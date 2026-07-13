#!/usr/bin/env bash
# Wrap any command in a transient systemd-run scope so OOM kills the
# wrapped process rather than evicting the rest of the desktop, and so
# GPU faults abort the run loudly, while blade-volume's device selection
# rejects llvmpipe instead of silently degrading to a CPU rasterizer.
#
# Usage:
#   etc/cgroup_run.sh [--mem 16G] [--cpu-weight 100] [--gpu-log PATH]
#                     [--icd PATH] [--allow-software] [--cpu-only]
#                     [--probe-timeout SECONDS] [--no-xid-watch]
#                     -- <command> [args...]
#
# Defaults:
#   --mem 16G                memory cap before scope OOMs
#   --cpu-weight 100         cgroup CPU weight
#   --gpu-log <auto>         Vulkan, GPU, and cgroup-memory telemetry; default
#                            /tmp/blade-volume-gpu-<pid>.log
#   --icd PATH               pin one Vulkan ICD explicitly (optional)
#   --allow-software         permit a software-only Vulkan installation
#   --cpu-only               skip Vulkan and GPU probes for CPU-only commands
#   --probe-timeout SECONDS  abort if a GPU probe stalls; default 10
#   --no-xid-watch           skip the background GPU-fault watcher
#
# Example:
#   etc/cgroup_run.sh --mem 12G -- cargo run --release --bin train_colmap -- ...
#
# What the instrumentation does:
#   - Rejects software-only Vulkan before launching; `--icd` can pin a
#     specific physical adapter on multi-GPU hosts.
#   - Watches kernel logs for NVIDIA Xid and AMD GPU fault/reset events.
#   - Aborts if Vulkan or NVIDIA telemetry stops responding, without waiting
#     for an unkillable driver task.
#   - Samples the named cgroup's current/peak/swap memory every second,
#     plus NVIDIA telemetry when nvidia-smi is available.
set -euo pipefail

MEM_CAP="16G"
CPU_WEIGHT="100"
GPU_LOG=""
ICD=""
ALLOW_LLVMPIPE=0
WATCH_XID=1
HAS_NVIDIA=0
CPU_ONLY=0
PROBE_TIMEOUT_SECONDS=10

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mem) MEM_CAP="$2"; shift 2 ;;
    --cpu-weight) CPU_WEIGHT="$2"; shift 2 ;;
    --gpu-log) GPU_LOG="$2"; shift 2 ;;
    --icd) ICD="$2"; shift 2 ;;
    --allow-software|--allow-llvmpipe) ALLOW_LLVMPIPE=1; shift ;;
    --cpu-only) CPU_ONLY=1; shift ;;
    --probe-timeout) PROBE_TIMEOUT_SECONDS="$2"; shift 2 ;;
    --no-xid-watch) WATCH_XID=0; shift ;;
    --) shift; break ;;
    -h|--help)
      sed -n '1,/^set -e/p' "$0" | sed 's/^# \?//'
      exit 0
      ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [ $# -eq 0 ]; then
  echo "no command given (use -- to separate command from flags)" >&2
  exit 2
fi

if ! [[ "$PROBE_TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
  echo "probe timeout must be a positive integer" >&2
  exit 2
fi

if [ "$CPU_ONLY" -eq 1 ] && [ -n "$ICD" ]; then
  echo "--cpu-only and --icd cannot be used together" >&2
  exit 2
fi

: "${GPU_LOG:=/tmp/blade-volume-gpu-$$.log}"
UNIT="blade-volume-run-$$"
TEMP_FILES=()

remove_temp_files() {
  for f in "${TEMP_FILES[@]:-}"; do
    [ -n "$f" ] && rm -f "$f"
  done
  return 0
}
trap remove_temp_files EXIT

# Run a diagnostic command without ever blocking on a driver task indefinitely.
# On timeout, remove the process from bash's job table before returning: a GPU
# ioctl may be in uninterruptible sleep and therefore ignore even SIGKILL.
run_with_deadline() {
  local output_path="$1"
  shift
  local probe_pid
  local ticks=0
  local max_ticks=$((PROBE_TIMEOUT_SECONDS * 10))

  "$@" >"$output_path" 2>&1 &
  probe_pid=$!
  while kill -0 "$probe_pid" 2>/dev/null && [ "$ticks" -lt "$max_ticks" ]; do
    sleep 0.1
    ticks=$((ticks + 1))
  done

  if kill -0 "$probe_pid" 2>/dev/null; then
    kill -KILL "$probe_pid" 2>/dev/null || true
    disown "$probe_pid" 2>/dev/null || true
    return 124
  fi

  wait "$probe_pid"
}

# ---------------------------------------------------------------- pre-flight
if [ "$CPU_ONLY" -eq 0 ] && [ -n "$ICD" ]; then
  if [ ! -f "$ICD" ]; then
    echo "cgroup_run: Vulkan ICD missing ($ICD)" >&2
    exit 3
  fi
  export VK_ICD_FILENAMES="$ICD"
  echo "cgroup_run: pinned VK_ICD_FILENAMES=$VK_ICD_FILENAMES" >&2
fi

VULKAN_SUMMARY=""
NVIDIA_BASELINE=""
if [ "$CPU_ONLY" -eq 0 ]; then
  if command -v vulkaninfo >/dev/null 2>&1; then
    VULKAN_SUMMARY=$(mktemp /tmp/blade-volume-vulkan-XXXXXX)
    TEMP_FILES+=("$VULKAN_SUMMARY")
    if run_with_deadline "$VULKAN_SUMMARY" vulkaninfo --summary; then
      :
    else
      rc=$?
      if [ "$rc" -eq 124 ]; then
        echo "cgroup_run: vulkaninfo pre-flight timed out after ${PROBE_TIMEOUT_SECONDS}s" >&2
      else
        echo "cgroup_run: vulkaninfo pre-flight failed (rc=$rc)" >&2
      fi
      exit 3
    fi
  elif [ "$ALLOW_LLVMPIPE" -eq 0 ]; then
    echo "cgroup_run: vulkaninfo is required for the physical-GPU pre-flight" >&2
    exit 3
  fi

  if [ "$ALLOW_LLVMPIPE" -eq 0 ] && ! grep -E \
      >/dev/null \
      'deviceType[[:space:]]*=[[:space:]]*PHYSICAL_DEVICE_TYPE_(DISCRETE|INTEGRATED)_GPU' \
      "$VULKAN_SUMMARY"; then
    echo "cgroup_run: no physical Vulkan GPU found; pass --allow-software to override" >&2
    exit 3
  fi

  if command -v nvidia-smi >/dev/null 2>&1; then
    NVIDIA_BASELINE=$(mktemp /tmp/blade-volume-nvidia-XXXXXX)
    TEMP_FILES+=("$NVIDIA_BASELINE")
    if run_with_deadline "$NVIDIA_BASELINE" nvidia-smi -q; then
      HAS_NVIDIA=1
      if grep -E "GPU requires reset|Pending GPU Reset|ERR!" "$NVIDIA_BASELINE" >/dev/null; then
        echo "cgroup_run: nvidia-smi reports the GPU needs reset:" >&2
        grep -E "GPU requires reset|Pending GPU Reset|ERR!" "$NVIDIA_BASELINE" >&2
        exit 3
      fi
    else
      rc=$?
      if [ "$rc" -eq 124 ]; then
        echo "cgroup_run: nvidia-smi pre-flight timed out after ${PROBE_TIMEOUT_SECONDS}s" >&2
        exit 3
      fi
      echo "cgroup_run: nvidia-smi unavailable (rc=$rc); NVIDIA sampling disabled" >&2
    fi
  fi
fi

# Capture initial state so the post-mortem has a baseline.
{
  echo "=== cgroup_run start $(date -Is) ==="
  echo "command: $*"
  echo "cpu_only=$CPU_ONLY"
  echo "VK_ICD_FILENAMES=${VK_ICD_FILENAMES:-(unset)}"
  echo "--- Vulkan summary ---"
  if [ "$CPU_ONLY" -eq 1 ]; then
    echo "skipped (--cpu-only)"
  elif [ -n "$VULKAN_SUMMARY" ]; then
    sed -n '1,240p' "$VULKAN_SUMMARY"
  else
    echo "unavailable"
  fi
  echo "--- host memory ---"
  free -h 2>&1 || true
  if [ "$HAS_NVIDIA" -eq 1 ]; then
    echo "--- NVIDIA baseline ---"
    sed -n '1,1000p' "$NVIDIA_BASELINE"
  fi
  echo "--- samples ---"
} >>"$GPU_LOG"
echo "cgroup_run: telemetry → $GPU_LOG" >&2

# ---------------------------------------------------------------- watchers
CHILD_PIDS=()
cleanup() {
  for p in "${CHILD_PIDS[@]:-}"; do
    [ -n "$p" ] && kill "$p" 2>/dev/null || true
  done
  if systemctl --user is-active --quiet "$UNIT.scope" 2>/dev/null; then
    systemctl --user kill --kill-whom=all "$UNIT.scope" 2>/dev/null || true
  fi
  remove_temp_files
  return 0
}
trap cleanup EXIT

# Periodic NVIDIA sampler when the tool is available. Cgroup memory is sampled
# separately for every vendor below.
if [ "$HAS_NVIDIA" -eq 1 ]; then
  NVIDIA_SAMPLE=$(mktemp /tmp/blade-volume-nvidia-sample-XXXXXX)
  TEMP_FILES+=("$NVIDIA_SAMPLE")
  PARENT_PID=$$
  (
    while sleep 5; do
      ts=$(date -Is)
      if run_with_deadline "$NVIDIA_SAMPLE" nvidia-smi \
          --query-gpu=temperature.gpu,power.draw,memory.used,utilization.gpu,ecc.errors.uncorrected.volatile.total \
          --format=csv,noheader,nounits; then
        while IFS= read -r row; do
          echo "$ts,nvidia,$row" >>"$GPU_LOG"
        done <"$NVIDIA_SAMPLE"
      else
        rc=$?
        {
          echo "=== NVIDIA TELEMETRY FAILURE $ts rc=$rc ==="
          if [ "$rc" -eq 124 ]; then
            echo "nvidia-smi timed out after ${PROBE_TIMEOUT_SECONDS}s"
          else
            sed -n '1,80p' "$NVIDIA_SAMPLE"
          fi
        } >>"$GPU_LOG"
        echo "cgroup_run: NVIDIA telemetry failed (rc=$rc), killing scope" >&2
        systemctl --user kill --kill-whom=all "$UNIT.scope" 2>/dev/null || true
        kill -TERM "$PARENT_PID" 2>/dev/null || true
        break
      fi
    done
  ) &
  CHILD_PIDS+=("$!")
fi

# Xid watcher: if any NVRM Xid fires, capture the line and kill our scope
# group so the wrapped run aborts loudly. Triggered by Xid 62 (PMU halt),
# 109 (CTX switch timeout), 154 (reset required), and anything else the
# kernel emits during the run.
WATCH_PID=""
if [ "$WATCH_XID" -eq 1 ] && [ "$CPU_ONLY" -eq 0 ]; then
  PARENT_PID=$$
  (
    journalctl -k -f -o cat --since=now 2>/dev/null | while IFS= read -r line; do
      if echo "$line" | grep -qiE \
          "NVRM:.*Xid|amdgpu.*(fault|reset|timeout|hang|ring .*stalled)"; then
        {
          echo "=== GPU FAULT DETECTED $(date -Is) ==="
          echo "$line"
        } >>"$GPU_LOG"
        echo "cgroup_run: GPU fault detected, killing scope: $line" >&2
        systemctl --user kill --kill-whom=all "$UNIT.scope" 2>/dev/null || true
        kill -TERM "$PARENT_PID" 2>/dev/null || true
        break
      fi
    done
  ) &
  WATCH_PID="$!"
  CHILD_PIDS+=("$WATCH_PID")
fi

# ---------------------------------------------------------------- exec
# Don't `exec` — we need the trap to fire and the watchers to drain.
START_TIME=$(date -Is)
(
  # The scope appears shortly after systemd-run starts. Once present, sample
  # kernel-backed cgroup counters until the command exits.
  for _ in {1..50}; do
    if systemctl --user show "$UNIT.scope" -p ActiveState >/dev/null 2>&1; then
      break
    fi
    sleep 0.1
  done
  while systemctl --user is-active --quiet "$UNIT.scope" 2>/dev/null; do
    {
      echo "=== cgroup sample $(date -Is) ==="
      systemctl --user show "$UNIT.scope" \
        -p MemoryCurrent -p MemoryPeak -p MemorySwapCurrent -p MemorySwapPeak \
        -p ActiveState -p SubState 2>/dev/null || true
      control_group=$(systemctl --user show "$UNIT.scope" -p ControlGroup --value 2>/dev/null || true)
      if [ -n "$control_group" ] && [ -r "/sys/fs/cgroup${control_group}/memory.events" ]; then
        sed -n '1,20p' "/sys/fs/cgroup${control_group}/memory.events"
      fi
    } >>"$GPU_LOG"
    sleep 1
  done
) &
MEMORY_WATCH_PID="$!"
CHILD_PIDS+=("$MEMORY_WATCH_PID")

set +e
systemd-run --user --scope --unit="$UNIT" \
  -p MemoryMax="$MEM_CAP" \
  -p MemorySwapMax=0 \
  -p CPUWeight="$CPU_WEIGHT" \
  --description="blade-volume cgroup_run: $*" \
  -- "$@"
RC=$?
set -e
wait "$MEMORY_WATCH_PID" 2>/dev/null || true

{
  echo "=== cgroup_run end $(date -Is) rc=$RC ==="
  journalctl --user -u "$UNIT.scope" --since="$START_TIME" --no-pager 2>&1 || true
  if [ "$HAS_NVIDIA" -eq 1 ]; then
    echo "--- NVIDIA final ---"
    if run_with_deadline "$NVIDIA_BASELINE" nvidia-smi; then
      sed -n '1,1000p' "$NVIDIA_BASELINE"
    else
      echo "nvidia-smi failed or timed out after the run"
    fi
  fi
} >>"$GPU_LOG"

exit "$RC"
