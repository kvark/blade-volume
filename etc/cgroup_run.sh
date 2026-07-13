#!/usr/bin/env bash
# Wrap any command in a transient systemd-run scope so OOM kills the
# wrapped process rather than evicting the rest of the desktop, and so
# GPU faults abort the run loudly, while blade-volume's device selection
# rejects llvmpipe instead of silently degrading to a CPU rasterizer.
#
# Usage:
#   etc/cgroup_run.sh [--mem 16G] [--cpu-weight 100] [--gpu-log PATH]
#                     [--icd PATH] [--allow-software] [--no-xid-watch]
#                     -- <command> [args...]
#
# Defaults:
#   --mem 16G                memory cap before scope OOMs
#   --cpu-weight 100         cgroup CPU weight
#   --gpu-log <auto>         Vulkan, GPU, and cgroup-memory telemetry; default
#                            /tmp/blade-volume-gpu-<pid>.log
#   --icd PATH               pin one Vulkan ICD explicitly (optional)
#   --allow-software         permit a software-only Vulkan installation
#   --no-xid-watch           skip the background GPU-fault watcher
#
# Example:
#   etc/cgroup_run.sh --mem 12G -- cargo run --release --bin train_colmap -- ...
#
# What the instrumentation does:
#   - Rejects software-only Vulkan before launching; `--icd` can pin a
#     specific physical adapter on multi-GPU hosts.
#   - Watches kernel logs for NVIDIA Xid and AMD GPU fault/reset events.
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

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mem) MEM_CAP="$2"; shift 2 ;;
    --cpu-weight) CPU_WEIGHT="$2"; shift 2 ;;
    --gpu-log) GPU_LOG="$2"; shift 2 ;;
    --icd) ICD="$2"; shift 2 ;;
    --allow-software|--allow-llvmpipe) ALLOW_LLVMPIPE=1; shift ;;
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

: "${GPU_LOG:=/tmp/blade-volume-gpu-$$.log}"
UNIT="blade-volume-run-$$"

# ---------------------------------------------------------------- pre-flight
if [ -n "$ICD" ]; then
  if [ ! -f "$ICD" ]; then
    echo "cgroup_run: Vulkan ICD missing ($ICD)" >&2
    exit 3
  fi
  export VK_ICD_FILENAMES="$ICD"
  echo "cgroup_run: pinned VK_ICD_FILENAMES=$VK_ICD_FILENAMES" >&2
fi

if [ "$ALLOW_LLVMPIPE" -eq 0 ]; then
  if ! command -v vulkaninfo >/dev/null 2>&1 || ! vulkaninfo --summary >/dev/null 2>&1; then
    echo "cgroup_run: vulkaninfo pre-flight failed" >&2
    exit 3
  fi
  if ! vulkaninfo --summary 2>/dev/null | grep -E \
      >/dev/null \
      'deviceType[[:space:]]*=[[:space:]]*PHYSICAL_DEVICE_TYPE_(DISCRETE|INTEGRATED)_GPU'; then
    echo "cgroup_run: no physical Vulkan GPU found; pass --allow-software to override" >&2
    exit 3
  fi
fi

if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  HAS_NVIDIA=1
  if nvidia-smi -q 2>&1 | grep -E \
      "GPU requires reset|Pending GPU Reset|ERR!" >/dev/null; then
    echo "cgroup_run: nvidia-smi reports the GPU needs reset:" >&2
    nvidia-smi -q 2>&1 | grep -E "GPU requires reset|Pending GPU Reset|ERR!" >&2
    exit 3
  fi
fi

# Capture initial state so the post-mortem has a baseline.
{
  echo "=== cgroup_run start $(date -Is) ==="
  echo "command: $*"
  echo "VK_ICD_FILENAMES=${VK_ICD_FILENAMES:-(unset)}"
  echo "--- Vulkan summary ---"
  vulkaninfo --summary 2>&1 || true
  echo "--- host memory ---"
  free -h 2>&1 || true
  if [ "$HAS_NVIDIA" -eq 1 ]; then
    echo "--- NVIDIA baseline ---"
    nvidia-smi 2>&1 || true
  fi
  echo "--- samples ---"
} >>"$GPU_LOG"
echo "cgroup_run: GPU telemetry → $GPU_LOG" >&2

# ---------------------------------------------------------------- watchers
CHILD_PIDS=()
cleanup() {
  for p in "${CHILD_PIDS[@]:-}"; do
    [ -n "$p" ] && kill "$p" 2>/dev/null || true
  done
  if systemctl --user is-active --quiet "$UNIT.scope" 2>/dev/null; then
    systemctl --user kill --kill-whom=all "$UNIT.scope" 2>/dev/null || true
  fi
}
trap cleanup EXIT

# Periodic NVIDIA sampler when the tool is available. Cgroup memory is sampled
# separately for every vendor below.
if [ "$HAS_NVIDIA" -eq 1 ]; then
  (
    while sleep 5; do
      ts=$(date -Is)
      row=$(nvidia-smi --query-gpu=temperature.gpu,power.draw,memory.used,utilization.gpu,ecc.errors.uncorrected.volatile.total \
                       --format=csv,noheader,nounits 2>/dev/null || echo "nvidia-smi-failed")
      echo "$ts,nvidia,$row" >>"$GPU_LOG"
    done
  ) &
  CHILD_PIDS+=("$!")
fi

# Xid watcher: if any NVRM Xid fires, capture the line and kill our scope
# group so the wrapped run aborts loudly. Triggered by Xid 62 (PMU halt),
# 109 (CTX switch timeout), 154 (reset required), and anything else the
# kernel emits during the run.
WATCH_PID=""
if [ "$WATCH_XID" -eq 1 ]; then
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
    nvidia-smi 2>&1 || true
  fi
} >>"$GPU_LOG"

exit "$RC"
