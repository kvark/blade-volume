#!/usr/bin/env bash
# Wrap any command in a transient systemd-run scope so OOM kills the
# wrapped process rather than evicting the rest of the desktop, and so
# GPU faults (Xid errors, llvmpipe fallback) abort the run loudly
# instead of silently degrading to a 25× CPU rasterizer path.
#
# Usage:
#   etc/cgroup_run.sh [--mem 16G] [--cpu-weight 100] [--gpu-log PATH]
#                     [--allow-llvmpipe] [--no-xid-watch] -- <command> [args...]
#
# Defaults:
#   --mem 16G                memory cap before scope OOMs
#   --cpu-weight 100         cgroup CPU weight
#   --gpu-log <auto>         periodic nvidia-smi samples; default
#                            /tmp/blade-volume-gpu-<pid>.log
#   --allow-llvmpipe         skip the NVIDIA-ICD pin and pre-flight
#                            (set this only if you actually want CPU)
#   --no-xid-watch           skip the background Xid watcher
#
# Example:
#   etc/cgroup_run.sh --mem 12G -- cargo run --release --bin train_colmap -- ...
#
# What the instrumentation does:
#   - Pins VK_ICD_FILENAMES to the NVIDIA ICD so Vulkan can never fall
#     back to llvmpipe (the failure mode that wasted ~43h after the
#     May-18 Xid 62 crash).
#   - Pre-flight: nvidia-smi must succeed and report no "ERR" /
#     "GPU requires reset" before we launch.
#   - Background watcher: tails `journalctl -k -f` for `NVRM: Xid` and
#     `pkill`s the scope if anything fires. Captures the offending
#     dmesg line into the gpu-log for post-mortem.
#   - Background sampler: writes `nvidia-smi --query-gpu` rows every
#     5s into the gpu-log (temp, power, mem, util, ecc, xid). This is
#     the timeline you want when the GPU misbehaves next time.
set -euo pipefail

MEM_CAP="16G"
CPU_WEIGHT="100"
GPU_LOG=""
ALLOW_LLVMPIPE=0
WATCH_XID=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mem) MEM_CAP="$2"; shift 2 ;;
    --cpu-weight) CPU_WEIGHT="$2"; shift 2 ;;
    --gpu-log) GPU_LOG="$2"; shift 2 ;;
    --allow-llvmpipe) ALLOW_LLVMPIPE=1; shift ;;
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

# ---------------------------------------------------------------- pre-flight
if [ "$ALLOW_LLVMPIPE" -eq 0 ]; then
  ICD="/usr/share/vulkan/icd.d/nvidia_icd.json"
  if [ ! -f "$ICD" ]; then
    echo "cgroup_run: NVIDIA ICD missing ($ICD); pass --allow-llvmpipe to force CPU" >&2
    exit 3
  fi
  if ! nvidia-smi >/dev/null 2>&1; then
    echo "cgroup_run: nvidia-smi failed; GPU likely wedged. Try sudo nvidia-smi -r or reboot." >&2
    exit 3
  fi
  # `nvidia-smi -q -d PERFORMANCE,POWER,TEMPERATURE` returns "ERR!" tokens
  # when the GPU is in a fault state but still partly responsive.
  if nvidia-smi -q 2>&1 | grep -qE "GPU requires reset|Pending GPU Reset|ERR!"; then
    echo "cgroup_run: nvidia-smi reports the GPU needs reset:" >&2
    nvidia-smi -q 2>&1 | grep -E "GPU requires reset|Pending GPU Reset|ERR!" >&2
    exit 3
  fi
  export VK_ICD_FILENAMES="$ICD"
  echo "cgroup_run: pinned VK_ICD_FILENAMES=$VK_ICD_FILENAMES" >&2
fi

# Capture initial state so the post-mortem has a baseline.
{
  echo "=== cgroup_run start $(date -Is) ==="
  echo "command: $*"
  echo "VK_ICD_FILENAMES=${VK_ICD_FILENAMES:-(unset)}"
  nvidia-smi 2>&1 || true
  echo "--- samples (ts,temp,power.draw,mem.used,util,ecc.errors,xid) ---"
} >>"$GPU_LOG"
echo "cgroup_run: GPU telemetry → $GPU_LOG" >&2

# ---------------------------------------------------------------- watchers
CHILD_PIDS=()
cleanup() {
  for p in "${CHILD_PIDS[@]:-}"; do
    [ -n "$p" ] && kill "$p" 2>/dev/null || true
  done
}
trap cleanup EXIT

# Periodic GPU sampler.
(
  while sleep 5; do
    ts=$(date -Is)
    row=$(nvidia-smi --query-gpu=temperature.gpu,power.draw,memory.used,utilization.gpu,ecc.errors.uncorrected.volatile.total \
                     --format=csv,noheader,nounits 2>/dev/null || echo "nvidia-smi-failed")
    echo "$ts,$row" >>"$GPU_LOG"
  done
) &
CHILD_PIDS+=("$!")

# Xid watcher: if any NVRM Xid fires, capture the line and kill our scope
# group so the wrapped run aborts loudly. Triggered by Xid 62 (PMU halt),
# 109 (CTX switch timeout), 154 (reset required), and anything else the
# kernel emits during the run.
WATCH_PID=""
if [ "$WATCH_XID" -eq 1 ]; then
  PARENT_PID=$$
  (
    journalctl -k -f -o cat --since=now 2>/dev/null | while IFS= read -r line; do
      if echo "$line" | grep -qE "NVRM:.*Xid"; then
        {
          echo "=== XID DETECTED $(date -Is) ==="
          echo "$line"
        } >>"$GPU_LOG"
        echo "cgroup_run: NVRM Xid detected, killing scope: $line" >&2
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
set +e
systemd-run --user --scope \
  -p MemoryMax="$MEM_CAP" \
  -p CPUWeight="$CPU_WEIGHT" \
  --description="blade-volume cgroup_run: $*" \
  -- "$@"
RC=$?
set -e

{
  echo "=== cgroup_run end $(date -Is) rc=$RC ==="
  nvidia-smi 2>&1 || true
} >>"$GPU_LOG"

exit "$RC"
