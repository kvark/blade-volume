#!/usr/bin/env bash
# Wrap any command in a transient systemd-run scope so OOM kills the
# wrapped process rather than evicting the rest of the desktop.
#
# Usage:
#   etc/cgroup_run.sh [MEM_CAP] -- <command> [args...]
#
# Defaults:
#   MEM_CAP=16G  (override with --mem 24G etc.)
#
# Example:
#   etc/cgroup_run.sh --mem 12G -- cargo run --release --bin train_colmap -- ...
#
# Why systemd-run --user --scope:
#   - --scope runs in the caller's terminal (no daemonization)
#   - --user puts the cgroup under user.slice/user@UID.service
#   - kills cascade: if the scope hits MemoryMax, all processes in the
#     scope die together, leaving the rest of your session alone.
set -euo pipefail

MEM_CAP="16G"
CPU_WEIGHT="100"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mem) MEM_CAP="$2"; shift 2 ;;
    --cpu-weight) CPU_WEIGHT="$2"; shift 2 ;;
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

exec systemd-run --user --scope \
  -p MemoryMax="$MEM_CAP" \
  -p CPUWeight="$CPU_WEIGHT" \
  --description="blade-volume cgroup_run: $*" \
  -- "$@"
