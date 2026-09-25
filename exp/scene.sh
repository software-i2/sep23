#!/usr/bin/env bash
# Synthetic scene on the simulated arm. usage: exp/scene.sh [mode = 2: 0 none, 1 up to 50%, 2 up to 100%] [pick.launch args]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
MODE="${1:-2}"
shift || true
BAG="${BAG:-data/captures_grasp_poses.bag}"
AMPLITUDE_M="${AMPLITUDE_M:-0.02}"  # scene sway; 0 holds it still
case "$MODE" in 0) COVER=0 ;; 1) COVER=50 ;; 2) COVER=100 ;; *) echo "[scene] mode $MODE is none of 0, 1, 2" >&2; exit 1 ;; esac
OCCLUDED=$([ "$MODE" = 0 ] && echo false || echo true)

PIDS=()
trap 'kill "${PIDS[@]}" 2>/dev/null; wait' EXIT
roslaunch sep23 pick.launch sim:=true synthetic:=true occluded:="$OCCLUDED" "$@" &
PIDS+=($!)
python3 exp/synthetic_drift.py _bag:="$PWD/$BAG" _amplitude_m:="$AMPLITUDE_M" &
PIDS+=($!)
if [ "$OCCLUDED" = true ]; then
    python3 exp/occluder.py _cover:="$COVER" &
    PIDS+=($!)
fi
wait -n
