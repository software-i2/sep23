#!/usr/bin/env bash
# A shell in the Noetic container with the workspace sourced; run again for another shell in the same one.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IMAGE:-sep23:noetic}"
NAME="${CONTAINER:-sep23_dev}"
PORT="$(sed -n 's/^ *serial_port: *\([^ #]*\).*/\1/p' "$ROOT/src/sep23/config/robot.yaml")"

if [ -n "$(docker ps -q --filter "name=^${NAME}$")" ]; then
    exec docker exec -it -w "$ROOT" "$NAME" bash
fi
DEVICE=()
[ -e "$PORT" ] && DEVICE=(--device "$PORT:$PORT") || echo "[!] $PORT not present: sim:=true only"
exec docker run --rm -it --name "$NAME" --network host "${DEVICE[@]}" -e WS="$ROOT" -v "$ROOT:$ROOT" -w "$ROOT" "$IMAGE" bash -c "echo '[*] roslaunch sep23 pick.launch sim:=true bag:=\$PWD/data/captures_grasp_poses.bag    then in a second shell: scripts/soak.sh'
    exec bash"
