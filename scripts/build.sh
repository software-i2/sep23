#!/usr/bin/env bash
# Builds the workspace in the Noetic image.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IMAGE:-sep23:noetic}"
docker image inspect "$IMAGE" >/dev/null 2>&1 || docker build -t "$IMAGE" "$ROOT/docker"
docker run --rm -u "$(id -u):$(id -g)" -e HOME=/tmp -v "$ROOT:$ROOT" -w "$ROOT" "$IMAGE" bash -c "
    source /opt/ros/noetic/setup.bash
    catkin config --extend /opt/ros/noetic >/dev/null
    catkin build $*"
