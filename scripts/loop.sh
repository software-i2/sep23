#!/usr/bin/env bash
# loops the mission for X seconds, with random yaw each time
set -uo pipefail
DURATION="${1:-600}"
STEER="${2:-true}"
ATTEMPT_TIMEOUT="${ATTEMPT_TIMEOUT:-120}"
YAW_DEG="${YAW_DEG:-0}"
case "$STEER" in true | false) ;; *) echo "[loop] steer $STEER is not true or false" >&2; exit 1 ;; esac
rosparam set /pick/track/steer "$STEER"  # the pick reads it on every start
field() { rostopic echo -n1 "$1" 2>/dev/null | head -1 | tr -d '"'; }

started=$SECONDS attempts=0 success=0 grabbed=0 failed=0 timed_out=0
while [ $((SECONDS - started)) -lt "$DURATION" ]; do
    yaw=$((RANDOM % (2 * YAW_DEG + 1) - YAW_DEG))
    [ "$YAW_DEG" -gt 0 ] && rosparam set /synthetic/yaw_deg "$yaw"  # before the reset, so the next look already sees it turned
    rosservice call /pick/reset >/dev/null 2>&1
    sleep 1
    attempts=$((attempts + 1))
    rosservice call /pick/start >/dev/null 2>&1
    begin=$SECONDS state=""
    while [ $((SECONDS - begin)) -lt "$ATTEMPT_TIMEOUT" ]; do
        state="$(field /pick/state/state)"
        case "$state" in SUCCESS | FAIL | ESTOP) break ;; esac
        sleep 1
    done
    case "$state" in
    SUCCESS)
        success=$((success + 1))
        [ "$(field /pick/state/grabbed)" = True ] && grabbed=$((grabbed + 1))
        ;;
    FAIL | ESTOP) failed=$((failed + 1)) ;;
    *) timed_out=$((timed_out + 1)); rosservice call /pick/stop >/dev/null 2>&1; state="timed out in ${state:-?}" ;;
    esac
    echo "[*] attempt $attempts, yaw $yaw deg: $state in $((SECONDS - begin)) s"
done
rosservice call /pick/stop >/dev/null 2>&1
echo "steer $STEER, yaw +-$YAW_DEG deg, over $((SECONDS - started)) s: $attempts attempts, $success SUCCESS ($grabbed with a grip), $failed failed, $timed_out timed out"
