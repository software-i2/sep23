#!/usr/bin/env bash
# Repeats the pick from the same place for a while and counts the outcomes. Run next to a running pick.launch.
# usage: scripts/soak.sh [seconds=600]   env: ATTEMPT_TIMEOUT=120
set -uo pipefail
DURATION="${1:-600}"
ATTEMPT_TIMEOUT="${ATTEMPT_TIMEOUT:-120}"
field() { rostopic echo -n1 "$1" 2>/dev/null | head -1 | tr -d '"'; }

reset() {
    rosservice call /pick/reset >/dev/null 2>&1  # also places the simulated arm home with the jaw open
    sleep 1
}

started=$SECONDS attempts=0 success=0 grabbed=0 on_handle=0 failed=0 timed_out=0
while [ $((SECONDS - started)) -lt "$DURATION" ]; do
    reset
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
        [ "$(field /pick/state/on_handle)" = True ] && on_handle=$((on_handle + 1)) && state="$state on the handle"
        ;;
    FAIL | ESTOP) failed=$((failed + 1)) ;;
    *) timed_out=$((timed_out + 1)); rosservice call /pick/stop >/dev/null 2>&1; state="timed out in ${state:-?}" ;;
    esac
    echo "[*] attempt $attempts: $state in $((SECONDS - begin)) s"
done
rosservice call /pick/stop >/dev/null 2>&1
echo "over $((SECONDS - started)) s: $attempts attempts, $success SUCCESS ($grabbed with a grip, $on_handle on the handle), $failed failed, $timed_out timed out"
