#!/usr/bin/env bash
# Sequential GPU job queue for the journal re-runs.
#
# Each non-comment line of the queue file is "<project dir> <args...>". For every
# line the runner trains (train.py <args>) and then evaluates the final epoch
# (test.py <args>) in that project directory. The queue file is re-read before
# each job, so it can be edited while the runner is going. Finished lines are
# appended to logs/queue.done, failed ones to logs/queue.failed (delete a line
# there to retry it). TEST_ARGS is appended to test.py only, e.g. TEST_ARGS='--split val'
# to select ISIC configurations on the official validation split.
#
#   setsid nohup bash run_queue.sh queue_attn_noid.txt >> logs/queue.out 2>&1 < /dev/null &
set -u
EXP="$(cd "$(dirname "$0")" && pwd)"
QUEUE="$(realpath "${1:-$EXP/queue_attn_noid.txt}")"
PY="${PY:-$HOME/venvs/torch/bin/python}"
TEST_ARGS="${TEST_ARGS:-}"
LOGS="$EXP/logs"
mkdir -p "$LOGS"
touch "$LOGS/queue.done" "$LOGS/queue.failed"

next_job() {
    grep -v -e '^[[:space:]]*#' -e '^[[:space:]]*$' "$QUEUE" | while IFS= read -r line; do
        if ! grep -qxF "$line" "$LOGS/queue.done" "$LOGS/queue.failed"; then
            echo "$line"
            break
        fi
    done
}

while job="$(next_job)"; [ -n "$job" ]; do
    proj="${job%% *}"
    args="${job#* }"
    name="$(echo "$job" | sed -e 's/--//g' -e 's/ \+/_/g')"
    echo "[$(date '+%F %T')] start  $job" | tee -a "$LOGS/queue.history"
    (cd "$EXP/$proj" && $PY train.py $args) > "$LOGS/$name.train.log" 2>&1 &&
        (cd "$EXP/$proj" && $PY test.py $args $TEST_ARGS) > "$LOGS/$name.test.log" 2>&1
    status=$?
    if [ $status -eq 0 ]; then
        echo "$job" >> "$LOGS/queue.done"
        result="$(grep -o 'Testing performance.*' "$LOGS/$name.test.log" | tail -1)"
    else
        echo "$job" >> "$LOGS/queue.failed"
        result="FAILED (see $LOGS/$name.*.log)"
    fi
    echo "[$(date '+%F %T')] exit=$status $job :: $result" | tee -a "$LOGS/queue.history"
done
echo "[$(date '+%F %T')] queue empty" | tee -a "$LOGS/queue.history"
