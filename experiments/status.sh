#!/usr/bin/env bash
# One-screen status of run_queue.sh: progress, current job, results so far.
#   bash status.sh [queue file]          watch -n 60 bash status.sh
# Results list every finished job in logs/queue.history (all datasets).
EXP="$(cd "$(dirname "$0")" && pwd)"
# default: the queue the running runner was started with, else queue_synapse.txt
running="$(pgrep -af "run_queue.sh" | grep -o "queue_[A-Za-z0-9_]*\.txt" | head -1)"
QUEUE="${1:-$EXP/${running:-queue_synapse.txt}}"
LOGS="$EXP/logs"

jobs="$(grep -v -e '^[[:space:]]*#' -e '^[[:space:]]*$' "$QUEUE")"
total=$(echo "$jobs" | grep -c .)
done_n=$(grep -cxF -f <(echo "$jobs") "$LOGS/queue.done")
failed_n=$(grep -cxF -f <(echo "$jobs") "$LOGS/queue.failed")
echo "Queue $(basename "$QUEUE"): $done_n/$total done, $failed_n failed"
pgrep -f run_queue.sh > /dev/null && echo "Runner: running" || echo "Runner: NOT running"

current=$(grep -E '\] (start|exit=)' "$LOGS/queue.history" | tail -1)
if [[ "$current" == *"] start "* ]]; then
    log=$(ls -t "$EXP"/model/*/*/log.txt | head -1)
    echo "Current: ${current#*start  }"
    grep -o 'epoch [0-9]*/[0-9]*  loss: [0-9.]*.*elapsed: [0-9.]*h' "$log" | tail -1 | sed 's/^/         /'
fi

echo
echo "Results (Dice / HD95; first 3 jobs use their retest):"
grep -E 'exit=0 |retest exit=0 ' "$LOGS/queue.history" | sed -E \
    -e 's/.*exit=0 (ApproxDA-TransUNet|DA-TransUNet) --dataset ([A-Za-z]+) --max_epochs [0-9]+ ?(.*) :: .*mean_dice : ([0-9.]+) mean_hd95 : ([0-9.]+).*/\1 \2 \3|\4|\5/' |
    awk -F'|' '{r[$1] = sprintf("%6.2f  %6.2f", 100 * $2, $3); if (!($1 in o)) o[$1] = ++n; k[o[$1]] = $1}
               END {for (i = 1; i <= n; i++) printf "  %s  %s\n", r[k[i]], k[i]}'
if [ "$failed_n" -gt 0 ]; then
    echo; echo "Failed:"; grep -xF -f <(echo "$jobs") "$LOGS/queue.failed" | sed 's/^/  /'
fi
