#!/usr/bin/env bash
# Re-evaluate jobs whose test ran before the test.py eval()-before-FLOP-count fix.
cd "$(dirname "$0")/.."
PY=$HOME/venvs/torch/bin/python
while IFS= read -r job; do
    proj="${job%% *}"; args="${job#* }"
    name="$(echo "$job" | sed -e 's/--//g' -e 's/ \+/_/g')"
    (cd "$proj" && $PY test.py $args) > "logs/$name.retest.log" 2>&1
    echo "[$(date '+%F %T')] retest exit=$? $job :: $(grep -o 'Testing performance.*' "logs/$name.retest.log" | tail -1)" | tee -a logs/queue.history
done <<'JOBS'
DA-TransUNet --dataset Synapse --max_epochs 300
ApproxDA-TransUNet --dataset Synapse --max_epochs 300 --window_size 28 --rank 32 --gate_mode pam
ApproxDA-TransUNet --dataset Synapse --max_epochs 300 --window_size 7 --rank 32 --gate_mode pam
JOBS
