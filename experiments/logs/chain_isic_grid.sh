#!/usr/bin/env bash
# ISIC: DA-TransUNet RGB check -> choose input on val -> ApproxDA ablation grid (val) -> test-split evaluation.
set -u
cd "$(dirname "$0")/.."
PY=$HOME/venvs/torch/bin/python
H=logs/queue.history
TEST_ARGS='--split val' bash run_queue.sh queue_isic.txt
val() { grep "exit=0 DA-TransUNet --dataset ISIC --max_epochs 300$1 ::" "$H" | grep -o "mean_dice : [0-9.]*" | tail -1 | awk '{print $3}'; }
gray=$(val ""); rgb=$(val " --rgb")
if [ -z "$rgb" ]; then echo "[$(date '+%F %T')] ISIC RGB check missing/failed; stopping" >> "$H"; exit 1; fi
mode=""; $PY -c "import sys; sys.exit(0 if $rgb >= $gray else 1)" && mode=" --rgb"
echo "[$(date '+%F %T')] ISIC input choice on val: gray $gray vs rgb $rgb -> '${mode:- grayscale}'" >> "$H"
C="--dataset ISIC --max_epochs 300$mode"
{
  echo ""
  echo "# ===== ApproxDA ablation round (input chosen on val: gray $gray vs rgb $rgb) ====="
  echo "# Window size M (r=32, PAM)"
  for m in 7 28 56 112; do echo "ApproxDA-TransUNet $C --window_size $m --rank 32 --gate_mode pam"; done
  echo "# Rank at M=112; full attention (M=112, r=0); window-only (M=28, r=0)"
  for r in 16 64 0; do echo "ApproxDA-TransUNet $C --window_size 112 --rank $r --gate_mode pam"; done
  echo "ApproxDA-TransUNet $C --window_size 28 --rank 0 --gate_mode pam"
  echo "# Routing at M=28; learned gate logs gate statistics (J6)"
  for g in cam fixed learn; do echo "ApproxDA-TransUNet $C --window_size 28 --rank 32 --gate_mode $g"; done
  echo "# CAM groups G under CAM-only"
  for g in 1 4; do echo "ApproxDA-TransUNet $C --window_size 28 --rank 32 --groups $g --gate_mode cam"; done
} >> queue_isic.txt
TEST_ARGS='--split val' bash run_queue.sh queue_isic.txt
# one evaluation of every finished ISIC run on the 1000-image test split
grep -v -e '^#' -e '^$' queue_isic.txt | grep -xF -f logs/queue.done | while IFS= read -r job; do
    proj="${job%% *}"; args="${job#* }"; name="$(echo "$job" | sed -e 's/--//g' -e 's/ \+/_/g')"
    (cd "$proj" && $PY test.py $args --split test) > "logs/$name.testsplit.log" 2>&1
    echo "[$(date '+%F %T')] ISIC test split: $job :: $(grep -o 'Testing performance.*' "logs/$name.testsplit.log" | tail -1)" >> "$H"
done
echo "[$(date '+%F %T')] ISIC chain finished" >> "$H"
