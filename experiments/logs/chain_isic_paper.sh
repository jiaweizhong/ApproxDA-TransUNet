#!/usr/bin/env bash
# ISIC paper protocol: DA-TransUNet seeds -> best seed -> ApproxDA grid if DA is within 1 pp of the paper (88.88).
set -u
cd "$(dirname "$0")/.."
PY=$HOME/venvs/torch/bin/python
H=logs/queue.history
C="--dataset ISIC --max_epochs 50 --optimizer adam --base_lr 0.001 --rgb --img_size 256"
bash run_queue.sh queue_isic.txt
best=$(for s in "" " --seed 2345" " --seed 3456"; do
    d=$(grep -F "exit=0 DA-TransUNet $C$s ::" "$H" | grep -o "mean_dice : [0-9.]*" | tail -1 | awk '{print $3}')
    [ -n "$d" ] && echo "$d|$s"
done | sort -t'|' -k1 -gr | head -1)
[ -z "$best" ] && { echo "[$(date '+%F %T')] ISIC paper protocol: no DA result; stopping" >> "$H"; exit 1; }
dice=${best%%|*}; seedarg=${best#*|}
if [ "$($PY -c "print(int($dice >= 0.8788))")" != 1 ]; then
    echo "[$(date '+%F %T')] ISIC paper protocol: best DA Dice $dice <$seedarg> below 87.88; ApproxDA grid NOT started" >> "$H"
    exit 0
fi
echo "[$(date '+%F %T')] ISIC paper protocol: best DA Dice $dice <${seedarg:- seed 1234}>; starting ApproxDA grid" >> "$H"
G="$C$seedarg"
{
  echo ""
  echo "# ===== ApproxDA ablation round, setup of the best DA run (Dice $dice,${seedarg:- seed 1234}) ====="
  echo "# Window size M (r=32, PAM); maps at 256 px are 16/32/64/128"
  for m in 8 32 64 128; do echo "ApproxDA-TransUNet $G --window_size $m --rank 32 --gate_mode pam"; done
  echo "# Rank at M=128; full attention (M=128, r=0); window-only (M=32, r=0)"
  for r in 16 64 0; do echo "ApproxDA-TransUNet $G --window_size 128 --rank $r --gate_mode pam"; done
  echo "ApproxDA-TransUNet $G --window_size 32 --rank 0 --gate_mode pam"
  echo "# Routing at M=32; learned gate logs gate statistics (J6)"
  for g in cam fixed learn; do echo "ApproxDA-TransUNet $G --window_size 32 --rank 32 --gate_mode $g"; done
  echo "# CAM groups G under CAM-only"
  for g in 1 4; do echo "ApproxDA-TransUNet $G --window_size 32 --rank 32 --groups $g --gate_mode cam"; done
} >> queue_isic.txt
bash run_queue.sh queue_isic.txt
echo "[$(date '+%F %T')] ISIC paper-protocol chain finished" >> "$H"
