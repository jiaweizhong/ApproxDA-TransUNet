#!/bin/bash
# DA-TransUNet ISIC 2018 training — persistent tmux session.
# Usage: bash run_isic.sh
# Attach:  tmux attach -t isic
# Monitor: tail -f DA2.log

SESSION="isic"
LOG="DA2.log"

tmux new-session -d -s "$SESSION" \
  "python train.py \
    --dataset ISIC \
    --max_epochs 300 \
    --batch_size 24 \
    --img_size 224 \
    --n_skip 3 \
    --val_interval 30 \
    2>&1 | tee $LOG"

echo "Started in tmux session '$SESSION'"
echo "Attach : tmux attach -t $SESSION"
echo "Log    : tail -f $LOG"
