#!/usr/bin/env bash
# Phase D — Kvasir Window Sensitivity Ablation (gate=pam, M=7/28/56/112)
#
# Train log (per run): ../model/AdaDA_Kvasir224/<snapshot>/log.txt
# Test  log (per run): ./test_log/test_log_AdaDA_Kvasir224/<snapshot>.txt
# Combined stdout log: ./logs/phase_d_kvasir_<timestamp>.log
#
# Run from experiments/Ada-DA-TransUNet/:
#   bash run_phase_d_kvasir.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$SCRIPT_DIR/logs"
RUNLOG="$SCRIPT_DIR/logs/phase_d_kvasir_$(date +%Y%m%d_%H%M%S).log"

# Tee all stdout + stderr to the combined log
exec > >(tee -a "$RUNLOG") 2>&1

echo "================================================================"
echo "  Phase D — Kvasir Window Sensitivity Ablation"
echo "  Configs: gate=pam, r=32, G=8, epochs=300, bs=24"
echo "  Window sizes: M=7 (D3), M=28 (D4), M=56 (D5), M=112 (D6)"
echo "  Started: $(date)"
echo "================================================================"

# Common args shared by all runs
TRAIN_ARGS=(
    --dataset Kvasir
    --max_epochs 300
    --batch_size 24
    --gate_mode pam
    --rank 32
    --groups 8
    --val_interval 15
)

TEST_ARGS=(
    --dataset Kvasir
    --max_epochs 300
    --batch_size 24
    --gate_mode pam
    --rank 32
    --groups 8
)

# Map M values to Phase D run labels
declare -A RUN_LABEL=([7]="D3" [28]="D4" [56]="D5" [112]="D6")

for M in 7 28 56 112; do
    LABEL="${RUN_LABEL[$M]}"

    echo ""
    echo "----------------------------------------------------------------"
    echo "  ${LABEL}: Training   gate=pam  M=${M}  r=32  Kvasir"
    echo "  $(date)"
    echo "----------------------------------------------------------------"

    python train.py "${TRAIN_ARGS[@]}" --window_size "$M"
    TRAIN_EXIT=$?

    if [ $TRAIN_EXIT -ne 0 ]; then
        echo "  [ERROR] ${LABEL} training failed (exit $TRAIN_EXIT) — skipping inference, continuing to next M"
        continue
    fi

    echo ""
    echo "----------------------------------------------------------------"
    echo "  ${LABEL}: Inference  gate=pam  M=${M}  r=32  Kvasir"
    echo "  $(date)"
    echo "----------------------------------------------------------------"

    python test.py "${TEST_ARGS[@]}" --window_size "$M"
    TEST_EXIT=$?

    if [ $TEST_EXIT -ne 0 ]; then
        echo "  [ERROR] ${LABEL} inference failed (exit $TEST_EXIT) — continuing to next M"
    else
        echo "  [OK] ${LABEL} complete  $(date)"
    fi
done

echo ""
echo "================================================================"
echo "  Phase D complete: $(date)"
echo "  Combined log : $RUNLOG"
echo "  Train logs   : ../model/AdaDA_Kvasir224/*/log.txt"
echo "  Test  logs   : ./test_log/test_log_AdaDA_Kvasir224/*.txt"
echo "================================================================"
