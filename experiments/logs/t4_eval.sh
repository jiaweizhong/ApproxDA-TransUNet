#!/usr/bin/env bash
# Evaluate the T4 workshop best checkpoints (/data/checkpoints/ApproxDA-best-porch) once the ISIC seed runs are done.
cd "$(dirname "$0")/../ApproxDA-TransUNet"
H=../logs/queue.history; PY=$HOME/venvs/torch/bin/python; D=/data/checkpoints/ApproxDA-best-porch
# (ISIC seeds already finished)
echo "[$(date '+%F %T')] T4 checkpoint evaluation start" >> $H
while read -r f ds M g logged; do
    r=$($PY /tmp/claude-1000/-home-jiawei-projects-ApproxDA-TransUNet/0efe6c43-a856-47d2-b971-3a85dfba26f3/scratchpad/t4eval/eval_t4.py "$D/$f" $ds $M 32 $g 2>&1 | tee -a ../logs/t4_eval.out | grep '^RESULT')
    echo "[$(date '+%F %T')] t4 $f :: ${r:-FAILED (see logs/t4_eval.out)} | logged $logged" >> $H
done <<'JOBS'
ApproxDA_Synapse224_pretrain_R50-ViT-B_16_skip3_epo300_bs12_224_M28_pam_best_model.pth Synapse 28 pam 80.94
ApproxDA_Synapse224_pretrain_R50-ViT-B_16_skip3_epo300_bs12_224_M112_pam.pth Synapse 112 pam 79.44
ApproxDA_Synapse224_pretrain_R50-ViT-B_16_skip3_epo300_bs24_224_best_model.pth Synapse 7 learn 77.78
ApproxDA_ISIC224_pretrain_R50-ViT-B_16_skip3_epo300_bs24_224_best_model.pth ISIC 7 learn 89.58
ApproxDA_Kvasir224_pretrain_R50-ViT-B_16_skip3_epo300_bs24_224_best_model.pth Kvasir 7 learn 89.24
JOBS
echo "[$(date '+%F %T')] T4 checkpoint evaluation finished" >> $H
