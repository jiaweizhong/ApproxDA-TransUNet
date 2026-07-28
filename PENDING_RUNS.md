# Pending Experiments — Journal Extension

> All runs on Lightning AI. Default config: SGD, lr=0.01 poly, 300ep, bs=24, 224×224, val_interval=15, seed=1234.
> Estimated total: ~70h compute + ~1h analysis.

---

## 0 — Figure Regeneration (no training, ~0.5h)

**Trigger:** ISIC GCS 0.70 → 0.50pp already updated in both scripts.

```bash
cd experiments
python analyze_gcs_causal.py \
    --synapse_dir ../data/Synapse/train_npz \
    --acdc_dir    ../data/ACDC/ACDC_training_slices \
    --kvasir_dir  ../data/Kvasir-SEG \
    --isic_dir    ../data/ISIC2018 \
    --cvc_dir     ../data/CVC-ClinicDB \
    --n_images 200 --window_size 28 --resize 256

python analyze_gcs_mask.py \
    --synapse_dir ../data/Synapse/train_npz \
    --kvasir_dir  ../data/Kvasir-SEG \
    --isic_dir    ../data/ISIC2018 \
    --n_images 200 --resize 256
```

- [ ] Copy `gcs_causal_sanity.png` → `paper-journal/figures/`
- [ ] Copy `gcs_mask_sanity.png` → `paper-journal/figures/`

---

## 1 — DA-TransUNet Baselines: ACDC + CVC (~11h)

### 1a — DA-TransUNet ACDC Baseline (~7h) 🔴

**Why（revised decision）:** 加入主结果表，即使我们不赢。理由：
- ACDC 是 GCS 理论的**预测性验证**：low-GCS + 同心环结构 → 理论预测提升有限 → 实验结果与预测一致 → falsifiability 增强
- 只展示赢的数据集会被 reviewer 质疑 cherry-picking；ACDC 作为"honest negative"使结论更可信
- 在 `06_analysis.tex` 加解释段落，在 `07_conclusion.tex` 加 limitation 条目

```bash
cd experiments/ApproxDA-TransUNet
python train_DA.py --dataset ACDC --vit_name R50-ViT-B_16 \
    --max_epochs 300 --batch_size 24 --val_interval 15 \
    2>&1 | tee ../../logs/acdc_DA_300ep.log
```

- [ ] Record mean DSC, per-class (RV/Myo/LV), HD95
- [ ] Add `tab:acdc` to `05_experiments.tex`; "four" → "five benchmarks" (L5, L9, L25)
- [ ] Add analysis paragraph in `06_analysis.tex`: ACDC 表现持平的机制解释（low GCS / 同心环不需要 locality prior / Myo 薄环精度受 r=32 限制）
- [ ] Add limitation bullet in `07_conclusion.tex`

### 1b — DA-TransUNet CVC Baseline (~4h)

**Why:** ApproxDA CVC ablation ran under our conditions (SGD 300ep 80/20 224×224); need matching DA-TransUNet baseline for fair comparison. DA-TransUNet paper used different settings.

```bash
cd experiments/ApproxDA-TransUNet
python train_DA.py --dataset CVC --vit_name R50-ViT-B_16 \
    --max_epochs 300 --batch_size 24 --val_interval 15 \
    2>&1 | tee ../../logs/cvc_DA_300ep.log
```

- [ ] Record DSC, IoU, HD95
- [ ] Add to `tab:cvc` in `05_experiments.tex`

---

## 4 — F8: Entropy Gate Ablation on Synapse (~12h)

**Why:** Validates H3 root-cause claim — symmetry-breaking gate avoids collapse. 1 run only; result goes into §5 "Alternative Gate Designs."

**Prerequisite:** Add `gate_mode='entropy'` branch to `ApproxDABlock.forward()` in `Architecture/ApproxDATransUNet.py`.

```bash
cd experiments/ApproxDA-TransUNet
python train.py --dataset Synapse --vit_name R50-ViT-B_16 \
    --max_epochs 300 --batch_size 24 \
    --gate_mode entropy --window_size 7 --rank 32 --groups 8 \
    --val_interval 15 \
    2>&1 | tee ../../logs/synapse_entropy_M7_300ep.log
```

| Run | Config | Est. | Status |
|-----|--------|------|--------|
| Synapse entropy gate | M=7, r=32 | ~12h | ⏳ |

Compare vs: gate=learn M=7: **77.78%** / gate=pam M=7: **78.64%**

- [ ] Check gate value distribution (does g avoid 0.5?)
- [ ] Add 1-row to gate ablation table in `05_experiments.tex`

---

## 5 — Post-experiment: Re-run Causal Analysis (~0.5h)

**After run #0 figure regeneration**, re-run on all 5 datasets to update Spearman ρ and regenerate figures.

```bash
cd experiments
python analyze_gcs_causal.py \
    --synapse_dir ../data/Synapse/train_npz \
    --acdc_dir    ../data/ACDC/ACDC_training_slices \
    --kvasir_dir  ../data/Kvasir-SEG \
    --isic_dir    ../data/ISIC2018 \
    --cvc_dir     ../data/CVC-ClinicDB \
    --n_images 200 --window_size 28 --resize 256
```

- [ ] Verify SC1–SC5 ρ values (5 datasets) in `journal_gcs_mechanism.tex`
- [ ] Regenerate and copy `gcs_causal_sanity.png` → `paper-journal/figures/`

---

## 6 — F2: Dataset Size Study (~40h, lower priority)

```bash
cd experiments/ApproxDA-TransUNet
for FRAC in 0.2 0.4 0.6 0.8; do
  python train_DA.py --dataset Kvasir --max_epochs 300 --batch_size 24 \
      --train_fraction $FRAC --val_interval 15 \
      2>&1 | tee ../../logs/kvasir_DA_frac${FRAC}_300ep.log
  python train.py --dataset Kvasir --gate_mode pam --window_size 56 --rank 32 \
      --max_epochs 300 --batch_size 24 \
      --train_fraction $FRAC --val_interval 15 \
      2>&1 | tee ../../logs/kvasir_pam_M56_frac${FRAC}_300ep.log
done
```

- [ ] Plot Δ(ApproxDA − DA) vs fraction → regularization hypothesis
- [ ] Add §5 "Dataset Size Analysis" subsection

---

## Summary

| # | Experiment | Est. | Priority |
|---|-----------|------|----------|
| 0 | Regenerate causal/mask PNG (script only) | 0.5h | 🔴 Now |
| 1a | DA-TransUNet ACDC baseline | 7h | 🔴 High |
| 1b | DA-TransUNet CVC baseline | 4h | 🔴 High |
| 4 | F8 Entropy gate Synapse | 12h | 🟡 Medium |
| 5 | Re-run causal analysis (5 datasets) | 0.5h | 🟡 After #0 |
| 6 | F2 Dataset size study | 40h | 🟢 Low |
| **Total** | | **~24h** | |

> Kvasir-Instrument **已移除**：SC5=0（binary），与 Kvasir-SEG/CVC/ISIC 同 tier，不扩展 GCS 谱线；5 个数据集的谱线已完整。|
