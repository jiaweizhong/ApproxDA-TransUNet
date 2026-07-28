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

## 1 — DA-TransUNet CVC Baseline (~4h)

**Why:** ApproxDA CVC ablation ran under our conditions (SGD 300ep 80/20 224×224); need matching DA-TransUNet baseline for fair Fig 3 comparison. DA-TransUNet paper used different settings.

```bash
cd experiments/ApproxDA-TransUNet
python train_DA.py --dataset CVC --vit_name R50-ViT-B_16 \
    --max_epochs 300 --batch_size 24 --val_interval 15 \
    2>&1 | tee ../../logs/cvc_DA_300ep.log
```

- [ ] Record DSC, IoU, HD95
- [ ] Add to `tab:cvc` in `05_experiments.tex`

---

## 2 — Kvasir-Instrument (~15h)

**Data path:** `../data/Kvasir-Instrument/` (590 images, images/ + masks/)

```bash
cd experiments/ApproxDA-TransUNet

# Generate lists (once)
python datasets/generate_lists.py --dataset KvasirInstrument \
    --data_dir ../data/Kvasir-Instrument

# DA-TransUNet baseline (~3h)
python train_DA.py --dataset KvasirInstrument --vit_name R50-ViT-B_16 \
    --max_epochs 300 --batch_size 24 --val_interval 15 \
    2>&1 | tee ../../logs/ki_DA_300ep.log

# ApproxDA window ablation — M=7 is also the main result
for M in 7 28 56 112; do
  python train.py --dataset KvasirInstrument --vit_name R50-ViT-B_16 \
    --max_epochs 300 --batch_size 24 \
    --gate_mode pam --window_size $M --rank 32 --groups 8 \
    --val_interval 15 \
    2>&1 | tee ../../logs/ki_pam_M${M}_300ep.log
done
```

| Run | Config | Est. | Status |
|-----|--------|------|--------|
| DA-TransUNet baseline | 300ep | ~3h | ⏳ |
| ApproxDA M=7 | gate=pam, r=32 (main result) | ~3h | ⏳ |
| ApproxDA M=28 | gate=pam, r=32 | ~3h | ⏳ |
| ApproxDA M=56 | gate=pam, r=32 | ~3h | ⏳ |
| ApproxDA M=112 | gate=pam, r=32 | ~3h | ⏳ |

- [ ] Record ΔDSC across M → GCS for Kvasir-Instrument
- [ ] Add `tab:kvasir_instrument` in `05_experiments.tex`
- [ ] Add GCS row to `tab:gcs_spectrum` in `journal_gcs_mechanism.tex`

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

**After runs 2 and 3 complete**, re-run with 7 datasets to update Spearman ρ and regenerate figures.

```bash
cd experiments
python analyze_gcs_causal.py \
    --synapse_dir ../data/Synapse/train_npz \
    --acdc_dir    ../data/ACDC/ACDC_training_slices \
    --kvasir_dir  ../data/Kvasir-SEG \
    --isic_dir    ../data/ISIC2018 \
    --cvc_dir     ../data/CVC-ClinicDB \
    --ki_dir      ../data/Kvasir-Instrument \
    --cxr_dir     ../data/Montgomery \
    --n_images 200 --window_size 28 --resize 256
```

- [ ] Update SC1–SC5 ρ values (5→7 datasets) in `journal_gcs_mechanism.tex`
- [ ] Regenerate and copy `gcs_causal_sanity.png`

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
| 1 | DA-TransUNet CVC baseline | 4h | 🔴 High |
| 2 | Kvasir-Instrument (5 runs) | 15h | 🔴 High |
| 4 | F8 Entropy gate Synapse | 12h | 🟡 Medium |
| 5 | Re-run causal analysis (6 datasets) | 0.5h | 🟡 After #2 |
| 6 | F2 Dataset size study | 40h | 🟢 Low |
| **Total** | | **~32h** | |
