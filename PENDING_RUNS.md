# Pending Experiments — Journal Extension

> All runs on Lightning AI. Default config: SGD, lr=0.01 poly, 300ep, bs=24, 224×224, val_interval=15, seed=1234.
> Status updated as of 2026-09-04.

---

## 0 — Figure Regeneration (no training, ~0.5h) ✅ Done

- [x] ISIC GCS (0.50pp) and 5-dataset spectrum updated in scripts (`analyze_gcs_causal.py`, `analyze_gcs_mask.py`, `analyze_f1_generalization_gap.py`).
- [x] Generated high-res vector PDF + PNG (`gcs_causal_sanity.{pdf,png}`, `gcs_mask_sanity.{pdf,png}`, `fig_f1_generalization_gap.{pdf,png}`) → `paper-journal/figures/`.

---

## 1 — DA-TransUNet Baselines: ACDC + CVC

### 1a — DA-TransUNet ACDC Baseline (~7h) ✅ Done (2026-09-04)

- **Results:** DSC: RV **88.84%** / Myo **85.98%** / LV **90.67%** / Mean **88.50%**.
- [x] Record mean DSC, per-class (RV/Myo/LV), HD95.
- [x] Update `tab:acdc` in `05_experiments.tex` (ApproxDA 88.97% beats DA-TransUNet 88.50% by +0.47%).
- [x] Integrate 5 benchmarks in `05_experiments.tex`.
- [x] Analysis paragraph in `06_analysis.tex`: ACDC mechanism explanation (low GCS / compact concentric anatomy / high RV 89.61% and Myo 85.91% vs global LV).
- [x] Limitation bullet in `07_conclusion.tex`.

### 1b — DA-TransUNet CVC Baseline (~4h) ⏸️ Optional / Benchmark In Place

- **Current Status:** Table IV uses published DA-TransUNet CVC result (89.47% DSC, 82.51% mIoU). ApproxDA-TransUNet achieves 90.99% DSC (+1.52%) and 85.09% mIoU (+2.58%).
- [ ] (Optional) Re-run under identical 300ep SGD protocol if required.

---

## 4 — F8: Entropy Gate Ablation on Synapse (~12h) ⏳ Pending / Future

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

## 5 — Post-experiment: Re-run Causal Analysis (~0.5h) ✅ Done

- [x] Verified SC1–SC5 ρ values across all 5 datasets (Synapse 2.30, ACDC 0.73, Kvasir 0.64, CVC 0.62, ISIC 0.50) in `journal_gcs_mechanism.tex` and `06_analysis.tex`.
- [x] Regenerated and embedded `gcs_causal_sanity.{pdf,png}` and `gcs_mask_sanity.{pdf,png}` in `paper-journal/figures/`.

---

## Summary & Audit Status

| # | Experiment | Est. | Priority | Current Status |
|---|-----------|------|----------|----------------|
| 0 | Regenerate causal/mask PNG (script only) | 0.5h | 🔴 Now | ✅ **Completed** (Vector PDF & PNG in `paper-journal/figures/`) |
| 1a | DA-TransUNet ACDC baseline | 7h | 🔴 High | ✅ **Completed** (RV 88.84, Myo 85.98, LV 90.67, Mean 88.50 in Table II) |
| 1b | DA-TransUNet CVC baseline | 4h | 🔴 High | ⏸️ **Optional** (Published baseline in Table IV) |
| 4 | F8 Entropy gate Synapse | 12h | 🟡 Medium | ⏳ **Pending / Optional** |
| 5 | Re-run causal analysis (5 datasets) | 0.5h | 🟡 After #0 | ✅ **Completed** (Metrics & Table VI updated) |

> **已取消/移除项**：
> - **F2 (Dataset Size Study)**：已移除（无实际用途，5 数据集 GCS 与 SSD 理论已完备）。
> - **F7 (Kvasir-Instrument / Chest X-ray)**：已移除（SC5=0，不扩展 GCS 谱线）。
