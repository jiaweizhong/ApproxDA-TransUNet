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

---

## E1 — Minimal Seed Robustness Check (~8h) ⏳ Strongly Recommended, Not a Blocker

**Why:** Reviewer comment #2 (revised framing): with the paper's claims already downgraded from "proof/causal" to "observed correlation/candidate," this is no longer a submission blocker. It remains the single most reviewer-defensible addition: low-GCS spans (ISIC 0.50pp, CVC 0.62pp, Kvasir 0.64pp, ACDC 0.73pp) are small enough that a reviewer may ask whether they're distinguishable from single-run training noise relative to the high-GCS Synapse span (2.30pp).

**Text fallback: ✅ Already applied (2026-09-04)** — "$4.6\times$ spectrum" softened to "an observed $4.6\times$ difference in single-run window-sensitivity spans" in `01_abstract.tex` and `06_analysis.tex`; single-run limitation disclosed in `07_conclusion.tex` Limitations. The experiment below remains optional — only needed if reviewers push back on the fallback wording.

**Minimal plan (not the full 18-run sweep):** Only 8 runs — 2 seeds each for the 4 configs that anchor the high-vs-low contrast:
- Synapse $M{=}7$ (worst) and $M{=}28$ (best) — 2 seeds each = 4 runs
- Kvasir or CVC: $M$ at curve max and curve min — 2 seeds each = 4 runs

```bash
# Example for Synapse M=28 seed 2
python train.py --dataset Synapse --vit_name R50-ViT-B_16 \
    --max_epochs 300 --batch_size 24 --seed 2 \
    --gate_mode pam --window_size 28 --rank 32 --groups 8 \
    --val_interval 15 \
    2>&1 | tee ../../logs/synapse_M28_pam_seed2.log
```

**Output:** Would confirm whether the high-GCS span (2.30pp) is significantly larger than low-GCS spans (~0.6pp) under training-seed variance.

---

## E2 — Clean M=14 Ablation ✅ Resolved via Alternative (2026-09-04)

**Why:** Reviewer comment #3: Table VII's M=14 row had r=64, gate=learn — three axes changed simultaneously alongside the window-size sweep.

**Resolution:** Took the no-training alternative — removed the M=14 row from `tab:window_ablation` (Table VII, now clean $M\in\{7,28,56,112\}$, gate=pam, $r{=}32$ throughout). The M=14/r=64/gate=learn data point is kept only in `tab:gate_ablation` (Table XII, "Gate Configurations and Gate-Collapse Robustness Check"), where its role — confirming gate collapse persists outside the default $M{=}7,r{=}32$ setting — is unambiguous. No training run needed.

---

## Summary & Audit Status

| # | Experiment | Est. | Priority | Current Status |
|---|-----------|------|----------|----------------|
| 0 | Regenerate causal/mask PNG (script only) | 0.5h | 🔴 Now | ✅ **Completed** (Vector PDF & PNG in `paper-journal/figures/`) |
| 1a | DA-TransUNet ACDC baseline | 7h | 🔴 High | ✅ **Completed** (RV 88.84, Myo 85.98, LV 90.67, Mean 88.50 in Table II) |
| 1b | DA-TransUNet CVC baseline | 4h | 🔴 High | ⏸️ **Optional** (Published baseline in Table IV) |
| 4 | F8 Entropy gate Synapse | 12h | 🟡 Medium | ⏳ **Pending / Optional** |
| 5 | Re-run causal analysis (5 datasets) | 0.5h | 🟡 After #0 | ✅ **Completed** (Metrics & Table VI updated) |
| E1 | Minimal seed robustness check (8 runs) | 8h | 🟢 Optional | ⏳ Text fallback ✅ applied; experiment optional (reviewer #2) |
| E2 | Clean M=14 ablation / Table VII | 0h | — | ✅ **Completed** (row removed, kept only in Table XII gate ablation) |

> **已取消/移除项**：
> - **F2 (Dataset Size Study)**：已移除（无实际用途，5 数据集 GCS 与 SSD 理论已完备）。
> - **F7 (Kvasir-Instrument / Chest X-ray)**：已移除（SC5=0，不扩展 GCS 谱线）。
