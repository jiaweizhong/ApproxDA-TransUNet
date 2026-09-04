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

## E3 — Statistical Validation (Paired Significance + Bootstrap CI) ✅ Mostly Done from Existing Logs (2026-09-04)

**Why:** BSPC-targeted reviewer advice: paired significance test for performance + bootstrap CI for GCS is a cheap, high-value addition — more systematic than DA-TransUNet's own paper, which only did paired stats on Synapse (n=12).

**Done, zero new training (parsed existing per-case logs):**
- [x] Paired bootstrap 95% CI + Wilcoxon + paired $t$-test, ApproxDA vs.\ DA-TransUNet: **Kvasir-SEG** (n=200, $p$=0.0011) and **ISIC 2018** (n=519, $p$<0.0001) — both significant at α=0.05. Written into `tab:paired_significance` in `05_experiments.tex`.
- [x] Bootstrap 95% CI for GCS: **Synapse** (2.31, CI [1.33, 4.08]pp) and **Kvasir-SEG** (0.64, CI [0.26, 1.80]pp). Written into `tab:gcs_spectrum` in `06_analysis.tex`. Note: CIs overlap partially — honestly disclosed, not oversold.
- [x] Synapse organ-level supplementary check (n=8, using DA-TransUNet's *published* per-organ numbers from `Reference/DA-TransUNet.md` paired with our own M=28 per-organ log): mean +1.12pp, **not significant** (Wilcoxon p=0.109, t-test p=0.092), 6/8 organs favor ApproxDA. Reported transparently as underpowered, not as confirmatory evidence.

**Found a data-integrity issue while doing this:** `results/DA-TransUNet/Synapse/inference-06162026.txt` does NOT reproduce the paper's reported 79.80% test DSC — its own summary line says 72.03%. The training log's val curve (best 79.52%) matches the paper, so this looks like a stale/wrong checkpoint was loaded at test time, not a training failure. `experiments/synapse_volatility_sweep.py` has 79.80 hardcoded as a constant, confirming the correct number was recorded by hand at some point but the matching per-case log is not the one currently committed.

**Still pending (blocked on data, not analysis):**
- [ ] **Synapse case-level paired significance** — needs a fresh, verified DA-TransUNet Synapse checkpoint + test-only inference re-run to regenerate a trustworthy per-case log (no retraining needed *if* the original checkpoint can be recovered; otherwise a full ~11.4h retrain per `training-06162026.txt`, since no weights are currently available). **Decision (2026-09-04): not pursuing before submission** — cost (~11-12h retrain) outweighs benefit given Kvasir/ISIC already carry the significance burden and the organ-level check is disclosed honestly instead.
- [ ] **ACDC GCS bootstrap CI + paired significance** — no per-case DA-TransUNet ACDC log was ever committed (only summary numbers). Would need a per-case-logging re-run of DA-TransUNet ACDC (~7h, already have the checkpoint config, see 1a above) and, for GCS CI, per-case ApproxDA logs across ACDC's tested $M$ values (already exist if a matching per-case-logging test script is used — check `results/ApproxDA-TransUNet/ACDC/test_acdc_M*.txt` format).
- [ ] **CVC-ClinicDB GCS bootstrap CI + paired significance** — DA-TransUNet CVC was never re-run locally (published baseline used); would need a full re-run (~4h, per 1b above) to get a per-case log.

**Placeholders left in the manuscript** (search for `[PLACEHOLDER` in `05_experiments.tex`, `06_analysis.tex`, `07_conclusion.tex`) mark exactly where these three items would slot in if run before submission.

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
| E3 | Paired significance (Kvasir/ISIC) + GCS bootstrap CI (Synapse/Kvasir) | 0h | 🔴 High | ✅ **Completed from existing logs** (Table `tab:paired_significance`, `tab:gcs_spectrum`) |
| E3a | Synapse case-level significance (fix/re-verify baseline log) | 0-11.4h | 🟢 Not pursuing | ⏳ **Deferred** — organ-level check (n=8, honestly non-significant) used instead |
| E3b | ACDC per-case DA-TransUNet log (for GCS CI + paired sig.) | ~7h | 🟡 Optional | ⏳ **Pending / Placeholder in text** |
| E3c | CVC per-case DA-TransUNet log (for GCS CI + paired sig.) | ~4h | 🟡 Optional | ⏳ **Pending / Placeholder in text** |

> **已取消/移除项**：
> - **F2 (Dataset Size Study)**：已移除（无实际用途，5 数据集 GCS 与 SSD 理论已完备）。
> - **F7 (Kvasir-Instrument / Chest X-ray)**：已移除（SC5=0，不扩展 GCS 谱线）。
