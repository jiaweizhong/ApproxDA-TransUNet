# Pending Experiments — Journal Extension

> Default config: SGD, lr=0.01 poly, 300ep, bs=24, 224×224, val_interval=15, seed=1234.
> Cleaned up 2026-09-20 — completed items removed. This file now lists ONLY what's still genuinely missing.
> For the full audit trail of what was already resolved (and how), see git history of this file.

---

## 1 — ISIC 2018: ApproxDA-TransUNet, M=28, gate=pam (missing per-case test log)

**Model:** ApproxDA-TransUNet (ours) · **Dataset:** ISIC 2018 · **Type:** test-only IF checkpoint recoverable, otherwise full retrain

**Why:** Completes the GCS bootstrap-CI table (`tab:gcs_spectrum`) — currently 4/5 datasets done (Synapse, ACDC, Kvasir-SEG, CVC-ClinicDB); ISIC is the only one missing. Training already completed once (best val DSC 0.8955 @ epoch 210, checkpoint saved to `epoch_299.pth`), but no per-case test/inference output was ever saved — only a training log (`results/ApproxDA-TransUNet/ISIC18/test_pam_M28.log`, no per-case DSC lines).

**Checkpoint status (checked 2026-09-20):** Not in `C:\Users\jiawe\Downloads\best-porch\` — that folder's `ApproxDA_ISIC224_..._best_model.pth` was verified (via `gate_fc` presence + `proj_r.weight` shape) to actually be **M=7, gate=learn**, not M=28, gate=pam. The correct checkpoint would be at `../model/ApproxDA_ISIC224/ApproxDA_pretrain_R50-ViT-B_16_skip3_epo300_bs24_224_M28_pam/epoch_299.pth` on whatever machine ran the training — check there first.

**If checkpoint found — test-only (minutes):**
```bash
cd experiments/ApproxDA-TransUNet
python test.py \
    --volume_path ../data/ISIC2018 --dataset ISIC --num_classes 2 \
    --list_dir ./lists/lists_ISIC --max_epochs 300 --batch_size 24 --img_size 224 \
    --n_skip 3 --vit_name R50-ViT-B_16 \
    --window_size 28 --rank 32 --groups 8 --gate_mode pam \
    2>&1 | tee ../../results/ApproxDA-TransUNet/ISIC18/test_pam_M28.txt
```

**If checkpoint lost — full retrain (~44.85h, single GPU** — per the original training log's own reported wall-clock time, not an estimate; ISIC training is unusually slow):
```bash
cd experiments/ApproxDA-TransUNet
python train.py --dataset ISIC --vit_name R50-ViT-B_16 \
    --max_epochs 300 --batch_size 24 \
    --gate_mode pam --window_size 28 --rank 32 --groups 8 \
    --val_interval 15 \
    2>&1 | tee ../../logs/isic_pam_M28_300ep.log
# then run the test-only command above once training finishes
```

**Decision needed:** worth 44.85h for one ISIC GCS-CI cell? (Kvasir/Synapse/ACDC/CVC already have CIs; this is the last gap.)

---

## 2 — Synapse: DA-TransUNet baseline (weights lost, committed log doesn't match paper)

**Model:** DA-TransUNet (baseline) · **Dataset:** Synapse · **Type:** full retrain (no usable checkpoint)

**Why:** Enables Synapse case-level (n=12) paired significance testing (`tab:paired_significance`), matching what's already done for Kvasir-SEG/ISIC/ACDC.

**Problem:** `results/DA-TransUNet/Synapse/inference-06162026.txt` does not reproduce the paper's reported 79.80% test DSC (its own summary line says 72.03%) — a stale/wrong checkpoint was evidently loaded at test time. The training log's validation curve (best 79.52%) does match the paper, so training itself was fine, but no usable checkpoint currently exists.

```bash
cd experiments/DA-TransUNet   # or wherever the DA-TransUNet baseline code lives
python train.py --dataset Synapse --vit_name R50-ViT-B_16 \
    --max_epochs 300 --batch_size 24 \
    --val_interval 15 \
    2>&1 | tee ../../logs/synapse_datransunet_300ep.log
# then test-only, saving per-case DSC to a log for pairing
```

**Est.:** ~11.4h (per the original training log).

**Current decision (2026-09-04, unchanged):** **Not pursuing before submission** — cost outweighs benefit. Kvasir-SEG + ISIC already carry the paper's significance-testing burden; a supplementary Synapse organ-level check (n=8, honestly reported as not significant) is used instead. Revisit only if a reviewer specifically pushes back on the missing Synapse case-level test.

---

## 3 — CVC-ClinicDB: DA-TransUNet baseline (never run locally)

**Model:** DA-TransUNet (baseline) · **Dataset:** CVC-ClinicDB · **Type:** full retrain (no prior run exists at all)

**Why:** Enables CVC-ClinicDB case-level paired significance testing. (Its GCS bootstrap CI is already done — that only needed ApproxDA's own logs, not DA-TransUNet's.) The paper currently cites DA-TransUNet's *published* CVC numbers (89.47% DSC, 82.51% mIoU) with no locally-run per-case log to pair against.

```bash
cd experiments/DA-TransUNet
python train.py --dataset CVC --vit_name R50-ViT-B_16 \
    --max_epochs 300 --batch_size 24 \
    --val_interval 15 \
    2>&1 | tee ../../logs/cvc_datransunet_300ep.log
# then test-only, saving per-case DSC to a log for pairing
```

**Est.:** ~4h.

**Current decision:** Optional / not pursued — Kvasir-SEG + ISIC already carry the significance burden; this would be a nice-to-have third confirmed dataset, not a blocker.

---

## 4 — Synapse: Entropy-gate ablation (code not in repo, result unverifiable)

**Model:** ApproxDA-TransUNet, `gate_mode='entropy'` · **Dataset:** Synapse · **Type:** code recovery/reimplementation, then full retrain

**Why:** Would validate H3 (symmetry-breaking gate avoids the g≈0.5 collapse) with a concrete DSC number, for a 1-paragraph "Alternative Gate Designs" subsection.

**Problem (audited 2026-09-20):** A run at `results/ApproxDA-TransUNet/M7R32-ENTROPY-09062026/` reports test DSC **78.87%** (beats gate=pam 78.64% and gate=learn 77.78%) — but `Architecture/block.py`'s `ApproxDABlock` has **no `'entropy'` branch** in its entire git history. Any mode other than `'pam'/'cam'/'fixed'` (including the literal string `'entropy'`) silently falls into the `else: # 'learn'` branch, which requires `self.gate_fc` — a layer only constructed `if gate_mode == "learn"`. Confirmed locally: instantiating `ApproxDABlock(gate_mode='entropy')` and running `forward()` raises `AttributeError: 'ApproxDABlock' object has no attribute 'gate_fc'` immediately. The code that actually produced 78.87% on Lightning AI was never committed back — only the log files were. No gate-value distribution was logged either, so even the code being recovered wouldn't answer F8's core question ("does g avoid 0.5?") without a re-run.

**Status:** Someone who ran this originally is merging the code back (in progress as of 2026-09-20). Once merged:
1. **Code review** (free) — confirm it matches the documented design below, not something else.
2. **If the checkpoint still exists** — test-only re-run to confirm 78.87% actually reproduces (minutes).
3. **Gate-value check** (not previously done, required either way) — load the checkpoint, run inference, log the actual `g` distribution. Without this, "does entropy gating avoid collapse" — F8's entire point — remains unanswered even with a verified DSC number.

Only after all three: add 1 paragraph + 1 table row. **Do not cite 78.87% anywhere until then.**

**Documented design** (for whoever reimplements, if the original code can't be recovered):
```python
def entropy_gate(attn_map):
    # attn_map: (B, N, r) — softmaxed attention weights
    p = attn_map.clamp(min=1e-8)
    H = -(p * p.log()).sum(dim=-1).mean(dim=-1)  # (B,) per-sample entropy
    return H  # lower H -> more focused -> higher gate weight

# In ApproxDABlock.forward():
H_pam = entropy_gate(pam_attn)
H_cam = entropy_gate(cam_attn)
g = torch.softmax(torch.stack([-H_pam, -H_cam], dim=1), dim=1)[:, 0]
g = g.view(B, 1, 1, 1)
fused = self.fusion(g * pam_out + (1 - g) * cam_out)
```
```bash
cd experiments/ApproxDA-TransUNet
python train.py --dataset Synapse --vit_name R50-ViT-B_16 \
    --max_epochs 300 --batch_size 24 \
    --gate_mode entropy --window_size 7 --rank 32 --groups 8 \
    --val_interval 15 \
    2>&1 | tee ../../logs/synapse_entropy_M7_300ep.log
```

**Est.:** ~12h if a full re-run is needed.

---

## Summary

| # | What's missing | Model | Dataset | Inference or retrain? | Est. | Decision |
|---|---|---|---|---|---|---|
| 1 | GCS CI last gap | ApproxDA-TransUNet | ISIC 2018 | Test-only if checkpoint found, else retrain | ~min / 44.85h | Checkpoint search in progress |
| 2 | Case-level significance | DA-TransUNet | Synapse | Retrain (weights lost) | ~11.4h | **Not pursuing** |
| 3 | Case-level significance | DA-TransUNet | CVC-ClinicDB | Retrain (never run) | ~4h | Optional, not pursued |
| 4 | H3 validation number | ApproxDA-TransUNet (entropy gate) | Synapse | Code recovery + possible retrain | ~12h if re-run needed | Blocked on code merge, then re-verify |
