# AdaDA-TransUNet Experiment Plan

## Goal

Run experiments on 3 datasets to support submission to **ACCV 2026** (Osaka, Japan, deadline Jul 5, 2026).
Both DA-TransUNet (baseline) and AdaDA-TransUNet are run under identical conditions for fair comparison.

---

## Target Datasets

| # | Dataset | Task | Classes | Split | Metric |
|---|---------|------|---------|-------|--------|
| 1 | **Synapse** | Multi-organ CT segmentation | 9 | 18 train / 12 test | DSC, HD95 |
| 2 | **Kvasir-SEG** | Polyp segmentation (endoscopy) | 2 (binary) | 800 train / 200 test | DSC, mIoU |
| 3 | **ISIC 2018** | Skin lesion segmentation (dermoscopy) | 2 (binary) | 1815 train / 259 val / 520 test | DSC, mIoU |

---

## Status

| Dataset | DA-TransUNet Train | DA-TransUNet Test | AdaDA Train | AdaDA Test |
|---------|-------------------|-------------------|-------------|------------|
| Synapse | ✅ Done (~11.4h pure train, 22.35h wall-clock, T4×1, 300ep, peak VRAM pending) | ✅ Done (DSC 80.51%, HD95 25.41mm, GFLOPs 25.5, Params 107.95M, Infer 121.8s/vol, Infer VRAM 0.5GB) | ✅ Done (T4×1, 3-skip, 300ep, 11.51h pure train, Peak VRAM 10.6 GB, best val epoch 45) | ✅ Done (DSC 77.93%, HD95 33.96mm, Params 114.90M, GFLOPs 27.2, Infer 118.4s/vol, Infer VRAM 0.5GB) |
| Kvasir-SEG | ⏳ Pending | ⏳ Pending | ⏳ Pending | ⏳ Pending |
| ISIC 2018 | ⏳ Pending | ⏳ Pending | ⏳ Pending | ⏳ Pending |

---

## Hardware & Training Settings

All runs on **Lightning AI** (persistent studio, not Kaggle).
- DA-TransUNet baseline: single **NVIDIA Tesla T4 (15 GB VRAM)**
- AdaDA multi-GPU: **T4 × 2** via `torchrun --nproc_per_node=2` (DDP, NCCL, per-GPU batch=12, total=24, LR 0.01 unchanged)
- AdaDA 4×GPU: `torchrun --nproc_per_node=4 train.py --batch_size 6 --n_gpu 4 ...` (per-GPU batch=6, total=24)
- DA-TransUNet multi-GPU failure: **Primary cause is a code bug, not OOM.** `DataParallel` replicates the full model onto each device including the zero-element `query_conv.weight [0,4,1,1]` in `DANetHead(64,64)`, then `BroadcastBackward` during `loss.backward()` fails to synchronize gradients on a zero-element tensor: `RuntimeError: BroadcastBackward got [0] but expected [0,4,1,1]`. Crash occurs at the first backward call regardless of skip routing — confirmed on T4×2 with unmodified code. Even if this architectural issue were patched (clamping C_inter ≥ 8), full PAM at 112×112 would require ~7 GB for the attention matrix alone; combined with the ~10 GB already used by weights + buffers + activations, total ~17 GB exceeds T4 capacity (14.6 GB) by ~2.4 GB. Both issues are documented in the paper appendix.
- AdaDA DDP rationale: NCCL all-reduce avoids `BroadcastBackward`; `LowRankWindowedPAM` has no zero-channel collapse; ~390x memory reduction enables the 112x112 skip.

| Setting | Value |
|---------|-------|
| Optimizer | SGD (momentum=0.9, weight_decay=1e-4) |
| Learning rate | 0.01 with polynomial decay |
| Batch size | 24 (per-GPU 12 for AdaDA 2-GPU) |
| Epochs | 300 |
| val_interval | 15 |
| Checkpoint | best_model.pth (saved when val DSC improves) |
| Seed | 1234 |
| Image size | 224×224 |
| AdaDA baseline config | window=7, rank=32, groups=8 |
| AdaDA recovery config | window=14, rank=64, groups=8 |

---

## Kaggle GPU Budget

### Week 1 — Main Experiments

| Run | GPU | Est. Time | Dataset | Purpose |
|-----|-----|-----------|---------|---------|
| DA-TransUNet Synapse (test-only re-run) | T4 x1 | ~20min | Synapse | Captures Params, GFLOPs, Infer Time, Infer VRAM (upload new code first) |
| AdaDA Synapse | T4 x1 | ~7h | Synapse | Apples-to-apples vs DA baseline |
| AdaDA Synapse | T4 x2 | ~4h | Synapse | Multi-GPU row in efficiency table |
| DA-TransUNet Kvasir | T4 x1 | ~3h | Kvasir-SEG | |
| AdaDA Kvasir | T4 x2 | ~2h | Kvasir-SEG | |
| DA-TransUNet ISIC | T4 x1 | ~4h | ISIC 2018 | |
| AdaDA ISIC | T4 x2 | ~2.5h | ISIC 2018 | |
| **Week 1 Total** | | **~30h** | | Fits within 30h weekly quota |

### Week 2 — Ablation Runs (Synapse only, T4 x2)

Strategy: **recover PAM capability first, then run gate ablation, then decide on gate redesign.**

Gate collapse analysis showed g ≈ 0.5 throughout inference (Δg = 0.0000). Root cause is ambiguous:
- M=7 window → 0.4% spatial coverage → PAM ≈ local filter ≈ CAM → PAM−CAM ≈ 0 → gate gradient ≈ 0
- OR gate architecture is fundamentally broken regardless of PAM quality

Run order: M=14 first (cheapest test, highest expected gain), then gate ablation.

| Run | GPU | Est. Time | Config | Purpose |
|-----|-----|-----------|--------|---------|
| AdaDA, M=14, r=64 | T4 x2 | ~4h | `--window_size 14 --rank 64` | Recover global PAM context |
| AdaDA, gate=fixed | T4 x2 | ~4h | `--gate_mode fixed` | Gate ablation: fixed 0.5 |
| AdaDA, gate=pam | T4 x2 | ~4h | `--gate_mode pam` | Gate ablation: PAM only |
| AdaDA, gate=cam | T4 x2 | ~4h | `--gate_mode cam` | Gate ablation: CAM only |
| **Week 2 Total** | | **~16h** | | |

**Decision after gate ablation:**
- If `fixed ≈ pam ≈ cam ≈ learn` (within ~0.2%) → gate contributes nothing → remove in next version
- If gate=learn >> fixed/pam/cam after M=14 → gate works when PAM/CAM are truly different → keep and study entropy routing
- Entropy gate / diversity loss only if gate ablation confirms gate has value

Weekly quota: 30h. Week 1 uses ~23h, Week 2 uses ~16h — both within limit.

---

## Kaggle Datasets Required

| Purpose | Kaggle Slug | Notes |
|---------|-------------|-------|
| Code | `deepsotaai/adada-transunet-code` | Re-upload after each local code change |
| ViT weights | `deepsotaai/vit-pretrained-weights` | One-time upload |
| Synapse data | `dogcdt/synapse` | Public dataset, already attached |
| Kvasir-SEG | `debeshranaDS/kvasir-seg` | Search on Kaggle, attach to notebook |
| ISIC 2018 | `shonenkov/isic2018` | Search on Kaggle, attach to notebook |
| DA checkpoints | `deepsotaai/da-transunet-checkpoints` | Already uploaded |

---

## Code Changes Required

### 1. Synapse (done)
- `DA-TransUNet/datasets/dataset_synapse.py` — existing, no changes
- `Ada-DA-TransUNet/datasets/dataset_synapse.py` — existing, no changes

### 2. Kvasir-SEG (TODO)
- Add `datasets/dataset_kvasir.py` to both models
  - Load PNG images + PNG masks
  - Binary segmentation: mask > 0 → class 1
  - Augmentation: random flip, rotation, resize to 224×224
- Update `dataset_config` in `train.py` and `test.py`
- Add `lists/lists_Kvasir/train.txt` and `test.txt`

### 3. ISIC 2018 (TODO)
- Add `datasets/dataset_isic.py` to both models
  - Load JPEG images + PNG masks (`_segmentation.png`)
  - Binary segmentation
  - Augmentation: random flip, rotation, colour jitter, resize to 224×224
- Update `dataset_config` in `train.py` and `test.py`
- Add `lists/lists_ISIC/train.txt` and `test.txt`

---

## Run Guide

All experiments run from the Lightning AI Studio **terminal** (not Jupyter notebooks).

| File | Purpose |
|------|---------|
| `notebooks/lightning-ai-setup.md` | Step-by-step terminal commands: one-time setup, all training/test/ablation runs, monitoring |

---

## Run Order

### Week 1 (main results)
```
Night 1:  DA-TransUNet  Synapse  (T4 x1, 22.35h, 300ep)  ✅ DONE — DSC 80.51%, HD95 25.41mm, GFLOPs 25.5, Params=107.95M, Infer=121.8s/vol, VRAM=0.5GB (best_model.pth at epoch 270)
Night 2:  AdaDA         Synapse  (T4 x1, ~7h)    ← apples-to-apples efficiency row
Night 3:  AdaDA         Synapse  (T4 x2, ~4h)    back-to-back with
          DA-TransUNet  Kvasir   (T4 x1, ~3h)    ← ~7h total, within 9h limit
Night 4:  DA-TransUNet  ISIC     (T4 x1, ~4h)    back-to-back with
          AdaDA         Kvasir   (T4 x2, ~2h)    ← ~6h total
Night 5:  AdaDA         ISIC     (T4 x2, ~2.5h)
```
> Night 3 pairs AdaDA T4×2 Synapse (~4h) with DA-TransUNet Kvasir (~3h) in one session.
> Night 4 pairs DA-TransUNet ISIC (~4h) with AdaDA Kvasir (~2h).

### Week 2 (ablation, Synapse only)

**Strategy:** Recover PAM capability first, then isolate gate contribution.

```
Night 5:  AdaDA M=14 r=64   Synapse  (T4 x2, ~4h)   --window_size 14 --rank 64
Night 6:  AdaDA gate=fixed  Synapse  (T4 x2, ~4h)   --gate_mode fixed   (M=7, r=32)
Night 7:  AdaDA gate=pam    Synapse  (T4 x2, ~4h)   --gate_mode pam     (PAM only)
          AdaDA gate=cam    Synapse  (T4 x2, ~4h)   --gate_mode cam     (CAM only)
```

**Decision tree after Week 2:**

| Result | Next action |
|--------|-------------|
| M=14 DSC >> 77.93% (closes gap to ~80%) | Window size was the root cause. Gate ablation still needed to understand gate contribution. |
| gate=fixed ≈ gate=pam ≈ gate=cam ≈ gate=learn | Gate contributes nothing → remove gate in paper, simplify model. |
| gate=learn >> others (after M=14) | Gate works when PAM/CAM truly differ. Consider entropy gate Phase 2. |
| M=14 still << 80.51% | Both window and gate are problems. Investigate rank=128 or deeper redesign. |

---

## Expected Results Table (to fill in paper)

### Synapse Multi-Organ CT

| Method | DSC (%) | HD95 (mm) |
|--------|---------|-----------|
| U-Net | 74.68 | 36.87 |
| TransUNet | 77.48 | 31.69 |
| Swin-Unet | 79.13 | 21.55 |
| DA-TransUNet (paper) | 79.80 | 23.48 |
| DA-TransUNet (ours, 300ep best_model) | 80.51 | 25.41 |
| **AdaDA-TransUNet (ours, 3-skip, T4×1, 300ep)** | **77.93** | **33.96** |

### Kvasir-SEG Polyp

| Method | DSC (%) | mIoU (%) |
|--------|---------|---------|
| DA-TransUNet (ours) | XX | XX |
| **AdaDA-TransUNet (ours)** | **XX** | **XX** |

### ISIC 2018 Skin Lesion

| Method | DSC (%) | mIoU (%) |
|--------|---------|---------|
| DA-TransUNet (ours) | XX | XX |
| **AdaDA-TransUNet (ours)** | **XX** | **XX** |

### Efficiency (Synapse, T4)

| Method | Params (M) | GFLOPs | Train Time | Train VRAM | Infer Time (s/vol) | Infer VRAM | Multi-GPU |
|--------|-----------|--------|-----------|-----------|-------------------|-----------|-----------|
| DA-TransUNet | 107.95 | 25.5 | 22.35h (T4×1, 300ep) | ~11.5 GB | 121.8 | 0.5 GB | No (OOM on T4×2) |
| AdaDA (r=32, M=7, G=8) | 114.90 | 27.2 | 11.51h (T4×1, 300ep) | 10.6 GB | 118.4 | 0.5 GB | Yes (T4×2 pending) |

> Params and inference metrics are logged by test.py. Train VRAM is logged at end of train.py.

### Ablation Study (Synapse)

**Axis 1: Window size / rank** — recover PAM global context

| Config | Window | Rank | DSC (%) | HD95 (mm) | Notes |
|--------|--------|------|---------|-----------|-------|
| DA-TransUNet baseline | — (global) | — | 80.51 | 25.41 | Full PAM+CAM, no gate, T4×1 |
| AdaDA, `--gate_mode learn` | M=7 | r=32 | 77.93 | 33.96 | ✅ T4×1 done (best epoch 45) |
| AdaDA, `--gate_mode learn` | M=14 | r=64 | XX | XX | Week 2 Night 5 — expected main gain |

**Axis 2: Gate mode** — isolate gate contribution (M=7, r=32 baseline)

| Config | `--gate_mode` | DSC (%) | HD95 (mm) | Notes |
|--------|--------------|---------|-----------|-------|
| AdaDA, PAM only | `pam` (g=1) | XX | XX | Week 2 |
| AdaDA, CAM only | `cam` (g=0) | XX | XX | Week 2 |
| AdaDA, fixed blend | `fixed` (g=0.5) | XX | XX | Week 2 |
| AdaDA, learnable gate | `learn` | 77.93 | 33.96 | ✅ Done — gate collapsed (g≈0.5, Δg=0.0000) |

> Gate analysis (analyze_gate_entropy.py): Spearman r=0.787, but gate range [0.4976, 0.5003] (Δ=0.0027%).
> Gate is inert — confirmed collapse. Ablation Axis 2 will quantify whether this matters for accuracy.
> Entropy gate / redesign only considered after gate ablation results show gate has nonzero contribution.

---

## Conference Target

### 🎯 Primary: ACCV 2026
- **Full name:** 18th Asian Conference on Computer Vision
- **Location:** Osaka, Japan 🇯🇵
- **Dates:** December 14–18, 2026
- **Paper deadline:** July 5, 2026 (23:59 GMT) — **27 days away**
- **CORE ranking:** A
- **Acceptance rate:** ~28%
- **Fit:** Computer vision + pattern recognition, medical image segmentation regularly published

### Conference Ranking (medical image segmentation)

| Rank | Conference | CORE | Acceptance | Location | 2026 Deadline | Notes |
|------|-----------|------|------------|----------|---------------|-------|
| 1 | **ACCV** ⭐ | B | ~28% | Osaka, Japan 🇯🇵 | **Jul 5, 2026** | **Our target** — strong general CV, Asia |
| 2 | **PRICAI** | C | ~30-35% | Guangzhou, China | Jun–Jul | Pacific Rim AI |
| 3 | **BIBM 2026** | B | ~19-22% | Dallas, Tx | Jul–Aug | Signal/image processing |
| 4 | **ACPR 2026** | B | ~30% | Asian Pacific | Sep-Oct | Asian Conference on Pattern Recognition |
| 5 | **SPIE Medical Imaging 2027** | C | ~19-22% | Vancouver, CA | August 5 | https://spie.org/MI27/conferencedetails/medical-image-processing |

**Prestige note:** MICCAI ranks higher than ACCV for medical imaging work specifically (it is the dedicated specialist venue, widely cited in clinical AI). Both are CORE A. ACCV is the right target now because:
- MICCAI 2027 deadline has not opened yet
- Our paper's efficiency contribution (windowed attention, low-rank projection) appeals to the broader CV audience at ACCV
- Recommended path: **ACCV 2026 → MICCAI 2027** (extended version with full experiments)

### Journal Extension (after conference acceptance)
**Target:** IEEE JBHI or *Frontiers in Bioengineering and Biotechnology* (same journal as DA-TransUNet)
- Add remaining 2 datasets (Chest X-Ray, CVC-ClinicDB)
- Add full ablation study with real numbers
- Add sensitivity curves (rank r, window M, groups G)
- Extend to ~12 pages
