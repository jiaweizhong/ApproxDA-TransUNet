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
| Synapse | ✅ Done (6.3h) | ✅ Done (DSC 79.36%, HD95 26.64mm, Params 107.95M, Infer 112.4s/vol, VRAM 0.5GB) — GFLOPs missing (thop failed) | ⏳ Pending | ⏳ Pending |
| Kvasir-SEG | ⏳ Pending | ⏳ Pending | ⏳ Pending | ⏳ Pending |
| ISIC 2018 | ⏳ Pending | ⏳ Pending | ⏳ Pending | ⏳ Pending |

---

## Hardware & Training Settings

All runs on **single NVIDIA Tesla T4 (15 GB VRAM)** via Kaggle free tier.
AdaDA-TransUNet also tested with **T4 x2** to demonstrate multi-GPU scalability.

| Setting | Value |
|---------|-------|
| Optimizer | SGD (momentum=0.9, weight_decay=1e-4) |
| Learning rate | 0.01 with polynomial decay |
| Batch size | 24 |
| Epochs | 150 |
| Seed | 1234 |
| Image size | 224×224 |
| AdaDA config | window=7, rank=32, groups=8 |

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

| Run | GPU | Est. Time | Config |
|-----|-----|-----------|--------|
| AdaDA, no gate | T4 x2 | ~4h | `--disable_gate` |
| AdaDA, rank=8 | T4 x2 | ~4h | `--rank 8` |
| **Week 2 Total** | | **~8h** | |

Weekly quota: 30h. Week 1 uses ~23h, Week 2 uses ~8h — both within limit.

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

## Notebooks

| Notebook file | Purpose |
|--------------|---------|
| `ada-da-transunet.ipynb` | DA-TransUNet: Synapse train + test |
| `adada-transunet.ipynb` | AdaDA-TransUNet: Synapse train + test |
| `da-transunet-kvasir.ipynb` | DA-TransUNet: Kvasir-SEG train + test |
| `adada-transunet-kvasir.ipynb` | AdaDA-TransUNet: Kvasir-SEG train + test |
| `da-transunet-isic.ipynb` | DA-TransUNet: ISIC 2018 train + test |
| `adada-transunet-isic.ipynb` | AdaDA-TransUNet: ISIC 2018 train + test |

---

## Run Order

### Week 1 (main results)
```
Night 1:  DA-TransUNet  Synapse  (T4 x1, ~20min)  ✅ DONE — Params=107.95M, Infer=112.4s/vol, VRAM=0.5GB; GFLOPs still missing (thop failed, need re-run with explicit thop pip install)
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
```
Night 5:  AdaDA no-gate  Synapse  (T4 x2, ~4h)   --disable_gate
Night 6:  AdaDA rank=8   Synapse  (T4 x2, ~4h)   --rank 8
```

---

## Expected Results Table (to fill in paper)

### Synapse Multi-Organ CT

| Method | DSC (%) | HD95 (mm) |
|--------|---------|-----------|
| U-Net | 74.68 | 36.87 |
| TransUNet | 77.48 | 31.69 |
| Swin-Unet | 79.13 | 21.55 |
| DA-TransUNet (paper) | 81.03 | 17.84 |
| DA-TransUNet (ours) | 79.36 | 26.64 |
| **AdaDA-TransUNet (ours)** | **XX** | **XX** |

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
| DA-TransUNet | 107.95 | XX | 6.3h | XX GB | 112.4 | 0.5 GB | No (OOM on T4 x2) |
| AdaDA (r=32, M=7, G=8) | XX | XX | XX h | XX GB | XX | XX GB | Yes (~4h on T4 x2) |

> Params and inference metrics are logged by test.py. Train VRAM is logged at end of train.py.

### Ablation Study (Synapse, T4 x2)

| Config | DSC (%) | HD95 (mm) | Notes |
|--------|---------|-----------|-------|
| DA-TransUNet (full PAM, no gate) | 79.36 | 26.64 | Baseline — confirmed (epoch_149.pth from da-transunet-checkpoints) |
| AdaDA, r=32, fixed gate (0.5) | XX | XX | `--disable_gate` — gate contribution |
| AdaDA, r=8, learned gate | XX | XX | `--rank 8` — rank sensitivity |
| **AdaDA, r=32, learned gate** | **XX** | **XX** | **Full model (ours)** |

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
| 1 | **MICCAI** | A | ~30% | Varies | ~Feb 2027 | Premier medical imaging venue — gold standard in domain |
| 2 | **ACCV** ⭐ | A | ~28% | Osaka, Japan 🇯🇵 | **Jul 5, 2026** | **Our target** — strong general CV, Asia |
| 3 | **ICONIP** | B+ | ~35% | Asia-Pacific | Jun–Jul | Neural computing, Asia |
| 4 | **ACPR** | B | ~40% | Asia | Jul–Aug | Pattern recognition, Asia |
| 5 | **PRICAI** | B | ~40% | Asia-Pacific | Jun–Jul | Pacific Rim AI |
| 6 | **APSIPA ASC** | C | ~50% | Asia-Pacific | Jul–Aug | Signal/image processing |
| 7 | **BMEI** | C+ | ~45% | China | Jul–Aug | IEEE biomedical, China |

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
