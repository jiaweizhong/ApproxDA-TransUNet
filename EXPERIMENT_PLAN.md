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
- DA-TransUNet multi-GPU failure: `DataParallel` crashes at first backward call with `RuntimeError: Function BroadcastBackward returned an invalid gradient at index 395 — got [0] but expected shape compatible with [0, 4, 1, 1]`. Root cause: `DANetHead(64,64)` contains a zero-element tensor (`query_conv.weight [0,4,1,1]`); `DataParallel` replicates it across devices and `BroadcastBackward` cannot synchronize gradients on zero-element tensors. Crash is deterministic regardless of skip routing or batch size — confirmed on T4×2 with unmodified code.
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

### Week 2 — Ablation Runs (Synapse only) — IN PROGRESS

> **gate=fixed removed** — gate=learn already confirmed g≈0.5 (Δg=0.0000), making fixed redundant.

Runs completed / in progress:

| Run | GPUs | Config | Status | Best Val DSC |
|-----|------|--------|--------|-------------|
| AdaDA, M=14, r=64 | GPU 0,1 (DDP) | `--window_size 14 --rank 64 --gate_mode learn` | ✅ Done | DSC **78.04%**, HD95 **28.77mm**, 115.05M params, 27.3 GFLOPs, 6.65h, 6.4 GB/GPU |
| AdaDA, gate=pam | GPU 2,3 (DDP) | `--window_size 7 --rank 32 --gate_mode pam` | 🔄 Running | — |
| AdaDA, gate=cam | GPU 3 (1×GPU) | `--window_size 7 --rank 32 --gate_mode cam` | 🔄 Running | — |

### Phase A — Decision after Exp 1

**Exp 1 result (val DSC 78.04% at ep 210):** Marginal improvement over M=7/r=32 baseline (+0.11% vs 77.93% test DSC). Window scaling alone is not recovering the gap to DA-TransUNet (80.51%).

**Root cause hypothesis:** Even M=14 window covers only 14²/112² = 1.56% of spatial positions (vs 0.39% for M=7). Original DA-TransUNet uses full global attention (100% coverage). LowRankWindowedPAM is still fundamentally local — increasing M or r beyond this point yields diminishing returns.

> **Next action: Run test.py on best_model.pth to confirm official test DSC before making Phase A call.**
>
> Snapshot path: `../model/AdaDA_Synapse224/AdaDA_pretrain_R50-ViT-B_16_skip3_epo300_bs12_224_M14_r64/`
>
> ```bash
> cd /teamspace/studios/this_studio/AdaDA-TransUNet/experiments/Ada-DA-TransUNet
> python test.py --dataset Synapse --vit_name R50-ViT-B_16 --max_epochs 300 --batch_size 12 \
>   --n_skip 3 --img_size 224 --window_size 14 --rank 64 --groups 8 --gate_mode learn
> ```

| Exp 1 test DSC | Phase A decision |
|----------------|-----------------|
| **≥ 80.5%** | Window+LowRank validated. Enter Phase B (scaling study). |
| **79–80.4%** | Meaningful gain. Run M=14, r=128 to find upper bound. |
| **78–79%** | Modest gain. Window scaling has limits. Wait for PAM/CAM ablation results before deciding. |
| **≤ 78%** | Window+LowRank path has fundamental limits. Efficiency story is the paper (Route 1). |

**While waiting for test.py:** PAM and CAM ablation results will reveal whether the gate or the PAM architecture is the bottleneck. If PAM-only ≪ CAM-only, windowed PAM is fundamentally broken for global context.

### Phase B — Scaling Study (only if Exp 1 ≥ 80.5%)

Two targeted runs first to answer: **window or rank — which matters more?**

| Run | Config | Answers |
|-----|--------|---------|
| M=7, r=64 | `--window_size 7 --rank 64` | Rank effect in isolation |
| M=14, r=32 | `--window_size 14 --rank 32` | Window effect in isolation |

Then if trend is clear, add r=128 as upper bound. Do NOT run all 6 grid points upfront.

**Do not plan further grid runs until Phase B results arrive.**

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

### Week 2 (ablation, Synapse only) — IN PROGRESS

```
✅ Done:  AdaDA M=14 r=64   Synapse  DSC 78.04%, HD95 28.77mm (+0.11% DSC, -5.19mm HD vs M=7/r=32)
Running:  AdaDA gate=pam    Synapse  (4-GPU host, GPU 2+3)   --gate_mode pam
Running:  AdaDA gate=cam    Synapse  (4-GPU host, GPU 3, 1×GPU)  --gate_mode cam
```

**Exp 1 (M=14, r=64) DSC is the project pivot. Everything below is conditional on that result.**

| Exp 1 result | Weeks 3–4 path |
|-------------|----------------|
| ≥ 80.5% | Scaling study (M=7/r=64, M=14/r=32 first; r=128 only if trend is clear) → Kvasir + ISIC → BIBM/ACCV |
| ~79% | Run M=14, r=128 (find upper bound) → assess venue |
| ≤ 78% | Window+LowRank path has limits → SPIE/ACPR or pivot |

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
| TransUNet | 87.91 | 80.03 |
| DA-TransUNet (paper, 500ep Adam) | 88.47 | 81.02 |
| DA-TransUNet (ours) | XX | XX |
| **AdaDA-TransUNet (ours)** | **XX** | **XX** |

### ISIC 2018 Skin Lesion

| Method | DSC (%) | mIoU (%) |
|--------|---------|---------|
| TransUNet | 88.78 | 82.63 |
| DA-TransUNet (paper, 50ep Adam) | 88.88 | 82.78 |
| DA-TransUNet (ours) | XX | XX |
| **AdaDA-TransUNet (ours)** | **XX** | **XX** |

### Efficiency (Synapse, T4)

| Method | Params (M) | GFLOPs | Train Time | Train VRAM | Infer Time (s/vol) | Infer VRAM | Multi-GPU |
|--------|-----------|--------|-----------|-----------|-------------------|-----------|-----------|
| DA-TransUNet | 107.95 | 25.5 | 22.35h (T4×1, 300ep) | ~11.5 GB | 121.8 | 0.5 GB | No (OOM on T4×2) |
| AdaDA (r=32, M=7, G=8) | 114.90 | 27.2 | 11.51h (T4×1, 300ep) | 10.6 GB | 118.4 | 0.5 GB | Yes (T4×2 pending) |

> Params and inference metrics are logged by test.py. Train VRAM is logged at end of train.py.
> **Note:** DA-TransUNet paper includes a paired t-test (p=0.032, mean ΔDSC=3.96 vs TransUNet). Our paper should include the same for AdaDA vs DA-TransUNet.

### Ablation Study (Synapse)

**Axis 1: Window size / rank** — recover PAM global context

| Config | Window | Rank | DSC (%) | HD95 (mm) | Notes |
|--------|--------|------|---------|-----------|-------|
| DA-TransUNet baseline | — (global) | — | 80.51 | 25.41 | Full PAM+CAM, no gate, T4×1 |
| AdaDA, `--gate_mode learn` | M=7 | r=32 | 77.93 | 33.96 | ✅ T4×1 done (best epoch 45) |
| AdaDA, `--gate_mode learn` | M=14 | r=64 | **78.04** | **28.77** | ✅ Done (6.65h/2×GPU, 6.4GB/GPU VRAM, 115.05M, 27.3 GFLOPs) |

**Axis 2: Gate mode** — isolate gate contribution (M=7, r=32 baseline)

| Config | `--gate_mode` | DSC (%) | HD95 (mm) | Notes |
|--------|--------------|---------|-----------|-------|
| AdaDA, PAM only | `pam` (g=1) | XX | XX | 🔄 Running |
| AdaDA, CAM only | `cam` (g=0) | XX | XX | 🔄 Running (1×GPU) |
| AdaDA, learnable gate | `learn` | 77.93 | 33.96 | ✅ Done — gate collapsed (g≈0.5, Δg=0.0000) |

> gate=fixed skipped: gate=learn already confirmed g≈0.5 throughout, making fixed redundant.
> Gate collapse is an **analysis finding**, not a contribution. Will appear in Discussion/Analysis section.
> No new gate modules planned until scaling study confirms Window+LowRank path is viable.

---

## Conference Target

### Paper Routes (conditional on Exp 1)

**Route 1 — Efficient Dual Attention** (safest, Exp 1 any result)
- Contribution 1: LowRankWindowedPAM — O(N²) → O(N·M²·r)
- Contribution 2: GroupedCAM — O(C²) → O(C²/G)
- Analysis: gate collapse as mechanistic insight (not contribution)
- Target: BIBM / SPIE / ACPR

**Route 2 — Efficient Dual Attention + Scaling Analysis** (requires Exp 1 ≥ 80.5%)
- Contribution 1: LowRankWindowedPAM
- Contribution 2: GroupedCAM
- Contribution 3: Systematic scaling study (M × r → accuracy/efficiency tradeoff)
- Insight section: gate collapse + routing collapse analysis (why, not just what)
- Target: BIBM primary, ACCV if mechanism analysis is strong

**Route 3 — New gate modules** ❌ Do not pursue. Gate story is dead without evidence.

### 🎯 Conference Targets

| Venue | CORE | Deadline | Probability if Exp1 ≥ 80.5% |
|-------|------|----------|------------------------------|
| **BIBM 2026** | B | Jul–Aug | 75–85% |
| **PRICAI 2026** | C | Jun–Jul | 80% |
| **ACPR 2026** | B | Sep–Oct | 85% |
| **SPIE 2027** | C | Aug 5 | 95% |
| **ACCV 2026** | B | Jul 5, 2026 | 45–60% (needs strong mechanism story) |

> ACCV deadline is Jul 5, 2026. With experiments still running, this is extremely tight.
> Realistic primary target: **BIBM 2026** (Jul–Aug deadline, medical imaging focus).
> Recommended path: **BIBM 2026 → MICCAI 2027** (extended version).

### Journal Extension (after conference acceptance)
**Target:** IEEE JBHI or *Frontiers in Bioengineering and Biotechnology* (same journal as DA-TransUNet)
- Add remaining 2 datasets (Chest X-Ray, CVC-ClinicDB)
- Add full ablation study with real numbers
- Add sensitivity curves (rank r, window M, groups G)
- Extend to ~12 pages
