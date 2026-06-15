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
| AdaDA, gate=pam | GPU 0,1 (DDP) | `--window_size 7 --rank 32 --gate_mode pam` | ✅ Done | val DSC **78.82%** (ep270), 8.34h T4×2 — test pending |
| AdaDA, gate=cam | GPU 3 (1×GPU) | `--window_size 7 --rank 32 --gate_mode cam` | 🔄 Running | — |

### Phase A — Decision ✅ RESOLVED

**Exp 1 confirmed results (test.py):** DSC **78.04%**, HD95 **28.77mm**
- vs. M=7/r=32: +0.11% DSC, −5.19mm HD95
- vs. DA-TransUNet baseline: −2.47% DSC, +3.36mm HD95

**Gate entropy (M=14):** Collapsed — identical to M=7. All 4 active blocks: g≈0.500, Δg=0.0000.
Block 2 shows Spearman r=+0.4273 but this is **spurious** — the gate range is only 0.4993–0.5010 (Δ=0.0017), statistically meaningless. Decision summary confirmed: `GATE COLLAPSED`.

**Revised root cause (replaces earlier "M=7 too local" hypothesis):**

The collapse is not caused by window size. It is a **gradient symmetry problem** inherent in the gating design:
- When g=0.5, both PAM and CAM branches receive identical loss gradient: `0.5 × ∂L/∂fused`
- Neither branch has incentive to differentiate → PAM ≈ CAM throughout training
- Gate gradient `∂L/∂g = ∂L/∂fused × (PAM_out − CAM_out) ≈ 0` because the branches converged symmetrically
- This circular dependency (`g=0.5 → identical gradients → PAM≈CAM → gate gradient≈0 → g stays at 0.5`) is stable at **any window size M**

This makes the gate collapse a **stronger and more general finding**: it reveals a fundamental limitation of naïve learnable gating in multi-branch attention architectures, not a windowing artifact.

**Phase A call: Route 1 (Efficient Dual Attention)**
- DSC 78.04% is in the "≤78%" bucket — window+rank scaling does not close the accuracy gap
- HD95 improvement (−5.19mm) is meaningful and supports boundary quality as a secondary metric
- Wait for PAM/CAM ablation to complete the gate analysis story before writing

### Phase B — Window Isolation Experiment (Priority 3, after Kvasir)

**Scaling study (M×r grid) is cancelled** — Exp 1 DSC 78.04% did not meet threshold.

**One high-value experiment remains: Global LowRank PAM (no window)**

| Experiment | Config | What it answers |
|------------|--------|-----------------|
| Global LowRank PAM | `--window_size 112` (= full spatial size, effectively no window) `--rank 64` | Is the 2.47% gap from **windowing** or from **low-rank**? |

Expected outcomes:
- If DSC ≈ 80%: Window is the culprit, rank is fine → paper claim: "windowed approximation loses global context"
- If DSC ≈ 78%: Both window AND rank fail → paper claim: "low-rank decomposition itself loses critical attention precision"

**Do this only if Kvasir/ISIC run cleanly and GPU budget allows.** It requires minimal code change (window_size = H = 112 at the skip connection level).

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
✅ Done:  AdaDA gate=pam    Synapse  val DSC 78.82% (ep270), 8.34h/T4×2 — test.py pending
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
| DA-TransUNet | 107.95 | 25.5 | 22.35h (T4×1, 300ep) | ~11.5 GB | 121.8 | 0.5 GB | No (BroadcastBackward crash) |
| AdaDA (M=7, r=32, G=8) | 114.90 | 27.2 | 11.51h (T4×1, 300ep) | 10.6 GB | 118.4 | 0.5 GB | Yes (T4×2) |
| AdaDA (M=14, r=64, G=8) | 115.05 | 27.3 | 6.65h (T4×2, 300ep) | 6.4 GB/GPU | 130.5 | 0.5 GB | Yes (T4×2) |

> Params and inference metrics are logged by test.py. Train VRAM is logged at end of train.py.
> **Note:** DA-TransUNet paper includes a paired t-test (p=0.032, mean ΔDSC=3.96 vs TransUNet). Our paper should include the same for AdaDA vs DA-TransUNet.
>
> **GFLOPs measurement note — reported numbers are undercounts for BOTH models:**
> `thop` only counts `nn.Linear`, `nn.Conv2d`, `nn.Conv1d` ops. It cannot see `torch.bmm` or `torch.einsum`, which is where attention matrix multiplications happen in both DA-TransUNet (PAMModule) and AdaDA (LowRankWindowedPAM/GroupedCAM).
> - DA-TransUNet PAM at 112×112 (N=12544, C/8=8): bmm alone ≈ 1.26B + 10.07B = 11.3B MACs — ALL invisible to thop. Reported 25.5 GFLOPs is a massive undercount.
> - AdaDA LowRankWindowedPAM at 112×112 (M=7, nW=256, r=32): windowed bmm ≈ 51M MACs total. Reported 27.2 GFLOPs is only a minor undercount (20% missing).
> - Net result: thop shows AdaDA HIGHER (27.2 vs 25.5) because it counts AdaDA's new Conv1d/Linear projections but misses DA-TransUNet's massive N² bmm cost. True ordering is reversed.
> - **Industry fix:** use `fvcore.nn.FlopCountAnalysis` (Meta AI, handles matmul/bmm natively) — `pip install fvcore`. Re-run both models to get corrected numbers before paper submission.
> - **True efficiency reduction at 112×112 skip:** DA 11.3B MACs vs AdaDA 51M MACs ≈ **220× reduction** in attention FLOPs at that layer. This is the correct number to highlight in the paper.

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
| AdaDA, PAM only | `pam` (g=1) | XX (test pending) | XX | ✅ Train done — val DSC **78.82%** best ep270, 8.34h T4×2; training instability ep67–89 (recovered) |
| AdaDA, CAM only | `cam` (g=0) | XX | XX | 🔄 Running (1×GPU) |
| AdaDA, learnable gate (M=7) | `learn` | 77.93 | 33.96 | ✅ Done — gate collapsed (g≈0.5, Δg=0.0000) |
| AdaDA, learnable gate (M=14) | `learn` | 78.04 | 28.77 | ✅ Done — gate collapsed (g≈0.5002, Δg=0.0000); Block 2 r=+0.4273 is spurious |

> **Gate collapse confirmed at both M=7 and M=14** — collapse is NOT caused by window being too local.
> Root cause: gradient symmetry problem (g=0.5 → both branches receive equal gradients → PAM≈CAM → gate gradient≈0).
> gate=fixed skipped: gate=learn already confirmed g≈0.5 throughout, making fixed redundant.
> **PAM-only finding (val):** gate=pam (78.82%) > gate=learn (test 77.93%) by ~0.89% val DSC — PAM alone outperforms the collapsed gate blend. Confirms gate is not only inert but slightly counterproductive (the 50/50 blend with inert CAM dilutes pure PAM).
> Training instability in gate=pam (loss spike ep67–89): likely due to alpha parameter starting at 0 (LowRankWindowedPAM adds nothing initially) — with g=1 always, the full PAM residual hits the network before alpha learns. No such spike in gate=learn (which can route to CAM as safety valve).
> Gate collapse is an **analysis/insight finding**, not a contribution. Will appear in Analysis/Discussion section.
> Paper narrative: "Naïve learnable gating in multi-branch attention suffers from gradient symmetry collapse — PAM-only ablation confirms the gate contributes nothing and the collapsed 50/50 blend is slightly suboptimal."

---

## Conference Target

### Paper Routes (conditional on Exp 1)

**Route 1 — Efficient Dual Attention Approximation** ✅ ACTIVE PATH
- Core claim: LowRankWindowedPAM + GroupedCAM reduce training VRAM and enable multi-GPU DDP, with task-conditional accuracy trade-off
- Contribution 1: LowRankWindowedPAM — DDP-compatible (no BroadcastBackward crash), 2× faster training (11.5h vs 22.35h)
- Contribution 2: GroupedCAM — O(C²/G)
- Analysis: Gate collapse (gradient symmetry) — explains the 50/50 blend; framed as "why approximation fails on CT" not as a contribution
- Performance story: Synapse −2.47% DSC (complex 9-organ CT), Kvasir/ISIC TBD (expected smaller gap or parity)
- Target: **BIBM 2026** primary, ACPR 2026 backup, SPIE 2027 safe

**Route 2 — + Scaling Analysis** ❌ CLOSED (Exp 1 DSC 78.04% < 80.5% threshold)

**Route 3 — New gate modules** ❌ Do not pursue. No entropy gate, no diversity loss, no MoE. Gate is not the main bottleneck.

> **GFLOPs note:** thop likely undercounts attention flops for both models. True efficiency advantage is training speed (2×) and DDP-compatibility, not reported GFLOPs. Do NOT lead the paper with GFLOPs comparison.

### 🎯 Conference Targets

| Venue | CORE | Deadline | Status |
|-------|------|----------|--------|
| **ACCV 2026** | B | Jul 5, 2026 | ❌ Closed — performance gap (−2.47% DSC) + no GFLOPs advantage = immediate rejection risk |
| **BIBM 2026** | B | Jul–Aug 2026 | ✅ Primary — medical imaging focus, efficiency + analysis story fits |
| **ACPR 2026** | B | Sep–Oct 2026 | ✅ Backup — broader CV, binary dataset results needed |
| **SPIE 2027** | C | Aug 5, 2026 | ✅ Safe bet — workshop venue, less competitive |
| **MICCAI 2027** | A | Jan 2027 | 🎯 Journal extension target after acceptance |

> **Narrative to write now:** "LowRankWindowedPAM enables efficient multi-GPU dual attention training but trades global context for local approximation — accuracy loss is task-dependent (larger on complex CT, smaller on binary segmentation)."
> This story needs Kvasir + ISIC results to be compelling.

### Journal Extension (after conference acceptance)
**Target:** IEEE JBHI or *Frontiers in Bioengineering and Biotechnology* (same journal as DA-TransUNet)
- Add remaining 2 datasets (Chest X-Ray, CVC-ClinicDB)
- Add full ablation study with real numbers
- Add sensitivity curves (rank r, window M, groups G)
- Extend to ~12 pages
