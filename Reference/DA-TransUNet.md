# DA-TransUNet: Integrating Spatial and Channel Dual Attention with Transformer U-Net for Medical Image Segmentation

**Published:** 16 May 2024 | *Frontiers in Bioengineering and Biotechnology* 12:1398237  
**DOI:** 10.3389/fbioe.2024.1398237  
**Authors:** Guanqun Sun, Yizhi Pan, Weikun Kong, Zichang Xu, Jianhua Ma, Teeradaj Racharak, Le-Minh Nguyen, Junyi Xin  
**Code:** https://github.com/SUN-1024/DA-TransUnet

---

## 1. Overview

DA-TransUNet integrates Dual Attention (DA-Block = PAM + CAM) into a TransUNet backbone at two locations:
1. **Encoder**: DA-Block placed *before* the ViT Transformer layer to provide image-specific position/channel features as input to the global self-attention.
2. **Skip connections**: DA-Block placed in all 3 skip connection layers to filter redundant features from the encoder before they reach the decoder.

The decoder is **unchanged** from standard TransUNet (CNN up-sampling only).

---

## 2. Architecture Details

### DA-Block
Two parallel branches, each with a 1×C→C/16 bottleneck convolution:
- **PAM branch**: Position Attention Module — O(H²W²C) — captures pixel-wise spatial dependencies via self-attention on H×W position map. Output α₁.
- **CAM branch**: Channel Attention Module — O(C²HW) — captures channel-wise dependencies via self-attention on C×C channel map. Output α₂.
- **Fusion**: `output = Conv(α₁ + α₂)` — summed then projected back to C channels.

**Intermediate channel optimization:** The original DANet used C/4; DA-TransUNet uses **C/16** for medical imaging — empirically best (Table 6).

### Encoder
- 3 CNN convolution blocks (halve spatial resolution, double channels each time)
- DA-Block after convolutions, before embedding
- Embedding layer → Transformer (ViT) layers
- Feature map → 3 skip connections → decoder

### Decoder
Standard TransUNet decoder: 3 up-sampling convolution blocks, feature fusion with skip connections.

---

## 3. Results

### 3.1 Synapse Multi-Organ CT (Table 1)

Trained with SGD (lr=0.01, momentum=0.9, wd=1e-4), 500 epochs, batch=24, image=224×224, patch=16, R50-ViT-B/16 pretrain.

| Model | Year | DSC (%) ↑ | HD (mm) ↓ | Aorta | Gallbladder | Kidney(L) | Kidney(R) | Liver | Pancreas | Spleen | Stomach |
|-------|------|-----------|-----------|-------|-------------|-----------|-----------|-------|----------|--------|---------|
| U-Net | 2015 | 76.85 | 39.70 | 89.07 | 69.72 | 77.77 | 68.60 | 93.43 | 53.98 | 86.67 | 75.58 |
| U-Net++ | 2018 | 76.91 | 36.93 | 88.19 | 68.89 | 81.76 | 75.27 | 93.01 | 58.20 | 83.44 | 70.52 |
| Residual U-Net | 2018 | 76.95 | 38.44 | 87.06 | 66.05 | 83.43 | 76.83 | 93.99 | 51.86 | 85.25 | 70.13 |
| Att-Unet | 2018 | 77.77 | 36.02 | 89.55 | 68.88 | 77.98 | 71.11 | 93.57 | 58.04 | 87.30 | 75.75 |
| MultiResUNet | 2020 | 77.42 | 36.84 | 87.73 | 65.67 | 82.08 | 70.43 | 93.49 | 60.09 | 85.23 | 75.66 |
| TransUNet | 2021 | 77.48 | 31.69 | 87.23 | 63.13 | 81.87 | 77.02 | 94.08 | 55.86 | 85.08 | 75.62 |
| UCTransNet | 2022 | 78.23 | 26.75 | 84.25 | 64.65 | 82.35 | 77.65 | 94.36 | 58.18 | 84.74 | 79.66 |
| TransNorm | 2022 | 78.40 | 30.25 | 86.23 | 65.10 | 82.18 | 78.63 | 94.22 | 55.34 | 89.50 | 76.01 |
| MIM | 2022 | 78.59 | 26.59 | 87.92 | 64.99 | 81.47 | 77.29 | 93.06 | 59.46 | 87.75 | 76.81 |
| Swin-Unet | 2022 | 79.13 | 21.55 | 85.47 | 66.53 | 83.28 | 79.61 | 94.29 | 56.58 | 90.66 | 76.60 |
| **DA-TransUNet** | **2023** | **79.80** | **23.48** | 86.54 | 65.27 | 81.70 | 80.45 | **94.57** | **61.62** | 88.53 | 79.73 |
| Average relative improvement | | +2.03% | −9.00mm | −0.73% | −1.09% | +0.28% | +5.21% | +0.82% | +4.86% | +1.97% | +4.50% |

> DA-TransUNet outperforms TransUNet by **+2.32% DSC** and **−8.21mm HD**. Best in Right Kidney, Liver, Pancreas, Stomach vs TransUNet. Slightly lower in Aorta (−0.69%) and Left Kidney (−0.17%).

### 3.2 Five Additional Datasets (Table 2)

Trained with Adam (lr=1e-3), 500 epochs (50 epochs for Chest Xray and ISIC 2018), batch=24 (75/25 train/test split).

| Dataset | IoU ↑ | Dice ↑ | Notes |
|---------|--------|--------|-------|
| **CVC-ClinicDB** (DA-TransUNet) | **0.8251** | **0.8947** | Best among all compared |
| CVC-ClinicDB (TransUNet) | 0.8163 | 0.8901 | |
| **Chest Xray** (DA-TransUNet) | 0.9317 | **0.9538** | Best Dice |
| Chest Xray (TransUNet) | 0.9301 | 0.9535 | |
| **ISIC2018-Task** (DA-TransUNet) | 0.8278 | **0.8888** | Best Dice |
| ISIC2018-Task (TransUNet) | 0.8263 | 0.8878 | |
| **Kvasir-instrument** (DA-TransUNet) | **0.8973** | **0.9381** | Best |
| Kvasir-instrument (TransUNet) | 0.8926 | 0.9363 | |
| **Kvasir-seg** (DA-TransUNet) | **0.8102** | **0.8847** | Best |
| Kvasir-seg (TransUNet) | 0.8003 | 0.8791 | |

> DA-TransUNet outperforms TransUNet on all 5 datasets. Best on 4/5 datasets among all compared models.

### 3.3 Model Complexity (Table 3)

| Model | Params | Param increase | DSC improvement | HD improvement |
|-------|--------|----------------|-----------------|----------------|
| TransUNet | 105,276,066 (~105.3M) | — | — | — |
| DA-TransUNet | 107,950,840 (~107.95M) | +2.54% | +2.99% | +25.9% |

**Inference speed:** DA-TransUNet 35.98 ms/image; TransUNet 33.58 ms/image (negligible difference).

---

## 4. Ablation Studies

### 4.1 DA-Block Placement: Encoder vs Skip Connections (Table 4)

| Config | Encoder DA | Skip DA | DSC (%) | HD (mm) |
|--------|-----------|---------|---------|---------|
| TransUNet baseline | — | — | 77.48 | 31.69 |
| + Skip DA only | — | ✓ | 78.28 | 29.09 |
| + Encoder DA only | ✓ | — | 78.87 | 27.71 |
| + Both (full DA-TransUNet) | ✓ | ✓ | **79.80** | **23.48** |

**Finding:** Encoder DA (+1.39%) contributes more than Skip DA (+0.80%) alone, but combining both is essential (+2.32% total).

### 4.2 Skip Connection Layer-Wise Ablation (Table 5)

Starting from encoder-DA baseline (78.87%):

| 1st layer | 2nd layer | 3rd layer | DSC (%) | HD (mm) |
|-----------|-----------|-----------|---------|---------|
| — | — | — | 78.87 | 27.71 |
| ✓ | — | — | 79.36 | 25.80 |
| — | ✓ | — | 78.65 | 23.43 |
| — | — | ✓ | 79.49 | 30.71 |
| ✓ | ✓ | ✓ | **79.80** | **23.48** |

**Finding:** The 1st skip layer (+0.49%) and 3rd skip layer (+0.62%) contribute most. All 3 layers together achieves the best balance.

### 4.3 Intermediate Channel Size in DA-Block (Table 6)

| Intermediate channels (fraction of input) | DSC (%) | HD (mm) |
|-------------------------------------------|---------|---------|
| C/1 (same as input) | 78.55 | 28.22 |
| C/2 | 79.35 | 23.77 |
| C/4 (original DANet) | 79.71 | 25.90 |
| C/8 | 79.35 | 25.66 |
| **C/16 (DA-TransUNet choice)** | **79.80** | **23.48** |
| C/32 | 79.71 | 24.45 |

**Finding:** C/16 is optimal for medical image segmentation. Smaller intermediate layers reduce overfitting and focus on most discriminative features.

---

## 5. Statistical Validation (Table 7)

Evaluated on 12 Synapse subsets (40% of data), using U-Net as benchmark:

| Model | Mean DSC ± SD | 95% CI |
|-------|--------------|--------|
| DA-TransUNet | 79.80 ± 5.01 | [74.79, 84.81] |
| TransUNet | 75.84 ± 6.77 | [69.06, 82.61] |

Paired t-test (DA-TransUNet vs TransUNet improvement over U-Net):
- Mean difference: **3.96**, 95% CI: [0.40, 7.53]
- t-statistic: 2.45, **p = 0.032** (statistically significant)
- DA-TransUNet also shows **narrower CI** (more consistent than TransUNet)

---

## 6. Limitations and Future Directions

The paper explicitly states (Section 5.3 and Conclusion):

1. **Computational cost**: DA-Blocks increase complexity — potential obstacle for real-time/resource-constrained use.
2. **Unchanged decoder**: Decoder retains original U-Net architecture; not optimized for DA-TransUNet.
3. **Shallow feature utilization**: *"The utilization of image feature positions and channels is only superficial, with deeper exploration possible."* — **direct call for follow-up research.**
4. **Fine-grained detail loss**: Risk of losing fine-grained details during ViT tokenization (post-convolution/pooling), particularly for thin complex structures.

**Explicit future work directions:**
- Optimize the decoder part
- Reduce computational complexity of DA-Blocks without performance compromise

---

## 7. Implementation Settings

| Setting | Value |
|---------|-------|
| Framework | PyTorch, single NVIDIA RTX 3090 |
| Image size | 256×256 (other datasets), 224×224 (Synapse) |
| Patch size | 16 |
| Optimizer (Synapse) | SGD, lr=0.01, momentum=0.9, wd=1e-4 |
| Optimizer (other) | Adam, lr=1e-3, momentum=0.9, wd=1e-4 |
| Batch size | 24 |
| Epochs | 500 (Synapse, CVC-ClinicDB, Kvasir); 50 (Chest Xray, ISIC 2018) |
| Loss (Synapse) | 0.5 × CrossEntropy + 0.5 × DiceLoss |
| Loss (other) | 0.5 × BCE + 0.5 × DiceLoss |
| Train/test split | 75% / 25% |
| Pretrain | R50-ViT-B/16 |

---

## 8. Gaps — Findings Not Yet in EXPERIMENT_PLAN.md

The following paper results/insights are not captured in the current plan documents:

### 8.1 Missing Reference Numbers for Kvasir-SEG and ISIC 2018

EXPERIMENT_PLAN.md targets Kvasir-SEG and ISIC 2018 but does not list DA-TransUNet paper results as baseline targets. Add to the Expected Results tables:

| Dataset | DA-TransUNet (paper) DSC | DA-TransUNet (paper) IoU |
|---------|--------------------------|--------------------------|
| ISIC 2018 | 0.8888 | 0.8278 |
| Kvasir-seg | 0.8847 | 0.8102 |

These are the numbers to beat for the AdaDA paper.

### 8.2 Per-Organ Synapse Results

EXPERIMENT_PLAN.md only shows aggregate DSC/HD. The paper shows per-organ breakdown (Table 1 above). Our test results should also be broken down per organ — particularly relevant because DA-TransUNet gains most on Pancreas (+5.73%) and Stomach (+4.11%) vs TransUNet, which are the hardest organs. If AdaDA recovers those, it's a strong finding.

### 8.3 Statistical Significance

No statistical test (t-test, CI) is mentioned in EXPERIMENT_PLAN.md. The DA-TransUNet paper includes a paired t-test (p=0.032) validating improvements. Our paper should include the same for AdaDA vs DA-TransUNet.

### 8.4 Ablation Decomposition — DA-Block Encoder vs Skip

The paper's Table 4 decomposition is important context for AdaDA's design:
- Encoder DA alone: +1.39% DSC vs TransUNet baseline
- Skip DA alone: +0.80% DSC
- Both: +2.32% DSC

AdaDA replaces the DA-Block in *all 3 skip connections* (same as DA-TransUNet) plus in the encoder. Our efficiency comes from LowRankWindowedPAM + GroupedCAM, not from removing skip DA.

### 8.5 Paper's Own Future Work = Our Contribution

The paper's conclusion explicitly says: *"deeper exploration of image feature positions and channels is possible."* This is the exact gap AdaDA fills. This quote belongs in AdaDA's Introduction/Motivation section.

### 8.6 Intermediate Channel = C/16 Analogy

DA-TransUNet uses C/16 intermediate channels in DA-Block (vs original DANet's C/4). Our LowRankWindowedPAM uses rank=32 on C=64–768 channels, which is an even more aggressive bottleneck. This parallel should be explicit in the paper's Related Work or Method section.

### 8.7 Our DA-TransUNet Outperforms Paper's Own Numbers

Our trained DA-TransUNet: **80.51% DSC** (300 ep, T4×1) vs paper's **79.80% DSC** (500 ep, RTX 3090). We beat the published number with fewer epochs — worth noting in paper to establish baseline credibility.

### 8.8 Datasets Not in Our Plan (Informational)

The DA-TransUNet paper also evaluates on:
- **CVC-ClinicDB**: Dice 0.8947, IoU 0.8251 (polyp detection, colonoscopy)
- **Chest Xray**: Dice 0.9538, IoU 0.9317 (TB lung segmentation)
- **Kvasir-instrument**: Dice 0.9381, IoU 0.8973 (GI tool segmentation)

These are not in our current plan. Chest Xray and Kvasir-instrument are lower-priority; could add for journal extension (see EXPERIMENT_PLAN.md Journal Extension section).
