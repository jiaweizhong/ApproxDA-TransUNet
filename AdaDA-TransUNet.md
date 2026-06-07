# Research Plan: AdaDA-TransUNet

## 1. Title

**AdaDA-TransUNet: Hardware-Aware Adaptive Dual Attention Transformer UNet for Efficient and Precise Medical Image Segmentation**

---

## 2. Dual Attention Network (DANet)

DANet [1] was proposed by Fu et al. for natural scene image segmentation. Its core idea is to simultaneously model **spatial dependencies** (position attention) and **channel dependencies** (channel attention) via two self-attention modules, jointly enhancing semantic consistency and discriminability.

### 2.1 Position Attention Module (PAM)

- Input feature map $A \in \mathbb{R}^{C \times H \times W}$
- Convolutional projections produce $B, C \in \mathbb{R}^{C \times H \times W}$, reshaped to $\mathbb{R}^{C \times N}$ where $N = H \times W$
- Spatial attention map:

$$S_{ji} = \frac{\exp(B_i \cdot C_j)}{\sum_{i=1}^{N} \exp(B_i \cdot C_j)}$$

- Final output:

$$E_j = \alpha \sum_{i=1}^{N} S_{ji} D_i + A_j$$

- Provides **global receptive field** — captures long-range correlations between similar regions regardless of spatial distance.

### 2.2 Channel Attention Module (CAM)

- Input feature map $A \in \mathbb{R}^{C \times H \times W}$, reshaped to $A^{C \times N}$
- Channel attention map:

$$X_{ji} = \frac{\exp(A_i \cdot A_j)}{\sum_{i=1}^{C} \exp(A_i \cdot A_j)}$$

- Output:

$$E_j = \beta \sum_{i=1}^{C} X_{ji} A_i + A_j$$

- Goal: capture cross-channel inter-correlations among semantic feature maps, strengthening channel expressiveness.

---

## 3. DA-TransUNet and Its Limitations

DA-TransUNet [2, 3] incorporates DA blocks into a Transformer U-Net. Its limitations are:

| Limitation | Description |
|---|---|
| High computational complexity | PAM: $\mathcal{O}(N^2)$, CAM: $\mathcal{O}(C^2)$ |
| Fixed compression ratio | Channel compression ratio is hardcoded at 1/4 |
| No hardware-aware adaptation | Cannot adjust inference config based on available device memory |

---

## 4. Our Optimizations (AdaDA-TransUNet)

AdaDA-TransUNet improves DA-TransUNet via four key innovations:

### 4.1 Low-Rank + Local-Aware PAM (reduces $\mathcal{O}(N^2)$)

- Original DANet PAM is full global self-attention: $\mathcal{O}(N^2)$
- Improvement: **Swin-style window attention** [4] + **low-rank decomposition** ($A \approx PQ$)
- Complexity reduced: $\mathcal{O}(N^2) \rightarrow \mathcal{O}(Nr)$

### 4.2 Grouped CAM (reduces $\mathcal{O}(C^2)$)

- Original DANet CAM performs full cross-channel interaction: $\mathcal{O}(C^2)$
- Improvement: split channels into $G$ groups, each performing independent attention (well-suited for multi-class channel representations in medical imaging)
- Complexity reduced: $\mathcal{O}(C^2) \rightarrow \mathcal{O}(C^2/G)$

### 4.3 Learnable Compression Ratio

- DANet uses a fixed channel compression ratio of 1/4 for PAM/CAM computations
- Improvement: introduce a **learnable ratio**:

$$r = \sigma(W \cdot F)$$

  Automatically adapts the channel compression level based on global features, improving adaptability and hardware friendliness.

### 4.4 Hardware-Aware Configuration Module

- Dynamically adjusts **window size / rank / number of groups** based on available device memory at inference time.

---

## 5. Architecture Overview

```
Input Image
    |
[Encoder]  Conv → AdaDA Block → Transformer Layers
    |
[Skip Connections]  with DA Blocks at each scale
    |
[Decoder]  Upsampling → Feature Fusion → Segmentation Head
    |
Segmentation Mask
```

---

## 6. Methodology

### 6.1 Low-Rank PAM

The attention matrix $A$ is approximated via low-rank factorization:

$$A \approx PQ, \quad P \in \mathbb{R}^{N \times r},\ Q \in \mathbb{R}^{r \times N}$$

This reduces the $\mathcal{O}(N^2)$ cost to $\mathcal{O}(Nr)$ where $r \ll N$.

### 6.2 Grouped CAM

$$X_{ji}^{(g)} = \frac{\exp(A_i \cdot A_j)}{\sum_{k=1}^{C/G} \exp(A_i \cdot A_k)}$$

Each group of $C/G$ channels computes attention independently.

### 6.3 Learnable Channel Ratio

$$r = \sigma(W \cdot F), \quad W \in \mathbb{R}^{d \times 1}$$

A sigmoid-activated linear projection over global average-pooled features produces a dynamic compression ratio clamped to $[0.0625,\ 0.5]$.

### 6.4 Hardware-Aware Configuration

```python
def hardware_config(free_mem):
    if free_mem > 8:
        return {"rank": 64, "window": 14, "groups": 4}
    elif free_mem > 4:
        return {"rank": 32, "window": 7,  "groups": 8}
    else:
        return {"rank": 16, "window": 7,  "groups": 16}
```

Higher available memory → larger rank and window size (more expressive, more compute).  
Lower available memory → smaller rank and more groups (leaner, hardware-friendly).

### 6.5 Module Implementations (Corrected PyTorch)

#### 6.5.1 Low-Rank Windowed PAM

```python
class LowRankWindowedPAM(nn.Module):
    """
    Windowed PAM: O(N^2) -> O(N*M^2) via Swin-style windows,
    then Low-rank within each window: O(N*M^2) -> O(N*r).
    Key: keys are projected N -> r via self.proj_r,
         so attention scores are (N, r) instead of (N, N).
    """
    def __init__(self, channels, window_size=7, rank=32):
        super().__init__()
        self.M      = window_size
        N           = window_size ** 2
        self.conv_B = nn.Conv1d(channels, channels, 1)   # query projection
        self.conv_C = nn.Conv1d(channels, channels, 1)   # key projection
        self.conv_D = nn.Conv1d(channels, channels, 1)   # value projection
        self.proj_r = nn.Linear(N, rank, bias=False)     # low-rank: N -> r  (A ≈ PQ)
        self.alpha  = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape
        M = self.M

        # Step 1 — partition into non-overlapping M×M windows
        x_w  = window_partition(x, M)           # (B·nW, C, M, M)
        nBW  = x_w.shape[0]
        x_n  = x_w.view(nBW, C, M * M)          # (B·nW, C, N),  N = M²

        # Step 2 — project B (query), C (key), D (value)
        feat_B = self.conv_B(x_n)               # (B·nW, C, N)
        feat_C = self.conv_C(x_n)               # (B·nW, C, N) — full-rank keys
        feat_D = self.conv_D(x_n)               # (B·nW, C, N)

        # Step 3 — low-rank key/value projection: N -> r  (implements A ≈ PQ)
        C_r = self.proj_r(feat_C)               # (B·nW, C, r)
        D_r = self.proj_r(feat_D)               # (B·nW, C, r)

        # Step 4 — attention scores: B^T @ C_r  →  (B·nW, N, r),  cost O(N·C·r)
        scores = torch.bmm(feat_B.transpose(1, 2), C_r)     # (B·nW, N, r)
        scores = F.softmax(scores, dim=-1)

        # Step 5 — weighted value: scores @ D_r^T  →  (B·nW, N, C)
        E_out = torch.bmm(scores, D_r.transpose(1, 2))      # (B·nW, N, C)

        # Step 6 — residual and window reverse
        E_n = self.alpha * E_out.transpose(1, 2) + x_n      # (B·nW, C, N)
        return window_reverse(E_n.view(nBW, C, M, M), M, H, W)
```

> **Complexity:** standard PAM is $\mathcal{O}(N^2 C)$. With windowing $N_w = M^2 \ll N$, then low-rank projection: $\mathcal{O}(N \cdot r \cdot C)$ total, where $r \ll M^2$.

#### 6.5.2 Grouped CAM

```python
class GroupedCAM(nn.Module):
    """
    Channel attention split into G independent groups: O(C^2) -> O(C^2 / G).
    Each group computes its own Cg×Cg attention matrix (Cg = C // G),
    matching the formula X^(g)_ji = softmax(A_i · A_j) over k=1..C/G.
    """
    def __init__(self, channels, groups=8):
        super().__init__()
        self.G    = groups
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape
        G, Cg = self.G, C // self.G

        # Reshape: treat each group as an independent mini-CAM
        x_g = x.view(B * G, Cg, H * W)          # (B·G, Cg, N)

        # Per-group channel attention matrix: X^(g) ∈ R^(Cg × Cg)
        X = torch.bmm(x_g, x_g.transpose(1, 2)) # (B·G, Cg, Cg)
        X = F.softmax(X, dim=-1)

        # Weighted channel update: E = beta * X^T A + A
        E_g = torch.bmm(X.transpose(1, 2), x_g) # (B·G, Cg, N)
        E   = self.beta * E_g + x_g

        return E.view(B, C, H, W)
```

#### 6.5.3 AdaDA Block (full, corrected)

```python
class AdaDABlock(nn.Module):
    """
    Combines LowRankWindowedPAM + GroupedCAM with a differentiable
    per-channel soft gate that implements the learnable compression ratio.

    Key fix vs. original sketch:
      - No .item() / discrete indexing — gate stays in the computation graph.
      - PAM and CAM run in parallel; the gate blends their outputs adaptively.
    """
    def __init__(self, channels, window_size=7, rank=32, groups=8):
        super().__init__()
        self.pam     = LowRankWindowedPAM(channels, window_size, rank)
        self.cam     = GroupedCAM(channels, groups)

        # Learnable soft gate: r = sigmoid(W · F_global) ∈ (0,1)^C
        # Differentiable — replaces hard channel truncation
        self.pool    = nn.AdaptiveAvgPool2d(1)
        self.gate_fc = nn.Linear(channels, channels)

        self.fusion  = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        pam_out = self.pam(x)                             # (B, C, H, W)
        cam_out = self.cam(x)                             # (B, C, H, W)

        # Soft per-channel gate: g ∈ (0,1)^C — gradient flows through sigmoid
        g = torch.sigmoid(
            self.gate_fc(self.pool(x).view(x.shape[0], -1))
        ).view(x.shape[0], -1, 1, 1)                     # (B, C, 1, 1)

        # Adaptive blend: high gate → trust PAM (spatial); low gate → trust CAM (channel)
        fused = self.fusion(g * pam_out + (1.0 - g) * cam_out)
        return fused + x                                  # residual connection
```

#### 6.5.4 Required Utilities (from Swin Transformer)

```python
def window_partition(x, window_size):
    """(B, C, H, W) -> (B*nW, C, M, M)"""
    B, C, H, W = x.shape
    M = window_size
    x = x.view(B, C, H // M, M, W // M, M)
    # (B, C, nH, M, nW, M) -> (B*nH*nW, C, M, M)
    return x.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, C, M, M)

def window_reverse(windows, window_size, H, W):
    """(B*nW, C, M, M) -> (B, C, H, W)"""
    M  = window_size
    nW = (H // M) * (W // M)
    B  = windows.shape[0] // nW
    C  = windows.shape[1]
    x  = windows.view(B, H // M, W // M, C, M, M)
    return x.permute(0, 3, 1, 4, 2, 5).contiguous().view(B, C, H, W)
```

---

## 7. Loss Function

Combined Dice + Cross-Entropy loss with equal weighting:

$$\mathcal{L} = \frac{1}{2} \cdot \mathcal{L}_{\text{Dice}} + \frac{1}{2} \cdot \mathcal{L}_{\text{CE}}$$

---

## 8. Experiment Plan

### 8.1 Datasets

| Dataset | Type |
|---|---|
| Synapse | Multi-organ CT segmentation |
| ISIC 2018 | Skin lesion segmentation |
| Chest X-ray | Lung segmentation |
| CVC-ClinicDB | Polyp segmentation (colonoscopy) |
| Kvasir-Seg | Polyp segmentation (endoscopy) |

### 8.2 Evaluation Metrics

- **Dice** — overlap-based segmentation accuracy
- **IoU** — intersection over union
- **HD95** — 95th percentile Hausdorff distance (boundary accuracy)
- **Params** — model parameter count
- **FLOPs** — floating point operations (compute cost)
- **FPS** — inference speed

### 8.3 Ablation Studies

| Component | Variants |
|---|---|
| PAM | Global vs. Windowed vs. Low-rank |
| CAM | Full cross-channel vs. Grouped |
| Compression ratio | Fixed (1/4) vs. Learnable |
| Hardware config | With vs. Without |

---

## 9. Key Contributions

1. **First adaptive DA Transformer UNet with hardware-aware configuration** — dynamically scales computation to device constraints at inference time.
2. **Better accuracy-efficiency trade-off** — low-rank PAM and grouped CAM drastically reduce complexity while preserving representational power.
3. **Plug-and-play modules** — AdaDA blocks can be integrated into modern segmentation architectures beyond TransUNet.

### Novelty Map (vs. baselines)

```
DANet (CVPR 2019)
  ├─ PAM: O(N²) global spatial attention  ──► AdaDA: Windowed + Low-rank PAM → O(Nr)   [novel in DA context]
  └─ CAM: O(C²) full channel attention    ──► AdaDA: Grouped CAM → O(C²/G)             [novel in DA context]

DA-TransUNet (2024)
  ├─ Fixed 1/16 compression ratio         ──► AdaDA: Learnable gate r = σ(W·F)         [novel]
  ├─ No hardware adaptation               ──► AdaDA: Hardware-aware config              [engineering contribution]
  └─ No efficiency optimization           ──► AdaDA: All of the above combined          [combined novelty]
```

> **Core novelty claim:** Swin-style and low-rank optimizations are well-studied on vanilla Transformer, but have not been applied specifically to the DA-block in a medical segmentation context. This gap is confirmed by the DA-TransUNet review notes: *"一般的Swin优化都在vanilla Transformer，很少专门针对DA-block做"*.

---

## References

[1] J. Fu, J. Liu, H. Tian, Y. Li, Y. Bao, Z. Fang, and H. Lu, "Dual attention network for scene segmentation," *CVPR*, 2019, pp. 3146–3154.

[2] G. Sun et al., "DA-TransUNet: integrating spatial and channel dual attention with transformer U-Net for medical image segmentation," *Frontiers in Bioengineering and Biotechnology*, vol. 12, p. 1398237, 2024.

[3] A. Dosovitskiy et al., "An image is worth 16x16 words: Transformers for image recognition at scale," *arXiv:2010.11929*, 2020.

[4] Z. Liu et al., "Swin Transformer: Hierarchical vision transformer using shifted windows," *ICCV*, 2021, pp. 10012–10022.
