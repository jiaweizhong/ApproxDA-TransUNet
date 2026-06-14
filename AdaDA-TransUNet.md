# Research Plan: AdaDA-TransUNet

## 1. Title

**AdaDA-TransUNet: Entropy-Informed Adaptive Dual Attention Transformer UNet for Efficient Medical Image Segmentation**

---

## 2. Background: Dual Attention Network (DANet)

DANet [1] models **spatial dependencies** (Position Attention Module, PAM) and **channel dependencies** (Channel Attention Module, CAM) via two self-attention branches, jointly enhancing semantic consistency.

### 2.1 Position Attention Module (PAM)

- Input $A \in \mathbb{R}^{C \times H \times W}$; projections produce $B, C, D \in \mathbb{R}^{C \times N}$ where $N = HW$
- Spatial attention: $S_{ji} = \frac{\exp(B_i \cdot C_j)}{\sum_i \exp(B_i \cdot C_j)}$
- Output: $E_j = \alpha \sum_i S_{ji} D_i + A_j$ — global receptive field, $\mathcal{O}(N^2 C)$

### 2.2 Channel Attention Module (CAM)

- Channel attention: $X_{ji} = \frac{\exp(A_i \cdot A_j)}{\sum_i \exp(A_i \cdot A_j)}$
- Output: $E_j = \beta \sum_i X_{ji} A_i + A_j$ — cross-channel correlations, $\mathcal{O}(C^2 N)$

---

## 3. DA-TransUNet and Its Limitations

| Limitation | Description |
|---|---|
| High computational complexity | PAM: $\mathcal{O}(N^2)$, CAM: $\mathcal{O}(C^2)$ |
| Fixed equal weighting | PAM and CAM always contribute equally regardless of feature uncertainty or depth |
| No hardware-aware adaptation | Cannot scale computation to available device memory |

---

## 4. Our Contributions (AdaDA-TransUNet)

### Contribution 1: Low-Rank Windowed PAM — $\mathcal{O}(N^2) \rightarrow \mathcal{O}(Nr)$

Swin-style window partitioning restricts attention to local $M \times M$ patches; low-rank projection ($A \approx PQ$, rank $r \ll M^2$) further reduces per-window cost.

$$\text{Complexity: } \mathcal{O}(N^2 C) \rightarrow \mathcal{O}(N r C)$$

### Contribution 2: Grouped CAM — $\mathcal{O}(C^2) \rightarrow \mathcal{O}(C^2/G)$

Split $C$ channels into $G$ groups; each group computes an independent $(C/G) \times (C/G)$ attention matrix. Well-suited to multi-class semantic representations in medical imaging.

$$X_{ji}^{(g)} = \frac{\exp(A_i \cdot A_j)}{\sum_{k=1}^{C/G} \exp(A_i \cdot A_k)}$$

### Contribution 3: Entropy-Informed Adaptive Routing

The original DA-TransUNet assumes equal contribution of PAM and CAM:

$$F_{\text{DA}} = \text{PAM}(F) + \text{CAM}(F)$$

AdaDA replaces this with a differentiable routing gate that takes **feature entropy** as an additional signal:

$$F_{\text{AdaDA}} = g \cdot \text{PAM}(F) + (1 - g) \cdot \text{CAM}(F)$$

$$g = \sigma\!\left(W_g \cdot \left[\text{GAP}(F),\ H(F)\right]\right) \in \mathbb{R}^{B \times C}$$

where:
- $\text{GAP}(F) \in \mathbb{R}^{B \times C}$ — global average pooling (feature content signal)
- $H(F) \in \mathbb{R}^{B \times 1}$ — per-sample feature entropy:

$$p = \text{softmax}(F_{\text{flat}}),\quad H(F) = -\frac{1}{C}\sum_c \sum_i p_{c,i} \log p_{c,i}$$

- $W_g \in \mathbb{R}^{C \times (C+1)}$ — learnable weight; one extra input dimension for the entropy scalar

**Motivation:** High-entropy features (blurry boundaries, lesion borders, ambiguous transitions) carry high spatial uncertainty → gate should weight PAM more. Low-entropy features (homogeneous organs, stable anatomy) have confident semantics → gate should weight CAM more.

**Parameter cost:** adds one column to $W_g$ (C extra parameters) plus $\mathcal{O}(CHW)$ entropy computation — negligible vs. PAM/CAM.

---

## 5. Architecture

```
Input Feature F
       |
  +----+----+
  |         |
PAM(F)    CAM(F)
  |         |
  +----+----+
       |
  [Entropy Gate]
  GAP(F) ──┐
  H(F)  ───┤──► Wg ──► sigmoid ──► g (B,C,1,1)
            |
  g·PAM + (1−g)·CAM
       |
  Conv1×1 fusion
       |
   + residual
       |
   Output
```

---

## 6. Implementation

### 6.1 Low-Rank Windowed PAM

```python
class LowRankWindowedPAM(nn.Module):
    def __init__(self, channels, window_size=7, rank=32):
        super().__init__()
        self.M      = window_size
        N           = window_size ** 2
        self.conv_B = nn.Conv1d(channels, channels, 1)
        self.conv_C = nn.Conv1d(channels, channels, 1)
        self.conv_D = nn.Conv1d(channels, channels, 1)
        self.proj_r = nn.Linear(N, rank, bias=False)   # low-rank: N -> r
        self.alpha  = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape
        M = self.M
        x_w   = window_partition(x, M)                        # (B*nW, C, M, M)
        nBW   = x_w.shape[0]
        x_n   = x_w.view(nBW, C, M * M)                      # (B*nW, C, N)
        feat_B = self.conv_B(x_n)
        feat_C = self.conv_C(x_n)
        feat_D = self.conv_D(x_n)
        C_r    = self.proj_r(feat_C)                          # (B*nW, C, r)
        D_r    = self.proj_r(feat_D)
        scores = torch.bmm(feat_B.transpose(1, 2), C_r)      # (B*nW, N, r)
        scores = F.softmax(scores, dim=-1)
        E_out  = torch.bmm(scores, D_r.transpose(1, 2))      # (B*nW, N, C)
        E_n    = self.alpha * E_out.transpose(1, 2) + x_n
        return window_reverse(E_n.view(nBW, C, M, M), M, H, W)
```

### 6.2 Grouped CAM

```python
class GroupedCAM(nn.Module):
    def __init__(self, channels, groups=8):
        super().__init__()
        self.G    = groups
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape
        G, Cg = self.G, C // self.G
        x_g = x.contiguous().view(B * G, Cg, H * W)
        X   = torch.bmm(x_g, x_g.transpose(1, 2))
        X   = F.softmax(X, dim=-1)
        E_g = torch.bmm(X.transpose(1, 2), x_g)
        E   = self.beta * E_g + x_g
        return E.contiguous().view(B, C, H, W)
```

### 6.3 AdaDABlock with Entropy Gate

```python
class AdaDABlock(nn.Module):
    def __init__(self, channels, window_size=7, rank=32, groups=8, disable_gate=False):
        super().__init__()
        self.disable_gate = disable_gate
        self.pam    = LowRankWindowedPAM(channels, window_size, rank)
        self.cam    = GroupedCAM(channels, groups)
        self.pool   = nn.AdaptiveAvgPool2d(1)
        if not disable_gate:
            self.gate_fc = nn.Linear(channels + 1, channels)  # +1 for entropy scalar
        self.fusion = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        pam_out = self.pam(x)
        cam_out = self.cam(x)
        if self.disable_gate:
            g = 0.5
        else:
            gap  = self.pool(x).view(x.shape[0], -1)           # (B, C)
            prob = F.softmax(x.flatten(2), dim=-1)              # (B, C, N)
            ent  = (-prob * torch.log(prob + 1e-6)).sum(-1).mean(1, keepdim=True)  # (B, 1)
            g    = torch.sigmoid(
                self.gate_fc(torch.cat([gap, ent], dim=1))
            ).view(x.shape[0], -1, 1, 1)                       # (B, C, 1, 1)
        fused = self.fusion(g * pam_out + (1.0 - g) * cam_out)
        return fused + x
```

### 6.4 Hardware-Aware Configuration (Implementation Detail)

Adjusts rank / window size / groups at inference time based on free GPU memory. Not a claimed contribution — an engineering convenience for deployment on different hardware.

```python
def hardware_config(free_mem_gb):
    if free_mem_gb > 8:
        return {"rank": 64, "window_size": 14, "groups": 4}
    elif free_mem_gb > 4:
        return {"rank": 32, "window_size": 7,  "groups": 8}
    else:
        return {"rank": 16, "window_size": 7,  "groups": 16}
```

---

## 7. Loss Function

$$\mathcal{L} = \frac{1}{2} \mathcal{L}_{\text{Dice}} + \frac{1}{2} \mathcal{L}_{\text{CE}}$$

---

## 8. Ablation Study Design

### Gate input ablation (primary)

| Config | Gate input | Purpose |
|--------|-----------|---------|
| `--disable_gate` | fixed 0.5 | remove gate entirely |
| GAP-only gate | $\text{GAP}(F)$ | baseline learned gate (currently training) |
| **GAP + entropy gate** | $[\text{GAP}(F),\ H(F)]$ | **full model (Phase 2)** |

### Efficiency ablation

| Config | Purpose |
|--------|---------|
| `--rank 8` | rank sensitivity |
| `--groups 4` | group count sensitivity |

### Visualization (paper figures)

1. **Scatter plot**: $H(F)$ vs. mean gate $g$ per AdaDA block — should show positive Spearman correlation
2. **Spatial entropy map**: one CT slice with entropy heatmap — boundary pixels should have higher $H$ and higher $g$
3. **Gate distribution per depth**: histogram of $g$ values at each decoder stage — should shift toward PAM (higher $g$) in shallow stages if entropy is informative

---

## 9. Key Contributions (for paper submission)

1. **Low-Rank Windowed PAM**: reduces $\mathcal{O}(N^2 C)$ to $\mathcal{O}(NrC)$ via Swin-style windowing + low-rank key/value projection — first application to DA-block in medical segmentation.
2. **Grouped CAM**: reduces $\mathcal{O}(C^2)$ to $\mathcal{O}(C^2/G)$ — preserves multi-class channel structure.
3. **Entropy-Informed Adaptive Routing**: adds feature entropy $H(F)$ to the attention routing gate, making PAM/CAM allocation uncertainty-aware — high-entropy boundaries get more spatial attention, high-confidence regions get more channel attention.

**Why this matters for ACCV reviewers:** The windowed + low-rank PAM directly enables multi-GPU DDP training (no `BroadcastBackward` on zero-element weights); the entropy gate adds interpretable, principled uncertainty-awareness with ~$C$ extra parameters.

---

## 10. Verification Protocol (before Phase 2)

Run `analyze_gate_entropy.py` on the GAP-only checkpoint (`best_model.pth`).
The script computes **both** Spearman($H$, $g$) and Spearman($\text{Var}$, $g$) so all backup plans are covered in one pass.

```bash
python analyze_gate_entropy.py \
  --vit_name R50-ViT-B_16 --n_skip 3 --max_epochs 300 --batch_size 24 \
  --window_size 7 --rank 32 --groups 8
```

### Case 1 — Strong positive correlation ($r > 0.5$, $p < 0.01$)

Gate already tracks entropy implicitly. Narrative: *"Existing adaptive gate naturally correlates with feature uncertainty."* Adding $H(F)$ as explicit input formalises what the network already discovered.

**Action:** Proceed directly to Phase 2. Paper claim is strong.

### Case 2 — Weak positive correlation ($0.2 < r < 0.5$, $p < 0.01$)

Entropy signal has supplementary value but is not dominant. Most likely outcome (~50%).

**Action:** Proceed to Phase 2 with entropy gate. Narrative: *"Explicit entropy input strengthens the uncertainty signal already weakly captured by GAP."*

### Case 3 — Near-zero correlation ($|r| < 0.1$)

Gate does not track entropy. Two sub-cases:
- Entropy is the wrong proxy for what matters
- Gate has not learned a meaningful uncertainty signal at all

**Action:** Do **not** add entropy gate. Run variance correlation from the same script output. If Var($F$) correlates ($r > 0.2$), switch to Plan B1. Otherwise Plan B3.

### Case 4 — Negative correlation ($r < 0$)

High-entropy features produce **lower** gate values → CAM is weighted more at uncertain boundaries, not PAM. This inverts the hypothesis but is scientifically interesting.

**Action:** Investigate per-block. If consistent across blocks, revise Contribution 3 to: *"High-uncertainty regions preferentially activate channel attention for semantic disambiguation"* — still a publishable finding. Then implement with the same gate architecture (the gate learns the correct direction regardless of our prior).

---

## 11. Backup Plans (if entropy correlation is weak)

### Plan B1 — Feature Variance Gate (recommended first fallback)

Replace $H(F)$ with per-sample mean spatial variance, which is cheaper ($\mathcal{O}(CN)$, no softmax) and often more stable in medical images:

$$\text{Var}(F) = \frac{1}{C} \sum_c \text{Var}_{\text{spatial}}(F_c) \in \mathbb{R}^{B \times 1}$$

$$g = \sigma\!\left(W_g \cdot \left[\text{GAP}(F),\ \text{Var}(F)\right]\right)$$

Code change is identical to Phase 2 — swap `ent` computation:
```python
var = x.var(dim=[2, 3]).mean(dim=1, keepdim=True)   # (B, 1)
g = torch.sigmoid(self.gate_fc(torch.cat([gap, var], dim=1))).view(...)
```

### Plan B2 — Boundary-Aware Gate

Use Laplacian edge energy as the routing signal — directly meaningful for boundary-sensitive segmentation (Synapse, ISIC, Kvasir):

```python
lap_kernel = torch.tensor([[0,1,0],[1,-4,1],[0,1,0]], dtype=x.dtype, device=x.device)
lap_kernel = lap_kernel.view(1,1,3,3).expand(C,-1,-1,-1)
edge = F.conv2d(x, lap_kernel, groups=C, padding=1).abs().mean(dim=[2,3])  # (B, C)
# concat with GAP: gate_fc = Linear(2*C, C)
```

Higher parameter cost ($2C$ input); best suited if boundary-specific ablations are needed.

### Plan B3 — Multi-Statistic Gate (most robust, best for ablation story)

Concatenate GAP + entropy + variance; let the network learn which signals matter:

$$g = \sigma\!\left(W_g \cdot \left[\text{GAP}(F),\ H(F),\ \text{Var}(F)\right]\right), \quad W_g \in \mathbb{R}^{C \times (C+2)}$$

Produces a clean 4-row ablation table:

| Gate input | Params added | Purpose |
|-----------|-------------|---------|
| GAP only | 0 | current baseline |
| GAP + $H$ | $C$ | entropy contribution |
| GAP + Var | $C$ | variance contribution |
| GAP + $H$ + Var | $2C$ | full uncertainty gate |

**Recommended path:** run Phase 1 script → if $r_H > 0.3$ use Plan (Phase 2 entropy); else if $r_{\text{Var}} > 0.2$ use Plan B1; otherwise go Plan B3 and let ablation data speak.

---

## 12. Figure Plan (paper)

1. **Gate–Entropy scatter**: $H(F)$ vs. mean $g$ per AdaDA block — Spearman $r$ annotated
2. **Spatial entropy map**: CT slice with per-pixel $H$ heatmap overlaid — boundary pixels should be high-entropy
3. **Gate distribution per depth**: boxplot of $g$ values at each decoder stage (Encoder, Skip3, Skip2, Skip1) — tests depth-dependence hypothesis from ESDA
4. **Ablation bar chart**: DSC across gate configurations — visual summary of gate contribution

---

## References

[1] J. Fu et al., "Dual attention network for scene segmentation," *CVPR*, 2019.

[2] G. Sun et al., "DA-TransUNet: integrating spatial and channel dual attention with transformer U-Net for medical image segmentation," *Frontiers in Bioengineering and Biotechnology*, vol. 12, 2024.

[3] A. Dosovitskiy et al., "An image is worth 16x16 words," *arXiv:2010.11929*, 2020.

[4] Z. Liu et al., "Swin Transformer: Hierarchical vision transformer using shifted windows," *ICCV*, 2021.
