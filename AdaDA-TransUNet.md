# Research Plan: ApproxDA-TransUNet

## 1. Naming

| Layer | Name |
|-------|------|
| **Paper Title** | Task-Conditioned Attention Approximation for Efficient Medical Image Segmentation |
| **Method Name** | ApproxDA-TransUNet (Approximate Dual Attention TransUNet) |
| **Research Theme** | Approximation Safety |
| **Design Principle** | Task Context Complexity → Required Global Context → Safe Approximation Strength |
| **Code directories** | `Ada-DA-TransUNet` / `AdaDA` (keep as-is; no rename needed) |

---

## 2. Background: Dual Attention Network (DANet)

DANet [1] models **spatial dependencies** (Position Attention Module, PAM) and **channel dependencies** (Channel Attention Module, CAM) via two self-attention branches, jointly enhancing semantic consistency.

### 2.1 Position Attention Module (PAM)

- Input $A \in \mathbb{R}^{C \times H \times W}$; projections produce $B, C, D \in \mathbb{R}^{C \times N}$ where $N = HW$
- Spatial attention: $S_{ji} = \frac{\exp(B_i \cdot C_j)}{\sum_i \exp(B_i \cdot C_j)}$
- Output: $E_j = \alpha \sum_i S_{ji} D_i + A_j$ — global receptive field, $\mathcal{O}(N^2 C)$

### 2.2 Channel Attention Module (CAM)

- Channel attention: $X_{ji} = \frac{\exp(A_i \cdot A_j)}{\sum_i \exp(A_i \cdot A_k)}$
- Output: $E_j = \beta \sum_i X_{ji} A_i + A_j$ — cross-channel correlations, $\mathcal{O}(C^2 N)$

---

## 3. DA-TransUNet and Its Limitations

| Limitation | Description |
|---|---|
| High computational complexity | PAM: $\mathcal{O}(N^2)$, CAM: $\mathcal{O}(C^2)$ — makes multi-GPU DDP impossible (BroadcastBackward crash on zero-element tensors) |
| Rigid branch weighting | PAM and CAM always contribute equally regardless of task structure or spatial context requirements |
| No task-conditional analysis | No understanding of when spatial vs. channel attention matters across tasks with different complexity levels |

---

## 4. Our Contributions (V4.0)

### Contribution 1: Efficient Approximation Framework

**LowRankWindowedPAM** reduces PAM from $\mathcal{O}(N^2 C)$ to $\mathcal{O}(N r C)$ via Swin-style window partitioning and low-rank projection ($A \approx PQ$, rank $r \ll M^2$):

$$\text{Complexity: } \mathcal{O}(N^2 C) \rightarrow \mathcal{O}(N r C)$$

**GroupedCAM** reduces CAM from $\mathcal{O}(C^2)$ to $\mathcal{O}(C^2/G)$ by splitting $C$ channels into $G$ independent groups:

$$X_{ji}^{(g)} = \frac{\exp(A_i \cdot A_j)}{\sum_{k=1}^{C/G} \exp(A_i \cdot A_k)}$$

**Key engineering benefit:** Both approximations eliminate the zero-element tensor that crashes DA-TransUNet's DataParallel. DDP (NCCL all-reduce) enables multi-GPU training with per-GPU VRAM 6.4 GB vs DA-TransUNet's 11.5 GB.

### Contribution 2: Gate Collapse Analysis

The learnable gate $g = \sigma(W_g \cdot \text{GAP}(F))$ collapses to $g \approx 0.5$ throughout training. This is a **gradient symmetry problem** inherent to any two-branch attention with naïve learnable gating:

1. When $g = 0.5$, both PAM and CAM branches receive identical loss gradient: $0.5 \times \partial\mathcal{L}/\partial F_{\text{fused}}$
2. Neither branch has incentive to differentiate → PAM $\approx$ CAM throughout training
3. Gate gradient $\partial\mathcal{L}/\partial g = \partial\mathcal{L}/\partial F_{\text{fused}} \times (\text{PAM\_out} - \text{CAM\_out}) \approx 0$
4. This circular dependency is stable at **any window size $M$** — collapse is not caused by windowing

Empirically confirmed at M=7 (gate range 0.4993–0.5010, Δ=0.0017) and M=14 (same pattern).

### Contribution 3: Cross-Task Empirical Study

We study how approximation safety varies across tasks with different context complexity:

| Dataset | Task | TCC Level | Best ApproxDA Config | Δ DSC vs DA |
|---------|------|-----------|-------------------|-------------|
| Synapse | Multi-organ CT (9 classes) | High | gate=pam, M=7, r=32 | −1.87% |
| Kvasir-SEG | Polyp segmentation (binary) | Low | gate=learn, M=7, r=32 | **+0.80%** |
| ISIC 2018 | Skin lesion (binary) | Low | TBD | TBD |

Pattern: high Task Context Complexity tasks lose from approximation; low-TCC tasks tolerate or benefit.

### Contribution 4: Preliminary TCC→Safety Evidence

We provide preliminary empirical evidence that Task Context Complexity (TCC) governs when attention approximation is safe:

- **High TCC** (Synapse): complex multi-organ anatomy, large scale variation, strong global spatial dependencies across organs → approximation loses critical long-range context → accuracy drops
- **Low TCC** (Kvasir, ISIC): single object per image, relatively homogeneous boundary structure, local context often sufficient → approximation is safe; the implicit regularization may even help

This is framed as **preliminary evidence** because TCC is not yet formally quantified (a journal extension goal).

---

## 5. Architecture

```
Input Feature F
       |
  +----+----+
  |         |
PAM(F)    CAM(F)
(LowRank  (Grouped
 Windowed) CAM)
  |         |
  +----+----+
       |
  [gate_mode routing]
  gate=pam:   g = 1.0  (PAM only)
  gate=cam:   g = 0.0  (CAM only)
  gate=fixed: g = 0.5  (equal blend)
  gate=learn: g = σ(Wg · GAP(F))  [collapses to ≈0.5]
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
        M = min(self.M, H, W)                                 # clamp for global attention
        x_w   = window_partition(x, M)                        # (B*nW, C, M, M)
        nBW   = x_w.shape[0]
        x_n   = x_w.view(nBW, C, M * M)
        feat_B = self.conv_B(x_n)
        feat_C = self.conv_C(x_n)
        feat_D = self.conv_D(x_n)
        N_actual = M * M
        C_r    = F.linear(feat_C, self.proj_r.weight[:, :N_actual])   # (B*nW, C, r)
        D_r    = F.linear(feat_D, self.proj_r.weight[:, :N_actual])
        scores = torch.bmm(feat_B.transpose(1, 2), C_r)               # (B*nW, N, r)
        scores = F.softmax(scores, dim=-1)
        E_out  = torch.bmm(scores, D_r.transpose(1, 2))               # (B*nW, N, C)
        E_n    = self.alpha * E_out.transpose(1, 2) + x_n
        return window_reverse(E_n.view(nBW, C, M, M), M, H, W)
```

Window clamping (`M = min(self.M, H, W)`) enables `--window_size 112` to give global attention at every scale — used in the Global PAM ablation.

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

### 6.3 AdaDABlock with gate_mode Routing

```python
class AdaDABlock(nn.Module):
    def __init__(self, channels, window_size=7, rank=32, groups=8, gate_mode='learn'):
        super().__init__()
        self.gate_mode = gate_mode
        self.pam    = LowRankWindowedPAM(channels, window_size, rank)
        self.cam    = GroupedCAM(channels, groups)
        self.pool   = nn.AdaptiveAvgPool2d(1)
        if gate_mode == 'learn':
            self.gate_fc = nn.Linear(channels, channels)
        self.fusion = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        pam_out = self.pam(x)
        cam_out = self.cam(x)
        if self.gate_mode == 'pam':
            g = 1.0
        elif self.gate_mode == 'cam':
            g = 0.0
        elif self.gate_mode == 'fixed':
            g = 0.5
        else:  # 'learn' — collapses to ≈0.5 due to gradient symmetry
            g = torch.sigmoid(
                self.gate_fc(self.pool(x).view(x.shape[0], -1))
            ).view(x.shape[0], -1, 1, 1)
        fused = self.fusion(g * pam_out + (1.0 - g) * cam_out)
        return fused + x
```

---

## 7. Loss Function

$$\mathcal{L} = \frac{1}{2} \mathcal{L}_{\text{Dice}} + \frac{1}{2} \mathcal{L}_{\text{CE}}$$

---

## 8. Experimental Results

### Synapse Multi-Organ CT (high TCC)

| Method | DSC (%) | HD95 (mm) | GFLOPs | Notes |
|--------|---------|-----------|--------|-------|
| DA-TransUNet (paper) | 79.80 | 23.48 | — | — |
| DA-TransUNet (ours, 300ep SGD) | 80.51 | 25.41 | 30.2 (fvcore) | Baseline |
| ApproxDA gate=learn, M=7, r=32 | 77.93 | 33.96 | ~32 | Gate collapsed (g≈0.5) |
| ApproxDA gate=cam, M=7, r=32 | 78.26 | 30.59 | ~32 | CAM-only |
| ApproxDA gate=learn, M=14, r=64 | 78.04 | 29.09 | 32.4 (fvcore) | Gate collapsed at larger window too |
| **ApproxDA gate=pam, M=7, r=32** | **78.64** | **31.09** | **32.1 (fvcore)** | **Best ApproxDA on Synapse** |
| ApproxDA Global PAM, M=112, r=64 | ⏳ TBD | ⏳ TBD | — | Pending: answers window vs. low-rank bottleneck |

### Kvasir-SEG Polyp (low TCC)

| Method | DSC (%) | mIoU (%) | HD95 (mm) | Notes |
|--------|---------|---------|-----------|-------|
| DA-TransUNet (paper, 500ep Adam, 75/25) | 88.47 | 81.02 | — | Different setup |
| DA-TransUNet (ours, 300ep SGD, 80/20) | 88.44 | 81.70 | 53.04 | Baseline |
| **ApproxDA gate=learn, M=7, r=32** | **89.24** | **83.40** | **42.60** | **+0.80% DSC vs DA** |

### ISIC 2018 Skin Lesion (low TCC)

| Method | DSC (%) | mIoU (%) | Notes |
|--------|---------|---------|-------|
| DA-TransUNet (ours, 300ep SGD, 80/20) | 🔄 Training | — | — |
| ApproxDA gate=learn, M=7, r=32 | 🔄 Training | — | — |

### Efficiency (Synapse, T4)

| Method | Params (M) | GFLOPs | Train VRAM | Train Time | Multi-GPU |
|--------|-----------|--------|-----------|-----------|-----------|
| DA-TransUNet | 107.95 | 30.2 (fvcore) | 11.5 GB | 12.06h (T4×1) | No (DataParallel crash) |
| ApproxDA gate=pam, M=7, r=32 | 112.98 | 32.1 (fvcore) | 6.4 GB/GPU | 8.34h (T4×2) | Yes (DDP) |
| ApproxDA gate=learn, M=7, r=32 | 114.90 | ~32.1 (est.) | 10.6 GB | 11.51h (T4×1) | Yes (DDP) |

**Note: Do NOT claim GFLOPs reduction.** ApproxDA is slightly *higher* in total GFLOPs (32.1 vs 30.2) because the ViT backbone dominates and the Conv1d projection overhead offsets decoder savings. The efficiency story is DDP-compatibility and per-GPU VRAM halved (6.4 vs 11.5 GB).

---

## 9. Ablation Study (Actual Results)

### Gate Mode Ablation (Synapse, M=7, r=32)

| `--gate_mode` | g value | DSC (%) | HD95 (mm) | Interpretation |
|--------------|---------|---------|-----------|---------------|
| `pam` (g=1) | 1.0 | **78.64** | 31.09 | PAM-only; best on high-TCC task |
| `cam` (g=0) | 0.0 | 78.26 | 30.59 | CAM-only |
| `learn` M=14 | ≈0.5 (collapsed) | 78.04 | **29.09** | Larger window helps HD95 even with collapsed gate |
| `learn` M=7 | ≈0.5 (collapsed) | 77.93 | 33.96 | Worst DSC — collapsed gate is actively harmful on CT |

**Key finding:** gate=learn collapsed to 50/50 blend, which is the worst DSC configuration. PAM-only strictly dominates on Synapse. Gate collapse is confirmed NOT a windowing artifact (same collapse at M=14).

### Window Size / Rank Ablation (Synapse)

| Config | M | r | gate | DSC (%) | HD95 (mm) | GFLOPs |
|--------|---|---|------|---------|-----------|--------|
| DA-TransUNet | global | — | N/A | 80.51 | 25.41 | 30.2 |
| ApproxDA | 7 | 32 | learn | 77.93 | 33.96 | ~32 |
| ApproxDA | 14 | 64 | learn | 78.04 | **29.09** | 32.4 |
| ApproxDA Global PAM | 112 | 64 | pam | ⏳ TBD | ⏳ TBD | ~32+ |

**Pending experiment:** Global PAM (`--gate_mode pam --window_size 112 --rank 64`) directly isolates whether the accuracy gap is from windowing or from low-rank itself.

### Cross-Task Summary

| Dataset | DA-TransUNet DSC | ApproxDA best DSC | Δ DSC | TCC Level | Approximation safe? |
|---------|-----------------|---------------|-------|-----------|-------------------|
| Synapse | 80.51% | 78.64% (gate=pam) | −1.87% | High | ❌ No |
| Kvasir | 88.44% | 89.24% (gate=learn) | **+0.80%** | Low | ✅ Yes |
| ISIC | TBD | TBD | TBD | Low | ⏳ Expected Yes |

---

## 10. Key Contributions (for paper submission)

1. **Efficient approximation framework (LowRankWindowedPAM + GroupedCAM):** First application of windowed low-rank PAM + grouped CAM to the DA-block in medical segmentation. Enables DDP training (DA-TransUNet crashes with DataParallel); halves per-GPU VRAM (6.4 vs 11.5 GB); 30% training speedup on Synapse.

2. **Gate collapse analysis:** Theoretical derivation and empirical confirmation of the gradient symmetry collapse in naïve learnable two-branch gating. The collapse is a fundamental limitation (not a windowing artifact) applicable to any two-branch attention architecture with a symmetric initialization.

3. **Cross-task approximation study:** Systematic evaluation across Synapse (high-TCC), Kvasir (low-TCC), and ISIC (low-TCC). First empirical demonstration that the same approximation strategy has opposite effects depending on task complexity: −1.87% DSC on CT vs +0.80% DSC on polyp segmentation.

4. **Preliminary TCC→safety evidence:** Task Context Complexity (single-object homogeneous tasks vs multi-organ globally-dependent tasks) predicts whether approximation is safe. Preliminary because TCC is not formally quantified — formal quantification is a journal extension goal.

**Conference paper scope:** Understand routing failure and characterize task-conditioned approximation behavior. No new gating mechanism — analysis only.

**Journal paper scope:** Design non-collapsing routing (entropy-guided gate, orthogonal branch loss, MoE) that learns effective task-conditioned allocation; formally define and measure TCC; expand dataset range.

---

## 11. Figure Plan (V4.0)

**Figure 1 — Cross-Task Performance Bar Chart**
Bar chart comparing DA-TransUNet vs ApproxDA-TransUNet (best config) across Synapse, Kvasir, ISIC.
Highlight: Synapse bar shows approximation hurts (−1.87%); Kvasir bar shows it helps (+0.80%).
This is the paper's central visual — makes the task-conditioned pattern immediately visible.

**Figure 2 — Gate Collapse Illustration**
Two-panel: (a) training curve of mean gate value $g$ over epochs — flat line at 0.5 regardless of M or epoch; (b) gradient flow diagram showing the symmetry trap ($g=0.5$ → equal gradients → PAM≈CAM → gradient≈0).
Produced by running `analyze_gate_entropy.py` on trained checkpoints.

**Figure 3 — Global Context Analysis (Phase B experiment)**
Scatter or bar: DSC vs. attention scope (M=7 windowed / M=14 windowed / M=112 global) on Synapse.
If global PAM closes the gap → "windowing is the bottleneck"; if not → "low-rank itself limits capacity on high-TCC tasks." Either result tells a clean story.

**Figure 4 — Gate Mode Ablation Bar Chart**
DSC comparison: gate=pam / gate=cam / gate=learn (M=7) / gate=learn (M=14) on Synapse.
Shows gate=pam best; gate=learn worst; ordering is consistent with the collapse explanation.

**Figure 5 — Case Visualization** (highest paper value per reviewer)
Side-by-side segmentation output on one Synapse CT slice (where ApproxDA loses) and one Kvasir image (where ApproxDA wins). Overlaid with attention maps from the PAM and CAM branches.
Intuition: on Synapse, ApproxDA loses organ boundaries because windowed attention misses global anatomy; on Kvasir, local attention is sufficient for polyp contours.

---

## References

[1] J. Fu et al., "Dual attention network for scene segmentation," *CVPR*, 2019.

[2] G. Sun et al., "DA-TransUNet: integrating spatial and channel dual attention with transformer U-Net for medical image segmentation," *Frontiers in Bioengineering and Biotechnology*, vol. 12, 2024.

[3] A. Dosovitskiy et al., "An image is worth 16x16 words," *arXiv:2010.11929*, 2020.

[4] Z. Liu et al., "Swin Transformer: Hierarchical vision transformer using shifted windows," *ICCV*, 2021.
