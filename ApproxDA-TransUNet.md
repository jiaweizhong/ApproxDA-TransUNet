# Research Plan: ApproxDA-TransUNet

## 1. Naming

| Layer | Name |
|-------|------|
| **Paper Title** | ApproxDA-TransUNet: Understanding Context Sensitivity of Attention Approximation in Medical Image Segmentation |
| **Method Name** | ApproxDA-TransUNet (Approximate Dual Attention TransUNet) |
| **Research Theme** | Context Sensitivity (CS) as Governing Factor |
| **Design Principle** | Context Sensitivity (CS) → Window Selection Importance → Optimal Approximation Scale |
| **Code directories** | `experiments/ApproxDA-TransUNet/` |

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
| No context-dependent analysis | No understanding of when spatial vs. channel attention matters across tasks with different global context requirements |

---

## 4. Our Contributions (V4.0)

### Explicit Hypotheses (organizing framework)

| # | Hypothesis | Evidence |
|---|-----------|---------|
| **H1** | Attention approximation effectiveness depends on task characteristics — every task has an optimal approximation scale, and Context Sensitivity (CS) determines how important it is to find it | Synapse CS=2.30% (high sensitivity); Kvasir CS=0.64% (low sensitivity); both improved +1.14%/+1.73% at optimal M |
| **H2** | CS is operationally defined as DSC range across the window sensitivity ablation: CS = max(DSC) − min(DSC). High-CS tasks exhibit steep non-monotonic DSC-vs-M curves (bias-variance tradeoff: under-context → high bias; over-context → high variance; optimal M = sweet spot); low-CS tasks show a broad plateau. Mechanism: approximation changes **inductive bias** (locality prior alignment), causes **over-context suppression** (reduces spurious long-range correlations), and may act as **implicit regularization** | CS(Synapse)=2.30pp; CS(Kvasir)=0.64pp; Phase E: M=7 pam 62.3% on-mask vs M=112 34.2% (1.82× ratio, 9/10 samples) |
| **H3** | Symmetric dual-attention routing naturally collapses to a fixed 50/50 blend — a stable equilibrium of dual-branch gradient symmetry | Confirmed at M=7 and M=14; collapse independent of window size |

Everything in the paper maps to H1, H2, or H3.

### Contribution 1: Controllable Approximation Framework (ApproxDA as Scientific Instrument)

ApproxDA-TransUNet is designed **not** as a performance-optimized architecture, but as a controllable experimental instrument with **three independently adjustable approximation axes**:

| Axis | Operator | Hyperparameter | What it approximates |
|------|----------|---------------|-------------------|
| Spatial | LowRankWindowedPAM | window size $M$ | Spatial approximation (receptive field) |
| Representation | Low-rank projection | rank $r$ | Representation approximation (attention fidelity) |
| Channel | GroupedCAM | groups $G$ | Channel approximation (cross-channel interaction) |

$$\text{PAM complexity: } \mathcal{O}(N^2 C) \rightarrow \mathcal{O}(N r C) \quad \text{CAM complexity: } \mathcal{O}(C^2) \rightarrow \mathcal{O}(C^2/G)$$

**Engineering benefit:** Eliminates the zero-element tensor that crashes DA-TransUNet's DataParallel; enables DDP with per-GPU VRAM 6.4 GB vs DA-TransUNet's 11.5 GB.

### Contribution 2: Gate Collapse Analysis (H3 evidence)

The learnable gate collapses to $g \approx 0.5$ — a **stable equilibrium of symmetric dual-branch optimization**:

1. At $g = 0.5$, both branches receive identical gradient: $0.5 \times \partial\mathcal{L}/\partial F_{\text{fused}}$
2. Neither branch has incentive to differentiate → PAM $\approx$ CAM throughout training
3. Gate gradient $\partial\mathcal{L}/\partial g \approx 0$ because (PAM\_out − CAM\_out) ≈ 0
4. Circular dependency is stable at **any window size** — this is a gradient symmetry problem, not a windowing artifact

The collapsed routing's effectiveness is **task-dependent**: harmful on high-CS Synapse, beneficial on low-CS Kvasir. The finding is not "learnable gates are bad" — it is that symmetric initialization creates an unstable fixed point that any dual-branch architecture will converge to without explicit symmetry breaking.

Confirmed at M=7 (gate range 0.4993–0.5010, Δ=0.0017) and M=14 (same pattern).

### Contribution 3: Cross-Task Empirical Study (H1 + H2 evidence)

| Dataset | Task | CS Level | Best ApproxDA Config | Δ DSC vs DA |
|---------|------|-----------|-------------------|-------------|
| Synapse | Multi-organ CT (9 classes) | High CS (2.30pp) | gate=pam, M=28, r=32 | **+1.14%** |
| Kvasir-SEG | Polyp segmentation (binary) | Low CS (0.64pp) | gate=pam, M=56, r=32 | **+1.73%** |
| ISIC 2018 | Skin lesion (binary) | Low CS (TBD — window ablation pending) | gate=learn, M=7, r=32 | **+1.15%** (vs our re-run 88.43%; +0.70% vs paper 88.88%) |

The opposite directions (−0.36% vs +1.73%) confirm approximation safety is task-dependent (H1). CS appears to be an important explanatory factor (H2): high-CS tasks require global spatial interaction that approximation disrupts. For low-CS tasks, the improvement should **not** be interpreted as approximation creating additional information. Rather, windowed PAM imposes an **inductive bias** (locality prior) that better matches the intrinsic structure of local-boundary tasks — simultaneously reducing overfitting to spurious long-range correlations. This parallels the effectiveness of local-receptive-field CNNs on texture-dominant tasks: enforcing locality is a correct prior, not just tolerance of approximation.

### Contribution 4: CS Operationalized + Mechanisms for Why Approximation Improves

**Context Sensitivity (CS) is operationally defined**:
$$\text{CS} \triangleq \max_M \text{DSC}(M) - \min_M \text{DSC}(M)$$
measured under fixed gate=pam, r=32. Values: CS(Synapse) = 2.30 pp; CS(Kvasir) = 0.64 pp; ratio = 3.6×.

**Key insight:** High CS does NOT mean approximation is harmful. It means performance is *sensitive* to window choice. With the right window (M=28), ApproxDA *beats* DA-TransUNet on Synapse (+1.14%). Every task has an optimal attention context scale; CS determines how critical it is to find it.

**The non-monotonic Synapse curve is the central result.** The DSC-vs-M curve (M=7: 78.64% → M=28: 80.94% → M=112: 79.44%) is a direct analogue of the **bias-variance tradeoff**:
- Under-context (M=7): high bias — missing inter-organ dependencies
- Optimal (M=28): sweet spot — sufficient cross-organ context, suppressed background noise
- Over-context (M=112): high variance — spurious long-range correlations from unrelated regions

**Three mechanisms explaining why approximation improves (in priority order):**
1. **Inductive bias alignment** (strongest, direct Phase E evidence): windowed PAM enforces locality prior matching low-CS task structure. Evidence: M=7 pam 62.3% on-mask vs M=112 34.2% (1.82×, 9/10 samples). Analogous to CNN locality being a feature, not a bug.
2. **Over-context suppression** (Synapse M=28 > M=112 as direct evidence): fully-global attention introduces spurious variance. Intermediate window suppresses this.
3. **Implicit regularization** (hypothesis, no direct evidence yet): constrained receptive field reduces attention degrees of freedom → regularization effect. Requires train/test DSC gap analysis (journal goal).

**Conference-scope empirical validation (Phase E):** M=7 pam vs M=112 global, 10 Kvasir images. Result: 62.3% vs 34.2% on-mask concentration, 1.82× ratio, 9/10 samples confirm. Directly supports mechanism 1.

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

### 6.3 ApproxDABlock with gate_mode Routing

```python
class ApproxDABlock(nn.Module):
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

### Synapse Multi-Organ CT (high CS)

| Method | DSC (%) | HD95 (mm) | GFLOPs | Notes |
|--------|---------|-----------|--------|-------|
| DA-TransUNet (paper) | 79.80 | 23.48 | — | — |
| DA-TransUNet (paper reported) | 79.80 | 23.48 | 30.2 (fvcore) | Baseline |
| ApproxDA gate=learn, M=7, r=32 | 77.78 | 34.29 | 32.1 (fvcore) | Gate collapsed (g≈0.5) |
| ApproxDA gate=cam, M=7, r=32 | 78.26 | 30.59 | ~32 | CAM-only |
| ApproxDA gate=learn, M=14, r=64 | 78.04 | 29.09 | 32.4 (fvcore) | Gate collapsed at larger window too |
| ApproxDA gate=pam, M=7, r=32 | 78.64 | 31.09 | 32.1 (fvcore) | Gate ablation baseline |
| ApproxDA Global PAM, M=112, r=64 | 78.93 | 31.21 | 32.4 (fvcore) | SUPERSEDED — rank mismatch with Phase D |
| ApproxDA gate=pam, M=56, r=32 | 78.90 | 35.23 | 32.1 (fvcore) | Phase D D2 |
| ApproxDA Global PAM, M=112, r=32 | 79.44 | 27.26 | 32.1 (fvcore) | Phase D high-M anchor |
| **ApproxDA Phase D peak, M=28, r=32** | **80.94** | **27.49** | 32.1 (fvcore) | **New best AdaDA on Synapse** — non-monotonic peak; **+1.14% vs DA-TransUNet** |

### Kvasir-SEG Polyp (low CS)

| Method | DSC (%) | mIoU (%) | HD95 (mm) | Notes |
|--------|---------|---------|-----------|-------|
| DA-TransUNet (paper, 500ep Adam, 75/25) | 88.47 | 81.02 | — | Different setup |
| DA-TransUNet (ours, 300ep SGD, 80/20) | 88.44 | 81.70 | 53.04 | Baseline |
| ApproxDA gate=learn, M=7, r=32 | 89.24 | 83.40 | 42.60 | +0.80% DSC vs DA (gate collapse baseline) |
| ApproxDA gate=pam, M=7, r=32 | 89.53 | 83.52 | 43.87 | Phase D D3 |
| ApproxDA gate=pam, M=28, r=32 | 89.54 | 83.66 | 45.39 | Phase D D4 |
| **ApproxDA gate=pam, M=56, r=32** | **90.17** | **84.27** | **44.35** | **Phase D D5 — new best (+1.73% DSC vs DA)** |
| ApproxDA gate=pam, M=112, r=32 | 89.56 | 83.68 | 45.51 | Phase D D6 |

### ISIC 2018 Skin Lesion (low CS)

| Method | DSC (%) | mIoU (%) | Notes |
|--------|---------|---------|-------|
| DA-TransUNet (ours, 300ep SGD, 80/20) | 88.43 | 80.68 | Train: 83.00h, T4×1, best val DSC 0.8771 ep75, VRAM 11.5 GB, Params 107.95M, GFLOPs 30.2 (fvcore); test HD95 159.72mm, Infer VRAM 0.5GB, avg 3089ms/case |
| **ApproxDA gate=learn, M=7, r=32** | **89.58** | **82.66** | Train: 83.26h, T4×1, best val DSC **0.8956** ep90, VRAM 10.5 GB; test: Params 114.90M, GFLOPs 32.0 (fvcore), HD95 134.83mm, Infer VRAM 0.9GB, avg 3011ms/case. **Δ vs ours: +1.15% DSC, −24.89mm HD95, +1.98% mIoU** |

### Efficiency (Synapse, T4)

| Method | Params (M) | GFLOPs | Train VRAM | Train Time | Multi-GPU |
|--------|-----------|--------|-----------|-----------|-----------|
| DA-TransUNet | 107.95 | 30.2 (fvcore) | 11.5 GB | 12.06h (T4×1) | No (DataParallel crash) |
| ApproxDA gate=pam, M=7, r=32 | 112.98 | 32.1 (fvcore) | 6.4 GB/GPU | 8.34h (T4×2) | Yes (DDP) |
| ApproxDA gate=learn, M=7, r=32 | 114.90 | 32.1 (fvcore) | 10.6 GB | 11.51h (T4×1) | Yes (DDP) |

**Note: Do NOT claim GFLOPs reduction.** ApproxDA is slightly *higher* in total GFLOPs (32.1 vs 30.2) because the ViT backbone dominates and the Conv1d projection overhead offsets decoder savings. The efficiency story is DDP-compatibility and per-GPU VRAM halved (6.4 vs 11.5 GB).

---

## 9. Ablation Study (Actual Results)

### Gate Mode Ablation (Synapse, M=7, r=32)

| `--gate_mode` | g value | DSC (%) | HD95 (mm) | Interpretation |
|--------------|---------|---------|-----------|---------------|
| `pam` (g=1) | 1.0 | **78.64** | 31.09 | PAM-only; best on high-CS task |
| `cam` (g=0) | 0.0 | 78.26 | 30.59 | CAM-only |
| `learn` M=14 | ≈0.5 (collapsed) | 78.04 | **29.09** | Larger window helps HD95 even with collapsed gate |
| `learn` M=7 | ≈0.5 (collapsed) | 77.78 | 34.29 | Worst DSC — collapsed gate is actively harmful on CT |

**Key finding (H3):** gate=learn collapses to a stable 50/50 fixed routing — not because the gate "fails" but because symmetric initialization creates a gradient symmetry equilibrium. The collapsed routing is task-dependent in effectiveness: worst DSC on high-CS Synapse (below both single-branch modes), but best on low-CS Kvasir. Collapse is independent of window size (same at M=7 and M=14).

### Window Size / Rank Ablation (Synapse)

**Synapse (high CS) — M sweep, gate=pam, r=32:**

| Config | M | r | gate | DSC (%) | HD95 (mm) | GFLOPs |
|--------|---|---|------|---------|-----------|--------|
| DA-TransUNet (paper) | global | — | N/A | 79.80 | 23.48 | 30.2 |
| ApproxDA | 7 | 32 | pam | 78.64 | 31.09 | 32.1 |
| ApproxDA | 14 | 64 | learn | 78.04 | **29.09** | 32.4 |
| ApproxDA Global PAM | 112 | 64 | pam | 78.93 | 31.21 | 32.4 | ← SUPERSEDED |
| ApproxDA | 56 | 32 | pam | 78.90 | 35.23 | 32.1 | Phase D D2 |
| ApproxDA Global PAM | 112 | 32 | pam | 79.44 | 27.26 | 32.1 | Phase D high-M anchor |
| **ApproxDA Phase D peak** | **28** | **32** | **pam** | **80.94** | **27.49** | 32.1 | **BEST — +1.14% vs DA-TransUNet** |

**Synapse Finding:** DSC-vs-M curve is **non-monotonic** — peak at M=28 (80.94%, +1.14% vs DA-TransUNet). Both M=7 (78.64%) and M=112 (79.44%) underperform the intermediate optimum. DSC range = **2.30%** (high sensitivity, high CS). Window sensitivity slope M=7→M=112: +0.80%; but peak-to-min range 2.30% better captures the sensitivity. ApproxDA with optimal M beats DA-TransUNet on Synapse!

**Kvasir (low CS) — Phase D window sensitivity, gate=pam, r=32:**

| M | DSC (%) | HD95 (mm) | IoU (%) |
|---|---------|-----------|---------|
| 7 (D3) | 89.53 | 43.87 | 83.52 |
| 28 (D4) | 89.54 | 45.39 | 83.66 |
| **56 (D5)** | **90.17** | 44.35 | 84.27 |
| 112 (D6) | 89.56 | 45.51 | 83.68 |

**Kvasir Finding:** DSC-vs-M peaks at M=56 (non-monotonic). Window sensitivity M=7→M=112: **+0.03%** (vs +0.80% on Synapse). Near-flat curve confirms low CS.

### Cross-Task Summary

| Dataset | DA-TransUNet DSC | ApproxDA best DSC | Δ DSC | Window Sensitivity | CS Level |
|---------|-----------------|---------------|-------|-------------------|-----------|
| Synapse | 79.80% (paper reported) | **80.94%** (gate=pam, M=28, r=32) | **+1.14%** | 2.30% range (high) | High CS |
| Kvasir | 88.44% | **90.17%** (gate=pam, M=56, r=32) | **+1.73%** | 0.64% range (low) | Low CS |
| ISIC | 88.43% (ours, 300ep SGD) | **89.58%** (gate=learn, M=7, r=32) | **+1.15%** (vs ours) | TBD (window ablation pending) | Low CS |

---

## 10. Key Contributions (for paper submission)

> **Overarching shift:** This paper shifts the discussion from *how* to approximate attention toward *when* attention approximation should be applied.

1. **Controllable approximation framework (ApproxDA as scientific instrument):** Three independently adjustable axes — spatial scope (M), representation fidelity (r), channel interaction (G) — enable isolated analysis of each approximation operator. Engineering benefit: DDP-compatible, halves per-GPU VRAM (6.4 vs 11.5 GB). The framework is the telescope; CS hypothesis is the science.

2. **Gate collapse analysis (H3):** Theoretical derivation and empirical confirmation that symmetric two-branch gating collapses to a stable equilibrium — not a training failure, but a fundamental property of symmetric gradient flow. The collapse's effectiveness is task-dependent: the same fixed routing is harmful on high-CS tasks and beneficial on low-CS tasks. Applicable to any two-branch attention with symmetric initialization.

3. **Cross-task approximation study (H1 + H2):** Both tasks improved with optimal window: **+1.14% DSC on high-CS Synapse** (gate=pam M=28/r=32) and **+1.73% DSC on low-CS Kvasir** (gate=pam M=56/r=32). CS governs window *sensitivity*: Synapse range 2.30% vs Kvasir 0.64% (3.6× ratio). High-CS tasks require careful window tuning; low-CS tasks are window-robust. The low-CS improvement is explained by **inductive bias alignment**: windowed PAM imposes a locality prior matching local-boundary task structure.

4. **CS as governing factor + inductive bias hypothesis:** CS predicts **window sensitivity** (not direction). Synapse: non-monotonic DSC-vs-M (peak at M=28, 80.94%), range 2.30%; Kvasir: near-flat (peak at M=56, 90.17%), range 0.64%. Phase E attention maps (gate=pam M=7 vs M=112, 10 Kvasir images): M=7 pam attends **62.3% on-mask** vs M=112 global **34.2%** — 1.82× more concentrated on polyp boundary (9/10 samples confirm). Direct visual evidence for inductive bias alignment.

**Conference scope:** Characterize context-dependent approximation behavior + inductive bias hypothesis validation (Phase E: attention maps; Phase C: context-radius sensitivity). No new gating mechanism — scientific analysis.

**Journal scope:** Design non-collapsing routing; formally quantify CS; expand dataset range across the CS spectrum; validate inductive bias mechanism systematically (attention-distance analysis across M sweep; see `EXPERIMENT_PLAN.md` § *Phase D* and § *Phase E* for targeted experiments).

---

## 11. Figure Plan (V4.0)

**Figure 1 — Cross-Task Performance Bar Chart**
Bar chart comparing DA-TransUNet vs ApproxDA-TransUNet (best config) across Synapse, Kvasir, ISIC.
Highlight: Both bars show ApproxDA > DA-TransUNet — Synapse +1.14% (M=28), Kvasir +1.73% (M=56). Can add window sensitivity error bars or range annotation.
This is the paper's central visual — makes the context-dependent pattern immediately visible.

**Figure 2 — Gate Collapse Illustration**
Two-panel: (a) training curve of mean gate value $g$ over epochs — flat line at 0.5 regardless of M or epoch; (b) gradient flow diagram showing the symmetry trap ($g=0.5$ → equal gradients → PAM≈CAM → gradient≈0).
Produced by running `analyze_gate_entropy.py` on trained checkpoints.

**Figure 3 — Global Context Analysis (Phase B experiment)**
Scatter or bar: DSC vs. attention scope (M=7 windowed / M=14 windowed / M=112 global) on Synapse.
If global PAM closes the gap → "windowing is the bottleneck"; if not → "low-rank itself limits capacity on high-CS tasks." Either result tells a clean story.

**Figure 4 — Gate Mode Ablation Bar Chart**
DSC comparison: gate=pam / gate=cam / gate=learn (M=7) / gate=learn (M=14) on Synapse.
Shows gate=pam best; gate=learn worst; ordering is consistent with the collapse explanation.

**Figure 5 — Attention Map Visualization (Phase E, inductive bias evidence) ✅ Done**
10-image comparison: gate=pam M=7 (windowed) vs gate=pam M=112 (global) on Kvasir polyp images.
Result: M=7 pam attends **62.3% on-mask** vs M=112 global **34.2%** — 1.82× more concentrated on polyp boundary (9/10 samples). Gate=learn baseline: 54.2%.
Output: `results/AdaDA-TransUNet/attention_maps/attn_Kvasir_M7_r32_pam_vs_M112.png`
Generated by `analyze_attention_maps.py` — no training needed. Controlled comparison: same architecture, same r=32, same gate=pam — only M differs.

**Figure 6 — Case Visualization** (segmentation quality)
Side-by-side segmentation output on one Synapse CT slice (where ApproxDA loses) and one Kvasir image (where ApproxDA wins).
Intuition: on Synapse, ApproxDA loses organ boundaries because windowed attention misses global anatomy; on Kvasir, local attention is not only sufficient but imposes the correct prior.

**Figure 7 — Window Sensitivity (Phase D, CS proxy) ✅ Done (Kvasir); Pending (Synapse M=28, M=56)**
DSC vs window size M for Synapse (4-point: M=7/28/56/112) and Kvasir (4-point: M=7/28/56/112).
Synapse slope: +0.80% (M=7→112, high CS). Kvasir slope: +0.03% (near-flat, low CS).
Replaces Phase C (center-crop sensitivity) which was INVALID — full-label evaluation artifact on both datasets.
Note: Synapse M=28 and M=56 are still pending (D1 and D2); Kvasir 4-point curve is complete.

---

## References

[1] J. Fu et al., "Dual attention network for scene segmentation," *CVPR*, 2019.

[2] G. Sun et al., "DA-TransUNet: integrating spatial and channel dual attention with transformer U-Net for medical image segmentation," *Frontiers in Bioengineering and Biotechnology*, vol. 12, 2024.

[3] A. Dosovitskiy et al., "An image is worth 16x16 words," *arXiv:2010.11929*, 2020.

[4] Z. Liu et al., "Swin Transformer: Hierarchical vision transformer using shifted windows," *ICCV*, 2021.
