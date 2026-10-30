import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
import numpy as np
from os.path import join as pjoin
from collections import OrderedDict


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


class StdConv2d(nn.Conv2d):

    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-5)
        return F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )


def conv3x3(cin, cout, stride=1, groups=1, bias=False):
    return StdConv2d(
        cin, cout, kernel_size=3, stride=stride, padding=1, bias=bias, groups=groups
    )


def conv1x1(cin, cout, stride=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=1, stride=stride, padding=0, bias=bias)


class PreActBottleneck(nn.Module):
    """Pre-activation (v2) bottleneck block."""

    def __init__(self, cin, cout=None, cmid=None, stride=1):
        super().__init__()
        cout = cout or cin
        cmid = cmid or cout // 4

        self.gn1 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv1 = conv1x1(cin, cmid, bias=False)
        self.gn2 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv2 = conv3x3(
            cmid, cmid, stride, bias=False
        )  # Original code has it on conv1!!
        self.gn3 = nn.GroupNorm(32, cout, eps=1e-6)
        self.conv3 = conv1x1(cmid, cout, bias=False)
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or cin != cout:
            # Projection also with pre-activation according to paper.
            self.downsample = conv1x1(cin, cout, stride, bias=False)
            self.gn_proj = nn.GroupNorm(cout, cout)

    def forward(self, x):

        # Residual branch
        residual = x
        if hasattr(self, "downsample"):
            residual = self.downsample(x)
            residual = self.gn_proj(residual)

        # Unit's branch
        y = self.relu(self.gn1(self.conv1(x)))
        y = self.relu(self.gn2(self.conv2(y)))
        y = self.gn3(self.conv3(y))

        y = self.relu(residual + y)
        return y

    def load_from(self, weights, n_block, n_unit):
        conv1_weight = np2th(weights[pjoin(n_block, n_unit, "conv1/kernel")], conv=True)
        conv2_weight = np2th(weights[pjoin(n_block, n_unit, "conv2/kernel")], conv=True)
        conv3_weight = np2th(weights[pjoin(n_block, n_unit, "conv3/kernel")], conv=True)

        gn1_weight = np2th(weights[pjoin(n_block, n_unit, "gn1/scale")])
        gn1_bias = np2th(weights[pjoin(n_block, n_unit, "gn1/bias")])

        gn2_weight = np2th(weights[pjoin(n_block, n_unit, "gn2/scale")])
        gn2_bias = np2th(weights[pjoin(n_block, n_unit, "gn2/bias")])

        gn3_weight = np2th(weights[pjoin(n_block, n_unit, "gn3/scale")])
        gn3_bias = np2th(weights[pjoin(n_block, n_unit, "gn3/bias")])

        self.conv1.weight.copy_(conv1_weight)
        self.conv2.weight.copy_(conv2_weight)
        self.conv3.weight.copy_(conv3_weight)

        self.gn1.weight.copy_(gn1_weight.view(-1))
        self.gn1.bias.copy_(gn1_bias.view(-1))

        self.gn2.weight.copy_(gn2_weight.view(-1))
        self.gn2.bias.copy_(gn2_bias.view(-1))

        self.gn3.weight.copy_(gn3_weight.view(-1))
        self.gn3.bias.copy_(gn3_bias.view(-1))

        if hasattr(self, "downsample"):
            proj_conv_weight = np2th(
                weights[pjoin(n_block, n_unit, "conv_proj/kernel")], conv=True
            )
            proj_gn_weight = np2th(weights[pjoin(n_block, n_unit, "gn_proj/scale")])
            proj_gn_bias = np2th(weights[pjoin(n_block, n_unit, "gn_proj/bias")])

            self.downsample.weight.copy_(proj_conv_weight)
            self.gn_proj.weight.copy_(proj_gn_weight.view(-1))
            self.gn_proj.bias.copy_(proj_gn_bias.view(-1))


class ResNetV2(nn.Module):
    """Implementation of Pre-activation (v2) ResNet mode."""

    def __init__(self, block_units, width_factor):
        super().__init__()
        width = int(64 * width_factor)
        self.width = width

        self.root = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv",
                        StdConv2d(
                            3, width, kernel_size=7, stride=2, bias=False, padding=3
                        ),
                    ),
                    ("gn", nn.GroupNorm(32, width, eps=1e-6)),
                    ("relu", nn.ReLU(inplace=True)),
                    # ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=0))
                ]
            )
        )

        self.body = nn.Sequential(
            OrderedDict(
                [
                    (
                        "block1",
                        nn.Sequential(
                            OrderedDict(
                                [
                                    (
                                        "unit1",
                                        PreActBottleneck(
                                            cin=width, cout=width * 4, cmid=width
                                        ),
                                    )
                                ]
                                + [
                                    (
                                        f"unit{i:d}",
                                        PreActBottleneck(
                                            cin=width * 4, cout=width * 4, cmid=width
                                        ),
                                    )
                                    for i in range(2, block_units[0] + 1)
                                ],
                            )
                        ),
                    ),
                    (
                        "block2",
                        nn.Sequential(
                            OrderedDict(
                                [
                                    (
                                        "unit1",
                                        PreActBottleneck(
                                            cin=width * 4,
                                            cout=width * 8,
                                            cmid=width * 2,
                                            stride=2,
                                        ),
                                    )
                                ]
                                + [
                                    (
                                        f"unit{i:d}",
                                        PreActBottleneck(
                                            cin=width * 8,
                                            cout=width * 8,
                                            cmid=width * 2,
                                        ),
                                    )
                                    for i in range(2, block_units[1] + 1)
                                ],
                            )
                        ),
                    ),
                    (
                        "block3",
                        nn.Sequential(
                            OrderedDict(
                                [
                                    (
                                        "unit1",
                                        PreActBottleneck(
                                            cin=width * 8,
                                            cout=width * 16,
                                            cmid=width * 4,
                                            stride=2,
                                        ),
                                    )
                                ]
                                + [
                                    (
                                        f"unit{i:d}",
                                        PreActBottleneck(
                                            cin=width * 16,
                                            cout=width * 16,
                                            cmid=width * 4,
                                        ),
                                    )
                                    for i in range(2, block_units[2] + 1)
                                ],
                            )
                        ),
                    ),
                ]
            )
        )

    def forward(self, x):
        features = []
        b, c, in_size, _ = x.size()
        x = self.root(x)
        features.append(x)
        x = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)(x)
        for i in range(len(self.body) - 1):
            x = self.body[i](x)
            right_size = int(in_size / 4 / (i + 1))
            if x.size()[2] != right_size:
                pad = right_size - x.size()[2]
                assert pad < 3 and pad > 0, "x {} should {}".format(
                    x.size(), right_size
                )
                feat = torch.zeros(
                    (b, x.size()[1], right_size, right_size), device=x.device
                )
                feat[:, :, 0 : x.size()[2], 0 : x.size()[3]] = x[:]
            else:
                feat = x
            features.append(feat)
        x = self.body[-1](x)
        return x, features[::-1]


# ---------------------------------------------------------------------------
# ApproxDA-TransUNet: windowed low-rank PAM, grouped CAM, adaptive DA block
# ---------------------------------------------------------------------------


def window_partition(x, window_size):
    """(B, C, H, W) -> (B*nW, C, M, M)"""
    B, C, H, W = x.shape
    M = window_size
    x = x.contiguous().view(B, C, H // M, M, W // M, M)
    return x.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, C, M, M)


def window_reverse(windows, window_size, H, W):
    """(B*nW, C, M, M) -> (B, C, H, W)"""
    M = window_size
    nW = (H // M) * (W // M)
    B = windows.shape[0] // nW
    C = windows.shape[1]
    x = windows.contiguous().view(B, H // M, W // M, C, M, M)
    return x.permute(0, 3, 1, 4, 2, 5).contiguous().view(B, C, H, W)


def exact_window_attention(q, k, v, bias=None):
    """softmax(q^T k) v^T without scaling, as PAM_Module: q, k (B, d, N), v (B, C, N) -> (B, N, C).

    On CUDA this runs the memory-efficient SDPA kernel, which never stores the N x N
    scores (15 GB at M=112, bs 24). The kernel needs head dims divisible by 4, so q/k/v
    are zero-padded (q^T k is unchanged; padded value channels are sliced off), and it is
    forced so an unsupported case raises instead of falling back to the N x N math path.
    On CPU (fvcore FLOP audits) the explicit matmuls are kept.
    """
    if not q.is_cuda:
        logits = torch.bmm(q.transpose(1, 2), k)
        scores = F.softmax(logits if bias is None else logits + bias, dim=-1)
        return torch.bmm(scores, v.transpose(1, 2))
    C = v.shape[1]
    pad_qk, pad_v = -q.shape[1] % 4, -C % 4
    q = F.pad(q.transpose(1, 2), (0, pad_qk)).contiguous().unsqueeze(1)  # (B, 1, N, d')
    k = F.pad(k.transpose(1, 2), (0, pad_qk)).contiguous().unsqueeze(1)
    v = F.pad(v.transpose(1, 2), (0, pad_v)).contiguous().unsqueeze(1)  # (B, 1, N, C')
    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        mask = None if bias is None else bias.to(q.dtype).expand(q.shape[0], 1, q.shape[2], k.shape[2])
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=1.0)
    return out.squeeze(1)[..., :C]


class LowRankWindowedPAM(nn.Module):
    """
    Windowed PAM: O(N^2 C) -> O(N*r*C) via Swin-style windows + low-rank projection.
    Keys and values are projected from window_size^2 -> rank via self.proj_r.
    """

    def __init__(self, channels, window_size=7, rank=32, qk_channels=None, alpha_init=0.0, attn_norm=False,
                 identity=True, or_fusion=None, rel_pos=False, abs_pos=False, affinity_in=0, affinity_dim=32):
        super().__init__()
        self.M = window_size
        # or_fusion (journal, opt-in via --pam_fusion max|relu_add|lse): an OR-like fusion with the identity as a floor
        # instead of alpha * E + x:  max(x, E)  /  x + relu(E)  /  log(exp(x) + exp(E)).  alpha is unused; the
        # attention weights get gradients from the first step (no alpha = 0 dead-lock).
        self.or_fusion = or_fusion
        N = window_size**2
        # qk_channels=None keeps the legacy full-width queries/keys; the C/16 block
        # passes channels // 8 to match DA-TransUNet's PAM_Module.
        qk = channels if qk_channels is None else qk_channels
        # abs_pos (opt-in, --pam_abspos): the normalised global (y, x) coordinates are appended to the query/key input
        self.abs_pos = abs_pos
        # affinity_in > 0 (opt-in, --pam_affinity): Context Prior style attention (Yu et al., CVPR 2020). Queries and
        # keys (affinity_dim channels) come from an external feature with affinity_in channels (the block input, not
        # the C/16 compression, whose 1-channel queries give rank-1 logits); the attention is the sigmoid affinity
        # P = sigmoid(Q^T K / sqrt(d)) over all positions of the window (exact, no low-rank projection), the values are
        # aggregated with the row-normalised P, and the logits are kept for affinity_loss() during training.
        self.affinity_in = affinity_in
        self.affinity_logits = None
        if affinity_in > 0 and (rel_pos or abs_pos):
            raise ValueError("--pam_affinity cannot be combined with --pam_relpos / --pam_abspos")
        qk_src = affinity_in if affinity_in > 0 else channels + 2 * abs_pos
        qk = affinity_dim if affinity_in > 0 else qk
        self.conv_B = nn.Conv1d(qk_src, qk, 1)
        self.conv_C = nn.Conv1d(qk_src, qk, 1)
        # rel_pos (opt-in, --pam_relpos): Swin-style relative position bias b(dy, dx) added to the logits, one learnable
        # value per offset inside a window; with the low-rank path the biased logits are projected like the keys
        self.rel_bias = nn.Parameter(torch.zeros((2 * window_size - 1) ** 2)) if rel_pos else None
        self.conv_D = nn.Conv1d(channels, channels, 1)
        # rank <= 0: no low-rank projection, i.e. exact attention inside each window
        self.proj_r = nn.Linear(N, rank, bias=False) if rank > 0 and affinity_in == 0 else None
        # alpha = 0 (default, as DANet) blocks every gradient to the attention weights until alpha moves;
        # alpha_init > 0 and/or attn_norm (BatchNorm on the attention term, ReZero style) are opt-in fixes.
        self.alpha = nn.Parameter(torch.full((1,), float(alpha_init)))
        self.attn_norm = norm(channels) if attn_norm else None
        # identity=False drops the "+ x" path: the attention term is the only output (opt-in experiment)
        self.identity = identity
        self.ablate = False  # True: return the input unchanged (attention switched off)

    def _rel_bias(self, M):
        """(M^2, M^2) bias for a window of side M <= self.M, gathered from the (2*self.M-1)^2 table."""
        yy, xx = torch.meshgrid(torch.arange(M, device=self.rel_bias.device),
                                torch.arange(M, device=self.rel_bias.device), indexing="ij")
        yy, xx = yy.flatten(), xx.flatten()
        dy = yy[:, None] - yy[None, :] + self.M - 1
        dx = xx[:, None] - xx[None, :] + self.M - 1
        return self.rel_bias[dy * (2 * self.M - 1) + dx]

    def forward(self, x, qk_feat=None):
        if self.ablate:
            return x
        B, C, H, W = x.shape
        # Clamp window to feature-map size so window_size=112 gives global attention
        # at every scale without requiring H to be a multiple of self.M.
        M = min(self.M, H, W)
        x_w = window_partition(x, M)  # (B*nW, C, M, M)
        nBW = x_w.shape[0]
        x_n = x_w.view(nBW, C, M * M)  # (B*nW, C, N_actual)
        if self.affinity_in:
            qk_n = window_partition(qk_feat, M).view(nBW, -1, M * M)
            q, k = self.conv_B(qk_n), self.conv_C(qk_n)
            logits = torch.bmm(q.transpose(1, 2), k) / q.shape[1] ** 0.5  # (B*nW, N, N)
            self.affinity_logits = logits if self.training else None
            P = torch.sigmoid(logits)
            E_out = torch.bmm(P / P.sum(-1, keepdim=True).clamp_min(1e-6), self.conv_D(x_n).transpose(1, 2))
        else:
            if self.abs_pos:
                yy, xx = torch.meshgrid(torch.linspace(-1, 1, H, device=x.device, dtype=x.dtype),
                                        torch.linspace(-1, 1, W, device=x.device, dtype=x.dtype), indexing="ij")
                pos = window_partition(torch.stack([yy, xx])[None].expand(B, -1, -1, -1), M).view(nBW, 2, M * M)
                qk_in = torch.cat([x_n, pos], dim=1)
            else:
                qk_in = x_n
            feat_B = self.conv_B(qk_in)
            feat_C = self.conv_C(qk_in)
            feat_D = self.conv_D(x_n)
            # proj_r was registered with N=self.M**2; slice weights when clamped
            N_actual = M * M
            bias = self._rel_bias(M) if self.rel_bias is not None else None  # (N, N) or None
            if self.proj_r is None:
                # exact window attention
                E_out = exact_window_attention(feat_B, feat_C, feat_D, bias=bias)  # (B*nW, N, C)
            else:
                if N_actual == self.M**2:
                    C_r = self.proj_r(feat_C)  # (B*nW, C, r)
                    D_r = self.proj_r(feat_D)
                    w = self.proj_r.weight
                else:
                    w = self.proj_r.weight[:, :N_actual]  # (r, N_actual)
                    C_r = F.linear(feat_C, w)
                    D_r = F.linear(feat_D, w)
                scores = torch.bmm(feat_B.transpose(1, 2), C_r)  # (B*nW, N, r)
                if bias is not None:
                    scores = scores + F.linear(bias, w)  # (Q^T K + B) P = Q^T K P + B P
                scores = F.softmax(scores, dim=-1)
                E_out = torch.bmm(scores, D_r.transpose(1, 2))  # (B*nW, N, C)
        if self.or_fusion is not None:
            E_t = E_out.transpose(1, 2)  # (B*nW, C, N)
            if self.or_fusion == "max":
                y = torch.maximum(x_n, E_t)
            elif self.or_fusion == "relu_add":
                y = x_n + F.relu(E_t)
            else:  # "lse": smooth maximum, >= max(x, E)
                y = torch.logaddexp(x_n, E_t)
            return window_reverse(y.reshape(nBW, C, M, M), M, H, W)
        if self.attn_norm is None:
            E_n = self.alpha * E_out.transpose(1, 2)  # (B*nW, C, N)
            E_n = E_n + x_n if self.identity else E_n
            return window_reverse(E_n.reshape(nBW, C, M, M), M, H, W)
        E = window_reverse(E_out.transpose(1, 2).reshape(nBW, C, M, M), M, H, W)
        return self.alpha * self.attn_norm(E) + (x if self.identity else 0)


def affinity_loss(logits, label, num_classes, eps=1e-5):
    """Context Prior affinity loss (Yu et al., CVPR 2020) for the (B, N, N) logits of a PAM whose single window
    covers the h x w grid (N = h * w). Ideal affinity A = L L^T, L (B, N, K) = class proportions of each grid cell
    (one-hot labels average-pooled to h x w). Loss = BCE(sigmoid(logits), A) - mean over rows of
    [log precision + log recall + log specificity] (the paper's unary and global terms, weights 1 / 1)."""
    B, N, _ = logits.shape
    h = int(round(N**0.5))
    assert h * h == N and label.shape[0] == B, "affinity_loss needs one square window per image"
    onehot = F.one_hot(label.long(), num_classes).permute(0, 3, 1, 2).float()
    L = F.adaptive_avg_pool2d(onehot, (h, h)).flatten(2).transpose(1, 2)  # (B, N, K), row-major like the window
    A = torch.bmm(L, L.transpose(1, 2))
    logits = logits.float()
    P = torch.sigmoid(logits)
    unary = F.binary_cross_entropy_with_logits(logits, A)
    tp = (A * P).sum(-1)
    precision = torch.log((tp + eps) / (P.sum(-1) + eps))
    recall = torch.log((tp + eps) / (A.sum(-1) + eps))
    specificity = torch.log((((1 - A) * (1 - P)).sum(-1) + eps) / ((1 - A).sum(-1) + eps))
    return unary - (precision + recall + specificity).mean()


class TFWindowedPAM(nn.Module):
    """
    Transformer-style windowed low-rank position attention (journal experiment, opt-in via --pam_style tf).
    Pre-LayerNorm, full-width queries/keys with 1/sqrt(C) scaling, output projection, and either a LayerScale
    residual  x + gamma * E  (gamma per channel, initialised at ls_init and excluded from weight decay) or a
    per-pixel gate  g * x + (1 - g) * E  with g = sigmoid(W [x, E]).  Window size M and rank r act as in
    LowRankWindowedPAM (r = 0: exact attention inside each window).
    """

    def __init__(self, channels, window_size=7, rank=32, fusion="residual", ls_init=0.1):
        super().__init__()
        self.M = window_size
        N = window_size**2
        self.norm = nn.LayerNorm(channels)
        self.q = nn.Conv1d(channels, channels, 1)
        self.k = nn.Conv1d(channels, channels, 1)
        self.v = nn.Conv1d(channels, channels, 1)
        self.proj_r = nn.Linear(N, rank, bias=False) if rank > 0 else None
        self.proj_o = nn.Conv2d(channels, channels, 1)
        self.scale = channels**-0.5
        self.fusion = fusion
        if fusion == "residual":
            self.ls_gamma = nn.Parameter(torch.full((1, channels, 1, 1), float(ls_init)))
        elif fusion == "gate":
            self.gate = nn.Conv2d(2 * channels, channels, 1)
        else:
            raise ValueError(f"unknown fusion {fusion!r}")
        self.ablate = False  # True: return the input unchanged (attention switched off)

    def forward(self, x):
        if self.ablate:
            return x
        B, C, H, W = x.shape
        h = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        M = min(self.M, H, W)
        h_w = window_partition(h, M)
        nBW = h_w.shape[0]
        h_n = h_w.reshape(nBW, C, M * M)
        q, k, v = self.q(h_n) * self.scale, self.k(h_n), self.v(h_n)
        if self.proj_r is None:
            E_out = exact_window_attention(q, k, v)  # (B*nW, N, C)
        else:
            w = self.proj_r.weight[:, : M * M]  # leading rows when the window is clamped
            k_r, v_r = F.linear(k, w), F.linear(v, w)  # (B*nW, C, r)
            A = F.softmax(torch.bmm(q.transpose(1, 2), k_r), dim=-1)  # (B*nW, N, r)
            E_out = torch.bmm(A, v_r.transpose(1, 2))  # (B*nW, N, C)
        E = self.proj_o(window_reverse(E_out.transpose(1, 2).reshape(nBW, C, M, M), M, H, W))
        if self.fusion == "residual":
            return x + self.ls_gamma * E
        g = torch.sigmoid(self.gate(torch.cat([x, E], dim=1)))
        return g * x + (1.0 - g) * E


class GroupedCAM(nn.Module):
    """
    Channel attention split into G independent groups: O(C^2) -> O(C^2/G).
    Each group computes its own (C/G x C/G) attention matrix.
    """

    def __init__(self, channels, groups=8):
        super().__init__()
        # Narrow maps (e.g. 64/16 = 4 channels in the C/16 block) cannot hold
        # more groups than channels; fall back to one channel per group. Widths that
        # `groups` does not divide (LightViT-S: 192/16 = 12) use the largest divisor
        # below it (12 -> 6 groups); every width that divided before is unchanged.
        self.G = max(g for g in range(1, min(groups, channels) + 1) if channels % g == 0)
        self.beta = nn.Parameter(torch.zeros(1))
        self.ablate = False  # True: return the input unchanged (attention switched off)

    def forward(self, x):
        if self.ablate:
            return x
        B, C, H, W = x.shape
        G, Cg = self.G, C // self.G
        x_g = x.contiguous().view(B * G, Cg, H * W)  # (B*G, Cg, N)
        X = torch.bmm(x_g, x_g.transpose(1, 2))  # (B*G, Cg, Cg)
        # Match DA-TransUNet CAM_Module: softmax(max - energy), then aggregate
        # over the normalized axis, so G=1 reproduces the original CAM exactly.
        X = torch.max(X, -1, keepdim=True)[0].expand_as(X) - X
        X = F.softmax(X, dim=-1)
        E_g = torch.bmm(X, x_g)  # (B*G, Cg, N)
        E = self.beta * E_g + x_g
        return E.contiguous().view(B, C, H, W)


def norm(planes):
    """BatchNorm with the same settings as DA-TransUNet's DANetHead."""
    return nn.BatchNorm2d(planes, momentum=0.95, eps=1e-03)


def conv_bn_relu(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), norm(cout), nn.ReLU()
    )


class ApproxDABlock(nn.Module):
    """
    ApproxDA Block with DA-TransUNet's C/16 bottleneck (DANetHead layout).

    Identical to DANetHead except that PAM/CAM are replaced by
    LowRankWindowedPAM / GroupedCAM and the two branches are fused by a gate:
      x -> conv5a (3x3, C->C/16) -> windowed low-rank PAM -> conv51 (3x3) --\
      x -> conv5c (3x3, C->C/16) -> grouped CAM           -> conv52 (3x3) --+-> g-fusion
        -> conv8 (Dropout2d 0.05, 1x1 C/16->C, ReLU)          (no outer residual, as DA)
    PAM queries/keys use (C/16)/8 channels as in DA's PAM_Module (min 1).
    DANetHead's conv6/conv7 are omitted: their outputs are discarded in DA.
    gate_mode: 'pam' (g=1) / 'cam' (g=0) build only the branch they use;
    'fixed' g=0.5; 'learn' g = sigmoid(Linear(GAP(x))) per C/16 channel.
    """

    def __init__(self, channels, window_size=7, rank=32, groups=8, gate_mode="learn", alpha_init=0.0, attn_norm=False,
                 pam_identity=True, pam_style="danet", pam_fusion="residual", attn_off=False, ratio=16,
                 pam_relpos=False, pam_abspos=False, coord_gate=False, pam_affinity=False, third_branch="none",
                 num_classes=9):
        super().__init__()
        inter = channels // ratio  # DA's C/16; --skip_ratio widens the skip blocks
        if pam_affinity and (pam_style != "danet" or gate_mode == "cam"):
            raise ValueError("--pam_affinity is implemented for the DANet-style PAM with gate_mode != cam")
        self.pam_affinity = pam_affinity
        if (coord_gate or pam_relpos or pam_abspos) and pam_style != "danet":
            raise ValueError("--pam_relpos / --pam_abspos / --coord_gate are implemented for the DANet-style PAM")
        if coord_gate and gate_mode != "pam":
            raise ValueError("--coord_gate is implemented for gate_mode pam")
        self.gate_mode = gate_mode
        self.use_pam = gate_mode != "cam"
        self.use_cam = gate_mode != "pam"
        if self.use_pam:
            self.conv5a = conv_bn_relu(channels, inter)
            if pam_style == "tf":
                if pam_fusion not in ("residual", "gate"):
                    raise ValueError(f"--pam_fusion {pam_fusion} is for the DANet-style PAM only")
                self.pam = TFWindowedPAM(inter, window_size, rank, fusion=pam_fusion)
            else:
                self.pam = LowRankWindowedPAM(
                    inter, window_size, rank, qk_channels=max(1, inter // 8),
                    alpha_init=alpha_init, attn_norm=attn_norm, identity=pam_identity,
                    or_fusion=pam_fusion if pam_fusion in ("max", "relu_add", "lse") else None,
                    rel_pos=pam_relpos, abs_pos=pam_abspos, affinity_in=channels if pam_affinity else 0,
                )
            self.pam.ablate = attn_off  # conv-only head: PAM passes its input through
            # coord_gate (opt-in): direction-aware residual gate on the PAM output (third "branch", gate=pam only)
            self.coord = CoordGate(inter) if coord_gate else None
            self.conv51 = conv_bn_relu(inter, inter)
        if self.use_cam:
            self.conv5c = conv_bn_relu(channels, inter)
            self.cam = GroupedCAM(inter, groups)
            self.cam.ablate = attn_off  # conv-only head: with gate fixed / learn / cam the CAM branch is off as well
            self.conv52 = conv_bn_relu(inter, inter)
        if gate_mode == "learn":
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.gate_fc = nn.Linear(channels, inter)
        self.conv8 = nn.Sequential(
            nn.Dropout2d(0.05, False), nn.Conv2d(inter, channels, 1), nn.ReLU()
        )
        # third_branch (opt-in, bottleneck block only): organ-level attention or the triplet-attention control,
        # added to the block output at full width
        self.third = {"none": None, "organ": lambda: OrganBranch(channels, num_classes),
                      "triplet": lambda: TripletBranch(channels)}[third_branch]
        self.third = self.third() if self.third is not None else None

    def _pam(self, x):
        h = self.conv5a(x)
        return self.pam(h, qk_feat=x) if self.pam_affinity else self.pam(h)

    def forward(self, x):
        if self.gate_mode == "pam":
            h = self._pam(x)
            fused = self.conv51(h if self.coord is None else self.coord(h))
        elif self.gate_mode == "cam":
            fused = self.conv52(self.cam(self.conv5c(x)))
        else:
            p = self.conv51(self._pam(x))
            c = self.conv52(self.cam(self.conv5c(x)))
            if self.gate_mode == "fixed":
                g = 0.5
            else:  # 'learn'
                B = x.shape[0]
                g = torch.sigmoid(self.gate_fc(self.pool(x).view(B, -1))).view(B, -1, 1, 1)
            fused = g * p + (1.0 - g) * c
        if self.third is not None:
            return self.conv8(fused) + self.third(x)
        return self.conv8(fused)


class CoordGate(nn.Module):
    """Coordinate attention (Hou et al., CVPR 2021) as a residual, identity-initialised gate (--coord_gate):
    z_h(i) = mean_j x(:, i, j),  z_w(j) = mean_i x(:, i, j),  f = ReLU(BN(W1 [z_h; z_w])),
    y = x * (1 + tanh(a_h(i) + a_w(j))),  a_h = W_h f_h,  a_w = W_w f_w  (W_h, W_w zero-initialised).
    Direction-aware (row / column) gating that can suppress or amplify; the identity at initialisation."""

    def __init__(self, channels, reduction=4):
        super().__init__()
        mip = max(8, channels // reduction)
        self.conv1 = nn.Conv2d(channels, mip, 1, bias=False)
        self.bn = norm(mip)
        self.act = nn.ReLU()
        self.conv_h = nn.Conv2d(mip, channels, 1)
        self.conv_w = nn.Conv2d(mip, channels, 1)
        for c in (self.conv_h, self.conv_w):
            nn.init.zeros_(c.weight)
            nn.init.zeros_(c.bias)

    def forward(self, x):
        B, C, H, W = x.shape
        z = torch.cat([x.mean(3, keepdim=True), x.mean(2, keepdim=True).transpose(2, 3)], dim=2)  # (B, C, H+W, 1)
        f_h, f_w = self.act(self.bn(self.conv1(z))).split([H, W], dim=2)
        return x * (1 + torch.tanh(self.conv_h(f_h) + self.conv_w(f_w).transpose(2, 3)))


def organ_aux_loss(logits, label, num_classes):
    """Soft cross-entropy of the organ branch's coarse logits (B, K, h, w) against the class proportions of each
    grid cell (one-hot labels average-pooled to h x w)."""
    h, w = logits.shape[-2:]
    target = F.adaptive_avg_pool2d(F.one_hot(label.long(), num_classes).permute(0, 3, 1, 2).float(), (h, w))
    return -(target * F.log_softmax(logits.float(), dim=1)).sum(1).mean()


class OrganBranch(nn.Module):
    """Organ-level attention, the third branch of the bottleneck block (--third_branch organ). OCR-style (Yuan et
    al., ECCV 2020) plus an organ-organ relation step for the anatomical layout:
      f = 3x3 conv (C -> d);  coarse logits = 1x1 conv (d -> K), supervised by organ_aux_loss;
      soft region of class k = its per-pixel class probability p_k normalised over the positions (ACFNet-style class
      centre; OCR's spatial softmax stays near uniform at this resolution, measured); prototype f_k / centroid c_k =
      region-weighted mean of f / of the (y, x) grid in [-1, 1]; presence m_k = mean of p_k;
      organ-organ: f_k += Out(softmax(Q K^T / sqrt(d)) V) over the K tokens LayerNorm(f_k + Pos(c_k, m_k));
      pixel-organ: ctx_i = sum_k softmax_k(q_i . k_k / sqrt(d)) v_k;
      output = ReLU(BN(1x1 conv [f, ctx] -> C)), added to the block output (full width, not the C/16 path).
    ablate=True zeroes the organ context (the f path stays), for test-time contribution checks."""

    def __init__(self, channels, num_classes, dim=64):
        super().__init__()
        self.reduce = conv_bn_relu(channels, dim)
        self.aux = nn.Conv2d(dim, num_classes, 1)
        self.pos = nn.Linear(3, dim)  # (centroid y, centroid x, presence)
        self.rel_norm = nn.LayerNorm(dim)
        self.rel_qkv = nn.Linear(dim, 3 * dim)
        self.rel_out = nn.Linear(dim, dim)
        self.q = nn.Conv2d(dim, dim, 1)
        self.kv = nn.Linear(dim, 2 * dim)
        self.out = nn.Sequential(nn.Conv2d(2 * dim, channels, 1, bias=False), norm(channels), nn.ReLU())
        self.scale = dim**-0.5
        self.aux_logits = None
        self.ablate = False

    def forward(self, x):
        f = self.reduce(x)
        B, d, H, W = f.shape
        logits = self.aux(f)
        self.aux_logits = logits if self.training else None
        if self.ablate:
            return self.out(torch.cat([f, torch.zeros_like(f)], 1))
        prob = F.softmax(logits, dim=1).flatten(2)  # (B, K, N) per-pixel class probabilities
        presence = prob.mean(-1, keepdim=True)  # (B, K, 1)
        region = prob / (prob.sum(-1, keepdim=True) + 1e-6)
        proto = torch.bmm(region, f.flatten(2).transpose(1, 2))  # (B, K, d)
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, H, device=f.device, dtype=f.dtype),
                                torch.linspace(-1, 1, W, device=f.device, dtype=f.dtype), indexing="ij")
        centroid = torch.matmul(region, torch.stack([yy.flatten(), xx.flatten()], dim=-1))  # (B, K, 2)
        q, k, v = self.rel_qkv(self.rel_norm(proto + self.pos(torch.cat([centroid, presence], -1)))).chunk(3, dim=-1)
        proto = proto + self.rel_out(torch.bmm(F.softmax(torch.bmm(q, k.transpose(1, 2)) * self.scale, -1), v))
        k, v = self.kv(proto).chunk(2, dim=-1)  # (B, K, d)
        q = self.q(f).flatten(2).transpose(1, 2)  # (B, N, d)
        ctx = torch.bmm(F.softmax(torch.bmm(q, k.transpose(1, 2)) * self.scale, -1), v)  # (B, N, d)
        return self.out(torch.cat([f, ctx.transpose(1, 2).reshape(B, d, H, W)], 1))


class _ZPoolGate(nn.Module):
    """Triplet-attention gate: [max, mean] over dim 1 -> 7x7 conv -> BN -> sigmoid -> rescale."""

    def __init__(self, k=7):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(2, 1, k, padding=k // 2, bias=False), nn.BatchNorm2d(1))

    def forward(self, x):
        z = torch.cat([x.amax(1, keepdim=True), x.mean(1, keepdim=True)], dim=1)
        return x * torch.sigmoid(self.conv(z))


class TripletBranch(nn.Module):
    """Unsupervised third-branch control (--third_branch triplet): triplet attention (Misra et al., WACV 2021), the
    mean of three rotation-based gates over the (C, W), (H, C) and (H, W) interactions, with the same reduction,
    [f, attended] concatenation and full-width output projection as OrganBranch."""

    def __init__(self, channels, dim=64):
        super().__init__()
        self.reduce = conv_bn_relu(channels, dim)
        self.cw, self.hc, self.hw = _ZPoolGate(), _ZPoolGate(), _ZPoolGate()
        self.out = nn.Sequential(nn.Conv2d(2 * dim, channels, 1, bias=False), norm(channels), nn.ReLU())
        self.ablate = False

    def forward(self, x):
        f = self.reduce(x)
        if self.ablate:
            t = torch.zeros_like(f)
        else:
            t = (self.cw(f.permute(0, 2, 1, 3)).permute(0, 2, 1, 3) + self.hc(f.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
                 + self.hw(f)) / 3
        return self.out(torch.cat([f, t], 1))


class SliceAttention(nn.Module):
    """Inter-slice attention at the bottleneck (--slice_attn K): every token of the centre slice attends to the tokens
    of the 2K+1 adjacent slices (itself included; S*N keys), each encoded by the shared 2-D encoder, with a learned
    slice-offset embedding and a learned token-position embedding; pre-norm multi-head attention + MLP with plain
    residuals (no zero / LayerScale initialisation: gated residual attention never trained under SGD in this model).
    K = 0 is the control: the same layer over the centre slice only. ablate: True / "all" returns the centre tokens
    unchanged; "neighbours" restricts the keys to the centre slice (test-time contribution of the context)."""

    def __init__(self, dim, num_slices, num_tokens, heads=4, mlp_ratio=2):
        super().__init__()
        self.num_slices = num_slices
        self.slice_emb = nn.Parameter(torch.zeros(num_slices, 1, dim))
        self.pos_emb = nn.Parameter(torch.zeros(1, num_tokens, dim))
        nn.init.trunc_normal_(self.slice_emb, std=0.02)
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm_mlp = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * mlp_ratio), nn.GELU(), nn.Linear(dim * mlp_ratio, dim))
        self.ablate = False

    def forward(self, t):
        """t: (B, S, N, D) tokens of the S slices -> (B, N, D) centre-slice tokens with inter-slice context."""
        B, S, N, D = t.shape
        c = S // 2
        if self.ablate is True or self.ablate == "all":
            return t[:, c]
        e = t + self.slice_emb[None] + self.pos_emb[None]  # (B, S, N, D)
        q = e[:, c]
        kv = e[:, c] if self.ablate == "neighbours" else e.reshape(B, S * N, D)
        kv = self.norm_kv(kv)
        x = t[:, c] + self.attn(self.norm_q(q), kv, kv, need_weights=False)[0]
        return x + self.mlp(self.norm_mlp(x))


class PointRefineHead(nn.Module):
    """Native-resolution boundary refinement (--point_refine), PointRend-style (Kirillov et al., CVPR 2020). The network
    predicts at 224 x 224; at the most uncertain points (smallest top-1 / top-2 logit margin) a point-wise MLP re-labels
    the point from [coarse logits, decoder features (bilinear), 5 x 5 native-resolution intensity patch]. Coordinates are
    normalised with align_corners=True, the convention of scipy.ndimage.zoom used by the data pipeline, so 224 maps,
    the 512 training canvas and the native test slices refer to the same points. The patch step is one pixel of the
    512 canvas at every native size (same physical extent relative to the field of view).
    Training: per image, train_points chosen from oversample * train_points random points, a fraction `importance`
    by uncertainty and the rest uniform; cross-entropy against the 512-canvas label (nearest).
    Inference: refine() re-labels the test_points most uncertain native pixels (scaled with the slice area).
    Small-organ options (v2; both off = v1):
      residual (--pr_residual): the point logits are the coarse logits plus the MLP output, whose last layer starts
        at zero, so the head starts as the identity and only corrects; a class it never learned keeps its coarse label.
      balanced (--pr_balanced): class_share of the training points are spread evenly over the classes present in
        the image (uniform inside each class), and the point loss is 0.5 CE + 0.5 Dice over the points like the main
        loss. v1 put 0.01 - 0.02 % of the points on the adrenal glands and erased them at test time (JOURNAL 4.9)."""

    def __init__(self, num_classes, feat_channels, hidden=64, patch=5, canvas=512, train_points=2048, oversample=3,
                 importance=0.75, test_points=8192, residual=False, balanced=False, class_share=0.25):
        super().__init__()
        self.residual, self.balanced, self.class_share = residual, balanced, class_share
        self.canvas, self.train_points, self.oversample, self.importance, self.test_points = canvas, train_points, oversample, importance, test_points
        r = patch // 2
        d = torch.arange(-r, r + 1, dtype=torch.float32) * 2.0 / (canvas - 1)
        dy, dx = torch.meshgrid(d, d, indexing="ij")
        self.register_buffer("offsets", torch.stack([dx.flatten(), dy.flatten()], -1), persistent=False)  # (P, 2) as (x, y)
        cin = num_classes + feat_channels + patch * patch
        self.mlp = nn.Sequential(nn.Conv1d(cin, hidden, 1), nn.ReLU(), nn.Conv1d(hidden, hidden, 1), nn.ReLU(),
                                 nn.Conv1d(hidden, hidden, 1), nn.ReLU(), nn.Conv1d(hidden, num_classes, 1))
        if residual:
            nn.init.zeros_(self.mlp[-1].weight)
            nn.init.zeros_(self.mlp[-1].bias)

    @staticmethod
    def sample(x, coords, mode="bilinear"):
        """x (B, C, H, W), coords (B, N, 2) in [-1, 1] as (x, y) -> (B, C, N)."""
        return F.grid_sample(x, coords.unsqueeze(2), mode=mode, align_corners=True).squeeze(3)

    @staticmethod
    def uncertainty(logits):
        """logits (B, K, ...) -> -(top1 - top2) (B, 1, ...): largest where the two best classes are closest."""
        top2 = logits.topk(2, dim=1).values
        return -(top2[:, :1] - top2[:, 1:2])

    def point_logits(self, coarse, feat, image, coords):
        B, N = coords.shape[:2]
        patch = (coords.unsqueeze(2) + self.offsets.to(coords.dtype)).reshape(B, -1, 2)  # (B, N * P, 2)
        intens = self.sample(image, patch).reshape(B, N, -1).transpose(1, 2)  # (B, P, N)
        c = self.sample(coarse, coords)
        out = self.mlp(torch.cat([c, self.sample(feat, coords), intens], 1))
        return c + out if self.residual else out

    @staticmethod
    def class_points(label_hr, n):
        """n points per image spread evenly over the classes present, uniform inside each class (canvas pixels)."""
        B, S = label_hr.shape[0], label_hr.shape[-1]
        flat = label_hr.flatten(1)
        coords = torch.rand(B, n, 2, device=label_hr.device) * 2 - 1  # fallback: uniform
        for b in range(B):
            classes = torch.unique(flat[b])
            per = n // len(classes)
            picks = []
            for c in classes:
                idx = torch.nonzero(flat[b] == c).squeeze(1)
                picks.append(idx[torch.randint(len(idx), (per,), device=idx.device)])
            pix = torch.cat(picks)
            coords[b, : len(pix)] = torch.stack([2 * (pix % S).float() / (S - 1) - 1, 2 * (pix // S).float() / (S - 1) - 1], -1)
        return coords

    @staticmethod
    def point_dice(logits, target, num_classes):
        """1 - mean soft Dice (squared denominators, as DiceLoss) over the classes present among the points."""
        p = F.softmax(logits, dim=1)
        t = F.one_hot(target, num_classes).permute(0, 2, 1).float()
        inter, den = (p * t).sum((0, 2)), (p * p).sum((0, 2)) + (t * t).sum((0, 2))
        present = t.sum((0, 2)) > 0
        return 1 - ((2 * inter + 1e-5) / (den + 1e-5))[present].mean()

    def train_loss(self, coarse, feat, image_hr, label_hr):
        """coarse (B, K, h, w) logits, feat (B, C, h, w), image_hr / label_hr (B, S, S) on the training canvas."""
        B = coarse.shape[0]
        n_cls = int(self.class_share * self.train_points) if self.balanced else 0
        n_unc = self.train_points - n_cls
        n_imp = int(self.importance * n_unc)
        with torch.no_grad():
            cand = torch.rand(B, self.oversample * n_unc, 2, device=coarse.device) * 2 - 1
            u = self.uncertainty(self.sample(coarse.detach(), cand)).squeeze(1)  # (B, kN)
            idx = u.topk(n_imp, dim=1).indices
            parts = [torch.gather(cand, 1, idx.unsqueeze(-1).expand(-1, -1, 2)),
                     torch.rand(B, n_unc - n_imp, 2, device=coarse.device) * 2 - 1]
            if n_cls:
                parts.append(self.class_points(label_hr, n_cls))
            coords = torch.cat(parts, 1)
            target = self.sample(label_hr.unsqueeze(1).float(), coords, mode="nearest").squeeze(1).long()
        logits = self.point_logits(coarse, feat, image_hr.unsqueeze(1).float(), coords)
        if self.balanced:
            return 0.5 * F.cross_entropy(logits, target) + 0.5 * self.point_dice(logits, target, logits.shape[1])
        return F.cross_entropy(logits, target)

    def refine(self, coarse, feat, image):
        """coarse (1, K, h, w), feat (1, C, h, w), image (1, 1, H, W) native slice -> (1, K, H, W) logits."""
        H, W = image.shape[-2:]
        up = F.interpolate(coarse, size=(H, W), mode="bilinear", align_corners=True)
        n = min(H * W, int(round(self.test_points * H * W / self.canvas**2)))
        idx = self.uncertainty(up).flatten(1).topk(n, dim=1).indices  # (1, n)
        rows, cols = (idx // W).float(), (idx % W).float()
        coords = torch.stack([2 * cols / (W - 1) - 1, 2 * rows / (H - 1) - 1], -1)  # (1, n, 2)
        out = up.flatten(2)
        out[:, :, idx[0]] = self.point_logits(coarse, feat, image, coords)
        return out.view_as(up)


def _balanced_points(label_hr, n, num_classes):
    """PointRefineHead.class_points for the point loss: the same distribution (n // classes-present points per class,
    uniform inside each class, uniform fallback for the remainder) with one host sync per batch and one multinomial
    draw per image instead of a nonzero per class and image."""
    B, S = label_hr.shape[0], label_hr.shape[-1]
    flat = label_hr.flatten(1)
    coords = torch.rand(B, n, 2, device=label_hr.device) * 2 - 1  # fallback: uniform
    hist = torch.zeros(B, num_classes, device=label_hr.device).scatter_add_(1, flat, torch.ones_like(flat, dtype=torch.float))
    present = (hist > 0).cpu()
    for b in range(B):
        cls = torch.nonzero(present[b]).squeeze(1).to(label_hr.device)
        per = n // len(cls)
        pix = torch.multinomial((flat[b].unsqueeze(0) == cls.unsqueeze(1)).float(), per, replacement=True).flatten()
        coords[b, : len(pix)] = torch.stack([2 * (pix % S).float() / (S - 1) - 1, 2 * (pix // S).float() / (S - 1) - 1], -1)
    return coords


def point_supervision_loss(coarse, label_hr, train_points=2048, oversample=3, importance=0.75, class_share=0.0):
    """Native-resolution point supervision (--point_loss): a training-only loss, nothing is added to the model.
    PointRend's training-point selection (Kirillov et al., CVPR 2020) without its point head: per image, `importance`
    of the points are the most uncertain (smallest top-1 / top-2 logit margin) of oversample * train_points random
    points, the rest uniform; the coarse logits (B, K, h, w) are bilinearly sampled at the points (align_corners=True,
    as scipy zoom) and supervised by cross-entropy against the native-resolution label (B, S, S) sampled nearest.
    The main loss sees labels nearest-downsampled to h x w; this one asks the interpolated logits to place the
    boundaries where the native labels put them.
    class_share > 0 (--ptl_balanced): that share of the points is spread evenly over the classes present in the image
    (uniform inside each class) and the uncertainty / uniform split applies to the rest. Without it ~0.01 - 0.02 % of
    the points fall on the adrenal glands and the loss can erase them (JOURNAL 4.9, 4.10). class_share = 0 is the
    original loss, bit for bit."""
    B = coarse.shape[0]
    n_cls = int(class_share * train_points)
    n_unc = train_points - n_cls
    n_imp = int(importance * n_unc)
    with torch.no_grad():
        cand = torch.rand(B, oversample * n_unc, 2, device=coarse.device) * 2 - 1
        u = PointRefineHead.uncertainty(PointRefineHead.sample(coarse.detach(), cand)).squeeze(1)
        idx = u.topk(n_imp, dim=1).indices
        parts = [torch.gather(cand, 1, idx.unsqueeze(-1).expand(-1, -1, 2)),
                 torch.rand(B, n_unc - n_imp, 2, device=coarse.device) * 2 - 1]
        if n_cls:
            parts.append(_balanced_points(label_hr, n_cls, coarse.shape[1]))
        coords = torch.cat(parts, 1)
        target = PointRefineHead.sample(label_hr.unsqueeze(1).float(), coords, mode="nearest").squeeze(1).long()
    return F.cross_entropy(PointRefineHead.sample(coarse, coords), target)


class RefineBlock(nn.Module):
    """Attention-free skip refinement (--block_type refine): the convolution path of the DANetHead layout, which carries
    all of the measured gain of the DA / ApproxDA block (JOURNAL_EXPERIMENTS.md 3f). No residual: the feature passes a
    C -> C/ratio bottleneck.
      plain:   3x3 C->C' (BN, ReLU) -> 3x3 C'->C' (BN, ReLU) -> Dropout2d -> 1x1 C'->C (ReLU)
      dilated: the middle 3x3 becomes the sum of 3x3 convs with dilation 1, 2 and 4 (multi-scale local context)
      guided:  the decoder feature at the same resolution (guide_channels) is projected to C' and added after the first
               conv, so the skip is refined conditioned on the decoder's semantics (slim decoder only)."""

    def __init__(self, channels, ratio=16, refine_mode="plain", guide_channels=0):
        super().__init__()
        inter = max(1, channels // ratio)
        self.refine_mode = refine_mode
        self.conv_in = conv_bn_relu(channels, inter)
        if refine_mode == "dilated":
            self.mid = nn.ModuleList(
                [nn.Sequential(nn.Conv2d(inter, inter, 3, padding=d, dilation=d, bias=False), norm(inter)) for d in (1, 2, 4)]
            )
            self.mid_act = nn.ReLU()
        else:
            self.mid = conv_bn_relu(inter, inter)
        if refine_mode == "guided":
            self.guide = nn.Sequential(nn.Conv2d(guide_channels, inter, 1, bias=False), norm(inter))
        self.conv_out = nn.Sequential(nn.Dropout2d(0.05, False), nn.Conv2d(inter, channels, 1), nn.ReLU())

    def forward(self, x, guide=None):
        h = self.conv_in(x)
        if self.refine_mode == "guided":
            h = F.relu(h + self.guide(guide))
        if self.refine_mode == "dilated":
            h = self.mid_act(sum(m(h) for m in self.mid))
        else:
            h = self.mid(h)
        return self.conv_out(h)


class MedNeXtBlock(nn.Module):
    """MedNeXt-style residual conv block (Roy et al., MICCAI 2023) used in place of an ApproxDABlock
    (--block_type mednext): depthwise k x k conv -> GroupNorm (one group per channel) -> 1x1 expand (exp x C)
    -> GELU -> 1x1 back to C, plus the identity. Attention-free test of "only the convolutions matter"."""

    def __init__(self, channels, kernel_size=5, exp=2):
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size // 2, groups=channels)
        self.norm = nn.GroupNorm(channels, channels)
        self.pw1 = nn.Conv2d(channels, exp * channels, 1)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(exp * channels, channels, 1)

    def forward(self, x):
        return x + self.pw2(self.act(self.pw1(self.norm(self.dw(x)))))


class ApproxDABlockLegacy(nn.Module):
    """
    Legacy ApproxDA Block (BIBM 2026 submission): full-width, no C/16 bottleneck.
    LowRankWindowedPAM + GroupedCAM on all C channels, blended via a soft gate,
    1x1 fusion and an outer residual. Kept for reproducing the old results.
    Note: GroupedCAM was later aligned with DA-TransUNet's CAM_Module, so
    cam/fixed/learn results differ from the original BIBM runs.
    gate_mode controls routing for ablation:
      'learn' — per-channel learned gate (default)
      'fixed' — fixed 0.5 blend
      'pam'   — PAM only (g=1)
      'cam'   — CAM only (g=0)
    """

    def __init__(self, channels, window_size=7, rank=32, groups=8, gate_mode="learn"):
        super().__init__()
        self.gate_mode = gate_mode
        self.pam = LowRankWindowedPAM(channels, window_size, rank)
        self.cam = GroupedCAM(channels, groups)
        self.pool = nn.AdaptiveAvgPool2d(1)
        if gate_mode == "learn":
            self.gate_fc = nn.Linear(channels, channels)
        self.fusion = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        pam_out = self.pam(x)
        cam_out = self.cam(x)
        if self.gate_mode == "pam":
            g = 1.0
        elif self.gate_mode == "cam":
            g = 0.0
        elif self.gate_mode == "fixed":
            g = 0.5
        else:  # 'learn'
            g = torch.sigmoid(self.gate_fc(self.pool(x).view(x.shape[0], -1))).view(
                x.shape[0], -1, 1, 1
            )  # (B, C, 1, 1)
        fused = self.fusion(g * pam_out + (1.0 - g) * cam_out)
        return fused + x


def hardware_config(free_mem_gb):
    """Select rank/window_size/groups based on available GPU memory (GB)."""
    if free_mem_gb > 8:
        return {"rank": 64, "window_size": 14, "groups": 4}
    elif free_mem_gb > 4:
        return {"rank": 32, "window_size": 7, "groups": 8}
    else:
        return {"rank": 16, "window_size": 7, "groups": 16}
