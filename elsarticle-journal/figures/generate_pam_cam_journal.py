"""
Generate approxda_pam_journal.pdf and approxda_cam_journal.pdf — the computation inside the two attention branches of the
ApproxDA block, in the box style of generate_block_decoder_journal.py. Shapes follow LowRankWindowedPAM and GroupedCAM in
experiments/ApproxDA-TransUNet/Architecture/block.py (one window / one group shown).

Run:
    python elsarticle-journal/figures/generate_pam_cam_journal.py
"""

import os

import matplotlib.pyplot as plt

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
from matplotlib.patches import FancyBboxPatch  # noqa: E402

C_IO = "#EBEBEB"
C_PAM = "#E8D5F5"
C_CAM = "#D5F0E0"
C_OP = "#FDE9C9"
EDGE = "#444444"
PARAM = "#C0501A"
OUT = os.path.dirname(os.path.abspath(__file__))
FS, SFS = 15, 13


def box(ax, cx, cy, w, h, color, label, sublabel=""):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.07", linewidth=0.9,
                                edgecolor=EDGE, facecolor=color, zorder=3))
    if sublabel:
        ax.text(cx, cy + 0.15, label, ha="center", va="center", fontsize=FS, fontweight="normal", zorder=4)
        ax.text(cx, cy - 0.17, sublabel, ha="center", va="center", fontsize=SFS, color="#333333", zorder=4)
    else:
        ax.text(cx, cy, label, ha="center", va="center", fontsize=FS, fontweight="normal", zorder=4)


def arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", color="#333333", lw=1.2, shrinkA=0, shrinkB=0), zorder=5)


def poly(ax, pts):
    xs, ys = zip(*pts)
    ax.plot(xs[:-1], ys[:-1], color="#333333", lw=1.2, zorder=5)
    arrow(ax, *pts[-2], *pts[-1])


def circle(ax, cx, cy, r=0.19):
    ax.add_patch(plt.Circle((cx, cy), r, color="white", ec=EDGE, lw=0.9, zorder=5))
    ax.text(cx, cy, "⊕", ha="center", va="center", fontsize=13, zorder=6)


def note(ax, x, y, s, ha="center", fs=11, color="#222222", style="italic", weight="normal"):
    ax.text(x, y, s, ha=ha, va="center", fontsize=fs, color=color, style=style, fontweight=weight, zorder=6)


def save(fig, name):
    plt.tight_layout(pad=0.2)
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight", dpi=300)
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
# Low-rank windowed PAM (one window)
# ════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12.5, 4.0))
ax.set_xlim(0, 12.5)
ax.set_ylim(0.0, 4.0)
ax.axis("off")
W, H, R = 1.55, 0.80, 0.19
YQ, YK, YV = 3.00, 1.95, 0.90
YA, YE = (YQ + YK) / 2, 1.40
X0, X1, X2, X3, X4, X5, X6, X7 = 0.55, 1.90, 3.90, 5.70, 7.95, 9.80, 11.0, 11.95
WA, W0, W7, WW, WE = 2.30, 0.95, 1.0, 1.25, 1.35
box(ax, X0, YK, W0, H, C_IO, r"$X'$", r"$C'HW$")
box(ax, X1, YK, WW, H, C_PAM, "Window", "partition")
box(ax, X2, YQ, W, H, C_OP, r"$Q = W_Q X_w$", r"$C_q\times N_w$")
box(ax, X2, YK, W, H, C_OP, r"$K = W_K X_w$", r"$C_q\times N_w$")
box(ax, X2, YV, W, H, C_OP, r"$V = W_V X_w$", r"$C'\times N_w$")
box(ax, X3, YK, W, H, C_PAM, r"$\tilde{K} = KP$", r"$C_q\times r$")
box(ax, X3, YV, W, H, C_PAM, r"$\tilde{V} = VP$", r"$C'\times r$")
box(ax, X4, YA, WA, H, C_PAM, r"$A = \mathrm{softmax}(Q^\top\tilde{K})$", r"$N_w\times r$")
box(ax, X5, YE, WE, H, C_PAM, r"$E = A\tilde{V}^\top$", r"$N_w\times C'$")
circle(ax, X6, YE, R)
box(ax, X7, YE, W7, H, C_IO, r"$E_{\mathrm{PAM}}$", r"$C'HW$")
arrow(ax, X0 + W0 / 2, YK, X1 - WW / 2, YK)
xb = (X1 + WW / 2 + X2 - W / 2) / 2  # branch point midway between the window box and Q, K, V
ax.plot([X1 + WW / 2, xb], [YK, YK], color="#333333", lw=1.2, zorder=5)
ax.plot([xb, xb], [YV, YQ], color="#333333", lw=1.2, zorder=5)
ax.add_patch(plt.Circle((xb, YK), 0.045, color="#333333", zorder=6))
for y in (YQ, YK, YV):
    arrow(ax, xb, y, X2 - W / 2, y)
arrow(ax, X2 + W / 2, YK, X3 - W / 2, YK)
arrow(ax, X2 + W / 2, YV, X3 - W / 2, YV)
xa = X4 - WA / 2
xj = (X3 + W / 2 + xa) / 2  # joint column between the K~ box and the A box
poly(ax, [(X2 + W / 2, YQ), (xj, YQ), (xj, YA + 0.15), (xa, YA + 0.15)])
poly(ax, [(X3 + W / 2, YK), (xj, YK), (xj, YA - 0.15), (xa, YA - 0.15)])
poly(ax, [(X4 + WA / 2, YA), (X5, YA), (X5, YE + H / 2)])
poly(ax, [(X3 + W / 2, YV), (X5, YV), (X5, YE - H / 2)])
arrow(ax, X5 + WE / 2, YE, X6 - R, YE)
note(ax, X6, YE - R - 0.30, r"$\alpha E + X_w$", fs=13, style="normal")
poly(ax, [(X1, YK + H / 2), (X1, 3.72), (X6, 3.72), (X6, YE + R)])
note(ax, (X3 + X4) / 2, 3.87, r"identity $X_w$", fs=15)
arrow(ax, X6 + R, YE, X7 - W7 / 2, YE)
note(ax, X7, YE - H / 2 - 0.52, "window\nreverse", fs=12)
note(ax, X1, YK - H / 2 - 0.24, r"$M\times M$ windows", fs=12)
note(ax, X1 - 0.55, YK + H / 2 + 0.28, r"$M$", fs=19, color=PARAM, weight="normal", style="normal")
note(ax, X3, YK + H / 2 + 0.28, r"$r$", fs=19, color=PARAM, weight="normal", style="normal")
note(ax, X3, YV - H / 2 - 0.24, r"$P\in\mathbb{R}^{N_w\times r}$, shared", fs=12, style="normal")
note(ax, 7.1, 0.22, r"$N_w = M^2$,  $C_q = C'/8$,  $\alpha$ initialized to 0", ha="left", fs=12, style="normal")
save(fig, "approxda_pam_journal")

# ════════════════════════════════════════════════════════════════════════════
# Grouped CAM (one group)
# ════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12.5, 2.85))
ax.set_xlim(0, 12.5)
ax.set_ylim(0.0, 2.85)
ax.axis("off")
Y, W, H, R = 1.45, 1.55, 0.80, 0.19
XS = [0.55, 2.05, 3.80, 5.55, 7.30, 8.68, 9.90, 11.60]
box(ax, XS[0], Y, 0.95, H, C_IO, r"$X'$", r"$C'\times N$")
box(ax, XS[1], Y, W, H, C_CAM, "Group split", r"$C_g\times N$ each")
box(ax, XS[2], Y, W, H, C_CAM, "Energy", r"$S = X_g X_g^\top$")
box(ax, XS[3], Y, W, H, C_CAM, "Softmax", r"$A$: $C_g\times C_g$")
box(ax, XS[4], Y, W, H, C_CAM, r"$E = AX_g$", r"$C_g\times N$")
circle(ax, XS[5], Y, R)
box(ax, XS[6], Y, W, H, C_IO, "Concat", r"$C'\times N$")
box(ax, XS[7], Y, 1.3, H, C_IO, r"$E_{\mathrm{CAM}}$", r"$C'HW$")
arrow(ax, XS[0] + 0.95 / 2, Y, XS[1] - W / 2, Y)
for a, b in zip(XS[1:4], XS[2:5]):
    arrow(ax, a + W / 2, Y, b - W / 2, Y)
arrow(ax, XS[4] + W / 2, Y, XS[5] - R, Y)
note(ax, XS[5], Y - R - 0.30, r"$\beta E + X_g$", fs=13, style="normal")
arrow(ax, XS[5] + R, Y, XS[6] - W / 2, Y)
arrow(ax, XS[6] + W / 2, Y, XS[7] - 1.3 / 2, Y)
poly(ax, [(XS[1], Y + H / 2), (XS[1], 2.58), (XS[5], 2.58), (XS[5], Y + R)])
note(ax, (XS[2] + XS[4]) / 2, 2.72, r"identity $X_g$", fs=15)
poly(ax, [(XS[1] + 0.35, Y - H / 2), (XS[1] + 0.35, 0.50), (XS[4] - 0.35, 0.50), (XS[4] - 0.35, Y - H / 2)])
note(ax, (XS[2] + XS[3]) / 2, 0.30, r"$X_g$", fs=15)
note(ax, XS[3], Y + H / 2 + 0.26, r"over $\max S - S$", fs=14, style="normal")
note(ax, XS[1] - 0.6, Y + H / 2 + 0.28, r"$G$", fs=19, color=PARAM, weight="normal", style="normal")
note(ax, 8.2, 0.30, r"$C_g = C'/G$,  $N = HW$,  $\beta$ initialized to 0", ha="left", fs=12, style="normal")
save(fig, "approxda_cam_journal")
print("Saved approxda_pam_journal.pdf and approxda_cam_journal.pdf")
