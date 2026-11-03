"""
Generate approxda_decoder_journal.pdf (paper Fig. 2b) — one slim decoder stage, in the box style of
generate_overview_journal.py. Fig. 2a (the ApproxDA block) is plotnn/approxda_journal_block.py (PlotNeuralNet).

Decoder stage (SlimDecoderBlock, ApproxDATransUNet.py): up x2 (previous) + RB(ApproxDA(skip)) -> RB.

Run:
    python elsarticle-journal/figures/generate_block_decoder_journal.py
"""

import os

import matplotlib.pyplot as plt

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
from matplotlib.patches import FancyBboxPatch  # noqa: E402

C_ENC = "#D6E8F7"
C_CONV = "#FDE9C9"
C_PAM = "#E8D5F5"
C_CAM = "#D5F0E0"
C_GATE = "#D9C2EE"
C_DEC = "#C9EDE8"
C_IO = "#EBEBEB"
EDGE = "#444444"
PARAM = "#C0501A"
OUT = os.path.dirname(os.path.abspath(__file__))


def box(ax, cx, cy, w, h, color, label, sublabel="", fs=11, sfs=9.5):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.07", linewidth=0.9,
                                edgecolor=EDGE, facecolor=color, zorder=3))
    if sublabel:
        ax.text(cx, cy + 0.11, label, ha="center", va="center", fontsize=fs, fontweight="normal", zorder=4)
        ax.text(cx, cy - 0.13, sublabel, ha="center", va="center", fontsize=sfs, color="#333333", zorder=4)
    else:
        ax.text(cx, cy, label, ha="center", va="center", fontsize=fs, fontweight="normal", zorder=4)


def arrow(ax, x0, y0, x1, y1, lw=1.2):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", color="#333333", lw=lw, shrinkA=0, shrinkB=0), zorder=5)


def poly_arrow(ax, pts, lw=1.2):
    xs, ys = zip(*pts)
    ax.plot(xs[:-1], ys[:-1], color="#333333", lw=lw, zorder=5)
    arrow(ax, *pts[-2], *pts[-1], lw=lw)


def add_circle(ax, cx, cy, r=0.13):
    ax.add_patch(plt.Circle((cx, cy), r, color="white", ec=EDGE, lw=0.9, zorder=5))
    ax.text(cx, cy, "⊕", ha="center", va="center", fontsize=12, zorder=6)


def note(ax, x, y, s, ha="center", fs=9.5, color="#222222", style="italic", weight="normal"):
    ax.text(x, y, s, ha=ha, va="center", fontsize=fs, color=color, style=style, fontweight=weight, zorder=6)


def save(fig, name):
    plt.tight_layout(pad=0.2)
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight", dpi=300)
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
# (b) slim decoder stage, with the residual reduction block (RB); compact canvas so that the text stays large
# ════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10.2, 2.75))
ax.set_xlim(0, 10.2)
ax.set_ylim(0.0, 2.75)
ax.axis("off")

Y_S, Y_D = 2.05, 0.85
W, H = 1.12, 0.66
FS, SFS = 13, 11.5
XE, XA, XR, XP, XR2, XN = 0.62, 2.05, 3.48, 4.42, 5.38, 6.80
box(ax, XE, Y_S, W, H, C_ENC, "Skip", r"$(2h)^2\times C_s$", fs=FS, sfs=SFS)
box(ax, XA, Y_S, W, H, C_PAM, "ApproxDA", r"$(2h)^2\times C_s$", fs=FS, sfs=SFS)
box(ax, XR, Y_S, W, H, C_DEC, "RB", r"$C_s \to 64$", fs=FS, sfs=SFS)
box(ax, XE, Y_D, W, H, C_IO, "Previous", r"$h^2\times 64$", fs=FS, sfs=SFS)
box(ax, XA, Y_D, W, H, C_DEC, "Up ×2", "bilinear", fs=FS, sfs=SFS)
add_circle(ax, XP, Y_D, r=0.18)
box(ax, XR2, Y_D, W, H, C_DEC, "RB", r"$64 \to 64$", fs=FS, sfs=SFS)
box(ax, XN, Y_D, W, H, C_IO, "Next", r"$(2h)^2\times 64$", fs=FS, sfs=SFS)
arrow(ax, XE + W / 2, Y_S, XA - W / 2, Y_S)
arrow(ax, XA + W / 2, Y_S, XR - W / 2, Y_S)
poly_arrow(ax, [(XR + W / 2, Y_S), (XP, Y_S), (XP, Y_D + 0.18)])
arrow(ax, XE + W / 2, Y_D, XA - W / 2, Y_D)
arrow(ax, XA + W / 2, Y_D, XP - 0.18, Y_D)
arrow(ax, XP + 0.18, Y_D, XR2 - W / 2, Y_D)
arrow(ax, XR2 + W / 2, Y_D, XN - W / 2, Y_D)
note(ax, XP + 0.1, (Y_S + Y_D) / 2 + 0.05, "add", ha="left", fs=12)
note(ax, 0.05, 0.25, r"$C_s$ = 512 / 256 / 64 at the 1/8, 1/4, 1/2 skips", ha="left", fs=11.5, style="normal")

# RB detail
RX, RW = 7.62, 2.50
ax.add_patch(FancyBboxPatch((RX, 0.12), RW, 2.50, boxstyle="round,pad=0.05", linewidth=0.9, edgecolor="#999999",
                            facecolor="white", linestyle="dashed", zorder=2))
ax.text(RX + RW / 2, 2.46, "RB", ha="center", va="center", fontsize=13, fontweight="normal", color="#7A4800", zorder=6)
cx, sx = RX + 1.62, RX + 0.38
Y1, Y2, YS = 1.86, 1.16, 0.50
box(ax, cx, Y1, 1.36, 0.42, C_DEC, "3×3 · BN · ReLU", fs=10.5)
box(ax, cx, Y2, 1.36, 0.42, C_DEC, "3×3 · BN", fs=10.5)
box(ax, sx, (Y1 + Y2) / 2, 0.52, 0.42, C_IO, "1×1", sublabel="", fs=10.5)
add_circle(ax, cx, YS, r=0.14)
note(ax, cx + 0.22, YS, "ReLU", ha="left", fs=10.5, style="normal")
note(ax, RX + 0.16, 2.22, "in", ha="left", fs=11)
arrow(ax, cx, Y1 - 0.21, cx, Y2 + 0.21)
arrow(ax, cx, Y2 - 0.21, cx, YS + 0.14)
poly_arrow(ax, [(sx, 2.22), (cx, 2.22), (cx, Y1 + 0.21)])
arrow(ax, sx, 2.22, sx, (Y1 + Y2) / 2 + 0.21)
poly_arrow(ax, [(sx, (Y1 + Y2) / 2 - 0.21), (sx, YS), (cx - 0.14, YS)])
note(ax, sx + 0.06, (Y2 + YS) / 2 - 0.12, "shortcut", ha="left", fs=9.5)
save(fig, "approxda_decoder_journal")
print("Saved approxda_decoder_journal.pdf")
