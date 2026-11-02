"""
Generate approxda_overview_journal.pdf — ApproxDA-TransUNet (journal) architecture diagram, in the style of the
workshop figure (paper/figures/generate_overview.py).

Mirrors ApproxDATransUNet with --backbone lightvit_t --dec_channels 64 (experiments/ApproxDA-TransUNet/Architecture):
R50 root / stage 1 / stage 2 -> LightViT-T stages 1-3 -> ApproxDA block -> conv 3x3 (64 ch) -> three decoder stages with
additive skips (each skip through an ApproxDA block) -> up x2, two 3x3 conv (16 ch) and the segmentation head.

Run:
    python elsarticle-journal/figures/generate_overview_journal.py
"""

import os

import matplotlib.pyplot as plt

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})  # TrueType, not Type 3
import matplotlib.patches as mpatches  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

# ── colour palette (as the workshop figure) ──────────────────────────────────
C_ENC = "#D6E8F7"
C_VIT = "#FDE9C9"
C_ADB = "#E8D5F5"
C_DEC = "#C9EDE8"
C_IO = "#EBEBEB"
C_MLP = "#D5F0E0"
EDGE = "#444444"


def box(ax, cx, cy, w, h, color, label, sublabel="", fs=10, sfs=10):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0.04", linewidth=0.9,
                                edgecolor=EDGE, facecolor=color, zorder=3))
    if sublabel:
        ax.text(cx, cy + 0.10, label, ha="center", va="center", fontsize=fs, fontweight="bold", zorder=4)
        ax.text(cx, cy - 0.14, sublabel, ha="center", va="center", fontsize=sfs, fontweight="bold", color="#333333",
                zorder=4)
    else:
        ax.text(cx, cy, label, ha="center", va="center", fontsize=fs, fontweight="bold", zorder=4)


def arrow(ax, x0, y0, x1, y1, color="#333333", lw=1.3, dashed=False):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="->", color=color, lw=lw, linestyle=(0, (4, 2)) if dashed else "solid"),
                zorder=5)


def elbow(ax, x0, y0, x1, y1, color="#333333", lw=1.3):
    ax.plot([x0, x0, x1], [y0, y1, y1], color=color, lw=lw, zorder=5)
    ax.annotate("", xy=(x1, y1), xytext=(x0 + 0.001, y1), arrowprops=dict(arrowstyle="->", color=color, lw=lw),
                zorder=5)


def add_circle(ax, cx, cy, r=0.13):
    ax.add_patch(plt.Circle((cx, cy), r, color="white", ec=EDGE, lw=0.9, zorder=5))
    ax.text(cx, cy, "⊕", ha="center", va="center", fontsize=9.5, zorder=6)


def italic(ax, x, y, s, ha="left", fs=10.5):
    ax.text(x, y, s, ha=ha, va="center", fontsize=fs, color="#333333", style="italic", zorder=6)


# ── figure layout ────────────────────────────────────────────────────────────
fig_w, fig_h = 12.55, 6.0
fig, ax = plt.subplots(figsize=(11.0, 5.2))  # canvas smaller than the data range: larger text
ax.set_xlim(0.15, fig_w)
ax.set_ylim(0.25, 5.30)
ax.axis("off")

XS = [0.8, 2.55, 4.30, 6.05, 8.15]
Y_ENC, Y_ADB, Y_DEC = 4.55, 3.10, 1.65
BW, BH = 1.35, 0.62  # encoder / ApproxDA / decoder boxes
BW_VIT, BH_VIT = 1.55, 0.72
BW_IO, BH_IO = 1.15, 0.58

# ── encoder: truncated ResNet-50, then LightViT-T ───────────────────────────
box(ax, XS[0], Y_ENC, BW_IO, BH_IO, C_IO, "Input", "1×224²", fs=12, sfs=11)
box(ax, XS[1], Y_ENC, BW, BH, C_ENC, "R50 root", "64ch · 112²", fs=12, sfs=11)
box(ax, XS[2], Y_ENC, BW, BH, C_ENC, "R50 stage 1", "256ch · 56²", fs=12, sfs=11)
box(ax, XS[3], Y_ENC, BW, BH, C_ENC, "R50 stage 2", "512ch · 28²", fs=12, sfs=11)
box(ax, XS[4], Y_ENC, BW_VIT, BH_VIT, C_VIT, "LightViT-T", "128ch · 14²", fs=12, sfs=11)
italic(ax, XS[2], Y_ENC + BH / 2 + 0.17, "truncated ResNet-50 (stride 8)", ha="center", fs=11)
for i in range(4):
    arrow(ax, XS[i] + (BW_IO if i == 0 else BW) / 2, Y_ENC, XS[i + 1] - (BW_VIT if i == 3 else BW) / 2, Y_ENC)

# ── ApproxDA blocks: three skips + bottleneck (after LightViT-T) ─────────────
for xi, cp in zip(XS[1:4], ("4", "16", "32")):
    box(ax, xi, Y_ADB, BW, BH, C_ADB, "ApproxDA", f"C′ = {cp}", fs=12, sfs=11)
box(ax, XS[4], Y_ADB, BW, BH, C_ADB, "ApproxDA", "C′ = 8", fs=12, sfs=11)
italic(ax, XS[4] + 0.09, (Y_ENC - BH_VIT / 2 + Y_ADB + BH / 2) / 2, "bottleneck", fs=11)

skip_mid_y = (Y_ENC - BH / 2 + Y_ADB + BH / 2) / 2
for xi in XS[1:4]:
    arrow(ax, xi, Y_ENC - BH / 2, xi, Y_ADB + BH / 2, lw=1.1)
    italic(ax, xi + 0.09, skip_mid_y, "skip", fs=11)
arrow(ax, XS[4], Y_ENC - BH_VIT / 2, XS[4], Y_ADB + BH / 2, lw=1.3)

# ── slim decoder: 64 channels at every skip stage, additive skips ────────────
box(ax, XS[3], Y_DEC, BW, BH, C_DEC, "Decoder", "64ch · 28²", fs=12, sfs=11)
box(ax, XS[2], Y_DEC, BW, BH, C_DEC, "Decoder", "64ch · 56²", fs=12, sfs=11)
box(ax, XS[1], Y_DEC, BW, BH, C_DEC, "Decoder", "64ch · 112²", fs=12, sfs=11)
box(ax, XS[0], Y_DEC, BW_IO, BH_IO, C_IO, "Output", "K×224²", fs=12, sfs=11)
elbow(ax, XS[4], Y_ADB - BH / 2, XS[3] + BW / 2, Y_DEC)
italic(ax, XS[4] - 0.09, (Y_ADB - BH / 2 + Y_DEC) / 2 + 0.12, "conv 3×3 → 64ch", ha="right", fs=10.5)
for i in [3, 2, 1]:
    arrow(ax, XS[i] - BW / 2, Y_DEC, XS[i - 1] + (BW_IO if i == 1 else BW) / 2, Y_DEC)
italic(ax, XS[0] + 0.1, Y_DEC - BH_IO / 2 - 0.19, "up ×2, 2 conv (16ch), head", ha="center", fs=10)
for xi in XS[1:4]:
    arrow(ax, xi, Y_ADB - BH / 2, xi, Y_DEC + BH / 2, lw=1.3)
    italic(ax, xi + 0.09, (Y_ADB - BH / 2 + Y_DEC + BH / 2) / 2, "add", fs=11)

# ════════════════════════════════════════════════════════════════════════════
# INSET — LightViT-T (stages 1–3, ImageNet-1k weights)
# ════════════════════════════════════════════════════════════════════════════
VIX, VIW, VIH, VIY = 9.35, 3.10, 4.26, 0.58
VIXC = VIX + VIW / 2
VBW, BH_S = 2.56, 0.48
ax.add_patch(FancyBboxPatch((VIX, VIY), VIW, VIH, boxstyle="round,pad=0.07", linewidth=0.9, edgecolor="#999999",
                            facecolor="white", linestyle="dashed", zorder=2))
ax.text(VIXC, VIY + VIH - 0.14, "LightViT-T (stages 1–3)", ha="center", va="center", fontsize=13,
        fontweight="bold", color="#7A4800", zorder=6)

y_in, y_s1, y_s2, y_s3, y_fu, y_add, y_out = 4.27, 3.62, 2.92, 2.22, 1.58, 1.05, 0.70
box(ax, VIXC, y_in, VBW, 0.42, C_IO, "1×1 conv + 8 tokens", "28² × 64", fs=11, sfs=10.5)
box(ax, VIXC, y_s1, VBW, BH_S, C_VIT, "Stage 1 · 2 layers", "28² × 64", fs=11.5, sfs=10.5)
box(ax, VIXC, y_s2, VBW, BH_S, C_VIT, "Stage 2 · 6 layers", "merge ↓2 · 14² × 128", fs=11.5, sfs=10.5)
box(ax, VIXC, y_s3, VBW, BH_S, C_VIT, "Stage 3 · 6 layers", "merge ↓2 · 7² × 256", fs=11.5, sfs=10.5)
box(ax, VIXC, y_fu, VBW, BH_S, C_MLP, "1×1 conv + up ×2", "14² × 128", fs=11.5, sfs=10.5)
add_circle(ax, VIXC, y_add)
ax.text(VIXC, y_out, "Output (14² × 128)", ha="center", va="center", fontsize=11, color="#333333", zorder=6)
for y0, y1 in [(y_in - 0.19, y_s1 + BH_S / 2), (y_s1 - BH_S / 2, y_s2 + BH_S / 2), (y_s2 - BH_S / 2, y_s3 + BH_S / 2),
               (y_s3 - BH_S / 2, y_fu + BH_S / 2), (y_fu - BH_S / 2, y_add + 0.13), (y_add - 0.13, y_out + 0.10)]:
    arrow(ax, VIXC, y0, VIXC, y1, lw=1.0)
# stage-2 output joins the sum (top-down fusion)
sk_x = VIXC + VBW / 2 + 0.20
ax.plot([VIXC + VBW / 2, sk_x, sk_x], [y_s2, y_s2, y_add], color=EDGE, lw=1.1, zorder=5)
ax.annotate("", xy=(VIXC + 0.14, y_add), xytext=(sk_x + 0.001, y_add),
            arrowprops=dict(arrowstyle="->", color=EDGE, lw=1.1), zorder=5)
arrow(ax, XS[4] + BW_VIT / 2, Y_ENC, VIX + 0.10, Y_ENC, color="#AAAAAA", lw=1.1, dashed=True)

# ── legend ────────────────────────────────────────────────────────────────────
ax.legend(
    handles=[
        mpatches.Patch(facecolor=C_ENC, edgecolor=EDGE, label="CNN encoder"),
        mpatches.Patch(facecolor=C_VIT, edgecolor=EDGE, label="LightViT-T"),
        mpatches.Patch(facecolor=C_ADB, edgecolor=EDGE, label="ApproxDA block"),
        mpatches.Patch(facecolor=C_DEC, edgecolor=EDGE, label="Slim decoder"),
    ],
    loc="lower left", fontsize=12, framealpha=0.90, bbox_to_anchor=(0.0, 0.0), ncol=4, columnspacing=1.0,
    handlelength=2.0, handleheight=1.4,
)

out_dir = os.path.dirname(os.path.abspath(__file__))
plt.tight_layout(pad=0.2)
plt.savefig(os.path.join(out_dir, "approxda_overview_journal.pdf"), bbox_inches="tight", dpi=300)
print("Saved approxda_overview_journal.pdf")
