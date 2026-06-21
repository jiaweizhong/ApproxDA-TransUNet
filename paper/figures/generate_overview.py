"""
Generate approxda_overview.pdf  —  ApproxDA-TransUNet architecture diagram.

Run:
    python paper/figures/generate_overview.py

Outputs:
    paper/figures/approxda_overview.pdf
    paper/figures/approxda_overview.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
import os

# ── colour palette ───────────────────────────────────────────────────────────
C_ENC  = "#D6E8F7"
C_VIT  = "#FDE9C9"
C_ADB  = "#E8D5F5"
C_DEC  = "#C9EDE8"
C_IO   = "#EBEBEB"
C_MLP  = "#D5F0E0"
EDGE   = "#444444"

# ── helpers ──────────────────────────────────────────────────────────────────
def box(ax, cx, cy, w, h, color, label, sublabel="", fs=10, sfs=9):
    rect = FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle="round,pad=0.04",
        linewidth=0.9, edgecolor=EDGE, facecolor=color, zorder=3
    )
    ax.add_patch(rect)
    if sublabel:
        ax.text(cx, cy + 0.10, label, ha="center", va="center",
                fontsize=fs, fontweight="bold", zorder=4)
        ax.text(cx, cy - 0.14, sublabel, ha="center", va="center",
                fontsize=sfs, color="#333333", zorder=4)
    else:
        ax.text(cx, cy, label, ha="center", va="center",
                fontsize=fs, fontweight="bold", zorder=4)

def arrow(ax, x0, y0, x1, y1, color="#333333", lw=1.3, dashed=False, rad=0.0):
    ls = (0, (4, 2)) if dashed else "solid"
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="->", color=color, lw=lw,
                                linestyle=ls,
                                connectionstyle=f"arc3,rad={rad}"), zorder=5)

def elbow(ax, x0, y0, x1, y1, color="#333333", lw=1.3):
    ax.plot([x0, x0, x1], [y0, y1, y1], color=color, lw=lw, zorder=5)
    ax.annotate("", xy=(x1, y1), xytext=(x0 + 0.001, y1),
                arrowprops=dict(arrowstyle="->", color=color, lw=lw), zorder=5)

def add_circle(ax, cx, cy, r=0.13):
    circ = plt.Circle((cx, cy), r, color='white', ec=EDGE, lw=0.9, zorder=5)
    ax.add_patch(circ)
    ax.text(cx, cy, '⊕', ha='center', va='center', fontsize=9.5, zorder=6)


# ── figure layout ────────────────────────────────────────────────────────────
fig_w, fig_h = 12.5, 6.0
fig, ax = plt.subplots(figsize=(fig_w, fig_h))
ax.set_xlim(0, fig_w)
ax.set_ylim(0.40, 5.40)
ax.axis("off")

XS = [0.8, 2.55, 4.30, 6.05, 8.15]
Y_ENC = 4.55
Y_ADB = 3.10
Y_DEC = 1.65

BW_ENC = 1.35;  BH_ENC = 0.62
BW_VIT = 1.55;  BH_VIT = 0.72
BW_ADB = 1.35;  BH_ADB = 0.62
BW_DEC = 1.35;  BH_DEC = 0.62
BW_IO  = 1.15;  BH_IO  = 0.58

# ── encoder ──────────────────────────────────────────────────────────────────
box(ax, XS[0], Y_ENC, BW_IO,  BH_IO,  C_IO,  "Input",    "3×224²",      fs=11, sfs=11)
box(ax, XS[1], Y_ENC, BW_ENC, BH_ENC, C_ENC, "CNN",      "64ch · 112²", fs=11, sfs=11)
box(ax, XS[2], Y_ENC, BW_ENC, BH_ENC, C_ENC, "CNN",      "256ch · 56²", fs=11, sfs=11)
box(ax, XS[3], Y_ENC, BW_ENC, BH_ENC, C_ENC, "CNN",      "512ch · 28²", fs=11, sfs=11)
box(ax, XS[4], Y_ENC, BW_VIT, BH_VIT, C_VIT, "ViT-B/16", "12L · 14²",  fs=11, sfs=11)

for i in range(4):
    arrow(ax,
          XS[i] + (BW_IO if i == 0 else BW_ENC)/2, Y_ENC,
          XS[i+1] - (BW_VIT if i == 3 else BW_ENC)/2, Y_ENC)

# ── ApproxDABlocks ────────────────────────────────────────────────────────────
for xi in XS[1:]:
    box(ax, xi, Y_ADB, BW_ADB, BH_ADB, C_ADB, "ApproxDABlock", fs=10.5)

# skip connections — solid black arrows
skip_mid_y   = (Y_ENC - BH_ENC/2 + Y_ADB + BH_ADB/2) / 2
btlnk_mid_y  = (Y_ENC - BH_VIT/2 + Y_ADB + BH_ADB/2) / 2

for xi in XS[1:4]:
    arrow(ax, xi, Y_ENC - BH_ENC/2, xi, Y_ADB + BH_ADB/2, lw=1.1)
    ax.text(xi + 0.09, skip_mid_y, "skip", ha="left", va="center",
            fontsize=10.5, color="#333333", style="italic", zorder=6)

arrow(ax, XS[4], Y_ENC - BH_VIT/2, XS[4], Y_ADB + BH_ADB/2, lw=1.3)
ax.text(XS[4] + 0.11, btlnk_mid_y, "bottleneck", ha="left", va="center",
        fontsize=10.5, zorder=6)

# ── decoder ───────────────────────────────────────────────────────────────────
box(ax, XS[3], Y_DEC, BW_DEC, BH_DEC, C_DEC, "Decoder", "512ch · 28²", fs=11, sfs=11)
box(ax, XS[2], Y_DEC, BW_DEC, BH_DEC, C_DEC, "Decoder", "256ch · 56²", fs=11, sfs=11)
box(ax, XS[1], Y_DEC, BW_DEC, BH_DEC, C_DEC, "Decoder", "64ch · 112²", fs=11, sfs=11)
box(ax, XS[0], Y_DEC, BW_IO,  BH_IO,  C_IO,  "Output",  "224²",        fs=11, sfs=11)

# ViT ApproxDABlock → right-edge midpoint of Up-1
elbow(ax, XS[4], Y_ADB - BH_ADB/2, XS[3] + BW_DEC/2, Y_DEC)

for i in [3, 2, 1]:
    arrow(ax, XS[i] - BW_DEC/2, Y_DEC,
          XS[i-1] + (BW_IO if i == 1 else BW_DEC)/2, Y_DEC)

for xi in XS[1:4]:
    arrow(ax, xi, Y_ADB - BH_ADB/2, xi, Y_DEC + BH_DEC/2, lw=1.3)


# ════════════════════════════════════════════════════════════════════════════
# INSET — ViT-B/16 Transformer Block
# ════════════════════════════════════════════════════════════════════════════
VIX  = 9.35
VIW  = 2.80
VIH  = 4.26
VIY  = 0.58
VIXC = VIX + VIW / 2   # 10.60
VBW  = 2.00             # content boxes; res bar extends ~0.45 beyond right edge
BH_LN  = 0.33
BH_BLK = 0.46

ax.add_patch(FancyBboxPatch(
    (VIX, VIY), VIW, VIH,
    boxstyle="round,pad=0.07", linewidth=0.9,
    edgecolor="#999999", facecolor="white", linestyle="dashed", zorder=2
))
ax.text(VIXC, VIY + VIH - 0.14,
        "ViT-B/16 Transformer Block ×12",
        ha="center", va="center", fontsize=10,
        fontweight="bold", color="#7A4800", zorder=6)

# element y-centres — well-spaced so inter-block arrows are clearly visible
vit_xi_y   = 4.34
vit_ln1_y  = 3.82
vit_msa_y  = 3.28
vit_add1_y = 2.74
vit_ln2_y  = 2.24
vit_mlp_y  = 1.70
vit_add2_y = 1.16
vit_xo_y   = 0.73

box(ax, VIXC, vit_xi_y,  VBW, 0.38,   C_IO,  "Input tokens",
    "196 × 768",                fs=9, sfs=8)
box(ax, VIXC, vit_ln1_y, VBW, BH_LN,  C_IO,  "Layer Norm",         fs=9)
box(ax, VIXC, vit_msa_y, VBW, BH_BLK, C_VIT,
    "Multi-Head Self-Attention", "12 heads · d_h = 64", fs=9, sfs=8)
add_circle(ax, VIXC, vit_add1_y)
box(ax, VIXC, vit_ln2_y, VBW, BH_LN,  C_IO,  "Layer Norm",         fs=9)
box(ax, VIXC, vit_mlp_y, VBW, BH_BLK, C_MLP,
    "MLP / FFN",                 "768 → 3072 → 768",   fs=9, sfs=8)
add_circle(ax, VIXC, vit_add2_y)
ax.text(VIXC, vit_xo_y, "Output tokens (196 × 768)",
        ha="center", va="center", fontsize=8.5, color="#333333", zorder=6)

# vertical flow arrows
for y0, y1 in [
    (vit_xi_y   - 0.19,      vit_ln1_y  + BH_LN/2),
    (vit_ln1_y  - BH_LN/2,   vit_msa_y  + BH_BLK/2),
    (vit_msa_y  - BH_BLK/2,  vit_add1_y + 0.13),
    (vit_add1_y - 0.13,      vit_ln2_y  + BH_LN/2),
    (vit_ln2_y  - BH_LN/2,   vit_mlp_y  + BH_BLK/2),
    (vit_mlp_y  - BH_BLK/2,  vit_add2_y + 0.13),
    (vit_add2_y - 0.13,      vit_xo_y   + 0.13),
]:
    arrow(ax, VIXC, y0, VIXC, y1, lw=1.0)

# residual bar — black, right of boxes
res_x = VIXC + VBW/2 + 0.28
ax.plot([VIXC + VBW/2, res_x, res_x],
        [vit_xi_y, vit_xi_y, vit_add2_y],
        color=EDGE, lw=1.1, zorder=5)
ax.annotate("", xy=(VIXC + 0.14, vit_add1_y),
            xytext=(res_x, vit_add1_y),
            arrowprops=dict(arrowstyle="->", color=EDGE, lw=1.1), zorder=5)
ax.annotate("", xy=(VIXC + 0.14, vit_add2_y),
            xytext=(res_x + 0.001, vit_add2_y),
            arrowprops=dict(arrowstyle="->", color=EDGE, lw=1.1), zorder=5)
ax.text(res_x + 0.12, (vit_xi_y + vit_add2_y) / 2,
        "res.", ha="center", va="center",
        fontsize=8.5, color=EDGE, style="italic", rotation=90, zorder=6)


# ── connector: ViT main box → ViT inset ──────────────────────────────────────
ax.annotate("", xy=(VIX + 0.10, VIY + VIH / 2),
            xytext=(XS[4] + BW_VIT/2 + 0.05, Y_ENC),
            arrowprops=dict(arrowstyle="->", color="#AAAAAA", lw=0.9,
                            linestyle=(0, (3, 2)),
                            connectionstyle="arc3,rad=-0.2"), zorder=5)
ax.text(9.10, 3.72, "detail", ha="center", fontsize=7.5, color="#AAAAAA", zorder=6)


# ── legend ────────────────────────────────────────────────────────────────────
ax.legend(handles=[
    mpatches.Patch(facecolor=C_ENC, edgecolor=EDGE, label="CNN encoder"),
    mpatches.Patch(facecolor=C_VIT, edgecolor=EDGE, label="ViT / MHSA"),
    mpatches.Patch(facecolor=C_ADB, edgecolor=EDGE, label="ApproxDABlock"),
    mpatches.Patch(facecolor=C_DEC, edgecolor=EDGE, label="Decoder"),
    mpatches.Patch(facecolor=C_MLP, edgecolor=EDGE, label="MLP / FFN"),
], loc="lower left", fontsize=13, framealpha=0.90,
   bbox_to_anchor=(0.0, 0.010),
   ncol=5, columnspacing=1.2, handlelength=2.0, handleheight=1.4)


# ── save ──────────────────────────────────────────────────────────────────────
out_dir = os.path.dirname(os.path.abspath(__file__))
plt.tight_layout(pad=0.2)
plt.savefig(os.path.join(out_dir, "approxda_overview.pdf"),
            bbox_inches="tight", dpi=300)
plt.savefig(os.path.join(out_dir, "approxda_overview.png"),
            bbox_inches="tight", dpi=200)
print("Saved approxda_overview.pdf and .png")
plt.show()
