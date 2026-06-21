"""
Generate approxda_overview.pdf  —  ApproxDA-TransUNet architecture diagram.

Run:
    python paper/figures/generate_overview.py

Outputs:
    paper/figures/approxda_overview.pdf   (replaces current Fig 1)
    paper/figures/approxda_overview.png   (preview)

Requires: matplotlib, numpy  (pip install matplotlib)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import numpy as np
import os

# ── colour palette ───────────────────────────────────────────────────────────
C_ENC  = "#D6E8F7"   # blue-ish  — CNN encoder
C_VIT  = "#FDE9C9"   # orange    — ViT bottleneck
C_ADB  = "#E8D5F5"   # violet    — ApproxDABlock
C_DEC  = "#C9EDE8"   # teal      — decoder
C_IO   = "#EBEBEB"   # gray      — input / output
C_PAM  = "#C5D9F1"   # blue      — PAM branch (inset)
C_CAM  = "#C9EDD0"   # green     — CAM branch (inset)
C_GATE = "#FFF6C2"   # yellow    — gate (inset)
EDGE   = "#444444"

# ── helpers ──────────────────────────────────────────────────────────────────
def box(ax, cx, cy, w, h, color, label, sublabel="", fs=8, sfs=6.5):
    """Draw a rounded box centred at (cx, cy)."""
    rect = FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle="round,pad=0.04",
        linewidth=0.8, edgecolor=EDGE, facecolor=color, zorder=3
    )
    ax.add_patch(rect)
    if sublabel:
        ax.text(cx, cy + 0.07, label, ha="center", va="center",
                fontsize=fs, fontweight="bold", zorder=4)
        ax.text(cx, cy - 0.13, sublabel, ha="center", va="center",
                fontsize=sfs, color="#555555", zorder=4)
    else:
        ax.text(cx, cy, label, ha="center", va="center",
                fontsize=fs, fontweight="bold", zorder=4)
    return (cx - w/2, cy - h/2, cx + w/2, cy + h/2)  # bbox

def arrow(ax, x0, y0, x1, y1, color="#333333", lw=1.2,
          dashed=False, style="->", rad=0.0):
    ls = (0, (4, 2)) if dashed else "solid"
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(
                    arrowstyle=style, color=color,
                    lw=lw, linestyle=ls,
                    connectionstyle=f"arc3,rad={rad}"
                ), zorder=5)

def elbow(ax, x0, y0, x1, y1, color="#333333", lw=1.2):
    """L-shaped connector: (x0,y0) → down to y1 → right/left to (x1,y1)."""
    ax.plot([x0, x0, x1], [y0, y1, y1], color=color, lw=lw, zorder=5)
    ax.annotate("", xy=(x1, y1), xytext=(x0 + 0.001, y1),
                arrowprops=dict(arrowstyle="->", color=color, lw=lw), zorder=5)


# ── figure layout ────────────────────────────────────────────────────────────
fig_w, fig_h = 14.0, 4.2
fig, ax = plt.subplots(figsize=(fig_w, fig_h))
ax.set_xlim(0, fig_w)
ax.set_ylim(-0.5, fig_h - 0.5)
ax.axis("off")

# Column x-positions (left part of figure, right part = inset)
XS = [0.8, 2.55, 4.30, 6.05, 8.15]  # Input, E0, E1, E2, ViT
Y_ENC = 3.2   # encoder row y
Y_ADB = 2.0   # ApproxDA row y
Y_DEC = 0.8   # decoder row y

BW_ENC = 1.35   # box width  — encoder
BH_ENC = 0.65   # box height — encoder
BW_VIT = 1.5
BH_VIT = 0.80
BW_ADB = 1.35
BH_ADB = 0.50
BW_DEC = 1.35
BH_DEC = 0.65
BW_IO  = 1.10
BH_IO  = 0.60

# ── encoder ──────────────────────────────────────────────────────────────────
box(ax, XS[0], Y_ENC, BW_IO, BH_IO, C_IO,
    "Input", "3×224²", fs=7.5, sfs=6)
box(ax, XS[1], Y_ENC, BW_ENC, BH_ENC, C_ENC,
    "CNN", "64ch · 112²", fs=7.5, sfs=6)
box(ax, XS[2], Y_ENC, BW_ENC, BH_ENC, C_ENC,
    "CNN", "256ch · 56²", fs=7.5, sfs=6)
box(ax, XS[3], Y_ENC, BW_ENC, BH_ENC, C_ENC,
    "CNN", "512ch · 28²", fs=7.5, sfs=6)
box(ax, XS[4], Y_ENC, BW_VIT, BH_VIT, C_VIT,
    "ViT-B/16", "12 layers · 14²", fs=7.5, sfs=6)

# encoder flow arrows
for i in range(4):
    arrow(ax, XS[i] + BW_IO/2 if i == 0 else XS[i] + BW_ENC/2,
          Y_ENC,
          XS[i+1] - (BW_VIT/2 if i == 3 else BW_ENC/2),
          Y_ENC)

# ── ApproxDABlocks ────────────────────────────────────────────────────────────
# Indices: ab0=E0 col, ab1=E1, ab2=E2, ab3=ViT (bottleneck)
for xi in XS[1:]:
    box(ax, xi, Y_ADB, BW_ADB, BH_ADB, C_ADB,
        "ApproxDABlock", fs=6.5)

# skip: encoder → ApproxDA (dashed gray)
for xi in XS[1:4]:
    arrow(ax, xi, Y_ENC - BH_ENC/2,
          xi, Y_ADB + BH_ADB/2,
          color="#888888", lw=0.9, dashed=True)

# bottleneck: ViT → ApproxDA (solid)
arrow(ax, XS[4], Y_ENC - BH_VIT/2,
      XS[4], Y_ADB + BH_ADB/2, lw=1.2)

# ── decoder ───────────────────────────────────────────────────────────────────
# Decoder columns aligned under E2, E1, E0 (same x)
box(ax, XS[3], Y_DEC, BW_DEC, BH_DEC, C_DEC,
    "Up-1", "512ch · 28²", fs=7.5, sfs=6)
box(ax, XS[2], Y_DEC, BW_DEC, BH_DEC, C_DEC,
    "Up-2", "256ch · 56²", fs=7.5, sfs=6)
box(ax, XS[1], Y_DEC, BW_DEC, BH_DEC, C_DEC,
    "Up-3", "64ch · 112²", fs=7.5, sfs=6)
box(ax, XS[0], Y_DEC, BW_IO, BH_IO, C_IO,
    "Output", "224²", fs=7.5, sfs=6)

# bottleneck AB3 → Up-1: elbow (down then left)
elbow(ax, XS[4], Y_ADB - BH_ADB/2,
      XS[3] + BW_DEC/2, Y_DEC + BH_DEC/2)

# decoder flow: right → left
for i in [3, 2, 1]:
    x_from = XS[i] - BW_DEC/2
    x_to   = XS[i-1] + (BW_IO/2 if i == 1 else BW_DEC/2)
    arrow(ax, x_from, Y_DEC, x_to, Y_DEC)

# skip AB → decoder (same-column vertical arrows)
for xi in XS[1:4]:
    arrow(ax, xi, Y_ADB - BH_ADB/2,
          xi, Y_DEC + BH_DEC/2, lw=1.2)

# ── skip / bottleneck labels ──────────────────────────────────────────────────
for xi in XS[1:4]:
    ax.text(xi + 0.08, (Y_ENC - BH_ENC/2 + Y_ADB + BH_ADB/2) / 2,
            "skip", ha="left", va="center", fontsize=5.5,
            color="#888888", style="italic", zorder=6)
ax.text(XS[4] + 0.10, (Y_ENC - BH_VIT/2 + Y_ADB + BH_ADB/2) / 2,
        "btlnk", ha="left", va="center", fontsize=5.5, zorder=6)

# ── INSET: ApproxDABlock detail ───────────────────────────────────────────────
IX = 9.55   # inset left edge
IW = 4.10   # inset width
IH = 3.80   # inset height
IY = 0.15   # inset bottom

inset_rect = FancyBboxPatch(
    (IX, IY), IW, IH,
    boxstyle="round,pad=0.07",
    linewidth=0.8, edgecolor="#999999",
    facecolor="white", linestyle="dashed", zorder=2
)
ax.add_patch(inset_rect)
ax.text(IX + IW/2, IY + IH - 0.22,
        "ApproxDABlock", ha="center", va="center",
        fontsize=8, fontweight="bold", color="#6A0EAC", zorder=6)

# positions inside inset
IXC = IX + IW/2  # centre x of inset
xi_y  = IY + IH - 0.65
pam_y = IY + IH - 1.45
cam_y = IY + IH - 1.45
gt_y  = IY + IH - 2.30
xo_y  = IY + 0.40

BW_IB = 1.45  # inset branch box width

box(ax, IXC, xi_y, 0.60, 0.38, C_IO, "x", fs=9)

box(ax, IXC - 0.80, pam_y, BW_IB, 0.62, C_PAM,
    "LR-Windowed PAM", "win M, rank r", fs=6.5, sfs=5.8)
box(ax, IXC + 0.80, cam_y, BW_IB, 0.62, C_CAM,
    "Grouped CAM", "groups G", fs=6.5, sfs=5.8)
box(ax, IXC, gt_y, 1.80, 0.50, C_GATE,
    "Gate  g", "g·P̂ + (1−g)·Ĉ", fs=6.5, sfs=5.8)
box(ax, IXC, xo_y, 1.80, 0.42, C_IO,
    "Wf(…) + x", fs=6.5)

# inset arrows
arrow(ax, IXC, xi_y - 0.19,  IXC - 0.80, pam_y + 0.31, lw=0.9)
arrow(ax, IXC, xi_y - 0.19,  IXC + 0.80, cam_y + 0.31, lw=0.9)
arrow(ax, IXC - 0.80, pam_y - 0.31, IXC - 0.35, gt_y + 0.25, lw=0.9)
arrow(ax, IXC + 0.80, cam_y - 0.31, IXC + 0.35, gt_y + 0.25, lw=0.9)
arrow(ax, IXC, gt_y - 0.25, IXC, xo_y + 0.21, lw=0.9)
# residual bypass
ax.annotate("", xy=(IXC - 0.65, xo_y + 0.10),
            xytext=(IXC - 0.65, xi_y - 0.10),
            arrowprops=dict(
                arrowstyle="->", color="#AAAAAA", lw=0.8,
                connectionstyle="arc3,rad=-0.35",
                linestyle=(0, (3, 2))
            ), zorder=5)
ax.text(IXC - 1.05, (xi_y + xo_y) / 2, "res.",
        ha="center", va="center", fontsize=5.5, color="#AAAAAA", zorder=6)

# ── connector: main → inset ───────────────────────────────────────────────────
ax.annotate("", xy=(IX + 0.08, IY + IH/2),
            xytext=(XS[4] + BW_VIT/2 + 0.05, Y_ADB),
            arrowprops=dict(
                arrowstyle="->", color="#AAAAAA", lw=0.8,
                linestyle=(0, (3, 2)),
                connectionstyle="arc3,rad=-0.15"
            ), zorder=5)
ax.text((XS[4] + BW_VIT/2 + IX) / 2, Y_ADB + 0.25,
        "detail", ha="center", fontsize=5.5, color="#AAAAAA", zorder=6)

# ── legend ────────────────────────────────────────────────────────────────────
legend_items = [
    mpatches.Patch(facecolor=C_ENC, edgecolor=EDGE, label="CNN encoder stage"),
    mpatches.Patch(facecolor=C_VIT, edgecolor=EDGE, label="ViT bottleneck"),
    mpatches.Patch(facecolor=C_ADB, edgecolor=EDGE, label="ApproxDABlock"),
    mpatches.Patch(facecolor=C_DEC, edgecolor=EDGE, label="Decoder (Up)"),
]
ax.legend(handles=legend_items, loc="lower left",
          fontsize=6, framealpha=0.85,
          bbox_to_anchor=(0.0, -0.02),
          ncol=4, columnspacing=0.8, handlelength=1.0, handleheight=0.8)

# ── save ──────────────────────────────────────────────────────────────────────
out_dir = os.path.dirname(os.path.abspath(__file__))
plt.tight_layout(pad=0.1)
plt.savefig(os.path.join(out_dir, "approxda_overview.pdf"),
            bbox_inches="tight", dpi=300)
plt.savefig(os.path.join(out_dir, "approxda_overview.png"),
            bbox_inches="tight", dpi=200)
print("Saved approxda_overview.pdf and .png")
plt.show()
