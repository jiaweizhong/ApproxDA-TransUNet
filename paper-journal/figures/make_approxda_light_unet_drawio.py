"""Lightweight ApproxDA (R50 to 1/8 + LightViT-T + C = 64 slim decoder, ApproxDA blocks on the bottleneck and the
three skips) trained with a native-resolution point loss (PointRend point selection, Kirillov et al., CVPR 2020,
without the point head; training only), as a multi-page draw.io file in U-Net layout:

    (a) overview: U-Net on its side (encoder row, ApproxDA row, decoder row, heads) with the LightViT-T inset
    (b) ApproxDA block (PAM + CAM branches, fixed gate g = 1/2)   (c) slim decoder block
    (d) point supervision loss (training only)

Boxes carry module names only; tensor sizes (resolution x channels) are written on the connections. Writes
approxda_light_unet.drawio (one page per panel; draw.io / diagrams.net / VS Code Draw.io Integration) and one PNG
preview per page, both from the same coordinates. Shapes and channels traced from experiments/ApproxDA-TransUNet
(lightvit_t, dec_channels 64, gate_mode fixed, rank 32, groups 8, window 28, point_loss with ptl_balanced and
ptl_class_share 0.05; K classes).

    python make_approxda_light_unet_drawio.py
"""
import os
from xml.sax.saxutils import escape, quoteattr

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
# Bottleneck order: LightViT-T -> ApproxDA block (--bn_order post, the final model; pre was lower in every seed)
NAME = "approxda_light_unet"

FS_BOX, FS_LABEL, FS_NOTE, FS_TAG, FS_SUM = 15, 13, 13, 12, 18   # draw.io font sizes (px)
PT = 0.69                                                         # draw.io px -> preview pt at 100 px / inch

GREY = ("#f5f5f5", "#666666")
BLUE = ("#dae8fc", "#6c8ebf")      # R50
PURPLE = ("#e1d5e7", "#9673a6")    # LightViT-T
YELLOW = ("#fff2cc", "#d6b656")    # ApproxDA block
ORANGE = ("#ffe6cc", "#d79b00")    # attention operators inside the block
GREEN = ("#d5e8d4", "#82b366")     # decoder
RED = ("#f8cecc", "#b85450")       # point loss (training only)
SUM = ("#ffffff", "#333333")
NOTE = ("#ffffff", "#999999")
V2 = "#ae4132"
EDGE = "#333333"
SKIP = "#3d5a99"


class Page:
    def __init__(self, name, width, height):
        self.name, self.width, self.height = name, width, height
        self.nodes, self.edges, self.labels = {}, [], []

    def box(self, i, cx, cy, w, h, text, colour, kind="box", v2=False):
        """Node centred at (cx, cy). kind: box | dbox (dashed box) | sum (circle) | note | text | tag | group."""
        self.nodes[i] = dict(x=cx - w / 2, y=cy - h / 2, w=w, h=h, text=text, colour=colour, kind=kind, v2=v2)
        return i

    def tag(self, of, letter):
        n = self.nodes[of]
        self.box(f"tag_{of}", n["x"] + n["w"] - 2, n["y"] - 1, 30, 20, f"({letter})", NOTE, "tag")

    def anchor(self, i, side, frac=0.5):
        n = self.nodes[i]
        return {"left": (n["x"], n["y"] + frac * n["h"]), "right": (n["x"] + n["w"], n["y"] + frac * n["h"]),
                "top": (n["x"] + frac * n["w"], n["y"]), "bottom": (n["x"] + frac * n["w"], n["y"] + n["h"])}[side], \
            {"left": (0, frac), "right": (1, frac), "top": (frac, 0), "bottom": (frac, 1)}[side]

    def edge(self, a, a_side, b, b_side, via=(), label="", seg=-1, side="above", dashed=False, colour=EDGE, af=0.5, bf=0.5,
             lpos=0.5, dy=None):
        """Arrow a -> b through waypoints; label next to segment `seg` (above / below / left / right; dy overrides)."""
        self.edges.append((a, a_side, af, b, b_side, bf, list(via), dashed, colour))
        if label:
            pts = [self.anchor(a, a_side, af)[0]] + list(via) + [self.anchor(b, b_side, bf)[0]]
            k = seg % (len(pts) - 1)
            (x0, y0), (x1, y1) = pts[k], pts[k + 1]
            mx, my = x0 + lpos * (x1 - x0), y0 + lpos * (y1 - y0)
            ox, oy = {"above": (0, -12), "below": (0, 12), "left": (-7, 0), "right": (7, 0)}[side]
            if dy is not None:
                oy = dy
            ha = {"left": "right", "right": "left"}.get(side, "center")
            self.labels.append((mx + ox, my + oy, label, ha, colour if colour != EDGE else "#444444"))


def html(text):
    out = []
    for line in text.split("\n"):
        line = escape(line)
        if line.startswith("**") and "**" in line[2:]:
            head, rest = line[2:].split("**", 1)
            line = f"<b>{head}</b>{rest}"
        out.append(line)
    return "<br>".join(out)


def drawio_page(p, pid):
    cells = ['<mxCell id="0"/>', '<mxCell id="1" parent="0"/>']
    for i, n in p.nodes.items():
        fill, stroke = n["colour"]
        common = f"html=1;whiteSpace=wrap;fillColor={fill};strokeColor={V2 if n['v2'] else stroke};"
        common += "strokeWidth=3;" if n["v2"] else ""
        style = {"box": f"rounded=1;arcSize=14;fontSize={FS_BOX};" + common,
                 "dbox": f"rounded=1;arcSize=14;dashed=1;fontSize={FS_BOX};" + common,
                 "sum": f"ellipse;fontStyle=1;fontSize={FS_SUM};" + common,
                 "note": f"rounded=1;arcSize=6;dashed=1;align=left;spacingLeft=10;fontSize={FS_NOTE};" + common,
                 "text": f"text;html=1;align=left;verticalAlign=middle;fontSize={FS_NOTE};whiteSpace=wrap;",
                 "tag": f"rounded=1;arcSize=30;fontSize={FS_TAG};fontStyle=1;fontColor=#555555;" + common,
                 "group": f"rounded=1;arcSize=3;dashed=1;verticalAlign=top;align=left;spacingLeft=10;spacingTop=2;"
                          f"fontSize={FS_BOX};fontStyle=1;fontColor=#7a5a00;" + common}[n["kind"]]
        cells.append(f'<mxCell id="{pid}_{i}" value={quoteattr(html(n["text"]))} style="{style}" vertex="1" parent="1">'
                     f'<mxGeometry x="{n["x"]:.1f}" y="{n["y"]:.1f}" width="{n["w"]}" height="{n["h"]}" as="geometry"/></mxCell>')
    for k, (a, asd, af, b, bsd, bf, via, dashed, colour) in enumerate(p.edges):
        (_, (ex, ey)), (_, (nx, ny)) = p.anchor(a, asd, af), p.anchor(b, bsd, bf)
        style = (f"edgeStyle=none;html=1;rounded=0;endArrow=block;endFill=1;endSize=5;strokeColor={colour};strokeWidth=1.5;"
                 f"exitX={ex};exitY={ey};exitDx=0;exitDy=0;entryX={nx};entryY={ny};entryDx=0;entryDy=0;"
                 + ("dashed=1;" if dashed else ""))
        geo = '<mxGeometry relative="1" as="geometry">'
        if via:
            geo += '<Array as="points">' + "".join(f'<mxPoint x="{x:.1f}" y="{y:.1f}"/>' for x, y in via) + "</Array>"
        cells.append(f'<mxCell id="{pid}_e{k}" style="{style}" edge="1" parent="1" source="{pid}_{a}" target="{pid}_{b}">'
                     f"{geo}</mxGeometry></mxCell>")
    for k, (x, y, text, ha, colour) in enumerate(p.labels):
        w, h = 10 + 0.6 * FS_LABEL * len(text.replace("**", "")), FS_LABEL + 6
        left = x if ha == "left" else (x - w if ha == "right" else x - w / 2)
        align = {"left": "left", "right": "right"}.get(ha, "center")
        cells.append(f'<mxCell id="{pid}_l{k}" value={quoteattr(html(text))} style="text;html=1;align={align};verticalAlign=middle;'
                     f'fontSize={FS_LABEL};fontColor={colour};labelBackgroundColor=#ffffff;" vertex="1" parent="1">'
                     f'<mxGeometry x="{left:.1f}" y="{y - h / 2:.1f}" width="{w:.0f}" height="{h}" as="geometry"/></mxCell>')
    return (f'<diagram name={quoteattr(p.name)} id="{pid}"><mxGraphModel dx="{p.width}" dy="{p.height}" grid="1" gridSize="10" '
            f'guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="{p.width}" '
            f'pageHeight="{p.height}" math="0" shadow="0"><root>' + "".join(cells) + "</root></mxGraphModel></diagram>")


def preview(p, path):
    fig = plt.figure(figsize=(p.width / 100, p.height / 100), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, p.width); ax.set_ylim(p.height, 0); ax.axis("off")
    for (a, asd, af, b, bsd, bf, via, dashed, colour) in p.edges:
        pts = [p.anchor(a, asd, af)[0]] + via + [p.anchor(b, bsd, bf)[0]]
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=colour, lw=1.2, ls=(0, (4, 3)) if dashed else "-", zorder=2)
        ax.annotate("", xy=pts[-1], xytext=pts[-2], arrowprops=dict(arrowstyle="-|>,head_length=0.45,head_width=0.22",
                                                                    color=colour, lw=1.2, shrinkA=0, shrinkB=0), zorder=3)
    for n in p.nodes.values():
        fill, stroke = n["colour"]
        ec, lw = (V2, 2.4) if n["v2"] else (stroke, 1.1)
        if n["kind"] == "text":
            ax.text(n["x"], n["y"] + n["h"] / 2, n["text"].replace("**", ""), ha="left", va="center",
                    fontsize=FS_NOTE * PT, zorder=5, linespacing=1.35)
            continue
        if n["kind"] == "group":
            ax.add_patch(FancyBboxPatch((n["x"], n["y"]), n["w"], n["h"], boxstyle="round,pad=0,rounding_size=6",
                                        fc=fill, ec=ec, lw=lw, ls="--", zorder=1))
            ax.text(n["x"] + 10, n["y"] + 6, n["text"].replace("**", ""), ha="left", va="top", fontsize=FS_BOX * PT,
                    weight="bold", color="#7a5a00", zorder=5)
            continue
        if n["kind"] == "sum":
            ax.add_patch(Circle((n["x"] + n["w"] / 2, n["y"] + n["h"] / 2), n["w"] / 2, fc=fill, ec=ec, lw=lw, zorder=4))
        else:
            ax.add_patch(FancyBboxPatch((n["x"], n["y"]), n["w"], n["h"], boxstyle="round,pad=0,rounding_size=6",
                                        fc=fill, ec=ec, lw=lw, ls="--" if n["kind"] in ("note", "dbox") else "-", zorder=4))
        lines = n["text"].split("\n")
        left = n["kind"] == "note"
        fs = {"sum": FS_SUM, "tag": FS_TAG, "note": FS_NOTE}.get(n["kind"], FS_BOX)
        step = fs * 1.25
        y0 = n["y"] + n["h"] / 2 - (len(lines) - 1) * step / 2
        for j, line in enumerate(lines):
            ax.text(n["x"] + 10 if left else n["x"] + n["w"] / 2, y0 + step * j, line.replace("**", ""),
                    ha="left" if left else "center", va="center", fontsize=fs * PT,
                    weight="bold" if line.startswith("**") or n["kind"] in ("sum", "tag") else "normal",
                    color="#555555" if n["kind"] == "tag" else "black", zorder=5)
    for x, y, text, ha, colour in p.labels:
        ax.text(x, y, text.replace("**", ""), ha=ha, va="center", fontsize=FS_LABEL * PT, color=colour, zorder=6,
                bbox=dict(fc="white", ec="none", pad=0.6))
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def legend(p, y, items, x=20, step=190, prefix="lg"):
    for k, (lab, col) in enumerate(items):
        p.box(f"{prefix}{k}", x + 12 + step * k, y, 24, 16, "", col)
        p.box(f"{prefix}t{k}", x + 30 + 80 + step * k, y, 160, 22, lab, NOTE, "text")


# ===================================================================== (a) overview: U-Net on its side, LightViT-T inset
a = Page("(a) Overview", 1500, 515)
C = [52, 200, 400, 620, 840, 1060]     # columns: native | 224^2 | 112^2 | 56^2 | 28^2 | 14^2
WC = [80, 120, 172, 172, 172, 172]
R = [72, 192, 312, 432]                # rows: encoder | ApproxDA | decoder | heads
BH = 48
a.box("lvi", 1330, 210, 330, 395, "LightViT-T (stages 1–3, ImageNet-1k)", ("#fffdf5", "#999999"), "group")
a.box("nat", C[0], R[0], WC[0], 56, "Native\nslice", GREY)
a.box("inp", C[1], R[0], WC[1], BH, "Input", GREY)
for k, i in ((2, "e1"), (3, "e2"), (4, "e3")):
    a.box(i, C[k], R[0], WC[k], BH, "**Encoder block (R50)**", BLUE)
RL, RB = R[0], R[1]                     # rows of LightViT-T and the bottleneck block
a.box("lv", C[5], RL, WC[5], BH, "**LightViT-T**", PURPLE)
a.box("bn", C[5], RB, WC[5], BH, "**ApproxDA block**", YELLOW)
for k, i in ((2, "s1"), (3, "s2"), (4, "s3")):
    a.box(i, C[k], R[1], WC[k], BH, "**ApproxDA block**", YELLOW)
a.box("cm", C[5], R[2], WC[5], BH, "**Conv 3×3**", GREEN)
for k, i in ((4, "d1"), (3, "d2"), (2, "d3")):
    a.box(i, C[k], R[2], WC[k], BH, "**Decoder block**", GREEN)
a.box("d4", C[1], R[2], WC[1], BH, "**Up ×2**\n2 × Conv 3×3", GREEN)
a.box("hd", C[1], R[3], WC[1], BH, "**Seg head**", GREEN)
a.box("mk", C[0], R[3], WC[0], 56, "Mask", GREY)
a.box("pl", C[2], R[3], WC[2], BH, "**Point loss**\ntraining only", RED, "dbox")
a.box("nl", C[3], R[3], 120, 56, "Native\nlabel", GREY)
# encoder row (to the right): tensors above the boxes
a.edge("nat", "right", "inp", "left", label="resize", dy=30)
for u, v, lab in (("inp", "e1", "224²×3"), ("e1", "e2", "112²×64"), ("e2", "e3", "56²×256"),
                  ("e3", "lv", "28²×512")):
    a.edge(u, "right", v, "left", label=lab, dy=-32)
# skips down through the ApproxDA blocks
for e_, s_, d_ in (("e1", "s1", "d3"), ("e2", "s2", "d2"), ("e3", "s3", "d1")):
    a.edge(e_, "bottom", s_, "top", label="skip", side="right", colour=SKIP)
    a.edge(s_, "bottom", d_, "top", colour=SKIP)
a.edge("lv", "bottom", "bn", "top", label="14²×128", side="right")
a.edge("bn", "bottom", "cm", "top", label="14²×128", side="right")
# decoder row (to the left): tensors below the boxes
for u, v, lab in (("cm", "d1", "14²×64"), ("d1", "d2", "28²×64"), ("d2", "d3", "56²×64"), ("d3", "d4", "112²×64")):
    a.edge(u, "left", v, "right", label=lab, dy=32)
# heads
a.edge("d4", "bottom", "hd", "top", label="224²×16", side="left")
a.edge("hd", "left", "mk", "right", label="argmax, ↑ H×W", dy=40)
a.edge("hd", "right", "pl", "left", label="224²×K", dy=-30, dashed=True)
a.edge("nl", "left", "pl", "right", label="H×W", dy=-30, dashed=True)
# LightViT-T inset
X, WI = 1310, 160
for i, y, t in (("pj", 62, "**1×1 proj**"), ("t1", 124, "**Stage 1**"), ("t2", 186, "**Stage 2**"), ("t3", 248, "**Stage 3**"),
                ("fu", 310, "**1×1 conv**")):
    a.box(i, X, y, WI, 36, t, PURPLE)
a.box("lsm", X, 362, 30, 30, "+", SUM, "sum")
for u, v, lab in (("pj", "t1", "28²×64"), ("t1", "t2", "merge ↓2"), ("t2", "t3", "merge ↓2"), ("t3", "fu", "7²×256"),
                  ("fu", "lsm", "↑2")):
    a.edge(u, "bottom", v, "top", label=lab, side="right")
a.edge("t2", "right", "lsm", "right", via=[(1425, 186), (1425, 362)], label="14²×128", seg=1, side="right")
a.labels.append((X, 392, "out: 14²×128 · global tokens in every stage", "center", "#444444"))
a.edge("lv", "right", "lvi", "left", bf=(RL - 12.5) / 395, dashed=True)
# legend
legend(a, 436, [("encoder block (R50)", BLUE), ("LightViT-T", PURPLE), ("ApproxDA block (b)", YELLOW)], x=715, step=255)
legend(a, 470, [("decoder block (c)", GREEN), ("point loss (d)", RED)], x=715, step=255, prefix="lgb")
a.box("lgn", 1325, 470, 260, 22, "blue: skips · dashed: training only", NOTE, "text")
for i in ("s1", "s2", "s3", "bn"):
    a.tag(i, "b")       # every ApproxDA block has the structure of page (b)
for i in ("d1", "d2", "d3"):
    a.tag(i, "c")       # every decoder block has the structure of page (c)
a.tag("pl", "d")

# ===================================================================== (b) ApproxDA block (two branches, fixed gate)
c = Page("(b) ApproxDA block", 1220, 345)
YU, YL, YM = 80, 205, 142
c.box("x", 34, YM, 48, 60, "x", GREY)
xs = [140, 265, 390, 515, 640, 765]
for i, (k, t, col) in enumerate((("c5a", "**Conv 3×3**", YELLOW), ("wp", "**Window\npartition**", ORANGE),
                                 ("lr", "**Low-rank\nprojection**", ORANGE), ("pa", "**PAM\nattention**", ORANGE),
                                 ("wr", "**Window\nreverse**", ORANGE), ("c51", "**Conv 3×3**", YELLOW))):
    c.box(k, xs[i], YU, 100, 50, t, col)
for i, k, t, col in ((0, "c5c", "**Conv 3×3**", YELLOW), (1, "gs", "**Group\nsplit**", ORANGE),
                     (3, "ca", "**Channel\nattention**", ORANGE), (5, "c52", "**Conv 3×3**", YELLOW)):
    c.box(k, xs[i], YL, 100, 50, t, col)
c.box("g", 895, YM, 110, 56, "**Gate g = ½**\n½·P + ½·C", YELLOW)
c.box("c8", 1030, YM, 95, 50, "**Conv 1×1**", YELLOW)
c.box("y", 1150, YM, 48, 60, "y", GREY)
c.edge("x", "right", "c5a", "left", via=[(72, YM), (72, YU)])
c.edge("x", "right", "c5c", "left", via=[(72, YM), (72, YL)])
c.labels.append((22, YM - 46, "h×w×C", "left", "#444444"))
for u, v, lab in (("c5a", "wp", "C/16"), ("wp", "lr", "M² tokens"), ("lr", "pa", "K, V: M²→r"), ("pa", "wr", "αE + x"),
                  ("wr", "c51", "")):
    c.edge(u, "right", v, "left", label=lab, dy=-36)
for u, v, lab in (("c5c", "gs", "C/16"), ("gs", "ca", "G groups"), ("ca", "c52", "βE + x")):
    c.edge(u, "right", v, "left", label=lab, dy=-36 if u == "c5c" else None)
c.edge("c51", "right", "g", "top", via=[(895, YU)], label="P", seg=0)
c.edge("c52", "right", "g", "bottom", via=[(895, YL)], label="C", seg=0)
c.edge("g", "right", "c8", "left", label="C/16")
c.edge("c8", "right", "y", "left", label="C")
c.box("nc", 610, 297, 1190, 66,
      "M = 28 at 224² (32 at 256²), the side of the 1/8 map, clamped to each map: windows at the 1/2 and 1/4 skips, "
      "global attention\n    at the 1/8 skip and the bottleneck · r = 32: projection rank · G = 8: channel groups "
      "(capped by the C/16 width) · α, β init 0\nC/16: channel compression (DANetHead layout, no outer residual) · "
      "Conv 3×3 = conv–BN–ReLU · g fixed at ½ (the learned gate stayed ≈ ½)", NOTE, "text")

# ===================================================================== (c) slim decoder block
d = Page("(c) Decoder block", 1160, 270)
YS, YMn = 55, 160
d.box("sk", 70, YS, 125, 46, "**Encoder skip**", BLUE)
d.box("ad", 245, YS, 155, 46, "**ApproxDA block**", YELLOW)
d.box("rs", 420, YS, 125, 46, "**ReduceBlock**", GREEN)
d.box("pv", 70, YMn, 125, 46, "previous block", GREY)
d.box("up", 245, YMn, 115, 46, "**Up ×2**", GREEN)
d.box("sm", 420, YMn, 34, 34, "+", SUM, "sum")
d.box("rf", 565, YMn, 125, 46, "**ReduceBlock**", GREEN)
d.box("nx", 715, YMn, 110, 46, "next block", GREY)
d.edge("sk", "right", "ad", "left", colour=SKIP)
d.edge("ad", "right", "rs", "left", colour=SKIP)
d.labels.append((157, YS - 34, "(2h)²×C_s", "center", SKIP))
d.edge("rs", "bottom", "sm", "top", label="(2h)²×64", side="right")
d.edge("pv", "right", "up", "left", label="h²×64", dy=-32)
d.edge("up", "right", "sm", "left", label="(2h)²×64", dy=-32)
d.edge("sm", "right", "rf", "left")
d.edge("rf", "right", "nx", "left", label="(2h)²×64", dy=-32)
d.tag("ad", "b")
# ReduceBlock detail
d.box("rbn", 940, 20, 200, 22, "**ReduceBlock**", NOTE, "text")
d.box("ri", 890, 60, 70, 34, "in", GREY)
d.box("r1", 1055, 60, 190, 34, "**Conv 3×3·BN·ReLU**", GREEN)
d.box("r2", 1055, 115, 150, 34, "**Conv 3×3·BN**", GREEN)
d.box("rsc", 890, 115, 100, 34, "**Conv 1×1**", GREEN)
d.box("rp", 1055, 165, 30, 30, "+", SUM, "sum")
d.box("ro", 1120, 165, 64, 34, "ReLU", GREY)
d.edge("ri", "right", "r1", "left")
d.edge("r1", "bottom", "r2", "top")
d.edge("ri", "bottom", "rsc", "top")
d.edge("r2", "bottom", "rp", "top")
d.edge("rsc", "bottom", "rp", "left", via=[(890, 165)])
d.edge("rp", "right", "ro", "left")
d.box("nd", 520, 238, 1000, 24, "C_s = 512 / 256 / 64 for the 1/8, 1/4, 1/2 skips; the skip is added, not concatenated.", NOTE, "text")

# ===================================================================== (d) point supervision loss (training only)
e = Page("(d) Point loss", 1090, 480)
Y0, Y1, Y2 = 50, 165, 265
e.box("lg", 85, Y0, 150, 44, "**Coarse logits**", GREEN)
e.box("ml", 860, Y0, 190, 50, "**Main loss** (224²)\n0.5 CE + 0.5 Dice", GREEN)
for i, cx, w, t in (("rp", 85, 110, "**Uniform\ncandidates**"), ("un", 235, 120, "**Uncertainty**"),
                    ("sel", 390, 130, "**Point\nselection**"), ("bs", 560, 120, "**Bilinear\nsample**"),
                    ("ce", 715, 110, "**Cross-\nentropy**")):
    e.box(i, cx, Y1, w, 50, t, RED)
e.box("sm", 860, Y1, 36, 36, "+", SUM, "sum")
e.box("L", 980, Y1, 70, 50, "L", GREY)
e.box("nl", 85, Y2, 110, 50, "Native\nlabel", GREY)
e.box("cf", 235, Y2, 120, 50, "**Class\nfloor**", RED)
e.box("ns", 390, Y2, 130, 50, "**Nearest\nsample**", RED)
lx = e.anchor("lg", "bottom", 0.85)[0][0]
e.edge("lg", "right", "bs", "top", via=[(560, Y0)], label="224²×K", seg=0)
e.edge("lg", "bottom", "un", "top", via=[(lx, 108), (235, 108)], af=0.85, label="no grad", seg=1, dy=-10, dashed=True)
for u, v, lab in (("rp", "un", "3N′"), ("un", "sel", "3N′"), ("sel", "bs", "N"), ("bs", "ce", "N×K"), ("ce", "sm", "L_pt"),
                  ("sm", "L", ""), ("nl", "cf", "H×W")):
    e.edge(u, "right", v, "left", label=lab, dy=-34)
sx = e.anchor("sel", "bottom", 0.2)[0][0]
e.edge("cf", "top", "sel", "bottom", via=[(235, 215), (sx, 215)], bf=0.2, label="0.05 N", seg=1, dy=-10)
e.edge("sel", "bottom", "ns", "top", label="N", side="right")
e.edge("nl", "bottom", "ns", "bottom", via=[(85, 310), (390, 310)], label="labels", seg=1, dy=-10)
e.edge("ns", "right", "ce", "bottom", via=[(715, Y2)], label="N labels", seg=0)
e.edge("ml", "bottom", "sm", "top", label="L_seg", side="right")
e.box("tr", 545, 400, 1060, 130,
      "Training only: nothing is added to the network; at inference the 224² logits are argmax-ed and resized (nearest) to H×W.\n"
      "N = 2048 points per image. Class floor: 0.05 N points (≈ 10 per class present), uniform inside each class of the label;\n"
      "    the other N′ = 0.95 N: the 75 % most uncertain of 3N′ uniform candidates (−(top1 − top2), k = 3) plus 25 % uniform.\n"
      "Logits sampled bilinearly (align_corners, as the test zoom); labels nearest on the native canvas:\n"
      "    512² AMOS · 384² CVC · 256² ISIC / FeTA · 224² ACDC.  L = L_seg + L_pt.\n"
      "Point selection of PointRend (Kirillov et al., CVPR 2020) plus the class floor, without its point head.",
      NOTE, "note")

pages = [a, c, d, e]
xml = '<mxfile host="app.diagrams.net">' + "".join(drawio_page(p, f"p{k}") for k, p in enumerate(pages)) + "</mxfile>"
open(os.path.join(OUT, NAME + ".drawio"), "w").write(xml)
for k, p in enumerate(pages):
    preview(p, os.path.join(OUT, f"{NAME}_{'abcd'[k]}.png"))
print("written", os.path.join(OUT, NAME + ".drawio"), "+ previews", [f"{NAME}_{x}.png" for x in "abcd"])
