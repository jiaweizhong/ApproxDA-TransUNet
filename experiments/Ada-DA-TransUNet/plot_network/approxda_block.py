"""
ApproxDA-TransUNet — AdaDABlock Detail Diagram
===============================================
        ┌─→ LowRankWindowedPAM ──→ g  ────────┐
  x ────┤                                      ├─→ Fusion Conv ─→ + x ─→ out
        └─→ GroupedCAM         ──→ (1-g) ──────┘

Run:
    cd PlotNeuralNet/pyexamples
    bash ../tikzmake.sh approxda_block
"""

import sys
sys.path.append('../')
from pycore.tikzeng import *
from pycore.blocks  import *

# Block visual dimensions shared by branch layers
BW, BD = 4, 6   # width, depth

arch = [
    to_head('..'),
    to_cor(),
    # Extra color: green for the CAM branch
    r"\def\CAMColor{rgb:green,4;white,2}" + "\n",
    to_begin(),

    # ── Input  ─────────────────────────────────────────────────────────────
    to_Conv('x_in', ' ', 512,
            offset="(0,0,0)", to="(0,0,0)",
            width=2, height=10, depth=10,
            caption="$x$"),

    # ── PAM branch (upper, y=+5) — blue (FcColor) ──────────────────────────
    to_Conv('win_part', ' ', 512,
            offset="(3, 5, 0)", to="(x_in-east)",
            width=BW, height=BD, depth=BD,
            fill=r"\FcColor",
            caption="Win Partition"),

    to_Conv('proj_r', ' ', 32,
            offset="(1.5, 0, 0)", to="(win_part-east)",
            width=1, height=BD, depth=BD,
            fill=r"\FcColor",
            caption="LR Proj"),

    to_Conv('pam_attn', ' ', 512,
            offset="(1.5, 0, 0)", to="(proj_r-east)",
            width=BW, height=BD, depth=BD,
            fill=r"\FcColor",
            caption="PAM Attn"),

    to_Conv('win_rev', ' ', 512,
            offset="(1.5, 0, 0)", to="(pam_attn-east)",
            width=BW, height=BD, depth=BD,
            fill=r"\FcColor",
            caption="Win Reverse"),

    # ── CAM branch (lower, y=-5) — green (CAMColor) ────────────────────────
    to_Conv('grp_split', ' ', 512,
            offset="(3, -5, 0)", to="(x_in-east)",
            width=BW, height=BD, depth=BD,
            fill=r"\CAMColor",
            caption="Grp Split"),

    to_Conv('cam_attn', ' ', 512,
            offset="(4.5, 0, 0)", to="(grp_split-east)",
            width=BW, height=BD, depth=BD,
            fill=r"\CAMColor",
            caption="Chan Attn"),

    # ── Gate (center, right of win_rev) — magenta (SoftmaxColor) ───────────
    # Positioned 2 units right and 5 units down from win_rev-east (back to y=0)
    to_Conv('gate', ' ', 1,
            offset="(2, -5, 0)", to="(win_rev-east)",
            width=1, height=4, depth=4,
            fill=r"\SoftmaxColor",
            caption="Gate $g$"),

    # ── Fusion conv  1x1 — yellow (ConvColor) ──────────────────────────────
    to_Conv('fusion', ' ', 512,
            offset="(2, 0, 0)", to="(gate-east)",
            width=BW, height=10, depth=10,
            caption="Fusion"),

    # ── Output ─────────────────────────────────────────────────────────────
    to_Conv('x_out', ' ', 512,
            offset="(2, 0, 0)", to="(fusion-east)",
            width=2, height=10, depth=10,
            caption="$+x$"),

    # ── Connections ────────────────────────────────────────────────────────
    to_connection('x_in',     'win_part'),
    to_connection('win_part', 'proj_r'),
    to_connection('proj_r',   'pam_attn'),
    to_connection('pam_attn', 'win_rev'),
    to_connection('win_rev',  'gate'),

    to_connection('x_in',     'grp_split'),
    to_connection('grp_split','cam_attn'),
    to_connection('cam_attn', 'gate'),

    to_connection('gate',   'fusion'),
    to_connection('fusion', 'x_out'),

    to_end()
]


def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex')


if __name__ == '__main__':
    main()
