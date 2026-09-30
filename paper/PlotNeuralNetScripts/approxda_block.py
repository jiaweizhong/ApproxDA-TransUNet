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

# Block visual dimensions — balanced flat aspect ratio
BW, BD = 6.2, 2.8  # width, height/depth for branch layers
BY = 2.0          # y-offset for upper/lower branches

arch = [
    to_head('..').replace('border=8pt', 'border={0pt 2pt 0pt 2pt}'),
    to_cor(),
    r"\def\CAMColor{rgb:green,4;white,2}" + "\n",
    r"\usetikzlibrary{calc}" + "\n",
    r"\definecolor{paramcolor}{rgb}{0.80,0.30,0.05}" + "\n",
    r"\tikzstyle{captionlabel}=[align=center, font=\scriptsize\bfseries]" + "\n",
    to_begin(),

    # ── Input  ─────────────────────────────────────────────────────────────
    to_Conv('x_in', ' ', 512,
            offset="(0,0,0)", to="(0,0,0)",
            width=3.2, height=3.4, depth=3.4,
            caption=r"$\mathbf{x}$"),

    # ── PAM branch (upper, y=+BY) — blue (FcColor) ─────────────────────────
    to_Conv('win_part', ' ', 512,
            offset="(2.8, 2.0, 0)", to="(x_in-east)",
            width=BW, height=BD, depth=BD,
            fill=r"\FcColor",
            caption="Window Partition"),

    to_Conv('proj_r', ' ', 32,
            offset="(1.8, 0, 0)", to="(win_part-east)",
            width=2.6, height=BD, depth=BD,
            fill=r"\FcColor",
            caption=r"\raisebox{-14pt}{\shortstack{Low-Rank \\ Projection}}"),

    to_Conv('pam_attn', ' ', 512,
            offset="(1.8, 0, 0)", to="(proj_r-east)",
            width=BW, height=BD, depth=BD,
            fill=r"\FcColor",
            caption="PAM Attention"),

    to_Conv('win_rev', ' ', 512,
            offset="(1.8, 0, 0)", to="(pam_attn-east)",
            width=BW, height=BD, depth=BD,
            fill=r"\FcColor",
            caption="Window Reverse"),

    # ── CAM branch (lower, y=-BY) — green (CAMColor) ───────────────────────
    to_Conv('grp_split', ' ', 512,
            offset="(2.8, -2.0, 0)", to="(x_in-east)",
            width=BW, height=BD, depth=BD,
            fill=r"\CAMColor",
            caption="Group Split"),

    to_Conv('cam_attn', ' ', 512,
            offset="(5.0, 0, 0)", to="(grp_split-east)",
            width=BW, height=BD, depth=BD,
            fill=r"\CAMColor",
            caption="Channel Attention"),

    # ── Gate (centre, 2.0 right and BY down from win_rev) — magenta ────────
    to_Conv('gate', ' ', 512,
            offset="(2.0, -2.0, 0)", to="(win_rev-east)",
            width=2.5, height=2.5, depth=2.5,
            fill=r"\SoftmaxColor",
            caption=r"Gate $\mathbf{g}$"),

    # ── Fusion conv — yellow (ConvColor) ───────────────────────────────────
    to_Conv('fusion', ' ', 512,
            offset="(2.0, 0, 0)", to="(gate-east)",
            width=BW, height=3.4, depth=3.4,
            caption="Fusion $1{\\times}1$"),

    # ── Output ─────────────────────────────────────────────────────────────
    to_Conv('x_out', ' ', 512,
            offset="(2.0, 0, 0)", to="(fusion-east)",
            width=3.2, height=3.4, depth=3.4,
            caption=r"$+\mathbf{x}$"),

    # ── Connections ────────────────────────────────────────────────────────
    to_connection('x_in',      'win_part'),
    to_connection('win_part',  'proj_r'),
    to_connection('proj_r',    'pam_attn'),
    to_connection('pam_attn',  'win_rev'),
    to_connection('win_rev',   'gate'),

    to_connection('x_in',      'grp_split'),
    to_connection('grp_split', 'cam_attn'),
    to_connection('cam_attn',  'gate'),

    to_connection('gate',   'fusion'),
    to_connection('fusion', 'x_out'),

    # ── Parameter labels (unified orange, large bold) + legend ─────────────
    r"""\node[above=5pt, font=\Large\bfseries\itshape, text=paramcolor]
        at (win_part-north) {$M$};
\node[above=5pt, font=\Large\bfseries\itshape, text=paramcolor]
        at (proj_r-north) {$r$};
\node[above=5pt, font=\Large\bfseries\itshape, text=paramcolor]
        at (grp_split-north) {$G$};
\node[draw=gray!50, thin, rounded corners=3pt, inner sep=4.5pt,
      anchor=north east, font=\normalsize\itshape, fill=white!95!gray]
      at ($(x_out-east |- cam_attn-south) + (0.0, -0.10)$) {
      \textcolor{paramcolor}{\textbf{\textit{M}}}: window size \quad
      \textcolor{paramcolor}{\textbf{\textit{r}}}: projection rank \quad
      \textcolor{paramcolor}{\textbf{\textit{G}}}: channel groups};
""",

    to_end()
]


def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex')


if __name__ == '__main__':
    main()
