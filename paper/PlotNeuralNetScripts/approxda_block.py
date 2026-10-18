"""
ApproxDA-TransUNet — ApproxDABlock detail diagram (C/16 DANetHead layout)
=========================================================================
        ┌─→ Conv3x3 C→C' ─→ LowRankWindowedPAM ─→ Conv3x3 ─→ g  ─────┐
  x ────┤                                                              ├─→ 1x1 Conv C'→C ─→ y
        └─→ Conv3x3 C→C' ─→ GroupedCAM         ─→ Conv3x3 ─→ (1-g) ─┘
  C' = C/16; no outer residual (as DANetHead). Mirrors ApproxDABlock in
  experiments/ApproxDA-TransUNet/Architecture/block.py.

Uses the PlotNeuralNet package (https://github.com/kgruiz/PlotNeuralNet).
Run from this directory:
    python approxda_block.py [path/to/PlotNeuralNet]   # writes approxda_block.tex
    pdflatex approxda_block.tex
    cp approxda_block.pdf ../figures/
"""

import os
import sys

PNN_ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/projects/PlotNeuralNet")
sys.path.insert(0, PNN_ROOT)
from PlotNeuralNet.PyCore.TikzGen import ToBegin, ToConnection, ToCor, ToEnd, ToGenerate, ToHead  # noqa: E402


def box(name, offset, to, width, height, depth, caption, dim="", fill=r"\FcColor"):
    """Box layer with a custom fill (TikzGen.ToConv has no fill) and its tensor width `dim` under the front edge."""
    return rf"""
\pic[shift={{{offset}}}] at {to} {{
    Box={{
        name={name},
        caption={caption},
        fill={fill},
        height={height},
        width={width},
        depth={depth}
    }}
}};
\node[below=0pt, font=\scriptsize] at ($({name}-nearsouthwest)!0.5!({name}-nearsoutheast)$) {{{dim}}};
"""


# Block visual dimensions
BW, BD = 5.0, 2.8  # width, height/depth for branch operator boxes
CW = 1.6  # width of 3x3 conv-BN-ReLU slabs
IO = 3.4  # height/depth of C-channel tensors (input, fusion, output)
GAP = 2.0  # horizontal gap between boxes in a branch
BY = 2.0  # y-offset of the upper/lower branches

PAM = r"\FcColor"
CAM = r"\CAMColor"
CONV = r"\ConvReluColor"

arch = [
    ToHead(os.path.join(PNN_ROOT, "PlotNeuralNet")).replace("border=8pt", "border={0pt 2pt 0pt 2pt}"),
    ToCor(),
    r"\def\CAMColor{rgb:green,4;white,2}" + "\n",
    r"\usetikzlibrary{calc}" + "\n",
    r"\definecolor{paramcolor}{rgb}{0.80,0.30,0.05}" + "\n",
    ToBegin(),
    # ── Input ──────────────────────────────────────────────────────────────
    box("x_in", "(0,0,0)", "(0,0,0)", 3.2, IO, IO, r"$\mathbf{x}$", r"$C$", r"\ConvColor"),
    # ── PAM branch (upper) ─────────────────────────────────────────────────
    box("conv5a", f"(2.4,{BY},0)", "(x_in-east)", CW, BD, BD,
        r"\raisebox{-14pt}{\shortstack{Conv\\$3{\times}3$}}", r"$C'$", CONV),
    box("win_part", f"({GAP},0,0)", "(conv5a-east)", BW, BD, BD,
        r"\raisebox{-14pt}{\shortstack{Window\\Partition}}", r"$C'$", PAM),
    box("proj_r", f"({GAP},0,0)", "(win_part-east)", 2.6, BD, BD,
        r"\raisebox{-14pt}{\shortstack{Low-Rank\\Projection}}", r"$r$", PAM),
    box("pam_attn", f"({GAP},0,0)", "(proj_r-east)", BW, BD, BD,
        r"\raisebox{-14pt}{\shortstack{PAM\\Attention}}", r"$C'$", PAM),
    box("win_rev", f"({GAP},0,0)", "(pam_attn-east)", BW, BD, BD,
        r"\raisebox{-14pt}{\shortstack{Window\\Reverse}}", r"$C'$", PAM),
    box("conv51", f"({GAP},0,0)", "(win_rev-east)", CW, BD, BD,
        r"\raisebox{-14pt}{\shortstack{Conv\\$3{\times}3$}}", r"$C'$", CONV),
    # ── CAM branch (lower) ─────────────────────────────────────────────────
    box("conv5c", f"(2.4,{-BY},0)", "(x_in-east)", CW, BD, BD,
        r"\raisebox{-14pt}{\shortstack{Conv\\$3{\times}3$}}", r"$C'$", CONV),
    box("grp_split", f"({GAP},0,0)", "(conv5c-east)", BW, BD, BD,
        r"\raisebox{-14pt}{\shortstack{Group\\Split}}", r"$C'$", CAM),
    box("cam_attn", f"(0,{-2 * BY},0)", "(pam_attn-west)", BW, BD, BD,
        r"\raisebox{-14pt}{\shortstack{Channel\\Attention}}", r"$C'$", CAM),
    box("conv52", f"(0,{-2 * BY},0)", "(conv51-west)", CW, BD, BD,
        r"\raisebox{-14pt}{\shortstack{Conv\\$3{\times}3$}}", r"$C'$", CONV),
    # ── Gate, fusion and output ────────────────────────────────────────────
    box("gate", f"(1.8,{-BY},0)", "(conv51-east)", 2.5, 2.5, 2.5, r"Gate $\mathbf{g}$", r"$C'$", r"\SoftmaxColor"),
    box("fusion", "(2.2,0,0)", "(gate-east)", CW, IO, IO,
        r"\raisebox{-14pt}{\shortstack{Conv $1{\times}1$\\$C'{\to}C$}}", r"$C$", r"\ConvColor"),
    box("y_out", "(2.2,0,0)", "(fusion-east)", 3.2, IO, IO, r"$\mathbf{y}$", r"$C$", r"\ConvColor"),
    # ── Connections ────────────────────────────────────────────────────────
    ToConnection("x_in", "conv5a"),
    ToConnection("conv5a", "win_part"),
    ToConnection("win_part", "proj_r"),
    ToConnection("proj_r", "pam_attn"),
    ToConnection("pam_attn", "win_rev"),
    ToConnection("win_rev", "conv51"),
    ToConnection("conv51", "gate"),
    ToConnection("x_in", "conv5c"),
    ToConnection("conv5c", "grp_split"),
    ToConnection("grp_split", "cam_attn"),
    ToConnection("cam_attn", "conv52"),
    ToConnection("conv52", "gate"),
    ToConnection("gate", "fusion"),
    ToConnection("fusion", "y_out"),
    # ── Parameter labels + legend ──────────────────────────────────────────
    r"""\node[above=5pt, font=\Large\bfseries\itshape, text=paramcolor]
        at (win_part-north) {$M$};
\node[above=5pt, font=\Large\bfseries\itshape, text=paramcolor]
        at (proj_r-north) {$r$};
\node[above=5pt, font=\Large\bfseries\itshape, text=paramcolor]
        at (grp_split-north) {$G$};
\node[draw=gray!50, thin, rounded corners=3pt, inner sep=4.5pt, align=left,
      anchor=north east, font=\normalsize\itshape, fill=white!95!gray]
      at ($(y_out-east |- conv52-south) + (0.5, 0.0)$) {
      \textcolor{paramcolor}{\textbf{\textit{M}}}: window size \quad
      \textcolor{paramcolor}{\textbf{\textit{r}}}: projection rank \\
      \textcolor{paramcolor}{\textbf{\textit{G}}}: channel groups \quad
      $C' = C/16$ \\
      Conv $3{\times}3$: conv--BN--ReLU};
""",
    ToEnd(),
]


def main():
    namefile = os.path.splitext(os.path.basename(sys.argv[0]))[0]
    ToGenerate(arch, namefile + ".tex")


if __name__ == "__main__":
    main()
