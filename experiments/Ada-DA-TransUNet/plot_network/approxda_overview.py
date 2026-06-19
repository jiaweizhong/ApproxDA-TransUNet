"""
ApproxDA-TransUNet — Overview Architecture Diagram (U-shape)
============================================================
Architecture (R50-ViT-B/16, n_skip=3):
  R50 Root (112x112, 64ch) -> Block1 (56x56, 256ch) -> Block2 (28x28, 512ch)
  -> Block3 + PatchEmbed + AdaDA (14x14, 768ch)
  -> ViT-B/16 x12 (14x14, 768ch)
  -> conv_more (14x14, 512ch)
  -> Dec1+ApproxDA (28x28, 256ch) <- skip 512 (from Block2)
  -> Dec2+ApproxDA (56x56, 128ch) <- skip 256 (from Block1)
  -> Dec3+ApproxDA (112x112, 64ch) <- skip 64  (from Root)
  -> Dec4 (224x224, 16ch)
  -> SegHead (224x224, 9ch)

Run:
    cd PlotNeuralNet/pyexamples
    bash ../tikzmake.sh approxda_overview
"""

import sys
sys.path.append('../')
from pycore.tikzeng import *
from pycore.blocks  import *

arch = [
    to_head('..'),
    to_cor(),
    r"\def\ConcatColor{rgb:blue,5;red,2.5;white,5}" + "\n",
    to_begin(),

    # ── Encoder (R50) — staircase descending ───────────────────────────────
    # Stage 1: Root  112x112, 64ch
    to_ConvConvRelu('cr1', 112, (64, 64),
                    offset="(0,0,0)", to="(0,0,0)",
                    width=(2, 2), height=40, depth=40),
    to_Pool('p1', offset="(1.2,-10,0)", to="(cr1-east)", height=32, depth=32),

    # Stage 2: Block1  56x56, 256ch
    to_ConvConvRelu('cr2', 56, (256, 256),
                    offset="(0,0,0)", to="(p1-east)",
                    width=(4, 4), height=32, depth=32),
    to_Pool('p2', offset="(1.2,-8.5,0)", to="(cr2-east)", height=24, depth=24),

    # Stage 3: Block2  28x28, 512ch
    to_ConvConvRelu('cr3', 28, (512, 512),
                    offset="(0,0,0)", to="(p2-east)",
                    width=(5, 5), height=24, depth=24),
    to_Pool('p3', offset="(1.2,-6.5,0)", to="(cr3-east)", height=14, depth=14),

    # Stage 4: Block3 + PatchEmbed + AdaDA  14x14, 768ch
    to_ConvConvRelu('cr4', 14, (768, 768),
                    offset="(0,0,0)", to="(p3-east)",
                    width=(6, 6), height=14, depth=14, caption="AdaDA"),

    # ── ViT Bottleneck (12 Transformer layers)  14x14 ──────────────────────
    to_Conv('vit', ' ', 768,
            offset="(2,-3,0)", to="(cr4-east)",
            width=12, height=8, depth=8,
            caption="ViT-B/16", fill=r"\FcColor"),

    # ── Bridge conv  14x14, 512ch ───────────────────────────────────────────
    to_ConvConvRelu('br', 14, (512, 512),
                    offset="(0,0,0)", to="(vit-east)",
                    width=(5, 5), height=8, depth=8),

    # ── Decoder 1: 28x28, 256ch  (ApproxDA Ball at cr3 level) ─────────────
    to_UnPool('up1', offset="(0,0,0)", to="(br-east)", height=14, depth=14),
    to_ConvConvRelu('ucr1', 28, (256, 256),
                    offset="(0,0,0)", to="(up1-east)",
                    width=(4, 4), height=14, depth=14),
    # Ball ada1 is +9.5 above ucr1-anchor → sits at cr3's y level
    r"""\pic[shift={(0,9.5,0)}] at (ucr1-anchor)
    {Ball={name=ada1,fill=\ConcatColor,radius=2.5,logo=$\mathcal{A}$}};
""",
    to_ConvConvRelu('ucr1a', 28, (256, 256),
                    offset="(1.4,0,0)", to="(ada1-east)",
                    width=(4, 4), height=14, depth=14),

    # ── Decoder 2: 56x56, 128ch  (ApproxDA Ball at cr2 level) ─────────────
    to_UnPool('up2', offset="(0,0,0)", to="(ucr1a-east)", height=24, depth=24),
    to_ConvConvRelu('ucr2', 56, (128, 128),
                    offset="(0,0,0)", to="(up2-east)",
                    width=(3, 3), height=24, depth=24),
    # Ball ada2 is +8.5 above ucr2-anchor → sits at cr2's y level
    r"""\pic[shift={(0,8.5,0)}] at (ucr2-anchor)
    {Ball={name=ada2,fill=\ConcatColor,radius=2.5,logo=$\mathcal{A}$}};
""",
    to_ConvConvRelu('ucr2a', 56, (128, 128),
                    offset="(1.8,0,0)", to="(ada2-east)",
                    width=(3, 3), height=24, depth=24),

    # ── Decoder 3: 112x112, 64ch  (ApproxDA Ball at cr1 level) ────────────
    to_UnPool('up3', offset="(0,0,0)", to="(ucr2a-east)", height=32, depth=32),
    to_ConvConvRelu('ucr3', 112, (64, 64),
                    offset="(0,0,0)", to="(up3-east)",
                    width=(2, 2), height=32, depth=32),
    # Ball ada3 is +10 above ucr3-anchor → sits at cr1's y level
    r"""\pic[shift={(0,10,0)}] at (ucr3-anchor)
    {Ball={name=ada3,fill=\ConcatColor,radius=2.5,logo=$\mathcal{A}$}};
""",
    to_ConvConvRelu('ucr3a', 112, (64, 64),
                    offset="(2,0,0)", to="(ada3-east)",
                    width=(2, 2), height=32, depth=32),

    # ── Decoder 4: 224x224, 16ch  (no skip, no ApproxDA) ───────────────────
    to_UnPool('up4', offset="(0,0,0)", to="(ucr3a-east)", height=40, depth=40),
    to_ConvConvRelu('ucr4', 224, (16, 16),
                    offset="(0,0,0)", to="(up4-east)",
                    width=(1, 1), height=40, depth=40),

    # ── Segmentation head ───────────────────────────────────────────────────
    to_ConvSoftMax('seg', 9, offset="(0.75,0,0)", to="(ucr4-east)",
                   width=1, height=40, depth=40, caption="Seg"),

    # ── Connections ─────────────────────────────────────────────────────────
    # Encoder forward (diagonal arrows for the descending staircase)
    to_connection('cr1', 'p1'),
    to_connection('p1', 'cr2'),
    to_connection('cr2', 'p2'),
    to_connection('p2', 'cr3'),
    to_connection('cr3', 'p3'),
    to_connection('p3', 'cr4'),
    to_connection('cr4', 'vit'),
    to_connection('vit', 'br'),
    # Decoder forward (ascending staircase)
    to_connection('br',     'up1'),
    to_connection('ucr1a',  'up2'),
    to_connection('ucr2a',  'up3'),
    to_connection('ucr3a',  'up4'),
    to_connection('up4',    'ucr4'),
    to_connection('ucr4',   'seg'),

    # Skip connections: encoder-east → Ball-west (horizontal),
    #   ucr-north → Ball-south (vertical up),  Ball-east → ucr_a-west
    r"""\draw [copyconnection] (cr3-east)  -- node {\copymidarrow} (ada1-west);
\draw [copyconnection] (ucr1-north) -- node {\copymidarrow} (ada1-south);
\draw [copyconnection] (ada1-east)  -- node {\copymidarrow} (ucr1a-west);
\draw [copyconnection] (cr2-east)   -- node {\copymidarrow} (ada2-west);
\draw [copyconnection] (ucr2-north) -- node {\copymidarrow} (ada2-south);
\draw [copyconnection] (ada2-east)  -- node {\copymidarrow} (ucr2a-west);
\draw [copyconnection] (cr1-east)   -- node {\copymidarrow} (ada3-west);
\draw [copyconnection] (ucr3-north) -- node {\copymidarrow} (ada3-south);
\draw [copyconnection] (ada3-east)  -- node {\copymidarrow} (ucr3a-west);
""",

    to_end()
]


def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex')


if __name__ == '__main__':
    main()
