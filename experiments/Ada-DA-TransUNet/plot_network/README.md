# PlotNeuralNet Figures — ApproxDA-TransUNet

Two diagram scripts based on the actual architecture in `Architecture/`.

## Setup

```bash
git clone https://github.com/HarisIqbal88/PlotNeuralNet
cp approxda_overview.py PlotNeuralNet/pyexamples/
cp approxda_block.py    PlotNeuralNet/pyexamples/
```

## Generate

```bash
cd PlotNeuralNet/pyexamples

# Full encoder-decoder overview
python approxda_overview.py && pdflatex approxda_overview.tex

# AdaDABlock internals
python approxda_block.py && pdflatex approxda_block.tex
```

## What each file shows

### approxda_overview.py
Full U-Net-style encoder-decoder:
- **R50 encoder**: Root (112×64) → Block1 (56×256) → Block2 (28×512) → Block3+PatchEmbed+AdaDA (14×768)
- **ViT bottleneck**: 12 transformer layers (14×768)
- **Decoder**: 3 upconv blocks each with ApproxDA applied to the skip feature before concat → output 224×9
- **Skip connections** (dashed arcs): Block2→Dec1, Block1→Dec2, Root→Dec3

### approxda_block.py
AdaDABlock internals (two parallel branches + gate):
- **PAM branch**: Window partition → Low-rank projection → Attention bmm → Window reverse
- **CAM branch**: Group split → Channel attention per group
- **Gate**: sigmoid scalar `g` blending PAM and CAM outputs
- **Fusion**: 1×1 conv + residual add

## Why you can't auto-generate from the PyTorch files

PlotNeuralNet is purely declarative — the Python scripts describe *what to draw*, not the actual computation graph. The layer sizes and connections in these scripts were derived by reading `Architecture/AdaDATransUNet.py` and `Architecture/block.py` manually.
