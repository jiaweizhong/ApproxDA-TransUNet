"""2-D CNN baselines for the journal tables (--arch unet / attunet / unetpp / resunet), trained from scratch with the
same data pipeline, loss (0.5 CE + 0.5 Dice), optimiser and evaluation as every other model in this project.

    unet     U-Net (Ronneberger et al., MICCAI 2015): 4 poolings, 64 - 1024 channels, two 3x3 conv-BN-ReLU per level,
             2x2 transposed-conv upsampling, concatenated skips.
    attunet  Attention U-Net (Oktay et al., MIDL 2018): the U-Net above with additive attention gates on the skips.
    unetpp   U-Net++ (Zhou et al., DLMIA 2018): nested dense skip pathways, 32 - 512 channels, bilinear upsampling,
             no deep supervision (the final node X^{0,4} is the output).
    resunet  Residual U-Net (Zhang et al., IEEE GRSL 2018): pre-activation residual units, 64 - 512 channels,
             strided-conv downsampling, bilinear upsampling, concatenated skips.

Grayscale inputs are fed as one channel; with --rgb the models take three channels (a one-channel tensor, e.g. the
FLOP-count dummy in test.py, is repeated to three)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class _Base(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

    def _prep(self, x):
        return x.repeat(1, self.in_channels, 1, 1) if x.size(1) == 1 and self.in_channels > 1 else x


class UNetPlusPlus(_Base):
    def __init__(self, in_channels, num_classes, base=32):
        super().__init__(in_channels)
        c = [base * 2 ** i for i in range(5)]
        self.c = c
        self.nodes = nn.ModuleDict()
        for i in range(5):  # X^{i,j}: depth i, j previous nodes at the same depth
            for j in range(5 - i):
                cin = in_channels if (i, j) == (0, 0) else (c[i - 1] if j == 0 else c[i] * j + c[i + 1])
                self.nodes[f"{i}_{j}"] = conv_block(cin, c[i])
        self.head = nn.Conv2d(c[0], num_classes, 1)

    def forward(self, x):
        x = self._prep(x)
        X = {}
        for j in range(5):
            for i in range(5 - j):
                if j == 0:
                    X[i, 0] = self.nodes[f"{i}_0"](x if i == 0 else F.max_pool2d(X[i - 1, 0], 2))
                else:
                    up = F.interpolate(X[i + 1, j - 1], size=X[i, 0].shape[-2:], mode="bilinear", align_corners=True)
                    X[i, j] = self.nodes[f"{i}_{j}"](torch.cat([X[i, k] for k in range(j)] + [up], 1))
        return self.head(X[0, 4])
