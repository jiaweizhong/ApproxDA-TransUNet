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


class UNet(_Base):
    def __init__(self, in_channels, num_classes, base=64):
        super().__init__(in_channels)
        c = [base * 2 ** i for i in range(5)]
        self.enc = nn.ModuleList([conv_block(in_channels, c[0])] + [conv_block(c[i], c[i + 1]) for i in range(4)])
        self.up = nn.ModuleList([nn.ConvTranspose2d(c[i + 1], c[i], 2, stride=2) for i in reversed(range(4))])
        self.dec = nn.ModuleList([conv_block(2 * c[i], c[i]) for i in reversed(range(4))])
        self.head = nn.Conv2d(c[0], num_classes, 1)

    def gate(self, i, g, skip):  # identity for U-Net, attention gate for Attention U-Net
        return skip

    def forward(self, x):
        x = self._prep(x)
        skips = []
        for i, enc in enumerate(self.enc):
            x = enc(x if i == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        x = skips.pop()
        for i, (up, dec) in enumerate(zip(self.up, self.dec)):
            g = up(x)
            x = dec(torch.cat([self.gate(i, g, skips[-1 - i]), g], 1))
        return self.head(x)
