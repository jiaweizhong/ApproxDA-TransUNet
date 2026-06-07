"""Smoke test for AdaDA-TransUNet modules. Run from Ada-DA-TransUNet/."""
import torch
from Architecture.block import (
    AdaDABlock, LowRankWindowedPAM, GroupedCAM,
    hardware_config, window_partition, window_reverse,
)

print("=== window_partition / window_reverse ===")
x = torch.randn(2, 64, 112, 112)
xw = window_partition(x, 7)
assert xw.shape == (2 * 16 * 16, 64, 7, 7), xw.shape
xr = window_reverse(xw, 7, 112, 112)
assert xr.shape == x.shape
print("  PASS")

print("=== LowRankWindowedPAM ===")
for C, H in [(768, 14), (512, 28), (256, 56), (64, 112)]:
    pam = LowRankWindowedPAM(C, window_size=7, rank=32)
    y = pam(torch.randn(2, C, H, H))
    assert y.shape == (2, C, H, H), f"PAM {C}@{H}: {y.shape}"
    print(f"  C={C} H={H} PASS")

print("=== GroupedCAM ===")
for C, H in [(768, 14), (512, 28), (256, 56), (64, 112)]:
    cam = GroupedCAM(C, groups=8)
    y = cam(torch.randn(2, C, H, H))
    assert y.shape == (2, C, H, H), f"CAM {C}@{H}: {y.shape}"
    print(f"  C={C} H={H} PASS")

print("=== AdaDABlock ===")
for C, H in [(768, 14), (512, 28), (256, 56), (64, 112)]:
    blk = AdaDABlock(C, window_size=7, rank=32, groups=8)
    y = blk(torch.randn(2, C, H, H))
    assert y.shape == (2, C, H, H), f"AdaDA {C}@{H}: {y.shape}"
    print(f"  C={C} H={H} PASS")

print("=== hardware_config ===")
assert hardware_config(10) == {"rank": 64, "window_size": 14, "groups": 4}
assert hardware_config(6)  == {"rank": 32, "window_size": 7,  "groups": 8}
assert hardware_config(2)  == {"rank": 16, "window_size": 7,  "groups": 16}
print("  PASS")

print("=== Full model forward pass ===")
from Architecture.AdaDATransUNet import CONFIGS, AdaDATransUNet
cfg = CONFIGS['R50-ViT-B_16']
cfg.n_classes = 9
cfg.n_skip = 3
cfg.patches.grid = (14, 14)
net = AdaDATransUNet(cfg, img_size=224, num_classes=9)
x = torch.randn(1, 3, 224, 224)
out = net(x)
assert out.shape == (1, 9, 224, 224), f"Model output shape: {out.shape}"
print(f"  Output shape: {out.shape}  PASS")

print("\nAll tests passed.")
