"""
PAM contribution maps: windowed vs map-wide PAM (paper Fig. 5 and its statistics).

For every LowRankWindowedPAM module a forward hook records ||output - input||_2 over channels,
i.e. the per-pixel magnitude of the attention term alpha * E. The maps of all blocks are
upsampled to the input size and averaged. On every test image the script reports the share of
this contribution that falls inside the ground-truth mask ("mass on target"), for each model,
and how many images have a larger share under the first (windowed) model.

Inputs follow test.py / utils.test_single_volume (resize to img_size with order 3, no other
normalisation); checkpoints are loaded strictly (final epoch of the c16 runs).

    python analyze_pam_contribution.py --dataset CVC --img_size 256 --windows 8 128 \
        --ckpts ../model/ApproxDA_CVC256/..._M8_pam_c16_adam_dl/epoch_499.pth \
                ../model/ApproxDA_CVC256/..._M128_pam_c16_adam_dl/epoch_499.pth \
        --out ../../paper/figures/attn_CVC_M8_vs_M128
"""

import argparse

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import zoom

from Architecture.ApproxDATransUNet import ApproxDATransUNet, CONFIGS
from Architecture.block import LowRankWindowedPAM

DATA = {
    "CVC": ("datasets.dataset_cvc", "CVC_dataset", "../data/CVC-ClinicDB", "./lists/lists_CVC"),
    "ISIC": ("datasets.dataset_isic", "ISIC_dataset", "../data/ISIC2018", "./lists/lists_ISIC"),
}

p = argparse.ArgumentParser()
p.add_argument("--dataset", default="CVC", choices=list(DATA))
p.add_argument("--img_size", type=int, default=256)
p.add_argument("--rgb", action="store_true")
p.add_argument("--windows", type=int, nargs=2, required=True, help="windowed M, map-wide M")
p.add_argument("--ckpts", nargs=2, required=True)
p.add_argument("--rank", type=int, default=32)
p.add_argument("--n_show", type=int, default=3)
p.add_argument("--out", required=True, help="output path without extension (.pdf and .png are written)")
args = p.parse_args()

mod, cls, base_dir, list_dir = DATA[args.dataset]
Dataset = getattr(__import__(mod, fromlist=[cls]), cls)
db = Dataset(base_dir=base_dir, list_dir=list_dir, split="test_vol", **({"rgb": True} if args.rgb else {}))


def build(window, ckpt):
    cfg = CONFIGS["R50-ViT-B_16"]
    cfg.n_classes, cfg.n_skip = 2, 3
    cfg.window_size, cfg.rank, cfg.groups, cfg.gate_mode, cfg.block_version = window, args.rank, 8, "pam", "c16"
    cfg.patches.size = (16, 16)
    cfg.patches.grid = (args.img_size // 16, args.img_size // 16)
    net = ApproxDATransUNet(cfg, img_size=args.img_size, num_classes=2).cuda()
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    net.load_state_dict({k[len("module."):] if k.startswith("module.") else k: v for k, v in state.items()}, strict=True)
    net.eval()
    maps = []
    for m in net.modules():
        if isinstance(m, LowRankWindowedPAM):
            m.register_forward_hook(lambda _m, inp, out: maps.append((out - inp[0]).norm(dim=1, keepdim=True)))
    return net, maps


def contribution(net, maps, image):
    s = args.img_size
    x, y = image.shape[:2]
    img = zoom(image, (s / x, s / y) + (1,) * (image.ndim - 2), order=3)
    t = torch.from_numpy(img).float()
    t = (t.permute(2, 0, 1) if t.ndim == 3 else t.unsqueeze(0)).unsqueeze(0).cuda()
    maps.clear()
    with torch.no_grad():
        pred = net(t).argmax(1)[0].cpu().numpy()
    up = [F.interpolate(m, size=(s, s), mode="bilinear", align_corners=False) for m in maps]
    return torch.cat(up, 1).mean(1)[0].cpu().numpy(), img, pred


models = [build(w, c) for w, c in zip(args.windows, args.ckpts)]
records = []
for i in range(len(db)):
    sample = db[i]
    image, label = np.asarray(sample["image"], dtype=np.float32), np.asarray(sample["label"])
    gt = zoom(label, (args.img_size / label.shape[0], args.img_size / label.shape[1]), order=0) > 0
    rec = dict(case=sample["case_name"], gt=gt, area=gt.mean())
    for k, (net, maps) in enumerate(models):
        cmap, img, pred = contribution(net, maps, image)
        rec[f"map{k}"], rec[f"pred{k}"], rec["img"] = cmap, pred, img
        rec[f"share{k}"] = float(cmap[gt].sum() / cmap.sum())
    records.append(rec)

share = np.array([[r["share0"], r["share1"]] for r in records])
area = np.array([r["area"] for r in records])
n_more = int((share[:, 0] > share[:, 1]).sum())
print(f"{args.dataset}: {len(records)} test images, mean target area {100 * area.mean():.1f}%")
for k, w in enumerate(args.windows):
    print(f"  M={w}: mean share of PAM contribution on target {100 * share[:, k].mean():.1f}% "
          f"(median {100 * np.median(share[:, k]):.1f}%)")
print(f"  M={args.windows[0]} > M={args.windows[1]} on {n_more}/{len(records)} images")

# figure: images at the 25th / 50th / 75th percentile of target area (independent of the models)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

order = np.argsort(area)
picks = [order[int(q * (len(order) - 1))] for q in np.linspace(0.25, 0.75, args.n_show)]
plt.rcParams.update({"font.size": 7, "font.family": "serif"})
width = 3.45
fig, axes = plt.subplots(len(picks), 3, figsize=(width, width / 3 * len(picks) + 0.35),
                         gridspec_kw=dict(wspace=0.03, hspace=0.04, left=0.0, right=1.0, top=0.93, bottom=0.09))
titles = ["Input + ground truth", f"$M{{=}}{args.windows[0]}$ (windowed)", f"$M{{=}}{args.windows[1]}$ (map-wide)"]
for row, i in enumerate(picks):
    r = records[i]
    disp = r["img"] if r["img"].ndim == 3 else np.repeat(r["img"][..., None], 3, 2)
    disp = (disp - disp.min()) / (disp.max() - disp.min() + 1e-8)
    for c in range(3):
        ax = axes[row, c]
        if c == 0:
            ax.imshow(disp)
        else:
            m = r[f"map{c - 1}"]
            ax.imshow(disp.mean(2), cmap="gray")
            im = ax.imshow((m - m.min()) / (m.max() - m.min() + 1e-8), cmap="hot", alpha=0.6, vmin=0, vmax=1)
            ax.text(0.97, 0.04, f"{100 * r[f'share{c - 1}']:.0f}%", transform=ax.transAxes, ha="right",
                    va="bottom", color="white", fontsize=6.5)
        ax.contour(r["gt"], levels=[0.5], colors="lime", linewidths=0.8)
        ax.set_xticks([]), ax.set_yticks([])
        if row == 0:
            ax.set_title(titles[c], fontsize=7, pad=2)
cax = fig.add_axes([0.25, 0.035, 0.5, 0.018])
cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[0, 0.5, 1])
cb.ax.tick_params(labelsize=6, length=2, pad=1)
fig.savefig(args.out + ".pdf", dpi=300)
fig.savefig(args.out + ".png", dpi=300)
print("written:", args.out + ".pdf", "cases:", [records[i]["case"] for i in picks])
