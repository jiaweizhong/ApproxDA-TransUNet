"""
Predict selected test cases with one trained model (helper for make_qualitative_figure.py).

Runs inside one project directory (ApproxDA-TransUNet or DA-TransUNet) so that project's own
Architecture / datasets packages are imported; the two projects cannot share one process.
Preprocessing and prediction follow utils.test_single_volume exactly (resize to img_size with
order 3, argmax, nearest-neighbour resize back to the label size). For Synapse, the displayed
slice is chosen from the ground truth only (most organ classes, then largest foreground), so it
is the same for every model.

Writes one .npz per case: image, label, pred, slice (Synapse) and the per-image Dice of the
displayed slice / image (mean over foreground classes present in the label).
"""

import argparse
import os
import sys

import numpy as np
import torch
from scipy.ndimage import zoom

p = argparse.ArgumentParser()
p.add_argument("--project", required=True, choices=["ApproxDA-TransUNet", "DA-TransUNet"])
p.add_argument("--dataset", required=True, choices=["Synapse", "CVC", "ISIC"])
p.add_argument("--ckpt", required=True)
p.add_argument("--cases", required=True, help="comma-separated case names")
p.add_argument("--img_size", type=int, required=True)
p.add_argument("--rgb", action="store_true")
p.add_argument("--window_size", type=int, default=7)
p.add_argument("--rank", type=int, default=32)
p.add_argument("--groups", type=int, default=8)
p.add_argument("--gate_mode", default="pam")
p.add_argument("--out_dir", required=True)
p.add_argument("--slice", type=int, default=-1, help="Synapse: show this slice instead of the GT-based choice")
p.add_argument("--all_slices", action="store_true",
               help="Synapse: write the per-slice Dice of every slice of the given cases to a .json and exit")
args = p.parse_args()

proj = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.project)
os.chdir(proj)
sys.path.insert(0, proj)

DATA = {
    "Synapse": ("datasets.dataset_synapse", "Synapse_dataset", "../data/Synapse/test_vol_h5", "./lists/lists_Synapse", 9),
    "CVC": ("datasets.dataset_cvc", "CVC_dataset", "../data/CVC-ClinicDB", "./lists/lists_CVC", 2),
    "ISIC": ("datasets.dataset_isic", "ISIC_dataset", "../data/ISIC2018", "./lists/lists_ISIC", 2),
}
mod_name, cls_name, base_dir, list_dir, num_classes = DATA[args.dataset]
Dataset = getattr(__import__(mod_name, fromlist=[cls_name]), cls_name)
db = Dataset(base_dir=base_dir, list_dir=list_dir, split="test_vol", **({"rgb": True} if args.rgb else {}))

if args.project == "ApproxDA-TransUNet":
    from Architecture.ApproxDATransUNet import ApproxDATransUNet as Net, CONFIGS
else:
    from Architecture.DATransUNet import DA_Transformer as Net, CONFIGS
cfg = CONFIGS["R50-ViT-B_16"]
cfg.n_classes = num_classes
cfg.n_skip = 3
cfg.patches.size = (16, 16)
cfg.patches.grid = (args.img_size // 16, args.img_size // 16)
if args.project == "ApproxDA-TransUNet":
    cfg.window_size, cfg.rank, cfg.groups = args.window_size, args.rank, args.groups
    cfg.gate_mode, cfg.block_version = args.gate_mode, "c16"
net = Net(cfg, img_size=args.img_size, num_classes=num_classes).cuda()
state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
state = {(k[len("module."):] if k.startswith("module.") else k).replace(".ada_block.", ".approx_block."): v
         for k, v in state.items()}
net.load_state_dict(state, strict=True)
net.eval()


def predict_2d(image):
    """image: (H, W) or (H, W, 3) float in [0, 1] -> (H, W) label map, as in test_single_volume."""
    x, y = image.shape[:2]
    s = args.img_size
    img = zoom(image, (s / x, s / y) + (1,) * (image.ndim - 2), order=3) if (x, y) != (s, s) else image
    t = torch.from_numpy(img).float()
    t = (t.permute(2, 0, 1) if t.ndim == 3 else t.unsqueeze(0)).unsqueeze(0).cuda()
    with torch.no_grad():
        out = torch.argmax(torch.softmax(net(t), dim=1), dim=1).squeeze(0).cpu().numpy()
    return zoom(out, (x / s, y / s), order=0) if (x, y) != (s, s) else out


def dice(pred, label):
    """Mean Dice over foreground classes present in the label (calculate_metric_percase convention)."""
    ds = []
    for c in range(1, num_classes):
        g, q = label == c, pred == c
        if g.sum() == 0:
            continue
        ds.append(0.0 if q.sum() == 0 else 2.0 * (g & q).sum() / (g.sum() + q.sum()))
    return float(np.mean(ds)) if ds else float("nan")


names = [l.strip() for l in db.sample_list]
os.makedirs(args.out_dir, exist_ok=True)

if args.all_slices:
    import json

    table = {}
    for case in args.cases.split(","):
        sample = db[names.index(case)]
        image, label = np.asarray(sample["image"]), np.asarray(sample["label"])
        table[case] = [dict(slice=k, dice=dice(predict_2d(image[k].astype(np.float32)), label[k]),
                            n_cls=int(len(np.unique(label[k][label[k] > 0]))), fg=int((label[k] > 0).sum()))
                       for k in range(label.shape[0]) if (label[k] > 0).any()]
    out = os.path.join(args.out_dir, f"{args.dataset}_slices_{args.project}.json")
    json.dump(table, open(out, "w"))
    print("written:", out)
    sys.exit(0)
for case in args.cases.split(","):
    sample = db[names.index(case)]
    image, label = np.asarray(sample["image"]), np.asarray(sample["label"])
    sl = -1
    if label.ndim == 3 and args.slice >= 0:
        sl = args.slice
        image, label = image[sl], label[sl]
    elif label.ndim == 3:  # Synapse volume: pick the slice from the ground truth only
        n_cls = [len(np.unique(label[k][label[k] > 0])) for k in range(label.shape[0])]
        fg = [(label[k] > 0).sum() for k in range(label.shape[0])]
        sl = max(range(label.shape[0]), key=lambda k: (n_cls[k], fg[k]))
        image, label = image[sl], label[sl]
    pred = predict_2d(image.astype(np.float32))
    d = dice(pred, label)
    out = os.path.join(args.out_dir, f"{args.dataset}_{case}_{args.project}.npz")
    np.savez_compressed(out, image=image, label=label, pred=pred.astype(np.uint8), slice=sl, dice=d)
    print(f"{args.dataset} {case} slice={sl} {args.project}: Dice {100 * d:.2f}")
