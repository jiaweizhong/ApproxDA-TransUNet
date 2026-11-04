"""
Qualitative figure of the journal: one test example per dataset (Synapse, ACDC, CVC-ClinicDB, ISIC 2018, FeTA 2.4),
columns image / ground truth / DA-TransUNet / ApproxDA-TransUNet ("Ours"; seed 1234, final epoch).

The example shows where ApproxDA-TransUNet improves on DA-TransUNet, without picking failures of either model:
  - candidates: every test image, or every slice of a test volume, whose ground truth contains at least half of the
    dataset's foreground classes and covers at least 1% of the image, and on which both models reach a Dice of 0.5;
    FeTA sub-052 (out-of-distribution ventriculomegaly, discussed in the paper) is excluded;
  - the candidate with the largest Dice gain of ApproxDA-TransUNet over DA-TransUNet is shown.
Each panel is cropped to the ground-truth foreground box (with a 35% margin), identically for all columns, and the Dice of
the displayed slice / image (mean over the classes present in its ground truth) is printed on the prediction panels.

Inputs: experiments/predictions/qualitative_journal/<project>_<dataset>/ from qualitative_journal_predict.py.
Run from experiments/ApproxDA-TransUNet (for its datasets package):
    python ../make_qualitative_figure_journal.py OUT.pdf
"""

import json
import os
import sys

sys.path.insert(0, os.getcwd())

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from datasets.dataset_acdc import ACDC_dataset  # noqa: E402
from datasets.dataset_cvc import CVC_dataset  # noqa: E402
from datasets.dataset_feta import FeTA_dataset  # noqa: E402
from datasets.dataset_isic import ISIC_dataset  # noqa: E402
from datasets.dataset_synapse import Synapse_dataset  # noqa: E402

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 9})
PRED = "../predictions/qualitative_journal"
DATA = [  # name, dataset class, base dir, list dir, rgb
    ("Synapse", Synapse_dataset, "../data/Synapse/test_vol_h5", "./lists/lists_Synapse", False),
    ("ACDC", ACDC_dataset, "../data/ACDC_MTUNet", "./lists/lists_ACDC", False),
    ("CVC", CVC_dataset, "../data/CVC-ClinicDB", "./lists/lists_CVC", False),
    ("ISIC", ISIC_dataset, "../data/ISIC2018", "./lists/lists_ISIC", True),
    ("FeTA", FeTA_dataset, "../data/FeTA_2d/test_vol_h5", "./lists/lists_FeTA", False),
]
NCLS = {"Synapse": 8, "ACDC": 3, "CVC": 1, "ISIC": 1, "FeTA": 7}
EXCLUDE = {"sub-052"}
LABEL = {"Synapse": "Synapse", "ACDC": "ACDC", "CVC": "CVC-ClinicDB", "ISIC": "ISIC 2018", "FeTA": "FeTA 2.4"}
COLORS = np.array([[0, 0, 0], [230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200], [245, 130, 48],
                   [145, 30, 180], [70, 240, 240], [240, 50, 230]], float) / 255


def slice_dice(pred, label):
    ds = [2 * ((pred == c) & (label == c)).sum() / ((pred == c).sum() + (label == c).sum())
          for c in np.unique(label) if c != 0]
    return float(np.mean(ds)) if ds else float("nan")


def to_rgb(img):
    img = img.astype(float)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return img if img.ndim == 3 else np.repeat(img[..., None], 3, -1)


def overlay(img, seg, alpha=0.5):
    out = img.copy()
    m = seg > 0
    out[m] = (1 - alpha) * out[m] + alpha * COLORS[seg[m]]
    return out


def crop_box(label, margin=0.35):
    ys, xs = np.nonzero(label)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    c, r = ((y0 + y1) / 2, (x0 + x1) / 2), max(y1 - y0, x1 - x0) / 2 * (1 + 2 * margin)
    H, W = label.shape
    r = min(r, H / 2, W / 2)
    cy, cx = min(max(c[0], r), H - r), min(max(c[1], r), W - r)
    return slice(int(cy - r), int(cy + r)), slice(int(cx - r), int(cx + r))


def case_id(name, case, k):
    """Short identifier of the shown example, printed under the dataset name."""
    if name == "ACDC":  # case_002_volume_ES -> patient 002 ES
        _, pid, _, phase = case.split("_")
        case = f"patient {pid} {phase}"
    elif name == "CVC":
        case = f"image {case}"
    return case if k is None else f"{case}, slice {k}"


def main():
    out = sys.argv[1]
    fig, axes = plt.subplots(len(DATA), 4, figsize=(7.2, 1.85 * len(DATA)))
    log = {}
    for row, (name, cls, base, lst, rgb) in enumerate(DATA):
        db = cls(base_dir=base, list_dir=lst, split="test_vol", **({"rgb": True} if rgb else {}))
        best = None
        for i in range(len(db)):
            s = db[i]
            case = s["case_name"]
            if case in EXCLUDE:
                continue
            vol_label = np.asarray(s["label"]).astype(int)
            vol_preds = {p: np.load(f"{PRED}/{p}_{name}/{case}.npz")["pred"].astype(int)
                         for p in ("DA-TransUNet", "ApproxDA-TransUNet")}
            for k in (range(vol_label.shape[0]) if vol_label.ndim == 3 else [None]):
                lab = vol_label if k is None else vol_label[k]
                if len(np.unique(lab)) - 1 < (NCLS[name] + 1) // 2 or (lab > 0).mean() < 0.01:
                    continue
                pr = {p: (v if k is None else v[k]) for p, v in vol_preds.items()}
                d = {p: slice_dice(v, lab) for p, v in pr.items()}
                if min(d.values()) < 0.5:
                    continue
                gain = d["ApproxDA-TransUNet"] - d["DA-TransUNet"]
                if best is None or gain > best[0]:
                    img = np.asarray(s["image"]) if k is None else np.asarray(s["image"])[k]
                    best = (gain, case, k, img, lab, pr)
        _, case, k, image, label, preds = best
        ys, xs = crop_box(label)
        img = to_rgb(image)[ys, xs]
        panels = [("Image", img, None), ("Ground truth", overlay(img, label[ys, xs]), None)]
        for p, title in (("DA-TransUNet", "DA-TransUNet"), ("ApproxDA-TransUNet", "Ours")):
            panels.append((title, overlay(img, preds[p][ys, xs]), slice_dice(preds[p], label)))
        log[name] = {"case": case, "slice": k,
                     **{p: slice_dice(preds[p], label) for p in preds}}
        for col, (title, im, d) in enumerate(panels):
            ax = axes[row, col]
            ax.imshow(im, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(title, fontsize=12)
            if col == 0:
                ax.set_ylabel(LABEL[name], fontsize=12, labelpad=19)
                ax.text(-0.035, 0.5, case_id(name, case, k), transform=ax.transAxes, rotation=90, ha="right",
                        va="center", fontsize=8.5, color="#444444")
            if d is not None:
                ax.text(0.03, 0.04, f"DSC {d * 100:.1f}", transform=ax.transAxes, color="white", fontsize=9.5,
                        bbox=dict(facecolor="black", alpha=0.6, pad=1.5, linewidth=0))
    plt.subplots_adjust(wspace=0.03, hspace=0.06, left=0.09, right=0.995, top=0.955, bottom=0.005)
    fig.savefig(out, dpi=300)
    json.dump(log, open(os.path.splitext(out)[0] + "_selection.json", "w"), indent=1)
    print(json.dumps(log, indent=1))


if __name__ == "__main__":
    main()
