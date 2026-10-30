"""
Qualitative comparison figure (paper Fig. 3): Synapse / CVC-ClinicDB / ISIC 2018 rows,
columns Input | Ground truth | DA-TransUNet | ApproxDA-TransUNet.

Models are the final-epoch checkpoints of the runs reported in Tables I-III. Case selection:
  - Synapse ("gain"): among all slices of the 12 test volumes with 2-3 organ classes in the ground
    truth (easy to read), >= 8000 foreground pixels and ApproxDA slice Dice >= 0.80, the slice with
    the largest ApproxDA - DA gain (case0032, slice 131: DA labels the stomach as spleen);
  - CVC-ClinicDB / ISIC 2018 ("median"): the test case whose ApproxDA per-case Dice (from its test
    log) is closest to the median.
Panel labels give the per-image Dice of the displayed image.

    ~/venvs/torch/bin/python make_qualitative_figure.py   # from experiments/
"""

import os
import re
import subprocess
import sys

import numpy as np

EXP = os.path.dirname(os.path.abspath(__file__))
PY = os.path.expanduser("~/venvs/torch/bin/python")
WORK = os.path.join(EXP, "..", "results", "qualitative")
OUT = os.path.join(EXP, "..", "paper", "figures", "fig_qualitative_cross_task")

ROWS = [
    dict(name="Synapse", dataset="Synapse", img_size=224, rgb=False, select="gain",
         approx=dict(ckpt="model/ApproxDA_Synapse224/ApproxDA_pretrain_R50-ViT-B_16_skip3_epo300_bs24_224_M112_pam_c16/epoch_299.pth",
                     log="ApproxDA-TransUNet/test_log/test_log_ApproxDA_Synapse224/ApproxDA_pretrain_R50-ViT-B_16_skip3_epo300_bs24_224_M112_pam_c16.txt",
                     window_size=112, rank=32, groups=8, gate_mode="pam"),
         da="model/TU_Synapse224/TU_pretrain_R50-ViT-B_16_skip3_epo300_bs24_224/epoch_299.pth"),
    dict(name="CVC-ClinicDB", dataset="CVC", img_size=256, rgb=False,
         approx=dict(ckpt="model/ApproxDA_CVC256/ApproxDA_pretrain_R50-ViT-B_16_skip3_epo500_bs24_lr0.001_256_M32_G1_cam_c16_adam_dl/epoch_499.pth",
                     log="ApproxDA-TransUNet/test_log/test_log_ApproxDA_CVC256/ApproxDA_pretrain_R50-ViT-B_16_skip3_epo500_bs24_lr0.001_256_M32_G1_cam_c16_adam_dl.txt",
                     window_size=32, rank=32, groups=1, gate_mode="cam"),
         da="model/TU_CVC256/TU_pretrain_R50-ViT-B_16_skip3_epo500_bs24_lr0.001_256_adam_dl/epoch_499.pth"),
    dict(name="ISIC 2018", dataset="ISIC", img_size=256, rgb=True,
         approx=dict(ckpt="model/ApproxDA_ISIC256/ApproxDA_pretrain_R50-ViT-B_16_skip3_epo50_bs24_256_M64_pam_c16_rgb_dl/epoch_49.pth",
                     log="ApproxDA-TransUNet/test_log/test_log_ApproxDA_ISIC256/ApproxDA_pretrain_R50-ViT-B_16_skip3_epo50_bs24_256_M64_pam_c16_rgb_dl.txt",
                     window_size=64, rank=32, groups=8, gate_mode="pam"),
         da="model/TU_ISIC256/TU_pretrain_R50-ViT-B_16_skip3_epo50_bs24_256_rgb_dl/epoch_49.pth"),
]

SYNAPSE_COLORS = np.array([[0, 0, 0], [255, 100, 100], [100, 200, 100], [100, 100, 255], [255, 165, 0],
                           [180, 0, 180], [0, 200, 200], [255, 215, 0], [0, 150, 255]]) / 255.0
BINARY_COLOR = np.array([255, 200, 50]) / 255.0


def median_case(log_path):
    """Per-case Dice from a test log (last occurrence wins) -> case closest to the median."""
    per_case = {}
    for line in open(os.path.join(EXP, log_path)):
        m = re.search(r"idx \d+ case (\S+) mean_dice ([0-9.]+)", line)
        if m:
            per_case[m.group(1)] = float(m.group(2))
    med = float(np.median(list(per_case.values())))
    case = min(per_case, key=lambda c: abs(per_case[c] - med))
    print(f"  {log_path.split('/')[-1]}: {len(per_case)} cases, median {100 * med:.2f} -> {case}")
    return case


REUSE = "--reuse" in sys.argv  # reuse the cached predictions in WORK (only relabelling / restyling the figure)


def predict(row, project, case, extra=()):
    cached = os.path.join(WORK, f"Synapse_slices_{project}.json" if "--all_slices" in extra else f"{row['dataset']}_{case}_{project}.npz")
    if REUSE and os.path.exists(cached):
        return None if "--all_slices" in extra else dict(np.load(cached))
    cmd = [PY, os.path.join(EXP, "qualitative_predict.py"), "--project", project, "--dataset", row["dataset"],
           "--cases", case, "--img_size", str(row["img_size"]), "--out_dir", WORK, *extra]
    if row["rgb"]:
        cmd.append("--rgb")
    if project == "ApproxDA-TransUNet":
        a = row["approx"]
        cmd += ["--ckpt", os.path.join(EXP, a["ckpt"]), "--window_size", str(a["window_size"]),
                "--rank", str(a["rank"]), "--groups", str(a["groups"]), "--gate_mode", a["gate_mode"]]
    else:
        cmd += ["--ckpt", os.path.join(EXP, row["da"])]
    subprocess.run(cmd, check=True)
    if "--all_slices" in extra:
        return None
    return dict(np.load(os.path.join(WORK, f"{row['dataset']}_{case}_{project}.npz")))


def gain_slice(row, n_cls=(2, 3), min_fg=8000, min_dice=0.80):
    """Synapse: (case, slice) with the largest ApproxDA - DA slice Dice among slices with few organs."""
    import json

    cases = ",".join(l.strip() for l in open(os.path.join(EXP, "ApproxDA-TransUNet/lists/lists_Synapse/test_vol.txt")) if l.strip())
    for project in ("DA-TransUNet", "ApproxDA-TransUNet"):
        predict(row, project, cases, extra=("--all_slices",))
    da = json.load(open(os.path.join(WORK, "Synapse_slices_DA-TransUNet.json")))
    ap = json.load(open(os.path.join(WORK, "Synapse_slices_ApproxDA-TransUNet.json")))
    cand = [(a["dice"] - d["dice"], c, d["slice"]) for c in da for d, a in zip(da[c], ap[c])
            if d["n_cls"] in n_cls and d["fg"] >= min_fg and a["dice"] >= min_dice]
    cand.sort(reverse=True)
    for g, c, sl in cand[:5]:
        print(f"  top gain: {c} slice {sl} (+{100 * g:.1f})")
    return cand[0][1], cand[0][2]


def to_rgb(image):
    img = image if image.ndim == 3 else np.repeat(image[..., None], 3, axis=2)
    lo, hi = img.min(), img.max()
    return (img - lo) / (hi - lo + 1e-8)


def overlay(image, seg, binary, alpha=0.55):
    out = to_rgb(image).copy()
    for c in range(1, int(seg.max()) + 1):
        m = seg == c
        if m.any():
            color = BINARY_COLOR if binary else SYNAPSE_COLORS[c]
            out[m] = (1 - alpha) * out[m] + alpha * color
    return out


def main():
    os.makedirs(WORK, exist_ok=True)
    panels = []
    for row in ROWS:
        print(row["name"])
        if row.get("select") == "gain":
            case, sl = gain_slice(row)
            extra = ("--slice", str(sl))
        else:
            case, extra = median_case(row["approx"]["log"]), ()
        da = predict(row, "DA-TransUNet", case, extra)
        ap = predict(row, "ApproxDA-TransUNet", case, extra)
        assert np.array_equal(da["label"], ap["label"])
        panels.append((row, case, da, ap))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import patheffects

    # pdf.fonttype 42: TrueType instead of matplotlib's default Type 3 fonts (IEEE PDF eXpress)
    plt.rcParams.update({"font.size": 7, "font.family": "serif", "pdf.fonttype": 42, "ps.fonttype": 42})
    # manual layout in inches: every cell has its image's aspect ratio and a fixed gap, so there is no padding
    ratios = [p[2]["label"].shape[0] / p[2]["label"].shape[1] for p in panels]
    W, gap, title_h = 3.45, 0.03, 0.14  # \columnwidth, gap between panels, column-title band
    wc = (W - 3 * gap) / 4
    H = title_h + wc * sum(ratios) + gap * (len(panels) - 1)
    fig = plt.figure(figsize=(W, H))

    def cell(r, c):
        top = H - title_h - wc * sum(ratios[:r]) - gap * r
        return fig.add_axes([c * (wc + gap) / W, (top - wc * ratios[r]) / H, wc / W, wc * ratios[r] / H])

    titles = ["Input", "Ground truth", "DA-TransUNet", "Ours"]
    stroke = [patheffects.withStroke(linewidth=1.5, foreground="black")]
    for r, (row, case, da, ap) in enumerate(panels):
        binary = row["dataset"] != "Synapse"
        imgs = [to_rgb(da["image"]), overlay(da["image"], da["label"], binary),
                overlay(da["image"], da["pred"], binary), overlay(ap["image"], ap["pred"], binary)]
        if not binary:  # Synapse h5 slices are stored transposed; show the standard axial view
            imgs = [im.transpose(1, 0, 2) for im in imgs]
        for c, im in enumerate(imgs):
            ax = cell(r, c)
            ax.imshow(im, interpolation="nearest")
            ax.set_xticks([]), ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if r == 0:
                ax.set_title(titles[c], fontsize=7, pad=2)
            if c >= 2:
                d = (da if c == 2 else ap)["dice"]
                ax.text(0.97, 0.04, f"{100 * d:.1f}", transform=ax.transAxes, ha="right", va="bottom",
                        color="white", fontsize=6.5, path_effects=stroke)
        sl = f", slice {int(da['slice'])}" if int(da["slice"]) >= 0 else ""
        print(f"  {row['name']}: case {case}{sl}  DA {100 * float(da['dice']):.2f}  ApproxDA {100 * float(ap['dice']):.2f}")
    # crop all outer whitespace
    # 600 dpi: the panels are resampled at the savefig dpi; at 300 they end up at ~297 ppi after LaTeX scales the
    # figure to \columnwidth, below IEEE's 300-dpi minimum
    fig.savefig(OUT + ".pdf", dpi=600, bbox_inches="tight", pad_inches=0)
    fig.savefig(OUT + ".png", dpi=300, bbox_inches="tight", pad_inches=0)
    print("written:", OUT + ".pdf")


if __name__ == "__main__":
    sys.exit(main())
