"""Where do the segmentation errors come from? Per organ, from the label maps test.py saves for AMOS
(../predictions/<exp>/<run>_npz/<case>.npz) and the ground-truth test volumes.

For every organ c and case:
  * false positives (pred c, truth not c) and false negatives (truth c, pred not c), in mL;
  * their 3-D distance to the ground-truth surface of c, in mm (FP: distance outside the organ, FN: depth inside it;
    an FP of an organ absent from the case counts as "absent"), binned <=2 / 2-5 / 5-15 / >15 mm;
  * z-extent errors: FP voxels on slices where the truth has no c (organ predicted beyond its ends) and FN voxels on
    slices where the prediction has no c (organ ends missed);
  * resolution floor: the ground truth resized slice-wise to 224 x 224 and back with nearest neighbour, as the test
    pipeline does with the prediction; its Dice is the best a perfect 224 x 224 prediction could reach.
Prints, per organ: mean Dice (cases containing it), its share of the mean-Dice deficit, error volume, FP / FN split,
distance bins, z-extent share and the resolution-floor Dice; then the same bins pooled over all organs.

Usage (cwd = experiments/ApproxDA-TransUNet):
    python analyze_error_sources.py ../predictions/ApproxDA_AMOS224/<run>_npz [--out summary.json]
"""
import argparse
import json
import multiprocessing as mp
import os
import sys

import h5py
import numpy as np
from scipy import ndimage
from scipy.ndimage import zoom

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datasets.dataset_amos import AMOS_CLASSES  # noqa: E402

BINS = (2.0, 5.0, 15.0)  # mm
BIN_NAMES = ("<=2mm", "2-5mm", "5-15mm", ">15mm", "absent")


def roundtrip(lab, size=224):
    out = np.empty_like(lab)
    for z in range(lab.shape[0]):
        x, y = lab[z].shape
        out[z] = zoom(zoom(lab[z], (size / x, size / y), order=0), (x / size, y / size), order=0)
    return out


def dice(a, b):
    s = a.sum() + b.sum()
    return 2.0 * (a & b).sum() / s if s else np.nan


def analyse(job):
    case, pred_path, gt_path = job
    with h5py.File(gt_path) as f:
        lab, sp = f["label"][:], f["spacing"][:]
    pred = np.load(pred_path)["prediction"]
    rt = roundtrip(lab)
    vox_ml = float(np.prod(sp)) / 1000.0
    pad = np.ceil(BINS[-1] / sp).astype(int) + 1
    res = {}
    for c in range(1, len(AMOS_CLASSES) + 1):
        g, p = lab == c, pred == c
        r = {"present": bool(g.any()), "dice": dice(g, p) if g.any() else np.nan,
             "rt_dice": dice(g, rt == c) if g.any() else np.nan,
             "fp_ml": 0.0, "fn_ml": 0.0, "fp_bins": [0.0] * 5, "fn_bins": [0.0] * 5, "fp_z_ml": 0.0, "fn_z_ml": 0.0}
        fp, fn = p & ~g, g & ~p
        if fp.any() or fn.any():
            r["fp_ml"], r["fn_ml"] = float(fp.sum()) * vox_ml, float(fn.sum()) * vox_ml
            g_z, p_z = g.reshape(g.shape[0], -1).any(1), p.reshape(p.shape[0], -1).any(1)
            r["fp_z_ml"] = float(fp[~g_z].sum()) * vox_ml
            r["fn_z_ml"] = float(fn[~p_z].sum()) * vox_ml
            if not g.any():
                r["fp_bins"][4] = r["fp_ml"]
            else:
                box = tuple(slice(max(int(i.min()) - q, 0), int(i.max()) + q + 1) for i, q in zip(np.nonzero(g | p), pad))
                gb = g[box]
                if fp.any():
                    d = ndimage.distance_transform_edt(~gb, sampling=sp)[fp[box]]
                    r["fp_bins"][:4] = [float(n) * vox_ml for n in np.bincount(np.searchsorted(BINS, d), minlength=4)]
                if fn.any():
                    d = ndimage.distance_transform_edt(gb, sampling=sp)[fn[box]]
                    r["fn_bins"][:4] = [float(n) * vox_ml for n in np.bincount(np.searchsorted(BINS, d), minlength=4)]
        res[AMOS_CLASSES[c - 1]] = r
    return case, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pred_dir")
    ap.add_argument("--gt_dir", default="../data/AMOS_CT_2d/test_vol_h5")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="", help="JSON output (default: <pred_dir>_errsrc.json)")
    a = ap.parse_args()
    cases = sorted(f[:-4] for f in os.listdir(a.pred_dir) if f.endswith(".npz"))
    jobs = [(c, os.path.join(a.pred_dir, c + ".npz"), os.path.join(a.gt_dir, c + ".npy.h5")) for c in cases]
    with mp.get_context("fork").Pool(a.workers) as pool:
        R = dict(pool.map(analyse, jobs))

    rows, tot_fp, tot_fn = {}, np.zeros(5), np.zeros(5)
    tot_err = tot_z = 0.0
    for name in AMOS_CLASSES:
        rs = [R[c][name] for c in R]
        dice_m = np.nanmean([r["dice"] for r in rs])
        rt_m = np.nanmean([r["rt_dice"] for r in rs])
        fp, fn = sum(r["fp_ml"] for r in rs), sum(r["fn_ml"] for r in rs)
        fpb, fnb = np.sum([r["fp_bins"] for r in rs], 0), np.sum([r["fn_bins"] for r in rs], 0)
        z = sum(r["fp_z_ml"] + r["fn_z_ml"] for r in rs)
        tot_fp += fpb; tot_fn += fnb; tot_err += fp + fn; tot_z += z
        rows[name] = {"dice": round(100 * dice_m, 2), "deficit_pts": round(100 * (1 - dice_m) / len(AMOS_CLASSES), 2),
                      "rt_dice": round(100 * rt_m, 2), "err_ml": round(fp + fn, 1), "fp_pct": round(100 * fp / max(fp + fn, 1e-9), 1),
                      "bins_pct": [round(100 * v / max(fp + fn, 1e-9), 1) for v in fpb + fnb],
                      "z_extent_pct": round(100 * z / max(fp + fn, 1e-9), 1)}
    summary = {"pred_dir": os.path.abspath(a.pred_dir), "cases": len(R), "bins": BIN_NAMES, "organs": rows,
               "all": {"err_ml": round(tot_err, 1), "fp_pct": round(100 * tot_fp.sum() / tot_err, 1),
                       "bins_pct": [round(100 * v / tot_err, 1) for v in tot_fp + tot_fn],
                       "fp_bins_pct": [round(100 * v / tot_err, 1) for v in tot_fp],
                       "fn_bins_pct": [round(100 * v / tot_err, 1) for v in tot_fn],
                       "z_extent_pct": round(100 * tot_z / tot_err, 1),
                       "mean_dice": round(np.mean([r["dice"] for r in rows.values()]), 2),
                       "mean_rt_dice": round(np.mean([r["rt_dice"] for r in rows.values()]), 2)}}
    out = a.out or a.pred_dir.rstrip("/") + "_errsrc.json"
    json.dump(summary, open(out, "w"), indent=1)
    print(f"{'organ':16s} {'Dice':>6s} {'deficit':>7s} {'224-floor':>9s} {'err mL':>8s} {'FP%':>5s}  "
          + " ".join(f"{b:>7s}" for b in BIN_NAMES) + f" {'z-ends%':>8s}")
    for name, r in sorted(rows.items(), key=lambda kv: -kv[1]["deficit_pts"]):
        print(f"{name:16s} {r['dice']:6.2f} {r['deficit_pts']:7.2f} {r['rt_dice']:9.2f} {r['err_ml']:8.1f} {r['fp_pct']:5.1f}  "
              + " ".join(f"{v:7.1f}" for v in r["bins_pct"]) + f" {r['z_extent_pct']:8.1f}")
    s = summary["all"]
    print(f"ALL: mean Dice {s['mean_dice']}, 224 floor {s['mean_rt_dice']}, error {s['err_ml']} mL, FP {s['fp_pct']}%, "
          f"bins {dict(zip(BIN_NAMES, s['bins_pct']))}, z-ends {s['z_extent_pct']}%")
    print("  FP bins", dict(zip(BIN_NAMES, s["fp_bins_pct"])), " FN bins", dict(zip(BIN_NAMES, s["fn_bins_pct"])))
    print("written", out)


if __name__ == "__main__":
    main()
