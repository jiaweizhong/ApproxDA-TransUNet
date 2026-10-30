"""Whole-organ misidentification on AMOS, from the label maps test.py saves (../predictions/<exp>/<run>_npz/<case>.npz).
Model-independent: only the predictions and the ground-truth test volumes are read.

Two views of the errors (voxels with prediction != ground truth):
1. Voxel view. An error voxel predicted as organ b is *far* when its 3-D distance to the ground truth of b is larger
   than --far_mm (default 15 mm, about the 20 px used for Synapse at 512^2), or b is absent from the case. Far errors
   are split into organ->organ (true label a != 0) and background->organ.
2. Component view. A 3-D connected component (26-connectivity, >= --min_ml) of predicted organ b that overlaps the
   ground truth of b by less than --max_overlap of its volume is a *misplaced* component; it is attributed to the
   true label covering most of it. Symmetrically, a ground-truth component of organ a that the prediction covers by
   less than --max_overlap is a *missed* component, attributed to the predicted label covering most of it.

Usage (cwd = experiments/ApproxDA-TransUNet):
    python analyze_misidentification.py ../predictions/ApproxDA_AMOS224/<run>_npz [--out summary.json]
"""
import argparse
import json
import multiprocessing as mp
import os
import sys

import h5py
import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datasets.dataset_amos import AMOS_CLASSES  # noqa: E402

NAMES = ("bg",) + AMOS_CLASSES
S26 = ndimage.generate_binary_structure(3, 3)


def components(mask, min_vox):
    cc, n = ndimage.label(mask, structure=S26)
    sizes = np.bincount(cc.ravel(), minlength=n + 1)
    for k, sl in enumerate(ndimage.find_objects(cc), 1):
        if sizes[k] >= min_vox:
            yield sl, cc[sl] == k


def analyse(job):
    case, pred_path, gt_path, a = job
    with h5py.File(gt_path) as f:
        lab, sp = f["label"][:], f["spacing"][:]
    pred = np.load(pred_path)["prediction"]
    assert pred.shape == lab.shape, case
    vox_ml = float(np.prod(sp)) / 1000.0
    min_vox = int(np.ceil(a.min_ml / vox_ml))
    err = pred != lab
    r = {"case": case, "err_ml": float(err.sum()) * vox_ml, "far_oo_ml": 0.0, "far_bo_ml": 0.0, "far_pairs": {},
         "misplaced": [], "missed": []}
    pad = np.ceil(a.far_mm / sp).astype(int) + 1
    for b in range(1, len(NAMES)):
        m = err & (pred == b)
        if m.any():
            box = tuple(slice(max(int(i.min()) - p, 0), int(i.max()) + p + 1) for i, p in zip(np.nonzero(m), pad))
            gt_b = lab[box] == b
            if gt_b.any():  # every ground-truth voxel within far_mm of an error voxel lies inside the padded box
                far = ndimage.distance_transform_edt(~gt_b, sampling=sp)[m[box]] > a.far_mm
            else:
                far = np.ones(int(m.sum()), dtype=bool)
            true_far = lab[box][m[box]][far]
            for t, n in enumerate(np.bincount(true_far, minlength=len(NAMES))):
                if n:
                    key = NAMES[t] + "->" + NAMES[b]
                    r["far_pairs"][key] = r["far_pairs"].get(key, 0.0) + float(n) * vox_ml
                    r["far_oo_ml" if t else "far_bo_ml"] += float(n) * vox_ml
        for sl, comp in components(pred == b, min_vox):
            under = lab[sl][comp]
            if (under == b).mean() < a.max_overlap:
                t = int(np.bincount(under, minlength=len(NAMES)).argmax())
                r["misplaced"].append({"pred": NAMES[b], "true": NAMES[t], "ml": round(float(comp.sum()) * vox_ml, 2),
                                       "err_ml": round(float((under != b).sum()) * vox_ml, 2), "z": [int(sl[0].start), int(sl[0].stop)]})
        for sl, comp in components(lab == b, min_vox):
            over = pred[sl][comp]
            if (over == b).mean() < a.max_overlap:
                t = int(np.bincount(over, minlength=len(NAMES)).argmax())
                r["missed"].append({"true": NAMES[b], "pred": NAMES[t], "ml": round(float(comp.sum()) * vox_ml, 2),
                                    "z": [int(sl[0].start), int(sl[0].stop)]})
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pred_dir")
    ap.add_argument("--gt_dir", default="../data/AMOS_CT_2d/test_vol_h5")
    ap.add_argument("--far_mm", type=float, default=15.0)
    ap.add_argument("--min_ml", type=float, default=1.0)
    ap.add_argument("--max_overlap", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="", help="JSON output (default: <pred_dir>_misid.json)")
    a = ap.parse_args()
    cases = sorted(f[:-4] for f in os.listdir(a.pred_dir) if f.endswith(".npz"))
    jobs = [(c, os.path.join(a.pred_dir, c + ".npz"), os.path.join(a.gt_dir, c + ".npy.h5"), a) for c in cases]
    with mp.get_context("fork").Pool(a.workers) as pool:
        R = pool.map(analyse, jobs)

    err = max(sum(r["err_ml"] for r in R), 1e-9)
    far_oo, far_bo = sum(r["far_oo_ml"] for r in R), sum(r["far_bo_ml"] for r in R)
    pairs = {}
    for r in R:
        for k, v in r["far_pairs"].items():
            pairs[k] = pairs.get(k, 0.0) + v
    mis = [dict(m, case=r["case"]) for r in R for m in r["misplaced"]]
    miss = [dict(m, case=r["case"]) for r in R for m in r["missed"]]

    def by(items, key):
        out = {}
        for m in items:
            k = key(m)
            n, ml = out.get(k, (0, 0.0))
            out[k] = (n + 1, round(ml + m["ml"], 2))
        return dict(sorted(out.items(), key=lambda kv: -kv[1][1]))

    summary = {
        "pred_dir": os.path.abspath(a.pred_dir), "cases": len(R),
        "params": {"far_mm": a.far_mm, "min_ml": a.min_ml, "max_overlap": a.max_overlap},
        "error_ml": round(err, 1),
        "far_organ_to_organ_pct": round(100 * far_oo / err, 2), "far_background_to_organ_pct": round(100 * far_bo / err, 2),
        "far_pairs_pct": {k: round(100 * v / err, 2) for k, v in sorted(pairs.items(), key=lambda kv: -kv[1])},
        "misplaced": {"n": len(mis), "cases": len({m["case"] for m in mis}),
                      "err_pct": round(100 * sum(m["err_ml"] for m in mis) / err, 2),
                      "by_pair (n, ml)": by(mis, lambda m: m["true"] + "->" + m["pred"]),
                      "by_case (n, ml)": by(mis, lambda m: m["case"])},
        "missed": {"n": len(miss), "cases": len({m["case"] for m in miss}),
                   "by_pair (n, ml)": by(miss, lambda m: m["true"] + "->" + m["pred"])},
        "misplaced_components": mis, "missed_components": miss,
    }
    out = a.out or a.pred_dir.rstrip("/") + "_misid.json"
    json.dump(summary, open(out, "w"), indent=1)
    print(f"{len(R)} cases, errors {err:.0f} mL; far errors: organ->organ {summary['far_organ_to_organ_pct']}%, "
          f"background->organ {summary['far_background_to_organ_pct']}%")
    print("top far pairs (% of error volume):", dict(list(summary["far_pairs_pct"].items())[:10]))
    m = summary["misplaced"]
    print(f"misplaced components: {m['n']} in {m['cases']} cases, {m['err_pct']}% of error volume")
    print("  by pair:", dict(list(m["by_pair (n, ml)"].items())[:10]))
    print("  by case:", dict(list(m["by_case (n, ml)"].items())[:5]))
    print(f"missed components: {summary['missed']['n']} in {summary['missed']['cases']} cases;",
          dict(list(summary["missed"]["by_pair (n, ml)"].items())[:10]))
    print("written", out)


if __name__ == "__main__":
    main()
