"""Convert the AMOS 2022 CT cases into the Synapse format of TransUNet (train_npz slices + test_vol_h5 volumes).

Training set: the 200 CT cases of imagesTr (ids < 500). Test set: the 100 CT cases of imagesVa (the 240 imagesTs cases
have no labels). Every volume is reoriented to LPS and stored as (z, R->L, A->P) with z running inferior->superior,
the array layout of the Synapse data, then clipped to the abdominal window [-125, 275] HU and scaled to [0, 1].
Labels keep the AMOS indices (0 background, 1 spleen, 2 right kidney, 3 left kidney, ... 15 prostate/uterus). Test
volumes also store the voxel spacing in mm (same axis order as the array), so HD95 can be reported in mm.

Usage (read-only on the source; run isolated because the source is downloaded data):
    python -I datasets/prepare_amos.py /data/datasets/amos/amos22 /data/datasets/AMOS_CT_2d lists/lists_AMOS
"""
import argparse
import json
import multiprocessing as mp
import os

import h5py
import numpy as np
import SimpleITK as sitk

HU_MIN, HU_MAX = -125, 275


def load(path):
    img = sitk.DICOMOrient(sitk.ReadImage(path), "LPS")
    arr = np.ascontiguousarray(sitk.GetArrayFromImage(img).transpose(0, 2, 1))  # (z, y, x) -> (z, R->L, A->P)
    sx, sy, sz = img.GetSpacing()
    return arr, np.array([sz, sx, sy])


def convert(job):
    split, img_path, lab_path, out_dir = job
    case = os.path.basename(img_path)[: -len(".nii.gz")]
    img, spacing = load(img_path)
    lab, lab_spacing = load(lab_path)
    assert img.shape == lab.shape and np.allclose(spacing, lab_spacing, atol=1e-3), case
    img = ((np.clip(img.astype(np.float32), HU_MIN, HU_MAX) - HU_MIN) / (HU_MAX - HU_MIN)).astype(np.float32)
    lab = lab.astype(np.uint8)
    if split == "train":
        names = [f"{case}_slice{z:03d}" for z in range(img.shape[0])]
        for z, name in enumerate(names):
            np.savez(os.path.join(out_dir, "train_npz", name + ".npz"), image=img[z], label=lab[z])
        return split, case, names
    with h5py.File(os.path.join(out_dir, "test_vol_h5", case + ".npy.h5"), "w") as f:
        f.create_dataset("image", data=img)
        f.create_dataset("label", data=lab)
        f.create_dataset("spacing", data=spacing)
    return split, case, [case]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("src", help="AMOS root containing dataset.json, imagesTr, labelsTr, imagesVa, labelsVa")
    p.add_argument("out", help="output root (train_npz/, test_vol_h5/)")
    p.add_argument("list_dir", help="where train.txt and test_vol.txt are written")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=0, help="convert only the first N cases of each split (smoke test)")
    args = p.parse_args()

    meta = json.load(open(os.path.join(args.src, "dataset.json")))
    jobs = []
    for split, key in (("train", "training"), ("test_vol", "validation")):
        entries = [e for e in meta[key] if int(os.path.basename(e["image"])[5:9]) < 500]  # CT only
        entries = sorted(entries, key=lambda e: e["image"])[: args.limit or None]
        for e in entries:
            jobs.append((split, os.path.join(args.src, e["image"].lstrip("./")),
                         os.path.join(args.src, e["label"].lstrip("./")), args.out))
    for sub in ("train_npz", "test_vol_h5"):
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)
    os.makedirs(args.list_dir, exist_ok=True)

    lists = {"train": [], "test_vol": []}
    with mp.get_context("fork").Pool(args.workers) as pool:
        for i, (split, case, names) in enumerate(pool.imap_unordered(convert, jobs), 1):
            lists[split] += names
            print(f"[{i}/{len(jobs)}] {split} {case}: {len(names)} entries", flush=True)
    for split, names in lists.items():
        with open(os.path.join(args.list_dir, split + ".txt"), "w") as f:
            f.write("".join(n + "\n" for n in sorted(names)))
    with open(os.path.join(args.out, "prepare_info.json"), "w") as f:
        json.dump({"source": os.path.abspath(args.src), "hu_window": [HU_MIN, HU_MAX], "orientation": "LPS, array (z, R->L, A->P)",
                   "train_cases": len({n.split("_slice")[0] for n in lists["train"]}), "train_slices": len(lists["train"]),
                   "test_cases": len(lists["test_vol"])}, f, indent=1)


if __name__ == "__main__":
    main()
