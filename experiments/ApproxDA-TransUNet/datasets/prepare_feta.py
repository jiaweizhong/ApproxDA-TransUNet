"""Convert FeTA 2.4 (BIDS) into the Synapse format of TransUNet (train_npz slices + test_vol_h5 volumes) on the
64 / 8 / 8 split of the EffiDec3D protocol (split_feta.json, drawn by AdaDec3D/EffiDec3D/prepare_feta.py with seed 0,
stratified by super-resolution method and pathology; the same split as the 3D baselines run there).

Preprocessing follows EffiDec3D's FeTA transforms (load_datasets_transforms.py, identical for train / val / test):
RAS orientation, intensity clipped to [0, 1000] and scaled to [0, 1], crop to the bounding box of the image
foreground (> 0, MONAI CropForegroundd defaults). The volumes are 0.5 mm isotropic and are not resampled. Arrays are
stored as (z: I->S, y: P->A, x: L->R) and sliced axially. Labels keep the FeTA indices (0 background, 1 external CSF,
2 grey matter, 3 white matter, 4 lateral ventricles, 5 cerebellum, 6 deep grey matter, 7 brainstem).

Output: train_npz/ (slices of the 64 training subjects), test_vol_h5/ (the 8 test and the 8 validation volumes, with
the voxel spacing in mm in array axis order); lists train.txt, test_vol.txt (8 test), val.txt (8 validation).

Usage (read-only on the source; run isolated because the source is downloaded data):
    python -I datasets/prepare_feta.py /data/datasets/feta_2.4 /data/datasets/feta_effidec3d/split_feta.json \
        /data/datasets/FeTA_2d lists/lists_FeTA
"""
import argparse
import json
import multiprocessing as mp
import os

import h5py
import nibabel as nib
import numpy as np

I_MIN, I_MAX = 0.0, 1000.0
N_LABELS = 8


def load(path):
    img = nib.as_closest_canonical(nib.load(path))  # RAS+, as MONAI Orientationd(axcodes="RAS")
    arr = np.asanyarray(img.dataobj)
    assert arr.ndim == 3, (path, arr.shape)
    return np.ascontiguousarray(arr.transpose(2, 1, 0)), np.array(img.header.get_zooms()[:3], dtype=float)[::-1]


def convert(job):
    split, sub, img_path, lab_path, out_dir = job
    img, spacing = load(img_path)
    lab, lab_spacing = load(lab_path)
    assert img.shape == lab.shape and np.allclose(spacing, lab_spacing, atol=1e-3), sub
    img = (np.clip(img.astype(np.float32), I_MIN, I_MAX) - I_MIN) / (I_MAX - I_MIN)
    lab = np.rint(lab).astype(np.uint8)
    assert lab.max() < N_LABELS, (sub, lab.max())
    fg = np.nonzero(img > 0)
    box = tuple(slice(int(a.min()), int(a.max()) + 1) for a in fg)
    lab_outside = int((lab > 0).sum() - (lab[box] > 0).sum())
    full_shape = img.shape
    img, lab = np.ascontiguousarray(img[box]), np.ascontiguousarray(lab[box])
    info = dict(subject=sub, split=split, full_shape=list(full_shape), crop_shape=list(img.shape),
                spacing=spacing.tolist(), label_voxels_outside_crop=lab_outside,
                classes_present=sorted(int(c) for c in np.unique(lab) if c > 0))
    if split == "train":
        names = [f"{sub}_slice{z:03d}" for z in range(img.shape[0])]
        for z, name in enumerate(names):
            np.savez(os.path.join(out_dir, "train_npz", name + ".npz"), image=img[z], label=lab[z])
        return split, names, info
    with h5py.File(os.path.join(out_dir, "test_vol_h5", sub + ".npy.h5"), "w") as f:
        f.create_dataset("image", data=img)
        f.create_dataset("label", data=lab)
        f.create_dataset("spacing", data=spacing)
    return split, [sub], info


def main():
    p = argparse.ArgumentParser()
    p.add_argument("bids", help="FeTA BIDS root (sub-XXX/anat/sub-XXX_rec-*_T2w.nii.gz, *_dseg.nii.gz)")
    p.add_argument("split_json", help="split_feta.json (subjects with split train / val / test and rec)")
    p.add_argument("out", help="output root (train_npz/, test_vol_h5/)")
    p.add_argument("list_dir", help="where train.txt, test_vol.txt and val.txt are written")
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    meta = json.load(open(args.split_json))
    jobs = []
    for s in meta["subjects"]:
        sub, rec = s["subject"], s["rec"]
        anat = os.path.join(args.bids, sub, "anat")
        jobs.append(({"train": "train", "test": "test_vol", "val": "val"}[s["split"]], sub,
                     os.path.join(anat, f"{sub}_rec-{rec}_T2w.nii.gz"), os.path.join(anat, f"{sub}_rec-{rec}_dseg.nii.gz"),
                     args.out))
    for sub in ("train_npz", "test_vol_h5"):
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)
    os.makedirs(args.list_dir, exist_ok=True)

    lists, infos = {"train": [], "test_vol": [], "val": []}, []
    with mp.get_context("fork").Pool(args.workers) as pool:
        for i, (split, names, info) in enumerate(pool.imap_unordered(convert, jobs), 1):
            lists[split] += names
            infos.append(info)
            print(f"[{i}/{len(jobs)}] {split} {info['subject']}: {info['crop_shape']} of {info['full_shape']}, "
                  f"{len(names)} entries, label voxels outside crop {info['label_voxels_outside_crop']}", flush=True)
    for split, names in lists.items():
        with open(os.path.join(args.list_dir, split + ".txt"), "w") as f:
            f.write("".join(n + "\n" for n in sorted(names)))
    with open(os.path.join(args.out, "prepare_info.json"), "w") as f:
        json.dump({"source": os.path.abspath(args.bids), "split_json": os.path.abspath(args.split_json),
                   "intensity_window": [I_MIN, I_MAX], "orientation": "RAS, array (z: I->S, y: P->A, x: L->R)",
                   "crop": "bounding box of image > 0 (MONAI CropForegroundd)", "train_slices": len(lists["train"]),
                   "cases": {k: len(v) if k != "train" else len({n.split('_slice')[0] for n in v}) for k, v in lists.items()},
                   "subjects": sorted(infos, key=lambda d: d["subject"])}, f, indent=1)


if __name__ == "__main__":
    main()
