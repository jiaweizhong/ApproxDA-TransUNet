#!/usr/bin/env python3
"""
Generate train.txt / test.txt list files for Kvasir-SEG, ISIC 2018, and DRIVE.

Run ONCE on Lightning AI after attaching the dataset volumes:

  cd experiments/ApproxDA-TransUNet

  # Kvasir-SEG: the committed lists are the official 880/120 split used in the
  # re-runs -- do NOT regenerate them (this script refuses to overwrite existing
  # lists unless --force is given; its random 80/20 split is NOT the official one).

  # ISIC 2018 Task 1, official train (2594) / val (100) / test (1000) split.
  # --isic_official_root must contain the six official ISIC2018_Task1* folders;
  # images and masks are copied into <data_dir>/images and <data_dir>/masks.
  python datasets/generate_lists.py --dataset ISIC --data_dir ../data/ISIC2018 \
      --isic_official_root /path/to/ISIC2018_official --force

  # ISIC 2018 (legacy: all images in a flat folder; random 80/20 split)
  python datasets/generate_lists.py --dataset ISIC --data_dir ../data/ISIC2018

Written lists are mirrored to ../DA-TransUNet/lists/<name>/ (if that folder
exists) so that DA-TransUNet and ApproxDA-TransUNet always use identical splits.

  # DRIVE retinal vessels (fixed official 20/20 train/test split)
  python datasets/generate_lists.py --dataset DRIVE --data_dir ../data/DRIVE

List format: one image stem per line, no extension.
  Kvasir  -> stem = filename without extension  (e.g.  cju0qkwl9qokg0993l0dewei2)
  ISIC    -> stem = image filename without .jpg (e.g.  ISIC_0024306)
             masks are expected as  <stem>_segmentation.png
  DRIVE   -> stem = image filename without extension (e.g.  21_training, 01_test)
             masks derived as  <number>_manual1.{gif,png}  in 1st_manual/
"""

import argparse
import glob
import os
import random
import shutil

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    choices=["Kvasir", "ISIC", "DRIVE", "CVC", "KvasirInstrument", "ChestXray"],
    required=True,
)
parser.add_argument("--data_dir", required=True, help="root data directory")
parser.add_argument("--seed", type=int, default=42, help="split seed (default 42)")
parser.add_argument(
    "--train_ratio",
    type=float,
    default=0.8,
    help="fraction used for training (default 0.8); ignored for DRIVE",
)
parser.add_argument(
    "--isic_official_root",
    default=None,
    help="ISIC only: folder with the official ISIC2018_Task1* train/val/test folders; "
    "writes train/val/test lists from the official split instead of a random split",
)
parser.add_argument(
    "--force", action="store_true", help="overwrite existing (non-placeholder) lists"
)
args = parser.parse_args()

random.seed(args.seed)
val_names = None  # only the ISIC official split has a validation set


def _is_placeholder(path):
    """True if a list file is missing or only holds comments/blank lines."""
    if not os.path.isfile(path):
        return True
    with open(path) as f:
        return all(not l.strip() or l.strip().startswith("#") for l in f)


def _isic_official(root, data_dir):
    """Read the official ISIC 2018 Task 1 split and flatten files into data_dir."""
    img_out = os.path.join(data_dir, "images")
    mask_out = os.path.join(data_dir, "masks")
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(mask_out, exist_ok=True)
    splits = {}
    for split, tag in [("train", "Training"), ("val", "Validation"), ("test", "Test")]:
        img_dirs = glob.glob(os.path.join(root, f"*Task1*{tag}_Input*"))
        gt_dirs = glob.glob(os.path.join(root, f"*Task1*{tag}_GroundTruth*"))
        if len(img_dirs) != 1 or len(gt_dirs) != 1:
            raise SystemExit(f"expected one {tag} input and one {tag} GroundTruth folder under {root}")
        stems = sorted(
            os.path.splitext(f)[0]
            for f in os.listdir(img_dirs[0])
            if f.lower().endswith(".jpg")
        )
        for s in stems:
            mask = os.path.join(gt_dirs[0], s + "_segmentation.png")
            if not os.path.isfile(mask):
                raise SystemExit(f"missing ground truth for {s} in {gt_dirs[0]}")
            for src, dst_dir in [(os.path.join(img_dirs[0], s + ".jpg"), img_out), (mask, mask_out)]:
                dst = os.path.join(dst_dir, os.path.basename(src))
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)
        splits[split] = stems
    return splits["train"], splits["val"], splits["test"]

if args.dataset == "Kvasir":
    list_dir = "./lists/lists_Kvasir"
    img_dir = os.path.join(args.data_dir, "images")
    names = sorted(
        [
            os.path.splitext(f)[0]
            for f in os.listdir(img_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
    )
    random.shuffle(names)
    n_train = int(len(names) * args.train_ratio)
    train_names = names[:n_train]
    test_names = names[n_train:]

elif args.dataset == "ISIC" and args.isic_official_root:
    list_dir = "./lists/lists_ISIC"
    train_names, val_names, test_names = _isic_official(
        args.isic_official_root, args.data_dir
    )

elif args.dataset == "ISIC":
    list_dir = "./lists/lists_ISIC"
    img_dir = os.path.join(args.data_dir, "images")
    names = sorted(
        [
            os.path.splitext(f)[0]
            for f in os.listdir(img_dir)
            if f.lower().endswith((".jpg", ".jpeg"))
        ]
    )
    random.shuffle(names)
    n_train = int(len(names) * args.train_ratio)
    train_names = names[:n_train]
    test_names = names[n_train:]

elif args.dataset == "CVC":
    list_dir = "./lists/lists_CVC"
    # Support both Kaggle layout (PNG/Original/) and flat layout (images/)
    kaggle_dir = os.path.join(args.data_dir, "PNG", "Original")
    flat_dir = os.path.join(args.data_dir, "images")
    img_dir = kaggle_dir if os.path.isdir(kaggle_dir) else flat_dir
    names = sorted(
        [
            os.path.splitext(f)[0]
            for f in os.listdir(img_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif"))
        ],
        key=lambda x: int(x) if x.isdigit() else x,
    )
    random.shuffle(names)
    n_train = int(len(names) * args.train_ratio)
    train_names = names[:n_train]
    test_names = names[n_train:]

elif args.dataset == "KvasirInstrument":
    list_dir = "./lists/lists_KvasirInstrument"
    img_dir = os.path.join(args.data_dir, "images")
    names = sorted(
        [
            os.path.splitext(f)[0]
            for f in os.listdir(img_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
    )
    random.shuffle(names)
    n_train = int(len(names) * args.train_ratio)
    train_names = names[:n_train]
    test_names = names[n_train:]

elif args.dataset == "ChestXray":
    list_dir = "./lists/lists_ChestXray"
    # Support Layout A (images/) and Layout B (CXR_png/)
    img_dir_a = os.path.join(args.data_dir, "images")
    img_dir_b = os.path.join(args.data_dir, "CXR_png")
    img_dir = img_dir_a if os.path.isdir(img_dir_a) else img_dir_b
    names = sorted(
        [
            os.path.splitext(f)[0]
            for f in os.listdir(img_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
    )
    random.shuffle(names)
    n_train = int(len(names) * args.train_ratio)
    train_names = names[:n_train]
    test_names = names[n_train:]

elif args.dataset == "DRIVE":
    # Public DRIVE releases omit test-set GT (held for challenge server).
    # We do an 80/20 split of the 20 labelled training images instead.
    list_dir = "./lists/lists_DRIVE"
    train_img_dir = os.path.join(args.data_dir, "training", "images")
    names = sorted(
        [
            os.path.splitext(f)[0]
            for f in os.listdir(train_img_dir)
            if f.lower().endswith((".tif", ".tiff", ".png"))
        ]
    )
    random.shuffle(names)
    n_train = int(len(names) * args.train_ratio)  # default 0.8 → 16 train / 4 val
    train_names = names[:n_train]
    test_names = names[n_train:]

lists = {"train": train_names, "test": test_names}
if val_names is not None:
    lists["val"] = val_names

existing = [
    s for s in lists if not _is_placeholder(os.path.join(list_dir, s + ".txt"))
]
if existing and not args.force:
    raise SystemExit(
        f"{list_dir} already has {existing} lists; refusing to overwrite the split used "
        "in the experiments. Re-run with --force if you really want a new split."
    )

# Mirror to DA-TransUNet so both models always train/test on identical splits.
da_list_dir = os.path.join("..", "DA-TransUNet", "lists", os.path.basename(list_dir))
targets = [list_dir] + ([da_list_dir] if os.path.isdir(os.path.dirname(da_list_dir)) else [])
for target in targets:
    os.makedirs(target, exist_ok=True)
    for split, names in lists.items():
        with open(os.path.join(target, split + ".txt"), "w") as f:
            f.write("\n".join(names) + "\n")

sizes = " / ".join(f"{len(n)} {s}" for s, n in lists.items())
source = "official split" if val_names is not None else f"seed={args.seed}"
print(f"{args.dataset}: {sizes}  ({source})")
print("Written to " + ", ".join(t + "/" for t in targets))
