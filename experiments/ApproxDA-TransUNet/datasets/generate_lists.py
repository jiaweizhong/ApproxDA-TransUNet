#!/usr/bin/env python3
"""
Generate train.txt / test.txt list files for Kvasir-SEG, ISIC 2018, and DRIVE.

Run ONCE on Lightning AI after attaching the dataset volumes:

  cd experiments/ApproxDA-TransUNet

  # Kvasir-SEG (all 1000 images; 80/20 split)
  python datasets/generate_lists.py --dataset Kvasir --data_dir ../data/Kvasir-SEG

  # ISIC 2018 (all images in a flat folder; 80/20 split)
  python datasets/generate_lists.py --dataset ISIC --data_dir ../data/ISIC2018

  # DRIVE retinal vessels (fixed official 20/20 train/test split)
  python datasets/generate_lists.py --dataset DRIVE --data_dir ../../data/DRIVE

List format: one image stem per line, no extension.
  Kvasir  -> stem = filename without extension  (e.g.  cju0qkwl9qokg0993l0dewei2)
  ISIC    -> stem = image filename without .jpg (e.g.  ISIC_0024306)
             masks are expected as  <stem>_segmentation.png
  DRIVE   -> stem = image filename without extension (e.g.  21_training, 01_test)
             masks derived as  <number>_manual1.{gif,png}  in 1st_manual/
"""

import argparse
import os
import random

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
args = parser.parse_args()

random.seed(args.seed)

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

os.makedirs(list_dir, exist_ok=True)
with open(os.path.join(list_dir, "train.txt"), "w") as f:
    f.write("\n".join(train_names) + "\n")
with open(os.path.join(list_dir, "test.txt"), "w") as f:
    f.write("\n".join(test_names) + "\n")

print(
    f"{args.dataset}: {len(train_names)} train / {len(test_names)} test  (seed={args.seed})"
)
print(f"Written to {list_dir}/")
