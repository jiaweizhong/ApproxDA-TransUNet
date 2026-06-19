#!/usr/bin/env python3
"""
Generate train.txt / test.txt list files for Kvasir-SEG and ISIC 2018.

Run ONCE on Lightning AI after attaching the dataset volumes:

  cd experiments/ApproxDA-TransUNet

  # Kvasir-SEG (all 1000 images; 80/20 split)
  python datasets/generate_lists.py --dataset Kvasir --data_dir ../data/Kvasir-SEG

  # ISIC 2018 (all images in a flat folder; 80/20 split)
  python datasets/generate_lists.py --dataset ISIC --data_dir ../data/ISIC2018

List format: one image stem per line, no extension.
  Kvasir  -> stem = filename without extension  (e.g.  cju0qkwl9qokg0993l0dewei2)
  ISIC    -> stem = image filename without .jpg (e.g.  ISIC_0024306)
             masks are expected as  <stem>_segmentation.png
"""

import argparse
import os
import random

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', choices=['Kvasir', 'ISIC'], required=True)
parser.add_argument('--data_dir', required=True, help='root data directory')
parser.add_argument('--seed', type=int, default=42, help='split seed (default 42)')
parser.add_argument('--train_ratio', type=float, default=0.8,
                    help='fraction used for training (default 0.8)')
args = parser.parse_args()

random.seed(args.seed)

img_dir = os.path.join(args.data_dir, 'images')

if args.dataset == 'Kvasir':
    list_dir = './lists/lists_Kvasir'
    names = sorted([
        os.path.splitext(f)[0]
        for f in os.listdir(img_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

elif args.dataset == 'ISIC':
    list_dir = './lists/lists_ISIC'
    names = sorted([
        os.path.splitext(f)[0]
        for f in os.listdir(img_dir)
        if f.lower().endswith(('.jpg', '.jpeg'))
    ])

random.shuffle(names)
n_train = int(len(names) * args.train_ratio)
train_names = names[:n_train]
test_names  = names[n_train:]

os.makedirs(list_dir, exist_ok=True)
with open(os.path.join(list_dir, 'train.txt'), 'w') as f:
    f.write('\n'.join(train_names) + '\n')
with open(os.path.join(list_dir, 'test.txt'), 'w') as f:
    f.write('\n'.join(test_names) + '\n')

print(f"{args.dataset}: {len(train_names)} train / {len(test_names)} test  (seed={args.seed})")
print(f"Written to {list_dir}/")
