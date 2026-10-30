#!/usr/bin/env python3
"""
Resize the flattened ISIC 2018 Task 1 images/masks to the network input size once.

The originals are up to ~6000x4000 px; resizing them on the fly every iteration
makes ISIC training CPU-bound (the old run took ~45 h). Images use antialiased
bicubic resampling (saved as quality-100 JPEG, the loader expects .jpg), masks use
nearest neighbour. Training and evaluation then run at 224x224, so ISIC metrics
are computed at that resolution.

  python datasets/resize_isic.py --src /data/datasets/ISIC2018_full \
      --dst /data/datasets/ISIC2018_224 --size 224
  # DA-TransUNet paper protocol (random 75/25 of the official training set, 256 px):
  python datasets/resize_isic.py --src /data/datasets/ISIC2018_full \
      --dst /data/datasets/ISIC2018_256 --size 256 --names <official train.txt>
"""
import argparse
import os
from multiprocessing import Pool

from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--src", required=True, help="dir with images/ and masks/ (generate_lists.py output)")
parser.add_argument("--dst", required=True, help="output dir (images/ and masks/ are created)")
parser.add_argument("--size", type=int, default=224)
parser.add_argument("--workers", type=int, default=16)
parser.add_argument("--names", default=None, help="optional list file: only resize these stems (e.g. the official training set)")
args = parser.parse_args()


def resize(job):
    src, dst, is_mask = job
    if os.path.exists(dst):
        return
    im = Image.open(src)
    if is_mask:
        im.convert("L").resize((args.size, args.size), Image.NEAREST).save(dst)
    else:
        im.convert("RGB").resize((args.size, args.size), Image.BICUBIC).save(
            dst, quality=100, subsampling=0
        )


if __name__ == "__main__":
    keep = set(open(args.names).read().split()) if args.names else None
    jobs = []
    for sub, is_mask in [("images", False), ("masks", True)]:
        os.makedirs(os.path.join(args.dst, sub), exist_ok=True)
        for f in sorted(os.listdir(os.path.join(args.src, sub))):
            if keep is not None and f.rsplit(".", 1)[0].removesuffix("_segmentation") not in keep:
                continue
            jobs.append((os.path.join(args.src, sub, f), os.path.join(args.dst, sub, f), is_mask))
    with Pool(args.workers) as pool:
        for i, _ in enumerate(pool.imap_unordered(resize, jobs, chunksize=16), 1):
            if i % 1000 == 0 or i == len(jobs):
                print(f"{i}/{len(jobs)}", flush=True)
