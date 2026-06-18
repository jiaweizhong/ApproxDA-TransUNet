"""
GCR Context Sensitivity Analysis — inference only, no retraining.

Applies center-crop at increasing context sizes to measure how much each
segmentation task depends on global spatial context (Global Context Requirement).
A steep DSC drop as crop shrinks -> high GCR (Synapse).
Near-flat curve -> low GCR (Kvasir, ISIC).

Run from experiments/Ada-DA-TransUNet/:
  python analyze_gcr_context.py --dataset Synapse --gate_mode pam --max_epochs 300
  python analyze_gcr_context.py --dataset Kvasir  --gate_mode learn --max_epochs 300
  python analyze_gcr_context.py --dataset ISIC    --gate_mode learn --max_epochs 300

Context crop sizes (square center crop, then resized back to 224x224):
  56  -> 25% of image width   112 -> 50%   168 -> 75%   224 -> 100% (no crop)
"""

import argparse
import logging
import os
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from scipy.ndimage import zoom as scipy_zoom
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.dataset_synapse import Synapse_dataset
from datasets.dataset_kvasir import Kvasir_dataset
from datasets.dataset_isic import ISIC_dataset
from utils import calculate_metric_percase
from Architecture.AdaDATransUNet import AdaDATransUNet
from Architecture.AdaDATransUNet import CONFIGS as CONFIGS_ViT_seg


CONTEXT_SIZES = [56, 112, 168, 224]   # crop side-length in pixels; 224 = no crop


# ---------------------------------------------------------------------------
# Center-crop helpers
# ---------------------------------------------------------------------------

def center_crop_and_resize(image_224: np.ndarray, crop_size: int, patch_size: int = 224) -> np.ndarray:
    """
    Center-crop a patch_size x patch_size image to crop_size x crop_size,
    then bicubic-resize back to patch_size x patch_size.
    crop_size == patch_size is a no-op.
    """
    if crop_size >= patch_size:
        return image_224
    half = crop_size // 2
    cx = cy = patch_size // 2
    cropped = image_224[cy - half: cy + half, cx - half: cx + half]
    scale = patch_size / cropped.shape[0]
    resized = scipy_zoom(cropped, (scale, scale), order=3)
    if resized.shape[0] != patch_size:
        resized = scipy_zoom(cropped,
                             (patch_size / cropped.shape[0], patch_size / cropped.shape[1]),
                             order=3)
    return resized


# ---------------------------------------------------------------------------
# Inference with context masking
# ---------------------------------------------------------------------------

def infer_with_crop(image, label, model, num_classes: int, patch_size: int, crop_size: int):
    """
    Like test_single_volume() from utils.py but applies center_crop_and_resize
    after the initial zoom to patch_size x patch_size, before model forward.

    Handles both:
      - 3D (Synapse): image shape (D, H, W) -> per-slice inference
      - 2D (Kvasir / ISIC): image shape (H, W)
    """
    image = image.squeeze(0).cpu().detach().numpy()
    label = label.squeeze(0).cpu().detach().numpy()

    model.eval()
    with torch.no_grad():
        if len(image.shape) == 3:
            # 3D volume: process each axial slice independently
            prediction = np.zeros_like(label)
            for idx in range(image.shape[0]):
                slc = image[idx]
                h, w = slc.shape

                if h != patch_size or w != patch_size:
                    slc224 = scipy_zoom(slc, (patch_size / h, patch_size / w), order=3)
                else:
                    slc224 = slc.copy()

                slc224 = center_crop_and_resize(slc224, crop_size, patch_size)

                inp = torch.from_numpy(slc224).unsqueeze(0).unsqueeze(0).float().cuda()
                out = torch.argmax(torch.softmax(model(inp), dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()

                if h != patch_size or w != patch_size:
                    pred = scipy_zoom(out, (h / patch_size, w / patch_size), order=0)
                else:
                    pred = out
                prediction[idx] = pred
        else:
            # 2D image
            h, w = image.shape
            if h != patch_size or w != patch_size:
                img224 = scipy_zoom(image, (patch_size / h, patch_size / w), order=3)
            else:
                img224 = image.copy()

            img224 = center_crop_and_resize(img224, crop_size, patch_size)

            inp = torch.from_numpy(img224).unsqueeze(0).unsqueeze(0).float().cuda()
            out = torch.argmax(torch.softmax(model(inp), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
            if h != patch_size or w != patch_size:
                prediction = scipy_zoom(prediction, (h / patch_size, w / patch_size), order=0)

    metrics = []
    for cls in range(1, num_classes):
        metrics.append(calculate_metric_percase(prediction == cls, label == cls))
    return metrics   # list of (dsc, hd95, jc) per class


# ---------------------------------------------------------------------------
# Sweep over all context sizes
# ---------------------------------------------------------------------------

def run_context_sweep(args, model):
    """Return dict: crop_size -> {'dsc': float, 'hd95': float}."""
    db_test = args.Dataset(base_dir=args.volume_path, split="test_vol",
                           list_dir=args.list_dir)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=2)
    n_samples = len(db_test)
    logging.info(f"Dataset: {args.dataset}  |  Test samples: {n_samples}")

    results = {}
    for crop_size in CONTEXT_SIZES:
        pct = round(100 * crop_size / 224)
        logging.info(f"\n--- crop={crop_size}x{crop_size} ({pct}% of image) ---")
        accum = np.zeros((args.num_classes - 1, 3))   # (n_classes-1, [dsc, hd95, jc])

        for batch in tqdm(testloader, desc=f"crop={crop_size}", leave=False):
            image = batch["image"]
            label = batch["label"]
            m = infer_with_crop(image, label, model,
                                num_classes=args.num_classes,
                                patch_size=args.img_size,
                                crop_size=crop_size)
            accum += np.array(m)

        accum /= n_samples
        mean_dsc  = float(np.mean(accum[:, 0]))
        mean_hd95 = float(np.mean(accum[:, 1]))
        results[crop_size] = {"dsc": mean_dsc, "hd95": mean_hd95}
        logging.info(f"  DSC={mean_dsc:.4f}   HD95={mean_hd95:.2f}")

    return results


# ---------------------------------------------------------------------------
# Snapshot path (mirrors Ada-DA-TransUNet train.py naming)
# ---------------------------------------------------------------------------

def build_snapshot_path(args) -> str:
    exp  = 'AdaDA_' + args.dataset + str(args.img_size)
    snap = f'../model/{exp}/AdaDA_pretrain_{args.vit_name}_skip{args.n_skip}'
    if args.vit_patches_size != 16:
        snap += f'_vitpatch{args.vit_patches_size}'
    if args.max_epochs != 30:
        snap += f'_epo{args.max_epochs}'
    snap += f'_bs{args.batch_size}'
    if args.base_lr != 0.01:
        snap += f'_lr{args.base_lr}'
    snap += f'_{args.img_size}'
    if args.seed != 1234:
        snap += f'_s{args.seed}'
    if args.window_size != 7:
        snap += f'_M{args.window_size}'
    if args.rank != 32:
        snap += f'_r{args.rank}'
    if args.gate_mode != 'learn':
        snap += f'_{args.gate_mode}'
    best = os.path.join(snap, 'best_model.pth')
    return best if os.path.exists(best) else os.path.join(snap, f'epoch_{args.max_epochs - 1}.pth')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GCR Context Sensitivity Analysis (AdaDA)")
    parser.add_argument('--dataset',          type=str,   default='Synapse',
                        choices=['Synapse', 'Kvasir', 'ISIC'])
    parser.add_argument('--vit_name',         type=str,   default='R50-ViT-B_16')
    parser.add_argument('--img_size',         type=int,   default=224)
    parser.add_argument('--n_skip',           type=int,   default=3)
    parser.add_argument('--vit_patches_size', type=int,   default=16)
    parser.add_argument('--max_epochs',       type=int,   default=300)
    parser.add_argument('--batch_size',       type=int,   default=24)
    parser.add_argument('--base_lr',          type=float, default=0.01)
    parser.add_argument('--seed',             type=int,   default=1234)
    parser.add_argument('--window_size',      type=int,   default=7)
    parser.add_argument('--rank',             type=int,   default=32)
    parser.add_argument('--groups',           type=int,   default=8)
    parser.add_argument('--gate_mode',        type=str,   default='learn',
                        choices=['learn', 'fixed', 'pam', 'cam'])
    parser.add_argument('--snapshot',         type=str,   default='',
                        help='Direct path to best_model.pth. Auto-computed if empty.')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

    dataset_config = {
        'Synapse': {
            'Dataset':     Synapse_dataset,
            'volume_path': '../data/Synapse/test_vol_h5',
            'list_dir':    './lists/lists_Synapse',
            'num_classes': 9,
        },
        'Kvasir': {
            'Dataset':     Kvasir_dataset,
            'volume_path': '../data/Kvasir-SEG',
            'list_dir':    './lists/lists_Kvasir',
            'num_classes': 2,
        },
        'ISIC': {
            'Dataset':     ISIC_dataset,
            'volume_path': '../data/ISIC2018',
            'list_dir':    './lists/lists_ISIC',
            'num_classes': 2,
        },
    }
    cfg = dataset_config[args.dataset]
    args.Dataset     = cfg['Dataset']
    args.volume_path = cfg['volume_path']
    args.list_dir    = cfg['list_dir']
    args.num_classes = cfg['num_classes']

    snapshot = args.snapshot if args.snapshot else build_snapshot_path(args)

    # Logging
    tag = f"{args.dataset}_M{args.window_size}_r{args.rank}_{args.gate_mode}"
    log_dir = './gcr_analysis'
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f'gcr_context_{tag}.log')
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(message)s',
        datefmt='%H:%M:%S',
        handlers=[logging.FileHandler(log_path, mode='w'),
                  logging.StreamHandler(sys.stdout)],
    )
    logging.info(f"Checkpoint: {snapshot}")
    if not os.path.exists(snapshot):
        logging.error(f"Checkpoint not found: {snapshot}")
        sys.exit(1)

    # Build model
    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes  = args.num_classes
    config_vit.n_skip     = args.n_skip
    config_vit.window_size = args.window_size
    config_vit.rank       = args.rank
    config_vit.groups     = args.groups
    config_vit.gate_mode  = args.gate_mode
    config_vit.patches.size = (args.vit_patches_size, args.vit_patches_size)
    if 'R50' in args.vit_name:
        g = args.img_size // args.vit_patches_size
        config_vit.patches.grid = (g, g)

    model = AdaDATransUNet(config_vit, img_size=args.img_size,
                           num_classes=config_vit.n_classes).cuda()
    model.load_state_dict(torch.load(snapshot, map_location='cuda'))
    logging.info(f"Model loaded. Params: {sum(p.numel() for p in model.parameters())/1e6:.2f} M")

    # Run sweep
    results = run_context_sweep(args, model)

    # Summary table
    sep = '=' * 62
    logging.info(f'\n{sep}')
    logging.info(f'  GCR Context Sensitivity — {args.dataset}  (AdaDA M={args.window_size} r={args.rank} gate={args.gate_mode})')
    logging.info(sep)
    logging.info(f'  {"Crop size":>12}  {"% image":>8}  {"DSC":>8}  {"HD95":>8}')
    logging.info(f'  {"-"*54}')
    for cs in CONTEXT_SIZES:
        pct = round(100 * cs / 224)
        r = results[cs]
        marker = ' <-- full context' if cs == 224 else ''
        logging.info(f'  {cs:>12}  {pct:>7}%  {r["dsc"]:>8.4f}  {r["hd95"]:>8.2f}{marker}')
    logging.info(sep)

    drop = results[224]["dsc"] - results[56]["dsc"]
    logging.info(f'  DSC drop (224->56): {drop:+.4f}  '
                 f'({"HIGH GCR signal" if drop > 0.05 else "low GCR signal"})')
    logging.info(sep)

    # CSV
    csv_path = os.path.join(log_dir, f'gcr_context_{tag}.csv')
    with open(csv_path, 'w') as f:
        f.write('dataset,gate_mode,window_size,rank,crop_size,pct_image,dsc,hd95\n')
        for cs in CONTEXT_SIZES:
            pct = round(100 * cs / 224)
            r = results[cs]
            f.write(f'{args.dataset},{args.gate_mode},{args.window_size},{args.rank},'
                    f'{cs},{pct},{r["dsc"]:.6f},{r["hd95"]:.4f}\n')
    logging.info(f'Results -> {csv_path}')
    logging.info(f'Log     -> {log_path}')


if __name__ == '__main__':
    main()
