"""
Phase E — PAM Attention Map Visualization

Captures the spatial effect of LowRankWindowedPAM at every active decoder scale
and visualizes which image regions the attention module modifies most strongly.
This is a proxy for "where is the model attending": pixels with high modification
magnitude received strong attention, pixels with near-zero modification were
essentially bypassed by the PAM branch.

Methodology:
  For each LowRankWindowedPAM module, register a forward hook that computes
      effect = (output - input).norm(dim=1)    # (B, H, W)
  This equals alpha * E_out (the attention-weighted residual) since the module
  output is alpha*E_out + x. So the map shows the magnitude of the learned
  attention contribution at each spatial position, upsampled to image size.

Usage — single model (run now, gate=learn M=7 checkpoint exists):
  python analyze_attention_maps.py --dataset Kvasir \\
    --gate_mode learn --window_size 7 --rank 32 --max_epochs 300

Usage — comparison (M=7 vs M=112, once Phase D D3+D6 checkpoints are ready):
  python analyze_attention_maps.py --dataset Kvasir \\
    --gate_mode pam --window_size 7 --rank 32 --max_epochs 300 \\
    --compare_snapshot ../model/ApproxDA_Kvasir224/ApproxDA_pretrain_R50-ViT-B_16_skip3_epo300_bs24_224_M112_pam/best_model.pth \\
    --compare_label "M=112 (global)"

Outputs (in ./attention_maps/):
  attn_{tag}.png         — panel figure (N images × columns)
  attn_{tag}_stats.csv   — per-image mean attention intensity + mask overlap
"""

import argparse
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from scipy.ndimage import zoom as scipy_zoom
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.transforms import Bbox

from datasets.dataset_kvasir import Kvasir_dataset
from datasets.dataset_isic import ISIC_dataset
from Architecture.ApproxDATransUNet import ApproxDATransUNet
from Architecture.ApproxDATransUNet import CONFIGS as CONFIGS_ViT_seg
from Architecture.block import LowRankWindowedPAM


N_SAMPLES   = 10   # number of test images to visualize
PATCH_SIZE  = 224


# ---------------------------------------------------------------------------
# Checkpoint loading (handles both zip and legacy pickle format)
# ---------------------------------------------------------------------------

def load_checkpoint(path, map_location='cuda'):
    try:
        return torch.load(path, map_location=map_location)
    except RuntimeError:
        import pickle
        with open(path, 'rb') as f:
            return pickle.load(f, encoding='latin1')


# ---------------------------------------------------------------------------
# Snapshot path (mirrors train.py naming)
# ---------------------------------------------------------------------------

def build_snapshot_path(args):
    exp  = 'ApproxDA_' + args.dataset + str(args.img_size)
    snap = f'../model/{exp}/ApproxDA_pretrain_{args.vit_name}_skip{args.n_skip}'
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
# Model builder
# ---------------------------------------------------------------------------

def build_model(args, num_classes, window_size, gate_mode, snapshot_path):
    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes   = num_classes
    config_vit.n_skip      = args.n_skip
    config_vit.window_size = window_size
    config_vit.rank        = args.rank
    config_vit.groups      = args.groups
    config_vit.gate_mode   = gate_mode
    config_vit.patches.size = (args.vit_patches_size, args.vit_patches_size)
    if 'R50' in args.vit_name:
        g = args.img_size // args.vit_patches_size
        config_vit.patches.grid = (g, g)

    model = ApproxDATransUNet(config_vit, img_size=args.img_size,
                           num_classes=num_classes).cuda()
    if not os.path.exists(snapshot_path):
        print(f'[ERROR] Checkpoint not found: {snapshot_path}')
        sys.exit(1)
    ckpt = load_checkpoint(snapshot_path)
    ckpt = {k.replace('.ada_block.', '.approx_block.'): v for k, v in ckpt.items()}
    model.load_state_dict(ckpt)
    model.eval()
    print(f'  Loaded: {snapshot_path}')
    return model


# ---------------------------------------------------------------------------
# Attention effect hook
# ---------------------------------------------------------------------------

class PAMEffectCollector:
    """Registers forward hooks on all LowRankWindowedPAM modules.
    Computes effect = (output - input).norm(dim=1) per scale.
    """
    def __init__(self, model):
        self.effects = {}   # name -> (B, H, W) tensor (last forward call)
        self._hooks  = []
        for name, module in model.named_modules():
            if isinstance(module, LowRankWindowedPAM):
                h = module.register_forward_hook(self._make_hook(name))
                self._hooks.append(h)

    def _make_hook(self, name):
        def hook(module, inp, output):
            # inp[0]: (B, C, H, W), output: (B, C, H, W)
            effect = (output - inp[0]).norm(dim=1)   # (B, H, W)
            self.effects[name] = effect.detach().cpu()
        return hook

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def aggregate(self, target_h=PATCH_SIZE, target_w=PATCH_SIZE):
        """Average all collected effect maps after upsampling to (target_h, target_w).
        Returns (B, target_h, target_w) tensor."""
        if not self.effects:
            return None
        upsampled = []
        for name, eff in self.effects.items():
            # eff: (B, H, W)
            eff_4d = eff.unsqueeze(1).float()   # (B, 1, H, W)
            up = F.interpolate(eff_4d, size=(target_h, target_w),
                               mode='bilinear', align_corners=False)
            upsampled.append(up.squeeze(1))     # (B, H, W)
        return torch.stack(upsampled, dim=0).mean(dim=0)   # (B, H, W)


# ---------------------------------------------------------------------------
# Inference + attention extraction for one image
# ---------------------------------------------------------------------------

def run_single_image(image_np, label_np, model, collector):
    """
    image_np: (H, W) numpy, label_np: (H, W) numpy
    Returns: prediction (H, W), attention_map (H, W) in [0,1]
    """
    h, w = image_np.shape
    if h != PATCH_SIZE or w != PATCH_SIZE:
        img224 = scipy_zoom(image_np, (PATCH_SIZE / h, PATCH_SIZE / w), order=3)
    else:
        img224 = image_np.copy()

    inp = torch.from_numpy(img224).unsqueeze(0).unsqueeze(0).float().cuda()

    collector.effects.clear()
    with torch.no_grad():
        logits = model(inp)
        pred224 = torch.argmax(torch.softmax(logits, dim=1), dim=1).squeeze(0).cpu().numpy()

    # Aggregate PAM effects
    agg = collector.aggregate(PATCH_SIZE, PATCH_SIZE)   # (1, 224, 224)
    agg_np = agg[0].numpy()

    # Return raw values; global normalisation applied after all samples collected.

    # Resize prediction back to original label size
    if h != PATCH_SIZE or w != PATCH_SIZE:
        pred = scipy_zoom(pred224, (h / PATCH_SIZE, w / PATCH_SIZE), order=0)
    else:
        pred = pred224

    return pred, agg_np, pred224, img224


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def overlay_heatmap(ax, image_224, heatmap, title, alpha=0.55):
    ax.imshow(image_224, cmap='gray', vmin=0, vmax=1)
    ax.imshow(heatmap, cmap='jet', alpha=alpha, vmin=0, vmax=1)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=4)
    ax.axis('off')


def _contour_with_halo(ax, mask, color, linewidth=2.0, halo_width=3.5):
    """Draw a contour with a dark halo for contrast on any background."""
    ax.contour(mask, levels=[0.5], colors=['black'], linewidths=halo_width,  zorder=2)
    ax.contour(mask, levels=[0.5], colors=[color],  linewidths=linewidth,    zorder=3)


def show_mask(ax, image_224, gt_224, pred_224, title):
    """Overlay GT (green) and prediction (cyan) contours on image."""
    ax.imshow(image_224, cmap='gray', vmin=0, vmax=1)
    _contour_with_halo(ax, gt_224,   'lime', linewidth=2.0)
    _contour_with_halo(ax, pred_224, 'cyan', linewidth=2.0)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=4)
    ax.axis('off')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase E — PAM Attention Map Visualization")
    parser.add_argument('--dataset',          type=str,  default='Kvasir',
                        choices=['Kvasir', 'ISIC'])
    parser.add_argument('--vit_name',         type=str,  default='R50-ViT-B_16')
    parser.add_argument('--img_size',         type=int,  default=224)
    parser.add_argument('--n_skip',           type=int,  default=3)
    parser.add_argument('--vit_patches_size', type=int,  default=16)
    parser.add_argument('--max_epochs',       type=int,  default=300)
    parser.add_argument('--batch_size',       type=int,  default=24)
    parser.add_argument('--base_lr',          type=float,default=0.01)
    parser.add_argument('--seed',             type=int,  default=1234)
    parser.add_argument('--window_size',      type=int,  default=7)
    parser.add_argument('--rank',             type=int,  default=32)
    parser.add_argument('--groups',           type=int,  default=8)
    parser.add_argument('--gate_mode',        type=str,  default='learn',
                        choices=['learn', 'fixed', 'pam', 'cam'])
    parser.add_argument('--snapshot',         type=str,  default='',
                        help='Override checkpoint path')
    # Optional second model for side-by-side comparison
    parser.add_argument('--compare_snapshot', type=str,  default='',
                        help='Path to second checkpoint for comparison (e.g. M=112)')
    parser.add_argument('--compare_window',   type=int,  default=112,
                        help='Window size of the comparison model')
    parser.add_argument('--compare_label',    type=str,  default='',
                        help='Label for comparison model in figure')
    parser.add_argument('--n_samples',        type=int,  default=N_SAMPLES)
    parser.add_argument('--seed_sample',      type=int,  default=42,
                        help='Random seed for image selection')
    parser.add_argument('--select_samples',   type=int,  nargs='+', default=None,
                        help='Plot only these 0-based indices from the collected samples '
                             '(e.g. --select_samples 0 7 9). Runs inference on all '
                             'n_samples but saves a figure with only the chosen rows.')
    parser.add_argument('--split_rows',       action='store_true',
                        help='Save each row as a separate PNG (row0.png, row1.png, …) '
                             'instead of one combined grid. Useful for per-sample LaTeX figures.')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

    dataset_config = {
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
    cfg         = dataset_config[args.dataset]
    num_classes = cfg['num_classes']

    snapshot = args.snapshot if args.snapshot else build_snapshot_path(args)
    tag = f"{args.dataset}_M{args.window_size}_r{args.rank}_{args.gate_mode}"

    # --- Load primary model ---
    print(f'\n[Primary] M={args.window_size} gate={args.gate_mode}')
    model_a = build_model(args, num_classes, args.window_size, args.gate_mode, snapshot)
    collector_a = PAMEffectCollector(model_a)

    # --- Load comparison model (optional) ---
    model_b = collector_b = None
    if args.compare_snapshot:
        compare_label = args.compare_label or f'M={args.compare_window}'
        print(f'\n[Compare] {compare_label}')
        model_b    = build_model(args, num_classes, args.compare_window, 'pam', args.compare_snapshot)
        collector_b = PAMEffectCollector(model_b)
        tag += f'_vs_M{args.compare_window}'
    else:
        compare_label = None

    # --- Load test data ---
    db_test = cfg['Dataset'](base_dir=cfg['volume_path'], split='test_vol',
                              list_dir=cfg['list_dir'])
    loader  = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)

    # Randomly select N_SAMPLES indices
    rng   = np.random.RandomState(args.seed_sample)
    total = len(db_test)
    idxs  = sorted(rng.choice(total, size=min(args.n_samples, total), replace=False))

    # --- Collect samples ---
    samples = []
    for i, batch in enumerate(loader):
        if i not in idxs:
            continue
        image_raw = batch['image'].squeeze(0).cpu().numpy()   # (H, W)
        label_raw = batch['label'].squeeze(0).cpu().numpy()   # (H, W)

        # Normalise image to [0,1] for display
        img_disp = (image_raw - image_raw.min()) / (image_raw.max() - image_raw.min() + 1e-8)

        # Resize GT to 224 for display
        h, w = img_disp.shape
        if h != PATCH_SIZE or w != PATCH_SIZE:
            gt224  = scipy_zoom(label_raw.astype(float), (PATCH_SIZE / h, PATCH_SIZE / w), order=0)
            img224 = scipy_zoom(img_disp, (PATCH_SIZE / h, PATCH_SIZE / w), order=3)
        else:
            gt224  = label_raw.astype(float)
            img224 = img_disp

        pred_a, attn_a, pred224_a, _ = run_single_image(img_disp, label_raw, model_a, collector_a)
        if h != PATCH_SIZE or w != PATCH_SIZE:
            pred224_a = scipy_zoom(pred_a.astype(float), (PATCH_SIZE / h, PATCH_SIZE / w), order=0)

        result = {
            'img224': img224, 'gt224': gt224,
            'pred224_a': pred224_a, 'attn_a': attn_a,
            'label': f'M={args.window_size} {args.gate_mode}',
        }

        if model_b is not None:
            pred_b, attn_b, pred224_b, _ = run_single_image(img_disp, label_raw, model_b, collector_b)
            if h != PATCH_SIZE or w != PATCH_SIZE:
                pred224_b = scipy_zoom(pred_b.astype(float), (PATCH_SIZE / h, PATCH_SIZE / w), order=0)
            result['pred224_b'] = pred224_b
            result['attn_b']    = attn_b

        samples.append(result)
        if len(samples) >= args.n_samples:
            break

    collector_a.remove()
    if collector_b is not None:
        collector_b.remove()

    # --- Global normalisation across all samples and both models ---
    all_maps = [s['attn_a'] for s in samples]
    if any('attn_b' in s for s in samples):
        all_maps += [s['attn_b'] for s in samples if 'attn_b' in s]
    g_min = min(m.min() for m in all_maps)
    g_max = max(m.max() for m in all_maps)
    for s in samples:
        s['attn_a'] = (s['attn_a'] - g_min) / (g_max - g_min + 1e-8)
        if 'attn_b' in s:
            s['attn_b'] = (s['attn_b'] - g_min) / (g_max - g_min + 1e-8)

    # --- Filter samples for figure ---
    if args.select_samples is not None:
        plot_samples = [samples[i] for i in args.select_samples if i < len(samples)]
        print(f'Plotting {len(plot_samples)}/{len(samples)} selected samples: {args.select_samples}')
    else:
        plot_samples = samples

    # --- Build figure ---
    has_compare = model_b is not None
    # Columns: image+GT | pred_a+attn_a | [pred_b+attn_b]
    n_cols = 4 if has_compare else 2   # (img, attn_a) or (img, attn_a, attn_b_side_by_side)
    if has_compare:
        n_cols = 3   # image | model_a heatmap | model_b heatmap

    n_rows = len(plot_samples)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.8, n_rows * 3.8))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    primary_label = f'M={args.window_size} {args.gate_mode}'

    for row, s in enumerate(plot_samples):
        col = 0

        # Column 0: image + GT contour
        ax = axes[row, col]
        ax.imshow(s['img224'], cmap='gray', vmin=0, vmax=1)
        _contour_with_halo(ax, s['gt224'], 'lime', linewidth=2.0)
        if row == 0:
            ax.set_title('Image + GT', fontsize=13, fontweight='bold', pad=4)
        ax.axis('off')
        col += 1

        # Column 1: primary model attention heatmap + prediction contour
        ax = axes[row, col]
        overlay_heatmap(ax, s['img224'], s['attn_a'],
                        title=primary_label if row == 0 else '')
        _contour_with_halo(ax, s['pred224_a'], 'cyan', linewidth=2.0)
        col += 1

        # Column 2 (optional): comparison model attention heatmap
        if has_compare:
            ax = axes[row, col]
            overlay_heatmap(ax, s['img224'], s['attn_b'],
                            title=compare_label if row == 0 else '')
            _contour_with_halo(ax, s['pred224_b'], 'cyan', linewidth=2.0)

    fig.suptitle(
        f'PAM Attention Effect — {args.dataset}  |  hot=high PAM modification  |  cyan=prediction  |  green=GT',
        fontsize=12, y=1.01
    )
    plt.tight_layout()

    # --- Save ---
    out_dir = './attention_maps'
    os.makedirs(out_dir, exist_ok=True)
    fig_suffix = f'_sel{"_".join(str(i) for i in args.select_samples)}' if args.select_samples else ''
    fig_path = os.path.join(out_dir, f'attn_{tag}{fig_suffix}.png')
    if args.split_rows:
        row_dir = os.path.join(out_dir, f'attn_{tag}{fig_suffix}_rows')
        os.makedirs(row_dir, exist_ok=True)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for row in range(n_rows):
            tight_bboxes = [axes[row, c].get_tightbbox(renderer) for c in range(n_cols)]
            bbox_inches = Bbox.union(tight_bboxes).transformed(fig.dpi_scale_trans.inverted())
            fig.savefig(os.path.join(row_dir, f'row{row}.png'), dpi=200, bbox_inches=bbox_inches)
        plt.close(fig)
        print(f'\nRows saved to: {row_dir}')
    else:
        plt.savefig(fig_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f'\nFigure saved: {fig_path}')

    # --- CSV stats ---
    csv_path = os.path.join(out_dir, f'attn_{tag}_stats.csv')
    with open(csv_path, 'w') as f:
        f.write('sample_idx,model,mean_attn,max_attn,attn_on_mask,attn_off_mask\n')
        for si, s in enumerate(samples):
            mask = s['gt224'] > 0.5
            bg   = ~mask
            for model_tag, attn in [(primary_label, s['attn_a'])]:
                mean_a   = float(attn.mean())
                max_a    = float(attn.max())
                on_mask  = float(attn[mask].mean()) if mask.any()  else 0.0
                off_mask = float(attn[bg].mean())   if bg.any()    else 0.0
                f.write(f'{si},{model_tag},{mean_a:.4f},{max_a:.4f},{on_mask:.4f},{off_mask:.4f}\n')
            if has_compare and 'attn_b' in s:
                attn = s['attn_b']
                mean_a   = float(attn.mean())
                max_a    = float(attn.max())
                on_mask  = float(attn[mask].mean()) if mask.any()  else 0.0
                off_mask = float(attn[bg].mean())   if bg.any()    else 0.0
                f.write(f'{si},{compare_label},{mean_a:.4f},{max_a:.4f},{on_mask:.4f},{off_mask:.4f}\n')

    print(f'Stats CSV : {csv_path}')
    print('\nDone.')


if __name__ == '__main__':
    main()
