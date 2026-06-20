"""
Verify whether feature entropy H(F) correlates with learned gate value g
in ApproxDABlock. Uses register_forward_hook on an existing checkpoint —
no changes to block.py or ApproxDATransUNet.py.

Usage (from experiments/ApproxDA-TransUNet/):
  # Synapse (default)
  python analyze_gate_entropy.py \\
    --vit_name R50-ViT-B_16 --n_skip 3 --max_epochs 300 --batch_size 24 \\
    --window_size 7 --rank 32 --groups 8

  # Kvasir-SEG or ISIC 2018 (once those checkpoints exist)
  python analyze_gate_entropy.py --dataset Kvasir \\
    --vit_name R50-ViT-B_16 --n_skip 3 --max_epochs 300 --batch_size 24 \\
    --window_size 7 --rank 32 --groups 8

Decision threshold (from plan):
  |r| > 0.3 and p < 0.01  -> proceed to Phase 2 (entropy gate implementation)
  |r| < 0.1               -> gate ignores entropy; narrative revision needed
  r < 0                   -> revise claim (channel attention stronger at high entropy)

Per-class outputs (Synapse):
  Fig E  — per-organ normalised mean H and mean g (shows which organs drive entropy)
  Fig F  — per-organ Spearman r_s (H vs g, slices containing that organ only)

Per-category outputs (binary datasets):
  Fig E  — foreground-fraction quartile mean H and mean g
  Fig F  — per-quartile Spearman r_s
"""

import argparse
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from scipy.ndimage import zoom as nd_zoom
from torch.utils.data import DataLoader

from Architecture.ApproxDATransUNet import ApproxDATransUNet
from Architecture.ApproxDATransUNet import CONFIGS as CONFIGS_ViT_seg
from Architecture.block import ApproxDABlock
from datasets.dataset_synapse import Synapse_dataset

# ── organ map (Synapse) ────────────────────────────────────────────────────────
SYNAPSE_ORGANS = {
    1: 'Aorta', 2: 'Gallbladder', 3: 'Kidney(L)',
    4: 'Kidney(R)', 5: 'Liver', 6: 'Pancreas',
    7: 'Spleen', 8: 'Stomach',
}

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Synapse',
                    choices=['Synapse', 'Kvasir', 'ISIC'],
                    help='which dataset checkpoint to analyse')
parser.add_argument('--volume_path', type=str, default='../data/Synapse/test_vol_h5')
parser.add_argument('--list_dir', type=str, default='./lists/lists_Synapse')
parser.add_argument('--num_classes', type=int, default=9)
parser.add_argument('--img_size', type=int, default=224)
parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16')
parser.add_argument('--vit_patches_size', type=int, default=16)
parser.add_argument('--n_skip', type=int, default=3)
parser.add_argument('--max_epochs', type=int, default=300)
parser.add_argument('--batch_size', type=int, default=24)
parser.add_argument('--base_lr', type=float, default=0.01)
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--window_size', type=int, default=7)
parser.add_argument('--rank', type=int, default=32)
parser.add_argument('--groups', type=int, default=8)
parser.add_argument('--gate_mode', type=str, default='learn',
                    choices=['learn', 'fixed', 'pam', 'cam'],
                    help='Must match the gate_mode used during training')
parser.add_argument('--out_dir', type=str, default='figures',
                    help='directory to save paper figures (created if absent)')
args = parser.parse_args()

if args.gate_mode != 'learn':
    sys.exit("Gate entropy analysis only applies to gate_mode='learn' (other modes have no learned gate).")

# ---------- build snapshot path (mirrors train.py logic exactly) ----------
dataset_tag = {'Synapse': 'Synapse224', 'Kvasir': 'Kvasir224', 'ISIC': 'ISIC224'}[args.dataset]
args.exp = 'ApproxDA_' + dataset_tag
snapshot_path = '../model/{}/{}'.format(args.exp, 'ApproxDA')
snapshot_path += '_pretrain'
snapshot_path += '_' + args.vit_name
snapshot_path += '_skip' + str(args.n_skip)
snapshot_path += '_epo' + str(args.max_epochs) if args.max_epochs != 30 else ''
snapshot_path += '_bs' + str(args.batch_size)
snapshot_path += '_lr' + str(args.base_lr) if args.base_lr != 0.01 else ''
snapshot_path += '_' + str(args.img_size)
snapshot_path += '_s' + str(args.seed) if args.seed != 1234 else ''
snapshot_path += '_M' + str(args.window_size) if args.window_size != 7 else ''
snapshot_path += '_r' + str(args.rank) if args.rank != 32 else ''

checkpoint = os.path.join(snapshot_path, 'best_model.pth')
if not os.path.exists(checkpoint):
    checkpoint = os.path.join(snapshot_path, 'epoch_{}.pth'.format(args.max_epochs - 1))
if not os.path.exists(checkpoint):
    sys.exit("Checkpoint not found at: {}".format(checkpoint))
print("Loading checkpoint:", checkpoint)

# ---------- build model ----------
config_vit = CONFIGS_ViT_seg[args.vit_name]
config_vit.n_classes = args.num_classes
config_vit.n_skip = args.n_skip
config_vit.window_size = args.window_size
config_vit.rank = args.rank
config_vit.groups = args.groups
config_vit.gate_mode = args.gate_mode
if args.vit_name.find('R50') != -1:
    config_vit.patches.grid = (
        args.img_size // args.vit_patches_size,
        args.img_size // args.vit_patches_size,
    )

net = ApproxDATransUNet(config_vit, img_size=args.img_size, num_classes=args.num_classes)
state = torch.load(checkpoint, map_location='cpu')
state = {k.replace('.ada_block.', '.approx_block.'): v for k, v in state.items()}
net.load_state_dict(state)
net.eval()
if torch.cuda.is_available():
    net = net.cuda()

# ---------- find all ApproxDABlock instances ----------
approxda_blocks = [(name, m) for name, m in net.named_modules() if isinstance(m, ApproxDABlock)]
print("Found {} ApproxDABlock(s):".format(len(approxda_blocks)))
for name, _ in approxda_blocks:
    print(" ", name)

# ---------- register hooks ----------
records = {i: {'H': [], 'Var': [], 'g': []} for i in range(len(approxda_blocks))}

def make_hook(idx, blk):
    def hook(module, inp, out):
        x = inp[0].detach()
        prob = F.softmax(x.flatten(2), dim=-1)
        H = (-prob * torch.log(prob + 1e-6)).sum(-1).mean(-1).cpu()
        Var = x.var(dim=[2, 3]).mean(dim=1).cpu()
        with torch.no_grad():
            gap = module.pool(x).view(x.shape[0], -1)
            g = torch.sigmoid(module.gate_fc(gap)).mean(dim=1).cpu()
        records[idx]['H'].append(H)
        records[idx]['Var'].append(Var)
        records[idx]['g'].append(g)
    return hook

hooks = []
for i, (name, blk) in enumerate(approxda_blocks):
    hooks.append(blk.register_forward_hook(make_hook(i, blk)))

# ---------- run inference slice-by-slice, recording organ context ----------
# We iterate manually (not via test_single_volume) so we can capture per-slice
# ground truth class composition alongside the hook-recorded H/g values.

slice_contexts = []  # one dict per forward pass (per 2-D slice)

def _infer_volume(image_vol, label_vol, num_fg_classes):
    """Iterate slices of one volume, fire model forward, record slice_contexts."""
    if image_vol.ndim == 2:
        image_vol = image_vol[np.newaxis]
        label_vol = label_vol[np.newaxis]
    for s in range(image_vol.shape[0]):
        slice_img = image_vol[s].astype(np.float32)
        slice_lbl = label_vol[s]
        # resize to model input
        h, w = slice_img.shape
        if h != args.img_size or w != args.img_size:
            slice_img = nd_zoom(slice_img, (args.img_size / h, args.img_size / w), order=3)
        # record class context
        organs_in_slice = {
            c: int((slice_lbl == c).sum())
            for c in range(1, num_fg_classes + 1)
            if (slice_lbl == c).any()
        }
        fg_frac = float((slice_lbl > 0).mean())
        slice_contexts.append({'organs': organs_in_slice, 'fg_fraction': fg_frac})
        # forward pass — hooks fire here
        img_t = torch.from_numpy(slice_img.copy()).unsqueeze(0).unsqueeze(0)
        img_t = img_t.repeat(1, 3, 1, 1)
        if torch.cuda.is_available():
            img_t = img_t.cuda()
        with torch.no_grad():
            net(img_t)


if args.dataset == 'Synapse':
    db = Synapse_dataset(base_dir=args.volume_path, split='test_vol', list_dir=args.list_dir)
    loader = DataLoader(db, batch_size=1, shuffle=False, num_workers=1)
    print("Running inference on {} Synapse volumes …".format(len(db)))
    for sample in loader:
        img_vol = sample['image'].squeeze(0).cpu().numpy()
        lbl_vol = sample['label'].squeeze(0).cpu().numpy()
        _infer_volume(img_vol, lbl_vol, num_fg_classes=8)

elif args.dataset in ('Kvasir', 'ISIC'):
    # ── binary datasets: add dataset_kvasir / dataset_isic when available ───
    # Until those loaders exist, load from a flat image+mask directory.
    # Expected layout:
    #   Kvasir: ../data/Kvasir/images/*.png, ../data/Kvasir/masks/*.png
    #   ISIC:   ../data/ISIC/images/*.jpg,   ../data/ISIC/masks/*_segmentation.png
    import glob
    from PIL import Image
    dataset_root = args.volume_path.replace('Synapse/test_vol_h5', args.dataset)
    img_paths = sorted(glob.glob(os.path.join(dataset_root, 'images', '*')))
    msk_paths = sorted(glob.glob(os.path.join(dataset_root, 'masks', '*')))
    if not img_paths:
        sys.exit("No images found under: " + dataset_root + "/images/  — run Synapse first.")
    print("Running inference on {} {} images …".format(len(img_paths), args.dataset))
    for ip, mp in zip(img_paths, msk_paths):
        img = np.array(Image.open(ip).convert('L').resize(
            (args.img_size, args.img_size))).astype(np.float32) / 255.0
        msk = (np.array(Image.open(mp).resize(
            (args.img_size, args.img_size))) > 0).astype(np.uint8)
        # binary: class 1 = foreground
        img_vol = img[np.newaxis]
        lbl_vol = msk[np.newaxis]
        _infer_volume(img_vol, lbl_vol, num_fg_classes=1)

for h in hooks:
    h.remove()

# ---------- filter to only blocks actually called during forward ----------
# DecoderBlock.forward routes skip to da/da2/da3 based on channel count, so
# only 1 of the 3 DA blocks per DecoderBlock fires per forward pass.
active_idx = [i for i in range(len(approxda_blocks)) if len(records[i]['H']) > 0]
print(f"\n{len(active_idx)} of {len(approxda_blocks)} blocks active during inference:")
for i in active_idx:
    print(f"  [{i}] {approxda_blocks[i][0]}")

if not active_idx:
    import sys; sys.exit("No hooks fired — check that best_model.pth loaded correctly.")

# ---------- aggregate per-slice averages across active blocks ----------
n_slices = len(slice_contexts)
H_avg = np.zeros(n_slices)
g_avg = np.zeros(n_slices)
for i in active_idx:
    H_avg += torch.cat(records[i]['H']).numpy()
    g_avg += torch.cat(records[i]['g']).numpy()
H_avg /= len(active_idx)
g_avg /= len(active_idx)

# ---------- compute Spearman correlations per active block ----------
print("\n=== Spearman correlation: H(F) vs gate g (per block) ===")
spearman_results = []
for i in active_idx:
    name = approxda_blocks[i][0]
    H_all   = torch.cat(records[i]['H']).numpy()
    Var_all = torch.cat(records[i]['Var']).numpy()
    g_all   = torch.cat(records[i]['g']).numpy()
    r_H,   p_H   = stats.spearmanr(H_all,   g_all)
    r_Var, p_Var = stats.spearmanr(Var_all,  g_all)
    spearman_results.append((name, r_H, p_H, r_Var, p_Var, H_all, Var_all, g_all))
    flag = "PROCEED" if abs(r_H) > 0.3 and p_H < 0.01 else ("WEAK" if abs(r_H) < 0.1 else "CHECK")
    print(f"  {name:50s}  H: r={r_H:+.4f} p={p_H:.2e}  [{flag}]  |  Var: r={r_Var:+.4f} p={p_Var:.2e}")

# ---------- Figure D stats: high vs low entropy group comparison ----------
print("\n=== Figure D: high-H vs low-H gate comparison (top/bottom 20%) ===")
figD_data = []
for i, (name, r_H, p_H, r_Var, p_Var, H_all, Var_all, g_all) in enumerate(spearman_results):
    thresh_hi = np.percentile(H_all, 80)
    thresh_lo = np.percentile(H_all, 20)
    g_hi = g_all[H_all >= thresh_hi]
    g_lo = g_all[H_all <= thresh_lo]
    delta = g_hi.mean() - g_lo.mean()
    figD_data.append((g_hi.mean(), g_lo.mean(), delta))
    print(f"  Block {i:2d}  high-H mean_g={g_hi.mean():.4f}  "
          f"low-H mean_g={g_lo.mean():.4f}  delta={delta:+.4f}")

# ---------- per-class / per-category analysis ----------
print(f"\n=== Per-class analysis ({args.dataset}) ===")

if args.dataset == 'Synapse':
    cat_names = []
    cat_H_vals = []
    cat_g_vals = []
    cat_r = []

    for cls_id, organ_name in SYNAPSE_ORGANS.items():
        mask = np.array([cls_id in ctx['organs'] for ctx in slice_contexts])
        if mask.sum() < 10:
            continue
        h_sub = H_avg[mask]
        g_sub = g_avg[mask]
        r, p = stats.spearmanr(h_sub, g_sub) if len(h_sub) > 5 else (float('nan'), float('nan'))
        cat_names.append(organ_name)
        cat_H_vals.append(h_sub)
        cat_g_vals.append(g_sub)
        cat_r.append((r, p))
        print(f"  {organ_name:15s}  n={mask.sum():4d}  "
              f"mean_H={h_sub.mean():.4f}  mean_g={g_sub.mean():.4f}  "
              f"r_s={r:+.4f}  p={p:.2e}")

else:
    # Binary dataset: group by foreground-fraction quartile
    fg_fracs = np.array([ctx['fg_fraction'] for ctx in slice_contexts])
    q25, q50, q75 = np.percentile(fg_fracs, [25, 50, 75])
    boundaries = [0, q25, q50, q75, 1.01]
    raw_names = ['Q1 small/absent', 'Q2 small-med', 'Q3 med-large', 'Q4 large']
    cat_names = []
    cat_H_vals = []
    cat_g_vals = []
    cat_r = []

    for qi in range(4):
        mask = (fg_fracs >= boundaries[qi]) & (fg_fracs < boundaries[qi + 1])
        if mask.sum() < 5:
            continue
        h_sub = H_avg[mask]
        g_sub = g_avg[mask]
        r, p = stats.spearmanr(h_sub, g_sub) if len(h_sub) > 5 else (float('nan'), float('nan'))
        cat_names.append(raw_names[qi])
        cat_H_vals.append(h_sub)
        cat_g_vals.append(g_sub)
        cat_r.append((r, p))
        print(f"  {raw_names[qi]:20s}  n={mask.sum():4d}  FG<={boundaries[qi+1]:.3f}  "
              f"mean_H={h_sub.mean():.4f}  mean_g={g_sub.mean():.4f}  "
              f"r_s={r:+.4f}  p={p:.2e}")

# ---------- save separate paper figures ----------
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(args.out_dir, exist_ok=True)
    DPI = 300
    FS  = 11

    n      = len(spearman_results)
    cols   = min(n, 4)
    rows   = (n + cols - 1) // cols
    labels = ['.'.join(name.split('.')[-2:]) if '.' in name else name
              for name, *_ in spearman_results]
    w = 0.35
    x = np.arange(n)

    # ── Figure A: gate distribution boxplot per block ──────────────────────
    fig, ax = plt.subplots(figsize=(max(4, n * 1.4), 3.5))
    g_data = [torch.cat(records[i]['g']).numpy() for i in active_idx]
    bp = ax.boxplot(g_data, labels=labels, patch_artist=True, notch=False)
    for patch in bp['boxes']:
        patch.set_facecolor('#a8c8e8')
    ax.set_ylabel('Mean gate value $g$', fontsize=FS)
    ax.set_xlabel('ApproxDA block', fontsize=FS)
    ax.tick_params(axis='x', rotation=25, labelsize=FS - 1)
    fig.tight_layout()
    path_A = os.path.join(args.out_dir, 'fig_A_gate_distribution.png')
    fig.savefig(path_A, dpi=DPI)
    plt.close(fig)
    print(f"\nFig A saved: {path_A}")

    # ── Figure B: entropy H(F) vs gate g scatter ───────────────────────────
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
    axes = np.array(axes).flatten() if n > 1 else [axes]
    for i, (name, r_H, p_H, r_Var, p_Var, H_all, Var_all, g_all) in enumerate(spearman_results):
        ax = axes[i]
        ax.scatter(H_all, g_all, alpha=0.25, s=5, color='steelblue', rasterized=True)
        ax.set_xlabel('Feature entropy $H(F)$', fontsize=FS)
        ax.set_ylabel('Mean gate $g$', fontsize=FS)
        sig = '***' if p_H < 0.001 else ('**' if p_H < 0.01 else ('*' if p_H < 0.05 else 'n.s.'))
        ax.set_title(f'{labels[i]}\n$r_s$={r_H:+.3f} {sig}', fontsize=FS)
    for ax in axes[n:]:
        ax.set_visible(False)
    fig.tight_layout()
    path_B = os.path.join(args.out_dir, 'fig_B_entropy_scatter.png')
    fig.savefig(path_B, dpi=DPI)
    plt.close(fig)
    print(f"Fig B saved: {path_B}")

    # ── Figure B2: variance Var(F) vs gate g scatter ───────────────────────
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
    axes = np.array(axes).flatten() if n > 1 else [axes]
    for i, (name, r_H, p_H, r_Var, p_Var, H_all, Var_all, g_all) in enumerate(spearman_results):
        ax = axes[i]
        ax.scatter(Var_all, g_all, alpha=0.25, s=5, color='darkorange', rasterized=True)
        ax.set_xlabel('Feature variance $\\mathrm{Var}(F)$', fontsize=FS)
        ax.set_ylabel('Mean gate $g$', fontsize=FS)
        sig = '***' if p_Var < 0.001 else ('**' if p_Var < 0.01 else ('*' if p_Var < 0.05 else 'n.s.'))
        ax.set_title(f'{labels[i]}\n$r_s$={r_Var:+.3f} {sig}', fontsize=FS)
    for ax in axes[n:]:
        ax.set_visible(False)
    fig.tight_layout()
    path_B2 = os.path.join(args.out_dir, 'fig_B2_variance_scatter.png')
    fig.savefig(path_B2, dpi=DPI)
    plt.close(fig)
    print(f"Fig B2 saved: {path_B2}")

    # ── Figure C: Spearman r bar chart (H and Var side by side) ───────────
    r_H_vals   = [r_H   for _, r_H, p_H, r_Var, p_Var, *_ in spearman_results]
    r_Var_vals = [r_Var for _, r_H, p_H, r_Var, p_Var, *_ in spearman_results]
    fig, ax = plt.subplots(figsize=(max(5, n * 1.6), 3.5))
    ax.bar(x - w/2, r_H_vals,   w, label='Entropy $H(F)$',    color='steelblue')
    ax.bar(x + w/2, r_Var_vals, w, label='Variance $\\mathrm{Var}(F)$', color='darkorange')
    ax.axhline(0,    color='black', linewidth=0.8)
    ax.axhline( 0.3, color='green', linewidth=1.0, linestyle='--', alpha=0.7, label='$r$=0.3 threshold')
    ax.axhline(-0.3, color='green', linewidth=1.0, linestyle='--', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha='right', fontsize=FS - 1)
    ax.set_ylabel('Spearman $r_s$', fontsize=FS)
    ax.set_xlabel('ApproxDA block', fontsize=FS)
    ax.legend(fontsize=FS - 1)
    fig.tight_layout()
    path_C = os.path.join(args.out_dir, 'fig_C_spearman_bars.png')
    fig.savefig(path_C, dpi=DPI)
    plt.close(fig)
    print(f"Fig C saved: {path_C}")

    # ── Figure D: mean gate g — high-H vs low-H groups per block ──────────
    g_hi_vals = [d[0] for d in figD_data]
    g_lo_vals = [d[1] for d in figD_data]
    fig, ax = plt.subplots(figsize=(max(5, n * 1.6), 3.5))
    ax.bar(x - w/2, g_hi_vals, w, label='High entropy (top 20%)',  color='#d62728')
    ax.bar(x + w/2, g_lo_vals, w, label='Low entropy (bottom 20%)', color='#1f77b4')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha='right', fontsize=FS - 1)
    ax.set_ylabel('Mean gate value $g$', fontsize=FS)
    ax.set_xlabel('ApproxDA block', fontsize=FS)
    ax.legend(fontsize=FS - 1)
    for xi, (g_hi, g_lo, delta) in zip(x, figD_data):
        ymax = max(g_hi, g_lo)
        ax.annotate(f'Δ={delta:+.3f}', xy=(xi, ymax + 0.01),
                    ha='center', fontsize=FS - 2, color='dimgray')
    fig.tight_layout()
    path_D = os.path.join(args.out_dir, 'fig_D_group_comparison.png')
    fig.savefig(path_D, dpi=DPI)
    plt.close(fig)
    print(f"Fig D saved: {path_D}")

    # ── Figures E and F: per-class / per-category analysis ────────────────
    if cat_names:
        n_cats = len(cat_names)
        x_cat  = np.arange(n_cats)
        label_suffix = args.dataset

        mean_H_cat = np.array([np.mean(v) for v in cat_H_vals])
        mean_g_cat = np.array([np.mean(v) for v in cat_g_vals])

        # normalise to [0,1] for visual comparison on same axis
        def _norm(arr):
            lo, hi = arr.min(), arr.max()
            return (arr - lo) / (hi - lo + 1e-8)

        H_norm = _norm(mean_H_cat)
        g_norm = _norm(mean_g_cat)

        # Figure E: normalised mean H vs mean g per class/category
        fig, ax = plt.subplots(figsize=(max(6, n_cats * 1.3), 4))
        ax.bar(x_cat - w/2, H_norm, w, label='Normalised mean $H(F)$', color='steelblue')
        ax.bar(x_cat + w/2, g_norm, w, label='Normalised mean $g$',     color='#d62728')
        ax.set_xticks(x_cat)
        ax.set_xticklabels(cat_names, rotation=30, ha='right', fontsize=FS - 1)
        ax.set_ylabel('Normalised value (0–1)', fontsize=FS)
        ax.set_xlabel(f'Class / category ({label_suffix})', fontsize=FS)
        ax.legend(fontsize=FS - 1)
        ax.set_title('Per-class: feature entropy vs gate activation (block avg)', fontsize=FS)
        fig.tight_layout()
        path_E = os.path.join(args.out_dir, f'fig_E_{label_suffix}_per_class.png')
        fig.savefig(path_E, dpi=DPI)
        plt.close(fig)
        print(f"Fig E saved: {path_E}")

        # Figure F: per-class Spearman r_s (H vs g, subset of slices)
        r_cat  = [v[0] for v in cat_r]
        p_cat  = [v[1] for v in cat_r]
        colors_r = ['#d62728' if r < 0 else 'steelblue' for r in r_cat]
        fig, ax = plt.subplots(figsize=(max(6, n_cats * 1.3), 4))
        ax.bar(x_cat, r_cat, color=colors_r, alpha=0.85)
        for xi, (r, p) in zip(x_cat, zip(r_cat, p_cat)):
            if not np.isnan(r):
                sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
                ax.text(xi, r + (0.02 if r >= 0 else -0.05), sig,
                        ha='center', fontsize=FS - 1)
        ax.axhline(0,    color='black', linewidth=0.8)
        ax.axhline( 0.3, color='green', linewidth=1.0, linestyle='--', alpha=0.7, label='$r$=0.3')
        ax.axhline(-0.3, color='green', linewidth=1.0, linestyle='--', alpha=0.7)
        ax.set_xticks(x_cat)
        ax.set_xticklabels(cat_names, rotation=30, ha='right', fontsize=FS - 1)
        ax.set_ylabel('Spearman $r_s$ (H vs g, class subset)', fontsize=FS)
        ax.set_xlabel(f'Class / category ({label_suffix})', fontsize=FS)
        ax.legend(fontsize=FS - 1)
        ax.set_title('Per-class entropy–gate correlation', fontsize=FS)
        fig.tight_layout()
        path_F = os.path.join(args.out_dir, f'fig_F_{label_suffix}_per_class_spearman.png')
        fig.savefig(path_F, dpi=DPI)
        plt.close(fig)
        print(f"Fig F saved: {path_F}")

    print(f"\nAll figures written to: {args.out_dir}/")

except ImportError:
    print("\nmatplotlib not available — skipping figures (Spearman results above are still valid)")

# ---------- summary decision ----------
print("\n=== Decision summary ===")
max_r_H   = max(abs(r_H)   for _, r_H, p_H, r_Var, p_Var, *_ in spearman_results)
max_r_Var = max(abs(r_Var) for _, r_H, p_H, r_Var, p_Var, *_ in spearman_results)
best_p_H  = min((p_H  for _, r_H, p_H, *_ in spearman_results if abs(r_H) > 0.3), default=1.0)

# Gate collapse check: if high-H vs low-H delta is negligible across all blocks,
# the gate is not actually adapting regardless of Spearman r value.
# Spearman r can be high on a collapsed gate because rank ordering exists even
# within a [0.497, 0.500] range — the MAGNITUDE of adaptation is what matters.
max_delta = max(abs(d[2]) for d in figD_data)
gate_collapsed = max_delta < 0.005   # less than 0.5% absolute gate swing

print(f"  Gate collapse check: max |delta| across blocks = {max_delta:.4f}  "
      f"({'COLLAPSED — gate is inert' if gate_collapsed else 'OK — gate is adapting'})")

if gate_collapsed:
    print("\nCase 0 — GATE COLLAPSED (g ≈ 0.5 regardless of entropy, delta ≈ 0).")
    print("  Spearman r may look high but reflects rank ordering on negligible magnitude variation.")
    print("  The gate_fc learned near-zero weights → σ(0) ≈ 0.5 for all inputs.")
    print("  -> Phase 2 entropy gate will NOT help — the gate itself is the problem.")
    print("  -> Accuracy gap is from windowed attention losing global context, not from gating.")
    print("  -> Recommended ablation: --gate_mode fixed to confirm gate is inert,")
    print("     then investigate larger window (M=14) or higher rank (r=64).")
elif max_r_H > 0.5 and best_p_H < 0.01:
    print("Case 1 — STRONG H correlation: gate already tracks entropy implicitly.")
    print("  -> Proceed to Phase 2 (entropy gate). Paper narrative is strong.")
elif max_r_H > 0.2 and best_p_H < 0.01:
    print("Case 2 — WEAK H correlation: entropy has supplementary value.")
    print("  -> Proceed to Phase 2. Narrative: explicit entropy strengthens implicit signal.")
elif max_r_H < 0.1:
    if max_r_Var > 0.2:
        print("Case 3 — H near zero, but Var correlates.")
        print("  -> Implement Plan B1 (variance gate): gate_fc = Linear(C+1, C) with var input.")
    else:
        print("Case 3 — H near zero, Var also weak.")
        print("  -> Implement Plan B3 (multi-statistic gate): gate_fc = Linear(C+2, C).")
else:
    neg_blocks = [name for name, r_H, *_ in spearman_results if r_H < -0.2]
    if neg_blocks:
        print("Case 4 — NEGATIVE H correlation in:", neg_blocks)
        print("  -> High entropy -> more CAM (inverts prior). Revise narrative.")
    else:
        print("Case 2/3 mixed — check per-block plots to decide.")

print(f"\n  max |r_H| = {max_r_H:.3f}  |  max |r_Var| = {max_r_Var:.3f}  |  max gate |delta| = {max_delta:.4f}")

if cat_names:
    print("\n=== Per-class decision hints ===")
    for cname, (r, p) in zip(cat_names, cat_r):
        if np.isnan(r):
            continue
        flag = "STRONG" if abs(r) > 0.3 and p < 0.01 else ("weak" if abs(r) > 0.1 else "none")
        print(f"  {cname:20s}  r_s={r:+.4f}  p={p:.2e}  [{flag}]")
