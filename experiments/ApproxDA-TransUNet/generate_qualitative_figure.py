"""
Generate qualitative segmentation figure for the paper (fig:qualitative).

Two modes selected by --mode:
  window_ablation  (default)
      Shows the SAME Synapse case under M=7, M=28, M=112 (gate=pam) to visually
      demonstrate the non-monotonic DSC-vs-M curve and the CS hypothesis.
      Columns: Input | GT | M=7 (78.64%) | M=28 (80.94%) | M=112 (79.44%)
      Needs: --ckpt_syn_m7_pam, --ckpt_syn_m28_pam, --ckpt_syn_m112_pam

  cross_task
      One representative case per dataset (Synapse / Kvasir / ISIC).
      Columns: Input | GT | Prediction
      Needs: --ckpt_syn_best, --ckpt_kvasir_best, --ckpt_isic_best

Common args:
  --volume_path_syn   path to Synapse test_vol_h5 (default ../data/Synapse/test_vol_h5)
  --volume_path_kv    path to Kvasir-SEG dir     (default ../data/Kvasir-SEG)
  --volume_path_isic  path to ISIC2018 dir        (default ../data/ISIC2018)
  --list_dir_syn      (default ./lists/lists_Synapse)
  --list_dir_kv       (default ./lists/lists_Kvasir)
  --list_dir_isic     (default ./lists/lists_ISIC)
  --out_dir           (default ../results/paper_figures)
  --dpi               (default 300)

Examples:
  # window ablation (most compelling for CS story)
  python generate_qualitative_figure.py --mode window_ablation \
    --ckpt_syn_m7_pam   <path/m7_pam/best_model.pth> \
    --ckpt_syn_m28_pam  <path/m28_pam/best_model.pth> \
    --ckpt_syn_m112_pam <path/m112_pam/best_model.pth>

  # cross-task (shows all three datasets)
  python generate_qualitative_figure.py --mode cross_task \
    --ckpt_syn_best   <path/m28_pam/best_model.pth> \
    --ckpt_kvasir_best <path/kvasir_m56_pam/best_model.pth> \
    --ckpt_isic_best  <path/isic_m7_learn/best_model.pth>
"""

import argparse, os, sys
import numpy as np
import torch
from scipy.ndimage import zoom as nd_zoom
from torch.utils.data import DataLoader

from Architecture.ApproxDATransUNet import ApproxDATransUNet
from Architecture.ApproxDATransUNet import CONFIGS as CONFIGS_ViT_seg

# ── Synapse organ colours (one per class 1–8) ──────────────────────────────────
SYNAPSE_COLORS = np.array([
    [0,   0,   0  ],   # 0 bg
    [255, 100, 100],   # 1 aorta
    [100, 200, 100],   # 2 gallbladder
    [100, 100, 255],   # 3 kidney L
    [255, 165,   0],   # 4 kidney R
    [180,   0, 180],   # 5 liver
    [  0, 200, 200],   # 6 pancreas
    [255, 215,   0],   # 7 spleen
    [200, 200, 200],   # 8 stomach
], dtype=np.uint8)

BINARY_COLORS = np.array([
    [0,   0,   0  ],   # 0 bg
    [255, 200,  50],   # 1 foreground
], dtype=np.uint8)

# ── args ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='window_ablation',
                    choices=['window_ablation', 'cross_task'])

# window_ablation checkpoints
parser.add_argument('--ckpt_syn_m7_pam',   default=None)
parser.add_argument('--ckpt_syn_m28_pam',  default=None)
parser.add_argument('--ckpt_syn_m112_pam', default=None)

# cross_task checkpoints
parser.add_argument('--ckpt_syn_best',    default=None)
parser.add_argument('--ckpt_kvasir_best', default=None)
parser.add_argument('--ckpt_isic_best',   default=None)

# data paths
parser.add_argument('--volume_path_syn',  default='../data/Synapse/test_vol_h5')
parser.add_argument('--volume_path_kv',   default='../data/Kvasir-SEG')
parser.add_argument('--volume_path_isic', default='../data/ISIC2018')
parser.add_argument('--list_dir_syn',     default='./lists/lists_Synapse')
parser.add_argument('--list_dir_kv',      default='./lists/lists_Kvasir')
parser.add_argument('--list_dir_isic',    default='./lists/lists_ISIC')

parser.add_argument('--img_size', type=int, default=224)
parser.add_argument('--n_skip',   type=int, default=3)
parser.add_argument('--groups',   type=int, default=8)
parser.add_argument('--out_dir',  default='../results/paper_figures')
parser.add_argument('--dpi',      type=int, default=300)
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)


# ── model loader ───────────────────────────────────────────────────────────────
def load_model(ckpt_path, window_size, rank=32, gate_mode='pam', num_classes=9):
    cfg = CONFIGS_ViT_seg['R50-ViT-B_16']
    cfg.n_classes = num_classes
    cfg.n_skip = args.n_skip
    cfg.window_size = window_size
    cfg.rank = rank
    cfg.groups = args.groups
    cfg.gate_mode = gate_mode
    cfg.patches.grid = (args.img_size // 16, args.img_size // 16)

    net = ApproxDATransUNet(cfg, img_size=args.img_size, num_classes=num_classes)
    state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state = {k.replace('.ada_block.', '.approx_block.'): v for k, v in state.items()}
    net.load_state_dict(state, strict=False)
    net.eval()
    if torch.cuda.is_available():
        net = net.cuda()
    print(f"  Loaded: {ckpt_path}")
    return net


# ── per-slice prediction ───────────────────────────────────────────────────────
def predict_slice(net, img_slice):
    """img_slice: H×W float32 numpy [0,1]. Returns pred: H×W uint8."""
    h, w = img_slice.shape
    if h != args.img_size or w != args.img_size:
        img_slice = nd_zoom(img_slice, (args.img_size / h, args.img_size / w), order=3)
    t = torch.from_numpy(img_slice.astype(np.float32)).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
    if torch.cuda.is_available():
        t = t.cuda()
    with torch.no_grad():
        out = net(t)
    pred = out.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    if h != args.img_size or w != args.img_size:
        pred = nd_zoom(pred, (h / args.img_size, w / args.img_size), order=0).astype(np.uint8)
    return pred


def dsc_binary(pred, gt):
    p = (pred > 0).flatten().astype(float)
    g = (gt > 0).flatten().astype(float)
    inter = (p * g).sum()
    return 2 * inter / (p.sum() + g.sum() + 1e-6)


def seg_to_rgb(seg, colors):
    h, w = seg.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(len(colors)):
        mask = seg == c
        rgb[mask] = colors[c]
    return rgb


def overlay(img_gray, seg_rgb, alpha=0.55):
    """Blend grayscale image with coloured segmentation overlay."""
    bg = np.stack([img_gray] * 3, axis=-1)   # H×W×3, float [0,1]
    out = (1 - alpha) * bg + alpha * (seg_rgb / 255.0)
    return np.clip(out, 0, 1)


# ── Synapse: pick a representative slice from a volume ────────────────────────
def pick_synapse_slice(img_vol, lbl_vol):
    """Return (s, img_slice, lbl_slice) for the slice with the most foreground pixels."""
    fg_counts = [(lbl_vol[s] > 0).sum() for s in range(img_vol.shape[0])]
    # pick slice near the 75th percentile of fg_counts (a rich but not-extreme case)
    thresh = np.percentile([c for c in fg_counts if c > 0], 75)
    candidates = [s for s, c in enumerate(fg_counts) if c >= thresh * 0.9]
    s = candidates[len(candidates) // 2]
    return s, img_vol[s], lbl_vol[s]


# ── load Synapse volumes ───────────────────────────────────────────────────────
def load_synapse_cases(n_cases=2):
    from datasets.dataset_synapse import Synapse_dataset
    db = Synapse_dataset(base_dir=args.volume_path_syn, split='test_vol',
                         list_dir=args.list_dir_syn)
    loader = DataLoader(db, batch_size=1, shuffle=False, num_workers=1)
    cases = []
    for i, s in enumerate(loader):
        if i >= n_cases:
            break
        img = s['image'].squeeze(0).cpu().numpy()
        lbl = s['label'].squeeze(0).cpu().numpy()
        name = s['case_name'][0]
        cases.append((name, img, lbl))
    return cases


# ── load Kvasir/ISIC images ───────────────────────────────────────────────────
def load_binary_cases(dataset, n_cases=2):
    if dataset == 'Kvasir':
        from datasets.dataset_kvasir import Kvasir_dataset
        db = Kvasir_dataset(base_dir=args.volume_path_kv, list_dir=args.list_dir_kv,
                            split='test_vol')
    else:
        from datasets.dataset_isic import ISIC_dataset
        db = ISIC_dataset(base_dir=args.volume_path_isic, list_dir=args.list_dir_isic,
                          split='test_vol')
    loader = DataLoader(db, batch_size=1, shuffle=False, num_workers=1)
    cases = []
    for i, s in enumerate(loader):
        if i >= n_cases:
            break
        img = s['image'].squeeze().cpu().numpy()   # H×W float
        lbl = s['label'].squeeze().cpu().numpy().astype(np.uint8)
        name = s['case_name'][0]
        # skip nearly-blank images (fg < 1%)
        if (lbl > 0).mean() < 0.01:
            n_cases += 1
            if i < n_cases:
                continue
        cases.append((name, img, lbl))
    return cases[:n_cases]


# ══════════════════════════════════════════════════════════════════════════════
# MODE 1: window_ablation — same Synapse case under M=7, M=28, M=112
# ══════════════════════════════════════════════════════════════════════════════
if args.mode == 'window_ablation':
    for arg_name in ('ckpt_syn_m7_pam', 'ckpt_syn_m28_pam', 'ckpt_syn_m112_pam'):
        if getattr(args, arg_name) is None:
            sys.exit(f"--{arg_name} is required for window_ablation mode. "
                     f"(use --ckpt_syn_m7_pam, --ckpt_syn_m28_pam, --ckpt_syn_m112_pam)")

    configs = [
        (7,   32, 'pam', args.ckpt_syn_m7_pam,   '78.64%'),
        (28,  32, 'pam', args.ckpt_syn_m28_pam,  '80.94%'),
        (112, 32, 'pam', args.ckpt_syn_m112_pam, '79.44%'),
    ]

    print("\nLoading Synapse cases …")
    cases = load_synapse_cases(n_cases=2)

    # collect predictions per config
    all_preds = {}  # (M, case_idx) -> pred_slice
    all_slices = {}  # case_idx -> (img_slice, lbl_slice)
    for M, r, gm, ckpt, _ in configs:
        net = load_model(ckpt, window_size=M, rank=r, gate_mode=gm, num_classes=9)
        for ci, (name, img_vol, lbl_vol) in enumerate(cases):
            s_idx, img_sl, lbl_sl = pick_synapse_slice(img_vol, lbl_vol)
            all_slices[ci] = (img_sl, lbl_sl)
            pred = predict_slice(net, img_sl)
            all_preds[(M, ci)] = pred
            d_gt  = sum(dsc_binary((lbl_sl == c).astype(np.uint8), (lbl_sl == c))
                        for c in range(1, 9)) / 8
            d_our = sum(dsc_binary((pred == c).astype(np.uint8), (lbl_sl == c))
                        for c in range(1, 9)) / 8
            print(f"  M={M}, case {name} slice {s_idx}: DSC={d_our:.3f}")
        del net
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # draw figure
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n_cases = len(cases)
    n_cols = 2 + len(configs)   # Input, GT, + one per M
    fig, axes = plt.subplots(n_cases, n_cols,
                             figsize=(3.0 * n_cols, 3.2 * n_cases))
    if n_cases == 1:
        axes = axes[np.newaxis, :]

    col_titles = ['Input', 'Ground Truth'] + [f'$M$={M}\n(DSC {dsc})'
                                               for M, _, _, _, dsc in configs]
    for ci, (name, img_vol, lbl_vol) in enumerate(cases):
        img_sl, lbl_sl = all_slices[ci]
        norm_img = (img_sl - img_sl.min()) / (img_sl.max() - img_sl.min() + 1e-6)

        for col, ax in enumerate(axes[ci]):
            ax.axis('off')
            if col == 0:
                # input image
                ax.imshow(norm_img, cmap='gray', vmin=0, vmax=1)
                if ci == 0:
                    ax.set_title(col_titles[col], fontsize=10, fontweight='bold')
                ax.set_ylabel(f'Case {ci+1}', fontsize=9, rotation=90, va='center')
            elif col == 1:
                # ground truth overlay
                gt_rgb = seg_to_rgb(lbl_sl, SYNAPSE_COLORS)
                ax.imshow(overlay(norm_img, gt_rgb))
                if ci == 0:
                    ax.set_title(col_titles[col], fontsize=10, fontweight='bold')
            else:
                M = configs[col - 2][0]
                pred = all_preds[(M, ci)]
                pred_rgb = seg_to_rgb(pred, SYNAPSE_COLORS)
                ax.imshow(overlay(norm_img, pred_rgb))
                # per-case DSC annotation
                d = sum(dsc_binary((pred == c).astype(np.uint8), (lbl_sl == c))
                        for c in range(1, 9)) / 8
                ax.set_title(col_titles[col] if ci == 0 else '', fontsize=10, fontweight='bold')
                ax.text(0.03, 0.03, f'DSC={d:.3f}', transform=ax.transAxes,
                        fontsize=8, color='white', va='bottom',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.6))

    fig.suptitle('Window sensitivity on Synapse (high CS): non-monotonic DSC-vs-$M$ curve\n'
                 'Under-context ($M$=7) and over-context ($M$=112) both underperform optimal ($M$=28)',
                 fontsize=11)
    fig.tight_layout()
    out_path = os.path.join(args.out_dir, 'fig_qualitative_window_ablation.pdf')
    fig.savefig(out_path, dpi=args.dpi, bbox_inches='tight')
    fig.savefig(out_path.replace('.pdf', '.png'), dpi=args.dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"\nFig:qualitative (window_ablation) saved → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# MODE 2: cross_task — one case per dataset
# ══════════════════════════════════════════════════════════════════════════════
elif args.mode == 'cross_task':
    required = {
        'Synapse':     ('ckpt_syn_best',    9, 28,  32, 'pam'),
        'Kvasir-SEG':  ('ckpt_kvasir_best', 2, 56,  32, 'pam'),
        'ISIC 2018':   ('ckpt_isic_best',   2,  7,  32, 'learn'),
    }
    for ds, (arg_name, *_) in required.items():
        if getattr(args, arg_name) is None:
            sys.exit(f"--{arg_name} is required for cross_task mode.")

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n_rows = 3
    n_cols = 3   # Input | GT | Prediction
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(9.5, 3.5 * n_rows))

    row_labels = list(required.keys())
    col_labels  = ['Input', 'Ground Truth', 'ApproxDA Prediction']
    dscs_row = {}

    row = 0
    # ── Synapse ──────────────────────────────────────────────────────────────
    arg_name, num_cls, M, r, gm = required['Synapse']
    net = load_model(getattr(args, arg_name), M, r, gm, num_cls)
    cases = load_synapse_cases(n_cases=1)
    name, img_vol, lbl_vol = cases[0]
    s_idx, img_sl, lbl_sl = pick_synapse_slice(img_vol, lbl_vol)
    pred = predict_slice(net, img_sl)
    del net; torch.cuda.empty_cache() if torch.cuda.is_available() else None

    norm_img = (img_sl - img_sl.min()) / (img_sl.max() - img_sl.min() + 1e-6)
    d = sum(dsc_binary((pred == c).astype(np.uint8), (lbl_sl == c)) for c in range(1, 9)) / 8
    dscs_row['Synapse'] = d

    axes[row, 0].imshow(norm_img, cmap='gray')
    axes[row, 1].imshow(overlay(norm_img, seg_to_rgb(lbl_sl, SYNAPSE_COLORS)))
    axes[row, 2].imshow(overlay(norm_img, seg_to_rgb(pred,   SYNAPSE_COLORS)))
    axes[row, 2].text(0.03, 0.03, f'DSC={d:.3f}', transform=axes[row, 2].transAxes,
                      fontsize=9, color='white', va='bottom',
                      bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.6))

    # ── Kvasir ───────────────────────────────────────────────────────────────
    row = 1
    arg_name, num_cls, M, r, gm = required['Kvasir-SEG']
    net = load_model(getattr(args, arg_name), M, r, gm, num_cls)
    cases_kv = load_binary_cases('Kvasir', n_cases=1)
    name_kv, img_kv, lbl_kv = cases_kv[0]
    pred_kv = predict_slice(net, img_kv)
    del net; torch.cuda.empty_cache() if torch.cuda.is_available() else None

    norm_kv = (img_kv - img_kv.min()) / (img_kv.max() - img_kv.min() + 1e-6)
    d_kv = dsc_binary(pred_kv, lbl_kv)
    dscs_row['Kvasir-SEG'] = d_kv

    axes[row, 0].imshow(norm_kv, cmap='gray')
    axes[row, 1].imshow(overlay(norm_kv, seg_to_rgb(lbl_kv, BINARY_COLORS)))
    axes[row, 2].imshow(overlay(norm_kv, seg_to_rgb(pred_kv, BINARY_COLORS)))
    axes[row, 2].text(0.03, 0.03, f'DSC={d_kv:.3f}', transform=axes[row, 2].transAxes,
                      fontsize=9, color='white', va='bottom',
                      bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.6))

    # ── ISIC ─────────────────────────────────────────────────────────────────
    row = 2
    arg_name, num_cls, M, r, gm = required['ISIC 2018']
    net = load_model(getattr(args, arg_name), M, r, gm, num_cls)
    cases_isic = load_binary_cases('ISIC', n_cases=1)
    name_isic, img_isic, lbl_isic = cases_isic[0]
    pred_isic = predict_slice(net, img_isic)
    del net; torch.cuda.empty_cache() if torch.cuda.is_available() else None

    norm_isic = (img_isic - img_isic.min()) / (img_isic.max() - img_isic.min() + 1e-6)
    d_isic = dsc_binary(pred_isic, lbl_isic)
    dscs_row['ISIC 2018'] = d_isic

    axes[row, 0].imshow(norm_isic, cmap='gray')
    axes[row, 1].imshow(overlay(norm_isic, seg_to_rgb(lbl_isic, BINARY_COLORS)))
    axes[row, 2].imshow(overlay(norm_isic, seg_to_rgb(pred_isic, BINARY_COLORS)))
    axes[row, 2].text(0.03, 0.03, f'DSC={d_isic:.3f}', transform=axes[row, 2].transAxes,
                      fontsize=9, color='white', va='bottom',
                      bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.6))

    # ── labels and formatting ─────────────────────────────────────────────────
    for r_idx, (ax_row, ds) in enumerate(zip(axes, row_labels)):
        ax_row[0].set_ylabel(ds, fontsize=10, fontweight='bold', rotation=90, va='center',
                              labelpad=6)
        for ax in ax_row:
            ax.axis('off')
    for c_idx, title in enumerate(col_labels):
        axes[0, c_idx].set_title(title, fontsize=10, fontweight='bold')

    fig.suptitle('Cross-task qualitative segmentation results\n'
                 'Synapse (high CS, $M$=28) | Kvasir-SEG (low CS, $M$=56) | ISIC (low CS, $M$=7)',
                 fontsize=11)
    fig.tight_layout()
    out_path = os.path.join(args.out_dir, 'fig_qualitative_cross_task.pdf')
    fig.savefig(out_path, dpi=args.dpi, bbox_inches='tight')
    fig.savefig(out_path.replace('.pdf', '.png'), dpi=args.dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"\nFig:qualitative (cross_task) saved → {out_path}")
    for ds, d in dscs_row.items():
        print(f"  {ds}: DSC={d:.4f}")
