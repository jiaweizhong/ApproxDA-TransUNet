"""
Generate paper figures for gate collapse (fig:gate_collapse) and entropy scatter
(fig:entropy) from one or two gate=learn checkpoints.

Works with ANY dataset (Synapse, Kvasir, ISIC) — use whichever gate=learn
checkpoint is intact. The gate collapse phenomenon is dataset-agnostic.

Outputs (saved to --out_dir, default ../results/paper_figures/):
  fig_gate_collapse.pdf   -- gate value per val-sample, one panel per checkpoint
  fig_entropy_scatter.pdf -- gate g vs feature entropy H scatter per active block

Usage (from experiments/ApproxDA-TransUNet/):

  # ISIC gate=learn M=7 (recommended -- checkpoint definitely intact)
  python generate_gate_figures.py \\
    --dataset ISIC \\
    --ckpt ../model/ApproxDA_ISIC224/ApproxDA_pretrain_R50-ViT-B_16_skip3_epo300_bs24_224/best_model.pth

  # Synapse M=7 gate=learn (if intact)
  python generate_gate_figures.py \\
    --dataset Synapse \\
    --ckpt ../model/ApproxDA_Synapse224/ApproxDA_pretrain_R50-ViT-B_16_skip3_epo300_bs24_224/best_model.pth

  # Two checkpoints side-by-side (e.g. ISIC M=7 + Synapse M=14)
  python generate_gate_figures.py \\
    --dataset  ISIC    --ckpt  <path1> --ckpt_label  "M=7 (ISIC)" \\
    --dataset2 Synapse --ckpt2 <path2> --ckpt2_label "M=14 (Synapse)" \\
    --window_size2 14 --rank2 64
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

# ── args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--ckpt',         type=str, required=True,
                    help='Path to a gate=learn best_model.pth (primary)')
parser.add_argument('--ckpt_label',   type=str, default=None,
                    help='Display label for primary ckpt (e.g. "M=7, ISIC")')
parser.add_argument('--ckpt2',        type=str, default=None,
                    help='Optional second gate=learn checkpoint for side-by-side')
parser.add_argument('--ckpt2_label',  type=str, default=None)
parser.add_argument('--dataset',      type=str, default='ISIC',
                    choices=['Synapse', 'Kvasir', 'ISIC'],
                    help='Dataset for primary checkpoint inference')
parser.add_argument('--dataset2',     type=str, default=None,
                    choices=['Synapse', 'Kvasir', 'ISIC'],
                    help='Dataset for second checkpoint (defaults to --dataset)')
parser.add_argument('--window_size',  type=int, default=7)
parser.add_argument('--rank',         type=int, default=32)
parser.add_argument('--window_size2', type=int, default=14)
parser.add_argument('--rank2',        type=int, default=64)
parser.add_argument('--img_size',     type=int, default=224)
parser.add_argument('--n_skip',       type=int, default=3)
parser.add_argument('--groups',       type=int, default=8)
# data paths (match Lightning AI layout)
parser.add_argument('--volume_path_syn',  default='../data/Synapse/test_vol_h5')
parser.add_argument('--list_dir_syn',     default='./lists/lists_Synapse')
parser.add_argument('--volume_path_kv',   default='../data/Kvasir-SEG')
parser.add_argument('--list_dir_kv',      default='./lists/lists_Kvasir')
parser.add_argument('--volume_path_isic', default='../data/ISIC2018')
parser.add_argument('--list_dir_isic',    default='./lists/lists_ISIC')
parser.add_argument('--out_dir',      default='../results/paper_figures')
args = parser.parse_args()

if args.dataset2 is None:
    args.dataset2 = args.dataset
os.makedirs(args.out_dir, exist_ok=True)


# ── dataset helpers ────────────────────────────────────────────────────────────
_DS_CFG = {
    'Synapse': ('volume_path_syn', 'list_dir_syn', 9,  True),
    'Kvasir':  ('volume_path_kv',  'list_dir_kv',  2,  False),
    'ISIC':    ('volume_path_isic', 'list_dir_isic', 2, False),
}

def _ds_args(name):
    vp_key, ld_key, n_cls, is_vol = _DS_CFG[name]
    return getattr(args, vp_key), getattr(args, ld_key), n_cls, is_vol

def make_loader(name):
    vol_path, list_dir, n_cls, is_vol = _ds_args(name)
    if name == 'Synapse':
        from datasets.dataset_synapse import Synapse_dataset
        db = Synapse_dataset(base_dir=vol_path, split='test_vol', list_dir=list_dir)
    elif name == 'Kvasir':
        from datasets.dataset_kvasir import Kvasir_dataset
        db = Kvasir_dataset(base_dir=vol_path, split='test_vol', list_dir=list_dir)
    else:
        from datasets.dataset_isic import ISIC_dataset
        db = ISIC_dataset(base_dir=vol_path, split='test_vol', list_dir=list_dir)
    return DataLoader(db, batch_size=1, shuffle=False, num_workers=1), n_cls, is_vol


# ── safe checkpoint loader ─────────────────────────────────────────────────────
def safe_load(ckpt_path):
    if not os.path.exists(ckpt_path):
        sys.exit(f"Checkpoint not found:\n  {ckpt_path}")
    try:
        return torch.load(ckpt_path, map_location='cpu', weights_only=False)
    except RuntimeError as e:
        if 'zip' in str(e).lower() or 'central directory' in str(e).lower():
            sys.exit(
                f"Checkpoint corrupted (truncated zip):\n  {ckpt_path}\n"
                f"Use --dataset ISIC --ckpt <isic_best_model.pth> instead.\n"
                f"Error: {e}"
            )
        raise


# ── model builder ─────────────────────────────────────────────────────────────
def build_model(ckpt_path, window_size, rank, num_classes):
    cfg = CONFIGS_ViT_seg['R50-ViT-B_16']
    cfg.n_classes   = num_classes
    cfg.n_skip      = args.n_skip
    cfg.window_size = window_size
    cfg.rank        = rank
    cfg.groups      = args.groups
    cfg.gate_mode   = 'learn'
    cfg.patches.grid = (args.img_size // 16, args.img_size // 16)

    net = ApproxDATransUNet(cfg, img_size=args.img_size, num_classes=num_classes)
    state = safe_load(ckpt_path)
    state = {k.replace('.ada_block.', '.approx_block.'): v for k, v in state.items()}
    net.load_state_dict(state)
    net.eval()
    if torch.cuda.is_available():
        net = net.cuda()
    print(f"  Loaded ({num_classes} cls): {ckpt_path}")
    return net


# ── hook-based gate + entropy collection ──────────────────────────────────────
def collect_gate_entropy(net, dataset_name):
    loader, num_classes, is_vol = make_loader(dataset_name)
    blocks = [(n, m) for n, m in net.named_modules() if isinstance(m, ApproxDABlock)]
    records = {i: {'H': [], 'g': []} for i in range(len(blocks))}

    def make_hook(idx):
        def hook(module, inp, _out):
            x = inp[0].detach()
            prob = F.softmax(x.flatten(2), dim=-1)
            H = (-prob * torch.log(prob + 1e-6)).sum(-1).mean(-1).cpu()
            with torch.no_grad():
                gap = module.pool(x).view(x.shape[0], -1)
                g = torch.sigmoid(module.gate_fc(gap)).mean(dim=1).cpu()
            records[idx]['H'].append(H)
            records[idx]['g'].append(g)
        return hook

    hooks = [blk.register_forward_hook(make_hook(i)) for i, (_, blk) in enumerate(blocks)]

    def _fwd(img_2d):
        h, w = img_2d.shape
        if h != args.img_size or w != args.img_size:
            img_2d = nd_zoom(img_2d, (args.img_size / h, args.img_size / w), order=3)
        t = torch.from_numpy(img_2d.astype(np.float32)).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
        if torch.cuda.is_available():
            t = t.cuda()
        with torch.no_grad():
            net(t)

    print(f"  Running inference on {len(loader)} {dataset_name} samples ...")
    for sample in loader:
        if is_vol:
            vol = sample['image'].squeeze(0).cpu().numpy()
            for s in range(vol.shape[0]):
                _fwd(vol[s].astype(np.float32))
        else:
            img = sample['image'].squeeze(0).cpu().numpy()
            _fwd((img[0] if img.ndim == 3 else img).astype(np.float32))

    for h in hooks:
        h.remove()

    active = [i for i in range(len(blocks)) if records[i]['H']]
    if not active:
        sys.exit("No hooks fired — check that model loaded correctly.")

    n = len(torch.cat(records[active[0]]['H']))
    g_avg, H_avg = np.zeros(n), np.zeros(n)
    for i in active:
        g_avg += torch.cat(records[i]['g']).numpy()
        H_avg += torch.cat(records[i]['H']).numpy()
    g_avg /= len(active)
    H_avg /= len(active)

    block_data = []
    for i in active:
        short = '.'.join(blocks[i][0].split('.')[-2:]) if '.' in blocks[i][0] else blocks[i][0]
        H_all = torch.cat(records[i]['H']).numpy()
        g_all = torch.cat(records[i]['g']).numpy()
        r, p  = stats.spearmanr(H_all, g_all)
        block_data.append((short, H_all, g_all, r, p))

    print(f"  Gate range [{g_avg.min():.4f}, {g_avg.max():.4f}]  "
          f"delta={g_avg.max()-g_avg.min():.4f}  mean={g_avg.mean():.4f}")
    return g_avg, H_avg, block_data


# ── collect data ───────────────────────────────────────────────────────────────
configs = []

print(f"\n=== Primary: {args.dataset} M={args.window_size} ===")
_, n_cls1, _ = make_loader(args.dataset)
net1 = build_model(args.ckpt, args.window_size, args.rank, n_cls1)
g1, H1, bd1 = collect_gate_entropy(net1, args.dataset)
label1 = args.ckpt_label or f"M={args.window_size} ({args.dataset}, gate=learn)"
configs.append((label1, g1, H1, bd1))
del net1
if torch.cuda.is_available():
    torch.cuda.empty_cache()

if args.ckpt2:
    print(f"\n=== Secondary: {args.dataset2} M={args.window_size2} ===")
    _, n_cls2, _ = make_loader(args.dataset2)
    net2 = build_model(args.ckpt2, args.window_size2, args.rank2, n_cls2)
    g2, H2, bd2 = collect_gate_entropy(net2, args.dataset2)
    label2 = args.ckpt2_label or f"M={args.window_size2} ({args.dataset2}, gate=learn)"
    configs.append((label2, g2, H2, bd2))
    del net2
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ── figures ────────────────────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    DPI    = 300
    FS     = 11
    COLORS = ['#d62728', '#1f77b4']

    # fig:gate_collapse
    n_panels = len(configs)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 3.5), sharey=True)
    if n_panels == 1:
        axes = [axes]

    for ax, (lbl, g, H, _), col in zip(axes, configs, COLORS):
        x = np.arange(len(g))
        ax.plot(x, g, color=col, linewidth=0.5, alpha=0.7)
        ax.fill_between(x, g.min(), g.max(), alpha=0.12, color=col)
        ax.axhline(0.5, color='black', linewidth=1.1, linestyle='--', alpha=0.7, label='g = 0.5')
        ax.set_xlabel('Validation sample index', fontsize=FS)
        ax.set_ylabel('Mean gate value $g$', fontsize=FS)
        ax.set_ylim(max(0.0, g.mean() - 0.04), min(1.0, g.mean() + 0.04))
        ax.set_title(
            f'{lbl}\nrange [{g.min():.4f}, {g.max():.4f}]  '
            r'$\Delta g$=' + f'{g.max()-g.min():.4f}',
            fontsize=FS
        )
        ax.legend(fontsize=FS - 1)
        ax.tick_params(labelsize=FS - 1)

    fig.suptitle(
        'Gate value $g$ across all validation samples (gate=learn)\n'
        r'gate\_fc learns near-zero weights $\Rightarrow$ $\sigma(0)\approx 0.5$ for all inputs',
        fontsize=FS
    )
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(args.out_dir, f'fig_gate_collapse.{ext}'),
                    dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"\nfig_gate_collapse -> {args.out_dir}/fig_gate_collapse.pdf")

    # fig:entropy scatter (primary checkpoint)
    bd = configs[0][3]
    n_blocks = len(bd)
    cols = min(n_blocks, 4)
    rows = (n_blocks + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
    axes_flat = np.array(axes).flatten() if n_blocks > 1 else [axes]

    for i, (blk_lbl, H_all, g_all, r, p) in enumerate(bd):
        ax = axes_flat[i]
        ax.scatter(H_all, g_all, alpha=0.2, s=4, color='steelblue', rasterized=True)
        ax.set_xlabel('Feature entropy $H(F)$', fontsize=FS)
        ax.set_ylabel('Mean gate $g$', fontsize=FS)
        sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'n.s.'))
        delta = g_all.max() - g_all.min()
        ax.set_title(f'{blk_lbl}\n$r_s$={r:+.3f} {sig}  ' + r'$\Delta g$=' + f'{delta:.4f}',
                     fontsize=FS)
        ax.tick_params(labelsize=FS - 1)

    for ax in axes_flat[n_blocks:]:
        ax.set_visible(False)

    fig.suptitle(
        r'Gate $g$ vs.\ feature entropy $H(F)$ -- ' + configs[0][0] + '\n'
        r'$r_s$ may appear non-zero but $\Delta g < 0.002$ (gate is inert)',
        fontsize=FS
    )
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(args.out_dir, f'fig_entropy_scatter.{ext}'),
                    dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"fig_entropy_scatter -> {args.out_dir}/fig_entropy_scatter.pdf")
    print(f"\nAll gate figures written to: {args.out_dir}/")

except ImportError as e:
    print(f"\nmatplotlib not available ({e}) — gate stats printed above are still valid.")
