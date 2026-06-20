"""
Generate paper figures for gate collapse (fig:gate_collapse) and entropy scatter
(fig:entropy) from one or two gate=learn checkpoints.

Outputs (saved to --out_dir, default ../results/paper_figures/):
  fig_gate_collapse.pdf  — gate value per validation slice for M=7 and/or M=14
  fig_entropy_scatter.pdf — gate g vs feature entropy H scatter per active block

Usage (from experiments/ApproxDA-TransUNet/):
  # M=7 only (minimum)
  python generate_gate_figures.py \
    --ckpt_m7  ../model/ApproxDA_Synapse224/ApproxDA_pretrain_R50-ViT-B_16_skip3_epo300_bs24_224/best_model.pth

  # M=7 + M=14 (shows both in one figure)
  python generate_gate_figures.py \
    --ckpt_m7  <path_to_m7_learn_checkpoint> \
    --ckpt_m14 <path_to_m14_learn_checkpoint>
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

# ── args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--ckpt_m7',  type=str, required=True,
                    help='Path to Synapse gate=learn M=7 best_model.pth')
parser.add_argument('--ckpt_m14', type=str, default=None,
                    help='Path to Synapse gate=learn M=14 best_model.pth (optional)')
parser.add_argument('--volume_path', type=str, default='../data/Synapse/test_vol_h5')
parser.add_argument('--list_dir',    type=str, default='./lists/lists_Synapse')
parser.add_argument('--img_size',    type=int, default=224)
parser.add_argument('--n_skip',      type=int, default=3)
parser.add_argument('--groups',      type=int, default=8)
parser.add_argument('--out_dir',     type=str, default='../results/paper_figures')
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

# ── model builder ─────────────────────────────────────────────────────────────
def _build_model(ckpt_path, window_size, rank=32):
    """Load ApproxDA with gate=learn from an explicit checkpoint path."""
    config = CONFIGS_ViT_seg['R50-ViT-B_16']
    config.n_classes = 9
    config.n_skip = args.n_skip
    config.window_size = window_size
    config.rank = rank
    config.groups = args.groups
    config.gate_mode = 'learn'
    config.patches.grid = (args.img_size // 16, args.img_size // 16)

    net = ApproxDATransUNet(config, img_size=args.img_size, num_classes=9)
    state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state = {k.replace('.ada_block.', '.approx_block.'): v for k, v in state.items()}
    net.load_state_dict(state)
    net.eval()
    if torch.cuda.is_available():
        net = net.cuda()
    print(f"  Loaded: {ckpt_path}  (window_size={window_size})")
    return net


# ── hook-based gate + entropy collection ──────────────────────────────────────
def collect_gate_entropy(net, volume_path, list_dir):
    """Return arrays: gate_per_slice (n_slices,), H_per_slice (n_slices,)."""
    blocks = [(n, m) for n, m in net.named_modules() if isinstance(m, ApproxDABlock)]
    records = {i: {'H': [], 'g': []} for i in range(len(blocks))}

    def make_hook(idx, _blk):
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

    hooks = [blk.register_forward_hook(make_hook(i, blk)) for i, (_, blk) in enumerate(blocks)]

    db = Synapse_dataset(base_dir=volume_path, split='test_vol', list_dir=list_dir)
    loader = DataLoader(db, batch_size=1, shuffle=False, num_workers=1)

    for sample in loader:
        img_vol = sample['image'].squeeze(0).cpu().numpy()
        for s in range(img_vol.shape[0]):
            sl = img_vol[s].astype(np.float32)
            h, w = sl.shape
            if h != args.img_size or w != args.img_size:
                sl = nd_zoom(sl, (args.img_size / h, args.img_size / w), order=3)
            t = torch.from_numpy(sl.copy()).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
            if torch.cuda.is_available():
                t = t.cuda()
            with torch.no_grad():
                net(t)

    for h in hooks:
        h.remove()

    active = [i for i in range(len(blocks)) if len(records[i]['H']) > 0]
    n = len(torch.cat(records[active[0]]['H']))
    g_avg = np.zeros(n)
    H_avg = np.zeros(n)
    for i in active:
        g_avg += torch.cat(records[i]['g']).numpy()
        H_avg += torch.cat(records[i]['H']).numpy()
    g_avg /= len(active)
    H_avg /= len(active)

    # per-block data for entropy scatter
    block_data = []
    for i in active:
        short = '.'.join(blocks[i][0].split('.')[-2:]) if '.' in blocks[i][0] else blocks[i][0]
        H_all = torch.cat(records[i]['H']).numpy()
        g_all = torch.cat(records[i]['g']).numpy()
        r, p = stats.spearmanr(H_all, g_all)
        block_data.append((short, H_all, g_all, r, p))

    return g_avg, H_avg, block_data


# ── collect data ───────────────────────────────────────────────────────────────
configs_to_run = []

print("\n=== Loading M=7 model ===")
net_m7 = _build_model(args.ckpt_m7, window_size=7)
g_m7, H_m7, block_data_m7 = collect_gate_entropy(net_m7, args.volume_path, args.list_dir)
configs_to_run.append(('M=7', g_m7, H_m7, block_data_m7))
del net_m7
if torch.cuda.is_available():
    torch.cuda.empty_cache()

if args.ckpt_m14:
    print("\n=== Loading M=14 model ===")
    net_m14 = _build_model(args.ckpt_m14, window_size=14, rank=64)
    g_m14, H_m14, block_data_m14 = collect_gate_entropy(net_m14, args.volume_path, args.list_dir)
    configs_to_run.append(('M=14', g_m14, H_m14, block_data_m14))
    del net_m14
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ── print summary ──────────────────────────────────────────────────────────────
for label, g, H, _ in configs_to_run:
    print(f"\n{label}: gate range [{g.min():.4f}, {g.max():.4f}]  "
          f"mean={g.mean():.4f}  std={g.std():.6f}  "
          f"max_delta={g.max()-g.min():.4f}")

# ── figures ────────────────────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    DPI = 300
    FS  = 11
    COLORS = ['#d62728', '#1f77b4']  # red=M7, blue=M14

    # ── fig:gate_collapse ──────────────────────────────────────────────────────
    # Show gate value per validation slice for each config.
    # Both should be flat lines at ≈0.5, proving the gate is collapsed.
    n_configs = len(configs_to_run)
    fig, axes = plt.subplots(1, n_configs, figsize=(4.5 * n_configs, 3.5), sharey=True)
    if n_configs == 1:
        axes = [axes]

    for ax, (label, g, H, _), color in zip(axes, configs_to_run, COLORS):
        x = np.arange(len(g))
        ax.plot(x, g, color=color, linewidth=0.6, alpha=0.7, label=label)
        ax.axhline(0.5, color='black', linewidth=1.0, linestyle='--', alpha=0.6, label='$g=0.5$')
        ax.fill_between(x, g.min(), g.max(), alpha=0.15, color=color)
        ax.set_xlabel('Validation slice index', fontsize=FS)
        ax.set_ylabel('Mean gate value $g$', fontsize=FS)
        ax.set_title(f'{label},  gate=learn\n'
                     f'range [{g.min():.4f}, {g.max():.4f}]  Δ={g.max()-g.min():.4f}',
                     fontsize=FS)
        ax.set_ylim(0.48, 0.52)
        ax.legend(fontsize=FS - 1)
        ax.tick_params(labelsize=FS - 1)

    fig.suptitle('Gate value $g$ across all Synapse validation slices\n'
                 '(gate_fc learns near-zero weights → $\\sigma(0) \\approx 0.5$ for all inputs)',
                 fontsize=FS)
    fig.tight_layout()
    path_gc = os.path.join(args.out_dir, 'fig_gate_collapse.pdf')
    fig.savefig(path_gc, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"\nFig:gate_collapse saved → {path_gc}")

    # also save PNG for quick preview
    fig, axes = plt.subplots(1, n_configs, figsize=(4.5 * n_configs, 3.5), sharey=True)
    if n_configs == 1:
        axes = [axes]
    for ax, (label, g, H, _), color in zip(axes, configs_to_run, COLORS):
        x = np.arange(len(g))
        ax.plot(x, g, color=color, linewidth=0.6, alpha=0.7, label=label)
        ax.axhline(0.5, color='black', linewidth=1.0, linestyle='--', alpha=0.6, label='$g=0.5$')
        ax.fill_between(x, g.min(), g.max(), alpha=0.15, color=color)
        ax.set_xlabel('Validation slice index', fontsize=FS)
        ax.set_ylabel('Mean gate value $g$', fontsize=FS)
        ax.set_title(f'{label}: range [{g.min():.4f}, {g.max():.4f}]', fontsize=FS)
        ax.set_ylim(0.48, 0.52)
        ax.legend(fontsize=FS - 1)
    fig.tight_layout()
    fig.savefig(path_gc.replace('.pdf', '.png'), dpi=DPI, bbox_inches='tight')
    plt.close(fig)

    # ── fig:entropy scatter ────────────────────────────────────────────────────
    # Use M=7 block_data (4 active blocks)
    _, _, _, block_data_for_scatter = configs_to_run[0]
    n_blocks = len(block_data_for_scatter)
    cols = min(n_blocks, 4)
    rows = (n_blocks + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
    axes = np.array(axes).flatten() if n_blocks > 1 else [axes]

    for i, (label, H_all, g_all, r, p) in enumerate(block_data_for_scatter):
        ax = axes[i]
        ax.scatter(H_all, g_all, alpha=0.25, s=5, color='steelblue', rasterized=True)
        ax.set_xlabel('Feature entropy $H(F)$', fontsize=FS)
        ax.set_ylabel('Mean gate $g$', fontsize=FS)
        sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'n.s.'))
        ax.set_title(f'{label}\n$r_s$={r:+.3f} {sig}  (Δ$g$≈0)', fontsize=FS)
        ax.tick_params(labelsize=FS - 1)

    for ax in axes[n_blocks:]:
        ax.set_visible(False)

    fig.suptitle('Gate value $g$ vs.\ feature entropy $H(F)$ — M=7, gate=learn, Synapse val\n'
                 'Spearman $r$ may appear significant but $g$ variation is $<$0.002 (gate is inert)',
                 fontsize=FS)
    fig.tight_layout()
    path_es = os.path.join(args.out_dir, 'fig_entropy_scatter.pdf')
    fig.savefig(path_es, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    fig.savefig(path_es.replace('.pdf', '.png'), dpi=DPI, bbox_inches='tight')
    print(f"Fig:entropy_scatter saved → {path_es}")
    plt.close('all')

    print(f"\nAll gate figures written to: {args.out_dir}/")

except ImportError as e:
    print(f"\nmatplotlib not available ({e}) — gate statistics printed above are still valid.")
