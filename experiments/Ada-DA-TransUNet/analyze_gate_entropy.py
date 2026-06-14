"""
Verify whether feature entropy H(F) correlates with learned gate value g
in AdaDABlock. Uses register_forward_hook on an existing checkpoint —
no changes to block.py or AdaDATransUNet.py.

Usage (from experiments/Ada-DA-TransUNet/):
  python analyze_gate_entropy.py \\
    --vit_name R50-ViT-B_16 --n_skip 3 --max_epochs 300 --batch_size 24 \\
    --window_size 7 --rank 32 --groups 8

Decision threshold (from plan):
  |r| > 0.3 and p < 0.01  -> proceed to Phase 2 (entropy gate implementation)
  |r| < 0.1               -> gate ignores entropy; narrative revision needed
  r < 0                   -> revise claim (channel attention stronger at high entropy)
"""

import argparse
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy import stats

from Architecture.AdaDATransUNet import AdaDATransUNet
from Architecture.AdaDATransUNet import CONFIGS as CONFIGS_ViT_seg
from Architecture.block import AdaDABlock
from datasets.dataset_synapse import Synapse_dataset
from utils import test_single_volume

parser = argparse.ArgumentParser()
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
parser.add_argument('--out_dir', type=str, default='figures',
                    help='directory to save paper figures (created if absent)')
args = parser.parse_args()


# ---------- build snapshot path (mirrors test.py logic) ----------
args.exp = 'AdaDA_Synapse' + str(args.img_size)
snapshot_path = '../model/{}/{}'.format(args.exp, 'AdaDA')
snapshot_path += '_pretrain'
snapshot_path += '_' + args.vit_name
snapshot_path += '_skip' + str(args.n_skip)
snapshot_path += '_epo' + str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
snapshot_path += '_bs' + str(args.batch_size)
snapshot_path += '_lr' + str(args.base_lr) if args.base_lr != 0.01 else ''
snapshot_path += '_' + str(args.img_size)

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
config_vit.disable_gate = False
if args.vit_name.find('R50') != -1:
    config_vit.patches.grid = (
        args.img_size // args.vit_patches_size,
        args.img_size // args.vit_patches_size,
    )

net = AdaDATransUNet(config_vit, img_size=args.img_size, num_classes=args.num_classes)
state = torch.load(checkpoint, map_location='cpu')
net.load_state_dict(state)
net.eval()
if torch.cuda.is_available():
    net = net.cuda()

# ---------- find all AdaDABlock instances ----------
adada_blocks = [(name, m) for name, m in net.named_modules() if isinstance(m, AdaDABlock)]
print("Found {} AdaDABlock(s):".format(len(adada_blocks)))
for name, _ in adada_blocks:
    print(" ", name)

# ---------- register hooks ----------
records = {i: {'H': [], 'Var': [], 'g': []} for i in range(len(adada_blocks))}

def make_hook(idx, blk):
    def hook(module, inp, out):
        x = inp[0].detach()
        # entropy: softmax over spatial dim, mean over channels -> (B,)
        prob = F.softmax(x.flatten(2), dim=-1)
        H = (-prob * torch.log(prob + 1e-6)).sum(-1).mean(-1).cpu()
        # variance: spatial variance per channel, mean over channels -> (B,)
        Var = x.var(dim=[2, 3]).mean(dim=1).cpu()
        # gate value: recompute with no grad -> (B,)
        with torch.no_grad():
            gap = module.pool(x).view(x.shape[0], -1)
            g = torch.sigmoid(module.gate_fc(gap)).mean(dim=1).cpu()
        records[idx]['H'].append(H)
        records[idx]['Var'].append(Var)
        records[idx]['g'].append(g)
    return hook

hooks = []
for i, (name, blk) in enumerate(adada_blocks):
    hooks.append(blk.register_forward_hook(make_hook(i, blk)))

# ---------- run inference through test volumes ----------
db = Synapse_dataset(base_dir=args.volume_path, split='test_vol', list_dir=args.list_dir)
loader = DataLoader(db, batch_size=1, shuffle=False, num_workers=1)
print("Running inference on {} volumes …".format(len(db)))

with torch.no_grad():
    for sample in loader:
        image = sample['image']   # (1, slices, H, W) or (1, 1, H, W)
        label = sample['label']
        # test_single_volume handles slice-by-slice inference; hooks fire per slice
        test_single_volume(
            image, label, net,
            classes=args.num_classes,
            patch_size=[args.img_size, args.img_size],
        )

for h in hooks:
    h.remove()

# ---------- compute Spearman correlations per block ----------
print("\n=== Spearman correlation: H(F) vs gate g ===")
spearman_results = []
for i, (name, _) in enumerate(adada_blocks):
    H_all   = torch.cat(records[i]['H']).numpy()
    Var_all = torch.cat(records[i]['Var']).numpy()
    g_all   = torch.cat(records[i]['g']).numpy()
    r_H,   p_H   = stats.spearmanr(H_all,   g_all)
    r_Var, p_Var = stats.spearmanr(Var_all,  g_all)
    spearman_results.append((name, r_H, p_H, r_Var, p_Var, H_all, Var_all, g_all))
    flag = "PROCEED" if abs(r_H) > 0.3 and p_H < 0.01 else ("WEAK" if abs(r_H) < 0.1 else "CHECK")
    print(f"  Block {i:2d} ({name:40s})  H: r={r_H:+.4f} p={p_H:.2e}  [{flag}]  |  Var: r={r_Var:+.4f} p={p_Var:.2e}")

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

# ---------- save separate paper figures ----------
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(args.out_dir, exist_ok=True)
    DPI = 300
    FS  = 11   # base font size

    n      = len(spearman_results)
    cols   = min(n, 4)
    rows   = (n + cols - 1) // cols
    # short block labels: last two path components
    labels = ['.'.join(name.split('.')[-2:]) if '.' in name else name
              for name, *_ in spearman_results]

    # ── Figure A: gate distribution boxplot per block ──────────────────────
    fig, ax = plt.subplots(figsize=(max(4, n * 1.4), 3.5))
    g_data = [torch.cat(records[i]['g']).numpy() for i in range(n)]
    bp = ax.boxplot(g_data, labels=labels, patch_artist=True, notch=False)
    for patch in bp['boxes']:
        patch.set_facecolor('#a8c8e8')
    ax.set_ylabel('Mean gate value $g$', fontsize=FS)
    ax.set_xlabel('AdaDA block', fontsize=FS)
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
    x = np.arange(n)
    w = 0.35
    r_H_vals   = [r_H   for _, r_H, p_H, r_Var, p_Var, *_ in spearman_results]
    r_Var_vals = [r_Var for _, r_H, p_H, r_Var, p_Var, *_ in spearman_results]
    fig, ax = plt.subplots(figsize=(max(5, n * 1.6), 3.5))
    ax.bar(x - w/2, r_H_vals,   w, label='Entropy $H(F)$',    color='steelblue')
    ax.bar(x + w/2, r_Var_vals, w, label='Variance $\\mathrm{Var}(F)$', color='darkorange')
    ax.axhline(0,   color='black', linewidth=0.8)
    ax.axhline( 0.3, color='green',  linewidth=1.0, linestyle='--', alpha=0.7, label='$r$=0.3 threshold')
    ax.axhline(-0.3, color='green',  linewidth=1.0, linestyle='--', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha='right', fontsize=FS - 1)
    ax.set_ylabel('Spearman $r_s$', fontsize=FS)
    ax.set_xlabel('AdaDA block', fontsize=FS)
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
    ax.set_xlabel('AdaDA block', fontsize=FS)
    ax.legend(fontsize=FS - 1)
    # annotate delta on each pair
    for xi, (g_hi, g_lo, delta) in zip(x, figD_data):
        ymax = max(g_hi, g_lo)
        ax.annotate(f'Δ={delta:+.3f}', xy=(xi, ymax + 0.01),
                    ha='center', fontsize=FS - 2, color='dimgray')
    fig.tight_layout()
    path_D = os.path.join(args.out_dir, 'fig_D_group_comparison.png')
    fig.savefig(path_D, dpi=DPI)
    plt.close(fig)
    print(f"Fig D saved: {path_D}")

    print(f"\nAll figures written to: {args.out_dir}/")

except ImportError:
    print("\nmatplotlib not available — skipping figures (Spearman results and group stats above are still valid)")

# ---------- summary decision ----------
print("\n=== Decision summary ===")
max_r_H   = max(abs(r_H)   for _, r_H, p_H, r_Var, p_Var, *_ in spearman_results)
max_r_Var = max(abs(r_Var) for _, r_H, p_H, r_Var, p_Var, *_ in spearman_results)
best_p_H  = min((p_H  for _, r_H, p_H, *_ in spearman_results if abs(r_H)   > 0.3), default=1.0)

if max_r_H > 0.5 and best_p_H < 0.01:
    print("Case 1 — STRONG H correlation: gate already tracks entropy implicitly.")
    print("  -> Proceed to Phase 2 (entropy gate). Paper narrative is strong.")
elif max_r_H > 0.2 and best_p_H < 0.01:
    print("Case 2 — WEAK H correlation: entropy has supplementary value.")
    print("  -> Proceed to Phase 2. Narrative: explicit entropy strengthens implicit signal.")
elif max_r_H < 0.1:
    if max_r_Var > 0.2:
        print("Case 3 — H near zero, but Var correlates.")
        print("  -> Skip entropy gate. Implement Plan B1 (variance gate): gate_fc = Linear(C+1, C) with var input.")
    else:
        print("Case 3 — H near zero, Var also weak.")
        print("  -> Implement Plan B3 (multi-statistic gate): gate_fc = Linear(C+2, C) with [GAP, H, Var].")
        print("     Let ablation table determine which signal contributes.")
else:
    neg_blocks = [name for name, r_H, *_ in spearman_results if r_H < -0.2]
    if neg_blocks:
        print("Case 4 — NEGATIVE H correlation in:", neg_blocks)
        print("  -> High entropy -> more CAM (inverts prior). Investigate per block.")
        print("     Revise narrative: 'uncertain boundaries prefer channel disambiguation'.")
    else:
        print("Case 2/3 mixed — check per-block plots to decide.")
print(f"\n  max |r_H| = {max_r_H:.3f}  |  max |r_Var| = {max_r_Var:.3f}")
