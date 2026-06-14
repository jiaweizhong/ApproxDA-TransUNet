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
parser.add_argument('--out', type=str, default='gate_entropy_scatter.png')
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

# ---------- scatter plots ----------
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n = len(spearman_results)
    cols = min(n, 4)
    rows_H   = (n + cols - 1) // cols
    rows_Var = (n + cols - 1) // cols
    fig, all_axes = plt.subplots(rows_H + rows_Var, cols,
                                 figsize=(4 * cols, 4 * (rows_H + rows_Var)))
    all_axes = np.array(all_axes).reshape(-1, cols)
    axes_H   = all_axes[:rows_H].flatten()
    axes_Var = all_axes[rows_H:].flatten()

    for i, (name, r_H, p_H, r_Var, p_Var, H_all, Var_all, g_all) in enumerate(spearman_results):
        short = name.split('.')[-2] + '.' + name.split('.')[-1] if '.' in name else name
        for ax, x_vals, x_label, r, p in [
            (axes_H[i],   H_all,   'Entropy H(F)',   r_H,   p_H),
            (axes_Var[i], Var_all, 'Variance Var(F)', r_Var, p_Var),
        ]:
            ax.scatter(x_vals, g_all, alpha=0.3, s=4)
            ax.set_xlabel(x_label)
            ax.set_ylabel('Mean gate g')
            ax.set_title(f'{short}\nr={r:+.3f} p={p:.1e}')

    for ax in list(axes_H[n:]) + list(axes_Var[n:]):
        ax.set_visible(False)

    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f"\nScatter plot saved to: {args.out}")
except ImportError:
    print("\nmatplotlib not available — skipping scatter plot (Spearman results above are still valid)")

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
