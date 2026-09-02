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

import argparse, os, sys, json, importlib.util
import numpy as np
import torch
from scipy.ndimage import zoom as nd_zoom
from torch.utils.data import DataLoader

from Architecture.ApproxDATransUNet import ApproxDATransUNet
from Architecture.ApproxDATransUNet import CONFIGS as CONFIGS_ViT_seg

# path to the DA-TransUNet experiment directory (for optional DA column)
DA_ARCH_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "DA-TransUNet")
)

# tracks which cases/slices were selected — written to cases_used.json at the end
CASES_LOG = {}

# ── Synapse organ colours (one per class 1–8) ──────────────────────────────────
SYNAPSE_COLORS = np.array(
    [
        [0, 0, 0],  # 0 bg
        [255, 100, 100],  # 1 aorta
        [100, 200, 100],  # 2 gallbladder
        [100, 100, 255],  # 3 kidney L
        [255, 165, 0],  # 4 kidney R
        [180, 0, 180],  # 5 liver
        [0, 200, 200],  # 6 pancreas
        [255, 215, 0],  # 7 spleen
        [200, 200, 200],  # 8 stomach
    ],
    dtype=np.uint8,
)

BINARY_COLORS = np.array(
    [
        [0, 0, 0],  # 0 bg
        [255, 200, 50],  # 1 foreground
    ],
    dtype=np.uint8,
)

# ── args ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode",
    default="window_ablation",
    choices=["window_ablation", "cross_task", "da_only"],
)
# da_only: re-run DA-TransUNet on the same cases recorded in cases_used.json
# Needs: --ckpt_da_syn, --ckpt_da_kvasir, --ckpt_da_isic
# Reads case selection from --cases_log (default ../results/paper_figures/cases_used.json)

# window_ablation checkpoints
parser.add_argument("--ckpt_syn_m7_pam", default=None)
parser.add_argument("--ckpt_syn_m28_pam", default=None)
parser.add_argument("--ckpt_syn_m112_pam", default=None)

# cross_task checkpoints
parser.add_argument("--ckpt_syn_best", default=None)
parser.add_argument("--ckpt_kvasir_best", default=None)
parser.add_argument("--ckpt_isic_best", default=None)

# data paths
parser.add_argument("--volume_path_syn", default="../../data/Synapse/test_vol_h5")
parser.add_argument("--volume_path_kv", default="../../data/Kvasir-SEG")
parser.add_argument("--volume_path_isic", default="../../data/ISIC2018")
parser.add_argument("--list_dir_syn", default="./lists/lists_Synapse")
parser.add_argument("--list_dir_kv", default="./lists/lists_Kvasir")
parser.add_argument("--list_dir_isic", default="./lists/lists_ISIC")

parser.add_argument("--img_size", type=int, default=224)
parser.add_argument("--n_skip", type=int, default=3)
parser.add_argument("--groups", type=int, default=8)
parser.add_argument("--out_dir", default="../results/paper_figures")
parser.add_argument("--dpi", type=int, default=300)

# DA-TransUNet checkpoints (for da_only mode)
parser.add_argument("--ckpt_da_syn", default=None, help="DA-TransUNet Synapse ckpt")
parser.add_argument("--ckpt_da_kvasir", default=None, help="DA-TransUNet Kvasir ckpt")
parser.add_argument("--ckpt_da_isic", default=None, help="DA-TransUNet ISIC ckpt")
parser.add_argument(
    "--cases_log",
    default="../results/paper_figures/cases_used.json",
    help="JSON written by cross_task, read by da_only",
)
parser.add_argument(
    "--syn_case",
    default=None,
    help="Force a specific Synapse case, e.g. case0022 (default: first in test_vol.txt)",
)
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)


# ── model loader ───────────────────────────────────────────────────────────────
def infer_window_size(state):
    """Read M from proj_r.weight shape: [rank, M*M] -> M = sqrt(shape[1])."""
    for k, v in state.items():
        if "pam.proj_r.weight" in k:
            m = int(round(v.shape[1] ** 0.5))
            return m
    return None


def load_model(ckpt_path, window_size, rank=32, gate_mode="pam", num_classes=9):
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = {k.replace(".ada_block.", ".approx_block."): v for k, v in state.items()}

    detected_m = infer_window_size(state)
    if detected_m is not None and detected_m != window_size:
        print(f"  [auto] window_size {window_size} -> {detected_m} (from checkpoint)")
        window_size = detected_m

    detected_gate = "learn" if any("gate_fc" in k for k in state) else gate_mode
    if detected_gate != gate_mode:
        print(
            f"  [auto] gate_mode '{gate_mode}' -> '{detected_gate}' (from checkpoint)"
        )
        gate_mode = detected_gate

    cfg = CONFIGS_ViT_seg["R50-ViT-B_16"]
    cfg.n_classes = num_classes
    cfg.n_skip = args.n_skip
    cfg.window_size = window_size
    cfg.rank = rank
    cfg.groups = args.groups
    cfg.gate_mode = gate_mode
    cfg.patches.grid = (args.img_size // 16, args.img_size // 16)

    net = ApproxDATransUNet(cfg, img_size=args.img_size, num_classes=num_classes)
    net.load_state_dict(state, strict=False)
    net.eval()
    if torch.cuda.is_available():
        net = net.cuda()
    print(f"  Loaded: {ckpt_path}  (M={window_size}, gate={gate_mode})")
    return net


def load_da_model(ckpt_path, num_classes=9):
    """Load original DA-TransUNet from the sibling DA-TransUNet/Architecture/ directory.

    DATransUNet.py uses relative imports (`from . import configs`), so we must
    add DA_ARCH_PATH to sys.path so Python treats Architecture/ as a package.
    """
    if DA_ARCH_PATH not in sys.path:
        sys.path.insert(0, DA_ARCH_PATH)
    # Force a clean import in case a stale module entry exists
    for key in list(sys.modules.keys()):
        if key.startswith("Architecture"):
            del sys.modules[key]
    from Architecture.DATransUNet import DA_Transformer, CONFIGS as DA_CONFIGS

    cfg = DA_CONFIGS["R50-ViT-B_16"]
    cfg.n_classes = num_classes
    cfg.n_skip = args.n_skip
    cfg.patches.grid = (args.img_size // 16, args.img_size // 16)

    net = DA_Transformer(cfg, img_size=args.img_size, num_classes=num_classes)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    net.load_state_dict(state, strict=False)
    net.eval()
    if torch.cuda.is_available():
        net = net.cuda()
    print(f"  [DA-TransUNet] Loaded: {ckpt_path}")
    return net


# ── per-slice prediction ───────────────────────────────────────────────────────
def predict_slice(net, img_slice):
    """img_slice: H×W float32 numpy (any range). Returns pred: H×W uint8."""
    h, w = img_slice.shape
    if h != args.img_size or w != args.img_size:
        img_slice = nd_zoom(img_slice, (args.img_size / h, args.img_size / w), order=3)
    # normalize to [0, 1] — matches training preprocessing for all datasets
    lo, hi = img_slice.min(), img_slice.max()
    img_slice = (img_slice - lo) / (hi - lo + 1e-6)
    t = (
        torch.from_numpy(img_slice.astype(np.float32))
        .unsqueeze(0)
        .unsqueeze(0)
        .repeat(1, 3, 1, 1)
    )
    if torch.cuda.is_available():
        t = t.cuda()
    with torch.no_grad():
        out = net(t)
    pred = out.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    if h != args.img_size or w != args.img_size:
        pred = nd_zoom(pred, (h / args.img_size, w / args.img_size), order=0).astype(
            np.uint8
        )
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
    bg = np.stack([img_gray] * 3, axis=-1)  # H×W×3, float [0,1]
    out = (1 - alpha) * bg + alpha * (seg_rgb / 255.0)
    return np.clip(out, 0, 1)


# ── Synapse: pick a representative slice from a volume ────────────────────────
def pick_synapse_slice(img_vol, lbl_vol, net=None):
    """Return (s, img_slice, lbl_slice) with the best per-slice DSC.

    If net is provided, runs inference on candidate slices and picks the one
    with the highest mean per-class DSC. Otherwise falls back to fg density.
    """
    fg_counts = [(lbl_vol[s] > 0).sum() for s in range(img_vol.shape[0])]
    non_empty = [c for c in fg_counts if c > 0]
    lo = np.percentile(non_empty, 40)
    hi = np.percentile(non_empty, 90)
    candidates = [s for s, c in enumerate(fg_counts) if lo <= c <= hi]
    if not candidates:
        candidates = list(range(img_vol.shape[0]))

    if net is None:
        s = candidates[len(candidates) // 2]
        return s, img_vol[s], lbl_vol[s]

    best_s, best_dsc = candidates[0], -1.0
    for s in candidates:
        pred = predict_slice(net, img_vol[s].astype(np.float32))
        lbl_s = lbl_vol[s]
        per_class = [
            dsc_binary((pred == c).astype(np.uint8), (lbl_s == c).astype(np.uint8))
            for c in range(1, 9)
            if (lbl_s == c).any()
        ]
        d = float(np.mean(per_class)) if per_class else 0.0
        if d > best_dsc:
            best_dsc, best_s = d, s
    print(f"  Best Synapse slice: {best_s}  per-slice DSC={best_dsc:.3f}")
    CASES_LOG.setdefault("synapse", {})["slice_idx"] = int(best_s)
    return best_s, img_vol[best_s], lbl_vol[best_s]


# ── load Synapse volumes ───────────────────────────────────────────────────────
def load_synapse_cases(n_cases=2, target_name=None):
    """Load Synapse volumes. If target_name is given, load ONLY that case by index
    (avoids iterating through earlier cases whose .h5 files may not exist)."""
    from datasets.dataset_synapse import Synapse_dataset

    db = Synapse_dataset(
        base_dir=args.volume_path_syn, split="test_vol", list_dir=args.list_dir_syn
    )

    if target_name is not None:
        names = [l.strip("\n").strip() for l in db.sample_list]
        if target_name not in names:
            print(f"  [warn] '{target_name}' not in Synapse sample list")
            return []
        idx = names.index(target_name)
        sample = db[idx]  # loads only case0022.npy.h5
        img = sample["image"]
        lbl = sample["label"]
        if hasattr(img, "numpy"):
            img = img.numpy()
        if hasattr(lbl, "numpy"):
            lbl = lbl.numpy()
        return [(target_name, np.array(img), np.array(lbl))]

    loader = DataLoader(db, batch_size=1, shuffle=False, num_workers=0)
    cases = []
    for i, s in enumerate(loader):
        if i >= n_cases:
            break
        img = s["image"].squeeze(0).cpu().numpy()
        lbl = s["label"].squeeze(0).cpu().numpy()
        name = s["case_name"][0]
        cases.append((name, img, lbl))
    return cases


# ── load Kvasir/ISIC images ───────────────────────────────────────────────────
def _make_binary_db(dataset, split):
    if dataset == "Kvasir":
        from datasets.dataset_kvasir import Kvasir_dataset

        return Kvasir_dataset(
            base_dir=args.volume_path_kv, list_dir=args.list_dir_kv, split=split
        )
    else:
        from datasets.dataset_isic import ISIC_dataset

        return ISIC_dataset(
            base_dir=args.volume_path_isic, list_dir=args.list_dir_isic, split=split
        )


def load_binary_cases(dataset, n_cases=1):
    # try common split names until one returns data
    for split in ("test_vol", "test", "val", "train"):
        try:
            db = _make_binary_db(dataset, split)
            if len(db) == 0:
                continue
            print(f"  {dataset}: using split='{split}', {len(db)} samples")
            break
        except Exception:
            continue
    else:
        sys.exit(f"Could not find any valid split for {dataset} dataset.")

    loader = DataLoader(db, batch_size=1, shuffle=False, num_workers=0)
    cases = []
    for s in loader:
        img = s["image"].squeeze().cpu().numpy()
        if img.ndim == 3:
            img = img[0]
        lbl = s["label"].squeeze().cpu().numpy().astype(np.uint8)
        name = s["case_name"][0]
        if (lbl > 0).mean() < 0.01:
            continue
        cases.append((name, img, lbl))
        if len(cases) >= n_cases:
            break

    if not cases:
        # fallback: take the very first sample regardless of fg content
        for s in DataLoader(db, batch_size=1, shuffle=False, num_workers=0):
            img = s["image"].squeeze().cpu().numpy()
            if img.ndim == 3:
                img = img[0]
            lbl = s["label"].squeeze().cpu().numpy().astype(np.uint8)
            cases.append((s["case_name"][0], img, lbl))
            break

    if not cases:
        sys.exit(
            f"No samples loaded for {dataset}. Check --volume_path_kv / --volume_path_isic."
        )
    return cases


# ══════════════════════════════════════════════════════════════════════════════
# MODE 1: window_ablation — same Synapse case under M=7, M=28, M=112
# ══════════════════════════════════════════════════════════════════════════════
if args.mode == "window_ablation":
    for arg_name in ("ckpt_syn_m7_pam", "ckpt_syn_m28_pam", "ckpt_syn_m112_pam"):
        if getattr(args, arg_name) is None:
            sys.exit(
                f"--{arg_name} is required for window_ablation mode. "
                f"(use --ckpt_syn_m7_pam, --ckpt_syn_m28_pam, --ckpt_syn_m112_pam)"
            )

    configs = [
        (7, 32, "pam", args.ckpt_syn_m7_pam, "78.64%"),
        (28, 32, "pam", args.ckpt_syn_m28_pam, "80.94%"),
        (112, 32, "pam", args.ckpt_syn_m112_pam, "79.44%"),
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
            d_gt = (
                sum(
                    dsc_binary((lbl_sl == c).astype(np.uint8), (lbl_sl == c))
                    for c in range(1, 9)
                )
                / 8
            )
            d_our = (
                sum(
                    dsc_binary((pred == c).astype(np.uint8), (lbl_sl == c))
                    for c in range(1, 9)
                )
                / 8
            )
            print(f"  M={M}, case {name} slice {s_idx}: DSC={d_our:.3f}")
        del net
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # draw figure
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_cases = len(cases)
    n_cols = 2 + len(configs)  # Input, GT, + one per M
    fig, axes = plt.subplots(n_cases, n_cols, figsize=(3.0 * n_cols, 3.2 * n_cases))
    if n_cases == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["Input", "Ground Truth"] + [
        f"$M$={M}\n(DSC {dsc})" for M, _, _, _, dsc in configs
    ]
    for ci, (name, img_vol, lbl_vol) in enumerate(cases):
        img_sl, lbl_sl = all_slices[ci]
        norm_img = (img_sl - img_sl.min()) / (img_sl.max() - img_sl.min() + 1e-6)

        for col, ax in enumerate(axes[ci]):
            ax.axis("off")
            if col == 0:
                # input image
                ax.imshow(norm_img, cmap="gray", vmin=0, vmax=1)
                if ci == 0:
                    ax.set_title(col_titles[col], fontsize=10, fontweight="bold")
                ax.set_ylabel(f"Case {ci+1}", fontsize=9, rotation=90, va="center")
            elif col == 1:
                # ground truth overlay
                gt_rgb = seg_to_rgb(lbl_sl, SYNAPSE_COLORS)
                ax.imshow(overlay(norm_img, gt_rgb))
                if ci == 0:
                    ax.set_title(col_titles[col], fontsize=10, fontweight="bold")
            else:
                M = configs[col - 2][0]
                pred = all_preds[(M, ci)]
                pred_rgb = seg_to_rgb(pred, SYNAPSE_COLORS)
                ax.imshow(overlay(norm_img, pred_rgb))
                # per-case DSC annotation
                d = (
                    sum(
                        dsc_binary((pred == c).astype(np.uint8), (lbl_sl == c))
                        for c in range(1, 9)
                    )
                    / 8
                )
                ax.set_title(
                    col_titles[col] if ci == 0 else "", fontsize=10, fontweight="bold"
                )
                ax.text(
                    0.03,
                    0.03,
                    f"DSC={d:.3f}",
                    transform=ax.transAxes,
                    fontsize=8,
                    color="white",
                    va="bottom",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.6),
                )

    fig.suptitle(
        "Window sensitivity on Synapse (high CS): non-monotonic DSC-vs-$M$ curve\n"
        "Under-context ($M$=7) and over-context ($M$=112) both underperform optimal ($M$=28)",
        fontsize=11,
    )
    fig.tight_layout()
    out_path = os.path.join(args.out_dir, "fig_qualitative_window_ablation.pdf")
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    fig.savefig(out_path.replace(".pdf", ".png"), dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFig:qualitative (window_ablation) saved → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# MODE 2: cross_task — one case per dataset
# ══════════════════════════════════════════════════════════════════════════════
elif args.mode == "cross_task":
    required = {
        "Synapse": ("ckpt_syn_best", 9, 28, 32, "pam"),
        "Kvasir-SEG": ("ckpt_kvasir_best", 2, 56, 32, "pam"),
        "ISIC 2018": ("ckpt_isic_best", 2, 7, 32, "learn"),
    }
    for ds, (arg_name, *_) in required.items():
        if getattr(args, arg_name) is None:
            sys.exit(f"--{arg_name} is required for cross_task mode.")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_rows = 3
    n_cols = 3  # Input | GT | Prediction
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(9.5, 3.5 * n_rows))

    row_labels = list(required.keys())
    col_labels = ["Input", "Ground Truth", "ApproxDA Prediction"]
    dscs_row = {}

    row = 0
    # ── Synapse ──────────────────────────────────────────────────────────────
    arg_name, num_cls, M, r, gm = required["Synapse"]
    net = load_model(getattr(args, arg_name), M, r, gm, num_cls)
    cases = load_synapse_cases(n_cases=1)
    name, img_vol, lbl_vol = cases[0]
    s_idx, img_sl, lbl_sl = pick_synapse_slice(img_vol, lbl_vol, net=net)
    pred = predict_slice(net, img_sl)
    del net
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    CASES_LOG["synapse"] = {"case_name": name, "slice_idx": int(s_idx)}
    print(f"  [LOG] Synapse: case={name}, slice={s_idx}")

    norm_img = (img_sl - img_sl.min()) / (img_sl.max() - img_sl.min() + 1e-6)
    d = (
        sum(
            dsc_binary((pred == c).astype(np.uint8), (lbl_sl == c)) for c in range(1, 9)
        )
        / 8
    )
    dscs_row["Synapse"] = d

    axes[row, 0].imshow(norm_img, cmap="gray")
    axes[row, 1].imshow(overlay(norm_img, seg_to_rgb(lbl_sl, SYNAPSE_COLORS)))
    axes[row, 2].imshow(overlay(norm_img, seg_to_rgb(pred, SYNAPSE_COLORS)))
    # Synapse: skip per-slice DSC (misleading — volume mean is 80.94%, see Table)

    # ── Kvasir ───────────────────────────────────────────────────────────────
    row = 1
    arg_name, num_cls, M, r, gm = required["Kvasir-SEG"]
    net = load_model(getattr(args, arg_name), M, r, gm, num_cls)
    cases_kv = load_binary_cases("Kvasir", n_cases=1)
    name_kv, img_kv, lbl_kv = cases_kv[0]
    pred_kv = predict_slice(net, img_kv)
    del net
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    CASES_LOG["kvasir"] = {"case_name": name_kv}
    print(f"  [LOG] Kvasir: case={name_kv}")

    norm_kv = (img_kv - img_kv.min()) / (img_kv.max() - img_kv.min() + 1e-6)
    d_kv = dsc_binary(pred_kv, lbl_kv)
    dscs_row["Kvasir-SEG"] = d_kv

    axes[row, 0].imshow(norm_kv, cmap="gray")
    axes[row, 1].imshow(overlay(norm_kv, seg_to_rgb(lbl_kv, BINARY_COLORS)))
    axes[row, 2].imshow(overlay(norm_kv, seg_to_rgb(pred_kv, BINARY_COLORS)))
    axes[row, 2].text(
        0.03,
        0.03,
        f"DSC={d_kv:.3f}",
        transform=axes[row, 2].transAxes,
        fontsize=9,
        color="white",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.6),
    )

    # ── ISIC ─────────────────────────────────────────────────────────────────
    row = 2
    arg_name, num_cls, M, r, gm = required["ISIC 2018"]
    net = load_model(getattr(args, arg_name), M, r, gm, num_cls)
    cases_isic = load_binary_cases("ISIC", n_cases=1)
    name_isic, img_isic, lbl_isic = cases_isic[0]
    pred_isic = predict_slice(net, img_isic)
    del net
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    CASES_LOG["isic"] = {"case_name": name_isic}
    print(f"  [LOG] ISIC: case={name_isic}")

    norm_isic = (img_isic - img_isic.min()) / (img_isic.max() - img_isic.min() + 1e-6)
    d_isic = dsc_binary(pred_isic, lbl_isic)
    dscs_row["ISIC 2018"] = d_isic

    axes[row, 0].imshow(norm_isic, cmap="gray")
    axes[row, 1].imshow(overlay(norm_isic, seg_to_rgb(lbl_isic, BINARY_COLORS)))
    axes[row, 2].imshow(overlay(norm_isic, seg_to_rgb(pred_isic, BINARY_COLORS)))
    axes[row, 2].text(
        0.03,
        0.03,
        f"DSC={d_isic:.3f}",
        transform=axes[row, 2].transAxes,
        fontsize=9,
        color="white",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.6),
    )

    # ── labels and formatting ─────────────────────────────────────────────────
    for r_idx, (ax_row, ds) in enumerate(zip(axes, row_labels)):
        ax_row[0].set_ylabel(
            ds, fontsize=10, fontweight="bold", rotation=90, va="center", labelpad=6
        )
        for ax in ax_row:
            ax.axis("off")
    for c_idx, title in enumerate(col_labels):
        axes[0, c_idx].set_title(title, fontsize=10, fontweight="bold")

    fig.suptitle(
        "Cross-task qualitative segmentation results\n"
        "Synapse (high CS, $M$=28) | Kvasir-SEG (low CS, $M$=56) | ISIC (low CS, $M$=7)",
        fontsize=11,
    )
    fig.tight_layout()
    out_path = os.path.join(args.out_dir, "fig_qualitative_cross_task.pdf")
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    fig.savefig(out_path.replace(".pdf", ".png"), dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFig:qualitative (cross_task) saved → {out_path}")
    for ds, d in dscs_row.items():
        print(f"  {ds}: DSC={d:.4f}")

    # ── save case log ─────────────────────────────────────────────────────────
    log_path = args.cases_log
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w") as f:
        json.dump(CASES_LOG, f, indent=2)
    print(f"\n[LOG] Case selection saved → {log_path}")
    print(json.dumps(CASES_LOG, indent=2))


# ══════════════════════════════════════════════════════════════════════════════
# MODE 3: da_only — re-run DA-TransUNet on the exact same cases as cross_task
#   Reads:  cases_used.json  (written by cross_task mode)
#   Writes: da_pred_synapse.png / da_pred_kvasir.png / da_pred_isic.png
#           (individual prediction images — edit together with cross_task fig)
# ══════════════════════════════════════════════════════════════════════════════
elif args.mode == "da_only":
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not os.path.exists(args.cases_log):
        sys.exit(
            f"cases_used.json not found at {args.cases_log}.\n"
            f"Run --mode cross_task first to record which cases were used."
        )

    with open(args.cases_log) as f:
        log = json.load(f)
    print(f"[da_only] Loaded case log: {log}")

    for ds_key, ckpt_arg, num_cls in [
        ("synapse", args.ckpt_da_syn, 9),
        ("kvasir", args.ckpt_da_kvasir, 2),
        ("isic", args.ckpt_da_isic, 2),
    ]:
        if ckpt_arg is None:
            print(f"  Skipping {ds_key} (no --ckpt_da_{ds_key} provided)")
            continue
        if ds_key not in log:
            print(f"  Skipping {ds_key} (not in cases_log)")
            continue

        net = load_da_model(ckpt_arg, num_classes=num_cls)
        info = log[ds_key]

        if ds_key == "synapse":
            target = info.get("case_name", "")
            cases = load_synapse_cases(target_name=target)
            if not cases:
                print(f"  [warn] case '{target}' not found; skipping synapse")
                del net
                continue
            _, img_vol, lbl_vol = cases[0]
            s_idx = info.get("slice_idx", img_vol.shape[0] // 2)
            img_sl, lbl_sl = img_vol[s_idx], lbl_vol[s_idx]
            pred = predict_slice(net, img_sl.astype(np.float32))
            norm = (img_sl - img_sl.min()) / (img_sl.max() - img_sl.min() + 1e-6)
            colors = SYNAPSE_COLORS

        else:  # kvasir / isic
            ds_label = "Kvasir" if ds_key == "kvasir" else "ISIC"
            cases = load_binary_cases(ds_label, n_cases=50)
            target = info.get("case_name", "")
            match = [(n, im, lb) for n, im, lb in cases if n == target]
            if not match:
                print(f"  [warn] case '{target}' not found; using first loaded")
                match = [cases[0]]
            _, img_sl, lbl_sl = match[0]
            pred = predict_slice(net, img_sl)
            norm = (img_sl - img_sl.min()) / (img_sl.max() - img_sl.min() + 1e-6)
            colors = BINARY_COLORS

        del net
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # save input / GT / DA-pred as individual PNGs (easy to combine in any editor)
        for tag, arr, is_overlay in [
            ("input", norm, False),
            ("gt", seg_to_rgb(lbl_sl.astype(np.uint8), colors), True),
            ("da_pred", seg_to_rgb(pred, colors), True),
        ]:
            fig_s, ax_s = plt.subplots(1, 1, figsize=(3, 3))
            ax_s.axis("off")
            if is_overlay:
                ax_s.imshow(overlay(norm, arr))
            else:
                ax_s.imshow(arr, cmap="gray")
            out = os.path.join(args.out_dir, f"da_{ds_key}_{tag}.png")
            fig_s.savefig(out, dpi=args.dpi, bbox_inches="tight", pad_inches=0)
            plt.close(fig_s)
            print(f"  Saved {out}")

    print("\n[da_only] Done. Use da_*_da_pred.png files to add DA-TransUNet column.")
