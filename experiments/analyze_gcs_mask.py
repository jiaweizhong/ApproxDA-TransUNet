"""
Mask-based GCS Sanity Check

Replaces Fourier Energy with mask-derived spatial statistics as GCS proxies.
Metrics computed per image, then averaged across dataset:

  n_classes        — unique foreground semantic classes per image
                     (Synapse: up to 8 organs; Kvasir/ISIC: 1)
  n_components     — connected foreground components per image
                     (Synapse: many organs; Kvasir/ISIC: typically 1)
  centroid_spread  — normalised std of per-class/per-component centroids
                     (how spread across the image the foreground regions are)
  bbox_coverage    — area of foreground bounding box / image area
                     (organs span most of the abdomen; polyps are localised)

Expected direction: all four metrics higher for Synapse (high GCS) than
Kvasir / ISIC (low GCS).  Spearman rho vs known delta-DSC is computed for each.

Usage (Lightning AI, from experiments/):
    python analyze_gcs_mask.py \
        --synapse_dir  ../data/Synapse/train_npz \
        --kvasir_dir   ../data/Kvasir-SEG \
        --isic_dir     ../data/ISIC2018 \
        --n_images 200
"""

import argparse
import os
import glob
import random
import numpy as np

try:
    import h5py
except ImportError:
    h5py = None
from PIL import Image
from scipy import ndimage
from scipy.stats import spearmanr
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── known GCS values from conference paper ────────────────────────────────────
KNOWN_GCS = {
    "Synapse": 2.30,
    "Kvasir": 0.64,
    "ISIC": 0.50,
}

COLORS = {"Synapse": "#D97C6B", "Kvasir": "#6BAD8F", "ISIC": "#6B8FAD"}


# ─────────────────────────────────────────────────────────────────────────────
# Mask statistics
# ─────────────────────────────────────────────────────────────────────────────


def mask_stats(label: np.ndarray, is_multiclass: bool) -> dict:
    """
    Compute spatial statistics from a single segmentation mask.

    label: 2D int array.
      is_multiclass=True  → values 0 (bg) to N (organs); Synapse format
      is_multiclass=False → binary (0/1); Kvasir / ISIC format
    Returns dict with n_classes, n_components, centroid_spread, bbox_coverage.
    """
    H, W = label.shape
    diag = np.sqrt(H**2 + W**2)

    if is_multiclass:
        classes = np.unique(label)
        classes = classes[classes > 0]
        n_classes = int(len(classes))

        centroids = []
        for c in classes:
            yx = np.argwhere(label == c)
            cy, cx = yx.mean(axis=0)
            centroids.append([cy / H, cx / W])

        if len(centroids) >= 2:
            centroids = np.array(centroids)
            centroid_spread = float(centroids.std(axis=0).mean())
        elif len(centroids) == 1:
            centroid_spread = 0.0
        else:
            return None  # no foreground at all

        binary = label > 0

    else:  # binary
        n_classes = 1
        binary = label > 0
        if binary.sum() == 0:
            return None

    # connected components (shared for both modes)
    labeled_arr, n_comp = ndimage.label(binary)
    n_components = int(n_comp)

    if not is_multiclass:
        if n_comp >= 2:
            comp_centroids = []
            for c in range(1, n_comp + 1):
                yx = np.argwhere(labeled_arr == c)
                comp_centroids.append(yx.mean(axis=0) / np.array([H, W]))
            comp_centroids = np.array(comp_centroids)
            centroid_spread = float(comp_centroids.std(axis=0).mean())
        else:
            # single component: spread of foreground pixels
            yx = np.argwhere(binary)
            centroid_spread = float((yx / np.array([H, W])).std(axis=0).mean())

    # bounding box coverage
    yx = np.argwhere(binary)
    if len(yx) == 0:
        bbox_coverage = 0.0
    else:
        y0, x0 = yx.min(axis=0)
        y1, x1 = yx.max(axis=0)
        bbox_h = max(y1 - y0 + 1, 1)
        bbox_w = max(x1 - x0 + 1, 1)
        bbox_coverage = float((bbox_h * bbox_w) / (H * W))

    return {
        "n_classes": n_classes,
        "n_components": n_components,
        "centroid_spread": centroid_spread,
        "bbox_coverage": bbox_coverage,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Dataset loaders
# ─────────────────────────────────────────────────────────────────────────────


def load_synapse(root: str, n: int) -> list[np.ndarray]:
    files = sorted(glob.glob(os.path.join(root, "*.npz")))
    if not files:
        files = sorted(glob.glob(os.path.join(root, "*.h5")))
    random.shuffle(files)
    labels = []
    for f in files:
        if len(labels) >= n:
            break
        try:
            if f.endswith(".npz"):
                label = np.load(f)["label"]
            else:
                with h5py.File(f, "r") as hf:
                    label = hf["label"][()]
            labels.append(label.astype(np.int32))
        except Exception as e:
            print(f"  Warning: {f}: {e}")
    return labels


def _find_mask(mask_dir: str, stem: str) -> str | None:
    """Find mask file: try stem.png, stem.jpg, stem_segmentation.png."""
    for suffix in ("_segmentation.png", ".png", ".jpg", ".jpeg"):
        p = os.path.join(mask_dir, stem + suffix)
        if os.path.exists(p):
            return p
    return None


def load_binary(base_dir: str, n: int, resize: int = 0) -> list[np.ndarray]:
    img_dir = os.path.join(base_dir, "images")
    mask_dir = os.path.join(base_dir, "masks")
    if not os.path.isdir(img_dir) or not os.path.isdir(mask_dir):
        return []

    stems = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG"):
        for p in glob.glob(os.path.join(img_dir, ext)):
            stem = os.path.splitext(os.path.basename(p))[0]
            if _find_mask(mask_dir, stem):
                stems.append(stem)

    random.shuffle(stems)
    labels = []
    for stem in stems[:n]:
        mp = _find_mask(mask_dir, stem)
        if mp is None:
            continue
        try:
            pil = Image.open(mp).convert("L")
            if resize > 0:
                pil = pil.resize((resize, resize), Image.NEAREST)
            mask = (np.array(pil) > 127).astype(np.uint8)
            labels.append(mask)
        except Exception as e:
            print(f"  Warning: {mp}: {e}")
    return labels


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

PRECOMPUTED_MASK = {
    "Synapse": {
        "stats": {
            "n_classes": 3.991,
            "n_components": 3.604,
            "centroid_spread": 0.089,
            "bbox_coverage": 0.181,
        },
        "delta_dsc": 2.30,
    },
    "Kvasir": {
        "stats": {
            "n_classes": 1.000,
            "n_components": 1.040,
            "centroid_spread": 0.110,
            "bbox_coverage": 0.226,
        },
        "delta_dsc": 0.64,
    },
    "ISIC": {
        "stats": {
            "n_classes": 1.000,
            "n_components": 3.810,
            "centroid_spread": 0.124,
            "bbox_coverage": 0.295,
        },
        "delta_dsc": 0.50,
    },
}


MASK_OFFSETS = {
    "n_classes": {
        "Synapse": (-10, 6, "right", "bottom"),
        "Kvasir": (8, 6, "left", "bottom"),
        "ISIC": (-8, 6, "right", "bottom"),
    },
    "n_components": {
        "Synapse": (-10, 6, "right", "bottom"),
        "Kvasir": (8, 6, "left", "bottom"),
        "ISIC": (8, 6, "left", "bottom"),
    },
    "centroid_spread": {
        "Synapse": (8, 6, "left", "bottom"),
        "Kvasir": (8, 6, "left", "bottom"),
        "ISIC": (-8, 6, "right", "bottom"),
    },
    "bbox_coverage": {
        "Synapse": (8, 6, "left", "bottom"),
        "Kvasir": (8, 6, "left", "bottom"),
        "ISIC": (-10, 6, "right", "bottom"),
    },
}


def plot_results(results: dict, out_path: str):
    metrics = ["n_classes", "n_components", "centroid_spread", "bbox_coverage"]
    metric_labels = {
        "n_classes": "Mean class count",
        "n_components": "Mean component count",
        "centroid_spread": "Centroid spread (normalised)",
        "bbox_coverage": "Bbox coverage (fraction)",
    }

    names = list(results.keys())
    delta_dscs = [results[n]["delta_dsc"] for n in names]
    colors = [COLORS[n] for n in names]

    fig = plt.figure(figsize=(13.5, 5.6))
    gs = gridspec.GridSpec(2, 4, hspace=0.36, wspace=0.22)

    for col, metric in enumerate(metrics):
        # top row: bar chart (sleek, thinner bars)
        ax_bar = fig.add_subplot(gs[0, col])
        vals = [results[n]["stats"][metric] for n in names]
        bars = ax_bar.bar(
            names, vals, width=0.46, color=colors, edgecolor="#333", linewidth=0.8
        )
        for bar, v in zip(bars, vals):
            ax_bar.text(
                bar.get_x() + bar.get_width() / 2,
                v + max(vals) * 0.03,
                f"{v:.3f}",
                ha="center",
                fontsize=8.5,
                fontweight="bold",
            )
        ax_bar.set_title(metric_labels[metric], fontsize=10, fontweight="bold")
        ax_bar.tick_params(axis="x", labelsize=9)
        ax_bar.tick_params(axis="y", labelsize=8.5)
        ax_bar.set_ylim(0, max(vals) * 1.25)
        ax_bar.grid(True, axis="y", alpha=0.25)

        # bottom row: scatter vs delta_dsc
        ax_sc = fig.add_subplot(gs[1, col])
        rho, pval = spearmanr(vals, delta_dscs)
        for name, v, d, c in zip(names, vals, delta_dscs, colors):
            ax_sc.scatter(
                v, d, s=110, color=c, edgecolors="#333", linewidth=0.8, zorder=3
            )
            if metric in MASK_OFFSETS and name in MASK_OFFSETS[metric]:
                ox, oy, ha, va = MASK_OFFSETS[metric][name]
            else:
                ox, oy, ha, va = 6, 2, "left", "bottom"
            ax_sc.annotate(
                name,
                (v, d),
                textcoords="offset points",
                xytext=(ox, oy),
                ha=ha,
                va=va,
                fontsize=8.5,
                fontweight="bold",
                arrowprops=(
                    dict(arrowstyle="-", color="#aaa", lw=0.4)
                    if (abs(ox) > 10 or abs(oy) > 8)
                    else None
                ),
            )

        x_min, x_max = min(vals), max(vals)
        x_span = x_max - x_min or 1.0
        ax_sc.set_xlim(x_min - x_span * 0.25, x_max + x_span * 0.25)
        ax_sc.set_ylim(0.10, 2.75)

        ax_sc.set_xlabel(metric_labels[metric], fontsize=9)
        if col == 0:
            ax_sc.set_ylabel("ΔDSC (pp) — GCS", fontsize=9.5, fontweight="bold")
        rho_color = "#2a7a2a" if rho > 0.5 else ("#cc4400" if rho < -0.3 else "#666")
        ax_sc.set_title(
            f"ρ = {rho:+.3f}   p = {pval:.3f}",
            fontsize=9.5,
            color=rho_color,
            fontweight="bold",
        )
        ax_sc.tick_params(labelsize=8.5)
        ax_sc.grid(True, alpha=0.25)

    fig.suptitle(
        "Mask-based GCS Proxies vs Known ΔDSC", fontsize=12.5, fontweight="bold"
    )

    repo_root = Path(__file__).resolve().parent.parent
    out_dir = repo_root / "paper-journal" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "gcs_mask_sanity.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(out_dir / "gcs_mask_sanity.png", bbox_inches="tight", dpi=200)
    print(f"Saved to {out_dir}/gcs_mask_sanity.{{pdf,png}}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synapse_dir", default="../data/Synapse/train_npz")
    parser.add_argument("--kvasir_dir", default="../data/Kvasir-SEG")
    parser.add_argument("--isic_dir", default="../data/ISIC2018")
    parser.add_argument("--n_images", type=int, default=200)
    parser.add_argument(
        "--resize",
        type=int,
        default=256,
        help="Resize masks to NxN before analysis (0=no resize). "
        "Speeds up loading of high-res datasets like ISIC.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="gcs_mask_sanity.png")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    dataset_configs = [
        ("Synapse", args.synapse_dir, True),
        ("Kvasir", args.kvasir_dir, False),
        ("ISIC", args.isic_dir, False),
    ]

    results = {}
    for name, path, is_multi in dataset_configs:
        print(f"\n[{name}] Loading masks from: {path}")
        if not os.path.isdir(path):
            print("  [WARN] Directory not found, skipping.")
            continue

        labels = (
            load_synapse(path, args.n_images)
            if is_multi
            else load_binary(path, args.n_images, resize=args.resize)
        )

        if not labels:
            print("  [WARN] No labels loaded, skipping.")
            continue

        print(f"  Loaded {len(labels)} masks. Computing stats...")
        stats = compute_dataset_stats(labels, is_multi)
        delta_dsc = KNOWN_GCS.get(name)

        print(f"  n_classes:       {stats['n_classes']:.3f}")
        print(f"  n_components:    {stats['n_components']:.3f}")
        print(f"  centroid_spread: {stats['centroid_spread']:.4f}")
        print(f"  bbox_coverage:   {stats['bbox_coverage']:.4f}")
        print(f"  ΔDSC (known):    {delta_dsc} pp")

        results[name] = {"stats": stats, "delta_dsc": delta_dsc}

    if len(results) < 2:
        print("\nUsing precomputed mask analysis data.")
        results = PRECOMPUTED_MASK

    plot_results(results, args.out)


if __name__ == "__main__":
    main()
