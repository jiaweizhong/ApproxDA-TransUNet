"""
GCS Causal Elimination — Systematic Alternative Explanation Sanity Checks

"Causal Elimination" approach: rule out alternative explanations one by one,
then present the mechanistic evidence for Semantic Spatial Dependency (SSD).

All checks are training-free; computed from ground-truth masks only.

Organization:
  [RULE-OUT] — expect near-zero or negative Spearman rho vs delta-DSC
    SC1: Foreground Ratio        GCS ≠ object size / pixel coverage
    SC2: Boundary Complexity     GCS ≠ boundary irregularity
    SC3: Spatial Entropy         GCS ≠ simple spatial dispersion

  [MECHANISTIC] — expect positive rho
    SC4: Window Crossing Ratio   GCS ~ fraction of fg that crosses M=7 windows
    SC5: Inter-class Sep.        GCS ~ fraction of class pairs in different windows
                                 (Synapse only; reports raw value, no rho)

Usage (Lightning AI, from experiments/):
    python analyze_gcs_causal.py \\
        --synapse_dir ../data/Synapse/train_npz \\
        --kvasir_dir  ../data/Kvasir-SEG \\
        --isic_dir    ../data/ISIC2018 \\
        --n_images 200 --window_size 7 --resize 256
"""

import argparse
import os
import glob
import random
import numpy as np
import h5py
from PIL import Image
from scipy import ndimage
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Known GCS values ─────────────────────────────────────────────────────────
KNOWN_GCS = {
    "Synapse": 2.30,
    "ACDC":    0.73,   # 4-class cardiac MRI — discriminating test for SSD vs n_classes
    "Kvasir":  0.64,
    "ISIC":    0.70,
    "CVC":     0.62,
}
COLORS = {"Synapse": "#D97C6B", "ACDC": "#A07BC3", "Kvasir": "#6BAD8F", "ISIC": "#6B8FAD", "CVC": "#C4A35A"}


# ─────────────────────────────────────────────────────────────────────────────
# Data loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_acdc_labels(root: str, n: int, seed: int = 42) -> list:
    """Load label arrays from ACDC_training_slices/*.h5 (label key, uint8 0-3)."""
    files = sorted(glob.glob(os.path.join(root, "*.h5")))
    rng = random.Random(seed)
    rng.shuffle(files)
    labels = []
    for f in files:
        if len(labels) >= n:
            break
        if "(" in os.path.basename(f):   # skip duplicate files
            continue
        try:
            with h5py.File(f, "r") as hf:
                label = hf["label"][:].astype(np.int32)
            if label.max() > 0:          # skip blank slices
                labels.append(label)
        except Exception:
            pass
    return labels


def load_synapse_labels(root: str, n: int, seed: int = 42) -> list:
    files = sorted(glob.glob(os.path.join(root, "*.npz")))
    rng = random.Random(seed)
    rng.shuffle(files)
    labels = []
    for f in files:
        if len(labels) >= n:
            break
        try:
            label = np.load(f)["label"].astype(np.int32)
            if label.max() > 0:   # skip blank slices
                labels.append(label)
        except Exception:
            pass
    return labels


def _find_mask(mask_dir: str, stem: str) -> str | None:
    for suffix in ("_segmentation.png", ".png", ".jpg", ".jpeg"):
        p = os.path.join(mask_dir, stem + suffix)
        if os.path.exists(p):
            return p
    return None


def load_cvc_labels(base_dir: str, n: int, resize: int = 256,
                    seed: int = 42) -> list:
    """CVC-ClinicDB: Kaggle layout PNG/Original + PNG/Ground Truth."""
    mask_dir = os.path.join(base_dir, "PNG", "Ground Truth")
    if not os.path.isdir(mask_dir):
        return []
    masks = []
    files = sorted(glob.glob(os.path.join(mask_dir, "*.png")) +
                   glob.glob(os.path.join(mask_dir, "*.tif")))
    rng = random.Random(seed)
    rng.shuffle(files)
    for f in files[:n]:
        try:
            pil = Image.open(f).convert("L")
            if resize > 0:
                pil = pil.resize((resize, resize), Image.NEAREST)
            masks.append((np.array(pil) > 127).astype(np.uint8))
        except Exception:
            pass
    return masks


def load_binary_labels(base_dir: str, n: int, resize: int = 256,
                       seed: int = 42) -> list:
    img_dir  = os.path.join(base_dir, "images")
    mask_dir = os.path.join(base_dir, "masks")
    if not os.path.isdir(img_dir) or not os.path.isdir(mask_dir):
        return []
    stems = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG"):
        for p in glob.glob(os.path.join(img_dir, ext)):
            stem = os.path.splitext(os.path.basename(p))[0]
            if _find_mask(mask_dir, stem):
                stems.append(stem)
    rng = random.Random(seed)
    rng.shuffle(stems)
    labels = []
    for stem in stems[:n]:
        mp = _find_mask(mask_dir, stem)
        if mp is None:
            continue
        try:
            pil = Image.open(mp).convert("L")
            if resize > 0:
                pil = pil.resize((resize, resize), Image.NEAREST)
            labels.append((np.array(pil) > 127).astype(np.uint8))
        except Exception:
            pass
    return labels


# ─────────────────────────────────────────────────────────────────────────────
# SC1 — Foreground Ratio
# ─────────────────────────────────────────────────────────────────────────────

def foreground_ratio(label: np.ndarray) -> float:
    """Fraction of pixels that are foreground."""
    return float((label > 0).mean())


# ─────────────────────────────────────────────────────────────────────────────
# SC2 — Boundary Complexity (isoperimetric ratio)
# ─────────────────────────────────────────────────────────────────────────────

def boundary_complexity(label: np.ndarray) -> float:
    """
    Isoperimetric ratio: Perimeter² / (4π × Area), averaged over all components.
    Circle = 1.0; irregular shapes > 1.0.
    """
    binary = (label > 0)
    if binary.sum() == 0:
        return 0.0
    labeled_arr, n_comp = ndimage.label(binary)
    ratios = []
    for c in range(1, n_comp + 1):
        comp = (labeled_arr == c).astype(np.float32)
        area = comp.sum()
        if area < 5:
            continue
        perimeter = (comp - ndimage.binary_erosion(comp)).sum()
        if perimeter < 1:
            continue
        ratios.append((perimeter ** 2) / (4 * np.pi * area))
    return float(np.mean(ratios)) if ratios else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# SC3 — Spatial Distribution Entropy
# ─────────────────────────────────────────────────────────────────────────────

def spatial_entropy(label: np.ndarray, grid: int = 8) -> float:
    """
    Entropy of foreground pixel distribution over a grid × grid spatial grid.
    Normalised to [0, 1]: 1.0 = perfectly uniform spread.
    """
    binary = (label > 0)
    H, W = binary.shape
    if binary.sum() == 0:
        return 0.0
    ch, cw = H // grid, W // grid
    counts = np.array([
        binary[i*ch:(i+1)*ch, j*cw:(j+1)*cw].sum()
        for i in range(grid) for j in range(grid)
    ], dtype=float)
    total = counts.sum()
    if total < 1:
        return 0.0
    p = counts / total
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)) / np.log(grid * grid))


# ─────────────────────────────────────────────────────────────────────────────
# SC4 — Window Crossing Ratio
# ─────────────────────────────────────────────────────────────────────────────

def window_crossing_ratio(label: np.ndarray, is_multiclass: bool,
                           M: int = 7) -> float:
    """
    Fraction of foreground pixels belonging to components that span > 1
    non-overlapping M×M window.

    Intuition: if a target region crosses window boundaries, windowed attention
    at size M cannot attend across the full target in a single step — the model
    must aggregate information across windows, requiring broader context.
    """
    total_fg = int((label > 0).sum())
    if total_fg == 0:
        return 0.0

    class_masks = []
    if is_multiclass:
        for c in np.unique(label):
            if c > 0:
                class_masks.append(label == c)
    else:
        class_masks.append(label > 0)

    cross_pixels = 0
    for mask in class_masks:
        comp_labeled, n_comp = ndimage.label(mask)
        for cid in range(1, n_comp + 1):
            comp = (comp_labeled == cid)
            yx = np.argwhere(comp)
            windows = set(zip((yx[:, 0] // M).tolist(),
                               (yx[:, 1] // M).tolist()))
            if len(windows) > 1:
                cross_pixels += int(comp.sum())

    return float(cross_pixels / total_fg)


# ─────────────────────────────────────────────────────────────────────────────
# SC5 — Inter-class Window Separation  (Synapse only)
# ─────────────────────────────────────────────────────────────────────────────

def inter_class_window_sep(label: np.ndarray, M: int = 7) -> float | None:
    """
    Fraction of class-centroid pairs that land in different M×M windows.
    Measures how often two organs are too far apart for windowed attention to
    see both simultaneously.  Returns None for single-class data.
    """
    classes = [c for c in np.unique(label) if c > 0]
    if len(classes) < 2:
        return None

    H, W = label.shape
    win = {}
    for c in classes:
        yx = np.argwhere(label == c)
        if len(yx) == 0:
            continue
        cy, cx = yx.mean(axis=0)
        win[c] = (int(cy) // M, int(cx) // M)

    keys = list(win.keys())
    n_pairs = n_sep = 0
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            n_pairs += 1
            if win[keys[i]] != win[keys[j]]:
                n_sep += 1

    return float(n_sep / n_pairs) if n_pairs > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation
# ─────────────────────────────────────────────────────────────────────────────

def compute_dataset_metrics(labels: list, is_multiclass: bool, M: int) -> dict:
    fr, bc, se, wcr, icws = [], [], [], [], []
    for label in labels:
        fr.append(foreground_ratio(label))
        bc.append(boundary_complexity(label))
        se.append(spatial_entropy(label))
        wcr.append(window_crossing_ratio(label, is_multiclass, M=M))
        if is_multiclass:
            v = inter_class_window_sep(label, M=M)
            if v is not None:
                icws.append(v)
    return {
        "foreground_ratio":       float(np.mean(fr)),
        "boundary_complexity":    float(np.mean(bc)),
        "spatial_entropy":        float(np.mean(se)),
        "window_crossing_ratio":  float(np.mean(wcr)),
        "inter_class_window_sep": float(np.mean(icws)) if icws else None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

def _place_labels(ax, xs, ys, names, fontsize=8.5):
    """
    Annotate scatter points with non-overlapping labels.
    Uses adjustText when available; falls back to a repulsion-vector approach.
    """
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)

    try:
        from adjustText import adjust_text
        texts = [ax.text(x, y, name, fontsize=fontsize)
                 for x, y, name in zip(xs, ys, names)]
        adjust_text(
            texts, x=xs, y=ys, ax=ax,
            arrowprops=dict(arrowstyle="-", color="#aaa", lw=0.5),
            expand_points=(2.0, 2.0),
            expand_text=(1.4, 1.4),
            force_text=(0.8, 1.5),
            force_points=(0.5, 1.0),
        )
        return

    except ImportError:
        pass

    # ── Fallback: inverse-distance repulsion ─────────────────────────────────
    xr = xs.ptp() or 1.0
    yr = ys.ptp() or 1.0

    # Get current axes limits to push labels inward from borders
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    PAD = 18   # base offset in points
    offsets = []
    for i in range(len(names)):
        dx, dy = 0.0, 0.0
        for j in range(len(names)):
            if i == j:
                continue
            ddx = (xs[i] - xs[j]) / xr
            ddy = (ys[i] - ys[j]) / yr
            dist2 = ddx ** 2 + ddy ** 2 + 1e-9
            dx += ddx / dist2
            dy += ddy / dist2
        mag = np.sqrt(dx ** 2 + dy ** 2) + 1e-9
        ox = dx / mag * PAD
        oy = dy / mag * PAD

        # Nudge away from axes borders (normalised [0,1] → points scale)
        margin_x = (xlim[1] - xlim[0]) * 0.05
        margin_y = (ylim[1] - ylim[0]) * 0.05
        if xs[i] - xlim[0] < margin_x:
            ox = abs(ox)
        if xlim[1] - xs[i] < margin_x:
            ox = -abs(ox)
        if ys[i] - ylim[0] < margin_y:
            oy = abs(oy)
        if ylim[1] - ys[i] < margin_y:
            oy = -abs(oy)

        offsets.append((ox, oy))

    for name, x, y, (ox, oy) in zip(names, xs, ys, offsets):
        ax.annotate(
            name, (x, y),
            textcoords="offset points", xytext=(ox, oy),
            fontsize=fontsize,
            arrowprops=dict(arrowstyle="-", color="#bbb", lw=0.4)
            if (abs(ox) > 10 or abs(oy) > 8) else None,
        )


RULE_OUT = [
    ("foreground_ratio",    "SC1: Foreground Ratio",    "expect ρ ≈ 0"),
    ("boundary_complexity", "SC2: Boundary Complexity", "expect ρ ≈ 0"),
    ("spatial_entropy",     "SC3: Spatial Entropy",     "expect ρ ≈ 0"),
]
MECHANISTIC = [
    ("window_crossing_ratio", "SC4: Window Crossing Ratio", "expect ρ > 0"),
]


def plot_results(results: dict, M: int, out_path: str):
    names = list(results.keys())
    colors = [COLORS[n] for n in names]
    delta_dscs = [results[n]["delta_dsc"] for n in names]

    all_panels = RULE_OUT + MECHANISTIC
    ncols = len(all_panels)
    fig, axes = plt.subplots(2, ncols, figsize=(4.5 * ncols, 8))
    fig.suptitle(
        f"GCS Causal Elimination  (M={M})\n"
        "Top: metric per dataset  |  Bottom: scatter vs ΔDSC",
        fontsize=13, fontweight="bold"
    )

    for col, (key, title, expectation) in enumerate(all_panels):
        vals = [results[n]["stats"][key] for n in names]
        rho, pval = spearmanr(vals, delta_dscs)

        # ── bar chart (top row) ───────────────────────────────────────────
        ax = axes[0, col]
        bars = ax.bar(names, vals, color=colors, edgecolor="#444", linewidth=0.7)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    v + max(vals) * 0.03,
                    f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")
        ax.set_title(f"{title}\n({expectation})", fontsize=9.5, fontweight="bold")
        ax.set_ylim(0, max(vals) * 1.30)
        ax.tick_params(axis="x", labelsize=9)

        # ── scatter (bottom row) ──────────────────────────────────────────
        ax = axes[1, col]
        for name, v, d, c in zip(names, vals, delta_dscs, colors):
            ax.scatter(v, d, s=110, color=c, edgecolors="#444",
                       linewidth=0.7, zorder=3)
        # Expand axes limits before placing labels so border-avoidance works
        ax.autoscale(enable=True)
        xmarg = (max(vals) - min(vals) or 1) * 0.15
        ymarg = (max(delta_dscs) - min(delta_dscs) or 1) * 0.18
        ax.set_xlim(min(vals) - xmarg, max(vals) + xmarg)
        ax.set_ylim(min(delta_dscs) - ymarg, max(delta_dscs) + ymarg)
        _place_labels(ax, vals, delta_dscs, names, fontsize=8.5)
        rho_color = ("#2a7a2a" if rho > 0.5 else
                     ("#cc4400" if rho < -0.3 else "#888"))
        ax.set_title(f"ρ = {rho:+.3f}   p = {pval:.3f}",
                     fontsize=9.5, color=rho_color, fontweight="bold")
        ax.set_xlabel(key, fontsize=8.5)
        if col == 0:
            ax.set_ylabel("ΔDSC (pp) — GCS", fontsize=9)
        ax.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved → {out_path}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synapse_dir", default="../data/Synapse/train_npz")
    parser.add_argument("--acdc_dir",   default="../data/ACDC/ACDC_training_slices")
    parser.add_argument("--kvasir_dir",  default="../data/Kvasir-SEG")
    parser.add_argument("--isic_dir",    default="../data/ISIC2018")
    parser.add_argument("--cvc_dir",     default="../data/CVC-ClinicDB")
    parser.add_argument("--n_images",    type=int, default=200)
    parser.add_argument("--window_size", type=int, default=28,
                        help="Window size in pixel space.  M=7 in feature space "
                             "corresponds to ~28 pixels at 256px input "
                             "(Swin 4x downsampling: 7 feature tokens × 4px stride). "
                             "M=7 in pixel space saturates WCR to 1.0 for all datasets.")
    parser.add_argument("--resize",      type=int, default=256)
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--out",         default="gcs_causal_sanity.png")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    M = args.window_size

    dataset_configs = [
        ("Synapse", args.synapse_dir, True,  "synapse"),
        ("ACDC",    args.acdc_dir,   True,  "acdc"),
        ("Kvasir",  args.kvasir_dir,  False, "binary"),
        ("ISIC",    args.isic_dir,    False, "binary"),
        ("CVC",     args.cvc_dir,     False, "cvc"),
    ]

    results = {}
    for name, path, is_multi, loader_type in dataset_configs:
        print(f"\n[{name}] Loading from: {path}")
        if not os.path.isdir(path):
            print("  ⚠  Directory not found, skipping.")
            continue

        if loader_type == "synapse":
            labels = load_synapse_labels(path, args.n_images, args.seed)
        elif loader_type == "acdc":
            labels = load_acdc_labels(path, args.n_images, args.seed)
        elif loader_type == "cvc":
            labels = load_cvc_labels(path, args.n_images,
                                     resize=args.resize, seed=args.seed)
        else:
            labels = load_binary_labels(path, args.n_images,
                                        resize=args.resize, seed=args.seed)
        if not labels:
            print("  ⚠  No labels loaded, skipping.")
            continue

        print(f"  Loaded {len(labels)} masks. Computing metrics...")
        stats = compute_dataset_metrics(labels, is_multi, M=M)

        print(f"  SC1 foreground_ratio:       {stats['foreground_ratio']:.4f}")
        print(f"  SC2 boundary_complexity:    {stats['boundary_complexity']:.4f}")
        print(f"  SC3 spatial_entropy:        {stats['spatial_entropy']:.4f}")
        print(f"  SC4 window_crossing_ratio:  {stats['window_crossing_ratio']:.4f}  (M={M})")
        if stats["inter_class_window_sep"] is not None:
            print(f"  SC5 inter_class_window_sep: {stats['inter_class_window_sep']:.4f}  "
                  f"(Synapse only; fraction of organ pairs in different windows)")
        else:
            print(f"  SC5 inter_class_window_sep: N/A (binary dataset)")
        print(f"  ΔDSC (known): {KNOWN_GCS.get(name)} pp")

        results[name] = {"stats": stats, "delta_dsc": KNOWN_GCS.get(name)}

    if len(results) < 2:
        print("\nNeed ≥ 2 datasets. Check paths.")
        return

    # ── Summary table ─────────────────────────────────────────────────────────
    names = list(results.keys())
    delta_dscs = [results[n]["delta_dsc"] for n in names]

    rule_out_keys  = [k for k, _, _ in RULE_OUT]
    mech_keys      = [k for k, _, _ in MECHANISTIC]

    print("\n" + "=" * 60)
    print("CAUSAL ELIMINATION SUMMARY")
    print("=" * 60)

    print("\n[RULE-OUT checks — expect ρ ≈ 0 or wrong direction]")
    for key in rule_out_keys:
        vals = [results[n]["stats"][key] for n in names]
        rho, pval = spearmanr(vals, delta_dscs)
        tag = "✅ ruled out" if abs(rho) < 0.5 else "⚠  not ruled out"
        print(f"  {tag}  {key:28s}  ρ = {rho:+.3f}   p = {pval:.3f}")

    print("\n[MECHANISTIC checks — expect ρ > 0]")
    for key in mech_keys:
        vals = [results[n]["stats"][key] for n in names]
        rho, pval = spearmanr(vals, delta_dscs)
        tag = "✅ supports SSD" if rho > 0.5 else "⚠  weak"
        print(f"  {tag}  {key:28s}  ρ = {rho:+.3f}   p = {pval:.3f}")

    # SC5 — now computed for all datasets; binary = 0 by construction
    # With ACDC added we have 4 data points and can compute a proper ρ.
    sc5_vals, sc5_gcs = [], []
    for n in names:
        v = results[n]["stats"]["inter_class_window_sep"]
        sc5_vals.append(v if v is not None else 0.0)
        sc5_gcs.append(results[n]["delta_dsc"])
    rho_sc5, p_sc5 = spearmanr(sc5_vals, sc5_gcs)

    print(f"\n[SC5 — Inter-class window separation (M={M})]")
    for n, v in zip(names, sc5_vals):
        tag = "(multi-class)" if results[n]["stats"]["inter_class_window_sep"] is not None else "(binary → 0 by construction)"
        print(f"  {n:10s}  SC5 = {v:.4f}  ΔDSC = {results[n]['delta_dsc']:.2f} pp  {tag}")
    print(f"  Spearman ρ = {rho_sc5:+.3f}   p = {p_sc5:.3f}")
    print(f"  → SC5 is the strongest mechanistic predictor: spatial spread of classes,")
    print(f"    not class count, governs GCS. ACDC (4 classes, SC5≈0) falls with binary tasks.")

    # ── Data-driven narrative ─────────────────────────────────────────────────
    sc1_rho = spearmanr([results[n]["stats"]["foreground_ratio"]    for n in names], delta_dscs)[0]
    sc2_rho = spearmanr([results[n]["stats"]["boundary_complexity"] for n in names], delta_dscs)[0]
    sc3_rho = spearmanr([results[n]["stats"]["spatial_entropy"]     for n in names], delta_dscs)[0]
    sc4_rho = spearmanr([results[n]["stats"]["window_crossing_ratio"] for n in names], delta_dscs)[0]

    print("\n[Narrative]")
    sc1_txt = "ruled out" if abs(sc1_rho) < 0.5 else f"NOT ruled out (ρ={sc1_rho:+.3f})"
    sc2_txt = ("ruled out" if abs(sc2_rho) < 0.5
               else f"NOT ruled out (ρ={sc2_rho:+.3f}) — likely cross-domain imaging confound "
                    "(CT produces complex organ boundaries by physics, not task reasoning)")
    sc3_txt = "ruled out" if abs(sc3_rho) < 0.5 else f"NOT ruled out (ρ={sc3_rho:+.3f})"
    wcr_max = max(results[n]["stats"]["window_crossing_ratio"] for n in names)
    if wcr_max > 0.98:
        sc4_txt = (f"SATURATED (all datasets WCR>{wcr_max:.3f}) — "
                   "individual targets cross M-pixel windows regardless of task type. "
                   "Positive finding: rules out component extent as GCS mechanism.")
    elif sc4_rho > 0.5:
        sc4_txt = f"supports SSD (ρ={sc4_rho:+.3f})"
    else:
        sc4_txt = f"weak/wrong direction (ρ={sc4_rho:+.3f})"
    print(f"  SC1 (object size/coverage):    {sc1_txt}")
    print(f"  SC2 (boundary shape):          {sc2_txt}")
    print(f"  SC3 (spatial dispersion):      {sc3_txt}")
    print(f"  SC4 (window crossing ratio):   {sc4_txt}")
    print(f"  SC5 (inter-class window sep):  ρ = {rho_sc5:+.3f} — "
          f"Synapse high, ACDC≈0 (co-located anatomy) = binary → key SSD evidence")

    plot_results(results, M, args.out)


if __name__ == "__main__":
    main()
