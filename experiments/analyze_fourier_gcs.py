"""
Fourier Energy vs GCS Sanity Check

Computes High-Frequency Ratio (HFR) for each dataset as a proxy for
Global Context Sensitivity (GCS). If HFR correlates negatively with ΔDSC,
the predictive GCS hypothesis is supported.

Usage (Lightning AI, from experiments/):
    python analyze_fourier_gcs.py \
        --synapse_dir  ../data/Synapse/train_npz \
        --kvasir_dir   ../data/Kvasir/images \
        --isic_dir     ../data/ISIC2018/images \
        --n_images 200

Output:
    Prints per-dataset HFR, then Spearman ρ vs known ΔDSC.
    Saves fourier_gcs_sanity.png (spectrum + scatter plot).
"""

import argparse
import os
import glob
import random
import numpy as np
import h5py
from PIL import Image
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

# ── known GCS values from conference paper ────────────────────────────────────
# ΔDSC = max(DSC) − min(DSC) across M ∈ {7,28,56,112}, gate=pam, r=32
KNOWN_GCS = {
    "Synapse": 2.30,  # M=7→28→56→112: −1.16, +1.14, −0.90, −0.36 vs DA-TransUNet
    "Kvasir": 0.64,
    "ISIC": 0.70,  # approximate: only M=7 tested; gain similar to Kvasir
}

THRESHOLD_RATIO = 0.15  # frequencies above 15% of Nyquist counted as "high"


def high_freq_ratio(
    image_gray: np.ndarray, threshold_ratio: float = THRESHOLD_RATIO
) -> float:
    """
    Compute fraction of 2D FFT power in frequencies above threshold_ratio × Nyquist.

    image_gray: 2D float array, any size.
    Returns scalar in [0, 1].
    """
    H, W = image_gray.shape
    fft = np.fft.fft2(image_gray)
    fft_shift = np.fft.fftshift(fft)
    power = np.abs(fft_shift) ** 2

    # Build radius map (normalised: 0 = DC, 1 = corner)
    cy, cx = H / 2, W / 2
    y, x = np.ogrid[:H, :W]
    radius = np.sqrt(((y - cy) / cy) ** 2 + ((x - cx) / cx) ** 2)

    high_mask = radius > threshold_ratio
    hfr = power[high_mask].sum() / (power.sum() + 1e-12)
    return float(hfr)


def load_synapse_images(root: str, n: int) -> list[np.ndarray]:
    """Load grayscale slices from Synapse .npz or .h5 files."""
    files = sorted(glob.glob(os.path.join(root, "*.npz")))
    if not files:
        files = sorted(glob.glob(os.path.join(root, "*.h5")))
    random.shuffle(files)
    imgs = []
    for f in files:
        if len(imgs) >= n:
            break
        try:
            if f.endswith(".npz"):
                data = np.load(f)
                img = data["image"]  # (H, W) float32, already grayscale
            else:
                with h5py.File(f, "r") as hf:
                    img = hf["image"][()]
                    if img.ndim == 3:  # (C, H, W) or (H, W, C)
                        img = img[0] if img.shape[0] < img.shape[-1] else img[..., 0]
            imgs.append(img.astype(np.float32))
        except Exception as e:
            print(f"  Warning: could not load {f}: {e}")
    return imgs


def load_rgb_images(root: str, n: int) -> list[np.ndarray]:
    """Load PNG/JPG images as grayscale arrays."""
    exts = ["*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG"]
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(root, ext)))
    random.shuffle(files)
    imgs = []
    for f in files[:n]:
        try:
            img = np.array(Image.open(f).convert("L"), dtype=np.float32) / 255.0
            imgs.append(img)
        except Exception as e:
            print(f"  Warning: could not load {f}: {e}")
    return imgs


def compute_dataset_hfr(
    images: list[np.ndarray], threshold: float = THRESHOLD_RATIO
) -> tuple[float, float]:
    """Return (mean HFR, std HFR) across all images."""
    hfrs = [high_freq_ratio(img, threshold) for img in images]
    return float(np.mean(hfrs)), float(np.std(hfrs))


def plot_results(results: dict, out_path: str):
    """
    results: {dataset_name: {"hfr_mean": ..., "hfr_std": ..., "delta_dsc": ...}}
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # ── left: HFR bar chart ────────────────────────────────────────────────────
    ax = axes[0]
    names = list(results.keys())
    means = [results[n]["hfr_mean"] for n in names]
    stds = [results[n]["hfr_std"] for n in names]
    colors = ["#D97C6B", "#6BAD8F", "#6B8FAD"]
    bars = ax.bar(
        names,
        means,
        yerr=stds,
        capsize=5,
        color=colors,
        edgecolor="#444",
        linewidth=0.8,
    )
    ax.set_ylabel("High-Frequency Ratio (HFR)", fontsize=12)
    ax.set_title("Spatial Frequency Profile per Dataset", fontsize=12)
    ax.set_ylim(0, 1)
    for bar, m in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            m + 0.02,
            f"{m:.3f}",
            ha="center",
            fontsize=10,
            fontweight="bold",
        )

    # ── right: HFR vs ΔDSC scatter ────────────────────────────────────────────
    ax = axes[1]
    hfrs = [results[n]["hfr_mean"] for n in names]
    deltas = [results[n]["delta_dsc"] for n in names]
    rho, pval = spearmanr(hfrs, deltas)

    for hfr, delta, name, color in zip(hfrs, deltas, names, colors):
        ax.scatter(
            hfr,
            delta,
            s=120,
            color=color,
            edgecolors="#444",
            linewidth=0.8,
            zorder=3,
            label=name,
        )
        ax.annotate(
            name, (hfr, delta), textcoords="offset points", xytext=(6, 4), fontsize=10
        )

    ax.set_xlabel("High-Frequency Ratio (HFR)", fontsize=12)
    ax.set_ylabel("ΔDSC (pp) — GCS proxy", fontsize=12)
    ax.set_title(
        f"HFR vs GCS  |  Spearman ρ = {rho:.3f}  (p = {pval:.3f})", fontsize=12
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved → {out_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synapse_dir", default="../data/Synapse/train_npz")
    parser.add_argument("--kvasir_dir", default="../data/Kvasir/images")
    parser.add_argument("--isic_dir", default="../data/ISIC2018/images")
    parser.add_argument(
        "--n_images",
        type=int,
        default=200,
        help="Images to sample per dataset (for speed)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=THRESHOLD_RATIO,
        help="HFR threshold: fraction of Nyquist",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="fourier_gcs_sanity.png")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    datasets = {
        "Synapse": (args.synapse_dir, "synapse"),
        "Kvasir": (args.kvasir_dir, "rgb"),
        "ISIC": (args.isic_dir, "rgb"),
    }

    results = {}
    for name, (path, kind) in datasets.items():
        print(f"\n[{name}] Loading from: {path}")
        if not os.path.isdir(path):
            print(f"  ⚠  Directory not found, skipping.")
            continue

        if kind == "synapse":
            imgs = load_synapse_images(path, args.n_images)
        else:
            imgs = load_rgb_images(path, args.n_images)

        if not imgs:
            print(f"  ⚠  No images loaded, skipping.")
            continue

        print(f"  Loaded {len(imgs)} images. Computing HFR...")
        mean_hfr, std_hfr = compute_dataset_hfr(imgs, args.threshold)
        delta_dsc = KNOWN_GCS.get(name, None)

        print(f"  HFR: {mean_hfr:.4f} ± {std_hfr:.4f}  |  ΔDSC (known): {delta_dsc} pp")
        results[name] = {
            "hfr_mean": mean_hfr,
            "hfr_std": std_hfr,
            "delta_dsc": delta_dsc,
        }

    if len(results) < 2:
        print("\nNeed at least 2 datasets for correlation. Check paths.")
        return

    # Spearman correlation
    hfrs = [results[n]["hfr_mean"] for n in results]
    deltas = [results[n]["delta_dsc"] for n in results]
    rho, pval = spearmanr(hfrs, deltas)

    print("\n" + "=" * 50)
    print("SANITY CHECK RESULT")
    print("=" * 50)
    print(f"Spearman ρ (HFR vs ΔDSC) = {rho:.3f}   p = {pval:.3f}")
    if rho < -0.7:
        print("✅ Strong negative correlation: HFR predicts GCS (high HFR → low ΔDSC)")
        print("   Predictive GCS claim is SUPPORTED. Safe to build Contribution 2.")
    elif rho < -0.3:
        print(
            "⚠  Moderate correlation. GCS characterization claim OK; 'prediction' framing risky."
        )
        print(
            "   Recommend expanding to 5-6 datasets before committing to Contribution 2."
        )
    else:
        print(
            "❌ Weak or positive correlation. Fourier Energy may not be the right proxy."
        )
        print(
            "   Try alternative metrics: spatial entropy, object-size ratio, boundary fractal dim."
        )

    plot_results(results, args.out)


if __name__ == "__main__":
    main()
