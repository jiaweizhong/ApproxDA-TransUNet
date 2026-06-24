"""
F1 — Generalization Gap Analysis
Parses existing training logs and plots val DSC + train loss curves for
DA-TransUNet vs ApproxDA-TransUNet on Kvasir-SEG and ISIC 2018.

Run from repo root:
    python experiments/analyze_f1_generalization_gap.py

Outputs:
    paper/figures/fig_f1_generalization_gap.pdf
    paper/figures/fig_f1_generalization_gap.png
"""

import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

LOGS = {
    "Synapse (high CS)": {
        "DA-TransUNet":                "results/DA-TransUNet/Synapse/training-06162026.txt",
        "ApproxDA (gate=pam, M=28)":   "results/ApproxDA-TransUNet/M28R32-PAM-06182026/training_SynapseM28R32PAM.txt",
    },
    "Kvasir-SEG (low CS)": {
        "DA-TransUNet":                "results/DA-TransUNet/Kvasir/training_kvasir-06162026.txt",
        "ApproxDA (gate=learn, M=7)":  "results/ApproxDA-TransUNet/Kvasir/training_kvasir-06162026.txt",
    },
    "ISIC 2018 (low CS)": {
        "DA-TransUNet":                "results/DA-TransUNet/ISIC2018/training_da_isic.txt",
        "ApproxDA (gate=learn, M=7)":  "results/ApproxDA-TransUNet/ISIC18/training_ada_da_isic.txt",
    },
}

# Test DSC from test.py runs (%)
TEST_DSC = {
    "Synapse (high CS)":    {"DA-TransUNet": 79.80, "ApproxDA (gate=pam, M=28)":  80.94},
    "Kvasir-SEG (low CS)":  {"DA-TransUNet": 88.44, "ApproxDA (gate=learn, M=7)": 89.24},
    "ISIC 2018 (low CS)":   {"DA-TransUNet": 88.43, "ApproxDA (gate=learn, M=7)": 89.58},
}

COLORS = {
    "DA-TransUNet":               "#2979C8",
    "ApproxDA (gate=pam, M=28)":  "#E07B00",
    "ApproxDA (gate=learn, M=7)": "#E07B00",
}
LABELS = {
    "DA-TransUNet":               "DA-TransUNet",
    "ApproxDA (gate=pam, M=28)":  "ApproxDA (gate=pam, M=28)",
    "ApproxDA (gate=learn, M=7)": "ApproxDA (gate=learn, M=7)",
}


def parse_log(path):
    epochs, losses, val_epochs, val_dscs = [], [], [], []
    val_idx = 0
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = re.search(r'epoch (\d+)/\d+\s+loss:\s*([0-9.]+)', line)
            if m:
                epochs.append(int(m.group(1)))
                losses.append(float(m.group(2)))
            # Format A: "Validation DSC: 0.8102  best: ..." (Kvasir/ISIC)
            m2 = re.match(r'Validation DSC:\s*([0-9.]+)', line)
            if m2:
                val_idx += 1
                val_epochs.append(val_idx * 15)
                val_dscs.append(float(m2.group(1)) * 100)
            # Format B: "epoch 15 val_dice: 0.6844  best: ..." (Synapse)
            m3 = re.match(r'epoch\s+(\d+)\s+val_dice:\s*([0-9.]+)', line)
            if m3:
                val_epochs.append(int(m3.group(1)))
                val_dscs.append(float(m3.group(2)) * 100)
    return {
        "epochs": np.array(epochs),
        "loss":   np.array(losses),
        "val_epochs": np.array(val_epochs),
        "val_dsc":    np.array(val_dscs),
    }


def gap_stats(data, model, dataset):
    vd = data["val_dsc"]
    best_val = vd.max()
    test_dsc = TEST_DSC[dataset][model]
    # Volatility: std of val DSC in last 30% of epochs
    tail = vd[int(len(vd) * 0.7):]
    volatility = tail.std()
    return best_val, test_dsc, volatility, best_val - test_dsc


def main():
    datasets = list(LOGS.keys())
    models   = list(next(iter(LOGS.values())).keys())

    fig, axes = plt.subplots(3, 2, figsize=(11, 10), constrained_layout=True)
    fig.suptitle("F1 — Generalization Gap: DA-TransUNet vs ApproxDA-TransUNet",
                 fontsize=13, fontweight="bold")

    print("=" * 70)
    print(f"{'Dataset':<14} {'Model':<30} {'BestVal':>8} {'TestDSC':>8} {'Gap':>7} {'Volatility':>10}")
    print("-" * 70)

    for row, dataset in enumerate(datasets):
        ax_dsc  = axes[row][0]
        ax_loss = axes[row][1]

        for model, logpath in LOGS[dataset].items():
            fullpath = REPO_ROOT / logpath
            if not fullpath.exists():
                print(f"  WARNING: log not found: {fullpath}")
                continue

            d = parse_log(fullpath)
            col = COLORS[model]
            lbl = LABELS[model]

            # Val DSC curve
            ax_dsc.plot(d["val_epochs"], d["val_dsc"], color=col, lw=1.8,
                        marker="o", markersize=3, label=lbl)
            # Test DSC horizontal dashed line
            test_val = TEST_DSC[dataset][model]
            ax_dsc.axhline(test_val, color=col, lw=1.0, linestyle="--", alpha=0.7)

            # Train loss curve (smoothed with rolling mean)
            loss = d["loss"]
            if len(loss) > 10:
                smoothed = np.convolve(loss, np.ones(10)/10, mode="valid")
                ax_loss.plot(d["epochs"][9:], smoothed, color=col, lw=1.6, label=lbl)
            else:
                ax_loss.plot(d["epochs"], loss, color=col, lw=1.6, label=lbl)

            # Stats
            best_val, test_dsc, vol, gap = gap_stats(d, model, dataset)
            print(f"{dataset:<14} {lbl:<30} {best_val:>8.2f} {test_dsc:>8.2f} {gap:>+7.2f} {vol:>10.4f}")

        # Val DSC subplot styling
        ax_dsc.set_title(f"{dataset} — Val DSC (%) over epochs", fontsize=10)
        ax_dsc.set_xlabel("Epoch", fontsize=9)
        ax_dsc.set_ylabel("Val DSC (%)", fontsize=9)
        ax_dsc.set_xlim(0, 305)
        ax_dsc.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
        ax_dsc.grid(True, which="major", alpha=0.3)
        ax_dsc.grid(True, which="minor", alpha=0.1)
        ax_dsc.legend(fontsize=8, loc="lower right")
        # Annotate "test DSC" label on right edge
        for model in LOGS[dataset]:
            test_val = TEST_DSC[dataset][model]
            ax_dsc.annotate(f"test={test_val:.2f}%", xy=(305, test_val),
                            fontsize=7, color=COLORS[model], va="center",
                            ha="left", annotation_clip=False)

        # Train loss subplot styling
        ax_loss.set_title(f"{dataset} — Train Loss (10-ep rolling avg)", fontsize=10)
        ax_loss.set_xlabel("Epoch", fontsize=9)
        ax_loss.set_ylabel("Loss", fontsize=9)
        ax_loss.set_xlim(0, 305)
        ax_loss.grid(True, alpha=0.3)
        ax_loss.legend(fontsize=8, loc="upper right")

    print("=" * 70)
    print("\nInterpretation guide:")
    print("  Gap  = best val DSC - test DSC.  Positive = val > test (mild overfit).")
    print("  Volatility = std of val DSC in last 30% of epochs (higher = more unstable).")
    print("  Dashed lines = test DSC (from test.py runs).")

    out_dir = REPO_ROOT / "paper" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "fig_f1_generalization_gap.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(out_dir / "fig_f1_generalization_gap.png", bbox_inches="tight", dpi=200)
    print(f"\nSaved to {out_dir}/fig_f1_generalization_gap.{{pdf,png}}")
    plt.show()


if __name__ == "__main__":
    main()
