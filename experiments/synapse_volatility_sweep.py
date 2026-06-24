"""
Quick Synapse volatility sweep — all ApproxDA configs vs DA-TransUNet.
Answers: is the higher-volatility finding specific to M=28, or universal?

Run from repo root:
    python experiments/synapse_volatility_sweep.py
"""

import re
import numpy as np
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# All Synapse training logs with their test DSC
CONFIGS = [
    {
        "label": "DA-TransUNet (baseline)",
        "log":   "results/DA-TransUNet/Synapse/training-06162026.txt",
        "test_dsc": 79.80,
        "note": "baseline",
    },
    {
        "label": "ApproxDA gate=learn M=7  r=32",
        "log":   "results/ApproxDA-TransUNet/M7R32-06142026/training-log-3skip-06142026.txt",
        "test_dsc": 77.78,
        "note": "gate collapsed",
    },
    {
        "label": "ApproxDA gate=learn M=14 r=64",
        "log":   "results/ApproxDA-TransUNet/M14R64-06152026/training_M14_r64-06152026.txt",
        "test_dsc": 78.04,
        "note": "gate collapsed",
    },
    {
        "label": "ApproxDA gate=cam  M=7  r=32",
        "log":   "results/ApproxDA-TransUNet/M7R32-CAM-06150226/training_M7_r32_cam-06152026.txt",
        "test_dsc": 78.26,
        "note": "",
    },
    {
        "label": "ApproxDA gate=pam  M=7  r=32",
        "log":   "results/ApproxDA-TransUNet/M7R32-PAM-06152026/training_M7_r32_pam-06152026.txt",
        "test_dsc": 78.64,
        "note": "",
    },
    {
        "label": "ApproxDA gate=pam  M=28 r=32  <- best DSC",
        "log":   "results/ApproxDA-TransUNet/M28R32-PAM-06182026/training_SynapseM28R32PAM.txt",
        "test_dsc": 80.94,
        "note": "Phase D best",
    },
    {
        "label": "ApproxDA gate=pam  M=56 r=32",
        "log":   "results/ApproxDA-TransUNet/M56R32-PAM-06192026/training_M56_r32_pam.txt",
        "test_dsc": None,   # may not have test DSC
        "note": "",
    },
    {
        "label": "ApproxDA gate=pam  M=112 r=32",
        "log":   "results/ApproxDA-TransUNet/M112R32-PAM-06182026/training_M112_r32_pam.txt",
        "test_dsc": 79.44,
        "note": "Phase D global anchor",
    },
]


def parse_val_dsc(path):
    val_epochs, val_dscs = [], []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = re.match(r'epoch\s+(\d+)\s+val_dice:\s*([0-9.]+)', line)
            if m:
                val_epochs.append(int(m.group(1)))
                val_dscs.append(float(m.group(2)) * 100)
    return np.array(val_epochs), np.array(val_dscs)


def volatility(arr):
    tail = arr[int(len(arr) * 0.7):]
    return tail.std() if len(tail) > 0 else float("nan")


print()
print("=" * 85)
print(f"{'Config':<42} {'N_val':>5} {'BestVal':>7} {'TestDSC':>7} {'Volatility':>10}  Note")
print("-" * 85)

baseline_vol = None

for cfg in CONFIGS:
    path = REPO / cfg["log"]
    if not path.exists():
        print(f"  [MISSING] {cfg['label']}")
        continue
    _, dscs = parse_val_dsc(path)
    if len(dscs) == 0:
        print(f"  [NO VAL DATA] {cfg['label']}")
        continue
    vol = volatility(dscs)
    best = dscs.max()
    test = cfg["test_dsc"]
    test_str = f"{test:7.2f}" if test is not None else "     ?"
    marker = ""
    if cfg["note"] == "baseline":
        baseline_vol = vol
        marker = " <- DA-TransUNet"
    elif baseline_vol is not None:
        ratio = vol / baseline_vol
        marker = f"  ({ratio:+.1f}×)" if ratio != 1 else ""
    print(f"  {cfg['label']:<42} {len(dscs):>5} {best:>7.2f} {test_str} {vol:>10.4f}{marker}")

print("=" * 85)
print()
print("Volatility = std of val DSC in last 30% of epochs.")
print("Ratio = ApproxDA volatility / DA-TransUNet volatility.")
print("> 1.0 -> more unstable than baseline on Synapse (high-CS task).")
