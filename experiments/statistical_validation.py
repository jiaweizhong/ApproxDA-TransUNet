"""
Statistical validation for BSPC submission, computed from EXISTING result logs
(no new training). Produces:
  1. Paired bootstrap 95% CI + Wilcoxon signed-rank test for ApproxDA vs
     DA-TransUNet mean DSC, per dataset (Kvasir-SEG, ISIC 2018 -- Synapse
     excluded here, see warning below; ACDC and CVC-ClinicDB excluded: no
     per-case DA-TransUNet log exists locally, published baselines only).
  2. Bootstrap 95% CI for GCS (= max_M DSC - min_M DSC) on Synapse and
     Kvasir-SEG, resampling test cases (unaffected by the Synapse warning
     below -- this uses ApproxDA-only logs, not the DA-TransUNet baseline).
  3. Synapse organ-level (n=8) supplementary paired check, using DA-TransUNet's
     *published* per-organ numbers (Reference/DA-TransUNet.md) against our own
     M=28 per-organ log -- NOT significant at alpha=0.05, reported for
     transparency, not as confirmatory evidence (see docstring in main()).

KNOWN ISSUE (2026-09-04): results/DA-TransUNet/Synapse/inference-06162026.txt
does NOT reproduce the paper's reported 79.80% DA-TransUNet Synapse test DSC
(its own summary line says 72.03%). The matching training log's best val DSC
(79.52%) DOES match the paper, so this looks like a stale/mismatched
checkpoint at test time, not a training failure. experiments/
synapse_volatility_sweep.py has 79.80 hardcoded, confirming the correct number
was recorded by hand at some point, but the per-case log for that run is not
the one currently committed. Case-level Synapse pairing is therefore excluded
from Part 1 until this is fixed (needs a re-verified checkpoint + test-only
inference re-run; see PENDING_RUNS.md E3a).

All paths relative to repo root.
"""
import re
import numpy as np
from pathlib import Path

REPO = Path(r"c:\Users\jiawe\Repos\AdaDA-TransUNet")
RNG = np.random.default_rng(42)
N_BOOT = 10000

LINE_RE = re.compile(r"case\s+(\S+)\s+mean_dice\s+([0-9.]+)")


def parse_percase(path: Path) -> dict:
    """Return {case_name: dice} from a test log."""
    text = path.read_text(errors="ignore")
    out = {}
    for m in LINE_RE.finditer(text):
        case, dice = m.group(1), float(m.group(2))
        out[case] = dice
    return out


def paired_arrays(da: dict, approx: dict):
    common = sorted(set(da) & set(approx))
    a = np.array([da[c] for c in common])
    b = np.array([approx[c] for c in common])
    return common, a, b


def bootstrap_paired_diff_ci(a, b, n_boot=N_BOOT, alpha=0.05, rng=RNG):
    """Bootstrap 95% CI for mean(b - a), resampling paired cases with replacement."""
    n = len(a)
    diffs = b - a
    boot_means = np.empty(n_boot)
    idx_pool = np.arange(n)
    for i in range(n_boot):
        idx = rng.choice(idx_pool, size=n, replace=True)
        boot_means[i] = diffs[idx].mean()
    lo, hi = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return diffs.mean(), lo, hi


def wilcoxon_signed_rank(a, b):
    from scipy.stats import wilcoxon
    diffs = b - a
    # drop exact ties (wilcoxon errors on all-zero diffs; unlikely here)
    try:
        stat, p = wilcoxon(diffs)
    except ValueError:
        stat, p = np.nan, np.nan
    return stat, p


def paired_ttest(a, b):
    from scipy.stats import ttest_rel
    stat, p = ttest_rel(b, a)
    return stat, p


def bootstrap_gcs_ci(percase_by_M: dict, n_boot=N_BOOT, alpha=0.05, rng=RNG):
    """
    percase_by_M: {M: {case: dice}}. Resamples the common case set jointly
    across all M (same resampled case indices applied to every M), computes
    GCS = max_M(mean dice) - min_M(mean dice) per resample.
    """
    Ms = sorted(percase_by_M.keys())
    common = sorted(set.intersection(*[set(d) for d in percase_by_M.values()]))
    n = len(common)
    mats = {M: np.array([percase_by_M[M][c] for c in common]) for M in Ms}

    # point estimate
    means = {M: mats[M].mean() for M in Ms}
    gcs_point = (max(means.values()) - min(means.values())) * 100  # pp

    idx_pool = np.arange(n)
    boot_gcs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.choice(idx_pool, size=n, replace=True)
        m_means = [mats[M][idx].mean() for M in Ms]
        boot_gcs[i] = (max(m_means) - min(m_means)) * 100
    lo, hi = np.percentile(boot_gcs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return n, gcs_point, lo, hi, means


def main():
    print("=" * 78)
    print("PART 1 — Paired ApproxDA vs DA-TransUNet: bootstrap CI + significance")
    print("=" * 78)

    # NOTE: Synapse intentionally excluded here -- see KNOWN ISSUE in module
    # docstring. The DA-TransUNet Synapse inference log does not reproduce
    # the reported 79.80% test DSC; pairing against it would silently
    # understate the true improvement. Re-add once a re-verified log exists:
    #   "Synapse": (
    #       REPO / "results/DA-TransUNet/Synapse/inference-06162026.txt",  # <- fix this file first
    #       REPO / "results/ApproxDA-TransUNet/M28R32-PAM-06182026/test_SynapseM28R32PAM.txt",
    #   ),
    pairs = {
        "Kvasir-SEG": (
            REPO / "results/DA-TransUNet/Kvasir/inference_kvasir-06162026.txt",
            REPO / "results/ApproxDA-TransUNet/KvasirD5-M56R32-PAM-06182026/test_kvasir_M56_r32_pam.txt",
        ),
        "ISIC 2018": (
            REPO / "results/DA-TransUNet/ISIC2018/test_da_isic.txt",
            REPO / "results/ApproxDA-TransUNet/ISIC18/test_learn_M7.txt",
        ),
    }

    summary_rows = []
    for name, (da_path, approx_path) in pairs.items():
        da = parse_percase(da_path)
        approx = parse_percase(approx_path)
        common, a, b = paired_arrays(da, approx)
        n_da, n_approx, n_common = len(da), len(approx), len(common)
        mean_diff, lo, hi = bootstrap_paired_diff_ci(a, b)
        w_stat, w_p = wilcoxon_signed_rank(a, b)
        t_stat, t_p = paired_ttest(a, b)
        print(f"\n[{name}]")
        print(f"  DA cases logged: {n_da}   ApproxDA cases logged: {n_approx}   paired (common): {n_common}")
        print(f"  DA-TransUNet mean DSC (paired subset): {a.mean()*100:.2f}%")
        print(f"  ApproxDA     mean DSC (paired subset): {b.mean()*100:.2f}%")
        print(f"  Mean paired difference: {mean_diff*100:+.2f} pp   95% bootstrap CI: [{lo*100:+.2f}, {hi*100:+.2f}] pp")
        print(f"  Wilcoxon signed-rank: W={w_stat:.1f}, p={w_p:.4g}")
        print(f"  Paired t-test:        t={t_stat:.3f}, p={t_p:.4g}")
        summary_rows.append((name, n_common, a.mean()*100, b.mean()*100, mean_diff*100, lo*100, hi*100, w_p, t_p))

    print("\n" + "=" * 78)
    print("PART 2 — Bootstrap 95% CI for GCS (all datasets with per-M per-case logs)")
    print("=" * 78)

    gcs_configs = {
        "Synapse": {
            7: REPO / "results/ApproxDA-TransUNet/M7R32-PAM-06152026/inference_M7_r32_pam-06152026.txt",
            28: REPO / "results/ApproxDA-TransUNet/M28R32-PAM-06182026/test_SynapseM28R32PAM.txt",
            56: REPO / "results/ApproxDA-TransUNet/M56R32-PAM-06192026/test_M56_r32_pam.txt",
            112: REPO / "results/ApproxDA-TransUNet/M112R32-PAM-06182026/test_M112_r32_pam.txt",
        },
        "Kvasir-SEG": {
            7: REPO / "results/ApproxDA-TransUNet/KvasirD3-M7R32-PAM-06182026/test_kvasir_M7_r32_pam.txt",
            28: REPO / "results/ApproxDA-TransUNet/KvasirD4-M28R32-PAM-06182026/test_kvasir_M28_r32_pam.txt",
            56: REPO / "results/ApproxDA-TransUNet/KvasirD5-M56R32-PAM-06182026/test_kvasir_M56_r32_pam.txt",
            112: REPO / "results/ApproxDA-TransUNet/KvasirD6-M112R32-PAM-06182026/test_kvasir_M112_r32_pam.txt",
        },
        "ACDC": {
            7: REPO / "results/ApproxDA-TransUNet/ACDC/test_acdc_M7.txt",
            28: REPO / "results/ApproxDA-TransUNet/ACDC/test_acdc_M28.txt",
            56: REPO / "results/ApproxDA-TransUNet/ACDC/test_acdc_M56.txt",
            112: REPO / "results/ApproxDA-TransUNet/ACDC/test_acdc_M112.txt",
        },
        "CVC-ClinicDB": {
            7: REPO / "results/ApproxDA-TransUNet/CVC/test_cvc_M7.txt",
            28: REPO / "results/ApproxDA-TransUNet/CVC/test_cvc_M28.txt",
            56: REPO / "results/ApproxDA-TransUNet/CVC/test_cvc_M56.txt",
            112: REPO / "results/ApproxDA-TransUNet/CVC/test_cvc_M112.txt",
        },
    }

    gcs_rows = []
    for name, m_paths in gcs_configs.items():
        percase_by_M = {M: parse_percase(p) for M, p in m_paths.items()}
        for M, d in percase_by_M.items():
            print(f"  [{name}] M={M}: {len(d)} cases logged")
        n, gcs_point, lo, hi, means = bootstrap_gcs_ci(percase_by_M)
        print(f"\n[{name}] common cases across all M: {n}")
        print(f"  Per-M mean DSC: " + ", ".join(f"M={M}: {v*100:.2f}%" for M, v in sorted(means.items())))
        print(f"  GCS (point estimate, common-case subset): {gcs_point:.2f} pp")
        print(f"  95% bootstrap CI: [{lo:.2f}, {hi:.2f}] pp\n")
        gcs_rows.append((name, n, gcs_point, lo, hi))

    print("=" * 78)
    print("PART 3 — Synapse organ-level (n=8) supplementary check")
    print("Uses DA-TransUNet's PUBLISHED per-organ numbers (Reference/DA-TransUNet.md,")
    print("their 79.80% DSC row) paired with our own M=28 per-organ log (Table X /")
    print("tab:per_organ). NOT a substitute for case-level pairing -- organs are")
    print("fixed anatomical categories aggregated over all 12 volumes, not")
    print("independently resampled patients. Reported for transparency only.")
    print("=" * 78)

    organs = ["Aorta", "Gallbladder", "Kidney(L)", "Kidney(R)", "Liver", "Pancreas", "Spleen", "Stomach"]
    da_organ = np.array([86.54, 65.27, 81.70, 80.45, 94.57, 61.62, 88.53, 79.73])
    approx_organ = np.array([88.0, 68.6, 84.5, 82.2, 93.3, 62.7, 89.4, 78.7])
    organ_diffs = approx_organ - da_organ

    from scipy.stats import wilcoxon, ttest_rel
    w_stat, w_p = wilcoxon(organ_diffs)
    t_stat, t_p = ttest_rel(approx_organ, da_organ)
    n = len(organ_diffs)
    idx_pool = np.arange(n)
    boot_means = np.array([organ_diffs[RNG.choice(idx_pool, size=n, replace=True)].mean() for _ in range(N_BOOT)])
    lo, hi = np.percentile(boot_means, [2.5, 97.5])

    for o, d in zip(organs, organ_diffs):
        print(f"  {o:12s}: {d:+.2f} pp")
    print(f"  Mean diff: {organ_diffs.mean():+.3f} pp   ({(organ_diffs > 0).sum()}/8 organs favor ApproxDA)")
    print(f"  Wilcoxon signed-rank: W={w_stat:.1f}, p={w_p:.4f}  (NOT significant at alpha=0.05)")
    print(f"  Paired t-test:        t={t_stat:.3f}, p={t_p:.4f}  (NOT significant at alpha=0.05)")
    print(f"  Organ-resampling bootstrap 95% CI: [{lo:+.2f}, {hi:+.2f}] pp")

    print("\n" + "=" * 78)
    print("MACHINE-READABLE SUMMARY")
    print("=" * 78)
    print("\n-- Paired significance (case-level) --")
    for row in summary_rows:
        print(row)
    print("\n-- GCS bootstrap --")
    for row in gcs_rows:
        print(row)
    print("\n-- Synapse organ-level (supplementary, underpowered) --")
    print(("Synapse-organ", n, organ_diffs.mean(), lo, hi, w_p, t_p))


if __name__ == "__main__":
    main()
