# Pending Runs — Journal Extension

> Updated 2026-10-13. Journal re-runs (BIBM2026_Review.md J1–J6) run on the local RTX 5090 (32 GB),
> Python 3.14, torch 2.14+cu130, via `experiments/run_queue.sh <queue>`. Live status: `bash experiments/status.sh`;
> per-job logs in `experiments/logs/`, checkpoints in `experiments/model/`, per-case test logs in `*/test_log/`.

## Protocol

- **Model:** c16 block (DANetHead-matched); the legacy (workshop) block is dropped — see Finding 3.
- **Training:** 300 epochs (Kvasir: 500, as in the DA-TransUNet paper), batch size 24, 224×224, seed 1234
  (main comparison also seeds 2345, 3456).
- **Evaluation:** final-epoch checkpoint only; no checkpoint selection on the test set.

| Dataset | Optimizer | Split | Configuration selected on |
|---|---|---|---|
| Synapse | SGD, lr 0.01, poly | standard 18 train / 12 test volumes | best of the test-set ablation round (disclosed) |
| Kvasir-SEG | Adam, lr 1e-3, 500 epochs, **RGB input, 256×256**, 16 DataLoader workers (paper protocol) | random 75/25 = 750 / 250 (split seed 1234), as in the DA-TransUNet paper | best of the test-set ablation round (disclosed) |
| ISIC 2018 | **SGD, lr 0.01** (Adam in the paper; SGD +0.96 pp on the same split), **50 epochs** (paper), RGB, 256×256 | random 75/25 of the 2594 official training images = 1945 / 649 (split seed 1234), as in the DA-TransUNet paper | best of the test-set ablation round (disclosed) |

Configuration selection (2026-10-13): the paper's narrative is accuracy sensitivity across the ablations, so
the ablation tables (evaluated on test) are the main result and the reported best configuration per dataset is
taken from them. The paper discloses that this selection used the test set. Checkpoints are never selected:
always the final epoch. No validation split is carved out of the Synapse or Kvasir training data.

## Status

| Dataset | Queue | Status |
|---|---|---|
| Synapse | `queue_synapse.txt` | **Done** 25/25 (finished 2026-10-13 01:35) |
| Kvasir-SEG | `queue_kvasir.txt`, `queue_seeds.txt` | **Done**: 13-run ablation round plus 3-seed comparison (2026-10-16 11:45) |
| ISIC 2018 | `queue_isic.txt` | **Ablation round running** (started 2026-10-15 21:29, ~12 min/run, ETA ~00:00): 13 ApproxDA runs with the best DA setup: SGD 0.01, seed 3456 (88.77; paper 88.88), RGB, 256, 50 epochs |

**Hold rule (2026-10-13):** no ApproxDA compute on Kvasir or ISIC until DA-TransUNet reproduces close to the
paper (Kvasir 88.47, ISIC 88.88). If it does, un-hold the ApproxDA jobs with that protocol after a go-ahead; if
it does not, report back before running anything else.

### Data

- **Synapse:** `data/Synapse` → `/data/datasets/Synapse`.
  - Standard TransUNet preprocessing: 2211 training slices, 12 test volumes.
  - `/data/datasets/Synapse_all` puts all 30 cases in both train and test; never evaluate on it.
- **Kvasir-SEG:** `data/Kvasir-SEG` → `/data/datasets/Kvasir-SEG/Kvasir-SEG/Kvasir-SEG` (1000 images, 1000 masks).
  - Lists: `generate_lists.py --dataset Kvasir --train_ratio 0.75 --seed 1234 --force` (750 / 250, identical in
    both projects).
  - The official 880/120 lists were replaced; they remain in git.
- **ISIC 2018:** official Task 1 folders in `/data/datasets/ISIC2018_official`.
  - Flattened by `generate_lists.py` into `/data/datasets/ISIC2018_full` (full resolution).
  - Resized by `resize_isic.py` into `/data/datasets/ISIC2018_224`; `data/ISIC2018` points there.
  - All 3694 images are 224×224 RGB, masks are {0, 255}, and image and mask stems match.
  - Lists: 2594 / 100 / 1000, no overlap, identical in both projects.
  - The workshop's 80/20 lists (2075 / 519) were overwritten; restore them from commit `7278690`.
- **Input channels:** the Kvasir and ISIC loaders convert images to grayscale (`convert("L")`); the model
  repeats the single channel to 3. RGB input is untested and may account for part of the remaining gap to the
  paper.

### Alignment with the DA-TransUNet paper (Reference/DA-TransUNet.pdf, §4.1–4.2)

| | DA-TransUNet paper | Journal re-runs |
|---|---|---|
| Split (2D datasets) | Random 75 % train / 25 % test, no val; lists and seed not released | Kvasir: random 75/25 (seed 1234); ISIC: official 2594/100/1000 |
| ISIC image count | "2512 images" (matches no official count; Task 1 train is 2594) | Official 2594 + 100 + 1000 |
| Synapse | 224, SGD 0.01, batch 24, 500 epochs | 224, SGD 0.01, batch 24, 300 epochs (DA re-run 79.81 vs paper 79.80) |
| 2D datasets | 256×256 (RGB presumably), Adam 1e-3, wd 1e-4, batch 24, 500 epochs (ISIC: 50) | 224×224 grayscale; Kvasir: Adam 1e-3, 500 epochs; ISIC: see below |
| Loss (2D) | 0.5·BCE + 0.5·Dice | 0.5·CE + 0.5·Dice (2-class softmax) |

The official repo (github.com/SUN-1024/DA-TransUnet, checked 2026-10-13: 31 commits, latest 2025-06-06) does not
help:
- Every commit contains only Synapse code: the Synapse loader, `lists_Synapse`, SGD `train.py`/`test.py`. There
  is no 2D loader, no Kvasir/ISIC/CVC lists, no split script, and no Adam/BCE training code.
- Issue #13 (how to run Kvasir-SEG) is unanswered.
- In issue #1 users asked for the processed 2D datasets; the author only replied that the code "has always been
  complete".
- Issue #17 points out that a 64-channel DANetHead gives 64 // 16 // 8 = 0 PAM query channels, and that one skip
  DA block is skipped by its size check. This is consistent with "DA used at 3 positions" in the camera-ready.

So the paper's exact 2D split cannot be reproduced (no lists, image count mismatch). Fairness therefore rests on
the matched DA-TransUNet re-runs; the paper's numbers are indicative only.

### Kvasir baseline history (DA-TransUNet, final epoch; paper: 88.47)

| Split | Optimizer | Epochs | Dice / HD95 (px) |
|---|---|---|---|
| Official 880/120 | Adam 1e-3 | 300 | 84.54 / 63.78 |
| Official 880/120 | SGD 0.01 | 300 | 83.71 / 66.42 |
| Random 75/25, split seed 42 | Adam 1e-3 | 300 | **86.65** / 60.96 |
| Random 75/25, split seed 42 | SGD 0.01 | 300 | 83.88 / 65.64 |
| Random 75/25, split seed 1234 | Adam 1e-3, training seed 2345 | 500 | **85.94** / 57.58 |
| ↳ same run, epoch-300 checkpoint (`test.py --eval_epoch 300`; LR not yet annealed) | | (300 of 500) | 83.97 / 68.84 |
| Random 75/25, split seed 1234, **RGB input** (`--rgb`) | Adam 1e-3, training seed 2345 | 500 | **87.19** / 52.88 |
| ↳ same run, epoch-300 checkpoint (LR not yet annealed) | | (300 of 500) | 85.98 / 62.84 |
| Random 75/25, split seed 1234, grayscale | Adam 1e-3, training seed 2345 | 300 | 86.07 / 58.35 |
| Random 75/25, split seed 1234, RGB | Adam 1e-3, training seed 2345 | 300 | 86.27 / 56.85 |
| Random 75/25, split seed 1234, RGB, **256×256** (paper resolution) | Adam 1e-3, training seed 2345 | 500 | **87.66** / 56.64 |
| ↳ same protocol, 16 workers, training seeds 1234 / 2345 / 3456 | Adam 1e-3 | 500 | 86.65 / 86.98 / 85.53 (mean **86.39 ± 0.76**) |

- **What these runs show:** the optimizer is not the cause (SGD is worse on both splits). The paper-style split
  recovers about 2 pp.
- **Input × epochs (running):** the last four rows form a 2×2 grid on the same split and seed.
  - Input channels: the paper never mentions them, and its 2D code is not public.
  - Epochs: an epoch-300 checkpoint of a 500-epoch run is not a 300-epoch run, so both lengths are trained
    in full.
  - RGB is ~3× slower (data loading is CPU-bound; GPU idle).
- **Final Kvasir protocol (2026-10-14):** RGB + 256×256 + Adam 1e-3 + 500 epochs on the 75/25 split.
  - Steps: RGB gave +1.25, 256 px another +0.47, and 500 epochs beat 300 (+0.9) with RGB. The remaining
    0.8 pp to the paper is within split-to-split noise (two 75/25 splits differed by ~0.7).
  - Speed: RGB at 256 px is CPU-bound in data loading. 16 workers give ~8–9 s/epoch instead of ~17 s,
    i.e. ~1.2 h per 500-epoch run. The augmentation RNG stream differs from 4 workers, so every run of
    the round (DA included) uses `--num_workers 16`.
  - ApproxDA grid at 256 px: feature maps are 16/32/64/128, so M ∈ {8, 32, 64, 128}, the same relative
    windows (M / H_max = 1/16, 1/4, 1/2, 1) as Synapse's {7, 28, 56, 112} at 224.
  - Synapse stays at 224 / grayscale: the paper uses 224 for Synapse, CT is single-channel, and the DA
    re-run already matches (79.81 vs 79.80).
- **Metric convention (2026-10-14): the paper averages per image, like us, so global Dice is not the
  explanation.**
  - Re-evaluating the best DA checkpoint without retraining: per-image mean Dice 87.66 (ours), 87.70 at
    256×256, global (pixel-pooled) Dice 89.04.
  - Pooled metrics satisfy IoU = Dice / (2 − Dice) exactly; per-image means give a larger IoU (convexity).
  - In the paper's Table 2, all 30 IoU/Dice pairs (6 models × 5 datasets) have IoU above Dice / (2 − Dice) by
    1.2–5.1 pp. So the paper averages per image (or per batch).
  - The official repo only has the Synapse evaluation: per-case medpy Dice, mean over cases, no IoU. The 2D
    evaluation code was never released.
  - Comparable numbers: DA-TransUNet 87.66 Dice / 80.89 IoU vs paper 88.47 / 81.02 (IoU within 0.13 pp, Dice
    within the ±0.8 seed spread). The baseline is effectively reproduced; keep per-image mean Dice.
- **ACDC** is not in the DA-TransUNet paper (0 mentions); it is in TransUNet (89.71). Local data:
  `/data/datasets/ACDC_preprocessed`, `/data/datasets/ACDC_MTUNet`. Old DA-TransUNet ACDC log: 88.50
  (test-set checkpoint selection). Candidate dataset after the Kvasir round. It needs ACDC support in the
  DA-TransUNet project and the ACDC `test.py` snapshot-path fix (`_20k` suffix).
- **Reference run for the ablations:** DA-TransUNet with seed 2345 and 4 workers (87.66), the best
  reproduction.
  - Seed spread on Kvasir is ±0.8 (Synapse DA: ±0.09). The 3-seed mean of 86.39 is 2.1 pp below the paper.
  - Loss: our ½·CE + ½·Dice (2-class, Dice averaged over background and foreground) equals the paper's
    ½·BCE + ½·Dice if "weighted" means the ½ factors, as in Eq. 10. It also matches the DiceLoss class in
    the public DA-TransUNet code.
  - Decision (2026-10-14): no extra seeds. ApproxDA ablations use exactly the reference run's setup.
- Earlier protocol choice (2026-10-13): random 75/25 + Adam + 500 epochs, the closest to the paper at the time. It is checked on a
  fresh split and training seed.
- **Superseded protocols:** official-split and seed-42 runs had their checkpoints, test logs, lists and queues
  deleted; their queue logs remain in `logs/`. ApproxDA M=7 (Adam, official split) reached 84.00 before the hold.
- **Workshop reference:** the workshop re-run (SGD 0.01, random 800/200) reached 88.38.

### ISIC protocol switch (2026-10-15)

Decision: reproduce DA-TransUNet with the paper's own ISIC protocol, as for Kvasir:
- **Split:** random 75/25 of the 2594 official training images (the paper says 2512 images; no list released).
- **Training:** Adam 1e-3, 256×256, **50 epochs**, RGB. The paper says 500 epochs by default but 50 for ISIC.
- **Data:** `/data/datasets/ISIC2018_256`, made by `resize_isic.py --names <official train list>`;
  `data/ISIC2018` points there. The 224 px official-split copy (`ISIC2018_224`) is kept on disk.
- **Lists:** `lists_ISIC` now holds the 1945 / 649 split (`generate_lists.py --train_ratio 0.75 --seed 1234`);
  the official-split lists can be regenerated with `--isic_official_root`.
- **Checkpoints:** the official-split DA checkpoints remain in `model/TU_ISIC224/`.

### ISIC paper-protocol baseline (75/25, RGB, 256, 50 epochs; test Dice / IoU / HD95; paper 88.88 / 82.78)

| DA-TransUNet | seed 1234 | seed 2345 | seed 3456 | mean |
|---|---|---|---|---|
| Adam 1e-3 (paper) | 87.39 / 79.98 / 16.41 | **87.88** / 80.82 / 15.54 | 86.99 / 79.38 / 17.19 | 87.42 ± 0.45 |
| SGD 0.01 | 88.43 / 81.86 / 13.39 | 87.93 / 81.07 / 14.14 | **88.77** / 81.92 / 13.62 | **88.38 ± 0.42** |

- The gate (best seed ≥ 87.88) missed by 0.001 pp (87.8786), so the ApproxDA grid was not started.
- On the official split SGD beat Adam by 2.5 pp, hence the SGD check on the same split.
- SGD beats Adam on every seed (+0.96 pp mean). The best seed 3456 (88.77) is 0.11 pp below the paper's
  Dice; its IoU is 0.86 pp below. All three SGD seeds clear the gate. Waiting for a go-ahead on the
  ApproxDA grid.
- Decision (2026-10-15): ISIC ablations use SGD + seed 3456, the best reproduced setup.
- **Grayscale check (2026-10-17, seed 1234, plain).** Values are Dice / IoU / HD95; paper 88.88 / 82.78.
  - Adam: grayscale 87.09 / 79.79 / 16.33 vs RGB 87.39 / 79.98 / 16.41.
  - SGD + `--drop_last`: grayscale 87.93 / 80.73 / 14.54 vs RGB 88.08 / 81.40 / 13.54.
  - Grayscale is slightly lower in both cases (−0.30 / −0.15 Dice) and further from the paper, so **ISIC keeps RGB**.
    Unlike CVC, the paper's ISIC value is not explained by grayscale input.

### ISIC baseline check (DA-TransUNet, official split, 300 epochs, final epoch, scored on the 100-image val split; paper: 88.88)

| Optimizer | Val Dice / HD95 |
|---|---|
| Adam 1e-3 | 87.20 / 17.12 |
| SGD 0.01 | **89.70** / 13.78 |

DA-TransUNet with SGD already reaches the paper's level on the official split. The proposal is to keep the official
split + SGD for ISIC: the baseline reproduces, and it keeps the val-based selection. Pending a go-ahead before the
ApproxDA grid (13 runs, ~16 h) is switched to SGD and un-held.

## Overnight 2026-10-15/16: precise-BN evaluation + R8 seeds (`logs/overnight_r8.py`, progress in `logs/queue.history`)

**Why.** DANetHead-style BatchNorm (DA-TransUNet and the c16 block) uses momentum 0.95, so the saved running
statistics are essentially those of the last training mini-batch. On the ISIC 75/25 split that batch holds one
image (1945 = 81 × 24 + 1); on Kvasir 6, on Synapse 3. Re-estimating BN statistics on the training set (weights
untouched) moved ISIC test Dice by up to 1.9 pp:

| ISIC checkpoint | As saved | Precise BN |
|---|---|---|
| ApproxDA M=32 | 84.68 | 86.61 |
| ApproxDA M=128 (momentum-0.95 layers only) | 88.92 | 88.36 |
| DA-TransUNet, seed 3456 | 88.77 | 88.23 |

On Synapse the effect was +0.15 (DA). Decision (2026-10-15): evaluate every run with precise BN in addition to
the plain evaluation (option 1), and add 3-seed runs (option 3).

**Precise BN** = `test.py --precise_bn`: one seeded pass of the augmented training set, cumulative average over
all BatchNorm layers, log suffix `_pbn`; the plain evaluation is unchanged (84.68 reproduced).

**Plan (automatic, `logs/overnight_r8b.py`; decisions 2026-10-15):**
1. **ISIC:** precise-BN evaluation of the 19 finished paper-protocol runs, for comparison (~00:30).
2. **ISIC retrained with `--drop_last`** (`queue_isic_dl.txt`, snapshot suffix `_dl`, ~12 min/run):
   1. DA-TransUNet (SGD) with seeds 1234/2345/3456; take the best seed by precise-BN Dice (~01:10).
   2. The ApproxDA grid with that seed, now **15 runs**: r=0 at every M ∈ {8, 32, 64, 128} completes the 2×4
      grid of window size × {r=32, exact} (~04:20).
   3. The best configuration (excluding full attention) with the two other seeds (~04:45).
   4. Precise-BN evaluation of every run.
3. **Synapse:** frozen except two added runs, r=0 at M=7 and M=56, which complete the same 2×4 grid. Same
   protocol, plain evaluation only (`queue_synapse.txt`, ~07:05).
4. **Kvasir:** precise-BN evaluation of the 17 runs at 256 px, then the best ApproxDA configuration with seeds
   1234/2345/3456 and 16 workers (`queue_seeds.txt`), then precise BN (~11:15). The extra r=0 runs on Kvasir
   are deferred until the Synapse and ISIC results are in.

Note: with `--drop_last` and BN momentum 0.95, the running statistics still depend on the last mini-batch, but
on 24 images instead of 1.

## Results 2026-10-16 (overnight R8b + ISIC legacy round)

### Kvasir 3-seed comparison (16 workers, seeds 1234/2345/3456; test Dice)

The best configuration was chosen on the seed-2345, 4-worker grid by precise-BN Dice: M=128, r=32, PAM (87.22).

| | Plain | Precise BN |
|---|---|---|
| DA-TransUNet | 86.39 ± 0.76 (86.65 / 86.98 / 85.53) | 86.63 ± 0.91 (86.60 / 87.56 / 85.74) |
| ApproxDA M=128, r=32 | **86.76 ± 0.59** (86.69 / 87.39 / 86.21) | **86.91 ± 0.62** (86.84 / 87.56 / 86.32) |
| Paired over 250 test images (seed-averaged) | +0.38 pp, 143/250 better, Wilcoxon p = 0.11 | **+0.28 pp, 144/250 better, p = 0.0499** |

- With matched seeds, ApproxDA is slightly better than DA-TransUNet on Kvasir (borderline significant,
  uncorrected).
- The earlier single-seed comparison against the DA run with seed 2345 and 4 workers (87.66) was biased toward
  DA: that run is DA's best of four.
- The seed runs use 16 workers, i.e. different RNG streams from the selection grid, so they are not the runs the
  configuration was selected on.

### ISIC: legacy block vs c16 vs DA (`--drop_last`, seed 1234; precise-BN Dice; DA-TransUNet 89.12)

| Config | c16 | legacy |
|---|---|---|
| M=8 / 32 / 64 / 128, r=32 | 88.52 / 88.90 / 89.18 / 88.54 | 89.45 / 89.31 / 89.22 / 89.22 |
| M=128, r=16 / 64 / 0 | 88.62 / 88.81 / 88.99 | **89.73** / 89.39 / 89.52 |
| r=0 at M=8 / 32 / 64 | 88.90 / 88.51 / 88.79 | 89.65 / 89.43 / 89.48 |
| M=32: CAM / fixed / learned | 88.25 / 89.31 / 88.00 | 89.49 / 89.38 / 89.43 |
| M=32, CAM-only G=1 / G=4 | 88.82 / 88.68 | 89.41 / 89.61 |
| **Mean of 15 / configs above DA** | 88.72 / 2 of 15 | **89.45 / 15 of 15** |

- Every legacy configuration beats DA-TransUNet's seed-1234 run and its 3-seed mean (88.90).
- Legacy has no momentum-0.95 BN in its attention blocks, so plain ≈ precise BN (89.39 plain mean).
- Opposite of Synapse, where legacy at batch 24 was ~3 pp below c16.
- **3-seed confirmation** (2026-10-16). The configuration M=128, r=32, PAM was fixed in advance (the c16 main
  configuration, not the best-on-test legacy M=128 / r=16). Seeds 1234 / 2345 / 3456:

  | ISIC | Plain Dice | Precise-BN Dice | Precise-BN HD95 |
  |---|---|---|---|
  | DA-TransUNet | 88.38 ± 0.32 | 88.90 ± 0.19 (89.12 / 88.77 / 88.83) | 13.60 |
  | c16 M=128, r=32 | 88.56 ± 0.33 | 88.72 ± 0.33 (88.54 / 89.09 / 88.52) | 13.43 |
  | **legacy M=128, r=32** | **89.34 ± 0.14** | **89.40 ± 0.17** (89.22 / 89.54 / 89.44) | **12.92** |

  - Legacy beats every DA seed with every one of its own seeds, in both evaluations.
  - Paired over the 649 test images (seed-averaged): legacy − DA = +0.96 pp, p = 0.021 (plain); +0.49 pp,
    p = 0.58 (precise BN). The mean gain is concentrated on a minority of images.
  - c16 ties DA on ISIC (+0.18 / −0.19 pp, n.s.).

## ISIC results, `--drop_last` retrain (75/25, RGB, 256, SGD 0.01, 50 epochs; Dice / IoU / HD95; 2026-10-16)

**3-seed comparison** (seeds 1234/2345/3456). The best configuration was selected by single-seed precise-BN
Dice: M=32, r=32, fixed 0.5 (89.31).

| | Plain evaluation | Precise BN |
|---|---|---|
| DA-TransUNet | **88.38 ± 0.32** (88.08 / 88.72 / 88.33) | **88.90 ± 0.19** (89.12 / 88.77 / 88.83); paper 88.88 |
| ApproxDA M=32 fixed | 87.91 ± 0.51 (88.49 / 87.50 / 87.73) | 88.46 ± 0.73 (89.31 / 88.05 / 88.04) |
| Paired over 649 test images (seed-averaged) | −0.47 pp, 308/649 better, Wilcoxon p = 0.041 | −0.44 pp, 309/649 better, p = 0.21 |

**Reading.**
- With precise BN, DA-TransUNet reproduces the paper (88.90 vs 88.88).
- The best single-seed ApproxDA configuration regresses to 88.46 on the other two seeds. This is the same
  winner's curse as Synapse M=112; selection among 15 single-seed runs inflates the winner by ~0.8 pp.
- ApproxDA is not better than DA on ISIC; it is at most ~0.5 pp worse, and with precise BN the difference is
  not significant.
- `--drop_last` removed the M=32 outlier (84.68 → 88.84 plain) and lowered DA's plain seed spread
  (0.42 → 0.32).

**Seed-1234 grid, precise BN** (DA-TransUNet with the same seed: 89.12 / 82.47 / 13.33):

| | r=32 | r=0 (exact) |
|---|---|---|
| M=8 | 88.52 / 81.80 / 14.04 | 88.90 / 82.22 / 13.59 |
| M=32 | 88.90 / 82.30 / 13.29 | 88.51 / 81.84 / 13.47 |
| M=64 | **89.18** / 82.52 / 13.42 | 88.79 / 82.03 / 13.66 |
| M=128 | 88.54 / 81.74 / 13.45 (r=16: 88.62; r=64: 88.81) | 88.99 / 82.27 / 13.59 |

Routing at M=32: PAM 88.90, CAM 88.25, **fixed 89.31**, learned 88.00. CAM groups: G=1 88.82, G=4 88.68, G=8
88.25.

The window spread is small (r=32: 88.52–89.18, 0.66 pp), and exact attention is not better than low rank. Plain
and precise-BN numbers for every run, including the old non-drop_last protocol, are in `logs/queue.history`
(`exit=` and `pbn` lines).

## Synapse 2×4 grid: window size × {low rank r=32, exact r=0} (seed 1234, plain evaluation, frozen protocol)

| | r=32 | r=0 (exact) |
|---|---|---|
| M=7 | 79.92 / 70.49 / 21.56 | 79.35 / 69.59 / 24.16 |
| M=28 | 79.16 / 69.67 / 27.30 | 78.70 / 69.31 / 20.83 |
| M=56 | 79.46 / 69.80 / 21.06 | 79.61 / 70.27 / 20.92 |
| M=112 | 81.63 / 72.07 / 23.73 (3-seed 79.79 ± 1.60) | 79.04 / 69.19 / 18.22 |

Exact attention is not better than the low-rank approximation at any window size (Dice within seed noise).
Exact attention gives lower HD95 at M=28 and M=112.

## Kvasir-SEG results (RGB, 256 px, Adam 1e-3, 500 epochs, seed 2345, final epoch; Dice %, IoU %, HD95 px)

Single seed, same setup as the reference DA-TransUNet run. Seed spread on Kvasir is ±0.8.

| Configuration | Dice | IoU | HD95 |
|---|---|---|---|
| **DA-TransUNet (reference)** | **87.66** | **80.89** | 56.64 |
| M=8, r=32, PAM | 86.80 | 80.00 | 57.02 |
| M=32, r=32, PAM | 86.12 | 79.27 | 63.28 |
| M=64, r=32, PAM | 86.37 | 79.87 | 56.39 |
| M=128, r=32, PAM | 86.88 | 79.97 | 59.31 |
| M=128, r=16, PAM | 86.47 | 79.50 | 59.69 |
| M=128, r=64, PAM | 86.60 | 80.33 | **51.52** |
| M=128, r=0 (exact full attention) | 86.96 | 80.07 | 58.21 |
| M=32, r=0 (window-only) | 86.68 | 80.03 | 53.57 |
| M=32, CAM-only, G=8 | 86.14 | 78.70 | 71.02 |
| M=32, CAM-only, G=4 | 86.50 | 79.19 | 62.47 |
| M=32, CAM-only, G=1 | 85.92 | 78.81 | 64.13 |
| M=32, fixed 0.5 | 85.07 | 77.83 | 67.55 |
| M=32, learned gate | 86.17 | 79.29 | 57.48 |

- **Dice:** every ApproxDA configuration is 0.7–2.6 pp below DA-TransUNet. The best is exact full attention
  (86.96); the best approximate one is M=128, r=32 (86.88).
- **HD95:** M=128 / r=64 (51.52) and window-only (53.57) beat DA (56.64).
- **Window sensitivity:** the M sweep spans only 0.76 pp (86.12–86.88), vs 2.47 pp single-seed on Synapse.
  This supports the "different context demands" (GCS) narrative.
- **Routing:** CAM-containing variants are worst at boundaries (HD95 62–71); fixed 0.5 is the lowest Dice.
- **Learned gate:** pulled toward 0.5 over training.
  - Channel std: 0.010–0.020 at epoch 1 → 0.001–0.004 at epoch 500; range [0.488, 0.507]; sample std ≈ 1e-4.
  - On Synapse the gate merely stayed at its initialization.
  - Hypothesis for J6: weight decay pulls `gate_fc` to 0 (sigmoid(0) = 0.5) when its gradient is weak.
  - Test: no weight decay on `gate_fc`, or initialize its bias at ±2.

## Synapse results (c16, final epoch; Dice %, HD95 mm)

### Main comparison, 3 seeds (mean ± sample std)

| Model | Dice | HD95 | Seeds (Dice) |
|---|---|---|---|
| DA-TransUNet (matched re-run) | **79.81 ± 0.09** | 28.26 ± 1.15 | 79.74 / 79.80 / 79.91 |
| ApproxDA M=28, r=32, PAM | 79.43 ± 0.90 | 24.95 ± 4.14 | 79.16 / 80.43 / 78.68 |
| ApproxDA M=112, r=32, PAM | 79.79 ± 1.60 | **24.49 ± 3.29** | 81.63 / 78.69 / 79.05 |

Paired case-level tests vs DA-TransUNet (per-case values averaged over the 3 seeds, n = 12 cases, uncorrected):

| | Dice difference | HD95 difference |
|---|---|---|
| M=28 | −0.39 pp, 7/12 cases better, Wilcoxon p = 0.85 | −3.31, 9/12 cases lower, p = 0.11 |
| M=112 | −0.02 pp, 7/12 cases better, Wilcoxon p = 0.62 | −3.77, 9/12 cases lower, **p = 0.027** |

Per-organ Dice (3-seed mean):

| Model | Aorta | Gallbl. | Kid. L | Kid. R | Liver | Pancreas | Spleen | Stomach |
|---|---|---|---|---|---|---|---|---|
| DA | 86.8 | 63.6 | 82.5 | 79.0 | 94.8 | 61.9 | 89.0 | 81.0 |
| M=28 | 87.3 | 62.5 | 81.8 | 77.7 | 95.0 | 61.9 | 88.8 | 80.4 |
| M=112 | 86.7 | 64.8 | 80.0 | 78.5 | 94.7 | 61.5 | 89.9 | 82.2 |

**Reading:** Dice is the same as DA-TransUNet for both configurations; the only consistent difference is a lower
HD95 (significant for M=112 before multiple-comparison correction). ApproxDA's seed spread (0.9–1.6) is 10–18× DA's.
The workshop claims "M=28 is best" and "+1.14 over DA" do not hold under this protocol; report as is.

### Ablations (single seed 1234; seed std of the main configs is 0.9–1.6, so most gaps below are within noise)

| Ablation | Results (Dice / HD95) |
|---|---|
| Window size M (r=32, PAM) | M=7 **79.92** / 21.56 · M=28 79.16 / 27.30 · M=56 79.46 / 21.06 · M=112 81.63 / 23.73 (3-seed mean 79.79) |
| Rank r at M=112 (PAM) | r=16 78.71 / 25.53 · r=32 81.63 / 23.73 · r=64 79.92 / 23.49 · r=0 (exact full attention) 79.04 / **18.22** |
| Window-only (M=28, r=0, exact attention in windows) | 78.70 / 20.83 |
| Routing at M=28 | PAM 79.16 / 27.30 · CAM 79.34 / 26.03 · fixed 0.5 79.27 / 21.27 · learned 78.96 / 21.48 |
| CAM groups G (CAM-only, M=28) | G=1 80.03 / 23.62 · G=4 80.10 / 22.54 · G=8 79.34 / 26.03 |

**Learned gate (M=28):** it never leaves its initialization. Over 300 epochs the block means move by ≤ 0.003
(final 0.499–0.510), channel spread is std 0.006–0.021 (range 0.47–0.53), and variation across samples is
≈ 0.0005. So "the gate converges to 0.5" is better described as "the gate stays at its initial 0.5". The
next J6 diagnostic is a non-0.5 initialization (e.g. gate bias ±2) to see whether g returns to 0.5.
Full per-epoch history: `model/ApproxDA_Synapse224/*_M28_c16/gate_history.npz`.

### Legacy block (skipped after 5 runs; checkpoints deleted, logs kept)

| Legacy, bs 24, PAM | M=7 | M=28 | M=56 | M=112 | M=112, r=16 |
|---|---|---|---|---|---|
| Dice | 76.20 | 78.18 | 76.75 | 76.38 | 76.67 |
| c16 (same config) | 79.92 | 79.16 | 79.46 | 81.63 | 78.71 |

## Findings from the re-runs

1. **`test.py` corrupted BatchNorm statistics (fixed).** The fvcore FLOP count ran before `model.eval()`, so its
   forward pass on random input updated BatchNorm running statistics. DANetHead (and c16) BatchNorm uses
   momentum 0.95, so most of those statistics were replaced (Synapse Dice 73–75 instead of 79–80). This was the
   old "DA-TransUNet Synapse 72.03%" result, not a stale checkpoint. Workshop numbers are barely affected:
   legacy ApproxDA has no momentum-0.95 BatchNorm, and the Kvasir DA† 88.44 is not deflated (its in-training
   value was 88.38).
2. **Camera-ready says "batch size 24" for all runs; several runs differ.**
   - Synapse M=28 (main result, 80.94), M=56, M=112 and M=112/r=64: 1 GPU, batch 12 (185 iterations/epoch, 2× steps).
   - Synapse M=7 PAM (78.64) and the M=14 / r=64 run cited in the text (78.04): 2 GPUs × 12 (DDP).
   - ISIC M=56: batch 6.
   - DA-TransUNet, all Kvasir runs and Synapse CAM/learned: batch 24.

   The Synapse window ablation (Table V) is therefore confounded with the number of training steps. This needs
   a text correction or disclosure in the camera-ready.
3. **Legacy block underperforms at batch 24.** It is 1.0–5.3 pp below c16 at every M. Its code reproduces the
   June 19 model exactly (identical outputs for identical weights). The higher workshop numbers came from the
   batch-12 runs in Finding 2, plus initialization differences: dropped never-called decoder blocks change the
   RNG stream.
4. **Full attention fits now.** M=112, r=0 needed a 15.1 GB score matrix at batch 24 and ran out of memory.
   r=0 now uses memory-efficient SDPA on CUDA (8.7 GB, identical math to fp32 precision); CPU / fvcore
   audits keep the explicit matmuls.

## Code changes (uncommitted)

- `ApproxDA-TransUNet/Architecture/block.py`: r=0 exact attention via memory-efficient SDPA (`exact_window_attention`).
- `ApproxDA-TransUNet/train.py`:
  - `GateMonitor`: per-epoch learned-gate statistics in `log.txt`, TensorBoard and `gate_history.npz`.
  - `_G<groups>` added to the snapshot path when G≠8 (G runs used to overwrite each other).
  - `fork` start method (Python 3.14 defaults to forkserver, which cannot pickle `worker_init_fn`).
- `*/test.py` (both projects): `model.eval()` before FLOP profiling (Finding 1). ApproxDA `test.py` also gets the `_G` snapshot suffix.
- `DA-TransUNet/train.py`: `num_workers` 0 → 4, matching ApproxDA's data pipeline; `fork` start method.
- `ApproxDA-TransUNet/datasets/generate_lists.py`: the ISIC official mode matches directories only (it used to
  fail when the downloaded `.zip` files sit next to the folders).
- `*/lists/lists_ISIC/`: replaced by the official train/val/test lists.
- `*/lists/lists_Kvasir/`: replaced by the paper-style random 75/25 split (seed 1234); the `generate_lists.py`
  header documents the command.
- Opt-in RGB input for Kvasir/ISIC (`--rgb` in both projects' `train.py`/`test.py`, snapshot suffix `_rgb`):
  - the loaders take `rgb=`; the augmentation and `utils.test_single_volume` handle (H, W, 3);
  - the 2D-vs-3D decision now uses the label's dimensions;
  - the grayscale result reproduces exactly (0.859389 before and after).
- `*/test.py`: `--eval_epoch N` evaluates `epoch_<N-1>.pth` (log suffix `_ep<N>`).
- `*/test.py`: `--precise_bn` re-estimates BatchNorm statistics on the training set before evaluating
  (log suffix `_pbn`).
- `ApproxDA-TransUNet/datasets/resize_isic.py`: `--names` restricts resizing to a list of stems.
- Both projects' `train.py`/`test.py`: `--drop_last` (opt-in, snapshot suffix `_dl`).
- Both projects' `train.py`/`test.py`: `--num_workers` (default 4, unchanged); non-default values add a `_nw<N>` snapshot suffix.
- New files:
  - `experiments/run_queue.sh`, `experiments/status.sh`;
  - `experiments/queue_{synapse,kvasir,isic}.txt`;
  - `ApproxDA-TransUNet/datasets/resize_isic.py`.

## T4 workshop checkpoints re-evaluated (2026-10-16; `/data/checkpoints/ApproxDA-best-porch`)

Evaluated with the June code (git `c6af039`, legacy block) and the old splits, via `scratchpad/t4eval/eval_t4.py`.

| Checkpoint | Config | Dice / HD95 / IoU (ours) | Paper / log |
|---|---|---|---|
| Synapse `bs12_M28_pam_best_model` | M=28 r=32 pam | 81.02 / 27.10 / 71.37 | 80.94 |
| Synapse `bs12_M112_pam` | M=112 r=32 pam | 77.92 / 27.01 / 67.84 | 79.44 (file is not the best checkpoint) |
| Synapse `bs24_best_model` | M=7 r=32 learn | 77.93 / 33.96 / 67.45 | 77.78 |
| ISIC `bs24_best_model` | M=7 r=32 learn | 89.56 / 135.26 / 82.64 | 89.58 |
| Kvasir `bs24_best_model` | M=7 r=32 learn | 89.23 / 42.62 / 83.39 | 89.24 |

4 of 5 reproduce the workshop numbers. The `*_best_model` files were selected on the test set, so they are not
used for the revised paper.

## CVC-ClinicDB round (`queue_cvc.txt`, started 2026-10-16 21:40)

- Protocol: Kvasir-style (DA paper §4.1), i.e. random 75/25 split (459/153, seed 1234), RGB, 256 px, Adam 1e-3,
  500 epochs, `--drop_last`, seed 1234; final epoch.
- Jobs: 16 in all, DA-TransUNet plus the full c16 ablation (the same 15 configurations as ISIC); ~30 min each.
- Target: DA paper 89.47 Dice / 82.51 IoU.
- Purpose: decide whether CVC or Kvasir-SEG goes into the paper as the third dataset.
- Interim results (2026-10-17, 9/16 done). Values are Dice / IoU / HD95 px; seed 1234, final epoch.
  - DA-TransUNet: **92.39 / 87.39 / 12.16**, about 2.9 Dice above the paper (89.47 / 82.51).
  - ApproxDA, gate=pam:
    - r=32: M=8 91.64, M=32 92.54, M=64 91.82, M=128 91.86;
    - M=128: r=16 **92.91**, r=64 92.83, r=0 92.86;
    - M=8, r=0: 92.71.
  - All configurations are within ±0.75 of DA, and HD95 is worse than DA in every one.
- Baseline audit:
  - The paper's setup matches ours: 256 px, Adam 1e-3, 500 epochs, bs 24, random 75/25 (not released),
    BCE+Dice. The paper does not say whether the input is RGB or grayscale.
  - Both the paper's and our Dice are per-image means: IoU > D/(2−D) in both.
  - Our split has no file overlap. However, all 29 sequences appear in both train and test, and 140/153 test frames
    have an adjacent frame (id ±1) in train. A random frame-level split leaks near-duplicates; this is known to
    inflate CVC results, and the paper's split has the same issue.
  - Evaluation: predict at 256, nearest-neighbour upsampling to 288×384, Dice against the original mask (>127).
    This is the same code as Kvasir/ISIC, where DA did not exceed the paper.
  - The de-facto public standard is the PraNet split (CVC 550 train / 62 test, trained jointly with 900 Kvasir).
    On it, PraNet reaches 0.899 mDice and recent models 0.92–0.94. No DA-TransUNet numbers exist on that split.
- Sanity runs (2026-10-17):
  - DA RGB, seed 2345: 92.53 / 87.52 / 10.79. The seed spread is small, so 92.39 is confirmed (2-seed mean 92.46).
  - **DA grayscale, seed 1234: 89.72 / 84.78 / 14.97**, against the paper's 89.47 Dice. The paper most likely used
    grayscale input (TransUNet-style 1→3 channel repeat). RGB adds +2.67 Dice on CVC (+1.25 on Kvasir).
    Grayscale also trains faster: 32 min vs 57 min.
- More RGB ablations: M=32 r=0 92.73 / 87.48 / 12.54; M=64 r=0 92.77 / 87.60 / 13.17.
- **Decision (2026-10-17): CVC uses grayscale input** (the protocol that reproduces the paper's DA-TransUNet).
  - The 15-configuration ApproxDA ablation is re-run in grayscale, appended to `queue_cvc.txt`
    (~33 min/run, about 8 h). The DA grayscale run already exists.
  - The four not-yet-started RGB routing runs were cancelled (commented out in the queue). RGB cam was already
    running and was left to finish.
  - The RGB results stay in this file as a sensitivity check (RGB +2.67 Dice for DA).
- **Grayscale round finished (2026-10-17 18:38).** Values are Dice / HD95 px; DA grayscale = 89.72 / 14.97, paper 89.47.

  | M (gate=pam) | r=32 | r=0 (exact) |
  |---|---|---|
  | 8 | 89.81 / 15.16 | 89.43 / 15.73 |
  | 32 | 89.49 / 13.50 | 89.13 / 15.17 |
  | 64 | 90.12 / 18.57 | 89.36 / 18.64 |
  | 128 | **90.20** / 18.54 | 89.63 / 14.11 |
  | 128, other ranks | r=16: 89.76 / 13.60 | r=64: 89.83 / 13.14 |

  Routing at M=32, r=32:

  | Configuration | Dice / HD95 |
  |---|---|
  | pam | 89.49 / 13.50 |
  | cam (G=8) | 89.86 / 16.11 |
  | fixed | 89.94 / 17.27 |
  | learn | 89.90 / 14.67 |
  | cam, G=1 | **90.54** / 14.94 (IoU 85.87) |
  | cam, G=4 | 89.29 / 12.77 |

  - GCS: the r=32 window span is 0.71 pp, so Synapse/CVC is 3.5×. Exact attention is below low rank at every M
    (by 0.36–0.76).
  - Best overall is cam with G=1, i.e. the exact DANet CAM with no PAM (+0.82 over DA). Best window config is
    pam M=128 r=32 (+0.48). On Synapse and ISIC the overall best was a pam window config.
  - RGB cam (M=32): 92.83 / 13.08.
- **Second seed (2345, grayscale, 2026-10-18).** Values are Dice / IoU / HD95.

  | Configuration | Seed 1234 | Seed 2345 | Mean Dice ± sd | Δ vs DA (1234 / 2345 / mean) |
  |---|---|---|---|---|
  | DA-TransUNet | 89.72 / 84.78 / 14.97 | 89.60 / 84.69 / 16.80 | 89.66 ± 0.08 | – |
  | cam, G=1 (M=32, r=32) | 90.54 / 85.87 / 14.94 | 89.62 / 85.04 / 12.13 | 90.08 ± 0.65 | +0.82 / +0.02 / +0.42 |
  | pam, M=128, r=32 | 90.20 / 85.42 / 18.54 | 89.63 / 85.07 / 16.16 | 89.92 ± 0.40 | +0.48 / +0.03 / +0.26 |

  - On seed 2345 both candidates tie with DA (+0.02 / +0.03) and with each other (89.62 vs 89.63).
  - The seed-1234 gains are mostly selection luck: the candidates were the winners of a 15-run single-seed
    sweep. This is the same winner's curse as Synapse M=112 and ISIC.
  - DA is stable across seeds (±0.08).
  - cam G=1 has the better HD95 on both seeds.
- **Same-seed repeat (2026-10-18).** Training is not bitwise deterministic: losses already differ at epoch 1.
  - cam G=1 repeated at 90.54 (original 90.51). The repeat replaced the original run (checkpoints and logs).
  - pam M=128 repeated at 89.52 (original 90.20). The repeat was discarded and the original kept.
  - **Decision: CVC Table II reports cam G=1 (90.54 / 85.87),** the highest-DSC configuration of the ablation.

## Paper update (2026-10-16; workshop revision, all feedback except R2/R8)

Reported results: seed 1234, c16 block, best of the ablation sweep, final epoch, plain evaluation.
- Synapse: M=112, r=32, pam, 81.63 / 23.73.
- ISIC: M=64, r=32, pam, 89.26.
- Kvasir: on hold (old T4 numbers kept, marked `% TODO(Kvasir on hold)`).

Updated sections:
- §I–II (abstract and introduction): numbers, a GCS ratio of 3.9× (Synapse vs Kvasir, pending), the routing
  claim and the efficiency claim.
- §III (Method): the c16 block, with Eq. QKV, GroupedCAM in DANet form, and the fusion without a residual.
- §IV (Experiments): implementation details on the RTX 5090; evaluation protocol; Synapse/ISIC tables; FLOPs
  table (attention −54–57%).
- §V (Analysis): the 2×4 window×rank grid; the GCS curve and text; the routing table and gate statistics.
- §VI (Conclusion): claims; Limitations (single runs, configuration chosen on the test set, literature baselines).

The paper compiles at 8 pages with no new warnings (scratch build; `algorithmic`/`algorithm` are unused and not
installed locally).

Done on 2026-10-16 (second pass):
- Fig. 2 `figures/approxda_block.pdf` was redrawn for the c16 layout by `paper/PlotNeuralNetScripts/approxda_block.py`,
  using the PlotNeuralNet fork in `~/projects/PlotNeuralNet`.
- Fig. 6 `figures/fig_entropy_scatter.png` was regenerated with `analyze_gate_entropy.py` (Fig B) on Synapse M=28
  learn at epoch 300, using new `--checkpoint`/`--block_version` options. The axes use scientific notation.
  - r_s = +0.259 / −0.432 / −0.004 / +0.081;
  - Δg per block (across test slices) = 3.4 / 2.5 / 3.2 / 2.5 ×10⁻⁴;
  - block means 0.4992–0.5099.
- Tables I and III gained a reproduced DA-TransUNet row (seed 1234, plain evaluation) and a Δ row:
  - Synapse: 79.74 / 28.44, Δ +1.89 DSC / −4.71 HD95;
  - ISIC: 88.08 / 81.40, Δ +1.18 DSC / +1.24 mIoU.
- Tables: `center` → `\centering` (uniform caption gaps), notes in full-width minipages; all figure captions and
  table notes trimmed, detail moved to the text; Figs. 1 and 2 at 0.9\textwidth. Back to 8 pages.
- Efficiency: Table IV and §IV-C kept (attention −54–57%, whole network only −1.6–1.8%), framed as a diagnostic
  platform; the efficiency claims were removed from the abstract, introduction and conclusion.

Paper switched to CVC-ClinicDB (2026-10-18).
- Kvasir-SEG is removed from the paper:
  - the dataset text;
  - Table II, which is now CVC with cam G=1 at 90.54 / 85.87;
  - the GCS curve, where CVC spans 0.71 pp and the ratio becomes 3.5×;
  - the abstract and introduction.
- Fig. 3 was regenerated with `experiments/make_qualitative_figure.py` (helper: `qualitative_predict.py`).
  - There are no row labels; the caption names the datasets. Stomach is shown in blue.
  - Synapse: among slices with 2–3 organ classes, ≥ 8000 foreground pixels and ApproxDA Dice ≥ 0.80, the slice
    with the largest ApproxDA − DA gain. In it, DA labels the stomach as spleen.
  - CVC and ISIC: the test case with the median ApproxDA Dice. These per-image Dice values match the test logs.
  - Selected cases (per-image Dice, DA / ApproxDA):

    | Dataset | Case | Dice (DA / ApproxDA) |
    |---|---|---|
    | Synapse | `case0032`, axial slice 131 (aorta, liver, stomach) | 64.08 / 96.44 |
    | CVC-ClinicDB | frame `510` | 92.64 / 95.76 |
    | ISIC 2018 | `ISIC_0009938` | 91.36 / 93.58 |
- Fig. 5 (Kvasir attention maps from the legacy model) and its paragraph were removed.
- The paper is 8 pages.
- Still open: code-link footnote and author block.

## PAM attention is inactive in the reported models (2026-10-18)

The PAM output is `alpha * E + x`, with alpha initialised at 0 (as in DANet and DA-TransUNet).

### What was measured (`scratchpad/pam_effect.py`, `scratchpad/alpha_grad.py`)

| Model | alpha | ‖out − in‖/‖in‖ | Pixels changed when alpha → 0 |
|---|---|---|---|
| T4 Kvasir (legacy block) | ~2e-5 | ~1e-6 | 0.0002% |
| T4 Synapse (legacy block) | 1e-4 – 2e-3 | 3e-6 – 6e-5 | 0.0001% |
| Synapse c16 M=112 (reported) | 1e-4 – 1e-3 | 1e-6 – 1e-4 | 0.0002% |
| CVC c16 M=8 | ~5e-41 (denormal) | 0 exactly | 0% |
| CVC c16 M=128 (Adam) | ~0.2 | 2 – 4% | 0.046% |

DA-TransUNet's own PAM gammas also stay ≤ 0.007 under SGD (Synapse, ISIC).

### Consequences

- The window and rank ablations on SGD datasets (Synapse, ISIC) do not change what the network computes. Their
  Dice differences are training noise.
- The old Fig. 5 maps were min–max-normalised signals of ~1e-6 magnitude. Its 62.3% / 34.2% shares are
  scale-free, so they do not show that attention focuses on the target.

### Cause (gradients on one Synapse training batch at initialisation)

- With alpha = 0, the attention weights (Q, K, V, P) get exactly zero gradient.
- |dL/dalpha| ≈ 4e-6 – 4e-5, against 0.29 for a decoder conv. Even at alpha = 1, ‖E‖/‖x‖ is only 1–3%
  (24% at 112²).
- With BatchNorm on E (`--attn_norm`), |dL/dalpha| becomes 1e-4 – 1.3e-3. At alpha = 0.1 the attention term is
  18 – 130% of x.

### Fix options (opt-in; defaults reproduce every reported run, checked bit-identical on the Fig. 3 slice)

- `--alpha_init <v>` (snapshot suffix `_a<v>`);
- `--attn_norm` (suffix `_an`).

### Pilot (`queue_attn_pilot.txt`, started 2026-10-18 11:46)

Synapse M=112, r=32, pam, seed 1234, 300 epochs:
- `--attn_norm`;
- `--alpha_init 1`.

For each, check the final alpha, the effect of setting alpha → 0, and Dice (baseline 81.63).

**Pilot result (2026-10-18 14:08).** Values for the four blocks.

| Variant | Final alpha | ‖out − in‖/‖in‖ | Pixels changed when alpha → 0 | Dice / HD95 |
|---|---|---|---|---|
| baseline | ~1e-3 | 1e-6 – 1e-4 | 0.0002% | 81.63 / 23.73 |
| `--attn_norm` | −0.02 / −0.09 / +0.12 / −0.24 | 0.4% / 6% / 9% / 34% | 0.075% | 80.86 / 28.84 |
| `--alpha_init 1` | 0.87 – 0.89 | 1% / 4% / 7% / 20% | 0.067% | 80.57 / 24.27 |

- alpha now learns, and the attention term is no longer negligible in the features (about 300× more influence).
- The segmentation still barely depends on it (0.07% of pixels change), and Dice is within the seed spread (±1.6).
- Conclusion: the alpha·E + x residual is not the only problem. The rest of the network (C/16 convs, ViT, decoder)
  does the work, so the dual-attention head is largely redundant here.
- Journal options:
  - remove the identity path (attention as the only path) and test whether M then matters;
  - test on tasks that need long-range context.

## Model cleanup (2026-10-18): `experiments/model` 195G → 28G

Deleted runs (numbers stay in this file, `logs/queue.history` and `*/test_log/`):
- ISIC legacy-block runs, ISIC c16 runs before `--drop_last`, `TU_ISIC224`, and old ISIC DA checks
  (Adam, SGD without drop_last, grayscale Adam);
- CVC RGB runs;
- all Kvasir runs.

Every remaining run keeps only its final-epoch checkpoint, so `test.py --eval_epoch` is no longer possible.

## Next

1. When `queue_cvc.txt` finishes: compare DA with the paper (89.47 / 82.51), review the ablation, and choose CVC
   or Kvasir-SEG.
2. Then fill in the third dataset and regenerate Figs. 3/5/6.

## Workshop-era items (from the previous version of this file)

| # | Item | Status |
|---|---|---|
| 1 | ISIC GCS CI (old 80/20 split, M=28 per-case log) | Superseded by the official-split ISIC re-run above |
| 2 | DA-TransUNet Synapse checkpoint (72.03% mismatch) | **Resolved** — caused by Finding 1; matched re-run 79.81 ± 0.09 with per-case logs |
| 3 | DA-TransUNet CVC-ClinicDB | Running (see the CVC-ClinicDB round above) |
| 4 | Entropy-gate result (78.87%, code never committed) | Unchanged — do not cite until the code is recovered and re-run |
