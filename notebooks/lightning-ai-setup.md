# Lightning AI Studio — Run Guide

All commands run in the **Studio terminal** (not Jupyter). Use `tmux` so sessions survive browser disconnects.

---

## One-Time Setup

### 1. Open a persistent tmux session

```bash
tmux new -s adada
# Ctrl+B, D  to detach
# tmux attach -t adada  to reattach later
```

### 2. Clone the repository

```bash
cd /teamspace/studios/this_studio
git clone https://github.com/jiaweizhong/AdaDA-TransUNet.git
```

To pull latest changes in future sessions:

```bash
git -C /teamspace/studios/this_studio/AdaDA-TransUNet pull
```

### 3. Install dependencies

```bash
pip install timm einops ml-collections medpy SimpleITK tensorboardX thop h5py scipy fvcore
```

### 4. Download data

#### Synapse — via Kaggle API

Upload your regenerated `kaggle.json` to `/teamspace/studios/this_studio/` via the Studio file browser, then:

```bash
pip install -q kaggle
mkdir -p ~/.kaggle
cp /teamspace/studios/this_studio/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

BASE=/teamspace/studios/this_studio
mkdir -p $BASE/data/raw_synapse
kaggle datasets download -d dogcdt/synapse -p $BASE/data/raw_synapse --unzip
mkdir -p $BASE/data/Synapse
mv $BASE/data/raw_synapse/Synapse/train_npz $BASE/data/Synapse/train_npz
mv $BASE/data/raw_synapse/Synapse/test_vol_h5 $BASE/data/Synapse/test_vol_h5
rm -rf $BASE/data/raw_synapse
```

#### ViT pretrained weights — Google public bucket (no auth needed)

```bash
mkdir -p /teamspace/studios/this_studio/model/vit_checkpoint/imagenet21k
wget -q --show-progress \
  -O /teamspace/studios/this_studio/model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz \
  https://storage.googleapis.com/vit_models/imagenet21k/R50+ViT-B_16.npz
```

### 5. Create symlinks (data path resolution)

`train.py` and `test.py` resolve paths relative to their own directory (`../data/`, `../model/`). `experiments/data/` is a **directory** (not a single symlink) containing per-dataset symlinks. Create them individually:

```bash
REPO=/teamspace/studios/this_studio/AdaDA-TransUNet/experiments
BASE=/teamspace/studios/this_studio

mkdir -p $REPO/data $REPO/model

ln -sfn $BASE/data/Synapse    $REPO/data/Synapse
ln -sfn $BASE/data/Kvasir-SEG $REPO/data/Kvasir-SEG
ln -sfn $BASE/data/ISIC2018   $REPO/data/ISIC2018
ln -sfn $BASE/model            $REPO/model

# verify
ls -la $REPO/data/Synapse/
ls -la $REPO/data/Kvasir-SEG/
ls -la $REPO/data/ISIC2018/
ls -la $REPO/model/vit_checkpoint/imagenet21k/
```

Expected: `R50+ViT-B_16.npz` (with `+` intact — Lightning AI does not strip it).

> **Do NOT run** `ln -sfn $BASE/data $REPO/data` — this creates a nested `$REPO/data/data` symlink and breaks relative path resolution.

### 6. Fix datasets/ package shadowing

```bash
REPO=/teamspace/studios/this_studio/AdaDA-TransUNet/experiments
touch $REPO/DA-TransUNet/datasets/__init__.py
touch $REPO/Ada-DA-TransUNet/datasets/__init__.py
```

---

## Running Experiments

> **CRITICAL — use tmux for ALL training runs.** `nohup ... &` does NOT protect `torchrun` from SIGHUP: when the Lightning AI browser disconnects, `torchrun`'s own signal handler catches the hangup and kills all workers. The only reliable fix is to run inside a tmux session and detach with `Ctrl+B, D`.

### Session lifecycle (do this every time)

```bash
# Create session if it doesn't exist, or attach to it if it does
tmux new-session -A -s adada

# To detach (training keeps running):  Ctrl+B, then D
# To reattach later:                   tmux attach -t adada
# To list sessions:                    tmux ls
```

Check GPU availability before starting:

```bash
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
```

> All commands below must be run **inside the tmux session**. Run them in the foreground with `| tee logfile.log` — do NOT add `&` at the end.

---

### DA-TransUNet — Synapse (single T4, baseline)

```bash
# Inside tmux:
cd /teamspace/studios/this_studio/AdaDA-TransUNet/experiments/DA-TransUNet

python -u train.py \
  --dataset Synapse --vit_name R50-ViT-B_16 \
  --max_epochs 300 --batch_size 24 \
  --base_lr 0.01 --n_skip 3 --img_size 224 \
  --seed 1234 --val_interval 15 \
2>&1 | tee run_da.log && \
python -u test.py \
  --dataset Synapse --vit_name R50-ViT-B_16 \
  --num_classes 9 --max_epochs 300 --img_size 224 --is_savenii \
2>&1 | tee -a run_da.log

# Detach: Ctrl+B, D
```

---

### AdaDA-TransUNet — Synapse, single T4 (apples-to-apples efficiency row)

```bash
# Inside tmux:
cd /teamspace/studios/this_studio/AdaDA-TransUNet/experiments/Ada-DA-TransUNet

python -u train.py \
  --dataset Synapse --vit_name R50-ViT-B_16 \
  --max_epochs 300 --batch_size 24 \
  --base_lr 0.01 --n_skip 3 --img_size 224 \
  --window_size 7 --rank 32 --groups 8 \
  --seed 1234 --val_interval 15 \
2>&1 | tee run_adada_1gpu.log && \
python -u test.py \
  --dataset Synapse --vit_name R50-ViT-B_16 \
  --num_classes 9 --max_epochs 300 --img_size 224 \
  --window_size 7 --rank 32 --groups 8 --is_savenii \
2>&1 | tee -a run_adada_1gpu.log

# Detach: Ctrl+B, D
```

---

### AdaDA-TransUNet — Synapse, 2×T4 DDP (multi-GPU efficiency row)

```bash
# Inside tmux:
cd /teamspace/studios/this_studio/AdaDA-TransUNet/experiments/Ada-DA-TransUNet

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train.py \
  --dataset Synapse --vit_name R50-ViT-B_16 \
  --max_epochs 300 --batch_size 12 \
  --base_lr 0.01 --n_skip 3 --img_size 224 \
  --window_size 7 --rank 32 --groups 8 \
  --seed 1234 --val_interval 15 \
2>&1 | tee run_adada_2gpu.log && \
python -u test.py \
  --dataset Synapse --vit_name R50-ViT-B_16 \
  --num_classes 9 --max_epochs 300 --batch_size 12 --img_size 224 \
  --window_size 7 --rank 32 --groups 8 --is_savenii \
2>&1 | tee -a run_adada_2gpu.log

# Detach: Ctrl+B, D
```

> Per-GPU batch=12, total=24 — same as single-GPU baseline, LR unchanged.

---

## Phase 1 — Entropy Verification

Run **after** `best_model.pth` exists from either single-GPU or 2-GPU run:

```bash
cd /teamspace/studios/this_studio/AdaDA-TransUNet/experiments/Ada-DA-TransUNet

python analyze_gate_entropy.py \
  --vit_name R50-ViT-B_16 --n_skip 3 \
  --max_epochs 300 --batch_size 24 \
  --window_size 7 --rank 32 --groups 8
```

Output: Spearman r (entropy) and r (variance) per AdaDA block + `gate_entropy_scatter.png`.

See `AdaDA-TransUNet.md §10–11` for the 4-case decision table and backup plans (B1/B2/B3).

---

## Phase 2 Ablation Runs (Week 2) — ✅ COMPLETE

> **`--disable_gate` is removed.** The flag was replaced by `--gate_mode {learn,fixed,pam,cam}`.
> All gate ablation runs used `--gate_mode` and are now complete.

### Results summary

| Config | Test DSC | Test HD95 | Status |
|--------|---------|-----------|--------|
| `--gate_mode learn --window_size 7 --rank 32` | 77.93% | 33.96mm | ✅ Done (T4×1, 300ep) |
| `--gate_mode pam --window_size 7 --rank 32` | 78.64% | 31.09mm | ✅ Done (T4×2, 300ep) |
| `--gate_mode cam --window_size 7 --rank 32` | 78.26% | 30.59mm | ✅ Done (T4×1, 300ep) |
| `--gate_mode learn --window_size 14 --rank 64` | 78.04% | 29.09mm | ✅ Done (T4×2, 300ep) |

**Finding:** Gate collapsed (g≈0.5, Δg=0.0000) in all `learn` runs. gate=learn is strictly worse than PAM-only. See `EXPERIMENT_PLAN.md §Ablation` for full analysis.

### Global PAM experiment (Phase B, after Kvasir/ISIC)

```bash
# Inside tmux:
cd /teamspace/studios/this_studio/AdaDA-TransUNet/experiments/Ada-DA-TransUNet

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train.py --dataset Synapse --vit_name R50-ViT-B_16 --max_epochs 300 --batch_size 12 --base_lr 0.01 --n_skip 3 --img_size 224 --gate_mode pam --window_size 112 --rank 64 --groups 8 --seed 1234 --val_interval 15 2>&1 | tee run_adada_global_pam.log

# After training finishes:
python -u test.py --dataset Synapse --vit_name R50-ViT-B_16 --num_classes 9 --max_epochs 300 --batch_size 12 --img_size 224 --window_size 112 --rank 64 --groups 8 --gate_mode pam 2>&1 | tee -a run_adada_global_pam.log

# Detach: Ctrl+B, D
```

---

## Monitoring

```bash
# Is anything still running?
ps aux | grep -E 'train\.py|torchrun' | grep -v grep

# Tail a specific log
tail -f /teamspace/studios/this_studio/AdaDA-TransUNet/experiments/Ada-DA-TransUNet/run_adada_2gpu.log

# Quick status across all logs
BASE=/teamspace/studios/this_studio/AdaDA-TransUNet/experiments
for f in $BASE/DA-TransUNet/run_da.log \
          $BASE/DA-TransUNet/run_da_kvasir.log \
          $BASE/DA-TransUNet/run_da_isic.log \
          $BASE/Ada-DA-TransUNet/run_adada_1gpu.log \
          $BASE/Ada-DA-TransUNet/run_adada_2gpu.log \
          $BASE/Ada-DA-TransUNet/run_adada_kvasir.log \
          $BASE/Ada-DA-TransUNet/run_adada_isic.log \
          $BASE/Ada-DA-TransUNet/run_adada_global_pam.log; do
  echo "=== $(basename $f) ===";
  tail -3 "$f" 2>/dev/null || echo "(not started)";
done

# Final DSC numbers
grep "Testing performance" $BASE/DA-TransUNet/run_da*.log 2>/dev/null
grep "Testing performance" $BASE/Ada-DA-TransUNet/run_*.log 2>/dev/null
```

---

## Checkpoint Locations

| Artifact | Path |
|---|---|
| AdaDA best checkpoint | `experiments/model/AdaDA_Synapse224/AdaDA/AdaDA_pretrain_R50-ViT-B_16_skip3_epo300_bs*/best_model.pth` |
| DA-TransUNet best checkpoint | `experiments/model/TU_Synapse224/TU/TU_pretrain_R50-ViT-B_16_skip3_epo300_bs24_224/best_model.pth` |
| Test logs | `experiments/{DA-TransUNet,Ada-DA-TransUNet}/test_log/` |
| Gate entropy scatter | `experiments/Ada-DA-TransUNet/gate_entropy_scatter.png` |

To pull a file to your local machine:

```bash
# From your LOCAL terminal:
scp <studio-ssh>:/teamspace/studios/this_studio/AdaDA-TransUNet/experiments/Ada-DA-TransUNet/gate_entropy_scatter.png .
```

(Find the SSH address in Lightning AI → Studio settings → SSH.)

---

## Kvasir-SEG and ISIC 2018

Dataset code is ready. Follow these steps before running training.

### 1. Download datasets (Kaggle API)

Make sure `~/.kaggle/kaggle.json` is installed (see One-Time Setup §4 above).

```bash
BASE=/teamspace/studios/this_studio

# --- Kvasir-SEG (by original paper authors, 151MB, 1000 images) ---
mkdir -p $BASE/data/raw_kvasir
kaggle datasets download -d debeshjha1/kvasirseg -p $BASE/data/raw_kvasir --unzip

# Zip unpacks as Kvasir-SEG/Kvasir-SEG/{images,masks}/
mkdir -p $BASE/data/Kvasir-SEG
mv $BASE/data/raw_kvasir/Kvasir-SEG/Kvasir-SEG/images $BASE/data/Kvasir-SEG/images
mv $BASE/data/raw_kvasir/Kvasir-SEG/Kvasir-SEG/masks  $BASE/data/Kvasir-SEG/masks
rm -rf $BASE/data/raw_kvasir

# Verify (should see ~1000 JPEG files each)
ls $BASE/data/Kvasir-SEG/images/ | wc -l
ls $BASE/data/Kvasir-SEG/masks/  | wc -l

# --- ISIC 2018 Task 1 (lesion segmentation) ---
# Slug: tschandl/isic2018-challenge-task1-data-segmentation (13.8GB, 8319 downloads)
# Contains train/val/test splits; only training set has ground truth masks.
# We use training set only (2594 images) and do our own 80/20 split.

mkdir -p $BASE/data/raw_isic
kaggle datasets download -d tschandl/isic2018-challenge-task1-data-segmentation \
  -p $BASE/data/raw_isic --unzip

# Folder structure after unzip:
#   ISIC2018_Task1-2_Training_Input/          <- images (.jpg)
#   ISIC2018_Task1_Training_GroundTruth/      <- masks (*_segmentation.png)
#   ISIC2018_Task1-2_Validation_Input/        <- no masks, skip
#   ISIC2018_Task1-2_Test_Input/              <- no masks, skip

mkdir -p $BASE/data/ISIC2018/images $BASE/data/ISIC2018/masks
mv $BASE/data/raw_isic/ISIC2018_Task1-2_Training_Input/*.jpg \
   $BASE/data/ISIC2018/images/
mv $BASE/data/raw_isic/ISIC2018_Task1_Training_GroundTruth/*_segmentation.png \
   $BASE/data/ISIC2018/masks/
rm -rf $BASE/data/raw_isic

# Verify
ls $BASE/data/ISIC2018/images/ | wc -l   # expect 2594
ls $BASE/data/ISIC2018/masks/  | wc -l   # expect 2594
```

> **If the Kaggle slug or folder names differ**, check with `ls $BASE/data/raw_*/` after download and adjust the `mv` paths.

### 2. Generate list files

Run once after downloading (creates `train.txt` and `test.txt`):

```bash
cd /teamspace/studios/this_studio/AdaDA-TransUNet/experiments/Ada-DA-TransUNet

# Use absolute paths — relative ../data/ fails because the symlink is in experiments/data/,
# not in the script's CWD (experiments/Ada-DA-TransUNet/).
python datasets/generate_lists.py --dataset Kvasir \
  --data_dir /teamspace/studios/this_studio/data/Kvasir-SEG
# Output: lists/lists_Kvasir/train.txt (800) + test.txt (200)  [seed=42, 80/20]

python datasets/generate_lists.py --dataset ISIC \
  --data_dir /teamspace/studios/this_studio/data/ISIC2018
# Output: lists/lists_ISIC/train.txt (~2075) + test.txt (~519)  [seed=42, 80/20]
```

Copy the same list files to DA-TransUNet so both models use identical splits:

```bash
REPO=/teamspace/studios/this_studio/AdaDA-TransUNet/experiments
cp $REPO/Ada-DA-TransUNet/lists/lists_Kvasir/train.txt $REPO/DA-TransUNet/lists/lists_Kvasir/train.txt
cp $REPO/Ada-DA-TransUNet/lists/lists_Kvasir/test.txt  $REPO/DA-TransUNet/lists/lists_Kvasir/test.txt
cp $REPO/Ada-DA-TransUNet/lists/lists_ISIC/train.txt   $REPO/DA-TransUNet/lists/lists_ISIC/train.txt
cp $REPO/Ada-DA-TransUNet/lists/lists_ISIC/test.txt    $REPO/DA-TransUNet/lists/lists_ISIC/test.txt
```

### 3. Training + inference commands

#### DA-TransUNet — Kvasir (single T4, ~3h)

```bash
cd /teamspace/studios/this_studio/AdaDA-TransUNet/experiments/DA-TransUNet

python -u train.py \
  --dataset Kvasir --vit_name R50-ViT-B_16 \
  --max_epochs 300 --batch_size 24 \
  --base_lr 0.01 --n_skip 3 --img_size 224 \
  --seed 1234 --val_interval 15 \
2>&1 | tee run_da_kvasir.log && \
python -u test.py \
  --dataset Kvasir --vit_name R50-ViT-B_16 \
  --num_classes 2 --max_epochs 300 --batch_size 24 --img_size 224 \
2>&1 | tee -a run_da_kvasir.log
```

#### DA-TransUNet — ISIC (single T4, ~4h)

```bash
cd /teamspace/studios/this_studio/AdaDA-TransUNet/experiments/DA-TransUNet

python -u train.py \
  --dataset ISIC --vit_name R50-ViT-B_16 \
  --max_epochs 300 --batch_size 24 \
  --base_lr 0.01 --n_skip 3 --img_size 224 \
  --seed 1234 --val_interval 15 \
2>&1 | tee run_da_isic.log && \
python -u test.py \
  --dataset ISIC --vit_name R50-ViT-B_16 \
  --num_classes 2 --max_epochs 300 --batch_size 24 --img_size 224 \
2>&1 | tee -a run_da_isic.log
```

#### AdaDA — Kvasir, gate=pam, 2×T4 (~2h)

```bash
cd /teamspace/studios/this_studio/AdaDA-TransUNet/experiments/Ada-DA-TransUNet

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train.py \
  --dataset Kvasir --vit_name R50-ViT-B_16 \
  --max_epochs 300 --batch_size 12 \
  --base_lr 0.01 --n_skip 3 --img_size 224 \
  --window_size 7 --rank 32 --groups 8 --gate_mode pam \
  --seed 1234 --val_interval 15 \
2>&1 | tee run_adada_kvasir.log && \
python -u test.py \
  --dataset Kvasir --vit_name R50-ViT-B_16 \
  --num_classes 2 --max_epochs 300 --batch_size 12 --img_size 224 \
  --window_size 7 --rank 32 --groups 8 --gate_mode pam \
2>&1 | tee -a run_adada_kvasir.log
```

#### AdaDA — ISIC, gate=pam, 2×T4 (~2.5h)

```bash
cd /teamspace/studios/this_studio/AdaDA-TransUNet/experiments/Ada-DA-TransUNet

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train.py \
  --dataset ISIC --vit_name R50-ViT-B_16 \
  --max_epochs 300 --batch_size 12 \
  --base_lr 0.01 --n_skip 3 --img_size 224 \
  --window_size 7 --rank 32 --groups 8 --gate_mode pam \
  --seed 1234 --val_interval 15 \
2>&1 | tee run_adada_isic.log && \
python -u test.py \
  --dataset ISIC --vit_name R50-ViT-B_16 \
  --num_classes 2 --max_epochs 300 --batch_size 12 --img_size 224 \
  --window_size 7 --rank 32 --groups 8 --gate_mode pam \
2>&1 | tee -a run_adada_isic.log
```

#### AdaDA — Global PAM (M=112, r=64), 2×T4 (~8h, after Kvasir/ISIC)

```bash
cd /teamspace/studios/this_studio/AdaDA-TransUNet/experiments/Ada-DA-TransUNet

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train.py \
  --dataset Synapse --vit_name R50-ViT-B_16 \
  --max_epochs 300 --batch_size 12 \
  --base_lr 0.01 --n_skip 3 --img_size 224 \
  --window_size 112 --rank 64 --groups 8 --gate_mode pam \
  --seed 1234 --val_interval 15 \
2>&1 | tee run_adada_global_pam.log && \
python -u test.py \
  --dataset Synapse --vit_name R50-ViT-B_16 \
  --num_classes 9 --max_epochs 300 --batch_size 12 --img_size 224 \
  --window_size 112 --rank 64 --groups 8 --gate_mode pam \
2>&1 | tee -a run_adada_global_pam.log
```
