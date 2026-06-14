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
pip install timm einops ml-collections medpy SimpleITK tensorboardX thop h5py scipy
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

`train.py` and `test.py` resolve paths relative to their own directory (`../data/`, `../model/`). Symlink once:

```bash
REPO=/teamspace/studios/this_studio/AdaDA-TransUNet/experiments

ln -sfn /teamspace/studios/this_studio/data  $REPO/data
ln -sfn /teamspace/studios/this_studio/model $REPO/model

# verify
ls -la $REPO/data/Synapse/
ls -la $REPO/model/vit_checkpoint/imagenet21k/
```

Expected: `R50+ViT-B_16.npz` (with `+` intact — Lightning AI does not strip it).

### 6. Fix datasets/ package shadowing

```bash
REPO=/teamspace/studios/this_studio/AdaDA-TransUNet/experiments
touch $REPO/DA-TransUNet/datasets/__init__.py
touch $REPO/Ada-DA-TransUNet/datasets/__init__.py
```

---

## Running Experiments

All training commands use `nohup … &` so they survive browser disconnects. Logs are written to a `.log` file in the experiment directory.

Check GPU availability first:

```bash
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
```

---

### DA-TransUNet — Synapse (single T4, baseline)

```bash
cd /teamspace/studios/this_studio/AdaDA-TransUNet/experiments/DA-TransUNet

cat > run_da.sh << 'EOF'
python -u train.py \
  --dataset Synapse --vit_name R50-ViT-B_16 \
  --max_epochs 300 --batch_size 24 --n_gpu 1 \
  --base_lr 0.01 --n_skip 3 --img_size 224 \
  --seed 1234 --val_interval 15 \
&& \
python -u test.py \
  --dataset Synapse --vit_name R50-ViT-B_16 \
  --num_classes 9 --max_epochs 300 --img_size 224 --is_savenii
EOF

nohup bash run_da.sh > run_da.log 2>&1 &
echo "DA PID: $!"
```

---

### AdaDA-TransUNet — Synapse, single T4 (apples-to-apples efficiency row)

```bash
cd /teamspace/studios/this_studio/AdaDA-TransUNet/experiments/Ada-DA-TransUNet

cat > run_adada_1gpu.sh << 'EOF'
python -u train.py \
  --dataset Synapse --vit_name R50-ViT-B_16 \
  --max_epochs 300 --batch_size 24 --n_gpu 1 \
  --base_lr 0.01 --n_skip 3 --img_size 224 \
  --window_size 7 --rank 32 --groups 8 \
  --seed 1234 --val_interval 15 \
&& \
python -u test.py \
  --dataset Synapse --vit_name R50-ViT-B_16 \
  --num_classes 9 --max_epochs 300 --img_size 224 \
  --window_size 7 --rank 32 --groups 8 --is_savenii
EOF

nohup bash run_adada_1gpu.sh > run_adada_1gpu.log 2>&1 &
echo "AdaDA 1GPU PID: $!"
```

---

### AdaDA-TransUNet — Synapse, 2×T4 DDP (multi-GPU efficiency row)

```bash
cd /teamspace/studios/this_studio/AdaDA-TransUNet/experiments/Ada-DA-TransUNet

cat > run_adada_2gpu.sh << 'EOF'
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train.py \
  --dataset Synapse --vit_name R50-ViT-B_16 \
  --max_epochs 300 --batch_size 12 --n_gpu 2 \
  --base_lr 0.01 --n_skip 3 --img_size 224 \
  --window_size 7 --rank 32 --groups 8 \
  --seed 1234 --val_interval 15 \
&& \
python -u test.py \
  --dataset Synapse --vit_name R50-ViT-B_16 \
  --num_classes 9 --max_epochs 300 --batch_size 12 --img_size 224 \
  --window_size 7 --rank 32 --groups 8 --is_savenii
EOF

nohup bash run_adada_2gpu.sh > run_adada_2gpu.log 2>&1 &
echo "AdaDA 2GPU PID: $!"
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

## Phase 2 Ablation Runs (Week 2)

### No-gate ablation (`--disable_gate`, fixed g=0.5)

```bash
cd /teamspace/studios/this_studio/AdaDA-TransUNet/experiments/Ada-DA-TransUNet

cat > run_nogate.sh << 'EOF'
python -u train.py \
  --dataset Synapse --vit_name R50-ViT-B_16 \
  --max_epochs 300 --batch_size 12 --n_gpu 2 \
  --base_lr 0.01 --n_skip 3 --img_size 224 \
  --window_size 7 --rank 32 --groups 8 \
  --disable_gate --seed 1234 --val_interval 15 \
&& \
CUDA_VISIBLE_DEVICES=0,1 python -u test.py \
  --dataset Synapse --vit_name R50-ViT-B_16 \
  --num_classes 9 --max_epochs 300 --batch_size 12 --img_size 224 \
  --window_size 7 --rank 32 --groups 8 --is_savenii
EOF

CUDA_VISIBLE_DEVICES=0,1 nohup bash run_nogate.sh > run_nogate.log 2>&1 &
echo "No-gate PID: $!"
```

### Full model — entropy gate (Phase 2, after code change to block.py)

```bash
# Implement Phase 2 first: gate_fc = Linear(channels+1, channels) in block.py
# Then retrain from scratch (checkpoint incompatible with GAP-only checkpoint)

cat > run_entropy_gate.sh << 'EOF'
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train.py \
  --dataset Synapse --vit_name R50-ViT-B_16 \
  --max_epochs 300 --batch_size 12 --n_gpu 2 \
  --base_lr 0.01 --n_skip 3 --img_size 224 \
  --window_size 7 --rank 32 --groups 8 \
  --seed 1234 --val_interval 15 \
&& \
python -u test.py \
  --dataset Synapse --vit_name R50-ViT-B_16 \
  --num_classes 9 --max_epochs 300 --batch_size 12 --img_size 224 \
  --window_size 7 --rank 32 --groups 8 --is_savenii
EOF

nohup bash run_entropy_gate.sh > run_entropy_gate.log 2>&1 &
echo "Entropy gate PID: $!"
```

### Rank sensitivity (`--rank 8` on full model)

```bash
cat > run_rank8.sh << 'EOF'
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train.py \
  --dataset Synapse --vit_name R50-ViT-B_16 \
  --max_epochs 300 --batch_size 12 --n_gpu 2 \
  --base_lr 0.01 --n_skip 3 --img_size 224 \
  --window_size 7 --rank 8 --groups 8 \
  --seed 1234 --val_interval 15 \
&& \
python -u test.py \
  --dataset Synapse --vit_name R50-ViT-B_16 \
  --num_classes 9 --max_epochs 300 --batch_size 12 --img_size 224 \
  --window_size 7 --rank 8 --groups 8 --is_savenii
EOF

nohup bash run_rank8.sh > run_rank8.log 2>&1 &
echo "Rank-8 PID: $!"
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
          $BASE/Ada-DA-TransUNet/run_adada_1gpu.log \
          $BASE/Ada-DA-TransUNet/run_adada_2gpu.log \
          $BASE/Ada-DA-TransUNet/run_nogate.log \
          $BASE/Ada-DA-TransUNet/run_entropy_gate.log; do
  echo "=== $(basename $f) ===";
  tail -3 "$f" 2>/dev/null || echo "(not started)";
done

# Final DSC numbers
grep "Testing performance" $BASE/DA-TransUNet/run_da.log 2>/dev/null
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

## Future Datasets (Kvasir-SEG, ISIC 2018)

Code changes required before running — see `EXPERIMENT_PLAN.md §Code Changes Required`. Once `dataset_kvasir.py` and `dataset_isic.py` exist, the commands follow the same pattern: replace `--dataset Synapse` with `--dataset Kvasir` or `--dataset ISIC`.
