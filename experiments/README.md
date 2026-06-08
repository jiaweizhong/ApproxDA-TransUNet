# AdaDA-TransUNet Experiments

This folder contains two models:

| Folder | Description |
|---|---|
| `DA-TransUNet/` | Original baseline — reproduce paper results |
| `Ada-DA-TransUNet/` | Our model — AdaDA-TransUNet implementation |

Shared resources at the experiment root:

| Path | Contents |
|---|---|
| `data/Synapse/train_npz/` | 2211 training slices (.npz) |
| `data/Synapse/test_vol_h5/` | 12 test volumes (.h5) |
| `model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz` | Pretrained ViT weights |

---

## Running on Kaggle (Free GPU)

Kaggle provides a free **T4 (15 GB VRAM)** or **P100 (16 GB VRAM)** GPU with a 30 hr/week limit and up to 9 hr per session. Both models fit at batch=24 within the T4's 15 GB.

### Step 1 — Upload to Kaggle Datasets

The Synapse dataset is already publicly available on Kaggle — no need to upload it. You only need to upload two private datasets once:

1. Go to [kaggle.com/datasets](https://www.kaggle.com/datasets) → **New Dataset**
2. Create a dataset for the weights: upload `model/` → name it e.g. `vit-pretrained-weights`
3. Create a dataset for the code: upload `DA-TransUNet/` and `Ada-DA-TransUNet/` → name it e.g. `adada-transunet-code`

### Step 2 — Create a Kaggle Notebook

1. Go to [kaggle.com/code](https://www.kaggle.com/code) → **New Notebook**
2. **Settings → Accelerator**: select **GPU T4 x1**
3. **Data → Add Data**: attach all three datasets:
   - Search **dogcdt/synapse** (public) — Synapse multi-organ CT data
   - Your `vit-pretrained-weights` dataset
   - Your `adada-transunet-code` dataset
4. Kaggle mounts datasets at `/kaggle/input/<dataset-name>/`

### Step 3 — Install dependencies

In a Python notebook cell (prefix with `!`):

```python
!pip install -q timm einops ml-collections medpy SimpleITK tensorboardX
```

> PyTorch, torchvision, scipy, h5py, numpy, and tqdm are pre-installed on Kaggle.

### Step 4 — One-time Kaggle setup

Run all of the following in a single cell. These only need to be done once per session:

```python
# 1. Copy code and weights to writable directory
!cp -r /kaggle/input/datasets/deepsotaai/adada-transunet-code/DA-TransUNet     /kaggle/working/DA-TransUNet
!cp -r /kaggle/input/datasets/deepsotaai/adada-transunet-code/Ada-DA-TransUNet /kaggle/working/Ada-DA-TransUNet
!cp -r /kaggle/input/datasets/deepsotaai/vit-pretrained-weights/model           /kaggle/working/model

# 2. Kaggle strips '+' from filenames on upload — rename the weights file back
!mv /kaggle/working/model/vit_checkpoint/imagenet21k/R50ViT-B_16.npz \
    /kaggle/working/model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz

# 3. Fix conflict with Kaggle's pre-installed HuggingFace 'datasets' library
!touch /kaggle/working/DA-TransUNet/datasets/__init__.py
!touch /kaggle/working/Ada-DA-TransUNet/datasets/__init__.py

# 4. Symlink Synapse data — train.py hardcodes '../data/Synapse/' and ignores --root_path
!mkdir -p /kaggle/working/data/Synapse
!ln -sfn /kaggle/input/datasets/dogcdt/synapse/Synapse/train_npz  /kaggle/working/data/Synapse/train_npz
!ln -sfn /kaggle/input/datasets/dogcdt/synapse/Synapse/test_vol_h5 /kaggle/working/data/Synapse/test_vol_h5
```

> Replace `deepsotaai` with your Kaggle username. Datasets attached via kagglehub are always under `/kaggle/input/datasets/<owner>/<dataset-name>/`.

---

## Reproduce DA-TransUNet (Baseline)

**Train:**

```bash
%%bash
cd /kaggle/working/DA-TransUNet

python train.py \
  --dataset Synapse \
  --vit_name R50-ViT-B_16 \
  --max_epochs 150 \
  --batch_size 24 \
  --base_lr 0.01 \
  --n_skip 3 \
  --img_size 224 \
  --seed 1234
```

> OOM: use `--batch_size 12 --base_lr 0.005`. Multi-GPU (`--n_gpu 2`) does not work — the O(N²) PAM attention OOMs during backward even at small batch sizes.

**Test:**

```bash
%%bash
cd /kaggle/working/DA-TransUNet

python test.py \
  --dataset Synapse \
  --vit_name R50-ViT-B_16 \
  --num_classes 9 \
  --img_size 224 \
  --is_savenii
```

To test with the original paper's pretrained weights instead of your own checkpoint, download the `.pth` file from the [Google Drive link](https://drive.google.com/drive/folders/1UqIEPcohjIZdpT5bIc0NPcxkvI8i4ily), place it as `model_out/TU_Synapse224/best_model.pth`, then run `python test.py --dataset Synapse --vit_name R50-ViT-B_16`.

---

## Run AdaDA-TransUNet (Our Model)

### Smoke test first

Run before committing GPU hours to training — verifies all module shapes are correct:

```bash
%%bash
cd /kaggle/working/Ada-DA-TransUNet
python quick_test.py
```

Expected output:
```
=== window_partition / window_reverse ===  PASS
=== LowRankWindowedPAM ===  C=768 H=14 PASS  ...
=== GroupedCAM ===           C=768 H=14 PASS  ...
=== AdaDABlock ===           C=768 H=14 PASS  ...
=== hardware_config ===      PASS
=== Full model forward pass ===
  Output shape: torch.Size([1, 9, 224, 224])  PASS
All tests passed.
```

### Train:

```bash
%%bash
cd /kaggle/working/Ada-DA-TransUNet

python train.py \
  --dataset Synapse \
  --vit_name R50-ViT-B_16 \
  --num_classes 9 \
  --max_epochs 150 \
  --batch_size 24 \
  --base_lr 0.01 \
  --n_skip 3 \
  --img_size 224 \
  --window_size 7 \
  --rank 32 \
  --groups 8
```

**AdaDA-specific arguments:**

| Argument | Default | Description |
|---|---|---|
| `--window_size` | `7` | Window size for LowRankWindowedPAM |
| `--rank` | `32` | Low-rank projection dimension (r ≪ window²=49) |
| `--groups` | `8` | Number of channel groups for GroupedCAM |

**Ablation variants** (change one argument at a time to isolate each contribution):

```bash
# Lean config — lower memory, faster, expected lower accuracy
--window_size 7 --rank 16 --groups 16

# Rich config — requires >8 GB free VRAM
--window_size 14 --rank 64 --groups 4
```

### Test:

```bash
%%bash
cd /kaggle/working/Ada-DA-TransUNet

python test.py \
  --dataset Synapse \
  --vit_name R50-ViT-B_16 \
  --num_classes 9 \
  --img_size 224 \
  --is_savenii
```

> `test.py` auto-detects free GPU memory via `hardware_config()` and sets `window_size/rank/groups` automatically — no manual flags needed at test time.

---

## VRAM Reference

| Model | batch=24 | batch=12 |
|---|---|---|
| DA-TransUNet (baseline) | ~13 GB | ~7 GB |
| AdaDA (rank=32, window=7, groups=8) — default | ~11 GB | ~6 GB |
| AdaDA (rank=16, window=7, groups=16) — lean | ~9 GB | ~5 GB |

- **Use `--n_gpu 1` on T4**. The O(N²) PAM in DA-TransUNet makes multi-GPU DataParallel OOM during backward. Use T4 x1 for both models to ensure a fair comparison.
- **P100 is not supported**: Kaggle's current PyTorch (2.x) requires sm_70+ and P100 is sm_60 — use T4 instead.
- **1x T4 (15 GB)**: batch=24 fits for both models. Training ~8 hrs.
- First checkpoint saved at epoch 99; final at epoch 149. Do not stop before epoch 99 or all progress is lost.

---

## Saving Checkpoints out of Kaggle

`/kaggle/working/` is writable but lost after the session unless saved to output:

```python
import shutil
# In a notebook cell — copy to the persistent output folder
shutil.copytree('/kaggle/working/Ada-DA-TransUNet/model_out', '/kaggle/output/ada_checkpoints')
```

Download from the notebook's **Output** tab after the session.

---

## Original DA-TransUNet Reference

> Sun, G. et al. "DA-TransUNet: integrating spatial and channel dual attention with transformer U-net for medical image segmentation." *Frontiers in Bioengineering and Biotechnology* 12 (2024): 1398237.
> https://doi.org/10.3389/fbioe.2024.1398237

- [Pretrained weights (Google Drive)](https://drive.google.com/drive/folders/1UqIEPcohjIZdpT5bIc0NPcxkvI8i4ily)
- [Preprocessed Synapse data (Google Drive)](https://drive.google.com/drive/folders/1ACJEoTp-uqfFJ73qS3eUObQh52nGuzCd)
- [Original code (GitHub)](https://github.com/SUN-1024/DA-TransUNet)

```bibtex
@article{sun2024transunet,
  title={DA-TransUNet: integrating spatial and channel dual attention with transformer U-net for medical image segmentation},
  author={Sun, Guanqun and Pan, Yizhi and Kong, Weikun and Xu, Zichang and Ma, Jianhua and Racharak, Teeradaj and Nguyen, Le-Minh and Xin, Junyi},
  journal={Frontiers in Bioengineering and Biotechnology},
  volume={12},
  pages={1398237},
  year={2024},
  publisher={Frontiers Media SA}
}
```
