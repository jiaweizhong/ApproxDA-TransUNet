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

Kaggle Notebooks cannot read local files. Upload everything once as private Kaggle Datasets:

1. Go to [kaggle.com/datasets](https://www.kaggle.com/datasets) → **New Dataset**
2. Create a dataset for the data: upload `data/Synapse/` → name it e.g. `synapse-medical`
3. Create a dataset for the weights: upload `model/` → name it e.g. `vit-pretrained-weights`
4. Create a dataset for the code: upload `DA-TransUNet/` and `Ada-DA-TransUNet/` → name it e.g. `adada-transunet-code`

### Step 2 — Create a Kaggle Notebook

1. Go to [kaggle.com/code](https://www.kaggle.com/code) → **New Notebook**
2. **Settings → Accelerator**: select **GPU T4 x1**
3. **Data → Add Data**: attach all three datasets from Step 1
4. Kaggle mounts datasets at `/kaggle/input/<dataset-name>/`

### Step 3 — Install dependencies

```bash
%%bash
pip install -q timm einops ml-collections medpy SimpleITK tensorboardX
```

> PyTorch, torchvision, scipy, h5py, numpy, and tqdm are pre-installed on Kaggle.

### Step 4 — Copy code to a writable directory

Kaggle input is read-only. Copy code to `/kaggle/working/` before running:

```bash
%%bash
cp -r /kaggle/input/adada-transunet-code/DA-TransUNet     /kaggle/working/DA-TransUNet
cp -r /kaggle/input/adada-transunet-code/Ada-DA-TransUNet /kaggle/working/Ada-DA-TransUNet
```

---

## Reproduce DA-TransUNet (Baseline)

**Train:**

```bash
%%bash
cd /kaggle/working/DA-TransUNet

python train.py \
  --dataset Synapse \
  --vit_name R50-ViT-B_16 \
  --root_path /kaggle/input/synapse-medical/Synapse/train_npz \
  --list_dir ./lists/lists_Synapse \
  --max_epochs 150 \
  --batch_size 24 \
  --base_lr 0.01 \
  --n_skip 3 \
  --img_size 224 \
  --seed 1234
```

> If OOM: use `--batch_size 12 --base_lr 0.005` (halve lr when halving batch size).

**Test:**

```bash
%%bash
cd /kaggle/working/DA-TransUNet

python test.py \
  --dataset Synapse \
  --vit_name R50-ViT-B_16 \
  --volume_path /kaggle/input/synapse-medical/Synapse/test_vol_h5 \
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
  --root_path /kaggle/input/synapse-medical/Synapse/train_npz \
  --list_dir ./lists/lists_Synapse \
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
  --volume_path /kaggle/input/synapse-medical/Synapse/test_vol_h5 \
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

Kaggle T4 (15 GB): batch=24 fits for both models.

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
