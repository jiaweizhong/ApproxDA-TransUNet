# Generate Qualitative Figure (Fig 3 / Fig 4)

This guide walks through downloading data, arranging directories, and running
`generate_qualitative_figure.py` to produce both the ApproxDA-TransUNet cross-task
figure and the matching DA-TransUNet predictions for manual comparison.

---

## 0. Prerequisites

```bash
cd experiments/ApproxDA-TransUNet
pip install matplotlib h5py scipy pillow numpy torch
```

Checkpoint files you need (already trained):

| Model | Dataset | Checkpoint path (example) |
|---|---|---|
| ApproxDA (M=28, gate=pam) | Synapse | `../../results/approxda_syn_m28_pam/best_model.pth` |
| ApproxDA (M=56, gate=pam) | Kvasir-SEG | `../../results/approxda_kv_m56_pam/best_model.pth` |
| ApproxDA (M=7, gate=learn) | ISIC 2018 | `../../results/approxda_isic_m7_learn/best_model.pth` |
| DA-TransUNet | Synapse | `../../results/da_transunet_syn/best_model.pth` |
| DA-TransUNet | Kvasir-SEG | `../../results/da_transunet_kv/best_model.pth` |
| DA-TransUNet | ISIC 2018 | `../../results/da_transunet_isic/best_model.pth` |

---

## 1. Synapse Multi-Organ CT

### Download from Kaggle

```bash
# Dataset: https://www.kaggle.com/datasets/dogcdt/synapse
kaggle datasets download -d dogcdt/synapse -p ../../data --unzip
```

The zip unpacks directly as `Synapse/test_vol_h5/` and `Synapse/train_npz/` — no further rearrangement needed. Only `test_vol_h5/` is required for inference.

### Expected layout after download

```
data/
└── Synapse/
    └── test_vol_h5/
        ├── case0001.npy.h5
        ├── case0002.npy.h5
        ├── case0003.npy.h5
        ├── case0004.npy.h5
        ├── case0008.npy.h5
        ├── case0022.npy.h5
        ├── case0025.npy.h5
        ├── case0029.npy.h5
        ├── case0032.npy.h5
        ├── case0035.npy.h5
        ├── case0036.npy.h5
        └── case0038.npy.h5
```

The list file `lists/lists_Synapse/test_vol.txt` already contains the correct case names.
**No extra setup needed for Synapse.**

---

## 2. Kvasir-SEG

### Download from Kaggle

```bash
pip install kaggle
# place your kaggle.json in ~/.kaggle/  (download from kaggle.com → Account → API)

# Slug: debeshjha1/kvasirseg  (original paper authors, 151 MB)
# Zip unpacks as Kvasir-SEG/Kvasir-SEG/{images,masks}/ — double-nested
mkdir -p ../../data/raw_kvasir
kaggle datasets download -d debeshjha1/kvasirseg -p ../../data/raw_kvasir --unzip
mkdir -p ../../data/Kvasir-SEG
mv ../../data/raw_kvasir/Kvasir-SEG/Kvasir-SEG/images ../../data/Kvasir-SEG/images
mv ../../data/raw_kvasir/Kvasir-SEG/Kvasir-SEG/masks  ../../data/Kvasir-SEG/masks
rm -rf ../../data/raw_kvasir
```

### Expected layout after download

```
data/
└── Kvasir-SEG/
    ├── images/
    │   ├── cju0qkwl9qokg0993l0dewei2.jpg
    │   └── ...   (1000 images total)
    └── masks/
        ├── cju0qkwl9qokg0993l0dewei2.jpg
        └── ...   (matching masks, same filename)
```

### Generate list files

```bash
python datasets/generate_lists.py --dataset Kvasir --data_dir ../../data/Kvasir-SEG
# → writes lists/lists_Kvasir/train.txt  (800 cases)
# → writes lists/lists_Kvasir/test.txt   (200 cases)
```

> **Important**: use `--seed 42` (default) to reproduce the same train/test split
> that was used during training. Wrong seed = test cases seen during training.

---

## 3. ISIC 2018

### Download from Kaggle

```bash
# Slug: tschandl/isic2018-challenge-task1-data-segmentation  (13.8 GB)
# Contains train/val/test splits; only training set has ground-truth masks.
# We use training set only (2594 images) and do our own 80/20 split.
mkdir -p ../../data/raw_isic
kaggle datasets download -d tschandl/isic2018-challenge-task1-data-segmentation \
  -p ../../data/raw_isic --unzip

# After unzip:
#   ISIC2018_Task1-2_Training_Input/          ← images (.jpg)
#   ISIC2018_Task1_Training_GroundTruth/      ← masks (*_segmentation.png)
#   ISIC2018_Task1-2_Validation_Input/        ← no masks, skip
#   ISIC2018_Task1-2_Test_Input/              ← no masks, skip

mkdir -p ../../data/ISIC2018/images ../../data/ISIC2018/masks
mv ../../data/raw_isic/ISIC2018_Task1-2_Training_Input/*.jpg \
   ../../data/ISIC2018/images/
mv ../../data/raw_isic/ISIC2018_Task1_Training_GroundTruth/*_segmentation.png \
   ../../data/ISIC2018/masks/
rm -rf ../../data/raw_isic
```

### Expected layout after download

```
data/
└── ISIC2018/
    ├── images/
    │   ├── ISIC_0024306.jpg
    │   └── ...   (2594 images)
    └── masks/
        ├── ISIC_0024306_segmentation.png
        └── ...   (matching masks, note _segmentation suffix)
```

### Generate list files

```bash
python datasets/generate_lists.py --dataset ISIC --data_dir ../../data/ISIC2018
# → writes lists/lists_ISIC/train.txt  (~2075 cases)
# → writes lists/lists_ISIC/test.txt   (~519 cases)
```

---

## 4. Verify directory structure

```
AdaDA-TransUNet/
├── data/
│   ├── Synapse/test_vol_h5/          ← 12 × .npy.h5
│   ├── Kvasir-SEG/images/ + masks/   ← 1000 images each
│   └── ISIC2018/images/ + masks/     ← 2594 images, masks with _segmentation.png
├── experiments/
│   └── ApproxDA-TransUNet/
│       ├── lists/
│       │   ├── lists_Synapse/test_vol.txt  ← already present
│       │   ├── lists_Kvasir/test.txt       ← generated in Step 2
│       │   └── lists_ISIC/test.txt         ← generated in Step 3
│       └── generate_qualitative_figure.py
└── results/   ← checkpoints live here
```

---

## 5. Generate ApproxDA cross-task figure

```bash
cd experiments/ApproxDA-TransUNet

python generate_qualitative_figure.py --mode cross_task \
  --ckpt_syn_best    ../../results/approxda_syn_m28_pam/best_model.pth \
  --ckpt_kvasir_best ../../results/approxda_kv_m56_pam/best_model.pth \
  --ckpt_isic_best   ../../results/approxda_isic_m7_learn/best_model.pth \
  --volume_path_syn  ../../data/Synapse/test_vol_h5 \
  --volume_path_kv   ../../data/Kvasir-SEG \
  --volume_path_isic ../../data/ISIC2018 \
  --out_dir          ../../results/paper_figures
```

**Outputs:**
```
results/paper_figures/
├── fig_qualitative_cross_task.pdf    ← paper figure (3 rows × 3 cols)
├── fig_qualitative_cross_task.png    ← preview
└── cases_used.json                   ← records which case + slice was chosen
```

`cases_used.json` example:
```json
{
  "synapse": { "case_name": "case0008", "slice_idx": 47 },
  "kvasir":  { "case_name": "cju0qkwl9qokg0993l0dewei2" },
  "isic":    { "case_name": "ISIC_0024306" }
}
```

---

## 6. Generate DA-TransUNet predictions on the same cases

```bash
python generate_qualitative_figure.py --mode da_only \
  --ckpt_da_syn    ../../results/da_transunet_syn/best_model.pth \
  --ckpt_da_kvasir ../../results/da_transunet_kv/best_model.pth \
  --ckpt_da_isic   ../../results/da_transunet_isic/best_model.pth \
  --volume_path_syn  ../../data/Synapse/test_vol_h5 \
  --volume_path_kv   ../../data/Kvasir-SEG \
  --volume_path_isic ../../data/ISIC2018 \
  --cases_log      ../../results/paper_figures/cases_used.json \
  --out_dir        ../../results/paper_figures
```

**Outputs** (individual 3×3 inch PNG per panel, one per dataset):
```
results/paper_figures/
├── da_synapse_input.png
├── da_synapse_gt.png
├── da_synapse_da_pred.png     ← DA-TransUNet prediction, same case + slice as Step 5
├── da_kvasir_input.png
├── da_kvasir_gt.png
├── da_kvasir_da_pred.png
├── da_isic_input.png
├── da_isic_gt.png
└── da_isic_da_pred.png
```

> Input and GT are the same as Step 5 — you can use either copy.

---

## 7. Combine into final figure

Open `fig_qualitative_cross_task.png` and the `da_*_da_pred.png` files in
any image editor (Photoshop / GIMP / Figma / PowerPoint) and insert a
**DA-TransUNet** column between GT and ApproxDA prediction.

Final layout:

```
Row             | Input | GT | DA-TransUNet | ApproxDA (ours)
----------------|-------|----|--------------|----------------
Synapse (M=28)  |  ✓   | ✓  |   da_pred    |      ✓
Kvasir  (M=56)  |  ✓   | ✓  |   da_pred    |      ✓
ISIC    (M=7)   |  ✓   | ✓  |   da_pred    |      ✓
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `FileNotFoundError: .npy.h5` | Check `--volume_path_syn` points to the folder containing `case0008.npy.h5` etc. |
| `No valid split found for Kvasir` | Run `generate_lists.py` first (Step 2) |
| `cases_used.json not found` | Run Step 5 (cross_task) before Step 6 (da_only) |
| DA-TransUNet import error | Confirm `experiments/DA-TransUNet/Architecture/DATransUNet.py` exists |
| Wrong case loaded in da_only | Delete `cases_used.json`, re-run Step 5 to regenerate |
| CUDA OOM | Add `--img_size 224` (already default); or run on CPU (slower, ~5 min/dataset) |
