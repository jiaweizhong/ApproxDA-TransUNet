# ApproxDA-TransUNet

Official implementation of **ApproxDA-TransUNet: Approximate Dual Attention for Efficient Medical Image Segmentation** (BIBM 2026).

ApproxDA-TransUNet reformulates the dual-attention module of DA-TransUNet along three independently controllable approximation axes — spatial window size (M), low-rank projection (r), and channel grouping (G) — achieving state-of-the-art segmentation across three medical imaging benchmarks while enabling native multi-GPU distributed training.

## Results

| Dataset | Method | DSC (%) | HD95 (mm) |
|---|---|---|---|
| Synapse Multi-Organ CT | DA-TransUNet | 79.80 | 23.48 |
| | **ApproxDA-TransUNet (M=28)** | **80.94** | 27.49 |
| Kvasir-SEG Polyp | DA-TransUNet | 88.44 | 53.04 |
| | **ApproxDA-TransUNet (M=56)** | **90.17** | 44.35 |
| ISIC 2018 Skin Lesion | DA-TransUNet | 88.88 | — |
| | **ApproxDA-TransUNet (M=7)** | **89.58** | — |

## Pretrained Checkpoints

Download from Kaggle: **[deepsotaai/approxda-transunet-checkpoints](https://www.kaggle.com/datasets/deepsotaai/approxda-transunet-checkpoints)**

| File | Dataset | Config | DSC |
|---|---|---|---|
| `ApproxDA_Synapse224_pretrain_R50-ViT-B_16_skip3_epo300_bs12_224_M28_pam_best_model.pth` | Synapse | M=28, gate=pam, r=32 | 80.94% |
| `ApproxDA_Kvasir224_pretrain_R50-ViT-B_16_skip3_epo300_bs24_224_best_model.pth` | Kvasir-SEG | M=56, gate=pam, r=32 | 90.17% |
| `ApproxDA_ISIC224_pretrain_R50-ViT-B_16_skip3_epo300_bs24_224_best_model.pth` | ISIC 2018 | M=7, gate=learn, r=32 | 89.58% |

```bash
pip install kaggle
kaggle datasets download -d deepsotaai/approxda-transunet-checkpoints --unzip
```

## Requirements

```bash
pip install torch torchvision numpy scipy pillow h5py ml_collections
```

Tested with Python 3.9+, PyTorch 2.x, CUDA 11.8+.

Also download the ViT-B/16 pretrained weights (R50+ViT-B_16.npz) from the
[TransUNet model zoo](https://console.cloud.google.com/storage/browser/vit_models) and place at:

```
model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz
```

## Dataset Setup

### Synapse Multi-Organ CT

```bash
kaggle datasets download -d dogcdt/synapse --unzip -p ../data
# Unpacks as data/Synapse/test_vol_h5/ and data/Synapse/train_npz/
```

### Kvasir-SEG

```bash
mkdir -p ../data/raw_kvasir
kaggle datasets download -d debeshjha1/kvasirseg -p ../data/raw_kvasir --unzip
mkdir -p ../data/Kvasir-SEG
mv ../data/raw_kvasir/Kvasir-SEG/Kvasir-SEG/images ../data/Kvasir-SEG/images
mv ../data/raw_kvasir/Kvasir-SEG/Kvasir-SEG/masks  ../data/Kvasir-SEG/masks
rm -rf ../data/raw_kvasir

python datasets/generate_lists.py --dataset Kvasir --data_dir ../data/Kvasir-SEG
```

### ISIC 2018

```bash
mkdir -p ../data/raw_isic
kaggle datasets download -d tschandl/isic2018-challenge-task1-data-segmentation \
  -p ../data/raw_isic --unzip
mkdir -p ../data/ISIC2018/images ../data/ISIC2018/masks
mv ../data/raw_isic/ISIC2018_Task1-2_Training_Input/*.jpg   ../data/ISIC2018/images/
mv ../data/raw_isic/ISIC2018_Task1_Training_GroundTruth/*_segmentation.png ../data/ISIC2018/masks/
rm -rf ../data/raw_isic

python datasets/generate_lists.py --dataset ISIC --data_dir ../data/ISIC2018
```

Expected layout:

```
data/
├── Synapse/
│   ├── train_npz/
│   └── test_vol_h5/
├── Kvasir-SEG/
│   ├── images/
│   └── masks/
└── ISIC2018/
    ├── images/
    └── masks/
```

## Training

```bash
# Synapse (best config: M=28, gate=pam)
python train.py --dataset Synapse --window_size 28 --gate_mode pam \
  --rank 32 --max_epochs 300 --batch_size 12 --img_size 224 --n_skip 3

# Kvasir-SEG (best config: M=56, gate=pam)
python train.py --dataset Kvasir --window_size 56 --gate_mode pam \
  --rank 32 --max_epochs 300 --batch_size 24 --img_size 224 --n_skip 3

# ISIC 2018 (best config: M=7, gate=learn)
python train.py --dataset ISIC --window_size 7 --gate_mode learn \
  --rank 32 --max_epochs 300 --batch_size 24 --img_size 224 --n_skip 3
```

Multi-GPU training (DDP) is supported natively:

```bash
torchrun --nproc_per_node=2 train.py --dataset Synapse --window_size 28 \
  --gate_mode pam --rank 32 --max_epochs 300 --batch_size 12
```

## Inference

```bash
# Synapse
python test.py --dataset Synapse \
  --volume_path ../data/Synapse/test_vol_h5 \
  --snapshot <path/to/best_model.pth> \
  --window_size 28 --gate_mode pam --rank 32

# Kvasir-SEG
python test.py --dataset Kvasir \
  --volume_path ../data/Kvasir-SEG \
  --snapshot <path/to/best_model.pth> \
  --window_size 56 --gate_mode pam --rank 32

# ISIC 2018
python test.py --dataset ISIC \
  --volume_path ../data/ISIC2018 \
  --snapshot <path/to/best_model.pth> \
  --window_size 7 --gate_mode learn --rank 32
```

## Citation

```bibtex
@inproceedings{approxdatransunet2026,
  title     = {ApproxDA-TransUNet: Approximate Dual Attention for Efficient Medical Image Segmentation},
  booktitle = {IEEE International Conference on Bioinformatics and Biomedicine (BIBM)},
  year      = {2026}
}
```
