# AdaDA Ablation Experiment Commands

Run from the experiment directory — `cd` first:

```bash
cd /teamspace/studios/this_studio/AdaDA-TransUNet/experiments/Ada-DA-TransUNet
```

Adjust `CUDA_VISIBLE_DEVICES` to whichever GPUs are free (`nvidia-smi` to check).

> **Note:** `gate_mode=fixed` (g=0.5) is skipped — the original `gate_mode=learn` run (77.93%)
> already confirmed gate collapsed to g≈0.5 (Δg=0.0000), making fixed redundant.

---

## Exp 1 — Architecture Recovery: M=14, r=64

```bash
# 2-GPU DDP (faster)
CUDA_VISIBLE_DEVICES=0,1 nohup torchrun --nproc_per_node=2 --master_port=29501 train.py --dataset Synapse --vit_name R50-ViT-B_16 --max_epochs 300 --batch_size 12 --base_lr 0.01 --n_skip 3 --img_size 224 --window_size 14 --rank 64 --groups 8 --gate_mode learn --seed 1234 --val_interval 15 > run_M14_r64.log 2>&1 &
echo "Exp1 PID: $!"
```

```bash
# Single GPU fallback
CUDA_VISIBLE_DEVICES=0 nohup python train.py --dataset Synapse --vit_name R50-ViT-B_16 --max_epochs 300 --batch_size 12 --base_lr 0.01 --n_skip 3 --img_size 224 --window_size 14 --rank 64 --groups 8 --gate_mode learn --seed 1234 --val_interval 15 > run_M14_r64.log 2>&1 &
echo "Exp1 PID: $!"
```

---

## Exp 2 — Gate Ablation: PAM Only (g=1)

```bash
CUDA_VISIBLE_DEVICES=2 nohup python train.py --dataset Synapse --vit_name R50-ViT-B_16 --max_epochs 300 --batch_size 24 --base_lr 0.01 --n_skip 3 --img_size 224 --window_size 7 --rank 32 --groups 8 --gate_mode pam --seed 1234 --val_interval 15 > run_gate_pam.log 2>&1 &
echo "Exp2 PID: $!"
```

```bash
CUDA_VISIBLE_DEVICES=2,3 nohup torchrun --nproc_per_node=2 --master_port=29510 train.py --dataset Synapse --vit_name R50-ViT-B_16 --max_epochs 300 --batch_size 12 --base_lr 0.01 --n_skip 3 --img_size 224 --window_size 7 --rank 32 --groups 8 --gate_mode pam --seed 1234 --val_interval 15 > run_gate_pam.log
```

---

## Exp 3 — Gate Ablation: CAM Only (g=0)

```bash
CUDA_VISIBLE_DEVICES=3 nohup python train.py --dataset Synapse --vit_name R50-ViT-B_16 --max_epochs 300 --batch_size 24 --base_lr 0.01 --n_skip 3 --img_size 224 --window_size 7 --rank 32 --groups 8 --gate_mode cam --seed 1234 --val_interval 15 > run_gate_cam.log 2>&1 &
echo "Exp3 PID: $!"
```

---

## Monitor

```bash
# Follow all logs
tail -f run_M14_r64.log run_gate_pam.log run_gate_cam.log

# GPU usage
nvidia-smi

# Check running processes
ps aux | grep train.py
```
