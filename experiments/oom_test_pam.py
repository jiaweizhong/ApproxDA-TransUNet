"""
Quantifies PAM memory at each skip resolution and demonstrates that full PAM
at 112x112 causes OOM during training on a T4 (14.6 GB), even though the
attention tensor alone fits on an empty GPU.

Key distinction:
  Empty-GPU test: only allocates the attention tensor → fits (7 GB < 14.6 GB)
  During training: model already uses ~10 GB → 10 + 7 = 17 GB → OOM

Run on the target GPU:
    cd experiments && python oom_test_pam.py
"""
import torch


MODEL_VRAM_GB = 9.8   # measured peak VRAM during AdaDA training (proxy for DA-TransUNet)


def human_mb(n_elements):
    return n_elements * 4 / 1024 ** 2


def human_gb(n_elements):
    return n_elements * 4 / 1024 ** 3


def test_pam_allocation(batch_size, h, w, label):
    N = h * w
    attn_gb = human_gb(batch_size * N * N)
    print(f"\n[{label}]")
    print(f"  Spatial: {h}x{w}   N={N:,}   batch/GPU={batch_size}")
    print(f"  Full PAM attention (B x N x N): {attn_gb:.2f} GB")

    if not torch.cuda.is_available():
        print("  (no CUDA — skipping allocation test)")
        return

    gpu_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    torch.cuda.reset_peak_memory_stats()
    try:
        q = torch.randn(batch_size, N, 1, device="cuda")   # minimal channel dim
        k = torch.randn(batch_size, 1, N, device="cuda")
        attn = torch.bmm(q, k)                             # B x N x N
        peak_gb = torch.cuda.max_memory_allocated() / 1024 ** 3
        combined = MODEL_VRAM_GB + peak_gb
        oom_status = "OOM" if combined > gpu_gb else "OK"
        print(f"  Standalone peak:  {peak_gb:.2f} GB  (GPU has {gpu_gb:.1f} GB) -> standalone OK")
        print(f"  Model (~{MODEL_VRAM_GB} GB) + attention: {combined:.1f} GB -> training {oom_status}")
        del q, k, attn
        torch.cuda.empty_cache()
    except RuntimeError as e:
        peak_gb = torch.cuda.max_memory_allocated() / 1024 ** 3
        print(f"  -> CUDA OOM at {peak_gb:.2f} GB: {e}")


if __name__ == "__main__":
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name}  ({props.total_memory / 1024**3:.1f} GB VRAM)")
    else:
        print("GPU: none")

    print(f"(Model baseline VRAM assumed: {MODEL_VRAM_GB} GB, measured from AdaDA training logs)")

    batch = 12  # per-GPU batch size (DataParallel on 2 GPUs, total=24)

    test_pam_allocation(batch, h=28,  w=28,  label="512ch skip  28x28  — DA-TransUNet block 0")
    test_pam_allocation(batch, h=56,  w=56,  label="256ch skip  56x56  — DA-TransUNet block 1")
    test_pam_allocation(batch, h=112, w=112, label="64ch  skip 112x112 — DA-TransUNet block 2 (NEVER RUNS in original code)")

    # AdaDA LowRankWindowedPAM memory at 112x112
    print("\n--- AdaDA-TransUNet windowed + low-rank PAM at 112x112 ---")
    window_size = 7
    rank = 32
    n_windows = (112 // window_size) ** 2   # 256 windows
    N_w = window_size ** 2                   # 49 tokens per window

    full_pam_elements   = batch * (112 * 112) ** 2
    windowed_elements   = batch * n_windows * N_w * N_w      # N_w x N_w per window
    lowrank_elements    = batch * n_windows * N_w * rank      # N_w x rank per window (actual AdaDA scores)

    print(f"  Full PAM (B x N x N):                {human_gb(full_pam_elements)*1024:.0f} MB")
    print(f"  Windowed only (B x nW x Nw x Nw):   {human_mb(windowed_elements):.1f} MB  (~{full_pam_elements / windowed_elements:.0f}x smaller)")
    print(f"  Windowed + low-rank (B x nW x Nw x r): {human_mb(lowrank_elements):.1f} MB  (~{full_pam_elements / lowrank_elements:.0f}x smaller)")
