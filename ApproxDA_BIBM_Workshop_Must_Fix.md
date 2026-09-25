# ApproxDA-TransUNet — BIBM Workshop 必改清单（时间优先版）

> 目标：在时间有限的情况下，优先修复 **4 位 reviewer 重复指出、最可能再次导致 reject 的问题**。  
> 原则：**先修 validity / fairness / incorrect claim，再做可选增强。**

---

## 修改状态（2026-09-24 更新）

- **稿件**：`paper/ApproxDA-TransUNet.tex`（IEEE conference 模板，7 页，含 22 条参考文献），编译没有 warning。原 BIBM 版可以从 git history 找回。
- **标记**：红色 `\tbd{}` 是待重跑后替换的数字，红色 `\todo{}` 是依赖实验的文字；定稿时把 `\revisionmarkstrue` 改成 `false`。
- **和 ISBI 版共用**：架构（c16 block）、训练协议和实验清单，见 `ApproxDA_ISBI_Reduction_and_Revision_Plan.md` §3–§5。

**总结：P0 里所有“改文字就能解决”的部分都已完成；剩下的是重跑实验、回填数字和重新出图。**

### A. ✅ 已完成（不需要实验）

| # | 项目 | 做了什么 |
|---|---|---|
| 1 | Validation / test protocol（文字） | 删除 “every 15 epochs / best checkpoint”。**和本清单原本的写法不同，按后来的决定执行**：不做 checkpoint 选择，一律取最后一个 epoch（和 TransUNet 一致）；ISIC 用官方 val（2594/100/1000）选 M/r/G/gate；Synapse、Kvasir 没有 val，从消融结果里选，并在 §V-C Limitations 里如实说明，同时给出偏差上限 |
| 2 | Single-variable ablation（表格设计） | Table IV 分为 rank / window-only / groups / routing 四块，每块只变一个因素；删除 M=14/r=64/learn 这一行，不再用它论证 “rank is not the bottleneck”；G 改在 CAM-only 下做；加入 exact attention（no projection）对照 |
| 3 | Method / Eq.(1) | 完整推导：C/16 压缩 → full PAM → window → low-rank（Q/K/V 形状、共享投影 P、N_w×r 的 attention、softmax 方向、无 scaling、α 残差、window 合并、M 截断）→ grouped CAM → g 融合、1×1 投回 C；写明在 exact 极限下等价于 DA-TransUNet 的 block |
| 4 | Gate Collapse 降级 | 删除 Eq.(5)、Fig.6，以及 “structural / gradient symmetry / fundamental property” 这些说法；只保留一句 empirical observation；g 统一写成 per-channel 向量 |
| 5 | GCS 降级 | 全文不再使用 GCS，改为 empirical window sensitivity（保留 S_window 定义），明确它不是 predictor、也不是 task property；Related work 里两处默认 “任务需要多少上下文” 的句子也已改写 |
| 6 | Efficiency claim | 删除原 Table IV（混用 1/2 GPU）和 DDP 讨论；390× 改为 “theoretical reduction of the attention-matrix term”；新 Table II 用同一工具、同一输入做静态统计（params / GFLOPs / attention 块 GFLOPs），写明不主张整网加速。这些数字不依赖训练，不需要重跑 |
| 7 | Matched DA-TransUNet（文字） | 主表分为 literature（†，仅供参考）和 re-trained 两部分；所有 claim 改为 “improves over DA-TransUNet under matched evaluation”；写明 DA 原文协议（256、Adam、500/50 epochs）和我们的差异 |
| 9 | SOTA / metric claims | 删除所有 SOTA 措辞（包括 Related work）；主动写出 Synapse HD95 和 ISIC mIoU 的退步 |
| 10 | Reporting（文字部分） | g 统一为 per-channel；Kvasir/ISIC 不报告 HD95；window ablation 给出原始数值（Table III） |
| — | 标题 | 保留原结构，只把 GCS 换掉：“ApproxDA-TransUNet: Understanding Window Sensitivity of Attention Approximation for Medical Image Segmentation”（审稿人没有要求改标题，改动是为了和正文保持一致） |
| — | 代码 | GroupedCAM 对齐 DA 原版；新的 c16 block（exact 极限下已验证和 DANetHead 等价）；删除从没调用的块；legacy block 已备份；新增 `--block_version`、`--rank 0`（详见 ISBI 追踪文档 §5） |

**保留的图**：Fig.1 overview、Fig.2 block、Fig.3 qualitative、Fig.4 DSC-vs-M 曲线、Fig.5 PAM contribution map。**删除的图**：原 Fig.6 gate entropy。

### B. ⬜ 待完成（需要实验 / 修图）

**B1. 重跑前的准备工作**

| 项目 | 状态 | 说明 |
|---|---|---|
| `--optimizer {sgd, adam}` | ✅ 脚本已改 | ApproxDA 和 DA-TransUNet 的 `train.py` / `test.py` 都已支持。adam 的 snapshot 目录加 `_adam` 后缀，`test.py` 要传同样的参数才能找到模型 |
| 防止用 test 做选择 | ✅ 脚本已改 | `--val_interval > 0` 时只会在 `val.txt` 上验证；Synapse、或没有 `val.txt` 的数据集会直接报错退出。`test.py` 默认加载**最后一个 epoch**（`--checkpoint last`），不再自动读取可能残留的 `best_model.pth`；新增 `--split val`（只在有 `val.txt` 时可用）。`test.py` 的 `--vit_name` 默认值改为 `R50-ViT-B_16`，和训练一致 |
| Kvasir list | ✅ 已修复并同步 | 原来的 list 有 **1 张图同时出现在 train 和 test**（`cju8bh8surexp0987o5pzklk1`，test 是 121 张），已从 test 删掉。现在是 880 / 120，没有重叠，合计 1000 张。DA-TransUNet 的 list 已换成同一份 |
| Synapse list | ✅ 已确认 | 两个项目的 list 完全相同 |
| `generate_lists.py` | ✅ 脚本已改 | 已有 list 时拒绝覆盖（需要 `--force`）；新增 ISIC 官方划分模式 `--isic_official_root`（生成 train/val/test 三个 list，并把官方文件整理到 `images/` 和 `masks/`）；写完后自动同步到 `DA-TransUNet/lists/`。已在临时目录里测试过 |
| ISIC 官方数据 | ⬜ 待下载 | 下载官方 Task 1 的 Training / Validation / Test Input 和 GroundTruth 六个文件夹，然后运行下面的命令 |
| 数据路径 | ✅ 已统一 | 两个项目现在都读 **`experiments/data/`**（在项目目录里写作 `../data`）。原来 DA-TransUNet 的 `train.py` 读仓库根目录下的 `data/`（`../../data`），而它的 `test.py` 读 `experiments/data/`，训练和测试用的不是同一个位置；`generate_qualitative_figure.py`、`analyze_attention_maps.py`（Synapse）和 `GENERATE_QUALITATIVE_FIGURE.md` 也一并改了。**训练机上要把数据放到 `experiments/data/` 下** |
| DA-TransUNet 在 Synapse 上只有 72.03% | ⬜ 待排查 | 旧的重跑结果远低于原文的 79.80。一个可疑线索：当时 DA 的训练读 `../../data/Synapse/train_npz`，测试读 `../data/Synapse/test_vol_h5`，如果这两个位置的数据版本或预处理不同，就可能导致这个问题（路径现已统一）。重跑前可以先检查训练机上两个位置的数据是否一致 |

命令示例（在对应项目目录下运行）：
```bash
# ISIC 官方划分（只需一次；会覆盖旧的随机 80/20 list，并同步到 DA-TransUNet）
python datasets/generate_lists.py --dataset ISIC --data_dir ../data/ISIC2018 \
    --isic_official_root /path/to/ISIC2018_official --force

# Synapse：SGD 0.01，300 epochs，取最后一个 epoch
python train.py --dataset Synapse --max_epochs 300 --window_size 28 --rank 32 --gate_mode pam
python test.py  --dataset Synapse --max_epochs 300 --window_size 28 --rank 32 --gate_mode pam

# Kvasir / ISIC：Adam 1e-3，300 epochs
python train.py --dataset Kvasir --max_epochs 300 --optimizer adam --base_lr 0.001 --window_size 56 --gate_mode pam
python test.py  --dataset Kvasir --max_epochs 300 --optimizer adam --base_lr 0.001 --window_size 56 --gate_mode pam

# ISIC：先在 val 上比较各个配置，选定后再在 test 上只跑一次
python test.py --dataset ISIC --max_epochs 300 --optimizer adam --base_lr 0.001 --window_size 28 --gate_mode pam --split val
```
DA-TransUNet 用同样的参数（不需要 `--window_size`、`--gate_mode` 等 ApproxDA 专用参数）。

**B2. 要跑的实验**（协议：300 epochs，取最后一个 epoch，224 输入；Synapse 用 SGD 0.01，Kvasir/ISIC 用 Adam 1e-3）

| 实验 | 对应清单 | 回填到论文 |
|---|---|---|
| DA-TransUNet 在三个数据集上重跑（matched） | #7 | Table I 的 re-trained 行，Table III 最后一行 |
| ApproxDA，PAM-only，r=32，M ∈ {7, 28, 56, 112}，三个数据集 | #1、#2 | Table III（window sensitivity），Table I 的 ApproxDA 行 |
| Rank：M=112，PAM，r ∈ {16, 32, 64, 0}；window-only：M=28，r=0（Synapse） | #2 | Table IV 的 rank / window-only 块 |
| Groups：CAM-only，G ∈ {1, 4, 8}（Synapse） | #2 | Table IV 的 groups 块，以及正文 `\todo{grouping result}` |
| Routing：M=7，r=32，{PAM, CAM, fixed 0.5, learned}（Synapse） | #2、#4 | Table IV 的 routing 块 |
| 主对比跑 3 seeds（DA-TransUNet + 最佳 ApproxDA） | #8 | Table I 的 mean ± std |
| Synapse per-organ DSC + paired test / bootstrap CI | #8 | 正文 `\todo`（可以用 `experiments/statistical_validation.py`） |
| 可选：learned gate 的 per-channel 分布或轨迹 | #4 | 正文 `\todo`（可选） |

**B3. 跑完后要回填和更新的内容**

| 项目 | 说明 |
|---|---|
| 所有红色 `\tbd{}` 数字 | 包括摘要、Table I / III / IV、正文里的差值和倍数（例如 +1.73、3.6–4.6×、0.64 / 2.30 这些 range）、各数据集选定的 M |
| 正文里的结论方向 | “exceeds / worse / below / not improve” 这些词都标了 `\tbd`，要按新结果核对，结论反过来的话要改写 |
| 硬件 | `\tbd{one NVIDIA Tesla T4}`，按实际训练环境填写 |
| 🖼 Fig.2 | 按 c16 block 重画：C/16 压缩、3×3 conv 块、gate 维度标 C/16、去掉外层残差（ISBI 追踪文档 §4.1 有详细说明） |
| 🖼 Fig.3 | 用重跑后的 checkpoint，通过 `generate_qualitative_figure.py` 重新出图 |
| 🖼 Fig.4 | 用新的 Table III 重新计算 ΔDSC（相对于重跑的 DA-TransUNet） |
| 🖼 Fig.5 | 用新的 Kvasir checkpoint 重新生成 contribution map，62.3% / 34.2% 这组数字也要重算 |
| 定稿 | 删掉 LaTeX 里的 NOTE / TODO 注释，把 `\revisionmarkstrue` 改成 `false` |

---

## P0 — 必须修改

### 1. 明确 Validation / Test Protocol
**Reviewer 问题：**  
论文只写了 train/test split，同时说每 15 epochs evaluation 并选择 best checkpoint；但没有说明 checkpoint、M、r、G、gate 是在什么数据上选择的。Reviewer 明确担心 test-set leakage。

**必须改：**
- 明确每个 dataset 的 **train / val / test**。
- 写清楚：
  - checkpoint 用 val 选择；
  - M、r、G、gate 用 val 选择；
  - test 只在 configuration 固定后做最终 evaluation。
- 如果原实验实际上使用 test 选 checkpoint / hyperparameter：
  - **不能只改文字，必须重跑**；
  - 至少重跑主 baseline + final ApproxDA configuration。

**论文中要删/改：**
- 不再写模糊的 “best checkpoint selected by evaluating every 15 epochs”。
- 改成明确的 validation-based model selection protocol。

---

### 2. 修正 Ablation：禁止一次改多个变量
**Reviewer 问题：**  
当前 ablation 同时改变 M、r、gate；G 没有系统测试，因此无法支持“rank 不重要”“三个 approximation axes 独立”等结论。

**必须改：**
至少补一组真正的 **single-variable ablation**：

- **Window**：固定 r、G、gate，只变 M
- **Rank**：固定 M、G、gate，只变 r
- **Groups**：固定 M、r、gate，只变 G
- **Gate**：固定 M、r、G，比较
  - PAM only
  - CAM only
  - fixed g=0.5
  - learnable gate

**最低要求：**
- 删掉/替换当前 Table V 中 `M=14, r=64, gate=learn` 这种 confounded row。
- 不再用它证明 “rank is not the bottleneck”。

---

### 3. 修正 Method / Eq. (1)
**Reviewer 问题：**  
原公式维度不一致；即使当前版本已部分修正，完整 attention path 仍然没有写清楚。

**必须改：**
完整写出：
- input shape；
- Q / K / V；
- low-rank projection；
- attention score shape；
- normalization / softmax；
- value aggregation；
- output 如何恢复到原 token grid；
- 是否共享 projection。

**最低要求：**
让 reviewer 可以只看 paper 就复现 tensor flow，不再需要猜 implementation。

---

### 4. 降低 Gate Collapse Claim
**Reviewer 问题：**  
g≈0.5 + gradient symmetry **不能证明** PAM/CAM representation convergence，也不能证明这是 symmetric dual-branch gating 的 structural failure。当前 Figure 6 与正文 correlation 描述也不一致。

**必须改：**
- 删除：
  - “structural failure”
  - “gradient symmetry proves collapse”
  - “fundamental property of symmetric two-branch architectures”
- 删除或大幅弱化 Eq. (5) 的 causal interpretation。
- 改成 observation：

> *The learnable gate converged close to 0.5 in our Synapse experiments and did not outperform PAM-only routing.*

**如果来得及补：**
- gate trajectory
- per-channel gate distribution
- fixed g=0.5 control

**不建议现在花时间：**
- branch-Jacobian theory
- 大规模 mechanism proof
- 多 dataset Gate Collapse theory

---

### 5. 降低 GCS Claim
**Reviewer 问题：**  
GCS 是从同一组 window ablation 的 DSC range 定义出来，再用来“预测 window tuning importance”，存在循环定义；3 个 dataset 也不足以证明它是 task intrinsic property。

**必须改：**
把：

> Global Context Sensitivity (GCS) as a task-level predictor / intrinsic property

改成：

> **empirical window sensitivity**

保留：

`S_window = max_M DSC(M) - min_M DSC(M)`

但只解释为：

> 当前 architecture 在该 dataset 上对 window size 的 empirical response。

**删除：**
- “predicts contextual requirement”
- “intrinsic property of the task”
- “high-GCS task requires long-range reasoning” 这类强因果语言

可以保留较弱表述：

> *The observed sensitivity is consistent with different contextual demands across datasets.*

---

### 6. 修正 Efficiency Claim
**Reviewer 问题：**  
当前 full model：
- Params 更多；
- GFLOPs 更高；
- PAM latency 没更快；
- 训练时间比较混用了 1 GPU / 2 GPU；
- 390× 只是 attention-matrix term，不是 whole-model speedup。

**必须改：**
- 不再宣称 **whole-network efficiency / acceleration**。
- 把 390× 改成：

> theoretical reduction of the attention-matrix term

- 明确：

> CNN–ViT backbone dominates end-to-end computation, so ApproxDA does not claim whole-network acceleration.

**最好：**
- 删除当前 DDP / multi-GPU “efficiency advantage” 讨论。
- 如果保留 efficiency table，只保留公平、同硬件条件下的数据。
- 若时间不够，宁可删整张 Table IV。

---

### 7. 统一重跑 / 重新表述 DA-TransUNet Baseline
**Reviewer 问题：**  
部分 baseline 数值来自 literature，split / preprocessing / training protocol 不一致；尤其 ISIC 的 DA-TransUNet 是否重跑不清楚。

**必须改：**
- 明确每个 DA-TransUNet 数值是：
  - **rerun**
  - 还是 **literature**
- 至少保证主要 claim 使用 **matched DA-TransUNet rerun**。
- matched protocol 至少一致：
  - split
  - preprocessing
  - input size
  - pretraining
  - optimizer
  - epochs
  - model selection

**论文 claim 改成：**
> improves over DA-TransUNet under matched evaluation

不要再依赖跨论文不统一的 SOTA comparison 作为主要证据。

---

## P1 — 强烈建议，但时间不够可简化

### 8. Multi-seed
**Reviewer 问题：**  
当前主结果多为 single run，提升幅度只有约 0.7–1.7 Dice，可能受随机性影响。

**建议：**
至少对：
- DA-TransUNet
- best ApproxDA

跑 **3 seeds**，报告 mean ± std。

**如果时间实在不够：**
优先 Synapse + 一个 2D dataset，不必所有 ablation 都跑 3 seeds。

---

### 9. 收紧 SOTA / Metric Claims
**必须文字修正：**
- “state-of-the-art across all three benchmarks” → 删除
- 改为 metric-specific：
  - highest DSC among compared methods
  - improves Dice over DA-TransUNet
- 主动承认：
  - Synapse HD95 变差
  - ISIC mIoU 没提升

---

### 10. 修正 Reporting 小问题
快速修：
- gate 是 scalar 还是 per-channel vector，统一 notation
- Figure 6 correlation 数值与正文一致
- Kvasir HD95 没有 physical calibration 时不要写 mm
- Kvasir / ISIC window ablation 给 raw numeric values，不只画 curve

---

# 最低可接受 Workshop Revision

如果时间非常紧，只做下面 7 件：

1. **明确 train/val/test + model selection**
2. **重做 clean single-variable ablation**
3. **修正完整 attention formulation**
4. **Gate Collapse 降级为 empirical observation**
5. **GCS 降级为 empirical window sensitivity**
6. **删掉 whole-model efficiency claim / DDP advantage**
7. **matched DA-TransUNet rerun + 明确 baseline 来源**

这 7 项是 reviewer 最集中的问题，也是最可能再次影响 workshop decision 的部分。

---

# 不建议现在投入时间的内容

除非以上 P0 已全部完成，否则不要优先做：

- 新增大量 dataset
- 独立 global-context predictor
- Gate Collapse 完整理论证明
- 大量 feature-map visualization
- 多个新 baseline
- 大篇幅 qualitative analysis
- 新 architecture 组件

这些更适合后续 ICME / journal 扩展，而不是当前 workshop rescue revision。

---

# 建议的 Workshop 版核心 Story

> **ApproxDA-TransUNet provides a controlled attention-approximation framework for medical image segmentation. Under matched evaluation, different datasets exhibit markedly different empirical sensitivity to spatial approximation, and appropriately selected windowing can improve Dice over the original DA-TransUNet.**

这个版本只保留当前数据真正能支持的结论：

- controlled approximation
- dataset-dependent window sensitivity
- matched improvement over DA-TransUNet

不再承担：
- universal GCS theory
- structural Gate Collapse theory
- whole-model efficiency claim
