# ApproxDA-TransUNet：BIBM 2026 审稿意见 → ISBI 2027 修改追踪

本文档合并了原来的 `BIBM_2026_Review_Comments_Categorized.md`（审稿意见）和 ISBI 精简方案，最后更新于 2026-09-24。

- **ISBI 源文件**：`ISBI_2027/ApproxDA-TransUNet.tex` + `ISBI_2027/sections/`（spconf 模板）。目前 6 页，含 22 条参考文献。**页数暂不处理**，等实验结果出来再压到 4 页 + 1 页参考文献。
- **标记约定**：红色 `\tbd{}` 是待重跑后替换的数字；红色 `\todo{}` 是依赖实验的文字。定稿时把 `\revisionmarkstrue` 改成 `\revisionmarksfalse`。
- **图例**：✅ 已完成 ｜ ⬜ 待实验 ｜ 🖼 待修图 ｜ ❓ 待决定

**结论：不需要实验就能修的部分已全部完成。** 剩下的是实验（§3，Phase 1：E1–E8 回应审稿意见；Phase 2：E9–E12 模型轻量化）、Fig.2 重画（§4.1）和 Fig.3 重新出图（§6）。代码已改为 c16 block（§5.4）。

---

# 1. ISBI 主线

**主 story**：Controlled attention approximation reveals substantial dataset-dependent window sensitivity in medical image segmentation. Under matched evaluation, ApproxDA-TransUNet with a task-appropriate window improves Dice over DA-TransUNet, while the degree of sensitivity varies considerably across datasets.

**只保留两条 contribution**：
1. **Controlled ApproxDA framework**：用 M、r、G 分别控制 spatial scope、affinity resolution 和 channel interaction，每次只变一个。它的定位是实验框架，不主张整网效率。
2. **Dataset-dependent window sensitivity**：Synapse 对 window size 明显敏感，Kvasir 和 ISIC 相对稳定；选对 window 后，三个数据集上都超过 DA-TransUNet。

**Claim 收紧（全部 ✅）**

| BIBM 原 claim | ISBI 版本 |
|---|---|
| state-of-the-art across three benchmarks | improves Dice over DA-TransUNet under matched evaluation；按指标分别说明 |
| approximation 优于 full dual attention | 改为优于 **DA-TransUNet**，并写明这个比较同时改变了 approximation 和 routing（PAM-only 去掉了 CAM）；approximation 本身的作用由 exact-PAM 对照行单独衡量 |
| GCS predicts contextual requirement / 是 intrinsic task property | 改为 **empirical window sensitivity**（DSC range），明确不是独立的 predictor |
| High-GCS task requires global context | observed sensitivity is consistent with different contextual demands |
| task-aligned inductive bias | results suggest a locality-induced bias |
| Gate Collapse 由 gradient symmetry 导致，是 structural limitation | 删除；只留一句 “learned gate converged near 0.5 in our Synapse experiments and did not outperform PAM-only” |
| Efficient segmentation framework / 390× | attention approximation framework；390× 只用于 attention matrix，明确不主张整网加速 |

---

# 2. 逐条审稿意见与处理状态

### R1. Model-selection protocol unclear（Must fix）
> Paper only describes train/test splits, but says the best checkpoint is selected by evaluation every 15 epochs. It is unclear which data are used to select checkpoint, M, r, G, and gate mode; if test data are used, results may have optimistic bias.

**已确认存在 leakage**：`--val_interval>0` 时，`_validate_synapse` / `_validate_2d` 读的是测试集，而 test.py 会优先加载 `best_model.pth`。例如 ISIC M=28 的 89.55% 就是训练过程中测试集上的最高分，从没单独跑过 test。
- ✅ checkpoint：一律取最后一个 epoch，不做 checkpoint 选择（和 TransUNet 一致）
- ✅ M/r/G/gate：ISIC 用官方 val 选；Synapse/Kvasir 没有 val，从消融（test）里选，在 Model selection、discussion 和 Limitations 里如实披露，并给出偏差上限（§3.0）
- ⬜ 按 §3.0 的协议全部重跑

### R2. Baseline comparisons may not be fair（Must fix）
> Most baseline values appear to come from previous papers under potentially different splits/training protocols. In particular, it is unclear whether DA-TransUNet on ISIC was rerun under the same 80/20 split.

- ✅ 主表分成 literature（†，仅供参考）和 re-run（matched）两部分；统一写明 split、增强、输入、预训练、optimizer、epochs、硬件
- ⬜ 在同一 protocol 下重跑 DA-TransUNet（三个数据集）。注意：旧的 Synapse re-run 只有 72.03% DSC（原论文报告 79.80%），需要排查；ISIC 表里 DA† 的 88.88/82.78 来源需要核实

### R3. Ablations do not isolate individual factors（Must fix）
> Several experiments change more than one variable simultaneously; r is changed together with M/gate, G is not systematically tested, and M=112 is still not true full attention because low-rank projection remains.

- ✅ 表格改为一次只变一个因素，删掉 M14/r64/learn 这类混淆行；注明 M=112 仍是 low-rank，不是 full PAM
- ✅ G 的 ablation 改在 CAM-only 下做（PAM-only 时 CAM 的输出被丢弃，改 G 没有效果）
- ✅ 覆盖审稿人列出的全部对照组：full attention = M=112 no projection（exact global PAM）；window-only = M=28 no projection；low-rank-only = M=112 r=32；group-only = CAM-only 下变 G；PAM-only / CAM-only / fixed / learned 在 routing block 里
- ✅ 代码修复后 G=1 严格等价于 DA-TransUNet 的 CAM（§5.1）
- ⬜ 待跑：r=16、no projection、G=1/4、fixed g=0.5（见 §3）

### R4. Eq.(1) dimensionally inconsistent, attention incomplete（Must fix）
> Given the stated tensor shapes, W_rCᵀ and W_rDᵀ are invalid; value projection and reconstruction to the original token grid are also unclear.

- ✅ 保留原稿的推导结构（full PAM → window → low-rank；CAM → grouped CAM），按代码写全：C/16 压缩、query/key/value 的维度（C′=max(1, C/8)，和 DA 一致）、共享 token 投影 P、N_w×r 的 attention 矩阵、softmax 方向、无 scaling、α/β 残差、window 合并、M 截断、r=0 表示 exact attention、g 融合 + 1×1 投回 C；并写明在 exact 极限下 block 等价于 DA-TransUNet 的 block
- pseudocode 放到 journal 版本，ISBI 不加

### R5. Gate Collapse not established by Eq.(5)（Must fix）
> Equal gate coefficients at g=0.5 do not imply equal parameter updates or representation convergence because PAM and CAM have different computations/Jacobians. Current evidence is also mainly from Synapse.

- ✅ 降级为一句 empirical 观察；删除 Eq.(5)、Fig.6 和 causal / structural 的说法。完整的机制分析留给 journal 版本

### R6. GCS does not independently predict contextual requirement（Strongly recommended）
> It is defined from the Dice range over the same window-size experiments that it is then used to explain, and three datasets are insufficient to establish it as an intrinsic task property.

- ✅ 改为 empirical window sensitivity，明确不是 predictor；Limitations 里写明
- 更多数据集（CVC、ACDC 的结果已有）放到 journal 版本

### R7. Efficiency claim does not hold at full-model level（Must fix）
> The proposed model has more parameters and higher GFLOPs than DA-TransUNet; the ~390× reduction applies only to the attention-matrix term. Training-time comparison mixes 2-GPU and 1-GPU settings.

- ✅ 删除 full-model efficiency 表和 DDP 讨论；写明所有模型在同一硬件上训练
- ✅ 审计查明：legacy 版比 DA-TransUNet 重，原因是没做 C/16 压缩、有闲置块，和近似本身无关（§5.3）。改成 c16 block 后比 DA-TransUNet 轻（105.99M / 29.73 GFLOPs 对比 107.95M / 30.20；attention 块 0.40 对比 0.87 GFLOPs），论文 Method 里如实写了整网和 attention 块的数字，以及 ViT 的占比
- ⬜ 可选：block-level 的 latency / memory（同一 GPU）

### R8. Statistical reliability insufficient（Strongly recommended）
> Main gains are based on single runs; Synapse has only 12 test cases, and differences of ~0.5–2 Dice points may be affected by initialization, data order, or checkpoint selection.

- ✅ 正文预留了 3 seeds、per-organ、paired test / bootstrap 的 `\todo`
- ⬜ 3 seeds（mean ± std）、Synapse per-organ 表、统计检验（可用 `experiments/statistical_validation.py`）

### R9. Claims / reporting too strong or inconsistent（Strongly recommended）
> “State-of-the-art” is not true across all metrics; Synapse HD95 and ISIC mIoU regress; Figure 6 correlation values conflict with the text; gate described inconsistently as scalar/vector; Kvasir HD95 should not be reported in mm without calibration.

- ✅ 删除 SOTA 措辞，按指标分别说明；正文主动写出 Synapse HD95 和 ISIC mIoU 的退步；Fig.6 已删；ablation 给出原始 DSC；Kvasir/ISIC 不报告 HD95；敏感度倍数改为准确的 3.6–4.6×；正文 gate 统一为 per-channel 向量 g
- 🖼 Fig.2 需要按 c16 架构重画（包括把 gate 标注从 “1” 改为 C/16），见 §4.1

---

# 3. 重跑实验清单

### 3.0 训练协议对齐原文（2026-09-24 查证）

**已决定（Synapse）**：沿用 TransUNet 的协议（224、SGD 0.01、batch 24、½CE+½Dice、18/12 划分），**训练长度改为 300 epochs**（比原文的 150 略长）。**没有 eval 集，一律取最后一个 epoch 的 checkpoint**。DA-TransUNet 按完全相同的协议重跑，所以主要对比是 matched 的；literature 行因为训练长度不同，只作参考。

**已决定（Kvasir / ISIC）**：尽量对齐 DA-TransUNet 的协议（Adam lr 1e-3、wd 1e-4、½BCE+½Dice），**训练长度统一 300 epochs**；数据集如果有官方 eval 集就用官方的。
- **Kvasir-SEG**：没有官方的 train/val/test 三分。Kvasir-SEG 原论文推荐 880/120，**仓库现在的 list 就是 880/120**（commit `7278690`，2026-08-30）。建议用它作为 train/test，和 Synapse 一样取最后一个 epoch。⚠️ BIBM 的 Kvasir 实验实际用的是 `generate_lists.py` 生成的 **800/200**（日志里是 200 个 test iteration），和现在的 list 不一致，论文里写的也是 800/200。重跑后以 880/120 为准，论文要同步修改。
- **ISIC 2018 Task 1**：官方有 **train 2594 / val 100 / test 1000**，都有标注。用官方 val 选 M/r/G/gate，最后在官方 test 上只跑一次。现在的 list 是从 2594 张训练图里随机分 80/20（2075/519），需要重新生成；Kaggle 上的数据集要确认有没有包含官方 val 和 test 的标注。
- **loss**：DA 用的是单通道的 ½BCE+½Dice，我们用 2 类 softmax CE + Dice。2 类 softmax CE 和对 logit 差做 BCE 在数学上等价，可以不改。
- **学习率调度**：原文没写 Adam 的调度方式，沿用我们的 poly decay。batch 原文也没写，沿用 24。
- ✅ **输入尺寸：三个数据集都用 224**（已决定）。DA 用的是 256，但窗口大小 M 必须整除特征图边长：256 输入时特征图是 16/32/64/128，M=7 会直接报错，和 Synapse 的 M ∈ {7, 28, 56, 112} 对不上。window 敏感度的跨数据集对比是 ISBI 的核心结论，所以选择统一用 224，论文里写明和 DA 原文的这处差异及原因。
- ✅ **Kvasir 划分：880/120**（已决定）。

**Kvasir / ISIC 最终协议**：224 输入，Adam lr 1e-3（β=(0.9, 0.999)），wd 1e-4，poly decay，batch 24，2 类 CE + Dice，300 epochs。Kvasir：880/120，取最后一个 epoch。ISIC：官方 2594 / 100 / 1000，用 val 选超参数，最后在 test 上只跑一次。

**需要的代码和数据准备（重跑前）**：✅ 脚本部分已完成（2026-09-24）。详细说明和命令示例见 `ApproxDA_BIBM_Workshop_Must_Fix.md` 的 B1。
- ✅ 两个项目都加了 `--optimizer {sgd, adam}`（adam 的 snapshot 目录加 `_adam` 后缀）。
- ✅ `--val_interval` 只会在 `val.txt` 上验证，没有 val 的数据集直接报错；`test.py` 默认加载最后一个 epoch，新增 `--checkpoint` 和 `--split val`。
- ✅ Kvasir list 修复了一张 train/test 重叠的图（test 从 121 张变成 120 张），并已同步到 DA-TransUNet；Synapse list 两边一致。
- ✅ `generate_lists.py` 有防覆盖保护，新增 ISIC 官方划分模式，并会自动同步到 DA-TransUNet。
- ⬜ ISIC 官方 val/test 数据需要下载后运行生成命令；⬜ 先排查 DA-TransUNet 在 Synapse 上只有 72% 的原因。

**待决定**：
- ✅ **超参数选择（已决定）**：ISIC 用官方 val 选 M/r/G/gate。Synapse 和 Kvasir 没有 val，按 TransUNet / DA-TransUNet 的惯例**从消融结果（test）里选最好的组合**，并在论文里如实说明：
  - Model selection 段落和主表脚注写明选择方式，并说明消融表列出了所有配置；
  - Window sensitivity 段落（discussion）：低敏感度数据集选哪个 M 都差不多（最多差 0.64），Synapse 本来就需要仔细选窗口；
  - Limitations：写明这两个数据集的结果可能偏乐观，偏差上限就是 window range（Kvasir 约 0.64，Synapse 最多约 2.30），并建议实际使用时在 held-out 数据上调 M。
  - 注意：R1 的回应因此只算部分解决（ISIC 完全解决；Synapse/Kvasir 靠如实披露），rebuttal 或 cover letter 里要主动说明。

目的：协议和原文一致，literature 行的数字才能直接拿来比。依据是 `Reference/DA-TransUNet.pdf` §4.2、`Reference/transunet.pdf` §4.2，以及 `experiments/DA-TransUNet/train.py` 的默认值。

| 项目 | TransUNet / DA-TransUNet：**Synapse** | DA-TransUNet：**Kvasir / ISIC 等 5 个数据集** | 我们现在（BIBM） |
|---|---|---|---|
| 输入 | 224×224，patch 16 | **256×256**，patch 16 | 224×224 |
| 优化器 | SGD，lr 0.01，momentum 0.9，wd 1e-4，poly decay | **Adam，lr 1e-3**，momentum 0.9，wd 1e-4 | SGD 0.01 |
| batch | 24 | 原文没写 | 24 |
| 训练长度 | **14k iterations ≈ 150 epochs**（代码默认 `max_epochs=150`） | **500 epochs；ISIC 和 Chest X-ray 只训 50 epochs** | 300 epochs |
| loss | ½CE + ½Dice | ½BCE + ½Dice | ½CE + ½Dice |
| 划分 | 18 / 12（固定 list） | **随机 75% / 25%**（3:1），没有公开 split list | Kvasir 800/200、ISIC 2075/519（80/20） |
| checkpoint | 最后一个 epoch，没有 val | 原文没写 | 训练中在 test 上选 best（leakage） |
| HD95 | medpy，**没传 spacing**，所以单位其实是像素，只是按惯例写成 mm | — | 同左 |

**结论**：
- **Synapse**：只要把 epoch 改成 150、改用最后一个 epoch，就和整个 TransUNet 系列的协议完全一致，Table 1 里所有 Synapse 的 literature 数字都能直接比。**但这和现有的 “一律 300 epochs” 规则冲突**，需要决定。
- **Kvasir / ISIC**：DA-TransUNet 用的是另一套协议（256、Adam、500/50 epochs、随机 75/25），我们现在的设置和它完全不同，所以 BIBM 表里 DA†（88.47 / 88.88）以及 U-Net、TransUNet 等 literature 行（都来自 DA-TransUNet 的 Table 2）都不能直接比。即使超参数全部对齐，原文的随机划分没有公开，也做不到严格一致，只能 “按原文协议重跑 DA-TransUNet + 我们的模型”，literature 行仍然只作参考。
- **R1 的 val 要求和 “最后一个 epoch” 可以同时满足**：先在 train 里划出 val，用它选 M/r/G/gate；选定后用**完整的 train** 按原文协议重新训练，取**最后一个 epoch**，test 只跑一次。这样既没有 leakage，又和原文协议一致（原文也是全 train、取最后一个 epoch）。

**通用设置**：同一硬件、只用 val 选超参数、test 只跑一次。epochs、输入尺寸、优化器和划分按 §3.0 的决定来定。

**代码参数（§5.4）**：ApproxDA 默认 `--block_version c16`（新的 C/16 block），snapshot 目录自动加 `_c16` 后缀，不会读到旧目录里的 checkpoint。`--block_version legacy` 是 BIBM 版的全通道 block。“no projection” 用 `--rank 0`。`--gate_mode pam` / `cam` 只会建用到的那个分支。

| # | 实验 | 数据集 | 用于 |
|---|---|---|---|
| E1 | DA-TransUNet re-run（先排查 Synapse 只有 72% 的原因） | Synapse / Kvasir / ISIC | Table 1、2 |
| E2 | ApproxDA，PAM-only，r=32，M ∈ {7, 28, 56, 112} | Synapse / Kvasir / ISIC | Table 2（window sensitivity） |
| E3 | Rank：M=112，PAM-only，r ∈ {16, 32, 64, 0 (no projection)}；另加 window-only：M=28，PAM-only，r=0 | Synapse | Table 3 |
| E4 | Groups：CAM-only，G ∈ {1, 4, 8}（c16 block 里的 CAM 只作用在 C/16 通道上：768→48、512→32、256→16、64→4，所以 64 通道处的 G 会被截到 4） | Synapse | Table 3 |
| E5 | Routing：M=7，r=32，G=8，{PAM, CAM, fixed, learned} | Synapse | Table 3 |
| E6 | 主对比跑 3 seeds（DA-TransUNet 和各数据集的最佳 ApproxDA） | Synapse / Kvasir / ISIC | Table 1 mean ± std |
| E7 | Synapse per-organ DSC + paired test / bootstrap | Synapse | R8 |
| E8 | 可选：block-level FLOPs / memory / latency | — | R7 |

**Phase 2：模型轻量化（等 E1–E8 跑完再开始，详见 §7）**

| # | 实验 | 数据集 | 前置工作 |
|---|---|---|---|
| E9 | ViT-B 深度：保留前 {12, 8, 6, 4} 层 | Synapse | 加 `--vit_layers` 参数（现有权重可直接用） |
| E10 | decoder 通道减半 (128, 64, 32, 16)，叠加在 E9 最好的深度上 | Synapse | 加 `--decoder_channels` 参数 |
| E11 | R50 + ViT-S/16 ×12（可选 ×6） | Synapse | 下载 AugReg ViT-S/16 i21k 权重；`load_from` 改成合并两份 npz（R50 用现有权重，ViT 用 ViT-S 权重，patch embedding 随机初始化） |
| E12 | 最佳轻量配置在另外两个数据集上验证，并给 DA-TransUNet 换同样的 backbone 做对照 | Kvasir / ISIC（+ Synapse 的 DA 对照） | E9–E11 的结果 |

E9–E12 的 M、r、G 和 gate 沿用 Phase 1 在 val 上选出的最佳组合，不必固定为 M=28、r=32、pam。

⚠️ 默认 block 已换成 c16（§5.4），**所有旧 ApproxDA 结果都不能直接用于 ISBI**（旧结果属于 legacy 架构）。分析脚本（`analyze_gate_entropy.py`、`generate_gate_figures.py`、`analyze_attention_maps.py`、`generate_qualitative_figure.py`）如果要读旧 checkpoint，需要设 `config.block_version = "legacy"`，并用 `strict=False` 加载，因为旧 checkpoint 里还有那 9 个已删除的闲置块。

---

# 4. 待修图

repo 里没有 `approxda_block.pdf`（Fig.2）的源文件；Fig.1（`approxda_overview.pdf`）由 `paper/figures/generate_overview.py` 生成。

### 4.1 Fig.2 ApproxDABlock：按 c16 架构重画

代码见 `experiments/ApproxDA-TransUNet/Architecture/block.py` 中的 `ApproxDABlock`。以 512 通道的 skip 为例（图中数字沿用现在的写法，每个方块下面标通道数）：

```
x (512) ─┬─ [3×3 Conv + BN + ReLU] 512→32 ─ Window Partition (M) ─ Low-Rank Proj. (r) ─ PAM Attention ─ Window Reverse ─ [3×3 Conv + BN + ReLU] 32→32 ─┐
         │                                                                                                                                       ├─ Gate g (32, per-channel) ─ [Dropout + 1×1 Conv + ReLU] 32→512 ─ out (512)
         └─ [3×3 Conv + BN + ReLU] 512→32 ─ Group Split (G) ─ Channel Attention ────────────────────── [3×3 Conv + BN + ReLU] 32→32 ─┘
```

和现在这张图相比要改的地方：
1. **两个分支入口各加一个 “3×3 Conv + BN + ReLU, C→C/16” 方块**（对应代码里的 `conv5a` / `conv5c`）。分支内部的方块（Window Partition、PAM Attention、Window Reverse、Group Split、Channel Attention）维度从 512 改成 **32**（=C/16）。
2. **两个分支出口各加一个 “3×3 Conv + BN + ReLU, C/16→C/16” 方块**（`conv51` / `conv52`）。
3. **Gate 方块的维度从 “1” 改为 “32”（C/16）**，并注明 per-channel。learned 模式下 `g = sigmoid(Linear(C→C/16)(GAP(x)))`，依赖输入；pam / cam / fixed 模式下是常数 1 / 0 / 0.5。
4. **“Fusion” 改成 “Dropout + 1×1 Conv + ReLU, C/16→C”**（`conv8`），输出 512。
5. **删掉最后的 “+x” 残差**：新 block 和 DA-TransUNet 一样没有外层残差。PAM/CAM 内部仍然有 α/β 残差，图里不用画。
6. Low-Rank Projection 旁边的 “32” 表示 r，保留；建议把 r 的标注和通道数区分开（例如写成 “r=32”），因为现在分支通道数也恰好是 32，容易混淆。
7. 可选：在 PAM Attention 旁边注明 query/key 为 C/16/8（512 通道时是 4），和 DANet 一致。
8. 图注已在论文里更新（C/16 压缩、g 是 per-channel、1×1 conv 投回 C）。

### 4.2 Fig.1 overview（可选）

图本身不需要改。ApproxDABlock 的 4 个位置（768@14²、512@28²、256@56²、64@112²）和代码一致。

---

# 5. 代码

### 5.1 ✅ GroupedCAM 对齐 DA-TransUNet 的 CAM_Module
`experiments/ApproxDA-TransUNet/Architecture/block.py` 原来有两处和原版不同，已修改：
1. **聚合方向（bug）**：原来是 `bmm(X.transpose(1,2), x_g)`，求和的维度没有被归一化。现改为 `bmm(X, x_g)`，每个输出通道的权重和为 1。
2. **能量项**：原来直接对 energy 做 softmax。现在和原版一样，先算 `max(energy) - energy`，等价于 softmax(−energy)。

验证：用随机张量对比，G=1 和 `CAM_Module` 的最大误差为 0.0；G=8 和逐组套用 `CAM_Module` 的最大误差也为 0.0。ISBI Eq.(4) 和 `paper-journal/sections/04_method.tex` 都已同步修改。

### 5.2 DA-TransUNet 和 ApproxDA 在结构上的差异（已查明，论文已如实描述）

**(a) 通道压缩有两层。** DA-TransUNet 原论文 §4.4.3 / Table 6 有相应说明，代码见 `DA-TransUNet/Architecture/block.py`：
1. **Block 级**：`DANetHead` 先用 3×3 conv + norm + ReLU 把 C 压到 **C/16**，再送进 PAM/CAM。DANet 原版是 C/4，而 DANet 是为道路场景设计的，DA-TransUNet 认为 C/4 不适合医学图像，就对这个超参做了 ablation。结果：C/1 为 78.55，C/2 为 79.35，C/4 为 79.71，C/8 为 79.35，**C/16 为 79.80**，C/32 为 79.71。论文给的理由是 “mitigate overfitting, optimize computational resources, focus on the most critical features”，这纯粹是经验性结论。而且除了 C/1 明显偏低，其余设置之间的差距都在 0.5 以内，都是单次运行。
2. **PAM 内部**：query/key 再压到 inter/8。这个做法沿用自 SAGAN（代码注释写着 “Ref from SAGAN”），DANet 论文的公式里并没有写。因此在 512 通道的 skip 上，实际是 512 → 32 → query/key 只有 4 维。

**(b) 64 通道 skip 的 bug**（commit `8c6423d`，2026-06-13）：DA-TransUNet 的 decoder 按 `x.size(1)`（decoder 特征的通道数）来选 DA 模块，所以 64@112² 那个 skip 永远不会调用 DA。用 hook 实测，**只有 encoder（768@14²）、512@28²、256@56² 这三处调用了 DA**。当时决定 DA-TransUNet 保持原样（作为公开 baseline），我们的版本修成按 `skip.size(1)` 判断，4 处都会调用。顺带一提：就算 DA-TransUNet 真的调用了 64 通道的块，64/16 = 4，4/8 = 0，query/key 的通道数会变成 0，直接报错。

**(c) 【已由 §5.4 解决，下面是当时的记录】legacy 版 ApproxDA 和 DA-TransUNet 的差异不只是 attention 近似**。c16 版之后，剩下的唯一结构差异是多了 64@112² 这个位置。legacy 当时的差异还包括：块作用在完整的 C 通道上而不是 C/16；query/key 是 C 维而不是 C/128；多了 64@112² 这个位置；融合方式不同（DA 用 conv51/52 + conv6/7/8 + dropout，ApproxDA 用 1×1 fusion）。
- 论文 Method §2.1 已如实写明：64 通道的 skip 是额外加的；DA-TransUNet 的 C/16 压缩在 ApproxDA 里没有。“drop-in replacement” 的说法已删除。
- 要把 approximation 本身的作用单独拿出来，依靠的是 E3 里 ApproxDA 自身的 no-projection 对照，而不是和 DA-TransUNet 比。
- 参数量：两边都有从没被调用的块。每个 decoder block 都建了 3 个 DA 模块，实际只用到 1 个。ApproxDA（M=112 配置）总共 118.18M 参数，其中 7.60M 属于从没被调用的块。这也是 DDP 需要 `find_unused_parameters=True` 的原因。如果将来报告参数量，应该去掉这部分。

### 5.3 Params / FLOPs 审计（2026-09-24，`experiments/audit_params_flops.py`）

口径和论文 / test.py 相同：fvcore，1×224×224 输入，一次乘加计为 1 FLOP，attention 的 bmm 也计入。总数和论文完全一致：DA-TransUNet 107.95M / 30.20 GFLOPs；ApproxDA（M=7，pam）112.98M / 32.07 GFLOPs；ApproxDA（M=7，learn）114.90M。

**各部分占比（DA-TransUNet → ApproxDA M=7 pam）**

| 组件 | Params (M) | Params 占比 | GFLOPs | GFLOPs 占比 |
|---|---|---|---|---|
| ViT encoder（12 层） | 85.06 → 85.06 | 78.8% → 75.3% | 17.37 → 17.37 | 57.5% → 54.2% |
| CNN backbone（R50） | 11.89 | ~11% | 3.96 | ~13% |
| Decoder convs | 7.39 | ~7% | 7.78 | ~25% |
| 实际调用的 attention 块 | 1.28 → 3.70 | 1.2% → 3.3% | 0.87 → 2.73 | 2.9% → 8.5% |
| 从没调用的 attention 块 | 1.40（10 个）→ 4.01（9 个） | 1.3% → 3.5% | 0 | 0 |

→ “ViT 占大部分”成立：ViT encoder 占 params 的 75–80%、FLOPs 的 54–58%。backbone 加 decoder 占 FLOPs 的 90% 以上。（更正：第一版审计脚本漏算了非叶子模块里直接执行的运算，ViT 的 FLOPs 曾被写成 16.67；脚本已修正，总数不受影响。）

**ViT 内部的拆分**：线性层（QKV、out-proj、MLP）占 16.65 GFLOPs（95.8%），全局注意力的 matmul（QKᵀ、AV）只占 0.71 GFLOPs（4.1%，约为整网的 2.4%）。参数方面，attention 的 Q/K/V/out 共 28.35M，MLP 共 56.67M。只有 196 个 token（14×14），所以全局注意力本身很便宜；ViT 的开销主要来自 hidden size 768 的线性层。

**增量从哪里来**
- **Params +5.03M**：从没调用的块贡献 +2.61M，实际调用的块贡献 +2.42M。ApproxDABlock 在完整的 C 通道上做三个 C×C 的 Q/K/V 1×1 conv，再加一个 C×C 的 fusion；DA 则先压到 C/16，所以块里的参数主要只是两个 3×3 C→C/16 conv。此外 `proj_r` 的大小是 M²×r，随 M 增长：M=112 时每个块的 `proj_r` 从 49×32 涨到 12544×32（约 0.4M），13 个块（含没调用的）一共多出约 5.2M，总数变成 118.18M。
- **GFLOPs +1.87**：几乎全部来自 PAM 的 Q/K/V 1×1 投影。ApproxDA 是 **1.734** GFLOPs，DA 只有 **0.003**，因为 DA 的 PAM 在 C/16 的特征上运行，query/key 又只有 C/128。attention 本身（算 attention 矩阵加加权求和）：DA 共 0.201，ApproxDA 共 0.276，其中 0.103 来自新增的 64@112² 位置；在三个共同位置上，ApproxDA 是 0.173，DA 是 0.201。
- 近似确实起了作用，但只在大分辨率上：256@56² 这一处，attention 本身从 0.177 降到 0.103。在 512@28² 和 768@14²，DA 的 full attention 因为通道已经压得很窄，本来就很便宜（0.022 / 0.002），我们在完整 C 通道上做的 low-rank attention 反而更贵（0.051 / 0.019）。
- **FLOPs 和 M 无关**（M=7 到 112 都是 32.07）：r 固定时，代价是 O(NrC)，与 M 无关。
- pam 模式下 CAM 仍然会被计算，只是结果被丢弃了，每次 forward 白白多算约 0.14 GFLOPs。

**结论**：params 和 FLOPs 增加，**不是 attention 近似造成的**，而是三个设计差异造成的：(1) 不做 C/16 压缩，在完整 C 通道上做 Q/K/V 投影；(2) 从没调用的块被计入了参数量；(3) 多了 64@112² 这个位置。论文里的 “N/r≈390×” 是相对于**同样通道数下的 full PAM** 算的，不是相对于 DA-TransUNet 实际的 PAM。

以上三项修复都已在 §5.4 完成。

### 5.4 ✅ 代码重构：删除闲置块 + legacy 备份 + C/16 block（2026-09-24）

**改动**（`experiments/ApproxDA-TransUNet/`）：
- `Architecture/block.py`
  - 原来的 block 改名为 `ApproxDABlockLegacy`，完整保留（全通道、1×1 fusion、外层残差）。
  - 新的 `ApproxDABlock` 按 DANetHead 的结构实现：`conv5a`/`conv5c`（3×3，C→C/16，BN，ReLU）→ `LowRankWindowedPAM` / `GroupedCAM` → `conv51`/`conv52`（3×3）→ g 融合 → `conv8`（Dropout2d 0.05，1×1 投回 C，ReLU）。**没有外层残差**。PAM 的 query/key 为 max(1, (C/16)/8)，和 DA 的 `PAM_Module` 一致。DA 里的 `conv6`/`conv7` 输出没被用到，不保留。pam / cam 模式只建用到的那个分支。
  - `LowRankWindowedPAM`：新增 `qk_channels`（legacy 仍然是全通道）；`rank<=0` 表示不投影，即窗口内 exact attention。
  - `GroupedCAM`：group 数最多等于通道数（C/16 后 64 通道处只剩 4 个通道）。
- `Architecture/ApproxDATransUNet.py`：`make_approx_block()` 根据 `config.block_version`（`c16` | `legacy`）选择 block。每个 `DecoderBlock` **只建它的 skip 用到的那一个块**，属性名仍然是 `da`/`da2`/`da3`。
- `train.py` / `test.py`：新增 `--block_version`（默认 `c16`）；c16 的 snapshot 目录加 `_c16` 后缀，legacy 的路径和旧的一样。
- `experiments/audit_params_flops.py`：同时审计 legacy 和 c16。

**验证**：
- **legacy 和 git HEAD 的旧代码等价**：加载旧权重后（丢掉 99 个闲置块的 key），M=7 / M=112 的输出误差都是 0。参数从 112.98M 降到 108.98M（M=7），从 118.18M 降到 110.58M（M=112）。
- **c16 能正常训练**：12 种配置（M ∈ {7, 28, 112}；r ∈ {0, 16, 32, 64}；G ∈ {1, 4, 8}；4 种 gate）下，4 个块全部被调用，**每个参数都有梯度**（没有闲置参数，DDP 不再需要 `find_unused_parameters`）。
- **c16 和 DANetHead 等价**：在 exact 设置下（窗口覆盖整张图、r=0、G=1、fixed g=0.5，conv8 权重 ×2），768 / 512 / 256 通道的输出误差都在 1e-7 量级（浮点精度）；参数量和去掉 conv6/7 之后的 DANetHead 完全相同。

**审计结果（fvcore，1×224×224）**：

| 模型 | Params | GFLOPs | attention 块 GFLOPs |
|---|---|---|---|
| DA-TransUNet | 107.95M（去掉闲置块后 106.56M） | 30.20 | 0.87 |
| legacy M=28 pam（已删闲置块） | 109.07M | 32.07 | 2.73 |
| **c16 M=28 r=32 pam** | **105.99M** | **29.73** | **0.40** |
| c16 M=7 r=32 learn（两个分支都在） | 106.51M | 30.08 | 0.74 |
| c16 M=112 r=0 pam（exact attention） | 105.89M | 30.71 | 1.37 |

→ c16 版的 ApproxDA **比 DA-TransUNet 更轻**，而且多了 64@112² 这一处。approximation 本身的效果现在也能直接看到：exact 和 r=32 相比，attention 块从 1.37 降到 0.40 GFLOPs。M=112 时 params 会多一些（105.99M → 107.50M），因为 `proj_r` 的大小是 M²×r。论文 Method 的 complexity 段落已按这些数字改写，并删掉了 “C′=C vs C/8” 和 “ApproxDA 作用在全通道上” 这两处已过时的描述。

---

# 6. 内容取舍记录

**已删除**：独立的 Related Work 小节（压缩成 Intro 第一段，22 条引用全部保留）、Gate Collapse 的因果分析、Eq.(5)、Fig.6（gate entropy）、full-model efficiency 表、DDP 讨论、bias–variance 类比、大段逐数据集的叙述。

**按用户要求保留**：Fig.1（overview，含 ViT inset，不和 Fig.2 合并）；Fig.2（block，改为通栏）；Fig.3（qualitative 对比，放在 Main results；🖼 **重跑后要用新的 c16 checkpoint，通过 `generate_qualitative_figure.py` 重新出图**，现在这张用的是 legacy checkpoint，而且 ISIC 那一行是 gate=learn、M=7）；Fig.5（PAM contribution map，改为描述性表述，不作因果证据）；PAM/CAM 公式推导；全部 literature baseline 行。

**审计中顺带修正的原稿问题**：CNN stride 原写 4/8/16，实际为 2/4/8；Fig.2 在单栏宽度下看不清，已改为通栏。

---

# 7. 模型轻量化探索（2026-09-24，已排期为 Phase 2，对应 §3 的 E9–E12；等 E1–E8 完成后开始）

**本地权重**：只有 `experiments/model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz`（约 440MB，被 `.gitignore` 的 `*.npz` 排除，换机器训练需要单独拷过去）。configs 里的路径是相对 `experiments/ApproxDA-TransUNet/` 的 `../model/...`。ViT-S / Ti 的权重本地没有，需要下载。

**目标**：让 “efficient” 的说法有实际数据支撑。审计（§5.3）表明，attention 块只占整网的 1–3%，开销主要在 backbone 和 decoder，所以轻量化必须从这两部分入手。

**ViT 内部的开销**：线性层（QKV、out-proj、MLP）占 95.8%，全局注意力的 matmul 只占 4.1%（196 个 token）。换成 Swin 窗口注意力最多省 0.7 GFLOPs，参数一个都不少，所以**不考虑换 Swin**。有效的做法是缩小 ViT 的宽度或深度。

**测算（c16 block，M=28，r=32，pam；fvcore，1×224×224）**，对照 DA-TransUNet 107.95M / 30.20G：

| 变体 | Params | GFLOPs | 预训练权重 |
|---|---|---|---|
| R50 + ViT-B ×12（当前） | 105.99M | 29.73 | 现有 `R50+ViT-B_16.npz` |
| R50 + ViT-B ×8（保留前 8 层） | 77.64M | 23.94 | 同上（`load_from` 按层加载，直接可用） |
| R50 + ViT-B ×6 | 63.47M | 21.04 | 同上 |
| R50 + ViT-B ×4 | 49.29M | 18.15 | 同上 |
| R50 + ViT-S ×12（384 维） | 39.70M | 16.40 | 需要下载 AugReg ViT-S/16 i21k 权重；R50 沿用现有权重；1024→384 的 patch embedding 随机初始化 |
| R50 + ViT-S ×6 | 29.05M | 14.14 | 同上 |
| R50 + ViT-Ti ×12（192 维） | 22.55M | 12.87 | 需要 AugReg ViT-Ti/16 |
| ViT-B ×12 + decoder 通道减半 (128,64,32,16) | 103.78M | 25.44 | decoder 本来就是随机初始化，不损失预训练 |
| ViT-S ×12 + decoder 通道减半 | 37.49M | 12.11 | 同 ViT-S |

- ViT 变小后，decoder（约 7.7G）就成了主要开销，其中 28² 和 56² 两层最重（各约 2.5G）。
- 代码：encoder 里的 attention 块原来写死 768 通道，已改为读 `config.hidden_size`。层数、宽度、decoder 通道目前都只能改 config，还没做成 CLI 参数。

**实验顺序**：见 §3 的 E9–E12。先在 Synapse 上做，按 val protocol，M/r/G/gate 用 Phase 1 选出的组合。

**写作上要注意**：轻量化带来的收益来自 backbone，不是 attention 近似本身，论文里要分开报告，避免重演 R7 的问题。

---

# 8. 留给 journal 版本

Gate Collapse 机制研究（gate 轨迹、梯度、分支相似度、非对称初始化）、GCS 的独立预测指标与更多数据集（CVC、ACDC）、pseudocode、完整的 efficiency 分析。
