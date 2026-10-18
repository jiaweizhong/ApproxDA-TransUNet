# ApproxDA-TransUNet 修改追踪：ML4BMI camera-ready + Journal

最后更新：2026-10-18。审稿意见来自 BIBM 2026 主会（ML4BMI 没有单独的审稿意见）。实验细节和全部数值见 `PENDING_RUNS.md`。

- **Camera-ready（workshop）**：稿件在 `paper/`。原计划只改文字；实际做法是在本机 RTX 5090 上按新协议重跑全部实验，再按新结果改写。正文报 seed 1234 的单次运行，配置取消融中最好的。
- **Journal**：补多 seed、不在 test 上选配置、修复 PAM 不起作用的问题等，需要新证据的内容放在这里。
- **图例**：✅ 已完成 ｜ ⚠️ 只做到披露 ｜ 🟡 进行中 ｜ ⬜ 未开始

**当前状态**：camera-ready 已按 R1–R7 和 R9 修改完毕；R8（多 seed）未回应。8 页，22 条参考文献。作者信息和代码链接（可点击，hyperref hidelinks）都已填好。

---

## 1. 逐条审稿意见

### R1. 模型选择协议不清楚（Must fix）
> 审稿原意：论文只写了 train/test，却说“每 15 epochs 评估并选 best checkpoint”；不清楚 checkpoint、M、r、G、gate 是用什么数据选的，如果用了 test，结果会偏乐观。

- ✅ **Checkpoint**：全部模型重跑，一律取最后一个 epoch，不在 test 上选 checkpoint。
  - Synapse 18/12；ISIC 从 2594 张官方训练图中随机 75/25（1945/649）；CVC 随机 75/25（459/153）。
  - 三个划分都没有 val 集。Implementation Details 写明了这一点。
- ⚠️ **配置（M、r、G、gate）**：取 test 上消融结果最好的那个。Implementation Details 写明“reported configurations are those with the highest DSC in the ablations”。2026-10-18 起，Limitations 不再写“结果偏乐观”。
- ⬜ **Journal**：配置选择不再使用 test（加 val 划分或交叉验证）。

### R2. Baseline 比较可能不公平（Must fix）
> 审稿原意：大部分 baseline 数字抄自文献，split 和训练协议可能不同；尤其不清楚 ISIC 上的 DA-TransUNet 是否在同样的 80/20 split 下重跑过。

- ✅ **重训 DA-TransUNet**：三张主表都加了“DA-TransUNet (reproduced)†”一行，与 ApproxDA 用完全相同的划分、设置和 seed。正文写明“To faithfully reproduce the baseline and ensure a fair comparison…”。其余 baseline 标为文献值，仅供参考。

  | 数据集 | 复现的 DA | 原文 DA | ApproxDA（seed 1234） |
  |---|---|---|---|
  | Synapse（灰度，SGD，300 ep，224） | 79.74 / 28.44 mm | 79.80 / 23.48 mm | 81.63 / 23.73 mm（pam，M=112，r=32） |
  | CVC-ClinicDB（灰度，Adam，500 ep，256） | 89.72 / 84.78 | 89.47 / 82.51 | 90.54 / 85.87（cam，G=1） |
  | ISIC 2018（RGB，SGD，50 ep，256，drop_last） | 88.08 / 81.40 | 88.88 / 82.78 | 89.26 / 82.64（pam，M=64，r=32） |

- ✅ **各数据集的协议**：按“能复现 DA 原文数值”来选：
  - CVC 用灰度：RGB 会让 DA 达到 92.39，比原文高 2.9；
  - ISIC 用 SGD：Adam 只有 87.4，比原文低约 1.5。
- ✅ **旧问题已查清**：DA 在 Synapse 上只有 72% 是 `test.py` 的 bug，FLOP 统计会改写 BN 的统计量，现已修复。

### R3. 消融没有隔离单个因素（Must fix）
> 审稿原意：好几组消融同时改了多个变量（r 和 M/gate 一起变），G 没有系统测试；M=112 仍然带 low-rank 投影，不是真正的 full attention。

- ✅ **Table V（Synapse，pam，G=8）**：M ∈ {7, 28, 56, 112} × {r=32, r=0 (exact)} 的 2×4 网格，加 M=112 下 r ∈ {16, 64}。正文写明 M=112、r=0 是 full attention。
- ✅ **Table VI（Synapse，M=28，r=32）**：pam / cam / fixed / learn 四种 routing，CAM-only 下 G ∈ {1, 4, 8}。
- ✅ **CVC 和 ISIC**：同样跑完整组消融，窗口消融的原始数值写进 §V-B。

### R4. Eq.(1) 维度不一致，attention 计算不完整（Must fix）
> 审稿原意：按文中给的张量形状，W_rCᵀ 和 W_rDᵀ 不合法；value 的投影以及如何还原到原始 token 网格也没交代清楚。

- ✅ **§III 按 c16 block 重写**：DANetHead 布局，C → C/16。
  - Q、K ∈ R^{C_q×N_w}（C_q = C′/8），V ∈ R^{C′×N_w}，P ∈ R^{N_w×r}，A = softmax(Qᵀ K̃) ∈ R^{N_w×r}，E = α(AṼᵀ)ᵀ + X。维度全部一致。
  - 窗口合并、M 截断、DANet 形式的 grouped CAM（G=1 即原版 CAM）都写明了。
  - 融合为 1×1 conv，没有残差。r=0、M ≥ H、G=1 时，block 等价于 DANetHead。
- ✅ **Fig. 2**：按 c16 结构重画（`paper/PlotNeuralNetScripts/approxda_block.py`）。

### R5. Gate Collapse 没被 Eq.(5) 证明（Must fix）
> 审稿原意：g=0.5 不代表 PAM/CAM 的参数更新相同，更不能推出表示收敛，因为两个分支的计算和 Jacobian 都不同；证据也主要来自 Synapse。

- ✅ **只保留经验观察，写明不分析原因**：训练中记录了 gate（`gate_history.npz`），block 均值在 0.499–0.512 之间，per-channel std ≤ 0.022。
- ✅ **Fig. 6 重画**：c16，M=28，learn。r_s 在 −0.43 到 +0.26 之间，Δg ≤ 3.4×10⁻⁴。
- ✅ **对照**：Table VI 有 fixed 0.5。CVC 上 learned gate 也没有胜出（89.90，低于 cam G=1 的 90.54）。
- ⬜ **Journal**：换 gate 初始化或正则化，分析原因。

### R6. GCS 不能独立预测任务的上下文需求（Strongly recommended）
> 审稿原意：GCS 是用 window 消融的 Dice 差值定义的，又拿来解释同一批实验，属于循环论证；3 个数据集也不足以说明它是任务本身的属性。

- ✅ **只作描述性指标**：GCS 写成经验诊断，不是 predictor。删掉了 inductive bias 和 locality-induced bias 的说法。Limitations 写明需要更多数据集。
- ✅ **新数据**：r=32 时的 Dice 跨度为 Synapse 2.47、CVC 0.71、ISIC 1.22，比值为 3.5× 和 2.0×。
- ⚠️ **风险**：见第 3 节。Synapse 和 ISIC 的 PAM 实际上不起作用，窗口之间的差异主要是训练噪声。
- ⬜ **Journal**：修好 attention 后重跑窗口消融；或者定义一个独立于 Dice 曲线的上下文指标。

### R7. 效率的说法在整网层面不成立（Must fix）
> 审稿原意：模型参数和 GFLOPs 都比 DA-TransUNet 多；约 390× 的节省只对应 attention 矩阵那一项；训练时间比较混用了 2-GPU 和 1-GPU。

- ✅ **只写 attention 矩阵**：390× 只描述为 attention 矩阵的理论节省。
- ✅ **Table IV（fvcore，c16）**：

  | 输入 | 模型 | Params | GFLOPs | Attention GFLOPs |
  |---|---|---|---|---|
  | 224 | DA | 106.56M | 30.20 | 0.87 |
  | 224 | ApproxDA M=112 | 107.50M | 29.73 | 0.40 |
  | 256 | DA | 106.60M | 39.75 | 1.22 |
  | 256 | ApproxDA M=64 | 106.46M | 39.05 | 0.52 |

  attention 部分省 54–57%，整网只省 1.6–1.8%。
- ✅ **不主张效率优势**：表中没有训练时间、显存，也没有 DDP 对比。摘要、引言、结论都不提效率优势，定位为诊断平台。
- ⬜ **Journal**：在同一块 GPU 上测 latency、显存和吞吐。

### R8. 统计可靠性不足（Strongly recommended）
> 审稿原意：主要提升只来自单次运行；Synapse 只有 12 个测试病例，0.5–2 个 Dice 点的差距可能只是初始化、数据顺序或 checkpoint 选择造成的。

- ⬜ **Camera-ready 未回应**：2026-10-18 按作者决定，Limitations 只保留两条（GCS 机制尚不清楚；ViT 主导计算量、需要轻量化）。论文里不再提单次运行或 seed。
- 🟡 **已有的多 seed 数据**：显示单 seed 的增益在多 seed 平均后基本消失，期刊版必须如实报告。

  | 数据集 | DA | ApproxDA | 备注 |
  |---|---|---|---|
  | Synapse（3 seed） | 79.81 ± 0.09 | M=112：79.79 ± 1.60 | 配对 Wilcoxon，Dice p = 0.62 |
  | ISIC（3 seed） | 88.38 ± 0.32 | M=128：88.56 ± 0.33 | p = 0.46 |
  | CVC（2 seed） | 89.66 ± 0.08 | cam G=1：90.08 ± 0.65；pam M=128：89.92 ± 0.40 | |

- ⬜ **Journal**：主配置每个数据集至少 3 个 seed，报 paired test 或 bootstrap CI；Synapse 报 per-organ 结果。

### R9. 表述过强或前后不一致（Strongly recommended）
> 审稿原意：“state-of-the-art” 并非在所有指标上成立，Synapse HD95 和 ISIC mIoU 反而变差；Figure 6 的相关系数和正文冲突；gate 一会儿是标量一会儿是向量；Kvasir 的 HD95 没有物理标定却写成 mm。

- ✅ **措辞**：没有 SOTA 或 top performance 的说法。主动写出 Synapse HD95 不如原文 DA 和 Swin-Unet，ISIC mIoU 略低于原文 DA。
- ✅ **数值一致**：Fig. 6 的数值和正文一致；gate 统一写成向量 g ∈ [0,1]^{C′}。
- ✅ **单位**：HD95 统一用 mm。Kvasir 已整体移除，不再有 px 的问题。
- ✅ **前后一致**：§III-D 原来写“primary benchmarks 用 PAM-only”，现在 CVC 主结果是 CAM-only，已改为“window/rank 消融用 PAM-only，routing 在 §V-C 比较”。

---

## 2. Camera-ready 收尾

- ✅ **作者信息**：按 IEEE/BIBM 模板填写。Hui Zhang 为通讯作者，脚注放在 `\title` 的 `\thanks` 里；放在 `\author` 里会多出约 11 pt 空白。
- ✅ **代码链接**：https://github.com/deepsota-ai/ApproxDA-TransUNet
- ✅ **数据集**：Kvasir-SEG 换成 CVC-ClinicDB，因为 Kvasir 上 DA 的复现比原文低 2.1（3 seed）。新增 Bernal 2015 引用。
  - ⚠️ camera-ready 阶段换数据集，是否需要告知 workshop chair，待定。
- ✅ **图**：
  - Fig. 3 重新生成：Synapse 选增益最大且只有 2–3 种器官的切片，CVC 和 ISIC 选 Dice 中位数的 case。
  - Fig. 5（旧 Kvasir attention map）已删除，见第 3 节。
  - Fig. 6 改为矢量 PDF。
- ✅ **措辞与排版**：去掉 inductive bias 的说法；表注统一为整宽 minipage；表格间距已统一（`\centering`）；Fig. 1 和 Fig. 2 为 100% 栏宽。

## 3. 新发现的问题：PAM 几乎不起作用（2026-10-18）

- **现象**：PAM 输出为 α·E + x，α 初始化为 0。
  - Synapse 和 ISIC 是 SGD 训练，α 一直停在约 10⁻³，旧的 T4 模型也一样。把 α 置 0 后，预测只有约 0.0002% 的像素改变。
  - DA-TransUNet 原版的 gamma 也 ≤ 0.007。
  - CVC 是 Adam 训练：M=128 时 α 约 0.2，但置 0 后也只改变 0.046% 的像素；M=8 时 α 衰减到约 5×10⁻⁴¹。
- **影响**：
  - Synapse 和 ISIC 上窗口和 rank 消融之间的差异，基本是训练噪声。
  - 旧 Fig. 5 是把约 10⁻⁶ 的信号做 min-max 归一化后画出来的，不能说明 attention 关注了目标，所以已删除。
- **原因**：
  - α = 0 时，Q、K、V、P 的梯度恰好为 0。
  - α 的梯度比 decoder 卷积小 4–5 个数量级。
  - 即使 α = 1，‖E‖/‖x‖ 也只有 1–3%。
- 🟡 **修复试点**（`queue_attn_pilot.txt`）：
  - 两个新选项 `--attn_norm`（对 E 做 BatchNorm，ReZero 风格）和 `--alpha_init 1`，默认关闭。在 Fig. 3 那张切片上验证过，默认设置得到的 Dice（96.44）和修改前相同。
  - 在 Synapse M=112 上各跑一次；如果有效，期刊版用它重跑窗口消融。

## 4. Journal 待办

- ⬜ J1：配置选择不用 test（val 或交叉验证）。
- ⬜ J4：多 seed，加统计检验。
- ⬜ J5、J6：修好 attention 后重跑窗口消融；分析 gate 的原因。
- ⬜ J7：测 latency、显存、吞吐；可选轻量化 backbone，并和近似带来的收益分开报告。
- ⬜ J8：分层定性分析（目标大小、边界和内部），包括失败案例。
- ⬜ J10：发布可复现材料：split、配置、seed、评估脚本和 checkpoint，放到上面的公开仓库。
