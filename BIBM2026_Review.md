# ApproxDA-TransUNet 修改追踪：ML4BMI camera-ready + Journal

最后更新：2026-09-30。审稿意见来自 BIBM 2026 主会（ML4BMI 没有单独的审稿意见）。

- **Camera-ready（workshop）**：保留已录用的实验结果和 8 页结构，只改文字，让论文“说的正好是现有证据能支持的”。稿件在 `paper/`（从原投稿 `b97d2fc~1` 改起）。
- **Journal**：用实验真正回答审稿意见，包括重跑、对照、多 seed。代码在 `experiments/`（已按新协议改好）。
- **判断规则**：改写能解决的放 camera-ready；需要新证据的放 journal。
- **图例**：✅ 已完成 ｜ ⚠️ 只做到披露 ｜ 🟡 代码已就绪，待跑 ｜ ⬜ 未开始 ｜ 🖼 待改图

**当前状态**：camera-ready 的文字修改与学术语调润色已全部完成（8 页，引用与图表排版对齐，22 条参考文献），Fig 2 架构图已重绘修复（512 维向量 $\mathbf{g}$ 与高密度布局优化完毕）。Journal 的代码准备大部分完成，实验待跑。

---

## 1. 逐条审稿意见

### R1. 模型选择协议不清楚（Must fix）
> 审稿原意：论文只写了 train/test，却说“每 15 epochs 评估并选 best checkpoint”；不清楚 checkpoint、M、r、G、gate 是用什么数据选的，如果用了 test，结果会偏乐观。

- **事实**：原实验确实用 test 选的 checkpoint（训练中的“验证”读的是 test），M/r/G/gate 也是从 test 上的消融结果里选的。
- ⚠️ **Camera-ready**：如实披露，不改写成 val。§IV-A 新增 “Evaluation Protocol” 一段，写明用 test 每 15 epochs 选 checkpoint，配置也从消融中选，结果可能偏乐观、且为单次运行；Conclusion 的 Limitations 再说一次。按原计划的规则，“用了 test 选模型就不能靠改措辞修好”，所以这一条在 workshop 版只做到披露。
- 🟡 **Journal（J1）**：
  - ✅ 代码：`--val_interval` 只在 `val.txt` 上验证，没有 val 的数据集直接报错；`test.py` 默认加载最后一个 epoch；新增 `--split val` 和 `--checkpoint`。
  - ✅ 协议已定：一律 300 epochs、取最后一个 epoch；Synapse 18/12；Kvasir 官方 880/120（已修掉一张 train/test 重叠的图）；ISIC 官方 2594/100/1000，用 val 选超参数；Synapse/Kvasir 没有 val，从消融里选并在论文中披露。
  - ⬜ 下载 ISIC 官方 val/test，运行 `generate_lists.py --isic_official_root ... --force`；⬜ 全部重跑。

### R2. Baseline 比较可能不公平（Must fix）
> 审稿原意：大部分 baseline 数字抄自文献，split 和训练协议可能不同；尤其不清楚 ISIC 上的 DA-TransUNet 是否在同样的 80/20 split 下重跑过。

- ✅ **Camera-ready**：逐表标注来源。Kvasir 的 DA 是我们重跑的（†）；Synapse 和 ISIC 的 DA 是文献值（ISIC 取自 DA-TransUNet 原文 Table 2，协议为 256、Adam、随机 75/25，与我们不同），正文和脚注都写明“indicative, not strictly matched”。
- 🟡 **Journal（J2）**：
  - ✅ 两个项目都加了 `--optimizer {sgd, adam}`，使用同一份 list、同一个数据根目录 `experiments/data/`。
  - ✅ 新的 c16 block 在 exact 极限下与 DANetHead 严格等价（误差约 1e-7），ApproxDA 和 DA 之间只剩“近似”和“多一个 64@112² 位置”两处差异。
  - ⬜ 先查 DA-TransUNet 旧重跑在 Synapse 上只有 72.03% 的原因（线索：当时训练读 `../../data`、测试读 `../data`）；⬜ 在三个数据集上 matched 重跑 DA-TransUNet。

### R3. 消融没有隔离单个因素（Must fix）
> 审稿原意：好几组消融同时改了多个变量（r 和 M/gate 一起变），G 没有系统测试；M=112 仍然带 low-rank 投影，不是真正的 full attention。

- ✅ **Camera-ready**：Table V/VI 删掉 M=14/r=64/learn 这两行混淆的配置（只在正文里描述性地提到）；补一组已有的干净 rank 对照（M=112，r=32 对 r=64：79.44 对 78.93）；删掉 bias–variance 类比和 “rank is not the bottleneck” 的说法；注明 M=112 仍是 low-rank，不是 full attention。
- 🟡 **Journal（J3）**：
  - ✅ 代码：`--rank 0` 表示窗口内 exact attention（可做 full attention 和 window-only 对照）；pam/cam 模式只建用到的分支；G 的消融要在 CAM-only 下做（PAM-only 时 G 无效）。
  - ⬜ 跑单变量消融：M ∈ {7, 28, 56, 112}；r ∈ {16, 32, 64, 0}（M=112）；window-only（M=28，r=0）；G ∈ {1, 4, 8}（CAM-only）；routing ∈ {PAM, CAM, fixed 0.5, learned}。

### R4. Eq.(1) 维度不一致，attention 计算不完整（Must fix）
> 审稿原意：按文中给的张量形状，W_rCᵀ 和 W_rDᵀ 不合法；value 的投影以及如何还原到原始 token 网格也没交代清楚。

- ✅ **Camera-ready**：§III 按实际跑实验的 legacy 代码重写完整张量流：Q/K/V 都是 C×N_w（全通道 C）；共享 token 投影 P ∈ R^{N_w×r}；attention A ∈ R^{N_w×r}，按行 softmax、无 scaling；α 残差；窗口合并回原网格；M 截断；grouped CAM 按当时的实现写，并说明它和 DANet 的归一化方向不同（主要结果为 PAM-only，不受影响）；per-channel gate、1×1 conv 加残差融合。另外修正 CNN stride（2/4/8），并写明 DA-TransUNet 公开实现只在 3 处用了 DA。
- ✅ **Journal**：代码已修（GroupedCAM 对齐 DANet；c16 block 对齐 DANetHead；删掉从没调用的块；legacy block 备份在 `--block_version legacy`）。⬜ 写作时按 c16 版重写公式；可选加 pseudocode。

### R5. Gate Collapse 没被 Eq.(5) 证明（Must fix）
> 审稿原意：g=0.5 不代表 PAM/CAM 的参数更新相同，更不能推出表示收敛，因为两个分支的计算和 Jacobian 都不同；证据也主要来自 Synapse。

- ✅ **Camera-ready**：删掉 Eq.(5)、梯度对称的推理和 “structural / fundamental” 的说法；只保留经验观察：“in our Synapse experiments the learnable gate converged close to 0.5 and did not outperform PAM-only routing”；写明没有分析原因，fixed 0.5 等对照留待以后。
- ⬜ **Journal（J6）**：⬜ 在 `train.py` 里记录 gate 随训练的变化（目前没有这项功能）；⬜ fixed 0.5、不同初始化、多数据集；可选：分支特征相似度（CKA）、梯度统计。

### R6. GCS 不能独立预测任务的上下文需求（Strongly recommended）
> 审稿原意：GCS 是用 window 消融的 Dice 差值定义的，又拿来解释同一批实验，属于循环论证；3 个数据集也不足以说明它是任务本身的属性。

- ✅ **Camera-ready**：标题保持录用时的原样；GCS 明确为经验指标（DSC range），写明它不是 predictor，也不是任务的固有属性；只保留 “consistent with different contextual demands” 这样的弱表述；Limitations 写明需要更多数据集。
- ⬜ **Journal（J5）**：二选一。(A) 保持描述性，加数据集检验其可复现性（CVC、ACDC 的 window 消融已有 legacy 结果，需要按新协议重跑）；(B) 定义一个独立于 Dice 曲线的上下文指标（目标大小、结构间距离等），检验它能否预测窗口敏感度。

### R7. 效率的说法在整网层面不成立（Must fix）
> 审稿原意：模型参数和 GFLOPs 都比 DA-TransUNet 多；约 390× 的节省只对应 attention 矩阵那一项；训练时间比较混用了 2-GPU 和 1-GPU。

- ✅ **Camera-ready**：390× 只描述为 attention 矩阵的理论节省；删掉原 Table IV 里的训练时间、显存、推理时间和 Multi-GPU/DDP；新表只保留同一工具统计的 params / GFLOPs / attention 块 GFLOPs（params 只计实际执行的模块：DA 106.56M、ApproxDA pam 108.98M、learn 109.90M；原表把从没调用过的块也算进去了，分别是 107.95 / 112.98 / 114.90M），并如实写明 legacy block 比 DA 的更重（2.73 对 0.87 attention GFLOPs，原因是全通道 Q/K/V 且多一个 block），不主张任何效率优势。
- 🟡 **Journal（J7）**：
  - ✅ 审计查明了原因（`experiments/audit_params_flops.py`）；c16 版比 DA-TransUNet 更轻：105.99M / 29.73 GFLOPs 对 106.56M / 30.20（参数量都只计实际执行的模块；DA 的公开实现另有 1.40M 从没调用过的块，含在内是 107.95M），attention 块 0.40 对 0.87；exact 对照为 1.37，说明近似本身确实省计算。ViT encoder 占 85.1M 参数、17.4 GFLOPs。
  - ⬜ 在同一 GPU 上测 latency / 显存 / 吞吐。
  - ⬜ 可选：轻量化（ViT-B 只保留前 12/8/6/4 层，现有权重可直接用；decoder 通道减半；ViT-S/16 需要下载权重）。最轻组合约 37.5M / 12.1 GFLOPs。收益来自 backbone，写作时要和近似的收益分开报告。

### R8. 统计可靠性不足（Strongly recommended）
> 审稿原意：主要提升只来自单次运行；Synapse 只有 12 个测试病例，0.5–2 个 Dice 点的差距可能只是初始化、数据顺序或 checkpoint 选择造成的。

- ✅ **Camera-ready**：Limitations 写明所有结果都是单次运行。（佐证：ISIC M=28 legacy 配置先后两次运行，best 分别为 89.55 和 89.37，相差 0.18。）
- ⬜ **Journal（J4）**：DA-TransUNet 和最佳 ApproxDA 各跑 3 个 seed（mean ± std）；Synapse per-organ 结果；paired test / bootstrap CI（可用 `experiments/statistical_validation.py`）。

### R9. 表述过强或前后不一致（Strongly recommended）
> 审稿原意：“state-of-the-art” 并非在所有指标上成立，Synapse HD95 和 ISIC mIoU 反而变差；Figure 6 的相关系数和正文冲突；gate 一会儿是标量一会儿是向量；Kvasir 的 HD95 没有物理标定却写成 mm。

- ✅ **Camera-ready**：
  - 删掉所有 SOTA 措辞（包括 Related Work），改为逐项指标说明，主动写出 Synapse HD95 和 ISIC mIoU 没有提升；
  - Fig 6 按图中数值重写：r_s = 0.05 / 0.23 / 0.16 / 0.79，Δg ≤ 2×10⁻⁴，gate 近乎恒定；
  - gate 统一写成 per-channel 向量 g ∈ [0,1]^C；
  - Kvasir 的 HD95 改为像素单位；
  - 补上 Kvasir / ISIC window 消融的原始数值。
- ✅ Fig 2 已重绘完成：gate 标注修正为 512 维向量 $\mathbf{g}$，排版与留白已彻底优化。
- ⬜ **Journal（J9）**：按新结果重新核对所有指标的表述。

---

## 2. Camera-ready 收尾
- ✅ 文字按上面 R1–R9 修改完毕，学术语调与篇幅优化完成（严格控制在 8 页内）。
- ✅ Fig 2 的 gate 标注与布局优化已完成。
- ⬜ 代码链接：脚注仍是 “will be made available upon acceptance”，待换成实际的公开 URL。
- ⬜ 作者信息：还是模板占位。
- ⚠️ 注意：camera-ready 的描述对应**原来的实验**（legacy block、旧 CAM、Kvasir 800/200、ISIC 2075/519、在 test 上选模型）；不要把 journal 的新协议或 c16 block 写进去。

## 3. Journal 实验前的准备
- ✅ 代码已推到 `main`（c16 block、协议相关参数、list 保护和 ISIC 官方划分模式、数据路径统一），另有未提交的 `test.py` 加载修复（去掉 `module.` 前缀、兼容 legacy checkpoint）。
- ⬜ AutoDL 上 `git pull origin main`；⬜ 补 Kvasir-SEG 数据（目前 `experiments/data/` 下只有 ISIC2018 和 Synapse）；⬜ 下载 ISIC 官方 val/test 并生成 list。
- ⬜ 先排查 DA-TransUNet 在 Synapse 上只有 72% 的问题，再开始正式跑。
- 常用命令：
  ```bash
  # Synapse：SGD 0.01，300 epochs，取最后一个 epoch
  python train.py --dataset Synapse --max_epochs 300 --window_size 28 --rank 32 --gate_mode pam
  python test.py  --dataset Synapse --max_epochs 300 --window_size 28 --rank 32 --gate_mode pam
  # Kvasir / ISIC：Adam 1e-3，300 epochs
  python train.py --dataset Kvasir --max_epochs 300 --optimizer adam --base_lr 0.001 --window_size 56 --gate_mode pam
  python test.py  --dataset Kvasir --max_epochs 300 --optimizer adam --base_lr 0.001 --window_size 56 --gate_mode pam
  # ISIC：先在 val 上比较各配置，选定后在 test 上只跑一次
  python test.py  --dataset ISIC --max_epochs 300 --optimizer adam --base_lr 0.001 --window_size 28 --gate_mode pam --split val
  ```
  DA-TransUNet 用同样的参数（不需要 ApproxDA 专用的 `--window_size`、`--gate_mode` 等）。

## 4. Journal 其他加分项（非审稿硬性要求）
- ⬜ J8：更有说服力的定性分析，例如按目标大小或边界/内部分层、失败案例，说明近似在哪里有帮助、在哪里有害。
- ⬜ J10：可复现性材料，包括完整的 split、配置、seed、硬件、评估脚本，以及发布 checkpoint。
