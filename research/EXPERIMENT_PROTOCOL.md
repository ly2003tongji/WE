# 交通模型分歧实验协议

最后更新：2026-07-21

## 1. 总原则

- 先验证分歧的增量预测价值，再设计后训练方法。
- 固定场景、历史、地图、自车候选、奖励器和时域，只改变交通行为来源。
- 数据划分、统计和 bootstrap 以原始 scene 为单位；同一 scene 的候选、裁剪和增强变体不得跨 split。
- 288 个 `rare navtest` 尽量保留为最终测试集，不用于反复设计分歧公式。
- 第一阶段不要求 3DGS；固定帧结构化评分成立后，再做完整视觉闭环。
- 任何 held-out 模型不得参与公式、阈值、超参数、早停、checkpoint 或场景选择。
- **不**将复现论文 Table 1 / full WE 列为本协议阶段目标；叙事与基线见 `PAPER_POSITIONING.md`。
- **统一 cutoff 已锁定为 4**（2026-07-21）。三源（Replay / IDM restore-physics / Nexus）全部使用该锚点；禁止与旧 cutoff=3 结果并表。

## 2. 配对样本定义

每个诊断样本至少固定：

```text
原始 scene token
历史截止时刻
地图与交通灯状态
agent 集合和车辆尺寸
一批相同的 ego candidate trajectories
评估时域与时间频率
统一 PDM Scorer
交通模型随机种子
```

主分析只在各来源都能运行的 common-support scenes 上进行，同时报告每个来源的覆盖率和执行失败率。

## 3. 阶段 0：代码和数据可观测性

### 目标

证明现有 WorldEngine 输出足以构建研究数据集。

### 任务

1. 固定上游 commit、配置、checkpoint 和环境。
2. 检查 NR/R rollout 输出的：
   - `plan_traj/`；
   - `meta_datas/`；
   - `pdms_pkl/`；
   - 每帧 candidate/token 对应关系。
3. 对一个 `pdms_pkl` 打印 key、shape、dtype 和取值范围，确认是否包含：
   - NOC/no-at-fault collision；
   - DAC；
   - TTC；
   - Comfort；
   - Progress；
   - overall score。
4. 确认 8192 候选共享哪一组他车未来，是否有 candidate-conditioned reaction。
5. 检查 BWM augmented `all_scenarios.pkl`：
   - 顶层 scene 数量与 key；
   - agent trajectory shape、频率和有效 mask；
   - original source token；
   - 同一源场景的 variant 数；
   - 是否能与 original scene 配对。

### 产物

- `reports/DISAGREEMENT_DATA_AUDIT.md`
- 一个只打印 schema/统计、不导出大文件的小脚本
- 1 个 scene 的可重复奖励导出命令

### 通过条件

- 能把同一 scene、同一候选的多来源奖励对齐；
- 或能明确证明当前输出缺少哪些字段，并给出最小补丁位置。

## 4. 阶段 1：现有来源分歧 smoke test

### 行为来源

- Log Replay（NR / 冻结态 Replay future）；
- IDM（**必须** restore-physics 协议，禁止用未恢复物理的 cold/warm 冒充严格配对）；
- Nexus sidecar（固定 seed；与 Replay/IDM **同一 cutoff**）；
- 可选：3–5 组随机参数 IDM，仅用于模型内/同族波动；
- BWM-Offline：**不**纳入同场景配对（等级 C），仅可作外部域旁证。

### 规模

- 先 10 scenes 验证数据对齐；
- 再 50 scenes 观察趋势；
- 不据 50 scenes 的弱结果直接证伪总 idea。

### 统计

对每个 scene、candidate、reward component 计算：

1. 安全标签翻转；
2. reward range / standard deviation；
3. 最坏奖励；
4. Top-1 是否变化；
5. Top-K 重合率；
6. 候选排序相关性；
7. 模型内随机种子方差与模型间方差。

必须把以下分开报告：

- 一致安全；
- 一致危险；
- 意见冲突。

### 产物

- `reports/DISAGREEMENT_SMOKE.md`
- 两两奖励相关图；
- NOC/TTC 翻转率；
- Top-1 一致率；
- 典型冲突与反例。

## 5. 阶段 2：在线学习式交通模型接入

### 2A. SMART

1. 核验 `InteractiveClosedLoop` 是否可获取；
2. 若可获取，先在原生 nuPlan 运行单场景；
3. 对齐 WorldEngine 的坐标、频率、agent 类型和状态更新；
4. 固定 ego 轨迹，检查 SMART agents 是否发生合理响应；
5. 记录 checkpoint、训练数据来源和随机采样方式。

### 2B. Nexus

最小旁路验证：

```text
WorldEngine/nuPlan scene
→ Nexus SceneTensor
→ 固定 ego 历史和一条 ego future candidate
→ 采样其他 agents
→ 解码到 nuPlan/WorldEngine agent states
```

至少使用两条明显不同的 ego candidate，验证其他 agents 输出是否变化。单场景成功后再做在线 receding-horizon wrapper。

### 通过条件

- 至少获得三个独立在线模型族：IDM、SMART、Nexus；
- 若短期只能获得两个，阶段 3 只能定位为 preliminary，不得宣称充分跨在线模型泛化。

## 6. 阶段 3：分歧预测 held-out 失败

### 数据划分

按 original scene 划分：

- scene-train：拟合可靠性函数；
- scene-validation：选择阈值和超参数；
- scene-test：最终一次性评估。

BWM 变体和同一日志裁剪必须跟随 original scene。

### 留族轮换

```text
IDM + SMART → held-out Nexus
IDM + Nexus → held-out SMART
SMART + Nexus → held-out IDM
```

Replay 和 BWM-Offline作为单独外部域报告。

### 预测目标

1. **False-safe transfer failure**
   - 源模型认为安全；
   - held-out 模型发生碰撞、TTC violation 或其他硬安全失败。

2. **排序翻转**
   - Top-1 是否变化；
   - Top-K safe recall；
   - held-out 候选排序相关性。

3. **Decision regret**
   - held-out 最优奖励；
   - 减去源模型选中动作在 held-out 中的奖励。

### 分歧指标

前期使用可解释指标：

- NOC/TTC disagreement；
- 各子奖励 range / variance；
- worst reward；
- ranking disagreement；
- Top-1 vote；
- 模型间差异减去模型内随机波动。

### 必须比较的简单基线

- 最小 TTC；
- 车辆数；
- 最近交互距离；
- 场景类别；
- 源模型平均奖励；
- 源模型最坏奖励；
- 只有基础风险；
- 基础风险 + 分歧。

如果分歧不能超过简单难度指标，就不应继续建设复杂多模型方法。

### 评估指标

- AUROC；
- AUPRC（随机基线等于正样本比例）；
- calibration / Brier / ECE；
- risk-coverage curve；
- regret correlation；
- 高分歧 top-20% 捕获的 held-out failure 比例；
- scene-level bootstrap 95% CI。

### 项目 go / narrow / stop

以下阈值是项目预注册建议，不是领域定律。

**Go：**

- 至少 5%–10% 独立 scenes 出现有意义的安全或排序翻转；
- 模型间差异大于同模型随机波动；
- held-out AUROC 约 0.70 或更高，95% CI 下界高于 0.5；
- AUPRC 明显超过失败基础比例；
- 高分歧组失败率至少约为低分歧组 2 倍；
- 在至少两个 held-out 在线模型族上重复；
- 分歧相对 TTC 等简单基线有增量。

**Narrow：**

- 只在加塞、汇入、无保护转弯、多车博弈等子集稳定；
- AUROC 约 0.60–0.70；
- 只对碰撞/TTC有效。

此时将问题缩小到强交互长尾场景，并使用新的独立场景验证。

**Stop：**

- 分歧虽存在但无法预测 held-out；
- 模型间差异不超过模型内随机性；
- AUROC 长期接近 0.5–0.55；
- 换 held-out 模型后关系消失或反转；
- 效果完全被 TTC、车辆数等解释；
- 大量 held-out 失败发生在源模型高度一致样本中。

## 7. 阶段 4：后训练方法

在阶段 3 通过前，不设计复杂网络。

### 必须实现的训练基线

1. 单一交通模型监督；
2. 多来源均匀混合；
3. 平均奖励/多数投票；
4. 简单分歧过滤；
5. worst-model / GroupDRO；
6. CVaR 或类似尾部风险训练；
7. 提出的分歧自适应方法。

### 候选方法方向

- 可靠性加权子奖励；
- reward distribution prediction；
- 低分歧使用平均监督、高分歧切换到尾部风险；
- 区分仿真失真与合理多模态未来；
- 高分歧场景调用高保真/额外交通模型复核。

不得默认删除高分歧样本。

## 8. 阶段 5：最终策略评测

必须报告：

- held-out 在线交通模型；
- 最坏测试模型性能；
- rare collision、TTC、SR、EP、PDMS*；
- common PDMS 与能力遗忘；
- 停车率、低进度和过度保守性；
- 至少 3 个策略训练随机种子；
- 场景级配对统计和置信区间；
- 相同 rollout/训练预算下的公平比较。

只有“在参与训练的交通模型上平均分更高”不能支持最终主张。

## 9. 数据与算力约束

- 大文件、权重、assets 和输出不得提交 Git。
- 任何新增下载前列出远端文件、预计大小、落盘位置和清理方式。
- 固定帧结构化诊断优先，不先扩大 3DGS 资产。
- SMART/Nexus 接入先做 1 scene，再做 10/50 scenes。
- TrafficBots V1.5 仅作为后备：
  - 车辆-only、1 秒历史、4 秒未来；
  - 从原始 nuPlan DB 导出 10 Hz；
  - 自车 player override；
  - WorldEngine 0.5 秒 tick 内部执行 5 个 0.1 秒子步；
  - 通过最小 gate 后再扩大。

## 10. 每次实验记录模板

```text
日期：
协作仓库 commit：
WorldEngine upstream commit：
交通模型 repo/commit/checkpoint：
场景 split 与数量：
source/held-out 定义：
候选轨迹来源与数量：
交通模型随机种子：
策略训练随机种子：
配置与命令：
GPU/CPU/耗时/显存：
输出路径：
指标：
失败与异常：
结论类型：运行事实 / 代码事实 / 推断
下一步：
```
