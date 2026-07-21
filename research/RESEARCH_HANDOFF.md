# 交通模型分歧研究交接

最后更新：2026-07-21

论文叙事与基线定位见 `PAPER_POSITIONING.md`；后训开源审计见 `reports/POST_TRAINING_OPENSOURCE_AUDIT.md`。当前权威下一阶段以 `CURRENT_STATE.md` 为准。

## 1. 研究问题

WorldEngine 使用仿真生成的闭环经验和 PDM 子奖励训练端到端驾驶策略。但同一场景、同一条自车候选轨迹，在不同交通行为模型下可能得到冲突的碰撞、TTC 和轨迹排序结论。

本项目研究：

> 多个交通行为模型对同一自车动作的分歧，能否预测合成经验在未见交通模型中的失效风险；若能，如何利用该分歧进行风险敏感后训练，使策略在未见交通模型中仍保持安全、进度和普通场景能力？

通俗地说：

> 不无条件相信单个虚拟环境给出的答案；先判断答案换一种合理交通行为后是否稳定，再决定驾驶模型如何学习。

## 2. 三层证据链

研究必须依次通过，不能跳步：

1. **分歧存在**：不同交通模型会改变 NOC、TTC、候选排序或最终动作。
2. **分歧可预测迁移失败**：源模型分歧越高，未见模型中的错误安全、排序翻转或决策遗憾越大。
3. **利用分歧改善策略**：提出的方法优于单模型训练和多模型均匀混合，并提高未见模型与最坏模型性能。

若只满足第 1 条，不足以继续做“分歧可靠性加权”。

## 3. WorldEngine 中的监督关系

### 3.1 数据生成

```text
Action Policy 生成自车候选轨迹
交通行为来源生成其他 agents 的未来
地图、车辆框和动力学提供结构化真值
PDM Scorer 计算候选级子奖励
SimEngine 导出传感器观察、metadata 和 pdms_pkl
```

交通模型本身不直接给出“正确动作”。它只生成他车行为；PDM Scorer 才负责根据结构化状态判卷。

### 3.2 后训练

公开模型具有：

- Imitation Head；
- NOC Head；
- DAC Head；
- TTC Head；
- Comfort Head；
- Progress Head。

各奖励头对候选轨迹预测对应子奖励，并与 PDM 标签计算训练损失。合成反事实样本通常屏蔽 imitation loss。

因此本研究首先比较：

```text
不同交通行为来源
→ 不同的 NOC / TTC 等监督标签
→ 不同的候选轨迹排序
→ 不同的策略选择
```

### 3.3 当前代码边界

- 默认 rare-log 配置曾审计为 `rl_finetuning=False`，主要执行 LoRA + reward-head BCE；上游近期文档又补充了 PG/entropy 配置，H20 端必须以固定 commit 的真实 data flow 为准。
- 论文完整 BWM 未在主仓中提供可调用的模型、权重和完整在线接口。
- 官方数据提供 `scenarios/augmented/` 下的 BWM-generated scenarios；可作为冻结的离线行为来源。
- `openscene-synthetic/{sensor_blobs,meta_datas,pdms_pkl}` 需通过 SimEngine rollout/export 生成，不能假设最终奖励标签已经全部打包。

## 4. 概念边界

### 4.1 分歧不等于难度

必须分开记录：

- **基础风险**：已知模型总体认为动作有多危险；
- **模型分歧**：结论对交通行为假设有多敏感。

三类基本样本：

1. 一致安全：低基础风险、低分歧；
2. 一致危险：高基础风险、低分歧；
3. 意见冲突：高模型敏感性。

不能把所有低分歧样本称为低风险。

### 4.2 分歧不一定是坏数据

高分歧可能来自：

- 某个交通模型错误；
- 场景本身存在多种合理反应；
- 模型处于 OOD 状态；
- 随机采样噪声。

目标不是强迫模型一致，而是判断分歧是否具有迁移预测价值，并在合理不确定性下选择稳健动作。

### 4.3 声明范围

首篇工作应称：

- 跨交通模型稳健性；
- 异构交通行为来源下的经验可靠性；
- traffic-model-aware post-training。

不得在没有独立执行引擎或真实道路证据时声称：

- 跨仿真器泛化；
- sim-to-real；
- 真实世界安全保证。

## 5. 当前交通行为来源

### 5.1 已经可用或有数据

1. **Log Replay**
   - 非反应式；
   - 其他 agents 重放真实日志轨迹；
   - 适合作为 deterministic/non-reactive anchor，不是真实反事实 oracle。

2. **IDM**
   - 规则式在线反应模型；
   - 当前 WorldEngine SimEngine 已支持 `R` 模式；
   - 不同参数 IDM 属于同一模型族，不能伪装成多个独立模型。

3. **BWM-Offline**
   - 官方 Hugging Face 提供 BWM 预生成的 augmented scenarios；
   - 可读取冻结轨迹并重新计算 PDM 标签；
   - 不能自由重采样，也不能保证为每条候选 ego action 重新生成反应；
   - 论文中应称 `BWM-generated frozen trajectories`，不能声称运行了完整 BWM。

### 5.2 需核验或接入

4. **SMART**
   - SMART 模型代码公开；
   - *When Planners Meet Reality* 声称提供 nuPlan drop-in agents，但公开的 `InteractiveClosedLoop` 接口状态仍需在 H20 端核验；
   - 若只有模型代码而无 drop-in wrapper，仍需完成 nuPlan/WorldEngine 适配。

5. **Nexus（优先新增）**
   - 原生 nuPlan 特征与公开 `nuplan.ckpt`；
   - 噪声解耦扩散模型，与 IDM/SMART 属于独立模型族；
   - 可通过固定 ego future token、采样其他 agents 的方式实现 ego-conditioned behavior generation；
   - 公开仓库闭环 `WorldModelAgents` 配置存在缺失类，不能视为即插即用；
   - 推荐先做单场景旁路验证，再接入 WorldEngine。

### 5.3 暂不作为前期依赖

- **TrafficBots V1.5**：无公开权重，面向 WOMD；可信使用需从原始 nuPlan DB 重建 10 Hz 数据并重新训练，预计工程风险中高。
- **GUMP**：论文能力符合，但公开 checkpoint、旧闭环接口和新版输出不匹配。
- **nuPlan-R**：论文符合目标，但截至调研日未发现公开代码与权重。
- **BITS/TBSim、CTG++**：有闭环代码或权重，但主要面向 Lyft/nuScenes，需较重的 nuPlan 适配。
- **BehaviorGPT、SceneDiffuser**：缺少可核验的完整代码/权重。

## 6. 推荐最终实验结构

理想的三个在线反应模型族：

```text
IDM：规则式
SMART：next-token 学习式
Nexus：扩散式生成模型
```

辅助来源：

```text
Log Replay：非反应式锚点
BWM-Offline：冻结生成轨迹外部验证
```

主要留族实验：

```text
IDM + SMART 训练/建模 → Nexus 完全留出
IDM + Nexus 训练/建模 → SMART 完全留出
SMART + Nexus 训练/建模 → IDM 完全留出
```

Replay 和 BWM-Offline 单独报告，不与在线模型四折简单平均。

## 7. 论文贡献目标

理想论文包含：

1. **诊断贡献**：配对量化交通模型对子奖励和轨迹选择的系统性影响；
2. **预测贡献**：证明分歧相对 TTC、车辆数、平均/最坏奖励提供额外 held-out 风险信息；
3. **方法贡献**：提出分歧校准的可靠性加权、奖励分布建模或自适应风险目标；
4. **泛化贡献**：在未见在线模型、最坏模型、rare/common 场景中验证，且不依赖停车获得虚假安全。

平均指标提升只是证据，不是贡献本身。

## 8. 当前最重要的未知项

1. BWM augmented scenario 是否能映射回 original source scene；
2. BWM 轨迹是否针对特定 ego plan 生成，以及是否可用于严格配对；
3. SimEngine 的 `pdms_pkl` 能否稳定导出 8192 候选的各子奖励；
4. SMART drop-in 代码是否实际可获得和运行；
5. Nexus 给定不同 ego candidates 时，其他 agents 输出是否显著改变；
6. 当前上游 commit 中 reward head、PG、importance weighting 的真实训练路径；
7. 模型间差异是否显著大于同模型多随机种子差异。

这些未知项由 `research/EXPERIMENT_PROTOCOL.md` 中的阶段门逐步解决。
