# WorldEngine 交通模型分歧研究：当前状态

最后更新：2026-07-23（用户决策：**先扩充 ~50**；下一 H20 任务见 `prompts/H20_TRAIN_SIDE_LE50_PROMPT.md`）

## 1. 文档用途与权威顺序

本文件是项目的**当前决策来源**，用于防止长对话压缩、Agent切换和旧计划残留造成状态漂移。

信息优先级：

1. `research/CURRENT_STATE.md`：当前研究问题、有效结论、下一阶段；
2. `HANDOFF.md`：H20环境、工程进度和详细交接；
3. `reports/`：具体实验的原始证据；
4. `research/RESEARCH_HANDOFF.md`、`EXPERIMENT_PROTOCOL.md`、`PAPER_POSITIONING.md`：稳定研究定义、协议与论文叙事；
5. 聊天记录：讨论过程，不作为最终事实来源。

只有完成一个正式里程碑、改变研究判断或改变下一阶段时，才更新本文件。普通调试细节留在报告中。

---

## 2. 固定版本

- 协作仓库分支：`h20/reproduction`
- 最新有效协作提交：`54dac5f`（train-side≤10 初步效应量）
- WorldEngine upstream：`fc79b937050ed9d68e18add2b480ae72578a7ea5`
- 官方数据revision：`8728616abaf090d195b3bdc7af6aacde40271145`
- 本地姊妹仓 SimScale（仅审计，非训练依赖）：`相关论文/World Engine/SimScale` @ `df99d45`

后续计划和实验必须记录实际commit；若版本发生变化，先更新本节。

---

## 3. 当前研究问题

> 异构交通行为来源是否会使同一场景、同一自车候选轨迹获得不稳定的奖励监督；这种分歧能否预测一个未见行为来源中的错误安全、排序翻转或决策失败？

若前提成立，再研究：

> 如何利用分歧校准合成经验可靠性，并进行风险敏感后训练，使策略在未见交通模型上仍保持安全和正常进度？

当前还没有进入后训练方法阶段。

---

## 4. 当前允许的论文主张

现阶段只允许研究和表述：

- 异构交通行为来源下的候选级奖励稳定性；
- 交通模型敏感性；
- 未见行为来源的迁移风险；
- traffic-model-aware synthetic supervision。

当前不能声称：

- 在线交通模型held-out泛化已经成立；
- 分歧能够预测未见模型失败；
- 分歧方法能够改善后训练；
- 跨仿真器泛化；
- sim-to-real或真实道路安全提升；
- 已复现或超越论文 Table 1 / full World Engine 数字。

工程可用源（单场景已验证，多场景未做研究结论）：Log Replay、IDM（restore-physics）、Nexus sidecar（固定 seed）。BWM-Offline 定级 C，不入同场景配对。SMART 当前不可直接使用。

**论文叙事与基线（2026-07-21 决策）：** 详见 `research/PAPER_POSITIONING.md`。WE 是平台与问题来源，不是必须打败的 Table 1 靶心；不启动忠实 Table 1 复现；方法阶段若需要开源弱基线，使用 OpenWE-SFT（`rl_finetuning=False`），并明确不等于论文 RL。后训开源审计见 `reports/POST_TRAINING_OPENSOURCE_AUDIT.md`。

---

## 5. 已完成且仍然有效的工程结论

### 5.1 WorldEngine复现

- H20环境、依赖、checkpoint、trajectory vocabulary、maps和最小3DGS assets已验证；
- 1-scene与10-scene NR/R闭环已跑通；
- WorldEngine upstream未修改；
- 详细证据见`reports/SMOKE_TEST.md`与`reports/CODE_PAPER_GAPS.md`。

### 5.2 候选级奖励可观测性

- DenseRewardManager可生成每帧8192候选的NOC、DAC、TTC、Comfort、Progress、Direction和Score；
- 奖励数组下标与`test_8192_kmeans.npy`第0维一致；
- 原始DenseRewardManager即使运行R模式，候选评分仍使用日志`from_scene` future，不直接使用在线IDM future；
- 不能把普通NR/R dense输出解释为交通模型奖励分歧。

### 5.3 冻结态Replay/IDM评分工具

已经实现并验证：

```text
同一冻结输入状态
├── Replay future
└── IDM future

同一8192候选 + 同一PDM Scorer
→ 可严格逐候选比较
```

最终采用的有效协议：

1. IDM从完整场景warm-start，保留policy/navigation；
2. 在cutoff使用公开setter恢复日志物理状态；
3. 调用`navigation.update_localization()`；
4. 不重建policy/navigation，不修改private状态；
5. 恢复后状态门控为A；
6. 执行相同ego conditioning；
7. 生成source-conditioned、非candidate-conditioned IDM future；
8. 使用统一PDM Scorer评分8192候选。

### 5.4 单场景工程现象

在唯一工程smoke scene中：

- NOC：Replay安全→IDM危险 `2658/8192`；
- TTC：Replay安全→IDM危险 `2316/8192`；
- 无反向安全翻转；
- 分歧全部由agent `44df645d1b5b584b` 的IDM轨迹造成；
- 该车在严格恢复后仍近静止振荡，Replay则继续前驶；
- DAC、Comfort和Direction保持一致；
- PDM导出的`ego_progress`会受NOC×DAC乘性门控影响，不是raw progress；
- 结果仅为单场景工程证据，不能用于研究go/no-go。

详细证据：

- `reports/FROZEN_STATE_PAIRED_SCORING.md`
- `reports/FROZEN_DISAGREEMENT_ATTRIBUTION.md`
- `reports/IDM_WARM_START_VALIDATION.md`
- `reports/IDM_RESTORE_PHYSICS_VALIDATION.md`

### 5.5 Nexus单场景sidecar

Nexus公开代码和nuPlan checkpoint已完成单场景工程验证：

- checkpoint strict-load通过，无missing/unexpected keys；
- 使用cutoff=4的5帧真实历史，非ego future不存在日志GT泄漏；
- 官方codec与adapter数值一致，历史round-trip误差约`3e-6 m`；
- ego全部16帧conditioning保持，同noise重复结果一致；
- 使用静态可行且物理合理的预注册candidate pairs；
- 至少一个通过物理门的agent对ego candidate产生响应：
  - pair2；
  - agent `44df645d1b5b584b`；
  - first-4s ADE约`0.574 m`；
- future pack可被PDM Scorer读取，conditioning candidate行`DAC=1`、`Comfort=1`、score约`0.770`；
- 14个生成车辆中2个未通过预注册物理门，PDM接口烟测对其使用明确记录的Log fallback；
- Nexus在该场景和强conditioning下对不同noise近乎确定性，后续统一固定seed，不把随机多样性作为当前必要能力；
- WorldEngine场景缺少`LANE_CONNECTOR`与`STOP_LINE`；CROSSWALK polygon映射已修正。

允许结论：

> Nexus公开模型在本单场景上支持ego-conditioned sidecar推理，并能输出可映射到WorldEngine/PDM的他车future。

不允许外推到多场景、held-out或泛化结论。

详细证据见`reports/NEXUS_SIDECAR_FEASIBILITY.md`。

### 5.6 BWM-Offline数据审计

已完成对官方`augmented/navtrain_50pct_collision/all_scenarios.pkl`的schema与配对能力审计（两轮，含证据链收尾）：

- 文件revision/size/SHA-256核验通过；受限Unpickler安全加载；
- 796个scenarios；顶层与metadata一级字段做796/796全量扫描，嵌套键在收尾轮亦按全量枚举；
- 无显式source/original/parent映射、无ego conditioning/plan、无cutoff/generation window、无候选级奖励、无original/BWM双轨；
- scenario id/token仅含`goal_conditional_*`/`intent_attack_*`等命名后缀，只能作为heuristic分组，不构成source事实；
- 仅VEHICLE，位置/朝向/速度/valid/尺寸齐全，地图存在，约49.5%含非空交通灯；
- `sample_rate=2`频率解释仍标注为待运动学消歧；
- futures/map字段可convert，但ego conditioning和统一锚点missing；
- 本层结论不需要19GB original文件。

判定：**C（仅外部生成域）**。允许作为独立生成域/辅助held-out描述；禁止同场景配对奖励归因。

详细证据见`reports/BWM_OFFLINE_DATA_AUDIT.md`。

---

## 6. 已废弃或不得再使用的结论

以下旧结论已被后续审计推翻或降级：

1. **直接比较NR/R dense rewards**
   - 无效：两者评分future都来自日志。

2. **step≥4同名frame是严格配对**
   - 无效：闭环状态已经分叉，只能称rollout-conditioned difference。

3. **73.4米是正式ADE**
   - 无效：包含Replay invalid/padding时间步；
   - valid-aware正式最大ADE为15.30米。

4. **Top-1未变化**
   - 无效：双方存在大量最高分并列；
   - 正式使用max-score集合、Jaccard和Kendall tau-b。

5. **warm-start完全停滞证明cold-start奖励是伪影**
   - 无效：warm-start在cutoff物理状态不配对，门控为C，不能比较奖励。

6. **cold-start的32%翻转一定是初始化伪影**
   - 已推翻：restore-physics严格配对门控A后，翻转数值和单车归因仍然存在。

---

## 7. 当前交通行为来源状态

### 7.1 已验证

#### IDM

- 当前唯一已验证的在线反应式交通模型；
- 已具备严格冻结态future生成与PDM评分管线；
- 属于规则式模型；
- 单场景中暴露近静止死锁/振荡行为，需要后续多场景分类，而不能默认代表普遍效应。

#### Log Replay

- 非反应式行为锚点；
- 可确定性重放日志轨迹；
- 不能对新的ego动作作出反应；
- 不能作为真实反事实oracle。

#### Nexus

- 已完成公开nuPlan代码与checkpoint的单场景sidecar验证；
- 属于噪声解耦扩散模型；
- 可以固定ego future并生成其他agents；
- 当前按固定seed作为近确定性在线生成来源；
- 后续进入统一奖励比较前，必须与Replay/IDM使用相同锚点、候选、fingerprint和PDM协议；
- 当前仍只有单场景工程证据。

### 7.2 已审计，仅外部生成域

#### BWM-Offline

- 官方数据提供BWM-generated augmented scenarios；
- 没有可调用的完整BWM模型和权重；
- 已审计（见5.6）：无source/original映射、无ego条件、无cutoff、无候选级奖励；
- 定级C：只能作为独立外部生成域，不纳入同场景交通模型共识或源模型分歧计算；
- 不作为核心分歧实验的配对来源；不再投入BWM模型复现。

### 7.3 当前阻塞

#### SMART

截至2026-07-15：

- 原始SMART模型代码公开；
- 论文给出nuPlan相关训练思路；
- `InteractiveClosedLoop`只有README，写明`Code is under internal approval`；
- 没有公开nuPlan闭环wrapper；
- 没有nuPlan-adapted checkpoint；
- 不能继续计入当前可用在线模型。

若Nexus与初步分歧信号成立，SMART应优先于TrafficBots进行nuPlan重训练和接入，因为：

- 模型较小；
- 有nuPlan相关配方；
- 与当前论文链路直接；
- 审稿人容易理解；
- 相比Waymo体系的TrafficBots，数据和接口混杂更少。

### 7.4 后备

- TrafficBots / TrafficBots V1.5：Waymo体系、无可用权重、需nuPlan重训练和重接口适配；
- BITS/TBSim：可作为不同模型族后备，但同样需要nuPlan适配；
- GUMP、nuPlan-R：论文能力相关，但当前公开物不足以直接使用。

---

## 8. 最终期望模型结构

### 核心在线模型

```text
IDM：规则式
Nexus：扩散式
SMART：next-token式，后续计划自行重训练
```

### 辅助来源

```text
Log Replay：非反应式锚点
BWM-Offline：离线生成域
```

若最终获得IDM、Nexus和SMART，可进行：

```text
IDM + Nexus → held-out SMART
IDM + SMART → held-out Nexus
Nexus + SMART → held-out IDM
```

这是强“未见在线交通模型泛化”主张的最低合理结构。

在SMART完成前，只能进行异构行为来源的较窄诊断与初步held-out实验。

---

## 9. 分歧验证前期工作状态

分歧验证所需的“测量与数据准备”前期工作已完成：

- Replay（非反应式）严格冻结态评分：可用；
- IDM（反应式）warm-navigation + restore-physics门控A：可用；
- Nexus（扩散式）单场景ego-conditioned sidecar：可用（固定seed，近确定性）；
- BWM-Offline：已审计，定级C，仅外部生成域；
- 候选级PDM子奖励导出与token映射：可用；
- 统一future-pack schema：Replay/IDM/Nexus已对齐，BWM不纳入配对。

尚未开始的正式研究工作：

- ~~统一时间锚点决定（cutoff=3 vs 4）~~ **已锁定 cutoff=4**；
- ~~Replay/IDM 管线迁移到 cutoff=4 并与 Nexus 三源对齐冒烟~~ **已完成**（单场景 engineering；见 `reports/CUTOFF4_THREE_SOURCE_SMOKE.md`）；
- ~~train-side长尾场景多来源候选奖励比较（≤10）~~ **已完成初步**（见 `reports/TRAIN_SIDE_LE10_DISAGREEMENT.md`）；
- 是否扩至 ~50 稳 R↔N 分位数；R↔I≈0 是否协议/场景特异；
- 分歧是否可预测held-out失败；
- 第三个独立在线模型（SMART）；
- 后训练方法。

## 10. 当前唯一下一阶段

> **train-side ~50 扩样（用户已拍板：先扩充）**

同协议（cutoff=4；静态可行中位数 conditioning；Replay / Restored-IDM / Nexus）将样本扩到约 50，以：

1. 稳定 **R↔N / I↔N** flip 分位数；
2. 检验 **R↔I≈0** 是否持续（若持续，主张改为日志/规则式 vs 学习式）；
3. 抽样 Nexus 多种子对照；仍**不作** go/no-go / held-out / 后训 / SMART。

### 10.1 已完成的初步效应量（≤10）

证据：`reports/TRAIN_SIDE_LE10_DISAGREEMENT.md`（`54dac5f`）。主信号 R↔N；R↔I≈0；多种子 ≪ 模型间（抽样）。

### 10.2 开展顺序

1. ~~锁 cutoff=4~~；~~三源冒烟~~；~~train-side≤10~~；
2. **当前：发 `prompts/H20_TRAIN_SIDE_LE50_PROMPT.md` → H20 Plan Mode → Mac 批后执行**；
3. 扩样报告后再议 SMART / 主张收缩 / held-out。

发给 H20：复制 `prompts/H20_TRAIN_SIDE_LE50_PROMPT.md` 中 `---` 之间正文。

## 11. 该阶段之后的决策

### 若出现稳定、可归因、超越简单难度基线的分歧

1. 扩到更多train-side场景做效应量分布；
2. 启动SMART nuPlan小规模训练与adapter，获得第三个独立在线模型；
3. 建立严格held-out（留族）分歧预测协议；
4. 前提成立后才设计风险敏感后训练。

### 若分歧微弱或完全由场景难度解释

- 缩小主张为异构行为来源奖励可靠性诊断；
- 重新评估是否值得投入SMART；
- 不强行进入后训练。

---

## 12. 稳定研究门槛

后续进入分歧预测前必须满足：

- 至少两个可严格配对的独立行为模型；
- 正式开发使用train-side场景；
- 288 rare navtest不参与公式、阈值和超参数选择；
- 模型间差异大于同模型随机种子差异；
- 分歧相对TTC、车辆数、平均/最坏奖励有增量；
- held-out模型不参与可靠性公式、调参、早停和场景选择。

进入后训练前必须进一步满足：

- 分歧能稳定预测至少一个未见行为来源中的错误安全或决策遗憾；
- 结论在独立scene split上成立；
- 简单均匀混合、平均奖励、最坏奖励、CVaR/GroupDRO等基线不足以完全解释收益。

---

## 13. 协作与审核流程

以后固定采用滚动单阶段流程：

```text
Mac侧确定唯一下一任务
→ H20 Agent只在Plan Mode制定该阶段计划
→ 用户将计划发回Mac侧
→ Mac侧审核并给执行提示词
→ H20执行、commit并push
→ Mac侧拉取代码和报告审核
→ 更新CURRENT_STATE.md
→ 决定下一个唯一阶段
```

规则：

- 不再一次预写多阶段计划；
- H20 Agent不自行扩展下一阶段；
- 普通工程调试由H20 Agent内部解决，不单独升级为研究阶段；
- 每次向Mac侧汇报前，H20结果必须commit并push；
- 未进入Git的本地结果不能作为Mac侧最终审核依据。
