# 交通行为模型与接入状态

最后更新：2026-07-15

## 1. 状态总览

| 来源 | 类型 | nuPlan/WE状态 | 权重/数据 | 反应性 | 当前用途 |
|---|---|---|---|---|---|
| Log Replay | 日志重放 | 已支持 NR | 原始日志 | 无 | 非反应式锚点 |
| IDM | 规则模型 | 已支持 R | 无需权重 | 在线 | 当前在线基线 |
| BWM-Offline | 冻结生成轨迹 | WE发布 augmented scenarios | 有轨迹，无模型权重 | 取决于预生成条件 | 离线生成域 |
| SMART | next-token Transformer | 模型代码公开；drop-in接口待核验 | 模型训练/权重状态需核验 | 在线 | 优先接入 |
| Nexus | 噪声解耦扩散 | 原生nuPlan特征；需WE adapter | 公开nuplan.ckpt | 可ego条件生成 | 优先新增模型 |
| TrafficBots V1.5 | CVAE + Transformer | WOMD专用；需重训练和适配 | 无官方权重 | 在线 | 后备扩展 |
| BITS/TBSim | 分层模仿 | Lyft/nuScenes；需nuPlan适配 | 有权重 | 在线 | 第二后备 |
| CTG++ | 条件扩散 | nuScenes/TBSim；需适配 | 有权重 | 在线 | BITS备选 |
| GUMP | 生成式world model | 公开闭环接口与权重不匹配 | 有部分nuPlan权重 | 论文支持 | 暂不计入 |
| nuPlan-R | 扩散式reactive agents | 论文支持，代码未核验发布 | 未核验 | 在线 | 等待开源 |

## 2. 已支持来源

### 2.1 Log Replay

- WorldEngine `NR` 模式；
- 其他 agents 按日志轨迹运动；
- 当 ego 偏离日志时不会产生反应；
- 可检查策略是否利用 IDM 的让行规律；
- 不能作为真实反事实 oracle。

### 2.2 IDM

- WorldEngine `R` 模式；
- 车辆根据规则对 ego 和其他 agents 反应；
- 行人/骑行者通常仍重放日志；
- 参数变体用于同族敏感性和模型内波动，不能冒充独立模型族。

### 2.3 BWM-Offline

官方数据位置：

```text
data/sim_engine/scenarios/augmented/
├── navtrain_50pct_collision/
├── navtrain_50pct_ep_1pct/
└── navtrain_50pct_offroad/
```

使用边界：

- 可读取作者预生成的场景/agent轨迹；
- 可使用统一 PDM Scorer重新生成候选标签；
- 不可调用BWM重新采样；
- 不可假设为8192条候选分别生成了动作条件反应；
- 必须核验 original scene 映射、频率、trajectory schema 和筛选偏差。

**2026-07-16 审计结论（`navtrain_50pct_collision` only；证据链收尾）**：

- 配对等级 **C（仅外部生成域）**；详见 `reports/BWM_OFFLINE_DATA_AUDIT.md`、`reports/hf_bwm_offline_provenance.json`。
- 796 scenarios；预注册 source/original/ego-conditioning/cutoff/双轨/候选级奖励覆盖率均为显式 **0.0**。
- 嵌套键枚举 796/796；命名分组仅 `heuristic_grouping_from_name`，不得升级配对。
- 运动学倾向 dt=0.5s（≈2Hz）；弱 metadata 线索不足以升 B。
- 第一层 schema 不依赖 19GB original（远端元数据体量，未下载）；当前禁止同场景配对奖励归因。

推荐论文名称：

- `BWM-generated frozen trajectories`
- `BWM-Offline`

## 3. 优先接入来源

### 3.1 SMART

资源：

- 模型：<https://github.com/rainmaker22/SMART>
- nuPlan反应式研究：*When Planners Meet Reality*
- 声称的drop-in仓库：<https://github.com/shgd95/InteractiveClosedLoop>

核验项：

1. 仓库是否真正提供可运行代码；
2. checkpoint来源与许可；
3. nuPlan版本、频率、地图和agent接口；
4. 是否允许外部ego控制；
5. 随机种子和采样方差；
6. 如何输出WorldEngine需要的agent boxes/trajectories。

### 3.2 Nexus

资源：

- 仓库：<https://github.com/OpenDriveLab/Nexus>
- 权重：`OpenDriveLab-org/Nexus/nuplan.ckpt`

优势：

- 原生nuPlan数据与feature builder；
- 与SMART、IDM属于不同模型族；
- 支持token级inpainting，可固定ego future并生成其他agents；
- 权重公开，避免从零训练。

已知阻断：

- `world_model_agents_observation.yaml` 指向缺失/不匹配类；
- 不能直接当作nuPlan IDM replacement；
- 需要自己完成SceneTensor、ego mask、输出解码和receding-horizon wrapper。

最小gate：

```text
同一nuPlan场景
候选ego A → 其他agents输出 A'
候选ego B → 其他agents输出 B'
```

若 A' 与 B' 合理变化，并能回写WorldEngine agent状态，再进入完整接入。

## 4. TrafficBots V1.5迁移评估

### 4.1 为什么不能直接使用

- 官方数据管线读取 WOMD；
- 无公开预训练权重；
- Waymo与nuPlan地图语义不同；
- agent历史、目标、交通灯和track管理接口不同；
- WorldEngine公开scenario常按2 Hz使用，而TrafficBots固定10 Hz；
- 直接用Waymo权重做nuPlan仅能算强OOD smoke，不适合作为可信基线。

### 4.2 可信迁移的最小范围

```text
TrafficBots V1.5
车辆-only
1秒历史
4秒未来
10 Hz内部rollout
行人/骑行者Log Replay
ego player_override
关闭3DGS做结构化验证
```

需要实现：

1. 从原始nuPlan DB导出10 Hz；
2. agent/map/traffic-light packer；
3. 坐标和yaw统一；
4. destination/goal构造；
5. 联合 `TrafficBotsAgentManager`；
6. WorldEngine每0.5秒tick执行5个0.1秒子步；
7. nuPlan重训练与验证。

### 4.3 粗略成本

以下为工程估计，不是官方承诺：

- 单H20完整项目：约8–12周，风险高；
- 8×H20：约6–9周，接口工程仍是主要成本；
- 最小迁移M0：约3–5周工程加一轮训练。

只在现有分歧信号与后训练方法已经成立、但缺少额外在线模型证据时投入。

## 5. 其他候选判断

### 5.1 BITS/TBSim

- 有闭环代码和公开权重；
- `trajdata`可读取多数据集，但官方闭环CLI主要支持Lyft/nuScenes；
- 需要nuPlan scene/map转换和WorldEngine adapter；
- 与SMART/Nexus归纳偏置不同，适合成为第5模型；
- 比TrafficBots权重条件更好，但域适配仍重。

### 5.2 CTG++

- nuScenes + TBSim；
- 有扩散权重和闭环代码；
- 默认控制全部agents，外部ego覆盖需改；
- 与BITS共用TBSim基础，二选一优先，不同时首做。

### 5.3 GUMP

- 论文和代码中存在nuPlan reactive world model设计；
- 公开checkpoint对应新版输出，闭环wrapper依赖旧接口/缺失类；
- 官方issue承认相关内容未完整发布；
- 在单场景证明公开checkpoint能随ego变化输出`next_agents`前，不计入可用模型。

### 5.4 暂不使用

- STRIVE：短时离线场景优化，不是持续reactive agent；
- CCDiff：nuScenes/TBSim、无公开checkpoint；
- BehaviorGPT：未核验到官方完整代码与权重；
- SceneDiffuser：驾驶版无官方代码/权重；
- Waymax：框架可插模型，但不额外提供现成学习式agent；
- nuPlan-R：等待公开代码和权重；
- VectorWorld：仓库仍在审批/占位状态。

## 6. 推荐决策顺序

1. 先完成 Replay / IDM / BWM-Offline schema与标签诊断；
2. 核验并接入 SMART；
3. 优先做 Nexus 单场景 adapter；
4. 形成 IDM / SMART / Nexus 三个在线独立家族；
5. 以 Replay、BWM-Offline作辅助域；
6. 只有证据不足时再做 TrafficBots V1.5 或 BITS。

## 7. 接入验收清单

每个新模型必须记录：

- repo commit与checkpoint hash；
-训练数据与许可；
- 输入历史、输出时域、频率；
- 坐标和地图schema；
- 是否显式/隐式条件于ego；
- 是否联合控制多agent；
- 同场景多seed方差；
- agent spawn/despawn和类型支持；
- 单scene运行命令与输出；
- 与IDM/Replay的公平配对方式；
- 执行失败和不支持场景的处理规则。
