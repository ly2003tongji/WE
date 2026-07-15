# WorldEngine 精读、复现与选题路线

## 总目标

在已跑通 WorldEngine 最小闭环的基础上，验证交通模型分歧能否预测合成经验在未见模型中的迁移失败，并在前提成立后提出分歧校准的风险敏感后训练方法。

## 当前判断

- 先读 WorldEngine，不先系统补一批前置论文。先读 Fig.1、§1、§2；遇到具体模块再定向补课。
- 导师所指链路是：`真实日志失败发现 → 3DGS 数字孪生 → 他车行为变化 → action policy 闭环 rollout → 策略后训练`。
- 3DGS 是环境渲染器，不与 action policy 联合反向传播。
- 单张 96GB H20 足够做 rare-case 闭环评测、LoRA 后训练和中等规模消融。
- CPFS 仅余约 1.5TB，必须选择性下载，不能直接拉全量数据。

## 工作流 A：Mac 论文精读

### A1. 开始前导读

先输出并暂停：

1. 标题、作者、年份/状态、研究方向与发展脉络；
2. 论文类型及相应阅读重点；
3. 整体写作逻辑；
4. P0/P1/P2 阅读大纲；
5. 读完后必须能回答的 3–5 个问题。

### A2. 精读顺序

- P0：Fig.1、§2、§3.4、Appendix B.2/B.3、Table 1。
- P1：§3.2 的 3DGS、§3.3/B.1 的 Behaviour World Model、§5 局限。
- P2：base model 结构与产业路测细节。

### A3. 定向前置知识

- 闭环：nuPlan/NAVSIM、open-loop/closed-loop、PDMS/SR/EP、replay/IDM。
- 3DGS：Gaussian 参数、投影、alpha compositing、动态 scene graph、MTGS。
- 后训练：行为克隆、轨迹词表、POMDP、policy gradient、KL 约束、offline distribution shift。

## 工作流 B：H20 复现

### B1. 只读审计

1. 记录硬件、驱动、CUDA、Python/conda、磁盘和网络条件。
2. 克隆官方仓库并固定 commit。
3. 核查文档、脚本、checkpoint 和数据 manifest。
4. 在下载前给出最小数据子集与准确/保守体积预算。

### B2. 环境与最小 baseline

1. 使用隔离环境兼容项目所需 Python 3.9、PyTorch 2.0.1/cu118。
2. 先运行 import/编译检查。
3. 只下载少量场景所需 checkpoint、map、metadata 和 3DGS assets。
4. 运行 1–10 个场景 smoke test。
5. 稳定后运行官方 288 rare-case baseline。

### B3. 最小消融

1. Non-reactive replay 与 IDM reactive agents。
2. 对齐同一场景、候选轨迹和 PDM 子奖励。
3. 审计 BWM-Offline augmented scenarios 的可配对性。
4. 接入至少一个独立学习式在线交通模型。
5. 记录 SR、EP、PDMS*、失败类型、吞吐、显存和随机性。

## 工作流 C：交通模型分歧研究

完整协议见 `research/EXPERIMENT_PROTOCOL.md`。

### C1. 数据可观测性

- 确认 `pdms_pkl` 的字段、shape和候选索引；
- 确认8192候选是否共享他车future；
- 确认BWM augmented scene与original scene映射。

### C2. 前提诊断

- 区分基础风险与模型分歧；
- 比较模型间和同模型多seed波动；
- 测量NOC/TTC翻转、Top-1变化和排序分歧；
- 使用scene级划分和统计。

### C3. 严格held-out验证

优先形成三个独立在线模型族：

```text
IDM + SMART → held-out Nexus
IDM + Nexus → held-out SMART
SMART + Nexus → held-out IDM
```

Replay和BWM-Offline作为辅助外部域单独报告。

### C4. 后训练

只有分歧相对TTC、车辆数、平均/最坏奖励具有增量预测力后，才比较：

- 单模型训练；
- 多模型均匀混合；
- 平均/最坏奖励；
- GroupDRO/CVaR；
- 分歧自适应方法。

## 论文—代码重点核查

1. 官方 roadmap 中 Behaviour World Model integration 尚未完成。
2. `e2e_vadv2_50pct_rlft_rare_log.py` 默认 `rl_finetuning=False`。
3. `e2e_vadv2_50pct_rlft_rare_rollout_bwm.py` 的 synthetic folders 是占位路径。
4. 主仓只有 MTGS 渲染消费端，重建训练依赖外部 MTGS 仓库。
5. 区分论文完整系统、当前开源仓库和产业内部验证三者的能力边界。

## 已锁定方向

**交通模型分歧校准的闭环后训练。**

目标不是简单提高 WorldEngine 上限，而是解决：

> 当合成监督来自不完美交通模型时，如何识别具有迁移风险的经验，并避免策略过拟合特定交通行为生成机制。

暂缓：

- 完整 BWM 或 MTGS 重建；
- 3DGS离轨不确定性；
- 从零训练TrafficBots；
- 修改驾驶backbone；
- 在分歧前提未通过时设计复杂后训练网络。

## 决策门槛

首选 idea 必须同时满足：

- 模型间分歧大于模型内随机波动；
- 分歧能在至少两个held-out在线模型族上预测失败或决策遗憾；
- 相比TTC、车辆数等简单基线有增量；
- 后训练优于多模型均匀混合；
- 未见模型最坏性能提高；
- common能力、进度和非保守性不明显退化；
- 代码改动、数据谱系和对照实验可审计。

