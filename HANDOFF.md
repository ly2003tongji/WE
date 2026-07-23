# WorldEngine 双 Agent 交接

最后更新：2026-07-21（**cutoff=4 单场景三源冒烟已完成**；下一阶段待批：train-side≤10）

## 共同目标

以 WorldEngine 为实验平台，研究不同交通行为模型生成的奖励监督是否稳定迁移到未见模型，并在前提成立后提出分歧校准的风险敏感后训练方法。

## Mac 论文会话状态

- 已完成 WorldEngine 正文、附录、公开代码边界和相关领域文献调研。
- 已确定唯一主方向：交通模型分歧能否预测合成经验在未见模型中的迁移失败，并用于稳健后训练。
- 研究定义、实验协议、模型清单与**论文叙事**已写入：
  - `research/RESEARCH_HANDOFF.md`
  - `research/EXPERIMENT_PROTOCOL.md`
  - `research/TRAFFIC_MODEL_INVENTORY.md`
  - `research/PAPER_POSITIONING.md`（2026-07-21 新增）
  - `reports/POST_TRAINING_OPENSOURCE_AUDIT.md`（2026-07-21 新增）
- H20 新会话使用：
  - **当前唯一阶段**：`prompts/H20_TRAIN_SIDE_LE10_PROMPT.md`（train-side≤10 效应量；先 Plan Mode）
  - 已完成冒烟批准副本：`prompts/H20_CUTOFF4_PLAN_APPROVAL.md`
  - 冒烟任务原文：`prompts/H20_CUTOFF4_THREE_SOURCE_PROMPT.md`
  - 总览/旧全量：`prompts/H20_DISAGREEMENT_AGENT_PROMPT.md`（勿覆盖锁定决策）
- **叙事决策**：WE = 平台/问题来源，不复现 Table 1 全消融；方法阶段可选 OpenWE-SFT（`rl_finetuning=False`）作弱基线。详见定位文档。

## 已确认的论文事实

- 论文是系统与方法框架论文，不是单纯 benchmark。
- 主链路：policy-aligned rare-case discovery → 3DGS digital twin → Behaviour World Model 扩增 → closed-loop rollout → behaviour-regularized RL post-training。
- 3DGS 和 Behaviour World Model 是环境侧模块；主要被更新的是驾驶策略。
- nuPlan 实验从 navtrain 发现 5,340 个 long-tail scenes，生成 31,508 frames 用于后训练。
- 论文报告 full WorldEngine 将 rare closed-loop SR 从 73.66% 提升至 88.89%，但 rare set 仅 288 个短场景。
- 论文明确承认：离轨时 3DGS 质量退化、BWM 对行人/非结构化交互不足、奖励手工设计、多轮后训练不稳定。
- Table 1 是后训**数据配方**消融（common/rare log、synthetic replay、rollout±BWM），不是 traffic-model held-out 协议。

## 已确认的代码事实

- 上游：<https://github.com/OpenDriveLab/WorldEngine>
- 主仓包含 SimEngine 和 AlgEngine，可见完整的训练/评测脚本链；数据、权重和 3DGS assets 外置。
- Behaviour World Model 尚未集成，roadmap 仍为未完成。
- MTGS 重建训练不在主仓，主仓主要加载并渲染预建 Gaussian assets。
- `projects/AlgEngine/configs/worldengine/e2e_vadv2_50pct_rlft_rare_log.py` 默认：
  - `reward_shaping=True`
  - `use_lora=True`
  - `rl_finetuning=False`
  - `importance_sampling=True`
- 因此默认所谓 RLFT 配置不会进入 `compute_RL_loss`；默认实际为 LoRA + reward-head BCE + 选择性 imitation。
- BWM rollout 配置中的 synthetic folder 路径仍为 `/path/to/...` 占位符。
- 官方数据发布了 `data/sim_engine/scenarios/augmented/` 下的 BWM-generated scenarios，但 `data/alg_engine/openscene-synthetic/` 仍需通过 SimEngine 生成。
- 当前可把 BWM 预生成轨迹作为 `BWM-Offline` 行为来源，不能声称可调用完整 BWM。
- Replay/IDM 已在 SimEngine 中分别对应 NR/R；Nexus sidecar 旁路在 1-scene 上判定 **可用**；SMART 尚未可运行接入。

## Mac 侧代码/论文深度问答新增确认事实（2026-07-16）

基于本地 `fc79b937050ed9d68e18add2b480ae72578a7ea5` 官方仓库克隆直接读码 + 论文原文 + 官方 GitHub issues。

- **官方确认 BWM 基于 Nexus 实现**：WorldEngine GitHub issue #8（2026-06-17）。
- **预训练 IL**：`reward_shaping=False` 时只有 `imi` 头；奖励头不存在。
- **imitation / reward 监督**：L2 最近邻 + CE；5 个独立 BCE 对 PDM 缓存列。
- **PDM 标签**：离线缓存，日志他车 future；换交通模型 ≈ 换一套缓存语义。
- **数据管线**：augmented → SimEngine rollout（硬编码 NR）→ openscene-synthetic；部分 synthetic 路径为占位符。
- **无显式策略 KL**；防遗忘 ≈ IS clip + LoRA + 混真实 log。
- **`rl_finetuning=True` 非 stub**，但 WE 全部 `*rlft*` 默认为 False；WE 仓内 `configs/simscale/` 为 True。
- 3DGS 分段与 scenario↔asset 查找见官方 issue #6/#10。

## Mac 侧后训/叙事补充（2026-07-21）

- 克隆并审计 SimScale：`相关论文/World Engine/SimScale` @ `df99d45`——无策略 KL、无 `TrajScoringHeadRL`；`rl_finetuning=True` 仅在 **WE 仓内** `configs/simscale/`。
- KL ≠ LoRA/混数据/IS clip；公开两仓无 Eq.9；无法裁定“未开源”vs“论文与实做不一致”。
- 文档写 True、WE `*rlft*` 配 False；True 时追加 PG+rank+entropy+IS clip，字段齐全可跑。
- 奖励头后训新建全量可训；LoRA 管旧模块。
- **决策**：不复现 Table 1；故事见 `research/PAPER_POSITIONING.md`；审计见 `reports/POST_TRAINING_OPENSOURCE_AUDIT.md`。
- 当前仍不启动后训/OpenWE 基线；**统一 cutoff=4 已锁定**；下一阶段 = 迁移 Replay/IDM 到 4 + 三源分歧冒烟。

## H20 已知资源

- Ubuntu 22.04.5 LTS，内核 5.10 Alibaba。
- 192 CPU cores，2.0TiB RAM。
- 1× NVIDIA H20，约 96GB，当前空闲。
- Driver 570.133.20，宿主显示 CUDA 12.8。
- `/mnt/cpfs` 总盘接近满载，只余约 1.5TB。
- 另有 Linux 5090 工作站，后续可能申请多 H20。

## H20 Agent 状态

- 阶段 1–5 与分歧工程链路（0+1+1.5–1.9、Nexus 可用、BWM-Offline 等级 C）已完成；细节见既有 `reports/*`。
- Upstream 固定 `fc79b937…`；未改 upstream。
- **cutoff=4 三源冒烟（2026-07-21）已完成（工程 smoke ≠ research go/no-go）**：
  - Replay/Restored-IDM：门控 **A**；`pre_cutoff_transition_flags` 全 False；
  - ego conditioning 锁定 `plan_idx.csv` step=5 → **1333**（禁用旧 Nexus 710）；
  - 三源 fingerprint **对齐**；
  - Replay↔IDM：NOC flip 2884（safe→danger 2746），TTC flip 2524；单车 `44df645d…` 主导（2746/2884）；
  - Replay↔Nexus / IDM↔Nexus 亦有显著 NOC/TTC 翻转（见报告）；
  - 报告：`reports/CUTOFF4_THREE_SOURCE_SMOKE.md`；摘要 json/csv 同目录；大数组在 `data/frozen_paired/smoke1_cutoff4_restore_v1/` 与 `nexus_sidecar/outputs/smoke_cutoff4_three_source/`（Git 外）。
- 旧 cutoff=3 产物保留未覆盖；**禁止与 4 并表**。
- BWM-Offline 禁止同场景配对奖励。

## 尚未解决的问题

- ~~最终统一 cutoff 未决定~~ **已锁定 `cutoff=4`**；~~Replay/IDM→4 + 三源冒烟~~ **已完成（单场景工程）**。
- train-side 多源分歧与 held-out 预测尚未做。
- SMART 可用性未核验；BWM 不可配对；Table 1 / full WE 无法开源对齐。
- 288 全量耗时与 part001/002 索引等工程项仍开放。

## 新研究假设与阶段门

1. 不同交通模型对同一场景和 ego candidate 产生非微小的 NOC/TTC/排序分歧；
2. 分歧相对难度基线提供额外 held-out 风险信息；
3. 分歧方法优于均匀混合，改善未见/最坏模型；
4. rare 提升不以 common 遗忘或停车为代价。

若第 2 条不成立，停止分歧可靠性后训练。

## H20 下一步

**单场景三源冒烟已完成并 push（`df99739`）。**

**下一批准项：** 发 `prompts/H20_TRAIN_SIDE_LE10_PROMPT.md` → H20 先 Plan Mode，Mac 批 scene 池后再执行。明确不做：Table 1、后训、SMART 训练、BWM 配对；288 navtest 不进抽样池。

**IDM 口径：** 规则式；path 先沿日志参考轨迹，纵向由 IDM 重算；`enable_lane_change=False`。

## 同步协议

- 任一 Agent 得到影响另一侧的重要结论时，更新本文件。
- 实验细节写入 `reports/`，本文件只保留结论和下一步。
- 不在本文件保存 token、主机密码、内部下载凭据或个人信息。

<!-- TRAIN_SIDE_LE10_BLOCK -->
## train-side ≤10（2026-07-21）

- 报告：[`reports/TRAIN_SIDE_LE10_DISAGREEMENT.md`](reports/TRAIN_SIDE_LE10_DISAGREEMENT.md)
- scene list：[`reports/train_side_le10_scene_list.md`](reports/train_side_le10_scene_list.md)
- 摘要：[`reports/train_side_le10_summary.json`](reports/train_side_le10_summary.json)
- 门控 A / degraded：7 / 3
- conditioning：`static_feasible_median_path_length`（**≠** 冒烟 NR 1333；不可直接数值对比）
- 声称边界：初步效应量 only；未作 go/no-go / held-out 主张
- **停止线**：本批不扩 50、不后训、不 Table1、不 SMART
- 建议（未执行）：初步：主表 Replay↔Nexus NOC 仍见非微小分歧（R↔I 近 0），可考虑扩至 ~50 稳分位数；SMART 仅当扩样后仍有剩余信号再议。本批不执行。
<!-- TRAIN_SIDE_LE10_BLOCK -->

