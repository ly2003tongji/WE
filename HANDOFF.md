# WorldEngine 双 Agent 交接

最后更新：2026-07-15（阶段 0+1+1.5–1.9 + **Nexus sidecar 可行性（部分通过）**；暂停）

## 共同目标

以 WorldEngine 为实验平台，研究不同交通行为模型生成的奖励监督是否稳定迁移到未见模型，并在前提成立后提出分歧校准的风险敏感后训练方法。

## Mac 论文会话状态

- 已完成 WorldEngine 正文、附录、公开代码边界和相关领域文献调研。
- 已确定唯一主方向：交通模型分歧能否预测合成经验在未见模型中的迁移失败，并用于稳健后训练。
- 研究定义、实验协议和模型清单已写入：
  - `research/RESEARCH_HANDOFF.md`
  - `research/EXPERIMENT_PROTOCOL.md`
  - `research/TRAFFIC_MODEL_INVENTORY.md`
- H20 新会话使用 `prompts/H20_DISAGREEMENT_AGENT_PROMPT.md`，不要继续使用旧的基础复现提示作为主任务。

## 已确认的论文事实

- 论文是系统与方法框架论文，不是单纯 benchmark。
- 主链路：policy-aligned rare-case discovery → 3DGS digital twin → Behaviour World Model 扩增 → closed-loop rollout → behaviour-regularized RL post-training。
- 3DGS 和 Behaviour World Model 是环境侧模块；主要被更新的是驾驶策略。
- nuPlan 实验从 navtrain 发现 5,340 个 long-tail scenes，生成 31,508 frames 用于后训练。
- 论文报告 full WorldEngine 将 rare closed-loop SR 从 73.66% 提升至 88.89%，但 rare set 仅 288 个短场景。
- 论文明确承认：离轨时 3DGS 质量退化、BWM 对行人/非结构化交互不足、奖励手工设计、多轮后训练不稳定。

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
- 因此默认所谓 RLFT 配置不会进入 `compute_RL_loss`；这一点必须通过运行和完整 loss/data flow 再验证，不能仅凭命名下结论。
- BWM rollout 配置中的 synthetic folder 路径仍为 `/path/to/...` 占位符。
- 官方数据发布了 `data/sim_engine/scenarios/augmented/` 下的 BWM-generated scenarios，但 `data/alg_engine/openscene-synthetic/` 仍需通过 SimEngine 生成。
- 当前可把 BWM 预生成轨迹作为 `BWM-Offline` 行为来源，不能声称可调用完整 BWM。
- Replay/IDM 已在 SimEngine 中分别对应 NR/R；Nexus sidecar 旁路在 1-scene 上部分通过；SMART 尚未可运行接入。

## H20 已知资源

- Ubuntu 22.04.5 LTS，内核 5.10 Alibaba。
- 192 CPU cores，2.0TiB RAM。
- 1× NVIDIA H20，约 96GB，当前空闲。
- Driver 570.133.20，宿主显示 CUDA 12.8。
- `/mnt/cpfs` 总盘接近满载，只余约 1.5TB。
- 另有 Linux 5090 工作站，后续可能申请多 H20。

## H20 Agent 状态

- 阶段 1–5 已完成，详见：
  - `reports/H20_ENV_AUDIT.md`
  - `reports/DATA_MANIFEST.md`
  - `reports/ENV_SETUP.md`
  - `reports/SMOKE_TEST.md`
  - `reports/CODE_PAPER_GAPS.md`
  - `scripts/audit_data_manifest.py`
- 官方代码固定在 `fc79b937050ed9d68e18add2b480ae72578a7ea5`。
- Hugging Face 数据版本 `8728616abaf090d195b3bdc7af6aacde40271145`：174 文件、4.854 TB；ModelScope 完整包含这些数据，另有 6 个文档/脚本文件。
- 默认 quick test 实际运行完整 288 个 `navtest_failures`，文档中的 “1/10” 不是脚本默认值；现有 `debug_scene_name` 没有被 runner 使用。
- 1–10 scene 最小可操作下载约 30.2–33.2 GB：完整 checkpoint、trajectory vocabulary、单体 rare scenario pickle、nuPlan maps，以及一个 rare asset shard。下载后还需生成过滤后的 scenario pickle。
- 任意指定 scene 因缺少 scene→shard 远端索引，保守需三个 rare asset shards；288 baseline 下载约 90.8 GB，工作盘保守需 380–500 GB。
- 闭环 smoke test 可避免 OpenScene 全量 metadata/sensor blobs；SimEngine metric 仍需要外部 nuPlan maps。
- 两个持久环境位于 `/workspace/worldengine/envs/{simengine,algengine}`，源码位于 `/workspace/worldengine/src/`；总占用约 25 GB，低于 `/workspace` 40 GB 限额。临时 cache 放 `/tmp`。
- Driver 570 保持不变；两个环境均已验证 PyTorch 2.0.1+cu118 在 H20 `sm_90` 上运行。
- gsplat v1.4.0 已完成真实 rasterization；MMCV full 1.6.2 已通过官方 CPU/CUDA 检查和 H20 CUDA op 测试。
- 已选择性下载并逐项校验 33.0 GB；其中 navtest PDMS cache 2.791 GB 经后续代码追踪和无-cache 复跑确认并非闭环必需，实际最小集合为 30.208 GB。没有下载 OpenScene blobs 或全量数据。
- part003 已确认包含 90 个完整 asset；已构造 1-scene（337.9 MB assets）和 10-scene（3.532 GB assets）filtered subsets。
- 1-scene NR/R、10-scene NR/R 均完成真实闭环，runner technical success 均为 100%。10-scene NR/R 均无 collision、drivable compliance/SR 均为 70%；R 相对 NR 的 EP 为 0.51100 vs 0.48689，score 为 0.62125 vs 0.61120。
- 1-scene profile 峰值显存 13,638 MiB、峰值 GPU util 97%；10-scene NR/R 墙钟分别约 8.9/9.5 分钟。
- 代码核查确认：主仓只有 BWM synthetic data consumer、没有 BWM generator；MTGS 重建指向外部仓库。
- 默认 rare-log `RLFT` 的 `rl_finetuning=False`，不调用 `compute_RL_loss`。默认实际为 LoRA + reward-head BCE；normal imitation 使用 importance ratio，rare/hard imitation 被 mask。
- 阶段 0+1（分歧数据可观测性）已完成，详见 `reports/DISAGREEMENT_DATA_AUDIT.md`。
- 1-scene NR 基础 smoke 复跑通过；NR/R dense-reward 均通过 60min/64GB/5GB 资源门并写出 8192 维 `pdms_pkl`。
- **DenseRewardManager 不能直接比较交通模型**：R 模式评分仍读日志 future；NR/R step≥4 差异是 rollout-conditioned。
- **阶段 1.5–1.9 frozen-state**：详见 `reports/FROZEN_STATE_PAIRED_SCORING.md`、`reports/FROZEN_DISAGREEMENT_ATTRIBUTION.md`、`reports/IDM_WARM_START_VALIDATION.md`、`reports/IDM_RESTORE_PHYSICS_VALIDATION.md`。
  - 1.5–1.7：cold-start sidecar 跑通；2658 NOC / 2316 TTC 翻转归因到 `44df645d` 近静止振荡。
  - **1.8 warm-start**：完整场景官方 IDM 从 step 0 滚动、不做 cutoff 物理恢复；cutoff 状态门控 **C**（step 1 起分叉，物理漂移 1–4m）；未跑奖励。`44df645d` 在 warm 完全停滞（净位移 0），与 cold 的振荡不同模式——当时无法判断 32% 翻转是否为伪影。
  - **1.9 restore-physics（最终结论）**：新增 `validate_idm_restore_physics.py`，保留 warm 阶段已构建的 `IDMPolicy`/`IDMNavigation` 对象（identity 恢复前后完全不变），在 cutoff 用公开 setter（`set_position/set_heading_theta/set_velocity/set_angular_velocity`）+ `navigation.update_localization()` 把物理状态**严格恢复**为日志冻结态。安全审计：仅用公开接口，`enable_lane_change=False` 硬编码使车道变更缓存为死代码，无需处理；cutoff 前 40/40 token 均无不可逆车道切换。
    - 恢复后 cutoff 门控 = **A**（38 个共同 agent 全部严格通过）。
    - `44df645d` 恢复后仍近静止振荡（净位移 1.25m，与 cold 最大偏差仅 **0.000233 m**），非完全停滞，`current_lane` 有效（`CenterLane`，`lat=0`）。
    - 严格配对下 Replay vs Restored-IDM：**NOC flip 2658、TTC flip 2316**，与 cold-start 数值**完全一致**；single-agent/LOO 归因确认**仍完全由 `44df645d` 单独造成**。
    - **最终判定：该振荡与翻转是官方 IDM 在此场景下的真实行为，不是 sidecar 冷启动 bug**（判定类别 B：严格配对下仍有分歧）。仍只是 1-scene 工程证据。
  - **扩样**：技术条件已具备（restore-physics 协议、门控 A、归因清晰），**仍需用户明确批准**；批准后必须使用 restore-physics 协议而非 `slice_scene_from_cutoff`。
- 摘要：`reports/frozen_paired_compare_summary_v3.*`、`reports/frozen_disagreement_attribution_summary.*`、`reports/idm_warm_start_summary.*`、`reports/idm_restore_physics_summary.*`；原始数组 Git 外 `data/frozen_paired/`。
- 未改 upstream。BWM / SMART / 10-scene 扩样均未执行。
- **Nexus 单场景 ego-conditioned 可行性（本阶段）**：**部分通过**。详见 `reports/NEXUS_SIDECAR_FEASIBILITY.md`。
  - 固定 Nexus `71c31ca…`、ckpt SHA256 `679f6ccf…`、nuPlan fork `e2aa9f34…`、MTR `a5ba7bda…`；隔离环境在 `nexus_sidecar/`（未污染 simengine/algengine）。
  - 新样本 `cutoff=4`；5 帧真实历史；非 ego future 槽已清零并无日志 GT 泄漏；strict-load `missing/unexpected=[]`。
  - ego 16 帧条件保持与同噪声复现通过；预注册 3 组 vocabulary A/B pairs 均显示候选敏感性（最大 ADE≈9.9 m）。
  - 噪声敏感性未达 0.1 m 阈值（≈0.001 m，但 `z_T` 已确认传入）；2/14 生成车辆物理超阈并对 PDM pack 做 log fallback；地图缺 LANE_CONNECTOR/STOP_LINE/CROSSWALK 编码。
  - conditioning candidate 单行 PDM 接口烟测通过（仅读 plan_idx 行，未解释其余 8191）。
  - 候选 A 未复用 `plan_idx.csv` step=5（与 cutoff=4 fingerprint 不一致）；未新增 Action Policy 链。

## 尚未解决的问题

- part003 的 90-scene 映射已可从 tar 获取；part001/part002 尚未下载和索引。
- AlgEngine 官方 pins 与依赖 metadata 有两个已知冲突：`networkx==2.5` 对 mmdet3d 的 `<2.3`，`Shapely==2.0.4` 对 nuscenes-devkit 的 `<2.0`；当前 import/CUDA op 通过，需在真实数据路径继续观察。
- 288 scenes 的实际解压体积、运行耗时、输出体积和 quick-start “5–10 分钟”说法仍需运行验证。
- 论文 full WorldEngine 所用 BWM/checkpoint/训练开关与当前默认开源配置的精确对应关系仍未交代。
- 冻结态 Replay/IDM 在 1-scene smoke 上已见 NOC/TTC/排序分歧，但这 **不是** 研究假设成立的证据；需 train-side 长尾与 held-out 协议。
- BWM augmented scenarios 是否包含 original source token、是否能严格配对、是否针对特定 ego plan 生成，尚未核验。
- SMART 的公开 nuPlan drop-in 实现是否实际可获取、checkpoint 是否可复现，尚未核验。
- Nexus 旁路接口在本 1-scene 上部分通过，但仍缺可直接替换 IDM 的闭环 wrapper；噪声波动门与地图覆盖需在后续场景复核。
- 交通模型分歧是否大于同模型随机波动、是否能预测 held-out failure，尚无实验结论；且不得用 `navtest_failures` 工程 smoke 代替研究判定。

## 新研究假设与阶段门

研究必须依次满足：

1. 不同交通模型对同一场景和 ego candidate 产生非微小的 NOC/TTC/排序分歧；
2. 分歧相对 TTC、车辆数、平均/最坏奖励提供额外 held-out 风险信息；
3. 分歧方法优于多模型均匀混合，并改善未见在线模型和最坏模型性能；
4. rare 安全提升不能以 common 遗忘、低进度或停车为代价。

若第 2 条不成立，停止“分歧可靠性后训练”，不得仅因模型意见不同继续堆方法。

## H20 下一步

阶段 0+1+1.5–1.9 与 Nexus sidecar 可行性（部分通过）已完成并暂停。下一步需用户明确批准后择一推进：

1. 用 restore-physics 协议扩到 10-scene 工程子集（仍非研究结论）；
2. 在更多场景复核 Nexus 噪声敏感性/物理门/地图覆盖，或评估闭环 wrapper；
3. 阶段 2：BWM-Offline 审计（仍为暂缓项，除非另行批准）；
4. SMART 公开性/可用性复核；
5. 正式假设验证改用 train-side 长尾场景；288 rare navtest 保留最终测试。

## 同步协议

- 任一 Agent 得到影响另一侧的重要结论时，更新本文件。
- 实验细节写入 `reports/`，本文件只保留结论和下一步。
- 不在本文件保存 token、主机密码、内部下载凭据或个人信息。
