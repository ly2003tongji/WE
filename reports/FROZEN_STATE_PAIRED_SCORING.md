# Frozen-State Paired Scoring（阶段 1.5）

**工程验证 only**。不得用本报告对研究假设做 go / narrow / no-go。场景来自 `navtest_failures` 1-scene smoke。

最后更新：2026-07-15

## 硬性警告（审查必读）

- **当前仅为 `navtest_failures` 单场景工程验证**，只证明管线可审计、可配对；**不能据此判断研究假设成立**。
- **8192 候选共享一组 source-conditioned 他车 future**：每个交通来源只生成一组 future，全体候选共用；**不是** candidate-conditioned IDM reaction（未对每个候选做 8192 次 IDM rollout）。
- **覆盖范围**：仅 **4 个动态 VEHICLE** 由 `IDMPolicy` 推进；11 个静态车走 `trajectory_policy`；行人/骑行者等 **44** 个 agent 为 log fallback。比较解读必须带上该 coverage。
- **max ADE = 73.4 m**（token `7cd47126ba8f584e`）异常偏大，**必须后续检查**是否来自 IDM 路线漂移、坐标/valid mask、agent 匹配错误或 future 构造 bug，不能直接当作“真实交互差异”。
- **NOC/TTC 大幅翻转**（约 32% / 28%）**可能是**真实交通模型差异，**也可能是** IDM 路线、valid mask、agent 匹配或 future 构造伪影；在未完成代码与异常审查前，不得当作效应量结论。
- **Top-1 未改变**（idx=2）；排序分歧主要体现在 Top-K / Kendall，不宜过度解读为“最优计划翻转”。
- **下一步必须先审查代码与上述异常，再扩大场景**；暂停新实验扩样，直至审查闭环。

## 结论摘要

| 项 | 结果 |
|---|---|
| frozen input fingerprint 是否严格一致 | **是**（Replay/IDM 共用同一 fingerprint） |
| IDM future 是否真正来自 IDM | **是**（4 辆动态车 `IDMPolicy`；其余按 WorldEngine 规则 fallback） |
| Replay/IDM future 是否不同 | **是**（4 agents max ADE>5cm；future hash 不同） |
| 8192 候选奖励能否严格逐项比较 | **是**（同 vocab、同 scorer、同 fingerprint） |
| 是否需要 upstream patch | **否**（协作层 sidecar 完成） |
| 资源门 | build ~9s / score ~70s；RSS <2GB；输出 ~89MB |

## 只读设计核查（阶段 1.5 第一部分）

| 主题 | 位置 / 结论 |
|---|---|
| DenseReward 读 from_scene future | `dense_reward_manager.py` `before_step`：`convert_to_detections_tracks_from_scene` 预载 `observations_list`；`compute_pdm_scores` 用该列表，**不用**在线 IDM agent |
| R 模式 IDM 推进 | `idm_policy.py` `act` + `idm_navigation`；依赖完整 engine/map |
| 冻结初始化最小字段 | `object_track` 位置/朝向/速度/valid/尺寸；地图 `map_features`；TL；ego conditioning；vocab/scorer 配置 |
| PDM Scorer 接口 | ego state + `DetectionsTracks` futures + map_api + route roadblocks；vocab `[8192,T,3]` |
| horizon / 频率 / 尺寸 / mask | `reward_sampling_poses=8`（0.1s×40）；obs buffer=9×0.5s；Pacifica；agent valid 由 position/valid 推出 |
| 无 3DGS | **可行**：`with_render_manager=false` 可完成结构化 future + densereward 评分 |

**为何 DenseRewardManager 不能直接比较交通模型**：即使 R 模式，评分仍读日志 future；NR/R step≥4 的差异是 rollout-conditioned（ego plan 已分叉），不是冻结态交通模型分歧。

## 实验设置

- Scene：`2021.09.29.15.23.04_veh-28_00601_00802-6326d00e52115da4`
- Cutoff：`step=3`（`num_history-1`）
- Ego conditioning：NR Action Policy `plan_idx=1333`（vocab 局部 → WE center）
- 模式：**source-conditioned**（每源一组他车 future，8192 候选共享；非 8192 次 candidate-conditioned IDM）
- 无渲染、无 AlgEngine、未改 upstream

`input_state_fingerprint`：

`d02737044abc8e8001499b5a7f036a1caa9df8c99bf2e6f7401dd2aa2be515af`

（不含交通模型 future；Replay/IDM 评分 provenance 一致。）

## 两种他车 future

| | Replay | IDM |
|---|---|---|
| source | `log_replay` | `idm`（4 车）+ fallback |
| future hash | `4dc058b9…844c` | `151fb2b8…3237` |
| ego conditioning hash | `e414b16f…c55c`（相同） | 同左 |
| coverage | 59 valid agents；15 VEHICLE | IDM rolled 4；static traj 11；non-vehicle log fallback 44 |

变化 agent（max ADE>5cm，均为 `source=idm`）：

| token | mean ADE (m) | max ADE (m) |
|---|---:|---:|
| `7cd47126ba8f584e` | 8.39 | 73.35 |
| `44df645d1b5b584b` | 6.71 | 15.30 |
| `74c0b539dabe5e9b` | 4.48 | 11.24 |
| `3a6b749e38305b9d` | 5.46 | 10.39 |

全体共同 agent 轨迹偏差：mean 0.42 m，max 73.35 m。

Coverage 可解释：`BaseAgentManager.reset` 跳过 PEDESTRIAN/CYCLIST；静态车强制 `trajectory_policy`。与官方 R 行为一致，非管线失败。

## 统一 PDM 评分

- 同一 `test_8192_kmeans.npy`（sha256 `cc44a31e…8ad35`）
- 同一 DenseReward / PDM scorer 路径（注入 future → `trajectory_policy` + `with_dense_reward_manager=true`）
- Replay step=3 分数与历史 NR dense-reward step=3 **逐元素相同**（max abs diff = 0）→ 评分路径校准通过
- IDM 分数 sha256 不同：`383d2d3b…ce9d` vs Replay `0e2c656b…4623`

## 最小比较（工程）

| 指标 | 结果 |
|---|---|
| Future 是否不同 | 是 |
| 变化 agent 数 | 4 |
| NOC flip | 2658 / 8192 = **32.4%** |
| NOC 一致安全 / 危险 / 冲突 | 2395 / 3139 / 2658 |
| TTC flip | 2316 / 8192 = **28.3%** |
| DAC | **完全相同** |
| Comfort | **完全相同** |
| Direction | **完全相同** |
| Score max/mean abs diff | 1.0 / 0.240 |
| Top-1 | **未变**（idx=2） |
| Top-K 重合 | k=1:1.0；k=5:0.2；k=10:0.1；k=50:0.02 |
| Kendall τ-b (score) | **0.657** |

解读（工程，非研究）：本 scene 在冻结态下 Replay/IDM 已产生大量 NOC/TTC/排序分歧；DAC/Comfort 稳定符合“他车 future 主要冲击碰撞/TTC 类项”的预期。不得外推到 held-out 失效预测。NOC/TTC 翻转与 max ADE=73.4 m 均需先做异常/coverage 审查，再谈扩样。

## 产物路径

**Git 外（原始数组）**

`/mnt/cpfs/prediction/lyyy/myself/WE/data/frozen_paired/smoke1_cutoff3/`

含：`future_{log_replay,idm}.{pkl,npz}`、`scores_*.pkl`、fingerprint / provenance JSON、workdir。

**Git 内（脚本 + 摘要）**

- `scripts/frozen_state_lib.py`
- `scripts/build_frozen_traffic_futures.py`
- `scripts/score_frozen_disagreement.py`
- `scripts/compare_frozen_rewards.py`
- `scripts/run_frozen_paired_scoring.sh`
- `reports/frozen_paired_compare_summary.{json,csv}`
- 本文件 `reports/FROZEN_STATE_PAIRED_SCORING.md`

## Upstream patch

**不需要。** 未修改 `upstream/WorldEngine`。

## 下一步建议（不执行）

0. **先审查代码与异常**（max ADE、coverage、future 构造、IDM 路线），再考虑扩样。
1. 审查通过后，才可将同一 sidecar 扩展到 10-scene 工程子集，仅估效应量分布（仍非研究判定）。
2. 阶段 2：审计 BWM augmented pkl 能否冻结配对。
3. Nexus/SMART sidecar：schema 已通，可在批准后接旁路。
4. 正式假设验证改用 train-side 长尾场景；`navtest_failures` 仅保留最终测试。
