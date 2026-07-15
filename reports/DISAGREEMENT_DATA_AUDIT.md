# 交通模型分歧数据与奖励 Schema 审计

审计时间：2026-07-15（UTC+8）
协作仓：`ly2003tongji/WE@326bb28c19358d6727529bfe382b08654d10f8fb`（`h20/reproduction`）
上游：`OpenDriveLab/WorldEngine@fc79b937050ed9d68e18add2b480ae72578a7ea5`
范围：阶段 0 + 阶段 1（1-scene `navtest_failures` 工程/schema smoke）
声明：本报告**不**对研究假设做 go / narrow / no-go 判断；`navtest_failures` 仅用于工程链路验证。

## 1. 结论摘要

1. **阶段 0 通过**：1-scene NR 基础 smoke technical success 1/1；metric 与历史 smoke 一致。
2. **NR / R dense reward 均通过资源门**：墙钟约 3 分钟、峰值 RSS 约 10.3 GB、表观输出约 0.4 GB，均低于 60 分钟 / 64 GB / 5 GB 硬门。
3. **候选级 `pdms_pkl` 已生成**：每 source 9 个文件（step 3–11），字段齐全，主奖励向量 shape=`(8192,)`，index 对应 `test_8192_kmeans.npy` 第 0 维。
4. **严格配对未实现**：缺少完整 canonical state fingerprint 所需字段；`meta_datas/` 在本轮输出中未落盘；同名 `scene/step` 不能替代状态一致性。
5. **他车 future 在 8192 批评分中共享日志轨迹**：`DenseRewardManager` 在 step 0 用 `convert_to_detections_tracks_from_scene` 预载 observation，评分不读取在线 IDM agent 状态。因此现有 NR/R `pdms_pkl` **不能**解释为“同一冻结状态下 Replay vs IDM 交通分歧”。
6. **本轮只定位最小 patch 与 frozen-state 离线评分方案，未修改 upstream，未应用 patch。**

## 2. 精确 commit / 配置 / 命令

### 2.1 固定版本

| 项 | 值 |
| --- | --- |
| 协作仓 commit | `326bb28c19358d6727529bfe382b08654d10f8fb` |
| 上游 commit | `fc79b937050ed9d68e18add2b480ae72578a7ea5` |
| AlgEngine config | `projects/AlgEngine/configs/worldengine/e2e_vadv2_50pct.py` |
| checkpoint | `data/alg_engine/ckpts/e2e_vadv2_50pct_ep8.pth` |
| vocabulary | `data/alg_engine/test_8192_kmeans.npy` |
| vocab SHA-256 | `cc44a31e75a53406db59f026f0358de97931e726f10254542f98d2a87a38ad35` |
| vocab shape/dtype | `(8192, 40, 3)` / `float32` |
| scene | `2021.09.29.15.23.04_veh-28_00601_00802-6326d00e52115da4` |
| subset | `smoke_1scene`（来自 `navtest_failures` / part003） |

### 2.2 阶段 0 命令

```bash
SMOKE_SUBSET_NAME=smoke_1scene \
SMOKE_MODEL_NAME=e2e_vadv2_50pct-stage0-smoke1 \
SMOKE_REACT_TYPE=NR \
ENABLE_RESUME=false \
bash scripts/run_smoke_1scene.sh
```

输出：`experiments/closed_loop_exps/e2e_vadv2_50pct-stage0-smoke1/navtest_failures_NR/`

### 2.3 阶段 1 命令（串行）

```bash
# NR
SMOKE_SUBSET_NAME=smoke_1scene \
DISAGREEMENT_SOURCE=NR \
DISAGREEMENT_MODEL_NAME=e2e_vadv2_50pct-disagreement-smoke1-NR-20260715 \
bash scripts/run_disagreement_rollout.sh

# 通过资源门后再跑 R
SMOKE_SUBSET_NAME=smoke_1scene \
DISAGREEMENT_SOURCE=R \
DISAGREEMENT_MODEL_NAME=e2e_vadv2_50pct-disagreement-smoke1-R-20260715 \
bash scripts/run_disagreement_rollout.sh

# schema / fingerprint 审计
python scripts/audit_disagreement_schema.py \
  --nr-root .../e2e_vadv2_50pct-disagreement-smoke1-NR-20260715/navtest_failures_NR \
  --r-root  .../e2e_vadv2_50pct-disagreement-smoke1-R-20260715/navtest_failures_R \
  --vocab data/hf/data/alg_engine/test_8192_kmeans.npy \
  --require-state-fingerprint-match \
  --out-json data/logs/nr_r_schema_audit.json
```

协作层脚本通过直接调用 `run_simulation.py` / `sim_test.py` 并设置 `with_dense_reward_manager=true`，**未修改** `upstream/WorldEngine`。

## 3. 阶段 0 结果

| 指标 | 值 |
| --- | ---: |
| runner `succeeded` | true |
| duration_s | 58.1 |
| no-at-fault-collision | 1.00000 |
| drivable-area compliance | 1.00000 |
| ego progress | 0.25179 |
| TTC within bound | 1.00000 |
| comfort | 1.00000 |
| driving-direction compliance | 1.00000 |
| score | 0.68825 |

与历史 `e2e_vadv2_50pct-smoke1` NR 一致。环境：simengine/algengine 均为 PyTorch `2.0.1+cu118`，GPU H20，Driver `570.133.20`。
注：外层 `tee` 曾因日志目录瞬时缺失返回非零；仿真本身 1/1 成功。后续阶段 1 已预先创建 `data/logs/`。

## 4. PDM key / shape / dtype / range

来源：`DenseRewardManager` 写出
`{data_output_dir}/pdms_pkl/{scene_id}_step_{step}_scores.pkl`
（`dense_reward_manager.py:189-210`）。

示例（NR，step=3，首个评分帧）：

| key | shape | dtype | min | max | mean |
| --- | --- | --- | ---: | ---: | ---: |
| `no_at_fault_collisions` | (8192,) | bool | 0 | 1 | 0.6168 |
| `drivable_area_compliance` | (8192,) | bool | 0 | 1 | 0.8701 |
| `lane_keeping` | (8192,) | float64 | 0 | 1 | 0.0698 |
| `ego_progress` | (8192,) | float64 | 0 | 1 | 0.5218 |
| `time_to_collision_within_bound` | (8192,) | float64 | 0 | 1 | 0.5359 |
| `comfort` | (8192,) | float64 | 0 | 1 | 0.4414 |
| `driving_direction_compliance` | (8192,) | float64 | 0 | 1 | 0.9360 |
| `score` | (8192,) | float64 | 0 | 1 | 0.4916 |
| `IL_plan_idx` | scalar | int64 | 4642 | 4642 | — |
| `target_traj` | (8, 2) | float64 | -7.14 | 4.58 | — |

NR/R 均写出 9 个 pkl（step 3–11）；两侧 key 集合完全一致；`candidate_count=8192`；schema 审计 `all_pdm_shape_ok=True`。

字段合并逻辑见 `dense_reward_manager.py:311-320,657-667`（`pred_idx=1` 去掉 PDM-closed baseline 后保留 8192 条 vocab 分数）。

## 5. 候选与 vocabulary 对应关系

- vocabulary：`test_8192_kmeans.npy`，shape `(8192, 40, 3)`。
- AlgEngine 选轨：`scores.argmax` → `chosen_ind` → `vocab[chosen_ind]`（见 `traj_scoring_head_RL.py` / `post_processor.py`）。
- `plan_traj/plan_idx.csv` 列：`prefix,step,plan_idx`；`plan_idx ∈ [0,8191]`，直接索引 vocabulary 第 0 维。
- `pdms_pkl` 中向量下标 `i` 与 vocabulary 第 `i` 条候选一一对应。
- `IL_plan_idx` 是 log 最近邻 imitation 标签（`dense_reward_manager.py:202-206`），**不是**闭环 executed `plan_idx`。

本轮 NR `plan_idx` 与阶段 0 / 历史 smoke1 一致（step4=1333…）。R 从 step4 起分叉（step4=7666），与历史 reactive 行为一致。

## 6. State fingerprint 定义与配对结果

### 6.1 严格配对要求（计划定义）

必须相等：

1. ego 当前/历史状态
2. 按 track token 排序的 agents 当前/历史、类型、尺寸、valid mask
3. 地图版本与相关 lane/roadblock 标识
4. 交通灯 ID / 状态 / 时间
5. 候选词表 hash、shape、dtype
6. PDM scorer commit/config、采样频率、评分时域

交通模型 future / seed / coverage 是比较对象，不进入“必须相等”指纹。

### 6.2 本轮实际可观测性

| 字段 | 本轮输出是否具备 |
| --- | --- |
| `pdms_pkl` 8192 子奖励 | 是 |
| `plan_idx.csv` | 是 |
| vocabulary hash | 是（外部文件） |
| `meta_datas/*.pkl` | **否**（目录不存在） |
| ego/agents 历史完整状态 | **否**（缺 meta 落盘） |
| map / traffic-light 指纹 | **否** |
| scorer config / horizon 记录 | **否**（未写入 pdm） |
| traffic future provenance | **否** |

`data_manager.save_data()` 只在 `reset()` 时刷新上一 episode（`data_manager.py:271-308`）。单 scene 跑完后没有再次 `reset`/`flush`，因此 `meta_datas/` 未写出。`base_env.close()`（`base_env.py:323`）未调用该 flush。

### 6.3 配对判定

- **严格配对：未实现。**
- 同名 `scene_id + step` 仅是弱对齐键。
- step≥4 的 NR/R 差异只能标为 **`rollout-conditioned difference`**（ego executed trajectory 已分叉），不能称为交通模型分歧。
- step=3 上 NR/R 的 8192 奖励数组**完全相等**（见下节），但这来自“共享日志他车 future + 尚未分叉的历史 ego”，**不是** Replay/IDM 对比证据。

## 7. 他车 future 是否共享

`代码事实`：

```145:148:projects/SimEngine/worldengine/manager/dense_reward_manager.py
obs_len = ...
for i in range(obs_len): # TODO: NR or R
    observation = self.converter.convert_to_detections_tracks_from_scene(i)
    self.observations_list.append(observation)
```

```233:259:projects/SimEngine/worldengine/manager/dense_reward_manager.py
initial_observation = self.observations_list[self.current_step]
...
interpolated_observations = self._interpolate_observations(
    self.observations_list[self.current_step:self.current_step + self.buffer_size]
)
```

- `convert_to_detections_tracks_from_scene` 读预录 scene（`WE2pdm_utils.py:108-185`）。
- `convert_to_detections_tracks_from_agent_input` 读在线 agent（`WE2pdm_utils.py:187+`），虽在 `dense_reward_manager.py:184-187` 被追加到 `detection_tracks_list`，但 **评分路径不使用它**。
- 因此：同一帧内 8192 候选共享同一组他车 future；且当前实现下 NR 与 R 的 dense-reward future 都来自日志，而不是 IDM 在线反应。

运行旁证：step=3 时 NR/R 全字段 `equal=True`；step≥4 出现差异时，`plan_idx` 已分叉，差异与 ego rollout 状态改变一致。

## 8. 运行时间 / 显存 / RSS / 输出体积

| 实验 | wall_s | sim duration_s | peak RSS | peak GPU mem | peak GPU util | 表观输出 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 阶段0 NR smoke | ~89（含启动） | 58.1 | — | — | — | ~32–393 MB 量级 |
| dense NR | 174 | 146.9 | 10.303 GB | 14647 MiB | 97% | ~395 MB（`du -sh`） |
| dense R | 181 | 152.9 | 10.320 GB | 13699 MiB | 94% | ~429 MB（`du -sh`） |

资源门：墙钟≪3600s，RSS≪64GB，输出≪5GB，无 hard gate / OOM。
说明：监控脚本的 `du -sb` 在该 CPFS 上显著低估（约 36 MB），以 `du -sh` 表观体积为准；两者均低于 5 GB。
`pdms_pkl` 各约 37 MB；`sensor_blobs` 各约 385 MB（96 张图）。

## 9. NR/R 同名 step 奖励差异（工程观察，非研究结论）

| step | score 是否全等 | NOC flip_rate | 备注 |
| ---: | --- | ---: | --- |
| 3 | 全字段 equal | 0 | 历史末帧；共享 log future |
| 4 | 否 | 0.0022 | `plan_idx` NR=1333 vs R=7666 |
| 7 | 否 | 0.0021 | rollout-conditioned |
| 11 | 否 | 0.0087 | rollout-conditioned |

这些数字只说明“闭环分叉后评分会变”，**不能**支撑“Replay/IDM 交通分歧存在/不存在”的研究判断。

## 10. 缺失字段、代码位置与建议 patch（本轮不应用）

### 10.1 缺口

1. `meta_datas` 单 scene 不 flush。
2. `pdms_pkl` 不含 state fingerprint / future provenance。
3. dense reward 他车 future 固定为 log，R 模式未接入 IDM future。
4. 闭环 NR/R 在首个 executed step 后 ego 状态分叉，无法用同名 frame 做严格配对。

### 10.2 建议最小 patch 位置（仅定位）

| 目的 | 文件:行 | 建议 |
| --- | --- | --- |
| 按交通源选择 observation | `dense_reward_manager.py:145-148,233-259` | R 使用 `from_agent_input` 或显式 future buffer；NR 保持 scene；写入 `future_provenance` |
| 末 episode flush meta | `data_manager.py:271-308` + `base_env.close` / executor teardown | 场景结束强制 `save_data()` |
| 保存 fingerprint | `dense_reward_manager.py:209-210` 邻近 | 在写 scores 前写入旁路 json：ego/agents/valid/map/lights/vocab_hash/scorer/horizon |
| 禁止伪配对 | 协作层审计脚本（已做） | fingerprint 不等则标记 `rollout-conditioned difference` |

本轮**未**修改 upstream，**未**应用上述 patch。

### 10.3 Frozen-state Replay/IDM + 离线 PDM 最小方案（设计-only）

```text
1. 在 cutoff=num_history-1（本配置 step=3）冻结：
   ego 当前/历史、agents/valid、地图、交通灯、vocab、scorer、seed、评分时域
2. 固定一条 ego conditioning trajectory（不是 8192 条分别在线反应）
3. 从同一冻结状态分别生成：
   future_replay = log/trajectory_policy
   future_idm    = idm_policy/navigation
4. 对同一 8192 vocab，分别用 future_replay / future_idm 做离线 PDM 评分
5. 比较子奖励 / Top-1 / 排序；记录 source-conditioned、非 candidate-conditioned
```

该方案明确**不是** 8192 次 candidate-conditioned IDM rollout。实现应放在协作层或独立 patch，待下一阶段批准后再做。

## 11. 已确认 / 未确认 / 下一阶段阻塞点

### 已确认（代码事实 / 运行事实）

- 1-scene 基础闭环与 dense-reward 导出可复现。
- `pdms_pkl` schema 与 8192↔vocab 对齐成立。
- 现有 dense reward 共享 log 他车 future。
- 严格 state fingerprint 无法从本轮输出构造。
- NR/R 资源门通过。

### 未确认

- Replay vs IDM 在**同一冻结状态**下是否产生非微小候选级 NOC/TTC/排序分歧。
- BWM-Offline 是否可映射 original / 严格配对（未下载）。
- SMART / Nexus 可运行性（本轮不做）。
- train-side 长尾场景上的研究效应量。

### 下一阶段阻塞点（需新批准）

1. 实现并验证 frozen-state 双 future + 离线 PDM（或经批准应用最小 upstream patch）。
2. 阶段 2：BWM augmented 最小下载与加载审计（19GB original 仍需单独批准）。
3. 阶段 3：仅作为工程配对 smoke / 效应量初估，仍不得用 `navtest_failures` 做研究 go/no-go。
4. 正式假设验证必须另立 train-side 长尾 split。

## 12. 产物路径

| 产物 | 路径 |
| --- | --- |
| 本报告 | `reports/DISAGREEMENT_DATA_AUDIT.md` |
| rollout 脚本 | `scripts/run_disagreement_rollout.sh` |
| 审计脚本 | `scripts/audit_disagreement_schema.py` |
| 阶段0输出 | `upstream/WorldEngine/experiments/closed_loop_exps/e2e_vadv2_50pct-stage0-smoke1/` |
| NR dense 输出 | `.../e2e_vadv2_50pct-disagreement-smoke1-NR-20260715/` |
| R dense 输出 | `.../e2e_vadv2_50pct-disagreement-smoke1-R-20260715/` |
| 审计 JSON | `/mnt/cpfs/prediction/lyyy/myself/WE/data/logs/nr_r_schema_audit.json` |

大输出 / pkl / 图像 / 日志不进入 Git。
