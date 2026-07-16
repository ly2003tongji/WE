# BWM-Offline 增强场景 schema 与配对能力审计

**工程验证 only。** 不运行 BWM、不评分、不下载 original/其他 split/sensor/3DGS；不决定最终统一 cutoff。

## 结论

**配对等级：C（仅外部生成域）。**

公开 `navtrain_50pct_collision` augmented pickle 可安全加载，顶层为标准 WorldEngine scenario dict（796 条），具备 `object_track` / `map_features` / `log_length` 等可消费轨迹结构，可设计映射到现有 `TrafficFuturePack`。但：

- 显式 `source/original/parent/base` 字段：**absent**
- 显式 ego conditioning / plan_idx / plan hash：**absent**
- 显式 cutoff / generation window / 双轨轨迹：**absent**
- 候选级 PDM 奖励：**absent**

因此**不能**主张与 original 严格/弱配对，也**不能**把 `token`/`id` 字符串后缀或 `goal_conditional_*` 字样当作配对证据。

本阶段**不需要**下载 19GB original 即可完成第一层 schema 与等级判定。

## 文件与版本门

| 项 | 值 |
|---|---|
| HF repo | `OpenDriveLab/WorldEngine` |
| revision | `8728616abaf090d195b3bdc7af6aacde40271145` |
| path | `data/sim_engine/scenarios/augmented/navtrain_50pct_collision/all_scenarios.pkl` |
| size | `3,130,339,854` bytes |
| SHA-256 / LFS oid | `55328d2aefe231eee36ae82521223b2bb0e9061952682db06ea5bc0a02120d1a` |
| license | `CC-BY-NC-SA-4.0` |
| 落盘 | `/mnt/cpfs/prediction/lyyy/myself/WE/data/bwm_offline_audit/hf/.../all_scenarios.pkl` |

未下载：另外两个 augmented split、original collision（19GB）、OpenScene blobs、3DGS assets。

## Pickle 安全与资源

| 阶段 | wall | peak RSS | 结果 |
|---|---|---|---|
| scan（pickletools 流式） | 90.1 s | 0.038 GB | ok；protocol 4；仅见 numpy reconstruct/ndarray；**不能代替 schema** |
| load（RestrictedUnpickler） | 18.3 s（脚本）/ 22.8 s（controller） | ≈5.0–5.2 GB | ok；未知 global 会停止，未改用普通 `pickle.load` |

宿主无 `/usr/bin/time`，使用等价 controller（`/proc` RSS + `resource.RUSAGE_CHILDREN`）记录。

## 顶层 schema

| 项 | 结果 |
|---|---|
| 顶层类型 | `dict` |
| scenario 数量 | **796** |
| key 类型 | `str` |
| 单 scenario | `dict` |
| 必填键覆盖 | `object_track/id/dynamic_map_states/map_features/log_length` = **100%** |
| 另见键 | `sample_rate`, `sdc_id`, `base_timestamp`, `name`, `token`, `cameras`, `lidar`, `metadata`, `map`, `dataset` |

### source / original / variant

对候选字段名的全局与抽样深扫描结果均为 **absent**（覆盖率 0）：

`source`, `source_token`, `original*`, `parent*`, `base_scene_id`, `variant*`, `sample_id`, `seed`, `augmentation_type`, `goal`, `intent`, `attack`（作为**字典键**）。

**观察（不得用于提升等级）**：部分 `token` 字符串含 `goal_conditional_copy_with_noise` 或 `intent_attack_with_goal_selection`；部分 `id`/`name` 带 `-NNN` 后缀。这只是字符串内容，不是结构化 source/variant 字段。

## 时间锚点（最高优先级）

| 字段 | pickle 事实 |
|---|---|
| `sample_rate` | 全部 **2**（796/796） |
| `log_length` | **21**（769）、19（17）、15（10） |
| 显式 history/current/cutoff/future/generation_window | **全部 absent** |
| 完整轨迹 vs 仅 BWM future | **无法区分**；只能记为完整轨迹数组，无作者 future 边界 |

SimEngine 代码契约 `dt = sample_rate * 0.05` 在 `sample_rate=2` 时给出 **0.1 s / 10 Hz**。同时 `log_length≈21` 与常见 2 Hz×约 10 s 窗口同形。  
**本阶段不裁决真实频率**；只记录：pickle 字面 `sample_rate=2`，代码契约暗示 10 Hz，帧数分布与 2 Hz×21 帧亦相容。统一 cutoff **未决定**。

若把整段轨迹视为可切片历史：

- cutoff=4：可取出 5 帧（索引 0..4）——**长度上可行**，非作者锚点；
- cutoff=3：只有 4 帧历史，**不满足**“5 帧历史”长度定义。

未发现 ego plan 帧数/频率/坐标系的显式 conditioning 记录；**不能**证明 BWM future 条件于某条 ego plan。

## Ego / agents

| 项 | 结果 |
|---|---|
| `sdc_id` | `ego` |
| agent 类型 | 仅 **VEHICLE**（抽样与全局计数一致） |
| state | `position/heading/velocity/valid/length/width/height` 齐全；`angular_velocity` 多数 absent |
| shape（典型） | position `(T,3)`，heading/valid `(T,)`，velocity `(T,2)`，T=`log_length` |
| 双轨 original/BWM | **absent** |
| per-agent source 标签 | **absent** |
| 能否只替换 BWM agents | **无字段支持** |

## 地图 / 灯 / 坐标

| 项 | 结果 |
|---|---|
| `map_features` | 100% 场景非空；flat dict |
| LANE / CROSSWALK | 有（抽样见数百 lane、数十 crosswalk） |
| LANE_CONNECTOR / STOP_LINE / ROADBLOCK | 抽样 interest 计数为 0（类型名未命中） |
| `dynamic_map_states` | 约 **49.5%** 场景非空；灯态多为 `traffic_light_state: list` |
| `old_origin_in_current_coordinate` | metadata 中有（ndarray shape `[2]`） |
| `digitaltwin_ego2globals` | metadata 中有（list） |
| 显式 `coordinate` | absent |

足以作为现有 PDM scene 输入的**字段骨架**存在；本阶段未跑 PDM。

## Ego conditioning / 泄漏

| 检查 | 结果 |
|---|---|
| 生成时 ego future / plan_idx / hash | **absent** |
| 真实 future / BWM future 分轨 | **absent** |
| 能否证明同 history 只改 traffic future | **不能** → `leakage_unproven` |
| 与新 ego candidate 重配对 | **禁止**（conditioning 未知） |

## 奖励 / 传感器

| 项 | 结果 |
|---|---|
| 8192 候选 PDM 子奖励 / pdms_pkl / overall score | **absent** |
| reward provenance | **absent** |
| `cameras` / `lidar` | 顶层键存在（路径/元数据容器，**未**跟随下载 blob） |
| `metadata.openscene_data_infos_dict` | 100% |
| 3DGS asset 实体 | **未下载**；未把引用当本地资产存在 |

不得把 scene-level 任何东西误认为候选级奖励（本文件中奖励字段本就缺失）。

## TrafficFuturePack 兼容性（只设计，不评分）

| 字段 | 状态 |
|---|---|
| agent futures | **direct/convert**：可按 `extract_replay_futures` 切片；`source` 需 adapter 标为 `bwm-offline` |
| token/type/size/valid | **direct**（size 留在 scene） |
| map / lights | **direct**（灯覆盖不全） |
| ego conditioning | **missing** |
| frequency / horizon | **需澄清/可能插值**（`sample_rate=2` vs 现网 2 Hz 锚点）；本阶段不执行插值 |
| provenance / fingerprint | **convert**（可计算 hash；缺 BWM/model/ego 显式 provenance） |
| 是否需要 19GB original（本阶段） | **否**（第一层 schema/等级已闭合） |
| 与 Replay/IDM/Nexus 同一锚点 | **未定**；数据未给出作者 cutoff |

## A/B/C/D

**C — 仅外部生成域**

证据：

1. source/original 字段 absent（禁止文件名/后缀推断升级）
2. ego conditioning absent（禁止轨迹相似度推断）
3. 无显式 cutoff / BWM future 边界
4. 轨迹/地图结构可被现有 scene 管线消费，故不是 D

允许：独立生成域的数据质量/鲁棒性描述。  
禁止：同场景配对效应、严格/弱配对奖励归因。

若未来出现显式 source + 可证明同 history/ego/map，才可复议 B/A；**不得**用本次 `token` 字符串启发式提前升级。

## 产物

- 脚本：`scripts/audit_bwm_augmented_schema.py`
- 测试：`tests/test_bwm_offline_schema_audit.py`（synthetic + integration skip）
- 报告：`reports/BWM_OFFLINE_DATA_AUDIT.md`
- 摘要：`reports/bwm_offline_schema_summary.json`
- 原始 pickle/cache/time 日志：Git 外 `data/bwm_offline_audit/`

## 暂停点

本阶段结束。不下载 original，不进入 PDM/统一实验/分歧评分。
