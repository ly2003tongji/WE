# BWM-Offline 增强场景 schema 与配对能力审计

**工程验证 only。** 不运行 BWM、不评分、不下载 original/其他 split/sensor/3DGS；不决定最终统一 cutoff。

## 结论

**配对等级：C（仅外部生成域）。** 证据链收尾后仍为 C。

公开 `navtrain_50pct_collision` augmented pickle 可安全加载，顶层为标准 WorldEngine scenario dict（**796** 条），具备可消费轨迹/地图结构。但：

- 预注册 `source/original/parent/base_*` 候选字段：覆盖率全部 **显式 0.0**
- 预注册 ego conditioning / plan_idx / plan hash：全部 **0.0**
- 显式 cutoff / generation window / 双轨轨迹 / 候选级奖励：全部 **absent / 0.0**
- 命名启发式与弱 metadata 线索**不得**升级等级

本阶段**仍不需要**下载 19GB original 即可闭合第一层 schema 与等级判定。

## 扫描覆盖口径（本轮明确）

| 口径 | 覆盖 |
|---|---|
| 顶层 + metadata 一级字段扫描 | **796/796** |
| 旧版深扫（固定+随机 sample） | **13/796 ≈ 1.6%** |
| 本轮嵌套键枚举（ndarray 为叶） | **796/796** |

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

**远端 API 验证**（下载前已执行；本轮未重下）：revision/path/size/LFS oid 一致。
**本地验证**：size + SHA256 与期望一致。
详见 `reports/hf_bwm_offline_provenance.json`。

未下载：另外两个 augmented split、original collision（**19GB 体量来自 HF 远端文件元数据，未下载**）、OpenScene/3DGS blob 实体。

## Pickle 安全与资源

| 阶段 | wall | peak RSS | 结果 |
|---|---|---|---|
| scan（pickletools 流式） | ~90 s | ~0.04 GB | ok；protocol 4；仅见 numpy reconstruct/ndarray |
| audit（RestrictedUnpickler + 全量嵌套枚举） | ~39 s | ~5.0 GB | ok；未知 global 拒绝；未回退普通 `pickle.load` |

## 顶层 schema

| 项 | 结果 |
|---|---|
| 顶层类型 | `dict` |
| scenario 数量 | **796** |
| 必填键 | `object_track/id/dynamic_map_states/map_features/log_length` = **100%** |
| 另见键 | `sample_rate`, `sdc_id`, `base_timestamp`, `name`, `token`, `cameras`, `lidar`, `metadata`, `map`(**str**), `dataset`(**str**) |

### 显式零覆盖（field_coverage）

所有预注册候选均写入摘要；未命中为 **0.0**（不用空 dict 表示“检查过但不存在”）。

- `source_original`：12 项全部 **0.0**
- `ego_conditioning`：9 项全部 **0.0**
- `variant` / `bwm_provenance` / `reward_*` / `cutoff_window`：全部 **0.0**
- `sensor_top_or_metadata`：`cameras`/`lidar`/`openscene_data_infos_dict` = **1.0**；其余传感器候选 **0.0**

分类器要求：source/ego 需 **覆盖率 ≥ 0.95 且结构证据**；极少数命中不会自动升 B。本轮无结构证据 → 保持 C。

## 全量嵌套键枚举（796/796）

ndarray 始终为叶节点（只记 shape/dtype/nbytes），实例 ID 字典折叠为 `[*]` schema。

### metadata 一级（100%）

| 键 | 类型/示例 |
|---|---|
| `actual_past_timesteps` | int，示例 **4** |
| `original_log_length` | int，示例 **21** |
| `total_frames` | int，示例 **21** |
| `log_name` | str，nuPlan log 前缀 |
| `scenario_token` | str，16-hex |
| `openscene_data_infos_dict` | dict，约 21 帧 token 键 |
| `nuplan_lidar_pc_tokens` | list[len≈21] |
| `old_origin_in_current_coordinate` | ndarray[2] |
| `digitaltwin_ego2globals` | list |
| `ego_agent_angle_stats` | dict |

### openscene 帧 schema（`[*]` 键并集）

含 `token/log_name/log_token/scene_token/timestamp/frame_idx`、`cams`(8 路)、`anns`、`can_bus`、`ego2global*`、`lidar_*`、以及 `lidar_path` / `*_path` 等**路径字符串**。判定为 **path/calibration 容器**，不是已下载的传感器 blob 实体。

### map / dataset

| 键 | 事实 |
|---|---|
| `map` | **str**（如 `us-nv-las-vegas-strip`），非嵌套 dict |
| `dataset` | **str**（如 `scenegen.nuplan`） |
| `map_features` | 实例 dict；schema：`type/polyline/polygon/entry_lanes/...` |

完整路径表见 `reports/bwm_offline_schema_summary.json` → `nested_key_enumeration`（含 `schema_highlights`）。

## 命名结构（仅 heuristic）

标签：**`heuristic_grouping_from_name`**

| 项 | 结果 |
|---|---|
| id 解析成功 | **796/796** |
| 基础分组数 | **102** |
| variant/base 分布 | 多为 10 variants（57 组）；亦有 1–9 |
| token 增广类型 | `goal_conditional` **574**；`intent_attack` **222** |

允许用于数据构成与 variant 分层描述。
**禁止**据此提升配对等级或做奖励归因。命名分组 ≠ source 事实。

## sample_rate 频率消歧

字面：`sample_rate=2`（796/796）。代码契约：`dt=sample_rate*0.05` → 0.1 s / 10 Hz。

运动学探针（valid VEHICLE；60 agents / 986 step pairs）：

| 假设 | 位移 vs 速度×dt 误差 median | p90 |
|---|---|---|
| dt=0.1 s | 0.158 | 3.26 |
| dt=0.5 s | **0.064** | **0.25** |

结论：**dt=0.5 s 明显更一致**（median 约 2.5× 更低，且过 1.5× 边际）。
heading↔velocity 方向一致性较好（median abs err ≈ 0.005 rad），支持速度向量可信，但不单独裁决 dt。

解释倾向：轨迹行为更像 **2 Hz**（间隔 0.5 s），而非把 `sample_rate=2` 直接读成“10 Hz 采样间隔 2”。
**不把作者 cutoff 发明出来**；`metadata.actual_past_timesteps=4` 只记为弱时间线索，不升格为显式 cutoff 字段。

## 传感器 / metadata 容器

| 容器 | 判定 |
|---|---|
| `cameras` | 8 路标定/小数组；`small_numeric_metadata` |
| `lidar` | 标定小数组；同上 |
| `openscene_data_infos_dict` | 路径字符串 + 标定/标注数组；`path_or_calibration_container` |

**修正表述**：pickle 内嵌路径/小数组 ≠ 已下载 OpenScene/3DGS blob；本轮未下载任何新 blob。

## 弱映射线索（不升级等级）

1. `metadata.scenario_token` / `log_name` / openscene `scene_token`（谱系 token，非 `source_token` 字段）
2. `metadata.original_log_length`（长度元数据，非 original scene 指针）
3. `metadata.actual_past_timesteps=4`（弱 past 线索，非显式 cutoff）
4. id/token 命名启发式（102 基础组 × variants）

以上均可描述数据构成；**均不足以**满足 B 的“≥95% 显式 source + 结构证据”。

## TrafficFuturePack 兼容性（只设计，不评分）

| 字段 | 状态 |
|---|---|
| agent futures | direct/convert；adapter 标 `bwm-offline` |
| map / lights | map_features direct；灯覆盖约半 |
| ego conditioning | missing |
| frequency | 运动学倾向 2 Hz；与现网锚点对齐仍需消费侧约定 |
| 是否需要 19GB original | **否**（等级 C 闭合）；仅当未来出现显式 source 且要证 equality 时再议 |
| 统一 cutoff | **未定** |

## A/B/C/D

**C — 仅外部生成域**

1. source/original 预注册字段全 0.0；无升 B 结构证据
2. ego conditioning 全 0.0
3. 无显式 cutoff/双轨/候选奖励
4. 轨迹+地图可消费 → 不是 D
5. 命名启发式 / 弱 metadata **不得**升 B/A

允许：独立生成域质量/鲁棒性描述。
禁止：同场景配对效应、严格/弱配对奖励归因。

## 产物

- `scripts/audit_bwm_augmented_schema.py`
- `tests/test_bwm_offline_schema_audit.py`（A/B/C/D + e2e synthetic + numpy RestrictedUnpickler）
- `reports/BWM_OFFLINE_DATA_AUDIT.md`
- `reports/bwm_offline_schema_summary.json`
- `reports/hf_bwm_offline_provenance.json`
- 原始 pickle/cache/time：Git 外 `data/bwm_offline_audit/`

## 暂停点

证据链收尾完成。不下载 original，不进入 PDM/评分/下一阶段。
