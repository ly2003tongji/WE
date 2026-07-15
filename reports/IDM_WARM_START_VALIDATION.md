# IDM Warm-Start Validation（阶段 1.8）

**工程验证 only**。不得据此判断研究假设成立。仅当前 1-scene；未扩样、未下载新数据、未改 upstream、未跑 BWM/SMART/Nexus。

最后更新：2026-07-15

## 核心问题

cold-start sidecar（`slice_scene_from_cutoff` 后重新初始化 IDM）中，`44df645d1b5b584b` 近静止振荡并单独造成全部 NOC/TTC 翻转。该行为在从完整原始场景正常跑到 cutoff 的官方 IDM 中是否仍存在？

## 预先声明的 cutoff 配对容差（事后未放宽）

| 量 | 严格 (A) | 软 (B) |
|---|---:|---:|
| 位置 | 0.05 m | 0.10 m |
| 航向 | 0.5° | 1.0° |
| 速度 | 0.05 m/s | 0.20 m/s |
| 尺寸 | atol 1e-3 m | 同左 |

- **A**：共同 agents 均在严格容差内，且 map / 交通灯一致 → 可比较 future 与奖励  
- **B**：仅数值在软容差 → 完整报告误差后可比较  
- **C**：超出软容差或类别字段不一致 → **禁止比较奖励**

## 1. 小修正

- `diagnose_idm_agents.py`：`geometry_label_consistent` 与汇总均使用 **Replay 全时域最小中心距**（不再用 `min_dist_r[0]`）；单测 `test_diagnose_geometry.py`。
- `run_hybrid_attribution.py`：去掉「必须恰好 4 个 IDM agents」硬断言；记录实际数量（本 warm 场景动态 IDM 为 **5**，含 `dfcaa5d389bc56ef`）。

## 2. Warm-start 构造（协作层，无 upstream 修改）

脚本：`scripts/validate_idm_warm_start.py`、`scripts/run_idm_warm_start_validation.sh`。

| 项 | 记录 |
|---|---|
| 场景 | 完整原始 scene 前缀 `truncate_scene_prefix(cutoff+horizon)`，**非** `slice_scene_from_cutoff` |
| 模式 | `agent_policy=idm_policy`，`agent_navigation=idm_navigation`，无 3DGS |
| IDM 真正开始 | **step 0**（`env.reset` 即为 `IDMPolicy`） |
| cutoff 前 ego | **原始日志** `[0, cutoff)` |
| cutoff 后 ego | 与 cold 相同 vocab `plan_idx=1333`；requested/executed 误差 **0** |
| 输出 | Git 外 `data/frozen_paired/smoke1_cutoff3_warm_v1/` |

## 3. Cutoff 状态门 → **C（不配对）**

- ego：位置/航向误差 **0**（日志跟随正确）  
- 共同 agents：从 **step 1** 起超出软容差；cutoff 时最大位置误差约 **4.06 m**（`dfcaa5d389bc56ef`）  
- 焦点车 `44df645d` 在 cutoff：pos_err ≈ **1.01 m**，speed_err ≈ **1.84 m/s**（日志在动，warm 卡住）  
- map / 交通灯：与冻结日志一致（门控使用日志 TL 快照）  
- **结论：grade C → 未运行 warm-start 奖励评分，未做公平 cold/warm PDM 对比**

### 差异起点与原因（诊断）

| 检查 | 结果 |
|---|---|
| 首次非 A | **step 1**（IDM 第一步后） |
| IDM 是否提前介入 | 是：从 step 0 起执行官方 IDM |
| ego 历史 | 与日志一致（非原因） |
| 初始化 | cold 在 cutoff 截断重初始化；warm 从 scene 起点连续滚动 → 物理状态在 cutoff 已分叉 |

### 保留 warm navigation、在 cutoff 恢复物理冻结态（设计 only，本轮未实现）

见 `data/frozen_paired/smoke1_cutoff3_warm_v1/restore_freeze_physics_design.json`：

1. Warm-start 滚到 cutoff  
2. 对 IDM agents 用公开 `set_position` / `set_heading_theta` / `set_velocity` 写回日志冻结态  
3. 调用 `navigation.update_localization()`，**不**重建 `IDMNavigation` / `original_route`  
4. 再接相同 post-cutoff ego conditioning  

若 `set_*` + `update_localization` 不足以同步 `IDMPolicy` 内部路由状态，则需改 upstream/私有状态 → **暂停，不自行实现**。

## 4. Cold vs Warm 轨迹（诊断 only；非奖励配对）

焦点 `44df645d1b5b584b`：

| | Cold-start（截断重初始化） | Warm-start（完整场景） |
|---|---|---|
| 近静止振荡 | **是**（速度 ~0.03↔1.84 交替） | **否** |
| 完全停滞 | 否 | **是**（speed≡0，净位移 **0 m**，卡在 **scene 起点** pose） |
| 相对 Replay 净位移 | cold≈1.25 m vs Replay≈16.54 m | warm post-cutoff 净位移 **0**（且不在 cutoff 走廊） |
| `current_lane` | 记录为字符串 `"None"`（实为 `CenterLane` 且 `id=None`） | `CenterLane`，`following_original_traj=True` |
| cold↔warm max ADE | — | ≈ **2.56 m**（诊断；门控 C） |

其余 warm IDM 车（`3a6b749e` / `74c0b539` / `7cd47126` / `dfcaa5d`）可正常前进（净位移约 21–39 m），**唯独** `44df645d` 在 warm 中全程卡死。

**直接回答核心问题**：cold 中的「近静止振荡 + 造成 32% NOC 翻转」**不能**在「从完整场景正常跑到 cutoff 的官方 IDM」中复现为同一模式；warm 表现为另一种病理（完全停滞于起点）。因此该振荡/翻转机制属于 **sidecar 截断冷启动伪影（外加门控 C 下不可公平比奖）**，不能当作官方连续 IDM 在冻结态的效应量。

## 5. Warm-start 奖励评分

**跳过**（cutoff 门控 C）。无 warm NOC/TTC flip 数字可报。

## 6. 判定

| 标准 | 结果 |
|---|---|
| 分类 | **`cold_start_artifact`** |
| 阶段 1.5–1.7 | 仅保留为「发现冷启动伪影」的工程记录 |
| 32% NOC / 28% TTC 翻转 | **不得**用于后续效应量 / 扩样统计 |
| 扩样 | **禁止**（须先改为 warm-start 或「保留 warm nav + cutoff 恢复冻结物理」协议，并通过门控 A/B 后再评分） |

补充：warm 上该车完全停滞说明完整前缀 IDM 对该 token 也有控制失效，但与 cold 的振荡机制不同；修复协议前两者都不可用作研究效应量。

## 产物

- 本报告；`reports/idm_warm_start_summary.{json,csv}`
- 脚本：`validate_idm_warm_start.py`、`run_idm_warm_start_validation.sh`、`test_warm_start_helpers.py`、`test_diagnose_geometry.py`；更新 `frozen_state_lib.py`、`diagnose_idm_agents.py`、`run_hybrid_attribution.py`
- Git 外：`data/frozen_paired/smoke1_cutoff3_warm_v1/`（含 future/pkl/log；不入库）
