# IDM Restore-Physics Validation（阶段 1.9）

**工程验证 only**。仅当前 1-scene；未扩样、未下载新数据、未改 upstream、未执行 BWM/SMART/Nexus、未设计后训练。

最后更新：2026-07-15

## 目标

阶段 1.8 已确认：cold-start（`slice_scene_from_cutoff` 后重建 IDM）在 `44df645d1b5b584b` 上出现近静止振荡，warm-start（完整场景连续跑 IDM、不做任何 cutoff 干预）在同一车上则完全停滞于 scene 起点，且 cutoff 状态门控为 **C**（不可比较奖励）。二者都不能单独回答：

> 若保留 warm-start 的导航/策略历史、但在 cutoff 严格恢复日志冻结物理状态，原始 32% NOC 翻转是否依然存在？

本阶段构造「warm navigation + cutoff 恢复冻结物理」的 restore-physics sidecar，尝试在严格配对下重新回答该问题。

## 一、只读安全审计（写代码前完成）

逐项核查 upstream（只读，未修改）：

| 接口/状态 | 位置 | 结论 |
|---|---|---|
| `BaseAgent.set_position/set_heading_theta/set_velocity/set_angular_velocity` | `components/agents/base_agent.py` | 均为公开方法，仅写 `_cur_pos/_cur_heading_theta/_cur_velocity/_cur_angular_velocity`，无副作用 |
| `IDMNavigation.update_localization()` | `components/agents/navigation/idm_navigation.py` | 公开方法，从 `agent.current_position` 重新计算 `current_lane/following_original_traj/_route_completion/checkpoints`；不重建 `original_route`/`CenterLane` 对象 |
| `IDMPolicy.act()` 跨 step 状态 | `components/agents/policy/idm_policy.py` | 仅用到 `routing_target_lane`（`move_to_next_road()` 每次 `act()` 自愈，读取 `navigation.current_lane`）；`target_speed/overtake_timer/available_routing_index_range` 在 `enable_lane_change=False`（`__init__` 硬编码，全仓库 grep 确认无任何配置覆盖）下为死代码，从不被 `lane_change_policy()` 使用 |
| `LogPlayController.step()` | `components/agents/controller/log_play_controller.py` | 每步仅读 `agent.trajectory`（由当步 policy 输出刷新），无跨 step 缓存 |
| `BaseAgent.before_step()` | 同上 | 每步自动用（已恢复的）`_cur_pos/_cur_velocity` 刷新 `last_position/last_velocity`，恢复后下一步自动生效，无需手动处理 |
| 非 ego `BaseAgent` 的 `acceleration` | 全仓库 grep | 不存在此持久字段；IDM 车没有需要恢复的 acceleration 状态 |

**内部判断：可安全恢复。** 仅用公开 setter + `navigation.update_localization()` 即可让下一步 IDM 只依赖恢复后的物理状态与保留的 navigation；未触碰任何 private 字段、未重建 navigation、未替换 policy 对象、未改 upstream。

**已记录的残余风险**：`update_localization()` 只做单向（前进）车道/路线切换；若 cutoff 前 warm 阶段已发生 `following_original_traj` 翻转或车道切换，物理回退不会撤销该标志。已在代码中显式检测（`pre_cutoff_transition_flags`），本次运行**全部 40 个 token 均为 False**（cutoff=3 仅 3 步，未发生任何不可逆切换）——风险未触发。

## 二、Restore-physics 构造

新增协作层脚本（不改 upstream）：

- `scripts/frozen_state_lib.py`：新增 `extract_log_agent_full_state_at`（位置/朝向/速度矢量/角速度，角速度用与上游 `compute_angular_velocity` 相同的前向差分公式）与 `restore_agent_physics_via_public_api`（仅调用上述公开接口，返回身份与车道兼容性诊断）。
- `scripts/validate_idm_restore_physics.py`：主流程。
- `scripts/test_restore_physics_helpers.py`：6 个单测（提取正确性、恢复精确写入、身份不变、`update_localization()` 恰好调用一次并如实报告不可逆切换风险）。
- `scripts/run_idm_restore_physics_validation.sh`：编排（单测 → 主流程）。

流程：

1. `truncate_scene_prefix(orig_scene, cutoff+horizon)` 保留完整场景前缀（非 `slice_scene_from_cutoff`）；`agent_policy=idm_policy`/`agent_navigation=idm_navigation`；无 3DGS。
2. `env.reset()` 后 IDM 从 **step 0** 真正开始执行（`idm_truly_starts_at_step=0`，与 1.8 一致）。
3. `env.step()` 推进到 `episode_step==cutoff`（本轮 cutoff=3）。
4. **恢复**：对 `am.all_agents` 中每个在原始日志 object_track 存在且 cutoff 有效的 token（含 ego，共 38 个 common agents），从**完整原始 scene**（非截断）读取 cutoff 帧的 position/heading/velocity/angular_velocity，调用 `set_position/set_heading_theta/set_velocity(value=None)/set_angular_velocity`；再对有 navigation 的 agent 调用 `navigation.update_localization()`。**不重建** policy/navigation。
5. 记录恢复前后 `id(agent.policy)`、`id(agent.navigation)`；**全部相同**（`identity_violations=[]`）。
6. 继续执行与 cold/warm 相同的 post-cutoff ego conditioning（`plan_idx` 来源：`resolve_plan_idx` 显式解析，本轮为 smoke 默认 1333，非静默硬编码，来源已记录在 `plan_idx_resolution.json`）。
7. 生成 Restored-IDM future（cutoff..cutoff+horizon）。

## 三、恢复后严格状态门 → **A**

| 检查 | 结果 |
|---|---|
| 共同 agents 数 | **38** |
| 门控等级 | **A**（严格通过） |
| 声明容差（沿用阶段 1.8，未放宽） | 位置≤0.05m / 航向≤0.5° / 速度≤0.05m/s（严格）；0.10m/1.0°/0.20m/s（软） |
| 超出软容差的 agent | **0** |
| ego 请求/执行误差 | **0.0 m / 0.0°**（hash 相同） |
| policy 对象 identity 恢复前后相同 | **是**（全部 agent） |
| navigation 对象 identity 恢复前后相同 | **是**（全部 agent） |
| route/checkpoint 是否丢失 | 否（`checkpoint_lanes` 恢复前后不变） |
| `update_localization()` 后 lane 与恢复位置相容 | 是（焦点车 `lat_after_restore_m=0.0`，恢复位置精确落在同一 `CenterLane` 上） |
| cutoff 前是否发生不可逆车道切换 | **否**（40/40 token 均 False） |
| map / 交通灯 | 与冻结日志一致 |
| score-stage fingerprint 与 cold-dir 复用的 Replay 分数 fingerprint | **相同**（`fingerprint_matches_cold_replay_fingerprint=true`） |

**结论：门控 A，可直接比较 Replay 与 Restored-IDM future 的奖励。**

## 四、三方轨迹对比（cold / warm / restored）

重点车 `44df645d1b5b584b`：

| | Cold-start（截断重初始化） | Warm-start（完整场景，无恢复） | **Restored-physics（本阶段）** |
|---|---|---|---|
| 门控 | — | C（不可比奖励） | **A** |
| 近静止振荡 | 是 | 否 | **是** |
| 完全停滞 | 否 | 是（净位移 0，卡在起点） | 否 |
| 速度范围 | ~0.03↔1.84 m/s | 恒 0 | **~0.0307↔1.844 m/s** |
| 净位移 | 1.25 m | 0 m | **1.250 m** |
| `current_lane` | `CenterLane`（记录为字符串 "None"） | `CenterLane` | **`CenterLane`（`lat=0.0`，与恢复位置精确相容）** |
| 相对 Replay 净位移 | — | — | Replay **16.54 m** vs Restored **1.25 m** |
| 相对 Replay max ADE | 15.30 m | — | **15.30 m** |
| **cold 与 restored 最大偏差** | — | — | **0.000233 m**（几乎完全重合） |
| 是否有大步瞬移（>15m/0.5s） | 无 | — | **无** |

**回答核心问题**：

- 恢复后 `44df645d` **仍近静止振荡**，**未**完全停滞；
- 恢复后 **拥有有效 lane**（`CenterLane`，位置相容，`lat=0.0`）；
- 恢复后相对 Replay 净位移 **1.25 m**（Replay 16.54 m）；
- **cold 与 restored 最大偏差仅 0.000233 m**——在严格恢复导航历史、物理配对通过（门控 A）的条件下，该车的振荡轨迹与 cold-start sidecar 的轨迹**几乎完全一致**。

这说明：cold-start 的「重截断重建导航」这一工程简化，**在本 token 上并未实质改变轨迹结果**；而阶段 1.8 warm-start 观察到的「完全停滞」，是**未做 cutoff 物理恢复、让 IDM 从 step 0 连续控制到 cutoff 时物理漂移累积**导致的**另一种**（更严重的）伪影，与 cold-start 的振荡是两种不同机制。

## 五、奖励评分（门控 A，已执行）

Replay vs Restored-IDM，相同 8192 候选、PDM Scorer、horizon、地图/交通灯/尺寸、score-stage fingerprint：

| 指标 | 值 |
|---|---|
| NOC：Replay 安全 → Restored 危险 | **2658** |
| NOC：Replay 危险 → Restored 安全 | **0** |
| TTC：Replay 安全 → Restored 危险 | **2316** |
| TTC：Replay 危险 → Restored 安全 | **0** |
| DAC / Comfort / Direction | 全部 **equal**（未异常改变） |
| `ego_progress` | 不等（max diff=1.0），原因同阶段 1.6：PDM 乘性门控副作用，非注入 bug |
| max-score 并列集合大小（Replay / Restored） | 3007 / 2026 |
| 交集 / Jaccard | 2026 / **0.674**（高于 cold-start 的 0.107，说明微小轨迹差异会显著改变并列边界，非矛盾） |
| Restored max-set ⊆ Replay max-set | 是 |
| Kendall τ-b | **0.657** |
| Replay 校准 | 复用 cold-dir 已验证（阶段 1.6 与历史 NR step3 逐元素一致）的 `scores_log_replay.pkl`；fingerprint 严格核对**相同** |

### Single-agent / leave-one-out 归因（5 个动态 IDM agent）

| Hybrid | NOC flip | TTC flip |
|---|---:|---:|
| full_restored（5 车） | 2658 | 2316 |
| **single `44df645d`** | **2658** | **2316** |
| single 其余 4 车 | 0 | 0 |
| **loo 去掉 `44df645d`** | **0** | **0** |
| loo 去掉其余任一车 | 2658 | 2316 |

**确认**：严格配对后，2658 个 NOC 翻转（及 2316 个 TTC 翻转）**仍完全由 `44df645d1b5b584b` 单独造成**，与阶段 1.7 cold-start 归因结论**数值完全一致**。

## 六、判定（按第十节标准）

- cutoff 严格配对：**通过（门控 A）**；
- Restored-IDM 轨迹连续、无瞬移、导航合理（`lat=0`，checkpoint/route 未丢失）；
- NOC/TTC 翻转仍存在并可归因到具体单车；

→ 属于**判定 B：严格配对下仍有分歧**。

> **该差异应视为官方 IDM 在此场景（该车初始近乎静止、且检测到 10.5m 内前方对象触发 `calculate_target_acc` 的零加速度死锁分支）下的真实控制行为，而不是 sidecar 冷启动 bug。**
>
> 仍然只能作为**单场景工程证据**；不构成研究假设成立的证据，不得用于研究 go/narrow/no-go 判定。

## 七、报告措辞修正（第十一节要求）

已将 `reports/IDM_WARM_START_VALIDATION.md` 与 `reports/FROZEN_DISAGREEMENT_ATTRIBUTION.md` 中：

> "32%翻转属于冷启动伪影"

修正为：

> "cold-start振荡模式确认属于截断重初始化伪影相关的工程观察；由于warm-start在cutoff状态不配对（门控C），当时无法直接确定32% NOC/TTC翻转的真实效应量；阶段1.9通过restore-physics严格配对（门控A）重新计算后，确认该翻转在严格配对下依然完全存在且可归因到同一车辆，应视为官方IDM在此场景下的真实行为而非冷启动伪影。"

## 八、扩样判断

- 严格配对（门控 A）下 32% NOC / 28% TTC 翻转**得到确认**，不再是"未知效应量"；
- 但**仍是 1-scene 工程证据**，且该车的异常（近静止 IDM 死锁）本身可能是这个具体场景的边界情况（几乎零速+近距离前方对象），是否代表 IDM 族的普遍行为尚需多场景验证；
- **建议**：技术上已具备扩样条件（restore-physics 协议可复用、门控 A、归因清晰），但**扩样仍需用户明确批准**；扩样时必须使用 restore-physics 协议（而非 cold-start `slice_scene_from_cutoff`），并继续监测/分类近静止振荡类样本，不能默认其为伪影或默认其为普遍效应量。

## 产物

- 本报告；`reports/idm_restore_physics_summary.{json,csv}`
- 脚本：`validate_idm_restore_physics.py`、`run_idm_restore_physics_validation.sh`、`test_restore_physics_helpers.py`；`frozen_state_lib.py` 新增 `extract_log_agent_full_state_at`/`restore_agent_physics_via_public_api`
- Git 外：`data/frozen_paired/smoke1_cutoff3_restore_v1/`（future/scores/pkl/npz/workdir；不入库）
- 更新：`reports/IDM_WARM_START_VALIDATION.md`、`reports/FROZEN_DISAGREEMENT_ATTRIBUTION.md`、`HANDOFF.md`
