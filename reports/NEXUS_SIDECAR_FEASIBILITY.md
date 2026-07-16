# Nexus 单场景 ego-conditioned sidecar 可行性（收尾修正）

**工程验证 only**。仅当前 1-scene、`cutoff=4`；未扩样、未改 upstream、未跑 BWM/SMART、未做三模型分歧结论。

## 结论

**可用。**

在物理过滤 + PDM 静态可行门预筛后的**适中** ego candidate pair 上，至少一个 **physics-valid** common-support agent（`44df645d`，pair2 ADE≈0.574 m）对 candidate conditioning 产生可测响应；future pack 与 conditioning 单行 PDM 接口通过（DAC=1、Comfort=1、score 有限）。

噪声敏感性仍近确定性（ADE≈0.001 m），**不再**作为「可用」必要条件；后续统一固定 `seed=0`。

允许主张：本单场景上，合理静态可行 ego candidate 可驱动至少一台通过 Nexus 物理门的 agent 产生 candidate-conditioned 响应，且旁路接口闭环。  
禁止主张：泛化、held-out、或多场景奖励分歧结论。

## 判定对照

| 判定 | 条件 | 本轮 |
|---|---|---|
| **可用** | 合理 pair + ≥1 physics-valid 响应 + pack/PDM 过 | **是**（pair2） |
| 接口可用但行为未验证 | 仅物理失败 agent 响应，或合理 pair 无响应 | 否 |
| 失败 | 筛选/地图/物理/schema 无法闭环 | 否 |

## 固定版本

| 项 | 值 |
|---|---|
| Nexus | `71c31ca848da94c969322a40f0f4ae2af8ca8129` |
| ckpt | SHA256 `679f6ccf…` / 139,612,912 B |
| 场景 | `2021.09.29…-6326d00e52115da4` |
| cutoff | **4** |

## 1. 候选过滤修正

| 项 | 状态 |
|---|---|
| near-stationary 速度不一致 | **`continue`（已修；原 `pass` bug）** |
| heading↔velocity | **已实现并强制** |
| acceleration ≤12 | **已实现并强制** |
| yaw-rate ≤π | **已实现并强制** |
| 连续性 / heading jump | **已实现并强制** |
| 地图可行性 | **不在 vocab 物理过滤中声称**；由 PDM DAC/Direction 静态门承担 |

物理过滤后保留 7868/8192；拒绝：`outside_100m`=324，`heading_vel_misaligned`=3。

## 2. 静态可行预筛（GPU 前）

同 cutoff=4、log-agent futures，对 8192 计算 DAC / Comfort / Direction / lane_keeping（记录）/ 轨迹物理连续性（已在上一步）。

硬保留：`DAC=1 ∧ Comfort=1 ∧ Direction=1` → **3100** 条；与物理过滤交集 **3097**。  
lane_keeping=1 仅 560（**非**硬门）。不做 Replay/Nexus 分歧分析。

## 3. 适中 candidate pairs（非全库最远）

锚点：静态可行集中位路径长度；规则见 manifest。

| pair | A | B | endpoint / RMS (m) | lon / lat Δ | 规则 |
|---|---|---|---|---|---|
| 0 | 710 | 6527 | 2.48 / 1.52 | 2.39 / 0.66 | small_sep |
| 1 | 710 | 5729 | 12.90 / 7.52 | 12.42 / 3.46 | mid_longitudinal |
| 2 | 710 | 2278 | 8.38 / 6.29 | 6.91 / 4.75 | reasonable_lateral |

A/B 均通过静态门。不可变 manifest：`reports/nexus_candidate_pair_manifest.json`。

## 4. 敏感性（两套）

| 套 | 定义 | 本轮 |
|---|---|---|
| raw 全部生成 agent | 含物理失败车 | pair1 上 `7cd47126` ADE≈1.93 m 等 |
| **physics-valid common-support** | 无 NaN；过速度/加速度/heading 门；非 Replay fallback；token 正确 | **pair2：`44df645d` ADE≈0.574 m（≥0.5），>5×同噪误差（0）** |

技术可用结论**仅**基于第 2 套。

## 5. 物理失败诊断（只诊断，不放宽门）

### `7cd47126ba8f584e`（acc 12.63 > 12）

- **来源：模型速度通道** `spd[current]=10.32 → spd[first_future]=4.00`，差分 acc≈12.63。
- 位置差分门：`max_acc_from_pos≈4.31`（不过 12）。
- **不是**单纯坐标变换伪影；**不得**事后放宽 12 m/s²。

### `803cff565b865578`（heading jump ≈179°）

- 近零速（current spd=0）。
- `θ→θ+π` 后 jump≈0.013 rad → **疑似 box 方向 π 等价 / 近零速不稳定**。
- **未**做 heading canonicalization；门保持。

## 6. 地图覆盖

WorldEngine 原始类型：有 `CROSSWALK`（仅 polygon、无 polyline）；**无** `LANE_CONNECTOR` / `STOP_LINE`。

| 类型 | 原因 | 处理 |
|---|---|---|
| CROSSWALK | **adapter 曾遗漏**（只读 polyline） | **已修**：polygon ring 回退；本轮编码 **CROSSWALK=8** |
| LANE_CONNECTOR / STOP_LINE | **上游 OpenScene 场景缺失** | 只记录，不扩任务 |
| LANE | `LANE_SURFACE_*` 映射 | 编码 52 |

## 7. 门控摘要

| 门控 | 结果 |
|---|---|
| strict-load | 通过 |
| future 泄漏 / ego16 / 同噪复现 / token | 通过 |
| 静态门 + 适中 pairs | 通过 |
| physics-valid 敏感性 | **通过**（pair2） |
| 噪声敏感性 | 未达 0.1 m；如实近确定性 |
| 物理/schema | 2 agent 超阈 → pack log fallback；其余可打包 |
| PDM conditioning 行 | **DAC=1, Comfort=1, score≈0.770**（plan_idx=710） |

## 产物

- 代码：`adapters/traffic_models/nexus.py`，`scripts/run_nexus_sidecar_smoke.py`，`tests/test_nexus_adapter.py`（synthetic unit + integration skip）
- 摘要：`reports/nexus_sidecar_feasibility_summary.{json,csv}`，`reports/nexus_candidate_pair_manifest.json`
- 原始输出（Git 外）：`nexus_sidecar/outputs/smoke_cutoff4_closing/`
