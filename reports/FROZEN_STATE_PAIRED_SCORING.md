# Frozen-State Paired Scoring（阶段 1.5 → 1.6 有效性修正）

**工程验证 only**。不得用本报告对研究假设做 go / narrow / no-go。场景来自 `navtest_failures` 1-scene smoke。

最后更新：2026-07-15（阶段 1.6 validity_v2 + **1.7 归因**）

> **阶段 1.7 更新**：tie 指标已修正为 v3；NOC/TTC 方向性与 hybrid 归因见 `reports/FROZEN_DISAGREEMENT_ATTRIBUTION.md`。2658 NOC flip 由单车 `44df645d1b5b584b`（IDM 近静止振荡）造成；扩样技术门 **工程通过**（仍非研究结论）。

## 硬性警告（审查必读）

- **当前仅为 `navtest_failures` 单场景工程验证**，只证明管线可审计、可配对；**不能据此判断研究假设成立**。
- **8192 候选共享一组 source-conditioned 他车 future**：每个交通来源只生成一组 future，全体候选共用；**不是** candidate-conditioned IDM reaction。
- **覆盖范围**：仅 **4 个动态 VEHICLE** 由 `IDMPolicy` 推进；11 个静态车走 `trajectory_policy`；行人/骑行者等 **44** 个 agent 为 log fallback。
- **阶段 1.5 的 max ADE=73.4 m 已确认为 artifact**（见下）：未对 invalid/padding 时间步做 mask。正式指标改用 **common-valid ADE**。
- **NOC/TTC 大幅翻转仍存在**（约 32% / 28%），**可能是**真实交通模型差异，**也可能是** IDM 路线、valid mask、agent 匹配或 future 构造伪影；在未完成代码与异常审查前，不得当作效应量结论。
- **不得简单声称「Top-1 未变化」**：双方在 max score=1.0 处存在大规模并列（Replay 3007 / IDM 321）；`np.argmax` 为 index-tie-broken。阶段 1.5 的 Top-1 结论已更正。
- **下一步必须先审查代码与上述异常，再扩大场景**。

## 阶段 1.6 修正摘要

| 项 | 阶段 1.5 | 阶段 1.6 (validity_v2) |
|---|---|---|
| ADE 定义 | 未 mask（含 invalid） | **common-valid ADE（正式）**；保留 unmasked 仅作诊断 |
| 正式 max ADE | 73.35 m（误） | **15.30 m** |
| token `7cd47126…` | unmasked 73.35 m | common-valid **0.92 m**；73.35 = IDM 续写 vs Replay `valid=0` 的 `[0,0]` |
| ego conditioning | 仅存 live positions | requested/executed hash **一致**；pos/heading 误差 **0** |
| fingerprint | 生成一次再赋给两路 | **阶段内独立双路**（build 对 / score 对）均相等 |
| Top-1 | 声称未变（idx=2） | **并列打破后不可如此解释**；argmax 2 vs 23；Jaccard(max-set)=0.107 |
| NOC flip | 2658 | **2658（不变）** |
| TTC flip | 2316 | **2316（不变）** |
| Replay vs 历史 NR step3 | 逐元素一致 | **仍逐元素一致** |

输出目录（未覆盖 1.5）：`data/frozen_paired/smoke1_cutoff3_validity_v2/`

## 1. Valid 与 ADE

- `agent_valid_at`：有 `valid` 字段时以之为准；**不再**仅因 `[0,0]` 判无效。
- `compare_futures`：仅在双方均 valid 的共同时间步计算正式 ADE；无共同 valid 的 agent 不进入汇总。
- **73.4 m 原因**：`7cd47126ba8f584e` 在 horizon 末步（相对 step=8 / 绝对 step=11）Replay `valid=0` 且位置为 `[0,0]`，IDM 仍输出约 (54.1, -49.5)；unmasked ADE 把该步算进去 → 73.4 m。**common-valid 仅 8 步，max ADE=0.92 m** → **artifact**。

## 2. Ego conditioning 验证

- `plan_idx` 来源：`plan_idx.csv` **step=4 → 1333**（cutoff=3 后首个 Action Policy 计划）；CLI 显式传入，非静默硬编码。
- requested hash = executed hash = `24c92e28…7143`
- pos max error = **0.0 m**；heading max error = **0.0°**
- IDM 确实对 **requested conditioning** 作出反应（轨迹跟踪无偏差）。

## 3. 独立 fingerprint

| 标签 | fingerprint | 阶段内配对 |
|---|---|---|
| build Replay / IDM（hydra scorer @ densereward=false） | `6a8a81e9…ebf5` | **相等** |
| score Replay / IDM（pre-inject，densereward=true） | `0249ede0…3e92` | **相等** |
| 四者全部相等 | 否（预期：build/score 的 hydra 块含 `with_dense_reward_manager` 不同） | 阶段内配对已独立验证 |

fingerprint 含：ego/agents 历史+当前、类型尺寸 valid、地图/灯、vocab hash/shape/dtype、cutoff/horizon/频率、**实际 Hydra scorer 配置 hash**、plan_idx + requested traj hash；**不含**交通 future。

## 4. 并列排名（tie-aware）

- Replay 最高分 1.0，并列 **3007** 个候选；IDM 最高分 1.0，并列 **321**。
- max-score 集合交集 321；Jaccard **0.107**。
- `argmax`（index-tie-broken）：Replay=2，IDM=23 → **选定 index 不同**。
- 双方均非唯一最优 → **不能**解释为「Top-1 未变化」。
- Top-K 同时报告：index-tie-broken / optimistic / pessimistic overlap。
- Kendall τ-b 保留（见 `frozen_paired_compare_summary_v2.json`）。

## 5. ego_progress 为何随他车 future 变化

代码路径（upstream，只读）：

1. `pdm_scorer.py` `_aggregate_scores`（约 L174–L195）：`normalized_progress *= multiplicate_metric_scores`（NOC×DAC 等乘性项）。
2. `dense_reward_manager.py` `_score_proposals_impl`（约 L662）：导出的 `ego_progress` 是 **gated 后的** `_weighted_metrics[PROGRESS]`，不是原始中心线进度。

因此仅改变他车 future → NOC 翻转 → 对应候选的 `ego_progress` 被乘零，可出现 **2256** 条变化。**不是**注入副作用，而是 scorer 合并定义。理论上 raw progress（同 ego 候选）应不变；导出字段会变。

## 6. 四辆 IDM 车诊断（要点）

| token | common-valid max ADE | unmasked max | 备注 |
|---|---:|---:|---|
| `44df645d1b5b584b` | 15.30 | 15.30 | 正式最大偏差来源；需后续查路线 |
| `74c0b539dabe5e9b` | ~11.2 | ~11.2 | IDM 分叉 |
| `3a6b749e38305b9d` | ~10.4 | ~10.4 | IDM 分叉 |
| `7cd47126ba8f584e` | **0.92** | **73.35** | 73.35 为 invalid 伪影；无大步跳变 |

`7cd47126…` 末步 Replay invalid；IDM 沿 lane `47774` 继续，速度约 9.8 m/s，无 >15 m/0.5s 跳变。

## 7. 扩样技术门

| 条件 | 状态 |
|---|---|
| 无 upstream patch 可跑配对 | 通过 |
| Replay 校准（=历史 NR step3） | 通过 |
| ego conditioning 执行一致 | 通过 |
| 独立 fingerprint（阶段内） | 通过 |
| valid-aware ADE | 通过；73.4 artifact 已解释 |
| 并列排名正确报告 | 通过 |
| 15 m 级真实 ADE / NOC 翻转归因 | **未完成** → **暂缓无审查扩样** |
| 多场景 CLI（cutoff/plan_idx） | 已支持；默认仅 smoke convenience |

**结论：具备继续审查与单场景复现的技术条件；不具备“直接扩到多场景做效应量”的条件。** 扩样前须审查 15 m ADE 车与 NOC 翻转机制。

## 产物

**Git 外**：`/mnt/cpfs/prediction/lyyy/myself/WE/data/frozen_paired/smoke1_cutoff3_validity_v2/`
（含 futures/scores、`ego_conditioning_verification.json`、`idm_agent_diagnostics.json`、独立 fingerprint JSON）

**Git 内**：本报告；`reports/frozen_paired_compare_summary_v2.{json,csv}`；修正后脚本。

阶段 1.5 目录 `smoke1_cutoff3/` **保留未覆盖**。
