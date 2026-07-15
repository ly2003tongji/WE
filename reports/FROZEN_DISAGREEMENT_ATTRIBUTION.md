# Frozen Disagreement Attribution（阶段 1.7）

**工程验证 only**。不得据此判断研究假设成立。仅 1-scene frozen-state；未扩样、未下载新数据、未改 upstream。

最后更新：2026-07-15

## 结论（一句话）

2658 个 NOC 翻转（及 2316 个 TTC 翻转）**几乎完全由单车 `44df645d1b5b584b` 的 IDM 轨迹造成**；该车在 IDM 下近静止振荡，而 Replay 继续前驶，几何上更贴近/碰撞 ego 候选。

## 1. 修正后的 tie 指标（v3）

| 指标 | 值 |
|---|---|
| Replay 最高分 | 1.0 |
| IDM 最高分 | 1.0 |
| Replay max-set 大小 | **3007** |
| IDM max-set 大小 | **321** |
| 交集 | **321** |
| 并集 | 3007 |
| Jaccard | **321/3007 ≈ 0.10675** |
| IDM max-set ⊆ Replay max-set | **是**（`frac_idm_max_in_replay_max=1.0`） |
| 可声称唯一 Top-1 翻转 | **否** |

已删除错误的 optimistic/pessimistic Top-K（曾出现 >1）。保留 index-tie-broken Top-K **仅作实现行为**；正式排序用 max-set / Jaccard / Kendall τ-b。单元测试：`scripts/test_ranking_metrics.py`（无并列/部分并列/全并列，比例∈[0,1]）。

## 2. NOC / TTC 方向性翻转（Full IDM vs Replay）

| | NOC | TTC |
|---|---:|---:|
| Replay 安全 → IDM 危险 | **2658** | **2316** |
| Replay 危险 → IDM 安全 | **0** | **0** |
| 双方安全 | 2395 | 2074 |
| 双方危险 | 3139 | 3802 |
| 总 flip | 2658 (32.4%) | 2316 (28.3%) |

全部为单向：Replay 更“安全标签” → IDM 更“危险标签”。

## 3. 四车 hybrid 归因

IDM 动态车：`3a6b749e38305b9d`, `44df645d1b5b584b`, `74c0b539dabe5e9b`, `7cd47126ba8f584e`。

相对 Replay baseline：

| Hybrid | NOC flip | 占 Full 比例 | TTC flip |
|---|---:|---:|---:|
| Full（4 车 IDM） | 2658 | 1.00 | 2316 |
| **Single `44df645d`** | **2658** | **1.00** | **2316** |
| Single `3a6b749e` | 0 | 0 | 0 |
| Single `74c0b539` | 0 | 0 | 0 |
| Single `7cd47126` | 0 | 0 | 0 |
| LOO 去掉 `44df645d` | **0** | 0 | 0 |
| LOO 去掉其他任一 | 2658 | 1.00 | 2316 |

**回答**：2658 个 NOC 翻转由 **`44df645d1b5b584b` 单独**造成；其余三车在本 scene 的 NOC/TTC 翻转上无增量贡献。

校准：所有 hybrid 共用同一 score-stage fingerprint；DAC/Comfort/Direction 在 Full/Single 中相对 Replay **未异常改变**（equal）；Replay 与历史 NR step3 仍逐元素一致。

## 4. 15.30 m ADE（`44df645d1b5b584b`）原因

| 检查 | 结果 |
|---|---|
| Token 全程不变 | 是 |
| common-valid | 9/9，无内部空洞 |
| 最大单步位移 | < 1 m（无瞬移） |
| Replay 净位移 | **16.54 m** |
| IDM 净位移 | **1.25 m** |
| 行为 | **近静止振荡**：速度/位移在 ~0.03 与 ~1.8 m/s 间交替；`current_lane` 多为 `None` |
| 分类 | **`idm_route_or_control_oscillation_near_stationary`** |

**不是** token 错配、坐标变换错误、valid/padding 伪影或非物理瞬移。
**是** IDM 路线/控制失效导致停滞振荡，Replay 驶离 → ADE 累积至 15.3 m。该停滞轨迹留在 ego 候选走廊内，解释了 NOC/TTC 单向恶化。

其余三车：路径/速度剖面分叉（ADE 约 10–11 m）或轻微差（`7cd47126` common-valid max 0.92 m）；本 scene 不驱动 NOC flip。

## 5. 碰撞几何验证（轻量）

对 `44df645d`：随机抽 5 个「Replay 安全→IDM 危险」候选，用 vocab ego 轨迹 + 定向包围盒 SAT：

- **5/5** 满足 IDM 下更近或发生近似盒碰撞（rate=1.0）
- 双方安全 / 双方危险各抽 5 个作对照

结论：flip 与 **该 agent 轨迹靠近 ego** 一致，而非尺寸/坐标/valid 标签错误。
（注：此为几何可行性检查，非完整 PDM at-fault 复刻。）

## 6. 扩样技术门

| 条件 | 状态 |
|---|---|
| tie 指标数学有效且有测试 | **通过** |
| 15.3 m 可解释、无 token/valid/坐标/瞬移错误 | **通过**（IDM 振荡） |
| NOC/TTC 可归因到具体 IDM agent | **通过**（单车 `44df645d`） |
| 代表候选碰撞几何与标签一致 | **通过** |
| Replay 校准 / ego 条件 / fingerprint | **通过** |

**建议：具备扩到多场景做工程效应量分布的技术条件**；仍不得用 `navtest_failures` 下研究 go/no-go。扩样时应继续报告方向性 flip、max-set Jaccard，并警惕 IDM 停滞/振荡类伪影。

## 产物

- 本报告；`reports/frozen_paired_compare_summary_v3.{json,csv}`；`reports/frozen_disagreement_attribution_summary.{json,csv}`
- 脚本：`ranking_metrics.py`, `test_ranking_metrics.py`, `run_hybrid_attribution.py`, `diagnose_idm_agents.py`；更新 `compare_frozen_rewards.py`
- Git 外：`data/frozen_paired/smoke1_cutoff3_attr_v1/`（hybrid scores/诊断；不入库）
