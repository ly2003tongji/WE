# train-side ≤10 三源分歧（初步效应量）

**初步效应量与描述性难度对照 only；不作研究 go/no-go；不作 held-out 可预测失败主张。**

最后更新：2026-07-21

## 协议指针与样本隔离（硬条件）

| 项 | 值 |
|---|---|
| 协作仓 HEAD | `abe131da3be35f735a6515428a37b5765e86c805` |
| WorldEngine upstream | `fc79b937050ed9d68e18add2b480ae72578a7ea5`（未改） |
| cutoff / horizon | **4** / 9 |
| 协议冒烟指针 | [`reports/CUTOFF4_THREE_SOURCE_SMOKE.md`](CUTOFF4_THREE_SOURCE_SMOKE.md)（**仅协议**，非本轮样本） |
| 本轮样本 | train-side rare：5×collision-exclusive + 5×ep-exclusive |
| 与冒烟数值 | **不可直接对比**：本轮 ego conditioning = `static_feasible_median_path_length`，冒烟 = NR `plan_idx=1333` |
| offroad 桶 | 本批不进主表；保留后续池；**不用** BWM/augmented |
| 禁止并表 | navtest_failures / engineering smoke / cutoff=3 |

**本轮 conditioning ≠ 冒烟 NR plan_idx=1333，故效应量不可与单场景冒烟数字直接数值对比。**

- scene list: [`reports/train_side_le10_scene_list.md`](train_side_le10_scene_list.md)
- ∩ navtest tokens: `[]`
- ∩ smoke scene-ids: `[]`

## 门控与样本量

- 门控 A 入主表：**7**
- degraded（非 A 或 transition flag）：**3**
- plan_idx 规则：静态可行集 DAC∧Comfort∧Direction=1 上 path-length 中位数 → vocab index（三源同一 index）

## 主表 pairwise 分歧分布（仅门控 A）

| pair | NOC total_flip | TTC total_flip | Kendall τ_b |
|---|---|---|---|
| replay_vs_idm | n=7 min=0 med=0 mean=0.0 max=0 | n=7 min=0 med=0 mean=0.6 max=4 | n=7 min=1 med=1 mean=1.0 max=1 |
| replay_vs_nexus | n=7 min=1 med=67 mean=237.3 max=1311 | n=7 min=0 med=45 mean=239.9 max=1355 | n=7 min=1 med=1 mean=0.9 max=1 |
| idm_vs_nexus | n=7 min=1 med=67 mean=237.3 max=1311 | n=7 min=0 med=45 mean=240.4 max=1355 | n=7 min=1 med=1 mean=0.9 max=1 |

## Nexus 多种子 vs 模型间

Nexus 多种子相对 seed0 的 Replay↔Nexus NOC_flip |Δ| 中位=19.0；同批模型间 Replay↔Nexus NOC_flip 中位=105.0（Replay↔IDM 中位=0.0）。初步：模型间（尤其 vs Nexus）明显大于多种子波动。

## 难度对照（描述性）

描述性：主表 n=7，R↔I NOC_flip 几乎全 0（中位 0），与难度代理的相关不可估；R↔N 分歧为主信号。不作因果结论。

## 逐 scene（含 degraded）

| token | 桶 | gate_A | degraded | plan_idx | R↔I NOC | R↔N NOC | I↔N NOC |
|---|---|---|---|---|---|---|---|
| `0008e2e718e15240` | collision | True | False | 2709 | 0 | 67 | 67 |
| `028a2f461cfe5f1c` | collision | True | False | 5680 | 0 | 105 | 105 |
| `031a7e846efb505b` | collision | False | True | 710 | — | — | — |
| `26bf0f9e0f245afe` | collision | True | False | 201 | 0 | 141 | 141 |
| `0aff3a7c4652586c` | collision | True | False | 5063 | 0 | 31 | 31 |
| `38e8a4b341b7575c` | ep | True | False | 1804 | 0 | 1311 | 1311 |
| `1ecef78a8bb85ddd` | ep | False | True | 4523 | — | — | — |
| `0a6c2c37c5335ad2` | ep | False | True | 4477 | — | — | — |
| `1836fa024ead5671` | ep | True | False | 1842 | 0 | 5 | 5 |
| `49218363cc6b530f` | ep | True | False | 898 | 0 | 1 | 1 |

## 建议（本批不执行）

初步：主表 Replay↔Nexus NOC 仍见非微小分歧（R↔I 近 0），可考虑扩至 ~50 稳分位数；SMART 仅当扩样后仍有剩余信号再议。本批不执行。

## 产物

- `reports/train_side_le10_summary.json`
- `reports/train_side_le10_summary.csv`
- 每 scene：`data/frozen_paired/train_side_le10/<token>/`
