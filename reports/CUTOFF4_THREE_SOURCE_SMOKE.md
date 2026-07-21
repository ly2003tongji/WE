# cutoff=4 三源对齐冒烟（Replay / Restored-IDM / Nexus）

**工程冒烟 only（engineering smoke ≠ research go/no-go）。**  
场景来自 `navtest_failures` 工程子集；**禁止**与旧 cutoff=3 数字并表；本报告不作研究假设成立/否决结论。

最后更新：2026-07-21

## Provenance（硬条件）

| 项 | 值 |
|---|---|
| 协作仓 HEAD | `47d0e4f4cd52f02167523970b0e00d357e13802c` |
| WorldEngine upstream | `fc79b937050ed9d68e18add2b480ae72578a7ea5`（未改） |
| scene | `2021.09.29.15.23.04_veh-28_00601_00802-6326d00e52115da4` |
| cutoff / horizon | **4** / 9 |
| ego conditioning | `plan_idx.csv` **step=5 → plan_idx=1333** |
| plan_idx.csv | `/mnt/cpfs/prediction/lyyy/myself/WE/WE/upstream/WorldEngine/experiments/closed_loop_exps/e2e_vadv2_50pct-disagreement-smoke1-NR-20260715/navtest_failures_NR/plan_traj/plan_idx.csv` |
| 禁止 | Nexus 旧 conditioning `plan_idx=710` |
| SCORER_CONFIG.num_history | **未改**（仍为 4）；研究锚点 cutoff=4 ≠ 声称等于 num_history-1 |
| Nexus seed | `0` |
| Nexus physics→log fallback tokens | `['7cd47126ba8f584e', '803cff565b865578']` |

## 1. Replay / IDM 门控（cutoff=4）

| 检查 | 结果 |
|---|---|
| cutoff gate | **A** |
| allow reward compare | True |
| `pre_cutoff_transition_flags` True 数 | **0** / 43 |
| 产物目录 | `/mnt/cpfs/prediction/lyyy/myself/WE/data/frozen_paired/smoke1_cutoff4_restore_v1`（未覆盖任何 `smoke1_cutoff3_*`） |

## 2. 三源 fingerprint

| 源 | fingerprint |
|---|---|
| Replay | `fcc78476ccb1e2972dc834db967c912f3a5ef2a78fe946962b49f792ce449a56` |
| Restored-IDM | `fcc78476ccb1e2972dc834db967c912f3a5ef2a78fe946962b49f792ce449a56` |
| Nexus | `fcc78476ccb1e2972dc834db967c912f3a5ef2a78fe946962b49f792ce449a56` |
| 三者对齐 | **True** |
| ego requested hash 与 restore 一致 | True |

## 3. Pairwise 分歧（8192 候选）

| Pair | NOC safe→danger | NOC danger→safe | NOC total | TTC safe→danger | TTC total |
|---|---:|---:|---:|---:|---:|
| Replay vs IDM | 2746 | 138 | 2884 | 2517 | 2524 |
| Replay vs Nexus | 1316 | 200 | 1516 | 1177 | 1394 |
| IDM vs Nexus | 579 | 2071 | 2650 | 259 | 2068 |

Kendall τ-b / Top-K / score range：见 `reports/cutoff4_three_source_summary.json`（并列按既有 tie-aware 协议）。

## 4. 归因提示（仅 Replay–IDM；工程级）

- full Restored-IDM NOC flip：2884
- top single-agent hybrid：`single_44df645d` NOC flip=2746 tokens=['44df645d1b5b584b']（约占 full 的 95%）
- 结论口径：**仍为单车主导**（`44df645d1b5b584b`）

与旧 cutoff=3 现象的关系：**仅文字描述、不混表**——cutoff=4 下仍出现显著 Replay→IDM 的 NOC/TTC 安全翻转，且归因仍由同一焦点车主导；这是同类工程现象的再现描述，**不得**与 cutoff=3 数字并表或外推为研究 go/no-go。

## 5. 冒烟问答（不作研究结论）

1. IDM 截止门控是否仍为 A？→ **A**（transition True=0）
2. 三源 fingerprint 是否对齐？→ **True**
3. 能否导出逐候选 pairwise 子奖励差异？→ **能**（scores pkl 在 Git 外；摘要 json/csv 入库）
4. 与旧 cutoff=3 Replay–IDM 现象？→ 同类工程现象再现（单车主导翻转），**禁止并表**

## 6. 命令与脚本入口

- Restore/IDM：`scripts/run_idm_restore_physics_validation.sh`（`DEFAULT_CUTOFF=4`，`--plan-idx-csv` step=5）
- 三源：`scripts/run_cutoff4_three_source_smoke.sh`（Phase A nexus env / Phase B simengine）
- 摘要：`reports/cutoff4_three_source_summary.{json,csv}`；`reports/idm_restore_physics_cutoff4_summary.{json,csv}`

## 7. 停止线

本批到此停止。不扩 train-side、不后训、不 Table 1、不 SMART、不 BWM 配对。未自动 commit/push。
