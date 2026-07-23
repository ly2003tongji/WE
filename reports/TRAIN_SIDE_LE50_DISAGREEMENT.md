# train-side ~50 三源分歧（效应量 / 分位数）

**效应量与分位数描述 only；不作研究 go/no-go；不作 held-out 可预测失败主张。**

最后更新：2026-07-23

## 协议指针与样本隔离

| 项 | 值 |
|---|---|
| 协作仓 HEAD | `2b3040bc76a51d265011baf09956dcc89e503df5` |
| WorldEngine upstream | `fc79b937050ed9d68e18add2b480ae72578a7ea5`（未改） |
| cutoff / horizon | **4** / 9 |
| 协议继承 | [`TRAIN_SIDE_LE10_DISAGREEMENT.md`](TRAIN_SIDE_LE10_DISAGREEMENT.md) |
| conditioning | `static_feasible_median_path_length` |
| 冒烟对比 | **不可直接对比** NR `plan_idx=1333` |

**本轮 conditioning ≠ 冒烟 NR plan_idx=1333，故效应量不可与单场景冒烟数字直接数值对比。**

- scene list: [`train_side_le50_scene_list.md`](train_side_le50_scene_list.md)
- ∩ navtest: `[]`
- ∩ smoke: `[]`

## 样本量与尝试

- 复用 le10 (symlink skip): **7**
- 新 restore 尝试: **55**；总尝试（含 le10 10）: **65** / 65
- 门控 A 主表: **36**；degraded: **29**
- 激活 reserve: `['0528e164f23c5529', '0ecec41277a8548a', '17c973648597575e', '1e8c214d813954a5', '24c304d148185e84', '3474b21e76d45316', '42fe4d68e9e450d7', '537c9917c20a56a9', '721751577c985b51', '7ef64baec0a45e86', '8f97cba77de256df', '94f83439fcae590c', '9b131890f4585196', 'cc30f7e179a757b8', 'fd06be612af256c4']`
- degraded 原因计数: `{'gate_grade=C': 10, 'transition_true=2': 1, 'missing_restore_scores': 10, 'pre_cutoff_transition_true=1': 5, 'restore_failed': 13, "'NoneType' object is not subscriptable": 13, 'transition_true=3': 3, 'transition_true=1': 5, 'pre_cutoff_transition_true=2': 1, 'transition_true=5': 1}`

## 全样本 pairwise 分位数（仅门控 A）

### 全样本

| pair | NOC total_flip | TTC total_flip |
|---|---|---|
| replay_vs_idm | n=36 min=0 p25=0 med=0 p75=66 IQR=66 max=720 | n=36 min=0 p25=0 med=0 p75=71 IQR=71 max=2791 |
| replay_vs_nexus | n=36 min=0 p25=30 med=135 p75=361 IQR=332 max=1815 | n=36 min=0 p25=36 med=116 p75=345 IQR=309 max=2894 |
| idm_vs_nexus | n=36 min=0 p25=63 med=142 p75=440 IQR=377 max=1854 | n=36 min=0 p25=49 med=132 p75=350 IQR=302 max=1642 |

**R↔I NOC=0 比例（门控 A）**: `0.5555555555555556`

### le10 子集（门控 A）

| pair | NOC total_flip | TTC total_flip |
|---|---|---|
| replay_vs_idm | n=7 min=0 p25=0 med=0 p75=0 IQR=0 max=0 | n=7 min=0 p25=0 med=0 p75=0 IQR=0 max=4 |
| replay_vs_nexus | n=7 min=1 p25=18 med=67 p75=123 IQR=105 max=1311 | n=7 min=0 p25=21 med=45 p75=118 IQR=98 max=1355 |
| idm_vs_nexus | n=7 min=1 p25=18 med=67 p75=123 IQR=105 max=1311 | n=7 min=0 p25=21 med=45 p75=120 IQR=100 max=1355 |

### 新增子集（门控 A）

| pair | NOC total_flip | TTC total_flip |
|---|---|---|
| replay_vs_idm | n=29 min=0 p25=0 med=3 p75=135 IQR=135 max=720 | n=29 min=0 p25=0 med=2 p75=173 IQR=173 max=2791 |
| replay_vs_nexus | n=29 min=0 p25=52 med=168 p75=392 IQR=340 max=1815 | n=29 min=0 p25=50 med=129 p75=409 IQR=359 max=2894 |
| idm_vs_nexus | n=29 min=0 p25=100 med=168 p75=479 IQR=379 max=1854 | n=29 min=0 p25=78 med=140 p75=351 IQR=273 max=1642 |

## Nexus 多种子 vs 模型间

- 多种子 scene 数（有 seed1/2）: **10**

Nexus 多种子相对 seed0 的 R↔N NOC |Δ| 中位=17.0；同批模型间 R↔N NOC 中位=135.0（R↔I 中位=0.0）。初步：模型间（尤其 vs Nexus）明显大于多种子波动。

## 难度对照（描述性）

描述性：主表 n=36，Replay score_min 与 R↔N NOC_flip Pearson≈nan；不作因果结论。

## 建议（本批不执行）

门控 A=36 未达目标 48–50（尝试已触顶 65；degraded 偏高，多见 restore/gateC）；建议优先修 IDM routing/门控再扩，SMART 暂缓。本批不执行。

## 产物

- `reports/train_side_le50_summary.json`
- `reports/train_side_le50_summary.csv`
- 每 scene：`data/frozen_paired/train_side_le50/<token>/`
