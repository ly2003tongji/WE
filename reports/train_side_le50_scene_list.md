# train-side ~50 scene list

**工程开发样本（train rare）；禁止与 navtest_failures / engineering smoke 并表。**

- planned: 50（collision 25 + ep 25，含全部 le10）
- reserve: 16（每桶 8）
- reuse_from_le10 (symlink, skip rerun): **7**
- le10 degraded retained (no rerun): **3**
- new planned runs: **40**
- ∩ navtest tokens: `[]`
- ∩ smoke scene-ids: `[]`
- attempt budget: le10已计10；总 restore 尝试 ≤65；目标门控 A 48–50

| # | 桶 | token | reuse | from_le10 | veh | 理由 |
|---|---|---|---|---|---|---|
| 1 | collision | `0008e2e718e15240` | True | True | veh-28 | retained_le10 |
| 2 | collision | `028a2f461cfe5f1c` | True | True | veh-14 | retained_le10 |
| 3 | collision | `031a7e846efb505b` | False | True | veh-28 | retained_le10 |
| 4 | collision | `26bf0f9e0f245afe` | True | True | veh-08 | retained_le10 |
| 5 | collision | `0aff3a7c4652586c` | True | True | veh-16 | retained_le10 |
| 6 | collision | `40dea91a29fd5e6c` | False | False | veh-08 | veh_log_diverse_fill |
| 7 | collision | `06767db02fb25a07` | False | False | veh-14 | veh_log_diverse_fill |
| 8 | collision | `153106fff89b5e7b` | False | False | veh-17 | veh_log_diverse_fill |
| 9 | collision | `02a4563606ae5a9d` | False | False | veh-28 | veh_log_diverse_fill |
| 10 | collision | `08c0eb09b63b5e0f` | False | False | veh-38 | veh_log_diverse_fill |
| 11 | collision | `2e8f657222765e4f` | False | False | veh-40 | veh_log_diverse_fill |
| 12 | collision | `416ff2910bc253e8` | False | False | veh-43 | veh_log_diverse_fill |
| 13 | collision | `03a4d87fa15e5043` | False | False | veh-50 | veh_log_diverse_fill |
| 14 | collision | `61a68d58ad285312` | False | False | veh-08 | veh_log_diverse_fill |
| 15 | collision | `160ce25b71c05a9c` | False | False | veh-14 | veh_log_diverse_fill |
| 16 | collision | `18a37481e755500e` | False | False | veh-17 | veh_log_diverse_fill |
| 17 | collision | `04b64c9b37f455f8` | False | False | veh-28 | veh_log_diverse_fill |
| 18 | collision | `11ad169e82fa5ded` | False | False | veh-38 | veh_log_diverse_fill |
| 19 | collision | `5981605a73a55c75` | False | False | veh-40 | veh_log_diverse_fill |
| 20 | collision | `6c341178033756b7` | False | False | veh-43 | veh_log_diverse_fill |
| 21 | collision | `07ddd211494e5080` | False | False | veh-50 | veh_log_diverse_fill |
| 22 | collision | `221bd26c26935eca` | False | False | veh-12 | veh_log_diverse_fill |
| 23 | collision | `03c3d0be71495130` | False | False | veh-26 | veh_log_diverse_fill |
| 24 | collision | `11e75aa566b754c5` | False | False | veh-39 | veh_log_diverse_fill |
| 25 | collision | `082bd81928755688` | False | False | veh-45 | veh_log_diverse_fill |
| 26 | ep | `38e8a4b341b7575c` | True | True | veh-08 | retained_le10 |
| 27 | ep | `1ecef78a8bb85ddd` | False | True | veh-14 | retained_le10 |
| 28 | ep | `0a6c2c37c5335ad2` | False | True | veh-26 | retained_le10 |
| 29 | ep | `1836fa024ead5671` | True | True | veh-35 | retained_le10 |
| 30 | ep | `49218363cc6b530f` | True | True | veh-40 | retained_le10 |
| 31 | ep | `64e4220c28b95dcf` | False | False | veh-08 | veh_log_diverse_fill |
| 32 | ep | `55490fa5c1345476` | False | False | veh-14 | veh_log_diverse_fill |
| 33 | ep | `185dc66ec14d596c` | False | False | veh-26 | veh_log_diverse_fill |
| 34 | ep | `190cae7cabd55f50` | False | False | veh-35 | veh_log_diverse_fill |
| 35 | ep | `929c65cfdd615e87` | False | False | veh-40 | veh_log_diverse_fill |
| 36 | ep | `19432f1b6e2858f5` | False | False | veh-50 | veh_log_diverse_fill |
| 37 | ep | `8aab1d0397465557` | False | False | veh-14 | veh_log_diverse_fill |
| 38 | ep | `32b1e375e496597d` | False | False | veh-26 | veh_log_diverse_fill |
| 39 | ep | `291f9fb96513531d` | False | False | veh-35 | veh_log_diverse_fill |
| 40 | ep | `27888e5904615499` | False | False | veh-50 | veh_log_diverse_fill |
| 41 | ep | `0877c1e35805579c` | False | False | veh-12 | veh_log_diverse_fill |
| 42 | ep | `32322d5141715ff8` | False | False | veh-28 | veh_log_diverse_fill |
| 43 | ep | `9904435837f6575f` | False | False | veh-45 | veh_log_diverse_fill |
| 44 | ep | `801cd45a49295ce9` | False | False | veh-12 | veh_log_diverse_fill |
| 45 | ep | `de3e10d777025fbc` | False | False | veh-14 | veh_log_diverse_fill |
| 46 | ep | `599725a7061a5741` | False | False | veh-26 | veh_log_diverse_fill |
| 47 | ep | `79312446c13c5a14` | False | False | veh-28 | veh_log_diverse_fill |
| 48 | ep | `34fd3a3e7a1e5008` | False | False | veh-35 | veh_log_diverse_fill |
| 49 | ep | `e818d53c9628556e` | False | False | veh-45 | veh_log_diverse_fill |
| 50 | ep | `2ae75b3fc86e5896` | False | False | veh-50 | veh_log_diverse_fill |

## Reserve（默认不跑；E2b 按需）

| 桶 | token | veh | 理由 |
|---|---|---|---|
| collision | `9b131890f4585196` | veh-08 | reserve_veh_log_diverse_fill |
| collision | `24c304d148185e84` | veh-12 | reserve_veh_log_diverse_fill |
| collision | `3474b21e76d45316` | veh-14 | reserve_veh_log_diverse_fill |
| collision | `1e8c214d813954a5` | veh-17 | reserve_veh_log_diverse_fill |
| collision | `0ecec41277a8548a` | veh-26 | reserve_veh_log_diverse_fill |
| collision | `0528e164f23c5529` | veh-28 | reserve_veh_log_diverse_fill |
| collision | `17c973648597575e` | veh-38 | reserve_veh_log_diverse_fill |
| collision | `537c9917c20a56a9` | veh-39 | reserve_veh_log_diverse_fill |
| ep | `8f97cba77de256df` | veh-17 | reserve_veh_log_diverse_fill |
| ep | `cc30f7e179a757b8` | veh-08 | reserve_veh_log_diverse_fill |
| ep | `fd06be612af256c4` | veh-12 | reserve_veh_log_diverse_fill |
| ep | `94f83439fcae590c` | veh-17 | reserve_veh_log_diverse_fill |
| ep | `721751577c985b51` | veh-26 | reserve_veh_log_diverse_fill |
| ep | `7ef64baec0a45e86` | veh-28 | reserve_veh_log_diverse_fill |
| ep | `42fe4d68e9e450d7` | veh-35 | reserve_veh_log_diverse_fill |
| ep | `6255219031f65c23` | veh-50 | reserve_veh_log_diverse_fill |

## 资产

- collision pkl: `/mnt/cpfs/prediction/lyyy/myself/WE/data/hf/data/sim_engine/scenarios/original/navtrain_50pct_collision/all_scenarios.pkl`
- ep pkl: `/mnt/cpfs/prediction/lyyy/myself/WE/data/hf/data/sim_engine/scenarios/original/navtrain_ep_per1/all_scenarios.pkl`
- 根目录: `/mnt/cpfs/prediction/lyyy/myself/WE/data/frozen_paired/train_side_le50/<token>/`（reuse → symlink `../train_side_le10/<token>`）

JSON: `/mnt/cpfs/prediction/lyyy/myself/WE/WE/reports/train_side_le50_scene_list.json`
