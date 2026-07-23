# train-side ≤10 scene list

**工程开发样本（train rare）；禁止与 navtest_failures / engineering smoke / cutoff4 冒烟并表。**

- collision-exclusive: 5；ep-exclusive: 5；offroad: 0（保留后续池；不用 BWM/augmented）
- ∩ navtest_failures tokens: `[]` （必须为空）
- ∩ smoke scene-ids: `[]` （必须为空）
- rare-train ∪ ∩ navtest = 0（yaml 级已核验）

| # | 桶 | token | scene_id | veh | 入选理由 |
|---|---|---|---|---|---|
| 1 | collision | `0008e2e718e15240` | `2021.09.15.14.00.15_veh-28_01274_01543-0008e2e718e15240` | veh-28 | preferred_token |
| 2 | collision | `028a2f461cfe5f1c` | `2021.07.16.01.22.41_veh-14_02626_04289-028a2f461cfe5f1c` | veh-14 | preferred_token |
| 3 | collision | `031a7e846efb505b` | `2021.09.15.13.16.40_veh-28_00366_00631-031a7e846efb505b` | veh-28 | preferred_token |
| 4 | collision | `26bf0f9e0f245afe` | `2021.08.17.18.11.12_veh-08_00629_01599-26bf0f9e0f245afe` | veh-08 | veh_diverse_fill |
| 5 | collision | `0aff3a7c4652586c` | `2021.06.23.15.56.12_veh-16_01308_04289-0aff3a7c4652586c` | veh-16 | veh_diverse_fill |
| 6 | ep | `38e8a4b341b7575c` | `2021.08.17.16.57.11_veh-08_01200_01636-38e8a4b341b7575c` | veh-08 | veh_diverse_fill |
| 7 | ep | `1ecef78a8bb85ddd` | `2021.07.16.01.22.41_veh-14_00572_01716-1ecef78a8bb85ddd` | veh-14 | veh_diverse_fill |
| 8 | ep | `0a6c2c37c5335ad2` | `2021.07.09.23.23.48_veh-26_04648_06327-0a6c2c37c5335ad2` | veh-26 | veh_diverse_fill |
| 9 | ep | `1836fa024ead5671` | `2021.05.12.23.36.44_veh-35_01133_01535-1836fa024ead5671` | veh-35 | veh_diverse_fill |
| 10 | ep | `49218363cc6b530f` | `2021.08.31.14.01.15_veh-40_00407_00497-49218363cc6b530f` | veh-40 | veh_diverse_fill |

## 资产

- collision pkl: `/mnt/cpfs/prediction/lyyy/myself/WE/data/hf/data/sim_engine/scenarios/original/navtrain_50pct_collision/all_scenarios.pkl`
- ep pkl: `/mnt/cpfs/prediction/lyyy/myself/WE/data/hf/data/sim_engine/scenarios/original/navtrain_ep_per1/all_scenarios.pkl`
- 单 scene 抽取: `/mnt/cpfs/prediction/lyyy/myself/WE/data/frozen_paired/train_side_le10/<token>/scene/all_scenarios.pkl`
- 地图: 既有 `data/maps/extracted`；本批不下载 3DGS

JSON: `/mnt/cpfs/prediction/lyyy/myself/WE/WE/reports/train_side_le10_scene_list.json`
