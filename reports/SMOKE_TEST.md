# WorldEngine 单 H20 闭环 Smoke Test

执行时间：2026-07-10（UTC+8）  
官方代码：`OpenDriveLab/WorldEngine@fc79b937050ed9d68e18add2b480ae72578a7ea5`  
设备：1× NVIDIA H20，Driver 570.133.20  
状态：1 scene、10 scenes、NR/R 均成功

## 当前结论

- `运行事实`：官方开放代码已在单 H20 上完成真实闭环：
  `SimEngine 3DGS render → AlgEngine NAVFormer inference → trajectory → simulator step → PDM metric`。
- `运行事实`：1-scene 和 10-scene runner 均为 100% 技术执行成功，无进程、CUDA、asset、map、checkpoint 或数据格式失败。
- `运行事实`：10-scene NR 与 IDM reactive 均无 at-fault collision；按 README “无碰撞且无 off-road” 的 SR 定义，两者均为 7/10 = 70%，因为相同 3 scenes 的 drivable-area compliance 为 0。
- `运行事实`：10-scene IDM 相对 NR 的平均 ego progress 从 0.48689 增至 0.51100，PDMS-style score 从 0.61120 增至 0.62125；样本仅 10 个且来自同一 asset shard，不能外推为正式效果结论。
- `运行事实`：1-scene profile 峰值显存 13,638 MiB、峰值 GPU util 97%，说明单 H20 显存余量充足。
- `代码/运行事实`：不需要 OpenScene 原始 metadata/sensor blobs 或 navtest PDMS cache；当前闭环需要完整 E2E checkpoint、trajectory vocabulary、scenario pickle、matching 3DGS asset 和 nuPlan maps。

## 数据与完整性

### 下载集合

WorldEngine 数据版本：Hugging Face commit `8728616abaf090d195b3bdc7af6aacde40271145`。

| 文件 | bytes | SHA-256 |
| --- | ---: | --- |
| `e2e_vadv2_50pct_ep8.pth` | 434,886,551 | `c335e7ba1b92464a3bd11daa2e3c2174aa929cb9c59b59b4bfdbd8bb9fb9fb0f` |
| `pdm_8192_gt_cache_navtest.pkl` | 2,791,314,758 | `58d82e73bc3dfaa6464ea7d90dee86f8e0139ea1679145d610a9babf86fe7316` |
| `test_8192_kmeans.npy` | 3,932,288 | `cc44a31e75a53406db59f026f0358de97931e726f10254542f98d2a87a38ad35` |
| `navtest_failures/all_scenarios.pkl` | 911,923,264 | `27203d52454f2538e6a70e95136b85ee8df35822f7707dc0091b4943ca7ad41d` |
| `navtest_failures/assets/part003.tar.gz` | 27,886,330,535 | `102d16b9f608ac8f2314aef2ba3e1907cee9db7e2a3d094b1833b4ff60de0d05` |
| `nuplan-maps-v1.1.zip` | 970,997,691 | `444860429f9a3bcf89a6459d683fa82eb9219aa259d8d9b8cdefdb37f0b56b05` |
| **合计** | **32,999,385,087** | 每项均通过本地 `sha256sum` |

前 5 项本地 hash 与官方 Hugging Face LFS oid 逐项一致。

`pdm_8192_gt_cache_navtest.pkl` 已在首次清单中下载并校验，但后续完整代码追踪确认它对 closed-loop inference 非必需：`NavSimOpenSceneE2EClosedLoop.load_pdm_infos()` 为空实现，并在 `get_data_info()` 中填充零 PDM vectors（`navsim_openscene_closed_loop.py:68-82,121-126`）。因此实际最小远端 payload 应从上表合计减去 2,791,314,758 bytes，即 **30,208,070,329 bytes**。

nuPlan 官方 NAVSIM 脚本指向 Motional S3，但本机对该域名的 HTTPS/HTTP 多次出现 connection reset。地图改从公开 Hugging Face mirror `pengxiang/nuplan_maps@a7afa514750a61fd2646745661612c25b556d044` 下载；文件名、精确大小和 LFS SHA-256 与公开元数据一致。该 mirror 不是 WorldEngine 官方数据仓，正式长期归档时应从 Motional 官方源重新获取并复核相同 hash。

### 子集构造

- part003 包含 90 个完整 scene asset 目录（630 个 tar members，每 scene 7 members）。
- 1-scene 子集：
  - scene：`2021.09.29.15.23.04_veh-28_00601_00802-6326d00e52115da4`
  - asset：337,869,033 bytes
  - filtered scenario pickle：1 key，约 3.0 MB
- 10-scene 子集：
  - 10 个 scene 均从 part003 选择；
  - asset：3,532,229,491 bytes；
  - filtered scenario pickle：10 keys。
- 使用 `scripts/prepare_smoke_subset.py` 做路径穿越检查，只提取普通文件/目录；没有把压缩包完整展开。
- 使用 `scripts/setup_smoke_links.sh` 把 CPFS 数据以 symlink 接入官方 `data/` 路径，没有复制权重或资产。

## 配置

| 项目 | 值 |
| --- | --- |
| AlgEngine config | `projects/AlgEngine/configs/worldengine/e2e_vadv2_50pct.py` |
| checkpoint | `data/alg_engine/ckpts/e2e_vadv2_50pct_ep8.pth` |
| split | 从官方 `navtest_failures` 过滤的 1/10 scenes |
| asset shard | `navtest_failures/assets/part003.tar.gz` |
| NR | logged trajectory replay |
| R | `agent_policy=idm_policy` + `agent_navigation=idm_navigation` |
| SimEngine seed | 默认 `0`（`default_common.yaml`） |
| AlgEngine seed | `sim_test.py --seed` 默认 `0` |
| Alg deterministic | 未传 `--deterministic` |
| history/future | 4 history + 8 simulation；日志显示 step 0–11 |

模型预检在正式闭环前完成：

```text
model_class=NAVFormer
checkpoint loaded successfully
GPU allocated=226.5 MiB
GPU reserved=240.0 MiB
```

## 精确命令

### 下载

```bash
hf download OpenDriveLab/WorldEngine \
  data/alg_engine/ckpts/e2e_vadv2_50pct_ep8.pth \
  data/alg_engine/pdms_cache/pdm_8192_gt_cache_navtest.pkl \
  data/alg_engine/test_8192_kmeans.npy \
  data/sim_engine/scenarios/original/navtest_failures/all_scenarios.pkl \
  data/sim_engine/assets/navtest_failures/assets/part003.tar.gz \
  --repo-type dataset \
  --revision 8728616abaf090d195b3bdc7af6aacde40271145 \
  --local-dir /mnt/cpfs/prediction/lyyy/myself/WE/data/hf \
  --max-workers 3
```

WorldEngine 5 个文件实际下载耗时 636 秒；地图另耗时约 28 秒，总远端 payload 约 33.0 GB。没有下载全量数据。

### 构造子集

```bash
/workspace/worldengine/envs/simengine/bin/python \
  scripts/prepare_smoke_subset.py \
  --scenario-pkl /mnt/cpfs/prediction/lyyy/myself/WE/data/hf/data/sim_engine/scenarios/original/navtest_failures/all_scenarios.pkl \
  --asset-tar /mnt/cpfs/prediction/lyyy/myself/WE/data/hf/data/sim_engine/assets/navtest_failures/assets/part003.tar.gz \
  --output-root /mnt/cpfs/prediction/lyyy/myself/WE/data/smoke_1scene \
  --scene 2021.09.29.15.23.04_veh-28_00601_00802-6326d00e52115da4
```

1-scene 与 10-scene 的完整 scene 参数保存在 shell history 和 `scripts/prepare_smoke_subset.py` 的可重复调用方式中；报告不重复 10 个长 ID，实际结果 CSV 列出了固定顺序。

### 运行

```bash
# 1 scene NR
bash scripts/run_smoke_1scene.sh

# 1 scene R
SMOKE_MODEL_NAME=e2e_vadv2_50pct-smoke1-reactive \
SMOKE_REACT_TYPE=R \
bash scripts/run_smoke_1scene.sh

# 10 scenes NR
SMOKE_SUBSET_NAME=smoke_10scenes \
SMOKE_MODEL_NAME=e2e_vadv2_50pct-smoke10 \
SMOKE_REACT_TYPE=NR \
bash scripts/run_smoke_1scene.sh

# 10 scenes R
SMOKE_SUBSET_NAME=smoke_10scenes \
SMOKE_MODEL_NAME=e2e_vadv2_50pct-smoke10-reactive \
SMOKE_REACT_TYPE=R \
bash scripts/run_smoke_1scene.sh
```

wrapper 只设置持久 env、干净 PATH、环境变量、symlink 和官方 `run_testing.sh` 参数，不修改上游代码。

## 运行结果

### 1 scene

| 指标 | NR | IDM reactive |
| --- | ---: | ---: |
| runner technical success | 1/1 | 1/1 |
| no-at-fault-collision | 1.00000 | 1.00000 |
| drivable-area compliance | 1.00000 | 1.00000 |
| ego progress | 0.25179 | 0.41051 |
| TTC within bound | 1.00000 | 1.00000 |
| comfort | 1.00000 | 1.00000 |
| driving-direction compliance | 1.00000 | 1.00000 |
| score | 0.68825 | 0.75438 |

`推断`：单场景差异不能说明 IDM 普遍优于 replay，只证明 reactive 分支确实改变了 rollout/metric。

### 10 scenes

| 指标 | NR | IDM reactive | R - NR |
| --- | ---: | ---: | ---: |
| runner technical success | 10/10 | 10/10 | 0 |
| no-at-fault-collision | 1.00000 | 1.00000 | 0 |
| drivable-area compliance | 0.70000 | 0.70000 | 0 |
| closed-loop SR（无 collision/off-road） | 0.70000 | 0.70000 | 0 |
| ego progress | 0.48689 | 0.51100 | +0.02411 |
| TTC within bound | 1.00000 | 1.00000 | 0 |
| comfort | 1.00000 | 1.00000 | 0 |
| driving-direction compliance | 0.95000 | 0.95000 | 0 |
| score / PDMS-style | 0.61120 | 0.62125 | +0.01005 |

三个 off-road scenes 在 NR/R 中相同；其中两个 first violation step 从 NR 的 7/6 变为 R 的 7/7，第三个均为 7。R 主要通过个别场景 progress 改变平均分，没有改变本 10-scene 子集的 SR。

### 耗时、吞吐与显存

| 项目 | 结果 |
| --- | ---: |
| 1-scene runner duration | 55.566 s |
| 1-scene profile wall time | 79.7 s |
| 1-scene peak GPU memory | 13,638 MiB |
| 1-scene peak GPU util | 97% |
| 1-scene GPU samples | 145（0.5 s 间隔） |
| 10-scene NR runner durations sum | 459.509 s |
| 10-scene NR mean per scene | 45.951 s |
| 10-scene NR wall time | 533.422 s |
| 10-scene R runner durations sum | 483.732 s |
| 10-scene R mean per scene | 48.373 s |
| 10-scene R wall time | 571.879 s |

R 比 NR 的 runner time 增加约 5.3%，wall time 增加约 7.2%。10-scene NR 按 120 simulator steps 计算，runner 吞吐约 0.261 step/s，端到端墙钟约 0.225 step/s。

### 输出

| 实验 | 输出体积 | camera images |
| --- | ---: | ---: |
| 1-scene profile NR | 393 MB | 96 |
| 1-scene R | 393 MB | 96 |
| 10-scene NR | 3.1 GB | 960 |
| 10-scene R | 2.8 GB | 960 |

每 scene 为 12 frames × 8 cameras。`clean_temp_files=True` 后 `frames/`、`merged_ann_files/` 和 trajectory `.npy` 被清理，保留 `plan_idx.csv`、sensor blobs、metric CSV、runner report 和 completion flag。

关键输出：

```text
upstream/WorldEngine/experiments/closed_loop_exps/
├── e2e_vadv2_50pct-smoke1-profile/navtest_failures_NR/
├── e2e_vadv2_50pct-smoke1-reactive/navtest_failures_R/
├── e2e_vadv2_50pct-smoke10/navtest_failures_NR/
└── e2e_vadv2_50pct-smoke10-reactive/navtest_failures_R/
```

## 发现的实现细节

1. `运行事实`：`configs.tar.gz` 对本闭环入口不必需；只有匹配 scene 的 `background/*.ckpt` 与 `road_height_map/*` 被使用。
2. `代码/运行事实`：navtest PDMS cache 对 closed-loop inference 非必需；移除 cache symlink 后，`e2e_vadv2_50pct-smoke1-nocache` 仍以 1/1 成功完成。PDM metric 由 SimEngine `MetricManager` 实时计算。
3. `运行事实`：metric 在 simulator step 11 保存，而 AlgEngine 在检测 completion flag 前仍处理最后一个 `_11.pkl` 并保存 trajectory step 12。
4. `运行事实`：runner `succeeded=true` 仅表示技术执行成功，不等同于 closed-loop SR；本 10-scene 例子 technical success 100%，但按 no collision + no off-road 的 SR 为 70%。
5. `运行事实`：README 的 “5–10 分钟” 对本机 10 scenes 基本成立（NR 8.9 分钟、R 9.5 分钟），但默认脚本是 288 scenes，不能据此推断完整默认运行只需 5–10 分钟。

## 阶段 4 判定

阶段 4 已完成。1–10 scene 最小闭环、NR replay 和 IDM reactive 均有真实运行证据。下一步可在不重新下载 asset 的情况下做这 10 scenes 的重复种子/稳定性检查；正式 288 baseline 仍需下载另外两个 rare asset shards（额外约 60.6 GB）并预留数小时运行和约 90 GB 输出。
