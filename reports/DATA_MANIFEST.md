# WorldEngine 数据与权重 Manifest

审计时间：2026-07-10（UTC+8）  
官方代码：`OpenDriveLab/WorldEngine@fc79b937050ed9d68e18add2b480ae72578a7ea5`  
Hugging Face 数据版本：`8728616abaf090d195b3bdc7af6aacde40271145`  
ModelScope 数据版本：`ca77ed5c425640dfa413f1bcbe5e8f5a64ba90b5`

## 当前结论

- `运行事实`：Hugging Face 有 174 个文件、4,854,353,280,839 bytes（4.854 TB）；ModelScope 有 180 个文件、4,854,359,953,202 bytes。未下载任何数据文件。
- `运行事实`：ModelScope 完整包含 Hugging Face 的 174 个路径，另外有 6 个文档/脚本文件；共享数据文件大小一致。两端仅 `.gitattributes` 和 `README.md` 大小不同。
- `代码事实`：默认 single-GPU quick test 不是 10 scenes，而是整个 288-scene `navtest_failures`。
- `代码事实`：当前入口没有生效的场景数过滤参数；scenario 是一个 911,923,264-byte 单体 pickle，rare assets 是 3 个 27.9–30.9 GB 的 tar.gz 分片。
- `推断`：可实现的最小 1–10 scene 路径是先取一个 rare-asset 分片、从其中选择 scene 目录，再把单体 scenario pickle 过滤成小 pickle。远端没有公开 “scene → shard” 索引，因此在下载/列 tar 前不能指定任意 scene。
- `代码事实`：闭环 smoke test 不需要官方 OpenScene 原始 sensor blobs、预处理 `nuplan_openscene_navtest.pkl` 或 navtest PDMS cache；SimEngine 会生成 AlgEngine 客户端读取的图像和 annotation，closed-loop dataset 覆写 `load_pdm_infos()` 并填充零 PDM vectors。

## 审计方法与完整文件清单

只读 API：

- Hugging Face：`/api/datasets/OpenDriveLab/WorldEngine/tree/main?recursive=true&expand=true`，按 `Link: rel=next` 分页。
- ModelScope：dataset id `184530` 的 `/api/v1/datasets/184530/repo/tree`，`Recursive=True` 分页。

可重复命令：

```bash
python3 scripts/audit_data_manifest.py --summary
python3 scripts/audit_data_manifest.py --source huggingface
python3 scripts/audit_data_manifest.py --source modelscope
```

后两条命令逐文件输出 `source<TAB>remote_path<TAB>exact_bytes`，是本报告所固定版本的完整 174/180 文件清单；脚本只读远端元数据，不下载 LFS 内容。

### 远端目录汇总

| 远端路径 | 文件数 | 精确 bytes | 十进制体积 | 用途 |
| --- | ---: | ---: | ---: | --- |
| HF 全仓 | 174 | 4,854,353,280,839 | 4.854 TB | 全量发布 |
| `data/alg_engine/ckpts/` | 4 | 1,318,822,815 | 1.319 GB | 预训练/训练/闭环 checkpoint |
| `data/alg_engine/merged_infos_navformer/` | 2 | 16,974,010,355 | 16.974 GB | OpenScene/NavFormer 预处理 annotation |
| `data/alg_engine/pdms_cache/` | 2 | 26,528,290,896 | 26.528 GB | navtrain/navtest PDM cache |
| `data/alg_engine/test_8192_kmeans.npy` | 1 | 3,932,288 | 3.932 MB | trajectory vocabulary 与闭环 post-process |
| `data/sim_engine/assets/navtest/` | 16 | 489,280,880,322 | 489.281 GB | common navtest 3DGS assets |
| `data/sim_engine/assets/navtest_failures/` | 4 | 88,471,241,017 | 88.471 GB | 288 rare scenes 3DGS assets |
| `data/sim_engine/assets/navtrain/` | 130 | 4,183,112,803,554 | 4.183 TB | navtrain 3DGS assets |
| `data/sim_engine/scenarios/original/` | 5 | 32,636,363,656 | 32.636 GB | 原始日志 scenario pickle |
| `data/sim_engine/scenarios/augmented/` | 3 | 15,998,816,912 | 15.999 GB | BWM/增强 scenario pickle |
| 文档图片/GIF及仓库元文件 | 7 | 28,119,024 | 28.119 MB | 非运行数据 |

### AlgEngine 全部文件

| 远端路径 | 精确 bytes | 用途 | rare closed-loop 必需 |
| --- | ---: | --- | --- |
| `data/alg_engine/ckpts/bevformerv2-r50-t1-base_epoch_48.pth` | 244,021,691 | backbone/base checkpoint | 否 |
| `data/alg_engine/ckpts/e2e_vadv2_50pct_ep8.pth` | 434,886,551 | quick test 完整 checkpoint | **是** |
| `data/alg_engine/ckpts/track_map_nuplan_r50_navtrain_100pct_bs1x8.pth` | 319,958,048 | 100% 训练初始化 | 否 |
| `data/alg_engine/ckpts/track_map_nuplan_r50_navtrain_50pct_bs1x8.pth` | 319,956,525 | 50% 训练初始化 | 否；`sim_test.py` 直接加载完整 checkpoint |
| `data/alg_engine/merged_infos_navformer/nuplan_openscene_navtest.pkl` | 848,516,627 | open-loop navtest annotation | 否；闭环时被生成 annotation 覆写 |
| `data/alg_engine/merged_infos_navformer/nuplan_openscene_navtrain.pkl` | 16,125,493,728 | 训练 annotation | 否 |
| `data/alg_engine/pdms_cache/pdm_8192_gt_cache_navtest.pkl` | 2,791,314,758 | navtest PDM token cache | 否；closed-loop subclass 跳过加载 |
| `data/alg_engine/pdms_cache/pdm_8192_gt_cache_navtrain.pkl` | 23,736,976,138 | navtrain PDM token cache | 否 |
| `data/alg_engine/test_8192_kmeans.npy` | 3,932,288 | 8192 trajectory vocabulary/post-process | **是** |

精确引用：

- 模型和 post-process 都读取 vocabulary：`configs/worldengine/e2e_vadv2_50pct.py:246-253`、`closed_loop/sim_test.py:236-248`。
- 闭环 loop 在建 dataset 前覆写 annotation 和 image root：`closed_loop/sim_test.py:142-159`。
- 基类会读取 cache，但 closed-loop subclass 覆写为空实现并生成零 PDM vectors：`mmdet3d_plugin/datasets/navsim_openscene_closed_loop.py:68-82,121-126`。

### Rare closed-loop 文件

| 远端路径 | 精确 bytes | 用途 | 必需性 |
| --- | ---: | --- | --- |
| `data/sim_engine/scenarios/original/navtest_failures/all_scenarios.pkl` | 911,923,264 | 288 rare scenarios | **是；远端不可按 scene 拆分** |
| `data/sim_engine/assets/navtest_failures/assets/part001.tar.gz` | 30,904,422,680 | rare 3DGS asset shard | 1–10 scenes 至少一个 shard；288 全部必需 |
| `data/sim_engine/assets/navtest_failures/assets/part002.tar.gz` | 29,680,462,746 | rare 3DGS asset shard | 同上 |
| `data/sim_engine/assets/navtest_failures/assets/part003.tar.gz` | 27,886,330,535 | rare 3DGS asset shard | 同上 |
| `data/sim_engine/assets/navtest_failures/configs.tar.gz` | 25,056 | reconstruction configs | quick test 代码未引用；暂定非必需 |

三个 asset shards 合计 **88,471,215,961 bytes（88.471 GB / 82.395 GiB）**。

### 其他 scenario 文件

| 远端路径 | 精确 bytes |
| --- | ---: |
| `data/sim_engine/scenarios/original/navtrain_50pct_collision/all_scenarios.pkl` | 19,078,046,784 |
| `data/sim_engine/scenarios/original/navtrain_ep_per1/all_scenarios.pkl` | 2,498,094,267 |
| `data/sim_engine/scenarios/original/navtrain_failures_per1/all_scenarios.pkl` | 2,613,988,099 |
| `data/sim_engine/scenarios/original/navtrain_hydramdp_failures/all_scenarios.pkl` | 7,534,311,242 |
| `data/sim_engine/scenarios/augmented/navtrain_50pct_collision/all_scenarios.pkl` | 3,130,339,854 |
| `data/sim_engine/scenarios/augmented/navtrain_50pct_ep_1pct/all_scenarios.pkl` | 4,998,586,847 |
| `data/sim_engine/scenarios/augmented/navtrain_50pct_offroad/all_scenarios.pkl` | 7,869,890,211 |

### Asset 分片清单

| 数据组 | 文件名范围 | 分片数 | 精确合计 bytes | 单分片范围 |
| --- | --- | ---: | ---: | ---: |
| navtest | `assets/part001.tar.gz`–`part015.tar.gz` + `configs.tar.gz` | 16 | 489,280,880,322 | 28,926,669,837–34,691,933,998 |
| navtest_failures | `assets/part001.tar.gz`–`part003.tar.gz` + `configs.tar.gz` | 4 | 88,471,241,017 | 27,886,330,535–30,904,422,680 |
| navtrain | `assets/part001.tar.gz`–`part129.tar.gz` + `configs.tar.gz` | 130 | 4,183,112,803,554 | 20,110,952,897–36,215,440,526 |

每个分片的精确文件名与 byte size 由 `python3 scripts/audit_data_manifest.py --source huggingface` 输出。它们全部是独立远端文件，可按 shard 下载；平台没有公开 archive 内的 scene 索引。

### ModelScope 独有文件

| 远端路径 | 精确 bytes |
| --- | ---: |
| `extract_datasets.sh` | 3,049 |
| `docs/imgs/README_overall.png` | 533,290 |
| `docs/imgs/nuplan_1.png` | 1,387,872 |
| `docs/imgs/nuplan_2.png` | 1,455,883 |
| `docs/imgs/nuplan_3.png` | 1,403,067 |
| `docs/imgs/nuplan_4.png` | 1,862,622 |

ModelScope 的 `.gitattributes` 为 27,034 bytes、README 为 13,491 bytes；Hugging Face 对应为 2,563 和 11,382 bytes。数据 payload 无差异。

## Quick test 反向路径追踪

| 类别 | 代码读取路径 | 结论 |
| --- | --- | --- |
| checkpoint | `scripts/closed_loop_test.sh:3-8` → `closed_loop/sim_test.py:269-286` | 只需 `e2e_vadv2_50pct_ep8.pth` |
| trajectory vocabulary | config `vocab_path` + `sim.post_process_path` | `test_8192_kmeans.npy` 必需 |
| map | `default_runner.yaml:73-75` → `metric_manager.py:94-100` | nuPlan maps 必需；官方 WorldEngine 数据仓未包含 |
| scenario pkl | `run_testing.sh:25-26,53-59` → `run_simulation.py:22-26` | `navtest_failures/all_scenarios.pkl` 必需，且完整载入 |
| rare 3DGS assets | `run_testing.sh:24-26` → `render/mtgs/mtgs.py:436-474` | 对被选 scene 必需 |
| OpenScene metadata | 闭环 loop 动态合并 SimEngine 生成 annotation | smoke test 不需远端 merged info |
| OpenScene sensor blobs | `run_testing.sh:88` 覆写 `data_root`，loop 覆写 image root | smoke test 不需原始 blobs |
| PDMS cache | closed-loop subclass 覆写 `load_pdm_infos()` | smoke test 不需要；训练/开环需要 |

### 地图缺口

- `代码事实`：SimEngine 默认找 `${WORLDENGINE_ROOT}/data/raw/nuplan/dataset/maps`（`default_runner.yaml:73-75`）。
- `文档事实`：WorldEngine 数据仓要求用户另行准备 nuPlan/OpenScene raw data（`docs/data_organization.md:47-87`）。
- `外部元数据`：公开镜像中的 nuPlan map zip 报告为 970,997,691 bytes（约 0.971 GB）；它不属于本次两端官方 WorldEngine manifest，正式下载前应以官方 nuPlan 源重新核对许可和 hash。
- `运行事实`：在 `/mnt/cpfs/prediction/lyyy` 下没有发现可复用的 `nuplan-maps-v1.0.json`、quick-test checkpoint、vocabulary 或 rare scenario pickle。

## 选择性下载能力

| 对象 | 可按文件选择 | 可按 scene 选择 | 结论 |
| --- | --- | --- | --- |
| checkpoint/vocab/scenario | 是 | scenario pickle 否 | HF `include` 或 ModelScope file path 可取单文件 |
| rare assets | 可按 3 个 tar.gz shard | 远端否 | 下载一个 shard 后可按 tar 内 scene 目录选择解压 |
| maps | 官方外部 archive | 未交代 | map archive 很小，应完整准备 |
| OpenScene blobs/meta | 外部数据 | 通常可按 log，但本 smoke 不需 | 不下载 |

当前代码还存在第二个阻碍：`debug_scene_name` 没有被 runner 使用，`env_builder.py:62-69` 的 scene filter 是 TODO。因此下载后必须生成小 scenario pickle，不能靠 Hydra 现有参数限制到 1–10 scenes。

## 三档空间预算

预算统一使用十进制 GB；括号内说明假设。地图 0.971 GB 是外部估计，不计入“官方文件精确合计”。

### 固定必需项

| 项目 | 精确 bytes |
| --- | ---: |
| 完整 E2E checkpoint | 434,886,551 |
| trajectory vocabulary | 3,932,288 |
| rare scenario 单体 pickle | 911,923,264 |
| **官方固定项合计** | **1,350,742,103（1.351 GB）** |
| nuPlan maps archive | 约 0.971 GB（外部估计） |

### 1 scene

- 最小可操作下载：固定项 1.351 GB + 最大单个 asset shard 30.904 GB + map 约 0.971 GB = **约 33.226 GB**。
- 条件：不预先指定 scene；下载一个 shard 后，从 shard 内选择一个有 scenario key 的 scene。
- 若必须命中任意指定 rare scene，因缺少 scene→shard 映射，保守下载为三个 shards，合计 **约 90.793 GB**。
- 工作盘保守预算：**42–52 GB**（保留一个 shard、解压 ≤1 GB/scene、地图解压和 ≤10 GB 输出/临时文件）。

### 10 scenes

- 若选择同一 shard 中的 10 scenes，下载仍约 **33.226 GB**。
- 若任意指定 10 scenes 跨 shard，保守下载约 **90.793 GB**。
- 工作盘保守预算：**62–127 GB**（每 scene 按 ≤1 GB 解压，输出/临时文件预留 20 GB；范围取决于保留 1 还是 3 个压缩 shard）。

### 288 scenes

- 精确官方 payload：固定项 1,350,742,103 + rare assets 88,471,215,961 = **89,821,958,064 bytes（89.822 GB）**；加地图约为 **90.793 GB**。
- 工作盘保守预算：**380–500 GB**。依据是官方仅称每 scene “几百 MB”，本预算按最高 1 GB/scene 的解压资产上界、保留 88.5 GB 压缩包，并为地图、生成帧、annotation、轨迹、metric 和失败重跑预留 50–120 GB。
- 单个远端文件均小于 100 GB，总下载小于 300 GB；因此不触发下载确认阈值。但阶段 2 不下载数据，阶段 4 前仍先完成环境 import/ABI 检查。

## 缺失与文档不一致

1. `代码事实`：官方 WorldEngine 仓库不含 nuPlan maps、OpenScene raw data；闭环 metric 实际需要 maps。
2. `代码事实`：quick-start 示例显示 “1/10”，默认脚本和 README 实际是 288 scenes。
3. `代码事实`：文档称 SimEngine/AlgEngine 通过 socket 连接，当前默认入口实际使用共享目录文件监控。
4. `代码事实`：文档提供 `debug_scene_name` 印象，但代码没有用它过滤；现有 quick test 不能直接做 1–10 scene。
5. `代码事实`：`configs.tar.gz` 在数据仓存在，但 quick-test asset loader 路径不引用它；用途更像重建/资产配置，是否运行必需官方未交代。
6. `代码事实`：BWM synthetic 数据目录不在本次 smoke test 最小集合；Behaviour World Model integration 在 README roadmap 仍未完成。

## 阶段 2 判定

阶段 2 已完成：两端全部远端文件已通过只读 API 列出和比对；没有下载全量数据。1–10 scenes 可在约 30.2–33.2 GB 下载预算内构造，但需要先下载一个 rare shard 才能建立 scene→shard 映射，并需要生成过滤后的 scenario pickle。288 rare baseline 下载约 90.8 GB、工作盘保守 380–500 GB。
