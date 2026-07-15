# Nexus 单场景 ego-conditioned sidecar 可行性

**工程验证 only**。仅当前 1-scene、`cutoff=4`；未扩样、未改 upstream、未跑 BWM/SMART、未做三模型分歧结论。

## 结论

**部分通过。**

公开 Nexus `71c31ca…` + `nuplan.ckpt`（SHA256 `679f6ccf…`）可在本场景上完成：strict-load、5 帧真实历史、非 ego future 无日志 GT 泄漏、官方 encode/decode 等价、ego 16 帧条件保持、同噪声可复现、预注册 3 组 A/B 候选敏感性（最大 ADE≈9.9 m）、token 回映、以及 conditioning candidate 单行 PDM 接口烟测。

未达「技术通过」的原因：

1. **噪声敏感性**：`A+noise0` vs `A+noise1` 最大 ADE≈0.001 m（阈值 ≥0.1 m）。已确认 `z_T` hash 不同且传入 `sample(z_t=…)`；在强 keep_mask 条件下本场景扩散采样几乎确定性。
2. **物理门**：14 个生成车辆中 2 个超出预注册阈值（`7cd47126` 加速度 12.63>12；`803cff56` heading jump>90°）。PDM pack 完整性对这些 token 使用 log fallback；候选敏感性在 fallback **之前**的原始 Nexus 输出上测量。
3. **地图覆盖**：编码得到 LANE=52；`LANE_CONNECTOR`/`STOP_LINE`/`CROSSWALK` 在编码集合中为 0（OpenScene 类型映射与半径筛选限制），按计划降为部分通过。

允许主张：本单场景上 Nexus 公开模型支持 ego-conditioned sidecar 旁路推理接口。  
禁止主张：泛化、held-out、或多场景奖励分歧结论。

## 固定版本

| 项 | 值 |
|---|---|
| 协作仓 | `a5227045cd64b9709f0c279711a37499ffd3a666` |
| WorldEngine | `fc79b937050ed9d68e18add2b480ae72578a7ea5` |
| Nexus | `71c31ca848da94c969322a40f0f4ae2af8ca8129` |
| nuplan fork | `e2aa9f34c6d129424a0d4c70f195c23c55a0666d` |
| MTR | `a5ba7bdafa09a1a355cc34f8a895499a2b14ddb3` |
| ckpt | HF `OpenDriveLab-org/Nexus@6eec52e8…` / 139,612,912 B / SHA256 `679f6ccf…` |
| 场景 | `2021.09.29.15.23.04_veh-28_00601_00802-6326d00e52115da4` |
| cutoff | **4**（历史索引 0..4；不复用 cutoff=3 artifact） |

## 关键门控结果

| 门控 | 结果 |
|---|---|
| strict-load | **通过** `missing=[]` `unexpected=[]`；`num_max_agents=[128,0,0]`；tensor `(129,21,8)` |
| 5 帧历史 | **通过**；14 辆车全程有效 |
| future 泄漏 | **通过**；非 ego future raw=0、`task_mask=0`、normalized=encode(zeros)；hash 已记录 |
| 官方 codec 等价 | **通过**；encode/decode max abs err = 0 |
| 历史 round-trip | **通过**；max pos≈3e-6 m |
| ego 16 帧保持 | **通过**；A/B 全 16 帧 pos max <1e-5 m |
| 同噪声复现 | **通过**；RNG hash 一致，first-4s max err = 0 |
| 候选敏感性 | **通过**；3 组预注册 pairs 均敏感（pair0 ADE 5.25 m @`7cd47126`） |
| 噪声敏感性 | **未达阈值**；ADE≈0.001 m；`z_T` 已审计传入 |
| 物理/schema | **部分**；2 agent 超阈；其余可打包 |
| token 回映 | **通过**；14/14 |
| PDM 单行 | **通过**；plan_idx=1807 行 8 字段均可计算且有限 |

## 候选来源

`plan_idx.csv` step=5（1333）与 cutoff=4 冻结 fingerprint **未**证明一致 → **未**接入 Action Policy。  
按 vocabulary 预注册规则在 GPU 前写入不可变 manifest（见 `reports/nexus_candidate_pair_manifest.json`）。

## 产物位置

- 代码：`adapters/traffic_models/{base,nexus}.py`，`scripts/run_nexus_sidecar_smoke.py`，`tests/test_nexus_adapter.py`
- 轻量摘要：`reports/nexus_sidecar_feasibility_summary.json`
- 原始输出（Git 外）：`/mnt/cpfs/prediction/lyyy/myself/WE/nexus_sidecar/outputs/smoke_cutoff4/`
- 隔离环境/权重/第三方源码（Git 外）：`/mnt/cpfs/prediction/lyyy/myself/WE/nexus_sidecar/`

## 资源

- sidecar 落盘约 22 GiB（硬上限 40 GiB 内）
- GPU 烟测峰值 RSS ≈1.5 GB；PDM ≈1.4 GB / 52 s
- 未修改 `/workspace/worldengine/envs/{simengine,algengine}`，未修改 upstream
