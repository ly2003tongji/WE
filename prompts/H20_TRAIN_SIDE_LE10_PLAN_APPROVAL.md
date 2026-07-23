# H20：train-side≤10 计划有条件批准（直接粘贴）

将下面 `---` 之间全文发给 H20 Agent（Agent Mode 执行；勿再空转 Plan）。

---

你是 WorldEngine 研究项目的 H20 实验 Agent。全程使用简体中文。

## 批准状态

Mac 侧已**有条件批准**你提交的计划：`train-side ≤10 三源分歧效应量计划`（T0→T5）。  
现在切换到 **Agent Mode**，按该计划执行；下列硬条件**高于**计划中的模糊表述。

当前协作仓 tip 在批准时应为 `abe131d…` 或其后（先 `git pull --ff-only`）；WorldEngine upstream 保持 `fc79b937…`，禁止升级/改 upstream。

## 批准范围（可做）

1. **下载**（本批唯一新数据，≈21.6GB，已批准）：
   - `navtrain_50pct_collision/all_scenarios.pkl`
   - `navtrain_ep_per1/all_scenarios.pkl`  
   落盘到计划中的 HF/sim_engine `scenarios/original/...` 路径；**不**下载 offroad original、**不**下载 3DGS、**不**用 augmented/BWM 冒充。
2. **固化** `reports/train_side_le10_scene_list.md`：5×collision-exclusive + 5×ep-exclusive；加载 pkl 后按后缀匹配 long id；缺则同桶替补并记录。
3. **硬校验**：最终 10 token 与 `navtest_failures`（288）及全部 engineering smoke scene-id 的交集必须为 **∅**；写进 scene list。
4. **逐 scene** cutoff=4 三源（Replay / Restored-IDM / Nexus seed0）；门控 A 入主表；transition flags 全记；degraded 单列。
5. **Nexus 波动**：≥3 个门控 A scene 再跑 seed=1、2；与模型间 flip 对照。
6. **汇总** → `reports/TRAIN_SIDE_LE10_DISAGREEMENT.md` + `train_side_le10_summary.{json,csv}`；短更 `HANDOFF.md`；**停止**。

## 硬条件（必须遵守）

1. **样本隔离**：禁止 navtest_failures / smoke_1scene / cutoff4 冒烟 scene 进入 ≤10；禁止与冒烟数字、cutoff=3 **并表**为同一研究样本。报告开头引用 `CUTOFF4_THREE_SOURCE_SMOKE.md` 仅作协议指针。
2. **offroad 桶**：本批不进主表；scene list 可写“O 保留为后续池”；禁止用 BWM/augmented offroad。
3. **ego conditioning（无 NR plan_idx.csv 时）**：锁定你计划的 **静态可行集（DAC∧Comfort∧Direction=1）上 path-length 中位数 → vocab index**；**三源必须用同一 index**；每 scene `scene_meta.json` + 主报告写死：
   - `source=static_feasible_median_path_length`
   - `plan_idx`、可行集大小、path-length 分位数说明、相关 hash  
   禁止静默 1333；禁止默认旧 Nexus 710（除非算出的 index 恰为 710 并写明）。  
   **报告必须单列一句**：本轮 conditioning ≠ 冒烟 NR `plan_idx=1333`，故效应量**不可与单场景冒烟数字直接数值对比**。
4. **门控**：非 A 或 `pre_cutoff_transition_flags` 有 True → 该 scene **degraded**，不进主效应量表（可附录）。
5. **IDM 参数扰动**：本批不做（按计划）。
6. **停止线**：T5 报告 + HANDOFF 短更后即停。不扩 50、不后训、不 Table1、不 SMART 训练、不自动 commit/push（除非用户另嘱）。
7. **声称边界**：只写初步效应量与描述性难度对照；**禁止**声称研究 go/no-go、held-out 可预测失败。

## 执行顺序（锁定）

```text
T0 冒烟指针 + 隔离声明
T1 下载两 pkl → 固化 10 scene list（含 ∩navtest=∅）
T2 逐 scene restore + Nexus seed0 + PDM
T3 ≥3 scene Nexus seed1/2
T4 汇总 + 难度对照
T5 HANDOFF → 停止
```

## 完成后只回传

```text
1. HEAD SHA；下载是否完成；scene list 路径
2. 10 scene 桶构成；∩navtest 校验结果
3. 门控 A 数 / degraded 数；plan_idx 规则一句话
4. 跨 scene 三 pairwise 分歧分布摘要
5. Nexus 多种子 vs 模型间（一句话）
6. 难度对照一句话
7. 报告路径
8. 是否建议扩 ~50 / SMART（仅建议，本批不执行）
```

开始：先确认 `git pull` 与门禁，再按 T0→T5 执行。

---
