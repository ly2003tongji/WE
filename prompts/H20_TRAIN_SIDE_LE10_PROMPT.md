# H20 提示词：train-side ≤10 三源分歧效应量（Plan Mode → 批准后执行）

将下面 `---` 之间的整段发送给 H20 工作站上的 Cursor Agent。

前提：cutoff=4 单场景三源冒烟已完成（`reports/CUTOFF4_THREE_SOURCE_SMOKE.md`）。本轮只做 **train-side ≤10** 初步效应量，不作研究 go/no-go，不后训。

---

你是 WorldEngine 研究项目的 H20 实验 Agent。全程使用简体中文。

## 工作模式：必须先进入 Plan Mode

收到本提示词后**立即切换到 Plan Mode**。批准前只读：核查文档、脚本、数据覆盖、资产是否够跑 train-side；**禁止**改代码、下载大包、跑批量评分、commit/push。

Plan 必须包含：

1. 当前 HEAD / 是否已有 `CUTOFF4_THREE_SOURCE_SMOKE.md`；若缺失，说明如何从本机未推送提交或远端恢复；
2. **train-side 场景候选池怎么定义**（来源、过滤、与 navtest_failures 的隔离证明）；
3. 拟选 ≤10 个 scene 的名单或抽样规则 + 资产/地图是否已在盘；
4. 每 scene 的三源流水线（复用 cutoff4 冒烟脚本）与输出目录约定；
5. 同模型波动方案（至少 Nexus 多 seed；IDM 扰动可选）；
6. 难度基线字段（车辆数、TTC 相关、最坏/均值奖励等——与既有协议对齐）；
7. 汇总指标与报告路径；GPU/磁盘/耗时；风险与回滚；
8. **第一批停止线**：≤10 scene 汇总报告写完即停（不自动扩到 50）。

提交 Plan 后暂停，等 Mac/用户批准再执行。

## 启动门禁

```bash
cd /mnt/cpfs/prediction/lyyy/myself/WE/WE
git fetch origin && git pull --ff-only origin h20/reproduction
git log -5 --oneline && git status -sb
```

必须确认：

- 可读 `research/CURRENT_STATE.md`（**cutoff=4 已锁定**）；
- 可读 `reports/CUTOFF4_THREE_SOURCE_SMOKE.md` 与 `reports/cutoff4_three_source_summary.json`（若仅在未 push 的本地提交，先 `git status`/`git log` 说清，Plan 里写如何纳入本轮，**不要**重跑整条 cutoff4 冒烟除非文件损坏）；
- WorldEngine upstream 仍为 `fc79b937050ed9d68e18add2b480ae72578a7ea5`；
- 复用协作层 cutoff=4 管线（restore-physics、三源并表脚本），**禁止**改 upstream。

## 已锁定决策（不得再议）

- **cutoff=4**；三源：Replay / IDM（restore-physics）/ Nexus（默认 seed=0；波动实验另加 seed）。
- ego conditioning：延续冒烟 provenance 原则——**显式 plan_idx csv + step**，写入每 scene 报告；禁止静默默认；禁止沿用 Nexus 旧 710（除非某 train scene 的 csv 真是该值且写明）。
- BWM-Offline：**不**入同场景配对。
- **288 `navtest_failures` / 既有 engineering smoke scene：不得进入本轮 ≤10 的抽样池**，也不得用于调公式。冒烟数字只可在讨论中作协议对照，**禁止与 train-side 结果并表混报为同一研究样本**。
- 不做：后训练、Table 1、SMART 训练、OpenWE 基线、论文叙事改写。
- IDM 口径不变：规则式；path 先沿日志；纵向 IDM；`enable_lane_change=False`。

## 本轮唯一目标

> 在 **train-side 长尾**上，用与冒烟**相同**的 cutoff=4 冻结协议，对 **≤10** 个 scene 跑 Replay/IDM/Nexus 候选级奖励，估计效应量量级，并对照同模型波动与简单难度基线——**仅初步**，不作 go/no-go。

本轮要回答（初步）：

1. 多 scene 上是否仍常见非微小 NOC/TTC/排序分歧？
2. 分歧是否常被单车/单模式主导，还是更分散？
3. 模型间差异是否明显大于 Nexus 多种子（及可选 IDM 扰动）波动？
4. 相对简单难度代理，是否还有剩余信号值得扩到 ~50 / 考虑 SMART？

## 场景选择约束（Plan 里必须写清）

1. 来源必须是 **train split / navtrain 侧长尾**（发现 yaml、既有 train rare 列表、或文档允许的 train-side 池）——**不是** `navtest_failures`。
2. 每个候选 scene：检查 3DGS asset / maps / scenario pkl 是否已在盘；缺资产则换 scene 或在 Plan 中列出最小下载（>100GB 须暂停请示）。
3. 优先多样（交互类型/日志），避免 10 个全是同一 log 切片；最终名单 ≤10。
4. 记录：scene token、来源清单、资产路径、为何入选。

## 建议执行分解（批准后）

### T0 — 固化冒烟基线指针

- 在新报告开头引用 cutoff4 冒烟报告路径与 HEAD；声明本轮样本与 engineering smoke **隔离**。

### T1 — 选定 ≤10 train-side scenes

- 产出 `reports/train_side_le10_scene_list.md`（或 json）：token、来源、资产状态。

### T2 — 逐 scene 三源评分

- 协议同冒烟：cutoff=4、restore-physics、门控 A 才入主分析；transition flags 全记录。
- 输出目录示例：`data/frozen_paired/train_side_le10/<token>/`（Git 外大文件）。
- 门控非 A 或 Nexus 大规模 fallback：该 scene 标 `degraded`，主表可单列，不删原始日志。

### T3 — 同模型波动（最低要求）

- 同一批 scene（可先子集 ≥3）：Nexus **至少 2 个额外 seed**（如 1、2）相对 seed=0，报 NOC/TTC flip 或 reward 差异分布。
- 可选：小幅 IDM 参数扰动（若成本高，Plan 里可降为附录/第二批）。

### T4 — 汇总与难度对照

- 跨 scene 汇总：pairwise（R↔I、R↔N、I↔N）安全翻转率/计数分布、排序相关、单车主导比例。
- 难度代理：每 scene 记录车辆数、与 TTC/最坏奖励等相关标量（能从现有 scorer/meta 稳定取得的字段）；做**描述性**对照（分歧 vs 难度），不作正式因果结论。
- 产出：
  - `reports/TRAIN_SIDE_LE10_DISAGREEMENT.md`
  - `reports/train_side_le10_summary.{json,csv}`（小文件可入库）

### T5 — 交接停止

- 短更 `HANDOFF.md` +（可选）`CURRENT_STATE.md` 下一阶段指针。
- **停止**。不扩 50、不后训、不自动 commit（除非用户另嘱）。

## 禁止事项

- 把 navtest engineering smoke 或 288 rare test 混进 ≤10 研究开发样本。
- 与 cutoff=3 旧表并表。
- 修改 WorldEngine / Nexus upstream。
- 声称 held-out 可预测失败、或研究假设已成立。
- 下载超大新数据未经确认。

## 完成后回传 Mac 侧

```text
1. HEAD SHA；冒烟报告是否在树内
2. ≤10 scene 名单与 train-side 来源证明
3. 门控 A 通过数 / degraded 数
4. 跨 scene 主分歧分布摘要（三 pairwise）
5. Nexus 多种子波动 vs 模型间差异（一句话）
6. 难度对照一句话
7. 报告路径
8. 是否建议扩 ~50 或启动 SMART 可用性（仅建议）
```

---

## Mac 侧备注（不必发给 H20）

- 权威：`research/CURRENT_STATE.md` §10 第 3 步。
- 先批 Plan（尤其是 scene 池与资产），再批执行。
- 若 H20 冒烟提交未 push，先让其 push 或 Mac pull 后再发本提示词，避免门禁卡在缺报告。
