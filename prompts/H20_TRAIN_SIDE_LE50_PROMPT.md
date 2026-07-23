# H20 提示词：train-side ~50 扩样（Plan Mode → 批准后执行）

将下面 `---` 之间的整段发送给 H20 工作站上的 Cursor Agent。

用户决策（2026-07-23）：**先扩充**。在 ≤10 初步效应量之上，同协议扩到约 **50** 个 train-side scene，稳定 R↔N 分位数并检验 R↔I≈0 是否持续。不作 go/no-go、不后训、不上 SMART。

---

你是 WorldEngine 研究项目的 H20 实验 Agent。全程使用简体中文。

## 工作模式：必须先进入 Plan Mode

收到本提示词后**立即切换到 Plan Mode**。批准前只读：盘点 ≤10 产物、脚本、pkl 覆盖、磁盘与耗时；**禁止**改代码、下载（除 Plan 中明示且 <100GB 的替补）、批量跑分、commit/push。

Plan 必须包含：

1. 当前 HEAD；确认 `reports/TRAIN_SIDE_LE10_DISAGREEMENT.md` 可读；
2. **~50 名单规则**：如何从 collision / ep yaml 池抽样；是否保留全部 ≤10；如何保证 ∩navtest=∅、∩smoke=∅；
3. 桶比例建议（默认锁定见下）与 veh/log 多样性策略；
4. 复用哪些脚本（`run_train_side_le10.sh` 等）及需改的 CLI/并行度；
5. Nexus 多种子：全量 50 太贵时的抽样方案（最低要求见下）；
6. 输出目录、汇总报告路径、GPU/磁盘/墙钟估计；
7. 风险（门控 degraded 率、磁盘、Nexus OOM）与回滚；
8. **停止线**：~50 主汇总报告写完即停（不自动 held-out / 后训 / SMART）。

提交 Plan 后暂停，等 Mac/用户批准再执行。

## 启动门禁

```bash
cd /mnt/cpfs/prediction/lyyy/myself/WE/WE
git fetch origin && git pull --ff-only origin h20/reproduction
git log -5 --oneline && git status -sb
```

必须确认：

- 可读 `research/CURRENT_STATE.md`（扩样决策）、`reports/TRAIN_SIDE_LE10_DISAGREEMENT.md`、`reports/train_side_le10_scene_list.md`；
- collision / ep original pkl 仍在盘（路径见 le10 scene list）；**本批默认不新下 offroad / 3DGS / BWM**；
- WorldEngine upstream = `fc79b937050ed9d68e18add2b480ae72578a7ea5`（勿改）；
- 复用 train-side≤10 协作层管线；禁止改 upstream。

## 已锁定决策（不得再议）

- **cutoff=4**，horizon=9；三源：Replay / Restored-IDM / Nexus。
- **ego conditioning**：`static_feasible_median_path_length`（与 ≤10 相同）；三源同一 `plan_idx`；每 scene 写 provenance。禁止静默 1333；禁止默认 710（除非算出恰为 710 并写明）。
- **样本**：仅 train rare yaml（collision + ep）；**保留并纳入**已有 ≤10（勿重跑已成功门控 A 的 scene，除非产物损坏——Plan 写清复用策略）。
- ∩ `navtest_failures` = ∅；∩ engineering smoke scene-id = ∅；禁止与冒烟 / cutoff=3 / ≤10 数字**混成同一统计表时不标注来源**（汇总可分层：le10 子集 vs 扩样新增 vs 全 ~50）。
- **offroad**：仍不进主表（无 original pkl）；保留后续池。
- **不做**：后训、Table 1、SMART 训练、held-out 预测主张、BWM 配对、IDM 参数扰动（除非 Plan 附录且另批）。
- 声称边界：效应量与分位数描述；**禁止**研究 go/no-go。

## 本轮唯一目标

> 将同协议 train-side 样本扩到约 **50** 个门控可分析 scene（含已有 ≤10），报告 **R↔N / I↔N** 分歧分位数是否稳定，并检验 **R↔I≈0** 是否在更大样本上持续；Nexus 多种子波动对照按抽样执行。

要回答：

1. 全样本（~50）上 R↔N NOC/TTC flip 的中位/IQR/尾部是否仍非微？
2. R↔I 是否仍近 0？若否，翻转集中在哪些 scene？
3. 模型间（尤其 R↔N）是否仍明显大于 Nexus 多种子波动（抽样）？
4. degraded 率是否可接受？是否建议下一步 SMART 或收缩主张？

## 抽样规则（Plan 可微调，须写死）

默认目标：**约 50 个计划 scene**（含已有 10），主分析以 **门控 A** 为准；若 degraded 过多，可同桶替补使 A 数接近 50，但总尝试上限与替补规则须在 Plan 写明（建议尝试上限 ≤60–65，避免无限换 scene）。

桶比例（默认）：

- **~25 collision-exclusive + ~25 ep-exclusive**（含已有各 5）；
- 已有 le10 名单全部保留；新增 token 从同 yaml 池抽，**veh-* / log 多样性**优先；
- 硬校验写入 `reports/train_side_le50_scene_list.md`。

数据：优先只用已下载的 collision + ep pkl；**禁止**为扩样下载 >100GB 新数据；若某 token 不在 pkl，同桶替补。

## 建议执行分解（批准后）

### E0 — 指针与复用清单

- 报告声明继承 le10 协议；列出可复用的 `data/frozen_paired/train_side_le10/<token>/`（门控 A 且产物完整者跳过重跑）。

### E1 — 固化 ~50 scene list

- `reports/train_side_le50_scene_list.{md,json}`：token、桶、long id、是否复用 le10、∩navtest=∅。

### E2 — 逐 scene 三源（增量）

- 协议同 le10；输出根目录建议：
  - 复用：`data/frozen_paired/train_side_le10/<token>/`（只读引用）
  - 新增：`data/frozen_paired/train_side_le50/<token>/`
- 或统一迁到 `train_side_le50/` 并用 symlink/清单记录复用——Plan 选一种并写清。
- 门控非 A → degraded，主表可单列。

### E3 — Nexus 多种子（抽样）

- **最低**：在门控 A 的 scene 中分层抽 **≥10** 个（C/E 尽量均衡），跑 seed=1、2，对照 seed=0。
- 指标：相对 seed0 的 R↔N NOC/TTC |Δ| 分布 vs 模型间 R↔N 分布。

### E4 — 汇总

- `reports/TRAIN_SIDE_LE50_DISAGREEMENT.md`
- `reports/train_side_le50_summary.{json,csv}`
- 必须分节报告：
  - 全样本 R↔I / R↔N / I↔N 分位数；
  - **R↔I≈0 是否持续**（A 中 R↔I NOC=0 的比例）；
  - le10 子集 vs 新增子集（可选，防混读）；
  - 多种子 vs 模型间；
  - degraded 率与原因摘要；
  - 描述性难度对照（同 le10 字段）。
- 明确：**不可与冒烟 NR-1333 数字直接对比**。

### E5 — 交接停止

- 短更 `HANDOFF.md` + `CURRENT_STATE.md` 下一决策指针；
- **停止**。不自动 commit（除非用户另嘱）；不启 SMART/后训。

## 禁止事项

- 把 navtest / smoke 混进名单；
- 用 BWM/augmented offroad 凑数；
- 改 upstream；
- 声称 held-out 可预测或假设已成立；
- 无批准确地下载 >100GB。

## 完成后回传 Mac 侧

```text
1. HEAD；复用 le10 scene 数 / 新跑数 / 门控 A 数 / degraded 数
2. scene list 路径；∩navtest 校验
3. 全样本三 pairwise 分位数摘要（尤其 R↔I 是否仍≈0，R↔N 中位/IQR）
4. 多种子抽样结论一句话
5. 报告路径
6. 建议：SMART / 再扩 / 收缩主张（仅建议）
```

---

## Mac 侧备注（不必发给 H20）

- 权威：`research/CURRENT_STATE.md` §10（先扩充）。
- 先批 Plan（名单规模、复用策略、多种子抽样、耗时），再批执行。
- 预期墙钟：在 le10「1–2 天/10 scene」量级上线性外推约数日；Plan 须给如实估计。
