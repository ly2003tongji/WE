# H20 提示词：修 IDM restore/门控（Plan Mode → 批准后执行）

将下面 `---` 之间的整段发送给 H20 工作站上的 Cursor Agent。

用户决策（2026-07-23）：**做 1 — 优先修 IDM restore/routing（NoneType + gate C）**，提高门控 A，再谈补跑。不作 SMART/后训；冻结现有 n=36 主信号结论，本轮以工程修复与再评为主。

---

你是 WorldEngine 研究项目的 H20 实验 Agent。全程使用简体中文。

## 工作模式：必须先进入 Plan Mode

收到后**立即 Plan Mode**。批准前只读：排查 le50 degraded 日志/traceback、`validate_idm_restore_physics.py`、IDM navigation/policy 调用链；**禁止**改代码、批量重跑、commit/push。

Plan 必须包含：

1. 当前 HEAD；确认 `reports/TRAIN_SIDE_LE50_DISAGREEMENT.md` 与 summary 可读；
2. **失败模式分类**（至少）：
   - `'NoneType' object is not subscriptable` / restore_failed（约 13）
   - `gate_grade=C`（约 10）
   - `pre_cutoff_transition_true*` / `transition_true*`
   - `missing_restore_scores`（是否多为上游失败连锁）
3. 每种模式的**根因假设**（协作层 vs upstream 行为）与拟改文件（默认只改协作层 `scripts/`）；
4. 最小复现：各模式挑 ≥2 个 token，给出复现命令；
5. 修复方案与**不改变研究协议**的证明（仍 cutoff=4、restore-physics、公开 API；不放宽门控 A 定义冒充通过）；
6. 修复后重跑范围：优先 **degraded 子集**（及必要 smoke）；成功后再决定是否补跑至 A≈48–50；
7. 风险/回滚；停止线。

提交 Plan 后暂停，等批准再改代码/重跑。

## 启动门禁

```bash
cd /mnt/cpfs/prediction/lyyy/myself/WE/WE
git fetch origin && git pull --ff-only origin h20/reproduction
git log -3 --oneline && git status -sb
```

确认：

- HEAD 含 `a36d0cb`（train-side~50）或其后；
- upstream = `fc79b937050ed9d68e18add2b480ae72578a7ea5`（**默认不改**）；
- le50 产物与 degraded 原因在 `reports/train_side_le50_summary.json`。

## 已锁定决策

- **目标**：降低 restore 崩溃与可修复的 gate C；提高可分析门控 A 数量。
- **协议不变**：cutoff=4；`static_feasible_median_path_length`；三源同一 plan_idx；门控 **A** 才入主分析；transition flags 仍全记录。
- **禁止**：放宽 `classify_cutoff_gate` 容差把 C 改叫 A；改 WorldEngine upstream（除非 Plan 证明必须且 Mac 另批）；后训 / Table1 / SMART；新下 >100GB 数据。
- **样本**：优先重跑 le50 **degraded** token；已门控 A 的 36 个默认不重跑（除非修复改变 restore 语义——若重跑须在报告说明前后一致性）。

## 背景事实（来自 le50）

- 尝试 65 触顶；A=36 / degraded=29。
- degraded 原因计数大致：`NoneType…subscriptable`≈13、`restore_failed`≈13、`gate_grade=C`≈10、transition 相关若干、`missing_restore_scores`≈10（常与失败连锁）。
- le10 已记录部分失败与 **IDM routing NoneType** 相关。
- 研究主信号（R↔N）在 n=36 上已非微；本轮是**工程产能**，不是推翻 R↔N。

## 本轮唯一目标

> 定位并修复协作层 IDM warm→restore 路径上的崩溃与可修门控失败；在 degraded 子集上验证门控 A 提升；产出修复报告。可选：修复后对仍失败者分类（不可修 / 需 upstream）。

## 建议工作分解（批准后）

### F0 — 失败审计表

- 从 `train_side_le50_summary.json` + 各 token 目录日志抽出：token、桶、异常栈顶、restore 阶段（warmup / restore / post-gate / score）。
- 产出：`reports/IDM_RESTORE_FAILURE_AUDIT.md`（可先 Plan 草稿结构）。

### F1 — NoneType / restore_failed

- 最小复现 2+ token；定位是 `current_lane is None`、`routing_target_lane`、某 list/dict 下标，还是 scene 抽取缺字段。
- 协作层修复优先：空值守卫、warmup 前 navigation.reset 完整性、缺 lane 时安全 skip agent（须记录）——**不得**静默改物理门控定义。
- 单测或脚本级回归：固定 token 修复前后对比。

### F2 — gate_grade=C

- 区分：物理漂移过大 vs agent 集合不齐 vs 其他。
- 可修项（例）：warmup 步与 cutoff 对齐、restore 后 `update_localization` 顺序、无效 agent 过滤一致性。
- **不可修则保留 C**，报告说明；禁止改阈值灌水。

### F3 — transition flags

- 统计 cutoff=4 下不可逆导航切换频率；评估是否仅记录、或缩短无效 warmup、或标记 degraded（现状可接受则维持）。

### F4 — 重跑 degraded

- 修复后批量重跑 degraded（串行）；更新 per-scene 状态；**不要**无故重跑全部 36 个 A。
- 汇总：修复前/后 A 数、仍失败原因分布。

### F5 — 报告与停止

- `reports/IDM_RESTORE_FIX.md`：根因、diff 文件列表、前后 A/degraded、是否建议补跑至 A≈48–50。
- 短更 `HANDOFF.md` / `CURRENT_STATE.md` 下一指针。
- **停止**。不自动 commit（除非用户另嘱）；不启 SMART/后训。

## 禁止事项

- 把门控 C「改标准」变成 A；
- 修改 upstream 私有状态绕过 restore 协议；
- 声称研究 go/no-go；
- 扩样抽新 yaml 大池（本轮不是再抽 50，是修管线）。

## 完成后回传

```text
1. HEAD；改动的协作层文件列表
2. NoneType / gateC / transition 根因各一句话
3. degraded 重跑：尝试数 / 新升 A 数 / 仍失败数
4. 报告路径
5. 是否建议下一步：补跑 A→48–50 / 冻结 n=36 分析 / 仍需 upstream（仅建议）
```

---

## Mac 侧备注

- 权威：`research/CURRENT_STATE.md` §10.2 选项 1。
- 先批 Plan（根因与拟改文件），再批打补丁与重跑。
