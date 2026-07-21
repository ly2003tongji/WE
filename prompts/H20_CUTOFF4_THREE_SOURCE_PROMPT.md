# H20 提示词：cutoff=4 三源分歧对齐（Plan Mode → 批准后执行）

将下面 `---` 之间的整段发送给 H20 工作站上的 Cursor Agent。

Mac 侧已锁定决策；本提示词只推进**当前唯一阶段**，不要回退到旧的全量分歧总提示词。

---

你是 WorldEngine 研究项目的 H20 实验 Agent。全程使用简体中文。

## 工作模式：必须先进入 Plan Mode

收到本提示词后**立即切换到 Plan Mode**。在用户（或 Mac 侧转发的批准）之前，只允许：

- 阅读交接文档、代码、配置和现有报告；
- 检查 Git、环境、数据与既有 `data/frozen_paired/`、`nexus_sidecar/` 输出是否仍在；
- 只读核查脚本入口、cutoff 硬编码位置、依赖与风险。

批准前禁止：

- 改代码/配置/报告、下载新数据、装依赖、跑长任务、commit/push。

Plan Mode 交付一份可审批计划，至少包含：

1. 可复用资产（脚本、报告、smoke scene、Nexus ckpt）与缺口；
2. 建议拆成的子步骤、每步输入/输出/通过条件；
3. 拟改动的协作仓文件路径（禁止改 WorldEngine upstream，除非计划里单列且说明为何不可避免）；
4. 关键命令草稿；
5. GPU/磁盘/耗时粗估；
6. 风险与回滚；
7. **第一批建议执行到哪一步就停**（默认：单场景三源冒烟通过即停，扩 train-side 另批批准）。

提交计划后**暂停**，等批准再进 Agent Mode，且只做批准范围内的步骤。

## 启动

协作仓分支：`h20/reproduction`。发送本提示词前 Mac 侧应已 push 含本文档与 `cutoff=4` 决策的提交；H20 **必须先 pull 到最新**再做 Plan。

```bash
# 路径以机器上实际 WE 协作为准（常见 /mnt/cpfs/.../WE 或 HANDOFF 记载路径）
cd "$WE_REPO"   # 若不确定：find /mnt/cpfs -maxdepth 4 -type d -name WE 2>/dev/null | head
git fetch origin
git checkout h20/reproduction
git pull --ff-only origin h20/reproduction
git log -3 --oneline
git status -sb
```

启动门禁（任一失败则 Plan 里写明并暂停，勿瞎改）：

1. `git log -1` 的提交说明或 diff 中应能看到 **cutoff=4 已锁定** / 本提示词文件已存在；
2. 确认下列文件可读：
   - `prompts/H20_CUTOFF4_THREE_SOURCE_PROMPT.md`
   - `research/PAPER_POSITIONING.md`
   - `reports/POST_TRAINING_OPENSOURCE_AUDIT.md`
3. 记录协作仓完整 HEAD SHA；WorldEngine upstream 仍固定：`fc79b937050ed9d68e18add2b480ae72578a7ea5`（勿擅自升级）。

必须先完整阅读（不要靠聊天记忆；以 pull 后的文件为准）：

1. `AGENTS.md`（若存在）
2. `research/CURRENT_STATE.md`（权威；**cutoff=4 已锁定**）
3. `HANDOFF.md`
4. `research/EXPERIMENT_PROTOCOL.md`
5. `research/PAPER_POSITIONING.md`
6. `reports/POST_TRAINING_OPENSOURCE_AUDIT.md`（本阶段不训后训，只知边界）
7. `reports/IDM_RESTORE_PHYSICS_VALIDATION.md`
8. `reports/FROZEN_STATE_PAIRED_SCORING.md`
9. `reports/NEXUS_SIDECAR_FEASIBILITY.md`
10. 相关脚本：`scripts/validate_idm_restore_physics.py` 及 frozen/nexus 旁路入口

**不要**使用过时的 `prompts/H20_DISAGREEMENT_AGENT_PROMPT.md` 作为本轮主任务清单（可查阅历史阶段，但不得覆盖本提示词的锁定决策与停止线）。
## 已锁定决策（不得再议）

- **统一 `cutoff=4`**（对齐 Nexus `N_PAST=5` → `cutoff=N_PAST-1`）。
- 三源：Log **Replay**、**IDM**（restore-physics）、**Nexus**（固定 seed）。
- BWM-Offline 等级 C：**禁止**同场景配对奖励。
- 不做：SMART 训练、后训练、Table 1 复现、OpenWE 基线训练、改论文叙事。
- 288 `navtest_failures`：**不**用于本阶段公式/调参；单场景冒烟可用既有工程 scene，但报告必须标明 **engineering smoke ≠ research go/no-go**。
- IDM 口径：规则式；path 先沿日志参考轨迹，纵向由 IDM 重算；`enable_lane_change=False`。

## 本轮唯一目标

> 在 **cutoff=4** 下，使 Replay / IDM / Nexus 对**同一冻结状态、同一 8192 候选、同一 PDM scorers** 产出可逐候选对齐的子奖励，完成**至少 1 个 scene** 的三源对齐冒烟；旧 cutoff=3 结果**禁止**与 4 并表。

回答的问题（冒烟级，不作研究结论）：

1. Replay/IDM 迁到 4 后，IDM 截止门控是否仍为 **A**？
2. 三源 fingerprint（agent 集合、尺寸、时频、ego conditioning、vocab、scorer）是否对齐？
3. 能否导出逐候选 NOC/TTC/DAC/Comfort/Progress/score 的 pairwise 差异表？
4. 与旧 cutoff=3 的 Replay–IDM 现象是否仍出现（仅描述，不混表）？

## 建议工作分解（计划里可微调，但不得跳过门控）

### A. 只读盘点

- 列出所有硬编码 `cutoff=3` / `cutoff=4` 的协作层脚本与配置。
- 确认 Nexus adapter 仍要求 `cutoff==4`。
- 确认既有 restore-physics 协议步骤可参数化到 4（含 `pre_cutoff_transition_flags` 检查：cutoff 更大 → warm 步数更多，不可逆导航切换风险上升，必须重跑检测）。

### B. 迁移 Replay/IDM → cutoff=4

- 协作层改参/改脚本，**默认不改 upstream**。
- 同一工程 smoke scene：Replay future + Restored-IDM future + 离线 PDM。
- 通过条件：cutoff 门控 **A**；写出与 v3 同级的 summary（新目录，如 `…/smoke*_cutoff4_*`，勿覆盖 cutoff=3 产物）。

### C. 接入 Nexus（同 scene、同 cutoff=4）

- 固定 seed；非 ego future 禁止日志 GT 泄漏（沿用现有 future-leakage 门）。
- 物理门/静态门失败策略与 `NEXUS_SIDECAR_FEASIBILITY.md` 一致并写明。
- ego conditioning 与 Replay/IDM 使用的 plan/ conditioning 对齐方式必须在报告里写死。

### D. 三源并表冒烟

- 统一 future-pack schema + 同一 8192 + 同一 PDM。
- 输出：
  - `reports/CUTOFF4_THREE_SOURCE_SMOKE.md`（结论 + 代码/命令引用）；
  - 摘要 json/csv（小文件可入库；大数组 Git 外）；
- 指标至少：NOC/TTC 安全翻转计数、score range、Top-K / Kendall（按既有协议处理并列）。
- **明确声明**：单场景工程证据，非研究假设成立。

### E. 本批默认停止线

冒烟报告写完即停。  
**不要**自动扩到 10/50 train-side；扩样作为下一批准项，计划里只写“若批准则如何做”的附录即可。

## 禁止事项

- 修改 WorldEngine / Nexus upstream（除非计划单列且用户批准）。
- 下载 >100GB 新数据未经确认。
- 将 NR/R 普通 dense-reward、或未 restore 的 IDM、或 cutoff=3 旧表当作三源正式结果。
- 声称 held-out 泛化、分歧可预测失败、或后训该怎么做。
- 启动后训练 / Table 1 / SMART。

## 同步与提交

- 更新 `HANDOFF.md` 本阶段结论与下一步（短）。
- 大数组、pkl、图像、日志：**不**进 Git。
- 需要 commit 时：只暂存报告/脚本/小摘要；先 `git status`，等用户明确要求再 commit（或按仓库惯例仅在用户要求时提交）。
- 推送仅在用户明确要求时。

## 完成后回传 Mac 侧的摘要格式

```text
1. 协作仓 commit
2. cutoff=4 Replay/IDM 门控结果
3. 三源 fingerprint 是否对齐
4. 主分歧数字（NOC/TTC flip 等）与归因是否仍单车主导
5. 报告路径
6. 阻塞项 / 下一步是否建议开 train-side≤10
```

---

## Mac 侧备注（不必发给 H20）

- 权威状态：`research/CURRENT_STATE.md` §10。
- 批准 Plan 后，H20 只应执行到「单场景三源冒烟」；扩样另开提示词。
- 若 Plan 提出必须改 upstream，Mac 先审再批。
