# H20：cutoff=4 计划有条件批准（直接粘贴）

将下面 `---` 之间全文发给 H20 Agent（Agent Mode 执行；勿再开一轮空 Plan）。

---

你是 WorldEngine 研究项目的 H20 实验 Agent。全程使用简体中文。

## 批准状态

Mac 侧已**有条件批准**你提交的计划：`cutoff=4 三源对齐冒烟计划`（S0→S5）。  
现在切换到 **Agent Mode**，严格按该计划执行；下列条件**高于**计划原文中的模糊表述。

## 启动（第 0 步，必须先完成）

```bash
cd /mnt/cpfs/prediction/lyyy/myself/WE/WE
git fetch origin && git pull --ff-only origin h20/reproduction
git log -3 --oneline && git status -sb
```

通过条件：

- HEAD 为 `47d0e4f`（或其后仅含文档指针的等效 tip；完整 SHA 记入报告）；
- 可读：`prompts/H20_CUTOFF4_THREE_SOURCE_PROMPT.md`、`research/PAPER_POSITIONING.md`、`reports/POST_TRAINING_OPENSOURCE_AUDIT.md`、`research/CURRENT_STATE.md`；
- WorldEngine upstream 仍为 `fc79b937050ed9d68e18add2b480ae72578a7ea5`。

任一失败 → **整批暂停**，只回报，不改代码。

## 执行范围（按你的计划）

1. **S1** 只读盘点 cutoff 硬编码；确认 Nexus `cutoff==4`；确认 restore 可 `--cutoff 4`。
2. **S2** Replay + Restored-IDM → cutoff=4；新目录 `data/frozen_paired/smoke1_cutoff4_*`；**禁止覆盖**任何 `cutoff3` 产物。
3. **S3** Nexus 同 scene、cutoff=4、**plan_idx=1333**、seed=0；新 out：`nexus_sidecar/outputs/smoke_cutoff4_three_source/`；不用旧 710。
4. **S4** 三源并表 → `reports/CUTOFF4_THREE_SOURCE_SMOKE.md` + 小 summary json/csv；大数组 Git 外。
5. **S5** 短更 `HANDOFF.md` → **停止**。

## 硬条件（必须写进报告 provenance）

1. ego conditioning：**plan_idx.csv 的 step=5 → plan_idx=1333**（你已核 step4/5 同为 1333）；报告写死 csv 路径、step、plan_idx；**禁止** Nexus 旧 710。
2. 若改 `DEFAULT_CUTOFF` 3→4：同步改注释——写明**研究锁定 cutoff=4**，不再假装等于 WE `num_history-1`。
3. **不要**为迁 cutoff 擅自改 `SCORER_CONFIG.num_history`；S2 核对 fingerprint/horizon=9 与从 cutoff 起的 future pack 一致即可。若必须改 scorer，先停并单列说明等 Mac 批。
4. `pre_cutoff_transition_flags` 若有 True，或门控非 **A**：只交诊断，**禁止**并表冒充严格可比通过。
5. 报告标明：engineering smoke（navtest 工程子集）≠ research go/no-go；**禁止**与 cutoff=3 数字并表。
6. **禁止**改 WorldEngine / Nexus upstream。
7. **停止线**：S5 冒烟报告写完即停。不做 train-side 扩样、后训、Table 1、SMART、BWM 配对、自动 commit/push（除非用户另嘱）。

## 完成后只回传

```text
1. 协作仓完整 HEAD SHA
2. S0 pull 是否成功
3. cutoff=4 Replay/IDM 门控 + transition flags 摘要
4. 三源 fingerprint 是否对齐；plan_idx provenance
5. 主分歧数字（各 pairwise NOC/TTC flip 等）与是否单车主导
6. 报告路径
7. 阻塞项；是否建议另批 train-side≤10（仅建议，本批不执行）
```

开始执行：先 S0，再按 S1→S5 推进。

---
