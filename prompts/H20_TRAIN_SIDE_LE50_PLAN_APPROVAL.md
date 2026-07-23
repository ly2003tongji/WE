# H20：train-side~50 计划有条件批准（直接粘贴）

将下面 `---` 之间全文发给 H20 Agent（Agent Mode 执行；勿再空转 Plan）。

---

你是 WorldEngine 研究项目的 H20 实验 Agent。全程使用简体中文。

## 批准状态

Mac 侧已**有条件批准**你提交的计划：`train-side ~50 扩样计划（E0→E5）`。  
现在切换到 **Agent Mode**，按该计划执行；下列硬条件**高于**计划模糊表述。

先确认：

```bash
cd /mnt/cpfs/prediction/lyyy/myself/WE/WE
git pull --ff-only origin h20/reproduction
# 期望含 2b3040b 或其后；upstream 仍为 fc79b937…（勿改）
```

## 批准范围（可做）

1. **E0–E1**：固化 `reports/train_side_le50_scene_list.{md,json}`（25C+25E 含全部 le10；reserve 每桶 8；∩navtest=∅、∩smoke=∅）。
2. **目录**：`data/frozen_paired/train_side_le50/<token>/`；7 个门控 A 完整产物 **symlink** 到 `../train_side_le10/<token>/`，`reuse_from_le10=true`，**禁止重跑**。
3. **E2–E2b**：新增 ~40 + reserve 替补；协议同 le10（cutoff=4、`static_feasible_median_path_length`、三源同一 plan_idx）；串行；总 restore 尝试（含 le10 已跑 10）**≤65**；A 目标 48–50，不足则报告写明实际 A。
4. **E3**：门控 A 中分层多种子 ≥10（尽量 5C+5E）；计入已有 le10 的 3 个多种子 scene，再从新增 A 补 ≥7（优先 ep）；seed=1、2 vs seed=0。
5. **E4–E5**：`reports/TRAIN_SIDE_LE50_DISAGREEMENT.md` + `train_side_le50_summary.{json,csv}`；短更 HANDOFF/CURRENT_STATE；**停止**。不自动 commit/push；不启 SMART/后训/held-out。

## 硬条件（必须遵守）

1. **3 个 degraded le10** 保留在名单、**不重跑**（原因已记录）。
2. **可复用 7 token** 仅当产物完整且 `gate_a=true`（按你计划的文件清单）；缺一件 → 不得标 reuse，须当新跑或标 degraded。
3. conditioning：**禁止**静默 1333；**禁止**默认 710（除非算出恰为 710 并写明）；报告**单列**：不可与冒烟 NR-1333 数值对比。
4. 汇总必须分节：全样本三 pairwise 分位数；**R↔I≈0 比例**；le10 子集 vs 新增；多种子 vs 模型间；degraded 率与原因；难度描述。
5. **声称边界**：只写效应量/分位数；**禁止** go/no-go、held-out 可预测失败。
6. **不做**：offroad/BWM/3DGS 下载、IDM 参数扰动、多进程抢 GPU、改 upstream、>100GB 新下载。

## 执行顺序（锁定）

```text
E0 协议指针与复用清单
E1 固化 le50 scene list（含 reserve、∩ 校验）
E2 增量三源 seed0（跳过 reuse）
E2b reserve 替补至 A≈48 或尝试上限 65
E3 Nexus 多种子抽样 ≥10
E4 汇总报告
E5 HANDOFF/CURRENT_STATE 短更 → 停止
```

## 完成后只回传

```text
1. HEAD；复用数 / 新跑数 / 门控 A / degraded / 总尝试次数
2. scene list 路径；∩navtest / ∩smoke
3. 全样本 R↔I / R↔N / I↔N 分位数；R↔I NOC=0 比例
4. 多种子一句
5. 报告路径
6. 建议：SMART / 再扩 / 收缩主张（仅建议）
```

开始：按 E0→E5 执行。

---
