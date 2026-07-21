# WorldEngine 后训练开源审计（2026-07-21）

审计范围：本地 WorldEngine `@fc79b937…`、姊妹仓 [OpenDriveLab/SimScale](https://github.com/OpenDriveLab/SimScale) `@df99d45`（克隆于 `相关论文/World Engine/SimScale`）、论文 PDF、`docs/config_guide.md`。

本报告沉淀 Mac 侧后训相关问答结论；**不改变**当前唯一下一阶段（分歧验证）。研究叙事与基线策略见 `research/PAPER_POSITIONING.md`。

---

## 1. 结论摘要

1. 论文 Eq.9 策略 \(\mathrm{KL}(\pi\|\pi_{\mathrm{ref}})\) 在 WE 与 SimScale **均无实现**；不能在公开信息下裁定“故意未开源”还是“论文理想化 / 与实做不一致”。
2. WE 官方 `configs/worldengine/*rlft*` 全部 `rl_finetuning=False`；文档写 True；`configs/simscale/*` 为 True。默认开源后训 = **LoRA + 奖励头 BCE + 选择性 imitation**，不是 PG。
3. `rl_finetuning=True` 在字段齐全时**可跑通**，追加 PG + rank + entropy，并用 IS ratio clip；**仍不是**显式 KL。
4. 预训练无奖励头；后训新建奖励头并 **全量** `requires_grad=True`；LoRA 管冻结的旧模块（如 imi/embedding）。
5. **不建议**以复现论文 Table 1 全消融为项目目标；方法阶段基线用 OpenWE-SFT（见定位文档）。

---

## 2. KL vs 开源“防遗忘”

| 机制 | 对象 | 与 Eq.9 KL |
|------|------|------------|
| 论文 \(\mathrm{KL}(\pi\|\pi_{\mathrm{ref}})\) | 策略分布 | 正文有公式；开源无对应 loss |
| LoRA | 参数低秩增量，冻 \(W_0\) | 不等价 |
| 真实 log 混合（`normal_ratio`） | 数据域 | 不等价 |
| IS ratio + `clamp(max=10)` | 相对 `orig_` 的概率比加权 | 近 trust-region / PPO 味道，**不是** KL 项 |
| `peft.py` 注释掉的 `kl_divergence` | 贝叶斯 LoRA **权重先验** | 数学对象不同，且未启用 |

论文同时写 data-level（混 log）与 policy-level（KL）；开源明确有前者，缺失后者。

**认识论：** 可钉死“公开代码无策略 KL”；不可钉死作者内部是否用过 KL。

---

## 3. LoRA 与奖励头

- LoRA：\(y = W_0 x + (\alpha/r) BAx\)，\(W_0\) 冻，只训 \(A,B\)。
- `finetuning_detach` 后，若 `orig_IL and reward_shaping`，对非 `imi` 的 head **重新打开全梯度**（新建头从零训）。
- 推断：后训不是“整网只 LoRA”；奖励头全量、旧表示 LoRA。

---

## 4. `rl_finetuning`：文档 vs 配置

| 来源 | 值 |
|------|-----|
| `docs/config_guide.md` RLFT 表 | True |
| `configs/worldengine/e2e_vadv2_50pct_rlft_*.py`（5 个） | **False** |
| `configs/simscale/e2e_vadv2_simscale_*.py`（2 个） | **True** |
| `TrajScoringHeadRL` 默认 | False |

文档仍描述 `rl_loss_weight`，并称 **not fully stabilized**；还建议 real-log 可试 `PG=1.0, entropy=0.2`，与 config 内 `0.01/1.0` 不一致。

复现建议：

- 对齐开源默认可跑路径 → **False**；
- 对齐文档 RL 叙事 → **True**，并当消融变量，不默认 = 论文。

---

## 5. `rl_finetuning=True` 时数据流（不自动报错）

与 False 相同的前向与 ①imi ②奖励 BCE；仅在 `loss()` 末尾调用 `compute_RL_loss`：

```text
batch: img, ego, PDM[B,8192], fail_mask
→ BEV → LoRA 路径 π + forward_origin → orig_*
→ loss: imi CE + 5×BCE + [True?] PG/rank/entropy
→ 反传: LoRA + 新奖励头
```

`forward_origin` 几乎总跑（相关 if 已注释）；开关只决定是否写入 RL loss。

易炸条件：缺 `fail_mask`（且 `use_lora`）、PDM 列为 `None`、在纯 IL 无稠密标签 config 上硬开。官方 RLFT/simscale 管道字段齐全时一般可训。

### PG / rank / entropy / IS clip（离散 8192 动作）

- **IS clip**：`detach(π/π_orig)`，fail 样本 ratio=1，再 `clamp(max=10)`。
- **PG**：`-mean(clipped_IS * score * log π)`（reward-weighted log-likelihood）。
- **rank**：`score==1` 的 max π 至少比非完美 max π 高 margin 0.2；权重常为 0。
- **entropy**：最大化 \(H(\pi)\) 防塌缩。

---

## 6. SimScale 独立仓

路径：`/Users/luoye060/Desktop/相关论文/World Engine/SimScale`。

- **无** `rl_finetuning` / `TrajScoringHeadRL` / 策略 KL；主线是 sim–real 共训练（navsim）。
- 论文 PDF 中 KL 出现次数为 0。
- 此前“姊妹项目开 True”指的是 **WE 仓内** `configs/simscale/`，共享同一 `traj_scoring_head_RL.py`，同样无 Eq.9 KL。

---

## 7. Table 1 与复现可行性

Table 1 行（论文）：base；SFT rare logs；post-train common/rare logs；rare synthetic replays；rollouts w/o BWM；full World Engine。

| 行/环节 | 开源可行性 |
|---------|------------|
| base 评测 | 可行（已有 ckpt） |
| rare/common log 开源 config | 可训，但是 False 配方，≠ 论文保证 |
| synthetic / rollout / BWM / full WE | 生成端/BWM/开关未齐，**不可承诺** |
| 闭环交通是否 IDM | 论文未钉死，数字可能对不齐 |

**决策：** 不把“预训练→后训→对齐 Table 1”列为项目目标。时间上单卡 288 闭环约数小时量级/次；多配方全表可达数周且仍可能对不齐——机会成本高于当前分歧验证。

硬件粗估见 `research/PAPER_POSITIONING.md` 与历史 `CODE_PAPER_GAPS.md`（10-scene≈9min → 288≈4–5h 外推）。

---

## 8. 对研究的直接含义

- **现在：** 不启动后训 / Table 1；继续统一锚点多源分歧。
- **方法阶段：** OpenWE-SFT 作可复现弱基线；可选 OpenWE-PG；主对比是朴素混合与稳健启发式 + held-out 交通协议。
- **叙事：** WE = 平台与问题来源，不是必须超越的成绩单。
