# WorldEngine 精读、复现与选题路线

## 总目标

先用论文建立系统地图，再按需补齐闭环评测、3D Gaussian Splatting（3DGS）与策略后训练知识，并以单张 H20 上的最小闭环实验筛选方向。重点不是从头复现整套系统，而是验证“仿真经验如何有效改进 action policy”。

## 当前判断

- 先读 WorldEngine，不先系统补一批前置论文。先读 Fig.1、§1、§2；遇到具体模块再定向补课。
- 导师所指链路是：`真实日志失败发现 → 3DGS 数字孪生 → 他车行为变化 → action policy 闭环 rollout → 策略后训练`。
- 3DGS 是环境渲染器，不与 action policy 联合反向传播。
- 单张 96GB H20 足够做 rare-case 闭环评测、LoRA 后训练和中等规模消融。
- CPFS 仅余约 1.5TB，必须选择性下载，不能直接拉全量数据。

## 工作流 A：Mac 论文精读

### A1. 开始前导读

先输出并暂停：

1. 标题、作者、年份/状态、研究方向与发展脉络；
2. 论文类型及相应阅读重点；
3. 整体写作逻辑；
4. P0/P1/P2 阅读大纲；
5. 读完后必须能回答的 3–5 个问题。

### A2. 精读顺序

- P0：Fig.1、§2、§3.4、Appendix B.2/B.3、Table 1。
- P1：§3.2 的 3DGS、§3.3/B.1 的 Behaviour World Model、§5 局限。
- P2：base model 结构与产业路测细节。

### A3. 定向前置知识

- 闭环：nuPlan/NAVSIM、open-loop/closed-loop、PDMS/SR/EP、replay/IDM。
- 3DGS：Gaussian 参数、投影、alpha compositing、动态 scene graph、MTGS。
- 后训练：行为克隆、轨迹词表、POMDP、policy gradient、KL 约束、offline distribution shift。

## 工作流 B：H20 复现

### B1. 只读审计

1. 记录硬件、驱动、CUDA、Python/conda、磁盘和网络条件。
2. 克隆官方仓库并固定 commit。
3. 核查文档、脚本、checkpoint 和数据 manifest。
4. 在下载前给出最小数据子集与准确/保守体积预算。

### B2. 环境与最小 baseline

1. 使用隔离环境兼容项目所需 Python 3.9、PyTorch 2.0.1/cu118。
2. 先运行 import/编译检查。
3. 只下载少量场景所需 checkpoint、map、metadata 和 3DGS assets。
4. 运行 1–10 个场景 smoke test。
5. 稳定后运行官方 288 rare-case baseline。

### B3. 最小消融

1. Non-reactive replay 与 IDM reactive agents。
2. rare-log LoRA：reward-shaped imitation learning baseline。
3. 有可靠 rollout 数据后，再尝试 rollout post-training。
4. 记录 SR、EP、PDMS*、失败类型、吞吐、显存和随机性。

## 论文—代码重点核查

1. 官方 roadmap 中 Behaviour World Model integration 尚未完成。
2. `e2e_vadv2_50pct_rlft_rare_log.py` 默认 `rl_finetuning=False`。
3. `e2e_vadv2_50pct_rlft_rare_rollout_bwm.py` 的 synthetic folders 是占位路径。
4. 主仓只有 MTGS 渲染消费端，重建训练依赖外部 MTGS 仓库。
5. 区分论文完整系统、当前开源仓库和产业内部验证三者的能力边界。

## 候选方向

1. **首选：闭环经验质量与真正的策略后训练。** 比较 reward-shaped IL、policy-gradient 分支、KL/参考策略约束和 hard-case weighting。
2. **次选：交通行为模型敏感性与泛化。** 比较 replay、IDM 和不同交互强度，研究 simulator-policy overfitting。
3. **高风险：3DGS 离轨渲染可靠性。** 估计视角/轨迹偏移下的不确定性，并对低可信帧降权。
4. **暂缓：完整 BWM 或 MTGS 重建。** 当前开源边界和工程成本不适合作为第一个目标。

## 决策门槛

首选 idea 必须同时满足：

- 现象在小规模实验中稳定；
- 闭环实验相对开环提供不可替代的信息；
- 单 H20 数天内能完成一轮；
- 代码改动和对照实验可审计；
- 能清晰区分论文已有贡献与新增贡献。

