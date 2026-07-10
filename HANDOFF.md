# WorldEngine 双 Agent 交接

最后更新：2026-07-10

## 共同目标

理解并验证 WorldEngine 中“3DGS 数字孪生环境 + action policy 闭环 rollout + post-training”的技术链，最终形成一个可复现、可消融、可投稿的研究问题。

## Mac 论文会话状态

- WorldEngine 论文 PDF 已获取，准备按 `PLAN.md` 进行正式导读。
- 已完成全文和附录的预读，但尚未开始面向用户的逐节讲解。
- 下一步：输出基本信息、论文类型、写作逻辑、分级大纲和核心问题，然后暂停等待用户“继续”。

## 已确认的论文事实

- 论文是系统与方法框架论文，不是单纯 benchmark。
- 主链路：policy-aligned rare-case discovery → 3DGS digital twin → Behaviour World Model 扩增 → closed-loop rollout → behaviour-regularized RL post-training。
- 3DGS 和 Behaviour World Model 是环境侧模块；主要被更新的是驾驶策略。
- nuPlan 实验从 navtrain 发现 5,340 个 long-tail scenes，生成 31,508 frames 用于后训练。
- 论文报告 full WorldEngine 将 rare closed-loop SR 从 73.66% 提升至 88.89%，但 rare set 仅 288 个短场景。
- 论文明确承认：离轨时 3DGS 质量退化、BWM 对行人/非结构化交互不足、奖励手工设计、多轮后训练不稳定。

## 已确认的代码事实

- 上游：<https://github.com/OpenDriveLab/WorldEngine>
- 主仓包含 SimEngine 和 AlgEngine，可见完整的训练/评测脚本链；数据、权重和 3DGS assets 外置。
- Behaviour World Model 尚未集成，roadmap 仍为未完成。
- MTGS 重建训练不在主仓，主仓主要加载并渲染预建 Gaussian assets。
- `projects/AlgEngine/configs/worldengine/e2e_vadv2_50pct_rlft_rare_log.py` 默认：
  - `reward_shaping=True`
  - `use_lora=True`
  - `rl_finetuning=False`
  - `importance_sampling=True`
- 因此默认所谓 RLFT 配置不会进入 `compute_RL_loss`；这一点必须通过运行和完整 loss/data flow 再验证，不能仅凭命名下结论。
- BWM rollout 配置中的 synthetic folder 路径仍为 `/path/to/...` 占位符。

## H20 已知资源

- Ubuntu 22.04.5 LTS，内核 5.10 Alibaba。
- 192 CPU cores，2.0TiB RAM。
- 1× NVIDIA H20，约 96GB，当前空闲。
- Driver 570.133.20，宿主显示 CUDA 12.8。
- `/mnt/cpfs` 总盘接近满载，只余约 1.5TB。
- 另有 Linux 5090 工作站，后续可能申请多 H20。

## H20 Agent 当前任务

1. 克隆本协作仓库并阅读 `AGENTS.md`、`PLAN.md`。
2. 运行 `scripts/bootstrap_h20.sh`，生成资源审计结果。
3. 审计官方数据仓库的文件清单与体积，不下载全量数据。
4. 提出能运行 1–10 个 rare scenes 的最小数据集合。
5. 评估 conda/cu118 与当前驱动兼容方案，不修改宿主驱动。
6. 在 `reports/H20_ENV_AUDIT.md` 记录结论，并更新本文件。

## 尚未解决的问题

- 官方 Hugging Face/ModelScope 数据是否真的包含 quick test 所需全部 checkpoint、rare scenario 和 matching assets？
- 最小 1–10 场景下载能否避免 OpenScene/nuPlan 全量 sensor blobs？
- H20 上旧版 MMCV custom CUDA ops 与当前驱动/编译器是否兼容？
- quick start 文档声称的 5–10 分钟是少量场景还是完整 288 场景？
- 默认 `rl_finetuning=False` 是发布配置失误，还是论文中的“post-training”主要依赖 reward-shaped supervised objectives？

## 同步协议

- 任一 Agent 得到影响另一侧的重要结论时，更新本文件。
- 实验细节写入 `reports/`，本文件只保留结论和下一步。
- 不在本文件保存 token、主机密码、内部下载凭据或个人信息。

