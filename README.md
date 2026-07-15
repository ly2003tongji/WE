# WorldEngine 研究协作仓库

本仓库用于同步 WorldEngine 论文精读、代码复现、H20 实验和研究选题。它不镜像官方代码，不存放论文 PDF、数据集、权重或密钥。

## 仓库内容

- `PLAN.md`：总体阅读、复现和选题路线
- `HANDOFF.md`：Mac 论文会话与 H20 工程会话的共享状态
- `AGENTS.md`：两个 Agent 都必须遵守的协作约束
- `prompts/H20_AGENT_PROMPT.md`：在 H20 工作站启动工程 Agent 时使用的提示词
- `prompts/H20_DISAGREEMENT_AGENT_PROMPT.md`：启动交通模型分歧实验会话
- `research/RESEARCH_HANDOFF.md`：研究问题、边界和当前结论
- `research/EXPERIMENT_PROTOCOL.md`：阶段门、留出协议、指标和停止条件
- `research/TRAFFIC_MODEL_INVENTORY.md`：各交通模型的可用性与接入成本
- `scripts/bootstrap_h20.sh`：只读检查机器资源并克隆官方 WorldEngine

## 两条并行工作流

### Mac：论文与方向

1. 按 `PLAN.md` 精读论文。
2. 严格区分论文事实、代码事实和推断。
3. 把对工程实验有影响的结论写入 `HANDOFF.md`。

### H20：复现与实验

1. 基础复现已经完成，新会话使用 `prompts/H20_DISAGREEMENT_AGENT_PROMPT.md`。
2. 先读取 `research/` 下三份研究交接，不从聊天记忆猜测实验定义。
3. 目标顺序：奖励schema审计 → BWM-Offline审计 → Replay/IDM分歧 → SMART/Nexus接入 → held-out验证。
4. 288 rare navtest保留为最终测试；开发优先使用train-side长尾场景。
5. 将环境、命令、失败原因和结果写入 `HANDOFF.md` 或 `reports/`。

## 上游资源

- 论文：<https://arxiv.org/abs/2606.19836>
- 官方代码：<https://github.com/OpenDriveLab/WorldEngine>
- 官方数据：<https://huggingface.co/datasets/OpenDriveLab/WorldEngine>
- 国内数据源：<https://www.modelscope.cn/datasets/OpenDriveLab/WorldEngine>

## 安全要求

- 禁止提交 GitHub token、SSH 私钥、`.env`、内部主机凭据。
- 禁止提交数据集、模型权重、仿真资产和实验大文件。
- GitHub 认证使用 SSH key、系统凭据管理器或交互式 `gh auth login`。

