# Agent 协作规则

## 语言与证据

- 全程使用简体中文，保留必要英文术语并解释中文含义。
- 严格区分并标注：
  - `论文事实`：论文正文或附录明确写出；
  - `代码事实`：由官方仓库、配置或运行结果确认；
  - `推断`：合理假设、研究判断或领域惯例。
- 论文或代码未交代的实现细节必须明确写“未交代”，不得猜测成事实。

## 分工

- Mac Agent：论文精读、前置知识讲解、实验假设和研究方向。
- H20 Agent：环境、依赖、数据清单、复现、训练和评测。
- 两边通过 `HANDOFF.md`、`research/` 和 `reports/` 交换状态，不假设聊天记忆自动同步。
- 交通模型分歧实验开始前必须阅读：
  - `research/RESEARCH_HANDOFF.md`
  - `research/EXPERIMENT_PROTOCOL.md`
  - `research/TRAFFIC_MODEL_INVENTORY.md`

## H20 执行约束

- 已知资源：Ubuntu 22.04、1× NVIDIA H20 约 96GB、Driver 570.133.20、宿主 CUDA 12.8、192 CPU、2TiB RAM。
- CPFS `/mnt/cpfs` 只剩约 1.5TB；下载任何数据前必须列出文件与预计大小。
- 不修改或降级宿主 NVIDIA 驱动，不破坏系统 Python，不删除共享数据。
- 环境必须隔离，优先使用 conda/mamba 或容器解决项目要求的 CUDA 11.8。
- 先单卡和小场景验证；仅当证据表明需要时才使用多 H20。
- 不从零预训练 base model，不先实现完整 BWM 或 MTGS 重建。

## Git 与安全

- 不在文件、命令、日志或远程 URL 中保存 token。
- 大数据、权重、仿真资产和输出目录不得进入 Git。
- 修改前读取最新 `HANDOFF.md`；阶段完成后更新它。
- 每次实验记录上游 commit、配置、命令、数据 split、随机种子、GPU 和指标。
- 未经用户明确要求，不修改上游官方仓库；研究改动使用独立分支。

## 当前优先级

1. 保持已跑通的 1/10-scene NR/R 最小闭环可重复。
2. 审计候选级 PDM 子奖励、轨迹词表和 synthetic data schema。
3. 验证 Replay / IDM / BWM-Offline 的配对条件与初步奖励分歧。
4. 核验并接入 SMART；优先评估 Nexus 单场景 ego-conditioned adapter。
5. 只有分歧能够增量预测 held-out 失败后，才设计风险敏感后训练。
6. 288 rare navtest 保留为最终测试，不用于反复调整分歧公式。

