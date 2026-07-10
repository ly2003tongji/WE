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

## H20 Agent 状态

- 阶段 1–5 已完成，详见：
  - `reports/H20_ENV_AUDIT.md`
  - `reports/DATA_MANIFEST.md`
  - `reports/ENV_SETUP.md`
  - `reports/SMOKE_TEST.md`
  - `reports/CODE_PAPER_GAPS.md`
  - `scripts/audit_data_manifest.py`
- 官方代码固定在 `fc79b937050ed9d68e18add2b480ae72578a7ea5`。
- Hugging Face 数据版本 `8728616abaf090d195b3bdc7af6aacde40271145`：174 文件、4.854 TB；ModelScope 完整包含这些数据，另有 6 个文档/脚本文件。
- 默认 quick test 实际运行完整 288 个 `navtest_failures`，文档中的 “1/10” 不是脚本默认值；现有 `debug_scene_name` 没有被 runner 使用。
- 1–10 scene 最小可操作下载约 36.0 GB：完整 checkpoint、trajectory vocabulary、navtest PDMS cache、单体 rare scenario pickle、nuPlan maps，以及一个约 31 GB 的 rare asset shard。下载后还需生成过滤后的 scenario pickle。
- 任意指定 scene 因缺少 scene→shard 远端索引，保守需三个 rare asset shards；288 baseline 下载约 93.6 GB，工作盘保守需 380–500 GB。
- 闭环 smoke test 可避免 OpenScene 全量 metadata/sensor blobs；SimEngine metric 仍需要外部 nuPlan maps。
- 两个持久环境位于 `/workspace/worldengine/envs/{simengine,algengine}`，源码位于 `/workspace/worldengine/src/`；总占用约 25 GB，低于 `/workspace` 40 GB 限额。临时 cache 放 `/tmp`。
- Driver 570 保持不变；两个环境均已验证 PyTorch 2.0.1+cu118 在 H20 `sm_90` 上运行。
- gsplat v1.4.0 已完成真实 rasterization；MMCV full 1.6.2 已通过官方 CPU/CUDA 检查和 H20 CUDA op 测试。
- 已选择性下载并逐项校验 33.0 GB：checkpoint、vocabulary、navtest PDMS cache、288-scene pickle、rare asset part003 和 nuPlan maps；没有下载 OpenScene blobs 或全量数据。
- part003 已确认包含 90 个完整 asset；已构造 1-scene（337.9 MB assets）和 10-scene（3.532 GB assets）filtered subsets。
- 1-scene NR/R、10-scene NR/R 均完成真实闭环，runner technical success 均为 100%。10-scene NR/R 均无 collision、drivable compliance/SR 均为 70%；R 相对 NR 的 EP 为 0.51100 vs 0.48689，score 为 0.62125 vs 0.61120。
- 1-scene profile 峰值显存 13,638 MiB、峰值 GPU util 97%；10-scene NR/R 墙钟分别约 8.9/9.5 分钟。
- 代码核查确认：主仓只有 BWM synthetic data consumer、没有 BWM generator；MTGS 重建指向外部仓库。
- 默认 rare-log `RLFT` 的 `rl_finetuning=False`，不调用 `compute_RL_loss`。默认实际为 LoRA + reward-head BCE；normal imitation 使用 importance ratio，rare/hard imitation 被 mask。
- 下一步是下载剩余两个 rare asset shards 并运行正式 288 NR/R baseline；尚未启动该约 60.6 GB 增量下载。

## 尚未解决的问题

- part003 的 90-scene 映射已可从 tar 获取；part001/part002 尚未下载和索引。
- AlgEngine 官方 pins 与依赖 metadata 有两个已知冲突：`networkx==2.5` 对 mmdet3d 的 `<2.3`，`Shapely==2.0.4` 对 nuscenes-devkit 的 `<2.0`；当前 import/CUDA op 通过，需在真实数据路径继续观察。
- 288 scenes 的实际解压体积、运行耗时、输出体积和 quick-start “5–10 分钟”说法仍需运行验证。
- 论文 full WorldEngine 所用 BWM/checkpoint/训练开关与当前默认开源配置的精确对应关系仍未交代。

## 同步协议

- 任一 Agent 得到影响另一侧的重要结论时，更新本文件。
- 实验细节写入 `reports/`，本文件只保留结论和下一步。
- 不在本文件保存 token、主机密码、内部下载凭据或个人信息。
