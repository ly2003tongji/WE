# H20 交通模型分歧实验 Agent 提示词

将下面整段发送给 H20 工作站上的 Cursor Agent。

---
你是 WorldEngine 研究项目的 H20 实验 Agent。全程使用简体中文。当前任务不是重复环境搭建，而是在已经跑通的 WorldEngine 最小闭环上，推进“交通模型分歧能否预测未见模型失效，并用于稳健后训练”的研究。

## 工作模式：必须先进入 Plan Mode

收到提示词后先切换到 **Plan Mode**。在用户批准计划前，只允许：

- 阅读交接文档、代码、配置和现有报告；
- 检查当前 Git、环境、数据和输出状态；
- 进行不会写文件、下载数据、启动长任务或改变环境的只读核查；
- 识别依赖、风险、缺失信息和阶段门。

用户批准计划前禁止：

- 修改代码、配置、报告或 Git 状态；
- 下载数据、模型或新仓库；
- 安装/升级依赖；
- 启动 smoke test、rollout、训练或批量分析；
- 创建 commit 或 push。

Plan Mode 最终必须给出一份可审批计划，包含：

1. 当前可复用资产与缺失项；
2. 阶段0～5各自要回答的问题、输入、输出和通过条件；
3. 预计修改的仓库文件；
4. 每一步拟执行的关键命令；
5. 新增下载的文件、大小和落盘位置；
6. 预计GPU/CPU/磁盘需求和耗时；
7. 风险、回滚方式和需要用户确认的决策点；
8. 第一批建议执行到哪个阶段以及为什么。

提交计划后暂停，等待用户批准。获得批准后再切换到 Agent Mode，并且只执行用户批准的阶段。

## 启动步骤
分支：h20/reproduction
提交：326bb28 完善交通模型分歧实验交接

```bash
git clone --branch h20/reproduction https://github.com/ly2003tongji/WE.git
cd WE
git pull --ff-only
```

必须先完整阅读：

1. `AGENTS.md`
2. `HANDOFF.md`
3. `research/RESEARCH_HANDOFF.md`
4. `research/EXPERIMENT_PROTOCOL.md`
5. `research/TRAFFIC_MODEL_INVENTORY.md`
6. `reports/SMOKE_TEST.md`
7. `reports/CODE_PAPER_GAPS.md`

不要依赖聊天历史补全事实。

## 已完成基础

- 官方 WorldEngine 固定 commit、双环境和约30.2 GB最小数据已经验证；
- 1-scene和10-scene NR/R真实闭环均已跑通；
- H20显存和吞吐不是当前阻塞；
-主仓只有BWM synthetic consumer，没有完整BWM generator；
- 现有工作区和输出路径以`HANDOFF.md`为准。

## 本轮核心目标

按顺序回答：

1. 当前输出能否对齐同一scene、同一候选在不同交通行为来源下的子奖励？
2. BWM-Offline augmented scenarios包含什么字段，能否映射回original scene？
3. Replay和IDM是否产生非微小的NOC/TTC/排序差异？
4. SMART与Nexus的公开代码实际能否接入？
5. 是否具备进入严格held-out实验的前提？

在前四项没有证据前，不设计新的后训练网络。

## 阶段 0：同步与只读核查

1. 记录协作仓库commit和上游WorldEngine commit；
2. 确认本地数据、环境、assets和实验输出仍完整；
3. 运行现有1-scene smoke命令，确认环境未漂移；
4. 不升级上游依赖，不修改宿主驱动。

## 阶段 1：奖励数据schema审计

对现有NR/R输出：

1. 定位`plan_traj`、`meta_datas`和`pdms_pkl`；
2. 编写小型只读脚本，打印：
   - scenario/frame/token；
   - candidate数量；
   - 每个PDM字段的key、shape、dtype、范围；
   - 候选索引与轨迹词表的对应关系；
3. 追踪PDM标签在SimEngine生成、导出和AlgEngine读取的位置；
4. 确认奖励数组是否对应8192候选；
5. 确认他车future是否在8192候选间共享；
6. 输出`reports/DISAGREEMENT_DATA_AUDIT.md`，每条结论引用代码路径和行号。

只提交脚本和报告，不提交pkl、npy、图像或日志。

## 阶段 2：BWM-Offline数据审计

在下载前先检查本地是否已有：

```text
data/sim_engine/scenarios/augmented/
```

若缺失：

1. 通过只读API列出三个`all_scenarios.pkl`大小；
2. 给出最小必要下载、落盘位置和空间预算；
3. 单次下载超过100 GB或新增总量超过300 GB时暂停请求用户确认。

读取一个augmented split时，避免无目的导出大对象。记录：

- 顶层scene数量和key格式；
- original/source token；
- variant映射；
- agent trajectory、valid mask、频率和时域；
- ego plan或条件字段；
- 地图、traffic light和agent类型；
- 是否可与original同初始状态严格配对；
- 是否存在预计算PDM标签。

若不能严格配对，明确将BWM-Offline降级为外部生成域，不得把它纳入“同一场景四模型共识”。

## 阶段 3：Replay / IDM最小分歧数据集

1. 使用相同10个scene；
2. 对齐NR和R的同一frame与候选；
3. 固定PDM Scorer和奖励定义；
4. 导出轻量统计，不复制原始大数组；
5. 至少统计：
   - NOC/TTC翻转；
   - DAC/Comfort稳定性；
   - overall与各子奖励差；
   - Top-1变化；
   - Top-K重合；
   - 候选排序相关性；
   - 一致安全/一致危险/冲突分类。

若现有输出不足，先做最小代码插桩并以独立patch记录；不要直接大改训练管线。

输出：

- `reports/DISAGREEMENT_SMOKE.md`
- 可复现命令；
- 必要的小型分析脚本；
- go/no-go判断，但不要用10 scenes否定总idea。

## 阶段 4：SMART可用性核验

核验：

- `rainmaker22/SMART`官方模型仓库；
- `shgd95/InteractiveClosedLoop`是否真正公开代码；
- checkpoint、nuPlan版本、运行命令和许可；
- 是否能由外部ego控制；
- 输出能否转换为WorldEngine agent states。

若drop-in代码不可用，写出最小adapter接口，不先承诺完整接入工期。

## 阶段 5：Nexus单场景旁路验证

固定Nexus repo commit和`nuplan.ckpt`。

最小目标：

```text
同一scene + ego candidate A → sampled other-agent future A'
同一scene + ego candidate B → sampled other-agent future B'
```

验证：

- 输入SceneTensor构造；
- ego token inpainting/mask；
- 输出轨迹解码；
- 坐标和频率；
- 相同seed下不同ego candidate是否引发合理响应；
- 同一candidate多seed模型内波动；
- 能否回写为WorldEngine agent boxes。

先关闭3DGS，只做结构化轨迹与碰撞/TTC检查。

输出`reports/NEXUS_ADAPTER_FEASIBILITY.md`。若公开闭环类缺失，用旁路wrapper验证，不为修复整个Nexus仓库而偏离研究目标。

## 统计与实验纪律

- Log Replay、IDM、BWM-Offline不是三个对等在线模型；
- 不同IDM参数不是独立模型族；
- 8192候选不是8192个独立统计样本；
- 以original scene划分和bootstrap；
- 288 rare navtest保留为最终测试，不用于调公式；
-先使用train-side长尾场景开发；
- held-out模型不得用于阈值、公式、调参或早停；
-同时报告基础风险和模型分歧；
- 分歧必须与TTC、车辆数、平均/最坏奖励等简单基线比较。

## Git交付

1. 研究分支继续使用`h20/reproduction`，除非用户另行指定；
2. 只提交Markdown、小脚本、配置diff和必要patch；
3. 禁止提交数据、权重、环境、图片、实验大日志和密钥；
4. 每阶段更新`HANDOFF.md`；
5. commit前运行status、diff和secret scan；
6. push后返回commit hash；
7. 任何外部仓库修改保留为独立patch或fork，不污染其默认分支。

## 首次回复要求

在 Plan Mode 中先报告：

1. 当前分支和commit；
2. 已阅读的交接文件；
3. 本地现有NR/R与BWM数据状态；
4. 当前仍需只读核验的事实；
5. 是否预计涉及新增下载；
6. 计划准备采用的阶段边界。

随后完成只读调查并提交完整实施计划，不得自动进入阶段0或阶段1的实际执行。用户明确批准后，再切换到 Agent Mode 开始工作。

现在先进入 Plan Mode。