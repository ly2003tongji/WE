# WorldEngine 代码—论文能力边界核查

审计时间：2026-07-10（UTC+8）  
官方代码：`OpenDriveLab/WorldEngine@fc79b937050ed9d68e18add2b480ae72578a7ea5`

## 结论摘要

1. `代码事实`：当前主仓没有开放 Behaviour World Model 生成/推理集成；开放的是预生成 synthetic rollout 的消费和训练数据加载路径。
2. `代码事实`：当前主仓的 MTGS 部分是预建 Gaussian asset 的加载与 gsplat 渲染端；重建训练指向外部 `OpenDriveLab/MTGS`。
3. `代码事实`：默认名为 `RLFT Rare Log` 的配置设置 `rl_finetuning=False`，不会调用 `compute_RL_loss`；默认训练主体是 LoRA + reward-shaped BCE + imitation loss，不是 policy-gradient update。
4. `代码事实`：默认配置下 `importance_sampling=True` 只对正常样本的 imitation loss 产生非平凡权重；reward heads 在 `orig_IL=True` 时得到全 1 权重，rare/hard cases 又因 `hard_case_no_imi=True` 不使用 imitation loss。
5. `代码/运行事实`：quick-start 的 “1/10、5–10 分钟” 示例与默认脚本的 288 scenes 不一致。本机 10 scenes 实测 8.9–9.5 分钟，但默认 288 baseline 尚未运行。
6. `代码/运行事实`：NR 是日志轨迹 replay，R 只把其他动态 agent 改为 IDM policy/navigation；10-scene 实测证明分支生效，但没有改变本子集 70% SR，只小幅改变 EP/score。

## 1. Behaviour World Model 开放范围

### BWM 证据

- `代码事实`：README roadmap 明确把 `Behavior World Model integration` 标为未完成（`README.md:132-142`）。
- `代码事实`：安装文档把 BWM 标为 `coming soon`（`docs/installation.md:36-42`）。
- `代码事实`：BWM rollout 配置的四个输入目录仍为：
  `/path/to/synthetic/rollouts/...`（`configs/worldengine/e2e_vadv2_50pct_rlft_rare_rollout_bwm.py:73-78`）。
- `代码事实`：同一配置把这些外部路径交给 synthetic dataset 的 `folder_name`（`e2e_vadv2_50pct_rlft_rare_rollout_bwm.py:406-414`）。
- `代码事实`：synthetic dataset 只读取已有 `plan_idx.csv`、metadata pickle、sensor blobs 和 PDM pickle，并做过滤（`mmdet3d_plugin/datasets/navsim_openscene_synthetic.py:27-47,49-158`）；没有调用 BWM 生成器。

### BWM 判定

`代码事实`：当前开源仓支持“消费 BWM/其他来源预先生成的 rollout”，不支持“在 WorldEngine 主仓中运行 BWM 产生行为变体”。因此论文中的完整链路不能仅靠主仓和已发布数据重新生成；预生成 augmented scenarios 不等价于开放 BWM 模型、checkpoint 和生成代码。

## 2. MTGS / 3DGS 重建开放范围

### MTGS 证据

- `代码事实`：README 的 Scene Reconstruction 段落把多遍历重建、asset generation 指向外部 `OpenDriveLab/MTGS` 仓库（`README.md:216-224`）。
- `代码事实`：主仓 `MTGSAssetManager` 根据 scene id 定位 `background/<scene>.ckpt` 并用 `torch.load(..., map_location=device)` 加载（`projects/SimEngine/worldengine/render/mtgs/mtgs.py:436-476`）。
- `代码事实`：road height map 加载代码目前被注释（`mtgs.py:478-482`）。
- `代码事实`：主仓搜索不到 Gaussian reconstruction 训练入口；`projects/SimEngine/worldengine/render/mtgs/` 提供 renderer、Gaussian model 表示和 asset manager，但没有训练 CLI/runner。
- `运行事实`：smoke test 只使用预建 `background/*.ckpt` 即完成 3DGS render；`configs.tar.gz` 不必需。

### MTGS 判定

`代码事实`：主仓是 MTGS runtime consumer，不是重建训练发布。  
`推断`：若研究离轨视角可靠性，可以直接在 renderer/asset 层做评测；若要复现从 nuPlan raw logs 到 asset 的训练，应单独审计外部 MTGS 仓库，不能把主仓能力表述为完整重建开放。

## 3. 默认 RLFT 配置是否运行 policy gradient

### 配置证据

`e2e_vadv2_50pct_rlft_rare_log.py`：

- planning head 为 `TrajScoringHeadRL`（`:247-249`）；
- `reward_shaping=True`、`use_lora=True`、`trans_use_lora=True`（`:249-251`）；
- **`rl_finetuning=False`**（`:252`）；
- `importance_sampling=True`、`orig_IL=True`（`:253-254`）；
- `rl_loss_weight={bce:0, rank:0, PG:0.01, entropy:1}`（`:255-260`）；
- `hard_case_no_imi=True`（`:261`）。

`代码事实`：不只是 rare-log；`configs/worldengine/` 下 5 个 `e2e_vadv2_50pct_rlft_*.py` 全部设置 `rl_finetuning=False`（common log `:252`、rare log `:252`、rare rollout `:258`、BWM rollout `:259`、synthetic replay `:258`）。这与 `docs/config_guide.md:95-118,173-181,233-239` 把 RLFT 描述为 `rl_finetuning=True`、hard cases “only RL losses” 直接矛盾。

训练 pipeline 收集六个 PDM fields 和 `fail_mask`（`:321-350`）；训练 dataset 是 `NavSimOpenSceneE2EFineTune`，并加载三个 rare YAML（`:48-71,394-419`）。

### 实际 loss data flow

1. `代码事实`：dataset 从 PDM cache 读取每个 token 的 8192-trajectory metrics（`navsim_openscene_nuplan.py:172-175,496-515,860-864`）。
2. `代码事实`：fine-tune dataset 把 rare token 标成 `fail_mask=1`，并按同 log 随机混入 normal samples（`navsim_openscene_finetuning.py:67-113,116-152,154-160`）。
3. `代码事实`：`NAVFormer.forward_train` 把 PDM fields 组成 `pdm_dict`，传入 `planning_head.loss`（`navformer.py:662-685,721-748`）。
4. `代码事实`：planning loss 先用 vocabulary 与 expert trajectory 距离得到 imitation label（`traj_scoring_head_RL.py:260-275`）。
5. `代码事实`：`hard_case_no_imi=True` 时，rare/hard 和 synthetic samples 的 imitation mask 被置 0（`:277-288`）。
6. `代码事实`：imitation 使用 label-smoothed cross entropy，并经过 `IS_weight`（`:292-302`）。
7. `代码事实`：`reward_shaping=True` 无条件加入 noc、drivable area、TTC、comfort、progress 五个 BCE losses（`:304-324`）。
8. `代码事实`：只有 `self.rl_finetuning` 为真时才调用 `compute_RL_loss`（`:326-329`）。

### `compute_RL_loss` 的开放实现

- `代码事实`：函数确实存在，计算 current/original policy、detached ratio、clipped ratio、ranking、reward-weighted log probability 和 entropy（`traj_scoring_head_RL.py:176-233`）。
- `代码事实`：默认 rare-log config 的 `rl_finetuning=False` 使该整段不可达。
- `代码事实`：配置中的 `rl_loss_weight['bce']` 在该文件没有被 loss 使用；BCE weights 是代码中直接写死的 `3,3,2,1,1`（`:304-324`）。
- `代码事实`：配置即使改为 `rl_finetuning=True`，默认 `rank=0` 会关闭 ranking contribution，而 `PG=0.01`、`entropy=1.0` 会启用对应项（config `:255-260`，head `:227-230`）。

### Importance sampling 的真实作用

`IS_weight`（`traj_scoring_head_RL.py:236-258`）：

- imitation head 使用 current/original imitation logits 的 probability ratio（`:240-243`）；
- 非 imitation head 在 `orig_IL=True` 时直接使用全 1 weight（`:243-247`）；
- ratio detach 后裁剪到 `[0.01, 10]`，hard/fail mask 强制设为 1（`:249-256`）。

因此默认 rare-log config 的实际优化结构为：

```text
normal samples:
  importance-weighted imitation CE
  + unweighted reward-head BCE

rare/hard samples:
  imitation masked out
  + unweighted reward-head BCE

all samples:
  LoRA/new reward-head parameters updated
  no compute_RL_loss / no PG / no entropy term
```

### RLFT 判定

`代码事实`：默认发布的 `RLFT Rare Log` 是 reward-shaped supervised fine-tuning with LoRA，而不是 policy-gradient RL。  
`推断`：论文“RL-based post-training”可能对应内部配置、未发布运行参数，或需要用户显式把 `rl_finetuning` 改为 true；仅凭当前默认配置无法复现论文中 policy-gradient 分支的训练结果。

## 4. Reward-shaped inference 与训练

- `代码事实`：reward shaping 为 noc、DA、TTC、comfort、progress 和 imitation 分别建立 heads（`traj_scoring_head_RL.py:74-119`）。
- `代码事实`：LoRA 开启时原模块冻结，新 reward heads 在 `orig_IL + reward_shaping` 下显式保持可训练（`:158-174`）。
- `代码事实`：inference score 用 imitation log-softmax 与五个 reward head 的加权 log score 组合，再 argmax 选 vocabulary trajectory（`:427-455`）。
- `代码事实`：同一次 forward 还运行 frozen/original 分支并产生 `orig_*` outputs（`:457-468,471-568`），供 importance ratio/对照使用。

`推断`：这更接近“学习型 trajectory scorer + reward-shaped supervised objectives”，而不是从 simulator transition 上在线反向传播。3DGS renderer 与 action policy 之间也没有联合梯度路径。

## 5. Quick test 场景数与耗时

### 文档/脚本

- `代码事实`：`docs/quick_start.md` 标题称 “Quick Test (5 Minutes)”（`:15-17`），示例输出显示 `scenario 1/10`（`:41-50`），并称约 5–10 分钟（`:52`）。
- `代码事实`：同一文档稍后又注明原始 `navtest_failures` 有 288 rare cases（`:128-139`）。
- `代码事实`：README 明确说默认 quick test 运行 288 rare-case scenarios（`README.md:165-184`）。
- `代码事实`：`scripts/closed_loop_test.sh:3-8` 直接传 `navtest_failures`，没有 scene count。
- `代码事实`：runner 读取整个 `all_scenarios.pkl`（`worldengine/runner/run_simulation.py:22-26`），`env_builder.py:62-69` 的 scene filter 仍是 TODO。

### 运行核查

- `运行事实`：过滤后的 10 scenes，单 H20 NR 为 533.4 s（8.9 min），R 为 571.9 s（9.5 min），与文档对 “10 scenes” 的时间描述一致。
- `运行事实`：未过滤默认 pickle 是 288 keys；本项目必须自行生成 filtered pickle 才能限制到 1/10 scenes。
- `推断`：按 10-scene 实测近似线性估计，单卡 288 scenes 是约 4.3–4.6 小时量级，不是 5–10 分钟；正式值仍需运行，不把线性外推标为事实。

## 6. NR replay 与 IDM reactive

### 代码路径

- `代码事实`：默认 other-agent config 是 `trajectory_policy` + `trajectory_navigation`，controller 为 log replay（`default_runner.yaml:68-71`）。
- `代码事实`：`run_testing.sh` 仅在 `REACT_TYPE=R` 时覆写为 `idm_policy` 和 `idm_navigation`（`:50-74`）。
- `代码事实`：policy/navigation builders 分别实例化 `IDMPolicy` 和 `IDMNavigation`（`build_policy.py:12-29`、`build_navigation.py:10-23`）。
- `代码事实`：IDM policy 只控制相对前车的纵向 progress/velocity（`idm_policy.py:9-13`）；IDM navigation 先跟原轨迹、再接地图 lane（`idm_navigation.py:17-21`）。
- `代码事实`：metric manager 依据 `agent_policy == idm_policy` 把结果写成 `_R.csv`，否则 `_NR.csv`（`metric_manager.py:387-397`）。

### 运行路径

相同 10 scenes：

| 指标 | NR replay | IDM reactive |
| --- | ---: | ---: |
| no-at-fault-collision | 1.00000 | 1.00000 |
| drivable-area compliance / SR | 0.70000 | 0.70000 |
| ego progress | 0.48689 | 0.51100 |
| score | 0.61120 | 0.62125 |
| runner time sum | 459.509 s | 483.732 s |

`运行事实`：R 分支改变了个别 rollout 和 progress，但没有改变这 10 scenes 的 collision/off-road outcome。  
`推断`：论文或后续实验若只报告 NR，可能高估对固定 replay 的适配；若只报告 IDM，也不能代表 learned BWM traffic。两者都应保留，并明确 IDM reactive ≠ Behaviour World Model。

## 7. 论文、当前开源仓库与产业内部验证

| 层级 | 可确认能力 |
| --- | --- |
| 论文系统 | long-tail discovery → 3DGS twin → BWM augmentation → rollout → behaviour-regularized RL post-training |
| 当前开源主仓 | SimEngine/AlgEngine 闭环、预建 MTGS asset 渲染、rare/synthetic 数据消费、reward-shaped/PG loss 代码、NR/IDM |
| 当前默认发布配置 | BWM 未集成、synthetic folder 占位、`rl_finetuning=False`、stable checkpoint roadmap 未完成 |
| 本次运行验证 | base checkpoint 的 1/10-scene NR/R 闭环；未验证训练和论文完整 WorldEngine checkpoint |

`推断`：不能把论文表中的 full WorldEngine 数字归因于当前默认开源配置的直接可运行结果。首个研究实验应把“当前可运行 baseline”与“论文完整系统”分开命名。

## 阶段 5 判定

阶段 5 已完成。最关键落差得到代码级证据：BWM/MTGS reconstruction 未在主仓完整开放，默认 RLFT 不执行 policy-gradient loss，quick-start 时间只适用于过滤后小子集。下一研究步骤应先复现 288 base NR/R，再决定是否把 `rl_finetuning=True` 作为新实验变量；不能直接宣称已复现论文完整 RL post-training。
