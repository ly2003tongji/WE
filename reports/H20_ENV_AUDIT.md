# H20 环境与代码审计

审计时间：2026-07-10（UTC+8）  
协作仓库分支：`h20/reproduction`  
官方代码：`OpenDriveLab/WorldEngine@fc79b937050ed9d68e18add2b480ae72578a7ea5`  
原始审计日志：`reports/bootstrap_20260710_204402.txt`

## 当前结论

- `运行事实`：单张 NVIDIA H20 可见且空闲，Driver 570.133.20，显存 97,871 MiB；宿主 `nvidia-smi` 显示 CUDA 12.8，系统 `nvcc` 实际为 12.1。
- `代码事实`：官方明确要求两个独立 Python 3.9 环境，均使用 PyTorch 2.0.1+cu118；SimEngine 另需 gsplat v1.4.0，AlgEngine 另需从源码编译 MMCV v1.6.2 custom ops（`docs/installation.md:3-8,45-82,111-174`）。
- `推断`：570 驱动向后兼容 cu118 runtime，无需也不得降级宿主驱动。主要风险不是驱动，而是旧 PyTorch/MMCV/Numba 与较新依赖、Hopper `sm_90` 和构建工具链的组合。
- `运行事实`：GitHub、Hugging Face、ModelScope 的 HTTPS 均可访问；GitHub TCP/22 和 Hugging Face TCP/443 可达，未检测到代理环境变量。
- `代码事实`：所谓“5 分钟 quick test”脚本默认运行完整 `navtest_failures`，README 明确称其为 288 场景，不是 1–10 场景 smoke test（`README.md:165-184`、`scripts/closed_loop_test.sh:3-8`）。`docs/quick_start.md:41-52` 展示的 “1/10” 输出与脚本默认值不一致。

## 机器资源

### 计算资源

| 项目 | 运行审计结果 |
| --- | --- |
| OS / kernel | Ubuntu 22.04.5 LTS；Linux 5.10.134-16.3.al8.x86_64 |
| CPU | 2× Intel Xeon Platinum 8469C；96 cores / 192 threads；2 NUMA nodes |
| RAM | 2.0 TiB total；约 1.4 TiB available；无 swap |
| GPU | 1× NVIDIA H20；97,871 MiB；审计时 0 MiB、0% util；MIG disabled |
| Driver | 570.133.20 |
| Driver CUDA 上限 | 12.8（来自 `nvidia-smi`，不是项目 runtime 版本） |
| 系统 CUDA toolkit | `nvcc` 12.1.105 |
| GCC / G++ | 11.4.0 |
| CMake / Ninja | 3.25.2 / 1.13.0 |
| 系统 Python | 3.11.11；不得用于项目环境 |
| Conda | 24.1.2；当前没有 `simengine`/`algengine` |

### 存储与推荐用途

| 路径 | 类型与容量 | 性能证据 | 推荐用途 |
| --- | --- | --- | --- |
| `/root`、`/home`、`/tmp` 所在 `/` | container overlay；1008 GiB total，217 GiB available；inode 9% used | 未做写入型 benchmark；overlay 位于容器本地层 | Conda 环境、源码构建和小型 cache；需控制在约 40 GiB 内，避免挤占系统盘 |
| `/dev/shm` | tmpfs；2.0 TiB available | 内存文件系统 | 临时编译/解压中转或短生命周期 IPC；重启/容器结束即丢失，不能存持久数据 |
| `/mnt/cpfs` | `fuse.aliyun-alinas-efc` 网络文件系统；70 TiB total，1.5 TiB available，98% used；inode 14% used | 挂载参数 `max_read=1048576`；未做破坏性写入 benchmark | 数据、权重、3DGS assets、实验输出；通过 symlink 接入，避免复制。大量小文件解压可能有 metadata 开销 |
| 裸设备 `nvme2n1`–`nvme5n1` | 各 3.5 TiB，当前未见挂载点 | 仅 `lsblk` 可见；是否允许使用未交代 | 不擅自分区或挂载 |

`推断`：环境放本地 Conda 根目录更适合频繁 import/编译；大数据放 CPFS。由于未获授权进行大文件写入压测，本报告不把理论存储类型当作实测吞吐。

### 网络与端口

- `运行事实`：GitHub、Hugging Face、ModelScope HTTPS 返回成功；GitHub SSH 22 可连接。
- `运行事实`：未发现 `HTTP_PROXY`、`HTTPS_PROXY`、`ALL_PROXY`、`NO_PROXY`。
- `运行事实`：本机 8080、8888、6006、2222 已被现有服务占用。
- `代码事实`：默认闭环脚本以监控目录和 `.pkl`/`.npy` 文件在 SimEngine 与 AlgEngine 间交换数据（`projects/SimEngine/scripts/run_testing.sh:50-88`、`projects/AlgEngine/closed_loop/sim_test.py:103-176`），未看到 quick test 依赖固定 TCP 监听端口。文档“socket”描述与当前入口实现不一致。

## 官方代码入口审计

### 固定版本

```text
repository: https://github.com/OpenDriveLab/WorldEngine
branch: main
commit: fc79b937050ed9d68e18add2b480ae72578a7ea5
commit date: 2026-06-27T12:48:33+08:00
subject: Update README.md
```

官方仓库保持 `main...origin/main`、工作区干净；本项目不修改其默认分支。

### Quick test 实际调用链

1. `scripts/closed_loop_test.sh:3-8` 固定配置、checkpoint、`navtest_failures` 和 `NR`。
2. `projects/SimEngine/scripts/run_testing.sh:25-29,50-88` 启动 SimEngine，再启动 AlgEngine 文件监控客户端。
3. SimEngine 读取整个 `all_scenarios.pkl`（`worldengine/runner/run_simulation.py:22-26`），当前 builder 的场景过滤仍是 TODO（`worldengine/runner/builders/env_builder.py:62-69`）。
4. 配置中的 `debug_scene_name` 虽然存在（`worldengine/configs/default_runner.yaml:12-17`），当前 Python 代码没有消费该字段。
5. 默认 `NR` 使用日志回放；传 `R` 时脚本覆写为 IDM policy/navigation（`run_testing.sh:61-67`）。

`代码事实`：现有 quick test 没有可直接传入的 “前 N 场景” 参数。因此阶段 4 前需在协作侧生成只含 1–10 个 key 的小型 scenario pickle，或增加不侵入上游的 wrapper；仅修改 `debug_scene_name` 不会生效。

## 兼容性风险

### cu118 与 Driver 570

- `运行事实`：Driver 570.133.20 足以加载 CUDA 11.8 构建的用户态二进制；宿主 CUDA 12.8 标识只是驱动能力上限。
- `推断`：环境内必须让 PyTorch wheel 与编译扩展统一使用 CUDA 11.8。若构建时误用 `/usr/local/cuda` 的 12.1 `nvcc`，PyTorch 会报告 CUDA 版本不匹配。
- 检查门槛：安装后记录 `torch.version.cuda`、`torch.cuda.get_device_capability()`、`torch.cuda.get_arch_list()`、`CUDA_HOME` 和 `nvcc --version`。

### PyTorch 2.0.1+cu118 与 H20

- `代码事实`：官方固定 PyTorch 2.0.1+cu118（`docs/installation.md:58-67,118-127`）。
- `推断`：CUDA 11.8 是首个正式支持 Hopper 的 toolkit，H20 的 `sm_90` 理论可用；但必须用真实 kernel/import smoke test 验证 wheel 是否包含所需 arch 或 PTX fallback，不能只以 `torch.cuda.is_available()` 下结论。

### MMCV 1.6.2

- `代码事实`：官方要求从 `v1.6.2` 源码、`MMCV_WITH_OPS=1` 编译并运行 `.dev_scripts/check_installation.py`（`docs/installation.md:129-151`）。
- `风险`：MMCV 1.x、MMDet 2.x、MMDet3D 1.0.0rc6 与 PyTorch 2.0 是旧组合；custom CUDA ops 可能在 C++ API、编译 arch 或 GCC/ABI 上失败。
- 控制措施：固定 commit/tag；先只编译 MMCV；显式设置 Conda CUDA 11.8 的 `CUDA_HOME`；必要时设置 `TORCH_CUDA_ARCH_LIST=9.0+PTX`；保存完整 build log。不得先大范围升级 OpenMMLab 栈。

### gsplat v1.4.0

- `代码事实`：官方从 Git tag `v1.4.0`、`--no-build-isolation` 安装（`docs/installation.md:70-76`）。
- `风险`：它会针对当前 torch/CUDA 构建扩展；系统 12.1 `nvcc` 与 torch cu118 混用是首要 ABI 风险，H20 arch 是次要风险。
- 控制措施：在 SimEngine 环境中先安装 torch cu118，再固定 gsplat tag 编译，并做一次最小 rasterization CUDA 调用。

### Python 包约束

- `代码事实`：AlgEngine 固定 `numba==0.53.0`、`numpy==1.23.4`；SimEngine 固定 `numpy==1.23.0`，同时要求较新的 pandas/scipy/matplotlib（两个 `requirements.txt`）。
- `运行事实`：PyPI 元数据表明 Numba 0.53.0 支持 Python `<3.10`，因此 Python 3.9 合法；但其年代早于 NumPy 1.23，仍需 import/JIT 验证。
- `风险`：老旧 binary wheels（Fiona/Rasterio/Pyogrio/Numba）和最新 pip resolver 可能产生构建或运行时冲突。必须分层安装并保存 `pip check` 结果。

## 推荐环境方案

1. 在本机 Conda 根下创建两个独立环境：`simengine`、`algengine`，均为 Python 3.9。
2. 环境和构建 cache 放本地 `/root/anaconda3`；数据和实验输出放 CPFS。
3. 每层设置停止点：
   - Python/pip；
   - torch/torchvision cu118 + GPU kernel；
   - SimEngine：gsplat；
   - AlgEngine：MMCV custom ops；
   - 各自 requirements；
   - 项目 import。
4. 不写入 `~/.bashrc`；运行时由激活脚本或命令显式导出 `WORLDENGINE_ROOT`、`NUPLAN_MAPS_ROOT`、`PYTHONPATH`。
5. 环境建立前导出 explicit/history 文件；失败时只删除对应 Conda env 和本地 build tree，不改系统 Python、驱动或共享数据。

### 回滚

```bash
conda env remove -n simengine
conda env remove -n algengine
```

以上仅删除项目隔离环境。上游源码位于被 `.gitignore` 排除的 `upstream/WorldEngine`，可重新克隆；CPFS 数据通过 symlink 接入，不随环境删除。

## 阶段 1 判定

阶段 1 已完成：机器、官方 commit、安装文档、quick start、入口和配置已审计。没有驱动级阻塞；存在需要在阶段 3 逐层验证的扩展编译与依赖冲突风险。当前不应下载全量 4.85 TB 数据。
