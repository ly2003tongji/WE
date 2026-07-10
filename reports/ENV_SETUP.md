# H20 隔离环境搭建记录

执行时间：2026-07-10（UTC+8）  
官方代码：`OpenDriveLab/WorldEngine@fc79b937050ed9d68e18add2b480ae72578a7ea5`  
状态：阶段 3 完成；尚未下载 checkpoint、地图、scenario 或 3DGS assets

## 结论

- `运行事实`：两个持久 Python 3.9 环境已创建并可调用 H20：
  - `/workspace/worldengine/envs/simengine`
  - `/workspace/worldengine/envs/algengine`
- `运行事实`：两套环境均为 Python 3.9.19、CUDA toolkit 11.8.89、PyTorch 2.0.1+cu118、torchvision 0.15.2+cu118。
- `运行事实`：PyTorch wheel 明确包含 `sm_90`，两个环境都在 H20 上完成了 1024×1024 CUDA matmul。
- `运行事实`：gsplat v1.4.0 在 `sm_90` 上编译成功，并完成真实 64×64 antialiased rasterization。
- `运行事实`：MMCV full 1.6.2 custom ops 编译成功；官方 `.dev_scripts/check_installation.py` 的 CPU/CUDA 检查均通过，额外的 H20 `box_iou_rotated` CUDA 调用返回 `[[1.0]]`。
- `运行事实`：SimEngine import、Ray 本地任务和项目模块 import 通过；AlgEngine 的 OpenMMLab 栈、`mmdet3d_plugin` 和 NAVSIM/nuPlan import 通过。
- `运行事实`：持久占用精确为 25,827,864,576 bytes（SimEngine 12,662,599,680；AlgEngine 13,049,319,424；MMCV/NAVSIM 源码 115,945,472），低于用户告知的 `/workspace` 40 GB 限额。

## 持久与临时路径

| 用途 | 路径 | 持久性 |
| --- | --- | --- |
| SimEngine 环境 | `/workspace/worldengine/envs/simengine` | 持久 |
| AlgEngine 环境 | `/workspace/worldengine/envs/algengine` | 持久 |
| MMCV v1.6.2 源码 | `/workspace/worldengine/src/mmcv` | 持久；editable install 需要 |
| NAVSIM v1.1 源码 | `/workspace/worldengine/src/navsim` | 持久；通过 `PYTHONPATH` 使用 |
| Conda package cache | `/tmp/worldengine-conda-pkgs` | 非持久、可删除 |
| pip cache | `/tmp/worldengine-pip-cache*` | 非持久、可删除 |
| 数据/权重/输出 | 计划放 `/mnt/cpfs` | 持久数据卷 |

`运行事实`：底层 `df` 显示 `/workspace` 有较大共享容量，但用户明确给出 40 GB 逻辑限额，本项目按 40 GB 控制。

## 固定版本

### 共同层

| 包/组件 | 版本 |
| --- | --- |
| Python | 3.9.19 |
| CUDA toolkit / nvcc | 11.8.89 |
| PyTorch | 2.0.1+cu118 |
| torchvision | 0.15.2+cu118 |
| GPU / capability | NVIDIA H20 / `(9, 0)` |
| PyTorch arch list | 含 `sm_90` |

### SimEngine

| 包 | 版本 |
| --- | --- |
| gsplat | 1.4.0，git commit `4d3a3b69db4de0326f983ccf7b7b255271a17b01` |
| numpy | 1.23.0 |
| opencv-python | 4.10.0.84 |
| ray | 2.48.0 |
| hydra-core | 1.2.0 |
| nuplan-devkit | 1.2.0，upstream commit `ce3c323af01c0d7ec5672f7832ef53f9c679aab0` |
| pandas / scipy | 2.3.3 / 1.13.1 |
| Fiona / Rasterio / Pyogrio | 1.8.21 / 1.3.2 / 0.4.1 |
| pytorch-lightning | 2.2.1 |

`运行事实`：`pip check` 返回 `No broken requirements found`。

### AlgEngine

| 包/源码 | 版本 |
| --- | --- |
| MMCV full | 1.6.2，git commit `847612dfffbcd917b6c267fe9decd4f8afed676e` |
| mmcls | 0.25.0 |
| mmdet | 2.25.3 |
| mmdet3d | 1.0.0rc6 |
| mmsegmentation | 0.29.1 |
| numpy / numba / llvmlite | 1.23.4 / 0.53.0 / 0.36.0 |
| opencv-python | 4.9.0.80 |
| Shapely | 2.0.4 |
| NAVSIM | branch v1.1，commit `3e8291bfa89ff247231e0227778840cd0a036896` |
| nuplan-devkit | 1.2.0 |

## 执行命令

以下命令省略重复的工作目录切换，但保留影响复现的参数。

### 1. 创建持久环境

首次直接使用 `conda create -n simengine python=3.9` 失败，见“失败与处理”。最终命令：

```bash
CONDA_PKGS_DIRS=/tmp/worldengine-conda-pkgs \
conda create --prefix /workspace/worldengine/envs/simengine \
  python=3.9.19 pip=24.0 setuptools=69.5.1 wheel=0.43.0 -y

CONDA_PKGS_DIRS=/tmp/worldengine-conda-pkgs \
conda create --prefix /workspace/worldengine/envs/algengine \
  python=3.9.19 pip=24.0 setuptools=69.5.1 wheel=0.43.0 -y
```

### 2. 安装独立 CUDA 11.8 toolkit

两个环境分别执行：

```bash
CONDA_PKGS_DIRS=/tmp/worldengine-conda-pkgs \
conda install --prefix <ENV_PREFIX> \
  -c "nvidia/label/cuda-11.8.0" cuda-toolkit -y
```

每个完整 toolkit 安装后约占 9.7 GB。选择独立安装而非跨环境共享 `CUDA_HOME`，避免 MMCV/gsplat 构建互相耦合。

### 3. 安装并验证 PyTorch

```bash
CUDA_HOME=<ENV_PREFIX> PATH=<ENV_PREFIX>/bin:/usr/bin:/bin \
<ENV_PREFIX>/bin/python -m pip install \
  torch==2.0.1+cu118 torchvision==0.15.2+cu118 \
  --index-url https://download.pytorch.org/whl/cu118
```

安装后立即恢复官方 NumPy pin：

```bash
/workspace/worldengine/envs/simengine/bin/python -m pip install \
  numpy==1.23.0 pillow==10.4.0
/workspace/worldengine/envs/algengine/bin/python -m pip install \
  numpy==1.23.4 pillow==10.4.0
```

两个环境的 CUDA 测试结果：

```text
torch=2.0.1+cu118
torch.version.cuda=11.8
device=NVIDIA H20
capability=(9, 0)
arch_list includes sm_90
1024x1024 CUDA matmul: PASS
```

### 4. SimEngine 与 gsplat

```bash
CUDA_HOME=/workspace/worldengine/envs/simengine \
PATH=/workspace/worldengine/envs/simengine/bin:/usr/bin:/bin \
TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=16 \
/workspace/worldengine/envs/simengine/bin/python -m pip install ninja

CUDA_HOME=/workspace/worldengine/envs/simengine \
PATH=/workspace/worldengine/envs/simengine/bin:/usr/bin:/bin \
TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=16 \
/workspace/worldengine/envs/simengine/bin/python -m pip install \
  "git+https://github.com/nerfstudio-project/gsplat.git@v1.4.0" \
  --no-build-isolation

/workspace/worldengine/envs/simengine/bin/python -m pip install \
  -r projects/SimEngine/requirements.txt
```

gsplat 构建 wheel：`gsplat-1.4.0-cp39-cp39-linux_x86_64.whl`。真实 CUDA 测试：

```text
render shape=(1, 64, 64, 3), max=0.9526093
alpha shape=(1, 64, 64, 1), max=0.9526093
radii=[11]
```

Ray 测试启动本地 2-CPU instance，remote function `41 + 1` 返回 `42` 后正常 shutdown。

### 5. AlgEngine 与 MMCV

```bash
git clone --branch v1.6.2 --depth 1 \
  https://github.com/open-mmlab/mmcv.git \
  /workspace/worldengine/src/mmcv

CUDA_HOME=/workspace/worldengine/envs/algengine \
PATH=/workspace/worldengine/envs/algengine/bin:/usr/bin:/bin \
TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=16 MMCV_WITH_OPS=1 \
/workspace/worldengine/envs/algengine/bin/python -m pip install \
  -v -e /workspace/worldengine/src/mmcv
```

官方检查命令：

```bash
CUDA_HOME=/workspace/worldengine/envs/algengine \
PATH=/workspace/worldengine/envs/algengine/bin:/usr/bin:/bin \
/workspace/worldengine/envs/algengine/bin/python \
  /workspace/worldengine/src/mmcv/.dev_scripts/check_installation.py
```

最终结果：

```text
CPU ops were compiled successfully.
CUDA ops were compiled successfully.
MMCV Compiler: GCC 11.4
MMCV CUDA Compiler: 11.8
box_iou_rotated H20 CUDA result: [[1.0]]
```

完整编译 stdout 原先位于非持久终端记录中，没有写入协作仓库；本报告保留 commit、构建参数、官方检查输出和可重复命令。后续重建时应把 `-v` stdout 重定向到 CPFS 日志目录。

### 6. OpenMMLab、AlgEngine 和 NAVSIM

```bash
/workspace/worldengine/envs/algengine/bin/python -m pip install \
  mmcls==0.25.0 mmdet==2.25.3 mmdet3d==1.0.0rc6 \
  mmsegmentation==0.29.1

/workspace/worldengine/envs/algengine/bin/python -m pip install \
  -r projects/AlgEngine/requirements.txt
/workspace/worldengine/envs/algengine/bin/python -m pip install shapely==2.0.4

git clone --branch v1.1 --depth 1 \
  https://github.com/autonomousvision/navsim.git \
  /workspace/worldengine/src/navsim

/workspace/worldengine/envs/algengine/bin/python -m pip install \
  "nuplan-devkit @ git+https://github.com/motional/nuplan-devkit/@nuplan-devkit-v1.2"
```

项目 import 需要同时提供：

```bash
export WORLDENGINE_ROOT=/mnt/cpfs/prediction/lyyy/myself/WE/WE/upstream/WorldEngine
export NAVSIM_DEVKIT_ROOT=/workspace/worldengine/src/navsim
export PYTHONPATH=$WORLDENGINE_ROOT/projects/AlgEngine:$NAVSIM_DEVKIT_ROOT:$PYTHONPATH
```

## 失败、风险与处理

### Conda 最新 Python 3.9 package 校验失败

- `运行事实`：首次 `conda create -n simengine python=3.9` 选择 `python-3.9.25-h0dcde21_1`，报大量 `CondaVerificationError`，package manifest 所列 `.pyc` 在共享 cache 中缺失。
- 仅执行 `conda clean --packages --tarballs` 后问题复现。
- 处理：删除本次创建的 16 KB 不完整 prefix；使用隔离的 `/tmp/worldengine-conda-pkgs` cache，并固定 Python 3.9.19。两个环境随后创建成功。

### `conda run` 受宿主 PATH 顺序影响

- `运行事实`：当前宿主 PATH 把 `/usr/local/bin` 放在目标 env 之前，`conda run --prefix ... python` 错误调用 Python 3.11.11。
- 处理：安装和检查使用绝对 `<ENV_PREFIX>/bin/python`；运行上游脚本前使用干净 PATH：

```bash
PATH=/root/anaconda3/condabin:/usr/bin:/bin \
CONDA_ENVS_PATH=/workspace/worldengine/envs \
conda run --prefix /workspace/worldengine/envs/simengine python --version
```

`推断`：上游 `run_testing.sh` 固定使用 `conda run -n simengine/algengine`，阶段 4 需提供 wrapper 设置 `CONDA_ENVS_PATH` 和干净 PATH，或对协作侧脚本做最小适配；不能直接在当前宿主 PATH 下运行。

### NumPy 被依赖解析器升级

- PyTorch/torchvision 初装拉取 NumPy 2.0.2；MMCV/OpenCV 依赖解析也曾再次升级。
- MMCV 官方检查第一次因此报 `_ARRAY_API not found` / `Numpy is not available`。
- 处理：按官方 requirements 在每一层结束后重新固定 SimEngine 1.23.0、AlgEngine 1.23.4。官方 MMCV 检查随后通过。

### AlgEngine 的两个 metadata 冲突

`pip check` 仍报告：

```text
mmdet3d 1.0.0rc6 requires networkx<2.3,>=2.2, but networkx 2.5 is installed.
nuscenes-devkit 1.1.11 requires Shapely<2.0.0, but Shapely 2.0.4 is installed.
```

这是上游自身明确 pin 的组合：

- AlgEngine `requirements.txt` 固定 `networkx==2.5`；
- WorldEngine 安装文档在 requirements 后明确执行 `shapely==2.0.4`。

因此没有擅自降级。`运行事实`：`numba`、全部 OpenMMLab 包、nuPlan/NAVSIM、`mmdet3d_plugin` 和 MMCV CUDA op 均可 import/运行。后续若出现对应 API 运行错误，再做最小兼容 patch 并记录。

## 运行入口约定

不修改 `~/.bashrc`。阶段 4 每次显式设置：

```bash
export WORLDENGINE_ROOT=/mnt/cpfs/prediction/lyyy/myself/WE/WE/upstream/WorldEngine
export SIMENGINE_ROOT=$WORLDENGINE_ROOT/projects/SimEngine
export ALGENGINE_ROOT=$WORLDENGINE_ROOT/projects/AlgEngine
export NAVSIM_DEVKIT_ROOT=/workspace/worldengine/src/navsim
export NUPLAN_MAPS_ROOT=$WORLDENGINE_ROOT/data/raw/nuplan/dataset/maps
export CONDA_ENVS_PATH=/workspace/worldengine/envs
export PYTHONPATH=$SIMENGINE_ROOT:$ALGENGINE_ROOT:$NAVSIM_DEVKIT_ROOT
export PATH=/root/anaconda3/condabin:/usr/bin:/bin
```

阶段 4 开始前还需通过 symlink 接入 CPFS 上的 checkpoint、vocabulary、过滤后的 scenario pickle、匹配的 3DGS assets 和 nuPlan maps。Closed-loop dataset 不读取预计算 PDMS cache。

## 回滚

仅删除本项目隔离环境和源码，不影响宿主驱动、系统 Python 或共享数据：

```bash
conda env remove --prefix /workspace/worldengine/envs/simengine
conda env remove --prefix /workspace/worldengine/envs/algengine
rm -rf /workspace/worldengine/src/mmcv
rm -rf /workspace/worldengine/src/navsim
```

数据和权重将位于 CPFS，不属于环境回滚范围，禁止随环境删除。

## 阶段 3 判定

阶段 3 已完成。H20、PyTorch cu118、gsplat v1.4.0 和 MMCV v1.6.2 custom CUDA ops 均已有真实运行证据。当前阻塞从“环境兼容性”转移为“尚未下载最小数据集合”；下一步应按 `reports/DATA_MANIFEST.md` 选择性下载约 36 GB 的 1-shard smoke-test 集合，而不是全量数据。
