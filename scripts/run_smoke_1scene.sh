#!/usr/bin/env bash

set -euo pipefail

export WORLDENGINE_ROOT="${WORLDENGINE_ROOT:-/mnt/cpfs/prediction/lyyy/myself/WE/WE/upstream/WorldEngine}"
export SIMENGINE_ROOT="${WORLDENGINE_ROOT}/projects/SimEngine"
export ALGENGINE_ROOT="${WORLDENGINE_ROOT}/projects/AlgEngine"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-/workspace/worldengine/src/navsim}"
export NUPLAN_MAPS_ROOT="${WORLDENGINE_ROOT}/data/raw/nuplan/dataset/maps"
export CONDA_ENVS_PATH="${CONDA_ENVS_PATH:-/workspace/worldengine/envs}"
export CUDA_HOME="/workspace/worldengine/envs/simengine"
export PYTHONPATH="${SIMENGINE_ROOT}:${ALGENGINE_ROOT}:${NAVSIM_DEVKIT_ROOT}"
SMOKE_MODEL_NAME="${SMOKE_MODEL_NAME:-e2e_vadv2_50pct-smoke1}"
SMOKE_REACT_TYPE="${SMOKE_REACT_TYPE:-NR}"

# 当前宿主把 /usr/local/bin 放在 Conda env 前面，会使 `conda run` 调错 Python。
export PATH="/root/anaconda3/condabin:/usr/bin:/bin"

bash "$(dirname "${BASH_SOURCE[0]}")/setup_smoke_links.sh"

bash "${SIMENGINE_ROOT}/scripts/run_testing.sh" \
  "${ALGENGINE_ROOT}/configs/worldengine/e2e_vadv2_50pct.py" \
  "${WORLDENGINE_ROOT}/data/alg_engine/ckpts/e2e_vadv2_50pct_ep8.pth" \
  "${SMOKE_MODEL_NAME}" \
  "navtest_failures" \
  "${SMOKE_REACT_TYPE}"
