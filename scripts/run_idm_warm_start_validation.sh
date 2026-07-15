#!/usr/bin/env bash
# Stage 1.8: IDM warm-start continuity validation (1-scene, no expand).
set -euo pipefail

export WORLDENGINE_ROOT="${WORLDENGINE_ROOT:-/mnt/cpfs/prediction/lyyy/myself/WE/WE/upstream/WorldEngine}"
export SIMENGINE_ROOT="${WORLDENGINE_ROOT}/projects/SimEngine"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-${WORLDENGINE_ROOT}/data/raw/nuplan/dataset/maps}"
export CONDA_ENVS_PATH="${CONDA_ENVS_PATH:-/workspace/worldengine/envs}"
export PATH="/root/anaconda3/condabin:/usr/bin:/bin"
export PYTHONPATH="${SIMENGINE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${CONDA_ENVS_PATH}/simengine/bin/python"
OUT_DIR="${WARM_OUT_DIR:-/mnt/cpfs/prediction/lyyy/myself/WE/data/frozen_paired/smoke1_cutoff3_warm_v1}"
COLD_DIR="${WARM_COLD_DIR:-/mnt/cpfs/prediction/lyyy/myself/WE/data/frozen_paired/smoke1_cutoff3_validity_v2}"
SCENE_PKL="${FROZEN_SCENE_PKL:-/mnt/cpfs/prediction/lyyy/myself/WE/data/smoke_1scene/scenarios/original/navtest_failures/all_scenarios.pkl}"
VOCAB="${FROZEN_VOCAB:-/mnt/cpfs/prediction/lyyy/myself/WE/data/hf/data/alg_engine/test_8192_kmeans.npy}"
CUTOFF="${FROZEN_CUTOFF:-3}"

mkdir -p "${OUT_DIR}"
echo "[0] unit tests"
"${PY}" "${SCRIPT_DIR}/test_diagnose_geometry.py"
"${PY}" "${SCRIPT_DIR}/test_warm_start_helpers.py"
"${PY}" "${SCRIPT_DIR}/test_ranking_metrics.py"

echo "[1] validate_idm_warm_start"
cd "${SIMENGINE_ROOT}"
"${PY}" "${SCRIPT_DIR}/validate_idm_warm_start.py" \
  --scene-pkl "${SCENE_PKL}" \
  --vocab "${VOCAB}" \
  --cold-dir "${COLD_DIR}" \
  --out-dir "${OUT_DIR}" \
  --cutoff "${CUTOFF}" \
  2>&1 | tee "${OUT_DIR}/warm_start.log"

echo "DONE out_dir=${OUT_DIR}"
