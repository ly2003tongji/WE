#!/usr/bin/env bash
# Orchestrate frozen-state Replay/IDM paired scoring (collaboration layer only).
set -euo pipefail

export WORLDENGINE_ROOT="${WORLDENGINE_ROOT:-/mnt/cpfs/prediction/lyyy/myself/WE/WE/upstream/WorldEngine}"
export SIMENGINE_ROOT="${WORLDENGINE_ROOT}/projects/SimEngine"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-${WORLDENGINE_ROOT}/data/raw/nuplan/dataset/maps}"
export CONDA_ENVS_PATH="${CONDA_ENVS_PATH:-/workspace/worldengine/envs}"
export PATH="/root/anaconda3/condabin:/usr/bin:/bin"
export PYTHONPATH="${SIMENGINE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${FROZEN_OUT_DIR:-/mnt/cpfs/prediction/lyyy/myself/WE/data/frozen_paired/smoke1_cutoff4_validity_v1}"
SCENE_PKL="${FROZEN_SCENE_PKL:-/mnt/cpfs/prediction/lyyy/myself/WE/data/smoke_1scene/scenarios/original/navtest_failures/all_scenarios.pkl}"
VOCAB="${FROZEN_VOCAB:-/mnt/cpfs/prediction/lyyy/myself/WE/data/hf/data/alg_engine/test_8192_kmeans.npy}"
CUTOFF="${FROZEN_CUTOFF:-4}"
PLAN_IDX_CSV="${FROZEN_PLAN_IDX_CSV:-}"
PLAN_IDX_STEP="${FROZEN_PLAN_IDX_STEP:-5}"
PY="${CONDA_ENVS_PATH}/simengine/bin/python"

mkdir -p "${OUT_DIR}"
echo "[1/3] build_frozen_traffic_futures (cutoff=${CUTOFF})"
BUILD_ARGS=(--scene-pkl "${SCENE_PKL}" --vocab "${VOCAB}" --out-dir "${OUT_DIR}" --cutoff "${CUTOFF}")
if [[ -n "${PLAN_IDX_CSV}" ]]; then
  BUILD_ARGS+=(--plan-idx-csv "${PLAN_IDX_CSV}" --plan-idx-step "${PLAN_IDX_STEP}")
elif [[ -n "${FROZEN_PLAN_IDX:-}" ]]; then
  BUILD_ARGS+=(--plan-idx "${FROZEN_PLAN_IDX}")
fi
"${PY}" "${SCRIPT_DIR}/build_frozen_traffic_futures.py" "${BUILD_ARGS[@]}"

echo "[2/3] score_frozen_disagreement"
"${PY}" "${SCRIPT_DIR}/score_frozen_disagreement.py" \
  --scene-pkl "${SCENE_PKL}" \
  --futures-dir "${OUT_DIR}" \
  --vocab "${VOCAB}" \
  --cutoff "${CUTOFF}"

echo "[3/3] compare_frozen_rewards"
"${PY}" "${SCRIPT_DIR}/compare_frozen_rewards.py" \
  --futures-dir "${OUT_DIR}"

echo "DONE out_dir=${OUT_DIR}"
