#!/usr/bin/env bash
# Restore-physics IDM validation (collaboration layer). Default: research-locked cutoff=4.
set -euo pipefail

export WORLDENGINE_ROOT="${WORLDENGINE_ROOT:-/mnt/cpfs/prediction/lyyy/myself/WE/WE/upstream/WorldEngine}"
export SIMENGINE_ROOT="${WORLDENGINE_ROOT}/projects/SimEngine"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-${WORLDENGINE_ROOT}/data/raw/nuplan/dataset/maps}"
export CONDA_ENVS_PATH="${CONDA_ENVS_PATH:-/workspace/worldengine/envs}"
export PATH="/root/anaconda3/condabin:/usr/bin:/bin"
export PYTHONPATH="${SIMENGINE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${CONDA_ENVS_PATH}/simengine/bin/python"
OUT_DIR="${RESTORE_OUT_DIR:-/mnt/cpfs/prediction/lyyy/myself/WE/data/frozen_paired/smoke1_cutoff4_restore_v1}"
# Legacy cutoff=3 dirs kept only for optional traj diagnostics (not scoring).
COLD_DIR="${RESTORE_COLD_DIR:-/mnt/cpfs/prediction/lyyy/myself/WE/data/frozen_paired/smoke1_cutoff3_validity_v2}"
WARM_DIR="${RESTORE_WARM_DIR:-/mnt/cpfs/prediction/lyyy/myself/WE/data/frozen_paired/smoke1_cutoff3_warm_v1}"
SCENE_PKL="${FROZEN_SCENE_PKL:-/mnt/cpfs/prediction/lyyy/myself/WE/data/smoke_1scene/scenarios/original/navtest_failures/all_scenarios.pkl}"
VOCAB="${FROZEN_VOCAB:-/mnt/cpfs/prediction/lyyy/myself/WE/data/hf/data/alg_engine/test_8192_kmeans.npy}"
CUTOFF="${FROZEN_CUTOFF:-4}"
PLAN_IDX_CSV="${FROZEN_PLAN_IDX_CSV:-${WORLDENGINE_ROOT}/experiments/closed_loop_exps/e2e_vadv2_50pct-disagreement-smoke1-NR-20260715/navtest_failures_NR/plan_traj/plan_idx.csv}"
PLAN_IDX_STEP="${FROZEN_PLAN_IDX_STEP:-5}"

mkdir -p "${OUT_DIR}"
echo "[0] unit tests"
"${PY}" "${SCRIPT_DIR}/test_diagnose_geometry.py"
"${PY}" "${SCRIPT_DIR}/test_warm_start_helpers.py"
"${PY}" "${SCRIPT_DIR}/test_restore_physics_helpers.py"
"${PY}" "${SCRIPT_DIR}/test_ranking_metrics.py"

echo "[1] validate_idm_restore_physics cutoff=${CUTOFF} plan_idx_csv step=${PLAN_IDX_STEP}"
cd "${SIMENGINE_ROOT}"
"${PY}" "${SCRIPT_DIR}/validate_idm_restore_physics.py" \
  --scene-pkl "${SCENE_PKL}" \
  --vocab "${VOCAB}" \
  --cold-dir "${COLD_DIR}" \
  --warm-dir "${WARM_DIR}" \
  --out-dir "${OUT_DIR}" \
  --cutoff "${CUTOFF}" \
  --plan-idx-csv "${PLAN_IDX_CSV}" \
  --plan-idx-step "${PLAN_IDX_STEP}" \
  2>&1 | tee "${OUT_DIR}/restore_physics.log"

echo "DONE out_dir=${OUT_DIR}"
