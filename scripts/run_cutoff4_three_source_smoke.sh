#!/usr/bin/env bash
# cutoff=4 three-source smoke orchestrator (collaboration layer only).
set -euo pipefail

WE_ROOT="${WE_ROOT:-/mnt/cpfs/prediction/lyyy/myself/WE/WE}"
SIDECAR="${SIDECAR:-/mnt/cpfs/prediction/lyyy/myself/WE/nexus_sidecar}"
NEXUS_PY="${NEXUS_PY:-${SIDECAR}/env/nexus/bin/python}"
SIM_PY="${SIM_PY:-/workspace/worldengine/envs/simengine/bin/python}"
SCRIPT="${WE_ROOT}/scripts/run_cutoff4_three_source_smoke.py"
NEXUS_OUT="${NEXUS_OUT:-${SIDECAR}/outputs/smoke_cutoff4_three_source}"
RESTORE_DIR="${RESTORE_DIR:-/mnt/cpfs/prediction/lyyy/myself/WE/data/frozen_paired/smoke1_cutoff4_restore_v1}"

export WORLDENGINE_ROOT="${WORLDENGINE_ROOT:-${WE_ROOT}/upstream/WorldEngine}"
export SIMENGINE_ROOT="${SIMENGINE_ROOT:-${WORLDENGINE_ROOT}/projects/SimEngine}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-${WORLDENGINE_ROOT}/data/raw/nuplan/dataset/maps}"
export PYTHONPATH="${WE_ROOT}:${WE_ROOT}/scripts:${SIMENGINE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p "${NEXUS_OUT}"

echo "[A] Nexus generate plan_idx=1333 seed=0"
"${NEXUS_PY}" "${SCRIPT}" --phase A --nexus-out "${NEXUS_OUT}" 2>&1 | tee "${NEXUS_OUT}/phase_a.log"

echo "[B] PDM score + three-source compare"
cd "${SIMENGINE_ROOT}"
"${SIM_PY}" "${SCRIPT}" --phase B --nexus-out "${NEXUS_OUT}" --restore-dir "${RESTORE_DIR}" 2>&1 | tee "${NEXUS_OUT}/phase_b.log"

echo "DONE nexus_out=${NEXUS_OUT}"
