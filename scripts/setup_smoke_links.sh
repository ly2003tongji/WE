#!/usr/bin/env bash

set -euo pipefail

WORLDENGINE_ROOT="${WORLDENGINE_ROOT:-/mnt/cpfs/prediction/lyyy/myself/WE/WE/upstream/WorldEngine}"
DATA_ROOT="${WORLDENGINE_DATA_ROOT:-/mnt/cpfs/prediction/lyyy/myself/WE/data}"
SMOKE_SUBSET_NAME="${SMOKE_SUBSET_NAME:-smoke_1scene}"
SMOKE_ROOT="${DATA_ROOT}/${SMOKE_SUBSET_NAME}"
HF_ROOT="${DATA_ROOT}/hf"
MAP_ROOT="${DATA_ROOT}/maps/extracted/nuplan-maps-v1.0"

link_path() {
  local source_path="$1"
  local target_path="$2"

  if [[ ! -e "${source_path}" ]]; then
    echo "缺少源路径: ${source_path}" >&2
    exit 1
  fi
  mkdir -p "$(dirname "${target_path}")"
  if [[ -L "${target_path}" ]]; then
    ln -sfn "${source_path}" "${target_path}"
  elif [[ -e "${target_path}" ]]; then
    echo "目标已存在且不是 symlink，拒绝覆盖: ${target_path}" >&2
    exit 1
  else
    ln -s "${source_path}" "${target_path}"
  fi
}

link_path \
  "${HF_ROOT}/data/alg_engine/ckpts/e2e_vadv2_50pct_ep8.pth" \
  "${WORLDENGINE_ROOT}/data/alg_engine/ckpts/e2e_vadv2_50pct_ep8.pth"
link_path \
  "${HF_ROOT}/data/alg_engine/pdms_cache/pdm_8192_gt_cache_navtest.pkl" \
  "${WORLDENGINE_ROOT}/data/alg_engine/pdms_cache/pdm_8192_gt_cache_navtest.pkl"
link_path \
  "${HF_ROOT}/data/alg_engine/test_8192_kmeans.npy" \
  "${WORLDENGINE_ROOT}/data/alg_engine/test_8192_kmeans.npy"
link_path \
  "${SMOKE_ROOT}/scenarios/original/navtest_failures" \
  "${WORLDENGINE_ROOT}/data/sim_engine/scenarios/original/navtest_failures"
link_path \
  "${SMOKE_ROOT}/assets/navtest_failures/assets" \
  "${WORLDENGINE_ROOT}/data/sim_engine/assets/navtest_failures/assets"
link_path \
  "${MAP_ROOT}" \
  "${WORLDENGINE_ROOT}/data/raw/nuplan/dataset/maps"

echo "smoke test symlink 已就绪"
echo "WORLDENGINE_ROOT=${WORLDENGINE_ROOT}"
echo "WORLDENGINE_DATA_ROOT=${DATA_ROOT}"
