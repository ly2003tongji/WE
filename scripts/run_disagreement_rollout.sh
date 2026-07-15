#!/usr/bin/env bash
# Collaboration-layer wrapper for 1-scene dense-reward NR/R rollouts.
# Does NOT modify upstream/WorldEngine. Enables with_dense_reward_manager=true
# by calling SimEngine/AlgEngine entrypoints directly (mirrors run_testing.sh).

set -euo pipefail

export WORLDENGINE_ROOT="${WORLDENGINE_ROOT:-/mnt/cpfs/prediction/lyyy/myself/WE/WE/upstream/WorldEngine}"
export SIMENGINE_ROOT="${WORLDENGINE_ROOT}/projects/SimEngine"
export ALGENGINE_ROOT="${WORLDENGINE_ROOT}/projects/AlgEngine"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-/workspace/worldengine/src/navsim}"
export NUPLAN_MAPS_ROOT="${WORLDENGINE_ROOT}/data/raw/nuplan/dataset/maps"
export CONDA_ENVS_PATH="${CONDA_ENVS_PATH:-/workspace/worldengine/envs}"
export CUDA_HOME="${CUDA_HOME:-/workspace/worldengine/envs/simengine}"
export PYTHONPATH="${SIMENGINE_ROOT}:${ALGENGINE_ROOT}:${NAVSIM_DEVKIT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
# Host PATH puts /usr/local/bin ahead of conda; keep a clean PATH for conda run.
export PATH="/root/anaconda3/condabin:/usr/bin:/bin"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WE_REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_ROOT="${WORLDENGINE_DATA_ROOT:-/mnt/cpfs/prediction/lyyy/myself/WE/data}"
LOG_ROOT="${DISAGREEMENT_LOG_ROOT:-${DATA_ROOT}/logs}"
mkdir -p "${LOG_ROOT}"

SMOKE_SUBSET_NAME="${SMOKE_SUBSET_NAME:-smoke_1scene}"
DISAGREEMENT_SOURCE="${DISAGREEMENT_SOURCE:-NR}"
DISAGREEMENT_MODEL_NAME="${DISAGREEMENT_MODEL_NAME:-}"
CFG="${DISAGREEMENT_CFG:-${ALGENGINE_ROOT}/configs/worldengine/e2e_vadv2_50pct.py}"
CKPT="${DISAGREEMENT_CKPT:-${WORLDENGINE_ROOT}/data/alg_engine/ckpts/e2e_vadv2_50pct_ep8.pth}"
DATA_TYPE="${DISAGREEMENT_DATA_TYPE:-navtest_failures}"
ASSET_NAME="${DISAGREEMENT_ASSET_NAME:-${DATA_TYPE}}"
WALL_LIMIT_SEC="${DISAGREEMENT_WALL_LIMIT_SEC:-3600}"
RSS_LIMIT_GB="${DISAGREEMENT_RSS_LIMIT_GB:-64}"
OUTPUT_LIMIT_GB="${DISAGREEMENT_OUTPUT_LIMIT_GB:-5}"
MONITOR_INTERVAL_SEC="${DISAGREEMENT_MONITOR_INTERVAL_SEC:-5}"

if [[ "${DISAGREEMENT_SOURCE}" != "NR" && "${DISAGREEMENT_SOURCE}" != "R" ]]; then
  echo "DISAGREEMENT_SOURCE must be NR or R, got: ${DISAGREEMENT_SOURCE}" >&2
  exit 1
fi

if [[ -z "${DISAGREEMENT_MODEL_NAME}" ]]; then
  TS="$(date +%Y%m%d_%H%M%S)"
  DISAGREEMENT_MODEL_NAME="e2e_vadv2_50pct-disagreement-${SMOKE_SUBSET_NAME}-${DISAGREEMENT_SOURCE}-${TS}"
fi

export SMOKE_SUBSET_NAME
bash "${SCRIPT_DIR}/setup_smoke_links.sh"

TEST_PATH="${WORLDENGINE_ROOT}/experiments/closed_loop_exps/${DISAGREEMENT_MODEL_NAME}/${DATA_TYPE}_${DISAGREEMENT_SOURCE}"
if [[ -d "${TEST_PATH}" ]]; then
  echo "ERROR: output path already exists (refuse overwrite): ${TEST_PATH}" >&2
  exit 1
fi

ASSET_FOLDER_PATH="${WORLDENGINE_ROOT}/data/sim_engine/assets/${ASSET_NAME}/assets"
DATAFILE_FOLDER_PATH="data/sim_engine/scenarios/original/${DATA_TYPE}"

mkdir -p "${TEST_PATH}/plan_traj" "${TEST_PATH}/frames" "${TEST_PATH}/merged_ann_files" "${TEST_PATH}/WE_output" "${TEST_PATH}/resource_monitor"

RUN_META="${TEST_PATH}/resource_monitor/run_meta.txt"
MONITOR_CSV="${TEST_PATH}/resource_monitor/resource_samples.csv"
MONITOR_SUMMARY="${TEST_PATH}/resource_monitor/resource_summary.txt"
PID_FILE="${TEST_PATH}/resource_monitor/pids.txt"
GATE_FILE="${TEST_PATH}/resource_monitor/hard_gate.txt"
STDOUT_LOG="${LOG_ROOT}/${DISAGREEMENT_MODEL_NAME}.stdout.log"

{
  echo "start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "start_epoch=$(date +%s)"
  echo "collaboration_commit=$(git -C "${WE_REPO_ROOT}" rev-parse HEAD 2>/dev/null || true)"
  echo "upstream_commit=$(git -C "${WORLDENGINE_ROOT}" rev-parse HEAD 2>/dev/null || true)"
  echo "model_name=${DISAGREEMENT_MODEL_NAME}"
  echo "source=${DISAGREEMENT_SOURCE}"
  echo "subset=${SMOKE_SUBSET_NAME}"
  echo "cfg=${CFG}"
  echo "ckpt=${CKPT}"
  echo "test_path=${TEST_PATH}"
  echo "with_dense_reward_manager=true"
  echo "wall_limit_sec=${WALL_LIMIT_SEC}"
  echo "rss_limit_gb=${RSS_LIMIT_GB}"
  echo "output_limit_gb=${OUTPUT_LIMIT_GB}"
} > "${RUN_META}"

echo "timestamp_epoch,wall_s,sim_rss_kb,alg_rss_kb,gpu_mem_used_mib,gpu_util_pct,output_bytes" > "${MONITOR_CSV}"
: > "${PID_FILE}"
: > "${GATE_FILE}"

START_EPOCH="$(date +%s)"
export START_EPOCH TEST_PATH MONITOR_CSV MONITOR_SUMMARY GATE_FILE PID_FILE
export WALL_LIMIT_SEC RSS_LIMIT_GB OUTPUT_LIMIT_GB MONITOR_INTERVAL_SEC

monitor_loop() {
  while true; do
    if [[ -f "${GATE_FILE}" ]] && grep -q HARD_GATE "${GATE_FILE}" 2>/dev/null; then
      exit 0
    fi
    now="$(date +%s)"
    wall=$((now - START_EPOCH))
    sim_rss=0
    alg_rss=0
    while read -r pid rss cmd; do
      [[ -z "${pid:-}" ]] && continue
      case "${cmd}" in
        *run_simulation.py*"${TEST_PATH}"*|*run_simulation.py*navtest_failures*"${DISAGREEMENT_MODEL_NAME}"*)
          sim_rss=$((sim_rss + rss))
          ;;
        *sim_test.py*"${TEST_PATH}"*)
          alg_rss=$((alg_rss + rss))
          ;;
      esac
    done < <(ps -eo pid=,rss=,args= 2>/dev/null || true)

    gpu_mem=0
    gpu_util=0
    if command -v nvidia-smi >/dev/null 2>&1; then
      read -r gpu_mem gpu_util < <(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr -d ' ' | tr ',' ' ')
      gpu_mem="${gpu_mem:-0}"
      gpu_util="${gpu_util:-0}"
    fi
    out_bytes="$(du -sb "${TEST_PATH}" 2>/dev/null | awk '{print $1}')"
    out_bytes="${out_bytes:-0}"
    echo "${now},${wall},${sim_rss},${alg_rss},${gpu_mem},${gpu_util},${out_bytes}" >> "${MONITOR_CSV}"

    rss_kb=${sim_rss}
    if (( alg_rss > sim_rss )); then rss_kb=${alg_rss}; fi
    rss_gb="$(awk -v k="${rss_kb}" 'BEGIN{printf "%.3f", k/1024/1024}')"
    out_gb="$(awk -v b="${out_bytes}" 'BEGIN{printf "%.3f", b/1024/1024/1024}')"

    if (( wall > WALL_LIMIT_SEC )); then
      echo "HARD_GATE: wall_clock ${wall}s exceeded ${WALL_LIMIT_SEC}s" | tee -a "${GATE_FILE}" "${MONITOR_SUMMARY}"
      if [[ -f "${PID_FILE}" ]]; then
        # shellcheck disable=SC2046
        kill $(awk '{print $1}' "${PID_FILE}" | tr '\n' ' ') 2>/dev/null || true
      fi
      pkill -f "${TEST_PATH}" 2>/dev/null || true
      exit 0
    fi
    if awk -v a="${rss_gb}" -v b="${RSS_LIMIT_GB}" 'BEGIN{exit !(a+0>b+0)}'; then
      echo "HARD_GATE: peak RSS ${rss_gb}GB exceeded ${RSS_LIMIT_GB}GB" | tee -a "${GATE_FILE}" "${MONITOR_SUMMARY}"
      if [[ -f "${PID_FILE}" ]]; then
        kill $(awk '{print $1}' "${PID_FILE}" | tr '\n' ' ') 2>/dev/null || true
      fi
      pkill -f "${TEST_PATH}" 2>/dev/null || true
      exit 0
    fi
    if awk -v a="${out_gb}" -v b="${OUTPUT_LIMIT_GB}" 'BEGIN{exit !(a+0>b+0)}'; then
      echo "HARD_GATE: output ${out_gb}GB exceeded ${OUTPUT_LIMIT_GB}GB" | tee -a "${GATE_FILE}" "${MONITOR_SUMMARY}"
      if [[ -f "${PID_FILE}" ]]; then
        kill $(awk '{print $1}' "${PID_FILE}" | tr '\n' ' ') 2>/dev/null || true
      fi
      pkill -f "${TEST_PATH}" 2>/dev/null || true
      exit 0
    fi
    sleep "${MONITOR_INTERVAL_SEC}"
  done
}

monitor_loop &
MONITOR_PID=$!

cleanup() {
  local code=$?
  if [[ -n "${MONITOR_PID:-}" ]] && kill -0 "${MONITOR_PID}" 2>/dev/null; then
    kill "${MONITOR_PID}" 2>/dev/null || true
    wait "${MONITOR_PID}" 2>/dev/null || true
  fi
  if [[ -n "${SIM_PID:-}" ]] && kill -0 "${SIM_PID}" 2>/dev/null; then
    kill "${SIM_PID}" 2>/dev/null || true
    wait "${SIM_PID}" 2>/dev/null || true
  fi
  if [[ -n "${ALG_PID:-}" ]] && kill -0 "${ALG_PID}" 2>/dev/null; then
    kill "${ALG_PID}" 2>/dev/null || true
    wait "${ALG_PID}" 2>/dev/null || true
  fi
  echo "end_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${RUN_META}"
  echo "end_epoch=$(date +%s)" >> "${RUN_META}"
  echo "exit_code=${code}" >> "${RUN_META}"
}
trap cleanup EXIT INT TERM

SIM_CMD=(
  python worldengine/runner/run_simulation.py
  debug_mode=True
  debug_scene_name=null
  "data_file_folder_path=${DATAFILE_FOLDER_PATH}"
  "asset_folder_path=${ASSET_FOLDER_PATH}"
  data_pkl_file_name=all_scenarios.pkl
  "output_dir=${TEST_PATH}/WE_output"
  "job_name=${DATA_TYPE}_${DISAGREEMENT_SOURCE}_${DISAGREEMENT_MODEL_NAME}"
  use_planner_actions=true
  ego_controller=log_play_controller
  ego_policy=env_input_policy
  ego_client=navformer_client
  ego_navigation=trajectory_navigation
  "planner_data_path=${TEST_PATH}/plan_traj"
  "planner_client_folder=${TEST_PATH}/frames"
  with_metric_manager=true
  with_dense_reward_manager=true
  distributed_mode=SINGLE_NODE
  enable_resume=false
  "completed_scenarios_dir=${TEST_PATH}/completed_scenarios"
)
if [[ "${DISAGREEMENT_SOURCE}" == "R" ]]; then
  SIM_CMD+=(agent_policy=idm_policy agent_navigation=idm_navigation)
fi

cd "${SIMENGINE_ROOT}"
conda run --no-capture-output -n simengine "${SIM_CMD[@]}" \
  > >(tee -a "${STDOUT_LOG}") 2>&1 &
SIM_PID=$!

cd "${ALGENGINE_ROOT}"
conda run --no-capture-output -n algengine \
  python closed_loop/sim_test.py \
    "${CFG}" \
    "${CKPT}" \
    --log-dir "${TEST_PATH}/WE_output" \
    --cfg-options \
      "sim.monitored_folder=${TEST_PATH}/frames" \
      "sim.plan_save_path=${TEST_PATH}/plan_traj" \
      "sim.merged_ann_save_dir=${TEST_PATH}/merged_ann_files" \
      sim.clean_temp_files=True \
      sim.clean_record_data=False \
      "data_root=${TEST_PATH}/WE_output/openscene_format/" \
  > >(tee -a "${STDOUT_LOG}") 2>&1 &
ALG_PID=$!

{
  echo "${SIM_PID} sim"
  echo "${ALG_PID} alg"
  echo "${MONITOR_PID} monitor"
} > "${PID_FILE}"

echo "sim_pid=${SIM_PID}" >> "${RUN_META}"
echo "alg_pid=${ALG_PID}" >> "${RUN_META}"
echo "monitor_pid=${MONITOR_PID}" >> "${RUN_META}"
echo "stdout_log=${STDOUT_LOG}" >> "${RUN_META}"
echo "Started dense-reward rollout: source=${DISAGREEMENT_SOURCE} model=${DISAGREEMENT_MODEL_NAME}"
echo "TEST_PATH=${TEST_PATH}"

set +e
wait "${ALG_PID}"
ALG_RC=$?
wait "${SIM_PID}"
SIM_RC=$?
set -e

kill "${MONITOR_PID}" 2>/dev/null || true
wait "${MONITOR_PID}" 2>/dev/null || true

END_EPOCH="$(date +%s)"
WALL_S=$((END_EPOCH - START_EPOCH))
OUT_BYTES="$(du -sb "${TEST_PATH}" 2>/dev/null | awk '{print $1}')"
OUT_BYTES="${OUT_BYTES:-0}"

python3 - "${MONITOR_CSV}" "${MONITOR_SUMMARY}" "${WALL_S}" "${OUT_BYTES}" "${SIM_RC}" "${ALG_RC}" "${GATE_FILE}" <<'PY'
import csv, sys
from pathlib import Path
path, summary, wall_s, out_bytes, sim_rc, alg_rc, gate_file = sys.argv[1:8]
rows = []
with open(path) as f:
    for row in csv.DictReader(f):
        rows.append(row)
peak_sim = max((int(float(r["sim_rss_kb"])) for r in rows), default=0)
peak_alg = max((int(float(r["alg_rss_kb"])) for r in rows), default=0)
peak_gpu = max((float(r["gpu_mem_used_mib"]) for r in rows), default=0.0)
peak_util = max((float(r["gpu_util_pct"]) for r in rows), default=0.0)
peak_out = max((int(float(r["output_bytes"])) for r in rows), default=int(out_bytes))
gate = Path(gate_file).read_text().strip() if Path(gate_file).exists() else ""
with open(summary, "w") as f:
    f.write(f"wall_s={wall_s}\n")
    f.write(f"sim_rc={sim_rc}\n")
    f.write(f"alg_rc={alg_rc}\n")
    f.write(f"peak_sim_rss_kb={peak_sim}\n")
    f.write(f"peak_alg_rss_kb={peak_alg}\n")
    f.write(f"peak_rss_gb={max(peak_sim, peak_alg)/1024/1024:.3f}\n")
    f.write(f"peak_gpu_mem_mib={peak_gpu:.1f}\n")
    f.write(f"peak_gpu_util_pct={peak_util:.1f}\n")
    f.write(f"final_output_bytes={out_bytes}\n")
    f.write(f"peak_output_bytes={peak_out}\n")
    f.write(f"final_output_gb={int(out_bytes)/1024/1024/1024:.3f}\n")
    f.write(f"hard_gate={gate or 'none'}\n")
print(open(summary).read())
PY

echo "alg_rc=${ALG_RC}" >> "${RUN_META}"
echo "sim_rc=${SIM_RC}" >> "${RUN_META}"
echo "DISAGREEMENT_MODEL_NAME=${DISAGREEMENT_MODEL_NAME}"
echo "TEST_PATH=${TEST_PATH}"

if grep -q HARD_GATE "${GATE_FILE}" 2>/dev/null; then
  echo "ERROR: hard gate triggered; see ${GATE_FILE}" >&2
  exit 91
fi
if [[ "${ALG_RC}" -ne 0 || "${SIM_RC}" -ne 0 ]]; then
  echo "ERROR: rollout failed alg_rc=${ALG_RC} sim_rc=${SIM_RC}" >&2
  exit 1
fi
