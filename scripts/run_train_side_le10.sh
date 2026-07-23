#!/usr/bin/env bash
# Train-side ≤10 cutoff=4 three-source pipeline (T1 post-download → T4 per-scene).
set -euo pipefail

WE_ROOT="${WE_ROOT:-/mnt/cpfs/prediction/lyyy/myself/WE/WE}"
DATA_ROOT="${DATA_ROOT:-/mnt/cpfs/prediction/lyyy/myself/WE/data}"
SIDECAR="${SIDECAR:-/mnt/cpfs/prediction/lyyy/myself/WE/nexus_sidecar}"
NEXUS_PY="${NEXUS_PY:-${SIDECAR}/env/nexus/bin/python}"
# SimEngine scoring / restore must use conda simengine (has nuplan)
PY="${PY:-/workspace/worldengine/envs/simengine/bin/python}"
EXTRACT_ROOT="${EXTRACT_ROOT:-${DATA_ROOT}/frozen_paired/train_side_le10}"
ASSET_FOLDER="${ASSET_FOLDER:-${WE_ROOT}/upstream/WorldEngine/data/sim_engine/assets/navtest_failures/assets}"
VOCAB="${VOCAB:-${DATA_ROOT}/hf/data/alg_engine/test_8192_kmeans.npy}"
SCENE_LIST_JSON="${SCENE_LIST_JSON:-${WE_ROOT}/reports/train_side_le10_scene_list.json}"
MAPS="${MAPS:-${DATA_ROOT}/maps/extracted}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-${MAPS}}"
export WORLDENGINE_ROOT="${WORLDENGINE_ROOT:-${WE_ROOT}/upstream/WorldEngine}"
export SIMENGINE_ROOT="${SIMENGINE_ROOT:-${WORLDENGINE_ROOT}/projects/SimEngine}"
export CONDA_ENVS_PATH="${CONDA_ENVS_PATH:-/workspace/worldengine/envs}"
export PATH="/root/anaconda3/condabin:/usr/bin:/bin:${PATH:-}"
export PYTHONPATH="${WE_ROOT}:${WE_ROOT}/scripts:${SIMENGINE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

if [[ ! -x "${PY}" ]]; then
  echo "ERROR: simengine python missing: ${PY}" >&2
  exit 1
fi

cd "${WE_ROOT}"
mkdir -p "${EXTRACT_ROOT}"
LOG="${EXTRACT_ROOT}/_pipeline.log"
exec > >(tee -a "${LOG}") 2>&1

echo "[gate] HEAD=$(git rev-parse HEAD)"
echo "[gate] upstream=$(git -C upstream/WorldEngine rev-parse HEAD)"

COLL_PKL="${DATA_ROOT}/hf/data/sim_engine/scenarios/original/navtrain_50pct_collision/all_scenarios.pkl"
EP_PKL="${DATA_ROOT}/hf/data/sim_engine/scenarios/original/navtrain_ep_per1/all_scenarios.pkl"

# Wait for downloads if needed
for i in $(seq 1 240); do
  if [[ -f "${COLL_PKL}" && -f "${EP_PKL}" ]]; then
    cs=$(stat -c%s "${COLL_PKL}" 2>/dev/null || echo 0)
    es=$(stat -c%s "${EP_PKL}" 2>/dev/null || echo 0)
    if [[ "${cs}" == "19078046784" && "${es}" == "2498094267" ]]; then
      echo "[T1] downloads size_ok"
      break
    fi
  fi
  if (( i == 240 )); then
    echo "[T1] ERROR downloads not ready after wait" >&2
    exit 1
  fi
  echo "[T1] waiting downloads... (${i})"
  sleep 30
done

echo "[T1] build scene list"
if [[ -f "${SCENE_LIST_JSON}" ]]; then
  echo "[T1] reuse existing ${SCENE_LIST_JSON}"
else
  "${PY}" "${WE_ROOT}/scripts/build_train_side_le10_scene_list.py" \
    --collision-pkl "${COLL_PKL}" \
    --ep-pkl "${EP_PKL}" \
    --extract-root "${EXTRACT_ROOT}" \
    --out-md "${WE_ROOT}/reports/train_side_le10_scene_list.md" \
    --out-json "${SCENE_LIST_JSON}"
fi

cd "${SIMENGINE_ROOT}"

mapfile -t TOKENS < <("${PY}" - <<'PY'
import json
from pathlib import Path
p = Path("/mnt/cpfs/prediction/lyyy/myself/WE/WE/reports/train_side_le10_scene_list.json")
d = json.loads(p.read_text())
for s in d["scenes"]:
    print(s["token"])
PY
)

GATE_A_TOKENS=()

for tok in "${TOKENS[@]}"; do
  echo "========== SCENE ${tok} =========="
  SCENE_DIR="${EXTRACT_ROOT}/${tok}"
  SCENE_PKL="${SCENE_DIR}/scene/all_scenarios.pkl"
  META="${SCENE_DIR}/scene_meta.json"
  RESTORE="${SCENE_DIR}/restore"
  NEXUS0="${SCENE_DIR}/nexus_seed0"
  SUM0="${SCENE_DIR}/three_source_seed0.json"

  bucket=$("${PY}" -c "import json; d=json.load(open('${SCENE_LIST_JSON}')); print(next(s['bucket'] for s in d['scenes'] if s['token']=='${tok}'))")
  sid=$("${PY}" -c "import json; d=json.load(open('${SCENE_LIST_JSON}')); print(next(s['scene_id'] for s in d['scenes'] if s['token']=='${tok}'))")

  if [[ ! -f "${META}" ]]; then
    echo "[T2] select plan_idx ${tok}"
    "${PY}" "${WE_ROOT}/scripts/select_static_feasible_median_plan_idx.py" \
      --scene-pkl "${SCENE_PKL}" \
      --out-meta "${META}" \
      --work-dir "${SCENE_DIR}/static_gate" \
      --asset-folder "${ASSET_FOLDER}" \
      --vocab "${VOCAB}" \
      --token "${tok}" \
      --bucket "${bucket}" \
      --scene-id "${sid}"
  fi
  plan_idx=$("${PY}" -c "import json; print(json.load(open('${META}'))['ego_conditioning']['plan_idx'])")
  echo "[T2] plan_idx=${plan_idx} token=${tok}"

  if [[ ! -f "${RESTORE}/restore_physics_summary.json" ]]; then
    echo "[T2] restore ${tok}"
    "${PY}" "${WE_ROOT}/scripts/validate_idm_restore_physics.py" \
      --scene-pkl "${SCENE_PKL}" \
      --out-dir "${RESTORE}" \
      --cutoff 4 \
      --horizon 9 \
      --plan-idx "${plan_idx}" \
      --vocab "${VOCAB}" \
      --asset-folder "${ASSET_FOLDER}" \
      --skip-attribution \
      --cold-dir "${RESTORE}/_unused_cold" \
      --warm-dir "${RESTORE}/_unused_warm" || {
        echo "[T2] restore FAILED ${tok} (continue)"
        continue
      }
  fi
  # overwrite CLI provenance with static_feasible provenance
  "${PY}" - "${META}" "${RESTORE}/plan_idx_resolution.json" <<'PY'
import json, sys
from pathlib import Path
meta=json.loads(Path(sys.argv[1]).read_text())
out=Path(sys.argv[2])
ego=meta.get("ego_conditioning") or {}
payload={
  "plan_idx": ego.get("plan_idx"),
  "source": "static_feasible_median_path_length",
  "n_feasible": ego.get("n_feasible"),
  "path_length_m": ego.get("path_length_m"),
  "path_length_percentiles": ego.get("path_length_percentiles"),
  "feasible_plan_idx_sha256": ego.get("feasible_plan_idx_sha256"),
  "path_length_vec_sha256": ego.get("path_length_vec_sha256"),
  "requested_ego_real_hash": ego.get("requested_ego_real_hash"),
  "note": ego.get("note"),
  "equals_legacy_710": ego.get("equals_legacy_710"),
  "equals_smoke_1333": ego.get("equals_smoke_1333"),
}
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(payload, indent=2, ensure_ascii=False)+"\n")
print("wrote", out)
PY

  if [[ ! -f "${RESTORE}/scores_log_replay.pkl" ]]; then
    echo "[T2] gate without scores; skip nexus, write degraded stub ${tok}"
    "${PY}" - <<PY
import json
from pathlib import Path
meta=json.loads(Path("${META}").read_text()) if Path("${META}").exists() else {}
gate={}
p=Path("${RESTORE}/restore_physics_summary.json")
if p.exists():
    gate=json.loads(p.read_text())
tr={}
tp=Path("${RESTORE}/pre_cutoff_irreversible_transition_flags.json")
if tp.exists():
    tr=json.loads(tp.read_text())
n_true=sum(1 for v in tr.values() if v is True)
out={
  "status":"degraded_no_scores",
  "token":"${tok}","bucket":"${bucket}","scene_id":"${sid}",
  "gate_a":False,"degraded":True,
  "degraded_reasons":[f"gate_grade={gate.get('cutoff_gate_grade')}", f"transition_true={n_true}", "missing_restore_scores"],
  "plan_idx_provenance": meta.get("ego_conditioning"),
  "conditioning_not_comparable_to_smoke_nr_1333": True,
  "idm_gate":{"grade":gate.get("cutoff_gate_grade"),"pre_cutoff_transition_true_count":n_true,"transition_flags":tr},
  "pairwise":{},
}
Path("${SUM0}").write_text(json.dumps(out,indent=2,ensure_ascii=False)+"\n")
print("wrote degraded stub", "${SUM0}")
PY
    continue
  fi

  if [[ ! -f "${NEXUS0}/future_nexus_seed0_meta.json" ]]; then
    echo "[T2] nexus seed0 ${tok}"
    "${NEXUS_PY}" "${WE_ROOT}/scripts/run_train_side_nexus_generate.py" \
      --scene-pkl "${SCENE_PKL}" \
      --scene-id "${sid}" \
      --out-dir "${NEXUS0}" \
      --plan-idx "${plan_idx}" \
      --seed 0 \
      --vocab "${VOCAB}" \
      --sidecar-root "${SIDECAR}" || {
        echo "[T2] nexus FAILED ${tok} (continue)"
        continue
      }
  fi

  if [[ ! -f "${SUM0}" ]]; then
    echo "[T2] score+compare seed0 ${tok}"
    "${PY}" "${WE_ROOT}/scripts/score_train_side_le10_scene.py" \
      --scene-pkl "${SCENE_PKL}" \
      --scene-id "${sid}" \
      --token "${tok}" \
      --bucket "${bucket}" \
      --restore-dir "${RESTORE}" \
      --nexus-dir "${NEXUS0}" \
      --scene-meta "${META}" \
      --out-summary "${SUM0}" \
      --plan-idx "${plan_idx}" \
      --seed 0 \
      --vocab "${VOCAB}" \
      --asset-folder "${ASSET_FOLDER}" || {
        echo "[T2] score FAILED ${tok} (continue)"
        continue
      }
  fi

  gate_a=$("${PY}" -c "import json; print(json.load(open('${SUM0}')).get('gate_a', False))")
  if [[ "${gate_a}" == "True" ]]; then
    GATE_A_TOKENS+=("${tok}")
  fi
done

echo "[T3] gate_a tokens: ${GATE_A_TOKENS[*]:-none}"
# pick first 3 gate-A for multi-seed
count=0
for tok in "${GATE_A_TOKENS[@]:-}"; do
  if (( count >= 3 )); then break; fi
  SCENE_DIR="${EXTRACT_ROOT}/${tok}"
  SCENE_PKL="${SCENE_DIR}/scene/all_scenarios.pkl"
  META="${SCENE_DIR}/scene_meta.json"
  RESTORE="${SCENE_DIR}/restore"
  plan_idx=$("${PY}" -c "import json; print(json.load(open('${META}'))['ego_conditioning']['plan_idx'])")
  sid=$("${PY}" -c "import json; d=json.load(open('${SCENE_LIST_JSON}')); print(next(s['scene_id'] for s in d['scenes'] if s['token']=='${tok}'))")
  bucket=$("${PY}" -c "import json; d=json.load(open('${SCENE_LIST_JSON}')); print(next(s['bucket'] for s in d['scenes'] if s['token']=='${tok}'))")
  for seed in 1 2; do
    NDIR="${SCENE_DIR}/nexus_seed${seed}"
    SUM="${SCENE_DIR}/three_source_seed${seed}.json"
    if [[ ! -f "${NDIR}/future_nexus_plan${plan_idx}_seed${seed}_meta.json" ]]; then
      echo "[T3] nexus seed=${seed} ${tok}"
      "${NEXUS_PY}" "${WE_ROOT}/scripts/run_train_side_nexus_generate.py" \
        --scene-pkl "${SCENE_PKL}" \
        --scene-id "${sid}" \
        --out-dir "${NDIR}" \
        --plan-idx "${plan_idx}" \
        --seed "${seed}" \
        --vocab "${VOCAB}" \
        --sidecar-root "${SIDECAR}"
    fi
    if [[ ! -f "${SUM}" ]]; then
      echo "[T3] score seed=${seed} ${tok}"
      "${PY}" "${WE_ROOT}/scripts/score_train_side_le10_scene.py" \
        --scene-pkl "${SCENE_PKL}" \
        --scene-id "${sid}" \
        --token "${tok}" \
        --bucket "${bucket}" \
        --restore-dir "${RESTORE}" \
        --nexus-dir "${NDIR}" \
        --scene-meta "${META}" \
        --out-summary "${SUM}" \
        --plan-idx "${plan_idx}" \
        --seed "${seed}" \
        --vocab "${VOCAB}" \
        --asset-folder "${ASSET_FOLDER}"
    fi
  done
  count=$((count + 1))
done

echo "[T4] aggregate"
"${PY}" "${WE_ROOT}/scripts/aggregate_train_side_le10.py" \
  --extract-root "${EXTRACT_ROOT}" \
  --scene-list "${SCENE_LIST_JSON}" \
  --out-md "${WE_ROOT}/reports/TRAIN_SIDE_LE10_DISAGREEMENT.md" \
  --out-json "${WE_ROOT}/reports/train_side_le10_summary.json" \
  --out-csv "${WE_ROOT}/reports/train_side_le10_summary.csv"

echo "[T5] HANDOFF short update"
"${PY}" "${WE_ROOT}/scripts/patch_handoff_train_side_le10.py"

echo "PIPELINE COMPLETE"
