#!/usr/bin/env bash
# Train-side ~50 cutoff=4 three-source pipeline (E1→E5). Serial; no GPU contention.
set -euo pipefail

WE_ROOT="${WE_ROOT:-/mnt/cpfs/prediction/lyyy/myself/WE/WE}"
DATA_ROOT="${DATA_ROOT:-/mnt/cpfs/prediction/lyyy/myself/WE/data}"
SIDECAR="${SIDECAR:-/mnt/cpfs/prediction/lyyy/myself/WE/nexus_sidecar}"
NEXUS_PY="${NEXUS_PY:-${SIDECAR}/env/nexus/bin/python}"
PY="${PY:-/workspace/worldengine/envs/simengine/bin/python}"
EXTRACT_ROOT="${EXTRACT_ROOT:-${DATA_ROOT}/frozen_paired/train_side_le50}"
LE10_ROOT="${LE10_ROOT:-${DATA_ROOT}/frozen_paired/train_side_le10}"
ASSET_FOLDER="${ASSET_FOLDER:-${WE_ROOT}/upstream/WorldEngine/data/sim_engine/assets/navtest_failures/assets}"
VOCAB="${VOCAB:-${DATA_ROOT}/hf/data/alg_engine/test_8192_kmeans.npy}"
SCENE_LIST_JSON="${SCENE_LIST_JSON:-${WE_ROOT}/reports/train_side_le50_scene_list.json}"
MAPS="${MAPS:-${DATA_ROOT}/maps/extracted}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-${MAPS}}"
export WORLDENGINE_ROOT="${WORLDENGINE_ROOT:-${WE_ROOT}/upstream/WorldEngine}"
export SIMENGINE_ROOT="${SIMENGINE_ROOT:-${WORLDENGINE_ROOT}/projects/SimEngine}"
export CONDA_ENVS_PATH="${CONDA_ENVS_PATH:-/workspace/worldengine/envs}"
export PATH="/root/anaconda3/condabin:/usr/bin:/bin:${PATH:-}"
export PYTHONPATH="${WE_ROOT}:${WE_ROOT}/scripts:${SIMENGINE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

MAX_ATTEMPTS="${MAX_ATTEMPTS:-65}"
TARGET_A_MIN="${TARGET_A_MIN:-48}"
# le10 already consumed 10 attempts
LE10_ATTEMPTS=10

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
cs=$(stat -c%s "${COLL_PKL}")
es=$(stat -c%s "${EP_PKL}")
[[ "${cs}" == "19078046784" && "${es}" == "2498094267" ]] || { echo "ERROR pkl size"; exit 1; }

echo "[E1] build scene list"
if [[ -f "${SCENE_LIST_JSON}" ]]; then
  echo "[E1] reuse existing ${SCENE_LIST_JSON}"
else
  "${PY}" "${WE_ROOT}/scripts/build_train_side_le50_scene_list.py" \
    --collision-pkl "${COLL_PKL}" \
    --ep-pkl "${EP_PKL}" \
    --extract-root "${EXTRACT_ROOT}" \
    --out-md "${WE_ROOT}/reports/train_side_le50_scene_list.md" \
    --out-json "${SCENE_LIST_JSON}"
fi

cd "${SIMENGINE_ROOT}"

ATTEMPT_STATE="${EXTRACT_ROOT}/_attempt_state.json"
if [[ ! -f "${ATTEMPT_STATE}" ]]; then
  echo "{\"new_restore_attempts\":0,\"le10_attempts\":${LE10_ATTEMPTS}}" > "${ATTEMPT_STATE}"
fi

run_one_scene() {
  local tok="$1"
  local bucket="$2"
  local sid="$3"
  local reuse="$4"
  local SCENE_DIR="${EXTRACT_ROOT}/${tok}"
  local SCENE_PKL="${SCENE_DIR}/scene/all_scenarios.pkl"
  local META="${SCENE_DIR}/scene_meta.json"
  local RESTORE="${SCENE_DIR}/restore"
  local NEXUS0="${SCENE_DIR}/nexus_seed0"
  local SUM0="${SCENE_DIR}/three_source_seed0.json"

  echo "========== SCENE ${tok} reuse=${reuse} =========="

  if [[ "${reuse}" == "True" || "${reuse}" == "true" ]]; then
    if [[ -f "${SUM0}" ]]; then
      echo "[E2] SKIP reuse_from_le10 ${tok}"
      return 0
    fi
    echo "[E2] WARN reuse flagged but missing summary; will not invent results" >&2
    return 0
  fi

  # degraded le10 retained: already have summary via symlink
  if [[ -f "${SUM0}" ]] && [[ -L "${SCENE_DIR}" ]]; then
    echo "[E2] SKIP retained le10 symlink with existing summary ${tok}"
    return 0
  fi

  # budget
  local new_att
  new_att=$("${PY}" -c "import json; print(json.load(open('${ATTEMPT_STATE}'))['new_restore_attempts'])")
  local total=$((LE10_ATTEMPTS + new_att))
  if (( total >= MAX_ATTEMPTS )); then
    echo "[E2] attempt budget exhausted total=${total}"
    return 2
  fi

  if [[ ! -f "${META}" ]]; then
    echo "[E2] select plan_idx ${tok}"
    "${PY}" "${WE_ROOT}/scripts/select_static_feasible_median_plan_idx.py" \
      --scene-pkl "${SCENE_PKL}" \
      --out-meta "${META}" \
      --work-dir "${SCENE_DIR}/static_gate" \
      --asset-folder "${ASSET_FOLDER}" \
      --vocab "${VOCAB}" \
      --token "${tok}" \
      --bucket "${bucket}" \
      --scene-id "${sid}" || {
        echo "[E2] plan_idx FAILED ${tok}"
        "${PY}" - "${tok}" "${bucket}" "${sid}" "${SUM0}" <<'PY'
import json,sys
from pathlib import Path
tok,bucket,sid,out=sys.argv[1:5]
Path(out).write_text(json.dumps({
  "status":"degraded_plan_idx_failed","token":tok,"bucket":bucket,"scene_id":sid,
  "gate_a":False,"degraded":True,"degraded_reasons":["plan_idx_failed"],"pairwise":{},
  "conditioning_not_comparable_to_smoke_nr_1333":True,
},indent=2,ensure_ascii=False)+"\n")
PY
        return 0
      }
  fi
  local plan_idx
  plan_idx=$("${PY}" -c "import json; print(json.load(open('${META}'))['ego_conditioning']['plan_idx'])")

  # count restore attempt
  "${PY}" - "${ATTEMPT_STATE}" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]); d=json.loads(p.read_text()); d["new_restore_attempts"]=int(d.get("new_restore_attempts",0))+1
p.write_text(json.dumps(d,indent=2)+"\n")
print("new_restore_attempts", d["new_restore_attempts"], "total", 10+d["new_restore_attempts"])
PY

  if [[ ! -f "${RESTORE}/restore_physics_summary.json" ]]; then
    echo "[E2] restore ${tok} plan_idx=${plan_idx}"
    "${PY}" "${WE_ROOT}/scripts/validate_idm_restore_physics.py" \
      --scene-pkl "${SCENE_PKL}" \
      --out-dir "${RESTORE}" \
      --cutoff 4 --horizon 9 \
      --plan-idx "${plan_idx}" \
      --vocab "${VOCAB}" \
      --asset-folder "${ASSET_FOLDER}" \
      --skip-attribution \
      --cold-dir "${RESTORE}/_unused_cold" \
      --warm-dir "${RESTORE}/_unused_warm" || {
        echo "[E2] restore FAILED ${tok}"
        "${PY}" - "${META}" "${RESTORE}" "${tok}" "${bucket}" "${sid}" "${SUM0}" <<'PY'
import json,sys
from pathlib import Path
meta_p,restore,tok,bucket,sid,out=sys.argv[1:7]
meta=json.loads(Path(meta_p).read_text()) if Path(meta_p).exists() else {}
err={}
ep=Path(restore)/"restore_error.json"
if ep.exists(): err=json.loads(ep.read_text())
Path(out).write_text(json.dumps({
  "status":"degraded_restore_failed","token":tok,"bucket":bucket,"scene_id":sid,
  "gate_a":False,"degraded":True,
  "degraded_reasons":["restore_failed", str(err.get("error","unknown"))[:200]],
  "plan_idx_provenance": meta.get("ego_conditioning"),
  "conditioning_not_comparable_to_smoke_nr_1333": True,
  "pairwise":{},
},indent=2,ensure_ascii=False)+"\n")
PY
        return 0
      }
  fi

  # overwrite plan_idx provenance
  "${PY}" - "${META}" "${RESTORE}/plan_idx_resolution.json" <<'PY'
import json,sys
from pathlib import Path
meta=json.loads(Path(sys.argv[1]).read_text()); out=Path(sys.argv[2]); ego=meta.get("ego_conditioning") or {}
out.write_text(json.dumps({
  "plan_idx": ego.get("plan_idx"), "source":"static_feasible_median_path_length",
  "n_feasible": ego.get("n_feasible"), "path_length_m": ego.get("path_length_m"),
  "path_length_percentiles": ego.get("path_length_percentiles"),
  "feasible_plan_idx_sha256": ego.get("feasible_plan_idx_sha256"),
  "path_length_vec_sha256": ego.get("path_length_vec_sha256"),
  "requested_ego_real_hash": ego.get("requested_ego_real_hash"),
  "note": ego.get("note"),
  "equals_legacy_710": ego.get("equals_legacy_710"),
  "equals_smoke_1333": ego.get("equals_smoke_1333"),
},indent=2,ensure_ascii=False)+"\n")
PY

  if [[ ! -f "${RESTORE}/scores_log_replay.pkl" ]]; then
    echo "[E2] no scores (gate C?) degraded stub ${tok}"
    "${PY}" - "${META}" "${RESTORE}" "${tok}" "${bucket}" "${sid}" "${SUM0}" <<'PY'
import json,sys
from pathlib import Path
meta=json.loads(Path(sys.argv[1]).read_text()) if Path(sys.argv[1]).exists() else {}
gate={}
p=Path(sys.argv[2])/"restore_physics_summary.json"
if p.exists(): gate=json.loads(p.read_text())
tr={}
tp=Path(sys.argv[2])/"pre_cutoff_irreversible_transition_flags.json"
if tp.exists(): tr=json.loads(tp.read_text())
n_true=sum(1 for v in tr.values() if v is True)
Path(sys.argv[6]).write_text(json.dumps({
  "status":"degraded_no_scores","token":sys.argv[3],"bucket":sys.argv[4],"scene_id":sys.argv[5],
  "gate_a":False,"degraded":True,
  "degraded_reasons":[f"gate_grade={gate.get('cutoff_gate_grade')}", f"transition_true={n_true}","missing_restore_scores"],
  "plan_idx_provenance": meta.get("ego_conditioning"),
  "conditioning_not_comparable_to_smoke_nr_1333": True,
  "idm_gate":{"grade":gate.get("cutoff_gate_grade"),"pre_cutoff_transition_true_count":n_true,"transition_flags":tr},
  "pairwise":{},
},indent=2,ensure_ascii=False)+"\n")
PY
    return 0
  fi

  if [[ ! -f "${NEXUS0}/future_nexus_seed0_meta.json" ]]; then
    echo "[E2] nexus seed0 ${tok}"
    "${NEXUS_PY}" "${WE_ROOT}/scripts/run_train_side_nexus_generate.py" \
      --scene-pkl "${SCENE_PKL}" --scene-id "${sid}" --out-dir "${NEXUS0}" \
      --plan-idx "${plan_idx}" --seed 0 --vocab "${VOCAB}" --sidecar-root "${SIDECAR}" || {
        echo "[E2] nexus FAILED ${tok}"
        return 0
      }
  fi

  if [[ ! -f "${SUM0}" ]]; then
    echo "[E2] score+compare seed0 ${tok}"
    "${PY}" "${WE_ROOT}/scripts/score_train_side_le10_scene.py" \
      --scene-pkl "${SCENE_PKL}" --scene-id "${sid}" --token "${tok}" --bucket "${bucket}" \
      --restore-dir "${RESTORE}" --nexus-dir "${NEXUS0}" --scene-meta "${META}" \
      --out-summary "${SUM0}" --plan-idx "${plan_idx}" --seed 0 \
      --vocab "${VOCAB}" --asset-folder "${ASSET_FOLDER}" || echo "[E2] score FAILED ${tok}"
  fi
  return 0
}

echo "[E2] planned scenes"
mapfile -t PLANNED < <("${PY}" - "${SCENE_LIST_JSON}" <<'PY'
import json,sys
d=json.loads(open(sys.argv[1]).read())
for s in d["scenes"]:
    print(f"{s['token']}\t{s['bucket']}\t{s['scene_id']}\t{s['reuse_from_le10']}")
PY
)

for line in "${PLANNED[@]}"; do
  IFS=$'\t' read -r tok bucket sid reuse <<<"${line}"
  run_one_scene "${tok}" "${bucket}" "${sid}" "${reuse}" || true
done

count_gate_a() {
  "${PY}" - "${EXTRACT_ROOT}" "${SCENE_LIST_JSON}" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1]); sl=json.loads(Path(sys.argv[2]).read_text())
n=0
for s in sl["scenes"]+sl.get("reserve",[]):
    p=root/s["token"]/"three_source_seed0.json"
    if p.exists() and json.loads(p.read_text()).get("gate_a") is True:
        n+=1
print(n)
PY
}

echo "[E2b] reserve refill if needed"
GATE_A=$(count_gate_a)
echo "[E2b] gate_a=${GATE_A} target_min=${TARGET_A_MIN}"

mapfile -t RESERVE < <("${PY}" - "${SCENE_LIST_JSON}" <<'PY'
import json,sys
d=json.loads(open(sys.argv[1]).read())
for s in d.get("reserve",[]):
    print(f"{s['token']}\t{s['bucket']}\t{s['scene_id']}\tFalse")
PY
)

for line in "${RESERVE[@]}"; do
  GATE_A=$(count_gate_a)
  new_att=$("${PY}" -c "import json; print(json.load(open('${ATTEMPT_STATE}'))['new_restore_attempts'])")
  total=$((LE10_ATTEMPTS + new_att))
  if (( GATE_A >= TARGET_A_MIN )); then
    echo "[E2b] reached gate_a=${GATE_A}; stop refill"
    break
  fi
  if (( total >= MAX_ATTEMPTS )); then
    echo "[E2b] attempt budget exhausted total=${total}"
    break
  fi
  IFS=$'\t' read -r tok bucket sid reuse <<<"${line}"
  # mark as activated in a side list for aggregate
  echo "${tok}" >> "${EXTRACT_ROOT}/_activated_reserve.txt"
  run_one_scene "${tok}" "${bucket}" "${sid}" "False" || true
done

GATE_A=$(count_gate_a)
new_att=$("${PY}" -c "import json; print(json.load(open('${ATTEMPT_STATE}'))['new_restore_attempts'])")
echo "[E2b] final gate_a=${GATE_A} new_attempts=${new_att} total_attempts=$((LE10_ATTEMPTS+new_att))"

echo "[E3] multiseed sample ≥10 (5C+5E prefer; credit 3 existing le10)"
"${PY}" - "${EXTRACT_ROOT}" "${SCENE_LIST_JSON}" "${LE10_ROOT}" <<'PY' > "${EXTRACT_ROOT}/_multiseed_tokens.txt"
import json,sys
from pathlib import Path
root=Path(sys.argv[1]); sl=json.loads(Path(sys.argv[2]).read_text()); le10=Path(sys.argv[3])
# already have seed1 at le10 for these
have={"0008e2e718e15240","028a2f461cfe5f1c","26bf0f9e0f245afe"}
gate_a=[]
for s in sl["scenes"]+sl.get("reserve",[]):
    p=root/s["token"]/"three_source_seed0.json"
    if not p.exists():
        continue
    sm=json.loads(p.read_text())
    if sm.get("gate_a") is True:
        gate_a.append(s)
# ensure have tokens are included if gate_a
selected=[]
for tok in sorted(have):
    for s in gate_a:
        if s["token"]==tok:
            selected.append(s); break
# fill C then E from NEW scenes preferentially for remaining
need_c=5-sum(1 for s in selected if s["bucket"]=="collision")
need_e=5-sum(1 for s in selected if s["bucket"]=="ep")
for bucket,need in [("collision",need_c),("ep",need_e)]:
    cands=[s for s in gate_a if s["bucket"]==bucket and s["token"] not in {x["token"] for x in selected}]
    # prefer not-from-le10
    cands.sort(key=lambda s: (s.get("from_le10",False), s["token"]))
    selected.extend(cands[:max(0,need)])
# if still short, fill any
while len(selected)<10:
    rest=[s for s in gate_a if s["token"] not in {x["token"] for x in selected}]
    if not rest: break
    selected.append(rest[0])
for s in selected[:10]:
    print(s["token"])
print("SELECTED", len(selected[:10]), file=sys.stderr)
PY

while read -r tok; do
  [[ -z "${tok}" ]] && continue
  SCENE_DIR="${EXTRACT_ROOT}/${tok}"
  # resolve real path if symlink
  SCENE_DIR="$(readlink -f "${SCENE_DIR}")"
  META="${SCENE_DIR}/scene_meta.json"
  RESTORE="${SCENE_DIR}/restore"
  SCENE_PKL="${SCENE_DIR}/scene/all_scenarios.pkl"
  plan_idx=$("${PY}" -c "import json; print(json.load(open('${META}'))['ego_conditioning']['plan_idx'])")
  sid=$("${PY}" -c "import json; d=json.load(open('${SCENE_LIST_JSON}'));
ss=d['scenes']+d.get('reserve',[]); print(next(s['scene_id'] for s in ss if s['token']=='${tok}'))")
  bucket=$("${PY}" -c "import json; d=json.load(open('${SCENE_LIST_JSON}'));
ss=d['scenes']+d.get('reserve',[]); print(next(s['bucket'] for s in ss if s['token']=='${tok}'))")
  for seed in 1 2; do
    if [[ -f "${SCENE_DIR}/three_source_seed${seed}.json" ]]; then
      echo "[E3] SKIP existing seed${seed} ${tok}"
      continue
    fi
    NDIR="${SCENE_DIR}/nexus_seed${seed}"
    echo "[E3] nexus seed=${seed} ${tok}"
    "${NEXUS_PY}" "${WE_ROOT}/scripts/run_train_side_nexus_generate.py" \
      --scene-pkl "${SCENE_PKL}" --scene-id "${sid}" --out-dir "${NDIR}" \
      --plan-idx "${plan_idx}" --seed "${seed}" --vocab "${VOCAB}" --sidecar-root "${SIDECAR}"
    echo "[E3] score seed=${seed} ${tok}"
    "${PY}" "${WE_ROOT}/scripts/score_train_side_le10_scene.py" \
      --scene-pkl "${SCENE_PKL}" --scene-id "${sid}" --token "${tok}" --bucket "${bucket}" \
      --restore-dir "${RESTORE}" --nexus-dir "${NDIR}" --scene-meta "${META}" \
      --out-summary "${SCENE_DIR}/three_source_seed${seed}.json" \
      --plan-idx "${plan_idx}" --seed "${seed}" \
      --vocab "${VOCAB}" --asset-folder "${ASSET_FOLDER}"
  done
done < "${EXTRACT_ROOT}/_multiseed_tokens.txt"

echo "[E4] aggregate"
"${PY}" "${WE_ROOT}/scripts/aggregate_train_side_le50.py" \
  --extract-root "${EXTRACT_ROOT}" \
  --scene-list "${SCENE_LIST_JSON}" \
  --attempt-state "${ATTEMPT_STATE}" \
  --out-md "${WE_ROOT}/reports/TRAIN_SIDE_LE50_DISAGREEMENT.md" \
  --out-json "${WE_ROOT}/reports/train_side_le50_summary.json" \
  --out-csv "${WE_ROOT}/reports/train_side_le50_summary.csv"

echo "[E5] handoff"
"${PY}" "${WE_ROOT}/scripts/patch_handoff_train_side_le50.py"

echo "PIPELINE COMPLETE"
