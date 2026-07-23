#!/usr/bin/env python3
"""Score Nexus + pairwise Replay/IDM/Nexus for one train-side scene; write per-scene summary."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np

WE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WE_ROOT))
sys.path.insert(0, str(WE_ROOT / "scripts"))

from frozen_state_lib import (  # noqa: E402
    DEFAULT_CUTOFF,
    DEFAULT_HORIZON,
    inject_ego_conditioning_at_cutoff,
    load_scene_dict,
    save_json,
)
from ranking_metrics import (  # noqa: E402
    analyze_max_score_ties,
    directional_binary_flips,
    index_tie_broken_topk_overlap,
    kendall_tau_b,
)
from run_cutoff4_three_source_smoke import PDM_KEYS, _pairwise, _rss_gb  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-pkl", type=Path, required=True)
    ap.add_argument("--scene-id", type=str, required=True)
    ap.add_argument("--token", type=str, required=True)
    ap.add_argument("--bucket", type=str, required=True)
    ap.add_argument("--restore-dir", type=Path, required=True)
    ap.add_argument("--nexus-dir", type=Path, required=True)
    ap.add_argument("--scene-meta", type=Path, required=True)
    ap.add_argument("--out-summary", type=Path, required=True)
    ap.add_argument("--plan-idx", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--vocab",
        type=Path,
        default=Path("/mnt/cpfs/prediction/lyyy/myself/WE/data/hf/data/alg_engine/test_8192_kmeans.npy"),
    )
    ap.add_argument("--asset-folder", type=str, required=True)
    ap.add_argument("--cutoff", type=int, default=DEFAULT_CUTOFF)
    ap.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    args = ap.parse_args()

    from score_frozen_disagreement import score_one_source

    we_root = WE_ROOT / "upstream/WorldEngine"
    os.environ.setdefault("WORLDENGINE_ROOT", str(we_root))
    os.environ.setdefault("SIMENGINE_ROOT", str(we_root / "projects/SimEngine"))
    maps = Path("/mnt/cpfs/prediction/lyyy/myself/WE/data/maps/extracted")
    if maps.exists():
        os.environ["NUPLAN_MAPS_ROOT"] = str(maps)
    if os.environ["SIMENGINE_ROOT"] not in sys.path:
        sys.path.insert(0, os.environ["SIMENGINE_ROOT"])

    scene = load_scene_dict(args.scene_pkl)
    vocab = np.load(args.vocab)
    cutoff = int(args.cutoff)
    horizon = int(args.horizon)
    plan_idx = int(args.plan_idx)
    seed = int(args.seed)
    scene_meta = json.loads(args.scene_meta.read_text())

    nexus_pkl = args.nexus_dir / f"future_nexus_plan{plan_idx}_seed{seed}.pkl"
    nexus_meta_path = args.nexus_dir / f"future_nexus_plan{plan_idx}_seed{seed}_meta.json"
    if not nexus_pkl.exists() and seed == 0:
        nexus_pkl = args.nexus_dir / "future_nexus_seed0.pkl"
        nexus_meta_path = args.nexus_dir / "future_nexus_seed0_meta.json"
    with nexus_pkl.open("rb") as f:
        nexus_pack = pickle.load(f)
    nexus_meta = json.loads(nexus_meta_path.read_text())

    restore_dir = args.restore_dir
    scene_scored = copy.deepcopy(scene)
    ego_meta = inject_ego_conditioning_at_cutoff(scene_scored, cutoff, plan_idx, vocab)
    req_hash = ego_meta["requested_ego_conditioning_hash"]
    nexus_pack = dict(nexus_pack)
    nexus_pack["ego_conditioning"] = {
        k: v for k, v in ego_meta.items() if k not in ("requested_center_traj", "requested_heading_traj")
    }

    t0 = time.time()
    score_out = args.nexus_dir / f"score_seed{seed}"
    score_out.mkdir(parents=True, exist_ok=True)
    # score_one_source writes beside work_dir parent conventionally; use dedicated out
    # Patch: score writes to out_dir parent of work_dir named scores_<source>.pkl via frozen helper.
    # Use nexus_dir as parent by setting work_dir = score_out / "workdir"
    # Actually score_frozen writes to work_dir.parent / f"scores_{source_name}.pkl"
    work = score_out / "workdir"
    prov = score_one_source(
        scene=scene,
        futures_pack=nexus_pack,
        source_name=f"nexus_seed{seed}",
        cutoff=cutoff,
        horizon=horizon,
        work_dir=work,
        asset_folder=args.asset_folder,
        vocab_path=args.vocab,
        vocab=vocab,
        plan_idx=plan_idx,
        requested_ego_hash=req_hash,
        wall_limit_sec=3600,
        rss_limit_gb=64.0,
        t0=t0,
    )
    scores_path = score_out / f"scores_nexus_seed{seed}.pkl"
    cand = score_out / f"scores_nexus_seed{seed}.pkl"
    # score_one_source places at work_dir.parent
    produced = score_out / f"scores_nexus_seed{seed}.pkl"
    if not produced.exists():
        # fallthrough name
        alt = list(score_out.glob("scores_*.pkl"))
        if not alt:
            raise RuntimeError(f"missing nexus scores under {score_out}; prov={prov}")
        produced = alt[0]
    with produced.open("rb") as f:
        scores_nexus = pickle.load(f)
    save_json(score_out / f"scores_nexus_seed{seed}_run_meta.json", prov)

    with (restore_dir / "scores_log_replay.pkl").open("rb") as f:
        scores_replay = pickle.load(f)
    with (restore_dir / "scores_idm_restored.pkl").open("rb") as f:
        scores_idm = pickle.load(f)

    fp_restore = json.loads((restore_dir / "fingerprint_cross_check.json").read_text())
    fp_replay = fp_restore.get("replay_run_fingerprint")
    fp_idm = fp_restore.get("restored_run_fingerprint")
    fp_nexus = prov["input_state_fingerprint"]
    fingerprint_aligned = bool(fp_replay == fp_idm == fp_nexus)

    pairs = {
        "replay_vs_idm": _pairwise(scores_replay, scores_idm, "replay", "idm"),
        "replay_vs_nexus": _pairwise(scores_replay, scores_nexus, "replay", "nexus"),
        "idm_vs_nexus": _pairwise(scores_idm, scores_nexus, "idm", "nexus"),
    }

    restore_summary = {}
    rs_path = restore_dir / "restore_physics_summary.json"
    if rs_path.exists():
        restore_summary = json.loads(rs_path.read_text())
    transition = {}
    tr_path = restore_dir / "pre_cutoff_irreversible_transition_flags.json"
    if tr_path.exists():
        transition = json.loads(tr_path.read_text())
    n_true = sum(1 for v in transition.values() if v is True)

    gate_grade = restore_summary.get("cutoff_gate_grade")
    allow = bool(restore_summary.get("cutoff_gate_allow_reward_compare"))
    gate_a = gate_grade == "A" and allow and n_true == 0
    degraded = not gate_a
    degraded_reasons = []
    if gate_grade != "A":
        degraded_reasons.append(f"gate_grade={gate_grade}")
    if not allow:
        degraded_reasons.append("allow_reward_compare=False")
    if n_true > 0:
        degraded_reasons.append(f"pre_cutoff_transition_true={n_true}")

    # difficulty proxies
    n_agents = int(scores_replay.get("n_agents") or len(scene.get("object_track", {})))
    try:
        ttc = np.asarray(scores_replay["time_to_collision_within_bound"], dtype=np.float64).reshape(-1)
        score_r = np.asarray(scores_replay["score"], dtype=np.float64).reshape(-1)
        difficulty = {
            "n_object_tracks": len(scene.get("object_track", {})),
            "n_agents_scored_hint": n_agents,
            "replay_ttc_mean": float(np.mean(ttc)),
            "replay_ttc_min": float(np.min(ttc)),
            "replay_score_mean": float(np.mean(score_r)),
            "replay_score_min": float(np.min(score_r)),
            "replay_score_p10": float(np.percentile(score_r, 10)),
        }
    except Exception as e:
        difficulty = {"error": str(e)}

    summary: Dict[str, Any] = {
        "status": "ok",
        "research_claim_boundary": "preliminary_effect_size_only",
        "token": args.token,
        "bucket": args.bucket,
        "scene_id": args.scene_id,
        "cutoff": cutoff,
        "horizon": horizon,
        "seed": seed,
        "gate_a": gate_a,
        "degraded": degraded,
        "degraded_reasons": degraded_reasons,
        "enter_main_effect_table": gate_a and seed == 0,
        "plan_idx_provenance": scene_meta.get("ego_conditioning"),
        "conditioning_not_comparable_to_smoke_nr_1333": True,
        "idm_gate": {
            "grade": gate_grade,
            "allow_reward_compare": allow,
            "pre_cutoff_transition_true_count": n_true,
            "pre_cutoff_transition_any_true": n_true > 0,
            "transition_flags": transition,
        },
        "fingerprint": {
            "replay": fp_replay,
            "idm_restored": fp_idm,
            "nexus": fp_nexus,
            "aligned": fingerprint_aligned,
        },
        "nexus": {
            "seed": seed,
            "noise_hash": nexus_meta.get("noise_hash"),
            "physics_log_fallback_tokens": nexus_meta.get("physics_log_fallback_tokens", []),
            "coverage_counts": nexus_meta.get("coverage_counts"),
            "hold_ok": nexus_meta.get("hold_ok"),
            "future_hash": nexus_meta.get("future_hash"),
        },
        "pairwise": {
            k: {
                "noc_safe_to_danger": v["noc_flips"]["a_safe_to_b_danger"],
                "noc_danger_to_safe": v["noc_flips"]["a_danger_to_b_safe"],
                "noc_total_flip": v["noc_flips"]["total_flip"],
                "ttc_safe_to_danger": v["ttc_flips"]["a_safe_to_b_danger"],
                "ttc_danger_to_safe": v["ttc_flips"]["a_danger_to_b_safe"],
                "ttc_total_flip": v["ttc_flips"]["total_flip"],
                "kendall_tau_b": v["kendall_tau_b"],
                "topk_overlap": v["topk_overlap_index_tie_broken"],
            }
            for k, v in pairs.items()
        },
        "difficulty": difficulty,
        "ego_conditioning_hash": {
            "requested": req_hash,
            "match_scene_meta_note": scene_meta.get("ego_conditioning", {}).get("note"),
        },
        "paths": {
            "restore_dir": str(restore_dir),
            "nexus_dir": str(args.nexus_dir),
            "scores_nexus": str(produced),
        },
        "wall_s": round(time.time() - t0, 3),
        "rss_gb": round(_rss_gb(), 3),
    }
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    save_json(args.out_summary, summary)

    # tiny csv for this scene seed0
    if seed == 0:
        rows = []
        for pair_name, v in summary["pairwise"].items():
            rows.append(
                {
                    "token": args.token,
                    "bucket": args.bucket,
                    "gate_a": gate_a,
                    "degraded": degraded,
                    "plan_idx": plan_idx,
                    "pair": pair_name,
                    "noc_total_flip": v["noc_total_flip"],
                    "ttc_total_flip": v["ttc_total_flip"],
                    "kendall_tau_b": v["kendall_tau_b"],
                }
            )
        csv_path = args.out_summary.with_suffix(".csv")
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(json.dumps({"token": args.token, "gate_a": gate_a, "degraded": degraded, "out": str(args.out_summary)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
