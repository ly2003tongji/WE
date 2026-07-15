#!/usr/bin/env python3
"""Construct hybrid Replay/IDM futures and score them for agent-level attribution."""

from __future__ import annotations

import argparse
import copy
import json
import os
import pickle
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from frozen_state_lib import (  # noqa: E402
    build_input_state_fingerprint,
    extract_scorer_config_from_hydra,
    inject_agent_futures_into_scene,
    load_scene_dict,
    save_json,
    scene_as_dict,
    sha256_bytes,
    sha256_file,
)
from ranking_metrics import (  # noqa: E402
    analyze_max_score_ties,
    assert_ratios_in_unit_interval,
    directional_binary_flips,
    kendall_tau_b,
)


def _rss_gb() -> float:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / (1024.0 ** 2)
    except Exception:
        return -1.0
    return -1.0


def make_hybrid(
    replay: Dict[str, Any],
    idm: Dict[str, Any],
    idm_tokens: List[str],
    name: str,
) -> Dict[str, Any]:
    """Start from Replay futures; replace listed tokens with IDM futures."""
    futures = copy.deepcopy(replay["futures"])
    replaced = []
    for tok in idm_tokens:
        if tok not in idm["futures"]:
            raise KeyError(f"token {tok} missing from IDM futures")
        futures[tok] = copy.deepcopy(idm["futures"][tok])
        replaced.append(tok)
    flat = []
    for token in sorted(futures.keys()):
        f = futures[token]
        flat.append(token.encode("utf-8"))
        flat.append(np.ascontiguousarray(f["position"]).tobytes())
        flat.append(np.ascontiguousarray(f["heading"]).tobytes())
    pack = {
        "source": f"hybrid:{name}",
        "cutoff": replay["cutoff"],
        "horizon": replay["horizon"],
        "frequency_hz": 2.0,
        "dt_s": 0.5,
        "n_agents": len(futures),
        "futures": futures,
        "future_hash": sha256_bytes(b"".join(flat)),
        "hybrid_idm_tokens": replaced,
        "ego_conditioning": replay.get("ego_conditioning"),
        "reaction_mode": "source_conditioned_not_candidate_conditioned",
        "coverage": {
            "idm_tokens_in_hybrid": replaced,
            "n_idm_tokens": len(replaced),
        },
    }
    return pack


def score_hybrid(
    scene: Dict[str, Any],
    futures_pack: Dict[str, Any],
    name: str,
    cutoff: int,
    horizon: int,
    work_dir: Path,
    asset_folder: str,
    vocab_path: Path,
    vocab: np.ndarray,
    plan_idx: int,
    requested_ego_hash: str,
    expected_fp: str,
) -> Dict[str, Any]:
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from worldengine.envs.build_env import build_env
    from worldengine.engine.engine_utils import close_engine, engine_initialized

    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    max_step = cutoff + 2
    out_root = work_dir / "sim_out"
    simengine_root = Path(os.environ["SIMENGINE_ROOT"])
    if engine_initialized():
        close_engine()
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(simengine_root / "worldengine/configs"), version_base="1.2"):
        cfg = compose(
            config_name="default_runner",
            overrides=[
                "debug_mode=True",
                "with_render_manager=false",
                "with_data_manager=false",
                "with_metric_manager=false",
                "with_dense_reward_manager=true",
                "agent_policy=trajectory_policy",
                "agent_navigation=trajectory_navigation",
                "ego_policy=trajectory_policy",
                "ego_navigation=trajectory_navigation",
                "ego_controller=log_play_controller",
                "use_planner_actions=false",
                "distributed_mode=SINGLE_NODE",
                f"data_file_folder_path={work_dir}",
                f"asset_folder_path={asset_folder}",
                "data_pkl_file_name=all_scenarios.pkl",
                f"output_dir={out_root}",
                f"job_name=hybrid_{name}",
                f"max_step={max_step}",
                "enable_resume=false",
                "exit_on_failure=true",
            ],
        )
    hydra_scorer = extract_scorer_config_from_hydra(cfg)
    fp_pre = build_input_state_fingerprint(
        scene,
        cutoff,
        vocab_path,
        plan_idx,
        scorer_config=hydra_scorer,
        horizon=horizon,
        requested_ego_traj_hash=requested_ego_hash,
        vocab=vocab,
        fingerprint_label=f"hybrid_{name}_pre_inject",
    )
    if fp_pre["input_state_fingerprint"] != expected_fp:
        raise RuntimeError(
            f"HARD_PAUSE fingerprint mismatch for hybrid {name}: "
            f"{fp_pre['input_state_fingerprint'][:16]} vs expected {expected_fp[:16]}"
        )

    injected = inject_agent_futures_into_scene(scene, futures_pack, cutoff)
    pkl_path = work_dir / "all_scenarios.pkl"
    with pkl_path.open("wb") as f:
        pickle.dump(scene_as_dict(injected), f)

    data = pickle.load(open(pkl_path, "rb"))
    env = build_env(cfg, name=f"hybrid_{name}", data=data)
    reports = env.run()
    close_engine()
    if not reports or not reports[0].succeeded:
        raise RuntimeError(f"hybrid score failed: {name}")

    pdms_dir = out_root / "openscene_format" / "pdms_pkl"
    target = pdms_dir / f"{scene['id']}_step_{cutoff}_scores.pkl"
    if not target.exists():
        cands = sorted(pdms_dir.glob(f"*_step_{cutoff}_scores.pkl"))
        if not cands:
            raise FileNotFoundError(f"no scores for {name}")
        target = cands[0]
    with target.open("rb") as f:
        scores = pickle.load(f)

    out_pkl = work_dir.parent / f"scores_hybrid_{name}.pkl"
    with out_pkl.open("wb") as f:
        pickle.dump(scores, f)

    return {
        "name": name,
        "scores_path": str(out_pkl),
        "future_hash": futures_pack["future_hash"],
        "hybrid_idm_tokens": futures_pack["hybrid_idm_tokens"],
        "input_state_fingerprint": fp_pre["input_state_fingerprint"],
        "scorer_config_hash": hydra_scorer.get("scorer_config_hash"),
        "scores": scores,
    }


def compare_to_baseline(baseline: Dict[str, Any], other: Dict[str, Any], name: str) -> Dict[str, Any]:
    br = baseline
    ot = other
    noc = directional_binary_flips(br["no_at_fault_collisions"], ot["no_at_fault_collisions"])
    ttc = directional_binary_flips(
        br["time_to_collision_within_bound"], ot["time_to_collision_within_bound"]
    )
    score_b = np.asarray(br["score"], dtype=np.float64).reshape(-1)
    score_o = np.asarray(ot["score"], dtype=np.float64).reshape(-1)
    ties = analyze_max_score_ties(score_b, score_o)
    out = {
        "hybrid": name,
        "noc": {
            "replay_safe_to_hybrid_danger": noc["a_safe_to_b_danger"],
            "replay_danger_to_hybrid_safe": noc["a_danger_to_b_safe"],
            "both_safe": noc["both_safe"],
            "both_danger": noc["both_danger"],
            "total_flip": noc["total_flip"],
        },
        "ttc": {
            "replay_safe_to_hybrid_danger": ttc["a_safe_to_b_danger"],
            "replay_danger_to_hybrid_safe": ttc["a_danger_to_b_safe"],
            "both_safe": ttc["both_safe"],
            "both_danger": ttc["both_danger"],
            "total_flip": ttc["total_flip"],
        },
        "score": {
            "max_abs_diff": float(np.max(np.abs(score_o - score_b))),
            "mean_abs_diff": float(np.mean(np.abs(score_o - score_b))),
            "n_changed": int(np.sum(np.abs(score_o - score_b) > 1e-6)),
            "kendall_tau_b": kendall_tau_b(score_b, score_o),
        },
        "max_score_sets": {
            "baseline_size": ties["side_a_max_set_size"],
            "hybrid_size": ties["side_b_max_set_size"],
            "intersection": ties["intersection_size"],
            "jaccard": ties["jaccard"],
            "frac_baseline_max_in_hybrid_max": ties["frac_a_max_in_b_max"],
            "frac_hybrid_max_in_baseline_max": ties["frac_b_max_in_a_max"],
        },
        "dac_equal": bool(
            np.array_equal(br["drivable_area_compliance"], ot["drivable_area_compliance"])
        ),
        "comfort_equal": bool(np.array_equal(br["comfort"], ot["comfort"])),
        "direction_equal": bool(
            np.array_equal(br["driving_direction_compliance"], ot["driving_direction_compliance"])
        ),
    }
    assert_ratios_in_unit_interval(out)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--futures-dir",
        type=Path,
        default=Path("/mnt/cpfs/prediction/lyyy/myself/WE/data/frozen_paired/smoke1_cutoff3_validity_v2"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/mnt/cpfs/prediction/lyyy/myself/WE/data/frozen_paired/smoke1_cutoff3_attr_v1"),
    )
    parser.add_argument(
        "--scene-pkl",
        type=Path,
        default=Path(
            "/mnt/cpfs/prediction/lyyy/myself/WE/data/smoke_1scene/scenarios/original/navtest_failures/all_scenarios.pkl"
        ),
    )
    parser.add_argument(
        "--vocab",
        type=Path,
        default=Path("/mnt/cpfs/prediction/lyyy/myself/WE/data/hf/data/alg_engine/test_8192_kmeans.npy"),
    )
    parser.add_argument("--cutoff", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=9)
    parser.add_argument("--skip-score-if-exists", action="store_true")
    args = parser.parse_args()

    t0 = time.time()
    we_root = Path(
        os.environ.get(
            "WORLDENGINE_ROOT",
            "/mnt/cpfs/prediction/lyyy/myself/WE/WE/upstream/WorldEngine",
        )
    )
    os.environ.setdefault("WORLDENGINE_ROOT", str(we_root))
    os.environ.setdefault("SIMENGINE_ROOT", str(we_root / "projects/SimEngine"))
    os.environ.setdefault("NUPLAN_MAPS_ROOT", str(we_root / "data/raw/nuplan/dataset/maps"))
    sys.path.insert(0, os.environ["SIMENGINE_ROOT"])
    asset_folder = str(we_root / "data/sim_engine/assets/navtest_failures/assets")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fut_dir = args.futures_dir

    with (fut_dir / "future_log_replay.pkl").open("rb") as f:
        replay = pickle.load(f)
    with (fut_dir / "future_idm.pkl").open("rb") as f:
        idm = pickle.load(f)
    with (fut_dir / "scores_log_replay.pkl").open("rb") as f:
        scores_replay = pickle.load(f)
    with (fut_dir / "scores_idm.pkl").open("rb") as f:
        scores_full = pickle.load(f)

    fp_doc = json.loads((fut_dir / "input_state_fingerprint.json").read_text())
    score_fp = json.loads((fut_dir / "independent_fingerprints_all.json").read_text())[
        "score_replay_pre_inject"
    ]
    plan_idx = int(fp_doc["plan_idx_resolution"]["plan_idx"])
    req_hash = fp_doc["requested_ego_conditioning_hash"]
    idm_tokens = list(idm["coverage"]["idm_rolled"])
    if not idm_tokens:
        raise RuntimeError("No IDM-rolled agents in future_idm coverage")
    # Multi-scene: allow any count; do not hard-require 4.
    print(f"[hybrid] idm_rolled_count={len(idm_tokens)} tokens={idm_tokens}", flush=True)

    scene = load_scene_dict(args.scene_pkl)
    vocab = np.load(args.vocab)

    hybrids_spec = [("baseline_replay", []), ("full_idm", idm_tokens)]
    for tok in idm_tokens:
        short = tok[:8]
        hybrids_spec.append((f"single_{short}", [tok]))
    for tok in idm_tokens:
        short = tok[:8]
        others = [t for t in idm_tokens if t != tok]
        hybrids_spec.append((f"loo_without_{short}", others))

    # Save hybrid future packs (lightweight meta + pkl outside git)
    hybrid_results = {}
    comparisons = {}

    for name, toks in hybrids_spec:
        print(f"[hybrid] {name} idm_tokens={toks}", flush=True)
        if name == "baseline_replay":
            pack = make_hybrid(replay, idm, [], name)
            # use existing scores
            scores = scores_replay
            prov = {
                "name": name,
                "scores_path": str(fut_dir / "scores_log_replay.pkl"),
                "future_hash": pack["future_hash"],
                "hybrid_idm_tokens": [],
                "input_state_fingerprint": score_fp,
                "reused_existing_scores": True,
            }
        elif name == "full_idm":
            pack = make_hybrid(replay, idm, idm_tokens, name)
            scores = scores_full
            prov = {
                "name": name,
                "scores_path": str(fut_dir / "scores_idm.pkl"),
                "future_hash": pack["future_hash"],
                "hybrid_idm_tokens": idm_tokens,
                "input_state_fingerprint": score_fp,
                "reused_existing_scores": True,
            }
        else:
            pack = make_hybrid(replay, idm, toks, name)
            score_pkl = args.out_dir / f"scores_hybrid_{name}.pkl"
            if args.skip_score_if_exists and score_pkl.exists():
                with score_pkl.open("rb") as f:
                    scores = pickle.load(f)
                prov = {
                    "name": name,
                    "scores_path": str(score_pkl),
                    "future_hash": pack["future_hash"],
                    "hybrid_idm_tokens": toks,
                    "input_state_fingerprint": score_fp,
                    "reused_existing_scores": True,
                }
            else:
                scored = score_hybrid(
                    scene=scene,
                    futures_pack=pack,
                    name=name,
                    cutoff=args.cutoff,
                    horizon=args.horizon,
                    work_dir=args.out_dir / f"workdir_{name}",
                    asset_folder=asset_folder,
                    vocab_path=args.vocab,
                    vocab=vocab,
                    plan_idx=plan_idx,
                    requested_ego_hash=req_hash,
                    expected_fp=score_fp,
                )
                scores = scored["scores"]
                prov = {k: v for k, v in scored.items() if k != "scores"}

        with (args.out_dir / f"future_hybrid_{name}.pkl").open("wb") as f:
            pickle.dump(pack, f)
        save_json(
            args.out_dir / f"future_hybrid_{name}_meta.json",
            {k: v for k, v in pack.items() if k != "futures"},
        )
        hybrid_results[name] = prov
        if name != "baseline_replay":
            comparisons[name] = compare_to_baseline(scores_replay, scores, name)
            comparisons[name]["hybrid_idm_tokens"] = prov["hybrid_idm_tokens"]
            print(
                f"  NOC flip {comparisons[name]['noc']['total_flip']} "
                f"(safe→danger {comparisons[name]['noc']['replay_safe_to_hybrid_danger']})",
                flush=True,
            )

    # Attribution table
    full_noc = comparisons["full_idm"]["noc"]["total_flip"]
    attribution_rows = []
    for name, comp in comparisons.items():
        attribution_rows.append(
            {
                "hybrid": name,
                "n_idm_agents": len(comp.get("hybrid_idm_tokens", [])),
                "idm_tokens": comp.get("hybrid_idm_tokens", []),
                "noc_total_flip": comp["noc"]["total_flip"],
                "noc_safe_to_danger": comp["noc"]["replay_safe_to_hybrid_danger"],
                "noc_danger_to_safe": comp["noc"]["replay_danger_to_hybrid_safe"],
                "noc_flip_frac_of_full": (
                    float(comp["noc"]["total_flip"]) / full_noc if full_noc else float("nan")
                ),
                "ttc_total_flip": comp["ttc"]["total_flip"],
                "ttc_safe_to_danger": comp["ttc"]["replay_safe_to_hybrid_danger"],
                "score_mean_abs_diff": comp["score"]["mean_abs_diff"],
                "kendall_tau_b": comp["score"]["kendall_tau_b"],
                "max_set_jaccard_vs_replay": comp["max_score_sets"]["jaccard"],
                "dac_equal": comp["dac_equal"],
                "comfort_equal": comp["comfort_equal"],
                "direction_equal": comp["direction_equal"],
            }
        )

    # Rank single-agent by NOC flip contribution
    singles = [r for r in attribution_rows if r["hybrid"].startswith("single_")]
    singles_sorted = sorted(singles, key=lambda x: -x["noc_total_flip"])

    summary = {
        "engineering_validation_only": True,
        "baseline_fingerprint": score_fp,
        "full_idm_noc_flip": full_noc,
        "idm_tokens": idm_tokens,
        "hybrids": hybrid_results,
        "comparisons": comparisons,
        "attribution_table": attribution_rows,
        "single_agent_ranked_by_noc_flip": singles_sorted,
        "wall_s": round(time.time() - t0, 3),
        "rss_gb": round(_rss_gb(), 3),
    }
    # validate ratios in attribution
    for row in attribution_rows:
        if not np.isnan(row["noc_flip_frac_of_full"]):
            assert 0.0 <= row["noc_flip_frac_of_full"] <= 1.0 + 1e-9

    save_json(args.out_dir / "hybrid_attribution_summary.json", summary)
    # lightweight git-facing copy path decided by caller
    print(json.dumps({"full_noc": full_noc, "singles": singles_sorted}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
