#!/usr/bin/env python3
"""Build frozen-state Replay and IDM traffic futures (no 3DGS, no upstream edits)."""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from frozen_state_lib import (  # noqa: E402
    DEFAULT_CUTOFF,
    DEFAULT_HORIZON,
    DEFAULT_PLAN_IDX,
    DEFAULT_SEED,
    SCORER_CONFIG,
    build_input_state_fingerprint,
    compare_futures,
    extract_replay_futures,
    extract_scorer_config_from_hydra,
    futures_to_numpy_bundle,
    inject_ego_conditioning,
    load_scene_dict,
    resolve_plan_idx,
    save_json,
    scene_as_dict,
    sha256_bytes,
    sha256_file,
    slice_scene_from_cutoff,
    verify_ego_conditioning_execution,
)


def _rss_gb() -> float:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / (1024.0 ** 2)
    except Exception:
        pass
    return -1.0


def _compose_idm_cfg(work_dir: Path, asset_folder: str, horizon: int):
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    simengine_root = Path(os.environ["SIMENGINE_ROOT"])
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(simengine_root / "worldengine/configs"), version_base="1.2"):
        cfg = compose(
            config_name="default_runner",
            overrides=[
                "debug_mode=True",
                "with_render_manager=false",
                "with_data_manager=false",
                "with_metric_manager=false",
                "with_dense_reward_manager=false",
                "agent_policy=idm_policy",
                "agent_navigation=idm_navigation",
                "ego_policy=trajectory_policy",
                "ego_navigation=trajectory_navigation",
                "ego_controller=log_play_controller",
                "use_planner_actions=false",
                "distributed_mode=SINGLE_NODE",
                f"data_file_folder_path={work_dir}",
                f"asset_folder_path={asset_folder}",
                "data_pkl_file_name=all_scenarios.pkl",
                f"output_dir={work_dir / 'out'}",
                "job_name=frozen_idm_future",
                f"max_step={horizon}",
                "enable_resume=false",
                "exit_on_failure=true",
            ],
        )
    return cfg


def run_idm_futures(
    orig_scene: Dict[str, Any],
    cutoff: int,
    horizon: int,
    plan_idx: int,
    vocab: np.ndarray,
    work_dir: Path,
    asset_folder: str,
    seed: int,
) -> Dict[str, Any]:
    """Roll IDM from frozen cutoff with fixed ego conditioning; no render."""
    from worldengine.envs.build_env import build_env
    from worldengine.engine.engine_utils import close_engine, engine_initialized

    truncated = slice_scene_from_cutoff(orig_scene, cutoff, horizon)
    ego_meta = inject_ego_conditioning(truncated, plan_idx, vocab)

    work_dir.mkdir(parents=True, exist_ok=True)
    pkl_path = work_dir / "all_scenarios.pkl"
    with pkl_path.open("wb") as f:
        pickle.dump(scene_as_dict(truncated), f)

    if engine_initialized():
        close_engine()

    cfg = _compose_idm_cfg(work_dir, asset_folder, horizon)
    hydra_scorer = extract_scorer_config_from_hydra(cfg)

    data = pickle.load(open(pkl_path, "rb"))
    env = build_env(cfg, name="frozen_idm_future", data=data)
    env.reset(seed=seed)

    records: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    policy_by_token: Dict[str, Optional[str]] = {}
    nav_by_token: Dict[str, Any] = {}

    def snap(step: int) -> None:
        am = env.engine.agent_manager
        for aid, ag in am.all_agents.items():
            pos = np.asarray(ag.current_position, dtype=np.float64).reshape(-1)
            vel = np.asarray(ag.current_velocity, dtype=np.float64).reshape(-1)
            pname = (
                type(getattr(ag, "policy", None)).__name__
                if getattr(ag, "policy", None) is not None
                else None
            )
            policy_by_token[aid] = pname
            nav_info = {}
            nav = getattr(ag, "navigation", None)
            if nav is not None:
                for attr in ("current_lane", "routing_target_lane", "checkpoint_lanes"):
                    obj = getattr(nav, attr, None)
                    if obj is None:
                        continue
                    if attr == "checkpoint_lanes" and hasattr(obj, "__len__"):
                        nav_info[attr] = [
                            str(getattr(x, "id", getattr(x, "index", type(x).__name__))) for x in list(obj)[:8]
                        ]
                    else:
                        nav_info[attr] = str(getattr(obj, "id", getattr(obj, "index", type(obj).__name__)))
            nav_by_token[aid] = nav_info
            speed = float(np.linalg.norm(vel[:2])) if vel.size else 0.0
            records[aid].append(
                {
                    "step": int(step),
                    "pos": pos.copy(),
                    "heading": float(ag.current_heading),
                    "vel": vel.copy(),
                    "speed": speed,
                    "policy": pname,
                    "navigation": nav_info,
                }
            )

    snap(0)
    for _ in range(horizon - 1):
        env.step(None)
        snap(int(env.engine.episode_step))

    # Add acceleration from speed diffs
    for aid, seq in records.items():
        for i, s in enumerate(seq):
            if i == 0:
                s["acceleration"] = 0.0
            else:
                s["acceleration"] = (s["speed"] - seq[i - 1]["speed"]) / 0.5

    ego_verify = verify_ego_conditioning_execution(
        requested_center=ego_meta["requested_center_traj"],
        requested_heading=ego_meta["requested_heading_traj"],
        live_positions=[s["pos"] for s in records.get("ego", [])],
        live_headings=[s["heading"] for s in records.get("ego", [])],
    )

    replay_ref = extract_replay_futures(orig_scene, cutoff, horizon)
    futures: Dict[str, Any] = {}
    coverage = {
        "idm_rolled": [],
        "static_trajectory_follow": [],
        "log_fallback": [],
        "invalid_at_cutoff": [],
        "missing_live_agent": [],
    }
    fallback_reasons: Dict[str, str] = {}
    idm_step_diagnostics: Dict[str, Any] = {}

    for token, rf in replay_ref["futures"].items():
        ot = orig_scene["object_track"][token]
        typ = ot.get("type")
        if typ != "VEHICLE":
            futures[token] = {**rf, "source": "log_fallback_non_vehicle"}
            coverage["log_fallback"].append(token)
            fallback_reasons[token] = "non_vehicle_not_spawned_by_agent_manager"
            continue
        if token not in records:
            futures[token] = {**rf, "source": "log_fallback_missing_live"}
            coverage["missing_live_agent"].append(token)
            coverage["log_fallback"].append(token)
            fallback_reasons[token] = "vehicle_not_in_live_agents"
            continue
        seq = sorted(records[token], key=lambda x: x["step"])
        pos = np.zeros((horizon, 3), dtype=np.float64)
        heading = np.zeros(horizon, dtype=np.float64)
        vel = np.zeros((horizon, 2), dtype=np.float64)
        valid = np.zeros(horizon, dtype=np.float64)
        step_rows = []
        for s in seq:
            i = int(s["step"])
            if i >= horizon:
                continue
            p = s["pos"]
            pos[i, : min(3, len(p))] = p[: min(3, len(p))]
            heading[i] = s["heading"]
            vv = s["vel"]
            if len(vv) >= 2:
                vel[i, :2] = vv[:2]
            else:
                vel[i, 0] = float(vv[0]) if len(vv) else 0.0
            valid[i] = 1.0
            # replay counterpart
            rpos = np.asarray(rf["position"])
            rvalid = np.asarray(rf.get("valid", np.ones(len(rpos)))).reshape(-1)
            step_rows.append(
                {
                    "step": i,
                    "idm_pos": pos[i, :2].tolist(),
                    "idm_heading": float(heading[i]),
                    "idm_speed": float(s["speed"]),
                    "idm_acceleration": float(s["acceleration"]),
                    "idm_valid": 1.0,
                    "replay_pos": rpos[i, :2].tolist() if i < len(rpos) else None,
                    "replay_valid": float(rvalid[i]) if i < len(rvalid) else 0.0,
                    "policy": s.get("policy"),
                    "navigation": s.get("navigation"),
                }
            )
        pname = policy_by_token.get(token, "")
        if pname == "IDMPolicy":
            src = "idm"
            coverage["idm_rolled"].append(token)
            idm_step_diagnostics[token] = {
                "policy": pname,
                "navigation_at_reset": nav_by_token.get(token),
                "steps": step_rows,
            }
        else:
            src = "static_trajectory_follow"
            coverage["static_trajectory_follow"].append(token)
        futures[token] = {
            "type": typ,
            "position": pos,
            "heading": heading,
            "velocity": vel,
            "valid": valid,
            "source": src,
            "policy": pname,
        }

    flat_parts = []
    for token in sorted(futures.keys()):
        f = futures[token]
        flat_parts.append(token.encode("utf-8"))
        flat_parts.append(np.ascontiguousarray(f["position"]).tobytes())
        flat_parts.append(np.ascontiguousarray(f["heading"]).tobytes())
    future_hash = sha256_bytes(b"".join(flat_parts))

    close_engine()

    # strip heavy arrays from ego_meta for json meta
    ego_meta_light = {
        k: v
        for k, v in ego_meta.items()
        if k not in ("requested_center_traj", "requested_heading_traj")
    }
    ego_meta_light["requested_ego_conditioning_hash"] = ego_meta["requested_ego_conditioning_hash"]
    ego_meta_light["executed_ego_conditioning_hash"] = ego_verify["executed_ego_conditioning_hash"]

    return {
        "source": "idm",
        "cutoff": cutoff,
        "horizon": horizon,
        "frequency_hz": 2.0,
        "dt_s": 0.5,
        "n_agents": len(futures),
        "futures": futures,
        "coverage": coverage,
        "fallback_reasons": fallback_reasons,
        "future_hash": future_hash,
        "ego_conditioning": ego_meta_light,
        "ego_conditioning_full": ego_meta,
        "ego_verification": ego_verify,
        "ego_live_positions": [s["pos"].tolist() for s in records.get("ego", [])],
        "ego_live_headings": [s["heading"] for s in records.get("ego", [])],
        "idm_agent_diagnostics": idm_step_diagnostics,
        "hydra_scorer_config": hydra_scorer,
        "reaction_mode": "source_conditioned_not_candidate_conditioned",
        "seed": seed,
        "notes": [
            "IDM rolls structured vehicle state only; with_render_manager=false.",
            "PEDESTRIAN/CYCLIST are not spawned by BaseAgentManager; log fallback.",
            "Static agents use trajectory_policy internally.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
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
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/mnt/cpfs/prediction/lyyy/myself/WE/data/frozen_paired/smoke1_cutoff3_validity_v2"),
    )
    parser.add_argument(
        "--cutoff",
        type=int,
        default=None,
        help="Frozen cutoff step. Default smoke convenience = 3 (num_history-1).",
    )
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--plan-idx", type=int, default=None, help="Explicit conditioning plan index.")
    parser.add_argument(
        "--plan-idx-csv",
        type=Path,
        default=None,
        help="plan_idx.csv from closed-loop run; used with --plan-idx-step.",
    )
    parser.add_argument(
        "--plan-idx-step",
        type=int,
        default=None,
        help="Step in plan_idx.csv (default cutoff+1 = first action after freeze).",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--asset-folder", type=str, default=None)
    parser.add_argument("--wall-limit-sec", type=int, default=3600)
    parser.add_argument("--rss-limit-gb", type=float, default=64.0)
    args = parser.parse_args()

    cutoff = args.cutoff if args.cutoff is not None else DEFAULT_CUTOFF
    cutoff_source = "cli --cutoff" if args.cutoff is not None else "smoke_default_DEFAULT_CUTOFF(=num_history-1)"

    t0 = time.time()
    we_root = Path(os.environ.get("WORLDENGINE_ROOT", "/mnt/cpfs/prediction/lyyy/myself/WE/WE/upstream/WorldEngine"))
    os.environ.setdefault("WORLDENGINE_ROOT", str(we_root))
    os.environ.setdefault("SIMENGINE_ROOT", str(we_root / "projects/SimEngine"))
    os.environ.setdefault("NUPLAN_MAPS_ROOT", str(we_root / "data/raw/nuplan/dataset/maps"))
    sys.path.insert(0, os.environ["SIMENGINE_ROOT"])

    asset_folder = args.asset_folder or str(we_root / "data/sim_engine/assets/navtest_failures/assets")
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    plan_res = resolve_plan_idx(args.plan_idx, args.plan_idx_csv, args.plan_idx_step, cutoff)
    plan_idx = int(plan_res["plan_idx"])

    scene = load_scene_dict(args.scene_pkl)
    vocab = np.load(args.vocab)

    # Requested ego traj hash from freeze pose (without mutating scoring scene)
    ego_probe = slice_scene_from_cutoff(scene, cutoff, args.horizon)
    ego_req = inject_ego_conditioning(ego_probe, plan_idx, vocab)
    req_hash = ego_req["requested_ego_conditioning_hash"]

    def gate(msg: str) -> None:
        raise RuntimeError(f"HARD_PAUSE: {msg}")

    # --- Independent fingerprint 1: before Replay future extraction ---
    fp_replay_build = build_input_state_fingerprint(
        scene,
        cutoff,
        args.vocab,
        plan_idx,
        scorer_config=dict(SCORER_CONFIG),
        horizon=args.horizon,
        requested_ego_traj_hash=req_hash,
        vocab=vocab,
        fingerprint_label="replay_future_build_pre",
    )

    replay = extract_replay_futures(scene, cutoff, args.horizon)
    replay["ego_conditioning"] = {
        k: v for k, v in ego_req.items() if k not in ("requested_center_traj", "requested_heading_traj")
    }
    replay["reaction_mode"] = "source_conditioned_not_candidate_conditioned"
    replay["note"] = "Replay does not react to ego; same requested ego conditioning recorded."
    replay["input_state_fingerprint"] = fp_replay_build["input_state_fingerprint"]

    if time.time() - t0 > args.wall_limit_sec:
        gate("wall clock exceeded before IDM")
    if _rss_gb() > args.rss_limit_gb:
        gate(f"RSS {_rss_gb():.1f}GB exceeded limit")

    # --- Independent fingerprint 2: before IDM env init (original freeze, separate call) ---
    fp_idm_build = build_input_state_fingerprint(
        scene,
        cutoff,
        args.vocab,
        plan_idx,
        scorer_config=dict(SCORER_CONFIG),
        horizon=args.horizon,
        requested_ego_traj_hash=req_hash,
        vocab=vocab,
        fingerprint_label="idm_env_init_pre",
    )

    try:
        idm = run_idm_futures(
            scene,
            cutoff,
            args.horizon,
            plan_idx,
            vocab,
            work_dir=out_dir / "idm_workdir",
            asset_folder=asset_folder,
            seed=args.seed,
        )
    except Exception as e:
        save_json(out_dir / "idm_error.json", {"error": str(e), "trace": traceback.format_exc()})
        print("IDM future generation failed:", e)
        print(traceback.format_exc())
        return 2

    # Replace placeholder scorer config on IDM path with actual hydra extract
    hydra_scorer = idm["hydra_scorer_config"]
    fp_idm_build_actual = build_input_state_fingerprint(
        scene,
        cutoff,
        args.vocab,
        plan_idx,
        scorer_config=hydra_scorer,
        horizon=args.horizon,
        requested_ego_traj_hash=req_hash,
        vocab=vocab,
        fingerprint_label="idm_env_init_pre_hydra_scorer",
    )
    # Rebuild replay fingerprint with same hydra-derived scorer fields for apples-to-apples
    # (IDM build itself doesn't use densereward; keep both documented)
    fp_replay_build_hydra = build_input_state_fingerprint(
        scene,
        cutoff,
        args.vocab,
        plan_idx,
        scorer_config=hydra_scorer,
        horizon=args.horizon,
        requested_ego_traj_hash=req_hash,
        vocab=vocab,
        fingerprint_label="replay_future_build_pre_hydra_scorer",
    )

    ego_v = idm["ego_verification"]
    save_json(out_dir / "ego_conditioning_verification.json", ego_v)
    if ego_v["warnings"]:
        print("WARNING ego conditioning:", ego_v["warnings"])
    if ego_v["hard_fail"]:
        save_json(out_dir / "HARD_PAUSE_ego_conditioning.json", ego_v)
        print("HARD_PAUSE: executed ego trajectory diverges from requested conditioning")
        print(
            "requested_hash",
            ego_v["requested_ego_conditioning_hash"],
            "executed_hash",
            ego_v["executed_ego_conditioning_hash"],
        )
        print("IDM reacted to EXECUTED trajectory (see ego_conditioning_verification.json)")
        return 3

    idm["input_state_fingerprint"] = fp_idm_build_actual["input_state_fingerprint"]
    replay["input_state_fingerprint"] = fp_replay_build_hydra["input_state_fingerprint"]

    if replay["ego_conditioning"]["requested_ego_conditioning_hash"] != idm["ego_conditioning"][
        "requested_ego_conditioning_hash"
    ]:
        gate("requested ego conditioning hash mismatch between Replay and IDM paths")

    r_cov = set(replay["coverage"]["idm_capable_vehicle"])
    i_rolled = set(idm["coverage"]["idm_rolled"])
    if not i_rolled.issubset(r_cov):
        gate(f"IDM rolled tokens not subset of replay vehicles: {sorted(i_rolled - r_cov)}")

    fut_cmp = compare_futures(replay, idm)

    # Per-agent diagnostics for IDM-rolled (enrich with ADE)
    ade_by_tok = {e["token"]: e for e in fut_cmp.get("per_agent", [])}
    for tok, diag in idm.get("idm_agent_diagnostics", {}).items():
        if tok in ade_by_tok:
            diag["ade"] = {
                k: ade_by_tok[tok][k]
                for k in (
                    "replay_valid_count",
                    "idm_valid_count",
                    "common_valid_count",
                    "common_valid_mean_ade_m",
                    "common_valid_max_ade_m",
                    "unmasked_mean_ade_m",
                    "unmasked_max_ade_m",
                    "first_fork_step_common_valid",
                    "first_fork_step_unmasked",
                )
            }
            # jump detection
            jumps = []
            for i in range(1, len(diag["steps"])):
                a = np.asarray(diag["steps"][i - 1]["idm_pos"])
                b = np.asarray(diag["steps"][i]["idm_pos"])
                dist = float(np.linalg.norm(b - a))
                if dist > 15.0:  # >15m in 0.5s ~ 30m/s implausible jump flag
                    jumps.append({"from_step": i - 1, "to_step": i, "step_dist_m": dist})
            diag["large_step_jumps"] = jumps

    save_json(out_dir / "idm_agent_diagnostics.json", idm.get("idm_agent_diagnostics", {}))

    fingerprints = {
        "replay_future_build_pre": fp_replay_build["input_state_fingerprint"],
        "idm_env_init_pre_constant_scorer": fp_idm_build["input_state_fingerprint"],
        "replay_future_build_pre_hydra_scorer": fp_replay_build_hydra["input_state_fingerprint"],
        "idm_env_init_pre_hydra_scorer": fp_idm_build_actual["input_state_fingerprint"],
        "all_four_equal_with_matching_scorer_block": (
            fp_replay_build_hydra["input_state_fingerprint"] == fp_idm_build_actual["input_state_fingerprint"]
        ),
        "note": (
            "Four independent build_input_state_fingerprint() calls. "
            "Scoring-stage fingerprints are computed later in score_frozen_disagreement.py "
            "before future injection."
        ),
        "hydra_scorer_config": hydra_scorer,
        "plan_idx_resolution": plan_res,
        "cutoff": cutoff,
        "cutoff_source": cutoff_source,
        "requested_ego_traj_hash": req_hash,
    }
    save_json(out_dir / "independent_fingerprints_build.json", fingerprints)
    save_json(
        out_dir / "input_state_fingerprint.json",
        {
            "input_state_fingerprint": fp_replay_build_hydra["input_state_fingerprint"],
            "independent_fingerprints": fingerprints,
            "cutoff": cutoff,
            "cutoff_source": cutoff_source,
            "plan_idx_resolution": plan_res,
            "vocab_sha256": sha256_file(args.vocab),
            "scorer_config": hydra_scorer,
            "requested_ego_conditioning_hash": req_hash,
            "executed_ego_conditioning_hash": ego_v["executed_ego_conditioning_hash"],
            "note": "Fingerprint excludes traffic-model futures.",
        },
    )
    save_json(out_dir / "input_state_payload_replay_build.json", fp_replay_build_hydra["payload"])

    for name, pack in (("log_replay", replay), ("idm", idm)):
        meta = {
            k: v
            for k, v in pack.items()
            if k
            not in (
                "futures",
                "ego_conditioning_full",
                "idm_agent_diagnostics",
            )
        }
        # drop large per-step from meta ego verify duplicate
        save_json(out_dir / f"future_{name}_meta.json", meta)
        np.savez_compressed(out_dir / f"future_{name}.npz", **futures_to_numpy_bundle(pack))
        with (out_dir / f"future_{name}.pkl").open("wb") as f:
            # strip non-pickle-friendly huge optional keys already ok
            dump = {k: v for k, v in pack.items() if k != "ego_conditioning_full"}
            if "ego_conditioning" in dump and "requested_center_traj" in dump.get("ego_conditioning", {}):
                dump["ego_conditioning"] = {
                    kk: vv
                    for kk, vv in dump["ego_conditioning"].items()
                    if kk not in ("requested_center_traj", "requested_heading_traj")
                }
            pickle.dump(dump, f)

    save_json(out_dir / "future_compare.json", fut_cmp)

    summary = {
        "status": "ok",
        "wall_s": round(time.time() - t0, 3),
        "rss_gb": round(_rss_gb(), 3),
        "cutoff": cutoff,
        "cutoff_source": cutoff_source,
        "plan_idx_resolution": plan_res,
        "input_state_fingerprint": fp_replay_build_hydra["input_state_fingerprint"],
        "independent_fingerprints_equal": fingerprints["all_four_equal_with_matching_scorer_block"],
        "requested_ego_conditioning_hash": req_hash,
        "executed_ego_conditioning_hash": ego_v["executed_ego_conditioning_hash"],
        "ego_pos_error_max_m": ego_v["pos_error_max_m"],
        "ego_heading_error_max_deg": ego_v["heading_error_max_deg"],
        "replay_future_hash": replay["future_hash"],
        "idm_future_hash": idm["future_hash"],
        "futures_differ": not fut_cmp["futures_equal"],
        "n_changed_agents_common_valid": fut_cmp["n_changed_agents_common_valid_maxade_gt_5cm"],
        "traj_dev_common_valid_mean_m": fut_cmp["traj_dev_common_valid_mean_m"],
        "traj_dev_common_valid_max_m": fut_cmp["traj_dev_common_valid_max_m"],
        "legacy_unmasked_max_ade_m": fut_cmp["legacy_unmasked"]["traj_dev_max_m"],
        "artifact_diagnosis": fut_cmp.get("artifact_diagnosis"),
        "idm_rolled": len(idm["coverage"]["idm_rolled"]),
        "log_fallback": len(idm["coverage"]["log_fallback"]),
        "reaction_mode": "source_conditioned_not_candidate_conditioned",
    }
    save_json(out_dir / "build_summary.json", summary)
    import json as _json

    print(_json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
