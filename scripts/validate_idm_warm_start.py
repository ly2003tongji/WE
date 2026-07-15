#!/usr/bin/env python3
"""Stage 1.8: warm-start IDM continuity vs cold-start truncation artifact check.

Uses full original scene (not slice_scene_from_cutoff). No upstream edits.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from frozen_state_lib import (  # noqa: E402
    CUTOFF_GATE_TOLERANCES,
    DEFAULT_CUTOFF,
    DEFAULT_HORIZON,
    DEFAULT_SEED,
    capture_live_agent_nav,
    classify_cutoff_gate,
    compare_futures,
    extract_log_agent_state_at,
    extract_replay_futures,
    extract_scorer_config_from_hydra,
    futures_to_numpy_bundle,
    inject_ego_conditioning_at_cutoff,
    live_agents_to_cutoff_state,
    load_scene_dict,
    resolve_plan_idx,
    save_json,
    scene_as_dict,
    sha256_bytes,
    truncate_scene_prefix,
    verify_ego_conditioning_execution,
)
from ranking_metrics import (  # noqa: E402
    analyze_max_score_ties,
    directional_binary_flips,
    kendall_tau_b,
)

FOCUS_TOKEN = "44df645d1b5b584b"
PDM_KEYS = [
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "time_to_collision_within_bound",
    "comfort",
    "ego_progress",
    "driving_direction_compliance",
    "score",
]


def _rss_gb() -> float:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / (1024.0 ** 2)
    except Exception:
        pass
    return -1.0


def _compose_warm_cfg(work_dir: Path, asset_folder: str, max_step: int):
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
                "job_name=frozen_idm_warm_start",
                f"max_step={max_step}",
                "enable_resume=false",
                "exit_on_failure=true",
            ],
        )
    return cfg


def _snap_agent(ag: Any, step: int) -> Dict[str, Any]:
    pos = np.asarray(ag.current_position, dtype=np.float64).reshape(-1)
    vel = np.asarray(ag.current_velocity, dtype=np.float64).reshape(-1)
    speed = float(np.linalg.norm(vel[:2])) if vel.size else 0.0
    pname = type(getattr(ag, "policy", None)).__name__ if getattr(ag, "policy", None) is not None else None
    return {
        "step": int(step),
        "pos": pos.copy(),
        "heading": float(ag.current_heading),
        "vel": vel.copy(),
        "speed": speed,
        "policy": pname,
        "navigation": capture_live_agent_nav(ag),
    }


def detect_oscillation(speeds: List[float], positions: np.ndarray) -> Dict[str, Any]:
    sp = np.asarray(speeds, dtype=np.float64)
    net = float(np.linalg.norm(positions[-1, :2] - positions[0, :2])) if len(positions) >= 2 else 0.0
    path = 0.0
    for i in range(1, len(positions)):
        path += float(np.linalg.norm(positions[i, :2] - positions[i - 1, :2]))
    # near-stationary oscillation: low net travel, alternating high/low speed
    alt = 0
    for i in range(2, len(sp)):
        if (sp[i] - sp[i - 1]) * (sp[i - 1] - sp[i - 2]) < 0:
            alt += 1
    osc = bool(net < 3.0 and path > net * 1.5 and alt >= 2 and float(np.max(sp)) > 0.5)
    complete_stall = bool(net < 0.5 and path < 0.5 and (len(sp) == 0 or float(np.max(sp)) < 0.1))
    return {
        "near_stationary_oscillation": osc,
        "complete_stall": complete_stall,
        "net_displacement_m": net,
        "path_length_m": path,
        "speed_alternations": int(alt),
        "speed_min": float(np.min(sp)) if len(sp) else None,
        "speed_max": float(np.max(sp)) if len(sp) else None,
        "speed_mean": float(np.mean(sp)) if len(sp) else None,
    }


def run_warm_start_idm(
    orig_scene: Dict[str, Any],
    cutoff: int,
    horizon: int,
    plan_idx: int,
    vocab: np.ndarray,
    work_dir: Path,
    asset_folder: str,
    seed: int,
) -> Dict[str, Any]:
    from worldengine.envs.build_env import build_env
    from worldengine.engine.engine_utils import close_engine, engine_initialized

    length = cutoff + horizon
    warm_scene = truncate_scene_prefix(orig_scene, length)
    ego_meta = inject_ego_conditioning_at_cutoff(warm_scene, cutoff, plan_idx, vocab)

    work_dir.mkdir(parents=True, exist_ok=True)
    pkl_path = work_dir / "all_scenarios.pkl"
    with pkl_path.open("wb") as f:
        pickle.dump(scene_as_dict(warm_scene), f)

    if engine_initialized():
        close_engine()

    max_step = length - 1
    cfg = _compose_warm_cfg(work_dir, asset_folder, max_step)
    hydra_scorer = extract_scorer_config_from_hydra(cfg)

    data = pickle.load(open(pkl_path, "rb"))
    env = build_env(cfg, name="frozen_idm_warm_start", data=data)
    env.reset(seed=seed)

    records: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    idm_start_step: Optional[int] = None
    pre_cutoff_vs_log: List[Dict[str, Any]] = []

    def snap_all(step: int) -> None:
        nonlocal idm_start_step
        am = env.engine.agent_manager
        for aid, ag in am.all_agents.items():
            row = _snap_agent(ag, step)
            records[aid].append(row)
            if row["policy"] == "IDMPolicy" and idm_start_step is None:
                idm_start_step = int(step)

    snap_all(0)
    # Pre-cutoff: compare live vs log each step to find first divergence
    cold_log_states = {
        s: extract_log_agent_state_at(orig_scene, s, include_ego=True) for s in range(cutoff + 1)
    }

    def record_vs_log(step: int) -> None:
        live = live_agents_to_cutoff_state(
            env,
            step,
            map_name=orig_scene.get("map"),
            traffic_lights=cold_log_states[step]["traffic_lights"],
        )
        # attach types from log for size compare — sizes from live agents already
        gate = classify_cutoff_gate(cold_log_states[step], live)
        # fill missing sizes from log for agents present in both
        pre_cutoff_vs_log.append(
            {
                "step": step,
                "grade": gate["grade"],
                "n_issues": len(gate["issues"]),
                "max_pos_err_m": max((a["pos_err_m"] for a in gate["per_agent"].values()), default=0.0),
                "worst_agents": sorted(
                    (
                        {"token": t, "pos_err_m": a["pos_err_m"], "heading_err_deg": a["heading_err_deg"]}
                        for t, a in gate["per_agent"].items()
                    ),
                    key=lambda x: -x["pos_err_m"],
                )[:5],
            }
        )

    record_vs_log(0)
    for _ in range(max_step):
        env.step(None)
        step = int(env.engine.episode_step)
        snap_all(step)
        if step <= cutoff:
            record_vs_log(step)

    # Cutoff gate: cold freeze log vs warm live at cutoff (from records)
    agents_at_cut = {}
    for aid, seq in records.items():
        rows = [r for r in seq if r["step"] == cutoff]
        if not rows:
            continue
        r = rows[0]
        length_v = 0.0
        width_v = 0.0
        typ = None
        if aid in orig_scene["object_track"]:
            ot = orig_scene["object_track"][aid]
            typ = ot.get("type")
            length_v = float(np.asarray(ot["state"]["length"])[cutoff, 0])
            width_v = float(np.asarray(ot["state"]["width"])[cutoff, 0])
        agents_at_cut[aid] = {
            "type": typ,
            "pos_xy": r["pos"][:2].tolist(),
            "heading": r["heading"],
            "speed": r["speed"],
            "valid": True,
            "length": length_v,
            "width": width_v,
            "policy": r["policy"],
            "navigation": r["navigation"],
        }
    warm_at_cutoff = {
        "step": cutoff,
        "map": orig_scene.get("map"),
        "agent_tokens_sorted": sorted(agents_at_cut.keys()),
        "agents": agents_at_cut,
        "traffic_lights": cold_log_states[cutoff]["traffic_lights"],
    }
    cold_at_cutoff = extract_log_agent_state_at(orig_scene, cutoff, include_ego=True)
    cutoff_gate = classify_cutoff_gate(cold_at_cutoff, warm_at_cutoff)

    first_div = None
    for row in pre_cutoff_vs_log:
        if row["grade"] != "A":
            first_div = row
            break

    # Post-cutoff ego verification
    ego_seq = sorted(records.get("ego", []), key=lambda x: x["step"])
    ego_post = [r for r in ego_seq if r["step"] >= cutoff]
    ego_verify = verify_ego_conditioning_execution(
        requested_center=ego_meta["requested_center_traj"],
        requested_heading=ego_meta["requested_heading_traj"],
        live_positions=[r["pos"] for r in ego_post],
        live_headings=[r["heading"] for r in ego_post],
    )

    # Build post-cutoff futures (horizon frames starting at cutoff)
    replay_ref = extract_replay_futures(orig_scene, cutoff, horizon)
    futures: Dict[str, Any] = {}
    coverage = {
        "idm_rolled": [],
        "static_trajectory_follow": [],
        "log_fallback": [],
        "missing_live_agent": [],
    }
    idm_diag: Dict[str, Any] = {}
    fallback_reasons: Dict[str, str] = {}

    for token, rf in replay_ref["futures"].items():
        ot = orig_scene["object_track"][token]
        typ = ot.get("type")
        if typ != "VEHICLE":
            futures[token] = {**rf, "source": "log_fallback_non_vehicle"}
            coverage["log_fallback"].append(token)
            fallback_reasons[token] = "non_vehicle"
            continue
        if token not in records:
            futures[token] = {**rf, "source": "log_fallback_missing_live"}
            coverage["missing_live_agent"].append(token)
            coverage["log_fallback"].append(token)
            fallback_reasons[token] = "not_in_live_agents"
            continue
        seq = [r for r in sorted(records[token], key=lambda x: x["step"]) if cutoff <= r["step"] < cutoff + horizon]
        pos = np.zeros((horizon, 3), dtype=np.float64)
        heading = np.zeros(horizon, dtype=np.float64)
        vel = np.zeros((horizon, 2), dtype=np.float64)
        valid = np.zeros(horizon, dtype=np.float64)
        step_rows = []
        for r in seq:
            i = int(r["step"]) - cutoff
            if i < 0 or i >= horizon:
                continue
            p = r["pos"]
            pos[i, : min(3, len(p))] = p[: min(3, len(p))]
            heading[i] = r["heading"]
            vv = r["vel"]
            if len(vv) >= 2:
                vel[i, :2] = vv[:2]
            valid[i] = 1.0
            step_rows.append(
                {
                    "step": i,
                    "global_step": int(r["step"]),
                    "idm_pos": pos[i, :2].tolist(),
                    "idm_heading": float(heading[i]),
                    "idm_speed": float(r["speed"]),
                    "idm_valid": 1.0,
                    "policy": r.get("policy"),
                    "navigation": r.get("navigation"),
                }
            )
        pname = seq[0]["policy"] if seq else None
        if pname == "IDMPolicy":
            src = "idm_warm"
            coverage["idm_rolled"].append(token)
            speeds = [s["idm_speed"] for s in step_rows]
            osc = detect_oscillation(speeds, pos)
            lane_none = sum(
                1
                for s in step_rows
                if s.get("navigation", {}).get("current_lane") is None
                and s.get("navigation", {}).get("current_lane_kind") in (None, "none", "CenterLane")
            )
            # CenterLane with id=None counted as synthetic / no map lane id
            map_lane_steps = sum(
                1
                for s in step_rows
                if s.get("navigation", {}).get("current_lane_kind") not in (None, "none", "CenterLane")
                and s.get("navigation", {}).get("current_lane") is not None
            )
            idm_diag[token] = {
                "policy": pname,
                "navigation_at_cutoff": (seq[0].get("navigation") if seq else None),
                "steps": step_rows,
                "oscillation": osc,
                "n_steps_centerlane_or_none": int(lane_none),
                "n_steps_map_lane_id": int(map_lane_steps),
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

    ego_meta_light = {
        k: v
        for k, v in ego_meta.items()
        if k not in ("requested_center_traj", "requested_heading_traj")
    }
    ego_meta_light["executed_ego_conditioning_hash"] = ego_verify["executed_ego_conditioning_hash"]

    # Pre-cutoff focus agent trail
    focus_pre = []
    if FOCUS_TOKEN in records:
        for r in records[FOCUS_TOKEN]:
            if r["step"] > cutoff:
                break
            focus_pre.append(
                {
                    "step": r["step"],
                    "pos": r["pos"][:2].tolist(),
                    "speed": r["speed"],
                    "policy": r["policy"],
                    "navigation": r["navigation"],
                }
            )

    return {
        "source": "idm_warm_start",
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
        "ego_verification": ego_verify,
        "idm_agent_diagnostics": idm_diag,
        "hydra_scorer_config": hydra_scorer,
        "seed": seed,
        "idm_truly_starts_at_step": idm_start_step,
        "idm_start_note": (
            "IDMPolicy is attached at env.reset for dynamic vehicles; "
            "official IDM control executes from scene start (step 0), not only after cutoff."
        ),
        "pre_cutoff_ego_trajectory_source": "original_log_frames_[0,cutoff)",
        "post_cutoff_ego_trajectory_source": "vocab_plan_same_as_cold_start",
        "cutoff_gate": cutoff_gate,
        "cold_at_cutoff": cold_at_cutoff,
        "warm_at_cutoff": {
            "step": warm_at_cutoff["step"],
            "map": warm_at_cutoff["map"],
            "agent_tokens_sorted": warm_at_cutoff["agent_tokens_sorted"],
            "traffic_lights": warm_at_cutoff["traffic_lights"],
            "agents": {
                t: {k: v for k, v in a.items() if k != "navigation"} | {"navigation": a.get("navigation")}
                for t, a in warm_at_cutoff["agents"].items()
                if t == "ego" or t in coverage["idm_rolled"] or t == FOCUS_TOKEN
            },
        },
        "pre_cutoff_divergence": {
            "per_step": pre_cutoff_vs_log,
            "first_non_A": first_div,
        },
        "focus_agent_pre_cutoff": focus_pre,
        "restore_freeze_physics_design": {
            "goal": "Keep warm-start navigation continuity; restore log physical state at cutoff before post-cutoff roll.",
            "collaboration_layer_sketch": [
                "1. Run warm-start IDM from scene start to cutoff (as here).",
                "2. At cutoff, for each IDM agent call public BaseAgent.set_position / set_heading_theta / set_velocity from log freeze.",
                "3. Call navigation.update_localization() so lane longitudinal state matches restored pose.",
                "4. Do NOT re-construct IDMNavigation / do NOT re-init original_route from truncated traj.",
                "5. Continue with same post-cutoff ego conditioning.",
            ],
            "upstream_or_private_risk": (
                "If set_* + update_localization is insufficient (e.g. IDMPolicy.routing_target_lane "
                "or internal IDM state desync), would need upstream/private-state changes — pause, do not implement."
            ),
            "implemented_this_round": False,
        },
        "notes": [
            "Warm-start uses full original scene prefix, not slice_scene_from_cutoff.",
            "No 3DGS; agent_policy=idm_policy; agent_navigation=idm_navigation.",
        ],
    }


def compare_cold_warm_agents(
    cold_diag: Dict[str, Any],
    warm_pack: Dict[str, Any],
    cold_futures: Dict[str, Any],
    focus: str = FOCUS_TOKEN,
) -> Dict[str, Any]:
    warm_diag = warm_pack.get("idm_agent_diagnostics", {})
    warm_fut = warm_pack["futures"]
    out: Dict[str, Any] = {"focus": focus, "agents": {}}
    tokens = sorted(set(warm_pack["coverage"]["idm_rolled"]) | set(cold_diag.keys()))
    for tok in tokens:
        wd = warm_diag.get(tok, {})
        cd = cold_diag.get(tok, {})
        cf = cold_futures.get(tok, {})
        wf = warm_fut.get(tok, {})
        row: Dict[str, Any] = {
            "cold_policy": cd.get("policy") or cf.get("policy"),
            "warm_policy": wd.get("policy") or wf.get("policy"),
            "cold_nav_at_reset": cd.get("navigation_at_reset"),
            "warm_nav_at_cutoff": wd.get("navigation_at_cutoff"),
            "warm_oscillation": wd.get("oscillation"),
        }
        if cf and wf and "position" in cf and "position" in wf:
            cp = np.asarray(cf["position"], dtype=np.float64)
            wp = np.asarray(wf["position"], dtype=np.float64)
            cv = np.asarray(cf.get("valid", np.ones(len(cp))), dtype=np.float64).reshape(-1)
            wv = np.asarray(wf.get("valid", np.ones(len(wp))), dtype=np.float64).reshape(-1)
            n = min(len(cp), len(wp), len(cv), len(wv))
            mask = (cv[:n] > 0) & (wv[:n] > 0)
            if mask.any():
                d = np.linalg.norm(cp[:n, :2] - wp[:n, :2], axis=1)
                d_m = d[mask]
                first = None
                for i, ok in enumerate(mask):
                    if ok and d[i] > 0.05:
                        first = i
                        break
                row["common_valid_ade_mean_m"] = float(np.mean(d_m))
                row["common_valid_ade_max_m"] = float(np.max(d_m))
                row["first_fork_step_gt_5cm"] = first
                row["cold_net_disp_m"] = float(np.linalg.norm(cp[n - 1, :2] - cp[0, :2]))
                row["warm_net_disp_m"] = float(np.linalg.norm(wp[n - 1, :2] - wp[0, :2]))
        # cold oscillation from steps if present
        if cd.get("steps"):
            speeds = [s.get("idm_speed", 0.0) for s in cd["steps"]]
            pos = np.asarray([s["idm_pos"] for s in cd["steps"]], dtype=np.float64)
            row["cold_oscillation"] = detect_oscillation(speeds, pos)
        out["agents"][tok] = row

    f = out["agents"].get(focus, {})
    wo = f.get("warm_oscillation") or {}
    co = f.get("cold_oscillation") or {}
    out["focus_summary"] = {
        "token": focus,
        "warm_near_stationary_oscillation": wo.get("near_stationary_oscillation"),
        "warm_complete_stall": wo.get("complete_stall"),
        "warm_net_disp_m": wo.get("net_displacement_m") if wo.get("net_displacement_m") is not None else f.get("warm_net_disp_m"),
        "warm_path_length_m": wo.get("path_length_m"),
        "warm_speed_max": wo.get("speed_max"),
        "cold_near_stationary_oscillation": co.get("near_stationary_oscillation"),
        "cold_complete_stall": co.get("complete_stall"),
        "cold_net_disp_m": co.get("net_displacement_m") if co.get("net_displacement_m") is not None else f.get("cold_net_disp_m"),
        "warm_lane_kind_at_cutoff": (f.get("warm_nav_at_cutoff") or {}).get("current_lane_kind"),
        "warm_following_original_traj_at_cutoff": (f.get("warm_nav_at_cutoff") or {}).get(
            "following_original_traj"
        ),
        "cold_lane_at_reset": (f.get("cold_nav_at_reset") or {}).get("current_lane"),
        "cold_warm_max_ade_m": f.get("common_valid_ade_max_m"),
        # backward-compatible aliases
        "warm_near_stationary": wo.get("near_stationary_oscillation"),
        "cold_near_stationary": co.get("near_stationary_oscillation"),
    }
    return out


def maybe_score_warm(
    scene: Dict[str, Any],
    warm_pack: Dict[str, Any],
    cold_out_dir: Path,
    out_dir: Path,
    vocab_path: Path,
    vocab: np.ndarray,
    cutoff: int,
    horizon: int,
    plan_idx: int,
    asset_folder: str,
) -> Optional[Dict[str, Any]]:
    """Score warm-start future vs existing Replay scores if gate allows."""
    from score_frozen_disagreement import score_one_source  # noqa: WPS433

    with (cold_out_dir / "scores_log_replay.pkl").open("rb") as f:
        replay_scores = pickle.load(f)
    req_hash = warm_pack["ego_conditioning"]["requested_ego_conditioning_hash"]
    t0 = time.time()
    provenance = score_one_source(
        scene=scene,
        futures_pack=warm_pack,
        source_name="idm_warm",
        cutoff=cutoff,
        horizon=horizon,
        work_dir=out_dir / "score_workdir_idm_warm",
        asset_folder=asset_folder,
        vocab_path=vocab_path,
        vocab=vocab,
        plan_idx=plan_idx,
        requested_ego_hash=req_hash,
        wall_limit_sec=3600,
        rss_limit_gb=64.0,
        t0=t0,
    )
    with (out_dir / "scores_idm_warm.pkl").open("rb") as f:
        warm_rewards = pickle.load(f)
    save_json(out_dir / "scores_idm_warm_run_meta.json", {k: v for k, v in provenance.items()})

    r = replay_scores
    i = warm_rewards
    metrics = {}
    for k in PDM_KEYS:
        a = np.asarray(r[k]).astype(np.float64).reshape(-1)
        b = np.asarray(i[k]).astype(np.float64).reshape(-1)
        metrics[k] = {
            "equal": bool(np.array_equal(a, b)),
            "max_abs_diff": float(np.max(np.abs(a - b))),
            "mean_abs_diff": float(np.mean(np.abs(a - b))),
        }
    noc_r = np.asarray(r["no_at_fault_collisions"]).astype(np.float64).reshape(-1)
    noc_i = np.asarray(i["no_at_fault_collisions"]).astype(np.float64).reshape(-1)
    ttc_r = np.asarray(r["time_to_collision_within_bound"]).astype(np.float64).reshape(-1)
    ttc_i = np.asarray(i["time_to_collision_within_bound"]).astype(np.float64).reshape(-1)
    flips = {
        "noc": directional_binary_flips(noc_r, noc_i, safe_threshold=1.0),
        "ttc": directional_binary_flips(ttc_r, ttc_i, safe_threshold=1.0),
    }
    score_r = np.asarray(r["score"]).astype(np.float64).reshape(-1)
    score_i = np.asarray(i["score"]).astype(np.float64).reshape(-1)
    ties = analyze_max_score_ties(score_r, score_i)
    tau = kendall_tau_b(score_r, score_i)
    return {
        "metrics": metrics,
        "flips": flips,
        "max_score_ties": ties,
        "kendall_tau_b": tau,
        "n_candidates": int(len(score_r)),
        "fingerprint": provenance.get("input_state_fingerprint"),
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
        "--cold-dir",
        type=Path,
        default=Path("/mnt/cpfs/prediction/lyyy/myself/WE/data/frozen_paired/smoke1_cutoff3_validity_v2"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/mnt/cpfs/prediction/lyyy/myself/WE/data/frozen_paired/smoke1_cutoff3_warm_v1"),
    )
    parser.add_argument("--cutoff", type=int, default=DEFAULT_CUTOFF)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--plan-idx", type=int, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--asset-folder", type=str, default=None)
    parser.add_argument("--skip-score", action="store_true")
    args = parser.parse_args()

    t0 = time.time()
    we_root = Path(os.environ.get("WORLDENGINE_ROOT", "/mnt/cpfs/prediction/lyyy/myself/WE/WE/upstream/WorldEngine"))
    os.environ.setdefault("WORLDENGINE_ROOT", str(we_root))
    os.environ.setdefault("SIMENGINE_ROOT", str(we_root / "projects/SimEngine"))
    os.environ.setdefault("NUPLAN_MAPS_ROOT", str(we_root / "data/raw/nuplan/dataset/maps"))
    sys.path.insert(0, os.environ["SIMENGINE_ROOT"])

    asset_folder = args.asset_folder or str(we_root / "data/sim_engine/assets/navtest_failures/assets")
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    plan_res = resolve_plan_idx(args.plan_idx, None, None, args.cutoff)
    plan_idx = int(plan_res["plan_idx"])
    scene = load_scene_dict(args.scene_pkl)
    vocab = np.load(args.vocab)

    print(
        f"[warm] cutoff={args.cutoff} horizon={args.horizon} plan_idx={plan_idx} "
        f"tol={CUTOFF_GATE_TOLERANCES}",
        flush=True,
    )

    try:
        warm = run_warm_start_idm(
            scene,
            args.cutoff,
            args.horizon,
            plan_idx,
            vocab,
            work_dir=out_dir / "idm_warm_workdir",
            asset_folder=asset_folder,
            seed=args.seed,
        )
    except Exception as e:
        save_json(out_dir / "warm_error.json", {"error": str(e), "trace": traceback.format_exc()})
        print(traceback.format_exc())
        return 2

    gate = warm["cutoff_gate"]
    save_json(out_dir / "cutoff_state_gate.json", gate)
    save_json(out_dir / "ego_conditioning_verification_warm.json", warm["ego_verification"])
    save_json(out_dir / "idm_warm_agent_diagnostics.json", warm["idm_agent_diagnostics"])
    save_json(out_dir / "pre_cutoff_divergence.json", warm["pre_cutoff_divergence"])
    save_json(out_dir / "restore_freeze_physics_design.json", warm["restore_freeze_physics_design"])

    # Load cold diagnostics / futures for comparison
    cold_diag = {}
    cold_diag_path = args.cold_dir / "idm_agent_diagnostics.json"
    if cold_diag_path.exists():
        cold_diag = json.loads(cold_diag_path.read_text())
    with (args.cold_dir / "future_idm.pkl").open("rb") as f:
        cold_idm = pickle.load(f)
    with (args.cold_dir / "future_log_replay.pkl").open("rb") as f:
        replay = pickle.load(f)

    traj_cmp = None
    if gate["allow_future_reward_compare"]:
        traj_cmp = compare_cold_warm_agents(cold_diag, warm, cold_idm["futures"])
        # also vs replay for focus net disp
        fut_cmp = compare_futures(replay, warm)
        save_json(out_dir / "future_compare_replay_vs_warm.json", fut_cmp)
    else:
        # Still emit warm-only focus diagnostics (not a paired reward claim)
        traj_cmp = compare_cold_warm_agents(cold_diag, warm, cold_idm["futures"])
        traj_cmp["note"] = (
            "Cutoff gate grade C: cold/warm trajectory numbers are diagnostic only; "
            "do NOT interpret as fair paired reward comparison."
        )
    save_json(out_dir / "cold_warm_traj_compare.json", traj_cmp)

    meta = {k: v for k, v in warm.items() if k not in ("futures", "cold_at_cutoff", "warm_at_cutoff")}
    # keep light warm_at_cutoff already trimmed
    meta["warm_at_cutoff"] = warm["warm_at_cutoff"]
    meta["cold_at_cutoff_tokens"] = warm["cold_at_cutoff"]["agent_tokens_sorted"]
    save_json(out_dir / "future_idm_warm_meta.json", meta)
    np.savez_compressed(out_dir / "future_idm_warm.npz", **futures_to_numpy_bundle(warm))
    with (out_dir / "future_idm_warm.pkl").open("wb") as f:
        dump = {k: v for k, v in warm.items() if k not in ("cold_at_cutoff",)}
        if "ego_conditioning" in dump:
            dump["ego_conditioning"] = {
                kk: vv
                for kk, vv in dump["ego_conditioning"].items()
                if kk not in ("requested_center_traj", "requested_heading_traj")
            }
        pickle.dump(dump, f)

    score_cmp = None
    if gate["allow_future_reward_compare"] and not args.skip_score:
        print("[warm] cutoff gate allows scoring; running PDM...", flush=True)
        try:
            score_cmp = maybe_score_warm(
                scene,
                warm,
                args.cold_dir,
                out_dir,
                args.vocab,
                vocab,
                args.cutoff,
                args.horizon,
                plan_idx,
                asset_folder,
            )
            save_json(out_dir / "warm_vs_replay_score_compare.json", score_cmp)
        except Exception as e:
            save_json(out_dir / "warm_score_error.json", {"error": str(e), "trace": traceback.format_exc()})
            print(traceback.format_exc())
            return 3
    else:
        print(
            f"[warm] skip reward scoring (gate={gate['grade']} allow={gate['allow_future_reward_compare']} skip={args.skip_score})",
            flush=True,
        )

    focus = traj_cmp.get("focus_summary", {}) if traj_cmp else {}
    warm_osc = bool(focus.get("warm_near_stationary_oscillation") or focus.get("warm_near_stationary"))
    cold_osc = bool(focus.get("cold_near_stationary_oscillation") or focus.get("cold_near_stationary"))
    warm_stall = bool(focus.get("warm_complete_stall"))
    verdict_bits = {
        "cutoff_gate_grade": gate["grade"],
        "allow_reward_compare": gate["allow_future_reward_compare"],
        "warm_focus_near_stationary_oscillation": warm_osc,
        "warm_focus_complete_stall": warm_stall,
        "cold_focus_near_stationary_oscillation": cold_osc,
        "warm_focus_net_disp_m": focus.get("warm_net_disp_m"),
        "cold_focus_net_disp_m": focus.get("cold_net_disp_m"),
        "warm_lane_kind": focus.get("warm_lane_kind_at_cutoff"),
        "idm_starts_at_step": warm["idm_truly_starts_at_step"],
        "first_pre_cutoff_divergence": warm["pre_cutoff_divergence"]["first_non_A"],
    }
    # Classification (engineering only)
    if gate["grade"] == "C":
        if cold_osc and not warm_osc:
            classification = "cold_start_artifact"
            reason = (
                "Cold-start near-stationary oscillation of 44df645d is NOT reproduced under warm-start "
                f"(warm_complete_stall={warm_stall}, warm_net_disp_m={focus.get('warm_net_disp_m')}). "
                "Cutoff physics unpaired from step 1 (gate C); cold reinit at truncated scene changes "
                "control continuum. Stage 1.5–1.7 32% NOC flips must not be used as effect size; "
                "expand blocked until warm-start or restore-freeze-physics protocol."
            )
        elif cold_osc and warm_osc:
            classification = "inconclusive_needs_restore_physics"
            reason = (
                "Both cold and warm show stagnation-like behavior but cutoff unpaired (gate C); "
                "cannot attribute flips to official continuous IDM without restore-physics."
            )
        else:
            classification = "inconclusive_needs_restore_physics"
            reason = (
                "Cutoff physics diverged (gate C); cannot fair-compare rewards. "
                "See restore_freeze_physics_design; do not expand sample yet."
            )
    else:
        if score_cmp is not None:
            noc_flip = score_cmp["flips"]["noc"]["a_safe_to_b_danger"]
            if warm_osc and noc_flip > 1000:
                classification = "official_idm_behavior"
                reason = "gate paired; warm still oscillates near-stationary; NOC flips persist"
            elif warm_stall and noc_flip > 1000:
                classification = "official_idm_behavior"
                reason = "gate paired; warm complete stall; NOC flips persist"
            elif (not warm_osc and not warm_stall) and noc_flip < 500:
                classification = "cold_start_artifact"
                reason = "gate paired; warm moves normally; flips largely gone"
            else:
                classification = "mixed_engineering_evidence"
                reason = "gate paired but mixed trajectory/flip signals"
        else:
            classification = "paired_traj_only"
            reason = "gate paired; scoring skipped"

    expand_allowed = False  # stage 1.8: never allow expand this round; report recommendation
    expand_recommendation = "block_until_warm_or_restore"
    if classification == "official_idm_behavior":
        expand_recommendation = "allow_with_warm_start_protocol"
    elif classification == "cold_start_artifact":
        expand_recommendation = "block_cold_start_32pct_invalid"

    summary = {
        "status": "ok",
        "wall_s": round(time.time() - t0, 3),
        "rss_gb": round(_rss_gb(), 3),
        "cutoff": args.cutoff,
        "plan_idx": plan_idx,
        "cutoff_gate_grade": gate["grade"],
        "cutoff_gate_allow_reward_compare": gate["allow_future_reward_compare"],
        "tolerances_declared_a_priori": CUTOFF_GATE_TOLERANCES,
        "idm_truly_starts_at_step": warm["idm_truly_starts_at_step"],
        "ego_pre_cutoff_source": warm["pre_cutoff_ego_trajectory_source"],
        "ego_post_cutoff_pos_error_max_m": warm["ego_verification"]["pos_error_max_m"],
        "ego_post_cutoff_heading_error_max_deg": warm["ego_verification"]["heading_error_max_deg"],
        "focus_token": FOCUS_TOKEN,
        "focus_summary": focus,
        "warm_vs_replay_flips": None
        if score_cmp is None
        else {
            "noc_replay_safe_idm_danger": score_cmp["flips"]["noc"]["a_safe_to_b_danger"],
            "ttc_replay_safe_idm_danger": score_cmp["flips"]["ttc"]["a_safe_to_b_danger"],
            "noc_flip_rate": score_cmp["flips"]["noc"]["flip_ratio"],
            "ttc_flip_rate": score_cmp["flips"]["ttc"]["flip_ratio"],
        },
        "classification": classification,
        "classification_reason": reason,
        "expand_allowed_this_round": expand_allowed,
        "expand_recommendation": expand_recommendation,
        "verdict_bits": verdict_bits,
        "idm_rolled_count": len(warm["coverage"]["idm_rolled"]),
        "idm_rolled": warm["coverage"]["idm_rolled"],
    }
    save_json(out_dir / "warm_start_summary.json", summary)

    # light CSV for git reports
    csv_path = out_dir / "warm_start_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["key", "value"])
        for k, v in summary.items():
            if isinstance(v, (dict, list)):
                w.writerow([k, json.dumps(v, ensure_ascii=False, default=str)])
            else:
                w.writerow([k, v])

    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
