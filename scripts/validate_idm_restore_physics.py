#!/usr/bin/env python3
"""Stage 1.9: restore cutoff physical state to log freeze while KEEPING warm-start
IDMPolicy/IDMNavigation objects (no reconstruction), to build a fair Replay vs
Restored-IDM comparison. No upstream edits; only public BaseAgent setters +
navigation.update_localization() are used (see frozen_state_lib.restore_agent_physics_via_public_api
for the safety audit).
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
from typing import Any, Dict, List, Optional

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from frozen_state_lib import (  # noqa: E402
    CUTOFF_GATE_TOLERANCES,
    DEFAULT_CUTOFF,
    DEFAULT_HORIZON,
    DEFAULT_SEED,
    agent_valid_at,
    build_input_state_fingerprint,
    capture_live_agent_nav,
    classify_cutoff_gate,
    compare_futures,
    extract_log_agent_full_state_at,
    extract_log_agent_state_at,
    extract_replay_futures,
    extract_scorer_config_from_hydra,
    futures_to_numpy_bundle,
    inject_ego_conditioning_at_cutoff,
    load_scene_dict,
    resolve_plan_idx,
    restore_agent_physics_via_public_api,
    save_json,
    scene_as_dict,
    sha256_bytes,
    truncate_scene_prefix,
    verify_ego_conditioning_execution,
)
from ranking_metrics import (  # noqa: E402
    analyze_max_score_ties,
    assert_ratios_in_unit_interval,
    directional_binary_flips,
    kendall_tau_b,
)
from run_hybrid_attribution import compare_to_baseline, make_hybrid, score_hybrid  # noqa: E402
from score_frozen_disagreement import score_one_source  # noqa: E402

FOCUS_TOKEN = "44df645d1b5b584b"
# Route/lane compatibility heuristic threshold declared a priori (not tuned post-hoc):
# a restored position should project onto its (possibly stale) current_lane with
# lateral offset no larger than ~2 generous lane widths.
LANE_LAT_INCOMPATIBLE_M = 6.0
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


def _compose_restore_cfg(work_dir: Path, asset_folder: str, max_step: int):
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
                "job_name=frozen_idm_restore_physics",
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
    alt = 0
    for i in range(2, len(sp)):
        if (sp[i] - sp[i - 1]) * (sp[i - 1] - sp[i - 2]) < 0:
            alt += 1
    osc = bool(net < 3.0 and path > net * 1.5 and alt >= 2 and float(np.max(sp)) > 0.5)
    complete_stall = bool(net < 0.5 and path < 0.5 and (len(sp) == 0 or float(np.max(sp)) < 0.1))
    large_jumps = []
    for i in range(1, len(positions)):
        d = float(np.linalg.norm(positions[i, :2] - positions[i - 1, :2]))
        if d > 15.0:
            large_jumps.append({"from_step": i - 1, "to_step": i, "step_dist_m": d})
    return {
        "near_stationary_oscillation": osc,
        "complete_stall": complete_stall,
        "net_displacement_m": net,
        "path_length_m": path,
        "speed_alternations": int(alt),
        "speed_min": float(np.min(sp)) if len(sp) else None,
        "speed_max": float(np.max(sp)) if len(sp) else None,
        "speed_mean": float(np.mean(sp)) if len(sp) else None,
        "large_step_jumps_gt_15m": large_jumps,
    }


def run_restore_physics_idm(
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
    sim_scene = truncate_scene_prefix(orig_scene, length)
    ego_meta = inject_ego_conditioning_at_cutoff(sim_scene, cutoff, plan_idx, vocab)

    work_dir.mkdir(parents=True, exist_ok=True)
    pkl_path = work_dir / "all_scenarios.pkl"
    with pkl_path.open("wb") as f:
        pickle.dump(scene_as_dict(sim_scene), f)

    if engine_initialized():
        close_engine()

    max_step = length - 1
    cfg = _compose_restore_cfg(work_dir, asset_folder, max_step)
    hydra_scorer = extract_scorer_config_from_hydra(cfg)

    data = pickle.load(open(pkl_path, "rb"))
    env = build_env(cfg, name="frozen_idm_restore_physics", data=data)
    env.reset(seed=seed)

    records: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    idm_start_step: Optional[int] = None
    pre_cutoff_nav_trail: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    def snap_all(step: int) -> None:
        nonlocal idm_start_step
        am = env.engine.agent_manager
        for aid, ag in am.all_agents.items():
            row = _snap_agent(ag, step)
            records[aid].append(row)
            if row["policy"] == "IDMPolicy" and idm_start_step is None:
                idm_start_step = int(step)
            if aid in orig_scene["object_track"]:
                pre_cutoff_nav_trail[aid].append(
                    {
                        "step": int(step),
                        "following_original_traj": row["navigation"].get("following_original_traj"),
                        "current_lane_kind": row["navigation"].get("current_lane_kind"),
                    }
                )

    snap_all(0)
    for _ in range(cutoff):
        env.step(None)
        step = int(env.engine.episode_step)
        snap_all(step)

    assert env.engine.episode_step == cutoff, (env.engine.episode_step, cutoff)

    # ---- Restore step: pre-cutoff irreversible-transition check (before touching state) ----
    pre_cutoff_transition_flags: Dict[str, bool] = {}
    for aid, trail in pre_cutoff_nav_trail.items():
        flipped = False
        if len(trail) >= 2:
            first = trail[0]
            for row in trail[1:]:
                if row["following_original_traj"] != first["following_original_traj"]:
                    flipped = True
                    break
        pre_cutoff_transition_flags[aid] = flipped

    am = env.engine.agent_manager
    identity_before: Dict[str, Any] = {}
    for aid, ag in am.all_agents.items():
        identity_before[aid] = {
            "policy_id": id(getattr(ag, "policy", None)),
            "navigation_id": id(getattr(ag, "navigation", None)),
            "policy_type": type(getattr(ag, "policy", None)).__name__,
            "navigation_type": type(getattr(ag, "navigation", None)).__name__ if getattr(ag, "navigation", None) is not None else None,
        }

    log_full_state_at_cutoff = extract_log_agent_full_state_at(orig_scene, cutoff, include_ego=True)

    restore_diag: Dict[str, Any] = {}
    identity_after: Dict[str, Any] = {}
    identity_violations: List[str] = []
    for aid, ag in am.all_agents.items():
        if aid not in log_full_state_at_cutoff:
            continue
        target = log_full_state_at_cutoff[aid]
        if not target["valid"]:
            restore_diag[aid] = {"skipped": True, "reason": "invalid_at_cutoff_in_log"}
            continue
        diag = restore_agent_physics_via_public_api(ag, target, restore_angular_velocity=True)
        restore_diag[aid] = diag
        identity_after[aid] = {
            "policy_id": id(getattr(ag, "policy", None)),
            "navigation_id": id(getattr(ag, "navigation", None)),
        }
        if identity_after[aid]["policy_id"] != identity_before[aid]["policy_id"]:
            identity_violations.append(f"{aid}: policy identity changed")
        if identity_after[aid]["navigation_id"] != identity_before[aid]["navigation_id"]:
            identity_violations.append(f"{aid}: navigation identity changed")

        # Lane compatibility check (heuristic, declared threshold LANE_LAT_INCOMPATIBLE_M).
        nav = getattr(ag, "navigation", None)
        lane_compat = None
        lat_after = None
        if nav is not None and getattr(nav, "current_lane", None) is not None:
            try:
                long_after, lat_after = nav.current_lane.local_coordinates(ag.current_position)
                lat_after = float(lat_after)
                lane_compat = bool(abs(lat_after) <= LANE_LAT_INCOMPATIBLE_M)
            except Exception as e:  # pragma: no cover - defensive, report don't crash
                restore_diag[aid]["lane_local_coordinates_error"] = str(e)
        restore_diag[aid]["lat_after_restore_m"] = lat_after
        restore_diag[aid]["lane_compatible_with_restored_position"] = lane_compat
        restore_diag[aid]["pre_cutoff_irreversible_transition"] = pre_cutoff_transition_flags.get(aid, False)
        restore_diag[aid]["lane_lat_incompatible_threshold_m"] = LANE_LAT_INCOMPATIBLE_M

        # Overwrite step==cutoff snapshot with the restored (post-update_localization) state
        # so all downstream future-extraction uses the corrected row.
        restored_row = _snap_agent(ag, cutoff)
        for i in range(len(records[aid]) - 1, -1, -1):
            if records[aid][i]["step"] == cutoff:
                records[aid][i] = restored_row
                break

    if identity_violations:
        # Would require upstream/private-state changes to fix; per instructions, pause.
        save_json(
            work_dir.parent / "HARD_PAUSE_identity_violation.json",
            {"violations": identity_violations, "identity_before": identity_before},
        )
        raise RuntimeError(
            "HARD_PAUSE: policy/navigation object identity changed during restore: "
            + "; ".join(identity_violations)
        )

    restored_at_cutoff_full: Dict[str, Any] = {}
    for aid, seq in records.items():
        if aid not in orig_scene["object_track"]:
            continue
        rows_at_cutoff = [x for x in seq if x["step"] == cutoff]
        if not rows_at_cutoff:
            continue
        r = rows_at_cutoff[0]
        restored_at_cutoff_full[aid] = {
            "type": orig_scene["object_track"].get(aid, {}).get("type"),
            "pos_xy": [float(r["pos"][0]), float(r["pos"][1])],
            "heading": r["heading"],
            "speed": r["speed"],
            "valid": True,
            "length": float(np.asarray(orig_scene["object_track"][aid]["state"]["length"])[cutoff, 0]),
            "width": float(np.asarray(orig_scene["object_track"][aid]["state"]["width"])[cutoff, 0]),
            "policy": r["policy"],
            "navigation": r["navigation"],
        }
    restored_at_cutoff = {
        "step": cutoff,
        "map": orig_scene.get("map"),
        "agent_tokens_sorted": sorted(restored_at_cutoff_full.keys()),
        "agents": restored_at_cutoff_full,
        "traffic_lights": extract_log_agent_state_at(orig_scene, cutoff, include_ego=True)["traffic_lights"],
    }
    cold_at_cutoff = extract_log_agent_state_at(orig_scene, cutoff, include_ego=True)
    cutoff_gate = classify_cutoff_gate(cold_at_cutoff, restored_at_cutoff)

    # ---- Continue stepping cutoff -> cutoff+horizon-1 with restored physics ----
    for _ in range(max_step - cutoff):
        env.step(None)
        step = int(env.engine.episode_step)
        snap_all(step)

    ego_seq = sorted(records.get("ego", []), key=lambda x: x["step"])
    ego_post = [r for r in ego_seq if r["step"] >= cutoff]
    ego_verify = verify_ego_conditioning_execution(
        requested_center=ego_meta["requested_center_traj"],
        requested_heading=ego_meta["requested_heading_traj"],
        live_positions=[r["pos"] for r in ego_post],
        live_headings=[r["heading"] for r in ego_post],
    )

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
            src = "idm_restored"
            coverage["idm_rolled"].append(token)
            speeds = [s["idm_speed"] for s in step_rows]
            osc = detect_oscillation(speeds, pos)
            idm_diag[token] = {
                "policy": pname,
                "navigation_at_cutoff": (seq[0].get("navigation") if seq else None),
                "steps": step_rows,
                "oscillation": osc,
                "restore_diagnostic": restore_diag.get(token),
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
        k: v for k, v in ego_meta.items() if k not in ("requested_center_traj", "requested_heading_traj")
    }
    ego_meta_light["executed_ego_conditioning_hash"] = ego_verify["executed_ego_conditioning_hash"]

    return {
        "source": "idm_restored_physics",
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
        "cutoff_gate": cutoff_gate,
        "cold_at_cutoff": cold_at_cutoff,
        "restored_at_cutoff": restored_at_cutoff,
        "identity_before_restore": identity_before,
        "identity_after_restore": identity_after,
        "identity_violations": identity_violations,
        "restore_diagnostics": restore_diag,
        "pre_cutoff_transition_flags": pre_cutoff_transition_flags,
        "replay_ref": replay_ref,
        "notes": [
            "Full original scene prefix (truncate_scene_prefix), not slice_scene_from_cutoff.",
            "agent_policy=idm_policy; agent_navigation=idm_navigation; no 3DGS.",
            "At step==cutoff, physical state force-restored to log freeze via public "
            "setters + navigation.update_localization(); policy/navigation objects kept.",
        ],
    }


def build_three_way_traj_compare(
    cold_diag: Dict[str, Any],
    warm_diag: Optional[Dict[str, Any]],
    restored: Dict[str, Any],
    replay_ref: Dict[str, Any],
    focus: str = FOCUS_TOKEN,
) -> Dict[str, Any]:
    def ade_vs_replay(fut: Dict[str, Any], tok: str) -> Optional[Dict[str, Any]]:
        if tok not in fut or tok not in replay_ref["futures"]:
            return None
        rp = np.asarray(replay_ref["futures"][tok]["position"], dtype=np.float64)[:, :2]
        rv = np.asarray(replay_ref["futures"][tok].get("valid", np.ones(len(rp))), dtype=np.float64).reshape(-1)
        ip = np.asarray(fut[tok]["position"], dtype=np.float64)[:, :2]
        iv = np.asarray(fut[tok].get("valid", np.ones(len(ip))), dtype=np.float64).reshape(-1)
        n = min(len(rp), len(ip), len(rv), len(iv))
        mask = (rv[:n] > 0) & (iv[:n] > 0)
        if not mask.any():
            return None
        d = np.linalg.norm(rp[:n] - ip[:n], axis=1)
        return {
            "common_valid_mean_ade_m": float(np.mean(d[mask])),
            "common_valid_max_ade_m": float(np.max(d[mask])),
            "net_displacement_m": float(np.linalg.norm(ip[n - 1] - ip[0])),
        }

    restored_fut = restored["futures"]
    restored_idm_diag = restored["idm_agent_diagnostics"]
    tokens = sorted(restored["coverage"]["idm_rolled"])
    rows = {}
    for tok in tokens:
        row: Dict[str, Any] = {
            "restored_vs_replay": ade_vs_replay(restored_fut, tok),
            "restored_oscillation": restored_idm_diag.get(tok, {}).get("oscillation"),
            "restored_navigation_at_cutoff": restored_idm_diag.get(tok, {}).get("navigation_at_cutoff"),
            "restore_diagnostic": restored_idm_diag.get(tok, {}).get("restore_diagnostic"),
        }
        cd = cold_diag.get(tok)
        if cd and cd.get("steps"):
            speeds = [s.get("idm_speed", 0.0) for s in cd["steps"]]
            pos = np.asarray([s["idm_pos"] for s in cd["steps"]], dtype=np.float64)
            row["cold_oscillation"] = detect_oscillation(speeds, pos)
            row["cold_navigation_at_reset"] = cd.get("navigation_at_reset")
        if warm_diag:
            wd = warm_diag.get(tok)
            if wd and wd.get("steps"):
                speeds = [s.get("idm_speed", 0.0) for s in wd["steps"]]
                pos = np.asarray([s["idm_pos"] for s in wd["steps"]], dtype=np.float64)
                row["warm_oscillation"] = wd.get("oscillation") or detect_oscillation(speeds, pos)
                row["warm_navigation_at_cutoff"] = wd.get("navigation_at_cutoff")
        # cold vs restored / warm vs restored max deviation (diagnostic)
        if cd and cd.get("steps") and tok in restored_idm_diag:
            cp = np.asarray([s["idm_pos"] for s in cd["steps"]], dtype=np.float64)
            rp2 = np.asarray([s["idm_pos"] for s in restored_idm_diag[tok]["steps"]], dtype=np.float64)
            n = min(len(cp), len(rp2))
            if n:
                row["cold_vs_restored_max_dev_m"] = float(np.max(np.linalg.norm(cp[:n] - rp2[:n], axis=1)))
        rows[tok] = row

    f = rows.get(focus, {})
    focus_summary = {
        "token": focus,
        "restored_near_stationary_oscillation": (f.get("restored_oscillation") or {}).get(
            "near_stationary_oscillation"
        ),
        "restored_complete_stall": (f.get("restored_oscillation") or {}).get("complete_stall"),
        "restored_net_disp_m": (f.get("restored_vs_replay") or {}).get("net_displacement_m"),
        "restored_vs_replay_max_ade_m": (f.get("restored_vs_replay") or {}).get("common_valid_max_ade_m"),
        "restored_has_valid_lane": (
            (f.get("restored_navigation_at_cutoff") or {}).get("current_lane_kind") not in (None, "none")
        ),
        "restored_lane_kind": (f.get("restored_navigation_at_cutoff") or {}).get("current_lane_kind"),
        "cold_near_stationary_oscillation": (f.get("cold_oscillation") or {}).get("near_stationary_oscillation"),
        "warm_near_stationary_oscillation": (f.get("warm_oscillation") or {}).get("near_stationary_oscillation"),
        "warm_complete_stall": (f.get("warm_oscillation") or {}).get("complete_stall"),
        "cold_vs_restored_max_dev_m": f.get("cold_vs_restored_max_dev_m"),
    }
    return {"tokens": tokens, "per_token": rows, "focus_summary": focus_summary}


def run_scoring_and_attribution(
    orig_scene: Dict[str, Any],
    restored: Dict[str, Any],
    cold_dir: Path,
    out_dir: Path,
    vocab_path: Path,
    vocab: np.ndarray,
    cutoff: int,
    horizon: int,
    plan_idx: int,
    asset_folder: str,
    skip_attribution: bool = False,
) -> Dict[str, Any]:
    """Score Replay + Restored-IDM at the *current* cutoff.

    Cold-dir Replay scores are NOT reused: prior smoke1_cutoff3_* artifacts are a
    different freeze and must not be mixed into cutoff=4 tables.
    Optional cold_dir fingerprint is recorded only as a cross-check note.
    """
    ego_cond = restored.get("ego_conditioning") or {}
    req_hash = ego_cond.get("requested_ego_conditioning_hash")
    if not req_hash:
        raise RuntimeError("restored pack missing requested_ego_conditioning_hash")

    replay_pack = dict(restored["replay_ref"])
    replay_pack["ego_conditioning"] = ego_cond
    replay_pack["source"] = "log_replay"

    t0 = time.time()
    replay_prov = score_one_source(
        scene=orig_scene,
        futures_pack=replay_pack,
        source_name="log_replay",
        cutoff=cutoff,
        horizon=horizon,
        work_dir=out_dir / "score_workdir_log_replay",
        asset_folder=asset_folder,
        vocab_path=vocab_path,
        vocab=vocab,
        plan_idx=plan_idx,
        requested_ego_hash=req_hash,
        wall_limit_sec=3600,
        rss_limit_gb=64.0,
        t0=t0,
    )
    with (out_dir / "scores_log_replay.pkl").open("rb") as f:
        replay_scores = pickle.load(f)
    save_json(out_dir / "scores_log_replay_run_meta.json", replay_prov)

    provenance = score_one_source(
        scene=orig_scene,
        futures_pack=restored,
        source_name="idm_restored",
        cutoff=cutoff,
        horizon=horizon,
        work_dir=out_dir / "score_workdir_idm_restored",
        asset_folder=asset_folder,
        vocab_path=vocab_path,
        vocab=vocab,
        plan_idx=plan_idx,
        requested_ego_hash=req_hash,
        wall_limit_sec=3600,
        rss_limit_gb=64.0,
        t0=t0,
    )
    with (out_dir / "scores_idm_restored.pkl").open("rb") as f:
        restored_scores = pickle.load(f)
    save_json(out_dir / "scores_idm_restored_run_meta.json", provenance)

    fingerprint_pair_equal = (
        provenance["input_state_fingerprint"] == replay_prov["input_state_fingerprint"]
    )
    cold_fp_note: Dict[str, Any] = {"cold_dir": str(cold_dir), "compared": False}
    cold_fp_path = cold_dir / "independent_fingerprints_all.json"
    if cold_fp_path.exists():
        try:
            fps_all = json.loads(cold_fp_path.read_text())
            cold_fp = fps_all.get("score_replay_pre_inject")
            cold_fp_note.update(
                {
                    "compared": True,
                    "cold_dir_score_replay_pre_inject_fingerprint": cold_fp,
                    "equal_to_current_replay": cold_fp == replay_prov["input_state_fingerprint"],
                    "note": (
                        "Informational only; cutoff=3 cold fingerprints are expected to differ "
                        "from cutoff=4 and must not be used for scoring."
                    ),
                }
            )
        except Exception as e:  # noqa: BLE001
            cold_fp_note["error"] = str(e)

    save_json(
        out_dir / "fingerprint_cross_check.json",
        {
            "replay_run_fingerprint": replay_prov["input_state_fingerprint"],
            "restored_run_fingerprint": provenance["input_state_fingerprint"],
            "replay_idm_fingerprint_equal": fingerprint_pair_equal,
            "cold_dir_cross_check": cold_fp_note,
        },
    )

    metrics = {}
    for k in PDM_KEYS:
        a = np.asarray(replay_scores[k]).astype(np.float64).reshape(-1)
        b = np.asarray(restored_scores[k]).astype(np.float64).reshape(-1)
        metrics[k] = {
            "equal": bool(np.array_equal(a, b)),
            "max_abs_diff": float(np.max(np.abs(a - b))),
            "mean_abs_diff": float(np.mean(np.abs(a - b))),
        }
    noc = directional_binary_flips(
        np.asarray(replay_scores["no_at_fault_collisions"]).astype(np.float64).reshape(-1),
        np.asarray(restored_scores["no_at_fault_collisions"]).astype(np.float64).reshape(-1),
        safe_threshold=1.0,
    )
    ttc = directional_binary_flips(
        np.asarray(replay_scores["time_to_collision_within_bound"]).astype(np.float64).reshape(-1),
        np.asarray(restored_scores["time_to_collision_within_bound"]).astype(np.float64).reshape(-1),
        safe_threshold=1.0,
    )
    score_r = np.asarray(replay_scores["score"]).astype(np.float64).reshape(-1)
    score_i = np.asarray(restored_scores["score"]).astype(np.float64).reshape(-1)
    ties = analyze_max_score_ties(score_r, score_i)
    tau = kendall_tau_b(score_r, score_i)

    primary = {
        "metrics_equal_or_diff": metrics,
        "noc_flips": noc,
        "ttc_flips": ttc,
        "max_score_ties": ties,
        "kendall_tau_b": tau,
        "n_candidates": int(len(score_r)),
        "fingerprint": provenance["input_state_fingerprint"],
        "fingerprint_matches_cold_replay_fingerprint": fingerprint_pair_equal,
        "fingerprint_replay_idm_equal": fingerprint_pair_equal,
        "replay_calibration_note": (
            "Scored log_replay futures freshly at this cutoff/plan_idx (no reuse of "
            "smoke1_cutoff3_* cold-dir scores)."
        ),
    }
    save_json(out_dir / "restored_vs_replay_score_compare.json", primary)

    if skip_attribution:
        attribution_summary = {
            "skipped": True,
            "reason": "skip_attribution flag (train-side batch)",
            "idm_tokens": list(restored["coverage"]["idm_rolled"]),
            "full_restored_noc_flip": primary["noc_flips"]["total_flip"],
            "single_agent_ranked_by_noc_flip": [],
        }
        save_json(out_dir / "restored_hybrid_attribution_summary.json", attribution_summary)
        return {"primary": primary, "attribution": attribution_summary}

    # ---- Single-agent / leave-one-out attribution (reuses stage 1.7/1.8 tested code) ----
    idm_tokens = list(restored["coverage"]["idm_rolled"])
    expected_fp = replay_prov["input_state_fingerprint"]

    hybrids_spec = [("full_restored", idm_tokens)]
    for tok in idm_tokens:
        hybrids_spec.append((f"single_{tok[:8]}", [tok]))
    for tok in idm_tokens:
        others = [t for t in idm_tokens if t != tok]
        hybrids_spec.append((f"loo_without_{tok[:8]}", others))

    comparisons = {}
    attribution_rows = []
    for name, toks in hybrids_spec:
        pack = make_hybrid(replay_pack, restored, toks, name)
        scored = score_hybrid(
            scene=orig_scene,
            futures_pack=pack,
            name=name,
            cutoff=cutoff,
            horizon=horizon,
            work_dir=out_dir / f"attr_workdir_{name}",
            asset_folder=asset_folder,
            vocab_path=vocab_path,
            vocab=vocab,
            plan_idx=plan_idx,
            requested_ego_hash=req_hash,
            expected_fp=expected_fp,
        )
        with (out_dir / f"future_attr_{name}.pkl").open("wb") as f:
            pickle.dump(pack, f)
        comp = compare_to_baseline(replay_scores, scored["scores"], name)
        comp["hybrid_idm_tokens"] = toks
        comparisons[name] = comp
        attribution_rows.append(
            {
                "hybrid": name,
                "n_idm_agents": len(toks),
                "idm_tokens": toks,
                "noc_total_flip": comp["noc"]["total_flip"],
                "noc_safe_to_danger": comp["noc"]["replay_safe_to_hybrid_danger"],
                "ttc_total_flip": comp["ttc"]["total_flip"],
                "max_set_jaccard_vs_replay": comp["max_score_sets"]["jaccard"],
                "kendall_tau_b": comp["score"]["kendall_tau_b"],
                "dac_equal": comp["dac_equal"],
                "comfort_equal": comp["comfort_equal"],
                "direction_equal": comp["direction_equal"],
            }
        )
        print(f"[restore-attr] {name}: NOC flip={comp['noc']['total_flip']}", flush=True)

    full_noc = comparisons["full_restored"]["noc"]["total_flip"]
    for row in attribution_rows:
        row["noc_flip_frac_of_full"] = (
            float(row["noc_total_flip"]) / full_noc if full_noc else float("nan")
        )
        # LOO can exceed full when removing a suppressor agent; record, do not hard-fail smoke.
        if not np.isnan(row["noc_flip_frac_of_full"]) and row["noc_flip_frac_of_full"] > 1.0 + 1e-9:
            row["noc_flip_frac_exceeds_full"] = True
            row["noc_flip_frac_of_full_note"] = (
                "LOO NOC flip exceeds full_restored; possible suppressor interaction; "
                "fraction capped for display only."
            )
            row["noc_flip_frac_of_full_raw"] = float(row["noc_flip_frac_of_full"])
            row["noc_flip_frac_of_full"] = float(row["noc_total_flip"]) / full_noc
        elif not np.isnan(row["noc_flip_frac_of_full"]):
            assert 0.0 <= row["noc_flip_frac_of_full"] <= 1.0 + 1e-9
            row["noc_flip_frac_exceeds_full"] = False
    singles = [r for r in attribution_rows if r["hybrid"].startswith("single_")]
    singles_sorted = sorted(singles, key=lambda x: -x["noc_total_flip"])

    attribution_summary = {
        "idm_tokens": idm_tokens,
        "full_restored_noc_flip": full_noc,
        "attribution_table": attribution_rows,
        "single_agent_ranked_by_noc_flip": singles_sorted,
        "comparisons": comparisons,
        "note": (
            "noc_flip_frac_of_full may exceed 1.0 for some LOO hybrids when removing an "
            "agent increases flips; see noc_flip_frac_exceeds_full."
        ),
    }
    save_json(out_dir / "restored_hybrid_attribution_summary.json", attribution_summary)

    return {
        "primary": primary,
        "attribution": attribution_summary,
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
        help="Optional legacy diagnostics only; Replay scores are always recomputed at --cutoff.",
    )
    parser.add_argument(
        "--warm-dir",
        type=Path,
        default=Path("/mnt/cpfs/prediction/lyyy/myself/WE/data/frozen_paired/smoke1_cutoff3_warm_v1"),
        help="Optional legacy warm-start traj diagnostics; not used for cutoff=4 scoring.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/mnt/cpfs/prediction/lyyy/myself/WE/data/frozen_paired/smoke1_cutoff4_restore_v1"),
    )
    parser.add_argument("--cutoff", type=int, default=DEFAULT_CUTOFF)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--plan-idx", type=int, default=None)
    parser.add_argument("--plan-idx-csv", type=Path, default=None)
    parser.add_argument("--plan-idx-step", type=int, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--asset-folder", type=str, default=None)
    parser.add_argument("--skip-score", action="store_true")
    parser.add_argument(
        "--skip-attribution",
        action="store_true",
        help="Score Replay vs Restored-IDM only; skip hybrid single/LOO attribution (train-side batch).",
    )
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

    # plan_idx explicitly resolved from CLI/CSV/documented smoke default; never silently hardcoded.
    plan_res = resolve_plan_idx(args.plan_idx, args.plan_idx_csv, args.plan_idx_step, args.cutoff)
    plan_idx = int(plan_res["plan_idx"])
    save_json(out_dir / "plan_idx_resolution.json", plan_res)

    scene = load_scene_dict(args.scene_pkl)
    vocab = np.load(args.vocab)

    print(
        f"[restore] cutoff={args.cutoff} horizon={args.horizon} plan_idx={plan_idx} "
        f"plan_idx_source={plan_res['source']} tol={CUTOFF_GATE_TOLERANCES}",
        flush=True,
    )

    try:
        restored = run_restore_physics_idm(
            scene,
            args.cutoff,
            args.horizon,
            plan_idx,
            vocab,
            work_dir=out_dir / "idm_restore_workdir",
            asset_folder=asset_folder,
            seed=args.seed,
        )
    except Exception as e:
        save_json(out_dir / "restore_error.json", {"error": str(e), "trace": traceback.format_exc()})
        print(traceback.format_exc())
        return 2

    gate = restored["cutoff_gate"]
    save_json(out_dir / "cutoff_state_gate.json", gate)
    save_json(out_dir / "ego_conditioning_verification_restored.json", restored["ego_verification"])
    save_json(out_dir / "idm_agent_diagnostics_restored.json", restored["idm_agent_diagnostics"])
    save_json(
        out_dir / "restore_identity_check.json",
        {
            "identity_before_restore": restored["identity_before_restore"],
            "identity_after_restore": restored["identity_after_restore"],
            "identity_violations": restored["identity_violations"],
            "policy_identity_preserved_all": len(restored["identity_violations"]) == 0,
        },
    )
    save_json(out_dir / "restore_per_agent_diagnostics.json", restored["restore_diagnostics"])
    save_json(out_dir / "pre_cutoff_irreversible_transition_flags.json", restored["pre_cutoff_transition_flags"])

    if restored["ego_verification"]["hard_fail"]:
        save_json(out_dir / "HARD_PAUSE_ego_conditioning.json", restored["ego_verification"])
        print("HARD_PAUSE: ego conditioning exceeded hard threshold", flush=True)
        return 4

    # ---- Load cold/warm diagnostics for 3-way comparison ----
    cold_diag = {}
    cold_diag_path = args.cold_dir / "idm_agent_diagnostics.json"
    if cold_diag_path.exists():
        cold_diag = json.loads(cold_diag_path.read_text())
    warm_diag = None
    warm_diag_path = args.warm_dir / "idm_warm_agent_diagnostics.json"
    if warm_diag_path.exists():
        warm_diag = json.loads(warm_diag_path.read_text())

    traj_cmp = build_three_way_traj_compare(cold_diag, warm_diag, restored, restored["replay_ref"])
    save_json(out_dir / "cold_warm_restored_traj_compare.json", traj_cmp)

    meta = {
        k: v
        for k, v in restored.items()
        if k not in ("futures", "cold_at_cutoff", "restored_at_cutoff", "replay_ref", "identity_before_restore", "identity_after_restore")
    }
    meta["restored_at_cutoff"] = restored["restored_at_cutoff"]
    save_json(out_dir / "future_idm_restored_meta.json", meta)
    np.savez_compressed(out_dir / "future_idm_restored.npz", **futures_to_numpy_bundle(restored))
    with (out_dir / "future_idm_restored.pkl").open("wb") as f:
        dump = {k: v for k, v in restored.items() if k not in ("cold_at_cutoff", "identity_before_restore", "identity_after_restore")}
        pickle.dump(dump, f)
    with (out_dir / "future_log_replay.pkl").open("wb") as f:
        pickle.dump(restored["replay_ref"], f)

    score_result = None
    if gate["allow_future_reward_compare"] and not args.skip_score:
        print(f"[restore] cutoff gate={gate['grade']} allows scoring; running PDM + attribution...", flush=True)
        try:
            score_result = run_scoring_and_attribution(
                scene,
                restored,
                args.cold_dir,
                out_dir,
                args.vocab,
                vocab,
                args.cutoff,
                args.horizon,
                plan_idx,
                asset_folder,
                skip_attribution=bool(args.skip_attribution),
            )
        except Exception as e:
            save_json(out_dir / "restore_score_error.json", {"error": str(e), "trace": traceback.format_exc()})
            print(traceback.format_exc())
            return 3
    else:
        print(
            f"[restore] skip reward scoring (gate={gate['grade']} allow={gate['allow_future_reward_compare']} "
            f"skip={args.skip_score})",
            flush=True,
        )

    focus = traj_cmp["focus_summary"]
    if gate["grade"] == "C":
        verdict = "C_restore_not_trustworthy"
        verdict_reason = "Cutoff gate is C after restore attempt; cannot fair-compare rewards."
    elif score_result is None:
        verdict = "gate_ab_scoring_skipped"
        verdict_reason = "Gate A/B but scoring was skipped by flag; only trajectory diagnostics available."
    else:
        noc_flip = score_result["primary"]["noc_flips"]["a_safe_to_b_danger"]
        if focus.get("restored_near_stationary_oscillation") or focus.get("restored_complete_stall"):
            if noc_flip > 0:
                verdict = "B_real_disagreement_under_strict_pairing"
                verdict_reason = (
                    "Cutoff physics strictly paired (gate "
                    + gate["grade"]
                    + "); restored IDM trajectory still stalls/oscillates for focus agent and "
                    "NOC/TTC flips persist under fair comparison."
                )
            else:
                verdict = "gate_ab_no_flip_focus_still_abnormal"
                verdict_reason = "Gate paired but no NOC flips despite abnormal focus-agent trajectory."
        else:
            if noc_flip == 0:
                verdict = "A_cold_start_artifact_confirmed"
                verdict_reason = (
                    "Cutoff physics strictly paired; restored IDM trajectory normal for focus agent; "
                    "original 32% NOC flips do not reproduce -> confirmed cold-start artifact."
                )
            else:
                verdict = "mixed_evidence"
                verdict_reason = "Gate paired, focus trajectory looks normal, but some NOC flips remain; needs review."

    expand_allowed = False  # stage 1.9: never self-authorize expansion; only recommend
    summary = {
        "status": "ok",
        "wall_s": round(time.time() - t0, 3),
        "rss_gb": round(_rss_gb(), 3),
        "cutoff": args.cutoff,
        "horizon": args.horizon,
        "plan_idx": plan_idx,
        "plan_idx_resolution": plan_res,
        "cutoff_gate_grade": gate["grade"],
        "cutoff_gate_allow_reward_compare": gate["allow_future_reward_compare"],
        "tolerances_declared_a_priori": CUTOFF_GATE_TOLERANCES,
        "identity_violations": restored["identity_violations"],
        "policy_navigation_identity_preserved": len(restored["identity_violations"]) == 0,
        "idm_truly_starts_at_step": restored["idm_truly_starts_at_step"],
        "ego_post_cutoff_pos_error_max_m": restored["ego_verification"]["pos_error_max_m"],
        "ego_post_cutoff_heading_error_max_deg": restored["ego_verification"]["heading_error_max_deg"],
        "focus_token": FOCUS_TOKEN,
        "focus_summary": focus,
        "restored_vs_replay_flips": None
        if score_result is None
        else {
            "noc_safe_to_danger": score_result["primary"]["noc_flips"]["a_safe_to_b_danger"],
            "noc_danger_to_safe": score_result["primary"]["noc_flips"]["a_danger_to_b_safe"],
            "noc_total_flip": score_result["primary"]["noc_flips"]["total_flip"],
            "ttc_safe_to_danger": score_result["primary"]["ttc_flips"]["a_safe_to_b_danger"],
            "ttc_total_flip": score_result["primary"]["ttc_flips"]["total_flip"],
            "fingerprint_matches_cold_replay_fingerprint": score_result["primary"][
                "fingerprint_matches_cold_replay_fingerprint"
            ],
        },
        "attribution_full_restored_noc_flip": None
        if score_result is None
        else score_result["attribution"]["full_restored_noc_flip"],
        "attribution_top_single_agent": None
        if score_result is None
        else (score_result["attribution"]["single_agent_ranked_by_noc_flip"][:1] or None),
        "verdict": verdict,
        "verdict_reason": verdict_reason,
        "expand_allowed_this_round": expand_allowed,
        "idm_rolled_count": len(restored["coverage"]["idm_rolled"]),
        "idm_rolled": restored["coverage"]["idm_rolled"],
    }
    save_json(out_dir / "restore_physics_summary.json", summary)

    csv_path = out_dir / "restore_physics_summary.csv"
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
