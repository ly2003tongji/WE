#!/usr/bin/env python3
"""Diagnose IDM agent trajectories and lightweight NOC collision geometry."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from frozen_state_lib import (  # noqa: E402
    REAR_AXLE_TO_CENTER,
    _as_1d,
    future_valid_mask,
    load_scene_dict,
    save_json,
    vocab_plan_to_center_traj,
)


def oriented_box_corners(cx: float, cy: float, heading: float, length: float, width: float) -> np.ndarray:
    """Return 4 corners of oriented box (center-based)."""
    hl, hw = length / 2.0, width / 2.0
    corners_local = np.array([[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]], dtype=np.float64)
    c, s = np.cos(heading), np.sin(heading)
    R = np.array([[c, -s], [s, c]], dtype=np.float64)
    return corners_local @ R.T + np.array([cx, cy], dtype=np.float64)


def sat_polygons_intersect(poly_a: np.ndarray, poly_b: np.ndarray) -> bool:
    """Separating Axis Theorem for two convex polygons (Nx2)."""
    def axes(poly):
        out = []
        n = len(poly)
        for i in range(n):
            e = poly[(i + 1) % n] - poly[i]
            out.append(np.array([-e[1], e[0]], dtype=np.float64))
        return out

    for axis in axes(poly_a) + axes(poly_b):
        nrm = np.linalg.norm(axis)
        if nrm < 1e-12:
            continue
        axis = axis / nrm
        pa = poly_a @ axis
        pb = poly_b @ axis
        if pa.max() < pb.min() or pb.max() < pa.min():
            return False
    return True


def min_center_distance(ego_xy: np.ndarray, agent_xy: np.ndarray) -> float:
    n = min(len(ego_xy), len(agent_xy))
    return float(np.min(np.linalg.norm(ego_xy[:n] - agent_xy[:n], axis=1)))


def diagnose_agent(
    token: str,
    replay_f: Dict[str, Any],
    idm_f: Dict[str, Any],
    scene: Dict[str, Any],
    cutoff: int,
) -> Dict[str, Any]:
    rp = np.asarray(replay_f["position"], dtype=np.float64)
    ip = np.asarray(idm_f["position"], dtype=np.float64)
    rh = _as_1d(np.asarray(replay_f["heading"], dtype=np.float64))
    ih = _as_1d(np.asarray(idm_f["heading"], dtype=np.float64))
    rv = future_valid_mask(replay_f)
    iv = future_valid_mask(idm_f)
    n = min(len(rp), len(ip), len(rh), len(ih), len(rv), len(iv))
    steps = []
    cum_dev = 0.0
    first = {"0.05": None, "1.0": None, "5.0": None}
    for t in range(n):
        dxy = ip[t, :2] - rp[t, :2]
        step_dev = float(np.linalg.norm(dxy))
        if t == 0:
            disp_idm = 0.0
            disp_rep = 0.0
            spd_idm = 0.0
            spd_rep = 0.0
            acc_idm = 0.0
        else:
            disp_idm = float(np.linalg.norm(ip[t, :2] - ip[t - 1, :2]))
            disp_rep = float(np.linalg.norm(rp[t, :2] - rp[t - 1, :2]))
            spd_idm = disp_idm / 0.5
            spd_rep = disp_rep / 0.5
            prev_spd = float(np.linalg.norm(ip[t - 1, :2] - ip[t - 2, :2]) / 0.5) if t >= 2 else spd_idm
            acc_idm = (spd_idm - prev_spd) / 0.5
        if bool(rv[t] and iv[t]):
            cum_dev = step_dev  # instantaneous ADE at t; also track max
        for thr, key in [(0.05, "0.05"), (1.0, "1.0"), (5.0, "5.0")]:
            if first[key] is None and bool(rv[t] and iv[t]) and step_dev > thr:
                first[key] = t
        # motion vs heading consistency for IDM
        motion_heading = None
        heading_motion_diff_deg = None
        if t > 0 and disp_idm > 0.05:
            motion_heading = float(np.arctan2(ip[t, 1] - ip[t - 1, 1], ip[t, 0] - ip[t - 1, 0]))
            dh = (motion_heading - float(ih[t]) + np.pi) % (2 * np.pi) - np.pi
            heading_motion_diff_deg = float(abs(np.rad2deg(dh)))
        steps.append(
            {
                "t": t,
                "abs_step": cutoff + t,
                "replay_xy": rp[t, :2].tolist(),
                "idm_xy": ip[t, :2].tolist(),
                "replay_heading": float(rh[t]),
                "idm_heading": float(ih[t]),
                "replay_valid": bool(rv[t]),
                "idm_valid": bool(iv[t]),
                "common_valid": bool(rv[t] and iv[t]),
                "ade_m": step_dev if (rv[t] and iv[t]) else None,
                "idm_step_disp_m": disp_idm,
                "replay_step_disp_m": disp_rep,
                "idm_speed_from_disp": spd_idm,
                "replay_speed_from_disp": spd_rep,
                "idm_acc_from_disp": acc_idm,
                "heading_motion_diff_deg": heading_motion_diff_deg,
            }
        )

    common = [s for s in steps if s["common_valid"]]
    ades = [s["ade_m"] for s in common if s["ade_m"] is not None]
    idm_disps = [s["idm_step_disp_m"] for s in steps[1:]]
    # consistency checks
    checks = {
        "token_constant": True,
        "common_valid_contiguous_prefix": False,
        "no_teleport_idm_max_step_disp_lt_20m": max(idm_disps) < 20.0 if idm_disps else True,
        "speed_disp_roughly_consistent": True,
        "severe_heading_motion_conflict_count": 0,
    }
    # contiguous common-valid from t0
    cv = [s["common_valid"] for s in steps]
    if any(cv):
        first_false = next((i for i, v in enumerate(cv) if not v), len(cv))
        checks["common_valid_contiguous_prefix"] = all(cv[:first_false]) and (
            first_false == len(cv) or not any(cv[first_false:])
        ) or all(cv)
        # more lenient: no holes in middle of common-valid span
        idxs = [i for i, v in enumerate(cv) if v]
        if idxs:
            checks["common_valid_no_internal_holes"] = idxs == list(range(idxs[0], idxs[-1] + 1))
    for s in steps:
        if s["heading_motion_diff_deg"] is not None and s["heading_motion_diff_deg"] > 90:
            checks["severe_heading_motion_conflict_count"] += 1
    # oscillation detector: alternating step displacement high/low
    osc = False
    if len(idm_disps) >= 4:
        pattern = [d > 0.3 for d in idm_disps]
        flips = sum(pattern[i] != pattern[i + 1] for i in range(len(pattern) - 1))
        osc = flips >= len(pattern) - 2 and (max(idm_disps) < 3.0)
    checks["oscillatory_near_stationary"] = osc

    # classification
    mean_ade = float(np.mean(ades)) if ades else None
    max_ade = float(np.max(ades)) if ades else None
    replay_travel = float(np.linalg.norm(rp[n - 1, :2] - rp[0, :2])) if n > 1 else 0.0
    idm_travel = float(np.linalg.norm(ip[n - 1, :2] - ip[0, :2])) if n > 1 else 0.0

    if checks.get("no_teleport_idm_max_step_disp_lt_20m") is False:
        cause = "non_physical_teleport"
    elif osc and idm_travel < 0.4 * max(replay_travel, 1.0):
        cause = "idm_route_or_control_oscillation_near_stationary"
    elif abs(idm_travel - replay_travel) > 5.0 and max_ade and max_ade > 5.0:
        cause = "path_or_speed_profile_divergence"
    else:
        cause = "moderate_dynamics_difference"

    # scene metadata
    ot = scene["object_track"][token]
    st = ot["state"]
    length = float(np.asarray(st["length"]).reshape(-1)[0])
    width = float(np.asarray(st["width"]).reshape(-1)[0])

    return {
        "token": token,
        "type": ot.get("type"),
        "length": length,
        "width": width,
        "n_steps": n,
        "common_valid_count": len(common),
        "mean_ade_m": mean_ade,
        "max_ade_m": max_ade,
        "first_exceed": first,
        "replay_net_travel_m": replay_travel,
        "idm_net_travel_m": idm_travel,
        "max_idm_step_disp_m": float(max(idm_disps)) if idm_disps else 0.0,
        "checks": checks,
        "likely_cause": cause,
        "steps": steps,
    }


def collision_geometry_samples(
    scene: Dict[str, Any],
    cutoff: int,
    vocab: np.ndarray,
    replay_scores: Dict[str, Any],
    idm_scores: Dict[str, Any],
    agent_token: str,
    replay_fut: Dict[str, Any],
    idm_fut: Dict[str, Any],
    n_each: int = 5,
) -> Dict[str, Any]:
    ego = scene["object_track"]["ego"]["state"]
    ego_xy = np.asarray(ego["position"][cutoff], dtype=np.float64)[:2]
    ego_h = float(_as_1d(np.asarray(ego["heading"], dtype=np.float64))[cutoff])
    ego_len = float(np.asarray(ego["length"]).reshape(-1)[0])
    ego_w = float(np.asarray(ego["width"]).reshape(-1)[0])
    ag = scene["object_track"][agent_token]["state"]
    ag_len = float(np.asarray(ag["length"]).reshape(-1)[0])
    ag_w = float(np.asarray(ag["width"]).reshape(-1)[0])

    noc_r = np.asarray(replay_scores["no_at_fault_collisions"], dtype=np.float64).reshape(-1)
    noc_i = np.asarray(idm_scores["no_at_fault_collisions"], dtype=np.float64).reshape(-1)
    safe_r, safe_i = noc_r >= 1.0, noc_i >= 1.0

    buckets = {
        "replay_safe_idm_danger": np.where(safe_r & ~safe_i)[0],
        "both_safe": np.where(safe_r & safe_i)[0],
        "both_danger": np.where(~safe_r & ~safe_i)[0],
    }

    rp = np.asarray(replay_fut["position"], dtype=np.float64)[:, :2]
    ip = np.asarray(idm_fut["position"], dtype=np.float64)[:, :2]
    rh = _as_1d(np.asarray(replay_fut["heading"], dtype=np.float64))
    ih = _as_1d(np.asarray(idm_fut["heading"], dtype=np.float64))
    rv = future_valid_mask(replay_fut)
    iv = future_valid_mask(idm_fut)

    rng = np.random.default_rng(0)
    out_buckets = {}
    for bname, idxs in buckets.items():
        if len(idxs) == 0:
            out_buckets[bname] = []
            continue
        take = idxs if len(idxs) <= n_each else rng.choice(idxs, size=n_each, replace=False)
        samples = []
        for ci in take:
            center, head = vocab_plan_to_center_traj(vocab[int(ci)], ego_xy, ego_h)
            # align lengths to agent horizon (0.5s)
            T = min(len(center), len(rp), len(ip))
            min_dist_r = []
            min_dist_i = []
            collide_r = False
            collide_i = False
            first_coll_r = None
            first_coll_i = None
            for t in range(T):
                ec = oriented_box_corners(center[t, 0], center[t, 1], float(head[t]), ego_len, ego_w)
                if t < len(rv) and rv[t]:
                    ac = oriented_box_corners(rp[t, 0], rp[t, 1], float(rh[t]), ag_len, ag_w)
                    d = float(np.linalg.norm(center[t, :2] - rp[t, :2]))
                    min_dist_r.append(d)
                    hit = sat_polygons_intersect(ec, ac)
                    if hit and first_coll_r is None:
                        first_coll_r = t
                    collide_r = collide_r or hit
                if t < len(iv) and iv[t]:
                    ac = oriented_box_corners(ip[t, 0], ip[t, 1], float(ih[t]), ag_len, ag_w)
                    d = float(np.linalg.norm(center[t, :2] - ip[t, :2]))
                    min_dist_i.append(d)
                    hit = sat_polygons_intersect(ec, ac)
                    if hit and first_coll_i is None:
                        first_coll_i = t
                    collide_i = collide_i or hit
                    min_center_r = float(min(min_dist_r)) if min_dist_r else None
                    min_center_i = float(min(min_dist_i)) if min_dist_i else None
                    # Use full-horizon min distances for both summary and per-sample consistency.
                    if noc_i[ci] < 1.0:
                        geom_ok = bool(
                            collide_i
                            or (
                                min_center_i is not None
                                and min_center_r is not None
                                and min_center_i < min_center_r - 0.05
                            )
                        )
                    else:
                        geom_ok = True
                    samples.append(
                        {
                            "candidate_idx": int(ci),
                            "noc_replay": float(noc_r[ci]),
                            "noc_idm": float(noc_i[ci]),
                            "min_center_dist_replay_agent_m": min_center_r,
                            "min_center_dist_idm_agent_m": min_center_i,
                            "approx_box_collide_replay": collide_r,
                            "approx_box_collide_idm": collide_i,
                            "first_box_collide_step_replay": first_coll_r,
                            "first_box_collide_step_idm": first_coll_i,
                            "geometry_label_consistent": geom_ok,
                        }
                    )
        out_buckets[bname] = samples

    # summary consistency rates
    consistency = {}
    for bname, samples in out_buckets.items():
        if not samples:
            consistency[bname] = None
            continue
        if bname == "replay_safe_idm_danger":
            # expect IDM closer (full-horizon min) or colliding more often
            ok = 0
            for s in samples:
                if s.get("geometry_label_consistent"):
                    ok += 1
            consistency[bname] = {
                "n": len(samples),
                "idm_closer_or_collide": ok,
                "rate": ok / len(samples),
                "distance_definition": "full_horizon_min_center_distance",
            }
        else:
            consistency[bname] = {"n": len(samples)}

    return {
        "agent_token": agent_token,
        "ego_size": {"length": ego_len, "width": ego_w},
        "agent_size": {"length": ag_len, "width": ag_w},
        "note": (
            "Lightweight center-distance + SAT oriented-box overlap vs vocab ego traj; "
            "not a full PDM at-fault collision replica, but checks geometric plausibility."
        ),
        "buckets": out_buckets,
        "consistency": consistency,
    }


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
    parser.add_argument("--focus-token", type=str, default="44df645d1b5b584b")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    scene = load_scene_dict(args.scene_pkl)
    vocab = np.load(args.vocab)
    with (args.futures_dir / "future_log_replay.pkl").open("rb") as f:
        replay = pickle.load(f)
    with (args.futures_dir / "future_idm.pkl").open("rb") as f:
        idm = pickle.load(f)
    with (args.futures_dir / "scores_log_replay.pkl").open("rb") as f:
        scores_r = pickle.load(f)
    with (args.futures_dir / "scores_idm.pkl").open("rb") as f:
        scores_i = pickle.load(f)

    tokens = list(idm["coverage"]["idm_rolled"])
    agent_reports = {}
    for tok in tokens:
        agent_reports[tok] = diagnose_agent(
            tok, replay["futures"][tok], idm["futures"][tok], scene, args.cutoff
        )

    # Prefer focus token for geometry; also try top single-agent contributor if known later
    geom = collision_geometry_samples(
        scene,
        args.cutoff,
        vocab,
        scores_r,
        scores_i,
        args.focus_token,
        replay["futures"][args.focus_token],
        idm["futures"][args.focus_token],
        n_each=5,
    )

    # Also run geometry for all 4 agents on safe→danger bucket only (3 samples)
    geom_all = {}
    for tok in tokens:
        g = collision_geometry_samples(
            scene, args.cutoff, vocab, scores_r, scores_i, tok,
            replay["futures"][tok], idm["futures"][tok], n_each=3,
        )
        geom_all[tok] = {
            "consistency": g["consistency"],
            "sample_safe_to_danger": g["buckets"]["replay_safe_idm_danger"],
        }

    out = {
        "focus_token": args.focus_token,
        "agents": agent_reports,
        "collision_geometry_focus": geom,
        "collision_geometry_all_idm_agents": geom_all,
    }
    save_json(args.out_dir / "idm_trajectory_and_geometry_diagnosis.json", out)
    # compact print
    for tok, rep in agent_reports.items():
        print(
            tok[:8],
            "max_ade",
            rep["max_ade_m"],
            "cause",
            rep["likely_cause"],
            "osc",
            rep["checks"].get("oscillatory_near_stationary"),
            "travel_r/i",
            round(rep["replay_net_travel_m"], 2),
            round(rep["idm_net_travel_m"], 2),
        )
    print("focus geom consistency", geom["consistency"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
