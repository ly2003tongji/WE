#!/usr/bin/env python3
"""Shared helpers for frozen-state Replay/IDM paired PDM scoring (collaboration layer)."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

REAR_AXLE_TO_CENTER = 1.461
DEFAULT_CUTOFF = 3  # num_history - 1
DEFAULT_HORIZON = 9  # buffer_size / reward_buffer_size
DEFAULT_PLAN_IDX = 1333  # NR Action Policy at step=4 for smoke_1scene
DEFAULT_SEED = 0
SCORER_CONFIG = {
    "num_history": 4,
    "num_future": 8,
    "buffer_size": 9,
    "reward_sampling_poses": 8,
    "reward_buffer_size": 9,
    "proposal_interval_length": 0.1,
    "sim_sample_interval_s": 0.5,
    "sample_rate": 10,
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_json(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=_json_default)
    return sha256_bytes(payload.encode("utf-8"))


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def load_scene_dict(pkl_path: Path) -> Dict[str, Any]:
    data = pickle.load(open(pkl_path, "rb"))
    if isinstance(data, dict):
        if len(data) != 1:
            raise ValueError(f"Expected exactly 1 scene in {pkl_path}, got {len(data)}")
        return copy.deepcopy(next(iter(data.values())))
    if isinstance(data, list):
        if len(data) != 1:
            raise ValueError(f"Expected exactly 1 scene list entry in {pkl_path}, got {len(data)}")
        return copy.deepcopy(data[0])
    raise TypeError(f"Unsupported scenario pickle type: {type(data)}")


def scene_as_dict(scene: Dict[str, Any]) -> Dict[str, Any]:
    return {scene["id"]: scene}


def _as_1d(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr)
    if a.ndim == 2 and a.shape[1] == 1:
        return a[:, 0]
    return a


def agent_valid_at(state: Dict[str, Any], step: int) -> bool:
    """Validity at one step. Prefer explicit `valid` when present; never infer from [0,0] alone."""
    pos = np.asarray(state["position"])
    if step >= len(pos):
        return False
    if "valid" in state:
        v = _as_1d(np.asarray(state["valid"]))
        if step >= len(v):
            return False
        return float(v[step]) > 0
    # Heuristic only when valid field is absent.
    return not np.allclose(pos[step, :2], 0.0)


def future_valid_mask(fut: Dict[str, Any]) -> np.ndarray:
    """Per-timestep valid mask for a future dict (prefer explicit valid)."""
    pos = np.asarray(fut["position"])
    n = len(pos)
    if "valid" in fut:
        return (_as_1d(np.asarray(fut["valid"], dtype=np.float64))[:n] > 0)
    return ~np.all(np.isclose(pos[:n, :2], 0.0), axis=1)


def sorted_agent_tokens(scene: Dict[str, Any], include_ego: bool = False) -> List[str]:
    sdc = scene["sdc_id"]
    tokens = [t for t in scene["object_track"].keys() if include_ego or t != sdc]
    return sorted(tokens)


def extract_frozen_agent_snapshot(scene: Dict[str, Any], cutoff: int) -> Dict[str, Any]:
    """History+current only (indices 0..cutoff). No futures beyond cutoff."""
    agents = {}
    for token in sorted_agent_tokens(scene, include_ego=True):
        ot = scene["object_track"][token]
        st = ot["state"]
        pos = np.asarray(st["position"], dtype=np.float64)[: cutoff + 1]
        heading = _as_1d(np.asarray(st["heading"], dtype=np.float64))[: cutoff + 1]
        vel = np.asarray(st.get("velocity", np.zeros_like(pos[:, :2])), dtype=np.float64)[: cutoff + 1]
        valid = _as_1d(np.asarray(st.get("valid", np.ones(len(pos))), dtype=np.float64))[: cutoff + 1]
        length = float(np.asarray(st.get("length", [[0.0]]))[0].reshape(-1)[0])
        width = float(np.asarray(st.get("width", [[0.0]]))[0].reshape(-1)[0])
        height = float(np.asarray(st.get("height", [[0.0]]))[0].reshape(-1)[0])
        agents[token] = {
            "type": ot.get("type"),
            "position": pos.tolist(),
            "heading": heading.tolist(),
            "velocity": vel.tolist(),
            "valid": valid.tolist(),
            "length": length,
            "width": width,
            "height": height,
            "valid_at_cutoff": bool(agent_valid_at(st, cutoff)),
        }
    return agents


def extract_map_fingerprint_payload(scene: Dict[str, Any]) -> Dict[str, Any]:
    mf = scene.get("map_features", {})
    lane_ids = sorted(str(k) for k in (mf.get("lane", {}) or {}).keys())
    roadblock_ids = sorted(str(k) for k in (mf.get("roadblock", {}) or {}).keys())
    # keep counts + short hashes of id lists to avoid huge payloads
    return {
        "map": scene.get("map"),
        "dataset": scene.get("dataset"),
        "n_lane_ids": len(lane_ids),
        "n_roadblock_ids": len(roadblock_ids),
        "lane_ids_sha256": sha256_bytes(",".join(lane_ids).encode("utf-8")),
        "roadblock_ids_sha256": sha256_bytes(",".join(roadblock_ids).encode("utf-8")),
        "map_feature_keys": sorted(mf.keys()) if isinstance(mf, dict) else [],
    }


def extract_traffic_light_payload(scene: Dict[str, Any], cutoff: int) -> Dict[str, Any]:
    dms = scene.get("dynamic_map_states", {})
    if not dms:
        return {"n": 0, "states_at_cutoff": []}
    items = []
    if isinstance(dms, dict):
        for tid in sorted(dms.keys()):
            entry = dms[tid]
            state = entry.get("traffic_light_state") or entry.get("state") or entry
            if isinstance(state, (list, tuple, np.ndarray)):
                arr = list(state)
                val = arr[cutoff] if cutoff < len(arr) else None
            else:
                val = state
            items.append({"id": str(tid), "state_at_cutoff": str(val)})
    elif isinstance(dms, list):
        for i, entry in enumerate(dms):
            items.append({"id": str(i), "raw_type": type(entry).__name__})
    return {"n": len(items), "states_at_cutoff": items}


def vocab_meta(vocab_path: Path, vocab: Optional[np.ndarray] = None) -> Dict[str, Any]:
    arr = vocab if vocab is not None else np.load(vocab_path)
    return {
        "path": str(vocab_path),
        "sha256": sha256_file(vocab_path),
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
    }


def extract_scorer_config_from_hydra(cfg: Any) -> Dict[str, Any]:
    """Pull scorer-relevant knobs from an actual Hydra compose result."""
    keys = [
        "num_history",
        "num_future",
        "buffer_size",
        "reward_sampling_poses",
        "reward_buffer_size",
        "sample_rate",
        "frame_rate",
        "planning_frame_rate",
        "with_dense_reward_manager",
        "agent_policy",
        "ego_policy",
    ]
    out: Dict[str, Any] = {}
    for k in keys:
        if k in cfg:
            out[k] = cfg[k]
    out["proposal_interval_length"] = 0.1
    out["sim_sample_interval_s"] = float(cfg.get("frame_rate", 0.5))
    out["scorer_config_hash"] = sha256_json(out)
    return out


def build_input_state_fingerprint(
    scene: Dict[str, Any],
    cutoff: int,
    vocab_path: Path,
    plan_idx: int,
    scorer_config: Optional[Dict[str, Any]] = None,
    horizon: int = DEFAULT_HORIZON,
    requested_ego_traj_hash: Optional[str] = None,
    vocab: Optional[np.ndarray] = None,
    fingerprint_label: str = "",
) -> Dict[str, Any]:
    """Fingerprint of frozen inputs. Does NOT include traffic-model futures."""
    scorer_config = scorer_config or dict(SCORER_CONFIG)
    agents = extract_frozen_agent_snapshot(scene, cutoff)
    ego = agents[scene["sdc_id"]]
    vmeta = vocab_meta(vocab_path, vocab=vocab)
    payload = {
        "scene_id": scene.get("id"),
        "scene_token": scene.get("token"),
        "cutoff": cutoff,
        "horizon": horizon,
        "frequency_hz": 2.0,
        "dt_s": 0.5,
        "sample_rate": scene.get("sample_rate"),
        "log_length": scene.get("log_length"),
        "base_timestamp": scene.get("base_timestamp"),
        "old_origin_in_current_coordinate": np.asarray(
            scene.get("metadata", {}).get("old_origin_in_current_coordinate", [])
        ).tolist(),
        "ego": ego,
        "agents_sorted": {k: v for k, v in agents.items() if k != scene["sdc_id"]},
        "map": extract_map_fingerprint_payload(scene),
        "traffic_lights": extract_traffic_light_payload(scene, cutoff),
        "vocabulary": vmeta,
        "ego_conditioning": {
            "plan_idx": int(plan_idx),
            "requested_trajectory_hash": requested_ego_traj_hash,
        },
        "scorer_config": scorer_config,
    }
    fp = sha256_json(payload)
    return {
        "input_state_fingerprint": fp,
        "payload": payload,
        "fingerprint_label": fingerprint_label,
    }


def resolve_plan_idx(
    plan_idx: Optional[int],
    plan_idx_csv: Optional[Path],
    plan_idx_step: Optional[int],
    cutoff: int,
    default_plan_idx: int = DEFAULT_PLAN_IDX,
) -> Dict[str, Any]:
    """Resolve plan_idx from CLI or CSV. Documents smoke-default provenance."""
    if plan_idx is not None and plan_idx_csv is None:
        return {
            "plan_idx": int(plan_idx),
            "source": "cli --plan-idx",
            "plan_idx_step": plan_idx_step,
            "note": "Explicit CLI value.",
        }
    if plan_idx_csv is not None:
        import csv as _csv

        rows = []
        with open(plan_idx_csv, newline="", encoding="utf-8") as f:
            for row in _csv.DictReader(f):
                rows.append(row)
        if not rows:
            raise ValueError(f"Empty plan_idx csv: {plan_idx_csv}")
        step = plan_idx_step if plan_idx_step is not None else (cutoff + 1)
        match = [r for r in rows if int(r["step"]) == int(step)]
        if not match:
            raise ValueError(
                f"No plan_idx for step={step} in {plan_idx_csv}; "
                f"available steps={[int(r['step']) for r in rows[:12]]}"
            )
        return {
            "plan_idx": int(match[0]["plan_idx"]),
            "source": f"plan_idx.csv step={step}",
            "plan_idx_step": int(step),
            "csv": str(plan_idx_csv),
            "note": (
                f"Action Policy plan at step={step} used as ego conditioning for cutoff={cutoff} "
                f"(first executed plan after open-loop history; densereward first scores at cutoff)."
            ),
        }
    # smoke convenience default
    return {
        "plan_idx": int(default_plan_idx),
        "source": "smoke_default_DEFAULT_PLAN_IDX",
        "plan_idx_step": cutoff + 1,
        "note": (
            f"SMOKE CONVENIENCE: default plan_idx={default_plan_idx} "
            f"(NR disagreement smoke1 first action at step={cutoff + 1}). "
            "Prefer --plan-idx-csv for multi-scene."
        ),
        "warning": "plan_idx came from hardcoded smoke default",
    }


def verify_ego_conditioning_execution(
    requested_center: np.ndarray,
    requested_heading: np.ndarray,
    live_positions: List[np.ndarray],
    live_headings: List[float],
    dt_s: float = 0.5,
    warn_pos_m: float = 0.1,
    warn_heading_deg: float = 1.0,
    hard_pos_m: float = 0.5,
    hard_heading_deg: float = 5.0,
) -> Dict[str, Any]:
    """Compare requested ego conditioning vs live IDM-sim ego trajectory."""
    n = min(len(requested_center), len(live_positions), len(requested_heading), len(live_headings))
    req_xy = np.asarray(requested_center[:n], dtype=np.float64)[:, :2]
    live_xy = np.stack([np.asarray(p, dtype=np.float64).reshape(-1)[:2] for p in live_positions[:n]])
    req_h = np.asarray(requested_heading[:n], dtype=np.float64).reshape(-1)
    live_h = np.asarray(live_headings[:n], dtype=np.float64).reshape(-1)
    pos_err = np.linalg.norm(req_xy - live_xy, axis=1)
    # wrap heading error to [-pi, pi]
    head_err = (live_h - req_h + np.pi) % (2 * np.pi) - np.pi
    head_err_deg = np.rad2deg(np.abs(head_err))
    per_step = []
    for t in range(n):
        per_step.append(
            {
                "t": t,
                "time_s": float(t * dt_s),
                "requested_xy": req_xy[t].tolist(),
                "executed_xy": live_xy[t].tolist(),
                "pos_error_m": float(pos_err[t]),
                "requested_heading": float(req_h[t]),
                "executed_heading": float(live_h[t]),
                "heading_error_deg": float(head_err_deg[t]),
            }
        )
    requested_hash = sha256_bytes(np.ascontiguousarray(req_xy).tobytes() + np.ascontiguousarray(req_h).tobytes())
    executed_hash = sha256_bytes(np.ascontiguousarray(live_xy).tobytes() + np.ascontiguousarray(live_h).tobytes())
    warnings = []
    if float(np.max(pos_err)) > warn_pos_m:
        warnings.append(f"pos_max_error {float(np.max(pos_err)):.4f}m > {warn_pos_m}m")
    if float(np.max(head_err_deg)) > warn_heading_deg:
        warnings.append(f"heading_max_error {float(np.max(head_err_deg)):.4f}deg > {warn_heading_deg}deg")
    hard_fail = bool(float(np.max(pos_err)) > hard_pos_m or float(np.max(head_err_deg)) > hard_heading_deg)
    return {
        "n_steps": n,
        "dt_s": dt_s,
        "requested_ego_conditioning_hash": requested_hash,
        "executed_ego_conditioning_hash": executed_hash,
        "hashes_equal": requested_hash == executed_hash,
        "pos_error_mean_m": float(np.mean(pos_err)),
        "pos_error_max_m": float(np.max(pos_err)),
        "heading_error_mean_deg": float(np.mean(head_err_deg)),
        "heading_error_max_deg": float(np.max(head_err_deg)),
        "per_step": per_step,
        "warnings": warnings,
        "hard_fail": hard_fail,
        "hard_thresholds": {"pos_m": hard_pos_m, "heading_deg": hard_heading_deg},
        "warn_thresholds": {"pos_m": warn_pos_m, "heading_deg": warn_heading_deg},
    }


def vocab_plan_to_center_traj(
    vocab_traj: np.ndarray,
    ego_center_xy: np.ndarray,
    ego_heading: float,
    rear_axle_to_center: float = REAR_AXLE_TO_CENTER,
) -> Tuple[np.ndarray, np.ndarray]:
    """Map vocab (40,3) local rear-axle 0.1s poses -> WE center poses at 0.5s including t0."""
    raw = np.asarray(vocab_traj, dtype=np.float64)
    local = np.concatenate([np.zeros((1, 3), dtype=np.float64), raw], axis=0)
    sampled = local[::5]  # 0.5s
    ego_h = float(ego_heading)
    rear_xy = np.asarray(ego_center_xy, dtype=np.float64)[:2] - rear_axle_to_center * np.array(
        [np.cos(ego_h), np.sin(ego_h)]
    )
    R = np.array(
        [[np.cos(ego_h), -np.sin(ego_h)], [np.sin(ego_h), np.cos(ego_h)]],
        dtype=np.float64,
    )
    rear_traj = sampled[:, :2] @ R.T + rear_xy
    head_traj = sampled[:, 2] + ego_h
    center_traj = rear_traj + rear_axle_to_center * np.stack(
        [np.cos(head_traj), np.sin(head_traj)], axis=1
    )
    return center_traj, head_traj


def inject_ego_conditioning(
    scene: Dict[str, Any],
    plan_idx: int,
    vocab: np.ndarray,
    rear_axle_to_center: float = REAR_AXLE_TO_CENTER,
) -> Dict[str, Any]:
    """Overwrite ego future (and keep t0) with vocab plan in WE coordinates."""
    ego = scene["object_track"]["ego"]
    st = ego["state"]
    pos = np.asarray(st["position"], dtype=np.float64).copy()
    heading = np.asarray(st["heading"], dtype=np.float64).copy()
    ego_xy = pos[0, :2]
    ego_h = float(heading[0]) if heading.ndim == 1 else float(heading[0, 0])
    center_traj, head_traj = vocab_plan_to_center_traj(
        vocab[plan_idx], ego_xy, ego_h, rear_axle_to_center=rear_axle_to_center
    )
    n = min(len(pos), len(center_traj))
    pos[:n, :2] = center_traj[:n]
    if heading.ndim == 1:
        heading[:n] = head_traj[:n]
    else:
        heading[:n, 0] = head_traj[:n]
    vel = np.asarray(st["velocity"], dtype=np.float64).copy()
    for i in range(1, n):
        d = (center_traj[i] - center_traj[i - 1]) / 0.5
        if vel.ndim == 2 and vel.shape[1] >= 2:
            vel[i, :2] = d
        else:
            vel[i] = np.linalg.norm(d)
    if "angular_velocity" in st:
        av = np.asarray(st["angular_velocity"], dtype=np.float64).copy()
        for i in range(1, n):
            av[i] = (head_traj[i] - head_traj[i - 1]) / 0.5
        st["angular_velocity"] = av
    valid = np.asarray(st.get("valid", np.ones(len(pos))))
    if valid.ndim == 1:
        valid[:n] = 1
    else:
        valid[:n, ...] = 1
    st["position"] = pos
    st["heading"] = heading
    st["velocity"] = vel
    st["valid"] = valid
    return {
        "plan_idx": int(plan_idx),
        "requested_ego_conditioning_hash": sha256_bytes(
            np.ascontiguousarray(center_traj[:n]).tobytes()
            + np.ascontiguousarray(head_traj[:n]).tobytes()
        ),
        # backward-compatible alias
        "ego_conditioning_hash": sha256_bytes(np.ascontiguousarray(center_traj[:n]).tobytes()),
        "ego_conditioning_shape": list(center_traj[:n].shape),
        "n_poses": int(n),
        "requested_center_traj": center_traj[:n].copy(),
        "requested_heading_traj": head_traj[:n].copy(),
    }


def slice_scene_from_cutoff(scene: Dict[str, Any], cutoff: int, length: int) -> Dict[str, Any]:
    """Create a short scene whose index 0 == original cutoff."""
    sc = copy.deepcopy(scene)
    sc["log_length"] = int(length)
    for oid, ot in sc["object_track"].items():
        st = ot["state"]
        for key, val in list(st.items()):
            arr = np.asarray(val)
            if arr.ndim >= 1 and arr.shape[0] >= cutoff + length:
                st[key] = arr[cutoff : cutoff + length].copy()
            elif arr.ndim >= 1 and arr.shape[0] > cutoff:
                sl = arr[cutoff:].copy()
                if sl.shape[0] < length:
                    pad_shape = (length - sl.shape[0],) + sl.shape[1:]
                    st[key] = np.concatenate([sl, np.zeros(pad_shape, dtype=sl.dtype)], axis=0)
                else:
                    st[key] = sl[:length]
            # scalar-like length/width/height arrays of shape (T,1) already handled
        if isinstance(ot.get("metadata"), dict):
            ot["metadata"]["track_length"] = int(length)
    # truncate lidar token lists if present (best-effort)
    meta = sc.get("metadata", {})
    for mk in ("nuplan_lidar_pc_tokens", "digitaltwin_ego2globals"):
        if mk in meta:
            try:
                seq = meta[mk]
                if hasattr(seq, "__len__") and len(seq) >= cutoff + length:
                    meta[mk] = seq[cutoff : cutoff + length]
            except Exception:
                pass
    return sc


def extract_replay_futures(
    scene: Dict[str, Any],
    cutoff: int,
    horizon: int,
) -> Dict[str, Any]:
    """Log futures from cutoff inclusive for horizon frames."""
    futures: Dict[str, Any] = {}
    coverage = {"idm_capable_vehicle": [], "log_fallback": [], "invalid_at_cutoff": []}
    for token in sorted_agent_tokens(scene, include_ego=False):
        ot = scene["object_track"][token]
        st = ot["state"]
        pos = np.asarray(st["position"], dtype=np.float64)
        heading = _as_1d(np.asarray(st["heading"], dtype=np.float64))
        vel = np.asarray(st.get("velocity", np.zeros((len(pos), 2))), dtype=np.float64)
        valid = _as_1d(np.asarray(st.get("valid", np.ones(len(pos))), dtype=np.float64))
        end = min(cutoff + horizon, len(pos))
        if not agent_valid_at(st, cutoff):
            coverage["invalid_at_cutoff"].append(token)
            continue
        sl = slice(cutoff, end)
        futures[token] = {
            "type": ot.get("type"),
            "position": pos[sl].copy(),
            "heading": heading[sl].copy(),
            "velocity": vel[sl].copy() if len(vel) >= end else vel[cutoff:].copy(),
            "valid": valid[sl].copy() if len(valid) >= end else valid[cutoff:].copy(),
            "source": "log_replay",
        }
        if ot.get("type") == "VEHICLE":
            coverage["idm_capable_vehicle"].append(token)
        else:
            coverage["log_fallback"].append(token)
    flat = _futures_flat_bytes(futures)
    return {
        "source": "log_replay",
        "cutoff": cutoff,
        "horizon": horizon,
        "frequency_hz": 2.0,
        "dt_s": 0.5,
        "n_agents": len(futures),
        "futures": futures,
        "coverage": coverage,
        "future_hash": sha256_bytes(flat),
        "future_shape_note": "per-agent (horizon<=9, 2) position + heading",
    }


def _futures_flat_bytes(futures: Dict[str, Any]) -> bytes:
    parts = []
    for token in sorted(futures.keys()):
        f = futures[token]
        parts.append(token.encode("utf-8"))
        parts.append(np.ascontiguousarray(f["position"]).tobytes())
        parts.append(np.ascontiguousarray(f["heading"]).tobytes())
    return b"".join(parts)


def inject_agent_futures_into_scene(
    scene: Dict[str, Any],
    futures_pack: Dict[str, Any],
    cutoff: int,
) -> Dict[str, Any]:
    """Write futures into object_track starting at cutoff (inclusive). History before cutoff untouched."""
    sc = copy.deepcopy(scene)
    futures = futures_pack["futures"]
    for token, f in futures.items():
        if token not in sc["object_track"]:
            continue
        st = sc["object_track"][token]["state"]
        pos = np.asarray(st["position"], dtype=np.float64).copy()
        heading = np.asarray(st["heading"], dtype=np.float64).copy()
        vel = np.asarray(st["velocity"], dtype=np.float64).copy()
        valid = np.asarray(st["valid"], dtype=np.float64).copy()
        fpos = np.asarray(f["position"], dtype=np.float64)
        fhead = _as_1d(np.asarray(f["heading"], dtype=np.float64))
        fvel = np.asarray(f.get("velocity", np.zeros_like(fpos[:, :2] if fpos.ndim == 2 else fpos)), dtype=np.float64)
        fvalid = _as_1d(np.asarray(f.get("valid", np.ones(len(fpos))), dtype=np.float64))
        n = len(fpos)
        end = min(cutoff + n, len(pos))
        n_write = end - cutoff
        pos[cutoff:end, :2] = fpos[:n_write, :2]
        if heading.ndim == 1:
            heading[cutoff:end] = fhead[:n_write]
        else:
            heading[cutoff:end, 0] = fhead[:n_write]
        if vel.ndim == 2 and vel.shape[1] >= 2 and fvel.ndim == 2 and fvel.shape[1] >= 2:
            vel[cutoff:end, :2] = fvel[:n_write, :2]
        if valid.ndim == 1:
            valid[cutoff:end] = fvalid[:n_write]
        else:
            valid[cutoff:end, 0] = fvalid[:n_write]
        st["position"] = pos
        st["heading"] = heading
        st["velocity"] = vel
        st["valid"] = valid
    return sc


def compare_futures(
    replay: Dict[str, Any],
    idm: Dict[str, Any],
    fork_thresh_m: float = 0.05,
) -> Dict[str, Any]:
    """Valid-aware ADE (formal) + unmasked ADE (legacy, for artifact diagnosis)."""
    r_f = replay["futures"]
    i_f = idm["futures"]
    common = sorted(set(r_f.keys()) & set(i_f.keys()))
    only_r = sorted(set(r_f.keys()) - set(i_f.keys()))
    only_i = sorted(set(i_f.keys()) - set(r_f.keys()))

    per_agent = []
    masked_dists_all: List[float] = []
    unmasked_dists_all: List[float] = []
    changed_masked = []
    changed_unmasked = []

    for tok in common:
        rp = np.asarray(r_f[tok]["position"], dtype=np.float64)[:, :2]
        ip = np.asarray(i_f[tok]["position"], dtype=np.float64)[:, :2]
        n = min(len(rp), len(ip))
        if n == 0:
            continue
        rv = future_valid_mask(r_f[tok])[:n]
        iv = future_valid_mask(i_f[tok])[:n]
        # pad masks if short
        if len(rv) < n:
            rv = np.pad(rv, (0, n - len(rv)), constant_values=False)
        if len(iv) < n:
            iv = np.pad(iv, (0, n - len(iv)), constant_values=False)
        common_valid = rv & iv
        d_all = np.linalg.norm(rp[:n] - ip[:n], axis=1)
        unmasked_mean = float(np.mean(d_all))
        unmasked_max = float(np.max(d_all))
        unmasked_dists_all.extend(d_all.tolist())

        entry: Dict[str, Any] = {
            "token": tok,
            "type": r_f[tok].get("type"),
            "source_r": r_f[tok].get("source"),
            "source_i": i_f[tok].get("source"),
            "replay_valid_count": int(np.sum(rv)),
            "idm_valid_count": int(np.sum(iv)),
            "common_valid_count": int(np.sum(common_valid)),
            "unmasked_mean_ade_m": unmasked_mean,
            "unmasked_max_ade_m": unmasked_max,
            "first_fork_step_unmasked": int(np.argmax(d_all > fork_thresh_m)) if np.any(d_all > fork_thresh_m) else None,
        }
        if int(np.sum(common_valid)) == 0:
            entry["common_valid_mean_ade_m"] = None
            entry["common_valid_max_ade_m"] = None
            entry["first_fork_step_common_valid"] = None
            entry["excluded_from_formal_ade"] = True
        else:
            d_m = d_all[common_valid]
            masked_dists_all.extend(d_m.tolist())
            mean_m = float(np.mean(d_m))
            max_m = float(np.max(d_m))
            entry["common_valid_mean_ade_m"] = mean_m
            entry["common_valid_max_ade_m"] = max_m
            entry["excluded_from_formal_ade"] = False
            # first fork among common-valid steps
            idxs = np.where(common_valid)[0]
            fork_local = np.where(d_m > fork_thresh_m)[0]
            entry["first_fork_step_common_valid"] = int(idxs[fork_local[0]]) if len(fork_local) else None
            if max_m > fork_thresh_m:
                changed_masked.append(
                    {
                        "token": tok,
                        "type": entry["type"],
                        "mean_ade": mean_m,
                        "max_ade": max_m,
                        "source_r": entry["source_r"],
                        "source_i": entry["source_i"],
                        "common_valid_count": entry["common_valid_count"],
                    }
                )
        if unmasked_max > fork_thresh_m:
            changed_unmasked.append(
                {
                    "token": tok,
                    "type": entry["type"],
                    "mean_ade": unmasked_mean,
                    "max_ade": unmasked_max,
                    "source_r": entry["source_r"],
                    "source_i": entry["source_i"],
                }
            )
        per_agent.append(entry)

    changed_masked.sort(key=lambda x: -x["max_ade"])
    changed_unmasked.sort(key=lambda x: -x["max_ade"])
    per_agent.sort(key=lambda x: -(x["common_valid_max_ade_m"] or -1))

    # Diagnose 73.4m-style artifacts
    artifact_notes = []
    for e in per_agent:
        if (e["unmasked_max_ade_m"] or 0) > 50 and (
            e["common_valid_max_ade_m"] is None or e["common_valid_max_ade_m"] < e["unmasked_max_ade_m"] * 0.5
        ):
            artifact_notes.append(
                {
                    "token": e["token"],
                    "unmasked_max_ade_m": e["unmasked_max_ade_m"],
                    "common_valid_max_ade_m": e["common_valid_max_ade_m"],
                    "replay_valid_count": e["replay_valid_count"],
                    "idm_valid_count": e["idm_valid_count"],
                    "likely_cause": "unmasked_ade_includes_invalid_or_padded_timesteps",
                }
            )

    return {
        "futures_equal": replay["future_hash"] == idm["future_hash"],
        "replay_future_hash": replay["future_hash"],
        "idm_future_hash": idm["future_hash"],
        "n_common_agents": len(common),
        "only_replay": only_r,
        "only_idm": only_i,
        # Formal (valid-aware)
        "n_changed_agents_common_valid_maxade_gt_5cm": len(changed_masked),
        "changed_agents_common_valid": changed_masked,
        "traj_dev_common_valid_mean_m": float(np.mean(masked_dists_all)) if masked_dists_all else 0.0,
        "traj_dev_common_valid_max_m": float(np.max(masked_dists_all)) if masked_dists_all else 0.0,
        "n_agents_in_formal_ade": int(sum(1 for e in per_agent if not e["excluded_from_formal_ade"])),
        # Legacy unmasked (diagnostic only)
        "legacy_unmasked": {
            "note": "NOT a formal metric; retained to explain stage-1.5 73.4m artifact",
            "n_changed_agents_maxade_gt_5cm": len(changed_unmasked),
            "changed_agents": changed_unmasked,
            "traj_dev_mean_m": float(np.mean(unmasked_dists_all)) if unmasked_dists_all else 0.0,
            "traj_dev_max_m": float(np.max(unmasked_dists_all)) if unmasked_dists_all else 0.0,
        },
        # aliases for backward-looking summaries
        "n_changed_agents_maxade_gt_5cm": len(changed_masked),
        "changed_agents": changed_masked,
        "traj_dev_mean_m": float(np.mean(masked_dists_all)) if masked_dists_all else 0.0,
        "traj_dev_max_m": float(np.max(masked_dists_all)) if masked_dists_all else 0.0,
        "per_agent": per_agent,
        "artifact_diagnosis": artifact_notes,
        "coverage_replay": replay.get("coverage"),
        "coverage_idm": idm.get("coverage"),
    }


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_json_default)


def futures_to_numpy_bundle(futures_pack: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Flatten futures to arrays for npz (no huge nested pickles)."""
    tokens = sorted(futures_pack["futures"].keys())
    # pad to horizon
    horizon = int(futures_pack["horizon"])
    pos = np.zeros((len(tokens), horizon, 3), dtype=np.float64)
    heading = np.zeros((len(tokens), horizon), dtype=np.float64)
    valid = np.zeros((len(tokens), horizon), dtype=np.float64)
    types = []
    sources = []
    for i, tok in enumerate(tokens):
        f = futures_pack["futures"][tok]
        p = np.asarray(f["position"], dtype=np.float64)
        h = _as_1d(np.asarray(f["heading"], dtype=np.float64))
        v = _as_1d(np.asarray(f.get("valid", np.ones(len(p))), dtype=np.float64))
        n = min(horizon, len(p))
        pos[i, :n, : p.shape[1]] = p[:n]
        heading[i, :n] = h[:n]
        valid[i, :n] = v[:n]
        types.append(str(f.get("type")))
        sources.append(str(f.get("source")))
    return {
        "tokens": np.asarray(tokens),
        "types": np.asarray(types),
        "sources": np.asarray(sources),
        "position": pos,
        "heading": heading,
        "valid": valid,
    }
