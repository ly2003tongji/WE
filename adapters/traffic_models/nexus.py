"""Nexus sidecar adapter: WorldEngine frozen state <-> SceneTensor <-> future pack.

Uses official Nexus encode/decode/normalizer when importable. Does not modify
Nexus or WorldEngine upstream. Non-ego future slots are generation placeholders
and must not carry log GT (future-leakage gate).
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Pacifica rear-axle offset used by nuPlan / Nexus ego encoding.
REAR_AXLE_TO_CENTER = 1.461
N_PAST = 5  # indices 0..4
N_FUTURE = 16  # indices 5..20
N_TOTAL = N_PAST + N_FUTURE  # 21
N_FEATURES = 8
PDM_HORIZON = 9  # cutoff current + 8 futures

# Checkpoint training config (nexus.yaml)
DEFAULT_MAP_FEATURES = ["LANE", "LANE_CONNECTOR", "STOP_LINE", "CROSSWALK"]
DEFAULT_NUM_MAX_AGENTS = [128, 0, 0]  # VEHICLE / PED / BIKE
DEFAULT_MAP_NUM_MAX_POLYLINES = 128
DEFAULT_MAP_NUM_POINTS = 21
DEFAULT_MAP_DIM = 7
DEFAULT_RADIUS_M = 100.0

# Official FEATURE_MEANS/STD (must match Nexus scene_tensor.py)
FEATURE_MEANS = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.5, 2.0], dtype=np.float64)
FEATURE_STD = np.array([52.0, 52.0, 0.5, 0.5, 2.0, 2.0, 2.5, 0.8], dtype=np.float64)

# Pacifica defaults used by official encode_ego
EGO_LENGTH_PACIFICA = 5.176624298095703  # will be overwritten from Nexus if available
EGO_WIDTH_PACIFICA = 2.2978200912475586

# WE OpenScene-ish type → Nexus map feature
WE_TO_NEXUS_MAP = {
    "LANE_SURFACE_STREET": "LANE",
    "LANE_SURFACE_UNSTRUCTURE": "LANE",  # best-effort; reported in coverage
    "CROSSWALK": "CROSSWALK",
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_array(arr: np.ndarray) -> str:
    a = np.ascontiguousarray(arr)
    return sha256_bytes(a.tobytes() + str(a.shape).encode() + str(a.dtype).encode())


def sha256_json(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)
    return sha256_bytes(payload.encode("utf-8"))


def encode_scene_tensor_np(raw: np.ndarray) -> np.ndarray:
    return (raw - FEATURE_MEANS[: raw.shape[-1]]) / (2.0 * FEATURE_STD[: raw.shape[-1]])


def decode_scene_tensor_np(normed: np.ndarray) -> np.ndarray:
    return normed * 2.0 * FEATURE_STD[: normed.shape[-1]] + FEATURE_MEANS[: normed.shape[-1]]


def try_import_official_codec():
    """Import official encode/decode if Nexus is on PYTHONPATH."""
    try:
        from nuplan_extent.planning.training.preprocessing.features.scene_tensor import (
            FEATURE_MEANS as FM,
            FEATURE_STD as FS,
            EGO_LENGTH,
            EGO_WIDTH,
            decode_scene_tensor,
            encode_scene_tensor,
            N_SCENE_TENSOR_FEATURES,
        )

        return {
            "ok": True,
            "FEATURE_MEANS": np.asarray(FM, dtype=np.float64),
            "FEATURE_STD": np.asarray(FS, dtype=np.float64),
            "EGO_LENGTH": float(EGO_LENGTH),
            "EGO_WIDTH": float(EGO_WIDTH),
            "decode": decode_scene_tensor,
            "encode": encode_scene_tensor,
            "N_SCENE_TENSOR_FEATURES": int(N_SCENE_TENSOR_FEATURES),
        }
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": repr(e)}


@dataclass
class FreezeFrame:
    """SE(2) freeze frame at cutoff: origin = ego rear axle, x = ego heading."""

    origin_xy: np.ndarray  # (2,) world rear-axle
    heading: float
    cos_h: float
    sin_h: float

    def world_to_local_xy(self, xy: np.ndarray) -> np.ndarray:
        d = np.asarray(xy, dtype=np.float64)[..., :2] - self.origin_xy
        # R^T
        x = self.cos_h * d[..., 0] + self.sin_h * d[..., 1]
        y = -self.sin_h * d[..., 0] + self.cos_h * d[..., 1]
        return np.stack([x, y], axis=-1)

    def local_to_world_xy(self, xy: np.ndarray) -> np.ndarray:
        x = np.asarray(xy, dtype=np.float64)[..., 0]
        y = np.asarray(xy, dtype=np.float64)[..., 1]
        wx = self.cos_h * x - self.sin_h * y + self.origin_xy[0]
        wy = self.sin_h * x + self.cos_h * y + self.origin_xy[1]
        return np.stack([wx, wy], axis=-1)

    def world_to_local_heading(self, h: np.ndarray) -> np.ndarray:
        return np.asarray(h, dtype=np.float64) - self.heading

    def local_to_world_heading(self, h: np.ndarray) -> np.ndarray:
        return np.asarray(h, dtype=np.float64) + self.heading

    def world_to_local_vel(self, vel: np.ndarray) -> np.ndarray:
        v = np.asarray(vel, dtype=np.float64)[..., :2]
        vx = self.cos_h * v[..., 0] + self.sin_h * v[..., 1]
        vy = -self.sin_h * v[..., 0] + self.cos_h * v[..., 1]
        return np.stack([vx, vy], axis=-1)

    def local_to_world_vel(self, vel: np.ndarray) -> np.ndarray:
        v = np.asarray(vel, dtype=np.float64)[..., :2]
        wx = self.cos_h * v[..., 0] - self.sin_h * v[..., 1]
        wy = self.sin_h * v[..., 0] + self.cos_h * v[..., 1]
        return np.stack([wx, wy], axis=-1)


def build_freeze_frame(scene: Dict[str, Any], cutoff: int) -> FreezeFrame:
    sdc = scene["sdc_id"]
    st = scene["object_track"][sdc]["state"]
    pos = np.asarray(st["position"], dtype=np.float64)
    heading = float(np.asarray(st["heading"]).reshape(-1)[cutoff])
    center = pos[cutoff, :2]
    rear = center - REAR_AXLE_TO_CENTER * np.array([math.cos(heading), math.sin(heading)])
    return FreezeFrame(
        origin_xy=rear,
        heading=heading,
        cos_h=math.cos(heading),
        sin_h=math.sin(heading),
    )


def _as_1d(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr)
    if a.ndim == 2 and a.shape[1] == 1:
        return a[:, 0]
    return a.reshape(-1) if a.ndim > 1 and a.shape[-1] == 1 else a


def agent_valid_at(state: Dict[str, Any], step: int) -> bool:
    pos = np.asarray(state["position"])
    if step >= len(pos):
        return False
    if "valid" in state:
        v = _as_1d(np.asarray(state["valid"]))
        if step >= len(v):
            return False
        return float(v[step]) > 0
    return not np.allclose(pos[step, :2], 0.0)


@dataclass
class SlotMeta:
    token: str
    agent_type: str
    length: float
    width: float
    height: float
    generate: bool  # True => other-agent future is Nexus generation slot


@dataclass
class NexusSceneBundle:
    """Normalized SceneTensor arrays + metadata (CPU numpy)."""

    tensor: np.ndarray  # (A, 21, 8) normalized
    validity: np.ndarray  # (A, 21)
    road_graph: np.ndarray
    road_graph_validity: np.ndarray
    task_mask: np.ndarray  # (A, 21, 8) 1=keep
    slot_meta: List[SlotMeta]
    freeze: FreezeFrame
    map_coverage: Dict[str, Any]
    history_tokens: List[str]
    num_max_agents_cfg: List[int]
    agent_dim_assert: Dict[str, Any]
    future_leakage_audit: Dict[str, Any]
    raw_unnormalized: np.ndarray  # (A, 21, 8) before norm; non-ego future zeroed


def select_history_vehicles(scene: Dict[str, Any], cutoff: int) -> List[str]:
    """Vehicles with valid frames on all indices 0..cutoff (5 real history frames)."""
    sdc = scene["sdc_id"]
    out = []
    for token in sorted(scene["object_track"].keys()):
        if token == sdc:
            continue
        ot = scene["object_track"][token]
        if ot.get("type") != "VEHICLE":
            continue
        st = ot["state"]
        if all(agent_valid_at(st, t) for t in range(cutoff + 1)):
            out.append(token)
    return out


def _state_at_local(
    freeze: FreezeFrame,
    st: Dict[str, Any],
    t: int,
    *,
    is_ego: bool,
    ego_length: float,
    ego_width: float,
) -> np.ndarray:
    """Return unnormalized 8-d local feature at time t."""
    pos = np.asarray(st["position"], dtype=np.float64)
    heading = float(_as_1d(np.asarray(st["heading"], dtype=np.float64))[t])
    vel = np.asarray(st["velocity"], dtype=np.float64)
    if is_ego:
        # WE stores ego center; Nexus uses rear axle.
        center = pos[t, :2]
        rear = center - REAR_AXLE_TO_CENTER * np.array([math.cos(heading), math.sin(heading)])
        xy_local = freeze.world_to_local_xy(rear)
        length, width = ego_length, ego_width
    else:
        xy_local = freeze.world_to_local_xy(pos[t, :2])
        length = float(_as_1d(np.asarray(st["length"], dtype=np.float64))[t])
        width = float(_as_1d(np.asarray(st["width"], dtype=np.float64))[t])
    h_local = freeze.world_to_local_heading(heading)
    v_local = freeze.world_to_local_vel(vel[t, :2])
    return np.array(
        [
            xy_local[0],
            xy_local[1],
            math.cos(h_local),
            math.sin(h_local),
            v_local[0],
            v_local[1],
            length,
            width,
        ],
        dtype=np.float64,
    )


def _extract_map_polyline(obj: Dict[str, Any]) -> Optional[np.ndarray]:
    """Prefer polyline; for CROSSWALK-like features fall back to polygon ring."""
    poly = np.asarray(obj.get("polyline", []), dtype=np.float64)
    if poly.ndim == 2 and poly.shape[0] >= 2:
        return poly[:, :2]
    polygon = np.asarray(obj.get("polygon", []), dtype=np.float64)
    if polygon.ndim == 2 and polygon.shape[0] >= 2:
        # Close ring if needed so sampling covers full boundary.
        ring = polygon[:, :2]
        if not np.allclose(ring[0], ring[-1]):
            ring = np.concatenate([ring, ring[:1]], axis=0)
        return ring
    return None


def audit_map_coverage(scene: Dict[str, Any], freeze: FreezeFrame, radius_m: float = DEFAULT_RADIUS_M) -> Dict[str, Any]:
    """Explain missing Nexus map types: absent / unmapped / no-geometry / radius truncation."""
    mf = scene.get("map_features", {}) or {}
    we_counts: Dict[str, int] = {}
    reasons = {
        "LANE_CONNECTOR": {"we_absent": True, "note": "OpenScene scene has no LANE_CONNECTOR type"},
        "STOP_LINE": {"we_absent": True, "note": "OpenScene scene has no STOP_LINE type"},
        "CROSSWALK": {"we_count": 0, "with_polyline": 0, "with_polygon_only": 0, "in_radius": 0, "encoded_via_polygon": 0},
        "LANE": {"we_street": 0, "we_unstructure": 0, "in_radius": 0},
    }
    for mid, obj in mf.items():
        if not isinstance(obj, dict):
            continue
        t = str(obj.get("type", ""))
        we_counts[t] = we_counts.get(t, 0) + 1
        if t == "CROSSWALK":
            reasons["CROSSWALK"]["we_count"] += 1
            has_pl = np.asarray(obj.get("polyline", [])).ndim == 2 and len(obj.get("polyline", [])) >= 2
            has_pg = np.asarray(obj.get("polygon", [])).ndim == 2 and len(obj.get("polygon", [])) >= 2
            if has_pl:
                reasons["CROSSWALK"]["with_polyline"] += 1
            elif has_pg:
                reasons["CROSSWALK"]["with_polygon_only"] += 1
            geom = _extract_map_polyline(obj)
            if geom is not None:
                d = float(np.linalg.norm(geom - freeze.origin_xy[None, :], axis=1).min())
                if d <= radius_m:
                    reasons["CROSSWALK"]["in_radius"] += 1
        if t in ("LANE_SURFACE_STREET", "LANE_SURFACE_UNSTRUCTURE"):
            key = "we_street" if t.endswith("STREET") else "we_unstructure"
            reasons["LANE"][key] += 1
            geom = _extract_map_polyline(obj)
            if geom is not None:
                d = float(np.linalg.norm(geom - freeze.origin_xy[None, :], axis=1).min())
                if d <= radius_m:
                    reasons["LANE"]["in_radius"] += 1
    return {
        "we_type_counts": we_counts,
        "diagnosis": reasons,
        "conclusion": {
            "LANE_CONNECTOR": "upstream_absent",
            "STOP_LINE": "upstream_absent",
            "CROSSWALK": "adapter_omission_if_polygon_not_used_else_radius",
            "LANE": "mapped_from_LANE_SURFACE_*",
        },
    }


def encode_map_polylines(
    scene: Dict[str, Any],
    freeze: FreezeFrame,
    map_features: Sequence[str] = DEFAULT_MAP_FEATURES,
    num_max: int = DEFAULT_MAP_NUM_MAX_POLYLINES,
    num_points: int = DEFAULT_MAP_NUM_POINTS,
    radius_m: float = DEFAULT_RADIUS_M,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    mf = scene.get("map_features", {}) or {}
    dim = 2 + len(map_features) + 1
    road = np.zeros((num_max, num_points, dim), dtype=np.float32)
    road_v = np.zeros_like(road)
    counts_we: Dict[str, int] = {}
    counts_nexus: Dict[str, int] = {k: 0 for k in map_features}
    used = 0
    missing_types = set()
    skipped_no_geom = 0
    skipped_radius = 0
    used_polygon_fallback = 0

    # Collect candidates within radius of freeze origin
    candidates: List[Tuple[float, str, Dict[str, Any], str, np.ndarray]] = []
    for mid, obj in mf.items():
        if not isinstance(obj, dict):
            continue
        we_type = str(obj.get("type", ""))
        counts_we[we_type] = counts_we.get(we_type, 0) + 1
        nexus_type = WE_TO_NEXUS_MAP.get(we_type)
        if nexus_type is None or nexus_type not in map_features:
            if we_type:
                missing_types.add(we_type)
            continue
        geom = _extract_map_polyline(obj)
        if geom is None:
            skipped_no_geom += 1
            continue
        if "polyline" not in obj or len(np.asarray(obj.get("polyline", []))) < 2:
            if "polygon" in obj:
                used_polygon_fallback += 1
        d = float(np.linalg.norm(geom - freeze.origin_xy[None, :], axis=1).min())
        if d > radius_m:
            skipped_radius += 1
            continue
        candidates.append((d, str(mid), obj, nexus_type, geom))

    candidates.sort(key=lambda x: x[0])
    # Prefer diversity: encode lanes first then crosswalks within budget
    lanes = [c for c in candidates if c[3] == "LANE"]
    others = [c for c in candidates if c[3] != "LANE"]
    ordered = lanes[: max(0, num_max - min(32, len(others)))] + others
    ordered = ordered[:num_max]

    for _, mid, obj, nexus_type, geom in ordered:
        local = freeze.world_to_local_xy(geom)
        n = local.shape[0]
        idx = np.linspace(0, n - 1, num_points)
        xs = np.interp(idx, np.arange(n), local[:, 0])
        ys = np.interp(idx, np.arange(n), local[:, 1])
        coords = np.stack([xs, ys], axis=1).astype(np.float32)
        onehot = np.zeros((num_points, len(map_features)), dtype=np.float32)
        onehot[:, list(map_features).index(nexus_type)] = 1.0
        speed = np.full((num_points, 1), -1.0, dtype=np.float32)
        row = np.concatenate([coords, onehot, speed], axis=1)
        road[used] = row
        road_v[used] = 1.0
        road_v[used, :, -1] = 0.0
        counts_nexus[nexus_type] += 1
        used += 1

    road[..., :2] = (road[..., :2] - FEATURE_MEANS[:2]) / (2.0 * FEATURE_STD[:2])
    audit = audit_map_coverage(scene, freeze, radius_m=radius_m)

    coverage = {
        "we_type_counts": counts_we,
        "nexus_type_counts": counts_nexus,
        "n_encoded": used,
        "missing_requested": [t for t in map_features if counts_nexus.get(t, 0) == 0],
        "unmapped_we_types": sorted(missing_types),
        "skipped_no_geom": skipped_no_geom,
        "skipped_radius": skipped_radius,
        "polygon_fallback_used": used_polygon_fallback,
        "audit": audit,
        "note": (
            "LANE_CONNECTOR/STOP_LINE absent in OpenScene WE map; "
            "CROSSWALK uses polygon ring when polyline missing; "
            "traffic lights not in Nexus map builder."
        ),
    }
    return road, road_v, coverage


def build_nexus_scene_bundle(
    scene: Dict[str, Any],
    cutoff: int,
    ego_future_world: Dict[str, np.ndarray],
    *,
    ego_length: float = EGO_LENGTH_PACIFICA,
    ego_width: float = EGO_WIDTH_PACIFICA,
    num_max_agents: Sequence[int] = DEFAULT_NUM_MAX_AGENTS,
    map_features: Sequence[str] = DEFAULT_MAP_FEATURES,
) -> NexusSceneBundle:
    """Build normalized SceneTensor for ego-conditioned generation.

    ``ego_future_world`` must provide keys:
      position (16,2) world center, heading (16,), velocity (16,2)
    for Nexus future indices 5..20 (does not include history).
    """
    if cutoff != N_PAST - 1:
        raise ValueError(f"This adapter expects cutoff={N_PAST - 1} (5-frame history), got {cutoff}")

    sdc = scene["sdc_id"]
    freeze = build_freeze_frame(scene, cutoff)
    hist_tokens = select_history_vehicles(scene, cutoff)
    max_veh = int(num_max_agents[0])
    if len(hist_tokens) > max_veh:
        hist_tokens = hist_tokens[:max_veh]

    n_agents_pad = sum(int(x) for x in num_max_agents)  # without ego
    A = n_agents_pad + 1  # + ego
    raw = np.zeros((A, N_TOTAL, N_FEATURES), dtype=np.float64)
    validity = np.zeros((A, N_TOTAL), dtype=np.float64)
    slot_meta: List[SlotMeta] = []

    # --- ego slot 0: history from log + conditioning future ---
    ego_st = scene["object_track"][sdc]["state"]
    for t in range(N_PAST):
        raw[0, t] = _state_at_local(
            freeze, ego_st, t, is_ego=True, ego_length=ego_length, ego_width=ego_width
        )
        validity[0, t] = 1.0
    # future conditioning (world center -> rear axle local)
    epos = np.asarray(ego_future_world["position"], dtype=np.float64)
    ehead = _as_1d(np.asarray(ego_future_world["heading"], dtype=np.float64))
    evel = np.asarray(ego_future_world["velocity"], dtype=np.float64)
    if epos.shape[0] != N_FUTURE:
        raise ValueError(f"ego future must have {N_FUTURE} frames, got {epos.shape[0]}")
    for i in range(N_FUTURE):
        t = N_PAST + i
        heading = float(ehead[i])
        center = epos[i, :2]
        rear = center - REAR_AXLE_TO_CENTER * np.array([math.cos(heading), math.sin(heading)])
        xy_local = freeze.world_to_local_xy(rear)
        h_local = freeze.world_to_local_heading(heading)
        v_local = freeze.world_to_local_vel(evel[i, :2])
        raw[0, t] = [
            xy_local[0],
            xy_local[1],
            math.cos(h_local),
            math.sin(h_local),
            v_local[0],
            v_local[1],
            ego_length,
            ego_width,
        ]
        validity[0, t] = 1.0
    slot_meta.append(
        SlotMeta(
            token=str(sdc),
            agent_type="EGO",
            length=ego_length,
            width=ego_width,
            height=float(_as_1d(np.asarray(ego_st.get("height", [[1.5]])))[cutoff]),
            generate=False,
        )
    )

    # --- other vehicles: history only; future = empty generation slots ---
    for j, token in enumerate(hist_tokens):
        slot = 1 + j
        ot = scene["object_track"][token]
        st = ot["state"]
        length = float(_as_1d(np.asarray(st["length"]))[cutoff])
        width = float(_as_1d(np.asarray(st["width"]))[cutoff])
        height = float(_as_1d(np.asarray(st.get("height", [[1.5]])))[cutoff])
        for t in range(N_PAST):
            raw[slot, t] = _state_at_local(
                freeze, st, t, is_ego=False, ego_length=ego_length, ego_width=ego_width
            )
            validity[slot, t] = 1.0 if agent_valid_at(st, t) else 0.0
        # Future: intentionally ZERO / no log GT. Validity=1 so model generates.
        # Length/width channels left 0 in raw; after normalize they become mean-centered.
        # Official keep_mask zeros future so values should not condition; we still clear GT.
        for t in range(N_PAST, N_TOTAL):
            raw[slot, t] = 0.0
            validity[slot, t] = 1.0
        # Put frozen size into future length/width so decode has identity prior if needed,
        # but task_mask will be 0 on future so they are not kept. Actually plan says
        # length/width generated but WorldEngine uses metadata size. Keep zeros for leakage proof.
        slot_meta.append(
            SlotMeta(
                token=token,
                agent_type="VEHICLE",
                length=length,
                width=width,
                height=height,
                generate=True,
            )
        )

    # Normalize
    tensor = encode_scene_tensor_np(raw).astype(np.float32)
    validity = validity.astype(np.float32)

    # task_mask: 1=keep
    task_mask = np.zeros_like(tensor)
    # ego all frames keep
    task_mask[0, :, :] = 1.0
    # other agents: history keep only
    for j in range(len(hist_tokens)):
        slot = 1 + j
        task_mask[slot, :N_PAST, :] = 1.0
        task_mask[slot, N_PAST:, :] = 0.0

    road, road_v, map_cov = encode_map_polylines(scene, freeze, map_features=map_features)

    # Future leakage audit
    non_ego_future = tensor[1 : 1 + len(hist_tokens), N_PAST:, :]
    non_ego_future_mask = task_mask[1 : 1 + len(hist_tokens), N_PAST:, :]
    leakage = {
        "non_ego_future_tensor_hash": sha256_array(non_ego_future),
        "non_ego_future_task_mask_hash": sha256_array(non_ego_future_mask),
        "non_ego_future_raw_hash": sha256_array(raw[1 : 1 + len(hist_tokens), N_PAST:, :]),
        "non_ego_future_initial_source": "zeros_cleared_no_log_gt",
        "task_mask_future_all_zero": bool(np.all(non_ego_future_mask == 0)),
        "raw_future_all_zero": bool(np.allclose(raw[1 : 1 + len(hist_tokens), N_PAST:, :], 0.0)),
        "normalized_future_is_encode_of_zeros": True,
    }
    # Prove normalized future == encode(zeros)
    zeros_norm = encode_scene_tensor_np(np.zeros((1, N_FEATURES), dtype=np.float64)).astype(np.float32)
    if non_ego_future.size:
        expected = np.broadcast_to(zeros_norm, non_ego_future.shape)
        leakage["matches_encode_zeros"] = bool(np.allclose(non_ego_future, expected, atol=1e-6))
        if not leakage["matches_encode_zeros"] or not leakage["task_mask_future_all_zero"]:
            raise RuntimeError("FUTURE_LEAKAGE_GATE: non-ego future not cleared or task_mask != 0")

    agent_dim_assert = {
        "num_max_agents_cfg": list(num_max_agents),
        "tensor_shape": list(tensor.shape),
        "expected_A": A,
        "n_history_vehicles": len(hist_tokens),
        "ego_slot": 0,
        "padding_validity_zero": bool(np.all(validity[1 + len(hist_tokens) :] == 0)),
    }
    if tensor.shape[0] != A or tensor.shape[1] != N_TOTAL or tensor.shape[2] != N_FEATURES:
        raise RuntimeError(f"Agent dim assert failed: {agent_dim_assert}")

    return NexusSceneBundle(
        tensor=tensor,
        validity=validity,
        road_graph=road,
        road_graph_validity=road_v,
        task_mask=task_mask.astype(np.float32),
        slot_meta=slot_meta,
        freeze=freeze,
        map_coverage=map_cov,
        history_tokens=hist_tokens,
        num_max_agents_cfg=list(num_max_agents),
        agent_dim_assert=agent_dim_assert,
        future_leakage_audit=leakage,
        raw_unnormalized=raw.astype(np.float32),
    )


def apply_ego_future_to_bundle(
    bundle: NexusSceneBundle,
    ego_future_world: Dict[str, np.ndarray],
    ego_length: float,
    ego_width: float,
) -> NexusSceneBundle:
    """Return a deep-copied bundle with only ego future (and raw/tensor) replaced."""
    b = copy.deepcopy(bundle)
    freeze = b.freeze
    epos = np.asarray(ego_future_world["position"], dtype=np.float64)
    ehead = _as_1d(np.asarray(ego_future_world["heading"], dtype=np.float64))
    evel = np.asarray(ego_future_world["velocity"], dtype=np.float64)
    for i in range(N_FUTURE):
        t = N_PAST + i
        heading = float(ehead[i])
        center = epos[i, :2]
        rear = center - REAR_AXLE_TO_CENTER * np.array([math.cos(heading), math.sin(heading)])
        xy_local = freeze.world_to_local_xy(rear)
        h_local = freeze.world_to_local_heading(heading)
        v_local = freeze.world_to_local_vel(evel[i, :2])
        b.raw_unnormalized[0, t] = [
            xy_local[0],
            xy_local[1],
            math.cos(h_local),
            math.sin(h_local),
            v_local[0],
            v_local[1],
            ego_length,
            ego_width,
        ]
    b.tensor[0] = encode_scene_tensor_np(b.raw_unnormalized[0]).astype(np.float32)
    return b


def vocab_candidate_to_nexus_ego_future(
    vocab_traj: np.ndarray,
    ego_center_xy: np.ndarray,
    ego_heading: float,
    rear_axle_to_center: float = REAR_AXLE_TO_CENTER,
) -> Dict[str, Any]:
    """Build 16-frame world-center ego future from 4s vocabulary plan + CV extension.

    Vocabulary is (40, 3) at 0.1s rear-axle local. We take 0.5s samples for 4s
    (8 future points), then constant-velocity / constant-heading extend 8 more.
    """
    raw = np.asarray(vocab_traj, dtype=np.float64)
    # include t0 then take 0.5s steps → 9 poses (t0..t8); futures are [1:]
    local = np.concatenate([np.zeros((1, 3), dtype=np.float64), raw], axis=0)
    sampled = local[::5]  # 0,0.5,...,4.0 → 9
    ego_h = float(ego_heading)
    rear_xy = np.asarray(ego_center_xy, dtype=np.float64)[:2] - rear_axle_to_center * np.array(
        [math.cos(ego_h), math.sin(ego_h)]
    )
    R = np.array(
        [[math.cos(ego_h), -math.sin(ego_h)], [math.sin(ego_h), math.cos(ego_h)]],
        dtype=np.float64,
    )
    rear_traj = sampled[:, :2] @ R.T + rear_xy
    head_traj = sampled[:, 2] + ego_h
    center_traj = rear_traj + rear_axle_to_center * np.stack(
        [np.cos(head_traj), np.sin(head_traj)], axis=1
    )
    # futures: indices 1..8 of sampled → 8 real candidate points
    real_center = center_traj[1:9]
    real_head = head_traj[1:9]
    # velocities from consecutive centers (including t0)
    full_c = center_traj[:9]
    real_vel = np.zeros((8, 2), dtype=np.float64)
    for i in range(8):
        real_vel[i] = (full_c[i + 1] - full_c[i]) / 0.5

    # CV extension using last two real points
    if len(real_center) >= 2:
        v_term = (real_center[-1] - real_center[-2]) / 0.5
        h_term = float(real_head[-1])
    else:
        v_term = real_vel[-1]
        h_term = float(real_head[-1])
    ext_center = np.zeros((8, 2), dtype=np.float64)
    ext_head = np.full(8, h_term, dtype=np.float64)
    ext_vel = np.zeros((8, 2), dtype=np.float64)
    cur = real_center[-1].copy()
    for i in range(8):
        cur = cur + v_term * 0.5
        ext_center[i] = cur
        ext_vel[i] = v_term

    position = np.concatenate([real_center, ext_center], axis=0)
    heading = np.concatenate([real_head, ext_head], axis=0)
    velocity = np.concatenate([real_vel, ext_vel], axis=0)

    # Physics stats for registration / filtering (all gates actually enforced).
    steps = np.linalg.norm(np.diff(full_c, axis=0), axis=1)
    spd = np.linalg.norm(np.diff(full_c, axis=0), axis=1) / 0.5
    max_acc = 0.0
    if len(spd) >= 2:
        max_acc = float(np.max(np.abs(np.diff(spd) / 0.5)))
    head_real = head_traj[:9]
    dh = np.abs(np.arctan2(np.sin(np.diff(head_real)), np.cos(np.diff(head_real))))
    max_yaw_rate = float(np.max(dh / 0.5)) if len(dh) else 0.0
    max_heading_jump = float(np.max(dh)) if len(dh) else 0.0
    # heading ↔ velocity alignment on moving steps (speed > 0.5 m/s)
    aligned = True
    for i in range(len(real_vel)):
        v = real_vel[i]
        sp = float(np.linalg.norm(v))
        if sp < 0.5:
            continue
        v_h = math.atan2(float(v[1]), float(v[0]))
        err = abs(math.atan2(math.sin(v_h - float(real_head[i])), math.cos(v_h - float(real_head[i]))))
        if err > math.pi / 2:
            aligned = False
            break
    # Continuity: first real step from t0 should match first velocity * dt
    first_step = float(np.linalg.norm(full_c[1] - full_c[0]))
    continuity_ok = first_step <= 15.0 and max_heading_jump <= math.pi / 2 and max_acc <= 12.0
    lon_end = float(np.dot(real_center[-1] - center_traj[0], np.array([math.cos(ego_h), math.sin(ego_h)])))
    lat_end = float(
        np.dot(real_center[-1] - center_traj[0], np.array([-math.sin(ego_h), math.cos(ego_h)]))
    )
    stats = {
        "real_endpoint_from_t0_m": float(np.linalg.norm(real_center[-1] - center_traj[0])),
        "real_path_length_m": float(np.sum(steps)),
        "terminal_speed_m_s": float(np.linalg.norm(v_term)),
        "ext_endpoint_from_real_end_m": float(np.linalg.norm(ext_center[-1] - real_center[-1])),
        "max_step_m": float(np.max(np.linalg.norm(np.diff(position, axis=0), axis=1))),
        "max_acc_m_s2": max_acc,
        "max_yaw_rate_rad_s": max_yaw_rate,
        "max_heading_jump_rad": max_heading_jump,
        "within_100m_radius": bool(np.all(np.linalg.norm(position - center_traj[0], axis=1) <= 100.0)),
        "heading_vel_aligned": bool(aligned),
        "continuity_ok": bool(continuity_ok),
        "lon_endpoint_m": lon_end,
        "lat_endpoint_m": lat_end,
        # Map feasibility is NOT claimed here; PDM DAC/Direction provide static map gates.
        "map_feasibility_enforced_here": False,
    }
    return {
        "position": position,
        "heading": heading,
        "velocity": velocity,
        "real_position": real_center,
        "real_heading": real_head,
        "extended_position": ext_center,
        "stats": stats,
        "t0_center": center_traj[0],
        "t0_heading": float(head_traj[0]),
        "traj_hash": sha256_array(position) + ":" + sha256_array(heading),
        "real_hash": sha256_array(real_center) + ":" + sha256_array(real_head),
        "ext_hash": sha256_array(ext_center),
    }


def candidate_physics_reject_reasons(stats: Dict[str, Any]) -> List[str]:
    """Return reject reasons; empty list means pass. All listed gates are enforced."""
    reasons: List[str] = []
    if not stats.get("within_100m_radius", False):
        reasons.append("outside_100m_radius")
    if float(stats.get("max_step_m", 0.0)) > 15.0:
        reasons.append("max_step_gt_15m")
    if float(stats.get("terminal_speed_m_s", 0.0)) > 30.0:
        reasons.append("terminal_speed_gt_30")
    # near-stationary but inconsistent terminal speed → reject (must continue, not pass)
    if float(stats.get("real_path_length_m", 0.0)) < 0.5 and float(stats.get("terminal_speed_m_s", 0.0)) > 1.0:
        reasons.append("near_stationary_speed_inconsistent")
    if not stats.get("heading_vel_aligned", False):
        reasons.append("heading_velocity_misaligned")
    if float(stats.get("max_acc_m_s2", 0.0)) > 12.0:
        reasons.append("acceleration_gt_12")
    if float(stats.get("max_yaw_rate_rad_s", 0.0)) > math.pi:
        reasons.append("yaw_rate_gt_pi")
    if float(stats.get("max_heading_jump_rad", 0.0)) > math.pi / 2:
        reasons.append("heading_jump_gt_90deg")
    if not stats.get("continuity_ok", False):
        reasons.append("trajectory_discontinuity")
    return reasons


def filter_vocab_candidates(
    vocab: np.ndarray,
    ego_center: np.ndarray,
    ego_heading: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Filter vocabulary with enforced physics gates. Returns (kept, reject_counts)."""
    kept: List[Dict[str, Any]] = []
    reject_counts: Dict[str, int] = {}
    for idx in range(len(vocab)):
        fut = vocab_candidate_to_nexus_ego_future(vocab[idx], ego_center, ego_heading)
        reasons = candidate_physics_reject_reasons(fut["stats"])
        if reasons:
            for r in reasons:
                reject_counts[r] = reject_counts.get(r, 0) + 1
            continue
        kept.append({"plan_idx": int(idx), "future": fut})
    return kept, reject_counts


def pair_frame_separation(a: Dict[str, Any], b: Dict[str, Any], ego_heading: float) -> Dict[str, float]:
    sep = pair_separation(a, b)
    da = a["real_position"][-1] - a["t0_center"]
    db = b["real_position"][-1] - b["t0_center"]
    fwd = np.array([math.cos(ego_heading), math.sin(ego_heading)])
    left = np.array([-math.sin(ego_heading), math.cos(ego_heading)])
    lon_a, lat_a = float(np.dot(da, fwd)), float(np.dot(da, left))
    lon_b, lat_b = float(np.dot(db, fwd)), float(np.dot(db, left))
    sep["lon_delta_m"] = abs(lon_a - lon_b)
    sep["lat_delta_m"] = abs(lat_a - lat_b)
    sep["path_length_delta_m"] = abs(
        float(a["stats"]["real_path_length_m"]) - float(b["stats"]["real_path_length_m"])
    )
    return sep


def select_moderate_candidate_pairs(
    candidates: List[Dict[str, Any]],
    ego_heading: float,
    max_pairs: int = 3,
) -> List[Dict[str, Any]]:
    """Select up to 3 moderate pairs (small / mid-longitudinal / reasonable-lateral).

    Does NOT pick the global farthest pair in the full vocabulary.
    Both members must already be static-feasible (caller responsibility).
    """
    if len(candidates) < 2:
        raise RuntimeError("Not enough static-feasible candidates for pairs")
    cands = sorted(candidates, key=lambda c: c["future"]["stats"]["real_path_length_m"])
    anchor = cands[len(cands) // 2]
    scored = []
    for b in cands:
        if b["plan_idx"] == anchor["plan_idx"]:
            continue
        sep = pair_frame_separation(anchor["future"], b["future"], ego_heading)
        scored.append((sep, b))

    def _mk(i: int, a: Dict[str, Any], b: Dict[str, Any], sep: Dict[str, float], rule: str) -> Dict[str, Any]:
        return {
            "pair_id": i,
            "A": {
                "plan_idx": a["plan_idx"],
                "traj_hash": a["future"]["traj_hash"],
                "real_hash": a["future"]["real_hash"],
                "ext_hash": a["future"]["ext_hash"],
                "stats": a["future"]["stats"],
                "static_gate": a.get("static_gate"),
            },
            "B": {
                "plan_idx": b["plan_idx"],
                "traj_hash": b["future"]["traj_hash"],
                "real_hash": b["future"]["real_hash"],
                "ext_hash": b["future"]["ext_hash"],
                "stats": b["future"]["stats"],
                "static_gate": b.get("static_gate"),
            },
            "separation": sep,
            "selection_rule": rule,
            "meets_sep_thresholds": bool(sep["endpoint_m"] >= 1.0 and sep["rms_m"] >= 0.5),
        }

    pairs: List[Dict[str, Any]] = []
    used = {anchor["plan_idx"]}

    # 1) small difference
    small = [
        (s, b)
        for s, b in scored
        if 1.0 <= s["endpoint_m"] <= 5.0 and 0.5 <= s["rms_m"] <= 3.0
    ]
    small.sort(key=lambda x: abs(x[0]["endpoint_m"] - 2.5) + abs(x[0]["rms_m"] - 1.5))
    if small:
        s, b = small[0]
        pairs.append(_mk(0, anchor, b, s, "static_feasible_median_anchor_small_sep"))
        used.add(b["plan_idx"])

    # 2) medium longitudinal
    mid = [
        (s, b)
        for s, b in scored
        if b["plan_idx"] not in used
        and 5.0 <= s["path_length_delta_m"] <= 25.0
        and s["lat_delta_m"] <= 4.0
        and s["endpoint_m"] >= 3.0
    ]
    mid.sort(key=lambda x: abs(x[0]["path_length_delta_m"] - 12.0))
    if mid:
        s, b = mid[0]
        pairs.append(_mk(len(pairs), anchor, b, s, "static_feasible_median_anchor_mid_longitudinal"))
        used.add(b["plan_idx"])

    # 3) reasonable lateral
    lat = [
        (s, b)
        for s, b in scored
        if b["plan_idx"] not in used
        and 2.0 <= s["lat_delta_m"] <= 12.0
        and s["lon_delta_m"] <= 15.0
        and s["endpoint_m"] >= 2.0
    ]
    lat.sort(key=lambda x: abs(x[0]["lat_delta_m"] - 5.0))
    if lat:
        s, b = lat[0]
        pairs.append(_mk(len(pairs), anchor, b, s, "static_feasible_median_anchor_reasonable_lateral"))
        used.add(b["plan_idx"])

    # Fill remaining with moderate (not farthest) seps if needed
    if len(pairs) < max_pairs:
        rest = [(s, b) for s, b in scored if b["plan_idx"] not in used and s["endpoint_m"] >= 1.0]
        rest.sort(key=lambda x: x[0]["endpoint_m"] + x[0]["rms_m"])
        # take mid of remaining rather than max
        if rest:
            mid_i = len(rest) // 2
            for s, b in rest[mid_i:]:
                if len(pairs) >= max_pairs:
                    break
                pairs.append(_mk(len(pairs), anchor, b, s, "static_feasible_median_anchor_moderate_fill"))
                used.add(b["plan_idx"])

    if not pairs:
        raise RuntimeError("Failed to pre-register moderate static-feasible pairs")
    return pairs[:max_pairs]


def decoded_agent_physics_gate(agent: Dict[str, Any], dt: float = 0.5) -> Dict[str, Any]:
    """Physics gate on decoded Nexus agent (first 8 future frames + cutoff continuity)."""
    pos = np.asarray(agent["position"], dtype=np.float64)
    head = _as_1d(np.asarray(agent["heading"], dtype=np.float64))
    vel = np.asarray(agent["velocity"], dtype=np.float64)
    fut_pos = pos[N_PAST : N_PAST + 8]
    fut_head = head[N_PAST : N_PAST + 8]
    fut_vel = vel[N_PAST : N_PAST + 8]
    issues: List[str] = []
    if not (np.isfinite(fut_pos).all() and np.isfinite(fut_head).all() and np.isfinite(fut_vel).all()):
        return {"ok": False, "issues": ["nan_inf"], "max_acc_from_model_spd": None, "max_acc_from_pos": None,
                "max_heading_jump_rad": None, "cos_sin_norm_ok": True}
    # Include cutoff→first-future for continuity of model velocity channel
    spd = np.linalg.norm(vel[N_PAST - 1 : N_PAST + 8, :2], axis=1)
    acc_model = np.abs(np.diff(spd) / dt) if len(spd) >= 2 else np.array([0.0])
    pos_seg = pos[N_PAST - 1 : N_PAST + 8, :2]
    step = np.linalg.norm(np.diff(pos_seg, axis=0), axis=1)
    acc_pos = np.abs(np.diff(step / dt) / dt) if len(step) >= 2 else np.array([0.0])
    head_seg = head[N_PAST - 1 : N_PAST + 8]
    dh = np.abs(np.arctan2(np.sin(np.diff(head_seg)), np.cos(np.diff(head_seg))))
    if float(np.max(step)) > 15.0:
        issues.append("step_gt_15m")
    if float(np.max(np.linalg.norm(fut_vel[:, :2], axis=1))) > 30.0:
        issues.append("speed_gt_30")
    if float(np.max(acc_model)) > 12.0:
        issues.append("acc_gt_12")
    if float(np.max(dh)) > math.pi / 2:
        issues.append("heading_jump_gt_90deg")
    return {
        "ok": len(issues) == 0,
        "issues": issues,
        "max_acc_from_model_spd": float(np.max(acc_model)),
        "max_acc_from_pos": float(np.max(acc_pos)) if len(acc_pos) else 0.0,
        "max_heading_jump_rad": float(np.max(dh)) if len(dh) else 0.0,
        "max_step_m": float(np.max(step)) if len(step) else 0.0,
    }


def diagnose_agent_physics(
    token: str,
    decoded: Dict[str, Any],
    scene: Dict[str, Any],
    cutoff: int,
) -> Dict[str, Any]:
    """Diagnose physics failures without relaxing gates or mutating outputs."""
    if token not in decoded:
        return {"token": token, "error": "missing"}
    da = decoded[token]
    gate = decoded_agent_physics_gate(da)
    pos = np.asarray(da["position"], dtype=np.float64)
    head = _as_1d(np.asarray(da["heading"], dtype=np.float64))
    vel = np.asarray(da["velocity"], dtype=np.float64)
    # current (cutoff) vs first future
    i0, i1 = N_PAST - 1, N_PAST
    st = scene["object_track"][token]["state"]
    log_pos = np.asarray(st["position"], dtype=np.float64)[cutoff, :2]
    log_vel = np.asarray(st["velocity"], dtype=np.float64)[cutoff, :2]
    log_head = float(_as_1d(np.asarray(st["heading"]))[cutoff])
    model_spd0 = float(np.linalg.norm(vel[i0, :2]))
    model_spd1 = float(np.linalg.norm(vel[i1, :2]))
    pos_implied_acc = float(np.linalg.norm(pos[i1, :2] - pos[i0, :2]) / 0.5 - model_spd0) / 0.5
    # heading π-equivalence probe (diagnose only; do not canonicalize)
    dh = abs(math.atan2(math.sin(head[i1] - head[i0]), math.cos(head[i1] - head[i0])))
    dh_flip = abs(math.atan2(math.sin(head[i1] - (head[i0] + math.pi)), math.cos(head[i1] - (head[i0] + math.pi))))
    # Reconstruct local cos/sin norm from world heading (decode uses atan2 already)
    cos_sin_norm = 1.0  # world heading is scalar; check velocity vs heading at near-zero
    near_zero = model_spd0 < 0.5 and model_spd1 < 0.5
    return {
        "token": token,
        "gate": gate,
        "cutoff_state_match": {
            "pos_err_m": float(np.linalg.norm(pos[i0, :2] - log_pos)),
            "vel_err_mps": float(np.linalg.norm(vel[i0, :2] - log_vel)),
            "heading_err_rad": abs(math.atan2(math.sin(head[i0] - log_head), math.cos(head[i0] - log_head))),
        },
        "model_speed_channel": {
            "spd_current": model_spd0,
            "spd_first_future": model_spd1,
            "acc_from_model_spd": abs(model_spd1 - model_spd0) / 0.5,
            "acc_implied_by_position_step": abs(pos_implied_acc),
            "source_hypothesis": (
                "model_velocity_output_discontinuity"
                if abs(model_spd1 - model_spd0) / 0.5 > 12.0
                else "not_model_velocity_jump"
            ),
        },
        "heading_jump": {
            "dh_rad": dh,
            "dh_deg": dh * 180.0 / math.pi,
            "dh_if_theta_plus_pi_rad": dh_flip,
            "near_zero_speed": near_zero,
            "box_direction_pi_equivalence_suspect": bool(near_zero and dh > math.pi / 2 and dh_flip < 0.2),
            "note": "Diagnose only; no heading canonicalization applied.",
        },
        "cos_sin_norm_world_scalar": cos_sin_norm,
    }


def pair_separation(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, float]:
    pa, pb = a["real_position"], b["real_position"]
    rms = float(np.sqrt(np.mean(np.sum((pa - pb) ** 2, axis=1))))
    end = float(np.linalg.norm(pa[-1] - pb[-1]))
    return {"rms_m": rms, "endpoint_m": end}


def decode_sampled_to_world(
    sampled_norm: np.ndarray,
    bundle: NexusSceneBundle,
) -> Dict[str, Any]:
    """Decode normalized sampled tensor to world trajectories per slot."""
    raw = decode_scene_tensor_np(np.asarray(sampled_norm, dtype=np.float64))
    freeze = bundle.freeze
    agents = {}
    for slot, meta in enumerate(bundle.slot_meta):
        if slot >= raw.shape[0]:
            break
        traj = raw[slot]  # (21, 8)
        valid = bundle.validity[slot]
        centers = []
        headings = []
        velocities = []
        valids = []
        for t in range(N_TOTAL):
            x, y, cosh, sinh, vx, vy, length, width = traj[t]
            h_local = math.atan2(sinh, cosh)
            if meta.agent_type == "EGO":
                rear_local = np.array([x, y])
                rear_w = freeze.local_to_world_xy(rear_local)
                h_w = float(freeze.local_to_world_heading(h_local))
                center_w = rear_w + REAR_AXLE_TO_CENTER * np.array([math.cos(h_w), math.sin(h_w)])
            else:
                center_w = freeze.local_to_world_xy(np.array([x, y]))
                h_w = float(freeze.local_to_world_heading(h_local))
            v_w = freeze.local_to_world_vel(np.array([vx, vy]))
            centers.append(center_w)
            headings.append(h_w)
            velocities.append(v_w)
            valids.append(float(valid[t]))
            _ = length, width  # identity from metadata
        agents[meta.token] = {
            "type": meta.agent_type,
            "position": np.asarray(centers),
            "heading": np.asarray(headings),
            "velocity": np.asarray(velocities),
            "valid": np.asarray(valids),
            "length": meta.length,
            "width": meta.width,
            "slot": slot,
            "generate": meta.generate,
        }
    return agents


def ego_hold_errors(
    decoded_ego: Dict[str, np.ndarray],
    cond: Dict[str, Any],
) -> Dict[str, Any]:
    """Compare decoded ego future (16 frames) vs conditioning."""
    pos = decoded_ego["position"][N_PAST:]  # 16
    head = decoded_ego["heading"][N_PAST:]
    vel = decoded_ego["velocity"][N_PAST:]
    cpos = cond["position"]
    chead = cond["heading"]
    cvel = cond["velocity"]

    def _seg(sl: slice) -> Dict[str, float]:
        pe = np.linalg.norm(pos[sl] - cpos[sl], axis=1)
        he = np.abs(np.arctan2(np.sin(head[sl] - chead[sl]), np.cos(head[sl] - chead[sl])))
        ve = np.linalg.norm(vel[sl] - cvel[sl], axis=1)
        return {
            "pos_mean_m": float(np.mean(pe)),
            "pos_max_m": float(np.max(pe)),
            "heading_mean_rad": float(np.mean(he)),
            "heading_max_rad": float(np.max(he)),
            "vel_mean_mps": float(np.mean(ve)),
            "vel_max_mps": float(np.max(ve)),
            "pos_hash": sha256_array(pos[sl]),
            "cond_pos_hash": sha256_array(cpos[sl]),
        }

    return {
        "real_5_12": _seg(slice(0, 8)),
        "extended_13_20": _seg(slice(8, 16)),
        "all_16": _seg(slice(0, 16)),
    }


def to_future_pack(
    decoded_agents: Dict[str, Any],
    scene: Dict[str, Any],
    cutoff: int,
    horizon: int,
    provenance: Dict[str, Any],
    *,
    source: str = "nexus",
) -> Dict[str, Any]:
    """Build WorldEngine future pack: cutoff current + first (horizon-1) Nexus futures."""
    from .base import Provenance, TrafficFuturePack

    sdc = scene["sdc_id"]
    futures: Dict[str, Any] = {}
    coverage = {
        "nexus_generated": [],
        "log_fallback": [],
        "invalid_at_cutoff": [],
        "fallback_reasons": {},
    }
    # Pack only non-ego agents (PDM injects agent futures; ego conditioning separate)
    for token, ot in scene["object_track"].items():
        if token == sdc:
            continue
        st = ot["state"]
        if not agent_valid_at(st, cutoff):
            coverage["invalid_at_cutoff"].append(token)
            continue
        if token in decoded_agents and decoded_agents[token].get("generate"):
            da = decoded_agents[token]
            # Pack schema: horizon frames starting at cutoff inclusive.
            # Decoded timeline: index cutoff maps to N_PAST-1=4; future k>=1 -> N_PAST+(k-1).
            pos_list = []
            head_list = []
            vel_list = []
            valid_list = []
            for k in range(horizon):
                if k == 0:
                    ti = N_PAST - 1
                else:
                    ti = N_PAST + (k - 1)
                if ti >= da["position"].shape[0]:
                    break
                pos_list.append(da["position"][ti, :2])
                head_list.append(da["heading"][ti])
                vel_list.append(da["velocity"][ti, :2])
                valid_list.append(da["valid"][ti])
            # For k==0 must match frozen log state exactly — overwrite with log
            pos0 = np.asarray(st["position"], dtype=np.float64)[cutoff, :2]
            head0 = float(_as_1d(np.asarray(st["heading"]))[cutoff])
            vel0 = np.asarray(st["velocity"], dtype=np.float64)[cutoff, :2]
            pos_list[0] = pos0
            head_list[0] = head0
            vel_list[0] = vel0
            valid_list[0] = 1.0
            futures[token] = {
                "type": ot.get("type"),
                "position": np.asarray(pos_list, dtype=np.float64),
                "heading": np.asarray(head_list, dtype=np.float64),
                "velocity": np.asarray(vel_list, dtype=np.float64),
                "valid": np.asarray(valid_list, dtype=np.float64),
                "source": source,
            }
            coverage["nexus_generated"].append(token)
        else:
            # Replay fallback for pack completeness only
            pos = np.asarray(st["position"], dtype=np.float64)
            heading = _as_1d(np.asarray(st["heading"], dtype=np.float64))
            vel = np.asarray(st["velocity"], dtype=np.float64)
            valid = _as_1d(np.asarray(st.get("valid", np.ones(len(pos))), dtype=np.float64))
            end = min(cutoff + horizon, len(pos))
            sl = slice(cutoff, end)
            futures[token] = {
                "type": ot.get("type"),
                "position": pos[sl].copy(),
                "heading": heading[sl].copy(),
                "velocity": vel[sl].copy() if len(vel) >= end else vel[cutoff:].copy(),
                "valid": valid[sl].copy() if len(valid) >= end else valid[cutoff:].copy(),
                "source": "log_replay_fallback",
            }
            coverage["log_fallback"].append(token)
            coverage["fallback_reasons"][token] = (
                "not_in_nexus_slots" if token not in decoded_agents else "non_generate_type"
            )

    # hash
    parts = []
    for token in sorted(futures.keys()):
        f = futures[token]
        parts.append(token.encode())
        parts.append(np.ascontiguousarray(f["position"]).tobytes())
        parts.append(np.ascontiguousarray(f["heading"]).tobytes())
    fh = sha256_bytes(b"".join(parts))
    pack = TrafficFuturePack(
        source=source,
        cutoff=cutoff,
        horizon=horizon,
        frequency_hz=2.0,
        dt_s=0.5,
        futures=futures,
        coverage=coverage,
        future_hash=fh,
        provenance=Provenance(source=source, scene_id=str(scene.get("id")), cutoff=cutoff, extras=provenance),
        n_agents=len(futures),
    )
    return pack.to_dict()


def physics_check_futures(futures: Dict[str, Any], dt: float = 0.5) -> Dict[str, Any]:
    """Gate 6 physics on generated agents."""
    issues = []
    ok = True
    for token, f in futures.items():
        if f.get("source") != "nexus":
            continue
        pos = np.asarray(f["position"], dtype=np.float64)
        head = _as_1d(np.asarray(f["heading"], dtype=np.float64))
        vel = np.asarray(f["velocity"], dtype=np.float64)
        if not np.isfinite(pos).all() or not np.isfinite(head).all() or not np.isfinite(vel).all():
            issues.append({"token": token, "issue": "nan_inf"})
            ok = False
            continue
        if len(pos) >= 2:
            step = np.linalg.norm(np.diff(pos[:, :2], axis=0), axis=1)
            if float(np.max(step)) > 15.0:
                issues.append({"token": token, "issue": "step_gt_15m", "max": float(np.max(step))})
                ok = False
            spd = np.linalg.norm(vel[:, :2], axis=1)
            if float(np.max(spd)) > 30.0:
                issues.append({"token": token, "issue": "speed_gt_30", "max": float(np.max(spd))})
                ok = False
            if len(spd) >= 2:
                acc = np.abs(np.diff(spd) / dt)
                if float(np.max(acc)) > 12.0:
                    issues.append({"token": token, "issue": "acc_gt_12", "max": float(np.max(acc))})
                    ok = False
            dh = np.abs(np.arctan2(np.sin(np.diff(head)), np.cos(np.diff(head))))
            if float(np.max(dh)) > math.pi / 2:
                issues.append({"token": token, "issue": "heading_jump_gt_90deg", "max": float(np.max(dh))})
                ok = False
    return {"ok": ok, "n_issues": len(issues), "issues": issues[:50]}


def roundtrip_history_test(scene: Dict[str, Any], cutoff: int = 4) -> Dict[str, Any]:
    """WE history -> SceneTensor local -> world; check errors on history frames."""
    # Dummy ego future (zeros motion) just to build bundle
    sdc = scene["sdc_id"]
    st = scene["object_track"][sdc]["state"]
    pos0 = np.asarray(st["position"], dtype=np.float64)[cutoff, :2]
    h0 = float(_as_1d(np.asarray(st["heading"]))[cutoff])
    dummy = {
        "position": np.tile(pos0, (N_FUTURE, 1)),
        "heading": np.full(N_FUTURE, h0),
        "velocity": np.zeros((N_FUTURE, 2)),
    }
    bundle = build_nexus_scene_bundle(scene, cutoff, dummy)
    decoded = decode_sampled_to_world(bundle.tensor, bundle)
    errs = []
    for token, da in decoded.items():
        ot = scene["object_track"][token]
        st = ot["state"]
        for t in range(N_PAST):
            if not agent_valid_at(st, t):
                continue
            gt_pos = np.asarray(st["position"], dtype=np.float64)[t, :2]
            gt_h = float(_as_1d(np.asarray(st["heading"]))[t])
            gt_v = np.asarray(st["velocity"], dtype=np.float64)[t, :2]
            pe = float(np.linalg.norm(da["position"][t, :2] - gt_pos))
            he = abs(math.atan2(math.sin(da["heading"][t] - gt_h), math.cos(da["heading"][t] - gt_h)))
            ve = float(np.linalg.norm(da["velocity"][t, :2] - gt_v))
            errs.append((token, t, pe, he, ve))
    if not errs:
        return {"ok": False, "reason": "no_valid_history"}
    max_pe = max(e[2] for e in errs)
    max_he = max(e[3] for e in errs)
    max_ve = max(e[4] for e in errs)
    return {
        "ok": max_pe <= 1e-4 and max_he <= 1e-4 and max_ve <= 1e-4,
        "max_pos_m": max_pe,
        "max_heading_rad": max_he,
        "max_vel_mps": max_ve,
        "n_checks": len(errs),
        "leakage": bundle.future_leakage_audit,
        "map_coverage": bundle.map_coverage,
        "agent_dim_assert": bundle.agent_dim_assert,
    }


def official_codec_equivalence_test() -> Dict[str, Any]:
    """Compare local encode/decode vs official Nexus codec on random tensors."""
    info = try_import_official_codec()
    if not info.get("ok"):
        return {"ok": False, "reason": "official_import_failed", "error": info.get("error")}
    rng = np.random.RandomState(0)
    raw = rng.randn(4, 21, 8).astype(np.float64)
    local_n = encode_scene_tensor_np(raw)
    off_n = np.asarray(info["encode"](raw), dtype=np.float64)
    local_d = decode_scene_tensor_np(local_n)
    off_d = np.asarray(info["decode"](local_n), dtype=np.float64)
    means_match = np.allclose(info["FEATURE_MEANS"], FEATURE_MEANS)
    std_match = np.allclose(info["FEATURE_STD"], FEATURE_STD)
    enc_err = float(np.max(np.abs(local_n - off_n)))
    dec_err = float(np.max(np.abs(local_d - off_d)))
    return {
        "ok": means_match and std_match and enc_err <= 1e-6 and dec_err <= 1e-6,
        "means_match": means_match,
        "std_match": std_match,
        "encode_max_abs_err": enc_err,
        "decode_max_abs_err": dec_err,
        "official_ego_L": info["EGO_LENGTH"],
        "official_ego_W": info["EGO_WIDTH"],
        "N_SCENE_TENSOR_FEATURES": info["N_SCENE_TENSOR_FEATURES"],
    }
