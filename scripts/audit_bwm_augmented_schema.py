#!/usr/bin/env python3
"""BWM-Offline augmented scenario pickle schema & pairing-capability audit.

Only reads author-released frozen trajectories. Does NOT run BWM, PDM, or
download original/sensor/3DGS assets. Restricted Unpickler refuses unknown
globals. ndarray leaves record shape/dtype/finite stats only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import pickletools
import random
import resource
import signal
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

WE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEED = 20260716
JSON_HARD_LIMIT_BYTES = 10 * 1024 * 1024

# Explicit identity / pairing fields — absent must be recorded, never inferred.
SOURCE_FIELD_CANDIDATES = (
    "source",
    "source_token",
    "source_scene_id",
    "original",
    "original_token",
    "original_scene_id",
    "original_id",
    "parent",
    "parent_token",
    "parent_scene_id",
    "base_scene_id",
    "base_token",
)
VARIANT_FIELD_CANDIDATES = (
    "variant",
    "variant_id",
    "sample_id",
    "seed",
    "augmentation_type",
    "aug_type",
    "goal",
    "intent",
    "attack",
)
EGO_COND_FIELD_CANDIDATES = (
    "ego_conditioning",
    "ego_plan",
    "ego_future",
    "plan_idx",
    "plan_index",
    "requested_ego_conditioning_hash",
    "ego_traj_hash",
    "conditioning_plan",
    "bwm_ego_plan",
)
BWM_PROV_FIELD_CANDIDATES = (
    "bwm_provenance",
    "bwm_model",
    "bwm_revision",
    "generation_window",
    "generation_seed",
    "model_revision",
    "bwm_future",
    "traffic_future_source",
)
REWARD_FIELD_CANDIDATES = (
    "pdms",
    "pdms_pkl",
    "reward",
    "rewards",
    "score",
    "scores",
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "comfort",
    "driving_direction_compliance",
    "ego_progress",
    "time_to_collision_within_bound",
    "lane_keeping",
)
SENSOR_FIELD_CANDIDATES = (
    "sensor_blobs",
    "sensor_path",
    "cameras",
    "lidar",
    "digitaltwin_asset_id",
    "synthetic_scene_info",
    "openscene_data_infos_dict",
)

REQUIRED_FIRST_LEVEL = ("object_track", "id", "dynamic_map_states", "map_features", "log_length")
AGENT_STATE_KEYS = ("position", "heading", "velocity", "valid", "length", "width", "height", "angular_velocity")

# Restricted unpickle allowlist — data-only / numpy reconstruction.
ALLOWED_GLOBALS: Set[Tuple[str, str]] = {
    ("builtins", "dict"),
    ("builtins", "list"),
    ("builtins", "tuple"),
    ("builtins", "set"),
    ("builtins", "frozenset"),
    ("builtins", "bytearray"),
    ("builtins", "bytes"),
    ("builtins", "str"),
    ("builtins", "int"),
    ("builtins", "float"),
    ("builtins", "complex"),
    ("builtins", "bool"),
    ("builtins", "NoneType"),
    ("builtins", "type"),
    ("builtins", "object"),
    ("collections", "OrderedDict"),
    ("collections", "defaultdict"),
    ("numpy", "ndarray"),
    ("numpy", "dtype"),
    ("numpy", "generic"),
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "scalar"),
    ("numpy.core.multiarray", "dtype"),
    ("numpy.core._multiarray_umath", "_reconstruct"),
    ("numpy.core._multiarray_umath", "scalar"),
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "scalar"),
    ("numpy._core.multiarray", "dtype"),
    ("numpy", "float64"),
    ("numpy", "float32"),
    ("numpy", "float16"),
    ("numpy", "int64"),
    ("numpy", "int32"),
    ("numpy", "int16"),
    ("numpy", "int8"),
    ("numpy", "uint64"),
    ("numpy", "uint32"),
    ("numpy", "uint16"),
    ("numpy", "uint8"),
    ("numpy", "bool_"),
    ("numpy", "bool"),
    ("numpy", "complex128"),
    ("numpy", "complex64"),
    ("numpy.dtypes", "Float64DType"),
    ("numpy.dtypes", "Float32DType"),
    ("numpy.dtypes", "Int64DType"),
    ("numpy.dtypes", "Int32DType"),
    ("numpy.dtypes", "BoolDType"),
    ("numpy.dtypes", "UInt8DType"),
}


class RestrictedUnpickler(pickle.Unpickler):
    """Refuse unknown globals; never fall back to unrestricted pickle.load."""

    def find_class(self, module: str, name: str):  # noqa: D401
        key = (module, name)
        if key in ALLOWED_GLOBALS:
            return super().find_class(module, name)
        # numpy.dtype via np.dtype aliases
        if module.startswith("numpy") and name in {
            "dtype",
            "ndarray",
            "_reconstruct",
            "scalar",
        }:
            if key in ALLOWED_GLOBALS or (module, name) in ALLOWED_GLOBALS:
                return super().find_class(module, name)
            # allow numpy.core.numeric / numpy._core.numeric reconstruct paths already listed
        raise pickle.UnpicklingError(f"forbidden_global:{module}.{name}")


def restricted_load(path: Path) -> Any:
    with path.open("rb") as f:
        return RestrictedUnpickler(f).load()


def sha256_file(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def rss_gb() -> float:
    try:
        with open(f"/proc/{os.getpid()}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / (1024.0 ** 2)
    except Exception:
        pass
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux ru_maxrss is kB
    return float(usage) / (1024.0 ** 2)


def save_json(path: Path, obj: Any) -> None:
    text = json.dumps(obj, indent=2, ensure_ascii=False, default=_json_default)
    raw = text.encode("utf-8")
    if len(raw) > JSON_HARD_LIMIT_BYTES:
        raise RuntimeError(f"JSON exceeds hard limit {JSON_HARD_LIMIT_BYTES}: {len(raw)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        v = float(o)
        return v if math.isfinite(v) else str(v)
    if isinstance(o, np.ndarray):
        return {"__ndarray__": True, "shape": list(o.shape), "dtype": str(o.dtype)}
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, set):
        return sorted(o)
    return str(o)


def verify_file_gate(path: Path, expected_size: int, expected_sha256: str) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    size = path.stat().st_size
    if size != expected_size:
        raise RuntimeError(f"size_mismatch actual={size} expected={expected_size}")
    digest = sha256_file(path)
    if digest != expected_sha256:
        raise RuntimeError(f"sha256_mismatch actual={digest} expected={expected_sha256}")
    return {"path": str(path), "size": size, "sha256": digest}


# ---------------------------------------------------------------------------
# Phase 1: pickletools scan (not a schema substitute)
# ---------------------------------------------------------------------------

def scan_pickle_opcodes(path: Path, max_wall_s: float, max_rss_gb: float) -> Dict[str, Any]:
    """Stream opcodes. STACK_GLOBAL pushes module/name from prior SHORT_BINUNICODE/BINUNICODE."""
    t0 = time.time()
    opcode_counts: Counter = Counter()
    globals_seen: Counter = Counter()
    protocols: Set[int] = set()
    reduce_n = 0
    build_n = 0
    forbidden: List[str] = []
    n_ops = 0
    # minimal stack of recent unicode strings for STACK_GLOBAL reconstruction
    str_stack: List[str] = []
    with path.open("rb") as f:
        for op, arg, pos in pickletools.genops(f):
            n_ops += 1
            opcode_counts[op.name] += 1
            if op.name == "PROTO" and isinstance(arg, int):
                protocols.add(arg)
            if op.name in ("SHORT_BINUNICODE", "BINUNICODE", "UNICODE") and isinstance(arg, str):
                str_stack.append(arg)
                if len(str_stack) > 8:
                    str_stack = str_stack[-8:]
            if op.name == "STACK_GLOBAL":
                if len(str_stack) >= 2:
                    mod, name = str_stack[-2], str_stack[-1]
                    str_stack = str_stack[:-2]
                    # Only treat as global if module path looks like an importable module.
                    if not (
                        mod.startswith(("numpy", "builtins", "collections", "numpy."))
                        or "." in mod
                    ):
                        continue
                    key = f"{mod} {name}"
                    globals_seen[key] += 1
                    if (mod, name) not in ALLOWED_GLOBALS and not _maybe_numpy_allowed(mod, name):
                        forbidden.append(key)
                else:
                    forbidden.append("STACK_GLOBAL_without_two_strings")
            if op.name == "GLOBAL":
                if isinstance(arg, str):
                    globals_seen[arg] += 1
                    mod, _, name = arg.partition(" ")
                    if (mod, name) not in ALLOWED_GLOBALS and not _maybe_numpy_allowed(mod, name):
                        forbidden.append(arg)
            if op.name == "REDUCE":
                reduce_n += 1
            if op.name == "BUILD":
                build_n += 1
            if op.name in ("EXT1", "EXT2", "EXT4", "PERSID", "BINPERSID"):
                forbidden.append(f"unsupported_opcode:{op.name}")
            if n_ops % 200_000 == 0:
                if time.time() - t0 > max_wall_s:
                    raise TimeoutError("scan_wall_exceeded")
                if rss_gb() > max_rss_gb:
                    raise MemoryError("scan_rss_exceeded")
    return {
        "ok": len(forbidden) == 0,
        "n_opcodes": n_ops,
        "protocols": sorted(protocols),
        "opcode_top20": opcode_counts.most_common(20),
        "globals": dict(globals_seen.most_common(50)),
        "n_reduce": reduce_n,
        "n_build": build_n,
        "forbidden": forbidden[:50],
        "n_forbidden": len(forbidden),
        "wall_s": round(time.time() - t0, 3),
        "peak_rss_gb": round(rss_gb(), 3),
        "note": "pickletools scan is safety precheck only; NOT a schema substitute",
    }


def _maybe_numpy_allowed(module: str, name: str) -> bool:
    if (module, name) in ALLOWED_GLOBALS:
        return True
    if module.startswith("numpy") and name in {"dtype", "ndarray", "_reconstruct", "scalar"}:
        # still require exact allowlist match for load; for scan, flag if not listed
        return (module, name) in ALLOWED_GLOBALS
    return False


# ---------------------------------------------------------------------------
# Schema leaf / walk helpers
# ---------------------------------------------------------------------------

def ndarray_leaf_stats(arr: np.ndarray) -> Dict[str, Any]:
    """Treat ndarray as schema leaf — no recursive expansion."""
    a = np.asarray(arr)
    out: Dict[str, Any] = {
        "kind": "ndarray",
        "shape": list(a.shape),
        "dtype": str(a.dtype),
        "nbytes": int(a.nbytes),
        "ndim": int(a.ndim),
    }
    if a.size == 0:
        out["finite_frac"] = None
        out["nan_count"] = 0
        out["inf_count"] = 0
        return out
    if np.issubdtype(a.dtype, np.number):
        flat = a.ravel()
        # sample at most 1e6 elements for finite stats to bound cost
        if flat.size > 1_000_000:
            idx = np.linspace(0, flat.size - 1, 1_000_000).astype(np.int64)
            sample = flat[idx]
        else:
            sample = flat
        finite = np.isfinite(sample.astype(np.float64, copy=False))
        out["finite_frac"] = float(np.mean(finite))
        out["nan_count_sample"] = int(np.isnan(sample.astype(np.float64, copy=False)).sum())
        out["inf_count_sample"] = int(np.isinf(sample.astype(np.float64, copy=False)).sum())
        out["sample_n"] = int(sample.size)
    else:
        out["finite_frac"] = None
    return out


def type_name(x: Any) -> str:
    return type(x).__name__


def find_keys_recursive(
    obj: Any,
    candidates: Sequence[str],
    *,
    max_depth: int = 6,
    max_keys: int = 5000,
    path: str = "",
) -> Dict[str, Any]:
    """Locate candidate key names; ndarray is always a leaf."""
    found: Dict[str, List[Dict[str, Any]]] = {c: [] for c in candidates}
    n_visited = 0

    def walk(node: Any, depth: int, cur: str) -> None:
        nonlocal n_visited
        if n_visited >= max_keys or depth > max_depth:
            return
        n_visited += 1
        if isinstance(node, np.ndarray):
            return
        if isinstance(node, dict):
            for k, v in node.items():
                ks = str(k)
                if ks in found and len(found[ks]) < 5:
                    entry: Dict[str, Any] = {"path": f"{cur}.{ks}" if cur else ks, "value_type": type_name(v)}
                    if isinstance(v, np.ndarray):
                        entry["ndarray"] = ndarray_leaf_stats(v)
                    elif isinstance(v, (str, int, float, bool)) or v is None:
                        entry["value"] = v
                    elif isinstance(v, (list, tuple)):
                        entry["len"] = len(v)
                    found[ks].append(entry)
                if isinstance(v, np.ndarray):
                    continue
                if isinstance(v, (dict, list, tuple)):
                    walk(v, depth + 1, f"{cur}.{ks}" if cur else ks)
        elif isinstance(node, (list, tuple)):
            # only scan first few elements
            for i, v in enumerate(node[:3]):
                if isinstance(v, np.ndarray):
                    continue
                if isinstance(v, (dict, list, tuple)):
                    walk(v, depth + 1, f"{cur}[{i}]")

    walk(obj, 0, path)
    coverage = {k: (len(v) > 0) for k, v in found.items()}
    present = {k: v for k, v in found.items() if v}
    absent = sorted(k for k, v in found.items() if not v)
    return {"coverage": coverage, "present": present, "absent": absent, "n_visited": n_visited}


def field_presence(scenario: Dict[str, Any], names: Sequence[str]) -> Dict[str, str]:
    """Top-level + metadata presence only (not filename inference)."""
    meta = scenario.get("metadata") if isinstance(scenario.get("metadata"), dict) else {}
    out: Dict[str, str] = {}
    for n in names:
        if n in scenario:
            out[n] = "present_top"
        elif isinstance(meta, dict) and n in meta:
            out[n] = "present_metadata"
        else:
            out[n] = "absent"
    return out


# ---------------------------------------------------------------------------
# Per-scenario audit
# ---------------------------------------------------------------------------

def audit_agent_tracks(object_track: Dict[str, Any], log_length: Optional[int]) -> Dict[str, Any]:
    types: Counter = Counter()
    n_agents = 0
    state_shapes: Dict[str, Counter] = {k: Counter() for k in AGENT_STATE_KEYS}
    state_present: Counter = Counter()
    spawn_despawn = {"n_with_valid_switch": 0, "n_checked": 0}
    for token, obj in object_track.items():
        if not isinstance(obj, dict):
            continue
        n_agents += 1
        typ = str(obj.get("type", "UNKNOWN"))
        types[typ] += 1
        st = obj.get("state")
        if not isinstance(st, dict):
            continue
        for k in AGENT_STATE_KEYS:
            if k not in st:
                continue
            state_present[k] += 1
            v = st[k]
            if isinstance(v, np.ndarray):
                state_shapes[k][str(tuple(v.shape)) + "|" + str(v.dtype)] += 1
            else:
                state_shapes[k][f"non_ndarray:{type_name(v)}"] += 1
        if "valid" in st and isinstance(st["valid"], np.ndarray):
            spawn_despawn["n_checked"] += 1
            valid = np.asarray(st["valid"]).reshape(-1)
            if valid.size >= 2:
                switches = int(np.sum(valid[1:] != valid[:-1]))
                if switches > 0:
                    spawn_despawn["n_with_valid_switch"] += 1
    return {
        "n_agents": n_agents,
        "type_counts": dict(types),
        "state_key_present_counts": dict(state_present),
        "state_shape_dtype_top": {k: v.most_common(5) for k, v in state_shapes.items() if v},
        "spawn_despawn": spawn_despawn,
        "log_length_declared": log_length,
    }


def audit_map_features(mf: Dict[str, Any]) -> Dict[str, Any]:
    type_counts: Counter = Counter()
    geom = {"polyline": 0, "polygon": 0, "neither": 0}
    for _mid, feat in mf.items():
        if not isinstance(feat, dict):
            continue
        t = str(feat.get("type", "UNKNOWN"))
        type_counts[t] += 1
        has_pl = isinstance(feat.get("polyline"), np.ndarray) or (
            isinstance(feat.get("polyline"), (list, tuple)) and len(feat.get("polyline", [])) >= 2
        )
        has_pg = isinstance(feat.get("polygon"), np.ndarray) or (
            isinstance(feat.get("polygon"), (list, tuple)) and len(feat.get("polygon", [])) >= 2
        )
        if has_pl:
            geom["polyline"] += 1
        elif has_pg:
            geom["polygon"] += 1
        else:
            geom["neither"] += 1
    # interest types
    interest = {
        "LANE": sum(v for k, v in type_counts.items() if "LANE" in k.upper() and "CONNECTOR" not in k.upper()),
        "LANE_CONNECTOR": sum(v for k, v in type_counts.items() if "CONNECTOR" in k.upper()),
        "ROADBLOCK": sum(v for k, v in type_counts.items() if "ROADBLOCK" in k.upper() or "ROAD_BLOCK" in k.upper()),
        "CROSSWALK": sum(v for k, v in type_counts.items() if "CROSSWALK" in k.upper()),
        "STOP_LINE": sum(v for k, v in type_counts.items() if "STOP" in k.upper() and "LINE" in k.upper()),
    }
    return {"n_features": len(mf), "type_counts": dict(type_counts), "geometry": geom, "interest_counts": interest}


def audit_dynamic_map(dms: Dict[str, Any]) -> Dict[str, Any]:
    n = len(dms)
    shapes: Counter = Counter()
    for _lid, obj in list(dms.items())[:50]:
        if not isinstance(obj, dict):
            continue
        st = obj.get("state")
        if isinstance(st, dict):
            for k, v in st.items():
                if isinstance(v, np.ndarray):
                    shapes[f"{k}:{tuple(v.shape)}|{v.dtype}"] += 1
                else:
                    shapes[f"{k}:{type_name(v)}"] += 1
    return {"n_lights_or_dynamic": n, "state_shape_samples": shapes.most_common(20)}


def audit_time_anchor(scenario: Dict[str, Any]) -> Dict[str, Any]:
    """Highest-priority audit: do not invent cutoff."""
    meta = scenario.get("metadata") if isinstance(scenario.get("metadata"), dict) else {}
    sample_rate = scenario.get("sample_rate", meta.get("sample_rate") if isinstance(meta, dict) else None)
    log_length = scenario.get("log_length")
    base_ts = scenario.get("base_timestamp", meta.get("base_timestamp") if isinstance(meta, dict) else None)
    explicit_bounds = {}
    for k in (
        "history_start",
        "history_end",
        "current",
        "cutoff",
        "future_start",
        "future_end",
        "generation_window",
        "bwm_future_start",
        "bwm_future_end",
    ):
        if k in scenario:
            explicit_bounds[k] = {"where": "top", "value": scenario[k]}
        elif isinstance(meta, dict) and k in meta:
            explicit_bounds[k] = {"where": "metadata", "value": meta[k]}
        else:
            explicit_bounds[k] = "absent"

    # code-expected dt from SimEngine contract (not pickle fact for cutoff)
    code_dt_s = None
    code_hz = None
    if isinstance(sample_rate, (int, float)) and sample_rate:
        code_dt_s = float(sample_rate) * 0.05
        code_hz = 1.0 / code_dt_s if code_dt_s else None

    # Can we support 5-frame 2Hz history at candidate cutoffs? Only if log_length known.
    support = {}
    if isinstance(log_length, (int, np.integer)):
        ll = int(log_length)
        for c in (3, 4):
            # indices 0..c inclusive => c+1 frames
            support[f"cutoff_{c}_has_ge_5_history_frames"] = ll >= (c + 1) and (c + 1) >= 5
            support[f"cutoff_{c}_in_range"] = 0 <= c < ll
    else:
        support["note"] = "log_length_missing_cannot_assess_cutoff"

    return {
        "sample_rate": sample_rate if sample_rate is not None else "absent",
        "log_length": log_length if log_length is not None else "absent",
        "base_timestamp": base_ts if base_ts is not None else "absent",
        "explicit_time_bounds": explicit_bounds,
        "simengine_code_contract_dt_s": code_dt_s,
        "simengine_code_contract_hz": code_hz,
        "cutoff_support_if_full_trajectory": support,
        "note": (
            "No author cutoff is invented. If explicit bounds absent, scenario is "
            "treated as full trajectory without BWM-future boundary fact."
        ),
    }


def audit_ego(scenario: Dict[str, Any]) -> Dict[str, Any]:
    sdc = scenario.get("sdc_id", "absent")
    ot = scenario.get("object_track") if isinstance(scenario.get("object_track"), dict) else {}
    ego_token = None
    if isinstance(sdc, str) and sdc in ot:
        ego_token = sdc
    elif "ego" in ot:
        ego_token = "ego"
    ego_state_shapes = {}
    if ego_token and isinstance(ot.get(ego_token), dict):
        st = ot[ego_token].get("state")
        if isinstance(st, dict):
            for k, v in st.items():
                if isinstance(v, np.ndarray):
                    ego_state_shapes[k] = ndarray_leaf_stats(v)
                else:
                    ego_state_shapes[k] = type_name(v)
    return {
        "sdc_id": sdc if sdc != "absent" else "absent",
        "ego_token_resolved": ego_token if ego_token else "absent",
        "ego_state_shapes": ego_state_shapes,
    }


def audit_one_scenario(scenario: Dict[str, Any], scenario_key: str) -> Dict[str, Any]:
    top_keys = sorted(scenario.keys()) if isinstance(scenario, dict) else []
    required = {k: (k in scenario) for k in REQUIRED_FIRST_LEVEL}
    source_fields = field_presence(scenario, SOURCE_FIELD_CANDIDATES)
    variant_fields = field_presence(scenario, VARIANT_FIELD_CANDIDATES)
    ego_cond_fields = field_presence(scenario, EGO_COND_FIELD_CANDIDATES)
    bwm_prov_fields = field_presence(scenario, BWM_PROV_FIELD_CANDIDATES)
    reward_scan = find_keys_recursive(scenario, REWARD_FIELD_CANDIDATES, max_depth=5, max_keys=3000)
    sensor_scan = find_keys_recursive(scenario, SENSOR_FIELD_CANDIDATES, max_depth=5, max_keys=3000)
    # also deep-scan source/ego in nested metadata only via candidates
    deep_source = find_keys_recursive(scenario, SOURCE_FIELD_CANDIDATES, max_depth=4, max_keys=2000)
    deep_ego = find_keys_recursive(scenario, EGO_COND_FIELD_CANDIDATES, max_depth=4, max_keys=2000)

    ot = scenario.get("object_track") if isinstance(scenario.get("object_track"), dict) else {}
    mf = scenario.get("map_features") if isinstance(scenario.get("map_features"), dict) else {}
    dms = scenario.get("dynamic_map_states") if isinstance(scenario.get("dynamic_map_states"), dict) else {}
    log_length = scenario.get("log_length")
    if isinstance(log_length, np.integer):
        log_length = int(log_length)

    meta = scenario.get("metadata") if isinstance(scenario.get("metadata"), dict) else {}
    coord = meta.get("coordinate") if isinstance(meta, dict) else None
    transforms = {}
    for k in ("old_origin_in_current_coordinate", "digitaltwin_ego2globals", "coordinate"):
        if isinstance(meta, dict) and k in meta:
            v = meta[k]
            if isinstance(v, np.ndarray):
                transforms[k] = ndarray_leaf_stats(v)
            else:
                transforms[k] = {"type": type_name(v), "repr": str(v)[:200] if not isinstance(v, (dict, list)) else type_name(v)}
        else:
            transforms[k] = "absent"

    return {
        "scenario_key": scenario_key,
        "scenario_type": type_name(scenario),
        "top_keys": top_keys,
        "required_first_level": required,
        "id_field": scenario.get("id", "absent"),
        "name_field": scenario.get("name", "absent"),
        "token_field": scenario.get("token", "absent"),
        "source_fields": source_fields,
        "variant_fields": variant_fields,
        "ego_conditioning_fields": ego_cond_fields,
        "bwm_provenance_fields": bwm_prov_fields,
        "deep_source_scan": {"absent": deep_source["absent"], "present_keys": sorted(deep_source["present"].keys())},
        "deep_ego_scan": {"absent": deep_ego["absent"], "present_keys": sorted(deep_ego["present"].keys())},
        "time_anchor": audit_time_anchor(scenario),
        "ego": audit_ego(scenario),
        "agents": audit_agent_tracks(ot, log_length if isinstance(log_length, int) else None),
        "map": audit_map_features(mf) if mf else {"absent": True},
        "dynamic_map": audit_dynamic_map(dms) if dms else {"absent": True},
        "coordinate_transforms": transforms,
        "coordinate_declared": coord if coord is not None else "absent",
        "reward_scan": {"present_keys": sorted(reward_scan["present"].keys()), "absent": reward_scan["absent"]},
        "sensor_scan": {
            "present_keys": sorted(sensor_scan["present"].keys()),
            "absent": sensor_scan["absent"],
            "present_detail": {k: sensor_scan["present"][k][:2] for k in list(sensor_scan["present"])[:8]},
        },
        "dual_trajectory_tracks": {
            "original_track_key": "present" if "original_object_track" in scenario else "absent",
            "bwm_track_key": "present" if "bwm_object_track" in scenario else "absent",
            "note": "Only explicit dual-track keys counted; no trajectory-similarity inference.",
        },
    }


# ---------------------------------------------------------------------------
# Global pass + sampling
# ---------------------------------------------------------------------------

def reservoir_sample_keys(keys: Sequence[str], k: int, seed: int) -> List[str]:
    rng = random.Random(seed)
    if len(keys) <= k:
        return list(keys)
    sample: List[str] = []
    for i, key in enumerate(keys):
        if i < k:
            sample.append(key)
        else:
            j = rng.randint(0, i)
            if j < k:
                sample[j] = key
    return sample


def classify_pairing(summary: Dict[str, Any]) -> Dict[str, Any]:
    """A/B/C/D using only explicit evidence. No filename/suffix/similarity upgrades."""
    evidence: List[str] = []
    missing: List[str] = []

    source_cov = summary.get("field_coverage", {}).get("source_original", {})
    ego_cov = summary.get("field_coverage", {}).get("ego_conditioning", {})
    has_source = any(v for k, v in source_cov.items() if v and k != "note")
    # source_cov values are fractions; treat >0 as present somewhere
    source_present = any(isinstance(v, (int, float)) and v > 0 for v in source_cov.values())
    ego_present = any(isinstance(v, (int, float)) and v > 0 for v in ego_cov.values())

    n_sc = summary.get("n_scenarios", 0)
    required_ok = summary.get("required_first_level_all", False)
    has_tracks = summary.get("has_basic_tracks", False)
    has_map = summary.get("has_map_features", False)
    has_time = summary.get("has_sample_rate_and_log_length", False)
    explicit_cutoff = summary.get("explicit_cutoff_present", False)
    dual_tracks = summary.get("dual_trajectory_present", False)
    reward_present = summary.get("reward_fields_present", False)

    if n_sc <= 0 or not has_tracks:
        missing.append("basic_trajectory_structure")
        return {
            "grade": "D",
            "evidence": evidence + ["no_scenarios_or_missing_object_track"],
            "missing": missing,
            "allowed": ["record data/license/dependency blockers"],
            "forbidden": ["enter scoring or training pipeline"],
        }

    if not required_ok or not has_map:
        missing.append("required_first_level_or_map")
        grade = "D"
    else:
        grade = "C"  # default without source

    if source_present:
        evidence.append("explicit_source_or_original_field_nonzero_coverage")
    else:
        missing.append("explicit_source_original_parent_base_fields")
        evidence.append("source_fields_absent_no_filename_inference")

    if ego_present:
        evidence.append("explicit_ego_conditioning_field_nonzero_coverage")
    else:
        missing.append("explicit_ego_conditioning_fields")
        evidence.append("ego_conditioning_absent_no_trajectory_inference")

    if explicit_cutoff:
        evidence.append("explicit_cutoff_or_generation_window")
    else:
        missing.append("explicit_cutoff_generation_window")
        evidence.append("full_trajectory_without_author_cutoff_boundary")

    if dual_tracks:
        evidence.append("explicit_dual_original_bwm_tracks")
    else:
        missing.append("explicit_dual_tracks")

    # Upgrade rules (strict)
    if grade != "D":
        if source_present and has_time and required_ok and has_map:
            # still need same history/ego/map equality — without original download cannot prove A
            if ego_present and explicit_cutoff:
                grade = "B"
                evidence.append(
                    "has_source_and_ego_and_cutoff_but_strict_equality_vs_original_unproven_without_original_file"
                )
                missing.append("byte_or_field_equality_vs_original_requires_19gb_original")
            else:
                grade = "B"
                evidence.append("has_source_mapping_candidate_but_ego_or_cutoff_incomplete")
                if not ego_present:
                    missing.append("ego_conditioning_for_strict_pairing")
        else:
            grade = "C"
            evidence.append("external_generated_domain_without_explicit_source_pairing")

    # A requires ALL of: source, same history/current, same ego cond, same map/lights/agents,
    # explicit BWM future, map to future-pack. Without original we cannot prove equality → never A here
    # unless data itself embeds original snapshot + equality hashes (not observed as requirement).
    if (
        source_present
        and ego_present
        and explicit_cutoff
        and dual_tracks
        and summary.get("self_contained_original_equality_proof", False)
    ):
        grade = "A"
        evidence.append("self_contained_strict_pairing_proof")
    else:
        if grade == "A":
            grade = "B"
        missing.append("cannot_claim_A_without_proven_identical_history_ego_map_agents")

    allowed = {
        "A": [
            "use as strictly paired offline traffic source under same anchor/ego plan",
        ],
        "B": [
            "source-matched schema/coverage/domain-gap description",
            "future original confirmation if approved",
        ],
        "C": [
            "independent generated-domain quality/robustness description",
        ],
        "D": [
            "record blockers only",
        ],
    }
    forbidden = {
        "A": [
            "claim online reactive resampling",
            "pair to unsaved new ego candidates",
        ],
        "B": [
            "candidate-level causal reward comparison",
            "strict traffic-model disagreement claims",
            "treat unknown ego plan as unified conditioning",
        ],
        "C": [
            "any same-scene paired effect attribution",
            "strict or weak paired reward attribution",
        ],
        "D": [
            "enter existing scoring/training pipeline",
        ],
    }
    return {
        "grade": grade,
        "evidence": evidence,
        "missing": missing,
        "allowed": allowed[grade],
        "forbidden": forbidden[grade],
        "reward_fields_present": reward_present,
        "note": "Filename/ID-suffix/trajectory-similarity MUST NOT upgrade grade.",
    }


def build_future_pack_compat(summary: Dict[str, Any], pairing: Dict[str, Any]) -> Dict[str, Any]:
    def status(direct=False, convert=False, interpolate=False, missing=False, needs_original=False, note="") -> Dict[str, Any]:
        return {
            "direct": direct,
            "convert": convert,
            "interpolate": interpolate,
            "missing": missing,
            "needs_original": needs_original,
            "note": note,
        }

    has_tracks = summary.get("has_basic_tracks", False)
    has_map = summary.get("has_map_features", False)
    has_lights = summary.get("has_dynamic_map", False)
    has_sr = summary.get("has_sample_rate_and_log_length", False)
    ego_cond = any(
        isinstance(v, (int, float)) and v > 0
        for v in summary.get("field_coverage", {}).get("ego_conditioning", {}).values()
    )
    source = any(
        isinstance(v, (int, float)) and v > 0
        for v in summary.get("field_coverage", {}).get("source_original", {}).values()
    )
    code_hz = summary.get("dominant_code_contract_hz")
    hz_direct = code_hz == 2.0

    return {
        "scene_cutoff_history": status(
            convert=has_sr and has_tracks,
            missing=not has_sr,
            needs_original=False,
            note="Full trajectory present if log_length known; author cutoff absent → cutoff is consumer choice, not data fact.",
        ),
        "ego_conditioning": status(
            missing=not ego_cond,
            note="absent in pickle" if not ego_cond else "explicit field present",
        ),
        "agent_futures": status(
            direct=has_tracks,
            convert=has_tracks,
            note="Can slice object_track like extract_replay_futures; source label must be set to bwm-offline by adapter.",
        ),
        "token_type_size_valid": status(
            direct=has_tracks,
            note="size stays on scene object_track; futures carry type/pos/heading/vel/valid.",
        ),
        "map_traffic_lights": status(
            direct=has_map and has_lights,
            missing=not has_map,
            note="map_features flat dict; dynamic_map_states present if has_lights.",
        ),
        "frequency_horizon": status(
            direct=hz_direct,
            interpolate=has_sr and not hz_direct,
            note=f"dominant_code_contract_hz={code_hz}; interpolation rules only designed, not executed.",
        ),
        "coverage_future_hash_provenance": status(
            convert=True,
            missing=not source,
            note="future_hash/provenance can be computed; explicit BWM provenance fields mostly absent.",
        ),
        "needs_19gb_original_for_strict_equality": pairing.get("grade") in ("A", "B")
        or "byte_or_field_equality_vs_original_requires_19gb_original" in pairing.get("missing", []),
        "can_reuse_existing_pdm_sidecar_without_scoring_now": has_tracks and has_map,
        "same_anchor_as_replay_idm_nexus": status(
            missing=True,
            note="Unified cutoff NOT decided this stage; data does not force cutoff=3 or 4.",
        ),
    }


def audit_dataset(path: Path, seed: int, random_samples: int, max_wall_s: float, max_rss_gb: float) -> Dict[str, Any]:
    t0 = time.time()
    peak = rss_gb()

    def check_limits() -> None:
        nonlocal peak
        peak = max(peak, rss_gb())
        if time.time() - t0 > max_wall_s:
            raise TimeoutError("audit_wall_exceeded")
        if peak > max_rss_gb:
            raise MemoryError("audit_rss_exceeded")

    data = restricted_load(path)
    check_limits()
    if not isinstance(data, dict):
        return {
            "ok": False,
            "error": f"top_level_not_dict:{type_name(data)}",
            "wall_s": round(time.time() - t0, 3),
            "peak_rss_gb": round(peak, 3),
        }

    keys = list(data.keys())  # one key list; values remain in `data` only
    n = len(keys)
    key_types = Counter(type_name(k) for k in keys)
    # fixed samples: first, middle, last
    fixed_idx = []
    if n >= 1:
        fixed_idx.append(0)
    if n >= 2:
        fixed_idx.append(n // 2)
    if n >= 3:
        fixed_idx.append(n - 1)
    fixed_keys = [keys[i] for i in fixed_idx]
    rand_keys = [k for k in reservoir_sample_keys(keys, random_samples, seed) if k not in set(fixed_keys)]
    sample_keys = fixed_keys + rand_keys

    # Global scalar counters — do not retain scenario objects beyond loop body
    sample_rates: Counter = Counter()
    log_lengths: Counter = Counter()
    n_agents_hist: Counter = Counter()
    agent_types: Counter = Counter()
    required_ok_n = 0
    has_map_n = 0
    has_dms_n = 0
    has_tracks_n = 0
    source_field_hits: Counter = Counter()
    ego_field_hits: Counter = Counter()
    variant_field_hits: Counter = Counter()
    bwm_field_hits: Counter = Counter()
    reward_hits: Counter = Counter()
    sensor_hits: Counter = Counter()
    explicit_cutoff_n = 0
    dual_track_n = 0
    id_formats: Counter = Counter()

    sample_reports: List[Dict[str, Any]] = []

    for i, key in enumerate(keys):
        sc = data[key]
        if not isinstance(sc, dict):
            continue
        if all(k in sc for k in REQUIRED_FIRST_LEVEL):
            required_ok_n += 1
        if isinstance(sc.get("object_track"), dict) and sc["object_track"]:
            has_tracks_n += 1
            ot = sc["object_track"]
            n_agents_hist[len(ot)] += 1
            for _t, obj in ot.items():
                if isinstance(obj, dict):
                    agent_types[str(obj.get("type", "UNKNOWN"))] += 1
        if isinstance(sc.get("map_features"), dict) and sc["map_features"]:
            has_map_n += 1
        if isinstance(sc.get("dynamic_map_states"), dict) and sc["dynamic_map_states"]:
            has_dms_n += 1

        meta = sc.get("metadata") if isinstance(sc.get("metadata"), dict) else {}
        sr = sc.get("sample_rate", meta.get("sample_rate") if isinstance(meta, dict) else None)
        if sr is not None:
            sample_rates[str(sr)] += 1
        ll = sc.get("log_length")
        if ll is not None:
            log_lengths[str(int(ll) if isinstance(ll, (int, np.integer)) else ll)] += 1

        for name in SOURCE_FIELD_CANDIDATES:
            if name in sc or (isinstance(meta, dict) and name in meta):
                source_field_hits[name] += 1
        for name in EGO_COND_FIELD_CANDIDATES:
            if name in sc or (isinstance(meta, dict) and name in meta):
                ego_field_hits[name] += 1
        for name in VARIANT_FIELD_CANDIDATES:
            if name in sc or (isinstance(meta, dict) and name in meta):
                variant_field_hits[name] += 1
        for name in BWM_PROV_FIELD_CANDIDATES:
            if name in sc or (isinstance(meta, dict) and name in meta):
                bwm_field_hits[name] += 1

        # shallow reward/sensor at top/metadata only for global coverage (deep only on samples)
        for name in REWARD_FIELD_CANDIDATES:
            if name in sc or (isinstance(meta, dict) and name in meta):
                reward_hits[name] += 1
        for name in SENSOR_FIELD_CANDIDATES:
            if name in sc or (isinstance(meta, dict) and name in meta):
                sensor_hits[name] += 1

        for bname in ("cutoff", "generation_window", "future_start", "bwm_future_start"):
            if bname in sc or (isinstance(meta, dict) and bname in meta):
                explicit_cutoff_n += 1
                break
        if "original_object_track" in sc or "bwm_object_track" in sc:
            dual_track_n += 1

        sid = sc.get("id", key)
        id_formats[type_name(sid)] += 1

        if key in sample_keys:
            sample_reports.append(audit_one_scenario(sc, str(key)))

        # release reference promptly
        del sc
        if i % 200 == 0:
            check_limits()

    # Field coverage fractions
    def frac_map(counter: Counter) -> Dict[str, float]:
        return {k: round(v / max(n, 1), 6) for k, v in sorted(counter.items())}

    # Deep scans on samples for reward/sensor nested presence
    sample_reward_present = set()
    sample_sensor_present = set()
    sample_source_deep = set()
    sample_ego_deep = set()
    for rep in sample_reports:
        sample_reward_present.update(rep.get("reward_scan", {}).get("present_keys", []))
        sample_sensor_present.update(rep.get("sensor_scan", {}).get("present_keys", []))
        sample_source_deep.update(rep.get("deep_source_scan", {}).get("present_keys", []))
        sample_ego_deep.update(rep.get("deep_ego_scan", {}).get("present_keys", []))

    dominant_sr = None
    if sample_rates:
        dominant_sr = sample_rates.most_common(1)[0][0]
    code_hz = None
    try:
        if dominant_sr is not None:
            code_hz = 1.0 / (float(dominant_sr) * 0.05)
    except Exception:
        code_hz = None

    summary = {
        "ok": True,
        "n_scenarios": n,
        "top_level_type": "dict",
        "key_type_counts": dict(key_types),
        "required_first_level_all": required_ok_n == n and n > 0,
        "required_ok_frac": round(required_ok_n / max(n, 1), 6),
        "has_basic_tracks": has_tracks_n > 0,
        "tracks_frac": round(has_tracks_n / max(n, 1), 6),
        "has_map_features": has_map_n > 0,
        "map_frac": round(has_map_n / max(n, 1), 6),
        "has_dynamic_map": has_dms_n > 0,
        "dynamic_map_frac": round(has_dms_n / max(n, 1), 6),
        "has_sample_rate_and_log_length": bool(sample_rates) and bool(log_lengths),
        "sample_rate_distribution": dict(sample_rates.most_common(20)),
        "log_length_distribution": dict(log_lengths.most_common(20)),
        "dominant_code_contract_hz": code_hz,
        "n_agents_distribution_top": n_agents_hist.most_common(20),
        "agent_type_counts": dict(agent_types),
        "field_coverage": {
            "source_original": frac_map(source_field_hits),
            "ego_conditioning": frac_map(ego_field_hits),
            "variant": frac_map(variant_field_hits),
            "bwm_provenance": frac_map(bwm_field_hits),
            "reward_top_or_metadata": frac_map(reward_hits),
            "sensor_top_or_metadata": frac_map(sensor_hits),
            "note": "Coverage from explicit keys only; absent fields have 0 coverage and are not inferred.",
        },
        "source_field_hit_counts": dict(source_field_hits),
        "ego_field_hit_counts": dict(ego_field_hits),
        "explicit_cutoff_frac": round(explicit_cutoff_n / max(n, 1), 6),
        "explicit_cutoff_present": explicit_cutoff_n > 0,
        "dual_trajectory_frac": round(dual_track_n / max(n, 1), 6),
        "dual_trajectory_present": dual_track_n > 0,
        "id_type_counts": dict(id_formats),
        "sample_deep_reward_keys": sorted(sample_reward_present),
        "sample_deep_sensor_keys": sorted(sample_sensor_present),
        "sample_deep_source_keys": sorted(sample_source_deep),
        "sample_deep_ego_keys": sorted(sample_ego_deep),
        "reward_fields_present": bool(reward_hits) or bool(sample_reward_present),
        "self_contained_original_equality_proof": False,
        "fixed_sample_keys": [str(k) for k in fixed_keys],
        "random_sample_keys": [str(k) for k in rand_keys],
        "samples": sample_reports,
        "wall_s": round(time.time() - t0, 3),
        "peak_rss_gb": round(max(peak, rss_gb()), 3),
    }

    pairing = classify_pairing(summary)
    compat = build_future_pack_compat(summary, pairing)
    summary["pairing_grade"] = pairing
    summary["traffic_future_pack_compatibility"] = compat

    # Drop heavy sample detail if JSON would be huge — keep essential
    del data
    check_limits()
    summary["peak_rss_gb"] = round(max(peak, rss_gb()), 3)
    summary["wall_s"] = round(time.time() - t0, 3)
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cmd_scan(args: argparse.Namespace) -> int:
    verify_file_gate(args.input, args.expected_size, args.expected_sha256)
    result = scan_pickle_opcodes(args.input, args.max_wall_s, args.max_rss_gb)
    out = {
        "phase": "scan",
        "file": verify_file_gate(args.input, args.expected_size, args.expected_sha256),
        "scan": result,
    }
    if args.output:
        save_json(args.output, out)
    print(json.dumps({"ok": result["ok"], "n_forbidden": result["n_forbidden"], "wall_s": result["wall_s"], "peak_rss_gb": result["peak_rss_gb"], "protocols": result["protocols"], "globals": result["globals"]}, indent=2))
    return 0 if result["ok"] else 2


def cmd_audit(args: argparse.Namespace) -> int:
    file_meta = verify_file_gate(args.input, args.expected_size, args.expected_sha256)
    # safety: scan first in-process briefly? Plan uses separate scan command; still refuse if scan fails when --require-scan-ok
    if args.require_prior_scan:
        if not args.prior_scan_json or not Path(args.prior_scan_json).exists():
            raise RuntimeError("prior scan json required")
        prior = json.loads(Path(args.prior_scan_json).read_text())
        if not prior.get("scan", {}).get("ok", False):
            raise RuntimeError("prior_scan_not_ok")

    summary = audit_dataset(
        args.input,
        seed=args.seed,
        random_samples=args.random_samples,
        max_wall_s=args.max_wall_s,
        max_rss_gb=args.max_rss_gb,
    )
    out = {
        "phase": "audit",
        "provenance": {
            "hf_repo": "OpenDriveLab/WorldEngine",
            "hf_revision": "8728616abaf090d195b3bdc7af6aacde40271145",
            "relative_path": "data/sim_engine/scenarios/augmented/navtrain_50pct_collision/all_scenarios.pkl",
            "license": "CC-BY-NC-SA-4.0",
            "expected_size": args.expected_size,
            "expected_sha256": args.expected_sha256,
            **file_meta,
        },
        "summary": summary,
    }
    if args.output:
        save_json(args.output, out)
    slim = {
        "ok": summary.get("ok"),
        "n_scenarios": summary.get("n_scenarios"),
        "pairing_grade": summary.get("pairing_grade", {}).get("grade"),
        "source_coverage": summary.get("field_coverage", {}).get("source_original"),
        "ego_conditioning_coverage": summary.get("field_coverage", {}).get("ego_conditioning"),
        "explicit_cutoff_frac": summary.get("explicit_cutoff_frac"),
        "sample_rate_distribution": summary.get("sample_rate_distribution"),
        "log_length_distribution": summary.get("log_length_distribution"),
        "wall_s": summary.get("wall_s"),
        "peak_rss_gb": summary.get("peak_rss_gb"),
        "needs_19gb_original": summary.get("traffic_future_pack_compatibility", {}).get(
            "needs_19gb_original_for_strict_equality"
        ),
    }
    print(json.dumps(slim, indent=2, ensure_ascii=False))
    return 0 if summary.get("ok") else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--input", type=Path, required=True)
    common.add_argument("--expected-size", type=int, required=True)
    common.add_argument("--expected-sha256", type=str, required=True)
    common.add_argument("--max-rss-gb", type=float, default=64.0)
    common.add_argument("--max-wall-s", type=float, default=1800.0)
    common.add_argument("--output", type=Path, default=None)

    sp = sub.add_parser("scan", parents=[common])
    sp.set_defaults(func=cmd_scan)

    ap = sub.add_parser("audit", parents=[common])
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--random-samples", type=int, default=10)
    ap.add_argument("--require-prior-scan", action="store_true")
    ap.add_argument("--prior-scan-json", type=Path, default=None)
    ap.set_defaults(func=cmd_audit)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    # Pin threads
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("PYTHONHASHSEED", "0")
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except Exception as e:
        print(json.dumps({"ok": False, "error": repr(e), "trace": traceback.format_exc()[-2000:]}, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
