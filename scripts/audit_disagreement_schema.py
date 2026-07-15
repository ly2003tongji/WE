#!/usr/bin/env python3
"""Read-only audit for disagreement dense-reward outputs.

Prints schema / fingerprint summaries only. Does not copy or write raw PDM arrays.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

PDM_EXPECTED_KEYS = [
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "lane_keeping",
    "ego_progress",
    "time_to_collision_within_bound",
    "comfort",
    "driving_direction_compliance",
    "score",
    "IL_plan_idx",
    "target_traj",
]


def sha256_file(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_json(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)
    return sha256_bytes(payload.encode("utf-8"))


def summarize_array(arr: np.ndarray) -> Dict[str, Any]:
    flat = np.asarray(arr)
    info: Dict[str, Any] = {
        "shape": list(flat.shape),
        "dtype": str(flat.dtype),
        "nbytes": int(flat.nbytes),
        "sha256_of_bytes": sha256_bytes(np.ascontiguousarray(flat).tobytes()),
    }
    if flat.size == 0:
        info.update({"min": None, "max": None, "mean": None, "finite_ratio": None})
        return info
    finite = np.isfinite(flat.astype(np.float64, copy=False))
    info["finite_ratio"] = float(finite.mean())
    if finite.any():
        vals = flat.astype(np.float64, copy=False)[finite]
        info["min"] = float(vals.min())
        info["max"] = float(vals.max())
        info["mean"] = float(vals.mean())
    else:
        info.update({"min": None, "max": None, "mean": None})
    if flat.dtype == np.bool_ or set(np.unique(flat).tolist()).issubset({0, 1, 0.0, 1.0}):
        uniq = np.unique(flat)
        info["unique_preview"] = [ _to_py(u) for u in uniq[:8] ]
    return info


def _to_py(x: Any) -> Any:
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def summarize_value(v: Any) -> Dict[str, Any]:
    if isinstance(v, np.ndarray):
        return {"type": "ndarray", **summarize_array(v)}
    if isinstance(v, (list, tuple)):
        try:
            arr = np.asarray(v)
            if arr.dtype != object:
                return {"type": type(v).__name__, **summarize_array(arr)}
        except Exception:
            pass
        return {"type": type(v).__name__, "len": len(v)}
    if isinstance(v, (int, float, bool, str)) or v is None:
        return {"type": type(v).__name__, "value": v}
    if isinstance(v, (np.integer, np.floating, np.bool_)):
        return {"type": type(v).__name__, "value": _to_py(v)}
    return {"type": type(v).__name__, "repr": repr(v)[:200]}


def load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def parse_pdm_filename(name: str) -> Tuple[str, int]:
    # {scene_id}_step_{step}_scores.pkl
    if not name.endswith("_scores.pkl") or "_step_" not in name:
        raise ValueError(f"unexpected pdm filename: {name}")
    body = name[: -len("_scores.pkl")]
    scene_id, step_s = body.rsplit("_step_", 1)
    return scene_id, int(step_s)


def audit_vocab(vocab_path: Path) -> Dict[str, Any]:
    arr = np.load(vocab_path, mmap_mode="r")
    info = {
        "path": str(vocab_path),
        "sha256": sha256_file(vocab_path),
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "nbytes": int(Path(vocab_path).stat().st_size),
    }
    # Do not materialize full copy; sample endpoints only.
    info["sample_first_row_l2"] = float(np.linalg.norm(arr[0]))
    info["sample_last_row_l2"] = float(np.linalg.norm(arr[-1]))
    return info


def audit_plan_idx(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"exists": False, "path": str(path)}
    rows = []
    with path.open() as f:
        header = f.readline().strip().split(",")
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 3:
                continue
            rows.append({"prefix": parts[0], "step": int(parts[1]), "plan_idx": int(parts[2])})
    idxs = [r["plan_idx"] for r in rows]
    return {
        "exists": True,
        "path": str(path),
        "n_rows": len(rows),
        "header": header,
        "plan_idx_min": min(idxs) if idxs else None,
        "plan_idx_max": max(idxs) if idxs else None,
        "rows_preview": rows[:5],
    }


def meta_fingerprint_from_frame(frame: Dict[str, Any]) -> Dict[str, Any]:
    """Best-effort fingerprint fields available from meta_datas frame dict."""
    keys_of_interest = [
        "token",
        "frame_idx",
        "timestamp",
        "log_name",
        "scene_token",
        "lidar2ego",
        "ego2global",
        "can_bus",
        "gt_boxes",
        "gt_names",
        "gt_velocity_3d",
        "track_tokens",
        "instance_tokens",
        "original_track_tokens",
    ]
    present = {k: k in frame for k in keys_of_interest}
    payload: OrderedDict[str, Any] = OrderedDict()
    payload["token"] = frame.get("token")
    payload["frame_idx"] = frame.get("frame_idx")
    payload["timestamp"] = frame.get("timestamp")
    payload["log_name"] = frame.get("log_name")
    payload["scene_token"] = frame.get("scene_token")

    # ego / pose proxies
    for k in ("can_bus", "lidar2ego", "ego2global"):
        if k in frame and isinstance(frame[k], np.ndarray):
            payload[k] = {
                "shape": list(frame[k].shape),
                "dtype": str(frame[k].dtype),
                "sha256": sha256_bytes(np.ascontiguousarray(frame[k]).tobytes()),
            }
        elif k in frame:
            payload[k] = summarize_value(frame[k])

    # agents
    for k in ("gt_boxes", "gt_velocity_3d"):
        if k in frame and isinstance(frame[k], np.ndarray):
            payload[k] = {
                "shape": list(frame[k].shape),
                "dtype": str(frame[k].dtype),
                "sha256": sha256_bytes(np.ascontiguousarray(frame[k]).tobytes()),
            }
    if "gt_names" in frame:
        names = [str(x) for x in list(np.asarray(frame["gt_names"]).reshape(-1))]
        payload["gt_names"] = names
        payload["gt_names_sha256"] = sha256_json(names)
    for k in ("track_tokens", "instance_tokens", "original_track_tokens"):
        if k in frame:
            toks = [str(x) for x in list(frame[k])]
            payload[k] = toks
            payload[f"{k}_sha256"] = sha256_json(toks)

    # valid mask proxy: non-zero boxes / present tokens
    if "gt_boxes" in frame and isinstance(frame["gt_boxes"], np.ndarray):
        boxes = frame["gt_boxes"]
        if boxes.ndim >= 2 and boxes.shape[0] > 0:
            valid = ~np.all(boxes[:, :2] == 0, axis=1)
            payload["valid_mask_proxy_sha256"] = sha256_bytes(np.ascontiguousarray(valid.astype(np.uint8)).tobytes())
            payload["n_valid_agents_proxy"] = int(valid.sum())
        else:
            payload["n_valid_agents_proxy"] = 0

    missing_required = [
        "map_version",
        "map_lane_ids",
        "roadblock_ids",
        "traffic_light_ids_states",
        "vocabulary_hash",
        "scorer_config",
        "reward_horizon",
        "sample_rate",
        "agent_history_states",
        "ego_history_states",
    ]
    return {
        "present_keys": present,
        "available_fingerprint_fields": list(payload.keys()),
        "missing_for_strict_pairing": missing_required,
        "canonical_payload": payload,
        "sha256": sha256_json(payload),
    }


def audit_one_pdm(path: Path) -> Dict[str, Any]:
    obj = load_pickle(path)
    scene_id, step = parse_pdm_filename(path.name)
    keys = list(obj.keys()) if isinstance(obj, dict) else []
    field_summaries = {}
    candidate_count = None
    shape_ok = True
    problems = []
    if not isinstance(obj, dict):
        problems.append(f"pdm object is {type(obj)}, expected dict")
    else:
        for k in keys:
            field_summaries[k] = summarize_value(obj[k])
        for k in PDM_EXPECTED_KEYS:
            if k not in obj:
                problems.append(f"missing key: {k}")
                continue
            s = field_summaries[k]
            if k in ("IL_plan_idx",):
                if s.get("type") not in ("int", "int64", "int32", "ndarray") and "value" not in s:
                    problems.append(f"{k} unexpected type {s.get('type')}")
                continue
            if k == "target_traj":
                continue
            shape = s.get("shape")
            if not shape:
                problems.append(f"{k} has no shape")
                shape_ok = False
                continue
            if candidate_count is None:
                candidate_count = int(shape[0])
            if int(shape[0]) != 8192:
                problems.append(f"{k} shape[0]={shape[0]} != 8192")
                shape_ok = False
            if candidate_count is not None and int(shape[0]) != candidate_count:
                problems.append(f"{k} candidate dim mismatch")
                shape_ok = False
    return {
        "path": str(path),
        "scene_id": scene_id,
        "step": step,
        "keys": keys,
        "candidate_count": candidate_count,
        "shape_ok": shape_ok and candidate_count == 8192,
        "field_summaries": field_summaries,
        "problems": problems,
        # Future provenance cannot be recovered from pkl alone.
        "future_provenance_in_pkl": "absent",
    }


def find_pdms(root: Path) -> List[Path]:
    base = root / "WE_output" / "openscene_format" / "pdms_pkl"
    if not base.exists():
        # sometimes nested differently
        hits = list(root.rglob("*_scores.pkl"))
        return sorted(hits)
    return sorted(base.glob("*_scores.pkl"))


def find_meta(root: Path) -> List[Path]:
    base = root / "WE_output" / "openscene_format" / "meta_datas"
    if base.exists():
        return sorted(base.glob("*.pkl"))
    return sorted((root / "WE_output").rglob("meta_datas/*.pkl"))


def read_resource_summary(root: Path) -> Dict[str, Any]:
    p = root / "resource_monitor" / "resource_summary.txt"
    meta = root / "resource_monitor" / "run_meta.txt"
    gate = root / "resource_monitor" / "hard_gate.txt"
    out: Dict[str, Any] = {"exists": p.exists()}
    if p.exists():
        for line in p.read_text().splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                out[k] = v
    if meta.exists():
        out["run_meta"] = {
            k: v
            for line in meta.read_text().splitlines()
            if "=" in line
            for k, v in [line.split("=", 1)]
        }
    if gate.exists():
        out["hard_gate_text"] = gate.read_text().strip()
    return out


def evaluate_resource_gates(summary: Dict[str, Any]) -> Dict[str, Any]:
    wall = float(summary.get("wall_s", "nan"))
    rss = float(summary.get("peak_rss_gb", "nan"))
    out_gb = float(summary.get("final_output_gb", "nan"))
    hard = summary.get("hard_gate_text") or summary.get("hard_gate") or "none"
    checks = {
        "wall_le_3600": wall <= 3600 if wall == wall else False,
        "rss_le_64gb": rss <= 64 if rss == rss else False,
        "output_le_5gb": out_gb <= 5 if out_gb == out_gb else False,
        "no_hard_gate": (hard in ("", "none", "None") or hard is None),
    }
    checks["pass"] = all(checks.values())
    return checks


def compare_fingerprints(nr: Dict[str, Any], r: Dict[str, Any]) -> Dict[str, Any]:
    nr_fps = { (x["scene_id"], x["step"]): x for x in nr.get("frame_fingerprints", []) }
    r_fps = { (x["scene_id"], x["step"]): x for x in r.get("frame_fingerprints", []) }
    common = sorted(set(nr_fps) & set(r_fps))
    only_nr = sorted(set(nr_fps) - set(r_fps))
    only_r = sorted(set(r_fps) - set(nr_fps))
    strict_matches = []
    mismatches = []
    for key in common:
        a = nr_fps[key]["fingerprint_sha256"]
        b = r_fps[key]["fingerprint_sha256"]
        item = {"scene_id": key[0], "step": key[1], "nr_fp": a, "r_fp": b, "equal": a == b}
        if a == b:
            strict_matches.append(item)
        else:
            mismatches.append(item)
    return {
        "n_nr_frames": len(nr_fps),
        "n_r_frames": len(r_fps),
        "n_common_keys": len(common),
        "n_strict_matches": len(strict_matches),
        "n_mismatches": len(mismatches),
        "only_nr": only_nr[:20],
        "only_r": only_r[:20],
        "strict_match_preview": strict_matches[:10],
        "mismatch_preview": mismatches[:10],
        "common_support_coverage_nr": (len(common) / len(nr_fps)) if nr_fps else None,
        "common_support_coverage_r": (len(common) / len(r_fps)) if r_fps else None,
        "unaligned_reasons": {
            "missing_on_other_source": len(only_nr) + len(only_r),
            "fingerprint_mismatch_on_common_key": len(mismatches),
            "note": (
                "Same scene_id/step is NOT sufficient for strict pairing. "
                "Only equal canonical state fingerprints qualify. "
                "Diverged frames must be labeled rollout-conditioned difference."
            ),
        },
    }


def audit_root(root: Path, source: str, vocab_info: Dict[str, Any]) -> Dict[str, Any]:
    root = root.resolve()
    pdms = find_pdms(root)
    metas = find_meta(root)
    plan = audit_plan_idx(root / "plan_traj" / "plan_idx.csv")
    resource = read_resource_summary(root)
    gates = evaluate_resource_gates(resource)

    pdm_audits = [audit_one_pdm(p) for p in pdms]
    frame_fingerprints = []
    meta_summary = {"exists": bool(metas), "n_files": len(metas), "files": [str(p) for p in metas]}

    # Build best-effort fingerprints from meta if present.
    for mp in metas:
        data = load_pickle(mp)
        if not isinstance(data, list):
            continue
        for frame in data:
            if not isinstance(frame, dict):
                continue
            fp = meta_fingerprint_from_frame(frame)
            # Attach vocab/scorer static fields into canonical hash inputs conceptually,
            # but keep separate because meta does not store them.
            frame_fingerprints.append(
                {
                    "source": source,
                    "meta_file": str(mp),
                    "scene_id": frame.get("log_name") or frame.get("scene_token"),
                    "token": frame.get("token"),
                    "step": frame.get("frame_idx"),
                    "fingerprint_sha256": fp["sha256"],
                    "missing_for_strict_pairing": fp["missing_for_strict_pairing"],
                    "available_fingerprint_fields": fp["available_fingerprint_fields"],
                }
            )

    # If no meta fingerprints, still record pdm keys as alignment keys (NOT strict).
    alignment_keys = [{"scene_id": a["scene_id"], "step": a["step"]} for a in pdm_audits]

    shape_problems = [a for a in pdm_audits if a["problems"]]
    return {
        "source": source,
        "root": str(root),
        "vocab": vocab_info,
        "plan_idx": plan,
        "resource": resource,
        "resource_gates": gates,
        "pdms_n": len(pdms),
        "pdms": pdm_audits,
        "meta": meta_summary,
        "frame_fingerprints": frame_fingerprints,
        "alignment_keys_from_pdm": alignment_keys,
        "all_pdm_shape_ok": all(a["shape_ok"] for a in pdm_audits) if pdm_audits else False,
        "n_pdm_with_problems": len(shape_problems),
        "code_facts": {
            "dense_reward_writes": (
                "projects/SimEngine/worldengine/manager/dense_reward_manager.py:189-210"
            ),
            "observations_from_scene_log": (
                "dense_reward_manager.py:145-148 uses convert_to_detections_tracks_from_scene; "
                "TODO comment says 'NR or R'. Scoring uses observations_list "
                "(dense_reward_manager.py:233-259), so 8192-candidate futures are shared log "
                "futures unless patched."
            ),
            "agent_input_tracks_collected_but_not_used_in_score": (
                "dense_reward_manager.py:184-187 appends detection_tracks_list from agent_input, "
                "but compute_pdm_scores reads observations_list."
            ),
            "synthetic_consumer": (
                "projects/AlgEngine/mmdet3d_plugin/datasets/navsim_openscene_synthetic.py:100-118"
            ),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nr-root", type=str, default="")
    ap.add_argument("--r-root", type=str, default="")
    ap.add_argument("--vocab", type=str, required=True)
    ap.add_argument("--out-json", type=str, default="")
    ap.add_argument(
        "--require-state-fingerprint-match",
        action="store_true",
        help="Exit non-zero if no strict fingerprint matches exist when both roots are provided.",
    )
    args = ap.parse_args()

    vocab_info = audit_vocab(Path(args.vocab))
    report: Dict[str, Any] = {
        "vocab": vocab_info,
        "strict_pairing_definition": {
            "required_equal_fields": [
                "ego_current_and_history",
                "agents_current_and_history_sorted_by_track_token",
                "agent_types_sizes_valid_mask",
                "map_version_and_lane_roadblock_ids",
                "traffic_light_ids_states_times",
                "vocabulary_file_hash_shape_dtype",
                "pdm_scorer_commit_config_sample_rate_horizon",
            ],
            "compared_not_in_fingerprint": [
                "traffic_model_future_trajectories",
                "traffic_model_seed",
                "future_coverage",
            ],
            "note": (
                "Same scene/frame/index names cannot substitute for state equality. "
                "Diverged frames are rollout-conditioned difference."
            ),
        },
    }

    if args.nr_root:
        report["nr"] = audit_root(Path(args.nr_root), "NR", vocab_info)
    if args.r_root:
        report["r"] = audit_root(Path(args.r_root), "R", vocab_info)

    if "nr" in report and "r" in report:
        report["nr_r_fingerprint_compare"] = compare_fingerprints(report["nr"], report["r"])
        # Compare PDM field schemas on first available files.
        if report["nr"]["pdms"] and report["r"]["pdms"]:
            nr0 = report["nr"]["pdms"][0]
            r0 = report["r"]["pdms"][0]
            report["pdm_schema_compare_first"] = {
                "nr_keys": nr0["keys"],
                "r_keys": r0["keys"],
                "keys_equal": nr0["keys"] == r0["keys"],
                "nr_candidate_count": nr0["candidate_count"],
                "r_candidate_count": r0["candidate_count"],
            }
        report["strict_pairing_achieved"] = (
            report["nr_r_fingerprint_compare"]["n_strict_matches"] > 0
            and report["nr"].get("meta", {}).get("exists")
            and report["r"].get("meta", {}).get("exists")
            and not report["nr"]["frame_fingerprints"][0]["missing_for_strict_pairing"]
            if report["nr"].get("frame_fingerprints")
            else False
        )
        # Even if meta hashes match, missing map/light/vocab/scorer fields => not strict.
        if report["nr"].get("frame_fingerprints"):
            missing = report["nr"]["frame_fingerprints"][0]["missing_for_strict_pairing"]
            report["strict_pairing_achieved"] = False
            report["strict_pairing_blockers"] = missing
            report["strict_pairing_status"] = "not_achieved_incomplete_fingerprint_fields"
        else:
            report["strict_pairing_achieved"] = False
            report["strict_pairing_status"] = "not_achieved_no_meta_or_no_fingerprint"

    text = json.dumps(report, indent=2, ensure_ascii=False, default=str)
    print(text)
    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
        print(f"Wrote {out}", file=sys.stderr)

    if args.require_state_fingerprint_match:
        if not report.get("strict_pairing_achieved", False):
            print("require-state-fingerprint-match: STRICT PAIRING NOT ACHIEVED", file=sys.stderr)
            # Non-zero but do not fail the engineering audit hard; caller decides.
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
