#!/usr/bin/env python3
"""Select plan_idx = median path-length on DAC∧Comfort∧Direction=1 static feasible set.

Writes scene_meta.json provenance. Never silently defaults to 1333 or 710.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

WE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WE_ROOT))
sys.path.insert(0, str(WE_ROOT / "scripts"))

from frozen_state_lib import DEFAULT_CUTOFF, DEFAULT_HORIZON, load_scene_dict, save_json  # noqa: E402
from run_nexus_sidecar_smoke import score_static_pdm_gate  # noqa: E402


def path_length_m(vocab_row: np.ndarray, ego_center: np.ndarray, ego_heading: float) -> Tuple[float, str]:
    from adapters.traffic_models import nexus as nx

    fut = nx.vocab_candidate_to_nexus_ego_future(vocab_row, ego_center, ego_heading)
    pl = float(fut["stats"]["real_path_length_m"])
    h = str(fut.get("real_hash") or "")
    return pl, h


def select_median(
    keep_idx: List[int],
    vocab: np.ndarray,
    ego_center: np.ndarray,
    ego_heading: float,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for idx in keep_idx:
        pl, rh = path_length_m(vocab[int(idx)], ego_center, ego_heading)
        rows.append({"plan_idx": int(idx), "path_length_m": pl, "real_hash": rh})
    rows.sort(key=lambda r: (r["path_length_m"], r["plan_idx"]))
    n = len(rows)
    if n == 0:
        raise RuntimeError("empty static feasible set")
    mid = n // 2  # upper-median for even n (same as nexus median-anchor convention)
    chosen = rows[mid]
    pls = [r["path_length_m"] for r in rows]
    return {
        "plan_idx": int(chosen["plan_idx"]),
        "path_length_m": float(chosen["path_length_m"]),
        "real_hash": chosen["real_hash"],
        "n_feasible": n,
        "path_length_percentiles": {
            "p0": float(pls[0]),
            "p25": float(np.percentile(pls, 25)),
            "p50": float(np.percentile(pls, 50)),
            "p75": float(np.percentile(pls, 75)),
            "p100": float(pls[-1]),
            "selected_rank_0based": mid,
            "selected_is_list_median_index": True,
        },
        "feasible_plan_idx_sha256": hashlib.sha256(
            ",".join(str(r["plan_idx"]) for r in rows).encode()
        ).hexdigest(),
        "path_length_vec_sha256": hashlib.sha256(
            np.asarray(pls, dtype=np.float64).tobytes()
        ).hexdigest(),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-pkl", type=Path, required=True)
    ap.add_argument("--out-meta", type=Path, required=True)
    ap.add_argument(
        "--vocab",
        type=Path,
        default=Path("/mnt/cpfs/prediction/lyyy/myself/WE/data/hf/data/alg_engine/test_8192_kmeans.npy"),
    )
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--asset-folder", type=str, required=True)
    ap.add_argument("--cutoff", type=int, default=DEFAULT_CUTOFF)
    ap.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    ap.add_argument("--token", type=str, default="")
    ap.add_argument("--bucket", type=str, default="")
    ap.add_argument("--scene-id", type=str, default="")
    args = ap.parse_args()

    we_root = WE_ROOT / "upstream/WorldEngine"
    os.environ.setdefault("WORLDENGINE_ROOT", str(we_root))
    os.environ.setdefault("SIMENGINE_ROOT", str(we_root / "projects/SimEngine"))
    maps = Path("/mnt/cpfs/prediction/lyyy/myself/WE/data/maps/extracted")
    if maps.exists():
        os.environ["NUPLAN_MAPS_ROOT"] = str(maps)

    scene = load_scene_dict(args.scene_pkl)
    vocab = np.load(args.vocab)
    args.work_dir.mkdir(parents=True, exist_ok=True)

    gate = score_static_pdm_gate(
        scene=scene,
        vocab=vocab,
        vocab_path=args.vocab,
        work_dir=args.work_dir / "static_gate_workdir",
        asset_folder=args.asset_folder,
        cutoff=int(args.cutoff),
        horizon=int(args.horizon),
    )
    if not gate.get("ok") or gate.get("n_keep", 0) < 1:
        save_json(
            args.out_meta,
            {
                "ok": False,
                "error": "static_feasible_empty",
                "gate_counts": gate.get("counts"),
            },
        )
        return 2

    sdc = scene["sdc_id"]
    ego_st = scene["object_track"][sdc]["state"]
    ego_center = np.asarray(ego_st["position"], dtype=np.float64)[int(args.cutoff), :2]
    ego_heading = float(np.asarray(ego_st["heading"]).reshape(-1)[int(args.cutoff)])
    sel = select_median(gate["keep_idx"], vocab, ego_center, ego_heading)

    meta = {
        "ok": True,
        "token": args.token,
        "bucket": args.bucket,
        "scene_id": args.scene_id or str(scene.get("id", "")),
        "cutoff": int(args.cutoff),
        "horizon": int(args.horizon),
        "ego_conditioning": {
            "source": "static_feasible_median_path_length",
            "rule": "DAC==1 AND Comfort==1 AND Direction==1; median path_length_m on that set",
            "plan_idx": sel["plan_idx"],
            "n_feasible": sel["n_feasible"],
            "path_length_m": sel["path_length_m"],
            "path_length_percentiles": sel["path_length_percentiles"],
            "feasible_plan_idx_sha256": sel["feasible_plan_idx_sha256"],
            "path_length_vec_sha256": sel["path_length_vec_sha256"],
            "requested_ego_real_hash": sel["real_hash"],
            "not_smoke_nr_plan_idx_1333": True,
            "forbidden_silent_default_1333": True,
            "forbidden_legacy_nexus_710_unless_computed": True,
            "equals_legacy_710": bool(sel["plan_idx"] == 710),
            "equals_smoke_1333": bool(sel["plan_idx"] == 1333),
            "note": (
                "本轮 conditioning ≠ 冒烟 NR plan_idx=1333，故效应量不可与单场景冒烟数字直接数值对比。"
            ),
        },
        "static_gate_counts": gate.get("counts"),
        "static_gate_scores_pkl": gate.get("scores_pkl"),
        "vocab_path": str(args.vocab),
        "vocab_sha256": hashlib.sha256(Path(args.vocab).read_bytes()).hexdigest(),
    }
    args.out_meta.parent.mkdir(parents=True, exist_ok=True)
    save_json(args.out_meta, meta)
    print(json.dumps({"plan_idx": sel["plan_idx"], "n_feasible": sel["n_feasible"], "out": str(args.out_meta)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
