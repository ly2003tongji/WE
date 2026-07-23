#!/usr/bin/env python3
"""Build train-side ≤10 scene list from navtrain rare yaml + original pkls.

Hard isolation: ∩ navtest_failures tokens = ∅; ∩ engineering smoke scene-ids = ∅.
Bucket: 5 collision-exclusive + 5 ep-exclusive. Offroad deferred.
"""

from __future__ import annotations

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import yaml

WE_ROOT = Path(__file__).resolve().parents[1]
YAML_DIR = (
    WE_ROOT
    / "upstream/WorldEngine/projects/AlgEngine/configs/navsim_splits"
)
NAVTEST_YAML = YAML_DIR / "navtest_split/navtest_failures_filtered.yaml"
COLL_YAML = YAML_DIR / "navtrain_split/e2e_vadv2_50pct_rare/navtrain_50pct_collision.yaml"
EP_YAML = YAML_DIR / "navtrain_split/e2e_vadv2_50pct_rare/navtrain_50pct_ep_1pct.yaml"
OFF_YAML = YAML_DIR / "navtrain_split/e2e_vadv2_50pct_rare/navtrain_50pct_off_road.yaml"

SMOKE_IDS = {
    "2021.09.29.15.23.04_veh-28_00601_00802-6326d00e52115da4",
    "2021.10.06.08.16.17_veh-52_01949_02501-49074bfb7c9e5c26",
    "2021.10.06.07.26.10_veh-52_00422_00728-a59bd481d324594a",
    "2021.09.29.19.02.14_veh-28_00964_01689-70d9d518ad0f5382",
    "2021.09.29.18.19.40_veh-28_00844_01218-17a015f4ef9b56d0",
    "2021.10.06.07.26.10_veh-52_00953_01126-67debdeae60b5fa4",
    "2021.09.29.19.02.14_veh-28_03198_03360-398326681cd7500a",
    "2021.10.06.07.26.10_veh-52_00953_01126-5041cdf76ecb5ee7",
    "2021.09.29.18.19.40_veh-28_00438_00833-76a0c83f0b6453a0",
    "2021.10.06.07.26.10_veh-52_01245_02064-613c9ac33f6951ca",
    "2021.09.29.19.02.14_veh-28_03198_03360-3b1e0182cb145b8d",
}

PREFER_C = ["0008e2e718e15240", "028a2f461cfe5f1c", "031a7e846efb505b"]
PREFER_E = ["0064ab0c89485eea", "02cb6299682e51d6", "04e45066320e5414"]


def load_tokens(path: Path) -> Set[str]:
    d = yaml.safe_load(path.read_text())
    return set(str(t) for t in d["tokens"])


def load_pkl(pkl_path: Path) -> Dict[str, Any]:
    with pkl_path.open("rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict):
        raise TypeError(type(data))
    return data


def index_from_data(data: Dict[str, Any]) -> Dict[str, str]:
    """Map short token (suffix after last '-') -> long scene id."""
    out: Dict[str, str] = {}
    for sid in data.keys():
        tok = str(sid).rsplit("-", 1)[-1]
        out[tok] = str(sid)
    return out


def veh_from_sid(sid: str) -> str:
    parts = sid.split("_")
    for p in parts:
        if p.startswith("veh-"):
            return p
    return "unk"


def diversify_pick(
    candidates: List[Tuple[str, str]],
    n: int,
    preferred: List[str],
) -> List[Tuple[str, str, str]]:
    """Return list of (token, long_id, reason). Prefer preferred tokens then veh diversity."""
    by_tok = {t: sid for t, sid in candidates}
    chosen: List[Tuple[str, str, str]] = []
    used_veh: Set[str] = set()
    used_tok: Set[str] = set()

    for t in preferred:
        if t in by_tok and len(chosen) < n:
            sid = by_tok[t]
            chosen.append((t, sid, "preferred_token"))
            used_veh.add(veh_from_sid(sid))
            used_tok.add(t)

    by_veh: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for t, sid in sorted(candidates, key=lambda x: x[0]):
        if t in used_tok:
            continue
        by_veh[veh_from_sid(sid)].append((t, sid))

    vehs = sorted(by_veh.keys())
    i = 0
    while len(chosen) < n and any(by_veh.values()):
        v = vehs[i % len(vehs)]
        # prefer unused veh
        order = [x for x in vehs if x not in used_veh] + [x for x in vehs if x in used_veh]
        v = order[i % len(order)] if order else v
        if by_veh[v]:
            t, sid = by_veh[v].pop(0)
            if t not in used_tok:
                chosen.append((t, sid, "veh_diverse_fill"))
                used_veh.add(v)
                used_tok.add(t)
        i += 1
        if i > 100000:
            break
    return chosen[:n]


def extract_single_scene_from_data(data: Dict[str, Any], scene_id: str, dst: Path) -> None:
    if scene_id not in data:
        raise KeyError(scene_id)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("wb") as f:
        pickle.dump({scene_id: data[scene_id]}, f, protocol=pickle.HIGHEST_PROTOCOL)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--collision-pkl",
        type=Path,
        default=Path(
            "/mnt/cpfs/prediction/lyyy/myself/WE/data/hf/data/sim_engine/scenarios/original/navtrain_50pct_collision/all_scenarios.pkl"
        ),
    )
    ap.add_argument(
        "--ep-pkl",
        type=Path,
        default=Path(
            "/mnt/cpfs/prediction/lyyy/myself/WE/data/hf/data/sim_engine/scenarios/original/navtrain_ep_per1/all_scenarios.pkl"
        ),
    )
    ap.add_argument(
        "--out-md",
        type=Path,
        default=WE_ROOT / "reports/train_side_le10_scene_list.md",
    )
    ap.add_argument(
        "--out-json",
        type=Path,
        default=WE_ROOT / "reports/train_side_le10_scene_list.json",
    )
    ap.add_argument(
        "--extract-root",
        type=Path,
        default=Path("/mnt/cpfs/prediction/lyyy/myself/WE/data/frozen_paired/train_side_le10"),
    )
    args = ap.parse_args()

    navtest = load_tokens(NAVTEST_YAML)
    coll = load_tokens(COLL_YAML)
    ep = load_tokens(EP_YAML)
    off = load_tokens(OFF_YAML)
    c_only = coll - ep - off
    e_only = ep - coll - off

    assert len(coll & navtest) == 0
    assert len(ep & navtest) == 0
    assert len(off & navtest) == 0

    print("Loading collision pkl once...", flush=True)
    c_data = load_pkl(args.collision_pkl)
    c_idx = index_from_data(c_data)
    print(f"  n_scenes={len(c_idx)}", flush=True)
    print("Loading ep pkl once...", flush=True)
    e_data = load_pkl(args.ep_pkl)
    e_idx = index_from_data(e_data)
    print(f"  n_scenes={len(e_idx)}", flush=True)

    c_cands = [(t, c_idx[t]) for t in sorted(c_only) if t in c_idx and c_idx[t] not in SMOKE_IDS]
    e_cands = [(t, e_idx[t]) for t in sorted(e_only) if t in e_idx and e_idx[t] not in SMOKE_IDS]
    print(f"c_cands_in_pkl={len(c_cands)} e_cands_in_pkl={len(e_cands)}", flush=True)

    c_pick = diversify_pick(c_cands, 5, PREFER_C)
    e_pick = diversify_pick(e_cands, 5, PREFER_E)
    if len(c_pick) < 5 or len(e_pick) < 5:
        raise RuntimeError(f"Insufficient scenes after filter: C={len(c_pick)} E={len(e_pick)}")

    scenes: List[Dict[str, Any]] = []
    for bucket, picks, pkl, data in [
        ("collision", c_pick, args.collision_pkl, c_data),
        ("ep", e_pick, args.ep_pkl, e_data),
    ]:
        for tok, sid, reason in picks:
            single = args.extract_root / tok / "scene" / "all_scenarios.pkl"
            extract_single_scene_from_data(data, sid, single)
            scenes.append(
                {
                    "bucket": bucket,
                    "token": tok,
                    "scene_id": sid,
                    "veh": veh_from_sid(sid),
                    "source_pkl": str(pkl),
                    "single_scene_pkl": str(single),
                    "selection_reason": reason,
                    "yaml_bucket_exclusive": True,
                }
            )
    del c_data, e_data

    final_tokens = {s["token"] for s in scenes}
    final_ids = {s["scene_id"] for s in scenes}
    inter_navtest = sorted(final_tokens & navtest)
    inter_smoke = sorted(final_ids & SMOKE_IDS)
    assert not inter_navtest, inter_navtest
    assert not inter_smoke, inter_smoke

    payload = {
        "n": len(scenes),
        "buckets": {"collision": 5, "ep": 5, "offroad": 0},
        "offroad_note": "Deferred: no original offroad pkl; not using BWM/augmented.",
        "isolation": {
            "navtest_failures_n": len(navtest),
            "intersection_tokens_with_navtest": inter_navtest,
            "intersection_scene_ids_with_smoke": inter_smoke,
            "union_rare_train_tokens": len(coll | ep | off),
            "rare_train_intersect_navtest": 0,
        },
        "scenes": scenes,
        "conditioning_rule": "static_feasible_median_path_length",
        "cutoff": 4,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# train-side ≤10 scene list",
        "",
        "**工程开发样本（train rare）；禁止与 navtest_failures / engineering smoke / cutoff4 冒烟并表。**",
        "",
        f"- collision-exclusive: 5；ep-exclusive: 5；offroad: 0（保留后续池；不用 BWM/augmented）",
        f"- ∩ navtest_failures tokens: `{inter_navtest}` （必须为空）",
        f"- ∩ smoke scene-ids: `{inter_smoke}` （必须为空）",
        f"- rare-train ∪ ∩ navtest = 0（yaml 级已核验）",
        "",
        "| # | 桶 | token | scene_id | veh | 入选理由 |",
        "|---|---|---|---|---|---|",
    ]
    for i, s in enumerate(scenes, 1):
        lines.append(
            f"| {i} | {s['bucket']} | `{s['token']}` | `{s['scene_id']}` | {s['veh']} | {s['selection_reason']} |"
        )
    lines.extend(
        [
            "",
            "## 资产",
            "",
            f"- collision pkl: `{args.collision_pkl}`",
            f"- ep pkl: `{args.ep_pkl}`",
            f"- 单 scene 抽取: `{args.extract_root}/<token>/scene/all_scenarios.pkl`",
            f"- 地图: 既有 `data/maps/extracted`；本批不下载 3DGS",
            "",
            f"JSON: `{args.out_json}`",
            "",
        ]
    )
    args.out_md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"n": len(scenes), "out_md": str(args.out_md), "inter_navtest": inter_navtest}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
