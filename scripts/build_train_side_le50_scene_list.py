#!/usr/bin/env python3
"""Build train-side ~50 scene list: 25C+25E incl. all le10 + reserve 8/bucket.

Hard isolation: ∩ navtest_failures = ∅; ∩ engineering smoke scene-ids = ∅.
Reuse gate-A le10 via symlink; degraded le10 retained without rerun.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import yaml

WE_ROOT = Path(__file__).resolve().parents[1]
YAML_DIR = WE_ROOT / "upstream/WorldEngine/projects/AlgEngine/configs/navsim_splits"
NAVTEST_YAML = YAML_DIR / "navtest_split/navtest_failures_filtered.yaml"
COLL_YAML = YAML_DIR / "navtrain_split/e2e_vadv2_50pct_rare/navtrain_50pct_collision.yaml"
EP_YAML = YAML_DIR / "navtrain_split/e2e_vadv2_50pct_rare/navtrain_50pct_ep_1pct.yaml"
OFF_YAML = YAML_DIR / "navtrain_split/e2e_vadv2_50pct_rare/navtrain_50pct_off_road.yaml"
LE10_JSON = WE_ROOT / "reports/train_side_le10_scene_list.json"
LE10_ROOT = Path("/mnt/cpfs/prediction/lyyy/myself/WE/data/frozen_paired/train_side_le10")

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

REUSE_NEED = [
    "scene_meta.json",
    "restore/scores_log_replay.pkl",
    "restore/scores_idm_restored.pkl",
    "nexus_seed0/future_nexus_seed0_meta.json",
    "three_source_seed0.json",
]


def load_tokens(path: Path) -> Set[str]:
    return set(str(t) for t in yaml.safe_load(path.read_text())["tokens"])


def load_pkl(path: Path) -> Dict[str, Any]:
    with path.open("rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict):
        raise TypeError(type(data))
    return data


def index_from_data(data: Dict[str, Any]) -> Dict[str, str]:
    return {str(sid).rsplit("-", 1)[-1]: str(sid) for sid in data.keys()}


def veh_from_sid(sid: str) -> str:
    for p in sid.split("_"):
        if p.startswith("veh-"):
            return p
    return "unk"


def log_prefix(sid: str) -> str:
    # keep date+veh slice for diversity: first two underscore-groups roughly
    return "-".join(sid.rsplit("-", 1)[0].split("_")[:3])


def can_reuse_le10(tok: str) -> bool:
    d = LE10_ROOT / tok
    sum0 = d / "three_source_seed0.json"
    if not sum0.exists():
        return False
    sm = json.loads(sum0.read_text())
    if sm.get("gate_a") is not True:
        return False
    return all((d / rel).exists() for rel in REUSE_NEED)


def diversify_pick(
    candidates: List[Tuple[str, str]],
    n: int,
    preferred: List[str],
    exclude: Set[str],
) -> List[Tuple[str, str, str]]:
    by_tok = {t: sid for t, sid in candidates if t not in exclude}
    chosen: List[Tuple[str, str, str]] = []
    used_veh: Set[str] = set()
    used_log: Set[str] = set()
    used_tok: Set[str] = set()

    for t in preferred:
        if t in by_tok and len(chosen) < n:
            sid = by_tok[t]
            chosen.append((t, sid, "preferred_token"))
            used_veh.add(veh_from_sid(sid))
            used_log.add(log_prefix(sid))
            used_tok.add(t)

    by_veh: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for t, sid in sorted(by_tok.items()):
        if t in used_tok:
            continue
        by_veh[veh_from_sid(sid)].append((t, sid))

    vehs = sorted(by_veh.keys())
    i = 0
    while len(chosen) < n and any(by_veh.values()):
        order = [x for x in vehs if x not in used_veh] + [x for x in vehs if x in used_veh]
        if not order:
            break
        v = order[i % len(order)]
        # among this veh, prefer unseen log prefix
        pool = by_veh[v]
        if not pool:
            i += 1
            if i > 200000:
                break
            continue
        pool.sort(key=lambda x: (log_prefix(x[1]) in used_log, x[0]))
        t, sid = pool.pop(0)
        if t not in used_tok:
            chosen.append((t, sid, "veh_log_diverse_fill"))
            used_veh.add(v)
            used_log.add(log_prefix(sid))
            used_tok.add(t)
        i += 1
        if i > 200000:
            break
    return chosen[:n]


def extract_single(data: Dict[str, Any], scene_id: str, dst: Path) -> None:
    if scene_id not in data:
        raise KeyError(scene_id)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("wb") as f:
        pickle.dump({scene_id: data[scene_id]}, f, protocol=pickle.HIGHEST_PROTOCOL)


def ensure_symlink_reuse(tok: str, le50_root: Path) -> Path:
    dst = le50_root / tok
    src = LE10_ROOT / tok
    if dst.is_symlink() or dst.exists():
        if dst.is_symlink():
            return dst
        # if real dir already, leave it
        return dst
    dst.symlink_to(os.path.relpath(src, start=dst.parent))
    return dst


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
        "--extract-root",
        type=Path,
        default=Path("/mnt/cpfs/prediction/lyyy/myself/WE/data/frozen_paired/train_side_le50"),
    )
    ap.add_argument("--out-md", type=Path, default=WE_ROOT / "reports/train_side_le50_scene_list.md")
    ap.add_argument("--out-json", type=Path, default=WE_ROOT / "reports/train_side_le50_scene_list.json")
    ap.add_argument("--n-per-bucket", type=int, default=25)
    ap.add_argument("--reserve-per-bucket", type=int, default=8)
    args = ap.parse_args()

    le10 = json.loads(LE10_JSON.read_text())
    le10_scenes = le10["scenes"]
    le10_by_bucket = {"collision": [], "ep": []}
    for s in le10_scenes:
        le10_by_bucket[s["bucket"]].append(s)

    navtest = load_tokens(NAVTEST_YAML)
    coll = load_tokens(COLL_YAML)
    ep = load_tokens(EP_YAML)
    off = load_tokens(OFF_YAML)
    c_only = coll - ep - off
    e_only = ep - coll - off
    assert len(coll & navtest) == 0 and len(ep & navtest) == 0

    print("Loading collision pkl once...", flush=True)
    c_data = load_pkl(args.collision_pkl)
    c_idx = index_from_data(c_data)
    print(f"  n={len(c_idx)}", flush=True)
    print("Loading ep pkl once...", flush=True)
    e_data = load_pkl(args.ep_pkl)
    e_idx = index_from_data(e_data)
    print(f"  n={len(e_idx)}", flush=True)

    c_cands = [(t, c_idx[t]) for t in sorted(c_only) if t in c_idx and c_idx[t] not in SMOKE_IDS]
    e_cands = [(t, e_idx[t]) for t in sorted(e_only) if t in e_idx and e_idx[t] not in SMOKE_IDS]
    print(f"c_cands={len(c_cands)} e_cands={len(e_cands)}", flush=True)

    args.extract_root.mkdir(parents=True, exist_ok=True)
    scenes: List[Dict[str, Any]] = []
    reserve: List[Dict[str, Any]] = []
    used: Set[str] = set()

    for bucket, le10_list, cands, pkl, data in [
        ("collision", le10_by_bucket["collision"], c_cands, args.collision_pkl, c_data),
        ("ep", le10_by_bucket["ep"], e_cands, args.ep_pkl, e_data),
    ]:
        # 1) lock all le10
        for s in le10_list:
            tok = s["token"]
            sid = s["scene_id"]
            used.add(tok)
            reuse = can_reuse_le10(tok)
            if reuse:
                ensure_symlink_reuse(tok, args.extract_root)
                single = str(args.extract_root / tok / "scene" / "all_scenarios.pkl")
            else:
                # degraded retained: symlink whole dir for provenance, or extract scene only
                ensure_symlink_reuse(tok, args.extract_root)
                single = str(args.extract_root / tok / "scene" / "all_scenarios.pkl")
            scenes.append(
                {
                    "bucket": bucket,
                    "token": tok,
                    "scene_id": sid,
                    "veh": veh_from_sid(sid),
                    "source_pkl": str(pkl),
                    "single_scene_pkl": single,
                    "selection_reason": "retained_le10",
                    "from_le10": True,
                    "reuse_from_le10": reuse,
                    "yaml_bucket_exclusive": True,
                    "planned": True,
                }
            )

        need_new = args.n_per_bucket - len(le10_list)
        picks = diversify_pick(cands, need_new + args.reserve_per_bucket, [], used)
        if len(picks) < need_new:
            raise RuntimeError(f"bucket {bucket}: only {len(picks)} new candidates, need {need_new}")
        main_picks = picks[:need_new]
        res_picks = picks[need_new : need_new + args.reserve_per_bucket]

        for tok, sid, reason in main_picks:
            used.add(tok)
            single = args.extract_root / tok / "scene" / "all_scenarios.pkl"
            extract_single(data, sid, single)
            scenes.append(
                {
                    "bucket": bucket,
                    "token": tok,
                    "scene_id": sid,
                    "veh": veh_from_sid(sid),
                    "source_pkl": str(pkl),
                    "single_scene_pkl": str(single),
                    "selection_reason": reason,
                    "from_le10": False,
                    "reuse_from_le10": False,
                    "yaml_bucket_exclusive": True,
                    "planned": True,
                }
            )

        for tok, sid, reason in res_picks:
            used.add(tok)
            single = args.extract_root / tok / "scene" / "all_scenarios.pkl"
            extract_single(data, sid, single)
            reserve.append(
                {
                    "bucket": bucket,
                    "token": tok,
                    "scene_id": sid,
                    "veh": veh_from_sid(sid),
                    "source_pkl": str(pkl),
                    "single_scene_pkl": str(single),
                    "selection_reason": f"reserve_{reason}",
                    "from_le10": False,
                    "reuse_from_le10": False,
                    "yaml_bucket_exclusive": True,
                    "planned": False,
                    "reserve": True,
                }
            )

    del c_data, e_data

    all_for_iso = scenes + reserve
    final_tokens = {s["token"] for s in all_for_iso}
    final_ids = {s["scene_id"] for s in all_for_iso}
    inter_navtest = sorted(final_tokens & navtest)
    inter_smoke = sorted(final_ids & SMOKE_IDS)
    assert not inter_navtest, inter_navtest
    assert not inter_smoke, inter_smoke

    n_reuse = sum(1 for s in scenes if s["reuse_from_le10"])
    n_le10_deg = sum(1 for s in scenes if s["from_le10"] and not s["reuse_from_le10"])
    n_new = sum(1 for s in scenes if not s["from_le10"])

    payload = {
        "n_planned": len(scenes),
        "n_reserve": len(reserve),
        "buckets": {"collision": args.n_per_bucket, "ep": args.n_per_bucket, "offroad": 0},
        "reuse_from_le10_n": n_reuse,
        "le10_degraded_retained_n": n_le10_deg,
        "new_planned_n": n_new,
        "attempt_budget": {
            "le10_already_attempted": 10,
            "max_total_restore_attempts": 65,
            "target_gate_a": [48, 50],
        },
        "offroad_note": "Deferred; not using BWM/augmented.",
        "isolation": {
            "navtest_failures_n": len(navtest),
            "intersection_tokens_with_navtest": inter_navtest,
            "intersection_scene_ids_with_smoke": inter_smoke,
            "rare_train_intersect_navtest": 0,
        },
        "scenes": scenes,
        "reserve": reserve,
        "conditioning_rule": "static_feasible_median_path_length",
        "cutoff": 4,
        "horizon": 9,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# train-side ~50 scene list",
        "",
        "**工程开发样本（train rare）；禁止与 navtest_failures / engineering smoke 并表。**",
        "",
        f"- planned: {len(scenes)}（collision {args.n_per_bucket} + ep {args.n_per_bucket}，含全部 le10）",
        f"- reserve: {len(reserve)}（每桶 {args.reserve_per_bucket}）",
        f"- reuse_from_le10 (symlink, skip rerun): **{n_reuse}**",
        f"- le10 degraded retained (no rerun): **{n_le10_deg}**",
        f"- new planned runs: **{n_new}**",
        f"- ∩ navtest tokens: `{inter_navtest}`",
        f"- ∩ smoke scene-ids: `{inter_smoke}`",
        f"- attempt budget: le10已计10；总 restore 尝试 ≤65；目标门控 A 48–50",
        "",
        "| # | 桶 | token | reuse | from_le10 | veh | 理由 |",
        "|---|---|---|---|---|---|---|",
    ]
    for i, s in enumerate(scenes, 1):
        lines.append(
            f"| {i} | {s['bucket']} | `{s['token']}` | {s['reuse_from_le10']} | {s['from_le10']} | {s['veh']} | {s['selection_reason']} |"
        )
    lines.extend(
        [
            "",
            "## Reserve（默认不跑；E2b 按需）",
            "",
            "| 桶 | token | veh | 理由 |",
            "|---|---|---|---|",
        ]
    )
    for s in reserve:
        lines.append(f"| {s['bucket']} | `{s['token']}` | {s['veh']} | {s['selection_reason']} |")
    lines.extend(
        [
            "",
            "## 资产",
            "",
            f"- collision pkl: `{args.collision_pkl}`",
            f"- ep pkl: `{args.ep_pkl}`",
            f"- 根目录: `{args.extract_root}/<token>/`（reuse → symlink `../train_side_le10/<token>`）",
            "",
            f"JSON: `{args.out_json}`",
            "",
        ]
    )
    args.out_md.write_text("\n".join(lines), encoding="utf-8")
    print(
        json.dumps(
            {
                "n_planned": len(scenes),
                "n_reserve": len(reserve),
                "n_reuse": n_reuse,
                "n_le10_deg": n_le10_deg,
                "n_new": n_new,
                "inter_navtest": inter_navtest,
                "out": str(args.out_md),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
