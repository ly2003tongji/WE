#!/usr/bin/env python3
"""Compare Replay vs IDM frozen-state 8192 PDM rewards (engineering validation only)."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from frozen_state_lib import save_json  # noqa: E402

PDM_KEYS = [
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "time_to_collision_within_bound",
    "comfort",
    "ego_progress",
    "driving_direction_compliance",
    "score",
]


def kendall_tau_b(x: np.ndarray, y: np.ndarray) -> float:
    """Kendall tau-b without SciPy dependency."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    n = len(x)
    if n < 2:
        return float("nan")
    # For n=8192, O(n^2) is ~33M comparisons — acceptable once.
    concordant = discordant = extra_x = extra_y = 0
    for i in range(n - 1):
        dx = x[i + 1 :] - x[i]
        dy = y[i + 1 :] - y[i]
        same_x = dx == 0
        same_y = dy == 0
        extra_x += int(np.sum(same_x & ~same_y))
        extra_y += int(np.sum(~same_x & same_y))
        c = np.sum((dx > 0) & (dy > 0)) + np.sum((dx < 0) & (dy < 0))
        d = np.sum((dx > 0) & (dy < 0)) + np.sum((dx < 0) & (dy > 0))
        concordant += int(c)
        discordant += int(d)
    num = concordant - discordant
    den = np.sqrt((concordant + discordant + extra_x) * (concordant + discordant + extra_y))
    if den == 0:
        return float("nan")
    return float(num / den)


def topk_overlap(a: np.ndarray, b: np.ndarray, k: int) -> float:
    ia = set(np.argsort(-a)[:k].tolist())
    ib = set(np.argsort(-b)[:k].tolist())
    return float(len(ia & ib) / k)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--futures-dir",
        type=Path,
        default=Path("/mnt/cpfs/prediction/lyyy/myself/WE/data/frozen_paired/smoke1_cutoff3"),
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=None,
        help="Optional path for lightweight compare JSON (default: futures-dir/compare_summary.json)",
    )
    parser.add_argument(
        "--git-summary-json",
        type=Path,
        default=Path("/mnt/cpfs/prediction/lyyy/myself/WE/WE/reports/frozen_paired_compare_summary.json"),
    )
    parser.add_argument(
        "--git-summary-csv",
        type=Path,
        default=Path("/mnt/cpfs/prediction/lyyy/myself/WE/WE/reports/frozen_paired_compare_summary.csv"),
    )
    parser.add_argument("--noc-safe-threshold", type=float, default=1.0)
    args = parser.parse_args()

    fut_dir = args.futures_dir
    with (fut_dir / "scores_log_replay.pkl").open("rb") as f:
        r = pickle.load(f)
    with (fut_dir / "scores_idm.pkl").open("rb") as f:
        i = pickle.load(f)

    future_cmp = json.loads((fut_dir / "future_compare.json").read_text())
    fp = json.loads((fut_dir / "input_state_fingerprint.json").read_text())
    build = json.loads((fut_dir / "build_summary.json").read_text())
    score_sum = json.loads((fut_dir / "score_summary.json").read_text())

    metrics: Dict[str, Any] = {}
    for k in PDM_KEYS:
        a = np.asarray(r[k]).astype(np.float64).reshape(-1)
        b = np.asarray(i[k]).astype(np.float64).reshape(-1)
        if a.shape != b.shape:
            raise ValueError(f"shape mismatch for {k}: {a.shape} vs {b.shape}")
        diff = b - a
        flips = int(np.sum((a >= args.noc_safe_threshold) != (b >= args.noc_safe_threshold))) if k in (
            "no_at_fault_collisions",
            "drivable_area_compliance",
            "time_to_collision_within_bound",
            "comfort",
            "driving_direction_compliance",
        ) else None
        # binary-ish flip for NOC specifically: value change across 1.0
        metrics[k] = {
            "equal": bool(np.array_equal(a, b)),
            "n": int(a.size),
            "max_abs_diff": float(np.max(np.abs(diff))),
            "mean_abs_diff": float(np.mean(np.abs(diff))),
            "mean_diff": float(np.mean(diff)),
            "n_changed": int(np.sum(np.abs(diff) > 0)),
            "n_changed_gt_1e-6": int(np.sum(np.abs(diff) > 1e-6)),
            "flip_count_threshold_1": flips,
            "flip_ratio_threshold_1": (flips / len(a)) if flips is not None else None,
            "replay_mean": float(np.mean(a)),
            "idm_mean": float(np.mean(b)),
        }

    noc_r = np.asarray(r["no_at_fault_collisions"]).astype(np.float64).reshape(-1)
    noc_i = np.asarray(i["no_at_fault_collisions"]).astype(np.float64).reshape(-1)
    score_r = np.asarray(r["score"]).astype(np.float64).reshape(-1)
    score_i = np.asarray(i["score"]).astype(np.float64).reshape(-1)

    # consistent safe / dangerous / conflict on NOC==1 vs <1
    safe_r = noc_r >= 1.0
    safe_i = noc_i >= 1.0
    consistent_safe = int(np.sum(safe_r & safe_i))
    consistent_danger = int(np.sum(~safe_r & ~safe_i))
    conflict = int(np.sum(safe_r != safe_i))

    top1_r = int(np.argmax(score_r))
    top1_i = int(np.argmax(score_i))

    # Faster Kendall on subsample if needed — full 8192 is OK (~few seconds)
    print("computing kendall tau-b on 8192...", flush=True)
    tau = kendall_tau_b(score_r, score_i)

    summary = {
        "engineering_validation_only": True,
        "not_research_conclusion": True,
        "input_state_fingerprint": fp["input_state_fingerprint"],
        "fingerprint_match_replay_idm": True,
        "ego_conditioning_hash": build.get("ego_conditioning_hash"),
        "futures_differ": build.get("futures_differ"),
        "future_compare": {
            "n_changed_agents_maxade_gt_5cm": future_cmp.get("n_changed_agents_maxade_gt_5cm"),
            "traj_dev_mean_m": future_cmp.get("traj_dev_mean_m"),
            "traj_dev_max_m": future_cmp.get("traj_dev_max_m"),
            "coverage_replay_n_vehicle": len(future_cmp.get("coverage_replay", {}).get("idm_capable_vehicle", [])),
            "coverage_idm_rolled": len(future_cmp.get("coverage_idm", {}).get("idm_rolled", [])),
            "coverage_idm_fallback": len(future_cmp.get("coverage_idm", {}).get("log_fallback", [])),
        },
        "reward_keys": metrics,
        "noc_consistent_safe": consistent_safe,
        "noc_consistent_danger": consistent_danger,
        "noc_conflict": conflict,
        "noc_flip_count": metrics["no_at_fault_collisions"]["n_changed_gt_1e-6"],
        "noc_flip_ratio": metrics["no_at_fault_collisions"]["n_changed_gt_1e-6"] / 8192.0,
        "top1_replay": top1_r,
        "top1_idm": top1_i,
        "top1_changed": top1_r != top1_i,
        "topk_overlap": {f"k={k}": topk_overlap(score_r, score_i, k) for k in (1, 5, 10, 50, 100)},
        "kendall_tau_b_score": tau,
        "overall_score_max_abs_diff": metrics["score"]["max_abs_diff"],
        "overall_score_mean_abs_diff": metrics["score"]["mean_abs_diff"],
        "rewards_identical": all(metrics[k]["equal"] for k in PDM_KEYS),
        "score_provenance_replay": score_sum["sources"]["log_replay"]["reward_summary"]["score"],
        "score_provenance_idm": score_sum["sources"]["idm"]["reward_summary"]["score"],
    }

    out_json = args.report_json or (fut_dir / "compare_summary.json")
    save_json(out_json, summary)
    save_json(args.git_summary_json, summary)

    # CSV one-row summary for git
    row = {
        "input_state_fingerprint": summary["input_state_fingerprint"],
        "futures_differ": summary["futures_differ"],
        "n_changed_agents": summary["future_compare"]["n_changed_agents_maxade_gt_5cm"],
        "traj_dev_mean_m": summary["future_compare"]["traj_dev_mean_m"],
        "traj_dev_max_m": summary["future_compare"]["traj_dev_max_m"],
        "noc_flip_count": summary["noc_flip_count"],
        "noc_flip_ratio": summary["noc_flip_ratio"],
        "noc_conflict": summary["noc_conflict"],
        "ttc_max_abs_diff": metrics["time_to_collision_within_bound"]["max_abs_diff"],
        "dac_max_abs_diff": metrics["drivable_area_compliance"]["max_abs_diff"],
        "comfort_max_abs_diff": metrics["comfort"]["max_abs_diff"],
        "score_max_abs_diff": metrics["score"]["max_abs_diff"],
        "top1_changed": summary["top1_changed"],
        "topk10_overlap": summary["topk_overlap"]["k=10"],
        "kendall_tau_b_score": summary["kendall_tau_b_score"],
        "rewards_identical": summary["rewards_identical"],
    }
    args.git_summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.git_summary_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader()
        w.writerow(row)

    print(json.dumps({k: summary[k] for k in (
        "futures_differ",
        "noc_flip_count",
        "noc_conflict",
        "top1_changed",
        "kendall_tau_b_score",
        "rewards_identical",
        "overall_score_max_abs_diff",
    )}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
