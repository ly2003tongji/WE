#!/usr/bin/env python3
"""Compare Replay vs IDM frozen-state 8192 PDM rewards (engineering validation only)."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

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
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    n = len(x)
    if n < 2:
        return float("nan")
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


def topk_overlap_index_tie_broken(a: np.ndarray, b: np.ndarray, k: int) -> float:
    """np.argsort tie-breaks by index — mark as index-tie-broken."""
    ia = set(np.argsort(-a)[:k].tolist())
    ib = set(np.argsort(-b)[:k].tolist())
    return float(len(ia & ib) / k)


def max_score_set(scores: np.ndarray, atol: float = 1e-12) -> Set[int]:
    m = float(np.max(scores))
    return set(np.where(np.abs(scores - m) <= atol)[0].tolist())


def tie_aware_topk_overlap(a: np.ndarray, b: np.ndarray, k: int) -> Dict[str, Any]:
    """Optimistic / pessimistic Top-K overlap under score ties at the boundary."""
    # Rank by score desc; ties share rank band
    order_a = np.argsort(-a)
    order_b = np.argsort(-b)

    def boundary_tie_set(scores: np.ndarray, order: np.ndarray, k: int) -> Dict[str, Any]:
        if k >= len(scores):
            return {
                "core": set(range(len(scores))),
                "boundary_ties": set(),
                "kth_score": float(scores[order[min(k, len(scores)) - 1]]),
            }
        kth_score = float(scores[order[k - 1]])
        # all with score > kth are core; all with score == kth are boundary ties
        core = set(np.where(scores > kth_score + 1e-12)[0].tolist())
        boundary = set(np.where(np.abs(scores - kth_score) <= 1e-12)[0].tolist())
        return {"core": core, "boundary_ties": boundary, "kth_score": kth_score}

    ba = boundary_tie_set(a, order_a, k)
    bb = boundary_tie_set(b, order_b, k)
    # optimistic: include all boundary ties in both sets
    opt_a = ba["core"] | ba["boundary_ties"]
    opt_b = bb["core"] | bb["boundary_ties"]
    # pessimistic: only core (strictly above boundary); if core size < k, overlap denom still k
    pes_a = ba["core"]
    pes_b = bb["core"]
    return {
        "k": k,
        "index_tie_broken_overlap": topk_overlap_index_tie_broken(a, b, k),
        "optimistic_overlap": float(len(opt_a & opt_b) / k),
        "pessimistic_overlap": float(len(pes_a & pes_b) / k),
        "side_a_boundary_tie_count": len(ba["boundary_ties"]),
        "side_b_boundary_tie_count": len(bb["boundary_ties"]),
        "side_a_kth_score": ba["kth_score"],
        "side_b_kth_score": bb["kth_score"],
    }


def analyze_ties(score_r: np.ndarray, score_i: np.ndarray) -> Dict[str, Any]:
    set_r = max_score_set(score_r)
    set_i = max_score_set(score_i)
    inter = set_r & set_i
    union = set_r | set_i
    top1_idx_r = int(np.argmax(score_r))
    top1_idx_i = int(np.argmax(score_i))
    unique_r = len(set_r) == 1
    unique_i = len(set_i) == 1
    out = {
        "replay_max_score": float(np.max(score_r)),
        "idm_max_score": float(np.max(score_i)),
        "replay_n_tied_at_max": len(set_r),
        "idm_n_tied_at_max": len(set_i),
        "replay_max_score_set_size": len(set_r),
        "idm_max_score_set_size": len(set_i),
        "max_score_set_intersection_size": len(inter),
        "max_score_set_jaccard": float(len(inter) / len(union)) if union else float("nan"),
        "argmax_index_replay": top1_idx_r,
        "argmax_index_idm": top1_idx_i,
        "argmax_indices_equal": top1_idx_r == top1_idx_i,
        "both_unique_optimum": unique_r and unique_i,
        "interpretation": (
            "双方唯一最优且选定index相同"
            if (unique_r and unique_i and top1_idx_r == top1_idx_i)
            else (
                "选定index相同，但最优集合需看并列"
                if top1_idx_r == top1_idx_i
                else "选定index不同（index-tie-broken argmax）"
            )
        ),
    }
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--futures-dir",
        type=Path,
        default=Path("/mnt/cpfs/prediction/lyyy/myself/WE/data/frozen_paired/smoke1_cutoff3_validity_v2"),
    )
    parser.add_argument("--report-json", type=Path, default=None)
    parser.add_argument(
        "--git-summary-json",
        type=Path,
        default=Path(
            "/mnt/cpfs/prediction/lyyy/myself/WE/WE/reports/frozen_paired_compare_summary_v2.json"
        ),
    )
    parser.add_argument(
        "--git-summary-csv",
        type=Path,
        default=Path(
            "/mnt/cpfs/prediction/lyyy/myself/WE/WE/reports/frozen_paired_compare_summary_v2.csv"
        ),
    )
    parser.add_argument(
        "--stage15-dir",
        type=Path,
        default=Path("/mnt/cpfs/prediction/lyyy/myself/WE/data/frozen_paired/smoke1_cutoff3"),
        help="Stage 1.5 dir for old-vs-new comparison (read-only).",
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
    fps_all = json.loads((fut_dir / "independent_fingerprints_all.json").read_text())
    ego_v = json.loads((fut_dir / "ego_conditioning_verification.json").read_text())

    metrics: Dict[str, Any] = {}
    for k in PDM_KEYS:
        a = np.asarray(r[k]).astype(np.float64).reshape(-1)
        b = np.asarray(i[k]).astype(np.float64).reshape(-1)
        if a.shape != b.shape:
            raise ValueError(f"shape mismatch for {k}: {a.shape} vs {b.shape}")
        diff = b - a
        flips = (
            int(np.sum((a >= args.noc_safe_threshold) != (b >= args.noc_safe_threshold)))
            if k
            in (
                "no_at_fault_collisions",
                "drivable_area_compliance",
                "time_to_collision_within_bound",
                "comfort",
                "driving_direction_compliance",
            )
            else None
        )
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

    safe_r = noc_r >= 1.0
    safe_i = noc_i >= 1.0
    consistent_safe = int(np.sum(safe_r & safe_i))
    consistent_danger = int(np.sum(~safe_r & ~safe_i))
    conflict = int(np.sum(safe_r != safe_i))

    print("computing kendall tau-b on 8192...", flush=True)
    tau = kendall_tau_b(score_r, score_i)
    ties = analyze_ties(score_r, score_i)
    topk = {
        f"k={k}": {
            **tie_aware_topk_overlap(score_r, score_i, k),
            "index_tie_broken_note": "argsort/argmax breaks ties by smaller index",
        }
        for k in (1, 5, 10, 50, 100)
    }

    ego_progress_note = {
        "code_path": (
            "pdm_scorer.py:_aggregate_scores "
            "(~L174-L195): normalized_progress *= multiplicate_metric_scores; "
            "dense_reward_manager.py:_score_proposals_impl L662 exports "
            "scorer._weighted_metrics[PROGRESS] after this gating"
        ),
        "mechanism": (
            "ego_progress exported by DenseReward is NOC/DAC-gated normalized progress, "
            "not raw centerline progress. Changing agent futures flips NOC → zeros progress "
            "for colliding proposals even if ego trajectory identical."
        ),
        "n_changed": metrics["ego_progress"]["n_changed_gt_1e-6"],
        "expected_if_only_multiplicative_gate": True,
    }

    # old vs new
    old_vs_new = {"stage15_dir": str(args.stage15_dir), "available": False}
    if (args.stage15_dir / "compare_summary.json").exists():
        old = json.loads((args.stage15_dir / "compare_summary.json").read_text())
        old_vs_new = {
            "available": True,
            "stage15_noc_flip": old.get("noc_flip_count"),
            "v2_noc_flip": metrics["no_at_fault_collisions"]["n_changed_gt_1e-6"],
            "stage15_traj_dev_max_m_unmasked": old.get("future_compare", {}).get("traj_dev_max_m"),
            "v2_traj_dev_common_valid_max_m": future_cmp.get("traj_dev_common_valid_max_m"),
            "v2_legacy_unmasked_max_m": future_cmp.get("legacy_unmasked", {}).get("traj_dev_max_m"),
            "stage15_top1_changed": old.get("top1_changed"),
            "v2_tie_interpretation": ties["interpretation"],
        }

    summary = {
        "engineering_validation_only": True,
        "not_research_conclusion": True,
        "validity_revision": "v2",
        "input_state_fingerprint_doc": fp.get("input_state_fingerprint"),
        "independent_fingerprints": fps_all,
        "ego_conditioning_verification": {
            "requested_hash": ego_v.get("requested_ego_conditioning_hash"),
            "executed_hash": ego_v.get("executed_ego_conditioning_hash"),
            "pos_error_max_m": ego_v.get("pos_error_max_m"),
            "heading_error_max_deg": ego_v.get("heading_error_max_deg"),
            "warnings": ego_v.get("warnings"),
            "hard_fail": ego_v.get("hard_fail"),
        },
        "futures_differ": build.get("futures_differ"),
        "future_compare_formal": {
            "n_changed_agents_common_valid_maxade_gt_5cm": future_cmp.get(
                "n_changed_agents_common_valid_maxade_gt_5cm"
            ),
            "traj_dev_common_valid_mean_m": future_cmp.get("traj_dev_common_valid_mean_m"),
            "traj_dev_common_valid_max_m": future_cmp.get("traj_dev_common_valid_max_m"),
            "artifact_diagnosis": future_cmp.get("artifact_diagnosis"),
            "legacy_unmasked_max_m": future_cmp.get("legacy_unmasked", {}).get("traj_dev_max_m"),
            "coverage_idm_rolled": len(future_cmp.get("coverage_idm", {}).get("idm_rolled", [])),
        },
        "reward_keys": metrics,
        "noc_consistent_safe": consistent_safe,
        "noc_consistent_danger": consistent_danger,
        "noc_conflict": conflict,
        "noc_flip_count": metrics["no_at_fault_collisions"]["n_changed_gt_1e-6"],
        "noc_flip_ratio": metrics["no_at_fault_collisions"]["n_changed_gt_1e-6"] / 8192.0,
        "ranking_ties": ties,
        "topk_overlap": topk,
        "kendall_tau_b_score": tau,
        "ego_progress_analysis": ego_progress_note,
        "overall_score_max_abs_diff": metrics["score"]["max_abs_diff"],
        "overall_score_mean_abs_diff": metrics["score"]["mean_abs_diff"],
        "rewards_identical": all(metrics[k]["equal"] for k in PDM_KEYS),
        "old_vs_new": old_vs_new,
        "score_provenance_replay": score_sum["sources"]["log_replay"]["reward_summary"]["score"],
        "score_provenance_idm": score_sum["sources"]["idm"]["reward_summary"]["score"],
    }

    out_json = args.report_json or (fut_dir / "compare_summary.json")
    save_json(out_json, summary)
    save_json(args.git_summary_json, summary)

    row = {
        "validity_revision": "v2",
        "build_pair_fp_equal": fps_all.get("build_pair_equal"),
        "score_pair_fp_equal": fps_all.get("score_pair_equal"),
        "ego_pos_err_max_m": ego_v.get("pos_error_max_m"),
        "common_valid_max_ade_m": future_cmp.get("traj_dev_common_valid_max_m"),
        "legacy_unmasked_max_ade_m": future_cmp.get("legacy_unmasked", {}).get("traj_dev_max_m"),
        "noc_flip_count": summary["noc_flip_count"],
        "ttc_flip_count": metrics["time_to_collision_within_bound"]["n_changed_gt_1e-6"],
        "replay_n_tied_at_max": ties["replay_n_tied_at_max"],
        "idm_n_tied_at_max": ties["idm_n_tied_at_max"],
        "max_score_jaccard": ties["max_score_set_jaccard"],
        "both_unique_optimum": ties["both_unique_optimum"],
        "kendall_tau_b_score": tau,
        "topk10_optimistic": topk["k=10"]["optimistic_overlap"],
        "topk10_pessimistic": topk["k=10"]["pessimistic_overlap"],
    }
    args.git_summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.git_summary_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader()
        w.writerow(row)

    print(
        json.dumps(
            {
                "noc_flip_count": summary["noc_flip_count"],
                "common_valid_max_ade": future_cmp.get("traj_dev_common_valid_max_m"),
                "legacy_unmasked_max_ade": future_cmp.get("legacy_unmasked", {}).get("traj_dev_max_m"),
                "ties": ties,
                "fps": fps_all,
                "ego_err_max_m": ego_v.get("pos_error_max_m"),
            },
            indent=2,
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
