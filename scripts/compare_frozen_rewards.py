#!/usr/bin/env python3
"""Compare Replay vs IDM frozen-state 8192 PDM rewards (engineering validation only)."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from frozen_state_lib import save_json  # noqa: E402
from ranking_metrics import (  # noqa: E402
    analyze_max_score_ties,
    assert_ratios_in_unit_interval,
    directional_binary_flips,
    index_tie_broken_topk_overlap,
    kendall_tau_b,
)

PDM_KEYS = [
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "time_to_collision_within_bound",
    "comfort",
    "ego_progress",
    "driving_direction_compliance",
    "score",
]


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
            "/mnt/cpfs/prediction/lyyy/myself/WE/WE/reports/frozen_paired_compare_summary_v3.json"
        ),
    )
    parser.add_argument(
        "--git-summary-csv",
        type=Path,
        default=Path(
            "/mnt/cpfs/prediction/lyyy/myself/WE/WE/reports/frozen_paired_compare_summary_v3.csv"
        ),
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
        diff = b - a
        metrics[k] = {
            "equal": bool(np.array_equal(a, b)),
            "n": int(a.size),
            "max_abs_diff": float(np.max(np.abs(diff))),
            "mean_abs_diff": float(np.mean(np.abs(diff))),
            "n_changed_gt_1e-6": int(np.sum(np.abs(diff) > 1e-6)),
            "replay_mean": float(np.mean(a)),
            "idm_mean": float(np.mean(b)),
        }

    noc_dir = directional_binary_flips(
        r["no_at_fault_collisions"], i["no_at_fault_collisions"], args.noc_safe_threshold
    )
    noc_dir["labels"] = {
        "a": "replay",
        "b": "idm",
        "a_safe_to_b_danger": "replay_safe_to_idm_danger",
        "a_danger_to_b_safe": "replay_danger_to_idm_safe",
    }
    ttc_dir = directional_binary_flips(
        r["time_to_collision_within_bound"],
        i["time_to_collision_within_bound"],
        args.noc_safe_threshold,
    )
    ttc_dir["labels"] = {
        "a": "replay",
        "b": "idm",
        "a_safe_to_b_danger": "replay_safe_to_idm_danger",
        "a_danger_to_b_safe": "replay_danger_to_idm_safe",
    }

    score_r = np.asarray(r["score"], dtype=np.float64).reshape(-1)
    score_i = np.asarray(i["score"], dtype=np.float64).reshape(-1)
    print("computing kendall tau-b on 8192...", flush=True)
    tau = kendall_tau_b(score_r, score_i)
    ties = analyze_max_score_ties(score_r, score_i)
    # remap keys to replay/idm naming for report clarity
    ties_named = {
        "replay_max_score": ties["side_a_max_score"],
        "idm_max_score": ties["side_b_max_score"],
        "replay_max_set_size": ties["side_a_max_set_size"],
        "idm_max_set_size": ties["side_b_max_set_size"],
        "intersection_size": ties["intersection_size"],
        "union_size": ties["union_size"],
        "jaccard": ties["jaccard"],
        "frac_replay_max_in_idm_max": ties["frac_a_max_in_b_max"],
        "frac_idm_max_in_replay_max": ties["frac_b_max_in_a_max"],
        "idm_max_is_subset_of_replay_max": ties["b_is_subset_of_a"],
        "argmax_index_replay": ties["argmax_index_a"],
        "argmax_index_idm": ties["argmax_index_b"],
        "both_unique_optimum": ties["both_unique_optimum"],
        "can_claim_unique_top1_flip": ties["can_claim_unique_top1_flip"],
        "interpretation": ties["interpretation"],
        "index_tie_broken_note": ties["index_tie_broken_note"],
    }
    topk_index_tie_broken = {
        f"k={k}": {
            "overlap": index_tie_broken_topk_overlap(score_r, score_i, k),
            "note": "index-tie-broken implementation behavior only; not a stable ranking conclusion",
        }
        for k in (1, 5, 10, 50, 100)
    }

    summary = {
        "engineering_validation_only": True,
        "not_research_conclusion": True,
        "validity_revision": "v3",
        "input_state_fingerprint_doc": fp.get("input_state_fingerprint"),
        "independent_fingerprints": fps_all,
        "ego_conditioning_verification": {
            "requested_hash": ego_v.get("requested_ego_conditioning_hash"),
            "executed_hash": ego_v.get("executed_ego_conditioning_hash"),
            "pos_error_max_m": ego_v.get("pos_error_max_m"),
            "heading_error_max_deg": ego_v.get("heading_error_max_deg"),
        },
        "futures_differ": build.get("futures_differ"),
        "future_compare_formal": {
            "traj_dev_common_valid_max_m": future_cmp.get("traj_dev_common_valid_max_m"),
            "traj_dev_common_valid_mean_m": future_cmp.get("traj_dev_common_valid_mean_m"),
            "legacy_unmasked_max_m": future_cmp.get("legacy_unmasked", {}).get("traj_dev_max_m"),
            "artifact_diagnosis": future_cmp.get("artifact_diagnosis"),
        },
        "reward_keys": metrics,
        "noc_directional": {
            "replay_safe_to_idm_danger": noc_dir["a_safe_to_b_danger"],
            "replay_danger_to_idm_safe": noc_dir["a_danger_to_b_safe"],
            "both_safe": noc_dir["both_safe"],
            "both_danger": noc_dir["both_danger"],
            "total_flip": noc_dir["total_flip"],
            "flip_ratio": noc_dir["flip_ratio"],
        },
        "ttc_directional": {
            "replay_safe_to_idm_danger": ttc_dir["a_safe_to_b_danger"],
            "replay_danger_to_idm_safe": ttc_dir["a_danger_to_b_safe"],
            "both_safe": ttc_dir["both_safe"],
            "both_danger": ttc_dir["both_danger"],
            "total_flip": ttc_dir["total_flip"],
            "flip_ratio": ttc_dir["flip_ratio"],
        },
        "ranking_ties": ties_named,
        "topk_index_tie_broken_only": topk_index_tie_broken,
        "kendall_tau_b_score": tau,
        "ego_progress_note": (
            "Exported ego_progress is NOC/DAC-gated (pdm_scorer._aggregate_scores); "
            "changes with agent futures via multiplicative gating."
        ),
        "overall_score_max_abs_diff": metrics["score"]["max_abs_diff"],
        "rewards_identical": all(metrics[k]["equal"] for k in PDM_KEYS),
        "score_provenance_replay": score_sum["sources"]["log_replay"]["reward_summary"]["score"],
        "score_provenance_idm": score_sum["sources"]["idm"]["reward_summary"]["score"],
    }
    assert_ratios_in_unit_interval(summary)

    out_json = args.report_json or (fut_dir / "compare_summary_v3.json")
    save_json(out_json, summary)
    save_json(args.git_summary_json, summary)

    row = {
        "validity_revision": "v3",
        "noc_r_safe_to_i_danger": summary["noc_directional"]["replay_safe_to_idm_danger"],
        "noc_r_danger_to_i_safe": summary["noc_directional"]["replay_danger_to_idm_safe"],
        "noc_both_safe": summary["noc_directional"]["both_safe"],
        "noc_both_danger": summary["noc_directional"]["both_danger"],
        "ttc_r_safe_to_i_danger": summary["ttc_directional"]["replay_safe_to_idm_danger"],
        "ttc_r_danger_to_i_safe": summary["ttc_directional"]["replay_danger_to_idm_safe"],
        "replay_max_set": ties_named["replay_max_set_size"],
        "idm_max_set": ties_named["idm_max_set_size"],
        "max_set_intersection": ties_named["intersection_size"],
        "jaccard": ties_named["jaccard"],
        "idm_subset_of_replay": ties_named["idm_max_is_subset_of_replay_max"],
        "can_claim_unique_top1_flip": ties_named["can_claim_unique_top1_flip"],
        "kendall_tau_b": tau,
        "common_valid_max_ade": future_cmp.get("traj_dev_common_valid_max_m"),
    }
    args.git_summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.git_summary_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader()
        w.writerow(row)

    print(json.dumps({"noc_directional": summary["noc_directional"], "ties": ties_named}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
