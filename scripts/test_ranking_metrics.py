#!/usr/bin/env python3
"""Unit tests for ranking_metrics: no ties / partial ties / all ties; ratios in [0,1]."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from ranking_metrics import (  # noqa: E402
    analyze_max_score_ties,
    assert_ratios_in_unit_interval,
    directional_binary_flips,
    index_tie_broken_topk_overlap,
    max_score_set,
)


def test_no_ties() -> None:
    a = np.array([3.0, 2.0, 1.0, 0.0])
    b = np.array([3.0, 1.5, 1.0, 0.5])
    t = analyze_max_score_ties(a, b)
    assert t["side_a_max_set_size"] == 1
    assert t["side_b_max_set_size"] == 1
    assert t["intersection_size"] == 1
    assert t["jaccard"] == 1.0
    assert t["both_unique_optimum"] is True
    assert t["can_claim_unique_top1_flip"] is False
    assert 0 <= index_tie_broken_topk_overlap(a, b, 2) <= 1
    assert_ratios_in_unit_interval(t)


def test_partial_ties() -> None:
    # A: three tied at max=1; B: two tied at max=1, subset of A's
    a = np.array([1.0, 1.0, 1.0, 0.2, 0.1])
    b = np.array([1.0, 0.5, 1.0, 0.2, 0.0])
    t = analyze_max_score_ties(a, b)
    assert t["side_a_max_set_size"] == 3
    assert t["side_b_max_set_size"] == 2
    assert t["intersection_size"] == 2
    assert t["b_is_subset_of_a"] is True
    assert abs(t["jaccard"] - 2 / 3) < 1e-12
    assert abs(t["frac_b_max_in_a_max"] - 1.0) < 1e-12
    assert abs(t["frac_a_max_in_b_max"] - 2 / 3) < 1e-12
    assert t["can_claim_unique_top1_flip"] is False
    assert_ratios_in_unit_interval(t)
    ov = index_tie_broken_topk_overlap(a, b, 1)
    assert 0.0 <= ov <= 1.0


def test_all_ties() -> None:
    a = np.ones(8)
    b = np.ones(8)
    t = analyze_max_score_ties(a, b)
    assert t["side_a_max_set_size"] == 8
    assert t["side_b_max_set_size"] == 8
    assert t["jaccard"] == 1.0
    assert t["frac_a_max_in_b_max"] == 1.0
    assert t["frac_b_max_in_a_max"] == 1.0
    assert t["can_claim_unique_top1_flip"] is False
    assert_ratios_in_unit_interval(t)
    for k in (1, 4, 8):
        ov = index_tie_broken_topk_overlap(a, b, k)
        assert abs(ov - 1.0) < 1e-12


def test_smoke_like_subset() -> None:
    # Replay 3007 / IDM 321 / inter 321 pattern (scaled)
    a = np.zeros(30)
    a[:10] = 1.0
    b = np.zeros(30)
    b[:3] = 1.0
    t = analyze_max_score_ties(a, b)
    assert t["side_a_max_set_size"] == 10
    assert t["side_b_max_set_size"] == 3
    assert t["intersection_size"] == 3
    assert t["b_is_subset_of_a"] is True
    assert abs(t["jaccard"] - 3 / 10) < 1e-12
    assert_ratios_in_unit_interval(t)


def test_directional_flips() -> None:
    a = np.array([1, 1, 0, 0, 1], dtype=float)
    b = np.array([1, 0, 0, 1, 1], dtype=float)
    d = directional_binary_flips(a, b)
    assert d["both_safe"] == 2
    assert d["both_danger"] == 1
    assert d["a_safe_to_b_danger"] == 1
    assert d["a_danger_to_b_safe"] == 1
    assert d["total_flip"] == 2
    assert_ratios_in_unit_interval(d)


def test_max_score_set_helper() -> None:
    s = max_score_set(np.array([0.5, 1.0, 1.0, 0.2]))
    assert s == {1, 2}


def main() -> int:
    test_no_ties()
    test_partial_ties()
    test_all_ties()
    test_smoke_like_subset()
    test_directional_flips()
    test_max_score_set_helper()
    print("ALL ranking_metrics tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
