#!/usr/bin/env python3
"""Ranking / tie / directional-flip metrics for frozen-state reward comparison.

All overlap / ratio / Jaccard values are guaranteed in [0, 1] (or NaN if undefined).
"""

from __future__ import annotations

from typing import Any, Dict, Set

import numpy as np


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
        c = int(np.sum((dx > 0) & (dy > 0)) + np.sum((dx < 0) & (dy < 0)))
        d = int(np.sum((dx > 0) & (dy < 0)) + np.sum((dx < 0) & (dy > 0)))
        concordant += c
        discordant += d
    num = concordant - discordant
    den = np.sqrt((concordant + discordant + extra_x) * (concordant + discordant + extra_y))
    if den == 0:
        return float("nan")
    return float(num / den)


def max_score_set(scores: np.ndarray, atol: float = 1e-12) -> Set[int]:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    m = float(np.max(scores))
    return set(np.where(np.abs(scores - m) <= atol)[0].tolist())


def _ratio(num: int, den: int) -> float:
    if den <= 0:
        return float("nan")
    r = float(num) / float(den)
    if not (0.0 <= r <= 1.0):
        raise ValueError(f"ratio out of [0,1]: {r} = {num}/{den}")
    return r


def analyze_max_score_ties(score_a: np.ndarray, score_b: np.ndarray, atol: float = 1e-12) -> Dict[str, Any]:
    """Formal tie-aware max-score set statistics. All ratios/Jaccard in [0,1]."""
    set_a = max_score_set(score_a, atol=atol)
    set_b = max_score_set(score_b, atol=atol)
    inter = set_a & set_b
    union = set_a | set_b
    top1_a = int(np.argmax(score_a))
    top1_b = int(np.argmax(score_b))
    unique_a = len(set_a) == 1
    unique_b = len(set_b) == 1
    jaccard = _ratio(len(inter), len(union)) if union else float("nan")
    return {
        "side_a_max_score": float(np.max(score_a)),
        "side_b_max_score": float(np.max(score_b)),
        "side_a_max_set_size": len(set_a),
        "side_b_max_set_size": len(set_b),
        "intersection_size": len(inter),
        "union_size": len(union),
        "jaccard": jaccard,
        "frac_a_max_in_b_max": _ratio(len(inter), len(set_a)),
        "frac_b_max_in_a_max": _ratio(len(inter), len(set_b)),
        "b_is_subset_of_a": set_b.issubset(set_a),
        "a_is_subset_of_b": set_a.issubset(set_b),
        "argmax_index_a": top1_a,
        "argmax_index_b": top1_b,
        "argmax_indices_equal": top1_a == top1_b,
        "both_unique_optimum": unique_a and unique_b,
        "can_claim_unique_top1_flip": bool(unique_a and unique_b and top1_a != top1_b),
        "interpretation": (
            "双方唯一最优且选定index相同"
            if (unique_a and unique_b and top1_a == top1_b)
            else (
                "双方唯一最优且选定index不同（可称唯一Top-1翻转）"
                if (unique_a and unique_b and top1_a != top1_b)
                else (
                    "存在大规模并列：不能声称唯一最优动作翻转；"
                    f"A max-set={len(set_a)}, B max-set={len(set_b)}, "
                    f"交集={len(inter)}, Jaccard={jaccard:.6f}"
                )
            )
        ),
        "index_tie_broken_note": (
            "np.argmax/argsort break ties by smaller index; "
            "index-tie-broken Top-K is implementation behavior, not a stable ranking conclusion"
        ),
    }


def index_tie_broken_topk_overlap(a: np.ndarray, b: np.ndarray, k: int) -> float:
    """Overlap of index-tie-broken Top-K sets. Always in [0,1]."""
    if k <= 0:
        raise ValueError("k must be positive")
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    k = min(k, len(a), len(b))
    ia = set(np.argsort(-a, kind="mergesort")[:k].tolist())
    ib = set(np.argsort(-b, kind="mergesort")[:k].tolist())
    return _ratio(len(ia & ib), k)


def directional_binary_flips(
    a: np.ndarray,
    b: np.ndarray,
    safe_threshold: float = 1.0,
) -> Dict[str, Any]:
    """Directional safe/danger flips. safe := value >= threshold."""
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    safe_a = a >= safe_threshold
    safe_b = b >= safe_threshold
    n = len(a)
    both_safe = int(np.sum(safe_a & safe_b))
    both_danger = int(np.sum(~safe_a & ~safe_b))
    a_safe_b_danger = int(np.sum(safe_a & ~safe_b))
    a_danger_b_safe = int(np.sum(~safe_a & safe_b))
    total_flip = a_safe_b_danger + a_danger_b_safe
    return {
        "n": n,
        "both_safe": both_safe,
        "both_danger": both_danger,
        "a_safe_to_b_danger": a_safe_b_danger,
        "a_danger_to_b_safe": a_danger_b_safe,
        "total_flip": total_flip,
        "flip_ratio": _ratio(total_flip, n),
        "a_safe_to_b_danger_ratio": _ratio(a_safe_b_danger, n),
        "a_danger_to_b_safe_ratio": _ratio(a_danger_b_safe, n),
        "both_safe_ratio": _ratio(both_safe, n),
        "both_danger_ratio": _ratio(both_danger, n),
        "labels": {
            "a": "replay",
            "b": "other",
            "safe_threshold": safe_threshold,
        },
    }


def assert_ratios_in_unit_interval(obj: Any, path: str = "root") -> None:
    """Recursively assert any key containing overlap/ratio/jaccard is in [0,1] or NaN."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            lk = str(k).lower()
            if any(s in lk for s in ("overlap", "ratio", "jaccard", "frac_")):
                if isinstance(v, (int, float, np.floating)):
                    fv = float(v)
                    if not (np.isnan(fv) or (0.0 <= fv <= 1.0)):
                        raise AssertionError(f"{path}.{k}={fv} not in [0,1]")
            assert_ratios_in_unit_interval(v, f"{path}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            assert_ratios_in_unit_interval(v, f"{path}[{i}]")
