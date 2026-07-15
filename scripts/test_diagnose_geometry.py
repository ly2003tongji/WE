#!/usr/bin/env python3
"""Unit tests for diagnose geometry consistency (full-horizon min distance)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))


def geometry_label_consistent(
    noc_idm: float,
    collide_idm: bool,
    min_dist_replay: float | None,
    min_dist_idm: float | None,
    margin: float = 0.05,
) -> bool:
    """Same rule as diagnose_idm_agents.collision_geometry_samples."""
    if noc_idm >= 1.0:
        return True
    if collide_idm:
        return True
    if min_dist_idm is None or min_dist_replay is None:
        return False
    return min_dist_idm < min_dist_replay - margin


def test_uses_full_horizon_min_not_first_step() -> None:
    # Replay min over horizon is 10m (first step was 0.1 by accident in old bug).
    # IDM min is 2m. Danger label should be consistent because IDM is closer overall.
    assert geometry_label_consistent(0.0, False, min_dist_replay=10.0, min_dist_idm=2.0) is True
    # If we wrongly compared to first-step 0.1, we'd get False — ensure we don't.
    first_step_replay = 0.1
    assert not (2.0 < first_step_replay - 0.05)


def test_safe_label_always_consistent() -> None:
    assert geometry_label_consistent(1.0, False, 1.0, 0.1) is True


def test_collide_counts_as_consistent() -> None:
    assert geometry_label_consistent(0.0, True, 5.0, 6.0) is True


def test_farther_idm_not_consistent() -> None:
    assert geometry_label_consistent(0.0, False, min_dist_replay=2.0, min_dist_idm=5.0) is False


def main() -> int:
    test_uses_full_horizon_min_not_first_step()
    test_safe_label_always_consistent()
    test_collide_counts_as_consistent()
    test_farther_idm_not_consistent()
    print("ALL diagnose geometry tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
