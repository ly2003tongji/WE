#!/usr/bin/env python3
"""Unit tests for stage-1.9 restore-physics helpers (no WorldEngine env required)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from frozen_state_lib import (  # noqa: E402
    extract_log_agent_full_state_at,
    restore_agent_physics_via_public_api,
)


def _mini_scene(T: int = 6) -> dict:
    pos = np.zeros((T, 3), dtype=np.float64)
    pos[:, 0] = np.arange(T) * 2.0  # moves along +x at 4 m/s (dt=0.5)
    heading = np.linspace(0.0, 0.5, T)  # slowly turning, nonzero angular velocity
    vel = np.zeros((T, 2), dtype=np.float64)
    vel[:, 0] = 4.0
    valid = np.ones(T)
    ego = {
        "type": "VEHICLE",
        "state": {
            "position": pos.copy(),
            "heading": heading.copy(),
            "velocity": vel.copy(),
            "length": np.full((T, 1), 4.5),
            "width": np.full((T, 1), 2.0),
            "valid": valid.copy(),
        },
    }
    return {"id": "t", "object_track": {"ego": ego}}


def test_extract_full_state_position_heading_velocity() -> None:
    scene = _mini_scene()
    st = extract_log_agent_full_state_at(scene, step=2)
    ego = st["ego"]
    assert ego["position"] == [4.0, 0.0]
    assert abs(ego["heading"] - 0.2) < 1e-9
    assert ego["velocity"] == [4.0, 0.0]
    assert ego["valid"] is True
    assert abs(ego["length"] - 4.5) < 1e-9


def test_extract_full_state_angular_velocity_forward_diff() -> None:
    scene = _mini_scene()
    st = extract_log_agent_full_state_at(scene, step=2)
    ego = st["ego"]
    # heading is linspace(0,0.5,6) -> step size 0.1; dt=0.5 -> angvel=0.1/0.5=0.2
    assert abs(ego["angular_velocity"] - 0.2) < 1e-9


def test_extract_full_state_last_step_uses_backward_diff() -> None:
    scene = _mini_scene(T=6)
    st = extract_log_agent_full_state_at(scene, step=5)
    ego = st["ego"]
    assert abs(ego["angular_velocity"] - 0.2) < 1e-9


class _FakeLane:
    def __init__(self, name: str):
        self.name = name


class _FakeNav:
    def __init__(self):
        self.current_lane = _FakeLane("original_route")
        self.following_original_traj = True
        self.update_localization_calls = 0

    def update_localization(self):
        self.update_localization_calls += 1
        # simulate map-lane switch happening on first restore call
        if self.update_localization_calls == 1:
            self.following_original_traj = False


class _FakePolicy:
    pass


class _FakeAgent:
    """Duck-types BaseAgent's public restore surface only."""

    def __init__(self):
        self.policy = _FakePolicy()
        self.navigation = _FakeNav()
        self._pos = np.zeros(2)
        self._heading = 0.0
        self._vel = np.zeros(2)
        self._angvel = 0.0

    def set_position(self, position):
        assert len(position) in (2, 3)
        self._pos = np.asarray(position[:2], dtype=np.float64)

    def set_heading_theta(self, heading_theta, in_rad=True):
        self._heading = float(heading_theta)

    def set_velocity(self, direction, value=None, in_local_frame=False):
        direction = np.asarray(direction, dtype=np.float64)
        if value is not None:
            norm = np.linalg.norm(direction) + 1e-6
            direction = direction / norm * value
        self._vel = direction

    def set_angular_velocity(self, angular_velocity, in_rad=True):
        self._angvel = float(angular_velocity)


def test_restore_applies_exact_values_via_public_setters_only() -> None:
    ag = _FakeAgent()
    target = {"position": [3.0, -1.0], "heading": 0.42, "velocity": [1.5, -0.5], "angular_velocity": 0.05}
    diag = restore_agent_physics_via_public_api(ag, target)
    assert np.allclose(ag._pos, [3.0, -1.0])
    assert abs(ag._heading - 0.42) < 1e-9
    assert np.allclose(ag._vel, [1.5, -0.5])  # value=None -> exact vector, no renormalization
    assert abs(ag._angvel - 0.05) < 1e-9
    assert diag["policy_identity_preserved"] is True
    assert diag["navigation_identity_preserved"] is True


def test_restore_never_reassigns_policy_or_navigation_object() -> None:
    ag = _FakeAgent()
    policy_before = ag.policy
    nav_before = ag.navigation
    restore_agent_physics_via_public_api(ag, {"position": [0.0, 0.0], "heading": 0.0, "velocity": [0.0, 0.0]})
    assert ag.policy is policy_before
    assert ag.navigation is nav_before


def test_restore_calls_update_localization_exactly_once_and_reports_transition() -> None:
    ag = _FakeAgent()
    diag = restore_agent_physics_via_public_api(
        ag, {"position": [1.0, 1.0], "heading": 0.1, "velocity": [1.0, 0.0]}
    )
    assert ag.navigation.update_localization_calls == 1
    # This is the documented residual risk: update_localization only moves state
    # forward; a pre-existing following_original_traj flip is visible, not hidden.
    assert diag["following_original_traj_before"] is True
    assert diag["following_original_traj_after"] is False


def main() -> int:
    test_extract_full_state_position_heading_velocity()
    test_extract_full_state_angular_velocity_forward_diff()
    test_extract_full_state_last_step_uses_backward_diff()
    test_restore_applies_exact_values_via_public_setters_only()
    test_restore_never_reassigns_policy_or_navigation_object()
    test_restore_calls_update_localization_exactly_once_and_reports_transition()
    print("ALL restore-physics helper tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
