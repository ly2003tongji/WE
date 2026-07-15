#!/usr/bin/env python3
"""Unit tests for warm-start cutoff gate and ego inject-at-cutoff."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from frozen_state_lib import (  # noqa: E402
    CUTOFF_GATE_TOLERANCES,
    classify_cutoff_gate,
    inject_ego_conditioning,
    inject_ego_conditioning_at_cutoff,
    slice_scene_from_cutoff,
    truncate_scene_prefix,
    vocab_plan_to_center_traj,
)


def _mini_scene(T: int = 12) -> dict:
    pos = np.zeros((T, 3), dtype=np.float64)
    pos[:, 0] = np.arange(T) * 1.0
    heading = np.zeros(T, dtype=np.float64)
    vel = np.zeros((T, 2), dtype=np.float64)
    vel[:, 0] = 2.0
    length = np.full((T, 1), 4.5)
    width = np.full((T, 1), 2.0)
    valid = np.ones(T)
    ego = {
        "type": "VEHICLE",
        "state": {
            "position": pos.copy(),
            "heading": heading.copy(),
            "velocity": vel.copy(),
            "length": length.copy(),
            "width": width.copy(),
            "valid": valid.copy(),
        },
    }
    ag = copy.deepcopy(ego)
    ag["state"]["position"][:, 1] = 3.0
    return {
        "id": "test",
        "map": "us-ma-boston",
        "log_length": T,
        "object_track": {"ego": ego, "aa": ag},
        "dynamic_map_states": {
            "1": {"state": {"traffic_light_state": np.array(["TRAFFIC_LIGHT_RED"] * T)}}
        },
        "metadata": {},
    }


def test_gate_a_identical() -> None:
    cold = {
        "step": 3,
        "map": "m",
        "agent_tokens_sorted": ["ego", "a"],
        "traffic_lights": {"1": "RED"},
        "agents": {
            "ego": {"pos_xy": [0.0, 0.0], "heading": 0.0, "speed": 1.0, "valid": True, "length": 4.0, "width": 2.0},
            "a": {"pos_xy": [1.0, 0.0], "heading": 0.1, "speed": 2.0, "valid": True, "length": 4.5, "width": 2.0},
        },
    }
    warm = copy.deepcopy(cold)
    g = classify_cutoff_gate(cold, warm)
    assert g["grade"] == "A"
    assert g["allow_future_reward_compare"] is True


def test_gate_b_soft_only() -> None:
    cold = {
        "step": 3,
        "map": "m",
        "agent_tokens_sorted": ["ego"],
        "traffic_lights": {},
        "agents": {
            "ego": {"pos_xy": [0.0, 0.0], "heading": 0.0, "speed": 1.0, "valid": True, "length": 4.0, "width": 2.0},
        },
    }
    warm = copy.deepcopy(cold)
    # 0.07m within soft 0.10, beyond strict 0.05
    warm["agents"]["ego"]["pos_xy"] = [0.07, 0.0]
    g = classify_cutoff_gate(cold, warm)
    assert g["grade"] == "B"
    assert CUTOFF_GATE_TOLERANCES["strict_pos_m"] == 0.05
    assert CUTOFF_GATE_TOLERANCES["soft_pos_m"] == 0.10


def test_gate_c_beyond_soft() -> None:
    cold = {
        "step": 3,
        "map": "m",
        "agent_tokens_sorted": ["ego"],
        "traffic_lights": {},
        "agents": {
            "ego": {"pos_xy": [0.0, 0.0], "heading": 0.0, "speed": 1.0, "valid": True, "length": 4.0, "width": 2.0},
        },
    }
    warm = copy.deepcopy(cold)
    warm["agents"]["ego"]["pos_xy"] = [0.5, 0.0]
    g = classify_cutoff_gate(cold, warm)
    assert g["grade"] == "C"
    assert g["allow_future_reward_compare"] is False


def test_inject_at_cutoff_matches_cold_slice() -> None:
    scene = _mini_scene()
    cutoff, horizon, plan_idx = 3, 9, 0
    vocab = np.zeros((2, 40, 3), dtype=np.float64)
    # small forward motion in local frame
    vocab[0, :, 0] = np.linspace(0.1, 4.0, 40)
    cold = slice_scene_from_cutoff(scene, cutoff, horizon)
    cold_meta = inject_ego_conditioning(cold, plan_idx, vocab)
    warm = truncate_scene_prefix(scene, cutoff + horizon)
    warm_meta = inject_ego_conditioning_at_cutoff(warm, cutoff, plan_idx, vocab)
    assert cold_meta["requested_ego_conditioning_hash"] == warm_meta["requested_ego_conditioning_hash"]
    # pre-cutoff ego unchanged from log
    assert np.allclose(
        warm["object_track"]["ego"]["state"]["position"][:cutoff],
        scene["object_track"]["ego"]["state"]["position"][:cutoff],
    )
    # post-cutoff matches cold truncated ego
    assert np.allclose(
        warm["object_track"]["ego"]["state"]["position"][cutoff : cutoff + horizon, :2],
        cold["object_track"]["ego"]["state"]["position"][:horizon, :2],
    )


def main() -> int:
    test_gate_a_identical()
    test_gate_b_soft_only()
    test_gate_c_beyond_soft()
    test_inject_at_cutoff_matches_cold_slice()
    print("ALL warm-start helper tests passed")
    print("DECLARED_TOLERANCES", CUTOFF_GATE_TOLERANCES)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
