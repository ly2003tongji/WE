#!/usr/bin/env python3
"""CPU unit tests for Nexus adapter.

Portable synthetic fixtures run always. Tests that require local absolute data
paths are marked integration and skip when data is absent.
"""

from __future__ import annotations

import math
import pickle
import sys
import unittest
from pathlib import Path

import numpy as np

WE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WE_ROOT))
sys.path.insert(0, str(WE_ROOT / "scripts"))

from adapters.traffic_models import nexus as nx  # noqa: E402


SCENE_PKL = Path(
    "/mnt/cpfs/prediction/lyyy/myself/WE/data/smoke_1scene/scenarios/original/navtest_failures/all_scenarios.pkl"
)
VOCAB = Path("/mnt/cpfs/prediction/lyyy/myself/WE/data/hf/data/alg_engine/test_8192_kmeans.npy")


def _synthetic_scene() -> dict:
    """Minimal WE-like scene for portable unit tests."""
    n = 12
    t = np.arange(n, dtype=np.float64)
    ego_pos = np.stack([t * 2.0, np.zeros(n), np.zeros(n)], axis=1)
    ego_vel = np.stack([np.full(n, 2.0), np.zeros(n)], axis=1)
    ego_head = np.zeros(n)
    other_pos = ego_pos + np.array([0.0, 5.0, 0.0])
    return {
        "id": "synthetic_nexus_unit",
        "sdc_id": "ego",
        "log_length": n,
        "object_track": {
            "ego": {
                "type": "VEHICLE",
                "state": {
                    "position": ego_pos,
                    "heading": ego_head,
                    "velocity": ego_vel,
                    "valid": np.ones(n),
                    "length": np.full(n, 5.0),
                    "width": np.full(n, 2.0),
                    "height": np.full(n, 1.5),
                },
            },
            "agent_a": {
                "type": "VEHICLE",
                "state": {
                    "position": other_pos,
                    "heading": ego_head.copy(),
                    "velocity": ego_vel.copy(),
                    "valid": np.ones(n),
                    "length": np.full(n, 4.5),
                    "width": np.full(n, 2.0),
                    "height": np.full(n, 1.5),
                },
            },
        },
        "map_features": {
            "lane0": {
                "type": "LANE_SURFACE_STREET",
                "polyline": np.stack([np.linspace(-10, 40, 20), np.zeros(20)], axis=1),
            },
            "cw0": {
                "type": "CROSSWALK",
                # polygon only (no polyline) — adapter must encode via polygon fallback
                "polygon": np.array(
                    [[5.0, -2.0], [8.0, -2.0], [8.0, 2.0], [5.0, 2.0], [5.0, -2.0]],
                    dtype=np.float64,
                ),
            },
        },
    }


class TestNexusSynthetic(unittest.TestCase):
    def test_encode_decode_local(self):
        rng = np.random.RandomState(0)
        raw = rng.randn(3, 21, 8)
        n = nx.encode_scene_tensor_np(raw)
        d = nx.decode_scene_tensor_np(n)
        self.assertLessEqual(float(np.max(np.abs(d - raw))), 1e-6)

    def test_near_stationary_inconsistent_filtered(self):
        # Craft a vocab traj that barely moves but terminal speed would be high if
        # we incorrectly kept inconsistent near-stationary plans.
        # Near-zero path with a jump at the end → high terminal speed + short path.
        traj = np.zeros((40, 3), dtype=np.float64)
        traj[-1, 0] = 20.0  # last 0.1s sample far → after ::5 sampling may create speed
        # Build many small steps then one large: force path_length small via zeros then spike
        traj[:, 0] = 0.0
        traj[35:, 0] = np.linspace(0, 25, 5)  # late spike
        fut = nx.vocab_candidate_to_nexus_ego_future(traj, np.array([0.0, 0.0]), 0.0)
        reasons = nx.candidate_physics_reject_reasons(fut["stats"])
        # Either near_stationary inconsistent or step/acc/speed — must reject
        self.assertTrue(len(reasons) > 0, msg=f"stats={fut['stats']} reasons={reasons}")

        # Explicit near-stationary inconsistency unit: path < 0.5, speed > 1
        stats = {
            "within_100m_radius": True,
            "max_step_m": 1.0,
            "terminal_speed_m_s": 5.0,
            "real_path_length_m": 0.1,
            "heading_vel_aligned": True,
            "max_acc_m_s2": 1.0,
            "max_yaw_rate_rad_s": 0.1,
            "max_heading_jump_rad": 0.1,
            "continuity_ok": True,
        }
        r = nx.candidate_physics_reject_reasons(stats)
        self.assertIn("near_stationary_speed_inconsistent", r)

    def test_heading_vel_acc_yaw_continuity_filters(self):
        base = {
            "within_100m_radius": True,
            "max_step_m": 1.0,
            "terminal_speed_m_s": 5.0,
            "real_path_length_m": 10.0,
            "heading_vel_aligned": True,
            "max_acc_m_s2": 1.0,
            "max_yaw_rate_rad_s": 0.1,
            "max_heading_jump_rad": 0.1,
            "continuity_ok": True,
        }
        self.assertEqual(nx.candidate_physics_reject_reasons(base), [])
        s = dict(base, heading_vel_aligned=False)
        self.assertIn("heading_velocity_misaligned", nx.candidate_physics_reject_reasons(s))
        s = dict(base, max_acc_m_s2=13.0)
        self.assertIn("acceleration_gt_12", nx.candidate_physics_reject_reasons(s))
        s = dict(base, max_yaw_rate_rad_s=math.pi + 0.1)
        self.assertIn("yaw_rate_gt_pi", nx.candidate_physics_reject_reasons(s))
        s = dict(base, continuity_ok=False)
        self.assertIn("trajectory_discontinuity", nx.candidate_physics_reject_reasons(s))

    def test_crosswalk_polygon_fallback(self):
        scene = _synthetic_scene()
        cutoff = 4
        st = scene["object_track"]["ego"]["state"]
        center = np.asarray(st["position"], dtype=np.float64)[cutoff, :2]
        heading = float(np.asarray(st["heading"]).reshape(-1)[cutoff])
        fut = {
            "position": np.tile(center, (16, 1)),
            "heading": np.full(16, heading),
            "velocity": np.zeros((16, 2)),
        }
        bundle = nx.build_nexus_scene_bundle(scene, cutoff, fut)
        cov = bundle.map_coverage
        self.assertGreaterEqual(cov["nexus_type_counts"].get("CROSSWALK", 0), 1)
        self.assertGreaterEqual(cov.get("polygon_fallback_used", 0), 1)
        audit = cov["audit"]["diagnosis"]["CROSSWALK"]
        self.assertGreaterEqual(audit["with_polygon_only"], 1)
        # Upstream types absent
        self.assertEqual(cov["audit"]["conclusion"]["LANE_CONNECTOR"], "upstream_absent")
        self.assertEqual(cov["audit"]["conclusion"]["STOP_LINE"], "upstream_absent")

    def test_filter_vocab_continue_not_pass(self):
        vocab = np.zeros((4, 40, 3), dtype=np.float64)
        # idx0: good forward motion
        vocab[0, :, 0] = np.linspace(0.1, 8.0, 40)
        # idx1: near-stationary inconsistent (tiny path, but force high terminal via stats path)
        # Use reject_reasons directly for idx1 stats injection through constructed future
        kept, counts = nx.filter_vocab_candidates(vocab, np.array([0.0, 0.0]), 0.0)
        # At least the good one may remain; inconsistent reject reason must be counted when present
        # Craft explicit reject via candidate_physics_reject_reasons already tested;
        # here ensure filter returns list and never silently keeps path<0.5 & speed>1
        for c in kept:
            st = c["future"]["stats"]
            if st["real_path_length_m"] < 0.5:
                self.assertLessEqual(st["terminal_speed_m_s"], 1.0)


@unittest.skipUnless(SCENE_PKL.exists() and VOCAB.exists(), "integration data absent")
class TestNexusIntegration(unittest.TestCase):
    """Requires local CPFS absolute paths; skipped when data missing."""

    @classmethod
    def setUpClass(cls):
        cls.scene = next(iter(pickle.load(open(SCENE_PKL, "rb")).values()))
        cls.vocab = np.load(VOCAB)
        cls.cutoff = 4

    def test_history_vehicles(self):
        toks = nx.select_history_vehicles(self.scene, self.cutoff)
        self.assertGreaterEqual(len(toks), 1)
        self.assertLessEqual(len(toks), 128)

    def test_future_leakage_cleared(self):
        sdc = self.scene["sdc_id"]
        st = self.scene["object_track"][sdc]["state"]
        center = np.asarray(st["position"], dtype=np.float64)[self.cutoff, :2]
        heading = float(np.asarray(st["heading"]).reshape(-1)[self.cutoff])
        fut = nx.vocab_candidate_to_nexus_ego_future(self.vocab[100], center, heading)
        bundle = nx.build_nexus_scene_bundle(self.scene, self.cutoff, fut)
        self.assertTrue(bundle.future_leakage_audit["task_mask_future_all_zero"])
        self.assertTrue(bundle.future_leakage_audit["raw_future_all_zero"])

    def test_roundtrip_history(self):
        rt = nx.roundtrip_history_test(self.scene, self.cutoff)
        self.assertTrue(rt["ok"], msg=rt)

    def test_vocab_extension_16(self):
        sdc = self.scene["sdc_id"]
        st = self.scene["object_track"][sdc]["state"]
        center = np.asarray(st["position"], dtype=np.float64)[self.cutoff, :2]
        heading = float(np.asarray(st["heading"]).reshape(-1)[self.cutoff])
        fut = nx.vocab_candidate_to_nexus_ego_future(self.vocab[200], center, heading)
        self.assertEqual(fut["position"].shape, (16, 2))
        self.assertIn("real_hash", fut)


if __name__ == "__main__":
    unittest.main()
