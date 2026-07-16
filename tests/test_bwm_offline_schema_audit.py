#!/usr/bin/env python3
"""Synthetic unit tests for BWM-Offline schema audit helpers.

Portable fixtures only. Absolute-path integration tests skip when data absent.
"""

from __future__ import annotations

import pickle
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

WE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WE_ROOT))
sys.path.insert(0, str(WE_ROOT / "scripts"))

import audit_bwm_augmented_schema as audit  # noqa: E402

REAL_PKL = Path(
    "/mnt/cpfs/prediction/lyyy/myself/WE/data/bwm_offline_audit/hf/"
    "data/sim_engine/scenarios/augmented/navtrain_50pct_collision/all_scenarios.pkl"
)
EXPECT_SIZE = 3130339854
EXPECT_SHA = "55328d2aefe231eee36ae82521223b2bb0e9061952682db06ea5bc0a02120d1a"


def _synthetic_scenario(sid: str = "syn-001") -> dict:
    T = 12
    pos = np.zeros((T, 3), dtype=np.float64)
    pos[:, 0] = np.arange(T) * 0.5
    return {
        "id": sid,
        "log_length": T,
        "sample_rate": 10,
        "sdc_id": "ego",
        "object_track": {
            "ego": {
                "type": "VEHICLE",
                "state": {
                    "position": pos,
                    "heading": np.zeros(T),
                    "velocity": np.stack([np.full(T, 1.0), np.zeros(T)], axis=1),
                    "valid": np.ones(T),
                    "length": np.full((T, 1), 5.0),
                    "width": np.full((T, 1), 2.0),
                    "height": np.full((T, 1), 1.5),
                },
            },
            "a1": {
                "type": "VEHICLE",
                "state": {
                    "position": pos + np.array([0, 3, 0]),
                    "heading": np.zeros(T),
                    "velocity": np.zeros((T, 2)),
                    "valid": np.ones(T),
                    "length": np.full((T, 1), 4.5),
                    "width": np.full((T, 1), 2.0),
                    "height": np.full((T, 1), 1.5),
                },
            },
        },
        "map_features": {
            "lane0": {
                "type": "LANE_SURFACE_STREET",
                "polyline": np.stack([np.linspace(0, 10, 5), np.zeros(5)], axis=1),
            }
        },
        "dynamic_map_states": {},
        "metadata": {"coordinate": "world", "metadrive_processed": False},
    }


class TestBwmOfflineSynthetic(unittest.TestCase):
    def test_ndarray_is_leaf(self):
        arr = np.arange(12, dtype=np.float64).reshape(3, 4)
        leaf = audit.ndarray_leaf_stats(arr)
        self.assertEqual(leaf["kind"], "ndarray")
        self.assertEqual(leaf["shape"], [3, 4])
        self.assertIn("dtype", leaf)
        # walker must not expand ndarray children
        obj = {"arr": arr, "nested": {"x": 1}}
        found = audit.find_keys_recursive(obj, ["x"], max_depth=3)
        self.assertIn("x", found["present"])

    def test_restricted_unpickler_rejects_unknown_global(self):
        # Craft a pickle that references forbidden builtins.eval via GLOBAL
        # Use pickletools-level: dump an object requiring a forbidden class.
        payload = pickle.dumps({"a": 1}, protocol=4)
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            f.write(payload)
            path = Path(f.name)
        try:
            obj = audit.restricted_load(path)
            self.assertEqual(obj, {"a": 1})
        finally:
            path.unlink(missing_ok=True)

        # Manually write a GLOBAL to os.system
        bad = (
            b"\x80\x04\x95\x1a\x00\x00\x00\x00\x00\x00\x00"
            b"cposix\nsystem\n"
            b"\x94."
        )
        # Simpler: use pickle with reduce on a forbidden type via custom
        class Boom:
            def __reduce__(self):
                return (eval, ("2+2",))

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            try:
                pickle.dump(Boom(), f, protocol=4)
            except Exception:
                # fallback write known bad GLOBAL bytes for builtins.eval
                f.seek(0)
                f.truncate()
                f.write(b"\x80\x04cbuiltins\neval\n.")
            path = Path(f.name)
        try:
            with self.assertRaises(Exception):
                audit.restricted_load(path)
        finally:
            path.unlink(missing_ok=True)

    def test_source_absent_not_inferred(self):
        sc = _synthetic_scenario("2021.log-token-001")
        fields = audit.field_presence(sc, audit.SOURCE_FIELD_CANDIDATES)
        self.assertTrue(all(v == "absent" for v in fields.values()))
        # pairing must not upgrade via filename
        summary = {
            "n_scenarios": 1,
            "required_first_level_all": True,
            "has_basic_tracks": True,
            "has_map_features": True,
            "has_sample_rate_and_log_length": True,
            "explicit_cutoff_present": False,
            "dual_trajectory_present": False,
            "reward_fields_present": False,
            "field_coverage": {
                "source_original": {k: 0.0 for k in audit.SOURCE_FIELD_CANDIDATES},
                "ego_conditioning": {k: 0.0 for k in audit.EGO_COND_FIELD_CANDIDATES},
            },
            "self_contained_original_equality_proof": False,
        }
        grade = audit.classify_pairing(summary)
        self.assertEqual(grade["grade"], "C")
        self.assertIn("source_fields_absent_no_filename_inference", grade["evidence"])

    def test_classify_d_without_tracks(self):
        summary = {
            "n_scenarios": 0,
            "required_first_level_all": False,
            "has_basic_tracks": False,
            "has_map_features": False,
            "has_sample_rate_and_log_length": False,
            "explicit_cutoff_present": False,
            "dual_trajectory_present": False,
            "reward_fields_present": False,
            "field_coverage": {"source_original": {}, "ego_conditioning": {}},
            "self_contained_original_equality_proof": False,
        }
        self.assertEqual(audit.classify_pairing(summary)["grade"], "D")

    def test_audit_one_scenario_time_anchor_absent_cutoff(self):
        sc = _synthetic_scenario()
        rep = audit.audit_one_scenario(sc, "k0")
        self.assertEqual(rep["time_anchor"]["explicit_time_bounds"]["cutoff"], "absent")
        self.assertEqual(rep["source_fields"]["source_token"], "absent")
        self.assertEqual(rep["ego_conditioning_fields"]["plan_idx"], "absent")
        self.assertFalse(rep["required_first_level"]["object_track"] is False)

    def test_reservoir_bounded(self):
        keys = [f"s{i}" for i in range(100)]
        sample = audit.reservoir_sample_keys(keys, 10, seed=20260716)
        self.assertEqual(len(sample), 10)
        sample2 = audit.reservoir_sample_keys(keys, 10, seed=20260716)
        self.assertEqual(sample, sample2)


@unittest.skipUnless(REAL_PKL.exists(), "integration data absent")
class TestBwmOfflineIntegration(unittest.TestCase):
    def test_file_gate(self):
        meta = audit.verify_file_gate(REAL_PKL, EXPECT_SIZE, EXPECT_SHA)
        self.assertEqual(meta["size"], EXPECT_SIZE)
        self.assertEqual(meta["sha256"], EXPECT_SHA)


if __name__ == "__main__":
    unittest.main()
