#!/usr/bin/env python3
"""Synthetic unit tests for BWM-Offline schema audit helpers.

Portable fixtures only. Absolute-path integration tests skip when data absent.
"""

from __future__ import annotations

import json
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


def _synthetic_scenario(sid: str = "syn-001", *, with_source: bool = False, with_ego: bool = False) -> dict:
    T = 12
    pos = np.zeros((T, 3), dtype=np.float64)
    # 0.5 m / step at 1.0 m/s → consistent with dt=0.5
    pos[:, 0] = np.arange(T) * 0.5
    sc = {
        "id": sid,
        "name": sid,
        "token": sid + "-goal_conditional_copy_with_noise",
        "log_length": T,
        "sample_rate": 2,
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
                    "velocity": np.stack([np.full(T, 1.0), np.zeros(T)], axis=1),
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
        "cameras": {"cam0": {"data_path": "sensor_blobs/cam0/0.jpg", "cam_intrinsic": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}},
        "lidar": {},
        "metadata": {
            "coordinate": "world",
            "metadrive_processed": False,
            "openscene_data_infos_dict": [
                {"token": "f0", "cams": {"CAM": {"data_path": "x.jpg"}}, "can_bus": np.zeros(10)}
            ],
            "nested": {"inner_key": 1},
        },
    }
    if with_source:
        sc["source_token"] = "orig-token"
        sc["metadata"]["original_scene_id"] = "orig-scene"
    if with_ego:
        sc["plan_idx"] = 7
        sc["ego_conditioning"] = {"plan_idx": 7}
        sc["cutoff"] = 4
        sc["original_object_track"] = sc["object_track"]
        sc["bwm_object_track"] = sc["object_track"]
    return sc


def _base_summary(**overrides):
    n = overrides.pop("n_scenarios", 10)
    src = {k: 0.0 for k in audit.SOURCE_FIELD_CANDIDATES}
    ego = {k: 0.0 for k in audit.EGO_COND_FIELD_CANDIDATES}
    base = {
        "n_scenarios": n,
        "required_first_level_all": True,
        "has_basic_tracks": True,
        "has_map_features": True,
        "has_sample_rate_and_log_length": True,
        "explicit_cutoff_present": False,
        "dual_trajectory_present": False,
        "reward_fields_present": False,
        "source_structural_evidence": False,
        "ego_structural_evidence": False,
        "self_contained_original_equality_proof": False,
        "field_coverage": {"source_original": src, "ego_conditioning": ego},
    }
    base.update(overrides)
    return base


class TestBwmOfflineSynthetic(unittest.TestCase):
    def test_ndarray_is_leaf(self):
        arr = np.arange(12, dtype=np.float64).reshape(3, 4)
        leaf = audit.ndarray_leaf_stats(arr)
        self.assertEqual(leaf["kind"], "ndarray")
        reg = audit.NestedKeyRegistry()
        reg.observe_scenario("s0", {"metadata": {"arr": arr, "x": 1}})
        paths = {p["path"] for p in reg.to_report(1)["paths"]}
        self.assertIn("metadata.arr", paths)
        self.assertIn("metadata.x", paths)

    def test_explicit_zero_coverage(self):
        cov = audit.explicit_zero_coverage(("a", "b", "c"), audit.Counter({"a": 5}), 10)
        self.assertEqual(cov["a"], 0.5)
        self.assertEqual(cov["b"], 0.0)
        self.assertEqual(cov["c"], 0.0)

    def test_restricted_unpickler_rejects_unknown_global(self):
        payload = pickle.dumps({"a": 1}, protocol=4)
        obj = audit.restricted_loads(payload)
        self.assertEqual(obj, {"a": 1})
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            f.write(b"\x80\x04cbuiltins\neval\n.")
            path = Path(f.name)
        try:
            with self.assertRaises(Exception):
                audit.restricted_load(path)
        finally:
            path.unlink(missing_ok=True)

    def test_restricted_unpickler_accepts_numpy(self):
        payload = pickle.dumps({"x": np.arange(4, dtype=np.float64)}, protocol=4)
        obj = audit.restricted_loads(payload)
        self.assertTrue(isinstance(obj["x"], np.ndarray))
        self.assertEqual(obj["x"].shape, (4,))

    def test_name_heuristic_does_not_upgrade_grade(self):
        sid = "2021.05.12.23.36.44_veh-35_00785_01041-6c61fae57b175318-000"
        sc = _synthetic_scenario(sid)
        parsed = audit.parse_scenario_names(sid, sc)
        self.assertEqual(parsed["label"], "heuristic_grouping_from_name")
        self.assertEqual(parsed["variant_index"], "000")
        grade = audit.classify_pairing(_base_summary())
        self.assertEqual(grade["grade"], "C")

    def test_classify_d_without_tracks(self):
        s = _base_summary(n_scenarios=0, has_basic_tracks=False, required_first_level_all=False, has_map_features=False)
        self.assertEqual(audit.classify_pairing(s)["grade"], "D")

    def test_classify_d_missing_map(self):
        s = _base_summary(has_map_features=False, required_first_level_all=False)
        self.assertEqual(audit.classify_pairing(s)["grade"], "D")

    def test_classify_c_default(self):
        self.assertEqual(audit.classify_pairing(_base_summary())["grade"], "C")

    def test_classify_c_rare_source_hit_no_upgrade(self):
        src = {k: 0.0 for k in audit.SOURCE_FIELD_CANDIDATES}
        src["source_token"] = 0.01  # rare hit
        s = _base_summary(field_coverage={"source_original": src, "ego_conditioning": {k: 0.0 for k in audit.EGO_COND_FIELD_CANDIDATES}})
        s["source_structural_evidence"] = True
        self.assertEqual(audit.classify_pairing(s)["grade"], "C")

    def test_classify_b_majority_source(self):
        src = {k: 0.0 for k in audit.SOURCE_FIELD_CANDIDATES}
        src["source_token"] = 1.0
        ego = {k: 0.0 for k in audit.EGO_COND_FIELD_CANDIDATES}
        s = _base_summary(
            field_coverage={"source_original": src, "ego_conditioning": ego},
            source_structural_evidence=True,
        )
        self.assertEqual(audit.classify_pairing(s)["grade"], "B")

    def test_classify_a_full_proof(self):
        src = {k: 0.0 for k in audit.SOURCE_FIELD_CANDIDATES}
        src["source_token"] = 1.0
        ego = {k: 0.0 for k in audit.EGO_COND_FIELD_CANDIDATES}
        ego["plan_idx"] = 1.0
        s = _base_summary(
            field_coverage={"source_original": src, "ego_conditioning": ego},
            source_structural_evidence=True,
            ego_structural_evidence=True,
            explicit_cutoff_present=True,
            dual_trajectory_present=True,
            self_contained_original_equality_proof=True,
        )
        self.assertEqual(audit.classify_pairing(s)["grade"], "A")

    def test_kinematics_prefers_dt(self):
        # Build mini dataset consistent with dt=0.5
        sc = _synthetic_scenario("2021.05.12.23.36.44_veh-35_00785_01041-aaaaaaaaaaaaaaaa-000")
        data = {"k": sc}
        out = audit.kinematics_dt_probe(data, ["k"], max_agents=5)
        self.assertGreater(out["n_step_pairs"], 0)
        self.assertIn(out["better_dt"], ("dt_0.5s_better", "dt_0.1s_better", "ambiguous_no_clear_margin", "ambiguous"))

    def test_end_to_end_synthetic_pickle_audit(self):
        data = {}
        for i in range(3):
            sid = f"2021.05.12.23.36.44_veh-35_00785_01041-aaaaaaaaaaaaaaa{i}-{i:03d}"
            data[sid] = _synthetic_scenario(sid)
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            pickle.dump(data, f, protocol=4)
            path = Path(f.name)
        try:
            digest = audit.sha256_file(path)
            size = path.stat().st_size
            # bypass size gate by calling audit_dataset directly after restricted_load path
            summary = audit.audit_dataset(path, seed=0, random_samples=2, max_wall_s=60, max_rss_gb=4)
            self.assertTrue(summary["ok"])
            self.assertEqual(summary["n_scenarios"], 3)
            # explicit zeros present
            for k in audit.SOURCE_FIELD_CANDIDATES:
                self.assertIn(k, summary["field_coverage"]["source_original"])
                self.assertEqual(summary["field_coverage"]["source_original"][k], 0.0)
            self.assertEqual(summary["pairing_grade"]["grade"], "C")
            self.assertEqual(summary["heuristic_grouping_from_name"]["label"], "heuristic_grouping_from_name")
            # JSON serializable under limit
            payload = {"summary": summary}
            text = json.dumps(payload, default=audit._json_default)
            self.assertLess(len(text.encode()), audit.JSON_HARD_LIMIT_BYTES)
            # restricted roundtrip of file
            obj = audit.restricted_load(path)
            self.assertEqual(len(obj), 3)
            _ = digest, size
        finally:
            path.unlink(missing_ok=True)


@unittest.skipUnless(REAL_PKL.exists(), "integration data absent")
class TestBwmOfflineIntegration(unittest.TestCase):
    def test_file_gate(self):
        meta = audit.verify_file_gate(REAL_PKL, EXPECT_SIZE, EXPECT_SHA)
        self.assertEqual(meta["size"], EXPECT_SIZE)
        self.assertEqual(meta["sha256"], EXPECT_SHA)


if __name__ == "__main__":
    unittest.main()
