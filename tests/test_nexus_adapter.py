#!/usr/bin/env python3
"""CPU unit tests for Nexus adapter (no GPU, no checkpoint)."""

from __future__ import annotations

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


class TestNexusAdapter(unittest.TestCase):
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
        self.assertTrue(bundle.future_leakage_audit["matches_encode_zeros"])
        # Ensure we did not copy log GT into non-ego future
        for j, token in enumerate(bundle.history_tokens):
            slot = 1 + j
            ot = self.scene["object_track"][token]["state"]
            if len(np.asarray(ot["position"])) > self.cutoff + 1:
                # Even if log has future, our raw future is zero
                self.assertTrue(np.allclose(bundle.raw_unnormalized[slot, nx.N_PAST :], 0.0))

    def test_roundtrip_history(self):
        rt = nx.roundtrip_history_test(self.scene, self.cutoff)
        self.assertTrue(rt["ok"], msg=rt)

    def test_encode_decode_local(self):
        rng = np.random.RandomState(0)
        raw = rng.randn(3, 21, 8)
        n = nx.encode_scene_tensor_np(raw)
        d = nx.decode_scene_tensor_np(n)
        self.assertLessEqual(float(np.max(np.abs(d - raw))), 1e-6)

    def test_vocab_extension_16(self):
        sdc = self.scene["sdc_id"]
        st = self.scene["object_track"][sdc]["state"]
        center = np.asarray(st["position"], dtype=np.float64)[self.cutoff, :2]
        heading = float(np.asarray(st["heading"]).reshape(-1)[self.cutoff])
        fut = nx.vocab_candidate_to_nexus_ego_future(self.vocab[200], center, heading)
        self.assertEqual(fut["position"].shape, (16, 2))
        self.assertEqual(fut["heading"].shape, (16,))
        self.assertIn("real_hash", fut)
        self.assertIn("ext_hash", fut)


if __name__ == "__main__":
    unittest.main()
