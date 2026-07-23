#!/usr/bin/env python3
"""Train-side Nexus generate at arbitrary plan_idx/seed (cutoff=4). Nexus env python."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

WE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WE_ROOT))
sys.path.insert(0, str(WE_ROOT / "scripts"))

from frozen_state_lib import DEFAULT_CUTOFF, DEFAULT_HORIZON, load_scene_dict, save_json  # noqa: E402
from run_cutoff4_three_source_smoke import apply_physics_log_fallback  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-pkl", type=Path, required=True)
    ap.add_argument("--scene-id", type=str, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--plan-idx", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--vocab",
        type=Path,
        default=Path("/mnt/cpfs/prediction/lyyy/myself/WE/data/hf/data/alg_engine/test_8192_kmeans.npy"),
    )
    ap.add_argument(
        "--sidecar-root",
        type=Path,
        default=Path("/mnt/cpfs/prediction/lyyy/myself/WE/nexus_sidecar"),
    )
    ap.add_argument("--cutoff", type=int, default=DEFAULT_CUTOFF)
    ap.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    ap.add_argument("--device", type=str, default="cuda:0")
    args = ap.parse_args()

    from adapters.traffic_models import nexus as nx
    from run_nexus_sidecar_smoke import (
        bundle_to_features,
        load_nexus_model,
        run_inference,
        setup_nexus_path,
    )

    if int(args.cutoff) != 4:
        raise SystemExit(f"Nexus adapter requires cutoff=4, got {args.cutoff}")

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_nexus_path(args.sidecar_root)

    scene = load_scene_dict(args.scene_pkl)
    sid = str(scene.get("id", ""))
    if args.scene_id not in sid and sid not in args.scene_id:
        # accept if suffix token matches
        if not sid.endswith(args.scene_id.split("-")[-1]) and args.scene_id not in sid:
            save_json(out_dir / "nexus_fail.json", {"reason": "scene_id_mismatch", "got": sid, "want": args.scene_id})
            return 2

    plan_idx = int(args.plan_idx)
    seed = int(args.seed)
    cutoff = int(args.cutoff)
    horizon = int(args.horizon)
    vocab = np.load(args.vocab)

    sdc = scene["sdc_id"]
    ego_st = scene["object_track"][sdc]["state"]
    ego_center = np.asarray(ego_st["position"], dtype=np.float64)[cutoff, :2]
    ego_heading = float(np.asarray(ego_st["heading"]).reshape(-1)[cutoff])
    ego_fut = nx.vocab_candidate_to_nexus_ego_future(vocab[plan_idx], ego_center, ego_heading)

    rt = nx.roundtrip_history_test(scene, cutoff)
    if not rt.get("ok"):
        save_json(out_dir / "nexus_fail.json", {"reason": "roundtrip", "rt": rt})
        return 3
    if not rt.get("leakage", {}).get("task_mask_future_all_zero", False):
        save_json(out_dir / "nexus_fail.json", {"reason": "future_leakage", "rt": rt})
        return 4

    import torch

    device = args.device if torch.cuda.is_available() else "cpu"
    model, load_info = load_nexus_model(args.sidecar_root, device)
    codec2 = nx.try_import_official_codec()
    ego_L = float(codec2["EGO_LENGTH"]) if codec2.get("ok") else nx.EGO_LENGTH_PACIFICA
    ego_W = float(codec2["EGO_WIDTH"]) if codec2.get("ok") else nx.EGO_WIDTH_PACIFICA
    try:
        fb = model.get_list_of_required_feature()[0]
        fb_agents = list(getattr(fb, "_num_max_agents", nx.DEFAULT_NUM_MAX_AGENTS))
    except Exception:
        fb_agents = list(nx.DEFAULT_NUM_MAX_AGENTS)

    bundle = nx.build_nexus_scene_bundle(
        scene,
        cutoff,
        ego_fut,
        ego_length=ego_L,
        ego_width=ego_W,
        num_max_agents=fb_agents,
    )
    if not bundle.future_leakage_audit.get("task_mask_future_all_zero"):
        save_json(out_dir / "nexus_fail.json", {"reason": "bundle_leakage", "audit": bundle.future_leakage_audit})
        return 4

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    noise_shape = (1,) + bundle.tensor.shape
    z0 = torch.randn(noise_shape, device=device)
    noise0_hash = hashlib.sha256(z0.detach().cpu().numpy().tobytes()).hexdigest()
    cpu_state = torch.get_rng_state()
    cuda_states = (
        [torch.cuda.get_rng_state(i) for i in range(torch.cuda.device_count())]
        if torch.cuda.is_available()
        else None
    )
    feats = bundle_to_features(bundle, device)
    out, pre_h, post_h = run_inference(model, feats, z0, cpu_state, cuda_states)
    sampled = out["sampled_tensor"].detach().cpu().numpy()[0]
    decoded = nx.decode_sampled_to_world(sampled, bundle)
    hold = nx.ego_hold_errors(decoded[sdc], ego_fut)
    hold_ok = hold["all_16"]["pos_max_m"] <= 1e-4 and hold["all_16"]["heading_max_rad"] <= 1e-4

    pack = nx.to_future_pack(
        decoded,
        scene,
        cutoff,
        horizon,
        provenance={
            "nexus_commit": "71c31ca848da94c969322a40f0f4ae2af8ca8129",
            "ckpt_sha256": load_info["ckpt_sha256"],
            "plan_idx": plan_idx,
            "plan_idx_source": "static_feasible_median_path_length",
            "seed": seed,
            "noise_hash": noise0_hash,
            "slot_tokens": [m.token for m in bundle.slot_meta],
            "map_coverage": bundle.map_coverage,
            "future_leakage": bundle.future_leakage_audit,
            "note": "train_side_le10; conditioning ≠ smoke NR 1333",
        },
    )
    physics = nx.physics_check_futures(pack["futures"])
    pack = apply_physics_log_fallback(pack, scene, cutoff, horizon, physics)
    physics_after = nx.physics_check_futures(pack["futures"])

    stem = f"future_nexus_plan{plan_idx}_seed{seed}"
    with (out_dir / f"{stem}.pkl").open("wb") as f:
        pickle.dump(pack, f)
    meta: Dict[str, Any] = {
        "cutoff": cutoff,
        "horizon": horizon,
        "plan_idx": plan_idx,
        "plan_idx_source": "static_feasible_median_path_length",
        "seed": seed,
        "noise_hash": noise0_hash,
        "future_hash": pack["future_hash"],
        "n_agents": pack["n_agents"],
        "coverage_counts": {
            k: len(v) if isinstance(v, list) else v
            for k, v in pack["coverage"].items()
            if k != "fallback_reasons"
        },
        "physics_before_fallback": physics,
        "physics_after_fallback": physics_after,
        "physics_log_fallback_tokens": pack.get("physics_log_fallback_tokens", []),
        "ego_hold": hold,
        "hold_ok": hold_ok,
        "strict_load": load_info,
        "future_leakage": bundle.future_leakage_audit,
        "roundtrip": {k: rt[k] for k in rt if k != "leakage"},
        "pre_rng_hash": pre_h,
        "post_rng_hash": post_h,
        "requested_ego_real_hash": ego_fut.get("real_hash"),
        "scene_id": sid,
    }
    save_json(out_dir / f"{stem}_meta.json", meta)
    # convenience symlink-like copy for seed0 default name used by scorer
    if seed == 0:
        with (out_dir / "future_nexus_seed0.pkl").open("wb") as f:
            pickle.dump(pack, f)
        save_json(out_dir / "future_nexus_seed0_meta.json", meta)
    print(json.dumps({"ok": hold_ok, "seed": seed, "plan_idx": plan_idx, "future_hash": pack["future_hash"]}, indent=2))
    return 0 if hold_ok else 6


if __name__ == "__main__":
    raise SystemExit(main())
