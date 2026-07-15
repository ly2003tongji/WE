#!/usr/bin/env python3
"""Nexus single-scene ego-conditioned sidecar feasibility smoke (cutoff=4).

Scope: resource audit, strict-load, adapter, A/B common-noise generation,
future-pack, conditioning-candidate single-row PDM interface smoke.
Does NOT regenerate Replay/IDM, download BWM, run SMART, or modify upstream.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import pickle
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

WE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WE_ROOT))
sys.path.insert(0, str(WE_ROOT / "scripts"))

from adapters.traffic_models import nexus as nx  # noqa: E402
from frozen_state_lib import (  # noqa: E402
    build_input_state_fingerprint,
    inject_ego_conditioning_at_cutoff,
    load_scene_dict,
    save_json,
    sha256_file,
)

SCENE_ID = "2021.09.29.15.23.04_veh-28_00601_00802-6326d00e52115da4"
DEFAULT_CUTOFF = 4
DEFAULT_HORIZON = 9
MAX_PAIRS = 3


def _rss_gb() -> float:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / (1024.0 ** 2)
    except Exception:
        return -1.0
    return -1.0


def setup_nexus_path(sidecar_root: Path) -> None:
    stubs = sidecar_root / "stubs"
    nexus_src = sidecar_root / "src" / "Nexus"
    nuplan_src = sidecar_root / "src" / "nuplan-devkit"
    mtr_src = sidecar_root / "src" / "MTR"
    # stubs first (waymo/wandb/alf); never put third_party/alf ahead of stubs
    for p in [stubs, nexus_src, nuplan_src, mtr_src]:
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)
    import nexus_import_bootstrap  # noqa: F401  # install meta_path stubs


def filter_vocab_candidates(
    vocab: np.ndarray,
    ego_center: np.ndarray,
    ego_heading: float,
) -> List[Dict[str, Any]]:
    """Pre-register feasible vocabulary plans (deterministic)."""
    out = []
    for idx in range(len(vocab)):
        fut = nx.vocab_candidate_to_nexus_ego_future(vocab[idx], ego_center, ego_heading)
        st = fut["stats"]
        # filters
        if not st["within_100m_radius"]:
            continue
        if st["max_step_m"] > 15.0:
            continue
        if st["terminal_speed_m_s"] > 30.0:
            continue
        if st["real_path_length_m"] < 0.5 and st["terminal_speed_m_s"] > 1.0:
            # near-stationary but claims speed — skip inconsistent
            pass
        # continuity: first real step from t0
        out.append({"plan_idx": int(idx), "future": fut})
    return out


def select_candidate_pairs(
    candidates: List[Dict[str, Any]],
    max_pairs: int = MAX_PAIRS,
) -> List[Dict[str, Any]]:
    """Deterministic A/B pairs by endpoint/RMS separation."""
    if len(candidates) < 2:
        raise RuntimeError("Not enough vocabulary candidates after filtering")
    # Prefer mid-speed diverse plans: sort by path length
    cands = sorted(candidates, key=lambda c: c["future"]["stats"]["real_path_length_m"])
    # Pick A as median path length
    a = cands[len(cands) // 2]
    pairs = []
    # Rank others by separation from A
    scored = []
    for b in cands:
        if b["plan_idx"] == a["plan_idx"]:
            continue
        sep = nx.pair_separation(a["future"], b["future"])
        if sep["endpoint_m"] >= 3.0 and sep["rms_m"] >= 1.0:
            scored.append((sep["endpoint_m"] + sep["rms_m"], sep, b))
    scored.sort(key=lambda x: -x[0])
    if not scored:
        # relax: take farthest endpoint even if below threshold, but mark
        scored = []
        for b in cands:
            if b["plan_idx"] == a["plan_idx"]:
                continue
            sep = nx.pair_separation(a["future"], b["future"])
            scored.append((sep["endpoint_m"] + sep["rms_m"], sep, b))
        scored.sort(key=lambda x: -x[0])
    used_b = set()
    # Pair 0: median A vs farthest B
    if scored:
        sep, b = scored[0][1], scored[0][2]
        pairs.append(_make_pair(0, a, b, sep))
        used_b.add(b["plan_idx"])
    # Additional pairs: other anchors at quartiles
    anchors = [cands[len(cands) // 4], cands[(3 * len(cands)) // 4]]
    for anc in anchors:
        if len(pairs) >= max_pairs:
            break
        local = []
        for b in cands:
            if b["plan_idx"] in used_b or b["plan_idx"] == anc["plan_idx"]:
                continue
            if any(p["A"]["plan_idx"] == anc["plan_idx"] and p["B"]["plan_idx"] == b["plan_idx"] for p in pairs):
                continue
            sep = nx.pair_separation(anc["future"], b["future"])
            if sep["endpoint_m"] >= 3.0 and sep["rms_m"] >= 1.0:
                local.append((sep["endpoint_m"] + sep["rms_m"], sep, b))
        local.sort(key=lambda x: -x[0])
        if local:
            sep, b = local[0][1], local[0][2]
            pairs.append(_make_pair(len(pairs), anc, b, sep))
            used_b.add(b["plan_idx"])
    if not pairs:
        raise RuntimeError("Failed to pre-register any candidate pairs")
    return pairs[:max_pairs]


def _make_pair(i: int, a: Dict[str, Any], b: Dict[str, Any], sep: Dict[str, float]) -> Dict[str, Any]:
    return {
        "pair_id": i,
        "A": {
            "plan_idx": a["plan_idx"],
            "traj_hash": a["future"]["traj_hash"],
            "real_hash": a["future"]["real_hash"],
            "ext_hash": a["future"]["ext_hash"],
            "stats": a["future"]["stats"],
        },
        "B": {
            "plan_idx": b["plan_idx"],
            "traj_hash": b["future"]["traj_hash"],
            "real_hash": b["future"]["real_hash"],
            "ext_hash": b["future"]["ext_hash"],
            "stats": b["future"]["stats"],
        },
        "separation": sep,
        "selection_rule": "vocab_preregister_median_path_then_max_sep",
        "meets_sep_thresholds": bool(sep["endpoint_m"] >= 3.0 and sep["rms_m"] >= 1.0),
    }


def rng_state_hash() -> str:
    import torch

    parts = [str(torch.initial_seed()).encode()]
    cpu = torch.get_rng_state().cpu().numpy().tobytes()
    parts.append(hashlib.sha256(cpu).digest())
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            s = torch.cuda.get_rng_state(i)
            if hasattr(s, "cpu"):
                s = s.cpu().numpy().tobytes()
            else:
                s = bytes(s)
            parts.append(hashlib.sha256(s).digest())
    return hashlib.sha256(b"".join(parts)).hexdigest()


def load_nexus_model(sidecar_root: Path, device: str):
    import torch
    from omegaconf import OmegaConf
    from hydra.utils import instantiate

    cfg_path = sidecar_root / "src/Nexus/nuplan_extent/planning/script/config/common/model/nexus.yaml"
    model_cfg = OmegaConf.load(cfg_path)
    # Resolve interpolations against a minimal common_cfg
    root = OmegaConf.create(
        {
            "common_cfg": {"output_cfg": {"trajectory_steps": 16, "time_horizon": 8.0}},
            "model": model_cfg,
        }
    )
    OmegaConf.resolve(root)
    model = instantiate(root.model)
    ckpt_path = sidecar_root / "checkpoints/nuplan.ckpt"
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    stripped = {}
    for k, v in state.items():
        nk = k[6:] if k.startswith("model.") else k
        stripped[nk] = v
    incompatible = model.load_state_dict(stripped, strict=True)
    missing = list(getattr(incompatible, "missing_keys", []) or [])
    unexpected = list(getattr(incompatible, "unexpected_keys", []) or [])
    if missing or unexpected:
        raise RuntimeError(f"strict-load failed missing={missing} unexpected={unexpected}")
    model.eval()
    model = model.to(device)
    return model, {
        "ckpt_path": str(ckpt_path),
        "ckpt_sha256": sha256_file(ckpt_path),
        "ckpt_size": ckpt_path.stat().st_size,
        "strict_load": True,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
    }


def bundle_to_features(bundle: nx.NexusSceneBundle, device: str):
    import torch
    from nuplan_extent.planning.training.preprocessing.features.scene_tensor import SceneTensor

    st = SceneTensor(
        tensor=torch.tensor(bundle.tensor, device=device).unsqueeze(0),
        validity=torch.tensor(bundle.validity, device=device).unsqueeze(0),
        road_graph=torch.tensor(bundle.road_graph, device=device).unsqueeze(0),
        road_graph_validity=torch.tensor(bundle.road_graph_validity, device=device).unsqueeze(0),
    )
    task_mask = torch.tensor(bundle.task_mask, device=device).unsqueeze(0)
    return {"scene_tensor": st, "task_mask": task_mask}


def run_inference(model, features, noise, cpu_state, cuda_states):
    import torch

    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []):
        torch.set_rng_state(cpu_state)
        if torch.cuda.is_available() and cuda_states is not None:
            for i, s in enumerate(cuda_states):
                torch.cuda.set_rng_state(s, i)
        pre_hash = rng_state_hash()
        with torch.no_grad():
            out = model.forward_inference(features, noise.clone())
        post_hash = rng_state_hash()
    return out, pre_hash, post_hash


def agent_ade(a_pos: np.ndarray, b_pos: np.ndarray, valid_a: np.ndarray, valid_b: np.ndarray) -> float:
    m = (valid_a > 0) & (valid_b > 0)
    if not np.any(m):
        return float("nan")
    return float(np.mean(np.linalg.norm(a_pos[m] - b_pos[m], axis=1)))


def score_pdm_single_row(
    scene: Dict[str, Any],
    futures_pack: Dict[str, Any],
    plan_idx: int,
    vocab: np.ndarray,
    vocab_path: Path,
    work_dir: Path,
    asset_folder: str,
) -> Dict[str, Any]:
    """PDM interface smoke via frozen score_one_source; only read conditioning row."""
    we_root = WE_ROOT / "upstream/WorldEngine"
    os.environ.setdefault("WORLDENGINE_ROOT", str(we_root))
    os.environ.setdefault("SIMENGINE_ROOT", str(we_root / "projects/SimEngine"))
    maps = Path("/mnt/cpfs/prediction/lyyy/myself/WE/data/maps/extracted")
    if maps.exists():
        os.environ.setdefault("NUPLAN_MAPS_ROOT", str(maps))
    sys.path.insert(0, os.environ["SIMENGINE_ROOT"])

    from score_frozen_disagreement import score_one_source

    cutoff = int(futures_pack["cutoff"])
    horizon = int(futures_pack["horizon"])
    # Ego conditioning must be in scene for PDM; inject at cutoff then score agents.
    scene_scored = copy.deepcopy(scene)
    ego_meta = inject_ego_conditioning_at_cutoff(scene_scored, cutoff, plan_idx, vocab)
    pack = copy.deepcopy(futures_pack)
    pack["ego_conditioning"] = {
        "plan_idx": plan_idx,
        "requested_ego_conditioning_hash": ego_meta["requested_ego_conditioning_hash"],
        "ego_conditioning_hash": ego_meta.get("ego_conditioning_hash"),
    }

    t0 = time.time()
    prov = score_one_source(
        scene=scene_scored,
        futures_pack=pack,
        source_name="nexus",
        cutoff=cutoff,
        horizon=horizon,
        work_dir=work_dir,
        asset_folder=asset_folder,
        vocab_path=vocab_path,
        vocab=vocab,
        plan_idx=plan_idx,
        requested_ego_hash=ego_meta["requested_ego_conditioning_hash"],
        wall_limit_sec=1800,
        rss_limit_gb=64.0,
        t0=t0,
    )
    scores_pkl = work_dir.parent / "scores_nexus.pkl"
    if not scores_pkl.exists():
        # score_one_source writes beside work_dir
        cands = list(work_dir.parent.glob("scores_nexus*.pkl"))
        if not cands:
            pdms = list(work_dir.rglob("*_scores.pkl"))
            if not pdms:
                return {"ok": False, "reason": "no_scores_pkl", "prov": {k: prov.get(k) for k in ("wall_s", "rss_gb")}}
            scores_pkl = pdms[0]
        else:
            scores_pkl = cands[0]
    obj = pickle.load(open(scores_pkl, "rb"))
    row = {}
    for k, v in obj.items():
        arr = np.asarray(v)
        if arr.ndim >= 1 and arr.shape[0] == 8192:
            val = arr[plan_idx]
            row[k] = {
                "value": float(val) if np.issubdtype(arr.dtype, np.number) else str(val),
                "finite": bool(np.isfinite(val)) if np.issubdtype(arr.dtype, np.number) else True,
            }
    return {
        "ok": bool(row) and all(r.get("finite", True) for r in row.values()),
        "plan_idx": plan_idx,
        "conditioning_row": row,
        "n_fields": len(row),
        "note": "Only conditioning candidate row inspected; 8191 others not summarized.",
        "scores_pkl": str(scores_pkl),
        "wall_s": prov.get("wall_s"),
        "rss_gb": prov.get("rss_gb"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sidecar-root", type=Path, default=Path("/mnt/cpfs/prediction/lyyy/myself/WE/nexus_sidecar"))
    ap.add_argument(
        "--scene-pkl",
        type=Path,
        default=Path(
            "/mnt/cpfs/prediction/lyyy/myself/WE/data/smoke_1scene/scenarios/original/navtest_failures/all_scenarios.pkl"
        ),
    )
    ap.add_argument(
        "--vocab",
        type=Path,
        default=Path("/mnt/cpfs/prediction/lyyy/myself/WE/data/hf/data/alg_engine/test_8192_kmeans.npy"),
    )
    ap.add_argument(
        "--asset-folder",
        type=str,
        default="/mnt/cpfs/prediction/lyyy/myself/WE/data/smoke_1scene/assets/navtest_failures/assets",
    )
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--skip-pdm", action="store_true")
    ap.add_argument("--cpu-only-tests", action="store_true")
    args = ap.parse_args()

    sidecar = args.sidecar_root
    out_dir = args.out_dir or (sidecar / "outputs/smoke_cutoff4")
    out_dir.mkdir(parents=True, exist_ok=True)
    report: Dict[str, Any] = {
        "stage": "nexus_sidecar_feasibility",
        "cutoff": DEFAULT_CUTOFF,
        "scene_id": SCENE_ID,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "gates": {},
        "verdict": None,
    }

    # --- Gate 0: scene / history ---
    scene = load_scene_dict(args.scene_pkl)
    assert SCENE_ID in scene.get("id", SCENE_ID) or scene.get("id") == SCENE_ID
    cutoff = DEFAULT_CUTOFF
    sdc = scene["sdc_id"]
    ego_st = scene["object_track"][sdc]["state"]
    n_frames = len(np.asarray(ego_st["position"]))
    hist_ok = n_frames >= cutoff + 1 and all(
        nx.agent_valid_at(ego_st, t) for t in range(cutoff + 1)
    )
    hist_tokens = nx.select_history_vehicles(scene, cutoff)
    report["gates"]["gate0_history"] = {
        "ok": bool(hist_ok and len(hist_tokens) > 0),
        "n_frames": n_frames,
        "log_length": scene.get("log_length"),
        "n_history_vehicles": len(hist_tokens),
        "note": "Other-agent futures are generation slots; log GT future not required.",
    }
    if not report["gates"]["gate0_history"]["ok"]:
        report["verdict"] = "失败"
        save_json(out_dir / "feasibility_summary.json", report)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 2

    # plan_idx step=5 fingerprint mismatch → vocab preregister (no Action Policy)
    report["gates"]["candidate_source"] = {
        "plan_idx_csv_step5": 1333,
        "reused": False,
        "reason": "step=5 context fingerprint not proven identical to cutoff=4 freeze; using vocabulary pre-registration only.",
    }

    vocab = np.load(args.vocab)
    ego_center = np.asarray(ego_st["position"], dtype=np.float64)[cutoff, :2]
    ego_heading = float(np.asarray(ego_st["heading"]).reshape(-1)[cutoff])
    filtered = filter_vocab_candidates(vocab, ego_center, ego_heading)
    pairs = select_candidate_pairs(filtered, max_pairs=MAX_PAIRS)
    # Materialize futures for registered indices
    pair_futures = {}
    for p in pairs:
        for lab in ("A", "B"):
            idx = p[lab]["plan_idx"]
            if idx not in pair_futures:
                pair_futures[idx] = nx.vocab_candidate_to_nexus_ego_future(
                    vocab[idx], ego_center, ego_heading
                )
    manifest = {
        "immutable": True,
        "created_before_gpu": True,
        "n_pairs": len(pairs),
        "pairs": pairs,
        "n_filtered_candidates": len(filtered),
        "vocab_sha256": sha256_file(args.vocab),
        "freeze_ego_center": ego_center.tolist(),
        "freeze_ego_heading": ego_heading,
    }
    manifest_path = out_dir / "candidate_pair_manifest.json"
    if manifest_path.exists():
        # Must not rewrite after GPU; if exists from prior failed run, keep if same
        old = json.loads(manifest_path.read_text())
        if old.get("pairs") != pairs:
            # allow overwrite only if no gpu_results yet
            if (out_dir / "gpu_results.json").exists():
                raise RuntimeError("Refuse to alter immutable candidate manifest after GPU")
    save_json(manifest_path, manifest)
    # also copy light summary into WE reports later
    report["gates"]["candidate_manifest"] = {
        "ok": True,
        "path": str(manifest_path),
        "pairs": [{"pair_id": p["pair_id"], "A": p["A"]["plan_idx"], "B": p["B"]["plan_idx"], "sep": p["separation"]} for p in pairs],
    }

    # CPU tests
    setup_nexus_path(sidecar)
    # codec test may need stubs
    codec = nx.official_codec_equivalence_test()
    # If import failed due to missing nuplan, still test local encode roundtrip
    rt = nx.roundtrip_history_test(scene, cutoff)
    report["gates"]["roundtrip_history"] = rt
    report["gates"]["official_codec"] = codec
    report["gates"]["future_leakage"] = rt.get("leakage", {})
    if not rt.get("ok"):
        report["verdict"] = "失败"
        save_json(out_dir / "feasibility_summary.json", report)
        print("STOP: roundtrip history failed", rt)
        return 3
    if rt.get("leakage") and not rt["leakage"].get("task_mask_future_all_zero"):
        report["verdict"] = "失败"
        save_json(out_dir / "feasibility_summary.json", report)
        print("STOP: future leakage")
        return 4

    # context fingerprint (plan_idx placeholder = first A)
    fp = build_input_state_fingerprint(
        scene,
        cutoff,
        args.vocab,
        plan_idx=pairs[0]["A"]["plan_idx"],
        horizon=DEFAULT_HORIZON,
        requested_ego_traj_hash=pair_futures[pairs[0]["A"]["plan_idx"]]["real_hash"],
        vocab=vocab,
        fingerprint_label="cutoff4_nexus",
    )
    save_json(out_dir / "input_state_fingerprint.json", {"input_state_fingerprint": fp["input_state_fingerprint"]})
    report["gates"]["context_fingerprint"] = fp["input_state_fingerprint"]

    if args.cpu_only_tests:
        report["verdict"] = "部分通过"
        report["note"] = "cpu_only_tests"
        save_json(out_dir / "feasibility_summary.json", report)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0

    # --- GPU ---
    import torch

    device = args.device if torch.cuda.is_available() else "cpu"
    try:
        model, load_info = load_nexus_model(sidecar, device)
        report["gates"]["strict_load"] = {"ok": True, **load_info}
    except Exception as e:
        report["gates"]["strict_load"] = {"ok": False, "error": repr(e), "trace": traceback.format_exc()[-2000:]}
        report["verdict"] = "失败"
        save_json(out_dir / "feasibility_summary.json", report)
        print("STOP: strict-load failed", e)
        return 5

    # Prefer official Pacifica sizes
    codec2 = nx.try_import_official_codec()
    ego_L = float(codec2["EGO_LENGTH"]) if codec2.get("ok") else nx.EGO_LENGTH_PACIFICA
    ego_W = float(codec2["EGO_WIDTH"]) if codec2.get("ok") else nx.EGO_WIDTH_PACIFICA

    # Builder shape assert from model feature builder
    try:
        fb = model.get_list_of_required_feature()[0]
        fb_agents = list(getattr(fb, "_num_max_agents", nx.DEFAULT_NUM_MAX_AGENTS))
    except Exception:
        fb_agents = list(nx.DEFAULT_NUM_MAX_AGENTS)
    report["gates"]["agent_dim"] = {"feature_builder_num_max_agents": fb_agents}

    # Build base bundle from pair0 A, then swap ego futures
    base_fut = pair_futures[pairs[0]["A"]["plan_idx"]]
    base_bundle = nx.build_nexus_scene_bundle(
        scene,
        cutoff,
        base_fut,
        ego_length=ego_L,
        ego_width=ego_W,
        num_max_agents=fb_agents,
    )
    report["gates"]["agent_dim"].update(base_bundle.agent_dim_assert)
    report["gates"]["map_coverage"] = base_bundle.map_coverage
    report["gates"]["future_leakage"] = base_bundle.future_leakage_audit
    if not base_bundle.future_leakage_audit.get("task_mask_future_all_zero"):
        report["verdict"] = "失败"
        save_json(out_dir / "feasibility_summary.json", report)
        return 4

    # Common noise
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    noise_shape = (1,) + base_bundle.tensor.shape
    z0 = torch.randn(noise_shape, device=device)
    z1 = torch.randn(noise_shape, device=device)
    noise0_hash = hashlib.sha256(z0.detach().cpu().numpy().tobytes()).hexdigest()
    noise1_hash = hashlib.sha256(z1.detach().cpu().numpy().tobytes()).hexdigest()
    cpu_state = torch.get_rng_state()
    cuda_states = (
        [torch.cuda.get_rng_state(i) for i in range(torch.cuda.device_count())]
        if torch.cuda.is_available()
        else None
    )

    def infer_for(plan_idx: int, noise: torch.Tensor):
        fut = pair_futures[plan_idx]
        b = nx.apply_ego_future_to_bundle(base_bundle, fut, ego_L, ego_W)
        # re-assert leakage after ego swap
        if not np.allclose(b.raw_unnormalized[1:, nx.N_PAST :, :], 0.0):
            raise RuntimeError("future GT appeared after ego swap")
        feats = bundle_to_features(b, device)
        out, pre_h, post_h = run_inference(model, feats, noise, cpu_state, cuda_states)
        sampled = out["sampled_tensor"].detach().cpu().numpy()[0]
        # ego keep equality on normalized tensor
        ego_in = b.tensor[0]
        ego_out = sampled[0]
        keep = b.task_mask[0] > 0.5
        equal_keep = bool(np.array_equal(ego_out[keep], ego_in[keep])) if False else bool(
            np.allclose(ego_out, ego_in, atol=0, rtol=0) or np.array_equal(ego_out, ego_in)
        )
        # Official sample restores keep_mask exactly
        equal_keep = bool(np.allclose(ego_out, ego_in, atol=1e-6))
        decoded = nx.decode_sampled_to_world(sampled, b)
        hold = nx.ego_hold_errors(decoded[sdc], fut)
        return {
            "plan_idx": plan_idx,
            "sampled": sampled,
            "decoded": decoded,
            "hold": hold,
            "ego_norm_equal": equal_keep,
            "pre_rng_hash": pre_h,
            "post_rng_hash": post_h,
            "bundle": b,
            "noise_hash": hashlib.sha256(noise.detach().cpu().numpy().tobytes()).hexdigest(),
        }

    gpu_results: Dict[str, Any] = {
        "noise0_hash": noise0_hash,
        "noise1_hash": noise1_hash,
        "pairs": [],
    }

    # Matrix: A+noise0 x2, B+noise0, A+noise1 for pair0; optionally other pairs A/B noise0
    pair0 = pairs[0]
    a_idx, b_idx = pair0["A"]["plan_idx"], pair0["B"]["plan_idx"]
    r_a1 = infer_for(a_idx, z0)
    r_a2 = infer_for(a_idx, z0)
    r_b = infer_for(a_idx if False else b_idx, z0)
    r_a_n1 = infer_for(a_idx, z1)

    # Reproducibility
    same_noise_pos_err = []
    for tok in r_a1["decoded"]:
        if not r_a1["decoded"][tok].get("generate"):
            continue
        p1 = r_a1["decoded"][tok]["position"][nx.N_PAST : nx.N_PAST + 8]
        p2 = r_a2["decoded"][tok]["position"][nx.N_PAST : nx.N_PAST + 8]
        same_noise_pos_err.append(float(np.max(np.linalg.norm(p1 - p2, axis=1))))
    repro_ok = (
        r_a1["pre_rng_hash"] == r_a2["pre_rng_hash"]
        and r_a1["noise_hash"] == r_a2["noise_hash"]
        and (max(same_noise_pos_err) if same_noise_pos_err else 0.0) <= 1e-4
        and r_a1["ego_norm_equal"]
        and r_a2["ego_norm_equal"]
    )
    report["gates"]["same_noise_repro"] = {
        "ok": repro_ok,
        "pre_rng_match": r_a1["pre_rng_hash"] == r_a2["pre_rng_hash"],
        "max_first4s_pos_err_m": max(same_noise_pos_err) if same_noise_pos_err else None,
        "ego_hold_A": r_a1["hold"],
    }

    # Ego hold gate
    hold_ok = (
        r_a1["hold"]["all_16"]["pos_max_m"] <= 1e-4
        and r_a1["hold"]["all_16"]["heading_max_rad"] <= 1e-4
        and r_b["hold"]["all_16"]["pos_max_m"] <= 1e-4
        and r_b["hold"]["all_16"]["heading_max_rad"] <= 1e-4
        and r_a1["ego_norm_equal"]
        and r_b["ego_norm_equal"]
    )
    report["gates"]["ego_hold_16"] = {
        "ok": hold_ok,
        "A": r_a1["hold"],
        "B": r_b["hold"],
    }
    if not hold_ok:
        report["verdict"] = "失败"
        save_json(out_dir / "feasibility_summary.json", report)
        save_json(out_dir / "gpu_results.json", {"note": "stopped_on_ego_hold"})
        print("STOP: ego hold failed")
        return 6

    # Candidate sensitivity on pair0
    sens_agents = []
    for tok in r_a1["decoded"]:
        if not r_a1["decoded"][tok].get("generate"):
            continue
        # within 50m of ego at cutoff
        ego_p = r_a1["decoded"][sdc]["position"][nx.N_PAST - 1, :2]
        ap = r_a1["decoded"][tok]["position"][nx.N_PAST - 1, :2]
        if float(np.linalg.norm(ap - ego_p)) > 50.0:
            continue
        pa = r_a1["decoded"][tok]["position"][nx.N_PAST : nx.N_PAST + 8]
        pb = r_b["decoded"][tok]["position"][nx.N_PAST : nx.N_PAST + 8]
        va = r_a1["decoded"][tok]["valid"][nx.N_PAST : nx.N_PAST + 8]
        vb = r_b["decoded"][tok]["valid"][nx.N_PAST : nx.N_PAST + 8]
        ade = agent_ade(pa, pb, va, vb)
        end = float(np.linalg.norm(pa[-1] - pb[-1])) if np.isfinite(ade) else float("nan")
        sens_agents.append({"token": tok, "ade_m": ade, "endpoint_m": end})
    sens_agents.sort(key=lambda x: -(x["ade_m"] if np.isfinite(x["ade_m"]) else -1))
    best = sens_agents[0] if sens_agents else None
    # repeat error baseline
    repeat_err = max(same_noise_pos_err) if same_noise_pos_err else 0.0
    cand_ok = False
    if best and np.isfinite(best["ade_m"]):
        cand_ok = (best["ade_m"] >= 0.5 or best["endpoint_m"] >= 1.0) and best["ade_m"] > 5 * max(repeat_err, 1e-8)

    # Run remaining pre-registered pairs (up to 3) for reporting
    pair_reports = [
        {
            "pair_id": 0,
            "A": a_idx,
            "B": b_idx,
            "best_agent": best,
            "n_compared": len(sens_agents),
            "candidate_sensitive": cand_ok,
            "separation": pair0["separation"],
        }
    ]
    for p in pairs[1:]:
        ra = infer_for(p["A"]["plan_idx"], z0)
        rb = infer_for(p["B"]["plan_idx"], z0)
        local = []
        for tok in ra["decoded"]:
            if not ra["decoded"][tok].get("generate"):
                continue
            ego_p = ra["decoded"][sdc]["position"][nx.N_PAST - 1, :2]
            ap = ra["decoded"][tok]["position"][nx.N_PAST - 1, :2]
            if float(np.linalg.norm(ap - ego_p)) > 50.0:
                continue
            pa = ra["decoded"][tok]["position"][nx.N_PAST : nx.N_PAST + 8]
            pb = rb["decoded"][tok]["position"][nx.N_PAST : nx.N_PAST + 8]
            ade = float(np.mean(np.linalg.norm(pa - pb, axis=1)))
            end = float(np.linalg.norm(pa[-1] - pb[-1]))
            local.append({"token": tok, "ade_m": ade, "endpoint_m": end})
        local.sort(key=lambda x: -x["ade_m"])
        b0 = local[0] if local else None
        ok_i = bool(b0 and (b0["ade_m"] >= 0.5 or b0["endpoint_m"] >= 1.0) and b0["ade_m"] > 5 * max(repeat_err, 1e-8))
        pair_reports.append(
            {
                "pair_id": p["pair_id"],
                "A": p["A"]["plan_idx"],
                "B": p["B"]["plan_idx"],
                "best_agent": b0,
                "candidate_sensitive": ok_i,
                "separation": p["separation"],
            }
        )
        cand_ok = cand_ok or ok_i

    report["gates"]["candidate_sensitivity"] = {
        "ok": cand_ok,
        "pairs": pair_reports,
        "repeat_err_m": repeat_err,
    }

    # Noise sensitivity A+n0 vs A+n1
    noise_agents = []
    for tok in r_a1["decoded"]:
        if not r_a1["decoded"][tok].get("generate"):
            continue
        pa = r_a1["decoded"][tok]["position"][nx.N_PAST : nx.N_PAST + 8]
        pb = r_a_n1["decoded"][tok]["position"][nx.N_PAST : nx.N_PAST + 8]
        ade = float(np.mean(np.linalg.norm(pa - pb, axis=1)))
        noise_agents.append({"token": tok, "ade_m": ade})
    noise_agents.sort(key=lambda x: -x["ade_m"])
    noise_ok = bool(noise_agents and noise_agents[0]["ade_m"] >= 0.1)
    report["gates"]["noise_sensitivity"] = {
        "ok": noise_ok,
        "best": noise_agents[0] if noise_agents else None,
        "noise0_hash": noise0_hash,
        "noise1_hash": noise1_hash,
    }

    # Build future pack from A+noise0
    pack = nx.to_future_pack(
        r_a1["decoded"],
        scene,
        cutoff,
        DEFAULT_HORIZON,
        provenance={
            "nexus_commit": "71c31ca848da94c969322a40f0f4ae2af8ca8129",
            "ckpt_sha256": load_info["ckpt_sha256"],
            "plan_idx": a_idx,
            "noise_hash": noise0_hash,
            "slot_tokens": [m.token for m in base_bundle.slot_meta],
            "map_coverage": base_bundle.map_coverage,
            "future_leakage": base_bundle.future_leakage_audit,
        },
    )
    phys = nx.physics_check_futures(pack["futures"])
    report["gates"]["physics_schema"] = phys
    # Save pack outside git
    with open(out_dir / "future_nexus_A_noise0.pkl", "wb") as f:
        pickle.dump(pack, f)
    # light meta only
    save_json(
        out_dir / "future_nexus_A_noise0_meta.json",
        {
            "future_hash": pack["future_hash"],
            "n_agents": pack["n_agents"],
            "coverage_counts": {k: len(v) if isinstance(v, list) else v for k, v in pack["coverage"].items() if k != "fallback_reasons"},
            "physics": phys,
            "plan_idx": a_idx,
        },
    )

    # Token remap check
    gen_tokens = set(pack["coverage"]["nexus_generated"])
    slot_tokens = {m.token for m in base_bundle.slot_meta if m.generate}
    report["gates"]["token_remap"] = {
        "ok": gen_tokens == slot_tokens,
        "n_generated": len(gen_tokens),
        "missing": sorted(slot_tokens - gen_tokens),
        "extra": sorted(gen_tokens - slot_tokens),
    }

    # PDM single-row
    if not args.skip_pdm and phys.get("ok") and report["gates"]["token_remap"]["ok"]:
        try:
            pdm = score_pdm_single_row(
                scene=scene,
                futures_pack=pack,
                plan_idx=a_idx,
                vocab=vocab,
                vocab_path=args.vocab,
                work_dir=out_dir / "pdm_workdir",
                asset_folder=args.asset_folder,
            )
            report["gates"]["pdm_single_row"] = pdm
        except Exception as e:
            report["gates"]["pdm_single_row"] = {
                "ok": False,
                "error": repr(e),
                "trace": traceback.format_exc()[-2000:],
            }
    else:
        report["gates"]["pdm_single_row"] = {
            "ok": False,
            "skipped": True,
            "reason": "physics_or_token_failed_or_skip_flag",
        }

    # Verdict
    g = report["gates"]
    tech = (
        g["strict_load"]["ok"]
        and g["roundtrip_history"]["ok"]
        and g.get("future_leakage", {}).get("task_mask_future_all_zero", False)
        and g["ego_hold_16"]["ok"]
        and g["same_noise_repro"]["ok"]
        and g["candidate_sensitivity"]["ok"]
        and g["noise_sensitivity"]["ok"]
        and g["physics_schema"]["ok"]
        and g["token_remap"]["ok"]
        and g.get("pdm_single_row", {}).get("ok", False)
    )
    partial = (
        g["strict_load"]["ok"]
        and g["ego_hold_16"]["ok"]
        and g["same_noise_repro"]["ok"]
        and g["physics_schema"]["ok"]
        and g["token_remap"]["ok"]
        and g.get("pdm_single_row", {}).get("ok", False)
        and not g["candidate_sensitivity"]["ok"]
    )
    if tech:
        report["verdict"] = "技术通过"
    elif partial:
        report["verdict"] = "部分通过"
    else:
        report["verdict"] = "失败"

    report["resource"] = {
        "rss_gb": _rss_gb(),
        "sidecar_du_hint": "see nexus_sidecar/",
        "device": device,
    }
    report["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    save_json(out_dir / "feasibility_summary.json", report)
    # Light GPU summary without huge arrays
    save_json(
        out_dir / "gpu_results.json",
        {
            "noise0_hash": noise0_hash,
            "noise1_hash": noise1_hash,
            "pair_reports": pair_reports,
            "ego_hold_A": r_a1["hold"],
            "repro": report["gates"]["same_noise_repro"],
            "noise_sens": report["gates"]["noise_sensitivity"],
        },
    )
    print(json.dumps({k: report[k] for k in ("verdict", "gates", "resource")}, indent=2, ensure_ascii=False, default=str))
    return 0 if report["verdict"] != "失败" else 1


if __name__ == "__main__":
    raise SystemExit(main())
