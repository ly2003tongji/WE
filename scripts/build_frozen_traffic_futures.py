#!/usr/bin/env python3
"""Build frozen-state Replay and IDM traffic futures (no 3DGS, no upstream edits)."""

from __future__ import annotations

import argparse
import copy
import os
import pickle
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from frozen_state_lib import (  # noqa: E402
    DEFAULT_CUTOFF,
    DEFAULT_HORIZON,
    DEFAULT_PLAN_IDX,
    DEFAULT_SEED,
    SCORER_CONFIG,
    build_input_state_fingerprint,
    compare_futures,
    extract_replay_futures,
    futures_to_numpy_bundle,
    inject_ego_conditioning,
    save_json,
    sha256_bytes,
    sha256_file,
    slice_scene_from_cutoff,
    load_scene_dict,
    scene_as_dict,
)


def _rss_gb() -> float:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    kb = float(line.split()[1])
                    return kb / (1024.0 ** 2)
    except Exception:
        pass
    return -1.0


def run_idm_futures(
    orig_scene: Dict[str, Any],
    cutoff: int,
    horizon: int,
    plan_idx: int,
    vocab: np.ndarray,
    work_dir: Path,
    asset_folder: str,
    seed: int,
) -> Dict[str, Any]:
    """Roll IDM from frozen cutoff with fixed ego conditioning; no render."""
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from worldengine.envs.build_env import build_env
    from worldengine.engine.engine_utils import close_engine, engine_initialized

    truncated = slice_scene_from_cutoff(orig_scene, cutoff, horizon)
    ego_meta = inject_ego_conditioning(truncated, plan_idx, vocab)

    work_dir.mkdir(parents=True, exist_ok=True)
    pkl_path = work_dir / "all_scenarios.pkl"
    with pkl_path.open("wb") as f:
        pickle.dump(scene_as_dict(truncated), f)

    if engine_initialized():
        close_engine()

    simengine_root = Path(os.environ["SIMENGINE_ROOT"])
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(simengine_root / "worldengine/configs"), version_base="1.2"):
        cfg = compose(
            config_name="default_runner",
            overrides=[
                "debug_mode=True",
                "with_render_manager=false",
                "with_data_manager=false",
                "with_metric_manager=false",
                "with_dense_reward_manager=false",
                "agent_policy=idm_policy",
                "agent_navigation=idm_navigation",
                "ego_policy=trajectory_policy",
                "ego_navigation=trajectory_navigation",
                "ego_controller=log_play_controller",
                "use_planner_actions=false",
                "distributed_mode=SINGLE_NODE",
                f"data_file_folder_path={work_dir}",
                f"asset_folder_path={asset_folder}",
                "data_pkl_file_name=all_scenarios.pkl",
                f"output_dir={work_dir / 'out'}",
                "job_name=frozen_idm_future",
                f"max_step={horizon}",
                "enable_resume=false",
                "exit_on_failure=true",
            ],
        )

    data = pickle.load(open(pkl_path, "rb"))
    env = build_env(cfg, name="frozen_idm_future", data=data)
    env.reset(seed=seed)

    records: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    policy_by_token: Dict[str, Optional[str]] = {}

    def snap(step: int) -> None:
        am = env.engine.agent_manager
        for aid, ag in am.all_agents.items():
            pos = np.asarray(ag.current_position, dtype=np.float64).reshape(-1)
            vel = np.asarray(ag.current_velocity, dtype=np.float64).reshape(-1)
            pname = (
                type(getattr(ag, "policy", None)).__name__
                if getattr(ag, "policy", None) is not None
                else None
            )
            policy_by_token[aid] = pname
            records[aid].append(
                {
                    "step": int(step),
                    "pos": pos.copy(),
                    "heading": float(ag.current_heading),
                    "vel": vel.copy(),
                    "policy": pname,
                }
            )

    snap(0)
    for _ in range(horizon - 1):
        env.step(None)
        snap(int(env.engine.episode_step))

    # Build futures aligned to original tokens.
    # WorldEngine IDM only simulates VEHICLE (ped/cyclist skipped at spawn).
    replay_ref = extract_replay_futures(orig_scene, cutoff, horizon)
    futures: Dict[str, Any] = {}
    coverage = {
        "idm_rolled": [],
        "static_trajectory_follow": [],
        "log_fallback": [],
        "invalid_at_cutoff": [],
        "missing_live_agent": [],
    }
    fallback_reasons: Dict[str, str] = {}

    for token, rf in replay_ref["futures"].items():
        ot = orig_scene["object_track"][token]
        typ = ot.get("type")
        if typ != "VEHICLE":
            futures[token] = {**rf, "source": "log_fallback_non_vehicle"}
            coverage["log_fallback"].append(token)
            fallback_reasons[token] = "non_vehicle_not_spawned_by_agent_manager"
            continue
        if token not in records:
            futures[token] = {**rf, "source": "log_fallback_missing_live"}
            coverage["missing_live_agent"].append(token)
            coverage["log_fallback"].append(token)
            fallback_reasons[token] = "vehicle_not_in_live_agents"
            continue
        seq = sorted(records[token], key=lambda x: x["step"])
        # Ensure length == horizon; pad with last if short
        pos = np.zeros((horizon, 3), dtype=np.float64)
        heading = np.zeros(horizon, dtype=np.float64)
        vel = np.zeros((horizon, 2), dtype=np.float64)
        valid = np.zeros(horizon, dtype=np.float64)
        for s in seq:
            i = int(s["step"])
            if i >= horizon:
                continue
            p = s["pos"]
            pos[i, : min(3, len(p))] = p[: min(3, len(p))]
            heading[i] = s["heading"]
            vv = s["vel"]
            if len(vv) >= 2:
                vel[i, :2] = vv[:2]
            else:
                vel[i, 0] = float(vv[0]) if len(vv) else 0.0
            valid[i] = 1.0
        pname = policy_by_token.get(token, "")
        if pname == "IDMPolicy":
            src = "idm"
            coverage["idm_rolled"].append(token)
        else:
            # Static vehicles forced to trajectory_policy inside agent_manager.
            src = "static_trajectory_follow"
            coverage["static_trajectory_follow"].append(token)
        futures[token] = {
            "type": typ,
            "position": pos,
            "heading": heading,
            "velocity": vel,
            "valid": valid,
            "source": src,
            "policy": pname,
        }

    # Also include ego record for provenance only (not scored as agent future)
    ego_seq = records.get("ego", [])

    flat_parts = []
    for token in sorted(futures.keys()):
        f = futures[token]
        flat_parts.append(token.encode("utf-8"))
        flat_parts.append(np.ascontiguousarray(f["position"]).tobytes())
        flat_parts.append(np.ascontiguousarray(f["heading"]).tobytes())
    future_hash = sha256_bytes(b"".join(flat_parts))

    close_engine()

    return {
        "source": "idm",
        "cutoff": cutoff,
        "horizon": horizon,
        "frequency_hz": 2.0,
        "dt_s": 0.5,
        "n_agents": len(futures),
        "futures": futures,
        "coverage": coverage,
        "fallback_reasons": fallback_reasons,
        "future_hash": future_hash,
        "ego_conditioning": ego_meta,
        "ego_live_positions": [s["pos"].tolist() for s in ego_seq],
        "reaction_mode": "source_conditioned_not_candidate_conditioned",
        "seed": seed,
        "notes": [
            "IDM rolls structured vehicle state only; with_render_manager=false.",
            "PEDESTRIAN/CYCLIST are not spawned by BaseAgentManager; log fallback.",
            "Static agents use trajectory_policy internally; treated as log_fallback if not IDM.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scene-pkl",
        type=Path,
        default=Path("/mnt/cpfs/prediction/lyyy/myself/WE/data/smoke_1scene/scenarios/original/navtest_failures/all_scenarios.pkl"),
    )
    parser.add_argument(
        "--vocab",
        type=Path,
        default=Path("/mnt/cpfs/prediction/lyyy/myself/WE/data/hf/data/alg_engine/test_8192_kmeans.npy"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/mnt/cpfs/prediction/lyyy/myself/WE/data/frozen_paired/smoke1_cutoff3"),
    )
    parser.add_argument("--cutoff", type=int, default=DEFAULT_CUTOFF)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--plan-idx", type=int, default=DEFAULT_PLAN_IDX)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--asset-folder",
        type=str,
        default=None,
        help="SimEngine asset folder (defaults to WORLDENGINE_ROOT/.../navtest_failures/assets)",
    )
    parser.add_argument("--wall-limit-sec", type=int, default=3600)
    parser.add_argument("--rss-limit-gb", type=float, default=64.0)
    args = parser.parse_args()

    t0 = time.time()
    we_root = Path(os.environ.get("WORLDENGINE_ROOT", "/mnt/cpfs/prediction/lyyy/myself/WE/WE/upstream/WorldEngine"))
    os.environ.setdefault("WORLDENGINE_ROOT", str(we_root))
    os.environ.setdefault("SIMENGINE_ROOT", str(we_root / "projects/SimEngine"))
    os.environ.setdefault("NUPLAN_MAPS_ROOT", str(we_root / "data/raw/nuplan/dataset/maps"))
    sys.path.insert(0, os.environ["SIMENGINE_ROOT"])

    asset_folder = args.asset_folder or str(we_root / "data/sim_engine/assets/navtest_failures/assets")

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    scene = load_scene_dict(args.scene_pkl)
    vocab = np.load(args.vocab)

    fp_pack = build_input_state_fingerprint(scene, args.cutoff, args.vocab, args.plan_idx, SCORER_CONFIG)
    save_json(out_dir / "input_state_fingerprint.json", {
        "input_state_fingerprint": fp_pack["input_state_fingerprint"],
        "cutoff": args.cutoff,
        "plan_idx": args.plan_idx,
        "vocab_sha256": sha256_file(args.vocab),
        "scorer_config": SCORER_CONFIG,
        "scene_id": scene.get("id"),
        "note": "Fingerprint excludes traffic-model futures (comparison target).",
        "payload_digest_only": True,
        "payload_sha256": sha256_bytes(
            __import__("json").dumps(fp_pack["payload"], sort_keys=True, default=str).encode("utf-8")
        ),
    })
    # Keep full payload separately for debugging (still lightweight relative to arrays)
    save_json(out_dir / "input_state_payload.json", fp_pack["payload"])

    # Replay future (same ego conditioning recorded for provenance)
    ego_tmp = slice_scene_from_cutoff(scene, args.cutoff, args.horizon)
    ego_meta_replay = inject_ego_conditioning(ego_tmp, args.plan_idx, vocab)
    replay = extract_replay_futures(scene, args.cutoff, args.horizon)
    replay["ego_conditioning"] = ego_meta_replay
    replay["reaction_mode"] = "source_conditioned_not_candidate_conditioned"
    replay["note"] = "Replay does not react to ego; same ego conditioning hash recorded for pairing."

    # Hard gate checks
    def gate(msg: str) -> None:
        raise RuntimeError(f"HARD_PAUSE: {msg}")

    if time.time() - t0 > args.wall_limit_sec:
        gate("wall clock exceeded before IDM")
    if _rss_gb() > args.rss_limit_gb:
        gate(f"RSS {_rss_gb():.1f}GB exceeded limit")

    try:
        idm = run_idm_futures(
            scene,
            args.cutoff,
            args.horizon,
            args.plan_idx,
            vocab,
            work_dir=out_dir / "idm_workdir",
            asset_folder=asset_folder,
            seed=args.seed,
        )
    except Exception as e:
        save_json(out_dir / "idm_error.json", {"error": str(e), "trace": traceback.format_exc()})
        print("IDM future generation failed:", e)
        print(traceback.format_exc())
        return 2

    # Both paths must share fingerprint (futures excluded by construction)
    idm["input_state_fingerprint"] = fp_pack["input_state_fingerprint"]
    replay["input_state_fingerprint"] = fp_pack["input_state_fingerprint"]

    if replay["ego_conditioning"]["ego_conditioning_hash"] != idm["ego_conditioning"]["ego_conditioning_hash"]:
        gate("ego conditioning hash mismatch between Replay and IDM paths")

    # Coverage sanity: IDM vehicles should be subset of replay vehicles
    r_cov = set(replay["coverage"]["idm_capable_vehicle"])
    i_rolled = set(idm["coverage"]["idm_rolled"])
    if not i_rolled.issubset(r_cov):
        gate(f"IDM rolled tokens not subset of replay vehicles: {sorted(i_rolled - r_cov)}")

    fut_cmp = compare_futures(replay, idm)

    # Persist futures (npz outside git) + light meta (json)
    for name, pack in (("log_replay", replay), ("idm", idm)):
        meta = {k: v for k, v in pack.items() if k != "futures"}
        # strip large live positions detail if any
        save_json(out_dir / f"future_{name}_meta.json", meta)
        bundle = futures_to_numpy_bundle(pack)
        np.savez_compressed(out_dir / f"future_{name}.npz", **bundle)
        # also pickle futures for exact reinjection
        with (out_dir / f"future_{name}.pkl").open("wb") as f:
            pickle.dump(pack, f)

    save_json(out_dir / "future_compare.json", fut_cmp)

    summary = {
        "status": "ok",
        "wall_s": round(time.time() - t0, 3),
        "rss_gb": round(_rss_gb(), 3),
        "input_state_fingerprint": fp_pack["input_state_fingerprint"],
        "ego_conditioning_hash": replay["ego_conditioning"]["ego_conditioning_hash"],
        "plan_idx": args.plan_idx,
        "replay_future_hash": replay["future_hash"],
        "idm_future_hash": idm["future_hash"],
        "futures_differ": not fut_cmp["futures_equal"],
        "n_changed_agents": fut_cmp["n_changed_agents_maxade_gt_5cm"],
        "traj_dev_mean_m": fut_cmp["traj_dev_mean_m"],
        "traj_dev_max_m": fut_cmp["traj_dev_max_m"],
        "idm_rolled": len(idm["coverage"]["idm_rolled"]),
        "log_fallback": len(idm["coverage"]["log_fallback"]),
        "reaction_mode": "source_conditioned_not_candidate_conditioned",
    }
    save_json(out_dir / "build_summary.json", summary)
    print(json_dumps(summary))
    return 0


def json_dumps(obj: Any) -> str:
    import json

    return json.dumps(obj, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    raise SystemExit(main())
