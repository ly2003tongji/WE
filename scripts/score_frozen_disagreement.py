#!/usr/bin/env python3
"""Score the same 8192 candidates under Replay vs IDM frozen futures via DenseRewardManager path."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from frozen_state_lib import (  # noqa: E402
    DEFAULT_CUTOFF,
    DEFAULT_HORIZON,
    build_input_state_fingerprint,
    extract_scorer_config_from_hydra,
    inject_agent_futures_into_scene,
    load_scene_dict,
    resolve_plan_idx,
    save_json,
    scene_as_dict,
    sha256_bytes,
    sha256_file,
)

PDM_KEYS = [
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "lane_keeping",
    "ego_progress",
    "time_to_collision_within_bound",
    "comfort",
    "driving_direction_compliance",
    "score",
]


def _rss_gb() -> float:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / (1024.0 ** 2)
    except Exception:
        pass
    return -1.0


def summarize_reward_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k in PDM_KEYS:
        if k not in d:
            continue
        arr = np.asarray(d[k])
        out[k] = {
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "min": float(np.min(arr.astype(np.float64))),
            "max": float(np.max(arr.astype(np.float64))),
            "mean": float(np.mean(arr.astype(np.float64))),
            "sha256": sha256_bytes(np.ascontiguousarray(arr).tobytes()),
        }
    return out


def score_one_source(
    scene: Dict[str, Any],
    futures_pack: Dict[str, Any],
    source_name: str,
    cutoff: int,
    horizon: int,
    work_dir: Path,
    asset_folder: str,
    vocab_path: Path,
    vocab: np.ndarray,
    plan_idx: int,
    requested_ego_hash: str,
    wall_limit_sec: int,
    rss_limit_gb: float,
    t0: float,
) -> Dict[str, Any]:
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from worldengine.envs.build_env import build_env
    from worldengine.engine.engine_utils import close_engine, engine_initialized

    if time.time() - t0 > wall_limit_sec:
        raise RuntimeError("HARD_PAUSE: wall clock exceeded before scoring")
    if _rss_gb() > rss_limit_gb:
        raise RuntimeError(f"HARD_PAUSE: RSS {_rss_gb():.1f}GB")

    # Independent fingerprint BEFORE future injection (non-future inputs only)
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    max_step = cutoff + 2
    out_root = work_dir / "sim_out"
    simengine_root = Path(os.environ["SIMENGINE_ROOT"])
    if engine_initialized():
        close_engine()
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(simengine_root / "worldengine/configs"), version_base="1.2"):
        cfg = compose(
            config_name="default_runner",
            overrides=[
                "debug_mode=True",
                "with_render_manager=false",
                "with_data_manager=false",
                "with_metric_manager=false",
                "with_dense_reward_manager=true",
                "agent_policy=trajectory_policy",
                "agent_navigation=trajectory_navigation",
                "ego_policy=trajectory_policy",
                "ego_navigation=trajectory_navigation",
                "ego_controller=log_play_controller",
                "use_planner_actions=false",
                "distributed_mode=SINGLE_NODE",
                f"data_file_folder_path={work_dir}",
                f"asset_folder_path={asset_folder}",
                "data_pkl_file_name=all_scenarios.pkl",
                f"output_dir={out_root}",
                f"job_name=frozen_score_{source_name}",
                f"max_step={max_step}",
                "enable_resume=false",
                "exit_on_failure=true",
            ],
        )
    hydra_scorer = extract_scorer_config_from_hydra(cfg)

    fp_pre = build_input_state_fingerprint(
        scene,
        cutoff,
        vocab_path,
        plan_idx,
        scorer_config=hydra_scorer,
        horizon=horizon,
        requested_ego_traj_hash=requested_ego_hash,
        vocab=vocab,
        fingerprint_label=f"score_{source_name}_pre_inject",
    )

    injected = inject_agent_futures_into_scene(scene, futures_pack, cutoff)
    pkl_path = work_dir / "all_scenarios.pkl"
    with pkl_path.open("wb") as f:
        pickle.dump(scene_as_dict(injected), f)

    data = pickle.load(open(pkl_path, "rb"))
    env = build_env(cfg, name=f"frozen_score_{source_name}", data=data)
    reports = env.run()
    close_engine()

    if not reports or not reports[0].succeeded:
        raise RuntimeError(f"scoring run failed for {source_name}: {reports}")

    pdms_dir = out_root / "openscene_format" / "pdms_pkl"
    target = pdms_dir / f"{scene['id']}_step_{cutoff}_scores.pkl"
    if not target.exists():
        cands = sorted(pdms_dir.glob(f"*_step_{cutoff}_scores.pkl"))
        if not cands:
            raise FileNotFoundError(f"No step_{cutoff} scores under {pdms_dir}")
        target = cands[0]

    with target.open("rb") as f:
        scores = pickle.load(f)

    scores_out = work_dir.parent / f"scores_{source_name}.pkl"
    with scores_out.open("wb") as f:
        pickle.dump(scores, f)

    ego_cond = futures_pack.get("ego_conditioning", {})
    provenance = {
        "input_state_fingerprint": fp_pre["input_state_fingerprint"],
        "fingerprint_label": f"score_{source_name}_pre_inject",
        "source": source_name,
        "source_future_hash": futures_pack.get("future_hash"),
        "requested_ego_conditioning_hash": ego_cond.get("requested_ego_conditioning_hash")
        or ego_cond.get("ego_conditioning_hash"),
        "executed_ego_conditioning_hash": ego_cond.get("executed_ego_conditioning_hash"),
        "agent_coverage": futures_pack.get("coverage"),
        "vocabulary_sha256": sha256_file(vocab_path),
        "scorer_config": hydra_scorer,
        "scorer_config_hash": hydra_scorer.get("scorer_config_hash"),
        "cutoff": cutoff,
        "horizon": horizon,
        "scores_path": str(scores_out),
        "reward_summary": summarize_reward_dict(scores),
        "reaction_mode": futures_pack.get(
            "reaction_mode", "source_conditioned_not_candidate_conditioned"
        ),
        "wall_s": round(time.time() - t0, 3),
        "rss_gb": round(_rss_gb(), 3),
    }
    save_json(work_dir.parent / f"scores_{source_name}_provenance.json", provenance)
    save_json(work_dir.parent / f"fingerprint_score_{source_name}.json", {
        "input_state_fingerprint": fp_pre["input_state_fingerprint"],
        "label": provenance["fingerprint_label"],
        "scorer_config_hash": hydra_scorer.get("scorer_config_hash"),
    })
    return provenance


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scene-pkl",
        type=Path,
        default=Path(
            "/mnt/cpfs/prediction/lyyy/myself/WE/data/smoke_1scene/scenarios/original/navtest_failures/all_scenarios.pkl"
        ),
    )
    parser.add_argument(
        "--futures-dir",
        type=Path,
        default=Path("/mnt/cpfs/prediction/lyyy/myself/WE/data/frozen_paired/smoke1_cutoff3_validity_v2"),
    )
    parser.add_argument(
        "--vocab",
        type=Path,
        default=Path("/mnt/cpfs/prediction/lyyy/myself/WE/data/hf/data/alg_engine/test_8192_kmeans.npy"),
    )
    parser.add_argument("--cutoff", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--plan-idx", type=int, default=None)
    parser.add_argument("--plan-idx-csv", type=Path, default=None)
    parser.add_argument("--plan-idx-step", type=int, default=None)
    parser.add_argument("--asset-folder", type=str, default=None)
    parser.add_argument("--wall-limit-sec", type=int, default=3600)
    parser.add_argument("--rss-limit-gb", type=float, default=64.0)
    args = parser.parse_args()

    t0 = time.time()
    we_root = Path(
        os.environ.get(
            "WORLDENGINE_ROOT",
            "/mnt/cpfs/prediction/lyyy/myself/WE/WE/upstream/WorldEngine",
        )
    )
    os.environ.setdefault("WORLDENGINE_ROOT", str(we_root))
    os.environ.setdefault("SIMENGINE_ROOT", str(we_root / "projects/SimEngine"))
    os.environ.setdefault("NUPLAN_MAPS_ROOT", str(we_root / "data/raw/nuplan/dataset/maps"))
    sys.path.insert(0, os.environ["SIMENGINE_ROOT"])

    asset_folder = args.asset_folder or str(
        we_root / "data/sim_engine/assets/navtest_failures/assets"
    )

    fp_path = args.futures_dir / "input_state_fingerprint.json"
    fp_doc = json.loads(fp_path.read_text())
    cutoff = args.cutoff if args.cutoff is not None else int(fp_doc.get("cutoff", DEFAULT_CUTOFF))
    plan_res = resolve_plan_idx(
        args.plan_idx if args.plan_idx is not None else fp_doc.get("plan_idx_resolution", {}).get("plan_idx"),
        args.plan_idx_csv,
        args.plan_idx_step,
        cutoff,
    )
    plan_idx = int(plan_res["plan_idx"])
    requested_ego_hash = fp_doc.get("requested_ego_conditioning_hash")
    if not requested_ego_hash:
        raise RuntimeError("missing requested_ego_conditioning_hash in input_state_fingerprint.json")

    scene = load_scene_dict(args.scene_pkl)
    vocab = np.load(args.vocab)

    provenances = {}
    score_fps = {}
    for source, pkl_name in (
        ("log_replay", "future_log_replay.pkl"),
        ("idm", "future_idm.pkl"),
    ):
        with (args.futures_dir / pkl_name).open("rb") as f:
            futures_pack = pickle.load(f)
        try:
            prov = score_one_source(
                scene=scene,
                futures_pack=futures_pack,
                source_name=source,
                cutoff=cutoff,
                horizon=args.horizon,
                work_dir=args.futures_dir / f"score_workdir_{source}",
                asset_folder=asset_folder,
                vocab_path=args.vocab,
                vocab=vocab,
                plan_idx=plan_idx,
                requested_ego_hash=requested_ego_hash,
                wall_limit_sec=args.wall_limit_sec,
                rss_limit_gb=args.rss_limit_gb,
                t0=t0,
            )
            provenances[source] = prov
            score_fps[source] = prov["input_state_fingerprint"]
            print(f"scored {source}: score_sha={prov['reward_summary']['score']['sha256'][:16]} fp={prov['input_state_fingerprint'][:16]}")
        except Exception as e:
            save_json(
                args.futures_dir / f"score_{source}_error.json",
                {"error": str(e), "trace": traceback.format_exc()},
            )
            print("score failed", source, e)
            print(traceback.format_exc())
            return 2

    if score_fps["log_replay"] != score_fps["idm"]:
        save_json(args.futures_dir / "HARD_PAUSE_fingerprint_mismatch.json", score_fps)
        print("HARD_PAUSE: score-stage independent fingerprints differ")
        print(score_fps)
        return 3

    # Merge four-way report (2 build + 2 score)
    build_fps = fp_doc.get("independent_fingerprints", {})
    four_way = {
        "build_replay": build_fps.get("replay_future_build_pre_hydra_scorer"),
        "build_idm": build_fps.get("idm_env_init_pre_hydra_scorer"),
        "score_replay_pre_inject": score_fps["log_replay"],
        "score_idm_pre_inject": score_fps["idm"],
    }
    # Note: build fingerprints may differ from score because densereward=true changes hydra scorer block.
    # Within each stage (build-only / score-only) they must match.
    four_way["build_pair_equal"] = four_way["build_replay"] == four_way["build_idm"]
    four_way["score_pair_equal"] = four_way["score_replay_pre_inject"] == four_way["score_idm_pre_inject"]
    four_way["all_four_equal"] = len(set(v for k, v in four_way.items() if k.startswith(("build_", "score_")) and isinstance(v, str))) == 1
    four_way["note"] = (
        "Build-stage Hydra has with_dense_reward_manager=false; score-stage has true. "
        "Expect build fingerprint != score fingerprint when scorer_config includes that flag; "
        "Replay/IDM pairs within each stage must match."
    )
    save_json(args.futures_dir / "independent_fingerprints_all.json", four_way)

    if not four_way["score_pair_equal"] or not four_way["build_pair_equal"]:
        print("HARD_PAUSE: within-stage fingerprint mismatch")
        return 3

    save_json(
        args.futures_dir / "score_summary.json",
        {
            "status": "ok",
            "input_state_fingerprint_score_stage": score_fps["log_replay"],
            "independent_fingerprints": four_way,
            "sources": provenances,
            "wall_s": round(time.time() - t0, 3),
            "rss_gb": round(_rss_gb(), 3),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
