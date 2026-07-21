#!/usr/bin/env python3
"""Cutoff=4 three-source smoke: Nexus@plan_idx=1333 + pairwise Replay/IDM/Nexus compare.

Phase A (nexus env): generate ego-conditioned Nexus futures (seed=0).
Phase B (simengine env): full 8192 PDM score + pairwise tables / report.

Does NOT modify WorldEngine/Nexus upstream. Does NOT expand train-side.
"""

from __future__ import annotations

import argparse
import copy
import csv
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

from frozen_state_lib import (  # noqa: E402
    DEFAULT_CUTOFF,
    DEFAULT_HORIZON,
    inject_ego_conditioning_at_cutoff,
    load_scene_dict,
    save_json,
    sha256_file,
)
from ranking_metrics import (  # noqa: E402
    analyze_max_score_ties,
    directional_binary_flips,
    index_tie_broken_topk_overlap,
    kendall_tau_b,
)

SCENE_ID = "2021.09.29.15.23.04_veh-28_00601_00802-6326d00e52115da4"
PLAN_IDX_CSV_DEFAULT = (
    WE_ROOT
    / "upstream/WorldEngine/experiments/closed_loop_exps/"
    / "e2e_vadv2_50pct-disagreement-smoke1-NR-20260715/navtest_failures_NR/plan_traj/plan_idx.csv"
)
PDM_KEYS = [
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "time_to_collision_within_bound",
    "comfort",
    "ego_progress",
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
        return -1.0
    return -1.0


def apply_physics_log_fallback(
    pack: Dict[str, Any],
    scene: Dict[str, Any],
    cutoff: int,
    horizon: int,
    physics: Dict[str, Any],
) -> Dict[str, Any]:
    """Replace physics-failing Nexus agents with log futures; record tokens."""
    from adapters.traffic_models import nexus as nx

    fail_tokens = sorted({iss["token"] for iss in physics.get("issues", [])})
    sdc = scene["sdc_id"]
    replaced = []
    for token in fail_tokens:
        if token not in pack["futures"]:
            continue
        if pack["futures"][token].get("source") != "nexus":
            continue
        ot = scene["object_track"][token]
        st = ot["state"]
        pos = np.asarray(st["position"], dtype=np.float64)
        heading = nx._as_1d(np.asarray(st["heading"], dtype=np.float64))
        vel = np.asarray(st["velocity"], dtype=np.float64)
        valid = nx._as_1d(np.asarray(st.get("valid", np.ones(len(pos))), dtype=np.float64))
        end = min(cutoff + horizon, len(pos))
        sl = slice(cutoff, end)
        pack["futures"][token] = {
            "type": ot.get("type"),
            "position": pos[sl].copy(),
            "heading": heading[sl].copy(),
            "velocity": vel[sl].copy() if len(vel) >= end else vel[cutoff:].copy(),
            "valid": valid[sl].copy() if len(valid) >= end else valid[cutoff:].copy(),
            "source": "log_replay_fallback",
        }
        cov = pack.setdefault("coverage", {})
        gen = list(cov.get("nexus_generated", []))
        if token in gen:
            gen.remove(token)
        cov["nexus_generated"] = gen
        fb = list(cov.get("log_fallback", []))
        if token not in fb:
            fb.append(token)
        cov["log_fallback"] = fb
        reasons = dict(cov.get("fallback_reasons", {}))
        reasons[token] = "physics_gate_fail_log_fallback"
        cov["fallback_reasons"] = reasons
        replaced.append(token)

    # re-hash
    parts = []
    for token in sorted(pack["futures"].keys()):
        f = pack["futures"][token]
        parts.append(token.encode())
        parts.append(np.ascontiguousarray(f["position"]).tobytes())
        parts.append(np.ascontiguousarray(f["heading"]).tobytes())
    pack["future_hash"] = hashlib.sha256(b"".join(parts)).hexdigest()
    pack["physics_log_fallback_tokens"] = replaced
    return pack


def phase_a_nexus_generate(args: argparse.Namespace) -> int:
    from adapters.traffic_models import nexus as nx
    from run_nexus_sidecar_smoke import (
        bundle_to_features,
        load_nexus_model,
        run_inference,
        setup_nexus_path,
    )

    sidecar = args.sidecar_root
    out_dir = args.nexus_out
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_nexus_path(sidecar)

    scene = load_scene_dict(args.scene_pkl)
    assert SCENE_ID in str(scene.get("id", SCENE_ID))
    cutoff = int(args.cutoff)
    if cutoff != 4:
        raise SystemExit(f"Nexus adapter requires cutoff=4, got {cutoff}")
    horizon = int(args.horizon)
    plan_idx = int(args.plan_idx)
    seed = int(args.seed)

    vocab = np.load(args.vocab)
    sdc = scene["sdc_id"]
    ego_st = scene["object_track"][sdc]["state"]
    ego_center = np.asarray(ego_st["position"], dtype=np.float64)[cutoff, :2]
    ego_heading = float(np.asarray(ego_st["heading"]).reshape(-1)[cutoff])
    ego_fut = nx.vocab_candidate_to_nexus_ego_future(vocab[plan_idx], ego_center, ego_heading)

    # Leakage / roundtrip gates
    rt = nx.roundtrip_history_test(scene, cutoff)
    if not rt.get("ok"):
        save_json(out_dir / "phase_a_fail.json", {"reason": "roundtrip", "rt": rt})
        return 3
    if not rt.get("leakage", {}).get("task_mask_future_all_zero", False):
        save_json(out_dir / "phase_a_fail.json", {"reason": "future_leakage", "rt": rt})
        return 4

    import torch

    device = args.device if torch.cuda.is_available() else "cpu"
    model, load_info = load_nexus_model(sidecar, device)
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
        save_json(out_dir / "phase_a_fail.json", {"reason": "bundle_leakage", "audit": bundle.future_leakage_audit})
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
            "plan_idx_csv": str(args.plan_idx_csv),
            "plan_idx_step": int(args.plan_idx_step),
            "seed": seed,
            "noise_hash": noise0_hash,
            "slot_tokens": [m.token for m in bundle.slot_meta],
            "map_coverage": bundle.map_coverage,
            "future_leakage": bundle.future_leakage_audit,
            "note": "three_source_smoke; NOT legacy plan_idx=710",
        },
    )
    physics = nx.physics_check_futures(pack["futures"])
    pack = apply_physics_log_fallback(pack, scene, cutoff, horizon, physics)
    physics_after = nx.physics_check_futures(pack["futures"])

    with (out_dir / "future_nexus_plan1333_seed0.pkl").open("wb") as f:
        pickle.dump(pack, f)
    save_json(
        out_dir / "future_nexus_plan1333_seed0_meta.json",
        {
            "cutoff": cutoff,
            "horizon": horizon,
            "plan_idx": plan_idx,
            "plan_idx_csv": str(args.plan_idx_csv),
            "plan_idx_step": int(args.plan_idx_step),
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
        },
    )
    print(
        json.dumps(
            {
                "phase": "A",
                "hold_ok": hold_ok,
                "physics_fallback": pack.get("physics_log_fallback_tokens", []),
                "future_hash": pack["future_hash"],
                "out": str(out_dir),
            },
            indent=2,
        )
    )
    return 0 if hold_ok else 6


def _pairwise(a: Dict[str, Any], b: Dict[str, Any], name_a: str, name_b: str) -> Dict[str, Any]:
    noc = directional_binary_flips(
        a["no_at_fault_collisions"], b["no_at_fault_collisions"], safe_threshold=1.0
    )
    ttc = directional_binary_flips(
        a["time_to_collision_within_bound"],
        b["time_to_collision_within_bound"],
        safe_threshold=1.0,
    )
    score_a = np.asarray(a["score"], dtype=np.float64).reshape(-1)
    score_b = np.asarray(b["score"], dtype=np.float64).reshape(-1)
    ties = analyze_max_score_ties(score_a, score_b)
    tau = kendall_tau_b(score_a, score_b)
    topk = {
        str(k): index_tie_broken_topk_overlap(score_a, score_b, k)
        for k in (1, 10, 50, 100)
    }
    metrics = {}
    for key in PDM_KEYS:
        xa = np.asarray(a[key], dtype=np.float64).reshape(-1)
        xb = np.asarray(b[key], dtype=np.float64).reshape(-1)
        diff = xb - xa
        metrics[key] = {
            "equal": bool(np.array_equal(xa, xb)),
            "max_abs_diff": float(np.max(np.abs(diff))),
            "mean_abs_diff": float(np.mean(np.abs(diff))),
            "n_changed_gt_1e-6": int(np.sum(np.abs(diff) > 1e-6)),
            f"{name_a}_mean": float(np.mean(xa)),
            f"{name_b}_mean": float(np.mean(xb)),
            f"{name_a}_min": float(np.min(xa)),
            f"{name_a}_max": float(np.max(xa)),
            f"{name_b}_min": float(np.min(xb)),
            f"{name_b}_max": float(np.max(xb)),
        }
    return {
        "pair": f"{name_a}_vs_{name_b}",
        "noc_flips": noc,
        "ttc_flips": ttc,
        "max_score_ties": ties,
        "kendall_tau_b": tau,
        "topk_overlap_index_tie_broken": topk,
        "metrics": metrics,
        "score_range": {
            name_a: {"min": float(score_a.min()), "max": float(score_a.max())},
            name_b: {"min": float(score_b.min()), "max": float(score_b.max())},
        },
    }


def phase_b_score_and_compare(args: argparse.Namespace) -> int:
    from score_frozen_disagreement import score_one_source

    we_root = WE_ROOT / "upstream/WorldEngine"
    os.environ.setdefault("WORLDENGINE_ROOT", str(we_root))
    os.environ.setdefault("SIMENGINE_ROOT", str(we_root / "projects/SimEngine"))
    os.environ.setdefault(
        "NUPLAN_MAPS_ROOT",
        os.environ.get(
            "NUPLAN_MAPS_ROOT",
            str(we_root / "data/raw/nuplan/dataset/maps"),
        ),
    )
    if os.environ["SIMENGINE_ROOT"] not in sys.path:
        sys.path.insert(0, os.environ["SIMENGINE_ROOT"])

    out_dir = args.nexus_out
    restore_dir = args.restore_dir
    report_md = args.report_md
    summary_json = args.summary_json
    summary_csv = args.summary_csv

    scene = load_scene_dict(args.scene_pkl)
    vocab = np.load(args.vocab)
    cutoff = int(args.cutoff)
    horizon = int(args.horizon)
    plan_idx = int(args.plan_idx)
    asset_folder = args.asset_folder

    with (out_dir / "future_nexus_plan1333_seed0.pkl").open("rb") as f:
        nexus_pack = pickle.load(f)
    nexus_meta = json.loads((out_dir / "future_nexus_plan1333_seed0_meta.json").read_text())

    # Align ego conditioning hash with restore run if present
    restore_ego = {}
    ego_path = restore_dir / "ego_conditioning_verification_restored.json"
    plan_res_path = restore_dir / "plan_idx_resolution.json"
    if (restore_dir / "future_idm_restored_meta.json").exists():
        restore_meta = json.loads((restore_dir / "future_idm_restored_meta.json").read_text())
        restore_ego = restore_meta.get("ego_conditioning") or {}
    plan_res = json.loads(plan_res_path.read_text()) if plan_res_path.exists() else {}

    scene_scored = copy.deepcopy(scene)
    ego_meta = inject_ego_conditioning_at_cutoff(scene_scored, cutoff, plan_idx, vocab)
    req_hash = ego_meta["requested_ego_conditioning_hash"]
    nexus_pack = dict(nexus_pack)
    nexus_pack["ego_conditioning"] = {
        k: v
        for k, v in ego_meta.items()
        if k not in ("requested_center_traj", "requested_heading_traj")
    }

    t0 = time.time()
    prov = score_one_source(
        scene=scene,
        futures_pack=nexus_pack,
        source_name="nexus",
        cutoff=cutoff,
        horizon=horizon,
        work_dir=out_dir / "score_workdir_nexus",
        asset_folder=asset_folder,
        vocab_path=args.vocab,
        vocab=vocab,
        plan_idx=plan_idx,
        requested_ego_hash=req_hash,
        wall_limit_sec=3600,
        rss_limit_gb=64.0,
        t0=t0,
    )
    with (out_dir / "scores_nexus.pkl").open("rb") as f:
        scores_nexus = pickle.load(f)
    save_json(out_dir / "scores_nexus_run_meta.json", prov)

    with (restore_dir / "scores_log_replay.pkl").open("rb") as f:
        scores_replay = pickle.load(f)
    with (restore_dir / "scores_idm_restored.pkl").open("rb") as f:
        scores_idm = pickle.load(f)

    fp_restore = json.loads((restore_dir / "fingerprint_cross_check.json").read_text())
    fp_replay = fp_restore.get("replay_run_fingerprint")
    fp_idm = fp_restore.get("restored_run_fingerprint")
    fp_nexus = prov["input_state_fingerprint"]
    fingerprint_aligned = bool(fp_replay == fp_idm == fp_nexus)

    pairs = {
        "replay_vs_idm": _pairwise(scores_replay, scores_idm, "replay", "idm"),
        "replay_vs_nexus": _pairwise(scores_replay, scores_nexus, "replay", "nexus"),
        "idm_vs_nexus": _pairwise(scores_idm, scores_nexus, "idm", "nexus"),
    }

    # Lightweight attribution hint: reuse restore attribution if present
    attr = {}
    attr_path = restore_dir / "restored_hybrid_attribution_summary.json"
    if attr_path.exists():
        attr = json.loads(attr_path.read_text())

    restore_summary = {}
    rs_path = restore_dir / "restore_physics_summary.json"
    if rs_path.exists():
        restore_summary = json.loads(rs_path.read_text())

    transition = {}
    tr_path = restore_dir / "pre_cutoff_irreversible_transition_flags.json"
    if tr_path.exists():
        transition = json.loads(tr_path.read_text())
    n_true = sum(1 for v in transition.values() if v is True)

    head_sha = os.popen(f"git -C {WE_ROOT} rev-parse HEAD").read().strip()
    upstream_sha = os.popen(f"git -C {WE_ROOT}/upstream/WorldEngine rev-parse HEAD").read().strip()

    summary = {
        "status": "ok",
        "engineering_smoke_not_research_go_nogo": True,
        "scene_id": SCENE_ID,
        "scene_note": "navtest_failures engineering smoke subset; NOT research go/no-go",
        "cutoff": cutoff,
        "horizon": horizon,
        "forbid_merge_with_cutoff3": True,
        "collaboration_head_sha": head_sha,
        "worldengine_upstream_sha": upstream_sha,
        "plan_idx_provenance": {
            "plan_idx": plan_idx,
            "plan_idx_step": int(args.plan_idx_step),
            "plan_idx_csv": str(args.plan_idx_csv),
            "restore_plan_idx_resolution": plan_res,
            "forbidden_legacy_nexus_plan_idx": 710,
            "note": "Locked to Action Policy plan_idx.csv step=5 → 1333 for all three sources.",
        },
        "fingerprint": {
            "replay": fp_replay,
            "idm_restored": fp_idm,
            "nexus": fp_nexus,
            "aligned": fingerprint_aligned,
            "scorer_num_history_unchanged": 4,
            "research_cutoff": 4,
        },
        "idm_gate": {
            "grade": restore_summary.get("cutoff_gate_grade"),
            "allow_reward_compare": restore_summary.get("cutoff_gate_allow_reward_compare"),
            "pre_cutoff_transition_true_count": n_true,
            "pre_cutoff_transition_any_true": n_true > 0,
        },
        "nexus": {
            "seed": int(args.seed),
            "noise_hash": nexus_meta.get("noise_hash"),
            "physics_log_fallback_tokens": nexus_meta.get("physics_log_fallback_tokens", []),
            "coverage_counts": nexus_meta.get("coverage_counts"),
            "hold_ok": nexus_meta.get("hold_ok"),
            "future_hash": nexus_meta.get("future_hash"),
        },
        "pairwise": {
            k: {
                "noc_safe_to_danger": v["noc_flips"]["a_safe_to_b_danger"],
                "noc_danger_to_safe": v["noc_flips"]["a_danger_to_b_safe"],
                "noc_total_flip": v["noc_flips"]["total_flip"],
                "ttc_safe_to_danger": v["ttc_flips"]["a_safe_to_b_danger"],
                "ttc_danger_to_safe": v["ttc_flips"]["a_danger_to_b_safe"],
                "ttc_total_flip": v["ttc_flips"]["total_flip"],
                "kendall_tau_b": v["kendall_tau_b"],
                "score_range": v["score_range"],
                "topk_overlap": v["topk_overlap_index_tie_broken"],
            }
            for k, v in pairs.items()
        },
        "replay_idm_attribution_hint": {
            "full_restored_noc_flip": attr.get("full_restored_noc_flip"),
            "top_single_agent": (attr.get("single_agent_ranked_by_noc_flip") or [None])[0],
        },
        "ego_conditioning_hash": {
            "requested": req_hash,
            "restore_requested": restore_ego.get("requested_ego_conditioning_hash"),
            "match_restore": req_hash == restore_ego.get("requested_ego_conditioning_hash"),
        },
        "paths": {
            "restore_dir": str(restore_dir),
            "nexus_out": str(out_dir),
            "report_md": str(report_md),
        },
        "wall_s": round(time.time() - t0, 3),
        "rss_gb": round(_rss_gb(), 3),
        "pairwise_detail": pairs,
    }
    save_json(summary_json, summary)
    save_json(out_dir / "three_source_compare_summary.json", summary)

    # small csv
    rows = []
    for pair_name, v in summary["pairwise"].items():
        rows.append(
            {
                "pair": pair_name,
                "noc_safe_to_danger": v["noc_safe_to_danger"],
                "noc_danger_to_safe": v["noc_danger_to_safe"],
                "noc_total_flip": v["noc_total_flip"],
                "ttc_safe_to_danger": v["ttc_safe_to_danger"],
                "ttc_danger_to_safe": v["ttc_danger_to_safe"],
                "ttc_total_flip": v["ttc_total_flip"],
                "kendall_tau_b": v["kendall_tau_b"],
            }
        )
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Markdown report
    ri = summary["pairwise"]["replay_vs_idm"]
    rn = summary["pairwise"]["replay_vs_nexus"]
    inn = summary["pairwise"]["idm_vs_nexus"]
    top_attr = summary["replay_idm_attribution_hint"].get("top_single_agent") or {}
    md = f"""# cutoff=4 三源对齐冒烟（Replay / Restored-IDM / Nexus）

**工程冒烟 only（engineering smoke ≠ research go/no-go）。**  
场景来自 `navtest_failures` 工程子集；**禁止**与旧 cutoff=3 数字并表；本报告不作研究假设成立/否决结论。

最后更新：2026-07-21

## Provenance（硬条件）

| 项 | 值 |
|---|---|
| 协作仓 HEAD | `{head_sha}` |
| WorldEngine upstream | `{upstream_sha}`（未改） |
| scene | `{SCENE_ID}` |
| cutoff / horizon | **{cutoff}** / {horizon} |
| ego conditioning | `plan_idx.csv` **step={args.plan_idx_step} → plan_idx={plan_idx}** |
| plan_idx.csv | `{args.plan_idx_csv}` |
| 禁止 | Nexus 旧 conditioning `plan_idx=710` |
| SCORER_CONFIG.num_history | **未改**（仍为 4）；研究锚点 cutoff=4 ≠ 声称等于 num_history-1 |
| Nexus seed | `{args.seed}` |
| Nexus physics→log fallback tokens | `{nexus_meta.get("physics_log_fallback_tokens", [])}` |

## 1. Replay / IDM 门控（cutoff=4）

| 检查 | 结果 |
|---|---|
| cutoff gate | **{restore_summary.get("cutoff_gate_grade")}** |
| allow reward compare | {restore_summary.get("cutoff_gate_allow_reward_compare")} |
| `pre_cutoff_transition_flags` True 数 | **{n_true}** / {len(transition)} |
| 产物目录 | `{restore_dir}`（未覆盖任何 `smoke1_cutoff3_*`） |

## 2. 三源 fingerprint

| 源 | fingerprint |
|---|---|
| Replay | `{fp_replay}` |
| Restored-IDM | `{fp_idm}` |
| Nexus | `{fp_nexus}` |
| 三者对齐 | **{fingerprint_aligned}** |
| ego requested hash 与 restore 一致 | {summary["ego_conditioning_hash"]["match_restore"]} |

## 3. Pairwise 分歧（8192 候选）

| Pair | NOC safe→danger | NOC danger→safe | NOC total | TTC safe→danger | TTC total |
|---|---:|---:|---:|---:|---:|
| Replay vs IDM | {ri["noc_safe_to_danger"]} | {ri["noc_danger_to_safe"]} | {ri["noc_total_flip"]} | {ri["ttc_safe_to_danger"]} | {ri["ttc_total_flip"]} |
| Replay vs Nexus | {rn["noc_safe_to_danger"]} | {rn["noc_danger_to_safe"]} | {rn["noc_total_flip"]} | {rn["ttc_safe_to_danger"]} | {rn["ttc_total_flip"]} |
| IDM vs Nexus | {inn["noc_safe_to_danger"]} | {inn["noc_danger_to_safe"]} | {inn["noc_total_flip"]} | {inn["ttc_safe_to_danger"]} | {inn["ttc_total_flip"]} |

Kendall τ-b / Top-K / score range：见 `reports/cutoff4_three_source_summary.json`（并列按既有 tie-aware 协议）。

## 4. 归因提示（仅 Replay–IDM；工程级）

- full Restored-IDM NOC flip：{summary["replay_idm_attribution_hint"].get("full_restored_noc_flip")}
- top single-agent hybrid：`{top_attr.get("hybrid")}` NOC flip={top_attr.get("noc_total_flip")} tokens={top_attr.get("idm_tokens")}

与旧 cutoff=3 现象的关系：**仅文字描述、不混表**——若本轮仍见显著 Replay→IDM NOC/TTC 安全翻转且单车主导，可记为“同类工程现象在 cutoff=4 再现”；数字本身不得与 cutoff=3 行并排。

## 5. 冒烟问答（不作研究结论）

1. IDM 截止门控是否仍为 A？→ **{restore_summary.get("cutoff_gate_grade")}**（transition True={n_true}）
2. 三源 fingerprint 是否对齐？→ **{fingerprint_aligned}**
3. 能否导出逐候选 pairwise 子奖励差异？→ **能**（scores pkl 在 Git 外；摘要 json/csv 入库）
4. 与旧 cutoff=3 Replay–IDM 现象？→ 见上节文字描述，**禁止并表**

## 6. 停止线

本批到此停止。不扩 train-side、不后训、不 Table 1、不 SMART、不 BWM 配对。
"""
    report_md.write_text(md, encoding="utf-8")
    print(json.dumps({"phase": "B", "fingerprint_aligned": fingerprint_aligned, "report": str(report_md)}, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["A", "B", "all"], default="all")
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
        "--sidecar-root",
        type=Path,
        default=Path("/mnt/cpfs/prediction/lyyy/myself/WE/nexus_sidecar"),
    )
    ap.add_argument(
        "--nexus-out",
        type=Path,
        default=Path("/mnt/cpfs/prediction/lyyy/myself/WE/nexus_sidecar/outputs/smoke_cutoff4_three_source"),
    )
    ap.add_argument(
        "--restore-dir",
        type=Path,
        default=Path("/mnt/cpfs/prediction/lyyy/myself/WE/data/frozen_paired/smoke1_cutoff4_restore_v1"),
    )
    ap.add_argument(
        "--asset-folder",
        type=str,
        default="/mnt/cpfs/prediction/lyyy/myself/WE/data/smoke_1scene/assets/navtest_failures/assets",
    )
    ap.add_argument("--cutoff", type=int, default=DEFAULT_CUTOFF)
    ap.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    ap.add_argument("--plan-idx", type=int, default=1333)
    ap.add_argument("--plan-idx-csv", type=Path, default=PLAN_IDX_CSV_DEFAULT)
    ap.add_argument("--plan-idx-step", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument(
        "--report-md",
        type=Path,
        default=WE_ROOT / "reports/CUTOFF4_THREE_SOURCE_SMOKE.md",
    )
    ap.add_argument(
        "--summary-json",
        type=Path,
        default=WE_ROOT / "reports/cutoff4_three_source_summary.json",
    )
    ap.add_argument(
        "--summary-csv",
        type=Path,
        default=WE_ROOT / "reports/cutoff4_three_source_summary.csv",
    )
    args = ap.parse_args()

    if args.plan_idx != 1333:
        raise SystemExit("Hard lock: plan_idx must be 1333 for this smoke")
    if args.cutoff != 4:
        raise SystemExit("Hard lock: cutoff must be 4")

    if args.phase in ("A", "all"):
        rc = phase_a_nexus_generate(args)
        if rc != 0:
            return rc
    if args.phase in ("B", "all"):
        # Phase B requires restore scores
        for req in ("scores_log_replay.pkl", "scores_idm_restored.pkl", "fingerprint_cross_check.json"):
            if not (args.restore_dir / req).exists():
                print(f"WAIT: missing {args.restore_dir / req}", flush=True)
                return 10
        return phase_b_score_and_compare(args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
