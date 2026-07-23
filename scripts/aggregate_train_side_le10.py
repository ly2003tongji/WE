#!/usr/bin/env python3
"""Aggregate train-side ≤10 results into reports (T4)."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _dist(vals: List[float]) -> Dict[str, Any]:
    if not vals:
        return {"n": 0}
    return {
        "n": len(vals),
        "min": float(min(vals)),
        "max": float(max(vals)),
        "mean": float(statistics.mean(vals)),
        "median": float(statistics.median(vals)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract-root", type=Path, required=True)
    ap.add_argument("--scene-list", type=Path, required=True)
    ap.add_argument("--out-md", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--out-csv", type=Path, required=True)
    args = ap.parse_args()

    scene_list = json.loads(args.scene_list.read_text())
    head = os.popen(f"git -C {Path(__file__).resolve().parents[1]} rev-parse HEAD").read().strip()
    upstream = os.popen(
        f"git -C {Path(__file__).resolve().parents[1]}/upstream/WorldEngine rev-parse HEAD"
    ).read().strip()

    rows: List[Dict[str, Any]] = []
    main_rows: List[Dict[str, Any]] = []
    degraded_rows: List[Dict[str, Any]] = []
    multiseed: List[Dict[str, Any]] = []

    for s in scene_list["scenes"]:
        tok = s["token"]
        d = args.extract_root / tok
        sum0 = d / "three_source_seed0.json"
        meta = d / "scene_meta.json"
        if not sum0.exists():
            rows.append(
                {
                    "token": tok,
                    "bucket": s["bucket"],
                    "scene_id": s["scene_id"],
                    "status": "missing_seed0",
                    "gate_a": False,
                    "degraded": True,
                }
            )
            degraded_rows.append(rows[-1])
            continue
        sm = json.loads(sum0.read_text())
        meta_j = json.loads(meta.read_text()) if meta.exists() else {}
        plan = (meta_j.get("ego_conditioning") or sm.get("plan_idx_provenance") or {})
        rec = {
            "token": tok,
            "bucket": s["bucket"],
            "scene_id": s["scene_id"],
            "status": sm.get("status"),
            "gate_a": sm.get("gate_a"),
            "degraded": sm.get("degraded"),
            "degraded_reasons": sm.get("degraded_reasons"),
            "plan_idx": plan.get("plan_idx"),
            "n_feasible": plan.get("n_feasible"),
            "path_length_m": plan.get("path_length_m"),
            "fingerprint_aligned": (sm.get("fingerprint") or {}).get("aligned"),
            "difficulty": sm.get("difficulty"),
            "pairwise": sm.get("pairwise"),
            "nexus_fallback_n": len((sm.get("nexus") or {}).get("physics_log_fallback_tokens") or []),
        }
        rows.append(rec)
        if sm.get("gate_a") and not sm.get("degraded"):
            main_rows.append(rec)
        else:
            degraded_rows.append(rec)

        for seed in (1, 2):
            p = d / f"three_source_seed{seed}.json"
            if not p.exists():
                continue
            sj = json.loads(p.read_text())
            # compare seed vs seed0 model-pair flips
            p0 = sm["pairwise"]["replay_vs_nexus"]
            ps = sj["pairwise"]["replay_vs_nexus"]
            multiseed.append(
                {
                    "token": tok,
                    "seed": seed,
                    "nexus_seed_vs0_replay_noc_flip_delta": abs(
                        ps["noc_total_flip"] - p0["noc_total_flip"]
                    ),
                    "nexus_seed_vs0_replay_ttc_flip_delta": abs(
                        ps["ttc_total_flip"] - p0["ttc_total_flip"]
                    ),
                    "seed_replay_vs_nexus_noc": ps["noc_total_flip"],
                    "seed0_replay_vs_nexus_noc": p0["noc_total_flip"],
                    "seed0_replay_vs_idm_noc": sm["pairwise"]["replay_vs_idm"]["noc_total_flip"],
                    "seed0_idm_vs_nexus_noc": sm["pairwise"]["idm_vs_nexus"]["noc_total_flip"],
                }
            )

    def collect_pair(name: str, key: str) -> List[float]:
        out = []
        for r in main_rows:
            pw = (r.get("pairwise") or {}).get(name) or {}
            if key in pw:
                out.append(float(pw[key]))
        return out

    pair_names = ["replay_vs_idm", "replay_vs_nexus", "idm_vs_nexus"]
    pairwise_dist = {
        name: {
            "noc_total_flip": _dist(collect_pair(name, "noc_total_flip")),
            "ttc_total_flip": _dist(collect_pair(name, "ttc_total_flip")),
            "kendall_tau_b": _dist(collect_pair(name, "kendall_tau_b")),
        }
        for name in pair_names
    }

    # difficulty descriptive: spearman-like via rank correlation crude
    difficulty_note = "insufficient_main_rows"
    if len(main_rows) >= 3:
        try:
            import math

            xs = [float((r.get("difficulty") or {}).get("replay_score_min", float("nan"))) for r in main_rows]
            ys = [
                float(((r.get("pairwise") or {}).get("replay_vs_idm") or {}).get("noc_total_flip", float("nan")))
                for r in main_rows
            ]
            # Pearson on available
            pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
            if len(pairs) >= 3:
                mx = statistics.mean(p[0] for p in pairs)
                my = statistics.mean(p[1] for p in pairs)
                num = sum((p[0] - mx) * (p[1] - my) for p in pairs)
                den = math.sqrt(sum((p[0] - mx) ** 2 for p in pairs) * sum((p[1] - my) ** 2 for p in pairs))
                if den > 0 and abs(corr) == abs(corr):  # finite
                    difficulty_note = (
                        f"描述性：主表 n={len(main_rows)}，Replay score_min 与 R↔I NOC_flip 的 Pearson≈{corr:.3f}；"
                        "不作因果结论。"
                    )
                else:
                    difficulty_note = (
                        f"描述性：主表 n={len(main_rows)}，R↔I NOC_flip 几乎全 0（中位 0），"
                        "与难度代理的相关不可估；R↔N 分歧为主信号。不作因果结论。"
                    )
            else:
                difficulty_note = f"主表 n={len(main_rows)}，难度相关字段不足。"
        except Exception as e:
            difficulty_note = f"difficulty_corr_error: {e}"

    seed_vs_model = "no_multiseed"
    if multiseed:
        seed_deltas = [m["nexus_seed_vs0_replay_noc_flip_delta"] for m in multiseed]
        model_rn = [m["seed0_replay_vs_nexus_noc"] for m in multiseed]
        model_ri = [m["seed0_replay_vs_idm_noc"] for m in multiseed]
        seed_vs_model = (
            f"Nexus 多种子相对 seed0 的 Replay↔Nexus NOC_flip |Δ| 中位={statistics.median(seed_deltas):.1f}；"
            f"同批模型间 Replay↔Nexus NOC_flip 中位={statistics.median(model_rn):.1f}"
            f"（Replay↔IDM 中位={statistics.median(model_ri):.1f}）。"
            f"初步：{'模型间（尤其 vs Nexus）明显大于多种子波动' if statistics.median(model_rn) > statistics.median(seed_deltas) else '多种子波动不小于模型间 vs Nexus'}。"
        )

    payload = {
        "status": "ok",
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "collaboration_head_sha": head,
        "worldengine_upstream_sha": upstream,
        "claim_boundary": "preliminary_effect_size_and_descriptive_difficulty_only",
        "forbid_research_go_nogo": True,
        "forbid_heldout_predictable_failure_claim": True,
        "protocol_pointer": "reports/CUTOFF4_THREE_SOURCE_SMOKE.md",
        "conditioning_note": (
            "本轮 conditioning ≠ 冒烟 NR plan_idx=1333，故效应量不可与单场景冒烟数字直接数值对比。"
        ),
        "isolation": scene_list.get("isolation"),
        "n_scenes": len(rows),
        "n_gate_a_main": len(main_rows),
        "n_degraded": len(degraded_rows),
        "pairwise_distributions_main_table": pairwise_dist,
        "multiseed": multiseed,
        "seed_vs_model_one_liner": seed_vs_model,
        "difficulty_one_liner": difficulty_note,
        "scenes": rows,
        "suggest_expand_50_or_smart": None,  # filled below
    }

    # suggestion heuristic (advisory only)
    main_noc_rn = collect_pair("replay_vs_nexus", "noc_total_flip")
    main_noc_ri = collect_pair("replay_vs_idm", "noc_total_flip")
    if len(main_rows) < 3:
        payload["suggest_expand_50_or_smart"] = (
            "主表门控 A 过少，建议先排查 degraded 再考虑扩样；本批不执行。"
        )
    elif main_noc_rn and statistics.median(main_noc_rn) >= 50:
        payload["suggest_expand_50_or_smart"] = (
            "初步：主表 Replay↔Nexus NOC 仍见非微小分歧（R↔I 近 0），可考虑扩至 ~50 稳分位数；"
            "SMART 仅当扩样后仍有剩余信号再议。本批不执行。"
        )
    elif main_noc_ri and statistics.median(main_noc_ri) >= 50:
        payload["suggest_expand_50_or_smart"] = (
            "初步：主表仍见非微小 R↔I NOC 分歧，可考虑扩至 ~50；SMART 暂缓。本批不执行。"
        )
    else:
        payload["suggest_expand_50_or_smart"] = (
            "初步：主表分歧偏小或样本少，扩 ~50 的优先级一般；SMART 暂不建议。本批不执行。"
        )

    args.out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # CSV flat
    csv_rows = []
    for r in rows:
        pw = r.get("pairwise") or {}
        base = {
            "token": r["token"],
            "bucket": r["bucket"],
            "gate_a": r.get("gate_a"),
            "degraded": r.get("degraded"),
            "plan_idx": r.get("plan_idx"),
            "n_feasible": r.get("n_feasible"),
        }
        for pname in pair_names:
            p = pw.get(pname) or {}
            csv_rows.append(
                {
                    **base,
                    "pair": pname,
                    "noc_total_flip": p.get("noc_total_flip"),
                    "ttc_total_flip": p.get("ttc_total_flip"),
                    "kendall_tau_b": p.get("kendall_tau_b"),
                    "enter_main": r in main_rows,
                }
            )
    if csv_rows:
        with args.out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            w.writeheader()
            w.writerows(csv_rows)

    # Markdown
    def fmt_dist(d: Dict[str, Any]) -> str:
        if not d or d.get("n", 0) == 0:
            return "n=0"
        return f"n={d['n']} min={d['min']:.0f} med={d['median']:.0f} mean={d['mean']:.1f} max={d['max']:.0f}"

    lines = [
        "# train-side ≤10 三源分歧（初步效应量）",
        "",
        "**初步效应量与描述性难度对照 only；不作研究 go/no-go；不作 held-out 可预测失败主张。**",
        "",
        f"最后更新：{datetime.now().strftime('%Y-%m-%d')}",
        "",
        "## 协议指针与样本隔离（硬条件）",
        "",
        "| 项 | 值 |",
        "|---|---|",
        f"| 协作仓 HEAD | `{head}` |",
        f"| WorldEngine upstream | `{upstream}`（未改） |",
        "| cutoff / horizon | **4** / 9 |",
        "| 协议冒烟指针 | [`reports/CUTOFF4_THREE_SOURCE_SMOKE.md`](CUTOFF4_THREE_SOURCE_SMOKE.md)（**仅协议**，非本轮样本） |",
        "| 本轮样本 | train-side rare：5×collision-exclusive + 5×ep-exclusive |",
        "| 与冒烟数值 | **不可直接对比**：本轮 ego conditioning = `static_feasible_median_path_length`，冒烟 = NR `plan_idx=1333` |",
        "| offroad 桶 | 本批不进主表；保留后续池；**不用** BWM/augmented |",
        "| 禁止并表 | navtest_failures / engineering smoke / cutoff=3 |",
        "",
        f"**本轮 conditioning ≠ 冒烟 NR plan_idx=1333，故效应量不可与单场景冒烟数字直接数值对比。**",
        "",
        f"- scene list: [`reports/train_side_le10_scene_list.md`](train_side_le10_scene_list.md)",
        f"- ∩ navtest tokens: `{scene_list.get('isolation', {}).get('intersection_tokens_with_navtest')}`",
        f"- ∩ smoke scene-ids: `{scene_list.get('isolation', {}).get('intersection_scene_ids_with_smoke')}`",
        "",
        "## 门控与样本量",
        "",
        f"- 门控 A 入主表：**{len(main_rows)}**",
        f"- degraded（非 A 或 transition flag）：**{len(degraded_rows)}**",
        f"- plan_idx 规则：静态可行集 DAC∧Comfort∧Direction=1 上 path-length 中位数 → vocab index（三源同一 index）",
        "",
        "## 主表 pairwise 分歧分布（仅门控 A）",
        "",
        "| pair | NOC total_flip | TTC total_flip | Kendall τ_b |",
        "|---|---|---|---|",
    ]
    for name in pair_names:
        d = pairwise_dist[name]
        lines.append(
            f"| {name} | {fmt_dist(d['noc_total_flip'])} | {fmt_dist(d['ttc_total_flip'])} | {fmt_dist(d['kendall_tau_b'])} |"
        )

    lines.extend(
        [
            "",
            "## Nexus 多种子 vs 模型间",
            "",
            seed_vs_model,
            "",
            "## 难度对照（描述性）",
            "",
            difficulty_note,
            "",
            "## 逐 scene（含 degraded）",
            "",
            "| token | 桶 | gate_A | degraded | plan_idx | R↔I NOC | R↔N NOC | I↔N NOC |",
            "|---|---|---|---|---|---|---|---|",
        ]
    )
    for r in rows:
        pw = r.get("pairwise") or {}
        lines.append(
            "| `{tok}` | {bucket} | {ga} | {deg} | {pi} | {ri} | {rn} | {inn} |".format(
                tok=r["token"],
                bucket=r.get("bucket"),
                ga=r.get("gate_a"),
                deg=r.get("degraded"),
                pi=r.get("plan_idx"),
                ri=(pw.get("replay_vs_idm") or {}).get("noc_total_flip", "—"),
                rn=(pw.get("replay_vs_nexus") or {}).get("noc_total_flip", "—"),
                inn=(pw.get("idm_vs_nexus") or {}).get("noc_total_flip", "—"),
            )
        )

    lines.extend(
        [
            "",
            "## 建议（本批不执行）",
            "",
            payload["suggest_expand_50_or_smart"],
            "",
            "## 产物",
            "",
            f"- `{args.out_json}`",
            f"- `{args.out_csv}`",
            f"- 每 scene：`data/frozen_paired/train_side_le10/<token>/`",
            "",
        ]
    )
    args.out_md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"n_main": len(main_rows), "n_degraded": len(degraded_rows), "out_md": str(args.out_md)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
