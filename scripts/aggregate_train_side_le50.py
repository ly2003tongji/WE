#!/usr/bin/env python3
"""Aggregate train-side ~50 results (E4). Stratified le10 vs new; multiseed vs model."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _pct(vals: List[float], q: float) -> Optional[float]:
    if not vals:
        return None
    xs = sorted(vals)
    if len(xs) == 1:
        return float(xs[0])
    pos = (len(xs) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(xs[lo])
    return float(xs[lo] * (hi - pos) + xs[hi] * (pos - lo))


def _dist(vals: List[float]) -> Dict[str, Any]:
    if not vals:
        return {"n": 0}
    return {
        "n": len(vals),
        "min": float(min(vals)),
        "p25": _pct(vals, 0.25),
        "median": float(statistics.median(vals)),
        "p75": _pct(vals, 0.75),
        "iqr": float(_pct(vals, 0.75) - _pct(vals, 0.25)),  # type: ignore
        "mean": float(statistics.mean(vals)),
        "max": float(max(vals)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract-root", type=Path, required=True)
    ap.add_argument("--scene-list", type=Path, required=True)
    ap.add_argument("--attempt-state", type=Path, required=True)
    ap.add_argument("--out-md", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--out-csv", type=Path, required=True)
    args = ap.parse_args()

    scene_list = json.loads(args.scene_list.read_text())
    attempt = json.loads(args.attempt_state.read_text()) if args.attempt_state.exists() else {}
    activated = set()
    act_path = args.extract_root / "_activated_reserve.txt"
    if act_path.exists():
        activated = {ln.strip() for ln in act_path.read_text().splitlines() if ln.strip()}

    head = os.popen(f"git -C {Path(__file__).resolve().parents[1]} rev-parse HEAD").read().strip()
    upstream = os.popen(
        f"git -C {Path(__file__).resolve().parents[1]}/upstream/WorldEngine rev-parse HEAD"
    ).read().strip()

    # include activated reserve into analysis set
    catalog = list(scene_list["scenes"])
    for r in scene_list.get("reserve", []):
        if r["token"] in activated:
            catalog.append({**r, "from_reserve_activated": True})

    rows: List[Dict[str, Any]] = []
    main_rows: List[Dict[str, Any]] = []
    degraded_rows: List[Dict[str, Any]] = []
    multiseed: List[Dict[str, Any]] = []

    for s in catalog:
        tok = s["token"]
        d = args.extract_root / tok
        # follow symlink
        d = d.resolve() if d.exists() else d
        sum0 = d / "three_source_seed0.json"
        meta = d / "scene_meta.json"
        base = {
            "token": tok,
            "bucket": s["bucket"],
            "scene_id": s["scene_id"],
            "from_le10": bool(s.get("from_le10")),
            "reuse_from_le10": bool(s.get("reuse_from_le10")),
            "from_reserve_activated": bool(s.get("from_reserve_activated")),
        }
        if not sum0.exists():
            rec = {**base, "status": "missing_seed0", "gate_a": False, "degraded": True}
            rows.append(rec)
            degraded_rows.append(rec)
            continue
        sm = json.loads(sum0.read_text())
        meta_j = json.loads(meta.read_text()) if meta.exists() else {}
        plan = (meta_j.get("ego_conditioning") or sm.get("plan_idx_provenance") or {})
        rec = {
            **base,
            "status": sm.get("status"),
            "gate_a": sm.get("gate_a"),
            "degraded": sm.get("degraded"),
            "degraded_reasons": sm.get("degraded_reasons"),
            "plan_idx": plan.get("plan_idx") if isinstance(plan, dict) else None,
            "n_feasible": plan.get("n_feasible") if isinstance(plan, dict) else None,
            "equals_legacy_710": plan.get("equals_legacy_710") if isinstance(plan, dict) else None,
            "difficulty": sm.get("difficulty"),
            "pairwise": sm.get("pairwise"),
        }
        rows.append(rec)
        if sm.get("gate_a") and not sm.get("degraded"):
            main_rows.append(rec)
        else:
            degraded_rows.append(rec)

        if not (sm.get("gate_a") and sm.get("pairwise")):
            continue
        for seed in (1, 2):
            p = d / f"three_source_seed{seed}.json"
            if not p.exists():
                continue
            sj = json.loads(p.read_text())
            if not sj.get("pairwise"):
                continue
            p0 = sm["pairwise"]["replay_vs_nexus"]
            ps = sj["pairwise"]["replay_vs_nexus"]
            multiseed.append(
                {
                    "token": tok,
                    "bucket": s["bucket"],
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

    def collect(subset: List[Dict[str, Any]], pair: str, key: str) -> List[float]:
        out = []
        for r in subset:
            pw = (r.get("pairwise") or {}).get(pair) or {}
            if key in pw and pw[key] is not None:
                out.append(float(pw[key]))
        return out

    pairs = ["replay_vs_idm", "replay_vs_nexus", "idm_vs_nexus"]
    le10_main = [r for r in main_rows if r.get("from_le10")]
    new_main = [r for r in main_rows if not r.get("from_le10")]

    def pair_block(subset: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            name: {
                "noc_total_flip": _dist(collect(subset, name, "noc_total_flip")),
                "ttc_total_flip": _dist(collect(subset, name, "ttc_total_flip")),
                "kendall_tau_b": _dist(collect(subset, name, "kendall_tau_b")),
            }
            for name in pairs
        }

    ri_noc = collect(main_rows, "replay_vs_idm", "noc_total_flip")
    ri_zero_frac = (sum(1 for x in ri_noc if x == 0.0) / len(ri_noc)) if ri_noc else None

    seed_vs_model = "no_multiseed"
    if multiseed:
        seed_deltas = [m["nexus_seed_vs0_replay_noc_flip_delta"] for m in multiseed]
        model_rn = [m["seed0_replay_vs_nexus_noc"] for m in multiseed]
        model_ri = [m["seed0_replay_vs_idm_noc"] for m in multiseed]
        seed_vs_model = (
            f"Nexus 多种子相对 seed0 的 R↔N NOC |Δ| 中位={statistics.median(seed_deltas):.1f}；"
            f"同批模型间 R↔N NOC 中位={statistics.median(model_rn):.1f}"
            f"（R↔I 中位={statistics.median(model_ri):.1f}）。"
            f"初步：{'模型间（尤其 vs Nexus）明显大于多种子波动' if statistics.median(model_rn) > statistics.median(seed_deltas) else '多种子波动不小于模型间 vs Nexus'}。"
        )

    # difficulty descriptive
    difficulty_note = "insufficient"
    if len(main_rows) >= 3:
        xs = [float((r.get("difficulty") or {}).get("replay_score_min", float("nan"))) for r in main_rows]
        ys = collect(main_rows, "replay_vs_nexus", "noc_total_flip")
        pairs_xy = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x)]
        if len(pairs_xy) >= 3 and len(set(p[1] for p in pairs_xy)) > 1:
            mx = statistics.mean(p[0] for p in pairs_xy)
            my = statistics.mean(p[1] for p in pairs_xy)
            num = sum((p[0] - mx) * (p[1] - my) for p in pairs_xy)
            den = math.sqrt(
                sum((p[0] - mx) ** 2 for p in pairs_xy) * sum((p[1] - my) ** 2 for p in pairs_xy)
            )
            corr = (num / den) if den > 0 else float("nan")
            difficulty_note = (
                f"描述性：主表 n={len(main_rows)}，Replay score_min 与 R↔N NOC_flip Pearson≈{corr:.3f}；不作因果结论。"
            )
        else:
            difficulty_note = (
                f"描述性：主表 n={len(main_rows)}；R↔I≈0 为主时难度相关可看 R↔N。"
                f"R↔I NOC=0 比例={ri_zero_frac}。不作因果结论。"
            )

    n_reuse = sum(1 for r in rows if r.get("reuse_from_le10"))
    n_new_run = int(attempt.get("new_restore_attempts", 0))
    total_attempts = 10 + n_new_run

    rn_med = pair_block(main_rows)["replay_vs_nexus"]["noc_total_flip"].get("median")
    ri_med = pair_block(main_rows)["replay_vs_idm"]["noc_total_flip"].get("median")
    if len(main_rows) < 40:
        suggest = (
            f"门控 A={len(main_rows)} 未达目标 48–50（尝试已触顶 65；degraded 偏高，多见 restore/gateC）；"
            "建议优先修 IDM routing/门控再扩，SMART 暂缓。本批不执行。"
        )
    elif rn_med is not None and rn_med >= 50 and (ri_zero_frac or 0) >= 0.7:
        suggest = (
            "初步：R↔N 中位仍非微且多数 scene R↔I NOC=0，可考虑 SMART 接入；"
            "再扩优先级一般。本批不执行。"
        )
    elif rn_med is not None and rn_med >= 50:
        suggest = (
            f"初步：R↔N 中位仍非微（med={rn_med}），但 R↔I NOC=0 仅约 {ri_zero_frac:.0%}（中位 {ri_med}）；"
            "主张应写清“中位近 0、尾部非空”。SMART 可议但先消化 IDM 尾部。本批不执行。"
        )
    elif rn_med is not None and rn_med < 20:
        suggest = "初步：R↔N 中位偏小，建议收缩“普遍大效应”主张，SMART 暂缓。本批不执行。"
    else:
        suggest = "初步：效应量中等，可小步再扩或启动 SMART 可行性；不作 go/no-go。本批不执行。"

    degraded_reasons: Dict[str, int] = {}
    for r in degraded_rows:
        for reason in r.get("degraded_reasons") or [r.get("status") or "unknown"]:
            key = str(reason).split(",")[0][:80]
            degraded_reasons[key] = degraded_reasons.get(key, 0) + 1

    payload = {
        "status": "ok",
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "collaboration_head_sha": head,
        "worldengine_upstream_sha": upstream,
        "claim_boundary": "effect_size_and_percentiles_only",
        "forbid_research_go_nogo": True,
        "forbid_heldout_predictable_failure_claim": True,
        "protocol_pointer_le10": "reports/TRAIN_SIDE_LE10_DISAGREEMENT.md",
        "conditioning_note": (
            "本轮 conditioning ≠ 冒烟 NR plan_idx=1333，故效应量不可与单场景冒烟数字直接数值对比。"
        ),
        "counts": {
            "planned_n": scene_list.get("n_planned"),
            "analyzed_n": len(rows),
            "reuse_from_le10": n_reuse,
            "new_restore_attempts": n_new_run,
            "total_restore_attempts_incl_le10": total_attempts,
            "n_gate_a_main": len(main_rows),
            "n_degraded": len(degraded_rows),
            "activated_reserve": sorted(activated),
        },
        "isolation": scene_list.get("isolation"),
        "pairwise_distributions_main_all": pair_block(main_rows),
        "pairwise_distributions_main_le10": pair_block(le10_main),
        "pairwise_distributions_main_new": pair_block(new_main),
        "replay_vs_idm_noc_zero_fraction": ri_zero_frac,
        "multiseed": multiseed,
        "multiseed_n_scenes": len({m["token"] for m in multiseed}),
        "seed_vs_model_one_liner": seed_vs_model,
        "difficulty_one_liner": difficulty_note,
        "degraded_reason_counts": degraded_reasons,
        "suggest_smart_or_expand_or_shrink": suggest,
        "scenes": rows,
    }
    args.out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    csv_rows = []
    for r in rows:
        pw = r.get("pairwise") or {}
        for pname in pairs:
            p = pw.get(pname) or {}
            csv_rows.append(
                {
                    "token": r["token"],
                    "bucket": r["bucket"],
                    "from_le10": r.get("from_le10"),
                    "reuse_from_le10": r.get("reuse_from_le10"),
                    "gate_a": r.get("gate_a"),
                    "degraded": r.get("degraded"),
                    "plan_idx": r.get("plan_idx"),
                    "pair": pname,
                    "noc_total_flip": p.get("noc_total_flip"),
                    "ttc_total_flip": p.get("ttc_total_flip"),
                    "kendall_tau_b": p.get("kendall_tau_b"),
                }
            )
    if csv_rows:
        with args.out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            w.writeheader()
            w.writerows(csv_rows)

    def fmt(d: Dict[str, Any]) -> str:
        if not d or d.get("n", 0) == 0:
            return "n=0"
        return (
            f"n={d['n']} min={d['min']:.0f} p25={d['p25']:.0f} med={d['median']:.0f} "
            f"p75={d['p75']:.0f} IQR={d['iqr']:.0f} max={d['max']:.0f}"
        )

    def table_block(title: str, block: Dict[str, Any]) -> List[str]:
        lines = [f"### {title}", "", "| pair | NOC total_flip | TTC total_flip |", "|---|---|---|"]
        for name in pairs:
            d = block[name]
            lines.append(f"| {name} | {fmt(d['noc_total_flip'])} | {fmt(d['ttc_total_flip'])} |")
        lines.append("")
        return lines

    md: List[str] = [
        "# train-side ~50 三源分歧（效应量 / 分位数）",
        "",
        "**效应量与分位数描述 only；不作研究 go/no-go；不作 held-out 可预测失败主张。**",
        "",
        f"最后更新：{datetime.now().strftime('%Y-%m-%d')}",
        "",
        "## 协议指针与样本隔离",
        "",
        "| 项 | 值 |",
        "|---|---|",
        f"| 协作仓 HEAD | `{head}` |",
        f"| WorldEngine upstream | `{upstream}`（未改） |",
        "| cutoff / horizon | **4** / 9 |",
        "| 协议继承 | [`TRAIN_SIDE_LE10_DISAGREEMENT.md`](TRAIN_SIDE_LE10_DISAGREEMENT.md) |",
        "| conditioning | `static_feasible_median_path_length` |",
        "| 冒烟对比 | **不可直接对比** NR `plan_idx=1333` |",
        "",
        f"**本轮 conditioning ≠ 冒烟 NR plan_idx=1333，故效应量不可与单场景冒烟数字直接数值对比。**",
        "",
        f"- scene list: [`train_side_le50_scene_list.md`](train_side_le50_scene_list.md)",
        f"- ∩ navtest: `{scene_list.get('isolation', {}).get('intersection_tokens_with_navtest')}`",
        f"- ∩ smoke: `{scene_list.get('isolation', {}).get('intersection_scene_ids_with_smoke')}`",
        "",
        "## 样本量与尝试",
        "",
        f"- 复用 le10 (symlink skip): **{n_reuse}**",
        f"- 新 restore 尝试: **{n_new_run}**；总尝试（含 le10 10）: **{total_attempts}** / 65",
        f"- 门控 A 主表: **{len(main_rows)}**；degraded: **{len(degraded_rows)}**",
        f"- 激活 reserve: `{sorted(activated)}`",
        f"- degraded 原因计数: `{degraded_reasons}`",
        "",
        "## 全样本 pairwise 分位数（仅门控 A）",
        "",
    ]
    md.extend(table_block("全样本", pair_block(main_rows)))
    md.append(f"**R↔I NOC=0 比例（门控 A）**: `{ri_zero_frac}`")
    md.append("")
    md.extend(table_block("le10 子集（门控 A）", pair_block(le10_main)))
    md.extend(table_block("新增子集（门控 A）", pair_block(new_main)))
    md.extend(
        [
            "## Nexus 多种子 vs 模型间",
            "",
            f"- 多种子 scene 数（有 seed1/2）: **{payload['multiseed_n_scenes']}**",
            "",
            seed_vs_model,
            "",
            "## 难度对照（描述性）",
            "",
            difficulty_note,
            "",
            "## 建议（本批不执行）",
            "",
            suggest,
            "",
            "## 产物",
            "",
            f"- `{args.out_json}`",
            f"- `{args.out_csv}`",
            f"- 每 scene：`data/frozen_paired/train_side_le50/<token>/`",
            "",
        ]
    )
    args.out_md.write_text("\n".join(md), encoding="utf-8")
    print(
        json.dumps(
            {
                "n_main": len(main_rows),
                "n_degraded": len(degraded_rows),
                "ri_zero": ri_zero_frac,
                "out": str(args.out_md),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
