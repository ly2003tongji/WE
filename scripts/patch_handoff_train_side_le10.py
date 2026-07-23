#!/usr/bin/env python3
"""Short HANDOFF.md update for train-side ≤10 completion (T5)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

WE_ROOT = Path(__file__).resolve().parents[1]
HANDOFF = WE_ROOT / "HANDOFF.md"
SUMMARY = WE_ROOT / "reports/train_side_le10_summary.json"
MARKER = "<!-- TRAIN_SIDE_LE10_BLOCK -->"


def main() -> int:
    import json

    summary = json.loads(SUMMARY.read_text()) if SUMMARY.exists() else {}
    block = f"""{MARKER}
## train-side ≤10（{datetime.now().strftime('%Y-%m-%d')}）

- 报告：[`reports/TRAIN_SIDE_LE10_DISAGREEMENT.md`](reports/TRAIN_SIDE_LE10_DISAGREEMENT.md)
- scene list：[`reports/train_side_le10_scene_list.md`](reports/train_side_le10_scene_list.md)
- 摘要：[`reports/train_side_le10_summary.json`](reports/train_side_le10_summary.json)
- 门控 A / degraded：{summary.get('n_gate_a_main', '?')} / {summary.get('n_degraded', '?')}
- conditioning：`static_feasible_median_path_length`（**≠** 冒烟 NR 1333；不可直接数值对比）
- 声称边界：初步效应量 only；未作 go/no-go / held-out 主张
- **停止线**：本批不扩 50、不后训、不 Table1、不 SMART
- 建议（未执行）：{summary.get('suggest_expand_50_or_smart', '见报告')}
{MARKER}
"""
    text = HANDOFF.read_text(encoding="utf-8") if HANDOFF.exists() else "# HANDOFF\n"
    if MARKER in text:
        parts = text.split(MARKER)
        # parts[0] ... parts[1]=old body ... parts[2]=after
        if len(parts) >= 3:
            text = parts[0] + block + parts[2]
        else:
            text = text.rstrip() + "\n\n" + block
    else:
        text = text.rstrip() + "\n\n" + block
    HANDOFF.write_text(text, encoding="utf-8")
    print(f"updated {HANDOFF}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
