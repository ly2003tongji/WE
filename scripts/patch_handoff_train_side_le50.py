#!/usr/bin/env python3
"""Short HANDOFF + CURRENT_STATE update for train-side ~50 (E5)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

WE_ROOT = Path(__file__).resolve().parents[1]
HANDOFF = WE_ROOT / "HANDOFF.md"
CURRENT = WE_ROOT / "research/CURRENT_STATE.md"
SUMMARY = WE_ROOT / "reports/train_side_le50_summary.json"
MARKER = "<!-- TRAIN_SIDE_LE50_BLOCK -->"


def main() -> int:
    import json
    import re

    summary = json.loads(SUMMARY.read_text()) if SUMMARY.exists() else {}
    counts = summary.get("counts") or {}
    block = f"""{MARKER}
## train-side ~50（{datetime.now().strftime('%Y-%m-%d')}）

- 报告：[`reports/TRAIN_SIDE_LE50_DISAGREEMENT.md`](reports/TRAIN_SIDE_LE50_DISAGREEMENT.md)
- scene list：[`reports/train_side_le50_scene_list.md`](reports/train_side_le50_scene_list.md)
- 摘要：[`reports/train_side_le50_summary.json`](reports/train_side_le50_summary.json)
- 复用 / 新尝试 / 门控 A / degraded：{counts.get('reuse_from_le10','?')} / {counts.get('new_restore_attempts','?')} / {counts.get('n_gate_a_main','?')} / {counts.get('n_degraded','?')}
- 总 restore 尝试（含 le10）：{counts.get('total_restore_attempts_incl_le10','?')} / 65
- conditioning：`static_feasible_median_path_length`（≠冒烟 NR 1333）
- 声称边界：效应量/分位数 only；未作 go/no-go / held-out 主张
- **停止线**：不自动 SMART / 后训 / held-out
- 建议（未执行）：{summary.get('suggest_smart_or_expand_or_shrink', '见报告')}
{MARKER}
"""
    text = HANDOFF.read_text(encoding="utf-8") if HANDOFF.exists() else "# HANDOFF\n"
    if MARKER in text:
        parts = text.split(MARKER)
        text = parts[0] + block + (parts[2] if len(parts) >= 3 else "")
    else:
        text = text.rstrip() + "\n\n" + block
    HANDOFF.write_text(text, encoding="utf-8")

    if CURRENT.exists():
        cur = CURRENT.read_text(encoding="utf-8")
        # bump last-updated line if present
        cur2 = re.sub(
            r"最后更新：.*",
            f"最后更新：{datetime.now().strftime('%Y-%m-%d')}（train-side~50 汇总完成；下一决策见建议字段）",
            cur,
            count=1,
        )
        pointer = (
            "\n\n## 下一决策指针（train-side ~50 后）\n\n"
            f"- 报告：`reports/TRAIN_SIDE_LE50_DISAGREEMENT.md`\n"
            f"- 门控 A / degraded：{counts.get('n_gate_a_main')} / {counts.get('n_degraded')}\n"
            f"- 建议（未执行）：{summary.get('suggest_smart_or_expand_or_shrink')}\n"
            "- 停止：不自动启动 SMART / 后训 / held-out；待 Mac 决策。\n"
        )
        if "## 下一决策指针（train-side ~50 后）" in cur2:
            cur2 = re.sub(
                r"## 下一决策指针（train-side ~50 后）[\s\S]*?(?=\n## |\Z)",
                pointer.strip() + "\n\n",
                cur2,
                count=1,
            )
        else:
            cur2 = cur2.rstrip() + "\n" + pointer
        # update latest commit note lightly
        cur2 = re.sub(
            r"最新有效协作提交：`[^`]+`",
            "最新有效协作提交：见 `git log -1`（~50 报告可能尚未 commit）",
            cur2,
            count=1,
        )
        CURRENT.write_text(cur2, encoding="utf-8")

    print(f"updated {HANDOFF} and {CURRENT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
