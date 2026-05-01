"""
Terminal presentation helpers for demos and judging (no CSV side-effects).
"""

from __future__ import annotations

import json
import shutil
from typing import Any

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"
BLUE = "\033[94m"


def _c(txt: str, clr: str) -> str:
    return f"{clr}{txt}{RESET}"


def term_cols(default: int = 96) -> int:
    try:
        return max(52, shutil.get_terminal_size((default, 20)).columns)
    except Exception:  # pragma: no cover
        return default


def rule(char: str = "━", muted: bool = False) -> str:
    w = term_cols() - 1
    line = char * max(12, min(w, 88))
    return _c(line, DIM if muted else CYAN)


def progress_line(current: int, total: int, label: str) -> str:
    total = max(1, total)
    ratio = current / total
    short = label[:56] + ("…" if len(label) > 56 else "")
    bar_w = max(8, min(44, term_cols() - 28 - len(short)))
    filled = min(bar_w - 1, max(1, int(bar_w * ratio)))
    empty = max(0, bar_w - filled)
    bar = "[" + "=" * filled + ">" + "." * empty + "]"
    pct = int(100 * ratio)
    return f"{_c(bar, BLUE)} {_c(str(pct) + '%', DIM)} {_c(short, MAGENTA)}"


def confidence_meter(top1: float, margin: float, rag_ok: bool) -> tuple[str, str]:
    """Lexical-derived confidence headline for demos (telemetry — not calibrated probability)."""
    t = float(top1)

    if t >= 58.0 and margin >= 0.035:
        label = "Strong lexical relevance"
        color_g = GREEN
        filled = 4
    elif t >= 14.0 and margin >= 0.02:
        label = "Moderate relevance"
        color_g = CYAN
        filled = 3
    elif t >= 7.0:
        label = "Marginal retrieval"
        color_g = YELLOW
        filled = 2
    else:
        label = "Weak retrieval"
        color_g = RED
        filled = 1

    if not rag_ok:
        label += " · quality gate flagged"

    bar = "".join((_c("█", color_g) if i < filled else _c("░", DIM)) for i in range(4))
    return f"{bar}  {_c(label, color_g)}", label


def print_decision_explainer(tri: dict[str, Any], stats: dict[str, float], rag_ok: bool) -> None:
    """Judge-friendly one-screen summary."""
    esc = ""
    er = tri.get("escalation_reason") if isinstance(tri, dict) else None
    if isinstance(er, str) and er.strip():
        esc = f" · {_c('Escalation: ' + er.strip(), YELLOW)}"

    headline, _ = confidence_meter(stats.get("top1", 0.0), stats.get("margin", 0.0), rag_ok)

    summary = json.dumps(
        {
            k: tri.get(k)
            for k in ("status", "request_type", "product_area")
            if isinstance(tri, dict) and k in tri
        },
        indent=2,
        ensure_ascii=False,
    )
    route = ""
    if isinstance(tri, dict) and tri.get("debug_routing_label"):
        route = f" {_c('Route:', DIM)} {_c(str(tri['debug_routing_label']), CYAN)}"

    print(rule())
    print(f"{_c('DECISION', BOLD)}  {route}{esc}")
    st = isinstance(tri, dict) and tri.get("status") == "replied"
    print(_c(summary, GREEN if st else YELLOW))
    print(f"{_c('Retrieval telemetry', DIM)}  {headline}")
    trace = tri.get("trace") if isinstance(tri, dict) else None
    if isinstance(trace, dict) and trace.get("evidence"):
        print(_c("\nEvidence (ranked excerpts)", BOLD))
        for i, row in enumerate(trace["evidence"][:5], 1):
            if not isinstance(row, dict):
                continue
            ttl = row.get("title", "")[:100]
            dm = row.get("domain")
            sc = row.get("score")
            print(f"  {_c(str(i), DIM)} {_c(str(dm), CYAN)}  {_c(str(sc)[:10], BLUE)}  {ttl}")
    print(rule(muted=True))


def print_banner_subtitle(extra: str) -> None:
    print(rule())
    print(_c(extra.strip(), MAGENTA))
    print(rule(muted=True))
