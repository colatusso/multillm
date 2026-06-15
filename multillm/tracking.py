"""Tracking of which models the judge picks over time (persisted in JSON).

On each mixture, records per model:
  runs  -> participated (was a proposer)
  wins  -> came 1st in the synthesizer's ranking
  borda -> points by position (1st of N = N points, 2nd = N-1, ...)

This lets you see who wins the most -- the basis for 'learning' which models are
worth it (e.g. retiring the one that never wins).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .mixture import MixtureResult


def load_stats(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def save_stats(stats: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")


def record_result(result: "MixtureResult", path: str | Path) -> dict:
    """Updates the tracking with a MixtureResult and persists it. Returns the stats."""
    stats = load_stats(path)
    n = len(result.candidates)
    for c in result.candidates:
        s = stats.setdefault(c.model, {"runs": 0, "wins": 0, "borda": 0})
        s["runs"] += 1
    for pos, idx in enumerate(result.ranking or []):
        if 0 <= idx < n:
            s = stats[result.candidates[idx].model]
            if pos == 0:
                s["wins"] += 1
            s["borda"] += n - pos
    save_stats(stats, path)
    return stats


def _win_rate(s: dict) -> float:
    runs = s.get("runs", 0)
    return s.get("wins", 0) / runs if runs else 0.0


def format_stats(path: str | Path) -> str:
    stats = load_stats(path)
    if not stats:
        return "Tracking: (no data yet)"
    # best by win %; tie-break by runs (more samples = more reliable)
    order = sorted(stats.items(), key=lambda kv: (_win_rate(kv[1]), kv[1].get("runs", 0)), reverse=True)
    w = max(len(m) for m, _ in order)
    rows = ["Tracking (best by win %):"]
    for model, s in order:
        runs = s.get("runs", 0)
        pct = f"{_win_rate(s) * 100:.0f}%" if runs else "-"
        rows.append(f"  {model:<{w}}  {pct:>4}  ({s.get('wins', 0)}/{runs} wins)  borda={s.get('borda', 0)}")
    return "\n".join(rows)
