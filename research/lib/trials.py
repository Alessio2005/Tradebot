"""Append-only trial log. M for the deflated Sharpe = PRIOR_M + number of distinct names logged here.

PRIOR_M (13 + 4 = 17) is booked from the work BEFORE this programme:
  * 13 trend/carry variants of the uploaded session: T1, T2, C1 and the other 10 single/ensemble
    lookback x long-short/long-flat cells of its sensitivity table (cost multiples are not variants);
  * 4 meta-label variants of the weekly campaign (reference, logreg, forest, ensemble; AUC ~ 0.49-0.51).
"""
from __future__ import annotations

import json
from pathlib import Path

PRIOR_M = 17
LOG = Path(__file__).resolve().parents[1] / "trials.jsonl"


def _read() -> list[dict]:
    if not LOG.exists():
        return []
    return [json.loads(x) for x in LOG.read_text().splitlines() if x.strip()]


def log_trial(name: str, family: str, note: str = "", **metrics) -> bool:
    """Book a variant once. Returns True if it was new (and so raised M)."""
    names = {t["name"] for t in _read()}
    if name in names:
        return False
    with LOG.open("a") as fh:
        fh.write(json.dumps({"name": name, "family": family, "note": note, **metrics}, default=float) + "\n")
    return True


def M() -> int:
    return PRIOR_M + len({t["name"] for t in _read()})
