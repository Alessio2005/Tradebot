"""Robuustheidscheck: dezelfde config op meerdere opgeslagen markten (geen parameter-grid).

Elke markt draait als gewone replay naar `out/<naam van de marktmap>/`; de statistiek van
alle markten komt in `out/sweep.json`. Bij meer dan één worker draaien de markten in een
procespool (zelfde startmethode als de keten-pool: `forkserver` waar die bestaat); binnen zo'n
worker rekent de keten in-proces (`pricing.workers = 1`), zodat de pools elkaar niet
verdringen. De uitkomst van een markt hangt daar niet van af: de poolketen is bit-gelijk aan de
in-proces keten.
"""
from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from ...schemas.options import OptionsConfig
from ...utils.failfast import require
from ..chain.builder import _mp_context
from ..data.store import read_market
from ..pricing.settings import resolve_workers
from .replay import run_replay

__all__ = ["resolve_sweep_workers", "run_sweep"]


def resolve_sweep_workers(workers: int | None, n_markets: int) -> int:
    """`None` (auto): min(4, aantal cores), zie `resolve_workers`, nooit meer dan markten."""
    require(workers is None or workers >= 1, "sweep: --workers moet minstens 1 zijn")
    return max(1, min(n_markets, resolve_workers("auto") if workers is None else workers))


def _run_market(job: tuple[OptionsConfig, Path, Path]) -> dict[str, Any]:
    """Eén markt als replay (module-level: picklebaar voor de pool)."""
    cfg, market, out = job
    res = asyncio.run(run_replay(cfg, read_market(market), out))
    rec = res.as_record()
    return {"start_equity": rec["start_equity"], "end_equity": rec["end_equity"],
            "positions_opened": rec["positions_opened"],
            "positions_closed": rec["positions_closed"],
            "open_short_legs_past_expiry": rec["open_short_legs_past_expiry"],
            "halted": rec["halted"], "stats": rec["stats"]}


def run_sweep(cfg: OptionsConfig, markets: Sequence[Path], out: Path,
              workers: int | None = None) -> dict[str, Any]:
    """Draai alle `markets`, schrijf `out/sweep.json` en geef dat record terug."""
    names = [m.name for m in markets]
    require(len(markets) > 0 and len(set(names)) == len(names),
            "sweep: geef minstens één markt, met elk een andere mapnaam")
    n = resolve_sweep_workers(workers, len(markets))
    if n > 1:
        cfg = cfg.model_copy(update={"pricing": cfg.pricing.model_copy(update={"workers": 1})})
    jobs = [(cfg, m, out / m.name) for m in markets]
    if n == 1:
        results = [_run_market(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=n, mp_context=_mp_context()) as pool:
            results = list(pool.map(_run_market, jobs))
    record = {"markets": dict(zip(names, results, strict=True))}
    out.mkdir(parents=True, exist_ok=True)
    (out / "sweep.json").write_text(json.dumps(record, indent=1, sort_keys=True),
                                    encoding="utf-8")
    return record
