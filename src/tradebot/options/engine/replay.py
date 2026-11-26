"""Replay: de engine op een opgeslagen markt met een virtuele klok (detailontwerp 12.2).

De klok loopt per handelsdag van de opening tot de sluiting in stappen van
`execution.walk_interval_seconds`, plus één ronde op `schedule.end_of_day`. Er is geen echte
tijd en geen netwerk: de broker is de `PaperBroker`, het geld is fictief.

Het universum wordt de namen uit de markt, met beta 1,0 voor elke naam die de config niet kent:
een synthetische markt heeft geen SPY-beta.

Met `workers > 1` worden alle ketens vooraf parallel gebouwd (IV en greeks per contract zijn
het dure deel, ruim 90 % van de tijd) en daarna uit een cache gelezen. Het bouwen van een keten
hangt alleen af van de snapshot en de carry, dus de uitkomst is gelijk aan `workers=1`.
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from ...schemas.options import OptionsConfig
from ..calendar import NY, TradingCalendar
from ..chain.builder import Chain, ChainBuilder, RawOption
from ..data.feed import ReplayFeed
from ..data.store import Market, Snapshot
from ..execution.paper import PaperBroker
from ..rates import Carry
from ..strategies.common import parse_time
from .service import Engine, Paths
from .state import EngineState

__all__ = ["REPLAY_MODE", "CachedBuilder", "ReplayResult", "prebuild_chains", "replay_config",
           "run_replay", "session_clock"]

REPLAY_MODE = "replay"
PAPER_HASH = "paper"


@dataclass(frozen=True, slots=True)
class ReplayResult:
    start_equity: float
    end_equity: float
    equity: tuple[tuple[str, float], ...]
    positions_opened: int
    positions_closed: int
    open_short_legs_past_expiry: int
    halted: bool
    journal: Path

    def as_record(self) -> dict[str, Any]:
        return {"start_equity": self.start_equity, "end_equity": round(self.end_equity, 2),
                "equity": [list(x) for x in self.equity],
                "positions_opened": self.positions_opened,
                "positions_closed": self.positions_closed,
                "open_short_legs_past_expiry": self.open_short_legs_past_expiry,
                "halted": self.halted, "journal": str(self.journal)}


def replay_config(cfg: OptionsConfig, market: Market) -> OptionsConfig:
    """`cfg` met het universum van de markt (en de halve dagen van de kalender)."""
    names = tuple(sorted(market.bars))
    beta = {n: cfg.universe.beta_to_spy.get(n, 1.0) for n in names}
    share = tuple(n for n in cfg.universe.share_eligible if n in names)
    uni = cfg.universe.model_copy(update={"underlyings": names, "beta_to_spy": beta,
                                          "share_eligible": share})
    return cfg.model_copy(update={"universe": uni})


def session_clock(cal: TradingCalendar, day: date, step_seconds: float,
                  eod: datetime) -> list[datetime]:
    """De tijdstippen van één handelsdag: opening tot en met sluiting per stap, dan `eod`."""
    start, end = cal.session(day)
    out, t, step = [], start, timedelta(seconds=step_seconds)
    while t <= end:
        out.append(t)
        t += step
    if eod > out[-1]:
        out.append(eod)
    return out


class CachedBuilder:
    """Een `ChainBuilder` die vooraf gebouwde ketens teruggeeft (zelfde snapshot en carry)."""

    def __init__(self, inner: ChainBuilder, cache: dict[tuple[str, datetime], Chain]) -> None:
        self.inner, self.cache = inner, cache

    def build(self, underlying: str, spot: float, raw: Sequence[RawOption], asof: datetime,
              carry: Carry) -> Chain:
        hit = self.cache.get((underlying, asof))
        if hit is not None and hit.spot == spot and hit.carry == carry:
            return hit
        return self.inner.build(underlying, spot, raw, asof, carry)


def _build_one(job: tuple[ChainBuilder, Snapshot, Carry]) -> Chain:
    builder, s, carry = job
    return builder.build(s.underlying, s.spot, s.options, s.asof, carry)


def prebuild_chains(builder: ChainBuilder, market: Market,
                    workers: int) -> dict[tuple[str, datetime], Chain]:
    """Alle ketens van de markt, parallel over `workers` processen."""
    carries = {u: Carry(market.rate, dividends=e.dividends) for u, e in market.events.items()}
    jobs = [(builder, s, carries.get(s.underlying, Carry(market.rate)))
            for s in market.snapshots]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        chains = list(pool.map(_build_one, jobs, chunksize=4))
    return {(c.underlying, c.asof): c for c in chains}


async def run_replay(cfg: OptionsConfig, market: Market, out: Path,
                     workers: int = 1) -> ReplayResult:
    cfg = replay_config(cfg, market)
    feed = ReplayFeed(market)
    p = cfg.paper
    broker = PaperBroker(cfg.account.capital_usd, p.fill_at_concession,
                         p.commission_per_contract, PAPER_HASH)
    first = market.days[0]
    state = EngineState(REPLAY_MODE, PAPER_HASH,
                        datetime.combine(first, parse_time("00:00"), tzinfo=NY)
                        .astimezone(UTC).isoformat())
    state.iv_history = {k: list(v) for k, v in market.iv_history.items()}
    engine = Engine(cfg, feed, broker, Paths(out), state)
    if workers > 1:
        engine.builder = CachedBuilder(engine.builder,                 # type: ignore[assignment]
                                       prebuild_chains(engine.builder, market, workers))
    eod_t = parse_time(cfg.schedule.end_of_day)
    for day in market.days:
        eod = datetime.combine(day, eod_t, tzinfo=NY)
        for now in session_clock(engine.cal, day, cfg.execution.walk_interval_seconds, eod):
            await engine.tick(now)
    last = datetime.combine(market.days[-1], eod_t, tzinfo=NY)
    await engine.stop(last)
    past = sum(1 for pos in state.open_positions() for leg in pos.option_legs
               if leg.qty < 0 and leg.contract is not None and leg.contract.expiry <= market.days[-1])
    rows = engine.journal.read()
    res = ReplayResult(cfg.account.capital_usd, broker_equity(state, cfg),
                       tuple(state.equity),
                       sum(1 for r in rows if r["event"] == "position_opened"),
                       sum(1 for r in rows if r["event"] == "position_closed"),
                       past, engine.halted,
                       engine.paths.journal)
    (out / "summary.json").write_text(json.dumps(res.as_record(), indent=1, sort_keys=True),
                                      encoding="utf-8")
    return res


def broker_equity(state: EngineState, cfg: OptionsConfig) -> float:
    return state.equity[-1][1] if state.equity else cfg.account.capital_usd
