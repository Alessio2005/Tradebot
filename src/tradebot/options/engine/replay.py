"""Replay: de engine op een opgeslagen markt met een virtuele klok (detailontwerp 12.2).

De klok loopt per handelsdag van de opening tot de sluiting in stappen van
`execution.walk_interval_seconds`, plus één ronde op `schedule.end_of_day`. Er is geen echte
tijd en geen netwerk: de broker is de `PaperBroker`, het geld is fictief.

Het universum wordt de namen uit de markt, met beta 1,0 voor elke naam die de config niet kent:
een synthetische markt heeft geen SPY-beta.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from ...schemas.options import OptionsConfig
from ..calendar import NY, TradingCalendar
from ..data.feed import ReplayFeed
from ..data.store import Market
from ..execution.paper import PaperBroker
from ..strategies.common import parse_time
from .service import Engine, Paths
from .state import EngineState

__all__ = ["REPLAY_MODE", "ReplayResult", "replay_config", "run_replay", "session_clock"]

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


async def run_replay(cfg: OptionsConfig, market: Market, out: Path) -> ReplayResult:
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
