"""Gedeelde regels voor alle strategieën: expiratie- en strikekeuze, poorten, sizing en exits.

Detailontwerp 7.2, 7.3 en 7.5. Alles hier is een pure functie van de invoer: geen klok (alleen
`market.now`), geen I/O. Gelijke stand wordt altijd op een vaste manier gebroken, zodat twee
aanroepen met dezelfde invoer dezelfde uitvoer geven.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, time
from typing import Protocol

from ...schemas.options import OptionsConfig
from ..chain.builder import Chain, PricedOption
from ..contracts import Right
from ..iv_metrics import IVCondition, IVRuleSettings, check_iv_condition, iv_rule_settings
from ..positions import (
    Action,
    ActionKind,
    Book,
    Candidate,
    LegIntent,
    MarketState,
    StrategyPosition,
    Urgency,
)

__all__ = ["Rules", "Strategy", "close", "dte", "entry_blockers", "expiry_close_due",
           "iv_ok", "otm_step", "parse_time", "pick_expiry", "rules_from_config", "size",
           "strike_at_delta", "strike_nearest"]


class Strategy(Protocol):
    name: str

    def scan(self, market: MarketState, book: Book) -> list[Candidate]: ...

    def manage(self, pos: StrategyPosition, market: MarketState) -> list[Action]: ...


def parse_time(hhmm: str) -> time:
    h, m = hhmm.split(":")
    return time(int(h), int(m))


@dataclass(frozen=True, slots=True)
class Rules:
    """De gedeelde instellingen uit config, één keer omgezet."""

    iv: IVRuleSettings
    close_before_expiry: time = time(15, 0)
    earnings_blackout: bool = True
    max_calendar_data_age_days: int = 1
    max_loss_frac_capital: float = 0.02
    max_loss_per_underlying_frac: float = 0.05


def rules_from_config(cfg: OptionsConfig) -> Rules:
    return Rules(iv_rule_settings(cfg.iv_metrics),
                 parse_time(cfg.schedule.close_before_expiry_time),
                 cfg.filters.earnings_blackout, cfg.filters.max_calendar_data_age_days,
                 cfg.sizing.max_loss_frac_capital, cfg.risk.max_loss_per_underlying_frac)


def dte(expiry: date, today: date) -> int:
    """Kalenderdagen tot de expiratie."""
    return (expiry - today).days


def pick_expiry(chain: Chain, lo: int, hi: int, today: date) -> date | None:
    """De expiratie met DTE in `[lo, hi]` die het dichtst bij het midden ligt; gelijk: de
    vroegste."""
    mid = (lo + hi) / 2
    inside = [e for e in chain.expiries() if lo <= dte(e, today) <= hi]
    return min(inside, key=lambda e: (abs(dte(e, today) - mid), e), default=None)


def _otm_distance(o: PricedOption, spot: float) -> float:
    k = o.contract.strike
    return spot - k if o.contract.right is Right.PUT else k - spot


def strike_at_delta(chain: Chain, expiry: date, right: Right, target: float,
                    band: tuple[float, float] | None = None) -> PricedOption | None:
    """Het contract met `| |delta| − target |` minimaal (binnen `band`, als gegeven); gelijk:
    het verste OTM."""
    rows = [o for o in chain.by_expiry(expiry, right) if o.delta is not None
            and (band is None or band[0] <= abs(o.delta) <= band[1])]
    return min(rows, key=lambda o: (abs(abs(o.delta) - target),  # type: ignore[arg-type]
                                    -_otm_distance(o, chain.spot)), default=None)


def strike_nearest(chain: Chain, expiry: date, right: Right, level: float) -> PricedOption | None:
    """Het contract met de strike het dichtst bij `level`; gelijk: de laagste strike."""
    rows = chain.by_expiry(expiry, right)
    return min(rows, key=lambda o: (abs(o.contract.strike - level), o.contract.strike),
               default=None)


def otm_step(chain: Chain, short: PricedOption, n: int) -> PricedOption | None:
    """Het contract `n` strikes verder OTM dan `short`, in dezelfde expiratie en kant."""
    c = short.contract
    rows = chain.by_expiry(c.expiry, c.right)
    strikes = [o.contract.strike for o in rows]
    i = strikes.index(c.strike)
    j = i - n if c.right is Right.PUT else i + n
    return rows[j] if 0 <= j < len(rows) else None


def iv_ok(market: MarketState, underlying: str, cond: IVCondition,
          rules: Rules) -> tuple[bool, str]:
    """(voldaan, spoorregel). De spoorregel zegt of de vervangende regel is gebruikt."""
    m = market.iv.get(underlying)
    if m is None:
        return False, "iv:none"
    ok, fallback = check_iv_condition(m, cond, rules.iv)
    return ok, f"iv:{cond.value}:{'fallback' if fallback else 'ivr'}:{'ok' if ok else 'no'}"


def entry_blockers(market: MarketState, underlying: str, until: date, rules: Rules) -> list[str]:
    """Redenen om op deze onderliggende niets te openen dat tot `until` loopt (leeg: geen)."""
    ev = market.events.get(underlying)
    if ev is None or not ev.usable(market.today, rules.max_calendar_data_age_days):
        return ["events:stale"]
    if rules.earnings_blackout and ev.earnings_between(market.today, until):
        return ["events:earnings_in_term"]
    return []


def size(max_loss_per_unit: float, market: MarketState, book: Book, underlying: str,
         rules: Rules) -> int:
    """Eenheden volgens 7.3: het kleinste van het budget per positie en de ruimte op de
    onderliggende, gedeeld door het maximale verlies per eenheid."""
    if not max_loss_per_unit > 0 or math.isinf(max_loss_per_unit):
        return 0
    room = rules.max_loss_per_underlying_frac * market.capital_usd - book.open_max_loss(underlying)
    budget = min(rules.max_loss_frac_capital * market.capital_usd, room)
    return max(math.floor(budget / max_loss_per_unit), 0)


def expiry_close_due(expiry: date, now: datetime, market: MarketState, rules: Rules) -> bool:
    """Op of na de handelsdag vóór `expiry`, vanaf `close_before_expiry` (detailontwerp 7.5)."""
    last = market.calendar.previous_trading_day(expiry)
    return now.date() > last or (now.date() == last and now.time() >= rules.close_before_expiry)


def close(pos: StrategyPosition, urgency: Urgency, reason: str,
          legs: Sequence[LegIntent] | None = None) -> Action:
    return Action(ActionKind.CLOSE if legs is None else ActionKind.CLOSE_LEG, pos.position_id,
                  tuple(pos.intents() if legs is None else legs), urgency, reason)
