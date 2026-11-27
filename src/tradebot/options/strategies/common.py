"""Gedeelde regels voor alle strategieën: expiratie- en strikekeuze, poorten, sizing en exits.

Detailontwerp 7.2, 7.3 en 7.5. Alles hier is een pure functie van de invoer: geen klok (alleen
`market.now`), geen I/O. Gelijke stand wordt altijd op een vaste manier gebroken, zodat twee
aanroepen met dezelfde invoer dezelfde uitvoer geven.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
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
    position_greek,
)
from ..risk.gamma import sigma_1d

__all__ = ["Rules", "Strategy", "close", "drawdown_factor", "dte", "entry_blockers",
           "expiry_close_due", "iv_ok", "otm_step", "parse_time", "pick_expiry",
           "rules_from_config", "sigma_unit_1d", "size", "strike_at_delta", "strike_nearest",
           "vrp_gate"]


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
    daily_risk_frac: float | None = None              # None: geen sizing op dag-σ
    vol_of_vol_daily_points: float = 1.0
    drawdown_brake: tuple[float, float] | None = (0.05, 0.10)


def rules_from_config(cfg: OptionsConfig) -> Rules:
    s = cfg.sizing
    return Rules(iv_rule_settings(cfg.iv_metrics),
                 parse_time(cfg.schedule.close_before_expiry_time),
                 cfg.filters.earnings_blackout, cfg.filters.max_calendar_data_age_days,
                 s.max_loss_frac_capital, cfg.risk.max_loss_per_underlying_frac,
                 s.daily_risk_frac, s.vol_of_vol_daily_points, s.drawdown_brake)


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


def vrp_gate(market: MarketState, underlying: str, expiry: date,
             legs: Iterable[LegIntent], min_ratio: float | None) -> tuple[bool, tuple[str, ...]]:
    """VRP-poort per looptijd voor premie verkopen: (toegelaten, spoorregels).

    Uit (`min_ratio` None): altijd toegelaten, geen spoor. Aan: de gemiddelde IV van de short
    legs (mid-IV, alleen status OK) gedeeld door σ(h) van de volstate, h = de kalenderdagen van
    `market.today` tot `expiry`, moet ≥ `min_ratio` zijn (spoorregel `vrp_h:<ratio>`). Een
    ontbrekende IV, volstate of RV-verwachting weigert de kandidaat.
    """
    if min_ratio is None:
        return True, ()
    chain, vol = market.chains.get(underlying), market.vol.get(underlying)
    ivs = []
    for leg in legs:
        c = leg.contract
        o = chain.find(c.expiry, c.strike, c.right) if chain is not None and leg.ratio < 0 else None
        if o is not None and o.iv is not None:
            ivs.append(o.iv)
    vrp = (vol.vrp_at(sum(ivs) / len(ivs), dte(expiry, market.today))
           if vol is not None and ivs else None)
    if vrp is None or vrp[0] < min_ratio:
        return False, ()
    return True, (f"vrp_h:{vrp[0]:.3f}",)


def entry_blockers(market: MarketState, underlying: str, until: date, rules: Rules) -> list[str]:
    """Redenen om op deze onderliggende niets te openen dat tot `until` loopt (leeg: geen)."""
    ev = market.events.get(underlying)
    if ev is None or not ev.usable(market.today, rules.max_calendar_data_age_days):
        return ["events:stale"]
    if rules.earnings_blackout and ev.earnings_between(market.today, until):
        return ["events:earnings_in_term"]
    return []


def sigma_unit_1d(market: MarketState, underlying: str, legs: Sequence[LegIntent],
                  vol_of_vol_points: float) -> float | None:
    """De verwachte 1-dags P&L-σ van één eenheid in USD; None zonder Greeks of RV-verwachting.

    ```text
    σ_unit = √( (Δ·S·σ_1d)² + 2·(½·Γ·S²·σ_1d²)² + (V·volvol_1d)² )
    ```

    Δ, Γ en V per eenheid uit de keten (`position_greek`, × 100; V in USD per volpunt),
    σ_1d uit `risk.gamma.sigma_1d` en volvol_1d in volpunten per dag. De gamma-term is de σ van
    ½·Γ·(ΔS)² bij een normale ΔS: Var = 2·(½·Γ·(S·σ_1d)²)².
    """
    chain, vol = market.chains.get(underlying), market.vol.get(underlying)
    if not legs or chain is None or vol is None or vol.rv is None:
        return None
    delta, gamma, vega = (position_greek(chain, legs, g) for g in ("delta", "gamma", "vega"))
    if delta is None or gamma is None or vega is None:
        return None
    s, sig = chain.spot, sigma_1d(vol.rv)
    return math.sqrt((delta * s * sig) ** 2 + 2.0 * (0.5 * gamma * s * s * sig * sig) ** 2
                     + (vega * vol_of_vol_points) ** 2)


def drawdown_factor(market: MarketState, brake: tuple[float, float] | None) -> float:
    """De factor op het aantal eenheden: 1 tot drawdown `brake[0]`, dan lineair naar 0,5 bij
    `brake[1]` en daarna 0,5. Zonder rem of zonder equity-gegevens: 1.

    Drawdown = (piek − equity) / piek, met de piek uit `EngineState.equity` (de engine geeft
    die en de lopende equity mee in `MarketState`)."""
    peak, equity = market.equity_peak_usd, market.equity_usd
    if brake is None or peak is None or equity is None or not peak > 0:
        return 1.0
    lo, hi = brake
    dd = (peak - equity) / peak
    return 1.0 - 0.5 * min(max((dd - lo) / (hi - lo), 0.0), 1.0)


def size(max_loss_per_unit: float, market: MarketState, book: Book, underlying: str,
         rules: Rules, legs: Sequence[LegIntent]) -> int:
    """Eenheden volgens 7.3: het kleinste van het budget per positie en de ruimte op de
    onderliggende, gedeeld door het maximale verlies per eenheid.

    Met `daily_risk_frac` (E1) ook hoogstens `floor(daily_risk_frac × kapitaal / σ_unit)`
    (`sigma_unit_1d` van `legs`); ontbreken de Greeks of de RV-verwachting, dan 0. Daarna maal
    de drawdown-rem (`drawdown_factor`), naar beneden afgerond en minimaal 0. Zonder
    `daily_risk_frac` en zonder drawdown is de uitkomst die van de regel van 7.3 alleen.
    """
    if not max_loss_per_unit > 0 or math.isinf(max_loss_per_unit):
        return 0
    room = rules.max_loss_per_underlying_frac * market.capital_usd - book.open_max_loss(underlying)
    budget = min(rules.max_loss_frac_capital * market.capital_usd, room)
    units = max(math.floor(budget / max_loss_per_unit), 0)
    if rules.daily_risk_frac is not None:
        sigma = sigma_unit_1d(market, underlying, legs, rules.vol_of_vol_daily_points)
        if sigma is None:
            return 0
        if sigma > 0:
            units = min(units, _floor(rules.daily_risk_frac * market.capital_usd / sigma))
    factor = drawdown_factor(market, rules.drawdown_brake)
    return units if factor == 1.0 else max(_floor(units * factor), 0)


_FLOOR_EPS = 1e-9


def _floor(x: float) -> int:
    """Naar beneden afronden, maar een getal dat door float-ruis net onder een geheel getal
    valt (7,9999999999 uit 10 × 0,7999999999) telt als dat gehele getal."""
    return math.floor(x + _FLOOR_EPS)


def expiry_close_due(expiry: date, now: datetime, market: MarketState, rules: Rules) -> bool:
    """Op of na de handelsdag vóór `expiry`, vanaf `close_before_expiry` (detailontwerp 7.5)."""
    last = market.calendar.previous_trading_day(expiry)
    return now.date() > last or (now.date() == last and now.time() >= rules.close_before_expiry)


def close(pos: StrategyPosition, urgency: Urgency, reason: str,
          legs: Sequence[LegIntent] | None = None) -> Action:
    return Action(ActionKind.CLOSE if legs is None else ActionKind.CLOSE_LEG, pos.position_id,
                  tuple(pos.intents() if legs is None else legs), urgency, reason)
