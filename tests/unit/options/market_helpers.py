"""Een kleine, echte markt voor de strategietests: BAW-geprijsde ketens rond een vaste klok."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from datetime import date, datetime
from functools import lru_cache

from tests.unit.options.helpers import NOW
from tradebot.options.calendar import TradingCalendar
from tradebot.options.chain.builder import Chain, ChainBuilder, PricedOption, RawOption
from tradebot.options.contracts import OptionContract, Quote, Right
from tradebot.options.events import Events
from tradebot.options.iv_metrics import IVMetrics, IVRStatus
from tradebot.options.positions import (
    MULTIPLIER,
    Candidate,
    Leg,
    MarketState,
    StrategyPosition,
    Trend,
)
from tradebot.options.pricing.engines import Method
from tradebot.options.pricing.implied_vol import IVResult, IVStatus
from tradebot.options.pricing.settings import pricer_settings
from tradebot.options.pricing.vanilla import OptionInputs, price
from tradebot.options.rates import Carry
from tradebot.options.units import GreeksPerShare
from tradebot.schemas.options import load_options_config

_BASE = load_options_config()
# De testmarkt heeft TST en TSTB; geef ze een beta van 1 zodat de risicopoort ze kent.
CFG = _BASE.model_copy(update={"universe": _BASE.universe.model_copy(update={
    "beta_to_spy": {**_BASE.universe.beta_to_spy, "TST": 1.0, "TSTB": 1.0}})})
# In-proces: elke hulpketen krijgt een eigen builder, een procespool per keten loont niet.
SETTINGS = replace(pricer_settings(CFG.pricing), workers=1)
CAL = TradingCalendar(CFG.universe.half_days)
# DTE vanaf NOW (2026-11-24): 24, 37, 52 en 87 dagen.
EXPIRIES = (date(2026, 12, 18), date(2026, 12, 31), date(2027, 1, 15), date(2027, 2, 19))
STRIKES = tuple(float(k) for k in range(70, 131))


def flat(v: float) -> Callable[[float, date], float]:
    return lambda k, e: v


def skewed(base: float = 0.25, slope: float = -0.3) -> Callable[[float, date], float]:
    """Vol stijgt naar lage strikes (aandelenskew), in `k = K/100 − 1`."""
    return lambda k, e: base + slope * (k / 100.0 - 1.0)


def make_chain(spot: float = 100.0, vol: Callable[[float, date], float] = flat(0.25),
               now: datetime = NOW, expiries: tuple[date, ...] = EXPIRIES,
               strikes: tuple[float, ...] = STRIKES, underlying: str = "TST",
               half: float = 0.01) -> Chain:
    carry = Carry(0.04)
    rows = []
    for e in expiries:
        if e <= now.date():
            continue
        for k in strikes:
            for r in (Right.PUT, Right.CALL):
                theo = price(OptionInputs(spot, k, r, e, now, carry), vol(k, e), Method.BAW)
                rows.append(RawOption(OptionContract(underlying, e, k, r),
                                      Quote(max(theo - half, 0.0), theo + half, now,
                                            volume=1000, open_interest=5000)))
    return ChainBuilder(SETTINGS, 1e9).build(underlying, spot, rows, now, carry)


@lru_cache(maxsize=32)
def cached_chain(spot: float = 100.0, base: float = 0.25, slope: float = 0.0,
                 now: datetime = NOW) -> Chain:
    return make_chain(spot, skewed(base, slope), now)


HIGH_IV = IVMetrics(0.25, 60.0, 70.0, IVRStatus.OK, 1.4, term_slope=0.01)
LOW_IV = IVMetrics(0.25, 10.0, 15.0, IVRStatus.OK, 0.8, term_slope=0.01)


def market(chain: Chain | None = None, iv: IVMetrics = HIGH_IV, trend: Trend = Trend.UP,
           events: Events | None = None, now: datetime | None = None,
           capital: float = 25_000.0) -> MarketState:
    chain = chain or cached_chain()
    und = chain.underlying
    now = now or chain.asof
    return MarketState(now, CAL, {und: chain}, {und: iv}, {und: trend},
                       {und: events or Events(und, now.date())}, capital)


def open_position(c: Candidate, opened_at: datetime = NOW, position_id: str = "p1",
                  **kw: object) -> StrategyPosition:
    """De positie die ontstaat als kandidaat `c` tegen mid vult."""
    chain = cached_chain()
    legs = []
    for leg in c.legs:
        k = leg.contract
        o = chain.find(k.expiry, k.strike, k.right)
        px = o.quote.mid if o is not None and o.quote.mid is not None else 0.0
        legs.append(Leg(k, c.underlying, leg.ratio * c.quantity, px))
    return StrategyPosition(position_id, c.strategy, c.underlying, tuple(legs), opened_at,
                            c.quantity, c.net_credit * MULTIPLIER * c.quantity,
                            min(c.max_loss, 1e12), variant=c.variant, **kw)  # type: ignore[arg-type]


def fake(strike: float, right: Right, expiry: date = EXPIRIES[1], delta: float | None = 0.3,
         mid: float = 1.0) -> PricedOption:
    """Een contract met opgegeven delta en mid, zonder pricing (voor de keuzeregels)."""
    g = GreeksPerShare(delta, 0.01, 0.1, -0.01, 0.01) if delta is not None else None
    res = IVResult(IVStatus.OK, 0.25, Method.BAW) if delta is not None else IVResult(
        IVStatus.NO_CONVERGENCE)
    return PricedOption(OptionContract("TST", expiry, strike, right),
                        Quote(mid - 0.01, mid + 0.01, NOW), res, IVResult(IVStatus.NO_QUOTE),
                        IVResult(IVStatus.NO_QUOTE), g, Method.BAW, False)


def fake_chain(*rows: PricedOption, spot: float = 100.0) -> Chain:
    return Chain("TST", spot, NOW, Carry(0.04), tuple(rows))
