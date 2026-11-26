"""Eén waardering van een vanilla optie met QuantLib.

`value` bouwt per aanroep een nieuw QuantLib-object binnen `evaluation_date` en bewaart
niets buiten dat blok. Dat is trager dan objecten hergebruiken, maar kan niet lekken: een
waardering hangt alleen af van haar invoer.

Delta en gamma komen uit de engine waar die ze levert (analytisch Europees, binomiaal, FD).
BAW levert ze niet; dan zijn ze `None` en rekent `greeks` ze met spotbumps uit.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import date, datetime

import QuantLib as ql

from ...utils.failfast import DataContractError, require
from ..calendar import expiry_moment
from ..contracts import Right
from ..rates import Carry, CashDividend
from .engines import EngineSettings, Method
from .ql_env import (
    ScaledTime,
    evaluation_date,
    make_process,
    make_vol_process,
    scaled_time,
    to_ql_date,
)

__all__ = ["OptionInputs", "Valuation", "price", "value", "vol_pricer"]

_QL_TYPE = {Right.CALL: ql.Option.Call, Right.PUT: ql.Option.Put}


@dataclass(frozen=True, slots=True)
class OptionInputs:
    """Alles wat een waardering nodig heeft, behalve de volatiliteit en de engine."""

    spot: float
    strike: float
    right: Right
    expiry: date
    now: datetime
    carry: Carry

    def __post_init__(self) -> None:
        require(self.spot > 0, "spot moet positief zijn", DataContractError, spot=self.spot)
        require(self.strike > 0, "strike moet positief zijn", DataContractError,
                strike=self.strike)
        require(self.now.tzinfo is not None, "`now` moet een tijdzone hebben",
                DataContractError)

    @property
    def expired(self) -> bool:
        return self.now >= expiry_moment(self.expiry)

    @property
    def intrinsic(self) -> float:
        if self.right is Right.CALL:
            return max(self.spot - self.strike, 0.0)
        return max(self.strike - self.spot, 0.0)

    @property
    def discrete_dividends(self) -> tuple[CashDividend, ...]:
        """De dividenden die de pricer meeneemt: ex-datum in `(eval_date, expiry]`.

        Dit is HET venster voor de keuze van de engine. `eval_date` is vandaag, behalve op de
        expiratiedag zelf (dan gisteren); een dividend met ex-datum vandaag telt dan mee.
        """
        return self.carry.dividends_in(scaled_time(self.now, self.expiry).eval_date,
                                       self.expiry)

    def with_spot(self, spot: float) -> OptionInputs:
        return replace(self, spot=spot)

    def with_rate(self, rate: float) -> OptionInputs:
        return replace(self, carry=replace(self.carry, rate=rate))

    def with_now(self, now: datetime) -> OptionInputs:
        return replace(self, now=now)


@dataclass(frozen=True, slots=True)
class Valuation:
    npv: float
    delta: float | None
    gamma: float | None


def _engine(method: Method, process: ql.BlackScholesMertonProcess, settings: EngineSettings,
            dividends: ql.DividendVector | None) -> ql.PricingEngine:
    if method is Method.EUROPEAN:
        return ql.AnalyticEuropeanEngine(process)
    if method is Method.BAW:
        return ql.BaroneAdesiWhaleyApproximationEngine(process)
    if method is Method.FD:
        if dividends is None:
            return ql.FdBlackScholesVanillaEngine(process, settings.fd_time_steps,
                                                  settings.fd_grid_points)
        return ql.FdBlackScholesVanillaEngine(process, dividends, settings.fd_time_steps,
                                              settings.fd_grid_points)
    tree = "lr" if method is Method.BINOMIAL_LR else "crr"
    return ql.BinomialVanillaEngine(process, tree, settings.binomial_steps)


def _dividends(inp: OptionInputs, st: ScaledTime) -> ql.DividendVector | None:
    divs = inp.carry.dividends_in(st.eval_date, inp.expiry)    # == inp.discrete_dividends
    if not divs:
        return None
    return ql.DividendVector([to_ql_date(d.ex_date) for d in divs], [d.amount for d in divs])


def _option(inp: OptionInputs, method: Method, settings: EngineSettings, st: ScaledTime,
            today: ql.Date, process: ql.BlackScholesMertonProcess) -> ql.VanillaOption:
    dividends = _dividends(inp, st)
    require(dividends is None or method.supports_discrete_dividends,
            "een discreet dividend in de looptijd vraagt de FD-engine", DataContractError,
            method=method.value)
    expiry = to_ql_date(inp.expiry)
    exercise = (ql.AmericanExercise(today, expiry) if method.american
                else ql.EuropeanExercise(expiry))
    option = ql.VanillaOption(ql.PlainVanillaPayoff(_QL_TYPE[inp.right], inp.strike), exercise)
    option.setPricingEngine(_engine(method, process, settings, dividends))
    return option


def _evaluate(inp: OptionInputs, vol: float, method: Method, settings: EngineSettings,
              with_greeks: bool) -> Valuation:
    st = scaled_time(inp.now, inp.expiry)
    with evaluation_date(st.eval_date) as today:
        process = make_process(inp.spot, inp.carry.rate, inp.carry.div_yield, vol, st)
        option = _option(inp, method, settings, st, today, process)
        npv = float(option.NPV())
        if not with_greeks or method is Method.BAW:
            return Valuation(npv, None, None)
        return Valuation(npv, float(option.delta()), float(option.gamma()))


@contextmanager
def vol_pricer(inp: OptionInputs, method: Method,
               settings: EngineSettings = EngineSettings()) -> Iterator[Callable[[float], float]]:
    """Eén QuantLib-optie, herwaardeerd bij elke vol: `with vol_pricer(...) as f: f(0.25)`.

    Voor de IV-solver, die dezelfde optie tientallen keren bij een andere vol waardeert;
    zo wordt het object één keer gebouwd. Geldig alleen binnen het `with`-blok.
    """
    st = scaled_time(inp.now, inp.expiry)
    with evaluation_date(st.eval_date) as today:
        process, set_vol = make_vol_process(inp.spot, inp.carry.rate, inp.carry.div_yield, st)
        option = _option(inp, method, settings, st, today, process)

        def npv(vol: float) -> float:
            set_vol(vol)
            return float(option.NPV())

        yield npv


def value(inp: OptionInputs, vol: float, method: Method,
          settings: EngineSettings = EngineSettings()) -> Valuation:
    """Waarde, delta en gamma (per aandeel) bij volatiliteit `vol` met engine `method`.

    Bij BAW zijn delta en gamma `None` (de engine levert ze niet).
    """
    return _evaluate(inp, vol, method, settings, with_greeks=True)


def price(inp: OptionInputs, vol: float, method: Method,
          settings: EngineSettings = EngineSettings()) -> float:
    """Alleen de waarde. Voor de IV-solver en bumps: een binomiale boom bij een extreem
    lage vol levert geen delta, en die is daar ook niet nodig."""
    return _evaluate(inp, vol, method, settings, with_greeks=False).npv
