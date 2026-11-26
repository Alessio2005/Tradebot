"""Implied volatility met expliciete foutcodes (detailontwerp 5.4).

Een IV-solver die bij een onmogelijke prijs een NaN of een standaardwaarde teruggeeft,
voedt de strategieën met verzonnen getallen. Hier krijgt elke mislukking een `IVStatus`, en
alleen `IVStatus.OK` draagt een waarde. Een contract zonder `OK` op de mid krijgt geen Greeks
en valt buiten elke strategie.

Algoritme per prijs (mid, bid of ask):

1. verlopen → `EXPIRED`; geen prijs → `NO_QUOTE`; prijs 0 → `NO_BID`;
2. onder de ondergrens min één tick → `BELOW_INTRINSIC`; boven de bovengrens →
   `ABOVE_UPPER_BOUND`. Ligt de prijs binnen één tick onder de ondergrens (afronding bij
   diep-ITM), dan geldt de ondergrens plus een halve tick;
3. Brent op `waarde(σ) − prijs` met de gevraagde engine, over `[min_vol, max_vol]`;
4. lukt dat niet: Brent met de terugvalengine (BAW voor een boom of FD, de binomiale boom
   voor BAW). Met een discreet dividend in de looptijd is er geen terugval: alleen FD kan dat;
5. lukt dat ook niet → `NO_CONVERGENCE`. Ligt de prijs buiten het bereik van het model bij
   `[min_vol, max_vol]`, dan is er geen terugval: een andere engine verandert dat niet.

De ondergrens is de intrinsieke waarde (Amerikaans) of de verdisconteerde intrinsieke
waarde op de forward (Europees). De bovengrens is de spot (call) of de strike (put).

Met een `bracket` (een smaller interval rond een startwaarde, bv. voor de dure FD-engine)
gaat stap 3 eerst over dat interval. Omsluit het geen wortel, weigert QuantLib een punt of
convergeert Brent niet, dan volgt het volle pad hierboven, met dezelfde terugvallogica. Omdat
de waarde stijgt in σ, ligt een wortel in het smalle interval ook in `[min_vol, max_vol]`;
het smalle interval verandert dus alleen het aantal waarderingen, niet de uitkomst (binnen
`accuracy`).

Brent op de eigen engine (en niet `VanillaOption.impliedVolatility`, die intern een andere
engine kiest) houdt de round-trip prijs → IV → prijs exact voor de engine die beslist.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

from scipy.optimize import brentq

from ..calendar import expiry_moment, year_fraction
from ..contracts import Right
from .engines import EngineSettings, Method
from .vanilla import OptionInputs, vol_pricer

__all__ = ["IVResult", "IVSettings", "IVStatus", "implied_vol"]


class IVStatus(StrEnum):
    OK = "ok"
    NO_QUOTE = "no_quote"
    NO_BID = "no_bid"
    BELOW_INTRINSIC = "below_intrinsic"
    ABOVE_UPPER_BOUND = "above_upper_bound"
    NO_CONVERGENCE = "no_convergence"
    EXPIRED = "expired"


@dataclass(frozen=True, slots=True)
class IVResult:
    status: IVStatus
    value: float | None = None
    method: Method | None = None


@dataclass(frozen=True, slots=True)
class IVSettings:
    accuracy: float = 1.0e-6
    max_evaluations: int = 200
    min_vol: float = 1.0e-2
    max_vol: float = 5.0
    tick: float = 0.01


def _lower_bound(inp: OptionInputs, method: Method) -> float:
    if method.american:
        return inp.intrinsic
    t = year_fraction(inp.now, expiry_moment(inp.expiry))
    fwd_spot = inp.spot * math.exp(-inp.carry.div_yield * t)
    pv_strike = inp.strike * math.exp(-inp.carry.rate * t)
    if inp.right is Right.CALL:
        return max(fwd_spot - pv_strike, 0.0)
    return max(pv_strike - fwd_spot, 0.0)


#: Ondergrenzen die de solver probeert als QuantLib de vorige weigert. Onder ongeveer 1 % vol
#: geven BAW ("forward + displacement must be positive") en de binomiale boom ("invalid up
#: probability") fouten; een aandelenoptie heeft nooit zo'n IV. Daarom is 1 % de standaard.
_LOWER_BRACKETS = (2.0e-2, 5.0e-2)


class _OutOfRange:
    """De prijs ligt buiten wat het model bij [min_vol, max_vol] kan geven (bv. een diep-ITM
    call onder S − K·e^{−rT}). Een andere engine verandert dat niet: geen terugval."""


_OUT_OF_RANGE = _OutOfRange()


def _solve(inp: OptionInputs, target: float, method: Method, engines: EngineSettings,
           cfg: IVSettings) -> float | _OutOfRange | None:
    with vol_pricer(inp, method, engines) as npv:
        def f(sigma: float) -> float:
            return npv(sigma) - target

        lo_vol, lo = cfg.min_vol, None
        for candidate in (cfg.min_vol, *(b for b in _LOWER_BRACKETS if b > cfg.min_vol)):
            try:
                lo, lo_vol = f(candidate), candidate
                break
            except RuntimeError:          # QuantLib weigert dit extreme punt; probeer hoger
                continue
        if lo is None:
            return None
        if lo > 0 or f(cfg.max_vol) < 0:
            return _OUT_OF_RANGE
        try:
            sigma, info = brentq(f, lo_vol, cfg.max_vol, xtol=cfg.accuracy,
                                 maxiter=cfg.max_evaluations, full_output=True, disp=False)
        except RuntimeError:              # QuantLib weigert een tussenpunt
            return None
        return float(sigma) if info.converged else None


def _solve_in(inp: OptionInputs, target: float, method: Method, engines: EngineSettings,
              cfg: IVSettings, bracket: tuple[float, float]) -> float | None:
    """Brent op `bracket`; `None` als dat geen wortel omsluit of QuantLib een punt weigert."""
    lo_vol, hi_vol = bracket
    with vol_pricer(inp, method, engines) as npv:
        known: dict[float, float] = {}

        def f(sigma: float) -> float:          # de randen niet twee keer waarderen
            if sigma in known:
                return known[sigma]
            return npv(sigma) - target

        try:
            known[lo_vol], known[hi_vol] = f(lo_vol), f(hi_vol)
            if known[lo_vol] > 0 or known[hi_vol] < 0:
                return None
            sigma, info = brentq(f, lo_vol, hi_vol, xtol=cfg.accuracy,
                                 maxiter=cfg.max_evaluations, full_output=True, disp=False)
        except RuntimeError:              # QuantLib weigert een punt: het volle pad beslist
            return None
        return float(sigma) if info.converged else None


def implied_vol(inp: OptionInputs, price: float | None, method: Method,
                engines: EngineSettings = EngineSettings(),
                cfg: IVSettings = IVSettings(), *,
                bracket: tuple[float, float] | None = None) -> IVResult:
    """De IV bij `price` met engine `method`, of een foutcode.

    `bracket` (binnen `[min_vol, max_vol]`) wordt eerst geprobeerd; zie de moduledocstring.
    """
    if inp.expired:
        return IVResult(IVStatus.EXPIRED)
    if price is None:
        return IVResult(IVStatus.NO_QUOTE)
    if price <= 0:
        return IVResult(IVStatus.NO_BID)
    lower = _lower_bound(inp, method)
    upper = inp.spot if inp.right is Right.CALL else inp.strike
    if price < lower - cfg.tick:
        return IVResult(IVStatus.BELOW_INTRINSIC)
    if price > upper:
        return IVResult(IVStatus.ABOVE_UPPER_BOUND)
    if price <= lower:
        price = lower + 0.5 * cfg.tick
    if bracket is not None:
        narrow = _solve_in(inp, price, method, engines, cfg, bracket)
        if narrow is not None:
            return IVResult(IVStatus.OK, narrow, method)
    sigma = _solve(inp, price, method, engines, cfg)
    if isinstance(sigma, float):
        return IVResult(IVStatus.OK, sigma, method)
    if sigma is None and not inp.discrete_dividends:
        fallback = Method.BINOMIAL_LR if method is Method.BAW else Method.BAW
        sigma = _solve(inp, price, fallback, engines, cfg)
        if isinstance(sigma, float):
            return IVResult(IVStatus.OK, sigma, fallback)
    return IVResult(IVStatus.NO_CONVERGENCE)
