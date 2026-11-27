"""Volatiliteitsoppervlak per onderliggende: smile per expiratie, interpolatie en controles.

Dit is KALIBRATIE AAN DE HUIDIGE MARKT, geen training op historische data (detailontwerp 6.2).

* Forward per expiratie uit put-call-pariteit op de drie strikes rond spot (mediaan), zodat
  dividenden en borrow vanzelf meetellen: `F = K + (C − P)·e^{rT}`.
* Smile: OTM-IV's (puts onder de forward, calls erboven), als totale variantie
  `w = σ²T` tegen log-moneyness `k = ln(K/F)`, kubische spline; minstens 5 punten. Een punt
  telt alleen mee als de vega minstens `min_vega` (USD per volpunt per aandeel) is: quotes
  staan op centen, en bij een kleine vega is de IV van een afgeronde prijs ruis (een halve
  cent bij vega 0,01 is een half volpunt).
* Over de tijd: lineair in totale variantie bij gelijke `k`; buiten het bereik van de
  expiraties: vlakke vol van de dichtstbijzijnde expiratie.
* Controles: kalender (w stijgt in T op een k-raster, binnen het k-bereik dat beide
  expiraties met punten dekken) en butterfly (de dichtheid uit de
  callprijzen ≥ 0, prijzen via `ql.blackFormula`). Faalt een controle, dan is de status
  `ARBITRAGE` en gebruikt geen strategie het oppervlak.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum
from itertools import pairwise
from statistics import median

import numpy as np
import QuantLib as ql
from scipy.interpolate import CubicSpline

from ..calendar import expiry_moment, year_fraction
from ..chain.builder import Chain, PricedOption
from ..contracts import Right

__all__ = ["MIN_SMILE_POINTS", "MIN_VEGA", "Smile", "Surface", "SurfaceStatus", "build_surface",
           "forward", "skew_25d"]

MIN_SMILE_POINTS = 5
MIN_VEGA = 0.01
_K_GRID = np.linspace(-0.2, 0.2, 21)
_TOL = 1e-6


class SurfaceStatus(StrEnum):
    OK = "ok"
    ARBITRAGE = "arbitrage"
    INSUFFICIENT = "insufficient"


@dataclass(frozen=True, slots=True)
class Smile:
    expiry: date
    t: float
    forward: float
    discount: float
    k: tuple[float, ...]
    w: tuple[float, ...]
    #: de spline door (k, w), één keer gebouwd: de controles vragen hem tientallen keren
    _spline: CubicSpline = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_spline",
                           CubicSpline(np.asarray(self.k), np.asarray(self.w), bc_type="natural"))

    def total_variance(self, k: float) -> float:
        """Totale variantie bij log-moneyness `k`; buiten de punten vlak (geen extrapolatie)."""
        kk = float(np.clip(k, self.k[0], self.k[-1]))
        return float(self._spline(kk))

    def vol(self, k: float) -> float:
        return math.sqrt(max(self.total_variance(k), 0.0) / self.t)

    def call(self, strike: float) -> float:
        k = math.log(strike / self.forward)
        std = math.sqrt(max(self.total_variance(k), 1e-12))
        return float(ql.blackFormula(ql.Option.Call, strike, self.forward, std, self.discount))


@dataclass(frozen=True, slots=True)
class Surface:
    underlying: str
    status: SurfaceStatus
    smiles: tuple[Smile, ...]
    reason: str = ""

    def atm_vol(self, tenor_days: float) -> float | None:
        """ATM-vol (k = 0) op `tenor_days`, lineair in totale variantie over de tijd."""
        if self.status is not SurfaceStatus.OK:
            return None
        t = tenor_days / 365.0
        s = self.smiles
        if t <= s[0].t:
            return s[0].vol(0.0)
        if t >= s[-1].t:
            return s[-1].vol(0.0)
        i = next(j for j in range(1, len(s)) if s[j].t >= t)
        a, b = s[i - 1], s[i]
        x = (t - a.t) / (b.t - a.t)
        w = (1 - x) * a.total_variance(0.0) + x * b.total_variance(0.0)
        return math.sqrt(w / t)

    def term_slope(self, short_days: float = 30, long_days: float = 60) -> float | None:
        lo, hi = self.atm_vol(short_days), self.atm_vol(long_days)
        return None if lo is None or hi is None else hi - lo


def _forward(chain: Chain, expiry: date, t: float, discount: float) -> float | None:
    calls = {o.contract.strike: o for o in chain.by_expiry(expiry, Right.CALL)
             if o.quote.mid is not None}
    puts = {o.contract.strike: o for o in chain.by_expiry(expiry, Right.PUT)
            if o.quote.mid is not None}
    both = sorted(set(calls) & set(puts), key=lambda k: abs(k - chain.spot))[:3]
    if not both:
        return None
    fwd = [k + (calls[k].quote.mid - puts[k].quote.mid) / discount  # type: ignore[operator]
           for k in both]
    return float(median(fwd))


def forward(chain: Chain, expiry: date) -> float | None:
    """Forward van één expiratie uit put-call-pariteit (mediaan over drie strikes rond spot)."""
    t = year_fraction(chain.asof, expiry_moment(expiry))
    return _forward(chain, expiry, t, math.exp(-chain.carry.rate * t))


def _smile(chain: Chain, expiry: date, min_vega: float) -> Smile | None:
    t = year_fraction(chain.asof, expiry_moment(expiry))
    discount = math.exp(-chain.carry.rate * t)
    fwd = _forward(chain, expiry, t, discount)
    if fwd is None or fwd <= 0:
        return None
    otm: list[PricedOption] = [
        o for o in chain.by_expiry(expiry)
        if o.iv is not None and o.greeks is not None and o.greeks.vega >= min_vega
        and ((o.contract.right is Right.PUT and o.contract.strike < fwd)
             or (o.contract.right is Right.CALL and o.contract.strike >= fwd))]
    pts = sorted({math.log(o.contract.strike / fwd): o.iv**2 * t  # type: ignore[operator]
                  for o in otm}.items())
    if len(pts) < MIN_SMILE_POINTS:
        return None
    return Smile(expiry, t, fwd, discount, tuple(p[0] for p in pts), tuple(p[1] for p in pts))


def _calendar_ok(smiles: tuple[Smile, ...]) -> bool:
    for a, b in pairwise(smiles):
        lo, hi = max(a.k[0], b.k[0]), min(a.k[-1], b.k[-1])
        for k in _K_GRID[(_K_GRID >= lo) & (_K_GRID <= hi)]:
            if b.total_variance(float(k)) < a.total_variance(float(k)) - _TOL:
                return False
    return True


def _butterfly_ok(smile: Smile) -> bool:
    strikes = smile.forward * np.exp(np.linspace(smile.k[0], smile.k[-1], 41))
    prices = np.array([smile.call(float(k)) for k in strikes])
    dk = np.diff(strikes)
    slopes = np.diff(prices) / dk
    density = np.diff(slopes) / (0.5 * (dk[1:] + dk[:-1]))
    return bool(np.all(density >= -_TOL))


def build_surface(chain: Chain, min_vega: float = MIN_VEGA) -> Surface:
    """Oppervlak uit een (bij voorkeur al gefilterde) keten; zie de moduledocstring."""
    smiles = tuple(s for s in (_smile(chain, e, min_vega) for e in chain.expiries())
                   if s is not None)
    if not smiles:
        return Surface(chain.underlying, SurfaceStatus.INSUFFICIENT, (), "geen bruikbare smile")
    if not _calendar_ok(smiles):
        return Surface(chain.underlying, SurfaceStatus.ARBITRAGE, smiles, "kalenderarbitrage")
    bad = [s.expiry.isoformat() for s in smiles if not _butterfly_ok(s)]
    if bad:
        return Surface(chain.underlying, SurfaceStatus.ARBITRAGE, smiles,
                       f"butterfly-arbitrage in {', '.join(bad)}")
    return Surface(chain.underlying, SurfaceStatus.OK, smiles)


def skew_25d(chain: Chain, expiry: date) -> float | None:
    """IV(put delta −0,25) − IV(call delta 0,25) in één expiratie, of None zonder data."""
    def nearest(right: Right, target: float) -> PricedOption | None:
        rows = [o for o in chain.by_expiry(expiry, right) if o.iv is not None
                and o.delta is not None]
        return min(rows, key=lambda o: abs(o.delta - target), default=None)  # type: ignore[operator]

    put, call = nearest(Right.PUT, -0.25), nearest(Right.CALL, 0.25)
    if put is None or call is None:
        return None
    return put.iv - call.iv  # type: ignore[operator]
