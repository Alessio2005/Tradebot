"""Volatiliteitsregime per onderliggende: de verwachte RV, de volatility risk premium en het regime.

Drie vragen, elk met een vaste regel (geen training, geen fit op historische data):

1. **Hoeveel volatiliteit verwachten we?** `forecast_rv` mengt de gerealiseerde Yang-Zhang-vol
   over een kort, middel en lang venster (standaard 5, 20 en 60 handelsdagen) met VASTE
   gewichten, in variantie. Dat is de vorm van het HAR-RV-model (Corsi 2009) zonder de
   regressie: de gewichten staan in de config en worden nergens geschat. Ontbreekt het lange
   venster (te weinig dagen), dan tellen de andere twee, met hun gewichten herschaald. Zonder
   het middelste venster is er geen verwachting.
2. **Is de optiepremie aantrekkelijk?** De volatility risk premium (VRP) is de ATM-IV(30)
   tegenover die verwachting: als verhouding (`iv / rv`) en als verschil in volpunten.
3. **Welk regime is het?** `classify` loopt de regels in deze volgorde af; de eerste die geldt,
   beslist:

   | Regime    | Regel                                                                  |
   |-----------|------------------------------------------------------------------------|
   | UNKNOWN   | geen ATM-IV of geen RV-verwachting                                     |
   | STRESS    | backwardation (IV60 − IV30 ≤ `stress_term_slope`) of een vol-schok       |
   |           | (RV kort / RV middel ≥ `stress_rv_expansion`)                          |
   | RICH      | IV / RV ≥ `rich_vrp_ratio`, IV − RV ≥ `rich_min_vrp_points`, en geen    |
   |           | inverse termijnstructuur (als die bekend is: helling ≥ `rich_min_term_slope`) |
   | CHEAP     | IV / RV < `cheap_vrp_ratio`                                            |
   | NORMAL    | al het andere                                                          |

Welke strategieën in welk regime nieuwe posities mogen openen, staat in de config
(`vol_regime.allowed`). Het regime raakt ALLEEN nieuwe entries: sluiten, stops, rolls en
bijsturingen gaan altijd door. STRESS op de marktonderliggende (standaard SPY) geldt voor het
hele boek (`effective_regimes`).
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

import pandas as pd

from ..schemas.options import VolRegimeConfig
from .iv_metrics import realized_vol

__all__ = ["RVForecast", "RegimeSettings", "VolRegime", "VolState", "classify",
           "effective_regimes", "forecast_rv", "regime_settings"]


class VolRegime(StrEnum):
    STRESS = "stress"      # vol-schok of backwardation: geen nieuwe short premium
    RICH = "rich"          # IV ruim boven de verwachte RV: premie verkopen
    NORMAL = "normal"
    CHEAP = "cheap"        # IV onder de verwachte RV: premie kopen
    UNKNOWN = "unknown"    # te weinig data: niets nieuws openen


@dataclass(frozen=True, slots=True)
class RegimeSettings:
    rv_windows: tuple[int, int, int] = (5, 20, 60)
    rv_weights: tuple[float, float, float] = (0.2, 0.5, 0.3)
    stress_term_slope: float = -0.02
    stress_rv_expansion: float = 1.5
    rich_vrp_ratio: float = 1.15
    rich_min_vrp_points: float = 0.02
    rich_min_term_slope: float = -0.005
    cheap_vrp_ratio: float = 0.95


def regime_settings(cfg: VolRegimeConfig) -> RegimeSettings:
    return RegimeSettings(cfg.rv_windows, cfg.rv_weights, cfg.stress_term_slope,
                          cfg.stress_rv_expansion, cfg.rich_vrp_ratio, cfg.rich_min_vrp_points,
                          cfg.rich_min_term_slope, cfg.cheap_vrp_ratio)


@dataclass(frozen=True, slots=True)
class RVForecast:
    """Geannualiseerde RV per venster en de gemengde verwachting (alle als fractie, 0,20 = 20 %)."""

    rv_short: float
    rv_mid: float
    rv_long: float | None
    forecast: float

    @property
    def expansion(self) -> float:
        """RV kort / RV middel: boven 1 loopt de vol op."""
        return self.rv_short / self.rv_mid


def forecast_rv(daily_bars: pd.DataFrame, s: RegimeSettings) -> RVForecast | None:
    """De vaste HAR-vormige menging in variantie; None zonder het korte of middelste venster."""
    short_w, mid_w, long_w = s.rv_windows
    rv_s, rv_m = realized_vol(daily_bars, short_w), realized_vol(daily_bars, mid_w)
    if rv_s is None or rv_m is None:
        return None
    rv_l = realized_vol(daily_bars, long_w)
    parts = [(s.rv_weights[0], rv_s), (s.rv_weights[1], rv_m)]
    if rv_l is not None:
        parts.append((s.rv_weights[2], rv_l))
    var = sum(w * v * v for w, v in parts) / sum(w for w, _ in parts)
    return RVForecast(rv_s, rv_m, rv_l, math.sqrt(var))


@dataclass(frozen=True, slots=True)
class VolState:
    regime: VolRegime
    atm_iv: float | None
    rv: RVForecast | None
    term_slope: float | None
    reason: str

    @property
    def vrp_ratio(self) -> float | None:
        return None if self.atm_iv is None or self.rv is None else self.atm_iv / self.rv.forecast

    @property
    def vrp_points(self) -> float | None:
        """IV − verwachte RV, als fractie (0,03 = 3 volpunten)."""
        return None if self.atm_iv is None or self.rv is None else self.atm_iv - self.rv.forecast

    def as_event(self) -> dict[str, object]:
        def r(x: float | None) -> float | None:
            return None if x is None else round(x, 4)
        return {"regime": self.regime.value, "reason": self.reason, "atm_iv": r(self.atm_iv),
                "rv_forecast": r(self.rv.forecast if self.rv else None),
                "rv_expansion": r(self.rv.expansion if self.rv else None),
                "vrp_ratio": r(self.vrp_ratio), "vrp_points": r(self.vrp_points),
                "term_slope": r(self.term_slope)}


def classify(atm_iv: float | None, rv: RVForecast | None, term_slope: float | None,
             s: RegimeSettings) -> VolState:
    """Het regime volgens de tabel in de moduledocstring; de reden zegt welke regel besliste."""
    def state(regime: VolRegime, reason: str) -> VolState:
        return VolState(regime, atm_iv, rv, term_slope, reason)

    if atm_iv is None or rv is None:
        return state(VolRegime.UNKNOWN, "no_iv" if atm_iv is None else "no_rv")
    if term_slope is not None and term_slope <= s.stress_term_slope:
        return state(VolRegime.STRESS, "backwardation")
    if rv.expansion >= s.stress_rv_expansion:
        return state(VolRegime.STRESS, "rv_expansion")
    ratio, points = atm_iv / rv.forecast, atm_iv - rv.forecast
    contango = term_slope is None or term_slope >= s.rich_min_term_slope
    if ratio >= s.rich_vrp_ratio and points >= s.rich_min_vrp_points and contango:
        return state(VolRegime.RICH, "vrp_rich")
    if ratio < s.cheap_vrp_ratio:
        return state(VolRegime.CHEAP, "vrp_cheap")
    return state(VolRegime.NORMAL, "vrp_normal")


def effective_regimes(states: Mapping[str, VolState],
                      market_underlying: str | None) -> dict[str, VolRegime]:
    """Het regime per onderliggende, met STRESS op de marktonderliggende voor iedereen."""
    regimes = {u: st.regime for u, st in states.items()}
    market = states.get(market_underlying) if market_underlying is not None else None
    if market is not None and market.regime is VolRegime.STRESS:
        return dict.fromkeys(regimes, VolRegime.STRESS)
    return regimes
