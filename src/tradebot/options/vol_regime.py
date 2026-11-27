"""Volatiliteitsregime per onderliggende: de verwachte RV, de volatility risk premium en het regime.

Drie vragen, elk met een vaste regel (geen training, geen fit op historische data):

1. **Hoeveel volatiliteit verwachten we?** `forecast_rv` mengt de gerealiseerde Yang-Zhang-vol
   over een kort, middel en lang venster (standaard 5, 20 en 60 handelsdagen) met VASTE
   gewichten, in variantie. Dat is de vorm van het HAR-RV-model (Corsi 2009) zonder de
   regressie: de gewichten staan in de config en worden nergens geschat. Ontbreekt het lange
   venster (te weinig dagen), dan tellen de andere twee, met hun gewichten herschaald. Zonder
   het middelste venster is er geen verwachting.
2. **Is de optiepremie aantrekkelijk?** De volatility risk premium (VRP) is de ATM-IV(30)
   tegenover de verwachte RV OP DIE HORIZON, σ(30 d) = `RVForecast.at_horizon(tenor_days)`
   (niet de blend zelf, die bij h → 0 hoort): als verhouding (`iv / σ`) en als verschil in
   volpunten. Dezelfde σ(h) gebruiken `vrp_at` en `theta_cover`, dus regime en selectie
   spreken elkaar niet tegen.
3. **Welk regime is het?** `classify` loopt de regels in deze volgorde af; de eerste die geldt,
   beslist:

   | Regime    | Regel                                                                  |
   |-----------|------------------------------------------------------------------------|
   | UNKNOWN   | geen ATM-IV of geen RV-verwachting                                     |
   | STRESS    | backwardation (IV60 − IV30 ≤ `stress_term_slope`) of een vol-schok       |
   |           | (RV kort / RV middel ≥ `stress_rv_expansion`)                          |
   | RICH      | IV / σ(30) ≥ `rich_vrp_ratio`, IV − σ(30) ≥ `rich_min_vrp_points`, en geen    |
   |           | inverse termijnstructuur (als die bekend is: helling ≥ `rich_min_term_slope`) |
   | CHEAP     | IV / σ(30) < `cheap_vrp_ratio`                                          |
   | NORMAL    | al het andere                                                          |

Welke strategieën in welk regime nieuwe posities mogen openen, staat in de config
(`vol_regime.allowed`). Het regime raakt verder alleen nieuwe entries: sluiten, stops, rolls en
bijsturingen gaan altijd door. STRESS op de marktonderliggende (standaard SPY) geldt voor het
hele boek (`effective_regimes`).

Eén uitzondering, het enige effect op bestaande posities: is het effectieve regime van de
onderliggende STRESS, dan sluit de engine elke undefined-risk positie (de short
straddle/strangle, `max_loss` oneindig) op die onderliggende met `Urgency.STOP`, reden
`regime_stress` (`Engine._manage`, vóór de regels van de strategie en ook in HALT). Daarom
laat `selection.admit` in STRESS ook nooit een undefined-risk entry toe, wat
`vol_regime.enabled` of `allowed.stress` ook zeggen; beide lezen hetzelfde effectieve regime.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

import pandas as pd

from ..schemas.options import VolRegimeConfig
from .iv_metrics import ewma_vol, realized_vol

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
    ewma_lambda: float = 0.94
    ewma_weight: float = 0.0
    mean_reversion_kappa: float = 4.0
    tenor_days: float = 30.0           # de looptijd van de ATM-IV: σ(h) voor de VRP is op deze h


def regime_settings(cfg: VolRegimeConfig, tenor_days: float = 30.0) -> RegimeSettings:
    """`tenor_days`: `iv_metrics.atm_tenor_days` (de looptijd waarop de ATM-IV is gemeten)."""
    return RegimeSettings(cfg.rv_windows, cfg.rv_weights, cfg.stress_term_slope,
                          cfg.stress_rv_expansion, cfg.rich_vrp_ratio, cfg.rich_min_vrp_points,
                          cfg.rich_min_term_slope, cfg.cheap_vrp_ratio, cfg.ewma_lambda,
                          cfg.ewma_weight, cfg.mean_reversion_kappa, tenor_days)


@dataclass(frozen=True, slots=True)
class RVForecast:
    """Geannualiseerde RV per venster en de gemengde verwachting (alle als fractie, 0,20 = 20 %)."""

    rv_short: float
    rv_mid: float
    rv_long: float | None
    forecast: float
    kappa: float = 4.0                 # per jaar: snelheid van de mean reversion in `at_horizon`

    @property
    def expansion(self) -> float:
        """RV kort / RV middel: boven 1 loopt de vol op."""
        return self.rv_short / self.rv_mid

    def at_horizon(self, days: float) -> float:
        """σ(h) voor `days` kalenderdagen (h = days/365 jaar); bij `days` ≤ 0 de blend zelf."""
        if days <= 0:
            return self.forecast
        h = days / 365.0
        long_run = self.rv_long if self.rv_long is not None else self.rv_mid
        weight = -math.expm1(-self.kappa * h) / (self.kappa * h)
        var = long_run ** 2 + (self.forecast ** 2 - long_run ** 2) * weight
        return math.sqrt(var)


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
    rv_e = ewma_vol(daily_bars, s.ewma_lambda) if s.ewma_weight > 0 else None
    if rv_e is not None:
        scale = (1.0 - s.ewma_weight) / sum(w for w, _ in parts)
        parts = [(w * scale, v) for w, v in parts] + [(s.ewma_weight, rv_e)]
    var = sum(w * v * v for w, v in parts) / sum(w for w, _ in parts)
    return RVForecast(rv_s, rv_m, rv_l, math.sqrt(var), s.mean_reversion_kappa)


@dataclass(frozen=True, slots=True)
class VolState:
    regime: VolRegime
    atm_iv: float | None
    rv: RVForecast | None
    term_slope: float | None
    reason: str
    tenor_days: float = 30.0

    @property
    def vrp_ratio(self) -> float | None:
        """ATM-IV / σ(`tenor_days`)."""
        return (None if self.atm_iv is None or self.rv is None
                else self.atm_iv / self.rv.at_horizon(self.tenor_days))

    @property
    def vrp_points(self) -> float | None:
        """IV − σ(`tenor_days`), als fractie (0,03 = 3 volpunten)."""
        return (None if self.atm_iv is None or self.rv is None
                else self.atm_iv - self.rv.at_horizon(self.tenor_days))

    def vrp_at(self, iv: float, dte_days: float) -> tuple[float, float] | None:
        """(IV / σ(h), IV − σ(h)) voor een looptijd van `dte_days` kalenderdagen; None zonder RV."""
        if self.rv is None:
            return None
        sigma = self.rv.at_horizon(dte_days)
        return iv / sigma, iv - sigma

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
        return VolState(regime, atm_iv, rv, term_slope, reason, s.tenor_days)

    if atm_iv is None or rv is None:
        return state(VolRegime.UNKNOWN, "no_iv" if atm_iv is None else "no_rv")
    if term_slope is not None and term_slope <= s.stress_term_slope:
        return state(VolRegime.STRESS, "backwardation")
    if rv.expansion >= s.stress_rv_expansion:
        return state(VolRegime.STRESS, "rv_expansion")
    sigma = rv.at_horizon(s.tenor_days)
    ratio, points = atm_iv / sigma, atm_iv - sigma
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
