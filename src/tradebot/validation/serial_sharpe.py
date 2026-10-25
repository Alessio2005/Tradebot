"""De Sharpe van een autogecorreleerde reeks: per periode geaggregeerd en Lo (2002).

WAAROM (audit 2026-10-24, SK-1)
===============================
`mean / sd · sqrt(365)` op dagrendementen veronderstelt dat die onafhankelijk zijn. Voor
het carryboek zijn ze dat niet: funding is persistent, dus de dagrendementen ook
(v8-T2: rho_1..3 = 0,49 / 0,55 / 0,45). Dan groeit de variantie over een jaar sneller dan
365 x de dagvariantie, en overschat sqrt(365) de jaarlijkse Sharpe: 5,05 gerapporteerd,
1,55 op maandrendementen, 1,08 op kwartaalrendementen.

De Lo-SE in `inference.sharpe_with_se` corrigeert de STANDAARDFOUT voor autocorrelatie,
niet het PUNTSCHATTING. Deze module levert de puntschatting:

* `period_sharpe`: de reeks samengesteld tot maand of kwartaal, dan mean/sd · sqrt(k).
  Geen model nodig; wel minder observaties.
* `lo_annualized_sharpe`: Lo (2002, vgl. 19), `eta(q) = q / sqrt(q + 2 sum_{k<q} (q-k)
  w_k rho_k)`, met Bartlett-gewichten `w_k = 1 - k/(L+1)` tot `L = lags`. Zonder die
  gewichten (harde afkapping) kan de som bij negatieve steekproef-autocorrelaties onder
  nul zakken; met de gewichten is `(q-k) w_k` het product van twee driehoekskernen, dus
  positief-definiet (Schur), en de variantie gegarandeerd >= 0 (Newey-West 1987). Meer
  lags = strenger; bij trage autocorrelatie loopt de schatting door tot ver voorbij 10.

Geen van beide vervangt een bevroren poort: wat hier staat is rapportage naast de
bevroren metriek, niet een herdefinitie ervan.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..utils.failfast import DataContractError, require

__all__ = ["SerialSharpe", "autocorrelations", "lo_annualized_sharpe", "period_returns",
           "period_sharpe", "serial_sharpe"]

PERIODS = {"ME": 12, "QE": 4}


def _clean(r: pd.Series) -> pd.Series:
    require(isinstance(r.index, pd.DatetimeIndex), "Aggregeren vraagt een tijdindex.",
            DataContractError)
    return r.dropna()


def autocorrelations(r: pd.Series | np.ndarray, lags: int) -> np.ndarray:
    """rho_1 .. rho_lags (steekproef, gedeeld door de totale kwadratensom)."""
    x = np.asarray(pd.Series(r).dropna(), dtype=np.float64)
    require(lags >= 1 and x.size > lags, "Te weinig observaties voor die lags.",
            DataContractError, lags=lags, n=int(x.size))
    x = x - x.mean()
    den = float(x @ x)
    if den == 0.0:
        return np.zeros(lags)
    return np.array([float(x[k:] @ x[:-k]) / den for k in range(1, lags + 1)])


def lo_annualized_sharpe(r: pd.Series | np.ndarray, *, periods_per_year: int,
                         lags: int) -> float:
    """Lo (2002) met Bartlett-gewichten: de per-bar-Sharpe maal eta(q), q = `periods_per_year`."""
    x = np.asarray(pd.Series(r).dropna(), dtype=np.float64)
    sd = float(x.std(ddof=1)) if x.size > 1 else 0.0
    if sd == 0.0:
        return float("nan")
    q = int(periods_per_year)
    lags = min(int(lags), q - 1)
    rho = autocorrelations(x, lags)
    var = q + 2.0 * sum((q - k) * (1.0 - k / (lags + 1)) * rho[k - 1]
                        for k in range(1, lags + 1))
    require(var > 0.0, "Een Bartlett-gewogen variantie hoort positief te zijn.",
            DataContractError, var=var)
    return float(x.mean() / sd * q / math.sqrt(var))


def period_returns(r: pd.Series, rule: str) -> pd.Series:
    """Samengestelde rendementen per periode (`ME` = maand, `QE` = kwartaal)."""
    require(rule in PERIODS, "Onbekende periode.", DataContractError, rule=rule)
    return (1.0 + _clean(r)).resample(rule).prod() - 1.0


def period_sharpe(r: pd.Series, rule: str, *, min_periods: int = 6) -> float:
    """mean/sd van de periode-rendementen, geannualiseerd met sqrt(perioden per jaar)."""
    p = period_returns(r, rule)
    sd = float(p.std(ddof=1)) if len(p) > 1 else 0.0
    if len(p) < min_periods or sd == 0.0:
        return float("nan")
    return float(p.mean() / sd * math.sqrt(PERIODS[rule]))


@dataclass(frozen=True)
class SerialSharpe:
    sharpe_iid: float
    sharpe_monthly: float
    sharpe_quarterly: float
    sharpe_lo: dict[int, float]
    rho: tuple[float, ...]
    ann_vol_iid: float
    ann_vol_monthly: float

    def as_record(self) -> dict[str, float | list[float]]:
        out: dict[str, float | list[float]] = {
            "sharpe_iid": self.sharpe_iid, "sharpe_monthly": self.sharpe_monthly,
            "sharpe_quarterly": self.sharpe_quarterly, "rho_1_3": list(self.rho),
            "ann_vol_iid": self.ann_vol_iid, "ann_vol_monthly": self.ann_vol_monthly}
        out.update({f"sharpe_lo{k}": v for k, v in self.sharpe_lo.items()})
        return out


def serial_sharpe(r: pd.Series, *, bars_per_year: int,
                  lags: Sequence[int] = (10, 30, 60)) -> SerialSharpe:
    """Alle schattingen naast elkaar, voor rapportage."""
    x = _clean(r)
    sd = float(x.std(ddof=1)) if len(x) > 1 else 0.0
    iid = float(x.mean() / sd * math.sqrt(bars_per_year)) if sd > 0.0 else float("nan")
    m = period_returns(x, "ME")
    vol_m = float(m.std(ddof=1) * math.sqrt(12)) if len(m) > 1 else float("nan")
    rho = tuple(float(v) for v in autocorrelations(x, 3)) if len(x) > 3 else (float("nan"),) * 3
    lo = {int(k): (lo_annualized_sharpe(x, periods_per_year=bars_per_year, lags=int(k))
                   if len(x) > int(k) else float("nan")) for k in lags}
    return SerialSharpe(sharpe_iid=iid, sharpe_monthly=period_sharpe(x, "ME"),
                        sharpe_quarterly=period_sharpe(x, "QE"), sharpe_lo=lo, rho=rho,
                        ann_vol_iid=sd * math.sqrt(bars_per_year), ann_vol_monthly=vol_m)
