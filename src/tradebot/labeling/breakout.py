"""Het primaire signaal: een CUSUM-doorbraak op de log-slotkoers (spec §5).

Richting = richting van de doorbraak. De drempel op bar t is `k * sigma[t-1]`:
de volatiliteit van de eventbar zelf mag niet bepalen of die bar een event is.
`k` wordt vastgelegd op eventfrequentie en nooit op rendement; `calibrate_k`
leest daarom geen labels.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from ..utils.failfast import DataContractError, require
from .cusum import directional_cusum_filter

__all__ = ["breakout_side", "calibrate_k", "events_per_week"]


def breakout_side(close: pd.Series, sigma_daily: pd.Series, k: float) -> pd.Series:
    """+1 / -1 op een doorbraakbar, 0 elders. Leidende NaN's zijn de notering."""
    require(close.index.equals(sigma_daily.index),
            "close en sigma delen geen index.", DataContractError)
    require(k > 0.0, "k moet positief zijn.", DataContractError, k=k)
    log_p = np.log(close.to_numpy(dtype=np.float64))
    thr = k * sigma_daily.shift(1).to_numpy(dtype=np.float64)
    ok = np.isfinite(log_p) & np.isfinite(thr) & (thr > 0.0)
    side = np.zeros(len(close), dtype=np.float64)
    if int(ok.sum()) >= 2:
        start = int(np.argmax(ok))
        require(
            bool(ok[start:].all()),
            "Een gat na de eerste geldige bar. Een CUSUM over een gat telt de "
            "sprong als één beweging; dat is een datadefect, geen signaal.",
            DataContractError,
            first_gap=str(close.index[start + int(np.argmin(ok[start:]))]),
        )
        up, down = directional_cusum_filter(log_p[start:], thr[start:], thr[start:])
        side[start + up] = 1.0
        side[start + down] = -1.0
    return pd.Series(side, index=close.index, name="side")


def events_per_week(
    sides: Mapping[str, pd.Series], *, start: pd.Timestamp, end: pd.Timestamp,
) -> float:
    """Events per week voor het hele boek in `[start, end)`."""
    weeks = (end - start) / pd.Timedelta(days=7)
    require(weeks > 0.0, "Een leeg venster.", DataContractError,
            start=str(start), end=str(end))
    total = 0
    for side in sides.values():
        window = side.loc[(side.index >= start) & (side.index < end)]
        total += int((window != 0.0).sum())
    return float(total / weeks)


def calibrate_k(
    close: Mapping[str, pd.Series],
    sigma: Mapping[str, pd.Series],
    *,
    k_grid: Sequence[float],
    target_per_week: float,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[float, dict[float, float]]:
    """De k uit `k_grid` waarvan de eventfrequentie in `[start, end)` het dichtst bij het doel ligt.

    Alleen data vóór `end` wordt gelezen. Bij gelijke afstand wint de grotere k
    (minder events, dus voorzichtiger).
    """
    require(set(close) == set(sigma), "close en sigma dekken andere symbolen.",
            DataContractError)
    rates: dict[float, float] = {}
    for k in k_grid:
        sides = {
            s: breakout_side(close[s].loc[close[s].index < end],
                             sigma[s].loc[sigma[s].index < end], float(k))
            for s in close
        }
        rates[float(k)] = events_per_week(sides, start=start, end=end)
    best = min(rates, key=lambda k: (abs(rates[k] - target_per_week), -k))
    return best, rates
