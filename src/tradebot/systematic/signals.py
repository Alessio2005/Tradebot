"""De drie signalen van het boek. Elk is een pure, causale functie van data ≤ *t*.

* `trend_score`  -- tijdreeks-momentum over een vast lookbackraster, in [-1, +1].
* `known_carry`  -- de funding die een long de afgelopen `window` bars betaalde,
  `lag` bars verschoven: uitsluitend afrekeningen van vóór het besluit.
* `carry_ranks`  -- de dollar-neutrale rangschikking daarop.

De causaliteit wordt niet aangenomen maar getoetst in
`tests/lookahead/test_systematic_causality.py`: de toekomst verstoren mag geen enkele
waarde tot en met *t* veranderen.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from ..utils.failfast import DataContractError, require

__all__ = ["carry_ranks", "known_carry", "trend_score"]


def trend_score(
    close: pd.DataFrame,
    sigma_daily: pd.DataFrame,
    *,
    lookbacks: Sequence[int],
    z_clip: float,
) -> pd.DataFrame:
    """Gemiddelde van `clip(z_L, ±z_clip)/z_clip` over de lookbacks met volledige historie.

    `z_L = ln(P_t / P_{t-L}) / (σ_t √L)`: het L-daagse logrendement in eenheden van zijn
    eigen verwachte spreiding. Een lookback zonder volledige historie doet niet mee
    (NaN), zodat een jonge munt alleen op de korte lookbacks wordt beoordeeld en niet
    op een afgekapt venster dat als lang wordt gepresenteerd.
    """
    require(len(lookbacks) > 0 and all(int(x) >= 1 for x in lookbacks),
            "Lege of ongeldige lookbacks.", DataContractError)
    require(z_clip > 0.0, "z_clip moet positief zijn.", DataContractError)
    logp = np.log(close)
    parts = []
    for lb in lookbacks:
        z = (logp - logp.shift(int(lb))) / (sigma_daily * np.sqrt(float(lb)))
        parts.append((z.clip(-z_clip, z_clip) / z_clip).to_numpy())
    stack = np.stack(parts)
    with np.errstate(all="ignore"):
        count = np.isfinite(stack).sum(axis=0)
        total = np.nansum(stack, axis=0)
        score = np.where(count > 0, total / np.maximum(count, 1), np.nan)
    return pd.DataFrame(score, index=close.index, columns=close.columns)


def known_carry(
    funding: pd.DataFrame,
    listed: pd.DataFrame,
    *,
    window_bars: int,
    signal_lag_bars: int,
) -> pd.DataFrame:
    """De som van de funding over `window_bars` bars, `signal_lag_bars` verschoven.

    Positief = longs betaalden. Vóór er een volledig venster na de notering ligt: NaN.
    """
    require(window_bars >= 1 and signal_lag_bars >= 1,
            "Carryvenster en signaalvertraging moeten ≥ 1 zijn.", DataContractError,
            window_bars=window_bars, signal_lag_bars=signal_lag_bars)
    masked = funding.where(listed)
    summed = masked.rolling(int(window_bars), min_periods=int(window_bars)).sum()
    return summed.shift(int(signal_lag_bars))


def carry_ranks(carry: pd.DataFrame, live: pd.DataFrame) -> pd.DataFrame:
    """-1 voor de bovenste helft (hoogste carry: short), +1 voor de onderste, 0 ertussen.

    Gelijke waarden worden op symboolnaam gebroken, zodat de kolomvolgorde niets
    beslist. Bij een oneven aantal levende namen krijgt de middelste 0. Minder dan
    twee levende namen met een carry: geen positie.
    """
    out = pd.DataFrame(0.0, index=carry.index, columns=carry.columns)
    names = np.array(sorted(carry.columns))
    ordered = carry[names]
    alive = live[names]
    for ts in carry.index:
        row = ordered.loc[ts]
        ok = row.notna().to_numpy() & alive.loc[ts].to_numpy()
        n = int(ok.sum())
        if n < 2:
            continue
        vals = row.to_numpy()[ok]
        syms = names[ok]
        # Aflopend op carry; bij gelijke carry beslist de naam (lexsort: laatste sleutel eerst).
        order = np.lexsort((syms, -vals))
        half = n // 2
        for s in syms[order[:half]]:
            out.at[ts, s] = -1.0
        for s in syms[order[n - half:]]:
            out.at[ts, s] = 1.0
    return out
