# src/tradebot/backtest/dd_shape.py
"""Horizon-aware drawdown-shape reference distribution (Wave 28, step 0.3b).

WHAT THIS REPLACES
------------------
``tests/killgates/test_expansion_killgates.py`` derived its Calmar ceiling from
``E[MaxDD] = sigma**2 / (2*mu)``, i.e. ``Calmar <= 2*S**2``. That identity is
real but it is **not the expected maximum drawdown**: it is the mean of the
drawdown process at a *random time* — the stationary law of a reflected
Brownian motion with negative drift.

The *maximum* over a horizon is a different quantity, and for reflected BM the
all-time supremum is a.s. unbounded: it grows logarithmically in the horizon.
Measured (``scripts/w28_dd_shape_calibration.py``, seed 42, 20k paths, iid
normal, 5685 bars == 22.6y, Sharpe 0.40):

    drawdown at a random time   1.04 vol units   <- what 1/(2S) = 1.25 predicts
    MAXIMUM drawdown            3.25 vol units   <- what a gate actually sees

The old ceiling was therefore ~2.6x too generous at this horizon, which is how
a Calmar floor of 0.25 at a Sharpe floor of 0.40 passed the consistency check
while in fact rejecting **99.7%** of strategies that genuinely have Sharpe 0.40.

WHY THIS SIMULATES INSTEAD OF FITTING A FORMULA
-----------------------------------------------
The natural closed form is Magdon-Ismail & Atiya (2004), "On the Maximum
Drawdown of a Brownian Motion" (J. Applied Probability 41(1)): in the
positive-drift regime ``E[MaxDD] ~ (sigma**2/2mu) * Q(mu**2 T / sigma**2)``
with ``Q(x) ~ ln(x) + const``. Fitting that constant reproduced the simulation
to ~5% at low Sharpe but drifted to ~16% at Sharpe 1.0 and higher vol, because
the compounding drag and the quantile multiplier both move with S and sigma.

A gate whose threshold is a fitted approximation of the thing it is supposed to
measure can silently change meaning when the fit is off. So the reference here
IS the simulation: deterministic (fixed seed, R-5), cached, and exact for the
null it states. No constants to drift.

WHY A QUANTILE, NOT A MEAN
--------------------------
A gate compares ONE realised path against the threshold. Thresholding at the
mean of a right-skewed distribution rejects ~37-50% of qualifying strategies by
luck alone. The gate therefore thresholds at an upper quantile of the null
"this strategy really does have Sharpe S over this horizon", so its
false-rejection rate is the stated budget (5% at q=0.95) by construction.

The null is iid normal: no fat tails, no autocorrelation, no vol clustering.
Real strategies draw worse maxima than that, so the reference is CONSERVATIVE
(the cap is if anything tight, never generous).

SCOPE
-----
This is a SHAPE gate: "is this drawdown pathological *given the claimed
skill*". It is not a risk limit. Absolute drawdown caps (propfirm lines, ruin
prevention) are a separate, independent constraint and nothing here weakens
them.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

__all__ = [
    "expected_max_dd_over_vol",
    "max_dd_over_vol_quantile",
    "calmar_ceiling",
    "dd_shape_reference",
    "SEED",
    "N_PATHS",
]

TRADING_DAYS = 252
SEED = 42          # R-5 determinism
N_PATHS = 4000
_CHUNK = 500
_MIN_SHARPE = 0.05


@lru_cache(maxsize=256)
def _simulate(sharpe: float, years: float, ann_vol: float) -> tuple[float, ...]:
    """Sorted MaxDD/vol sample under the "genuinely Sharpe ``sharpe``" null.

    Cached on rounded arguments by the public wrappers, so a gate evaluation
    costs one simulation per distinct (Sharpe, horizon, vol) triple.
    """
    n_bars = max(int(round(years * TRADING_DAYS)), 2)
    rng = np.random.default_rng(SEED)
    mu_d = sharpe * ann_vol / TRADING_DAYS
    sd_d = ann_vol / np.sqrt(TRADING_DAYS)

    out = np.empty(N_PATHS)
    for i in range(0, N_PATHS, _CHUNK):
        r = rng.normal(mu_d, sd_d, size=(_CHUNK, n_bars))
        eq = np.cumprod(1.0 + r, axis=1)
        peak = np.maximum.accumulate(eq, axis=1)
        dd = np.abs((eq / peak - 1.0).min(axis=1))
        vol = r.std(axis=1, ddof=1) * np.sqrt(TRADING_DAYS)
        out[i:i + _CHUNK] = dd / vol
    return tuple(np.sort(out))


def dd_shape_reference(
    sharpe: float, years: float, ann_vol: float = 0.10
) -> np.ndarray:
    """The null distribution of MaxDD/vol, as a sorted array."""
    if years <= 0.0:
        raise ValueError(f"years must be positive, got {years}")
    if ann_vol <= 0.0:
        raise ValueError(f"ann_vol must be positive, got {ann_vol}")
    s = max(round(float(sharpe), 2), _MIN_SHARPE)
    return np.asarray(_simulate(s, round(float(years), 1), round(float(ann_vol), 3)))


def expected_max_dd_over_vol(
    sharpe: float, years: float, ann_vol: float = 0.10
) -> float:
    """Expected maximum drawdown over the horizon, in units of annualised vol."""
    return float(dd_shape_reference(sharpe, years, ann_vol).mean())


def max_dd_over_vol_quantile(
    sharpe: float, years: float, q: float = 0.95, ann_vol: float = 0.10
) -> float:
    """Upper-``q`` quantile of MaxDD/vol under the Sharpe-``sharpe`` null.

    A realised ``dd_over_vol`` above this is worse than ``q`` of the paths a
    genuinely-skilled strategy would produce — evidence that the drawdown shape
    is pathological, not merely unlucky. False-rejection budget is ``1 - q``.
    """
    if not 0.5 < q < 1.0:
        raise ValueError(f"q must lie in (0.5, 1.0), got {q}")
    return float(np.quantile(dd_shape_reference(sharpe, years, ann_vol), q))


def calmar_ceiling(
    sharpe: float, years: float, ann_vol: float = 0.10
) -> float:
    """Largest Calmar a strategy with this Sharpe can be *expected* to show.

    ``Calmar = CAGR / MaxDD`` and ``CAGR ~= S*sigma - sigma**2/2``, so in vol
    units ``Calmar = (S - sigma/2) / (MaxDD/sigma)``.

    A Calmar floor above this is unsatisfiable in expectation. The old
    ``2*S**2`` omitted the horizon term and overstated it by ~2.6x at 22y.
    """
    return (float(sharpe) - ann_vol / 2.0) / expected_max_dd_over_vol(
        sharpe, years, ann_vol
    )
