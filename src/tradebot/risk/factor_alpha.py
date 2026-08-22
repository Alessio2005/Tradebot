# src/tradebot/risk/factor_alpha.py
"""G4 factor lab — HAC/Newey-West residual-alpha regression (Mandate v3 §1.2).

ONE function, reused by every unit and by the book: regress daily returns on
the tradeable factorset of the unit's market and report the intercept
(residual alpha) with HAC errors. The binding rule:

    residual alpha > 0 with p < 0.05, at book level AND per sleeve.
    A sleeve that only retains factor loading after regression
    (|t(alpha)| < 2) is not alpha and does not count.

Promoted from ``scripts/true_alpha_gates.py::gate_g6`` into the package
(R-2), generalised to all markets. Factor *data* comes from the
point-in-time data layer (``tradebot.data.sources``); this module only
defines the per-market factorset names (G4 spec) and the regression.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.api as sm

__all__ = ["FactorAlphaResult", "factor_residual_alpha", "G4_FACTORSETS"]

# Mandate v3 §10 G4 — the per-market tradeable factorsets.
G4_FACTORSETS: dict[str, tuple[str, ...]] = {
    "crypto": ("MKT", "TSMOM"),
    "equities": ("Mkt-RF", "SMB", "HML", "RMW", "CMA", "MOM", "BAB"),
    "fx": ("DOLLAR", "CARRY", "TREND"),
    "commodities": ("MKT", "CARRY", "MOM"),
}

_T_ALPHA_FLOOR = 2.0  # |t(alpha)| >= 2 — below this a sleeve "is not alpha"


@dataclass(frozen=True)
class FactorAlphaResult:
    """Complete G4 row for the gate table."""

    unit: str
    market: str
    alpha_daily: float
    alpha_ann: float
    t_alpha: float
    p_alpha: float
    loadings: dict[str, float] = field(default_factory=dict)
    loading_tstats: dict[str, float] = field(default_factory=dict)
    r_squared: float = float("nan")
    n_obs: int = 0
    hac_maxlags: int = 0
    factors: tuple[str, ...] = ()

    @property
    def passes(self) -> bool:
        """G4 pass: positive residual alpha, p < 0.05, t >= 2."""
        return (
            self.alpha_daily > 0
            and self.p_alpha < 0.05
            and self.t_alpha >= _T_ALPHA_FLOOR
        )

    def gate_row(self) -> str:
        """One formatted G4 line for the wave gate table."""
        verdict = "PASS" if self.passes else "FAIL"
        return (
            f"G4 {self.unit} [{self.market}]: "
            f"ann_alpha={self.alpha_ann * 100:+.1f}%  t={self.t_alpha:+.2f}  "
            f"p={self.p_alpha:.4f}  R2={self.r_squared:.3f}  "
            f"n={self.n_obs}  factors={','.join(self.factors)}  [{verdict}]"
        )


def factor_residual_alpha(
    returns: pd.Series,
    factors: pd.DataFrame,
    unit: str,
    market: str,
    periods_per_year: int = 365,
    hac_maxlags: int | None = None,
) -> FactorAlphaResult:
    """Regress ``returns`` on ``factors`` with HAC (Newey-West) errors.

    Parameters
    ----------
    returns : daily (or per-period) net return series of the unit/book.
    factors : DataFrame of factor returns, columns = factor names, index
        aligned/alignable with ``returns``. Factors must be CAUSAL — built
        from data <= t-1, exactly like the unit's own signals (R-1).
    unit, market : identifiers for the gate-table row. ``market`` must be a
        key of ``G4_FACTORSETS`` or ``"book"`` (combined multi-market set).
    periods_per_year : 365 for crypto daily, 252 for equities/FX daily.
    hac_maxlags : Newey-West lag; default Bailey-standard ceil(0.75 * n^(1/3)),
        floored at 10 to match the audited crypto implementation.

    Notes
    -----
    Rows with any NaN are dropped after an inner join — the caller is
    responsible for ensuring the factor panel covers the return window
    (a short overlap silently weakens the test, so n_obs is reported and
    must be sanity-checked in the wave log).
    """
    if market != "book" and market in G4_FACTORSETS:
        missing = set(G4_FACTORSETS[market]) - set(factors.columns)
        if missing:
            raise ValueError(
                f"Factorset for market {market!r} incomplete: missing "
                f"{sorted(missing)} (G4 requires the full set)."
            )

    df = pd.concat([returns.rename("y"), factors], axis=1, join="inner")
    df = df.dropna()
    if len(df) < 60:
        raise ValueError(
            f"Only {len(df)} overlapping observations — too few for a "
            "meaningful HAC regression (need >= 60)."
        )

    if hac_maxlags is None:
        hac_maxlags = max(10, int(np.ceil(0.75 * len(df) ** (1 / 3))))

    x_cols = [c for c in df.columns if c != "y"]
    X = sm.add_constant(df[x_cols])
    model = sm.OLS(df["y"], X).fit(
        cov_type="HAC", cov_kwds={"maxlags": hac_maxlags}
    )

    alpha = float(model.params["const"])
    return FactorAlphaResult(
        unit=unit,
        market=market,
        alpha_daily=alpha,
        alpha_ann=alpha * periods_per_year,
        t_alpha=float(model.tvalues["const"]),
        p_alpha=float(model.pvalues["const"]),
        loadings={c: float(model.params[c]) for c in x_cols},
        loading_tstats={c: float(model.tvalues[c]) for c in x_cols},
        r_squared=float(model.rsquared),
        n_obs=int(len(df)),
        hac_maxlags=int(hac_maxlags),
        factors=tuple(x_cols),
    )
