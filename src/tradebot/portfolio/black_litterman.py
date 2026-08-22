# src/tradebot/portfolio/black_litterman.py
"""Black-Litterman portfolio construction with CPCV-derived views.

View construction:
  P  = selection matrix (which assets the view concerns)
  Q  = CPCV OOS-summed alpha estimates (annualised)
  Ω  = diagonal matrix with variance of OOS estimates

Posterior:
  μ_BL = [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹ [(τΣ)⁻¹π + P'Ω⁻¹Q]

π = implied market returns (market-cap-weighted; proxy via volume in crypto)
τ = 1/T (standard)

Reference: Black & Litterman (1992); He & Litterman (1999) implementation.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from ..utils.failfast import DataContractError

logger = logging.getLogger(__name__)

__all__ = ["BLViews", "black_litterman_weights"]


class BLViews:
    """Container for Black-Litterman view specification.

    Parameters
    ----------
    assets :
        List of asset names matching the columns of the returns DataFrame.
    """

    def __init__(self, assets: list[str]) -> None:
        self.assets = assets
        self._P: list[np.ndarray] = []
        self._Q: list[float]      = []
        self._omega: list[float]  = []

    def add_absolute_view(
        self,
        asset: str,
        expected_return: float,
        uncertainty: float,
    ) -> None:
        """Add an absolute return view for a single asset.

        Parameters
        ----------
        asset : target asset name.
        expected_return : annualised expected return (e.g. 0.10 = 10%).
        uncertainty : variance of the view (higher = less confident).
        """
        idx = self.assets.index(asset)
        p = np.zeros(len(self.assets))
        p[idx] = 1.0
        self._P.append(p)
        self._Q.append(expected_return)
        self._omega.append(max(uncertainty, 1e-8))

    def add_relative_view(
        self,
        long_asset: str,
        short_asset: str,
        expected_spread: float,
        uncertainty: float,
    ) -> None:
        """Add a relative view: long_asset outperforms short_asset."""
        li = self.assets.index(long_asset)
        si = self.assets.index(short_asset)
        p = np.zeros(len(self.assets))
        p[li], p[si] = 1.0, -1.0
        self._P.append(p)
        self._Q.append(expected_spread)
        self._omega.append(max(uncertainty, 1e-8))

    def from_cpcv_oos(
        self,
        oos_probs: pd.DataFrame,
        uncertainty_col: str = "prob_std",
    ) -> None:
        """Populate views from CPCV OOS predictions.

        ``oos_probs`` must have columns: ``symbol``, ``expected_return``,
        and optionally ``prob_std`` (variance proxy).
        """
        for _, row in oos_probs.iterrows():
            sym = str(row["symbol"])
            if sym not in self.assets:
                continue
            er   = float(row.get("expected_return", 0.0))
            unc  = float(row.get(uncertainty_col, 0.01)) ** 2
            self.add_absolute_view(sym, er, unc)

    @property
    def P(self) -> np.ndarray:
        return np.vstack(self._P) if self._P else np.zeros((0, len(self.assets)))

    @property
    def Q(self) -> np.ndarray:
        return np.array(self._Q)

    @property
    def Omega(self) -> np.ndarray:
        return np.diag(self._omega)


def black_litterman_weights(
    returns: pd.DataFrame,
    views: BLViews | None = None,
    tau: float | None = None,
    risk_aversion: float = 2.5,
) -> pd.Series:
    """Compute Black-Litterman posterior expected returns and MVO weights.

    Parameters
    ----------
    returns :
        Historical returns DataFrame (rows=time, cols=assets).
    views :
        BLViews with investor views.  If None, reverts to market-implied.
    tau :
        Uncertainty in prior.  Defaults to 1 / T.
    risk_aversion :
        Market risk-aversion coefficient (lambda).

    Returns
    -------
    pd.Series of portfolio weights indexed by asset names.
    """
    assets = returns.columns.tolist()
    n = len(assets)
    T = len(returns)

    # Drop NaN rows rather than filling with 0 — fillna(0) artificially
    # decorrelates assets during data gaps, inflating posterior confidence.
    clean = returns.dropna(how="any")
    if len(clean) < 5:
        clean = returns.fillna(returns.mean())
    lw = LedoitWolf()
    lw.fit(clean.values)
    cov = lw.covariance_

    if tau is None:
        # τ = 1/T makes the prior very dominant for large T (small τ →
        # large (τΣ)⁻¹ → prior precision dominates).  Clamp to [0.01, 0.05]
        # per He & Litterman (1999) convention; 0.025 is the standard default.
        tau = float(np.clip(1.0 / max(T, 1), 0.01, 0.05))

    # Implied equilibrium returns (reverse-optimised from equal weights)
    w_eq = np.ones(n) / n
    pi   = risk_aversion * cov @ w_eq  # implied excess returns

    if views is None or len(views._P) == 0:
        # No views — return equal-weight
        return pd.Series(w_eq, index=assets)

    P = views.P
    Q = views.Q
    Omega = views.Omega

    # BL posterior expected return
    tau_cov_inv = np.linalg.pinv(tau * cov)
    omega_inv   = np.linalg.pinv(Omega + np.eye(len(Q)) * 1e-8)

    lhs = tau_cov_inv + P.T @ omega_inv @ P
    rhs = tau_cov_inv @ pi + P.T @ omega_inv @ Q

    # Phase 0: `except LinAlgError: mu_bl = pi` gaf de MARKTPRIOR terug als
    # "Black-Litterman posterior". De views (P, Q) verdwenen dan volledig uit de
    # schatting terwijl het resultaat als BL werd gerapporteerd.
    try:
        mu_bl = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError as exc:
        raise DataContractError(
            "Black-Litterman posterior-solve singulier. Er wordt NIET "
            "teruggevallen op de marktprior: dat zou de views stilzwijgend "
            "wegstrepen en het resultaat toch als BL rapporteren."
        ) from exc

    # MVO on posterior (risk_aversion * Σ w = mu_bl → solve for w)
    # Phase 0: `except LinAlgError: w_raw = w_eq` degradeerde de MVO-oplossing
    # stilzwijgend naar gelijke gewichten (1/N).
    try:
        w_raw = np.linalg.solve(risk_aversion * cov, mu_bl)
    except np.linalg.LinAlgError as exc:
        raise DataContractError(
            "MVO-solve op de BL-posterior singulier. Er wordt NIET "
            "teruggevallen op gelijke gewichten."
        ) from exc

    # Long-only, normalised
    w_raw = np.maximum(w_raw, 0.0)
    total = w_raw.sum()
    if total < 1e-9:
        weights = w_eq
    else:
        weights = w_raw / total

    return pd.Series(weights, index=assets, name="bl_weight")
