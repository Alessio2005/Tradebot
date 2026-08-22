# src/tradebot/risk/factor_risk.py
"""Barra-style factor risk model for crypto portfolios.

Decomposes portfolio risk into:
  - Factor risk: systematic exposure to common risk factors
  - Idiosyncratic risk: asset-specific residual

Crypto factors used here:
  - Market (BTC beta)
  - Momentum (12-1 month cross-sectional)
  - Volatility (realised vol rank)
  - Liquidity (ADV rank)

Reference: Barra (1998) "Risk Models and Investment Management";
adapted for crypto by dropping industry factors.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)

__all__ = ["FactorExposure", "FactorRiskModel", "compute_factor_risk"]


@dataclass
class FactorExposure:
    """Factor loading (beta) for a single asset."""

    asset: str
    market_beta: float
    momentum_loading: float
    vol_loading: float
    r_squared: float
    idio_vol: float   # annualised idiosyncratic volatility


def _compute_loadings(
    asset_returns: pd.Series,
    factor_returns: pd.DataFrame,
    window: int = 126,
) -> FactorExposure:
    """OLS regression of asset returns on factor returns."""
    asset = str(asset_returns.name)
    common = asset_returns.index.intersection(factor_returns.index)
    y = asset_returns.loc[common].tail(window).values
    X = factor_returns.loc[common].tail(window).values

    if len(y) < 20 or X.shape[1] == 0:
        return FactorExposure(asset, 1.0, 0.0, 0.0, 0.0, float(np.std(y) * np.sqrt(252)))

    reg = LinearRegression(fit_intercept=True)
    reg.fit(X, y)

    residuals = y - reg.predict(X)
    r2 = float(max(0.0, reg.score(X, y)))
    idio_vol = float(np.std(residuals) * np.sqrt(252))

    betas = reg.coef_
    return FactorExposure(
        asset=asset,
        market_beta=float(betas[0]) if len(betas) > 0 else 1.0,
        momentum_loading=float(betas[1]) if len(betas) > 1 else 0.0,
        vol_loading=float(betas[2]) if len(betas) > 2 else 0.0,
        r_squared=r2,
        idio_vol=idio_vol,
    )


class FactorRiskModel:
    """Compute and cache factor exposures for a multi-asset portfolio.

    Parameters
    ----------
    factor_window :
        Number of bars used for OLS regression per asset.
    """

    def __init__(self, factor_window: int = 126) -> None:
        self.factor_window = factor_window
        self._exposures: dict[str, FactorExposure] = {}

    def fit(
        self,
        returns: pd.DataFrame,
        btc_returns: pd.Series | None = None,
    ) -> None:
        """Fit factor loadings for all assets.

        Parameters
        ----------
        returns :
            Asset returns DataFrame.
        btc_returns :
            BTC returns as market factor proxy.  If None, uses the
            equal-weighted mean of ``returns`` as the market.
        """
        if btc_returns is None:
            mkt = returns.mean(axis=1)
        else:
            mkt = btc_returns.reindex(returns.index).fillna(0.0)

        # Construct factor matrix
        log_rets = np.log1p(returns.fillna(0.0))
        mkt_factor = mkt.values

        # Momentum: 63-bar return (rank-normalised within universe)
        mom_raw = returns.rolling(63).sum()
        mom_ranks = mom_raw.rank(axis=1, pct=True).subtract(0.5)

        # Volatility factor: 21-bar rolling vol (rank)
        vol_raw = returns.rolling(21).std()
        vol_ranks = vol_raw.rank(axis=1, pct=True).subtract(0.5)

        for asset in returns.columns:
            factor_df = pd.DataFrame({
                "market":   mkt_factor,
                "momentum": mom_ranks[asset].values if asset in mom_ranks.columns else np.zeros(len(returns)),
                "vol":      vol_ranks[asset].values  if asset in vol_ranks.columns  else np.zeros(len(returns)),
            }, index=returns.index).dropna()

            self._exposures[asset] = _compute_loadings(
                log_rets[asset],
                factor_df,
                window=self.factor_window,
            )

    def exposures(self) -> pd.DataFrame:
        """Return factor exposures as a DataFrame (one row per asset)."""
        return pd.DataFrame([
            {
                "asset":             e.asset,
                "market_beta":       e.market_beta,
                "momentum_loading":  e.momentum_loading,
                "vol_loading":       e.vol_loading,
                "r_squared":         e.r_squared,
                "idio_vol":          e.idio_vol,
            }
            for e in self._exposures.values()
        ]).set_index("asset")

    def portfolio_factor_risk(self, weights: pd.Series) -> dict[str, float]:
        """Decompose portfolio risk into factor and idiosyncratic components."""
        assets = weights.index.tolist()
        w = weights.fillna(0.0).values

        total_var = 0.0
        idio_var  = 0.0
        for i, asset in enumerate(assets):
            exp = self._exposures.get(asset)
            if exp is None:
                continue
            idio_var  += (w[i] * exp.idio_vol / np.sqrt(252)) ** 2
            total_var += (w[i] ** 2) * ((exp.market_beta * 0.02) ** 2 + (exp.idio_vol / np.sqrt(252)) ** 2)

        factor_var = max(0.0, total_var - idio_var)
        return {
            "total_vol_ann":   float(np.sqrt(max(total_var, 0.0)) * np.sqrt(252)),
            "factor_vol_ann":  float(np.sqrt(factor_var) * np.sqrt(252)),
            "idio_vol_ann":    float(np.sqrt(idio_var) * np.sqrt(252)),
            "diversification": float(np.sqrt(idio_var / (total_var + 1e-9))),
        }


def compute_factor_risk(
    returns: pd.DataFrame,
    weights: pd.Series,
    btc_returns: pd.Series | None = None,
    factor_window: int = 126,
) -> dict[str, float]:
    """Convenience wrapper: fit FactorRiskModel and return portfolio risk decomposition."""
    model = FactorRiskModel(factor_window=factor_window)
    model.fit(returns, btc_returns=btc_returns)
    return model.portfolio_factor_risk(weights)
