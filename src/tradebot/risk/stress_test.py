# src/tradebot/risk/stress_test.py
"""Historical and synthetic stress-testing suite.

Scenarios
---------
  covid_crash     : 2020-02-20 – 2020-03-13  (-50% BTC in 22 days)
  luna_collapse   : 2022-05-04 – 2022-05-14  (-99% LUNA + contagion)
  ftx_collapse    : 2022-11-06 – 2022-11-11  (BTC -25% in 5 days)
  rate_shock      : 2022-01-01 – 2022-06-18  (Fed tightening bear)
  synthetic_3sigma: Monte Carlo ±3σ paths

Reference: BCBS (2009) "Principles for sound stress testing practices";
adapted for crypto by incorporating funding-rate and contagion scenarios.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["StressScenario", "StressResult", "StressTestSuite"]


_SCENARIOS: Dict[str, Optional[Tuple[str, str]]] = {
    "covid_crash":      ("2020-02-20", "2020-03-13"),
    "luna_collapse":    ("2022-05-04", "2022-05-14"),
    "ftx_collapse":     ("2022-11-06", "2022-11-11"),
    "rate_shock":       ("2022-01-01", "2022-06-18"),
    "synthetic_3sigma": None,
}


@dataclass(frozen=True)
class StressScenario:
    """Metadata for a single stress scenario."""

    name: str
    start: Optional[str]
    end: Optional[str]
    description: str = ""


@dataclass
class StressResult:
    """Result of running one stress scenario on a portfolio."""

    scenario_name: str
    portfolio_return: float        # cumulative return during scenario
    max_drawdown: float            # max intra-scenario drawdown
    var_99: float                  # 1-day 99% VaR during scenario
    asset_returns: pd.Series       # per-asset return during scenario
    path: Optional[pd.Series] = None  # daily equity path (if available)


class StressTestSuite:
    """Run historical and synthetic stress tests on a portfolio.

    Parameters
    ----------
    price_data :
        DataFrame of asset prices (rows=date, cols=assets).
        Must have a DatetimeIndex.
    """

    SCENARIOS = _SCENARIOS

    def __init__(self, price_data: pd.DataFrame) -> None:
        self.prices = price_data

    def run_historical(
        self,
        portfolio_weights: pd.Series,
        scenario_key: str,
    ) -> StressResult:
        """Run a named historical stress scenario.

        Parameters
        ----------
        portfolio_weights :
            Current portfolio weights (indexed by asset name).
        scenario_key :
            One of the keys in ``StressTestSuite.SCENARIOS``.
        """
        if scenario_key not in _SCENARIOS:
            raise ValueError(f"Unknown scenario: {scenario_key!r}. "
                             f"Valid: {list(_SCENARIOS.keys())}")

        date_range = _SCENARIOS[scenario_key]
        if date_range is None:
            return self.run_monte_carlo(portfolio_weights, n_paths=500, scenario_name=scenario_key)

        # Create tz-aware timestamps to compare with tz-aware price index.
        start = pd.Timestamp(date_range[0]).tz_localize("UTC")
        end   = pd.Timestamp(date_range[1]).tz_localize("UTC")
        prices_in_window = self.prices.loc[
            (self.prices.index >= start) & (self.prices.index <= end)
        ]

        if prices_in_window.empty:
            logger.warning("No price data for scenario %s (%s – %s).", scenario_key, start, end)
            return StressResult(
                scenario_name=scenario_key,
                portfolio_return=0.0,
                max_drawdown=0.0,
                var_99=0.0,
                asset_returns=pd.Series(dtype=float),
            )

        assets = portfolio_weights.index.tolist()
        available = [a for a in assets if a in prices_in_window.columns]
        if not available:
            return StressResult(scenario_key, 0.0, 0.0, 0.0, pd.Series(dtype=float))

        w = portfolio_weights.reindex(available).fillna(0.0)
        if w.sum() > 1e-9:
            w /= w.sum()

        p = prices_in_window[available]
        daily_ret = p.pct_change().fillna(0.0)
        asset_scenario_ret = (p.iloc[-1] / p.iloc[0] - 1).reindex(available)

        port_daily = (daily_ret * w.values).sum(axis=1)
        cum_ret = float((1 + port_daily).prod() - 1)

        equity = (1 + port_daily).cumprod()
        running_max = equity.cummax()
        dd = ((equity - running_max) / running_max).min()
        max_dd = float(dd)

        var_99 = float(np.percentile(port_daily.values, 1))

        return StressResult(
            scenario_name=scenario_key,
            portfolio_return=cum_ret,
            max_drawdown=max_dd,
            var_99=var_99,
            asset_returns=asset_scenario_ret,
            path=equity,
        )

    def run_monte_carlo(
        self,
        portfolio_weights: pd.Series,
        n_paths: int = 10_000,
        horizon_days: int = 22,
        sigma_multiplier: float = 3.0,
        scenario_name: str = "synthetic_3sigma",
        seed: int = 42,
        tail_correlation: float | None = None,
        t_copula_df: float | None = None,
        lookback_days: int = 252,
    ) -> StressResult:
        """Synthetic stress test via Monte Carlo (±3σ shock paths).

        CHIEF AUDIT-FIX (Sim-to-Reality #18):
          Previous Gaussian MC + historical covariance underestimates crisis
          tails for two reasons:
            1. **Tail correlation breakdown** — historical pairwise ρ ≈ 0.5
               in calm regimes but rises to ≈ 0.95 during liquidation events.
               Set ``tail_correlation=0.95`` to force a correlation floor
               on the off-diagonals of the shock covariance.
            2. **Gaussian tails are too thin** — crypto daily-return kurtosis
               sits around 10-20; even a 3σ Gaussian shock under-represents
               the empirical 99th-percentile move.  Set ``t_copula_df`` to
               draw from a multivariate Student-t (df ≈ 4-6 for crypto) so
               the empirical fat-tail behaviour is preserved.

        Draws ``n_paths`` random returns with σ = ``sigma_multiplier`` × historical
        daily vol and computes the portfolio tail statistics.

        Parameters
        ----------
        portfolio_weights : asset weights series.
        n_paths : Monte Carlo paths.
        horizon_days : path length.
        sigma_multiplier : scale on historical σ.
        scenario_name : label for the StressResult.
        seed : RNG seed.
        tail_correlation : when set (e.g. 0.95), all off-diagonal
            correlations below this value are lifted to it before
            sampling.  Models the empirical corr→1 effect.
        t_copula_df : when set (e.g. 5.0), shocks are drawn from a
            multivariate Student-t instead of Gaussian.  None = Gaussian.
        lookback_days : window for the covariance estimate.  CHIEF AUDIT
            2026-05-23 (P-5): eerder werd ``daily_rets.cov()`` over de
            HELE meegegeven price-series berekend — inclusief de tail-
            events die juist gesimuleerd worden. Resultaat: σ wordt
            opgeblazen door bv. de March-2020 / FTX-crash bars en het
            scenario verliest scherpte. Default 252 ≈ 1 jaar trailing
            kalmer-regime cov, een gangbare BIS/Basel-stijl keuze. Zet
            hoger (bv. 504) voor stabieler maar minder responsief, of
            None-equivalent door ``lookback_days=len(daily_rets)``.
        """
        assets = portfolio_weights.index.tolist()
        available = [a for a in assets if a in self.prices.columns]
        if not available:
            return StressResult(scenario_name, 0.0, 0.0, 0.0, pd.Series(dtype=float))

        w = portfolio_weights.reindex(available).fillna(0.0)
        if w.sum() > 1e-9:
            w /= w.sum()

        daily_rets = self.prices[available].pct_change().dropna()
        # CHIEF AUDIT 2026-05-23 (P-5): rolling lookback ipv full-series cov
        # om tail-events buiten de cov-estimator te houden.
        cov = daily_rets.tail(int(lookback_days)).cov().values * sigma_multiplier ** 2

        # CHIEF AUDIT-FIX #18a: tail-correlation floor on the off-diagonal.
        if tail_correlation is not None and len(available) > 1:
            std = np.sqrt(np.diag(cov))
            corr = cov / np.outer(std + 1e-12, std + 1e-12)
            np.fill_diagonal(corr, 1.0)
            # Lift weak correlations toward the floor, preserve sign
            floor = float(tail_correlation)
            mask = np.abs(corr) < floor
            np.fill_diagonal(mask, False)
            corr = np.where(mask, np.sign(corr) * floor, corr)
            corr = np.where(np.isfinite(corr), corr, 0.0)
            cov = corr * np.outer(std, std)

        rng = np.random.default_rng(seed)
        port_terminal: list[float] = []

        # CHIEF AUDIT-FIX #18b: optional Student-t copula via Gaussian scale-mixture
        # Z ~ N(0, cov);  X = Z / sqrt(W/df)  where  W ~ ChiSq(df)  → multivariate t.
        for _ in range(n_paths):
            z_shocks = rng.multivariate_normal(
                np.zeros(len(available)), cov, size=horizon_days,
            )
            if t_copula_df is not None and t_copula_df > 2.0:
                df_t = float(t_copula_df)
                chi2 = rng.chisquare(df_t, size=horizon_days).reshape(-1, 1)
                shocks = z_shocks / np.sqrt(chi2 / df_t)
            else:
                shocks = z_shocks
            path_ret = (shocks * w.values).sum(axis=1)
            port_terminal.append(float((1 + path_ret).prod() - 1))

        arr = np.array(port_terminal)
        return StressResult(
            scenario_name=scenario_name,
            portfolio_return=float(np.mean(arr)),
            max_drawdown=float(np.percentile(arr, 1)),
            var_99=float(np.percentile(arr, 1)),
            asset_returns=pd.Series(dtype=float),
        )

    def run_all(
        self,
        portfolio_weights: pd.Series,
        mc_paths: int = 500,
    ) -> Dict[str, StressResult]:
        """Run all scenarios and return a dict of results."""
        results: Dict[str, StressResult] = {}
        for key in _SCENARIOS:
            try:
                results[key] = self.run_historical(portfolio_weights, key)
            except Exception:
                logger.exception("Stress scenario failed: %s", key)
        return results

    def marginal_var(
        self,
        portfolio_weights: pd.Series,
        confidence: float = 0.99,
        window: int = 252,
    ) -> pd.Series:
        """Compute marginal 1-day VaR contribution per asset.

        MVaR_i = w_i * (∂VaR / ∂w_i) ≈ w_i * cov(r_i, r_p) / σ_p * z_α
        """
        assets = portfolio_weights.index.tolist()
        available = [a for a in assets if a in self.prices.columns]
        if not available:
            return pd.Series(dtype=float)

        w = portfolio_weights.reindex(available).fillna(0.0)
        if w.sum() > 1e-9:
            w /= w.sum()

        daily_rets = self.prices[available].pct_change().dropna().tail(window)
        cov = daily_rets.cov().values
        port_vol = float(np.sqrt(w.values @ cov @ w.values))

        from scipy.stats import norm
        z = float(norm.ppf(confidence))

        # Marginal VaR = z * w_i * cov(r_i, r_p) / σ_p
        cov_with_port = cov @ w.values
        mvar = z * w.values * cov_with_port / (port_vol + 1e-9)
        return pd.Series(mvar, index=available, name=f"mvar_{confidence:.0%}")
