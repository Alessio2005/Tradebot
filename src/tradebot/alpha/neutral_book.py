"""neutral_book.py — market-neutral multi-sleeve target-weight engine.

Single source of truth for BOTH research backtests and live/paper trading
(identical code -> no research/live divergence; CHIEF methodological rule).

Three dollar-neutral sleeves, combined by causal (trailing-vol) risk-parity:
  • statarb   : cross-sectional residual reversal (Avellaneda-Lee), k=3.
  • lowvol    : betting-against-vol — long low realised-vol, short high.
  • carry     : funding-carry — short high-funding, long low-funding.

Every weight at bar t uses ONLY information available at the close of t
(rolling windows ending at t; risk-parity vol uses sleeve P&L through t-1).
Execution must apply weights to the NEXT bar (handled by the caller) so there
is no look-ahead. Parameters are pre-committed constants — NO optimisation.

Returns weights that satisfy:  sum(w) ≈ 0 (dollar-neutral),  sum|w| ≈ 1 (gross).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = ["NeutralBookConfig", "NeutralBook"]


@dataclass(frozen=True)
class NeutralBookConfig:
    statarb_k: int = 3          # residual cumulative-return lookback (days)
    vol_window: int = 20        # realised-vol window for low-vol sleeve (days)
    carry_window: int = 3       # trailing funding smoothing (days)
    rp_lookback: int = 252      # risk-parity trailing-vol window (days)
    rp_min_periods: int = 60
    min_names: int = 8          # require >= this many assets present per bar


def _xs_dollar_neutral(sig: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional demean + gross-normalise each row -> sum(w)=0, sum|w|=1."""
    w = sig.sub(sig.mean(axis=1), axis=0)
    g = w.abs().sum(axis=1).replace(0, np.nan)
    return w.div(g, axis=0).fillna(0.0)


class NeutralBook:
    def __init__(self, config: NeutralBookConfig | None = None) -> None:
        self.cfg = config or NeutralBookConfig()

    # ── individual sleeves (return dollar-neutral weight panels) ────────────
    def _statarb_weights(self, R: pd.DataFrame) -> pd.DataFrame:
        m = R.mean(axis=1)                                   # equal-weight market factor
        resid = R.sub(m, axis=0)                             # market-neutral residual
        cum = resid.rolling(self.cfg.statarb_k).sum()
        z = cum.sub(cum.mean(axis=1), axis=0).div(cum.std(axis=1).replace(0, np.nan), axis=0)
        return _xs_dollar_neutral(-z)                        # long laggards

    def _lowvol_weights(self, R: pd.DataFrame) -> pd.DataFrame:
        vol = R.rolling(self.cfg.vol_window).std()
        z = vol.sub(vol.mean(axis=1), axis=0).div(vol.std(axis=1).replace(0, np.nan), axis=0)
        return _xs_dollar_neutral(-z)                        # long low-vol

    def _carry_weights(self, F: pd.DataFrame) -> pd.DataFrame:
        car = -F.rolling(self.cfg.carry_window).mean()       # short high funding
        z = car.sub(car.mean(axis=1), axis=0).div(car.std(axis=1).replace(0, np.nan), axis=0)
        return _xs_dollar_neutral(z)

    # ── full causal combination ─────────────────────────────────────────────
    def weight_history(self, prices: pd.DataFrame, funding: pd.DataFrame | None = None):
        """Return (combined_weights, sleeve_pnls) over the full history.

        prices  : wide DataFrame [date x symbol] of close prices (daily).
        funding : wide DataFrame [date x symbol] of per-day funding (long pays).
                  If None, the carry sleeve is omitted.

        NOTE (CHIEF 2026-05-31): SIMPLE returns are used for all portfolio P&L
        (Σ wᵢ·(Pₜ/Pₜ₋₁−1)). Log returns are INVALID for cross-sectional L/S P&L
        (they understate short losses on large moves -> inflated Sharpe). This is
        the correct accounting and matches the paper broker exactly.
        """
        R = prices / prices.shift(1) - 1.0
        R = R[R.notna().sum(axis=1) >= self.cfg.min_names]
        sleeves = {"statarb": self._statarb_weights(R), "lowvol": self._lowvol_weights(R)}
        if funding is not None:
            F = funding.reindex(R.index).reindex(columns=R.columns).fillna(0.0)
            sleeves["carry"] = self._carry_weights(F)
        else:
            F = None

        # sleeve P&L (weights applied to NEXT-day return -> shift(1); no lookahead)
        pnls = {}
        for name, W in sleeves.items():
            we = W.shift(1).fillna(0.0)
            pnl = (we * R).sum(axis=1)
            if name == "carry" and F is not None:
                pnl = pnl - (we * F).sum(axis=1)             # funding accrual
            pnls[name] = pnl
        pnl_df = pd.DataFrame(pnls)

        # causal risk-parity weights: inverse trailing-vol, info through t-1
        vol = pnl_df.rolling(self.cfg.rp_lookback, min_periods=self.cfg.rp_min_periods).std().shift(1)
        iv = (1.0 / vol.replace(0, np.nan))
        rp = iv.div(iv.sum(axis=1), axis=0)
        # before rp warms up, fall back to equal weight
        rp = rp.fillna(1.0 / len(sleeves))

        combined = None
        for name, W in sleeves.items():
            term = W.mul(rp[name], axis=0)
            combined = term if combined is None else combined.add(term, fill_value=0.0)
        combined = _xs_dollar_neutral(combined)              # re-normalise gross to 1
        return combined, pnl_df

    def target_weights(self, prices: pd.DataFrame, funding: pd.DataFrame | None = None) -> pd.Series:
        """Target dollar-neutral weights for the LATEST bar (to trade next bar)."""
        combined, _ = self.weight_history(prices, funding)
        if combined.empty:
            return pd.Series(0.0, index=prices.columns)
        return combined.iloc[-1].reindex(prices.columns).fillna(0.0)
