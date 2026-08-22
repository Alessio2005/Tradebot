# src/tradebot/alpha/combination.py
"""IC-weighted signal combination with IC-DAMP shrinkage.

Reference: López de Prado (2018) AFML §7 — signal combination.
IC-DAMP (Information Coefficient Decay Adjusted Mean Predictor):
    IC_shrunk = IC * (IC² / (IC² + (1 - IC²) / (T - 2)))
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
import warnings

from scipy.stats import ConstantInputWarning, spearmanr

logger = logging.getLogger(__name__)

__all__ = ["ICWeightedCombiner"]


def _ic_damp(ic: float, n_obs: int) -> float:
    """IC-DAMP shrinkage of a single IC estimate.

    Shrinks the IC towards zero proportionally to the estimation uncertainty.
    Parameters
    ----------
    ic : raw Spearman IC
    n_obs : number of observations used to compute ic
    """
    if n_obs <= 2 or not np.isfinite(ic):
        return 0.0
    ic2 = ic ** 2
    shrunk = ic2 / (ic2 + (1.0 - ic2) / max(n_obs - 2, 1))
    # Preserve sign
    return float(np.sign(ic) * np.sqrt(max(shrunk, 0.0)))


class ICWeightedCombiner:
    """Information Coefficient-weighted ensemble of alpha signals.

    Each period:
        IC_i = Spearman(signal_i[t-L..t-1], fwd_return[t-L..t-1])
        IC_shrunk_i = IC-DAMP(IC_i, L)
        w_i = IC_shrunk_i / sum(|IC_shrunk|)
        s_combined = sum(w_i * s_i)

    Parameters
    ----------
    signal_names :
        List of signal column names.  Must match columns of the DataFrame
        passed to ``fit`` / ``predict``.
    lookback :
        Rolling window for IC estimation (in bars).
    min_ic_abs :
        Signals with |IC_shrunk| below this threshold are zero-weighted.
    """

    def __init__(
        self,
        signal_names: list[str],
        lookback: int = 252,
        min_ic_abs: float = 0.02,
    ) -> None:
        self.signal_names = signal_names
        self.lookback = lookback
        self.min_ic_abs = min_ic_abs
        self._weights: Optional[pd.Series] = None
        self._ic_table: Optional[pd.DataFrame] = None

    def fit(
        self,
        signals: pd.DataFrame,
        fwd_returns: pd.Series,
    ) -> None:
        """Estimate IC weights from historical signals and forward returns.

        Parameters
        ----------
        signals :
            DataFrame with columns = signal names, index = timestamps.
        fwd_returns :
            Series of forward returns (same index as ``signals``).
        """
        common_idx = signals.index.intersection(fwd_returns.index)
        sigs = signals.loc[common_idx, self.signal_names].tail(self.lookback)
        rets = fwd_returns.loc[common_idx].tail(self.lookback)

        n_obs = len(sigs)
        records = []
        for name in self.signal_names:
            col = sigs[name].values
            mask = np.isfinite(col) & np.isfinite(rets.values)
            if mask.sum() < 10:
                records.append({"signal": name, "ic_raw": 0.0, "ic_shrunk": 0.0, "weight": 0.0})
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConstantInputWarning)
                ic_raw, _ = spearmanr(col[mask], rets.values[mask])
            ic_raw = float(ic_raw) if np.isfinite(ic_raw) else 0.0
            ic_shrunk = _ic_damp(ic_raw, int(mask.sum()))
            records.append({"signal": name, "ic_raw": ic_raw, "ic_shrunk": ic_shrunk, "weight": 0.0})

        self._ic_table = pd.DataFrame(records).set_index("signal")

        total_abs_ic = float(self._ic_table["ic_shrunk"].abs().sum())
        if total_abs_ic < 1e-9:
            # No predictive signals — equal-weight
            n = len(self.signal_names)
            self._ic_table["weight"] = 1.0 / n if n > 0 else 0.0
        else:
            raw_w = self._ic_table["ic_shrunk"] / total_abs_ic
            # Zero-out signals below minimum IC threshold
            raw_w[self._ic_table["ic_shrunk"].abs() < self.min_ic_abs] = 0.0
            total_w = raw_w.abs().sum()
            self._weights = raw_w / total_w if total_w > 1e-9 else raw_w
            self._ic_table["weight"] = self._weights

        logger.info(
            "ICWeightedCombiner fitted: n_signals=%d, n_obs=%d",
            len(self.signal_names),
            n_obs,
        )

    def predict(self, signals: pd.DataFrame) -> pd.Series:
        """Compute IC-weighted combined signal for each row in ``signals``.

        Parameters
        ----------
        signals : DataFrame with columns = signal names.

        Returns
        -------
        pd.Series — combined signal in [-1, +1].
        """
        if self._ic_table is None:
            raise RuntimeError("Call fit() before predict().")

        weights = self._ic_table["weight"]
        result = pd.Series(0.0, index=signals.index)
        for name in self.signal_names:
            if name in signals.columns:
                result += signals[name].fillna(0.0) * float(weights.get(name, 0.0))

        return result.clip(-1.0, 1.0).rename("combined_signal")

    def ic_table(self) -> pd.DataFrame:
        """Return IC diagnostics table (signal → ic_raw, ic_shrunk, weight)."""
        if self._ic_table is None:
            raise RuntimeError("Call fit() first.")
        return self._ic_table.copy()
