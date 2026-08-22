# src/tradebot/alpha/csm_volume_clock.py
"""Cross-Sectional Momentum on volume-clock bars (Tier 2 — T2.1).

Unlike ``CSMomentum`` (calendar-time), this signal measures momentum over
50 *macro-runs-bars* (volume-clock) so the lookback adapts to the pace
of trade activity rather than the passage of wall-clock time.

Why volume-clock (López de Prado AFML §2)?
  Calendar-time momentum is dominated by volatility clustering: a 30-day
  window in a high-activity period covers far more economic information
  than the same 30 calendar days in a low-activity period.  Volume-clock
  windows equalize information content, producing more stationary returns
  and better t-statistics on the momentum signal.

Design:
  1. Each bar represents a fixed amount of total volume (macro-runs-bar).
  2. Momentum = log(P[t]) / log(P[t - macro_window_bars]).
  3. Cross-sectional rank-normalization: within a multi-asset universe
     the raw momentums are rank-normalized to [-1, +1].  For single-asset
     backtests we fall back to own rolling z-score.
  4. Vol-scaling: momentum / realised_vol_annualized (TSMOM style).
  5. Judge gate: if ``prob_judge`` is supplied and below ``judge_threshold``,
     the signal is suppressed regardless of momentum magnitude.
"""
from __future__ import annotations

import hashlib
import logging

import numpy as np
import pandas as pd

from .base import SignalResult

logger = logging.getLogger(__name__)

__all__ = ["CSMVolumeClockSignal", "rank_normalize_cross_section"]


def _params_hash(**kwargs: object) -> str:
    s = "_".join(f"{k}={v}" for k, v in sorted(kwargs.items()))
    return hashlib.md5(s.encode()).hexdigest()[:8]


def rank_normalize_cross_section(
    signals: dict[str, float],
    min_universe: int = 2,
) -> dict[str, float]:
    """Rank-normalize a dict of {symbol: raw_signal} to [-1, +1].

    Uses fractional ranking (average ties) and maps ranks linearly to [-1, +1].

    Parameters
    ----------
    signals :
        Raw signal values per symbol.  NaN values are excluded from ranking.
    min_universe :
        Minimum number of non-NaN symbols needed to produce a meaningful
        cross-sectional rank.  If fewer, returns zeros for all.

    Returns
    -------
    Dict[str, float] : rank-normalized signals in [-1, +1].
    """
    valid = {k: v for k, v in signals.items() if np.isfinite(v)}
    if len(valid) < min_universe:
        return {k: 0.0 for k in signals}

    syms = list(valid.keys())
    vals = np.array([valid[s] for s in syms], dtype=np.float64)

    # Fractional ranking (argsort twice = rank)
    n = len(vals)
    sorted_idx = np.argsort(vals)
    ranks = np.empty(n)
    ranks[sorted_idx] = np.arange(1, n + 1, dtype=np.float64)

    # Map [1, n] → [-1, +1]
    if n == 1:
        normed = np.zeros(n)
    else:
        normed = 2.0 * (ranks - 1.0) / (n - 1.0) - 1.0

    result = {k: 0.0 for k in signals}
    for sym, norm_val in zip(syms, normed):
        result[sym] = float(norm_val)
    return result


class CSMVolumeClockSignal:
    """Cross-Sectional Momentum on volume-clock runs bars.

    Measures momentum over ``macro_window_bars`` macro-runs-bars.  When used
    in a multi-asset portfolio, call :func:`rank_normalize_cross_section` on
    a dict of ``.predict()`` signal values to get cross-sectionally normalised
    signals.

    Parameters
    ----------
    symbol :
        Ticker name.
    macro_window_bars :
        Lookback in macro-runs-bars.  Default 50 (≈ 50 × 4 h ≈ 8-day window
        at typical crypto macro-bar frequency).
    vol_scale :
        If True, scale by 63-bar realised vol (annualised) to make the signal
        stationary across volatility regimes.
    skip_bars :
        Number of most-recent macro-bars to skip (microstructure reversal).
        Default 2 (analogous to Jegadeesh-Titman skip-month on macro bars).
    judge_threshold :
        If ``prob_judge`` is passed to ``predict()`` and is below this value,
        the signal is zeroed (Judge gate).  Set to 0.0 to disable.
    """

    def __init__(
        self,
        symbol: str,
        macro_window_bars: int = 50,
        vol_scale: bool = True,
        skip_bars: int = 2,
        judge_threshold: float = 0.55,
    ) -> None:
        self.symbol = symbol
        self.macro_window_bars = macro_window_bars
        self.vol_scale = vol_scale
        self.skip_bars = skip_bars
        self.judge_threshold = judge_threshold
        self.signal_id = (
            f"{symbol}_csm_vc_"
            + _params_hash(mw=macro_window_bars, vs=vol_scale, sk=skip_bars)
        )
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> None:
        """Cache close prices for subsequent predict calls.

        ``df`` is expected to be a macro-runs-bar DataFrame with a ``close``
        column and a UTC-aware DatetimeIndex.
        """
        if "close" not in df.columns:
            raise ValueError("CSMVolumeClockSignal.fit: DataFrame must have 'close' column.")
        self._macro_closes: pd.Series = df["close"].copy()
        self._fitted = True

    def predict(
        self,
        df: pd.DataFrame,
        prob_judge: float | None = None,
    ) -> SignalResult:
        """Compute momentum signal on latest macro-bar.

        Parameters
        ----------
        df :
            Macro-bar DataFrame up to and including the current bar.  Must
            contain ``close``.
        prob_judge :
            Optional Judge probability.  If below ``self.judge_threshold``,
            the signal is zeroed (trade suppressed).

        Returns
        -------
        SignalResult with signal in [-1, +1].
        """
        ts = df.index[-1]
        prices: pd.Series = df["close"]
        needed = self.macro_window_bars + self.skip_bars + 1

        zero_result = SignalResult(self.symbol, ts, 0.0, 0.5, self.macro_window_bars, self.signal_id)

        if len(prices) < needed:
            return zero_result

        # Judge gate: suppress signal if Judge confidence is too low.
        if prob_judge is not None and float(prob_judge) < self.judge_threshold:
            logger.debug(
                "[%s] CSMVolumeClockSignal: Judge prob %.3f < %.2f → signal suppressed.",
                self.symbol, prob_judge, self.judge_threshold,
            )
            return zero_result

        # Lookback price (skip micro-reversal zone)
        p_now  = float(prices.iloc[-(self.skip_bars + 1)])
        p_past = float(prices.iloc[-(self.macro_window_bars + self.skip_bars + 1)])

        if p_past <= 0 or p_now <= 0:
            return zero_result

        raw_momentum = float(np.log(p_now / p_past))

        if self.vol_scale:
            log_rets = np.log(prices.values[1:] / prices.values[:-1])
            window = min(63, len(log_rets))
            ann_vol = float(np.std(log_rets[-window:]) * np.sqrt(252)) + 1e-9
            raw_momentum /= ann_vol

        # Map to [-1, +1] via tanh (steeper than 1.0 to create more conviction)
        signal = float(np.tanh(raw_momentum * 0.5))
        confidence = float(0.5 + 0.5 * signal)

        return SignalResult(
            symbol=self.symbol,
            timestamp=ts,
            signal=signal,
            confidence=confidence,
            horizon_bars=self.macro_window_bars,
            signal_id=self.signal_id,
        )

    def feature_names(self) -> list[str]:
        return ["close"]


# =============================================================================
# Portfolio-level cross-sectional ranking helper
# =============================================================================

def compute_csm_volume_clock_signals(
    signals: dict[str, CSMVolumeClockSignal],
    dfs: dict[str, pd.DataFrame],
    judge_probs: dict[str, float] | None = None,
    min_universe: int = 2,
    lag_bars: int = 1,
) -> dict[str, SignalResult]:
    """Compute and rank-normalize CSM signals across the full universe.

    Parameters
    ----------
    signals :
        Dict of ``{symbol: CSMVolumeClockSignal}`` instances.
    dfs :
        Dict of ``{symbol: macro_bar_df}`` for the current bar.
    judge_probs :
        Optional ``{symbol: prob_judge}`` from the Meta-Labeling Judge.
    min_universe :
        Minimum assets for cross-sectional ranking.
    lag_bars :
        CHIEF AUDIT 2026-05-23 (P-2): Number of bars to lag the input to
        guarantee causality.  When ``lag_bars > 0`` the per-symbol
        ``predict()`` call receives ``df.iloc[:-lag_bars]`` (features computed
        on bars ≤ T - lag_bars) while the SignalResult timestamp is forced to
        ``df.index[-1]`` (= bar T).  The caller may therefore use the returned
        signal for a trade entering at bar T + lag_bars without leakage.
        Default 1 = causal-safe (signal_t uses bars ≤ t-1, trade at t).

    Returns
    -------
    Dict[str, SignalResult] with rank-normalized signals.  The output signal
    is the signal for a trade at bar T + lag_bars (T = df.index[-1]).
    """
    raw: dict[str, SignalResult] = {}
    raw_vals: dict[str, float] = {}

    for sym, sig in signals.items():
        if sym not in dfs:
            continue
        pj = (judge_probs or {}).get(sym)
        df_sym = dfs[sym]
        # CHIEF AUDIT 2026-05-23 (P-2): apply explicit lag for causal safety.
        if lag_bars > 0 and len(df_sym) > lag_bars:
            df_pred = df_sym.iloc[:-lag_bars]
            ts_final = df_sym.index[-1]
        else:
            df_pred = df_sym
            ts_final = df_sym.index[-1] if len(df_sym) > 0 else None
        result = sig.predict(df_pred, prob_judge=pj)
        # Re-anchor the timestamp to the latest observed bar so the caller
        # knows which bar this lagged signal corresponds to.
        if ts_final is not None and result.timestamp != ts_final:
            result = SignalResult(
                symbol=result.symbol,
                timestamp=ts_final,
                signal=result.signal,
                confidence=result.confidence,
                horizon_bars=result.horizon_bars,
                signal_id=result.signal_id,
            )
        raw[sym] = result
        raw_vals[sym] = result.signal

    ranked = rank_normalize_cross_section(raw_vals, min_universe=min_universe)

    # Rebuild SignalResult with rank-normalized signal
    out: dict[str, SignalResult] = {}
    for sym, sr in raw.items():
        normed_signal = ranked.get(sym, 0.0)
        out[sym] = SignalResult(
            symbol=sr.symbol,
            timestamp=sr.timestamp,
            signal=normed_signal,
            confidence=float(0.5 + 0.5 * normed_signal),
            horizon_bars=sr.horizon_bars,
            signal_id=sr.signal_id,
        )
    return out
