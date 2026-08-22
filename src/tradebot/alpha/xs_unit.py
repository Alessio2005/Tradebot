# src/tradebot/alpha/xs_unit.py
"""Cross-sectional unit harness — the shared machinery for every XS premium.

Mandate v3 §5.6 discipline, baked in:
  * SIMPLE first: rank-demeaned, dollar-neutral, fixed rebalance calendar,
    fixed selection fraction. No tunable harvest layer (F12).
  * Causal by construction: weights computed from data <= t apply to the
    t -> t+1 return (``w.shift(1) * ret``). Guarded in tests/lookahead/.
  * PIT universe: a symbol only enters the cross-section when the
    membership calendar (built from ``asof_ts``) says it is tradeable.
  * Honest costs: commission + half-spread on turnover, borrow fee on the
    short book, all configurable per liquidity class — never zero.

Every equity/FX/commodity XS unit (eq_xsmom, eq_strev, eq_lowvol, ...) is a
thin signal definition on top of this harness.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = [
    "CostModel",
    "XSUnitResult",
    "decile_weights",
    "event_to_panel",
    "run_xs_unit",
    "ann_sharpe",
]

TRADING_DAYS = 252


@dataclass(frozen=True)
class CostModel:
    """Per-side trading costs in decimal terms (Mandate §5.2 cost model).

    commission_bps : per-side commission (IBKR-style retail ~0.5-1 bp).
    half_spread_bps: effective half-spread per liquidity class.
    borrow_fee_ann : annualised borrow fee on the short book (general
                     collateral ~0.25-0.50%; hard-to-borrow names must be
                     EXCLUDED from the short side upstream, not priced here).
    """

    commission_bps: float = 1.0
    half_spread_bps: float = 5.0
    borrow_fee_ann: float = 0.005

    @property
    def per_side(self) -> float:
        return (self.commission_bps + self.half_spread_bps) / 1e4


@dataclass(frozen=True)
class XSUnitResult:
    """Everything a wave-log entry and the G4 lab need from one unit run."""

    unit: str
    net_returns: pd.Series
    gross_returns: pd.Series
    weights: pd.DataFrame
    daily_turnover: pd.Series
    config: dict

    def summary(self, start: str | None = None) -> dict:
        """Metrics, optionally from ``start`` onward (evaluation window —
        e.g. 2000+ where the PIT membership history is reliable; the
        warmup/lookback before ``start`` still used the full panel)."""
        net, gross = self.net_returns.dropna(), self.gross_returns.dropna()
        if start is not None:
            net, gross = net.loc[start:], gross.loc[start:]
        years = net.groupby(net.index.year).apply(lambda r: (1 + r).prod() - 1)
        return {
            "unit": self.unit,
            "n_days": int(len(net)),
            "net_sharpe": ann_sharpe(net),
            "gross_sharpe": ann_sharpe(gross),
            "net_cagr": float((1 + net).prod() ** (TRADING_DAYS / max(len(net), 1)) - 1),
            "avg_daily_turnover": float(
                self.daily_turnover.loc[start:].mean() if start is not None
                else self.daily_turnover.mean()
            ),
            "eval_start": start,
            "per_year_net": {int(y): float(v) for y, v in years.items()},
            "config": self.config,
        }


def ann_sharpe(r: pd.Series, periods: int = TRADING_DAYS) -> float:
    r = r.dropna()
    if len(r) < 2 or r.std() == 0:
        return float("nan")
    return float(r.mean() / r.std() * np.sqrt(periods))


def decile_weights(
    scores: pd.Series, top_frac: float = 0.3, min_names: int = 10
) -> pd.Series:
    """Dollar-neutral long-top / short-bottom weights from one date's scores.

    Equal-weight within each leg, gross exposure = 1.0 (0.5 per leg).
    Returns an all-zero vector when the cross-section is too thin —
    a thin universe must show up as "no position", never as a 2-name bet.
    """
    s = scores.dropna()
    if len(s) < min_names:
        return pd.Series(0.0, index=scores.index)
    k = max(int(np.floor(len(s) * top_frac)), 1)
    ranked = s.rank(method="first")
    w = pd.Series(0.0, index=scores.index)
    w[ranked[ranked > len(s) - k].index] = 0.5 / k
    w[ranked[ranked <= k].index] = -0.5 / k
    return w


def event_to_panel(
    events: pd.DataFrame,
    index: pd.DatetimeIndex,
    columns: pd.Index,
    expiry_bdays: int,
    symbol_col: str = "symbol",
    asof_col: str = "asof_ts",
    value_col: str = "value",
) -> pd.DataFrame:
    """Sparse PIT events -> dense signal panel with expiry.

    Cell (t, s) = the latest event value for ``s`` whose ``asof_ts`` <= t,
    but only while the event is at most ``expiry_bdays`` trading days old —
    afterwards NaN (stale fundamentals/announcements must drop out, never
    linger silently). Strictly as-of: future events can never appear.
    """
    out = pd.DataFrame(np.nan, index=index, columns=columns)
    idx_vals = index.tz_convert(None).values if index.tz is not None else index.values
    ev = events.sort_values(asof_col, kind="stable")
    for sym, grp in ev.groupby(symbol_col):
        if sym not in out.columns:
            continue
        col = out[sym].to_numpy()
        for asof, val in zip(grp[asof_col], grp[value_col]):
            a = pd.Timestamp(asof)
            a64 = (a.tz_convert(None) if a.tzinfo else a).to_datetime64()
            i0 = int(np.searchsorted(idx_vals, a64, side="left"))
            if i0 >= len(index):
                continue
            col[i0 : i0 + expiry_bdays] = val
        out[sym] = col
    return out


def run_xs_unit(
    prices: pd.DataFrame,
    signal_panel: pd.DataFrame,
    unit: str,
    membership: pd.DataFrame | None = None,
    rebalance: str = "ME",
    cost: CostModel = CostModel(),
    top_frac: float = 0.3,
    min_names: int = 10,
    beta_panel: pd.DataFrame | None = None,
    config: dict | None = None,
) -> XSUnitResult:
    """Run one cross-sectional unit end-to-end (simple, fixed harvest).

    Parameters
    ----------
    prices : wide close panel (UTC DatetimeIndex x symbols), point-in-time.
    signal_panel : same shape; row t may use information up to AND INCLUDING
        the close of t (it is acted on at t+1 — the harness shifts).
        Causality of the panel itself is the unit's responsibility and is
        guarded by the lookahead suite.
    membership : optional boolean PIT calendar (dates x symbols); a symbol
        with False (or missing) at t gets no score at t.
    rebalance : pandas offset alias for the rebalance calendar ("ME"
        month-end per the literature; FIXED per unit, never swept — F12).
    beta_panel : optional rolling-beta panel (same shape, causal <= t).
        When given, each leg is scaled by the inverse of its average beta
        (Frazzini-Pedersen BAB construction): the book is ex-ante
        beta-NEUTRAL instead of dollar-neutral-but-short-beta. Without
        this, a vol/beta-sorted book structurally bleeds short-beta in a
        rising market (W22 run-3 lesson: lowvol Sharpe -0.82 was beta
        bleed, not premium absence).
    """
    if not isinstance(prices.index, pd.DatetimeIndex) or prices.index.tz is None:
        raise ValueError("prices must have a tz-aware UTC DatetimeIndex")
    if not prices.index.equals(signal_panel.index):
        raise ValueError("signal_panel must share the prices index")

    rets = prices.pct_change(fill_method=None)

    sig = signal_panel.copy()
    if membership is not None:
        mask = membership.reindex(index=sig.index, columns=sig.columns).fillna(False)
        sig = sig.where(mask)

    # Rebalance on the last trading day of each period, decided by comparing
    # with the NEXT bar only. Taking ``groupby(period).max()`` instead makes
    # the calendar depend on how far the data happens to run: the final bar of
    # a truncated panel is always the max of its period, so it becomes a
    # phantom rebalance that live trading would not do. Harmless once in a
    # full-sample backtest; fires at EVERY fold boundary under walk-forward.
    # At the ragged edge the next period is unknown, so the final bar never
    # rebalances. (Period grouping on the tz-naive view — calendar-based,
    # avoids the pandas tz-drop warning.) Guarded by
    # tests/lookahead/test_eq_units_causality.py::test_truncation_invariance.
    period_alias = {"ME": "M", "WE": "W", "D": "D"}.get(rebalance, rebalance)
    period = sig.index.tz_convert(None).to_period(period_alias)
    is_rb = np.zeros(len(period), dtype=bool)
    is_rb[:-1] = period[:-1] != period[1:]
    rb_dates = sig.index[is_rb]

    def _weights_at(t: pd.Timestamp) -> pd.Series:
        w = decile_weights(sig.loc[t], top_frac=top_frac, min_names=min_names)
        if beta_panel is not None and (w != 0).any():
            beta_t = beta_panel.loc[t].reindex(w.index)
            long_m, short_m = w > 0, w < 0
            for mask in (long_m, short_m):
                leg_beta = float(beta_t[mask].mean())
                # BAB leg-leverage to |beta| = 1. Unknown beta (NaN, early
                # window) => no scaling; floor 0.25 bounds the leverage.
                scale = leg_beta if np.isfinite(leg_beta) else 1.0
                w[mask] = w[mask] / max(scale, 0.25)
        return w

    w_rb = pd.DataFrame({t: _weights_at(t) for t in rb_dates}).T
    w_rb.index = rb_dates
    # hold between rebalances
    weights = w_rb.reindex(sig.index).ffill().fillna(0.0)

    held = weights.shift(1).fillna(0.0)          # w(t-1) earns r(t)
    gross = (held * rets).sum(axis=1)

    turnover = (weights - held).abs().sum(axis=1)
    trade_cost = turnover * cost.per_side
    borrow = held.clip(upper=0.0).abs().sum(axis=1) * (cost.borrow_fee_ann / TRADING_DAYS)
    net = gross - trade_cost - borrow

    return XSUnitResult(
        unit=unit,
        net_returns=net,
        gross_returns=gross,
        weights=weights,
        daily_turnover=turnover,
        config={
            "rebalance": rebalance,
            "top_frac": top_frac,
            "min_names": min_names,
            "beta_neutral": beta_panel is not None,
            "commission_bps": cost.commission_bps,
            "half_spread_bps": cost.half_spread_bps,
            "borrow_fee_ann": cost.borrow_fee_ann,
            **(config or {}),
        },
    )
