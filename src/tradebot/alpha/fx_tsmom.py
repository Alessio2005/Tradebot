# src/tradebot/alpha/fx_tsmom.py
"""Wave 26a — G10 FX time-series momentum (Moskowitz, Ooi & Pedersen 2012).

Prior: "Time Series Momentum" (JFE 2012) — an instrument's own trailing
12-month excess return predicts its next-month return; go long past winners
/ short past losers, size each leg to equal ex-ante risk. Documented across
58 instruments incl. FX, decades, and survives costs in liquid majors.

This is a DIFFERENT premium class than crypto trend (F7/F8 falsified crypto
trend specifically): the managed-futures TSMOM premium is academically real
in FX/commodities/rates (Mandate §5.3, §8). It is also TIME-SERIES, not
cross-sectional, so it does not use the XS harness — each currency is sized
on its own signal, the book is NOT dollar-neutral by construction.

Construction (literature-conform, FIXED):
  signal(t) = sign( TR(t-1) / TR(t-1-252) - 1 )   (own 12m trailing return)
  weight(t) = signal(t) / vol_i(t)   (inverse-vol risk scaling, 60d vol),
              normalised to gross 1, held to next month-end rebalance.
Costs: majors 1 bp half-spread on turnover (FX swap-implied; no borrow).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from tradebot.alpha.xs_unit import CostModel, ann_sharpe

__all__ = ["UNIT", "PRIOR", "run", "FXTSMomResult"]

UNIT = "fx_tsmom_g10"
PRIOR = "Moskowitz, Ooi & Pedersen (2012), J. Fin. Econ. 104(2), 228-250"

_LOOKBACK = 252
_VOL_WINDOW = 60
_COST = CostModel(commission_bps=0.0, half_spread_bps=1.0, borrow_fee_ann=0.0)
_TRADING_DAYS = 252


@dataclass(frozen=True)
class FXTSMomResult:
    unit: str
    net_returns: pd.Series
    gross_returns: pd.Series
    weights: pd.DataFrame
    daily_turnover: pd.Series
    config: dict

    def summary(self, start: str | None = None) -> dict:
        net = self.net_returns.dropna()
        gross = self.gross_returns.dropna()
        if start is not None:
            net, gross = net.loc[start:], gross.loc[start:]
        years = net.groupby(net.index.year).apply(lambda r: (1 + r).prod() - 1)
        return {
            "unit": self.unit,
            "n_days": int(len(net)),
            "net_sharpe": ann_sharpe(net, _TRADING_DAYS),
            "gross_sharpe": ann_sharpe(gross, _TRADING_DAYS),
            "net_cagr": float((1 + net).prod() ** (_TRADING_DAYS / max(len(net), 1)) - 1),
            "avg_daily_turnover": float(
                self.daily_turnover.loc[start:].mean() if start is not None
                else self.daily_turnover.mean()
            ),
            "eval_start": start,
            "per_year_net": {int(y): float(v) for y, v in years.items()},
            "config": self.config,
        }


def run(tr_index: pd.DataFrame, cost: CostModel = _COST) -> FXTSMomResult:
    """Time-series momentum book over the G10 TR panel (see module docstring)."""
    if not isinstance(tr_index.index, pd.DatetimeIndex) or tr_index.index.tz is None:
        raise ValueError("tr_index must have a tz-aware UTC DatetimeIndex")

    rets = tr_index.pct_change(fill_method=None)
    # own 12m trailing return, known at t (acted on t+1 via shift below)
    mom = tr_index / tr_index.shift(_LOOKBACK) - 1.0
    sig = np.sign(mom)
    # inverse-vol risk scaling, causal (vol from returns <= t)
    vol = rets.rolling(_VOL_WINDOW).std()
    raw = (sig / vol).replace([np.inf, -np.inf], np.nan)

    # Rebalance month-end; hold between. Decided by comparing with the NEXT
    # bar only: ``groupby(period).max()`` makes the final bar of a truncated
    # panel the max of its month and therefore a phantom rebalance that live
    # trading would not do (it fires at every fold boundary under walk-forward).
    # At the ragged edge the next month is unknown, so the final bar never
    # rebalances. Guarded by tests/lookahead/test_fx_tsmom_causality.py.
    period = raw.index.tz_convert(None).to_period("M")
    is_rb = np.zeros(len(period), dtype=bool)
    is_rb[:-1] = period[:-1] != period[1:]
    rb_dates = raw.index[is_rb]
    w_rb = raw.loc[rb_dates]
    # normalise each rebalance row to gross exposure 1 (sum |w| = 1)
    gross_abs = w_rb.abs().sum(axis=1).replace(0, np.nan)
    w_rb = w_rb.div(gross_abs, axis=0)
    weights = w_rb.reindex(raw.index).ffill().fillna(0.0)

    held = weights.shift(1).fillna(0.0)          # w(t-1) earns r(t)
    gross = (held * rets).sum(axis=1)
    turnover = (weights - held).abs().sum(axis=1)
    net = gross - turnover * cost.per_side

    return FXTSMomResult(
        unit=UNIT,
        net_returns=net,
        gross_returns=gross,
        weights=weights,
        daily_turnover=turnover,
        config={
            "lookback": _LOOKBACK,
            "vol_window": _VOL_WINDOW,
            "rebalance": "ME",
            "half_spread_bps": cost.half_spread_bps,
            "prior": PRIOR,
        },
    )
