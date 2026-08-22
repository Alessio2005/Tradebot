# src/tradebot/alpha/cm_tsmom.py
"""Wave 27 — cross-asset time-series momentum (Moskowitz, Ooi & Pedersen 2012).

Prior: "Time Series Momentum" (JFE 2012, 104(2) 228-250) — an instrument's own
trailing 12-month excess return predicts its next-month return, across 58
futures spanning commodities, rates, FX and equity indices. The managed-futures
premium is documented in THOSE markets; F7/F8 falsified crypto trend only, and
both name this asset class as the explicit reopening condition.

Design decisions, fixed BEFORE the first backtest (F12 — no tuning layer):
  * lookback 252d, vol window 60d, monthly rebalance, sign(momentum) — the
    literature parameterisation, not a swept one.
  * inverse-vol sizing, gross exposure normalised to 1.
  * TIME-SERIES, not cross-sectional: the book is not dollar-neutral, so it
    carries directional risk that G4 must price out.

Difference from ``fx_tsmom`` (which this deliberately mirrors): the traded
instruments are ETFs/ETCs, so the short leg pays a BORROW fee. FX shorts are
swap-implied and free; pretending that carries over would understate costs.

Why ETF proxies and not futures continuations: see
``tradebot.data.xasset_proxy`` — free continuous series omit 7-25%/yr of roll
drag, measured.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from tradebot.alpha.xs_unit import CostModel, ann_sharpe

__all__ = ["UNIT", "PRIOR", "COST", "run", "XAssetTSMomResult", "effective_breadth"]

UNIT = "cm_tsmom_xasset"
PRIOR = "Moskowitz, Ooi & Pedersen (2012), J. Fin. Econ. 104(2), 228-250"

_LOOKBACK = 252
_VOL_WINDOW = 60
_TRADING_DAYS = 252

# Liquid US-listed ETFs via IBKR: commission ~1bp, half-spread ~3bp
# (conservative — the thin names DBA/WEAT/CPER sit near 5-10bp).
# Borrow: general-collateral ETFs ~0.5%/yr. The ETF's own TER is ALREADY
# inside the adjusted NAV, so it is not charged again here.
COST = CostModel(commission_bps=1.0, half_spread_bps=3.0, borrow_fee_ann=0.005)


@dataclass(frozen=True)
class XAssetTSMomResult:
    unit: str
    net_returns: pd.Series
    gross_returns: pd.Series
    weights: pd.DataFrame
    daily_turnover: pd.Series
    borrow_cost: pd.Series
    config: dict

    def summary(self, start: str | None = None) -> dict:
        net, gross = self.net_returns.dropna(), self.gross_returns.dropna()
        if start is not None:
            net, gross = net.loc[start:], gross.loc[start:]
        years = net.groupby(net.index.year).apply(lambda r: (1 + r).prod() - 1)
        n_pos = int((years > 0).sum())
        eq = (1 + net).cumprod()
        dd = float((eq / eq.cummax() - 1.0).min())
        ann_vol = float(net.std() * np.sqrt(_TRADING_DAYS))
        cagr = float((1 + net).prod() ** (_TRADING_DAYS / max(len(net), 1)) - 1)
        return {
            "unit": self.unit,
            "n_days": len(net),
            "net_sharpe": ann_sharpe(net, _TRADING_DAYS),
            "gross_sharpe": ann_sharpe(gross, _TRADING_DAYS),
            "net_cagr": cagr,
            "ann_vol": ann_vol,
            "max_drawdown": dd,
            "calmar": float(cagr / abs(dd)) if dd < 0 else float("nan"),
            "dd_over_vol": float(abs(dd) / ann_vol) if ann_vol > 0 else float("nan"),
            "years_positive_frac": n_pos / max(len(years), 1),
            "n_years": len(years),
            "avg_daily_turnover": float(self.daily_turnover.loc[net.index].mean()),
            "borrow_drag_ann": float(
                self.borrow_cost.loc[net.index].mean() * _TRADING_DAYS
            ),
            "per_year_net": {int(y): float(v) for y, v in years.items()},
            "config": self.config,
        }


def effective_breadth(returns: pd.DataFrame) -> float:
    """N_eff = N / (1 + (N-1)*rho_bar) — independent bets, not instrument count.

    The Fundamental Law uses INDEPENDENT breadth. Counting 26 instruments as
    26 bets is the F10 error; six equity-index ETFs are close to one bet.
    """
    r = returns.dropna(how="all")
    n = r.shape[1]
    if n < 2:
        return float(n)
    corr = r.corr().to_numpy()
    off = corr[~np.eye(n, dtype=bool)]
    rho_bar = float(np.nanmean(off))
    return float(n / (1.0 + (n - 1) * rho_bar))


def run(tr_panel: pd.DataFrame, cost: CostModel = COST) -> XAssetTSMomResult:
    """TSMOM book over a roll-inclusive total-return panel.

    ``tr_panel`` is indexed by bar timestamp (tz-aware UTC); weights formed on
    bar t are held into bar t+1, so no row ever earns its own signal.
    """
    if not isinstance(tr_panel.index, pd.DatetimeIndex) or tr_panel.index.tz is None:
        raise ValueError("tr_panel must have a tz-aware UTC DatetimeIndex")

    rets = tr_panel.pct_change(fill_method=None)
    mom = tr_panel / tr_panel.shift(_LOOKBACK) - 1.0
    sig = np.sign(mom)
    vol = rets.rolling(_VOL_WINDOW).std()
    raw = (sig / vol).replace([np.inf, -np.inf], np.nan)

    # Rebalance on the FIRST bar of each month, decided by comparing with the
    # PREVIOUS bar only. Selecting month-END bars instead (``groupby(period)
    # .max()``, as fx_tsmom does) makes the calendar depend on how far the data
    # happens to run: the final bar of a truncated panel is always the max of
    # its month, so it becomes a phantom rebalance that live trading would not
    # do. Caught by test_truncation_invariance_no_future_leakage.
    period = raw.index.tz_convert(None).to_period("M")
    is_rb = pd.Series(period != np.roll(period, 1), index=raw.index)
    is_rb.iloc[0] = True
    w_rb = raw.loc[is_rb.to_numpy()]
    w_rb = w_rb.div(w_rb.abs().sum(axis=1).replace(0, np.nan), axis=0)
    weights = w_rb.reindex(raw.index).ffill().fillna(0.0)

    held = weights.shift(1).fillna(0.0)  # w(t-1) earns r(t) — causal
    gross = (held * rets).sum(axis=1)
    turnover = (weights - held).abs().sum(axis=1)
    short_notional = held.clip(upper=0.0).abs().sum(axis=1)
    borrow = short_notional * cost.borrow_fee_ann / _TRADING_DAYS
    net = gross - turnover * cost.per_side - borrow

    return XAssetTSMomResult(
        unit=UNIT,
        net_returns=net,
        gross_returns=gross,
        weights=weights,
        daily_turnover=turnover,
        borrow_cost=borrow,
        config={
            "lookback": _LOOKBACK,
            "vol_window": _VOL_WINDOW,
            "rebalance": "ME",
            "commission_bps": cost.commission_bps,
            "half_spread_bps": cost.half_spread_bps,
            "borrow_fee_ann": cost.borrow_fee_ann,
            "n_instruments": int(tr_panel.shape[1]),
            "prior": PRIOR,
        },
    )
