#!/usr/bin/env python
"""W28 step 0.2 — exact impact of the phantom-rebalance repair on archived units.

The repair changes the rebalance calendar ONLY at the ragged edge: the final
bar of the panel used to be a rebalance (it is always the max of its period),
now it never is. In ``run_xs_unit`` the final weights row enters net P&L
through exactly one term:

    held     = weights.shift(1)
    gross[t] = held[t] . rets[t]          -> uses weights[t-1], not weights[t]
    borrow[t]= f(held[t])                 -> uses weights[t-1], not weights[t]
    turnover[t] = |weights[t] - weights[t-1]|

so weights[-1] appears only in ``turnover[-1]``, i.e. the old code charged one
extra round of trading cost on the last bar and nothing else. The Sharpe delta
is therefore exactly computable without re-running anything, and the direction
is known a priori: removing a cost can only IMPROVE an archived number.

That direction is why this has to be measured. ``eq_strev_1m`` was archived at
net Sharpe +0.39 against a pre-registered lat of 0.40 (WAVE_LOG W22/W24) — 0.01
away from a verdict flip. "Negligible" is not an argument there; a number is.

The Wave-21 equity panel is not present in this working copy, so the bound is
established two ways:
  (1) EMPIRICALLY on the real 26-instrument cross-asset panel (2004-2026,
      n=5685) through the same ``run_xs_unit`` harness the equity units use;
  (2) ANALYTICALLY as a closed-form worst case for the equity configuration.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tradebot.alpha.xs_unit import (
    CostModel,
    ann_sharpe,
    decile_weights,
    run_xs_unit,
)

PANEL = "market_data_parquet/xasset/tr_panel.parquet"
TRADING_DAYS = 252


def _load_panel() -> pd.DataFrame:
    df = pd.read_parquet(PANEL)
    wide = df.pivot(index="event_ts", columns="symbol", values="close")
    wide.index = pd.to_datetime(wide.index, utc=True)
    return wide.sort_index()


def empirical() -> dict[str, float]:
    """Exact repair delta on a real 22-year daily panel."""
    prices = _load_panel()
    # 1m reversal signal, the eq_strev construction, on this panel
    sig = -(prices / prices.shift(21) - 1.0)
    cost = CostModel(half_spread_bps=6.0, commission_bps=1.0)

    top_frac, min_names = 0.3, 6
    res = run_xs_unit(
        prices, sig, "phantom_probe", cost=cost, rebalance="ME",
        top_frac=top_frac, min_names=min_names,
    )
    net_new = res.net_returns

    # What the OLD calendar would have done: the final bar rebalances. Same
    # top_frac/min_names as the run above, or the two paths are not comparable.
    w_phantom = decile_weights(
        sig.iloc[-1], top_frac=top_frac, min_names=min_names
    )
    w_prev = res.weights.iloc[-1]  # under the repair the last bar is a hold
    phantom_turnover = float((w_phantom - w_prev).abs().sum())
    extra_cost = phantom_turnover * cost.per_side

    net_old = net_new.copy()
    net_old.iloc[-1] -= extra_cost

    s_new, s_old = ann_sharpe(net_new), ann_sharpe(net_old)
    return {
        "n_bars": len(net_new),
        "phantom_turnover": phantom_turnover,
        "extra_cost_at_final_bar": extra_cost,
        "sharpe_repaired": s_new,
        "sharpe_with_phantom": s_old,
        "delta_sharpe": s_new - s_old,
    }


def analytic_worst_case(n_bars: int = 6500, ann_vol: float = 0.10) -> dict[str, float]:
    """Closed-form ceiling for the equity units (cannot be exceeded).

    Worst case = a complete flip of a fully-invested dollar-neutral decile
    book: turnover 2.0 at 7 bps per side. Sharpe ~ mean/sd*sqrt(252), and only
    the mean moves (one bar out of n).
    """
    turnover, per_side = 2.0, 7e-4
    extra_cost = turnover * per_side
    d_mean_daily = extra_cost / n_bars
    d_sharpe = d_mean_daily * np.sqrt(TRADING_DAYS) / (ann_vol / np.sqrt(TRADING_DAYS))
    return {
        "assumed_n_bars": n_bars,
        "assumed_ann_vol": ann_vol,
        "max_extra_cost": extra_cost,
        "max_delta_sharpe": float(d_sharpe),
    }


def main() -> int:
    emp = empirical()
    ana = analytic_worst_case()

    print("== empirical (real 26-instrument panel, run_xs_unit) ==")
    for k, v in emp.items():
        print(f"  {k:28s} {v}")
    print("\n== analytic worst case (equity config, full decile flip) ==")
    for k, v in ana.items():
        print(f"  {k:28s} {v}")

    gap = 0.40 - 0.39  # eq_strev_1m distance to its pre-registered lat
    print(f"\n  eq_strev_1m gap to lat            {gap:.4f}")
    print(f"  worst-case repair gain            {ana['max_delta_sharpe']:.6f}")
    verdict = "NO FLIP" if ana["max_delta_sharpe"] < gap else "FLIP POSSIBLE — RERUN"
    print(f"  verdict                           {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
