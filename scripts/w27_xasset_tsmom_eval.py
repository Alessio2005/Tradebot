#!/usr/bin/env python
"""Wave 27 evaluation — cross-asset TSMOM on the roll-inclusive proxy panel.

    python scripts/w27_xasset_tsmom_eval.py

Prints the full wave record: unit summary, per-year table, effective breadth,
sector attribution, in-sample / out-of-sample split, and the roll-contamination
diagnostic that motivated the ETF-proxy route (Mandate §5.4).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from tradebot.alpha.cm_tsmom import COST, UNIT, effective_breadth, run
from tradebot.data.xasset_proxy import XASSET_UNIVERSE, sector_of, to_tr_panel

PANEL = "market_data_parquet/xasset/tr_panel.parquet"
OOS_START = "2019-01-01"  # frozen split: ~2/3 in-sample, ~1/3 out-of-sample


def _fmt(d: dict, keys: tuple[str, ...]) -> str:
    out = []
    for k in keys:
        v = d[k]
        out.append(f"{k}={v:+.3f}" if isinstance(v, float) else f"{k}={v}")
    return "  ".join(out)


def main() -> int:
    long = pd.read_parquet(PANEL)
    panel = to_tr_panel(long)
    print(f"panel: {panel.shape[1]} instruments  "
          f"{panel.index.min().date()} -> {panel.index.max().date()}  "
          f"({len(panel):,} bars)\n")

    res = run(panel, cost=COST)
    full = res.summary()
    print("=" * 78)
    print(f"UNIT {UNIT}   (cost: {COST.commission_bps}bp comm + "
          f"{COST.half_spread_bps}bp half-spread/side, "
          f"{COST.borrow_fee_ann:.2%} borrow on shorts)")
    print("=" * 78)
    print(_fmt(full, ("net_sharpe", "gross_sharpe", "net_cagr", "ann_vol")))
    print(_fmt(full, ("max_drawdown", "calmar", "dd_over_vol", "years_positive_frac")))
    print(_fmt(full, ("avg_daily_turnover", "borrow_drag_ann")), f"  n_days={full['n_days']}")

    print("\nper-year net return")
    for y, v in full["per_year_net"].items():
        bar = "#" * min(int(abs(v) * 100), 40)
        print(f"  {y}  {v * 100:+7.2f}%  {'' if v >= 0 else '-'}{bar}")

    # ---- breadth: independent bets, not instrument count (F10) -------------
    rets = panel.pct_change(fill_method=None)
    n_eff = effective_breadth(rets)
    print(f"\neffective breadth: N={panel.shape[1]}  N_eff={n_eff:.2f}  "
          f"(implied IR at IC=0.05, 12 rebal/yr: "
          f"{0.05 * np.sqrt(n_eff * 12):.2f})")
    for sec in sorted({sector_of(c) for c in panel.columns}):
        cols = [c for c in panel.columns if sector_of(c) == sec]
        print(f"  {sec:12s} N={len(cols):2d}  N_eff={effective_breadth(rets[cols]):.2f}")

    # ---- sector attribution ------------------------------------------------
    held = res.weights.shift(1).fillna(0.0)
    print("\nsector attribution (annualised gross contribution)")
    for sec in sorted({sector_of(c) for c in panel.columns}):
        cols = [c for c in panel.columns if sector_of(c) == sec]
        contrib = (held[cols] * rets[cols]).sum(axis=1)
        print(f"  {sec:12s} {contrib.mean() * 252 * 100:+6.2f}%/yr  "
              f"(sharpe {contrib.mean() / contrib.std() * np.sqrt(252):+.2f})")

    # ---- frozen IS / OOS split --------------------------------------------
    print(f"\nIS / OOS split at {OOS_START}")
    net = res.net_returns.dropna()
    is_r, oos_r = net.loc[:OOS_START], net.loc[OOS_START:]
    for nm, r in (("in-sample ", is_r), ("out-sample", oos_r)):
        s = r.mean() / r.std() * np.sqrt(252)
        print(f"  {nm}  n={len(r):5d}  sharpe={s:+.3f}  "
              f"cagr={((1 + r).prod() ** (252 / len(r)) - 1) * 100:+.2f}%")
    is_s = is_r.mean() / is_r.std() * np.sqrt(252)
    oos_s = oos_r.mean() / oos_r.std() * np.sqrt(252)
    print(f"  decay = {(is_s - oos_s) / abs(is_s) * 100:+.1f}%")

    # ---- the data trap this wave exists to avoid --------------------------
    print("\nroll-contamination diagnostic (why proxies, not continuations)")
    try:
        import yfinance as yf
        fut = yf.download(["CL=F", "NG=F"], start="2010-01-01", progress=False,
                          auto_adjust=True, threads=False)["Close"].dropna()
        etf = yf.download(["USO", "UNG"], start="2010-01-01", progress=False,
                          auto_adjust=True, threads=False)["Close"].dropna()
        both = pd.concat([fut, etf], axis=1).dropna()
        lr = np.log(both).diff().dropna()
        yrs = len(lr) / 252
        for f, e in (("CL=F", "USO"), ("NG=F", "UNG")):
            gap = (lr[f].sum() - lr[e].sum()) / yrs * 100
            print(f"  {f} vs {e}: continuation overstates by {gap:+.2f}%/yr")
    except Exception as exc:  # network optional
        print(f"  (skipped: {exc})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
