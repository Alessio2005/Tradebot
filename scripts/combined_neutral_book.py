"""combined_neutral_book.py — CHIEF /goal: regime-independent alpha via
DIVERSIFICATION of two market-neutral sleeves with complementary regime risk.

No single price-taker strategy is regime-independent (each earns a premium for
bearing some regime risk). But two market-neutral sleeves whose adverse regimes
DON'T overlap combine into an all-weather book:

  Sleeve A — cross-sectional funding CARRY (5 local assets w/ funding history).
             Earns funding premium; loses in deleveraging (2022).
  Sleeve B — broad cross-sectional residual REVERSAL stat-arb (75 perps, price).
             Earns dispersion mean-reversion; loses in calm/trending (2024/26),
             BOOMS in high-dispersion stress (2022).

Both beta~0. Combined by inverse-vol (risk-parity, no tuning). The test: is the
COMBINATION positive across all regimes, beta~0, cost-surviving?
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
ASSETS = ["ETHUSDT", "SOLUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT"]
COST = 5.0


def carry_pnl(k=3, cost_bps=COST):
    rets, fund = {}, {}
    for a in ASSETS:
        df = pd.read_parquet(ROOT / "artefacts/features" / f"{a}.parquet",
                             columns=["close", "fundingRate"]).sort_index()
        c = df["close"].resample("1D").last()
        rets[a] = np.log(c / c.shift(1))
        fund[a] = df["fundingRate"].resample("8h").last().resample("1D").sum()
    R = pd.DataFrame(rets).dropna(how="all"); F = pd.DataFrame(fund).reindex(R.index).fillna(0.0)
    car = -F.rolling(k).mean()
    W = {}
    for t in F.index:
        s = car.loc[t].dropna()
        z = (s - s.mean()) / (s.std() + 1e-12); z = z - z.mean()
        g = z.abs().sum(); W[t] = (z / g if g > 1e-12 else z).reindex(F.columns).fillna(0.0)
    W = pd.DataFrame(W).T
    we = W.shift(1).fillna(0.0)
    pnl = (we * R).sum(axis=1) - (we * F).sum(axis=1) * 0 + (-(we * F).sum(axis=1))
    cost = (we - we.shift(1).fillna(0.0)).abs().sum(axis=1) * cost_bps / 1e4
    return ((we * R).sum(axis=1) - (we * F).sum(axis=1) - cost).rename("carry")


def statarb_pnl(k=3, cost_bps=COST):
    P = pd.read_parquet(ROOT / "artefacts/broad_perp_daily_close.parquet").resample("1D").last()
    R = np.log(P / P.shift(1)); R = R[R.notna().sum(axis=1) >= 20]
    m = R.mean(axis=1); resid = R.sub(m, axis=0)
    s = resid.rolling(k).sum()
    z = s.sub(s.mean(axis=1), axis=0).div(s.std(axis=1).replace(0, np.nan), axis=0)
    w = (-z).sub((-z).mean(axis=1), axis=0); w = w.div(w.abs().sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    we = w.shift(1).fillna(0.0)
    cost = (we - we.shift(1).fillna(0.0)).abs().sum(axis=1) * cost_bps / 1e4
    return ((we * R).sum(axis=1) - cost).rename("statarb"), m


def report(name, net, mkt):
    sh = net.mean() / net.std() * np.sqrt(365) if net.std() > 0 else np.nan
    dd = ((1 + net).cumprod() / (1 + net).cumprod().cummax() - 1).min()
    beta = np.polyfit(mkt.reindex(net.index).fillna(0), net, 1)[0]
    line = f"{name:18s}"; neg = 0
    for y, g in net.groupby(net.index.year):
        r = (1 + g).prod() - 1; neg += r < 0; line += f"{y}:{r*100:+5.0f}% "
    print(line + f"| Sh={sh:+.2f} DD={dd*100:.0f}% beta={beta:+.3f} NEG={neg}")
    return sh


def main():
    carry = carry_pnl()
    statarb, mkt = statarb_pnl()
    idx = carry.index.intersection(statarb.index)
    carry, statarb = carry.reindex(idx).fillna(0.0), statarb.reindex(idx).fillna(0.0)

    # inverse-vol (risk-parity) combine — no tuning
    vc, vs = carry.std(), statarb.std()
    wc, ws = (1 / vc) / (1 / vc + 1 / vs), (1 / vs) / (1 / vc + 1 / vs)
    combo = (wc * carry + ws * statarb).rename("combo")
    corr = carry.corr(statarb)

    print(f"Sleeve correlation (carry vs statarb): {corr:+.3f}   risk-parity weights: carry={wc:.2f} statarb={ws:.2f}\n")
    print("=== Per-jaar (5 bps, market-neutral) ===")
    report("Carry only", carry, mkt)
    report("Statarb only", statarb, mkt)
    report("COMBINED (RP)", combo, mkt)
    try:
        from tradebot.backtest.metrics import deflated_sharpe
        sh = combo.mean() / combo.std() * np.sqrt(365)
        print(f"\nCombined DSR(n_trials=12, n={len(combo)}) = {deflated_sharpe(sh, n_trials=12, n_obs=len(combo)):.3f}")
    except Exception as e:
        print("DSR skip:", e)


if __name__ == "__main__":
    main()
