"""neutral_carry_reversal.py — dollar-neutral short-term-reversal + funding-carry.

CHIEF build (2026-05-31). Goal: a non-trend, every-year-profitable strategy on
measured edge, without overfit.

Design (low degrees of freedom — no hyperparameter optimisation):
  • Universe: ETH, SOL, AVAX, LINK, DOT (the 5 with built features).
  • Daily grid (resampled from event-bar features).
  • Factor 1 — short-term REVERSAL (Lehmann/Jegadeesh 1990):
        s_rev_i,t = -zscore_x( sum of past k_rev daily log-returns, lagged 1d )
        long recent losers, short recent winners.
  • Factor 2 — funding CARRY harvest:
        s_carry_i,t = -zscore_x( trailing k_carry-day mean 8h funding, lagged 1d )
        short high-funding perps (collect funding), long low/negative-funding.
  • Both DEMEANED cross-sectionally each day -> dollar-neutral (sum w = 0).
  • Combine 50/50 (equal risk, no tuned weight), re-demean, scale gross to 1.0.
  • Rebalance daily, fills next day (no lookahead). Costs on turnover.
  • Funding P&L: long pays positive funding, short receives.

Validation:
  • Per-year OOS return / Sharpe / MaxDD.
  • Cost sensitivity {0, 2.5, 5, 10} bps  -> Gate 1 (does edge survive costs?).
  • Deflated Sharpe (Bailey & Lopez de Prado) with a conservative trial count.
  • Lookback robustness (report, NOT select).

This is a RESEARCH harness (gross-of-financing realism caveats noted). It is the
gate before wiring into the production CPCV pipeline.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

ASSETS = ["ETHUSDT", "SOLUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT"]
FEAT = ROOT / "artefacts" / "features"

K_REV = 5      # reversal lookback (days)   — pre-committed standard
K_CARRY = 3    # funding smoothing (days)   — ~1 funding cycle
W_REV = 0.5    # equal factor weight (no tuning)
DAYS_PER_YR = 365.0


def build_daily_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (daily_logret, daily_funding_long_cost) panels [date x asset].

    daily_funding_long_cost_i,t = funding a LONG pays on day t (sum of the
    realised 8h rates that day). Long P&L from funding = -w * this.
    """
    rets, fund = {}, {}
    for a in ASSETS:
        df = pd.read_parquet(FEAT / f"{a}.parquet", columns=["close", "fundingRate"])
        df = df.sort_index()
        close_d = df["close"].resample("1D").last().dropna()
        rets[a] = np.log(close_d / close_d.shift(1))
        # realised 8h funding -> daily paid by a long
        f8 = df["fundingRate"].resample("8h").last()
        fund[a] = f8.resample("1D").sum()
    R = pd.DataFrame(rets).dropna(how="all")
    F = pd.DataFrame(fund).reindex(R.index).fillna(0.0)
    return R, F


def x_demean(s: pd.Series) -> pd.Series:
    """Cross-sectional demean -> dollar-neutral, then scale gross to 1."""
    s = s - s.mean()
    g = s.abs().sum()
    return s / g if g > 1e-12 else s


def x_zscore(s: pd.Series) -> pd.Series:
    sd = s.std()
    return (s - s.mean()) / sd if sd > 1e-12 else s * 0.0


def build_weights(R: pd.DataFrame, F: pd.DataFrame, k_rev: int, k_carry: int, w_rev: float) -> pd.DataFrame:
    """Daily dollar-neutral weights, set at close of day t (using data <= t)."""
    # Factor signals (lagged: use info through t to trade t+1)
    past_ret = R.rolling(k_rev).sum()                 # trailing cumulative return
    rev_raw = -past_ret                               # reversal: long losers
    carry_raw = -F.rolling(k_carry).mean()            # carry: short high funding
    weights = {}
    for t in R.index:
        rev = x_zscore(rev_raw.loc[t].dropna())
        car = x_zscore(carry_raw.loc[t].dropna())
        common = rev.index.union(car.index)
        combo = w_rev * rev.reindex(common).fillna(0.0) + (1 - w_rev) * car.reindex(common).fillna(0.0)
        weights[t] = x_demean(combo)
    W = pd.DataFrame(weights).T.reindex(columns=R.columns).fillna(0.0)
    return W


def backtest(R: pd.DataFrame, F: pd.DataFrame, W: pd.DataFrame, cost_bps: float) -> pd.Series:
    """Daily net P&L series. Weights at t applied to t+1 returns (next-day fill)."""
    w_eff = W.shift(1).fillna(0.0)                     # SK-2: trade next day
    price_pnl = (w_eff * R).sum(axis=1)
    funding_pnl = -(w_eff * F).sum(axis=1)             # long pays +funding, short receives
    turnover = (w_eff - w_eff.shift(1).fillna(0.0)).abs().sum(axis=1)
    cost = turnover * (cost_bps / 1e4)
    return (price_pnl + funding_pnl - cost).rename("pnl")


def stats(pnl: pd.Series) -> dict:
    eq = (1 + pnl).cumprod()
    yrs = len(pnl) / DAYS_PER_YR
    sharpe = pnl.mean() / pnl.std() * np.sqrt(DAYS_PER_YR) if pnl.std() > 0 else float("nan")
    mdd = (eq / eq.cummax() - 1).min()
    return {"sharpe": sharpe, "ann_ret": eq.iloc[-1] ** (1 / yrs) - 1,
            "total_ret": eq.iloc[-1] - 1, "maxdd": mdd}


def main() -> None:
    R, F = build_daily_panel()
    print(f"Daily panel: {R.index.min().date()} .. {R.index.max().date()} | {len(R)} days | {len(ASSETS)} assets\n")

    # ---- Headline config (pre-committed) ----
    W = build_weights(R, F, K_REV, K_CARRY, W_REV)
    print(f"=== GATE 1: cost sensitivity (k_rev={K_REV}, k_carry={K_CARRY}, 50/50, dollar-neutral) ===")
    print(f"{'cost_bps':>9s}{'Sharpe':>8s}{'AnnRet%':>9s}{'MaxDD%':>8s}{'TotRet%':>9s}")
    for cb in (0.0, 2.5, 5.0, 10.0):
        s = stats(backtest(R, F, W, cb))
        print(f"{cb:>9.1f}{s['sharpe']:>8.2f}{s['ann_ret']*100:>9.1f}{s['maxdd']*100:>8.1f}{s['total_ret']*100:>9.1f}")

    # ---- Per-year (at 5 bps) ----
    pnl5 = backtest(R, F, W, 5.0)
    print("\n=== Per-jaar OOS (5 bps kosten) ===")
    print(f"{'Year':6s}{'Return%':>10s}{'Sharpe':>8s}{'MaxDD%':>8s}")
    gross_net_neg = 0
    for y, g in pnl5.groupby(pnl5.index.year):
        eq = (1 + g).cumprod()
        sh = g.mean() / g.std() * np.sqrt(DAYS_PER_YR) if g.std() > 0 else float("nan")
        mdd = (eq / eq.cummax() - 1).min()
        ret = eq.iloc[-1] - 1
        if ret < 0:
            gross_net_neg += 1
        print(f"{y:<6d}{ret*100:>10.1f}{sh:>8.2f}{mdd*100:>8.1f}")
    print(f"\nNegatieve jaren @5bps: {gross_net_neg}")

    # ---- Deflated Sharpe (conservative trial count) ----
    try:
        from tradebot.backtest.metrics import deflated_sharpe
        # We tested ~ a handful of (k_rev,k_carry,sign,weight) structural choices.
        n_trials = 12
        dsr = deflated_sharpe(stats(pnl5)["sharpe"], n_trials=n_trials, n_obs=len(pnl5))
        print(f"\nDeflated Sharpe (n_trials={n_trials}, n_obs={len(pnl5)}): {dsr:.3f}  (promote if >0.95)")
    except Exception as e:
        print(f"\nDSR calc skipped: {e}")

    # ---- Lookback robustness (REPORT, do not select) ----
    print("\n=== Robuustheid over lookbacks (5 bps) — NIET geselecteerd, alleen gerapporteerd ===")
    print(f"{'k_rev':>6s}{'k_carry':>8s}{'Sharpe':>8s}{'AnnRet%':>9s}{'MaxDD%':>8s}")
    for kr in (3, 5, 10):
        for kc in (1, 3, 5):
            Wi = build_weights(R, F, kr, kc, W_REV)
            s = stats(backtest(R, F, Wi, 5.0))
            print(f"{kr:>6d}{kc:>8d}{s['sharpe']:>8.2f}{s['ann_ret']*100:>9.1f}{s['maxdd']*100:>8.1f}")


if __name__ == "__main__":
    main()
