"""xs_harvest.py — harvest the positive XS IC into Sharpe (Wave 15, no retrain).

The XS rework produced a robust positive rank-IC (+0.069, positive EVERY year) but
the naive decile book only made Sharpe 0.25 — the score loads on volatility, so the
long/short is partly an uncompensated vol-factor bet, and turnover/tail names eat
the edge. This loads the persisted OOS scores and builds properly engineered books:
  - rank signal, cross-sectionally demeaned (dollar-neutral)
  - NEUTRALISED vs the vol factor + market beta (regress out -> pure relative value)
  - EWMA-smoothed signal + no-trade band (turnover control)
  - whole-book vol-target, honest 10bps
and reports Sharpe / CAGR / MaxDD / beta / DSR / per-year for each.

Run:  python research/xs_harvest.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from tradebot.backtest.metrics import deflated_sharpe

DAYS = 365.0
COST_BPS = 10.0
TARGET_VOL = 0.40


def load():
    sc = pd.read_parquet(ROOT / "artefacts" / "xs_oos_scores.parquet")
    close = pd.read_parquet(ROOT / "artefacts" / "xs_close_panel.parquet")
    score = sc.pivot_table(index="__date", columns="__sym", values="__p", aggfunc="last")
    score = score.reindex(close.index).reindex(columns=close.columns)
    return score, close


def metrics(pnl, mkt):
    pnl = pnl.dropna()
    sh = pnl.mean() / pnl.std() * np.sqrt(DAYS); vol = pnl.std() * np.sqrt(DAYS)
    eq = (1 + pnl).cumprod(); cagr = eq.iloc[-1] ** (DAYS / len(pnl)) - 1
    dd = (eq / eq.cummax() - 1).min(); srd = pnl.mean() / pnl.std()
    b = pnl.reindex(mkt.index).cov(mkt) / mkt.var()
    yr = " ".join(f"{y}:{((1+g).prod()-1)*100:+.0f}%" for y, g in pnl.groupby(pnl.index.year))
    return sh, cagr, vol, dd, b, deflated_sharpe(srd, 2000, len(pnl)), yr


def book(weights, rets, target_vol=TARGET_VOL):
    g = weights.abs().sum(axis=1).replace(0, np.nan)
    w = weights.div(g, axis=0).fillna(0.0)
    pnl = (w.shift(1) * rets).sum(axis=1) - (w - w.shift(1)).abs().sum(axis=1) * COST_BPS / 1e4
    pnl = pnl.dropna()
    bv = pnl.std() * np.sqrt(DAYS)
    return pnl * (target_vol / bv) if bv > 0 else pnl


def main():
    score, close = load()
    rets = np.log(close / close.shift(1))
    mkt = rets.mean(axis=1)
    vol = rets.rolling(30).std().shift(1)
    logvol = np.log(vol.clip(lower=1e-4))
    btc_ret = rets["BTCUSDT"] if "BTCUSDT" in rets else mkt
    beta = rets.rolling(60).cov(btc_ret).div(btc_ret.rolling(60).var(), axis=0).shift(1)

    # cross-sectional rank signal, demeaned -> dollar-neutral in [-.5,.5]
    rank = score.rank(axis=1, pct=True)
    sig = rank.sub(rank.mean(axis=1), axis=0)

    print("Harvesting positive XS IC into Sharpe (10bps, 40% vol):\n")

    # 1. plain rank book
    print("[1 rank, dollar-neutral]            ", *("%s=%s" % x for x in zip(
        ["Sh", "CAGR", "vol", "MDD", "beta", "DSR"],
        ["%.2f" % v if not isinstance(v, str) else v for v in metrics(book(sig, rets), mkt)[:6]])))
    print("    per-year:", metrics(book(sig, rets), mkt)[6])

    # 2. vol+beta neutralised (regress signal on [logvol,beta] each day, take residual)
    def neutralise(s):
        out = s.copy() * np.nan
        for dt in s.index:
            y = s.loc[dt].dropna()
            if len(y) < 15:
                continue
            X = pd.DataFrame({"lv": logvol.loc[dt], "b": beta.loc[dt]}).reindex(y.index)
            X = X.fillna(X.mean()); X.insert(0, "c", 1.0)
            A = X.values; yy = y.values
            try:
                coef, *_ = np.linalg.lstsq(A, yy, rcond=None)
                out.loc[dt, y.index] = yy - A @ coef
            except Exception:
                out.loc[dt, y.index] = yy
        return out
    sig_n = neutralise(sig)
    m = metrics(book(sig_n, rets), mkt)
    print(f"\n[2 vol+beta-neutralised]             Sh={m[0]:.2f} CAGR={m[1]*100:+.0f}% vol={m[2]:.0%} "
          f"MDD={m[3]*100:.0f}% beta={m[4]:.2f} DSR={m[5]:.3f}")
    print("    per-year:", m[6])

    # 3. neutralised + EWMA smoothed (turnover control)
    sig_s = sig_n.ewm(span=5).mean()
    m = metrics(book(sig_s, rets), mkt)
    print(f"\n[3 neutralised + EWMA5 smooth]       Sh={m[0]:.2f} CAGR={m[1]*100:+.0f}% vol={m[2]:.0%} "
          f"MDD={m[3]*100:.0f}% beta={m[4]:.2f} DSR={m[5]:.3f}")
    print("    per-year:", m[6])

    # 4. neutralised + smoothed + no-trade band (hold unless |Δw| large)
    w = sig_s.copy()
    g = w.abs().sum(axis=1).replace(0, np.nan); wn = w.div(g, axis=0).fillna(0.0)
    held = wn.copy() * 0.0
    prev = pd.Series(0.0, index=wn.columns)
    for dt in wn.index:
        tgt = wn.loc[dt]
        move = (tgt - prev).abs() > 0.004      # 40bps band
        prev = prev.where(~move, tgt)
        held.loc[dt] = prev
    pnl = (held.shift(1) * rets).sum(axis=1) - (held - held.shift(1)).abs().sum(axis=1) * COST_BPS / 1e4
    pnl = pnl.dropna(); bv = pnl.std() * np.sqrt(DAYS); pnl = pnl * (TARGET_VOL / bv)
    m = metrics(pnl, mkt)
    print(f"\n[4 neutralised + smooth + band]      Sh={m[0]:.2f} CAGR={m[1]*100:+.0f}% vol={m[2]:.0%} "
          f"MDD={m[3]*100:.0f}% beta={m[4]:.2f} DSR={m[5]:.3f}")
    print("    per-year:", m[6])


if __name__ == "__main__":
    main()
