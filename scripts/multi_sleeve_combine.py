"""multi_sleeve_combine.py — stack orthogonal sleeves toward higher Sharpe (Wave 15).

The XS-ML sleeve (Sharpe ~0.8) is orthogonal to low-vol (corr +0.03). Sharpe stacks
as sqrt(N) over uncorrelated sleeves, so the real lever toward the target is a book
of decorrelated sleeves. Builds 4 (directional now allowed under mandate v2):
  1. ML-XS   : persisted cross-sectional model, neutralised+smoothed (market-neutral)
  2. LOWVOL  : long low realised-vol / short high (BAB)             (market-neutral)
  3. STATARB : residual (BTC-hedged) short-term reversal           (market-neutral)
  4. TSMOM   : time-series trend across the universe (net directional, allowed v2)
Reports each sleeve's Sharpe, the correlation matrix (deliverable), the risk-parity
combined book, and the honest sqrt(N) ceiling projection.

Run:  python scripts/multi_sleeve_combine.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from tradebot.backtest.metrics import deflated_sharpe

DAYS = 365.0; COST_BPS = 10.0; TARGET_VOL = 0.40


def vt(p, tv=TARGET_VOL):
    p = p.dropna(); bv = p.std() * np.sqrt(DAYS)
    return p * (tv / bv) if bv > 0 else p


def ann(p):
    p = p.dropna(); return p.mean() / p.std() * np.sqrt(DAYS) if p.std() > 0 else 0.0


def pnl_from_w(w, rets):
    g = w.abs().sum(axis=1).replace(0, np.nan); wn = w.div(g, axis=0).fillna(0.0)
    return (wn.shift(1) * rets).sum(axis=1) - (wn - wn.shift(1)).abs().sum(axis=1) * COST_BPS / 1e4


def main():
    sc = pd.read_parquet(ROOT / "artefacts" / "xs_oos_scores.parquet")
    close = pd.read_parquet(ROOT / "artefacts" / "xs_close_panel.parquet")
    score = sc.pivot_table(index="__date", columns="__sym", values="__p", aggfunc="last")
    score = score.reindex(close.index).reindex(columns=close.columns)
    rets = np.log(close / close.shift(1)); mkt = rets.mean(axis=1)
    vol = rets.rolling(30).std().shift(1); logvol = np.log(vol.clip(lower=1e-4))
    btc = rets["BTCUSDT"]; beta = rets.rolling(60).cov(btc).div(btc.rolling(60).var(), axis=0).shift(1)
    resid = rets.sub(beta.mul(btc, axis=0))             # BTC-hedged residual returns

    def demean(s):
        return s.sub(s.mean(axis=1), axis=0)

    # 1. ML-XS neutralised + smoothed
    rank = score.rank(axis=1, pct=True); sig = demean(rank)
    sig_n = sig * np.nan
    for dt in sig.index:
        y = sig.loc[dt].dropna()
        if len(y) < 15:
            continue
        X = pd.DataFrame({"lv": logvol.loc[dt], "b": beta.loc[dt]}).reindex(y.index).fillna(0.0)
        X.insert(0, "c", 1.0); A = X.values
        if not np.isfinite(A).all():
            sig_n.loc[dt, y.index] = y.values; continue
        try:
            coef, *_ = np.linalg.lstsq(A, y.values, rcond=None)
            sig_n.loc[dt, y.index] = y.values - A @ coef
        except np.linalg.LinAlgError:
            sig_n.loc[dt, y.index] = y.values
    ml = vt(pnl_from_w(sig_n.ewm(span=5).mean(), rets))

    # 2. LOWVOL / BAB
    lv = -demean(logvol.rank(axis=1, pct=True))
    lowvol = vt(pnl_from_w(lv.ewm(span=5).mean(), rets))

    # 3. STATARB residual short-term reversal
    rr = -demean(resid.rolling(5).sum().rank(axis=1, pct=True))
    statarb = vt(pnl_from_w(rr.ewm(span=3).mean(), rets))

    # 4. TSMOM (per-asset trend, net directional) — long if 60d return > 0, vol-scaled
    trend = np.sign(np.log(close / close.shift(60))).shift(1)
    w_ts = (trend / vol).replace([np.inf, -np.inf], 0.0)
    tsmom = vt(pnl_from_w(w_ts.ewm(span=5).mean(), rets))

    sleeves = {"ML-XS": ml, "LOWVOL": lowvol, "STATARB": statarb, "TSMOM": tsmom}
    idx = ml.dropna().index
    for k in sleeves:
        sleeves[k] = sleeves[k].reindex(idx).fillna(0.0)

    print("=== SLEEVE SHARPES (10bps, 40% vol each) ===")
    for k, p in sleeves.items():
        print(f"  {k:8s} Sharpe={ann(p):.2f}  beta={p.cov(mkt.reindex(idx))/mkt.reindex(idx).var():+.2f}")

    print("\n=== SLEEVE CORRELATION MATRIX ===")
    C = pd.DataFrame(sleeves).corr()
    print(C.round(2).to_string())

    # risk-parity combine (equal vol already via vt) — try MN-only and all-4
    for name, keys in (("MN (ML+LOWVOL+STATARB)", ["ML-XS", "LOWVOL", "STATARB"]),
                       ("ALL (+TSMOM directional)", ["ML-XS", "LOWVOL", "STATARB", "TSMOM"])):
        comb = vt(sum(sleeves[k] for k in keys))
        eq = (1 + comb.dropna()).cumprod(); cagr = eq.iloc[-1] ** (DAYS / len(comb.dropna())) - 1
        dd = (eq / eq.cummax() - 1).min(); srd = comb.dropna().mean() / comb.dropna().std()
        b = comb.dropna().cov(mkt.reindex(idx)) / mkt.reindex(idx).var()
        yr = " ".join(f"{y}:{((1+g).prod()-1)*100:+.0f}%" for y, g in comb.dropna().groupby(comb.dropna().index.year))
        print(f"\n=== COMBINED {name} (risk-parity) ===")
        print(f"  Sharpe={ann(comb):.2f} CAGR={cagr*100:+.0f}% MaxDD={dd*100:.0f}% beta={b:+.2f} "
              f"DSR(N2000)={deflated_sharpe(srd,2000,len(comb.dropna())):.3f}")
        print(f"  per-year: {yr}")

    # honest sqrt(N) ceiling
    avg_sh = np.mean([ann(sleeves[k]) for k in sleeves])
    print("\n=== sqrt(N) CEILING ===")
    print(f"  avg sleeve Sharpe={avg_sh:.2f}; for Sharpe=3 need N_orthogonal={ (3/max(avg_sh,1e-9))**2 :.0f} "
          f"equally-good uncorrelated sleeves (have ~{len(sleeves)}).")


if __name__ == "__main__":
    main()
