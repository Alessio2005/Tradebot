"""xs_diagnose.py — is the XS-ML book novel alpha or just the low-vol factor? (Wave 15)

Rebuilds the best engineered book ([4] neutralised+smooth+band) and stress-tests:
  - Sharpe ex-2022 (is it carried by one crash year?)
  - long-only vs short-only contribution
  - correlation to a pure BAB/low-vol book and to the existing carry sleeve
  - correlation to the market
to decide whether the ML rework adds an orthogonal track (combination -> sqrt(N)
Sharpe lift) or merely rediscovers the known low-vol premium.

Run:  python scripts/xs_diagnose.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from tradebot.backtest.metrics import deflated_sharpe

DAYS = 365.0; COST_BPS = 10.0; TARGET_VOL = 0.40


def ann_sharpe(p):
    p = p.dropna(); return p.mean() / p.std() * np.sqrt(DAYS) if p.std() > 0 else 0.0


def vt(pnl):
    pnl = pnl.dropna(); bv = pnl.std() * np.sqrt(DAYS)
    return pnl * (TARGET_VOL / bv) if bv > 0 else pnl


def main():
    sc = pd.read_parquet(ROOT / "artefacts" / "xs_oos_scores.parquet")
    close = pd.read_parquet(ROOT / "artefacts" / "xs_close_panel.parquet")
    score = sc.pivot_table(index="__date", columns="__sym", values="__p", aggfunc="last")
    score = score.reindex(close.index).reindex(columns=close.columns)
    rets = np.log(close / close.shift(1)); mkt = rets.mean(axis=1)
    vol = rets.rolling(30).std().shift(1); logvol = np.log(vol.clip(lower=1e-4))
    btc = rets["BTCUSDT"]; beta = rets.rolling(60).cov(btc).div(btc.rolling(60).var(), axis=0).shift(1)

    rank = score.rank(axis=1, pct=True); sig = rank.sub(rank.mean(axis=1), axis=0)

    # neutralise vs [logvol, beta]
    sig_n = sig * np.nan
    for dt in sig.index:
        y = sig.loc[dt].dropna()
        if len(y) < 15:
            continue
        X = pd.DataFrame({"lv": logvol.loc[dt], "b": beta.loc[dt]}).reindex(y.index)
        X = X.fillna(X.mean()).fillna(0.0); X.insert(0, "c", 1.0)
        A = X.values
        if not np.isfinite(A).all():
            sig_n.loc[dt, y.index] = y.values; continue
        try:
            coef, *_ = np.linalg.lstsq(A, y.values, rcond=None)
            sig_n.loc[dt, y.index] = y.values - A @ coef
        except np.linalg.LinAlgError:
            sig_n.loc[dt, y.index] = y.values
    sig_s = sig_n.ewm(span=5).mean()

    def to_pnl(w, long_only=False, short_only=False):
        if long_only:
            w = w.clip(lower=0)
        if short_only:
            w = w.clip(upper=0)
        g = w.abs().sum(axis=1).replace(0, np.nan); wn = w.div(g, axis=0).fillna(0.0)
        return (wn.shift(1) * rets).sum(axis=1) - (wn - wn.shift(1)).abs().sum(axis=1) * COST_BPS / 1e4

    ml = vt(to_pnl(sig_s))
    print("=== XS-ML book [neutralised+smooth] ===")
    print(f"  full Sharpe={ann_sharpe(ml):.2f}")
    ex22 = ml[ml.index.year != 2022]
    print(f"  Sharpe EX-2022={ann_sharpe(ex22):.2f}   (2022 contributes "
          f"{(ml[ml.index.year==2022]+1).prod()-1:+.0%})")
    lo = vt(to_pnl(sig_s, long_only=True)); so = vt(to_pnl(sig_s, short_only=True))
    print(f"  long-only Sharpe={ann_sharpe(lo):.2f}   short-only Sharpe={ann_sharpe(so):.2f}")

    # pure BAB / low-vol book: long low-vol, short high-vol (no ML)
    lv_rank = logvol.rank(axis=1, pct=True)
    bab_sig = -(lv_rank.sub(lv_rank.mean(axis=1), axis=0))   # +ve weight on low vol
    bab = vt(to_pnl(bab_sig.ewm(span=5).mean()))
    print(f"\n=== pure BAB/low-vol book (no ML) Sharpe={ann_sharpe(bab):.2f} ===")
    common = ml.dropna().index.intersection(bab.dropna().index)
    print(f"  corr(XS-ML, BAB)          = {ml.reindex(common).corr(bab.reindex(common)):+.2f}")
    print(f"  corr(XS-ML, market)       = {ml.reindex(common).corr(mkt.reindex(common)):+.2f}")

    # existing carry sleeve (if available)
    try:
        fund = pd.read_parquet(ROOT / "artefacts" / "funding_universe.parquet").reindex(close.index).reindex(columns=close.columns)
        fz = ((fund - fund.rolling(30).mean()) / fund.rolling(30).std()).shift(1)
        carry_sig = -(fz.sub(fz.mean(axis=1), axis=0))      # long -funding (receive carry)
        carry = vt(to_pnl(carry_sig.ewm(span=5).mean()))
        print(f"  corr(XS-ML, carry)        = {ml.reindex(common).corr(carry.reindex(common)):+.2f}  "
              f"(carry Sharpe={ann_sharpe(carry):.2f})")
    except Exception as e:
        print("  carry corr n/a:", str(e)[:40]); carry = None

    # combined equal-risk book
    parts = {"ml": ml, "bab": bab}
    if carry is not None:
        parts["carry"] = carry
    comb = sum(vt(p).reindex(ml.index).fillna(0.0) for p in parts.values())
    comb = vt(comb)
    eq = (1 + comb.dropna()).cumprod(); cagr = eq.iloc[-1] ** (DAYS / len(comb.dropna())) - 1
    dd = (eq / eq.cummax() - 1).min(); srd = comb.dropna().mean() / comb.dropna().std()
    yr = " ".join(f"{y}:{((1+g).prod()-1)*100:+.0f}%" for y, g in comb.dropna().groupby(comb.dropna().index.year))
    print(f"\n=== COMBINED equal-risk ({'+'.join(parts)}) ===")
    print(f"  Sharpe={ann_sharpe(comb):.2f} CAGR={cagr*100:+.0f}% MaxDD={dd*100:.0f}% "
          f"DSR(N2000)={deflated_sharpe(srd,2000,len(comb.dropna())):.3f}")
    print(f"  per-year: {yr}")


if __name__ == "__main__":
    main()
