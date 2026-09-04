"""robust_combine.py — best robust combination + the "every year >60%" math (Wave 17).

(1) RR sweep: trend entry + asymmetric ATR PT/SL barrier exits, several RR ratios.
(2) Combine complementary sleeves (robust trend + XS-MN reversal + low-vol) which
    cover different regimes, and report per-year robustness.
(3) The decisive arithmetic: what Sharpe does "every year > X%" actually require?

Run:  python research/robust_combine.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from tradebot.backtest.metrics import deflated_sharpe

PANEL = ROOT / "artefacts" / "broad_perp_ohlcv.parquet"
DAYS = 365.0; COST_BPS = 10.0; TARGET_VOL = 0.40


def vt(p, tv=TARGET_VOL):
    p = p.dropna(); bv = p.std() * np.sqrt(DAYS)
    return p * (tv / bv) if bv > 0 else p


def peryear(p):
    return {y: (1 + g).prod() - 1 for y, g in p.dropna().groupby(p.dropna().index.year)}


def ann(p):
    p = p.dropna(); return p.mean() / p.std() * np.sqrt(DAYS) if p.std() > 0 else 0.0


def main():
    panel = pd.read_parquet(PANEL)
    close = panel.pivot_table(index="date", columns="symbol", values="close").sort_index()
    close = close.dropna(axis=1, thresh=int(len(close) * 0.6))
    syms = list(close.columns)
    rets = np.log(close / close.shift(1)); mkt = rets.mean(axis=1); btc = rets["BTCUSDT"]
    vol = rets.rolling(30).std().shift(1)

    # ---- (1) RR ratio sweep: trend entry (50d MA) + asymmetric exits via ATR multiple ----
    print("=== (1) RR-RATIO SWEEP (trend entry, asymmetric vol-scaled hold) ===")
    print(f"{'PT:SL':>7s} {'Sharpe':>7s}  per-year")
    trendsig = np.sign(close / close.rolling(50).mean() - 1).shift(1)   # long/short trend
    for pt, sl in [(1, 1), (2, 1), (3, 1), (1, 2), (3, 2), (1, 3)]:
        # asymmetric sizing: scale winners up to pt, cap losers at sl (RR proxy on held pnl)
        w = (trendsig / vol.clip(lower=1e-4))
        g = w.abs().sum(axis=1).replace(0, np.nan); wn = w.div(g, axis=0).fillna(0.0)
        raw = (wn * rets)
        asym = raw.clip(lower=-sl * raw.abs().mean(axis=1).values.reshape(-1, 1) - 1,
                        upper=pt * raw.abs().mean(axis=1).values.reshape(-1, 1) + 1)
        pnl = vt(asym.sum(axis=1) - (wn - wn.shift(1)).abs().sum(axis=1) * COST_BPS / 1e4)
        ys = peryear(pnl)
        print(f"{pt}:{sl:>5d} {ann(pnl):7.2f}  " + " ".join(f"{y}:{v*100:+.0f}%" for y, v in ys.items()))

    # ---- (2) combine complementary sleeves ----
    print("\n=== (2) ROBUST COMBINATION (complementary regimes) ===")
    # robust trend (tf2-ish: 2d bars, fast lookbacks, long/short)
    c2 = close.resample("2D").last(); r2 = np.log(c2 / c2.shift(1)); v2 = r2.rolling(20).std().shift(1).clip(lower=1e-4)
    ts = sum(np.sign(np.log(c2 / c2.shift(max(1, round(L / 2))))) for L in (10, 20, 40)) / 3
    w2 = (ts / v2).shift(1); g2 = w2.abs().sum(axis=1).replace(0, np.nan); wn2 = w2.div(g2, axis=0).fillna(0.0)
    trend = (wn2 * r2).sum(axis=1) - (wn2 - wn2.shift(1)).abs().sum(axis=1) * COST_BPS / 1e4
    trend = trend.reindex(close.index).fillna(0.0)
    trend_d = vt(trend)

    # XS-MN reversal sleeve (from cached ML scores if present, else raw 5d reversal)
    try:
        sc = pd.read_parquet(ROOT / "artefacts" / "xs_oos_scores.parquet")
        score = sc.pivot_table(index="__date", columns="__sym", values="__p", aggfunc="last").reindex(close.index).reindex(columns=syms)
        rk = score.rank(axis=1, pct=True); sig = rk.sub(rk.mean(axis=1), axis=0).ewm(span=5).mean()
    except Exception:
        sig = -(rets.rolling(5).sum().rank(axis=1, pct=True)).sub(0.5)
    gx = sig.abs().sum(axis=1).replace(0, np.nan); wx = sig.div(gx, axis=0).fillna(0.0)
    xs = vt((wx.shift(1) * rets).sum(axis=1) - (wx - wx.shift(1)).abs().sum(axis=1) * COST_BPS / 1e4)

    # low-vol
    lv = -(np.log(vol.clip(lower=1e-4)).rank(axis=1, pct=True)).sub(0.5).ewm(span=5).mean()
    gl = lv.abs().sum(axis=1).replace(0, np.nan); wl = lv.div(gl, axis=0).fillna(0.0)
    lowvol = vt((wl.shift(1) * rets).sum(axis=1) - (wl - wl.shift(1)).abs().sum(axis=1) * COST_BPS / 1e4)

    idx = xs.dropna().index
    sleeves = {"TREND": trend_d.reindex(idx).fillna(0), "XS-REV": xs.reindex(idx).fillna(0), "LOWVOL": lowvol.reindex(idx).fillna(0)}
    for k, p in sleeves.items():
        print(f"  {k:8s} Sharpe={ann(p):.2f}  per-year " + " ".join(f"{y}:{v*100:+.0f}%" for y, v in peryear(p).items()))
    print("  corr:\n" + pd.DataFrame(sleeves).corr().round(2).to_string())
    comb = vt(sum(sleeves.values()))
    ys = peryear(comb)
    print(f"\n  COMBINED Sharpe={ann(comb):.2f}  MaxDD={((1+comb).cumprod()/(1+comb).cumprod().cummax()-1).min()*100:.0f}%  "
          f"DSR(N2000)={deflated_sharpe(comb.mean()/comb.std(),2000,len(comb)):.3f}")
    print("  per-year: " + " ".join(f"{y}:{v*100:+.0f}%" for y, v in ys.items()))
    print(f"  every year >60%? {'YES' if all(v>0.6 for v in ys.values()) else 'NO'}; "
          f"every year >0%? {'YES' if all(v>0 for v in ys.values()) else 'NO'}; worst={min(ys.values())*100:.0f}%")

    # ---- (3) the arithmetic of "every year > X" ----
    print("\n=== (3) WHAT DOES 'every year >60%' REQUIRE? (Gaussian annual returns) ===")
    print("  P(year>60%)=0.95 needs  mu - 1.645*sigma > 60%.  At Sharpe S, mu=S*sigma:")
    for S in (1.0, 1.5, 2.0, 3.0, 4.0):
        # find sigma s.t. S*sigma - 1.645*sigma = 0.60  -> sigma=0.60/(S-1.645)
        if S > 1.645:
            sig_need = 0.60 / (S - 1.645); mu = S * sig_need
            print(f"    Sharpe {S:.1f}: feasible at vol={sig_need*100:.0f}% (mean {mu*100:.0f}%) — 1-in-20 year still >60%")
        else:
            print(f"    Sharpe {S:.1f}: IMPOSSIBLE — cannot keep the worst year above 60% at any vol")
    print("  => 'every year >60%' ~ requires Sharpe >~ 2-3 (same barrier as 100% CAGR).")


if __name__ == "__main__":
    main()
