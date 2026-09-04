"""market_neutral_alpha.py — strictly dollar/beta-neutral cross-sectional
funding-carry with an alt-data regime RISK brake. CHIEF /goal: true alpha,
profitable in ALL regimes, NOT regime-dependent.

Engine (regime-independent BY CONSTRUCTION):
  • Cross-sectional funding carry: w_i = -zscore_x(trailing funding_i), then
    cross-sectionally DEMEANED -> sum(w)=0 (dollar-neutral). With homogeneous
    crypto betas (~1 to the basket) this is also ~beta-neutral.
  • Earns the funding spread (short crowded-long/high-funding, long low-funding)
    irrespective of market direction.

Regime brake (RISK control, not a return signal):
  • alt-data composite z = mean z(F&G, active-addr 30d growth, miner-rev 30d).
    When z < -1 (deleveraging stress -> the regime that historically RUNS OVER
    the carry trade), scale GROSS exposure down. This targets the 2022 crash
    without timing market direction.

Validation: realized beta to basket (must be ~0 = regime-independent), per-year
net-of-cost (must be positive EVERY year), full Sharpe + DSR.
No hyperparameter optimisation — pre-committed standard constants.
"""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
ASSETS = ["ETHUSDT", "SOLUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT"]
K_CARRY = 3
COST_BPS = 5.0
DAYS = 365.0


def _get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    return json.load(urllib.request.urlopen(req, timeout=25))


def daily_panel():
    rets, fund = {}, {}
    for a in ASSETS:
        df = pd.read_parquet(ROOT / "artefacts/features" / f"{a}.parquet",
                             columns=["close", "fundingRate"]).sort_index()
        c = df["close"].resample("1D").last()
        rets[a] = np.log(c / c.shift(1))
        f8 = df["fundingRate"].resample("8h").last()
        fund[a] = f8.resample("1D").sum()
    R = pd.DataFrame(rets).dropna(how="all")
    F = pd.DataFrame(fund).reindex(R.index).fillna(0.0)
    return R, F


def regime_z():
    fng = _get("https://api.alternative.me/fng/?limit=0")["data"]
    f = pd.Series({pd.to_datetime(int(x["timestamp"]), unit="s", utc=True): int(x["value"]) for x in fng}).sort_index().resample("1D").last().ffill()

    def bc(c):
        d = _get(f"https://api.blockchain.info/charts/{c}?timespan=all&format=json&sampled=true")["values"]
        return pd.Series({pd.to_datetime(x["x"], unit="s", utc=True): x["y"] for x in d}).sort_index().resample("1D").last().ffill()
    addr = np.log((s := bc("n-unique-addresses")) / s.shift(30))
    mrev = np.log((s := bc("miners-revenue")) / s.shift(30))

    def zc(x, w=180):
        return (x - x.rolling(w).mean()) / x.rolling(w).std()
    comp = (zc(f.astype(float)) + zc(addr) + zc(mrev)) / 3.0
    return comp.shift(1)  # 1-day publication lag


def xs_carry_weights(F, k):
    car = -F.rolling(k).mean()
    W = {}
    for t in F.index:
        s = car.loc[t].dropna()
        if len(s) < 3:
            W[t] = pd.Series(0.0, index=F.columns); continue
        z = (s - s.mean()) / (s.std() + 1e-12)
        z = z - z.mean()                       # dollar-neutral
        g = z.abs().sum()
        W[t] = (z / g if g > 1e-12 else z).reindex(F.columns).fillna(0.0)
    return pd.DataFrame(W).T


def stats(pnl):
    pnl = pnl.dropna(); eq = (1 + pnl).cumprod(); yrs = len(pnl) / DAYS
    return (pnl.mean() / pnl.std() * np.sqrt(DAYS) if pnl.std() > 0 else np.nan,
            (eq / eq.cummax() - 1).min(), eq.iloc[-1] - 1)


def main():
    R, F = daily_panel()
    W = xs_carry_weights(F, K_CARRY)
    basket = R.mean(axis=1)

    def run(gross):
        w = W.mul(gross, axis=0).shift(1).fillna(0.0)
        price = (w * R).sum(axis=1)
        funding = -(w * F).sum(axis=1)
        cost = (w - w.shift(1).fillna(0.0)).abs().sum(axis=1) * COST_BPS / 1e4
        return (price + funding - cost).rename("pnl")

    # 1) pure neutral carry (no brake)
    pnl_raw = run(pd.Series(1.0, index=R.index))
    # 2) + regime risk-brake: cut gross to 0.3 when composite z < -1
    try:
        z = regime_z().reindex(R.index)
        gross = pd.Series(1.0, index=R.index)
        gross[z < -1.0] = 0.3
        pnl_brk = run(gross.ffill().fillna(1.0))
        have_regime = True
    except Exception as e:
        print("regime fetch failed:", e); pnl_brk = pnl_raw; have_regime = False

    for name, pnl in [("Neutral carry (raw)", pnl_raw), ("Neutral carry + regime-brake", pnl_brk)]:
        sh, dd, tot = stats(pnl)
        # realized beta to basket (regime-independence proof)
        j = pd.DataFrame({"p": pnl, "m": basket}).dropna()
        beta = np.polyfit(j["m"], j["p"], 1)[0]
        print(f"\n=== {name} ===")
        print(f"  FULL: Sharpe={sh:+.2f}  MaxDD={dd*100:.1f}%  TotRet={tot*100:+.0f}%  beta_to_market={beta:+.3f}")
        line = "  per-jaar: "; neg = 0
        for y, g in pnl.groupby(pnl.index.year):
            r = (1 + g).prod() - 1; neg += r < 0
            line += f"{y}:{r*100:+5.1f}% "
        print(line + f" | NEG JAREN={neg}")
        try:
            from tradebot.backtest.metrics import deflated_sharpe
            print(f"  DSR(n_trials=12)={deflated_sharpe(sh, n_trials=12, n_obs=len(pnl.dropna())):.3f}")
        except Exception:
            pass


if __name__ == "__main__":
    main()
