"""broad_statarb.py — breadth-fixed market-neutral cross-sectional stat-arb.

CHIEF /goal: regime-independent alpha. The 5-asset residual reversal had no
edge because stat-arb needs cross-sectional BREADTH (idiosyncratic dispersion).
This fetches ~75 established Binance USDT perps (onboard < 2021-07, full multi-
regime history) and runs Avellaneda-Lee style residual mean-reversion:

  • market factor m_t = cross-sectional mean return (equal-weight).
  • residual_i = R_i - m_t  (beta~1 homogeneous crypto -> market-neutral).
  • score_i = -zscore_x( trailing k-day cumulative residual )  (long laggards).
  • dollar-neutral weights (demeaned), gross=1, next-day fill, costs on turnover.

Survivorship caveat: universe = perps STILL trading today (delisted excluded
-> mild upward bias). Noted, not corrected (Binance rarely delists majors).
No hyperparameter optimisation — report a small fixed grid, do not select.
"""
from __future__ import annotations
import json, time, urllib.request, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
CACHE = ROOT / "artefacts" / "broad_perp_daily_close.parquet"
START_MS = int(pd.Timestamp("2021-06-01", tz="UTC").timestamp() * 1000)


def _get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    return json.load(urllib.request.urlopen(req, timeout=30))


def universe():
    info = _get("https://fapi.binance.com/fapi/v1/exchangeInfo")
    out = []
    for s in info["symbols"]:
        if s.get("contractType") != "PERPETUAL" or s.get("quoteAsset") != "USDT" or s.get("status") != "TRADING":
            continue
        if s.get("onboardDate", 0) / 1000 < pd.Timestamp("2021-07-01").timestamp() and s["symbol"] != "BTCDOMUSDT":
            out.append(s["symbol"])
    return out


def fetch_daily_close(sym):
    rows, start = [], START_MS
    for _ in range(10):
        d = _get(f"https://fapi.binance.com/fapi/v1/klines?symbol={sym}&interval=1d&startTime={start}&limit=1000")
        if not d:
            break
        rows += [(k[0], float(k[4])) for k in d]
        if len(d) < 1000:
            break
        start = d[-1][0] + 86400000
    if not rows:
        return None
    s = pd.Series({pd.to_datetime(t, unit="ms", utc=True): c for t, c in rows}).sort_index()
    return s[~s.index.duplicated()]


def build_panel():
    if CACHE.exists():
        return pd.read_parquet(CACHE)
    syms = universe()
    print(f"Fetching {len(syms)} perps...", flush=True)
    cols = {}
    for i, sym in enumerate(syms):
        try:
            s = fetch_daily_close(sym)
            if s is not None and len(s) > 800:
                cols[sym] = s
        except Exception as e:
            print(f"  {sym} fail: {str(e)[:50]}")
        if i % 15 == 0:
            print(f"  {i}/{len(syms)}", flush=True)
        time.sleep(0.05)
    P = pd.DataFrame(cols).sort_index()
    P.to_parquet(CACHE)
    return P


def main():
    P = build_panel()
    P = P.resample("1D").last()
    R = np.log(P / P.shift(1))
    # require >=20 names present each day
    R = R[R.notna().sum(axis=1) >= 20]
    m = R.mean(axis=1)                      # market factor
    resid = R.sub(m, axis=0)
    print(f"Panel: {R.index.min().date()}..{R.index.max().date()} | {R.shape[1]} perps | {len(R)} days")

    def run(k, cost_bps):
        s = resid.rolling(k).sum()
        z = s.sub(s.mean(axis=1), axis=0).div(s.std(axis=1).replace(0, np.nan), axis=0)
        w = (-z).sub((-z).mean(axis=1), axis=0)
        w = w.div(w.abs().sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
        we = w.shift(1).fillna(0.0)
        pnl = (we * R).sum(axis=1)
        cost = (we - we.shift(1).fillna(0.0)).abs().sum(axis=1) * cost_bps / 1e4
        return (pnl - cost).dropna()

    print("\n=== Broad cross-sectional residual reversal ===")
    print(f"{'k':>3s}{'cost':>7s}{'Sharpe':>8s}{'MaxDD%':>9s}{'AnnRet%':>9s}{'beta':>8s}")
    for k in (1, 2, 3, 5):
        for cb in (0.0, 5.0, 10.0):
            net = run(k, cb)
            sh = net.mean() / net.std() * np.sqrt(365) if net.std() > 0 else np.nan
            dd = ((1 + net).cumprod() / (1 + net).cumprod().cummax() - 1).min()
            ann = (1 + net).prod() ** (365 / len(net)) - 1
            beta = np.polyfit(m.reindex(net.index).fillna(0), net, 1)[0]
            print(f"{k:>3d}{cb:>7.1f}{sh:>8.2f}{dd*100:>9.1f}{ann*100:>9.1f}{beta:>8.3f}")

    # per-year for the most cost-robust k at 10 bps
    print("\n=== per-jaar (k=1, 10bps) ===")
    net = run(1, 10.0); line = ""; neg = 0
    for y, g in net.groupby(net.index.year):
        r = (1 + g).prod() - 1; neg += r < 0; line += f"{y}:{r*100:+5.1f}% "
    print(line + f"| NEG JAREN={neg}")
    try:
        from tradebot.backtest.metrics import deflated_sharpe
        s5 = run(1, 10.0); shh = s5.mean()/s5.std()*np.sqrt(365)
        print(f"DSR(n_trials=12)={deflated_sharpe(shh, n_trials=12, n_obs=len(s5)):.3f}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
