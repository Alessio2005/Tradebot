"""trend_robust_sweep.py — all-weather long/short trend, multi-timeframe/RR (Wave 17).

User thesis: bigger timeframes show more trend; long/short trend can be all-weather
(long 2021, short 2022) and robust every year. This sweeps:
  - bar timeframe: 1d / 2d / 3d / 5d / 7d (resampled from daily OHLCV)
  - RR / trend lookbacks (fast..slow) and signal blends
  - long/short vs long/flat
on the 99-asset universe, vol-targeted, honest costs.

Methodologically watertight:
  - causal signals (position at bar t uses data <= t-1), turnover costs
  - WALK-FORWARD config selection: pick best on IS (<=2023), report OOS (2024-26)
    so the winning config is NOT chosen on the test window
  - DSR DEFLATED by the number of configs swept (multiple-testing honest)
  - per-year P&L + robustness (#neg years, worst year) reported for the winner

Run:  python scripts/trend_robust_sweep.py
"""
from __future__ import annotations
import sys, itertools, warnings
from pathlib import Path
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from tradebot.backtest.metrics import deflated_sharpe

PANEL = ROOT / "artefacts" / "broad_perp_ohlcv.parquet"
DAYS = 365.0; COST_BPS = 10.0; TARGET_VOL = 0.40
IS_END = "2023-12-31"


def resample_close(close, tf):
    if tf == 1:
        return close
    return close.resample(f"{tf}D").last()


def trend_book(close, lookbacks, longshort, tf):
    """Multi-lookback long/short trend book on a close panel (one bar = tf days)."""
    c = resample_close(close, tf)
    r = np.log(c / c.shift(1))
    vol = r.rolling(20).std().shift(1).clip(lower=1e-4)
    sig = pd.DataFrame(0.0, index=c.index, columns=c.columns)
    for L in lookbacks:
        s = np.sign(np.log(c / c.shift(L)))
        sig = sig.add(s, fill_value=0.0)
    sig = sig / len(lookbacks)                      # blended trend in [-1,1]
    if not longshort:
        sig = sig.clip(lower=0.0)
    w = (sig / vol).shift(1)                          # causal, inverse-vol sized
    g = w.abs().sum(axis=1).replace(0, np.nan); wn = w.div(g, axis=0).fillna(0.0)
    pnl = (wn * r).sum(axis=1) - (wn - wn.shift(1)).abs().sum(axis=1) * COST_BPS / 1e4
    pnl = pnl.dropna()
    # scale daily-equivalent: bar return over tf days -> annualise with DAYS/tf bars
    bars_per_year = DAYS / tf
    bv = pnl.std() * np.sqrt(bars_per_year)
    pnl = pnl * (TARGET_VOL / bv) if bv > 0 else pnl
    return pnl, bars_per_year


def stats(pnl, bpy):
    sh = pnl.mean() / pnl.std() * np.sqrt(bpy) if pnl.std() > 0 else 0.0
    eq = (1 + pnl).cumprod()
    yrs = {y: (1 + g).prod() - 1 for y, g in pnl.groupby(pnl.index.year)}
    dd = (eq / eq.cummax() - 1).min()
    return sh, yrs, dd


def main():
    panel = pd.read_parquet(PANEL)
    close = panel.pivot_table(index="date", columns="symbol", values="close").sort_index()
    close = close.dropna(axis=1, thresh=int(len(close) * 0.6))   # keep names with decent history

    tfs = [1, 2, 3, 5, 7]
    lb_sets = {
        "fast": [10, 20, 40],
        "mid": [20, 50, 100],
        "slow": [50, 100, 200],
        "blend": [10, 20, 50, 100, 200],
    }
    modes = [("LS", True), ("LF", False)]

    configs = list(itertools.product(tfs, lb_sets.items(), modes))
    print(f"Sweeping {len(configs)} configs (tf x lookbacks x mode)\n", flush=True)

    results = []
    for tf, (lbn, lbs), (mn, ls) in configs:
        lbs_b = [max(1, round(L / tf)) for L in lbs]     # convert day-lookback to bars
        pnl, bpy = trend_book(close, lbs_b, ls, tf)
        if len(pnl) < 50:
            continue
        is_p = pnl[pnl.index <= IS_END]; oos_p = pnl[pnl.index > IS_END]
        sh_is, yrs_is, _ = stats(is_p, bpy)
        sh_all, yrs_all, dd = stats(pnl, bpy)
        neg = sum(1 for v in yrs_all.values() if v < 0)
        worst = min(yrs_all.values()) if yrs_all else -1
        results.append({"cfg": f"tf{tf}/{lbn}/{mn}", "tf": tf, "pnl": pnl, "bpy": bpy,
                        "sh_is": sh_is, "sh_all": sh_all, "neg": neg, "worst": worst, "dd": dd})

    R = pd.DataFrame(results)
    n_trials = len(R)
    # rank by IN-SAMPLE robustness (min IS year), then evaluate OOS — walk-forward honest
    def is_minyear(row):
        is_p = row["pnl"][row["pnl"].index <= IS_END]
        ys = {y: (1 + g).prod() - 1 for y, g in is_p.groupby(is_p.index.year)}
        return min(ys.values()) if ys else -1
    R["is_minyear"] = R.apply(is_minyear, axis=1)
    R = R.sort_values("is_minyear", ascending=False)

    print("=== TOP 8 configs by IN-SAMPLE worst-year (walk-forward selection) ===")
    print(f"{'cfg':16s} {'Sh_all':>7s} {'#neg':>5s} {'worst':>7s} {'MaxDD':>7s}  per-year")
    for _, row in R.head(8).iterrows():
        _, yrs, _ = stats(row["pnl"], row["bpy"])
        ys = " ".join(f"{y}:{v*100:+.0f}%" for y, v in yrs.items())
        print(f"{row['cfg']:16s} {row['sh_all']:7.2f} {row['neg']:5d} {row['worst']*100:6.0f}% "
              f"{row['dd']*100:6.0f}%  {ys}")

    # the winner chosen on IS:
    win = R.iloc[0]
    pnl = win["pnl"]; bpy = win["bpy"]
    oos_p = pnl[pnl.index > IS_END]
    sh_oos = oos_p.mean() / oos_p.std() * np.sqrt(bpy) if oos_p.std() > 0 else 0.0
    srd = pnl.mean() / pnl.std()
    dsr = deflated_sharpe(srd, n_trials, len(pnl))
    print(f"\n=== WALK-FORWARD WINNER (selected on IS<=2023): {win['cfg']} ===")
    sh_all, yrs, dd = stats(pnl, bpy)
    print(f"  full Sharpe={sh_all:.2f}  OOS(2024-26) Sharpe={sh_oos:.2f}  MaxDD={dd*100:.0f}%")
    print(f"  per-year: " + " ".join(f"{y}:{v*100:+.0f}%" for y, v in yrs.items()))
    print(f"  DSR deflated by n_trials={n_trials}: {dsr:.3f}   (#neg years={win['neg']}, worst={win['worst']*100:.0f}%)")
    print(f"\n  TARGET CHECK: every year >60%? "
          f"{'YES' if all(v>0.60 for v in yrs.values()) else 'NO'}  "
          f"(every year positive? {'YES' if all(v>0 for v in yrs.values()) else 'NO'})")


if __name__ == "__main__":
    main()
