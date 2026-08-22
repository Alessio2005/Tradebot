"""diag_xsectional_ls.py — H3 capstone: heeft de SHORT-LEG edge in een
cross-sectionele (market-neutral) long/short constructie?

De barrier-sweep toonde aan: directionele shorts hebben geen universele edge, maar
er is sterke DISPERSIE tussen assets (DOT daalde -> short-positief; ETH steeg ->
short-negatief).  Een cross-sectionele L/S exploiteert juist die dispersie:
  - LONG  de relatieve winnaar(s)  (top-k trailing momentum)
  - SHORT de relatieve verliezer(s) (bottom-k)
De EV van het SHORT-been hangt dan af van relatieve underperformance, niet van
absolute marktrichting -> de seculiere long-drift valt grotendeels weg.

We meten 3 sleeves apart zodat de short-bijdrage zichtbaar is:
  1. long_only   : alleen het long-been (benchmark)
  2. short_leg    : alleen het short-been  (DIT is de short-alpha vraag)
  3. dollar_neutral: long - short (β-neutrale spread)

Plus een 'naive_short_all' controle: elke dag alle 5 short (de onvoorwaardelijke
short — hoort sterk negatief te zijn).

CAUSAAL: momentum-rank op trailing return t/m gisteren; positie aangehouden tot de
volgende rebalance; entry op de volgende bar.  Geen lookahead.

Gebruik:
    python -m scripts.diag_xsectional_ls
    python -m scripts.diag_xsectional_ls --lookback_d 30 --rebalance_d 7 --k 1
"""
from __future__ import annotations

import argparse
import glob
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("diag_xs_ls")

COST_BPS = {"ETHUSDT": 3.0, "SOLUSDT": 4.0, "AVAXUSDT": 6.0,
            "LINKUSDT": 6.0, "DOTUSDT": 5.0, "BTCUSDT": 2.0}
DEFAULT_SYMBOLS = ["ETHUSDT", "SOLUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT"]
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "market_data_parquet"


def load_daily_close(symbol: str) -> pd.Series:
    files = sorted(glob.glob(str(DATA_DIR / symbol / "**" / "*.parquet"), recursive=True))
    parts = []
    for f in files:
        df = pd.read_parquet(f, columns=["timestamp", "close"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        parts.append(df.set_index("timestamp")["close"].sort_index().resample("1D").last())
    s = pd.concat(parts).sort_index()
    return s[~s.index.duplicated(keep="last")].dropna()


def perf(daily_ret: pd.Series, label: str) -> dict:
    r = daily_ret.dropna()
    if len(r) < 30:
        return {"sleeve": label, "days": len(r), "ann_ret": np.nan,
                "ann_vol": np.nan, "sharpe": np.nan, "maxdd": np.nan, "hit": np.nan}
    eq = (1.0 + r).cumprod()
    dd = (eq / eq.cummax() - 1.0).min()
    return {
        "sleeve": label,
        "days": len(r),
        "ann_ret": float(r.mean() * 365),
        "ann_vol": float(r.std(ddof=1) * np.sqrt(365)),
        "sharpe": float(r.mean() / (r.std(ddof=1) + 1e-12) * np.sqrt(365)),
        "maxdd": float(dd),
        "hit": float((r > 0).mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    ap.add_argument("--lookback_d", type=int, default=30, help="momentum-lookback (dagen)")
    ap.add_argument("--rebalance_d", type=int, default=7, help="rebalance-interval (dagen)")
    ap.add_argument("--k", type=int, default=1, help="aantal assets per been (top-k / bottom-k)")
    ap.add_argument("--csv", default="reports/xsectional_ls.csv")
    args = ap.parse_args()

    closes = {}
    for s in args.symbols:
        try:
            closes[s] = load_daily_close(s)
        except Exception as e:
            logger.warning("%s overslaan: %s", s, e)
    px = pd.DataFrame(closes).dropna()
    logger.info("Common daily grid: %d dagen, %d assets %s -> %s",
                len(px), px.shape[1], px.index.min(), px.index.max())
    syms = list(px.columns)
    ret1d = px.pct_change()
    mom = px.pct_change(args.lookback_d)           # trailing momentum
    cost = np.array([COST_BPS.get(s, 5.0) for s in syms]) / 1e4

    n = len(px)
    # Doel-gewichten per dag (gehouden tussen rebalances).
    w_long = pd.DataFrame(0.0, index=px.index, columns=syms)
    w_short = pd.DataFrame(0.0, index=px.index, columns=syms)
    w_naive = pd.DataFrame(0.0, index=px.index, columns=syms)
    last = {}
    for i in range(n):
        if i < args.lookback_d + 1:
            continue
        if (i % args.rebalance_d) == 0 or not last:
            m = mom.iloc[i]                        # bekend t/m vandaag (causaal)
            order = m.sort_values()
            losers = order.index[:args.k]
            winners = order.index[-args.k:]
            wl = pd.Series(0.0, index=syms); wl[winners] = 1.0 / args.k
            ws = pd.Series(0.0, index=syms); ws[losers] = 1.0 / args.k
            wn = pd.Series(1.0 / len(syms), index=syms)
            last = {"wl": wl, "ws": ws, "wn": wn}
        w_long.iloc[i] = last["wl"].values
        w_short.iloc[i] = last["ws"].values
        w_naive.iloc[i] = last["wn"].values

    # Returns: positie aangehouden -> verdient de volgende-dag return (shift entry).
    fwd = ret1d.shift(-1)                           # next-day return per asset
    # Transactiekosten op gewichts-deltas.
    def tc(w: pd.DataFrame) -> pd.Series:
        return (w.diff().abs() * cost).sum(axis=1)

    long_ret = (w_long * fwd).sum(axis=1) - tc(w_long)
    short_ret = (-(w_short * fwd)).sum(axis=1) - tc(w_short)     # SHORT verdient -return
    naive_short = (-(w_naive * fwd)).sum(axis=1) - tc(w_naive)
    neutral = long_ret + short_ret                  # dollar-neutraal (long+short sleeves)

    rows = [
        perf(long_ret, "1_long_only (winnaars)"),
        perf(short_ret, "2_short_leg (verliezers)  <= short-alpha"),
        perf(neutral, "3_dollar_neutral L/S"),
        perf(naive_short, "4_naive_short_all (controle, hoort neg)"),
    ]
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 170)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print(f"\n===== CROSS-SECTIONAL L/S  (lookback={args.lookback_d}d, "
          f"rebalance={args.rebalance_d}d, k={args.k}) =====")
    print(res.to_string(index=False))
    print("\nLezing:")
    print("  sleeve 2 (short_leg) POSITIEVE Sharpe => cross-sectionele short HEEFT edge (H3 bevestigd).")
    print("  sleeve 4 (naive_short_all) hoort sterk NEGATIEF (onvoorwaardelijke short).")
    print("  sleeve 3 (dollar_neutral) = market-neutraal totaalproduct.")

    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(args.csv, index=False)


if __name__ == "__main__":
    main()
