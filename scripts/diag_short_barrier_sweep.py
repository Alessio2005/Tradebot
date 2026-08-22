"""diag_short_barrier_sweep.py — Grondige short-alpha analyse (model-vrij).

Test-suite die de short-kant ontkoppelt van de getrainde long-mirror pipeline en
direct de UITKOMSTEN meet onder verschillende keuzes.  Geen modeltrain nodig:
het simuleert een triple-barrier short-exit op de ruwe OHLC-data en boekt
NIET-OVERLAPPENDE trades (sequentieel, net als de echte backtest:
``active_trade_end_idx``).

Drie assen worden gevarieerd:

  1. BARRIER-GEOMETRIE (H1)  — (PT, SL) in ATR-eenheden + horizon t_max.
        symmetric   = PT 2.0 / SL 1.0   (huidige long-mirror default — fout voor shorts?)
        flipped     = PT 1.0 / SL 2.0   ("down the elevator": snelle PT, ruime SL)
        tight       = PT 1.0 / SL 1.0
        fast        = PT 1.5 / SL 1.0   + korte horizon
     ...

  2. ENTRY-CONDITIE (H4 vs trend) — wanneer mag een short open:
        all         = elke bar (onvoorwaardelijke baseline)
        bear        = close < EMA200            (trend/regime short — D3/H1)
        reversion   = close > EMA200·(1+k·atr)  (overshoot boven mean — H4 mean-reversion)
        down_cusum  = neerwaarts CUSUM-event    (event-driven, zoals het systeem)

  3. KOSTEN — per-asset round-trip taker bps (CHIEF AUDIT G3).

Per (asset × geometrie × conditie) rapporteren we: #trades, win-rate, gem. netto
short-return per trade, geannualiseerde Sharpe, gem. holdtijd, timeout-fractie.

CAUSAAL: entry op de NEXT-BAR open (signaal bekend op close[t], fill op open[t+1]),
ATR/EMA strikt backward-looking.  Forward-prijzen worden alleen voor de exit-uitkomst
gebruikt (geen lookahead in entry-beslissing).

Gebruik:
    python -m scripts.diag_short_barrier_sweep
    python -m scripts.diag_short_barrier_sweep --symbols ETHUSDT --bar 1h
    python -m scripts.diag_short_barrier_sweep --csv reports/short_barrier_sweep.csv
"""
from __future__ import annotations

import argparse
import glob
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("diag_short_barrier")

COST_BPS_PER_ASSET = {
    "ETHUSDT": 3.0, "SOLUSDT": 4.0, "AVAXUSDT": 6.0,
    "LINKUSDT": 6.0, "DOTUSDT": 5.0, "BTCUSDT": 2.0,
}
DEFAULT_SYMBOLS = ["ETHUSDT", "SOLUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT"]
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "market_data_parquet"

# Barrier-geometrieën: (label, PT_atr, SL_atr, t_max_bars)
GEOMETRIES = [
    ("symmetric_2_1_t24", 2.0, 1.0, 24),   # huidige long-mirror default
    ("flipped_1_2_t24",   1.0, 2.0, 24),   # down-the-elevator
    ("flipped_1_2_t12",   1.0, 2.0, 12),   # + korte horizon
    ("tight_1_1_t12",     1.0, 1.0, 12),
    ("fast_15_1_t8",      1.5, 1.0, 8),
    ("fast_1_15_t8",      1.0, 1.5, 8),
]
ENTRY_CONDS = ["all", "bear", "reversion", "down_cusum"]


# ---------------------------------------------------------------------------
def load_ohlc(symbol: str, bar: str) -> pd.DataFrame:
    files = sorted(glob.glob(str(DATA_DIR / symbol / "**" / "*.parquet"), recursive=True))
    if not files:
        raise FileNotFoundError(f"Geen prijs-parquets voor {symbol}")
    parts = []
    for f in files:
        df = pd.read_parquet(f, columns=["timestamp", "open", "high", "low", "close"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp").sort_index()
        agg = pd.DataFrame({
            "open": df["open"].resample(bar).first(),
            "high": df["high"].resample(bar).max(),
            "low": df["low"].resample(bar).min(),
            "close": df["close"].resample(bar).last(),
        }).dropna()
        parts.append(agg)
    out = pd.concat(parts).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    logger.info("[%s] %d %s-bars %s -> %s", symbol, len(out), bar,
                out.index.min(), out.index.max())
    return out


def add_indicators(df: pd.DataFrame, atr_win: int = 14, ema_span: int = 200,
                   cusum_k: float = 1.0) -> pd.DataFrame:
    """Voeg causale ATR-fractie, EMA200 en neerwaartse CUSUM-events toe."""
    c = df["close"]
    prev_c = c.shift(1)
    tr = np.maximum(df["high"] - df["low"],
                    np.maximum((df["high"] - prev_c).abs(), (df["low"] - prev_c).abs()))
    atr = tr.rolling(atr_win).mean()
    df["atr_frac"] = (atr / c).shift(1)          # frac van prijs, 1 bar gelagged
    df["ema"] = c.ewm(span=ema_span, adjust=False).mean().shift(1)
    df["bear"] = c.shift(1) < df["ema"]          # regime op vorige close (causaal)

    # Simpele symmetrische CUSUM op log-returns -> neerwaartse events.
    logret = np.log(c / prev_c).fillna(0.0)
    vol = logret.rolling(atr_win).std().shift(1).fillna(method="bfill")
    thr = (cusum_k * vol).to_numpy()
    lr = logret.to_numpy()
    s_pos = 0.0
    s_neg = 0.0
    down_event = np.zeros(len(df), dtype=bool)
    for i in range(len(df)):
        t = thr[i] if np.isfinite(thr[i]) and thr[i] > 0 else 1e9
        s_pos = max(0.0, s_pos + lr[i])
        s_neg = min(0.0, s_neg + lr[i])
        if s_neg < -t:
            down_event[i] = True
            s_neg = 0.0
        if s_pos > t:
            s_pos = 0.0
    df["down_cusum"] = down_event
    return df


def entry_mask(df: pd.DataFrame, cond: str, rev_k: float = 1.5) -> np.ndarray:
    if cond == "all":
        return np.ones(len(df), dtype=bool)
    if cond == "bear":
        return df["bear"].to_numpy()
    if cond == "reversion":
        # overshoot: close (vorige) ruim BOVEN EMA -> short de overextensie
        over = df["close"].shift(1) > df["ema"] * (1.0 + rev_k * df["atr_frac"])
        return over.fillna(False).to_numpy()
    if cond == "down_cusum":
        return df["down_cusum"].to_numpy()
    raise ValueError(cond)


# ---------------------------------------------------------------------------
def simulate_short(df: pd.DataFrame, enter: np.ndarray, pt_atr: float, sl_atr: float,
                   t_max: int, cost_bps: float, bar_hours: float) -> dict:
    """Sequentiële niet-overlappende short triple-barrier simulatie.

    Entry: open[i+1] (next-bar fill).  PT = entry·(1 - pt_atr·atr_frac),
    SL = entry·(1 + sl_atr·atr_frac).  Touch-detectie op low (PT) / high (SL);
    bij dubbele touch in 1 bar => SL (conservatief).  Timeout => close op t_max.
    """
    o = df["open"].to_numpy(np.float64)
    h = df["high"].to_numpy(np.float64)
    l = df["low"].to_numpy(np.float64)
    c = df["close"].to_numpy(np.float64)
    af = df["atr_frac"].to_numpy(np.float64)
    n = len(df)
    cost = cost_bps / 1e4

    rets, holds, wins, timeouts = [], [], 0, 0
    i = 0
    while i < n - 1:
        if not enter[i] or not np.isfinite(af[i]) or af[i] <= 0:
            i += 1
            continue
        entry = o[i + 1]                      # next-bar open fill
        if not np.isfinite(entry) or entry <= 0:
            i += 1
            continue
        pt = entry * (1.0 - pt_atr * af[i])   # short profit = prijs daalt
        sl = entry * (1.0 + sl_atr * af[i])   # short loss   = prijs stijgt
        end = min(i + 1 + t_max, n - 1)
        exit_price = c[end]
        exit_idx = end
        hit = "timeout"
        for j in range(i + 1, end + 1):
            if h[j] >= sl:                    # SL eerst (conservatief)
                exit_price, exit_idx, hit = sl, j, "sl"
                break
            if l[j] <= pt:
                exit_price, exit_idx, hit = pt, j, "pt"
                break
        gross = (entry - exit_price) / entry  # short return
        net = gross - cost
        rets.append(net)
        holds.append(exit_idx - (i + 1))
        wins += int(net > 0)
        timeouts += int(hit == "timeout")
        i = exit_idx + 1                      # non-overlapping: hervat na exit

    if len(rets) < 10:
        return {"N": len(rets), "win": np.nan, "mean_ret": np.nan,
                "sharpe_ann": np.nan, "avg_hold_h": np.nan, "timeout_frac": np.nan}
    r = np.array(rets)
    avg_hold_bars = float(np.mean(holds)) + 1.0
    trades_per_year = (365.0 * 24.0 / bar_hours) / max(avg_hold_bars, 1.0)
    return {
        "N": len(rets),
        "win": float(wins / len(rets)),
        "mean_ret": float(r.mean()),
        "sharpe_ann": float(r.mean() / (r.std(ddof=1) + 1e-12) * np.sqrt(trades_per_year)),
        "avg_hold_h": float(avg_hold_bars * bar_hours),
        "timeout_frac": float(timeouts / len(rets)),
    }


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Short barrier-geometrie sweep")
    ap.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    ap.add_argument("--bar", default="1h")
    ap.add_argument("--csv", default="reports/short_barrier_sweep.csv")
    args = ap.parse_args()

    bar_hours = pd.Timedelta(args.bar).total_seconds() / 3600.0
    rows = []
    for sym in args.symbols:
        try:
            df = load_ohlc(sym, args.bar)
        except FileNotFoundError as e:
            logger.warning("%s overslaan: %s", sym, e)
            continue
        df = add_indicators(df)
        cost = COST_BPS_PER_ASSET.get(sym, 5.0)
        for cond in ENTRY_CONDS:
            enter = entry_mask(df, cond)
            for gname, pt, sl, tmax in GEOMETRIES:
                st = simulate_short(df, enter, pt, sl, tmax, cost, bar_hours)
                rows.append({"symbol": sym, "cond": cond, "geom": gname, **st})
        logger.info("[%s] klaar.", sym)

    res = pd.DataFrame(rows)
    if res.empty:
        logger.error("Geen resultaten.")
        return

    pool = (res.dropna(subset=["mean_ret"])
            .groupby(["cond", "geom"])
            .apply(lambda g: pd.Series({
                "N": int(g["N"].sum()),
                "win": float(np.average(g["win"], weights=g["N"])),
                "mean_ret": float(np.average(g["mean_ret"], weights=g["N"])),
                "sharpe_ann": float(g["sharpe_ann"].mean()),
                "avg_hold_h": float(np.average(g["avg_hold_h"], weights=g["N"])),
                "timeout_frac": float(np.average(g["timeout_frac"], weights=g["N"])),
            }), include_groups=False)
            .reset_index())

    pd.set_option("display.width", 170)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print("\n================ POOLED OVER ASSETS (cond × geometrie) ================")
    print(pool.sort_values(["cond", "sharpe_ann"], ascending=[True, False]).to_string(index=False))
    print("\n================ PER-ASSET (beste geometrie per conditie) =============")
    best = (res.dropna(subset=["sharpe_ann"])
            .sort_values("sharpe_ann", ascending=False)
            .groupby(["symbol", "cond"]).head(1)
            .sort_values(["symbol", "cond"]))
    print(best[["symbol", "cond", "geom", "N", "win", "mean_ret", "sharpe_ann", "avg_hold_h"]].to_string(index=False))

    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(args.csv, index=False)
    logger.info("Volledige matrix -> %s", args.csv)
    print("\nLezing: mean_ret = netto short-PnL per niet-overlappende trade (na kosten).")
    print("  H1 bevestigd als 'flipped'/'fast' geometrie > 'symmetric_2_1'.")
    print("  H4 bevestigd als 'reversion' entry > 'bear'/'all'.")
    print("  Positieve mean_ret + Sharpe = exploiteerbare short-edge.")


if __name__ == "__main__":
    main()
