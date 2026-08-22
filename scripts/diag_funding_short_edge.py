"""diag_funding_short_edge.py — H2 diagnostic: is there positive SHORT edge in funding?

Research question (H2 uit docs/SHORT_ALPHA_RESEARCH_2026-05-30.md):
    Heeft het SHORTEN van een asset positieve forward-EV wanneer de funding-rate
    extreem positief is (crowded longs)?  De short-EV is hier de SOM van:
      (a) price-component  : -(close[t+h]/close[t] - 1)   (short wint als prijs daalt)
      (b) carry-component  : + cum funding over (t, t+h]   (short ontvangt funding>0)
      (c) kosten           : - round-trip taker cost (per-asset bps)

Het script is volledig CAUSAAL aan de signaal-kant:
  - funding z-score gebruikt enkel verleden (rate.shift(1) + rolling window),
    identiek aan tradebot.features.funding_carry.
  - entry op close[t] met het z-signaal dat op t al bekend is.
  - forward return gebruikt toekomstige prijs — dat is de UITKOMST die we meten,
    geen feature (geen lookahead in het signaal).

Sharpe wordt berekend op NIET-OVERLAPPENDE samples (stap = horizon) zodat
overlap-autocorrelatie de Sharpe niet kunstmatig opblaast.

Geen model-afhankelijkheid, geen retrain — draait op de ruwe parquet-data.

Gebruik:
    python -m scripts.diag_funding_short_edge
    python -m scripts.diag_funding_short_edge --symbols ETHUSDT SOLUSDT --bar 1h
    python -m scripts.diag_funding_short_edge --horizons 8 24 72 --csv out.csv
"""
from __future__ import annotations

import argparse
import glob
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("diag_funding_short")

# Per-asset round-trip taker kosten in bps (CHIEF AUDIT 2026-05-23 G3 + memory).
COST_BPS_PER_ASSET = {
    "ETHUSDT": 3.0, "SOLUSDT": 4.0, "AVAXUSDT": 6.0,
    "LINKUSDT": 6.0, "DOTUSDT": 5.0, "BTCUSDT": 2.0,
}
DEFAULT_SYMBOLS = ["ETHUSDT", "SOLUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT"]
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "market_data_parquet"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_price_close(symbol: str, bar: str) -> pd.Series:
    """Laad alle 5-sec parquets voor *symbol* en resample naar *bar* close.

    Per-file resampling houdt het geheugen begrensd (518k rijen/maand).
    """
    files = sorted(glob.glob(str(DATA_DIR / symbol / "**" / "*.parquet"), recursive=True))
    if not files:
        raise FileNotFoundError(f"Geen prijs-parquets voor {symbol} onder {DATA_DIR / symbol}")
    parts: list[pd.Series] = []
    for f in files:
        df = pd.read_parquet(f, columns=["timestamp", "close"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        s = df.set_index("timestamp")["close"].sort_index()
        parts.append(s.resample(bar).last())
    close = pd.concat(parts).sort_index()
    close = close[~close.index.duplicated(keep="last")].dropna()
    logger.info("[%s] price: %d %s-bars %s -> %s", symbol, len(close), bar,
                close.index.min(), close.index.max())
    return close


def load_funding(symbol: str) -> pd.Series:
    """Laad 8h funding-rate reeks (UTC-geindexeerd)."""
    f = DATA_DIR / "macro" / f"macro_crypto_{symbol}.parquet"
    df = pd.read_parquet(f)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    s = df.set_index("timestamp")["fundingRate"].sort_index()
    s = s[~s.index.duplicated(keep="last")].dropna()
    logger.info("[%s] funding: %d posts %s -> %s (mean=%.5f%%/8h)", symbol, len(s),
                s.index.min(), s.index.max(), 100 * float(s.mean()))
    return s


# ---------------------------------------------------------------------------
# Signal construction (causaal)
# ---------------------------------------------------------------------------
def build_panel(close: pd.Series, funding: pd.Series, bar_hours: float,
                z_window_days: int = 30, ema_span: int = 200) -> pd.DataFrame:
    """Bouw een per-bar paneel met causale funding-z, regime en forward-bouwstenen."""
    # Funding op 8h-grid -> causale z-score (shift(1), zelfde als funding_carry.py).
    rate_lagged = funding.shift(1)
    posts_per_day = 3
    zw = max(1, posts_per_day * z_window_days)
    mean_z = rate_lagged.rolling(zw, min_periods=posts_per_day).mean()
    std_z = rate_lagged.rolling(zw, min_periods=posts_per_day).std(ddof=0)
    fund_z = ((rate_lagged - mean_z) / (std_z + 1e-12)).clip(-6, 6)

    # Map 8h funding-grid -> bar-grid via forward-fill (alleen verleden funding).
    df = pd.DataFrame(index=close.index)
    df["close"] = close
    df["fund_rate_lagged"] = rate_lagged.reindex(df.index, method="ffill")
    df["fund_z"] = fund_z.reindex(df.index, method="ffill")
    # Per-bar carry die een SHORT ontvangt (funding>0 -> short income).
    # 8h funding verdeeld over de bars binnen elk 8h-venster.
    bars_per_funding = max(1.0, 8.0 / bar_hours)
    df["carry_per_bar"] = df["fund_rate_lagged"] / bars_per_funding

    # Regime (causale EMA op de bar-close).
    df["ema"] = df["close"].ewm(span=ema_span, adjust=False).mean()
    df["bear"] = df["close"] < df["ema"]

    return df.dropna(subset=["close", "fund_z"])


# ---------------------------------------------------------------------------
# EV evaluation (niet-overlappende samples)
# ---------------------------------------------------------------------------
def short_ev_table(df: pd.DataFrame, symbol: str, horizon_bars: int,
                   bar_hours: float, cost_bps: float) -> list[dict]:
    """Bereken short-EV per funding-bucket op niet-overlappende samples."""
    close = df["close"].to_numpy(np.float64)
    carry = df["carry_per_bar"].to_numpy(np.float64)
    zfund = df["fund_z"].to_numpy(np.float64)
    bear = df["bear"].to_numpy(bool)
    n = len(close)

    # Niet-overlappende entry-indices (stap = horizon).
    idx = np.arange(0, n - horizon_bars, horizon_bars, dtype=int)
    fwd = idx + horizon_bars

    price_short = -(close[fwd] / close[idx] - 1.0)          # short price-PnL
    carry_short = np.array([carry[i:i + horizon_bars].sum() for i in idx])  # short carry-income
    cost = cost_bps / 1e4                                     # round-trip kosten
    total = price_short + carry_short - cost
    z_at = zfund[idx]
    bear_at = bear[idx]

    periods_per_year = (365.0 * 24.0) / (bar_hours * horizon_bars)

    def stats(mask: np.ndarray, label: str) -> dict:
        m = mask & np.isfinite(total)
        nN = int(m.sum())
        if nN < 10:
            return {"symbol": symbol, "horizon_h": int(bar_hours * horizon_bars),
                    "bucket": label, "N": nN, "mean_ret": np.nan, "carry": np.nan,
                    "hit": np.nan, "sharpe_ann": np.nan}
        r = total[m]
        return {
            "symbol": symbol,
            "horizon_h": int(bar_hours * horizon_bars),
            "bucket": label,
            "N": nN,
            "mean_ret": float(r.mean()),
            "carry": float(carry_short[m].mean()),
            "hit": float((r > 0).mean()),
            "sharpe_ann": float(r.mean() / (r.std(ddof=1) + 1e-12) * np.sqrt(periods_per_year)),
        }

    rows = [
        stats(np.ones(len(idx), bool), "all"),
        stats(z_at >= 1.0, "fund_z>=1"),
        stats(z_at >= 2.0, "fund_z>=2"),
        stats((z_at >= 2.0) & bear_at, "fund_z>=2 & bear"),
        stats((z_at >= 1.0) & bear_at, "fund_z>=1 & bear"),
        stats(z_at <= -1.0, "fund_z<=-1 (ctrl)"),
    ]
    return rows


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="H2 funding-short-edge diagnostic")
    ap.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    ap.add_argument("--bar", default="1h", help="resample-frequentie (bv 1h, 4h, 8h)")
    ap.add_argument("--horizons", nargs="+", type=int, default=[8, 24, 72],
                    help="forward horizons in UREN")
    ap.add_argument("--csv", default=None, help="optioneel: schrijf resultaten naar CSV")
    args = ap.parse_args()

    bar_hours = pd.Timedelta(args.bar).total_seconds() / 3600.0
    all_rows: list[dict] = []

    for sym in args.symbols:
        try:
            close = load_price_close(sym, args.bar)
            funding = load_funding(sym)
        except FileNotFoundError as e:
            logger.warning("%s — overslaan: %s", sym, e)
            continue
        df = build_panel(close, funding, bar_hours)
        cost_bps = COST_BPS_PER_ASSET.get(sym, 5.0)
        for h in args.horizons:
            hb = max(1, int(round(h / bar_hours)))
            all_rows.extend(short_ev_table(df, sym, hb, bar_hours, cost_bps))

    res = pd.DataFrame(all_rows)
    if res.empty:
        logger.error("Geen resultaten — controleer data-paden.")
        return

    # Geaggregeerd over assets (N-gewogen mean_ret, gemiddelde Sharpe).
    pool = (res.dropna(subset=["mean_ret"])
            .groupby(["horizon_h", "bucket"])
            .apply(lambda g: pd.Series({
                "N": int(g["N"].sum()),
                "mean_ret": float(np.average(g["mean_ret"], weights=g["N"])),
                "carry": float(np.average(g["carry"], weights=g["N"])),
                "hit": float(np.average(g["hit"], weights=g["N"])),
                "sharpe_ann": float(g["sharpe_ann"].mean()),
            }), include_groups=False)
            .reset_index())

    pd.set_option("display.width", 160)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print("\n================ PER-ASSET SHORT-EV (price + carry - cost) ================")
    print(res.to_string(index=False))
    print("\n================ POOLED OVER ASSETS (N-weighted) =========================")
    print(pool.to_string(index=False))
    print("\nLezing: mean_ret = gem. short-PnL per niet-overlappende trade (decimaal).")
    print("        carry    = gem. funding-income-component (short).")
    print("        Positieve mean_ret + Sharpe in 'fund_z>=2'/'& bear' buckets => H2 bevestigd.")
    print("        'fund_z<=-1 (ctrl)' hoort NEGATIEF te zijn (short tegen de carry).")

    if args.csv:
        res.to_csv(args.csv, index=False)
        logger.info("Resultaten weggeschreven naar %s", args.csv)


if __name__ == "__main__":
    main()
