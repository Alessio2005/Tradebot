"""validate_hrp_pipeline.py — Synthetische end-to-end validatie van HRP + Portfolio.

Doel: bevestig Sharpe > 2 door de volledige machinery (HRP walk-forward,
PortfolioBacktester, PortfolioRiskManager) te testen zonder de zware ML-pipeline.

Signaal: EMA(20) > EMA(80) → LONG, anders FLAT (geen shorts in dit test).
Data: BTC + ETH + SOL, 15-min bars, 2022-2024 (3 jaar OOS-periode).

Resultaat wordt vergeleken met target Sharpe > 2.0.

Gebruik:
    cd C:\\Users\\algul\\Documents\\Tradebot
    python validate_hrp_pipeline.py
"""
from __future__ import annotations

import glob
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent
_SRC  = _ROOT / "src"
for p in (_SRC, _ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("validate_hrp")

DATA_DIR  = _ROOT / "market_data_parquet"
SYMBOLS   = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
YEAR_FROM = 2022
YEAR_TO   = 2025   # exclusief (laden t/m 2024)
BAR_SECS  = 900.0  # 15 minuten
EMA_FAST  = 20     # bars
EMA_SLOW  = 80     # bars


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA LADEN + RESAMPLEN
# ─────────────────────────────────────────────────────────────────────────────
def load_ohlcv(sym: str) -> pd.DataFrame:
    """Laad alle maand-parquets voor één symbol en resample naar 15-min OHLCV."""
    frames = []
    for yr in range(YEAR_FROM, YEAR_TO):
        pattern = str(DATA_DIR / sym / str(yr) / f"{sym}_{yr}-*.parquet")
        for fp in sorted(glob.glob(pattern)):
            try:
                df = pd.read_parquet(fp, columns=["timestamp", "open", "high", "low", "close", "real_volume"])
                df = df.set_index("timestamp")
                frames.append(df)
            except Exception as exc:
                logger.debug("[%s] skip %s: %s", sym, fp, exc)
    if not frames:
        raise FileNotFoundError(f"Geen data voor {sym} in {YEAR_FROM}-{YEAR_TO}")

    raw = pd.concat(frames).sort_index()
    raw.index = pd.to_datetime(raw.index, utc=True)

    # Resample tick-data → 15-min OHLCV
    rule = f"{int(BAR_SECS)}s"
    ohlcv = raw.resample(rule).agg(
        open=("open",   "first"),
        high=("high",   "max"),
        low=("low",    "min"),
        close=("close",  "last"),
        volume=("real_volume", "sum"),
    ).dropna(subset=["close"])

    logger.info("[%s] loaded %d 15-min bars (%s → %s)", sym,
                len(ohlcv), ohlcv.index[0], ohlcv.index[-1])
    return ohlcv


# ─────────────────────────────────────────────────────────────────────────────
# 2. SIGNAAL GENERATIE: EMA CROSSOVER (volledig causal)
# ─────────────────────────────────────────────────────────────────────────────
def generate_signals(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Genereer LONG/FLAT signalen op basis van EMA-crossover.

    entry : EMA_FAST kruist boven EMA_SLOW → side=+1
    exit  : EMA_FAST kruist onder EMA_SLOW → side=0
    """
    close = ohlcv["close"]
    ema_fast = close.ewm(span=EMA_FAST, adjust=False).mean()
    ema_slow = close.ewm(span=EMA_SLOW, adjust=False).mean()

    # Signaal: 1 wanneer fast > slow (LONG), 0 anders
    raw_signal = (ema_fast > ema_slow).astype(int)

    # Bereken per-bar returns
    log_ret = np.log(close / close.shift(1)).fillna(0.0)
    price_ret_simple = np.expm1(log_ret)

    # signed_return = price_ret * side (PnL-gesigneerd)
    signed_ret = price_ret_simple * raw_signal

    # requested_leverage: vaste ATR-gebaseerde schatting (eenvoudig voor validatie)
    atr_pct = (ohlcv["high"] - ohlcv["low"]).rolling(14).mean() / close
    atr_pct = atr_pct.fillna(atr_pct.mean())
    # Kelly sizing: target_risk=1% / (ATR_pct × gap_factor=2)
    gap_factor = 2.0  # BTC gap-risk multiple
    requested_lev = np.clip(0.01 / (atr_pct * gap_factor), 0.0, 2.0)
    requested_lev = requested_lev * raw_signal  # 0 wanneer flat

    df = pd.DataFrame({
        "close":             close,
        "signed_return":     signed_ret,
        "requested_leverage": requested_lev,
        "side":              raw_signal.astype(int),
    }, index=ohlcv.index)

    n_active = int(raw_signal.sum())
    logger.info("[signal] active bars=%d / %d (%.1f%%)",
                n_active, len(df), 100 * n_active / max(len(df), 1))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. ASSET TRACKS AANMAKEN
# ─────────────────────────────────────────────────────────────────────────────
def build_track(sym: str, df: pd.DataFrame):
    from tradebot.backtest.tracks import AssetTrack
    from typing import cast

    n = len(df)
    timestamps = cast(pd.DatetimeIndex, pd.to_datetime(df.index))
    return AssetTrack(
        symbol=sym,
        timestamps=timestamps,
        signed_returns=df["signed_return"].to_numpy(dtype=np.float64),
        requested_leverage=df["requested_leverage"].to_numpy(dtype=np.float64),
        side=df["side"].to_numpy(dtype=np.int64),
        per_asset_cap=2.0,
        cost_bps=5.0,   # 5 bp taker fee
        funding_rate=None,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 4. HRP WALK-FORWARD GEWICHTEN
# ─────────────────────────────────────────────────────────────────────────────
HRP_LOOKBACK   = 2000   # bars lookback window voor covariantie
REBALANCE_BARS = 500    # rebalanceer HRP elke 500 bars (~52 dagen)


def compute_hrp_weights_walkforward(tracks: list) -> dict[str, np.ndarray]:
    """Causale walk-forward HRP met price-returns (SK-HRP-RETURNS fix)."""
    from tradebot.portfolio.hrp import HrpResearchGate, hrp_weights
    from tradebot.utils.failfast import DataContractError

    # ---------------------------------------------------------------------------
    # PHASE 7/8 STAGE C-1: HRP is RESEARCH ONLY en technisch geblokkeerd voor
    # productie (fase-6 deliverable 23, no-go 15). `hrp_weights` eist daarom een
    # expliciet token. Dit is een researchscript, dus dat token hoort hier thuis --
    # met een reden die opschrijft waarvoor.
    # ---------------------------------------------------------------------------
    _HRP_GATE = HrpResearchGate(
        reason=("legacy researchscript: HRP-herweging over walk-forward vensters, "
                "als invoer voor de vergelijking met Inverse Volatility"),
        preregistration_id="legacy-miscellaneous-hrp-backtest",
    )

    n_assets = len(tracks)
    if n_assets == 0:
        return {}

    n_bars = len(tracks[0].signed_returns)

    # SK-HRP-RETURNS FIX: gebruik price-returns, niet PnL-returns
    # price_ret = signed_return × side (zie SK-HRP-RETURNS fix in run_hrp_backtest.py)
    price_returns_df = pd.DataFrame({
        t.symbol: (
            np.asarray(t.signed_returns, dtype=np.float64)
            * np.asarray(t.side, dtype=np.float64)
        )
        for t in tracks
    })
    price_returns_df = price_returns_df.replace(0.0, np.nan)

    equal = 1.0 / n_assets
    weight_arrays: dict[str, np.ndarray] = {
        t.symbol: np.full(n_bars, equal, dtype=np.float64) for t in tracks
    }
    current_weights: dict[str, float] = {t.symbol: equal for t in tracks}

    for bar_start in range(HRP_LOOKBACK, n_bars, REBALANCE_BARS):
        lookback_start = max(0, bar_start - HRP_LOOKBACK)
        historical = price_returns_df.iloc[lookback_start:bar_start].dropna(how="all")

        if len(historical) >= 5:
            try:
                alloc = hrp_weights(historical, research_gate=_HRP_GATE)
                current_weights = alloc.weights.to_dict()
            except DataContractError:
                # Zie run_hrp_backtest.py: een ontbrekende research-gate is een
                # contractschending, geen numeriek incident.
                raise
            except (ValueError, np.linalg.LinAlgError) as exc:
                logger.warning("HRP mislukt bij bar %d: %s", bar_start, exc)

        bar_end = min(bar_start + REBALANCE_BARS, n_bars)
        for sym, w_val in current_weights.items():
            if sym in weight_arrays:
                weight_arrays[sym][bar_start:bar_end] = float(w_val)

    logger.info("HRP eindgewichten: %s",
                " | ".join(f"{s}={weight_arrays[s][-1]:.3f}" for s in sorted(weight_arrays)))
    return weight_arrays


def apply_hrp_weights(tracks: list, weight_arrays: dict[str, np.ndarray]) -> list:
    import copy
    n = len(tracks)
    scaled = []
    for t in tracks:
        w_arr = weight_arrays.get(t.symbol, np.full(len(t.requested_leverage), 1.0 / n))
        scale_arr = n * w_arr
        new_lev = t.requested_leverage * scale_arr
        new_t = copy.copy(t)
        object.__setattr__(new_t, "requested_leverage", new_lev)
        logger.info("[%s] HRP mean_weight=%.3f mean_scale=%.3f",
                    t.symbol, float(np.mean(w_arr)), float(np.mean(scale_arr)))
        scaled.append(new_t)
    return scaled


# ─────────────────────────────────────────────────────────────────────────────
# 5. PORTFOLIO BACKTEST
# ─────────────────────────────────────────────────────────────────────────────
def run_portfolio(tracks: list):
    from tradebot.backtest.portfolio import PortfolioBacktester
    from tradebot.risk.portfolio import PortfolioRiskManager

    symbols = [t.symbol for t in tracks]
    n_bars = len(tracks[0].signed_returns)
    # Bereken daadwerkelijke bars-per-jaar uit de data
    ts0 = tracks[0].timestamps[0]
    ts1 = tracks[0].timestamps[-1]
    elapsed_years = max((ts1 - ts0).total_seconds() / (365.25 * 86400), 1e-9)
    bars_per_year = float(n_bars) / elapsed_years
    logger.info("bars_per_year (actueel) = %.0f", bars_per_year)

    risk_mgr = PortfolioRiskManager(
        symbols=symbols,
        target_annual_vol=0.25,
        vol_window_bars=500,
        corr_window_bars=1000,
        max_avg_pairwise_corr=0.90,
        max_gross_leverage=6.0,
        max_net_leverage=3.5,
        max_per_asset_leverage=3.0,
        dd_breaker_threshold=0.30,
        dd_resume_threshold=0.12,
        kelly_fraction=None,    # SK-KELLY-DOUBLE fix: pass-through
        starting_equity=1.0,
        bars_per_year=bars_per_year,
    )
    backtester = PortfolioBacktester(
        risk_manager=risk_mgr,
        bars_per_year=bars_per_year,
        bar_seconds=BAR_SECS,
    )
    logger.info("PortfolioBacktester draait over %d assets × %d bars...", len(tracks), n_bars)
    return backtester.run(tracks)


# ─────────────────────────────────────────────────────────────────────────────
# 6. RAPPORTAGE
# ─────────────────────────────────────────────────────────────────────────────
def report(result, tracks: list) -> None:
    print("\n" + "=" * 65)
    print("  VALIDATE_HRP_PIPELINE — EMA crossover (BTC + ETH + SOL)")
    print("=" * 65)
    print(f"  Sharpe (annualized)  : {result.sharpe:.3f}  {'OK > 2.0' if result.sharpe > 2.0 else 'FAIL < 2.0'}")
    print(f"  Deflated Sharpe      : {result.deflated_sharpe:.3f}")
    print(f"  Calmar               : {result.calmar:.3f}")
    print(f"  Max drawdown         : {result.max_dd:.2%}")
    print(f"  Total return         : {result.total_return:.2%}")
    print(f"  Avg gross leverage   : {result.avg_gross_leverage:.4f}")
    print(f"  Realized vol         : {result.realized_vol:.2%}")
    print(f"  DD breaker bars      : {result.n_dd_breaker_bars}")
    print(f"  N bars               : {result.n_bars}")
    print(f"  Total rebalance cost : {result.total_rebalance_cost:.4%}")
    print(f"  Total funding cost   : {result.total_funding_cost:.4%}")

    if result.bootstrap:
        bs = result.bootstrap
        print(f"  Bootstrap Sharpe CI  : [{bs.get('sharpe_lower_5', 0):.2f}, "
              f"{bs.get('sharpe_upper_95', 0):.2f}] (90% CI)")

    if result.per_asset_contribution:
        print("\n  Per-asset PnL contribution:")
        for sym, c in sorted(result.per_asset_contribution.items()):
            print(f"    {sym}: {c:.4%}")

    print("\n  Trade counts per asset:")
    for t in tracks:
        n_active = int(np.count_nonzero(t.side))
        print(f"    {t.symbol}: active_bars={n_active}")

    # Equity breakdown per jaar
    eq = result.equity_curve
    if isinstance(eq, pd.Series) and len(eq) > 0:
        eq_dt = eq.copy()
        if not isinstance(eq_dt.index, pd.DatetimeIndex):
            eq_dt.index = pd.to_datetime(eq_dt.index, utc=True)
        eq_df = eq_dt.to_frame("equity")
        eq_df["year"] = eq_df.index.year
        ann = (
            eq_df.groupby("year")["equity"]
            .agg(["first", "last"])
            .assign(annual_return=lambda d: d["last"] / d["first"] - 1.0)
        )
        print("\n  Jaarlijkse returns:")
        for yr, row in ann.iterrows():
            print(f"    {yr}: {row['annual_return']:.2%}")

    verdict = "GESLAAGD (PASS)" if result.sharpe > 2.0 else "GEZAKT (FAIL)"
    print(f"\n  AUDIT VERDICT: {verdict}  (target Sharpe > 2.0)")
    print("=" * 65 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    logger.info("=== VALIDATE_HRP_PIPELINE START ===")

    # Stap 1: data laden
    dfs: dict[str, pd.DataFrame] = {}
    for sym in SYMBOLS:
        try:
            ohlcv = load_ohlcv(sym)
            dfs[sym] = ohlcv
        except FileNotFoundError as exc:
            logger.error("[%s] data niet gevonden: %s", sym, exc)

    if not dfs:
        logger.error("Geen data geladen — abort.")
        return

    # Stap 2: signalen genereren
    sig_dfs: dict[str, pd.DataFrame] = {}
    for sym, ohlcv in dfs.items():
        sig_dfs[sym] = generate_signals(ohlcv)

    # Stap 3: gemeenschappelijk tijdsraster (intersectie zodat alle assets dezelfde range hebben)
    common_idx = None
    for df in sig_dfs.values():
        if common_idx is None:
            common_idx = df.index
        else:
            common_idx = common_idx.intersection(df.index)
    if common_idx is None or len(common_idx) == 0:
        logger.error("Geen gemeenschappelijk tijdsraster — abort.")
        return
    logger.info("Gemeenschappelijk raster: %d bars (%s → %s)",
                len(common_idx), common_idx[0], common_idx[-1])

    for sym in list(sig_dfs.keys()):
        sig_dfs[sym] = sig_dfs[sym].loc[common_idx]

    # Stap 4: asset tracks bouwen
    tracks = [build_track(sym, sig_dfs[sym]) for sym in SYMBOLS if sym in sig_dfs]
    if not tracks:
        logger.error("Geen tracks — abort.")
        return

    # Stap 5: HRP walk-forward gewichten berekenen + toepassen
    hrp_weights_arr = compute_hrp_weights_walkforward(tracks)
    tracks_scaled = apply_hrp_weights(tracks, hrp_weights_arr)

    # Stap 6: portfolio backtest runnen
    result = run_portfolio(tracks_scaled)

    # Stap 7: rapporteren
    report(result, tracks_scaled)

    logger.info("=== VALIDATE_HRP_PIPELINE KLAAR ===")
    return result


if __name__ == "__main__":
    main()
