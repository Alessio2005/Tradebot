"""run_hrp_backtest.py — Standalone HRP portfolio backtest with elevated vol target.

Usage:
    python run_hrp_backtest.py

Changes vs default backtest_portfolio.py:
  - method = "hrp"       : HRP weights pre-scale requested_leverage per asset
  - target_annual_vol    : 0.25  (was 0.12) → more position size in calm periods
  - max_gross_leverage   : 6.0   (was 4.0)
  - kelly_fraction       : 0.40  (was 0.25)
  - dd_breaker_threshold : 0.30  (was 0.20) → less aggressive halting

Data coverage: 2021-06 → 2026-05  (no 2020 data exists in market_data_parquet/).
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import joblib
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
logger = logging.getLogger("hrp_backtest")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
ARTEFACTS  = _ROOT / "artefacts"
REPORTS    = _ROOT / "reports"
SYMBOLS    = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
BAR_SECS   = 900.0                     # 15-min bars
BARS_PER_YEAR = 365.0 * 24 * 3600.0 / BAR_SECS  # ≈ 35 040

# Elevated risk parameters
TARGET_VOL      = 0.25    # annualised; was 0.12
MAX_GROSS_LEV   = 6.0     # was 4.0
MAX_NET_LEV     = 3.5     # was 2.5
MAX_ASSET_LEV   = 3.0     # was 2.0
# SK-KELLY-DOUBLE FIX (2026-05-18):
#   gap_risk_kelly_size(target_risk=0.01) in evaluation.py implementeert al
#   een conservatieve ATR-gebaseerde sizing (effectief ~1/20 Kelly bij normale
#   crypto-vol). Een tweede kelly_fraction=0.40 hier drukte de leverage
#   onnodig op 40% van de al-conservatieve waarde → factor 2.5× te klein.
#   apps/backtest_portfolio.py gebruikt kelly_fraction=None om dezelfde reden.
#   Fix: None = pass-through (PortfolioRiskManager.kelly_fraction resolves to 1.0).
#   Het vol-target (target_annual_vol=0.25) regelt de schaling.
KELLY_FRAC      = None    # was 0.40 — zie SK-KELLY-DOUBLE fix above
DD_BREAKER      = 0.30    # was 0.20
DD_RESUME       = 0.12    # was 0.10

HRP_LOOKBACK = 2000       # bars of signed_returns used to compute HRP weights


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD EXISTING ASSET TRACKS
# ─────────────────────────────────────────────────────────────────────────────
def load_tracks() -> list:
    tracks = []
    tracks_dir = ARTEFACTS / "tracks"
    for sym in SYMBOLS:
        p = tracks_dir / f"{sym}.joblib"
        if not p.exists():
            logger.warning("Track missing for %s at %s — skip", sym, p)
            continue
        t = joblib.load(p)
        logger.info(
            "[%s] loaded track: %d bars | "
            "signed_returns range [%.4f, %.4f] | "
            "active bars (side≠0): %d",
            sym,
            len(t.signed_returns),
            float(np.nanmin(t.signed_returns)),
            float(np.nanmax(t.signed_returns)),
            int(np.count_nonzero(t.side)),
        )
        tracks.append(t)
    return tracks


# ─────────────────────────────────────────────────────────────────────────────
# 2. COMPUTE HRP WEIGHTS WALK-FORWARD FROM TRACKS
#
# P0-A FIX: the old implementation called sr.tail(HRP_LOOKBACK) → always
# used the LAST 2000 bars of the full 5-year period, i.e. data from 2025-
# to compute weights that are then applied retroactively to 2021-06.
# That is pure lookahead: the HRP weight for bar 1 was determined by
# information that would not be available until bar N-2000.
#
# Correct approach: walk-forward — every REBALANCE_BARS steps, recompute
# HRP using only the bars observed up to that point.  Each new weight
# vector is applied strictly to the next rebalance window.
# ─────────────────────────────────────────────────────────────────────────────
REBALANCE_BARS = 500  # recompute HRP weights every 500 bars (~52 days at 15 min)


def compute_hrp_weights_walkforward(tracks: list) -> dict[str, np.ndarray]:
    """Walk-forward HRP: recompute weights every REBALANCE_BARS using only
    bars available up to that point (strictly causal).

    Returns
    -------
    dict[symbol → np.ndarray]  shape (T,) of per-bar HRP weight for each asset.
    The weights are held constant within each rebalance window and updated at
    the next boundary — exactly as a live system would do.
    """
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

    # SK-HRP-RETURNS FIX (2026-05-18):
    #   signed_returns = price_ret × side (PnL-gesigneerd: positief = winst).
    #   Wanneer een asset short staat (side=-1): signed_return = -price_ret.
    #   Dit spiegelt de correlatie ten opzichte van long-staande assets:
    #   BTC long (side=+1) en ETH short (side=-1) krijgen NEGATIEVE covariantie
    #   ook al stijgen ze samen — puur door de tekenconventie.  Gevolg: HRP ziet
    #   kunstmatige diversificatie en wijst een verkeerde allocatie toe.
    #
    #   Correctie: gebruik prijs-returns voor de covariantie-schatting.
    #   price_ret = signed_return × side  (want signed_ret × side = price_ret × side² = price_ret)
    #   Op flat bars (side=0): price_ret = 0 (geen info → bijdrage nul).
    #
    #   Bijkomend voordeel: de ~98% nul-bars (flat perioden) domineren anders de
    #   LedoitWolf-covariantie.  We bewaren alleen active bars (≥1 asset actief)
    #   als non-zero om de estimatie te concentreren op informatieve perioden.
    price_returns_df = pd.DataFrame(
        {
            t.symbol: (
                np.asarray(t.signed_returns, dtype=np.float64)
                * np.asarray(t.side, dtype=np.float64)
            )
            for t in tracks
        }
    )
    # Zet 0-returns om naar NaN zodat _cov_ledoit_wolf's dropna(how='any')
    # uitsluitend de bars retainiert waar ALLE assets tegelijk actief waren.
    # Als dat te weinig obs zijn (<5), valt hrp_weights terug op fillna(mean).
    price_returns_df = price_returns_df.replace(0.0, np.nan)

    # Per-bar weight arrays — initialise to equal weight
    equal = 1.0 / n_assets
    weight_arrays: dict[str, np.ndarray] = {
        t.symbol: np.full(n_bars, equal, dtype=np.float64) for t in tracks
    }
    current_weights: dict[str, float] = {t.symbol: equal for t in tracks}

    # Walk forward: start recomputing once we have HRP_LOOKBACK bars of history
    for bar_start in range(HRP_LOOKBACK, n_bars, REBALANCE_BARS):
        lookback_start = max(0, bar_start - HRP_LOOKBACK)
        historical = price_returns_df.iloc[lookback_start:bar_start]
        # Behoud alleen bars waar minstens één asset non-NaN is (actieve bars)
        historical = historical.dropna(how="all")

        if len(historical) >= 5:
            try:
                alloc = hrp_weights(historical, research_gate=_HRP_GATE)
                current_weights = alloc.weights.to_dict()
            except DataContractError:
                # Een contractschending (ontbrekende of ongeldige research-gate)
                # is geen numeriek incident en mag NIET worden opgeslokt: het
                # betekent dat HRP zonder verklaring wordt gedraaid.
                raise
            except (ValueError, np.linalg.LinAlgError) as exc:
                logger.warning(
                    "HRP compute failed at bar %d (%s) — keeping previous weights.",
                    bar_start, exc,
                )
            # else: keep current_weights from previous window

        # Apply weights to the next rebalance window
        bar_end = min(bar_start + REBALANCE_BARS, n_bars)
        for sym, w_val in current_weights.items():
            if sym in weight_arrays:
                weight_arrays[sym][bar_start:bar_end] = float(w_val)

    # Log the terminal (most-recent) weights for visibility
    logger.info(
        "Walk-forward HRP: final window weights — %s",
        " | ".join(
            f"{sym}={weight_arrays[sym][-1]:.3f}"
            for sym in sorted(weight_arrays)
        ),
    )
    return weight_arrays


# ─────────────────────────────────────────────────────────────────────────────
# 3. APPLY HRP WEIGHTS TO REQUESTED_LEVERAGE
#    Scale each bar's leverage by the time-varying HRP weight (causal).
#    Formula: lev_scaled[t] = lev_raw[t] * n_assets * hrp_weight[t]
#    The n_assets factor preserves aggregate portfolio-level leverage.
# ─────────────────────────────────────────────────────────────────────────────
def apply_hrp_weights(tracks: list, weight_arrays: dict[str, np.ndarray]) -> list:
    import copy

    n = len(tracks)
    scaled_tracks = []
    for t in tracks:
        w_arr = weight_arrays.get(
            t.symbol, np.full(len(t.requested_leverage), 1.0 / n)
        )
        # Time-varying scale per bar (replaces static scalar)
        scale_arr = n * w_arr
        new_lev = t.requested_leverage * scale_arr
        new_t = copy.copy(t)
        object.__setattr__(new_t, "requested_leverage", new_lev)
        logger.info(
            "[%s] HRP walk-forward: mean_weight=%.3f  mean_scale=%.3f",
            t.symbol, float(np.mean(w_arr)), float(np.mean(scale_arr)),
        )
        scaled_tracks.append(new_t)
    return scaled_tracks


# ─────────────────────────────────────────────────────────────────────────────
# 4. PRINT TRADE ACTIVITY STATISTICS PER PERIOD
# ─────────────────────────────────────────────────────────────────────────────
def print_activity_stats(tracks: list) -> None:
    logger.info("── Trade activity per year ─────────────────────────────────")
    for t in tracks:
        ts  = pd.to_datetime(t.timestamps)
        df  = pd.DataFrame({"side": t.side, "ret": t.signed_returns}, index=ts)
        df.index = pd.DatetimeIndex(df.index)
        df["year"] = df.index.year
        summary = (
            df.groupby("year")
            .agg(
                total_bars=("side", "count"),
                active_bars=("side", lambda x: (x != 0).sum()),
                mean_ret=("ret", "mean"),
            )
            .assign(active_pct=lambda d: 100.0 * d["active_bars"] / d["total_bars"])
        )
        print(f"\n{t.symbol}:")
        print(summary.to_string())


# ─────────────────────────────────────────────────────────────────────────────
# 5. RUN PORTFOLIO BACKTEST
# ─────────────────────────────────────────────────────────────────────────────
def run_portfolio(tracks: list):
    from tradebot.backtest.portfolio import PortfolioBacktester
    from tradebot.risk.portfolio import PortfolioRiskManager

    symbols = [t.symbol for t in tracks]
    risk_mgr = PortfolioRiskManager(
        symbols=symbols,
        target_annual_vol=TARGET_VOL,
        vol_window_bars=500,
        corr_window_bars=1000,
        max_avg_pairwise_corr=0.90,
        max_gross_leverage=MAX_GROSS_LEV,
        max_net_leverage=MAX_NET_LEV,
        max_per_asset_leverage=MAX_ASSET_LEV,
        dd_breaker_threshold=DD_BREAKER,
        dd_resume_threshold=DD_RESUME,
        kelly_fraction=KELLY_FRAC,
        starting_equity=1.0,
        bars_per_year=BARS_PER_YEAR,
    )
    backtester = PortfolioBacktester(
        risk_manager=risk_mgr,
        bars_per_year=BARS_PER_YEAR,
        bar_seconds=BAR_SECS,
    )

    logger.info("Running PortfolioBacktester over %d assets...", len(tracks))
    result = backtester.run(tracks)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 6. REPORT RESULTS
# ─────────────────────────────────────────────────────────────────────────────
def report(result, tracks: list) -> None:
    eq = result.equity_curve

    # Annual breakdown
    if isinstance(eq, pd.Series) and len(eq) > 0:
        eq_dt = eq.copy()
        eq_dt.index = pd.to_datetime(eq_dt.index)
        eq_df = eq_dt.to_frame("equity")
        eq_df["year"] = eq_df.index.year
        ann = (
            eq_df.groupby("year")["equity"]
            .agg(["first", "last"])
            .assign(annual_return=lambda d: d["last"] / d["first"] - 1.0)
        )
        print("\n── Annual equity returns ──────────────────────────────────────")
        print(ann[["annual_return"]].map(lambda x: f"{x:.2%}").to_string())

    print("\n── Portfolio metrics ──────────────────────────────────────────")
    print(f"  Sharpe            : {result.sharpe:.3f}")
    print(f"  Calmar            : {result.calmar:.3f}")
    print(f"  Max drawdown      : {result.max_dd:.2%}")
    print(f"  Total return      : {result.total_return:.2%}")
    print(f"  Deflated Sharpe   : {result.deflated_sharpe:.3f}")
    print(f"  Avg gross leverage: {result.avg_gross_leverage:.3f}")
    print(f"  Avg net leverage  : {result.avg_net_leverage:.3f}")
    print(f"  Avg correlation   : {result.avg_corr:.3f}")
    print(f"  Realized vol      : {result.realized_vol:.2%}")
    print(f"  DD breaker bars   : {result.n_dd_breaker_bars}")
    print(f"  N bars            : {result.n_bars}")
    print(f"  Rebalance cost    : {result.total_rebalance_cost:.4%}")
    print(f"  Funding cost      : {result.total_funding_cost:.4%}")

    if result.per_asset_contribution:
        print("\n── Per-asset PnL contribution ─────────────────────────────────")
        for sym, contrib in sorted(result.per_asset_contribution.items()):
            print(f"  {sym}: {contrib:.4%}")

    # N trades per asset
    print("\n── Trade counts ────────────────────────────────────────────────")
    for t in tracks:
        n_active = int(np.count_nonzero(t.side))
        n_long   = int((np.asarray(t.side) > 0).sum())
        n_short  = int((np.asarray(t.side) < 0).sum())
        print(f"  {t.symbol}: total={n_active}  long={n_long}  short={n_short}")

    # Save
    REPORTS.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS / "hrp_portfolio_metrics.json"
    metrics = {
        "config": {
            "target_annual_vol": TARGET_VOL,
            "max_gross_leverage": MAX_GROSS_LEV,
            "kelly_fraction": KELLY_FRAC,
            "dd_breaker_threshold": DD_BREAKER,
            "hrp_lookback_bars": HRP_LOOKBACK,
        },
        "sharpe":             result.sharpe,
        "calmar":             result.calmar,
        "max_drawdown":       result.max_dd,
        "total_return":       result.total_return,
        "deflated_sharpe":    result.deflated_sharpe,
        "avg_gross_leverage": result.avg_gross_leverage,
        "avg_net_leverage":   result.avg_net_leverage,
        "realized_vol":       result.realized_vol,
        "n_dd_breaker_bars":  result.n_dd_breaker_bars,
        "n_bars":             result.n_bars,
        "total_rebalance_cost": result.total_rebalance_cost,
        "total_funding_cost": result.total_funding_cost,
        "per_asset_contribution": result.per_asset_contribution,
        "n_trades": {
            t.symbol: int(np.count_nonzero(t.side)) for t in tracks
        },
    }
    out_path.write_text(json.dumps(metrics, indent=2, default=float))
    logger.info("Metrics saved → %s", out_path)

    # Save equity curve
    if isinstance(eq, pd.Series) and len(eq) > 0:
        hrp_dir = ARTEFACTS / "portfolio"
        hrp_dir.mkdir(parents=True, exist_ok=True)
        eq_save = eq.copy()
        eq_save.index = pd.to_datetime(eq_save.index)
        eq_save.to_frame("equity").to_parquet(hrp_dir / "hrp_equity_curve.parquet")
        logger.info("Equity curve saved → %s", hrp_dir / "hrp_equity_curve.parquet")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 70)
    print("  HRP Portfolio Backtest  (target_vol=25%, max_gross_lev=6×)")
    print("  Data coverage: 2021-06 → 2026-05  (no 2020 data available)")
    print("=" * 70)

    tracks = load_tracks()
    if not tracks:
        logger.error("No tracks loaded — abort.")
        return

    print_activity_stats(tracks)

    hrp_weight_arrays = compute_hrp_weights_walkforward(tracks)
    tracks_scaled = apply_hrp_weights(tracks, hrp_weight_arrays)

    result = run_portfolio(tracks_scaled)
    report(result, tracks_scaled)


if __name__ == "__main__":
    main()
