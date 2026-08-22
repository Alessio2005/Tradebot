# src/tradebot/backtest/tracks.py
"""Per-asset OOS track dataclasses and helpers.

Migrated from portfolio_backtest.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

import numpy as np
import pandas as pd

# CHIEF AUDIT 2026-05-23 (H6): optionele fee_schedule koppeling — laat callers
# een live-tier (VIP1-3) doorgeven i.p.v. de hardcoded VIP0-aanname.
from tradebot.execution.fees import FeeSchedule


@dataclass
class AssetTrack:
    """Per-asset OOS-track op event-grid.

    `signed_returns` is de echte bar-by-bar PnL van die asset (price_return
    * side, na spread en slippage). De portfolio-laag moet alleen nog de
    leverage-multiplier toepassen.

    `requested_leverage` is de raw leverage die de Kelly+CVaR-laag voorschreef
    (zonder portfolio-cap). De PortfolioRiskManager schaalt dit met vol/corr.

    `side` is +1/-1/0 per bar.

    REBALANCE-COST-FIX: ``cost_bps`` is de éénzijdige fee+spread in basispunten
    (1bp = 0.0001). De portfolio-laag rekent ``cost_bps`` af op de absolute
    delta van de signed-leverage van bar t-1 → t. Hiermee wordt het "gratis
    herbalanceren" door vol-target / correlatie-scaler corrigeerend belast.

    FUNDING-FIX: ``funding_rate`` is de per-bar funding rate (decimal, NIET
    geannualiseerd) — typisch 0 op alle bars behalve de funding-tick (elke
    8 uur op Bybit). Long betaalt funding bij positieve rate; short
    ontvangt het. ``signed_leverage * funding_rate`` wordt van de bar-PnL
    afgetrokken.
    """
    symbol: str
    timestamps: pd.DatetimeIndex
    signed_returns: np.ndarray            # shape (T,) raw price_return × side
    requested_leverage: np.ndarray        # shape (T,) raw Kelly leverage per bar
    side: np.ndarray                      # shape (T,) +1/-1/0
    per_asset_cap: float = 2.0
    cost_bps: float = 5.0                 # 5bp ≈ Bybit taker (0.04%) + 1bp spread
    funding_rate: np.ndarray | None = None  # shape (T,) of None = no funding
    # AUDIT-FIX (Round 3 — Funding Δt correctie):
    # False (default) = sparse convention: rate ≠ 0 slechts op 8h-ticks
    #   (aangemaakt door _load_per_bar_funding_rate). Geen Δt-schaling nodig.
    # True = dense convention: volledige 8h-rate ffill'd op ELKE bar.
    #   Backtester past automatisch Δt = bar_seconds/28800 toe per bar.
    funding_rate_is_dense: bool = False
    # ── BLUEPRINT-FIX (IV.7) — fill-metadata voor KalmanImpactObserver ──
    # Per-bar dictionary van arrays (alle shape (T,)):
    #   "realised_slippage" : gemeten slippage per bar (decimal); 0 op no-fill bars.
    #   "sigma_per_bar"     : bar-σ als decimaal.
    #   "order_size"        : Q van de fill (basis-asset); 0 op no-fill bars.
    #   "bar_volume"        : V van de bar.
    # Wanneer aangeleverd EN ``stack`` op de backtester is gezet, doet de
    # backtester per non-zero fill ``stack.eta_observer.update(s,σ,Q,V)``.
    # ``None`` ⇒ geen η-update (legacy gedrag).
    fill_metadata: dict[str, np.ndarray] | None = None
    # CHIEF AUDIT 2026-05-23 (H6): optionele live-tier fee schedule. Wanneer
    # aangeleverd gebruikt de portfolio-laag ``fee_schedule.taker_bps`` * 2
    # voor cost_bps i.p.v. de hardcoded VIP0-default (10 bps round-trip).
    # Live accounts zijn vaak VIP1-3 (taker 3.0-4.0 bps one-way) → bespaart
    # 2-4 bps round-trip per trade. Default None = legacy 10 bps gedrag.
    fee_schedule: FeeSchedule | None = None


@dataclass

class PortfolioBacktestResult:
    """Output van PortfolioBacktester.run()."""
    equity_curve: pd.Series
    portfolio_returns: pd.Series
    sharpe: float
    max_dd: float
    calmar: float
    total_return: float
    deflated_sharpe: float
    bootstrap: dict[str, float] = field(default_factory=dict)
    per_asset_contribution: dict[str, float] = field(default_factory=dict)
    avg_gross_leverage: float = 0.0
    avg_net_leverage: float = 0.0
    avg_corr: float = 0.0
    realized_vol: float = 0.0
    n_dd_breaker_bars: int = 0
    n_bars: int = 0
    # REBALANCE-COST-FIX + FUNDING-FIX diagnostics
    total_rebalance_cost: float = 0.0
    total_funding_cost: float = 0.0
    avg_turnover: float = 0.0


# =============================================================================
# PORTFOLIO BACKTESTER
# =============================================================================

def asset_track_from_backtest(
    symbol: str,
    df_test: pd.DataFrame,
    bar_returns: np.ndarray,
    sides: np.ndarray,
    requested_leverage: np.ndarray,
    per_asset_cap: float = 2.0,
    cost_bps: float = 5.0,
    funding_rate: np.ndarray | None = None,
    fee_schedule: FeeSchedule | None = None,
) -> AssetTrack:
    """Wrap de output van train_regime.internal_backtest / bidirectional_backtest.

    df_test          : DataFrame met DatetimeIndex over de OOS-periode.
    bar_returns      : per-bar PnL (al side-aware: positief = winst voor de positie).
    sides            : per-bar +1/-1/0.
    requested_leverage: per-bar raw Kelly leverage (=0.0 op flat bars).
    cost_bps         : éénzijdige fee+spread in bps (Bybit taker ~5bp).
    funding_rate     : optionele per-bar funding rate (decimal, niet
                       geannualiseerd) of None.
    """
    n = len(df_test)
    if not (len(bar_returns) == n and len(sides) == n and len(requested_leverage) == n):
        # Pad/truncate defensief
        m = min(n, len(bar_returns), len(sides), len(requested_leverage))
        df_test = df_test.iloc[:m]
        bar_returns = np.asarray(bar_returns)[:m]
        sides = np.asarray(sides)[:m]
        requested_leverage = np.asarray(requested_leverage)[:m]
        if funding_rate is not None and len(funding_rate) >= m:
            funding_rate = np.asarray(funding_rate)[:m]

    funding_arr: np.ndarray | None = None
    if funding_rate is not None:
        funding_arr = np.asarray(funding_rate, dtype=np.float64)
        if funding_arr.shape[0] != len(df_test):
            # Mismatch — drop rather than misalign
            import logging as _logging
            _logging.getLogger("portfolio_backtest").warning(
                "asset_track_from_backtest [%s]: funding_rate lengte %d != "
                "df_test lengte %d — funding genegeerd om misalignering te voorkomen.",
                symbol, funding_arr.shape[0], len(df_test),
            )
            funding_arr = None
        else:
            # AUDIT-FIX (issue #2 — Funding Validation):
            #   funding_rate moet een *per-event* waarde zijn (niet geannualiseerd,
            #   niet continu-geïntegreerd).  Typische Bybit 8h funding is ~0.01%
            #   per event → 1e-4 decimaal.  Als de waarden groter zijn dan 5% per
            #   event (= 0.05 decimaal) is er waarschijnlijk een schaalfout
            #   (bv. dagelijkse of geannualiseerde funding doorgegeven).
            _MAX_FUNDING_PER_EVENT = 0.05
            abs_max = float(np.nanmax(np.abs(funding_arr)))
            if abs_max > _MAX_FUNDING_PER_EVENT:
                import logging as _logging2
                _logging2.getLogger("portfolio_backtest").warning(
                    "asset_track_from_backtest [%s]: funding_rate max abs waarde "
                    "= %.6f (%.4f%%) overschrijdt de verwachte per-event drempel "
                    "van %.2f%%.  Controleer of je een geannualiseerde of dagelijkse "
                    "rate doorgeeft in plaats van een per-bar waarde.",
                    symbol, abs_max, abs_max * 100.0, _MAX_FUNDING_PER_EVENT * 100.0,
                )

            # AUDIT-FIX (K1 — Funding Sparsity Guard):
            #   funding moet sparse zijn (≤ 1 tick per 8h, dus ~3 ticks/dag bij
            #   continue bars). Als per ongeluk een ffill'd reeks wordt
            #   doorgegeven (zelfde rate herhaald op elke bar), dan loopt
            #   ``signed_lev * fr`` per bar de funding-cost vele malen op.
            #   We waarschuwen wanneer > 25% van de bars een non-zero rate
            #   heeft — bij realistische Bybit 8h cadence is dat onmogelijk.
            n_total = int(funding_arr.size)
            n_nonzero = int(np.count_nonzero(funding_arr))
            if n_total > 0 and (n_nonzero / n_total) > 0.25:
                import logging as _logging3
                _logging3.getLogger("portfolio_backtest").warning(
                    "asset_track_from_backtest [%s]: funding_rate non-zero op "
                    "%.1f%% van de bars (%d/%d). Bybit 8h-cadence verwacht "
                    "≤ 5%% bij 5-min bars. Lijkt een ffill'd reeks → funding "
                    "wordt vele malen geboekt. Gebruik _load_per_bar_funding_rate "
                    "(sparse mapping per tick) i.p.v. merge_asof+ffill.",
                    symbol,
                    100.0 * n_nonzero / max(n_total, 1),
                    n_nonzero, n_total,
                )

    # CHIEF AUDIT 2026-05-23 (H6): wanneer fee_schedule aangeleverd: override
    # cost_bps met round-trip taker fee (one-way taker_bps * 2). Live VIP1-3
    # accounts hebben taker 3.0-4.0 bps → round-trip 6-8 bps i.p.v. 10 bps.
    effective_cost_bps = float(cost_bps)
    if fee_schedule is not None:
        effective_cost_bps = float(fee_schedule.taker_bps) * 2.0

    return AssetTrack(
        symbol=symbol,
        # Gebruik cast en pd.to_datetime in plaats van een kale pd.DatetimeIndex() call over een list
        timestamps=cast(pd.DatetimeIndex, pd.to_datetime(df_test.index)),
        signed_returns=np.asarray(bar_returns, dtype=np.float64),
        requested_leverage=np.asarray(requested_leverage, dtype=np.float64),
        side=np.asarray(sides, dtype=np.int64),
        per_asset_cap=float(per_asset_cap),
        cost_bps=effective_cost_bps,
        funding_rate=funding_arr,
        fee_schedule=fee_schedule,
    )
