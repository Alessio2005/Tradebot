"""per_side.py — Single-side (LONG or SHORT) bandit ensemble backtest.

Extracted from train_regime.py lines 2405-2739.

Depends on:
  tradebot.backtest._kernels   — calc_non_overlapping_stats (Numba)
  tradebot.labeling.trend_scanning — TrendScanningLabeler
  tradebot.execution.spread    — compute_dynamic_spread_arr, compute_annualised_sharpe
  tradebot.backtest.evaluation — compute_bar_by_bar_mtm, gap_risk_kelly_size, compute_timeout_fraction
"""

from __future__ import annotations

import logging
from typing import Any, cast

import numpy as np
import pandas as pd

from tradebot.backtest._kernels import calc_non_overlapping_stats
from tradebot.execution.spread import compute_annualised_sharpe, compute_dynamic_spread_arr
from tradebot.labeling.trend_scanning import TrendScanningLabeler

from .evaluation import compute_bar_by_bar_mtm, compute_timeout_fraction, gap_risk_kelly_size

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dtype coercion helpers (local equivalents of train_regime._f64c / _i32c)
# ---------------------------------------------------------------------------

def _f64c(arr: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(arr, dtype=np.float64)


def _i32c(arr: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(arr, dtype=np.int32)


# =============================================================================
# INTERNAL BACKTEST
# =============================================================================
def internal_backtest(
    base_model: Any,
    X1: np.ndarray,
    X4: np.ndarray,
    Xd: np.ndarray,
    df_test: pd.DataFrame,
    account_size: float = 100_000.0,
    target_risk: float = 0.01,
    max_leverage: float = 2.0,
    min_conf_long: float = 0.35,
    min_conf_short: float = 0.35,
    t_min: int = 5,
    t_max: int = 24,
    min_tstat: float = 2.0,
    event_timestamps: pd.DatetimeIndex | None = None,
    # PORTFOLIO-MARGIN-FIX (Item 8): rollend gross-exposure plafond.
    # ``None`` = backwards-compatible (geen check). Een typische waarde is
    # ``max_leverage`` zelf (single-side, géén stapeling) of een hoger
    # multi-trade plafond (bv. 1.5× max_leverage).
    max_portfolio_leverage: float | None = None,
    # FUNDING-FIX (Item 2): per-bar funding rate (decimal, niet
    # geannualiseerd). Lengte = len(df_test). ``None`` = geen funding.
    funding_rates: np.ndarray | None = None,
    # REBALANCE-COST-FIX (Item 1): éénzijdige fee+spread in bps voor MTM
    # rebalance kosten. 0 = uit (default backwards-compatible). Bybit
    # taker ≈ 4–5 bp.
    rebalance_cost_bps: float = 0.0,
    symbol: str | None = None,
    # CHIEF AUDIT 2026-05-23 (H6): optionele live-tier fee schedule.
    fee_schedule: Any | None = None,
) -> dict[str, float]:

    logger.info(
        f"--- BACKTEST (TREND SCANNING) "
        f"[L>{min_conf_long:.2f} | S>{min_conf_short:.2f}] ---"
    )

    high, low, close = df_test['high'].values, df_test['low'].values, df_test['close'].values

    # Use to_numpy() to ensure these are pure ndarrays
    high = df_test['high'].to_numpy(dtype=np.float64)
    low = df_test['low'].to_numpy(dtype=np.float64)
    close = df_test['close'].to_numpy(dtype=np.float64)

    if 'feat_vol_gk' in df_test.columns:
        atr = df_test['feat_vol_gk'].ffill().fillna(1e-5).to_numpy(dtype=np.float64)
    else:
        prev_close = np.roll(close, 1)
        # By using to_numpy() above, the '-' operator and np.maximum now work for Pylance
        tr = np.maximum(high - low, np.abs(high - prev_close))
        atr = pd.Series(tr).rolling(14).mean().ffill().fillna(1e-5).to_numpy(dtype=np.float64)

    # Dynamische Spread / Fee Logica — SQUARE-ROOT IMPACT MODEL (issue #2)
    # Voor bid/ask data zijn de kosten al in de barriers verwerkt (spread_val = 0).
    # Voor mid-price data schaalt de spread mee met σ_bar en 1/sqrt(depth).
    if 'bid_close' in df_test.columns and 'ask_close' in df_test.columns:
        spread_val = 0.0000
        spread_arr = np.full(len(close), spread_val, dtype=np.float64)
    else:
        spread_arr = compute_dynamic_spread_arr(
            df_test, fallback=0.0010, fee_schedule=fee_schedule,
        )
        # Scalar proxy voor legacy consumers (TrendScanningLabeler etc.)
        spread_val = (
            float(np.nanmedian(spread_arr)) if spread_arr.size > 0 else 0.0010
        )
        if not np.isfinite(spread_val) or spread_val <= 0.0:
            spread_val = 0.0010

    # Hybride labeler: Trend Scanning (richting) + Triple Barrier (PT/SL returns).
    # SPREAD-FIX: geef spread_val door zodat entry-prijs correct gecorrigeerd wordt
    # voor mid-price data (Bybit). _triple_barrier_per_event past curr_price ± spread/2 aan.
    _ts_labeler = TrendScanningLabeler(t_min=t_min, t_max=t_max, min_tstat=min_tstat)
    _, t1_long, ret_long = _ts_labeler.label_data(df_test, side="LONG", spread=spread_val)
    _, t1_short, ret_short = _ts_labeler.label_data(df_test, side="SHORT", spread=spread_val)

    # Targets pre-calculeren voor de Bandit Feedback Loop
    n_bars = len(df_test)
    fallback_t1 = pd.Series(np.arange(n_bars) + t_max, index=df_test.index)

    t1_long_arr = t1_long.reindex(df_test.index).fillna(fallback_t1).to_numpy(dtype=np.int32)
    t1_short_arr = t1_short.reindex(df_test.index).fillna(fallback_t1).to_numpy(dtype=np.int32)

    t1_long_arr = np.clip(t1_long_arr, 0, n_bars - 1)
    t1_short_arr = np.clip(t1_short_arr, 0, n_bars - 1)

    ret_long_arr = ret_long.reindex(df_test.index).fillna(0.0).to_numpy(dtype=np.float64)
    ret_short_arr = ret_short.reindex(df_test.index).fillna(0.0).to_numpy(dtype=np.float64)

    # Pre-allocate probability arrays
    n_samples = len(X1)
    p_long = np.zeros(n_samples, dtype=np.float64)
    p_short = np.zeros(n_samples, dtype=np.float64)

    is_long_ensemble = base_model.models[0].min_conf_short >= 0.99
    target_pct = min_conf_long if is_long_ensemble else min_conf_short

    meta_veto_count = 0
    total_base_signals = 0

    dynamic_thresholds = np.full(n_samples, 2.0, dtype=np.float64)
    valid_events_set = set(event_timestamps) if event_timestamps is not None else None

    # Opslag voor Asynchrone Bandit Updates
    bandit_updates = {}

    # --- NIEUW: State tracking voor On-Policy Learning ---
    active_trade_end_idx = -1

    for i in range(n_samples):
        # 0. CHECK PENDING BANDIT UPDATES (Zijn er trades gesloten op deze bar?)
        if hasattr(base_model, 'update_bandit') and i in bandit_updates:
            for ctx, probs_raw, rew_l, rew_s in bandit_updates[i]:
                base_model.update_bandit(ctx, probs_raw, rew_l, rew_s)
            del bandit_updates[i]  # Geheugen opschonen

        x1_row = X1[i].reshape(1, -1)
        x4_row = X4[i].reshape(1, -1) if len(X4) > 0 else None
        xd_row = Xd[i].reshape(1, -1) if len(Xd) > 0 else None

        # 1. Update de scalers ALTIJD op elke bar
        base_model.update_live_scalers(x1_row, x4_row, xd_row)

        # 2. CUSUM EVENT CHECK
        if valid_events_set is not None:
            curr_ts = df_test.index[i]
            if curr_ts not in valid_events_set:
                continue

        res = base_model.predict_greybox_strategy(
            x1_row, x4_row, xd_row, min_confidence=target_pct
        )

        if res.get("prob_win", 0.0) >= res.get("threshold_used", 1.0):
            total_base_signals += 1

        if res.get("vetoed_by_meta", False):
            meta_veto_count += 1

        # 3. BANDIT FEEDBACK SCHEDULER (ON-POLICY)
        is_valid_signal = res.get("signal") == 1 and not res.get("vetoed_by_meta", False)

        # FIX: Controleer of we fysiek ruimte hebben in het portfolio
        is_overlapping = i <= active_trade_end_idx

        if is_valid_signal and not is_overlapping:
            if hasattr(base_model, 'update_bandit') and "raw_model_probs" in res:
                side_det = res.get("side_detected")
                t1_idx = t1_long_arr[i] if side_det == "LONG" else t1_short_arr[i]

                # --- NIEUW: Blokkeer nieuwe updates tot deze specifieke trade sluit ---
                active_trade_end_idx = t1_idx

                # ASYMMETRIE-FIX: geef beide werkelijke uitkomsten mee.
                # Vroeger: reward_short = -reward_long (fout bij PT≠SL asymmetrie).
                # Nu: elke kant krijgt zijn eigen netto-rendement uit de barrière.
                #
                # CHIEF AUDIT 2026-05-23 (K2): align bandit reward met
                # daadwerkelijk geboekte PnL — trek half-spread af per leg
                # zodat de bandit dezelfde net-PnL ziet als de executie. De
                # _kernels.py kernel rekent een extra half-spread aan bovenop
                # de TBM-spread (zie H2-comment); diezelfde aftrek hier zorgt
                # dat winstgevende armen niet op kunstmatig opgeblazen rewards
                # leren — de bandit zou anders trades selecteren die in de
                # equity-curve net negatief uitvallen na kosten.
                atr_norm = atr[i] / close[i] if close[i] > 0 else 1e-4
                net_ret_long  = ret_long_arr[i]  - spread_arr[i] * 0.5
                net_ret_short = ret_short_arr[i] - spread_arr[i] * 0.5
                reward_long  = net_ret_long  / atr_norm if net_ret_long  != 0 else 0.0
                reward_short = net_ret_short / atr_norm if net_ret_short != 0 else 0.0

                safe_t1_idx = max(t1_idx, i + 1)

                if safe_t1_idx not in bandit_updates:
                    bandit_updates[safe_t1_idx] = []
                bandit_updates[safe_t1_idx].append(
                    (res["macro_context_used"], res["raw_model_probs"], reward_long, reward_short)
                )

        # Sla de kansen op voor de uiteindelijke Vectorized Non-Overlapping performance berekening
        # OPMERKING: Ghost-trades krijgen kans '0.0', dus dit matcht de On-Policy logica
        final_prob = res.get("prob_win", 0.0) if is_valid_signal else 0.0

        if res.get("side_detected") == "LONG":
            p_long[i] = final_prob
            dynamic_thresholds[i] = res.get("threshold_used", min_conf_long)
        elif res.get("side_detected") == "SHORT":
            p_short[i] = final_prob
            dynamic_thresholds[i] = res.get("threshold_used", min_conf_short)

    bandit_updates.clear()

    if total_base_signals > 0:
        veto_rate = (meta_veto_count / total_base_signals) * 100
        logger.info(
            f"   Meta-Model Vetoes: {meta_veto_count} out of "
            f"{total_base_signals} base signals ({veto_rate:.1f}% blocked)."
        )
    else:
        logger.info("   Meta-Model Vetoes: 0 (No base signals generated).")

    dummy_high_thresh = np.full(n_samples, 2.0, dtype=np.float64)
    if is_long_ensemble:
        thresh_long_arr = dynamic_thresholds
        thresh_short_arr = dummy_high_thresh
    else:
        thresh_long_arr = dummy_high_thresh
        thresh_short_arr = dynamic_thresholds

    # 4. Filter op Non-Overlapping (Vectorized Finale Check)
    # H1-FIX: vang active_indices op (derde returnwaarde) voor correcte Sharpe-timestamps.
    # NUMBA-FIX (Item 11): force float64/int32 + C-contiguous before kernel.
    active_returns_pct, _active_sides, active_indices = calc_non_overlapping_stats(
        _f64c(p_short),
        _f64c(p_long),
        _f64c(ret_short_arr),
        _f64c(ret_long_arr),
        _i32c(t1_short_arr),
        _i32c(t1_long_arr),
        _f64c(spread_arr),
        _f64c(thresh_long_arr),
        _f64c(thresh_short_arr),
    )


    if len(active_returns_pct) == 0:
        return {
            "sharpe_ratio": 0.0, "total_pnl": 0.0,
            "final_equity": account_size, "n_trades": 0,
        }

    # Equity curve simulatie met Kelly-fractionele positiegrootte.
    # LOOKAHEAD-FIX: positiegrootte nu gesized op de VERWACHTE risico (SL-afstand bij entry),
    # niet op de gerealiseerde return achteraf.
    # GAP-RISK-FIX (Item 2): standaard Kelly veronderstelt dat de SL geraakt wordt
    # op het exacte SL-niveau. In de praktijk kunnen gaps over de SL springen. We
    # stressen daarom de verwachte risico met een ATR-multiplier en een empirische
    # CVaR-correctie op basis van de tot-nu-toe gerealiseerde returns.
    _sl_width_bt = 1.0  # Matcht TrendScanningLabeler default sl_width=1.0
    equity: list[float] = [float(account_size)]
    leverages_used: list[float] = []
    realised_buffer: list[float] = []  # ringbuffer voor CVaR
    capped_count = 0

    actual_indices = active_indices[:len(active_returns_pct)]
    # internal_backtest is single-side: we construeren expliciete signed-sides
    # uit is_long_ensemble (active_sides codeert L=2/S=0/F=1, niet bruikbaar
    # als signed schaal voor MTM-PnL).
    side_signed = 1 if is_long_ensemble else -1

    for r, idx in zip(active_returns_pct, actual_indices):
        curr_eq = equity[-1]
        hist_arr = (
            np.asarray(realised_buffer[-200:], dtype=np.float64)
            if len(realised_buffer) >= 20 else None
        )
        kres = gap_risk_kelly_size(
            target_risk=float(target_risk),
            # P1.5-FIX (CHIEF AUDIT 2026-05-23): Use the PRIOR bar's ATR
            # for sizing, not the signal bar itself.  bidirectional.py:348
            # already does this correctly (atr[max(idx-1, 0)]).  On a
            # vol-spike bar (news event that triggers the CUSUM signal),
            # atr[idx] is the post-spike inflated value — using it produces
            # up to 30-50% conservative sizing relative to the live path
            # where the prior-bar ATR would be used.  Aligning both
            # backtest paths eliminates this inconsistency.
            atr_at_entry=float(atr[max(int(idx) - 1, 0)]),
            price_at_entry=float(close[idx]),
            sl_width=_sl_width_bt,
            max_leverage=float(max_leverage),
            historical_returns=hist_arr,
        )
        if kres.capped:
            capped_count += 1
        trade_size = curr_eq * kres.leverage
        pnl = trade_size * float(r)
        equity.append(curr_eq + pnl)
        leverages_used.append(kres.leverage)
        realised_buffer.append(float(r))

    final_equity = equity[-1]
    total_profit = final_equity - account_size

    # H1-FIX: gebruik de werkelijke bar-posities van uitgevoerde trades als timestamps.
    # P0-21: Use next-bar timestamps for fills (signal bar ≠ fill bar).
    # Real fill = next-bar open; shift indices by +1, clamp to last bar.
    fill_indices = np.clip(actual_indices + 1, 0, len(df_test) - 1)
    trade_ts = df_test.index[fill_indices].values
    sharpe = compute_annualised_sharpe(active_returns_pct, trade_ts)

    # ─── HANGING BARRIER MONITOR (Item 4) ─────────────────────────────────
    # Een trade die op `entry + t_max` exit geldt als horizon-timeout.
    side_t1_arr = t1_long_arr if is_long_ensemble else t1_short_arr
    timeout_stats = compute_timeout_fraction(
        event_indices=actual_indices.astype(np.int64),
        horizons=np.full(actual_indices.size, int(t_max), dtype=np.int64),
        t1_indices=side_t1_arr[actual_indices].astype(np.int64),
        n_total_bars=n_bars,
        warn_threshold=0.15,
        max_horizon=int(t_max),
    )

    # ─── BAR-BY-BAR MTM CURVE (Item 6) ────────────────────────────────────
    if actual_indices.size > 0:
        exit_idx_arr = (t1_long_arr if is_long_ensemble else t1_short_arr)[actual_indices]
        sides_signed = np.full(actual_indices.size, side_signed, dtype=np.int64)
        mtm = compute_bar_by_bar_mtm(
            close=close,
            bar_index=cast(pd.DatetimeIndex, df_test.index),
            trade_entry_idx=actual_indices.astype(np.int64),
            trade_exit_idx=exit_idx_arr.astype(np.int64),
            trade_sides=sides_signed,
            trade_leverage=np.asarray(leverages_used, dtype=np.float64),
            account_size=float(account_size),
            # PORTFOLIO-MARGIN-FIX (Item 8): geef rollend gross-exposure cap door.
            max_portfolio_leverage=max_portfolio_leverage,
            # FUNDING-FIX (Item 2) + REBALANCE-COST-FIX (Item 1)
            funding_rates=funding_rates,
            rebalance_cost_bps=rebalance_cost_bps,
        )
    else:
        mtm = None

    logger.info(
        f"Backtest Result: Equity ${final_equity:,.2f} | "
        f"Sharpe (trade): {sharpe:.2f} | "
        f"MTM Sharpe: {mtm.sharpe_annualised if mtm else 0.0:.2f} | "
        f"MTM Calmar: {mtm.calmar if mtm else 0.0:.2f} | "
        f"MTM MaxDD: {mtm.max_drawdown if mtm else 0.0:.2%} | "
        f"Trades: {len(active_returns_pct)} | "
        f"Leverage capped: {capped_count} | "
        f"Timeouts: {timeout_stats['timeout_fraction']:.1%}"
    )

    return {
        "sharpe_ratio": float(sharpe),
        "total_pnl": float(total_profit),
        "final_equity": float(final_equity),
        "n_trades": len(active_returns_pct),
        "meta_vetoes": meta_veto_count,
        "leverage_capped_count": int(capped_count),
        "timeout_fraction": float(timeout_stats["timeout_fraction"]),
        "mtm_sharpe": float(mtm.sharpe_annualised) if mtm else 0.0,
        "mtm_calmar": float(mtm.calmar) if mtm else 0.0,
        "mtm_max_drawdown": float(mtm.max_drawdown) if mtm else 0.0,
        "mtm_intra_dd_p95": float(mtm.intra_drawdown_p95) if mtm else 0.0,
    }


# Backward-compat alias (test_imports.py public API)
run_per_side_backtest = internal_backtest
