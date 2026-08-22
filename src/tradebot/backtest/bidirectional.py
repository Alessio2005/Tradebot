"""bidirectional.py — Simultaneous LONG + SHORT bandit ensemble backtest.

Extracted from train_regime.py lines 2741-3158.
"""

from __future__ import annotations

import logging
from typing import Any, cast

import numpy as np
import pandas as pd

from tradebot.backtest._kernels import calc_non_overlapping_stats
from tradebot.execution.spread import compute_annualised_sharpe, compute_dynamic_spread_arr
from tradebot.labeling.trend_scanning import TrendScanningLabeler

from ..labeling.meta import CausalScout, MetaLabelFilter
from .evaluation import (
    compute_bar_by_bar_mtm,
    compute_timeout_fraction,
    gap_risk_kelly_size,
    resolve_long_short_collision,
)

logger = logging.getLogger(__name__)


def _f64c(arr: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(arr, dtype=np.float64)


def _i32c(arr: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(arr, dtype=np.int32)


# =============================================================================
# BIDIRECTIONELE BACKTEST — LONG + SHORT BANDIT ENSEMBLE TEGELIJK ACTIEF
# =============================================================================
def bidirectional_backtest(
    long_ensemble: Any,
    X1_long: np.ndarray,
    X4_long: np.ndarray,
    Xd_long: np.ndarray,
    short_ensemble: Any,
    X1_short: np.ndarray,
    X4_short: np.ndarray,
    Xd_short: np.ndarray,
    df_test: pd.DataFrame,
    # PORTFOLIO-MARGIN-FIX (Item 8): zie internal_backtest.
    max_portfolio_leverage: float | None = None,
    # FUNDING-FIX (Item 2) + REBALANCE-COST-FIX (Item 1) — zie internal_backtest.
    funding_rates: np.ndarray | None = None,
    rebalance_cost_bps: float = 0.0,
    symbol: str | None = None,
    account_size: float = 100_000.0,
    target_risk: float = 0.01,
    max_leverage: float = 2.0,
    min_conf_long: float = 0.35,
    min_conf_short: float = 0.35,
    t_min: int = 5,
    t_max: int = 24,
    min_tstat: float = 2.0,
    event_timestamps: pd.DatetimeIndex | None = None,
    max_uncertainty: float = 0.30,
    # ── Meta-Labeling (optioneel) ────────────────────────────────────
    use_meta_labeling: bool = False,
    scout_min_tstat: float = 1.5,
    scout_cusum_multiplier: float = 0.5,
    meta_prob_threshold: float = 0.35,
    # CHIEF AUDIT 2026-05-23 (H6): optionele live-tier fee schedule.
    fee_schedule: Any | None = None,
) -> dict[str, float | int | np.ndarray | pd.Index]:
    """Draait LONG- en SHORT-bandit ensemble bar-voor-bar tegelijk.

    In tegenstelling tot ``internal_backtest`` worden beide kanten in één loop
    aangeboden.  Op elke bar kiezen we de kant met de hoogste
    confidence-ratio (prob_win / threshold). Vervolgens bepaalt
    ``calc_non_overlapping_stats`` de definitieve trade-sequentie, net als
    bij de single-side variant — maar nu kan elke trade LONG of SHORT zijn.
    """
    logger.info(
        "--- BIDIRECTIONEEL BACKTEST (LONG + SHORT Bandit Ensemble tegelijk) ---"
    )

    close = df_test["close"].to_numpy(dtype=np.float64)
    high  = df_test["high"].to_numpy(dtype=np.float64)
    low   = df_test["low"].to_numpy(dtype=np.float64)

    if "feat_vol_gk" in df_test.columns:
        atr = df_test["feat_vol_gk"].ffill().fillna(1e-5).to_numpy(dtype=np.float64)
    else:
        prev_close = np.roll(close, 1)
        tr  = np.maximum(high - low, np.abs(high - prev_close))
        atr = pd.Series(tr).rolling(14).mean().ffill().fillna(1e-5).to_numpy(dtype=np.float64)

    # Dynamische Spread / Fee Logica (bidirectioneel pad) — square-root impact.
    if "bid_close" in df_test.columns and "ask_close" in df_test.columns:
        spread_val = 0.0000
        spread_arr = np.full(len(close), spread_val, dtype=np.float64)
    else:
        spread_arr = compute_dynamic_spread_arr(
            df_test, fallback=0.0010, fee_schedule=fee_schedule,
        )
        spread_val = (
            float(np.nanmedian(spread_arr)) if spread_arr.size > 0 else 0.0010
        )
        if not np.isfinite(spread_val) or spread_val <= 0.0:
            spread_val = 0.0010

    # Trend Scanning labels voor BEIDE kanten.
    # SPREAD-FIX: spread_val doorgeven voor correcte entry-prijs geometrie.
    # AUDIT B-2 FIX: pass event_timestamps (CUSUM events) to avoid labelling every bar.
    _tsl = TrendScanningLabeler(t_min=t_min, t_max=t_max, min_tstat=min_tstat)
    # CHIEF AUDIT 2026-05-23 (P1.1): use_fixed_horizon=True removes the oracle
    # exit bias.  The old default (False) set the Triple Barrier timeout to
    # TrendScan best_span — a forward-looking OLS span that "knows" how long
    # the profitable trend lasts.  With fixed horizon = t_max bars, the timeout
    # is causal; PT and SL barriers still trigger on actual future price paths
    # (which is correct — Triple Barrier inherently uses future prices for exits,
    # but the horizon itself must not be oracle-selected).
    _, t1_long_s,  ret_long_s  = _tsl.label_data(
        df_test, event_timestamps=event_timestamps, side="LONG",
        spread=spread_val, use_fixed_horizon=True,
    )
    _, t1_short_s, ret_short_s = _tsl.label_data(
        df_test, event_timestamps=event_timestamps, side="SHORT",
        spread=spread_val, use_fixed_horizon=True,
    )

    n_bars     = len(df_test)
    fallback   = pd.Series(np.arange(n_bars) + t_max, index=df_test.index)
    t1_l_arr   = np.clip(t1_long_s.reindex(df_test.index).fillna(fallback).to_numpy(dtype=np.int32),  0, n_bars - 1)
    t1_s_arr   = np.clip(t1_short_s.reindex(df_test.index).fillna(fallback).to_numpy(dtype=np.int32), 0, n_bars - 1)
    ret_l_arr  = ret_long_s.reindex(df_test.index).fillna(0.0).to_numpy(dtype=np.float64)
    ret_s_arr  = ret_short_s.reindex(df_test.index).fillna(0.0).to_numpy(dtype=np.float64)

    # ── META-LABELING: pre-compute Scout-signalen ────────────────────────────
    # De Scout bepaalt RICHTING (Long/Short/Flat) per bar met permissieve drempel.
    # De Judge (long_ensemble / short_ensemble) filtert daarna op prob_win.
    # MetaLabelFilter.batch_decide() combineert beide tot de finale beslissing.
    scout_dirs_full: np.ndarray = np.zeros(n_bars, dtype=np.int8)
    meta_filter: MetaLabelFilter | None = None

    if use_meta_labeling:
        logger.info(
            "Bidirectioneel backtest: Meta-Labeling ACTIEF "
            "(scout_min_tstat=%.2f, meta_prob_threshold=%.2f).",
            scout_min_tstat, meta_prob_threshold,
        )
        # LOOKAHEAD-FIX: PrimaryScout gebruikte TrendScanning (forward OLS over toekomstige
        # bars) om richting te bepalen. In backtest betekent dit dat de Scout "weet" wat
        # de markt de komende t_max bars gaat doen — lookahead bias.
        # CausalScout bepaalt richting via EMA-crossover (volledig backward-looking).
        # Triple Barrier uitkomsten (realized_ret, t1_indices) gebruiken wel toekomstige
        # bars — maar dit zijn de werkelijke marktuitkomsten, geen signaal.
        _scout = CausalScout(
            fast_ema=10,
            slow_ema=50,
            cusum_multiplier=scout_cusum_multiplier,
            t_horizon=t_max,
            pt_width=2.0,
            sl_width=1.0,
        )
        _scout_result = _scout.generate(df_test)
        # Zet event-gebaseerde richting om naar een per-bar array
        for _k, _ev_i in enumerate(_scout_result.event_indices):
            if 0 <= _ev_i < n_bars:
                scout_dirs_full[_ev_i] = _scout_result.directions[_k]
        meta_filter = MetaLabelFilter(prob_threshold=meta_prob_threshold)
        logger.info(
            "Scout gegenereerd: %d events op %d bars.",
            len(_scout_result.event_indices), n_bars,
        )
    # ────────────────────────────────────────────────────────────────────────

    n_samples             = len(X1_long)
    p_long                = np.zeros(n_samples, dtype=np.float64)
    p_short               = np.zeros(n_samples, dtype=np.float64)
    dyn_thresh_long       = np.full(n_samples, 2.0, dtype=np.float64)
    dyn_thresh_short      = np.full(n_samples, 2.0, dtype=np.float64)

    valid_events_set: set | None = set(event_timestamps) if event_timestamps is not None else None

    # Asynchrone bandit-updates per kant
    bu_long:  dict[int, list[tuple[Any, Any, float, float]]] = {}
    bu_short: dict[int, list[tuple[Any, Any, float, float]]] = {}

    active_trade_end_idx = -1

    for i in range(n_samples):
        # --- Verwerk openstaande bandit-updates ---
        if hasattr(long_ensemble, "update_bandit") and i in bu_long:
            for ctx, prb, rw_l, rw_s in bu_long[i]:
                long_ensemble.update_bandit(ctx, prb, rw_l, rw_s)
            del bu_long[i]
        if hasattr(short_ensemble, "update_bandit") and i in bu_short:
            for ctx, prb, rw_l, rw_s in bu_short[i]:
                short_ensemble.update_bandit(ctx, prb, rw_l, rw_s)
            del bu_short[i]

        # Feature-rijen per ensemble
        x1l = X1_long[i].reshape(1, -1)
        x4l = X4_long[i].reshape(1, -1)  if X4_long.shape[1]  > 0 else None
        xdl = Xd_long[i].reshape(1, -1)  if Xd_long.shape[1]  > 0 else None

        x1s = X1_short[i].reshape(1, -1)
        x4s = X4_short[i].reshape(1, -1) if X4_short.shape[1] > 0 else None
        xds = Xd_short[i].reshape(1, -1) if Xd_short.shape[1] > 0 else None

        # Scalers bijwerken voor beide ensembles
        long_ensemble.update_live_scalers(x1l, x4l, xdl)
        short_ensemble.update_live_scalers(x1s, x4s, xds)

        # CUSUM event-filter
        if valid_events_set is not None:
            if df_test.index[i] not in valid_events_set:
                continue

        # Voorspellingen opvragen
        res_l = long_ensemble.predict_greybox_strategy(
            x1l, x4l, xdl, min_confidence=min_conf_long, max_uncertainty=max_uncertainty
        )
        res_s = short_ensemble.predict_greybox_strategy(
            x1s, x4s, xds, min_confidence=min_conf_short, max_uncertainty=max_uncertainty
        )

        is_overlap = i <= active_trade_end_idx

        # ── META-LABELING FILTER ─────────────────────────────────────────────
        # Als meta-labeling actief is, checkt MetaLabelFilter EERST of de Scout
        # een signaal geeft; pas daarna beoordeelt de Judge of het de moeite waard is.
        # Bij use_meta_labeling=False valt de logica terug op het normale gedrag.
        if meta_filter is not None:
            # Scout-richting voor deze bar
            scout_dir = int(scout_dirs_full[i]) if i < len(scout_dirs_full) else 0

            # Judge-kansen (prob_win uit beide ensembles)
            pw_long  = float(res_l.get("prob_win", 0.0))
            pw_short = float(res_s.get("prob_win", 0.0))

            # MetaLabelFilter beslist: 1=Long OK, -1=Short OK, 0=geblokkeerd
            meta_decision_long  = meta_filter.decide(
                1 if scout_dir == 1 else 0, pw_long,
                judge_side_detected=res_l.get("side_detected"),
            )
            meta_decision_short = meta_filter.decide(
                1 if scout_dir == -1 else 0, pw_short,
                judge_side_detected=res_s.get("side_detected"),
            )
            long_active  = meta_decision_long  == 1 and not is_overlap
            short_active = meta_decision_short == 1 and not is_overlap
        else:
            # Origineel gedrag: Judge bepaalt alles (geen Scout-richtingcheck)
            long_active  = res_l.get("signal") == 1 and not is_overlap
            short_active = res_s.get("signal") == 1 and not is_overlap
        # ────────────────────────────────────────────────────────────────────

        if long_active or short_active:
            # COLLISION RESOLUTION (Item 5):
            # Vervangt het binary "winner-takes-all" gedrag door een delta-
            # neutrale netting wanneer beide signalen actief zijn. De resolver
            # weegt prob/threshold ratios en kiest een effectieve horizon =
            # min(h_l, h_s) bij dubbele activatie (duration_neutral=True).
            resolution = resolve_long_short_collision(
                long_active=bool(long_active),
                short_active=bool(short_active),
                prob_long=float(res_l.get("prob_win", 0.0)),
                prob_short=float(res_s.get("prob_win", 0.0)),
                threshold_long=float(res_l.get("threshold_used", min_conf_long)),
                threshold_short=float(res_s.get("threshold_used", min_conf_short)),
                horizon_long=int(t1_l_arr[i] - i),
                horizon_short=int(t1_s_arr[i] - i),
                ret_long=float(ret_l_arr[i]),
                ret_short=float(ret_s_arr[i]),
                duration_neutral=True,
            )

            if resolution.side == "FLAT":
                pass
            elif resolution.side == "NET":
                # Delta-neutrale dual-leg trade: we nemen geen netto direction,
                # maar registreren wel een trade met combined-return zodat de
                # P&L-curve dit segment meeneemt. Vermijd hier bandit-update
                # om geen tegengestelde gradiënten in te schieten.
                horizon_bars = max(int(resolution.horizon), 1)
                active_trade_end_idx = i + horizon_bars
                # Geen p_long/p_short marker — non-overlapping kernel slaat hem
                # over. Alleen state-tracking voor overlap-bescherming.
            elif resolution.net_weight > 0.0:
                # NET LONG (eventueel met short-leg gedempt)
                p_long[i]          = float(res_l.get("prob_win", 0.0))
                dyn_thresh_long[i] = float(res_l.get("threshold_used", min_conf_long))
                horizon_bars = max(int(resolution.horizon), 1)
                active_trade_end_idx = i + horizon_bars

                if hasattr(long_ensemble, "update_bandit") and "raw_model_probs" in res_l:
                    # CHIEF AUDIT 2026-05-23 (K2): align bandit reward met
                    # daadwerkelijk geboekte PnL — trek half-spread af per leg
                    # zodat de bandit dezelfde net-PnL ziet als de executie.
                    # Hoewel TBM al de FULL round-trip spread aftrekt voor de
                    # PnL-curve, gebruikt _kernels.py een EXTRA half-spread
                    # (zie H2-comment in _kernels.py). Voor de bandit-reward
                    # houden we hier eenzelfde half-spread aftrek per leg aan
                    # zodat de bandit niet leert op een ander netto dan de
                    # backtest-equity-curve.
                    atr_norm = atr[i] / close[i] if close[i] > 0 else 1e-4
                    net_ret_l = ret_l_arr[i] - spread_arr[i] * 0.5
                    net_ret_s = ret_s_arr[i] - spread_arr[i] * 0.5
                    rw_l = net_ret_l / atr_norm if net_ret_l != 0 else 0.0
                    rw_s = net_ret_s / atr_norm if net_ret_s != 0 else 0.0
                    safe_t1 = max(int(t1_l_arr[i]), i + 1)
                    bu_long.setdefault(safe_t1, []).append(
                        (res_l["macro_context_used"], res_l["raw_model_probs"], rw_l, rw_s)
                    )
            else:
                # NET SHORT
                p_short[i]          = float(res_s.get("prob_win", 0.0))
                dyn_thresh_short[i] = float(res_s.get("threshold_used", min_conf_short))
                horizon_bars = max(int(resolution.horizon), 1)
                active_trade_end_idx = i + horizon_bars

                if hasattr(short_ensemble, "update_bandit") and "raw_model_probs" in res_s:
                    # CHIEF AUDIT 2026-05-23 (K2): align bandit reward met
                    # daadwerkelijk geboekte PnL — half-spread per leg.
                    atr_norm = atr[i] / close[i] if close[i] > 0 else 1e-4
                    net_ret_l = ret_l_arr[i] - spread_arr[i] * 0.5
                    net_ret_s = ret_s_arr[i] - spread_arr[i] * 0.5
                    rw_l = net_ret_l / atr_norm if net_ret_l != 0 else 0.0
                    rw_s = net_ret_s / atr_norm if net_ret_s != 0 else 0.0
                    safe_t1 = max(int(t1_s_arr[i]), i + 1)
                    bu_short.setdefault(safe_t1, []).append(
                        (res_s["macro_context_used"], res_s["raw_model_probs"], rw_l, rw_s)
                    )

    bu_long.clear()
    bu_short.clear()

    # H1-FIX: vang active_indices op voor correcte Sharpe-timestamps (zie ook internal_backtest).
    # NUMBA-FIX (Item 11): force float64/int32 + C-contiguous before kernel.
    active_returns_pct, active_sides, active_indices = calc_non_overlapping_stats(
        _f64c(p_short), _f64c(p_long),
        _f64c(ret_s_arr), _f64c(ret_l_arr),
        _i32c(t1_s_arr), _i32c(t1_l_arr),
        _f64c(spread_arr),
        _f64c(dyn_thresh_long),
        _f64c(dyn_thresh_short),
    )

    if len(active_returns_pct) == 0:
        logger.info("BIDIR BACKTEST | Geen trades gegenereerd.")
        return {
            "sharpe_ratio": 0.0, "total_pnl": 0.0,
            "final_equity": account_size, "n_trades": 0,
        }

    # Equity-curve simulatie met Kelly-fractionele positiegrootte.
    # GAP-RISK-FIX (Item 2): zie ``gap_risk_kelly_size`` voor de uitbreiding.
    _sl_width_bt = 1.0  # Matcht CausalScout / TrendScanningLabeler default sl_width=1.0
    equity: list[float] = [float(account_size)]
    leverages_used_bd: list[float] = []
    realised_buffer_bd: list[float] = []
    capped_count_bd = 0

    actual_indices_bd = active_indices[:len(active_returns_pct)]
    for r, idx in zip(active_returns_pct, actual_indices_bd):
        curr_eq = equity[-1]
        hist_arr = (
            np.asarray(realised_buffer_bd[-200:], dtype=np.float64)
            if len(realised_buffer_bd) >= 20 else None
        )
        kres = gap_risk_kelly_size(
            target_risk=float(target_risk),
            # P0-C FIX: gebruik atr[idx-1] — GK-vol op de signal-bar zelf is pas
            # bekend na de close van die bar; sizing moet gebaseerd zijn op de
            # vorige bar's vol (strikt-causaal, info-before-close conventie).
            atr_at_entry=float(atr[max(int(idx) - 1, 0)]),
            price_at_entry=float(close[idx]),
            sl_width=_sl_width_bt,
            max_leverage=float(max_leverage),
            historical_returns=hist_arr,
        )
        if kres.capped:
            capped_count_bd += 1
        trade_size = curr_eq * kres.leverage
        equity.append(curr_eq + trade_size * float(r))
        leverages_used_bd.append(kres.leverage)
        realised_buffer_bd.append(float(r))

    final_equity = equity[-1]
    total_profit = final_equity - account_size
    # P0-21: Use next-bar timestamps for fills (signal bar ≠ fill bar).
    # Real fill = next-bar open; shift indices by +1, clamp to last bar.
    fill_indices_bd = np.clip(actual_indices_bd + 1, 0, len(df_test) - 1)
    trade_ts        = df_test.index[fill_indices_bd].values
    sharpe          = compute_annualised_sharpe(active_returns_pct, trade_ts)

    n_long  = int(np.sum(active_sides == 2))
    n_short = int(np.sum(active_sides == 0))

    # ─── HANGING BARRIER MONITOR (Item 4) ─────────────────────────────────
    # Selecteer per trade het juiste t1-array (long vs short side-codering 2/0).
    sides_codes_bd = active_sides[:len(active_returns_pct)]
    t1_per_trade = np.where(
        sides_codes_bd == 2,
        t1_l_arr[actual_indices_bd],
        t1_s_arr[actual_indices_bd],
    ).astype(np.int64)
    timeout_stats_bd = compute_timeout_fraction(
        event_indices=actual_indices_bd.astype(np.int64),
        horizons=np.full(actual_indices_bd.size, int(t_max), dtype=np.int64),
        t1_indices=t1_per_trade,
        n_total_bars=n_bars,
        warn_threshold=0.15,
        max_horizon=int(t_max),
    )

    # ─── BAR-BY-BAR MTM CURVE (Item 6) ────────────────────────────────────
    if actual_indices_bd.size > 0:
        sides_signed_bd = np.where(
            sides_codes_bd == 2, 1, -1
        ).astype(np.int64)
        mtm_bd = compute_bar_by_bar_mtm(
            close=close,
            bar_index=cast(pd.DatetimeIndex, df_test.index),
            trade_entry_idx=actual_indices_bd.astype(np.int64),
            trade_exit_idx=t1_per_trade,
            trade_sides=sides_signed_bd,
            trade_leverage=np.asarray(leverages_used_bd, dtype=np.float64),
            account_size=float(account_size),
            # PORTFOLIO-MARGIN-FIX (Item 8): bidirectionele Long+Short paths
            # kunnen MTM-overlap vertonen — cap nu actief.
            max_portfolio_leverage=max_portfolio_leverage,
            # FUNDING-FIX (Item 2) + REBALANCE-COST-FIX (Item 1)
            funding_rates=funding_rates,
            rebalance_cost_bps=rebalance_cost_bps,
        )
    else:
        mtm_bd = None

    logger.info(
        f"BIDIR BACKTEST | Trade-Sharpe: {sharpe:.2f} | "
        f"MTM Sharpe: {mtm_bd.sharpe_annualised if mtm_bd else 0.0:.2f} | "
        f"MTM Calmar: {mtm_bd.calmar if mtm_bd else 0.0:.2f} | "
        f"MTM MaxDD: {mtm_bd.max_drawdown if mtm_bd else 0.0:.2%} | "
        f"Trades: {len(active_returns_pct)} (LONG: {n_long}, SHORT: {n_short}) | "
        f"Lev capped: {capped_count_bd} | "
        f"Timeouts: {timeout_stats_bd['timeout_fraction']:.1%} | "
        f"PnL: ${total_profit:,.2f}"
    )

    # ─── PER-BAR TRACKS VOOR PORTFOLIO-LAAG (MARK-TO-MARKET) ─────────────
    # Spread de positie over de VOLLEDIGE hold-periode [entry, t1] zodat de
    # portfolio-risicolaag de echte intra-trade vol en correlatie ziet.
    #
    # bar_returns_arr[b] = price_ret_simple[b] × side  (unleveraged, PnL-
    # gesigneerd: positief = winst ongeacht richting). De portfolio-laag past
    # requested_leverage toe. Entry-/exit-kosten worden correct geboekt via
    # cost_bps op de delta van signed-exposure bij de transitiebars.
    #
    # Waarom dit correct is:
    #   Oud: cumulatieve trade-return op 1 bar → vol-target zag vrijwel
    #        vlakke reeks → leverage dalend naar ~0.001× → Sharpe inflated.
    #   Nu:  elke bar in de hold-periode heeft side ≠ 0 → risk-manager ziet
    #        realistische volatiliteit, correlaties en DD-breaker werken
    #        gedurende de volledige open positie.
    bar_returns_arr  = np.zeros(n_bars, dtype=np.float64)
    bar_sides_arr    = np.zeros(n_bars, dtype=np.int64)
    bar_leverage_arr = np.zeros(n_bars, dtype=np.float64)
    if actual_indices_bd.size > 0:
        # Close-to-close log-returns → eenvoudige returns voor MTM-bars.
        # log_ret_arr[b] = log(close[b] / close[b-1]); bar 0 = 0.
        safe_close_arr = np.where(close > 0.0, close, np.nan)
        log_ret_arr = np.zeros(n_bars, dtype=np.float64)
        log_ret_arr[1:] = np.log(close[1:] / safe_close_arr[:-1])
        log_ret_arr = np.nan_to_num(log_ret_arr, nan=0.0, posinf=0.0, neginf=0.0)
        price_ret_simple = np.expm1(log_ret_arr)  # log → simple return

        for idx, side, lev, t1_idx in zip(
            actual_indices_bd,
            sides_signed_bd,          # reuse: computed above for mtm_bd
            leverages_used_bd,
            t1_per_trade.tolist(),
        ):
            # P0-B FIX: entry = signal_bar + 1 (fill op open van bar na signaal).
            # price_ret_simple[b] = close[b]/close[b-1] - 1.
            # Bij entry=idx crediteert de positie close[idx]/close[idx-1] — de
            # move INTO de signal-bar die in live niet vangbaar is (fill is open[idx+1]).
            # Met entry=idx+1 begint de credit bij close[idx+1]/close[idx] ✓.
            entry = int(idx) + 1
            exit_ = min(int(t1_idx), n_bars - 1)
            if entry <= exit_:
                sl = slice(entry, exit_ + 1)
                bar_returns_arr[sl]  = price_ret_simple[sl] * float(side)
                bar_sides_arr[sl]    = int(side)
                bar_leverage_arr[sl] = float(lev)

    return {
        "sharpe_ratio": float(sharpe),
        "total_pnl":    float(total_profit),
        "final_equity": float(final_equity),
        "n_trades":     len(active_returns_pct),
        "n_long":       n_long,
        "n_short":      n_short,
        "leverage_capped_count": int(capped_count_bd),
        "timeout_fraction": float(timeout_stats_bd["timeout_fraction"]),
        "mtm_sharpe": float(mtm_bd.sharpe_annualised) if mtm_bd else 0.0,
        "mtm_calmar": float(mtm_bd.calmar) if mtm_bd else 0.0,
        "mtm_max_drawdown": float(mtm_bd.max_drawdown) if mtm_bd else 0.0,
        "mtm_intra_dd_p95": float(mtm_bd.intra_drawdown_p95) if mtm_bd else 0.0,
        # Per-bar tracks voor portfolio aggregation (multi-asset).
        "bar_returns": bar_returns_arr,
        "bar_sides":   bar_sides_arr,
        "bar_leverage": bar_leverage_arr,
        "df_test_index": df_test.index,
    }
