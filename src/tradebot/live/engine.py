# src/tradebot/live/engine.py
"""AsyncIO live trading engine — orchestrates all live components.

Event-flow per bar-close (target: < 500ms total latency):
  Bar(t) received
    → FeatureUpdater.update(bar_t)        [max 50ms]
    → SignalRunner.predict(features_t)    [max 100ms]
    → PortfolioController.optimise(...)   [max 200ms]
    → ExecutionController.size_orders(...)
    → OMS.place_orders(...)
    → AuditLog.record(...)
    → CircuitBreaker.check(...)

On HALT: OMS.close_all() → engine.stop() → CRITICAL alert.

The engine supports both paper mode (replay) and live mode (Bybit WS).
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import signal as _signal
import threading as _threading
import time
from pathlib import Path

import pandas as pd

from ..execution.fees import FeeSchedule, bybit_perp_schedule
from ..monitoring.metrics import EngineMetrics, start_metrics_server
from ..oms.audit_log import AuditLog
from ..oms.paper_oms import PaperOMS
from ..oms.router import OrderRouter
from ..portfolio.constraints import PortfolioConstraints
from ..risk.daily_loss_governor import (
    GovernorAction,
    PropfirmGovernor,
    PropfirmLimits,
    RegimeConfig,
)
from ..schemas.config import RiskConfig, load_config
from .circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from .cusum_filter import CUSUMFilter, build_cusum_filters
from .execution_controller import ExecutionController, ExecutionControllerConfig
from .feature_updater import FeatureUpdater, FeatureUpdaterConfig
from .feed import BarEvent, Feed
from .judge_gate import JudgeGate, build_judge_gates
from .portfolio_controller import PortfolioController, PortfolioControllerConfig
from .signal_runner import SignalRunner, SignalRunnerConfig
from .state import EngineMode, SystemState

logger = logging.getLogger(__name__)

__all__ = ["LiveEngineConfig", "LiveEngine"]


class LiveEngineConfig:
    """Top-level configuration for the LiveEngine.

    Parameters
    ----------
    symbols :
        List of trading pairs to trade.
    initial_equity :
        Starting NAV in USDT.
    mode :
        ``"paper"`` or ``"live"``.
    interval :
        Bar interval (e.g. ``"1h"``).
    audit_log_path :
        Path for the JSONL audit file.
    cb_config :
        CircuitBreakerConfig.  Defaults to conservative thresholds.
    ec_config :
        ExecutionControllerConfig.
    pc_config :
        PortfolioControllerConfig.
    sr_config :
        SignalRunnerConfig.
    fu_config :
        FeatureUpdaterConfig.
    metrics_port :
        Port for the Prometheus /metrics HTTP endpoint (0 = disabled).
    """

    def __init__(
        self,
        symbols: list[str],
        initial_equity: float = 100_000.0,
        mode: str = "paper",
        interval: str = "1h",
        audit_log_path: str = "artefacts/audit/audit.jsonl",
        cb_config: CircuitBreakerConfig | None = None,
        ec_config: ExecutionControllerConfig | None = None,
        pc_config: PortfolioControllerConfig | None = None,
        sr_config: SignalRunnerConfig | None = None,
        fu_config: FeatureUpdaterConfig | None = None,
        metrics_port: int = 8000,
        fee_schedule: FeeSchedule | None = None,
        judge_dir: Path | None = None,
        state_out_dir: Path | None = None,
        state_write_every_n: int = 50,
        propfirm_limits: PropfirmLimits | None = None,
        regime: RegimeConfig | None = None,
    ) -> None:
        self.symbols = symbols
        self.initial_equity = initial_equity
        self.mode = EngineMode(mode)
        self.interval = interval
        self.audit_log_path = audit_log_path
        # PHASE 7/8 STAGE D (C1/C3/C4): de soevereine policy wordt EEN keer
        # geladen en door alle drie de controllers gebruikt. Tot Stage D droeg
        # elk van hen zijn eigen kopie van de drempels -- `CircuitBreakerConfig`
        # had `max_drawdown_pct = 0.08` in code staan, gelijk aan `conf/` en
        # zonder enig mechanisme dat ze gelijk hield -- en bouwde
        # `ExecutionControllerConfig()` zich op zonder limiet, want
        # `_max_gross_notional` bleef `float("inf")` omdat niets in de boom hem
        # ooit zette. Zie `reports/phase7_divergence_map.md` §4.
        risk_policy = load_config(
            Path(__file__).resolve().parents[3]
            / "conf" / "risk" / "default.yaml",
            RiskConfig,
        )
        self.cb_config = cb_config or CircuitBreakerConfig.from_risk_config(
            risk_policy)
        self.ec_config = ec_config or ExecutionControllerConfig(risk=risk_policy)
        # PHASE 5: `PortfolioControllerConfig()` had een impliciete
        # concentratielimiet (`max_weight=0.40`) die NIET uit `conf/risk/` kwam
        # (wiring audit C5). De fallback bouwt hem nu uit de soevereine policy,
        # zodat L13 dezelfde drempel gebruikt als de backtest en meebeweegt
        # wanneer die drempel verandert.
        #
        # Dit is de contractgrens die fase-opdracht paragraaf 4.3 in Phase 5
        # vastlegt; de volledige `live/`-migratie blijft Phase 7.
        # PHASE 7/8 STAGE C-1: `method` had geen expliciete waarde en viel dus
        # terug op de oude default "hrp" - een research-only allocator in het
        # live-pad (fase-6 no-go 15). De fallback kiest nu ERC, de toegelaten
        # risk-parity-allocator die bij een mislukte solve terugvalt op inverse
        # volatility.
        self.pc_config = pc_config or PortfolioControllerConfig(
            method="erc",
            constraints=PortfolioConstraints.from_risk_config(risk_policy),
        )
        self.sr_config = sr_config or SignalRunnerConfig()
        self.fu_config = fu_config or FeatureUpdaterConfig()
        self.metrics_port = metrics_port
        self.judge_dir = judge_dir
        self.state_out_dir = state_out_dir or Path("artefacts/paper_trade")
        self.state_write_every_n = state_write_every_n
        # Propfirm governor (audit §3): daily-loss + static-DD enforcement.
        # When None the engine behaves exactly as before (no propfirm gating).
        self.propfirm_limits = propfirm_limits
        self.regime = regime or RegimeConfig.funded()
        # Per-symbol CUSUM multipliers (mirror conf_config.yaml)
        self.cusum_multipliers: dict[str, float] = {}
        # Rec 4 (Sim-to-Reality #19): account-tier fee schedule.
        # Reads TRADEBOT_BYBIT_TIER + TRADEBOT_FEE_DISCOUNT from env if not
        # provided explicitly.  (Bybit linear perpetuals.)
        if fee_schedule is not None:
            self.fee_schedule: FeeSchedule | None = fee_schedule
        else:
            import os
            tier = os.environ.get("TRADEBOT_BYBIT_TIER", "NONVIP")
            disc = float(os.environ.get("TRADEBOT_FEE_DISCOUNT", "0.0") or 0.0)
            self.fee_schedule = bybit_perp_schedule(tier=tier, fee_discount_pct=disc)


class LiveEngine:
    """AsyncIO event-driven live trading engine.

    Parameters
    ----------
    config :
        Engine configuration.
    signal_runner :
        Configured SignalRunner with all alpha signals loaded.
    feed :
        Market data Feed (paper-replay or live WS).
    """

    def __init__(
        self,
        config: LiveEngineConfig,
        signal_runner: SignalRunner,
        feed: Feed,
    ) -> None:
        self._cfg = config
        self._state = SystemState(
            mode=config.mode,
            equity=config.initial_equity,
            equity_peak=config.initial_equity,
            daily_pnl_open=config.initial_equity,
            # L-3 FIX (CHIEF AUDIT 2026-05-28): seed the intraday peak to the
            # real starting equity too.  It previously defaulted to 100k while
            # the account was 200k, so the first marked equity defined the
            # session peak inconsistently with the lifetime peak.
            intraday_peak_equity=config.initial_equity,
        )

        # Sub-components
        self._audit = AuditLog(config.audit_log_path)
        self._paper_oms = PaperOMS(
            initial_equity=config.initial_equity,
            audit_log=self._audit,
            fee_schedule=config.fee_schedule,
        )
        self._router = OrderRouter(
            paper_oms=self._paper_oms,
            live_mode=(config.mode == EngineMode.LIVE),
        )
        self._cb = CircuitBreaker(config.cb_config, self._state)
        # Propfirm governor (audit §3) — only active when limits are configured.
        self._governor: PropfirmGovernor | None = (
            PropfirmGovernor(config.propfirm_limits)
            if config.propfirm_limits is not None else None
        )
        if self._governor is not None:
            self._governor.start_day(config.initial_equity)
            logger.info(
                "Propfirm governor active: daily soft/hard=%.1f%%/%.1f%%, "
                "static-DD trip/resume=%.1f%%/%.1f%%, regime=%s.",
                config.propfirm_limits.daily_soft * 100,
                config.propfirm_limits.daily_hard * 100,
                config.propfirm_limits.dd_trigger * 100,
                config.propfirm_limits.dd_resume * 100,
                config.regime.regime.value,
            )
        self._fu = FeatureUpdater(config.fu_config)
        self._sr = signal_runner
        self._pc = PortfolioController(config.pc_config, config.symbols)
        self._ec = ExecutionController(config.ec_config)
        self._feed = feed

        # Queue for feed events
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
        self._feed._queue = self._queue

        # Metrics
        self._bars_processed: int = 0
        self._latencies: list[float] = []
        self._stop_event = asyncio.Event()
        self._metrics = EngineMetrics()

        # Wave 15 P0-5.1 — kill switch state
        self._kill_flag = _threading.Event()

        # Wave 15 P0-5.6 — feed sequence tracking
        self._last_bar_seq: dict[str, int] = {}

        # JudgeGate — loaded from judge_dir if provided
        # P0-3 FIX: pass artefacts_dir so JudgeGate can load the training
        # column layout and reindex live features to match (prevents
        # CatBoostError when live FeaturePipeline columns differ from training).
        self._judges: dict[str, dict[str, JudgeGate]] = {}
        if config.judge_dir is not None:
            _artefacts_dir = config.judge_dir.parent  # artefacts/judge_models → artefacts/
            self._judges = build_judge_gates(
                config.symbols, config.judge_dir, artefacts_dir=_artefacts_dir
            )

        # AFML-correct external CUSUMFilters (shadow mode only).
        # In paper/live mode the internal CUSUM in ModelSignal is used instead.
        self._cusum_filters: dict[str, CUSUMFilter] = {}
        if config.mode == EngineMode.SHADOW and config.cusum_multipliers:
            artefacts_dir = Path(config.audit_log_path).parent.parent
            self._cusum_filters = build_cusum_filters(
                config.symbols, config.cusum_multipliers, artefacts_dir
            )

        # Pending rebalance signals (judge-gated, event-driven)
        self._pending_rebalance: dict[str, object | None] = {
            s: None for s in config.symbols
        }

        # State file output
        self._state_out_dir = config.state_out_dir
        self._state_out_dir.mkdir(parents=True, exist_ok=True)
        self._state_write_every_n = config.state_write_every_n
        self._equity_history: list[float] = []
        self._start_ts: str = pd.Timestamp.now(tz="UTC").isoformat()

        # Start Prometheus HTTP server (P2-2)
        if config.metrics_port > 0:
            start_metrics_server(port=config.metrics_port)

        # Install OS-level kill switch (Wave 15 P0-5.1)
        self._install_kill_switch()

    # ------------------------------------------------------------------
    # Kill switch (Wave 15 P0-5.1)
    # ------------------------------------------------------------------

    def _install_kill_switch(self) -> None:
        """Install synchronous SIGUSR1 kill switch (Wave 15 P0-5.1).

        Sends SIGUSR1 to halt all orders immediately. Safe from signal context.
        """
        import platform
        if platform.system() == "Windows":
            # Windows: SIGTERM as fallback (SIGUSR1 not supported)
            _signal.signal(_signal.SIGTERM, self._kill_switch_handler)
        else:
            _signal.signal(_signal.SIGUSR1, self._kill_switch_handler)
            _signal.signal(_signal.SIGTERM, self._kill_switch_handler)

    def _kill_switch_handler(self, signum: int, frame) -> None:
        """Atomic kill-switch: set halt flag, drain queue, alert."""
        self._kill_flag.set()
        logger.critical(
            "KILL SWITCH ACTIVATED (signal=%d) — halting all trading immediately.",
            signum,
        )

    # ------------------------------------------------------------------
    # Public control
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Start the engine and process bars until stopped or halted."""
        logger.info("LiveEngine: starting in %s mode.", self._cfg.mode.value)
        # P0-3 FIX (CHIEF AUDIT 2026-05-27):
        #   Initialise last_feed_ts NOW so the CB feed-timeout clock starts
        #   counting from engine start, not from first bar.  Without this,
        #   last_feed_ts stays None forever when the WS connects but sends
        #   zero data (e.g. geo-blocked Futures endpoint), and the CB's
        #   feed_timeout check (guarded by `is not None`) never fires —
        #   producing a silent infinite loop with no state.json written.
        self._state.last_feed_ts = pd.Timestamp.now(tz="UTC")
        await self._feed.start()

        while not self._stop_event.is_set():
            # Wave 15 P0-5.1 — kill switch check (atomic threading.Event)
            if self._kill_flag.is_set():
                await self._emergency_halt()
                return

            try:
                event = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                # Check feed timeout
                now = pd.Timestamp.now(tz="UTC")
                self._cb.check(now)
                if self._cb.is_active:
                    await self._halt()
                    break
                continue

            if event is None:
                logger.info("LiveEngine: feed exhausted — stopping.")
                break

            await self._process_bar(event)

            if self._cb.is_active:
                await self._halt()
                break

        self._state.mode = EngineMode.STOPPED
        logger.info(
            "LiveEngine: stopped. bars_processed=%d, p95_latency=%.1fms",
            self._bars_processed,
            self._p95_latency(),
        )

    def request_stop(self) -> None:
        """Signal the engine to stop after the current bar."""
        self._stop_event.set()

    # ------------------------------------------------------------------
    # Per-bar processing
    # ------------------------------------------------------------------

    async def _process_bar(self, event: BarEvent) -> None:
        t0 = time.monotonic()
        now = pd.Timestamp.now(tz="UTC")
        symbol = event.symbol

        # Wave 15 P0-5.6 — feed sequence number gap detection
        if hasattr(event, "seq") and event.seq > 0:
            last_seq = self._last_bar_seq.get(symbol, -1)
            if last_seq >= 0 and event.seq != last_seq + 1:
                gap = event.seq - last_seq - 1
                logger.error(
                    "FEED GAP DETECTED: %s seq=%d expected=%d (gap=%d bars). "
                    "Tripping circuit breaker.",
                    symbol, event.seq, last_seq + 1, gap,
                )
                self._cb._trip(
                    f"feed_gap:{symbol}:seq={event.seq}", now
                )
                return
            self._last_bar_seq[symbol] = event.seq

        # Update feed timestamp (used by CB feed-timeout check)
        self._state.last_feed_ts = now

        # Mark-to-market + bar metadata for slippage wiring (Rec 2 / #6)
        # F1 fix (CHIEF AUDIT 2026-05-25):
        #   Prefer LIVE L1 half-spread from feed.get_best_quote() over the
        #   synthetic (H-L)/8 estimate.  Synthetic stays as fallback only
        #   when the bookTicker stream hasn't produced a fresh quote yet
        #   (<5s old, see Feed.get_best_quote).
        #
        # Double-count safeguard (per operator note 2026-05-25):
        #   cost_bps_per_asset in prod.yaml is consumed ONLY by
        #   apps/backtest_portfolio.py — NEVER by PaperOMS.  So injecting
        #   the live L1 half-spread here does not stack on top of the YAML
        #   cost.  The live cost stack is: half-spread (this value) +
        #   taker_fee_bps (PaperOMS.place_order), mirroring the backtest
        #   spread_arr × 0.5 + fee model.
        sigma_proxy = (event.high - event.low) / max(event.close, 1e-9)
        live_quote = None
        feed = getattr(self, "_feed", None)
        if feed is not None and hasattr(feed, "get_best_quote"):
            live_quote = feed.get_best_quote(symbol)
        if live_quote is not None:
            bid, ask = live_quote
            mid = 0.5 * (bid + ask)
            if mid > 0 and ask > bid:
                # Half-spread in bps.  Cap at 100 bps to guard against
                # malformed quotes; floor at 0 (no synthetic minimum).
                spread_half_bps = min(100.0, (ask - bid) / (2.0 * mid) * 1e4)
            else:
                spread_half_bps = max(0.0, sigma_proxy / 8.0 * 1e4)
        else:
            # Synthetic fallback — no 5-bps floor (was double-counting the
            # taker_fee which is applied separately by PaperOMS).
            spread_half_bps = max(0.0, sigma_proxy / 8.0 * 1e4)
        self._paper_oms.set_bar_prices(
            {symbol: event.close},
            sigma_map={symbol: sigma_proxy},
            volume_map={symbol: float(event.volume)},
            spread_bps_map={symbol: spread_half_bps},
        )

        # P1-1 — Accrue 8h perpetual funding cost when the bar carries a rate.
        # BarEvent.funding_rate == 0.0 on non-funding bars (99 % of 1h bars).
        if event.funding_rate != 0.0:
            payment = self._paper_oms.accrue_funding(
                symbol, event.funding_rate, event.ts
            )
            self._metrics.funding_accrued(symbol=symbol, payment_usdt=payment)
            logger.debug(
                "Funding accrued: %s rate=%.6f payment=%.4f USDT",
                symbol, event.funding_rate, payment,
            )

        # 1. Feature update
        # Include taker_buy_volume so FeaturePipeline can compute CVD features
        # (feat_cvd_flow_ffd, feat_cvd_session_z, feat_rvd_z) identically to
        # training — these 9/111 model features require taker-side volume.
        bar_series = pd.Series(
            {
                "open":             event.open,
                "high":             event.high,
                "low":              event.low,
                "close":            event.close,
                "volume":           event.volume,
                "taker_buy_volume": event.taker_buy_volume,
            },
            name=event.ts,
        )
        features = self._fu.update(symbol, bar_series)

        # ── 2. AFML bifurcation ───────────────────────────────────────────────
        #
        # SHADOW mode (live WebSocket):  AFML-correct event-driven path
        #   CUSUMFilter steps on every raw 5s bar (O(1)).
        #   When CUSUM fires → force_refresh() runs FeaturePipeline NOW
        #   → predict_on_event() on FRESH features at event time.
        #
        # PAPER / LIVE mode:  legacy path (internal CUSUM in ModelSignal)
        #   Features from FeatureUpdater (fixed 538-bar refresh or cached).
        #   ModelSignal.predict() steps its own CUSUM.
        #
        signal_result = None

        if self._cfg.mode == EngineMode.SHADOW and self._cusum_filters:
            # ── Shadow path: external CUSUM drives feature refresh ────────────
            cusum_filter = self._cusum_filters.get(symbol)
            cusum_fired = cusum_filter.step(event.close) if cusum_filter else False

            if cusum_fired:
                # Feature pipeline runs EXACTLY at the CUSUM event — no staleness.
                # W-2 (2026-05-27): use force_refresh_async() so the CPU-bound
                # FeaturePipeline (3–10 s for 1.5M-bar buffer → 3 000 HRB bars)
                # runs in the thread-pool executor instead of blocking the asyncio
                # event loop.  Thread safety guaranteed by force_refresh() taking a
                # list-snapshot of the buffer at entry.
                fresh_features = await self._fu.force_refresh_async(symbol)
                if fresh_features is not None:
                    # Update CUSUM ATR from fresh features for next interval
                    if cusum_filter is not None:
                        cusum_filter.update_atr(fresh_features)
                    # P1-5: pass event.ts so signal timestamp = bar close,
                    # not wall-clock (parity with backtest convention).
                    signal_result = self._sr.predict_on_event(
                        symbol, fresh_features, bar_ts=event.ts
                    )
                    features = fresh_features   # use for JudgeGate below
                else:
                    logger.debug(
                        "LiveEngine [%s]: CUSUM fired but force_refresh returned None "
                        "(buffer still warming up).", symbol,
                    )
            # features stays as the cached+close-injected row from FeatureUpdater
            # (used for mark-to-market below) but NOT for prediction.

        # ── Paper/live legacy path: internal CUSUM in ModelSignal ─────────
        elif features is not None:
            # P1-5: forward bar.ts for audit/TCA timestamp parity.
            signal_result = self._sr.predict(symbol, features, bar_ts=event.ts)

        # JudgeGate meta-labeling filter
        # B-1 (2026-05-27): bypass JudgeGate when runs-bar count from the
        # last force_refresh is below the full-burn-in threshold (500 bars).
        # With partial burn-in (100 usable bars vs 2000 full), degraded feature
        # values cause the judge model to score < tau even on genuine signals.
        # During partial burn-in we fall back to cal_prob >= min_conf as the
        # sole gate.  This allows paper-trade data to accumulate while the buffer
        # grows toward the 2000-bar full burn-in (takes ~14 days live at normal
        # density).  Bypasses are logged at INFO so they show up in paper_trade_stdout.log.
        _JUDGE_BURNIN_BYPASS_BARS: int = 500
        if (
            signal_result is not None
            and signal_result.signal != 0
            and features is not None
        ):
            side_str = "LONG" if signal_result.signal > 0 else "SHORT"
            gate = self._judges.get(symbol, {}).get(side_str)
            usable_bars = self._fu.get_last_usable_bars(symbol)
            burnin_bypass = usable_bars < _JUDGE_BURNIN_BYPASS_BARS

            if gate is not None and not burnin_bypass:
                passes, p_combined = gate.passes(features, signal_result.confidence, event.ts)
                if passes:
                    self._pending_rebalance[symbol] = signal_result
                    logger.info(
                        "JudgeGate PASS [%s/%s] p_combined=%.4f ts=%s",
                        symbol, side_str, p_combined, event.ts,
                    )
                else:
                    logger.debug(
                        "JudgeGate BLOCK [%s/%s] p_combined=%.4f ts=%s",
                        symbol, side_str, p_combined, event.ts,
                    )
            elif burnin_bypass:
                # Partial burn-in bypass: signal_result.signal != 0 guarantees
                # cal_prob >= min_conf was already checked inside ModelSignal
                # before returning a non-flat result.  No extra gate needed.
                self._pending_rebalance[symbol] = signal_result
                logger.info(
                    "JudgeGate BYPASS [%s/%s] (partial burn-in: %d usable bars) "
                    "cal_prob=%.4f -> PASS ts=%s",
                    symbol, side_str, usable_bars,
                    signal_result.confidence, event.ts,
                )
            else:
                # No judge model loaded — signal passes directly.
                self._pending_rebalance[symbol] = signal_result

        # Store bar state
        if features is not None:
            self._state.last_bar[symbol] = features
        self._state.last_bar_ts[symbol] = event.ts

        # Propfirm governor (audit §3) — evaluated BEFORE the circuit breaker.
        # HARD_FLATTEN/FAIL trip the CB (existing halt path flattens + stops);
        # SOFT_STOP blocks NEW risk this bar without a full halt.
        gov_soft = False
        if self._governor is not None:
            gov_eq = self._paper_oms.tracker.equity
            if gov_eq > 0:
                gd = self._governor.update(gov_eq)
                if gd.action in (GovernorAction.HARD_FLATTEN, GovernorAction.FAIL):
                    logger.critical(
                        "Propfirm governor %s — tripping CB: %s",
                        gd.action.value, gd.reason,
                    )
                    self._cb._trip(
                        f"propfirm_governor:{gd.action.value}:{gd.reason}", ts=now,
                    )
                elif gd.action == GovernorAction.SOFT_STOP:
                    gov_soft = True
                    logger.warning(
                        "Propfirm governor SOFT_STOP (%s) — no new risk this bar.",
                        gd.reason,
                    )

        # P1-2 — Circuit-breaker BEFORE order generation.
        # Equity is already marked-to-market; drawdown is current.
        # If CB trips here we record metrics and return immediately;
        # run() detects is_active on the next iteration and calls _halt().
        self._cb.check(now)
        self._metrics.circuit_breaker_state(active=self._cb.is_active)
        if self._cb.is_active:
            elapsed_ms = (time.monotonic() - t0) * 1000
            self._latencies.append(elapsed_ms)
            self._bars_processed += 1
            self._metrics.bar_processed(symbol=symbol, latency_s=elapsed_ms / 1000.0)
            logger.warning(
                "LiveEngine: CB active — skipping order generation for bar %s %s.",
                symbol, event.ts,
            )
            return

        # 3. Portfolio optimisation — only when a judge-gated signal fired this bar.
        # Event-driven: skip continuous rebalancing to avoid excessive turnover.
        all_prices = {
            s: self._paper_oms._last_close.get(s, 0.0)
            if hasattr(self._paper_oms, "_last_close") else 0.0
            for s in self._cfg.symbols
        }
        current_prices_ready = all(v > 0 for v in all_prices.values())

        any_pending = any(v is not None for v in self._pending_rebalance.values())
        if current_prices_ready and any_pending and not gov_soft:
            signals_map = {s: self._pending_rebalance.get(s) for s in self._cfg.symbols}

            target_weights = self._pc.optimise(signals_map, all_prices)
            self._state.target_weights = dict(target_weights)

            # Rec 3 (Sim-to-Reality #5): propagate crisis multiplier to OMS
            # so slippage and impact are scaled up during liquidity-stress regimes.
            self._paper_oms.set_crisis_multiplier(self._pc.crisis_multiplier)

            # 4. Size orders
            current_weights: dict[str, float] = {}
            eq = self._paper_oms.tracker.equity
            if eq > 0:
                for sym, pos in self._paper_oms.tracker.get_all_positions().items():
                    current_weights[sym] = pos.notional / eq

            signal_probs = {
                s: r.confidence
                for s, r in self._pending_rebalance.items()
                if r is not None
            }
            orders = self._ec.size_orders(
                target_weights=target_weights,
                current_weights=current_weights,
                prices=all_prices,
                equity=eq,
                signal_probs=signal_probs,
            )

            # 5. Route orders — reset fat-finger guard after each fill
            for order in orders:
                fill = await self._router.place_order(order)
                # Reset fat-finger guard so next rebalance isn't blocked
                self._ec._last_order_qty[order.symbol] = 1e9
                self._metrics.order_filled(
                    symbol=order.symbol, side=order.side.value
                )
                logger.info(
                    "FILL: %s %s qty=%.4f @ %.4f notional=$%.0f",
                    order.symbol, order.side.value,
                    fill.fill_qty, fill.fill_price, fill.notional_usdt,
                )

            # Clear pending signals after rebalance
            self._pending_rebalance = {s: None for s in self._cfg.symbols}

            # 6. Update equity state
            new_equity = self._paper_oms.tracker.equity
            self._state.update_equity(new_equity)
            self._state.positions = self._paper_oms.get_positions()
            self._equity_history.append(new_equity)

            # P2-2 — equity + drawdown gauges
            self._metrics.equity_updated(
                equity=new_equity,
                drawdown=self._state.current_drawdown,
            )

        # Periodic equity sample for rolling Sharpe (every bar, capped at 5000)
        eq_now = self._paper_oms.tracker.equity
        # CHIEF-4 (2026-05-28): track equity every bar (not only on rebalance)
        # so peak/intraday drawdown reflects intra-bar moves between trades.
        self._state.update_equity(eq_now)
        # UTC-day rollover → reset intraday DD tracker.
        cur_date = pd.Timestamp(event.ts).normalize()
        last_date = getattr(self, "_last_session_date", None)
        if last_date is None:
            self._last_session_date = cur_date
        elif cur_date != last_date:
            self._state.reset_daily()
            if self._governor is not None:
                self._governor.reset_day(eq_now)
            self._last_session_date = cur_date
        self._equity_history.append(eq_now)
        if len(self._equity_history) > 5000:
            self._equity_history = self._equity_history[-5000:]

        # P2-2 — queue depth gauge (best-effort; queue may not be accessible)
        self._metrics.queue_depth(depth=self._queue.qsize())

        # Record latency metrics
        elapsed_ms = (time.monotonic() - t0) * 1000
        self._latencies.append(elapsed_ms)
        self._bars_processed += 1
        self._metrics.bar_processed(symbol=symbol, latency_s=elapsed_ms / 1000.0)

        if elapsed_ms > 500:
            logger.warning(
                "LiveEngine: slow bar processing %.1fms (target <500ms).", elapsed_ms
            )

        # Periodic state file write for dashboard
        if self._bars_processed % self._state_write_every_n == 0:
            self._write_state(event.ts)

    # ------------------------------------------------------------------
    # State file writes (for Streamlit dashboard)
    # ------------------------------------------------------------------

    def _rolling_sharpe(self, window: int = 5000) -> float | None:
        """Annualised Sharpe from the last `window` equity samples.

        Returns None until at least 17 280 samples are available (= 1 full
        calendar day of 5s-bar data: 24 h × 3600 s / 5 s).  Below this
        threshold the annualised figure is meaningless: the annualisation
        factor √(252 × 24 × 720) ≈ 2 086 amplifies any short-window drift
        into triple- or double-digit values (observed: 41 at bar 3 500 /
        5.8 h with equity +0.43 %).  The dashboard renders "—" for None.

        S-1 (2026-05-27): initial threshold set to 2880 (≈ 4 h).
        S-1b (2026-05-27): raised to 17 280 (= 1 day) after 41-Sharpe
        artefact observed at bar 3 500.
        """
        hist = self._equity_history[-window:]
        # S-1b (2026-05-27): require ≥ 17 280 samples (1 full calendar day
        # of 5s bars) before reporting an annualised Sharpe.
        if len(hist) < 17_280:
            return None
        rets = [(hist[i] - hist[i - 1]) / hist[i - 1] for i in range(1, len(hist))]
        mu = sum(rets) / len(rets)
        sigma = math.sqrt(sum((r - mu) ** 2 for r in rets) / len(rets))
        if sigma < 1e-12:
            return None
        # 5s bars → 252 trading days × 24h × 720 bars/h
        bars_per_year = 252 * 24 * 720
        return float(mu / sigma * math.sqrt(bars_per_year))

    def _write_state(self, current_bar_ts: pd.Timestamp | None = None) -> None:
        """Atomic write of state.json + append to equity_curve.jsonl."""
        eq = self._paper_oms.tracker.equity
        peak = self._state.equity_peak
        # CHIEF-4 (2026-05-28): the dashboard "Max Drawdown" KPI used to
        # read ``current_drawdown`` (instantaneous peak-to-current), which
        # is 0 the moment equity recovers to a new high.  We now also
        # publish historic and intraday max-DD so the KPI matches what the
        # operator sees on the drawdown chart.
        cur_dd_pct = self._state.current_drawdown * 100.0
        peak_dd_pct = self._state.peak_drawdown * 100.0
        intraday_dd_pct = self._state.intraday_peak_drawdown * 100.0
        init_eq = self._cfg.initial_equity
        ret_pct = (eq - init_eq) / init_eq * 100.0

        positions: dict[str, dict] = {}
        for sym, pos in self._paper_oms.tracker.get_all_positions().items():
            # B-2 FIX (2026-05-27): PositionRecord uses .qty, not .size
            qty_val = getattr(pos, "qty", None) or getattr(pos, "size", 0.0)
            side_val = (
                pos.side.value if hasattr(pos, "side") and hasattr(pos.side, "value")
                else ("LONG" if qty_val >= 0 else "SHORT")
            )
            positions[sym] = {
                "qty": round(float(qty_val), 6),
                "notional": round(float(pos.notional), 2),
                "side": side_val,
            }

        prices = {}
        if hasattr(self._paper_oms, "_last_close"):
            prices = {s: round(float(v), 6) for s, v in self._paper_oms._last_close.items()}

        state_doc = {
            "mode": self._cfg.mode.value,
            "start_ts": self._start_ts,
            "current_bar_ts": current_bar_ts.isoformat() if current_bar_ts else None,
            "timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
            "equity": round(eq, 2),
            "initial_equity": init_eq,
            "peak_equity": round(peak, 2),
            "pnl_usdt": round(eq - init_eq, 2),
            "pnl_pct": round(ret_pct, 4),
            # CHIEF-4: ``drawdown_pct`` retained for backward compat
            # (=current/instantaneous).  New fields are the real maxima.
            "drawdown_pct": round(cur_dd_pct, 4),
            "current_drawdown_pct": round(cur_dd_pct, 4),
            "max_drawdown_pct": round(peak_dd_pct, 4),
            "intraday_drawdown_pct": round(intraday_dd_pct, 4),
            "total_return_pct": round(ret_pct, 4),
            "rolling_sharpe_32d": self._rolling_sharpe(),
            "n_trades": self._audit.records_written,
            "bars_processed": self._bars_processed,
            "n_bars_processed": self._bars_processed,
            "cb_active": self._cb.is_active,
            "positions": positions,
            "prices": prices,
        }

        # Atomic write via tmp → rename
        out_dir = self._state_out_dir
        tmp = out_dir / "state.tmp"
        tmp.write_text(json.dumps(state_doc, default=str))
        tmp.replace(out_dir / "state.json")

        # Append to equity curve
        ec_line = json.dumps({
            "ts": pd.Timestamp.now(tz="UTC").isoformat(),
            "equity": round(eq, 2),
            "bar": self._bars_processed,
        })
        with open(out_dir / "equity_curve.jsonl", "a", encoding="utf-8") as fh:
            fh.write(ec_line + "\n")

    # ------------------------------------------------------------------
    # Halt procedure
    # ------------------------------------------------------------------

    async def _halt(self) -> None:
        logger.critical(
            "LiveEngine: HALT triggered. reason=%s — closing all positions.",
            self._cb.halt_reason,
        )
        fills = self._paper_oms.close_all()
        logger.critical("LiveEngine: closed %d positions.", len(fills))
        self._stop_event.set()

    async def _emergency_halt(self) -> None:
        """Emergency halt: cancel all open orders, log state, alert.

        Called on KILL SWITCH signal (Wave 15 P0-5.1).
        """
        logger.critical("EMERGENCY HALT — cancelling all open orders.")
        if hasattr(self, "_router") and self._router is not None:
            await self._router.close_all()
        else:
            # Fallback to paper OMS
            self._paper_oms.close_all()
        self._stop_event.set()

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def _p95_latency(self) -> float:
        return self._latency_quantile(0.95)

    def _latency_quantile(self, q: float) -> float:
        """Return the q-quantile of recent latencies in seconds.

        CHIEF AUDIT-FIX (Sim-to-Reality #13):
          P95 only captures median-tail latency; P99/P99.9 is where the
          OS-pause / network-stall pathology lives.  These tail spikes
          queue follow-up orders behind late fill confirmations and
          double the effective latency observed in live trading.  We
          retain the full latency vector (already done) and expose
          arbitrary quantiles via this helper.
        """
        if not self._latencies:
            return 0.0
        sorted_lat = sorted(self._latencies)
        idx = int(float(q) * len(sorted_lat))
        return sorted_lat[min(idx, len(sorted_lat) - 1)]

    def _p99_latency(self) -> float:
        return self._latency_quantile(0.99)

    def _p999_latency(self) -> float:
        return self._latency_quantile(0.999)

    def latency_report(self) -> dict[str, float]:
        """Full latency tail report — P50/P95/P99/P99.9 and max.

        CHIEF AUDIT-FIX (Sim-to-Reality #13):
          Operators previously only saw P95.  Add P99/P99.9/max so
          tail-spike pathologies are visible.  An empirical rule of thumb
          for crypto MFT: P99.9 should stay within 4× the P95 — beyond
          that, the queue-of-orders-behind-confirmation pattern becomes
          dominant and the strategy needs throttling.
        """
        if not self._latencies:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "p999": 0.0, "max": 0.0}
        sorted_lat = sorted(self._latencies)
        n = len(sorted_lat)
        return {
            "p50":  sorted_lat[int(0.50 * n)],
            "p95":  sorted_lat[min(int(0.95 * n), n - 1)],
            "p99":  sorted_lat[min(int(0.99 * n), n - 1)],
            "p999": sorted_lat[min(int(0.999 * n), n - 1)],
            "max":  sorted_lat[-1],
        }

    @property
    def bars_processed(self) -> int:
        return self._bars_processed

    @property
    def state(self) -> SystemState:
        return self._state

    @property
    def latencies(self) -> list[float]:
        return list(self._latencies)

    @property
    def audit_log(self) -> AuditLog:
        return self._audit
