"""apps/paper_trade_runner.py — Historical feature replay paper trade runner.

Replays the last REPLAY_DAYS of pre-computed features through the trained
ModelSignals (LONG + SHORT per symbol), PortfolioController (HRP), and
PaperOMS to simulate live trading without Docker or a live WS connection.

Output (artefacts/paper_trade/):
  state.json        — live snapshot (equity, positions, metrics)
  equity_curve.jsonl — per-bar equity log
  audit.jsonl       — order fills (JSONL, same schema as live engine)

Usage:
    python apps/paper_trade_runner.py [--days 14] [--delay 0.0] [--symbols ETH,SOL,...]

Speed:
    delay=0.0 (default): max-speed replay, ~1000x real-time.
    delay=5.0           : 5s/bar = real shadow-trade cadence.

This runner uses artefacts/features/{SYM}.parquet (pre-computed by
build_features pipeline) so ModelSignals see the EXACT same features
as the backtest — zero feature distribution shift.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from tradebot.live.model_signal import ModelSignal, ModelSignalConfig
from tradebot.live.portfolio_controller import PortfolioController, PortfolioControllerConfig
from tradebot.live.execution_controller import ExecutionController, ExecutionControllerConfig
from tradebot.live.signal_runner import SignalRunner, SignalRunnerConfig
from tradebot.oms.paper_oms import PaperOMS
from tradebot.oms.audit_log import AuditLog
from tradebot.alpha.base import SignalResult

try:
    from catboost import CatBoostClassifier, Pool as CatPool
    _CB_OK = True
except ImportError:
    _CB_OK = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("paper_trade")

# Asymmetric judge thresholds (from backtest_portfolio.py — CHIEF fix)
_JUDGE_TAU_LONG  = 0.40
_JUDGE_TAU_SHORT = 0.75


class JudgeGate:
    """Secondary meta-labeling filter — mirrors backtest P_combined = P_primary × P_judge.

    The judge CatBoost model was trained on:
      - 126 raw feature columns (X_primary.values) in parquet order
      - prob_primary (Platt-calibrated probability from primary model)
      - hour_sin, hour_cos, dow_sin, dow_cos (time features)
    Total: 131 features passed as numpy float32 array.

    P_combined = P_primary × P_judge.
    LONG: trade if P_combined > 0.40.
    SHORT: trade if P_combined > 0.75.
    """

    def __init__(self, sym: str, side: str, judge_dir: Path) -> None:
        self._sym = sym
        self._side = side
        self._model: Optional[CatBoostClassifier] = None
        self._n_feat: int = 0
        self._tau = _JUDGE_TAU_LONG if side == "LONG" else _JUDGE_TAU_SHORT

        if not _CB_OK:
            logger.warning("JudgeGate [%s/%s]: CatBoost not available — gate disabled.", sym, side)
            return

        path = judge_dir / f"{sym}_{side}_judge.cbm"
        if not path.exists():
            logger.warning("JudgeGate [%s/%s]: model not found at %s — gate disabled.", sym, side, path)
            return

        self._model = CatBoostClassifier()
        self._model.load_model(str(path))
        self._n_feat = len(self._model.feature_names_)
        logger.info("JudgeGate [%s/%s]: loaded — %d features, tau=%.2f", sym, side, self._n_feat, self._tau)

    def passes(
        self,
        row_df: pd.DataFrame,
        cal_prob: float,
        ts: pd.Timestamp,
    ) -> tuple[bool, float]:
        """Return (passes_gate, p_combined).

        If judge model not loaded, always passes with p_combined=cal_prob.
        """
        if self._model is None:
            return True, cal_prob

        try:
            # Raw feature values in parquet column order
            x_raw = row_df.values.flatten()[:self._n_feat - 5].astype(np.float32)

            # Time features (same as build_judge_features)
            h = ts.hour
            dow = ts.dayofweek
            time_feats = np.array([
                np.sin(2 * np.pi * h / 24),
                np.cos(2 * np.pi * h / 24),
                np.sin(2 * np.pi * dow / 7),
                np.cos(2 * np.pi * dow / 7),
            ], dtype=np.float32)

            # prob_primary slot + time features: total = (n_feat - 5) + 1 + 4 = n_feat
            judge_x = np.concatenate([x_raw, [float(cal_prob)], time_feats]).reshape(1, -1)

            # NaN fill (CatBoost doesn't handle NaN in numpy input)
            judge_x = np.nan_to_num(judge_x, nan=0.0, posinf=0.0, neginf=0.0)

            p_judge = float(self._model.predict_proba(judge_x)[0][1])
            p_combined = cal_prob * p_judge
            passes = p_combined > self._tau
            return passes, p_combined

        except Exception as exc:
            logger.debug("JudgeGate [%s/%s]: error — %s. Passing.", self._sym, self._side, exc)
            return True, cal_prob

_ARTEFACTS = _ROOT / "artefacts"
_FEATURES  = _ARTEFACTS / "features"
_OUT_DIR   = _ARTEFACTS / "paper_trade"

_SYMBOLS_DEFAULT = ["ETHUSDT", "SOLUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT"]
_CUSUM_MULT = {
    "ETHUSDT": 2.5, "SOLUSDT": 2.5, "AVAXUSDT": 2.2,
    "LINKUSDT": 2.0, "DOTUSDT": 2.0,
}
_INITIAL_EQUITY = 200_000.0
_STATE_WRITE_EVERY = 10   # write state.json every N bars


# ── Helpers ────────────────────────────────────────────────────────────────────

def _load_features(sym: str, days: int) -> pd.DataFrame:
    """Load the last `days` worth of pre-computed features for `sym`."""
    path = _FEATURES / f"{sym}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Feature parquet not found: {path}")
    df = pd.read_parquet(path)
    # Ensure UTC-aware DatetimeIndex
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    # Bars per day at ~44.8-min cadence ≈ 32 bars/day
    bars_needed = max(days * 35, 200)
    return df.iloc[-bars_needed:]


_JUDGE_DIR = _ARTEFACTS / "judge_models"


def _build_judge_gates(symbols: List[str]) -> Dict[str, Dict[str, JudgeGate]]:
    """Load JudgeGate for each (symbol, side)."""
    gates: Dict[str, Dict[str, JudgeGate]] = {}
    for sym in symbols:
        gates[sym] = {}
        for side in ("LONG", "SHORT"):
            gates[sym][side] = JudgeGate(sym, side, _JUDGE_DIR)
    return gates


def _build_signals(symbols: List[str]) -> Dict[str, Dict[str, ModelSignal]]:
    """Return {sym: {"LONG": sig, "SHORT": sig}} for all loadable signals."""
    signals: Dict[str, Dict[str, ModelSignal]] = {}
    for sym in symbols:
        signals[sym] = {}
        mult = _CUSUM_MULT.get(sym, 2.0)
        for side in ("LONG", "SHORT"):
            try:
                cfg = ModelSignalConfig(
                    symbol=sym,
                    side=side,
                    artefacts_dir=_ARTEFACTS,
                    cusum_threshold_multiplier=mult,
                    ema_regime_span=200,
                )
                sig = ModelSignal(cfg)
                sig.fit(pd.DataFrame())  # warm-start from parquet
                signals[sym][side] = sig
                logger.info("Loaded ModelSignal [%s/%s]", sym, side)
            except FileNotFoundError as exc:
                logger.warning("Skipping [%s/%s]: %s", sym, side, exc)
    return signals


def _write_state(
    out_dir: Path,
    equity: float,
    peak: float,
    positions: Dict[str, float],
    prices: Dict[str, float],
    n_bars: int,
    n_trades: int,
    equity_history: List[float],
    cb_active: bool,
    start_ts: str,
    current_ts: str,
    symbols: List[str],
) -> None:
    """Atomic write of state.json for the Streamlit monitor."""
    # Rolling Sharpe on equity curve (last 30 obs = ~1 month)
    sharpe = float("nan")
    if len(equity_history) > 5:
        eq = np.array(equity_history[-60:], dtype=float)
        rets = np.diff(eq) / eq[:-1]
        if rets.std() > 1e-10:
            sharpe = float(rets.mean() / rets.std() * np.sqrt(252 * 32))

    drawdown = float((peak - equity) / peak) if peak > 0 else 0.0

    pos_detail = {}
    for sym, qty in positions.items():
        price = prices.get(sym, 0.0)
        notional = abs(qty) * price
        pos_detail[sym] = {
            "qty": round(qty, 6),
            "notional_usdt": round(notional, 2),
            "side": "LONG" if qty > 0 else ("SHORT" if qty < 0 else "FLAT"),
        }

    state = {
        "ts": pd.Timestamp.now(tz="UTC").isoformat(),
        "start_ts": start_ts,
        "current_bar_ts": current_ts,
        "equity": round(equity, 2),
        "initial_equity": _INITIAL_EQUITY,
        "peak_equity": round(peak, 2),
        "pnl_usdt": round(equity - _INITIAL_EQUITY, 2),
        "pnl_pct": round((equity / _INITIAL_EQUITY - 1.0) * 100, 3),
        "drawdown_pct": round(drawdown * 100, 3),
        "rolling_sharpe_32d": round(sharpe, 4) if not np.isnan(sharpe) else None,
        "bars_processed": n_bars,
        "n_trades": n_trades,
        "cb_active": cb_active,
        "mode": "paper_replay",
        "symbols": symbols,
        "positions": pos_detail,
        "prices": {s: round(p, 4) for s, p in prices.items()},
    }
    tmp = out_dir / "_state_tmp.json"
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp.replace(out_dir / "state.json")


def _append_equity(
    out_dir: Path,
    equity: float,
    n_bars: int,
    ts: str,
) -> None:
    with open(out_dir / "equity_curve.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"ts": ts, "equity": round(equity, 2), "bar": n_bars}) + "\n")


# ── Main replay loop ───────────────────────────────────────────────────────────

def _run_replay(
    symbols: List[str],
    days: int,
    delay: float,
) -> None:
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Clear previous run's equity curve (but keep audit for continuity)
    eq_path = _OUT_DIR / "equity_curve.jsonl"
    if eq_path.exists():
        eq_path.unlink()

    audit_log = AuditLog(_OUT_DIR / "audit.jsonl")

    # Load features per symbol
    feat: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            feat[sym] = _load_features(sym, days)
            logger.info("[%s] Loaded %d feature bars (%.1f days)",
                        sym, len(feat[sym]), len(feat[sym]) / 32)
        except FileNotFoundError as exc:
            logger.error("%s — removing from universe", exc)
    symbols = [s for s in symbols if s in feat]

    if not symbols:
        logger.error("No feature data found. Run build_features first.")
        return

    # Build signals + judge gates
    sig_map = _build_signals(symbols)
    judge_gates = _build_judge_gates(symbols)

    # Build SignalRunner (same wiring as live_trader.py)
    all_sigs = [
        sig_map[sym][side]
        for sym in symbols
        for side in ("LONG", "SHORT")
        if side in sig_map.get(sym, {})
    ]
    sr = SignalRunner(
        signals=all_sigs,
        config=SignalRunnerConfig(min_confidence=0.0, use_combiner=False),
    )
    logger.info("SignalRunner: %d signals wired", len(all_sigs))

    # Build portfolio + execution
    pc = PortfolioController(
        PortfolioControllerConfig(method="hrp", min_history_bars=30),
        symbols=symbols,
    )
    ec = ExecutionController(ExecutionControllerConfig(max_weight_change=0.25))
    # Disable fat-finger for paper replay — no manual override risk
    for sym in symbols:
        ec._last_order_qty[sym] = 1e9

    # PaperOMS
    oms = PaperOMS(
        initial_equity=_INITIAL_EQUITY,
        audit_log=audit_log,
        taker_fee_bps=4.0,
    )

    # Merge all feature rows into one timeline, sorted by timestamp
    all_bars: List[tuple] = []
    for sym, df in feat.items():
        for ts, row in df.iterrows():
            all_bars.append((ts, sym, row))
    all_bars.sort(key=lambda x: x[0])
    logger.info("Total bars to replay: %d across %d symbols", len(all_bars), len(symbols))

    equity_history: List[float] = [_INITIAL_EQUITY]
    peak_equity = _INITIAL_EQUITY
    prices: Dict[str, float] = {}
    n_bars = 0
    n_trades = 0
    start_ts = pd.Timestamp.now(tz="UTC").isoformat()
    last_signals: Dict[str, Optional[SignalResult]] = {s: None for s in symbols}

    logger.info("Paper trade replay starting — %d days, delay=%.1fs", days, delay)
    t_run_start = time.monotonic()

    for ts, sym, row in all_bars:
        n_bars += 1
        row_df = pd.DataFrame([row], index=[ts])

        # Update price feed
        close = float(row.get("close", 0.0))
        if close > 0:
            prices[sym] = close

        oms.set_bar_prices(
            prices={sym: close} if close > 0 else {},
            sigma_map={sym: float(row.get("feat_atr_rel", row.get("feat_atr", 0.001)))},
            volume_map={sym: float(row.get("tick_volume", 1.0))},
        )

        # Signal prediction via SignalRunner (same as live engine)
        result = sr.predict(sym, row_df)
        if result is not None and result.signal != 0:
            # Apply judge gate: P_combined = P_primary × P_judge
            side_str = "LONG" if result.signal > 0 else "SHORT"
            gate = judge_gates.get(sym, {}).get(side_str)
            passes, p_combined = gate.passes(row_df, result.confidence, ts) if gate else (True, result.confidence)
            if passes:
                last_signals[sym] = result
                logger.debug("[%s/%s] JUDGE PASS — p_prim=%.4f p_comb=%.4f",
                             sym, side_str, result.confidence, p_combined)
            else:
                logger.debug("[%s/%s] JUDGE REJECT — p_prim=%.4f p_comb=%.4f",
                             sym, side_str, result.confidence, p_combined)

        # Portfolio optimise ONLY when a new signal fires — matches backtest event-driven logic.
        # Continuous rebalancing on every bar causes fat-finger + excess turnover.
        any_new_signal = any(v is not None and v.signal != 0 for v in last_signals.values())
        if any_new_signal and len(prices) == len(symbols):
            signals_map = {s: last_signals.get(s) for s in symbols}
            target_weights = pc.optimise(signals_map, prices)

            # Current weights
            eq = oms.tracker.equity
            current_weights: Dict[str, float] = {}
            if eq > 0:
                for s, pos in oms.tracker.get_all_positions().items():
                    current_weights[s] = pos.notional / eq

            # Signal confidences
            sig_probs = {}
            for s, _sig_res in last_signals.items():
                if _sig_res is not None and _sig_res.signal != 0:
                    sig_probs[s] = _sig_res.confidence

            # Size orders
            orders = ec.size_orders(
                target_weights=target_weights,
                current_weights=current_weights,
                prices=prices,
                equity=eq,
                signal_probs=sig_probs,
            )

            for order in orders:
                try:
                    fill = oms.place_order(order)
                    n_trades += 1
                    ec._last_order_qty[order.symbol] = 1e9  # reset after fill
                    logger.info(
                        "FILL [%s] %s qty=%.4f @ %.4f equity=%.2f",
                        order.symbol, order.side.value,
                        fill.fill_qty, fill.fill_price, oms.tracker.equity,
                    )
                except Exception as exc:
                    logger.warning("Order failed [%s]: %s", order.symbol, exc)

            # After fills, clear stale signals to avoid re-triggering
            for s in symbols:
                if last_signals.get(s) is not None:
                    last_signals[s] = None

        # Equity tracking
        new_eq = oms.tracker.equity
        peak_equity = max(peak_equity, new_eq)
        equity_history.append(new_eq)

        # Write monitoring state
        if n_bars % _STATE_WRITE_EVERY == 0:
            _write_state(
                _OUT_DIR, new_eq, peak_equity,
                oms.get_positions(), prices,
                n_bars, n_trades, equity_history,
                cb_active=False,
                start_ts=start_ts,
                current_ts=ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
                symbols=symbols,
            )
            _append_equity(_OUT_DIR, new_eq, n_bars, str(ts))

        if delay > 0:
            time.sleep(delay)

    # Final state write
    final_eq = oms.tracker.equity
    _write_state(
        _OUT_DIR, final_eq, peak_equity,
        oms.get_positions(), prices,
        n_bars, n_trades, equity_history,
        cb_active=False,
        start_ts=start_ts,
        current_ts=str(pd.Timestamp.now(tz="UTC")),
        symbols=symbols,
    )
    _append_equity(_OUT_DIR, final_eq, n_bars, str(pd.Timestamp.now(tz="UTC")))

    elapsed = time.monotonic() - t_run_start
    returns = np.array(equity_history)
    rets = np.diff(returns) / returns[:-1]
    sharpe = float(rets.mean() / rets.std() * np.sqrt(252 * 32)) if rets.std() > 1e-10 else 0.0
    max_dd = float(np.max(np.maximum.accumulate(returns) - returns) / np.maximum.accumulate(returns).max())

    logger.info("=" * 60)
    logger.info("PAPER TRADE COMPLETED in %.1fs", elapsed)
    logger.info("  Bars replayed : %d", n_bars)
    logger.info("  Trades        : %d", n_trades)
    logger.info("  Final equity  : $%.2f", final_eq)
    logger.info("  PnL           : $%.2f (%.2f%%)",
                final_eq - _INITIAL_EQUITY,
                (final_eq / _INITIAL_EQUITY - 1) * 100)
    logger.info("  Annualised Sharpe : %.3f", sharpe)
    logger.info("  Max Drawdown  : %.2f%%", max_dd * 100)
    logger.info("  State written to  : %s", _OUT_DIR)
    logger.info("=" * 60)


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paper trade runner — historical feature replay"
    )
    parser.add_argument(
        "--days", type=int, default=14,
        help="Number of days to replay (default: 14)"
    )
    parser.add_argument(
        "--delay", type=float, default=0.0,
        help="Seconds to sleep between bars (0 = max speed)"
    )
    parser.add_argument(
        "--symbols", type=str, default="",
        help="Comma-separated symbols (default: all 5)"
    )
    args = parser.parse_args()

    symbols = (
        [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        if args.symbols else _SYMBOLS_DEFAULT
    )

    _run_replay(symbols=symbols, days=args.days, delay=args.delay)


if __name__ == "__main__":
    main()
