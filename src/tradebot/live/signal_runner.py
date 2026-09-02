# src/tradebot/live/signal_runner.py
"""Load models and run predict() per bar-close event.

Combines outputs from all registered AlphaSignals using the
ICWeightedCombiner to produce a single combined signal per symbol.
"""
from __future__ import annotations

import logging

import pandas as pd

from ..alpha.base import AlphaSignal, SignalResult
from ..alpha.combination import ICWeightedCombiner
from ..utils.failfast import TradebotContractError

logger = logging.getLogger(__name__)

__all__ = ["SignalRunnerConfig", "SignalRunner"]


class SignalRunnerConfig:
    """Configuration for the signal runner.

    `min_confidence` BESTAAT HIER NIET MEER
    ----------------------------------------
    Tot Stage D onderdrukte deze runner elk signaal met `confidence < 0.55`,
    met R-9 §9 als grondslag. Audit §14 sluit modelvertrouwen echter uit als
    parameter die de positiegrootte bepaalt, en no-go 8 van de fase maakt dat
    bindend. Een onderdrukkingspoort is daar een vorm van: een signaal dat wordt
    weggefilterd levert exposure 0 op in plaats van de gesizede exposure, en dus
    bepaalt het vertrouwen wel degelijk de grootte -- alleen in twee stappen.

    De richting en de sterkte van het signaal bepalen de exposure; het
    vertrouwen van het model doet dat niet meer. Wie een signaal wil
    onderdrukken, doet dat in de alphalaag waar het signaal wordt gemaakt, niet
    in de live-runner waar de backtest geen tegenhanger heeft.

    Parameters
    ----------
    use_combiner :
        If True, combine signals with ICWeightedCombiner before returning.
        When both a LONG (signal>0) and SHORT (signal<0) result pass the
        confidence gate, the combiner is NOT used — the signals would just
        cancel to 0 (FLAT).  Instead, direction disambiguation fires:
        return the higher-confidence direction if the conviction delta
        exceeds ``min_direction_delta``, else return None (ambiguous, FLAT).
        ICWeightedCombiner is only applied when all raw_results point the
        same direction (i.e. multiple independent alphas for one side).
    ic_lookback :
        IC estimation lookback window (bars) for the combiner.
    min_direction_delta :
        Minimum absolute confidence difference between the best LONG and
        best SHORT signal to elect a winner.  If
        |conf_LONG - conf_SHORT| < min_direction_delta the bar is treated
        as FLAT (no signal returned).  Default 0.02 (2 pp).
    """

    def __init__(
        self,
        use_combiner: bool = True,
        ic_lookback: int = 60,
        min_direction_delta: float = 0.02,
    ) -> None:
        self.use_combiner = use_combiner
        self.ic_lookback = ic_lookback
        self.min_direction_delta = min_direction_delta


class SignalRunner:
    """Runs all registered AlphaSignals and returns a combined prediction.

    Parameters
    ----------
    signals :
        List of AlphaSignal implementations.
    config :
        Runner configuration.
    """

    def __init__(
        self,
        signals: list[AlphaSignal],
        config: SignalRunnerConfig | None = None,
    ) -> None:
        self._signals = signals
        self._cfg = config or SignalRunnerConfig()
        self._combiner: ICWeightedCombiner | None = None  # lazy init on first predict
        # Rolling history for IC computation: signal_id → list of signal values
        self._signal_history: dict[str, list[float]] = {}
        self._return_history: list[float] = []

    # ------------------------------------------------------------------
    # Per-bar predict
    # ------------------------------------------------------------------

    def predict(
        self,
        symbol: str,
        features: pd.DataFrame,
        fwd_return: float | None = None,
        bar_ts: pd.Timestamp | None = None,
    ) -> SignalResult | None:
        """Run all signals on ``features`` and return a combined SignalResult.

        Parameters
        ----------
        symbol :
            Trading pair.
        features :
            Feature DataFrame ending at bar t (last row = current bar).
        fwd_return :
            Realised forward return for updating IC estimates (None at bar t).

        Returns
        -------
        Combined SignalResult, or None if no signal passes the confidence gate.
        """
        raw_results: list[SignalResult] = []
        for sig in self._signals:
            # Only run signals that are configured for this symbol.
            # Without this guard every bar (e.g. ETHUSDT) would step the CUSUM
            # accumulator and EMA-200 of SOLUSDT/AVAXUSDT/... with ETH close
            # prices, corrupting their internal state permanently.
            sig_symbol = getattr(getattr(sig, "_cfg", None), "symbol", symbol)
            if sig_symbol != symbol:
                continue
            try:
                # P1-5: forward bar_ts so SignalResult.timestamp = bar close,
                # not wall-clock.  Compatible with signals that ignore it.
                try:
                    result = sig.predict(features, bar_ts=bar_ts)
                except TypeError:
                    result = sig.predict(features)
                raw_results.append(result)
            except Exception as exc:
                # Phase 0: hier verdween een falend signaal STIL uit het boek.
                # De combiner rekende daarna met minder signalen dan
                # geconfigureerd, wat de effectieve weging van de overige
                # signalen verhoogt - een andere portefeuille dan de
                # gebacktestte, zonder dat een dashboard dat toont.
                raise TradebotContractError(
                    f"SignalRunner: {sig.signal_id}.predict() faalde: {exc}. "
                    f"Een signaal mag niet stil uit het boek vallen."
                ) from exc

        if not raw_results:
            return None

        if not self._cfg.use_combiner or len(raw_results) == 1:
            return raw_results[0]

        # ── Direction disambiguation ──────────────────────────────────────────
        # When LONG (signal>0) and SHORT (signal<0) results are both present,
        # running ICWeightedCombiner would average them to ~0 (FLAT) — which is
        # mathematically correct but useless: it suppresses all trades whenever
        # two opposing-direction signals fire simultaneously (e.g. Platt-
        # calibrator OOD bias where both sides output ~0.72 cal_prob).
        #
        # Fix: pick the higher-confidence direction if the conviction delta
        # (|conf_LONG - conf_SHORT|) exceeds min_direction_delta; otherwise
        # return None (genuinely ambiguous bar → FLAT, no position change).
        # ICWeightedCombiner is ONLY applied when all raw_results agree on
        # direction (multiple independent alpha sources for the same side).
        longs = [r for r in raw_results if r.signal > 0]
        shorts = [r for r in raw_results if r.signal < 0]
        if longs and shorts:
            best_long = max(longs, key=lambda r: r.confidence)
            best_short = max(shorts, key=lambda r: r.confidence)
            delta = abs(best_long.confidence - best_short.confidence)
            if delta < self._cfg.min_direction_delta:
                logger.debug(
                    "SignalRunner [%s] predict: direction ambiguous "
                    "(delta=%.4f < threshold=%.4f) → FLAT",
                    symbol, delta, self._cfg.min_direction_delta,
                )
                return None
            winner = (
                best_long if best_long.confidence >= best_short.confidence else best_short
            )
            logger.debug(
                "SignalRunner [%s] predict: direction=%s delta=%.4f",
                symbol, "LONG" if winner.signal > 0 else "SHORT", delta,
            )
            return winner

        # All signals agree on direction — safe to combine with ICWeightedCombiner.
        # Lazy-init combiner with discovered signal IDs
        signal_ids = [r.signal_id for r in raw_results]
        if self._combiner is None or self._combiner.signal_names != signal_ids:
            self._combiner = ICWeightedCombiner(signal_names=signal_ids)

        # Build a one-row signals DataFrame for the combiner
        signal_values = {r.signal_id: r.signal for r in raw_results}
        signals_df = pd.DataFrame([signal_values])

        # Update combiner with forward return if available
        if fwd_return is not None and len(self._return_history) >= 1:
            hist_df = pd.DataFrame(self._signal_history)
            if not hist_df.empty and len(hist_df) >= 5:
                returns_s = pd.Series(self._return_history)
                self._combiner.fit(
                    hist_df.iloc[-self._cfg.ic_lookback:],
                    returns_s.iloc[-self._cfg.ic_lookback:],
                    lookback=self._cfg.ic_lookback,
                )

        # Accumulate history
        for sid, val in signal_values.items():
            if sid not in self._signal_history:
                self._signal_history[sid] = []
            self._signal_history[sid].append(val)
        if fwd_return is not None:
            self._return_history.append(fwd_return)

        combined_series = self._combiner.predict(signals_df)  # type: ignore[union-attr]
        combined_val = float(combined_series.iloc[0])

        # Use the highest-confidence individual result as metadata template
        best = max(raw_results, key=lambda r: r.confidence)
        return SignalResult(
            symbol=best.symbol,
            timestamp=best.timestamp,
            signal=combined_val,
            confidence=best.confidence,
            horizon_bars=best.horizon_bars,
            signal_id=f"{symbol}_combined",
        )

    # ------------------------------------------------------------------
    # AFML event-driven prediction (CUSUM already fired externally)
    # ------------------------------------------------------------------

    def predict_on_event(
        self,
        symbol: str,
        features: pd.DataFrame,
        bar_ts: pd.Timestamp | None = None,
    ) -> SignalResult | None:
        """Run all signals for ``symbol`` using predict_on_event().

        Called by the engine after CUSUMFilter fires and FeaturePipeline has
        computed FRESH features.  Delegates to ModelSignal.predict_on_event()
        which skips the internal CUSUM step (already fired externally).

        Returns the first non-zero SignalResult, or None.
        """
        raw_results: list[SignalResult] = []
        for sig in self._signals:
            sig_symbol = getattr(getattr(sig, "_cfg", None), "symbol", symbol)
            if sig_symbol != symbol:
                continue
            predict_fn = getattr(sig, "predict_on_event", None)
            if predict_fn is None:
                # Fallback for signals that don't implement predict_on_event
                predict_fn = sig.predict
            try:
                # P1-5: forward bar timestamp; tolerate older signal signatures.
                try:
                    result = predict_fn(features, bar_ts=bar_ts)
                except TypeError:
                    result = predict_fn(features)
                if result is not None:
                    raw_results.append(result)
            except Exception as exc:
                # Phase 0: idem predict() - een falend signaal mag niet stil uit
                # het boek vallen; dat verandert de effectieve weging van de
                # overige signalen t.o.v. de backtest.
                raise TradebotContractError(
                    f"SignalRunner: {getattr(sig, 'signal_id', '?')}."
                    f"predict_on_event() faalde: {exc}."
                ) from exc

        if not raw_results:
            return None

        if not self._cfg.use_combiner or len(raw_results) == 1:
            return raw_results[0]

        # ── Direction disambiguation (same logic as predict()) ────────────────
        longs = [r for r in raw_results if r.signal > 0]
        shorts = [r for r in raw_results if r.signal < 0]
        if longs and shorts:
            best_long = max(longs, key=lambda r: r.confidence)
            best_short = max(shorts, key=lambda r: r.confidence)
            delta = abs(best_long.confidence - best_short.confidence)
            if delta < self._cfg.min_direction_delta:
                logger.debug(
                    "SignalRunner [%s] predict_on_event: direction ambiguous "
                    "(delta=%.4f < threshold=%.4f) → FLAT",
                    symbol, delta, self._cfg.min_direction_delta,
                )
                return None
            winner = (
                best_long if best_long.confidence >= best_short.confidence else best_short
            )
            logger.info(
                "SignalRunner [%s] predict_on_event: direction=%s delta=%.4f conf=%.4f",
                symbol, "LONG" if winner.signal > 0 else "SHORT", delta, winner.confidence,
            )
            return winner

        # All signals agree on direction — safe to combine.
        # Same combiner path as predict()
        signal_ids = [r.signal_id for r in raw_results]
        if self._combiner is None or self._combiner.signal_names != signal_ids:
            self._combiner = ICWeightedCombiner(signal_names=signal_ids)

        signal_values = {r.signal_id: r.signal for r in raw_results}
        signals_df = pd.DataFrame([signal_values])
        combined_val = float(self._combiner.predict(signals_df).iloc[0])  # type: ignore[union-attr]

        best = max(raw_results, key=lambda r: r.confidence)
        return SignalResult(
            symbol=best.symbol,
            timestamp=best.timestamp,
            signal=combined_val,
            confidence=best.confidence,
            horizon_bars=best.horizon_bars,
            signal_id=f"{symbol}_combined",
        )
