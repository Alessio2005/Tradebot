"""Meta-Labeling Architecture (López de Prado AFML ch. 10).

Migrated from legacy meta_labeling.py.
Imports from train_regime replaced with tradebot.labeling.* (Wave 5 complete).

Meta-Labeling Architectuur
==========================
Probleem: Directe richting-voorspelling (stijgt/daalt?) is laag-precisie.
Oplossing: Scheiding van *richting* (Scout) en *kwaliteitsoordeel* (Judge).

Werking:
  1. PrimaryScout        — permissieve CUSUM + TrendScan + Triple Barrier.
  2. MetaLabelingEngine  — maakt meta-labels (1=Scout winstgevend, 0=verlies).
  3. Judge (extern)      — ContextualBanditEnsemble getraind op meta-labels.
  4. MetaLabelFilter     — combineert Scout-richting + Judge-kans live/bt.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger("MetaLabeling")


# =============================================================================
# LEAKAGE-GUARD (Item 4): forbid forward-looking labelers as TRAINING SCOUT
# =============================================================================
# TrendScanningLabeler bepaalt richting via OLS over toekomstige bars
# (look-ahead). Als die labels een Judge voeden, leert de Judge de bias van de
# Scout te voorspellen i.p.v. de markt. CausalScout (EMA-crossover) is
# strikt backward-looking en daarom de juiste trainings-Scout.
_FORBIDDEN_TRAIN_SCOUT_CLASSES: frozenset[str] = frozenset({"TrendScanningLabeler"})


def assert_training_scout_is_causal(scout: Any) -> None:
    """Raise ValueError if *scout* is a forward-looking labeler used as Scout.

    Roep aan vóór elke trainingsfase die een Scout-instantie gebruikt om
    meta-labels te genereren. Backward-looking objecten (CausalScout,
    PrimaryScout met EMA-richting) passeren stilzwijgend.

    Args:
        scout : instantie of class. Wordt herkend op `type(scout).__name__`.

    Raises:
        ValueError : als de scout-klasse op de zwarte lijst staat of als
                     PrimaryScout wordt gebruikt zonder causal_only=True.
    """
    cls_name: str = type(scout).__name__ if not isinstance(scout, type) else scout.__name__
    if cls_name in _FORBIDDEN_TRAIN_SCOUT_CLASSES:
        raise ValueError(
            f"Item 4 leakage guard — '{cls_name}' is forward-looking "
            f"en mag NIET als trainings-Scout worden gebruikt. "
            f"Gebruik PrimaryScout(causal_only=True) of CausalScout."
        )
    # Functionele check: PrimaryScout zonder causal_only=True lekt TrendScan
    if cls_name == "PrimaryScout" and hasattr(scout, "causal_only") and not scout.causal_only:
        raise ValueError(
            "Item 4 leakage guard — PrimaryScout zonder causal_only=True "
            "gebruikt TrendScan intern (lookahead). Zet causal_only=True "
            "voor Judge-training of gebruik CausalScout."
        )


# CHIEF AUDIT-FIX (Sim-to-Reality #1):
#   The leakage guard `assert_training_scout_is_causal` only fired when the
#   caller of `build_dataset()` remembered to pass `scout_for_audit=`.  Any
#   forgotten kwarg silently allowed a forward-looking TrendScanningLabeler
#   to feed the Judge → "Silent Killer" (Sharpe collapses in live trading).
#
#   Defence: a strict mode env-var. When TRADEBOT_STRICT_CAUSAL=1 the guard
#   becomes mandatory in build_dataset() and raises if scout_for_audit is
#   None.  Default keeps backward compatibility for existing legitimate calls
#   that pre-validated externally.  Production runs MUST set the env-var.
import os as _os  # local import to avoid top-of-file churn

_STRICT_CAUSAL_ENV: str = "TRADEBOT_STRICT_CAUSAL"


def _strict_causal_required() -> bool:
    """Return True if the production strict-causal gate is active."""
    return _os.environ.get(_STRICT_CAUSAL_ENV, "0").lower() in {"1", "true", "yes"}


# ---------------------------------------------------------------------------
# Helper: Numba-vrije EMA (pandas-backed) — geen extra dep
# ---------------------------------------------------------------------------
def _ema(series: pd.Series, span: int) -> np.ndarray:
    return series.ewm(span=span, adjust=False).mean().to_numpy(dtype=np.float64)


# =============================================================================
# 1. DATA CLASSES
# =============================================================================
@dataclass
class ScoutResult:
    """Output van PrimaryScout.generate() voor één bar-reeks.

    Attributes:
        event_indices : bar-positie van elk CUSUM-event (int32).
        directions    : Scout-richting per event: -1=Short, 0=Flat, 1=Long (int8).
        realized_ret  : netto gerealiseerd rendement via Triple Barrier (float64).
        t1_indices    : exit-bar index per event, benodigd voor purging/embargo (int32).
        t_stats       : t-statistiek van de beste TrendScan-helling per event (float64).
    """

    event_indices: np.ndarray  # shape (n_events,), dtype int32
    directions: np.ndarray     # shape (n_events,), dtype int8
    realized_ret: np.ndarray   # shape (n_events,), dtype float64
    t1_indices: np.ndarray     # shape (n_events,), dtype int32
    t_stats: np.ndarray        # shape (n_events,), dtype float64


@dataclass
class MetaDataset:
    """Trainset klaar voor de Secondary Judge.

    Attributes:
        X_micro    : feature matrix micro-timeframe per event (float32).
        X_meso     : feature matrix meso-timeframe per event (float32).
        X_macro    : feature matrix macro-timeframe per event (float32).
        y_meta     : binaire meta-label — 1=winstgevend, 0=verlies (int8).
        directions : Scout-richting per event (int8), bruikbaar als extra feature.
        weights    : sample weights (uniqueness × returngrootte) (float64).
        t1_indices : exit-bar indices voor CPCV purging (int32).
        timestamps : pd.DatetimeIndex aligned op de event-bars.
    """

    X_micro: np.ndarray         # shape (n_events, n_micro_feats), dtype float32
    X_meso: np.ndarray          # shape (n_events, n_meso_feats),  dtype float32
    X_macro: np.ndarray         # shape (n_events, n_macro_feats), dtype float32
    y_meta: np.ndarray          # shape (n_events,), dtype int8
    directions: np.ndarray      # shape (n_events,), dtype int8
    weights: np.ndarray         # shape (n_events,), dtype float64
    t1_indices: np.ndarray      # shape (n_events,), dtype int32
    timestamps: pd.DatetimeIndex


# =============================================================================
# 2. PRIMARY SCOUT
# =============================================================================
class PrimaryScout:
    """Primair signaalmodel — hoge recall, lage precisie.

    De Scout heeft als enige taak: *veel* potentieel interessante
    trade-momenten opsporen.  Hij is opzettelijk permissief; de
    Secondary Judge filtert de slechte signalen daarna weg.

    Pipeline per aanroep van ``generate()``:

    Stap 1 — CUSUM filter
        Samples enkel bars waar de geaccumuleerde prijsverandering de
        dynamische drempel (ATR × cusum_multiplier) overschrijdt.
        Resultaat: subset ``event_indices``.

    Stap 2 — TrendScanning
        Per event: bereken OLS-helling over [t_min, t_max] bars vooruit.
        Als |t-stat| ≥ min_tstat → Long of Short.
        Als |t-stat| < min_tstat → Flat (geen trade).

    Stap 3 — Triple Barrier
        Entry = open van bar t+1 (geen lookahead bias).
        PT = pt_width × ATR, SL = sl_width × ATR, horizon = best_span.
        Resultaat: realized_ret en t1_indices per event.

    Args:
        t_min            : minimale TrendScan horizon (bars).
        t_max            : maximale TrendScan horizon (bars).
        min_tstat        : t-stat drempel — verlaag dit voor hogere recall.
        pt_width         : Take-Profit als veelvoud van ATR.
        sl_width         : Stop-Loss als veelvoud van ATR.
        cusum_multiplier : ATR-factor voor de CUSUM drempelwaarde.
    """

    def __init__(
        self,
        t_min: int = 3,
        t_max: int = 20,
        min_tstat: float = 1.5,
        pt_width: float = 2.0,
        sl_width: float = 1.0,
        cusum_multiplier: float = 0.5,
        half_spread: float = 0.0,
        # L2-NOTE: cusum_multiplier=0.5 is bewust permissiever dan de 3.5 die
        # get_cusum_events() in train_regime.py gebruikt.
        # Scout-doel: hoge RECALL → veel potentiële events opsporen (7× meer dan filter).
        # train_regime.py-filter: hoge PRECISIE → alleen sterke CUSUM-signalen als event_timestamps.
        # Als event_timestamps in de backtest van de strengere filter komen, evalueert
        # de backtest minder events dan het model getraind is op — houd hiermee rekening
        # bij het vergelijken van Scout-frequentie vs. backtest-frequentie.
        #
        # AUDIT-FIX (N11 — cusum_threshold_multiplier mismatch):
        #   Als Scout-multiplier (bv. 0.5) ≠ training-filter (bv. 1.5), dan
        #   produceert de Scout 3× meer events/dag dan het model getraind is op.
        #   Bandit's gamma_effective = gamma^3 per kalenderdag (3× sneller verouderd
        #   dan getraind). Warning wordt gelogd wanneer de ratio > 2 of < 0.5.
        #   Sync via conf/symbols/{symbol}.yaml: cusum_threshold_multiplier.
        #
        # AUDIT-FIX (K2 — Spread-val): half_spread (decimal, niet bps) wordt aan
        # _triple_barrier_per_event doorgegeven zodat entry/exit op mid-price data
        # (Bybit) realistisch worden afgerekend. Per Bybit perp 2024-2026 medians:
        #   BTC: 0.0001 (0.01%)   ETH: 0.00015   SOL: 0.0004
        # Bij echte bid/ask data (Dukascopy, depth-streams) → laat 0.0.
        training_cusum_multiplier: float | None = None,
        # Optioneel: training-filter multiplier voor N11-mismatch waarschuwing.
        # Stel in op de cusum_threshold_multiplier uit conf/symbols/{symbol}.yaml.
        causal_only: bool = False,
        # Wanneer True: vervang TrendScan-richting door EMA-crossover (geen lookahead).
        # Verplicht op True voor Judge-training. Default False = legacy-gedrag.
    ) -> None:
        self.t_min = t_min
        self.t_max = t_max
        self.min_tstat = min_tstat
        self.pt_width = pt_width
        self.sl_width = sl_width
        self.cusum_multiplier = cusum_multiplier
        self.half_spread: float = float(half_spread)
        self.causal_only: bool = bool(causal_only)

        # AUDIT-FIX (N11): log mismatch Scout vs training cusum multiplier
        if training_cusum_multiplier is not None:
            ratio = float(cusum_multiplier) / max(float(training_cusum_multiplier), 1e-9)
            if ratio < 0.5 or ratio > 2.0:
                logger.warning(
                    "PrimaryScout (N11 — cusum mismatch): Scout cusum_multiplier=%.3f "
                    "vs training_cusum_multiplier=%.3f → ratio=%.2f× (aanbeveling: 0.5–2.0). "
                    "Scout produceert ~%.1f× meer events/dag dan model op getraind is. "
                    "Bandit gamma_effective = gamma^%.1f per kalenderdag. "
                    "Sync via conf/symbols YAML of pas gamma aan.",
                    cusum_multiplier, training_cusum_multiplier, ratio,
                    ratio, ratio,
                )

    # ------------------------------------------------------------------
    @classmethod
    def from_config(
        cls,
        symbol: str,
        cfg_dir: Path | None = None,
        global_spread: float = 0.001,
        **scout_kwargs: Any,
    ) -> PrimaryScout:
        """Factory: bouw PrimaryScout met asset-specifieke half_spread automatisch ingebakken.

        AUDIT-FIX (N18 — K2 silent regression):
          Default half_spread=0.0 herstelt de K2-bug zodra de caller vergeet
          de waarde te passeren.  Deze factory leest spread uit
          conf/symbols/{symbol}.yaml en deelt door 2 voor half_spread.
          Zo kan de bug nooit stil terugkomen.

        Args:
            symbol      : asset-ticker (bv. "BTCUSDT").
            cfg_dir     : pad naar conf/symbols/ map.  None → probeert relatief
                          aan cwd ("conf/symbols").
            global_spread: fallback round-trip spread (decimal) als YAML niet
                          gevonden wordt.
            **scout_kwargs: overige PrimaryScout-kwargs.

        Returns:
            PrimaryScout met half_spread correct ingesteld.
        """
        import yaml  # soft dep: alleen hier nodig

        _cfg_dir = cfg_dir if cfg_dir is not None else Path("conf/symbols")
        sym_path = _cfg_dir / f"{symbol}.yaml"

        round_trip_spread = global_spread
        if sym_path.exists():
            with open(sym_path, encoding="utf-8") as fh:
                sym_cfg: dict[str, Any] = yaml.safe_load(fh) or {}
            round_trip_spread = float(
                sym_cfg.get("spread", sym_cfg.get("training", {}).get("spread", global_spread))
            )
        else:
            logger.warning(
                "PrimaryScout.from_config [%s]: %s niet gevonden. "
                "Gebruik global_spread=%.5f.",
                symbol, sym_path, global_spread,
            )

        half_spread = round_trip_spread / 2.0
        logger.info(
            "PrimaryScout.from_config [%s]: round_trip_spread=%.5f → half_spread=%.5f",
            symbol, round_trip_spread, half_spread,
        )
        # Overschrijf half_spread zodat caller-kwargs het niet kunnen nullen
        scout_kwargs["half_spread"] = half_spread
        return cls(**scout_kwargs)

    # ------------------------------------------------------------------
    def generate(
        self,
        df: pd.DataFrame,
        atr: np.ndarray | None = None,
    ) -> ScoutResult:
        """Genereer Scout-signalen voor alle bars in *df*.

        Args:
            df  : DataFrame met kolommen 'open', 'high', 'low', 'close'.
                  Optioneel: 'bid_high', 'bid_low', 'ask_high', 'ask_low',
                  'feat_vol_gk' (Garman-Klass vol voor ATR).
            atr : Vooraf berekende ATR-array (float64, lengte = len(df)).
                  Als None wordt ATR intern afgeleid.

        Returns:
            ScoutResult met signalen voor bars waar de CUSUM triggert.
        """
        # Migrated: formerly lazy-imported from train_regime.py (Wave 5 complete).
        from .cusum import symmetric_cusum_filter as _symmetric_cusum_filter
        from .trend_scanning import _trend_scan_core
        from .triple_barrier import _triple_barrier_per_event

        close_arr = df["close"].values.astype(np.float64)
        n = len(close_arr)

        # ── ATR ──────────────────────────────────────────────────────────
        atr_arr = self._resolve_atr(df, close_arr, atr)

        # ── Stap 1: CUSUM filter ────────────────────────────────────────
        thresholds = atr_arr * self.cusum_multiplier
        event_indices_raw: np.ndarray = _symmetric_cusum_filter(close_arr, thresholds)

        if len(event_indices_raw) == 0:
            logger.warning("PrimaryScout: CUSUM geeft 0 events terug.")
            return self._empty_result()

        # ── Stap 2: Richting (TrendScan of causal EMA-crossover) ────────────────
        if self.causal_only:
            # Causal EMA-crossover direction (no lookahead)
            fast_ema = pd.Series(close_arr).ewm(span=self.t_min, adjust=False).mean().to_numpy()
            slow_ema = pd.Series(close_arr).ewm(span=self.t_max, adjust=False).mean().to_numpy()
            directions_full = np.zeros(n, dtype=np.int8)
            directions_full[fast_ema > slow_ema] = 1
            directions_full[fast_ema < slow_ema] = -1
            ts_spans = np.full(n, (self.t_min + self.t_max) // 2, dtype=np.int32)
            ts_tstats = np.full(n, np.nan, dtype=np.float64)
        else:
            _, ts_tstats, ts_spans = _trend_scan_core(
                close_arr,
                int(self.t_min),
                int(self.t_max),
                float(self.min_tstat),
            )

            # Richting per bar (volledige reeks)
            directions_full = np.zeros(n, dtype=np.int8)
            directions_full[ts_tstats >= self.min_tstat] = 1
            directions_full[ts_tstats <= -self.min_tstat] = -1

        # Aligneer op events
        event_indices = event_indices_raw.astype(np.int32)
        event_directions = directions_full[event_indices]
        event_tstats = ts_tstats[event_indices]
        event_spans = np.clip(
            ts_spans[event_indices].astype(np.int32),
            self.t_min,
            self.t_max,
        )

        # ── Stap 3: Triple Barrier per richting ─────────────────────────
        open_arr  = df["open"].values.astype(np.float64)  if "open"     in df.columns else close_arr

        # AUDIT-FIX (issue #3 — Spread-val fallback):
        #   Wanneer bid_high/bid_low/ask_high/ask_low ontbreken, vallen we terug
        #   op de mid-market OHLC.  Dit simuleert nul spread — onrealistisch voor
        #   live trading (Bybit perps typisch 0.5–2 bps op de touch).
        #   Gebruik ``spread`` parameter van MetaLabelingEngine om een synthetische
        #   half-spread te bouwen als alternatief, maar log altijd een waarschuwing
        #   zodat de gebruiker weet dat de backtest te optimistisch is.
        _has_bid_ask = (
            "bid_high" in df.columns
            and "bid_low"  in df.columns
            and "ask_high" in df.columns
            and "ask_low"  in df.columns
        )
        if not _has_bid_ask:
            logger.warning(
                "PrimaryScout.generate(): bid_high/bid_low/ask_high/ask_low "
                "kolommen ontbreken — Triple Barrier gebruikt mid-market OHLC "
                "(spread = 0).  Dit onderschat transactiekosten en overschat "
                "realized returns.  Lever microstructuur-kolommen aan of stel "
                "``spread`` in op MetaLabelingEngine voor synthetische correctie.",
            )
        bid_high  = df["bid_high"].values.astype(np.float64) if "bid_high" in df.columns else df["high"].values.astype(np.float64)
        bid_low   = df["bid_low"].values.astype(np.float64)  if "bid_low"  in df.columns else df["low"].values.astype(np.float64)
        ask_high  = df["ask_high"].values.astype(np.float64) if "ask_high" in df.columns else df["high"].values.astype(np.float64)
        ask_low   = df["ask_low"].values.astype(np.float64)  if "ask_low"  in df.columns else df["low"].values.astype(np.float64)

        realized_ret = np.zeros(len(event_indices), dtype=np.float64)
        t1_indices   = np.full(len(event_indices), -1, dtype=np.int32)

        for side_int in (1, -1):
            mask: np.ndarray = event_directions == side_int
            if not mask.any():
                continue

            ev_idx   = event_indices[mask]
            ev_spans = event_spans[mask]

            # AUDIT-FIX (K2): half_spread doorgeven — voorheen default 0.0 →
            # mid-price hits → te optimistische realized_ret én meta-labels.
            # AUDIT-FIX (Round 3): pass required kernel args (execution_delay_bars,
            # jump_ratio_arr, sl_jump_max_mult). Neutral defaults: delay=1,
            # zeros-array = geen wick-aanpassing, mult=0.0 = uitgeschakeld.
            _, t1_sub, ret_sub = _triple_barrier_per_event(
                open_arr, bid_high, bid_low, ask_high, ask_low,
                close_arr, atr_arr,
                ev_idx, ev_spans,
                float(self.pt_width), float(self.sl_width), side_int,
                float(self.half_spread),
                1,
                np.zeros(len(ev_idx), dtype=np.float64),
                0.0,
            )
            realized_ret[mask] = ret_sub
            t1_indices[mask]   = t1_sub

        # Flat events: t1 = event + t_min (minimale horizon)
        flat_mask: np.ndarray = event_directions == 0
        if flat_mask.any():
            t1_indices[flat_mask] = np.minimum(
                event_indices[flat_mask] + self.t_min,
                n - 1,
            ).astype(np.int32)

        logger.info(
            "PrimaryScout: %d events | Long=%d Short=%d Flat=%d",
            len(event_indices),
            int((event_directions == 1).sum()),
            int((event_directions == -1).sum()),
            int(flat_mask.sum()),
        )

        return ScoutResult(
            event_indices=event_indices,
            directions=event_directions,
            realized_ret=realized_ret,
            t1_indices=t1_indices,
            t_stats=event_tstats,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_atr(
        df: pd.DataFrame,
        close_arr: np.ndarray,
        atr: np.ndarray | None,
    ) -> np.ndarray:
        """Geeft een ATR-array terug op basis van externe input of berekening."""
        if atr is not None:
            return np.asarray(atr, dtype=np.float64)
        if "feat_vol_gk" in df.columns:
            return df["feat_vol_gk"].ffill().fillna(1e-5).values.astype(np.float64)
        # Fallback: Wilder's ATR via True Range
        h        = df["high"].values.astype(np.float64)  if "high" in df.columns else close_arr
        low_arr  = df["low"].values.astype(np.float64)   if "low"  in df.columns else close_arr
        prev_c   = np.roll(close_arr, 1)
        tr       = np.maximum(h - low_arr, np.abs(h - prev_c))
        return pd.Series(tr).rolling(14).mean().ffill().fillna(1e-5).to_numpy(dtype=np.float64)

    @staticmethod
    def _empty_result() -> ScoutResult:
        return ScoutResult(
            event_indices=np.array([], dtype=np.int32),
            directions=np.array([], dtype=np.int8),
            realized_ret=np.array([], dtype=np.float64),
            t1_indices=np.array([], dtype=np.int32),
            t_stats=np.array([], dtype=np.float64),
        )


# =============================================================================
# 2b. CAUSAL SCOUT — backward-looking richting (geen lookahead bias)
# =============================================================================
class CausalScout:
    """Scout met uitsluitend backward-looking richting — geschikt voor backtest en live.

    Verschil met PrimaryScout:
      • PrimaryScout gebruikt TrendScanning (OLS over toekomstige bars) voor richting.
        Dit geeft lookahead bias in backtest: de Scout "weet" al wat de markt gaat doen.
      • CausalScout bepaalt richting via EMA-crossover (fast_ema > slow_ema → Long).
        Volledig backward-looking; identiek gedrag in backtest en live productie.

    Triple Barrier (realized_ret, t1_indices) gebruikt wél toekomstige bars — maar dit
    is de *uitkomst* (wat is er werkelijk gebeurd?), niet het *signaal*. Dit is correct
    voor backtesting en labeling.

    Args:
        fast_ema         : Periode voor de snelle EMA (richting-signaal).
        slow_ema         : Periode voor de trage EMA (richting-signaal).
        cusum_multiplier : ATR-factor voor de CUSUM drempelwaarde.
        t_horizon        : Vaste horizon in bars voor Triple Barrier (geen trend-scan).
        pt_width         : Take-Profit als veelvoud van ATR.
        sl_width         : Stop-Loss als veelvoud van ATR.
    """

    def __init__(
        self,
        fast_ema: int = 10,
        slow_ema: int = 50,
        cusum_multiplier: float = 0.5,
        t_horizon: int = 15,
        pt_width: float = 2.0,
        sl_width: float = 1.0,
        half_spread: float = 0.0,
        # AUDIT-FIX (K2): half_spread (decimal) → conservatieve entry/exit op mid-price data.
    ) -> None:
        self.fast_ema = fast_ema
        self.slow_ema = slow_ema
        self.cusum_multiplier = cusum_multiplier
        self.t_horizon = t_horizon
        self.pt_width = pt_width
        self.sl_width = sl_width
        self.half_spread: float = float(half_spread)

    # ------------------------------------------------------------------
    @classmethod
    def from_config(
        cls,
        symbol: str,
        cfg_dir: Path | None = None,
        global_spread: float = 0.001,
        **scout_kwargs: Any,
    ) -> CausalScout:
        """Factory: identiek aan PrimaryScout.from_config maar voor CausalScout.

        AUDIT-FIX (N18): zie PrimaryScout.from_config docstring.
        """
        # Hergebruik PrimaryScout's factory-logica voor YAML-laden
        tmp = PrimaryScout.from_config(
            symbol=symbol,
            cfg_dir=cfg_dir,
            global_spread=global_spread,
            **scout_kwargs,
        )
        # Zet half_spread uit het PrimaryScout-object over naar CausalScout
        scout_kwargs["half_spread"] = tmp.half_spread
        return cls(**scout_kwargs)

    # ------------------------------------------------------------------
    def generate(
        self,
        df: pd.DataFrame,
        atr: np.ndarray | None = None,
    ) -> ScoutResult:
        """Genereer backward-looking Scout-signalen voor alle bars in *df*.

        Args:
            df  : DataFrame met kolommen 'open', 'high', 'low', 'close'.
            atr : Vooraf berekende ATR-array (float64, lengte = len(df)).
                  Als None wordt ATR intern afgeleid.

        Returns:
            ScoutResult met richting bepaald via EMA-crossover (geen lookahead).
        """
        from .cusum import symmetric_cusum_filter as _symmetric_cusum_filter
        from .triple_barrier import _triple_barrier_per_event

        close_arr = df["close"].values.astype(np.float64)
        n = len(close_arr)

        if n < self.slow_ema + 2:
            logger.warning("CausalScout: onvoldoende bars (%d) voor EMA(%d).", n, self.slow_ema)
            return self._empty_result()

        # ── ATR ──────────────────────────────────────────────────────────
        atr_arr = PrimaryScout._resolve_atr(df, close_arr, atr)

        # ── Stap 1: CUSUM filter (backward-looking ✓) ───────────────────
        thresholds = atr_arr * self.cusum_multiplier
        event_indices_raw: np.ndarray = _symmetric_cusum_filter(close_arr, thresholds)

        if len(event_indices_raw) == 0:
            logger.warning("CausalScout: CUSUM geeft 0 events terug.")
            return self._empty_result()

        # ── Stap 2: EMA-crossover richting (backward-looking ✓) ─────────
        close_s = pd.Series(close_arr)
        ema_fast = _ema(close_s, self.fast_ema)
        ema_slow = _ema(close_s, self.slow_ema)

        # +1 = Long (snel boven traag), -1 = Short, 0 = te weinig data (warm-up)
        directions_full = np.zeros(n, dtype=np.int8)
        warmed = np.arange(n) >= self.slow_ema
        directions_full[warmed & (ema_fast > ema_slow)] = 1
        directions_full[warmed & (ema_fast <= ema_slow)] = -1

        event_indices = event_indices_raw.astype(np.int32)
        event_directions = directions_full[event_indices]

        # ── Stap 3: Triple Barrier per richting (uitkomst — mag toekomst gebruiken) ──
        open_arr  = df["open"].values.astype(np.float64)  if "open"     in df.columns else close_arr
        bid_high  = df["bid_high"].values.astype(np.float64) if "bid_high" in df.columns else df["high"].values.astype(np.float64)
        bid_low   = df["bid_low"].values.astype(np.float64)  if "bid_low"  in df.columns else df["low"].values.astype(np.float64)
        ask_high  = df["ask_high"].values.astype(np.float64) if "ask_high" in df.columns else df["high"].values.astype(np.float64)
        ask_low   = df["ask_low"].values.astype(np.float64)  if "ask_low"  in df.columns else df["low"].values.astype(np.float64)

        # Vaste horizon per event (geen trend-scan horizon nodig)
        horizons = np.full(len(event_indices), self.t_horizon, dtype=np.int32)

        realized_ret = np.zeros(len(event_indices), dtype=np.float64)
        t1_indices   = np.full(len(event_indices), -1, dtype=np.int32)

        for side_int in (1, -1):
            mask: np.ndarray = event_directions == side_int
            if not mask.any():
                continue

            ev_idx   = event_indices[mask]
            ev_horiz = horizons[mask]

            # AUDIT-FIX (K2): half_spread doorgegeven — labels en returns nu
            # mid-price-spread-realistisch i.p.v. zero-cost.
            # AUDIT-FIX (Round 3): pass required kernel args (zie eerste call).
            _, t1_sub, ret_sub = _triple_barrier_per_event(
                open_arr, bid_high, bid_low, ask_high, ask_low,
                close_arr, atr_arr,
                ev_idx, ev_horiz,
                float(self.pt_width), float(self.sl_width), side_int,
                float(self.half_spread),
                1,
                np.zeros(len(ev_idx), dtype=np.float64),
                0.0,
            )
            realized_ret[mask] = ret_sub
            t1_indices[mask]   = t1_sub

        # Flat events (EMA nog in warm-up): t1 = event + horizon
        flat_mask: np.ndarray = event_directions == 0
        if flat_mask.any():
            t1_indices[flat_mask] = np.minimum(
                event_indices[flat_mask] + self.t_horizon,
                n - 1,
            ).astype(np.int32)

        # t_stats = NaN (geen trend-scan → geen t-statistiek beschikbaar)
        t_stats_arr = np.full(len(event_indices), np.nan, dtype=np.float64)

        logger.info(
            "CausalScout: %d events | Long=%d Short=%d Flat=%d (EMA %d/%d)",
            len(event_indices),
            int((event_directions == 1).sum()),
            int((event_directions == -1).sum()),
            int(flat_mask.sum()),
            self.fast_ema, self.slow_ema,
        )

        return ScoutResult(
            event_indices=event_indices,
            directions=event_directions,
            realized_ret=realized_ret,
            t1_indices=t1_indices,
            t_stats=t_stats_arr,
        )

    @staticmethod
    def _empty_result() -> ScoutResult:
        return ScoutResult(
            event_indices=np.array([], dtype=np.int32),
            directions=np.array([], dtype=np.int8),
            realized_ret=np.array([], dtype=np.float64),
            t1_indices=np.array([], dtype=np.int32),
            t_stats=np.array([], dtype=np.float64),
        )


# =============================================================================
# 3. META-LABELING ENGINE
# =============================================================================
class MetaLabelingEngine:
    """Zet Scout-resultaten om naar meta-labels en bouwt de Judge-dataset.

    Meta-label definitie (López de Prado, AFML ch. 10):
        y = 1  als de Scout-trade winstgevend was (realized_ret > profit_threshold)
        y = 0  als de Scout-trade verliesgevend was

    De Secondary Judge leert NIET richting te voorspellen, maar de kans dat
    de Scout gelijk heeft.  Bij een lage Judge-kans wordt de trade geblokkeerd —
    ook als de Scout een signaal geeft.  Dit lost het overtrading-probleem op.

    Args:
        profit_threshold : minimaal netto rendement voor label=1.
                           0.0 = elke positieve return is een win.
        min_tstat_filter : events met |t-stat| < drempel krijgen label=0.
                           0.0 = geen extra filter (alles wordt meegenomen).
        include_flat     : als True worden Flat-events als label=0 opgenomen.
        spread           : **AUDIT-FIX (issue #3 — Spread-val)**
                           Half-spread als fractionele return (decimal, niet bps).
                           Standaard 0.0 = nul spread (gebruik dit ALLEEN wanneer
                           de Triple Barrier al bid/ask-gecorrigeerde prijzen
                           gebruikt via ``bid_high``/``ask_low`` kolommen in de
                           DataFrame).

                           Wanneer ``spread > 0`` en de DataFrame géén
                           bid_high/ask_low kolommen heeft:
                             * profit_threshold wordt automatisch verhoogd tot
                               max(profit_threshold, 2 × spread) zodat round-trip
                               transactiekosten zijn verwerkt in de labeldrempel.
                             * De Triple Barrier gebruikt mid-market OHLC met een
                               waarschuwing (zie PrimaryScout.generate()).

                           Vuistregels per asset (Bybit perpetuals, 2024):
                             * BTC:  spread ≈ 0.0002  (0.02% half-spread)
                             * ETH:  spread ≈ 0.0003
                             * SOL:  spread ≈ 0.0008  (minder liquide)

                           Laat op 0.0 als je echte bid/ask microstructuurdata
                           in de DataFrame hebt — dan is de spread al impliciet
                           verwerkt in de barrierprijzen.
    """

    def __init__(
        self,
        profit_threshold: float = 0.0,
        min_tstat_filter: float = 0.0,
        include_flat: bool = False,
        spread: float = 0.0,
        barrier_uses_spread: bool = False,
        vwpf_min_participation: float = 0.0,
        vwpf_max_participation: float = 0.20,
        # AUDIT-FIX (Issue 3 — Volume-Weighted Probability of Filling):
        #   Triple Barrier-hits in de backtest tellen vaak als win zonder dat
        #   de markt op dat moment de orderomvang kon absorberen.  In MFT is
        #   dat dodelijk: live komt de fill nooit door, of duwt jouw eigen
        #   order de prijs voorbij de barrier.  Met VWPF tellen barrier-hits
        #   alleen als win wanneer de orderomvang in het [vwpf_min, vwpf_max]
        #   participation-bereik van het bar-volume valt.
        #
        #     min < ratio < max  →  realistisch fillbaar (label = 1 mag)
        #     ratio < min        →  illiquide (label = 0)
        #     ratio > max        →  jouw eigen impact triggert de hit (label = 0)
        #
        #   Default vwpf_min=0.0 → backward compat (niemand wordt geblokkeerd
        #   tenzij caller een ondergrens > 0 instelt).
        # AUDIT-FIX (N25 — Spread Double-Booking):
        #   Wanneer PrimaryScout/CausalScout met half_spread > 0 is gebouwd
        #   (K2-fix), zijn barrier-prices al gecorrigeerd voor spread-kosten.
        #   De oude code telde dan NOGMAALS 2×spread op bij profit_threshold →
        #   effectief 4× spread geboekt op label-niveau.
        #
        #   Zet barrier_uses_spread=True als de Scout is gebouwd met
        #   half_spread > 0 (via PrimaryScout.from_config of expliciet).
        #   Dan wordt profit_threshold NIET verhoogd met 2×spread — de
        #   drempel blijft puur winst t.o.v. reeds-gecorrigeerde barrier.
        #
        #   barrier_uses_spread=False (default) = legacy-gedrag:
        #   mid-price barriers + 2×spread threshold (correct voor bare
        #   mid-price data zonder half_spread in Scout).
    ) -> None:
        # THRESHOLD-FIX: profit_threshold=0.0 labelt elke minimale positieve return
        # als "win", inclusief moves die na transaction costs een verlies zijn.
        # Als spread > 0 én barrier géén spread bevat: verhoog tot 2×spread.
        # Als barrier AL spread bevat (barrier_uses_spread=True): niet verhogen.
        if spread > 0.0 and not barrier_uses_spread:
            # Mid-price fallback: kosten nog niet in barrier → voeg toe aan drempel
            effective_threshold = max(profit_threshold, 2.0 * spread)
        else:
            # Spread al in barrier-prices verwerkt (of spread=0): geen dubbelboeking
            effective_threshold = max(profit_threshold, 0.0)
        self.profit_threshold = effective_threshold
        self.min_tstat_filter = min_tstat_filter
        self.include_flat     = include_flat
        self._barrier_uses_spread: bool = barrier_uses_spread
        self.vwpf_min_participation: float = float(vwpf_min_participation)
        self.vwpf_max_participation: float = float(vwpf_max_participation)

    # ------------------------------------------------------------------
    def make_meta_labels(
        self,
        scout: ScoutResult,
        *,
        bar_volumes: np.ndarray | None = None,
        order_sizes: np.ndarray | None = None,
        funding_per_event: np.ndarray | None = None,
        signed_directions_for_funding: np.ndarray | None = None,
    ) -> np.ndarray:
        """Bereken binaire meta-labels vanuit een ScoutResult.

        Args:
            scout       : Output van PrimaryScout.generate().
            bar_volumes : Optioneel — bar-volume op het event-moment
                          (zelfde unit als ``order_sizes``).  Vereist
                          voor VWPF execution-aware labels.
            order_sizes : Optioneel — voorgenomen orderomvang per event.
            funding_per_event : Optioneel — som van funding-rates die over
                          de hold-tijd van elk event betaald (long) of
                          ontvangen (short) zou worden.  Decimal per
                          event (bv. 0.0008 = 8 bps cumulatief).
            signed_directions_for_funding : Optioneel — 1/-1 per event;
                          long betaalt funding bij positieve rate, short
                          ontvangt het.  Default: ``scout.directions``.

        Returns:
            int8 array van meta-labels — 1=win, 0=verlies — per event.

        AUDIT-FIX (Issue 3 — VWPF execution-aware labels):
            Wanneer ``bar_volumes`` én ``order_sizes`` zijn opgegeven én
            ``self.vwpf_min_participation > 0`` of ``vwpf_max_participation < 1``,
            wordt elke barrier-hit gevalideerd door :func:`vwpf_check` uit
            ``market_impact``.  Hits buiten het fillbare participation-bereik
            worden gedegradeerd tot label=0 (verlies) — anders zou de Judge
            leren op trades die in productie niet uitvoerbaar zijn.
        """
        n  = len(scout.event_indices)
        y  = np.zeros(n, dtype=np.int8)

        # AUDIT-FIX (Issue 5 — Funding Rate Drag & Carry Risk):
        #   Y_adj = Y - signed_dir * funding_cum.  Bij long-positie in een
        #   oververhitte markt betaal je elke 8 uur funding; een long-trade
        #   die +2% stijgt maar 2.5% funding kost is een netto verlies.
        #   Hierdoor leert de Judge expliciet: "lange perpetuals in
        #   high-funding-regimes zijn niet winstgevend zonder veel grotere
        #   alpha".  Funding wordt afgetrokken van de realized return
        #   VOORDAT we tegen profit_threshold testen.
        _funding_active = funding_per_event is not None
        funding_arr: np.ndarray | None = None
        funding_dir: np.ndarray | None = None
        if _funding_active:
            funding_arr = np.asarray(funding_per_event, dtype=np.float64)
            if funding_arr.shape[0] != n:
                logger.warning(
                    "MetaLabelingEngine: funding_per_event lengte (%d) ≠ "
                    "events (%d) — funding adjustment uitgeschakeld.",
                    funding_arr.shape[0], n,
                )
                _funding_active = False
                funding_arr = None
            elif signed_directions_for_funding is not None:
                funding_dir = np.asarray(
                    signed_directions_for_funding, dtype=np.float64
                )
            else:
                funding_dir = scout.directions.astype(np.float64)

        # Lazy import — alleen als VWPF actief is
        _vwpf_active = (
            bar_volumes is not None
            and order_sizes is not None
            and (self.vwpf_min_participation > 0.0 or self.vwpf_max_participation < 1.0)
        )
        _vwpf_check = None
        if _vwpf_active:
            try:
                from ..execution.market_impact import vwpf_check as _vwpf_check_imported
                _vwpf_check = _vwpf_check_imported
            except ImportError:
                logger.warning(
                    "MetaLabelingEngine: VWPF gevraagd maar market_impact "
                    "module niet beschikbaar — VWPF-gate uitgeschakeld."
                )
                _vwpf_active = False

        n_vwpf_failed: int = 0
        n_funding_flipped: int = 0
        for i in range(n):
            direction = int(scout.directions[i])
            ret       = float(scout.realized_ret[i])
            tstat     = float(scout.t_stats[i])

            # Flat events tellen niet mee tenzij expliciet ingeschakeld
            if direction == 0 and not self.include_flat:
                continue

            # t-stat filter: te laag = geen betrouwbaar signaal
            if self.min_tstat_filter > 0.0 and abs(tstat) < self.min_tstat_filter:
                continue

            # AUDIT-FIX (Issue 5 — Subtract funding cost van realized_ret)
            ret_adj = ret
            if _funding_active and funding_arr is not None and funding_dir is not None:
                # signed * funding (long betaalt bij +funding, short ontvangt)
                ret_adj = ret - float(funding_dir[i]) * float(funding_arr[i])
                if (ret > self.profit_threshold) and (ret_adj <= self.profit_threshold):
                    n_funding_flipped += 1

            # Win als return (na funding) de drempel overschrijdt
            if ret_adj > self.profit_threshold:
                # AUDIT-FIX (Issue 3 — VWPF gate):
                # Hits zonder voldoende liquiditeit blijven label=0.
                if _vwpf_active and _vwpf_check is not None:
                    bar_vol_i  = float(bar_volumes[i])      # type: ignore[index]
                    ord_size_i = float(order_sizes[i])      # type: ignore[index]
                    fillable, _p = _vwpf_check(
                        target_size=ord_size_i,
                        bar_volume=bar_vol_i,
                        min_participation_ratio=self.vwpf_min_participation,
                        max_participation_ratio=self.vwpf_max_participation,
                    )
                    if not fillable:
                        n_vwpf_failed += 1
                        continue
                y[i] = 1

        win_pct = float(y.mean()) * 100.0 if n > 0 else 0.0
        logger.info(
            "Meta-labels: %d events | wins=%d (%.1f%%) | losses=%d | "
            "vwpf_blocked=%d | funding_flipped_to_loss=%d",
            n, int(y.sum()), win_pct, int((y == 0).sum()),
            n_vwpf_failed, n_funding_flipped,
        )
        return y

    # ------------------------------------------------------------------
    def build_dataset(
        self,
        scout: ScoutResult,
        X_micro: np.ndarray,
        X_meso: np.ndarray,
        X_macro: np.ndarray,
        df_index: pd.DatetimeIndex,
        sample_weight_fn: Callable[..., np.ndarray] | None = None,
        scout_for_audit: Any | None = None,
        train_indices: np.ndarray | None = None,
    ) -> MetaDataset:
        """Bouw de volledige trainset voor de Judge.

        Alignt de feature matrices op de event_indices zodat X[k] de features
        beschrijft van de bar op het moment dat de Scout een signaal gaf.

        Args:
            scout           : output van PrimaryScout.generate().
            X_micro         : (n_bars, n_feats)       micro feature matrix.
            X_meso          : (n_bars, n_meso_feats)   meso feature matrix.
            X_macro         : (n_bars, n_macro_feats)  macro feature matrix.
            df_index        : DatetimeIndex van de originele bar-reeks.
            sample_weight_fn: Callable(timestamps, uniqueness, returns) → ndarray.
                              Als None: uniforme gewichten (allemaal 1.0).
            train_indices   : integer posities van train-events binnen de CPCV-fold
                              (np.ndarray, int64). Wanneer opgegeven wordt de leakage-
                              safe `get_average_uniqueness_per_fold` gebruikt; anders
                              de globale variant met een waarschuwing (SK-2).
            scout_for_audit : Optionele Scout-instantie of -class. Indien
                              opgegeven wordt `assert_training_scout_is_causal`
                              uitgevoerd zodat per ongeluk een
                              `TrendScanningLabeler` als Scout het training-
                              pad niet kan binnenkomen (Item 4).

        Returns:
            MetaDataset klaar voor RegimeCatAgent.train() of CPCV cross-validatie.
        """
        # CHIEF AUDIT-FIX (Sim-to-Reality #1): enforce leakage guard in strict mode.
        # Production should run with TRADEBOT_STRICT_CAUSAL=1.  When set, the
        # `scout_for_audit` argument is REQUIRED — forgetting it raises rather
        # than silently skipping the lookahead check.
        if scout_for_audit is None:
            if _strict_causal_required():
                raise ValueError(
                    "Item 4 leakage guard (STRICT MODE) — build_dataset() called "
                    "without scout_for_audit while TRADEBOT_STRICT_CAUSAL=1. "
                    "Pass scout_for_audit=<your Scout instance> so the causal-Scout "
                    "check can run.  Forgetting this kwarg silently allowed "
                    "lookahead-Scout training in legacy callers."
                )
            else:
                logger.warning(
                    "build_dataset: scout_for_audit not supplied. "
                    "Forward-looking Scout detection is DISABLED for this call. "
                    "Set TRADEBOT_STRICT_CAUSAL=1 in production to make this fatal."
                )
        else:
            assert_training_scout_is_causal(scout_for_audit)
        n_bars = len(df_index)
        idx    = scout.event_indices
        idx_clipped = np.clip(idx, 0, n_bars - 1)

        y_meta = self.make_meta_labels(scout)

        # L1-FIX: Verwijder flat events expliciet uit de dataset als include_flat=False.
        # make_meta_labels() slaat flat events over (continue), zodat ze y=0 blijven.
        # Flat events zijn echter geen directionale trades — ze als y=0 in de Judge-
        # trainset laten staan voegt ruis toe (onterechte negatieve voorbeelden).
        #
        # SHAPE-FIX: directions_filtered en t1_indices_filtered worden in BEIDE takken
        # aangemaakt zodat de return-statement altijd gefilterde arrays terug kan geven.
        # Vorige versie retourneerde scout.directions / scout.t1_indices (ongefiltererd,
        # lengte n_events), terwijl X_micro/y_meta/weights/timestamps al gefilterd waren
        # op n_active — dit veroorzaakte stille shape-mismatches downstream.
        if not self.include_flat:
            active_mask: np.ndarray = scout.directions != 0  # behoud alleen Long/Short
            idx                     = idx[active_mask]
            idx_clipped             = idx_clipped[active_mask]
            y_meta                  = y_meta[active_mask]
            realized_ret_filtered   = scout.realized_ret[active_mask]
            t1_indices_filtered     = scout.t1_indices[active_mask]
            directions_filtered     = scout.directions[active_mask]      # SHAPE-FIX
        else:
            realized_ret_filtered   = scout.realized_ret
            t1_indices_filtered     = scout.t1_indices
            directions_filtered     = scout.directions                   # SHAPE-FIX

        # Feature extractie — selecteer rijen op event-posities
        def _slice(mat: np.ndarray) -> np.ndarray:
            if mat.shape[0] == 0:
                return np.zeros((len(idx), 0), dtype=np.float32)
            return mat[idx_clipped].astype(np.float32)

        Xm = _slice(X_micro)
        Xs = _slice(X_meso)
        Xc = _slice(X_macro)

        # Sample weights
        if sample_weight_fn is not None:
            timestamps_pd = pd.Series(df_index[idx_clipped])

            # M1-FIX: Bereken echte uniqueness via get_average_uniqueness() in plaats van
            # de placeholder np.ones(...) die elke sample gelijk woog (uniqueness=1.0).
            # Lazy import voorkomt circulaire import (meta_labeling → train_regime → meta_labeling).
            #
            # SK-2 FIX: Gebruik get_average_uniqueness_per_fold() wanneer train_indices
            # beschikbaar is (CPCV-loop). De globale variant lekt OOS test-fold events in
            # de train sample-weights — events die overlappen met test-events krijgen lagere
            # uniqueness, ook al zijn die test-events in productie onbekend.
            # Als train_indices=None (standalone use), val terug op globale variant met warning.
            try:
                from ..cv.uniqueness import (
                    get_average_uniqueness,
                    get_average_uniqueness_per_fold,
                )
                t1_series = pd.Series(
                    np.clip(t1_indices_filtered, 0, n_bars - 1).astype(np.int32),
                    index=df_index[idx_clipped],
                )
                if train_indices is not None:
                    real_uniqueness = get_average_uniqueness_per_fold(
                        df_index, t1_series, train_indices
                    )
                else:
                    logger.warning(
                        "build_dataset: train_indices niet opgegeven — globale uniqueness "
                        "gebruikt (leaky voor CPCV). Geef train_indices mee vanuit de "
                        "CPCV-split om leakage in sample-weights te vermijden."
                    )
                    real_uniqueness = get_average_uniqueness(df_index, t1_series)
            except ImportError:
                logger.warning(
                    "get_average_uniqueness niet beschikbaar — terugvallen op uniforme gewichten."
                )
                real_uniqueness = np.ones(len(idx), dtype=np.float64)

            w = sample_weight_fn(
                timestamps_pd,
                real_uniqueness,        # ← M1-FIX: echte uniqueness ipv placeholder 1.0
                realized_ret_filtered,
            )
            w = np.asarray(w, dtype=np.float64)
        else:
            w = np.ones(len(idx), dtype=np.float64)

        logger.info(
            "MetaDataset gebouwd: %d events | micro=%s meso=%s macro=%s",
            len(idx), Xm.shape, Xs.shape, Xc.shape,
        )

        return MetaDataset(
            X_micro=Xm,
            X_meso=Xs,
            X_macro=Xc,
            y_meta=y_meta,
            directions=directions_filtered,   # SHAPE-FIX: was scout.directions (ongefilterd)
            weights=w,
            t1_indices=t1_indices_filtered,   # SHAPE-FIX: was scout.t1_indices (ongefilterd)
            timestamps=df_index[idx_clipped],
        )

    # ------------------------------------------------------------------
    @staticmethod
    def filter_active_events(scout: ScoutResult) -> ScoutResult:
        """Verwijder Flat-events — behoud alleen Long en Short.

        Handig als de Judge enkel op directionale signalen traint
        en Flat-events ruis toevoegen aan de dataset.

        Args:
            scout: origineel ScoutResult van PrimaryScout.generate().

        Returns:
            Gefilterd ScoutResult zonder Flat-events (direction == 0).
        """
        active: np.ndarray = scout.directions != 0
        return ScoutResult(
            event_indices=scout.event_indices[active],
            directions=scout.directions[active],
            realized_ret=scout.realized_ret[active],
            t1_indices=scout.t1_indices[active],
            t_stats=scout.t_stats[active],
        )


# =============================================================================
# 4. META LABEL FILTER — live / backtesting inference
# =============================================================================
class MetaLabelFilter:
    """Combineert Scout-richting en Judge-kans tot een handelsbeslissing.

    De filter is de kern van de meta-labeling architectuur bij inference:

        Scout geeft richting →  Judge geeft prob_win
            prob_win ≥ threshold  →  trade doorgeven
            prob_win <  threshold →  trade blokkeren (Flat)

    Args:
        prob_threshold             : minimale Judge-kans voor goedkeuring.
        require_direction_agreement: als True, wordt de trade alleen
            goedgekeurd als Scout en Judge dezelfde kant op wijzen.
            (Alleen relevant als de Judge apart Long- en Short-modellen heeft.)
    """

    def __init__(
        self,
        prob_threshold: float = 0.35,
        require_direction_agreement: bool = False,
    ) -> None:
        self.prob_threshold              = prob_threshold
        self.require_direction_agreement = require_direction_agreement

    # ------------------------------------------------------------------
    def decide(
        self,
        scout_direction: int,
        judge_prob_win: float,
        judge_side_detected: str | None = None,
    ) -> int:
        """Geef de uiteindelijke beslissing terug voor één bar.

        Args:
            scout_direction    : -1 / 0 / 1 vanuit PrimaryScout.
            judge_prob_win     : kans dat de trade winstgevend is  [0.0, 1.0].
            judge_side_detected: 'LONG' of 'SHORT' zoals gerapporteerd door
                                  RegimeEnsembleCat / ContextualBanditEnsemble.

        Returns:
            1  = Long goedgekeurd
            -1 = Short goedgekeurd
            0  = Geblokkeerd (Judge filtert of Scout is Flat)
        """
        if scout_direction == 0:
            return 0

        if judge_prob_win < self.prob_threshold:
            return 0

        if self.require_direction_agreement and judge_side_detected is not None:
            judge_long = judge_side_detected.upper() == "LONG"
            if scout_direction == 1 and not judge_long:
                return 0
            if scout_direction == -1 and judge_long:
                return 0

        return int(scout_direction)

    # ------------------------------------------------------------------
    def batch_decide(
        self,
        scout_directions: np.ndarray,
        judge_probs_win: np.ndarray,
    ) -> np.ndarray:
        """Vectorised versie van decide() voor backtesting over een reeks bars.

        Args:
            scout_directions : int8 array (-1/0/1) per bar.
            judge_probs_win  : float64 array [0, 1] per bar.

        Returns:
            int8 array met uiteindelijke handelsbeslissingen per bar.
        """
        out: np.ndarray = np.zeros(len(scout_directions), dtype=np.int8)
        active: np.ndarray = scout_directions != 0
        passed: np.ndarray = active & (judge_probs_win >= self.prob_threshold)
        out[passed] = scout_directions[passed]
        return out

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"MetaLabelFilter("
            f"prob_threshold={self.prob_threshold}, "
            f"require_direction_agreement={self.require_direction_agreement})"
        )
