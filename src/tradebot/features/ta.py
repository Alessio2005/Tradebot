"""FeatureEngineer — TA + FFD + microstructure feature pipeline.

Migrated from legacy ta_features.py. Single responsibility:
  compute_features(df, timeframe) -> pd.DataFrame

All Numba kernels live in features._ta_kernels (single source of truth).
"""
from __future__ import annotations

import logging
import math
from typing import cast

import numpy as np
import pandas as pd

from ..volatility import get_garman_klass_volatility, get_jump_adjusted_volatility
from ._ta_kernels import (
    DEFAULT_BASE_WINDOW,
    DEFAULT_FFD_THRESHOLD,
    DEFAULT_VOLUME_CANDIDATES,
    apply_frac_diff_mask_fast,
    calculate_shannon_entropy,
    compute_dynamic_ffd_threshold,
    compute_hilbert_phase,
    get_optimal_d,
    get_weights_ffd,
    roll_autocorr_lag1,
    roll_autocorr_spearman,
)

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Engine voor het genereren van geavanceerde technische features."""

    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or {}
        self.window = int(self.cfg.get("base_window", DEFAULT_BASE_WINDOW))
        # Base FFD weight truncation threshold. Wordt per timeframe dynamisch
        # herberekend (compute_dynamic_ffd_threshold) om memory-leaks tijdens
        # macro-shocks te voorkomen. Zie _add_ffd_features.
        self.ffd_thres: float = float(
            self.cfg.get("ffd_thres", DEFAULT_FFD_THRESHOLD)
        )
        self._use_dynamic_ffd_thres: bool = bool(
            self.cfg.get("dynamic_ffd_threshold", True)
        )
        # Fractie van de data die voor FFD-kalibratie gebruikt wordt.
        # d wordt EENMALIG per timeframe bepaald en daarna GECACHT.
        # Caching voorkomt dat d bij elke aanroep herberekend wordt, wat
        # leidt tot niet-deterministische feature-waarden in productie.
        self._ffd_calib_frac: float = float(
            self.cfg.get("ffd_calib_frac", 0.80)
        )
        # Cache: {timeframe_str: float}  — gevuld bij eerste add_features()-aanroep
        self._fixed_d_cache: dict[str, float] = {}
        # Cache voor de effectieve (dynamische) threshold per timeframe zodat
        # de productie-inferentie exact dezelfde truncation reproduceert als
        # de training-run.
        self._ffd_thres_cache: dict[str, float] = {}

    # ------------------------------------------------------------------
    # CPCV-LEAKAGE-FIX (Item 1):
    # Per-fold FFD d-calibratie. De default cache-pad in _add_ffd_features
    # gebruikt de eerste 60% van de data — als CPCV test-folds vroeg in de
    # tijdlijn vallen, lekt macro-structuur uit de toekomst terug naar die
    # vroege folds. Roep `clear_d_cache()` + `fit_d_on_train_indices(...)`
    # AAN HET BEGIN VAN ELKE FOLD aan zodat d uitsluitend op train-bars
    # wordt gekalibreerd.
    # ------------------------------------------------------------------
    def clear_d_cache(self, timeframe: str | None = None) -> None:
        """Maak de FFD d- en threshold-cache leeg.

        Gebruik dit aan het begin van elke CPCV-fold om te voorkomen dat
        een d-waarde die op een vorige fold-train-set is gekalibreerd in
        de huidige fold blijft staan.

        Args:
            timeframe : Specifieke timeframe-key (bv. "micro"). None = alle.
        """
        if timeframe is None:
            self._fixed_d_cache.clear()
            self._ffd_thres_cache.clear()
            return
        self._fixed_d_cache.pop(timeframe, None)
        self._ffd_thres_cache.pop(timeframe, None)
        # Volume-varianten van de cache-keys (zie _add_ffd_volume_features).
        self._fixed_d_cache.pop(f"{timeframe}_vol", None)
        self._ffd_thres_cache.pop(f"{timeframe}_vol", None)
        for tau_key in [k for k in self._ffd_thres_cache if k.startswith(f"{timeframe}_")]:
            self._ffd_thres_cache.pop(tau_key, None)

    # ------------------------------------------------------------------
    # AUDIT-FIX (Issue 1 — FFD Memory & Purging Paradox):
    #   FFD-gewichten hebben wiskundig een **oneindige** staart.  Wanneer
    #   we ze afkappen op |w_k| < tau, blijft de effectieve memory-lengte
    #   gelijk aan de lengte van het overlevende gewichtsblok.  Als dat
    #   blok langer is dan het CPCV-embargo, lekt informatie van de
    #   test-set naar de train-set via het FFD-feature.
    #
    #   `get_ffd_max_lag(timeframe)` retourneert de **dynamische** lag-
    #   diepte (= lengte van de FFD-gewichten-array nadat tau is
    #   toegepast) per timeframe.  Caller kan deze waarde gebruiken om
    #   ``CombinatorialPurgedCV`` op een veilige `embargo_bars` te zetten:
    #
    #       max_lag = max(
    #           pipe.ta_engineer.get_ffd_max_lag("micro"),
    #           pipe.ta_engineer.get_ffd_max_lag("meso"),
    #           pipe.ta_engineer.get_ffd_max_lag("macro"),
    #       )
    #       cv = CombinatorialPurgedCV(
    #           embargo_bars=max(t_max_bars + 1, max_lag + 1),
    #           feature_max_lag=max_lag,
    #       )
    # ------------------------------------------------------------------
    def get_ffd_max_lag(
        self,
        timeframe: str = "micro",
        *,
        max_search_lim: int = 5000,
        epsilon: float = 1e-6,
    ) -> int:
        """Effectieve FFD-memory-diepte per timeframe (in bars).

        Berekent het aantal gewichten dat overblijft nadat de gecachete
        truncation-threshold τ is toegepast.  Wanneer een feature deze
        lag-diepte bereikt, neemt de feature géén informatie meer mee
        van bars verder terug (|w_k| < τ).

        Args:
            timeframe:    "micro" / "meso" / "macro" (zelfde sleutel als
                          ``self._fixed_d_cache``).
            max_search_lim: bovengrens op gewicht-zoeklengte.
            epsilon:      extra absolute drempel — neem het maximum van
                          τ_dyn en epsilon zodat je nooit oneindig zoekt
                          wanneer τ extreem klein is gekozen.

        Returns:
            Aantal effectieve FFD-gewichten (≥ 1).  Wanneer d of τ niet
            in cache staan: fallback op een conservatieve 256-bars-grens.
        """
        d = self._fixed_d_cache.get(timeframe)
        tau = self._ffd_thres_cache.get(timeframe)
        if d is None or tau is None:
            # Conservatieve fallback: 256 bars memory-budget.
            return 256
        tau_eff = max(float(tau), float(epsilon))
        weights = get_weights_ffd(float(d), tau_eff, int(max_search_lim))
        return int(weights.shape[0])

    def get_max_feature_lag(self, *, max_search_lim: int = 5000) -> int:
        """Veilige bovengrens over alle gecachete timeframes.

        Returns:
            Maximum FFD-lag-diepte over alle timeframes in de cache.
        """
        if not self._fixed_d_cache:
            return 256
        return max(
            self.get_ffd_max_lag(tf, max_search_lim=max_search_lim)
            for tf in self._fixed_d_cache
        )

    def fit_d_on_train_indices(
        self,
        df: pd.DataFrame,
        train_indices: np.ndarray,
        timeframe: str = "micro",
        max_d: float = 1.0,
        step: float = 0.05,
        p_thres: float = 0.05,
    ) -> float:
        """Kalibreer FFD d UITSLUITEND op de opgegeven train-indices.

        Schrijft het resultaat naar `self._fixed_d_cache[timeframe]`
        zodat een daaropvolgende `add_features()`-aanroep deterministisch
        dezelfde d-waarde gebruikt — zonder lekkage van toekomstige bars.

        Args:
            df            : Volledige micro/meso/macro DataFrame met "close".
            train_indices : Integer-posities (np.ndarray) van train-bars in df.
            timeframe     : Cache-key (bv. "micro", "meso", "macro").
            max_d, step,
            p_thres       : doorgegeven aan get_optimal_d.

        Returns:
            De gecalibreerde d. NaN-fallback wordt naar 0.4 herleid.
        """
        if "close" not in df.columns:
            raise KeyError("fit_d_on_train_indices vereist een 'close' kolom in df.")
        if len(train_indices) < 30:
            logger.warning(
                "fit_d_on_train_indices [%s]: te weinig train-bars (%d), "
                "fallback naar d=0.4.", timeframe, len(train_indices),
            )
            self._fixed_d_cache[timeframe] = 0.4
            return 0.4

        # Effectieve threshold per timeframe (consistent met _add_ffd_features).
        train_close: pd.Series = df["close"].iloc[train_indices]
        log_price_train: np.ndarray = np.log(
            train_close.clip(lower=1e-9).to_numpy(dtype=np.float64)
        )
        if self._use_dynamic_ffd_thres:
            ret_np = np.diff(log_price_train, prepend=log_price_train[0])
            tau_dyn: float = float(compute_dynamic_ffd_threshold(
                ret_np, base=self.ffd_thres,
            ))
        else:
            tau_dyn = float(self.ffd_thres)
        self._ffd_thres_cache[timeframe] = tau_dyn

        d_candidate: float = get_optimal_d(
            train_close,
            max_d=max_d,
            step=step,
            p_thres=p_thres,
            ffd_thres=tau_dyn,
        )
        if np.isnan(d_candidate):
            d_candidate = 0.4
            logger.warning(
                "fit_d_on_train_indices [%s]: get_optimal_d gaf NaN, "
                "fallback d=0.4 (n_train=%d, tau=%.2e).",
                timeframe, len(train_indices), tau_dyn,
            )
        self._fixed_d_cache[timeframe] = float(d_candidate)
        logger.info(
            "FFD [%s] (CPCV fold-cal): d=%.2f op %d train-bars (tau=%.2e).",
            timeframe, d_candidate, len(train_indices), tau_dyn,
        )
        return float(d_candidate)

    def _get_volume_col(self, df: pd.DataFrame) -> str:
        """Identificeert de meest relevante volumekolom."""
        for col in DEFAULT_VOLUME_CANDIDATES:
            if col in df.columns:
                return col

        raise KeyError(
            "Geen volume-kolom gevonden. Verwacht een van: "
            f"{', '.join(DEFAULT_VOLUME_CANDIDATES)}"
        )

    def add_features(self, df: pd.DataFrame, timeframe: str = "micro") -> pd.DataFrame:
        """Hoofdfunctie voor feature extractie."""
        if df.empty:
            return df

        df = df.copy()
        self._prepare_index(df)

        # Pre-calculaties
        close_np: np.ndarray = df["close"].to_numpy(dtype=np.float64)
        returns = df["close"].pct_change().fillna(0.0)
        returns_np: np.ndarray = returns.to_numpy(dtype=np.float64)
        vol_col = self._get_volume_col(df)

        # 1. Alpha & momentum
        # WINDOW-FIX: autocorr window was 15 → SE ≈ 1/√15 ≈ 0.26 (vrijwel pure ruis).
        # Een autocorrelatie van 0.15 (typisch informatief) valt volledig binnen de
        # steekproeffout bij n=15. Window=60 geeft SE ≈ 0.13 → statistisch bruikbaar.
        #
        # SPEARMAN-FIX: Pearson is extreem gevoelig voor één uitschieter in kleine
        # windows — één 5σ return tilt ρ van 0.0 naar 0.4. Spearman rank autocorr
        # is bounded door de ranks [1..w] en verplaatst hooguit 1/w per nieuwe bar,
        # dus drastisch stabieler voor momentum-features in crypto (fat tails).
        # De originele Pearson-versie blijft als diagnostiek beschikbaar onder
        # dezelfde kolomnaam achter een config-toggle.
        use_spearman_ac: bool = bool(self.cfg.get("use_spearman_autocorr", True))
        if use_spearman_ac:
            df["feat_autocorr_60"] = roll_autocorr_spearman(returns_np, 60)
        else:
            df["feat_autocorr_60"] = roll_autocorr_lag1(returns_np, 60)
        # feat_cycle_20 (atan2 op z-scores) verwijderd: financiële data heeft
        # zelden stabiele vaste cycli. Spectrale fase-features falen structureel
        # out-of-sample door constante fase-verschuivingen in de markt.
        #
        # HILBERT-FASE (optioneel): als scipy.signal.hilbert beschikbaar is,
        # voegen we een causale Hilbert-fase toe op een low-pass gefilterde
        # prijs (EMA). Dit is marketen-aware i.p.v. puur spectraal: de
        # Hilbert-transform werkt op elke gladde reeks, niet op de aanname van
        # een vaste dominante cyclus. Zie :func:`compute_hilbert_phase`.
        if bool(self.cfg.get("use_hilbert_phase", True)):
            try:
                df["feat_hilbert_phase"] = compute_hilbert_phase(
                    close_np, smoothing_span=20
                )
            except Exception as exc:
                logger.debug("Hilbert-phase faalde (%s) — feature overgeslagen.", exc)
        df["feat_entropy_150"] = calculate_shannon_entropy(close_np, 150, 20)

        # 2. Trend & breakout
        df = self._add_trend_features(df)

        # 3. Fractional differencing (FFD) — prijs + volume
        df = self._add_ffd_features(df, timeframe=timeframe)
        df = self._add_ffd_volume_features(df, vol_col=vol_col, timeframe=timeframe)

        # 4. Microstructure & VWAP
        df = self._add_microstructure_features(df, vol_col)

        # 5. Volatility & jumps
        df = self._add_volatility_features(df, returns_np)

        # 6. Session features
        df = self._add_session_features(df)

        # 7. Degenerate-bar flag (issue #10 — Zero-Variance Bars)
        # Een bar waarbij open=high=low=close is een 1-tick-bar. Delingen door
        # σ (Sharpe, z-scores, cycle_phase) exploderen of worden artificieel 0.
        # De 1e-9 numba clamp voorkomt NaN, maar wiskundig is de uitkomst ruis.
        # We markeren deze bars expliciet zodat het model ze kan negeren.
        if {"high", "low", "open", "close"}.issubset(df.columns):
            high_np  = df["high"].to_numpy(dtype=np.float64)
            low_np   = df["low"].to_numpy(dtype=np.float64)
            open_np  = df["open"].to_numpy(dtype=np.float64)
            close_np2 = df["close"].to_numpy(dtype=np.float64)
            hlco_equal = ((high_np - low_np) < 1e-9) & (
                np.abs(close_np2 - open_np) < 1e-9
            )
            df["feat_bar_is_degenerate"] = hlco_equal.astype(np.int8)
        else:
            df["feat_bar_is_degenerate"] = np.zeros(len(df), dtype=np.int8)

        # Cleanup
        return df.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)

    def _prepare_index(self, df: pd.DataFrame) -> None:
        """Zorgt voor een correcte DatetimeIndex."""
        if not isinstance(df.index, pd.DatetimeIndex):
            if "timestamp" in df.columns:
                df.set_index("timestamp", inplace=True)

            df.index = pd.to_datetime(df.index, utc=True)

        elif df.index.tz is None:
            df.index = df.index.tz_localize("UTC")

    def _add_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Trendsterkte en breakout-posities."""
        high, low, close = df["high"], df["low"], df["close"]
        prev_close = close.shift(1)

        tr = pd.concat(
            [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
            axis=1,
        ).max(axis=1)

        df["feat_atr"] = tr.rolling(self.window).mean()
        df["feat_atr_rel"] = df["feat_atr"] / close

        r_low = low.rolling(self.window).min()
        r_high = high.rolling(self.window).max()
        df["feat_breakout_pos"] = (
            (close - r_low) / (r_high - r_low).replace(0, 1e-9)
        )

        path = close.diff().abs().rolling(self.window).sum()
        df["feat_trend_strength"] = (
            close.diff(self.window).abs() / path
        ).fillna(0.0)

        return df

    def _add_ffd_features(
        self, df: pd.DataFrame, timeframe: str = "micro"
    ) -> pd.DataFrame:
        """Stabiele fractional differencing met gecachte vaste d per timeframe.

        LEAKAGE-FIX (was kritiek):
          d wordt nu EENMALIG per timeframe gecached in self._fixed_d_cache.
          Bij de eerste aanroep per timeframe wordt d berekend op de eerste
          min(60%, 5000 bars) van de data — nooit meer dan 60% om altijd een
          OOS-buffer te bewaren, zelfs bij kleine datasets.
          Alle volgende aanroepen (productie, incremental updates) hergebruiken
          exact dezelfde d-waarde → deterministisch, geen sluipende leakage.

        CPCV-LEAKAGE-FIX (Item 1):
          De default kalibratie hieronder gebruikt de eerste 60% van de data.
          Bij CPCV waarbij test-folds vroeg in de tijdlijn vallen, lekt macro-
          structuur uit de toekomst terug naar die folds. Roep daarom in de
          training-loop **vóór** elke fold-aanroep van add_features() de
          methode `fit_d_on_train_indices(df, sub_tr_idx, timeframe)` aan
          (samen met `clear_d_cache(timeframe)`). De cache is dan al gevuld
          met de fold-specifieke d en de 60%-fallback wordt overgeslagen.
        """
        try:
            log_price: np.ndarray = np.log(
                df["close"].clip(lower=1e-9).to_numpy(dtype=np.float64)
            )
            n = len(log_price)

            # --- Stap 0: Bepaal dynamische truncation-threshold (per timeframe) ---
            # In hoge-volatility regimes blijven de FFD-gewichten langer
            # significant: een statische threshold 1e-4 kapt dan te snel af,
            # waardoor lange-range memory juist verloren gaat tijdens macro-
            # schokken. De dynamische threshold schaalt omgekeerd evenredig
            # met recente σ en wordt per timeframe **gecacht** zodat live-
            # inferentie exact dezelfde truncatie toepast als training.
            if timeframe not in self._ffd_thres_cache:
                if self._use_dynamic_ffd_thres:
                    # Gebruik ruwe returns (niet log) voor thresholds — beide
                    # geven dezelfde *relatieve* variatie, maar rauwe pct-change
                    # is numeriek stabieler voor σ-schattingen.
                    ret_np = np.diff(log_price, prepend=log_price[0])
                    tau_dyn = compute_dynamic_ffd_threshold(
                        ret_np,
                        base=self.ffd_thres,
                    )
                else:
                    tau_dyn = float(self.ffd_thres)
                self._ffd_thres_cache[timeframe] = tau_dyn
                logger.info(
                    "FFD [%s]: effectieve truncation-threshold = %.2e "
                    "(base=%.2e, dynamic=%s).",
                    timeframe,
                    tau_dyn,
                    self.ffd_thres,
                    self._use_dynamic_ffd_thres,
                )

            ffd_thres_eff: float = float(self._ffd_thres_cache[timeframe])

            # --- Stap 1: Bepaal fixed_d (gecacht per timeframe) ---
            if timeframe not in self._fixed_d_cache:
                # CPCV-LEAKAGE-FIX (simpele variant): kalibreer d op het
                # allereerste burn-in venster (≤ 2000 bars). Dit venster valt
                # gegarandeerd vóór alle CPCV test-folds ongeacht fold-grenzen.
                #
                # Oud lek: min(60%, 5000) bars. Bij walk-forward CPCV waarbij
                # fold 1 begint op bar 0, zat de helft van de kalibratieset al
                # in een test-fold → d absorbeerde het macro-regime van de
                # toekomst. Nu kalibreren we op max 2000 bars (gelijk aan
                # drop_bars in features.py — bars die sowieso worden weggegooid
                # als burn-in) zodat de d-waarde structureel leakage-vrij is.
                #
                # Voor exacte per-fold kalibratie: roep voor elke CPCV-fold
                # `clear_d_cache(timeframe)` + `fit_d_on_train_indices(df,
                # sub_tr_idx, timeframe)` aan in train_regime.py. Dan is de
                # cache al gevuld door de fold-lus en wordt dit blok overgeslagen.
                _BURN_IN_CALIB_BARS: int = 2000
                calib_end = min(
                    _BURN_IN_CALIB_BARS,
                    int(n * 0.60),
                    5000,
                )
                calib_end = max(calib_end, 30)   # absolute minimum voor ADF

                if calib_end < n:
                    candidate_d = get_optimal_d(
                        df["close"].iloc[:calib_end],
                        max_d=1.0,
                        step=0.05,
                        p_thres=0.05,
                        ffd_thres=ffd_thres_eff,
                    )
                else:
                    # Dataset te klein om 60/40 split te maken — gebruik fallback
                    candidate_d = float("nan")
                    logger.warning(
                        "FFD [%s]: Dataset te klein voor OOS-kalibratie "
                        "(n=%d). Fallback naar d=0.4", timeframe, n
                    )

                if np.isnan(candidate_d):
                    candidate_d = 0.4

                self._fixed_d_cache[timeframe] = candidate_d
                logger.info(
                    "FFD [%s]: d=%.2f gekalibreerd op %d bars en gecacht.",
                    timeframe, candidate_d, calib_end,
                )

            fixed_d: float = float(self._fixed_d_cache[timeframe])

            # Sla d op als constante diagnostiekkolom
            df["feat_frac_diff_d"] = fixed_d
            df["feat_frac_diff_tau"] = ffd_thres_eff  # diagnostiek

            # --- Stap 2: Vaste d over de volledige reeks ---
            full_mask = np.ones(n, dtype=bool)
            weights_fixed = get_weights_ffd(fixed_d, ffd_thres_eff, n)
            df["feat_frac_diff"] = apply_frac_diff_mask_fast(
                log_price, weights_fixed, full_mask
            )

            # --- Stap 3: Multi-d ensemble (model leert eigen weging) ---
            for d_candidate in (0.2, 0.4, 0.6, 0.8):
                w_c = get_weights_ffd(d_candidate, ffd_thres_eff, n)
                fd_c = apply_frac_diff_mask_fast(log_price, w_c, full_mask)
                col = f"feat_ffd_d{str(d_candidate).replace('.', '')}"
                df[col] = fd_c

        except Exception as exc:
            logger.error("FFD error [%s]: %s", timeframe, exc)

        return df

    def _add_ffd_volume_features(
        self, df: pd.DataFrame, vol_col: str, timeframe: str = "micro"
    ) -> pd.DataFrame:
        """Fractional differencing op log(volume).

        Volume heeft langetermijngeheugen (clustering: hoge volume-periodes
        volgen elkaar op). Standaard-returns (d=1) wissen dit geheugen volledig.
        FFD met de minimale stationaire d behoudt de autocorrelatie-structuur
        en geeft het model informatie over regimewijzigingen in liquiditeit.

        Leakage-preventie: dezelfde calib-strategie als _add_ffd_features —
        d wordt eenmalig gecacht op de eerste 60% van de data.
        Cache-sleutel: f"vol_{timeframe}" om niet te botsen met prijs-d.
        """
        try:
            vol_series = df[vol_col].clip(lower=1e-9)
            log_vol: np.ndarray = np.log(vol_series.to_numpy(dtype=np.float64))
            n = len(log_vol)

            # Dynamische truncation-threshold voor volume (zelfde principe als prijs).
            vol_thres_key = f"vol_thres_{timeframe}"
            if vol_thres_key not in self._ffd_thres_cache:
                if self._use_dynamic_ffd_thres:
                    dlog = np.diff(log_vol, prepend=log_vol[0])
                    tau_dyn_vol = compute_dynamic_ffd_threshold(
                        dlog, base=self.ffd_thres
                    )
                else:
                    tau_dyn_vol = float(self.ffd_thres)
                self._ffd_thres_cache[vol_thres_key] = tau_dyn_vol

            ffd_thres_vol: float = float(self._ffd_thres_cache[vol_thres_key])

            vol_key = f"vol_{timeframe}"
            if vol_key not in self._fixed_d_cache:
                # P1.2-FIX (CHIEF AUDIT 2026-05-23): Apply the same
                # _BURN_IN_CALIB_BARS = 2000 cap that the price-FFD block
                # uses (see comment above).  The old min(60%, 5000) formula
                # let the volume-d calibration reach into test folds for
                # early CPCV folds, leaking future regime information into
                # the d parameter.  min(2000, 60%, ...) is always ≤ 2000
                # for datasets large enough to run CPCV.
                _BURN_IN_CALIB_BARS_VOL: int = 2000
                calib_end = min(_BURN_IN_CALIB_BARS_VOL, int(n * 0.60), 5000)
                calib_end = max(calib_end, 30)
                if calib_end < n:
                    candidate_d = get_optimal_d(
                        vol_series.iloc[:calib_end],
                        max_d=1.0,
                        step=0.05,
                        p_thres=0.05,
                        ffd_thres=ffd_thres_vol,
                    )
                else:
                    candidate_d = float("nan")
                    logger.warning(
                        "FFD-volume [%s]: dataset te klein voor OOS-kalibratie "
                        "(n=%d). Fallback naar d=0.3.",
                        timeframe, n,
                    )

                if np.isnan(candidate_d):
                    # Volume is al deels stationair → lagere fallback-d dan prijs
                    candidate_d = 0.3

                self._fixed_d_cache[vol_key] = candidate_d
                logger.info(
                    "FFD-volume [%s]: d=%.2f gekalibreerd op %d bars "
                    "(tau=%.2e).",
                    timeframe, candidate_d, calib_end, ffd_thres_vol,
                )

            fixed_d_vol = float(self._fixed_d_cache[vol_key])
            df["feat_ffd_vol_d"] = fixed_d_vol  # diagnostiek

            full_mask = np.ones(n, dtype=bool)
            weights_vol = get_weights_ffd(fixed_d_vol, ffd_thres_vol, n)
            df["feat_ffd_vol"] = apply_frac_diff_mask_fast(log_vol, weights_vol, full_mask)

        except Exception as exc:
            logger.error("FFD-volume error [%s]: %s", timeframe, exc)

        return df

    # _get_rolling_d verwijderd — zie commit history. Rolling ADF veroorzaakte
    # kunstmatige structurele breuken door springende d-waarden.

    def _add_microstructure_features(
        self,
        df: pd.DataFrame,
        vol_col: str,
    ) -> pd.DataFrame:
        """Orderflow- en VWAP-dynamiek."""
        tp = (df["high"] + df["low"] + df["close"]) / 3
        vwap_num = (tp * df[vol_col]).rolling(50).sum()
        vwap_den = df[vol_col].rolling(50).sum().replace(0, 1)
        vwap = vwap_num / vwap_den

        std = df["close"].rolling(50).std().replace(0, 1e-9)
        df["feat_vwap_dist"] = (df["close"] - vwap) / std

        bar_dir = np.sign(df["close"] - df["open"])

        if (
            "taker_buy_volume" in df.columns
            and "taker_sell_volume" in df.columns
        ):
            raw_cvd_flow = df["taker_buy_volume"] - df["taker_sell_volume"]
        else:
            raw_cvd_flow = df[vol_col] * bar_dir

        # ── Feature 1: Rolling Volume Delta (RVD, 50 bars) ──────────────────
        # Naam-fix: de vorige "feat_cvd_z" was geen echte CVD meer — rolling(50).sum()
        # heeft geen geheugen over meerdere sessies. Hernoemd naar feat_rvd_z (Rolling
        # Volume Delta) zodat de naam de semantiek weerspiegelt.
        # Korte-termijn orderflow-druk (stationair, geen leakage). ✓
        rvd_rolling = raw_cvd_flow.rolling(50).sum()
        rvd_std = rvd_rolling.rolling(50).std().replace(0, 1e-9)
        df["feat_rvd_z"] = (rvd_rolling - rvd_rolling.rolling(50).mean()) / rvd_std

        # ── Feature 2: Exponentially Weighted CVD  (vervangt daily-reset CVD) ──
        # RESET-FIX:
        #   De vorige implementatie hanteerde een `groupby(date).cumsum()` reset
        #   op middernacht UTC. Elke nieuwe dag sprong de CVD-stand discontinu
        #   van bv. +15 000 naar 0 → kunstmatige structural breaks waar het
        #   model valse regime-shifts uit leerde. Ook loopt de midnight-reset
        #   niet synchroon met crypto-markten (24/7, geen "open/close").
        #
        #   Oplossing: **exponentieel wegglijdende CVD**. Elke bar draagt bij
        #   aan een cumulatieve som met een geometrisch afvallend gewicht:
        #
        #       EW-CVD_t  =  λ · EW-CVD_{t-1}  +  flow_t,     λ = e^{−1/τ}
        #
        #   met τ = halflife_bars ≈ ½-dag van bars. Er is GEEN reset, dus
        #   geen valse structural break. Oude flow vervaagt natuurlijk
        #   (langetermijn-druk wordt automatisch uitgedoofd).
        #
        #   Z-score voor stationariteit: we delen door de EWMA-σ van de raw
        #   flow (zelfde redenering als voorheen — rolling std over de flow
        #   zelf is stationair, over de cumulatieve stand explodeert die).
        raw_cvd_flow_np = np.asarray(raw_cvd_flow.fillna(0.0), dtype=np.float64)
        halflife_bars = float(self.cfg.get("cvd_halflife_bars", 500.0))
        lam = math.exp(-math.log(2.0) / halflife_bars) if halflife_bars > 0 else 0.0
        ew_cvd = np.empty_like(raw_cvd_flow_np)
        if ew_cvd.size > 0:
            ew_cvd[0] = raw_cvd_flow_np[0]
            for _i in range(1, ew_cvd.size):
                ew_cvd[_i] = lam * ew_cvd[_i - 1] + raw_cvd_flow_np[_i]

        df["_cvd_ewma_raw"] = ew_cvd
        cvd_s_std = raw_cvd_flow.rolling(250, min_periods=20).std().replace(0, 1e-9)
        df["feat_cvd_session_z"] = df["_cvd_ewma_raw"] / cvd_s_std
        df.drop(columns=["_cvd_ewma_raw"], inplace=True)

        # ── Feature 2b: FFD(log(|CVD|+1) * sign) — regime-shift detector ──────
        # De EW-CVD is stationair-achtig maar behoudt korte-range memory. Voor
        # lange-range autocorrelatie-retentie fractioneel differentiëren we de
        # log-getransformeerde absolute flow (tekenbehoudend). Dit levert een
        # complementaire lange-horizon order-flow feature die dezelfde
        # leakage-preventie (gecachte d/threshold per timeframe) volgt als
        # _add_ffd_volume_features.
        try:
            signed = raw_cvd_flow_np
            log_signed = np.sign(signed) * np.log1p(np.abs(signed))
            n_flow = log_signed.size
            if n_flow > 50:
                tau_key = "cvd_flow_thres"
                if tau_key not in self._ffd_thres_cache:
                    self._ffd_thres_cache[tau_key] = (
                        compute_dynamic_ffd_threshold(signed, base=self.ffd_thres)
                        if self._use_dynamic_ffd_thres
                        else float(self.ffd_thres)
                    )
                tau_flow: float = float(self._ffd_thres_cache[tau_key])
                w_flow = get_weights_ffd(0.4, tau_flow, n_flow)
                mask_flow = np.ones(n_flow, dtype=bool)
                df["feat_cvd_flow_ffd"] = apply_frac_diff_mask_fast(
                    log_signed, w_flow, mask_flow
                )
        except Exception as exc:  # pragma: no cover
            logger.debug("FFD-CVD feature faalde (%s) — overgeslagen.", exc)

        return df

    def _add_volatility_features(
        self,
        df: pd.DataFrame,
        returns: np.ndarray,
    ) -> pd.DataFrame:
        """Bipower variation en jump intensity."""
        n = len(returns)
        rv = np.full(n, np.nan)
        bv = np.full(n, np.nan)
        pi_2 = math.pi / 2.0
        window = self.window

        for i in range(window, n):
            win_ret = returns[i - window + 1 : i + 1]
            rv[i] = float(np.sum(win_ret ** 2))
            # BNS 2004: multiply by n/(n-1) finite-sample correction.
            # Without it, BV is systematically low by ~1/window → jump intensity
            # overestimated by ~1.3% at window=75.
            abs_r = np.abs(win_ret)
            left = abs_r[1:]
            right = abs_r[:-1]
            pairs = left * right
            # ZERO-RETURN-BIAS FIX: tel alleen paren met beide returns ≠ 0
            # en herschaal met 1/p² (p = fractie niet-nul paren). Anders
            # onderschat BV in low-volume regimes → jump_intensity vals hoog.
            nonzero_pair = (left != 0.0) & (right != 0.0)
            bv_sum = float(pairs[nonzero_pair].sum()) * pi_2
            total_pairs = window - 1
            if total_pairs > 0:
                bv_sum = bv_sum * (window / (window - 1))
                n_nz = int(nonzero_pair.sum())
                if 0 < n_nz < total_pairs:
                    p = float(n_nz) / float(total_pairs)
                    bv_sum = bv_sum / (p * p)
            bv[i] = bv_sum

        df["feat_jump_intensity"] = (rv - bv) / np.maximum(rv, 1e-9)
        df["feat_jump_intensity"] = df["feat_jump_intensity"].clip(lower=0.0)

        if get_garman_klass_volatility is not None:
            # GK: absolute=True → feat_vol_gk in prijseenheden (σ_GK × close).
            # σ_GK = sqrt(½(lnH/L)² − (2ln2−1)(lnC/O)²) over rolling venster.
            # absolute=True: dist_pt = feat_vol_gk × pt_width (in dollars) —
            # identieke schaal als de True-Range-ATR fallback.
            # Causal: sliding window, geen vorige-bar referentie, geen lookahead.
            df["feat_vol_gk"] = get_garman_klass_volatility(df, window=20, absolute=True)

        if get_jump_adjusted_volatility is not None:
            # FIX (issue #9 — Jump Diffusion component):
            # Pure GK veronderstelt continue GBM. Crypto heeft fat tails en
            # sprongen (liquidatie-cascades, news). We voegen een BNS-bipower
            # jump-variantie toe en exposen de som als feat_vol_gk_jump,
            # zonder de bestaande feat_vol_gk semantiek te breken.
            df["feat_vol_gk_jump"] = get_jump_adjusted_volatility(
                df, window=20, absolute=True
            )

            # feat_vol_jump_ratio: fractie van totale variantie verklaard door
            # discontinue sprongen (BNS jump-component / totale GK+jump variantie).
            #
            # Logica voor de Judge (Meta-Labeler):
            #   - Hoge ratio (~1.0): beweging gedreven door illiquiditeit /
            #     grote marktorders (gaps). Scout-signaal minder betrouwbaar.
            #   - Lage ratio (~0.0): continue GBM-trending. Scout-signaal
            #     informatief → hogere meta-label kans.
            #   Cruciaal voor SOL waar liquiditeit soms plotseling wegvalt.
            #
            # Berekening (variantie-ontbinding):
            #   var_total = feat_vol_gk_jump²   (σ_GK² + σ_jump²)
            #   var_gk    = feat_vol_gk²         (σ_GK², alleen continue component)
            #   var_jump  = var_total − var_gk    (zuivere sprong-variantie ≥ 0)
            #   ratio     = var_jump / var_total  ∈ [0, 1]
            #
            # absolute=True: beide series in prijseenheden (σ × close) →
            # kwadratering geeft consistente variantieschaal voor de deling.
            gk_sq   = df["feat_vol_gk"] ** 2
            tot_sq  = df["feat_vol_gk_jump"] ** 2
            jump_sq = (tot_sq - gk_sq).clip(lower=0.0)
            df["feat_vol_jump_ratio"] = (
                jump_sq / tot_sq.clip(lower=1e-20)
            ).clip(0.0, 1.0).fillna(0.0)

        return df

    def _add_session_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tijdgebaseerde features (NY open/close)."""
        try:
            dti = cast(pd.DatetimeIndex, df.index)
            if dti.tz is None:
                ny_idx = dti.tz_localize("UTC").tz_convert(
                    "America/New_York"
                )
            else:
                ny_idx = dti.tz_convert("America/New_York")

            df["feat_hr"] = ny_idx.hour

            is_open = (ny_idx.hour == 9) & (ny_idx.minute >= 30)
            df["feat_is_ny_open"] = is_open.astype(float)
        except Exception:
            pass

        return df


# =============================================================================
# FEATURE ORTHOGONALIZER (PCA per cluster)
# =============================================================================
