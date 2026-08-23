# src/tradebot/alpha/momentum.py
"""Time-series and cross-sectional momentum signals (Moskowitz et al. 2012).

TSMomentum  : 12-1 log-return momentum with Jegadeesh-Titman skip-month.
CSMomentum  : Rank-normalised cross-sectional momentum within a universe.
"""
from __future__ import annotations

import hashlib
import logging

import numpy as np
import pandas as pd

from .base import SignalResult

logger = logging.getLogger(__name__)

__all__ = ["TSMomentum", "CSMomentum"]


def _params_hash(**kwargs: object) -> str:
    s = "_".join(f"{k}={v}" for k, v in sorted(kwargs.items()))
    return hashlib.md5(s.encode()).hexdigest()[:8]


class TSMomentum:
    """Time-series momentum: 12-month minus 1-month log return.

    Implements the Jegadeesh-Titman (1993) skip-month correction:
    signal = log(P[t-1] / P[t-lookback]) — excludes the most-recent month
    to avoid microstructure-driven mean reversion.

    Parameters
    ----------
    symbol :
        Ticker name (embedded in signal_id).
    lookback_bars :
        Total lookback window in bars (default 252 ≈ 12 months daily).
    skip_bars :
        Number of bars to skip at the end (default 21 ≈ 1 month daily).
    vol_scale :
        If True, scale the signal by annualised volatility (TSMOM variant
        from Moskowitz et al. 2012 — uses ex-ante vol to normalise).
    """

    def __init__(
        self,
        symbol: str,
        lookback_bars: int = 252,
        skip_bars: int = 21,
        vol_scale: bool = True,
    ) -> None:
        self.symbol = symbol
        self.lookback_bars = lookback_bars
        self.skip_bars = skip_bars
        self.vol_scale = vol_scale
        self.signal_id = (
            f"{symbol}_tsmom_"
            + _params_hash(lb=lookback_bars, sk=skip_bars, vs=vol_scale)
        )
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> None:
        self._df = df[["close"]].copy()
        self._fitted = True

    def predict(self, df: pd.DataFrame, lag_bars: int = 1) -> SignalResult:
        """Compute time-series momentum signal.

        Parameters
        ----------
        df : pd.DataFrame
            Bar DataFrame with ``close`` column.
        lag_bars : int, default 1
            CHIEF AUDIT 2026-05-23 (P-3): Number of trailing bars to exclude
            from the realised-vol computation.  Without this exclusion the
            most-recent return ends on bar T while the signal is timestamped
            at T → trade on bar T uses information from bar T (same-bar
            leakage).  Default 1 = causal-safe (signal at T uses returns
            ending at T-1, so trade can enter at T).
        """
        prices = df["close"]
        n = len(prices)
        needed = self.lookback_bars + 1

        if n < needed:
            return SignalResult(
                symbol=self.symbol,
                timestamp=prices.index[-1],
                signal=0.0,
                confidence=0.5,
                horizon_bars=21,
                signal_id=self.signal_id,
            )

        # 12-1 log return: price at (t - skip_bars) / price at (t - lookback_bars)
        p_now  = float(prices.iloc[-(self.skip_bars + 1)])
        p_past = float(prices.iloc[-(self.lookback_bars + 1)])
        if p_past <= 0:
            return SignalResult(self.symbol, prices.index[-1], 0.0, 0.5, 21, self.signal_id)

        raw_signal = float(np.log(p_now / p_past))

        if self.vol_scale:
            # CHIEF AUDIT 2026-05-23 (P-3): drop the last ``lag_bars`` returns
            # so realised-vol is computed strictly on bars ≤ T - lag_bars.
            lag = max(0, int(lag_bars))
            if lag > 0 and len(prices.values) > lag + 1:
                log_rets = np.log(prices.values[1:-lag] / prices.values[:-1-lag])
            else:
                log_rets = np.log(prices.values[1:] / prices.values[:-1])
            if len(log_rets) == 0:
                ann_vol = 1e-9
            else:
                ann_vol = float(np.std(log_rets[-63:]) * np.sqrt(252)) + 1e-9
            raw_signal /= ann_vol

        # Map to [-1, +1] via tanh with scale 1.0
        signal = float(np.tanh(raw_signal))
        confidence = float(0.5 + 0.5 * signal)

        return SignalResult(
            symbol=self.symbol,
            timestamp=prices.index[-1],
            signal=signal,
            confidence=confidence,
            horizon_bars=21,
            signal_id=self.signal_id,
        )

    def feature_names(self) -> list[str]:
        return ["close"]


class CSMomentum:
    """Cross-sectional rank momentum within a multi-asset universe.

    For use when running across multiple symbols simultaneously.
    Each instance holds one symbol; an external combiner must rank
    signals across the universe.

    Implements the Asness et al. (1997) cross-sectional standardisation:
    z-score of rolling n-bar return within the universe.
    """

    def __init__(
        self,
        symbol: str,
        lookback_bars: int = 126,
    ) -> None:
        self.symbol = symbol
        self.lookback_bars = lookback_bars
        self.signal_id = (
            f"{symbol}_csmom_" + _params_hash(lb=lookback_bars)
        )

    def fit(self, df: pd.DataFrame) -> None:
        self._df = df[["close"]].copy()

    def predict(self, df: pd.DataFrame, lag_bars: int = 1) -> SignalResult:
        """Compute cross-sectional momentum signal.

        Parameters
        ----------
        df : pd.DataFrame
            Bar DataFrame with ``close`` column.
        lag_bars : int, default 1
            CHIEF AUDIT 2026-05-23 (P-3): Number of trailing bars to exclude
            from the return computation so signal_t uses returns ending at
            T - lag_bars (causal-safe; signal at T can be traded at T).
        """
        prices = df["close"]
        lag = max(0, int(lag_bars))
        if len(prices) < self.lookback_bars + 1 + lag:
            return SignalResult(self.symbol, prices.index[-1], 0.0, 0.5, 21, self.signal_id)

        # CHIEF AUDIT 2026-05-23 (P-3): anchor the latest "now" price at
        # T - lag_bars rather than T to prevent same-bar leakage.
        if lag > 0:
            p_now = float(prices.iloc[-(1 + lag)])
            p_past = float(prices.iloc[-(self.lookback_bars + 1 + lag)])
        else:
            p_now = float(prices.iloc[-1])
            p_past = float(prices.iloc[-(self.lookback_bars + 1)])
        if p_past <= 0:
            return SignalResult(self.symbol, prices.index[-1], 0.0, 0.5, 21, self.signal_id)
        ret = float(np.log(p_now / p_past))

        # Without a full universe, normalise against own rolling distribution.
        # CHIEF AUDIT 2026-05-23 (P-3): also drop the last ``lag`` bars when
        # building the calibration window so it stays strictly historical.
        if lag > 0 and len(prices.values) > lag:
            prices_for_window = prices.values[:-lag]
        else:
            prices_for_window = prices.values
        if len(prices_for_window) <= self.lookback_bars:
            return SignalResult(self.symbol, prices.index[-1], 0.0, 0.5, 21, self.signal_id)
        window_rets = np.array([
            np.log(prices_for_window[i] / prices_for_window[i - self.lookback_bars])
            for i in range(self.lookback_bars, len(prices_for_window))
        ])
        if len(window_rets) < 5:
            return SignalResult(self.symbol, prices.index[-1], 0.0, 0.5, 21, self.signal_id)

        mu, sigma = float(np.mean(window_rets)), float(np.std(window_rets)) + 1e-9
        z = (ret - mu) / sigma
        signal = float(np.tanh(z / 2.0))
        confidence = float(0.5 + 0.5 * signal)

        return SignalResult(
            symbol=self.symbol,
            timestamp=prices.index[-1],
            signal=signal,
            confidence=confidence,
            horizon_bars=21,
            signal_id=self.signal_id,
        )

    def feature_names(self) -> list[str]:
        return ["close"]


# =========================================================================== #
# PHASE 3 - L4 CROSS-SECTIONAL MOMENTUM (Level 1 baseline)
# =========================================================================== #
# Alles BOVEN deze regel is legacy: `TSMomentum` en `CSMomentum` op het oude
# `AlphaSignal`-protocol, met een per-bar `predict()` die ook `confidence`,
# `horizon_bars` en - in het geval van TSMomentum - een VOL-SCHALING teruggeeft.
# Die vol-schaling is precies wat sectie 11.1 uit L4 verbant: het is
# positiegrootte, geen richting. De code is hier bewust ONGEWIJZIGD gelaten
# (26 modules en `alpha/__init__.py` hangen eraan) en valt onder DI-12.
#
# Alles HIERONDER is de Level-1 baseline op het nieuwe L4-contract:
#
#     a_t = rank-gedemeaned 60-bars log-rendement, geschaald naar [-1, +1]
#
# Nul risicologica, nul leverage, nul stop-losses, nul executiekennis. Wat er
# met `a_t` gebeurt, beslissen L7 en L8.
# --------------------------------------------------------------------------- #
from typing import Any as _Any
from typing import ClassVar as _ClassVar

from ..features.pipeline import PanelPipeline as _PanelPipeline
from ..features.pipeline import TransformStep as _TransformStep
from ..utils.failfast import DataContractError as _DataContractError
from ..utils.failfast import require as _require
from .base import AlphaUnit as _AlphaUnit


class CrossSectionalMomentum(_AlphaUnit):
    """Level-1 Cross-Sectional Momentum. De meetlat, niet het slimme model.

    De constructie is met opzet zo simpel als het onderwerp toelaat:

      1. het cumulatieve log-rendement over `lookback_bars`, eindigend
         `skip_bars` voor `t` (Jegadeesh-Titman skip tegen de korte-termijn
         reversal);
      2. per tijdstip een rangschikking over de assets, lineair geschaald naar
         `[-1, +1]`.

    Beide stappen zijn L1-transforms; deze unit voegt er geen wiskunde aan toe.
    Wat hij WEL bezit, is de **rebalance-kalender**: hoe vaak de view wordt
    ververst is een alpha-beslissing en staat in `conf/model/alpha.yaml`.

    WAAROM DE RANG EN NIET DE Z-SCORE
    ---------------------------------
    Een rang is per constructie begrensd op `[-1, +1]` en per rij som-nul bij
    symmetrische bezetting: hij is dus schaalvrij en dollar-neutraal zonder dat
    er ergens een normalisatie hoeft te worden bijgehouden. Een z-score zou dat
    bereik moeten afdwingen met een clip, en een clip die vaak bindt is een
    verborgen positielimiet - oftewel risicologica in L4.

    WAT HIER NIET IN ZIT, EN WAAROM
    -------------------------------
    Geen vol-schaling (dat is L7 vol-targeting), geen gewichten (L8), geen
    kosten (L9/L10), geen stop-loss (L7). De legacy `TSMomentum` hierboven doet
    het eerste wel; dat is exact de verstrengeling die Phase 4 moet ontwarren en
    die hier niet opnieuw wordt geintroduceerd.

    DE FANTOOM-HERBALANCERINGSFIX (RETAIN, audit sectie 24)
    -------------------------------------------------------
    `alpha/xs_unit.py` bevat de fix waarbij de rebalance-kalender wordt bepaald
    door met de VOLGENDE bar te vergelijken, zodat de laatste bar van een
    afgekapt panel nooit een rebalance wordt. Diezelfde eigenschap wordt hier
    op een andere manier bereikt en NIET verwaterd: de kalender telt posities
    vanaf het BEGIN van het panel (`i % rebalance_every_bars == 0`) en kijkt
    dus nergens naar het einde. Truncatie kan de kalender daarmee per
    constructie niet verschuiven - bewezen in
    `tests/lookahead/test_baseline_causality.py`.
    """

    name: _ClassVar[str] = "cross_sectional_momentum"

    def __init__(
        self,
        *,
        lookback_bars: int,
        skip_bars: int,
        min_assets: int,
        rebalance_every_bars: int,
        signal_floor: float,
        signal_cap: float,
    ) -> None:
        _require(
            lookback_bars > 0,
            "Een momentum-lookback moet positief zijn.",
            _DataContractError,
            lookback_bars=lookback_bars,
        )
        _require(
            skip_bars >= 0,
            "Een NEGATIEVE skip verschuift het formatievenster naar de TOEKOMST.",
            _DataContractError,
            skip_bars=skip_bars,
        )
        _require(
            min_assets > 1,
            "Een cross-sectie over minder dan twee assets bestaat niet.",
            _DataContractError,
            min_assets=min_assets,
        )
        _require(
            rebalance_every_bars > 0,
            "De rebalance-frequentie moet positief zijn.",
            _DataContractError,
            rebalance_every_bars=rebalance_every_bars,
        )
        super().__init__(
            signal_floor=signal_floor,
            signal_cap=signal_cap,
            params={
                "lookback_bars": int(lookback_bars),
                "skip_bars": int(skip_bars),
                "min_assets": int(min_assets),
                "rebalance_every_bars": int(rebalance_every_bars),
            },
        )

    # ------------------------------------------------------------- features
    def feature_steps(self) -> tuple[_TransformStep, ...]:
        """De L1-transforms die deze unit nodig heeft, in volgorde.

        De unit DECLAREERT zijn features maar berekent ze niet zelf: de
        berekening hoort in L1, waar hij getoetst en gehasht wordt.
        """
        return (
            _TransformStep(
                "rolling_log_return",
                {"window": int(self.params["lookback_bars"]),
                 "skip_bars": int(self.params["skip_bars"])},
            ),
            _TransformStep(
                "cross_sectional_rank",
                {"min_assets": int(self.params["min_assets"])},
            ),
        )

    def feature_pipeline(self) -> _PanelPipeline:
        return _PanelPipeline(self.feature_steps())

    # ------------------------------------------------------------- contract
    @property
    def burn_in_period(self) -> int:
        return int(self.params["lookback_bars"]) + int(self.params["skip_bars"])

    def _generate(self, features: pd.DataFrame) -> pd.DataFrame:
        every = int(self.params["rebalance_every_bars"])
        if every == 1:
            exposures = features.astype("float64")
        else:
            # Kalender geteld vanaf het BEGIN van het panel, nooit vanaf het
            # einde: truncatie kan hem daarmee niet verschuiven. `ffill` draagt
            # hier een GENOMEN beslissing vooruit (de positie wordt gehouden) en
            # vult geen ontbrekende schatting in - dat onderscheid is de reden
            # dat het geen contractbreuk is.
            is_rebalance = pd.Series(
                (np.arange(len(features)) % every) == 0, index=features.index
            )
            exposures = (
                features.astype("float64")
                .where(is_rebalance, other=np.nan)
                .ffill()
            )
        return exposures.clip(
            lower=float(self.params["signal_floor"]),
            upper=float(self.params["signal_cap"]),
        ).astype("float64")


def build_cross_sectional_momentum(cfg: _Any) -> CrossSectionalMomentum:
    """Bouw de baseline-unit uit `conf/model/alpha.yaml`.

    De enige plek waar deze unit wordt geinstantieerd voor productie- en
    testgebruik; er staat geen enkele parameter als literal in deze functie.
    """
    return CrossSectionalMomentum(
        lookback_bars=cfg.lookback_bars,
        skip_bars=cfg.skip_bars,
        min_assets=cfg.min_assets,
        rebalance_every_bars=cfg.rebalance_every_bars,
        signal_floor=cfg.signal_floor,
        signal_cap=cfg.signal_cap,
    )
