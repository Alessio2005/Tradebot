"""M0 Causal Vol-Buckets doet wat de baseline moet doen — deliverable 14.

De causaliteit staat in `tests/lookahead/test_m0_causality.py`. Hier gaat het om
het GEDRAG: binden de drempels, weigert de config wat zij hoort te weigeren, en
klopt de diagnostiek waarmee het M0-vs-HMM-rapport straks de turnover verklaart.

Elke drempel wordt van BEIDE kanten benaderd. Phase 5 §12 vond een wiring-test
met niet-bindende limietwaarden die daardoor niets bewees; een drempeltest die
alleen de doorlaatkant meet, heeft hetzelfde probleem.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from tradebot.regime import (
    BucketAssignment,
    VolBucket,
    causal_atr,
    causal_vol_zscore,
    classify_vol_buckets,
)
from tradebot.schemas.config import M0BucketConfig, RegimeConfig, load_config
from tradebot.utils.failfast import DataContractError

SEED = 20260826
LAM = 0.94
BURN_IN = 60


def _frame(close: np.ndarray, *, span_frac: float = 0.01) -> pd.DataFrame:
    open_ = np.concatenate([[close[0]], close[:-1]])
    span = span_frac * close
    return pd.DataFrame({
        "open": open_,
        "high": np.maximum(open_, close) + span,
        "low": np.minimum(open_, close) - span,
        "close": close,
    })


class TestShippedConfigIsTheOneUnderTest:
    def test_conf_model_regime_yaml_validates(self) -> None:
        cfg = load_config("conf/model/regime.yaml", RegimeConfig)
        assert cfg.m0.zscore_low < cfg.m0.zscore_high
        assert cfg.m0.atr_fast_window < cfg.m0.atr_slow_window


class TestConfigRefusesIncoherentThresholds:
    """Zonder deze validatie zou een bar tegelijk HOOG en LAAG kunnen zijn."""

    def test_inverted_zscore_band_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="NORMAAL-band"):
            M0BucketConfig(zscore_low=0.5, zscore_high=-0.5)

    def test_inverted_atr_band_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="atr_ratio_low"):
            M0BucketConfig(atr_ratio_low=1.5, atr_ratio_high=0.8)

    def test_fast_window_not_shorter_than_slow_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="korter"):
            M0BucketConfig(atr_fast_window=20, atr_slow_window=5)

    def test_equal_thresholds_are_refused_too(self) -> None:
        """De rand van de validatie, niet alleen de duidelijke omkering."""
        with pytest.raises(ValidationError):
            M0BucketConfig(zscore_low=0.0, zscore_high=0.0)


class TestCausalAtr:
    def test_bar_zero_carries_no_true_range(self) -> None:
        """Bar 0 heeft geen vorige close.

        Zou hij op `high - low` terugvallen, dan rust zijn True Range op ANDERE
        informatie dan die van elke volgende bar, en het venster dat hem
        meeneemt is stilzwijgend inhomogeen.
        """
        frame = _frame(np.array([100.0, 101.0, 102.0, 103.0]))
        atr = causal_atr(frame["high"], frame["low"], frame["close"], window=2)
        assert np.isnan(atr.iloc[0])
        assert np.isnan(atr.iloc[1])       # venster nog niet vol (bar 0 telt niet)
        assert np.isfinite(atr.iloc[2])

    def test_burn_in_is_not_forward_filled(self) -> None:
        frame = _frame(100.0 + np.arange(50, dtype=float))
        atr = causal_atr(frame["high"], frame["low"], frame["close"], window=20)
        assert atr.iloc[:20].isna().all()
        assert atr.iloc[20:].notna().all()

    def test_true_range_takes_the_gap_into_account(self) -> None:
        """Een gap-open hoort de True Range te vergroten, niet de high-low."""
        gapped = pd.DataFrame({
            "open": [100.0, 130.0], "high": [101.0, 131.0],
            "low": [99.0, 129.0], "close": [100.0, 130.0]})
        atr = causal_atr(
            gapped["high"], gapped["low"], gapped["close"], window=1)
        # |high_1 - close_0| = |131 - 100| = 31 domineert high-low = 2.
        assert atr.iloc[1] == pytest.approx(31.0)

    def test_a_zero_window_is_refused(self) -> None:
        frame = _frame(np.array([100.0, 101.0]))
        with pytest.raises(DataContractError, match="minder dan één bar"):
            causal_atr(frame["high"], frame["low"], frame["close"], window=0)


class TestCausalVolZscore:
    def test_undefined_before_min_periods(self) -> None:
        rng = np.random.default_rng(SEED)
        returns = pd.Series(rng.normal(0.0, 0.02, 400))
        z = causal_vol_zscore(
            returns, lam=LAM, burn_in_bars=BURN_IN, min_periods=250)
        assert z.iloc[:250].isna().all()
        assert z.iloc[-1:].notna().all()

    def test_a_constant_volatility_gives_no_zscore_rather_than_a_division(
        self,
    ) -> None:
        """Nul spreiding betekent geen z-score, geen deling door nul."""
        returns = pd.Series(np.full(400, 0.01))
        z = causal_vol_zscore(
            returns, lam=LAM, burn_in_bars=BURN_IN, min_periods=100)
        assert not np.isinf(z.to_numpy(dtype=np.float64)).any()

    def test_min_periods_below_two_is_refused(self) -> None:
        with pytest.raises(DataContractError, match="minder dan twee"):
            causal_vol_zscore(
                pd.Series([0.01] * 10), lam=LAM, burn_in_bars=5, min_periods=1)


def _regime_series(n: int = 900) -> pd.DataFrame:
    """Rustig, dan explosief, dan rustig — alle drie de buckets moeten voorkomen."""
    rng = np.random.default_rng(SEED + 3)
    sigma = np.concatenate([
        np.full(n // 3, 0.010),
        np.full(n // 3, 0.070),
        np.full(n - 2 * (n // 3), 0.008),
    ])
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, sigma)))
    open_ = np.concatenate([[100.0], close[:-1]])
    span = np.abs(rng.normal(0.0, sigma)) * close
    return pd.DataFrame({
        "open": open_,
        "high": np.maximum(open_, close) + span,
        "low": np.minimum(open_, close) - span,
        "close": close,
    })


def _classify(frame: pd.DataFrame, cfg: M0BucketConfig) -> BucketAssignment:
    return classify_vol_buckets(
        frame, cfg, ewma_lambda=LAM, ewma_burn_in_bars=BURN_IN, symbol="TEST")


class TestBothAxesMustAgree:
    """De conjunctie is een architectuurbesluit en wordt hier gemeten."""

    CFG = M0BucketConfig(zscore_min_periods=120)

    def test_all_three_buckets_occur_on_a_regime_switching_series(self) -> None:
        result = _classify(_regime_series(), self.CFG)
        assert all(result.occupancy[b.name] > 0 for b in VolBucket), (
            f"niet elk regime komt voor: {result.occupancy}")

    def test_an_unreachable_high_threshold_leaves_the_bucket_empty(self) -> None:
        """De andere kant van de drempel: hij BINDT.

        Zou HOOG ook bij een onbereikbare drempel voorkomen, dan classificeerde
        M0 op iets anders dan de drempel en zou de configwaarde niets doen.
        """
        cfg = M0BucketConfig(zscore_min_periods=120, zscore_high=99.0)
        assert _classify(_regime_series(), cfg).occupancy["HIGH"] == 0

    def test_an_unreachable_atr_threshold_alone_also_empties_high(self) -> None:
        """Bewijst dat de tweede as ECHT meedoet in de conjunctie.

        De z-score-drempel blijft hier normaal; alleen de ATR-eis wordt
        onbereikbaar. Zou HOOG toch voorkomen, dan wordt de ATR-as genegeerd en
        is M0 in werkelijkheid een een-assig model.
        """
        cfg = M0BucketConfig(zscore_min_periods=120, atr_ratio_high=99.0)
        assert _classify(_regime_series(), cfg).occupancy["HIGH"] == 0

    def test_the_same_holds_for_the_low_bucket(self) -> None:
        cfg = M0BucketConfig(zscore_min_periods=120, atr_ratio_low=1e-6)
        assert _classify(_regime_series(), cfg).occupancy["LOW"] == 0

    def test_a_wider_normal_band_switches_less(self) -> None:
        """Bredere band = minder wisselingen = minder turnover (§0.4)."""
        narrow = _classify(
            _regime_series(),
            M0BucketConfig(zscore_min_periods=120, zscore_low=-0.2,
                           zscore_high=0.2))
        wide = _classify(
            _regime_series(),
            M0BucketConfig(zscore_min_periods=120, zscore_low=-1.5,
                           zscore_high=1.5))
        assert wide.n_transitions < narrow.n_transitions


class TestDiagnosticsForTheReport:
    CFG = M0BucketConfig(zscore_min_periods=120)

    def test_occupancy_sums_to_the_defined_bars(self) -> None:
        result = _classify(_regime_series(), self.CFG)
        assert sum(result.occupancy.values()) == result.n_defined

    def test_mean_duration_is_consistent_with_the_transition_count(self) -> None:
        result = _classify(_regime_series(), self.CFG)
        assert result.mean_duration == pytest.approx(
            result.n_defined / (result.n_transitions + 1))

    def test_a_series_that_never_switches_reports_its_full_length(self) -> None:
        rng = np.random.default_rng(SEED + 5)
        calm = _frame(100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.012, 600))))
        result = _classify(calm, self.CFG)
        if result.n_transitions == 0:
            assert result.mean_duration == float(result.n_defined)

    def test_the_record_carries_what_the_report_needs(self) -> None:
        record = _classify(_regime_series(), self.CFG).as_record()
        assert set(record) >= {
            "symbol", "n_total", "n_defined", "coverage", "occupancy",
            "n_transitions", "mean_duration_bars"}


class TestInputContract:
    CFG = M0BucketConfig(zscore_min_periods=120)

    @pytest.mark.parametrize("missing", ["open", "high", "low", "close"])
    def test_a_missing_ohlc_column_is_refused(self, missing: str) -> None:
        frame = _regime_series().drop(columns=[missing])
        with pytest.raises(DataContractError, match=f"'{missing}' ontbreekt"):
            _classify(frame, self.CFG)

    def test_undefined_bars_are_nan_and_never_normal(self) -> None:
        result = _classify(_regime_series(), self.CFG)
        assert np.isnan(result.buckets.iloc[0])
        assert result.n_defined < result.n_total
