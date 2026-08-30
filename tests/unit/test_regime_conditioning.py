# tests/unit/test_regime_conditioning.py
"""De conditioneringslaag van H2: van een regimemodel naar een exposure-factor.

WAT HIER WORDT AFGEDWONGEN
===========================
De vergelijking M0 vs. M1 vs. M2 is alleen een vergelijking van MODELLEN als de
afbeelding van model naar positie voor alle drie IDENTIEK is. Zij is dat:

    a_geconditioneerd[t] = a_basis[t] * (1 - p_hoog[t])

M0 levert `p_hoog` als indicator, M1 als eenstapsvoorspelling, M2 als filtered
posterior. Er is geen vrije parameter in die afbeelding -- geen multiplier per
bucket, geen drempel op de kans -- want elke vrije parameter daarin zou een
extra trial zijn die niet is gepre-registreerd, en zou de winnaar bepalen zonder
dat het regimemodel iets deed.

Buiten het OOS-masker is de factor in ELKE arm exact 1. De twee armen zijn daar
dus bitidentiek, en elk verschil dat het rapport meldt komt van de bars waarop
het model out-of-sample wordt beoordeeld.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tradebot.cv.walk_forward import WalkForwardCV
from tradebot.regime.buckets import UNDEFINED_BUCKET, VolBucket
from tradebot.regime.conditioning import (
    CONDITIONER_SPECS,
    build_conditioner,
    m0_multiplier,
    oos_mask,
)
from tradebot.schemas.config import (
    AdequacyConfig,
    M0BucketConfig,
    load_config,
    regime_config,
)
from tradebot.utils.failfast import DataContractError

ADEQUACY = load_config(ROOT / "conf/model/adequacy.yaml", AdequacyConfig)
M2 = regime_config().m2
SYMBOLS = ("AAA", "BBB")
#: Trainvensters van 400 bars: de Data Adequacy Gate eist >= 100 observaties
#: per toestand per fold, en bij k = 3 is 395/3 = 131,7 het krappe geval.
CV = WalkForwardCV(train_size=400, test_size=100, step=100, mode="rolling",
                   min_train=400, embargo_bars=5)


def _panel(n: int = 900, seed: int = 5) -> tuple[pd.DataFrame, dict]:
    """Twee symbolen met een duidelijke rustige en een onrustige helft."""
    index = pd.date_range("2021-01-01", periods=n, freq="D", tz="UTC",
                          name="asof_ts")
    rng = np.random.default_rng(seed)
    ohlc: dict[str, pd.DataFrame] = {}
    returns: dict[str, pd.Series] = {}
    for offset, symbol in enumerate(SYMBOLS):
        scale = np.where(np.arange(n) % 120 < 60, 0.004, 0.025)
        step = rng.standard_normal(n) * scale
        close = 100.0 * np.exp(np.cumsum(step))
        span = np.abs(step) * close
        ohlc[symbol] = pd.DataFrame(
            {"open": close, "high": close + span, "low": close - span,
             "close": close, "volume": 1.0e6 + offset},
            index=index)
        returns[symbol] = pd.Series(
            np.log(close / np.roll(close, 1)), index=index).iloc[1:]
    return pd.DataFrame(returns), ohlc


def _buckets(n: int = 900) -> pd.DataFrame:
    """Een bucketpaneel met de eerste 30 bars ongedefinieerd."""
    index = pd.date_range("2021-01-01", periods=n, freq="D", tz="UTC",
                          name="asof_ts")
    pattern = np.tile(
        [float(VolBucket.LOW), float(VolBucket.NORMAL), float(VolBucket.HIGH)],
        n // 3 + 1)[:n]
    frame = pd.DataFrame({s: pattern for s in SYMBOLS}, index=index)
    frame.iloc[:30] = UNDEFINED_BUCKET
    return frame


class TestOosMask:
    def test_is_the_union_of_the_test_windows(self) -> None:
        mask, folds = oos_mask(900, CV)
        assert mask.sum() == sum(len(f.test_idx) for f in folds)
        assert mask.sum() == 500
        assert not mask[:400].any()

    def test_folds_never_train_on_their_own_test_window(self) -> None:
        _, folds = oos_mask(900, CV)
        for fold in folds:
            assert fold.train_end <= int(fold.test_idx[0])

    def test_the_embargo_is_actually_dropped(self) -> None:
        _, folds = oos_mask(900, CV)
        for fold in folds:
            assert int(fold.test_idx[0]) - fold.train_end == CV.embargo_bars


class TestM0:
    def test_is_one_minus_the_high_indicator(self) -> None:
        buckets = _buckets()
        mask, _ = oos_mask(len(buckets), CV)
        values = m0_multiplier(buckets, mask)
        is_high = buckets == float(VolBucket.HIGH)
        assert ((values == 0.0) == (is_high & mask[:, None])).all().all()

    def test_is_exactly_one_outside_the_oos_mask(self) -> None:
        buckets = _buckets()
        mask, _ = oos_mask(len(buckets), CV)
        values = m0_multiplier(buckets, mask)
        assert (values[~mask] == 1.0).all().all()

    def test_an_undefined_bucket_inside_the_mask_crashes(self) -> None:
        """'Geen oordeel' is een geldige uitkomst van M0, maar niet op een bar
        waarop hij wordt beoordeeld: daar zou hij stilzwijgend 'geen reductie'
        betekenen en dat is een BEWERING."""
        buckets = _buckets()
        mask, _ = oos_mask(len(buckets), CV)
        buckets.iloc[600, 0] = UNDEFINED_BUCKET
        with pytest.raises(DataContractError):
            m0_multiplier(buckets, mask)


class TestConditioners:
    def test_the_spec_list_is_the_preregistered_trial_space(self) -> None:
        """Zes trials: M1, M2-gaussian en M2-student_t maal k = 2 en k = 3."""
        assert len(CONDITIONER_SPECS) == 6
        assert len({s.label for s in CONDITIONER_SPECS}) == 6

    @pytest.mark.parametrize("spec", CONDITIONER_SPECS, ids=lambda s: s.label)
    def test_multiplier_is_a_probability_and_one_outside_the_mask(
        self, spec
    ) -> None:
        returns, ohlc = _panel()
        result = build_conditioner(
            spec, returns=returns, buckets=_buckets().iloc[1:], cv=CV,
            adequacy=ADEQUACY, m2_cfg=M2)
        values = result.values
        assert float(values.min().min()) >= 0.0
        assert float(values.max().max()) <= 1.0
        assert (values[~result.mask] == 1.0).all().all()

    @pytest.mark.parametrize("spec", CONDITIONER_SPECS, ids=lambda s: s.label)
    def test_the_multiplier_is_causal(self, spec) -> None:
        """De sterkste vorm van dit bewijs: kap de reeks af en herbereken. Wat
        op bar t staat, mag niet veranderen doordat er later bars bijkomen."""
        returns, _ = _panel()
        buckets = _buckets().iloc[1:]
        cut = 750
        full = build_conditioner(
            spec, returns=returns, buckets=buckets, cv=CV, adequacy=ADEQUACY,
            m2_cfg=M2)
        short = build_conditioner(
            spec, returns=returns.iloc[:cut], buckets=buckets.iloc[:cut],
            cv=CV, adequacy=ADEQUACY, m2_cfg=M2)
        shared = short.values.index[short.mask]
        assert len(shared) > 0
        assert np.allclose(
            full.values.loc[shared].to_numpy(),
            short.values.loc[shared].to_numpy(), atol=0.0, rtol=0.0)

    @pytest.mark.parametrize("spec", CONDITIONER_SPECS, ids=lambda s: s.label)
    def test_reports_the_rarest_filtered_occupancy(self, spec) -> None:
        """Dit getal voedt stop-criterium 1 van de pre-registratie. Het komt
        uit de FILTERED posterior op het trainvenster -- de bars waarop de
        parameters daadwerkelijk zijn geschat."""
        returns, _ = _panel()
        result = build_conditioner(
            spec, returns=returns, buckets=_buckets().iloc[1:], cv=CV,
            adequacy=ADEQUACY, m2_cfg=M2)
        assert result.rarest_state_obs_per_fold > 0.0
        assert result.rarest_state_obs_per_fold <= result.n_train_obs_per_fold

    def test_the_high_state_is_the_most_volatile_one(self) -> None:
        """Niet toestand k-1 en niet de eerste: de toestandsvolgorde van een EM
        is willekeurig. Op data waarvan de tweede helft van elke cyclus
        onrustig is, moet de factor daar systematisch lager liggen."""
        returns, _ = _panel()
        spec = next(s for s in CONDITIONER_SPECS
                    if s.label == "hmm2-diag-gaussian")
        result = build_conditioner(
            spec, returns=returns, buckets=_buckets().iloc[1:], cv=CV,
            adequacy=ADEQUACY, m2_cfg=M2)
        scored = result.values[result.mask]
        position = np.arange(len(returns))[result.mask] % 120
        calm = scored.to_numpy()[position < 60].mean()
        wild = scored.to_numpy()[position >= 60].mean()
        assert wild < calm

    def test_diagnostics_count_switches_on_the_scored_bars(self) -> None:
        returns, _ = _panel()
        spec = next(s for s in CONDITIONER_SPECS
                    if s.label == "hmm2-diag-gaussian")
        result = build_conditioner(
            spec, returns=returns, buckets=_buckets().iloc[1:], cv=CV,
            adequacy=ADEQUACY, m2_cfg=M2)
        for symbol in SYMBOLS:
            assert 0 <= result.n_transitions[symbol] < int(result.mask.sum())
            assert result.mean_duration_bars[symbol] > 0.0

    def test_a_bucket_panel_off_the_return_axis_crashes(self) -> None:
        returns, _ = _panel()
        spec = CONDITIONER_SPECS[0]
        with pytest.raises(DataContractError):
            build_conditioner(
                spec, returns=returns, buckets=_buckets().iloc[1:-3], cv=CV,
                adequacy=ADEQUACY, m2_cfg=M2)


class TestM0Config:
    def test_thresholds_still_come_from_conf(self) -> None:
        assert regime_config().m0 == M0BucketConfig(
            zscore_low=-0.5, zscore_high=0.5, zscore_min_periods=250,
            atr_fast_window=5, atr_slow_window=20,
            atr_ratio_low=0.85, atr_ratio_high=1.15)
