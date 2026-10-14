# tests/unit/test_schema_contracts.py
"""De dataframe-checks van de Pandera-schema's en de validators van `conf/`.

Elke check hieronder wordt RECHTSTREEKS aangeroepen, met een frame dat hem
laat slagen en een frame dat hem laat falen. Een check die alleen ooit `True`
heeft teruggegeven, is niet getoetst: hij kan net zo goed `return True` zijn.

Voor de configvalidators geldt hetzelfde: een geldige configuratie uit
`conf/` wordt op precies één veld kapotgemaakt, zodat de foutmelding van
DIE validator komt en niet van een andere.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from tradebot.schemas._common import (
    is_tz_aware_utc,
    monotonic_increasing_index,
    no_all_nan,
)
from tradebot.schemas.bars import BarSchema
from tradebot.schemas.config import (
    AlphaConfig,
    ConfigContractError,
    DataConfig,
    FeatureConfig,
    FracDiffConfig,
    ImpactConfig,
    M2HmmConfig,
    MetaLabelConfig,
    MetaLabelingAdequacyConfig,
    MomentumFeatureConfig,
    VolStateConfig,
    adequacy_config,
    load_config,
)
from tradebot.schemas.events import EventSchema
from tradebot.schemas.features import FeatureBlockSchema
from tradebot.schemas.folds import CPCVFoldSchema
from tradebot.schemas.labels import LabelSchema
from tradebot.schemas.oos_predictions import OOSPredictionSchema

ROOT = Path(__file__).resolve().parents[2]
UTC_INDEX = pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")
PLAIN = pd.DataFrame({"a": [1.0, 2.0, 3.0]})


class TestCommonHelpers:
    def test_monotonic_index(self) -> None:
        assert monotonic_increasing_index(pd.Index([1, 2, 3])) is True
        assert monotonic_increasing_index(pd.Index([2, 1, 3])) is False

    def test_an_all_nan_column_is_refused(self) -> None:
        series = pd.Series([1.0, np.nan], name="x")
        assert no_all_nan(series) is series
        with pytest.raises(AssertionError, match="entirely NaN"):
            no_all_nan(pd.Series([np.nan, np.nan], name="x"))

    def test_tz_awareness(self) -> None:
        assert is_tz_aware_utc(UTC_INDEX) is True
        assert is_tz_aware_utc(UTC_INDEX.tz_localize(None)) is False


class TestUtcChecksLeaveNonDatetimeIndicesToTheirOwnCheck:
    """Een index die geen DatetimeIndex is, zakt op `index_is_datetime` of op
    het Pandera-indexveld; de UTC-check zelf zegt daar niets over."""

    @pytest.mark.parametrize(
        "schema", [BarSchema, EventSchema, LabelSchema, OOSPredictionSchema])
    def test_a_plain_index_passes_the_utc_check(self, schema) -> None:
        assert schema.index_is_utc(PLAIN) is True

    @pytest.mark.parametrize("schema", [LabelSchema, OOSPredictionSchema])
    def test_a_naive_index_fails_the_utc_check(self, schema) -> None:
        naive = pd.DataFrame({"a": [1.0, 2.0, 3.0]},
                             index=UTC_INDEX.tz_localize(None))
        assert schema.index_is_utc(naive) is False
        assert schema.index_is_utc(PLAIN.set_index(UTC_INDEX)) is True


class TestEventSchema:
    def test_an_all_nan_feature_column_fails(self) -> None:
        frame = pd.DataFrame({"feat_a": [1.0, 2.0, 3.0],
                              "feat_b": [np.nan] * 3}, index=UTC_INDEX)
        assert EventSchema.no_all_nan_feat_columns(frame) is False
        assert EventSchema.no_all_nan_feat_columns(frame[["feat_a"]]) is True


class TestFeatureBlockSchema:
    def test_index_checks(self) -> None:
        frame = PLAIN.set_index(UTC_INDEX)
        assert FeatureBlockSchema.index_is_datetime(frame) is True
        assert FeatureBlockSchema.index_is_datetime(PLAIN) is False
        assert FeatureBlockSchema.index_monotonic_increasing(frame) is True
        assert FeatureBlockSchema.index_monotonic_increasing(
            frame.iloc[::-1]) is False

    @pytest.mark.parametrize("columns", [
        ["close", "feat_vol_gk", "feat_x_micro", "feat_y_meso", "feat_z_macro",
         "feat_macro_rate", "feat_plain"],
        ["feat_bad_mesoX"],
    ])
    def test_the_tier_suffix_check_accepts_every_name(
        self, columns: list[str],
    ) -> None:
        """Vastgelegd zoals hij IS: de check kan niet falen (zie de toelichting
        bij de `pragma` in `schemas/features.py`). Wordt hij ooit aangescherpt,
        dan hoort deze test mee te veranderen."""
        frame = pd.DataFrame({c: [1.0] for c in columns})
        assert FeatureBlockSchema.feat_columns_have_valid_tier_suffix(frame) is True

    def test_no_all_nan_columns(self) -> None:
        good = pd.DataFrame({"a": [1.0, np.nan], "b": [2.0, 3.0]})
        bad = pd.DataFrame({"a": [np.nan, np.nan], "b": [2.0, 3.0]})
        assert FeatureBlockSchema.no_all_nan_columns(good) is True
        assert FeatureBlockSchema.no_all_nan_columns(bad) is False


class TestFoldAndLabelChecks:
    def test_train_before_test_and_a_positive_embargo(self) -> None:
        folds = pd.DataFrame({"train_end": [10, 20], "test_start": [15, 25],
                              "effective_embargo_bars": [5, 5]})
        assert CPCVFoldSchema.train_before_test_or_no_overlap(folds) is True
        assert CPCVFoldSchema.embargo_covers_t_max(folds) is True
        inverted = folds.assign(test_start=[5, 25], effective_embargo_bars=[0, 5])
        assert CPCVFoldSchema.train_before_test_or_no_overlap(inverted) is False
        assert CPCVFoldSchema.embargo_covers_t_max(inverted) is False

    def test_label_checks(self) -> None:
        labels = pd.DataFrame({"t1_idx": [0, 3, 2], "realized_ret": [0.1, -0.2, 0.0]},
                              index=UTC_INDEX)
        assert LabelSchema.index_monotonic_increasing(labels) is True
        assert LabelSchema.t1_ge_t0(labels) is True
        assert LabelSchema.realized_ret_finite(labels) is True
        backward = labels.assign(t1_idx=[0, 0, 1])
        assert LabelSchema.t1_ge_t0(backward) is False
        assert LabelSchema.realized_ret_finite(
            labels.assign(realized_ret=[0.1, np.inf, 0.0])) is False


# --------------------------------------------------------------------------- #
# Configvalidators
# --------------------------------------------------------------------------- #
def _broken(cls, base, **changes):
    return cls.model_validate({**base.model_dump(), **changes})


class TestConfigValidators:
    def test_a_blank_symbol_is_refused(self) -> None:
        base = load_config(ROOT / "conf/data/default.yaml", DataConfig)
        with pytest.raises(ValidationError, match="lege string"):
            _broken(DataConfig, base, symbols=("BTCUSDT", "  "))

    def test_the_slow_ewma_must_exceed_the_fast_one(self) -> None:
        base = load_config(ROOT / "conf/features/default.yaml", FeatureConfig).momentum
        with pytest.raises(ValidationError, match="ewma_slow_span"):
            _broken(MomentumFeatureConfig, base,
                    ewma_slow_span=base.ewma_fast_span)

    def test_the_signal_cap_must_exceed_the_floor(self) -> None:
        base = AlphaConfig()
        with pytest.raises(ValidationError, match="signal_cap"):
            _broken(AlphaConfig, base, signal_cap=base.signal_floor)

    def test_eta_must_lie_inside_its_own_interval(self) -> None:
        base = load_config(ROOT / "conf/execution/impact.yaml", ImpactConfig)
        with pytest.raises(ValidationError, match="onzekerheidsband"):
            _broken(ImpactConfig, base, eta=base.eta_ci_high * 2.0 + 1.0)

    def test_the_class_ratio_band_must_be_ordered(self) -> None:
        base = MetaLabelingAdequacyConfig()
        with pytest.raises(ValidationError, match="min_positive_class_ratio"):
            _broken(MetaLabelingAdequacyConfig, base,
                    min_positive_class_ratio=base.max_positive_class_ratio)

    def test_the_fracdiff_range_must_be_ordered(self) -> None:
        base = FracDiffConfig()
        with pytest.raises(ValidationError, match="d_lo"):
            _broken(FracDiffConfig, base, d_lo=base.d_hi)

    @pytest.mark.parametrize(("changes", "match"), [
        ({"n_states_grid": ()}, "leeg"),
        ({"dof_min": 50.0, "dof_max": 10.0}, "dof_min"),
        ({"dof_init": 300.0}, "dof_init"),
    ])
    def test_the_hmm_grid_is_sane(self, changes: dict, match: str) -> None:
        with pytest.raises(ValidationError, match=match):
            _broken(M2HmmConfig, M2HmmConfig(), **changes)

    def test_an_empty_meta_label_grid_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="depths is leeg"):
            _broken(MetaLabelConfig, MetaLabelConfig(), depths=())

    def test_the_state_quantiles_must_be_ordered(self) -> None:
        with pytest.raises(ValidationError, match="low_q"):
            _broken(VolStateConfig, VolStateConfig(), low_q=0.8, high_q=0.2)


class TestLoadConfig:
    def test_a_document_that_is_not_a_mapping_is_refused(
        self, tmp_path: Path,
    ) -> None:
        path = tmp_path / "list.yaml"
        path.write_text("- a\n- b\n", encoding="utf-8")
        with pytest.raises(ConfigContractError, match="verwacht een mapping"):
            load_config(path, AlphaConfig)

    def test_the_adequacy_thresholds_come_from_conf(self) -> None:
        assert adequacy_config() == load_config(
            ROOT / "conf/model/adequacy.yaml", type(adequacy_config()))
