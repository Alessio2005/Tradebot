# tests/unit/test_features_small_edges.py
"""De kleine restgaten in `features/`: base, blocks, carry, OI, registry, transforms.

Elk gat is een eigen, kleine tak: een read-only view, een lengtecheck, een
lege uitvoer of een kolomkeuze. De grotere modules (regime, ta, orthogonalize,
cfi, ...) hebben hun eigen bestand.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tradebot.features.base import (
    BaseFeature,
    CertifiedFrame,
    CertifiedPanel,
    DataRegister,
    FeaturePipeline,
    FeatureResult,
    InputSpec,
)
from tradebot.features.blocks import split_feature_blocks, stack_feats
from tradebot.features.funding_carry import compute_funding_features
from tradebot.features.open_interest import compute_oi_features
from tradebot.features.positioning import amihud_illiquidity
from tradebot.features.regime_features import add_regime_features
from tradebot.features.registry import FeatureRegistry
from tradebot.features.transforms import burn_in_of
from tradebot.features.weekly_set import MIN_LIVE_SYMBOLS, market_features
from tradebot.utils.failfast import DataContractError


def _certified(n: int = 12, symbol: str = "BTC") -> CertifiedFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    ns = idx.astype("int64").to_numpy()
    frame = pd.DataFrame({"close": np.linspace(100.0, 110.0, n),
                          "event_ts_ns": ns, "asof_ts_ns": ns}, index=idx)
    return CertifiedFrame(frame=frame, symbol=symbol, data_hashes=(("ohlcv", "abc"),))


class _Echo(BaseFeature):
    """Identiteit op close, met een instelbare burn-in."""

    name = "echo"
    input_spec = InputSpec(datasets=("ohlcv",), columns=("close",))

    def __init__(self, burn_in: int = 0, column: str = "echo_close") -> None:
        super().__init__(params={"burn_in": burn_in, "column": column})
        self._burn_in = burn_in
        self._column = column

    @property
    def burn_in_period(self) -> int:
        return self._burn_in

    @property
    def output_columns(self) -> tuple[str, ...]:
        return (self._column,)

    def _compute(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame({self._column: frame["close"].astype("float64")}, index=frame.index)
        out.iloc[: self._burn_in] = np.nan
        return out


# --------------------------------------------------------------------------- #
# features/base.py
# --------------------------------------------------------------------------- #
class TestBaseViews:
    def test_the_register_exposes_its_hashes_read_only(self, tmp_path: Path) -> None:
        path = tmp_path / "data_hashes.json"
        path.write_text(json.dumps({"crypto/ohlcv/BTC/1d": "deadbeef"}), encoding="utf-8")
        hashes = DataRegister(path).hashes
        assert hashes["crypto/ohlcv/BTC/1d"] == "deadbeef"
        with pytest.raises(TypeError):
            hashes["x"] = "y"                                    # type: ignore[index]

    def test_a_panel_lists_its_symbols_in_column_order(self) -> None:
        idx = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
        panel = CertifiedPanel(values=pd.DataFrame({"B": [1.0, 2.0, 3.0], "A": [3.0, 2.0, 1.0]}, index=idx),
                               data_hashes=(("ohlcv", "h"),))
        assert panel.symbols == ("B", "A")

    def test_a_feature_result_reports_its_shape_and_its_gaps(self) -> None:
        res = _Echo(burn_in=3).transform(_certified(10))
        assert isinstance(res, FeatureResult)
        assert res.columns == ("echo_close",) and res.n_rows == 10 and res.n_nan == 3

    def test_a_pipeline_exposes_the_features_it_was_built_from(self) -> None:
        feat = _Echo()
        assert FeaturePipeline([feat]).features == (feat,)

    def test_an_empty_input_passes_the_contract_without_a_burn_in_check(self) -> None:
        res = _Echo(burn_in=5).transform(_certified(0))
        assert res.n_rows == 0 and res.n_nan == 0


class TestRegistryDefinitions:
    def test_definitions_follow_the_registration_order(self) -> None:
        reg = FeatureRegistry()
        reg.register_all([_Echo(burn_in=1), _Echo(burn_in=2, column="echo_other")])
        defs = reg.definitions()
        assert [d.burn_in_period for d in defs] == [1, 2]
        assert [d.output_columns for d in defs] == [("echo_close",), ("echo_other",)]
        assert all(d.class_name == "_Echo" for d in defs)


# --------------------------------------------------------------------------- #
# features/blocks.py
# --------------------------------------------------------------------------- #
class TestBlocks:
    def test_stack_feats_promotes_1d_blocks_and_skips_empty_or_missing_ones(self) -> None:
        x1 = np.arange(4.0)                                     # 1D -> (4, 1)
        x4 = np.arange(8.0).reshape(4, 2)
        out = stack_feats(x1, x4, None)
        assert out.shape == (4, 3)
        np.testing.assert_array_equal(out[:, 0], x1)
        assert stack_feats(x1, np.empty((4, 0)), x4).shape == (4, 3)

    def test_stack_feats_of_nothing_is_an_empty_array(self) -> None:
        assert stack_feats(None, np.array([]), None).size == 0

    def test_split_feature_blocks_returns_empty_tiers_with_the_right_row_count(self) -> None:
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        micro, meso, macro = split_feature_blocks(df, {"micro": ["a", "missing"], "macro": ["b"]})
        assert micro.shape == (3, 1) and meso.shape == (3, 0) and macro.shape == (3, 1)
        assert micro.dtype == np.float64


# --------------------------------------------------------------------------- #
# funding, open interest, positioning, regime features
# --------------------------------------------------------------------------- #
class TestLengthChecks:
    def test_funding_rejects_a_series_of_the_wrong_length(self) -> None:
        idx = pd.date_range("2024-01-01", periods=10, freq="h", tz="UTC")
        with pytest.raises(ValueError, match="funding_per_bar length 9"):
            compute_funding_features(idx, np.zeros(9))

    def test_open_interest_rejects_a_close_series_of_the_wrong_length(self) -> None:
        idx = pd.date_range("2024-01-01", periods=60, freq="h", tz="UTC")
        with pytest.raises(ValueError, match="close length"):
            compute_oi_features(idx, np.full(60, 1_000.0), close=np.ones(59))


class TestAmihud:
    @pytest.mark.parametrize("volume_col", ["real_volume", "tick_volume", "volume"])
    def test_it_reads_the_first_available_volume_column(self, volume_col: str) -> None:
        close = 100 * np.exp(np.cumsum(np.random.default_rng(0).normal(0, 0.01, 60)))
        df = pd.DataFrame({"close": close, volume_col: np.full(60, 50.0)})
        out = amihud_illiquidity(df, window=5)
        assert out.name == "feat_amihud" and len(out) == 60 and (out.dropna() >= 0).all()

    def test_thin_volume_is_more_illiquid_than_thick_volume(self) -> None:
        close = 100 * np.exp(np.cumsum(np.random.default_rng(1).normal(0, 0.01, 80)))
        thin = amihud_illiquidity(pd.DataFrame({"close": close, "volume": np.full(80, 10.0)}))
        thick = amihud_illiquidity(pd.DataFrame({"close": close, "volume": np.full(80, 1_000.0)}))
        assert thin.iloc[-1] > thick.iloc[-1]


class TestRegimeFeatures:
    def test_a_frame_without_close_is_returned_untouched_with_a_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        df = pd.DataFrame({"open": [1.0, 2.0]})
        with caplog.at_level(logging.WARNING, logger="tradebot.features.regime_features"):
            out = add_regime_features(df)
        assert out is df and list(out.columns) == ["open"] and "regime features skipped" in caplog.text


# --------------------------------------------------------------------------- #
# transforms.burn_in_of and the weekly market factor
# --------------------------------------------------------------------------- #
class TestBurnIn:
    def test_each_transform_declares_its_own_burn_in(self) -> None:
        assert burn_in_of("rolling_log_return", {"window": 20, "skip_bars": 5}) == 25
        assert burn_in_of("expanding_zscore", {"min_periods": 30}) == 29
        assert burn_in_of("rolling_quantile_winsorise", {"window": 60}) == 59
        assert burn_in_of("cross_sectional_rank", {}) == 0

    def test_an_unknown_transform_is_refused(self) -> None:
        with pytest.raises(DataContractError, match="Onbekende transform"):
            burn_in_of("not_a_transform", {})


class TestMarketFeatures:
    def test_a_window_with_a_constant_symbol_gets_no_market_factor(self) -> None:
        rng = np.random.default_rng(0)
        n, k = 40, MIN_LIVE_SYMBOLS + 1
        rets = pd.DataFrame(rng.normal(0, 0.01, (n, k)),
                            index=pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC"),
                            columns=[f"S{i}" for i in range(k)])
        rets["S0"] = 0.0                                        # std == 0 in elk venster
        out = market_features(rets, window=10)
        assert set(out) == {"mkt_ret5", "resid_z20", "avg_corr60"}
        assert all(panel.isna().all().all() for panel in out.values())
