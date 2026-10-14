# tests/unit/test_meta_labeling_edges.py
"""`labeling/meta.py`: de scouts, de meta-labelengine en de beslisfilter.

De engine krijgt hier eerst handgemaakte `ScoutResult`s, zodat elke labelregel
(flat, t-stat, funding, VWPF) een eigen, leesbare casus is; de scouts draaien op
een gesimuleerde prijsreeks met genoeg beweging voor CUSUM-events.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from tradebot.labeling import meta
from tradebot.labeling.meta import (
    CausalScout,
    MetaLabelFilter,
    MetaLabelingEngine,
    PrimaryScout,
    ScoutResult,
    assert_training_scout_is_causal,
)
from tradebot.labeling.trend_scanning import TrendScanningLabeler


def _market(n: int = 500, seed: int = 0, sigma: float = 0.01) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    close = 100 * np.exp(np.cumsum(rng.normal(0, sigma, n)))
    open_ = np.r_[close[0], close[:-1]]
    return pd.DataFrame({"open": open_, "high": np.maximum(open_, close) * 1.003,
                         "low": np.minimum(open_, close) * 0.997, "close": close}, index=idx)


def _scout(directions, rets, tstats=None, events=None) -> ScoutResult:
    n = len(directions)
    return ScoutResult(
        event_indices=np.asarray(events if events is not None else np.arange(n) * 3 + 10, dtype=np.int32),
        directions=np.asarray(directions, dtype=np.int8),
        realized_ret=np.asarray(rets, dtype=np.float64),
        t1_indices=np.asarray(events if events is not None else np.arange(n) * 3 + 10, dtype=np.int32) + 5,
        t_stats=np.asarray(tstats if tstats is not None else np.full(n, 3.0), dtype=np.float64),
    )


# --------------------------------------------------------------------------- #
# leakage guard
# --------------------------------------------------------------------------- #
class TestLeakageGuard:
    def test_a_forward_looking_labeler_is_refused_as_instance_and_as_class(self) -> None:
        with pytest.raises(ValueError, match="TrendScanningLabeler"):
            assert_training_scout_is_causal(TrendScanningLabeler())
        with pytest.raises(ValueError, match="TrendScanningLabeler"):
            assert_training_scout_is_causal(TrendScanningLabeler)

    def test_a_primary_scout_is_only_causal_with_the_flag(self) -> None:
        with pytest.raises(ValueError, match="causal_only=True"):
            assert_training_scout_is_causal(PrimaryScout(causal_only=False))
        assert_training_scout_is_causal(PrimaryScout(causal_only=True))
        assert_training_scout_is_causal(CausalScout())
        assert_training_scout_is_causal(object())

    @pytest.mark.parametrize(("value", "strict"), [("1", True), ("TRUE", True), ("yes", True),
                                                   ("0", False), ("", False), ("no", False)])
    def test_the_strict_mode_follows_the_environment(self, monkeypatch: pytest.MonkeyPatch,
                                                     value: str, strict: bool) -> None:
        monkeypatch.setenv("TRADEBOT_STRICT_CAUSAL", value)
        assert meta._strict_causal_required() is strict

    def test_the_default_mode_is_not_strict(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("TRADEBOT_STRICT_CAUSAL", raising=False)
        assert meta._strict_causal_required() is False


# --------------------------------------------------------------------------- #
# PrimaryScout
# --------------------------------------------------------------------------- #
class TestPrimaryScoutConfiguration:
    def test_a_cusum_multiplier_far_from_the_training_filter_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger=meta.logger.name):
            PrimaryScout(cusum_multiplier=0.5, training_cusum_multiplier=2.0)
            assert "cusum mismatch" in caplog.text
            caplog.clear()
            PrimaryScout(cusum_multiplier=1.0, training_cusum_multiplier=1.2)
        assert "cusum mismatch" not in caplog.text

    def test_the_factory_reads_the_spread_from_the_symbol_yaml(self, tmp_path: Path) -> None:
        (tmp_path / "BTCUSDT.yaml").write_text(yaml.safe_dump({"spread": 0.0008}))
        (tmp_path / "ETHUSDT.yaml").write_text(yaml.safe_dump({"training": {"spread": 0.0012}}))
        (tmp_path / "SOLUSDT.yaml").write_text("")
        assert PrimaryScout.from_config("BTCUSDT", tmp_path, t_min=4).half_spread == pytest.approx(0.0004)
        assert PrimaryScout.from_config("ETHUSDT", tmp_path).half_spread == pytest.approx(0.0006)
        assert PrimaryScout.from_config("SOLUSDT", tmp_path, global_spread=0.002).half_spread == pytest.approx(0.001)

    def test_the_factory_overrides_a_caller_supplied_half_spread(self, tmp_path: Path) -> None:
        (tmp_path / "BTCUSDT.yaml").write_text(yaml.safe_dump({"spread": 0.0008}))
        scout = PrimaryScout.from_config("BTCUSDT", tmp_path, half_spread=0.0)
        assert scout.half_spread == pytest.approx(0.0004)

    def test_a_missing_yaml_falls_back_to_the_global_spread_with_a_warning(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.chdir(tmp_path)
        with caplog.at_level(logging.WARNING, logger=meta.logger.name):
            scout = PrimaryScout.from_config("NOPE", global_spread=0.004)
        assert scout.half_spread == pytest.approx(0.002) and "niet gevonden" in caplog.text


class TestPrimaryScoutGenerate:
    DF = _market(600, seed=1, sigma=0.012)

    def test_the_trend_scan_scout_labels_directions_and_exits(self) -> None:
        out = PrimaryScout(t_min=3, t_max=12, min_tstat=1.5).generate(self.DF)
        n = len(out.event_indices)
        assert n > 10 and out.directions.shape == out.realized_ret.shape == out.t1_indices.shape == (n,)
        assert set(np.unique(out.directions)) <= {-1, 0, 1}
        live = (out.directions != 0) & (out.t1_indices >= 0)  # -1 = te weinig toekomst voor een label
        assert live.any() and (out.t1_indices[live] >= out.event_indices[live]).all()

    def test_the_causal_scout_mode_uses_the_ema_crossover_and_has_no_t_stats(self) -> None:
        out = PrimaryScout(t_min=5, t_max=30, causal_only=True).generate(self.DF)
        assert len(out.event_indices) > 10 and np.isnan(out.t_stats).all()
        assert set(np.unique(out.directions)) <= {-1, 0, 1}

    def test_a_threshold_nothing_reaches_gives_an_empty_result_and_a_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger=meta.logger.name):
            out = PrimaryScout(cusum_multiplier=1e6).generate(self.DF)
        assert out.event_indices.size == 0 and "0 events" in caplog.text

    def test_a_tstat_bar_nothing_clears_leaves_flat_events_with_the_minimum_horizon(self) -> None:
        scout = PrimaryScout(t_min=3, t_max=12, min_tstat=1e6)
        out = scout.generate(self.DF)
        assert (out.directions == 0).all()
        np.testing.assert_array_equal(out.t1_indices, np.minimum(out.event_indices + 3, len(self.DF) - 1))

    def test_bid_ask_columns_silence_the_mid_price_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        df = self.DF.assign(bid_high=self.DF["high"], bid_low=self.DF["low"], ask_high=self.DF["high"],
                            ask_low=self.DF["low"])
        with caplog.at_level(logging.WARNING, logger=meta.logger.name):
            PrimaryScout(half_spread=0.0002).generate(df)
            assert "mid-market" not in caplog.text
            PrimaryScout().generate(self.DF)
        assert "mid-market" in caplog.text

    def test_the_atr_comes_from_the_caller_a_vol_feature_or_the_true_range(self) -> None:
        close = self.DF["close"].to_numpy()
        given = PrimaryScout._resolve_atr(self.DF, close, np.arange(len(close), dtype=float))
        assert given[3] == 3.0
        feat = PrimaryScout._resolve_atr(self.DF.assign(feat_vol_gk=0.5), close, None)
        assert (feat == 0.5).all()
        tr = PrimaryScout._resolve_atr(self.DF, close, None)
        assert np.isfinite(tr).all() and (tr > 0).all()
        bare = PrimaryScout._resolve_atr(self.DF[["close"]], close, None)
        assert np.isfinite(bare).all()

    def test_an_events_frame_without_open_falls_back_to_the_close(self) -> None:
        out = PrimaryScout().generate(self.DF.drop(columns=["open"]))
        assert len(out.event_indices) > 0


# --------------------------------------------------------------------------- #
# CausalScout
# --------------------------------------------------------------------------- #
class TestCausalScout:
    DF = _market(600, seed=2, sigma=0.012)

    def test_too_few_bars_for_the_slow_ema_give_an_empty_result(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger=meta.logger.name):
            out = CausalScout(slow_ema=50).generate(self.DF.iloc[:40])
        assert out.event_indices.size == 0 and "onvoldoende bars" in caplog.text

    def test_a_threshold_nothing_reaches_gives_an_empty_result(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger=meta.logger.name):
            out = CausalScout(cusum_multiplier=1e6).generate(self.DF)
        assert out.event_indices.size == 0 and "0 events" in caplog.text

    def test_directions_follow_the_crossover_and_the_warm_up_is_flat(self) -> None:
        out = CausalScout(fast_ema=5, slow_ema=60, cusum_multiplier=0.3, t_horizon=10).generate(self.DF)
        early = out.event_indices < 60
        assert early.any() and (out.directions[early] == 0).all()
        assert set(np.unique(out.directions[~early])) <= {-1, 1}
        np.testing.assert_array_equal(
            out.t1_indices[early], np.minimum(out.event_indices[early] + 10, len(self.DF) - 1))
        assert np.isnan(out.t_stats).all()

    def test_bid_ask_columns_and_a_given_atr_are_used(self) -> None:
        df = self.DF.assign(bid_high=self.DF["high"], bid_low=self.DF["low"], ask_high=self.DF["high"],
                            ask_low=self.DF["low"]).drop(columns=["open"])
        atr = np.full(len(df), 0.5)
        out = CausalScout(cusum_multiplier=1.0).generate(df, atr=atr)
        assert len(out.event_indices) > 0

    def test_a_series_that_only_rises_never_produces_a_short(self) -> None:
        n = 300
        rng = np.random.default_rng(0)
        close = 100 + 0.25 * np.arange(n) + rng.normal(0, 0.05, n)
        df = pd.DataFrame({"open": close - 0.1, "high": close + 0.3, "low": close - 0.3, "close": close},
                          index=pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"))
        out = CausalScout(fast_ema=5, slow_ema=60, cusum_multiplier=0.5).generate(df)
        assert (out.directions >= 0).all() and (out.directions == 1).any()

    def test_the_factory_carries_the_symbol_spread_over(self, tmp_path: Path) -> None:
        (tmp_path / "BTCUSDT.yaml").write_text(yaml.safe_dump({"spread": 0.0006}))
        scout = CausalScout.from_config("BTCUSDT", tmp_path, fast_ema=7)
        assert scout.half_spread == pytest.approx(0.0003) and scout.fast_ema == 7


# --------------------------------------------------------------------------- #
# meta labels
# --------------------------------------------------------------------------- #
class TestMetaLabels:
    def test_the_threshold_includes_the_round_trip_spread_unless_the_barrier_already_has_it(self) -> None:
        assert MetaLabelingEngine(profit_threshold=0.0005, spread=0.001).profit_threshold == 0.002
        assert MetaLabelingEngine(profit_threshold=0.005, spread=0.001).profit_threshold == 0.005
        assert MetaLabelingEngine(profit_threshold=0.0005, spread=0.001,
                                  barrier_uses_spread=True).profit_threshold == 0.0005
        assert MetaLabelingEngine(profit_threshold=-0.01).profit_threshold == 0.0

    def test_wins_losses_and_flat_events(self) -> None:
        sc = _scout([1, -1, 0, 1, 1], [0.02, 0.03, 0.05, -0.01, 0.004])
        eng = MetaLabelingEngine(profit_threshold=0.005)
        assert eng.make_meta_labels(sc).tolist() == [1, 1, 0, 0, 0]
        assert MetaLabelingEngine(profit_threshold=0.005, include_flat=True).make_meta_labels(sc).tolist() == [1, 1, 1, 0, 0]

    def test_a_weak_trend_statistic_blocks_the_label(self) -> None:
        sc = _scout([1, 1, 1], [0.02, 0.02, 0.02], tstats=[3.0, -0.5, 2.0])
        assert MetaLabelingEngine(min_tstat_filter=1.0).make_meta_labels(sc).tolist() == [1, 0, 1]

    def test_funding_is_subtracted_before_the_threshold(self, caplog: pytest.LogCaptureFixture) -> None:
        sc = _scout([1, -1, 1], [0.02, 0.02, 0.02])
        eng = MetaLabelingEngine()
        with caplog.at_level(logging.INFO, logger=meta.logger.name):
            y = eng.make_meta_labels(sc, funding_per_event=np.array([0.03, 0.03, 0.0]))
        # long betaalt 3 %, short ontvangt 3 %, de laatste betaalt niets.
        assert y.tolist() == [0, 1, 1] and "funding_flipped_to_loss=1" in caplog.text
        override = eng.make_meta_labels(sc, funding_per_event=np.array([0.03, 0.03, 0.0]),
                                        signed_directions_for_funding=np.array([-1, -1, -1]))
        assert override.tolist() == [1, 1, 1] or override.tolist() == [1, 1, 1]

    def test_funding_of_the_wrong_length_is_ignored_with_a_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        sc = _scout([1, 1], [0.02, 0.02])
        with caplog.at_level(logging.WARNING, logger=meta.logger.name):
            y = MetaLabelingEngine().make_meta_labels(sc, funding_per_event=np.array([0.5]))
        assert y.tolist() == [1, 1] and "funding adjustment uitgeschakeld" in caplog.text

    def test_a_win_that_could_not_have_been_filled_stays_a_loss(self) -> None:
        sc = _scout([1, 1, 1, 1], [0.02] * 4)
        eng = MetaLabelingEngine(vwpf_min_participation=0.05, vwpf_max_participation=0.20)
        y = eng.make_meta_labels(sc, bar_volumes=np.array([100.0] * 4),
                                 order_sizes=np.array([1.0, 10.0, 50.0, 15.0]))
        assert y.tolist() == [0, 1, 0, 1]   # te dun, vulbaar, eigen impact, vulbaar
        # Zonder volumes of met een open band wordt er niet getoetst.
        assert eng.make_meta_labels(sc).tolist() == [1, 1, 1, 1]
        open_band = MetaLabelingEngine(vwpf_min_participation=0.0, vwpf_max_participation=1.0)
        assert open_band.make_meta_labels(sc, bar_volumes=np.ones(4), order_sizes=np.full(4, 5.0)).tolist() == [1] * 4

    def test_no_events_give_no_labels(self) -> None:
        empty = _scout([], [], events=[])
        assert MetaLabelingEngine().make_meta_labels(empty).size == 0


# --------------------------------------------------------------------------- #
# dataset
# --------------------------------------------------------------------------- #
class TestBuildDataset:
    DF = _market(600, seed=3, sigma=0.012)
    SCOUT = CausalScout(fast_ema=5, slow_ema=60, cusum_multiplier=0.3, t_horizon=10).generate(DF)

    def _features(self, k: int = 3):
        n = len(self.DF)
        rng = np.random.default_rng(0)
        return rng.normal(size=(n, k)), rng.normal(size=(n, 2)), rng.normal(size=(n, 1))

    def test_flat_events_are_dropped_and_every_array_is_aligned_to_the_remaining_events(self) -> None:
        micro, meso, macro = self._features()
        ds = MetaLabelingEngine().build_dataset(self.SCOUT, micro, meso, macro, self.DF.index,
                                                scout_for_audit=CausalScout())
        active = self.SCOUT.directions != 0
        n = int(active.sum())
        assert ds.X_micro.shape == (n, 3) and ds.X_micro.dtype == np.float32
        assert ds.y_meta.shape == ds.directions.shape == ds.weights.shape == ds.t1_indices.shape == (n,)
        assert len(ds.timestamps) == n and (ds.directions != 0).all() and (ds.weights == 1.0).all()

    def test_flat_events_stay_when_requested_and_empty_blocks_become_zero_width(self) -> None:
        micro, _, _ = self._features()
        ds = MetaLabelingEngine(include_flat=True).build_dataset(
            self.SCOUT, micro, np.zeros((0, 0)), np.zeros((0, 0)), self.DF.index,
            scout_for_audit=PrimaryScout(causal_only=True))
        assert ds.X_meso.shape == (len(self.SCOUT.event_indices), 0) and ds.X_macro.shape[1] == 0

    def test_the_leakage_guard_runs_when_an_audit_scout_is_given_and_is_fatal_in_strict_mode(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        micro, meso, macro = self._features()
        eng = MetaLabelingEngine()
        with pytest.raises(ValueError, match="TrendScanningLabeler"):
            eng.build_dataset(self.SCOUT, micro, meso, macro, self.DF.index,
                              scout_for_audit=TrendScanningLabeler())
        monkeypatch.delenv("TRADEBOT_STRICT_CAUSAL", raising=False)
        with caplog.at_level(logging.WARNING, logger=meta.logger.name):
            eng.build_dataset(self.SCOUT, micro, meso, macro, self.DF.index)
        assert "scout_for_audit not supplied" in caplog.text
        monkeypatch.setenv("TRADEBOT_STRICT_CAUSAL", "1")
        with pytest.raises(ValueError, match="STRICT MODE"):
            eng.build_dataset(self.SCOUT, micro, meso, macro, self.DF.index)

    def test_sample_weights_use_the_global_uniqueness_unless_train_indices_are_given(
        self, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from tradebot.cv import uniqueness

        micro, meso, macro = self._features()
        seen: dict = {}

        def weights(ts, uniq, rets):
            seen.update(n_ts=len(ts), n_uniq=len(uniq), n_rets=len(rets))
            return np.ones(len(ts))

        eng = MetaLabelingEngine()
        with caplog.at_level(logging.WARNING, logger=meta.logger.name):
            ds = eng.build_dataset(self.SCOUT, micro, meso, macro, self.DF.index,
                                   sample_weight_fn=weights, scout_for_audit=CausalScout())
        n = len(ds.y_meta)
        assert "train_indices niet opgegeven" in caplog.text
        assert seen == {"n_ts": n, "n_uniq": n, "n_rets": n} and (ds.weights == 1.0).all()

        calls: list[np.ndarray] = []

        def per_fold(index, t1, train):
            calls.append(np.asarray(train))
            return np.ones(len(train))

        monkeypatch.setattr(uniqueness, "get_average_uniqueness_per_fold", per_fold)
        train = np.arange(0, n // 2)
        eng.build_dataset(self.SCOUT, micro, meso, macro, self.DF.index, sample_weight_fn=weights,
                          scout_for_audit=CausalScout(), train_indices=train)
        assert len(calls) == 1 and np.array_equal(calls[0], train)

    def test_active_events_can_be_filtered_out_of_a_scout_result(self) -> None:
        sc = _scout([1, 0, -1, 0], [0.1, 0.2, 0.3, 0.4])
        out = MetaLabelingEngine.filter_active_events(sc)
        assert out.directions.tolist() == [1, -1] and out.realized_ret.tolist() == [0.1, 0.3]
        assert out.event_indices.tolist() == sc.event_indices[[0, 2]].tolist()


# --------------------------------------------------------------------------- #
# filter
# --------------------------------------------------------------------------- #
class TestMetaLabelFilter:
    def test_a_flat_scout_or_a_low_judge_probability_blocks_the_trade(self) -> None:
        f = MetaLabelFilter(prob_threshold=0.4)
        assert f.decide(0, 0.99) == 0 and f.decide(1, 0.39) == 0
        assert f.decide(1, 0.4) == 1 and f.decide(-1, 0.9) == -1

    def test_direction_agreement_is_only_enforced_when_asked_and_when_the_judge_reports_a_side(self) -> None:
        strict = MetaLabelFilter(prob_threshold=0.4, require_direction_agreement=True)
        assert strict.decide(1, 0.9, "LONG") == 1 and strict.decide(-1, 0.9, "SHORT") == -1
        assert strict.decide(1, 0.9, "SHORT") == 0 and strict.decide(-1, 0.9, "long") == 0
        assert strict.decide(1, 0.9, None) == 1
        assert MetaLabelFilter(prob_threshold=0.4).decide(1, 0.9, "SHORT") == 1

    def test_the_batch_version_matches_the_scalar_one(self) -> None:
        f = MetaLabelFilter(prob_threshold=0.5)
        dirs = np.array([1, -1, 0, 1, -1], dtype=np.int8)
        probs = np.array([0.9, 0.6, 0.99, 0.2, 0.5])
        batch = f.batch_decide(dirs, probs)
        assert batch.dtype == np.int8
        assert batch.tolist() == [f.decide(int(d), float(p)) for d, p in zip(dirs, probs, strict=True)]

    def test_the_repr_states_its_parameters(self) -> None:
        assert repr(MetaLabelFilter(0.3, True)) == "MetaLabelFilter(prob_threshold=0.3, require_direction_agreement=True)"
