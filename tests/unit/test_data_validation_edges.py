# tests/unit/test_data_validation_edges.py
"""Restgaten van de L0-validatielaag (`data/validation/`): schema, gaps, outliers en continuiteit.

Het hoofdcontract (elke validator raiset, niets wordt geïnterpoleerd) staat in `test_data_validation.py`; dit bestand
voegt de randen toe: ontbrekende kolommen, te korte reeksen, lege ledgers en de niet-fatale paden.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from tradebot.data.validation import (
    FUNDING_SPEC,
    OHLCV_SPEC,
    OPEN_INTEREST_SPEC,
    AdjustmentFactor,
    AdjustmentLedger,
    GapLedger,
    GapRecord,
    OutlierReport,
    SymbolLifecycle,
    apply_adjustments,
    detect_gaps,
    detect_outliers,
    enforce_gap_policy,
    enforce_outlier_policy,
    validate_continuity,
    validate_schema,
)
from tradebot.data.validation.gaps import cadence_ns
from tradebot.utils.failfast import DataContractError

DAY = 86_400_000_000_000


def _frame(n: int = 10) -> pd.DataFrame:
    ev = np.arange(n, dtype=np.int64) * DAY
    close = 100.0 + np.arange(n, dtype=float)
    return pd.DataFrame({"event_ts_ns": ev, "asof_ts_ns": ev + DAY, "open": close, "high": close + 1.0,
                         "low": close - 1.0, "close": close, "volume": np.full(n, 5.0)})


# --------------------------------------------------------------------------- #
# schema
# --------------------------------------------------------------------------- #
class TestSchemaEdges:
    @pytest.mark.parametrize("col", ["event_ts_ns", "asof_ts_ns"])
    def test_a_missing_time_column_is_named(self, col: str) -> None:
        with pytest.raises(DataContractError, match=col):
            validate_schema(_frame().drop(columns=[col]), OHLCV_SPEC)

    def test_a_float_asof_column_is_refused(self) -> None:
        df = _frame().astype({"asof_ts_ns": float})
        with pytest.raises(DataContractError, match="asof_ts_ns moet int64"):
            validate_schema(df, OHLCV_SPEC, context="bybit")

    def test_unsorted_event_times_are_refused(self) -> None:
        df = _frame().iloc[::-1].reset_index(drop=True)
        with pytest.raises(DataContractError, match="niet oplopend"):
            validate_schema(df, OHLCV_SPEC)

    def test_the_context_is_part_of_the_message(self) -> None:
        with pytest.raises(DataContractError, match=r"\[bybit/BTCUSDT\]"):
            validate_schema(_frame().iloc[0:0], OHLCV_SPEC, context="bybit/BTCUSDT")

    def test_a_missing_required_numeric_column_is_named(self) -> None:
        with pytest.raises(DataContractError, match="'volume' ontbreekt"):
            validate_schema(_frame().drop(columns=["volume"]), OHLCV_SPEC)

    def test_a_non_numeric_column_is_refused(self) -> None:
        df = _frame().astype({"close": str})
        with pytest.raises(DataContractError, match="niet numeriek"):
            validate_schema(df, OHLCV_SPEC)

    def test_infinite_values_count_as_missing(self) -> None:
        df = _frame()
        df.loc[3, "volume"] = np.inf
        with pytest.raises(DataContractError, match="1 NaN/inf"):
            validate_schema(df, OHLCV_SPEC)

    def test_the_funding_series_needs_its_interval_column(self) -> None:
        ev = np.arange(3, dtype=np.int64) * DAY
        df = pd.DataFrame({"event_ts_ns": ev, "asof_ts_ns": ev, "funding_rate": [0.0001, 0.0002, -0.0001]})
        with pytest.raises(DataContractError, match="funding_interval_hours"):
            validate_schema(df, FUNDING_SPEC)
        assert validate_schema(df.assign(funding_interval_hours=8), FUNDING_SPEC) is None

    def test_open_interest_may_be_zero_but_not_negative(self) -> None:
        ev = np.arange(3, dtype=np.int64) * DAY
        df = pd.DataFrame({"event_ts_ns": ev, "asof_ts_ns": ev, "open_interest": [0.0, 5.0, 7.0]})
        assert validate_schema(df, OPEN_INTEREST_SPEC) is None
        df.loc[1, "open_interest"] = -1.0
        with pytest.raises(DataContractError, match="negatieve"):
            validate_schema(df, OPEN_INTEREST_SPEC)

    def test_a_high_below_the_body_and_a_low_above_it_are_both_counted(self) -> None:
        df = _frame()
        df.loc[2, "high"] = df.loc[2, "close"] - 0.5
        df.loc[4, "low"] = df.loc[4, "open"] + 0.5
        with pytest.raises(DataContractError, match="OHLC-ordening") as err:
            validate_schema(df, OHLCV_SPEC)
        assert "n_high_violations=1" in str(err.value) and "n_low_violations=1" in str(err.value)


# --------------------------------------------------------------------------- #
# gaps
# --------------------------------------------------------------------------- #
class TestGapEdges:
    def test_the_cadence_is_known_for_every_granularity_in_the_config(self) -> None:
        assert cadence_ns("1h") == 3600 * 10**9 and cadence_ns("1d") == DAY

    def test_a_frame_without_event_times_is_refused(self) -> None:
        with pytest.raises(DataContractError, match="event_ts_ns"):
            detect_gaps(pd.DataFrame({"x": [1]}), asset_class="crypto", symbol="X", granularity="1d")

    def test_an_unsorted_frame_is_refused(self) -> None:
        df = pd.DataFrame({"event_ts_ns": [DAY, 0]})
        with pytest.raises(DataContractError, match="oplopend"):
            detect_gaps(df, asset_class="crypto", symbol="X", granularity="1d")

    def test_fewer_than_two_bars_cannot_have_a_gap(self) -> None:
        df = pd.DataFrame({"event_ts_ns": [0]})
        assert detect_gaps(df, asset_class="crypto", symbol="X", granularity="1d") == []

    def test_a_late_bar_that_does_not_skip_a_whole_cadence_is_not_a_gap(self) -> None:
        df = pd.DataFrame({"event_ts_ns": [0, DAY + DAY // 2, 3 * DAY]})
        assert detect_gaps(df, asset_class="crypto", symbol="X", granularity="1d") == []

    def test_the_iso_properties_render_utc_timestamps(self) -> None:
        rec = GapRecord("crypto", "X", "1d", 0, 3 * DAY, 2)
        assert rec.gap_start_iso.startswith("1970-01-01 00:00:00") and rec.gap_end_iso.startswith("1970-01-04")

    def test_an_empty_ledger_reads_as_an_empty_typed_frame(self, tmp_path) -> None:
        ledger = GapLedger(tmp_path / "sub" / "gaps.jsonl")
        ledger.append([])
        assert not ledger.path.exists()
        assert ledger.read_all().empty and "n_missing" in ledger.read_all().columns
        assert ledger.summary().empty

    def test_blank_lines_in_the_ledger_are_ignored(self, tmp_path) -> None:
        ledger = GapLedger(tmp_path / "gaps.jsonl")
        ledger.append([GapRecord("crypto", "X", "1d", 0, 3 * DAY, 2)])
        with open(ledger.path, "a", encoding="utf-8") as fh:
            fh.write("\n   \n")
        assert len(ledger.read_all()) == 1

    def test_the_reject_message_names_the_context_and_the_first_gap(self, tmp_path) -> None:
        ledger = GapLedger(tmp_path / "gaps.jsonl")
        records = [GapRecord("crypto", "X", "1d", 0, 3 * DAY, 2), GapRecord("crypto", "X", "1d", 5 * DAY, 8 * DAY, 2)]
        with pytest.raises(DataContractError, match=r"2 gat\(en\), samen 4 .* in bybit") as err:
            enforce_gap_policy(records, policy="reject", ledger=ledger, context="bybit")
        assert "1970-01-01" in str(err.value) and len(ledger.read_all()) == 2

    def test_without_gaps_the_reject_policy_passes(self, tmp_path) -> None:
        assert enforce_gap_policy([], policy="reject", ledger=GapLedger(tmp_path / "g.jsonl")) is None


# --------------------------------------------------------------------------- #
# outliers
# --------------------------------------------------------------------------- #
class TestOutlierEdges:
    def test_a_frame_without_close_is_refused(self) -> None:
        with pytest.raises(DataContractError, match="close-kolom"):
            detect_outliers(pd.DataFrame({"x": [1.0]}), symbol="X", granularity="1d", max_abs_log_return=0.5)

    @pytest.mark.parametrize("threshold", [0.0, -0.1])
    def test_a_non_positive_threshold_is_refused(self, threshold: float) -> None:
        with pytest.raises(DataContractError, match="positief"):
            detect_outliers(_frame(), symbol="X", granularity="1d", max_abs_log_return=threshold)

    def test_a_single_bar_has_no_returns_and_no_outliers(self) -> None:
        report = detect_outliers(_frame(1), symbol="X", granularity="1d", max_abs_log_return=0.5)
        assert report == OutlierReport("X", "1d", 1, 0, 0, 0, 0.0) and report.total == 0

    def test_frames_without_volume_or_range_columns_are_measured_on_close_only(self) -> None:
        df = pd.DataFrame({"event_ts_ns": np.arange(4) * DAY, "close": [100.0, 100.0, 200.0, 200.0]})
        report = detect_outliers(df, symbol="X", granularity="1d", max_abs_log_return=0.5)
        assert (report.n_price_jumps, report.n_zero_volume, report.n_inverted_range) == (1, 0, 0)
        assert report.max_abs_log_return == pytest.approx(math.log(2))

    def test_the_worst_jumps_are_ranked_and_capped(self) -> None:
        close = np.array([100.0, 150.0, 100.0, 300.0, 120.0, 110.0])     # |returns|: .41 .41 1.10 .92 .09
        df = pd.DataFrame({"event_ts_ns": np.arange(6) * DAY, "close": close})
        report = detect_outliers(df, symbol="X", granularity="1d", max_abs_log_return=0.3, top_n=2)
        assert report.n_price_jumps == 4 and len(report.worst_jumps) == 2
        assert report.worst_jumps[0][1] == pytest.approx(math.log(3)) and report.worst_jumps[0][0].startswith("1970-01-04")
        assert report.worst_jumps[1][1] == pytest.approx(math.log(0.4)) and report.worst_jumps[1][0].startswith("1970-01-05")

    def test_a_zero_close_is_neutralised_instead_of_producing_infinite_returns(self) -> None:
        df = pd.DataFrame({"event_ts_ns": np.arange(3) * DAY, "close": [100.0, 0.0, 100.0]})
        report = detect_outliers(df, symbol="X", granularity="1d", max_abs_log_return=0.5)
        assert np.isfinite(report.max_abs_log_return)

    def test_the_policy_names_the_first_jumps_when_they_are_not_allowed(self) -> None:
        report = OutlierReport("X", "1d", 10, 2, 0, 0, 0.7, worst_jumps=(("1970-01-03", 0.7),))
        with pytest.raises(DataContractError, match="2 prijssprong"):
            enforce_outlier_policy(report, allow_price_jumps=False)
        assert enforce_outlier_policy(report, allow_price_jumps=True) is None


# --------------------------------------------------------------------------- #
# continuity
# --------------------------------------------------------------------------- #
class TestContinuityEdges:
    def test_a_frame_without_close_is_refused(self) -> None:
        with pytest.raises(DataContractError, match="close-kolom"):
            validate_continuity(pd.DataFrame({"event_ts_ns": [0]}), symbol="X", max_abs_log_return=0.5)

    def test_a_single_bar_has_no_candidates(self) -> None:
        assert validate_continuity(_frame(1), symbol="X", max_abs_log_return=0.5) == []

    def test_an_active_symbol_has_no_upper_bound(self) -> None:
        life = SymbolLifecycle("X", listed_ts_ns=0)
        assert life.is_active and validate_continuity(_frame(), symbol="X", lifecycle=life, max_abs_log_return=0.5) == []
        assert not SymbolLifecycle("X", 0, delisted_ts_ns=DAY, renamed_to="Y").is_active

    def test_candidates_can_be_logged_to_a_ledger_and_read_back_per_symbol(self, tmp_path) -> None:
        df = _frame()
        df.loc[5:, ["open", "high", "low", "close"]] *= 3.0
        ledger = AdjustmentLedger(tmp_path / "led" / "adj.jsonl")
        found = validate_continuity(df, symbol="X", max_abs_log_return=0.5, ledger=ledger)
        assert len(found) == 1 and found[0].reason == "roll" and found[0].effective_ts_ns == 5 * DAY
        assert "NIET automatisch toegepast" in found[0].note
        ledger.append([AdjustmentFactor("Y", 2 * DAY, 2.0, "split"), AdjustmentFactor("X", DAY, 0.5, "rename", "n")])
        assert [f.effective_ts_ns for f in ledger.for_symbol("X")] == [DAY, 5 * DAY]
        assert ledger.for_symbol("Y")[0].effective_iso.startswith("1970-01-03")

    def test_an_absent_ledger_reads_as_empty_and_empty_appends_are_noops(self, tmp_path) -> None:
        ledger = AdjustmentLedger(tmp_path / "none.jsonl")
        ledger.append([])
        assert ledger.read_all() == [] and not ledger.path.exists()

    def test_blank_ledger_lines_are_skipped(self, tmp_path) -> None:
        ledger = AdjustmentLedger(tmp_path / "adj.jsonl")
        ledger.append([AdjustmentFactor("X", DAY, 2.0, "split")])
        with open(ledger.path, "a", encoding="utf-8") as fh:
            fh.write("\n\n")
        assert len(ledger.read_all()) == 1

    def test_adjustments_need_event_times(self) -> None:
        with pytest.raises(DataContractError, match="event_ts_ns"):
            apply_adjustments(pd.DataFrame({"close": [1.0]}), [AdjustmentFactor("X", DAY, 2.0, "split")])

    def test_no_factors_returns_an_untouched_copy(self) -> None:
        raw = _frame()
        out = apply_adjustments(raw, [])
        pd.testing.assert_frame_equal(out, raw)
        assert out is not raw

    def test_missing_price_columns_are_skipped_and_factors_chain_oldest_first(self) -> None:
        raw = _frame(6)[["event_ts_ns", "close", "volume"]]
        out = apply_adjustments(raw, [AdjustmentFactor("X", 4 * DAY, 2.0, "split"),
                                      AdjustmentFactor("X", 2 * DAY, 10.0, "roll")])
        assert out["close"].tolist() == [100.0 * 20, 101.0 * 20, 102.0 * 2, 103.0 * 2, 104.0, 105.0]
        assert out["volume"].tolist() == raw["volume"].tolist()
