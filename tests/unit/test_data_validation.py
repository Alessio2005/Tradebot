"""Phase 1, deliverable 7 - contracttests voor de L0 validatielaag.

Kernregel die hier wordt bewezen: **elke validator raiset bij schending en
retourneert geen boolean.**
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradebot.data.validation import (
    OHLCV_SPEC,
    AdjustmentFactor,
    AdjustmentLedger,
    GapLedger,
    SymbolLifecycle,
    apply_adjustments,
    detect_gaps,
    detect_outliers,
    enforce_gap_policy,
    enforce_outlier_policy,
    validate_continuity,
    validate_schema,
)
from tradebot.utils.failfast import DataContractError

DAY = 86_400_000_000_000


def frame(n: int = 30, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ev = np.arange(n, dtype=np.int64) * DAY
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return pd.DataFrame({
        "event_ts_ns": ev,
        "asof_ts_ns": ev + DAY,
        "open": c * 0.999, "high": c * 1.01, "low": c * 0.99, "close": c,
        "volume": rng.uniform(1, 10, n),
    })


class TestSchema:
    def test_clean_frame_passes(self) -> None:
        assert validate_schema(frame(), OHLCV_SPEC) is None

    def test_returns_none_never_a_boolean(self) -> None:
        """Een aanroeper mag de uitkomst niet als vlag kunnen gebruiken."""
        assert validate_schema(frame(), OHLCV_SPEC) is None

    def test_empty_frame_rejected(self) -> None:
        with pytest.raises(DataContractError, match="Lege dataset"):
            validate_schema(frame().iloc[0:0], OHLCV_SPEC)

    def test_missing_column_rejected(self) -> None:
        with pytest.raises(DataContractError, match="Verplichte kolom"):
            validate_schema(frame().drop(columns=["volume"]), OHLCV_SPEC)

    def test_non_int64_timestamp_rejected(self) -> None:
        df = frame()
        df["event_ts_ns"] = pd.to_datetime(df["event_ts_ns"], unit="ns", utc=True)
        with pytest.raises(DataContractError, match="int64"):
            validate_schema(df, OHLCV_SPEC)

    def test_duplicate_timestamps_rejected(self) -> None:
        df = frame()
        df.loc[5, "event_ts_ns"] = int(df.loc[4, "event_ts_ns"])
        with pytest.raises(DataContractError, match="duplicaten"):
            validate_schema(df, OHLCV_SPEC)

    def test_nan_rejected_not_interpolated(self) -> None:
        df = frame()
        df.loc[7, "close"] = np.nan
        with pytest.raises(DataContractError, match="NaN"):
            validate_schema(df, OHLCV_SPEC)

    def test_non_positive_price_rejected(self) -> None:
        df = frame()
        df.loc[3, "close"] = 0.0
        with pytest.raises(DataContractError, match="niet-positieve"):
            validate_schema(df, OHLCV_SPEC)

    def test_negative_volume_rejected(self) -> None:
        df = frame()
        df.loc[3, "volume"] = -1.0
        with pytest.raises(DataContractError, match="negatieve"):
            validate_schema(df, OHLCV_SPEC)

    def test_ohlc_ordering_enforced(self) -> None:
        """high < max(open, close) breekt elke range-schatter."""
        df = frame()
        df.loc[3, "high"] = df.loc[3, "low"] * 0.5
        with pytest.raises(DataContractError, match="OHLC-ordening"):
            validate_schema(df, OHLCV_SPEC)

    def test_asof_before_event_rejected(self) -> None:
        df = frame()
        df.loc[4, "asof_ts_ns"] = int(df.loc[4, "event_ts_ns"]) - 1
        with pytest.raises(DataContractError, match="vóór event_ts_ns"):
            validate_schema(df, OHLCV_SPEC)


class TestGaps:
    def test_no_gaps_on_a_complete_series(self) -> None:
        assert detect_gaps(frame(), asset_class="crypto", symbol="B",
                           granularity="1d") == []

    def test_gap_is_measured_exactly(self) -> None:
        g = frame().drop(index=[5, 6, 7]).reset_index(drop=True)
        recs = detect_gaps(g, asset_class="crypto", symbol="B", granularity="1d")
        assert len(recs) == 1
        assert recs[0].n_missing == 3

    def test_reject_policy_raises(self, tmp_path) -> None:
        g = frame().drop(index=[5]).reset_index(drop=True)
        recs = detect_gaps(g, asset_class="crypto", symbol="B", granularity="1d")
        with pytest.raises(DataContractError, match="ontbrekende bar"):
            enforce_gap_policy(
                recs, policy="reject", ledger=GapLedger(tmp_path / "gaps.jsonl"))

    def test_gap_is_written_to_the_ledger_even_when_rejected(self, tmp_path) -> None:
        """Er bestaat geen pad waarin een gat verdwijnt zonder spoor."""
        led = GapLedger(tmp_path / "gaps.jsonl")
        g = frame().drop(index=[5]).reset_index(drop=True)
        recs = detect_gaps(g, asset_class="crypto", symbol="B", granularity="1d")
        with pytest.raises(DataContractError):
            enforce_gap_policy(recs, policy="reject", ledger=led)
        assert len(led.read_all()) == 1

    def test_register_policy_records_and_continues(self, tmp_path) -> None:
        led = GapLedger(tmp_path / "gaps.jsonl")
        g = frame().drop(index=[5]).reset_index(drop=True)
        recs = detect_gaps(g, asset_class="crypto", symbol="B", granularity="1d")
        enforce_gap_policy(recs, policy="register", ledger=led)
        assert len(led.read_all()) == 1

    def test_unknown_policy_rejected(self, tmp_path) -> None:
        with pytest.raises(DataContractError, match="gap_policy"):
            enforce_gap_policy([], policy="interpolate",
                               ledger=GapLedger(tmp_path / "g.jsonl"))

    def test_unknown_granularity_rejected(self) -> None:
        with pytest.raises(DataContractError, match="granulariteit"):
            detect_gaps(frame(), asset_class="crypto", symbol="B",
                        granularity="7s")

    def test_ledger_summary_aggregates(self, tmp_path) -> None:
        led = GapLedger(tmp_path / "gaps.jsonl")
        g = frame().drop(index=[5, 6, 15]).reset_index(drop=True)
        led.append(detect_gaps(g, asset_class="crypto", symbol="B",
                               granularity="1d"))
        s = led.summary()
        assert int(s["n_missing_bars"].iloc[0]) == 3


class TestOutliers:
    def test_clean_frame_has_no_outliers(self) -> None:
        r = detect_outliers(frame(), symbol="B", granularity="1d",
                            max_abs_log_return=0.30)
        assert r.total == 0

    def test_price_jump_detected(self) -> None:
        df = frame()
        df.loc[10, "close"] = df.loc[10, "close"] * 2.0
        r = detect_outliers(df, symbol="B", granularity="1d",
                            max_abs_log_return=0.30)
        assert r.n_price_jumps >= 1
        assert r.worst_jumps

    def test_zero_volume_is_always_fatal(self) -> None:
        df = frame()
        df.loc[4, "volume"] = 0.0
        r = detect_outliers(df, symbol="B", granularity="1d",
                            max_abs_log_return=0.30)
        with pytest.raises(DataContractError, match="volume 0"):
            enforce_outlier_policy(r, allow_price_jumps=True)

    def test_inverted_range_is_always_fatal(self) -> None:
        df = frame()
        df.loc[4, "high"] = df.loc[4, "low"] - 1.0
        r = detect_outliers(df, symbol="B", granularity="1d",
                            max_abs_log_return=0.30)
        with pytest.raises(DataContractError, match="high < low"):
            enforce_outlier_policy(r, allow_price_jumps=True)

    def test_price_jumps_are_configurable(self) -> None:
        """Op crypto is een dagelijkse 30%-beweging een echte gebeurtenis."""
        df = frame()
        df.loc[10, "close"] = df.loc[10, "close"] * 2.0
        r = detect_outliers(df, symbol="B", granularity="1d",
                            max_abs_log_return=0.30)
        with pytest.raises(DataContractError, match="prijssprong"):
            enforce_outlier_policy(r, allow_price_jumps=False)
        assert enforce_outlier_policy(r, allow_price_jumps=True) is None

    def test_threshold_has_no_default(self) -> None:
        """max_abs_log_return moet expliciet worden meegegeven."""
        import inspect

        p = inspect.signature(detect_outliers).parameters["max_abs_log_return"]
        assert p.default is inspect.Parameter.empty


class TestContinuity:
    def test_clean_series_has_no_roll_candidates(self) -> None:
        assert validate_continuity(frame(), symbol="B",
                                   max_abs_log_return=0.30) == []

    def test_jump_becomes_a_candidate_not_an_applied_correction(self) -> None:
        """Kandidaten worden NIET automatisch toegepast."""
        df = frame()
        df.loc[10, "close"] = df.loc[10, "close"] * 2.0
        cands = validate_continuity(df, symbol="B", max_abs_log_return=0.30)
        assert cands
        assert all("NIET automatisch toegepast" in c.note for c in cands)

    def test_bars_before_listing_rejected(self) -> None:
        lc = SymbolLifecycle(symbol="B", listed_ts_ns=int(5 * DAY))
        with pytest.raises(DataContractError, match="vóór de listing"):
            validate_continuity(frame(), symbol="B", lifecycle=lc,
                                max_abs_log_return=0.30)

    def test_bars_after_delisting_rejected(self) -> None:
        """Survivorship: bars na een delisting bestaan niet."""
        lc = SymbolLifecycle(symbol="B", listed_ts_ns=0,
                             delisted_ts_ns=int(10 * DAY))
        with pytest.raises(DataContractError, match="ná de delisting"):
            validate_continuity(frame(), symbol="B", lifecycle=lc,
                                max_abs_log_return=0.30)

    def test_adjustment_applies_only_before_effective_date(self) -> None:
        df = frame()
        adj = apply_adjustments(
            df, [AdjustmentFactor("B", int(15 * DAY), 0.5, "split")])
        assert adj.loc[0, "close"] == pytest.approx(df.loc[0, "close"] * 0.5)
        assert adj.loc[20, "close"] == pytest.approx(df.loc[20, "close"])

    def test_adjustment_does_not_mutate_the_raw_series(self) -> None:
        """De ruwe reeks blijft intact; dat maakt de correctie inspecteerbaar."""
        df = frame()
        before = df["close"].copy()
        apply_adjustments(df, [AdjustmentFactor("B", int(15 * DAY), 0.5, "split")])
        pd.testing.assert_series_equal(before, df["close"])

    def test_non_positive_factor_rejected(self) -> None:
        with pytest.raises(DataContractError, match="strikt positief"):
            apply_adjustments(frame(),
                              [AdjustmentFactor("B", int(5 * DAY), 0.0, "roll")])

    def test_ledger_roundtrip(self, tmp_path) -> None:
        led = AdjustmentLedger(tmp_path / "adj.jsonl")
        led.append([AdjustmentFactor("B", int(5 * DAY), 0.5, "split", "note")])
        back = led.for_symbol("B")
        assert len(back) == 1
        assert back[0].factor == 0.5
        assert back[0].reason == "split"
