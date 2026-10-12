# tests/unit/test_tca_components.py
"""De TCA-bouwstenen los: arrival price, IS-decompositie, post-trade, rapport.

De roundtrip-closure zelf staat in `tests/integration/test_tca_roundtrip.py`;
hier staan de randen van de losse functies -- een beslismoment tussen twee
bars, een terugkijkvenster, een VWAP met onbekende herkomst, een leeg rapport.
"""
from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tradebot.tca import (
    TCAReport,
    analyse_execution,
    compute_arrival_price_benchmark,
    decompose_implementation_shortfall,
)
from tradebot.tca.post_trade import CostDecomposition, TcaClosure, _marks_at
from tradebot.utils.failfast import DataContractError

IDX = pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC")
PRICES = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0], index=IDX)


class TestArrivalPrice:
    def test_a_decision_between_bars_takes_the_last_known_price(self) -> None:
        between = IDX[2] + pd.Timedelta(minutes=30)
        bench = compute_arrival_price_benchmark(
            "BTC", between, execution_price=103.0, signed_qty=1.0,
            price_series=PRICES)
        assert bench.arrival_price == 102.0
        assert bench.implementation_shortfall == pytest.approx(-1.0)

    def test_a_decision_before_the_first_bar_falls_back_to_execution(
        self,
    ) -> None:
        early = IDX[0] - pd.Timedelta(hours=1)
        bench = compute_arrival_price_benchmark(
            "BTC", early, execution_price=99.0, signed_qty=2.0,
            price_series=PRICES)
        assert bench.arrival_price == 99.0
        assert bench.implementation_shortfall == 0.0

    def test_the_offset_looks_back_a_number_of_bars(self) -> None:
        bench = compute_arrival_price_benchmark(
            "BTC", IDX[3], execution_price=104.0, signed_qty=1.0,
            price_series=PRICES, arrival_offset=2)
        assert bench.arrival_price == 101.0

    def test_the_offset_is_clipped_at_the_first_bar(self) -> None:
        bench = compute_arrival_price_benchmark(
            "BTC", IDX[1], execution_price=104.0, signed_qty=1.0,
            price_series=PRICES, arrival_offset=10)
        assert bench.arrival_price == 100.0

    def test_a_non_positive_arrival_falls_back_to_execution(self) -> None:
        broken = PRICES.copy()
        broken.iloc[2] = 0.0
        bench = compute_arrival_price_benchmark(
            "BTC", IDX[2], execution_price=102.5, signed_qty=1.0,
            price_series=broken)
        assert bench.arrival_price == 102.5


class TestShortfallTimingComponent:
    @pytest.mark.parametrize("source", ["mid_vwap", "unknown"])
    def test_a_vwap_without_trade_prints_is_flagged_biased(
        self, source: str, caplog: pytest.LogCaptureFixture,
    ) -> None:
        with caplog.at_level(logging.WARNING):
            decomp = decompose_implementation_shortfall(
                "BTC", decision_price=100.0, arrival_price=101.0,
                execution_price=102.0, signed_qty=1.0, vwap_price=101.5,
                vwap_source=source)  # type: ignore[arg-type]
        assert decomp.timing_is_biased is True
        assert decomp.timing_component == pytest.approx(0.5 - (-1.0))
        assert source in caplog.text

    def test_a_trade_print_vwap_is_not_biased(self) -> None:
        decomp = decompose_implementation_shortfall(
            "BTC", decision_price=100.0, arrival_price=101.0,
            execution_price=102.0, signed_qty=1.0, vwap_price=101.5,
            vwap_source="trade_print")
        assert decomp.timing_is_biased is False
        assert decomp.total_is == pytest.approx(
            decomp.delay_component + decomp.market_impact
            + decomp.timing_component)


def _record(symbol: str, signal: float = 0.0):
    return analyse_execution(
        symbol, IDX[1], signed_qty=2.0, execution_price=101.2,
        decision_price=100.0, arrival_price=101.0, volatility=0.02,
        adv=1e6, pre_trade_signal=signal)


class TestPostTrade:
    def test_alpha_capture_is_the_signal_left_after_costs(self) -> None:
        record = _record("BTC", signal=0.01)
        notional = 2.0 * 101.2
        expected = 1.0 - (record.is_decomp.total_is / notional) / 0.01
        assert record.alpha_capture == pytest.approx(
            float(np.clip(expected, 0.0, 2.0)), rel=1e-6)

    def test_without_a_signal_all_alpha_counts_as_captured(self) -> None:
        assert _record("BTC").alpha_capture == 1.0

    def test_the_closure_record_carries_the_decomposition(self) -> None:
        closure = TcaClosure(
            arrival_equity=1000.0, realised_equity=998.0, gap=2.0,
            decomposition=CostDecomposition(
                spread=0.5, impact=1.0, fees=0.5, funding=0.25, timing=3.0),
            residual=0.0, tolerance=0.01, closes=True)
        record = closure.as_record()
        assert record["gap"] == 2.0
        assert record["closes"] is True
        assert record["execution_total"] == 2.0
        assert record["total_with_funding"] == 2.25
        assert record["timing_opportunity_cost"] == 3.0

    def test_marks_for_an_unknown_timestamp_are_refused(self) -> None:
        result = SimpleNamespace(
            snapshots=[SimpleNamespace(ts=IDX[0])],
            audit={"marks": [{"BTC": 100.0}]})
        assert _marks_at(result, IDX[0]) == {"BTC": 100.0}  # type: ignore[arg-type]
        with pytest.raises(DataContractError, match="Geen markprijzen"):
            _marks_at(result, IDX[1])  # type: ignore[arg-type]


class TestReport:
    def test_an_empty_report_has_an_empty_summary_and_breakdown(self) -> None:
        report = TCAReport([])
        assert report.summary().empty
        assert report.by_symbol().empty

    def test_the_breakdown_groups_per_symbol_and_caches_the_frame(self) -> None:
        report = TCAReport([_record("BTC"), _record("ETH"), _record("BTC")])
        frame = report.to_dataframe()
        assert report.to_dataframe() is frame
        breakdown = report.by_symbol()
        assert breakdown.loc["BTC", "n_trades"] == 2
        assert breakdown.loc["ETH", "n_trades"] == 1
        assert report.summary()["n_trades"] == 3
