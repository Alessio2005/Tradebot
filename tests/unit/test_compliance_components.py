# tests/unit/test_compliance_components.py
"""De randgevallen van de compliance-bouwstenen.

`tests/unit/test_wave9.py` toetst het hoofdpad van elk onderdeel. Hier staan
de paden die bij een audit het eerst worden gelezen: een te korte reeks in de
DM-toets, een log met een kapotte regel, een productierapport zonder
goedkeurder, een short positie, een schaduwhandelaar zonder records.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tradebot.compliance.champion_challenger import ChampionChallenger
from tradebot.compliance.circuit_log import CircuitLog
from tradebot.compliance.mrm_report import MRMReport, require_signed_approval
from tradebot.compliance.position_report import generate_position_report
from tradebot.compliance.shadow_trader import ShadowTrader
from tradebot.oms.position_tracker import PositionRecord


class TestChampionChallengerHelpers:
    def test_the_dm_test_needs_four_observations(self) -> None:
        stat, pvalue = ChampionChallenger._dm_test(np.array([0.1, -0.2, 0.3]))
        assert np.isnan(stat)
        assert pvalue == 1.0

    def test_a_single_bar_has_no_sharpe(self) -> None:
        assert ChampionChallenger._sharpe(pd.Series([0.01])) == 0.0

    def test_a_flat_pnl_has_no_sharpe(self) -> None:
        """Een spreiding van nul is geen oneindige Sharpe maar geen meting."""
        assert ChampionChallenger._sharpe(pd.Series([0.002] * 50)) == 0.0


class TestCircuitLog:
    def test_a_log_that_was_never_written_is_empty(self, tmp_path: Path) -> None:
        log = CircuitLog(tmp_path / "logs" / "cb.jsonl")
        assert log.read_all() == []
        assert log.as_dataframe().empty

    def test_blank_and_malformed_lines_are_skipped(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        log = CircuitLog(tmp_path / "cb.jsonl")
        log.record("drawdown", equity=90.0, equity_peak=100.0,
                   current_drawdown=0.1, daily_pnl_pct=-0.05,
                   open_positions={"BTC": 1.0})
        with open(tmp_path / "cb.jsonl", "a", encoding="utf-8") as fh:
            fh.write("\n{not json\n")
            fh.write('{"halt_reason": "only one field"}\n')
        with caplog.at_level(logging.WARNING):
            entries = log.read_all()
        assert [e.halt_reason for e in entries] == ["drawdown"]
        assert caplog.text.count("skipped malformed line") == 2


class TestMrmApproval:
    def test_a_production_report_without_an_approver_is_refused(self) -> None:
        with pytest.raises(ValueError, match="FORBIDDEN"):
            MRMReport(model_id="m", git_sha="g", feature_hash="f",
                      dvc_hash="d", _production=True)

    def test_a_production_report_with_an_approver_is_accepted(self) -> None:
        report = MRMReport(model_id="m", git_sha="g", feature_hash="f",
                           dvc_hash="d", approved_by="alice", _production=True)
        assert report.approved_by == "alice"

    @pytest.mark.parametrize(("approver", "secret"), [("", "s"), ("alice", "")])
    def test_signing_needs_both_an_approver_and_a_secret(
        self, approver: str, secret: str,
    ) -> None:
        with pytest.raises(ValueError, match="signed approval"):
            require_signed_approval({"model_id": "m"}, approver, secret)

    def test_the_signature_depends_on_the_secret(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr("tradebot.compliance.mrm_report.time.time",
                            lambda: 1_700_000_000.0)
        data = {"model_id": "m", "sharpe": 1.2}
        signed = require_signed_approval(data, "alice", "s1")
        again = require_signed_approval(data, "alice", "s1")
        other = require_signed_approval(data, "alice", "s2")
        assert signed["approved_by"] == "alice"
        assert signed["approval_ts"] == 1_700_000_000
        assert signed["approval_signature"] == again["approval_signature"]
        assert signed["approval_signature"] != other["approval_signature"]
        assert "approved_by" not in data


class TestPositionReport:
    def test_flat_positions_are_dropped_and_shorts_count_short(self) -> None:
        positions = {
            "BTC": PositionRecord(symbol="BTC", qty=0.5, avg_entry_price=40_000.0,
                                  last_mark_price=40_000.0),
            "ETH": PositionRecord(symbol="ETH", qty=-10.0, avg_entry_price=2_000.0,
                                  last_mark_price=2_000.0),
            "SOL": PositionRecord(symbol="SOL", qty=0.0, avg_entry_price=100.0,
                                  last_mark_price=100.0),
        }
        report = generate_position_report(
            equity=100_000.0, positions=positions, daily_pnl=0.0,
            report_date="2024-01-01")
        assert [p.symbol for p in report.positions] == ["BTC", "ETH"]
        assert report.gross_leverage == pytest.approx(0.4)
        assert report.net_imbalance_pct == pytest.approx(0.0)
        frame = report.to_dataframe()
        assert list(frame["symbol"]) == ["BTC", "ETH"]


class TestShadowTrader:
    def test_a_quoted_spread_is_added_to_the_slippage(self, tmp_path: Path) -> None:
        trader = ShadowTrader("v1", "v2", output_root=tmp_path)
        record = trader.observe(
            pd.Timestamp("2024-01-01", tz="UTC"), "BTC", champion_signal=0.0,
            challenger_signal=1.0, close_price=100.0, champion_pnl=0.0,
            next_open_price=101.0, spread_bps=3.0, slippage_bps=2.0)
        assert record.hyp_fill_price == pytest.approx(101.0 * (1.0 + 5e-4))

    def test_saving_without_records_writes_nothing(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        trader = ShadowTrader("v1", "v2", output_root=tmp_path / "shadow")
        with caplog.at_level(logging.WARNING):
            assert trader.save("BTC") == Path()
        assert not (tmp_path / "shadow").exists()
        assert "no records" in caplog.text
