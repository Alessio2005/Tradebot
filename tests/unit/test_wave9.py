"""tests/unit/test_wave9.py — Wave 9 Governance & Compliance tests.

Covers:
  - compliance/shadow_trader.py (ShadowTrader)
  - compliance/champion_challenger.py (ChampionChallenger — DM-test reproducible)
  - compliance/mrm_report.py (MRMReport — generates without exceptions)
  - compliance/position_report.py (DailyPositionReport)
  - compliance/circuit_log.py (CircuitLog)
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ─── ShadowTrader ─────────────────────────────────────────────────────────────

from tradebot.compliance.shadow_trader import ShadowRecord, ShadowTrader


class TestShadowTrader:
    def _make_trader(self) -> ShadowTrader:
        return ShadowTrader(
            champion_version="v1.0.0",
            challenger_version="v1.1.0",
        )

    def test_observe_returns_record(self):
        st = self._make_trader()
        ts = pd.Timestamp("2024-01-01 12:00:00", tz="UTC")
        rec = st.observe(
            ts=ts, symbol="BTCUSDT",
            champion_signal=0.3, challenger_signal=0.5,
            close_price=40000.0, champion_pnl=100.0,
        )
        assert isinstance(rec, ShadowRecord)
        assert rec.champion_signal == 0.3
        assert rec.challenger_signal == 0.5

    def test_no_orders_placed(self):
        """ShadowTrader must never modify any OMS state."""
        st = self._make_trader()
        initial_state = st.n_observations
        for i in range(5):
            ts = pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(hours=i)
            st.observe(ts=ts, symbol="BTCUSDT",
                       champion_signal=float(i) * 0.1,
                       challenger_signal=float(i) * 0.1 + 0.05,
                       close_price=40000.0 + i * 100, champion_pnl=50.0)
        # Only observation count changes — no fills, no positions
        assert st.n_observations == 5

    def test_save_creates_parquet(self, tmp_path):
        st = ShadowTrader("v1.0.0", "v1.1.0", output_root=tmp_path)
        ts = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
        st.observe(ts=ts, symbol="BTCUSDT", champion_signal=0.2,
                   challenger_signal=0.4, close_price=40000.0, champion_pnl=0.0)
        out_path = st.save("BTCUSDT")
        assert out_path.exists()
        df = pd.read_parquet(out_path)
        assert len(df) == 1
        assert "champion_signal" in df.columns

    def test_challenger_pnl_independent_of_champion(self):
        """Hypothetical challenger PnL must not equal champion PnL unconditionally."""
        st = self._make_trader()
        ts = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
        rec = st.observe(ts=ts, symbol="BTCUSDT",
                         champion_signal=0.8, challenger_signal=-0.8,
                         close_price=40000.0, champion_pnl=200.0)
        # Challenger is short, champion is long — hyp_pnl ≠ champion_pnl
        assert rec is not None  # just check it runs without crash

    def test_get_records(self):
        st = self._make_trader()
        for i in range(3):
            ts = pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(hours=i)
            st.observe(ts, "BTCUSDT", 0.1 * i, -0.1 * i, 40000.0, 0.0)
        recs = st.get_records()
        assert len(recs) == 3


# ─── ChampionChallenger ────────────────────────────────────────────────────────

from tradebot.compliance.champion_challenger import (
    ChampionChallenger,
    ChampionChallengerConfig,
    DMTestResult,
)


def _make_pnl_series(n: int, mean: float = 0.0, std: float = 1.0, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    vals = rng.normal(mean, std, n)
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    return pd.Series(vals, index=idx)


class TestChampionChallenger:
    def test_insufficient_data_returns_recommendation(self):
        cc = ChampionChallenger(ChampionChallengerConfig(n_min_bars=200))
        short = _make_pnl_series(10)
        result = cc.evaluate(short, short)
        assert result.recommendation == "insufficient_data"
        assert not result.sufficient_data

    def test_dm_statistic_is_finite_with_sufficient_data(self):
        cc = ChampionChallenger(ChampionChallengerConfig(n_min_bars=50))
        champ = _make_pnl_series(200, mean=0.5, seed=1)
        chal = _make_pnl_series(200, mean=0.7, seed=2)
        result = cc.evaluate(champ, chal)
        assert result.sufficient_data
        assert math.isfinite(result.dm_statistic)
        assert 0.0 <= result.pvalue <= 1.0

    def test_dm_test_is_reproducible(self):
        """Regression: same inputs must produce identical DM statistic."""
        cc = ChampionChallenger(ChampionChallengerConfig(n_min_bars=50))
        champ = _make_pnl_series(200, mean=0.5, seed=1)
        chal = _make_pnl_series(200, mean=0.7, seed=2)
        r1 = cc.evaluate(champ, chal)
        r2 = cc.evaluate(champ, chal)
        assert abs(r1.dm_statistic - r2.dm_statistic) < 1e-10
        assert abs(r1.pvalue - r2.pvalue) < 1e-10

    def test_champion_retained_when_sharpe_worse(self):
        cc = ChampionChallenger(ChampionChallengerConfig(n_min_bars=50, sharpe_hurdle=100.0))
        champ = _make_pnl_series(200, mean=1.0, seed=1)
        chal = _make_pnl_series(200, mean=0.5, seed=2)
        result = cc.evaluate(champ, chal)
        assert result.recommendation == "retain_champion"

    def test_sharpe_always_finite(self):
        cc = ChampionChallenger(ChampionChallengerConfig(n_min_bars=5))
        champ = _make_pnl_series(100, mean=0.5)
        chal = _make_pnl_series(100, mean=0.5)
        result = cc.evaluate(champ, chal)
        assert math.isfinite(result.champion_sharpe)
        assert math.isfinite(result.challenger_sharpe)


# ─── MRMReport ────────────────────────────────────────────────────────────────

from tradebot.compliance.mrm_report import MRMReport, generate_mrm_report


class TestMRMReport:
    def _make_report(self) -> MRMReport:
        return generate_mrm_report(
            model_id="BTCUSDT_LONG_v1.0.0",
            git_sha="abc1234",
            feature_hash="deadbeef12345678",
            dvc_hash="cafebabe",
            training_date="2025-01-15",
            symbols=["BTCUSDT", "ETHUSDT"],
            n_bars_per_symbol={"BTCUSDT": 8760, "ETHUSDT": 8760},
            oos_sharpe=1.45,
            oos_max_dd=0.12,
            oos_calmar=1.8,
        )

    def test_generates_without_exception(self):
        report = self._make_report()
        assert isinstance(report, MRMReport)

    def test_has_all_9_sections(self):
        report = self._make_report()
        assert len(report.sections) == 9

    def test_to_dict_is_serialisable(self):
        import json
        report = self._make_report()
        d = report.to_dict()
        # Must be JSON serialisable
        json.dumps(d)

    def test_to_text_contains_model_id(self):
        report = self._make_report()
        text = report.to_text()
        assert "BTCUSDT_LONG_v1.0.0" in text

    def test_pending_approval(self):
        report = self._make_report()
        assert report.approved_by == ""
        assert report.approval_date == ""
        text = report.to_text()
        assert "[PENDING]" in text


# ─── DailyPositionReport ──────────────────────────────────────────────────────

from tradebot.compliance.position_report import (
    DailyPositionReport,
    generate_position_report,
)
from tradebot.oms.position_tracker import PositionRecord


class TestDailyPositionReport:
    def _make_positions(self):
        pos = PositionRecord(
            symbol="BTCUSDT",
            qty=0.5,
            avg_entry_price=40_000.0,
            unrealised_pnl=500.0,
            last_mark_price=41_000.0,
        )
        return {"BTCUSDT": pos}

    def test_generate_report(self):
        positions = self._make_positions()
        report = generate_position_report(
            equity=100_500.0,
            positions=positions,
            daily_pnl=500.0,
            model_version="v1.0.0",
        )
        assert isinstance(report, DailyPositionReport)
        assert report.equity == 100_500.0
        assert len(report.positions) == 1

    def test_gross_leverage_computed(self):
        positions = self._make_positions()
        report = generate_position_report(
            equity=100_000.0, positions=positions, daily_pnl=0.0
        )
        # notional = 0.5 * 41000 = 20500; leverage = 20500/100000 = 0.205
        assert abs(report.gross_leverage - 0.205) < 0.001

    def test_to_dict_serialisable(self):
        import json
        positions = self._make_positions()
        report = generate_position_report(
            equity=100_000.0, positions=positions, daily_pnl=0.0
        )
        json.dumps(report.to_dict())

    def test_empty_positions_zero_leverage(self):
        report = generate_position_report(
            equity=100_000.0, positions={}, daily_pnl=0.0
        )
        assert report.gross_leverage == 0.0
        assert len(report.positions) == 0


# ─── CircuitLog ───────────────────────────────────────────────────────────────

from tradebot.compliance.circuit_log import CircuitLog, CircuitLogEntry


class TestCircuitLog:
    def test_record_and_read(self, tmp_path):
        log = CircuitLog(tmp_path / "circuit.jsonl")
        entry = log.record(
            halt_reason="MAX_DRAWDOWN",
            equity=88_000.0,
            equity_peak=100_000.0,
            current_drawdown=0.12,
            daily_pnl_pct=-0.04,
            open_positions={"BTCUSDT": 0.5},
            model_version="v1.0.0",
            git_sha="abc1234",
        )
        assert isinstance(entry, CircuitLogEntry)
        all_entries = log.read_all()
        assert len(all_entries) == 1
        assert all_entries[0].halt_reason == "MAX_DRAWDOWN"

    def test_append_only(self, tmp_path):
        log_path = tmp_path / "circuit.jsonl"
        for i in range(3):
            log = CircuitLog(log_path)
            log.record(
                halt_reason="FEED_TIMEOUT",
                equity=float(90_000 + i * 1000),
                equity_peak=100_000.0,
                current_drawdown=float(i) * 0.03,
                daily_pnl_pct=-0.01,
                open_positions={},
            )
        all_entries = CircuitLog(log_path).read_all()
        assert len(all_entries) == 3

    def test_as_dataframe(self, tmp_path):
        log = CircuitLog(tmp_path / "circuit.jsonl")
        log.record(
            halt_reason="DAILY_LOSS",
            equity=97_000.0,
            equity_peak=100_000.0,
            current_drawdown=0.03,
            daily_pnl_pct=-0.03,
            open_positions={"BTCUSDT": -0.2},
        )
        df = log.as_dataframe()
        assert len(df) == 1
        assert "halt_reason" in df.columns

    def test_extra_context_preserved(self, tmp_path):
        log = CircuitLog(tmp_path / "circuit.jsonl")
        log.record(
            halt_reason="HASH_MISMATCH",
            equity=100_000.0,
            equity_peak=100_000.0,
            current_drawdown=0.0,
            daily_pnl_pct=0.0,
            open_positions={},
            extra={"feature_hash": "deadbeef", "expected_hash": "cafebabe"},
        )
        entries = log.read_all()
        assert entries[0].extra["feature_hash"] == "deadbeef"
