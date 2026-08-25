"""De vectorized engine telt nooit als promotiebewijs — fase-opdracht §15, §17.

Twee eisen, en de tweede is de moeilijke:

1. *"Iedere output moet expliciet `NOT_ADMISSIBLE_AS_PROMOTION_EVIDENCE`
   bevatten."*
2. *"De promotion gate moet technisch weigeren om vectorized resultaten als
   promotiebewijs te accepteren. Test deze weigering."*

Een markering die niemand controleert, is documentatie. Deze suite toetst dat
de gate crasht, en dat hij dat ook doet wanneer het resultaat als platte dict
uit een JSON-artefact komt - want dat is de vorm waarin een getal een week
later terugkomt.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradebot.backtest.vectorized import (
    EVIDENCE_KEY,
    NOT_ADMISSIBLE,
    VectorizedResult,
    is_vectorized_evidence,
    reject_vectorized_evidence,
    run_vectorized,
)
from tradebot.registry.catalog import ModelCatalog, ModelRecord
from tradebot.registry.promotion import promote
from tradebot.utils.failfast import DataContractError

SYMBOLS = ["BTCUSDT", "ETHUSDT"]


@pytest.fixture
def result() -> VectorizedResult:
    idx = pd.date_range("2024-01-01", periods=30, freq="D", tz="UTC")
    rng = np.random.default_rng(3)
    prices = pd.DataFrame(
        {s: 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.02, 30))) for s in SYMBOLS},
        index=idx)
    weights = pd.DataFrame(0.25, index=idx, columns=SYMBOLS)
    return run_vectorized(weights, prices, cost_per_side=0.00065,
                          initial_equity=100_000.0)


# =========================================================================== #
# 1. De markering zit op elke output
# =========================================================================== #
class TestTheMarkIsAlwaysThere:
    def test_the_result_carries_the_label(self, result: VectorizedResult) -> None:
        assert result.evidence_class == NOT_ADMISSIBLE
        assert result.evidence_class == "NOT_ADMISSIBLE_AS_PROMOTION_EVIDENCE"

    def test_the_record_carries_the_label(self, result: VectorizedResult) -> None:
        record = result.as_record()
        assert record[EVIDENCE_KEY] == NOT_ADMISSIBLE
        assert record["engine"] == "vectorized"
        assert record["counts_as_promotion_evidence"] is False

    def test_the_property_is_always_false(self, result: VectorizedResult) -> None:
        assert result.counts_as_promotion_evidence is False

    def test_the_label_cannot_be_overridden(self) -> None:
        """Er is geen constructie waarin het resultaat zichzelf vrijpleit."""
        idx = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
        series = pd.Series([1.0, 1.0, 1.0], index=idx)
        with pytest.raises(DataContractError):
            VectorizedResult(
                equity_curve=series, returns=series, gross_returns=series,
                turnover=series,
                evidence_class="ADMISSIBLE",  # type: ignore[arg-type]
            )

    def test_the_result_is_frozen(self, result: VectorizedResult) -> None:
        with pytest.raises((AttributeError, TypeError)):
            result.evidence_class = "ADMISSIBLE"  # type: ignore[misc]


# =========================================================================== #
# 2. De detectie werkt op elke vorm waarin een resultaat reist
# =========================================================================== #
class TestDetection:
    def test_the_dataclass_is_detected(self, result: VectorizedResult) -> None:
        assert is_vectorized_evidence(result)

    def test_the_flat_record_is_detected(self, result: VectorizedResult) -> None:
        assert is_vectorized_evidence(result.as_record())

    def test_a_json_roundtrip_is_still_detected(
        self, result: VectorizedResult
    ) -> None:
        """Dit is de vorm waarin een getal een week later terugkomt."""
        import json

        payload = json.loads(json.dumps(result.as_record(), default=str))
        assert is_vectorized_evidence(payload)

    @pytest.mark.parametrize("payload", [
        {EVIDENCE_KEY: NOT_ADMISSIBLE},
        {"engine": "vectorized"},
        {"counts_as_promotion_evidence": False},
    ])
    def test_every_marker_alone_is_enough(self, payload: dict) -> None:
        assert is_vectorized_evidence(payload)

    @pytest.mark.parametrize("payload", [
        {"engine": "event_driven", "sharpe": 1.2},
        {"sharpe": 0.8, "max_dd": 0.1},
        {},
    ])
    def test_event_driven_output_is_not_flagged(self, payload: dict) -> None:
        """Anders zou de gate ALLES weigeren en niets bewijzen."""
        assert not is_vectorized_evidence(payload)


# =========================================================================== #
# 3. De weigering is technisch, niet documentair
# =========================================================================== #
class TestTheGateRefuses:
    def test_reject_crashes_on_a_vectorized_result(
        self, result: VectorizedResult
    ) -> None:
        with pytest.raises(DataContractError):
            reject_vectorized_evidence(result)

    def test_reject_crashes_on_a_flat_record(
        self, result: VectorizedResult
    ) -> None:
        with pytest.raises(DataContractError):
            reject_vectorized_evidence(result.as_record())

    def test_reject_passes_event_driven_output(self) -> None:
        reject_vectorized_evidence({"engine": "event_driven", "sharpe": 1.0})

    def test_the_promotion_gate_refuses_vectorized_metrics(
        self, tmp_path
    ) -> None:
        """§17: 'vectorized promotion attempt -> FAIL'.

        Niet 'de gate logt een waarschuwing'. Hij crasht, en het model
        verandert niet van stage.
        """
        catalog = ModelCatalog(tmp_path / "catalog.json")
        catalog.register(ModelRecord(
            symbol="BTCUSDT", side="LONG", version=1, stage="research",
            artifact_path="m.joblib", git_sha="abc", dvc_hash="d",
            feature_hash="f",
            # Metrics die de gate RUIM zou halen - zodat de test niet per
            # ongeluk groen wordt omdat de drempels niet gehaald zijn.
            metrics={"oos_logloss": 0.40, "sharpe": 2.0, EVIDENCE_KEY: NOT_ADMISSIBLE},
            extra={},
        ))
        with pytest.raises(DataContractError):
            promote(catalog, "BTCUSDT", "LONG", "staging")

    def test_the_promotion_gate_refuses_a_vectorized_marker_in_extra(
        self, tmp_path
    ) -> None:
        catalog = ModelCatalog(tmp_path / "catalog.json")
        catalog.register(ModelRecord(
            symbol="ETHUSDT", side="LONG", version=1, stage="research",
            artifact_path="m.joblib", git_sha="abc", dvc_hash="d",
            feature_hash="f",
            metrics={"oos_logloss": 0.40, "sharpe": 2.0},
            extra={"engine": "vectorized"},
        ))
        with pytest.raises(DataContractError):
            promote(catalog, "ETHUSDT", "LONG", "staging")

    def test_the_gate_still_promotes_event_driven_evidence(
        self, tmp_path
    ) -> None:
        """De weigering is gericht, niet totaal.

        Zonder deze test zou een gate die alles weigert ook groen zijn.
        """
        catalog = ModelCatalog(tmp_path / "catalog.json")
        catalog.register(ModelRecord(
            symbol="SOLUSDT", side="LONG", version=1, stage="research",
            artifact_path="m.joblib", git_sha="abc", dvc_hash="d",
            feature_hash="f",
            metrics={"oos_logloss": 0.40, "sharpe": 2.0},
            extra={"engine": "event_driven"},
        ))
        promoted = promote(catalog, "SOLUSDT", "LONG", "staging")
        assert promoted is not None
        assert promoted.stage == "staging"


# =========================================================================== #
# 4. De vectorized berekening zelf blijft de Phase 3-conventie
# =========================================================================== #
class TestTheComputation:
    def test_it_uses_the_shift_one_convention(self) -> None:
        idx = pd.date_range("2024-01-01", periods=4, freq="D", tz="UTC")
        prices = pd.DataFrame({"BTCUSDT": [100.0, 110.0, 121.0, 121.0]}, index=idx)
        weights = pd.DataFrame({"BTCUSDT": [1.0, 1.0, 1.0, 1.0]}, index=idx)
        out = run_vectorized(weights, prices, cost_per_side=0.0,
                             initial_equity=100.0)
        # Bar 0: geen positie (shift). Bar 1: +10%. Bar 2: +10%.
        assert out.gross_returns.iloc[0] == pytest.approx(0.0)
        assert out.gross_returns.iloc[1] == pytest.approx(0.10)
        assert out.gross_returns.iloc[2] == pytest.approx(0.10)

    def test_turnover_is_charged(self) -> None:
        idx = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
        prices = pd.DataFrame({"BTCUSDT": [100.0, 100.0, 100.0]}, index=idx)
        weights = pd.DataFrame({"BTCUSDT": [1.0, 0.0, 1.0]}, index=idx)
        out = run_vectorized(weights, prices, cost_per_side=0.001,
                             initial_equity=100.0)
        assert out.turnover.iloc[0] == pytest.approx(1.0)
        assert out.turnover.iloc[1] == pytest.approx(1.0)
        assert out.returns.iloc[0] == pytest.approx(-0.001)

    def test_misaligned_inputs_are_a_hard_failure(self) -> None:
        idx = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
        prices = pd.DataFrame({"BTCUSDT": [1.0, 1.0, 1.0]}, index=idx)
        with pytest.raises(DataContractError):
            run_vectorized(
                pd.DataFrame({"BTCUSDT": [1.0, 1.0]}, index=idx[:2]),
                prices, cost_per_side=0.0, initial_equity=100.0)

    def test_negative_costs_are_a_hard_failure(self) -> None:
        idx = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
        prices = pd.DataFrame({"BTCUSDT": [1.0, 1.0, 1.0]}, index=idx)
        weights = pd.DataFrame({"BTCUSDT": [1.0, 1.0, 1.0]}, index=idx)
        with pytest.raises(DataContractError):
            run_vectorized(weights, prices, cost_per_side=-0.001,
                           initial_equity=100.0)


def test_the_config_still_declares_vectorized_inadmissible() -> None:
    """`conf/backtest/default.yaml` zegt hetzelfde als de code."""
    from pathlib import Path

    from tradebot.schemas.config import BacktestConfig, load_config

    root = Path(__file__).resolve().parents[2]
    cfg = load_config(root / "conf/backtest/default.yaml", BacktestConfig)
    assert cfg.counts_as_promotion_evidence is False
