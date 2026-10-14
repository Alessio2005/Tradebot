# tests/unit/test_package_root.py
"""Het pakket zelf: zijn versie en zijn foutklassen.

`tradebot.__version__` komt uit de geïnstalleerde metadata. Zonder die metadata
-- een checkout die niet met `pip install -e .` is geïnstalleerd -- valt hij
terug op een vaste string, en de versie-tuple moet daar nog steeds uit te
lezen zijn.
"""
from __future__ import annotations

import importlib
import importlib.metadata

import pytest

import tradebot
from tradebot.exceptions import (
    FeatureSchemaMismatch,
    LookaheadError,
    PublicationLagViolation,
    SchemaViolationError,
    StageInputMissing,
    TradebotError,
)


def test_the_version_falls_back_without_installed_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _missing(name: str) -> str:
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", _missing)
    try:
        importlib.reload(tradebot)
        assert tradebot.__version__ == "0.4.0"
        assert tradebot.__version_tuple__ == (0, 4, 0)
    finally:
        monkeypatch.undo()
        importlib.reload(tradebot)
    assert tradebot.__version__ == importlib.metadata.version("tradebot")


class TestExceptions:
    def test_a_schema_violation_names_its_schema(self) -> None:
        error = SchemaViolationError("kolom ontbreekt", schema_name="bars")
        assert error.schema_name == "bars"
        assert str(error) == "[bars] kolom ontbreekt"
        assert str(SchemaViolationError("kolom ontbreekt")) == "kolom ontbreekt"

    def test_a_publication_lag_violation_is_a_lookahead_error(self) -> None:
        error = PublicationLagViolation("CPI", "2024-01-01", "2024-01-15")
        assert isinstance(error, LookaheadError)
        assert str(error) == "Pub-lag: 'CPI' at 2024-01-01 published at 2024-01-15"

    def test_a_feature_schema_mismatch_lists_at_most_five_columns(self) -> None:
        missing = [f"m{i}" for i in range(7)]
        error = FeatureSchemaMismatch("BTC", missing=missing)
        assert error.symbol == "BTC"
        assert error.missing == missing
        assert error.extra == []
        assert "m4" in str(error) and "m5" not in str(error)

    def test_a_missing_stage_input_names_the_stage_and_the_path(self) -> None:
        error = StageInputMissing("features", "/data/x.parquet")
        assert isinstance(error, TradebotError)
        assert str(error) == "Stage 'features' missing: /data/x.parquet"
