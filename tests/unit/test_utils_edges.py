# tests/unit/test_utils_edges.py
"""De randen van `tradebot.utils`: Numba-invoer, hashing, Parquet-I/O en tijd.

Dit zijn de grenspoorten van de pijplijn. Wat hier stil doorgaat -- een NaN
naar een Numba-kernel, een naieve timestamp, een leeg frame met een
willekeurige hash -- komt verderop terug als een getal dat niemand kan
herleiden.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pandera.pandas as pa
import pytest
from pandera.typing import Series

from tradebot.utils.arrays import numba_array, numba_array_nonan
from tradebot.utils.failfast import DataContractError, has_module
from tradebot.utils.hashing import dataframe_content_hash
from tradebot.utils.parquet_io import (
    read_parquet_metadata,
    read_validated_parquet,
    write_validated_parquet,
)
from tradebot.utils.time import (
    asof_join,
    from_utc_ns,
    to_utc,
    to_utc_ns,
    to_utc_ns_series,
)


class TestNumbaArrays:
    def test_nan_is_refused_before_a_kernel(self) -> None:
        with pytest.raises(ValueError, match="NaN values in column 'x'"):
            numba_array(pd.Series([1.0, np.nan], name="x"), dtype="f64")

    def test_a_strided_column_is_copied_once_and_logged(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        strided = pd.Series(np.arange(10, dtype=np.float64)[::2], name="s")
        assert not strided.to_numpy(copy=False).flags["C_CONTIGUOUS"]
        with caplog.at_level(logging.WARNING):
            arr = numba_array(strided, dtype="f64")
        assert arr.flags["C_CONTIGUOUS"]
        assert np.array_equal(arr, [0.0, 2.0, 4.0, 6.0, 8.0])
        assert "Non-contiguous" in caplog.text

    def test_the_nonan_variant_fills_explicitly(self) -> None:
        arr = numba_array_nonan(pd.Series([1.0, np.nan, 3.0], name="x"), dtype="f64", fill=-1.0)
        assert np.array_equal(arr, [1.0, -1.0, 3.0])


def test_has_module_looks_at_the_top_level_package() -> None:
    assert has_module("numpy.linalg") is True
    assert has_module("a_package_that_does_not_exist_anywhere") is False


def test_an_empty_frame_has_a_fixed_content_hash() -> None:
    a = dataframe_content_hash(pd.DataFrame())
    b = dataframe_content_hash(pd.DataFrame({"x": []}))
    assert a == b
    assert a != dataframe_content_hash(pd.DataFrame({"x": [1.0]}))


class _Tiny(pa.DataFrameModel):
    x: Series[float] = pa.Field(ge=0.0)


class TestValidatedParquet:
    def test_a_roundtrip_carries_the_schema_and_the_metadata(
        self, tmp_path: Path,
    ) -> None:
        frame = pd.DataFrame({"x": [0.0, 1.5, 2.0]})
        path = write_validated_parquet(
            frame, tmp_path / "nested" / "tiny.parquet", _Tiny,
            metadata={"symbol": "BTC", "rows": 3})
        assert path.is_absolute() and path.exists()
        meta = read_parquet_metadata(path)
        assert meta["tradebot_schema"] == "_Tiny"
        assert meta["symbol"] == "BTC"
        assert meta["rows"] == "3"
        back = read_validated_parquet(path, _Tiny)
        pd.testing.assert_frame_equal(back, frame)

    def test_an_invalid_frame_is_never_written(self, tmp_path: Path) -> None:
        with pytest.raises(pa.errors.SchemaErrors):
            write_validated_parquet(pd.DataFrame({"x": [-1.0]}),
                                    tmp_path / "bad.parquet", _Tiny)
        assert not (tmp_path / "bad.parquet").exists()

    def test_reading_without_metadata_or_file(self, tmp_path: Path) -> None:
        plain = tmp_path / "plain.parquet"
        pd.DataFrame({"x": [1.0]}).to_parquet(plain)
        assert "tradebot_schema" not in read_parquet_metadata(plain)
        with pytest.raises(FileNotFoundError, match="Artefact not found"):
            read_validated_parquet(tmp_path / "missing.parquet", _Tiny)


class TestTime:
    def test_to_utc_accepts_every_representation(self) -> None:
        expected = pd.Timestamp("2024-01-01 12:00", tz="UTC")
        assert to_utc(pd.Timestamp("2024-01-01 12:00")) == expected
        assert to_utc(pd.Timestamp("2024-01-01 13:00", tz="Europe/Amsterdam")) == expected
        assert to_utc(datetime(2024, 1, 1, 12, 0)) == expected
        assert to_utc(datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)) == expected

    def test_to_utc_ns_accepts_aware_inputs_only(self) -> None:
        expected = pd.Timestamp("2024-01-01", tz="UTC").value
        assert to_utc_ns("2024-01-01T00:00:00Z") == expected
        assert to_utc_ns(pd.Timestamp("2024-01-01", tz="UTC")) == expected
        assert to_utc_ns(datetime(2024, 1, 1, tzinfo=timezone.utc)) == expected
        with pytest.raises(DataContractError):
            to_utc_ns("2024-01-01")
        with pytest.raises(DataContractError, match="verwacht str"):
            to_utc_ns(1_700_000_000)  # type: ignore[arg-type]

    def test_the_series_variant_refuses_naive_and_non_datetime(self) -> None:
        aware = pd.Series(pd.date_range("2024-01-01", periods=2, freq="D", tz="UTC"))
        ns = to_utc_ns_series(aware)
        assert list(ns) == [t.value for t in aware]
        with pytest.raises(DataContractError):
            to_utc_ns_series(aware.dt.tz_localize(None))
        with pytest.raises(DataContractError):
            to_utc_ns_series(pd.Series([1, 2]))

    def test_from_utc_ns_inverts_both_shapes(self) -> None:
        stamp = pd.Timestamp("2024-03-01 08:00", tz="UTC")
        assert from_utc_ns(stamp.value) == stamp
        assert from_utc_ns(np.int64(stamp.value)) == stamp
        back = from_utc_ns([stamp.value, stamp.value + 1])
        assert isinstance(back, pd.Series)
        assert back.iloc[0] == stamp

    def test_asof_join_renames_overlapping_columns(self) -> None:
        index = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC",
                              name="ts")
        left = pd.DataFrame({"value": [1.0, 2.0, 3.0]}, index=index)
        right = pd.DataFrame({
            "asof_ts": pd.date_range("2023-12-31", periods=3, freq="D", tz="UTC"),
            "value": [10.0, 20.0, 30.0]})
        joined = asof_join(left, right, tolerance=pd.Timedelta(days=2))
        assert list(joined["value_r"]) == [20.0, 30.0, 30.0]
        renamed = asof_join(left, right, suffix="_pit",
                            tolerance=pd.Timedelta(days=2))
        assert "value_pit" in renamed.columns
