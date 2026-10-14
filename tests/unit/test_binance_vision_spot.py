"""Het spotbeen van v5: de koppeling perp -> spot en de dagpanelen in perp-eenheden."""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tradebot.data.binance_vision import SPOT_KLINES, build_spot_panels, spot_pair_for
from tradebot.utils.failfast import DataContractError

SPOT = ("BTCUSDT", "PEPEUSDT", "1000SATSUSDT", "SATSUSDT", "BABYDOGEUSDT", "MOGUSDT")


@pytest.mark.parametrize(("perp", "expected"), [
    ("BTCUSDT", ("BTCUSDT", 1.0)),
    ("1000PEPEUSDT", ("PEPEUSDT", 1e3)),
    # De perpnaam bestaat zelf op spot: dan is dat het paar, zonder factor.
    ("1000SATSUSDT", ("1000SATSUSDT", 1.0)),
    ("1MBABYDOGEUSDT", ("BABYDOGEUSDT", 1e6)),
    ("1000000MOGUSDT", ("MOGUSDT", 1e6)),
    ("NVDAUSDT", None),
    ("1INCHUSDT", None),
])
def test_spot_pair_for(perp, expected):
    assert spot_pair_for(perp, SPOT) == expected


def _zip(path: Path, rows: list[list[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(",".join(str(v) for v in r) for r in rows).encode()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(path.stem + ".csv", body)
    path.write_bytes(buf.getvalue())


def _row(open_ms: int, close: float, qv: float, *, micro: bool = False) -> list[float]:
    t = open_ms * 1000 if micro else open_ms
    return [t, close, close * 1.1, close * 0.9, close, 1.0, t + 1, qv, 10, 0.5, qv / 2, 0]


def test_build_spot_panels_scales_prices_and_reads_microseconds(tmp_path: Path):
    day = 86_400_000
    t0 = int(pd.Timestamp("2024-12-30", tz="UTC").value // 1_000_000)
    raw = tmp_path / "raw"
    pepe = raw / SPOT_KLINES.format(s="PEPEUSDT")
    _zip(pepe / "PEPEUSDT-1d-2024-12.zip", [_row(t0, 2e-5, 1e6), _row(t0 + day, 2.2e-5, 2e6)])
    # Vanaf 2025 schrijft Binance spot-tijdstempels in microseconden.
    _zip(pepe / "PEPEUSDT-1d-2025-01.zip", [_row(t0 + 2 * day, 2.4e-5, 3e6, micro=True)])
    panels = build_spot_panels({"1000PEPEUSDT": ("PEPEUSDT", 1e3)}, raw_dir=raw,
                               panel_dir=tmp_path / "panels")
    close = panels["close"]["1000PEPEUSDT"]
    # Het raster is de SLUITtijd: open + 1 dag.
    assert close.index[0] == pd.Timestamp("2024-12-31", tz="UTC")
    assert np.allclose(close.to_numpy(), [0.02, 0.022, 0.024])
    assert np.allclose(panels["quote_volume"]["1000PEPEUSDT"].to_numpy(), [1e6, 2e6, 3e6])
    assert (tmp_path / "panels" / "close.parquet").is_file()


def test_build_spot_panels_without_data_fails(tmp_path: Path):
    with pytest.raises(DataContractError):
        build_spot_panels({"XUSDT": ("XUSDT", 1.0)}, raw_dir=tmp_path, panel_dir=tmp_path / "p")
