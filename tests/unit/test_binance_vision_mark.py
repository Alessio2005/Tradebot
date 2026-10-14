"""De mark-price-dagbars (v6): waar Binance op liquideert, op het perpraster."""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tradebot.data.binance_vision import MARK_KLINES, build_mark_panels
from tradebot.utils.failfast import DataContractError

DAY = 86_400_000


def _zip(path: Path, rows: list[list[float]], *, header: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [",".join(str(v) for v in r) for r in rows]
    if header:
        lines.insert(0, "open_time,open,high,low,close,volume,close_time,quote_volume,count,"
                        "taker_buy_volume,taker_buy_quote_volume,ignore")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(path.stem + ".csv", "\n".join(lines).encode())
    path.write_bytes(buf.getvalue())


def _row(open_ms: int, close: float) -> list[float]:
    # Mark-price-klines hebben geen volume: die kolommen zijn 0.
    return [open_ms, close, close * 1.05, close * 0.97, close, 0, open_ms + DAY - 1, 0,
            86400, 0, 0, 0]


def test_build_mark_panels_on_the_close_grid(tmp_path: Path):
    t0 = int(pd.Timestamp("2021-05-18", tz="UTC").value // 1_000_000)
    raw = tmp_path / "raw"
    d = raw / MARK_KLINES.format(s="BTCUSDT")
    _zip(d / "BTCUSDT-1d-2021-05.zip", [_row(t0, 100.0), _row(t0 + DAY, 80.0)])
    _zip(d / "BTCUSDT-1d-2021-06.zip", [_row(t0 + 2 * DAY, 90.0)], header=True)
    e = raw / MARK_KLINES.format(s="ETHUSDT")
    _zip(e / "ETHUSDT-1d-2021-05.zip", [_row(t0 + DAY, 10.0)])
    panels = build_mark_panels(["BTCUSDT", "ETHUSDT", "NOPEUSDT"], raw_dir=raw,
                               panel_dir=tmp_path / "panels")
    assert set(panels) == {"high", "low", "close"}
    high = panels["high"]
    # Het raster is de SLUITtijd (open + 1 dag), zoals de perp- en spotpanelen.
    assert high.index[0] == pd.Timestamp("2021-05-19", tz="UTC")
    assert list(high.columns) == ["BTCUSDT", "ETHUSDT"]
    assert np.allclose(high["BTCUSDT"].to_numpy(), [105.0, 84.0, 94.5])
    assert np.isnan(high["ETHUSDT"].iloc[0]) and high["ETHUSDT"].iloc[1] == pytest.approx(10.5)
    assert np.allclose(panels["low"]["BTCUSDT"].to_numpy(), [97.0, 77.6, 87.3])
    assert (tmp_path / "panels" / "high.parquet").is_file()


def test_build_mark_panels_without_data_fails(tmp_path: Path):
    with pytest.raises(DataContractError):
        build_mark_panels(["XUSDT"], raw_dir=tmp_path, panel_dir=tmp_path / "p")
