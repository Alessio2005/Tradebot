"""`data/positioning_series.py`: spot-close en CFTC-TFF, zonder netwerk."""
from __future__ import annotations

import io
import json
import runpy
import sys
import zipfile
from pathlib import Path

import pandas as pd
import pytest
import requests

from tradebot.data import binance_vision as bv
from tradebot.data import positioning_series as ps
from tradebot.utils.failfast import DataContractError


def _row(date: str, market: str, code: str, am_long: str = "100", **kw: str) -> dict:
    row = {k: "0" for k in ps.TFF_COLUMNS}
    row.update({"report_date_as_yyyy_mm_dd": f"{date}T00:00:00.000",
                "market_and_exchange_names": market, "cftc_contract_market_code": code,
                "open_interest_all": "1000", "asset_mgr_positions_long": am_long,
                "asset_mgr_positions_short": "40"})
    row.update(kw)
    return row


BTC = "BITCOIN - CHICAGO MERCANTILE EXCHANGE"
ETH = "ETHER CASH SETTLED - CHICAGO MERCANTILE EXCHANGE"
ROWS = [
    _row("2021-01-05", BTC, "133741"),
    _row("2021-01-05", BTC, "133741", am_long="120"),          # dubbel: de laatste telt
    _row("2021-01-05", ETH, "146021", am_long=""),             # leeg = 0
    _row("2021-01-05", "EURO FX - CHICAGO MERCANTILE EXCHANGE", "099741"),
    _row("2019-06-04", BTC, "133741"),                         # vóór de cyclus
    _row("2026-10-06", BTC, "133741"),                         # vooruit-sample
]


class _Resp:
    def __init__(self, payload: list) -> None:
        self._payload = payload

    def json(self) -> list:
        return self._payload


def _get(rows: list | None = ROWS):
    def get(url: str, params: dict[str, str]) -> _Resp:
        assert url == ps.CFTC_TFF_URL and params["$where"] == ps.TFF_WHERE
        if rows is None:
            raise requests.ConnectionError("CFTC onbereikbaar")
        return _Resp(rows)
    return get


def _spot(coin: str) -> pd.Series:
    idx = pd.to_datetime(["2021-01-01", "2021-01-02"]).rename("date")
    return pd.Series([100.0, 110.0], index=idx, name=coin)


def test_parse_tff_keeps_cme_crypto_inside_the_cycle():
    df = ps.parse_tff(ROWS)
    assert list(df["code"]) == ["133741", "146021"]
    assert df.loc[0, "asset_mgr_long"] == 120 and df.loc[1, "asset_mgr_long"] == 0
    assert (df["report_date"] == pd.Timestamp("2021-01-05")).all()
    with pytest.raises(DataContractError, match="geen TFF"):
        ps.parse_tff([])
    with pytest.raises(DataContractError, match="ontbreken"):
        ps.parse_tff([{"report_date_as_yyyy_mm_dd": "2021-01-05"}])
    with pytest.raises(DataContractError, match="Geen CME"):
        ps.parse_tff(ROWS[3:4])
    with pytest.raises(DataContractError, match="negatieve"):
        ps.parse_tff([_row("2021-01-05", BTC, "133741", am_long="-5")])


def _zip(csv: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("f.csv", csv)
    return buf.getvalue()


def test_spot_close_is_indexed_by_the_close_and_stops_at_the_cycle(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    day = 1609459200000  # 2021-01-01 UTC
    rows = "".join(f"{day + i * bv.DAY_MS},1,1,1,{100 + i},1,0,{5 + i},1,1,1,0\n"
                   for i in range(3))
    files = {"2021-01": _zip(rows),
             "2026-10": _zip(f"{1790812800000},1,1,1,9,1,0,1,1,1,1,0\n"),
             "2019-01": _zip(f"{1546300800000},1,1,1,9,1,0,1,1,1,1,0\n")}
    objs = [bv.S3Object(key=f"{bv.SPOT_KLINES.format(s='BTCUSDT')}BTCUSDT-1d-{m}.zip",
                        etag="x-1", size=len(b)) for m, b in files.items()]

    def fetch(o: bv.S3Object, raw_dir: Path) -> Path:
        p = raw_dir / Path(o.key).name
        p.write_bytes(files[o.key[-11:-4]])
        return p

    monkeypatch.setattr(bv, "list_objects", lambda prefix: objs)
    monkeypatch.setattr(bv, "_fetch", fetch)
    s = ps._binance_spot("BTC", raw_dir=tmp_path)
    assert s.name == "BTC"
    assert s.to_dict() == {pd.Timestamp("2021-01-02"): 100.0, pd.Timestamp("2021-01-03"): 101.0,
                           pd.Timestamp("2021-01-04"): 102.0}
    with pytest.raises(DataContractError, match="Geen spot-zips"):
        ps.spot_close("BTCUSDT", [])
    bad = tmp_path / "bad.zip"
    bad.write_bytes(_zip(f"{day},1,1,1,0,1,0,1,1,1,1,0\n"))
    with pytest.raises(DataContractError, match="niet-positieve"):
        ps.spot_close("BTCUSDT", [bad])


def test_fetch_all_writes_verified_files(tmp_path: Path):
    out = ps.fetch_all(tmp_path, get=_get(), spot=_spot)
    assert set(out["files"]) == {str(ps.SPOT_CSV), str(ps.TFF_CSV)} and out["errors"] == {}
    assert out["files"][str(ps.TFF_CSV)]["markets"] == {"133741": BTC, "146021": ETH}
    assert json.loads((tmp_path / ps.MANIFEST).read_text(encoding="utf-8")) == out
    spot = ps.load_spot_close(tmp_path)
    assert list(spot.columns) == ["BTC", "ETH"] and spot.loc["2021-01-02", "ETH"] == 110.0
    tff = ps.load_tff(tmp_path)
    assert list(tff["code"]) == ["133741", "146021"]
    assert tff["report_date"].dtype.kind == "M"
    assert ps.fetch_all(tmp_path, get=_get(), spot=_spot) == out
    (tmp_path / ps.SPOT_CSV).write_text("date,BTC\n2021-01-01,1\n", encoding="utf-8")
    with pytest.raises(DataContractError, match="manifest"):
        ps.load_spot_close(tmp_path)


def test_unreachable_sources_are_recorded_not_fatal(tmp_path: Path):
    def broken(coin: str) -> pd.Series:
        raise requests.ConnectionError("geen route")
    out = ps.fetch_all(tmp_path, get=_get(None), spot=broken)
    assert out["files"] == {} and set(out["errors"]) == {"spot_BTC", "spot_ETH", "cftc_tff"}
    with pytest.raises(DataContractError, match="manifest"):
        ps.load_tff(tmp_path)


@pytest.mark.filterwarnings("ignore:.*found in sys.modules after import of package:RuntimeWarning")
def test_main_and_the_script_guard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys):
    monkeypatch.setattr(ps, "ROOT", tmp_path)
    monkeypatch.setattr(ps.fetch_all, "__defaults__", (tmp_path,))
    monkeypatch.setattr(ps.fetch_all, "__kwdefaults__", {"get": _get(), "spot": _spot})
    out = ps.main([])
    assert str(ps.TFF_CSV) in json.loads(capsys.readouterr().out)["files"]
    assert out["errors"] == {}
    monkeypatch.setattr(sys, "argv", ["positioning_series", "x"])
    with pytest.raises(DataContractError, match="Gebruik"):
        runpy.run_module("tradebot.data.positioning_series", run_name="__main__")


def test_the_real_transport_raises_on_http_errors(monkeypatch: pytest.MonkeyPatch):
    class R:
        def __init__(self, status: int) -> None:
            self.status = status

        def raise_for_status(self) -> None:
            if self.status >= 400:
                raise requests.HTTPError(str(self.status))

    def fake(url, params, timeout):
        assert timeout == 120
        return R(params.pop("status"))
    monkeypatch.setattr(requests, "get", fake)
    assert ps._get(ps.CFTC_TFF_URL, {"status": 200}).status == 200  # type: ignore[dict-item]
    with pytest.raises(requests.HTTPError):
        ps._get(ps.CFTC_TFF_URL, {"status": 500})  # type: ignore[dict-item]
