"""`data/external_series.py`: T-bill en funding per beurs, zonder netwerk."""
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
from tradebot.data import external_series as ex
from tradebot.utils.failfast import DataContractError

FRED = "observation_date,DTB3\n2019-08-30,1.95\n2020-04-01,0.09\n2020-04-02,.\n" \
       "2020-04-03,0.10\n2026-09-30,3.80\n2026-10-01,3.70\n"
ECB = "KEY,FREQ,TIME_PERIOD,OBS_VALUE\nEST.B,B,2019-10-01,-0.549\nEST.B,B,2020-04-01,-0.45\n" \
      "EST.B,B,2026-09-30,1.93\nEST.B,B,2026-10-01,1.92\n"


def _kraken(rates: list[tuple[str, float]]) -> dict:
    return {"result": "success",
            "rates": [{"timestamp": t, "fundingRate": 1.0, "relativeFundingRate": r}
                      for t, r in rates]}


class _Resp:
    def __init__(self, *, text: str = "", payload: dict | None = None) -> None:
        self.text, self._payload = text, payload

    def json(self) -> dict:
        assert self._payload is not None
        return self._payload


def _get(payloads: dict[str, dict], *, fred: str = FRED, ecb: str | None = ECB):
    def get(url: str, params: dict[str, str]) -> _Resp:
        if url == ex.FRED_URL:
            assert params == {"id": "DTB3"}
            return _Resp(text=fred)
        if url == ex.ECB_ESTR_URL:
            assert params == {"format": "csvdata"}
            if ecb is None:
                raise requests.ConnectionError("ECB onbereikbaar")
            return _Resp(text=ecb)
        sym = params["symbol"]
        if sym not in payloads:
            raise requests.HTTPError(f"403 voor {sym}")
        return _Resp(payload=payloads[sym])
    return get


KRAKEN = {"PF_XBTUSD": _kraken([("2023-01-01T00:00:00.000Z", 1e-5),
                                ("2023-01-01T01:00:00.000Z", 2e-5),
                                ("2023-01-02T00:00:00.000Z", -1e-5),
                                ("2026-10-01T00:00:00.000Z", 9.0)])}


def _binance(coin: str) -> pd.Series:
    idx = pd.to_datetime(["2023-01-01", "2023-01-02"])
    return pd.Series([3e-4, 1e-4], index=idx.rename("date"), name=f"binance_{coin}")


def test_parse_dtb3_takes_fractions_drops_gaps_and_stops_at_the_cycle():
    s = ex.parse_dtb3(FRED)
    assert s.index.min() == pd.Timestamp("2020-04-01")
    assert s.index.max() == ex.LAST_DAY
    assert s.loc["2020-04-03"] == pytest.approx(0.001)
    assert pd.Timestamp("2020-04-02") not in s.index
    legacy = ex.parse_dtb3(FRED.replace("observation_date", "DATE"))
    assert legacy.equals(s)
    with pytest.raises(DataContractError, match="Geen DTB3"):
        ex.parse_dtb3("DATE,DGS10\n2020-04-01,1.0\n")
    with pytest.raises(DataContractError, match="buiten"):
        ex.parse_dtb3("DATE,DTB3\n2020-04-01,40.0\n")
    with pytest.raises(DataContractError, match="zonder waarnemingen"):
        ex.parse_dtb3("DATE,DTB3\n2027-01-01,4.0\n")


def test_parse_estr_takes_fractions_and_refuses_other_files():
    s = ex.parse_estr(ECB)
    assert s.to_dict() == {pd.Timestamp("2019-10-01"): pytest.approx(-0.00549),
                           pd.Timestamp("2020-04-01"): pytest.approx(-0.0045),
                           pd.Timestamp("2026-09-30"): pytest.approx(0.0193)}
    with pytest.raises(DataContractError, match="Geen €STR"):
        ex.parse_estr("DATE,X\n2020-01-01,1\n")
    with pytest.raises(DataContractError, match="buiten"):
        ex.parse_estr("TIME_PERIOD,OBS_VALUE\n2020-01-01,30\n")


def test_kraken_funding_is_summed_per_utc_day():
    s = ex.parse_kraken_funding(KRAKEN["PF_XBTUSD"], "kraken_BTC")
    assert s.to_dict() == {pd.Timestamp("2023-01-01"): pytest.approx(3e-5),
                           pd.Timestamp("2023-01-02"): pytest.approx(-1e-5)}
    with pytest.raises(DataContractError, match="geen fundinghistorie"):
        ex.parse_kraken_funding({"result": "error"}, "kraken_BTC")


def _zip(csv: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("f.csv", csv)
    return buf.getvalue()


def test_binance_funding_reads_the_month_zips_up_to_september_2026(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    day = 1672531200000  # 2023-01-01 UTC
    files = {
        "2023-01": _zip("calc_time,funding_interval_hours,last_funding_rate\n"
                        f"{day},8,0.0001\n{day + 8 * 3600_000},8,0.0002\n"
                        f"{(day + 86_400_000) * 1000},8,-0.0001\n"),
        "2026-10": _zip(f"calc_time,funding_interval_hours,last_funding_rate\n{day},8,9\n"),
    }
    objs = [bv.S3Object(key=f"{bv.FUNDING.format(s='BTCUSDT')}BTCUSDT-fundingRate-{m}.zip",
                        etag="x-1", size=len(b)) for m, b in files.items()]

    def fetch(o: bv.S3Object, raw_dir: Path) -> Path:
        p = raw_dir / Path(o.key).name
        p.write_bytes(files[o.key[-11:-4]])
        return p

    monkeypatch.setattr(bv, "list_objects", lambda prefix: objs)
    monkeypatch.setattr(bv, "_fetch", fetch)
    s = ex.binance_daily_funding("BTC", raw_dir=tmp_path)
    assert s.to_dict() == {pd.Timestamp("2023-01-01"): pytest.approx(3e-4),
                           pd.Timestamp("2023-01-02"): pytest.approx(-1e-4)}
    monkeypatch.setattr(bv, "list_objects", lambda prefix: objs[1:])
    with pytest.raises(DataContractError, match="geen funding-zips"):
        ex.binance_daily_funding("BTC", raw_dir=tmp_path)


def test_fetch_all_writes_verified_files_and_records_unreachable_sources(tmp_path: Path):
    out = ex.fetch_all(tmp_path, get=_get(KRAKEN), binance=_binance)
    assert set(out["files"]) == {str(ex.DTB3_CSV), str(ex.ESTR_CSV), str(ex.FUNDING_CSV)}
    assert out["files"][str(ex.FUNDING_CSV)]["columns"] == ["kraken_BTC", "binance_BTC",
                                                           "binance_ETH"]
    assert "403" in out["errors"]["kraken_ETH"]
    assert json.loads((tmp_path / ex.MANIFEST).read_text(encoding="utf-8")) == out
    dtb3 = ex.load_dtb3(tmp_path)
    assert dtb3.index.max() == ex.LAST_DAY and dtb3.loc["2020-04-02"] == pytest.approx(0.0009)
    estr = ex.load_estr(tmp_path)
    assert estr.index.max() == ex.LAST_DAY and estr.loc["2020-04-02"] == pytest.approx(-0.0045)
    funding = ex.load_funding(tmp_path)
    assert funding.loc["2023-01-01", "kraken_BTC"] == pytest.approx(3e-5)
    again = ex.fetch_all(tmp_path, get=_get(KRAKEN), binance=_binance)
    assert again == out  # dezelfde bronnen geven dezelfde bytes
    (tmp_path / ex.DTB3_CSV).write_text("date,dtb3\n2020-04-01,0.5\n", encoding="utf-8")
    with pytest.raises(DataContractError, match="manifest"):
        ex.load_dtb3(tmp_path)


def test_without_any_funding_only_the_t_bill_is_written(tmp_path: Path):
    def broken(coin: str) -> pd.Series:
        raise requests.ConnectionError("geen route")
    out = ex.fetch_all(tmp_path, get=_get({}, ecb=None), binance=broken)
    assert set(out["files"]) == {str(ex.DTB3_CSV)} and len(out["errors"]) == 5
    assert "ECB" in out["errors"]["estr"]
    assert not (tmp_path / ex.FUNDING_CSV).exists()
    with pytest.raises(DataContractError, match="manifest"):
        ex.load_estr(tmp_path)


@pytest.mark.filterwarnings("ignore:.*found in sys.modules after import of package:RuntimeWarning")
def test_main_and_the_script_guard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys):
    monkeypatch.setattr(ex, "ROOT", tmp_path)
    monkeypatch.setattr(ex, "_get", _get(KRAKEN))
    monkeypatch.setattr(ex, "binance_daily_funding", _binance)
    monkeypatch.setattr(ex.fetch_all, "__defaults__", (tmp_path,))
    monkeypatch.setattr(ex.fetch_all, "__kwdefaults__", {"get": _get(KRAKEN),
                                                         "binance": _binance})
    out = ex.main([])
    assert str(ex.DTB3_CSV) in json.loads(capsys.readouterr().out)["files"]
    assert out["errors"].keys() == {"kraken_ETH"}
    monkeypatch.setattr(sys, "argv", ["external_series", "x"])
    with pytest.raises(DataContractError, match="Gebruik"):
        runpy.run_module("tradebot.data.external_series", run_name="__main__")


def test_the_real_transport_raises_on_http_errors(monkeypatch: pytest.MonkeyPatch):
    class R:
        def __init__(self, status: int) -> None:
            self.status = status

        def raise_for_status(self) -> None:
            if self.status >= 400:
                raise requests.HTTPError(str(self.status))
    seen = {}

    def fake(url, params, timeout):
        seen.update(url=url, params=params, timeout=timeout)
        return R(params.pop("status"))
    monkeypatch.setattr(requests, "get", fake)
    assert ex._get(ex.FRED_URL, {"id": "DTB3", "status": 200}).status == 200  # type: ignore[dict-item]
    with pytest.raises(requests.HTTPError):
        ex._get(ex.FRED_URL, {"id": "DTB3", "status": 500})  # type: ignore[dict-item]
    assert seen == {"url": ex.FRED_URL, "params": {"id": "DTB3"}, "timeout": 60}
