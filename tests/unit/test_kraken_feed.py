"""`data/kraken_feed.py`: publieke Kraken-data -> de panelen en de markt van H3, zonder netwerk."""
from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import requests

from tradebot.data import kraken_feed as kf
from tradebot.schemas.robust_book_v2 import robust_book_v5_config
from tradebot.utils.failfast import DataContractError

ROOT = Path(__file__).resolve().parents[2]
NOW = pd.Timestamp("2026-10-27 06:00", tz="UTC")
#: Dagbars met start 2025-07-01 .. 2026-10-26 (de laatste is nog open op NOW).
STARTS = pd.date_range("2025-07-01", "2026-10-27", freq="D", tz="UTC")


def _path(seed: int, base: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return base * np.exp(np.cumsum(rng.normal(0.0, 0.03, len(STARTS))))


PX = {"BTCUSDT": _path(1, 60_000.0), "ETHUSDT": _path(2, 3_000.0)}
SPOT_KEY = {"XBTUSD": "XXBTZUSD", "ETHUSD": "XETHZUSD"}


class _Resp:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def json(self) -> dict:
        return self._payload


def _spot(pair: str, *, zero_vwap: bool = False) -> dict:
    sym = "BTCUSDT" if pair == "XBTUSD" else "ETHUSDT"
    # Een dagomzet van ~$50 mln per munt, in munten.
    rows = [[int(t.timestamp()), str(p), str(p * 1.01), str(p * 0.99), str(p),
             "0" if zero_vwap else str(p), str(5e7 / p), 100] for t, p in zip(STARTS, PX[sym])]
    return {"error": [], "result": {SPOT_KEY[pair]: rows[-720:], "last": 1}}


def _candles(contract: str, frm: int, to: int, *, page: int = 400,
             usd: float = 5e7) -> dict:
    sym = "BTCUSDT" if contract == "PF_XBTUSD" else "ETHUSDT"
    rows = [{"time": int(t.timestamp()) * 1000, "open": str(p), "high": str(p * 1.01),
             "low": str(p * 0.99), "close": str(p), "volume": usd / p}
            for t, p in zip(STARTS, PX[sym]) if frm <= t.timestamp() <= to]
    return {"candles": rows[:page], "more_candles": len(rows) > page}


def _funding(contract: str) -> dict:
    hours = pd.date_range(pd.Timestamp("2025-10-15", tz="UTC"), NOW + pd.Timedelta(hours=5),
                          freq="h")
    r = 2e-6 if contract == "PF_XBTUSD" else 3e-6
    return {"result": "success",
            "rates": [{"timestamp": h.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                       "relativeFundingRate": r} for h in hours]}


def _get(**kw):
    calls: list[tuple[str, dict]] = []

    def get(url: str, params: dict[str, str]) -> _Resp:
        calls.append((url, dict(params)))
        if url == kf.SPOT_OHLC_URL:
            return _Resp(_spot(params["pair"], zero_vwap=kw.get("zero_vwap", False)))
        if url == kf.FUNDING_URL:
            return _Resp(_funding(params["symbol"]))
        contract = url.split("/")[-2]
        return _Resp(_candles(contract, int(params["from"]), int(params["to"]),
                              usd=kw.get("usd", 5e7)))
    get.calls = calls  # type: ignore[attr-defined]
    return get


@pytest.fixture(scope="module")
def panels() -> kf.KrakenPanels:
    return kf.fetch_panels(_get(), now=NOW, days=600)


def test_the_panels_hold_only_settled_bars_on_the_close_time(panels):
    close = panels.perp["close"]
    assert list(close.columns) == ["BTCUSDT", "ETHUSDT"]
    # De bar die op 2026-10-26 begon, sloot op 2026-10-27 00:00 en is af; die van de 27e niet.
    assert close.index.max() == pd.Timestamp("2026-10-27", tz="UTC")
    assert close.loc["2026-10-27 00:00+00:00", "BTCUSDT"] == pytest.approx(PX["BTCUSDT"][-2])
    assert panels.spot["close"].index.equals(close.index)
    # funding[t] = de som over [t − 1 dag, t): 24 uur van 2e-6.
    assert panels.perp["funding"].loc["2026-10-27 00:00+00:00", "BTCUSDT"] == pytest.approx(
        24 * 2e-6)
    assert panels.last == pd.Timestamp("2026-10-27", tz="UTC")


def test_perp_candles_are_paged_until_the_last(panels):
    get = _get()
    kf.fetch_panels(get, now=NOW, days=600)
    pages = [c for c in get.calls if "charts" in c[0] and "PF_XBTUSD" in c[0]]
    assert len(pages) == 2 and int(pages[1][1]["from"]) > int(pages[0][1]["from"])


def test_the_market_is_h3s_market_from_the_first_funding_day(panels):
    cfg = robust_book_v5_config(ROOT / "conf/model/robust_book_v5.yaml")
    m = kf.kraken_market(panels, cfg)
    assert m.symbols == ("BTCUSDT", "ETHUSDT")
    assert m.index[0] == pd.Timestamp("2025-10-16", tz="UTC")
    assert m.perp.book.funding.loc["2026-10-27 00:00+00:00", "ETHUSDT"] == pytest.approx(
        24 * 3e-6)
    assert bool(m.perp.universe.iloc[-1].all())   # 90 dagen historie, liquide, beweeglijk
    assert m.basis.abs().max().max() < 1e-9       # spot en perp hebben hier dezelfde koers


def test_a_volume_in_the_wrong_unit_is_refused():
    with pytest.raises(DataContractError, match="plausibele bereik"):
        kf.fetch_panels(_get(usd=1.0), now=NOW, days=600)


def test_a_zero_vwap_falls_back_to_the_close():
    p = kf.fetch_panels(_get(zero_vwap=True), now=NOW, days=600)
    assert p.spot["quote_volume"].iloc[-1, 0] == pytest.approx(5e7)


@pytest.mark.parametrize(("parse", "payload", "match"), [
    (kf.parse_spot_ohlc, {"error": ["EQuery:Unknown asset pair"]}, "geen OHLC"),
    (kf.parse_spot_ohlc, {"error": [], "result": {"A": [], "last": 1}}, "precies één paar"),
    (kf.parse_spot_ohlc, {"error": [], "result": {"A": [[0, 1, 1, 1, -1, 1, 1, 1]], "last": 1}},
     "niet-positieve koers"),
    (kf.parse_perp_candles, {"candles": []}, "geen candles"),
    (kf.parse_funding, {"result": "error"}, "geen fundinghistorie"),
])
def test_malformed_answers_are_refused(parse, payload, match):
    with pytest.raises(DataContractError, match=match):
        parse(payload)


def test_panels_round_trip_through_their_manifest(panels, tmp_path: Path):
    manifest = kf.write_panels(panels, tmp_path)
    assert set(manifest["files"]) == {f"perp_{k}.csv" for k in kf.PERP_FIELDS} | {
        f"spot_{k}.csv" for k in kf.SPOT_FIELDS}
    back = kf.load_panels(tmp_path)
    pd.testing.assert_frame_equal(back.perp["close"], panels.perp["close"], check_freq=False,
                                  rtol=1e-11)
    (tmp_path / "spot_close.csv").write_text("asof_ts,BTCUSDT\n", encoding="utf-8")
    with pytest.raises(DataContractError, match="manifest"):
        kf.load_panels(tmp_path)


@pytest.mark.filterwarnings("ignore:.*found in sys.modules after import of package:RuntimeWarning")
def test_main_writes_the_panels(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys):
    monkeypatch.setattr(kf.fetch_panels, "__defaults__", (_get(),))
    monkeypatch.setattr(kf.fetch_panels, "__kwdefaults__", {"now": NOW, "days": 600})
    out = kf.main([str(tmp_path)])
    assert json.loads(capsys.readouterr().out)["last_bar"] == out["last_bar"]
    monkeypatch.setattr(sys, "argv", ["kraken_feed"])
    with pytest.raises(DataContractError, match="Gebruik"):
        runpy.run_module("tradebot.data.kraken_feed", run_name="__main__")


def test_the_real_transport_raises_on_http_errors(monkeypatch: pytest.MonkeyPatch):
    class R:
        def __init__(self, status: int) -> None:
            self.status = status

        def raise_for_status(self) -> None:
            if self.status >= 400:
                raise requests.HTTPError(str(self.status))

    monkeypatch.setattr(requests, "get", lambda url, params, timeout: R(params["s"]))
    assert kf._get(kf.SPOT_OHLC_URL, {"s": 200}).status == 200  # type: ignore[dict-item]
    with pytest.raises(requests.HTTPError):
        kf._get(kf.SPOT_OHLC_URL, {"s": 500})  # type: ignore[dict-item]
