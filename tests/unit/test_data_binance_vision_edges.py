# tests/unit/test_data_binance_vision_edges.py
"""`data/binance_vision.py`: de S3-listing, de verifieerde download, `sync*`, de dagpanelen en de CLI.

Geen netwerk: `_SESSION` is een nagebootste bucket die lijst en objecten uit een dict bedient (met
pagina's, zodat de `marker`-lus echt loopt). De zips zijn klein en worden hier zelf gebouwd. De
mark-price-panelen en de spot-hulpfuncties staan in `test_binance_vision_mark.py` / `_spot.py`.
"""
from __future__ import annotations

import hashlib
import io
import json
import runpy
import sys
import zipfile
from pathlib import Path
from typing import ClassVar

import pandas as pd
import pytest
import requests
from tenacity import wait_none

from tradebot.data import binance_vision as bv
from tradebot.utils.failfast import DataContractError

DAY = bv.DAY_MS
KLINE_HEADER = ("open_time,open,high,low,close,volume,close_time,quote_volume,count,"
                "taker_buy_volume,taker_buy_quote_volume,ignore")


# --------------------------------------------------------------------------- #
# bouwstenen
# --------------------------------------------------------------------------- #
def _zip_bytes(csv: str, name: str = "data.csv") -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(name, csv)
    return buf.getvalue()


def _klines_csv(first_day_ms: int, closes: list[float], *, header: bool = False) -> str:
    rows = [f"{first_day_ms + i * DAY},{c},{c * 1.01},{c * 0.99},{c},10,{first_day_ms + (i + 1) * DAY - 1},"
            f"{1000 + i},5,1,1,0" for i, c in enumerate(closes)]
    return "\n".join(([KLINE_HEADER] if header else []) + rows)


def _funding_csv(rows: list[tuple[int, float]], *, header: bool = True) -> str:
    lines = [f"{ms},8,{rate}" for ms, rate in rows]
    return "\n".join((["calc_time,funding_interval_hours,last_funding_rate"] if header else []) + lines)


def _write(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


class _Resp:
    def __init__(self, text: str = "", content: bytes = b"") -> None:
        self.text, self.content = text, content

    def raise_for_status(self) -> None:
        return None


class _Bucket:
    """Een S3-bucket in het geheugen met pagina's van `page` items."""

    def __init__(self, files: dict[str, bytes], page: int = 2, etags: dict[str, str] | None = None) -> None:
        self.files, self.page, self.etags = files, page, etags or {}
        self.downloads: list[str] = []
        self.listings: list[dict[str, str]] = []
        self.corrupt: set[str] = set()

    def etag(self, key: str) -> str:
        return self.etags.get(key) or hashlib.md5(self.files[key]).hexdigest()

    def get(self, url: str, params: dict[str, str] | None = None, timeout: float | None = None) -> _Resp:
        key = url[len(bv.BUCKET_URL):].lstrip("/")
        if key:
            self.downloads.append(key)
            return _Resp(content=b"corrupt" if key in self.corrupt else self.files[key])
        params = params or {}
        self.listings.append(dict(params))
        prefix, marker = params["prefix"], params.get("marker", "")
        if params.get("delimiter") == "/":
            names = sorted({k[len(prefix):].split("/", 1)[0] for k in self.files
                            if k.startswith(prefix) and "/" in k[len(prefix):]})
            names = [n for n in names if f"{prefix}{n}/" > marker]
            page, more = names[: self.page], len(names) > self.page
            xml = "".join(f"<CommonPrefixes><Prefix>{prefix}{n}/</Prefix></CommonPrefixes>" for n in page)
            tail = f"<IsTruncated>true</IsTruncated><NextMarker>{prefix}{page[-1]}/</NextMarker>" if more \
                else "<IsTruncated>false</IsTruncated>"
            return _Resp(text=f"<ListBucketResult>{xml}{tail}</ListBucketResult>")
        keys = sorted(k for k in self.files if k.startswith(prefix) and k > marker)
        page, more = keys[: self.page], len(keys) > self.page
        xml = "".join(f"<Contents><Key>{k}</Key><ETag>&quot;{self.etag(k)}&quot;</ETag>"
                      f"<Size>{len(self.files[k])}</Size></Contents>" for k in page)
        tail = "<IsTruncated>true</IsTruncated>" if more else "<IsTruncated>false</IsTruncated>"
        return _Resp(text=f"<ListBucketResult>{xml}{tail}</ListBucketResult>")


def _install(monkeypatch: pytest.MonkeyPatch, files: dict[str, bytes], **kw) -> _Bucket:
    bucket = _Bucket(files, **kw)
    monkeypatch.setattr(bv, "_SESSION", bucket)
    return bucket


def _um_files(symbols: tuple[str, ...] = ("AAAUSDT", "BBBUSDT"), months: tuple[str, ...] = ("2021-05", "2021-06", "2021-07"),
              ) -> dict[str, bytes]:
    t0 = int(pd.Timestamp("2021-05-01", tz="UTC").value // 1_000_000)
    files: dict[str, bytes] = {}
    for s in symbols:
        for m in months:
            files[f"{bv.KLINES.format(s=s)}{s}-1d-{m}.zip"] = _zip_bytes(_klines_csv(t0, [100.0, 101.0]))
            files[f"{bv.KLINES.format(s=s)}{s}-1d-{m}.zip.CHECKSUM"] = b"checksum"
        files[f"{bv.FUNDING.format(s=s)}{s}-fundingRate-{months[0]}.zip"] = _zip_bytes(
            _funding_csv([(t0, 0.0001)]))
    return files


# --------------------------------------------------------------------------- #
# HTTP en listing
# --------------------------------------------------------------------------- #
class TestGet:
    def test_the_bucket_root_and_an_object_key_build_their_urls(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: list[tuple[str, dict | None, float]] = []

        class Session:
            def get(self, url, params=None, timeout=None):
                seen.append((url, params, timeout))
                return _Resp(text="ok")

        monkeypatch.setattr(bv, "_SESSION", Session())
        bv._get({"prefix": "x"})
        bv._get(key="a/b.zip")
        assert seen == [(bv.BUCKET_URL, {"prefix": "x"}, 60), (f"{bv.BUCKET_URL}/a/b.zip", None, 60)]

    def test_a_failing_request_is_retried_five_times_without_waiting_and_then_raised(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls: list[int] = []

        class Session:
            def get(self, url, params=None, timeout=None):
                calls.append(1)
                raise requests.ConnectionError("down")

        monkeypatch.setattr(bv, "_SESSION", Session())
        monkeypatch.setattr(bv._get.retry, "wait", wait_none())
        with pytest.raises(requests.ConnectionError):
            bv._get()
        assert len(calls) == 5


class TestListing:
    FILES: ClassVar[dict[str, bytes]] = {
        f"data/futures/um/monthly/klines/{s}/1d/{s}-1d-2021-05.zip": b"x"
        for s in ("AAAUSDT", "BBBUSDT", "CCCUSDT_240628", "DDDBUSD", "EEEUSDT")}

    def test_common_prefixes_are_followed_across_pages(self, monkeypatch: pytest.MonkeyPatch) -> None:
        bucket = _install(monkeypatch, self.FILES, page=2)
        prefixes = bv.list_common_prefixes("data/futures/um/monthly/klines/")
        assert [p.rstrip("/").rsplit("/", 1)[-1] for p in prefixes] == [
            "AAAUSDT", "BBBUSDT", "CCCUSDT_240628", "DDDBUSD", "EEEUSDT"]
        assert [x["marker"] for x in bucket.listings] == ["", "data/futures/um/monthly/klines/BBBUSDT/",
                                                          "data/futures/um/monthly/klines/DDDBUSD/"]

    def test_only_perpetual_usdt_symbols_are_listed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install(monkeypatch, self.FILES)
        assert bv.list_symbols() == ["AAAUSDT", "BBBUSDT", "EEEUSDT"]

    def test_objects_carry_key_etag_and_size_and_paging_continues_from_the_last_key(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        files = {f"p/{i}.zip": bytes(i + 1) for i in range(5)}
        bucket = _install(monkeypatch, files, page=2)
        objs = bv.list_objects("p/")
        assert [o.key for o in objs] == [f"p/{i}.zip" for i in range(5)]
        assert objs[2] == bv.S3Object(key="p/2.zip", etag=hashlib.md5(bytes(3)).hexdigest(), size=3)
        assert [x["marker"] for x in bucket.listings] == ["", "p/1.zip", "p/3.zip"]

    def test_spot_symbols_are_all_usdt_pairs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        files = {f"data/spot/monthly/klines/{s}/1d/{s}-1d-2021-05.zip": b"x"
                 for s in ("BTCUSDT", "ETHBTC", "PEPEUSDT")}
        _install(monkeypatch, files)
        assert bv.list_spot_symbols() == ["BTCUSDT", "PEPEUSDT"]


# --------------------------------------------------------------------------- #
# verifieerde download
# --------------------------------------------------------------------------- #
class TestVerifiedDownload:
    DATA = b"some zip bytes"

    def _obj(self, etag: str | None = None, size: int | None = None) -> bv.S3Object:
        return bv.S3Object("k/a.zip", etag or hashlib.md5(self.DATA).hexdigest(),
                           len(self.DATA) if size is None else size)

    def test_a_file_is_verified_by_size_and_md5(self, tmp_path: Path) -> None:
        path = tmp_path / "a.zip"
        assert not bv._verified(path, self._obj())                            # bestaat niet
        path.write_bytes(self.DATA)
        assert bv._verified(path, self._obj())
        assert not bv._verified(path, self._obj(size=3))                      # andere grootte
        assert not bv._verified(path, self._obj(etag="0" * 32))               # andere MD5

    def test_a_multipart_etag_is_not_an_md5_so_the_size_decides(self, tmp_path: Path) -> None:
        path = _write(tmp_path / "a.zip", self.DATA)
        assert bv._verified(path, self._obj(etag="abc-3"))
        assert not bv._verified(path, self._obj(etag="abc-3", size=1))

    def test_a_verified_file_is_not_downloaded_again(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        bucket = _install(monkeypatch, {"k/a.zip": self.DATA})
        _write(tmp_path / "k/a.zip", self.DATA)
        assert bv._fetch(self._obj(), tmp_path) == tmp_path / "k/a.zip"
        assert bucket.downloads == []

    def test_a_download_is_checked_against_the_etag_and_written_through_a_part_file(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        bucket = _install(monkeypatch, {"k/a.zip": self.DATA})
        path = bv._fetch(self._obj(), tmp_path)
        assert path.read_bytes() == self.DATA and bucket.downloads == ["k/a.zip"]
        assert not list(tmp_path.rglob("*.part"))

    def test_a_download_that_differs_from_its_etag_is_refused_and_leaves_nothing_behind(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        bucket = _install(monkeypatch, {"k/a.zip": self.DATA})
        bucket.corrupt.add("k/a.zip")
        with pytest.raises(DataContractError, match="wijkt af van de S3-ETag"):
            bv._fetch(self._obj(), tmp_path)
        assert not list(tmp_path.rglob("*"))

    def test_a_multipart_object_is_accepted_without_an_md5_check(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        _install(monkeypatch, {"k/a.zip": self.DATA})
        assert bv._fetch(self._obj(etag="abc-3"), tmp_path).read_bytes() == self.DATA


# --------------------------------------------------------------------------- #
# sync
# --------------------------------------------------------------------------- #
class TestSync:
    def test_klines_and_funding_up_to_the_last_month_are_downloaded_and_manifested(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        files = _um_files()
        bucket = _install(monkeypatch, files)
        manifest_path = tmp_path / "out" / "manifest.json"
        m = bv.sync(["BBBUSDT", "AAAUSDT"], raw_dir=tmp_path / "raw", workers=2, last_month="2021-06",
                    manifest_path=manifest_path)
        wanted = sorted(k for k in files if k.endswith(".zip") and "2021-07" not in k)
        assert sorted(bucket.downloads) == wanted
        assert all((tmp_path / "raw" / k).is_file() for k in wanted)
        digest = hashlib.sha256("\n".join(f"{k} {bucket.etag(k)}" for k in wanted).encode()).hexdigest()
        assert m == {"source": bv.BUCKET_URL, "last_month": "2021-06", "n_symbols": 2,
                     "n_objects": len(wanted), "objects_sha256": digest,
                     "symbols": ["BBBUSDT", "AAAUSDT"], "bytes": sum(len(files[k]) for k in wanted)}
        assert json.loads(manifest_path.read_text(encoding="utf-8")) == m

    def test_a_second_run_downloads_nothing_and_gives_the_same_digest(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        bucket = _install(monkeypatch, _um_files(("AAAUSDT",)))
        kw = {"raw_dir": tmp_path / "raw", "workers": 2, "last_month": "2021-07",
              "manifest_path": tmp_path / "m.json"}
        first = bv.sync(["AAAUSDT"], **kw)
        bucket.downloads.clear()
        second = bv.sync(["AAAUSDT"], **kw)
        assert bucket.downloads == [] and first["objects_sha256"] == second["objects_sha256"]

    def test_without_symbols_the_whole_bucket_is_listed(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        _install(monkeypatch, _um_files())
        m = bv.sync(raw_dir=tmp_path / "raw", workers=2, last_month="2021-05", manifest_path=tmp_path / "m.json")
        assert m["symbols"] == ["AAAUSDT", "BBBUSDT"] and m["n_objects"] == 2 * 2


class TestSyncSpotAndMark:
    def test_spot_pairs_are_mapped_downloaded_and_manifested(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        t0 = int(pd.Timestamp("2021-05-01", tz="UTC").value // 1_000_000)
        files = {}
        for s in ("BTCUSDT", "PEPEUSDT"):
            for m in ("2019-11", "2021-05", "2021-08"):
                files[f"{bv.SPOT_KLINES.format(s=s)}{s}-1d-{m}.zip"] = _zip_bytes(_klines_csv(t0, [1.0]))
        bucket = _install(monkeypatch, files)
        m = bv.sync_spot(["BTCUSDT", "1000PEPEUSDT", "NOSPOTUSDT"], raw_dir=tmp_path / "raw", workers=2,
                         last_month="2021-06", manifest_path=tmp_path / "spot.json")
        # Voor SPOT_FIRST_MONTH (2019-12) en na last_month (2021-06) wordt niets gehaald.
        assert sorted(bucket.downloads) == sorted(k for k in files if "-2021-05" in k)
        assert m["n_perps"] == 3 and m["n_mapped"] == 2 and m["n_objects"] == 2
        assert m["mapping"] == {"1000PEPEUSDT": ["PEPEUSDT", 1000.0], "BTCUSDT": ["BTCUSDT", 1.0]}
        assert json.loads((tmp_path / "spot.json").read_text(encoding="utf-8")) == m

    def test_mark_price_symbols_without_data_are_counted_apart(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        t0 = int(pd.Timestamp("2021-05-01", tz="UTC").value // 1_000_000)
        files = {f"{bv.MARK_KLINES.format(s='AAAUSDT')}AAAUSDT-1d-{m}.zip": _zip_bytes(_klines_csv(t0, [1.0]))
                 for m in ("2021-05", "2021-09")}
        bucket = _install(monkeypatch, files)
        m = bv.sync_mark(["AAAUSDT", "NODATAUSDT"], raw_dir=tmp_path / "raw", workers=2, last_month="2021-06",
                         manifest_path=tmp_path / "mark.json")
        assert bucket.downloads == [f"{bv.MARK_KLINES.format(s='AAAUSDT')}AAAUSDT-1d-2021-05.zip"]
        assert (m["n_symbols"], m["n_symbols_with_data"], m["n_objects"]) == (2, 1, 1)
        assert json.loads((tmp_path / "mark.json").read_text(encoding="utf-8")) == m


# --------------------------------------------------------------------------- #
# parsen en panelen
# --------------------------------------------------------------------------- #
class TestParse:
    def test_funding_without_a_header_gets_its_column_names(self, tmp_path: Path) -> None:
        path = _write(tmp_path / "f.zip", _zip_bytes(_funding_csv([(DAY, 0.0001), (2 * DAY, -0.0002)], header=False)))
        out = bv.parse_funding(path)
        assert list(out.columns) == ["calc_time", "last_funding_rate"]
        assert out["last_funding_rate"].tolist() == [0.0001, -0.0002]

    def test_funding_with_a_header_and_microsecond_timestamps_is_normalised_to_milliseconds(self, tmp_path: Path) -> None:
        ms = int(pd.Timestamp("2021-05-01", tz="UTC").value // 1_000_000)
        path = _write(tmp_path / "f.zip", _zip_bytes(_funding_csv([(ms * 1000, 0.0003), (ms + DAY, 0.0004)])))
        assert bv.parse_funding(path)["calc_time"].tolist() == [ms, ms + DAY]

    def test_a_zip_with_more_than_one_csv_is_refused(self, tmp_path: Path) -> None:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("a.csv", "1,2")
            zf.writestr("b.csv", "3,4")
        with pytest.raises(DataContractError, match="iets anders dan één CSV"):
            bv.parse_funding(_write(tmp_path / "x.zip", buf.getvalue()))

    def test_a_scientific_notation_data_row_is_not_mistaken_for_a_header(self, tmp_path: Path) -> None:
        path = _write(tmp_path / "f.zip", _zip_bytes(f"{DAY},8,2e-05\n{2 * DAY},8,1e-04"))
        assert bv.parse_funding(path)["last_funding_rate"].tolist() == [2e-05, 1e-04]


class TestBuildPanels:
    T0 = int(pd.Timestamp("2021-05-01", tz="UTC").value // 1_000_000)

    def _raw(self, tmp_path: Path) -> Path:
        raw = tmp_path / "raw"
        a = bv.KLINES.format(s="AAAUSDT")
        b = bv.KLINES.format(s="BBBUSDT")
        _write(raw / a / "AAAUSDT-1d-2021-05.zip", _zip_bytes(_klines_csv(self.T0, [100, 101, 102, 103, 104])))
        # BBB noteert pas op dag 2 en heeft een dubbele rij.
        _write(raw / b / "BBBUSDT-1d-2021-05.zip",
               _zip_bytes(_klines_csv(self.T0 + 2 * DAY, [10, 11, 12]) + "\n" + _klines_csv(self.T0 + 2 * DAY, [10])))
        f = bv.FUNDING.format(s="AAAUSDT")
        _write(raw / f / "AAAUSDT-fundingRate-2021-05.zip", _zip_bytes(_funding_csv([
            (self.T0, 0.001),                              # precies middernacht van dag 0: bar die op dag 1 sluit
            (self.T0 + 8 * 3600_000, 0.002),               # zelfde dag: opgeteld
            (self.T0 + DAY + 16 * 3600_000, 0.004),        # dag 1 om 16:00 -> bar die op dag 2 sluit
        ])))
        return raw

    def test_the_daily_panels_sit_on_the_close_time_and_funding_follows_the_strictly_after_rule(
        self, tmp_path: Path,
    ) -> None:
        panels = bv.build_panels(["BBBUSDT", "AAAUSDT"], raw_dir=self._raw(tmp_path), panel_dir=tmp_path / "panels")
        assert set(panels) == {"open", "high", "low", "close", "quote_volume", "funding"}
        close, funding = panels["close"], panels["funding"]
        assert list(close.columns) == ["AAAUSDT", "BBBUSDT"]
        assert close.index[0] == pd.Timestamp("2021-05-02", tz="UTC") and close.index.name == "asof_ts"
        assert close["AAAUSDT"].tolist() == [100, 101, 102, 103, 104]
        assert close["BBBUSDT"].isna().tolist() == [True, True, False, False, False]
        assert close["BBBUSDT"].dropna().tolist() == [10, 11, 12]                    # de dubbele rij is weg
        # Afrekeningen op dag 0 (00:00 en 08:00) tellen mee voor de bar die op dag 1 middernacht sluit.
        assert funding["AAAUSDT"].tolist() == pytest.approx([0.003, 0.004, 0.0, 0.0, 0.0])
        # Zonder funding-bestand: nul zodra genoteerd, NaN ervoor.
        assert funding["BBBUSDT"].isna().tolist() == [True, True, False, False, False]
        assert (funding["BBBUSDT"].dropna() == 0.0).all()
        for name in panels:
            assert (tmp_path / "panels" / f"{name}.parquet").is_file()

    def test_without_symbols_every_directory_under_klines_is_a_symbol(self, tmp_path: Path) -> None:
        raw = self._raw(tmp_path)
        (raw / bv.KLINES.format(s="EMPTYUSDT")).mkdir(parents=True)                  # map zonder zips: overgeslagen
        panels = bv.build_panels(raw_dir=raw, panel_dir=tmp_path / "panels")
        assert list(panels["close"].columns) == ["AAAUSDT", "BBBUSDT"]


# --------------------------------------------------------------------------- #
# vooruit-sample en CLI
# --------------------------------------------------------------------------- #
class TestBuildForward:
    @pytest.fixture()
    def calls(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[tuple]:
        calls: list[tuple] = []
        monkeypatch.setattr(bv, "FORWARD_MANIFEST_DIR", tmp_path / "manifests")
        monkeypatch.setattr(bv, "FORWARD_PANEL_ROOT", tmp_path / "forward")
        close = pd.DataFrame({"BTCUSDT": [1.0, 2.0, 3.0], "ETHUSDT": [1.0, 2.0, 3.0]})

        def sync(**kw):
            calls.append(("sync", kw["last_month"], Path(kw["manifest_path"]).name))
            return {}

        def build_panels(**kw):
            calls.append(("build_panels", Path(kw["panel_dir"]).name))
            return {"close": close}

        def sync_spot(perps, **kw):
            calls.append(("sync_spot", tuple(perps), kw["last_month"], Path(kw["manifest_path"]).name))
            return {"mapping": {"BTCUSDT": ["BTCUSDT", 1], "ETHUSDT": ["ETHUSDT", 1]}}

        def build_spot_panels(mapping, **kw):
            calls.append(("build_spot_panels", dict(mapping), Path(kw["panel_dir"]).name))
            return {}

        def sync_mark(perps, **kw):
            calls.append(("sync_mark", tuple(perps), Path(kw["manifest_path"]).name))
            return {}

        def build_mark_panels(perps, **kw):
            calls.append(("build_mark_panels", tuple(perps), Path(kw["panel_dir"]).name))
            return {"close": close.iloc[:2]}

        for fn in (sync, build_panels, sync_spot, build_spot_panels, sync_mark, build_mark_panels):
            monkeypatch.setattr(bv, fn.__name__, fn)
        return calls

    @pytest.mark.parametrize("bad", ["2026-09", "2026-08", "2027-1", "next month", "2027-03-01"])
    def test_only_a_month_after_the_frozen_sample_is_accepted(self, bad: str) -> None:
        with pytest.raises(DataContractError, match="begint na 2026-09"):
            bv.build_forward(bad)

    def test_perps_spot_and_mark_are_built_into_their_own_forward_tree(self, calls: list[tuple], tmp_path: Path) -> None:
        shapes = bv.build_forward("2027-03")
        assert shapes == {"perp": (3, 2), "mark": (2, 2)}
        assert [c[0] for c in calls] == ["sync", "build_panels", "sync_spot", "build_spot_panels",
                                         "sync_mark", "build_mark_panels"]
        assert calls[0] == ("sync", "2027-03", "binance_um_manifest.json")
        assert calls[3][1] == {"BTCUSDT": ("BTCUSDT", 1.0), "ETHUSDT": ("ETHUSDT", 1.0)}
        assert (tmp_path / "manifests").is_dir()


class TestMain:
    @pytest.fixture()
    def stubs(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[tuple]:
        calls: list[tuple] = []
        frame = pd.DataFrame({"AAAUSDT": [1.0, 2.0], "BBBUSDT": [3.0, 4.0]})
        monkeypatch.setattr(bv, "PANEL_DIR", tmp_path / "panels")
        (tmp_path / "panels").mkdir()
        frame.to_parquet(tmp_path / "panels" / "close.parquet")
        monkeypatch.setattr(bv, "SPOT_MANIFEST", tmp_path / "spot.json")
        (tmp_path / "spot.json").write_text(json.dumps({"mapping": {"AAAUSDT": ["AAAUSDT", 1]}}), encoding="utf-8")

        def rec(name, result):
            def fn(*a, **kw):
                calls.append((name, a, kw))
                return result
            monkeypatch.setattr(bv, name, fn)

        rec("sync", {"symbols": ["AAAUSDT"], "n_objects": 3})
        rec("build_panels", {"close": frame})
        rec("sync_spot", {"mapping": {"AAAUSDT": ["AAAUSDT", 1]}, "n_mapped": 1})
        rec("build_spot_panels", {"close": frame})
        rec("sync_mark", {"n_objects": 5})
        rec("build_mark_panels", {"close": frame})
        rec("build_forward", {"perp": (1, 2)})
        return calls

    def test_the_forward_command_takes_a_month(self, stubs: list[tuple], capsys: pytest.CaptureFixture[str]) -> None:
        bv.main(["forward", "2027-03"])
        assert stubs == [("build_forward", ("2027-03",), {})]
        assert "'perp': (1, 2)" in capsys.readouterr().out

    @pytest.mark.parametrize("argv", [[], ["sync", "extra"], ["nonsense"], ["forward"]])
    def test_bad_usage_is_refused(self, argv: list[str], stubs: list[tuple]) -> None:
        with pytest.raises(DataContractError, match="Gebruik"):
            bv.main(argv)
        assert stubs == []

    def test_sync_prints_the_manifest_without_the_symbol_list(self, stubs: list[tuple], capsys: pytest.CaptureFixture[str]) -> None:
        bv.main(["sync"])
        assert json.loads(capsys.readouterr().out) == {"n_objects": 3}

    def test_panels_prints_the_shapes(self, stubs: list[tuple], capsys: pytest.CaptureFixture[str]) -> None:
        bv.main(["panels"])
        assert capsys.readouterr().out.strip() == "{'close': (2, 2)}"

    def test_spot_sync_uses_the_perps_of_the_built_panel(self, stubs: list[tuple], capsys: pytest.CaptureFixture[str]) -> None:
        bv.main(["spot-sync"])
        assert stubs[0][0] == "sync_spot" and stubs[0][1] == (["AAAUSDT", "BBBUSDT"],)
        assert json.loads(capsys.readouterr().out) == {"n_mapped": 1}

    def test_spot_panels_rebuilds_from_the_spot_manifest(self, stubs: list[tuple], capsys: pytest.CaptureFixture[str]) -> None:
        bv.main(["spot-panels"])
        assert stubs == [("build_spot_panels", ({"AAAUSDT": ("AAAUSDT", 1.0)},), {})]
        assert capsys.readouterr().out.strip() == "{'close': (2, 2)}"

    def test_mark_sync_and_mark_panels_use_the_perps_of_the_built_panel(
        self, stubs: list[tuple], capsys: pytest.CaptureFixture[str],
    ) -> None:
        bv.main(["mark-sync"])
        assert json.loads(capsys.readouterr().out) == {"n_objects": 5}
        bv.main(["mark-panels"])
        assert capsys.readouterr().out.strip() == "{'close': (2, 2)}"
        assert [c[0] for c in stubs] == ["sync_mark", "build_mark_panels"]
        assert all(c[1] == (["AAAUSDT", "BBBUSDT"],) for c in stubs)

    @pytest.mark.filterwarnings(
        "ignore:.*found in sys.modules after import of package:RuntimeWarning")
    def test_the_module_runs_as_a_script(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "argv", ["binance_vision", "nonsense"])
        with pytest.raises(DataContractError, match="Gebruik"):
            runpy.run_module("tradebot.data.binance_vision", run_name="__main__")
