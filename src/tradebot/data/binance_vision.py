"""Binance USDⓈ-M-geschiedenis uit de publieke `data.binance.vision`-bucket.

WAAROM DEZE BRON
================
De gecertificeerde store bevat zes Bybit-perps: overlevers van nu, met een gemiddelde
onderlinge correlatie van 0,73 (≈ 1,3 onafhankelijke weddenschappen). Breedte is de
bottleneck van elk onderzoek op die data. Binance publiceert de volledige historie van
ELKE USDT-M-perp die ooit heeft gehandeld -- ook gedeliste (LUNA, FTT, SRM, ...) -- als
maand-zips. Daarmee is een survivorship-vrij universum te bouwen.

De bucket is dezelfde als achter `data.binance.vision`; hij wordt hier via het
path-style S3-endpoint gelezen. Elke zip wordt tegen zijn S3-ETag (MD5) geverifieerd;
een afwijkende hash crasht.

WAT DIT MODULE NIET DOET
========================
Het schrijft niets in de gecertificeerde PIT-store en verandert `data_hashes.json` niet.
Het bouwt een APART, reproduceerbaar paneel met een eigen manifest
(`artefacts/data/binance_um_manifest.json`).

    python -I -m tradebot.data.binance_vision sync     # download + verifieer (idempotent)
    python -I -m tradebot.data.binance_vision panels   # bouw de dagpanelen
"""
from __future__ import annotations

import hashlib
import io
import json
import re
import sys
import zipfile
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from ..utils.failfast import DataContractError, require

__all__ = ["BUCKET_URL", "S3Object", "build_panels", "build_spot_panels", "list_objects",
           "list_spot_symbols", "list_symbols", "spot_pair_for", "sync", "sync_spot"]

BUCKET_URL = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = ROOT / "data" / "binance_vision" / "raw"
PANEL_DIR = ROOT / "data" / "binance_vision" / "panels"
MANIFEST = ROOT / "artefacts" / "data" / "binance_um_manifest.json"

KLINES = "data/futures/um/monthly/klines/{s}/1d/"
FUNDING = "data/futures/um/monthly/fundingRate/{s}/"
KLINE_COLUMNS = ("open_time", "open", "high", "low", "close", "volume", "close_time",
                 "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume",
                 "ignore")
DAY_MS = 86_400_000
_SESSION = requests.Session()


@dataclass(frozen=True)
class S3Object:
    key: str
    etag: str
    size: int


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, max=30), reraise=True)
def _get(params: dict[str, str] | None = None, key: str = "") -> requests.Response:
    url = f"{BUCKET_URL}/{key}" if key else BUCKET_URL
    r = _SESSION.get(url, params=params, timeout=60)
    r.raise_for_status()
    return r


def list_common_prefixes(prefix: str) -> list[str]:
    out: list[str] = []
    marker = ""
    while True:
        text = _get({"delimiter": "/", "prefix": prefix, "marker": marker}).text
        out += re.findall(r"<CommonPrefixes><Prefix>([^<]+)</Prefix></CommonPrefixes>", text)
        if "<IsTruncated>true</IsTruncated>" not in text:
            return out
        marker = re.search(r"<NextMarker>([^<]+)</NextMarker>", text).group(1)  # type: ignore[union-attr]


def list_objects(prefix: str) -> list[S3Object]:
    out: list[S3Object] = []
    marker = ""
    while True:
        text = _get({"prefix": prefix, "marker": marker}).text
        for body in re.findall(r"<Contents>(.*?)</Contents>", text, flags=re.S):
            key = re.search(r"<Key>([^<]+)</Key>", body).group(1)  # type: ignore[union-attr]
            etag = re.search(r"<ETag>&quot;([^&]+)&quot;</ETag>", body).group(1)  # type: ignore[union-attr]
            size = int(re.search(r"<Size>(\d+)</Size>", body).group(1))  # type: ignore[union-attr]
            out.append(S3Object(key=key, etag=etag, size=size))
        if "<IsTruncated>true</IsTruncated>" not in text:
            return out
        marker = out[-1].key


def list_symbols() -> list[str]:
    """Elke USDT-M-perp die ooit dagbars had: eindigt op USDT, geen leveringscontract."""
    names = [p.rstrip("/").split("/")[-1]
             for p in list_common_prefixes("data/futures/um/monthly/klines/")]
    return sorted(s for s in names if s.endswith("USDT") and "_" not in s)


def _verified(path: Path, obj: S3Object) -> bool:
    if not path.is_file() or path.stat().st_size != obj.size:
        return False
    if "-" in obj.etag:  # multipart-ETag is geen MD5; grootte is dan de toets
        return True
    return hashlib.md5(path.read_bytes()).hexdigest() == obj.etag


def _fetch(obj: S3Object, raw_dir: Path) -> Path:
    path = raw_dir / obj.key
    if _verified(path, obj):
        return path
    data = _get(key=obj.key).content
    if "-" not in obj.etag:
        got = hashlib.md5(data).hexdigest()
        require(got == obj.etag, "Download wijkt af van de S3-ETag.", DataContractError,
                key=obj.key, etag=obj.etag, md5=got)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    tmp.write_bytes(data)
    tmp.replace(path)
    return path


def _objects_for(symbol: str) -> list[S3Object]:
    objs = []
    for tmpl in (KLINES, FUNDING):
        objs += [o for o in list_objects(tmpl.format(s=symbol)) if o.key.endswith(".zip")]
    return objs


def sync(symbols: Sequence[str] | None = None, *, raw_dir: Path = RAW_DIR,
         workers: int = 32, last_month: str = "2026-09") -> dict:
    """Download (idempotent) en verifieer klines + funding tot en met `last_month`."""
    syms = list(symbols) if symbols is not None else list_symbols()
    with ThreadPoolExecutor(workers) as pool:
        listings = list(pool.map(_objects_for, syms))
    objs = [o for lst in listings for o in lst
            if re.search(r"-(\d{4}-\d{2})\.zip$", o.key).group(1) <= last_month]  # type: ignore[union-attr]
    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(lambda o: _fetch(o, raw_dir), objs))
    digest = hashlib.sha256("\n".join(f"{o.key} {o.etag}" for o in sorted(
        objs, key=lambda o: o.key)).encode()).hexdigest()
    manifest = {"source": BUCKET_URL, "last_month": last_month, "n_symbols": len(syms),
                "n_objects": len(objs), "objects_sha256": digest,
                "symbols": syms, "bytes": int(sum(o.size for o in objs))}
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


# --------------------------------------------------------------------------- #
# Parsen -- de zips zijn onvertrouwde invoer: alleen CSV-getallen worden gelezen.
# --------------------------------------------------------------------------- #
def _read_zip_csv(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as zf:
        names = [n for n in zf.namelist() if n.endswith(".csv")]
        require(len(names) == 1, "Een zip met iets anders dan één CSV.", DataContractError,
                path=str(path), names=names)
        raw = zf.read(names[0])
    first = raw.split(b"\n", 1)[0]
    # Een kopregel BEGINT met een naam; een datarij met wetenschappelijke notatie (2e-05)
    # bevat ook een letter en mag niet als kop worden weggegooid.
    header = 0 if re.match(rb"\s*[A-Za-z_]", first) else None
    return pd.read_csv(io.BytesIO(raw), header=header)


def parse_klines(path: Path) -> pd.DataFrame:
    df = _read_zip_csv(path)
    if df.columns[0] != "open_time":
        df.columns = list(KLINE_COLUMNS)[: df.shape[1]]
    df = df[["open_time", "open", "high", "low", "close", "quote_volume"]].astype("float64")
    # Microseconde-tijdstempels (nieuwere bestanden) terug naar milliseconden.
    df["open_time"] = np.where(df["open_time"] > 1e14, df["open_time"] // 1000, df["open_time"])
    return df


def parse_funding(path: Path) -> pd.DataFrame:
    df = _read_zip_csv(path)
    if "calc_time" not in df.columns:
        df.columns = ["calc_time", "funding_interval_hours", "last_funding_rate"][: df.shape[1]]
    out = df[["calc_time", "last_funding_rate"]].astype("float64")
    out["calc_time"] = np.where(out["calc_time"] > 1e14, out["calc_time"] // 1000, out["calc_time"])
    return out


def _symbol_frames(symbol: str, raw_dir: Path) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    kdir = raw_dir / KLINES.format(s=symbol)
    fdir = raw_dir / FUNDING.format(s=symbol)
    kfiles = sorted(kdir.glob("*.zip")) if kdir.is_dir() else []
    ffiles = sorted(fdir.glob("*.zip")) if fdir.is_dir() else []
    k = pd.concat([parse_klines(p) for p in kfiles]) if kfiles else None
    f = pd.concat([parse_funding(p) for p in ffiles]) if ffiles else None
    return k, f


def build_panels(symbols: Iterable[str] | None = None, *, raw_dir: Path = RAW_DIR,
                 panel_dir: Path = PANEL_DIR) -> dict[str, pd.DataFrame]:
    """Dagpanelen op de SLUITtijd (open + 1 dag), zoals de gecertificeerde store.

    funding[T] = som van de afrekeningen met calc_time in [T − 1 dag, T): wat een
    positie die over de bar met sluiting T werd gehouden, betaalde (long) of ontving.
    """
    syms = sorted(symbols) if symbols is not None else sorted(
        p.name for p in (raw_dir / "data/futures/um/monthly/klines").iterdir())
    fields: dict[str, dict[str, pd.Series]] = {
        k: {} for k in ("open", "high", "low", "close", "quote_volume", "funding")}
    for s in syms:
        k, f = _symbol_frames(s, raw_dir)
        if k is None or k.empty:
            continue
        k = k.drop_duplicates("open_time").sort_values("open_time")
        idx = pd.to_datetime(k["open_time"].to_numpy() + DAY_MS, unit="ms", utc=True)
        for col in ("open", "high", "low", "close", "quote_volume"):
            fields[col][s] = pd.Series(k[col].to_numpy(), index=idx)
        if f is not None and not f.empty:
            f = f.drop_duplicates("calc_time").sort_values("calc_time")
            # Een afrekening op tijdstip c hoort bij de bar die sluit op de eerste
            # middernacht STRIKT na c.
            bar = (np.floor(f["calc_time"].to_numpy() / DAY_MS) + 1) * DAY_MS
            fields["funding"][s] = pd.Series(
                f["last_funding_rate"].to_numpy(),
                index=pd.to_datetime(bar, unit="ms", utc=True)).groupby(level=0).sum()
    grid = pd.date_range(min(s.index.min() for s in fields["close"].values()),
                         max(s.index.max() for s in fields["close"].values()),
                         freq="D", tz="UTC", name="asof_ts")
    panels: dict[str, pd.DataFrame] = {}
    for name, cols in fields.items():
        frame = pd.DataFrame({s: v.reindex(grid) for s, v in cols.items()}, index=grid)
        panels[name] = frame.reindex(columns=sorted(fields["close"]))
    listed = panels["close"].notna()
    panels["funding"] = panels["funding"].where(listed, np.nan).fillna(0.0).where(listed)
    panel_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in panels.items():
        frame.astype("float64").to_parquet(panel_dir / f"{name}.parquet", compression="zstd")
    return panels


# --------------------------------------------------------------------------- #
# Spot: het hedgebeen van de basis-carry (v5). Kolommen heten naar de PERP.
# --------------------------------------------------------------------------- #
SPOT_KLINES = "data/spot/monthly/klines/{s}/1d/"
SPOT_PANEL_DIR = ROOT / "data" / "binance_vision" / "spot_panels"
SPOT_MANIFEST = ROOT / "artefacts" / "data" / "binance_spot_manifest.json"
SPOT_FIRST_MONTH = "2019-12"
_SCALED = re.compile(r"^(1000000|1000|1M)([A-Z0-9]+USDT)$")
_MULTIPLIER = {"1000000": 1e6, "1000": 1e3, "1M": 1e6}


def list_spot_symbols() -> list[str]:
    names = [p.rstrip("/").split("/")[-1]
             for p in list_common_prefixes("data/spot/monthly/klines/")]
    return sorted(s for s in names if s.endswith("USDT"))


def spot_pair_for(perp: str, spot: Iterable[str]) -> tuple[str, float] | None:
    """De spotmunt achter een perp en de prijsfactor: 1000PEPEUSDT -> (PEPEUSDT, 1000).

    Bestaat de perpnaam zelf op spot (1000SATSUSDT), dan is dat het paar, factor 1."""
    names = set(spot)
    if perp in names:
        return perp, 1.0
    m = _SCALED.match(perp)
    if m and m.group(2) in names:
        return m.group(2), _MULTIPLIER[m.group(1)]
    return None


def sync_spot(perps: Sequence[str], *, raw_dir: Path = RAW_DIR, workers: int = 32,
              last_month: str = "2026-09") -> dict:
    """Download (idempotent) en verifieer de spot-dagklines achter `perps`."""
    spot = list_spot_symbols()
    mapping = {p: pair for p in perps if (pair := spot_pair_for(p, spot)) is not None}
    pairs = sorted({s for s, _ in mapping.values()})

    def objects(s: str) -> list[S3Object]:
        return [o for o in list_objects(SPOT_KLINES.format(s=s)) if o.key.endswith(".zip")]

    with ThreadPoolExecutor(workers) as pool:
        listings = list(pool.map(objects, pairs))
    month = re.compile(r"-(\d{4}-\d{2})\.zip$")
    objs = [o for lst in listings for o in lst
            if SPOT_FIRST_MONTH <= month.search(o.key).group(1) <= last_month]  # type: ignore[union-attr]
    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(lambda o: _fetch(o, raw_dir), objs))
    digest = hashlib.sha256("\n".join(f"{o.key} {o.etag}" for o in sorted(
        objs, key=lambda o: o.key)).encode()).hexdigest()
    manifest = {"source": BUCKET_URL, "first_month": SPOT_FIRST_MONTH, "last_month": last_month,
                "n_perps": len(perps), "n_mapped": len(mapping), "n_objects": len(objs),
                "objects_sha256": digest, "bytes": int(sum(o.size for o in objs)),
                "mapping": {p: [s, f] for p, (s, f) in sorted(mapping.items())}}
    SPOT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    SPOT_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def build_spot_panels(mapping: dict[str, tuple[str, float]], *, raw_dir: Path = RAW_DIR,
                      panel_dir: Path = SPOT_PANEL_DIR) -> dict[str, pd.DataFrame]:
    """Spot-dagpanelen op de SLUITtijd, kolom = perpnaam, prijs in perp-eenheden (× factor)."""
    fields: dict[str, dict[str, pd.Series]] = {k: {} for k in ("close", "high", "low",
                                                               "quote_volume")}
    for perp, (spot, factor) in sorted(mapping.items()):
        kdir = raw_dir / SPOT_KLINES.format(s=spot)
        files = sorted(kdir.glob("*.zip")) if kdir.is_dir() else []
        if not files:
            continue
        k = pd.concat([parse_klines(p) for p in files])
        k = k.drop_duplicates("open_time").sort_values("open_time")
        idx = pd.to_datetime(k["open_time"].to_numpy() + DAY_MS, unit="ms", utc=True)
        for col in ("close", "high", "low"):
            fields[col][perp] = pd.Series(k[col].to_numpy() * factor, index=idx)
        fields["quote_volume"][perp] = pd.Series(k["quote_volume"].to_numpy(), index=idx)
    require(bool(fields["close"]), "Geen spotdata gevonden.", DataContractError)
    grid = pd.date_range(min(s.index.min() for s in fields["close"].values()),
                         max(s.index.max() for s in fields["close"].values()),
                         freq="D", tz="UTC", name="asof_ts")
    panels = {name: pd.DataFrame({s: v.reindex(grid) for s, v in cols.items()}, index=grid)
              .reindex(columns=sorted(fields["close"])) for name, cols in fields.items()}
    panel_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in panels.items():
        frame.astype("float64").to_parquet(panel_dir / f"{name}.parquet", compression="zstd")
    return panels


def main(argv: Sequence[str]) -> None:
    cmds = ("sync", "panels", "spot-sync", "spot-panels")
    require(len(argv) == 1 and argv[0] in cmds,
            "Gebruik: python -I -m tradebot.data.binance_vision "
            "{sync|panels|spot-sync|spot-panels}", DataContractError)
    if argv[0] == "sync":
        m = sync()
        print(json.dumps({k: v for k, v in m.items() if k != "symbols"}, indent=2))
    elif argv[0] == "panels":
        p = build_panels()
        print({k: v.shape for k, v in p.items()})
    elif argv[0] == "spot-sync":
        perps = sorted(pd.read_parquet(PANEL_DIR / "close.parquet").columns)
        m = sync_spot(perps)
        print(json.dumps({k: v for k, v in m.items() if k != "mapping"}, indent=2))
    else:
        manifest = json.loads(SPOT_MANIFEST.read_text(encoding="utf-8"))
        p = build_spot_panels({k: (s, float(f)) for k, (s, f) in manifest["mapping"].items()})
        print({k: v.shape for k, v in p.items()})


if __name__ == "__main__":
    main(sys.argv[1:])
