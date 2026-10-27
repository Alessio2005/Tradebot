"""Externe reeksen voor het carryboek op Kraken: de T-bill-rente en de funding per beurs.

    python -I -m tradebot.data.external_series     # -> data/external/*.csv + manifest

Draait op GitHub Actions (`.github/workflows/research-run.yml`, taak `external-series`): de
ontwikkelomgeving bereikt FRED, Kraken en Binance Vision niet.

* `dtb3.csv`: de 3-maands T-bill (FRED DTB3, discontobasis, als fractie per jaar). Wat
  onbelegde dollars in een geldmarktfonds ongeveer opbrengen: de kasrente van v9.
* `estr.csv`: de euro short-term rate (ECB, €STR, vanaf oktober 2019). Voor een belegger in
  euro's die de dollar afdekt, is DTB3 − €STR wat de afdekking ongeveer kost.
* `funding_daily.csv`: de dagsom van de funding per beurs en munt (BTC, ETH), als fractie
  van de notional. Kraken (PF_XBTUSD, PF_ETHUSD, `relativeFundingRate`) naast Binance
  (BTCUSDT, ETHUSDT, de maand-zips van Binance Vision): hoeveel scheelt de carry tussen
  de beurs van de backtest en de beurs waar het boek kan draaien.

Alles stopt op `LAST_DAY` (2026-09-30), het einde van de onderzoekscyclus: het
vooruit-sample (vanaf 2026-10-01) blijft ongelezen. Een fundingbron die niet bereikbaar is,
staat met de fout in het manifest en ontbreekt in de CSV; de T-bill is verplicht.
"""
from __future__ import annotations

import hashlib
import io
import json
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from ..utils.failfast import DataContractError, require
from . import binance_vision as bv

__all__ = ["BINANCE_PERPS", "KRAKEN_PERPS", "LAST_DAY", "binance_daily_funding", "fetch_all",
           "fetch_dtb3", "kraken_daily_funding", "load_dtb3", "load_estr", "load_funding",
           "main", "parse_dtb3", "parse_estr", "parse_kraken_funding"]

ROOT = Path(__file__).resolve().parents[3]
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
ECB_ESTR_URL = "https://data-api.ecb.europa.eu/service/data/EST/B.EU000A2X2A25.WT"
KRAKEN_FUNDING_URL = "https://futures.kraken.com/derivatives/api/v4/historicalfundingrates"
OUT_DIR = Path("data/external")
DTB3_CSV = OUT_DIR / "dtb3.csv"
ESTR_CSV = OUT_DIR / "estr.csv"
FUNDING_CSV = OUT_DIR / "funding_daily.csv"
MANIFEST = Path("artefacts/data/external_manifest.json")
LAST_DAY = pd.Timestamp("2026-09-30")
FIRST_DAY = pd.Timestamp("2019-09-01")
KRAKEN_PERPS = {"BTC": "PF_XBTUSD", "ETH": "PF_ETHUSD"}
BINANCE_PERPS = {"BTC": "BTCUSDT", "ETH": "ETHUSDT"}


def _get(url: str, params: Mapping[str, str]) -> requests.Response:
    r = requests.get(url, params=dict(params), timeout=60)
    r.raise_for_status()
    return r


def _window(s: pd.Series, last_day: pd.Timestamp = LAST_DAY) -> pd.Series:
    s = s[(s.index >= FIRST_DAY) & (s.index <= last_day)].sort_index()
    require(not s.empty, "Een reeks zonder waarnemingen in de onderzoekscyclus.",
            DataContractError, name=s.name)
    return s


def parse_dtb3(text: str, *, last_day: pd.Timestamp = LAST_DAY) -> pd.Series:
    """FRED-CSV (`observation_date,DTB3` of `DATE,DTB3`, `.` = geen notering) -> fractie.

    `last_day` is voor het onderzoek het einde van de cyclus; alleen het Kraken-boek
    (`carry.u1`, shadow en live) leest verder."""
    df = pd.read_csv(io.StringIO(text), na_values=["."])
    require(df.shape[1] == 2 and df.columns[1] == "DTB3", "Geen DTB3-CSV van FRED.",
            DataContractError, columns=list(df.columns))
    s = pd.Series(df.iloc[:, 1].to_numpy() / 100.0, index=pd.to_datetime(df.iloc[:, 0]),
                  name="dtb3").dropna()
    s.index.name = "date"
    s = _window(s, pd.Timestamp(last_day))
    require(bool(((s > -0.01) & (s < 0.25)).all()), "Een T-bill-rente buiten -1 .. 25 %.",
            DataContractError)
    return s


def parse_estr(text: str) -> pd.Series:
    """ECB-CSV (`format=csvdata`: TIME_PERIOD, OBS_VALUE in procent) -> fractie per jaar."""
    df = pd.read_csv(io.StringIO(text))
    require({"TIME_PERIOD", "OBS_VALUE"} <= set(df.columns), "Geen €STR-CSV van de ECB.",
            DataContractError, columns=list(df.columns))
    s = pd.Series(df["OBS_VALUE"].to_numpy(dtype=float) / 100.0,
                  index=pd.to_datetime(df["TIME_PERIOD"]), name="estr").dropna()
    s.index.name = "date"
    s = _window(s)
    require(bool(((s > -0.02) & (s < 0.25)).all()), "Een €STR buiten -2 .. 25 %.",
            DataContractError)
    return s


def parse_kraken_funding(payload: Mapping[str, Any], name: str) -> pd.Series:
    """`historicalfundingrates` -> de dagsom van `relativeFundingRate` (UTC-dag)."""
    require(payload.get("result") == "success" and bool(payload.get("rates")),
            "Kraken gaf geen fundinghistorie.", DataContractError, name=name,
            result=payload.get("result"))
    ts = pd.to_datetime([r["timestamp"] for r in payload["rates"]], utc=True)
    rate = pd.Series([float(r["relativeFundingRate"]) for r in payload["rates"]],
                     index=ts.tz_convert(None).normalize(), name=name)
    s = rate.groupby(level=0).sum()
    s.index.name = "date"
    return _window(s)


def kraken_daily_funding(coin: str, get: Callable[..., Any] = _get) -> pd.Series:
    sym = KRAKEN_PERPS[coin]
    return parse_kraken_funding(get(KRAKEN_FUNDING_URL, {"symbol": sym}).json(),
                                f"kraken_{coin}")


def binance_daily_funding(coin: str, raw_dir: Path = bv.RAW_DIR) -> pd.Series:
    """De maand-zips van Binance Vision tot en met september 2026 -> dagsom."""
    sym = BINANCE_PERPS[coin]
    last = LAST_DAY.strftime("%Y-%m")
    objs = [o for o in bv.list_objects(bv.FUNDING.format(s=sym))
            if (m := re.search(r"-(\d{4}-\d{2})\.zip$", o.key)) and m.group(1) <= last]
    require(bool(objs), "Binance Vision gaf geen funding-zips.", DataContractError, symbol=sym)
    f = pd.concat([bv.parse_funding(bv._fetch(o, raw_dir)) for o in objs])
    day = pd.to_datetime(f["calc_time"].to_numpy(), unit="ms").normalize()
    s = pd.Series(f["last_funding_rate"].to_numpy(), index=day, name=f"binance_{coin}")
    s = s.groupby(level=0).sum()
    s.index.name = "date"
    return _window(s)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(path: Path, root: Path, frame: pd.DataFrame | pd.Series) -> dict[str, Any]:
    return {"sha256": _sha256(root / path), "rows": len(frame),
            "first": str(frame.index.min().date()), "last": str(frame.index.max().date())}


def fetch_all(root: Path = ROOT, *, get: Callable[..., Any] = _get,
              binance: Callable[[str], pd.Series] = binance_daily_funding) -> dict[str, Any]:
    """Haal alles op, schrijf de CSV's en het manifest. De T-bill moet lukken."""
    (root / OUT_DIR).mkdir(parents=True, exist_ok=True)
    dtb3 = parse_dtb3(get(FRED_URL, {"id": "DTB3"}).text)
    dtb3.to_frame().to_csv(root / DTB3_CSV, float_format="%.6g", lineterminator="\n")
    errors: dict[str, str] = {}
    files = {str(DTB3_CSV): _record(DTB3_CSV, root, dtb3)}
    try:
        estr = parse_estr(get(ECB_ESTR_URL, {"format": "csvdata"}).text)
        estr.to_frame().to_csv(root / ESTR_CSV, float_format="%.6g", lineterminator="\n")
        files[str(ESTR_CSV)] = _record(ESTR_CSV, root, estr)
    except (requests.RequestException, DataContractError, ValueError) as e:
        errors["estr"] = f"{type(e).__name__}: {e}"[:300]
    cols: list[pd.Series] = []
    for coin in KRAKEN_PERPS:
        for name, pull in ((f"kraken_{coin}", lambda c: kraken_daily_funding(c, get)),
                           (f"binance_{coin}", binance)):
            try:
                cols.append(pull(coin))
            except (requests.RequestException, DataContractError, KeyError, ValueError) as e:
                errors[name] = f"{type(e).__name__}: {e}"[:300]
    manifest: dict[str, Any] = {
        "last_day": str(LAST_DAY.date()),
        "sources": {"dtb3": f"{FRED_URL}?id=DTB3", "estr": ECB_ESTR_URL,
                    "kraken": KRAKEN_FUNDING_URL, "kraken_symbols": KRAKEN_PERPS,
                    "binance": bv.BUCKET_URL, "binance_symbols": BINANCE_PERPS},
        "files": files, "errors": errors}
    if cols:
        funding = pd.concat(cols, axis=1).sort_index()
        funding.to_csv(root / FUNDING_CSV, float_format="%.10g", lineterminator="\n")
        manifest["files"][str(FUNDING_CSV)] = _record(FUNDING_CSV, root, funding) | {
            "columns": list(funding.columns)}
    (root / MANIFEST).parent.mkdir(parents=True, exist_ok=True)
    (root / MANIFEST).write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n",
                                 encoding="utf-8")
    return manifest


def _verified(root: Path, path: Path) -> Path:
    manifest = json.loads((root / MANIFEST).read_text(encoding="utf-8"))
    rec = manifest["files"].get(str(path))
    require(rec is not None and (root / path).is_file()
            and rec["sha256"] == _sha256(root / path),
            "Een externe reeks wijkt af van haar manifest.", DataContractError, path=str(path))
    return root / path


def load_dtb3(root: Path = ROOT) -> pd.Series:
    """De T-bill als fractie per jaar, per kalenderdag (weekend en feestdag: vorige noteringsdag)."""
    return _daily(pd.read_csv(_verified(root, DTB3_CSV), index_col=0, parse_dates=True)["dtb3"])


def _daily(s: pd.Series, last_day: pd.Timestamp = LAST_DAY) -> pd.Series:
    days = pd.date_range(s.index.min(), last_day, freq="D", name="date")
    return s.reindex(days).ffill()


def fetch_dtb3(last_day: pd.Timestamp, get: Callable[..., Any] = _get) -> pd.Series:
    """De T-bill per kalenderdag tot en met `last_day`, live van FRED (niet gecommit).

    Voor het Kraken-boek, dat elke dag tot gisteren rekent; de laatste notering geldt tot
    er een nieuwe is."""
    last = pd.Timestamp(last_day).normalize()
    return _daily(parse_dtb3(get(FRED_URL, {"id": "DTB3"}).text, last_day=last), last)


def load_estr(root: Path = ROOT) -> pd.Series:
    """€STR als fractie per jaar, per kalenderdag (zoals `load_dtb3`)."""
    return _daily(pd.read_csv(_verified(root, ESTR_CSV), index_col=0, parse_dates=True)["estr"])


def load_funding(root: Path = ROOT) -> pd.DataFrame:
    return pd.read_csv(_verified(root, FUNDING_CSV), index_col=0, parse_dates=True)


def main(argv: Sequence[str]) -> dict[str, Any]:
    require(len(argv) == 0, "Gebruik: python -I -m tradebot.data.external_series",
            DataContractError)
    out = fetch_all(ROOT)
    print(json.dumps({"files": out["files"], "errors": out["errors"]}, indent=1))
    return out


if __name__ == "__main__":
    main(sys.argv[1:])
