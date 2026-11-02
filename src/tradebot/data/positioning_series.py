"""Invoer voor een richtings-sleeve naast de carry: spotprijzen en institutionele positionering.

    python -I -m tradebot.data.positioning_series   # -> data/external/{spot_close,cftc_tff}.csv

Draait op GitHub Actions (`.github/workflows/research-run.yml`, taak `positioning-series`):
de ontwikkelomgeving bereikt Binance Vision en de CFTC niet.

* `spot_close.csv`: de dag-close van BTCUSDT en ETHUSDT op Binance spot (de maand-zips van
  Binance Vision, MD5 tegen de S3-ETag). De index is het SLUITmoment van de bar (de
  UTC-dag na `open_time`), zoals `asof_ts` in de dagreeksen van v5 en v9.
* `cftc_tff.csv`: het wekelijkse Traders-in-Financial-Futures-rapport van de CFTC (alleen
  futures) voor de crypto-contracten op de CME. Per rapportdatum (de dinsdag) de open
  interest en de long- en shortposities van asset managers, leveraged funds en dealers,
  in contracten. Dit is de enige openbare, gereguleerde meting van "instituties netto
  long": asset managers zijn pensioenfondsen, verzekeraars en fondsen; leveraged funds zijn
  hedgefondsen (die op de CME vaak short staan als been van een basistrade).

  Point-in-time: het rapport van dinsdag verschijnt op vrijdag 15:30 ET. Wie het gebruikt,
  laat het pas ingaan na `TFF_RELEASE_LAG_DAYS` (zie `systematic.flow_sleeve`). Bekend
  risico: tijdens een sluiting van de Amerikaanse overheid (2018-12/2019-01, 2025-10/11)
  verschenen rapporten weken later; de CFTC publiceert geen historische releasedatums,
  dus die weken lezen in de backtest iets te vroeg.

Alles stopt op `LAST_DAY` (2026-09-30), het einde van de onderzoekscyclus: het
vooruit-sample blijft ongelezen. Een bron die niet bereikbaar is, staat met de fout in het
manifest en ontbreekt; geen van beide is verplicht.
"""
from __future__ import annotations

import hashlib
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
from .external_series import LAST_DAY

__all__ = ["CFTC_TFF_URL", "SPOT_PAIRS", "TFF_COLUMNS", "fetch_all", "load_spot_close",
           "load_tff", "main", "parse_tff", "spot_close"]

ROOT = Path(__file__).resolve().parents[3]
CFTC_TFF_URL = "https://publicreporting.cftc.gov/resource/gpe5-46if.json"
OUT_DIR = Path("data/external")
SPOT_CSV = OUT_DIR / "spot_close.csv"
TFF_CSV = OUT_DIR / "cftc_tff.csv"
MANIFEST = Path("artefacts/data/positioning_manifest.json")
FIRST_DAY = pd.Timestamp("2019-09-01")
SPOT_PAIRS = {"BTC": "BTCUSDT", "ETH": "ETHUSDT"}
#: Socrata-veld -> kolom in `cftc_tff.csv`. Alle velden zijn verplicht.
TFF_COLUMNS = {
    "report_date_as_yyyy_mm_dd": "report_date",
    "market_and_exchange_names": "market",
    "cftc_contract_market_code": "code",
    "open_interest_all": "open_interest",
    "asset_mgr_positions_long": "asset_mgr_long",
    "asset_mgr_positions_short": "asset_mgr_short",
    "lev_money_positions_long": "lev_money_long",
    "lev_money_positions_short": "lev_money_short",
    "dealer_positions_long_all": "dealer_long",
    "dealer_positions_short_all": "dealer_short",
}
_NUMERIC = [c for c in TFF_COLUMNS.values() if c not in ("report_date", "market", "code")]
_CRYPTO = re.compile(r"\b(BITCOIN|ETHER)\b.*CHICAGO MERCANTILE", re.I)
TFF_WHERE = ("upper(market_and_exchange_names) like '%CHICAGO MERCANTILE%' AND "
             "(upper(market_and_exchange_names) like '%BITCOIN%' OR "
             "upper(market_and_exchange_names) like '%ETHER%')")


def _get(url: str, params: Mapping[str, str]) -> requests.Response:
    r = requests.get(url, params=dict(params), timeout=120)
    r.raise_for_status()
    return r


# --------------------------------------------------------------------------- #
# CFTC Traders in Financial Futures
# --------------------------------------------------------------------------- #
def parse_tff(rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Socrata-JSON -> één rij per (rapportdatum, contract), alleen CME-crypto, ≤ LAST_DAY."""
    require(bool(rows), "De CFTC gaf geen TFF-rijen.", DataContractError)
    missing = sorted(set(TFF_COLUMNS) - set(rows[0]))
    require(not missing, "Velden ontbreken in het TFF-rapport.", DataContractError,
            missing=missing, available=sorted(rows[0])[:60])
    df = pd.DataFrame([{TFF_COLUMNS[k]: r.get(k) for k in TFF_COLUMNS} for r in rows])
    df = df[df["market"].astype(str).str.contains(_CRYPTO)].copy()
    require(not df.empty, "Geen CME-cryptocontract in het TFF-rapport.", DataContractError)
    df["report_date"] = pd.to_datetime(df["report_date"]).dt.tz_localize(None).dt.normalize()
    df[_NUMERIC] = df[_NUMERIC].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    require(bool((df[_NUMERIC] >= 0).all().all()), "Een negatieve positie in het TFF-rapport.",
            DataContractError)
    df = df[(df["report_date"] >= FIRST_DAY) & (df["report_date"] <= LAST_DAY)]
    df = df.drop_duplicates(["report_date", "code"], keep="last")
    return df.sort_values(["code", "report_date"]).reset_index(drop=True)


def fetch_tff(get: Callable[..., Any] = _get) -> pd.DataFrame:
    params = {"$where": TFF_WHERE, "$limit": "50000",
              "$order": "report_date_as_yyyy_mm_dd"}
    return parse_tff(get(CFTC_TFF_URL, params).json())


# --------------------------------------------------------------------------- #
# Spot-close van Binance
# --------------------------------------------------------------------------- #
def spot_close(pair: str, files: Sequence[Path]) -> pd.Series:
    """Dag-zips -> close per sluitmoment (UTC-dag na `open_time`), ≤ LAST_DAY."""
    require(bool(files), "Geen spot-zips.", DataContractError, pair=pair)
    k = pd.concat([bv.parse_klines(p) for p in files])
    k = k.drop_duplicates("open_time").sort_values("open_time")
    idx = pd.to_datetime(k["open_time"].to_numpy() + bv.DAY_MS, unit="ms").normalize()
    s = pd.Series(k["close"].to_numpy(), index=idx, name=pair)
    s.index.name = "date"
    s = s[(s.index >= FIRST_DAY) & (s.index <= LAST_DAY)]
    require(not s.empty and bool((s > 0).all()), "Een lege of niet-positieve spotreeks.",
            DataContractError, pair=pair)
    return s


def _binance_spot(coin: str, raw_dir: Path = bv.RAW_DIR) -> pd.Series:
    pair = SPOT_PAIRS[coin]
    last = LAST_DAY.strftime("%Y-%m")
    objs = [o for o in bv.list_objects(bv.SPOT_KLINES.format(s=pair))
            if (m := re.search(r"-(\d{4}-\d{2})\.zip$", o.key))
            and bv.SPOT_FIRST_MONTH <= m.group(1) <= last]
    return spot_close(pair, [bv._fetch(o, raw_dir) for o in objs]).rename(coin)


# --------------------------------------------------------------------------- #
# Ophalen, manifest, laden
# --------------------------------------------------------------------------- #
def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(path: Path, root: Path, index: pd.Index) -> dict[str, Any]:
    return {"sha256": _sha256(root / path), "rows": len(index),
            "first": str(index.min().date()), "last": str(index.max().date())}


def fetch_all(root: Path = ROOT, *, get: Callable[..., Any] = _get,
              spot: Callable[[str], pd.Series] = _binance_spot) -> dict[str, Any]:
    """Haal beide bronnen op, schrijf de CSV's en het manifest."""
    (root / OUT_DIR).mkdir(parents=True, exist_ok=True)
    errors: dict[str, str] = {}
    files: dict[str, Any] = {}
    closes: list[pd.Series] = []
    for coin in SPOT_PAIRS:
        try:
            closes.append(spot(coin))
        except (requests.RequestException, DataContractError, KeyError, ValueError) as e:
            errors[f"spot_{coin}"] = f"{type(e).__name__}: {e}"[:300]
    if closes:
        frame = pd.concat(closes, axis=1).sort_index()
        frame.to_csv(root / SPOT_CSV, float_format="%.10g", lineterminator="\n")
        files[str(SPOT_CSV)] = _record(SPOT_CSV, root, frame.index) | {
            "columns": list(frame.columns)}
    try:
        tff = fetch_tff(get)
        tff.to_csv(root / TFF_CSV, index=False, float_format="%.10g", lineterminator="\n",
                   date_format="%Y-%m-%d")
        files[str(TFF_CSV)] = _record(TFF_CSV, root, pd.DatetimeIndex(tff["report_date"])) | {
            "markets": {str(c): str(m) for c, m in
                        tff.drop_duplicates("code", keep="last")[["code", "market"]]
                        .itertuples(index=False)}}
    except (requests.RequestException, DataContractError, KeyError, ValueError) as e:
        errors["cftc_tff"] = f"{type(e).__name__}: {e}"[:600]
    manifest = {"last_day": str(LAST_DAY.date()),
                "sources": {"spot": bv.BUCKET_URL, "spot_pairs": SPOT_PAIRS,
                            "cftc_tff": CFTC_TFF_URL, "cftc_where": TFF_WHERE},
                "files": files, "errors": errors}
    (root / MANIFEST).parent.mkdir(parents=True, exist_ok=True)
    (root / MANIFEST).write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n",
                                 encoding="utf-8")
    return manifest


def _verified(root: Path, path: Path) -> Path:
    manifest = json.loads((root / MANIFEST).read_text(encoding="utf-8"))
    rec = manifest["files"].get(str(path))
    require(rec is not None and (root / path).is_file()
            and rec["sha256"] == _sha256(root / path),
            "Een positioneringsreeks wijkt af van haar manifest.", DataContractError,
            path=str(path))
    return root / path


def load_spot_close(root: Path = ROOT) -> pd.DataFrame:
    """Close per sluitdag (kolommen BTC, ETH), gecontroleerd tegen het manifest."""
    return pd.read_csv(_verified(root, SPOT_CSV), index_col=0, parse_dates=True)


def load_tff(root: Path = ROOT) -> pd.DataFrame:
    """Het TFF-rapport, gecontroleerd tegen het manifest."""
    df = pd.read_csv(_verified(root, TFF_CSV), parse_dates=["report_date"],
                     dtype={"code": str})
    return df


def main(argv: Sequence[str]) -> dict[str, Any]:
    require(len(argv) == 0, "Gebruik: python -I -m tradebot.data.positioning_series",
            DataContractError)
    out = fetch_all(ROOT)
    print(json.dumps({"files": out["files"], "errors": out["errors"]}, indent=1))
    return out


if __name__ == "__main__":
    main(sys.argv[1:])
