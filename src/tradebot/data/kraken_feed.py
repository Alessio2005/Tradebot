"""Publieke Kraken-data voor het majorsboek (U1): spot, perp en funding van BTC en ETH.

    python -I -m tradebot.data.kraken_feed OUT_DIR     # -> OUT_DIR/*.csv + manifest.json

Het boek dat op Kraken EU kan draaien (v9-U1, de regels van v5-H3) rekent hier op de
prijzen en de funding van KRAKEN, niet op die van Binance. Drie publieke bronnen, geen
sleutel:

* spot: `api.kraken.com/0/public/OHLC` (dagbars, de laatste 720 dagen), XBTUSD en ETHUSD;
* perp: `futures.kraken.com/api/charts/v1/trade/{PF_XBTUSD,PF_ETHUSD}/1d`;
* funding: `futures.kraken.com/derivatives/api/v4/historicalfundingrates` (per uur,
  `relativeFundingRate`), als dagsom.

DE CONVENTIES VAN DE PANELEN (gelijk aan `binance_vision`, zodat `harvest` ongewijzigd
draait):
* de index is de SLUITTIJD van de dagbar (start + 1 dag), UTC, naam `asof_ts`;
* `funding[t]` is de som van de afrekeningen in `[t − 1 dag, t)`;
* alleen afgesloten bars: een bar die na `now` sluit, bestaat nog niet;
* de kolommen heten zoals de majors in `conf/model/robust_book_v5.yaml` (`BTCUSDT`,
  `ETHUSDT`): dat zijn boeksymbolen, de adapter vertaalt ze naar Kraken.

NIET GETOETST TEGEN DE ECHTE API: de ontwikkelomgeving bereikt Kraken niet. De parsers
weigeren alles wat niet de verwachte vorm heeft, en de dagomzet moet in een plausibel bereik
liggen (een volume in contracten dat als dollars wordt gelezen, valt daar buiten).
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from ..schemas.robust_book_v2 import RobustBookV2Config
from ..systematic.breadth import breadth_from_frames
from ..systematic.harvest import BasisMarket, basis_market
from ..utils.failfast import DataContractError, require
from . import external_series as ex

__all__ = ["COINS", "KrakenPanels", "fetch_panels", "kraken_market", "load_panels", "main",
           "parse_funding", "parse_perp_candles", "parse_spot_ohlc", "write_panels"]

SPOT_OHLC_URL = "https://api.kraken.com/0/public/OHLC"
PERP_CHART_URL = "https://futures.kraken.com/api/charts/v1/trade/{symbol}/1d"
FUNDING_URL = ex.KRAKEN_FUNDING_URL
#: Boeksymbool -> (spotpaar, perp) op Kraken.
COINS: dict[str, tuple[str, str]] = {"BTCUSDT": ("XBTUSD", "PF_XBTUSD"),
                                     "ETHUSDT": ("ETHUSD", "PF_ETHUSD")}
DAY = pd.Timedelta(days=1)
#: Een plausibele dagomzet in USD voor BTC/ETH op Kraken (spot en perp): daarbuiten is de
#: eenheid fout (contracten als dollars, of andersom), geen marktbeweging.
ADV_BOUNDS_USD = (1e5, 1e12)
PERP_FIELDS = ("close", "high", "low", "quote_volume", "funding")
SPOT_FIELDS = ("close", "quote_volume")

Get = Callable[[str, Mapping[str, str]], requests.Response]


def _get(url: str, params: Mapping[str, str]) -> requests.Response:
    r = requests.get(url, params=dict(params), timeout=60)
    r.raise_for_status()
    return r


def _frame(rows: list[tuple[pd.Timestamp, float, float, float, float]]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["asof_ts", "close", "high", "low", "quote_volume"])
    df = df.drop_duplicates("asof_ts", keep="last").set_index("asof_ts").sort_index()
    require(bool((df[["close", "high", "low"]] > 0.0).all().all())
            and bool((df["quote_volume"] >= 0.0).all()), "Een niet-positieve koers of een "
            "negatieve omzet.", DataContractError)
    return df


def parse_spot_ohlc(payload: Mapping[str, Any]) -> pd.DataFrame:
    """`0/public/OHLC`: `[tijd, open, high, low, close, vwap, volume, count]`, tijd = start."""
    require(not payload.get("error") and isinstance(payload.get("result"), Mapping),
            "Kraken spot gaf geen OHLC.", DataContractError, error=payload.get("error"))
    series = [v for k, v in payload["result"].items() if k != "last"]
    require(len(series) == 1 and len(series[0]) > 0, "Een OHLC-antwoord met precies één "
            "paar verwacht.", DataContractError, keys=list(payload["result"]))
    rows = []
    for r in series[0]:
        close, vwap, vol = float(r[4]), float(r[5]), float(r[6])
        rows.append((pd.Timestamp(int(r[0]), unit="s", tz="UTC") + DAY, close,
                     float(r[2]), float(r[3]), (vwap if vwap > 0.0 else close) * vol))
    return _frame(rows)


def parse_perp_candles(payload: Mapping[str, Any]) -> tuple[pd.DataFrame, bool]:
    """`charts/v1/trade/.../1d`: `{time (ms, start), open, high, low, close, volume}`.

    Een PF-contract is één eenheid van de munt: omzet in USD = volume x close."""
    candles = payload.get("candles")
    require(isinstance(candles, list) and len(candles) > 0, "Kraken futures gaf geen "
            "candles.", DataContractError, keys=list(payload))
    rows = [(pd.Timestamp(int(c["time"]), unit="ms", tz="UTC") + DAY, float(c["close"]),
             float(c["high"]), float(c["low"]), float(c["volume"]) * float(c["close"]))
            for c in candles]
    return _frame(rows), bool(payload.get("more_candles", False))


def parse_funding(payload: Mapping[str, Any]) -> pd.Series:
    """`historicalfundingrates` -> dagsom van `relativeFundingRate`, geïndexeerd op het einde
    van de dag (de som over `[t − 1 dag, t)` staat op `t`)."""
    require(payload.get("result") == "success" and bool(payload.get("rates")),
            "Kraken gaf geen fundinghistorie.", DataContractError, result=payload.get("result"))
    ts = pd.to_datetime([r["timestamp"] for r in payload["rates"]], utc=True)
    rate = pd.Series([float(r["relativeFundingRate"]) for r in payload["rates"]],
                     index=ts.floor("D") + DAY)
    out = rate.groupby(level=0).sum()
    out.index.name = "asof_ts"
    return out


def _settled(obj: pd.DataFrame | pd.Series, now: pd.Timestamp) -> Any:
    return obj.loc[obj.index <= now]


def _sane_adv(qv: pd.Series, name: str) -> None:
    adv = float(qv.tail(30).mean())
    lo, hi = ADV_BOUNDS_USD
    require(lo <= adv <= hi, "Een dagomzet buiten het plausibele bereik: de eenheid klopt "
            "niet.", DataContractError, name=name, adv_usd=adv)


@dataclass(frozen=True)
class KrakenPanels:
    """De dagpanelen per veld (kolommen = boeksymbolen)."""

    perp: Mapping[str, pd.DataFrame]
    spot: Mapping[str, pd.DataFrame]

    @property
    def last(self) -> pd.Timestamp:
        return pd.Timestamp(self.perp["close"].index.max())


def fetch_panels(get: Get = _get, *, now: pd.Timestamp | None = None,
                 days: int = 800) -> KrakenPanels:
    """Haal alles op, tot en met de laatste bar die vóór `now` sloot."""
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    start = now.floor("D") - pd.Timedelta(days=int(days))
    perp: dict[str, dict[str, pd.Series]] = {f: {} for f in PERP_FIELDS}
    spot: dict[str, dict[str, pd.Series]] = {f: {} for f in SPOT_FIELDS}
    for sym, (pair, contract) in COINS.items():
        s = _settled(parse_spot_ohlc(get(SPOT_OHLC_URL, {"pair": pair, "interval": "1440"})
                                     .json()), now)
        parts, cursor = [], start
        while True:
            frame, more = parse_perp_candles(get(PERP_CHART_URL.format(symbol=contract), {
                "from": str(int(cursor.timestamp())), "to": str(int(now.timestamp()))}).json())
            parts.append(frame)
            # De volgende pagina begint bij de start van de bar na de laatste.
            nxt = frame.index.max()
            if not more or nxt <= cursor:
                break
            cursor = nxt
        p = _settled(pd.concat(parts).pipe(lambda d: d[~d.index.duplicated(keep="last")])
                     .sort_index(), now)
        f = _settled(parse_funding(get(FUNDING_URL, {"symbol": contract}).json()), now)
        _sane_adv(p["quote_volume"], f"{contract} perp")
        _sane_adv(s["quote_volume"], f"{pair} spot")
        for k in ("close", "high", "low", "quote_volume"):
            perp[k][sym] = p[k]
        perp["funding"][sym] = f
        for k in SPOT_FIELDS:
            spot[k][sym] = s[k]
    grid = pd.date_range(min(x.index.min() for x in perp["close"].values()),
                         max(x.index.max() for x in perp["close"].values()),
                         freq="D", name="asof_ts")
    cols = sorted(COINS)
    return KrakenPanels(
        perp={k: pd.DataFrame(v).reindex(index=grid, columns=cols) for k, v in perp.items()},
        spot={k: pd.DataFrame(v).reindex(index=grid, columns=cols) for k, v in spot.items()})


def kraken_market(panels: KrakenPanels, cfg: RobustBookV2Config) -> BasisMarket:
    """De `BasisMarket` van H3 op Kraken. Het raster begint bij de eerste dag met funding
    voor elke munt: vóór de fundinghistorie zou `harvest` nullen als carry lezen."""
    f = panels.perp["funding"]
    first = max(f[c].first_valid_index() for c in f.columns)
    require(first is not None, "Geen funding voor een van de munten.", DataContractError)
    cut = {k: v.loc[first:] for k, v in panels.perp.items()}
    breadth = breadth_from_frames(cut["close"], cut["high"], cut["low"], cut["quote_volume"],
                                  cut["funding"], cfg, source="kraken", min_members=len(COINS))
    return basis_market(breadth, {k: v.loc[first:] for k, v in panels.spot.items()}, cfg)


# --------------------------------------------------------------------------- #
# Opslaan en laden (de vooruit-lezing bewaart wat ze las)
# --------------------------------------------------------------------------- #
def _files(panels: KrakenPanels) -> dict[str, pd.DataFrame]:
    return {**{f"perp_{k}.csv": v for k, v in panels.perp.items()},
            **{f"spot_{k}.csv": v for k, v in panels.spot.items()}}


def write_panels(panels: KrakenPanels, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, frame in _files(panels).items():
        path = out_dir / name
        frame.to_csv(path, float_format="%.12g", lineterminator="\n")
        files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {"last_bar": panels.last.isoformat(), "coins": COINS, "files": files}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n",
                                           encoding="utf-8")
    return manifest


def load_panels(out_dir: Path) -> KrakenPanels:
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    frames = {}
    for name, sha in manifest["files"].items():
        path = out_dir / name
        require(hashlib.sha256(path.read_bytes()).hexdigest() == sha,
                "Een Kraken-paneel wijkt af van zijn manifest.", DataContractError, path=name)
        frames[name] = pd.read_csv(path, index_col=0, parse_dates=True)
    return KrakenPanels(
        perp={k: frames[f"perp_{k}.csv"] for k in PERP_FIELDS},
        spot={k: frames[f"spot_{k}.csv"] for k in SPOT_FIELDS})


def main(argv: Sequence[str]) -> dict[str, Any]:
    require(len(argv) == 1, "Gebruik: python -I -m tradebot.data.kraken_feed OUT_DIR",
            DataContractError)
    out = write_panels(fetch_panels(), Path(argv[0]))
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    main(sys.argv[1:])
