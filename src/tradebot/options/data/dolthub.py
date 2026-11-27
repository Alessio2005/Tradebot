"""Een historische replay-markt uit echte optieketens van DoltHub (`post-no-preference/options`).

De database heeft einde-dag-ketens (bid, ask en IV per contract) voor US-aandelen en een paar
ETF's, vanaf februari 2019. Deze module haalt ze op, bewaart ze per dag in een lokale cache en
bouwt er een `Market` van in het formaat van `data.store`, zodat `replay` en `sweep` ze afspelen.

Wat de bron WEL en NIET geeft, en wat dat voor de uitkomst betekent (ook in de CHANGELOG):

* **Eén snapshot per dag.** De quotes zijn van rond de slotkoers. De markt zet ze op
  `DoltSpec.snapshot_time` (standaard 15:15 New York) en `replay_config` legt de scantijd en
  het sluitmoment vóór expiratie op datzelfde tijdstip (`Market.daily_snapshot`). Er is dus
  geen intraday-pad: alles van een dag gebeurt op één prijs. Dat is iets optimistisch
  (slotquote om 15:15), maar zonder look-ahead op de koers.
* **Niet elke dag.** In 2019 is er één keten per week (op zaterdag, met de quotes van
  vrijdag); later meestal maandag, woensdag en vrijdag. Een datum die geen handelsdag is,
  schuift naar de handelsdag ervoor. Op handelsdagen zonder keten staat er een snapshot met
  alleen de slotkoers: posities kunnen dan niet sluiten, wel expireren en settelen.
* **Weinig expiraties en strikes.** Meestal drie expiraties en strikes in grove stappen; verre
  vleugels ontbreken vaak. Strategieën die een precieze delta of vleugel zoeken, vinden dan
  niets of iets verder weg.
* **Geen volume of open interest.** Die velden staan vast op `DoltSpec.volume` en
  `DoltSpec.open_interest` (boven de liquiditeitsfilters). Een fill komt uit de paper-broker
  op bid/ask: optimistischer dan de echte markt, waar een order in een dunne keten niet vult.
* **IV, greeks en dividenden.** De IV en greeks van DoltHub worden NIET gebruikt: de engine
  rekent ze uit de quotes met QuantLib (Amerikaans, met dividenden). Dagbars en dividenden komen uit
  een andere bron (Yahoo's chart-API, teruggerekend naar de koersen van toen), earnings uit
  SEC EDGAR (8-K item 2.02). De koers bij de keten is de slotkoers uit
  die bron. Kleine verschillen met de koers waarop DoltHub de
  quotes nam, zijn er dus.
* **Universum.** QQQ en IWM staan niet in de database; SPY, AAPL en MSFT wel. Een naam zonder
  ketens valt met een waarschuwing uit de markt.
* **Rente.** Eén vaste rente voor de hele periode (`DoltSpec.rate`).

`python -I -m tradebot.options dolthub --cache DIR --out DIR --start D --end D` haalt de
ontbrekende dagen op (alleen lezen, openbare API), bouwt de markt en schrijft hem weg.
"""
from __future__ import annotations

import csv
import json
import time as _time
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from ...utils.failfast import DataContractError, require
from ..calendar import NY, TradingCalendar
from ..chain.builder import RawOption
from ..contracts import OptionContract, Quote, Right
from ..events import Events
from ..pricing.engines import Method
from ..pricing.ql_env import scaled_time
from ..pricing.vanilla import OptionInputs, price
from ..rates import Carry, CashDividend
from .store import Market, Snapshot

__all__ = ["API", "COLUMNS", "DEFAULT_RATE", "DEFAULT_SYMBOLS", "DoltSpec", "UnderlyingData",
           "build_market", "cache_days", "download", "fetch_day", "make_market", "read_cache",
           "sec_earnings", "yahoo_underlying"]

API = "https://www.dolthub.com/api/v1alpha1/post-no-preference/options/master"
COLUMNS = ("date", "act_symbol", "expiration", "strike", "call_put", "bid", "ask", "vol",
           "delta", "gamma", "theta", "vega", "rho")
DEFAULT_SYMBOLS = ("SPY", "QQQ", "IWM", "AAPL", "MSFT")
FIRST_DAY = date(2019, 2, 9)            # de eerste keten in de database
DEFAULT_RATE = 0.03                     # één vaste rente voor 2019-2026 (gemiddeld)

HttpGet = Callable[[str], Mapping[str, Any]]


@dataclass(frozen=True, slots=True)
class DoltSpec:
    start: date
    end: date
    rate: float = DEFAULT_RATE
    snapshot_time: time = time(15, 15)
    open_interest: int = 5000
    volume: int = 1000
    half_days: tuple[date, ...] = ()
    fill_days: int = 60          # handelsdagen dat een verdwenen contract aangevuld wordt; 0 = uit
    fill_moneyness: float = 0.20  # alleen aanvullen binnen |K/S − 1| ≤ dit (verder: ~intrinsiek)


@dataclass(frozen=True, slots=True)
class UnderlyingData:
    """Dagbars (open, high, low, close, volume; datumindex) en de kalender van één naam."""
    name: str
    bars: pd.DataFrame
    dividends: tuple[CashDividend, ...] = ()
    earnings: tuple[date, ...] = ()


# --- ophalen (netwerk) ---------------------------------------------------------------------

def _http_get(url: str) -> Mapping[str, Any]:  # pragma: no cover - echt netwerk
    with urllib.request.urlopen(url, timeout=120) as r:
        out: Mapping[str, Any] = json.load(r)
    return out


def _sql(day: date, symbols: Sequence[str]) -> str:
    for s in symbols:
        require(s.isalnum(), "ongeldig symbool", DataContractError, symbol=s)
    names = ",".join(f"'{s}'" for s in symbols)
    return (f"SELECT * FROM option_chain WHERE date='{day.isoformat()}' "
            f"AND act_symbol IN ({names})")


def fetch_day(day: date, symbols: Sequence[str], get: HttpGet = _http_get,
              retries: int = 5, backoff: float = 5.0,
              sleep: Callable[[float], None] = _time.sleep) -> list[dict[str, Any]]:
    """Alle rijen van `option_chain` op `day` voor `symbols`, via de SQL-API van DoltHub."""
    url = f"{API}?{urllib.parse.urlencode({'q': _sql(day, symbols)})}"
    last = ""
    for attempt in range(retries + 1):
        try:
            res = get(url)
        except OSError as e:
            last = repr(e)
        else:
            if res.get("query_execution_status") == "Success":
                return [dict(r) for r in res.get("rows", [])]
            last = str(res.get("query_execution_message", ""))
        if attempt < retries:
            sleep(backoff * 2 ** attempt)
    raise DataContractError(f"DoltHub-query mislukt voor {day.isoformat()}: {last}")


def _days(start: date, end: date) -> list[date]:
    """Kandidaatdata: werkdagen, plus de zaterdagen van 2019 (toen kwam de keten op zaterdag)."""
    out, d = [], max(start, FIRST_DAY)
    while d <= end:
        if d.weekday() < 5 or (d.weekday() == 5 and d.year == 2019):
            out.append(d)
        d += timedelta(days=1)
    return out


def download(cache: Path, symbols: Sequence[str], start: date, end: date,
             get: HttpGet = _http_get, workers: int = 4,
             log: Callable[[str], None] = print) -> list[date]:
    """Haal de dagen in `[start, end]` op die nog niet in `cache` staan.

    Per datum één bestand: `YYYY-MM-DD.csv` met de rijen, of `YYYY-MM-DD.empty` als er die
    dag niets is. Een bestand wordt pas onder zijn naam gezet als het compleet is, dus een
    afgebroken download hervat waar hij stopte. Geeft de data met ketens terug.
    """
    cache.mkdir(parents=True, exist_ok=True)
    todo = [d for d in _days(start, end) if not (cache / f"{d}.csv").exists()
            and not (cache / f"{d}.empty").exists()]

    def one(d: date) -> None:
        rows = fetch_day(d, symbols, get)
        if not rows:
            (cache / f"{d}.empty").touch()
            return
        tmp = cache / f"{d}.part"
        with tmp.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, COLUMNS, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        tmp.replace(cache / f"{d}.csv")
        log(f"{d}: {len(rows)} rijen")

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        list(ex.map(one, todo))
    return cache_days(cache, start, end)


def cache_days(cache: Path, start: date, end: date) -> list[date]:
    return sorted(d for d in (date.fromisoformat(p.stem) for p in cache.glob("*.csv"))
                  if start <= d <= end)


def read_cache(cache: Path, start: date, end: date) -> pd.DataFrame:
    """De rijen van alle gecachete dagen in `[start, end]` als één tabel."""
    days = cache_days(cache, start, end)
    require(bool(days), "geen DoltHub-ketens in de cache voor de periode", DataContractError,
            cache=str(cache), start=start.isoformat(), end=end.isoformat())
    df = pd.concat([pd.read_csv(cache / f"{d}.csv", dtype={"act_symbol": str})
                    for d in days], ignore_index=True)
    return df[list(COLUMNS[:8])]


CHART = "https://query2.finance.yahoo.com/v8/finance/chart/"
EarningsSource = Callable[[str, date, date], tuple[date, ...]]


def _yahoo_get(url: str) -> Mapping[str, Any]:  # pragma: no cover - echt netwerk
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        out: Mapping[str, Any] = json.load(r)
    return out


SEC = "https://data.sec.gov/submissions/"
# CIK-nummers bij de SEC. ETF's (SPY, QQQ, IWM, ...) hebben geen earnings en staan er niet in.
CIK = {"AAPL": 320193, "MSFT": 789019}
ETFS = frozenset({"SPY", "QQQ", "IWM", "DIA", "XLF"})


def _sec_get(url: str) -> Mapping[str, Any]:  # pragma: no cover - echt netwerk
    req = urllib.request.Request(url, headers={"User-Agent": "tradebot-research"})
    with urllib.request.urlopen(req, timeout=60) as r:
        out: Mapping[str, Any] = json.load(r)
    return out


def sec_earnings(symbol: str, start: date, end: date, get: HttpGet = _sec_get,
                 log: Callable[[str], None] = print) -> tuple[date, ...]:
    """Earningsdata uit SEC EDGAR: de datum van elk 8-K met item 2.02 (Results of Operations).

    Dat is de dag van het persbericht; bij AAPL en MSFT is dat na de slotbel, dus de
    koerssprong valt de handelsdag erna. Een ETF heeft geen earnings; een naam zonder CIK
    in de tabel krijgt er geen (met een melding).
    """
    if symbol in ETFS:
        return ()
    cik = CIK.get(symbol)
    if cik is None:
        log(f"{symbol}: geen CIK bekend, geen earningsdata")
        return ()
    root = get(f"{SEC}CIK{cik:010d}.json")
    tables = [root["filings"]["recent"]]
    for f in root["filings"].get("files", []):
        if date.fromisoformat(f["filingTo"]) >= start:
            tables.append(get(f"{SEC}{f['name']}"))
    out = {date.fromisoformat(d) for t in tables
           for form, d, items in zip(t["form"], t["filingDate"], t["items"], strict=True)
           if form == "8-K" and "2.02" in str(items).split(",")}
    return tuple(sorted(d for d in out if start <= d <= end + timedelta(days=120)))


def yahoo_underlying(symbol: str, start: date, end: date, get: HttpGet = _yahoo_get,
                     earnings: EarningsSource = sec_earnings,
                     today: date | None = None) -> UnderlyingData:
    """Dagbars en dividenden uit Yahoo's chart-API, terug naar de koersen van toen.

    Yahoo corrigeert koersen en dividenden voor splitsingen; de strikes van DoltHub zijn de
    echte strikes van die dag. Vóór elke split gaan prijzen en dividenden dus × de ratio
    (en het volume ÷ de ratio). Dividenden blijven verder ongecorrigeerd. Yahoo geeft
    alleen de splits binnen de gevraagde periode, dus de vraag loopt tot vandaag (`today`)
    en de bars worden daarna op `end` afgekapt.
    """
    require(symbol.isalnum(), "ongeldig symbool", DataContractError, symbol=symbol)
    p1 = int(datetime.combine(start, time(0), tzinfo=NY).timestamp())
    until = max(end, today or datetime.now(NY).date()) + timedelta(days=1)
    p2 = int(datetime.combine(until, time(0), tzinfo=NY).timestamp())
    res = get(f"{CHART}{symbol}?period1={p1}&period2={p2}&interval=1d&events=div%2Csplits")
    rows = (res.get("chart") or {}).get("result") or []
    require(bool(rows) and bool(rows[0].get("timestamp")), "geen koersen van Yahoo",
            DataContractError, symbol=symbol)
    r = rows[0]
    q = r["indicators"]["quote"][0]

    def day(ts: Any) -> date:
        return datetime.fromtimestamp(int(ts), tz=NY).date()

    splits = [(day(v["date"]), float(v["numerator"]) / float(v["denominator"]))
              for v in (r.get("events") or {}).get("splits", {}).values()]

    def factor(d: date) -> float:
        f = 1.0
        for when, ratio in splits:
            if d < when:
                f *= ratio
        return f

    out = []
    for i, ts in enumerate(r["timestamp"]):
        d = day(ts)
        if q["close"][i] is None:
            continue
        f = factor(d)
        out.append({"date": pd.Timestamp(d), **{k: float(q[k][i]) * f
                                               for k in ("open", "high", "low", "close")},
                    "volume": float(q["volume"][i] or 0.0) / f})
    bars = pd.DataFrame(out).set_index("date")
    bars = bars[~bars.index.duplicated(keep="last")].sort_index()
    bars = bars[bars.index <= pd.Timestamp(end)]
    divs = tuple(sorted((CashDividend(day(v["date"]), float(v["amount"]) * factor(day(v["date"])))
                         for v in (r.get("events") or {}).get("dividends", {}).values()
                         if float(v["amount"]) > 0
                         and day(v["date"]) <= end + timedelta(days=120)),
                        key=lambda x: x.ex_date))
    return UnderlyingData(symbol, bars, divs, earnings(symbol, start, end))


# --- markt bouwen --------------------------------------------------------------------------

def _price(v: Any, floor: float) -> float | None:
    """Een prijs uit de bron, of `None` als hij ontbreekt of onder `floor` ligt."""
    if v is None or pd.isna(v):
        return None
    x = float(v)
    return x if x >= floor else None


def _snapshot_day(cal: TradingCalendar, d: date) -> date:
    return d if cal.is_trading_day(d) else cal.previous_trading_day(d)


def build_market(chains: pd.DataFrame, underlyings: Iterable[UnderlyingData], spec: DoltSpec,
                 warn: Callable[[str], None] = print) -> Market:
    """Een replay-markt: elke handelsdag in de periode met koers één snapshot, met de
    DoltHub-keten van die dag, aangevuld met de contracten die de laatste `fill_days`
    handelsdagen wél in de keten stonden (zie `_fill`). Zonder keten en zonder recente
    contracten is de snapshot leeg: alleen de slotkoers."""
    cal = TradingCalendar(spec.half_days)
    data = {u.name: u for u in underlyings}
    have = set(chains["act_symbol"].astype(str))
    for name in sorted(set(data) - have):
        warn(f"{name}: geen DoltHub-ketens, valt uit de markt")
    names = sorted(set(data) & have)
    require(bool(names), "geen onderliggende met ketens en koersen", DataContractError)
    bars: dict[str, pd.DataFrame] = {}
    for n in names:
        b = data[n].bars.copy()
        b.index = pd.DatetimeIndex(pd.to_datetime(b.index), name="date")
        bars[n] = b[b.index <= pd.Timestamp(spec.end)][["open", "high", "low", "close",
                                                         "volume"]]
    common = set.intersection(*({t.date() for t in b.index} for b in bars.values()))
    days = tuple(sorted(d for d in common if spec.start <= d <= spec.end
                        and cal.is_trading_day(d)))
    require(bool(days), "geen handelsdagen met koersen in de periode", DataContractError,
            start=spec.start.isoformat(), end=spec.end.isoformat())
    df = chains.copy()
    df["day"] = [_snapshot_day(cal, date.fromisoformat(str(d))) for d in df["date"]]
    # Twee data op dezelfde handelsdag (een zaterdag én de vrijdag ervoor): de laatste wint.
    df["src"] = df["date"].astype(str)
    last = df.groupby(["act_symbol", "day"])["src"].transform("max")
    df = df[df["src"] == last]
    groups = {(str(k[0]), k[1]): g for k, g in df.groupby(["act_symbol", "day"], sort=False)}
    events = {n: Events(n, spec.end, tuple(sorted(data[n].earnings)),
                        tuple(sorted(data[n].dividends, key=lambda d: d.ex_date)))
              for n in names}
    snapshots = []
    for n in names:
        closes = bars[n]["close"]
        carry = Carry(spec.rate, dividends=events[n].dividends)
        seen: dict[OptionContract, _Seen] = {}
        for i, day in enumerate(days):
            ts = datetime.combine(day, spec.snapshot_time, tzinfo=NY)
            spot = float(closes.loc[pd.Timestamp(day)])
            g = groups.get((n, day))
            real = [] if g is None else _options(n, g, ts, spec, seen, i)
            have_c = {o.contract for o in real}
            fill = [_fill(c, v, spot, ts, carry) for c, v in sorted(seen.items(), key=_key)
                    if c not in have_c and c.expiry >= day and i - v.index <= spec.fill_days
                    and abs(c.strike / spot - 1.0) <= spec.fill_moneyness]
            opts = sorted(real + fill, key=lambda o: _key((o.contract, None)))
            snapshots.append(Snapshot(n, ts, spot, tuple(opts)))
    snapshots.sort(key=lambda s: (s.asof, s.underlying))
    return Market(spec.rate, days, bars, tuple(snapshots), events,
                  daily_snapshot=spec.snapshot_time.strftime("%H:%M"))


@dataclass(frozen=True, slots=True)
class _Seen:
    """De laatste echte quote van een contract: IV van DoltHub, spread en de dagindex."""
    iv: float
    spread: float
    index: int


def _key(item: tuple[OptionContract, Any]) -> tuple[date, float, str]:
    c = item[0]
    return (c.expiry, c.strike, c.right.value)


def _options(name: str, g: pd.DataFrame, ts: datetime, spec: DoltSpec,
             seen: dict[OptionContract, _Seen], index: int) -> list[RawOption]:
    out = []
    for r in g.itertuples(index=False):
        expiry = date.fromisoformat(str(r.expiration))
        if expiry < ts.date():
            continue
        right = Right.CALL if str(r.call_put).lower().startswith("c") else Right.PUT
        bid, ask = _price(r.bid, 0.0), _price(r.ask, 0.01)
        if bid is not None and ask is not None and ask < bid:
            bid, ask = None, None              # gekruist boek: niet bruikbaar
        c = OptionContract(name, expiry, round(float(r.strike), 2), right)
        iv = _price(r.vol, 0.01)
        if iv is not None and bid is not None and ask is not None:
            seen[c] = _Seen(iv, ask - bid, index)
        out.append(RawOption(c, Quote(bid, ask, ts, volume=spec.volume,
                                      open_interest=spec.open_interest)))
    return out


def _fill(c: OptionContract, v: _Seen, spot: float, ts: datetime, carry: Carry) -> RawOption:
    """Een aangevulde quote: QuantLib (BAW, of FD bij een dividend) op de koers van vandaag met de laatste IV en spread
    van het contract. Volume en open interest 0: de filters laten hem niet toe voor een nieuwe
    entry, alleen een open positie (die de filters omzeilt) kan erop waarderen en sluiten."""
    start = scaled_time(ts, c.expiry).eval_date      # hetzelfde venster als de pricer
    method = Method.FD if carry.dividends_in(start, c.expiry) else Method.BAW
    theo = price(OptionInputs(spot, c.strike, c.right, c.expiry, ts, carry), v.iv, method)
    half = 0.5 * max(v.spread, 0.01)
    bid = max(round(theo - half, 2), 0.0)
    ask = max(round(theo + half, 2), round(bid + 0.01, 2))
    return RawOption(c, Quote(bid, ask, ts, volume=0, open_interest=0))


UnderlyingSource = Callable[[str, date, date], UnderlyingData]


def make_market(cache: Path, spec: DoltSpec, symbols: Sequence[str] = DEFAULT_SYMBOLS,
                fetch: bool = True, get: HttpGet = _http_get,
                underlying: UnderlyingSource = yahoo_underlying, warmup_days: int = 150,
                log: Callable[[str], None] = print) -> Market:
    """Vul de cache aan (tenzij `fetch` uit staat), lees hem en bouw de markt. De dagbars
    beginnen `warmup_days` kalenderdagen vóór de start, voor RV60 en de trendfilters."""
    if fetch:
        download(cache, symbols, spec.start, spec.end, get, log=log)
    chains = read_cache(cache, spec.start, spec.end)
    present = sorted(set(chains["act_symbol"].astype(str)) & set(symbols))
    for s in sorted(set(symbols) - set(present)):
        log(f"{s}: geen DoltHub-ketens, valt uit de markt")
    data = [underlying(s, spec.start - timedelta(days=warmup_days), spec.end) for s in present]
    return build_market(chains[chains["act_symbol"].isin(present)], data, spec, warn=log)
