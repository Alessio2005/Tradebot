"""Kraken als venue voor het carryboek: spot (api.kraken.com) en perps (futures.kraken.com).

Eigenaarsbesluit 2026-10-25: Binance eruit, Kraken erin. Waarom Kraken en niet Bybit,
en wat het kost: `conf/execution/venues.yaml` en het rapport van 2026-10-25.

NIET GETOETST TEGEN DE ECHTE API
================================
De ontwikkelomgeving kan geen beurs bereiken. Getoetst is wat zonder beurs te toetsen is:
- de spot-signatuur, tegen het gepubliceerde voorbeeld van Kraken;
- elk verzoek dat de adapter bouwt;
- elk antwoord dat hij leest (tegen een nep-transport).

Daarom staat de adapter standaard op `dry_run`: hij rondt af, legt vast wat hij zou sturen
en vult tegen de referentiekoers, zonder één privé-aanroep. Vóór echt geld, in deze
volgorde:
1. `demo=True`: de futures-demo (demo-futures.kraken.com) met demo-sleutels;
2. spot met de kleinste ordergrootte;
3. de futures-signatuur (`futures_signature`) tegen een echte aanroep. Daar is geen
   gepubliceerd voorbeeld van gevonden.

EU-STRUCTUUR (de reden voor `collateral_assets`)
================================================
Spot en margin vallen onder de MiCA-entiteit (Ierland), perps onder de MiFID-entiteit
(Cyprus). Er is geen unified margin. BTC en ETH tellen als onderpand in de futureswallet
(met haircut), dus voor die munten verhuist de adapter de gekochte spot naar de
futureswallet; daar is het tegelijk het lange been en het onderpand van de short. Voor
alle andere munten blijft de spot in de spotwallet, en de short heeft eigen USD-marge.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import math
import time
import urllib.parse
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from decimal import ROUND_DOWN, Decimal
from typing import Any

from ..oms.order import OrderSide
from ..utils.failfast import DataContractError, TradebotContractError, require
from .orders import Leg, LegOrder
from .venue import LegFill, VenueBook

__all__ = ["FUTURES_DEMO_URL", "FUTURES_URL", "SPOT_URL", "KrakenError", "KrakenMarkets",
           "KrakenVenue", "futures_signature", "spot_signature"]

SPOT_URL = "https://api.kraken.com"
FUTURES_URL = "https://futures.kraken.com/derivatives"
FUTURES_DEMO_URL = "https://demo-futures.kraken.com/derivatives"
#: Kraken noemt bitcoin XBT en doge XDG (spot); de rest heet zoals elders.
ALIASES = {"BTC": "XBT", "DOGE": "XDG"}
#: Eindstatussen van een spotorder (QueryOrders).
SPOT_DONE = ("closed", "canceled", "expired")
#: Futures-endpoints zonder sleutel (ook in dry-run).
FUTURES_PUBLIC = ("/api/v3/instruments", "/api/v3/tickers")

#: Een HTTP-transport: (methode, url, headers, form-body, query) -> JSON.
Http = Callable[[str, str, Mapping[str, str], str | None, Mapping[str, str] | None],
                dict[str, Any]]


class KrakenError(TradebotContractError):
    """Kraken weigerde, of meldde iets dat geen fill is."""


def spot_signature(secret_b64: str, uri_path: str, nonce: str, postdata: str) -> str:
    """API-Sign: base64(HMAC-SHA512(secret, uri_path + SHA256(nonce + postdata)))."""
    msg = uri_path.encode() + hashlib.sha256((nonce + postdata).encode()).digest()
    return base64.b64encode(hmac.new(base64.b64decode(secret_b64), msg,
                                     hashlib.sha512).digest()).decode()


def futures_signature(secret_b64: str, endpoint_path: str, nonce: str, postdata: str) -> str:
    """Authent: base64(HMAC-SHA512(secret, SHA256(postdata + nonce + endpoint_path))).

    `endpoint_path` zonder het voorvoegsel `/derivatives` (bv. `/api/v3/sendorder`)."""
    digest = hashlib.sha256((postdata + nonce + endpoint_path).encode()).digest()
    return base64.b64encode(hmac.new(base64.b64decode(secret_b64), digest,
                                     hashlib.sha512).digest()).decode()


def _requests_http(method: str, url: str, headers: Mapping[str, str], body: str | None,
                   query: Mapping[str, str] | None) -> dict[str, Any]:
    import requests  # harde dependency (requirements.lock)

    r = requests.request(method, url, headers=dict(headers), data=body, params=query,
                         timeout=15)
    r.raise_for_status()
    out: dict[str, Any] = r.json()
    return out


def _floor(qty: float, decimals: int) -> Decimal:
    """`qty` naar beneden afgerond op `decimals` decimalen, exact (Decimal)."""
    step = Decimal(1).scaleb(-int(decimals))
    return (Decimal(str(qty)) / step).to_integral_value(rounding=ROUND_DOWN) * step


def _qty(d: Decimal | float) -> str:
    """Een hoeveelheid zoals Kraken haar krijgt: exact, zonder exponent of nullen erachter.

    `format(x, "f")` op een float geeft zes decimalen en rondt af, ook naar boven
    (0,51234567 -> "0.512346"): een grotere order dan bedoeld."""
    return format(Decimal(str(d)).normalize(), "f")


@dataclass(frozen=True)
class KrakenMarkets:
    """Van een boeksymbool (`BTCUSDT`) naar Kraken: spotpaar, spotasset, perp, precisie."""

    spot_pair: Mapping[str, str]
    spot_asset: Mapping[str, str]
    spot_decimals: Mapping[str, int]
    spot_min: Mapping[str, float]
    perp: Mapping[str, str]
    perp_decimals: Mapping[str, int]

    @staticmethod
    def build(symbols: Sequence[str], asset_pairs: Mapping[str, Any],
              instruments: Sequence[Mapping[str, Any]]) -> KrakenMarkets:
        by_alt = {p["altname"]: (key, p) for key, p in asset_pairs.items()}
        live = {i["symbol"].upper(): i for i in instruments if i.get("tradeable", True)}
        out: dict[str, dict[str, Any]] = {k: {} for k in (
            "spot_pair", "spot_asset", "spot_decimals", "spot_min", "perp", "perp_decimals")}
        for sym in symbols:
            require(sym.endswith("USDT"), "Een boeksymbool eindigt op USDT.", DataContractError,
                    symbol=sym)
            base = sym[: -len("USDT")]
            alias = ALIASES.get(base, base)
            hit = by_alt.get(f"{alias}USD")
            if hit is not None:
                key, p = hit
                out["spot_pair"][sym] = key
                out["spot_asset"][sym] = p["base"]
                out["spot_decimals"][sym] = int(p["lot_decimals"])
                out["spot_min"][sym] = float(p.get("ordermin", 0.0))
            for name in (f"PF_{alias}USD", f"PF_{base}USD"):
                if name in live:
                    out["perp"][sym] = name
                    out["perp_decimals"][sym] = int(
                        live[name].get("contractValueTradePrecision", 0))
                    break
        return KrakenMarkets(**out)

    def tradable(self, symbol: str) -> bool:
        """Beide benen bestaan op Kraken."""
        return symbol in self.spot_pair and symbol in self.perp


@dataclass
class KrakenVenue:
    """`carry.venue.Venue` op Kraken (zie de moduledocstring: standaard dry-run)."""

    symbols: Sequence[str]
    spot_key: str = ""
    spot_secret: str = ""
    futures_key: str = ""
    futures_secret: str = ""
    dry_run: bool = True
    demo: bool = False
    collateral_assets: frozenset[str] = frozenset({"BTC", "ETH"})
    http: Http = _requests_http
    polls: int = 20
    poll_interval_s: float = 0.5
    sleep: Callable[[float], None] = time.sleep
    nonce: Callable[[], str] = field(default=lambda: str(time.time_ns() // 1_000))
    is_paper: bool = False
    #: De equity die de dry-run meldt (er is geen account om haar uit te lezen).
    dry_equity_usd: float = 0.0
    dry_book: dict[str, dict[str, float]] = field(
        default_factory=lambda: {"spot": {}, "perp": {}})
    sent: list[dict[str, Any]] = field(default_factory=list)
    _markets: KrakenMarkets | None = None

    # ------------------------------------------------------------------ transport
    def _futures_base(self) -> str:
        return FUTURES_DEMO_URL if self.demo else FUTURES_URL

    def _public_spot(self, method: str, query: Mapping[str, str] | None = None) -> Any:
        res = self.http("GET", f"{SPOT_URL}/0/public/{method}", {}, None, query)
        if res.get("error"):
            raise KrakenError(f"Kraken spot {method}: {res['error']}")
        return res["result"]

    def _private_spot(self, method: str, data: Mapping[str, Any]) -> Any:
        require(not self.dry_run, "Een privé-aanroep in dry-run.", KrakenError, method=method)
        nonce = self.nonce()
        uri = f"/0/private/{method}"
        body = urllib.parse.urlencode({"nonce": nonce, **data})
        headers = {"API-Key": self.spot_key,
                   "API-Sign": spot_signature(self.spot_secret, uri, nonce, body),
                   "Content-Type": "application/x-www-form-urlencoded"}
        res = self.http("POST", f"{SPOT_URL}{uri}", headers, body, None)
        if res.get("error"):
            raise KrakenError(f"Kraken spot {method}: {res['error']}")
        return res["result"]

    def _futures(self, method: str, path: str, data: Mapping[str, Any] | None = None) -> Any:
        """Futures v3. Publiek als er geen sleutel nodig is (`FUTURES_PUBLIC`)."""
        url = f"{self._futures_base()}{path}"
        body = urllib.parse.urlencode(dict(data or {}))
        headers: dict[str, str] = {}
        if path not in FUTURES_PUBLIC:
            require(not self.dry_run, "Een privé-aanroep in dry-run.", KrakenError, path=path)
            nonce = self.nonce()
            headers = {"APIKey": self.futures_key, "Nonce": nonce,
                       "Authent": futures_signature(self.futures_secret, path, nonce, body)}
        res = self.http(method, url, headers, body if method == "POST" else None,
                        None if method == "POST" else dict(data or {}))
        if res.get("result") != "success":
            raise KrakenError(f"Kraken futures {path}: {res}")
        return res

    # ------------------------------------------------------------------ markten
    def markets(self) -> KrakenMarkets:
        if self._markets is None:
            pairs = self._public_spot("AssetPairs")
            instruments = self._futures("GET", "/api/v3/instruments")["instruments"]
            self._markets = KrakenMarkets.build(self.symbols, pairs, instruments)
        return self._markets

    # ------------------------------------------------------------------ publiek
    def prices(self) -> tuple[dict[str, float], dict[str, float]]:
        """De live koers per boeksymbool: (spot laatste trade, perp markprijs). Publiek."""
        mk = self.markets()
        pairs = {mk.spot_pair[s]: s for s in self.symbols if s in mk.spot_pair}
        spot: dict[str, float] = {}
        if pairs:
            ticks = self._public_spot("Ticker", {"pair": ",".join(pairs)})
            spot = {pairs[k]: float(v["c"][0]) for k, v in ticks.items() if k in pairs}
        contracts = {mk.perp[s].upper(): s for s in self.symbols if s in mk.perp}
        perp = {}
        for t in self._futures("GET", "/api/v3/tickers")["tickers"]:
            sym = contracts.get(str(t.get("symbol", "")).upper())
            if sym is not None:
                perp[sym] = float(t.get("markPrice") or t["last"])
        return spot, perp

    def server_time(self) -> float:
        """De klok van Kraken (unix-seconden): de nonce leunt op de eigen klok."""
        return float(self._public_spot("Time")["unixtime"])

    # ------------------------------------------------------------------ stand
    def book(self) -> VenueBook:
        """Spot = spotwallet + onderpand in de futureswallet; equity in USD."""
        if self.dry_run:
            return VenueBook(spot=dict(self.dry_book["spot"]),
                             perp_short=dict(self.dry_book["perp"]),
                             equity_usd=self.dry_equity_usd)
        mk = self.markets()
        bal = {k: float(v) for k, v in self._private_spot("Balance", {}).items()}
        flex = self._futures("GET", "/api/v3/accounts")["accounts"]["flex"]
        coll = {k: float(v.get("quantity", 0.0)) for k, v in flex["currencies"].items()}
        spot: dict[str, float] = {}
        for sym, asset in mk.spot_asset.items():
            base = sym[: -len("USDT")]
            q = bal.get(asset, 0.0) + coll.get(ALIASES.get(base, base), 0.0)
            if q > 0.0:
                spot[sym] = q
        to_book = {v: k for k, v in mk.perp.items()}
        shorts: dict[str, float] = {}
        for p in self._futures("GET", "/api/v3/openpositions")["openPositions"]:
            sym = to_book.get(str(p["symbol"]).upper())
            require(sym is not None, "Een perp-positie buiten het boek.", KrakenError,
                    symbol=p["symbol"])
            assert sym is not None
            signed = float(p["size"]) * (-1.0 if p["side"] == "short" else 1.0)
            require(signed <= 0.0, "Een LONGE perp in een carryboek.", KrakenError,
                    symbol=p["symbol"])
            shorts[sym] = -signed
        held = [mk.spot_pair[s] for s in mk.spot_pair if bal.get(mk.spot_asset[s], 0.0) > 0.0]
        prices = ({k: float(v["c"][0]) for k, v in self._public_spot(
            "Ticker", {"pair": ",".join(held)}).items()} if held else {})
        spot_value = sum(bal.get(mk.spot_asset[s], 0.0) * prices[mk.spot_pair[s]]
                         for s in mk.spot_pair if mk.spot_pair[s] in prices)
        equity = bal.get("ZUSD", 0.0) + spot_value + float(flex["portfolioValue"])
        return VenueBook(spot=spot, perp_short=shorts, equity_usd=equity)

    # ------------------------------------------------------------------ orders
    def execute(self, orders: Sequence[LegOrder]) -> list[LegFill]:
        mk = self.markets()
        fills = []
        for o in orders:
            require(mk.tradable(o.symbol), "Deze munt heeft op Kraken niet beide benen.",
                    KrakenError, symbol=o.symbol)
            fills.append(self._spot(o, mk) if o.leg is Leg.SPOT else self._perp(o, mk))
        return fills

    def _collateral(self, symbol: str) -> str | None:
        base = symbol[: -len("USDT")]
        return ALIASES.get(base, base) if base in self.collateral_assets else None

    def _spot(self, o: LegOrder, mk: KrakenMarkets) -> LegFill:
        dq = _floor(o.qty, mk.spot_decimals[o.symbol])
        qty = float(dq)
        require(qty >= mk.spot_min[o.symbol], "Onder de minimale ordergrootte van Kraken.",
                KrakenError, symbol=o.symbol, qty=qty, min=mk.spot_min[o.symbol])
        order = {"pair": mk.spot_pair[o.symbol], "type": o.side.value.lower(),
                 "ordertype": "market", "volume": _qty(dq),
                 "cl_ord_id": str(uuid.uuid4())}
        coll = self._collateral(o.symbol)
        if self.dry_run:
            self.sent.append({"leg": "spot", **order})
            book = self.dry_book["spot"]
            book[o.symbol] = book.get(o.symbol, 0.0) + (qty if o.side is OrderSide.BUY else -qty)
            return LegFill(order=o, qty=qty, price=o.ref_price, fee_usd=0.0)
        if coll is not None and o.side is OrderSide.SELL:
            # Het onderpand staat in de futureswallet: eerst terug naar spot.
            self._futures("POST", "/api/v3/withdrawal",
                          {"currency": coll, "amount": _qty(dq)})
        txid = self._private_spot("AddOrder", order)["txid"][0]
        row: Mapping[str, Any] = {}
        for attempt in range(self.polls):
            if attempt:
                self.sleep(self.poll_interval_s)
            row = self._private_spot("QueryOrders", {"txid": txid, "trades": "true"})[txid]
            if row["status"] in SPOT_DONE:
                break
        else:
            raise KrakenError(f"Spotorder {txid} heeft na {self.polls} pogingen geen "
                              f"eindstatus ({row.get('status')!r}); de stand is onbekend.")
        filled, price = float(row["vol_exec"]), float(row.get("price") or 0.0)
        if filled <= 0.0 or not (math.isfinite(price) and price > 0.0):
            raise KrakenError(f"Spotorder {txid} eindigde {row['status']!r} zonder fill.")
        if coll is not None and o.side is OrderSide.BUY:
            self._private_spot("WalletTransfer", {"asset": coll, "from": "Spot Wallet",
                                                  "to": "Futures Wallet",
                                                  "amount": _qty(filled)})
        return LegFill(order=o, qty=filled, price=price, fee_usd=float(row.get("fee") or 0.0))

    def _perp(self, o: LegOrder, mk: KrakenMarkets) -> LegFill:
        dsize = _floor(o.qty, mk.perp_decimals[o.symbol])
        size = float(dsize)
        require(size > 0.0, "Een perp-order van nul contracten.", KrakenError, symbol=o.symbol)
        order = {"orderType": "mkt", "symbol": mk.perp[o.symbol],
                 "side": o.side.value.lower(), "size": _qty(dsize),
                 "cliOrdId": str(uuid.uuid4())}
        if o.reduce_only:
            order["reduceOnly"] = "true"
        if self.dry_run:
            self.sent.append({"leg": "perp", **order})
            book = self.dry_book["perp"]
            book[o.symbol] = book.get(o.symbol, 0.0) + (
                size if o.side is OrderSide.SELL else -size)
            return LegFill(order=o, qty=size, price=o.ref_price, fee_usd=0.0)
        status = self._futures("POST", "/api/v3/sendorder", order)["sendStatus"]
        execs = [e for e in status.get("orderEvents", []) if e.get("type") == "EXECUTION"]
        filled = sum(float(e["amount"]) for e in execs)
        if status.get("status") != "placed" or filled <= 0.0:
            raise KrakenError(f"Perp-order {order['cliOrdId']} niet uitgevoerd: {status}")
        price = sum(float(e["price"]) * float(e["amount"]) for e in execs) / filled
        # De perp-fee staat niet in het orderantwoord maar in het accountlog; de equity die
        # `book()` meldt, is er al na. Hier dus 0, niet geschat.
        return LegFill(order=o, qty=filled, price=price, fee_usd=0.0)
