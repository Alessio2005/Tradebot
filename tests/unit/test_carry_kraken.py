"""De Kraken-adapter (`carry/kraken.py`) tegen een nep-Kraken.

Er gaat niets over het netwerk. Getoetst wordt: de spot-signatuur tegen het gepubliceerde
voorbeeld van Kraken, de opbouw van elk verzoek, het lezen van elk antwoord, en dat de
adapter zonder eindstatus of zonder fill nooit een fill meldt.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import urllib.parse
from typing import Any

import pytest

from tradebot.carry.kraken import (
    FUTURES_DEMO_URL,
    FUTURES_URL,
    SPOT_URL,
    KrakenError,
    KrakenMarkets,
    KrakenVenue,
    futures_signature,
    spot_signature,
)
from tradebot.carry.orders import Leg, LegOrder
from tradebot.oms.order import OrderSide
from tradebot.utils.failfast import DataContractError

SECRET = base64.b64encode(b"k" * 64).decode()

PAIRS = {"XXBTZUSD": {"altname": "XBTUSD", "base": "XXBT", "lot_decimals": 8,
                      "ordermin": "0.0001"},
         "XETHZUSD": {"altname": "ETHUSD", "base": "XETH", "lot_decimals": 8,
                      "ordermin": "0.002"},
         "SOLUSD": {"altname": "SOLUSD", "base": "SOL", "lot_decimals": 8, "ordermin": "0.02"},
         "XXDGZUSD": {"altname": "XDGUSD", "base": "XXDG", "lot_decimals": 8,
                      "ordermin": "50"}}
INSTRUMENTS = [{"symbol": "PF_XBTUSD", "contractValueTradePrecision": 4, "tradeable": True},
               {"symbol": "PF_ETHUSD", "contractValueTradePrecision": 3, "tradeable": True},
               {"symbol": "PF_SOLUSD", "contractValueTradePrecision": 2, "tradeable": True},
               {"symbol": "PF_DOGEUSD", "contractValueTradePrecision": 0, "tradeable": False}]


class FakeKraken:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, str], str | None]] = []
        self.statuses: list[dict[str, Any]] = [
            {"status": "closed", "vol_exec": "0.5", "price": "40000", "fee": "16.0"}]
        self.send: dict[str, Any] = {"status": "placed", "orderEvents": [
            {"type": "EXECUTION", "amount": "0.3", "price": "40100"},
            {"type": "EXECUTION", "amount": "0.2", "price": "40050"},
            {"type": "PLACE"}]}
        self.positions = [{"side": "short", "symbol": "pf_xbtusd", "size": "0.5"}]

    def __call__(self, method, url, headers, body, query):
        self.calls.append((method, url, dict(headers), body))
        path = url.split(".com", 1)[1]
        if path.endswith("/0/public/AssetPairs"):
            return {"error": [], "result": PAIRS}
        if path.endswith("/api/v3/instruments"):
            return {"result": "success", "instruments": INSTRUMENTS}
        if path.endswith("/0/public/Ticker"):
            return {"error": [], "result": {"XXBTZUSD": {"c": ["40000", "1"]}}}
        if path.endswith("/0/private/Balance"):
            return {"error": [], "result": {"XXBT": "0.25", "ZUSD": "1000"}}
        if path.endswith("/api/v3/accounts"):
            return {"result": "success", "accounts": {"flex": {
                "currencies": {"XBT": {"quantity": 0.25}, "USD": {"quantity": 500}},
                "portfolioValue": 10_500.0}}}
        if path.endswith("/api/v3/openpositions"):
            return {"result": "success", "openPositions": self.positions}
        if path.endswith("/0/private/AddOrder"):
            return {"error": [], "result": {"txid": ["OABC-1"]}}
        if path.endswith("/0/private/QueryOrders"):
            row = self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]
            return {"error": [], "result": {"OABC-1": row}}
        if path.endswith("/0/private/WalletTransfer"):
            return {"error": [], "result": {"refid": "T1"}}
        if path.endswith("/api/v3/withdrawal"):
            return {"result": "success"}
        if path.endswith("/api/v3/sendorder"):
            return {"result": "success", "sendStatus": self.send}
        raise AssertionError(f"onverwacht: {url}")

    def paths(self) -> list[str]:
        return [u.split(".com", 1)[1] for _, u, _, _ in self.calls]


def _venue(fake: FakeKraken, **kw: Any) -> KrakenVenue:
    base = dict(symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT"], spot_key="SK",
                spot_secret=SECRET, futures_key="FK", futures_secret=SECRET, dry_run=False,
                http=fake, sleep=lambda s: None, nonce=lambda: "1700000000000")
    return KrakenVenue(**{**base, **kw})


def _order(sym="BTCUSDT", leg=Leg.SPOT, side=OrderSide.BUY, qty=0.512345678,
           reduce_only=False) -> LegOrder:
    return LegOrder(sym, leg, side, qty, qty * 40000.0, 40000.0, reduce_only)


def _form(body: str | None) -> dict[str, str]:
    return dict(urllib.parse.parse_qsl(body or ""))


# --------------------------------------------------------------------------- #
def test_the_spot_signature_reproduces_krakens_published_example():
    secret = ("kQH5HW/8p1uGOVjbgWA7FunAmGO8lsSUXNsu3eow76sz84Q18fWxnyRzBHCd3pd5nE9qa99HA"
              "ZtuZuj6F1huXg==")
    body = "nonce=1616492376594&ordertype=limit&pair=XBTUSD&price=37500&type=buy&volume=1.25"
    assert spot_signature(secret, "/0/private/AddOrder", "1616492376594", body) == (
        "4/dpxb3iT4tp/ZCVEwSnEsLxx0bqyhLpdfOpc6fn7OR8+UClSV5n9E6aSS8MPtnRfp32bAb0nmbRn6H8nd"
        "wLUQ==")


def test_the_futures_signature_follows_the_documented_construction():
    digest = hashlib.sha256(b"a=1" + b"42" + b"/api/v3/sendorder").digest()
    want = base64.b64encode(hmac.new(base64.b64decode(SECRET), digest,
                                     hashlib.sha512).digest()).decode()
    assert futures_signature(SECRET, "/api/v3/sendorder", "42", "a=1") == want


def test_markets_map_book_symbols_to_both_kraken_legs():
    mk = KrakenMarkets.build(["BTCUSDT", "ETHUSDT", "DOGEUSDT", "SOLUSDT"], PAIRS, INSTRUMENTS)
    assert mk.spot_pair["BTCUSDT"] == "XXBTZUSD" and mk.perp["BTCUSDT"] == "PF_XBTUSD"
    assert mk.spot_pair["DOGEUSDT"] == "XXDGZUSD" and "DOGEUSDT" not in mk.perp   # niet tradeable
    assert mk.tradable("SOLUSDT") and not mk.tradable("DOGEUSDT")
    assert mk.perp_decimals["ETHUSDT"] == 3 and mk.spot_min["ETHUSDT"] == 0.002
    with pytest.raises(DataContractError, match="USDT"):
        KrakenMarkets.build(["BTCUSD"], PAIRS, INSTRUMENTS)


def test_a_spot_buy_of_collateral_is_signed_polled_and_moved_to_the_futures_wallet():
    fake = FakeKraken()
    fake.statuses = [{"status": "open", "vol_exec": "0", "price": "0"},
                     {"status": "closed", "vol_exec": "0.5", "price": "40000", "fee": "16"}]
    venue = _venue(fake)
    fill = venue.execute([_order()])[0]
    assert (fill.qty, fill.price, fill.fee_usd) == (0.5, 40000.0, 16.0)
    method, url, headers, body = next(c for c in fake.calls if c[1].endswith("AddOrder"))
    form = _form(body)
    assert form["pair"] == "XXBTZUSD" and form["type"] == "buy"
    assert form["ordertype"] == "market" and form["volume"] == "0.51234567"   # afgerond
    assert headers["API-Sign"] == spot_signature(SECRET, "/0/private/AddOrder",
                                                 "1700000000000", body)
    assert fake.paths().count("/0/private/QueryOrders") == 2
    transfer = _form(fake.calls[-1][3])
    assert fake.paths()[-1] == "/0/private/WalletTransfer"
    assert transfer == {"nonce": "1700000000000", "asset": "XBT", "from": "Spot Wallet",
                        "to": "Futures Wallet", "amount": "0.5"}


def test_selling_collateral_first_withdraws_it_from_the_futures_wallet():
    fake = FakeKraken()
    _venue(fake).execute([_order(side=OrderSide.SELL, reduce_only=True)])
    paths = fake.paths()
    assert paths.index("/derivatives/api/v3/withdrawal") < paths.index("/0/private/AddOrder")
    assert "/0/private/WalletTransfer" not in paths


def test_an_alt_stays_in_the_spot_wallet():
    fake = FakeKraken()
    fake.statuses = [{"status": "closed", "vol_exec": "3", "price": "150", "fee": "0.5"}]
    _venue(fake).execute([_order(sym="SOLUSDT", qty=3.0)])
    assert "/0/private/WalletTransfer" not in fake.paths()


@pytest.mark.parametrize("row, match", [
    ({"status": "open", "vol_exec": "0"}, "geen eindstatus"),
    ({"status": "canceled", "vol_exec": "0", "price": "0"}, "zonder fill"),
    ({"status": "closed", "vol_exec": "0.5", "price": "0"}, "zonder fill"),
])
def test_no_terminal_status_or_no_fill_is_never_a_fill(row, match):
    fake = FakeKraken()
    fake.statuses = [row]
    with pytest.raises(KrakenError, match=match):
        _venue(fake, polls=3).execute([_order()])


def test_a_spot_order_below_krakens_minimum_is_refused():
    with pytest.raises(KrakenError, match="minimale ordergrootte"):
        _venue(FakeKraken()).execute([_order(sym="ETHUSDT", qty=0.001)])


def test_a_perp_short_is_signed_and_its_executions_are_averaged():
    fake = FakeKraken()
    fill = _venue(fake).execute([_order(leg=Leg.PERP, side=OrderSide.SELL, qty=0.50009)])[0]
    assert fill.qty == pytest.approx(0.5) and fill.price == pytest.approx(40080.0)
    _, url, headers, body = next(c for c in fake.calls if c[1].endswith("sendorder"))
    assert url == f"{FUTURES_URL}/api/v3/sendorder"
    form = _form(body)
    assert form["orderType"] == "mkt" and form["symbol"] == "PF_XBTUSD"
    assert form["side"] == "sell" and form["size"] == "0.5" and "reduceOnly" not in form
    assert headers["Authent"] == futures_signature(SECRET, "/api/v3/sendorder",
                                                   "1700000000000", body)


def test_a_perp_buyback_is_reduce_only_and_an_unfilled_one_is_an_error():
    fake = FakeKraken()
    fake.send = {"status": "placed", "orderEvents": []}
    with pytest.raises(KrakenError, match="niet uitgevoerd"):
        _venue(fake).execute([_order(leg=Leg.PERP, side=OrderSide.BUY, reduce_only=True)])
    assert _form(fake.calls[-1][3])["reduceOnly"] == "true"
    with pytest.raises(KrakenError, match="nul contracten"):
        _venue(FakeKraken()).execute([_order(sym="SOLUSDT", leg=Leg.PERP, qty=0.001)])


def test_a_coin_without_both_legs_on_kraken_is_refused():
    with pytest.raises(KrakenError, match="niet beide benen"):
        _venue(FakeKraken()).execute([_order(sym="DOGEUSDT")])


def test_the_book_adds_futures_collateral_to_spot_and_values_the_account():
    fake = FakeKraken()
    book = _venue(fake).book()
    assert book.spot == {"BTCUSDT": 0.5}                 # 0,25 spotwallet + 0,25 onderpand
    assert book.perp_short == {"BTCUSDT": 0.5}
    assert book.equity_usd == pytest.approx(1000.0 + 0.25 * 40000.0 + 10_500.0)


def test_the_book_refuses_a_long_perp_or_a_foreign_position():
    fake = FakeKraken()
    fake.positions = [{"side": "long", "symbol": "PF_XBTUSD", "size": "0.1"}]
    with pytest.raises(KrakenError, match="LONGE perp"):
        _venue(fake).book()
    fake.positions = [{"side": "short", "symbol": "PF_ADAUSD", "size": "1"}]
    with pytest.raises(KrakenError, match="buiten het boek"):
        _venue(fake).book()


def test_a_book_without_spot_holdings_asks_no_prices():
    fake = FakeKraken()
    fake.positions = []
    venue = _venue(fake)
    fake_balance = {"error": [], "result": {"ZUSD": "250"}}
    original = fake.__call__

    def answer(method, url, headers, body, query):
        if url.endswith("/0/private/Balance"):
            fake.calls.append((method, url, dict(headers), body))
            return fake_balance
        return original(method, url, headers, body, query)
    venue.http = answer
    book = venue.book()
    assert book.spot == {"BTCUSDT": 0.25} and "/0/public/Ticker" not in fake.paths()
    assert book.equity_usd == pytest.approx(250.0 + 10_500.0)


def test_dry_run_sends_nothing_private_and_keeps_its_own_book():
    fake = FakeKraken()
    venue = _venue(fake, dry_run=True, dry_equity_usd=1e6)
    fills = venue.execute([_order(), _order(leg=Leg.PERP, side=OrderSide.SELL),
                           _order(leg=Leg.PERP, side=OrderSide.BUY, qty=0.1,
                                  reduce_only=True),
                           _order(side=OrderSide.SELL, qty=0.1)])
    assert [f.price for f in fills] == [40000.0] * 4
    assert all("/0/private" not in p and "sendorder" not in p for p in fake.paths())
    assert [s["leg"] for s in venue.sent] == ["spot", "perp", "perp", "spot"]
    assert venue.sent[2]["reduceOnly"] == "true"
    book = venue.book()
    assert book.spot == {"BTCUSDT": pytest.approx(0.41234567)}
    assert book.perp_short == {"BTCUSDT": pytest.approx(0.4123)}
    assert book.equity_usd == 1e6
    with pytest.raises(KrakenError, match="dry-run"):
        venue._private_spot("Balance", {})
    with pytest.raises(KrakenError, match="dry-run"):
        venue._futures("GET", "/api/v3/accounts")


def test_errors_from_kraken_are_raised_and_demo_uses_the_demo_host():
    def refusing(method, url, headers, body, query):
        if "futures" in url:
            return {"result": "error", "error": "apiLimitExceeded"}
        return {"error": ["EGeneral:Invalid arguments"]}
    venue = KrakenVenue(symbols=["BTCUSDT"], http=refusing, dry_run=False, demo=True)
    with pytest.raises(KrakenError, match="Invalid arguments"):
        venue._public_spot("AssetPairs")
    with pytest.raises(KrakenError, match="Invalid arguments"):
        venue._private_spot("Balance", {})
    with pytest.raises(KrakenError, match="apiLimitExceeded"):
        venue._futures("GET", "/api/v3/instruments")
    assert venue._futures_base() == FUTURES_DEMO_URL and SPOT_URL.startswith("https://")


def test_the_default_transport_uses_requests(monkeypatch: pytest.MonkeyPatch):
    import requests

    from tradebot.carry import kraken

    seen = {}

    class Resp:
        def raise_for_status(self) -> None:
            seen["checked"] = True

        def json(self) -> dict:
            return {"ok": 1}

    def fake_request(method, url, **kw):
        seen.update(method=method, url=url, **kw)
        return Resp()
    monkeypatch.setattr(requests, "request", fake_request)
    out = kraken._requests_http("GET", "https://x.example/y", {"H": "1"}, None, {"q": "2"})
    assert out == {"ok": 1} and seen["checked"] and seen["params"] == {"q": "2"}
    assert seen["timeout"] == 15


def test_the_default_nonce_is_monotone_milliseconds():
    v = KrakenVenue(symbols=["BTCUSDT"])
    a, b = int(v.nonce()), int(v.nonce())
    assert b >= a > 1_600_000_000_000_000
