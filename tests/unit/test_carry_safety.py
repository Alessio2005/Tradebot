"""`carry/safety.py` en `carry/alerts.py`: de harde grenzen, de reparatie, de lock, de alarmen."""
from __future__ import annotations

import math
from pathlib import Path

import pytest
import requests

from tradebot.carry.alerts import Alerter
from tradebot.carry.orders import Leg, LegOrder
from tradebot.carry.safety import (
    Deferred,
    Limits,
    SingleInstance,
    check_orders,
    check_prices,
    hedge_gaps,
    load_limits,
    margin_headroom,
    marked_pnl,
    repair_orders,
)
from tradebot.carry.venue import VenueBook
from tradebot.oms.order import OrderSide
from tradebot.utils.failfast import ConfigContractError

ROOT = Path(__file__).resolve().parents[2]
LIM = Limits(max_capital_usd=1e5, max_order_usd=2e4, max_gross=1.5, max_price_deviation=0.05,
             hedge_tolerance_usd=50.0, max_step_loss=0.02, margin_warn=1.6,
             margin_critical=1.2)


def test_the_example_config_has_sane_limits():
    lim = load_limits(ROOT / "conf/live/u1.example.yaml")
    assert lim.max_gross >= 1.4 and lim.initial_margin == 0.5
    assert lim.margin_critical < lim.margin_warn


@pytest.mark.parametrize(("change", "match"), [
    ({"max_order_usd": 0.0}, "eindig en positief"),
    ({"hedge_tolerance_usd": math.nan}, "eindig en positief"),
    ({"margin_critical": 2.0}, "kritieke marge"),
    ({"max_price_deviation": 0.6}, "50 %"),
])
def test_limits_refuse_nonsense(change, match):
    with pytest.raises(ConfigContractError, match=match):
        Limits(**{**LIM.__dict__, **change})


def test_unknown_limits_and_a_config_without_limits_are_refused(tmp_path: Path):
    with pytest.raises(ConfigContractError, match="Onbekende grenzen"):
        Limits.from_mapping({**LIM.__dict__, "max_leverage": 3})
    (tmp_path / "c.yaml").write_text("mode: dry_run\n", encoding="utf-8")
    with pytest.raises(ConfigContractError, match="zonder `limits:`"):
        load_limits(tmp_path / "c.yaml")


def _order(sym: str, usd: float) -> LegOrder:
    return LegOrder(symbol=sym, leg=Leg.SPOT, side=OrderSide.BUY, qty=1.0, notional_usd=usd,
                    ref_price=usd, reduce_only=False)


def test_check_orders_names_every_breach():
    assert check_orders([_order("BTCUSDT", 1e4)], LIM, 5e4) == []
    out = check_orders([_order("BTCUSDT", 3e4), _order("ETHUSDT", 1e3)], LIM, 2e5)
    assert len(out) == 2 and "plafond" in out[0] and "BTCUSDT spot" in out[1]


def test_check_prices_flags_deviation_and_missing_or_broken_quotes():
    dec = {"BTCUSDT": 100.0, "ETHUSDT": 10.0, "XUSDT": 1.0}
    out = check_prices(dec, {"BTCUSDT": 104.0, "ETHUSDT": 11.0, "XUSDT": math.nan}, LIM)
    assert out == ["ETHUSDT: live 11.00 tegen besluit 10.00 (+10.0%)", "XUSDT: geen live koers"]


def test_hedge_gaps_and_the_reduce_only_repair():
    book = VenueBook(spot={"BTCUSDT": 1.0, "ETHUSDT": 10.0, "SOLUSDT": 5.0},
                     perp_short={"BTCUSDT": 1.0, "ETHUSDT": 12.0, "SOLUSDT": 5.0001},
                     equity_usd=1e5)
    gaps = hedge_gaps(book, {"BTCUSDT": 60_000.0, "ETHUSDT": 3_000.0, "SOLUSDT": 150.0}, 50.0)
    assert gaps == {"ETHUSDT": pytest.approx(-2.0)}      # SOL: 0,015 USD verschil is geen gat
    assert hedge_gaps(book, {"BTCUSDT": 6e4, "ETHUSDT": 3e3}, 50.0).keys() == {
        "ETHUSDT", "SOLUSDT"}                             # zonder koers: tel het als gat
    fix = repair_orders({"ETHUSDT": -2.0, "BTCUSDT": 0.5}, {"BTCUSDT": 6e4, "ETHUSDT": 3e3},
                        {"BTCUSDT": 6e4, "ETHUSDT": 3e3})
    assert [(o.symbol, o.leg, o.side, o.qty, o.reduce_only) for o in fix] == [
        ("ETHUSDT", Leg.PERP, OrderSide.BUY, 2.0, True),
        ("BTCUSDT", Leg.SPOT, OrderSide.SELL, 0.5, True)]


def test_margin_headroom_and_marked_pnl():
    book = VenueBook(spot={"BTCUSDT": 1.0}, perp_short={"BTCUSDT": 1.0}, equity_usd=60_000.0)
    assert margin_headroom(book, {"BTCUSDT": 60_000.0}, 0.5) == pytest.approx(2.0)
    assert math.isinf(margin_headroom(VenueBook({}, {}, 1.0), {}, 0.5))
    then = {"BTCUSDT": (60_000.0, 60_010.0)}
    now = {"BTCUSDT": (54_000.0, 54_020.0)}
    # Gehedged: alleen de basis telt; die liep op van +10 naar +20, de short verliest 10.
    assert marked_pnl({"BTCUSDT": 1.0}, {"BTCUSDT": 1.0}, then, now) == pytest.approx(-10.0)
    assert marked_pnl({"BTCUSDT": 1.0}, {}, then, now) == pytest.approx(-6_000.0)
    assert marked_pnl({"ETHUSDT": 1.0}, {}, then, now) == 0.0    # geen markering: niets


def test_single_instance_keeps_a_second_holder_out(tmp_path: Path):
    lock = tmp_path / "s.lock"
    with SingleInstance(lock), pytest.raises(Deferred, match="ander proces"):
        SingleInstance(lock).__enter__()
    with SingleInstance(lock):          # vrijgegeven: weer te nemen
        pass


# --------------------------------------------------------------------------- #
# Alarmen
# --------------------------------------------------------------------------- #
class _Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def test_alerts_go_to_every_channel_and_are_deduplicated():
    sent, logs, clock = [], [], _Clock()
    a = Alerter(webhook_url="https://hooks.example/abc", telegram_token="T0K",
                telegram_chat_id="42", min_interval_s=3600.0,
                post=lambda url, p: sent.append((url, dict(p))), clock=clock,
                log=logs.append)
    assert a.channels == ["webhook", "telegram"]
    assert a.send("warning", "k", "eerste")
    assert [u for u, _ in sent] == ["https://hooks.example/abc",
                                    "https://api.telegram.org/botT0K/sendMessage"]
    assert sent[1][1] == {"chat_id": "42", "text": "[tradebot/u1] WARNING: eerste"}
    assert not a.send("warning", "k", "herhaling")            # binnen het uur: onderdrukt
    clock.t += 901
    assert a.send("critical", "k", "kritiek")                 # kritiek: na een kwartier
    clock.t += 60
    assert not a.send("critical", "k", "kritiek")
    assert a.send("bogus-level", "other", "x") and sent[-1][1]["text"].startswith(
        "[tradebot/u1] WARNING")
    assert any("onderdrukt" in line for line in logs)


def test_a_failing_channel_never_raises_and_never_leaks_its_secret():
    logs: list[str] = []

    def boom(url, payload):
        raise requests.HTTPError(f"500 Server Error for url: {url}")
    a = Alerter(webhook_url="https://hooks.example/SECRET", telegram_token="T0K",
                telegram_chat_id="42", post=boom, log=logs.append)
    assert a.send("critical", "x", "test") is False
    joined = "\n".join(logs)
    assert "faalde" in joined and "SECRET" not in joined and "T0K" not in joined
    quiet = Alerter.from_env({}, log=logs.append)
    assert quiet.channels == [] and quiet.send("info", "y", "alleen het log") is False


def test_the_default_transport_posts_json(monkeypatch: pytest.MonkeyPatch):
    seen = {}

    class R:
        def raise_for_status(self) -> None:
            seen["ok"] = True

    def post(url, json, timeout):
        seen.update(url=url, json=json, timeout=timeout)
        return R()
    monkeypatch.setattr(requests, "post", post)
    a = Alerter.from_env({"ALERT_WEBHOOK_URL": "https://hooks.example/x"}, log=lambda s: None)
    assert a.send("info", "z", "hallo")
    assert seen == {"url": "https://hooks.example/x",
                    "json": {"text": "[tradebot/u1] INFO: hallo"}, "timeout": 10, "ok": True}
