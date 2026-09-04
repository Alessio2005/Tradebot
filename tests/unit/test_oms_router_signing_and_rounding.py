# tests/unit/test_oms_router_signing_and_rounding.py
"""De twee zuivere functies op de live-orderweg, en waarom zij ertoe doen.

Phase 9, stap 11. `oms/router.py` staat op **27,9 % dekking over 315 LOC** en is
daarmee de zwakste plek van het `oms/`-pakket (69,8 %, net onder de drempel).
Het grootste deel van dat bestand is async HTTP naar Bybit; dat vraagt een
nagebouwde beurs en is geen unit-test.

Twee functies zijn dat wel, en zij dragen het scherpste risico van het hele
bestand omdat zij **stil** verkeerd kunnen zijn:

* `_round_to_step` — Bybit weigert orders waarvan de hoeveelheid de `qtyStep`
  schendt, dus rondt de router lokaal af. Naar BENEDEN: naar boven afronden
  maakt de order groter dan de risicolaag heeft toegestaan. Een float-afronding
  in plaats van `Decimal` levert precies dat op bij stappen als 0,1.
* `_auth_headers` — de handtekening is
  `HMAC_SHA256(secret, ts + api_key + recv_window + payload)`. Verwissel twee
  velden in die volgorde en de authenticatie faalt bij de beurs, niet hier. Er
  is geen lokale poort die dat merkt.

WAT DEZE TESTS NIET DOEN
========================
Zij raken de divergentie uit `PROJECT_STATE.md` §3.2 niet aan. Dat
`oms/router.py::place_order` geen risicobesluit vraagt, is fence 5 van deze
fase: in kaart brengen, niet oplossen. Hier wordt uitsluitend vastgelegd wat de
bestaande code doet.
"""
from __future__ import annotations

import hashlib
import hmac
from decimal import Decimal

import pytest

from tradebot.oms.router import _RECV_WINDOW, OrderRouter


@pytest.fixture
def router(monkeypatch: pytest.MonkeyPatch) -> OrderRouter:
    monkeypatch.setenv("BYBIT_API_KEY", "test-key")
    monkeypatch.setenv("BYBIT_API_SECRET", "test-secret")
    return OrderRouter(paper_oms=None, live_mode=False)


# ═════════════════════════════════════════════════════════════════════════════
# _round_to_step — naar beneden, en exact
# ═════════════════════════════════════════════════════════════════════════════
class TestRoundToStep:
    @pytest.mark.parametrize(
        "qty,step,expected",
        [
            (1.27, "0.1", "1.2"),
            (1.20, "0.1", "1.2"),
            (0.09, "0.1", "0.0"),
            (123.456, "0.001", "123.456"),
            (123.4567, "0.001", "123.456"),
            (7.0, "1", "7"),
            (7.9, "1", "7"),
        ],
    )
    def test_it_floors_to_the_step(self, qty: float, step: str, expected: str) -> None:
        assert OrderRouter._round_to_step(qty, step) == Decimal(expected)

    def test_it_never_rounds_up(self) -> None:
        """Naar boven afronden vergroot de order buiten het risicobesluit om."""
        for qty in (0.1999, 1.9999, 12.34999):
            out = OrderRouter._round_to_step(qty, "0.01")
            assert out <= Decimal(str(qty))

    def test_a_float_representable_only_in_binary_still_floors_correctly(self) -> None:
        """`Decimal(str(qty))` is de reden dat dit klopt.

        In binaire drijvende komma is 0,1 + 0,2 gelijk aan 0,30000000000000004.
        Deelt de router die waarde door stap 0,1 als float, dan geeft de
        vloerdeling 3 -- maar op 2,9999999999999996 zou zij 2 geven. De
        Decimal-route maakt de uitkomst onafhankelijk van die representatie.
        """
        assert OrderRouter._round_to_step(0.1 + 0.2, "0.1") == Decimal("0.3")

    def test_a_step_of_zero_returns_the_quantity_untouched(self) -> None:
        """Deling door nul mag geen crash zijn; de beurs valideert dan zelf."""
        assert OrderRouter._round_to_step(1.234, "0") == Decimal("1.234")

    def test_a_negative_step_is_treated_as_no_step(self) -> None:
        assert OrderRouter._round_to_step(1.234, "-0.1") == Decimal("1.234")

    def test_the_result_is_a_decimal_not_a_float(self) -> None:
        """Het type is onderdeel van het contract: str(Decimal) gaat naar de beurs."""
        assert isinstance(OrderRouter._round_to_step(1.27, "0.1"), Decimal)

    # ── negatieve controle ───────────────────────────────────────────────────
    def test_a_naive_float_implementation_disagrees_on_a_real_case(self) -> None:
        """Bewijs dat de Decimal-route iets oplost in plaats van iets te lijken.

        Zonder deze controle zouden de asserties hierboven net zo groen zijn op
        een implementatie die met floats werkt.
        """
        qty, step = 2.67, 0.01
        naive = (qty // step) * step                       # 2.66 in plaats van 2.67
        exact = OrderRouter._round_to_step(qty, "0.01")

        assert exact == Decimal("2.67")
        assert round(naive, 10) != float(exact), (
            "de naïeve float-variant hoort hier af te wijken"
        )


# ═════════════════════════════════════════════════════════════════════════════
# _auth_headers — de handtekening die pas bij de beurs faalt
# ═════════════════════════════════════════════════════════════════════════════
class TestAuthHeaders:
    def test_the_signature_is_hmac_over_timestamp_key_window_payload_in_that_order(
        self, router: OrderRouter
    ) -> None:
        """De volgorde IS het contract; hij is hier onafhankelijk nagerekend."""
        payload = '{"symbol":"BTCUSDT","qty":"0.01"}'
        headers = router._auth_headers(payload)

        expected = hmac.new(
            b"test-secret",
            (headers["X-BAPI-TIMESTAMP"] + "test-key" + _RECV_WINDOW + payload).encode(),
            hashlib.sha256,
        ).hexdigest()
        assert headers["X-BAPI-SIGN"] == expected

    def test_every_header_bybit_requires_is_present(self, router: OrderRouter) -> None:
        headers = router._auth_headers("")
        assert set(headers) == {
            "X-BAPI-API-KEY", "X-BAPI-TIMESTAMP", "X-BAPI-RECV-WINDOW",
            "X-BAPI-SIGN", "Content-Type",
        }
        assert headers["X-BAPI-API-KEY"] == "test-key"
        assert headers["X-BAPI-RECV-WINDOW"] == _RECV_WINDOW
        assert headers["Content-Type"] == "application/json"

    def test_the_timestamp_is_milliseconds(self, router: OrderRouter) -> None:
        """Bybit verwacht ms; seconden geeft een recv_window-afwijzing."""
        ts = int(router._auth_headers("")["X-BAPI-TIMESTAMP"])
        assert ts > 1_000_000_000_000

    def test_a_different_payload_yields_a_different_signature(
        self, router: OrderRouter
    ) -> None:
        """Anders zou de handtekening de order niet binden."""
        a = router._auth_headers('{"qty":"1"}')["X-BAPI-SIGN"]
        b = router._auth_headers('{"qty":"2"}')["X-BAPI-SIGN"]
        assert a != b

    def test_a_different_secret_yields_a_different_signature(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("BYBIT_API_KEY", "test-key")
        monkeypatch.setenv("BYBIT_API_SECRET", "secret-one")
        one = OrderRouter(paper_oms=None)._auth_headers("x")["X-BAPI-SIGN"]
        monkeypatch.setenv("BYBIT_API_SECRET", "secret-two")
        two = OrderRouter(paper_oms=None)._auth_headers("x")["X-BAPI-SIGN"]
        assert one != two

    # ── negatieve controle ───────────────────────────────────────────────────
    def test_a_swapped_field_order_produces_a_different_signature(
        self, router: OrderRouter
    ) -> None:
        """Bewijs dat de volgorde-assertie werkelijk de volgorde toetst."""
        payload = "abc"
        headers = router._auth_headers(payload)
        ts = headers["X-BAPI-TIMESTAMP"]

        swapped = hmac.new(
            b"test-secret",
            ("test-key" + ts + _RECV_WINDOW + payload).encode(),   # key en ts omgewisseld
            hashlib.sha256,
        ).hexdigest()
        assert headers["X-BAPI-SIGN"] != swapped


class TestRouterConstruction:
    def test_it_starts_in_paper_mode_by_default(self, router: OrderRouter) -> None:
        assert router._live_mode is False

    def test_the_idempotency_cache_and_filter_cache_start_empty(
        self, router: OrderRouter
    ) -> None:
        """Wave 15 P0-5.4: de cache voorkomt dubbele plaatsing bij een retry."""
        assert router._order_id_cache == {}
        assert router._symbol_filters == {}

    def test_missing_credentials_give_empty_strings_not_a_crash(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Paper-modus hoort te werken zonder sleutels in de omgeving."""
        monkeypatch.delenv("BYBIT_API_KEY", raising=False)
        monkeypatch.delenv("BYBIT_API_SECRET", raising=False)
        r = OrderRouter(paper_oms=None)
        assert r._api_key == ""
        assert r._api_secret == ""
