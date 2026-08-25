"""Dubbele boekhouding — de twee invarianten, en de gevallen die ze breken.

Phase 5, teststrategie stap 1: accounting gaat vóór alles, omdat elke latere
laag (execution, TCA, engine, baseline) zijn getallen hierop stapelt.

De tests zijn zo geschreven dat ze FALEN op de manier waarop de legacy-engines
rekenden: `equity *= (1 + r - costs)` heeft geen kas, geen positie in stuks en
geen tegenrekening, en zou elke test hieronder die naar `cash` of naar de
attributie kijkt niet kunnen doorstaan.
"""
from __future__ import annotations

import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from tradebot.backtest.accounting import (
    AccountingError,
    Fill,
    Ledger,
    Position,
)
from tradebot.utils.failfast import DataContractError

TS = pd.Timestamp("2024-01-01", tz="UTC")
EQ = 100_000.0


def _fill(symbol: str, qty: float, price: float, *, fee: float = 0.0,
          ts: pd.Timestamp = TS, oid: str = "o1") -> Fill:
    return Fill(symbol=symbol, ts=ts, qty=qty, price=price, fee=fee,
                liquidity="taker", order_id=oid)


# =========================================================================== #
# 1. Het contract van Fill
# =========================================================================== #
class TestFillContract:
    def test_a_zero_quantity_fill_is_rejected(self) -> None:
        with pytest.raises(DataContractError):
            _fill("BTCUSDT", 0.0, 100.0)

    def test_a_non_positive_price_is_rejected(self) -> None:
        with pytest.raises(DataContractError):
            _fill("BTCUSDT", 1.0, 0.0)

    def test_a_negative_fee_is_rejected(self) -> None:
        # Een rebate is geen negatieve kost; hij hoort apart geboekt.
        with pytest.raises(DataContractError):
            _fill("BTCUSDT", 1.0, 100.0, fee=-1.0)

    def test_a_naive_timestamp_is_rejected(self) -> None:
        with pytest.raises(DataContractError):
            Fill(symbol="BTCUSDT", ts=pd.Timestamp("2024-01-01"), qty=1.0,
                 price=100.0, fee=0.0, liquidity="taker", order_id="o")

    def test_notional_is_signed(self) -> None:
        assert _fill("BTCUSDT", -2.0, 50.0).notional == pytest.approx(-100.0)


# =========================================================================== #
# 2. De balansidentiteit
# =========================================================================== #
class TestBalanceIdentity:
    def test_a_fresh_ledger_balances(self) -> None:
        snap = Ledger(EQ).mark({}, TS)
        assert snap.equity == pytest.approx(EQ)
        assert snap.assets == pytest.approx(snap.liabilities + snap.equity)

    def test_opening_a_long_moves_cash_into_position_value(self) -> None:
        led = Ledger(EQ)
        led.apply_fill(_fill("BTCUSDT", 2.0, 1_000.0))
        snap = led.mark({"BTCUSDT": 1_000.0}, TS)
        assert snap.cash == pytest.approx(EQ - 2_000.0)
        assert snap.position_value == pytest.approx(2_000.0)
        assert snap.equity == pytest.approx(EQ)
        assert snap.liabilities == pytest.approx(0.0)

    def test_a_short_puts_the_obligation_on_the_liability_side(self) -> None:
        led = Ledger(EQ)
        led.apply_fill(_fill("BTCUSDT", -2.0, 1_000.0))
        snap = led.mark({"BTCUSDT": 1_000.0}, TS)
        assert snap.cash == pytest.approx(EQ + 2_000.0)
        assert snap.position_value == pytest.approx(-2_000.0)
        assert snap.liabilities == pytest.approx(2_000.0)
        assert snap.assets == pytest.approx(snap.liabilities + snap.equity)
        assert snap.equity == pytest.approx(EQ)

    def test_the_identity_survives_a_price_move(self) -> None:
        led = Ledger(EQ)
        led.apply_fill(_fill("BTCUSDT", 2.0, 1_000.0))
        snap = led.mark({"BTCUSDT": 1_500.0}, TS)
        assert snap.equity == pytest.approx(EQ + 1_000.0)
        assert snap.assets == pytest.approx(snap.liabilities + snap.equity)


# =========================================================================== #
# 3. De PnL-attributie
# =========================================================================== #
class TestPnlAttribution:
    def test_an_unrealised_gain_is_attributed(self) -> None:
        led = Ledger(EQ)
        led.apply_fill(_fill("BTCUSDT", 2.0, 1_000.0))
        snap = led.mark({"BTCUSDT": 1_100.0}, TS)
        assert snap.unrealized_pnl == pytest.approx(200.0)
        assert snap.realized_pnl == pytest.approx(0.0)
        assert snap.equity - EQ == pytest.approx(200.0)

    def test_closing_converts_unrealised_into_realised(self) -> None:
        led = Ledger(EQ)
        led.apply_fill(_fill("BTCUSDT", 2.0, 1_000.0))
        led.apply_fill(_fill("BTCUSDT", -2.0, 1_100.0, oid="o2"))
        snap = led.mark({"BTCUSDT": 1_100.0}, TS)
        assert snap.realized_pnl == pytest.approx(200.0)
        assert snap.unrealized_pnl == pytest.approx(0.0)
        assert snap.position_value == pytest.approx(0.0)
        assert snap.cash == pytest.approx(EQ + 200.0)

    def test_a_short_that_falls_makes_money(self) -> None:
        led = Ledger(EQ)
        led.apply_fill(_fill("BTCUSDT", -2.0, 1_000.0))
        led.apply_fill(_fill("BTCUSDT", 2.0, 900.0, oid="o2"))
        snap = led.mark({"BTCUSDT": 900.0}, TS)
        assert snap.realized_pnl == pytest.approx(200.0)
        assert snap.equity == pytest.approx(EQ + 200.0)

    def test_fees_leave_the_book_and_stay_attributed(self) -> None:
        led = Ledger(EQ)
        led.apply_fill(_fill("BTCUSDT", 2.0, 1_000.0, fee=1.10))
        snap = led.mark({"BTCUSDT": 1_000.0}, TS)
        assert snap.fees_paid == pytest.approx(1.10)
        assert snap.equity == pytest.approx(EQ - 1.10)
        # Dit is de test die "geen verdwenen kosten" afdwingt.
        assert snap.equity - EQ == pytest.approx(
            snap.realized_pnl + snap.unrealized_pnl - snap.fees_paid
            - snap.funding_paid
        )

    def test_funding_is_paid_by_the_long_and_received_by_the_short(self) -> None:
        long_led, short_led = Ledger(EQ), Ledger(EQ)
        long_led.apply_fill(_fill("BTCUSDT", 2.0, 1_000.0))
        short_led.apply_fill(_fill("BTCUSDT", -2.0, 1_000.0))
        paid = long_led.apply_funding("BTCUSDT", rate=0.0001, mark_price=1_000.0, ts=TS)
        recv = short_led.apply_funding("BTCUSDT", rate=0.0001, mark_price=1_000.0, ts=TS)
        assert paid == pytest.approx(0.20)
        assert recv == pytest.approx(-0.20)
        assert long_led.mark({"BTCUSDT": 1_000.0}, TS).equity == pytest.approx(EQ - 0.20)
        assert short_led.mark({"BTCUSDT": 1_000.0}, TS).equity == pytest.approx(EQ + 0.20)

    def test_funding_on_a_flat_book_is_a_no_op(self) -> None:
        led = Ledger(EQ)
        assert led.apply_funding("BTCUSDT", rate=0.01, mark_price=1.0, ts=TS) == 0.0
        assert led.mark({}, TS).equity == pytest.approx(EQ)


# =========================================================================== #
# 4. De vier fill-gevallen, inclusief de flip die geen legacy-engine kende
# =========================================================================== #
class TestPositionArithmetic:
    def test_adding_to_a_long_takes_the_weighted_average(self) -> None:
        led = Ledger(EQ)
        led.apply_fill(_fill("BTCUSDT", 1.0, 100.0))
        led.apply_fill(_fill("BTCUSDT", 3.0, 200.0, oid="o2"))
        pos = led.position("BTCUSDT")
        assert pos.qty == pytest.approx(4.0)
        assert pos.avg_price == pytest.approx((100.0 + 600.0) / 4.0)
        assert led.realized_pnl == pytest.approx(0.0)

    def test_partially_reducing_realises_only_the_closed_part(self) -> None:
        led = Ledger(EQ)
        led.apply_fill(_fill("BTCUSDT", 4.0, 100.0))
        realized = led.apply_fill(_fill("BTCUSDT", -1.0, 150.0, oid="o2"))
        assert realized == pytest.approx(50.0)
        pos = led.position("BTCUSDT")
        assert pos.qty == pytest.approx(3.0)
        # De instapprijs van het RESTANT verandert niet bij verkleinen.
        assert pos.avg_price == pytest.approx(100.0)

    def test_a_flip_closes_the_old_position_and_reopens_at_the_fill_price(self) -> None:
        led = Ledger(EQ)
        led.apply_fill(_fill("BTCUSDT", 2.0, 100.0))
        realized = led.apply_fill(_fill("BTCUSDT", -5.0, 120.0, oid="o2"))
        # De hele long van 2 wordt gesloten op +20 per stuk.
        assert realized == pytest.approx(40.0)
        pos = led.position("BTCUSDT")
        assert pos.qty == pytest.approx(-3.0)
        assert pos.avg_price == pytest.approx(120.0)
        snap = led.mark({"BTCUSDT": 120.0}, TS)
        assert snap.unrealized_pnl == pytest.approx(0.0)
        assert snap.equity == pytest.approx(EQ + 40.0)

    def test_an_exact_close_removes_the_symbol_from_the_register(self) -> None:
        led = Ledger(EQ)
        led.apply_fill(_fill("BTCUSDT", 2.0, 100.0))
        led.apply_fill(_fill("BTCUSDT", -2.0, 100.0, oid="o2"))
        assert "BTCUSDT" not in led.positions()
        assert led.position("BTCUSDT") == Position(0.0, 0.0)


# =========================================================================== #
# 5. Fail-fast: geen stille waardering
# =========================================================================== #
class TestNoSilentValuation:
    def test_marking_without_a_price_for_an_open_position_crashes(self) -> None:
        led = Ledger(EQ)
        led.apply_fill(_fill("BTCUSDT", 1.0, 100.0))
        with pytest.raises(DataContractError):
            led.mark({"ETHUSDT": 50.0}, TS)

    def test_a_stale_price_is_not_silently_reused_for_a_new_symbol(self) -> None:
        led = Ledger(EQ)
        led.mark({"BTCUSDT": 100.0}, TS)
        led.apply_fill(_fill("ETHUSDT", 1.0, 50.0))
        with pytest.raises(DataContractError):
            led.mark({"BTCUSDT": 100.0}, TS)

    def test_a_non_finite_mark_is_rejected(self) -> None:
        led = Ledger(EQ)
        led.apply_fill(_fill("BTCUSDT", 1.0, 100.0))
        with pytest.raises(DataContractError):
            led.mark({"BTCUSDT": float("nan")}, TS)

    def test_a_zero_initial_balance_is_rejected(self) -> None:
        with pytest.raises(DataContractError):
            Ledger(0.0)

    def test_verify_rejects_a_tampered_snapshot(self) -> None:
        """De controle is echt en niet decoratief."""
        led = Ledger(EQ)
        snap = led.mark({}, TS)
        broken = type(snap)(**{**snap.as_record(), "ts": snap.ts,
                               "equity": snap.equity + 1_000.0})
        with pytest.raises(AccountingError):
            led.verify(broken)


# =========================================================================== #
# 6. Property: de invarianten houden over willekeurige fill-reeksen
# =========================================================================== #
_qty = st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False)
_price = st.floats(min_value=10.0, max_value=1_000.0, allow_nan=False,
                   allow_infinity=False)
_fee = st.floats(min_value=0.0, max_value=5.0, allow_nan=False, allow_infinity=False)


@given(
    trades=st.lists(st.tuples(_qty, _price, _fee), min_size=1, max_size=25),
    final_mark=_price,
)
@settings(max_examples=200, deadline=None, derandomize=True,
          suppress_health_check=[HealthCheck.too_slow])
def test_both_invariants_hold_over_arbitrary_fill_sequences(
    trades: list[tuple[float, float, float]], final_mark: float
) -> None:
    """Elke reeks fills laat de balans en de attributie sluiten.

    `derandomize=True` is hier bewust gezet. De rest van de suite draait
    non-derandomised, en `reports/phase5_exit_report.md` legt uit waarom dat
    daar een reproduceerbaarheidsprobleem is; een boekhoudingstest hoort niet
    per run een ander voorbeeld te toetsen.
    """
    led = Ledger(EQ)
    for i, (q, p, f) in enumerate(trades):
        if abs(q) <= 1e-9:
            continue
        led.apply_fill(_fill("BTCUSDT", q, p, fee=f, oid=f"o{i}"))
        # mark() controleert beide invarianten en crasht bij afwijking.
        led.mark({"BTCUSDT": p}, TS)
    snap = led.mark({"BTCUSDT": final_mark}, TS)
    assert snap.assets == pytest.approx(snap.liabilities + snap.equity, rel=1e-9,
                                        abs=1e-6)
    assert snap.equity - EQ == pytest.approx(
        snap.realized_pnl + snap.unrealized_pnl - snap.fees_paid - snap.funding_paid,
        rel=1e-9, abs=1e-6,
    )


@given(
    q1=st.floats(min_value=0.1, max_value=5.0),
    p1=_price,
    p2=_price,
)
@settings(max_examples=100, deadline=None, derandomize=True)
def test_a_full_round_trip_realises_exactly_the_price_difference(
    q1: float, p1: float, p2: float
) -> None:
    """Open, sluit: de gerealiseerde winst is `qty * (exit - entry)`, precies."""
    led = Ledger(EQ)
    led.apply_fill(_fill("BTCUSDT", q1, p1))
    led.apply_fill(_fill("BTCUSDT", -q1, p2, oid="o2"))
    snap = led.mark({"BTCUSDT": p2}, TS)
    assert snap.realized_pnl == pytest.approx(q1 * (p2 - p1), rel=1e-9, abs=1e-6)
    assert snap.unrealized_pnl == pytest.approx(0.0, abs=1e-9)
    assert snap.cash == pytest.approx(EQ + q1 * (p2 - p1), rel=1e-9, abs=1e-6)


# =========================================================================== #
# 7. Wat de legacy-engines niet konden
# =========================================================================== #
def test_the_ledger_has_no_equity_setter() -> None:
    """`portfolio/legacy_sizing.py` had `self.equity` publiek muteerbaar.

    Zolang equity een INVOER kan zijn, is de balans niet te controleren: elke
    afwijking kan worden weggeschreven. Hier is equity uitsluitend afgeleid.
    """
    led = Ledger(EQ)
    with pytest.raises(AttributeError):
        led.equity = 1.0  # type: ignore[assignment,misc]
    # En er is ook geen achterdeur om een willekeurig veld bij te zetten.
    with pytest.raises(AttributeError):
        led.some_new_field = 1.0  # type: ignore[attr-defined]
    assert callable(led.equity)


def test_every_fill_is_retained_for_the_audit_trail() -> None:
    led = Ledger(EQ)
    for i in range(5):
        led.apply_fill(_fill("BTCUSDT", 1.0, 100.0 + i, fee=0.1, oid=f"o{i}"))
    assert len(led.fills) == 5
    assert [f.order_id for f in led.fills] == [f"o{i}" for i in range(5)]
    assert all(isinstance(f.as_record(), dict) for f in led.fills)


def test_funding_is_not_capped_by_the_ledger() -> None:
    """De +/-2 % venue-cap hoort in de executielaag, niet in de boekhouding."""
    led = Ledger(EQ)
    led.apply_fill(_fill("BTCUSDT", 1.0, 1_000.0))
    paid = led.apply_funding("BTCUSDT", rate=0.08, mark_price=1_000.0, ts=TS)
    assert paid == pytest.approx(80.0)
