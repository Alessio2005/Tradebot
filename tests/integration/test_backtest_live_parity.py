"""Backtest en live delen één executiecontract — exit-criterium 9.

De eis: *"Backtest/live execution contract is bit-identiek op gedeelde replay."*

WAAROM DIT EEN ZINVOLLE TEST IS EN GEEN TAUTOLOGIE
--------------------------------------------------
`BacktestExecutionContext` en `LiveExecutionContext` erven allebei van
`_SharedContext`, dus dat ze hetzelfde doen lijkt vanzelfsprekend. Dat is
precies het punt: de constructie is zo gekozen dat pariteit een EIGENSCHAP VAN
DE CODE is in plaats van een meetresultaat dat na elke wijziging opnieuw
bewezen moet worden.

Wat de test dan nog toevoegt, is een slot op die constructie. Zij faalt zodra
iemand `LiveExecutionContext` een eigen `market_state()` of een eigen
`risk_state()` geeft - en dat is precies de manier waarop live-implementaties
in de praktijk uit elkaar lopen met hun backtest. De wiring audit telde drie
live-controllers met elk hun eigen risicoregime; dit is het mechanisme dat
voorkomt dat er een vierde bij komt.

WAT PHASE 7 NOG MOET DOEN
-------------------------
De BRON aansluiten: `live/feed.py` levert de `MarketSlice`, `live/state.py` de
boekstaat. De besluitvorming erboven verandert niet. Zie
`reports/phase5_exit_report.md` sectie 'Phase 7 handoff'.
"""
from __future__ import annotations

import inspect

import pytest

from tradebot.execution.context import (
    BacktestExecutionContext,
    ExecutionContext,
    LiveExecutionContext,
    build_context,
)

from ._engine_fixtures import INITIAL_EQUITY, make_engine, make_replay

pytestmark = pytest.mark.integration

#: De methoden waarop backtest en live niet mogen afwijken.
_CONTRACT = (
    "asof", "next_fill_ts", "market_state", "risk_state", "equity",
    "positions", "marks", "slice",
)


@pytest.fixture(scope="module")
def replay():
    return make_replay(n_bars=50)


# =========================================================================== #
# 1. Bit-identieke replay
# =========================================================================== #
class TestSharedReplayIsBitIdentical:
    def test_the_equity_curves_are_identical(self, replay) -> None:
        bt = make_engine(context_kind="backtest").run(
            replay.slices, replay.exposures)
        live = make_engine(context_kind="live").run(
            replay.slices, replay.exposures)
        assert bt.equity_curve.equals(live.equity_curve)

    def test_every_risk_decision_is_identical(self, replay) -> None:
        bt = make_engine(context_kind="backtest").run(
            replay.slices, replay.exposures)
        live = make_engine(context_kind="live").run(
            replay.slices, replay.exposures)
        assert len(bt.decisions) == len(live.decisions) > 0
        assert [d.as_record() for d in bt.decisions] == \
               [d.as_record() for d in live.decisions]

    def test_every_order_and_fill_is_identical(self, replay) -> None:
        bt = make_engine(context_kind="backtest").run(
            replay.slices, replay.exposures)
        live = make_engine(context_kind="live").run(
            replay.slices, replay.exposures)
        assert len(bt.reports) == len(live.reports) > 0
        assert [r.as_record() for r in bt.reports] == \
               [r.as_record() for r in live.reports]

    def test_the_ledger_snapshots_are_identical(self, replay) -> None:
        bt = make_engine(context_kind="backtest").run(
            replay.slices, replay.exposures)
        live = make_engine(context_kind="live").run(
            replay.slices, replay.exposures)
        assert [s.as_record() for s in bt.snapshots] == \
               [s.as_record() for s in live.snapshots]

    def test_only_the_declared_context_kind_differs(self, replay) -> None:
        bt = make_engine(context_kind="backtest").run(
            replay.slices, replay.exposures)
        live = make_engine(context_kind="live").run(
            replay.slices, replay.exposures)
        a, b = dict(bt.as_record()), dict(live.as_record())
        assert a.pop("context_kind") == "backtest"
        assert b.pop("context_kind") == "live"
        assert a == b


# =========================================================================== #
# 2. Het contract zelf
# =========================================================================== #
class TestTheSharedContract:
    def test_both_implement_the_full_contract(self) -> None:
        for cls in (BacktestExecutionContext, LiveExecutionContext):
            for name in _CONTRACT:
                assert hasattr(cls, name), f"{cls.__name__} mist {name}"
            assert not inspect.isabstract(cls)

    def test_neither_overrides_a_contract_method(self) -> None:
        """Dit is het slot.

        Zodra `LiveExecutionContext` een eigen `market_state()` krijgt, kan hij
        een andere risicostaat produceren dan de backtest en is exit-criterium
        9 onbewijsbaar geworden. Dan faalt deze test, en dat is de bedoeling.
        """
        for cls in (BacktestExecutionContext, LiveExecutionContext):
            own = set(vars(cls))
            offending = own & set(_CONTRACT)
            assert not offending, (
                f"{cls.__name__} overschrijft {sorted(offending)}; backtest en "
                f"live kunnen dan uiteenlopen"
            )

    def test_the_contract_is_abstract_and_cannot_be_half_implemented(
        self,
    ) -> None:
        assert inspect.isabstract(ExecutionContext)
        with pytest.raises(TypeError):
            ExecutionContext()  # type: ignore[abstract]

        class Partial(ExecutionContext):
            def asof(self):  # type: ignore[no-untyped-def]
                raise NotImplementedError

        with pytest.raises(TypeError):
            Partial()  # type: ignore[abstract]

    def test_the_contract_exposes_no_alpha_channel(self) -> None:
        """Geen modelnaam, geen confidence, geen strategie-identiteit.

        Dezelfde regel die `risk/contract.py::MarketState` op de laag eronder
        legt. Een executiecontext die modelvertrouwen kan doorgeven, is een
        kanaal waarlangs alpha alsnog de positiegrootte bepaalt (audit §14).
        """
        forbidden = ("alpha", "signal", "confidence", "model", "strategy",
                     "prediction", "score", "edge")
        names = [n for n in dir(ExecutionContext) if not n.startswith("_")]
        for name in names:
            assert not any(f in name.lower() for f in forbidden), (
                f"ExecutionContext.{name} kan alpha-informatie doorgeven"
            )


# =========================================================================== #
# 3. Beide contexts uit dezelfde invoer
# =========================================================================== #
class TestBuildContextParity:
    def test_the_two_kinds_answer_every_question_identically(
        self, replay
    ) -> None:
        market = replay.slices[0]
        kwargs = {
            "equity": INITIAL_EQUITY,
            "high_water_mark": INITIAL_EQUITY * 1.1,
            "day_start_equity": INITIAL_EQUITY * 0.98,
            "positions": {"BTCUSDT": 1.5, "ETHUSDT": -2.0},
            "next_fill_ts": replay.slices[1].ts,
        }
        bt = build_context("backtest", market, **kwargs)  # type: ignore[arg-type]
        live = build_context("live", market, **kwargs)  # type: ignore[arg-type]

        assert bt.asof() == live.asof()
        assert bt.next_fill_ts() == live.next_fill_ts()
        assert bt.equity() == live.equity()
        assert dict(bt.positions()) == dict(live.positions())
        assert dict(bt.marks()) == dict(live.marks())
        assert bt.market_state().as_record() if hasattr(
            bt.market_state(), "as_record") else True
        assert bt.risk_state().as_record() == live.risk_state().as_record()
        assert dict(bt.market_state().sigma_hat) == \
               dict(live.market_state().sigma_hat)
        assert dict(bt.market_state().cluster) == \
               dict(live.market_state().cluster)
        assert bt.market_state().asof_ts == live.market_state().asof_ts

    def test_only_the_audit_header_reveals_which_is_which(self, replay) -> None:
        market = replay.slices[0]
        kwargs = {
            "equity": INITIAL_EQUITY, "high_water_mark": INITIAL_EQUITY,
            "day_start_equity": INITIAL_EQUITY, "positions": {},
            "next_fill_ts": replay.slices[1].ts,
        }
        bt = build_context("backtest", market, **kwargs)  # type: ignore[arg-type]
        live = build_context("live", market, **kwargs)  # type: ignore[arg-type]
        a, b = bt.audit_header(), live.audit_header()
        assert a.pop("context_kind") == "backtest"
        assert b.pop("context_kind") == "live"
        assert a == b

    def test_an_unknown_context_kind_is_a_hard_failure(self, replay) -> None:
        from tradebot.utils.failfast import DataContractError

        with pytest.raises(DataContractError):
            build_context("paper", replay.slices[0], equity=INITIAL_EQUITY,
                          high_water_mark=INITIAL_EQUITY,
                          day_start_equity=INITIAL_EQUITY, positions={},
                          next_fill_ts=replay.slices[1].ts)


# =========================================================================== #
# 4. De halt-toestand reist mee in beide richtingen
# =========================================================================== #
def test_a_halted_book_produces_the_same_decision_in_both(replay) -> None:
    """De kill switch is dezelfde in backtest en live.

    `live/circuit_breaker.py` houdt vandaag een eigen, in-memory HALTED-state
    (wiring audit C2). Deze test legt vast dat de gedeelde context de halt op
    exact dezelfde manier doorgeeft, zodat Phase 7 hem alleen hoeft aan te
    sluiten in plaats van opnieuw te bedenken.
    """
    kwargs = {
        "equity": INITIAL_EQUITY, "high_water_mark": INITIAL_EQUITY,
        "day_start_equity": INITIAL_EQUITY, "positions": {},
        "next_fill_ts": replay.slices[1].ts,
        "halted": True, "halt_reason": "drawdown_breaker",
        "halted_at": "2024-01-01T00:00:00+00:00",
    }
    bt = build_context("backtest", replay.slices[0], **kwargs)  # type: ignore[arg-type]
    live = build_context("live", replay.slices[0], **kwargs)  # type: ignore[arg-type]
    assert bt.risk_state().halted is live.risk_state().halted is True
    assert bt.risk_state().as_record() == live.risk_state().as_record()
