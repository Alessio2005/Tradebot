"""Het QuantLib-deel van de keten over een procespool (`pricing.workers`).

Met workers=2 exact dezelfde keten als in-proces (dataclass-gelijkheid, dus ook de floats),
de pool lui gemaakt, hergebruikt en door `close()` gesloten; `auto` resolveert.
"""
from __future__ import annotations

import multiprocessing
from dataclasses import replace

import pytest

from tests.unit.options.helpers import NOW
from tests.unit.options.test_chain_batch import (
    CARRY,
    DIV_CARRY,
    ON,
    SPOT,
    _div_chain,
    _plain_chain,
)
from tradebot.options.chain import builder as builder_mod
from tradebot.options.chain.builder import Chain, ChainBuilder, solve_quantlib
from tradebot.options.pricing import settings as settings_mod
from tradebot.options.pricing.engines import Method, Use
from tradebot.options.pricing.settings import pricer_settings, resolve_workers
from tradebot.schemas.options import PricingConfig, load_options_config

ONE = replace(ON, workers=1)
TWO = replace(ON, workers=2, pool_min_contracts=0)


def _build(builder: ChainBuilder, rows, carry) -> Chain:
    return builder.build("TST", SPOT, rows, NOW, carry)


@pytest.mark.parametrize(("rows", "carry", "use"), [
    (_div_chain(), DIV_CARRY, Use.SCAN),            # FD met startwaarde, plus een BAW
    (_plain_chain()[:8], CARRY, Use.DECISION),      # binomiaal
])
def test_two_workers_give_exactly_the_in_process_chain(rows, carry, use):
    pooled = ChainBuilder(TWO, 10.0, use=use, spread_iv=True)
    try:
        on = _build(pooled, rows, carry)
        pool = pooled._pool
        assert pool is not None                     # echt door de pool
        assert _build(pooled, rows, carry) == on and pooled._pool is pool   # hergebruikt
    finally:
        pooled.close()
    assert pooled._pool is None
    assert on == _build(ChainBuilder(ONE, 10.0, use=use, spread_iv=True), rows, carry)
    assert {o.method for o in on.options} >= ({Method.FD} if use is Use.SCAN
                                             else {Method.BINOMIAL_LR})


def test_the_pool_only_above_pool_min_contracts_and_close_is_idempotent():
    rows = _div_chain()                              # 6 FD-contracten
    b = ChainBuilder(replace(TWO, pool_min_contracts=6), 10.0)
    _build(b, rows, DIV_CARRY)
    assert b._pool is None                           # 6 is niet meer dan 6
    b.close()
    b.settings = replace(TWO, pool_min_contracts=5)
    _build(b, rows, DIV_CARRY)
    assert b._pool is not None
    b.close()
    b.close()
    assert b._pool is None


def test_solve_quantlib_runs_in_process():
    assert solve_quantlib(ONE, []) == []


def test_auto_resolves_to_at_most_four_cores(monkeypatch):
    monkeypatch.setattr(settings_mod.os, "cpu_count", lambda: 16)
    assert resolve_workers("auto") == 4
    monkeypatch.setattr(settings_mod.os, "cpu_count", lambda: 2)
    assert resolve_workers("auto") == 2
    monkeypatch.setattr(settings_mod.os, "cpu_count", lambda: None)
    assert resolve_workers("auto") == 1
    assert resolve_workers(7) == 7
    base = load_options_config().pricing
    assert base.workers == "auto" and base.pool_min_contracts == 16
    cfg = base.model_copy(update={"workers": 3, "pool_min_contracts": 8})
    assert (pricer_settings(cfg).workers, pricer_settings(cfg).pool_min_contracts) == (3, 8)
    with pytest.raises(ValueError, match="workers"):
        PricingConfig(risk_free_rate=0.04, workers=0)


def test_the_start_method_is_fork_free_where_possible(monkeypatch):
    assert builder_mod._mp_context().get_start_method() == "forkserver"
    monkeypatch.setattr(builder_mod.multiprocessing, "get_all_start_methods", lambda: ["spawn"])
    assert builder_mod._mp_context() is multiprocessing.get_context()
