"""Benchmark: een keten van 2.000 contracten (BAW-waardering plus IV) in minder dan 2 s."""
from __future__ import annotations

import time

import pytest

from tests.unit.options.helpers import inputs
from tradebot.options.contracts import Right
from tradebot.options.pricing.engines import Method
from tradebot.options.pricing.implied_vol import IVStatus, implied_vol
from tradebot.options.pricing.vanilla import price


@pytest.mark.benchmark
@pytest.mark.slow
def test_a_chain_of_2000_contracts_prices_in_under_two_seconds():
    chain = [inputs(strike=70.0 + 0.03 * i, right=Right.CALL if i % 2 else Right.PUT)
             for i in range(2000)]
    # Quotes zoals de markt ze geeft: op centen, en minstens één tick.
    quotes = [max(round(price(c, 0.25, Method.BAW), 2), 0.01) for c in chain]
    start = time.perf_counter()
    results = [implied_vol(c, q, Method.BAW) for c, q in zip(chain, quotes, strict=True)]
    elapsed = time.perf_counter() - start
    assert sum(r.status is IVStatus.OK for r in results) > 1500
    assert elapsed < 2.0, f"{elapsed:.2f} s"
