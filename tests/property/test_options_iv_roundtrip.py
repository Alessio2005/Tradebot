"""Property: prijs(σ) → IV → σ binnen 1e-6, over spot, strike, looptijd, vol en rente."""
from __future__ import annotations

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from tests.unit.options.helpers import inputs_with_t
from tradebot.options.contracts import Right
from tradebot.options.pricing.engines import Method
from tradebot.options.pricing.implied_vol import IVStatus, implied_vol
from tradebot.options.pricing.vanilla import price
from tradebot.options.rates import Carry

pytestmark = pytest.mark.property


@settings(max_examples=60, deadline=None)
@given(moneyness=st.floats(0.8, 1.25), t=st.floats(7 / 365, 1.0), vol=st.floats(0.08, 1.2),
       rate=st.floats(0.0, 0.08), right=st.sampled_from(list(Right)),
       method=st.sampled_from([Method.BAW, Method.EUROPEAN]))
def test_iv_round_trip(moneyness, t, vol, rate, right, method):
    inp = inputs_with_t(t, strike=100.0 * moneyness, right=right, carry=Carry(rate))
    p = price(inp, vol, method)
    vega = (price(inp, vol + 0.01, method) - price(inp, vol - 0.01, method)) / 2
    assume(vega > 1e-3 and p > inp.intrinsic + 0.01)
    res = implied_vol(inp, p, method)
    assert res.status is IVStatus.OK
    assert res.value == pytest.approx(vol, abs=1e-6)
