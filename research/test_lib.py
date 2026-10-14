"""Causality and accounting checks for the research lib (run: pytest research/test_lib.py -q)."""
import sys; sys.path.insert(0, "research")
from dataclasses import replace
import numpy as np, pandas as pd
import pytest
from lib.data import load_panel
from lib.engine import run_book, cap_gross
from lib.sleeves import tsmom_score, ts_weights, xs_weights, xs_mom_score, pv_target, fund_z

@pytest.fixture(scope="module")
def panel():
    return load_panel()

def _corrupt_after(p, cut):
    """Replace every price/volume/funding/OI value strictly after `cut` with noise; the past must not move."""
    rng = np.random.default_rng(0)
    def f(df):
        out = df.copy(); m = out.index > cut
        out.loc[m] = out.loc[m].to_numpy() * rng.uniform(0.5, 1.5, size=out.loc[m].shape)
        return out
    close = f(p.close)
    return replace(p, close=close, open=f(p.open), high=f(p.high), low=f(p.low), turnover=f(p.turnover),
                   funding=f(p.funding), oi=f(p.oi), ret=close.pct_change(), sigma_ann=f(p.sigma_ann))

def _full_pipeline(p):
    s_mix = 0.5 * tsmom_score(p) + 0.5 * tsmom_score(p, long_flat=True)
    w_t = pv_target(ts_weights(p, s_mix), p)
    w_x = pv_target(xs_weights(p, xs_mom_score(p)), p)
    return pv_target(cap_gross(0.5 * w_t + 0.5 * w_x), p), fund_z(p)

def test_weights_do_not_depend_on_the_future(panel):
    cut = pd.Timestamp("2024-06-01", tz="UTC")
    w0, z0 = _full_pipeline(panel)
    w1, z1 = _full_pipeline(_corrupt_after(panel, cut))
    pd.testing.assert_frame_equal(w0.loc[:cut], w1.loc[:cut], check_exact=False, rtol=1e-9, atol=1e-12)
    pd.testing.assert_frame_equal(z0.loc[:cut], z1.loc[:cut], check_exact=False, rtol=1e-9, atol=1e-12)

def test_book_accounting_identity(panel):
    w, _ = _full_pipeline(panel)
    b = run_book(w, panel)
    np.testing.assert_allclose(b["net"], b["gross"] + b["funding"] + b["costs"], atol=1e-12)
    assert (b["gross_lev"] <= 4.0 + 1e-9).all()

def test_zero_weights_earn_nothing(panel):
    b = run_book(pd.DataFrame(0.0, index=panel.index, columns=panel.close.columns), panel)
    assert b["net"].abs().max() == 0.0

def test_a_one_day_delay_never_sees_the_return_it_trades(panel):
    """w_t decided at close t earns only from t+1: a weight that is +1 only on day t earns ret[t+1]."""
    w = pd.DataFrame(0.0, index=panel.index, columns=panel.close.columns)
    t = panel.index[1000]; w.loc[t, "BTCUSDT"] = 1.0
    b = run_book(w, panel, cost=0.0, funding=False)
    assert b["net"].loc[t] == 0.0
    assert abs(b["net"].iloc[1001] - panel.ret["BTCUSDT"].iloc[1001]) < 1e-12
