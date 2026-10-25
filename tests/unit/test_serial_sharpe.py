"""De Sharpe van een autogecorreleerde reeks (`validation/serial_sharpe.py`).

Wat moet gelden: zonder autocorrelatie vallen alle schattingen samen met de bevroren
iid-Sharpe; met een persistente reeks (AR(1), phi = 0,6) zakt de Lo-Sharpe naar
sqrt((1 - phi) / (1 + phi)) = 0,5 keer de iid-Sharpe, en de maand-Sharpe mee.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradebot.utils.failfast import DataContractError
from tradebot.validation.serial_sharpe import (
    autocorrelations,
    lo_annualized_sharpe,
    period_returns,
    period_sharpe,
    serial_sharpe,
)


def _daily(x: np.ndarray) -> pd.Series:
    return pd.Series(x, index=pd.date_range("2000-01-01", periods=x.size, freq="D", tz="UTC"))


def _ar1(phi: float, n: int, seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    e = rng.standard_normal(n) * 0.01
    x = np.empty(n)
    x[0] = e[0]
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    return _daily(x + 0.002)


def test_without_autocorrelation_every_estimate_agrees():
    r = _daily(np.random.default_rng(1).standard_normal(20_000) * 0.01 + 0.0005)
    s = serial_sharpe(r, bars_per_year=365, lags=(10, 60))
    assert abs(s.rho[0]) < 0.03
    for v in (s.sharpe_lo[10], s.sharpe_lo[60], s.sharpe_monthly):
        assert v == pytest.approx(s.sharpe_iid, rel=0.2)


def test_a_persistent_series_is_deflated_by_the_lo_factor():
    r = _ar1(0.6, 40_000, seed=3)
    s = serial_sharpe(r, bars_per_year=365, lags=(120,))
    assert s.rho[0] == pytest.approx(0.6, abs=0.02)
    assert s.sharpe_lo[120] / s.sharpe_iid == pytest.approx(0.5, rel=0.1)
    assert s.sharpe_monthly < 0.7 * s.sharpe_iid
    assert s.ann_vol_monthly > 1.5 * s.ann_vol_iid
    rec = s.as_record()
    assert rec["sharpe_lo120"] == s.sharpe_lo[120] and len(rec["rho_1_3"]) == 3


def test_period_returns_compound():
    r = _daily(np.full(62, 0.001))
    m = period_returns(r, "ME")
    assert m.iloc[0] == pytest.approx(1.001 ** 31 - 1.0)
    assert len(period_returns(r, "QE")) == 1


def test_too_few_or_flat_periods_give_no_sharpe():
    assert np.isnan(period_sharpe(_daily(np.full(100, 0.001)), "ME"))      # 4 maanden
    assert np.isnan(period_sharpe(_daily(np.zeros(400)), "ME"))            # sd = 0
    assert np.isnan(lo_annualized_sharpe(np.zeros(50), periods_per_year=365, lags=5))
    flat = serial_sharpe(_daily(np.zeros(3)), bars_per_year=365, lags=(10,))
    assert np.isnan(flat.sharpe_iid) and all(np.isnan(flat.rho)) and np.isnan(flat.sharpe_lo[10])
    one = serial_sharpe(_daily(np.zeros(1)), bars_per_year=365, lags=(10,))
    assert np.isnan(one.ann_vol_monthly)


def test_a_constant_series_has_zero_autocorrelation():
    assert np.array_equal(autocorrelations(np.ones(10), 3), np.zeros(3))


def test_the_lags_are_capped_at_the_horizon():
    r = _ar1(0.3, 2_000, seed=5)
    assert lo_annualized_sharpe(r, periods_per_year=12, lags=500) == pytest.approx(
        lo_annualized_sharpe(r, periods_per_year=12, lags=11))


@pytest.mark.parametrize("call, match", [
    (lambda: autocorrelations(np.ones(3), 5), "Te weinig observaties"),
    (lambda: period_returns(pd.Series([0.1, 0.2]), "ME"), "tijdindex"),
    (lambda: period_returns(_daily(np.ones(5)), "W"), "Onbekende periode"),
])
def test_contract_breaches_are_refused(call, match):
    with pytest.raises(DataContractError, match=match):
        call()
