"""Spec §6.2: één kostendefinitie, drie toepassingen."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tradebot.execution.impact_model import square_root_impact
from tradebot.execution.trade_costs import TradeCostModel, load_impact_params
from tradebot.schemas.config import ExecutionConfig, load_config
from tradebot.utils.failfast import DataContractError

ROOT = Path(__file__).resolve().parents[2]
EXEC = load_config(ROOT / "conf/execution/fees.yaml", ExecutionConfig)


def _model(impact=None) -> TradeCostModel:
    return TradeCostModel.from_config(EXEC, stop_slippage_bps=5.0, impact=impact)


def test_the_fixed_part_is_13_bps_from_the_fee_config() -> None:
    m = _model()
    assert m.per_leg == pytest.approx(0.00065)
    assert m.fixed_round_trip == pytest.approx(0.0013)
    assert m.stop_slippage == pytest.approx(0.0005)


def test_label_costs_add_the_funding_of_the_held_bars_only() -> None:
    funding = np.zeros(8)
    funding[1] = 0.05   # de entrybar zelf: de positie hield die bar niet vast
    funding[3] = 0.001  # binnen de houdtijd
    funding[5] = 0.07   # na de exit
    n = 1
    kw = dict(funding=funding, adv=np.full(8, 1e9), sigma_daily=np.full(8, 0.03),
              reference_notional=np.full(n, 1e4))
    long_cost = _model().label_costs([1.0], [1], [4], **kw)
    short_cost = _model().label_costs([-1.0], [1], [4], **kw)
    assert long_cost[0] == pytest.approx(0.0013 + 0.001)
    assert short_cost[0] == pytest.approx(0.0013 - 0.001)


def test_missing_funding_inside_the_hold_crashes() -> None:
    funding = np.zeros(8)
    funding[2] = np.nan
    with pytest.raises(DataContractError, match="Funding"):
        _model().label_costs([1.0], [1], [4], funding=funding, adv=np.full(8, 1e9),
                             sigma_daily=np.full(8, 0.03), reference_notional=np.full(1, 1e4))


def test_the_ex_ante_bound_never_counts_funding_income() -> None:
    m = _model()
    kw = dict(horizon_bars=10, max_notional=8e4, adv=1e9, sigma_daily=0.03)
    receives = m.ex_ante_cost(side=-1.0, funding_recent_mean=0.0002, **kw)
    pays = m.ex_ante_cost(side=1.0, funding_recent_mean=0.0002, **kw)
    assert receives == pytest.approx(0.0013 + 0.0005)
    assert pays == pytest.approx(0.0013 + 0.0005 + 10 * 0.0002)


def test_impact_is_the_square_root_model_or_zero_without_parameters() -> None:
    assert _model().impact_fraction(1e6, adv=1e9, sigma_daily=0.03) == 0.0
    params = load_impact_params(ROOT / "conf/execution/impact.yaml")
    m = _model(params)
    expected = square_root_impact(order_notional=1e6, adv_notional=1e9, sigma_daily=0.03,
                                  params=params).impact_fraction
    assert m.impact_fraction(1e6, adv=1e9, sigma_daily=0.03) == pytest.approx(expected)
    assert m.impact_fraction(-1e6, adv=1e9, sigma_daily=0.03) == pytest.approx(expected)


def test_impact_without_a_valid_adv_crashes() -> None:
    m = _model(load_impact_params(ROOT / "conf/execution/impact.yaml"))
    with pytest.raises(DataContractError, match="ADV"):
        m.impact_fraction(1e6, adv=float("nan"), sigma_daily=0.03)
