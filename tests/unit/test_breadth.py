"""Breedte per constructie, met interval. Fase 11, breedte en tijdschaal, stap 2.

Vier eigenschappen dragen de meting:

1. de constructies doen wat hun naam zegt: dollar-neutraal sommeert per bar tot
   nul en verliest precies één rang; de bèta-hedge laat een residu achter dat
   ongecorreleerd is met het mandje;
2. het aantal weddenschappen draagt een interval (R-8), en het puntgetal ligt
   daarbinnen;
3. het ontwerpeffect staat ernaast onder zijn eigen naam, en wordt niet
   stilzwijgend ingevuld wanneer het niet gedefinieerd is;
4. gelijke invoer en gelijke seed geven een bit-identiek resultaat (R-5).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tradebot.utils.failfast import DataContractError
from tradebot.validation.breadth import (
    BreadthConfig,
    breadth_config,
    breadth_measurement,
    construct,
    independent_bets,
)
from tradebot.validation.inference import effective_breadth

ROOT = Path(__file__).resolve().parents[2]
IDX = pd.date_range("2022-01-01", periods=800, freq="D", tz="UTC")


def _one_factor(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    market = rng.standard_normal((len(IDX), 1))
    loadings = np.array([1.0, 1.1, 1.3, 1.2, 0.9, 1.4])
    values = 0.02 * market * loadings + 0.01 * rng.standard_normal((len(IDX), 6))
    return pd.DataFrame(values, index=IDX, columns=list("ABCDEF"))


def test_the_committed_config_loads_and_names_only_known_constructions() -> None:
    cfg = breadth_config(ROOT / "conf" / "research" / "breadth.yaml")
    assert isinstance(cfg, BreadthConfig)
    assert set(cfg.constructions) == {"directional", "dollar_neutral", "beta_hedged_ew"}
    assert cfg.signal_clock.iact_window_c > 0
    assert list(cfg.wall.horizons_bars) == sorted(cfg.wall.horizons_bars)


def test_directional_is_the_identity() -> None:
    panel = _one_factor(1)
    pd.testing.assert_frame_equal(construct(panel, "directional"), panel)


def test_dollar_neutral_sums_to_zero_and_loses_one_rank() -> None:
    residual = construct(_one_factor(2), "dollar_neutral")
    assert np.allclose(residual.sum(axis=1), 0.0, atol=1e-15)
    assert np.linalg.matrix_rank(residual.to_numpy()) == residual.shape[1] - 1


def test_the_beta_hedge_leaves_a_residual_orthogonal_to_the_basket() -> None:
    panel = _one_factor(3)
    residual = construct(panel, "beta_hedged_ew")
    basket = panel.mean(axis=1)
    for column in residual:
        assert abs(np.cov(residual[column], basket)[0, 1]) < 1e-15


def test_an_unknown_construction_is_refused() -> None:
    with pytest.raises(DataContractError):
        construct(_one_factor(4), "market_neutral_by_magic")


def test_the_point_estimate_lies_inside_its_interval() -> None:
    row = breadth_measurement(
        construct(_one_factor(5), "dollar_neutral"),
        construction="dollar_neutral", n_boot=400, seed=7, ci_level=0.95,
        block_length=None,
    )
    low, high = row.independent_bets_ci
    assert low <= row.independent_bets <= high
    assert row.independent_bets <= row.rank + 1e-9
    assert row.n_obs == len(IDX) and row.n_names == 6


def test_the_design_effect_is_reported_under_its_own_name() -> None:
    panel = construct(_one_factor(6), "directional")
    row = breadth_measurement(
        panel, construction="directional", n_boot=200, seed=8, ci_level=0.95,
        block_length=None,
    )
    corr = panel.corr().to_numpy()
    assert row.design_effect == pytest.approx(effective_breadth(corr))
    assert row.independent_bets == pytest.approx(independent_bets(corr))
    assert row.design_effect_undefined_replicates == 0
    record = row.to_dict()
    assert "design_effect" in record and "independent_bets" in record
    assert "n_eff" not in record


def test_same_input_and_seed_give_a_bit_identical_row() -> None:
    panel = construct(_one_factor(9), "dollar_neutral")
    kwargs = {"construction": "dollar_neutral", "n_boot": 300, "seed": 11,
              "ci_level": 0.95, "block_length": None}
    assert breadth_measurement(panel, **kwargs).to_dict() == \
        breadth_measurement(panel, **kwargs).to_dict()


def test_a_panel_with_a_gap_is_refused_and_not_filled() -> None:
    panel = _one_factor(10)
    panel.iloc[5, 2] = np.nan
    with pytest.raises(DataContractError):
        breadth_measurement(panel, construction="directional", n_boot=50, seed=1,
                            ci_level=0.95, block_length=None)
