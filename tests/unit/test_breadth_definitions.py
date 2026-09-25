"""Twee grootheden, twee namen — en het bewijs dat de oude naam de verkeerde meet.

`validation/inference.py::effective_breadth` is Kish' ontwerpeffect van een
gepoold, gelijkgewogen gemiddelde: hoeveel onafhankelijke waarnemingen het
gemiddelde van N gecorreleerde reeksen waard is. Dat is de juiste grootheid om
een gepoolde t te defleren en de verkeerde om weddenschappen mee te tellen. Op een
paneel dat over de namen tot nul sommeert, gaat haar noemer `1 + (N-1) rho_bar`
naar nul; op het dollar-neutrale residu van de zes gecertificeerde namen geeft zij
123,58 (`fase_11_breedte_en_tijdschaal.md` §2.1).

Het aantal onafhankelijke weddenschappen uit de fundamentele wet is begrensd door
de rang van de correlatiematrix. `validation/breadth.py::independent_bets` is de
naam waaronder de validatielaag die grootheid gebruikt; de implementatie blijft
in `portfolio/covariance.py::effective_n_assets` (DI-10, AD-30).

Negatieve controle, met de hand gedraaid en niet gecommit: laat
`independent_bets` de rho-formule aanroepen, en
`test_independent_bets_never_exceed_the_rank` wordt rood.
"""
from __future__ import annotations

import numpy as np
import pytest

from tradebot.utils.failfast import DataContractError
from tradebot.validation.breadth import independent_bets
from tradebot.validation.inference import effective_breadth

N_OBS = 2_000
N_NAMES = 6


def _one_factor(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    market = rng.standard_normal((N_OBS, 1))
    return 0.85 * market + 0.5 * rng.standard_normal((N_OBS, N_NAMES))


def _dollar_neutral(panel: np.ndarray) -> np.ndarray:
    return panel - panel.mean(axis=1, keepdims=True)


def test_independent_bets_never_exceed_the_rank() -> None:
    residual = _dollar_neutral(_one_factor(35))
    corr = np.corrcoef(residual, rowvar=False)
    assert np.linalg.matrix_rank(residual) == N_NAMES - 1
    assert independent_bets(corr) <= N_NAMES - 1 + 1e-9


def test_clones_are_one_bet_and_independents_are_n() -> None:
    rng = np.random.default_rng(36)
    clones = np.repeat(rng.standard_normal((N_OBS, 1)), N_NAMES, axis=1)
    clones = clones + 1e-6 * rng.standard_normal(clones.shape)
    independents = rng.standard_normal((N_OBS, N_NAMES))
    assert independent_bets(np.corrcoef(clones, rowvar=False)) < 1.05
    assert independent_bets(np.corrcoef(independents, rowvar=False)) > 5.8


def test_it_delegates_and_does_not_reimplement() -> None:
    from tradebot.portfolio.covariance import effective_n_assets

    corr = np.corrcoef(_one_factor(37), rowvar=False)
    assert independent_bets(corr) == effective_n_assets(corr)


@pytest.mark.parametrize(
    "bad",
    [
        np.ones((3, 2)),
        np.array([[1.0]]),
        np.array([[1.0, 0.2], [0.3, 1.0]]),
        np.array([[1.0, np.nan], [np.nan, 1.0]]),
    ],
)
def test_it_refuses_what_is_not_a_correlation_matrix(bad: np.ndarray) -> None:
    with pytest.raises(DataContractError):
        independent_bets(bad)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "DI-35: effective_breadth is the design effect of a pooled mean, not "
        "breadth; on a dollar-neutral panel it exceeds the number of names. "
        "Rename after the ledger amendment of H-11.1 "
        "(fase_11_breedte_en_tijdschaal.md, step 1.6)."
    ),
)
def test_the_design_effect_is_not_breadth() -> None:
    residual = _dollar_neutral(_one_factor(35))
    corr = np.corrcoef(residual, rowvar=False)
    assert effective_breadth(corr) <= N_NAMES
