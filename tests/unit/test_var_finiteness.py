"""VaR-schatters zijn eindig op degenerate invoer — Phase 6, stap 0.

`test_evt_gpd_var_less_than_cf_at_extreme` was rood op `84273ca`. Het
foutbericht wees naar ``evt_gpd_var``, maar de oorzaak lag een niveau dieper.
Deze suite legt de hele keten vast, omdat een reparatie op de verkeerde plek
groen wordt zonder het defect te sluiten.

DE KETEN, ZOALS GEMETEN
-----------------------
    arr = [1.0] * 50, threshold_q = 0.5
      -> returns = -arr, u = quantile(returns, 0.5) = -1.0
      -> tail = returns[returns < -1.0] = leeg
      -> len(tail) < 10  -> cornish_fisher_var(arr, 0.99)
      -> scipy.stats.skew op een reeks zonder spreiding = 0/0 = NaN
      -> mean + NaN * 0.0 = NaN

``evt_gpd_var`` deed dus niets fout; hij gaf een NaN door die hij van
``cornish_fisher_var`` kreeg. Een `np.nan_to_num` in ``evt_gpd_var`` had de test
groen gemaakt en het defect laten staan.
"""
from __future__ import annotations

import numpy as np
import pytest
import scipy.stats as _stats

from tradebot.risk.var import cornish_fisher_var, evt_gpd_var

#: Het exacte tegenvoorbeeld dat Hypothesis vond.
_COUNTEREXAMPLE = -np.ones(50, dtype=np.float64)


def test_the_historical_counterexample_is_now_finite() -> None:
    assert np.isfinite(evt_gpd_var(_COUNTEREXAMPLE, confidence=0.99,
                                   threshold_quantile=0.5))


def test_scipy_still_returns_nan_here_so_the_guard_has_work_to_do() -> None:
    """Negatieve controle (§0.10) — bewijst dat de oorzaak nog bestaat.

    Zonder deze test is er geen bewijs dat de guard iets doet: als een latere
    scipy-versie skew/kurtosis op een puntmassa als 0.0 zou definiëren, was het
    defect vanzelf verdwenen en zou de guard dode code zijn geworden zonder dat
    iemand het merkt. Deze assertie maakt dat zichtbaar op het moment dat het
    gebeurt.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        skew = _stats.skew(_COUNTEREXAMPLE)
        kurt = _stats.kurtosis(_COUNTEREXAMPLE)
    assert not np.isfinite(skew) and not np.isfinite(kurt), (
        "scipy geeft hier geen NaN meer; de degenerate-guard in "
        "cornish_fisher_var meet dan niet langer het defect dat hij sloot"
    )


def test_the_point_mass_answer_is_the_exact_value_not_a_placeholder() -> None:
    """Een constante reeks heeft een BEKEND kwantiel: de constante zelf.

    Dit onderscheidt de reparatie van een `return 0.0`-vangnet. Het antwoord is
    exact en schaalt mee met de invoer.
    """
    for level in (-1.0, 0.0, 3.5, -42.25):
        arr = np.full(60, level)
        assert cornish_fisher_var(arr, confidence=0.99) == pytest.approx(level)


def test_the_guard_fires_on_underflowed_central_moments_too() -> None:
    """``std == 0`` was het VERKEERDE criterium geweest.

    Op deze reeks is ``np.std(ddof=1)`` ~ 8e-17 en dus niet nul, terwijl het
    centrale moment waar scipy op deelt WEL naar nul onderloopt. Een guard op
    ``std == 0`` had deze invoer laten passeren en alsnog NaN teruggegeven.
    """
    arr = np.ones(50) + np.array([1e-16] * 25 + [-1e-16] * 25)
    assert np.std(arr, ddof=1) > 0.0
    assert np.isfinite(cornish_fisher_var(arr, confidence=0.99))


@pytest.mark.parametrize("threshold_q", [0.5, 0.6, 0.75, 0.9, 0.95])
@pytest.mark.parametrize(
    ("name", "arr"),
    [
        ("constant", np.ones(80)),
        ("twee waarden", np.where(np.arange(80) % 2 == 0, 1.0, 1.0000001)),
        ("één uitschieter", np.concatenate([np.ones(79), [100.0]])),
        ("uniform", np.linspace(1.0, 100.0, 120)),
        ("zware staart", np.concatenate([np.ones(100), np.linspace(2, 50, 20)])),
    ],
)
def test_evt_is_finite_across_degenerate_and_normal_shapes(
    name: str, arr: np.ndarray, threshold_q: float
) -> None:
    result = evt_gpd_var(-arr, confidence=0.99, threshold_quantile=threshold_q)
    assert np.isfinite(result), f"{name} @ q={threshold_q} gaf {result}"


def test_a_healthy_sample_still_gets_the_cornish_fisher_correction() -> None:
    """De guard mag de normale route niet kapen.

    Zonder deze test zou een guard die ALTIJD ``mean`` teruggeeft ook groen
    zijn — en de CF-correctie stilzwijgend hebben uitgeschakeld.
    """
    rng = np.random.default_rng(20260825)
    arr = rng.standard_t(df=4, size=4000) * 0.01
    cf = cornish_fisher_var(arr, confidence=0.99)
    assert np.isfinite(cf)
    assert cf < float(np.mean(arr)), (
        "de 99%-VaR ligt niet onder het gemiddelde; de CF-correctie is "
        "waarschijnlijk overgeslagen"
    )
    # De correctie moet ook echt van de Gaussische waarde afwijken op scheve,
    # zware-staart-data.
    gaussian = float(np.mean(arr) + _stats.norm.ppf(0.01) * np.std(arr, ddof=1))
    assert cf != pytest.approx(gaussian, rel=1e-6)
