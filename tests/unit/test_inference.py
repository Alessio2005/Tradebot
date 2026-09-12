"""De inferentiekern. Elke test hier falsifieert een aanname die revisie 1
stilzwijgend maakte.

Deze tests zijn opzettelijk streng op de RICHTING van de correcties, niet op
hun exacte waarde: een correctie die de verkeerde kant op werkt, is erger dan
geen correctie, want zij ziet eruit als zorgvuldigheid.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradebot.backtest.metrics import deflated_sharpe
from tradebot.validation.inference import (
    clustered_mean,
    neff_deflation,
    sharpe_difference,
    sharpe_with_se,
)

IDX = pd.date_range("2022-01-01", periods=1615, freq="D", tz="UTC")


def test_naive_se_is_recovered_under_iid_normality() -> None:
    """De Lo-correctie MOET degenereren naar 1/sqrt(T) wanneer de aannames van
    1/sqrt(T) gelden. Doet zij dat niet, dan is zij verkeerd geïmplementeerd."""
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0.0, 0.01, 1615), index=IDX)
    est = sharpe_with_se(returns, bars_per_year=365, nw_lags=0)
    naive = 1.0 / np.sqrt(1615 / 365)
    assert abs(est.se - naive) / naive < 0.10


def test_fat_tails_inflate_the_standard_error() -> None:
    """Dik-staartige rendementen maken een Sharpe ONZEKERDER. Een SE die daar
    niet op reageert, onderschat de onzekerheid stelselmatig."""
    rng = np.random.default_rng(1)
    thin = pd.Series(rng.normal(0.001, 0.01, 1615), index=IDX)
    fat = pd.Series(rng.standard_t(3, 1615) * 0.01 + 0.001, index=IDX)
    assert (
        sharpe_with_se(fat, bars_per_year=365).se
        > sharpe_with_se(thin, bars_per_year=365).se
    )


def test_positive_autocorrelation_inflates_the_standard_error() -> None:
    """Newey-West, in de goede richting. Positief autogecorreleerde P&L bevat
    minder informatie per bar dan een i.i.d.-reeks van dezelfde lengte."""
    rng = np.random.default_rng(2)
    innovation = rng.normal(0.0, 0.01, 1615)
    ar = np.zeros(1615)
    for i in range(1, 1615):
        ar[i] = 0.4 * ar[i - 1] + innovation[i]
    series = pd.Series(ar + 0.001, index=IDX)
    assert (
        sharpe_with_se(series, bars_per_year=365).se
        > sharpe_with_se(series, bars_per_year=365, nw_lags=0).se
    )


def test_sharpe_difference_is_paired_and_smaller_than_the_level_se() -> None:
    """De kern van stap 10. Twee sterk gecorreleerde reeksen hebben een
    verschil-SE die veel kleiner is dan de SE van elk niveau -- dat is precies
    waarom de gepaarde toets een onderscheid kan maken dat de niveaus niet
    maken."""
    rng = np.random.default_rng(3)
    base = rng.normal(0.0005, 0.01, 1615)
    a = pd.Series(base, index=IDX)
    b = pd.Series(base + rng.normal(0.0, 0.0005, 1615), index=IDX)
    diff = sharpe_difference(a, b, bars_per_year=365, n_boot=2000, seed=7)
    level = sharpe_with_se(a, bars_per_year=365)
    assert diff.se < level.se
    assert diff.n_paired_obs == 1615


def test_sharpe_difference_uses_only_bars_where_both_are_active() -> None:
    """De halt-correctie uit §5.2. Als de ene keten 1559 bars gehalteerd is,
    is een 'gepaard' verschil over alle bars niet gepaard."""
    rng = np.random.default_rng(4)
    a = pd.Series(rng.normal(0.0005, 0.01, 1615), index=IDX)
    b = a.copy()
    b.iloc[200:] = np.nan
    diff = sharpe_difference(a, b, bars_per_year=365, n_boot=500, seed=7)
    assert diff.n_paired_obs == 200


def test_identical_series_have_zero_difference_and_zero_t() -> None:
    rng = np.random.default_rng(5)
    series = pd.Series(rng.normal(0.0, 0.01, 1615), index=IDX)
    diff = sharpe_difference(series, series, bars_per_year=365, n_boot=500, seed=7)
    assert diff.delta == 0.0
    assert diff.t_stat == 0.0


def test_clustering_on_date_shrinks_the_t_statistic_on_a_correlated_panel() -> None:
    """Q4. Zes namen met rho-bar 0,74 zijn geen zes onafhankelijke reeksen.
    Een gepoolde t die daar niet op reageert, overschat met ongeveer een
    factor twee."""
    rng = np.random.default_rng(6)
    common = rng.normal(0.001, 0.01, 1615)
    panel = pd.DataFrame(
        {f"S{i}": common + rng.normal(0.0, 0.004, 1615) for i in range(6)},
        index=IDX,
    )
    pooled = panel.to_numpy().ravel()
    naive_t = pooled.mean() / (pooled.std(ddof=1) / np.sqrt(pooled.size))
    assert abs(clustered_mean(panel).t_stat) < abs(naive_t)


def test_neff_deflation_matches_the_measured_panel() -> None:
    """De gemeten waarde: N_eff = 1,271 bij zes namen geeft factor 0,460."""
    corr = np.full((6, 6), 0.7442)
    np.fill_diagonal(corr, 1.0)
    assert abs(neff_deflation(corr) - 0.460) < 0.02


def test_dsr_requires_its_moments_explicitly() -> None:
    """Q5. De DSR heeft V[SR_m], scheefheid en kurtosis nodig. Een aanroep
    zonder die argumenten mag niet stilzwijgend de normale benadering pakken."""
    with pytest.raises(TypeError):
        deflated_sharpe(0.01, n_obs=1615, n_trials=25)  # type: ignore[call-arg]


def test_dsr_is_stricter_when_the_trial_sharpes_are_more_dispersed() -> None:
    """Hoe wilder de zoektocht spreidde, hoe hoger de lat. Dat is de hele
    gedachte achter de DSR en zij hoort in de code te staan."""
    common = dict(n_obs=1615, n_trials=25, skew=0.0, kurtosis=3.0,
                  bars_per_year=365)
    tight = deflated_sharpe(0.05, sr_variance=1.0 / 1615, **common)
    wide = deflated_sharpe(0.05, sr_variance=4.0 / 1615, **common)
    assert wide.dsr < tight.dsr


# --------------------------------------------------------------------------- #
# Ruling P47 (fase 10, stap 10) -- de ontaardingstak van `sharpe_difference_test`
# --------------------------------------------------------------------------- #
# De tak stond op `np.array_equal(x, y)`: LETTERLIJKE gelijkheid. De analytisch
# ontaarde verzameling is groter. Sharpe is schaal-invariant, dus bij `y = c*x`
# met `c > 0` is het verschil voor ELKE steekproef exact nul en zijn variantie
# ook: de vier momenten `(x, y, x^2, y^2)` zijn perfect afhankelijk en
# `grad' Psi grad` valt term voor term weg. `_lw_delta_and_se` geeft daar
# `delta = 0,0` met `se = nan`, waarna `require(se > 0)` een geval weigerde
# waarvan het antwoord in gesloten vorm bekend is.


def test_a_positive_multiple_has_exactly_zero_sharpe_difference() -> None:
    """`y = 2x` is dezelfde keten in een andere maatvoering, niet een andere
    keten. Sharpe is schaal-invariant, dus het verschil is nul MET ZEKERHEID en
    niet nul met een standaardfout. Dit gaf een `DataContractError` tot P47."""
    rng = np.random.default_rng(3)
    a = pd.Series(rng.normal(0.0005, 0.010, 1615), index=IDX)
    for factor in (2.0, 0.5, 1e-6):
        diff = sharpe_difference(
            a, pd.Series(a.to_numpy() * factor, index=IDX),
            bars_per_year=365, n_boot=500, seed=7,
        )
        assert diff.delta_sharpe == 0.0
        assert diff.se == 0.0
        assert diff.t_stat == 0.0
        assert diff.p_value == 1.0
        assert diff.ci_low == 0.0 and diff.ci_high == 0.0
        assert diff.sharpe_a == pytest.approx(diff.sharpe_b, rel=1e-12)


def test_the_identity_case_still_answers_exactly_as_before() -> None:
    """`c = 1` is het bijzondere geval van P47's evenredigheidstak. Het
    antwoord dat de oude `array_equal`-tak gaf, moet er onveranderd uit komen;
    een 'reparatie' die het identiteitsgeval verandert, is een regressie."""
    rng = np.random.default_rng(5)
    series = pd.Series(rng.normal(0.0, 0.01, 1615), index=IDX)
    diff = sharpe_difference(series, series.copy(), bars_per_year=365,
                             n_boot=500, seed=7)
    assert (diff.delta_sharpe, diff.se, diff.t_stat, diff.p_value) == (
        0.0, 0.0, 0.0, 1.0)


def test_a_sign_flip_is_not_a_degenerate_pair() -> None:
    """De negatieve controle die bewijst dat de tak niet te veel opslokt.
    Bij `y = -x` is het Sharpe-verschil `2*SR(x)` en dus in het algemeen NIET
    nul; een evenredigheidstak die ook negatieve veelvouden vangt, geeft geen
    reparatie maar een fout antwoord."""
    rng = np.random.default_rng(3)
    a = pd.Series(rng.normal(0.0005, 0.010, 1615), index=IDX)
    diff = sharpe_difference(a, pd.Series(-a.to_numpy(), index=IDX),
                             bars_per_year=365, n_boot=500, seed=7)
    assert diff.sharpe_b == pytest.approx(-diff.sharpe_a, rel=1e-12)
    assert abs(diff.delta_sharpe) > 1.0
    assert diff.se > 0.0
    assert diff.p_value < 0.05
