"""De Data Adequacy Gate is een POORT en geen waarschuwing — Phase 6, §3.

Een gate die niet kan weigeren, is decoratie. Elke test hieronder toetst één van
de twee eigenschappen die de poort moet hebben:

    (a) hij LAAT DOOR wat de eis haalt;
    (b) hij WEIGERT wat de eis niet haalt, met cijfers in het bericht.

De tweede is de belangrijke. `reports/phase5_exit_report.md` §12 vond vijf van
negen defecten in de bewijsvoering zelf, waaronder een wiring-test met
niet-bindende limietwaarden die dus niets bewees. Elke drempel hieronder wordt
daarom van BEIDE kanten benaderd.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradebot.schemas.config import AdequacyConfig, load_config
from tradebot.utils.failfast import DataContractError
from tradebot.validation.data_adequacy import (
    INSUFFICIENT_DATA,
    AdequacyVerdict,
    assess_garch,
    assess_har_rv,
    assess_hmm,
    assess_hrp,
    assess_meta_labeling,
    auc_minimum_detectable,
    effective_independent_series,
    mean_difference_power,
    require_adequacy,
    sharpe_difference_power,
)

CFG = load_config("conf/model/adequacy.yaml", AdequacyConfig)


# --------------------------------------------------------------------------- #
# Het oordeel zelf
# --------------------------------------------------------------------------- #
def test_an_inadequate_verdict_without_a_shortfall_is_rejected() -> None:
    """`UNPROVEN — insufficient data` moet een METING zijn, geen vermoeden."""
    with pytest.raises(DataContractError, match="vermoeden"):
        AdequacyVerdict(model_class="x", adequate=False, requirement="r",
                        measured={}, shortfall="")


def test_an_adequate_verdict_with_a_shortfall_is_rejected() -> None:
    with pytest.raises(DataContractError, match="MET een tekort"):
        AdequacyVerdict(model_class="x", adequate=True, requirement="r",
                        measured={}, shortfall="te weinig")


def test_the_verdict_string_is_the_one_the_ledger_uses() -> None:
    bad = AdequacyVerdict(model_class="x", adequate=False, requirement="r",
                          measured={"n": 1}, shortfall="te weinig")
    assert bad.verdict == INSUFFICIENT_DATA
    assert AdequacyVerdict(model_class="x", adequate=True, requirement="r",
                           measured={}).verdict == ""


def test_require_adequacy_crashes_and_carries_the_numbers() -> None:
    """Het bericht moet zelfstandig diagnosticeerbaar zijn."""
    bad = AdequacyVerdict(model_class="hmm_k3", adequate=False,
                          requirement=">= 100 per state",
                          measured={"rarest_state_obs": 41.0},
                          shortfall="te weinig bezetting.")
    with pytest.raises(DataContractError) as exc:
        require_adequacy(bad)
    message = str(exc.value)
    assert "hmm_k3" in message
    assert "41.0" in message
    assert INSUFFICIENT_DATA in message
    assert "telt niet mee in M" in message


def test_require_adequacy_is_silent_when_the_gate_opens() -> None:
    require_adequacy(AdequacyVerdict(
        model_class="ok", adequate=True, requirement="r", measured={"n": 1000}))


# --------------------------------------------------------------------------- #
# Per modelklasse — beide kanten van elke drempel
# --------------------------------------------------------------------------- #
def test_garch_passes_above_and_fails_below_the_window_requirement() -> None:
    need = CFG.garch.min_obs_per_fit_window
    assert assess_garch([need, need + 100], CFG).adequate
    failed = assess_garch([need - 1, need + 100], CFG)
    assert not failed.adequate
    assert str(need - 1) in failed.shortfall, (
        "het tekort noemt het gemeten getal niet; het oordeel is dan niet "
        "onderbouwd met cijfers"
    )


def test_garch_binds_on_the_smallest_window_not_the_average() -> None:
    """Een model wordt PER FOLD gefit; het gemiddelde venster fit niets."""
    need = CFG.garch.min_obs_per_fit_window
    assert not assess_garch([need - 200, need + 5000], CFG).adequate


def test_har_rv_fails_on_this_universe_because_there_is_no_intraday_data() -> None:
    """De bevinding die deze fase op DEZE store hoort te doen."""
    verdict = assess_har_rv(1743, 0, CFG, observed_granularity="geen")
    assert not verdict.adequate
    assert verdict.measured["coverage_pct"] == 0.0
    assert "realized variance" in verdict.shortfall


def test_har_rv_would_pass_with_sufficient_coverage() -> None:
    """Negatieve controle: de eis is haalbaar en niet onmogelijk gesteld."""
    assert assess_har_rv(1000, 900, CFG).adequate


def test_hmm_uses_the_optimistic_uniform_assumption_by_default() -> None:
    """En zegt dat er ook bij, zodat een PASS niet als bewijs wordt gelezen."""
    verdict = assess_hmm([495] * 12, 3, CFG)
    assert "optimistisch" in verdict.measured["occupancy_source"]
    assert verdict.measured["rarest_state_occupancy_fraction"] == pytest.approx(1 / 3)


def test_hmm_fails_once_the_real_occupancy_is_supplied() -> None:
    """De poort die er op dit universum toe doet.

    Onder de uniforme aanname haalt een 3-state HMM op 495 bars per fold de eis
    ruim. Zodra de FEITELIJKE bezetting erin gaat — een crisisregime dat 8 % van
    de tijd actief is — valt hij om. Dat verschil is precies waarom de gate de
    bron van de aanname rapporteert.
    """
    optimistic = assess_hmm([495] * 12, 3, CFG)
    assert optimistic.adequate

    realistic = assess_hmm([495] * 12, 3, CFG,
                           expected_occupancy=[0.62, 0.30, 0.08])
    assert not realistic.adequate
    assert "uitschieterdetector" in realistic.shortfall


def test_hmm_rejects_an_occupancy_that_is_not_a_distribution() -> None:
    with pytest.raises(DataContractError, match="kansverdeling"):
        assess_hmm([495], 2, CFG, expected_occupancy=[0.3, 0.3])


def test_meta_labeling_needs_effective_events_not_just_nominal_ones() -> None:
    """De correctie die op deze opzet de uitkomst bepaalt.

    Met een event op elke bar en een horizon van tien bars is het nominale
    aantal een factor ~6 te optimistisch. Een gate die alleen nominaal telt,
    laat een fold met 3.000 sterk overlappende labels door alsof het 3.000
    onafhankelijke waarnemingen zijn.
    """
    nominal = [3000] * 6
    ratios = [0.5] * 6
    generous = assess_meta_labeling(
        nominal, ratios, CFG, effective_events_per_fold=[450.0] * 6)
    assert generous.adequate

    overlapping = assess_meta_labeling(
        nominal, ratios, CFG, effective_events_per_fold=[80.0] * 6)
    assert not overlapping.adequate
    assert "EFFECTIEVE" in overlapping.shortfall


def test_meta_labeling_refuses_to_pass_when_uniqueness_was_never_measured() -> None:
    """Niet meten is niet hetzelfde als voldoen."""
    verdict = assess_meta_labeling([3000] * 6, [0.5] * 6, CFG)
    assert not verdict.adequate
    assert "NIET gemeten" in verdict.shortfall


def test_meta_labeling_fails_on_an_unbalanced_fold() -> None:
    verdict = assess_meta_labeling(
        [3000] * 6, [0.5, 0.5, 0.96, 0.5, 0.5, 0.5], CFG,
        effective_events_per_fold=[450.0] * 6)
    assert not verdict.adequate
    assert "positieve-klasseratio" in verdict.shortfall


def test_hrp_fails_on_a_near_rank_one_universe() -> None:
    """De situatie uit §0.7, nagebouwd: alles beweegt met één factor."""
    rng = np.random.default_rng(3)
    factor = rng.normal(0.0, 0.02, 400)
    frame = pd.DataFrame({
        f"A{i}": factor + rng.normal(0.0, 0.0005, 400) for i in range(6)
    })
    verdict = assess_hrp(frame, CFG, n_folds=12)
    assert not verdict.adequate
    assert verdict.measured["mean_pairwise_correlation"] > 0.95
    assert "conditiegetal" in verdict.shortfall


def test_hrp_passes_on_a_well_conditioned_universe() -> None:
    rng = np.random.default_rng(4)
    frame = pd.DataFrame(rng.normal(0.0, 0.02, (2000, 6)),
                         columns=[f"A{i}" for i in range(6)])
    assert assess_hrp(frame, CFG, n_folds=6).adequate


# --------------------------------------------------------------------------- #
# Power-analyse
# --------------------------------------------------------------------------- #
def test_six_correlated_perpetuals_are_worth_about_one_and_a_third_series() -> None:
    """De correctie waarzonder elke power-berekening in deze fase fout is."""
    assert effective_independent_series(6, 0.0) == pytest.approx(6.0)
    assert effective_independent_series(6, 1.0) == pytest.approx(1.0)
    assert effective_independent_series(6, 0.7485) == pytest.approx(1.265, abs=1e-3)


def test_the_minimum_detectable_effect_shrinks_with_the_square_root_of_n() -> None:
    small = mean_difference_power(100, 0.1, CFG.power)
    large = mean_difference_power(400, 0.1, CFG.power)
    assert large.minimum_detectable_effect == pytest.approx(
        small.minimum_detectable_effect / 2.0, rel=1e-9)


def test_a_power_analysis_reports_when_the_test_cannot_see_the_effect() -> None:
    """De eigenschap die exit-criterium 10 uitvraagt."""
    blind = mean_difference_power(100, 0.01, CFG.power)
    assert not blind.informative
    assert blind.power_deficit > 1.0

    sighted = mean_difference_power(10_000, 0.10, CFG.power)
    assert sighted.informative
    assert sighted.power_deficit < 1.0


def test_the_sharpe_difference_mde_depends_on_the_pairing() -> None:
    """Twee sterk gecorreleerde strategieën zijn preciezer te vergelijken.

    Zonder die correctie zou H2 op 4,8 jaar bij voorbaat kansloos lijken; met
    haar is de vraag hoe sterk het regimefilter de returnreeks verandert.
    """
    independent = sharpe_difference_power(4.77, 0.5, 0.0, 0.10, CFG.power)
    paired = sharpe_difference_power(4.77, 0.5, 0.95, 0.10, CFG.power)
    assert paired.minimum_detectable_effect < independent.minimum_detectable_effect
    assert paired.minimum_detectable_effect == pytest.approx(
        independent.minimum_detectable_effect * np.sqrt(0.05 / 1.0), rel=1e-9)


def test_the_auc_mde_falls_as_the_event_count_rises() -> None:
    few = auc_minimum_detectable(200, 200, CFG.power)
    many = auc_minimum_detectable(2000, 2000, CFG.power)
    assert many.minimum_detectable_effect < few.minimum_detectable_effect
    assert few.expected_effect == pytest.approx(0.08)
