# tests/unit/test_hmm_fit_contract.py
"""Wat `fit_hmm` belooft: reproduceerbaar, gedeeld startpunt, geen derde model.

DE MEETING DIE DEZE TESTS AFDWINGT
===================================
`hmmlearn` initialiseert `GaussianHMM` met `sklearn.cluster.KMeans`. Die
parallelliseert over OpenMP-threads, en de reductievolgorde ligt niet vast.
Gemeten op 900 bars van twee overlappende toestanden, hmmlearn 0.3.3, zestig
identieke fits met `random_state=20260830`:

    aantal verschillende uitkomsten  =  2

Twee. Niet in de laatste decimaal van een rapportgetal, maar in de PARAMETERS
waarop het hele H2-oordeel rust. `deterministic_start` haalt die k-means uit de
keten; wat overblijft is de EM-recursie zelf en die is puur numpy.

`test_fit_is_bit_reproducible` herhaalt de fit dertig keer en eist EEN uitkomst.
Hij is de negatieve controle op de reparatie: zonder de reparatie wordt hij rood
op precies dezelfde manier waarop de meting hierboven rood was.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tradebot.regime.markov import (
    HmmParameters,
    HmmSpec,
    deterministic_start,
    filtered_occupancy,
    fit_hmm,
    forward_filter,
)
from tradebot.schemas.config import (
    AdequacyConfig,
    M2HmmConfig,
    load_config,
    regime_config,
)
from tradebot.utils.failfast import DataContractError

ADEQUACY = load_config(ROOT / "conf/model/adequacy.yaml", AdequacyConfig)
M2 = regime_config().m2
FOLDS = (500,) * 12


def _series(n: int = 900, seed: int = 7, dof: float = 3.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    scales = np.array([0.004, 0.020])
    trans = np.array([[0.95, 0.05], [0.05, 0.95]])
    state, out = 0, np.empty(n)
    for t in range(n):
        state = int(rng.choice(2, p=trans[state]))
        out[t] = scales[state] * rng.standard_t(dof)
    return out[:, None]


def _fit(spec: HmmSpec, observations: np.ndarray, seed: int = 20260830):
    return fit_hmm(
        observations, spec, ADEQUACY, symbol="TEST", fold_id=0,
        train_end=600, n_obs_per_fold=FOLDS, seed=seed, m2_cfg=M2)


class TestDeterministicStart:
    def test_splits_on_scale_and_not_on_sign(self) -> None:
        """K-means op een 1-D returnreeks splitst op NIVEAU en zet daarmee de
        negatieve returns in de ene toestand. Een vol-regimemodel gaat over
        SCHAAL; deze test faalt zodra de split terugvalt op het niveau."""
        rng = np.random.default_rng(0)
        calm = rng.normal(0.0, 0.001, 400)
        wild = rng.normal(0.0, 0.05, 400)
        start = deterministic_start(
            np.concatenate([calm, wild])[:, None], 2, "diag")
        variances = start["covars"].ravel()
        assert variances[1] > 10.0 * variances[0]
        assert abs(start["means"]).max() < 0.02

    def test_start_and_transitions_are_uniform(self) -> None:
        """De initialisatie beweert niets over persistentie: die moet uit de
        data komen en niet uit een startwaarde."""
        start = deterministic_start(_series(), 3, "diag")
        assert np.allclose(start["start_prob"], 1.0 / 3.0)
        assert np.allclose(start["trans_mat"], 1.0 / 3.0)

    def test_is_bit_identical_across_repeats(self) -> None:
        observations = _series()
        first = deterministic_start(observations, 2, "diag")
        for _ in range(20):
            again = deterministic_start(observations, 2, "diag")
            assert np.array_equal(first["means"], again["means"])
            assert np.array_equal(first["covars"], again["covars"])

    def test_a_group_without_spread_crashes(self) -> None:
        constant = np.zeros((60, 1))
        with pytest.raises(DataContractError):
            deterministic_start(constant, 2, "diag")

    def test_full_covariance_start_is_diagonal(self) -> None:
        start = deterministic_start(_series(), 2, "full")
        assert start["covars"].shape == (2, 1, 1)


class TestReproducibility:
    def test_fit_is_bit_reproducible(self) -> None:
        """Dertig identieke fits, één uitkomst. Dit is de test die de gemeten
        twee-uit-zestig van `hmmlearn`s k-means-initialisatie afvangt."""
        observations = _series()
        spec = HmmSpec(n_states=2)
        seen = {_fit(spec, observations).means.tobytes() for _ in range(30)}
        assert len(seen) == 1

    def test_the_seed_no_longer_changes_the_outcome(self) -> None:
        """Het startpunt komt niet meer uit een RNG. De seed blijft in de
        handtekening zodat `hmmlearn` niet op een globale RNG terugvalt, maar
        hij mag het resultaat niet meer bepalen -- anders is `seed` een
        verborgen trial."""
        observations = _series()
        spec = HmmSpec(n_states=2)
        a = _fit(spec, observations, seed=1)
        b = _fit(spec, observations, seed=999_999)
        assert np.array_equal(a.means, b.means)
        assert a.loglikelihood == b.loglikelihood

    def test_student_t_fit_is_bit_reproducible(self) -> None:
        observations = _series()
        spec = HmmSpec(n_states=2, distribution="student_t")
        seen = {_fit(spec, observations).means.tobytes() for _ in range(10)}
        assert len(seen) == 1


class TestStudentTPath:
    def test_the_variant_is_no_longer_refused(self) -> None:
        """De pre-registratie H2 boekte `hmm2-diag-student_t` als trial. Een
        variant weigeren die al in `M` staat, laat `M` betalen voor werk dat
        nooit is gedaan."""
        spec = HmmSpec(n_states=2, distribution="student_t")
        assert spec.label == "hmm2-diag-student_t"

    def test_fit_carries_degrees_of_freedom(self) -> None:
        params = _fit(HmmSpec(n_states=2, distribution="student_t"), _series())
        assert params.dof is not None
        assert params.dof.shape == (2,)
        assert np.all(params.dof > M2.dof_min - 1e-9)
        assert params.as_record()["dof"] is not None

    def test_gaussian_fit_carries_no_degrees_of_freedom(self) -> None:
        params = _fit(HmmSpec(n_states=2), _series())
        assert params.dof is None
        assert params.as_record()["dof"] is None

    def test_a_gaussian_spec_with_dof_is_refused(self) -> None:
        """Een Gaussische fit met een t-filter is een DERDE model dat in geen
        enkele pre-registratie staat. Het contract weigert de combinatie in
        plaats van er stilzwijgend een van de twee van te maken."""
        params = _fit(HmmSpec(n_states=2), _series())
        with pytest.raises(DataContractError):
            HmmParameters(
                spec=params.spec, symbol="TEST", fold_id=0,
                start_prob=params.start_prob, trans_mat=params.trans_mat,
                means=params.means, covars=params.covars, converged=True,
                n_train_obs=600, loglikelihood=0.0, dof=np.array([5.0, 5.0]))

    def test_a_student_t_spec_without_dof_is_refused(self) -> None:
        params = _fit(HmmSpec(n_states=2, distribution="student_t"), _series())
        with pytest.raises(DataContractError):
            HmmParameters(
                spec=params.spec, symbol="TEST", fold_id=0,
                start_prob=params.start_prob, trans_mat=params.trans_mat,
                means=params.means, covars=params.covars, converged=True,
                n_train_obs=600, loglikelihood=0.0, dof=None)

    def test_the_filter_uses_the_t_density(self) -> None:
        """Beide varianten delen het startpunt, dus een verschil in de filtered
        posterior komt van de VERDELING. Zonder deze test zou een dispatch die
        stilzwijgend Gaussisch blijft er nooit uitkomen."""
        observations = _series(dof=2.5, seed=11)
        gaussian = forward_filter(
            observations, _fit(HmmSpec(n_states=2), observations))
        student = forward_filter(
            observations,
            _fit(HmmSpec(n_states=2, distribution="student_t"), observations))
        assert np.abs(gaussian.values - student.values).max() > 1e-6

    def test_the_student_t_filter_is_still_causal(self) -> None:
        """Dezelfde afkaptest als voor de Gaussische variant: de rijen tot aan
        de snede mogen niet veranderen wanneer er latere bars bij komen."""
        observations = _series(dof=2.5, seed=11)
        params = _fit(
            HmmSpec(n_states=2, distribution="student_t"), observations)
        full = forward_filter(observations, params)
        cut = forward_filter(observations[:700], params)
        assert np.allclose(full.values[:700], cut.values, atol=0.0, rtol=0.0)

    def test_occupancy_comes_from_the_filtered_posterior(self) -> None:
        params = _fit(
            HmmSpec(n_states=2, distribution="student_t"), _series())
        occupancy = filtered_occupancy(forward_filter(_series(), params))
        assert occupancy.shape == (2,)
        assert occupancy.sum() == pytest.approx(1.0)


class TestConfigIsTheSearchSpace:
    def test_the_grid_matches_the_preregistered_trial_count(self) -> None:
        """Drie varianten maal `n_states_grid` moet de zes trials van
        pre-registratie 3d3af28730a6c7f9da48d13139522a05 opleveren."""
        assert 3 * len(M2.n_states_grid) == 6

    def test_a_duplicated_state_count_is_refused(self) -> None:
        with pytest.raises(Exception, match="dubbelen"):
            M2HmmConfig(n_states_grid=(2, 2))

    def test_a_single_state_is_refused(self) -> None:
        with pytest.raises(Exception, match="regimemodel"):
            M2HmmConfig(n_states_grid=(1, 2))
