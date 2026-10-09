# tests/unit/test_student_t_hmm.py
"""De Student-t emissies en hun EM-stap. Deliverable 15, de M2-variant.

WAAROM DEZE TESTS BESTAAN
==========================
`hmmlearn` levert geen Student-t HMM. De pre-registratie H2
(`3d3af28730a6c7f9da48d13139522a05`) boekte hem wel: twee van de zes trials zijn
`hmm2-diag-student_t` en `hmm3-diag-student_t`, en die zes staan sinds het
bevriezen in `M`. Er is dus geen route waarlangs deze variant "later" komt
zonder dat er trials zijn geboekt voor werk dat nooit is gedaan.

De verleiding is een Gaussische fit met een t-filter of andersom. Dat is een
DERDE model dat in geen enkele pre-registratie staat. De EM staat daarom hier,
compleet, met de dichtheid getoetst tegen `scipy` in plaats van tegen zichzelf.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import integrate, stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tradebot.regime.markov import deterministic_start
from tradebot.regime.student_t import (
    StudentTFit,
    fit_student_t_hmm,
    student_t_log_density,
)
from tradebot.schemas.config import M2HmmConfig
from tradebot.utils.failfast import DataContractError

CFG = M2HmmConfig()


def _fit(
    observations: np.ndarray, n_states: int, cfg: M2HmmConfig = CFG
) -> StudentTFit:
    """De echte route: warme start binnen, EM eruit.

    De start is hier de DETERMINISTISCHE initialisatie in plaats van een
    Gaussische fit. Dat scheelt de test een `hmmlearn`-afhankelijkheid en meet
    precies wat dit module doet -- de campagne levert straks de Gaussische
    uitkomst aan, en dat is dezelfde vorm.
    """
    start = deterministic_start(observations, n_states, cfg.covariance_type)
    return fit_student_t_hmm(
        observations, start=start, gaussian_loglikelihood=float("nan"), cfg=cfg)


def _two_regime_series(
    n: int = 900, seed: int = 7, dof: float = 3.0
) -> np.ndarray:
    """Twee toestanden met een factor 5 in schaal en zware staarten.

    Persistent (0,95 op de diagonaal) omdat een HMM zonder persistentie een
    mengsel is en de transitiematrix dan niets meet.
    """
    rng = np.random.default_rng(seed)
    scales = np.array([0.004, 0.020])
    trans = np.array([[0.95, 0.05], [0.05, 0.95]])
    state = 0
    out = np.empty(n)
    for t in range(n):
        state = int(rng.choice(2, p=trans[state]))
        out[t] = scales[state] * rng.standard_t(dof)
    return out[:, None]


class TestDensity:
    """De dichtheid wordt tegen `scipy` getoetst, niet tegen zichzelf."""

    def test_matches_scipy_in_one_dimension(self) -> None:
        y = np.linspace(-4.0, 4.0, 41)[:, None]
        means = np.array([[0.3]])
        covars = np.array([[2.0]])
        dof = np.array([5.0])
        got = student_t_log_density(y, means, covars, dof, "diag")[:, 0]
        want = stats.t.logpdf(
            y[:, 0], df=5.0, loc=0.3, scale=math.sqrt(2.0))
        assert np.allclose(got, want, atol=1e-12)

    def test_large_dof_converges_to_the_gaussian(self) -> None:
        """Bij nu -> oneindig is de t een normale. Dat is de sanity check die
        aantoont dat de schaalmatrix een SCHAAL is en geen variantie."""
        y = np.linspace(-3.0, 3.0, 31)[:, None]
        means = np.array([[0.0]])
        covars = np.array([[1.5]])
        got = student_t_log_density(
            y, means, covars, np.array([1e7]), "diag")[:, 0]
        want = stats.norm.logpdf(y[:, 0], loc=0.0, scale=math.sqrt(1.5))
        assert np.allclose(got, want, atol=1e-5)

    def test_density_integrates_to_one(self) -> None:
        means, covars, dof = (
            np.array([[-0.2]]), np.array([[0.7]]), np.array([4.0]))

        def pdf(x: float) -> float:
            return float(np.exp(student_t_log_density(
                np.array([[x]]), means, covars, dof, "diag")[0, 0]))

        total, _ = integrate.quad(pdf, -60.0, 60.0, limit=400)
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_full_and_diag_agree_in_one_dimension(self) -> None:
        y = np.linspace(-2.0, 2.0, 21)[:, None]
        means = np.array([[0.1]])
        dof = np.array([6.0])
        diag = student_t_log_density(y, means, np.array([[0.8]]), dof, "diag")
        full = student_t_log_density(
            y, means, np.array([[[0.8]]]), dof, "full")
        assert np.allclose(diag, full, atol=1e-12)

    def test_non_positive_scale_crashes(self) -> None:
        """Geen fallback op een epsilon: een toestand zonder spreiding heeft
        geen dichtheid, en doorrekenen levert een getal zonder betekenis."""
        with pytest.raises(DataContractError):
            student_t_log_density(
                np.zeros((3, 1)), np.array([[0.0]]), np.array([[0.0]]),
                np.array([5.0]), "diag")


class TestEm:
    def test_recovers_the_two_scales_in_the_right_order(self) -> None:
        fit = _fit(_two_regime_series(), 2)
        scales = np.sqrt(np.sort(fit.covars.ravel()))
        assert scales[1] / scales[0] > 2.5
        assert fit.n_states == 2

    def test_loglikelihood_never_decreases(self) -> None:
        """EM is monotoon. Een dalende likelihood betekent een fout in de
        M-stap, en die fout is anders alleen zichtbaar als een slecht getal."""
        fit = _fit(_two_regime_series(), 2)
        history = np.asarray(fit.loglikelihood_history)
        assert history.size >= 2
        assert np.all(np.diff(history) >= -1e-8)

    def test_heavy_tails_give_a_smaller_dof_than_normal_data(self) -> None:
        """De staartparameter moet de staarten MEten. Deze vergelijking is de
        toets daarop; een absolute drempel op nu zou hier alleen de
        steekproefruis van de schatter meten."""
        rng = np.random.default_rng(3)
        gaussian = np.concatenate([
            rng.normal(0.0, 0.004, 500), rng.normal(0.0, 0.02, 500)])[:, None]
        heavy = _two_regime_series(dof=2.5, seed=11)
        assert _fit(heavy, 2).dof.max() < _fit(gaussian, 2).dof.min()

    def test_beats_its_own_starting_point_on_heavy_tails(self) -> None:
        """De EM is monotoon vanaf de start, dus de eindlikelihood ligt boven
        die van de eerste iteratie. Dat is het getal dat zegt of de extra
        parameter iets deed."""
        fit = _fit(_two_regime_series(dof=2.5, seed=11), 2)
        assert fit.loglikelihood > fit.loglikelihood_history[0]

    def test_dof_stays_inside_the_configured_bounds(self) -> None:
        fit = _fit(_two_regime_series(), 2)
        assert np.all(fit.dof >= CFG.dof_min)
        assert np.all(fit.dof <= CFG.dof_max)

    def test_hitting_the_configured_ceiling_is_reported(self) -> None:
        """De grens raken is een RESULTAAT en wordt gerapporteerd, niet
        weggerond -- anders leest een nu van precies `dof_max` als een
        schatting terwijl het een randoplossing is."""
        rng = np.random.default_rng(3)
        observations = np.concatenate([
            rng.normal(0.0, 0.004, 500), rng.normal(0.0, 0.02, 500)])[:, None]
        fit = _fit(observations, 2,
                   M2HmmConfig(dof_min=2.1, dof_init=4.0, dof_max=5.0))
        assert fit.dof.max() == pytest.approx(5.0)
        assert fit.dof_at_bound is True

    def test_is_deterministic(self) -> None:
        observations = _two_regime_series()
        a = _fit(observations, 2)
        b = _fit(observations, 2)
        assert np.array_equal(a.means, b.means)
        assert np.array_equal(a.covars, b.covars)
        assert np.array_equal(a.dof, b.dof)
        assert a.loglikelihood == b.loglikelihood

    def test_transcendentals_never_see_a_strided_view(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Bitidentiek op identieke data, ongeacht de heap.

        `np.log` op een kolom-VIEW van een (n, k)-array koos in numpy 1.26 de
        AVX-512-kernel of scalaire libm afhankelijk van waar de uitvoerbuffer
        op de heap landde, met 1 ULP verschil. `test_the_multiplier_is_causal
        [hmm3-diag-student_t]` faalde daarop: twee fits op DEZELFDE trainbars
        gaven andere `nu`. Zie de toelichting bij `log_u` in `student_t.py`.
        Deze test bewaakt de regel die dat oplost, en is in tegenstelling tot
        het symptoom niet afhankelijk van de heapgeschiedenis of de CPU.
        """
        from tradebot.regime import student_t as module

        strided: list[tuple[str, tuple[int, ...]]] = []

        class _Spy:
            def __getattr__(self, name: str) -> object:
                return getattr(np, name)

        spy = _Spy()
        for name in ("exp", "log", "log1p"):
            def watched(x, *args, _fn=getattr(np, name), _name=name, **kw):
                if (isinstance(x, np.ndarray) and x.ndim > 0
                        and not x.flags.c_contiguous):
                    strided.append((_name, x.strides))
                return _fn(x, *args, **kw)
            setattr(spy, name, watched)
        monkeypatch.setattr(module, "np", spy)

        fit = _fit(_two_regime_series(), 3)
        assert fit.n_iter_used > 1 and not fit.degenerate
        assert strided == []

    def test_transition_rows_sum_to_one(self) -> None:
        fit = _fit(_two_regime_series(), 3)
        assert np.allclose(fit.trans_mat.sum(axis=1), 1.0)
        assert np.isclose(fit.start_prob.sum(), 1.0)

    def test_non_finite_input_crashes(self) -> None:
        observations = _two_regime_series()
        observations[17, 0] = np.nan
        with pytest.raises(DataContractError):
            _fit(observations, 2)

    def test_too_few_observations_crashes(self) -> None:
        """Er wordt niet 'zo goed als het gaat' gefit: onder een handvol
        observaties per toestand is elke parameter ruis."""
        rng = np.random.default_rng(1)
        with pytest.raises(DataContractError):
            _fit(rng.normal(0.0, 0.01, 8)[:, None], 3)

    def test_a_collapsing_state_is_recorded_and_not_raised(self) -> None:
        """De t-mengselverdeling heeft een singulariteit: bij een `k` groter
        dan het aantal regimes dat de data draagt, gaat een schaal naar nul en
        de likelihood naar oneindig. Dat is geen optimum. De EM stopt daar,
        houdt de laatste geldige parameters aan en meldt het -- zelfde contract
        als `fit_garch_window`, waar een numerieke mislukking een geteld
        resultaat is en geen afgebroken campagne."""
        rng = np.random.default_rng(19)
        # Zes IDENTIEKE uitschieters: een toestand kan daar exact op landen en
        # heeft dan spreiding nul. De likelihood loopt weg, de fit niet.
        observations = rng.normal(0.0, 0.01, 300)
        observations[100:106] = 0.0731
        fit = _fit(observations[:, None], 3)
        assert fit.degenerate is True
        assert fit.converged is False
        assert fit.as_record()["degenerate"] is True
        # En wat eruit komt is BRUIKBAAR: de teruggerolde parameters halen
        # nog een filter, zodat de campagne kan descopen in plaats van te
        # crashen op een schaal van 1e-128.
        scales = np.asarray(fit.covars).ravel()
        assert float(scales.min() / scales.max()) >= CFG.min_scale_ratio

    def test_a_healthy_fit_is_not_flagged_degenerate(self) -> None:
        assert _fit(_two_regime_series(), 2).degenerate is False

    def test_reports_whether_it_converged(self) -> None:
        fit = _fit(_two_regime_series(), 2,
                   M2HmmConfig(n_iter=2, em_tolerance=1e-12))
        assert isinstance(fit, StudentTFit)
        assert fit.converged is False
