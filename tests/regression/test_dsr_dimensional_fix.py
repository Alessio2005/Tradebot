"""Regression guard for the DSR dimensional fix (CHIEF 2026-06-08).

``metrics.deflated_sharpe`` previously subtracted the expected-max of N standard
normals (≈1.67 for N=12, in z units) directly from a per-observation Sharpe
(~0.07), giving z ≈ −68 and DSR ≡ 0 for EVERY realistic strategy — a silently
dead G2 gate.  The fix scales the expected-max into SHARPE units, so the
benchmark it is compared against is a Sharpe and not a z-score.

PHASE 10, STEP 4A: the fix survives the new signature, and these tests now pin
it against MEASUREMENT_CONTRACT.md §6 rather than against the old internal form.
§6 writes the same scaling explicitly::

    SR_0 = sqrt(sr_variance) * e_max_z
    DSR  = Phi( (SR_hat - SR_0) sqrt(T - 1)
                / sqrt(1 - g3 SR_hat + ((g4 - 1)/4) SR_hat^2) )

The two forms differ only in that §6 divides ``SR_0`` by the estimator SE of the
returns rather than by the plain ``sqrt(T-1)/denominator``; with the documented
normal approximation ``sr_variance = 1/n_obs`` the largest gap on the values
below is 2.3e-3 in DSR units (measured, not assumed).  The guarded property —
and the numbers that guarded it — are therefore unchanged.  What HAS changed is
that ``sr_variance``, ``skew``, ``kurtosis`` and ``bars_per_year`` can no longer
be omitted, and that the artefact records which of the two variances was used.
"""
from __future__ import annotations

import math

import pytest

from tradebot.backtest.metrics import deflated_sharpe
from tradebot.utils.failfast import DataContractError

# 3-sleeve neutral book baseline: per-day SR=0.0658, 1815 daily obs, 12 trials.
_SR_DAY = 0.0658
_N_OBS = 1815
#: MEASUREMENT_CONTRACT.md §1. De DSR-formule gebruikt de annualisatie niet; zij
#: draagt het venster (§10) en is daarom verplicht.
_BARS_PER_YEAR = 365.0


def _dsr(sr: float, *, n_trials: int, n_obs: int = _N_OBS):
    """De §6-aanroep met de gedocumenteerde normale benadering voor V[{SR_m}]."""
    return deflated_sharpe(
        sr,
        n_obs=n_obs,
        n_trials=n_trials,
        sr_variance=1.0 / n_obs,
        skew=0.0,
        kurtosis=3.0,
        bars_per_year=_BARS_PER_YEAR,
        approximation="normal",
    )


def test_dsr_is_not_dead_for_significant_strategy() -> None:
    """A t≈2.8 strategy must yield a meaningful DSR (~0.87), not ~0."""
    assert 0.80 < _dsr(_SR_DAY, n_trials=12).dsr < 0.95


def test_dsr_matches_the_measurement_contract_formula() -> None:
    """The §6 identity, term by term — no second formulation of the same thing."""
    from scipy import stats

    sr, n, m = _SR_DAY, _N_OBS, 12
    g, e = 0.5772156649015329, math.e
    e_max_z = (1 - g) * stats.norm.ppf(1 - 1 / m) + g * stats.norm.ppf(1 - 1 / (m * e))
    sr_zero = math.sqrt(1.0 / n) * e_max_z
    denominator = math.sqrt(1.0 - 0.0 * sr + ((3.0 - 1.0) / 4.0) * sr * sr)
    expected = float(stats.norm.cdf((sr - sr_zero) * math.sqrt(n - 1) / denominator))
    got = _dsr(sr, n_trials=m)
    assert abs(got.dsr - expected) < 1e-12, (got.dsr, expected)
    assert abs(got.sr_zero - sr_zero) < 1e-12


def test_the_expected_maximum_is_a_sharpe_and_not_a_z_score() -> None:
    """THE dimensional fix, stated as the property it protects.

    The pre-2026-06-08 bug compared a per-bar Sharpe (~0.07) against the
    expected maximum of M standard normals (~1.67).  ``sr_zero`` must therefore
    live on the scale of ``sr_hat``, not on the z-scale: at T = 1815 it is
    smaller by a factor sqrt(T).
    """
    got = _dsr(_SR_DAY, n_trials=12)
    assert 0.0 < got.sr_zero < 0.1, got.sr_zero
    assert got.sr_zero * math.sqrt(_N_OBS) > 1.0


def test_dsr_monotonic_decreasing_in_n_trials() -> None:
    """More tested hypotheses ⇒ harder to clear the deflation benchmark."""
    few = _dsr(_SR_DAY, n_trials=3).dsr
    many = _dsr(_SR_DAY, n_trials=2000).dsr
    assert few > many
    assert many < 0.5  # 2000 honest hypotheses should sink a t≈2.8 result


def test_dsr_strong_strategy_passes_gate() -> None:
    """A genuinely strong strategy (t≈5) clears the 0.95 promotion gate."""
    assert _dsr(0.118, n_trials=12).dsr > 0.95


def test_the_old_positional_call_no_longer_computes_anything() -> None:
    """Step 4A.4: the pre-phase-10 shape must be VISIBLE, not silently rescored.

    ``deflated_sharpe(sr, n_trials, n_obs)`` used to run with
    ``returns_skew=0.0, returns_kurt=3.0`` hidden in its defaults.  That is the
    assumption §6 forbids, so the call shape that carried it must raise.
    """
    with pytest.raises(TypeError, match=r"MEASUREMENT_CONTRACT\.md §6"):
        deflated_sharpe(_SR_DAY, 12, _N_OBS)  # type: ignore[call-arg]
    with pytest.raises(TypeError, match=r"MEASUREMENT_CONTRACT\.md §6"):
        deflated_sharpe(  # type: ignore[call-arg]
            sr_observed=_SR_DAY, n_trials=12, n_obs=_N_OBS,
            returns_skew=0.0, returns_kurt=3.0,
        )


def test_the_variance_of_the_trial_sharpes_is_recorded_as_measured_or_assumed() -> None:
    """§6: the approximation belongs in the artefact, not in a default."""
    assumed = _dsr(_SR_DAY, n_trials=12)
    assert assumed.approximation == "normal"
    assert assumed.to_dict()["approximation"] == "normal"

    measured = deflated_sharpe(
        _SR_DAY,
        n_obs=_N_OBS,
        n_trials=12,
        sr_variance=4.0 / _N_OBS,
        skew=0.0,
        kurtosis=3.0,
        bars_per_year=_BARS_PER_YEAR,
    )
    assert measured.approximation == "empirical"
    # Wider dispersion of the trial Sharpes ⇒ a higher bar (§6, and the whole
    # point of the DSR).
    assert measured.dsr < assumed.dsr


def test_an_explicit_mislabel_raises() -> None:
    """Fixronde 1, item 7 (ruling T4A-G): the explicit path must run the same
    contradiction check as the inferred one.  ``sr_variance=4/n_obs`` is an
    empirical variance; labelling it ``"normal"`` would write a wrong label
    into the measurement artefact -- exactly what §6 exists to prevent."""
    with pytest.raises(DataContractError):
        deflated_sharpe(
            _SR_DAY,
            n_obs=_N_OBS,
            n_trials=12,
            sr_variance=4.0 / _N_OBS,
            skew=0.0,
            kurtosis=3.0,
            bars_per_year=_BARS_PER_YEAR,
            approximation="normal",
        )
    # And the symmetric case: the documented approximation labelled "empirical".
    with pytest.raises(DataContractError):
        deflated_sharpe(
            _SR_DAY,
            n_obs=_N_OBS,
            n_trials=12,
            sr_variance=1.0 / _N_OBS,
            skew=0.0,
            kurtosis=3.0,
            bars_per_year=_BARS_PER_YEAR,
            approximation="empirical",
        )
