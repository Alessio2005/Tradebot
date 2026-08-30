# tests/unit/test_sharpe_difference.py
"""De gepaarde Sharpe-toets van H2 en de controles die haar geldig maken.

De toets wordt hier tegen zijn EIGENSCHAPPEN getoetst en niet tegen een tabel
met vaste getallen: dat hij nul geeft op een identiek paar, dat hij zijn
nominale niveau houdt op ruis, dat hij een ingebracht effect ziet zodra het
groot genoeg is, en dat hij op de gepaarde structuur leunt in plaats van op twee
losse Sharpes.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tradebot.utils.failfast import DataContractError
from tradebot.validation.sharpe_difference import (
    jobson_korkie_memmel,
    rescale_to_sharpe,
    run_sharpe_controls,
    stationary_bootstrap_indices,
)

BARS = 365.0
ALPHA = 0.05


def _pair(n: int = 1200, rho: float = 0.95, seed: int = 4):
    rng = np.random.default_rng(seed)
    base = rng.normal(0.0002, 0.02, n)
    noise = rng.normal(0.0, 0.02, n)
    other = rho * base + math.sqrt(1.0 - rho**2) * noise
    return base, other


class TestStatistic:
    def test_an_identical_pair_gives_no_difference(self) -> None:
        a, _ = _pair()
        result = jobson_korkie_memmel(
            a, a * 1.000001, bars_per_year=BARS, alpha=ALPHA)
        assert abs(result.difference) < 1e-6
        assert result.p_value > 0.99
        assert result.significant is False

    def test_a_large_injected_effect_is_detected(self) -> None:
        a, b = _pair()
        better = rescale_to_sharpe(a, 2.0, bars_per_year=BARS)
        result = jobson_korkie_memmel(
            better, b, bars_per_year=BARS, alpha=ALPHA, alternative="a-better")
        assert result.significant is True
        assert result.difference > 1.0

    def test_the_one_sided_form_does_not_reward_being_worse(self) -> None:
        """Een tweezijdige verwerping zegt dat de twee verschillen, niet dat de
        uitdager wint. Promotie gebruikt daarom de eenzijdige vorm."""
        a, b = _pair()
        worse = rescale_to_sharpe(a, -2.0, bars_per_year=BARS)
        two_sided = jobson_korkie_memmel(
            worse, b, bars_per_year=BARS, alpha=ALPHA)
        one_sided = jobson_korkie_memmel(
            worse, b, bars_per_year=BARS, alpha=ALPHA, alternative="a-better")
        assert two_sided.significant is True
        assert one_sided.significant is False

    def test_the_standard_error_shrinks_with_correlation(self) -> None:
        """Dit is de reden dat de pre-registratie een GEPAARDE toets eist: bij
        rho = 0,95 is het verschil vier keer preciezer te meten dan bij nul."""
        errors = []
        for rho in (0.0, 0.95):
            a, b = _pair(rho=rho)
            errors.append(jobson_korkie_memmel(
                a, b, bars_per_year=BARS, alpha=ALPHA).standard_error)
        assert errors[1] < 0.5 * errors[0]

    def test_annualisation_does_not_move_the_statistic(self) -> None:
        a, b = _pair()
        daily = jobson_korkie_memmel(a, b, bars_per_year=365.0, alpha=ALPHA)
        weekly = jobson_korkie_memmel(a, b, bars_per_year=52.0, alpha=ALPHA)
        assert daily.z_statistic == pytest.approx(weekly.z_statistic)
        assert daily.difference != pytest.approx(weekly.difference)

    def test_unequal_lengths_crash(self) -> None:
        a, b = _pair()
        with pytest.raises(DataContractError):
            jobson_korkie_memmel(a, b[:-5], bars_per_year=BARS, alpha=ALPHA)

    def test_too_few_observations_crash(self) -> None:
        a, b = _pair(n=20)
        with pytest.raises(DataContractError):
            jobson_korkie_memmel(a, b, bars_per_year=BARS, alpha=ALPHA)

    def test_a_flat_arm_crashes(self) -> None:
        a, _ = _pair()
        with pytest.raises(DataContractError):
            jobson_korkie_memmel(
                a, np.zeros_like(a), bars_per_year=BARS, alpha=ALPHA)


class TestRescale:
    def test_hits_the_target_exactly(self) -> None:
        a, _ = _pair()
        shifted = rescale_to_sharpe(a, 0.75, bars_per_year=BARS)
        got = (float(np.mean(shifted)) / float(np.std(shifted, ddof=1))
               * math.sqrt(BARS))
        assert got == pytest.approx(0.75)

    def test_only_the_mean_moves(self) -> None:
        a, _ = _pair()
        shifted = rescale_to_sharpe(a, 0.75, bars_per_year=BARS)
        assert np.std(shifted, ddof=1) == pytest.approx(np.std(a, ddof=1))
        assert np.allclose(np.diff(shifted), np.diff(a))


class TestBootstrap:
    def test_keeps_the_length_and_stays_in_range(self) -> None:
        rng = np.random.default_rng(0)
        idx = stationary_bootstrap_indices(500, 10.0, rng)
        assert idx.shape == (500,)
        assert idx.min() >= 0 and idx.max() < 500

    def test_draws_blocks_and_not_single_bars(self) -> None:
        """Bij een gemiddelde bloklengte van 10 hoort ongeveer 90 % van de
        stappen een voortzetting te zijn. Een i.i.d.-bootstrap zou de
        autocorrelatie vernietigen en de toets kunstmatig precies maken."""
        rng = np.random.default_rng(0)
        idx = stationary_bootstrap_indices(20_000, 10.0, rng)
        consecutive = np.mean(np.diff(idx) == 1)
        assert 0.85 < consecutive < 0.95

    def test_a_block_shorter_than_one_bar_crashes(self) -> None:
        with pytest.raises(DataContractError):
            stationary_bootstrap_indices(100, 0.5, np.random.default_rng(0))


class TestControls:
    def test_size_is_near_nominal_and_power_rises_with_the_effect(self) -> None:
        """De twee controles van exit-criterium 12 in één meting: de toets
        verwerpt zelden op een nul en vaker naarmate het effect groeit. Een
        toets die dat niet doet, meet niets."""
        a, b = _pair(rho=0.9)
        controls = run_sharpe_controls(
            a, b, bars_per_year=BARS, alpha=ALPHA, expected_effect=0.08,
            minimum_detectable_effect=1.5, target_power=0.8, n_replicates=200,
            mean_block_length=10.0, seed=11)
        assert controls.size_rejection_rate <= 2.0 * ALPHA
        assert controls.size_passed is True
        assert controls.power_at_mde > controls.power_at_expected_effect

    def test_a_tiny_expected_effect_fails_the_power_control(self) -> None:
        """Precies de situatie die de H2-pre-registratie vooraf beschrijft: het
        verwachte effect van 0,08 Sharpe-eenheden is te klein voor deze opzet,
        en dan draagt een niet-significante uitslag geen falsificatie."""
        a, b = _pair(rho=0.9)
        controls = run_sharpe_controls(
            a, b, bars_per_year=BARS, alpha=ALPHA, expected_effect=0.08,
            minimum_detectable_effect=1.5, target_power=0.8, n_replicates=200,
            mean_block_length=10.0, seed=11)
        assert controls.power_at_expected_effect < 0.8
        assert controls.power_passed is False

    def test_is_deterministic_given_the_seed(self) -> None:
        a, b = _pair()
        kwargs = dict(
            bars_per_year=BARS, alpha=ALPHA, expected_effect=0.08,
            minimum_detectable_effect=1.0, target_power=0.8, n_replicates=50,
            mean_block_length=10.0, seed=7)
        assert (run_sharpe_controls(a, b, **kwargs).as_record()
                == run_sharpe_controls(a, b, **kwargs).as_record())
