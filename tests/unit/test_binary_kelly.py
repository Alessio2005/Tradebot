from __future__ import annotations

import math

import numpy as np
import pytest

from tradebot.risk.binary_kelly import (
    break_even_probability,
    correlation_scale,
    drawdown_probability,
    first_passage_up_probability,
    kelly_fraction_binary,
    monte_carlo_drawdown_probability,
    no_exit_probability,
    posterior_lower_probability,
)


def test_without_drift_the_barrier_is_a_coin() -> None:
    assert first_passage_up_probability(0.0, 0.03, 0.07) == pytest.approx(0.5)
    p = first_passage_up_probability(0.004, 0.03, 0.07)
    assert p + first_passage_up_probability(-0.004, 0.03, 0.07) == pytest.approx(1.0)
    assert p > 0.5


def test_first_passage_matches_a_simulated_brownian_motion() -> None:
    rng = np.random.default_rng(0)
    mu, sigma, b, dt = 0.5, 1.0, 0.5, 1e-3
    n_paths, n_steps = 4000, 20000
    x = np.zeros(n_paths)
    hit = np.zeros(n_paths)
    alive = np.ones(n_paths, dtype=bool)
    for _ in range(n_steps):
        x[alive] += mu * dt + sigma * math.sqrt(dt) * rng.normal(size=int(alive.sum()))
        up, down = alive & (x >= b), alive & (x <= -b)
        hit[up] = 1.0
        alive &= ~(up | down)
        if not alive.any():
            break
    assert hit.mean() == pytest.approx(first_passage_up_probability(mu, sigma, b), abs=0.03)


def test_the_vertical_barrier_share_of_the_spec() -> None:
    assert no_exit_probability(math.sqrt(0.5)) == pytest.approx(0.108, abs=1e-3)
    assert no_exit_probability(0.5) < no_exit_probability(1.0) < no_exit_probability(3.0) <= 1.0


def test_break_even_and_kelly_meet_at_zero() -> None:
    p_be = break_even_probability(0.09, 0.0025)
    assert p_be == pytest.approx(0.5 + 0.0025 / 0.18)
    assert kelly_fraction_binary(p_be, 0.09, 0.0025) == pytest.approx(0.0, abs=1e-12)
    assert kelly_fraction_binary(0.6, 0.09, 0.0) == pytest.approx(0.2)
    assert kelly_fraction_binary(0.4, 0.09, 0.0) == 0.0


def test_the_posterior_lower_bound_shrinks_toward_the_estimate_with_data() -> None:
    lo_small = posterior_lower_probability(0.58, 10, 0.25)
    lo_big = posterior_lower_probability(0.58, 2000, 0.25)
    assert lo_small < lo_big < 0.58
    assert posterior_lower_probability(0.58, 0, 0.25) == pytest.approx(0.25)


def test_the_equicorrelated_kelly_correction() -> None:
    assert correlation_scale(1, 0.75) == 1.0
    assert correlation_scale(3, 0.75) == pytest.approx(1.0 / 2.5)
    assert correlation_scale(3, -0.2) == 1.0


def test_the_fractional_kelly_drawdown_formula() -> None:
    assert drawdown_probability(0.25, 0.25) == pytest.approx(0.75 ** 7)
    assert drawdown_probability(0.5, 0.25) == pytest.approx(0.75 ** 3)


def test_monte_carlo_drawdown_is_certain_or_impossible_at_the_extremes() -> None:
    losses = np.full(50, -1.0)
    wins = np.full(50, 1.0)
    kw = dict(risk_fraction=0.1, n_trades=10, drawdown=0.25, block_length=2, n_paths=500, seed=1)
    assert monte_carlo_drawdown_probability(losses, **kw) == 1.0
    assert monte_carlo_drawdown_probability(wins, **kw) == 0.0
