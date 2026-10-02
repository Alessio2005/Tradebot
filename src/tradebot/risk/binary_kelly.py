"""Kansrekening voor een 1:1-barrière (spec §10, §17.7).

Een trade met symmetrische barrières op ±b is een binaire weddenschap. Zonder
drift is de kans om eerst boven uit te komen precies 1/2; met drift mu en
volatiliteit sigma is zij 1 / (1 + exp(-2 mu b / sigma^2)). Break-even, Kelly,
de onzekerheid in p, de correlatie tussen gelijktijdige trades en de kans op
een drawdown volgen daaruit in gesloten vorm; de Monte Carlo controleert de
laatste op de echte tradeverdeling.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import stats
from scipy.special import expit

from ..utils.failfast import DataContractError, require
from ..validation.inference import circular_block_indices

__all__ = [
    "break_even_probability", "correlation_scale", "drawdown_probability",
    "first_passage_up_probability", "kelly_fraction_binary",
    "monte_carlo_drawdown_probability", "no_exit_probability",
    "posterior_lower_probability",
]


def first_passage_up_probability(mu: float, sigma: float, barrier: float) -> float:
    """P(+b eerst) voor een Brownse beweging met drift, barrières op ±b."""
    require(sigma > 0.0 and barrier > 0.0, "sigma en barrier moeten positief zijn.",
            DataContractError, sigma=sigma, barrier=barrier)
    return float(expit(2.0 * mu * barrier / sigma ** 2))


def no_exit_probability(a: float, n_terms: int = 50) -> float:
    """P(sup_{s<=1} |W_s| < a) voor standaard-Brownse beweging (reeksformule)."""
    require(a > 0.0, "a moet positief zijn.", DataContractError, a=a)
    k = np.arange(n_terms)
    terms = ((-1.0) ** k) / (2 * k + 1) * np.exp(-((2 * k + 1) ** 2) * math.pi ** 2 / (8.0 * a * a))
    return float(min(1.0, 4.0 / math.pi * terms.sum()))


def break_even_probability(barrier: float, cost: float) -> float:
    """p_be = 1/2 + c / (2b): winst b - c, verlies b + c."""
    require(barrier > 0.0 and 0.0 <= cost < barrier, "Nodig: 0 <= cost < barrier.",
            DataContractError, barrier=barrier, cost=cost)
    return 0.5 + cost / (2.0 * barrier)


def kelly_fraction_binary(p: float, barrier: float, cost: float) -> float:
    """Kelly-fractie van het vermogen dat verloren gaat bij de stop; 0 zonder edge."""
    require(0.0 <= p <= 1.0, "p buiten [0, 1].", DataContractError, p=p)
    require(barrier > 0.0 and 0.0 <= cost < barrier, "Nodig: 0 <= cost < barrier.",
            DataContractError, barrier=barrier, cost=cost)
    eps = cost / barrier
    f = (p * (1.0 - eps) - (1.0 - p) * (1.0 + eps)) / ((1.0 - eps) * (1.0 + eps))
    return float(max(f, 0.0))


def posterior_lower_probability(p_hat: float, n: float, quantile: float) -> float:
    """Kwantiel van Beta(1 + p_hat n, 1 + (1 - p_hat) n): uniforme prior plus n waarnemingen."""
    require(0.0 <= p_hat <= 1.0 and n >= 0.0 and 0.0 < quantile < 1.0,
            "Ongeldige posterior-invoer.", DataContractError, p_hat=p_hat, n=n, q=quantile)
    return float(stats.beta.ppf(quantile, 1.0 + p_hat * n, 1.0 + (1.0 - p_hat) * n))


def correlation_scale(k_same: int, rho: float) -> float:
    """Multivariate Kelly bij k gelijk-gecorreleerde, gelijk-renderende bets: 1 / (1 + (k-1) rho)."""
    require(k_same >= 1, "Ten minste één positie.", DataContractError, k_same=k_same)
    return 1.0 / (1.0 + (k_same - 1) * max(float(rho), 0.0))


def drawdown_probability(kelly_multiple: float, drawdown: float) -> float:
    """P(ooit onder (1 - drawdown) maal het huidige niveau) onder fractie-Kelly: x^(2/c - 1).

    Geldt vanaf een gegeven moment, in een continu model met juist geschatte
    edge. Over een horizon met veel nieuwe pieken is de kans op minstens één
    zo'n drawdown hoger; daarvoor is `monte_carlo_drawdown_probability`.
    """
    require(0.0 < kelly_multiple < 2.0 and 0.0 < drawdown < 1.0,
            "Nodig: 0 < c < 2 en 0 < drawdown < 1.", DataContractError)
    return float((1.0 - drawdown) ** (2.0 / kelly_multiple - 1.0))


def monte_carlo_drawdown_probability(
    r_multiples: np.ndarray,
    *,
    risk_fraction: float,
    n_trades: int,
    drawdown: float,
    block_length: int,
    n_paths: int,
    seed: int,
) -> float:
    """Kans op een maximale drawdown >= `drawdown` binnen `n_trades` trades.

    `r_multiples` is het rendement per eenheid risico, in tijdvolgorde (winst
    ~ +1, verlies ~ -1). Circulaire blokbootstrap uit `validation.inference`,
    zodat trades die dicht op elkaar liggen hun samenhang houden.
    """
    r = np.asarray(r_multiples, dtype=np.float64)
    require(r.size > 0 and bool(np.isfinite(r).all()), "Lege of niet-eindige trades.",
            DataContractError)
    require(1 <= n_trades <= r.size, "De horizon vraagt meer trades dan er gemeten zijn.",
            DataContractError, n_trades=n_trades, n_measured=int(r.size))
    rng = np.random.default_rng(seed)
    idx = circular_block_indices(r.size, max(1, int(block_length)), int(n_paths), rng)
    draws = r[idx[:, :n_trades]]
    log_eq = np.cumsum(np.log1p(np.clip(risk_fraction * draws, -0.999999, None)), axis=1)
    peak = np.maximum.accumulate(np.maximum(log_eq, 0.0), axis=1)
    worst = 1.0 - np.exp((log_eq - peak).min(axis=1))
    return float((worst >= drawdown).mean())
