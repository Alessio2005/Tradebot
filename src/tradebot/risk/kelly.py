# src/tradebot/risk/kelly.py
"""Fractional Kelly position sizing.

The Kelly criterion maximises long-run log-wealth.  A full-Kelly bet is
highly aggressive and sensitive to estimation error.  Fractional Kelly
(f*_frac = f*_full / divisor) trades optimality for robustness.

Three sizing functions are provided:
  - ``kelly_fraction``      : given edge (μ) and variance (σ²) → optimal f*
  - ``gap_risk_kelly_size`` : Kelly adjusted for gap/jump risk (CVaR overlay)
  - ``meta_label_kelly``    : Kelly weighted by meta-label confidence score
"""
from __future__ import annotations

import logging
import math

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["gap_risk_kelly_size", "kelly_fraction", "meta_label_kelly"]


# CHIEF AUDIT-FIX (Sim-to-Reality #7):
#   Default divisor 6.0 implicitly assumes ~15% μ-estimation error.  Crypto
#   alpha-decay studies (Bouchaud-Bonart 2020, internal post-mortems on the
#   2022 funding-rate regime shift) show real μ-error closer to 30-50% over
#   any 30-90 day window.  At 30% error full Kelly is risk-of-ruin, even
#   1/6 Kelly draws down 40-60% during regime change.  We expose `mu_error`
#   as an explicit knob and derive the divisor:
#       divisor ≈ 1 / (1 - mu_error)^2
#   so 15% error → divisor=1.38, 30% → 2.0, 50% → 4.0, on top of the base
#   1/4 Kelly safety.  Effective divisor = base_divisor × mu_error_factor.
#   Backward-compatible: when ``mu_error`` is left at None we keep the
#   legacy 6.0 default.
def _mu_error_divisor_factor(mu_error: float) -> float:
    """Inflate the divisor based on assumed μ-estimation error.

    Derivation: an over-stated μ̂ = μ * (1 + e) inserted in Kelly yields
    f̂ ≈ f* * (1 + e).  To keep the expected log-utility on the right side
    of the bias we scale the divisor by (1 / (1 - e))² — the variance of
    log-utility around the true μ.
    """
    e = float(np.clip(mu_error, 0.0, 0.95))
    return 1.0 / max((1.0 - e) ** 2, 1e-6)


def kelly_fraction(
    mu: float,
    sigma_sq: float,
    # Wave 16 P0-19 / CHIEF AUDIT #7:
    # crypto μ-estimation error = 30-50% → effective divisor 12-24 (base 6 × inflate).
    kelly_divisor: float = 6.0,
    max_fraction: float = 0.25,
    mu_error: float | None = 0.35,  # crypto μ-error=35% → effectieve divisor ≈ 14.2 (was 6.0)
) -> float:
    """Fractional Kelly bet size for a Gaussian payoff distribution.

    Full Kelly: f* = μ / σ²
    Fractional: f = f* / kelly_divisor

    Parameters
    ----------
    mu : expected return per bar (decimal).
    sigma_sq : variance per bar (decimal²).
    kelly_divisor : base divisor for fractional Kelly (>= 1).
    max_fraction : hard upper cap on the fraction (e.g. 0.25 = 25 % notional).
    mu_error : assumed relative μ-estimation error (e.g. 0.30 = 30%).
        When provided, the effective divisor is multiplied by
        ``1 / (1 - mu_error)^2`` (see :func:`_mu_error_divisor_factor`).
        ``None`` keeps legacy behaviour.

    Returns
    -------
    float : position size fraction in [0, max_fraction].
    """
    if sigma_sq <= 1e-12 or not math.isfinite(mu) or not math.isfinite(sigma_sq):
        return 0.0
    f_full = float(mu) / float(sigma_sq)
    eff_divisor = max(float(kelly_divisor), 1.0)
    if mu_error is not None:
        eff_divisor *= _mu_error_divisor_factor(float(mu_error))
    f_frac = f_full / eff_divisor
    return float(np.clip(f_frac, 0.0, max_fraction))


def gap_risk_kelly_size(
    expected_alpha: float,
    vol: float,
    jump_sigma: float = 0.0,
    # Wave 16 P0-19 / CHIEF AUDIT #7:
    # crypto μ-estimation error = 30-50% → use mu_error parameter to inflate divisor.
    kelly_divisor: float = 6.0,
    max_fraction: float = 0.25,
    cvar_multiplier: float = 1.5,
    # v3 SK-5 FIX: mu_error=0.35 → effectieve divisor = 6.0 × (1/0.65²) ≈ 14.2.
    # Crypto μ-schattingsfout is 30-50% (regime shifts, fat-tail alpha decay).
    # Dit reduceert live leverage met ~58% tov de legacy mu_error=None (divisor=6).
    mu_error: float | None = 0.35,  # crypto μ-error=35% → effectieve divisor ≈ 14.2
) -> float:
    """Kelly size with CVaR overlay for jump/gap risk.

    Uses combined diffusive + jump volatility as the risk denominator:

        sigma_total = sqrt(vol² + jump_sigma²)
        f* = expected_alpha / (cvar_multiplier * sigma_total²)

    Parameters
    ----------
    expected_alpha : per-bar edge (e.g. from meta-label score).
    vol : diffusive vol (GK or Parkinson σ).
    jump_sigma : BNS jump-diffusion component (from volatility package).
    kelly_divisor : fractional Kelly divisor.
    max_fraction : hard cap.
    cvar_multiplier : penalty for fat tails (> 1 reduces size near jumps).

    Returns
    -------
    float : position fraction in [0, max_fraction].
    """
    sigma_total_sq = max(float(vol) ** 2 + float(jump_sigma) ** 2, 1e-12)
    adjusted_var = float(cvar_multiplier) * sigma_total_sq
    return kelly_fraction(
        mu=expected_alpha,
        sigma_sq=adjusted_var,
        kelly_divisor=kelly_divisor,
        max_fraction=max_fraction,
        mu_error=mu_error,
    )


def meta_label_kelly(
    base_size: float,
    meta_prob: float,
    meta_threshold: float = 0.5,
    scale_below_threshold: bool = False,
) -> float:
    """Scale a base position size by the meta-label confidence.

    When meta_prob >= meta_threshold, scales linearly:
        size = base_size * (meta_prob - threshold) / (1 - threshold)

    When below threshold and scale_below_threshold is False, returns 0
    (meta-label filter rejects the trade).

    Parameters
    ----------
    base_size : base Kelly fraction (already fractional Kelly).
    meta_prob : meta-label probability of profitable trade in [0, 1].
    meta_threshold : minimum probability to enter.
    scale_below_threshold : if True, scale down below threshold rather than flat-zero.

    Returns
    -------
    float : scaled position size >= 0.
    """
    p = float(np.clip(meta_prob, 0.0, 1.0))
    thr = float(np.clip(meta_threshold, 0.0, 1.0))
    b = float(base_size)

    if p < thr:
        if scale_below_threshold:
            # Linearly scale from 0 at p=0 to base_size at p=threshold
            return b * (p / max(thr, 1e-9))
        return 0.0

    # Scale from 0 at threshold to base_size at p=1
    span = max(1.0 - thr, 1e-9)
    return b * ((p - thr) / span)
