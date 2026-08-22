# src/tradebot/train/reward.py
"""Net-alpha reward and Kalman impact observer.

Extracted from ``quant_architect.py`` (audit items III.6 and IV.7).

Contents:
    NetAlphaResult          — frozen result dataclass
    NetAlphaReward          — R = LogReturn − Slippage − Fees
    ImpactObservationRecord — one realised slippage vs. theoretical prediction
    KalmanImpactObserver    — 1-D Kalman filter for real-time η calibration

Uses :func:`~execution.market_impact.square_root_impact` via a soft-import.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from ..execution.market_impact import ImpactResult

logger = logging.getLogger("train.reward")

# Phase 0: `..execution.market_impact` is een INTERNE module binnen dit pakket en
# kan niet legitiem ontbreken. De try/except zette de vlag _MARKET_IMPACT_AVAILABLE
# op False, waarna het model stilzwijgend zonder de betreffende correctie draaide.
from ..execution.market_impact import square_root_impact


# =============================================================================
# III.6 NET-ALPHA REWARD  — R = LogReturn − Slippage − Fees
# =============================================================================
@dataclass(frozen=True)
class NetAlphaResult:
    """Output of :meth:`NetAlphaReward.compute`."""

    reward: float
    log_return: float
    slippage: float
    fees: float
    impact_eats_alpha: bool


class NetAlphaReward:
    """Real-world reward function for the Bandit (audit III.6).

    Replaces the tanh-MTM illusion with a mathematically correct net-alpha::

        R = log(P_exit / P_entry) − slippage_in − slippage_out − fees

    Slippage is computed via the square-root impact (Bouchaud-Bonart) and gives
    the Bandit a **negative reward** when market impact eats the alpha — even if
    the price moved in the right direction. This closes the feedback loop required
    by the blueprint: the bandit learns to automatically scale down order size on
    illiquid bars.

    Args:
        eta:           impact coefficient (Bouchaud ≈ 0.142).
        fee_bps:       round-trip exchange fee in basis points.
        eat_threshold: threshold above which impact_eats_alpha=True (0.8 = 80%).
    """

    def __init__(
        self,
        eta: float = 0.142,
        fee_bps: float = 4.0,
        eat_threshold: float = 0.80,
    ) -> None:
        self.eta: float = float(eta)
        self.fee_bps: float = float(fee_bps)
        self.eat_threshold: float = float(eat_threshold)

    def compute(
        self,
        price_entry: float,
        price_exit: float,
        order_size: float,
        bar_volume: float,
        sigma_per_bar: float,
        side: int = 1,
    ) -> NetAlphaResult:
        """Compute the net-alpha reward.

        Args:
            price_entry:   entry price (> 0).
            price_exit:    exit price.
            order_size:    placed size (base asset, one side).
            bar_volume:    bar volume during execution.
            sigma_per_bar: bar-σ as decimal.
            side:          +1 long, -1 short.

        Returns:
            ``NetAlphaResult`` — reward in log-return units.
        """
        if price_entry <= 0.0 or price_exit <= 0.0:
            return NetAlphaResult(
                reward=0.0, log_return=0.0, slippage=0.0, fees=0.0,
                impact_eats_alpha=True,
            )
        log_ret = float(side) * (math.log(price_exit) - math.log(price_entry))

        impact_in: ImpactResult = square_root_impact(
            order_size=order_size,
            bar_volume=bar_volume,
            sigma_per_bar=sigma_per_bar,
            eta=self.eta,
        )
        # Round-trip slippage: symmetric in/out impact.
        slippage_total = 2.0 * impact_in.impact_fraction
        fees = self.fee_bps * 1e-4

        reward = log_ret - slippage_total - fees

        gross_alpha = abs(log_ret)
        if gross_alpha > 1e-12:
            eaten_share = (slippage_total + fees) / gross_alpha
            eats = bool(eaten_share >= self.eat_threshold)
        else:
            eats = True

        return NetAlphaResult(
            reward=float(reward),
            log_return=float(log_ret),
            slippage=float(slippage_total),
            fees=float(fees),
            impact_eats_alpha=eats,
        )


# =============================================================================
# IV.7 KALMAN IMPACT OBSERVER  — η-update real-time
# =============================================================================
@dataclass
class ImpactObservationRecord:
    """One realisation of slippage vs. theoretical prediction."""

    realised_slippage: float
    sigma_per_bar: float
    participation: float   # Q / V
    eta_predicted: float
    eta_observed: float


class KalmanImpactObserver:
    """Real-time η calibration with 1-D Kalman filter (audit IV.7).

    Static η = 0.142 ignores that exchange order books thin out during news
    events — the same participation then produces double the slippage. We model
    η as a latent state with random-walk dynamics::

        η_{t+1} = η_t + w_t    , w_t ∼ N(0, Q)
        s_t     = η_t · σ · √(Q/V) + v_t , v_t ∼ N(0, R)

    where ``s_t`` is the realised slippage of own trades. Kalman update on each
    observation::

        η_obs = s_t / (σ · √(Q/V))      # implied η from the trade
        K = P_pred / (P_pred + R)
        η_t  = η_pred + K · (η_obs − η_pred)
        P_t  = (1 − K) · P_pred

    With this observer, :class:`~train.ensemble.ContextualBanditEnsemble` can
    feed :class:`NetAlphaReward` with an **adaptive** η instead of a fixed
    number, dynamically closing the simulation-to-reality gap.

    Args:
        eta_init:    starting value (Bouchaud default 0.142).
        process_var: Q (random-walk volatility of η-drift).
        obs_var:     R (measurement uncertainty of η_obs).
        eta_min/max: hard clip to catch degenerate observations (0.01—1.0 safe).
    """

    def __init__(
        self,
        eta_init: float = 0.142,
        process_var: float = 1e-5,
        obs_var: float = 1e-2,
        eta_min: float = 0.01,
        eta_max: float = 1.0,
    ) -> None:
        self.eta: float = float(np.clip(eta_init, eta_min, eta_max))
        self.P: float = 1.0  # state-covariance initial uncertainty
        self.Q: float = max(float(process_var), 1e-12)
        self.R: float = max(float(obs_var), 1e-12)
        self.eta_min: float = float(eta_min)
        self.eta_max: float = float(eta_max)
        self.history: list[ImpactObservationRecord] = []

    def update(
        self,
        realised_slippage: float,
        sigma_per_bar: float,
        order_size: float,
        bar_volume: float,
    ) -> float:
        """Update η with one realised trade.

        Args:
            realised_slippage: measured slippage as decimal (post-trade).
            sigma_per_bar:     σ of the bar on which the trade ran.
            order_size:        Q of the trade (base asset).
            bar_volume:        V of the bar.

        Returns:
            New η estimate (post-update).
        """
        sigma = max(float(sigma_per_bar), 1e-9)
        v = max(float(bar_volume), 1e-9)
        q = max(float(order_size), 0.0)
        if q <= 0.0:
            return self.eta

        participation = q / v
        sqrt_part = math.sqrt(participation)
        denom = sigma * sqrt_part
        if denom <= 1e-12:
            return self.eta

        eta_obs = float(realised_slippage) / denom
        eta_obs = float(np.clip(eta_obs, self.eta_min, self.eta_max))

        # σ²-adaptive observation variance: high-volatility bars produce noisier
        # slippage estimates (wide bid-ask, erratic fills).  Scale R inversely
        # with σ so flash-crash trades don't anchor the Kalman state to a
        # misleadingly low η_obs (which would under-estimate future impact).
        sigma_sq = sigma * sigma
        R_adaptive = self.R / max(sigma_sq, 1e-12)
        R_adaptive = float(np.clip(R_adaptive, self.R * 0.1, self.R * 100.0))

        eta_pred = self.eta
        P_pred = self.P + self.Q
        K = P_pred / (P_pred + R_adaptive)
        eta_post = eta_pred + K * (eta_obs - eta_pred)
        P_post = (1.0 - K) * P_pred

        self.eta = float(np.clip(eta_post, self.eta_min, self.eta_max))
        self.P = float(P_post)

        self.history.append(
            ImpactObservationRecord(
                realised_slippage=float(realised_slippage),
                sigma_per_bar=float(sigma),
                participation=float(participation),
                eta_predicted=float(eta_pred),
                eta_observed=float(eta_obs),
            )
        )
        if len(self.history) > 5000:
            self.history = self.history[-2500:]
        return self.eta

    def predicted_slippage(
        self,
        order_size: float,
        bar_volume: float,
        sigma_per_bar: float,
    ) -> float:
        """Theoretical slippage with current η — useful for pre-trade gating."""
        v = max(float(bar_volume), 1e-9)
        return self.eta * float(sigma_per_bar) * math.sqrt(max(order_size, 0.0) / v)


__all__ = [
    "ImpactObservationRecord",
    "KalmanImpactObserver",
    "NetAlphaResult",
    "NetAlphaReward",
]
