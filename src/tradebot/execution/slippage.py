# src/tradebot/execution/slippage.py
"""Slippage models for simulation-to-reality gap correction.

Three models of increasing sophistication:
  1. Fixed-bps    : constant taker fee + spread (simplest; use as floor)
  2. Linear vol   : slippage ∝ σ_bar (assumes spread ~ vol regime)
  3. Impact-aware : delegates to :func:`.market_impact.square_root_impact`
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger(__name__)

__all__ = ["SlippageModel", "compute_slippage", "fixed_bps_slippage", "vol_proportional_slippage"]


@dataclass(frozen=True)
class SlippageModel:
    """Configuration for a slippage model.

    Attributes
    ----------
    mode : "fixed" | "vol_linear" | "sqrt_impact"
        Slippage calculation mode.
    fixed_bps : float
        One-way cost in basis points used in "fixed" mode and as a floor
        in other modes.
    vol_scale : float
        Multiplier on bar volatility (sigma) in "vol_linear" mode.
    impact_eta : float
        Almgren-Chriss eta parameter (see ``square_root_impact``) used in
        "sqrt_impact" mode.
    floor_bps : float
        Hard floor on the slippage in bps regardless of mode.  Acts as a
        safety net against pathological low-vol inputs.

    CHIEF AUDIT-FIX (Sim-to-Reality #6):
      Floor was hard-coded to 2 bps in ``vol_proportional_slippage`` which
      systematically underestimated thin-book spread widening (5-10 bps on
      crypto perps during off-hours / weekends).  Default raised to 5 bps
      and observed bid/ask spread is now accepted as an additional floor
      via :func:`compute_slippage`.
    """

    mode: Literal["fixed", "vol_linear", "sqrt_impact"] = "fixed"
    fixed_bps: float = 5.0
    vol_scale: float = 1.0
    impact_eta: float = 0.1
    floor_bps: float = 5.0


def fixed_bps_slippage(bps: float, price: float) -> float:
    """Return one-way absolute slippage from a fixed basis-point cost.

    Parameters
    ----------
    bps : basis points (e.g. 5.0 = 5 bps = 0.05 %).
    price : mid-price.

    Returns
    -------
    float : absolute price slippage (positive = adverse fill).
    """
    return max(float(bps), 0.0) * 1e-4 * abs(float(price))


def vol_proportional_slippage(
    sigma_bar: float,
    price: float,
    vol_scale: float = 1.0,
    floor_bps: float = 5.0,
    observed_spread_bps: float = 0.0,
) -> float:
    """Slippage proportional to bar volatility, with a fixed-bps floor.

    CHIEF AUDIT-FIX (Sim-to-Reality #6):
      Old default floor=2 bps under-estimated thin-book widening on crypto
      perps during off-hours / weekends (observed 5-10 bps in BTC/ETH and
      up to 20 bps in SOL).  Raised to 5 bps and an optional
      ``observed_spread_bps`` adds a hard lower bound from realised market
      data — when the live spread is wider than the floor we use that.

    Parameters
    ----------
    sigma_bar : bar-level volatility (log-return units).
    price : mid-price.
    vol_scale : multiplier on sigma.
    floor_bps : minimum slippage in basis points.
    observed_spread_bps : half-spread from live order book; if supplied
        and larger than ``floor_bps``, it becomes the effective floor.

    Returns
    -------
    float : absolute price slippage.
    """
    vol_slip = max(float(sigma_bar), 0.0) * float(vol_scale) * abs(float(price))
    eff_floor_bps = max(float(floor_bps), float(observed_spread_bps))
    floor_slip = fixed_bps_slippage(eff_floor_bps, price)
    return max(vol_slip, floor_slip)


def compute_slippage(
    model: SlippageModel,
    price: float,
    order_size: float = 0.0,
    bar_volume: float = 1.0,
    sigma_bar: float = 0.0,
    observed_spread_bps: float = 0.0,
    crisis_multiplier: float = 1.0,
) -> float:
    """Dispatch to the configured slippage model.

    Parameters
    ----------
    model : SlippageModel configuration.
    price : mid-price at execution.
    order_size : size of the order in base-asset units (for impact models).
    bar_volume : realised bar volume in the same units as order_size.
                 Replaces the old ``adv`` parameter (P0-16 fix: match
                 ``square_root_impact`` signature which expects bar_volume,
                 not average daily volume).
    sigma_bar : bar volatility (for vol-proportional models).

    Returns
    -------
    float : one-way absolute price slippage (>= 0).
    """
    # CHIEF AUDIT-FIX (Sim-to-Reality #6): unified floor = max(model.floor_bps,
    # observed_spread_bps).  Every mode applies it; live callers can pass the
    # live half-spread to bring the simulation slippage in line with reality.
    eff_floor_bps = max(float(model.floor_bps), float(observed_spread_bps))
    floor_slip = fixed_bps_slippage(eff_floor_bps, price)

    if model.mode == "fixed":
        return max(fixed_bps_slippage(model.fixed_bps, price), floor_slip)

    if model.mode == "vol_linear":
        return vol_proportional_slippage(
            sigma_bar, price,
            vol_scale=model.vol_scale,
            floor_bps=eff_floor_bps,
            observed_spread_bps=observed_spread_bps,
        )

    if model.mode == "sqrt_impact":
        from .market_impact import square_root_impact
        # P0-16: use bar_volume (not adv) and sigma_per_bar (not sigma) to match
        # square_root_impact(order_size, bar_volume, sigma_per_bar, *, eta, ...).
        result = square_root_impact(
            order_size=order_size,
            bar_volume=bar_volume,
            sigma_per_bar=sigma_bar,
            eta=model.impact_eta,
            crisis_multiplier=crisis_multiplier,
        )
        impact_slip = result.impact_fraction * abs(float(price))
        return max(impact_slip, floor_slip)

    logger.warning("Unknown SlippageModel.mode=%r — falling back to fixed.", model.mode)
    return max(fixed_bps_slippage(model.fixed_bps, price), floor_slip)
