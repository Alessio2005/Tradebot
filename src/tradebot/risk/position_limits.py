# src/tradebot/risk/position_limits.py
"""Position limit enforcement — hard caps and utilisation checks.

These checks are applied per-asset BEFORE the PortfolioRiskManager scales
the aggregate portfolio.  They represent the innermost defensive layer:
    1. Single-asset gross leverage cap    (e.g. 2× notional)
    2. Aggregate gross leverage cap       (e.g. 3× across all assets)
    3. Concentration check                (single asset > X% of gross exposure)
    4. Side symmetry check                (max net imbalance long vs. short)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

__all__ = ["PositionLimits", "PositionViolation", "check_position_limits"]


@dataclass(frozen=True)
class PositionLimits:
    """Hard position limits for a portfolio.

    Attributes
    ----------
    per_asset_cap : max gross leverage per single asset.
    portfolio_gross_cap : max sum(|leverage_i|) across all assets.
    max_concentration : max fraction of gross exposure in one asset (0.8 = 80 %).
    max_net_imbalance : max |sum(signed_leverage_i)| / gross_exposure.
    """

    per_asset_cap: float = 2.0
    portfolio_gross_cap: float = 3.0
    max_concentration: float = 0.8
    max_net_imbalance: float = 0.9


@dataclass
class PositionViolation:
    """Records a limit breach for logging / circuit-breaker triggering."""

    asset: str
    rule: str
    value: float
    limit: float

    def __str__(self) -> str:
        return f"PositionViolation({self.asset}: {self.rule}={self.value:.4f} > {self.limit:.4f})"


def check_position_limits(
    leverages: dict[str, float],
    limits: PositionLimits,
) -> list[PositionViolation]:
    """Check all position limits and return any violations.

    Parameters
    ----------
    leverages : dict mapping asset symbol to signed leverage
        (positive = long, negative = short, 0 = flat).
    limits : position limit configuration.

    Returns
    -------
    list[PositionViolation] — empty list if all checks pass.
    """
    violations: list[PositionViolation] = []
    if not leverages:
        return violations

    gross_exposures = {sym: abs(lev) for sym, lev in leverages.items()}
    gross_total = sum(gross_exposures.values())
    net_total = sum(leverages.values())

    # Per-asset gross cap
    for sym, gross in gross_exposures.items():
        if gross > limits.per_asset_cap:
            violations.append(PositionViolation(sym, "per_asset_cap", gross, limits.per_asset_cap))

    # Portfolio gross cap
    if gross_total > limits.portfolio_gross_cap:
        violations.append(
            PositionViolation("PORTFOLIO", "portfolio_gross_cap", gross_total, limits.portfolio_gross_cap)
        )

    # Concentration
    if gross_total > 1e-9:
        for sym, gross in gross_exposures.items():
            conc = gross / gross_total
            if conc > limits.max_concentration:
                violations.append(PositionViolation(sym, "concentration", conc, limits.max_concentration))

    # Net imbalance
    if gross_total > 1e-9:
        imbalance = abs(net_total) / gross_total
        if imbalance > limits.max_net_imbalance:
            violations.append(
                PositionViolation("PORTFOLIO", "net_imbalance", imbalance, limits.max_net_imbalance)
            )

    for v in violations:
        logger.warning(str(v))

    return violations
