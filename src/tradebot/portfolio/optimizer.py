# src/tradebot/portfolio/optimizer.py
"""Unified portfolio optimisation API.

Single entry-point ``optimize()`` that dispatches to HRP, BL, ERC or MVO
based on the ``method`` parameter.

Design: thin dispatcher + constraint application.  Each sub-module owns
its own algorithm; this module owns only the routing and post-processing.
"""
from __future__ import annotations

import logging
from typing import Literal

import pandas as pd

from ..utils.failfast import ConfigContractError, require
from .black_litterman import BLViews, black_litterman_weights
from .constraints import PortfolioConstraints, apply_constraints
from .hrp import HrpResearchGate, hrp_weights
from .markowitz import min_variance_weights, mvo_weights
from .risk_parity import erc_weights

logger = logging.getLogger(__name__)

__all__ = ["optimize", "OptimisationMethod"]

OptimisationMethod = Literal["hrp", "bl", "erc", "mvo", "minvar"]


def optimize(
    returns: pd.DataFrame,
    constraints: PortfolioConstraints,
    method: OptimisationMethod,
    current_weights: pd.Series | None = None,
    views: BLViews | None = None,
    expected_returns: pd.Series | None = None,
    hrp_research_gate: HrpResearchGate | None = None,
) -> pd.Series:
    """Compute portfolio weights using the requested method.

    Parameters
    ----------
    returns :
        Asset returns DataFrame (rows=time, cols=assets).
        Minimum 5 rows; recommend 60+ for reliable estimation.
    method :
        Optimisation method.  One of:
        - ``"hrp"``    : Hierarchical Risk Parity — **RESEARCH ONLY**,
                         vereist `hrp_research_gate` (zie hieronder)
        - ``"bl"``     : Black-Litterman (requires ``views``)
        - ``"erc"``    : Equal Risk Contribution
        - ``"mvo"``    : Mean-Variance (requires ``expected_returns``)
        - ``"minvar"`` : Global Minimum Variance
    constraints :
        Portfolio constraints.  VERPLICHT sinds Phase 5: er is geen default
        meer.  De vorige default (`PortfolioConstraints()` met max_weight=0.40
        en max_leverage=1.00) was een tweede risicopolicy die stilzwijgend
        gold voor elke caller die het argument oversloeg, en die bovendien
        afweek van `conf/risk/gross_cap` (1.50).  Bouw hem met
        `PortfolioConstraints.from_risk_config(risk_cfg)`.
    hrp_research_gate :
        VERPLICHT wanneer `method='hrp'`. Er is geen default.
        `method` zelf heeft sinds Phase 7/8 evenmin een default: hij
        stond op `"hrp"`, waardoor elke aanroeper die het argument
        oversloeg een research-only allocator kreeg. Dat is exact het
        patroon dat Phase 5 bij `constraints` heeft opgeruimd.

    current_weights :
        Current portfolio weights for turnover limiting.
    views :
        BLViews (required for ``method="bl"``).
    expected_returns :
        Expected returns Series (optional for ``method="mvo"``).

    Returns
    -------
    pd.Series of constrained portfolio weights summing to 1.
    """
    require(
        isinstance(constraints, PortfolioConstraints),
        "optimize() vereist een expliciete PortfolioConstraints. De oude "
        "default gaf L8 zijn eigen concentratie- en leveragelimiet; zie "
        "reports/phase5_sovereign_wiring_audit.md D1/D2.",
        ConfigContractError,
        got=type(constraints).__name__,
    )

    if len(returns) < 5:
        logger.warning("Insufficient returns for optimisation (n=%d) — equal weight.", len(returns))
        n = len(returns.columns)
        raw = pd.Series(1.0 / n, index=returns.columns)
        return apply_constraints(raw, constraints, current_weights)

    if method == "hrp":
        # PHASE 7/8 STAGE C-1. Fase-6 no-go 15 luidde "HRP is
        # productie-toegankelijk zonder bewijs". Dat was een understatement:
        # HRP was de DEFAULT-methode van deze functie, dus `optimize(returns,
        # constraints)` zonder verdere argumenten leverde HRP-gewichten.
        #
        # `method` heeft daarom geen default meer, en dit pad eist het token.
        require(
            hrp_research_gate is not None,
            "optimize(method='hrp') zonder hrp_research_gate. HRP is Research "
            "Track (audit §13.1) en technisch geblokkeerd voor productie tot "
            "turnover-gecorrigeerde OOS-superioriteit boven Inverse Volatility "
            "is aangetoond én de boomordening stabiel is (fase-6 no-go 11). "
            "Kies een toegelaten methode, of geef een HrpResearchGate mee met "
            "de reden waarom dit onderzoek is.",
            ConfigContractError,
            method=method,
        )
        raw = hrp_weights(
            returns, research_gate=hrp_research_gate).weights
    elif method == "bl":
        raw = black_litterman_weights(returns, views=views)
    elif method == "erc":
        raw = erc_weights(returns)
    elif method == "mvo":
        raw = mvo_weights(returns, expected_returns=expected_returns)
    elif method == "minvar":
        raw = min_variance_weights(returns)
    else:
        raise ValueError(f"Unknown optimisation method: {method!r}")

    return apply_constraints(raw, constraints, current_weights)
