# src/tradebot/alpha/eq_strev.py
"""Wave 22b — equity short-term reversal, 1 month (Jegadeesh 1990).

Prior: "Evidence of Predictable Behavior of Security Returns" (JF 1990):
the prior one-month return reverses over the next month. Classically the
highest-turnover of the three OHLCV premia — the unit lives or dies on the
honest cost line (G10), which is precisely what the wave must measure.

Construction (literature-conform, FIXED):
  score(t) = -( P(t) / P(t-21) - 1 )    (short last month's winners)
  monthly rebalance, long top 30% / short bottom 30%, dollar-neutral.
"""
from __future__ import annotations

import pandas as pd

from tradebot.alpha.xs_unit import CostModel, XSUnitResult, run_xs_unit

__all__ = ["UNIT", "PRIOR", "signal_panel", "run"]

UNIT = "eq_strev_1m"
PRIOR = "Jegadeesh (1990), J. Finance 45(3), 881-898"

_LOOKBACK = 21


def signal_panel(prices: pd.DataFrame) -> pd.DataFrame:
    """Negative trailing 1-month return; row t uses closes <= t."""
    return -(prices / prices.shift(_LOOKBACK) - 1.0)


def run(
    prices: pd.DataFrame,
    membership: pd.DataFrame | None = None,
    cost: CostModel = CostModel(),
) -> XSUnitResult:
    return run_xs_unit(
        prices=prices,
        signal_panel=signal_panel(prices),
        unit=UNIT,
        membership=membership,
        rebalance="ME",
        cost=cost,
        top_frac=0.3,
        config={"lookback": _LOOKBACK, "prior": PRIOR},
    )
