# src/tradebot/alpha/fx_carry.py
"""Wave 25 — G10 FX carry (Koijen, Moskowitz, Pedersen & Vrugt 2018).

Prior: "Carry" (JFE 2018) — sorting currencies on interest-rate
differential vs USD delivers a positive premium (the classic carry trade;
also Lustig-Roussanov-Verdelhan 2011). Documented across decades and
universes, survives transaction costs in majors.

Construction (literature-conform, FIXED):
  signal(t) = i_fx(t) − i_usd(t)   (3m rates, PIT/as-of, percent)
  monthly rebalance, long top 1/3 / short bottom 1/3 over 9 G10-crosses,
  dollar-neutral, traded on FRED-based total-return indices.

Costs: majors 1 bp half-spread, no commission/borrow (FX swap-implied).
F7/F8 note: those falsifications cover crypto trend — FX carry is a
different, documented premium class (mandate §5.3).
"""
from __future__ import annotations

import pandas as pd

from tradebot.alpha.xs_unit import CostModel, XSUnitResult, run_xs_unit

__all__ = ["UNIT", "PRIOR", "run"]

UNIT = "fx_carry_g10"
PRIOR = "Koijen, Moskowitz, Pedersen & Vrugt (2018), J. Fin. Econ. 127(2), 197-225"

_COST = CostModel(commission_bps=0.0, half_spread_bps=1.0, borrow_fee_ann=0.0)


def run(
    tr_index: pd.DataFrame,
    carry_signal: pd.DataFrame,
    cost: CostModel = _COST,
) -> XSUnitResult:
    return run_xs_unit(
        prices=tr_index,
        signal_panel=carry_signal,
        unit=UNIT,
        membership=None,
        rebalance="ME",
        cost=cost,
        top_frac=1.0 / 3.0,
        min_names=6,
        config={"universe": "G10ex-USD", "rate_tenor": "3m", "prior": PRIOR},
    )
