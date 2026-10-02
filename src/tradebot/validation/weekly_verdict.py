"""Het oordeel valt op de BEVROREN stop-criteria van de preregistratie (spec §14).

Geen drempel staat in deze module. Een negatieve controle die bindt maakt de run
ongeldig; een falsify-criterium falsifieert; elk ander bindend criterium laat de
hypothese onbewezen. Een criterium zonder meting crasht: een ontbrekend getal
mag nooit als "niet bindend" worden gelezen.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from ..registry.preregistration import StopCriterion
from ..utils.failfast import DataContractError, require

__all__ = ["DEVELOPMENT_METRICS", "HOLDOUT_METRICS", "Verdict", "judge"]

DEVELOPMENT_METRICS = (
    "max_abs_shuffle_auc_deviation", "reversed_minus_baseline_sharpe",
    "ensemble_net_sharpe", "sharpe_diff_ci_low", "sharpe_ci_low",
    "hit_rate_ci_low_minus_break_even", "pbo", "mc_drawdown_probability_1y",
    "n_trades",
)
HOLDOUT_METRICS = ("holdout_brier_diff_ci_low",)
Status = Literal["PASS", "INVALID", "UNPROVEN", "FALSIFIED"]


@dataclass(frozen=True)
class Verdict:
    status: Status
    binding: tuple[str, ...]
    values: dict[str, float]

    def as_dict(self) -> dict[str, Any]:
        return {"status": self.status, "binding": list(self.binding), "values": self.values}


def judge(
    criteria: Sequence[StopCriterion],
    values: Mapping[str, float],
    *,
    stage: Literal["development", "holdout"],
) -> Verdict:
    in_stage = [c for c in criteria
                if c.action != "promote"
                and c.name.startswith("holdout_") == (stage == "holdout")]
    missing = sorted({c.metric for c in in_stage} - set(values))
    require(not missing, "Een stop-criterium zonder meting.", DataContractError,
            missing=missing, stage=stage)
    measured = {c.metric: float(values[c.metric]) for c in in_stage}
    require(all(np.isfinite(v) for v in measured.values()),
            "Een niet-eindige meting.", DataContractError, values=measured)
    binding = [c for c in in_stage if c.binds(measured[c.metric])]
    if any(c.name.startswith("negative_control") for c in binding):
        status: Status = "INVALID"
    elif any(c.action == "falsify" for c in binding):
        status = "FALSIFIED"
    elif binding:
        status = "UNPROVEN"
    else:
        status = "PASS"
    return Verdict(status=status, binding=tuple(c.name for c in binding), values=measured)
