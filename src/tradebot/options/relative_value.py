"""Relative value op de VRP: de rangorde van toegelaten kandidaten, nooit een poort.

Per scanronde wordt de VRP-ratio (ATM-IV30 / σ(30 d)) van elke onderliggende met bekende
waarde omgezet naar een z-score over het universum (populatie-σ). Een rijke onderliggende
(z > 0) krijgt een hogere rangscore voor kandidaten die premie ontvangen, en een lagere voor
kandidaten die premie betalen: verkopen waar de volatiliteit relatief duur is, kopen waar ze
relatief goedkoop is. Met minder dan drie namen of σ = 0 zijn alle z-scores 0.

`rank_score` wordt vermenigvuldigd met `1 + w·z` (credit) of `1 − w·z` (debit), begrensd op
[0,5, 1,5]. Er valt niets af: dit verandert alleen de volgorde.
"""
from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping

from .positions import Candidate
from .vol_regime import VolState

__all__ = ["MAX_FACTOR", "MIN_FACTOR", "MIN_NAMES", "rv_adjust", "vrp_zscores"]

MIN_NAMES = 3
MIN_FACTOR, MAX_FACTOR = 0.5, 1.5
VRP_HORIZON_DAYS = 30


def vrp_zscores(vol: Mapping[str, VolState]) -> dict[str, float]:
    """z-score van de VRP-ratio per onderliggende; onderliggenden zonder IV of RV krijgen 0."""
    ratios: dict[str, float] = {}
    for und, st in vol.items():
        if st.atm_iv is None:
            continue
        vrp = st.vrp_at(st.atm_iv, VRP_HORIZON_DAYS)
        if vrp is not None:
            ratios[und] = vrp[0]
    z = dict.fromkeys(vol, 0.0)
    if len(ratios) < MIN_NAMES:
        return z
    mean = sum(ratios.values()) / len(ratios)
    sd = math.sqrt(sum((r - mean) ** 2 for r in ratios.values()) / len(ratios))
    if sd == 0.0:
        return z
    z.update({u: (r - mean) / sd for u, r in ratios.items()})
    return z


def rv_adjust(c: Candidate, z: float, weight: float) -> Candidate:
    """`c` met herwogen `rank_score` en de spoorregel `rv_z:<z>`."""
    sign = 1.0 if c.net_credit > 0 else -1.0
    factor = min(MAX_FACTOR, max(MIN_FACTOR, 1.0 + sign * weight * z))
    return dataclasses.replace(c, rank_score=c.rank_score * factor,
                               rule_trace=(*c.rule_trace, f"rv_z:{z:.2f}"))
