"""Short straddle of strangle: undefined risk (detailontwerp 7.4).

De klasse bestaat en is getest als de rest, maar `registry.build_strategies` weigert hem zolang
`risk.allow_undefined_risk` false is. `max_loss` is `inf`. De sizing rekent met een stressverlies
in plaats daarvan: het verlies bij een beweging van `STRESS_MOVE` van spot, min het credit.
"""
from __future__ import annotations

import math
from datetime import date

from ...schemas.options import ShortVolConfig
from ..chain.builder import Chain
from ..contracts import Right
from ..iv_metrics import IVCondition
from ..positions import (
    MULTIPLIER,
    Action,
    Book,
    Candidate,
    LegIntent,
    MarketState,
    StrategyPosition,
    mark,
)
from .common import (
    Rules,
    dte,
    entry_blockers,
    iv_ok,
    pick_expiry,
    size,
    strike_at_delta,
    strike_nearest,
)
from .credit import manage_credit

__all__ = ["STRESS_MOVE", "ShortVol"]

STRESS_MOVE = 0.20


class ShortVol:
    name = "short_straddle_strangle"

    def __init__(self, cfg: ShortVolConfig, rules: Rules) -> None:
        self.cfg, self.rules = cfg, rules

    def _legs(self, chain: Chain, expiry: date) -> tuple[LegIntent, ...] | None:
        if self.cfg.style == "straddle":
            call = strike_nearest(chain, expiry, Right.CALL, chain.spot)
            put = (chain.find(expiry, call.contract.strike, Right.PUT)
                   if call is not None else None)
        else:
            call = strike_at_delta(chain, expiry, Right.CALL, self.cfg.strangle_delta)
            put = strike_at_delta(chain, expiry, Right.PUT, self.cfg.strangle_delta)
        if call is None or put is None:
            return None
        return LegIntent(put.contract, -1), LegIntent(call.contract, -1)

    def scan(self, market: MarketState, book: Book) -> list[Candidate]:
        cfg, out = self.cfg, []
        for und in sorted(market.chains):
            chain = market.chains[und]
            expiry = pick_expiry(chain, *cfg.dte, market.today)
            if expiry is None or entry_blockers(market, und, expiry, self.rules):
                continue
            ok, trace = iv_ok(market, und, IVCondition(cfg.iv_condition), self.rules)
            legs = self._legs(chain, expiry) if ok else None
            m = mark(chain, legs) if legs is not None else None
            if legs is None or m is None:
                continue
            credit = -m
            put_k, call_k = (leg.contract.strike for leg in legs)
            stress = max(put_k - chain.spot * (1 - STRESS_MOVE),
                         chain.spot * (1 + STRESS_MOVE) - call_k) - credit
            units = size(stress * MULTIPLIER, market, book, und, self.rules)
            if units == 0:
                continue
            out.append(Candidate(self.name, und, legs, units, credit, math.inf,
                                 credit / stress,
                                 (trace, self.cfg.style, f"dte:{dte(expiry, market.today)}",
                                  f"stress_loss_unit:{stress * MULTIPLIER:.2f}")))
        return sorted(out, key=lambda c: (-c.rank_score, c.underlying))

    def manage(self, pos: StrategyPosition, market: MarketState) -> list[Action]:
        c = self.cfg
        return manage_credit(pos, market, self.rules, c.take_profit_frac,
                             c.stop_loss_multiple_credit, c.close_dte)
