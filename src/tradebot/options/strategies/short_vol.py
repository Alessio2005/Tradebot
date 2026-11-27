"""Short straddle of strangle: undefined risk (detailontwerp 7.4).

De klasse bestaat en is getest als de rest, maar `registry.build_strategies` weigert hem zolang
`risk.allow_undefined_risk` false is. `max_loss` is `inf`. De sizing rekent met een stressverlies
in plaats daarvan: het verlies bij een beweging van `STRESS_MOVE` van spot, min het credit
(`stress_loss`; per positie in USD `position_stress_loss_usd`).

Harde grenzen:

- alleen op de onderliggenden in `underlyings` (allow-list), en hoogstens `max_per_underlying`
  open posities van deze strategie per onderliggende;
- boekbreed: de risicopoort weigert (of verkleint) een entry als de som van het stressverlies van
  alle open undefined-risk posities plus de nieuwe boven `risk.max_undefined_stress_loss_frac`
  van het kapitaal komt;
- STRESS als effectief regime van de onderliggende: geen entry (`selection.admit`), en de
  engine sluit een open positie (`regime_stress`).

Beheer, in deze volgorde: de stops van `manage_credit` (expiratie, kosten ≥
`stop_loss_multiple_credit` × credit), dan een short leg met `|delta| > short_delta_exit` →
CLOSE, STOP (`short_delta`), dan winst nemen en `close_dte` van `manage_credit`.

Optionele entrypoort `min_vrp_ratio` (`null` = uit): gemiddelde short-IV ≥ ratio × σ(h)
(`common.vrp_gate`, spoorregel `vrp_h:<ratio>`).
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
    PositionStatus,
    StrategyPosition,
    Urgency,
    mark,
)
from .common import (
    Rules,
    close,
    dte,
    entry_blockers,
    iv_ok,
    pick_expiry,
    size,
    strike_at_delta,
    strike_nearest,
    vrp_gate,
)
from .credit import manage_credit

__all__ = ["STRESS_MOVE", "ShortVol", "position_stress_loss_usd", "stress_loss"]

STRESS_MOVE = 0.20


def stress_loss(put_k: float, call_k: float, spot: float, credit: float) -> float:
    """Het verlies per aandeel bij een beweging van `STRESS_MOVE` van `spot`, min het credit."""
    return max(put_k - spot * (1 - STRESS_MOVE), spot * (1 + STRESS_MOVE) - call_k) - credit


def position_stress_loss_usd(pos: StrategyPosition, spot: float) -> float:
    """Het stressverlies van een undefined-risk positie in USD, bij `spot`.

    Optiedeel, per aandeel: het maximum over de aanwezige short legs (put: K − S·(1 − m), call:
    S·(1 + m) − K), min het credit van de entry, niet onder 0; × eenheden × 100. Aandelen-legs
    (bijvoorbeeld na een vroege assignment) tellen elk met |aantal| × S × m (de beweging tegen
    ze in). Zonder legs: 0.
    """
    moves = [c.strike - spot * (1 - STRESS_MOVE) if c.right is Right.PUT
             else spot * (1 + STRESS_MOVE) - c.strike
             for c in (leg.contract for leg in pos.intents() if leg.ratio < 0)]
    options = (max(max(moves) - pos.entry_credit, 0.0) * pos.units * MULTIPLIER
               if moves else 0.0)
    shares = sum(abs(leg.qty) for leg in pos.legs if leg.contract is None)
    return options + shares * spot * STRESS_MOVE


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

    def _held(self, book: Book, underlying: str) -> int:
        return sum(1 for p in book.positions if p.strategy == self.name
                   and p.underlying == underlying and p.status is not PositionStatus.CLOSED)

    def scan(self, market: MarketState, book: Book) -> list[Candidate]:
        cfg, out = self.cfg, []
        for und in sorted(market.chains):
            if und not in cfg.underlyings or self._held(book, und) >= cfg.max_per_underlying:
                continue
            chain = market.chains[und]
            expiry = pick_expiry(chain, *cfg.dte, market.today)
            if expiry is None or entry_blockers(market, und, expiry, self.rules):
                continue
            ok, trace = iv_ok(market, und, IVCondition(cfg.iv_condition), self.rules)
            legs = self._legs(chain, expiry) if ok else None
            m = mark(chain, legs) if legs is not None else None
            if legs is None or m is None:
                continue
            vrp_ok, vrp_trace = vrp_gate(market, und, expiry, legs, cfg.min_vrp_ratio)
            if not vrp_ok:
                continue
            credit = -m
            put_k, call_k = (leg.contract.strike for leg in legs)
            stress = stress_loss(put_k, call_k, chain.spot, credit)
            units = size(stress * MULTIPLIER, market, book, und, self.rules, legs)
            if units == 0:
                continue
            out.append(Candidate(self.name, und, legs, units, credit, math.inf,
                                 credit / stress,
                                 (trace, self.cfg.style, f"dte:{dte(expiry, market.today)}",
                                  f"stress_loss_unit:{stress * MULTIPLIER:.2f}", *vrp_trace)))
        return sorted(out, key=lambda c: (-c.rank_score, c.underlying))

    def manage(self, pos: StrategyPosition, market: MarketState) -> list[Action]:
        c = self.cfg
        acts = manage_credit(pos, market, self.rules, c.take_profit_frac,
                             c.stop_loss_multiple_credit, c.close_dte)
        if acts and acts[0].urgency is Urgency.STOP:
            return acts
        chain = market.chains.get(pos.underlying)
        if chain is not None and any(self._tested(chain, leg) for leg in pos.intents()
                                     if leg.ratio < 0):
            return [close(pos, Urgency.STOP, "short_delta")]
        return acts

    def _tested(self, chain: Chain, short: LegIntent) -> bool:
        k = short.contract
        o = chain.find(k.expiry, k.strike, k.right)
        return o is not None and o.delta is not None and abs(o.delta) > self.cfg.short_delta_exit
