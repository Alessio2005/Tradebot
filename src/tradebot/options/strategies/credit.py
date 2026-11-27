"""Credit-strategieën met beperkt risico: credit spread en iron condor (detailontwerp 7.4).

Beheer, in deze volgorde (de eerste die geldt, wint):

1. handelsdag vóór een expiratie, na `close_before_expiry_time` → CLOSE, STOP;
2. kosten om te sluiten ≥ `stop_loss_multiple_credit` × credit → CLOSE, STOP;
3. (alleen credit spread, als `short_delta_exit` gezet is) short leg met |delta| >
   `short_delta_exit` → CLOSE, STOP (`delta_breach`);
4. (alleen credit spread) short leg meer dan `itm_stop_frac` ITM → CLOSE, STOP;
5. kosten ≤ (1 − `take_profit_frac`) × credit → CLOSE, NORMAL (winst genomen);
6. DTE ≤ `close_dte` → CLOSE, NORMAL.

Heeft de positie aandelen-legs (na een vroege assignment), dan zeggen de optie-legs alleen niets
meer over de waarde: de regels 2 tot en met 5 (stop, delta, ITM, winst) zwijgen dan. De engine
sluit zo'n positie zelf met reden `assigned`; de expiratie- en DTE-regel blijven gelden.

Entry: met `min_vrp_ratio` gezet moet de ATM-IV van de gekozen expiratie ≥ die ratio × σ(h)
zijn (h = DTE van die expiratie; `common.vrp_gate`, spoorregel `vrp_h:<ratio>`). Een short
call (bear call, callkant van de condor) opent niet als er voor zijn expiratie een ex-dividend
valt waarbij zijn extrinsieke waarde kleiner is dan het dividend (`short_call_dividend_blockers`).

De credit spread rangschikt alle breedtes in `wing_width_strikes` die het credit-criterium
halen op (credit − kosten) / max verlies (kosten = commissie voor openen en sluiten van beide
legs) en neemt de beste; bij gelijke stand de smalste. Short strikes van de condor liggen binnen
`short_delta ± short_delta_tolerance`, net als de credit spread een deltaband heeft.

De iron condor stuurt daarna één keer bij: heeft een short leg `|delta| > adjust_short_delta`,
dan sluit hij de ongeteste kant en opent daar een nieuwe spread op de doel-delta, met dezelfde
breedte in strikes, in dezelfde expiratie (één ADJUST-actie).
"""
from __future__ import annotations

from ...schemas.options import CreditSpreadConfig, IronCondorConfig
from ..chain.builder import Chain, PricedOption
from ..contracts import Right
from ..iv_metrics import IVCondition
from ..positions import (
    MULTIPLIER,
    Action,
    ActionKind,
    Book,
    Candidate,
    LegIntent,
    MarketState,
    StrategyPosition,
    Trend,
    Urgency,
    mark,
)
from .common import (
    Rules,
    close,
    dte,
    entry_blockers,
    expiry_close_due,
    iv_ok,
    otm_step,
    pick_expiry,
    short_call_dividend_blockers,
    size,
    strike_at_delta,
    vrp_gate,
)

__all__ = ["CreditSpread", "IronCondor", "manage_credit"]


def manage_credit(pos: StrategyPosition, market: MarketState, rules: Rules, take_profit: float,
                  stop_multiple: float, close_dte: int,
                  itm_frac: float | None = None,
                  delta_exit: float | None = None) -> list[Action]:
    expiries = pos.expiries()
    if any(expiry_close_due(e, market.now, market, rules) for e in expiries):
        return [close(pos, Urgency.STOP, "expiry")]
    if pos.shares:                      # assigned: de optie-legs alleen zeggen niets (N17)
        return ([close(pos, Urgency.NORMAL, "dte")]
                if dte(expiries[0], market.today) <= close_dte else [])
    chain = market.chains.get(pos.underlying)
    credit = pos.entry_credit
    m = mark(chain, pos.intents()) if chain is not None else None
    if m is not None:
        cost = -m
        if cost >= stop_multiple * credit:
            return [close(pos, Urgency.STOP, "stop_loss")]
    if delta_exit is not None and chain is not None:
        for leg in pos.intents():
            c = leg.contract
            o = chain.find(c.expiry, c.strike, c.right) if leg.ratio < 0 else None
            if o is not None and o.delta is not None and abs(o.delta) > delta_exit:
                return [close(pos, Urgency.STOP, "delta_breach")]
    if itm_frac is not None and chain is not None:
        for leg in pos.intents():
            c = leg.contract
            itm = (chain.spot < c.strike * (1 - itm_frac) if c.right is Right.PUT
                   else chain.spot > c.strike * (1 + itm_frac))
            if leg.ratio < 0 and itm:
                return [close(pos, Urgency.STOP, "short_itm")]
    if m is not None and -m <= (1 - take_profit) * credit:
        return [close(pos, Urgency.NORMAL, "take_profit")]
    if dte(expiries[0], market.today) <= close_dte:
        return [close(pos, Urgency.NORMAL, "dte")]
    return []


def _spread(chain: Chain, short: PricedOption, widths: range, min_frac: float,
            cost: float = 0.0) -> tuple[tuple[LegIntent, ...], float, float] | None:
    """De beste spread vanaf `short` over alle `widths` die credit ≥ `min_frac` × breedte
    halen: de hoogste (credit − `cost`) / (breedte − credit), bij gelijke stand de smalste.
    `cost` is per aandeel (commissie voor openen en sluiten)."""
    best: tuple[float, tuple[LegIntent, ...], float, float] | None = None
    for n in widths:
        long = otm_step(chain, short, n)
        if long is None:
            break
        legs = (LegIntent(short.contract, -1), LegIntent(long.contract, 1))
        m = mark(chain, legs)
        width = abs(short.contract.strike - long.contract.strike)
        if m is None or not min_frac * width <= -m < width:
            continue
        score = (-m - cost) / (width + m)
        if best is None or score > best[0]:
            best = score, legs, -m, width
    return None if best is None else best[1:]


class CreditSpread:
    name = "credit_spread"

    def __init__(self, cfg: CreditSpreadConfig, rules: Rules) -> None:
        self.cfg, self.rules = cfg, rules

    def scan(self, market: MarketState, book: Book) -> list[Candidate]:
        cfg, out = self.cfg, []
        for und in sorted(market.chains):
            chain, trend = market.chains[und], market.trend.get(und)
            if trend not in (Trend.UP, Trend.DOWN):
                continue
            right = Right.PUT if trend is Trend.UP else Right.CALL
            expiry = pick_expiry(chain, *cfg.dte, market.today)
            if expiry is None or entry_blockers(market, und, expiry, self.rules):
                continue
            ok, iv_trace = iv_ok(market, und, IVCondition(cfg.iv_condition), self.rules)
            short = strike_at_delta(chain, expiry, right, cfg.short_delta_target,
                                    cfg.short_delta)
            if not ok or short is None:
                continue
            found = _spread(chain, short, range(cfg.wing_width_strikes[0],
                                                cfg.wing_width_strikes[1] + 1),
                            cfg.min_credit_frac_width, 4 * self.rules.commission_per_contract / MULTIPLIER)
            if found is None:
                continue
            legs, credit, width = found
            if short_call_dividend_blockers(market, und, legs):
                continue
            vrp_ok, vrp_trace = vrp_gate(market, und, expiry, cfg.min_vrp_ratio)
            if not vrp_ok:
                continue
            loss_unit = (width - credit) * MULTIPLIER
            units = size(loss_unit, market, book, und, self.rules, legs)
            if units == 0:
                continue
            kind = "bull_put" if right is Right.PUT else "bear_call"
            out.append(Candidate(self.name, und, legs, units, credit, loss_unit * units,
                                 credit / (width - credit),
                                 (f"trend:{trend.value}", kind, iv_trace,
                                  f"dte:{dte(expiry, market.today)}", f"width:{width:g}",
                                  *vrp_trace)))
        return sorted(out, key=lambda c: (-c.rank_score, c.underlying))

    def manage(self, pos: StrategyPosition, market: MarketState) -> list[Action]:
        c = self.cfg
        return manage_credit(pos, market, self.rules, c.take_profit_frac,
                             c.stop_loss_multiple_credit, c.close_dte, c.itm_stop_frac,
                             c.short_delta_exit)


class IronCondor:
    name = "iron_condor"

    def __init__(self, cfg: IronCondorConfig, rules: Rules) -> None:
        self.cfg, self.rules = cfg, rules

    def _band(self) -> tuple[float, float]:
        """De deltaband van de short strikes: `short_delta ± short_delta_tolerance`."""
        d, tol = self.cfg.short_delta, self.cfg.short_delta_tolerance
        return d - tol, d + tol

    def _condor(self, chain: Chain, sp: PricedOption, sc: PricedOption
                ) -> tuple[tuple[LegIntent, ...], float, float, int] | None:
        lo, hi = self.cfg.wing_width_strikes
        for n in range(lo, hi + 1):
            lp, lc = otm_step(chain, sp, n), otm_step(chain, sc, n)
            if lp is None or lc is None:
                return None
            legs = (LegIntent(lp.contract, 1), LegIntent(sp.contract, -1),
                    LegIntent(sc.contract, -1), LegIntent(lc.contract, 1))
            m = mark(chain, legs)
            width = max(sp.contract.strike - lp.contract.strike,
                        lc.contract.strike - sc.contract.strike)
            if m is not None and -m >= self.cfg.min_credit_frac_width * width:
                return legs, -m, width, n
        return None

    def scan(self, market: MarketState, book: Book) -> list[Candidate]:
        cfg, out = self.cfg, []
        for und in sorted(market.chains):
            chain = market.chains[und]
            expiry = pick_expiry(chain, *cfg.dte, market.today)
            if expiry is None or entry_blockers(market, und, expiry, self.rules):
                continue
            ok, iv_trace = iv_ok(market, und, IVCondition(cfg.iv_condition), self.rules)
            band = self._band()
            sp = strike_at_delta(chain, expiry, Right.PUT, cfg.short_delta, band)
            sc = strike_at_delta(chain, expiry, Right.CALL, cfg.short_delta, band)
            if not ok or sp is None or sc is None or sp.contract.strike >= sc.contract.strike:
                continue
            found = self._condor(chain, sp, sc)
            if found is None:
                continue
            legs, credit, width, n = found
            if short_call_dividend_blockers(market, und, legs):
                continue
            vrp_ok, vrp_trace = vrp_gate(market, und, expiry, cfg.min_vrp_ratio)
            if not vrp_ok:
                continue
            loss_unit = (width - credit) * MULTIPLIER
            units = size(loss_unit, market, book, und, self.rules, legs)
            if units == 0:
                continue
            out.append(Candidate(self.name, und, legs, units, credit, loss_unit * units,
                                 credit / (width - credit),
                                 (iv_trace, f"dte:{dte(expiry, market.today)}",
                                  f"wing_strikes:{n}", f"width:{width:g}", *vrp_trace)))
        return sorted(out, key=lambda c: (-c.rank_score, c.underlying))

    def manage(self, pos: StrategyPosition, market: MarketState) -> list[Action]:
        c = self.cfg
        acts = manage_credit(pos, market, self.rules, c.take_profit_frac,
                             c.stop_loss_multiple_credit, c.close_dte)
        if acts or pos.adjustments >= c.max_adjustments or pos.shares:
            return acts
        adj = self._adjust(pos, market)
        return [adj] if adj is not None else []

    def _adjust(self, pos: StrategyPosition, market: MarketState) -> Action | None:
        chain = market.chains.get(pos.underlying)
        if chain is None:
            return None
        legs = pos.intents()
        for short in (leg for leg in legs if leg.ratio < 0):
            o = chain.find(short.contract.expiry, short.contract.strike, short.contract.right)
            if o is None or o.delta is None or abs(o.delta) <= self.cfg.adjust_short_delta:
                continue
            untested = Right.CALL if short.contract.right is Right.PUT else Right.PUT
            side = tuple(leg for leg in legs if leg.contract.right is untested)
            old_short = next(leg for leg in side if leg.ratio < 0).contract
            old_long = next(leg for leg in side if leg.ratio > 0).contract
            strikes = [x.contract.strike for x in chain.by_expiry(old_short.expiry, untested)]
            if old_short.strike not in strikes or old_long.strike not in strikes:
                return None
            n = abs(strikes.index(old_short.strike) - strikes.index(old_long.strike))
            new_short = strike_at_delta(chain, old_short.expiry, untested, self.cfg.short_delta,
                                        self._band())
            new_long = otm_step(chain, new_short, n) if new_short is not None else None
            if new_short is None or new_long is None:
                return None
            return Action(ActionKind.ADJUST, pos.position_id, side, Urgency.NORMAL,
                          f"adjust:{short.contract.right.name.lower()}_tested",
                          (LegIntent(new_short.contract, -1), LegIntent(new_long.contract, 1)))
        return None
