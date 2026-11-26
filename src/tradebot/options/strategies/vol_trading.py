"""Volatiliteitshandel: gamma scalping (`strategies.gamma_scalp`).

Een long ATM-straddle, gekocht als de IV goedkoop is tegenover de verwachte RV over de looptijd,
en delta-neutraal gehouden door de hedger (`options/hedging.py`) met aandelen. De winst komt uit
het bijsturen: koop laag, verkoop hoog zolang de gerealiseerde vol boven de betaalde IV ligt.

Entry, per onderliggende in `universe.share_eligible`:

* de expiratie met DTE in `dte` (het dichtst bij het midden), strike het dichtst bij de forward;
* IV van die expiratie = het gemiddelde van de call- en de put-IV van de straddle;
* IV ≤ `max_iv_rv_ratio` × σ(h), σ(h) = `RVForecast.at_horizon(DTE)`;
* geen earnings in de looptijd (en verse events); sizing via `common.size` op het debit;
* `rank_score` = σ(h) / IV − 1.

Beheer, de eerste regel die geldt:

1. de expiratieregel (7.5);
2. waarde ≥ `take_profit_multiple_debit` × debit;
3. waarde ≤ `stop_loss_frac_debit` × debit;
4. IV ≥ σ(h) voor de resterende DTE: de goedkope vol is weg (`iv_normalized`);
5. DTE ≤ `close_dte`.

"Waarde" is per eenheid en per aandeel, inclusief de P&L van de hedge: wat het bijsturen al
vastlegde (`hedge_realized_usd`, na commissie) plus het open deel van de aandelen-legs:
`mark + (hedge_realized_usd + Σ_aandelen-legs qty × (spot − entry_price)) / (units × 100)`.
"""
from __future__ import annotations

from collections.abc import Collection, Sequence
from datetime import date

from ...schemas.options import GammaScalpConfig
from ..chain.builder import Chain
from ..contracts import Right
from ..positions import (
    MULTIPLIER,
    Action,
    Book,
    Candidate,
    LegIntent,
    MarketState,
    StrategyPosition,
    Urgency,
    mark,
)
from .common import (
    Rules,
    close,
    dte,
    entry_blockers,
    expiry_close_due,
    pick_expiry,
    size,
    strike_nearest,
)
from .debit import _level

__all__ = ["GammaScalp"]


def _iv(chain: Chain, legs: Sequence[LegIntent]) -> float | None:
    """Gemiddelde IV van de legs (de call en de put); None als een leg geen IV heeft."""
    ivs = []
    for leg in legs:
        c = leg.contract
        o = chain.find(c.expiry, c.strike, c.right)
        if o is None or o.iv is None:
            return None
        ivs.append(o.iv)
    return sum(ivs) / len(ivs)


def _hedge_pnl(pos: StrategyPosition, spot: float) -> float:
    """P&L van de hedge, per eenheid en per aandeel: het gerealiseerde deel
    (`hedge_realized_usd`) plus het open deel van de aandelen-legs tegen spot."""
    open_pnl = sum(leg.qty * (spot - leg.entry_price) for leg in pos.legs
                   if leg.contract is None)
    return (pos.hedge_realized_usd + open_pnl) / (pos.units * MULTIPLIER)


class GammaScalp:
    name = "gamma_scalp"

    def __init__(self, cfg: GammaScalpConfig, rules: Rules,
                 share_eligible: Collection[str]) -> None:
        self.cfg, self.rules, self.eligible = cfg, rules, frozenset(share_eligible)

    def _legs(self, chain: Chain, expiry: date) -> tuple[LegIntent, LegIntent] | None:
        call = strike_nearest(chain, expiry, Right.CALL, _level(chain, expiry))
        put = chain.find(expiry, call.contract.strike, Right.PUT) if call is not None else None
        if call is None or put is None:
            return None
        return LegIntent(put.contract, 1), LegIntent(call.contract, 1)

    def scan(self, market: MarketState, book: Book) -> list[Candidate]:
        cfg, today, out = self.cfg, market.today, []
        for und in sorted(market.chains):
            chain, vol = market.chains[und], market.vol.get(und)
            if und not in self.eligible or vol is None:
                continue
            expiry = pick_expiry(chain, *cfg.dte, today)
            if expiry is None or entry_blockers(market, und, expiry, self.rules):
                continue
            if market.events[und].earnings_between(today, expiry):
                continue                       # ook als de globale earnings-blackout uit staat
            legs = self._legs(chain, expiry)
            debit = mark(chain, legs) if legs is not None else None
            iv = _iv(chain, legs) if legs is not None else None
            days = dte(expiry, today)
            vrp = vol.vrp_at(iv, days) if iv is not None and iv > 0 else None
            if legs is None or debit is None or debit <= 0 or vrp is None:
                continue
            ratio = vrp[0]
            if ratio > cfg.max_iv_rv_ratio:
                continue
            units = size(debit * MULTIPLIER, market, book, und, self.rules)
            if units == 0:
                continue
            out.append(Candidate(self.name, und, legs, units, -debit,
                                 debit * MULTIPLIER * units, 1.0 / ratio - 1.0,
                                 (f"iv_rv:{ratio:.3f}", f"dte:{days}",
                                  f"strike:{legs[0].contract.strike:g}")))
        return sorted(out, key=lambda c: (-c.rank_score, c.underlying))

    def manage(self, pos: StrategyPosition, market: MarketState) -> list[Action]:
        cfg, (expiry, *_) = self.cfg, pos.expiries()
        if expiry_close_due(expiry, market.now, market, self.rules):
            return [close(pos, Urgency.STOP, "expiry")]
        chain = market.chains.get(pos.underlying)
        if chain is not None:
            debit = -pos.entry_credit
            m = mark(chain, pos.intents())
            value = m + _hedge_pnl(pos, chain.spot) if m is not None else None
            if value is not None and value >= cfg.take_profit_multiple_debit * debit:
                return [close(pos, Urgency.NORMAL, "take_profit")]
            if value is not None and value <= cfg.stop_loss_frac_debit * debit:
                return [close(pos, Urgency.STOP, "stop_loss")]
            vol, iv = market.vol.get(pos.underlying), _iv(chain, pos.intents())
            if (vol is not None and vol.rv is not None and iv is not None
                    and iv >= vol.rv.at_horizon(dte(expiry, market.today))):
                return [close(pos, Urgency.NORMAL, "iv_normalized")]
        if dte(expiry, market.today) <= cfg.close_dte:
            return [close(pos, Urgency.NORMAL, "dte")]
        return []
