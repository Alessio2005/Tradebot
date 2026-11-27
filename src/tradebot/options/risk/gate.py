"""De risicopoort: de enige plek waar een order goedgekeurd wordt (detailontwerp 8.1, 8.2).

`RiskGate.review` geeft een `Decision`. Alleen deze module kan een `ApprovedOrder` maken: de
constructor eist een module-private sentinel. `BrokerAdapter.submit` accepteert alleen een
`ApprovedOrder`, dus elke order is langs de poort geweest.

Volgorde:

1. **Hard** (`SafetyBreach`): aantal ≤ 0, een verlopen contract, premie boven `max_order_usd`,
   of een limietprijs buiten [intrinsiek − 1 tick, natuurlijke prijs + 1 tick]. Dat zijn fouten
   in code of config; de engine gaat dan in HALT.
2. **Toestand**: in HALT geen order die risico toevoegt; in UNWIND alleen sluitorders. Geen
   entry na de dagverliesgrens, op een onderliggende met een reconciliatieverschil, met
   undefined risk zonder `allow_undefined_risk`, of als de PDT-teller het verbiedt.
3. **Aantallen**: open posities en nieuwe posities vandaag.
4. **Pro forma**: de maten van het boek na de order, tegen elke grens.
   - Een entry met undefined risk (`max_loss` oneindig) krijgt er één check bij,
     `undefined_stress_loss`: de som van het stressverlies (`short_vol.STRESS_MOVE`, in USD, bij
     de huidige spot) van alle open undefined-risk posities, de lopende undefined-risk entries
     (`Book.pending_undefined`, nog niet gevuld) en de nieuwe, tegen
     `max_undefined_stress_loss_frac` × kapitaal. Zonder keten voor een open undefined-risk
     positie is die som onbekend en dus te groot.
   - Een entry wordt per eenheid verkleind tot hij past (RESIZE), of geweigerd.
   - Een volledige sluiting wordt altijd goedgekeurd: hij haalt een positie met beperkt
     risico weg, ook in HALT. Een stop gaat vóór de PDT-regel.
   - Bijsturen, rollen en een deel sluiten: goed als alles daarna past, of als de order
     risicoverkleinend is (elke nu overschreden maat daalt en geen andere gaat erover).
     In HALT alleen als hij risicoverkleinend is of alleen legs sluit. Nooit verkleind.

Een aandelenorder van de delta-hedger (`review_stock`) heeft zijn eigen, kortere pad:

1. **Hard** (`SafetyBreach`): aantal 0, een symbool buiten `universe.share_eligible`, of een
   limiet die verder dan de volledige spread (`paper.stock_spread_bps`) plus één cent van de spot
   ligt.
2. Een order die de aandelen van een gesloten positie naar nul brengt (`closing`), wordt altijd
   goedgekeurd: zonder opties eromheen is het alleen nog delta. De poort controleert dat hij
   echt afbouwt: op het symbool van de gesloten positie, tegengesteld aan haar eigen
   aandelen-leg (`Book.closing_shares`) en niet groter dan die; anders is het een
   `SafetyBreach`.
3. Anders: geen keten of geen volledige Greeks is weigeren; op een onderliggende met een
   reconciliatieverschil ook. In HALT of UNWIND alleen als de order de aandelen-legs op dat
   symbool in absolute waarde verkleint én de absolute beta-delta van het boek daalt.
   Plafond `stock_notional_usd`: |aandelen-legs na de order| × spot ≤ `hedging.max_notional_frac`
   × kapitaal (hetzelfde plafond als de hedger), tenzij de order de aandelen-legs verkleint.
   Normaal: goed als de beta-delta daarna binnen de grens ligt of daalt.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import date, datetime
from enum import StrEnum

from ...schemas.options import OptionsConfig
from ..calendar import expiry_moment
from ..chain.builder import Chain
from ..positions import (
    MULTIPLIER,
    Action,
    ActionKind,
    Book,
    Candidate,
    Leg,
    LegIntent,
    MarketState,
    StrategyPosition,
    Urgency,
)
from ..pricing.engines import EngineSettings, Method
from ..strategies.short_vol import position_stress_loss_usd
from .aggregate import aggregate
from .limits import BookMetrics, CheckResult, check, measure
from .pdt import pdt_blocks_entry
from .scenarios import grid

__all__ = ["ApprovedOrder", "ApprovedStockOrder", "Decision", "GateState", "OrderKind",
           "OrderProposal", "RiskGate", "SafetyBreach", "StockDecision", "StockProposal",
           "Verdict"]

_SENTINEL = object()


class SafetyBreach(RuntimeError):
    """Een order die nooit gemaakt had mogen worden: de engine gaat in HALT."""


class Verdict(StrEnum):
    APPROVE = "approve"
    RESIZE = "resize"
    REJECT = "reject"


class OrderKind(StrEnum):
    OPEN = "open"
    CLOSE = "close"
    CLOSE_LEG = "close_leg"
    ADJUST = "adjust"
    ROLL = "roll"


@dataclass(frozen=True, slots=True)
class OrderProposal:
    """Een combo-order in handelsrichting. `legs[i].ratio`: + kopen, − verkopen, per eenheid.

    `limit_net` is per eenheid en per aandeel: + ontvangen credit, − betaald debit.
    `max_loss`: bij OPEN het maximale verlies van de nieuwe positie, anders dat van de
    positie na de order (USD totaal).
    """

    kind: OrderKind
    strategy: str
    underlying: str
    legs: tuple[LegIntent, ...]
    quantity: int
    limit_net: float
    max_loss: float
    position_id: str
    urgency: Urgency = Urgency.NORMAL
    reason: str = ""
    variant: str = ""

    @classmethod
    def open(cls, c: Candidate, position_id: str) -> OrderProposal:
        return cls(OrderKind.OPEN, c.strategy, c.underlying, c.legs, c.quantity, c.net_credit,
                   c.max_loss, position_id, Urgency.NORMAL, "entry", c.variant)

    @classmethod
    def for_action(cls, a: Action, pos: StrategyPosition, limit_net: float,
                   max_loss_after: float = 0.0) -> OrderProposal:
        """De order bij een actie uit `manage`: de te sluiten legs omgedraaid, plus de nieuwe."""
        closing = tuple(LegIntent(leg.contract, -leg.ratio) for leg in a.legs)
        kind = {ActionKind.CLOSE: OrderKind.CLOSE, ActionKind.CLOSE_LEG: OrderKind.CLOSE_LEG,
                ActionKind.ADJUST: OrderKind.ADJUST, ActionKind.ROLL: OrderKind.ROLL}[a.kind]
        return cls(kind, pos.strategy, pos.underlying, closing + a.new_legs, pos.units,
                   limit_net, 0.0 if kind is OrderKind.CLOSE else max_loss_after,
                   pos.position_id, a.urgency, a.reason, pos.variant)

    @property
    def notional_usd(self) -> float:
        return abs(self.limit_net) * MULTIPLIER * self.quantity


@dataclass(frozen=True, slots=True)
class ApprovedOrder:
    proposal: OrderProposal
    approved_at: datetime
    checks: tuple[CheckResult, ...]
    _token: object = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._token is not _SENTINEL:
            raise TypeError("een ApprovedOrder maakt alleen de risicopoort")


@dataclass(frozen=True, slots=True)
class StockProposal:
    """Een aandelenorder voor de aandelen-leg van `position_id`: `qty` + kopen, − verkopen.

    `closing`: de positie is gesloten en haar aandelen gaan terug naar nul.
    """

    position_id: str
    strategy: str
    symbol: str
    qty: int
    limit: float              # per aandeel
    closing: bool = False


@dataclass(frozen=True, slots=True)
class ApprovedStockOrder:
    proposal: StockProposal
    approved_at: datetime
    checks: tuple[CheckResult, ...]
    _token: object = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._token is not _SENTINEL:
            raise TypeError("een ApprovedStockOrder maakt alleen de risicopoort")


@dataclass(frozen=True, slots=True)
class StockDecision:
    verdict: Verdict
    approved: ApprovedStockOrder | None
    checks: tuple[CheckResult, ...]
    reason: str


@dataclass(frozen=True, slots=True)
class Decision:
    verdict: Verdict
    approved: ApprovedOrder | None
    checks: tuple[CheckResult, ...]
    reason: str


@dataclass(frozen=True, slots=True)
class GateState:
    """Wat de engine over het boek weet, naast posities en markt."""

    halted: bool = False
    unwind: bool = False
    daily_loss_frac: float = 0.0
    equity: float = 0.0
    new_positions_today: int = 0
    day_trades: tuple[date, ...] = ()
    blocked_underlyings: frozenset[str] = frozenset()


def _combo_bounds(p: OrderProposal, chain: Chain) -> tuple[float, float] | None:
    """(gunstigst, natuurlijk) als debit per eenheid; None als een quote ontbreekt.

    Natuurlijk: kopen tegen de ask, verkopen tegen de bid (de slechtste prijs die nog vult).
    Gunstigst: andersom (kopen tegen de bid, verkopen tegen de ask). Een limiet buiten dit
    bereik (plus één tick) is geen prijs maar een fout.
    """
    best = natural = 0.0
    for leg in p.legs:
        c = leg.contract
        o = chain.find(c.expiry, c.strike, c.right)
        if o is None or o.quote.bid is None or o.quote.ask is None:
            return None
        natural += leg.ratio * (o.quote.ask if leg.ratio > 0 else o.quote.bid)
        best += leg.ratio * (o.quote.bid if leg.ratio > 0 else o.quote.ask)
    return best, natural


class RiskGate:
    def __init__(self, cfg: OptionsConfig, engines: EngineSettings,
                 method: Method = Method.BINOMIAL_LR) -> None:
        self.cfg, self.engines, self.method = cfg, engines, method
        self.risk = cfg.risk
        self.scenarios = grid(cfg.risk.scenario_spot, cfg.risk.scenario_vol_points)
        self.beta: Mapping[str, float] = cfg.universe.beta_to_spy

    # --- hulpfuncties ---------------------------------------------------------------------

    def _hard(self, p: OrderProposal, market: MarketState) -> str | None:
        """Gooit `SafetyBreach` bij een fout; geeft een reden als quotes ontbreken."""
        if p.quantity <= 0:
            raise SafetyBreach(f"aantal {p.quantity} voor {p.position_id}")
        for leg in p.legs:
            c = leg.contract
            if c.underlying != p.underlying:
                raise SafetyBreach(f"leg {c.key()} hoort niet bij {p.underlying}")
            if market.now >= expiry_moment(c.expiry):
                raise SafetyBreach(f"verlopen contract {c.key()}")
        if p.notional_usd > self.cfg.execution.max_order_usd:
            raise SafetyBreach(f"premie {p.notional_usd:.2f} USD boven max_order_usd")
        chain = market.chains.get(p.underlying)
        bounds = _combo_bounds(p, chain) if chain is not None else None
        if bounds is None:
            return "quotes:missing"
        tick = self.cfg.execution.ticks.combo
        debit = -p.limit_net
        best, natural = bounds
        if debit < best - tick - 1e-9 or debit > natural + tick + 1e-9:
            raise SafetyBreach(f"limiet {debit:.4f} buiten [{best:.4f}, {natural:.4f}]"
                               f" voor {p.position_id}")
        return None

    def _measure(self, positions: tuple[StrategyPosition, ...], book: Book,
                 market: MarketState) -> BookMetrics:
        return measure(positions, book.shares, market, self.beta, self.scenarios, self.method,
                       self.engines)

    @staticmethod
    def _stress_usd(positions: tuple[StrategyPosition, ...], market: MarketState) -> float:
        """Het stressverlies van de undefined-risk posities onder `positions` (USD)."""
        total = 0.0
        for x in positions:
            if not math.isinf(x.max_loss):
                continue
            chain = market.chains.get(x.underlying)
            if chain is None:
                return math.inf
            total += position_stress_loss_usd(x, chain.spot)
        return total

    def _approve(self, p: OrderProposal, verdict: Verdict, checks: tuple[CheckResult, ...],
                 market: MarketState, reason: str) -> Decision:
        return Decision(verdict, ApprovedOrder(p, market.now, checks, _SENTINEL), checks, reason)

    @staticmethod
    def _reject(checks: tuple[CheckResult, ...], reason: str) -> Decision:
        return Decision(Verdict.REJECT, None, checks, reason)

    # --- de ingang ------------------------------------------------------------------------

    def review(self, p: OrderProposal, book: Book, market: MarketState,
               state: GateState) -> Decision:
        missing = self._hard(p, market)
        if p.kind is OrderKind.CLOSE:
            if missing is not None:
                return self._reject((), missing)
            return self._approve(p, Verdict.APPROVE, (), market, "close")
        if state.unwind:
            return self._reject((), "state:unwind")
        if missing is not None:
            return self._reject((), missing)
        if p.kind is OrderKind.OPEN:
            return self._review_open(p, book, market, state)
        return self._review_change(p, book, market, state)

    def _review_open(self, p: OrderProposal, book: Book, market: MarketState,
                     state: GateState) -> Decision:
        r, cap = self.risk, market.capital_usd
        live = tuple(x for x in book.positions if x.status.value != "closed")
        gates = [
            (state.halted, "state:halt"),
            (state.daily_loss_frac >= r.daily_loss_stop_entries_frac, "daily_loss:entries"),
            (p.underlying in state.blocked_underlyings, "reconcile:diff"),
            (math.isinf(p.max_loss) and not r.allow_undefined_risk, "undefined_risk"),
            (len(live) >= r.max_open_positions, "count:open_positions"),
            (state.new_positions_today >= r.max_new_positions_per_day, "count:new_today"),
            (r.pdt_guard and pdt_blocks_entry(state.day_trades, market.now, state.equity,
                                              market.calendar), "pdt"),
        ]
        for blocked, why in gates:
            if blocked:
                return self._reject((), why)
        unit = StrategyPosition(p.position_id, p.strategy, p.underlying,
                                tuple(Leg(leg.contract, p.underlying, leg.ratio, 0.0)
                                      for leg in p.legs), market.now, 1,
                                p.limit_net * MULTIPLIER, p.max_loss / p.quantity,
                                variant=p.variant)
        base = self._measure(live, book, market)
        per_unit = self._measure((unit,), Book(), market)
        undefined = math.isinf(p.max_loss)
        pending = tuple(x for x in book.pending_undefined if x.position_id != p.position_id)
        stress_base = self._stress_usd(live + pending, market) if undefined else 0.0
        stress_unit = self._stress_usd((unit,), market) if undefined else 0.0
        checks: tuple[CheckResult, ...] = ()
        for q in range(p.quantity, 0, -1):
            checks = check(base.plus(per_unit, q), r, cap)
            if undefined:
                checks += (CheckResult("undefined_stress_loss", stress_base + q * stress_unit,
                                       r.max_undefined_stress_loss_frac * cap),)
            if all(c.ok for c in checks):
                if q == p.quantity:
                    return self._approve(p, Verdict.APPROVE, checks, market, "entry")
                return self._approve(replace(p, quantity=q, max_loss=p.max_loss / p.quantity * q),
                                     Verdict.RESIZE, checks, market, f"resized:{q}")
        failed = ",".join(c.name for c in checks if not c.ok)
        return self._reject(checks, f"limits:{failed}")

    def _after(self, p: OrderProposal, book: Book) -> tuple[StrategyPosition, ...]:
        """Het boek na een bijsturing, rol of gedeeltelijke sluiting."""
        out = []
        for x in book.positions:
            if x.status.value == "closed":
                continue
            if x.position_id != p.position_id:
                out.append(x)
                continue
            qty: dict[str, int] = {}
            legs: dict[str, Leg] = {}
            for leg in x.option_legs:
                qty[leg.contract.key()] = leg.qty  # type: ignore[union-attr]
                legs[leg.contract.key()] = leg  # type: ignore[union-attr]
            for leg in p.legs:
                k = leg.contract.key()
                qty[k] = qty.get(k, 0) + leg.ratio * p.quantity
                legs.setdefault(k, Leg(leg.contract, p.underlying, 0, 0.0))
            new_legs = tuple(replace(legs[k], qty=n) for k, n in sorted(qty.items()) if n)
            out.append(replace(x, legs=new_legs, max_loss=p.max_loss))
        return tuple(out)

    def _review_change(self, p: OrderProposal, book: Book, market: MarketState,
                       state: GateState) -> Decision:
        r, cap = self.risk, market.capital_usd
        live = tuple(x for x in book.positions if x.status.value != "closed")
        before = check(self._measure(live, book, market), r, cap)
        after = check(self._measure(self._after(p, book), book, market), r, cap)
        reducing = all(
            (a.measured < b.measured) if not b.ok else a.ok
            for b, a in zip(before, after, strict=True))
        if reducing and any(not b.ok for b in before):
            return self._approve(p, Verdict.APPROVE, after, market, "risk_reducing")
        if state.halted and p.kind is not OrderKind.CLOSE_LEG:
            return self._reject(after, "state:halt")
        if all(a.ok for a in after):
            return self._approve(p, Verdict.APPROVE, after, market, p.kind.value)
        failed = ",".join(c.name for c in after if not c.ok)
        return self._reject(after, f"limits:{failed}")

    # --- aandelen (delta-hedger) ----------------------------------------------------------

    def review_stock(self, p: StockProposal, book: Book, market: MarketState,
                     state: GateState) -> StockDecision:
        if p.qty == 0:
            raise SafetyBreach(f"aandelenorder van 0 voor {p.position_id}")
        if p.symbol not in self.cfg.universe.share_eligible:
            raise SafetyBreach(f"{p.symbol} mag niet als aandeel gehouden worden")
        chain = market.chains.get(p.symbol)
        if chain is not None:
            band = chain.spot * self.cfg.paper.stock_spread_bps / 1e4 + 0.01
            if not math.isfinite(p.limit) or abs(p.limit - chain.spot) > band + 1e-9:
                raise SafetyBreach(f"aandelenlimiet {p.limit:.4f} te ver van spot"
                                   f" {chain.spot:.4f} voor {p.position_id}")
        held = book.hedge_shares.get(p.symbol, 0)
        if p.closing:
            # Alleen afbouwen, tegen de eigen aandelen-leg van de gesloten positie (niet het
            # totaal van het symbool: een open positie kan het andere teken hebben).
            und, own = book.closing_shares.get(p.position_id, (p.symbol, 0))
            if und != p.symbol or own == 0 or (own > 0) == (p.qty > 0) or abs(p.qty) > abs(own):
                raise SafetyBreach(f"sluitorder {p.qty} {p.symbol} voor {p.position_id} past"
                                   f" niet bij zijn aandelen-leg ({own} {und})")
            return self._stock(p, Verdict.APPROVE, (), market, "close")
        if chain is None:
            return StockDecision(Verdict.REJECT, None, (), "quotes:missing")
        if p.symbol in state.blocked_underlyings:
            return StockDecision(Verdict.REJECT, None, (), "reconcile:diff")
        live = tuple(x for x in book.positions if x.status.value != "closed")
        before = aggregate(live, book.shares, market.chains, self.beta)
        shares = dict(book.shares)
        shares[p.symbol] = shares.get(p.symbol, 0) + p.qty
        after = aggregate(live, shares, market.chains, self.beta)
        shrinks = abs(held + p.qty) < abs(held)
        checks = (CheckResult("beta_delta_usd", abs(after.beta_delta_usd),
                              self.risk.max_abs_beta_delta_frac * market.capital_usd),
                  CheckResult("stock_notional_usd", abs(held + p.qty) * chain.spot,
                              self.cfg.hedging.max_notional_frac * market.capital_usd))
        if not (before.complete and after.complete):
            return StockDecision(Verdict.REJECT, None, checks, "greeks_incomplete")
        reducing = abs(after.beta_delta_usd) < abs(before.beta_delta_usd)
        # In HALT of UNWIND alleen afbouwen: minder aandelen én minder delta.
        if (state.halted or state.unwind) and not (reducing and shrinks):
            return StockDecision(Verdict.REJECT, None, checks,
                                 "state:halt" if state.halted else "state:unwind")
        if not (checks[1].ok or shrinks):
            return StockDecision(Verdict.REJECT, None, checks, "limits:stock_notional_usd")
        if reducing:
            return self._stock(p, Verdict.APPROVE, checks, market, "risk_reducing")
        if checks[0].ok:
            return self._stock(p, Verdict.APPROVE, checks, market, "hedge")
        return StockDecision(Verdict.REJECT, None, checks, "limits:beta_delta_usd")

    @staticmethod
    def _stock(p: StockProposal, verdict: Verdict, checks: tuple[CheckResult, ...],
               market: MarketState, reason: str) -> StockDecision:
        return StockDecision(verdict, ApprovedStockOrder(p, market.now, checks, _SENTINEL),
                             checks, reason)
