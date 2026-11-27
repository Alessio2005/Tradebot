"""De options-engine: één ronde per tijdstip, taken als ze aan de beurt zijn (detailontwerp 9).

De engine is een toestandsmachine (`EngineMode`) met taken die op een klok draaien. `tick(now)`
voert alle taken uit die op `now` aan de beurt zijn; een replay of test roept `tick` met een
virtuele klok aan, de paper-dienst met de echte klok. Er zit geen eigen tijd in: alleen `now`.

Taken (New York):

| Taak | Wanneer | Doet |
|---|---|---|
| market | elke ronde | ketens verversen als er een nieuwe snapshot is, paper-fills |
| manage | elke `manage_interval_seconds` in de sessie | `manage()` per positie, poort, orders |
| scan | op `scan_times` | `scan()` per strategie, beste kandidaat per strategie, poort, orders |
| walk | elke `walk_interval_seconds` per open order | price-walk (10.2) |
| unwind_shares | elke ronde in de sessie, na manage | aandelen van gesloten posities naar nul |
| hedge | elke `hedging.interval_seconds` (als `hedging.enabled`) | delta-hedge met aandelen |
| risk | elke `risk_interval_seconds` | dagverlies, drawdown (HALT), de-grossing |
| reconcile | bij start en elke `reconcile_interval_seconds` | broker tegen staat |
| end_of_day | `end_of_day` | ATM-IV, expiratie en assignment, equity, staat |

Aandelen (de delta-hedger, `options.hedging`): een aandelenorder staat met `"stock": True` in
`state.orders`, loopt niet mee in de price-walk en telt niet als "lopende order" van zijn
positie (sluiten blijft dus mogelijk). Elke fill wordt apart geboekt als de aandelen-leg van de
positie (één leg met `contract=None`, gemiddelde prijs); de geboekte executies staan in de
order (`execs`), zodat een herstart ze niet nog eens boekt. Een open hedge-order van een vorige
hedge-ronde wordt geannuleerd en pas de ronde daarna vervangen. Een gesloten positie met
aandelen krijgt in dezelfde ronde een aandelenorder naar nul (purpose EXIT); een lopende
hedge-order van die positie wordt eerst geannuleerd. Een EXIT die na `hedging.interval_seconds`
nog open staat, wordt geannuleerd en de ronde daarna tegen een verse limiet opnieuw gestuurd,
tot de positie vlak is. Aandelenfills worden ontdubbeld op `exec_id` (IBKR `execId`); een
correctie (zelfde id vóór de laatste punt) wordt nooit geboekt, wel gejournald en gealarmeerd. De reconciliatie vergelijkt per symbool
waarop gehedged is de aandelen bij de broker met de aandelen-legs: het verschil mag alleen uit
hele lots van 100 bestaan (assignment of uitoefening, zoals bij de wheel).

Fouten (9.3): een `SafetyBreach` zet de `HaltStore`; een onverwachte fout in een taak wordt
gelogd en gealarmeerd, en drie keer achter elkaar in dezelfde taak is HALT.
"""
from __future__ import annotations

import dataclasses
import logging
import math
import os
from collections.abc import Awaitable, Callable, Mapping
from datetime import date, datetime, time, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any

from ...risk.kill_switches import HaltRecord, HaltStore
from ...schemas.options import OptionsConfig
from ..calendar import NY, TradingCalendar, expiry_moment
from ..chain.builder import Chain, ChainBuilder
from ..chain.filters import build_filtered, filter_settings
from ..contracts import OptionContract, Right
from ..data.feed import MarketFeed
from ..execution.broker import BrokerAdapter, Execution, ExecutionKind
from ..execution.orders import (
    TERMINAL,
    ComboOrder,
    OrderStatus,
    Purpose,
    StockOrder,
    WalkStep,
    client_id,
    combo_fair,
    combo_prices,
    fair_walk_limit,
    position_id,
    tick_for,
    walk_limit,
)
from ..execution.paper import PaperBroker, settlement_kind
from ..hedging import hedge_orders, hedge_rules, stock_limit
from ..iv_metrics import IVHistory, IVMetrics, iv_rule_settings, realized_vol
from ..positions import (
    MULTIPLIER,
    Action,
    ActionKind,
    Book,
    Candidate,
    Leg,
    LegIntent,
    MarketState,
    PendingChange,
    PositionStatus,
    StrategyPosition,
    Trend,
    Urgency,
    mark,
    trend_of,
)
from ..pricing.engines import Method, Use
from ..pricing.settings import pricer_settings
from ..pricing.surface import Surface, SurfaceStatus, build_surface, skew_25d
from ..rates import Carry
from ..relative_value import rv_adjust, vrp_zscores
from ..risk.gamma import gamma_loss_1d, sigma_1d
from ..risk.gate import (
    GateState,
    OrderKind,
    OrderProposal,
    RiskGate,
    SafetyBreach,
    StockProposal,
    apply_changes,
)
from ..risk.limits import expiry_max_loss
from ..risk.monitor import assess, degross_plan
from ..selection import admit, selection_rules
from ..strategies.common import Strategy, parse_time
from ..strategies.registry import build_strategies
from ..vol_regime import (
    VolRegime,
    VolState,
    classify,
    effective_regimes,
    forecast_rv,
    regime_settings,
)
from .state import EngineState, Journal, intents_from, intents_record, save_state

__all__ = ["Engine", "EngineMode", "Paths"]

log = logging.getLogger(__name__)
MAX_TASK_FAILURES = 3
# Strategieën die de aandelen uit hun eigen assignment als losse aandelen houden (de wheel
# en haar losse delen: CSP → covered call), niet als aandelen-leg die wordt afgebouwd.
_OWN_SHARES = frozenset({"wheel", "covered_call", "cash_secured_put"})
LOT_STRATEGIES = _OWN_SHARES
Alert = Callable[[str, str, str], None]
Task = Callable[[datetime], Awaitable[None]]


class EngineMode(StrEnum):
    STARTING = "starting"
    RECONCILING = "reconciling"
    RUNNING = "running"
    BLOCKED = "blocked"
    HALTED = "halted"
    UNWINDING = "unwinding"
    STOPPING = "stopping"
    STOPPED = "stopped"


@dataclasses.dataclass(frozen=True, slots=True)
class Paths:
    root: Path

    @property
    def state(self) -> Path:
        return self.root / "state.json"

    @property
    def journal(self) -> Path:
        return self.root / "journal.jsonl"

    @property
    def halt_store(self) -> Path:
        return self.root / "halt" / "halt.json"

    @property
    def halt_file(self) -> Path:
        return self.root / "HALT"

    @property
    def unwind_file(self) -> Path:
        return self.root / "UNWIND"

    @property
    def heartbeat(self) -> Path:
        return self.root / "heartbeat"


def _write_heartbeat(path: Path, text: str) -> None:
    """Schrijf `text` als de volledige inhoud van `path` (de map moet bestaan).

    Overschrijven en daarna inkorten, in plaats van openen met afkappen (`write_text`): op
    sommige bestandssystemen kost afkappen bij het openen een paar honderd µs, en een replay
    schrijft de heartbeat elke ronde (tienduizenden keren). De inhoud is daarna dezelfde.
    """
    data = text.encode("utf-8")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT, 0o644)
    try:
        os.write(fd, data)
        os.ftruncate(fd, len(data))
    finally:
        os.close(fd)


def _no_alert(level: str, key: str, text: str) -> None:
    log.warning("[%s] %s: %s", level, key, text)


class Engine:
    def __init__(self, cfg: OptionsConfig, feed: MarketFeed, broker: BrokerAdapter,
                 paths: Paths, state: EngineState, alert: Alert = _no_alert,
                 strategies: tuple[Strategy, ...] | None = None) -> None:
        self.cfg, self.feed, self.broker, self.paths, self.state = cfg, feed, broker, paths, state
        self.alert = alert
        self.cal = TradingCalendar(cfg.universe.half_days)
        self.settings = pricer_settings(cfg.pricing)
        self.builder = ChainBuilder(self.settings, cfg.filters.max_quote_age_seconds, Use.SCAN)
        self.filters = filter_settings(cfg.filters)
        self.gate = RiskGate(cfg, self.settings.engines, Method(cfg.pricing.decision_engine))
        self.strategies = strategies if strategies is not None else build_strategies(cfg)
        self.by_name = {s.name: s for s in self.strategies}
        self.halts = HaltStore(paths.halt_store)
        self.journal = Journal(paths.journal)
        self.iv_rules = iv_rule_settings(cfg.iv_metrics)
        self.regime_rules = regime_settings(cfg.vol_regime, cfg.iv_metrics.atm_tenor_days)
        self.selection = selection_rules(cfg)
        self.hedge_rules = hedge_rules(cfg)
        sch = cfg.schedule
        self.scan_times = tuple(parse_time(t) for t in sch.scan_times)
        self.close_time = parse_time(sch.close_before_expiry_time)
        self.eod_time = parse_time(sch.end_of_day)
        self.mode = EngineMode.STARTING
        self.chains: dict[str, Chain] = {}
        self._snap_keys: dict[str, datetime] = {}
        self._iv: dict[str, IVMetrics] = {}
        self._surfaces: dict[str, Surface] = {}
        self._vol: dict[str, VolState] = {}
        self._trend: dict[str, Trend] = {}
        self._last: dict[str, datetime] = {}
        self._failures: dict[str, int] = {}
        self._done_today: set[tuple[date, str]] = set()
        self._shares: dict[str, int] = {}
        self._equity = cfg.account.capital_usd
        self._versions = (0, 0)
        self._book_version = 0
        self._market_version = 0
        self._last_risk: tuple[int, int] | None = None
        self._exec_since: datetime | None = None
        self._entries_blocked_day: date | None = None
        self._blocked: frozenset[str] = frozenset()
        self._seen: set[tuple[datetime, ExecutionKind, str, str, int, str]] = set()
        self._gamma_unknown: set[str] = set()      # posities waarvoor `gamma_unknown` al kwam

    # --- toestand -------------------------------------------------------------------------

    @property
    def halted(self) -> bool:
        return self.halts.is_halted() or self.paths.halt_file.exists()

    @property
    def unwinding(self) -> bool:
        return self.paths.unwind_file.exists()

    def _event(self, now: datetime, event: str, **kw: Any) -> None:
        self.journal.write(now, event, **kw)

    def _engage_halt(self, now: datetime, kind: str, reason: str, measured: float = 0.0,
                     threshold: float = 0.0, key: str = "") -> None:
        rec = self.halts.engage(HaltRecord(kind, reason, measured, threshold, key,
                                           now.isoformat()))
        self.mode = EngineMode.HALTED
        self._event(now, "halt", kind=rec.kind, reason=rec.reason)
        self.alert("critical", "options_halt", f"{rec.kind}: {rec.reason}")

    # --- de ronde -------------------------------------------------------------------------

    async def tick(self, now: datetime) -> None:
        if self.mode is EngineMode.STOPPED:
            return
        if self.mode is EngineMode.STARTING:
            await self._run("reconcile", now, self._reconcile)
        await self._run("market", now, self._market)
        if self.halted and self.mode not in (EngineMode.HALTED, EngineMode.UNWINDING):
            self.mode = EngineMode.HALTED
        if self.unwinding:
            self.mode = EngineMode.UNWINDING
        in_session = self.cal.is_open(now)
        if in_session:
            await self._every("walk", now, self.cfg.execution.walk_interval_seconds, self._walk)
            await self._every("risk", now, self.cfg.schedule.risk_interval_seconds, self._risk)
            await self._every("manage", now, self.cfg.schedule.manage_interval_seconds,
                              self._manage)
            await self._run("unwind_shares", now, self._unwind_shares)
            if self.cfg.hedging.enabled:
                await self._every("hedge", now, self.cfg.hedging.interval_seconds, self._hedge)
            await self._every("reconcile", now, self.cfg.schedule.reconcile_interval_seconds,
                              self._reconcile)
            for t in self.scan_times:
                await self._once("scan:" + t.isoformat(), now, t, self._scan)
        await self._once("eod", now, self.eod_time, self._end_of_day)
        self.paths.heartbeat.parent.mkdir(parents=True, exist_ok=True)
        _write_heartbeat(self.paths.heartbeat, now.isoformat())

    async def _every(self, name: str, now: datetime, seconds: float, fn: Task) -> None:
        last = self._last.get(name)
        if last is not None and (now - last).total_seconds() < seconds:
            return
        self._last[name] = now
        await self._run(name, now, fn)

    async def _once(self, name: str, now: datetime, at: time, fn: Task) -> None:
        key = (now.date(), name)
        if key in self._done_today or not self.cal.is_trading_day(now.date()):
            return
        if now < datetime.combine(now.date(), at, tzinfo=NY):
            return
        self._done_today.add(key)
        await self._run(name, now, fn)

    async def _run(self, name: str, now: datetime, fn: Task) -> None:
        name = name.split(":", maxsplit=1)[0]           # "scan:10:00" en "scan:13:30" tellen samen
        try:
            await fn(now)
            self._failures[name] = 0
        except SafetyBreach as e:
            self._engage_halt(now, "options_safety_breach", str(e))
        except Exception as e:
            log.exception("taak %s faalde", name)
            n = self._failures.get(name, 0) + 1
            self._failures[name] = n
            self._event(now, "task_error", task=name, error=repr(e), count=n)
            self.alert("error", f"options_task_{name}", repr(e))
            if n >= MAX_TASK_FAILURES:
                self._engage_halt(now, "options_task_failures",
                                  f"taak {name} faalde {n} keer achter elkaar", n,
                                  MAX_TASK_FAILURES)

    # --- markt ----------------------------------------------------------------------------

    def _carry(self, und: str) -> Carry:
        ev = self.feed.events(und)
        return Carry(self.feed.rate, dividends=ev.dividends if ev is not None else ())

    def _metrics(self, und: str, chain: Chain, today: date) -> IVMetrics:
        surf = build_surface(chain, self.cfg.pricing.surface_min_vega)
        self._surfaces[und] = surf
        atm = surf.atm_vol(self.cfg.iv_metrics.atm_tenor_days)
        rows = [(date.fromisoformat(d), v) for d, v in self.state.iv_history.get(und, [])]
        bars = self.feed.bars_before(und, today)
        rv = realized_vol(bars, self.cfg.iv_metrics.rv_window_days)
        expiries = chain.expiries()
        skew = skew_25d(chain, expiries[len(expiries) // 2]) if expiries else None
        slope = surf.term_slope() if surf.status is SurfaceStatus.OK else None
        return IVMetrics.compute(atm, IVHistory(rows), today, rv, self.iv_rules, slope, skew)

    def _vol_state(self, und: str, now: datetime) -> None:
        """Regime bijwerken; een wisseling komt in het journal."""
        m = self._iv[und]
        rv = forecast_rv(self.feed.bars_before(und, now.date()), self.regime_rules)
        st = classify(m.atm_iv, rv, m.term_slope, self.regime_rules)
        prev = self._vol.get(und)
        self._vol[und] = st
        if prev is None or prev.regime is not st.regime:
            self._event(now, "vol_regime", underlying=und, **st.as_event())

    async def _market(self, now: datetime) -> None:
        changed = False
        jobs = []
        for und in self.feed.underlyings():
            snap = self.feed.snapshot(und, now)
            if snap is None:
                self.chains.pop(und, None)
                continue
            if self._snap_keys.get(und) == snap.asof:
                continue
            jobs.append((und, snap))
        held = frozenset(leg.contract for pos in self.state.open_positions()
                         for leg in pos.option_legs if leg.contract is not None)
        built = [build_filtered(self.builder, self.filters, und, snap.spot, snap.options,
                                snap.asof, self._carry(und), held) for und, snap in jobs]
        for (und, snap), (chain, res) in zip(jobs, built, strict=True):
            self.chains[und] = chain
            self._snap_keys[und] = snap.asof
            self._iv[und] = self._metrics(und, chain, now.date())
            self._vol_state(und, now)
            t = trend_of(self.feed.bars_before(und, now.date()))
            if t is not None:
                self._trend[und] = t
            changed = True
            self._event(now, "chain", underlying=und, kept=len(chain.options),
                        rejected={k.value: v for k, v in sorted(res.rejected.items())})
        if changed:
            self._market_version += 1
        if isinstance(self.broker, PaperBroker):
            self.broker.update_market(now, self.chains)
        await self._executions(now)
        acct = await self.broker.account()
        self._equity = acct.equity
        await self._refresh_shares()

    async def _refresh_shares(self) -> None:
        self._shares = {p.symbol: p.qty for p in await self.broker.positions()
                        if p.contract is None}

    def market_state(self, now: datetime) -> MarketState:
        events = {u: e for u in self.chains if (e := self.feed.events(u)) is not None}
        peak = max(self.state.high_water_mark, self.cfg.account.capital_usd)
        return MarketState(now, self.cal, dict(self.chains), dict(self._iv), dict(self._trend),
                           events, self.cfg.account.capital_usd, dict(self._vol),
                           self._equity, peak)

    def book(self) -> Book:
        return Book(tuple(self.state.open_positions()), dict(self._shares),
                    dict(self.state.cost_basis), self._equity, dict(self.state.wheel),
                    self._hedge_shares(), self._pending_entries(),
                    {p.position_id: (p.underlying, p.shares) for p in self.state.positions
                     if p.status is PositionStatus.CLOSED and p.shares},
                    self._pending_changes())

    def _pending_changes(self) -> tuple[PendingChange, ...]:
        """Lopende (nog niet gevulde) ADJUST- en ROLL-orders, voor de poort (review N10)."""
        terminal = {s.value for s in TERMINAL}
        kinds = {OrderKind.ADJUST.value, OrderKind.ROLL.value}
        return tuple(
            PendingChange(o["position_id"], intents_from(o["legs"]), o["quantity"],
                          math.inf if o["max_loss"] is None else o["max_loss"])
            for o in self.state.orders.values()
            if not o.get("stock") and o["kind"] in kinds and o["status"] not in terminal)

    def _pending_entries(self) -> tuple[StrategyPosition, ...]:
        """Lopende (nog niet gevulde) entry-orders, als positie (voor de poort en de sizing)."""
        terminal = {s.value for s in TERMINAL}
        out = []
        for o in self.state.orders.values():
            if o.get("stock") or o["kind"] != OrderKind.OPEN.value or o["status"] in terminal:
                continue
            q = o["quantity"]
            out.append(StrategyPosition(
                o["position_id"], o["strategy"], o["underlying"],
                tuple(Leg(leg.contract, o["underlying"], leg.ratio * q, 0.0)
                      for leg in intents_from(o["legs"])),
                datetime.fromisoformat(o["submitted_at"]), q, -o["limit"] * MULTIPLIER * q,
                math.inf if o["max_loss"] is None else o["max_loss"], variant=o["variant"]))
        return tuple(out)

    def _hedge_shares(self) -> dict[str, int]:
        """Aandelen-legs per symbool, ook van gesloten posities die nog naar nul gaan."""
        out: dict[str, int] = {}
        for p in self.state.positions:
            n = p.shares
            if n:
                out[p.underlying] = out.get(p.underlying, 0) + n
        return out

    def gate_state(self, now: datetime, daily_loss: float = 0.0) -> GateState:
        today = now.date().isoformat()
        return GateState(self.halted, self.unwinding,
                         daily_loss if self._entries_blocked_day != now.date() else 1.0,
                         self._equity, self.state.new_positions.get(today, 0),
                         self.state.day_trade_dates(),
                         frozenset() if self.mode is not EngineMode.BLOCKED
                         else frozenset(self._blocked))

    # --- orders ---------------------------------------------------------------------------

    def _pending(self, pos_id: str) -> bool:
        """Een lopende combo-order voor de positie (aandelenorders tellen niet)."""
        return any(o["position_id"] == pos_id and o["status"] not in {s.value for s in TERMINAL}
                   and not o.get("stock") for o in self.state.orders.values())

    def _open_stock(self) -> dict[str, dict[str, Any]]:
        """Lopende aandelenorders per position-id."""
        return {o["position_id"]: o for o in self.state.orders.values()
                if o.get("stock") and o["status"] == OrderStatus.SUBMITTED.value}

    def _concession(self, purpose: Purpose, urgency: Urgency) -> float:
        e = self.cfg.execution
        if purpose is Purpose.ENTRY:
            return e.max_concession_entry
        return e.max_concession_exit if urgency is Urgency.STOP else e.max_concession_entry

    def _fair_anchored(self, purpose: Purpose) -> bool:
        return purpose is Purpose.ENTRY and self.cfg.execution.limit_anchor == "fair"

    def _limit(self, und: str, legs: tuple[LegIntent, ...], purpose: Purpose, urgency: Urgency,
               step: int) -> tuple[WalkStep, str | None] | None:
        """De limiet bij `step` (zie `execution.orders`) en, bij het fair-anker, de reden om een
        entry niet te sturen (`spread_vs_fair`, `max_edge`) of None; None zonder keten of quote."""
        chain = self.chains.get(und)
        prices = combo_prices(chain, legs) if chain is not None else None
        if chain is None or prices is None:
            return None
        mid, natural = prices
        e = self.cfg.execution
        if not self._fair_anchored(purpose):
            return walk_limit(mid, natural, step, tick_for(e.ticks, mid, legs),
                              self._concession(purpose, urgency)), None
        fair = combo_fair(chain, self._surfaces.get(und), legs, self.settings.engines)
        if fair is None:                         # zelfde quotes als combo_prices: kan niet
            return None
        s = fair_walk_limit(mid, natural, fair, step, tick_for(e.ticks, mid, legs),
                            e.max_edge_frac)
        if abs(natural - mid) * 2.0 > e.max_spread_frac_fair * abs(fair):
            return s, "spread_vs_fair"
        return s, "max_edge" if s.done else None

    async def _send(self, p: OrderProposal, now: datetime, purpose: Purpose) -> None:
        state = self.gate_state(now, self._daily_loss())
        d = self.gate.review(p, self.book(), self.market_state(now), state)
        self._event(now, "risk_decision", position_id=p.position_id, strategy=p.strategy,
                    underlying=p.underlying, kind=p.kind.value, verdict=d.verdict.value,
                    reason=d.reason,
                    checks=[[c.name, round(c.measured, 4), round(c.threshold, 4)]
                            for c in d.checks])
        if d.approved is None:
            return
        ap = d.approved.proposal
        first = self._limit(ap.underlying, ap.legs, purpose, ap.urgency, 0)
        if first is None:
            self._event(now, "order_skipped", position_id=ap.position_id, purpose=purpose.value,
                        strategy=ap.strategy, underlying=ap.underlying, reason="no_quote")
            return
        step, skip = first
        if skip is not None:
            self._event(now, "entry_skipped", position_id=ap.position_id, strategy=ap.strategy,
                        underlying=ap.underlying, reason=skip)
            return
        key = f"{ap.position_id}|{purpose.value}|{now.date().isoformat()}"
        attempt = self.state.attempts.get(key, 0)
        self.state.attempts[key] = attempt + 1
        cid = client_id(ap.position_id, purpose, now.date(), attempt)
        order = ComboOrder(cid, ap.position_id, ap.legs, ap.quantity, step.limit, purpose,
                           ap.urgency)
        self.state.orders[cid] = {
            "client_id": cid, "position_id": ap.position_id, "purpose": purpose.value,
            "kind": ap.kind.value, "strategy": ap.strategy, "underlying": ap.underlying,
            "legs": intents_record(ap.legs),
            "quantity": ap.quantity, "max_loss": None if math.isinf(ap.max_loss)
            else ap.max_loss, "variant": ap.variant, "urgency": ap.urgency.value,
            "reason": ap.reason, "step": 0, "limit": step.limit,
            "status": OrderStatus.SUBMITTED.value, "submitted_at": now.isoformat()}
        self._event(now, "order_submitted", client_id=cid, position_id=ap.position_id,
                    purpose=purpose.value, limit=step.limit, quantity=ap.quantity,
                    strategy=ap.strategy, underlying=ap.underlying)
        bo = await self.broker.submit(d.approved, order)
        self.state.orders[cid]["status"] = bo.status.value
        await self._executions(now)

    async def _walk(self, now: datetime) -> None:
        # In HALT of UNWIND geen nieuw risico: een lopende entry wordt geannuleerd in plaats
        # van doorgeschoven richting natural (review 2026-11-27, punt 2).
        stop_entries = self.halted or self.unwinding
        for cid, o in sorted(self.state.orders.items()):
            if o["status"] != OrderStatus.SUBMITTED.value or o.get("stock"):
                continue
            purpose = Purpose(o["purpose"])
            if stop_entries and purpose is Purpose.ENTRY:
                await self.broker.cancel(cid)
                o["status"] = OrderStatus.CANCELLED.value
                self._event(now, "order_cancelled", client_id=cid, position_id=o["position_id"],
                            reason="halt")
                continue
            nxt = self._limit(o["underlying"], intents_from(o["legs"]), purpose,
                              Urgency(o["urgency"]), o["step"] + 1)
            if nxt is None:
                continue
            step, _ = nxt
            if step.done:
                await self.broker.cancel(cid)
                o["status"] = OrderStatus.CANCELLED.value
                self._event(now, "order_cancelled", client_id=cid, position_id=o["position_id"],
                            reason="max_edge" if self._fair_anchored(purpose)
                            else "max_concession")
                continue
            o["step"], o["limit"] = o["step"] + 1, step.limit
            bo = await self.broker.modify(cid, step.limit)
            o["status"] = bo.status.value
            self._event(now, "order_modified", client_id=cid, limit=step.limit, step=o["step"])
        await self._executions(now)

    async def _executions(self, now: datetime) -> None:
        since = self._exec_since or datetime.min.replace(tzinfo=NY)
        execs = await self.broker.executions(since)
        self._exec_since = now          # inclusief: een fill later in deze ronde telt nog mee
        fills: dict[str, list[Execution]] = {}
        for e in execs:
            # Aandelenfills ook op `exec_id`: twee deelfills in dezelfde seconde zijn twee fills.
            key = (e.ts, e.kind, e.client_id, e.contract.key() if e.contract else e.symbol,
                   e.qty, "" if e.contract else e.exec_id)
            if key in self._seen:
                continue
            self._seen.add(key)
            if e.kind is ExecutionKind.FILL:
                fills.setdefault(e.client_id, []).append(e)
            else:
                self._assignment(e, now)
        for cid in sorted(fills):
            o = self.state.orders.get(cid)
            if o is not None and o.get("stock"):
                self._book_stock_fill(cid, o, fills[cid], now)
                continue
            if o is None or o.get("booked"):
                continue
            o["booked"] = True
            o["status"] = OrderStatus.FILLED.value
            self._book_fill(cid, o, fills[cid], now)

    def _book_fill(self, cid: str, o: dict[str, Any], execs: list[Execution], now: datetime) -> None:
        q = o["quantity"]
        limit = o["limit"]
        legs = intents_from(o["legs"])
        fee = sum(e.commission for e in execs)
        px = {e.contract.key(): e.price for e in execs}
        kind = OrderKind(o["kind"])
        self._book_version += 1
        self._event(now, "fill", client_id=cid, position_id=o["position_id"], price=limit,
                    quantity=q, commission=fee)
        if kind is OrderKind.OPEN:
            pos = StrategyPosition(
                o["position_id"], o["strategy"], o["underlying"],
                tuple(Leg(leg.contract, o["underlying"], leg.ratio * q,
                          px.get(leg.contract.key(), 0.0)) for leg in legs),
                now, q, -limit * MULTIPLIER * q,
                math.inf if o["max_loss"] is None else o["max_loss"], variant=o["variant"],
                phase=o["variant"] if o["strategy"] == "wheel" else "")
            self.state.positions.append(pos)
            day = now.date().isoformat()
            self.state.new_positions[day] = self.state.new_positions.get(day, 0) + 1
            if pos.strategy == "wheel":
                self.state.wheel[pos.underlying] = "csp" if pos.phase == "csp" else "cc"
            self._event(now, "position_opened", position_id=pos.position_id,
                        strategy=pos.strategy, underlying=pos.underlying,
                        entry_net=pos.entry_net, max_loss=o["max_loss"])
            return
        i = next(i for i, p in enumerate(self.state.positions)
                 if p.position_id == o["position_id"])
        pos = self.state.positions[i]
        qty = {leg.contract.key(): leg for leg in pos.option_legs}  # type: ignore[union-attr]
        for leg in legs:
            k = leg.contract.key()
            old = qty.get(k, Leg(leg.contract, pos.underlying, 0, px.get(k, 0.0)))
            qty[k] = dataclasses.replace(old, qty=old.qty + leg.ratio * q)
        rest = tuple(v for _, v in sorted(qty.items()) if v.qty)
        stock = tuple(leg for leg in pos.legs if leg.contract is None)
        if pos.opened_at.date() == now.date():
            self.state.day_trades.append(now.date().isoformat())
        if kind is OrderKind.CLOSE or not rest:
            self.state.positions[i] = dataclasses.replace(pos, status=PositionStatus.CLOSED)
            if pos.strategy == "wheel" and pos.phase == "csp":
                self.state.wheel[pos.underlying] = "idle"
            elif pos.strategy == "wheel":
                self.state.wheel[pos.underlying] = "shares"
            hedge = ({"hedge_realized_usd": round(pos.hedge_realized_usd, 2)}
                     if pos.hedge_realized_usd or pos.shares else {})
            self._event(now, "position_closed", position_id=pos.position_id,
                        reason=o["reason"], exit_net=-limit * MULTIPLIER * q, **hedge)
            return
        self.state.positions[i] = dataclasses.replace(
            pos, legs=rest + stock, max_loss=o["max_loss"] if o["max_loss"] is not None else math.inf,
            adjustments=pos.adjustments + (kind is OrderKind.ADJUST),
            entry_net=pos.entry_net - limit * MULTIPLIER * q)
        self._event(now, "position_changed", position_id=pos.position_id, kind=kind.value)

    def _warn_settlement(self, now: datetime, code: str, detail: str) -> None:
        self.alert("error", code, detail)
        self._event(now, "settlement_alert", code=code, detail=detail)

    def _assignment(self, e: Execution, now: datetime) -> None:
        """Reken `e` af op de open posities met dit contract (de broker doet het netto per
        contract).

        De broker rekent zijn hele netto positie in het contract af (`−e.qty`). Is dat precies de
        som van de legs van de open posities op dit contract, dan rekent elke positie haar eigen
        leg af, naar haar eigen teken: long uitgeoefend, short assigned (of allebei vervallen).
        Zo rekenen ook tegengestelde posities op één contract af, ook als ze netto nul zijn (die
        uitvoering komt van `_settle_netted`). Klopt de som niet, dan per teken: alleen posities
        met het teken van de afrekening, in positievolgorde, elk haar hele leg of niets; blijft
        er iets over, dan volgt een alert.
        """
        self._book_version += 1
        k = e.contract.key()
        net = -e.qty
        holders = []
        for i, p in enumerate(self.state.positions):
            leg = next((x for x in p.option_legs if x.contract.key() == k), None)  # type: ignore[union-attr]
            if p.status is not PositionStatus.CLOSED and leg is not None:
                holders.append((i, leg.qty))
        chosen, remaining = list(holders), 0
        if sum(q for _, q in holders) != net:
            chosen, remaining = [], abs(net)
            for i, q in holders:
                if net and (q > 0) == (net > 0) and remaining:
                    take = min(abs(q), remaining)      # gedeeltelijk: alleen dat deel (N7)
                    chosen.append((i, take if q > 0 else -take))
                    remaining -= take
        for i, qty in chosen:
            p = self.state.positions[i]
            kind = (e.kind if e.kind is ExecutionKind.EXPIRY
                    else ExecutionKind.EXERCISE if qty > 0 else ExecutionKind.ASSIGNMENT)
            left = tuple(dataclasses.replace(x, qty=x.qty - qty) if x.contract.key() == k else x  # type: ignore[union-attr]
                         for x in p.option_legs)
            rest = tuple(x for x in left if x.qty)
            stock = tuple(leg for leg in p.legs if leg.contract is None)
            settle_usd, spot = 0.0, e.contract.strike
            if kind is not ExecutionKind.EXPIRY:         # waarde van de levering tegen de slotkoers
                chain = self.chains.get(p.underlying)
                if chain is None:
                    self._warn_settlement(now, "options_settlement_no_chain",
                                          f"{k} van {p.position_id}: geen keten, settle_usd = 0")
                spot = chain.spot if chain is not None else e.contract.strike
                itm = spot - e.contract.strike if e.contract.right is Right.CALL \
                    else e.contract.strike - spot
                settle_usd = qty * itm * MULTIPLIER
            own = p.strategy in _OWN_SHARES
            if p.strategy == "wheel" and kind is ExecutionKind.ASSIGNMENT:
                self.state.wheel[p.underlying] = "shares" if p.phase == "csp" else "idle"
                self.state.cost_basis[p.underlying] = (
                    e.contract.strike - p.entry_credit if p.phase == "csp"
                    else self.state.cost_basis.get(p.underlying, 0.0))
            elif (p.strategy == "cash_secured_put" and kind is ExecutionKind.ASSIGNMENT
                  and p.underlying not in self.state.cost_basis):
                self.state.cost_basis[p.underlying] = e.contract.strike - p.entry_credit
            cur = (dataclasses.replace(p, legs=rest + stock,
                                       units=_units_after(rest, p.units)) if rest
                   else dataclasses.replace(p, status=PositionStatus.CLOSED))
            if kind is not ExecutionKind.EXPIRY and not own:
                # De geleverde aandelen worden een aandelen-leg van de positie, geboekt op de
                # slotkoers waartegen `settle_usd` al rekent. Is de positie dicht, dan bouwt
                # `_unwind_shares` ze de volgende ronde af; anders sluit `_manage` de positie
                # (reden `assigned`) en gaan ze mee. Zo blijft er geen losse aandelenpositie in
                # het boek staan (gezien in de DoltHub-replay: een vroege assignment voor
                # ex-dividend liet −100 SPY staan). De wheel, covered call en cash-secured put
                # houden hun aandelen juist als losse aandelen (ontwerp: CSP → covered call).
                delivered = qty * MULTIPLIER * (1 if e.contract.right is Right.CALL else -1)
                cur = _with_shares(cur, delivered, spot)
                cur = dataclasses.replace(cur, assigned=True)
            self.state.positions[i] = cur
            # Een assignment, of een uitoefening van een gehedgede strategie: alarm, want de
            # aandelen zijn er onverwacht (ze worden wel afgebouwd, zie hierboven).
            hedged = self.cfg.hedging.enabled and p.strategy in self.cfg.hedging.strategies
            if kind is not ExecutionKind.EXPIRY and not own and (qty < 0 or hedged):
                self.alert("critical", "options_assignment",
                           f"{'short' if qty < 0 else 'long'} {k} van {p.position_id}"
                           f" {kind.value}")
            self._event(now, kind.value, contract=k, qty=-qty, price=e.price,
                        position_id=p.position_id, settle_usd=round(settle_usd, 2))
            if not rest:
                hedge = ({"hedge_realized_usd": round(cur.hedge_realized_usd, 2)}
                         if cur.hedge_realized_usd or cur.shares or p.shares else {})
                self._event(now, "position_closed", position_id=p.position_id,
                            reason=kind.value, exit_net=round(settle_usd, 2), **hedge)
        if not chosen:
            self._event(now, e.kind.value, contract=k, qty=e.qty, price=e.price,
                        position_id=None, settle_usd=0.0)
        if remaining:
            self._warn_settlement(now, "options_settlement_remainder",
                                  f"{k}: {remaining} contract(en) van {e.kind.value} zonder positie")

    def _settle_netted(self, day: date, closes: Mapping[str, float], now: datetime) -> None:
        """Legs die op `day` expireren op een contract waarop de open posities netto nul zijn.

        De paper-broker houdt per contract alleen het netto aantal; bij nul heeft hij niets meer
        om af te rekenen en komt er geen uitvoering. Elke positie rekent dan haar eigen leg af
        tegen de slotkoers (`settlement_kind`), via `_assignment` met netto 0."""
        net: dict[str, tuple[OptionContract, int]] = {}
        for p in self.state.open_positions():
            for leg in p.option_legs:
                c = leg.contract
                if c.expiry == day:  # type: ignore[union-attr]
                    k = c.key()  # type: ignore[union-attr]
                    net[k] = (c, net.get(k, (c, 0))[1] + leg.qty)  # type: ignore[assignment]
        for _, (c, n) in sorted(net.items()):
            spot = closes.get(c.underlying)
            if n or spot is None:      # niet netto nul: de reconciliatie ziet het verschil
                continue
            kind = settlement_kind(c, spot, 1)
            price = 0.0 if kind is ExecutionKind.EXPIRY else c.strike
            self._assignment(Execution(now, kind, "", c, c.underlying, 0, price), now)

    # --- aandelen (delta-hedger) ----------------------------------------------------------

    def _book_stock_fill(self, cid: str, o: dict[str, Any], execs: list[Execution],
                         now: datetime) -> None:
        """Elke aandelenfill apart op de aandelen-leg van de positie (ook na annuleren)."""
        i = next(i for i, p in enumerate(self.state.positions)
                 if p.position_id == o["position_id"])
        for e in execs:
            rec: Any = e.exec_id or [e.ts.isoformat(), e.qty, e.price]
            if rec in o["execs"]:
                continue
            base = e.exec_id.rsplit(".", 1)[0] if "." in e.exec_id else ""
            if base and any(isinstance(x, str) and "." in x and x.rsplit(".", 1)[0] == base
                            for x in o["execs"]):
                o["execs"].append(rec)                  # een correctie: nooit twee keer boeken
                self._event(now, "hedge_fill_correction", client_id=cid, exec_id=e.exec_id,
                            position_id=o["position_id"], qty=e.qty, price=e.price)
                self.alert("error", "options_hedge_fill_correction",
                           f"{cid} {e.exec_id}: correctie niet geboekt")
                continue
            o["execs"].append(rec)
            o["filled"] += e.qty
            pos = self.state.positions[i] = _with_shares(self.state.positions[i], e.qty, e.price,
                                                         e.commission)
            self._book_version += 1
            self._event(now, "hedge_fill", client_id=cid, position_id=o["position_id"],
                        underlying=o["underlying"], purpose=o["purpose"], qty=e.qty,
                        price=e.price, commission=e.commission, shares=pos.shares,
                        hedge_realized_usd=round(pos.hedge_realized_usd, 2))
        if o["filled"] == o["qty"]:
            o["status"] = OrderStatus.FILLED.value

    async def _send_stock(self, pos: StrategyPosition, qty: int, now: datetime,
                          purpose: Purpose, delta: float | None = None) -> None:
        chain = self.chains[pos.underlying]
        limit = stock_limit(chain.spot, qty, self.cfg.paper.stock_spread_bps)
        p = StockProposal(pos.position_id, pos.strategy, pos.underlying, qty, limit,
                          closing=purpose is Purpose.EXIT)
        d = self.gate.review_stock(p, self.book(), self.market_state(now),
                                   self.gate_state(now, self._daily_loss()))
        cid = None
        if d.approved is not None:
            key = f"{pos.position_id}|{purpose.value}|{now.date().isoformat()}"
            attempt = self.state.attempts.get(key, 0)
            self.state.attempts[key] = attempt + 1
            cid = client_id(pos.position_id, purpose, now.date(), attempt)
            self.state.orders[cid] = {
                "client_id": cid, "stock": True, "position_id": pos.position_id,
                "purpose": purpose.value, "strategy": pos.strategy,
                "underlying": pos.underlying, "qty": qty, "limit": limit, "filled": 0,
                "execs": [], "status": OrderStatus.SUBMITTED.value,
                "submitted_at": now.isoformat()}
        self._event(now, "hedge_order", client_id=cid, position_id=pos.position_id,
                    strategy=pos.strategy, underlying=pos.underlying, purpose=purpose.value,
                    qty=qty, limit=limit, verdict=d.verdict.value, reason=d.reason,
                    delta_shares=None if delta is None else round(delta, 2),
                    checks=[[c.name, round(c.measured, 4), round(c.threshold, 4)]
                            for c in d.checks])
        if d.approved is None or cid is None:
            return
        bo = await self.broker.submit_stock(
            d.approved, StockOrder(cid, pos.position_id, pos.underlying, qty, limit, purpose))
        if self.state.orders[cid]["status"] == OrderStatus.SUBMITTED.value:
            self.state.orders[cid]["status"] = bo.status.value
        await self._executions(now)
        await self._refresh_shares()

    async def _cancel_stock(self, o: dict[str, Any], now: datetime, reason: str,
                            confirm: bool = False) -> bool:
        """Annuleer een aandelenorder. Met `confirm` alleen als de broker hem niet meer open
        heeft (anders kan hij nog vullen naast een nieuwe order); dan geeft dit False terug."""
        await self.broker.cancel(o["client_id"])
        if confirm and any(b.order.client_id == o["client_id"]
                           for b in await self.broker.open_orders()):
            self._event(now, "order_cancel_pending", client_id=o["client_id"],
                        position_id=o["position_id"], reason=reason)
            return False
        o["status"] = OrderStatus.CANCELLED.value
        self._event(now, "order_cancelled", client_id=o["client_id"],
                    position_id=o["position_id"], reason=reason)
        return True

    async def _unwind_shares(self, now: datetime) -> None:
        """Gesloten posities met aandelen: lopende hedge annuleren, dan een order naar nul."""
        closed = [p for p in self.state.positions
                  if p.status is PositionStatus.CLOSED and p.shares]
        if not closed and not self.cfg.hedging.enabled:
            return
        open_stock = self._open_stock()
        retry = self.cfg.hedging.interval_seconds
        for o in open_stock.values():
            pos = next(p for p in self.state.positions if p.position_id == o["position_id"])
            if pos.status is PositionStatus.CLOSED and o["purpose"] == Purpose.HEDGE.value:
                await self._cancel_stock(o, now, "position_closed")
            elif o["purpose"] == Purpose.EXIT.value and (
                    now - datetime.fromisoformat(o["submitted_at"])).total_seconds() >= retry:
                # De ronde erna opnieuw, en alleen na bevestigde annulering (review N30).
                await self._cancel_stock(o, now, "replaced", confirm=True)
        self._net_closed_shares(now)
        closed = [p for p in self.state.positions
                  if p.status is PositionStatus.CLOSED and p.shares]
        for pos in closed:
            if pos.position_id in open_stock or pos.underlying not in self.chains:
                continue
            await self._send_stock(pos, -pos.shares, now, Purpose.EXIT)

    def _net_closed_shares(self, now: datetime) -> None:
        """Tegengestelde aandelen-legs van gesloten posities op één symbool verrekenen tegen
        de spot, zonder order: de broker houdt toch al het netto aantal. Zo gaat alleen het
        netto aantal naar de markt en komt er geen koop én verkoop van hetzelfde aantal (dubbele
        kosten). Elke leg realiseert haar P&L tegen de spot, zonder commissie (N29)."""
        busy = {o["position_id"] for o in self._open_stock().values()}
        by_und: dict[str, list[int]] = {}
        for i, p in enumerate(self.state.positions):
            if (p.status is PositionStatus.CLOSED and p.shares and p.position_id not in busy
                    and p.underlying in self.chains):
                by_und.setdefault(p.underlying, []).append(i)
        for und, idx in sorted(by_und.items()):
            spot = self.chains[und].spot
            longs = [i for i in idx if self.state.positions[i].shares > 0]
            shorts = [i for i in idx if self.state.positions[i].shares < 0]
            for a in longs:
                for b in shorts:
                    n = min(self.state.positions[a].shares, -self.state.positions[b].shares)
                    if n <= 0:
                        continue
                    for j, q in ((a, -n), (b, n)):
                        pos = self.state.positions[j] = _with_shares(self.state.positions[j], q,
                                                                     spot)
                        self._event(now, "hedge_netted", position_id=pos.position_id,
                                    underlying=und, qty=q, price=spot, shares=pos.shares,
                                    hedge_realized_usd=round(pos.hedge_realized_usd, 2))
                    self._book_version += 1

    async def _hedge(self, now: datetime) -> None:
        """Eén hedge-ronde: oude hedge-orders annuleren, dan per onderliggende de band."""
        open_stock = self._open_stock()
        waiting = set()
        for o in open_stock.values():
            if o["purpose"] == Purpose.HEDGE.value:
                await self._cancel_stock(o, now, "replaced")
            waiting.add(o["underlying"])          # pas de volgende ronde opnieuw
        blocked = self._blocked if self.mode is EngineMode.BLOCKED else frozenset()
        live = tuple(p for p in self.state.open_positions()
                     if p.underlying not in waiting and p.underlying not in blocked)
        # In HALT of UNWIND alleen afbouwen: een order die de aandelen niet verkleint, komt niet
        # eens bij de poort (die blijft de laatste controle), dus ook geen afgewezen regel.
        reduce_only = self.halted or self.unwinding
        held = self._hedge_shares()
        for h in hedge_orders(live, self.chains, self.cfg.account.capital_usd,
                              self.hedge_rules):
            n = held.get(h.underlying, 0)
            if reduce_only and abs(n + h.qty) >= abs(n):
                continue
            pos = next(p for p in live if p.position_id == h.position_id)
            await self._send_stock(pos, h.qty, now, Purpose.HEDGE, h.delta_shares)

    # --- taken ----------------------------------------------------------------------------

    def _daily_loss(self) -> float:
        prev = self.state.equity[-1][1] if self.state.equity else self.cfg.account.capital_usd
        return max(prev - self._equity, 0.0) / prev if prev else 0.0

    def _entry_window(self, now: datetime) -> bool:
        start, end = self.cal.session(now.date())
        sch = self.cfg.schedule
        lo = start + timedelta(minutes=sch.no_entry_first_minutes)
        hi = end - timedelta(minutes=sch.no_entry_last_minutes)
        if now.date() in self.cal.half_days:
            hi = min(hi, datetime.combine(now.date(), time(12, 30), tzinfo=NY))
        return bool(lo <= now < hi)

    async def _scan(self, now: datetime) -> None:
        if self.mode in (EngineMode.HALTED, EngineMode.UNWINDING) or self.halted:
            self._event(now, "scan_summary", skipped="halted")
            return
        if not self._entry_window(now):
            self._event(now, "scan_summary", skipped="entry_window")
            return
        market, book = self.market_state(now), self.book()
        regimes = effective_regimes(self._vol, self.cfg.vol_regime.market_underlying)
        rv_z = vrp_zscores(self._vol) if self.cfg.relative_value.weight > 0 else None
        opened_today: set[str] = set()
        summary: dict[str, int] = {}
        blocked: dict[str, int] = {}
        for s in self.strategies:
            cands = self._admitted(s.scan(market, book), regimes, blocked, market.today, rv_z)
            summary[s.name] = len(cands)
            best = next((c for c in cands if c.underlying not in opened_today
                         and not self._opened_today(c.underlying, now.date())), None)
            if best is None:
                continue
            self._event(now, "candidate", strategy=best.strategy, underlying=best.underlying,
                        quantity=best.quantity, net_credit=round(best.net_credit, 4),
                        max_loss=best.max_loss, rank=round(best.rank_score, 4),
                        trace=list(best.rule_trace))
            pid = position_id(best.strategy, best.underlying, best.legs, now.date())
            if any(p.position_id == pid for p in self.state.positions) or self._pending(pid):
                continue
            await self._send(self._entry(best, pid), now, Purpose.ENTRY)
            opened_today.add(best.underlying)
        self._event(now, "scan_summary", candidates=summary, blocked=blocked,
                    regimes={u: r.value for u, r in sorted(regimes.items())})

    def _admitted(self, cands: list[Candidate], regimes: dict[str, VolRegime],
                  blocked: dict[str, int], today: date,
                  rv_z: Mapping[str, float] | None = None) -> list[Candidate]:
        """Kandidaten die het regime en de theta-harvest-poort halen, in dezelfde volgorde.

        Met `rv_z` (relative value aan) wordt de rangscore herwogen en opnieuw gesorteerd."""
        out = []
        for c in cands:
            a = admit(c, self.chains[c.underlying], self._vol.get(c.underlying),
                      regimes.get(c.underlying, VolRegime.UNKNOWN), self.selection, today)
            if a.admitted:
                out.append(a.candidate)
            else:
                key = f"{c.strategy}:{a.blocker}"
                blocked[key] = blocked.get(key, 0) + 1
        if rv_z is not None:
            w = self.cfg.relative_value.weight
            out = sorted((rv_adjust(c, rv_z.get(c.underlying, 0.0), w) for c in out),
                         key=lambda c: (-c.rank_score, c.underlying))
        return out

    def _entry(self, c: Candidate, pid: str) -> OrderProposal:
        chain = self.chains[c.underlying]
        m = mark(chain, c.legs)
        assert m is not None
        return OrderProposal.open(dataclasses.replace(c, net_credit=-m), pid)

    def _opened_today(self, und: str, day: date) -> bool:
        return any(p.underlying == und and p.opened_at.date() == day
                   for p in self.state.positions)

    async def _manage(self, now: datetime) -> None:
        """Beheer per open positie; loopt ook in HALT (sluiten verkleint risico).

        Vóór de regels van de strategie: een undefined-risk positie (`max_loss` oneindig) op een
        onderliggende met effectief regime STRESS gaat dicht (STOP, `regime_stress`); een positie
        met credit waarvan het gamma-verlies over één dag te groot is, ook (STOP, `gamma_risk`,
        alleen met `risk.max_gamma_loss_frac_credit`). Dezelfde
        bron als de toelating in `_scan` (`effective_regimes` over `_vol`, dat `_market` altijd
        bijwerkt, ook met `vol_regime.enabled: false`), dus wat hier sluit, opent daar niet."""
        market = self.market_state(now)
        regimes = effective_regimes(self._vol, self.cfg.vol_regime.market_underlying)
        for pos in list(self.state.open_positions()):
            if self._pending(pos.position_id):
                continue
            if self.mode is EngineMode.UNWINDING:
                acts = [Action(ActionKind.CLOSE, pos.position_id, pos.intents(), Urgency.STOP,
                               "unwind")]
            elif math.isinf(pos.max_loss) and regimes.get(pos.underlying) is VolRegime.STRESS:
                acts = [Action(ActionKind.CLOSE, pos.position_id, pos.intents(), Urgency.STOP,
                               "regime_stress")]
            elif pos.assigned:
                # Na een assignment of uitoefening is de positie niet meer wat de strategie
                # opende (aandelen-leg tegen de rest): meteen sluiten, voor elke strategie.
                acts = [Action(ActionKind.CLOSE, pos.position_id, pos.intents(), Urgency.STOP,
                               "assigned")]
            elif self._gamma_breach(pos, now):
                acts = [Action(ActionKind.CLOSE, pos.position_id, pos.intents(), Urgency.STOP,
                               "gamma_risk")]
            else:
                strat = self.by_name.get(pos.strategy)
                acts = strat.manage(pos, market) if strat is not None else []
            for a in acts[:1]:
                await self._act(a, pos, now)

    def _gamma_breach(self, pos: StrategyPosition, now: datetime) -> bool:
        """Gamma-verlies over één dag boven `risk.max_gamma_loss_frac_credit` × de netto credit
        (`entry_net`; na een ADJUST de lopende netto credit, niet de oorspronkelijke ontvangst)
        (D2); uit bij `None`, zonder credit of zonder RV-voorspelling. Is het verlies onbekend
        (geen keten of een leg zonder Greeks), dan sluit de regel niet, maar meldt hij dat één
        keer per positie (`gamma_unknown`): de stops van de strategie blijven gelden."""
        limit = self.cfg.risk.max_gamma_loss_frac_credit
        st = self._vol.get(pos.underlying)
        if limit is None or pos.entry_net <= 0.0 or st is None or st.rv is None:
            return False
        loss = gamma_loss_1d(pos, self.chains.get(pos.underlying), sigma_1d(st.rv))
        if loss is None:
            if pos.position_id not in self._gamma_unknown:
                self._gamma_unknown.add(pos.position_id)
                self._event(now, "gamma_unknown", position_id=pos.position_id,
                            strategy=pos.strategy, underlying=pos.underlying)
                self.alert("warning", "options_gamma_unknown",
                           f"{pos.position_id}: gamma-verlies onbekend (geen keten of Greeks)")
            return False
        return loss > limit * pos.entry_net

    async def _act(self, a: Action, pos: StrategyPosition, now: datetime) -> None:
        chain = self.chains.get(pos.underlying)
        if chain is None:
            self._event(now, "action_deferred", position_id=pos.position_id, reason="no_chain")
            return
        closing = tuple(LegIntent(leg.contract, -leg.ratio) for leg in a.legs) + a.new_legs
        m = mark(chain, closing)
        if m is None:
            self._event(now, "action_deferred", position_id=pos.position_id, reason="no_quote")
            return
        max_after = pos.max_loss
        if a.kind is ActionKind.ROLL:
            new = a.new_legs[0].contract
            max_after = new.strike * MULTIPLIER * pos.units
        elif a.kind is ActionKind.ADJUST:
            # Het maximale verlies na de bijsturing uit de nieuwe legs (oud 5), niet het oude.
            (after,) = apply_changes((pos,), (PendingChange(pos.position_id, closing, pos.units,
                                                            pos.max_loss),))
            max_after = expiry_max_loss(after.option_legs, pos.entry_net - m * MULTIPLIER * pos.units,
                                        default=pos.max_loss)
        p = OrderProposal.for_action(a, pos, -m, max_after)
        purpose = {ActionKind.CLOSE: Purpose.EXIT, ActionKind.CLOSE_LEG: Purpose.EXIT,
                   ActionKind.ADJUST: Purpose.ADJUST, ActionKind.ROLL: Purpose.ROLL}[a.kind]
        if self.mode is EngineMode.UNWINDING:
            purpose = Purpose.UNWIND
        self._event(now, "action", position_id=pos.position_id, kind=a.kind.value,
                    reason=a.reason, urgency=a.urgency.value)
        await self._send(p, now, purpose)

    async def _risk(self, now: datetime) -> None:
        version = (self._market_version, self._book_version)
        if version == self._last_risk:
            return
        self._last_risk = version
        prev = self.state.equity[-1][1] if self.state.equity else self.cfg.account.capital_usd
        hwm = max(self.state.high_water_mark, self.cfg.account.capital_usd)
        st = assess(self._equity, prev, hwm, self.cfg.risk, now)
        if st.halt is not None and not self.halts.is_halted():
            self._engage_halt(now, st.halt.kind, st.halt.reason, st.halt.measured,
                              st.halt.threshold, st.halt.config_key)
        if st.entries_blocked and self._entries_blocked_day != now.date():
            self._entries_blocked_day = now.date()
            self._event(now, "daily_loss_stop", loss_frac=round(st.daily_loss_frac, 4))
        if st.degross:
            g = self.gate
            plan = degross_plan(self.book(), self.market_state(now), g.beta, g.scenarios,
                                self.cfg.risk, Method.BAW, self.settings.engines)
            for pid in plan:
                pos = next(p for p in self.state.open_positions() if p.position_id == pid)
                if not self._pending(pid):
                    await self._act(Action(ActionKind.CLOSE, pid, pos.intents(), Urgency.STOP,
                                           "degross"), pos, now)

    async def _reconcile(self, now: datetime) -> None:
        held = await self.broker.positions()
        broker = {p.contract.key(): p.qty for p in held if p.contract is not None}
        mine: dict[str, int] = {}
        for p in self.state.open_positions():
            for leg in p.option_legs:
                k = leg.contract.key()  # type: ignore[union-attr]
                mine[k] = mine.get(k, 0) + leg.qty
        diff = sorted(k for k in set(broker) | set(mine) if broker.get(k, 0) != mine.get(k, 0))
        diff += self._share_breaks({p.symbol: p.qty for p in held if p.contract is None})
        if diff:
            self._blocked = frozenset(k.split(" ")[0] for k in diff)
            if self.mode is not EngineMode.BLOCKED:
                self.alert("critical", "options_recon_break", ", ".join(diff))
            if self.mode not in (EngineMode.HALTED, EngineMode.UNWINDING):
                self.mode = EngineMode.BLOCKED
            self._event(now, "recon_break", contracts=diff)
            return
        self._blocked = frozenset()
        if self.mode in (EngineMode.STARTING, EngineMode.RECONCILING, EngineMode.BLOCKED):
            self.mode = EngineMode.HALTED if self.halted else EngineMode.RUNNING
            self._event(now, "recon_ok", mode=self.mode.value)

    def _share_breaks(self, broker: Mapping[str, int]) -> list[str]:
        """Symbolen waarop gehedged is en waar de broker iets anders heeft dan de aandelen-legs.

        Alleen op symbolen waar losse aandelen verwacht worden (wheel, cash-secured put, covered
        call: assignment en uitoefening geven hele lots van 100) telt een veelvoud van 100 als
        goed. Elders is elk verschil een break (review 2026-11-27, N3): een crash-herstart kan
        anders precies een lot te veel of te weinig geven zonder alarm."""
        hedge = self._hedge_shares()
        names = set(hedge) | {o["underlying"] for o in self.state.orders.values()
                              if o.get("stock")}
        lots = set(self.state.wheel) | {p.underlying for p in self.state.positions
                                        if p.strategy in LOT_STRATEGIES}
        def off(n: str) -> bool:
            d = broker.get(n, 0) - hedge.get(n, 0)
            return d % MULTIPLIER != 0 if n in lots else d != 0

        return [f"{n} shares" for n in sorted(names) if off(n)]

    def _book_dividends(self, paid: list[tuple[str, int, float]],
                        divs: Mapping[str, Any], now: datetime) -> None:
        """Het dividend dat de paper-broker op aandelen boekte, ook op de aandelen-legs van de
        posities (`hedge_realized_usd`), zodat de rapport-P&L en de broker-equity gelijk blijven.
        Losse aandelen (wheel, covered call) hebben geen positie-leg: alleen de equity (N6)."""
        for und, shares, usd in paid:
            self._event(now, "dividend_paid", underlying=und, shares=shares,
                        amount=divs[und].amount, usd=round(usd, 2))
            for i, p in enumerate(self.state.positions):
                if p.underlying != und or not p.shares:
                    continue
                gain = p.shares * divs[und].amount
                pos = self.state.positions[i] = dataclasses.replace(
                    p, hedge_realized_usd=p.hedge_realized_usd + gain)
                self._event(now, "dividend", position_id=p.position_id, underlying=und,
                            shares=p.shares, amount=divs[und].amount, usd=round(gain, 2),
                            hedge_realized_usd=round(pos.hedge_realized_usd, 2))

    async def _end_of_day(self, now: datetime) -> None:
        today = now.date()
        for und, m in sorted(self._iv.items()):
            fresh = self._snap_keys.get(und)
            if m.atm_iv is not None and fresh is not None and fresh.date() == today:
                rows = self.state.iv_history.setdefault(und, [])
                if not rows or rows[-1][0] != today.isoformat():
                    rows.append((today.isoformat(), m.atm_iv))
        if isinstance(self.broker, PaperBroker):
            closes = {u: c.spot for u, c in self.chains.items()}
            self.broker.settle(today, closes, now)
            nxt = self.cal.next_trading_day(today)
            divs = {}
            for und in self.feed.underlyings():
                ev = self.feed.events(und)
                for d in (ev.dividends if ev is not None else ()):
                    if d.ex_date == nxt:
                        divs[und] = d
            self.broker.early_assign(divs, now)
            await self._executions(now)
            self._book_dividends(self.broker.pay_dividends(divs, now), divs, now)
            self._settle_netted(today, closes, now)
        for p in self.state.open_positions():
            for leg in p.option_legs:
                if leg.qty < 0 and now >= expiry_moment(leg.contract.expiry):  # type: ignore[union-attr]
                    self.alert("critical", "options_short_past_expiry", leg.contract.key())  # type: ignore[union-attr]
        acct = await self.broker.account()
        self._equity = acct.equity
        self.state.equity.append((today.isoformat(), acct.equity))
        self.state.high_water_mark = max(self.state.high_water_mark, acct.equity)
        for cid, o in self.state.orders.items():
            if o["status"] == OrderStatus.SUBMITTED.value:
                await self.broker.cancel(cid)
                o["status"] = OrderStatus.EXPIRED.value
        save_state(self.state, self.paths.state)
        self._event(now, "state_saved", equity=round(acct.equity, 2),
                    open_positions=len(self.state.open_positions()))

    async def stop(self, now: datetime) -> None:
        self.mode = EngineMode.STOPPING
        for cid, o in self.state.orders.items():
            if o["status"] == OrderStatus.SUBMITTED.value:
                await self.broker.cancel(cid)
                o["status"] = OrderStatus.CANCELLED.value
        save_state(self.state, self.paths.state)
        self.builder.close()
        self._event(now, "stopped")
        self.mode = EngineMode.STOPPED


def _units_after(rest: tuple[Leg, ...], units: int) -> int:
    """Het aantal eenheden na een gedeeltelijke afrekening: blijven de legs veelvouden van
    `units`, dan ongewijzigd, anders de grootste gemene deler van de legs (verhoudingen per
    eenheid blijven dan gehele getallen, zodat een sluitorder klopt)."""
    if all(x.qty % units == 0 for x in rest):
        return units
    return math.gcd(*(abs(x.qty) for x in rest))


def _with_shares(pos: StrategyPosition, qty: int, price: float,
                 commission: float = 0.0) -> StrategyPosition:
    """`pos` met `qty` aandelen erbij in haar ene aandelen-leg, volgens gemiddelde kostprijs.

    Bijkopen in dezelfde richting middelt de prijs; afbouwen houdt de oude prijs en realiseert
    `gesloten × (prijs − gemiddelde)` (short: omgekeerd teken); omdraaien realiseert de hele
    oude leg en de rest krijgt de fillprijs. De commissie van elke fill gaat van de gerealiseerde
    P&L af (`hedge_realized_usd`)."""
    old = pos.shares
    new = old + qty
    prev = next((leg.entry_price for leg in pos.legs if leg.contract is None), price)
    closed = 0 if old == 0 or (old > 0) == (qty > 0) else min(abs(qty), abs(old))
    realized = closed * (price - prev) * (1 if old > 0 else -1) - commission
    if old == 0 or (old > 0) != (new > 0):
        avg = price
    elif abs(new) > abs(old):
        avg = (prev * old + price * qty) / new
    else:
        avg = prev
    stock = (Leg(None, pos.underlying, new, avg),) if new else ()
    return dataclasses.replace(pos, legs=pos.option_legs + stock,
                               hedge_realized_usd=pos.hedge_realized_usd + realized)
