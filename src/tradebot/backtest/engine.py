# src/tradebot/backtest/engine.py
"""De authoritative event-driven backtester — L10.

Phase 5, deliverable 1. De ENIGE wettige autoriteit binnen het platform
(audit §16.1). De keten is strikt en zonder omweg:

    MarketEvent -> SignalEvent -> SOVEREIGN RISK DECISION -> portfolio intent
                -> order -> fill -> accounting

WAAROM ELKE STAP HIER STRUCTUREEL IS EN NIET AFGESPROKEN
--------------------------------------------------------
De vier legacy-engines hadden allemaal ergens een pad `signal -> lokale limiet
-> order`. Dat kon, omdat de order gewoon een getal was dat je kon uitrekenen.
Hier kan het niet:

* `OrderRouter.build_orders()` NEEMT een `RiskDecision` als verplicht argument
  en verwerpt er een met een andere `config_hash`. Er is geen tweede weg naar
  een `Order`.
* `Order.__post_init__` verwerpt een order die kan vullen op of vóór zijn eigen
  beslisbar. Lookahead is dus geen testresultaat maar een constructiefout.
* `Ledger.mark()` controleert na ELKE bar de balans en de PnL-attributie. Een
  kostenpost die nergens vandaan komt, crasht de run.

DE `shift(1)`-CONVENTIE, NU EXPLICIET
-------------------------------------
`backtest/baseline_runner.py` schrijft `held = weights.shift(1)`: een gewicht
bepaald op `t` rendeert op `t+1`. Dat is dezelfde causaliteit die deze engine
afdwingt, maar hier is zij een LATENCY: `venue.latency_bars = 1` betekent dat
een order besloten op de close van bar `t` vult op bar `t+1`. Het verschil is
dat de vectorized versie de conventie in een index-shift stopte, en deze in een
timestamp die de order zelf draagt en die bij overtreding crasht.

WAT DEZE ENGINE NIET DOET
-------------------------
Hij bouwt geen alpha. `desired_exposure` komt binnen als `a_t in [-1, +1]` van
een `AlphaUnit`, precies zoals het L4-contract voorschrijft. De engine kent geen
modelnaam, geen confidence en geen strategie-identiteit; zie
`execution/context.py` voor waarom dat geen omissie is.

Ref: ARCHITECTUUR_AUDIT_2026-08-22.md secties 15, 16, 16.1, 19 (L10), 26.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Any

import numpy as np
import pandas as pd

from ..execution.context import MarketSlice, build_context
from ..execution.order_router import ExecutionReport, OrderRouter
from ..risk.contract import RiskDecision
from ..risk.engine import RiskEngine
from ..utils.failfast import ConfigContractError, DataContractError, require
from .accounting import Ledger, LedgerSnapshot

__all__ = ["BacktestResult", "EventDrivenEngine"]


@dataclass(frozen=True, slots=True)
class BacktestResult:
    """De uitkomst van één run, met volledig auditspoor.

    `risk_policy_hash` staat hier omdat fase-opdracht §6 punt 5 het eist:
    *"backtest result bevat de gebruikte risk-policy identifier."* Zonder dat
    veld is een resultaat niet te koppelen aan het regime dat het produceerde.
    """

    equity_curve: pd.Series
    returns: pd.Series
    snapshots: tuple[LedgerSnapshot, ...]
    reports: tuple[ExecutionReport, ...]
    decisions: tuple[RiskDecision, ...]
    risk_policy_hash: str
    impact_status: str
    spread_status: str
    n_bars: int
    n_orders: int
    n_fills: int
    n_rejected: int
    n_partial: int
    n_sovereign_clipped: int
    n_sovereign_halted: int
    audit: dict[str, Any] = field(default_factory=dict)

    @property
    def total_fees(self) -> float:
        return float(self.snapshots[-1].fees_paid) if self.snapshots else 0.0

    @property
    def total_funding(self) -> float:
        return float(self.snapshots[-1].funding_paid) if self.snapshots else 0.0

    @property
    def total_spread_cost(self) -> float:
        return float(sum(r.spread_cost for r in self.reports))

    @property
    def total_impact_cost(self) -> float:
        return float(sum(r.impact_cost for r in self.reports))

    @property
    def fill_ratio(self) -> float:
        """Aandeel van de gevraagde notional dat daadwerkelijk is gevuld."""
        requested = sum(abs(r.requested_qty) * r.arrival_price for r in self.reports)
        filled = sum(abs(r.filled_qty) * r.arrival_price for r in self.reports)
        return float(filled / requested) if requested > 0.0 else 1.0

    def as_record(self) -> dict[str, Any]:
        return {
            "risk_policy_hash": self.risk_policy_hash,
            "impact_status": self.impact_status,
            "spread_status": self.spread_status,
            "n_bars": self.n_bars,
            "n_orders": self.n_orders,
            "n_fills": self.n_fills,
            "n_rejected": self.n_rejected,
            "n_partial": self.n_partial,
            "n_sovereign_clipped": self.n_sovereign_clipped,
            "n_sovereign_halted": self.n_sovereign_halted,
            "fill_ratio": self.fill_ratio,
            "total_fees": self.total_fees,
            "total_funding": self.total_funding,
            "total_spread_cost": self.total_spread_cost,
            "total_impact_cost": self.total_impact_cost,
            "final_equity": float(self.equity_curve.iloc[-1])
            if len(self.equity_curve) else None,
            **self.audit,
        }


class EventDrivenEngine:
    """De authoritative engine. Eén per run, puur in zijn invoer."""

    __slots__ = ("_context_kind", "_ledger", "_participation_cap", "_risk",
                 "_router")

    def __init__(
        self,
        *,
        risk_engine: RiskEngine,
        router: OrderRouter,
        initial_equity: float,
        context_kind: str = "backtest",
    ) -> None:
        require(
            isinstance(risk_engine, RiskEngine),
            "De engine vereist een soevereine RiskEngine. Er is geen modus "
            "waarin hij zonder draait (fase-opdracht §17: 'backtest zonder "
            "sovereign decision -> FAIL').",
            ConfigContractError,
            got=type(risk_engine).__name__,
        )
        require(
            isinstance(router, OrderRouter),
            "De engine vereist een OrderRouter.",
            ConfigContractError, got=type(router).__name__,
        )
        require(
            router.risk_config_hash == risk_engine.config_hash,
            "De router is gebouwd op een andere risicopolicy dan de engine. "
            "Twee regimes in één run is precies wat deze fase uitsluit.",
            ConfigContractError,
            router_hash=router.risk_config_hash,
            engine_hash=risk_engine.config_hash,
        )
        self._risk = risk_engine
        self._router = router
        self._ledger = Ledger(initial_equity)
        self._context_kind = context_kind
        self._participation_cap = float(risk_engine.config.adv_participation_cap)

    @property
    def ledger(self) -> Ledger:
        return self._ledger

    @property
    def risk_policy_hash(self) -> str:
        return self._risk.config_hash

    # ------------------------------------------------------------------ #
    # De event loop
    # ------------------------------------------------------------------ #
    def run(
        self,
        slices: Sequence[MarketSlice],
        exposures: Mapping[pd.Timestamp, Mapping[str, float]],
    ) -> BacktestResult:
        """Draai de keten bar voor bar.

        Parameters
        ----------
        slices : de marktsneden, chronologisch en strikt oplopend.
        exposures : beslismoment -> `a_t in [-1, +1]` per symbool. Bars zonder
            entry doen niets; dat is een vlak boek, geen fout.
        """
        require(
            len(slices) >= 2,
            "Een event-driven run heeft minstens twee bars nodig: één om te "
            "beslissen en één om op te vullen.",
            DataContractError, n_slices=len(slices),
        )
        timestamps = [s.ts for s in slices]
        require(
            all(b > a for a, b in pairwise(timestamps)),
            "De marktsneden staan niet strikt chronologisch. Een backtest die "
            "terugspringt in de tijd, meet niets.",
            DataContractError,
        )

        equity_points: list[float] = []
        snapshots: list[LedgerSnapshot] = []
        reports: list[ExecutionReport] = []
        decisions: list[RiskDecision] = []
        pending: list[Any] = []
        n_clipped = n_halted = 0

        hwm = self._ledger.initial_equity
        day_start = self._ledger.initial_equity

        for index, market in enumerate(slices):
            # ---------------------------------------------------------- #
            # 1. MARKET EVENT — waardeer tegen de prijzen van deze bar.
            #    Dit gebeurt VOOR alles, zodat elke beslissing hieronder op
            #    de boekstaat van dit moment werkt en niet op die van de
            #    vorige bar.
            # ---------------------------------------------------------- #
            snapshot = self._ledger.mark(dict(market.marks), market.ts)

            # ---------------------------------------------------------- #
            # 2. FILL EVENT — orders van eerdere bars die NU mogen vullen.
            #    Vóór de nieuwe beslissing, want de positie die daaruit volgt
            #    is de positie waarop straks wordt gesized.
            # ---------------------------------------------------------- #
            still_pending = []
            for order in pending:
                if order.ts_earliest_fill > market.ts:
                    still_pending.append(order)
                    continue
                report = self._fill(order, market)
                reports.append(report)
                if report.fill is not None:
                    self._ledger.apply_fill(report.fill)
            pending = still_pending

            # ---------------------------------------------------------- #
            # 3. FUNDING EVENT — settelt op de bar waarop hij valt.
            # ---------------------------------------------------------- #
            for symbol, rate in market.funding_rate.items():
                if symbol in self._ledger.positions():
                    self._ledger.apply_funding(
                        symbol,
                        rate=self._router.venue.cap_funding(rate),
                        mark_price=float(market.marks[symbol]),
                        ts=market.ts,
                    )

            snapshot = self._ledger.mark(dict(market.marks), market.ts)
            equity = snapshot.equity
            hwm = max(hwm, equity)
            equity_points.append(equity)
            snapshots.append(snapshot)

            # De laatste bar beslist niet: er is geen bar meer om op te vullen.
            if index + 1 >= len(slices):
                continue

            desired = exposures.get(market.ts)
            if not desired:
                continue

            # ---------------------------------------------------------- #
            # 4. SOVEREIGN RISK DECISION — vóór elke ordergedachte.
            # ---------------------------------------------------------- #
            context = build_context(
                self._context_kind, market,
                equity=equity, high_water_mark=hwm, day_start_equity=day_start,
                positions={s: p.qty for s, p in self._ledger.positions().items()},
                next_fill_ts=slices[index + 1].ts,
            )
            decision = self._risk.decide(
                desired, context.market_state(), context.risk_state()
            )
            decisions.append(decision)
            if decision.binding_constraints:
                n_clipped += 1
            if decision.risk_state_out.halted:
                n_halted += 1

            # ---------------------------------------------------------- #
            # 5. ORDER EVENT — alleen bereikbaar mét een besluit.
            # ---------------------------------------------------------- #
            pending.extend(self._router.build_orders(
                decision,
                equity=equity,
                marks=dict(market.marks),
                current_qty={s: p.qty for s, p in self._ledger.positions().items()},
                ts_decision=context.asof(),
                ts_earliest_fill=context.next_fill_ts(),
            ))

        require(
            not pending,
            "De run eindigt met openstaande orders die nooit konden vullen. "
            "Dat is een gat in de event loop, geen executie-eigenschap.",
            DataContractError, n_pending=len(pending),
        )

        index_ts = pd.DatetimeIndex([s.ts for s in slices], name="asof_ts")
        equity_curve = pd.Series(equity_points, index=index_ts, name="equity")
        returns = equity_curve.pct_change(fill_method=None).fillna(0.0).rename(
            "return")
        return BacktestResult(
            equity_curve=equity_curve,
            returns=returns,
            snapshots=tuple(snapshots),
            reports=tuple(reports),
            decisions=tuple(decisions),
            risk_policy_hash=self._risk.config_hash,
            impact_status=(reports[0].impact_status.value if reports
                           else "NO_ORDERS"),
            spread_status=(reports[0].spread_status.value if reports
                           else "NO_ORDERS"),
            n_bars=len(slices),
            n_orders=len(reports),
            n_fills=sum(1 for r in reports if r.fill is not None),
            n_rejected=sum(1 for r in reports if r.is_rejected),
            n_partial=sum(1 for r in reports if r.is_partial),
            n_sovereign_clipped=n_clipped,
            n_sovereign_halted=n_halted,
            audit={
                "engine": "event_driven",
                "context_kind": self._context_kind,
                "risk_audit_header": self._risk.audit_header(),
                "initial_equity": self._ledger.initial_equity,
            },
        )

    # ------------------------------------------------------------------ #
    def _fill(self, order: Any, market: MarketSlice) -> ExecutionReport:
        symbol = order.symbol
        require(
            symbol in market.marks,
            "Een openstaande order in een symbool dat op de fillbar geen "
            "prijs heeft. Er wordt niet gevuld tegen een aangenomen prijs.",
            DataContractError, symbol=symbol, ts=str(market.ts),
        )
        return self._router.execute(
            order,
            arrival_price=float(market.marks[symbol]),
            bar_volume_notional=float(market.bar_volume_notional.get(symbol, 0.0)),
            adv_notional=float(market.adv_notional.get(symbol, float("nan"))),
            sigma_daily=float(market.sigma_daily.get(symbol, float("nan"))),
            participation_cap=self._participation_cap,
        )


def build_slices(
    marks: pd.DataFrame,
    sigma_hat: pd.DataFrame,
    sigma_daily: pd.DataFrame,
    adv_notional: pd.DataFrame,
    bar_volume_notional: pd.DataFrame,
    clusters: Mapping[str, str],
    funding_rate: pd.DataFrame | None = None,
) -> list[MarketSlice]:
    """Zet uitgelijnde panels om in de replay-reeks die beide contexts voeden.

    Alle frames MOETEN dezelfde index en kolommen hebben. Dat wordt hier
    gecontroleerd en niet aangenomen: een stilzwijgende reindex is de meest
    onopvallende manier om een backtest een bar te laten kijken die er niet was.
    """
    frames = {
        "sigma_hat": sigma_hat, "sigma_daily": sigma_daily,
        "adv_notional": adv_notional, "bar_volume_notional": bar_volume_notional,
    }
    for name, frame in frames.items():
        require(
            bool(frame.index.equals(marks.index)),
            f"{name} staat niet op de tijdas van de markprijzen.",
            DataContractError, frame=name,
        )
        require(
            list(frame.columns) == list(marks.columns),
            f"{name} heeft andere kolommen dan de markprijzen.",
            DataContractError, frame=name,
        )
    funding = (funding_rate if funding_rate is not None
               else pd.DataFrame(0.0, index=marks.index, columns=marks.columns))

    out: list[MarketSlice] = []
    for ts in marks.index:
        row = marks.loc[ts]
        valid = [s for s in marks.columns if np.isfinite(row[s]) and row[s] > 0.0]
        if not valid:
            continue
        out.append(MarketSlice(
            ts=ts,
            marks={s: float(row[s]) for s in valid},
            sigma_hat={s: float(sigma_hat.at[ts, s]) for s in valid},
            sigma_daily={s: float(sigma_daily.at[ts, s]) for s in valid},
            adv_notional={s: float(adv_notional.at[ts, s]) for s in valid},
            bar_volume_notional={
                s: float(bar_volume_notional.at[ts, s]) for s in valid},
            clusters={s: str(clusters[s]) for s in valid if s in clusters},
            funding_rate={s: float(funding.at[ts, s]) for s in valid},
        ))
    return out


def exposures_from_frame(
    frame: pd.DataFrame,
) -> dict[pd.Timestamp, dict[str, float]]:
    """Zet een gewichtenpanel om in de `exposures`-mapping van `run()`.

    Rijen die volledig NaN zijn, leveren GEEN entry op: dat is een bar waarop de
    alpha-unit niets zegt, en de engine hoort dan niets te doen in plaats van
    een boek van nul af te dwingen.
    """
    out: dict[pd.Timestamp, dict[str, float]] = {}
    index = pd.DatetimeIndex(frame.index)
    for ts, (_, row) in zip(index, frame.iterrows(), strict=True):
        values = {str(s): float(v) for s, v in row.items() if np.isfinite(v)}
        if values:
            out[ts] = values
    return out


def iter_equity(result: BacktestResult) -> Iterable[tuple[pd.Timestamp, float]]:
    """Handige iterator voor rapportcode."""
    return zip(result.equity_curve.index, result.equity_curve.to_numpy(),
               strict=True)
