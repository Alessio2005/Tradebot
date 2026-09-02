# src/tradebot/monitoring/execution_drift.py
"""Wat de beslislaag verwachtte tegen wat de beurs deed. Stage D-3.

WAAROM DIT NAAST `live_drift_monitor.py` STAAT
===============================================
`live_drift_monitor.py` bewaakt de FEATUREVERDELING: lijkt de wereld nog op de
wereld waarop is getraind. Deze module bewaakt iets anders en veel directers:
per ORDER, of de fill is wat de beslislaag dacht te krijgen. Drie grootheden, en
zij falen op drie verschillende manieren:

* **prijs** -- een fill die ongunstiger is dan verwacht. Slippage is normaal;
  25 bp ongunstig op een aangenomen half-spread van 1,0 bp is dat niet. Dan is
  er iets anders aan de hand: een verkeerd symbool, een verkeerde zijde, een
  order in een liquiditeitsgat, of een stale prijs in de beslislaag;
* **tijd** -- de latency van beslisbar tot fillbevestiging. De backtest rekent
  met `latency_bars = 1`; loopt de werkelijkheid daar structureel boven, dan
  handelt live op oudere informatie dan de gesimuleerde tegenhanger en is elke
  pariteitsclaim onjuist;
* **hoeveelheid** -- een partiele fill. Het boek dat ontstaat is dan een ander
  boek dan het boek dat is besloten, en de risicolaag heeft over dat andere boek
  geen oordeel geveld.

DE RICHTING VAN DE PRIJSDRIFT IS HET HELE PUNT
===============================================
Een fill boven de verwachte prijs is voor een KOOP ongunstig en voor een VERKOOP
gunstig. Wie het absolute verschil meet, telt meevallers als problemen en
middelt ze weg tegen tegenvallers -- en dan meet een gemiddelde drift van nul
een boek dat systematisch te duur koopt en te goedkoop verkoopt.
:func:`adverse_drift_bps` rekent daarom per zijde, met een positief getal dat
altijd "slechter dan verwacht" betekent.

VERKLAARD EN ONVERKLAARD
=========================
D-6 laat de 60-daagse teller doorlopen bij een VERKLAARD verschil (een
exchange-outage die is gedocumenteerd) en herstart hem bij een onverklaard
verschil. Deze module kent dat onderscheid daarom expliciet: een
:class:`ExecutionDrift` draagt een `explanation`, en
:meth:`ExecutionDriftMonitor.unexplained` levert precies de gevallen die de
teller breken. Een verklaring is een handeling van een mens; zij wordt hier
vastgelegd, niet afgeleid.

Ref: fase-opdracht Stage D-3 en D-6; `conf/monitoring/default.yaml`;
`reports/phase7_divergence_map.md` §6.1.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from ..oms.order import Fill, Order, OrderSide
from ..schemas.config import MonitoringConfig, monitoring_config
from ..utils.failfast import DataContractError, require

__all__ = [
    "ExecutionDrift",
    "ExecutionDriftMonitor",
    "adverse_drift_bps",
]

#: De drie poortnamen. Zij staan als constante zodat een rapport en een test
#: dezelfde string gebruiken en een typefout niet stilzwijgend nooit matcht.
PRICE = "price_drift"
LATENCY = "fill_latency"
FILL_RATIO = "fill_ratio"


def adverse_drift_bps(
    side: OrderSide, expected_price: float, fill_price: float
) -> float:
    """Hoeveel basispunten ONGUNSTIGER de fill was dan verwacht.

    Positief is altijd slechter. Voor een koop is dat te duur betalen, voor een
    verkoop te goedkoop verkopen. Een negatieve waarde is een meevaller en
    breekt niets.
    """
    require(
        expected_price > 0.0 and fill_price > 0.0,
        "Een prijsdrift op een niet-positieve prijs. Dat is geen drift maar "
        "een kapotte prijs, en die hoort eerder in de keten te crashen.",
        DataContractError,
        expected_price=expected_price, fill_price=fill_price,
    )
    direction = 1.0 if side is OrderSide.BUY else -1.0
    return direction * (fill_price - expected_price) / expected_price * 10_000.0


@dataclass(frozen=True)
class ExecutionDrift:
    """Eén order: verwacht tegen werkelijk, met het oordeel erbij."""

    order_id: str
    symbol: str
    side: OrderSide
    expected_price: float
    fill_price: float
    adverse_bps: float
    expected_qty: float
    fill_qty: float
    fill_ratio: float
    decision_ts: pd.Timestamp
    fill_ts: pd.Timestamp
    latency_seconds: float
    breached: tuple[str, ...]
    #: Een gedocumenteerde oorzaak (exchange-outage, geplande maintenance).
    #: `None` betekent ONVERKLAARD, en dat is wat de 60-daagse teller breekt.
    explanation: str | None = None

    @property
    def is_breach(self) -> bool:
        return bool(self.breached)

    @property
    def breaks_the_clock(self) -> bool:
        """Onverklaard én boven een drempel. Zie D-6."""
        return self.is_breach and self.explanation is None

    def as_record(self) -> dict[str, Any]:
        return {
            "order_id": self.order_id, "symbol": self.symbol,
            "side": self.side.value,
            "expected_price": self.expected_price,
            "fill_price": self.fill_price, "adverse_bps": self.adverse_bps,
            "expected_qty": self.expected_qty, "fill_qty": self.fill_qty,
            "fill_ratio": self.fill_ratio,
            "decision_ts": self.decision_ts.isoformat(),
            "fill_ts": self.fill_ts.isoformat(),
            "latency_seconds": self.latency_seconds,
            "breached": list(self.breached),
            "explanation": self.explanation,
            "breaks_the_clock": self.breaks_the_clock,
        }


@dataclass
class ExecutionDriftMonitor:
    """Meet elke fill tegen de verwachting van de beslislaag.

    De drempels komen uit `conf/monitoring/default.yaml` en worden vóór de
    60-daagse klok bevroren (no-go 10). Zij staan niet in deze module.
    """

    cfg: MonitoringConfig = field(default_factory=monitoring_config)
    _drifts: list[ExecutionDrift] = field(default_factory=list, init=False)

    def record(
        self,
        order: Order,
        fill: Fill,
        *,
        expected_price: float,
        explanation: str | None = None,
    ) -> ExecutionDrift:
        """Leg één fill vast tegen de prijs waarop is besloten.

        `expected_price` is een ARGUMENT en geen veld op `Order`, omdat de
        beslisprijs bij de laag hoort die de order maakte. Hem hier afleiden uit
        de fill zou de meting circulair maken: dan meet je altijd nul.
        """
        require(
            order.order_id == fill.order_id,
            "Deze fill hoort niet bij deze order. Een drift over twee "
            "verschillende orders is geen meting maar een verwisseling.",
            DataContractError,
            order_id=order.order_id, fill_order_id=fill.order_id,
        )
        adverse = adverse_drift_bps(order.side, expected_price, fill.fill_price)
        ratio = (float(fill.fill_qty) / float(order.qty_base)
                 if order.qty_base else 0.0)
        latency = float(
            (pd.Timestamp(fill.fill_ts) - pd.Timestamp(order.created_at))
            .total_seconds())

        breached: list[str] = []
        if adverse > self.cfg.max_adverse_price_drift_bps:
            breached.append(PRICE)
        if latency > self.cfg.max_fill_latency_seconds:
            breached.append(LATENCY)
        if ratio < self.cfg.min_fill_ratio:
            breached.append(FILL_RATIO)

        drift = ExecutionDrift(
            order_id=order.order_id, symbol=order.symbol, side=order.side,
            expected_price=float(expected_price),
            fill_price=float(fill.fill_price), adverse_bps=adverse,
            expected_qty=float(order.qty_base), fill_qty=float(fill.fill_qty),
            fill_ratio=ratio, decision_ts=pd.Timestamp(order.created_at),
            fill_ts=pd.Timestamp(fill.fill_ts), latency_seconds=latency,
            breached=tuple(breached), explanation=explanation,
        )
        self._drifts.append(drift)
        return drift

    @property
    def drifts(self) -> Sequence[ExecutionDrift]:
        return tuple(self._drifts)

    def breaches(self) -> tuple[ExecutionDrift, ...]:
        """Elke order die minstens één drempel raakte."""
        return tuple(d for d in self._drifts if d.is_breach)

    def unexplained(self) -> tuple[ExecutionDrift, ...]:
        """De gevallen die de 60-daagse teller breken (D-6)."""
        return tuple(d for d in self._drifts if d.breaks_the_clock)

    def summary(self) -> dict[str, Any]:
        """Wat er in `PAPER_TRADING_LOG.md` hoort te staan.

        `worst_adverse_bps` is een MAXIMUM en geen gemiddelde: een gemiddelde
        drift middelt een enkele ernstige misfill weg tegen honderd schone
        fills, en het is precies die ene die iets betekent.
        """
        if not self._drifts:
            return {"n_orders": 0, "n_breaches": 0, "n_unexplained": 0,
                    "worst_adverse_bps": float("nan"),
                    "worst_latency_seconds": float("nan"),
                    "min_fill_ratio": float("nan")}
        return {
            "n_orders": len(self._drifts),
            "n_breaches": len(self.breaches()),
            "n_unexplained": len(self.unexplained()),
            "worst_adverse_bps": max(d.adverse_bps for d in self._drifts),
            "worst_latency_seconds": max(
                d.latency_seconds for d in self._drifts),
            "min_fill_ratio": min(d.fill_ratio for d in self._drifts),
        }
