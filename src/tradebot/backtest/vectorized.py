# src/tradebot/backtest/vectorized.py
"""De vectorized research-engine — screening, nooit bewijs.

Phase 5, deliverable 3. Audit §16.1 is onvoorwaardelijk:

> *"Vectorized Research Engine: toegestaan uitsluitend in de exploratieve fase
> voor snelle hypothese-screening. Vectorized resultaten worden nooit
> geaccepteerd als bewijs voor modelpromotie."*

WAT DEZE MODULE TOEVOEGT AAN `baseline_runner.py`
--------------------------------------------------
Niets aan de wiskunde. `baseline_runner.run_baseline_tracks()` blijft de
Phase 3-berekening, ongewijzigd, want de vergelijkbaarheid met de bestaande
baseline hangt daarvan af.

Wat hier bij komt, is de MARKERING - en die is niet cosmetisch. Fase-opdracht
§15 eist twee dingen:

1. *"Iedere output moet expliciet `NOT_ADMISSIBLE_AS_PROMOTION_EVIDENCE`
   bevatten."* Daarom draagt `VectorizedResult` dat veld en kan het niet worden
   weggelaten: het is geen optionele annotatie maar een `Literal` met precies
   één toegestane waarde.

2. *"De promotion gate moet technisch weigeren om vectorized resultaten als
   promotiebewijs te accepteren."* Daarom staat `reject_vectorized_evidence()`
   hier, wordt hij door `registry/promotion.py` aangeroepen, en crasht hij in
   plaats van te loggen.

WAAROM EEN MARKERING EN GEEN VERBOD
------------------------------------
De vectorized engine mag bestaan en zou zonder hem trager onderzoek opleveren:
een screening van vijftig hypothesen door de event-driven engine kost uren waar
de vectorized versie seconden nodig heeft. Het gevaar zit niet in het rekenen
maar in het CITEREN. Een getal zonder herkomst verhuist na een week naar een
rapport, en daar is niet meer te zien dat het uit een engine kwam die geen
spread, geen latency en geen partial fills kent.

Ref: ARCHITECTUUR_AUDIT_2026-08-22.md secties 16, 16.1, 18.1, 26.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd

from ..utils.failfast import DataContractError, require

__all__ = [
    "NOT_ADMISSIBLE",
    "VectorizedResult",
    "is_vectorized_evidence",
    "reject_vectorized_evidence",
    "run_vectorized",
]

#: Het label dat elke vectorized output draagt. Letterlijk de tekst uit
#: fase-opdracht §15, zodat een grep door rapporten en artefacten hem vindt.
NOT_ADMISSIBLE: Literal["NOT_ADMISSIBLE_AS_PROMOTION_EVIDENCE"] = (
    "NOT_ADMISSIBLE_AS_PROMOTION_EVIDENCE"
)

#: Sleutel waaronder het label in platte dicts en JSON-artefacten verschijnt.
EVIDENCE_KEY = "evidence_class"


@dataclass(frozen=True, slots=True)
class VectorizedResult:
    """Een vectorized backtestresultaat, permanent gemarkeerd.

    `evidence_class` is een `Literal` met precies één toegestane waarde. Er is
    dus geen constructie waarin een vectorized resultaat zichzelf als
    toelaatbaar bewijs kan presenteren - ook niet per ongeluk, ook niet door een
    caller die het veld overschrijft.
    """

    equity_curve: pd.Series
    returns: pd.Series
    gross_returns: pd.Series
    turnover: pd.Series
    evidence_class: Literal["NOT_ADMISSIBLE_AS_PROMOTION_EVIDENCE"] = NOT_ADMISSIBLE
    audit: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require(
            self.evidence_class == NOT_ADMISSIBLE,
            "Een vectorized resultaat dat zichzelf niet als "
            "NOT_ADMISSIBLE_AS_PROMOTION_EVIDENCE markeert (audit §16.1).",
            DataContractError,
            evidence_class=self.evidence_class,
        )

    @property
    def counts_as_promotion_evidence(self) -> bool:
        """Altijd `False`. Bestaat zodat consumenten ernaar kunnen vragen."""
        return False

    def as_record(self) -> dict[str, Any]:
        return {
            EVIDENCE_KEY: self.evidence_class,
            "engine": "vectorized",
            "counts_as_promotion_evidence": False,
            "n_bars": int(len(self.equity_curve)),
            "final_equity": float(self.equity_curve.iloc[-1])
            if len(self.equity_curve) else None,
            **self.audit,
        }


def run_vectorized(
    weights: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    cost_per_side: float,
    initial_equity: float,
) -> VectorizedResult:
    """De Phase 3-conventie: `held = weights.shift(1)`, kosten op turnover.

    Deze functie is met opzet SIMPEL en blijft dat. Elke poging haar realistisch
    te maken - een spread erbij, een latency erbij - verkleint het verschil met
    de authoritative engine en daarmee de reden dat die bestaat.

    Wat zij NIET modelleert, en wat `backtest/engine.py` wel doet: executie op
    de volgende bar in plaats van op de beslisbar, marktimpact, partial fills,
    geweigerde orders, funding, een kasregister en een sluitende balans. Het
    verschil dat alleen al het EXECUTIEMOMENT maakt, is gemeten op ~18 bp
    per-bar RMSE; zie `tests/integration/test_engine_parity.py`.
    """
    require(
        bool(weights.index.equals(prices.index)),
        "Gewichten en prijzen staan niet op dezelfde tijdas.",
        DataContractError, n_weights=len(weights), n_prices=len(prices),
    )
    require(
        list(weights.columns) == list(prices.columns),
        "Gewichten en prijzen hebben andere kolommen.",
        DataContractError,
    )
    require(
        cost_per_side >= 0.0,
        "Negatieve kosten per eenheid turnover.",
        DataContractError, cost_per_side=cost_per_side,
    )

    returns = prices.pct_change(fill_method=None).fillna(0.0)
    held = weights.shift(1).fillna(0.0)
    gross = (held * returns).sum(axis=1)
    turnover = (weights - held).abs().sum(axis=1)
    net = gross - turnover * float(cost_per_side)
    equity = float(initial_equity) * (1.0 + net).cumprod()
    return VectorizedResult(
        equity_curve=equity.rename("equity"),
        returns=net.rename("return"),
        gross_returns=gross.rename("gross_return"),
        turnover=turnover.rename("turnover"),
        audit={
            "cost_per_side": float(cost_per_side),
            "initial_equity": float(initial_equity),
            "convention": "held = weights.shift(1); executed at the decision bar close",
        },
    )


# --------------------------------------------------------------------------- #
# De technische weigering
# --------------------------------------------------------------------------- #
def is_vectorized_evidence(payload: Any) -> bool:
    """Draagt dit object of deze mapping het niet-toelaatbaar-label?

    Werkt op zowel `VectorizedResult` als op de platte dicts die uit
    `as_record()` en uit JSON-artefacten komen, want dat is de vorm waarin een
    resultaat de promotion gate feitelijk bereikt.
    """
    if isinstance(payload, VectorizedResult):
        return True
    if isinstance(payload, Mapping):
        if payload.get(EVIDENCE_KEY) == NOT_ADMISSIBLE:
            return True
        if payload.get("engine") == "vectorized":
            return True
        if payload.get("counts_as_promotion_evidence") is False:
            return True
    return bool(getattr(payload, "evidence_class", None) == NOT_ADMISSIBLE)


def reject_vectorized_evidence(payload: Any, *, context: str = "") -> None:
    """Crash wanneer vectorized output als promotiebewijs wordt aangeboden.

    Fase-opdracht §15: *"De promotion gate moet technisch weigeren."* Loggen en
    doorgaan is geen weigering; dat is een weigering die je kunt negeren.
    """
    require(
        not is_vectorized_evidence(payload),
        "Vectorized backtestoutput wordt aangeboden als promotiebewijs. Audit "
        "§16.1: de vectorized engine is uitsluitend toegestaan voor "
        "hypothese-screening en telt nooit als bewijs voor modelpromotie. "
        "Draai het kandidaatmodel door `backtest/engine.py`.",
        DataContractError,
        context=context or "promotion gate",
        evidence_class=NOT_ADMISSIBLE,
    )
