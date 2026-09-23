# src/tradebot/risk/binding_audit.py
"""Welke limiet bindt, op hoeveel bars, en met hoeveel. Fase 11, stap 3.

WAAROM DIT BESTAND BESTAAT
==========================
De ladder (`artefacts/baseline/phase5_revaluation.json`) telt dat de soevereine
laag op 1.742 van 1.743 bars ingrijpt. Welke limiet dat is, stond nergens: een
teller zonder inhoud. Dezelfde vraag had al twee andere namen -- DI-27 ("de
vol-target bindt niet") en DI-30 (een uniforme factor valt niet meer weg) -- en
na AD-26 kreeg zij een derde: failure 1 van fase 11 §3.1.

WAT HIER WORDT GEMETEN, EN HOE
==============================
`RiskEngine.trace` geeft de keten van `decide` stap voor stap terug: het boek
vóór en na elke limiet in `constraint_order`. Dit bestand bouwt die keten NIET
na; het leest haar. `TracingRiskEngine` is een `RiskEngine` die bij elk besluit
ook de trace bewaart, zodat de ladderlussen (L1 in
`backtest/phase5_baseline.py::_sovereign_weights`, L3 in de event-driven
engine) ongewijzigd kunnen worden gemeten via hun `engine_factory`.

Per stap twee grootheden, en hun verschil is zelf een meting:

* `changed`  -- het boek is na deze stap anders dan ervoor (boven
  `DUST_TOLERANCE`). Dat is wat de limiet DEED.
* `recorded` -- het auditspoor heeft voor deze stap een `BindingConstraint`.
  Dat is wat de limiet MELDDE.

`n_disagree` telt de bars waarop die twee uiteenlopen. Nul is het contract uit
`docs/RISK_CONTRACT.md` §5.2 (geen ingreep zonder registratie, geen registratie
zonder ingreep); een ander getal is een bevinding.

DE VOL-TARGET, APART
====================
`apply_volatility_target` schaalt met `w_t = min(1, min(max_leverage,
sigma_target / sigma_boek))`. De laag VERKLEINT uitsluitend, dus voor elke
`max_leverage >= 1` valt `max_leverage` uit de formule. `summarise_traces` telt
daarom twee dingen naast elkaar: op hoeveel besluiten de verhouding onder
`max_leverage` ligt (de lezing van de fase-11-prompt, stap 3.3) en op hoeveel
onder 1 (wat de code werkelijk doet). Het verschil tussen die twee tellingen is
het aantal besluiten waarop `max_leverage` had kunnen tellen en het niet deed.
"""
from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..schemas.config import RiskConfig
from ..utils.failfast import DataContractError, require
from .contract import MarketState, RiskDecision, RiskState
from .engine import DUST_TOLERANCE, ChainStep, RiskEngine, risk_config_hash
from .vol_targeting import book_sigma_hat

__all__ = [
    "StepEffect",
    "TraceRecord",
    "TracingRiskEngine",
    "audit_layers",
    "first_divergent_step",
    "policy_from_registry",
    "step_effects",
    "summarise_traces",
    "vol_ratio",
]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_REGISTRY = _REPO_ROOT / "artefacts/governance/risk_config_registry.json"

#: De stap waarna een uniforme factor hoort weg te vallen (REGEL V, AD-25).
_PIVOT = "vol_target"


@dataclass(frozen=True)
class TraceRecord:
    """Eén besluit, met de keten die het produceerde."""

    asof_ts: pd.Timestamp
    steps: tuple[ChainStep, ...]
    sigma_hat: Mapping[str, float]


@dataclass(frozen=True)
class StepEffect:
    """Wat één stap met het boek deed, naast wat hij daarover meldde."""

    name: str
    changed: bool
    recorded: bool
    gross_before: float
    gross_after: float

    @property
    def shrink(self) -> float:
        """Het deel van de gross dat deze stap wegnam; 0 op een vlak boek."""
        if self.gross_before <= DUST_TOLERANCE:
            return 0.0
        return 1.0 - self.gross_after / self.gross_before


class TracingRiskEngine(RiskEngine):
    """Een `RiskEngine` die bij elk besluit zijn keten bewaart.

    Het besluit zelf is bit-identiek aan dat van `RiskEngine` (getest): de
    trace is een tweede, pure aanroep van dezelfde lus. Zonder `HaltStore`
    gebruiken, anders schrijft een vurende kill switch twee keer weg.
    """

    def __init__(self, config: RiskConfig) -> None:
        super().__init__(config)
        self.records: list[TraceRecord] = []

    def decide(
        self,
        desired_exposure: Mapping[str, float],
        market_state: MarketState,
        risk_state: RiskState,
    ) -> RiskDecision:
        steps, _ = self.trace(desired_exposure, market_state, risk_state)
        self.records.append(TraceRecord(
            asof_ts=market_state.asof_ts, steps=steps,
            sigma_hat=dict(market_state.sigma_hat)))
        return super().decide(desired_exposure, market_state, risk_state)


def _gross(book: Mapping[str, float]) -> float:
    return float(sum(abs(float(w)) for w in book.values()))


def step_effects(steps: Sequence[ChainStep]) -> tuple[StepEffect, ...]:
    """Per stap: veranderde het boek, en meldde de stap dat?"""
    out: list[StepEffect] = []
    for step in steps:
        before, after = step.exposure_before, step.exposure_after
        changed = any(abs(float(after.get(s, 0.0)) - float(w)) > DUST_TOLERANCE
                      for s, w in before.items())
        out.append(StepEffect(
            name=step.name, changed=changed, recorded=bool(step.bound),
            gross_before=_gross(before), gross_after=_gross(after)))
    return tuple(out)


def vol_ratio(record: TraceRecord, cfg: RiskConfig) -> float:
    """`sigma_target / sigma_boek` op het boek zoals de vol-target het zag.

    REGEL V geldt op dit besluit voor een uniforme factor `c` dan en slechts
    dan als `c >= vol_ratio`: daaronder schaalt de vol-target het geschaalde
    boek niet meer terug, want hij verkleint uitsluitend.
    """
    step = next(s for s in record.steps if s.name == _PIVOT)
    sigma_book = book_sigma_hat(step.exposure_before, record.sigma_hat)
    return math.inf if sigma_book <= 0.0 else float(cfg.sigma_target) / sigma_book


@dataclass
class _Tally:
    n_changed: int = 0
    n_recorded: int = 0
    n_disagree: int = 0
    shrinks: list[float] = field(default_factory=list)


def summarise_traces(records: Iterable[TraceRecord], cfg: RiskConfig) -> dict[str, Any]:
    """Per limiet: op hoeveel besluiten hij het boek veranderde, en met hoeveel.

    `mean_shrink_when_changed` is de gemiddelde fractie van de gross die de
    stap wegnam OP de besluiten waar hij bond; `mean_shrink_all` middelt over
    alle besluiten. Een limiet die zelden bindt maar dan hard, en een die altijd
    een beetje bindt, zijn zo te onderscheiden.
    """
    records = list(records)
    tallies = {name: _Tally() for name in cfg.constraint_order}
    ratios: list[float] = []
    desired: list[float] = []
    permitted: list[float] = []
    for record in records:
        for effect in step_effects(record.steps):
            tally = tallies.setdefault(effect.name, _Tally())
            tally.n_changed += effect.changed
            tally.n_recorded += effect.recorded
            tally.n_disagree += effect.changed != effect.recorded
            tally.shrinks.append(effect.shrink)
        if any(s.name == _PIVOT for s in record.steps):
            ratios.append(vol_ratio(record, cfg))
        desired.append(_gross(record.steps[0].exposure_before))
        permitted.append(_gross(record.steps[-1].exposure_after))
    n = len(records)
    finite = np.array([r for r in ratios if math.isfinite(r)], dtype="float64")
    return {
        "risk_policy_hash": risk_config_hash(cfg),
        "n_decisions": n,
        "mean_gross_desired": float(np.mean(desired)) if desired else 0.0,
        "mean_gross_permitted": float(np.mean(permitted)) if permitted else 0.0,
        "steps": {
            name: {
                "n_changed": t.n_changed,
                "share_changed": t.n_changed / n if n else 0.0,
                "n_recorded": t.n_recorded,
                "n_disagree": t.n_disagree,
                "mean_shrink_when_changed": float(np.mean(
                    [s for s in t.shrinks if s > DUST_TOLERANCE] or [0.0])),
                "mean_shrink_all": float(np.mean(t.shrinks or [0.0])),
            }
            for name, t in tallies.items()
        },
        _PIVOT: {
            "sigma_target": float(cfg.sigma_target),
            "max_leverage": float(cfg.max_leverage),
            "n_ratio_below_one": int(sum(r < 1.0 for r in ratios)),
            "n_ratio_below_max_leverage": int(
                sum(r < float(cfg.max_leverage) for r in ratios)),
            "n_flat_book": int(sum(not math.isfinite(r) for r in ratios)),
            # REGEL V geldt op bar t voor factor c precies als c >= ratio(t).
            # Het maximum is dus de kleinste c waarvoor zij op ELKE bar houdt.
            "ratio_max": float(finite.max()) if finite.size else math.nan,
            "ratio_quantiles": (
                {f"q{int(q * 100):02d}": float(np.quantile(finite, q))
                 for q in (0.05, 0.25, 0.5, 0.75, 0.95)} if finite.size else {}),
        },
    }


#: De laddergetallen die naast de ketensamenvatting worden bewaard. Uit DEZELFDE
#: run, zodat een bindingstelling en de Sharpe ernaast nooit uit twee
#: verschillende runs kunnen komen.
LADDER_KEYS = ("net_sharpe", "gross_sharpe", "volatility", "max_drawdown",
               "mean_gross", "n_sovereign_halted", "n_sovereign_clipped")


def audit_layers(
    run: Callable[[Callable[[str, Any], RiskEngine]], Sequence[Any]],
    cfg: RiskConfig,
) -> dict[str, dict[str, Any]]:
    """Meet één ladderrun per laag: de keten per besluit, plus de laddergetallen.

    `run(engine_factory)` draait de ladder -- in de app is dat
    `run_ladder_track(..., engine_factory=...)`. Deze functie levert de factory
    die per laag een `TracingRiskEngine` inzet, en leest daarna wat die zagen.
    Zij kent de ladder dus niet en bouwt haar niet na.
    """
    tracers: dict[str, TracingRiskEngine] = {}

    def factory(layer: str, risk_cfg: Any) -> RiskEngine:
        tracers[layer] = TracingRiskEngine(risk_cfg)
        return tracers[layer]

    rows = {r.layer: r.as_record() for r in run(factory)}
    return {layer: {**summarise_traces(t.records, cfg),
                    "ladder": {k: rows[layer][k] for k in LADDER_KEYS
                               if k in rows[layer]}}
            for layer, t in tracers.items()}


def policy_from_registry(config_hash: str, path: Path | str | None = None) -> RiskConfig:
    """Reconstrueer een geregistreerd beleid, en bewijs dat het dat beleid IS.

    Het register bewaart per beleid de volledige gevalideerde configuratie.
    Een reconstructie die niet op haar eigen hash uitkomt, meet een ander
    beleid dan zij noemt -- precies het defect waar fase 11 over gaat -- en
    wordt daarom geweigerd in plaats van gebruikt.
    """
    p = Path(path) if path is not None else _REGISTRY
    doc = json.loads(p.read_text(encoding="utf-8"))
    entries = [e for e in doc.get("entries", []) if e.get("config_hash") == config_hash]
    require(
        len(entries) == 1,
        "Dit beleid staat niet (precies één keer) in het risicoconfiguratie-"
        "register. Een meting onder een niet-geregistreerd beleid is niet "
        "auditbaar (AD-27).",
        DataContractError, config_hash=config_hash, n_entries=len(entries),
    )
    cfg = RiskConfig(**entries[0]["config"])
    rebuilt = risk_config_hash(cfg)
    require(
        rebuilt == config_hash,
        f"De configuratie die het register onder {config_hash} bewaart, hasht "
        f"naar {rebuilt}. Het register is gewijzigd of beschadigd; een meting "
        "hierop zou onder een ander beleid draaien dan zij noemt.",
        DataContractError, config_hash=config_hash, rebuilt=rebuilt,
    )
    return cfg


def _close(a: Mapping[str, float], b: Mapping[str, float], scale: float) -> bool:
    """`a == scale * b`, op relatieve precisie."""
    keys = set(a) | set(b)
    ref = max((abs(float(b.get(k, 0.0))) for k in keys), default=0.0)
    tol = max(DUST_TOLERANCE, 1e-9 * abs(scale) * ref)
    return all(abs(float(a.get(k, 0.0)) - scale * float(b.get(k, 0.0))) <= tol
               for k in keys)


def first_divergent_step(
    full: Sequence[ChainStep], scaled: Sequence[ChainStep], factor: float,
) -> str | None:
    """De eerste stap waarna `factor * a` niet meer uitkomt waar REGEL V zegt.

    Vóór de vol-target hoort het geschaalde boek `factor` keer het volle te
    zijn; vanaf de vol-target hoort het er GELIJK aan te zijn, want de
    vol-target deelt de factor weer weg. De eerste stap waarop die verwachting
    breekt, is de stap die de invariantie breekt. `None` betekent dat zij
    over de hele keten houdt.
    """
    require(
        [s.name for s in full] == [s.name for s in scaled],
        "De twee ketens lopen niet door dezelfde stappen; dan is er niets te "
        "vergelijken.",
        DataContractError,
    )
    past_pivot = False
    for a, b in zip(full, scaled, strict=True):
        past_pivot = past_pivot or a.name == _PIVOT
        expected_scale = 1.0 if past_pivot else float(factor)
        if not _close(b.exposure_after, a.exposure_after, expected_scale):
            return a.name
    return None
