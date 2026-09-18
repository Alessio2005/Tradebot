# src/tradebot/risk/kill_switches.py
"""L7 kill switches — Drawdown Breaker (HWM, getrapt) en Daily Loss Governor.

Phase 4, deliverable 3 / stap 5. Twee mechanismen met verschillende aard:

* **Drawdown Breaker** op de High-Water Mark, met GETRAPTE de-grossing. Tot de
  harde grens is hij een multiplier, geen schakelaar. Een breaker die pas bij
  de eindlimiet iets doet, doet niets in het traject waarin ingrijpen nog
  goedkoop is.
* **Daily Loss Governor** als ONMIDDELLIJKE kill switch. Geen trappen: de
  governor is binair van aard. Hij WAS een propfirm-lijn; sinds de
  mandaatwijziging (`docs/RISK_MANDATE.md`) is hij een gap-containmentlijn op
  eigen kapitaal. De aard verandert niet mee: binair blijft binair, want een
  getrapte reactie op een sprong is te laat.

DE HALTED-TOESTAND IS EEN EENRICHTINGSDEUR
------------------------------------------
Exit-criterium 5: de `HALTED`-toestand overleeft een procesherstart en wordt
uitsluitend handmatig opgeheven. Vóór Phase 4 haalde geen van de vier
bestaande breakers dat (sectie 4.1 van `reports/phase4_entanglement_map.md`):
`live/circuit_breaker.py` schreef zijn trip weg naar een append-only log, maar
de TOESTAND zelf zat in `self._halt_reason` en was na een herstart weg. Een
kill switch die een crash niet overleeft, beschermt precies niet tegen het
scenario waarin hij het hardst nodig is - een proces dat omvalt tijdens een
verliesreeks en opnieuw opstart.

`HaltStore` maakt de toestand duurzaam:

* `engage()` schrijft een `HaltRecord` en is NIET overschrijvend. Wie al gehalt
  is, blijft gehalt met de OORSPRONKELIJKE reden: de eerste oorzaak is de ware
  oorzaak, en een tweede trigger mag hem niet maskeren.
* `release()` is de enige weg terug, vereist een operator en een motivering, en
  wordt zelf ook weggeschreven. Er is geen tijdgebaseerde, geen
  herstel-gebaseerde en geen automatische ontgrendeling.
* Er is bewust GEEN `engage()`-variant die een bestaande halt wist.

DE HWM IS CAUSAAL
-----------------
De drawdown wordt gemeten tegen `risk_state.high_water_mark`, en die wordt
voorwaarts opgebouwd (`advance_high_water_mark`). Een HWM over de volledige
sample is een lookahead-lek: de breaker vuurt dan systematisch te laat in het
eerste deel van de reeks en te vroeg in het tweede.

Ref: ARCHITECTUUR_AUDIT_2026-08-22.md secties 14, 14.1, 19 (L7/L13), 23.
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..utils.failfast import ConfigContractError, DataContractError, require
from .contract import BOOK_SCOPE, BindingConstraint, ConstraintKind, RiskState

__all__ = [
    "DAILY_LOSS_KEY",
    "DRAWDOWN_LEVELS_KEY",
    "MAX_DRAWDOWN_KEY",
    "HaltRecord",
    "HaltStore",
    "advance_high_water_mark",
    "apply_daily_loss_governor",
    "apply_drawdown_breaker",
    "apply_halt",
    "degrossing_multiplier",
]

DRAWDOWN_LEVELS_KEY = "risk.drawdown_breaker_levels"
MAX_DRAWDOWN_KEY = "risk.max_drawdown_pct"
DAILY_LOSS_KEY = "risk.daily_loss_limit"

_TOL = 1e-12


@dataclass(frozen=True, slots=True)
class HaltRecord:
    """Waarom het boek gehalt is. Onveranderlijk zodra geschreven."""

    kind: str
    reason: str
    measured: float
    threshold: float
    config_key: str
    halted_at: str

    def as_record(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "reason": self.reason,
            "measured": float(self.measured),
            "threshold": float(self.threshold),
            "config_key": self.config_key,
            "halted_at": self.halted_at,
        }

    @staticmethod
    def from_record(raw: Mapping[str, Any]) -> HaltRecord:
        missing = {
            "kind", "reason", "measured", "threshold", "config_key", "halted_at"
        } - set(raw)
        require(
            not missing,
            "Corrupte halt-state op schijf: verplichte velden ontbreken. Een "
            "onleesbare kill-switch-state wordt NIET als 'niet gehalt' "
            "geinterpreteerd - dat zou een crash tijdens het wegschrijven "
            "omzetten in toestemming om te handelen.",
            DataContractError,
            missing=sorted(missing),
        )
        return HaltRecord(
            kind=str(raw["kind"]),
            reason=str(raw["reason"]),
            measured=float(raw["measured"]),
            threshold=float(raw["threshold"]),
            config_key=str(raw["config_key"]),
            halted_at=str(raw["halted_at"]),
        )


class HaltStore:
    """Duurzame, onherroepelijke `HALTED`-toestand op schijf.

    De actieve halt staat in `path`; elke overgang (engage EN release) wordt
    daarnaast weggeschreven naar een append-only journaal ernaast, zodat de
    geschiedenis niet verloren gaat wanneer de actieve halt wordt opgeheven.
    """

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        self._journal = self._path.with_suffix(".journal.jsonl")

    @property
    def path(self) -> Path:
        return self._path

    @property
    def journal_path(self) -> Path:
        return self._journal

    def load(self) -> HaltRecord | None:
        """De actieve halt, of `None`. Crasht op een corrupt bestand."""
        if not self._path.is_file():
            return None
        text = self._path.read_text(encoding="utf-8").strip()
        require(
            bool(text),
            "Halt-state bestaat maar is leeg. Dat duidt op een afgebroken "
            "schrijfactie; het wordt niet als 'niet gehalt' gelezen.",
            DataContractError,
            path=str(self._path),
        )
        raw = json.loads(text)
        require(
            isinstance(raw, dict),
            "Halt-state is geen object.",
            DataContractError,
            path=str(self._path),
        )
        return HaltRecord.from_record(raw)

    def is_halted(self) -> bool:
        return self.load() is not None

    def engage(self, record: HaltRecord) -> HaltRecord:
        """Halt het boek. Een BESTAANDE halt wordt NOOIT overschreven.

        De eerste oorzaak is de ware oorzaak. Een tweede trigger die de reden
        overschrijft, maakt de post-mortem onmogelijk: je ziet dan de laatste
        drempel die raakte, niet die welke het boek stopte.
        """
        existing = self.load()
        if existing is not None:
            return existing
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(record.as_record(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self._append_journal({"event": "engage", **record.as_record()})
        return record

    def release(self, *, operator: str, justification: str) -> HaltRecord:
        """De ENIGE weg terug. Handmatig, met naam en motivering.

        Er is geen tijdgebaseerde, geen herstel-gebaseerde en geen
        automatische ontgrendeling. Audit sectie 14 laat geen bypass toe, en
        een kill switch die vanzelf opheft is een bypass met een klok eraan.
        """
        require(
            bool(operator.strip()) and bool(justification.strip()),
            "Een handmatige reset van de HALTED-toestand vereist een operator "
            "en een motivering. Zonder die twee is de reset niet auditbaar en "
            "dus niet toegestaan.",
            DataContractError,
            operator=operator,
            justification=justification,
        )
        active = self.load()
        require(
            active is not None,
            "Er is geen actieve HALTED-toestand om op te heffen.",
            DataContractError,
            path=str(self._path),
        )
        assert active is not None  # door require() gegarandeerd
        self._append_journal(
            {
                "event": "release",
                "operator": operator,
                "justification": justification,
                "released_at": pd.Timestamp.utcnow().isoformat(),
                "released_halt": active.as_record(),
            }
        )
        self._path.unlink()
        return active

    def _append_journal(self, entry: Mapping[str, Any]) -> None:
        self._journal.parent.mkdir(parents=True, exist_ok=True)
        with self._journal.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, sort_keys=True) + "\n")


def advance_high_water_mark(state: RiskState, equity: float) -> RiskState:
    """Werk de causale HWM voorwaarts bij met een nieuwe equity-waarde.

    De HWM is monotoon niet-dalend en gebruikt uitsluitend informatie tot en
    met `t`. `RiskState` weigert een HWM onder de equity, dus deze functie is
    de enige juiste manier om equity bij te werken.
    """
    eq = float(equity)
    require(
        np.isfinite(eq) and eq > 0.0,
        "Niet-eindige of niet-positieve equity aangeboden aan de HWM.",
        DataContractError,
        equity=eq,
    )
    return RiskState(
        equity=eq,
        high_water_mark=max(float(state.high_water_mark), eq),
        day_start_equity=float(state.day_start_equity),
        halted=state.halted,
        halt_reason=state.halt_reason,
        halted_at=state.halted_at,
    )


def apply_halt(
    exposures: Mapping[str, float], state: RiskState
) -> tuple[dict[str, float], list[BindingConstraint]]:
    """Een gehalt boek heeft geen exposure. Punt.

    Dit is de eerste stap in `constraint_order`: alles daarna is betekenisloos.
    """
    out = {str(k): float(v) for k, v in exposures.items()}
    if not state.halted:
        return out, []

    gross = float(sum(abs(v) for v in out.values()))
    bound = [
        BindingConstraint(
            kind=ConstraintKind.HALTED,
            scope=BOOK_SCOPE,
            measured=gross,
            threshold=0.0,
            exposure_before=gross,
            exposure_after=0.0,
            config_key="risk_state.halted",
        )
    ] if gross > _TOL else []
    return {s: 0.0 for s in out}, bound


def degrossing_multiplier(
    drawdown: float, levels: Sequence[Any]
) -> tuple[float, float]:
    """De getrapte de-grossing-multiplier bij een gegeven drawdown.

    Returns
    -------
    (multiplier, bindende_drempel)
        `multiplier` is 1.0 zolang geen trap is geraakt, en `drempel` dan 0.0.
        De DIEPSTE geraakte trap wint - de trappen lopen strikt op in
        `drawdown` en strikt af in `gross_multiplier`, afgedwongen door
        `RiskConfig._tiers_are_monotone`.
    """
    dd = float(drawdown)
    require(
        np.isfinite(dd),
        "Niet-eindige drawdown aangeboden aan de Drawdown Breaker.",
        DataContractError,
        drawdown=dd,
    )
    multiplier = 1.0
    threshold = 0.0
    for tier in levels:
        tier_dd = float(tier.drawdown)
        if dd >= tier_dd - _TOL:
            multiplier = float(tier.gross_multiplier)
            threshold = tier_dd
    return multiplier, threshold


def apply_drawdown_breaker(
    exposures: Mapping[str, float],
    state: RiskState,
    *,
    levels: Sequence[Any],
    max_drawdown_pct: float,
    store: HaltStore | None = None,
    asof_ts: pd.Timestamp | None = None,
) -> tuple[dict[str, float], list[BindingConstraint], RiskState]:
    """Drawdown Breaker op de High-Water Mark, getrapt tot de harde grens.

    Boven `max_drawdown_pct` is het geen de-grossing meer maar een HALT, en
    die is onherroepelijk tot een handmatige `HaltStore.release()`.
    """
    hard = float(max_drawdown_pct)
    require(
        np.isfinite(hard) and 0.0 < hard <= 1.0,
        "max_drawdown_pct moet een fractie in (0, 1] zijn.",
        ConfigContractError,
        key=MAX_DRAWDOWN_KEY,
        value=hard,
    )
    out = {str(k): float(v) for k, v in exposures.items()}
    dd = state.drawdown

    if dd >= hard - _TOL:
        return _engage_halt(
            out,
            state,
            kind=ConstraintKind.DRAWDOWN_BREAKER,
            reason=(
                f"Drawdown {dd:.4f} bereikte de harde grens {hard:.4f} op de "
                "High-Water Mark."
            ),
            measured=dd,
            threshold=hard,
            config_key=MAX_DRAWDOWN_KEY,
            store=store,
            asof_ts=asof_ts,
        )

    multiplier, tier_threshold = degrossing_multiplier(dd, levels)
    if multiplier >= 1.0:
        return out, [], state

    gross = float(sum(abs(v) for v in out.values()))
    scaled = {s: w * multiplier for s, w in out.items()}
    bound = [
        BindingConstraint(
            kind=ConstraintKind.DRAWDOWN_BREAKER,
            scope=BOOK_SCOPE,
            measured=dd,
            threshold=tier_threshold,
            exposure_before=gross,
            exposure_after=gross * multiplier,
            config_key=DRAWDOWN_LEVELS_KEY,
        )
    ] if gross > _TOL else []
    return scaled, bound, state


def apply_daily_loss_governor(
    exposures: Mapping[str, float],
    state: RiskState,
    *,
    daily_loss_limit: float,
    store: HaltStore | None = None,
    asof_ts: pd.Timestamp | None = None,
) -> tuple[dict[str, float], list[BindingConstraint], RiskState]:
    """Daily Loss Governor — onmiddellijke, onherroepelijke kill switch.

    Geen trappen. Het verlies wordt gemeten sinds `day_start_equity`.
    """
    limit = float(daily_loss_limit)
    require(
        np.isfinite(limit) and 0.0 < limit <= 1.0,
        "daily_loss_limit moet een fractie in (0, 1] zijn.",
        ConfigContractError,
        key=DAILY_LOSS_KEY,
        value=limit,
    )
    out = {str(k): float(v) for k, v in exposures.items()}
    loss = state.daily_loss
    if loss < limit - _TOL:
        return out, [], state

    return _engage_halt(
        out,
        state,
        kind=ConstraintKind.DAILY_LOSS_GOVERNOR,
        reason=(
            f"Dagverlies {loss:.4f} bereikte de limiet {limit:.4f} sinds de "
            "openingsequity van de dag."
        ),
        measured=loss,
        threshold=limit,
        config_key=DAILY_LOSS_KEY,
        store=store,
        asof_ts=asof_ts,
    )


def _engage_halt(
    exposures: dict[str, float],
    state: RiskState,
    *,
    kind: ConstraintKind,
    reason: str,
    measured: float,
    threshold: float,
    config_key: str,
    store: HaltStore | None,
    asof_ts: pd.Timestamp | None,
) -> tuple[dict[str, float], list[BindingConstraint], RiskState]:
    """Zet het boek op nul en maak de HALTED-toestand duurzaam."""
    stamp = (asof_ts or pd.Timestamp.utcnow()).isoformat()
    record = HaltRecord(
        kind=kind.value,
        reason=reason,
        measured=float(measured),
        threshold=float(threshold),
        config_key=config_key,
        halted_at=stamp,
    )
    if store is not None:
        # `engage` respecteert een bestaande halt: de eerste oorzaak blijft staan.
        record = store.engage(record)

    gross = float(sum(abs(v) for v in exposures.values()))
    bound = [
        BindingConstraint(
            kind=kind,
            scope=BOOK_SCOPE,
            measured=float(measured),
            threshold=float(threshold),
            exposure_before=gross,
            exposure_after=0.0,
            config_key=config_key,
        )
    ]
    halted_state = RiskState(
        equity=float(state.equity),
        high_water_mark=float(state.high_water_mark),
        day_start_equity=float(state.day_start_equity),
        halted=True,
        halt_reason=record.reason,
        halted_at=record.halted_at,
    )
    return {s: 0.0 for s in exposures}, bound, halted_state
