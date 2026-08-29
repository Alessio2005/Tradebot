# src/tradebot/registry/hypothesis_ledger.py
"""Append-only hypothesis ledger — the honest cumulative DSR trial counter.

Mandate v3 §11: every tested config (model variant, Optuna trial, sweep cell,
harvest config) increments ``total_n_hypotheses``, persistently and
cross-wave. DSR must always be deflated by this cumulative count, never by a
wave-local one (the Wave-19 lesson: local counts flatter you).

Concurrency model (STAPPENPLAN v3 §0): one writer on the main ledger.
Parallel waves append to their own *staging* file
(``hypothesis_ledger_staging_<wave>.json``) and are merged serially at wave
close. Writes are atomic (tmp file + ``os.replace``).

The ledger is append-only by construction: this module exposes no delete or
edit operation, and ``append`` refuses to shrink the entry list.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from tradebot.utils.hashing import hash_config
from tradebot.utils.time import now_utc

__all__ = [
    "LedgerEntry",
    "HypothesisLedger",
    "DEFAULT_LEDGER_PATH",
    "PROVENANCE_FIELDS",
]

DEFAULT_LEDGER_PATH = Path("artefacts/governance/hypothesis_ledger.json")

_VALID_RESULTS = frozenset({"accepted", "archived", "falsified", "interim"})
_VALID_MARKETS = frozenset(
    {"crypto", "equities", "fx", "commodities", "rates", "book"}
)


#: De vier herkomstvelden die audit §26 op elke ledger-entry eist.
#:
#: PHASE 7/8 STAGE B-8. Zij bestonden hiervoor niet als veld. De herkomst werd
#: als vrije tekst in `notes` gepropt:
#:
#:     notes="Baseline-resultaat; preregistration_id=56395fa2...; git_sha=42555d2"
#:
#: Dat is geen contract maar een gewoonte. Er was geen manier om een entry
#: ZONDER herkomst te weigeren, en dus geen manier om te weten of een entry uit
#: een reproduceerbare run kwam. De ledger telt `M` voor elke DSR in dit
#: platform; een entry die niet herleidbaar is tot code, data en configuratie,
#: maakt die telling een bewering.
PROVENANCE_FIELDS = ("git_sha", "config_hash", "data_hash", "preregistration_id")


@dataclass(frozen=True)
class LedgerEntry:
    """One immutable ledger row. ``n_trials`` is the number of distinct
    configs this entry accounts for (>= 1).

    Elk van de vier velden in `PROVENANCE_FIELDS` is VERPLICHT en niet-leeg. Er
    is geen default en geen `unknown`-waarde: een entry die niet te reproduceren
    is, hoort niet in een teller te belanden die promotiebesluiten draagt.
    """

    wave: int
    unit: str
    market: str
    config_hash: str
    n_trials: int
    result: str
    git_sha: str = ""
    data_hash: str = ""
    preregistration_id: str = ""
    ts_utc: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    #: De `config_hash` van de entry die deze entry AMENDEERT. Een amendement
    #: verandert het OORDEEL over onderzoek dat al is geteld, en telt daarom
    #: zelf nul trials.
    #:
    #: Waarom dit veld bestaat, gemeten op 2026-08-29: de 48 trials van H1 zijn
    #: bij het BEVRIEZEN van de pre-registratie al in `M` geboekt — terecht,
    #: want wie een parameterruimte vastlegt, heeft die kansen genomen. Het
    #: oordeel daarna nogmaals als 48 trials boeken zou `M` van 2.776 naar
    #: 2.824 brengen voor onderzoek dat één keer is gedaan. Ondertellen maakt
    #: elke DSR erna te gunstig; dubbeltellen maakt hem te streng. Beide zijn
    #: onwaar, en een append-only ledger heeft dus een manier nodig om een
    #: oordeel te herzien zonder de telling te raken.
    amends: str = ""

    def __post_init__(self) -> None:
        if self.amends:
            if self.n_trials != 0:
                raise ValueError(
                    f"Een amendement op {self.amends!r} draagt "
                    f"n_trials={self.n_trials}. Een amendement herziet een "
                    "OORDEEL over trials die al zijn geteld; trials meebrengen "
                    "maakt er een nieuwe zoektocht van met een etiket dat het "
                    "tegenovergestelde zegt. Zet n_trials=0, of boek een "
                    "gewone entry."
                )
        elif self.n_trials < 1:
            raise ValueError(f"n_trials must be >= 1, got {self.n_trials}")
        if self.result not in _VALID_RESULTS:
            raise ValueError(
                f"result {self.result!r} not in {sorted(_VALID_RESULTS)}"
            )
        if self.market not in _VALID_MARKETS:
            raise ValueError(
                f"market {self.market!r} not in {sorted(_VALID_MARKETS)}"
            )
        # De velden hebben een lege default zodat de FOUTMELDING alle vier de
        # ontbrekende velden in één keer noemt. Een TypeError op het eerste
        # ontbrekende keyword zou de aanroeper vier keer laten raden.
        missing = [
            name for name in PROVENANCE_FIELDS
            if not str(getattr(self, name)).strip()
        ]
        if missing:
            raise ValueError(
                f"LedgerEntry({self.unit!r}) mist verplichte herkomstvelden: "
                f"{missing}. Audit §26: een ledger-entry zonder resolvable "
                f"herkomst is INVALIDE — hij telt mee in M, en dus in elke DSR, "
                f"zonder dat iemand kan nagaan waarop hij is gebaseerd. Vul ze "
                f"met de werkelijke waarden; er is geen 'unknown'."
            )
        if not self.ts_utc:
            object.__setattr__(self, "ts_utc", now_utc().isoformat())

    @classmethod
    def from_config(
        cls,
        wave: int,
        unit: str,
        market: str,
        config: dict[str, Any],
        git_sha: str,
        data_hash: str,
        preregistration_id: str,
        n_trials: int = 1,
        result: str = "interim",
        metrics: dict[str, Any] | None = None,
        notes: str = "",
        amends: str = "",
    ) -> LedgerEntry:
        """Build an entry, deriving ``config_hash`` deterministically.

        `git_sha`, `data_hash` en `preregistration_id` zijn VERPLICHT en staan
        vóór de argumenten met een default: wie deze constructor gebruikt, kan
        de herkomst niet vergeten mee te geven.
        """
        return cls(
            wave=wave,
            unit=unit,
            market=market,
            config_hash=hash_config(config),
            n_trials=n_trials,
            result=result,
            git_sha=git_sha,
            data_hash=data_hash,
            preregistration_id=preregistration_id,
            metrics=metrics or {},
            notes=notes,
            amends=amends,
        )


class HypothesisLedger:
    """Atomic, append-only access to the hypothesis ledger JSON file."""

    def __init__(self, path: Path | str = DEFAULT_LEDGER_PATH) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(
                f"Ledger not found at {self.path}. The ledger is seeded "
                "manually (Wave 20, step 0.1) — never auto-created, so the "
                "seed reconstruction is always deliberate."
            )

    # -- reads ------------------------------------------------------------

    def load(self) -> dict[str, Any]:
        with open(self.path, encoding="utf-8") as fh:
            return json.load(fh)

    def total_n_hypotheses(self) -> int:
        """The honest cumulative trial count: seed + all appended trials."""
        doc = self.load()
        return int(doc["seed_total"]) + sum(
            int(e["n_trials"]) for e in doc["entries"]
        )

    def entries(self) -> list[dict[str, Any]]:
        return list(self.load()["entries"])

    # -- writes (append-only) ----------------------------------------------

    def append(self, entry: LedgerEntry) -> int:
        """Atomically append one entry; returns the new cumulative total.

        Een amendement (`entry.amends`) moet naar een BESTAANDE entry wijzen.
        Wijst het naar niets, dan hangt het oordeel aan niets en is niet na te
        gaan wát er is herzien.
        """
        doc = self.load()
        if entry.amends:
            known = {e["config_hash"] for e in doc["entries"]}
            if entry.amends not in known:
                raise ValueError(
                    f"Amendement wijst naar config_hash {entry.amends!r}, en "
                    f"die staat niet in deze ledger. Een oordeel over "
                    f"onderzoek dat hier niet is geboekt, hoort hier ook niet "
                    f"thuis."
                )
        doc["entries"].append(asdict(entry))
        self._write_atomic(doc)
        return int(doc["seed_total"]) + sum(
            int(e["n_trials"]) for e in doc["entries"]
        )

    def merge_staging(self, staging_path: Path | str) -> int:
        """Serially merge a parallel wave's staging file, then delete it.

        Staging format: JSON list of entry dicts. Duplicate
        (wave, unit, config_hash) triples already in the ledger are
        rejected — a config is only ever counted once.
        """
        staging_path = Path(staging_path)
        with open(staging_path, encoding="utf-8") as fh:
            staged = json.load(fh)
        if not isinstance(staged, list):
            raise ValueError(f"{staging_path} must contain a JSON list")

        doc = self.load()
        seen = {
            (e["wave"], e["unit"], e["config_hash"]) for e in doc["entries"]
        }
        for raw in staged:
            entry = LedgerEntry(**raw)  # validates
            key = (entry.wave, entry.unit, entry.config_hash)
            if key in seen:
                raise ValueError(
                    f"Duplicate ledger key {key} in {staging_path} — a "
                    "config is only counted once."
                )
            seen.add(key)
            doc["entries"].append(asdict(entry))
        self._write_atomic(doc)
        try:
            staging_path.unlink()
        except OSError:
            # some mounts forbid unlink — truncate instead so a re-merge
            # can never double-count
            staging_path.write_text("[]", encoding="utf-8")
        return int(doc["seed_total"]) + sum(
            int(e["n_trials"]) for e in doc["entries"]
        )

    @staticmethod
    def append_to_staging(
        staging_path: Path | str, entry: LedgerEntry
    ) -> None:
        """Append to a wave-local staging file (parallel-safe per wave)."""
        staging_path = Path(staging_path)
        staged: list[dict[str, Any]] = []
        if staging_path.exists():
            with open(staging_path, encoding="utf-8") as fh:
                staged = json.load(fh)
        staged.append(asdict(entry))
        tmp = staging_path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(staged, fh, indent=2, ensure_ascii=False)
        os.replace(tmp, staging_path)

    # -- internals ----------------------------------------------------------

    def _write_atomic(self, doc: dict[str, Any]) -> None:
        existing = self.load()
        if len(doc["entries"]) < len(existing["entries"]):
            raise ValueError(
                "Refusing to write: entry list would shrink (append-only)."
            )
        if doc["seed_total"] != existing["seed_total"]:
            raise ValueError("Refusing to write: seed_total is immutable.")
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=2, ensure_ascii=False)
        os.replace(tmp, self.path)
