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
]

DEFAULT_LEDGER_PATH = Path("artefacts/governance/hypothesis_ledger.json")

_VALID_RESULTS = frozenset({"accepted", "archived", "falsified", "interim"})
_VALID_MARKETS = frozenset(
    {"crypto", "equities", "fx", "commodities", "rates", "book"}
)


@dataclass(frozen=True)
class LedgerEntry:
    """One immutable ledger row. ``n_trials`` is the number of distinct
    configs this entry accounts for (>= 1)."""

    wave: int
    unit: str
    market: str
    config_hash: str
    n_trials: int
    result: str
    ts_utc: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def __post_init__(self) -> None:
        if self.n_trials < 1:
            raise ValueError(f"n_trials must be >= 1, got {self.n_trials}")
        if self.result not in _VALID_RESULTS:
            raise ValueError(
                f"result {self.result!r} not in {sorted(_VALID_RESULTS)}"
            )
        if self.market not in _VALID_MARKETS:
            raise ValueError(
                f"market {self.market!r} not in {sorted(_VALID_MARKETS)}"
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
        n_trials: int = 1,
        result: str = "interim",
        metrics: dict[str, Any] | None = None,
        notes: str = "",
    ) -> LedgerEntry:
        """Build an entry, deriving ``config_hash`` deterministically."""
        return cls(
            wave=wave,
            unit=unit,
            market=market,
            config_hash=hash_config(config),
            n_trials=n_trials,
            result=result,
            metrics=metrics or {},
            notes=notes,
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
        """Atomically append one entry; returns the new cumulative total."""
        doc = self.load()
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
