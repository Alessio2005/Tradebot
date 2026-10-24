"""De staat van het carryboek tussen twee runs, als JSON op schijf.

Het boek is een toestandsmachine: de benen (`a`, `b`), het doel van een munt in
beweging (`goal`, v7), de equity en de tellers. Wie de staat verliest, verliest de
spreiding en de hysterese, en het boek handelt bij de volgende run anders dan de backtest.

* Per SYMBOOL, niet per kolom: het paneel krijgt nieuwe munten, de volgorde schuift.
* Exact: floats gaan als `repr` door JSON en komen bit-identiek terug; NaN en inf worden
  `null`.
* Atomair: eerst een tijdelijk bestand, dan `replace`. Een crash halverwege laat de
  vorige staat staan.
* Gebonden aan de config: een andere config-hash weigert. Het bevroren boek verandert
  niet stil onder een draaiende staat.
"""
from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..systematic.leverage import LeveredBook
from ..utils.failfast import DataContractError, require

__all__ = ["STATE_VERSION", "CarryState", "load_state", "restore", "save_state", "snapshot"]

STATE_VERSION = 1


def _num(x: float) -> float | None:
    return float(x) if math.isfinite(x) else None


@dataclass(frozen=True)
class CarryState:
    candidate: str
    config_hash: str
    mode: str
    last_bar: str
    equity: float
    a: Mapping[str, float]
    b: Mapping[str, float]
    goal: Mapping[str, float]
    n_forced: int
    n_liquidations: int
    n_governed: int
    min_headroom: float | None
    # Wat het boek na de laatste stap op de beurs verwacht (munten): de reconciliatie
    # van de volgende run legt de beurs hiertegen.
    expected_spot: Mapping[str, float] = field(default_factory=dict)
    expected_perp: Mapping[str, float] = field(default_factory=dict)
    halted: str | None = None

    def as_record(self) -> dict[str, Any]:
        return {"version": STATE_VERSION, "candidate": self.candidate,
                "config_hash": self.config_hash, "mode": self.mode,
                "last_bar": self.last_bar, "equity": self.equity, "a": dict(self.a),
                "b": dict(self.b), "goal": dict(self.goal), "n_forced": self.n_forced,
                "n_liquidations": self.n_liquidations, "n_governed": self.n_governed,
                "min_headroom": self.min_headroom, "expected_spot": dict(self.expected_spot),
                "expected_perp": dict(self.expected_perp), "halted": self.halted}

    def with_halt(self, reason: str) -> CarryState:
        return CarryState(**{**self._fields(), "halted": reason})

    def _fields(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def snapshot(book: LeveredBook, *, candidate: str, config_hash: str, mode: str,
             last_bar: pd.Timestamp, expected_spot: Mapping[str, float] | None = None,
             expected_perp: Mapping[str, float] | None = None) -> CarryState:
    """De staat van `book` na de bar `last_bar`. Alleen niet-nul benen en lopende doelen."""
    syms = book.symbols
    return CarryState(
        candidate=candidate, config_hash=config_hash, mode=mode,
        last_bar=pd.Timestamp(last_bar).isoformat(), equity=float(book.equity),
        a={s: float(v) for s, v in zip(syms, book.a, strict=True) if v != 0.0},
        b={s: float(v) for s, v in zip(syms, book.b, strict=True) if v != 0.0},
        goal={s: float(v) for s, v in zip(syms, book.goal, strict=True) if np.isfinite(v)},
        n_forced=int(book.n_forced), n_liquidations=int(book.n_liq),
        n_governed=int(book.n_governed), min_headroom=_num(book.min_headroom),
        expected_spot=dict(expected_spot or {}), expected_perp=dict(expected_perp or {}))


def restore(state: CarryState, book: LeveredBook, *, config_hash: str) -> None:
    """Zet `book` (vers gebouwd op het huidige paneel) in de opgeslagen staat."""
    require(state.config_hash == config_hash, "De staat hoort bij een andere config: het "
            "bevroren boek is veranderd.", DataContractError, state=state.config_hash,
            config=config_hash)
    pos = {s: i for i, s in enumerate(book.symbols)}
    missing = sorted((set(state.a) | set(state.b) | set(state.goal)) - set(pos))
    require(not missing, "De staat houdt munten die het paneel niet meer kent.",
            DataContractError, missing=missing)
    n = len(book.symbols)
    a, b, goal = np.zeros(n), np.zeros(n), np.full(n, np.nan)
    for s, v in state.a.items():
        a[pos[s]] = v
    for s, v in state.b.items():
        b[pos[s]] = v
    for s, v in state.goal.items():
        goal[pos[s]] = v
    book.a, book.b, book.goal = a, b, goal
    book.equity = float(state.equity)
    book.n_forced, book.n_liq, book.n_governed = (
        state.n_forced, state.n_liquidations, state.n_governed)
    book.min_headroom = float("inf") if state.min_headroom is None else state.min_headroom


def save_state(state: CarryState, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state.as_record(), indent=1, sort_keys=True, allow_nan=False),
                   encoding="utf-8")
    tmp.replace(path)


def load_state(path: Path) -> CarryState:
    require(path.is_file(), "Geen staat: begin met `init`.", DataContractError, path=str(path))
    raw = json.loads(path.read_text(encoding="utf-8"))
    require(raw.get("version") == STATE_VERSION, "Onbekende versie van de staat.",
            DataContractError, version=raw.get("version"))
    raw.pop("version")
    return CarryState(**raw)
