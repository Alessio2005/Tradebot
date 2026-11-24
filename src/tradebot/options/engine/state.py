"""Staat (JSON, schema-versie 1) en journal (JSONL, append-only), detailontwerp 11.

* De staat wordt atomisch geschreven: naar `state.json.tmp`, `fsync`, dan `os.replace`. Een
  onderbroken schrijfactie laat de oude staat leesbaar.
* `mode` en `account_hash` moeten bij het laden gelijk zijn aan wat de dienst verwacht: een
  paper-staat wordt nooit live gebruikt (en omgekeerd).
* Het journal bevat nooit geheimen of het volledige account-id: alleen de hash.
"""
from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

from ...utils.failfast import DataContractError, require
from ..contracts import OptionContract, Right
from ..positions import Leg, LegIntent, PositionStatus, StrategyPosition

__all__ = ["SCHEMA_VERSION", "EngineState", "Journal", "contract_from", "contract_record",
           "intents_from", "intents_record", "load_state", "save_state", "to_record"]

SCHEMA_VERSION = 1


@dataclass(slots=True)
class EngineState:
    mode: str
    account_hash: str
    created_at: str
    positions: list[StrategyPosition] = field(default_factory=list)
    orders: dict[str, dict[str, Any]] = field(default_factory=dict)
    attempts: dict[str, int] = field(default_factory=dict)
    iv_history: dict[str, list[tuple[str, float]]] = field(default_factory=dict)
    equity: list[tuple[str, float]] = field(default_factory=list)
    high_water_mark: float = 0.0
    day_trades: list[str] = field(default_factory=list)
    wheel: dict[str, str] = field(default_factory=dict)
    cost_basis: dict[str, float] = field(default_factory=dict)
    new_positions: dict[str, int] = field(default_factory=dict)   # datum → aantal

    def open_positions(self) -> list[StrategyPosition]:
        return [p for p in self.positions if p.status is not PositionStatus.CLOSED]

    def day_trade_dates(self) -> tuple[date, ...]:
        return tuple(date.fromisoformat(d) for d in self.day_trades)


def contract_record(c: OptionContract) -> dict[str, Any]:
    return {"underlying": c.underlying, "expiry": c.expiry.isoformat(), "strike": c.strike,
            "right": c.right.value}


def contract_from(c: Mapping[str, Any]) -> OptionContract:
    return OptionContract(c["underlying"], date.fromisoformat(c["expiry"]), float(c["strike"]),
                          Right(c["right"]))


def intents_record(legs: tuple[LegIntent, ...]) -> list[list[Any]]:
    return [[contract_record(leg.contract), leg.ratio] for leg in legs]


def intents_from(rows: list[list[Any]]) -> tuple[LegIntent, ...]:
    return tuple(LegIntent(contract_from(c), int(r)) for c, r in rows)


def _leg(leg: Leg) -> dict[str, Any]:
    c = leg.contract
    return {"symbol": leg.symbol, "qty": leg.qty, "entry_price": leg.entry_price,
            "contract": None if c is None else contract_record(c)}


def _unleg(raw: Mapping[str, Any]) -> Leg:
    c = raw["contract"]
    return Leg(None if c is None else contract_from(c), raw["symbol"], int(raw["qty"]),
               float(raw["entry_price"]))


def _pos(p: StrategyPosition) -> dict[str, Any]:
    return {"position_id": p.position_id, "strategy": p.strategy, "underlying": p.underlying,
            "legs": [_leg(x) for x in p.legs], "opened_at": p.opened_at.isoformat(),
            "units": p.units, "entry_net": p.entry_net,
            "max_loss": None if p.max_loss == float("inf") else p.max_loss,
            "adjustments": p.adjustments, "phase": p.phase, "variant": p.variant,
            "status": p.status.value}


def _unpos(r: Mapping[str, Any]) -> StrategyPosition:
    return StrategyPosition(r["position_id"], r["strategy"], r["underlying"],
                            tuple(_unleg(x) for x in r["legs"]),
                            datetime.fromisoformat(r["opened_at"]), int(r["units"]),
                            float(r["entry_net"]),
                            float("inf") if r["max_loss"] is None else float(r["max_loss"]),
                            int(r["adjustments"]), r["phase"], r["variant"],
                            PositionStatus(r["status"]))


def to_record(s: EngineState) -> dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION, "mode": s.mode, "account_hash": s.account_hash,
            "created_at": s.created_at, "positions": [_pos(p) for p in s.positions],
            "orders": s.orders, "attempts": s.attempts,
            "iv_history": {k: [list(x) for x in v] for k, v in s.iv_history.items()},
            "equity": [list(x) for x in s.equity], "high_water_mark": s.high_water_mark,
            "day_trades": s.day_trades, "wheel": s.wheel, "cost_basis": s.cost_basis,
            "new_positions": s.new_positions}


def save_state(s: EngineState, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    data = json.dumps(to_record(s), indent=1, sort_keys=True)
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_state(path: Path, mode: str, account_hash: str) -> EngineState | None:
    """De staat, of None als er nog geen is. Een andere mode of account is een fout."""
    if not path.is_file():
        return None
    r = json.loads(path.read_text(encoding="utf-8"))
    require(r.get("schema_version") == SCHEMA_VERSION, "onbekende schema-versie van de staat",
            DataContractError, found=r.get("schema_version"))
    require(r["mode"] == mode, "de staat hoort bij een andere mode", DataContractError,
            state=r["mode"], expected=mode)
    require(r["account_hash"] == account_hash, "de staat hoort bij een ander account",
            DataContractError)
    return EngineState(r["mode"], r["account_hash"], r["created_at"],
                       [_unpos(p) for p in r["positions"]], dict(r["orders"]),
                       {k: int(v) for k, v in r["attempts"].items()},
                       {k: [(d, float(v)) for d, v in rows]
                        for k, rows in r["iv_history"].items()},
                       [(d, float(v)) for d, v in r["equity"]], float(r["high_water_mark"]),
                       list(r["day_trades"]), dict(r["wheel"]),
                       {k: float(v) for k, v in r["cost_basis"].items()},
                       {k: int(v) for k, v in r["new_positions"].items()})


class Journal:
    """Eén JSON-regel per gebeurtenis; alleen toevoegen."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, ts: datetime, event: str, **fields: Any) -> None:
        row = {"ts": ts.isoformat(), "event": event, **fields}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True, default=str) + "\n")

    def read(self) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines()]
