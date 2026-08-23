"""Continuïteit: futures rolls, delistings en symbol renames — deliverable 7d.

DOCTRINE (Phase 1, stap 8)
--------------------------
> *"Bouw de futures-roll-correctie als een **expliciete adjustment factor
> ledger**, niet als een stilzwijgende backward-ratio-aanpassing. De ledger is
> inspecteerbaar; een stilzwijgende aanpassing is dat niet."*

Een backward-ratio-aanpassing die direct op de prijsreeks wordt toegepast is
onzichtbaar: je kunt achteraf niet meer vaststellen wélke correctie is
toegepast, wanneer, en met welke factor. Deze module bewaart daarom de factoren
apart en past ze pas op verzoek toe, zodat de ruwe reeks intact blijft.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ...utils.failfast import DataContractError, require

__all__ = [
    "AdjustmentFactor",
    "AdjustmentLedger",
    "SymbolLifecycle",
    "apply_adjustments",
    "validate_continuity",
]


@dataclass(frozen=True)
class AdjustmentFactor:
    """Eén expliciete, inspecteerbare prijs-aanpassing."""

    symbol: str
    #: moment waarop de aanpassing van kracht wordt
    effective_ts_ns: int
    #: multiplicatieve factor op alle prijzen VOOR dit moment
    factor: float
    #: "roll" | "split" | "rename" | "redenomination"
    reason: str
    note: str = ""

    @property
    def effective_iso(self) -> str:
        return str(pd.Timestamp(self.effective_ts_ns, unit="ns", tz="UTC"))


@dataclass(frozen=True)
class SymbolLifecycle:
    """Levensloop van één symbool — de basis voor survivorship-correctie."""

    symbol: str
    listed_ts_ns: int
    #: None = nog actief
    delisted_ts_ns: int | None = None
    renamed_to: str | None = None

    @property
    def is_active(self) -> bool:
        return self.delisted_ts_ns is None


class AdjustmentLedger:
    """Append-only, inspecteerbaar register van prijs-aanpassingen."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def append(self, factors: list[AdjustmentFactor]) -> None:
        if not factors:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            for f in factors:
                row = asdict(f)
                row["effective_iso"] = f.effective_iso
                fh.write(json.dumps(row, sort_keys=True) + "\n")

    def read_all(self) -> list[AdjustmentFactor]:
        if not self.path.is_file():
            return []
        out: list[AdjustmentFactor] = []
        for ln in self.path.read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            d = json.loads(ln)
            d.pop("effective_iso", None)
            out.append(AdjustmentFactor(**d))
        return out

    def for_symbol(self, symbol: str) -> list[AdjustmentFactor]:
        return sorted((f for f in self.read_all() if f.symbol == symbol),
                      key=lambda f: f.effective_ts_ns)


def apply_adjustments(
    df: pd.DataFrame,
    factors: list[AdjustmentFactor],
    *,
    price_columns: tuple[str, ...] = ("open", "high", "low", "close"),
) -> pd.DataFrame:
    """Pas de ledger-factoren toe op een KOPIE van `df`.

    De ruwe reeks blijft ongemoeid; het resultaat is een nieuwe frame. Welke
    factoren zijn toegepast blijft afleidbaar uit de ledger, wat bij een
    in-place backward-ratio-aanpassing onmogelijk zou zijn.
    """
    require("event_ts_ns" in df.columns,
            "apply_adjustments vereist een event_ts_ns-kolom.",
            DataContractError, columns=list(df.columns)[:15])
    if not factors:
        return df.copy()

    out = df.copy()
    ev = out["event_ts_ns"].to_numpy()
    for f in sorted(factors, key=lambda x: x.effective_ts_ns):
        require(f.factor > 0,
                "Een adjustment factor moet strikt positief zijn; een factor "
                "van nul of lager maakt de prijsreeks betekenisloos.",
                DataContractError, symbol=f.symbol, factor=f.factor)
        mask = ev < f.effective_ts_ns
        for col in price_columns:
            if col in out.columns:
                out.loc[mask, col] = out.loc[mask, col] * f.factor
    return out


def validate_continuity(
    df: pd.DataFrame,
    *,
    symbol: str,
    lifecycle: SymbolLifecycle | None = None,
    max_abs_log_return: float,
    ledger: AdjustmentLedger | None = None,
) -> list[AdjustmentFactor]:
    """Detecteer discontinuïteiten die op een roll of split wijzen.

    Retourneert de *kandidaat*-factoren. Ze worden NIET automatisch toegepast:
    een sprong kan een echte marktbeweging zijn. Een kandidaat wordt pas een
    correctie wanneer hij bewust aan de ledger wordt toegevoegd.

    Raiset wanneer de reeks buiten het levensloop-venster van het symbool valt —
    data ná een delisting is per definitie survivorship-corrupt.
    """
    require("close" in df.columns,
            "validate_continuity vereist een close-kolom.",
            DataContractError, columns=list(df.columns)[:15])

    if lifecycle is not None:
        ev = df["event_ts_ns"]
        n_early = int((ev < lifecycle.listed_ts_ns).sum())
        require(n_early == 0,
                f"{n_early} bar(s) vóór de listing van {symbol}. Data die vóór "
                f"het bestaan van het instrument ligt is per definitie fout.",
                DataContractError, symbol=symbol,
                listed=str(pd.Timestamp(lifecycle.listed_ts_ns, unit="ns", tz="UTC")))
        if lifecycle.delisted_ts_ns is not None:
            n_late = int((ev > lifecycle.delisted_ts_ns).sum())
            require(n_late == 0,
                    f"{n_late} bar(s) ná de delisting van {symbol}. Zulke bars "
                    f"zijn survivorship-corrupt: ze bestaan alleen doordat een "
                    f"latere bron het symbool opnieuw gebruikt.",
                    DataContractError, symbol=symbol)

    close = df["close"].to_numpy(dtype=float)
    if close.size < 2:
        return []

    with np.errstate(divide="ignore", invalid="ignore"):
        lr = np.log(close[1:] / close[:-1])
    lr = np.nan_to_num(lr, nan=0.0, posinf=0.0, neginf=0.0)

    ev = df["event_ts_ns"].to_numpy()
    candidates: list[AdjustmentFactor] = []
    for i, v in enumerate(lr):
        if abs(v) > max_abs_log_return:
            candidates.append(AdjustmentFactor(
                symbol=symbol,
                effective_ts_ns=int(ev[i + 1]),
                factor=float(close[i + 1] / close[i]),
                reason="roll",
                note=(f"kandidaat: |log-return| {abs(v):.4f} > drempel "
                      f"{max_abs_log_return:.4f}. NIET automatisch toegepast."),
            ))
    if ledger is not None:
        ledger.append(candidates)
    return candidates
