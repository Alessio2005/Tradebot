"""Het contract van het robuuste boek v1. Zie `conf/model/robust_book.yaml`.

Apart van `schemas/config.py`, net als `schemas/weekly_meta.py`: dit domein hoort bij
één onderzoeksprogramma en verandert met hem mee. Ontwerp:
`docs/superpowers/specs/2026-10-07-robust-book-design.md`.
"""
from __future__ import annotations

from itertools import pairwise
from pathlib import Path
from typing import Annotated

import pandas as pd
from pydantic import Field, model_validator

from .config import StrictModel, load_config

__all__ = ["ROBUST_BOOK_CONFIG_PATH", "RobustBookConfig", "robust_book_config"]

ROBUST_BOOK_CONFIG_PATH = (
    Path(__file__).resolve().parents[3] / "conf" / "model" / "robust_book.yaml")

Positive = Annotated[float, Field(gt=0.0)]
PositiveInt = Annotated[int, Field(ge=1)]


class Windows(StrictModel):
    w_dev_start: str
    train_end: str
    validate_start: str
    w_dev_end: str
    gate_start: str

    @model_validator(mode="after")
    def _ordered(self) -> Windows:
        ts = [pd.Timestamp(v, tz="UTC") for v in (
            self.w_dev_start, self.train_end, self.validate_start, self.w_dev_end,
            self.gate_start)]
        if not all(a < b for a, b in pairwise(ts)):
            raise ValueError("de vensters moeten strikt oplopen: "
                             "w_dev_start < train_end < validate_start < w_dev_end < gate_start")
        return self


class Execution(StrictModel):
    #: 1 = besluit op close t, het nieuwe gewicht rendeert vanaf t+1. Nul bestaat niet.
    lag_bars: PositiveInt
    aum_usd: Positive


class Sizing(StrictModel):
    vol_span: PositiveInt
    min_history_bars: PositiveInt
    asset_vol_target: Positive
    portfolio_vol_target: Positive
    max_scale: Positive
    per_asset_cap: Positive
    gross_cap: Positive


class Trend(StrictModel):
    lookbacks: tuple[PositiveInt, ...]
    z_clip: Positive


class Core(StrictModel):
    assets: tuple[str, ...]


class Carry(StrictModel):
    window_bars: PositiveInt
    signal_lag_bars: PositiveInt
    rebalance_bars: PositiveInt


class Combo(StrictModel):
    inclusion_min_train_sharpe: float


class RobustBookConfig(StrictModel):
    """Alle instellingen van het programma. Niets hiervan wordt gezocht."""

    symbols: tuple[str, ...]
    windows: Windows
    execution: Execution
    sizing: Sizing
    trend: Trend
    core: Core
    carry: Carry
    combo: Combo
    planned_trials: PositiveInt
    known_prior_trials: Annotated[int, Field(ge=0)]

    @model_validator(mode="after")
    def _consistent(self) -> RobustBookConfig:
        if len(set(self.symbols)) != len(self.symbols):
            raise ValueError("dubbel symbool in het universum")
        if not set(self.core.assets) <= set(self.symbols):
            raise ValueError("de kern bevat een symbool buiten het universum")
        if list(self.trend.lookbacks) != sorted(set(self.trend.lookbacks)):
            raise ValueError("lookbacks moeten uniek en oplopend zijn")
        return self


def robust_book_config(path: Path | str = ROBUST_BOOK_CONFIG_PATH) -> RobustBookConfig:
    """Laad en valideer `conf/model/robust_book.yaml` (plat document, geen wrapper)."""
    return load_config(path, RobustBookConfig)
