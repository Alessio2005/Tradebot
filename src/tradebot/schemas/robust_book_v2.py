"""Het contract van robuust boek v2 (breedte). Zie `conf/model/robust_book_v2.yaml`."""
from __future__ import annotations

from itertools import pairwise
from pathlib import Path
from typing import Annotated

import pandas as pd
from pydantic import Field, model_validator

from .config import StrictModel, load_config

__all__ = ["ROBUST_BOOK_V2_CONFIG_PATH", "ROBUST_BOOK_V3_CONFIG_PATH", "RobustBookV2Config",
           "RobustBookV3Config", "robust_book_v2_config", "robust_book_v3_config"]

ROBUST_BOOK_V2_CONFIG_PATH = (
    Path(__file__).resolve().parents[3] / "conf" / "model" / "robust_book_v2.yaml")
ROBUST_BOOK_V3_CONFIG_PATH = (
    Path(__file__).resolve().parents[3] / "conf" / "model" / "robust_book_v3.yaml")

Positive = Annotated[float, Field(gt=0.0)]
PositiveInt = Annotated[int, Field(ge=1)]


class Universe(StrictModel):
    top_n: PositiveInt
    min_history_bars: PositiveInt
    adv_window: PositiveInt
    min_adv_usd: Positive
    vol_window: PositiveInt
    min_ann_vol: Positive


class Windows(StrictModel):
    train_start: str
    train_end: str
    validate_start: str
    w_dev_end: str
    holdout_start: str
    holdout_end: str

    @model_validator(mode="after")
    def _ordered(self) -> Windows:
        ts = [pd.Timestamp(v, tz="UTC") for v in (
            self.train_start, self.train_end, self.validate_start, self.w_dev_end,
            self.holdout_start, self.holdout_end)]
        if not all(a < b for a, b in pairwise(ts)):
            raise ValueError("de vensters moeten strikt oplopen")
        return self


class Execution(StrictModel):
    lag_bars: PositiveInt
    aum_usd: Positive


class Costs(StrictModel):
    taker_fee_bps: Positive
    min_half_spread_bps: Positive
    max_half_spread_bps: Positive
    spread_window: PositiveInt
    impact_y: Positive
    impact_y_stress: Positive


class Sizing(StrictModel):
    vol_span: PositiveInt
    asset_vol_target: Positive
    portfolio_vol_target: Positive
    max_scale: Positive
    per_asset_cap: Positive
    gross_cap: Positive


class XsMom(StrictModel):
    lookbacks: tuple[PositiveInt, ...]
    winsor: Positive
    rebalance_bars: PositiveInt


class XsCarry(StrictModel):
    window_bars: PositiveInt
    signal_lag_bars: PositiveInt
    rebalance_bars: PositiveInt


class Trend(StrictModel):
    lookbacks: tuple[PositiveInt, ...]
    z_clip: Positive


class Combo(StrictModel):
    candidates: tuple[str, ...]
    inclusion_min_train_sharpe: float


class RobustBookV2Config(StrictModel):
    universe: Universe
    windows: Windows
    execution: Execution
    costs: Costs
    sizing: Sizing
    xsmom: XsMom
    xscarry: XsCarry
    trend: Trend
    combo: Combo
    planned_trials: PositiveInt
    known_prior_trials: Annotated[int, Field(ge=0)]


def robust_book_v2_config(path: Path | str = ROBUST_BOOK_V2_CONFIG_PATH) -> RobustBookV2Config:
    return load_config(path, RobustBookV2Config)


class V3(StrictModel):
    half_spread_bps: Positive
    half_spread_stress_bps: Positive
    trade_rate: Annotated[float, Field(gt=0.0, le=1.0)]
    trade_rate_family: tuple[Annotated[float, Field(gt=0.0, le=1.0)], ...]


class RobustBookV3Config(RobustBookV2Config):
    """v2 plus het `v3`-blok: gecorrigeerde spread en partiële aanpassing."""

    v3: V3


def robust_book_v3_config(path: Path | str = ROBUST_BOOK_V3_CONFIG_PATH) -> RobustBookV3Config:
    return load_config(path, RobustBookV3Config)
