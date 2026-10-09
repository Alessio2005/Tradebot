"""Het contract van robuust boek v2 (breedte). Zie `conf/model/robust_book_v2.yaml`."""
from __future__ import annotations

from itertools import pairwise
from pathlib import Path
from typing import Annotated

import pandas as pd
from pydantic import Field, model_validator

from .config import StrictModel, load_config

__all__ = ["ROBUST_BOOK_V2_CONFIG_PATH", "ROBUST_BOOK_V3_CONFIG_PATH",
           "ROBUST_BOOK_V5_CONFIG_PATH", "ROBUST_BOOK_V6_CONFIG_PATH",
           "ROBUST_BOOK_V7_CONFIG_PATH", "ROBUST_BOOK_V8_CONFIG_PATH", "RobustBookV2Config",
           "RobustBookV3Config", "RobustBookV5Config", "RobustBookV6Config",
           "RobustBookV7Config", "RobustBookV8Config", "robust_book_v2_config",
           "robust_book_v3_config", "robust_book_v5_config", "robust_book_v6_config",
           "robust_book_v7_config", "robust_book_v8_config"]

ROBUST_BOOK_V2_CONFIG_PATH = (
    Path(__file__).resolve().parents[3] / "conf" / "model" / "robust_book_v2.yaml")
ROBUST_BOOK_V3_CONFIG_PATH = (
    Path(__file__).resolve().parents[3] / "conf" / "model" / "robust_book_v3.yaml")
ROBUST_BOOK_V5_CONFIG_PATH = (
    Path(__file__).resolve().parents[3] / "conf" / "model" / "robust_book_v5.yaml")
ROBUST_BOOK_V6_CONFIG_PATH = (
    Path(__file__).resolve().parents[3] / "conf" / "model" / "robust_book_v6.yaml")
ROBUST_BOOK_V7_CONFIG_PATH = (
    Path(__file__).resolve().parents[3] / "conf" / "model" / "robust_book_v7.yaml")
ROBUST_BOOK_V8_CONFIG_PATH = (
    Path(__file__).resolve().parents[3] / "conf" / "model" / "robust_book_v8.yaml")

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


Fraction = Annotated[float, Field(gt=0.0, le=1.0)]


class Basis(StrictModel):
    """De spot-perp-basiscarry (v5): long spot, short perp, de funding oogsten."""

    funding_span: PositiveInt
    enter_apr: Positive
    exit_apr: float
    slots: PositiveInt
    notional: Fraction
    band: Fraction
    hedge_tolerance: Fraction
    min_spot_adv_usd: Positive
    max_abs_basis: Fraction
    spot_taker_fee_bps: Positive
    spot_half_spread_bps: Positive
    spot_half_spread_stress_bps: Positive
    maintenance_margin: Fraction
    margin_floor: Fraction
    majors: tuple[str, ...]
    combo_trend_share: Fraction

    @model_validator(mode="after")
    def _consistent(self) -> Basis:
        if not self.exit_apr < self.enter_apr:
            raise ValueError("exit_apr moet onder enter_apr liggen (hysterese)")
        if not (1.0 - self.notional) >= self.margin_floor * self.notional:
            raise ValueError("bij de volle notional ligt de futureswallet al onder de "
                             "margevloer: het boek zou elke dag herbalanceren")
        return self


class Backcast(StrictModel):
    start: str
    end: str

    @model_validator(mode="after")
    def _ordered(self) -> Backcast:
        if not pd.Timestamp(self.start, tz="UTC") < pd.Timestamp(self.end, tz="UTC"):
            raise ValueError("backcast.start moet voor backcast.end liggen")
        return self


class RobustBookV5Config(RobustBookV3Config):
    """v3 plus `basis` (de nieuwe sleeve) en `backcast` (het ongemeten sample van 2020)."""

    basis: Basis
    backcast: Backcast

    @model_validator(mode="after")
    def _backcast_before_dev(self) -> RobustBookV5Config:
        if not (pd.Timestamp(self.backcast.end, tz="UTC")
                < pd.Timestamp(self.windows.train_start, tz="UTC")):
            raise ValueError("de backcast moet volledig voor W_DEV liggen")
        return self


def robust_book_v5_config(path: Path | str = ROBUST_BOOK_V5_CONFIG_PATH) -> RobustBookV5Config:
    return load_config(path, RobustBookV5Config)


Rate = Annotated[float, Field(ge=0.0, lt=1.0)]


class Leverage(StrictModel):
    """Het hefboomboek (v6): unified margin, financiering, haircut, liquidatie, governor."""

    #: Kandidaat -> notional per been als veelvoud van de equity.
    candidates: dict[str, Positive]
    majors: tuple[str, ...]
    haircut_major: Rate
    haircut_alt: Rate
    haircut_alt_stress: Rate
    maintenance_margin: Rate
    maintenance_margin_stress: Rate
    loan_maintenance: Rate
    min_uni_mmr: Annotated[float, Field(ge=1.0)]
    gross_cap: Positive
    liquidation_fee: Rate
    adv_participation_cap: Annotated[float, Field(gt=0.0, le=1.0)]
    financing_floor_apr: Rate
    financing_multiplier: Positive
    financing_stress_floor_apr: Rate
    financing_stress_multiplier: Positive
    financing_optimistic_apr: Rate

    @model_validator(mode="after")
    def _consistent(self) -> Leverage:
        if not self.candidates:
            raise ValueError("geen hefboomkandidaten")
        if max(self.candidates.values()) * 2.0 > self.gross_cap:
            raise ValueError("een kandidaat vraagt meer bruto exposure (beide benen) dan de cap")
        if not (self.haircut_alt_stress > self.haircut_alt
                and self.maintenance_margin_stress > self.maintenance_margin):
            raise ValueError("de margestress moet strenger zijn dan de basis")
        if not (self.financing_stress_floor_apr >= self.financing_floor_apr
                and self.financing_stress_multiplier >= self.financing_multiplier):
            raise ValueError("de financieringsstress moet duurder zijn dan de basis")
        return self


class Forward(StrictModel):
    """Het vooruit-sample: data die bij het bevriezen nog niet bestond."""

    start: str
    min_months: PositiveInt


class RobustBookV6Config(RobustBookV5Config):
    """v5 plus `leverage` (het hefboomboek) en `forward` (het vooruit-sample)."""

    leverage: Leverage
    forward: Forward

    @model_validator(mode="after")
    def _forward_after_holdout(self) -> RobustBookV6Config:
        if not (pd.Timestamp(self.forward.start, tz="UTC")
                > pd.Timestamp(self.windows.holdout_end, tz="UTC")):
            raise ValueError("het vooruit-sample moet na de holdout beginnen")
        return self


def robust_book_v6_config(path: Path | str = ROBUST_BOOK_V6_CONFIG_PATH) -> RobustBookV6Config:
    return load_config(path, RobustBookV6Config)


class V7(StrictModel):
    """Uitvoering en financiering (v7): gespreide uitvoering en de gefinancierde tranche."""

    #: Elke tranche (1 / slots) gaat in zoveel gelijke dagstappen in of uit de markt.
    slice_days: PositiveInt
    #: De gefinancierde tranche gaat aan bij carry >= leenrente + deze spread.
    lever_spread_apr: Rate
    #: De kandidaten met een gefinancierde tranche (hun hefboom per been is 2,0).
    tranche: tuple[str, ...]


class RobustBookV7Config(RobustBookV6Config):
    """v6 plus `v7`."""

    v7: V7

    @model_validator(mode="after")
    def _tranche_consistent(self) -> RobustBookV7Config:
        unknown = set(self.v7.tranche) - set(self.leverage.candidates)
        if unknown:
            raise ValueError(f"tranche-kandidaten zonder hefboomregel: {sorted(unknown)}")
        for name, lev in self.leverage.candidates.items():
            want = 2.0 if name in self.v7.tranche else 1.0
            if lev != want:
                raise ValueError(f"{name}: hefboom per been {lev}, verwacht {want} "
                                 "(basis 1, plus 1 als gefinancierde tranche)")
        return self


def robust_book_v7_config(path: Path | str = ROBUST_BOOK_V7_CONFIG_PATH) -> RobustBookV7Config:
    return load_config(path, RobustBookV7Config)


class V8(StrictModel):
    """De sprongrisicogrens (v8): geen carry in een munt met een dag-σ boven deze waarde."""

    max_daily_sigma: Annotated[float, Field(gt=0.0, lt=1.0)]


class RobustBookV8Config(RobustBookV7Config):
    """v7 plus `v8`."""

    v8: V8


def robust_book_v8_config(path: Path | str = ROBUST_BOOK_V8_CONFIG_PATH) -> RobustBookV8Config:
    return load_config(path, RobustBookV8Config)
