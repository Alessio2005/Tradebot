"""Het contract van de wekelijkse meta-label-strategie. Zie `conf/model/weekly_meta.yaml`.

Apart van `schemas/config.py`, dat al te groot is: dit domein hoort bij één
strategie en verandert met haar mee.
"""
from __future__ import annotations

from pathlib import Path
from typing import Annotated

import pandas as pd
from pydantic import Field, model_validator

from .config import StrictModel, load_config

__all__ = ["WEEKLY_META_CONFIG_PATH", "WeeklyMetaConfig", "weekly_meta_config"]

WEEKLY_META_CONFIG_PATH = (
    Path(__file__).resolve().parents[3] / "conf" / "model" / "weekly_meta.yaml")

Positive = Annotated[float, Field(gt=0.0)]
Fraction = Annotated[float, Field(gt=0.0, lt=1.0)]


class WeeklyMetaConfig(StrictModel):
    """Alle instellingen van de strategie. Niets hiervan wordt gezocht."""

    symbols: tuple[str, ...]
    first_test_start_utc: str
    holdout_split_utc: str
    test_bars: Annotated[int, Field(ge=28)]
    events_per_week_target: Positive
    k_grid: tuple[Positive, ...]
    #: Eén breedte voor winst- en verliesbarrière: 1:1 is een eigenschap van het
    #: schema, niet van twee getallen die toevallig gelijk staan.
    barrier_sigma: Positive
    horizon_bars: Annotated[int, Field(ge=1)]
    stop_slippage_bps: Annotated[float, Field(ge=0.0)]
    trades_per_week_target: Positive
    logreg_c: Positive
    forest_n_estimators: Annotated[int, Field(ge=10)]
    forest_max_depth: Annotated[int, Field(ge=1, le=5)]
    forest_min_samples_leaf: Annotated[int, Field(ge=1)]
    forest_max_features: Annotated[int, Field(ge=1)]
    calibration_fraction: Fraction
    kelly_multiple: Annotated[float, Field(gt=0.0, le=0.5)]
    posterior_quantile: Fraction
    n_probability_bins: Annotated[int, Field(ge=2)]
    baseline_risk_fraction: Fraction
    resize_band: Annotated[float, Field(ge=0.0, lt=1.0)]
    corr_window: Annotated[int, Field(ge=20)]
    account_equity: Positive
    impact_eta: Positive
    impact_eta_low: Positive
    impact_eta_high: Positive
    mc_paths: Annotated[int, Field(ge=1000)]
    mc_max_drawdown: Fraction
    mc_max_probability_1y: Fraction
    cpcv_n_groups: Annotated[int, Field(ge=2)]
    cpcv_n_test_groups: Annotated[int, Field(ge=1)]
    pbo_max: Fraction
    min_trades: Annotated[int, Field(ge=1)]
    n_shuffle_replicates: Annotated[int, Field(ge=1)]
    shuffle_auc_band: tuple[float, float]
    holdout_brier_margin: Annotated[float, Field(ge=0.0)]
    planned_trials: Annotated[int, Field(ge=2)]
    seed: Annotated[int, Field(ge=0)]

    @model_validator(mode="after")
    def _consistent(self) -> WeeklyMetaConfig:
        first = pd.Timestamp(self.first_test_start_utc)
        split = pd.Timestamp(self.holdout_split_utc)
        if first.tzinfo is None or split.tzinfo is None:
            raise ValueError("first_test_start_utc en holdout_split_utc moeten een tijdzone dragen")
        if not first < split:
            raise ValueError("de eerste testperiode moet voor de holdout-split liggen")
        if list(self.k_grid) != sorted(set(self.k_grid)):
            raise ValueError("k_grid moet strikt oplopend zijn")
        if self.trades_per_week_target >= self.events_per_week_target:
            raise ValueError("er moeten meer events dan trades zijn: het filter kiest")
        if self.cpcv_n_groups % self.cpcv_n_test_groups != 0:
            raise ValueError("cpcv_n_groups moet deelbaar zijn door cpcv_n_test_groups")
        lo, hi = self.shuffle_auc_band
        if not 0.0 < lo < 0.5 < hi < 1.0:
            raise ValueError("shuffle_auc_band moet 0,5 omsluiten")
        return self


def weekly_meta_config(path: Path | str = WEEKLY_META_CONFIG_PATH) -> WeeklyMetaConfig:
    """Laad en valideer `conf/model/weekly_meta.yaml` (plat document, geen wrapper)."""
    return load_config(path, WeeklyMetaConfig)
