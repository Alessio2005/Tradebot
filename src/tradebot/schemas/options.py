"""Config-contract van de options-setup (`conf/options/default.yaml`).

Elk getal dat een handelsbeslissing stuurt, staat in de config en wordt hier gevalideerd.
Een onbekende sleutel is een fout bij het laden, geen waarschuwing (`StrictModel`,
`extra="forbid"`). De secties groeien per fase mee (detailontwerp 3.1): in fase 2 zijn dat
`universe` en `pricing`.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import Field, ValidationError, model_validator

from ..utils.failfast import ConfigContractError
from .config import StrictModel

__all__ = ["DEFAULT_OPTIONS_CONFIG", "OptionsConfig", "PricingConfig", "UniverseConfig",
           "load_options_config"]

DEFAULT_OPTIONS_CONFIG = Path(__file__).resolve().parents[3] / "conf/options/default.yaml"

AmericanMethod = Literal["binomial_lr", "binomial_crr", "baw", "fd"]
Ticker = Annotated[str, Field(pattern=r"^[A-Z][A-Z.]{0,9}$")]


class UniverseConfig(StrictModel):
    """De onderliggende waarden, welke ervan als aandeel gehouden mogen worden, en beta's."""

    underlyings: tuple[Ticker, ...] = Field(min_length=1)
    share_eligible: tuple[Ticker, ...] = ()
    beta_to_spy: dict[Ticker, Annotated[float, Field(gt=0.0, le=5.0)]]
    half_days: tuple[date, ...] = ()

    @model_validator(mode="after")
    def _consistent(self) -> UniverseConfig:
        if len(set(self.underlyings)) != len(self.underlyings):
            raise ValueError("underlyings bevat dubbelen")
        if not set(self.share_eligible) <= set(self.underlyings):
            raise ValueError("share_eligible moet een deel van underlyings zijn")
        missing = set(self.underlyings) - set(self.beta_to_spy)
        if missing:
            raise ValueError(f"beta_to_spy mist {sorted(missing)}")
        return self


class PricingConfig(StrictModel):
    """Engines, roosters, bumps en de IV-solver (detailontwerp 5)."""

    scan_engine: AmericanMethod = "baw"
    decision_engine: AmericanMethod = "binomial_lr"
    binomial_steps: Annotated[int, Field(ge=11, le=5001)] = 401
    fd_time_steps: Annotated[int, Field(ge=10)] = 200
    fd_grid_points: Annotated[int, Field(ge=50)] = 400
    vega_bump_vol: Annotated[float, Field(gt=0.0, le=0.05)] = 0.01
    rho_bump_rate: Annotated[float, Field(gt=0.0, le=0.01)] = 0.0001
    iv_accuracy: Annotated[float, Field(gt=0.0, le=1e-3)] = 1.0e-6
    iv_max_evaluations: Annotated[int, Field(ge=10)] = 200
    # Onder 1 % is QuantLib numeriek onbetrouwbaar; geen aandelenoptie heeft zo'n IV.
    iv_min: Annotated[float, Field(ge=0.005)] = 1.0e-2
    iv_max: Annotated[float, Field(gt=0.0, le=10.0)] = 5.0
    risk_free_rate: Annotated[float, Field(ge=-0.05, le=0.25)]

    @model_validator(mode="after")
    def _consistent(self) -> PricingConfig:
        if self.binomial_steps % 2 == 0:
            raise ValueError("binomial_steps moet oneven zijn (Leisen-Reimer)")
        if self.iv_min >= self.iv_max:
            raise ValueError("iv_min moet kleiner zijn dan iv_max")
        return self


class OptionsConfig(StrictModel):
    universe: UniverseConfig
    pricing: PricingConfig


def load_options_config(path: Path = DEFAULT_OPTIONS_CONFIG) -> OptionsConfig:
    """Laad en valideer de options-config; elke fout is een `ConfigContractError`."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ConfigContractError(f"{path}: verwacht een mapping bovenaan")
    try:
        return OptionsConfig.model_validate(raw)
    except ValidationError as e:
        raise ConfigContractError(f"{path}: {e}") from e
