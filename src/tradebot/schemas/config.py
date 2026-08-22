"""Pydantic v2 configuratiecontracten - Phase 0, deliverable 4.

Elk configuratiedomein heeft hier een model. De regels zijn overal identiek:

    model_config = ConfigDict(extra="forbid", frozen=True)

`extra="forbid"`  - een onbekende sleutel is een fout, geen aanwijzing. Een typo
                    in `conf/` mag nooit stilzwijgend een default laten winnen.
`frozen=True`     - config is onveranderlijk na laden. Een module die tijdens een
                    run een drempel bijstelt, maakt het resultaat
                    onreproduceerbaar en dus onbruikbaar als bewijs.

Elke validatiefout wordt vertaald naar `ConfigContractError`, zodat
configuratieschendingen dezelfde fail-fast-tak volgen als data- en
dependency-schendingen.

Ref: ARCHITECTUUR_AUDIT_2026-08-22.md sectie 20, sectie 26 (AC-5).
"""
from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from ..utils.failfast import ConfigContractError

__all__ = [
    "BacktestConfig",
    "DataConfig",
    "ExecutionConfig",
    "RiskConfig",
    "StrictModel",
    "TradebotConfig",
    "ValidationConfig",
    "VolatilityConfig",
    "load_config",
    "validate_mapping",
]

Fraction = Annotated[float, Field(gt=0.0, le=1.0)]
PositiveInt = Annotated[int, Field(gt=0)]


class StrictModel(BaseModel):
    """Basis voor elk configuratiedomein: verbiedt extra sleutels, is frozen."""

    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)


# --------------------------------------------------------------------------- #
# L0 - Data
# --------------------------------------------------------------------------- #
class DataConfig(StrictModel):
    """Contract voor de datalaag (L0). Zie `conf/data/`."""

    market_data_root: Path = Field(
        description="Root van de ruwe/legacy parquet-boom.")
    pit_store_root: Path = Field(
        description="Root van de append-only Point-in-Time store (Phase 1).")
    feature_store_root: Path
    artefacts_root: Path

    symbols: tuple[str, ...] = Field(min_length=1)
    granularities: tuple[str, ...] = Field(min_length=1)

    venue: Literal["bybit", "binance"] = "bybit"
    quote_currency: str = "USDT"

    # Funding-settlementsemantiek. Een funding rate die om 08:00 UTC settelt is
    # pas OM 08:00 UTC bekend - niet om 00:00 van diezelfde dag. Deze waarde
    # wordt afgedwongen in de asof_join-tolerance (Phase 1, stap 6).
    funding_interval_hours: PositiveInt = 8

    # Verplichte, expliciete tolerance voor elke asof_join. Geen default in de
    # functie zelf: de keuze hoort in de config, niet in de code.
    asof_tolerance_seconds: PositiveInt = 60

    # Hoe om te gaan met gaten. NOOIT impliciet in een ingestion-functie.
    gap_policy: Literal["reject", "register"] = "reject"

    @field_validator("symbols", "granularities")
    @classmethod
    def _no_blanks(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        if any(not s.strip() for s in v):
            raise ValueError("lege string in lijst")
        return v


# --------------------------------------------------------------------------- #
# L2 - Volatility
# --------------------------------------------------------------------------- #
class VolatilityConfig(StrictModel):
    """Contract voor de volatiliteitsengines (L2). Zie `conf/model/volatility.yaml`.

    Level 1 (EWMA/RiskMetrics) is de BASELINE. GARCH is Level 2 en komt pas in
    Phase 6 (audit sectie 9.1 / 22).
    """

    estimator: Literal["ewma"] = Field(
        default="ewma",
        description="Phase 3 staat uitsluitend de Level-1 baseline toe.")

    # RiskMetrics-conventie. Geen magisch getal in de code.
    ewma_lambda: Annotated[float, Field(gt=0.0, lt=1.0)] = 0.94

    # Aantal observaties voordat de schatting bruikbaar is. Daarvoor propageert
    # de estimator NaN - nooit een impliciete constante volatiliteit.
    burn_in_bars: PositiveInt = 60

    annualisation_factor: Annotated[float, Field(gt=0.0)] = 365.0

    min_periods: PositiveInt = 2


# --------------------------------------------------------------------------- #
# L4 - Alpha
# --------------------------------------------------------------------------- #
class AlphaConfig(StrictModel):
    """Contract voor de alpha-laag (L4). Zie `conf/model/alpha.yaml`."""

    lookback_bars: PositiveInt = 60
    skip_bars: Annotated[int, Field(ge=0)] = 1
    min_assets: PositiveInt = 3
    rebalance_every_bars: PositiveInt = 1

    # Bereik waarbinnen elke AlphaUnit-output moet vallen (sectie 11.1).
    signal_floor: float = -1.0
    signal_cap: float = 1.0

    winsorise_quantile: Annotated[float, Field(gt=0.0, lt=0.5)] = 0.02

    @field_validator("signal_cap")
    @classmethod
    def _cap_above_floor(cls, v: float, info: Any) -> float:
        floor = info.data.get("signal_floor")
        if floor is not None and v <= floor:
            raise ValueError("signal_cap moet groter zijn dan signal_floor")
        return v


# --------------------------------------------------------------------------- #
# L7/L8 - Risk & Portfolio
# --------------------------------------------------------------------------- #
class RiskConfig(StrictModel):
    """Contract voor de onafhankelijke risicolaag (L7). Zie `conf/risk/`."""

    max_position_pct: Fraction = 0.25
    max_gross_leverage: Annotated[float, Field(gt=0.0)] = 1.5
    max_net_imbalance: Fraction = 0.40
    daily_var_limit_pct: Fraction = 0.02
    max_drawdown_pct: Fraction = 0.08
    max_daily_loss_pct: Fraction = 0.03
    max_position_age_h: PositiveInt = 48
    min_signal_confidence: Fraction = 0.55
    max_funding_cost_bps_day: Annotated[float, Field(ge=0.0)] = 30.0

    allocator: Literal["equal_weight", "risk_parity"] = "risk_parity"


# --------------------------------------------------------------------------- #
# L9 - Execution
# --------------------------------------------------------------------------- #
class ExecutionConfig(StrictModel):
    """Contract voor executie & TCA (L9). Zie `conf/execution/`."""

    maker_fee_bps: Annotated[float, Field(ge=0.0)]
    taker_fee_bps: Annotated[float, Field(ge=0.0)]

    # Conservatieve, VOORLOPIGE spread-aanname. De definitieve eta-kalibratie
    # volgt in Phase 5; tot die tijd wordt elk resultaat bruto EN netto
    # gerapporteerd met deze aanname expliciet vermeld.
    assumed_half_spread_bps: Annotated[float, Field(ge=0.0)] = 1.0

    slippage_model: Literal["none", "half_spread", "square_root"] = "half_spread"
    impact_eta: Annotated[float, Field(ge=0.0)] = 0.142
    cost_assumption_is_provisional: bool = True


# --------------------------------------------------------------------------- #
# L10 - Backtest
# --------------------------------------------------------------------------- #
class BacktestConfig(StrictModel):
    """Contract voor de backtest-engine (L10)."""

    engine: Literal["vectorized", "event_driven"] = "vectorized"
    initial_equity: Annotated[float, Field(gt=0.0)] = 100_000.0
    bars_per_year: Annotated[float, Field(gt=0.0)] = 365.0
    seed: Annotated[int, Field(ge=0)] = 42

    # Vectorized resultaten zijn screening-materiaal, nooit promotiebewijs
    # (sectie 16.1). Expliciet in de config zodat een rapport het kan citeren.
    counts_as_promotion_evidence: bool = False


# --------------------------------------------------------------------------- #
# L11 - Validation
# --------------------------------------------------------------------------- #
class ValidationConfig(StrictModel):
    """Contract voor de statistische validatielaag (L11). Zie `conf/validation/`."""

    scheme: Literal["purged_walk_forward"] = "purged_walk_forward"

    n_splits: PositiveInt = 6
    train_bars: PositiveInt = 500
    test_bars: PositiveInt = 100

    # De embargo-lengte is een FUNCTIE van de labelhorizon en hoort daarom in de
    # config, niet in de code (Phase 2, stap 5).
    label_horizon_bars: PositiveInt = 1
    embargo_bars: Annotated[int, Field(ge=0)] = 5

    dsr_alpha: Annotated[float, Field(gt=0.0, lt=1.0)] = 0.05
    spa_alpha: Annotated[float, Field(gt=0.0, lt=1.0)] = 0.05
    spa_bootstrap_reps: PositiveInt = 1000
    spa_block_length: PositiveInt = 10

    # CPCV en PBO zijn OPTIONAL-diagnostiek en mogen nooit een gate zijn
    # (sectie 17.1). Vastgelegd als config zodat de gate-compositie het kan
    # afdwingen in plaats van erop te vertrouwen.
    cpcv_is_gate: Literal[False] = False
    pbo_is_gate: Literal[False] = False

    @field_validator("embargo_bars")
    @classmethod
    def _embargo_covers_horizon(cls, v: int, info: Any) -> int:
        h = info.data.get("label_horizon_bars")
        if h is not None and v < h:
            raise ValueError(
                f"embargo_bars ({v}) < label_horizon_bars ({h}): overlappende "
                f"labels lekken dan van test naar train"
            )
        return v


# --------------------------------------------------------------------------- #
# Root
# --------------------------------------------------------------------------- #
class TradebotConfig(StrictModel):
    """Volledige, gevalideerde platformconfiguratie."""

    data: DataConfig
    volatility: VolatilityConfig
    alpha: AlphaConfig
    risk: RiskConfig
    execution: ExecutionConfig
    backtest: BacktestConfig
    validation: ValidationConfig

    seed: Annotated[int, Field(ge=0)] = 42


_M = TypeVar("_M", bound=BaseModel)

DOMAIN_SCHEMAS: dict[str, type[StrictModel]] = {
    "data": DataConfig,
    "volatility": VolatilityConfig,
    "alpha": AlphaConfig,
    "risk": RiskConfig,
    "execution": ExecutionConfig,
    "backtest": BacktestConfig,
    "validation": ValidationConfig,
}


def validate_mapping(model: type[_M], mapping: dict[str, Any], *, source: str = "") -> _M:
    """Valideer `mapping` tegen `model` of crash met `ConfigContractError`.

    Pydantic's `ValidationError` wordt bewust vertaald: configuratieschendingen
    horen in dezelfde fail-fast-hierarchie als data- en dependency-schendingen,
    zodat er precies een tak is die het platform kan doen crashen.
    """
    try:
        return model.model_validate(mapping)
    except ValidationError as exc:
        where = f" in {source}" if source else ""
        details = "; ".join(
            f"{'.'.join(str(p) for p in e['loc']) or '<root>'}: {e['msg']}"
            for e in exc.errors()
        )
        raise ConfigContractError(
            f"Configuratiecontract geschonden{where} voor {model.__name__}: "
            f"{details}"
        ) from exc


def load_config(path: str | Path, model: type[_M]) -> _M:
    """Laad een YAML-bestand en valideer het tegen `model`.

    Ondersteunt zowel een plat document als een document met precies een
    top-level sleutel die gelijk is aan de domeinnaam (de Hydra-conventie in
    `conf/`), bijvoorbeeld `risk:` in `conf/risk/default.yaml`.
    """
    import yaml

    p = Path(path)
    if not p.is_file():
        raise ConfigContractError(f"Configuratiebestand bestaat niet: {p}")

    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    if raw is None:
        raise ConfigContractError(f"Configuratiebestand is leeg: {p}")
    if not isinstance(raw, dict):
        raise ConfigContractError(
            f"Configuratiebestand {p} levert {type(raw).__name__}, verwacht een mapping"
        )

    # Hydra-specifieke sleutels horen niet bij het domeinmodel.
    payload = {k: v for k, v in raw.items() if k not in ("defaults", "_self_")}

    # `risk:` / `data:` wrapper afpellen wanneer dat de enige sleutel is.
    if len(payload) == 1:
        only_key, only_val = next(iter(payload.items()))
        if isinstance(only_val, dict) and DOMAIN_SCHEMAS.get(only_key) is model:
            payload = only_val

    return validate_mapping(model, payload, source=str(p))
