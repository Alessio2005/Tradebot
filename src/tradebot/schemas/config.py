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

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from ..utils.failfast import ConfigContractError

__all__ = [
    "BacktestConfig",
    "DataConfig",
    "ExecutionConfig",
    "FeatureConfig",
    "ImpactConfig",
    "MicrostructureFeatureConfig",
    "MomentumFeatureConfig",
    "RiskConfig",
    "StrictModel",
    "TradebotConfig",
    "ValidationConfig",
    "VolatilityConfig",
    "VolatilityFeatureConfig",
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

    # Drempel op |ln(close_t / close_{t-1})| waarboven een bar als uitschieter
    # wordt gemeld. Op crypto is een dagelijkse beweging van 30% een ECHTE
    # marktgebeurtenis, geen fout - vandaar dat de drempel hier staat en niet
    # in de code, en dat `allow_price_jumps` los configureerbaar is.
    max_abs_log_return: Annotated[float, Field(gt=0.0)] = 0.35
    allow_price_jumps: bool = True

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
# L3 - Feature Store & Causal Pipeline
# --------------------------------------------------------------------------- #
class VolatilityFeatureConfig(StrictModel):
    """Vensters en decay-factoren van de causale volatiliteitsfeatures (L3)."""

    #: Rolling realized-vol vensters in bars. `min_periods == window`: tijdens de
    #: burn-in blijft de waarde NaN. Er wordt NOOIT ingevuld - dat was DI-2.
    realized_windows: tuple[PositiveInt, ...] = Field(min_length=1)

    #: RiskMetrics-conventie, gelijk aan `VolatilityConfig.ewma_lambda`.
    ewma_lambda: Annotated[float, Field(gt=0.0, lt=1.0)]
    #: Aantal returns dat de causale opstartfase van de EWMA-variantie voedt.
    ewma_burn_in_bars: PositiveInt

    #: Venster voor de range-estimators (Parkinson, Garman-Klass).
    range_window: PositiveInt


class MomentumFeatureConfig(StrictModel):
    """Vensters van de causale momentum-features (L3)."""

    lookback_windows: tuple[PositiveInt, ...] = Field(min_length=1)
    #: Skip-periode tegen de 1-bar reversal. 0 = geen skip.
    skip_bars: Annotated[int, Field(ge=0)]
    ewma_fast_span: PositiveInt
    ewma_slow_span: PositiveInt

    @field_validator("ewma_slow_span")
    @classmethod
    def _slow_above_fast(cls, v: int, info: Any) -> int:
        fast = info.data.get("ewma_fast_span")
        if fast is not None and v <= fast:
            raise ValueError(
                f"ewma_slow_span ({v}) moet groter zijn dan ewma_fast_span ({fast}); "
                f"anders is de spread per constructie omgekeerd van teken"
            )
        return v


class MicrostructureFeatureConfig(StrictModel):
    """Vensters van de funding- en open-interest-features (L3)."""

    funding_windows: tuple[PositiveInt, ...] = Field(min_length=1)
    funding_zscore_min_periods: PositiveInt
    oi_change_windows: tuple[PositiveInt, ...] = Field(min_length=1)


class FeatureConfig(StrictModel):
    """Contract voor de feature-laag (L3). Zie `conf/features/default.yaml`.

    Phase 2, exit criterium 5: geen enkele vensterlengte, decay factor of drempel
    staat als literal in `src/tradebot/features/`. Elke waarde hier gaat mee in
    de `feature_hash`, zodat een parameterwijziging gegarandeerd een ander
    artefact oplevert.
    """

    #: `nan` is de ENIGE toegestane burn-in-politiek. Een feature die zijn
    #: burn-in vult met een sample-brede statistiek (DI-2) of met een constante
    #: is per definitie INVALID. De sleutel bestaat zodat het validatierapport
    #: hem kan citeren, niet om hem om te zetten.
    burn_in_policy: Literal["nan"] = "nan"

    #: Root van de L3 feature store. Bewust GESCHEIDEN van
    #: `DataConfig.feature_store_root`, dat naar de oudere live-featurestore in
    #: `artefacts/` wijst. Twee verschillende artefacten onder een sleutel laten
    #: vallen zou precies de verwarring opleveren die dit contract uitsluit.
    store_root: Path = Path("data/feature_store")

    annualisation_factor: Annotated[float, Field(gt=0.0)] = 365.0

    #: Minimaal aantal observaties voordat een EXPANDING statistiek wordt
    #: vrijgegeven. Dit is de causale vervanging van het DI-2-lek.
    min_expanding_periods: PositiveInt = 20

    funding_tolerance_hours: PositiveInt = 8
    open_interest_tolerance_hours: PositiveInt = 24

    volatility: VolatilityFeatureConfig
    momentum: MomentumFeatureConfig
    microstructure: MicrostructureFeatureConfig


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
class DrawdownTier(StrictModel):
    """Eén trap van de Drawdown Breaker: vanaf `drawdown` geldt `gross_multiplier`.

    Getrapte de-grossing i.p.v. een binaire schakelaar (Phase 4, deliverable 3).
    Een breaker die pas bij de eindlimiet vuurt, doet niets in het traject waar
    ingrijpen nog goedkoop is.
    """

    drawdown: Fraction
    gross_multiplier: Annotated[float, Field(ge=0.0, le=1.0)]


class RiskConfig(StrictModel):
    """Contract voor de soevereine risicolaag (L7). Zie `conf/risk/`.

    Phase 4 maakt dit de ENIGE bron van waarheid voor elke risicodrempel. Vóór
    deze fase stond `max_gross_leverage` op vijf plaatsen met vier verschillende
    waarden (`reports/phase4_entanglement_map.md` sectie 5); waar die uiteenliepen
    is hier consequent de STRENGSTE gekozen, conform de faseregel "conservatief
    bij twijfel" - de kosten van een te ruime limiet zijn asymmetrisch.
    """

    # -- L7 volatility targeting: w_t = min(max_leverage, sigma_target/sigma_hat) --
    sigma_target: Fraction = 0.08
    max_leverage: Annotated[float, Field(gt=0.0)] = 1.5

    # -- harde limieten --
    max_position_pct: Fraction = 0.25
    max_concentration: Fraction = 0.40
    max_cluster_concentration: Fraction = 0.60
    gross_cap: Annotated[float, Field(gt=0.0)] = 1.5
    net_cap: Annotated[float, Field(gt=0.0)] = 0.60
    adv_participation_cap: Fraction = 0.01

    # -- kill switches --
    drawdown_breaker_levels: tuple[DrawdownTier, ...] = ()
    max_drawdown_pct: Fraction = 0.08
    daily_loss_limit: Fraction = 0.03

    # -- overig --
    daily_var_limit_pct: Fraction = 0.02
    max_position_age_h: PositiveInt = 48

    #: Symbool -> sector/cluster-label voor de clusterlimiet. Een symbool dat
    #: hier ontbreekt terwijl de clusterlimiet bindt, is een crash en geen
    #: "overige"-emmer: een onbekend cluster maakt de limiet betekenisloos.
    clusters: dict[str, str] = Field(default_factory=dict)

    #: Toepassingsvolgorde van de limieten. Expliciet en geconfigureerd, nooit
    #: impliciet in de code-volgorde (Phase 4, stap 4). Zie docs/RISK_CONTRACT.md
    #: sectie 6 voor waarom de boekbrede caps als laatste komen.
    constraint_order: tuple[str, ...] = ()

    allocator: Literal["equal_weight", "risk_parity"] = "risk_parity"

    @field_validator("drawdown_breaker_levels")
    @classmethod
    def _tiers_are_monotone(cls, v: tuple[DrawdownTier, ...]) -> tuple[DrawdownTier, ...]:
        """Diepere drawdown mag nooit een RUIMERE multiplier krijgen."""
        for prev, nxt in zip(v, v[1:]):
            if nxt.drawdown <= prev.drawdown:
                raise ValueError(
                    "drawdown_breaker_levels moet strikt oplopen in `drawdown`; "
                    f"kreeg {prev.drawdown} gevolgd door {nxt.drawdown}."
                )
            if nxt.gross_multiplier > prev.gross_multiplier:
                raise ValueError(
                    "Een diepere drawdown mag geen ruimere gross_multiplier "
                    f"krijgen; {nxt.drawdown} geeft {nxt.gross_multiplier} "
                    f"tegen {prev.gross_multiplier} op {prev.drawdown}."
                )
        return v


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

    # PHASE 5: `impact_eta` stond hier met default 0.142 - een literatuurwaarde
    # (Bouchaud-Bonart, BTC-perps) die nooit op dit universum is gekalibreerd en
    # die bovendien door geen enkele module werd gelezen. Fase-opdracht §11
    # verbiedt exact die constructie: een ontbrekende eta is een
    # ConfigContractError en nooit een default. Het veld is verplaatst naar
    # `ImpactConfig` (conf/execution/impact.yaml), waar het GEEN default heeft
    # en verplicht een status en herkomst draagt.
    #
    # Gevolg dat de lezer moet kennen: een NIEUWE `freeze_preregistration`-run
    # levert een andere `preregistration_id` dan de Phase 3-bevriezing, omdat
    # `resolve_parameters()` deze config meehasht. Het bestaande bevroren
    # artefact is niet geraakt - dat leest zijn eigen opgeslagen JSON.

    cost_assumption_is_provisional: bool = True


class ImpactConfig(StrictModel):
    """Contract voor het marktimpactmodel (L9). Zie `conf/execution/impact.yaml`.

    GEEN VELD HEEFT EEN DEFAULT. Dat is het hele punt: fase-opdracht §11 eist dat
    een ontbrekende `eta` een `ConfigContractError` oplevert en nooit nul, een
    fallback of een stille schatting. Pydantic levert die fout hier gratis,
    omdat elk veld verplicht is.
    """

    #: `Impact = eta * sigma_daily * sqrt(order_notional / adv_notional)`.
    eta: Annotated[float, Field(gt=0.0)]

    #: De PERMANENTE fractie van de impact (Bouchaud-decompositie). Zie
    #: `execution/impact_model.py` voor waarom deze definitie hier staat en niet
    #: in de audit.
    kappa_d: Annotated[float, Field(ge=0.0, le=1.0)]

    #: `CALIBRATED` of `IMPACT_UNCALIBRATED`. Reist mee naar elk rapport.
    status: Literal["CALIBRATED", "IMPACT_UNCALIBRATED"]

    #: Volledige herkomst. Zonder deze velden is `eta` een getal zonder bron.
    method: str = Field(min_length=1)
    data_hash: str = Field(min_length=1)
    sample_size: PositiveInt
    period_start: str = Field(min_length=1)
    period_end: str = Field(min_length=1)
    instruments: tuple[str, ...] = Field(min_length=1)
    eta_ci_low: Annotated[float, Field(gt=0.0)]
    eta_ci_high: Annotated[float, Field(gt=0.0)]

    @model_validator(mode="after")
    def _ci_contains_point_estimate(self) -> ImpactConfig:
        if not (self.eta_ci_low <= self.eta <= self.eta_ci_high):
            raise ValueError(
                "eta ligt buiten zijn eigen onzekerheidsband "
                f"[{self.eta_ci_low}, {self.eta_ci_high}]."
            )
        return self


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
    features: FeatureConfig
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
    "features": FeatureConfig,
    "alpha": AlphaConfig,
    "risk": RiskConfig,
    "execution": ExecutionConfig,
    "impact": ImpactConfig,
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
