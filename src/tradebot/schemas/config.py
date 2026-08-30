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

from functools import lru_cache
from itertools import pairwise
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
    "AdequacyConfig",
    "BacktestConfig",
    "DataConfig",
    "EconometricsConfig",
    "ExecutionConfig",
    "FeatureConfig",
    "FracDiffConfig",
    "ImpactConfig",
    "LabelingConfig",
    "MicrostructureFeatureConfig",
    "MomentumFeatureConfig",
    "RiskConfig",
    "StrictModel",
    "TcaConfig",
    "GarchAdequacyConfig",
    "HarRvAdequacyConfig",
    "HmmAdequacyConfig",
    "HrpAdequacyConfig",
    "MetaLabelingAdequacyConfig",
    "PowerConfig",
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

    # --- Meerstaps GARCH-forecasts (Phase 6, stap 7) ----------------------- #
    # Voor EGARCH en APARCH BESTAAT er geen analytische meerstaps-forecast; de
    # recursie loopt in ln(sigma^2) respectievelijk sigma^delta en de
    # terugtransformatie heeft geen gesloten vorm. `arch` weigert daar terecht.
    # De meerstaps-forecast wordt daarom gesimuleerd, voor ELKE variant, zodat
    # geen enkel model op h > 1 een andersoortige forecast krijgt dan zijn
    # concurrent. Op h = 1 blijft de analytische forecast staan: daar bestaat
    # het exacte antwoord en hoort er geen Monte-Carlo-ruis in.
    forecast_simulations: PositiveInt = 5000

    # Zonder vaste seed is elke QLIKE-uitslag op h > 1 onherhaalbaar, en een
    # niet-reproduceerbaar getal is geen bewijs.
    forecast_seed: PositiveInt = 20260826


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
        for prev, nxt in pairwise(v):
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


class TcaConfig(StrictModel):
    """Contract voor Transaction Cost Analysis (L9). Zie `conf/tca/default.yaml`.

    De tolerantie staat in configuratie en niet in de code omdat fase-opdracht
    §13 eist dat zij VOORAF vastligt. Een tolerantie die de implementatie kiest,
    beweegt mee met de uitkomst die zij moet toetsen.
    """

    #: Numerieke tolerantie van de sluitingsidentiteit, in basispunten van de
    #: startequity.
    closure_tolerance_bps: Annotated[float, Field(gt=0.0)]

    #: De timing-component is contrafeitelijk en telt niet mee in de sluiting.
    #: `Literal[False]`: dit is geen schakelaar maar een vastgelegde beslissing.
    timing_counts_towards_closure: Literal[False] = False

    #: Kostencomponenten dragen altijd hun herkomststatus mee.
    require_cost_provenance: Literal[True] = True


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
# L11 - Data Adequacy Gate (Phase 6, §3)
# --------------------------------------------------------------------------- #
class GarchAdequacyConfig(StrictModel):
    """Minimumeisen voor een Level-2 GARCH-fit. Zie `conf/model/adequacy.yaml`."""

    min_obs_per_fit_window: PositiveInt = 250
    min_convergence_ratio: Fraction = 0.90
    persistence_boundary: Annotated[float, Field(gt=0.0, le=1.0)] = 0.999
    max_boundary_solution_ratio: Annotated[float, Field(ge=0.0, le=1.0)] = 0.10

    #: Een variantieforecast boven dit veelvoud van de gemiddelde gekwadrateerde
    #: return is geen forecast meer maar een numeriek artefact. GEMETEN op
    #: 2026-08-29: de gesimuleerde 5-staps EGARCH-forecast liep op BTCUSDT op
    #: tot 4,5e25 maal die referentie. De EGARCH-recursie loopt in ln(sigma^2)
    #: en de verwachting van exp van een zwaarstaartige random walk hoeft niet
    #: te bestaan; de simulatie schat dan een moment dat er niet is.
    max_forecast_level_ratio: Annotated[float, Field(gt=1.0)] = 100.0

    #: Een variantieforecast die dit veelvoud van de gemiddelde gekwadrateerde
    #: return overschrijdt, is geen forecast meer. GEMETEN op 2026-08-29: de
    #: gesimuleerde 5-staps EGARCH-forecast liep op BTCUSDT op tot 4,5e25 maal
    #: die referentie. De EGARCH-recursie loopt in ln(sigma^2) en de verwachting
    #: van exp van een zwaarstaartige random walk hoeft niet te bestaan; de
    #: simulatie schat dan een moment dat er niet is.
    #:
    #: 100 (tienmaal in volatiliteit) is ruim; de gemeten uitschieters liggen er
    #: vijftien ordes van grootte boven, dus de conclusie hangt niet aan deze
    #: waarde.
    max_forecast_level_ratio: Annotated[float, Field(gt=1.0)] = 100.0


class HarRvAdequacyConfig(StrictModel):
    """Minimumeisen voor Level-3 HAR-RV op realized variance."""

    intraday_granularity: str = "5m"
    min_bars_per_day: PositiveInt = 200
    min_days_with_coverage_pct: Annotated[float, Field(ge=0.0, le=100.0)] = 80.0
    min_obs_per_fit_window: PositiveInt = 500


class HmmAdequacyConfig(StrictModel):
    """Minimumeisen voor een M2 Filtered HMM per toestand per fold."""

    min_obs_per_state_per_fold: PositiveInt = 100
    min_state_occupancy_fraction: Fraction = 0.10


class MetaLabelingAdequacyConfig(StrictModel):
    """Minimumeisen voor een CatBoost secondary model per fold."""

    min_events_per_fold: PositiveInt = 200
    min_effective_events_per_fold: PositiveInt = 100
    min_positive_class_ratio: Fraction = 0.20
    max_positive_class_ratio: Fraction = 0.80

    @model_validator(mode="after")
    def _band_is_ordered(self) -> MetaLabelingAdequacyConfig:
        if self.min_positive_class_ratio >= self.max_positive_class_ratio:
            raise ValueError(
                "min_positive_class_ratio moet onder max_positive_class_ratio "
                f"liggen; kreeg {self.min_positive_class_ratio} >= "
                f"{self.max_positive_class_ratio}"
            )
        return self


class HrpAdequacyConfig(StrictModel):
    """Minimumeisen voor een stabiele correlatiematrix onder HRP."""

    min_obs_per_asset: PositiveInt = 10
    max_condition_number: Annotated[float, Field(gt=1.0)] = 100.0


class PowerConfig(StrictModel):
    """Parameters van de power-analyse die elke pre-registratie draagt."""

    alpha: Annotated[float, Field(gt=0.0, lt=1.0)] = 0.05
    target_power: Annotated[float, Field(gt=0.0, lt=1.0)] = 0.80
    two_sided: bool = True


class ProxyAdequacyConfig(StrictModel):
    """Contract voor de variantieproxy van de QLIKE-competitie (Phase 6 stap 7).

    QLIKE heeft zijn minimum op `forecast = E[proxy]`. Draagt de proxy een
    multiplicatieve factor ten opzichte van de grootheid die de modellen
    voorspellen -- de variantie van de close-to-close return -- dan verschuift
    dat minimum mee, en wint het model waarvan het NIVEAU bij die factor past in
    plaats van het model dat de dynamiek het beste voorspelt.
    """

    #: Hoeveel de gemiddelde proxy van de gemiddelde gekwadrateerde return mag
    #: afwijken voordat zij niet meer als zuivere proxy geldt. De
    #: gekwadrateerde return is de referentie omdat hij per constructie zuiver
    #: is voor precies de grootheid die de modellen voorspellen:
    #: `E[r_t^2 | F_{t-1}] = sigma_t^2`. Ruisig, maar zuiver.
    max_scale_deviation: Annotated[float, Field(gt=0.0, lt=1.0)] = 0.15


class AdequacyConfig(StrictModel):
    """Contract voor de Data Adequacy Gate (L11). Zie `conf/model/adequacy.yaml`.

    Phase 6 §3: geen enkel model mag worden gefit voordat zijn data-adequaatheid
    is gemeten en geregistreerd. De drempels staan hier en niet in code, zodat
    een drempel niet achteraf kan worden verlaagd om een model door de poort te
    krijgen.
    """

    garch: GarchAdequacyConfig = Field(default_factory=GarchAdequacyConfig)
    har_rv: HarRvAdequacyConfig = Field(default_factory=HarRvAdequacyConfig)
    hmm: HmmAdequacyConfig = Field(default_factory=HmmAdequacyConfig)
    meta_labeling: MetaLabelingAdequacyConfig = Field(
        default_factory=MetaLabelingAdequacyConfig)
    hrp: HrpAdequacyConfig = Field(default_factory=HrpAdequacyConfig)
    proxy: ProxyAdequacyConfig = Field(default_factory=ProxyAdequacyConfig)
    power: PowerConfig = Field(default_factory=PowerConfig)


# --------------------------------------------------------------------------- #
# L6 - Meta-labeling (Phase 6, deliverable 19)
# --------------------------------------------------------------------------- #
class LabelingConfig(StrictModel):
    """Contract voor triple-barrier labeling. Zie `conf/model/labeling.yaml`."""

    profit_target_sigma: Annotated[float, Field(gt=0.0)] = 2.0
    stop_loss_sigma: Annotated[float, Field(gt=0.0)] = 2.0
    horizon_bars: PositiveInt = 10

    #: >= 1 en niet configureerbaar naar 0. Zie `conf/model/labeling.yaml` voor
    #: waarom: 0 betekent handelen op de close waarop je besluit, en dat is de
    #: aanname die `reports/phase5_engine_diff.md` weerlegde.
    entry_lag_bars: Annotated[int, Field(ge=1)] = 1

    min_sigma_obs: PositiveInt = 60


# --------------------------------------------------------------------------- #
# L4 - Fractionele differentiering (Phase 6, deliverable 8)
# --------------------------------------------------------------------------- #
class FracDiffConfig(StrictModel):
    """Contract voor fractionele differentiering. Zie `conf/model/fracdiff.yaml`."""

    d_lo: Annotated[float, Field(ge=0.0, lt=1.0)] = 0.05
    d_hi: Annotated[float, Field(gt=0.0, le=1.0)] = 0.95
    weight_threshold: Annotated[float, Field(gt=0.0, lt=1.0)] = 1.0e-5
    adf_p_target: Annotated[float, Field(gt=0.0, lt=1.0)] = 0.05
    max_iter: PositiveInt = 24

    @model_validator(mode="after")
    def _range_is_ordered(self) -> FracDiffConfig:
        if self.d_lo >= self.d_hi:
            raise ValueError(
                f"d_lo ({self.d_lo}) moet onder d_hi ({self.d_hi}) liggen"
            )
        return self


# --------------------------------------------------------------------------- #
# L11 - Econometrische toetsdrempels (Phase 7/8, Stage A-3)
# --------------------------------------------------------------------------- #
class EconometricsConfig(StrictModel):
    """Drempels van de diagnostische toetslaag. Zie `conf/validation/econometrics.yaml`.

    WAAROM DIT BESTAAT
    ------------------
    Deze waarden stonden tot Stage A-3 als numerieke defaults in de signatuur van
    `validation/econometrics.py`, `validation/vol_metrics.py`,
    `validation/diagnostics_report.py`, `validation/data_adequacy.py` en
    `volatility/realized.py`. Vijf modules met ratchet-budget 0 hielden daarmee
    vijftien literals vast en `check_hardcoded_params.py --strict` stond rood.

    Een significantiedrempel is geen rekenkundig feit maar een BELEIDSKEUZE: hij
    bepaalt hoe vaak een toets ten onrechte verwerpt, en dat is precies het soort
    keuze dat achteraf kan worden bijgesteld om een model door een poort te
    krijgen. Daarom staat hij hier, gevalideerd, frozen, en in de `config_hash`.

    De waarden zijn ONGEWIJZIGD overgenomen van de defaults die zij vervangen.
    Stage A-3 verplaatst; het verandert geen enkel getal.
    """

    #: Significantieniveau van ADF, KPSS, Ljung-Box, Engle-ARCH en CUSUM.
    alpha: Annotated[float, Field(gt=0.0, lt=1.0)] = 0.05

    #: Aantal lags voor de Ljung-Box-toets op de niveaus.
    ljung_box_lags: PositiveInt = 20

    #: Minimaal aantal observaties voor een Mincer-Zarnowitz-regressie met HAC.
    mincer_zarnowitz_min_obs: PositiveInt = 30

    #: Minimale overlap tussen twee verliesreeksen voordat een DM-toets is
    #: toegestaan. Onder deze fractie vergelijkt de toets twee modellen op
    #: verschillende steekproeven en meet hij het verschil in steekproef in
    #: plaats van het verschil in model.
    dm_min_intersection_ratio: Annotated[float, Field(gt=0.0, le=1.0)] = 0.80

    #: De AUC-drempel uit audit §24 waartegen de minimaal detecteerbare
    #: effectgrootte wordt uitgedrukt.
    auc_expected: Annotated[float, Field(gt=0.5, lt=1.0)] = 0.58

    #: Vensterlengte van de Yang-Zhang-variantie-estimator. YZ is inherent een
    #: VENSTER-estimator; er bestaat geen per-bar variant.
    yang_zhang_window: PositiveInt = 20


# --------------------------------------------------------------------------- #
# L3 - Regime engines (Phase 6, deliverables 14 en 15)
# --------------------------------------------------------------------------- #
class M0BucketConfig(StrictModel):
    """Drempels van de M0 Causal Vol-Buckets. Zie `conf/model/regime.yaml`.

    Elke waarde hier is een DREMPEL en geen geschatte parameter. Dat is het hele
    punt van M0: nul latente toestanden, nul schattingen, nul lekrisico. Zij
    staan in config en niet in code, zodat een wijziging zichtbaar is in de
    diff en meetelt als een nieuwe trial.
    """

    #: Onder deze causale z-score van de log-EWMA-vol heet het regime LAAG.
    zscore_low: float = -0.5
    #: Boven deze z-score heet het regime HOOG.
    zscore_high: float = 0.5
    #: Observaties voordat de causale z-score bestaat. Daarvoor is het regime
    #: ONGEDEFINIEERD - nooit "normaal bij gebrek aan beter".
    zscore_min_periods: PositiveInt = 250

    atr_fast_window: PositiveInt = 5
    atr_slow_window: PositiveInt = 20
    #: ATR-ratio's die vol-EXPANSIE respectievelijk -CONTRACTIE aanwijzen.
    atr_ratio_low: Annotated[float, Field(gt=0.0)] = 0.85
    atr_ratio_high: Annotated[float, Field(gt=0.0)] = 1.15

    @model_validator(mode="after")
    def _thresholds_are_ordered(self) -> M0BucketConfig:
        if self.zscore_low >= self.zscore_high:
            raise ValueError(
                f"zscore_low ({self.zscore_low}) moet onder zscore_high "
                f"({self.zscore_high}) liggen; anders is er geen NORMAAL-band "
                "en classificeert M0 elke bar als extreem."
            )
        if self.atr_ratio_low >= self.atr_ratio_high:
            raise ValueError(
                f"atr_ratio_low ({self.atr_ratio_low}) moet onder "
                f"atr_ratio_high ({self.atr_ratio_high}) liggen"
            )
        if self.atr_fast_window >= self.atr_slow_window:
            raise ValueError(
                f"atr_fast_window ({self.atr_fast_window}) moet korter zijn "
                f"dan atr_slow_window ({self.atr_slow_window}); anders meet de "
                "ratio geen verandering in volatiliteit"
            )
        return self


class M2HmmConfig(StrictModel):
    """De M1/M2-parameterruimte. Zie `conf/model/regime.yaml`.

    Anders dan bij M0 zijn dit GEEN drempels maar de omvang van een ZOEKRUIMTE.
    Elk element van `n_states_grid` maal elke variant is een trial en telt mee
    in `M`; de pre-registratie `3d3af28730a6c7f9da48d13139522a05` boekte er zes.
    Wie deze lijst uitbreidt, doet een nieuwe pre-registratie -- het uitbreiden
    van een bevroren ruimte maakt elke DSR erna te gunstig.
    """

    #: Het aantal latente toestanden dat wordt geprobeerd. Zes trials = drie
    #: varianten (M1, M2-gaussian, M2-student_t) maal deze twee waarden.
    n_states_grid: tuple[PositiveInt, ...] = (2, 3)
    covariance_type: Literal["diag", "full"] = "diag"
    #: Maximaal aantal EM-iteraties. Niet-convergentie is een RESULTAAT dat
    #: wordt geregistreerd, geen probleem dat wordt weggevangen.
    n_iter: PositiveInt = 200
    #: Relatieve tolerantie op de log-likelihood waaronder de EM stopt.
    em_tolerance: Annotated[float, Field(gt=0.0)] = 1e-6
    #: Startwaarde van de vrijheidsgraden in de Student-t EM.
    dof_init: Annotated[float, Field(gt=2.0)] = 8.0
    #: Ondergrens. Onder 2 bestaat de variantie van een t-verdeling niet, en een
    #: emissie zonder tweede moment maakt de toestandsvergelijking betekenisloos.
    dof_min: Annotated[float, Field(gt=2.0)] = 2.1
    #: Bovengrens. Erboven is de t numeriek niet meer van een normale te
    #: onderscheiden; de fit rapporteert dat hij de grens raakte.
    dof_max: Annotated[float, Field(gt=2.0)] = 200.0
    #: Ondergrens op `min(schaal) / max(schaal)` over de toestanden
    #: (Hathaway 1985). De t-mengselverdeling heeft een singulariteit waarin
    #: een toestand op een handvol punten instort, zijn schaal naar nul gaat en
    #: de likelihood naar oneindig. Dat is geen optimum. Zakt de verhouding
    #: hieronder, dan stopt de EM en meldt hij `degenerate`.
    min_scale_ratio: Annotated[float, Field(gt=0.0, lt=1.0)] = 1e-4
    #: Seed van de gedeelde initialisatie (k-means in de Gaussische EM), zodat
    #: de Gaussische en de Student-t variant vanaf HETZELFDE punt starten en het
    #: verschil tussen beide de verdeling is en niet het startpunt.
    seed: Annotated[int, Field(ge=0)] = 20260830

    @model_validator(mode="after")
    def _grid_is_sane(self) -> M2HmmConfig:
        if not self.n_states_grid:
            raise ValueError(
                "n_states_grid is leeg; dan is er geen M2-variant om te toetsen"
            )
        if min(self.n_states_grid) < 2:
            raise ValueError(
                f"n_states_grid bevat {min(self.n_states_grid)}; een HMM met "
                "minder dan twee toestanden is geen regimemodel"
            )
        if len(set(self.n_states_grid)) != len(self.n_states_grid):
            raise ValueError(
                f"n_states_grid bevat dubbelen ({self.n_states_grid}); een "
                "trial die twee keer in de ruimte staat, telt twee keer in M "
                "zonder twee keer iets te meten"
            )
        if self.dof_min >= self.dof_max:
            raise ValueError(
                f"dof_min ({self.dof_min}) moet onder dof_max "
                f"({self.dof_max}) liggen"
            )
        if not self.dof_min <= self.dof_init <= self.dof_max:
            raise ValueError(
                f"dof_init ({self.dof_init}) ligt buiten "
                f"[{self.dof_min}, {self.dof_max}]"
            )
        return self


class RegimeConfig(StrictModel):
    """Contract voor de L3 regime-engines. Zie `conf/model/regime.yaml`."""

    m0: M0BucketConfig = Field(default_factory=M0BucketConfig)
    m2: M2HmmConfig = Field(default_factory=M2HmmConfig)


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
    "tca": TcaConfig,
    "backtest": BacktestConfig,
    "validation": ValidationConfig,
    "econometrics": EconometricsConfig,
    "adequacy": AdequacyConfig,
    "labeling": LabelingConfig,
    "fracdiff": FracDiffConfig,
    "regime": RegimeConfig,
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


# --------------------------------------------------------------------------- #
# Gedeelde, eenmalig geladen drempels (Phase 7/8, Stage A-3)
# --------------------------------------------------------------------------- #
#: Repository-root. Zelfde conventie als `features/base.py::_repo_root`.
_REPO_ROOT = Path(__file__).resolve().parents[3]

ECONOMETRICS_CONFIG_PATH = _REPO_ROOT / "conf" / "validation" / "econometrics.yaml"


@lru_cache(maxsize=1)
def econometrics_config() -> EconometricsConfig:
    """De gevalideerde econometrische drempels uit `conf/validation/`.

    Deze accessor bestaat zodat `validation/` en `volatility/` dezelfde drempels
    delen zonder dat de een de ander importeert — een L2-module die uit L11 zou
    moeten importeren is een laaginversie.

    FAIL-FAST. Ontbreekt het bestand of schendt het zijn contract, dan crasht dit
    met `ConfigContractError` bij import van de consumerende module. Dat is
    opzettelijk: audit §23 en de faseregel *"ontbrekende configuratie: halteren"*
    sluiten uit dat een toets stilzwijgend terugvalt op een ingebouwde drempel.
    Een drempel die uit code komt in plaats van uit `conf/`, is niet gehasht en
    dus niet auditbaar.

    De cache is bewust: de config is `frozen`, wordt tijdens een run niet
    herladen, en een tweede lezing zou alleen maar een tweede kans zijn om een
    andere waarde te zien dan de eerste lezing gaf.
    """
    return load_config(ECONOMETRICS_CONFIG_PATH, EconometricsConfig)


ADEQUACY_CONFIG_PATH = _REPO_ROOT / "conf" / "model" / "adequacy.yaml"


@lru_cache(maxsize=1)
def adequacy_config() -> AdequacyConfig:
    """De gevalideerde drempels van de Data Adequacy Gate uit `conf/model/`.

    Zelfde constructie en dezelfde reden als :func:`econometrics_config`: de
    QLIKE-competitie (`validation/vol_competition.py`) velt haar oordeel tegen
    de convergentie-, randoplossings- en power-drempels van Phase 6, en die
    horen uit `conf/` te komen zodat ze gehasht en auditbaar zijn. Een drempel
    die in een handtekening staat, kan achteraf worden bijgesteld om een model
    door de poort te krijgen.
    """
    return load_config(ADEQUACY_CONFIG_PATH, AdequacyConfig)


VOLATILITY_CONFIG_PATH = _REPO_ROOT / "conf" / "model" / "volatility.yaml"


@lru_cache(maxsize=1)
def volatility_config() -> VolatilityConfig:
    """De L2-baselineparameters uit `conf/model/volatility.yaml`.

    Zelfde constructie en dezelfde reden als :func:`econometrics_config`. De
    QLIKE-competitie zet EWMA(0.94) in als titelverdediger; die lambda en die
    burn-in horen uit `conf/` te komen, want zij zijn onderdeel van het
    baselinecontract en niet van de competitie die hen toetst.
    """
    return load_config(VOLATILITY_CONFIG_PATH, VolatilityConfig)


REGIME_CONFIG_PATH = _REPO_ROOT / "conf" / "model" / "regime.yaml"


@lru_cache(maxsize=1)
def regime_config() -> RegimeConfig:
    """De L3-regimeparameters uit `conf/model/regime.yaml`.

    Zelfde constructie en dezelfde reden als :func:`econometrics_config`. Voor
    M0 gaat het om DREMPELS, voor M1/M2 om de omvang van de ZOEKRUIMTE -- en dat
    tweede is precies waarom het uit `conf/` moet komen: het aantal toestanden
    dat wordt geprobeerd bepaalt hoeveel trials er in `M` horen, en een
    zoekruimte die in code staat is achteraf uit te breiden zonder dat een diff
    het laat zien.
    """
    return load_config(REGIME_CONFIG_PATH, RegimeConfig)
