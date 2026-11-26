"""Config-contract van de options-setup (`conf/options/default.yaml`).

Elk getal dat een handelsbeslissing stuurt, staat in de config en wordt hier gevalideerd.
Een onbekende sleutel is een fout bij het laden, geen waarschuwing (`StrictModel`,
`extra="forbid"`). De secties groeien per fase mee (detailontwerp 3.1): fase 2 bracht
`universe` en `pricing`, fase 3 `filters` en `iv_metrics`, fase 4 `strategies`, `sizing` en
`schedule`, fase 5 `risk` en `execution`; daarna `vol_regime` en `theta_harvest`.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import Field, ValidationError, model_validator

from ..utils.failfast import ConfigContractError
from .config import StrictModel

__all__ = ["DEFAULT_OPTIONS_CONFIG", "CalendarSpreadConfig", "CashSecuredPutConfig",
           "CoveredCallConfig", "CreditSpreadConfig", "DebitVerticalConfig", "ExecutionConfig",
           "FilterConfig",
           "IVFallbackConfig", "IVMetricsConfig", "IronCondorConfig", "LongVolConfig",
           "OptionsConfig", "PricingConfig", "RegimeAllowedConfig", "RiskConfig", "ScheduleConfig",
           "ShortVolConfig", "SizingConfig", "StrategiesConfig", "ThetaHarvestConfig",
           "UniverseConfig", "VolRegimeConfig", "WheelConfig",
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
    # Een smilepunt telt pas mee vanaf deze vega (USD per volpunt per aandeel), zie
    # `pricing.surface`: bij een kleinere vega is de IV van een op centen afgeronde quote ruis.
    surface_min_vega: Annotated[float, Field(ge=0.0, le=1.0)] = 0.01
    risk_free_rate: Annotated[float, Field(ge=-0.05, le=0.25)]
    # De keten in batches: BAW met de numpy-kernel (vastgepind tegen QuantLib), FD met een
    # startwaarde. Uit = elk contract apart door QuantLib.
    vectorized: bool = True
    # Het QuantLib-deel van de keten (FD, binomiaal, terugval) over zoveel processen; "auto" =
    # min(4, aantal cores), 1 = in het eigen proces. De pool pas boven `pool_min_contracts`
    # contracten: daaronder kost het verdelen meer dan het oplevert.
    workers: Annotated[int, Field(ge=1, le=64)] | Literal["auto"] = "auto"
    pool_min_contracts: Annotated[int, Field(ge=0)] = 16

    @model_validator(mode="after")
    def _consistent(self) -> PricingConfig:
        if self.binomial_steps % 2 == 0:
            raise ValueError("binomial_steps moet oneven zijn (Leisen-Reimer)")
        if self.iv_min >= self.iv_max:
            raise ValueError("iv_min moet kleiner zijn dan iv_max")
        return self


Fraction = Annotated[float, Field(gt=0.0, le=1.0)]
IVRLevel = Annotated[float, Field(ge=0.0, le=100.0)]
Ratio = Annotated[float, Field(gt=0.0, le=5.0)]


class FilterConfig(StrictModel):
    """Liquiditeit en datakwaliteit per contract en per onderliggende (detailontwerp 6.1)."""

    min_open_interest: Annotated[int, Field(ge=0)] = 500
    min_volume: Annotated[int, Field(ge=0)] = 100
    max_spread_abs: Annotated[float, Field(gt=0.0)] = 0.10
    max_spread_frac_mid: Fraction = 0.10
    min_underlying_adv_shares: Annotated[int, Field(ge=0)] = 1_000_000
    max_quote_age_seconds: Annotated[float, Field(gt=0.0)] = 10.0
    earnings_blackout: bool = True
    max_calendar_data_age_days: Annotated[int, Field(ge=0)] = 1


class IVFallbackConfig(StrictModel):
    """Vervangende regel zolang er geen 252 dagen IV-historie is (open vraag B)."""

    enabled: bool = True
    high_iv_ratio: Ratio = 1.20
    low_iv_ratio: Ratio = 1.00
    very_low_iv_ratio: Ratio = 0.90

    @model_validator(mode="after")
    def _ordered(self) -> IVFallbackConfig:
        if not self.very_low_iv_ratio <= self.low_iv_ratio <= self.high_iv_ratio:
            raise ValueError("verwacht very_low_iv_ratio <= low_iv_ratio <= high_iv_ratio")
        return self


class IVMetricsConfig(StrictModel):
    """IV rank, ATM-tenor, RV-venster en de drempels van de IV-voorwaarden (detailontwerp 6.3)."""

    ivr_lookback_days: Annotated[int, Field(ge=20, le=400)] = 252
    atm_tenor_days: Annotated[int, Field(ge=7, le=365)] = 30
    rv_window_days: Annotated[int, Field(ge=5, le=252)] = 20
    high_ivr: IVRLevel = 30.0
    low_ivr: IVRLevel = 30.0
    very_low_ivr: IVRLevel = 20.0
    fallback: IVFallbackConfig = IVFallbackConfig()

    @model_validator(mode="after")
    def _ordered(self) -> IVMetricsConfig:
        if not self.very_low_ivr <= self.low_ivr <= self.high_ivr:
            raise ValueError("verwacht very_low_ivr <= low_ivr <= high_ivr")
        return self


Delta = Annotated[float, Field(gt=0.0, lt=1.0)]
Days = Annotated[int, Field(ge=0, le=730)]
Multiple = Annotated[float, Field(gt=0.0, le=20.0)]
IVConditionName = Literal["high", "low", "very_low"]
ClockTime = Annotated[str, Field(pattern=r"^([01][0-9]|2[0-3]):[0-5][0-9]$")]


def _range(lo: float, hi: float, name: str) -> None:
    if lo > hi:
        raise ValueError(f"{name}: ondergrens boven bovengrens")


class _Strategy(StrictModel):
    enabled: bool = True

    @model_validator(mode="after")
    def _ranges(self) -> _Strategy:
        for name in type(self).model_fields:
            v = getattr(self, name)
            if isinstance(v, tuple) and len(v) == 2 and all(isinstance(x, int | float)
                                                            for x in v):
                _range(v[0], v[1], name)
        return self


class CreditSpreadConfig(_Strategy):
    """Bull put bij trend UP, bear call bij DOWN (detailontwerp 7.4)."""

    dte: tuple[Days, Days] = (30, 45)
    short_delta: tuple[Delta, Delta] = (0.20, 0.30)
    short_delta_target: Delta = 0.25
    wing_width_strikes: tuple[Annotated[int, Field(ge=1)], Annotated[int, Field(ge=1)]] = (1, 5)
    # credit / breedte is ongeveer de kans op ITM, dus ongeveer |short delta|: ⅓ is bij
    # delta 0,25 onhaalbaar (detailontwerp 7.4 noemde ⅓; zie CHANGELOG fase 4).
    min_credit_frac_width: Fraction = 0.20
    iv_condition: IVConditionName = "high"
    take_profit_frac: Fraction = 0.50
    stop_loss_multiple_credit: Multiple = 2.0
    itm_stop_frac: Fraction = 0.01
    close_dte: Days = 21


class IronCondorConfig(_Strategy):
    dte: tuple[Days, Days] = (30, 45)
    short_delta: Delta = 0.16
    wing_width_strikes: tuple[Annotated[int, Field(ge=1)], Annotated[int, Field(ge=1)]] = (5, 10)
    min_credit_frac_width: Fraction = 0.15          # totaal, beide kanten; zie credit spread
    iv_condition: IVConditionName = "high"
    take_profit_frac: Fraction = 0.50
    stop_loss_multiple_credit: Multiple = 2.0
    close_dte: Days = 21
    adjust_short_delta: Delta = 0.30
    max_adjustments: Annotated[int, Field(ge=0, le=3)] = 1


class DebitVerticalConfig(_Strategy):
    """Bull call bij trend UP, bear put bij DOWN."""

    dte: tuple[Days, Days] = (30, 60)
    long_delta: tuple[Delta, Delta] = (0.50, 0.60)
    long_delta_target: Delta = 0.55
    max_debit_frac_width: Fraction = 0.50
    max_width_strikes: Annotated[int, Field(ge=1, le=30)] = 10
    iv_condition: IVConditionName = "low"
    take_profit_frac_debit: Multiple = 1.0      # winst als fractie van het debit
    stop_loss_frac_debit: Fraction = 0.50       # sluiten als de waarde hieronder komt
    close_dte: Days = 21


class CalendarSpreadConfig(_Strategy):
    short_dte: tuple[Days, Days] = (20, 35)
    long_dte: tuple[Days, Days] = (50, 90)
    min_term_slope: float = 0.0
    max_ivp: IVRLevel = 50.0
    # Zonder IV-historie: IV/RV hieronder (vervangende regel, gemarkeerd).
    max_iv_rv_ratio_fallback: Ratio = 1.00
    take_profit_multiple_debit: Multiple = 1.25
    stop_loss_frac_debit: Fraction = 0.50


class LongVolConfig(_Strategy):
    """Long straddle of strangle, twee varianten die elk apart aan of uit staan."""

    style: Literal["straddle", "strangle"] = "straddle"
    strangle_delta: tuple[Delta, Delta] = (0.25, 0.30)
    earnings_variant: bool = True
    earnings_window_days: tuple[Days, Days] = (5, 15)
    earnings_min_dte: Days = 14
    low_iv_variant: bool = True
    low_iv_condition: IVConditionName = "very_low"
    low_iv_dte: tuple[Days, Days] = (30, 45)
    take_profit_multiple_debit: Multiple = 1.25
    stop_loss_frac_debit: Fraction = 0.50
    close_dte: Days = 7
    pre_earnings_close_time: ClockTime = "15:00"


class CashSecuredPutConfig(_Strategy):
    enabled: bool = False                       # open vraag C
    dte: tuple[Days, Days] = (30, 45)
    delta: tuple[Delta, Delta] = (0.20, 0.30)
    delta_target: Delta = 0.25
    iv_condition: IVConditionName = "high"
    take_profit_frac: Fraction = 0.50
    roll_dte: Days = 21


class CoveredCallConfig(_Strategy):
    enabled: bool = False
    dte: tuple[Days, Days] = (30, 45)
    delta: tuple[Delta, Delta] = (0.20, 0.30)
    delta_target: Delta = 0.25
    take_profit_frac: Fraction = 0.50
    roll_dte: Days = 21


class WheelConfig(_Strategy):
    """CSP en covered call als toestandsmachine; gebruikt hun instellingen."""

    enabled: bool = False


class ShortVolConfig(_Strategy):
    """Undefined risk: laadt alleen met `risk.allow_undefined_risk: true`."""

    enabled: bool = False
    style: Literal["straddle", "strangle"] = "strangle"
    strangle_delta: Delta = 0.16
    dte: tuple[Days, Days] = (30, 45)
    iv_condition: IVConditionName = "high"
    take_profit_frac: Fraction = 0.50
    stop_loss_multiple_credit: Multiple = 2.0
    close_dte: Days = 21


class StrategiesConfig(StrictModel):
    """De strategieën, in de vaste volgorde waarin de engine hun kandidaten neemt."""

    credit_spread: CreditSpreadConfig = CreditSpreadConfig()
    iron_condor: IronCondorConfig = IronCondorConfig()
    debit_vertical: DebitVerticalConfig = DebitVerticalConfig()
    calendar_spread: CalendarSpreadConfig = CalendarSpreadConfig()
    long_straddle_strangle: LongVolConfig = LongVolConfig()
    cash_secured_put: CashSecuredPutConfig = CashSecuredPutConfig()
    covered_call: CoveredCallConfig = CoveredCallConfig()
    wheel: WheelConfig = WheelConfig()
    short_straddle_strangle: ShortVolConfig = ShortVolConfig()


StrategyName = Literal["credit_spread", "iron_condor", "debit_vertical", "calendar_spread",
                       "long_straddle_strangle", "cash_secured_put", "covered_call", "wheel",
                       "short_straddle_strangle"]
_SHORT_PREMIUM: tuple[StrategyName, ...] = (
    "credit_spread", "iron_condor", "cash_secured_put", "covered_call", "wheel",
    "short_straddle_strangle")


class RegimeAllowedConfig(StrictModel):
    """Per regime de strategieën die NIEUWE posities mogen openen (sluiten mag altijd)."""

    rich: tuple[StrategyName, ...] = _SHORT_PREMIUM
    normal: tuple[StrategyName, ...] = (
        "credit_spread", "debit_vertical", "calendar_spread", "long_straddle_strangle",
        "cash_secured_put", "covered_call", "wheel")
    cheap: tuple[StrategyName, ...] = (
        "debit_vertical", "calendar_spread", "long_straddle_strangle")
    stress: tuple[StrategyName, ...] = ()
    unknown: tuple[StrategyName, ...] = ()


class VolRegimeConfig(StrictModel):
    """Verwachte RV, volatility risk premium en regime (`options.vol_regime`). Vaste regels."""

    enabled: bool = True
    rv_windows: tuple[int, int, int] = (5, 20, 60)        # handelsdagen: kort, middel, lang
    rv_weights: tuple[Fraction, Fraction, Fraction] = (0.2, 0.5, 0.3)   # in variantie, vast
    stress_term_slope: Annotated[float, Field(ge=-0.5, le=0.0)] = -0.02
    stress_rv_expansion: Annotated[float, Field(gt=1.0, le=5.0)] = 1.5
    rich_vrp_ratio: Ratio = 1.15
    rich_min_vrp_points: Annotated[float, Field(ge=0.0, le=0.5)] = 0.02
    rich_min_term_slope: Annotated[float, Field(ge=-0.5, le=0.5)] = -0.005
    cheap_vrp_ratio: Ratio = 0.95
    market_underlying: Ticker | None = "SPY"
    allowed: RegimeAllowedConfig = RegimeAllowedConfig()

    @model_validator(mode="after")
    def _consistent(self) -> VolRegimeConfig:
        short, mid, long = self.rv_windows
        if not 2 <= short < mid < long <= 252:
            raise ValueError("rv_windows moet oplopend zijn, tussen 2 en 252")
        if not self.cheap_vrp_ratio <= self.rich_vrp_ratio:
            raise ValueError("cheap_vrp_ratio moet <= rich_vrp_ratio")
        if self.stress_term_slope > self.rich_min_term_slope:
            raise ValueError("stress_term_slope moet <= rich_min_term_slope")
        return self


class ThetaHarvestConfig(StrictModel):
    """De poort voor kandidaten die premie ontvangen (`options.selection`)."""

    enabled: bool = True
    # (θ/dag × 365) / (½·|Γ|·S²·σ_RV²): boven 1 betaalt de premie het verwachte gamma-verlies.
    min_theta_gamma_cover: Annotated[float, Field(ge=0.0, le=10.0)] = 1.0
    min_theta_per_max_loss: Annotated[float, Field(ge=0.0, le=0.1)] = 0.0   # θ/dag / max verlies


class SizingConfig(StrictModel):
    max_loss_frac_capital: Fraction = 0.02
    ranking: Literal["credit_per_max_loss"] = "credit_per_max_loss"   # vast, detailontwerp 7.3


class RiskConfig(StrictModel):
    """Risicopoort (detailontwerp 8); fracties zijn van `capital_usd`."""

    max_abs_beta_delta_frac: Fraction = 0.30
    max_vega_frac: Fraction = 0.005
    max_gamma_loss_5pct_frac: Fraction = 0.03
    max_loss_per_underlying_frac: Fraction = 0.05
    max_buying_power_frac: Fraction = 0.50
    max_open_positions: Annotated[int, Field(ge=1, le=100)] = 10
    max_new_positions_per_day: Annotated[int, Field(ge=1, le=50)] = 3
    daily_loss_stop_entries_frac: Fraction = 0.02
    daily_loss_degross_frac: Fraction = 0.04
    drawdown_halt_frac: Fraction = 0.10
    scenario_spot: tuple[Annotated[float, Field(gt=-0.9, le=1.0)], ...] = (
        -0.20, -0.10, -0.05, 0.05, 0.10, 0.20)
    scenario_vol_points: tuple[Annotated[float, Field(ge=-50.0, le=100.0)], ...] = (
        -10.0, 0.0, 10.0, 20.0)
    max_scenario_loss_frac: Fraction = 0.08
    allow_undefined_risk: bool = False
    pdt_guard: bool = True

    @model_validator(mode="after")
    def _ordered(self) -> RiskConfig:
        if not self.daily_loss_stop_entries_frac <= self.daily_loss_degross_frac:
            raise ValueError("daily_loss_stop_entries_frac moet <= daily_loss_degross_frac")
        if not self.scenario_vol_points:
            raise ValueError("scenario_vol_points mag niet leeg zijn")
        return self


class TickConfig(StrictModel):
    below_3: Annotated[float, Field(gt=0.0, le=1.0)] = 0.01
    above_3: Annotated[float, Field(gt=0.0, le=1.0)] = 0.05


class TicksConfig(StrictModel):
    default: TickConfig = TickConfig()
    combo: Annotated[float, Field(gt=0.0, le=1.0)] = 0.01


class ExecutionConfig(StrictModel):
    """Orders en de price-walk (detailontwerp 10)."""

    walk_interval_seconds: Annotated[float, Field(gt=0.0, le=600.0)] = 15.0
    max_concession_entry: Fraction = 0.60      # fractie van mid naar de natuurlijke prijs
    max_concession_exit: Fraction = 1.00
    ticks: TicksConfig = TicksConfig()
    max_order_usd: Annotated[float, Field(gt=0.0)] = 5000.0


class ScheduleConfig(StrictModel):
    """Tijden in New York (detailontwerp 9.2)."""

    close_before_expiry_time: ClockTime = "15:00"
    scan_times: tuple[ClockTime, ...] = ("10:00", "13:30")
    end_of_day: ClockTime = "16:15"
    no_entry_first_minutes: Annotated[int, Field(ge=0, le=120)] = 15
    no_entry_last_minutes: Annotated[int, Field(ge=0, le=120)] = 30
    manage_interval_seconds: Annotated[float, Field(gt=0.0, le=3600.0)] = 60.0
    risk_interval_seconds: Annotated[float, Field(gt=0.0, le=3600.0)] = 60.0
    reconcile_interval_seconds: Annotated[float, Field(gt=0.0, le=86400.0)] = 300.0

    @model_validator(mode="after")
    def _scans(self) -> ScheduleConfig:
        if not self.scan_times:
            raise ValueError("scan_times mag niet leeg zijn")
        if list(self.scan_times) != sorted(set(self.scan_times)):
            raise ValueError("scan_times moet oplopend en zonder dubbelen zijn")
        return self


class AccountConfig(StrictModel):
    """Het fictieve paper-kapitaal; alle risicofracties gaan hierover."""

    capital_usd: Annotated[float, Field(gt=0.0, le=10_000_000.0)] = 25_000.0


class PaperConfig(StrictModel):
    """De gesimuleerde broker voor replay en paper (detailontwerp 10.4)."""

    fill_at_concession: Fraction = 0.40     # vult zodra de limiet zo ver richting natuurlijk is
    commission_per_contract: Annotated[float, Field(ge=0.0, le=10.0)] = 0.65


class OptionsConfig(StrictModel):
    universe: UniverseConfig
    filters: FilterConfig
    pricing: PricingConfig
    iv_metrics: IVMetricsConfig
    strategies: StrategiesConfig
    sizing: SizingConfig
    risk: RiskConfig
    execution: ExecutionConfig
    schedule: ScheduleConfig
    account: AccountConfig = AccountConfig()
    paper: PaperConfig = PaperConfig()
    vol_regime: VolRegimeConfig = VolRegimeConfig()
    theta_harvest: ThetaHarvestConfig = ThetaHarvestConfig()

    @model_validator(mode="after")
    def _market_underlying(self) -> OptionsConfig:
        mu = self.vol_regime.market_underlying
        if mu is not None and mu not in self.universe.underlyings:
            raise ValueError(f"vol_regime.market_underlying {mu} staat niet in universe")
        return self


def load_options_config(path: Path = DEFAULT_OPTIONS_CONFIG) -> OptionsConfig:
    """Laad en valideer de options-config; elke fout is een `ConfigContractError`."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ConfigContractError(f"{path}: verwacht een mapping bovenaan")
    try:
        return OptionsConfig.model_validate(raw)
    except ValidationError as e:
        raise ConfigContractError(f"{path}: {e}") from e
