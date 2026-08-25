"""Gedeelde bouwstenen voor de Phase 5 integratietests.

Eén plek waar een engine, een router en een replay worden gebouwd, zodat de
pariteits-, TCA-, causaliteits- en wiringtests aantoonbaar op DEZELFDE opstelling
draaien. Twee tests die elk hun eigen fixture bouwen, kunnen groen zijn op twee
verschillende systemen.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from tradebot.backtest.engine import EventDrivenEngine, build_slices, exposures_from_frame
from tradebot.execution.context import MarketSlice
from tradebot.execution.impact_model import ImpactParams, ImpactStatus
from tradebot.execution.order_router import (
    OrderRouter,
    SpreadModel,
    SpreadStatus,
    VenueSpec,
)
from tradebot.risk.engine import RiskEngine
from tradebot.schemas.config import (
    ExecutionConfig,
    ImpactConfig,
    RiskConfig,
    TcaConfig,
    load_config,
)

ROOT = Path(__file__).resolve().parents[2]
CONF = ROOT / "conf"

SYMBOLS: tuple[str, ...] = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "LINKUSDT")
INITIAL_EQUITY = 100_000.0


def risk_config() -> RiskConfig:
    return load_config(CONF / "risk/default.yaml", RiskConfig)


def impact_params(cfg: ImpactConfig | None = None) -> ImpactParams:
    c = cfg or load_config(CONF / "execution/impact.yaml", ImpactConfig)
    return ImpactParams(
        eta=c.eta, kappa_d=c.kappa_d, status=ImpactStatus(c.status),
        method=c.method, data_hash=c.data_hash, sample_size=c.sample_size,
        period_start=c.period_start, period_end=c.period_end,
        instruments=c.instruments, eta_ci_low=c.eta_ci_low,
        eta_ci_high=c.eta_ci_high,
    )


def venue_spec() -> VenueSpec:
    ex = load_config(CONF / "execution/fees.yaml", ExecutionConfig)
    return VenueSpec(
        maker_fee_bps=ex.maker_fee_bps, taker_fee_bps=ex.taker_fee_bps,
        funding_cap_abs=0.02, min_notional=10.0, latency_bars=1,
    )


def spread_model() -> SpreadModel:
    ex = load_config(CONF / "execution/fees.yaml", ExecutionConfig)
    return SpreadModel(
        half_spread_bps=ex.assumed_half_spread_bps,
        status=SpreadStatus.SPREAD_ASSUMED,
        source="conf/execution/fees.yaml::assumed_half_spread_bps",
    )


def tca_tolerance_bps() -> float:
    return load_config(CONF / "tca/default.yaml", TcaConfig).closure_tolerance_bps


def make_router(
    risk_engine: RiskEngine, *, impact: ImpactParams | None = None,
    spread: SpreadModel | None = None, venue: VenueSpec | None = None,
) -> OrderRouter:
    return OrderRouter(
        venue=venue or venue_spec(),
        spread=spread or spread_model(),
        impact=impact or impact_params(),
        risk_config_hash=risk_engine.config_hash,
    )


def make_engine(
    cfg: RiskConfig | None = None, *, context_kind: str = "backtest",
    router: OrderRouter | None = None,
) -> EventDrivenEngine:
    engine = RiskEngine(cfg or risk_config())
    return EventDrivenEngine(
        risk_engine=engine,
        router=router or make_router(engine),
        initial_equity=INITIAL_EQUITY,
        context_kind=context_kind,
    )


@dataclass(frozen=True)
class Replay:
    """Een deterministische marktreeks plus de bijbehorende alpha-exposures.

    Deterministisch via een vaste seed en NIET via Hypothesis: de parity-eis is
    bit-identiteit, en die is alleen betekenisvol op een reeks die twee runs
    gegarandeerd delen.
    """

    slices: list[MarketSlice]
    exposures: dict[pd.Timestamp, dict[str, float]]
    prices: pd.DataFrame
    index: pd.DatetimeIndex

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(self.prices.columns)


def make_replay(
    n_bars: int = 60,
    *,
    seed: int = 20260825,
    symbols: tuple[str, ...] = SYMBOLS,
    sigma_annual: float = 0.60,
    adv_notional: float = 5.0e8,
    bar_volume_notional: float | None = None,
    funding_rate: float = 0.0001,
    clusters: Mapping[str, str] | None = None,
    exposure: float = 1.0,
) -> Replay:
    """Bouw een replay die beide contexts en alle vier de tests delen."""
    rng = np.random.default_rng(seed)
    index = pd.DatetimeIndex(
        pd.date_range("2024-01-01", periods=n_bars, freq="D", tz="UTC"),
        name="asof_ts",
    )
    daily = sigma_annual / np.sqrt(365.0)
    prices = pd.DataFrame(
        {s: 100.0 * np.exp(np.cumsum(rng.normal(0.0, daily, n_bars)))
         for s in symbols},
        index=index,
    )
    sigma_hat = pd.DataFrame(sigma_annual, index=index, columns=list(symbols))
    sigma_daily = sigma_hat / np.sqrt(365.0)
    adv = pd.DataFrame(adv_notional, index=index, columns=list(symbols))
    volume = pd.DataFrame(
        bar_volume_notional if bar_volume_notional is not None else adv_notional,
        index=index, columns=list(symbols),
    )
    funding = pd.DataFrame(funding_rate, index=index, columns=list(symbols))
    labels = dict(clusters) if clusters is not None else {
        s: ("crypto_l1" if s != "LINKUSDT" else "crypto_oracle") for s in symbols
    }
    slices = build_slices(prices, sigma_hat, sigma_daily, adv, volume, labels,
                          funding)
    exposures = exposures_from_frame(
        pd.DataFrame(exposure, index=index, columns=list(symbols)))
    return Replay(slices=slices, exposures=exposures, prices=prices, index=index)
