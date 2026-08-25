"""Het gecertificeerde Phase 6-universum, één keer geladen, voor elke app.

WAAROM DIT BESTAAT
==================
Elke Phase 6-app heeft exact dezelfde vier dingen nodig: de gecertificeerde
prijzen, de OHLC-frames, de causale EWMA-sigma en het primaire Phase 3-signaal —
allemaal op HETZELFDE bruikbare venster. Dat venster is niet triviaal: het is
de doorsnede van de bars waarop de EWMA-vol bestaat (60 bars burn-in) en de bars
waarop de causale 30-daagse ADV bestaat. Op deze store zijn dat er 1.743,
2021-11-15 t/m 2026-08-23.

Wie dat venster per app opnieuw afleidt, krijgt er vroeg of laat twee. Twee
vensters betekent dat de QLIKE-competitie en de engine-benchmark op verschillende
data draaien, en dat is precies het soort verschil dat je pas ontdekt als de
getallen niet meer op elkaar aansluiten. De afleiding staat daarom één keer hier
en is identiek aan die in `apps/run_phase5_baseline.py`, zodat elk Phase
6-resultaat vergelijkbaar is met de Phase 5-basislijn.

WAT HET PRIMAIRE SIGNAAL IS
===========================
`side` is het TEKEN van de exposure van de ongewijzigde Phase 3-baseline-unit
(Cross-Sectional Momentum uit `conf/model/alpha.yaml`). Deze fase evalueert
modellen, geen signalen (§2): het primaire signaal komt ongewijzigd uit Phase 3
en wordt hier alleen tot een richting gereduceerd, omdat dat is wat een
secondary model volgens §12.1 als gegeven aanneemt.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..features.base import (
    ASOF_INDEX_NAME,
    CertifiedPanel,
    DataRegister,
    load_certified_close_panel,
    load_certified_series,
)
from ..utils.failfast import DataContractError, require
from ..volatility.ewma import ewma_volatility_panel
from .pit_store import PitStore

__all__ = ["Phase6Universe", "load_phase6_universe"]

#: Venster van de causale ADV-schatter, identiek aan `apps/run_phase5_baseline.py`.
_ADV_WINDOW = 30


@dataclass(frozen=True)
class Phase6Universe:
    """Alles wat een Phase 6-app van de data nodig heeft, op één venster."""

    #: Slotkoersen op het bruikbare venster.
    prices: pd.DataFrame
    #: Volledige OHLC per symbool, op hetzelfde venster.
    ohlc: Mapping[str, pd.DataFrame]
    #: Log-returns, eerste bar valt weg.
    log_returns: pd.DataFrame
    #: GEANNUALISEERDE causale EWMA-volatiliteit.
    sigma_annual: pd.DataFrame
    #: Dezelfde volatiliteit PER BAR — de eenheid waarin barrières schalen.
    sigma_bar: pd.DataFrame
    #: Teken van het Phase 3-baselinesignaal: +1 / -1 / 0.
    side: pd.DataFrame
    #: Causale 30-daagse ADV in quote-eenheden.
    adv: pd.DataFrame
    #: De `data_hash` per gecertificeerde reeks die dit universum draagt.
    data_hashes: Mapping[str, str]
    annualisation_factor: float

    @property
    def symbols(self) -> list[str]:
        return list(self.prices.columns)

    @property
    def n_bars(self) -> int:
        return int(len(self.prices))

    def as_record(self) -> dict[str, Any]:
        return {
            "n_bars": self.n_bars,
            "symbols": self.symbols,
            "period_start": str(self.prices.index[0].date()),
            "period_end": str(self.prices.index[-1].date()),
            "data_hashes": dict(self.data_hashes),
        }


def _turnover_panel(
    store: PitStore,
    register: DataRegister,
    symbols: Sequence[str],
    index: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Turnover en de volledige OHLC-frames uit dezelfde gecertificeerde reeks."""
    turnover: dict[str, pd.Series] = {}
    frames: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        df, _ = load_certified_series(
            store, register, asset_class="crypto", dataset="ohlcv",
            symbol=symbol, granularity="1d")
        idx = pd.DatetimeIndex(
            pd.to_datetime(df["asof_ts_ns"].to_numpy(), unit="ns", utc=True),
            name=ASOF_INDEX_NAME)
        turnover[symbol] = pd.Series(
            df["turnover"].to_numpy(dtype="float64"), index=idx)
        frames[symbol] = pd.DataFrame(
            {c: df[c].to_numpy(dtype="float64")
             for c in ("open", "high", "low", "close", "volume")},
            index=idx,
        ).reindex(index)
    return pd.DataFrame(turnover), frames


def load_phase6_universe(
    root: Any,
    cfg: Mapping[str, Any],
    alpha_unit: Any,
    *,
    git_sha: str,
) -> Phase6Universe:
    """Laad het universum op het bruikbare venster van 1.743 bars.

    `cfg` is de mapping uit `backtest.baseline_report.load_baseline_configs`,
    zodat elk getal uit `conf/` komt en geen enkele drempel hier als literal
    staat.
    """
    store = PitStore(root / cfg["data"].pit_store_root)
    register = DataRegister(root / "artefacts/governance/data_hashes.json")
    symbols = list(cfg["data"].symbols)

    panel: CertifiedPanel = load_certified_close_panel(
        store, register, symbols=symbols, granularity="1d", asset_class="crypto")
    prices_full = panel.values

    sigma_full = ewma_volatility_panel(
        prices_full, lam=cfg["vol"].ewma_lambda,
        burn_in_bars=cfg["vol"].burn_in_bars,
        annualisation_factor=cfg["vol"].annualisation_factor)

    volume, ohlc_full = _turnover_panel(
        store, register, symbols, prices_full.index)
    volume = volume.reindex(prices_full.index)
    # Causaal: de turnover van bar t is pas op zijn close bekend.
    adv_full = volume.rolling(_ADV_WINDOW, min_periods=_ADV_WINDOW).mean().shift(1)

    usable = sigma_full.dropna(how="any").index
    usable = usable[usable.isin(adv_full.dropna(how="any").index)]
    require(
        len(usable) > 0,
        "Het bruikbare venster is leeg: er is geen bar waarop zowel de "
        "EWMA-volatiliteit als de causale ADV bestaat.",
        DataContractError,
        n_sigma=int(sigma_full.notna().all(axis=1).sum()),
        n_adv=int(adv_full.notna().all(axis=1).sum()),
    )

    features = alpha_unit.feature_pipeline().transform(panel, git_sha=git_sha)
    exposures = alpha_unit.generate(features).exposures.reindex(prices_full.index)

    prices = prices_full.loc[usable]
    sigma_annual = sigma_full.loc[usable]
    sigma_bar = sigma_annual / np.sqrt(cfg["vol"].annualisation_factor)
    side = np.sign(exposures.loc[usable]).fillna(0.0)

    return Phase6Universe(
        prices=prices,
        ohlc={s: f.loc[usable] for s, f in ohlc_full.items()},
        log_returns=np.log(prices / prices.shift(1)).dropna(how="any"),
        sigma_annual=sigma_annual,
        sigma_bar=sigma_bar,
        side=side,
        adv=adv_full.loc[usable],
        data_hashes=dict(panel.data_hashes),
        annualisation_factor=float(cfg["vol"].annualisation_factor),
    )
