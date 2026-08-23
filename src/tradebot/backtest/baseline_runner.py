# src/tradebot/backtest/baseline_runner.py
"""L10 - de baseline-keten uit audit sectie 21, end-to-end.

Phase 3, deliverable 8.

    PIT-store (L0/L1)
        -> causaal close-panel + data_hash
        -> L1 transforms (rolling log-return, cross-sectionele rang)
        -> L2 EWMA-volatiliteit (lambda = 0.94, causale burn-in)
        -> L4 AlphaUnit          -> a_t in [-1, +1]
        -> L8 allocator          -> gewichten
        -> L10 backtest          -> bruto en netto rendement

WALK-FORWARD OP EEN REGELGEBASEERDE BASELINE, EERLIJK BENOEMD
-------------------------------------------------------------
Deze baseline FIT niets. Er is geen parameter die uit de data wordt geschat: de
lookback, de skip en lambda komen alle uit `conf/`. Elke bar na de burn-in is
daarmee per constructie out-of-sample, en Purged Walk-Forward voegt in strikte
zin geen bescherming toe die er niet al was.

Hij wordt toch gedraaid, om twee redenen die het rapport ook noemt:

  1. **Vergelijkbaarheid.** Een later model MOET door Purged Walk-Forward. Als
     de baseline over het volledige venster wordt gemeten en het model over de
     testfolds, vergelijk je twee verschillende periodes en wint het model op
     de kalender in plaats van op zijn inhoud.
  2. **De embargo is niet vrijblijvend.** Rond elke foldgrens vallen
     `embargo_bars` bars weg. Dat is precies de behandeling die een gefit model
     krijgt, en de baseline hoort dezelfde bars te missen.

Wat walk-forward hier NIET doet, is een overfit-risico wegnemen dat er niet is.
Dat pretenderen zou het getal geloofwaardiger laten lijken dan het is.

DE `shift(1)`-CONVENTIE (RETAIN, audit sectie 24)
--------------------------------------------------
Een gewicht dat op `t` wordt bepaald, rendeert op `t+1`. In code:
`held = weights.shift(1)`, en `gross_t = sum_i held_{i,t} * ret_{i,t}`. De
turnover - en dus de kosten - vallen op de bar waarop de trade gebeurt.
Deze conventie komt ongewijzigd uit `alpha/xs_unit.py` en is een RETAIN-item.

Ref: ARCHITECTUUR_AUDIT_2026-08-22.md secties 11.1, 13.1, 16.1, 19, 21, 26.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..alpha.base import AlphaUnit
from ..cv.walk_forward import WalkForwardCV
from ..features.base import CertifiedPanel
from ..portfolio.equal_weight import equal_weight_long_only, sized_by_equal_weight
from ..portfolio.risk_parity import inverse_volatility_long_only, sized_by_risk_parity
from ..utils.failfast import DataContractError, require
from ..volatility.ewma import ewma_volatility_panel
from .metrics import calmar_ratio, max_drawdown, sharpe_ratio

__all__ = [
    "BASELINE_TRACKS",
    "BaselineResult",
    "CostModel",
    "build_weight_tracks",
    "run_baseline_tracks",
]

#: De vier vooraf geregistreerde tracks (zie de pre-registratie, planned_trials).
BASELINE_TRACKS: tuple[str, ...] = (
    "xs_momentum_risk_parity",
    "xs_momentum_equal_weight",
    "long_only_equal_weight",
    "long_only_risk_parity",
)


@dataclass(frozen=True)
class CostModel:
    """Voorlopige kostenaanname (Phase 3, stap 9).

    De definitieve eta-kalibratie op orderboekdata volgt in Phase 5. Tot die tijd
    wordt elk resultaat BRUTO EN NETTO gerapporteerd, met deze aanname expliciet
    vermeld. `is_provisional` staat in `conf/execution/fees.yaml` en wordt
    meegeschreven naar het rapport, zodat een lezer nooit hoeft te raden of het
    getal definitief is.
    """

    taker_fee_bps: float
    half_spread_bps: float
    is_provisional: bool

    @property
    def per_side(self) -> float:
        """Kosten per eenheid turnover, in decimalen."""
        return (self.taker_fee_bps + self.half_spread_bps) / 1e4

    def as_dict(self) -> dict[str, float | bool]:
        return {
            "taker_fee_bps": float(self.taker_fee_bps),
            "half_spread_bps": float(self.half_spread_bps),
            "per_side_decimal": float(self.per_side),
            "is_provisional": bool(self.is_provisional),
        }


@dataclass(frozen=True)
class BaselineResult:
    """Het resultaat van een enkele track, bruto en netto."""

    track: str
    gross_returns: pd.Series
    net_returns: pd.Series
    turnover: pd.Series
    weights: pd.DataFrame
    n_folds: int
    n_oos_bars: int
    n_embargoed_bars: int

    def metrics(self, *, bars_per_year: int) -> dict[str, float]:
        net = self.net_returns.to_numpy(dtype="float64")
        gross = self.gross_returns.to_numpy(dtype="float64")
        equity = np.cumprod(1.0 + net)
        mdd, _, _ = max_drawdown(equity)
        return {
            "gross_sharpe": float(sharpe_ratio(gross, bars_per_year=bars_per_year)),
            "net_sharpe": float(sharpe_ratio(net, bars_per_year=bars_per_year)),
            "net_total_return": float(equity[-1] - 1.0) if equity.size else 0.0,
            "max_drawdown": float(mdd),
            "calmar": float(calmar_ratio(equity, bars_per_year=bars_per_year)),
            "avg_daily_turnover": float(np.mean(self.turnover.to_numpy())),
            "n_oos_bars": float(self.n_oos_bars),
            "n_folds": float(self.n_folds),
        }


def build_weight_tracks(
    panel: CertifiedPanel,
    unit: AlphaUnit,
    *,
    lam: float,
    vol_burn_in_bars: int,
    annualisation_factor: float,
    gross_target: float,
    git_sha: str,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Bouw de gewichten van elke track, plus de exposures van de alpha-unit.

    `gross_target` is een NORMALISATIECONVENTIE (elke track wordt op dezelfde
    bruto-exposure geschaald zodat de vergelijking over sizing gaat en niet over
    hefboom), geen risicolimiet. De limiet hoort in L7 en komt in Phase 4.
    """
    features = unit.feature_pipeline().transform(panel, git_sha=git_sha)
    exposures = unit.generate(features).exposures
    vol = ewma_volatility_panel(
        panel.values, lam=lam, burn_in_bars=vol_burn_in_bars,
        annualisation_factor=annualisation_factor,
    )
    # Long-only referenties krijgen alleen een positie zodra hun vol bestaat,
    # zodat alle vier de tracks op dezelfde bars handelen en de vergelijking
    # niet over verschillende startdata gaat.
    available = panel.values.where(vol.notna())
    tracks = {
        "xs_momentum_risk_parity": sized_by_risk_parity(
            exposures.where(vol.notna()), vol, gross_target=gross_target),
        "xs_momentum_equal_weight": sized_by_equal_weight(
            exposures.where(vol.notna()), gross_target=gross_target),
        "long_only_equal_weight": equal_weight_long_only(
            available, gross_target=gross_target),
        "long_only_risk_parity": inverse_volatility_long_only(
            vol, gross_target=gross_target),
    }
    require(
        set(tracks) == set(BASELINE_TRACKS),
        "De gebouwde tracks wijken af van de vooraf geregistreerde verzameling.",
        DataContractError,
        built=sorted(tracks),
        registered=sorted(BASELINE_TRACKS),
    )
    return tracks, exposures


def _oos_mask(
    n: int, *, train_bars: int, test_bars: int, embargo_bars: int, min_splits: int
) -> tuple[np.ndarray, int, int]:
    """Het OOS-masker van Purged Walk-Forward, plus fold- en embargotelling.

    De stap is `test_bars`: de testvensters sluiten dan precies op elkaar aan,
    zonder overlap en zonder gaten. Dat is geen vrije parameter maar wat een
    partitie van de tijdas betekent; `n_splits` uit `conf/validation/` fungeert
    als MINIMUM en de realisatie wordt eraan getoetst.
    """
    require(
        n > train_bars + test_bars,
        "Te weinig bars voor een enkele walk-forward fold.",
        DataContractError,
        n_bars=n, train_bars=train_bars, test_bars=test_bars,
    )
    cv = WalkForwardCV(
        train_size=train_bars, test_size=test_bars, step=test_bars,
        mode="rolling", min_train=train_bars, embargo_bars=embargo_bars,
    )
    mask = np.zeros(n, dtype=bool)
    n_folds = 0
    for fold in cv.split(n):
        # De embargo hoort AAN BEIDE kanten van de grens: de bars direct na de
        # overgang van train naar test dragen informatie die met de trainperiode
        # overlapt. Ze vallen dus uit de OOS-reeks.
        start = int(fold.test_idx[0]) + embargo_bars
        end = int(fold.test_idx[-1]) + 1
        if start < end:
            mask[start:end] = True
        n_folds += 1
    require(
        n_folds >= min_splits,
        "Minder walk-forward folds dan `n_splits` uit conf/validation/. De "
        "gecertificeerde reeks is te kort voor het geconfigureerde schema.",
        DataContractError,
        realised_folds=n_folds, required=min_splits,
    )
    embargoed = n_folds * embargo_bars
    return mask, n_folds, embargoed


def run_baseline_tracks(
    panel: CertifiedPanel,
    weight_tracks: Mapping[str, pd.DataFrame],
    *,
    cost: CostModel,
    train_bars: int,
    test_bars: int,
    embargo_bars: int,
    min_splits: int,
    warmup_bars: int,
) -> dict[str, BaselineResult]:
    """Draai elke track door Purged Walk-Forward en reken bruto en netto af.

    `warmup_bars` is de gezamenlijke burn-in van de keten; die bars bestaan niet
    als handelsdag en tellen niet mee in de walk-forward indeling.
    """
    require(
        len(weight_tracks) > 0,
        "Geen enkele track om te draaien.",
        DataContractError,
    )
    prices = panel.values
    returns = prices.pct_change(fill_method=None)
    tradable = prices.index[warmup_bars:]
    mask, n_folds, embargoed = _oos_mask(
        len(tradable), train_bars=train_bars, test_bars=test_bars,
        embargo_bars=embargo_bars, min_splits=min_splits,
    )
    oos_index = tradable[mask]

    results: dict[str, BaselineResult] = {}
    for name, weights in weight_tracks.items():
        require(
            bool(weights.index.equals(prices.index)),
            "Een track staat niet op de tijdas van het panel.",
            DataContractError, track=name,
        )
        w = weights.fillna(0.0)
        # RETAIN (sectie 24): het gewicht van t-1 verdient het rendement van t.
        held = w.shift(1).fillna(0.0)
        gross = (held * returns.fillna(0.0)).sum(axis=1)
        turnover = (w - held).abs().sum(axis=1)
        net = gross - turnover * cost.per_side
        results[name] = BaselineResult(
            track=name,
            gross_returns=gross.loc[oos_index],
            net_returns=net.loc[oos_index],
            turnover=turnover.loc[oos_index],
            weights=w.loc[oos_index],
            n_folds=n_folds,
            n_oos_bars=int(len(oos_index)),
            n_embargoed_bars=int(embargoed),
        )
    return results


def returns_matrix(
    results: Mapping[str, BaselineResult], tracks: Sequence[str]
) -> np.ndarray:
    """Netto-rendementen als (T, S)-matrix voor de SPA-toets."""
    require(
        len(tracks) > 0,
        "Geen tracks opgegeven voor de SPA-matrix.",
        DataContractError,
    )
    columns = [results[t].net_returns.to_numpy(dtype="float64") for t in tracks]
    lengths = {len(c) for c in columns}
    require(
        len(lengths) == 1,
        "De tracks hebben een verschillend aantal OOS-bars; ze zijn niet op "
        "dezelfde folds gemeten en dus niet vergelijkbaar.",
        DataContractError,
        lengths=sorted(lengths),
    )
    return np.column_stack(columns)
