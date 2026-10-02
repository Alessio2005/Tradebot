"""Wat een 1:1-trade met stops bij de exchange werkelijk oplevert (spec §6, §17.6).

`vol_barriers.label_triple_barrier` bepaalt WELKE barrière eerst raakt en op welke
bar. Zijn `realized_return` rekent met de slotkoers van die bar. Een stop of
take-profit bij de exchange vult op het barrièreniveau -- of op de open, als de
bar er met een gat voorbij opent. Een stop vult bovendien als marktorder, met
slippage tegen de positie in. Deze module rekent die fill uit, en zet het doel
op "netto na de round trip positief".
"""
from __future__ import annotations

import dataclasses

import numpy as np

from ..schemas.config import ExecutionConfig, LabelingConfig
from ..utils.failfast import DataContractError, require
from .vol_barriers import BarrierLabels

__all__ = ["barrier_fill_returns", "round_trip_cost", "select_events",
           "with_cost_aware_target"]


def round_trip_cost(exec_cfg: ExecutionConfig) -> float:
    """Eén definitie van de round trip: twee keer taker plus halve spread."""
    return 2.0 * (exec_cfg.taker_fee_bps + exec_cfg.assumed_half_spread_bps) / 1e4


def barrier_fill_returns(
    labels: BarrierLabels,
    open_: np.ndarray,
    close: np.ndarray,
    cfg: LabelingConfig,
    *,
    stop_slippage_bps: float,
) -> np.ndarray:
    """Positierendement per event bij een fill op de barrière (richting verrekend)."""
    open_ = np.asarray(open_, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    require(open_.size == close.size, "open en close zijn niet even lang.", DataContractError)
    lag = int(cfg.entry_lag_bars)
    slip = float(stop_slippage_bps) / 1e4
    out = np.empty(len(labels), dtype=np.float64)
    for i in range(len(labels)):
        s = float(labels.side[i])
        entry = float(close[int(labels.event_idx[i]) + lag])
        j = int(labels.exit_idx[i])
        outcome = int(labels.barrier_outcome[i])
        if outcome == 0:
            fill = float(close[j])
        else:
            o = float(open_[j])
            require(np.isfinite(o), "Een barrière-exit op een bar zonder open.",
                    DataContractError, exit_idx=j)
            if outcome == 1:
                level = entry * (1.0 + s * cfg.profit_target_sigma * float(labels.sigma[i]))
                gapped = o >= level if s > 0 else o <= level
                fill = o if gapped else level
            else:
                level = entry * (1.0 - s * cfg.stop_loss_sigma * float(labels.sigma[i]))
                gapped = o <= level if s > 0 else o >= level
                fill = (o if gapped else level) * (1.0 - s * slip)
        out[i] = s * (fill / entry - 1.0)
    return out


def with_cost_aware_target(
    labels: BarrierLabels, fill_returns: np.ndarray, *, round_trip_cost: float,
) -> BarrierLabels:
    """Labels met het fillrendement en het doel "netto na de round trip > 0"."""
    fill_returns = np.asarray(fill_returns, dtype=np.float64)
    require(fill_returns.size == len(labels), "Een fillrendement per event.",
            DataContractError, n=fill_returns.size, n_events=len(labels))
    return dataclasses.replace(
        labels,
        realized_return=fill_returns,
        meta_label=((fill_returns - float(round_trip_cost)) > 0.0).astype(np.int8),
    )


def select_events(labels: BarrierLabels, mask: np.ndarray) -> BarrierLabels:
    """Dezelfde labels, beperkt tot `mask`, met elk veld in de pas."""
    m = np.asarray(mask, dtype=bool)
    require(m.size == len(labels), "Het masker heeft niet één waarde per event.",
            DataContractError)
    return BarrierLabels(**{f.name: getattr(labels, f.name)[m]
                            for f in dataclasses.fields(labels)})
