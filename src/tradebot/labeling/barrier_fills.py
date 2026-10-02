"""Fills van een 1:1-trade volgens de dagdata-executieconventie (spec §12).

`vol_barriers.label_triple_barrier` bepaalt WELKE barrière eerst raakt en op
welke bar. Zijn `realized_return` rekent met de slotkoers van die bar. De
conventie van Plan 1 vult een target op het barrièreniveau (of op de open als
de bar er met een gat voorbij opent), een stop op het niveau of de open,
maal (1 ∓ slippage) tegen de positie in, en de verticale barrière op de close.
Het doel wordt "netto na de trade-specifieke kosten van spec §6.2a positief".
"""
from __future__ import annotations

import dataclasses

import numpy as np

from ..schemas.config import LabelingConfig
from ..utils.failfast import DataContractError, require
from .vol_barriers import BarrierLabels

__all__ = ["barrier_fill_returns", "select_events", "with_cost_aware_target"]


def barrier_fill_returns(
    labels: BarrierLabels,
    open_: np.ndarray,
    close: np.ndarray,
    cfg: LabelingConfig,
    *,
    stop_slippage_bps: float,
) -> np.ndarray:
    """Positierendement per event bij een fill volgens spec §12 (richting verrekend)."""
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
            require(bool(np.isfinite(o)), "Een barrière-exit op een bar zonder open.",
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
    labels: BarrierLabels, fill_returns: np.ndarray, *, costs: np.ndarray,
) -> BarrierLabels:
    """Labels met het fillrendement en het doel `fill - C_i^label > 0` (spec §6.1)."""
    fill_returns = np.asarray(fill_returns, dtype=np.float64)
    costs = np.asarray(costs, dtype=np.float64)
    require(fill_returns.size == costs.size == len(labels), "Eén fill en één kost per event.",
            DataContractError, n_fill=fill_returns.size, n_cost=costs.size, n_events=len(labels))
    return dataclasses.replace(
        labels,
        realized_return=fill_returns,
        meta_label=((fill_returns - costs) > 0.0).astype(np.int8),
    )


def select_events(labels: BarrierLabels, mask: np.ndarray) -> BarrierLabels:
    """Dezelfde labels, beperkt tot `mask`, met elk veld in de pas."""
    m = np.asarray(mask, dtype=bool)
    require(m.size == len(labels), "Het masker heeft niet één waarde per event.",
            DataContractError)
    return BarrierLabels(**{f.name: getattr(labels, f.name)[m]
                            for f in dataclasses.fields(labels)})
