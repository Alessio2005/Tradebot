# src/tradebot/validation/phase11_breadth_measurement.py
"""Fase 11, breedte en tijdschaal — de meetcampagne van stages A en B.

WAT HIER WOONT EN WAT NIET
==========================
`validation/breadth.py` bevat de DEFINITIES (constructies, onafhankelijke
weddenschappen, het ontwerpeffect onder zijn eigen naam) en
`validation/signal_clock.py` de klok van een besluitpaneel. Dit bestand zet ze
over de vensters en de jaren heen en maakt er één artefact van. Er staat hier
geen nieuwe statistiek (R-3).

DE VENSTERS WORDEN AFGELEID, NIET GEDEFINIEERD
==============================================
`W_FULL` is het bruikbare venster: de bars waarop elke naam een ex-ante
volatiliteit en een causale ADV heeft — dezelfde conjunctie als H-10.1.
`W_DEV` is `validation/holdout.py::development_slice` daarop, met de bevroren
split. `W_GATE` is het complement binnen `W_FULL`. Het poortsample wordt dus
NIET via `gate_slice` gelezen: er wordt geen hypothese getoetst en er wordt
geen eerste moment berekend, alleen correlaties (R-15 van de fase-opdracht).
Tweede momenten over het volle venster hebben precedent in
`phase10_state_diagnostics.json` en in de haltteller van H-10.1.

GEEN RISICOBESLUIT
==================
Niets hier gaat door de risicolaag. Het artefact draagt daarom geen
`risk_policy_hash` van een beleid, maar zegt expliciet waarom niet.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..backtest.vectorized import run_vectorized
from ..utils.failfast import DataContractError, require
from .breadth import BreadthConfig, breadth_measurement, construct
from .holdout import development_slice
from .signal_clock import decision_panel_clock

__all__ = ["baseline_weight_tracks", "measure_breadth", "measure_signal_clock", "split_windows"]


def split_windows(returns: pd.DataFrame, *, lock_path: Path) -> dict[str, pd.DataFrame]:
    """`W_FULL`, `W_DEV` en `W_GATE` op het aangeleverde bruikbare venster."""
    require(
        bool(np.isfinite(returns.to_numpy(dtype=np.float64)).all()),
        "Het rendementspaneel over het bruikbare venster bevat een gat. W_FULL "
        "is per constructie gebalanceerd (MEASUREMENT_CONTRACT §2); een gat "
        "betekent dat het venster verkeerd is gesneden.",
        DataContractError,
    )
    development = development_slice(returns, lock_path=lock_path)
    gate = returns.loc[returns.index.difference(development.index)]
    return {"W_DEV": development, "W_GATE": gate, "W_FULL": returns}


def _window_meta(panel: pd.DataFrame) -> dict[str, Any]:
    return {"n_obs": int(len(panel)), "first": str(panel.index[0].date()),
            "last": str(panel.index[-1].date())}


def measure_breadth(
    windows: dict[str, pd.DataFrame],
    *,
    cfg: BreadthConfig,
    n_boot: int,
    seed: int,
    ci_level: float,
    block_length: int | None,
) -> dict[str, Any]:
    """Elke constructie op elk venster, en op elk kalenderjaar van `W_FULL`."""
    kwargs = {"n_boot": n_boot, "seed": seed, "ci_level": ci_level,
              "block_length": block_length}
    rows = [
        {"window": name, **breadth_measurement(
            construct(panel, construction), construction=construction, **kwargs
        ).to_dict()}
        for name, panel in windows.items() for construction in cfg.constructions
    ]
    full = windows["W_FULL"]
    per_year = [
        {"year": int(year), **breadth_measurement(
            construct(group, construction), construction=construction, **kwargs
        ).to_dict()}
        for year, group in full.groupby(full.index.year)
        for construction in cfg.constructions
    ]
    return {
        "windows": {name: _window_meta(panel) for name, panel in windows.items()},
        "rows": rows,
        "per_year": per_year,
        "quantities": {
            "independent_bets": "participatieratio van de eigenwaarden, "
                                "validation/breadth.py::independent_bets (AD-30)",
            "design_effect": "Kish, validation/inference.py::effective_breadth; "
                             "GEEN breedte (DI-35)",
        },
        "beta_hedged_ew_note": "bèta op het volle venster van elke rij geschat: "
                               "diagnostiek, geen besluitgrootheid",
        "first_moments_computed": False,
    }


def baseline_weight_tracks(root: Path, base: dict[str, Any], *, git_sha: str) -> dict[str, pd.DataFrame]:
    """De gewichtspanelen van de vier baseline-tracks, precies zoals H-10.1 ze bouwt.

    Dezelfde loaders en dezelfde unit als `apps/run_h10_1_decision_frequency.py`,
    zodat de klok van stap 3 op dezelfde panelen wordt gemeten als het
    vasthouden van H-10.1.
    """
    from ..alpha.momentum import build_cross_sectional_momentum
    from ..backtest.baseline_runner import build_weight_tracks
    from ..data.pit_store import PitStore
    from ..features.base import DataRegister, load_certified_close_panel

    register = DataRegister(root / "artefacts/governance/data_hashes.json")
    panel = load_certified_close_panel(
        PitStore(root / base["data"].pit_store_root), register,
        symbols=list(base["data"].symbols), granularity="1d", asset_class="crypto")
    tracks, _ = build_weight_tracks(
        panel, build_cross_sectional_momentum(base["alpha"]),
        lam=base["vol"].ewma_lambda, vol_burn_in_bars=base["vol"].burn_in_bars,
        annualisation_factor=base["vol"].annualisation_factor,
        gross_target=1.0, git_sha=git_sha)
    return tracks


def measure_signal_clock(
    tracks: dict[str, pd.DataFrame],
    *,
    prices: pd.DataFrame,
    usable: pd.Index,
    development_index: pd.Index,
    cfg: BreadthConfig,
    bars_per_year: float,
) -> dict[str, Any]:
    """De klok van elk besluitpaneel op `W_DEV` (stap 3). Geen rendement.

    Het besluitpaneel is `weights.loc[usable].fillna(0.0)`, zoals H-10.1 het
    vasthield. De omzet komt uit `run_vectorized` met kosten nul: de enige
    omzetdefinitie (`backtest/vectorized.py`), inclusief de instapbar.
    """
    rows: dict[str, Any] = {}
    for name in sorted(tracks):
        decision = tracks[name].loc[usable].fillna(0.0)
        development = decision.loc[decision.index.isin(development_index)]
        turnover = run_vectorized(development, prices.loc[development.index],
                                  cost_per_side=0.0, initial_equity=1.0).turnover
        rows[name] = decision_panel_clock(
            development, turnover=turnover, window_c=cfg.signal_clock.iact_window_c,
            max_lag=cfg.signal_clock.iact_max_lag, bars_per_year=bars_per_year,
        ).to_dict()
    return {
        "window": "W_DEV",
        "tracks": rows,
        "estimator": {"iact_window_c": cfg.signal_clock.iact_window_c,
                      "iact_max_lag": cfg.signal_clock.iact_max_lag},
        "turnover_source": "backtest/vectorized.py::run_vectorized, cost_per_side=0, "
                           "tweezijdig, instapbar inbegrepen",
    }
