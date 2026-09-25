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

from ..utils.failfast import DataContractError, require
from .breadth import BreadthConfig, breadth_measurement, construct
from .holdout import development_slice

__all__ = ["measure_breadth", "split_windows"]


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
