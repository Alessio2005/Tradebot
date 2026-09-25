# src/tradebot/validation/phase11_breadth_measurement.py
"""Fase 11, breedte en tijdschaal — de meetcampagne van stages A t/m D.

WAT HIER WOONT EN WAT NIET
==========================
`validation/breadth.py` bevat de DEFINITIES (constructies, onafhankelijke
weddenschappen, het ontwerpeffect onder zijn eigen naam) en
`validation/signal_clock.py` de klok van een besluitpaneel, en
`validation/phase11_decision_clock_feasibility.py` de poort van stap 7. Dit
bestand zet ze over de vensters en de jaren heen en maakt er één artefact van.
Er staat hier geen nieuwe statistiek (R-3).

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

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..backtest.baseline_runner import CostModel
from ..backtest.vectorized import run_vectorized
from ..utils.failfast import DataContractError, require
from .breadth import (
    BreadthConfig,
    breadth_measurement,
    construct,
    dsr_hurdle,
    required_ic,
    simulate_wall,
    t_hurdle_sharpe,
)
from .holdout import development_slice
from .phase11_decision_clock_feasibility import (
    ladder_cost_mix,
    measure_feasibility,
    policy_book_scales,
)
from .signal_clock import decision_panel_clock

__all__ = [
    "baseline_weight_tracks",
    "build_artefact",
    "k_star_from_clock",
    "measure_breadth",
    "measure_signal_clock",
    "measure_wall",
    "split_windows",
]


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
    kwargs: dict[str, Any] = {"n_boot": n_boot, "seed": seed, "ci_level": ci_level,
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


def measure_wall(
    windows: dict[str, pd.DataFrame],
    breadth: dict[str, Any],
    *,
    cfg: BreadthConfig,
    bars_per_year: float,
    m_new: int,
    dsr_target: float,
) -> dict[str, Any]:
    """De IC-muur per constructie en horizon, gecontroleerd door simulatie (stap 5).

    De drempels gelden onder normaliteit (scheefheid 0, kurtosis 3,
    `sr_variance = 1/n_obs`); een hypothese rekent haar drempel voor haar eigen
    reeks opnieuw uit. De simulatie gebruikt alleen de covariantie van `W_DEV`,
    geen enkel gemiddeld rendement. Beslisregel, vooraf in
    `conf/research/breadth.yaml`: ligt de gesimuleerde Sharpe over het hele
    IC-rooster binnen `formula_tolerance` van de wet, dan is de formule de muur;
    anders wordt de formule gecorrigeerd met de mediane verhouding.
    """
    dev, gate = windows["W_DEV"], windows["W_GATE"]
    n_dev, n_gate = len(dev), len(gate)
    hurdles = {
        "W_DEV": {
            "n_obs": n_dev,
            "dsr": dsr_hurdle(n_obs=n_dev, n_trials=m_new, skew=0.0, kurtosis=3.0,
                              sr_variance=1.0 / n_dev, bars_per_year=bars_per_year,
                              dsr_target=dsr_target),
            "t": t_hurdle_sharpe(n_obs=n_dev, bars_per_year=bars_per_year,
                                 t=cfg.wall.t_hurdle),
        },
        "W_GATE": {"n_obs": n_gate,
                   "t": t_hurdle_sharpe(n_obs=n_gate, bars_per_year=bars_per_year,
                                        t=cfg.wall.t_hurdle)},
    }
    bets = {(row["window"], row["construction"]): row["independent_bets"]
            for row in breadth["rows"]}
    constructions: dict[str, Any] = {}
    for construction in cfg.constructions:
        covariance = construct(dev, construction).cov().to_numpy()
        sims = [simulate_wall(covariance, construction=construction, ic=ic,
                              n_obs=cfg.wall.simulation_n_obs,
                              seed=cfg.wall.simulation_seed,
                              bars_per_year=bars_per_year).to_dict()
                for ic in cfg.wall.simulation_ic_grid]
        ratios = [sim["ratio"] for sim in sims]
        holds = all(abs(r - 1.0) <= cfg.wall.formula_tolerance for r in ratios)
        correction = 1.0 if holds else float(np.median(ratios))
        table = []
        for h in cfg.wall.horizons_bars:
            dev_bets = bets[("W_DEV", construction)]
            gate_bets = bets[("W_GATE", construction)]
            formula = {
                "dsr_W_DEV": required_ic(hurdles["W_DEV"]["dsr"], independent_bets=dev_bets,
                                         horizon_bars=h, bars_per_year=bars_per_year),
                "t_W_DEV": required_ic(hurdles["W_DEV"]["t"], independent_bets=dev_bets,
                                       horizon_bars=h, bars_per_year=bars_per_year),
                "t_W_GATE": required_ic(hurdles["W_GATE"]["t"], independent_bets=gate_bets,
                                        horizon_bars=h, bars_per_year=bars_per_year),
            }
            table.append({"horizon_bars": int(h), "formula": formula,
                          "wall": {k: v / correction for k, v in formula.items()},
                          "bets_per_year_W_DEV": dev_bets * bars_per_year / h})
        constructions[construction] = {
            "simulations": sims, "ratio_min": min(ratios), "ratio_max": max(ratios),
            "formula_holds": holds, "correction": correction, "table": table,
        }
    return {
        "hurdles": hurdles,
        "hurdle_assumptions": "normaliteit: scheefheid 0, kurtosis 3, sr_variance = 1/n_obs",
        "m_new": int(m_new),
        "dsr_target": float(dsr_target),
        "formula_tolerance": cfg.wall.formula_tolerance,
        "forecast_noise_model": "onafhankelijk over de namen (zie simulate_wall)",
        "constructions": constructions,
    }


def k_star_from_clock(row: dict[str, Any]) -> int:
    """k* = ⌈τ_int⌉ van het besluitpaneel, alleen als die mediaan een meting is.

    Een naam waarvan het Sokal-venster niet is bereikt, mag boven de mediaan
    liggen (`signal_clock.median_is_determined`); een mediaan die zelf een
    ondergrens is, levert geen k*.
    """
    tau = float(row["tau_int_median"])
    require(math.isfinite(tau) and bool(row["tau_median_determined"]),
            "De signaalklok van de primaire cel is niet eindig, of haar mediaan "
            "hangt af van een afgekapte naam; dan is er geen k* en geen "
            "hypothese om te toetsen.", DataContractError, tau=tau)
    return int(math.ceil(tau))


def build_artefact(
    root: Path,
    *,
    market: dict[str, Any],
    base: dict[str, Any],
    cfg: BreadthConfig,
    inference: Any,
    lock_path: Path,
    m_new: int,
    current_policy_hash: str,
    git_sha: str,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """Stage A t/m D in één artefact: breedte, signaalklok, muur en de poort van stap 7."""
    usable = market["sigma"].dropna(how="any").index
    usable = usable[usable.isin(market["adv"].dropna(how="any").index)]
    windows = split_windows(np.log(market["prices"]).diff().loc[usable], lock_path=lock_path)
    bars_per_year = float(base["bt"].bars_per_year)
    breadth = measure_breadth(windows, cfg=cfg, n_boot=inference.n_boot, seed=inference.seed,
                              ci_level=inference.ci_level, block_length=inference.block_length)
    tracks = baseline_weight_tracks(root, base, git_sha=git_sha)
    clock = measure_signal_clock(tracks, prices=market["prices"], usable=usable,
                                 development_index=windows["W_DEV"].index, cfg=cfg,
                                 bars_per_year=bars_per_year)
    wall = measure_wall(windows, breadth, cfg=cfg, bars_per_year=bars_per_year,
                        m_new=m_new, dsr_target=1.0 - float(base["val"].dsr_alpha))
    feas = cfg.feasibility
    ladder = {**ladder_cost_mix(root / feas.ladder_artefact, track=feas.primary_track),
              "artefact": feas.ladder_artefact}
    cost = CostModel(taker_fee_bps=base["exec"].taker_fee_bps,
                     half_spread_bps=base["exec"].assumed_half_spread_bps,
                     is_provisional=base["exec"].cost_assumption_is_provisional)
    feasibility = measure_feasibility(
        tracks, prices=market["prices"], usable=usable, development=windows["W_DEV"],
        cfg=feas, k_star=k_star_from_clock(clock["tracks"][feas.primary_track]),
        cost_per_side=cost.per_side, ladder=ladder,
        scales=policy_book_scales(
            root / "artefacts/governance/risk_config_registry.json",
            ladder_hash=ladder["risk_policy_hash"], current_hash=current_policy_hash,
            ladder_mean_gross=ladder["mean_gross_l1"]),
        current_policy_hash=current_policy_hash,
        sr_level_required=wall["hurdles"]["W_DEV"]["dsr"], bars_per_year=bars_per_year,
        ci_level=inference.ci_level)
    return {
        "git_sha": git_sha,
        **provenance,
        "inference": {"n_boot": inference.n_boot, "seed": inference.seed,
                      "ci_level": inference.ci_level, "block_length": inference.block_length},
        "risk_policy_hash": None,
        "risk_policy_note": "geen risicobesluit: alles hier is L0/data (AD-27 n.v.t.); "
                            "de poort van stap 7 citeert de ladder met haar eigen beleid",
        "trials": 0,
        "breadth": breadth,
        "signal_clock": clock,
        "wall": wall,
        "feasibility": feasibility,
    }
