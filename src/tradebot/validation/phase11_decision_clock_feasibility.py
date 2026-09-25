# src/tradebot/validation/phase11_decision_clock_feasibility.py
"""Fase 11, breedte en tijdschaal — de haalbaarheidspoort van H-11.2 (stap 7).

WAT DEZE MODULE DOET, EN WAT NIET
=================================
Zij beslist VÓÓR registratie of H-11.2 ("de beslisklok volgt de signaalklok")
op deze sample iets kan meten. Zij meet geen enkel echt gemiddeld rendement en
kost dus geen trial. Drie ingrediënten:

1. **De detectiegrens.** Synthetische rendementen MET GEMIDDELDE NUL uit de
   covariantie van `W_DEV`, beide gewichtspanelen erop, en de verschiltoets
   van de kern (`inference.sharpe_difference_test`). De spreiding van het
   verschil onder de nulhypothese is de standaardfout die de echte toets zou
   gebruiken.
2. **De kostenwinst bij nul signaalverval.** Omzetdaling maal kosten per zijde,
   geannualiseerd, gedeeld door de boekvolatiliteit. Omzet komt uit
   `backtest/vectorized.py::run_vectorized`, de kosten per zijde uit
   `conf/execution/fees.yaml` (R-3).
3. **De L3-kostenmix en hoe zij met de boekgrootte schaalt.** Het impactmodel is
   de wortelwet (`execution/impact_model.py`): impact per eenheid schaalt met
   de wortel van de grootte, fees en spread niet. Een groter boek onder AD-26
   vergroot dus alleen het impactdeel.

Groen dan en slechts dan wanneer de kostenwinst bij nul signaalverval, op de
L3-kostenmix, groter is dan de detectiegrens. Hoe dat uitvalt, is een meting
en staat niet hier.

DE VOORWAARDE VAN STAP 7, EN WAT ZONDER HAAR AL VASTSTAAT
========================================================
De poort hoort te staan op de ladder onder het GELDENDE risicobeleid
(`artefacts/baseline/phase11_revaluation.json`, van de zusterfase). Zolang die
er niet is, rekent deze module op de ladder die er wel is, en vraagt zij het
beleidsregister hoeveel groter het boek onder het geldende beleid kan zijn: de
verhouding van de vol-targets (de verwachte schaal) en de bruto-cap gedeeld
door het gemeten boek (de bovengrens). Een rood dat ook bij die bovengrens rood
blijft, kan geen nieuwe ladder meer omdraaien. Elke andere uitkomst wacht op
stap 7.1 (`PENDING_7_1`).
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from ..backtest.vectorized import run_vectorized
from ..portfolio.decision_frequency import breakeven_cost_bps, hold_decision
from ..utils.failfast import DataContractError, require
from .breadth import FeasibilityConfig
from .inference import sharpe_difference_test

__all__ = [
    "cost_gain_sharpe",
    "gate_verdict",
    "impact_multiplier",
    "ladder_cost_mix",
    "measure_feasibility",
    "null_detection_limit",
    "policy_book_scales",
]

_COST_LAYER = "L3_execution"
_BOOK_LAYER = "L1_sovereign"
_COSTS = ("fees", "impact", "spread", "funding")


def null_detection_limit(
    reference: pd.DataFrame,
    held: pd.DataFrame,
    covariance: np.ndarray[Any, Any],
    *,
    n_paths: int,
    seed: int,
    n_boot: int,
    bars_per_year: float,
    ci_level: float,
) -> dict[str, Any]:
    """Het kleinste Sharpe-verschil dat de kern op deze sample kan onderscheiden.

    Beide panelen worden uitgevoerd volgens de vectorized-conventie
    (`held = weights.shift(1)`). De standaardfout is de analytische
    Ledoit-Wolf-SE van `sharpe_difference_test`; `n_boot` bepaalt alleen het
    interval dat de kern daarnaast altijd bootstrapt, en dat interval wordt hier
    niet gebruikt.
    """
    require(
        reference.index.equals(held.index) and list(reference.columns) == list(held.columns),
        "Beide gewichtspanelen moeten op dezelfde bars en namen liggen.",
        DataContractError,
    )
    cov = np.asarray(covariance, dtype=np.float64)
    require(cov.shape == (reference.shape[1], reference.shape[1]),
            "De covariantie hoort bij de namen van het paneel.", DataContractError,
            shape=tuple(cov.shape), n_names=reference.shape[1])
    executed_a = reference.shift(1).fillna(0.0).to_numpy(dtype=np.float64)
    executed_b = held.shift(1).fillna(0.0).to_numpy(dtype=np.float64)
    eigenvalues, vectors = np.linalg.eigh(0.5 * (cov + cov.T))
    loading = vectors * np.sqrt(np.clip(eigenvalues, 0.0, None))
    rng = np.random.default_rng(int(seed))
    deltas: list[float] = []
    ses: list[float] = []
    for path in range(int(n_paths)):
        returns = rng.standard_normal(executed_a.shape) @ loading.T
        result = sharpe_difference_test(
            (executed_b * returns).sum(axis=1), (executed_a * returns).sum(axis=1),
            bars_per_year=float(bars_per_year), n_boot=int(n_boot), seed=int(seed) + path)
        deltas.append(float(result.delta_sharpe))
        ses.append(float(result.se))
    median_se = float(np.median(ses))
    two_sided = float(norm.ppf(1.0 - (1.0 - float(ci_level)) / 2.0))
    one_sided = float(norm.ppf(float(ci_level)))
    return {
        "n_paths": int(n_paths),
        "mean_delta": float(np.mean(deltas)),
        "sd_delta": float(np.std(deltas, ddof=1)),
        "median_se": median_se,
        "detectable_two_sided": two_sided * median_se,
        "detectable_one_sided": one_sided * median_se,
        "detectable_two_sided_from_sd": two_sided * float(np.std(deltas, ddof=1)),
    }


def cost_gain_sharpe(
    *,
    turnover_reference: float,
    turnover_held: float,
    cost_per_side: float,
    book_volatility: float,
    bars_per_year: float,
) -> float:
    """De Sharpe-winst uit kosten alleen, bij nul signaalverval."""
    require(
        0.0 <= turnover_held <= turnover_reference and book_volatility > 0.0
        and cost_per_side >= 0.0,
        "Vasthouden kan de omzet alleen verlagen, en de boekvolatiliteit moet "
        "positief zijn.",
        DataContractError, turnover_reference=turnover_reference,
        turnover_held=turnover_held, book_volatility=book_volatility,
    )
    return float((turnover_reference - turnover_held) * cost_per_side * bars_per_year
                 / book_volatility)


def impact_multiplier(*, fees: float, impact: float, spread: float, scale: float) -> float:
    """`(fees + impact * sqrt(scale) + spread) / (fees + spread)`.

    De verhouding tussen de omzetkosten op L3 en die op de L0-kostenas (fees
    plus halve spread), bij een boek dat `scale` keer zo groot is als het boek
    waarop `impact` is gemeten. Fees en spread zijn lineair in de grootte en
    vallen in Sharpe-eenheden weg; impact per eenheid schaalt met de wortel.
    """
    require(fees + spread > 0.0 and impact >= 0.0 and scale > 0.0,
            "Kosten en schaal moeten positief zijn.", DataContractError,
            fees=fees, impact=impact, spread=spread, scale=scale)
    return float(1.0 + impact * math.sqrt(scale) / (fees + spread))


def ladder_cost_mix(path: Path, *, track: str) -> dict[str, Any]:
    """De L3-kostenmix en het L1-boek van één track, uit een ladderartefact."""
    rows = json.loads(Path(path).read_text(encoding="utf-8"))["rows"]
    layers = {row["layer"]: row for row in rows if row.get("track") == track}
    require(_COST_LAYER in layers and _BOOK_LAYER in layers,
            f"Het ladderartefact draagt voor {track!r} geen {_COST_LAYER} en "
            f"{_BOOK_LAYER}; zonder die twee is er geen kostenmix en geen boek.",
            DataContractError, path=str(path), layers=sorted(layers))
    cost, book = layers[_COST_LAYER], layers[_BOOK_LAYER]
    mix = {name: float(cost.get(f"cost_{name}", math.nan)) for name in _COSTS}
    require(all(math.isfinite(v) and v >= 0.0 for v in mix.values()),
            "Een kostenpost op L3 ontbreekt of is niet eindig; een poort op een "
            "halve kostenmix zou de omzetkosten onderschatten.",
            DataContractError, track=track, costs=mix)
    return {"artefact": str(path), "track": track,
            "risk_policy_hash": str(cost["risk_policy_hash"]), **mix,
            "net_sharpe_l3_full_window": float(cost["net_sharpe"]),
            "mean_gross_l1": float(book["mean_gross"])}


def policy_book_scales(
    registry_path: Path, *, ladder_hash: str, current_hash: str, ladder_mean_gross: float,
) -> dict[str, float]:
    """Hoeveel groter het boek onder het geldende beleid kan zijn dan op de ladder.

    `vol_target`: de verhouding van de vol-targets, de schaal zolang de
    vol-target bindt. `gross_cap_bound`: de bruto-cap van het geldende beleid
    gedeeld door het gemeten L1-boek, de grootste schaal die het beleid toelaat.
    """
    entries = {entry["config_hash"]: entry["audit_header"]
               for entry in json.loads(Path(registry_path).read_text(encoding="utf-8"))["entries"]}
    require(ladder_hash in entries and current_hash in entries,
            "Beide beleidsregels moeten in het register staan; een schaal tegen "
            "een onbekend beleid is een gok.", DataContractError,
            ladder_hash=ladder_hash, current_hash=current_hash)
    require(ladder_mean_gross > 0.0, "Het gemeten boek moet positief zijn.",
            DataContractError, ladder_mean_gross=ladder_mean_gross)
    old, new = entries[ladder_hash], entries[current_hash]
    return {"vol_target": float(new["sigma_target"]) / float(old["sigma_target"]),
            "gross_cap_bound": float(new["gross_cap"]) / float(ladder_mean_gross)}


def gate_verdict(
    *,
    gain_l0: float,
    multipliers: dict[str, float],
    detectable_two_sided: float,
    detectable_one_sided: float,
    precondition_met: bool,
) -> dict[str, Any]:
    """De poort van stap 7.3, zoals zij vooraf staat.

    Groen: de winst op de ladder (`multipliers["ladder"]`) is groter dan de
    TWEEZIJDIGE grens, want de toets van H-10.1 vraagt een 95 %-interval dat nul
    uitsluit. Zonder de nieuwe ladder is alleen een rood definitief dat ook bij
    de grootste schaal onder de EENZIJDIGE grens blijft: dan kan geen boek het
    omdraaien, ook niet bij de mildste lezing van de toets.
    """
    require(gain_l0 > 0.0 and "ladder" in multipliers,
            "De poort vraagt een positieve kostenwinst en de multiplier van de ladder.",
            DataContractError, gain_l0=gain_l0, multipliers=sorted(multipliers))
    gains = {name: m * gain_l0 for name, m in multipliers.items()}
    green_on_ladder = gains["ladder"] > detectable_two_sided
    red_under_every_book = max(gains.values()) <= detectable_one_sided
    if precondition_met:
        gate = "GREEN" if green_on_ladder else "RED"
    else:
        gate = "RED" if red_under_every_book else "PENDING_7_1"
    return {
        "gate": gate,
        "cost_gain_sharpe_l3": gains,
        "green_on_ladder": bool(green_on_ladder),
        "red_under_every_book": bool(red_under_every_book),
        "multiplier_required_two_sided": detectable_two_sided / gain_l0,
        "multiplier_required_one_sided": detectable_one_sided / gain_l0,
        "precondition_met": bool(precondition_met),
    }


def measure_feasibility(
    tracks: dict[str, pd.DataFrame],
    *,
    prices: pd.DataFrame,
    usable: pd.Index,
    development: pd.DataFrame,
    cfg: FeasibilityConfig,
    k_star: int,
    cost_per_side: float,
    ladder: dict[str, Any],
    scales: dict[str, float],
    current_policy_hash: str,
    sr_level_required: float,
    bars_per_year: float,
    ci_level: float,
) -> dict[str, Any]:
    """Stap 7.1 t/m 7.3 op de primaire cel. Geen enkel echt gemiddeld rendement.

    `sr_level_required` is de DSR-drempel van de muur. Het tweede criterium van
    H-10.1 is `dsr_development_best_k >= 0.95`: een DSR op de NIVEAU-Sharpe van
    de vastgehouden reeks, niet op het verschil. Het blok `level_criterion` zet
    die drempel naast de enige gepubliceerde niveau-Sharpe van deze cel (L3,
    k = 1, volle venster, uit de ladder) en de grootste kostenwinst; het is een
    citaat, geen meting.

    Het besluitpaneel is dat van stap 3; het vasthouden gebeurt met
    `hold_decision`, verankerd op de eerste bruikbare bar zoals H-10.1, en wordt
    daarna op `W_DEV` gesneden. Σ is de covariantie van de logrendementen van
    `W_DEV`, dezelfde als die van de muur.
    """
    decision = tracks[cfg.primary_track].loc[usable].fillna(0.0)
    panels = {"reference": hold_decision(decision, k=cfg.reference_k),
              "held": hold_decision(decision, k=int(k_star))}
    panels = {name: panel.loc[development.index] for name, panel in panels.items()}
    covariance = development.cov().to_numpy()
    turnover, volatility = {}, {}
    for name, panel in panels.items():
        turnover[name] = float(run_vectorized(panel, prices.loc[panel.index], cost_per_side=0.0,
                                              initial_equity=1.0).turnover.mean())
        executed = panel.shift(1).fillna(0.0).to_numpy(dtype=np.float64)
        volatility[name] = float(np.sqrt(
            np.einsum("ti,ij,tj->t", executed, covariance, executed).mean() * bars_per_year))
    null = null_detection_limit(panels["reference"], panels["held"], covariance,
                                n_paths=cfg.null_paths, seed=cfg.null_seed,
                                n_boot=cfg.null_n_boot, bars_per_year=bars_per_year,
                                ci_level=ci_level)
    gain_l0 = cost_gain_sharpe(turnover_reference=turnover["reference"],
                               turnover_held=turnover["held"], cost_per_side=cost_per_side,
                               book_volatility=volatility["reference"],
                               bars_per_year=bars_per_year)
    multipliers = {name: impact_multiplier(fees=ladder["fees"], impact=ladder["impact"],
                                           spread=ladder["spread"], scale=scale)
                   for name, scale in {"ladder": 1.0, **scales}.items()}
    verdict = gate_verdict(
        gain_l0=gain_l0, multipliers=multipliers,
        detectable_two_sided=null["detectable_two_sided"],
        detectable_one_sided=null["detectable_one_sided"],
        precondition_met=ladder["risk_policy_hash"] == current_policy_hash)
    detection_cost_bps = {
        side: breakeven_cost_bps(-null[f"detectable_{side}"],
                                 turnover["reference"] - turnover["held"],
                                 volatility_per_bar=volatility["reference"] / math.sqrt(bars_per_year),
                                 bars_per_year=bars_per_year)
        for side in ("two_sided", "one_sided")}
    gap = float(sr_level_required) - ladder["net_sharpe_l3_full_window"]
    level = {"sr_required": float(sr_level_required),
             "net_sharpe_l3_k1_full_window": ladder["net_sharpe_l3_full_window"],
             "gap": gap, "max_cost_gain_sharpe_l3": max(verdict["cost_gain_sharpe_l3"].values()),
             "gap_exceeds_max_cost_gain": bool(
                 gap > max(verdict["cost_gain_sharpe_l3"].values()))}
    return {
        "primary_track": cfg.primary_track, "window": "W_DEV",
        "n_obs": int(len(development)), "k_reference": cfg.reference_k, "k_star": int(k_star),
        "turnover_per_bar": turnover, "book_volatility_annual": volatility,
        "cost_per_side": float(cost_per_side), "cost_gain_sharpe_l0": gain_l0,
        "null": null, "ladder": ladder, "book_scales": scales,
        "impact_multipliers": multipliers,
        "detection_cost_per_side_bps": detection_cost_bps,
        "current_risk_policy_hash": current_policy_hash,
        **verdict,
        "level_criterion": level,
        "trials_spent": 0,
        "first_moments_computed": False,
    }
