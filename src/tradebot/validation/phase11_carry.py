# src/tradebot/validation/phase11_carry.py
"""H-11.1, de carrykandidaat. Fase 11, stage B.

Stap 6 legt hier vast wat vóór de eerste fit bekend moet zijn: de
ontwerpgetallen (uit `conf/research/h11_funding_carry.yaml`), de DSR-drempel,
de adequaatheidsvloer per houdduur, en het trialbudget. Die worden bij het
bevriezen van de pre-registratie ingespoten (`h11_parameters`), zodat de
pre-registratie-ID ze vastpint: wie na het bevriezen een getal wijzigt, krijgt
een andere ID en dus een pre-registratie die niet bevroren is.

WAT HIER BEWUST NIET STAAT
==========================
Geen enkel resultaat van de constructie. De primaire cel en het rooster zijn
gekozen op gepubliceerde, beschrijvende cijfers (masterprompt §3.6) en op
uitrekenbare vloeren, niet op een meting van het carryboek. Een keuze na zo'n
meting zou een trial zijn (R-2).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from ..backtest.metrics import deflated_sharpe
from ..registry.ledger_reset import active_trial_count
from ..registry.trial_budget import assert_within_budget
from ..utils.failfast import DataContractError, require
from .inference import effective_breadth

__all__ = [
    "DESIGN",
    "HOLDING_PERIODS",
    "PRIMARY_HOLDING_PERIOD",
    "H11Design",
    "adequacy_by_holding_period",
    "dsr_hurdle",
    "freeze_h11",
    "h11_design",
    "h11_parameters",
    "h11_preregistration",
    "measured_breadth",
    "spent_since_reset",
    "trial_budget_after",
]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DESIGN_PATH = _REPO_ROOT / "conf/research/h11_funding_carry.yaml"
SPEC_PATH = "conf/research/preregistration_h11_funding_carry.yaml"
_GOV = "artefacts/governance"

#: De ledger-entry waarmee de post-reset-telling begint (AD-24).
_RESET_UNIT = "ad24_ledger_reset"


@dataclass(frozen=True)
class H11Design:
    """De ontwerpgetallen van H-11.1, zoals `conf/` ze levert."""

    holding_periods: tuple[int, ...]
    primary_holding_period: int
    construction: str
    dropped_construction: str
    adequacy_floor: int
    dsr_target: float
    permutation_seed: int
    signal_lag_bars: int

    def __post_init__(self) -> None:
        require(
            self.primary_holding_period in self.holding_periods,
            "De primaire houdduur staat niet in het rooster. De primaire cel is "
            "per definitie een van de geregistreerde cellen.",
            DataContractError,
            primary=self.primary_holding_period, grid=list(self.holding_periods),
        )
        require(
            self.signal_lag_bars >= 1,
            "De carry die op bar t bekend is, mag niet de afrekeningen van bar t "
            "zelf bevatten (masterprompt stap 7.1, R-1).",
            DataContractError, signal_lag_bars=self.signal_lag_bars,
        )


def h11_design(path: Path | str | None = None) -> H11Design:
    """Lees de ontwerpgetallen. Onbekende of ontbrekende sleutels crashen."""
    p = Path(path) if path is not None else _DESIGN_PATH
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))["h11"]
    fields = set(H11Design.__dataclass_fields__)
    require(
        set(raw) == fields,
        "De ontwerpconfig van H-11.1 heeft niet precies de verwachte sleutels. "
        "Een typo mag nooit stil betekenen dat een ontwerpkeuze niet bestaat.",
        DataContractError,
        unknown=sorted(set(raw) - fields), missing=sorted(fields - set(raw)),
    )
    return H11Design(
        holding_periods=tuple(int(h) for h in raw["holding_periods"]),
        primary_holding_period=int(raw["primary_holding_period"]),
        construction=str(raw["construction"]),
        dropped_construction=str(raw["dropped_construction"]),
        adequacy_floor=int(raw["adequacy_floor"]),
        dsr_target=float(raw["dsr_target"]),
        permutation_seed=int(raw["permutation_seed"]),
        signal_lag_bars=int(raw["signal_lag_bars"]),
    )


DESIGN = h11_design()
HOLDING_PERIODS = DESIGN.holding_periods
PRIMARY_HOLDING_PERIOD = DESIGN.primary_holding_period


def dsr_hurdle(
    *, n_obs: int, n_trials: int, bars_per_year: float,
    dsr_target: float = DESIGN.dsr_target,
) -> float:
    """De geannualiseerde Sharpe die bij `n_trials` en `n_obs` `DSR >= dsr_target` haalt.

    Gaussische vorm: `sr_variance = 1/n_obs`, scheefheid 0, kurtosis 3, zoals
    de tabel in `registry/trial_budget.py` en de 1,8686 van H-10.1. De DSR zelf
    komt uit `backtest.metrics.deflated_sharpe`, de enige implementatie (R-3);
    dit is alleen de bisectie eromheen. De werkelijke toets gebruikt later de
    GEMETEN momenten van de rendementsreeks, en die maken de eis op
    dikstaartige crypto-rendementen hoger, niet lager.
    """
    def dsr(annual: float) -> float:
        return float(deflated_sharpe(
            sr_hat=annual / math.sqrt(bars_per_year), n_obs=n_obs,
            n_trials=n_trials, sr_variance=1.0 / n_obs, skew=0.0, kurtosis=3.0,
            bars_per_year=bars_per_year).dsr)

    lo, hi = 0.0, 10.0
    require(dsr(hi) >= dsr_target > dsr(lo),
            "De DSR-drempel ligt buiten het zoekinterval [0, 10].",
            DataContractError, n_obs=n_obs, n_trials=n_trials)
    while hi - lo > 1e-9:
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if dsr(mid) >= dsr_target else (mid, hi)
    return hi


def adequacy_by_holding_period(
    *, n_obs: int, n_eff: float,
    holding_periods: tuple[int, ...] = HOLDING_PERIODS,
    floor: int = DESIGN.adequacy_floor,
) -> dict[int, dict[str, Any]]:
    """Per houdduur: niet-overlappende perioden, de rho-gedefleerde telling, en de vloer.

    Het boek levert `floor(n_obs / h)` niet-overlappende perioden. De
    rho-gedefleerde paneltelling is dat getal maal de effectieve breedte
    `N_eff` van de zes namen. AD-20: het oordeel valt op de KLEINSTE van de
    twee. Een cel onder de vloer is `UNPROVEN -- insufficient data` en telt
    niet mee in het oordeel (stop-criterium 5).
    """
    table: dict[int, dict[str, Any]] = {}
    for h in holding_periods:
        periods = int(np.floor(n_obs / h))
        deflated = periods * float(n_eff)
        conservative = min(float(periods), deflated)
        table[int(h)] = {
            "nonoverlapping_periods": periods,
            "effective_bets_rho_deflated": deflated,
            "conservative": conservative,
            "floor": int(floor),
            "passes_floor": bool(conservative >= floor),
        }
    return table


def measured_breadth(log_returns: pd.DataFrame) -> float:
    """`N_eff` van het universum op het gegeven venster (`inference.effective_breadth`)."""
    corr = log_returns.dropna(how="any").corr().to_numpy(dtype="float64")
    return float(effective_breadth(corr))


def spent_since_reset(ledger_path: Path | str) -> int:
    """Het aantal trials dat na de AD-24-reset is besteed, uit de ledger.

    Na de reset boeken hypothesen als AMENDEMENT met `n_trials = 0`, want een
    gewone entry zou het ledgertotaal boven de bevroren `archived_total` tillen
    en `active_trial_count` laten weigeren. De besteding staat daarom in
    `metrics`: `trials_spent` als dat veld er is (een vervallen hypothese
    besteedt minder dan zij plande), anders `planned_trials`.
    """
    doc = json.loads(Path(ledger_path).read_text(encoding="utf-8"))
    units = [e["unit"] for e in doc["entries"]]
    require(
        _RESET_UNIT in units,
        "De ledger kent geen AD-24-reset. Zonder reset is er geen post-reset "
        "telling, en is `m_new` niet het budget.",
        DataContractError, ledger=str(ledger_path),
    )
    start = len(units) - 1 - units[::-1].index(_RESET_UNIT)
    spent = 0
    for entry in doc["entries"][start + 1:]:
        metrics = entry.get("metrics") or {}
        if "trials_spent" in metrics:
            spent += int(metrics["trials_spent"])
        elif "planned_trials" in metrics:
            spent += int(metrics["planned_trials"])
    return spent


def trial_budget_after(
    *, spent_before: int, planned: int, reset_path: Path,
) -> dict[str, int]:
    """Weiger een plan boven het bevroren budget, en geef de stand erna.

    De rem is `registry.trial_budget.assert_within_budget` op het CUMULATIEVE
    aantal: wat er al besteed is plus wat nu gepland wordt. Er komt geen tweede
    budgetcontrole bij (R-3).
    """
    after = int(spent_before) + int(planned)
    assert_within_budget(after, reset_path=reset_path)
    m_new = active_trial_count(
        reset_path=reset_path,
        ledger_path=_REPO_ROOT / "artefacts/governance/hypothesis_ledger.json",
    ).total
    return {"spent_before": int(spent_before), "planned": int(planned),
            "spent_after": after, "m_new": int(m_new),
            "remaining_after": int(m_new) - after}


def h11_parameters(root: Path) -> dict[str, Any]:
    """De parameterruimte die bij het bevriezen wordt ingespoten (stap 6.1).

    Alles wat de pre-registratie-ID moet vastpinnen en dat niet in het
    YAML-schema past: het ontwerp uit `conf/`, het beleid dat zal beslissen
    (AD-27), het venster, de kostenbasis, het budget, de drempels en de
    vooraf uitgerekende adequaatheid. Niets hiervan is een resultaat van de
    constructie: `N_eff` is de breedte van de zes PRIJSreeksen op W_DEV, die
    ook de nulmeting al rapporteerde.
    """
    from ..backtest.ladder_inputs import load_ladder_inputs
    from ..registry.policy_carry import current_policy_hash
    from .holdout import development_slice

    inputs = load_ladder_inputs(root)
    lock_path = root / _GOV / "holdout_lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    dev = development_slice(pd.DataFrame({"bar": 0.0}, index=inputs.usable),
                            lock_path=lock_path).index
    n_obs = int(len(dev))
    bars_per_year = float(inputs.cfg["bt"].bars_per_year)
    n_eff = measured_breadth(np.log(inputs.market["prices"].loc[dev]).diff())
    budget = trial_budget_after(
        spent_before=spent_since_reset(root / _GOV / "hypothesis_ledger.json"),
        planned=len(DESIGN.holding_periods),
        reset_path=root / _GOV / "ledger_reset.json")
    fees = inputs.cfg["exec"]
    fixed_rt = 2.0 * (float(fees.taker_fee_bps) + float(fees.assumed_half_spread_bps))
    return {
        "construction": DESIGN.construction,
        "dropped_construction": {
            "name": DESIGN.dropped_construction,
            "reason": "de hedge is het gelijkgewogen mandje van dezelfde zes "
                      "perpetuals, dat dezelfde funding betaalt: netto carry "
                      "c_bar*(1-beta_bar) ~ 0; spot valt buiten AD-23; "
                      "geschrapt op besluit van de eigenaar, 2026-09-23"},
        "holding_periods": list(DESIGN.holding_periods),
        "primary_cell": {"construction": DESIGN.construction,
                         "holding_period": DESIGN.primary_holding_period,
                         "layer": "L3_execution", "metric": "net_sharpe",
                         "window": "W_DEV"},
        "signal": {
            "known_carry": "som van de afrekeningen in dagbar t - signal_lag_bars "
                           "(data/funding_panel.py::daily_funding_panel, "
                           "verschoven); uitsluitend afrekeningen van voor het "
                           "besluit",
            "signal_lag_bars": DESIGN.signal_lag_bars,
            "ranking": "op elke herbalanceringsbar: aflopend op bekende carry, "
                       "gelijke waarden op symboolnaam; de bovenste helft short "
                       "(a = -1), de onderste helft long (a = +1)",
            "rebalance": "elke h bars vanaf de eerste W_DEV-bar waarop elk "
                         "symbool een bekende carry heeft; daartussen vast",
            "exposure_units": "a in {-1, +1}; de soevereine laag bepaalt de "
                              "grootte (vol-target)"},
        "risk_policy_hash": current_policy_hash(),
        "layer": "L3_execution",
        "window": {"name": "W_DEV", "start": str(dev[0].date()),
                   "end": str(dev[-1].date()), "n_bars": n_obs,
                   "split_utc": lock["split_utc"]},
        "gate_reads_at_freeze": list(lock["reads"]),
        "bars_per_year": bars_per_year,
        "cost_basis": {
            "taker_fee_bps": float(fees.taker_fee_bps),
            "maker_fee_bps": float(fees.maker_fee_bps),
            "assumed_half_spread_bps": float(fees.assumed_half_spread_bps),
            "fixed_round_trip_bps": fixed_rt,
            "impact_eta": float(inputs.impact.eta),
            "impact_kappa_d": float(inputs.impact.kappa_d),
            "impact_status": str(inputs.impact.status.value)},
        "trial_budget": budget,
        "dsr": {"m_new": budget["m_new"], "n_obs": n_obs,
                "bars_per_year": bars_per_year, "dsr_target": DESIGN.dsr_target,
                "approximation": "normal",
                "required_annual_sharpe_gaussian": dsr_hurdle(
                    n_obs=n_obs, n_trials=budget["m_new"],
                    bars_per_year=bars_per_year),
                "t2_hurdle": 2.0 / math.sqrt(n_obs / bars_per_year)},
        "n_eff_w_dev": n_eff,
        "adequacy": {str(h): cell for h, cell in adequacy_by_holding_period(
            n_obs=n_obs, n_eff=n_eff).items()},
        "negative_controls": {
            "sign_reversal": "hetzelfde boek met omgekeerd teken",
            "time_permutation": "de bekende carry per symbool geschud in de tijd",
            "permutation_seed": DESIGN.permutation_seed,
            "pass_rule": "HAC-t > 2 op het dagelijkse verschil in carry-P&L "
                         "(arm min controle), validation.inference.sharpe_with_se, "
                         "op de primaire houdduur"},
        "decomposition": {
            "carry_pnl": "ontvangen min betaalde funding, door de engine geboekt",
            "costs": "fees + spread + impact",
            "price_pnl": "netto P&L + kosten - carry_pnl",
            "carry_share": "carry_pnl / (carry_pnl + price_pnl); niet "
                           "evalueerbaar bij een niet-positief bruto"},
    }


def h11_preregistration(root: Path, parameters: dict[str, Any]) -> Any:
    """De pre-registratie uit de spec, met de gecertificeerde data_hashes."""
    from ..features.base import DataRegister
    from ..registry.preregistration import load_preregistration_spec

    register = DataRegister(root / _GOV / "data_hashes.json")
    series = yaml.safe_load((root / SPEC_PATH).read_text(encoding="utf-8"))[
        "preregistration"]["data_series"]
    return load_preregistration_spec(
        root / SPEC_PATH, parameters=parameters,
        data_hashes=[(s, register.hashes[s]) for s in series])


def freeze_h11(
    root: Path, *, directory: Path | None = None, git_sha: str | None = None,
) -> Path:
    """Bevries H-11.1. Eenmalig: een tweede aanroep met dezelfde inhoud is een no-op."""
    from ..features.registry import current_git_sha
    from ..registry.preregistration import freeze_preregistration

    prereg = h11_preregistration(root, h11_parameters(root))
    total = active_trial_count(
        reset_path=root / _GOV / "ledger_reset.json",
        ledger_path=root / _GOV / "hypothesis_ledger.json").archived_total
    return freeze_preregistration(
        prereg, git_sha=git_sha or current_git_sha(),
        ledger_total_at_freeze=total,
        directory=directory if directory is not None else root / _GOV)

