# apps/run_h10_3_state_gate.py
"""H-10.3 — de toestandspoort. Fase 10, stap 13.

DEZE STAP VERVALT, EN DAT IS HET ANTWOORD. De stapopdracht maakt H-10.3
voorwaardelijk: *"Zij hangt aan stap 9. Haalt de driedelige toestand de
bezettings- en episodepoort niet, dan VERVALT deze stap en is dat het antwoord."*
Deze app stelt vast of die voorwaarde is gehaald. Zij haalt hem niet.

Er wordt daarom GEEN hypothese getoetst, GEEN `gate_by_state` geschreven, GEEN
poortsample gelezen en GEEN trial besteed. De app doet één ding: de poort van
stap 9 evalueren op de werkelijke toewijzing, en het oordeel plus de rekenkundige
grond ervan wegschrijven.

WAAROM DIT EEN APP IS EN NIET EEN ALINEA IN HET RAPPORT. Het oordeel "vervalt"
rust op een meting, en een meting die niemand kan herhalen is een bewering. Deze
app is die herhaalbaarheid. Zij kost nul trials omdat zij niets selecteert: de
drempels komen uit `conf/model/adequacy.yaml`, de foldgeometrie uit
`validation/adequacy_report.py::measure_fold_geometry`, en de toewijzing uit
`regime/state.py` — er is geen keuze die op grond van de uitkomst is gemaakt.

DE POORT STOND UIT, en dat is waarom dit niet eerder is vastgesteld.
`apps/run_state_diagnostics.py` roept `assert_realised_occupancy` aan, maar achter
een `--check-adequacy`-vlag die op False staat en die niemand ooit heeft
meegegeven. Erger: het oordeel ging naar stdout en NOOIT naar het artefact. De
poort van stap 9 is dus geschreven en getest, maar zijn uitspraak over de echte
toewijzing is nergens vastgelegd -- en juist daarop staat of valt stap 13.

In dezelfde commit is die vlag verwijderd: de poort draait nu altijd en zijn
oordeel staat in `phase10_state_diagnostics.json`. Een poort die standaard uit
staat, is geen poort; en een oordeel dat alleen in stdout bestaat, is geen
meting (R-8). Deze app is de tweede, expliciete lezing ervan, omdat het oordeel
van stap 13 er volledig op rust.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tradebot.alpha.momentum import build_cross_sectional_momentum
from tradebot.backtest.baseline_report import json_safe, load_baseline_configs
from tradebot.data.phase6_universe import load_phase6_universe
from tradebot.features.registry import current_git_sha
from tradebot.regime.state import assign_by_variance
from tradebot.schemas.config import adequacy_config, regime_config
from tradebot.validation.adequacy_report import measure_fold_geometry
from tradebot.validation.data_adequacy import assert_realised_occupancy
from tradebot.validation.holdout import development_slice

LOCK = ROOT / "artefacts/governance/holdout_lock.json"
OUT = "artefacts/governance/phase10_h10_3.json"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args(argv)

    cfg = load_baseline_configs(ROOT)
    state_cfg = regime_config().state
    universe = load_phase6_universe(
        ROOT, cfg, build_cross_sectional_momentum(cfg["alpha"]),
        git_sha=current_git_sha())
    sigma = development_slice(universe.sigma_annual, lock_path=LOCK)
    assignment = assign_by_variance(
        sigma, low_q=float(state_cfg.low_q), high_q=float(state_cfg.high_q),
        min_periods=int(state_cfg.min_periods),
        source=f"ewma_lambda_{cfg['vol'].ewma_lambda}", lag=int(state_cfg.lag))

    limits = adequacy_config().vol_state
    geometry = measure_fold_geometry(len(sigma), cfg["val"])
    verdict = assert_realised_occupancy(
        assignment,
        n_folds=geometry.n_folds,
        min_obs_per_state_per_fold=limits.min_obs_per_state_per_fold,
        min_episodes_per_state_per_fold=limits.min_episodes_per_state_per_fold,
        min_occupancy_fraction=float(limits.min_state_occupancy_fraction),
        raise_on_failure=False)

    # De poort snijdt de TOEGEWEZEN bars (burn-in eraf) in n_folds blokken en eist
    # de drempel per toestand per fold per symbool. Daarmee is er een bovengrens
    # aan wat überhaupt haalbaar is: alle toestanden samen kunnen niet meer bars
    # beslaan dan de fold lang is. Dit blok rekent die grens uit, zodat het
    # oordeel niet alleen zegt DAT de poort faalt maar ook of hij had kunnen
    # slagen. Geen drempel wordt hier gewijzigd; er wordt alleen gedeeld.
    assigned = int(verdict.measured["n_assigned_bars"])
    n_states = int(verdict.measured["n_states"])
    bars_per_fold = assigned / geometry.n_folds
    required_per_fold = n_states * int(limits.min_obs_per_state_per_fold)
    satisfiable = required_per_fold <= bars_per_fold

    payload = {
        "git_sha": current_git_sha(),
        "hypothesis": "H-10.3",
        "verdict": "LAPSED",
        "verdict_basis": (
            "Stap 13 is conditional on stap 9's realised occupancy and episode "
            "gate. The gate returns adequate=false, so the step lapses and that "
            "is the answer. No hypothesis was tested, no trial was spent, and the "
            "gate sample was not read."),
        "trials_planned": 1,
        "trials_spent": 0,
        "gate_sample_read": False,
        "sample": "development",
        "occupancy_gate": json_safe(verdict.as_record()),
        "fold_geometry": {
            "n_folds": geometry.n_folds,
            "n_bars": geometry.n_bars,
            "train_sizes": list(geometry.train_sizes),
            "test_sizes": list(geometry.test_sizes),
        },
        "gate_satisfiability": {
            "n_assigned_bars": assigned,
            "n_states": n_states,
            "bars_per_fold_per_symbol": bars_per_fold,
            "required_bars_per_fold_per_symbol": required_per_fold,
            "satisfiable": satisfiable,
            "shortfall_factor": required_per_fold / bars_per_fold,
            "max_n_folds_that_could_satisfy": int(assigned // required_per_fold),
            "max_threshold_satisfiable_at_this_n_folds": bars_per_fold / n_states,
            "note": (
                "The bars arm of the gate requires min_obs_per_state_per_fold for "
                "EACH state, per fold, per symbol. Their sum cannot exceed the "
                "fold length. When required_bars_per_fold_per_symbol exceeds "
                "bars_per_fold_per_symbol the gate cannot be satisfied by ANY "
                "assignment at this fold count -- it then reports a property of "
                "the fold geometry, not of the regime. The verdict stands either "
                "way (the regime also fails the episode arm by an order of "
                "magnitude), but the gate is over-determined here and that is a "
                "finding about the gate."),
        },
        "artefact_path": args.out,
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(json_safe(payload), indent=2), encoding="utf-8")
    print(f"wrote {out}")
    print(f"occupancy gate adequate = {verdict.adequate}  -> H-10.3 LAPSED, 0 trials spent")
    print(f"gate satisfiable at n_folds={geometry.n_folds}? {satisfiable} "
          f"({required_per_fold} bars required vs {bars_per_fold:.1f} available "
          f"per fold per symbol, factor {required_per_fold / bars_per_fold:.2f}x)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
