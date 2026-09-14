# apps/run_state_diagnostics.py
"""Stap 6 -- de toestandsdiagnose op de ONTWIKKELSAMPLE. Selecteert niets.

De wetenschap staat in `regime/state_diagnostics.py`. Deze app draait haar TWEE
keer op hetzelfde venster en met DEZELFDE toewijzingsregel: gelagd (het contract
van stap 5) en met `lag=0`, dus gelijktijdig. Die tweede is per constructie een
lookahead en isoleert precies een ding -- hoeveel van de gemeten vol-separatie
uit gelijktijdigheid komt. Zij is GEEN reproductie van revisie 1 (fixronde 1,
bevinding I-1): revisie 1 mat met `regime/buckets.py::classify_vol_buckets`,
vaste +/-0,5-sigma-banden plus een ATR-as, bezetting 16,5/78,7/4,8 %, en niet
met `assign_by_variance`, bezetting 39,5/48,1/12,4 %. Nul trials: er wordt niets
uit geselecteerd.

Een vlag, en zij is additief (fase 10, stap 9, ruling P38). Zonder vlag doet
deze app exact wat zij deed en schrijft zij exact hetzelfde artefact; de
DVC-stage draait ongewijzigd. Met `--check-adequacy` legt zij daar de
bezettingspoort van stap 9 naast, op de GELAGDE toewijzing -- de ongelagde is
een lookahead en er valt niets over te oordelen. Het uitvoerpad blijft een
DVC-`out` en verschuift niet.

R-6 (apps <= 80 LOC) wordt hier overschreden. De poort zelf staat in
`validation/data_adequacy.py`; wat hier bij komt is bedrading en presentatie,
en die hoort bij de app die hem aanroept. Een module erbij om een `print` te
huisvesten zou R-3 zwaarder belasten dan R-6 hier wint.
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
from tradebot.regime.state import StateAssignment, assign_by_variance
from tradebot.regime.state_diagnostics import StateDiagnostics, diagnose
from tradebot.schemas.config import ValidationConfig, adequacy_config, regime_config
from tradebot.validation.adequacy_report import measure_fold_geometry
from tradebot.validation.data_adequacy import (
    OccupancyVerdict,
    assert_realised_occupancy,
)
from tradebot.validation.holdout import development_slice

LOCK = ROOT / "artefacts/governance/holdout_lock.json"
OUT = "artefacts/governance/phase10_state_diagnostics.json"
COLUMNS = ("state", "n", "occ", "epis", "dur", "vol", "ret", "t_pool", "t_clus", "t_defl")


def _print(label: str, result: StateDiagnostics) -> None:
    print(f"\n{label}  ({result.n_bars} bars, t_years {result.t_years:.4f}, "
          f"lag {result.params['lag']:.0f})")
    print("  " + "".join(f"{name:>9}" for name in COLUMNS))
    for state in ("LOW", "NORMAL", "HIGH"):
        vol, ret = result.separation_vol[state], result.separation_return[state]
        print("  " + f"{state:>9}{vol['n_bars']:>9}{vol['occupancy']:>9.3f}"
              f"{vol['n_episodes']:>9}{vol['mean_duration_bars']:>9.1f}"
              f"{vol['ann_vol']:>9.1%}{ret['ann_return']:>9.1%}"
              f"{ret['t_stat_pooled']:>9.2f}{ret['t_stat_clustered']:>9.2f}"
              f"{ret['t_stat_neff_deflated']:>9.2f}")


def _check_adequacy(assignment: StateAssignment, *, n_bars: int,
                    val: ValidationConfig) -> OccupancyVerdict:
    """Stap 9: de bezettingspoort op de ECHTE toewijzing. Meet en rapporteert.

    `raise_on_failure=False` en niet de crashende tak: deze app MEET, en het
    oordeel is de uitkomst van de meting en niet een fout in de run. Dat is het
    onderscheid dat `require_adequacy` in zijn eigen docstring maakt -- de
    runner mag registreren, de fit mag niet uitvoeren. De exitcode blijft dus 0;
    het oordeel staat in het rapport, niet in `$?`.

    `n_folds` komt uit `measure_fold_geometry`, de ENE plek waar dit project
    telt hoeveel walk-forward folds een reeks oplevert (R-3). Het is niet het
    geconfigureerde minimum `n_splits`.
    """
    limits = adequacy_config().vol_state
    verdict = assert_realised_occupancy(
        assignment,
        n_folds=measure_fold_geometry(n_bars, val).n_folds,
        min_obs_per_state_per_fold=limits.min_obs_per_state_per_fold,
        min_episodes_per_state_per_fold=limits.min_episodes_per_state_per_fold,
        # RULING P39. De config noemt deze drempel `min_state_occupancy_fraction`,
        # gelijk aan haar buurman in het `hmm:`-blok; de functie noemt hem
        # `min_occupancy_fraction`. Deze regel is de ENIGE vertaling tussen beide
        # namen, zodat er nergens anders twee namen voor een drempel rondgaan.
        min_occupancy_fraction=float(limits.min_state_occupancy_fraction),
        raise_on_failure=False,
    )
    print("\nbezettingspoort op de GEREALISEERDE bezetting (stap 9)")
    print(json.dumps(json_safe(verdict.as_record()), indent=2, ensure_ascii=False))
    return verdict


def main(argv: list[str] | None = None) -> int:
    # RULING (fase 10, stap 13). De bezettingspoort van stap 9 stond achter een
    # `--check-adequacy`-vlag met default False, en niemand heeft hem ooit
    # meegegeven: zijn oordeel over de ECHTE toewijzing was daarmee nergens
    # vastgelegd, terwijl stap 13 er volledig op rust. De vlag is verwijderd.
    # Een poort die standaard uit staat is geen poort, en een oordeel dat alleen
    # in stdout bestaat is geen meting (R-8) -- het gaat nu het artefact in.
    argparse.ArgumentParser(description="stap 6 -- toestandsdiagnose").parse_args(argv)

    cfg, state_cfg = load_baseline_configs(ROOT), regime_config().state
    universe = load_phase6_universe(ROOT, cfg,
        build_cross_sectional_momentum(cfg["alpha"]), git_sha=current_git_sha())
    sigma = development_slice(universe.sigma_annual, lock_path=LOCK)
    returns = development_slice(universe.log_returns, lock_path=LOCK)

    runs: dict[str, StateDiagnostics] = {}
    assignments: dict[str, StateAssignment] = {}
    for label, lag in (("lagged", int(state_cfg.lag)), ("unlagged_revision_1", 0)):
        assignments[label] = assign_by_variance(
            sigma, low_q=float(state_cfg.low_q), high_q=float(state_cfg.high_q),
            min_periods=int(state_cfg.min_periods),
            source=f"ewma_lambda_{cfg['vol'].ewma_lambda}", lag=lag)
        runs[label] = diagnose(
            assignments[label], returns, bars_per_year=float(cfg["bt"].bars_per_year))
        _print(label, runs[label])

    occupancy = _check_adequacy(
        assignments["lagged"], n_bars=len(sigma), val=cfg["val"])

    payload = {
        "git_sha": current_git_sha(), "selects_nothing": True, "trials": 0,
        "occupancy_gate": json_safe(occupancy.as_record()),
        "full_window": universe.as_record(),
        "development_window": {
            "split_utc": json.loads(LOCK.read_text(encoding="utf-8"))["split_utc"],
            "period_start": str(sigma.index[0].date()),
            "period_end": str(sigma.index[-1].date()),
            "n_sigma_bars": int(len(sigma)), "n_return_bars": int(len(returns))},
        "runs": {label: result.to_dict() for label, result in runs.items()}}
    (ROOT / OUT).write_text(json.dumps(json_safe(payload), indent=2), encoding="utf-8")
    print(f"\nartefact: {OUT}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
