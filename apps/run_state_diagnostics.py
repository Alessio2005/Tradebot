# apps/run_state_diagnostics.py
"""Stap 6 -- de toestandsdiagnose op de ONTWIKKELSAMPLE. Selecteert niets.

De wetenschap staat in `regime/state_diagnostics.py`. Deze app draait haar TWEE
keer op hetzelfde venster: gelagd (het contract van stap 5) en met `lag=0`, de
ongelagde toewijzing van revisie 1. Die tweede is per constructie een lookahead
en meet UITSLUITEND hoeveel van revisie 1's "separatie" uit gelijktijdigheid
kwam. Nul trials: er wordt niets uit geselecteerd.

Geen CLI-vlaggen (YAGNI): het uitvoerpad is een DVC-`out` en verschuift niet.
"""
from __future__ import annotations

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
from tradebot.regime.state_diagnostics import StateDiagnostics, diagnose
from tradebot.schemas.config import regime_config
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


def main() -> int:
    cfg, state_cfg = load_baseline_configs(ROOT), regime_config().state
    universe = load_phase6_universe(ROOT, cfg,
        build_cross_sectional_momentum(cfg["alpha"]), git_sha=current_git_sha())
    sigma = development_slice(universe.sigma_annual, lock_path=LOCK)
    returns = development_slice(universe.log_returns, lock_path=LOCK)

    runs: dict[str, StateDiagnostics] = {}
    for label, lag in (("lagged", int(state_cfg.lag)), ("unlagged_revision_1", 0)):
        runs[label] = diagnose(
            assign_by_variance(
                sigma, low_q=float(state_cfg.low_q), high_q=float(state_cfg.high_q),
                min_periods=int(state_cfg.min_periods),
                source=f"ewma_lambda_{cfg['vol'].ewma_lambda}", lag=lag),
            returns, bars_per_year=float(cfg["bt"].bars_per_year))
        _print(label, runs[label])

    payload = {
        "git_sha": current_git_sha(), "selects_nothing": True, "trials": 0,
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
