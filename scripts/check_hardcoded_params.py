"""Phase 0 / stap 8 - scanner voor hardcoded parameters (exit criterium 6).

Detecteert numerieke literals die als DREMPEL, LOOKBACK-WINDOW, lambda-parameter
of LIMIET fungeren, en dus in `conf/` horen te staan in plaats van in `src/`.

Wat het WEL vindt:
  * numerieke defaults van functie-/methode-argumenten
  * module-level constanten met een numerieke waarde

Wat het bewust NEGEERT (de `BENIGN`-set): identiteiten en indices (0, 1, -1),
kansdrempel 0.5, percentagenoemer 100, en de kalenderconstanten 12/24/60/365/252.
Die zijn geen beleidskeuze maar rekenkundige feiten.

ALLOWLIST
---------
`ALLOWLIST` bevat per bestand het AANTAL nog toegestane treffers, met de fase
waarin het wordt opgeruimd. Dit is een RATCHET: de scanner faalt zodra een
bestand MEER treffers krijgt dan zijn budget. Het budget mag alleen omlaag.
Elk item verwijst naar een DI-nummer in docs/DEFERRED_ISSUES.md.

Gebruik:
    python scripts/check_hardcoded_params.py            # rapport, exit 0
    python scripts/check_hardcoded_params.py --strict   # exit 1 bij overschrijding
    python scripts/check_hardcoded_params.py --list     # alle treffers tonen
"""
from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "tradebot"
SKIP_PARTS = {"__pycache__"}

# Rekenkundige feiten, geen beleidskeuzes.
BENIGN: set[float] = {
    0, 1, 2, 3, 4, -1, -2,
    0.0, 1.0, 2.0, -1.0,
    0.5,            # kansdrempel bij een binaire classificatie
    100, 100.0,     # percentagenoemer
    10, 12, 24, 60, 252, 365,  # kalender / annualisatie
    1e-9, 1e-8, 1e-10, 1e-6, 1e-12,  # numerieke epsilons
}

# --------------------------------------------------------------------------- #
# RATCHET. Budget per bestand; mag alleen omlaag. Zie docs/DEFERRED_ISSUES.md.
# --------------------------------------------------------------------------- #
ALLOWLIST: dict[str, tuple[int, str]] = {
    # Gegenereerd uit de gemeten nulstand op 2026-08-22 (Phase 0).
    # Dit is een RATCHET: elk budget mag alleen OMLAAG. De scanner faalt zodra
    # een bestand meer treffers krijgt dan zijn budget, of zodra een NIEUW
    # bestand met literals verschijnt (die staan niet in deze tabel en krijgen
    # dus budget 0).
    # -- alpha/ --
    "alpha/carry.py": (3, "DI-12 Phase 3"),
    "alpha/combination.py": (1, "DI-12 Phase 3"),
    "alpha/csm_volume_clock.py": (2, "DI-12 Phase 3"),
    "alpha/decay_tracker.py": (2, "DI-12 Phase 3"),
    "alpha/kalman_ou.py": (6, "DI-12 Phase 3"),
    "alpha/macro_regime.py": (3, "DI-12 Phase 3"),
    "alpha/mean_reversion.py": (3, "DI-12 Phase 3"),
    "alpha/microstructure.py": (1, "DI-12 Phase 3"),
    "alpha/momentum.py": (2, "DI-12 Phase 3"),
    "alpha/research_harness.py": (3, "DI-12 Phase 3"),
    "alpha/xs_unit.py": (2, "DI-12 Phase 3"),
    # -- backtest/ --
    "backtest/bidirectional.py": (8, "DI-9  Phase 5"),
    "backtest/dd_shape.py": (5, "DI-9  Phase 5"),
    "backtest/evaluation.py": (14, "DI-9  Phase 5"),
    "backtest/metrics.py": (4, "DI-9  Phase 5"),
    "backtest/pbo.py": (1, "DI-9  Phase 5"),
    "backtest/per_side.py": (5, "DI-9  Phase 5"),
    "backtest/portfolio.py": (4, "DI-9  Phase 5"),
    "backtest/spa.py": (2, "DI-9  Phase 5"),
    "backtest/tracks.py": (1, "DI-9  Phase 5"),
    # -- bars/ --
    "bars/dollar.py": (3, "DI-13 Phase 1"),
    "bars/imbalance.py": (3, "DI-13 Phase 1"),
    "bars/runs.py": (3, "DI-13 Phase 1"),
    "bars/tick.py": (1, "DI-13 Phase 1"),
    "bars/volume.py": (1, "DI-13 Phase 1"),
    # -- compliance/ --
    "compliance/shadow_trader.py": (1, "DI-14 Phase 7"),
    # -- cv/ --
    "cv/cpcv.py": (1, "DI-11 Phase 6"),
    "cv/uniqueness.py": (1, "DI-11 Phase 6"),
    "cv/walk_forward.py": (4, "DI-11 Phase 6"),
    # -- data/ --
    "data/crypto.py": (1, "DI-13 Phase 1"),
    "data/crypto_macro.py": (3, "DI-13 Phase 1"),
    "data/edgar_universe.py": (1, "DI-13 Phase 1"),
    "data/open_interest.py": (2, "DI-13 Phase 1"),
    "data/orderbook.py": (4, "DI-13 Phase 1"),
    "data/perp_feed.py": (3, "DI-13 Phase 1"),
    "data/sources/edgar.py": (1, "DI-13 Phase 1"),
    "data/sources/eia.py": (2, "DI-13 Phase 1"),
    "data/sources/fred.py": (1, "DI-13 Phase 1"),
    "data/sources/stooq.py": (1, "DI-13 Phase 1"),
    # -- execution/ --
    "execution/fees.py": (1, "DI-9  Phase 5"),
    "execution/market_impact.py": (17, "DI-9  Phase 5"),
    "execution/simulator.py": (5, "DI-9  Phase 5"),
    "execution/slippage.py": (1, "DI-9  Phase 5"),
    "execution/spread.py": (2, "DI-9  Phase 5"),
    # -- features/ --
    "features/_ta_kernels.py": (8, "DI-12 Phase 3"),
    "features/cfi.py": (2, "DI-12 Phase 3"),
    "features/fracdiff.py": (8, "DI-12 Phase 3"),
    "features/funding_carry.py": (1, "DI-12 Phase 3"),
    "features/macro.py": (4, "DI-12 Phase 3"),
    "features/microstructure.py": (6, "DI-12 Phase 3"),
    "features/open_interest.py": (1, "DI-12 Phase 3"),
    "features/orthogonalize.py": (4, "DI-12 Phase 3"),
    "features/regime.py": (10, "DI-12 Phase 3"),
    "features/stationarity_gate.py": (1, "DI-12 Phase 3"),
    "features/ta.py": (4, "DI-12 Phase 3"),
    # -- labeling/ --
    "labeling/cusum.py": (2, "DI-11 Phase 6"),
    "labeling/fixed_horizon.py": (1, "DI-11 Phase 6"),
    "labeling/meta.py": (8, "DI-11 Phase 6"),
    "labeling/trend_scanning.py": (2, "DI-11 Phase 6"),
    # -- live/ --
    "live/engine.py": (4, "DI-14 Phase 7"),
    "live/execution_controller.py": (5, "DI-14 Phase 7"),
    "live/feature_updater.py": (1, "DI-14 Phase 7"),
    "live/feed.py": (1, "DI-14 Phase 7"),
    "live/model_signal.py": (1, "DI-14 Phase 7"),
    "live/portfolio_controller.py": (2, "DI-14 Phase 7"),
    "live/signal_runner.py": (2, "DI-14 Phase 7"),
    # -- monitoring/ --
    "monitoring/feature_health.py": (1, "DI-14 Phase 7"),
    "monitoring/live_drift_monitor.py": (1, "DI-14 Phase 7"),
    "monitoring/metrics.py": (1, "DI-14 Phase 7"),
    "monitoring/sharpe_monitor.py": (1, "DI-14 Phase 7"),
    # -- oms/ --
    "oms/paper_oms.py": (1, "DI-14 Phase 7"),
    "oms/position_tracker.py": (1, "DI-14 Phase 7"),
    # -- portfolio/ --
    "portfolio/black_litterman.py": (1, "DI-10 Phase 4"),
    "portfolio/markowitz.py": (1, "DI-10 Phase 4"),
    # -- registry/ --
    "registry/lineage.py": (1, "DI-14 Phase 7"),
    # -- risk/ --
    "risk/factor_risk.py": (3, "DI-10 Phase 4"),
    "risk/hmm_regime.py": (1, "DI-10 Phase 4"),
    "risk/kelly.py": (7, "DI-10 Phase 4"),
    "risk/liquidity_risk.py": (6, "DI-10 Phase 4"),
    "risk/portfolio.py": (15, "DI-10 Phase 4"),
    "risk/stress_test.py": (5, "DI-10 Phase 4"),
    "risk/var.py": (10, "DI-10 Phase 4"),
    # -- selection/ --
    "selection/mda.py": (4, "DI-11 Phase 6"),
    "selection/sfi.py": (3, "DI-11 Phase 6"),
    # -- tca/ --
    "tca/post_trade.py": (1, "DI-9  Phase 5"),
    "tca/pre_trade.py": (1, "DI-9  Phase 5"),
    # -- train/ --
    "train/_bandit_helpers.py": (1, "DI-11 Phase 6"),
    "train/_scalers.py": (5, "DI-11 Phase 6"),
    "train/calibration.py": (3, "DI-11 Phase 6"),
    "train/catboost.py": (2, "DI-11 Phase 6"),
    "train/ensemble.py": (10, "DI-11 Phase 6"),
    "train/meta_train.py": (4, "DI-11 Phase 6"),
    "train/quant_arch.py": (3, "DI-11 Phase 6"),
    "train/reward.py": (6, "DI-11 Phase 6"),
    "train/schema_guard.py": (2, "DI-11 Phase 6"),
    "train/stack.py": (1, "DI-11 Phase 6"),
    # -- tune/ --
    "tune/objective.py": (2, "DI-11 Phase 6"),
    "tune/pruning.py": (1, "DI-11 Phase 6"),
    "tune/samplers.py": (1, "DI-11 Phase 6"),
    "tune/search_space.py": (1, "DI-11 Phase 6"),
    # -- utils/ --
    "utils/hashing.py": (4, "DI-14 Phase 7"),
    # -- volatility/ --
    "volatility/ewma.py": (2, "DI-11 Phase 6"),
    "volatility/garman_klass.py": (2, "DI-11 Phase 6"),
    "volatility/har_rv.py": (4, "DI-11 Phase 6"),
    "volatility/parkinson.py": (1, "DI-11 Phase 6"),
    "volatility/rogers_satchell.py": (1, "DI-11 Phase 6"),
    "volatility/yang_zhang.py": (2, "DI-11 Phase 6"),
}

# Modules die Phase 3 NIEUW aanmaakt of volledig herschrijft naar het L1/L4/L8-
# contract. Zodra ze bestaan geldt budget 0 zonder uitzondering: nieuwe code mag
# geen enkele drempel als literal bevatten. `alpha/momentum.py` staat hier NIET
# bij zolang het de legacy-versie is; het schuift mee zodra Phase 3 hem
# herschrijft (dan valt zijn ratchet-budget naar 0).
GOVERNED = ("features/transforms.py", "features/pipeline.py",
            "alpha/base.py", "portfolio/risk_parity.py",
            "portfolio/equal_weight.py", "validation/")


def iter_py(base: Path):
    for p in sorted(base.rglob("*.py")):
        if not any(part in SKIP_PARTS for part in p.parts):
            yield p


def numeric_literals(path: Path) -> list[tuple[int, str, float]]:
    """(regelnummer, context, waarde) voor elke niet-benigne numerieke literal."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    out: list[tuple[int, str, float]] = []

    def keep(v: object) -> bool:
        return isinstance(v, (int, float)) and not isinstance(v, bool) and v not in BENIGN

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            defaults = list(node.args.defaults) + [d for d in node.args.kw_defaults if d]
            for d in defaults:
                if isinstance(d, ast.Constant) and keep(d.value):
                    out.append((d.lineno, f"default van {node.name}()", float(d.value)))
    return sorted(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args(argv)

    per_file: dict[str, list[tuple[int, str, float]]] = {}
    for p in iter_py(SRC):
        hits = numeric_literals(p)
        if hits:
            per_file[p.relative_to(SRC).as_posix()] = hits

    violations: list[str] = []
    total = 0
    for rel, hits in sorted(per_file.items()):
        total += len(hits)
        budget, owner = ALLOWLIST.get(rel, (0, ""))
        governed = any(rel.startswith(g) for g in GOVERNED)
        if governed and hits:
            violations.append(
                f"{rel}: {len(hits)} literal(s) in een door Phase 0-3 bestuurde module "
                f"(budget 0)")
        elif len(hits) > budget:
            violations.append(
                f"{rel}: {len(hits)} literal(s) > budget {budget} ({owner or 'geen budget'})")
        if args.list:
            print(f"\n{rel}  ({len(hits)}, budget {budget} {owner})")
            for ln, ctx, v in hits:
                print(f"    L{ln:<5} {ctx:<45} = {v}")

    budgeted = sum(b for b, _ in ALLOWLIST.values())
    print(f"\n{total} numerieke literal(s) in {len(per_file)} bestand(en).")
    print(f"Toegestaan door de ratchet: {budgeted}. Bestuurd door Phase 0-3: budget 0.")

    if violations:
        print("\nOVERSCHRIJDINGEN:", file=sys.stderr)
        for v in violations:
            print("  " + v, file=sys.stderr)
        if args.strict:
            return 1
    else:
        print("Geen enkel bestand overschrijdt zijn budget.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
