"""run_parallel_pipeline.py — Parallel Stages 2-4 runner.

Runs Stage 2 (tune_hparams) and Stage 3 (train_cpcv) in parallel batches
of MAX_CONCURRENT pairs, then runs Stage 4 (backtest_portfolio) sequentially.

Usage:
    python run_parallel_pipeline.py

Log: run_parallel_pipeline.log
"""
from __future__ import annotations

import concurrent.futures
import logging
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("run_parallel_pipeline.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("parallel_pipeline")

PYTHON = sys.executable
ROOT = Path(__file__).parent

# 5-asset compliant universe (params.yaml). BTCUSDT excluded per CHIEF audit
# 2026-05-28 (G-1: negative alpha) and has no built features — keep it out so
# the retrain stays consistent with the deployable portfolio.
SYMBOLS = ["ETHUSDT", "SOLUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT"]

# Phase B (2026-05-30): full 200-trial retrain on the REMAINING sides only.
# The 3 structurally-negative sides were dropped in Phase A (zeroed in the
# backtest via portfolio.disabled_sides) so there is no point spending Optuna
# budget on them — exclude them here too.
DISABLED_SIDES = {"DOTUSDT_LONG", "SOLUSDT_LONG", "ETHUSDT_SHORT"}
PAIRS = [
    (sym, side)
    for sym in SYMBOLS
    for side in ("LONG", "SHORT")
    if f"{sym}_{side}" not in DISABLED_SIDES
]

# Phase B: run tune to the FULL 200-trial budget (no 12-min timeout). This is
# the "volledige Optuna retrain" the CHIEF audit said was needed; the earlier
# run was time-boxed at optuna_timeout_seconds=720 and reached 200 for only
# ETHUSDT_LONG. null disables the timeout so each study runs to n_trials=200.
TUNE_OVERRIDES = ["training.optuna_timeout_seconds=null"]

# Fastest stable parallelism for this box (6 physical cores / 16 GB RAM):
# CatBoost defaults to thread_count=-1 (all 6 cores per job). With 3 concurrent
# jobs the cores stay saturated via sub-linear-scaling efficiency gains while
# RAM stays ~9 GB (3 × ~3 GB peak). Going to 4+ oversubscribes CPU and risks
# paging — slower, not faster. thread_count is left at default to preserve
# CatBoost determinism (regression/bit-equivalence baselines).
MAX_CONCURRENT = 3


def run_stage(cmd: list[str], label: str, env_override: dict | None = None) -> bool:
    logger.info("START: %s", label)
    t0 = time.time()
    env = None
    if env_override:
        import os
        env = {**os.environ, **env_override}
    result = subprocess.run(cmd, cwd=str(ROOT), env=env)
    elapsed = time.time() - t0
    if result.returncode == 0:
        logger.info("DONE:  %s (%.0fs)", label, elapsed)
        return True
    else:
        logger.error("FAIL:  %s (exit=%d, %.0fs)", label, result.returncode, elapsed)
        return False


def run_pair_tune(sym: str, side: str) -> tuple[str, bool]:
    pair_key = f"{sym}_{side}"
    ok = run_stage(
        [PYTHON, "-m", "apps.tune_hparams", f"+pair={pair_key}", *TUNE_OVERRIDES],
        f"S2-tune {pair_key}",
        env_override={"PYTHONIOENCODING": "utf-8"},
    )
    return pair_key, ok


def run_pair_train(sym: str, side: str) -> tuple[str, bool]:
    pair_key = f"{sym}_{side}"
    ok = run_stage(
        [PYTHON, "-m", "apps.train_cpcv", f"+pair={pair_key}"],
        f"S3-train {pair_key}",
        env_override={"PYTHONIOENCODING": "utf-8"},
    )
    return pair_key, ok


def run_parallel(fn, pairs: list[tuple[str, str]], max_workers: int) -> dict[str, bool]:
    results: dict[str, bool] = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(fn, sym, side): f"{sym}_{side}" for sym, side in pairs}
        for fut in concurrent.futures.as_completed(futures):
            pair_key, ok = fut.result()
            results[pair_key] = ok
    return results


def main() -> None:
    logger.info("=== Parallel Pipeline START — %d pairs, %d concurrent ===",
                len(PAIRS), MAX_CONCURRENT)

    # ── Stage 2: tune_hparams (parallel) ────────────────────────────────────
    logger.info("=== Stage 2: tune_hparams — all %d pairs ===", len(PAIRS))
    tune_results = run_parallel(run_pair_tune, PAIRS, max_workers=MAX_CONCURRENT)

    tune_ok = [k for k, v in tune_results.items() if v]
    tune_fail = [k for k, v in tune_results.items() if not v]
    logger.info("Stage 2 done. OK=%d, FAIL=%d: %s", len(tune_ok), len(tune_fail), tune_fail)

    # ── Stage 3: train_cpcv (parallel, only successful pairs) ───────────────
    train_pairs = [(s, si) for s, si in PAIRS if f"{s}_{si}" in tune_ok]
    logger.info("=== Stage 3: train_cpcv — %d pairs ===", len(train_pairs))
    train_results = run_parallel(run_pair_train, train_pairs, max_workers=MAX_CONCURRENT)

    train_ok = [k for k, v in train_results.items() if v]
    train_fail = [k for k, v in train_results.items() if not v]
    logger.info("Stage 3 done. OK=%d, FAIL=%d: %s", len(train_ok), len(train_fail), train_fail)

    # ── Stage 4: backtest_portfolio ─────────────────────────────────────────
    logger.info("=== Stage 4: backtest_portfolio ===")
    ok = run_stage(
        [PYTHON, "-m", "apps.backtest_portfolio"],
        "S4-backtest",
        env_override={"PYTHONIOENCODING": "utf-8"},
    )
    if not ok:
        logger.error("Stage 4 FAILED.")
        sys.exit(1)

    logger.info("=== Parallel Pipeline COMPLETE ===")


if __name__ == "__main__":
    main()
