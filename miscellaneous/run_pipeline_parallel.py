"""run_pipeline_parallel.py — Parallel pipeline orchestrator for tradebot.

Resume-aware: stages whose output artefacts already exist are skipped.

Stages:
  1. build_features   (6 symbols in parallel)
  2. tune_hparams     (12 symbol×side pairs in parallel)
  3. train_cpcv       (12 symbol×side pairs in parallel)
  4. backtest_portfolio (single, after all Stage 3 complete)

Usage:
    python run_pipeline_parallel.py              # full pipeline (auto-resumes)
    python run_pipeline_parallel.py --stage 1    # only build_features
    python run_pipeline_parallel.py --stage 2    # only tune_hparams
    python run_pipeline_parallel.py --stage 3    # only train_cpcv
    python run_pipeline_parallel.py --stage 4    # only backtest
    python run_pipeline_parallel.py --force      # ignore existing artefacts, redo all
    python run_pipeline_parallel.py --max-workers 4
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed, Future
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("pipeline")

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT"]
SIDES   = ["LONG", "SHORT"]
PAIRS   = [(sym, side) for sym in SYMBOLS for side in SIDES]

ROOT       = Path(__file__).resolve().parent
ARTS       = ROOT / "artefacts"
REPORTS    = ROOT / "reports"


# =============================================================================
# Resume helpers — check whether a stage's output already exists
# =============================================================================

def _s1_done(sym: str) -> bool:
    return (
        (ARTS / "features" / f"{sym}.parquet").exists()
        and (ARTS / "events"   / f"{sym}.parquet").exists()
    )

def _s2_done(sym: str, side: str) -> bool:
    return (ARTS / "hparams" / f"{sym}_{side}.json").exists()

def _s3_done(sym: str, side: str) -> bool:
    return (ARTS / "oos_probs" / f"{sym}_{side}.parquet").exists()

def _s4_done() -> bool:
    return (ARTS / "portfolio" / "result.joblib").exists()


# =============================================================================
# Core runner
# =============================================================================

def _run(cmd: list[str], label: str) -> tuple[str, int, str, str]:
    t0 = time.time()
    logger.info("[START] %s", label)
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
    elapsed = time.time() - t0
    status = "OK" if result.returncode == 0 else "FAILED"
    logger.info("[%s] %s  (%.0fs)", status, label, elapsed)
    if result.returncode != 0:
        tail = "\n".join(result.stderr.strip().splitlines()[-35:])
        logger.error("[%s] stderr:\n%s", label, tail)
    return label, result.returncode, result.stdout, result.stderr


def _python() -> str:
    return sys.executable


# =============================================================================
# Stage 1 — build_features
# =============================================================================

def stage1_build_features(max_workers: int = 6, force: bool = False) -> dict[str, int]:
    pending = [s for s in SYMBOLS if force or not _s1_done(s)]
    skipped = [s for s in SYMBOLS if not force and _s1_done(s)]

    if skipped:
        logger.info("Stage 1: skipping %s (artefacts exist)", skipped)

    if not pending:
        logger.info("Stage 1: all symbols already built — nothing to do.")
        return {f"build_features[{s}]": 0 for s in SYMBOLS}

    logger.info("=" * 60)
    logger.info("STAGE 1: build_features  %d symbols  max_workers=%d", len(pending), min(len(pending), max_workers))
    logger.info("=" * 60)

    jobs = [
        ([_python(), "-m", "apps.build_features", f"+symbol={sym}"], f"build_features[{sym}]")
        for sym in pending
    ]

    results: dict[str, int] = {f"build_features[{s}]": 0 for s in skipped}
    with ProcessPoolExecutor(max_workers=min(max_workers, len(pending))) as ex:
        futures: list[Future] = [ex.submit(_run, cmd, label) for cmd, label in jobs]
        for f in as_completed(futures):
            label, rc, _o, _e = f.result()
            results[label] = rc

    _report("Stage 1", results)
    return results


# =============================================================================
# Stage 2 — tune_hparams
# =============================================================================

def stage2_tune_hparams(
    max_workers: int = 6,
    failed_stage1: Optional[set[str]] = None,
    force: bool = False,
) -> dict[str, int]:
    failed_stage1 = failed_stage1 or set()

    pending = [
        (sym, side) for sym, side in PAIRS
        if sym not in failed_stage1
        and (force or not _s2_done(sym, side))
    ]
    skipped = [
        (sym, side) for sym, side in PAIRS
        if sym not in failed_stage1
        and not force and _s2_done(sym, side)
    ]

    if skipped:
        logger.info("Stage 2: skipping %s (hparams exist)", [f"{s}/{d}" for s, d in skipped])

    if not pending:
        logger.info("Stage 2: all pairs already tuned — nothing to do.")
        return {f"tune_hparams[{s}/{d}]": 0 for s, d in PAIRS}

    logger.info("=" * 60)
    logger.info("STAGE 2: tune_hparams  %d pairs  max_workers=%d", len(pending), min(max_workers, len(pending)))
    logger.info("=" * 60)

    jobs = [
        (
            [_python(), "-m", "apps.tune_hparams", f"+pair={sym}_{side}"],
            f"tune_hparams[{sym}/{side}]",
        )
        for sym, side in pending
    ]

    results: dict[str, int] = {f"tune_hparams[{s}/{d}]": 0 for s, d in skipped}
    with ProcessPoolExecutor(max_workers=min(max_workers, len(pending))) as ex:
        futures = [ex.submit(_run, cmd, label) for cmd, label in jobs]
        for f in as_completed(futures):
            label, rc, _o, _e = f.result()
            results[label] = rc

    _report("Stage 2", results)
    return results


# =============================================================================
# Stage 3 — train_cpcv
# =============================================================================

def stage3_train_cpcv(
    max_workers: int = 6,
    failed_stage2: Optional[set[str]] = None,
    force: bool = False,
) -> dict[str, int]:
    failed_stage2 = failed_stage2 or set()

    pending = [
        (sym, side) for sym, side in PAIRS
        if f"{sym}_{side}" not in failed_stage2
        and (force or not _s3_done(sym, side))
    ]
    skipped = [
        (sym, side) for sym, side in PAIRS
        if f"{sym}_{side}" not in failed_stage2
        and not force and _s3_done(sym, side)
    ]

    if skipped:
        logger.info("Stage 3: skipping %s (oos_probs exist)", [f"{s}/{d}" for s, d in skipped])

    if not pending:
        logger.info("Stage 3: all pairs already trained — nothing to do.")
        return {f"train_cpcv[{s}/{d}]": 0 for s, d in PAIRS}

    logger.info("=" * 60)
    logger.info("STAGE 3: train_cpcv  %d pairs  max_workers=%d", len(pending), min(max_workers, len(pending)))
    logger.info("=" * 60)

    jobs = [
        (
            [_python(), "-m", "apps.train_cpcv", f"+pair={sym}_{side}"],
            f"train_cpcv[{sym}/{side}]",
        )
        for sym, side in pending
    ]

    results: dict[str, int] = {f"train_cpcv[{s}/{d}]": 0 for s, d in skipped}
    with ProcessPoolExecutor(max_workers=min(max_workers, len(pending))) as ex:
        futures = [ex.submit(_run, cmd, label) for cmd, label in jobs]
        for f in as_completed(futures):
            label, rc, _o, _e = f.result()
            results[label] = rc

    _report("Stage 3", results)
    return results


# =============================================================================
# Stage 4 — backtest_portfolio
# =============================================================================

def stage4_backtest_portfolio(force: bool = False) -> int:
    if not force and _s4_done():
        logger.info("Stage 4: portfolio result already exists — skipping.")
        _print_metrics()
        return 0

    logger.info("=" * 60)
    logger.info("STAGE 4: backtest_portfolio")
    logger.info("=" * 60)

    label, rc, _o, _e = _run(
        [_python(), "-m", "apps.backtest_portfolio"],
        "backtest_portfolio",
    )
    if rc == 0:
        _print_metrics()
    return rc


def _print_metrics() -> None:
    metrics_path = REPORTS / "portfolio_metrics.json"
    if metrics_path.exists():
        m = json.loads(metrics_path.read_text())
        logger.info("=" * 60)
        logger.info("PORTFOLIO METRICS:")
        for k, v in m.items():
            logger.info("  %-32s %s", k, v)
        logger.info("=" * 60)


# =============================================================================
# Helpers
# =============================================================================

def _report(stage: str, results: dict[str, int]) -> None:
    ok  = [k for k, v in results.items() if v == 0]
    bad = [k for k, v in results.items() if v != 0]
    logger.info("%s complete: %d OK, %d FAILED", stage, len(ok), len(bad))
    if bad:
        logger.warning("FAILED jobs: %s", bad)


def _failed_symbols(stage_results: dict[str, int]) -> set[str]:
    failed: set[str] = set()
    for label, rc in stage_results.items():
        if rc != 0 and "[" in label:
            token = label.split("[")[1].rstrip("]")
            failed.add(token.split("/")[0])
    return failed


def _failed_pairs(stage_results: dict[str, int]) -> set[str]:
    failed: set[str] = set()
    for label, rc in stage_results.items():
        if rc != 0 and "[" in label:
            token = label.split("[")[1].rstrip("]")
            parts = token.split("/")
            if len(parts) == 2:
                failed.add(f"{parts[0]}_{parts[1]}")
    return failed


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Resume-aware parallel tradebot pipeline")
    parser.add_argument("--stage", default="1,2,3,4",
                        help="Comma-separated stages to run (default: 1,2,3,4)")
    parser.add_argument("--max-workers", type=int, default=6,
                        help="Max parallel workers per stage (default: 6)")
    parser.add_argument("--force", action="store_true",
                        help="Ignore existing artefacts and redo everything")
    args = parser.parse_args()

    requested = {int(s.strip()) for s in args.stage.split(",")}
    mw = args.max_workers
    force = args.force

    # Print current artefact status
    logger.info("Artefact status:")
    for sym in SYMBOLS:
        s1 = "✓" if _s1_done(sym) else "✗"
        pairs_s2 = sum(_s2_done(sym, side) for side in SIDES)
        pairs_s3 = sum(_s3_done(sym, side) for side in SIDES)
        logger.info("  %s  S1=%s  S2=%d/2  S3=%d/2", sym, s1, pairs_s2, pairs_s3)

    t_total = time.time()
    s1_results: dict[str, int] = {}
    s2_results: dict[str, int] = {}
    s3_results: dict[str, int] = {}

    if 1 in requested:
        s1_results = stage1_build_features(max_workers=mw, force=force)
    if 2 in requested:
        s2_results = stage2_tune_hparams(
            max_workers=mw,
            failed_stage1=_failed_symbols(s1_results),
            force=force,
        )
    if 3 in requested:
        s3_results = stage3_train_cpcv(
            max_workers=mw,
            failed_stage2=_failed_pairs(s2_results),
            force=force,
        )
    if 4 in requested:
        stage4_backtest_portfolio(force=force)

    elapsed = time.time() - t_total
    logger.info("Pipeline finished in %.0f min %.0f sec", elapsed // 60, elapsed % 60)


if __name__ == "__main__":
    main()
