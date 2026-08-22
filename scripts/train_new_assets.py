"""scripts/train_new_assets.py
Parallel training pipeline for AVAXUSDT, LINKUSDT, DOTUSDT.

Stage 1 : build_features  — 3 parallel (AVAX | LINK | DOT)
Stage 2 : tune_hparams    — 6 parallel (3 symbols × 2 sides)
Stage 3 : train_cpcv      — 6 parallel (3 symbols × 2 sides)

Stages are sequential (2 waits on 1, 3 waits on 2).
Within each stage all jobs run in parallel via ThreadPoolExecutor.

Usage:
    python scripts/train_new_assets.py
    python scripts/train_new_assets.py --stage 1   # only build_features
    python scripts/train_new_assets.py --stage 2   # only tune_hparams
    python scripts/train_new_assets.py --stage 3   # only train_cpcv
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT   = Path(__file__).resolve().parents[1]   # Tradebot/
LOGS   = ROOT / "logs" / "train_new_assets"
LOGS.mkdir(parents=True, exist_ok=True)

PY     = sys.executable
SYMS   = ["AVAXUSDT", "LINKUSDT", "DOTUSDT"]
PAIRS  = [(s, side) for s in SYMS for side in ("LONG", "SHORT")]

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-7s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOGS / "orchestrator.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("orchestrate")


def run(args: list[str], label: str) -> bool:
    log_file = LOGS / f"{label.replace('/', '_').replace('[', '').replace(']', '')}.log"
    log.info("[START] %-36s -> %s", label, log_file.name)
    t0 = time.time()
    with open(log_file, "w", encoding="utf-8") as lf:
        proc = subprocess.run(
            args,
            cwd=str(ROOT),
            stdout=lf,
            stderr=subprocess.STDOUT,
        )
    elapsed = time.time() - t0
    if proc.returncode == 0:
        log.info("[DONE ] %-36s  %.0fs", label, elapsed)
        return True
    tail = log_file.read_text(encoding="utf-8", errors="replace")[-600:]
    log.error("[FAIL ] %-36s  %.0fs  rc=%d\n%s", label, elapsed, proc.returncode, tail)
    return False


def stage1() -> bool:
    log.info("=" * 60)
    log.info("STAGE 1 — build_features (%d symbols parallel)", len(SYMS))
    log.info("=" * 60)
    jobs = {
        (PY, "-m", "apps.build_features", f"+symbol={sym}"): f"build_features[{sym}]"
        for sym in SYMS
    }
    with ThreadPoolExecutor(max_workers=len(SYMS)) as pool:
        futs = {pool.submit(run, list(cmd), lbl): lbl for cmd, lbl in jobs.items()}
        results = {lbl: fut.result() for fut, lbl in [(f, futs[f]) for f in as_completed(futs)]}
    failed = [lbl for lbl, ok in results.items() if not ok]
    if failed:
        log.error("Stage 1 FAILED: %s", failed)
    return not failed


def stage2() -> bool:
    log.info("=" * 60)
    log.info("STAGE 2 — tune_hparams (%d pairs parallel)", len(PAIRS))
    log.info("=" * 60)
    jobs = {
        (PY, "-m", "apps.tune_hparams", f"+pair={sym}_{side}"): f"tune_hparams[{sym}_{side}]"
        for sym, side in PAIRS
    }
    with ThreadPoolExecutor(max_workers=len(PAIRS)) as pool:
        futs = {pool.submit(run, list(cmd), lbl): lbl for cmd, lbl in jobs.items()}
        results = {lbl: fut.result() for fut, lbl in [(f, futs[f]) for f in as_completed(futs)]}
    failed = [lbl for lbl, ok in results.items() if not ok]
    if failed:
        log.error("Stage 2 FAILED: %s", failed)
    return not failed


def stage3() -> bool:
    log.info("=" * 60)
    log.info("STAGE 3 — train_cpcv (%d pairs parallel)", len(PAIRS))
    log.info("=" * 60)
    jobs = {
        (PY, "-m", "apps.train_cpcv", f"+pair={sym}_{side}"): f"train_cpcv[{sym}_{side}]"
        for sym, side in PAIRS
    }
    with ThreadPoolExecutor(max_workers=len(PAIRS)) as pool:
        futs = {pool.submit(run, list(cmd), lbl): lbl for cmd, lbl in jobs.items()}
        results = {lbl: fut.result() for fut, lbl in [(f, futs[f]) for f in as_completed(futs)]}
    failed = [lbl for lbl, ok in results.items() if not ok]
    if failed:
        log.error("Stage 3 FAILED: %s", failed)
    return not failed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, choices=[1, 2, 3], default=None,
                        help="Run only this stage (default: all)")
    args = parser.parse_args()

    stages = [args.stage] if args.stage else [1, 2, 3]
    t_total = time.time()

    for s in stages:
        ok = {1: stage1, 2: stage2, 3: stage3}[s]()
        if not ok:
            log.error("Pipeline aborted at stage %d.", s)
            sys.exit(1)

    log.info("=" * 60)
    log.info("ALL STAGES COMPLETE  total=%.0fs", time.time() - t_total)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
