"""run_pipeline.py — Full training pipeline orchestrator.

Runs all 4 stages sequentially:
  Stage 1: build_features (all symbols)
  Stage 2: tune_hparams   (all symbol×side pairs, Optuna)
  Stage 3: train_cpcv     (all symbol×side pairs, CPCV ensemble)
  Stage 4: backtest_portfolio

Log: run_pipeline.log
"""
import subprocess
import sys
import time
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("run_pipeline.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("pipeline")

PYTHON = sys.executable
ROOT   = Path(__file__).parent

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
PAIRS   = [
    ("BTCUSDT", "LONG"),  ("BTCUSDT", "SHORT"),
    ("ETHUSDT", "LONG"),  ("ETHUSDT", "SHORT"),
    ("SOLUSDT", "LONG"),  ("SOLUSDT", "SHORT"),
]


def run(cmd: list[str], label: str) -> bool:
    logger.info("=== START: %s ===", label)
    t0 = time.time()
    result = subprocess.run(cmd, cwd=ROOT)
    elapsed = time.time() - t0
    if result.returncode == 0:
        logger.info("=== DONE:  %s (%.0fs) ===", label, elapsed)
        return True
    else:
        logger.error("=== FAIL:  %s (exit=%d, %.0fs) ===", label, result.returncode, elapsed)
        return False


def main() -> None:
    logger.info("Pipeline start — %d symbols, %d pairs", len(SYMBOLS), len(PAIRS))

    # ── Stage 1: build_features (all symbols in one call) ────────────────────
    ok = run(
        [PYTHON, "-m", "apps.build_features"],
        "Stage 1 — build_features (BTC+ETH+SOL)",
    )
    if not ok:
        logger.error("Stage 1 failed. Aborting.")
        sys.exit(1)

    # ── Stage 2: tune_hparams per pair ────────────────────────────────────────
    tune_failures = []
    for sym, side in PAIRS:
        pair_key = f"{sym}_{side}"
        ok = run(
            [PYTHON, "-m", "apps.tune_hparams", f"+pair={pair_key}"],
            f"Stage 2 — tune_hparams {pair_key}",
        )
        if not ok:
            tune_failures.append(pair_key)

    if tune_failures:
        logger.warning("Stage 2 failures (skipping train for these): %s", tune_failures)

    # ── Stage 3: train_cpcv per pair ─────────────────────────────────────────
    train_failures = []
    for sym, side in PAIRS:
        pair_key = f"{sym}_{side}"
        if pair_key in tune_failures:
            logger.warning("Skipping %s train — no hparams.", pair_key)
            continue
        ok = run(
            [PYTHON, "-m", "apps.train_cpcv", f"+pair={pair_key}"],
            f"Stage 3 — train_cpcv {pair_key}",
        )
        if not ok:
            train_failures.append(pair_key)

    if train_failures:
        logger.warning("Stage 3 failures: %s", train_failures)

    # ── Stage 4: backtest_portfolio ───────────────────────────────────────────
    run(
        [PYTHON, "-m", "apps.backtest_portfolio"],
        "Stage 4 — backtest_portfolio",
    )

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
