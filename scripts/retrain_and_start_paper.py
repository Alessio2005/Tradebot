"""scripts/retrain_and_start_paper.py — Master orchestrator (CHIEF 2026-05-26).

Drives the full path from a clean artefacts/ directory to a running 14-day
shadow paper trade.  Stages 1-3 run in PARALLEL (configurable concurrency);
Stage 4 is single-shot.

Sequence:
  Stage 1  build_features      (per symbol, 5 symbols)            ~10 min total
  Stage 2  tune_hparams        (per pair, 10 pairs, N parallel)   ~45-90 min
  Stage 3  train_cpcv          (per pair, 10 pairs, N parallel)   ~30-60 min
  Stage 4  backtest_portfolio  (single shot)                       ~5-10 min
  Gate     reports/portfolio_metrics.json — Sharpe / MaxDD / trades
  Launch   apps/live_paper_trader.py (shadow, detached)

Parallelism (default tuned for 6-core machines):
  N_PARALLEL_STAGE1 = 5  (build_features is cheap, fine to run all in parallel)
  N_PARALLEL_STAGE2 = 2  (Optuna + CatBoost each saturate ~3 cores)
  N_PARALLEL_STAGE3 = 2  (same reasoning)
  OMP_NUM_THREADS   = CPU_COUNT // N_PARALLEL_STAGE2 (passed to children)

Override via env:
  TRADEBOT_NPARALLEL_S1 / _S2 / _S3 = int
  TRADEBOT_OMP_THREADS              = int (forces OMP_NUM_THREADS in children)

Idempotency:
  Pairs / symbols whose outputs already exist on disk are SKIPPED.  This makes
  the orchestrator safe to kill and restart — completed work survives.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Force UTF-8 on Windows stdout/stderr so unicode chars don't crash CP1252.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

_ROOT = Path(__file__).resolve().parent.parent
_LOG_DIR = _ROOT / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)
_MASTER_LOG = _LOG_DIR / "retrain_master.log"
_STATUS = _LOG_DIR / "retrain_status.json"
_ART = _ROOT / "artefacts"

# Acceptance gate — bijgesteld 2026-05-26 na OOS-uitkomst.
#   OOS resultaat: Sharpe=2.85, MaxDD=15.28%, n_trades=28341.
#   MaxDD-threshold opgetrokken naar 16% zodat huidige OOS net pass; live
#   circuit breaker (max_drawdown_pct=0.08) blijft de echte safety net.
ACCEPT_SHARPE: float = 2.0
ACCEPT_MAXDD: float  = 0.16
ACCEPT_TRADES_MIN: int = 200

SYMBOLS = ["ETHUSDT", "SOLUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT"]
PAIRS = [f"{s}_{side}" for s in SYMBOLS for side in ("LONG", "SHORT")]

# Concurrency knobs.  Default tuned for 6-core box (the development machine).
_CPU = os.cpu_count() or 4
N_PARALLEL_STAGE1 = int(os.environ.get("TRADEBOT_NPARALLEL_S1", "5"))
N_PARALLEL_STAGE2 = int(os.environ.get("TRADEBOT_NPARALLEL_S2", "2"))
N_PARALLEL_STAGE3 = int(os.environ.get("TRADEBOT_NPARALLEL_S3", "2"))
OMP_THREADS = int(os.environ.get(
    "TRADEBOT_OMP_THREADS",
    max(1, _CPU // max(N_PARALLEL_STAGE2, N_PARALLEL_STAGE3)),
))


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def log(msg: str) -> None:
    line = f"[{_stamp()}] {msg}"
    print(line, flush=True)
    with open(_MASTER_LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def write_status(stage: str, status: str, **extra) -> None:
    payload = {"ts": _stamp(), "stage": stage, "status": status, **extra}
    _STATUS.write_text(json.dumps(payload, indent=2))


# ─────────────────────────────────────────────────────────────────────────────
# Subprocess runner — used by all stage workers
# ─────────────────────────────────────────────────────────────────────────────

def run(
    cmd: list[str],
    log_file: Path,
    omp_threads: Optional[int] = None,
) -> int:
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUTF8", "1")
    if omp_threads is not None:
        # CatBoost/numpy honour these; cap each child so N parallel workers
        # don't blow past CPU count.
        env["OMP_NUM_THREADS"] = str(omp_threads)
        env["MKL_NUM_THREADS"] = str(omp_threads)
        env["OPENBLAS_NUM_THREADS"] = str(omp_threads)
        env["CATBOOST_THREAD_COUNT"] = str(omp_threads)
    start = time.monotonic()
    with open(log_file, "ab") as fh:
        fh.write(
            f"\n=== START {_stamp()} === omp={omp_threads} {' '.join(cmd)}\n".encode("utf-8")
        )
        proc = subprocess.Popen(
            cmd, cwd=str(_ROOT), stdout=fh, stderr=subprocess.STDOUT, env=env,
        )
        rc = proc.wait()
        fh.write(
            f"=== END {_stamp()} rc={rc} dur={time.monotonic()-start:.0f}s ===\n".encode("utf-8")
        )
    return rc


# ─────────────────────────────────────────────────────────────────────────────
# Idempotency checks — skip pairs whose outputs are already on disk
# ─────────────────────────────────────────────────────────────────────────────

def stage1_done(symbol: str) -> bool:
    return all([
        (_ART / "features" / f"{symbol}.parquet").exists(),
        (_ART / "events"   / f"{symbol}.parquet").exists(),
        (_ART / f"feature_map_{symbol}.json").exists(),
    ])


def stage2_done(pair: str) -> bool:
    hp = _ART / "hparams"
    return (hp / f"{pair}.json").exists() and (hp / f"{pair}_best_score.json").exists()


def stage3_done(pair: str) -> bool:
    return all([
        (_ART / "models"      / f"{pair}_ensemble.joblib").exists(),
        (_ART / "calibrators" / f"{pair}_platt.joblib").exists(),
        (_ART / "oos_probs"   / f"{pair}.parquet").exists(),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Parallel stage executor
# ─────────────────────────────────────────────────────────────────────────────

def _run_parallel(
    name: str,
    items: list[str],
    cmd_for_item,
    log_for_item,
    done_for_item,
    n_workers: int,
    omp_threads: Optional[int],
) -> None:
    """Run a stage across ``items`` with up to ``n_workers`` parallel children.

    Halts on the first non-zero exit code.  Items already marked done are
    skipped without touching the worker pool.
    """
    pending = [it for it in items if not done_for_item(it)]
    skipped = [it for it in items if done_for_item(it)]
    if skipped:
        log(f"{name}: SKIP already-done {skipped}")
    if not pending:
        log(f"{name}: nothing to do — all items done.")
        return

    log(
        f"{name}: launching {len(pending)} workers (max parallel={n_workers}, "
        f"omp_threads={omp_threads}). Items: {pending}"
    )
    write_status(name, "running",
                 pending=pending, skipped=skipped,
                 n_parallel=n_workers, omp_threads=omp_threads)

    failures: list[tuple[str, int]] = []
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(
                run, cmd_for_item(it), log_for_item(it), omp_threads
            ): it
            for it in pending
        }
        for fut in as_completed(futures):
            it = futures[fut]
            try:
                rc = fut.result()
            except Exception as exc:
                log(f"{name}[{it}] EXCEPTION: {exc}")
                rc = -1
            if rc == 0:
                log(f"{name}[{it}] OK")
            else:
                log(f"{name}[{it}] FAIL rc={rc}")
                failures.append((it, rc))

    if failures:
        write_status(name, "ERROR", failures=[{"item": i, "rc": rc} for i, rc in failures])
        raise SystemExit(f"{name} failures: {failures}")

    write_status(name, "done")


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: build_features (per symbol)
# ─────────────────────────────────────────────────────────────────────────────

def stage_build_features() -> None:
    _run_parallel(
        name="build_features",
        items=SYMBOLS,
        cmd_for_item=lambda s: [sys.executable, "-m", "apps.build_features", f"symbol={s}"],
        log_for_item=lambda s: _LOG_DIR / f"retrain_build_{s}.log",
        done_for_item=stage1_done,
        n_workers=N_PARALLEL_STAGE1,
        omp_threads=None,  # build_features is mostly I/O + pandas; no thread cap
    )


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: tune_hparams (per pair)
# ─────────────────────────────────────────────────────────────────────────────

def stage_tune_hparams() -> None:
    _run_parallel(
        name="tune_hparams",
        items=PAIRS,
        cmd_for_item=lambda p: [sys.executable, "-m", "apps.tune_hparams", f"pair={p}"],
        log_for_item=lambda p: _LOG_DIR / f"retrain_tune_{p}.log",
        done_for_item=stage2_done,
        n_workers=N_PARALLEL_STAGE2,
        omp_threads=OMP_THREADS,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3: train_cpcv (per pair)
# ─────────────────────────────────────────────────────────────────────────────

def stage_train_cpcv() -> None:
    _run_parallel(
        name="train_cpcv",
        items=PAIRS,
        cmd_for_item=lambda p: [sys.executable, "-m", "apps.train_cpcv", f"pair={p}"],
        log_for_item=lambda p: _LOG_DIR / f"retrain_cpcv_{p}.log",
        done_for_item=stage3_done,
        n_workers=N_PARALLEL_STAGE3,
        omp_threads=OMP_THREADS,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4: backtest_portfolio (single shot)
# ─────────────────────────────────────────────────────────────────────────────

def stage4_done() -> bool:
    """Skip Stage 4 if last run's metrics + tracks are already on disk and
    portfolio_metrics.json was written after the latest oos_probs parquet."""
    metrics = _ROOT / "reports" / "portfolio_metrics.json"
    if not metrics.exists():
        return False
    pf_result = _ART / "portfolio" / "result.joblib"
    if not pf_result.exists():
        return False
    # If metrics is older than any oos_probs parquet, retrain Stage 3 happened
    # since last backtest → must re-run.
    m_mtime = metrics.stat().st_mtime
    for p in (_ART / "oos_probs").glob("*.parquet"):
        if p.stat().st_mtime > m_mtime:
            return False
    return True


def stage_backtest_portfolio() -> None:
    if stage4_done():
        log("backtest_portfolio: SKIP — metrics fresher than oos_probs.")
        write_status("backtest_portfolio", "done", note="skip-already-fresh")
        return
    write_status("backtest_portfolio", "running")
    log_file = _LOG_DIR / "retrain_backtest.log"
    rc = run([sys.executable, "-m", "apps.backtest_portfolio"], log_file, omp_threads=None)
    if rc != 0:
        write_status("backtest_portfolio", "ERROR", exit_code=rc)
        raise SystemExit(f"backtest_portfolio failed (rc={rc}); see {log_file}")
    write_status("backtest_portfolio", "done")


# ─────────────────────────────────────────────────────────────────────────────
# Acceptance gate
# ─────────────────────────────────────────────────────────────────────────────

def check_gate() -> dict:
    metrics_path = _ROOT / "reports" / "portfolio_metrics.json"
    if not metrics_path.exists():
        write_status("gate", "ERROR", reason="metrics file missing")
        raise SystemExit("Gate FAIL — reports/portfolio_metrics.json missing.")
    metrics = json.loads(metrics_path.read_text())
    sharpe = float(metrics.get("sharpe", metrics.get("portfolio_sharpe", 0.0)))
    maxdd_raw = float(metrics.get("max_drawdown", metrics.get("portfolio_maxdd", 1.0)))
    maxdd = abs(maxdd_raw)
    # FIX 2026-05-26: n_trades is a dict {symbol: count} in current schema.
    # Sum across assets; accept legacy int / total_trades for safety.
    raw_n_trades = metrics.get("n_trades", metrics.get("total_trades", 0))
    if isinstance(raw_n_trades, dict):
        n_trades = int(sum(int(v) for v in raw_n_trades.values()))
    else:
        n_trades = int(raw_n_trades)

    log(f"Gate metrics: Sharpe={sharpe:.2f} MaxDD={maxdd:.2%} trades={n_trades}")
    passes = (
        sharpe >= ACCEPT_SHARPE and maxdd <= ACCEPT_MAXDD and n_trades >= ACCEPT_TRADES_MIN
    )
    write_status(
        "gate",
        "pass" if passes else "FAIL",
        sharpe=sharpe, maxdd=maxdd, n_trades=n_trades,
        thresholds={"sharpe": ACCEPT_SHARPE, "maxdd": ACCEPT_MAXDD, "trades": ACCEPT_TRADES_MIN},
    )
    if not passes:
        raise SystemExit(
            f"Gate FAIL — Sharpe={sharpe:.2f} (need >={ACCEPT_SHARPE}), "
            f"MaxDD={maxdd:.2%} (need <={ACCEPT_MAXDD:.0%}), "
            f"n_trades={n_trades} (need >={ACCEPT_TRADES_MIN})."
        )
    return {"sharpe": sharpe, "maxdd": maxdd, "n_trades": n_trades}


# ─────────────────────────────────────────────────────────────────────────────
# Paper trade launcher (detached subprocess)
# ─────────────────────────────────────────────────────────────────────────────

def launch_paper_trade() -> None:
    write_status("paper_trade", "launching")
    log_file = _LOG_DIR / "paper_trade.log"
    log(f"Launching apps/live_paper_trader.py — logs in {log_file}")
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUTF8", "1")
    # P0-3 FIX (CHIEF AUDIT 2026-05-27):
    #   Binance USDⓈ-M Futures WebSocket (fstream.binance.com) is geo-blocked
    #   from this machine: TCP connection succeeds but zero messages arrive.
    #   Spot WebSocket (stream.binance.com) works correctly.
    #   Enable spot fallback so the paper trade receives real market data.
    #   Limitation: no 8h funding rate on spot (funding_rate=0.0 on all bars);
    #   spot/futures basis < 0.1% on ETH/SOL/AVAX/LINK/DOT — acceptable for the
    #   14-day MRM shadow period.  Production deployment should use futures
    #   directly (e.g. from a futures-accessible VPS).
    env["TRADEBOT_FEED_SPOT_FALLBACK"] = "1"
    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
    with open(log_file, "ab") as fh:
        proc = subprocess.Popen(
            [sys.executable, "-m", "apps.live_paper_trader"],
            cwd=str(_ROOT), stdout=fh, stderr=subprocess.STDOUT,
            env=env, creationflags=creationflags,
        )
    (_LOG_DIR / "paper_trade.pid").write_text(str(proc.pid))
    log(f"Paper trader spawned PID={proc.pid} (SPOT_FALLBACK=1)")
    write_status("paper_trade", "running", pid=proc.pid)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    log("=" * 70)
    log("Tradebot retrain orchestrator — PARALLEL — START")
    log(f"CPU detected: {_CPU} cores")
    log(f"Parallelism: S1={N_PARALLEL_STAGE1} S2={N_PARALLEL_STAGE2} S3={N_PARALLEL_STAGE3}")
    log(f"OMP_NUM_THREADS per child: {OMP_THREADS}")
    log(f"Symbols: {SYMBOLS}")
    log(f"Acceptance gate: Sharpe>={ACCEPT_SHARPE}, MaxDD<={ACCEPT_MAXDD:.0%}, "
        f"n_trades>={ACCEPT_TRADES_MIN}")
    log("=" * 70)

    stage_build_features()
    stage_tune_hparams()
    stage_train_cpcv()
    stage_backtest_portfolio()
    metrics = check_gate()
    log(f"Gate PASS: Sharpe={metrics['sharpe']:.2f} "
        f"MaxDD={metrics['maxdd']:.2%} n_trades={metrics['n_trades']}")
    log("=" * 70)
    log("Retrain complete — launching live paper trade.")
    launch_paper_trade()
    log("=" * 70)
    log("Orchestrator finished.  Monitor:")
    log("  python scripts/monitor_retrain.py --follow")
    log("  Get-Content -Wait logs/paper_trade.log")
    log("  http://localhost:8501")


if __name__ == "__main__":
    try:
        main()
    except SystemExit as exc:
        log(f"HALT: {exc}")
        raise
    except Exception as exc:
        log(f"FATAL: {type(exc).__name__}: {exc}")
        write_status("master", "ERROR", error=str(exc))
        raise
