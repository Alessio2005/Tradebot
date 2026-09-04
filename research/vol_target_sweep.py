"""vol_target_sweep.py — find the most aggressive risk setting that drives
total return toward the ~600% target (DD may exceed 15%; return is priority).

Context (2026-05-30): Phase A/B (neg-EV sides dropped, full-200 retrain) hit
Sharpe ~2.7-2.9 / MaxDD 6-8% but total return fell to 70-109% (down from
~600%+ on older, more aggressive runs) because the book runs very light
(realized_vol 7-9%, long_bear_factor 0.25, SHORT signals gated out).

It climbs aggression along three levers and detects the vol_mult=2.5 plateau
(where raising target_annual_vol stops lifting realized vol):
  - vol  = portfolio.target_annual_vol
  - bear = portfolio.long_bear_factor (1.0 = no bear-long cut -> more return)
  - sides= "A" keeps the Phase-A disabled_sides; "ALL" re-enables every side
           (more simultaneous positions -> higher achievable realized vol).

PARALLEL EXECUTION: combos run concurrently (MAX_CONCURRENT subprocesses). Each
run gets a unique machine.run_tag so its outputs (tracks{tag}/, portfolio{tag}/,
reports/portfolio_metrics{tag}.json) never clobber another run, plus a unique
hydra.run.dir. BLAS/OMP threads are capped per process so MAX_CONCURRENT x
THREADS_PER_PROC == physical cores (no oversubscription on the 6-core box).

Run AFTER the Phase B retrain completes so it operates on the final models.
Usage:  python research/vol_target_sweep.py
Output: reports/vol_target_sweep.csv  (+ console table)
"""
from __future__ import annotations

import concurrent.futures
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable
REPORTS = ROOT / "reports"
OUT_CSV = REPORTS / "vol_target_sweep.csv"

RETURN_TARGET = 6.0  # 600%

# 6 physical cores: 3 concurrent backtests x 2 BLAS threads each = full
# utilisation without oversubscription. Backtests are largely single-process
# Python loops, so concurrency is the dominant speedup.
MAX_CONCURRENT = 3
THREADS_PER_PROC = 2

# (target_annual_vol, long_bear_factor, sides). Ordered cheap->aggressive.
# (vol 0.24 vs 0.35 at fixed bear/sides reveals whether the 2.5 cap is pinning
# realized vol.)
COMBOS: list[tuple[float, float, str]] = [
    (0.16, 0.25, "A"),
    (0.24, 0.25, "A"),
    (0.24, 1.00, "A"),
    (0.35, 1.00, "A"),
    (0.24, 1.00, "ALL"),
    (0.35, 1.00, "ALL"),
]


def _tag(vol: float, bear: float, sides: str) -> str:
    return f"_v{int(round(vol*100)):03d}_b{int(round(bear*100)):03d}_{sides}"


def run_backtest(vol: float, bear: float, sides: str) -> dict | None:
    tag = _tag(vol, bear, sides)
    cmd = [
        PYTHON, "-m", "apps.backtest_portfolio",
        f"portfolio.target_annual_vol={vol}",
        f"portfolio.long_bear_factor={bear}",
        f"+machine.run_tag={tag}",
        f"hydra.run.dir=outputs/sweep{tag}",
    ]
    if sides == "ALL":
        cmd.append("portfolio.disabled_sides=[]")

    # Cap per-process math threads so MAX_CONCURRENT runs share the cores evenly.
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[var] = str(THREADS_PER_PROC)

    log_path = ROOT / f"sweep{tag}.log"
    print(f"START  vol={vol} bear={bear} sides={sides}  (tag {tag})", flush=True)
    with open(log_path, "w", encoding="utf-8") as lf:
        r = subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=lf, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        print(f"FAILED vol={vol} bear={bear} sides={sides} (exit {r.returncode}, see {log_path.name})", flush=True)
        return None

    metrics_path = REPORTS / f"portfolio_metrics{tag}.json"
    if not metrics_path.exists():
        print(f"FAILED vol={vol} bear={bear} sides={sides} (no metrics written)", flush=True)
        return None
    m = json.loads(metrics_path.read_text())
    row = {
        "target_vol": vol, "bear_factor": bear, "sides": sides,
        "sharpe": round(m.get("sharpe", float("nan")), 3),
        "deflated_sharpe": round(m.get("deflated_sharpe", float("nan")), 3),
        "max_drawdown": round(m.get("max_drawdown", float("nan")), 4),
        "total_return": round(m.get("total_return", float("nan")), 3),
        "realized_vol": round(m.get("realized_vol", float("nan")), 4),
        "calmar": round(m.get("calmar", float("nan")), 3),
    }
    print(
        f"DONE   vol={vol} bear={bear} sides={sides} -> "
        f"Sharpe={row['sharpe']} MaxDD={row['max_drawdown']*100:.2f}% "
        f"Return={row['total_return']*100:.0f}% realVol={row['realized_vol']*100:.1f}%",
        flush=True,
    )
    return row


def main() -> None:
    rows: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as ex:
        futures = [ex.submit(run_backtest, v, b, s) for v, b, s in COMBOS]
        for fut in concurrent.futures.as_completed(futures):
            r = fut.result()
            if r is not None:
                rows.append(r)

    if not rows:
        print("No successful runs.")
        return

    rows.sort(key=lambda r: r["total_return"])
    cols = list(rows[0].keys())
    OUT_CSV.write_text("\n".join([",".join(cols)] + [",".join(str(r[c]) for c in cols) for r in rows]) + "\n")

    print("\n================ SWEEP SUMMARY (return-priority) ================")
    for r in rows:
        print(
            f"vol={r['target_vol']:.2f} bear={r['bear_factor']:.2f} sides={r['sides']:3s} | "
            f"Sharpe={r['sharpe']:.2f} | MaxDD={r['max_drawdown']*100:5.2f}% | "
            f"realVol={r['realized_vol']*100:4.1f}% | Return={r['total_return']*100:5.0f}%"
        )
    hit = [r for r in rows if r["total_return"] >= RETURN_TARGET]
    best_ret = max(rows, key=lambda r: r["total_return"])
    if hit:
        pick = min(hit, key=lambda r: r["max_drawdown"])
        print(
            f"\n>=600% reached. Lowest-DD config at target: vol={pick['target_vol']} "
            f"bear={pick['bear_factor']} sides={pick['sides']} -> "
            f"Return={pick['total_return']*100:.0f}% Sharpe={pick['sharpe']:.2f} "
            f"MaxDD={pick['max_drawdown']*100:.2f}%"
        )
    else:
        print(
            f"\n600% NOT reached by these levers. Max return: vol={best_ret['target_vol']} "
            f"bear={best_ret['bear_factor']} sides={best_ret['sides']} -> "
            f"Return={best_ret['total_return']*100:.0f}% MaxDD={best_ret['max_drawdown']*100:.2f}% "
            f"(realVol={best_ret['realized_vol']*100:.1f}%). If realVol plateaus ~26%, the "
            f"vol_mult=2.5 cap is binding — raising it is the next lever."
        )
    print(f"\nCSV: {OUT_CSV}")


if __name__ == "__main__":
    main()
