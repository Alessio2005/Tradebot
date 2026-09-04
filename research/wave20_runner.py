# research/wave20_runner.py
"""One-shot Wave-20→22 evidence runner (0.4 rebaseline + smoke-fetches +
W21 universe + W22 unit evals).

Run from the repo root with the project venv active:

    python research/wave20_runner.py            # everything
    python research/wave20_runner.py --quick    # skip the full universe fetch
                                               # (25-symbol smoke build instead)

Every step's stdout/stderr is captured to artefacts/wave20_runlog/<step>.log
and a machine-readable status summary lands in
artefacts/wave20_runlog/status.json. Steps are independent unless marked;
a failure never silently skips evidence — it is recorded and the run
continues with the steps that don't depend on it.

This script only COLLECTS evidence. Accept/archive decisions and ledger
merges remain serial, deliberate actions (STAPPENPLAN §0).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV = {**os.environ, "PYTHONPATH": str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")}
LOGDIR = ROOT / "artefacts" / "wave20_runlog"
PY = sys.executable

SMOKE_SNIPPET = r"""
import sys
sys.path.insert(0, "src")
import pandas as pd
results = {}

def try_step(name, fn):
    try:
        out = fn()
        results[name] = f"OK ({out})"
    except Exception as e:
        results[name] = f"FAIL: {type(e).__name__}: {e}"

from tradebot.data.sources import stooq, kenfrench, fred, cboe, edgar, wiki_constituents

try_step("stooq", lambda: f"{len(stooq.fetch_daily('aapl.us'))} rows")
try_step("kenfrench", lambda: f"{len(kenfrench.fetch_factors_daily())} rows")
try_step("fred", lambda: f"{len(fred.fetch_series('DTB3', pd.Timedelta(days=1)))} rows")
try_step("cboe", lambda: f"{len(cboe.fetch_vix_term_structure())} rows")
try_step("edgar", lambda: f"{len(edgar.fetch_filing_index(320193))} filings")
try_step("wiki", lambda: f"{len(wiki_constituents.fetch_membership_events())} events")

for k, v in results.items():
    print(f"{k:10s} {v}")
# stooq is known-blocked upstream (2026-06-10) and has a registered
# yfinance fallback — log it, but only non-stooq failures fail the step.
n_fail = sum(v.startswith("FAIL") for k, v in results.items() if k != "stooq")
sys.exit(1 if n_fail else 0)
"""


def run_step(name: str, cmd: list[str], status: dict, cwd: Path = ROOT) -> bool:
    print(f"\n=== {name} ===", flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, env=ENV)
    log = LOGDIR / f"{name}.log"
    log.write_text(
        f"$ {' '.join(cmd)}\n\n--- stdout ---\n{proc.stdout}\n"
        f"--- stderr ---\n{proc.stderr}\n--- rc={proc.returncode} ---\n",
        encoding="utf-8",
    )
    ok = proc.returncode == 0
    status[name] = {
        "ok": ok,
        "rc": proc.returncode,
        "seconds": round(time.time() - t0, 1),
        "log": str(log.relative_to(ROOT)),
    }
    print(proc.stdout[-2000:] if proc.stdout else "(no stdout)")
    if not ok:
        print(f"!! {name} FAILED (rc={proc.returncode}) — see {log}", flush=True)
    return ok


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quick", action="store_true", help="25-symbol universe smoke build")
    p.add_argument("--skip-tests", action="store_true")
    p.add_argument("--skip-rebaseline", action="store_true",
                   help="skip 0.4 (already confirmed bit-identical)")
    p.add_argument("--skip-universe", action="store_true",
                   help="reuse the existing W21 panel parquet (no re-fetch)")
    p.add_argument("--phase2", action="store_true",
                   help="ONLY the phase-2 network ingests: FX (FRED), "
                        "factors (Ken French), EDGAR fundamentals+filings")
    args = p.parse_args(argv)

    LOGDIR.mkdir(parents=True, exist_ok=True)
    status: dict = {}

    if args.phase2:
        run_step("ingest_fx", [PY, "apps/ingest_fx.py"], status)
        run_step("fetch_factors", [PY, "apps/fetch_factors.py"], status)
        run_step("ingest_edgar", [PY, "apps/ingest_edgar.py"], status)
        (LOGDIR / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
        n_fail = sum(1 for v in status.values() if not v.get("ok", False))
        print(f"\n==== PHASE2 DONE: {len(status) - n_fail}/{len(status)} ok ====")
        return 1 if n_fail else 0

    # 0.4a — full test suite (incl. new 0.1/0.2/0.3 + eq-unit guards)
    if not args.skip_tests:
        run_step("pytest", [PY, "-m", "pytest", "tests/", "-q", "--tb=short"], status)

    # 0.4b — crypto rebaseline (twice: bit-identical reproduction check, R-5/G7)
    if not args.skip_rebaseline:
        rb = [PY, "research/wave_final_eval.py",
              "artefacts/broad_perp_daily_close_WIDE.parquet", "10", "0.35"]
        run_step("rebaseline_run1", rb, status)
        run_step("rebaseline_run2", rb, status)
        if status.get("rebaseline_run1", {}).get("ok") and status.get("rebaseline_run2", {}).get("ok"):
            a = (LOGDIR / "rebaseline_run1.log").read_text(encoding="utf-8")
            b = (LOGDIR / "rebaseline_run2.log").read_text(encoding="utf-8")
            ident = a.split("--- stdout ---")[1] == b.split("--- stdout ---")[1]
            status["rebaseline_bit_identical"] = {"ok": ident}
            print(f"\nrebaseline bit-identical: {ident}")

        run_step("multi_sleeve_combine", [PY, "research/multi_sleeve_combine.py"], status)

    # 0.2-rest — smoke-fetch every PIT source (network)
    run_step("smoke_fetches", [PY, "-c", SMOKE_SNIPPET], status)

    # W21 — universe build (full unless --quick). Stooq fail-fasts and the
    # build auto-falls-back to yfinance — make sure it's installed.
    uni_ok = True
    if not args.skip_universe:
        run_step("pip_yfinance", [PY, "-m", "pip", "install", "--quiet", "yfinance"], status)
        build = [PY, "apps/build_equity_universe.py", "build"]
        if args.quick:
            build += ["--max-symbols", "25"]
        uni_ok = run_step("w21_universe_build", build, status)
        if uni_ok:
            run_step("w21_universe_report", [PY, "apps/build_equity_universe.py", "report"], status)
    if uni_ok:

        # W22 — unit evals + ledger staging (depends on W21). Rotate any
        # stale staging file first: a rerun would otherwise append duplicate
        # keys and the serial merge correctly refuses those.
        stg = ROOT / "artefacts/governance/hypothesis_ledger_staging_w22.json"
        if stg.exists():
            stg.rename(stg.with_suffix(f".{int(time.time())}.bak"))
        # G4 factorset (FF5+MOM) — fetch once; w22 uses it when present
        run_step("fetch_factors", [PY, "apps/fetch_factors.py"], status)
        w22 = [PY, "apps/run_eq_units.py", "--stage",
               "artefacts/governance/hypothesis_ledger_staging_w22.json"]
        if (ROOT / "market_data_parquet/equities/factors_daily.parquet").exists():
            w22 += ["--factors", "market_data_parquet/equities/factors_daily.parquet"]
        run_step("w22_eq_units", w22, status)

    (LOGDIR / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    n_fail = sum(1 for v in status.values() if not v.get("ok", False))
    print(f"\n==== DONE: {len(status) - n_fail}/{len(status)} steps ok — "
          f"logs in {LOGDIR.relative_to(ROOT)} ====")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
