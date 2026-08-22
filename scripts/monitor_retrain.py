"""scripts/monitor_retrain.py — Live progress dashboard for the retrain run.

Run alongside scripts/retrain_and_start_paper.py to see per-stage progress.
Reads logs/retrain_status.json + log files and prints a compact summary.

Usage:
    python scripts/monitor_retrain.py            # one-shot snapshot
    python scripts/monitor_retrain.py --follow   # refresh every 30s
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_LOG_DIR = _ROOT / "logs"


def _tail(path: Path, n: int = 3) -> list[str]:
    if not path.exists():
        return []
    try:
        with open(path, "rb") as fh:
            # Read last ~16KB
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - 16_384))
            data = fh.read().decode("utf-8", errors="replace")
        lines = [ln for ln in data.splitlines() if ln.strip()]
        return lines[-n:]
    except Exception as exc:
        return [f"<read error: {exc}>"]


def snapshot() -> None:
    status_path = _LOG_DIR / "retrain_status.json"
    print("=" * 70)
    print(f"Tradebot retrain monitor — {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    if status_path.exists():
        status = json.loads(status_path.read_text())
        print(f"Stage:  {status.get('stage'):20s} Status: {status.get('status')}")
        for k, v in status.items():
            if k in ("stage", "status", "ts"):
                continue
            print(f"  {k}: {v}")
    else:
        print("No status file yet — retrain may not have started.")

    print()
    print("Recent stage logs (last 3 lines each):")
    # List all retrain_* log files, sort by mtime descending, show top 5
    logs = sorted(
        _LOG_DIR.glob("retrain_*.log"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )[:5]
    for log in logs:
        mtime = time.strftime("%H:%M:%S", time.localtime(log.stat().st_mtime))
        print(f"\n--- {log.name} (last write {mtime}) ---")
        for line in _tail(log, 3):
            print(f"  {line[-160:]}")

    # If paper trade has been launched, also show that
    pt_log = _LOG_DIR / "paper_trade.log"
    if pt_log.exists():
        print("\n--- paper_trade.log (last 5 lines) ---")
        for line in _tail(pt_log, 5):
            print(f"  {line[-160:]}")

    state_json = _ROOT / "artefacts" / "paper_trade" / "state.json"
    if state_json.exists():
        try:
            state = json.loads(state_json.read_text())
            print("\n--- paper_trade state.json ---")
            print(f"  mode={state.get('mode')} "
                  f"bars={state.get('bars_processed')} "
                  f"trades={state.get('n_trades')} "
                  f"equity=${state.get('equity'):.0f} "
                  f"DD={state.get('drawdown_pct'):.2f}%  "
                  f"Sharpe(rolling)={state.get('rolling_sharpe_32d')}")
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--follow", action="store_true", help="refresh every 30s")
    args = parser.parse_args()
    if not args.follow:
        snapshot()
        return
    while True:
        try:
            os.system("cls" if os.name == "nt" else "clear")
        except Exception:
            pass
        snapshot()
        try:
            time.sleep(30)
        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    main()
