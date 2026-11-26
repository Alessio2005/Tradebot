"""De opdrachtregel van de options-engine: `python -I -m tradebot.options <opdracht>`.

* `replay --market DIR --out DIR [--config PAD] [--workers N]`: speel een opgeslagen markt af
  met de paper-broker en schrijf journal, staat en `summary.json` naar `--out`. `--workers`
  bouwt de ketens vooraf parallel; de uitkomst is dezelfde.
* `historical --out DIR --start D --end D [--config PAD]`: bouw een historische markt uit de
  koersen en de IV van IBKR (alleen lezen; dezelfde `IB_*`-omgeving, client-id 18).
* `paper --run-dir DIR [--config PAD] [--healthcheck]`: de paper-dienst tegen IB Gateway.
  Verbinding uit de omgeving: `IB_HOST`, `IB_PORT`, `IB_CLIENT_ID`, `IB_ACCOUNT`. Alleen
  een paper-account (id begint met `DU`). `--healthcheck` controleert alleen de heartbeat.

Er is geen live-opdracht: live staat uit en kan hier niet aangezet worden.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from pathlib import Path

from ib_async import IB

from ..schemas.options import DEFAULT_OPTIONS_CONFIG, OptionsConfig, load_options_config
from .calendar import NY
from .data.historical import HistoricalSeries, HistoricalSpec, build_market, fetch_history
from .data.store import read_market, write_market
from .engine.paper import run_paper
from .engine.replay import run_replay
from .execution.ib_noise import quiet_ib

__all__ = ["HEARTBEAT_MAX_AGE_SECONDS", "healthy", "main"]

HEARTBEAT_MAX_AGE_SECONDS = 600.0


def healthy(run_dir: Path, now: datetime) -> bool:
    beat = run_dir / "heartbeat"
    if not beat.is_file():
        return False
    last = datetime.fromisoformat(beat.read_text(encoding="utf-8").strip())
    return (now - last).total_seconds() <= HEARTBEAT_MAX_AGE_SECONDS


async def _paper(cfg: OptionsConfig, run_dir: Path, env: Mapping[str, str]) -> None:
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)
    await run_paper(cfg, IB(), run_dir, stop, env.get("IB_ACCOUNT", ""),
                    env.get("IB_HOST", "127.0.0.1"), int(env.get("IB_PORT", "4002")),
                    int(env.get("IB_CLIENT_ID", "17")))


async def _history(cfg: OptionsConfig, spec: HistoricalSpec,
                   env: Mapping[str, str]) -> list[HistoricalSeries]:
    quiet_ib()
    ib = IB()
    await ib.connectAsync(env.get("IB_HOST", "127.0.0.1"), int(env.get("IB_PORT", "4002")),
                          int(env.get("IB_CLIENT_ID", "18")), readonly=True)
    try:
        return await fetch_history(ib, cfg.universe.underlyings, spec.start, spec.end)
    finally:
        ib.disconnect()


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="tradebot.options",
                                 description="Options-engine (alleen fictief geld).")
    sub = ap.add_subparsers(dest="command", required=True)
    rp = sub.add_parser("replay", help="speel een opgeslagen markt af met de paper-broker")
    rp.add_argument("--market", type=Path, required=True)
    rp.add_argument("--out", type=Path, required=True)
    rp.add_argument("--config", type=Path, default=DEFAULT_OPTIONS_CONFIG)
    rp.add_argument("--workers", type=int, default=1,
                    help="processen om de ketens vooraf parallel te bouwen (zelfde uitkomst)")
    hp = sub.add_parser("historical",
                        help="bouw een historische markt uit IBKR-koersen en -IV (alleen lezen)")
    hp.add_argument("--out", type=Path, required=True)
    hp.add_argument("--start", type=date.fromisoformat, required=True)
    hp.add_argument("--end", type=date.fromisoformat, required=True)
    hp.add_argument("--config", type=Path, default=DEFAULT_OPTIONS_CONFIG)
    pp = sub.add_parser("paper", help="de paper-dienst tegen IB Gateway (alleen paper)")
    pp.add_argument("--run-dir", type=Path, required=True)
    pp.add_argument("--config", type=Path, default=DEFAULT_OPTIONS_CONFIG)
    pp.add_argument("--healthcheck", action="store_true")
    args = ap.parse_args(argv)
    if args.command == "paper" and args.healthcheck:
        return 0 if healthy(args.run_dir, datetime.now(NY)) else 1
    cfg = load_options_config(args.config)
    if args.command == "paper":
        asyncio.run(_paper(cfg, args.run_dir, os.environ))
        return 0
    if args.command == "historical":
        spec = HistoricalSpec(args.start, args.end, half_days=tuple(cfg.universe.half_days))
        series = asyncio.run(_history(cfg, spec, os.environ))
        write_market(build_market(series, spec), args.out)
        print(f"historische markt {spec.start} t/m {spec.end} geschreven naar {args.out}")
        return 0
    res = asyncio.run(run_replay(cfg, read_market(args.market), args.out,
                                 args.workers))
    print(json.dumps(res.as_record(), indent=1, sort_keys=True))
    return 1 if res.open_short_legs_past_expiry or res.halted else 0
