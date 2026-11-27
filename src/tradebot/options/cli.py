"""De opdrachtregel van de options-engine: `python -I -m tradebot.options <opdracht>`.

* `replay --market DIR --out DIR [--config PAD] [--overwrite]`: speel een opgeslagen markt af
  met de paper-broker en schrijf journal, staat en `summary.json` naar `--out`.
* `sweep --markets DIR [DIR ...] --out DIR [--config PAD] [--workers N] [--overwrite]`: dezelfde
  config op meerdere opgeslagen markten (robuustheidscheck, geen parameter-grid); per markt een
  replay in `--out/<mapnaam>/` en de statistiek van alle markten in `--out/sweep.json`.
  `--workers` (standaard auto) is het aantal markten tegelijk.

`replay` en `sweep` weigeren een `--out` die al bestaat en niet leeg is (exit-code 2): het
journal wordt aangevuld, dus een tweede run erin zou elke telling verdubbelen. `--overwrite`
maakt hem eerst leeg, maar alleen als hij eruitziet als eerdere uitvoer (een `journal.jsonl` of
`sweep.json` erin, of een `journal.jsonl` in een submap); een andere map blijft staan.
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
import shutil
import signal
import sys
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
from .engine.sweep import run_sweep
from .execution.ib_noise import quiet_ib

__all__ = ["HEARTBEAT_MAX_AGE_SECONDS", "healthy", "main", "prepare_out"]

HEARTBEAT_MAX_AGE_SECONDS = 600.0


def healthy(run_dir: Path, now: datetime) -> bool:
    beat = run_dir / "heartbeat"
    if not beat.is_file():
        return False
    last = datetime.fromisoformat(beat.read_text(encoding="utf-8").strip())
    return (now - last).total_seconds() <= HEARTBEAT_MAX_AGE_SECONDS


def _looks_like_output(out: Path) -> bool:
    """Een eerdere replay (`journal.jsonl`) of sweep (`sweep.json`, of journals per markt)."""
    return ((out / "journal.jsonl").is_file() or (out / "sweep.json").is_file()
            or any((d / "journal.jsonl").is_file() for d in out.iterdir() if d.is_dir()))


def prepare_out(out: Path, overwrite: bool) -> str | None:
    """Maak `--out` klaar; geeft de fout als dat niet mag, anders None.

    Een niet-bestaande of lege map is goed. Een niet-lege map alleen met `overwrite`, en dan
    alleen als hij eruitziet als eerdere uitvoer: dan gaat hij eerst helemaal leeg."""
    if not out.exists() or (out.is_dir() and not any(out.iterdir())):
        return None
    if not overwrite:
        return (f"--out {out} bestaat al en is niet leeg (het journal zou worden aangevuld);"
                " kies een lege map of geef --overwrite")
    if not out.is_dir() or not _looks_like_output(out):
        return f"--overwrite: {out} is geen eerdere replay- of sweep-uitvoer; niets gewist"
    shutil.rmtree(out)
    return None


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
    sp = sub.add_parser("sweep", help="dezelfde config op meerdere opgeslagen markten")
    sp.add_argument("--markets", type=Path, nargs="+", required=True)
    sp.add_argument("--out", type=Path, required=True)
    sp.add_argument("--config", type=Path, default=DEFAULT_OPTIONS_CONFIG)
    sp.add_argument("--workers", type=int, default=None)
    for p in (rp, sp):
        p.add_argument("--overwrite", action="store_true",
                       help="maak een bestaande uitvoermap eerst leeg")
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
    error = prepare_out(args.out, args.overwrite)
    if error is not None:
        print(error, file=sys.stderr)
        return 2
    if args.command == "sweep":
        record = run_sweep(cfg, args.markets, args.out, args.workers)
        print(json.dumps(record, indent=1, sort_keys=True))
        return 1 if any(m["open_short_legs_past_expiry"] or m["halted"]
                        for m in record["markets"].values()) else 0
    res = asyncio.run(run_replay(cfg, read_market(args.market), args.out))
    print(json.dumps(res.as_record(), indent=1, sort_keys=True))
    return 1 if res.open_short_legs_past_expiry or res.halted else 0
