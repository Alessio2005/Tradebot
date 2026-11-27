"""De opdrachtregel van de options-engine: `python -I -m tradebot.options <opdracht>`.

* `replay --market DIR --out DIR [--config PAD] [--overwrite]`: speel een opgeslagen markt af
  met de paper-broker en schrijf journal, staat en `summary.json` naar `--out`.
* `sweep --markets DIR [DIR ...] --out DIR [--config PAD] [--workers N] [--overwrite]`: dezelfde
  config op meerdere opgeslagen markten (robuustheidscheck, geen parameter-grid); per markt een
  replay in `--out/<mapnaam>/` en de statistiek van alle markten in `--out/sweep.json`.
  `--workers` (standaard auto) is het aantal markten tegelijk.

`replay` en `sweep` weigeren een `--out` die al bestaat en niet leeg is (exit-code 2): het
journal wordt aangevuld, dus een tweede run erin zou elke telling verdubbelen. `--overwrite`
wist alleen de bekende bestanden van een eerdere replay- of sweep-uitvoer (zie `prepare_out`);
elke andere map, een map met HALT/UNWIND/`halt/` en een map met de markt erin blijven staan.
* `dolthub --cache DIR --out DIR --start D --end D [--symbols S ...] [--offline]`: bouw een
  replay-markt uit echte einde-dag-optieketens van DoltHub (openbare API, alleen lezen) met
  koersen en dividenden uit yfinance. `--cache` bewaart de ruwe ketens per dag; `--offline`
  gebruikt alleen wat daar al staat. Beperkingen: zie `data/dolthub.py` en de CHANGELOG.
* `historical --out DIR --start D --end D [--config PAD]`: bouw een historische markt uit de
  koersen en de IV van IBKR (alleen lezen; dezelfde `IB_*`-omgeving, client-id 18).
* `paper --run-dir DIR [--config PAD] [--healthcheck]`: de paper-dienst tegen IB Gateway.
  Verbinding uit de omgeving: `IB_HOST`, `IB_PORT`, `IB_CLIENT_ID`, `IB_ACCOUNT`. Alleen
  een paper-account (id begint met `DU`). `--healthcheck` controleert alleen de heartbeat.
  Alarmen: `ALERT_WEBHOOK_URL` en/of `TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHAT_ID` (altijd ook het
  log); `HEARTBEAT_URL` is de dead man's switch; `IB_RECONNECT_TIMEOUT_SECONDS` (900) is de
  tijd zonder verbinding met de gateway waarna HALT volgt.

Er is geen live-opdracht: live staat uit en kan hier niet aangezet worden.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import date, datetime
from pathlib import Path

import requests
from ib_async import IB

from ..carry.alerts import Alerter
from ..schemas.options import DEFAULT_OPTIONS_CONFIG, OptionsConfig, load_options_config
from .calendar import NY
from .data.dolthub import DEFAULT_RATE, DEFAULT_SYMBOLS, DoltSpec, make_market
from .data.historical import HistoricalSeries, HistoricalSpec, build_market, fetch_history
from .data.store import read_market, write_market
from .engine.paper import run_paper
from .engine.replay import REPLAY_MODE, run_replay
from .engine.sweep import run_sweep
from .execution.ib_noise import quiet_ib

__all__ = ["HEARTBEAT_MAX_AGE_SECONDS", "healthy", "main", "prepare_out"]

HEARTBEAT_MAX_AGE_SECONDS = 600.0
PULSE_SECONDS = 60.0
log = logging.getLogger(__name__)


def healthy(run_dir: Path, now: datetime) -> bool:
    beat = run_dir / "heartbeat"
    if not beat.is_file():
        return False
    last = datetime.fromisoformat(beat.read_text(encoding="utf-8").strip())
    return (now - last).total_seconds() <= HEARTBEAT_MAX_AGE_SECONDS


#: Wat een replay in zijn uitvoermap schrijft (zonder HALT: een gehaltte replay heeft `halt/`).
REPLAY_OUTPUT = frozenset({"journal.jsonl", "state.json", "summary.json", "heartbeat"})
#: Kill switches van de engine: een map met een van deze wordt nooit gewist.
KILL_SWITCHES = ("HALT", "UNWIND", "halt")


def _replay_files(d: Path) -> list[Path] | None:
    """De bestanden van replay-uitvoer in `d`, of None als `d` dat niet precies is: alleen
    bekende bestanden, en een `state.json` met mode `replay` (een paper-run heeft `paper`)."""
    entries = sorted(d.iterdir())
    if not entries or any(e.name not in REPLAY_OUTPUT or not e.is_file() for e in entries):
        return None
    try:
        mode = json.loads((d / "state.json").read_text(encoding="utf-8")).get("mode")
    except (OSError, ValueError, AttributeError):    # geen, onleesbare of vreemde staat
        return None
    return entries if mode == REPLAY_MODE else None


def _output_files(out: Path) -> list[Path] | None:
    """Alles wat `--overwrite` in `out` mag wissen (submappen na hun bestanden), of None.

    Een replay-map (zie `_replay_files`), of een sweep-map: `sweep.json` plus submappen die elk
    een replay-map zijn, en verder niets."""
    replay = _replay_files(out)
    if replay is not None:
        return replay
    entries = sorted(out.iterdir())
    if not (out / "sweep.json").is_file():
        return None
    files: list[Path] = [out / "sweep.json"]
    for e in entries:
        if e.name == "sweep.json":
            continue
        sub = _replay_files(e) if e.is_dir() else None
        if sub is None:
            return None
        files += [*sub, e]
    return files


def prepare_out(out: Path, overwrite: bool, markets: Sequence[Path] = ()) -> str | None:
    """Maak `--out` klaar; geeft de fout als dat niet mag, anders None.

    Een niet-bestaande of lege map is goed. Een niet-lege map alleen met `overwrite`, en dan
    alleen als hij precies eerdere replay- of sweep-uitvoer is (`_output_files`), zonder kill
    switch (`HALT`, `UNWIND`, `halt/`) erin en zonder een van de `markets` erin. Dan gaan
    alleen die bekende bestanden en submappen weg, nooit een hele boom."""
    if not out.exists() or (out.is_dir() and not any(out.iterdir())):
        return None
    if not overwrite:
        return (f"--out {out} bestaat al en is niet leeg (het journal zou worden aangevuld);"
                " kies een lege map of geef --overwrite")
    if not out.is_dir():
        return f"--overwrite: {out} is geen map; niets gewist"
    root = out.resolve()
    if any(m.resolve() == root or root in m.resolve().parents for m in markets):
        return f"--overwrite: {out} bevat de markt; niets gewist"
    if any((d / k).exists() for d in (out, *(e for e in out.iterdir() if e.is_dir()))
           for k in KILL_SWITCHES):
        return f"--overwrite: {out} bevat HALT, UNWIND of halt/; niets gewist"
    files = _output_files(out)
    if files is None:
        return f"--overwrite: {out} is niet precies een eerdere replay- of sweep-uitvoer; niets gewist"
    for f in files:
        if f.is_dir():
            f.rmdir()                     # leeg: zijn bestanden staan ervoor in de lijst
        else:
            f.unlink()
    return None


def _ping(url: str) -> None:
    requests.get(url, timeout=10).raise_for_status()


def make_pulse(url: str | None, ping: Callable[[str], None] = _ping,
               clock: Callable[[], float] = time.monotonic,
               every: float = PULSE_SECONDS) -> Callable[[], None] | None:
    """De dead man's switch: een GET op `HEARTBEAT_URL`, hoogstens eens per `every` seconden.

    Een falende ping stopt de dienst niet (alleen een log). Zonder URL: geen pulse."""
    if not url:
        return None
    last: list[float] = []

    def pulse() -> None:
        now = clock()
        if last and now - last[0] < every:
            return
        last[:] = [now]
        try:
            ping(url)
        except Exception as e:      # een dead man's switch die faalt, stopt de dienst niet
            log.warning("heartbeat-ping faalde: %s", type(e).__name__)

    return pulse


async def _paper(cfg: OptionsConfig, run_dir: Path, env: Mapping[str, str]) -> None:
    # Alarmen (HALT, recon-break, assignment): webhook en/of Telegram uit de omgeving, altijd
    # ook in het log; zonder kanaal blijft alleen het log over (`Alerter.send` gooit nooit).
    alerter = Alerter.from_env(env, log=log.warning)
    if not alerter.channels:
        log.warning("geen alarmkanaal ingesteld (ALERT_WEBHOOK_URL of TELEGRAM_*): alarmen "
                    "gaan alleen naar het log")
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)
    await run_paper(cfg, IB(), run_dir, stop, env.get("IB_ACCOUNT", ""),
                    env.get("IB_HOST", "127.0.0.1"), int(env.get("IB_PORT", "4002")),
                    int(env.get("IB_CLIENT_ID", "17")), alert=alerter.send,
                    reconnect_timeout=float(env.get("IB_RECONNECT_TIMEOUT_SECONDS", "900")),
                    pulse=make_pulse(env.get("HEARTBEAT_URL")))


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
    dp = sub.add_parser("dolthub", help="bouw een replay-markt uit DoltHub-optieketens")
    dp.add_argument("--cache", type=Path, required=True)
    dp.add_argument("--out", type=Path, required=True)
    dp.add_argument("--start", type=date.fromisoformat, required=True)
    dp.add_argument("--end", type=date.fromisoformat, required=True)
    dp.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS))
    dp.add_argument("--rate", type=float, default=DEFAULT_RATE)
    dp.add_argument("--offline", action="store_true", help="niets ophalen, alleen de cache")
    dp.add_argument("--config", type=Path, default=DEFAULT_OPTIONS_CONFIG)
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
    if args.command == "dolthub":
        dspec = DoltSpec(args.start, args.end, rate=args.rate,
                         half_days=tuple(cfg.universe.half_days))
        market = make_market(args.cache, dspec, tuple(args.symbols), fetch=not args.offline)
        write_market(market, args.out)
        print(f"DoltHub-markt {dspec.start} t/m {dspec.end} ({', '.join(sorted(market.bars))},"
              f" {len(market.days)} handelsdagen) geschreven naar {args.out}")
        return 0
    error = prepare_out(args.out, args.overwrite,
                        args.markets if args.command == "sweep" else [args.market])
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
