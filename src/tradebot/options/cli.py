"""De opdrachtregel van de options-engine: `python -I -m tradebot.options <opdracht>`.

* `replay --market DIR --out DIR [--config PAD]`: speel een opgeslagen markt af met de
  paper-broker en schrijf journal, staat en `summary.json` naar `--out`.

Er is geen live-opdracht: live staat uit en kan hier niet aangezet worden.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Sequence
from pathlib import Path

from ..schemas.options import DEFAULT_OPTIONS_CONFIG, load_options_config
from .data.store import read_market
from .engine.replay import run_replay

__all__ = ["main"]


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="tradebot.options",
                                 description="Options-engine (alleen fictief geld).")
    sub = ap.add_subparsers(dest="command", required=True)
    rp = sub.add_parser("replay", help="speel een opgeslagen markt af met de paper-broker")
    rp.add_argument("--market", type=Path, required=True)
    rp.add_argument("--out", type=Path, required=True)
    rp.add_argument("--config", type=Path, default=DEFAULT_OPTIONS_CONFIG)
    args = ap.parse_args(argv)
    cfg = load_options_config(args.config)
    res = asyncio.run(run_replay(cfg, read_market(args.market), args.out))
    print(json.dumps(res.as_record(), indent=1, sort_keys=True))
    return 1 if res.open_short_legs_past_expiry or res.halted else 0
