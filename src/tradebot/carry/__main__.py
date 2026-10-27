"""CLI van het carry-shadowboek (paper: het model is de venue).

    python -I -m tradebot.carry init   --state S [--panels P] [--start 2026-10-01]
    python -I -m tradebot.carry step   --state S [--panels P] [--log L]
    python -I -m tradebot.carry status --state S
    python -I -m tradebot.carry unwind --state S --reason TEKST [--panels P]

`--panels` wijst naar een paneelwortel zoals `binance_vision forward` hem bouwt
(panels/, spot_panels/, mark_panels/); standaard `data/binance_vision/forward`. `init`
zet de beginstand (de backtest tot de start), `step` verwerkt elke nieuwe gesettelde bar
en schrijft per bar één beslisregel naar het log (JSONL, standaard naast de staat).

Dit is het shadowboek van v8-T2 op Binance-panelen (de vooruit-lezing van v8). Het boek dat
op Kraken EU kan draaien, live of dry-run, heeft een eigen CLI: `python -I -m
tradebot.carry.u1` (zie `carry/u1.py`).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from ..systematic import programme_v6 as v6
from .runner import init, load_context, step, unwind
from .state import load_state, save_state
from .venue import PaperVenue

__all__ = ["main"]


def _paper(state_path: Path) -> PaperVenue:
    st = load_state(state_path)
    return PaperVenue(spot=dict(st.expected_spot), perp_short=dict(st.expected_perp),
                      equity_usd=st.equity)


def main(argv: Sequence[str] | None = None, *, root: Path = v6.ROOT) -> dict:
    ap = argparse.ArgumentParser(prog="python -I -m tradebot.carry")
    ap.add_argument("command", choices=("init", "step", "status", "unwind"))
    ap.add_argument("--state", type=Path, required=True)
    ap.add_argument("--panels", type=Path, default=None)
    ap.add_argument("--start", default=None)
    ap.add_argument("--log", type=Path, default=None)
    ap.add_argument("--reason", default=None)
    args = ap.parse_args(argv)
    panels = args.panels or root / v6.FORWARD_ROOT
    if args.command == "status":
        st = load_state(args.state)
        out = {"last_bar": st.last_bar, "equity": st.equity, "mode": st.mode,
               "halted": st.halted, "n_held": len(st.a), "moving": sorted(st.goal),
               "n_liquidations": st.n_liquidations}
    elif args.command == "init":
        ctx = load_context(root, panels)
        st = init(ctx, start=args.start or ctx.cfg.forward.start)
        save_state(st, args.state)
        out = {"initialised_through": st.last_bar, "equity": st.equity, "n_held": len(st.a)}
    elif args.command == "step":
        ctx = load_context(root, panels)
        records, st = step(ctx, load_state(args.state), _paper(args.state))
        log = args.log or args.state.with_suffix(".jsonl")
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec, sort_keys=True) + "\n")
        save_state(st, args.state)
        out = {"bars": len(records), "last_bar": st.last_bar, "equity": st.equity}
    else:
        if not args.reason:
            ap.error("unwind vraagt een --reason")
        ctx = load_context(root, panels)
        px = dict(zip(ctx.arrays.symbols, ctx.spot_px[-1], strict=True))
        pp = dict(zip(ctx.arrays.symbols, ctx.perp_px[-1], strict=True))
        fills, st = unwind(load_state(args.state), _paper(args.state), spot_price=px,
                           perp_price=pp, reason=args.reason)
        save_state(st, args.state)
        out = {"closed_legs": len(fills), "halted": st.halted}
    print(json.dumps(out, sort_keys=True))
    return out


if __name__ == "__main__":
    main(sys.argv[1:])
