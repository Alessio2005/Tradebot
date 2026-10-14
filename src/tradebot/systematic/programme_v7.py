"""Robuust boek v7: het hefboomboek van v6, met gespreide uitvoering en een gefinancierde tranche.

    python -I -m tradebot.systematic.programme_v7 freeze      # ledger + prereg (M = 26)
    python -I -m tradebot.systematic.programme_v7 run         # W_DEV, batterij, selectie
    python -I -m tradebot.systematic.programme_v7 oos         # backcast 2020 + holdout, één keer
    python -I -m tradebot.systematic.programme_v7 forward     # vanaf 2027-04: het vooruit-sample

WAAROM v7 BESTAAT. v6 (W_DEV) liet twee lekken zien, en geen gebrek aan edge:
1. Impact. De helft van de omzet is in- en uitstap van een heel slot (0,1-0,2 van de
   equity in één altcoin) in één keer. In het wortelmodel kost een trade q^1,5. Posities
   staan maanden (mediaan 113 dagen); een paar dagen spreiden kost bijna geen carry.
2. Financiering. 17 % van de notional zat in munten met carry onder de leenrente
   max(8 %, BTC-carry). Die verdienden ~16 %, de lening kostte gemiddeld 30-42 %.

v7 verandert precies die twee dingen, met parameters die vooraf vastliggen (elke tranche
in 5 dagstappen, κ = 0,2 zoals v3; de tranche aan bij carry ≥ rente + 10 pp, de hysterese
van v5). De orkestratie (bevriezen, meten, lezen) is die van v6 (`programme_v6.Programme`).
Ontwerp: `docs/superpowers/specs/2026-10-09-robust-book-v7-execution-design.md`.
"""
from __future__ import annotations

import json
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from ..schemas.robust_book_v2 import RobustBookV7Config, robust_book_v7_config
from ..utils.failfast import DataContractError, require
from . import programme_v5 as v5
from . import programme_v6 as v6
from .book import BookResult
from .harvest import BasisCosts
from .leverage import MarginMarket, levered_targets, run_levered, tranche_targets

__all__ = ["CANDIDATES", "V7", "simulate", "simulate_book"]

CONFIG = Path("conf/model/robust_book_v7.yaml")
SPEC_PATH = Path("conf/research/preregistration_robust_book_v7.yaml")
ARTEFACT_DIR = Path("artefacts/research/robust_book_v7")
CANDIDATES = ("S1_CARRY_PM_SLICED", "S2_CARRY_PM_TRANCHE")
HYPOTHESES = {
    "S1_CARRY_PM_SLICED": "K1 van v6 (de slotregel van v5-H1, 1x per been, unified margin), met "
                          "elke tranche in vijf gelijke dagstappen in en uit.",
    "S2_CARRY_PM_TRANCHE": "S1 plus een tweede, gefinancierde tranche per munt (tot 2x per "
                           "been), aan zolang de carry de leenrente met 10 pp verslaat "
                           "(hysterese tot de rente), ook gespreid.",
}


def simulate(name: str, m: MarginMarket, cfg: RobustBookV7Config, costs: BasisCosts, *,
             lag: int | None = None, ov: Mapping[str, Any] | None = None) -> BookResult:
    """Eén v7-kandidaat. Overrides als in v6, plus slice_days en lever_spread."""
    require(name in CANDIDATES, "Onbekende v7-kandidaat.", DataContractError, name=name)
    return simulate_book(name, m, cfg, costs, lag=lag, ov=ov)


def simulate_book(name: str, m: MarginMarket, cfg: RobustBookV7Config, costs: BasisCosts, *,
                  lag: int | None = None, ov: Mapping[str, Any] | None = None,
                  max_sigma: float | None = None) -> BookResult:
    """De simulatiekern van v7, gedeeld met v8 (`max_sigma`: de sprongrisicogrens; een
    override `max_sigma` in `ov` wint, `None` daar zet hem uit)."""
    o = dict(ov or {})
    max_sigma = o["max_sigma"] if "max_sigma" in o else max_sigma
    lg = int(cfg.execution.lag_bars if lag is None else lag)
    b, lv, x = cfg.basis, cfg.leverage, cfg.v7
    slots = int(o.get("slots", b.slots))
    rule = {"slots": slots, "span": int(o.get("funding_span", b.funding_span)),
            "enter_apr": float(o.get("enter_apr", b.enter_apr)),
            "exit_apr": float(o.get("exit_apr", b.exit_apr)),
            "min_spot_adv_usd": b.min_spot_adv_usd, "max_abs_basis": b.max_abs_basis,
            "carry_noise": v5._noise(m.basis, int(o["noise_seed"])) if "noise_seed" in o else None,
            "max_sigma": None if max_sigma is None else float(max_sigma)}
    rate = v6.financing(m.basis, cfg, str(o.get("financing", "base")))
    if name in x.tranche:
        tgt = tranche_targets(m.basis, financing=rate,
                              lever_spread=float(o.get("lever_spread", x.lever_spread_apr)),
                              **rule)
    else:
        tgt = levered_targets(m.basis, leverage=float(lv.candidates[name]), **rule)
    if "every" in o:
        tgt = v5._every(tgt, int(o["every"]))
    skip = v5._missing(m.index, int(o["missing_seed"])) if "missing_seed" in o else None
    margin = v6.margin_spec(cfg, stress=bool(o.get("margin_stress", False)))
    if o.get("governor_off", False):
        margin = replace(margin, min_uni_mmr=1.0, gross_cap=1e9)
    days = int(o.get("slice_days", x.slice_days))
    step = None if days <= 1 else (1.0 / slots) / days
    return run_levered(tgt, m, costs, lag=lg, band=float(o.get("band", b.band)),
                       hedge_tolerance=b.hedge_tolerance, margin=margin, financing=rate,
                       skip=skip, adv_cap=lv.adv_participation_cap, slice_step=step)


def _family(name: str, cfg: RobustBookV7Config | None = None) -> dict[str, dict[str, Any]]:
    """De verstoringen van v6, plus de twee nieuwe parameters (spreiding, en voor S2 de
    spread): ook die moeten op een plateau staan, niet op een piek."""
    fam = v6._family() | {"slice_days_3": {"slice_days": 3}, "slice_days_10": {"slice_days": 10}}
    if cfg is not None and name in cfg.v7.tranche:
        fam |= {"lever_spread_0.05": {"lever_spread": 0.05},
                "lever_spread_0.15": {"lever_spread": 0.15}}
    return fam


def _attribution(full: Callable[..., BookResult], brief: Callable[[BookResult], dict]) -> dict:
    """Wat de spreiding doet: dezelfde kandidaat ongespreid (alleen ter informatie)."""
    return {"execution_unsliced": brief(full(ov={"slice_days": 1}))}


V7 = v6.Programme(
    name="robust_book_v7", wave=7, config=CONFIG, spec=SPEC_PATH, artefact_dir=ARTEFACT_DIR,
    candidates=CANDIDATES, hypotheses=HYPOTHESES, load_config=robust_book_v7_config,
    simulate=simulate, family=_family, battery_extra=_attribution,
    notes="Twee geplande trials: S1 = v6-K1 met gespreide uitvoering (5 dagstappen per "
          "tranche), S2 = S1 plus een gefinancierde tranche (carry >= leenrente + 10 pp). "
          "Post-hoc na v6 (W_DEV), voor elke v6-lezing buiten W_DEV, en zo geboekt.")


def main(argv: Sequence[str]) -> None:
    cmds = {"freeze": lambda: print(v6.freeze(prog=V7)),
            "run": lambda: print(json.dumps(v6.run_programme(prog=V7), indent=2, default=float)),
            "oos": lambda: print(json.dumps(v6.read_oos(prog=V7), indent=2, default=float)),
            "forward": lambda: print(json.dumps(v6.read_forward(prog=V7), indent=2,
                                                default=float))}
    require(len(argv) == 1 and argv[0] in cmds,
            "Gebruik: python -I -m tradebot.systematic.programme_v7 {freeze|run|oos|forward}",
            DataContractError)
    cmds[argv[0]]()


if __name__ == "__main__":
    main(sys.argv[1:])
