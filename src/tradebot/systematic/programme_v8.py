"""Robuust boek v8: v7-S2 met een sprongrisicogrens (dag-σ ≤ 20 %).

    python -I -m tradebot.systematic.programme_v8 freeze      # ledger + prereg (M = 27)
    python -I -m tradebot.systematic.programme_v8 run         # W_DEV, batterij, selectie
    python -I -m tradebot.systematic.programme_v8 oos         # backcast + holdout (verbruikt)
    python -I -m tradebot.systematic.programme_v8 forward     # vanaf 2027-04: het vooruit-sample

WAAROM v8 BESTAAT. v6-K1 en v7-S2 faalden in de holdout 2025-26 (Sharpe −3,1 en −2,5). De
forensiek: zonder brede carry (BTC/ETH < 15 %) liet de carry-screen alleen nog
pump-and-dump-microcaps door (HIFI, BAKE, TUT, ALPINE; dag-σ 24-105 % bij instap). Hun
"carry" is een vergoeding voor sprongrisico; hun impact, basis en delistingrisico vraten
alles op. De grens is afgeleid uit al bevroren parameters (zie de config): bij de
1 %-ADV-cap en vijf dagstappen kost een rondgang 0,18·σ, een kwartaal carry op de
instapdrempel is 3,75 %, dus break-even bij σ ≈ 0,21 → 0,20.

Dit ontwerp kent de holdout. Die kan v8 alleen nog verwerpen, niet bevestigen. Het schone
bewijs is het vooruit-sample (slot bevroren 2026-10-09, vóór alle v6-v8-lezingen).
Ontwerp: `docs/superpowers/specs/2026-10-09-robust-book-v8-jump-risk-design.md`.
"""
from __future__ import annotations

import json
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from ..schemas.robust_book_v2 import RobustBookV8Config, robust_book_v8_config
from ..utils.failfast import DataContractError, require
from . import programme_v6 as v6
from . import programme_v7 as v7
from .book import BookResult
from .harvest import BasisCosts
from .leverage import MarginMarket

__all__ = ["CANDIDATES", "V8", "simulate"]

CONFIG = Path("conf/model/robust_book_v8.yaml")
SPEC_PATH = Path("conf/research/preregistration_robust_book_v8.yaml")
ARTEFACT_DIR = Path("artefacts/research/robust_book_v8")
CANDIDATES = ("T2_CARRY_TRANCHE_JUMPCAP",)
HYPOTHESES = {
    "T2_CARRY_TRANCHE_JUMPCAP": "v7-S2 (basiscarry, unified margin, gefinancierde tranche bij "
                                "carry >= leenrente + 10 pp, vijf dagstappen) met een "
                                "sprongrisicogrens: geen carry in een munt met dag-sigma > 20 %.",
}


def simulate(name: str, m: MarginMarket, cfg: RobustBookV8Config, costs: BasisCosts, *,
             lag: int | None = None, ov: Mapping[str, Any] | None = None) -> BookResult:
    """De v8-kandidaat: de kern van v7 met de grens uit `cfg.v8` (override `max_sigma`)."""
    require(name in CANDIDATES, "Onbekende v8-kandidaat.", DataContractError, name=name)
    return v7.simulate_book(name, m, cfg, costs, lag=lag, ov=ov,
                            max_sigma=cfg.v8.max_daily_sigma)


def _family(name: str, cfg: RobustBookV8Config | None = None) -> dict[str, dict[str, Any]]:
    """De verstoringen van v7 (voor een tranche-kandidaat) plus de grens zelf: 0,15 en 0,30."""
    return v7._family(name, cfg) | {"max_sigma_0.15": {"max_sigma": 0.15},
                                    "max_sigma_0.30": {"max_sigma": 0.30}}


def _attribution(full: Callable[..., BookResult], brief: Callable[[BookResult], dict]) -> dict:
    """Ter informatie: dezelfde kandidaat zonder grens (= v7-S2) en ongespreid."""
    return {"jump_cap_off": brief(full(ov={"max_sigma": None})),
            **v7._attribution(full, brief)}


V8 = v6.Programme(
    name="robust_book_v8", wave=8, config=CONFIG, spec=SPEC_PATH, artefact_dir=ARTEFACT_DIR,
    candidates=CANDIDATES, hypotheses=HYPOTHESES, load_config=robust_book_v8_config,
    simulate=simulate, family=_family, battery_extra=_attribution,
    notes="Eén geplande trial: T2 = v7-S2 met een sprongrisicogrens (dag-sigma <= 20 %). "
          "Post-hoc na de holdout-lezingen van v6 en v7 (de holdout is verbruikt), en zo "
          "geboekt. Het schone bewijs is het vooruit-sample.")


def main(argv: Sequence[str]) -> None:
    cmds = {"freeze": lambda: print(v6.freeze(prog=V8)),
            "run": lambda: print(json.dumps(v6.run_programme(prog=V8), indent=2, default=float)),
            "oos": lambda: print(json.dumps(v6.read_oos(prog=V8), indent=2, default=float)),
            "forward": lambda: print(json.dumps(v6.read_forward(prog=V8), indent=2,
                                                default=float))}
    require(len(argv) == 1 and argv[0] in cmds,
            "Gebruik: python -I -m tradebot.systematic.programme_v8 {freeze|run|oos|forward}",
            DataContractError)
    cmds[argv[0]]()


if __name__ == "__main__":
    main(sys.argv[1:])
