"""Hansen's SPA als promotiegate — Phase 2 deliverable, gebouwd in Stage B-2.

WAT DEZE MODULE TOEVOEGT AAN `backtest/spa.py`
==============================================
De rekenkern staat in `backtest/spa.py::spa_test` (Hansen 2005, stationary
bootstrap met Politis-White blokgrootte) en blijft daar. Deze module maakt er
een **poort** van.

WAAROM SPA NAAST DE DSR
=======================
De DSR corrigeert voor het AANTAL trials. SPA corrigeert voor iets anders: de
vraag of de BESTE van een verzameling strategieën de benchmark verslaat, gegeven
dat je de beste hebt uitgekozen. Twee verschillende data-snooping-problemen:

* DSR: *"ik heb 2776 dingen geprobeerd en dit is de beste Sharpe"*
* SPA: *"ik heb deze N kandidaten tegen de benchmark gezet en de beste wint"*

Een strategie die alleen de DSR haalt maar niet SPA, wint van het nulmodel maar
niet van de benchmark waartegen hij moet concurreren. Beide zijn nodig.

DE DRIE p-WAARDEN VAN HANSEN, EN WELKE TELT
===========================================
`spa_test` geeft `p_value_lower`, `p_value_consistent` en `p_value_upper`. Zij
verschillen in hoeveel ze aannemen over de slechte strategieën in de verzameling:

* `lower`      — meest liberaal; behandelt slechte modellen alsof ze er niet zijn
* `consistent` — Hansen's aanbevolen schatter
* `upper`      — meest conservatief; neemt alle modellen even serieus

**Deze gate gebruikt `p_value_consistent` als oordeel en eist dat `upper` het
niet tegenspreekt.** De reden staat in §4.2 van dit platform's doctrine: een
uitslag die alleen onder de meest liberale variant significant is, is een
uitslag die afhangt van hoe je de verliezers behandelt — en dat is geen
eigenschap van het winnende model.

Ref: audit §17.1; `fase_2_research_falsification.md`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..backtest.spa import spa_test
from ..schemas.config import ValidationConfig
from ..utils.failfast import DataContractError, require

__all__ = ["SpaResult", "spa_gate"]

#: Minimaal aantal observaties voor een zinvolle stationary bootstrap. Onder
#: deze grens bevat een blok een substantieel deel van de reeks en meet de
#: bootstrap de blokkeuze in plaats van de onzekerheid.
MIN_OBS_FOR_SPA = 50


@dataclass(frozen=True)
class SpaResult:
    """Een onveranderlijk SPA-oordeel."""

    p_value: float
    p_value_lower: float
    p_value_upper: float
    best_strategy_idx: int
    n_strategies: int
    n_obs: int
    alpha: float
    passed: bool
    #: True wanneer `consistent` slaagt maar `upper` niet: de uitslag hangt dan
    #: af van hoe de slechte modellen in de verzameling worden meegewogen.
    is_assumption_dependent: bool

    @property
    def verdict(self) -> str:
        if self.is_assumption_dependent:
            return "ASSUMPTION-DEPENDENT — lees als NIET significant"
        return "PASS" if self.passed else "FAIL"

    def as_dict(self) -> dict[str, Any]:
        return {
            "spa_p_value": self.p_value,
            "spa_p_value_lower": self.p_value_lower,
            "spa_p_value_upper": self.p_value_upper,
            "spa_best_strategy_idx": self.best_strategy_idx,
            "spa_n_strategies": self.n_strategies,
            "spa_n_obs": self.n_obs,
            "spa_alpha": self.alpha,
            "spa_passed": self.passed,
            "spa_assumption_dependent": self.is_assumption_dependent,
            "spa_verdict": self.verdict,
        }


def spa_gate(
    benchmark_returns: np.ndarray,
    strategy_returns: np.ndarray,
    *,
    config: ValidationConfig,
    seed: int | None = None,
) -> SpaResult:
    """Draai Hansen's SPA als poort.

    Parameters
    ----------
    benchmark_returns
        1D-reeks van de benchmark die verslagen moet worden.
    strategy_returns
        `(T, S)` — T perioden × S kandidaten. Een 1D-array wordt als S=1 gelezen.
    config
        `spa_alpha`, `spa_bootstrap_reps` en `spa_block_length` komen hiervandaan.
    seed
        Zet hem voor een reproduceerbare bootstrap. Een SPA-uitslag zonder seed
        is niet exact herhaalbaar.

    Raises
    ------
    DataContractError
        Bij niet-eindige input, te weinig observaties, of een lengteverschil
        tussen benchmark en kandidaten.
    """
    bench = np.asarray(benchmark_returns, dtype=np.float64).ravel()
    strat = np.asarray(strategy_returns, dtype=np.float64)
    if strat.ndim == 1:
        strat = strat.reshape(-1, 1)

    require(
        np.isfinite(bench).all() and np.isfinite(strat).all(),
        "SPA kreeg niet-eindige waarden. De bootstrap zou die meesamplen en een "
        "p-waarde geven over een reeks die deels niet bestaat.",
        DataContractError,
    )
    require(
        bench.shape[0] == strat.shape[0],
        f"SPA: benchmark heeft {bench.shape[0]} observaties, de kandidaten "
        f"{strat.shape[0]}. Twee reeksen van verschillende lengte vergelijken "
        f"betekent dat het verschil deels een verschil in STEEKPROEF is.",
        DataContractError,
    )
    require(
        bench.shape[0] >= MIN_OBS_FOR_SPA,
        f"SPA vereist minimaal {MIN_OBS_FOR_SPA} observaties, kreeg "
        f"{bench.shape[0]}. Daaronder beslaat een bootstrapblok een substantieel "
        f"deel van de reeks en meet de toets de blokkeuze.",
        DataContractError,
    )

    if seed is not None:
        np.random.seed(int(seed))

    raw = spa_test(
        benchmark_returns=bench,
        strategy_returns_matrix=strat,
        n_bootstrap=config.spa_bootstrap_reps,
        block_size=config.spa_block_length,
        significance=config.spa_alpha,
    )

    p_cons = float(raw["p_value_consistent"])       # type: ignore[arg-type]
    p_low = float(raw["p_value_lower"])             # type: ignore[arg-type]
    p_up = float(raw["p_value_upper"])              # type: ignore[arg-type]

    cons_ok = p_cons < config.spa_alpha
    upper_ok = p_up < config.spa_alpha
    assumption_dependent = bool(cons_ok and not upper_ok)

    return SpaResult(
        p_value=p_cons,
        p_value_lower=p_low,
        p_value_upper=p_up,
        best_strategy_idx=int(raw["best_strategy_idx"]),   # type: ignore[arg-type]
        n_strategies=int(strat.shape[1]),
        n_obs=int(bench.shape[0]),
        alpha=config.spa_alpha,
        passed=bool(cons_ok and upper_ok),
        is_assumption_dependent=assumption_dependent,
    )
