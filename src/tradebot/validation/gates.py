"""De promotiepoort — `run_promotion_gates` → onveranderlijk `GateResult`.

Phase 2 deliverable, gebouwd in Phase 7/8 Stage B-1. **Dit is het bestand dat
D-1 sluit.**

WAT D-1 WAS
===========
`docs/model_risk_policy.md` beweerde sinds Phase 2 dat promotie wordt geblokkeerd
door een lookahead-suite en een reeks statistische poorten. Die poorten bestonden
niet. `reports/phase3_exit_report.md` §0 en §5 meldden dat expliciet; drie fasen
lang heeft niemand het opgepakt. Het beleid was waar op papier en onwaar in de
machine, en dat is de gevaarlijkste van de twee combinaties: iedereen handelt
alsof de controle bestaat.

HET ONTWERP: ÉÉN POORT, VIJF EISEN, GEEN DEELSCORE
==================================================
`run_promotion_gates` geeft één `GateResult` terug. Dat object is `frozen` en
kent geen manier om een `False` in een `True` te veranderen. Er is geen
`force=`, geen `override=`, geen `warn_only=`. Een model dat de poort niet haalt,
haalt hem niet.

De vijf eisen, elk met zijn eigen falsificatierichting:

1. **Pre-registratie** — de hypothese lag vast vóór de meting. Zonder dit is
   elke andere toets een toets op een hypothese die uit de data is afgelezen.
2. **Data-adequaatheid** — er was genoeg data om iets te kunnen zien. Faalt
   deze, dan luidt het oordeel `UNPROVEN — insufficient data` en NOOIT
   `FALSIFIED` (fase-no-go 13).
3. **Lookahead-suite** — de zes causaliteitstests staan groen.
4. **DSR** — de Sharpe overleeft de correctie voor `M` trials.
5. **SPA** — de beste kandidaat verslaat de benchmark, ook onder de
   conservatieve variant van Hansen.

**Er is geen gewogen totaalscore.** Een model dat vier van de vijf haalt, is
geweigerd. Een score zou uitnodigen tot afwegen, en de hele reden dat deze
poorten bestaan is dat afwegen achteraf precies de vrijheidsgraad is die
data-snooping mogelijk maakt.

WAT DEZE MODULE NIET DOET
=========================
Hij **traint niets, fit niets en kiest niets**. Hij krijgt gemeten uitkomsten
aangeleverd en velt een oordeel. Dat is opzettelijk: een poort die zelf rekent,
kan zijn eigen invoer beïnvloeden.

Ref: audit §17.1, §18.1, §26; `fase_2_research_falsification.md`;
`backtest/baseline_report.py` (het patroon dat deze module generiek maakt).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np

from ..registry.trial_counter import TrialCount
from ..schemas.config import ValidationConfig
from ..utils.failfast import DataContractError, require
from .dsr import DsrResult, dsr_gate
from .spa import SpaResult, spa_gate

__all__ = [
    "GateResult",
    "GateVerdict",
    "PROMOTION_GATE_NAMES",
    "run_promotion_gates",
]

#: De vijf poorten, in de volgorde waarin ze worden geëvalueerd. De volgorde is
#: niet cosmetisch: pre-registratie en adequaatheid gaan VOORAF aan elke
#: statistische toets, omdat een toets op niet-geregistreerde of ontoereikende
#: data geen betekenis heeft die door een p-waarde kan worden gered.
PROMOTION_GATE_NAMES = (
    "preregistration",
    "data_adequacy",
    "lookahead_suite",
    "dsr",
    "spa",
)


class GateVerdict:
    """De drie mogelijke oordelen. Bewust geen `Enum` met een vierde optie."""

    PROMOTED = "PROMOTED"
    FALSIFIED = "FALSIFIED"
    #: Ontoereikende data. **Niet** `FALSIFIED` — no-go 13. Onbewezen is niet
    #: hetzelfde als bewezen slecht (audit §6).
    UNPROVEN = "UNPROVEN — insufficient data"


@dataclass(frozen=True)
class GateResult:
    """Onveranderlijk oordeel over één model. Er is geen weg om dit te wijzigen."""

    model_id: str
    verdict: str
    #: Per poort: geslaagd of niet. Alle vijf moeten True zijn voor PROMOTED.
    gates: dict[str, bool]
    #: Per poort de reden, ook bij slagen. Een geslaagde poort zonder toelichting
    #: is niet controleerbaar.
    reasons: dict[str, str]
    trial_count: TrialCount
    dsr_result: DsrResult | None
    spa_result: SpaResult | None
    #: Verplichte herkomstvelden. Een oordeel zonder deze is niet auditbaar.
    git_sha: str
    config_hash: str
    data_hash: str
    preregistration_id: str
    #: De twee labels die audit §16.1 op elke kostenclaim eist.
    cost_labels: tuple[str, ...] = ("IMPACT_UNCALIBRATED", "SPREAD_ASSUMED")
    ts_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self) -> None:
        missing = [
            name for name in ("git_sha", "config_hash", "data_hash",
                              "preregistration_id")
            if not str(getattr(self, name)).strip()
        ]
        require(
            not missing,
            f"GateResult mist verplichte herkomstvelden: {missing}. Een "
            f"promotieoordeel zonder resolvable herkomst is per audit §26 "
            f"INVALIDE - het kan niet worden gereproduceerd en dus niet worden "
            f"nagerekend.",
            DataContractError,
            model_id=self.model_id,
        )
        require(
            set(self.gates) == set(PROMOTION_GATE_NAMES),
            f"GateResult moet precies de vijf poorten dragen "
            f"{PROMOTION_GATE_NAMES}, kreeg {sorted(self.gates)}. Een ontbrekende "
            f"poort is geen geslaagde poort.",
            DataContractError,
        )

    @property
    def passed(self) -> bool:
        """Alle vijf, of niets. Er is geen deelscore."""
        return self.verdict == GateVerdict.PROMOTED

    @property
    def failed_gates(self) -> tuple[str, ...]:
        return tuple(n for n in PROMOTION_GATE_NAMES if not self.gates[n])

    def as_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "model_id": self.model_id,
            "verdict": self.verdict,
            "passed": self.passed,
            "gates": dict(self.gates),
            "reasons": dict(self.reasons),
            "failed_gates": list(self.failed_gates),
            "git_sha": self.git_sha,
            "config_hash": self.config_hash,
            "data_hash": self.data_hash,
            "preregistration_id": self.preregistration_id,
            "cost_labels": list(self.cost_labels),
            "ts_utc": self.ts_utc,
        }
        d.update(self.trial_count.as_dict())
        if self.dsr_result is not None:
            d.update(self.dsr_result.as_dict())
        if self.spa_result is not None:
            d.update(self.spa_result.as_dict())
        return d

    def summary(self) -> str:
        """Eén regel per poort, geschikt voor een rapport of CI-log."""
        lines = [f"{self.model_id}: {self.verdict}"]
        for name in PROMOTION_GATE_NAMES:
            mark = "PASS" if self.gates[name] else "FAIL"
            lines.append(f"  [{mark}] {name}: {self.reasons.get(name, '')}")
        return "\n".join(lines)


def run_promotion_gates(
    *,
    model_id: str,
    oos_returns: np.ndarray,
    benchmark_returns: np.ndarray,
    competitor_returns: np.ndarray,
    trial_count: TrialCount,
    config: ValidationConfig,
    preregistration_id: str,
    git_sha: str,
    config_hash: str,
    data_hash: str,
    data_is_adequate: bool,
    adequacy_reason: str,
    lookahead_suite_passed: bool,
    lookahead_reason: str,
    spa_seed: int | None = None,
) -> GateResult:
    """De ENIGE toegestane weg van een gemeten model naar een promotieoordeel.

    Elke parameter is verplicht. Er is geen default die een poort kan overslaan,
    en er is geen `force`-argument.

    Parameters
    ----------
    oos_returns
        De OUT-OF-SAMPLE returnreeks. In-sample returns hier aanleveren is een
        contractschending die deze functie niet kan detecteren — dat is wat de
        lookahead-suite en de purged walk-forward moeten afdwingen.
    competitor_returns
        `(T, S)` — de kandidatenverzameling voor SPA, inclusief het model zelf.
    data_is_adequate, adequacy_reason
        Uitkomst van `validation/data_adequacy.py`, gemeten VÓÓR de fit.
    lookahead_suite_passed, lookahead_reason
        Uitkomst van `tests/lookahead/`. Deze functie draait de tests niet zelf:
        een poort die zijn eigen bewijs produceert, is geen poort.

    Returns
    -------
    GateResult
        `verdict` is `UNPROVEN` zodra de data-adequaatheid faalt — ook wanneer
        andere poorten ook falen. Ontoereikende data maakt de overige uitslagen
        betekenisloos, en ze als falsificatie boeken zou een model veroordelen
        voor een tekort in de dataset (no-go 13).
    """
    require(
        bool(str(model_id).strip()),
        "run_promotion_gates vereist een niet-lege model_id.",
        DataContractError,
    )

    gates: dict[str, bool] = {}
    reasons: dict[str, str] = {}

    # -- 1. Pre-registratie ------------------------------------------------- #
    has_prereg = bool(str(preregistration_id).strip())
    gates["preregistration"] = has_prereg
    reasons["preregistration"] = (
        f"bevroren pre-registratie {preregistration_id}" if has_prereg
        else "GEEN pre-registratie — de hypothese lag niet vast vóór de meting, "
             "dus elke p-waarde hierna toetst een hypothese die uit de data is "
             "afgelezen"
    )

    # -- 2. Data-adequaatheid ----------------------------------------------- #
    gates["data_adequacy"] = bool(data_is_adequate)
    reasons["data_adequacy"] = adequacy_reason

    # -- 3. Lookahead-suite -------------------------------------------------- #
    gates["lookahead_suite"] = bool(lookahead_suite_passed)
    reasons["lookahead_suite"] = lookahead_reason

    # Ontoereikende data maakt DSR en SPA betekenisloos. Ze tóch draaien zou een
    # p-waarde produceren die iemand later citeert.
    dsr_result: DsrResult | None = None
    spa_result: SpaResult | None = None

    if not data_is_adequate:
        gates["dsr"] = False
        gates["spa"] = False
        reasons["dsr"] = (
            "NIET GEDRAAID — de Data Adequacy Gate faalde. Een DSR op "
            "ontoereikende data levert een getal dat geen significantie meet.")
        reasons["spa"] = "NIET GEDRAAID — idem."
        return GateResult(
            model_id=model_id,
            verdict=GateVerdict.UNPROVEN,
            gates=gates,
            reasons=reasons,
            trial_count=trial_count,
            dsr_result=None,
            spa_result=None,
            git_sha=git_sha,
            config_hash=config_hash,
            data_hash=data_hash,
            preregistration_id=preregistration_id or "GEEN",
        )

    # -- 4. DSR -------------------------------------------------------------- #
    dsr_result = dsr_gate(
        oos_returns, trial_count=trial_count, config=config)
    gates["dsr"] = dsr_result.passed
    reasons["dsr"] = (
        f"DSR={dsr_result.dsr:.4f} tegen drempel {1.0 - config.dsr_alpha:.4f} "
        f"bij M={trial_count.value} ({trial_count.source}) op "
        f"{dsr_result.n_obs} observaties — {dsr_result.verdict}"
    )

    # -- 5. SPA -------------------------------------------------------------- #
    spa_result = spa_gate(
        benchmark_returns, competitor_returns, config=config, seed=spa_seed)
    gates["spa"] = spa_result.passed
    reasons["spa"] = (
        f"SPA p_consistent={spa_result.p_value:.4f}, p_upper="
        f"{spa_result.p_value_upper:.4f} tegen alpha={config.spa_alpha} over "
        f"{spa_result.n_strategies} kandidaten — {spa_result.verdict}"
    )

    all_passed = all(gates[name] for name in PROMOTION_GATE_NAMES)
    verdict = GateVerdict.PROMOTED if all_passed else GateVerdict.FALSIFIED

    return GateResult(
        model_id=model_id,
        verdict=verdict,
        gates=gates,
        reasons=reasons,
        trial_count=trial_count,
        dsr_result=dsr_result,
        spa_result=spa_result,
        git_sha=git_sha,
        config_hash=config_hash,
        data_hash=data_hash,
        # Een ONTBREKENDE pre-registratie wordt als `"GEEN"` vastgelegd en niet
        # als lege string. Het verschil is wezenlijk: `__post_init__` weigert een
        # leeg herkomstveld, en dan zou een model zonder pre-registratie een
        # CRASH opleveren in plaats van een geregistreerde WEIGERING. De weigering
        # is het waardevolle artefact - een crash laat geen spoor na dat dit model
        # ooit is aangeboden.
        preregistration_id=preregistration_id or "GEEN",
    )
