"""Afdwingende DSR-wrapper — Phase 2 deliverable, gebouwd in Phase 7/8 Stage B-2.

WAT DEZE MODULE WEL EN NIET IS
==============================
Dit is **geen tweede DSR-implementatie**. De rekenkern staat in
`backtest/metrics.py::deflated_sharpe` (Bailey & López de Prado 2014, met de
correcte Euler-Mascheroni-term voor het expected maximum) en blijft daar. Die
implementatie is in Phase 5 geverifieerd en wordt hier **hergebruikt, niet
herbouwd** — een tweede implementatie zou twee getallen opleveren die uit elkaar
kunnen lopen.

Wat deze module toevoegt is **handhaving**:

1. `M` is VERPLICHT. Er is geen default. Een DSR met een impliciete `M` is een
   DSR met een impliciet aantal trials, en dat is precies de vrijheidsgraad die
   de toets hoort weg te nemen.
2. `M` moet BEVROREN zijn voor een rapporteerbaar resultaat. Een live-`M` uit de
   ledger geeft morgen een ander antwoord.
3. De skew en kurtosis worden GEMETEN, niet aangenomen. Tot fase 10 stap 4A
   defaultete de onderliggende functie naar `skew=0, kurt=3` (Gaussisch);
   crypto-returns zijn dat niet, en een Gaussische aanname maakt de variantie
   van de Sharpe-schatter te klein en de toets dus te soepel. Die defaults
   bestaan sinds stap 4A niet meer — `metrics.deflated_sharpe` WEIGERT een
   aanroep zonder momenten (MEASUREMENT_CONTRACT.md §6) — maar het meten
   gebeurt nog steeds hier.
4. Het oordeel draagt zijn eigen onzekerheid mee (`M_UNCERTAINTY_NOTE`).
5. **De variantie van de trial-Sharpes is een GEREGISTREERDE keuze, geen
   default.** `V[{SR_m}]` zou de empirische spreiding van de M trial-Sharpes
   moeten zijn. Die reeks bestaat hier niet: de `TrialCount` telt hypothesen,
   hij bewaart hun Sharpes niet. Deze gate gebruikt daarom de gedocumenteerde
   benadering `1/n_obs` uit §6 — en zegt dat, met `approximation="normal"`, in
   elk artefact dat zij produceert. Een benadering die in een JSON staat, is een
   keuze; een benadering die in een default staat, is een aanname.

WAAROM EEN RANDGEVAL ALS NIET-SIGNIFICANT WORDT GELEZEN
=======================================================
`M` is een ondergrens: de seed is gereconstrueerd uit een logboek en trials die
daar nooit in belandden, ontbreken. Een ontbrekende trial maakt de DSR te
OPTIMISTISCH. Elke DSR hier is daarom een bovengrens op de werkelijke
significantie. `is_marginal` markeert de zone waarin die onzekerheid het oordeel
kan omdraaien.

Ref: audit §17.1, §26 (AC-5); `fase_2_research_falsification.md`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..backtest.metrics import deflated_sharpe
from ..registry.trial_counter import M_UNCERTAINTY_NOTE, TrialCount
from ..schemas.config import ValidationConfig, backtest_config
from ..utils.failfast import DataContractError, require
from .inference import require_sharpe_triple

__all__ = ["DsrResult", "MARGINAL_BAND", "dsr_gate"]

#: Breedte van de zone rond `alpha` waarbinnen een uitslag als randgeval geldt.
#: Niet configureerbaar: dit is geen drempel maar een eerlijkheidsmarge op een
#: `M` waarvan bekend is dat hij een ondergrens is.
MARGINAL_BAND = 0.01

#: Minimaal aantal observaties. Onder deze grens is de variantie van de
#: Sharpe-schatter zo groot dat de DSR geen onderscheidend vermogen heeft.
#: Bailey-Lopez de Prado's afleiding steunt op de asymptotische normaliteit van
#: de Sharpe-schatter, en die geldt niet op een handvol punten.
MIN_OBS_FOR_DSR = 30


@dataclass(frozen=True)
class DsrResult:
    """Een onveranderlijk DSR-oordeel met alles wat nodig is om het te herhalen."""

    dsr: float
    #: De PER-BAR Sharpe waarop de DSR is berekend (Bailey-Lopez de Prado).
    sharpe_observed: float
    n_obs: int
    trial_count: TrialCount
    alpha: float
    skew: float
    kurtosis: float
    passed: bool
    is_marginal: bool
    #: `V[{SR_m}]` zoals aan `metrics.deflated_sharpe` meegegeven.
    sr_variance: float
    #: `"normal"` of `"empirical"` — welke herkomst die variantie had (§6).
    approximation: str
    #: Het venster waartegen de Sharpe is gemeten (§10). De DSR-formule
    #: gebruikt de annualisatie niet; het oordeel is zonder haar niet te
    #: vergelijken met de t-drempel die bij dit venster hoort.
    bars_per_year: float
    t_years: float

    def as_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "dsr": self.dsr,
            "sharpe_observed": self.sharpe_observed,
            "n_obs": self.n_obs,
            "alpha": self.alpha,
            "skew": self.skew,
            "kurtosis": self.kurtosis,
            "passed": self.passed,
            "is_marginal": self.is_marginal,
            "verdict": self.verdict,
            "sr_variance": self.sr_variance,
            "dsr_approximation": self.approximation,
            "bars_per_year": self.bars_per_year,
            "t_years": self.t_years,
        }
        d.update(self.trial_count.as_dict())
        # MEASUREMENT_CONTRACT.md §10: dit record draagt een Sharpe
        # (`sharpe_observed`) en mag dus niet zonder zijn drietal naar buiten.
        # De poort staat in `inference.py` en wordt hier AANGEROEPEN, niet
        # nagebouwd.
        require_sharpe_triple(d, where="DsrResult.as_dict")
        return d

    @property
    def verdict(self) -> str:
        if self.is_marginal:
            return "MARGINAL — lees als NIET significant"
        return "PASS" if self.passed else "FAIL"


def dsr_gate(
    returns: np.ndarray,
    *,
    trial_count: TrialCount,
    config: ValidationConfig,
    sharpe_observed: float | None = None,
    require_frozen_m: bool = True,
) -> DsrResult:
    """Draai de DSR als POORT en niet als rapportcijfer.

    Parameters
    ----------
    returns
        De out-of-sample returnreeks waarop de Sharpe is gemeten.
    trial_count
        VERPLICHT. Geen default, geen terugval naar de live-ledger. Bouw hem met
        `registry.trial_counter.frozen_trial_count(<preregistratie>)`.
    config
        `ValidationConfig`; `dsr_alpha` komt hiervandaan en niet uit deze module.
    sharpe_observed
        Optioneel al berekend. Wordt anders uit `returns` gemeten.
    require_frozen_m
        Zet dit alleen op False voor een EXPLORATIEVE meting die nergens wordt
        gerapporteerd. Een gerapporteerd resultaat met een live-`M` is niet
        reproduceerbaar.

    Raises
    ------
    DataContractError
        Bij een live-`M` onder `require_frozen_m`, bij te weinig observaties, of
        bij niet-eindige returns. Er is geen pad waarin deze gate een oordeel
        geeft op input die hij niet kan beoordelen.
    """
    arr = np.asarray(returns, dtype=np.float64).ravel()

    require(
        np.isfinite(arr).all(),
        f"DSR kreeg {int((~np.isfinite(arr)).sum())} niet-eindige waarde(n). "
        f"Een NaN stilzwijgend wegmiddelen verandert de steekproef waarover de "
        f"Sharpe is berekend, en dus de toets zelf.",
        DataContractError,
        n_obs=int(arr.size),
    )
    require(
        arr.size >= MIN_OBS_FOR_DSR,
        f"DSR vereist minimaal {MIN_OBS_FOR_DSR} observaties, kreeg {arr.size}. "
        f"De afleiding steunt op de asymptotische normaliteit van de "
        f"Sharpe-schatter; op minder punten meet de toets niets.",
        DataContractError,
        n_obs=int(arr.size),
    )
    require(
        isinstance(trial_count, TrialCount),
        "dsr_gate vereist een TrialCount, geen kaal getal. Een M zonder herkomst "
        "is niet auditbaar: het rapport moet kunnen tonen waar hij vandaan komt "
        "en welk deel gereconstrueerd is.",
        DataContractError,
    )
    if require_frozen_m:
        require(
            trial_count.is_reproducible,
            f"dsr_gate kreeg een LIVE trial-count (origin={trial_count.origin}). "
            f"Een gerapporteerde DSR moet uit een BEVROREN pre-registratie komen, "
            f"anders geeft dezelfde code op dezelfde data morgen een ander "
            f"antwoord zodra er een trial is bijgeschreven. Gebruik "
            f"`frozen_trial_count(<pad naar de pre-registratie>)`.",
            DataContractError,
            m_source=trial_count.source,
        )

    std = float(arr.std(ddof=1))
    # DEGENERATIE-CONTROLE, RELATIEF EN NIET TEGEN NUL.
    #
    # `np.full(200, 0.001).std(ddof=1)` geeft 2.17e-19 en niet 0.0: de variantie
    # wordt berekend als sum((x - mean)^2) en daarin blijft bij identieke
    # waarden een afrondingsrest staan (catastrophic cancellation). Een toets
    # `std > 0.0` laat zo'n reeks dus DOOR, en levert een Sharpe van ~4.6e15 op
    # die daarna keurig door de DSR wordt verwerkt.
    #
    # Gevonden door `test_constant_series_crashes_instead_of_returning_zero`,
    # die op de eerste versie van deze module groen had moeten zijn en dat niet
    # was. De drempel is daarom RELATIEF aan de schaal van de reeks.
    scale = max(float(np.abs(arr).mean()), 1e-300)
    require(
        std > 1e-12 * scale,
        f"DSR kreeg een (bijna) constante returnreeks: std={std:.3e} tegen een "
        f"schaal van {scale:.3e}. De Sharpe is dan numeriek betekenisloos - dit "
        f"is een datadefect, geen resultaat van nul.",
        DataContractError,
        std=std,
    )
    sr = float(arr.mean() / std) if sharpe_observed is None else float(sharpe_observed)

    # Skew en kurtosis worden GEMETEN. Crypto-returns zijn scheef en
    # dikstaartig, en de Gaussische aanname maakt var(SR) te klein en de toets
    # dus te soepel.
    from scipy import stats as _stats

    skew = float(_stats.skew(arr))
    kurt = float(_stats.kurtosis(arr, fisher=False))

    # V[{SR_m}]: de trial-Sharpes zijn hier NIET beschikbaar -- een `TrialCount`
    # telt hypothesen en bewaart hun Sharpes niet. §6 van het meetcontract staat
    # dan de benadering `1/n_obs` toe, MITS dat feit in het artefact staat.
    # Vandaar de expliciete vlag; zij loopt door tot in `as_dict()`.
    n_obs = int(arr.size)
    sr_variance = 1.0 / n_obs
    bars_per_year = float(backtest_config().bars_per_year)
    result = deflated_sharpe(
        sr,
        n_obs=n_obs,
        n_trials=trial_count.value,
        sr_variance=sr_variance,
        skew=skew,
        kurtosis=kurt,
        bars_per_year=bars_per_year,
        approximation="normal",
    )
    dsr = float(result.dsr)

    # De DSR is een KANS dat de waargenomen Sharpe de expected maximum onder de
    # nul overtreft. Slagen betekent dus dsr > 1 - alpha, niet dsr < alpha.
    threshold = 1.0 - config.dsr_alpha
    passed = dsr > threshold
    is_marginal = abs(dsr - threshold) <= MARGINAL_BAND

    return DsrResult(
        dsr=dsr,
        sharpe_observed=sr,
        n_obs=n_obs,
        trial_count=trial_count,
        alpha=config.dsr_alpha,
        skew=skew,
        kurtosis=kurt,
        passed=bool(passed and not is_marginal),
        is_marginal=bool(is_marginal),
        sr_variance=result.sr_variance,
        approximation=result.approximation,
        bars_per_year=result.bars_per_year,
        t_years=result.t_years,
    )


def dsr_from_preregistration(
    returns: np.ndarray,
    preregistration_path: Path | str,
    config: ValidationConfig,
) -> DsrResult:
    """Gemakspad: laad de bevroren `M` en draai de gate in één stap."""
    from ..registry.trial_counter import frozen_trial_count

    return dsr_gate(
        returns,
        trial_count=frozen_trial_count(preregistration_path),
        config=config,
    )


def uncertainty_note() -> str:
    """De tekst die elk artefact met een DSR woordelijk moet meedragen."""
    return M_UNCERTAINTY_NOTE
