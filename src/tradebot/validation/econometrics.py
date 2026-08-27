"""De econometrische toetsingsketen — Phase 6, deliverable 7 (§8.2).

Vijf toetsen die samen bepalen OF een reeks een bepaald model verdient, voordat
dat model wordt gefit:

    ADF          verwerpt een eenheidswortel        -> is de reeks stationair?
    KPSS         nulhypothese IS stationariteit     -> bevestigt of tegenspreekt
    Ljung-Box    autocorrelatie in de reeks         -> is er lineaire structuur?
    Engle ARCH   autocorrelatie in de KWADRATEN     -> is er vol-clustering?
    CUSUM        structurele breuk in het gemiddelde -> geldt één model over
                                                       het hele venster?

WAAROM ADF ÉN KPSS
==================
Zij hebben tegengestelde nulhypotheses en dat is precies waarom ze samen
informatiever zijn dan elk apart. Vier uitkomsten, vier betekenissen:

    ADF verwerpt, KPSS verwerpt niet   -> stationair. Eenduidig.
    ADF verwerpt niet, KPSS verwerpt   -> eenheidswortel. Eenduidig.
    Beide verwerpen                    -> tegenstrijdig; meestal een teken van
                                          lange-geheugengedrag of een
                                          structurele breuk in het venster.
    Geen van beide verwerpt            -> de data is niet informatief genoeg om
                                          te beslissen. Dat is een RESULTAAT.

Een pijplijn die alleen ADF draait, ziet de laatste twee gevallen als "prima" en
gaat door. Deze module rapporteert het onderscheid.

DE ARCH-TOETS IS DE POORTWACHTER VAN HET VOL-SPOOR
===================================================
Stap 3 van de fase-opdracht: *"Is er geen aantoonbare conditionele
heteroskedasticiteit, dan is een GARCH-structuur niet gerechtvaardigd en stopt
het spoor daar met een gedocumenteerd oordeel."*

Dat is geen formaliteit. GARCH modelleert een conditionele variantie die
varieert met het verleden. Ontbreekt die variatie, dan schat een GARCH-fit
omega, alpha en beta op ruis, convergeert hij naar alpha + beta ~ 0 of naar de
IGARCH-rand, en levert hij een forecast die niet beter kan zijn dan een
constante. Zo'n model FALSIFICEREN zou onterecht zijn: het is niet slechter
gebleken, het was niet van toepassing. Vandaar de actie `descope` in het
H1-stopcriterium.

WAT DEZE MODULE NIET DOET
==========================
Hij beslist niets. Elke functie geeft een :class:`DiagnosticOutcome` met de
statistiek, de p-waarde en de expliciete conclusie in woorden. De poort zelf
staat in de runner, zodat de toets en het besluit gescheiden blijven en het
besluit herleidbaar is naar een getal.

Ref: ARCHITECTUUR_AUDIT_2026-08-22.md §8.2; Engle (1982); Kwiatkowski et al.
(1992); Brown, Durbin & Evans (1975).
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from scipy import stats as _stats
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.stattools import adfuller, kpss

from ..schemas.config import econometrics_config
from ..utils.failfast import DataContractError, require

#: De econometrische drempels komen uit `conf/validation/econometrics.yaml`
#: en niet uit deze module. Zie Stage A-3 van het Phase 7/8-programma: een
#: significantieniveau is een beleidskeuze, geen rekenkundig feit, en hoort
#: daarom gehasht in de config te staan. Ontbreekt de config, dan crasht de
#: import - er is geen ingebouwde terugval.
_ECONO = econometrics_config()


__all__ = [
    "SeriesDiagnostics",
    "DiagnosticOutcome",
    "adf_test",
    "arch_gate_verdict",
    "cusum_test",
    "diagnose_series",
    "engle_arch_test",
    "kpss_test",
    "ljung_box_test",
]


@dataclass(frozen=True)
class DiagnosticOutcome:
    """Eén toets: wat er is gemeten, en wat dat betekent.

    Heet bewust niet ``TestOutcome``: pytest verzamelt elke klasse waarvan de
    naam met ``Test`` begint en klaagt dan dat zij een ``__init__`` heeft. Een
    waarschuwing die bij elke suite-run terugkomt, wordt genegeerd, en genegeerde
    waarschuwingen verbergen de volgende.
    """

    name: str
    statistic: float
    p_value: float
    #: De nulhypothese IN WOORDEN. Zonder haar is "p = 0,03" betekenisloos:
    #: bij ADF betekent verwerping stationariteit, bij KPSS het omgekeerde.
    null_hypothesis: str
    rejected: bool
    alpha: float
    detail: Mapping[str, Any]

    def __post_init__(self) -> None:
        require(
            bool(self.null_hypothesis),
            "Een toetsuitkomst zonder expliciete nulhypothese. Bij ADF betekent "
            "verwerping stationariteit en bij KPSS het omgekeerde; zonder die "
            "tekst is een p-waarde niet interpreteerbaar.",
            DataContractError,
            name=self.name,
        )

    def as_record(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "null_hypothesis": self.null_hypothesis,
            "rejected": self.rejected,
            "alpha": self.alpha,
            "detail": dict(self.detail),
        }


def _clean(series: np.ndarray, *, name: str, min_obs: int) -> np.ndarray:
    arr = np.asarray(series, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    require(
        arr.size >= min_obs,
        f"{name} kreeg te weinig eindige observaties. Er wordt NIET "
        "geïmputeerd: een gat in de reeks is een databevinding.",
        DataContractError,
        n_finite=int(arr.size), min_obs=min_obs, test=name,
    )
    require(
        float(np.std(arr)) > 0.0,
        f"{name} kreeg een reeks zonder variatie. Elke toets op een constante "
        "is degenerate en zou een p-waarde opleveren die niets betekent.",
        DataContractError,
        test=name, n=int(arr.size),
    )
    return arr


def adf_test(series: np.ndarray, *, alpha: float = _ECONO.alpha,
             regression: Literal["c", "ct", "n"] = "c") -> DiagnosticOutcome:
    """Augmented Dickey-Fuller. Verwerping = GEEN eenheidswortel = stationair."""
    arr = _clean(series, name="ADF", min_obs=50)
    stat, p_value, used_lag, n_obs, crit, _ = adfuller(
        arr, regression=regression, autolag="AIC")
    return DiagnosticOutcome(
        name="adf",
        statistic=float(stat),
        p_value=float(p_value),
        null_hypothesis="de reeks bevat een eenheidswortel (niet-stationair)",
        rejected=bool(p_value < alpha),
        alpha=alpha,
        detail={"used_lag": int(used_lag), "n_obs": int(n_obs),
                "regression": regression,
                "critical_values": {k: float(v) for k, v in crit.items()}},
    )


def kpss_test(series: np.ndarray, *, alpha: float = _ECONO.alpha,
              regression: Literal["c", "ct"] = "c") -> DiagnosticOutcome:
    """KPSS. Verwerping = de reeks is NIET stationair — omgekeerd aan ADF.

    ``statsmodels`` kapt de p-waarde af op [0,01, 0,10]; buiten dat bereik geeft
    hij de rand terug met een InterpolationWarning. Die afkapping wordt hier
    expliciet in `detail` gemeld, want een p van precies 0,01 betekent
    "<= 0,01" en niet "= 0,01", en dat verschil telt in een rapport.
    """
    arr = _clean(series, name="KPSS", min_obs=50)
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        stat, p_value, n_lags, crit = kpss(arr, regression=regression,
                                           nlags="auto")
        clipped = any("p-value" in str(w.message).lower() for w in caught)
    return DiagnosticOutcome(
        name="kpss",
        statistic=float(stat),
        p_value=float(p_value),
        null_hypothesis="de reeks IS stationair (rond een constante of trend)",
        rejected=bool(p_value < alpha),
        alpha=alpha,
        detail={"n_lags": int(n_lags), "regression": regression,
                "p_value_clipped_at_table_edge": bool(clipped),
                "critical_values": {k: float(v) for k, v in crit.items()}},
    )


def ljung_box_test(series: np.ndarray, *, lags: int = _ECONO.ljung_box_lags,
                   alpha: float = _ECONO.alpha) -> DiagnosticOutcome:
    """Ljung-Box op de NIVEAUS. Verwerping = lineaire autocorrelatie aanwezig."""
    arr = _clean(series, name="Ljung-Box", min_obs=max(50, 3 * lags))
    result = acorr_ljungbox(arr, lags=[lags], return_df=True)
    stat = float(result["lb_stat"].iloc[0])
    p_value = float(result["lb_pvalue"].iloc[0])
    return DiagnosticOutcome(
        name="ljung_box",
        statistic=stat,
        p_value=p_value,
        null_hypothesis=f"geen autocorrelatie in de eerste {lags} lags",
        rejected=bool(p_value < alpha),
        alpha=alpha,
        detail={"lags": int(lags)},
    )


def engle_arch_test(series: np.ndarray, *, lags: int = 12,
                    alpha: float = _ECONO.alpha) -> DiagnosticOutcome:
    """Engle (1982) LM-toets op ARCH-effecten. DE POORTWACHTER van het vol-spoor.

    Regressie van het gekwadrateerde residu op zijn eigen lags; de nulhypothese
    is dat alle coëfficiënten nul zijn, oftewel GEEN conditionele
    heteroskedasticiteit. Verwerping rechtvaardigt een GARCH-structuur; geen
    verwerping betekent dat GARCH op deze reeks niet van toepassing is.
    """
    arr = _clean(series, name="Engle ARCH", min_obs=max(50, 4 * lags))
    stat, p_value, f_stat, f_p = het_arch(arr, nlags=lags)
    return DiagnosticOutcome(
        name="engle_arch",
        statistic=float(stat),
        p_value=float(p_value),
        null_hypothesis=(
            f"geen ARCH-effecten t/m lag {lags}: de conditionele variantie is "
            "constant"
        ),
        rejected=bool(p_value < alpha),
        alpha=alpha,
        detail={"lags": int(lags), "f_statistic": float(f_stat),
                "f_p_value": float(f_p)},
    )


def cusum_test(series: np.ndarray, *, alpha: float = _ECONO.alpha) -> DiagnosticOutcome:
    """CUSUM van gestandaardiseerde recursieve residuen (Brown-Durbin-Evans 1975).

    Toetst of het GEMIDDELDE van de reeks over het venster constant blijft. De
    teststatistiek is het maximum van de genormaliseerde cumulatieve som; onder
    de nulhypothese volgt die een Brownse brug, waarvan de kritieke waarden de
    bekende Kolmogorov-Smirnov-achtige grenzen zijn:

        alpha = 0,10 -> 0,850    alpha = 0,05 -> 0,948    alpha = 0,01 -> 1,143

    Waarom dit ertoe doet in deze fase: elk model hier wordt op één venster van
    1.743 bars gefit en beoordeeld. Zit er een structurele breuk in, dan is de
    OOS-prestatie een gemiddelde over twee verschillende regimes en zegt zij
    niets over de toekomst van geen van beide. `reports/phase5_cluster_concentration_audit.md`
    vond al dat de clusterpartitie zes keer wisselde in zes jaar.
    """
    arr = _clean(series, name="CUSUM", min_obs=50)
    n = arr.size
    centred = arr - arr.mean()
    sigma = float(np.std(arr, ddof=1))
    cumulative = np.cumsum(centred) / (sigma * np.sqrt(n))
    statistic = float(np.max(np.abs(cumulative)))

    critical = {0.10: 0.850, 0.05: 0.948, 0.01: 1.143}
    require(
        alpha in critical,
        "CUSUM-kritieke waarden bestaan alleen voor alpha in {0,10, 0,05, 0,01}. "
        "Een geïnterpoleerde kritieke waarde zou een verzonnen drempel zijn.",
        DataContractError,
        alpha=alpha, available=sorted(critical),
    )
    threshold = critical[alpha]
    # Asymptotische p-waarde van de Kolmogorov-verdeling van de Brownse brug.
    p_value = float(_stats.kstwobign.sf(statistic))
    return DiagnosticOutcome(
        name="cusum",
        statistic=statistic,
        p_value=p_value,
        null_hypothesis="het gemiddelde van de reeks is constant over het venster",
        rejected=bool(statistic > threshold),
        alpha=alpha,
        detail={"critical_value": threshold, "n_obs": int(n),
                "argmax_index": int(np.argmax(np.abs(cumulative)))},
    )


@dataclass(frozen=True)
class SeriesDiagnostics:
    """De vijf toetsen op één reeks, met het gecombineerde stationariteitsoordeel."""

    series_name: str
    adf: DiagnosticOutcome
    kpss: DiagnosticOutcome
    ljung_box: DiagnosticOutcome
    engle_arch: DiagnosticOutcome
    cusum: DiagnosticOutcome

    @property
    def stationarity_verdict(self) -> str:
        """De vier mogelijke combinaties van ADF en KPSS, elk met eigen betekenis."""
        if self.adf.rejected and not self.kpss.rejected:
            return "stationair (ADF verwerpt, KPSS verwerpt niet)"
        if not self.adf.rejected and self.kpss.rejected:
            return "eenheidswortel (ADF verwerpt niet, KPSS verwerpt)"
        if self.adf.rejected and self.kpss.rejected:
            return (
                "tegenstrijdig (beide verwerpen) — meestal lange-geheugengedrag "
                "of een structurele breuk binnen het venster"
            )
        return (
            "onbeslist (geen van beide verwerpt) — de reeks is niet informatief "
            "genoeg om te beslissen; dit is een resultaat, geen groen licht"
        )

    @property
    def arch_gate_open(self) -> bool:
        """Mag er een GARCH-structuur op deze reeks worden gefit?"""
        return self.engle_arch.rejected

    def as_record(self) -> dict[str, Any]:
        return {
            "series": self.series_name,
            "stationarity_verdict": self.stationarity_verdict,
            "arch_gate_open": self.arch_gate_open,
            "tests": {t.name: t.as_record() for t in
                      (self.adf, self.kpss, self.ljung_box,
                       self.engle_arch, self.cusum)},
        }


def diagnose_series(
    series: np.ndarray,
    *,
    name: str,
    alpha: float = _ECONO.alpha,
    ljung_box_lags: int = _ECONO.ljung_box_lags,
    arch_lags: int = 12,
) -> SeriesDiagnostics:
    """Draai alle vijf toetsen op één reeks."""
    return SeriesDiagnostics(
        series_name=name,
        adf=adf_test(series, alpha=alpha),
        kpss=kpss_test(series, alpha=alpha),
        ljung_box=ljung_box_test(series, lags=ljung_box_lags, alpha=alpha),
        engle_arch=engle_arch_test(series, lags=arch_lags, alpha=alpha),
        cusum=cusum_test(series, alpha=alpha),
    )


def arch_gate_verdict(diagnostics: SeriesDiagnostics) -> str:
    """Het gedocumenteerde oordeel dat stap 3 uitvraagt, in woorden."""
    if diagnostics.arch_gate_open:
        return (
            f"OPEN — Engle ARCH verwerpt op alpha = {diagnostics.engle_arch.alpha} "
            f"(p = {diagnostics.engle_arch.p_value:.3g}); een GARCH-structuur is "
            f"op {diagnostics.series_name} gerechtvaardigd."
        )
    return (
        f"DICHT — Engle ARCH verwerpt NIET op alpha = "
        f"{diagnostics.engle_arch.alpha} (p = {diagnostics.engle_arch.p_value:.3g}). "
        f"Er is geen aantoonbare conditionele heteroskedasticiteit in "
        f"{diagnostics.series_name}; een GARCH-structuur is niet gerechtvaardigd "
        f"en de reeks wordt uit de competitie gede-scoped. Dat is GEEN "
        f"falsificatie: het model is niet slechter gebleken, het is niet van "
        f"toepassing."
    )
