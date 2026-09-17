# src/tradebot/validation/sharpe_difference.py
"""De gepaarde Sharpe-toets van H2, met de controles die haar geldig maken.

.. attention::

   **SUPERSEDED — BEVROREN METHODE. GEEN NIEUWE AANROEPPLEKKEN.**

   De ene Sharpe-verschiltoets van dit programma is sinds fase 10 stap 4A
   ``validation/inference.py::sharpe_difference_test``: Ledoit-Wolf (2008) met
   HAC-covariantie van de vier momenten en een gestudentiseerde CIRCULAIRE
   blokbootstrap, zoals `docs/MEASUREMENT_CONTRACT.md` §4 hem voorschrijft.
   **Elke nieuwe vergelijking bindt daaraan.**

   Deze module blijft staan en wordt NIET herschreven. Zij is de bevroren
   METHODE van een AFGESLOTEN, vooraf geregistreerd experiment: H2,
   pre-registratie ``3d3af28730a6c7f9da48d13139522a05``, met de resultaten in
   ``artefacts/governance/phase6_h2_regime_benchmark.json``. Zij implementeert
   Jobson-Korkie/Memmel met een STATIONAIRE bootstrap; dat is een andere toets
   dan Ledoit-Wolf, niet dezelfde met een andere naam. Haar hier vervangen door
   de nieuwe kern zou de methode van een gepubliceerd resultaat met terugwerkende
   kracht veranderen, en daarmee het verslag zelf falsificeren.

   Haar enige consument is ``validation/regime_benchmark.py``, die H2
   reproduceert en daarom bij deze methode hoort te blijven. R-3 (één
   implementatie per statistische grootheid) wordt niet geschonden doordat er
   twee bestanden zijn, maar zou wél worden geschonden zodra een NIEUWE meting
   hieraan bindt in plaats van aan ``inference.py``.

WAAROM GEPAARD EN NIET TWEE LOSSE SHARPES
==========================================
De twee armen conditioneren HETZELFDE primaire signaal. Hun returnreeksen zijn
daardoor sterk gecorreleerd, en die correlatie is precies wat een gepaarde toets
zoveel preciezer maakt dan twee intervallen naast elkaar leggen::

    SE(SR)   = sqrt((1 + SR^2 / 2) / T)                (Lo 2002)
    SE(dSR) ~ SE(SR) * sqrt(2 * (1 - rho))             (Jobson-Korkie, Memmel)

Twee losse Sharpes met overlappende betrouwbaarheidsintervallen zouden hier
"geen verschil" suggereren waar de gepaarde toets wel degelijk iets ziet -- of
andersom. De pre-registratie schrijft daarom de gepaarde toets voor, en dit
module implementeert hem in zijn Memmel-vorm (2003), die de asymptotische
variantie van Jobson-Korkie (1981) corrigeert.

DE CONTROLES ZIJN GEEN BIJVANGST
=================================
Exit-criterium 12 eist een negatieve controle per statistische toets. Voor deze
toets zijn dat er twee, en zij meten verschillende dingen:

    SIZE   verwerpt de toets ongeveer `alpha` van de tijd wanneer de twee
           Sharpes GELIJK zijn? Verwerpt hij vaker, dan is elke significante
           uitslag verdacht.
    POWER  verwerpt de toets wanneer er WEL een verschil is? Bij het verwachte
           effect van 0,08 Sharpe-eenheden en bij het minimaal detecteerbare
           effect uit de power-analyse.

Beide draaien op een gepaarde stationaire block-bootstrap van de ECHTE
returnreeksen, met hun eigen autocorrelatie, hun eigen staarten en hun eigen
onderlinge correlatie. Het nulpaar wordt gemaakt door de UITDAGER te
herschalen naar exact de Sharpe van de baseline: hetzelfde volatiliteitsniveau,
dezelfde afhankelijkheid, een verschil van nul per constructie. Dat is een
scherpere nul dan een simulatie uit een normale verdeling, want zij draagt de
eigenschappen die deze toets in de problemen kunnen brengen.

De power-controle is het getal waar H2 om draait. De pre-registratie stelde
VOOR de run analytisch vast dat deze opzet 0,08 niet kan zien; hier wordt dat
GEMETEN. Spreken de twee elkaar tegen, dan wint de meting -- dezelfde regel die
in H1 een FALSIFIED tegenhield.

Ref: Jobson & Korkie (1981); Memmel (2003); Lo (2002); Politis & Romano (1994)
voor de stationaire bootstrap; pre-registratie `3d3af28730a6c7f9da48d13139522a05`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from scipy import stats

from ..utils.failfast import DataContractError, require

__all__ = [
    "SharpeDifferenceResult",
    "SharpeTestControls",
    "jobson_korkie_memmel",
    "rescale_to_sharpe",
    "run_sharpe_controls",
    "stationary_bootstrap_indices",
]

#: Onder dit aantal gepaarde observaties is de asymptotiek van Memmel niet meer
#: te verdedigen; de toets weigert dan in plaats van een p-waarde te leveren die
#: nergens op slaat.
_MIN_OBS = 30

#: Ondergrens voor de asymptotische variantie, RELATIEF aan haar eigen leidende
#: term. Memmel eq. 5 is een verschil van termen van orde `sd_a^2 * sd_b^2 / n`.
#: Zijn de twee armen HETZELFDE spoor -- correlatie exact 1 en proportionele
#: momenten, zoals bij `a` tegen `c * a` -- dan vallen die termen per constructie
#: volledig tegen elkaar weg en is de echte variantie NUL. Wat overblijft is
#: afrondingsruis met een willekeurig teken, dus een toets `variance > 0` beslist
#: dan op afrondingsgeluk in plaats van op de data.
#:
#: GEMETEN (780 identieke-spoor-vergelijkingen, 60 seeds x 13 multipliers):
#: `variance > 0` weigerde 48,1% en liet 51,9% door. De doorgelaten gevallen
#: gaven een ONSCHULDIG oordeel (grootste |z| = 0,0000, kleinste p = 0,999995,
#: nooit significant), dus er is nooit een vals positief uit voortgekomen -- het
#: defect is dat dezelfde invoer twee verschillende gedragingen kreeg.
#:
#: De drempel ligt op 1e-12 maal de leidende term, ~4.500x de machineprecisie.
#: Gemeten scheiding: identiek spoor komt niet boven 1,5e-16 uit, terwijl het
#: minst gescheiden ECHTE paar (rho = 0,999999) op 8,8e-7 zit -- zes ordes
#: speling. De phase-10-vergelijkingen (rho 0,8 tot 0,99) liggen op 1e-2 tot 1e-3.
#:
#: Dit is GEEN onderzoeksparameter in de zin van R-2 en kost dus geen trial: de
#: drempel kan alleen een vergelijking WEIGEREN, nooit een gunstiger uitkomst
#: produceren. Een weigering is geen resultaat.
#:
#: WAAROM DEZE TAK ANDERS ANTWOORDT DAN DE LEDOIT-WOLF-TAK. Dezelfde ontaarding
#: is eerder gevonden en gerepareerd in `validation/inference.py`, RULING P47:
#: `sharpe_difference_test` vangt haar af met `_is_positive_multiple` en geeft de
#: GESLOTEN VORM terug (verschil 0, se 0, p = 1) in plaats van te weigeren. Die
#: reparatie is toen NIET op deze tweede implementatie toegepast; dit is dat
#: gemis, en niets meer. De twee antwoorden blijven met opzet verschillend:
#: Ledoit-Wolf kent het antwoord in gesloten vorm en schrijft het op, terwijl
#: Memmel eq. 5 hier een 0/0 rekent en de aanroeper hoort te dwingen zelf te
#: beslissen wat "hetzelfde spoor" in ZIJN context betekent. Dat is precies wat
#: `phase10_decision_frequency.py::_delta_block` doet: het zet `test_was_called`
#: op False met een reden, in plaats van een p-waarde te rapporteren voor een
#: vergelijking die nooit een vergelijking was.
_DEGENERATE_VARIANCE_REL_TOL = 1e-12


@dataclass(frozen=True)
class SharpeDifferenceResult:
    """Het gepaarde Sharpe-verschil met alles wat het beoordeelbaar maakt."""

    sharpe_a: float
    sharpe_b: float
    #: `sharpe_a - sharpe_b`, geannualiseerd.
    difference: float
    z_statistic: float
    p_value: float
    standard_error: float
    correlation: float
    n_obs: int
    alternative: str
    significant: bool

    def as_record(self) -> dict[str, Any]:
        return {
            "sharpe_a": self.sharpe_a, "sharpe_b": self.sharpe_b,
            "difference": self.difference, "z_statistic": self.z_statistic,
            "p_value": self.p_value, "standard_error": self.standard_error,
            "correlation": self.correlation, "n_obs": self.n_obs,
            "alternative": self.alternative, "significant": self.significant,
        }


def _moments(x: np.ndarray) -> tuple[float, float]:
    return float(np.mean(x)), float(np.std(x, ddof=1))


def jobson_korkie_memmel(
    a: np.ndarray,
    b: np.ndarray,
    *,
    bars_per_year: float,
    alpha: float,
    alternative: Literal["two-sided", "a-better"] = "two-sided",
) -> SharpeDifferenceResult:
    """Toets ``SR_a = SR_b`` op gepaarde returns. Memmel (2003).

    De statistiek staat op de PER-BAR Sharpes -- de annualisatie is een
    constante factor die in teller en noemer wegvalt en dus geen invloed heeft
    op `z`. Gerapporteerd worden wel de geannualiseerde Sharpes, want dat is de
    eenheid waarin de pre-registratie haar effect uitdrukt.

    `alternative="a-better"` is eenzijdig en is wat een PROMOTIE nodig heeft:
    een tweezijdige verwerping zegt dat de twee verschillen, niet dat de
    uitdager wint. Dezelfde regel als in `judge_challenger` voor H1.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    require(
        a.shape == b.shape,
        "Een gepaarde toets op reeksen van verschillende lengte. Zonder "
        "koppeling per bar is de correlatie -- en dus de standaardfout -- "
        "betekenisloos.",
        DataContractError, n_a=int(a.size), n_b=int(b.size),
    )
    finite = np.isfinite(a) & np.isfinite(b)
    a, b = a[finite], b[finite]
    require(
        a.size >= _MIN_OBS,
        "Te weinig gepaarde observaties voor de asymptotiek van Memmel.",
        DataContractError, n_obs=int(a.size), required=_MIN_OBS,
    )
    mu_a, sd_a = _moments(a)
    mu_b, sd_b = _moments(b)
    require(
        sd_a > 0.0 and sd_b > 0.0,
        "Een arm zonder variatie. De Sharpe is daar niet gedefinieerd en het "
        "verschil dus evenmin.",
        DataContractError, sd_a=sd_a, sd_b=sd_b,
    )
    n = int(a.size)
    covariance = float(np.cov(a, b, ddof=1)[0, 1])
    correlation = covariance / (sd_a * sd_b)

    # Memmel (2003), eq. 5: de asymptotische variantie van
    # theta = sd_b * mu_a - sd_a * mu_b.
    theta = sd_b * mu_a - sd_a * mu_b
    variance = (
        2.0 * sd_a**2 * sd_b**2
        - 2.0 * sd_a * sd_b * covariance
        + 0.5 * mu_a**2 * sd_b**2
        + 0.5 * mu_b**2 * sd_a**2
        - (mu_a * mu_b / (sd_a * sd_b)) * covariance**2
    ) / n
    leading_term = 2.0 * sd_a**2 * sd_b**2 / n
    require(
        variance > _DEGENERATE_VARIANCE_REL_TOL * leading_term,
        "De asymptotische variantie van het Sharpe-verschil is niet positief "
        "op de schaal van haar eigen leidende term. Dat gebeurt wanneer de twee "
        "armen HETZELFDE spoor zijn -- correlatie exact 1 en proportionele "
        "momenten, bijvoorbeeld een reeks tegen een veelvoud van zichzelf. Een "
        "uniforme factor deelt weg tegen de noemer van de Sharpe, dus er is geen "
        "verschil om te toetsen.",
        DataContractError, variance=variance, correlation=correlation,
        leading_term=leading_term,
        relative_variance=variance / leading_term if leading_term > 0.0 else float("nan"),
    )
    standard_error = math.sqrt(variance)
    z = theta / standard_error
    if alternative == "two-sided":
        p_value = float(2.0 * stats.norm.sf(abs(z)))
    else:
        p_value = float(stats.norm.sf(z))
    scale = math.sqrt(bars_per_year)
    return SharpeDifferenceResult(
        sharpe_a=mu_a / sd_a * scale, sharpe_b=mu_b / sd_b * scale,
        difference=(mu_a / sd_a - mu_b / sd_b) * scale,
        z_statistic=float(z), p_value=p_value,
        standard_error=standard_error, correlation=float(correlation),
        n_obs=n, alternative=alternative, significant=bool(p_value < alpha),
    )


def rescale_to_sharpe(
    x: np.ndarray, target_sharpe: float, *, bars_per_year: float,
) -> np.ndarray:
    """Verschuif `x` zodat zijn geannualiseerde Sharpe exact `target` wordt.

    Alleen het GEMIDDELDE beweegt: de standaardafwijking, de autocorrelatie van
    de afwijkingen en de staarten blijven wat zij waren. Dat maakt dit de juiste
    constructie voor een nul- en een alternatiefpaar -- het verschil tussen de
    twee is dan precies het Sharpe-verschil en niet ook nog een ander regime.
    """
    x = np.asarray(x, dtype=np.float64)
    sd = float(np.std(x, ddof=1))
    require(
        sd > 0.0,
        "Herschalen van een reeks zonder variatie.",
        DataContractError, n=int(x.size),
    )
    return x - float(np.mean(x)) + target_sharpe / math.sqrt(bars_per_year) * sd


def stationary_bootstrap_indices(
    n: int, mean_block_length: float, rng: np.random.Generator,
) -> np.ndarray:
    """Indexen van een stationaire bootstrap (Politis & Romano 1994).

    Blokken van geometrisch verdeelde lengte, circulair doorlopend. De
    blokstructuur is niet decoratief: deze returnreeksen zijn autogecorreleerd,
    en een i.i.d.-bootstrap zou die afhankelijkheid vernietigen en de toets
    kunstmatig precies laten lijken.
    """
    require(
        mean_block_length >= 1.0,
        "Een gemiddelde bloklengte onder 1 bar is geen blokbootstrap.",
        DataContractError, mean_block_length=mean_block_length,
    )
    p = 1.0 / mean_block_length
    out = np.empty(n, dtype=np.int64)
    current = int(rng.integers(0, n))
    for t in range(n):
        if t > 0:
            current = (int(rng.integers(0, n)) if rng.random() < p
                       else (current + 1) % n)
        out[t] = current
    return out


@dataclass(frozen=True)
class SharpeTestControls:
    """Wat de toets doet op een gemeten nul en op een gemeten alternatief."""

    n_replicates: int
    alpha: float
    #: Verwerpingsratio wanneer de twee Sharpes per constructie GELIJK zijn.
    size_rejection_rate: float
    #: Verwerpingsratio bij het VERWACHTE effect uit de pre-registratie.
    power_at_expected_effect: float
    #: Verwerpingsratio bij het minimaal detecteerbare effect uit de
    #: power-analyse. Hoort rond `target_power` te liggen; doet hij dat niet,
    #: dan klopt de analytische power-analyse niet op deze data.
    power_at_mde: float
    expected_effect: float
    minimum_detectable_effect: float
    target_power: float
    #: De toets houdt zijn nominale niveau. Zonder dit is elke significante
    #: uitslag verdacht.
    size_passed: bool
    #: De toets ziet het effect waar de hypothese over gaat. Is dit False, dan
    #: draagt een niet-significante uitslag GEEN falsificatie.
    power_passed: bool

    def as_record(self) -> dict[str, Any]:
        return {
            "n_replicates": self.n_replicates, "alpha": self.alpha,
            "size_rejection_rate": self.size_rejection_rate,
            "power_at_expected_effect": self.power_at_expected_effect,
            "power_at_mde": self.power_at_mde,
            "expected_effect": self.expected_effect,
            "minimum_detectable_effect": self.minimum_detectable_effect,
            "target_power": self.target_power,
            "size_passed": self.size_passed, "power_passed": self.power_passed,
        }


def run_sharpe_controls(
    baseline: np.ndarray,
    challenger: np.ndarray,
    *,
    bars_per_year: float,
    alpha: float,
    expected_effect: float,
    minimum_detectable_effect: float,
    target_power: float,
    n_replicates: int,
    mean_block_length: float,
    seed: int,
    size_tolerance: float = 2.0,
) -> SharpeTestControls:
    """Size en power van deze toets, gemeten op deze twee reeksen.

    Het paar wordt GEZAMENLIJK gebootstrapt, zodat de correlatie tussen de armen
    -- de grootheid die de standaardfout bepaalt -- in elke replicatie bewaard
    blijft. Losse bootstraps zouden die correlatie naar nul brengen en de toets
    veel conservatiever laten lijken dan hij hier is.

    `size_tolerance` is de factor waarmee de gemeten verwerpingsratio `alpha`
    mag overschrijden voordat de size als gefaald geldt. Twee is ruim, en dat is
    opzettelijk: deze controle hoort een toets af te vangen die zijn niveau
    GROF mist, niet een die er met bootstrapruis naast zit.
    """
    baseline = np.asarray(baseline, dtype=np.float64)
    challenger = np.asarray(challenger, dtype=np.float64)
    require(
        baseline.shape == challenger.shape,
        "De controles vragen om een gepaarde opzet.",
        DataContractError, n_a=int(baseline.size), n_b=int(challenger.size),
    )
    scale = math.sqrt(bars_per_year)
    sharpe_baseline = (float(np.mean(baseline))
                       / float(np.std(baseline, ddof=1)) * scale)
    null_partner = rescale_to_sharpe(
        challenger, sharpe_baseline, bars_per_year=bars_per_year)
    expected_partner = rescale_to_sharpe(
        challenger, sharpe_baseline + expected_effect,
        bars_per_year=bars_per_year)
    mde_partner = rescale_to_sharpe(
        challenger, sharpe_baseline + minimum_detectable_effect,
        bars_per_year=bars_per_year)

    rng = np.random.default_rng(seed)
    counts = {"size": 0, "expected": 0, "mde": 0}
    for _ in range(n_replicates):
        idx = stationary_bootstrap_indices(baseline.size, mean_block_length, rng)
        left = baseline[idx]
        for name, partner in (("size", null_partner),
                              ("expected", expected_partner),
                              ("mde", mde_partner)):
            result = jobson_korkie_memmel(
                partner[idx], left, bars_per_year=bars_per_year, alpha=alpha,
                alternative="a-better")
            counts[name] += int(result.significant)

    size_rate = counts["size"] / n_replicates
    return SharpeTestControls(
        n_replicates=n_replicates, alpha=alpha,
        size_rejection_rate=size_rate,
        power_at_expected_effect=counts["expected"] / n_replicates,
        power_at_mde=counts["mde"] / n_replicates,
        expected_effect=expected_effect,
        minimum_detectable_effect=minimum_detectable_effect,
        target_power=target_power,
        size_passed=size_rate <= size_tolerance * alpha,
        power_passed=counts["expected"] / n_replicates >= target_power,
    )
