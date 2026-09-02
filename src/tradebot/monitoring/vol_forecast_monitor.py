# src/tradebot/monitoring/vol_forecast_monitor.py
"""Doet de productie-vol-estimator live nog wat hij in de backtest deed?

Stage D-3. De productie-estimator is EWMA(0,94); H1 heeft hem niet verslagen
zien worden (`reports/GARCH_VS_EWMA_COMPETITION.md`). Deze monitor bewaakt niet
of hij GOED is -- dat is in H1 beslist -- maar of hij live nog hetzelfde doet.

WAAROM DE POORT RELATIEF IS EN NIET ABSOLUUT
=============================================
QLIKE is nul bij een perfecte forecast en onbegrensd daarboven. Er bestaat geen
natuurlijke absolute grens; elk getal dat hier zou staan, zou verzonnen zijn.
De poort vergelijkt daarom de LIVE QLIKE met de QLIKE die dezelfde estimator in
de backtest haalde: een verhouding met een betekenis, tegen een baseline die is
gemeten in plaats van gekozen.

De baseline is een ARGUMENT en heeft geen default. Een monitor die zijn eigen
referentie verzint, meet niets; wie hem aanroept moet zeggen waartegen.

WAAROM DE MINCER-ZARNOWITZ-TOETS ERNAAST STAAT
===============================================
QLIKE zegt hoe GROOT de fout is, MZ zegt of zij SYSTEMATISCH is. Een estimator
die structureel 30 % te laag voorspelt, kan een acceptabele QLIKE houden en
tegelijk elke positiegrootte te groot maken -- want de vol-targeting deelt door
precies dat getal. De twee samen vangen wat elk apart doorlaat, en dat is
dezelfde reden waarom §12.1 MDI en SFI allebei eist.

De rekenkern komt uit `validation/vol_metrics.py`, dezelfde functies waarmee H1
zijn oordeel velde. Een tweede implementatie hier zou betekenen dat live en
backtest op verschillende definities van QLIKE kunnen uitkomen -- precies de
klasse divergentie die `reports/phase7_divergence_map.md` in kaart brengt.

Ref: fase-opdracht Stage D-3; `conf/monitoring/default.yaml`;
`validation/vol_metrics.py`.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..schemas.config import MonitoringConfig, monitoring_config
from ..utils.failfast import DataContractError, require
from ..validation.vol_metrics import mincer_zarnowitz, qlike

__all__ = ["VolForecastMonitor", "VolForecastVerdict"]

QLIKE_DEGRADED = "qlike_degraded"
MZ_BIASED = "forecast_biased"


def _mz_verdict_is_bias(mz: Any, alpha_level: float) -> bool:
    """Is deze Mincer-Zarnowitz-uitkomst een ECHTE vertekening?

    Normaal is dat simpelweg `p < alpha`. Er is een randgeval waarin die vraag
    niet op de p-waarde te beantwoorden is, en het treedt juist op bij een
    forecast die perfect met de realisatie meebeweegt.

    GEMETEN, 2026-09-01. Bij nulresiduen gaat de HAC-covariantie naar nul en
    explodeert de Wald-statistiek; `mincer_zarnowitz` geeft dan p = 0 ongeacht
    de coefficienten. Dat treft twee tegengestelde gevallen tegelijk:

    * `forecast == rv` -> alpha = -8,1e-19, beta = 1,000000000000006, p = 6,8e-9.
      Volmaakt zuiver, en toch "vertekend". Een vals alarm dat het boek sluit.
    * `forecast == rv * 0,7` -> alpha ~ 0, beta = 1,4285714, p = 0,0. Perfecte
      FIT en maximale vertekening -- precies wat er wel gevangen moet worden.

    Beide hebben `r_squared == 1`. Een poort op de fitkwaliteit kan ze dus niet
    onderscheiden; een eerdere versie van deze functie deed dat wel en liet het
    tweede geval door. Waar het werkelijk om gaat is of de COEFFICIENTEN van
    (0, 1) afwijken. Zijn de residuen numeriek nul, dan is de regressie exact en
    is die afwijking rechtstreeks af te lezen -- er is geen toets nodig en de
    p-waarde is betekenisloos.

    `vol_metrics.py` wordt hiervoor NIET aangepast: die functie draagt het
    bevroren H1-oordeel, en haar gedrag veranderen zou dat onreproduceerbaar
    maken.
    """
    exact = (np.isfinite(mz.r_squared)
             and (1.0 - mz.r_squared) <= mz.n_obs * float(
                 np.finfo(np.float64).eps))
    if exact:
        tol = mz.n_obs * float(np.finfo(np.float64).eps)
        return not (abs(mz.alpha) <= tol and abs(mz.beta - 1.0) <= tol)
    return bool(mz.p_value < alpha_level)


@dataclass(frozen=True)
class VolForecastVerdict:
    """Het oordeel over het venster dat nu in de monitor zit."""

    n_obs: int
    live_qlike: float
    baseline_qlike: float
    qlike_ratio: float
    mz_alpha: float
    mz_beta: float
    mz_p_value: float
    breached: tuple[str, ...]
    #: `False` zolang er te weinig waarnemingen zijn. Dan is er GEEN oordeel --
    #: niet "in orde". Een monitor die zwijgt omdat hij niets weet, mag niet
    #: worden gelezen als een monitor die groen staat.
    is_conclusive: bool

    @property
    def is_breach(self) -> bool:
        return bool(self.breached)

    def as_record(self) -> dict[str, Any]:
        return {
            "n_obs": self.n_obs, "live_qlike": self.live_qlike,
            "baseline_qlike": self.baseline_qlike,
            "qlike_ratio": self.qlike_ratio,
            "mz_alpha": self.mz_alpha, "mz_beta": self.mz_beta,
            "mz_p_value": self.mz_p_value,
            "breached": list(self.breached),
            "is_conclusive": self.is_conclusive,
        }


@dataclass
class VolForecastMonitor:
    """Verzamelt (realisatie, forecast)-paren en velt er een oordeel over.

    `baseline_qlike` is de QLIKE die dezelfde estimator in de backtest haalde.
    Hij is verplicht; zie de moduledocstring.
    """

    baseline_qlike: float
    cfg: MonitoringConfig = field(default_factory=monitoring_config)
    #: Het venster waarover wordt geoordeeld. `maxlen` houdt het rollend, zodat
    #: een ontsporing van vandaag niet eeuwig wordt uitgemiddeld tegen een jaar
    #: schone historie.
    window: int = 1_000
    _rv: deque[float] = field(default_factory=deque, init=False)
    _forecast: deque[float] = field(default_factory=deque, init=False)

    def __post_init__(self) -> None:
        require(
            self.baseline_qlike > 0.0,
            "Een baseline-QLIKE van nul of lager bestaat niet: QLIKE is een "
            "Bregman-divergentie en strikt positief buiten de perfecte "
            "forecast. Een niet-positieve baseline maakt elke verhouding "
            "betekenisloos.",
            DataContractError, baseline_qlike=self.baseline_qlike,
        )
        self._rv = deque(maxlen=self.window)
        self._forecast = deque(maxlen=self.window)

    def record(self, realised_variance: float, forecast_variance: float) -> None:
        """Voeg één bar toe. Beide grootheden zijn VARIANTIES, geen vols.

        Dat onderscheid is niet cosmetisch: QLIKE op standaarddeviaties in
        plaats van varianties geeft een ander getal, en de baseline uit H1 is op
        varianties gemeten.
        """
        self._rv.append(float(realised_variance))
        self._forecast.append(float(forecast_variance))

    def verdict(self) -> VolForecastVerdict:
        """Oordeel over het huidige venster.

        Onder `vol_forecast_min_obs` is het oordeel NIET-CONCLUSIEF en zijn er
        per definitie geen breaches. Dat is geen groen licht; zie
        `is_conclusive`.
        """
        n = len(self._rv)
        if n < self.cfg.vol_forecast_min_obs:
            return VolForecastVerdict(
                n_obs=n, live_qlike=float("nan"),
                baseline_qlike=self.baseline_qlike, qlike_ratio=float("nan"),
                mz_alpha=float("nan"), mz_beta=float("nan"),
                mz_p_value=float("nan"), breached=(), is_conclusive=False)

        rv = np.fromiter(self._rv, dtype=np.float64, count=n)
        forecast = np.fromiter(self._forecast, dtype=np.float64, count=n)

        # `qlike` geeft NaN voor elk paar waar de forecast niet strikt positief
        # is. Tellen die NaN's te zwaar, dan is er geen oordeel te vellen -- en
        # dat is iets ANDERS dan een oordeel dat groen is. Zonder deze poort
        # kwam een volledig ontspoorde estimator (forecast overal nul of
        # negatief) er als schoon uit: `nanmean` van louter NaN is NaN,
        # `NaN > drempel` is False, `breached` bleef leeg en het oordeel ging
        # als CONCLUSIEF de deur uit. De klassedocstring van
        # `VolForecastVerdict.is_conclusive` verbiedt precies dat.
        terms = qlike(rv, forecast)
        n_usable = int(np.count_nonzero(np.isfinite(terms)))
        if n_usable < self.cfg.vol_forecast_min_obs:
            return VolForecastVerdict(
                n_obs=n, live_qlike=float("nan"),
                baseline_qlike=self.baseline_qlike, qlike_ratio=float("nan"),
                mz_alpha=float("nan"), mz_beta=float("nan"),
                mz_p_value=float("nan"), breached=(), is_conclusive=False)

        live = float(np.mean(terms[np.isfinite(terms)]))
        ratio = live / self.baseline_qlike
        mz = mincer_zarnowitz(rv, forecast, model="live")

        breached: list[str] = []
        # Expliciet op eindigheid toetsen. `NaN > drempel` is False, dus een
        # niet-eindige ratio zou anders als "niet overschreden" gelden.
        if not np.isfinite(ratio) or ratio > self.cfg.vol_qlike_degradation_ratio:
            breached.append(QLIKE_DEGRADED)
        if _mz_verdict_is_bias(mz, self.cfg.vol_mz_alpha):
            breached.append(MZ_BIASED)

        return VolForecastVerdict(
            n_obs=n, live_qlike=live, baseline_qlike=self.baseline_qlike,
            qlike_ratio=ratio, mz_alpha=mz.alpha, mz_beta=mz.beta,
            mz_p_value=mz.p_value, breached=tuple(breached),
            is_conclusive=True)
