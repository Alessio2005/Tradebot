"""Het trial-budget -- de rem op AD-24.

Bij `M_new = 25` kost elke VERDUBBELING van `M` ongeveer 0,09 tot 0,13 Sharpe
aan drempelhoogte, gemeten met dit fase's eigen instrumenten -- niet aan de
0,13-0,17 die de fasetekst noemde (ruling P21b hieronder legt uit waarom die
claim zelf onjuist was).

DE WISSELKOERS, OP HAAR JUISTE VENSTER (ruling P21a)
=====================================================
De fasetekst reproduceerde de tabel M -> DSR-eis op `n_obs = 1615`: de legacy
purged-walk-forward OOS-mask. `docs/MEASUREMENT_CONTRACT.md` §2.5 scoopt dat
venster uitdrukkelijk tot het LEZEN van legacy-artefacten -- "geen drempel die
een nieuwe meting mag kiezen" -- en ruling T1-B legt het ene venster van dit
programma vast op W_FULL (`n_obs = 1743`, `t_years = 4,7753`). Beide kolommen
staan hieronder, W_FULL primair, zodat een lezer die een oud artefact naast
deze tabel legt geen tegenspraak ziet die er niet is.

Gemeten met `tradebot.backtest.metrics.deflated_sharpe`, `sr_variance =
1/n_obs`, `skew = 0`, `kurtosis = 3`, `bars_per_year = 365`, drempel bij
DSR = 0,95 (reproductie: `tests/unit/test_ledger_reset.py`, dezelfde
bisectie):

    M                              25    50    100   250   500   2425  2776
    eis, n_obs=1743 (W_FULL)      1,67  1,80  1,91  2,05  2,15  2,36  2,37
    eis, n_obs=1615 (legacy mask) 1,73  1,87  1,99  2,13  2,24  2,45  2,47

Dat is de wisselkoers waarin een onderzoeksplan zich hoort uit te drukken.
Deze module maakt die koers afdwingbaar in plaats van adviserend.

WAAROM "0,09 TOT 0,13" EN NIET DE "0,13-0,17" UIT DE FASETEKST (ruling P21b)
=============================================================================
De claim in de fasetekst is fout, en fout in de flatterende richting: zij
maakt elke verdubbeling duurder lijken dan hij is. Gemeten over elke
verdubbeling van M=25 tot M=1000 (25->50, 50->100, 100->200, 200->400,
250->500, 400->800, 500->1000): de kost ligt tussen **0,097 en 0,133** op
`n_obs=1615`, tussen **0,093 en 0,128** op W_FULL. De eigen tabel van de
fasetekst laat dit al zien: haar gedrukte rij 1,74 -> 1,87 is een sprong van
0,13 -- de TOP van de werkelijke bandbreedte, niet de bodem van de beweerde.
Deze module citeert daarom het gemeten bereik, "ongeveer 0,09 tot 0,13", op
precies de plek waar de fasetekst een derde, ongemeten getal zou hebben
toegevoegd: het argument blijft overeind (trials zijn schaars, want elke
verdubbeling verhoogt de eigen lat), alleen de digits zijn gecorrigeerd.

R4, DE OVERERVINGSREGEL, BECIJFERD
===================================
AD-24-protocol R4: "Wie een oude fit hergebruikt, erft zijn trials." Het
CPCV-ensemble draagt 2.400 Optuna-trials (200 x 6 symbolen x 2 zijden). Wie
zo'n fit hergebruikt in plaats van opnieuw te fitten, erft die 2.400 trials en
geeft ze mee aan `assert_within_budget` als `planned` -- niet als nul, alsof
hergebruik geen kosten heeft. Dat tilt `M` van 25 naar 2.425 en de DSR-eis van
1,67 naar 2,36 (W_FULL). Deze module bouwt geen detectie van hergebruik: de
aanroeper die een oude fit hergebruikt, is degene die weet dat hij dat doet,
en is dus degene die de erfenis meegeeft.

DE REKENREGEL VOOR BESLISBOMEN
===============================
Een campagne met een voorwaardelijk terugvalpad kost het aantal takken dat op
de data wordt doorlopen, niet 1. "Als k=3 de bezettingspoort niet haalt, meet
dan k=2" is twee specificaties op dezelfde data. Boek ze beide, vooraf.

DE FILENOTFOUNDERROR VAN EEN ONLEESBARE LEDGER
================================================
`active_trial_count` roept `HypothesisLedger(ledger_path)` aan, en die raist
een kale `FileNotFoundError` -- geen `DataContractError` -- wanneer
`ledger_path` niet bestaat. Deze module vangt die niet op en wikkelt hem niet
in. Dat is een bewuste keuze, geen omissie: `FileNotFoundError` is hier al de
gevestigde conventie voor een ontbrekend verplicht bestand (zo ook
`HypothesisLedger.__init__` zelf, `live/model_signal.py`,
`train/checkpoints.py`), met een boodschap die al zegt wat ontbreekt en
waarom er geen automatische reconstructie is. Een tweede vertaling naar
`DataContractError` zou die boodschap dupliceren zonder informatie toe te
voegen, en zou twee echt verschillende faalmodi -- "budget overschreden" en
"ledger onbereikbaar" -- onder een en dezelfde exceptie verbergen.
"""
from __future__ import annotations

from pathlib import Path

from ..utils.failfast import DataContractError, require
from .hypothesis_ledger import DEFAULT_LEDGER_PATH
from .ledger_reset import active_trial_count

__all__ = ["assert_within_budget", "remaining"]


def assert_within_budget(planned: int, *, reset_path: Path) -> None:
    """Crash wanneer een gepland aantal trials het bevroren budget overschrijdt."""
    budget = active_trial_count(
        reset_path=reset_path,
        ledger_path=DEFAULT_LEDGER_PATH,
    ).total
    require(
        planned <= budget,
        "Het geplande aantal trials overschrijdt het bevroren budget. Elke "
        "verdubbeling van M kost ~0,09 tot 0,13 Sharpe aan drempelhoogte; een "
        "plan dat over het budget gaat, verhoogt zijn eigen lat. Tel ook elke "
        "tak van een voorwaardelijk pad mee, en erf de trials van een "
        "hergebruikte fit (R4). Herzie het plan, of leg een nieuw budget vast "
        "met een eigen grondslag.",
        DataContractError, planned=planned, budget=budget,
    )


def remaining(*, reset_path: Path, ledger_path: Path, booked: int) -> int:
    """Wat er van het budget over is na `booked` geboekte trials."""
    budget = active_trial_count(
        reset_path=reset_path, ledger_path=ledger_path
    ).total
    return budget - int(booked)
