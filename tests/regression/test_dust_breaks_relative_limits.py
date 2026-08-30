# tests/regression/test_dust_breaks_relative_limits.py
"""Stof in het boek mocht de relatieve limieten niet meer breken. DI-19.

DE MEETING
===========
`RiskEngine.decide` zette exposures onder `DUST_TOLERANCE` op nul NA de
limietketen en verifieerde daarna. De concentratielimiet lost echter een VAST
PUNT op waarin elke exposure meetelt in de gross:

    |w_i| <= c * sum_j |w_j|

Verdwijnt daarna een van die `w_j`, dan krimpt de gross en STIJGT het aandeel
van de grootste post. Gemeten op een boek uit de H2-campagne, 2025-10-11,
`hmm2-diag-gaussian`:

    exposure die verdween          8,3e-13
    gross waaruit hij verdween     9,1e-7
    concentratie voor de snap      0,40000000000000002
    concentratie na de snap        0,40000008838494777
    tolerantie in de verificatie   1e-9

De engine crashte daar terecht op -- er is geen pad waarlangs een limiet
stilzwijgend wordt afgerond -- maar de foutmelding wees de CLUSTERlimiet aan,
die er niets mee te maken had. Zes van de zeven H2-armen liepen; de zevende
crashte, en dat was geen eigenschap van dat model maar van deze volgorde.

De reparatie is twee regels: verifieer wat de keten heeft OPGELOST, snap daarna.
Deze test is de negatieve controle erop en draagt de exacte cijfers, zodat een
latere herordening niet stilletjes hetzelfde kan doen.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tradebot.risk.contract import MarketState, RiskState
from tradebot.risk.engine import DUST_TOLERANCE, RiskEngine
from tradebot.schemas.config import RiskConfig, load_config

RISK = load_config(ROOT / "conf/risk/default.yaml", RiskConfig)

#: Het boek van 2025-10-11 zoals de H2-arm het aanbood, cijfer voor cijfer.
#: SOLUSDT domineert; de andere vijf staan tussen 1e-5 en 1e-12 omdat het
#: filter daar `P(hoog) ~ 1` gaf.
FAILING_BOOK = {
    "BTCUSDT": -6.3519776086921642e-05,
    "ETHUSDT": +3.3439568770846173e-08,
    "SOLUSDT": +1.0,
    "AVAXUSDT": +2.6181744102260301e-12,
    "LINKUSDT": -2.3363504566720165e-06,
    "DOTUSDT": +0.0,
}
SIGMA = {
    "BTCUSDT": 0.4693, "ETHUSDT": 0.8481, "SOLUSDT": 1.0383,
    "AVAXUSDT": 1.6697, "LINKUSDT": 1.3168, "DOTUSDT": 1.6565,
}


def _decide(book):
    engine = RiskEngine(RISK)
    market = MarketState(
        asof_ts=pd.Timestamp("2025-10-11", tz="UTC"), sigma_hat=SIGMA,
        adv_usd={s: 5.0e8 for s in SIGMA}, cluster=dict(RISK.clusters))
    state = RiskState(equity=96_704.99809185612, high_water_mark=100_000.0,
                      day_start_equity=96_704.99809185612)
    return engine.decide(book, market, state)


def test_a_book_that_straddles_the_dust_tolerance_is_decided() -> None:
    """Het exacte boek dat de H2-campagne liet crashen."""
    decision = _decide(FAILING_BOOK)
    assert decision.config_hash == "1b60cb664fbf9a2a"


def test_the_returned_book_carries_no_dust() -> None:
    """De snap gebeurt nog steeds -- alleen na de verificatie in plaats van
    ervoor. Wat eruit komt bevat geen posities die niet kunnen bestaan."""
    permitted = _decide(FAILING_BOOK).permitted_exposure
    for symbol, exposure in permitted.items():
        assert exposure == 0.0 or abs(exposure) > DUST_TOLERANCE, symbol


def test_the_absolute_caps_still_hold_on_what_is_returned() -> None:
    """De snap kan alleen krimpen, dus gross, per-asset en concentratie kunnen
    alleen dalen. De netto exposure kan stijgen door een tegengestelde
    stofpost weg te halen, met ten hoogste `n * DUST_TOLERANCE` -- 6e-12 hier,
    ruim binnen de 1e-9 waarmee de netto-cap wordt geverifieerd."""
    permitted = _decide(FAILING_BOOK).permitted_exposure
    gross = sum(abs(w) for w in permitted.values())
    assert gross <= RISK.gross_cap + 1e-9
    assert abs(sum(permitted.values())) <= RISK.net_cap + 1e-9
    assert max(abs(w) for w in permitted.values()) <= RISK.max_position_pct + 1e-9


def test_dust_free_and_dusty_books_agree_on_the_real_positions() -> None:
    """De stofpost draagt 2,6e-12 van de exposure. Hem vooraf weglaten mag het
    boek dus niet meetbaar veranderen -- deed het dat wel, dan zou de snap een
    economische ingreep zijn geweest in plaats van een opruimactie."""
    dusty = _decide(FAILING_BOOK).permitted_exposure
    clean = _decide({s: (0.0 if abs(w) <= DUST_TOLERANCE else w)
                     for s, w in FAILING_BOOK.items()}).permitted_exposure
    for symbol in FAILING_BOOK:
        assert dusty[symbol] == pytest.approx(clean[symbol], abs=1e-11)
