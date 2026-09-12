# tests/unit/test_layer_transitions.py
"""Een verschil zonder interval is geen bevinding.

Revisie 1 gebruikte hier een gepaarde t-toets op RENDEMENTSverschillen. Dat is
de verkeerde toets: de sporen verschillen in VOLATILITEIT (L7 herschaalt naar
het vol-target, de haltketen zet 89 % van de bars vlak), en een t-toets op
rendementsverschillen negeert dat. De vraag is een SHARPE-verschil en die vraagt
Ledoit-Wolf (2008).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from tradebot.validation.inference import block_bootstrap_ci, sharpe_difference_test

IDX = pd.date_range("2021-01-01", periods=1615, freq="D", tz="UTC")


def test_two_identical_tracks_show_no_difference() -> None:
    rng = np.random.default_rng(1)
    track = pd.Series(rng.normal(0.0002, 0.02, 1615), index=IDX)
    result = sharpe_difference_test(track, track.copy(), bars_per_year=365)
    assert abs(result.delta_sharpe) < 1e-12
    assert result.p_value > 0.99


def test_a_real_difference_is_detected_with_an_interval() -> None:
    rng = np.random.default_rng(2)
    # RULING P46. De brief zette hier mu_a = 0.0010. Dat is een WARE
    # geannualiseerde delta van +0,955 tegen een SE van 0,632: een effect van
    # 1,5 sigma. Een tekenassertie op 1,5 sigma toetst de TREKKING en niet de
    # eigenschap -- precies wat de openingsdocstring van dit bestand verwerpt --
    # en viel bij default_rng(2) dan ook om (steekproefgemiddelde van `a` 2,08
    # sigma onder de trekking, delta_sharpe = -0,0169). De SEED BLIJFT 2: seeds
    # afzoeken tot het teken uitkomt is de aanpassing die R-2 en R-10 verbieden;
    # een verkeerd gekozen effectgrootte optrekken bij een VASTE seed kan dat
    # niet zijn. Bij 0.0040 is de ware delta +3,821 tegen dezelfde SE, dus 6,1
    # sigma, en is het teken een eigenschap van de constructie.
    a = pd.Series(rng.normal(0.0040, 0.02, 1615), index=IDX)
    b = pd.Series(rng.normal(0.0000, 0.02, 1615), index=IDX)
    result = sharpe_difference_test(a, b, bars_per_year=365)
    assert result.delta_sharpe > 0.0
    # "Wordt gedetecteerd" betekent: het interval sluit nul UIT, en niets
    # zwakkers. Een punt met het goede teken en een interval om nul heen is
    # onder R-8 geen bevinding.
    assert result.ci_low > 0.0
    assert result.ci_low < result.delta_sharpe < result.ci_high


def test_the_test_accounts_for_unequal_volatility() -> None:
    """De reden waarom een gepaarde t op rendementsverschillen hier fout is:
    twee sporen met hetzelfde gemiddelde en verschillende vol hebben een
    verschillende Sharpe, en die vraag moet de toets kunnen zien."""
    rng = np.random.default_rng(3)
    a = pd.Series(rng.normal(0.0005, 0.010, 1615), index=IDX)
    b = pd.Series(a.to_numpy() * 2.0, index=IDX)  # zelfde signaal, dubbele vol
    result = sharpe_difference_test(a, b, bars_per_year=365)
    assert abs(result.delta_sharpe) < 1e-9  # Sharpe is scale-invariant
    assert result.p_value > 0.5


def test_a_halted_track_is_compared_on_common_active_bars_only() -> None:
    """De haltketen zet 1.559 van de 1.743 bars vlak. Nul-rendementen
    meerekenen levert een SE die de resterende 184 actieve bars niet
    weerspiegelt, en dus een t die te groot is."""
    rng = np.random.default_rng(4)
    a = pd.Series(rng.normal(0.0005, 0.02, 1615), index=IDX)
    b = a.copy()
    b.iloc[184:] = 0.0
    result = sharpe_difference_test(
        a, b, bars_per_year=365, align="common_active",
    )
    assert result.n_effective <= 184


def test_the_bootstrap_ci_is_reported_alongside_the_analytic_one() -> None:
    """Twee onafhankelijke routes naar hetzelfde interval. Wijken ze sterk af,
    dan is dat de bevinding en niet een detail."""
    rng = np.random.default_rng(5)
    a = pd.Series(rng.normal(0.0008, 0.02, 1615), index=IDX)
    ci = block_bootstrap_ci(a, statistic="sharpe", block_length=20,
                            n_boot=2000, bars_per_year=365, seed=11)
    assert ci.low < ci.point < ci.high
