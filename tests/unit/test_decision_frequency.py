# tests/unit/test_decision_frequency.py
"""Vasthouden is niet hetzelfde als resamplen.

De valkuil: `.resample('5D').first()` en dan forward-fillen. Dat kiest per
blok de eerste bar en dat is een lookahead zodra het blok wordt uitgelijnd op
iets anders dan de eerste beschikbare bar.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradebot.portfolio.decision_frequency import breakeven_cost_bps, hold_decision
from tradebot.utils.failfast import DataContractError

IDX = pd.date_range("2022-01-01", periods=100, freq="D", tz="UTC")


def _exposures() -> pd.DataFrame:
    rng = np.random.default_rng(6)
    return pd.DataFrame(
        {"A": rng.uniform(-1, 1, 100), "B": rng.uniform(-1, 1, 100)}, index=IDX,
    )


def test_k_equals_one_is_the_identity() -> None:
    exposures = _exposures()
    pd.testing.assert_frame_equal(hold_decision(exposures, k=1), exposures)


def test_k_equals_five_holds_for_five_bars() -> None:
    held = hold_decision(_exposures(), k=5)
    for start in range(0, 95, 5):
        block = held.iloc[start:start + 5]
        assert (block.nunique() == 1).all()


def test_the_held_value_is_the_value_of_the_first_bar_in_the_block() -> None:
    exposures = _exposures()
    held = hold_decision(exposures, k=5)
    for start in range(0, 95, 5):
        pd.testing.assert_series_equal(
            held.iloc[start], exposures.iloc[start], check_names=False,
        )


def test_turnover_falls_roughly_as_one_over_k() -> None:
    exposures = _exposures()

    def turnover(frame: pd.DataFrame) -> float:
        return float(frame.diff().abs().sum(axis=1).mean())

    base = turnover(exposures)
    assert turnover(hold_decision(exposures, k=5)) < base / 3.0


def test_k_must_be_positive_and_integral() -> None:
    for bad in (0, -1, 2.5):
        with pytest.raises(DataContractError):
            hold_decision(_exposures(), k=bad)


# ---------------------------------------------------------------------------
# breakeven_cost_bps -- stap 11.7
#
# De hypothese is een KOSTENhypothese en `eta` is ongekalibreerd (AD-2,
# `IMPACT_UNCALIBRATED`). Een conclusie "k=5 is beter" hangt dan aan een
# getal dat niet is gemeten. De conclusie die dat overleeft, is "k=5 is beter
# zodra de werkelijke kosten boven X bp liggen", en X is wat deze functie
# uitrekent.
# ---------------------------------------------------------------------------


def test_the_breakeven_cost_zeroes_the_net_difference() -> None:
    """De definierende eigenschap, en niet een gekopieerd getal.

    Bij de teruggegeven kosten is het netto Sharpe-verschil per constructie
    nul. Dat is te controleren zonder de formule te herhalen: vul de kosten
    terug in de linearisatie en kijk of er nul uitkomt.
    """
    delta_gross, turnover_delta, sigma, bpy = -0.2, 0.05, 0.01, 365.0
    bps = breakeven_cost_bps(
        delta_gross, turnover_delta,
        volatility_per_bar=sigma, bars_per_year=bpy,
    )
    cost = bps / 1e4
    delta_net = delta_gross + cost * turnover_delta * np.sqrt(bpy) / sigma
    assert abs(delta_net) < 1e-12


def test_a_hold_that_loses_gross_but_saves_turnover_breaks_even_above_zero() -> None:
    """Het geval waar de hypothese op hoopt: bruto slechter, omzet lager."""
    bps = breakeven_cost_bps(
        -0.2, 0.05, volatility_per_bar=0.01, bars_per_year=365.0,
    )
    assert bps > 0.0


def test_a_hold_that_wins_gross_needs_no_cost_to_justify_it() -> None:
    """Een negatief breakevenniveau is een UITKOMST en geen fout: het betekent
    dat het vasthouden al bij nul kosten wint, en dus dat de kostenaanname de
    conclusie niet draagt."""
    bps = breakeven_cost_bps(
        +0.2, 0.05, volatility_per_bar=0.01, bars_per_year=365.0,
    )
    assert bps < 0.0


def test_a_zero_turnover_difference_has_no_breakeven() -> None:
    """Zonder omzetverschil kan geen kostenniveau het teken omdraaien. Dat is
    een BEVINDING over de keten van de hypothese en mag niet als een getal
    worden weggeschreven."""
    with pytest.raises(DataContractError):
        breakeven_cost_bps(
            -0.2, 0.0, volatility_per_bar=0.01, bars_per_year=365.0,
        )


def test_the_breakeven_refuses_a_degenerate_denominator() -> None:
    for sigma, bpy in ((0.0, 365.0), (-0.01, 365.0), (0.01, 0.0)):
        with pytest.raises(DataContractError):
            breakeven_cost_bps(
                -0.2, 0.05, volatility_per_bar=sigma, bars_per_year=bpy,
            )
