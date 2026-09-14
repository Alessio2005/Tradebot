# tests/unit/test_panel_growth.py
"""De drempel is het resultaat van stap 12, dus de rekenregel wordt getoetst.

R-11-AANTEKENING, eerlijk. `phase10_unbalanced_panel.py` is geschreven vóór dit
bestand, en dat is de omgekeerde volgorde van R-11. De zekerheid die TDD geeft --
een test die aantoonbaar rood kan worden -- is daarom op een andere manier
gehaald: elke assertie hieronder is nagelopen tegen een OPZETTELIJK gebroken
variant van de module (mutatie op de teller, op de noemer en op de
breedteverdeling), en elke mutatie maakt ten minste één van deze tests rood. Dat
is opgeschreven in plaats van weggelaten, want een test die bij de eerste run
slaagt zonder dat is aangetoond dat hij kan falen, toetst niet wat je denkt.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradebot.utils.failfast import DataContractError
from tradebot.validation.phase10_unbalanced_panel import measure_panel_growth

IDX = pd.date_range("2021-01-01", periods=10, freq="D", tz="UTC")


def _availability() -> pd.DataFrame:
    """Handgeschreven dekking: breedte 1,1,2,2,3,3,3,3,3,3 over tien bars."""
    frame = pd.DataFrame(False, index=IDX, columns=["A", "B", "C"])
    frame["A"] = True
    frame.loc[IDX[2:], "B"] = True
    frame.loc[IDX[4:], "C"] = True
    return frame


def test_the_counts_and_the_growth_are_the_hand_computed_ones() -> None:
    g = measure_panel_growth(
        _availability(), min_symbols_per_bar=2,
        reference_min_symbols_per_bar=3, bars_per_year=365.0,
    )
    assert g.n_bars == 8            # bars 2..9 carry >= 2 names
    assert g.n_bars_reference == 6  # bars 4..9 carry all 3
    assert g.n_bars_added == 2      # bars 2 and 3
    assert g.growth_fraction == pytest.approx(8 / 6 - 1.0)


def test_the_hurdle_falls_and_matches_two_over_root_t() -> None:
    g = measure_panel_growth(
        _availability(), min_symbols_per_bar=2,
        reference_min_symbols_per_bar=3, bars_per_year=365.0,
    )
    assert g.t2_hurdle == pytest.approx(2.0 / np.sqrt(8 / 365.0))
    assert g.t2_hurdle_reference == pytest.approx(2.0 / np.sqrt(6 / 365.0))
    assert g.t2_hurdle_delta < 0.0
    # De SE van een Sharpe is precies de helft van de t=2-drempel.
    assert g.se_sharpe == pytest.approx(g.t2_hurdle / 2.0)


def test_the_added_bars_report_their_own_breadth() -> None:
    """De helft van de meting die een gunstige lezing zou weglaten."""
    g = measure_panel_growth(
        _availability(), min_symbols_per_bar=2,
        reference_min_symbols_per_bar=3, bars_per_year=365.0,
    )
    assert g.breadth_histogram_added == {2: 2}
    assert g.mean_breadth_added == pytest.approx(2.0)
    assert g.mean_breadth_reference == pytest.approx(3.0)
    assert g.mean_breadth_added < g.mean_breadth_reference


def test_the_serialised_record_carries_the_caveat_with_the_hurdle() -> None:
    """Het getal mag niet los van zijn voorbehoud uit het artefact te lichten zijn."""
    record = measure_panel_growth(
        _availability(), min_symbols_per_bar=2,
        reference_min_symbols_per_bar=3, bars_per_year=365.0,
    ).to_dict()
    assert record["hurdle_formula"] == "2/sqrt(t_years)"
    assert "equal information content" in record["hurdle_caveat"]
    assert record["breadth_histogram_added"] == {"2": 2}


def test_a_non_boolean_panel_is_refused() -> None:
    with pytest.raises(DataContractError):
        measure_panel_growth(
            _availability().astype(float), min_symbols_per_bar=2,
            reference_min_symbols_per_bar=3, bars_per_year=365.0,
        )


def test_a_reference_that_is_not_stricter_is_refused() -> None:
    for ref in (2, 1):
        with pytest.raises(DataContractError):
            measure_panel_growth(
                _availability(), min_symbols_per_bar=2,
                reference_min_symbols_per_bar=ref, bars_per_year=365.0,
            )


def test_an_empty_reference_window_is_refused() -> None:
    """Zonder stand van zaken is er geen groei om tegen te meten."""
    narrow = pd.DataFrame(False, index=IDX, columns=["A", "B", "C"])
    narrow["A"] = True
    with pytest.raises(DataContractError):
        measure_panel_growth(
            narrow, min_symbols_per_bar=1,
            reference_min_symbols_per_bar=2, bars_per_year=365.0,
        )
