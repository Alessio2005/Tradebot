"""Eén vasthoudoperatie, of twee? Fase 11, breedte en tijdschaal, stap 4.

Er bestaan twee implementaties van "een besluit k bars vasthouden":

* `alpha/momentum.py::CrossSectionalMomentum` met `rebalance_every_bars = k`
  houdt de SIGNAALexposures vast, met een kalender vanaf de eerste bar van het
  featurepaneel;
* `portfolio/decision_frequency.py::hold_decision` houdt de GEWICHTEN vast, met
  blokken vanaf de eerste bar van het paneel dat hij krijgt.

Deze tests leggen vast waar dat dezelfde operatie is en waar niet:

1. bij gelijkgewogen sizing, met gelijk anker, is het boek BIT-IDENTIEK:
   normaliseren per rij laat identieke rijen identiek;
2. bij risicopariteit is het boek aantoonbaar ANDERS: de sizing leest een
   volatiliteit die elke bar verandert. Deze test is tegelijk de negatieve
   controle op de eerste: een vergelijking die hier ook "gelijk" zou zeggen,
   meet niets;
3. met een verschoven anker verschillen zij al door de fase van de kalender.
   H-10.1 verankerde `hold_decision` op de eerste VERHANDELBARE bar
   (ordeningsbesluit 1), de unit telt vanaf het begin van het featurepaneel.

En een vierde, gemeten tijdens het schrijven van deze tests: met de
geconfigureerde lookback (60 bars plus skip 1, dus een burn-in van 61)
WEIGERT de unit elke k behalve 1 en 61. De kalender van de unit valt dan niet
samen met het einde van de burn-in, de bars daartussen dragen een feature maar
een exposure die via `ffill` uit een NaN komt, en de eigen validatie van de unit
slaat daarop aan. 61 is priem: de vasthoudoptie van de unit is met de
geconfigureerde lookback voor geen enkele k tussen 2 en 60 bruikbaar. De
equivalentietests hieronder gebruiken daarom lookback 59 (burn-in 60, deelbaar
door K = 10).

Het besluit (DI-36): `hold_decision` is de ene implementatie van "een besluit
vasthouden"; `rebalance_every_bars` blijft 1 en verdwijnt uit de unit zodra een
fase de baseline-unit om een andere reden opnieuw afleidt.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tradebot.alpha.momentum import CrossSectionalMomentum
from tradebot.portfolio.decision_frequency import hold_decision
from tradebot.portfolio.equal_weight import sized_by_equal_weight
from tradebot.portfolio.risk_parity import sized_by_risk_parity

K = 10
LOOKBACK = 59  # burn-in 60, deelbaar door K; de geconfigureerde 60 is dat niet
BURN_IN = LOOKBACK + 1
IDX = pd.date_range("2022-01-01", periods=300, freq="D", tz="UTC")
COLUMNS = ["A", "B", "C", "D"]


def _features(burn_in: int = BURN_IN) -> SimpleNamespace:
    rng = np.random.default_rng(40)
    values = pd.DataFrame(rng.uniform(-1, 1, (len(IDX), 4)), index=IDX, columns=COLUMNS)
    values.iloc[:burn_in] = np.nan
    return SimpleNamespace(values=values, data_hashes=("synthetic",))


def _vol() -> pd.DataFrame:
    rng = np.random.default_rng(41)
    return pd.DataFrame(rng.uniform(0.3, 1.2, (len(IDX), 4)), index=IDX, columns=COLUMNS)


def _exposures(every: int) -> pd.DataFrame:
    unit = CrossSectionalMomentum(lookback_bars=LOOKBACK, skip_bars=1, min_assets=3,
                                  rebalance_every_bars=every, signal_floor=-1.0,
                                  signal_cap=1.0)
    return unit.generate(_features()).exposures


def test_equal_weight_books_are_bit_identical_under_the_same_anchor() -> None:
    held_in_unit = sized_by_equal_weight(_exposures(K), gross_target=1.0)
    held_on_book = hold_decision(sized_by_equal_weight(_exposures(1), gross_target=1.0), k=K)
    pd.testing.assert_frame_equal(held_in_unit, held_on_book)


def test_risk_parity_books_differ_because_the_sizing_reads_a_moving_vol() -> None:
    vol = _vol()
    held_in_unit = sized_by_risk_parity(_exposures(K), vol, gross_target=1.0)
    held_on_book = hold_decision(sized_by_risk_parity(_exposures(1), vol, gross_target=1.0),
                                 k=K)
    assert float((held_in_unit - held_on_book).abs().to_numpy().max()) > 1e-3


def test_a_shifted_anchor_changes_the_book_even_under_equal_weight() -> None:
    offset = 3
    held_in_unit = sized_by_equal_weight(_exposures(K), gross_target=1.0).iloc[offset:]
    book = sized_by_equal_weight(_exposures(1), gross_target=1.0).iloc[offset:]
    held_on_book = hold_decision(book, k=K)
    assert not held_in_unit.equals(held_on_book)


def test_the_configured_unit_refuses_every_k_that_does_not_divide_its_burn_in() -> None:
    from tradebot.utils.failfast import DataContractError

    def unit(every: int) -> CrossSectionalMomentum:
        return CrossSectionalMomentum(lookback_bars=60, skip_bars=1, min_assets=3,
                                      rebalance_every_bars=every, signal_floor=-1.0,
                                      signal_cap=1.0)

    features = _features(burn_in=61)
    with pytest.raises(DataContractError):
        unit(K).generate(features)
    assert unit(61).generate(features).exposures.shape == features.values.shape
