# tests/lookahead/test_unbalanced_panel_causality.py
"""R-1 op de breedtepoort — en de valkuil is niet de poort maar de KOLOMkeuze.

`normalise_weights` is rijgewijs: de uitvoer van bar t hangt uitsluitend af van
de waarden op bar t. Truncatie-invariantie is daarmee een gevolg van de
constructie. Wat NIET rijgewijs is, en wat een onevenwichtig paneel juist
uitnodigt, is het kiezen van de KOLOMMEN: "neem de symbolen die we hebben" is een
uitspraak over het einde van de steekproef, en dat einde is de toekomst.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradebot.portfolio.weights import normalise_weights

IDX = pd.date_range("2021-01-01", periods=300, freq="D", tz="UTC")


def _ragged() -> pd.DataFrame:
    """Een paneel dat aan BEIDE kanten rafelt: DOT begint laat en verdwijnt weer.

    Late instap is het geval dat deze stap wil kunnen dragen; het verdwijnen aan
    het eind is er bewust bij gezet, omdat juist dat de kolomkeuze van de
    negatieve controle hieronder laat afhangen van waar de steekproef ophoudt.
    """
    rng = np.random.default_rng(11)
    frame = pd.DataFrame(
        {"BTC": rng.uniform(-1, 1, 300),
         "LINK": rng.uniform(-1, 1, 300),
         "DOT": rng.uniform(-1, 1, 300)},
        index=IDX,
    )
    frame.loc[IDX[:40], "LINK"] = np.nan
    frame.loc[IDX[:90], "DOT"] = np.nan
    frame.loc[IDX[280:], "DOT"] = np.nan
    return frame


def test_truncation_cannot_change_an_earlier_bar() -> None:
    raw = _ragged()
    for m in (1, 2, 3):
        base = normalise_weights(raw, min_symbols_per_bar=m)
        for cut in (50, 100, 200, 279):
            pd.testing.assert_frame_equal(
                normalise_weights(raw.iloc[:cut], min_symbols_per_bar=m),
                base.iloc[:cut], check_freq=False,
            )


def test_a_perturbation_at_bar_t_cannot_move_bars_before_t() -> None:
    raw = _ragged()
    perturbed = raw.copy()
    perturbed.iloc[171, 0] = 99.0
    for m in (1, 2, 3):
        pd.testing.assert_frame_equal(
            normalise_weights(perturbed, min_symbols_per_bar=m).iloc[:171],
            normalise_weights(raw, min_symbols_per_bar=m).iloc[:171],
            check_freq=False,
        )


def test_a_bar_is_unaffected_by_whether_a_symbol_returns_later() -> None:
    """Afwezigheid later mag de samenstelling nu niet raken.

    Dit is de eigenschap waar het onevenwichtige paneel op staat of valt: bar 100
    kent DOT en moet dat blijven kennen, ook wanneer DOT op bar 280 verdwijnt.
    """
    raw = _ragged()
    never_delisted = raw.copy()
    never_delisted.loc[IDX[280:], "DOT"] = 0.5
    pd.testing.assert_frame_equal(
        normalise_weights(raw, min_symbols_per_bar=2).iloc[:280],
        normalise_weights(never_delisted, min_symbols_per_bar=2).iloc[:280],
        check_freq=False,
    )


def _gate_on_the_last_bar(raw: pd.DataFrame, min_symbols_per_bar: int) -> pd.DataFrame:
    """De valkuil, expliciet: kies de kolommen die op de LAATSTE bar bestaan.

    Dit is "neem de symbolen die we vandaag hebben", en het is verleidelijk omdat
    het per bar nog steeds netjes normaliseert — de lookahead zit niet in de
    rekenregel maar in de KOLOMverzameling, die van het einde van de steekproef
    afhangt. Alleen in productie zou dit een lookahead zijn; hier is het de
    negatieve controle.
    """
    keep = raw.notna().iloc[-1]
    sub = raw.loc[:, keep]
    n_available = sub.notna().sum(axis=1)
    gated = sub.mul(np.where(n_available >= min_symbols_per_bar, 1.0, 0.0), axis=0)
    gross = np.nansum(np.abs(gated.to_numpy()), axis=1)
    scale = np.where(gross > 0.0, 1.0 / np.where(gross > 0.0, gross, 1.0), 0.0)
    out = pd.DataFrame(gated.to_numpy() * scale[:, None],
                       index=sub.index, columns=sub.columns)
    return out.reindex(columns=raw.columns)


def test_the_truncation_guard_goes_red_on_last_bar_column_selection() -> None:
    """De negatieve controle op de test hierboven.

    Zonder dit bewijs toetst `test_truncation_cannot_change_an_earlier_bar` niet
    aantoonbaar iets: een vergelijking die altijd slaagt, bewijst niet dat de
    operatie causaal is. DOT verdwijnt op bar 280, dus op de volle steekproef
    valt DOT uit de kolomkeuze en op elke afknotting vóór 280 niet — en dan
    verschilt de samenstelling van bars die allebei al voorbij waren.
    """
    raw = _ragged()
    base = _gate_on_the_last_bar(raw, 2)
    for cut in (100, 200, 279):
        with pytest.raises(AssertionError):
            pd.testing.assert_frame_equal(
                _gate_on_the_last_bar(raw.iloc[:cut], 2),
                base.iloc[:cut], check_freq=False,
            )
