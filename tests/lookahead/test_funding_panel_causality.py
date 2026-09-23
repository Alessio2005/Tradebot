# tests/lookahead/test_funding_panel_causality.py
"""De dagaggregatie van funding is causaal, en de toets kan rood worden (R-1).

Fase 11, stap 5.4. `data/funding_panel.py` is de enige funding-route die na
stap 5.2 overblijft: zij boekt de L3-fundingkosten, en stage B leest er de
carry uit. De waarde op de dagbar die op `T` sluit, is de som van de
afrekeningen met `event_ts` in `[T - 1 dag, T)`, dus uitsluitend afrekeningen
van VÓÓR het sluitmoment, en dus vóór het besluit dat op `T` valt.

Twee toetsen, elk met een negatieve controle die bewijst dat de toets een lek
ziet:

* **truncatie** -- gooi alle afrekeningen vanaf een snijmoment `c` weg; de
  waarde op elke bar die op of vóór `c` sluit, mag niet veranderen;
* **perturbatie** -- verander de rates vanaf `c`; idem.

De negatieve controles zijn twee aggregaties die de afrekening van de EIGEN bar
meenemen: het venster `(T - 1 dag, T]` (de afrekening precies op het
sluitmoment) en `[T, T + 1 dag)` (de afrekeningen van de bar erna).
"""
from __future__ import annotations

import numpy as np
import pytest

from tradebot.data.funding_panel import settlement_sums

DAY_NS = np.int64(86_400 * 10**9)
HOUR_NS = np.int64(3_600 * 10**9)


def _ticks(n_days: int = 40, seed: int = 11) -> tuple[np.ndarray, np.ndarray]:
    """8h-afrekeningen om 00, 08 en 16 uur, met een blok 2h-afrekeningen erin
    (zoals SOLUSDT van 2022-11-10 t/m 2022-12-20)."""
    rng = np.random.default_rng(seed)
    base = np.int64(1_700_000_000) * 10**9 // DAY_NS * DAY_NS
    stamps = [base + d * DAY_NS + h * HOUR_NS
              for d in range(n_days)
              for h in (range(0, 24, 2) if 15 <= d < 20 else (0, 8, 16))]
    event = np.asarray(stamps, dtype="int64")
    return event, rng.normal(1e-4, 3e-4, event.size)


def _bar_ends(event: np.ndarray, n_bars: int = 38) -> np.ndarray:
    """Dagbars gelabeld op hun SLUITMOMENT, zoals het prijspaneel."""
    first_close = event[0] // DAY_NS * DAY_NS + DAY_NS
    return first_close + np.arange(n_bars, dtype="int64") * DAY_NS


def _leaky_closed_right(event, rate, bar_end):
    """NEGATIEVE CONTROLE 1: `(T - 1 dag, T]` neemt de afrekening OP `T` mee."""
    return np.array([rate[(event > t - DAY_NS) & (event <= t)].sum() for t in bar_end])


def _leaky_next_bar(event, rate, bar_end):
    """NEGATIEVE CONTROLE 2: `[T, T + 1 dag)`, de afrekeningen van de bar erna."""
    return np.array([rate[(event >= t) & (event < t + DAY_NS)].sum() for t in bar_end])


def _truncation_breaks(aggregate, event, rate, bar_end) -> int:
    """Het aantal (snijmoment, bar)-paren waarop truncatie de waarde verandert."""
    full = aggregate(event, rate, bar_end)
    breaks = 0
    for cut in bar_end[5:-5]:
        keep = event < cut
        trunc = aggregate(event[keep], rate[keep], bar_end)
        closed = bar_end <= cut
        breaks += int(np.sum(~np.isclose(full[closed], trunc[closed],
                                         rtol=0.0, atol=0.0)))
    return breaks


def _perturbation_breaks(aggregate, event, rate, bar_end) -> int:
    full = aggregate(event, rate, bar_end)
    breaks = 0
    for cut in bar_end[5:-5]:
        noisy = rate.copy()
        noisy[event >= cut] += 1.0
        pert = aggregate(event, noisy, bar_end)
        closed = bar_end <= cut
        breaks += int(np.sum(full[closed] != pert[closed]))
    return breaks


class TestTheDailyAggregationIsCausal:
    def test_truncation_leaves_every_closed_bar_unchanged(self) -> None:
        event, rate = _ticks()
        assert _truncation_breaks(settlement_sums, event, rate,
                                  _bar_ends(event)) == 0

    def test_perturbation_after_the_close_changes_nothing_before_it(self) -> None:
        event, rate = _ticks()
        assert _perturbation_breaks(settlement_sums, event, rate,
                                    _bar_ends(event)) == 0

    def test_it_sums_every_settlement_in_the_bar_not_the_last(self) -> None:
        """Op de 2h-dagen twaalf afrekeningen, op de gewone drie. Een
        `asof_join` zou er één boeken en de rest weggooien."""
        event, rate = _ticks()
        bar_end = _bar_ends(event)
        sums = settlement_sums(event, rate, bar_end)
        for i, t in enumerate(bar_end):
            window = (event >= t - DAY_NS) & (event < t)
            assert sums[i] == pytest.approx(rate[window].sum(), rel=1e-12, abs=1e-18)
        assert {int(((event >= t - DAY_NS) & (event < t)).sum()) for t in bar_end} \
            == {3, 12}


class TestTheToetsCanGoRed:
    """Zonder deze klasse bewijzen de nullen hierboven niets."""

    @pytest.mark.parametrize("leaky", [_leaky_closed_right, _leaky_next_bar],
                             ids=["own_close_tick", "next_bar"])
    def test_truncation_catches_an_aggregation_that_reads_its_own_bar(
        self, leaky
    ) -> None:
        event, rate = _ticks()
        assert _truncation_breaks(leaky, event, rate, _bar_ends(event)) > 0

    @pytest.mark.parametrize("leaky", [_leaky_closed_right, _leaky_next_bar],
                             ids=["own_close_tick", "next_bar"])
    def test_perturbation_catches_an_aggregation_that_reads_its_own_bar(
        self, leaky
    ) -> None:
        event, rate = _ticks()
        assert _perturbation_breaks(leaky, event, rate, _bar_ends(event)) > 0
