"""D-1 POORT 2 — TEMPORELE-SHIFT-INVARIANTIE.

    Verschuif de VOLLEDIGE reeks in de kalender. Dezelfde bars, dezelfde
    waarden, dezelfde onderlinge afstanden — alleen een andere absolute datum.
    Elke feature moet dan positie-voor-positie exact dezelfde waarden geven.

Deze poort bestond volgens `reports/phase3_exit_report.md` §2 **nergens**, en de
masterprompt noemt hem daarom expliciet als te bouwen (§B-2). Hij is gebouwd in
Phase 7/8 Stage B-2.

WAT HIJ VANGT DAT POORT 1 NIET VANGT
====================================
Truncatie-invariantie (poort 1) meet of een waarde op `t` verandert door data NA
`t`. Dat is de klassieke lookahead. Maar er is een tweede manier waarop een
feature informatie kan gebruiken die hij niet heeft: door te steunen op de
ABSOLUTE positie in de kalender.

Voorbeelden die poort 1 volledig doorlaat, omdat ze strikt causaal zijn:

* een drempel die is afgestemd op een specifieke periode
  (`if ts.year >= 2021: window = 40`) — gekalibreerd op de historie die de
  onderzoeker al had gezien, en dus in-sample-kennis in de vorm van een
  constante;
* een normalisatie per kalenderjaar of -kwartaal, die op de eerste dag van een
  jaar een sprong maakt die niets met de markt te maken heeft;
* een feature die de RIJPOSITIE in de dataset gebruikt in plaats van de
  waarden, en die daarmee stilzwijgend weet hoe lang de dataset is.

Alle drie overleven een truncatietest probleemloos en alle drie maken een
backtest onherhaalbaar op nieuwe data. Zij zijn niet zeldzaam: het zijn de
gebruikelijke resten van een handmatige kalibratie die daarna is blijven staan.

DE TOETS
========
`shift_calendar(frame, offset)` verschuift de index, `event_ts_ns` en
`asof_ts_ns` met dezelfde constante. De onderlinge afstanden blijven exact
behouden, dus voor een causale feature is er niets veranderd dat hij mag zien.
De vergelijking is BIT-identiek en POSITIE-gewijs: de output-index verschuift
mee, want `BaseFeature.transform` indexeert op `asof_ts_ns`.

De offsets zijn opzettelijk grof (jaren, geen dagen). Een kalenderafhankelijkheid
die pas na een jaar zichtbaar wordt, is er ook een.

Ref: `reports/phase3_exit_report.md` §2; `fase_7_8_consolidatie_productie.md` §B-2.
"""
from __future__ import annotations

from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

from tradebot.features.base import BaseFeature, CertifiedFrame, InputSpec
from tradebot.features.registry import build_default_registry

from .d1_harness import FEAT_CFG, SYMBOLS, certified_frame, requires_store

pytestmark = pytest.mark.lookahead

REGISTRY = build_default_registry(FEAT_CFG)
FEATURE_IDS = [f.feature_id for f in REGISTRY.features]

#: De verschuivingen zijn opzettelijk GEEN veelvoud van een jaar.
#:
#: De eerste versie van dit bestand gebruikte +365, -730 en +1461 dagen, en de
#: negatieve controle bleef groen: een verschuiving over hele jaren laat de
#: jaargrenzen op precies dezelfde POSITIES in de reeks vallen, dus een feature
#: die per kalenderjaar normaliseert komt er ongeschonden uit. De toets mat toen
#: niets, en zag er intussen degelijk uit.
#:
#: Deze offsets verschuiven jaar-, kwartaal- én maandgrenzen alle drie, en zijn
#: hele dagen zodat het 1d-barraster en de 8h-fundingjoin hun structuur houden.
OFFSETS: tuple[pd.Timedelta, ...] = (
    pd.Timedelta(days=187),       # ~ half jaar
    pd.Timedelta(days=-95),       # ruim een kwartaal terug
    pd.Timedelta(days=1003),      # ~2,75 jaar, kruist twee schrikkeljaren
)
OFFSET_IDS = ["plus_187d", "minus_95d", "plus_1003d"]


def shift_calendar(source: CertifiedFrame, offset: pd.Timedelta) -> CertifiedFrame:
    """Dezelfde reeks, dezelfde afstanden, een andere absolute datum.

    Alle drie de tijdsdragers schuiven met exact dezelfde constante mee: de
    index, `event_ts_ns` en `asof_ts_ns`. Zou er één achterblijven, dan
    verandert de latentie tussen gebeurtenis en beschikbaarheid en meet deze
    toets iets anders dan hij beweert.
    """
    frame = source.frame.copy()
    delta_ns = int(offset.value)
    frame.index = frame.index + offset
    frame.index.name = source.frame.index.name
    for col in ("event_ts_ns", "asof_ts_ns"):
        frame[col] = (frame[col].to_numpy(dtype="int64") + delta_ns).astype("int64")
    return CertifiedFrame(
        frame=frame, symbol=source.symbol, data_hashes=source.data_hashes)


def assert_same_values(
    shifted: pd.DataFrame, original: pd.DataFrame, *, what: str, offset: pd.Timedelta
) -> None:
    """Positie-voor-positie bit-identiek. De index MAG verschillen — hij hoort
    precies `offset` verschoven te zijn en niets anders."""
    assert list(shifted.columns) == list(original.columns), (
        f"{what}: kolommen wijken af na verschuiving met {offset}")
    assert len(shifted) == len(original), (
        f"{what}: rijaantal wijkt af na verschuiving met {offset} "
        f"({len(shifted)} vs {len(original)})")
    for col in original.columns:
        a = shifted[col].to_numpy(dtype="float64")
        b = original[col].to_numpy(dtype="float64")
        same_nan = np.isnan(a) == np.isnan(b)
        assert same_nan.all(), (
            f"{what}/{col}: NaN-patroon verandert door een verschuiving van "
            f"{offset}; {int((~same_nan).sum())} rij(en) afwijkend. De feature "
            f"kijkt naar de kalender, niet naar de data.")
        finite = ~np.isnan(a)
        diff = np.abs(a[finite] - b[finite])
        n_bad = int((diff != 0.0).sum())
        assert n_bad == 0, (
            f"{what}/{col}: {n_bad} waarde(n) veranderen door een verschuiving "
            f"van {offset} die de onderlinge afstanden intact laat. "
            f"max |diff| = {float(diff.max()) if diff.size else 0.0:.17g}. "
            f"Er zit een absolute-tijdafhankelijkheid in deze feature; op nieuwe "
            f"data reproduceert hij zichzelf niet.")


# --------------------------------------------------------------------------- #
# De poort
# --------------------------------------------------------------------------- #
@requires_store
class TestTemporalShiftInvariancePerFeature:
    """Elke geregistreerde feature, over drie verschuivingen."""

    @pytest.mark.parametrize("offset", OFFSETS, ids=OFFSET_IDS)
    @pytest.mark.parametrize("feature", REGISTRY.features, ids=FEATURE_IDS)
    def test_feature_survives_a_calendar_shift(
        self, feature: BaseFeature, offset: pd.Timedelta
    ) -> None:
        source = certified_frame("BTCUSDT")
        original = feature.transform(source).values
        shifted = feature.transform(shift_calendar(source, offset)).values
        assert_same_values(
            shifted, original,
            what=f"BTCUSDT/{feature.feature_id}", offset=offset)


@requires_store
class TestTemporalShiftInvarianceOfTheMatrix:
    """De volledige pijplijn, op elk gecertificeerd symbool."""

    @pytest.mark.parametrize("symbol", SYMBOLS)
    def test_matrix_survives_a_calendar_shift(self, symbol: str) -> None:
        source = certified_frame(symbol)
        pipeline = REGISTRY.pipeline()
        original = pipeline.transform(source).values
        for offset in OFFSETS:
            shifted = pipeline.transform(shift_calendar(source, offset)).values
            assert_same_values(
                shifted, original, what=f"{symbol}/matrix", offset=offset)

    def test_the_shift_actually_moved_the_index(self) -> None:
        """Controle op het instrument zelf.

        Zou `shift_calendar` per ongeluk de originele frame teruggeven, dan zou
        elke test hierboven groen zijn zonder iets te meten. Dat is precies de
        klasse fout die dit bestand hoort te vinden, en hij is hier op de toets
        zelf van toepassing.
        """
        source = certified_frame("BTCUSDT")
        moved = shift_calendar(source, OFFSETS[0])
        assert moved.frame.index[0] == source.frame.index[0] + OFFSETS[0]
        assert (moved.frame["asof_ts_ns"].to_numpy()
                - source.frame["asof_ts_ns"].to_numpy()
                == int(OFFSETS[0].value)).all()
        # De latentie tussen gebeurtenis en beschikbaarheid blijft intact.
        original_lag = (source.frame["asof_ts_ns"]
                        - source.frame["event_ts_ns"]).to_numpy()
        moved_lag = (moved.frame["asof_ts_ns"]
                     - moved.frame["event_ts_ns"]).to_numpy()
        assert (original_lag == moved_lag).all()


# --------------------------------------------------------------------------- #
# Negatieve controle
# --------------------------------------------------------------------------- #
class _CalendarKeyedFeature(BaseFeature):
    """Bewust kalender-afhankelijk, en volstrekt CAUSAAL.

    Deze feature kijkt geen enkele bar vooruit; poort 1 verklaart hem groen en
    heeft daarin gelijk. Hij normaliseert alleen per kalenderjaar — het idioom
    dat overblijft wanneer iemand ooit *"even per jaar standaardiseren"* heeft
    gedaan en het is blijven staan.

    Het gevolg is dat zijn waarde op een gegeven bar afhangt van welk jaartal
    erboven staat, en dat is informatie die de markt niet levert.
    """

    name: ClassVar[str] = "deliberately_calendar_keyed"
    input_spec: ClassVar[InputSpec] = InputSpec(datasets=("ohlcv",), columns=("close",))

    def __init__(self) -> None:
        super().__init__(params={})

    @property
    def burn_in_period(self) -> int:
        return 1

    @property
    def output_columns(self) -> tuple[str, ...]:
        return ("leak_year_scaled_close",)

    def _compute(self, frame: pd.DataFrame) -> pd.DataFrame:
        close = frame["close"].astype("float64")
        # Expanding binnen het kalenderjaar: strikt causaal, en toch verschuift
        # elke waarde zodra de reeks een ander jaar in valt.
        year = pd.Series(frame.index.year, index=frame.index)
        scaled = close / close.groupby(year).cummax()
        scaled.iloc[0] = np.nan
        return pd.DataFrame(
            {self.output_columns[0]: scaled.to_numpy(dtype="float64")},
            index=frame.index,
        )


@requires_store
class TestTheGateCanGoRed:
    """Bewijs door falen. Een poort die nooit rood is geweest, is geen poort."""

    def test_a_calendar_keyed_feature_is_caught(self) -> None:
        source = certified_frame("BTCUSDT")
        leaker = _CalendarKeyedFeature()
        original = leaker.transform(source).values
        caught = 0
        for offset in OFFSETS:
            shifted = leaker.transform(shift_calendar(source, offset)).values
            with pytest.raises(AssertionError, match="verschuiving"):
                assert_same_values(
                    shifted, original, what="leak/calendar", offset=offset)
            caught += 1
        assert caught == len(OFFSETS), (
            "de kalender-afhankelijke feature glipte door minstens één "
            "verschuiving heen")

    def test_the_calendar_leak_survives_the_truncation_gate(self) -> None:
        """Waarom deze poort NAAST poort 1 moet bestaan, en niet erin.

        Dezelfde lekkende feature is truncatie-invariant: kappen aan het eind
        verandert geen enkele eerdere waarde, want `cummax` binnen het jaar is
        strikt achterwaarts. Poort 1 verklaart hem dus terecht groen en zou hem
        nooit vinden.
        """
        from .d1_harness import assert_bit_identical, cut_points

        source = certified_frame("BTCUSDT")
        leaker = _CalendarKeyedFeature()
        full = leaker.transform(source).values
        for cut in cut_points(source):
            truncated = leaker.transform(source.truncate(cut)).values
            assert_bit_identical(
                truncated, full, what="leak/calendar-under-gate-1", cut=cut)
