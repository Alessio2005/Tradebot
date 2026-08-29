"""D-1 POORT 3 — TOEKOMST-KOLOMVERGIFTIGING.

    Plak een kolom vol zuivere toekomstinformatie aan het inputframe. Geen
    enkele geregistreerde feature mag daarvan iets merken.

Een feature declareert zijn invoer in `InputSpec`. Die declaratie is de basis
van elke causaliteitsclaim in dit platform: `data_hashes`, de `feature_hash` en
de hele provenance-keten gaan ervan uit dat een feature leest wat hij zegt te
lezen. Deze poort toetst die aanname in plaats van haar te geloven.

WAAROM DIT EEN EIGEN POORT IS
=============================
`reports/phase3_exit_report.md` §2 noemt deze poort als bestaand *"alleen als
negatieve-skip-variant, niet als generieke kolom-injectie"*. Het verschil is
wezenlijk. De bestaande variant toont dat één specifieke lekkende feature wordt
gevangen. Deze poort toont dat **geen enkele** feature gevoelig is voor een
kolom die hij niet heeft gedeclareerd — en dat is de eigenschap waarop de rest
van de keten steunt.

GEMETEN, NIET AANGENOMEN: `transform()` FILTERT NIET
====================================================
`BaseFeature.transform` controleert dat de gedeclareerde kolommen AANWEZIG zijn
en geeft daarna het VOLLEDIGE frame door aan `_compute`. Er is dus geen
mechanisme dat een feature ervan weerhoudt een niet-gedeclareerde kolom te
lezen; er is alleen de discipline van de auteur. Deze poort is daarom de enige
plek waar die discipline wordt gemeten.

Dat is geen theoretisch risico. Het idioom dat hem breekt is doodgewoon:

    frame.select_dtypes("number").mean(axis=1)      # "gemiddelde van alles"
    frame.drop(columns=["close"]).sum(axis=1)
    frame.iloc[:, 3:].rolling(20).mean()

Alle drie zijn stabiel zolang het frame stabiel is, en alle drie veranderen van
betekenis zodra er upstream een kolom bij komt — een nieuwe join, een extra
dataset, een debugkolom die iemand liet staan. Het lek ontstaat dan zonder dat
er ook maar één regel in de feature is gewijzigd.

DE VIER VERGIFTIGINGEN
======================
Elk is puur toekomst, en elk is een andere vorm ervan:

* `poison_next_close`   — de close van de volgende bar. Het klassieke lek.
* `poison_global_max`   — een constante gelijk aan het maximum over de HELE
                          reeks. Bevat geen enkele bar-specifieke informatie en
                          toch de uitkomst van de volledige steekproef.
* `poison_reversed`     — de reeks achterstevoren. Elke waarde is een andere
                          bar, en op de eerste helft is dat de verre toekomst.
* `poison_perfect_label`— het teken van het volgende rendement. Het label dat
                          een model hoort te voorspellen.

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

POISON_COLUMNS = (
    "poison_next_close",
    "poison_global_max",
    "poison_reversed",
    "poison_perfect_label",
)


def poison(source: CertifiedFrame, *, columns: tuple[str, ...] = POISON_COLUMNS
           ) -> CertifiedFrame:
    """Hetzelfde frame, plus kolommen die zuivere toekomstinformatie dragen.

    De originele kolommen worden NIET aangeraakt. Verandert er iets in de
    output, dan komt dat uitsluitend doordat een feature buiten zijn
    gedeclareerde invoer heeft gelezen.
    """
    frame = source.frame.copy()
    close = frame["close"].astype("float64")

    available: dict[str, np.ndarray] = {
        # bfill dekt alleen de laatste rij, die na shift(-1) geen opvolger heeft
        "poison_next_close": close.shift(-1).bfill().to_numpy(dtype="float64"),
        "poison_global_max": np.full(len(frame), float(close.max())),
        "poison_reversed": close.to_numpy(dtype="float64")[::-1].copy(),
        "poison_perfect_label": np.sign(
            close.shift(-1).bfill().to_numpy(dtype="float64")
            - close.to_numpy(dtype="float64")),
    }
    for name in columns:
        frame[name] = available[name]
    return CertifiedFrame(
        frame=frame, symbol=source.symbol, data_hashes=source.data_hashes)


def assert_unaffected(
    poisoned: pd.DataFrame, clean: pd.DataFrame, *, what: str
) -> None:
    """Bit-identiek. Eén afwijkende bit betekent dat de kolom is gelezen."""
    assert list(poisoned.columns) == list(clean.columns), (
        f"{what}: de vergiftigde run levert andere OUTPUT-kolommen op. De "
        f"feature genereert kolommen op grond van wat hij in het frame "
        f"aantreft.")
    assert poisoned.index.equals(clean.index), f"{what}: index wijkt af"
    for col in clean.columns:
        a = poisoned[col].to_numpy(dtype="float64")
        b = clean[col].to_numpy(dtype="float64")
        same_nan = np.isnan(a) == np.isnan(b)
        assert same_nan.all(), (
            f"{what}/{col}: NaN-patroon verandert door een kolom die deze "
            f"feature niet heeft gedeclareerd; "
            f"{int((~same_nan).sum())} rij(en) afwijkend.")
        finite = ~np.isnan(a)
        diff = np.abs(a[finite] - b[finite])
        n_bad = int((diff != 0.0).sum())
        assert n_bad == 0, (
            f"{what}/{col}: {n_bad} waarde(n) veranderen door een kolom met "
            f"zuivere toekomstinformatie die NIET in de input_spec staat. "
            f"max |diff| = {float(diff.max()) if diff.size else 0.0:.17g}. "
            f"De feature leest buiten zijn declaratie; zijn feature_hash en "
            f"data_hashes beschrijven dan niet waarop hij is berekend.")


# --------------------------------------------------------------------------- #
# De poort
# --------------------------------------------------------------------------- #
@requires_store
class TestNoFeatureReadsAnUndeclaredColumn:
    @pytest.mark.parametrize("feature", REGISTRY.features, ids=FEATURE_IDS)
    def test_all_four_poisons_at_once(self, feature: BaseFeature) -> None:
        """Alle vier tegelijk: als er iets doorlekt, moet dit het zien."""
        source = certified_frame("BTCUSDT")
        clean = feature.transform(source).values
        dirty = feature.transform(poison(source)).values
        assert_unaffected(dirty, clean, what=f"BTCUSDT/{feature.feature_id}")

    @pytest.mark.parametrize("poison_col", POISON_COLUMNS)
    def test_each_poison_separately_on_the_matrix(self, poison_col: str) -> None:
        """Eén tegelijk, zodat de melding aanwijst WELKE kolom is gelezen."""
        source = certified_frame("BTCUSDT")
        pipeline = REGISTRY.pipeline()
        clean = pipeline.transform(source).values
        dirty = pipeline.transform(poison(source, columns=(poison_col,))).values
        assert_unaffected(dirty, clean, what=f"matrix/{poison_col}")

    @pytest.mark.parametrize("symbol", SYMBOLS)
    def test_the_whole_matrix_on_every_symbol(self, symbol: str) -> None:
        source = certified_frame(symbol)
        pipeline = REGISTRY.pipeline()
        clean = pipeline.transform(source).values
        dirty = pipeline.transform(poison(source)).values
        assert_unaffected(dirty, clean, what=f"{symbol}/matrix")


@requires_store
class TestThePoisonIsActuallyPoisonous:
    """Controle op het instrument. Een vergiftiging die geen toekomst bevat,
    zou elke test hierboven groen maken zonder iets te meten."""

    def test_next_close_really_is_the_next_bar(self) -> None:
        source = certified_frame("BTCUSDT")
        f = poison(source).frame
        close = f["close"].to_numpy(dtype="float64")
        nxt = f["poison_next_close"].to_numpy(dtype="float64")
        assert (nxt[:-1] == close[1:]).all()

    def test_the_perfect_label_predicts_the_next_return_exactly(self) -> None:
        source = certified_frame("BTCUSDT")
        f = poison(source).frame
        close = f["close"].to_numpy(dtype="float64")
        label = f["poison_perfect_label"].to_numpy(dtype="float64")
        actual = np.sign(np.diff(close))
        moved = actual != 0.0
        assert (label[:-1][moved] == actual[moved]).all()

    def test_the_columns_are_not_declared_by_any_feature(self) -> None:
        """Als een feature ze wél declareerde, zou lezen legitiem zijn en zou
        deze poort een feature falsificeren die zich aan het contract houdt."""
        declared: set[str] = set()
        for feature in REGISTRY.features:
            declared |= set(type(feature).input_spec.columns)
        assert declared.isdisjoint(POISON_COLUMNS)


# --------------------------------------------------------------------------- #
# Negatieve controle
# --------------------------------------------------------------------------- #
class _ColumnAgnosticFeature(BaseFeature):
    """Bewust lek, en niet eens opzettelijk geschreven als lek.

    Deze feature middelt over ALLE numerieke kolommen die hij in het frame
    aantreft. Op het frame waarvoor hij is geschreven, is hij causaal en
    volstrekt onschuldig; hij wordt lekkend op de dag dat er upstream een kolom
    bij komt. Zijn `input_spec` blijft intussen keurig `("close",)` melden.

    Dat is precies waarom deze poort bestaat: het lek zit niet in de feature
    maar in de aanname dat het frame nooit verandert.
    """

    name: ClassVar[str] = "deliberately_column_agnostic"
    input_spec: ClassVar[InputSpec] = InputSpec(datasets=("ohlcv",), columns=("close",))

    def __init__(self) -> None:
        super().__init__(params={})

    @property
    def burn_in_period(self) -> int:
        return 1

    @property
    def output_columns(self) -> tuple[str, ...]:
        return ("agnostic_row_mean",)

    def _compute(self, frame: pd.DataFrame) -> pd.DataFrame:
        numeric = frame.select_dtypes("number").drop(
            columns=[c for c in ("event_ts_ns", "asof_ts_ns")
                     if c in frame.columns])
        out = numeric.mean(axis=1)
        out.iloc[0] = np.nan
        return pd.DataFrame(
            {self.output_columns[0]: out.to_numpy(dtype="float64")},
            index=frame.index,
        )


@requires_store
class TestTheGateCanGoRed:
    @pytest.mark.parametrize("poison_col", POISON_COLUMNS)
    def test_a_column_agnostic_feature_is_caught_by_every_poison(
        self, poison_col: str
    ) -> None:
        source = certified_frame("BTCUSDT")
        leaker = _ColumnAgnosticFeature()
        clean = leaker.transform(source).values
        dirty = leaker.transform(poison(source, columns=(poison_col,))).values
        with pytest.raises(AssertionError, match="input_spec"):
            assert_unaffected(dirty, clean, what=f"leak/{poison_col}")

    def test_the_agnostic_leak_survives_the_truncation_gate(self) -> None:
        """Waarom deze poort naast poort 1 moet bestaan.

        Op een ONVERGIFTIGD frame is deze feature truncatie-invariant: een
        rijgemiddelde kijkt niet vooruit. Poort 1 verklaart hem terecht groen en
        zou hem nooit vinden. Het lek ontstaat pas door een verandering in het
        FRAME, en dat is niet iets wat een invariantietoets op de feature kan
        zien.
        """
        from .d1_harness import assert_bit_identical, cut_points

        source = certified_frame("BTCUSDT")
        leaker = _ColumnAgnosticFeature()
        full = leaker.transform(source).values
        for cut in cut_points(source):
            truncated = leaker.transform(source.truncate(cut)).values
            assert_bit_identical(
                truncated, full, what="leak/agnostic-under-gate-1", cut=cut)
