"""D-1 POORT 1 — TRUNCATIE-INVARIANTIE.

    Een feature die op tijdstip t een andere waarde krijgt afhankelijk van wat
    er NA t in de dataset staat, gebruikt informatie uit de toekomst.

De toets is onbarmhartig simpel. Bereken de feature-matrix over het volledige
venster `T`. Kap daarna de INPUT af op drie punten `t1, t2, t3` en bereken
opnieuw. Voor elke `tau <= t_i` moet de waarde **bit-identiek** zijn. Wijkt er
ook maar één bit af, dan is de feature gefalsificeerd — ongeacht hoe plausibel
zijn formule oogt.

BIT-IDENTIEK, NIET `allclose`
=============================
Een tolerantie zou de vraag verschuiven van *"gebruikt deze feature toekomstige
data?"* naar *"hoeveel toekomstige data mag hij gebruiken?"*. Op die tweede
vraag bestaat geen verdedigbaar antwoord, en elke drempel die je kiest is de
drempel waaronder een lek onzichtbaar wordt.

HERKOMST VAN DIT BESTAND
========================
Deze poort bestond al, maar zonder naam: hij zat in `test_feature_causality.py`
tussen de burn-in- en join-toetsen. Stage B-2 van
`fase_7_8_consolidatie_productie.md` eist de zes D-1-poorten als **zelfstandige,
benoemde bestanden**, en schrijft daarbij expliciet extraheren-en-hernoemen voor
in plaats van dupliceren. De klassen hieronder zijn woordelijk verplaatst;
`test_feature_causality.py` houdt de toetsen die iets anders meten en importeert
de gedeelde harnas uit `d1_harness.py`.

Draait op de 18 gecertificeerde reeksen uit Phase 1 (6 symbolen x
ohlcv/funding/open_interest). Ontbreekt de PIT-store, dan wordt overgeslagen met
een expliciete reden — nooit stilzwijgend als geslaagd gerapporteerd.

Ref: `fase_2_research_falsification.md` stap 5, exit criterium 1; audit §17.1.
"""
from __future__ import annotations

import pytest

from tradebot.features.base import BaseFeature
from tradebot.features.registry import build_default_registry, current_git_sha

from .d1_harness import (
    FEAT_CFG,
    N_CUTS,
    SYMBOLS,
    assert_bit_identical,
    certified_frame,
    cut_points,
    leaking_references,
    requires_store,
)

pytestmark = pytest.mark.lookahead

REGISTRY = build_default_registry(FEAT_CFG)
FEATURE_IDS = [f.feature_id for f in REGISTRY.features]


@requires_store
class TestTruncationInvariancePerFeature:
    """Elke feature afzonderlijk, op elk gecertificeerd symbool."""

    @pytest.mark.parametrize("symbol", SYMBOLS)
    @pytest.mark.parametrize("feature", REGISTRY.features, ids=FEATURE_IDS)
    def test_feature_is_truncation_invariant(
        self, feature: BaseFeature, symbol: str
    ) -> None:
        source = certified_frame(symbol)
        full = feature.transform(source).values
        for cut in cut_points(source):
            truncated = feature.transform(source.truncate(cut)).values
            assert_bit_identical(
                truncated, full, what=f"{symbol}/{feature.feature_id}", cut=cut
            )


@requires_store
class TestTruncationInvarianceOfTheMatrix:
    """De volledige matrix in één keer: ook de COMPOSITIE mag niet lekken.

    Features kunnen elk afzonderlijk causaal zijn terwijl de pijplijn die ze
    samenvoegt dat niet is — een gedeelde normalisatiestap, een gezamenlijke
    dropna, een herindexering over het geheel. Per-feature groen is daarom geen
    bewijs voor de matrix.
    """

    @pytest.mark.parametrize("symbol", SYMBOLS)
    def test_matrix_is_truncation_invariant(self, symbol: str) -> None:
        source = certified_frame(symbol)
        pipeline = REGISTRY.pipeline()
        full = pipeline.transform(source).values
        assert len(full) > FEAT_CFG.microstructure.funding_zscore_min_periods
        for cut in cut_points(source):
            truncated = pipeline.transform(source.truncate(cut)).values
            assert_bit_identical(truncated, full, what=f"{symbol}/matrix", cut=cut)

    @pytest.mark.parametrize("symbol", SYMBOLS)
    def test_matrix_hash_is_stable_under_truncation(self, symbol: str) -> None:
        """De feature_hash beschrijft de DEFINITIE, niet het gerealiseerde venster.

        Truncatie van de input verandert de matrix-inhoud maar niet de identiteit
        van de features; anders zou elke walk-forward fold een ander artefact
        opleveren en zou de registry onbruikbaar zijn als sleutel.
        """
        source = certified_frame(symbol)
        sha = current_git_sha()
        full_hash = REGISTRY.matrix_hash(data_hashes=source.data_hashes, git_sha=sha)
        for cut in cut_points(source):
            cut_hash = REGISTRY.matrix_hash(
                data_hashes=source.truncate(cut).data_hashes, git_sha=sha
            )
            assert cut_hash == full_hash


@requires_store
class TestTheGateCanGoRed:
    """Bewijs door falen: een lekkende feature MOET deze poort rood maken.

    Zonder deze klasse bewijst een groene poort alleen dat hij groen is. Beide
    lekklassen worden gedekt — de grove `shift(-1)` en het stille sample-brede
    schaalprobleem — omdat een toets die alleen de eerste vangt, de tweede
    ongemerkt doorlaat, en de tweede is degene die in de praktijk voorkomt.
    """

    @pytest.mark.parametrize(
        "leaker", leaking_references(),
        ids=["shift_minus_one", "sample_wide_scaler"],
    )
    def test_leaking_feature_fails_truncation_invariance(
        self, leaker: BaseFeature
    ) -> None:
        source = certified_frame("BTCUSDT")
        full = leaker.transform(source).values
        failures = 0
        for cut in cut_points(source):
            truncated = leaker.transform(source.truncate(cut)).values
            with pytest.raises(AssertionError):
                assert_bit_identical(
                    truncated, full, what=f"leak/{leaker.name}", cut=cut
                )
            failures += 1
        assert failures == N_CUTS, (
            "De invariantietoets ving het lek niet op elk snijpunt. Een toets "
            "die een bekend lek doorlaat, bewijst niets over de features die hij "
            "groen verklaart."
        )
