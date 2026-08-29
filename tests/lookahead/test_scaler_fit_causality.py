"""D-1 POORT 4 — CAUSALITEIT VAN DE SCALER-FIT.

    Een scaler, normalisator of PCA die op de VOLLEDIGE steekproef is gefit,
    lekt de verdeling van de testperiode naar de trainingsperiode — ook wanneer
    er geen enkele waarde vooruit wordt gelezen.

Dit is de subtielste van de zes poorten, en de enige waarvan het lek per rij
onzichtbaar is. Er wordt geen bar uit de toekomst gelezen. Er wordt alleen een
GEMIDDELDE, een STANDAARDDEVIATIE of een QUANTIELGRENS gebruikt die mede door
toekomstige bars is bepaald. Het model weet daardoor hoe de wereld er gemiddeld
uit zal zien voordat hij hem heeft gezien.

WAAROM DIT EEN LEK IS EN GEEN DETAIL
====================================
Een `StandardScaler` die op alles is gefit, vertelt elke trainingsrij hoe groot
de spreiding over de hele periode is. In een markt met regimewisselingen is dat
precies de informatie die telt: de scaler verklapt dat er verderop een
volatiliteitspiek komt, want anders zou de gemeten sigma kleiner zijn geweest.
Het model leert dan een schaal die op het beslismoment niet bestond.

De omvang is niet klein. Dat is meetbaar en wordt hieronder gemeten in plaats
van beweerd (`TestTheLeakIsMaterial`).

WAT ER IS GEMETEN IN DEZE CODEBASE
==================================
Drie scaler-paden, drie verschillende toestanden:

| pad | fit op | gedrag zonder train-indices |
|---|---|---|
| `FeatureOrthogonalizer.fit_on_train_indices` | uitsluitend train | n.v.t. — train is verplicht |
| `FeatureOrthogonalizer.fit_transform` | eerste `calib_frac` bars | WAARSCHUWT en lekt door |
| `features/scaling.py::apply_symqs_overlay` | train, of niets | CRASHT (`CausalityViolationError`) |

Die tweede rij is de asymmetrie waar deze poort op let: één scaler crasht, de
andere waarschuwt. Een waarschuwing in een logregel is geen poort — hij houdt
niets tegen en niemand leest hem terug.

Gemeten en vastgelegd: de lekkende tak in `features/regime.py` staat achter
`cfg.feature_pipeline.use_pca_pre_cv`, en dat veld bestaat in **geen enkel
schema en geen enkele YAML**. De tak is daarmee dode code, niet een optie.
`TestTheLeakyBranchStaysUnreachable` zorgt dat dat zo blijft: iemand die het veld
alsnog toevoegt, maakt een lek activeerbaar en moet dat bewust doen.

Ref: audit §7.2; CHIEF AUDIT 2026-05-23 (P-7, P2.2); `reports/phase3_exit_report.md` §2.
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from tradebot.features.orthogonalize import FeatureOrthogonalizer
from tradebot.features.volatility import causal_expanding_std, log_returns
from tradebot.utils.failfast import CausalityViolationError

from .d1_harness import (
    FEAT_CFG,
    ROOT,
    assert_bit_identical,
    certified_frame,
    cut_points,
    requires_store,
)

pytestmark = pytest.mark.lookahead

N_BARS = 900
N_FEATS = 8
#: De train-fold ligt MIDDEN in de reeks, niet vooraan. Zou hij vooraan liggen,
#: dan valt hij samen met het kalibratievenster van `fit_transform` en is het
#: verschil tussen de twee paden niet te zien.
TRAIN_SLICE = slice(400, 700)


@pytest.fixture(scope="module")
def matrix() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Een feature-matrix met een REGIMEWISSEL buiten de train-fold.

    De laatste 200 bars hebben een vijfmaal zo grote spreiding. Dat is geen
    kunstgreep maar de normale situatie in crypto, en het is precies de
    eigenschap die een sample-brede fit naar de trainingsperiode lekt.
    """
    rng = np.random.default_rng(20260829)
    x = rng.normal(0.0, 1.0, (N_BARS, N_FEATS))
    x[-200:] *= 5.0
    train_idx = np.arange(N_BARS)[TRAIN_SLICE]
    return x, train_idx, [f"f{i}" for i in range(N_FEATS)]


def _orth() -> FeatureOrthogonalizer:
    return FeatureOrthogonalizer(n_components=0.95, min_components=2)


# --------------------------------------------------------------------------- #
# De poort: de fit mag alleen train-rijen zien
# --------------------------------------------------------------------------- #
@requires_store
class TestTheFitSeesOnlyTrainRows:
    """De beslissende toets, en hij is gedragsmatig en niet statisch.

    Vervang alles BUITEN de train-fold door onzin en fit opnieuw op exact
    dezelfde train-indices. Zijn de getransformeerde waarden OP DE TRAIN-RIJEN
    bit-identiek, dan heeft de fit die onzin niet gezien. Wijken zij af, dan
    wel — hoe de code er ook uitziet.
    """

    def test_corrupting_everything_outside_train_changes_nothing_on_train(
        self, matrix
    ) -> None:
        x, train_idx, names = matrix

        clean, _ = _orth().fit_on_train_indices(x, train_idx, names, prefix="pc")

        corrupted = x.copy()
        mask = np.ones(len(x), dtype=bool)
        mask[train_idx] = False
        rng = np.random.default_rng(7)
        corrupted[mask] = rng.normal(500.0, 250.0, corrupted[mask].shape)

        dirty, _ = _orth().fit_on_train_indices(
            corrupted, train_idx, names, prefix="pc")

        a = clean[train_idx]
        b = dirty[train_idx]
        assert a.shape == b.shape, (
            "het aantal componenten verandert door data buiten de train-fold; "
            "de fit heeft die data gezien")
        n_bad = int((a != b).sum())
        assert n_bad == 0, (
            f"{n_bad} van {a.size} getransformeerde waarde(n) op de TRAIN-rijen "
            f"veranderen doordat er buiten de train-fold andere data staat. De "
            f"scaler/PCA is dan niet uitsluitend op train gefit, en elke "
            f"OOS-meting die erop volgt is besmet.")

    def test_the_corruption_does_reach_the_non_train_rows(self, matrix) -> None:
        """Controle op het instrument: de vervuiling moet wél ergens aankomen.

        Zou `fit_on_train_indices` de matrix buiten train ongemoeid laten, dan
        zou de test hierboven groen zijn zonder iets te bewijzen.
        """
        x, train_idx, names = matrix
        clean, _ = _orth().fit_on_train_indices(x, train_idx, names, prefix="pc")

        corrupted = x.copy()
        mask = np.ones(len(x), dtype=bool)
        mask[train_idx] = False
        corrupted[mask] += 1000.0
        dirty, _ = _orth().fit_on_train_indices(
            corrupted, train_idx, names, prefix="pc")

        assert (clean[mask] != dirty[mask]).any(), (
            "de vervuiling is nergens zichtbaar; de toets hierboven meet niets")


class TestTheLeakIsMaterial:
    """NEGATIEVE CONTROLE, en tegelijk de meting die het belang aantoont.

    Het legacy-pad `fit_transform` fit op het KALIBRATIEVENSTER — de eerste
    `calib_frac` bars — en transformeert daarna alles. Binnen een CPCV- of
    walk-forward-context vallen die bars in andermans testfold.
    """

    def test_fit_transform_is_moved_by_data_outside_the_train_fold(
        self, matrix
    ) -> None:
        x, train_idx, names = matrix

        with pytest.warns(DeprecationWarning, match="fits on full"):
            clean, _ = _orth().fit_transform(x, names, prefix="pc")

        corrupted = x.copy()
        mask = np.ones(len(x), dtype=bool)
        mask[train_idx] = False
        rng = np.random.default_rng(7)
        corrupted[mask] = rng.normal(500.0, 250.0, corrupted[mask].shape)

        with pytest.warns(DeprecationWarning, match="fits on full"):
            dirty, _ = _orth().fit_transform(corrupted, names, prefix="pc")

        differs = (
            clean[train_idx].shape != dirty[train_idx].shape
            or bool((clean[train_idx] != dirty[train_idx]).any())
        )
        assert differs, (
            "het legacy-pad bleek hier ongevoelig voor data buiten de "
            "train-fold. Dan meet deze negatieve controle niets, en is de "
            "voorkeur voor fit_on_train_indices niet aangetoond.")

    def test_the_legacy_path_still_announces_itself(self, matrix) -> None:
        """De waarschuwing is geen poort, maar hij moet er wel zijn: hij is het
        enige spoor dat een lekkende run achterlaat."""
        x, _, names = matrix
        with pytest.warns(DeprecationWarning, match="fit_on_train_indices"):
            _orth().fit_transform(x, names, prefix="pc")


class TestTheScalerThatCrashesInsteadOfWarning:
    """`apply_symqs_overlay` doet het goed: geen train-indices is een CRASH.

    Dit is de vorm die de faseregel voorschrijft (*"fail-fast compliance"*), en
    hij staat hier zodat de asymmetrie met `fit_transform` zichtbaar blijft in
    plaats van weg te zakken.
    """

    def test_no_train_indices_is_a_causality_violation(self) -> None:
        from tradebot.features.scaling import apply_symqs_overlay

        rng = np.random.default_rng(3)
        names = ["rsi_14", "stoch_k", "close"]
        raw = rng.normal(0.0, 1.0, (300, len(names)))
        with pytest.raises(CausalityViolationError, match="train_indices"):
            apply_symqs_overlay(raw.copy(), raw, names, train_indices=None)


# --------------------------------------------------------------------------- #
# De causale normalisator in de featurelaag
# --------------------------------------------------------------------------- #
@requires_store
class TestTheCausalScalerInTheFeatureLayer:
    """`causal_expanding_std` versus de sample-brede variant, zij aan zij."""

    def test_the_causal_expanding_scaler_is_truncation_invariant(self) -> None:
        source = certified_frame("BTCUSDT")
        mp = int(FEAT_CFG.min_expanding_periods)

        def causal(frame):
            return causal_expanding_std(log_returns(frame["close"]), min_periods=mp)

        full = causal(source.frame)
        for cut in cut_points(source):
            truncated = causal(source.truncate(cut).frame)
            assert_bit_identical(
                truncated.to_frame("scale"), full.to_frame("scale"),
                what="BTCUSDT/causal_expanding_std", cut=cut)

    def test_a_sample_wide_scaler_is_not(self) -> None:
        """De negatieve controle op de featurelaag: `r / r.std()` schuift mee
        met elke bar die je achteraan toevoegt."""
        source = certified_frame("BTCUSDT")

        def sample_wide(frame):
            r = log_returns(frame["close"])
            return r / r.std()

        full = sample_wide(source.frame)
        cut = cut_points(source)[0]
        truncated = sample_wide(source.truncate(cut).frame)
        with pytest.raises(AssertionError):
            assert_bit_identical(
                truncated.to_frame("scale"), full.to_frame("scale"),
                what="BTCUSDT/sample_wide", cut=cut)


# --------------------------------------------------------------------------- #
# Statische controle op het promotiepad
# --------------------------------------------------------------------------- #
def _orthogonalizer_fit_transform_calls(path: Path) -> list[int]:
    """Regelnummers van `<iets met 'orth'>.fit_transform(...)`-aanroepen."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    hits: list[int] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "fit_transform"):
            continue
        receiver = node.func.value
        name = (receiver.id if isinstance(receiver, ast.Name)
                else receiver.attr if isinstance(receiver, ast.Attribute)
                else "")
        if "orth" in name.lower():
            hits.append(node.lineno)
    return hits


class TestThePromotionPathUsesThePerFoldFit:
    """De gedragstest hierboven bewijst dat het CAUSALE pad causaal is. Deze
    bewijst dat het promotiepad dat pad ook daadwerkelijk kiest."""

    @pytest.mark.parametrize("rel", [
        "src/tradebot/tune/objective.py",
        "apps/train_cpcv.py",
    ])
    def test_no_global_orthogonalizer_fit_in_the_training_path(
        self, rel: str
    ) -> None:
        path = ROOT / rel
        hits = _orthogonalizer_fit_transform_calls(path)
        assert hits == [], (
            f"{rel} roept FeatureOrthogonalizer.fit_transform aan op regel(s) "
            f"{hits}. Dat fit op het kalibratievenster in plaats van per fold "
            f"op de train-indices, en lekt de variantiestructuur van testbars "
            f"naar de fit. Gebruik fit_on_train_indices(X, train_idx, ...).")

    @pytest.mark.parametrize("rel", [
        "src/tradebot/tune/objective.py",
        "apps/train_cpcv.py",
    ])
    def test_the_training_path_does_call_the_per_fold_fit(self, rel: str) -> None:
        """Anders zou het bestand ook slagen door helemaal niet te schalen."""
        text = (ROOT / rel).read_text(encoding="utf-8")
        assert "fit_on_train_indices" in text, (
            f"{rel} gebruikt de per-fold fit niet; de test hierboven is dan "
            f"groen zonder iets te bewaken")


class TestTheLeakyBranchStaysUnreachable:
    """`features/regime.py` heeft een pre-CV PCA-tak achter een vlag die in geen
    enkel schema en geen enkele YAML voorkomt. De tak is dode code.

    Zodra iemand het veld toevoegt, wordt een bekend lek activeerbaar via
    configuratie — precies de vorm van stille degradatie die audit §23 verbiedt.
    Deze test maakt die stap zichtbaar in plaats van stil.
    """

    def test_use_pca_pre_cv_is_not_a_configurable_field(self) -> None:
        offenders: list[str] = []
        for path in list((ROOT / "conf").rglob("*.yaml")) + [
            ROOT / "src" / "tradebot" / "schemas" / "config.py"
        ]:
            if "use_pca_pre_cv" in path.read_text(encoding="utf-8"):
                offenders.append(str(path.relative_to(ROOT)))
        assert offenders == [], (
            f"`use_pca_pre_cv` is configureerbaar geworden in {offenders}. Die "
            f"vlag zet de pre-CV PCA aan, die op de volledige `df_merged` fit "
            f"(AUDIT C-3). Een lek dat je met een YAML-regel kunt aanzetten, is "
            f"een lek dat op een dag aan staat.")

    def test_the_branch_still_defaults_to_off(self) -> None:
        text = (ROOT / "src" / "tradebot" / "features" / "regime.py").read_text(
            encoding="utf-8")
        assert 'getattr(self.cfg.feature_pipeline, "use_pca_pre_cv", False)' in text, (
            "de default van use_pca_pre_cv is niet langer False, of de tak is "
            "herschreven; controleer of de pre-CV PCA nog steeds uit staat")
