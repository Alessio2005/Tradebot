# tests/unit/test_selection.py
"""`selection/` had nul tests. Dit bestand legt vast wat het pakket BELOOFT.

Phase 9, stap 11. `reports/phase9_coverage_baseline.md` meet `selection/` op
**0,0 % dekking over 293 LOC** — het enige pakket in `src/` zonder een enkele
test. Dat is bijzonder ongelukkig, want beide modules bestaan juist om een
lookahead-fout te REPAREREN die een audit heeft gevonden:

* `sfi._purged_timeseries_splits` — CHIEF AUDIT 2026-05-23, P-11.
  `sklearn.TimeSeriesSplit` kent geen purge, dus trainmonsters die eindigen op
  `fold_end - 1` overlappen in feature/label-geheugen met testmonsters die
  beginnen op `fold_end` (AFML §7.2). De vervanger houdt een gat aan.
* `mda._block_shuffle` — CHIEF AUDIT 2026-05-23, P-10. Zonder embargo lekt
  FFD-geheugen over de blokgrens en wordt permutatie-importance omhoog
  vertekend. De embargo-bars worden op **NaN** gezet en nadrukkelijk niet op
  nul: een nul is een GELDIGE observatie ("geen beweging") en zou de
  MDA-nulverdeling verschuiven.

WAAROM ER NEGATIEVE CONTROLES IN STAAN
======================================
`docs/PROJECT_STATE.md` opent met de les van de externe review: *een groen
criterium is pas bewijs als de bijbehorende test rood kan worden om de reden
waarvoor het criterium bestaat.* Voor de twee auditreparaties hierboven is dat
niet vanzelfsprekend — een test die alleen de vorm van de uitvoer controleert,
blijft groen wanneer iemand de purge of het embargo terugdraait. Daarom draagt
dit bestand per reparatie een expliciete negatieve controle die de OUDE,
foutieve variant nabouwt en aantoont dat de assertie daarop wél afgaat.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradebot.selection.mda import _block_shuffle, causal_mda, filter_by_mda
from tradebot.selection.sfi import _purged_timeseries_splits, rank_features_by_sfi


# ═════════════════════════════════════════════════════════════════════════════
# P-11 — de purge tussen train en test
# ═════════════════════════════════════════════════════════════════════════════
class TestPurgedSplitsKeepTrainAndTestApart:
    def test_every_fold_leaves_exactly_purge_bars_between_train_and_test(self) -> None:
        """Het gat IS de reparatie. Zonder gat overlapt het labelgeheugen."""
        purge = 10
        splits = _purged_timeseries_splits(n_samples=300, n_splits=4, purge_bars=purge)

        assert splits, "geen enkele fold opgeleverd; de test toetst dan niets"
        for train_idx, test_idx in splits:
            gap = int(test_idx[0]) - int(train_idx[-1]) - 1
            assert gap == purge, f"gat is {gap}, verwacht {purge}"

    def test_no_train_index_ever_lies_at_or_after_the_test_fold(self) -> None:
        """Causaliteit: trainen mag nooit op bars die na de teststart liggen."""
        for train_idx, test_idx in _purged_timeseries_splits(300, 4, 10):
            assert int(train_idx.max()) < int(test_idx.min())

    def test_a_larger_purge_pushes_the_test_fold_further_away(self) -> None:
        """De parameter doet werkelijk iets; hij is geen dode knop."""
        near = _purged_timeseries_splits(300, 4, 5)
        far = _purged_timeseries_splits(300, 4, 40)
        assert int(far[0][1][0]) - int(near[0][1][0]) == 35

    def test_too_few_samples_yield_no_folds_instead_of_a_crash(self) -> None:
        assert _purged_timeseries_splits(n_samples=4, n_splits=5, purge_bars=10) == []

    # ── negatieve controle ───────────────────────────────────────────────────
    def test_the_gap_assertion_would_catch_an_unpurged_splitter(self) -> None:
        """Bouw de OUDE variant na: train tot fold_end, test vanaf fold_end.

        Zonder deze controle bewijst de eerste test niets — hij zou net zo
        groen zijn op een splitter die het gat helemaal niet aanhoudt.
        """
        n, n_splits = 300, 4
        fold_size = n // (n_splits + 1)
        unpurged = []
        for k in range(1, n_splits + 1):
            fold_end = k * fold_size
            unpurged.append((
                np.arange(0, fold_end),
                np.arange(fold_end, min(fold_end + fold_size, n)),
            ))

        gaps = {int(te[0]) - int(tr[-1]) - 1 for tr, te in unpurged}
        assert gaps == {0}, "de nagebouwde variant hoort GEEN purge te hebben"
        assert gaps != {10}, "de assertie uit de eerste test gaat hierop af"


# ═════════════════════════════════════════════════════════════════════════════
# P-10 — het embargo tussen geshuffelde blokken
# ═════════════════════════════════════════════════════════════════════════════
class TestBlockShuffleEmbargo:
    def test_without_embargo_the_shuffle_is_a_pure_permutation(self) -> None:
        """Geen embargo betekent: dezelfde waarden, andere volgorde."""
        arr = np.arange(200, dtype=np.float64)
        out = _block_shuffle(arr, block_size=20, rng=np.random.default_rng(7))

        assert out.shape == arr.shape
        assert np.array_equal(np.sort(out), np.sort(arr))
        assert not np.isnan(out).any()

    def test_the_embargo_writes_nan_and_never_zero(self) -> None:
        """Een nul is een GELDIGE observatie; NaN is de ontbrekende waarde.

        Dit is P-10 letterlijk. Zou het embargo nullen schrijven, dan telt de
        MDA die bars mee als "geen beweging" en verschuift de nulverdeling.
        """
        arr = np.arange(1, 201, dtype=np.float64)   # geen enkele echte nul
        out = _block_shuffle(arr, block_size=20, rng=np.random.default_rng(7),
                             embargo_bars=3)

        assert np.isnan(out).any(), "embargo_bars > 0 hoort NaN te schrijven"
        assert not (out == 0).any(), "embargo schreef een nul in plaats van NaN"

    def test_the_embargo_blanks_exactly_three_bars_per_block_boundary(self) -> None:
        arr = np.arange(1, 201, dtype=np.float64)
        n_blocks = 200 // 20
        out = _block_shuffle(arr, block_size=20, rng=np.random.default_rng(7),
                             embargo_bars=3)
        assert int(np.isnan(out).sum()) == 3 * (n_blocks - 1)

    def test_a_zero_embargo_blanks_nothing(self) -> None:
        arr = np.arange(1, 201, dtype=np.float64)
        out = _block_shuffle(arr, block_size=20, rng=np.random.default_rng(7),
                             embargo_bars=0)
        assert int(np.isnan(out).sum()) == 0

    # ── negatieve controle ───────────────────────────────────────────────────
    def test_the_nan_assertion_would_catch_a_zero_filling_embargo(self) -> None:
        """De teruggedraaide variant: vul met 0.0 in plaats van NaN."""
        arr = np.arange(1, 201, dtype=np.float64)
        reverted = _block_shuffle(arr, block_size=20, rng=np.random.default_rng(7),
                                  embargo_bars=3).copy()
        reverted[np.isnan(reverted)] = 0.0

        assert (reverted == 0).any(), "de nagebouwde variant hoort nullen te hebben"
        assert not np.isnan(reverted).any(), "en geen NaN meer"


# ═════════════════════════════════════════════════════════════════════════════
# causal_mda — het oordeel over een feature
# ═════════════════════════════════════════════════════════════════════════════
class _ModelThatOnlyReadsColumnZero:
    """Een echte, deterministische classifier — geen mock.

    Hij leest uitsluitend kolom 0. Daarmee is het verwachte MDA-oordeel vooraf
    bekend: kolom 0 verliest AUC bij shuffling, kolom 1 kan dat per constructie
    niet.
    """

    @staticmethod
    def predict_proba(X: np.ndarray) -> np.ndarray:
        p = 1.0 / (1.0 + np.exp(-np.nan_to_num(X[:, 0])))
        return np.column_stack([1.0 - p, p])


@pytest.fixture
def separable_oos() -> tuple[np.ndarray, np.ndarray, list[str]]:
    rng = np.random.default_rng(11)
    n = 400
    signal = np.linspace(-3.0, 3.0, n)
    noise = rng.normal(size=n)
    X = np.column_stack([signal, noise])
    y = (signal > 0).astype(int)
    return X, y, ["signal", "noise"]


class TestCausalMDA:
    def test_the_feature_the_model_uses_scores_above_the_one_it_ignores(
        self, separable_oos
    ) -> None:
        X, y, names = separable_oos
        out = causal_mda(_ModelThatOnlyReadsColumnZero(), X, y, names,
                         block_size=20, n_repeats=5, random_seed=3)

        by = out.set_index("feature")["mda_mean"]
        assert by["signal"] > by["noise"]
        assert by["signal"] > 0.0, "shuffling van de gebruikte feature hoort AUC te kosten"

    def test_a_feature_the_model_never_reads_scores_exactly_zero(
        self, separable_oos
    ) -> None:
        """Kolom 1 raakt de voorspelling niet, dus shuffling verandert niets."""
        X, y, names = separable_oos
        out = causal_mda(_ModelThatOnlyReadsColumnZero(), X, y, names,
                         block_size=20, n_repeats=5, random_seed=3)
        assert out.set_index("feature").loc["noise", "mda_mean"] == pytest.approx(0.0)

    def test_an_insignificant_feature_is_marked_for_dropping(self, separable_oos) -> None:
        """`drop` is het oordeel: t-stat onder 2,0 betekent weg."""
        X, y, names = separable_oos
        out = causal_mda(_ModelThatOnlyReadsColumnZero(), X, y, names,
                         block_size=20, n_repeats=5, random_seed=3).set_index("feature")
        assert bool(out.loc["noise", "drop"]) is True

    def test_the_result_is_sorted_by_mda_mean_descending(self, separable_oos) -> None:
        X, y, names = separable_oos
        out = causal_mda(_ModelThatOnlyReadsColumnZero(), X, y, names,
                         block_size=20, n_repeats=3, random_seed=3)
        assert list(out["mda_mean"]) == sorted(out["mda_mean"], reverse=True)

    def test_it_is_reproducible_under_the_same_seed(self, separable_oos) -> None:
        X, y, names = separable_oos
        a = causal_mda(_ModelThatOnlyReadsColumnZero(), X, y, names, random_seed=5)
        b = causal_mda(_ModelThatOnlyReadsColumnZero(), X, y, names, random_seed=5)
        pd.testing.assert_frame_equal(a, b)


class TestFilterByMDA:
    def test_it_removes_exactly_the_features_marked_drop(self) -> None:
        X = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0], "c": [5.0, 6.0]})
        mda = pd.DataFrame({
            "feature": ["a", "b", "c"],
            "drop": [False, True, False],
        })
        assert list(filter_by_mda(X, mda).columns) == ["a", "c"]

    def test_it_preserves_the_original_column_order(self) -> None:
        X = pd.DataFrame({"z": [1.0], "a": [2.0], "m": [3.0]})
        mda = pd.DataFrame({"feature": ["a"], "drop": [True]})
        assert list(filter_by_mda(X, mda).columns) == ["z", "m"]

    def test_dropping_nothing_returns_every_column(self) -> None:
        X = pd.DataFrame({"a": [1.0], "b": [2.0]})
        mda = pd.DataFrame({"feature": ["a", "b"], "drop": [False, False]})
        assert list(filter_by_mda(X, mda).columns) == ["a", "b"]


# ═════════════════════════════════════════════════════════════════════════════
# SFI — de drempel doet werk
# ═════════════════════════════════════════════════════════════════════════════
class TestRankFeaturesBySFI:
    def test_only_features_above_the_threshold_survive_and_they_come_back_sorted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """De selectie- en sorteerlogica, los van de CatBoost-fit.

        `single_feature_importance` traint per feature per fold een model; die
        kosten horen niet in een unit-test van de DREMPEL. De AUC-tabel wordt
        daarom rechtstreeks aangeboden — de code die hier wordt getoetst is de
        filter- en sorteerstap, en die draait onveranderd.
        """
        import tradebot.selection.sfi as sfi_mod

        monkeypatch.setattr(
            sfi_mod, "single_feature_importance",
            lambda X, y, **kw: pd.Series({"good": 0.61, "edge": 0.52, "bad": 0.48}),
        )
        got = rank_features_by_sfi(pd.DataFrame(), np.array([]), min_auc=0.52)

        assert got == ["good"], "0,52 is de ondergrens en is strikt (>)"

    def test_lowering_the_threshold_admits_more_features(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import tradebot.selection.sfi as sfi_mod

        monkeypatch.setattr(
            sfi_mod, "single_feature_importance",
            lambda X, y, **kw: pd.Series({"good": 0.61, "edge": 0.52, "bad": 0.48}),
        )
        got = rank_features_by_sfi(pd.DataFrame(), np.array([]), min_auc=0.50)
        assert got == ["good", "edge"]


@pytest.mark.slow
class TestSingleFeatureImportanceEndToEnd:
    """Eén echte fit, zodat de CatBoost-weg niet uitsluitend op papier bestaat."""

    def test_a_predictive_feature_outscores_pure_noise(self) -> None:
        from tradebot.selection.sfi import single_feature_importance

        rng = np.random.default_rng(4)
        n = 400
        # Periodiek, niet monotoon: elke fold moet BEIDE klassen bevatten,
        # anders slaat de foldlus over en meet dit niets. Zie de test hieronder.
        signal = np.sin(np.linspace(0.0, 8.0 * np.pi, n))
        X = pd.DataFrame({"signal": signal, "noise": rng.normal(size=n)})
        y = (signal > 0).astype(int)

        auc = single_feature_importance(X, y, cv_splits=2, purge_bars=5)

        assert set(auc.index) == {"signal", "noise"}
        assert auc["signal"] > auc["noise"]
        assert list(auc) == sorted(auc, reverse=True)

    def test_a_monotone_label_leaves_every_feature_at_the_neutral_0_5(self) -> None:
        """Een gemeten degradatie die geen enkele foutmelding oplevert.

        Bij een label dat met de tijd meeloopt -- `y = signal > 0` op een
        oplopende reeks -- bevat elke trainfold maar één klasse. De lus slaat
        dan elke fold over en `feature_aucs[feat]` valt terug op **0,5 voor
        alles**, waarna elke feature even belangrijk lijkt.

        Dat is precies de faalvorm die de moduleheader beschrijft voor de
        weggehaalde `try/except` rond de catboost-import. De terugval is hier
        stil op een andere manier: hij komt uit de DATA, niet uit een import,
        en geen enkele poort merkt hem op. Vastgelegd zodat wie deze functie
        aanroept weet dat 0,5-over-de-hele-linie "niet gemeten" betekent en
        niet "geen signaal".
        """
        from tradebot.selection.sfi import single_feature_importance

        rng = np.random.default_rng(4)
        n = 400
        signal = np.linspace(-2.0, 2.0, n)      # monotoon
        X = pd.DataFrame({"signal": signal, "noise": rng.normal(size=n)})
        y = (signal > 0).astype(int)

        auc = single_feature_importance(X, y, cv_splits=2, purge_bars=5)

        assert auc.to_dict() == {"signal": 0.5, "noise": 0.5}
