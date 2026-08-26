"""Smoothed probabilities zijn onbereikbaar vanuit elke backtest — deliverable 17.

Exit-criterium 3 van de fase. De audit is onvoorwaardelijk (§10.2):

    Smoothed probabilities `P(S_t | F_T)` gebruiken de volledige dataset.
    GEBRUIK IN BACKTESTS IS STRENG VERBODEN.

Dit bestand bewijst drie dingen, en het derde is het belangrijkste:

    1. geen enkele backtest- of conditioneringsmodule NOEMT de smoothed variant;
    2. een GEÏNJECTEERDE poging crasht met `CausalityViolationError`;
    3. de toets zelf wordt rood wanneer de blokkade wordt weggehaald.

Zonder (3) is dit bestand een groene test die alles goedkeurt, inclusief een
platform waarin de blokkade nooit heeft bestaan. `reports/phase5_exit_report.md`
§12 vond precies dat: een wiring-test met niet-bindende limietwaarden die dus
niets bewees.

WAAROM DIT VOOR DE EERSTE FIT MOET STAAN
=========================================
Stap 9 van de fase-opdracht: *"Bouw de blokkade voordat je het model op data
loslaat -- anders sluipt de smoothed-variant er tijdens exploratie in en is elk
daarna gemeten resultaat besmet."* Bij het schrijven van dit bestand is er nog
geen HMM gefit. Dat is opzet.

DE OMVANG VAN WAT HIER WORDT TEGENGEHOUDEN
===========================================
Niet theoretisch. Gemeten met identieke parameters op dezelfde 600 bars is
``max |P(S_t | F_T) - P(S_t | F_t)|`` ongeveer **0,52** -- meer dan een halve
eenheid kansmassa. Op zo'n verschil kantelt een regime-oordeel. De test hieronder
meet het opnieuw en eist dat het substantieel is, want was het verwaarloosbaar,
dan bewaakte deze hele constructie niets.
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from tradebot.regime.markov import (
    DIAGNOSTICS_ONLY,
    DiagnosticsToken,
    FilteredProbabilities,
    HmmParameters,
    HmmSpec,
    SmoothedProbabilities,
    forward_filter,
    require_filtered,
    smoothed_probabilities,
)
from tradebot.utils.failfast import CausalityViolationError

SEED = 20260826
SRC = Path(__file__).resolve().parents[2] / "src" / "tradebot"

#: De mappen waarlangs een besluit de authoritative engine bereikt. Wat hier
#: staat, kan een positie beïnvloeden; smoothed probabilities horen er dus niet.
BACKTEST_SURFACE = ("backtest", "alpha", "portfolio", "risk", "execution")

#: Namen die ALTIJD de smoothed route betekenen -- het zijn de onze.
FORBIDDEN_ALWAYS = (
    "smoothed_probabilities", "SmoothedProbabilities", "as_diagnostic_array",
)

#: Namen die alleen smoothed betekenen WANNEER ZE OP EEN HMM STAAN.
#:
#: DEFECT IN MIJN EIGEN WERK, gevonden bij de eerste run. De eerste versie
#: verbood `predict_proba` overal, en sloeg prompt aan op
#: `alpha/adaptive_wf.py`, waar het de `predict_proba` van een
#: CatBoostClassifier is -- volstrekt legitiem, en straks ook de kern van het
#: meta-labeling-spoor (H3). Een scan die zulke false positives geeft, wordt
#: binnen een week uitgezet of leeggemaakt, en dan bewaakt hij niets meer.
#:
#: `predict` staat er ook in: bij hmmlearn is dat VITERBI, het meest
#: waarschijnlijke pad over de HELE reeks, en dus net zo goed smoothed als de
#: posterior. Dat is precies de val waar `risk/hmm_regime.py` in is getrapt.
FORBIDDEN_ON_HMM = ("predict_proba", "score_samples", "predict")

#: Modules die de smoothed route noemen en dat (nog) mogen, met de reden.
#:
#: `risk/hmm_regime.py` is een BEKENDE, GEREGISTREERDE overtreding: hij gebruikt
#: `GaussianHMM.predict` (Viterbi) en `predict_proba`. Phase 0 heeft dat
#: gedocumenteerd als DI-1 en bewust niet gerepareerd, omdat die fase geen
#: modelgedrag wijzigde. Audit §24 eist REDESIGN naar het M2-contract; dat is
#: deliverable 16 van deze fase.
#:
#: Deze lijst is geen ontsnapping maar een TRIPWIRE: de test hieronder eist dat
#: hij exact deze ene entry bevat. Een tweede overtreding -- of het stilzwijgend
#: toevoegen van een module aan deze lijst -- maakt hem rood.
#: Paden staan met forward slashes, ongeacht platform -- zie `_scan_surface`.
REGISTERED_VIOLATIONS = {
    "risk/hmm_regime.py": "deliverable 16 (§24 REDESIGN); Phase 0 DI-1",
}


def _spec_and_params() -> HmmParameters:
    spec = HmmSpec(n_states=2)
    return HmmParameters(
        spec=spec, symbol="TEST", fold_id=0,
        start_prob=np.array([0.5, 0.5]),
        trans_mat=np.array([[0.95, 0.05], [0.05, 0.95]]),
        means=np.array([[0.0], [0.6]]), covars=np.array([[1.0], [1.0]]),
        converged=True, n_train_obs=400, loglikelihood=-1.0,
    )


def _overlapping_observations(n: int = 600) -> np.ndarray:
    """Overlappende toestanden. Op gescheiden toestanden valt er niets te smoothen
    en zou elke meting hier triviaal nul zijn -- zie de moduledocstring van
    `regime/markov.py`."""
    rng = np.random.default_rng(SEED)
    state = np.zeros(n, dtype=int)
    for t in range(1, n):
        state[t] = state[t - 1] if rng.random() < 0.95 else 1 - state[t - 1]
    return rng.normal(np.where(state == 0, 0.0, 0.6), 1.0)


# --------------------------------------------------------------------------- #
# 1. Statisch: het backtest-oppervlak noemt de smoothed route niet
# --------------------------------------------------------------------------- #
def _scan_source(source: str, *, filename: str = "<planted>") -> set[str]:
    """Welke verboden namen komen in deze bron voor?

    `predict_proba` en consorten tellen alleen mee wanneer de module ook
    `hmmlearn` importeert -- anders is het de `predict_proba` van een
    sklearn- of CatBoost-classifier en volkomen legitiem.
    """
    tree = ast.parse(source, filename=filename)
    names: set[str] = set()
    imports_hmmlearn = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
                "hmmlearn"):
            imports_hmmlearn = True
        elif isinstance(node, ast.Import) and any(
                a.name.startswith("hmmlearn") for a in node.names):
            imports_hmmlearn = True

    forbidden = set(FORBIDDEN_ALWAYS)
    if imports_hmmlearn:
        forbidden |= set(FORBIDDEN_ON_HMM)
    return names & forbidden


def _scan_surface() -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for package in BACKTEST_SURFACE:
        root = SRC / package
        if not root.is_dir():
            continue
        for module in root.rglob("*.py"):
            hits = _scan_source(
                module.read_text(encoding="utf-8"), filename=str(module))
            if hits:
                found[module.relative_to(SRC).as_posix()] = sorted(hits)
    return found


class TestNoBacktestPathMentionsSmoothed:
    def test_only_registered_violations_remain(self) -> None:
        unregistered = {
            path: hits for path, hits in _scan_surface().items()
            if path not in REGISTERED_VIOLATIONS
        }
        assert unregistered == {}, (
            "Deze modules kunnen een positie beïnvloeden en noemen de smoothed "
            "route zonder registratie:\n"
            + "\n".join(f"  {p} -> {h}" for p, h in unregistered.items()))

    def test_the_register_holds_exactly_the_one_known_violation(self) -> None:
        """De tripwire.

        `risk/hmm_regime.py` gebruikt Viterbi en `predict_proba` en staat als
        DI-1 geregistreerd; audit §24 eist REDESIGN naar het M2-contract en dat
        is deliverable 16 van deze fase. Zolang die er niet is, staat hij hier —
        zichtbaar, met reden, en niet als stilzwijgende uitzondering.

        Deze test wordt rood zodra er een TWEEDE overtreding bijkomt, en ook
        zodra iemand deze lijst uitbreidt om een nieuwe module door te laten.
        Hij wordt ook rood wanneer deliverable 16 klaar is — dan moet de entry
        eruit, en dat is precies de bedoeling.
        """
        found = set(_scan_surface())
        assert found == set(REGISTERED_VIOLATIONS), (
            f"verwacht precies de geregistreerde overtredingen "
            f"{sorted(REGISTERED_VIOLATIONS)}, kreeg {sorted(found)}")

    def test_the_scan_actually_scanned_something(self) -> None:
        """Negatieve controle op de scan zelf.

        Een AST-scan over nul bestanden is groen en bewijst niets. Phase 5 §12
        vond dit defecttype al een keer; hier wordt het uitgesloten.
        """
        scanned = [
            module for package in BACKTEST_SURFACE
            if (SRC / package).is_dir()
            for module in (SRC / package).rglob("*.py")
        ]
        assert len(scanned) >= 10, f"slechts {len(scanned)} modules gescand"

    def test_the_scan_would_find_a_planted_reference(self) -> None:
        """Negatieve controle: plant de overtreding en zie dat de scan hem vindt."""
        planted = (
            "from tradebot.regime.markov import smoothed_probabilities\n"
            "def build(obs, params):\n"
            "    return smoothed_probabilities(obs, params, None)\n"
        )
        assert _scan_source(planted), (
            "De scan ziet een geplante `smoothed_probabilities`-aanroep NIET. "
            "Dan zegt zijn groene uitkomst op de echte modules ook niets.")

    def test_the_scan_would_find_a_planted_viterbi_call(self) -> None:
        """En de Viterbi-variant, die niet naar smoothing KLINKT."""
        planted = (
            "from hmmlearn.hmm import GaussianHMM\n"
            "def build(model, X):\n"
            "    return model.predict(X)\n"
        )
        assert "predict" in _scan_source(planted)

    def test_a_catboost_predict_proba_is_not_flagged(self) -> None:
        """De false positive die de eerste versie van deze scan wél gaf.

        Het meta-labeling-spoor (H3) draait volledig op
        `CatBoostClassifier.predict_proba`. Zou de scan die blijven melden, dan
        wordt hij uitgezet of leeggemaakt — en dan bewaakt hij niets meer.
        """
        planted = (
            "from catboost import CatBoostClassifier\n"
            "def build(model, X):\n"
            "    return model.predict_proba(X)[:, 1]\n"
        )
        assert _scan_source(planted) == set()


# --------------------------------------------------------------------------- #
# 2. Dynamisch: een geïnjecteerde poging crasht
# --------------------------------------------------------------------------- #
class TestInjectedAttemptCrashes:
    def test_smoothed_object_is_rejected_by_the_gate(self) -> None:
        params = _spec_and_params()
        obs = _overlapping_observations()
        smoothed = smoothed_probabilities(obs, params, DIAGNOSTICS_ONLY)
        with pytest.raises(CausalityViolationError, match="STRENG VERBODEN"):
            require_filtered(smoothed, context="conditioning layer")

    def test_the_error_names_the_reason_the_smoothed_variant_was_created(
        self,
    ) -> None:
        """De traceback moet de volgende lezer vertellen WIE hem heeft gemaakt."""
        params = _spec_and_params()
        token = DiagnosticsToken(
            reason="expliciet gemarkeerde diagnostiek voor het benchmarkrapport")
        smoothed = smoothed_probabilities(
            _overlapping_observations(), params, token)
        with pytest.raises(CausalityViolationError) as excinfo:
            require_filtered(smoothed, context="engine")
        assert "diagnostiek voor het benchmarkrapport" in str(excinfo.value)

    def test_a_bare_array_is_rejected_too(self) -> None:
        """De belangrijkste variant van de aanval.

        Wie `as_diagnostic_array()` aanroept, houdt een gewone ndarray over, en
        daaraan is niet meer te zien waar hij vandaan komt. Zou de poort kale
        arrays doorlaten, dan is één methodeaanroep genoeg om de hele
        constructie te omzeilen -- en zou zij alleen tegen ongelukken
        beschermen, niet tegen de weg van de minste weerstand.
        """
        params = _spec_and_params()
        smoothed = smoothed_probabilities(
            _overlapping_observations(), params, DIAGNOSTICS_ONLY)
        with pytest.raises(CausalityViolationError, match="kale array"):
            require_filtered(
                smoothed.as_diagnostic_array(), context="engine")

    def test_smoothed_has_no_array_interface(self) -> None:
        """Hij kan niet stilzwijgend in numerieke code belanden."""
        params = _spec_and_params()
        smoothed = smoothed_probabilities(
            _overlapping_observations(), params, DIAGNOSTICS_ONLY)
        assert not hasattr(smoothed, "values")
        assert not hasattr(smoothed, "__array__")
        with pytest.raises(TypeError):
            np.asarray(smoothed, dtype=np.float64)

    def test_smoothed_without_a_token_is_refused(self) -> None:
        params = _spec_and_params()
        with pytest.raises(CausalityViolationError, match="zonder DiagnosticsToken"):
            smoothed_probabilities(
                _overlapping_observations(), params, None)  # type: ignore[arg-type]

    def test_a_token_without_a_real_reason_is_refused(self) -> None:
        with pytest.raises(CausalityViolationError, match="inhoudelijke reden"):
            DiagnosticsToken(reason="diagnostiek")

    def test_filtered_passes_the_gate(self) -> None:
        """De poort moet ook DOORLATEN; anders blokkeert hij alles en zegt niets."""
        params = _spec_and_params()
        filtered = forward_filter(_overlapping_observations(), params)
        assert require_filtered(filtered, context="engine") is filtered
        assert isinstance(filtered, FilteredProbabilities)


# --------------------------------------------------------------------------- #
# 3. De blokkade bewaakt iets van omvang
# --------------------------------------------------------------------------- #
class TestTheLeakBeingPreventedIsSubstantial:
    def test_smoothed_differs_materially_from_filtered(self) -> None:
        """Was het verschil verwaarloosbaar, dan bewaakte deze hele constructie niets.

        Dit getal hoort in `reports/M0_VS_HMM_BENCHMARK.md`: het maakt van §10.2
        een gemeten grootheid in plaats van een voorschrift.
        """
        params = _spec_and_params()
        obs = _overlapping_observations()
        filtered = forward_filter(obs, params)
        smoothed = smoothed_probabilities(obs, params, DIAGNOSTICS_ONLY)
        gap = smoothed.max_absolute_difference(filtered)
        assert gap > 0.10, (
            f"filtered en smoothed verschillen slechts {gap:.4f}. Dan is de "
            "blokkade misschien overbodig — of, waarschijnlijker, meet deze "
            "opzet het verschil niet.")

    def test_the_last_bar_is_the_one_place_they_agree(self) -> None:
        """``P(S_T | F_T)`` is per definitie zowel filtered als smoothed.

        Dat is de analytische controle op de forward-backward-implementatie:
        klopt de laatste rij niet, dan is de backward-pass fout — en dan zou de
        gemeten 0,52 hierboven een implementatiefout kunnen zijn in plaats van
        het lek.
        """
        params = _spec_and_params()
        obs = _overlapping_observations()
        filtered = forward_filter(obs, params)
        smoothed = smoothed_probabilities(obs, params, DIAGNOSTICS_ONLY)
        np.testing.assert_allclose(
            smoothed.as_diagnostic_array()[-1], filtered.values[-1], atol=1e-12)


# --------------------------------------------------------------------------- #
# 4. Filtered is werkelijk causaal
# --------------------------------------------------------------------------- #
class TestForwardFilterIsTruncationInvariant:
    @pytest.mark.parametrize("cut", [200, 400, 599])
    def test_appending_future_bars_changes_nothing(self, cut: int) -> None:
        params = _spec_and_params()
        obs = _overlapping_observations()
        full = forward_filter(obs, params).values[: cut + 1]
        short = forward_filter(obs[: cut + 1], params).values
        np.testing.assert_allclose(full, short, rtol=0.0, atol=0.0)

    def test_the_smoothed_variant_fails_the_same_test(self) -> None:
        """Negatieve controle: bewijs dat de truncatietest een lek DETECTEERT.

        De smoothed variant hoort hier rood te zijn. Was hij dat niet, dan zou
        de groene uitkomst voor de filtered variant hierboven niets bewijzen.
        """
        params = _spec_and_params()
        obs = _overlapping_observations()
        cut = 400
        full = smoothed_probabilities(
            obs, params, DIAGNOSTICS_ONLY).as_diagnostic_array()[: cut + 1]
        short = smoothed_probabilities(
            obs[: cut + 1], params, DIAGNOSTICS_ONLY).as_diagnostic_array()
        assert not np.allclose(full, short, rtol=0.0, atol=1e-9), (
            "De smoothed variant kwam ONGESCHONDEN door de truncatietest. Dan "
            "meet deze toets niets en zegt hij ook niets over de filtered "
            "variant.")

    def test_isinstance_is_the_gate_and_not_a_duck_type(self) -> None:
        """Een object dat zich VOORDOET als filtered komt er niet doorheen."""
        class _Impostor:
            values = np.zeros((10, 2))
            causal = True
            symbol = "TEST"

        with pytest.raises(CausalityViolationError):
            require_filtered(_Impostor(), context="engine")


class TestSmoothedProbabilitiesType:
    def test_it_is_not_a_filtered_instance(self) -> None:
        assert not issubclass(SmoothedProbabilities, FilteredProbabilities)
