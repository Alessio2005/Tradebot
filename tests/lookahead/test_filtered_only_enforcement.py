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

WAT STAGE C-1 HIERAAN HEEFT TOEGEVOEGD
=======================================
Sectie 5 is nieuw en toetst niet het contract maar zijn consument in de
sizinglaag: `risk/hmm_regime.py`. Die module stond tot Stage C-1 als
GEREGISTREERDE overtreding in `REGISTERED_VIOLATIONS` hieronder -- hij gebruikte
`GaussianHMM.predict` (Viterbi) en `predict_proba` (smoothed). Die lijst is nu
leeg, en sectie 5 meet wat dat heeft opgeleverd: over 70 expanderende vensters
wijzigde het Viterbi-oordeel op de NIEUWSTE bar in 12 gevallen zodra er latere
bars bij kwamen, en het filtered oordeel in nul. Precies die nieuwste bar is de
waarde waarmee `portfolio/legacy_sizing.py` een LONG-positie halveert.

Daar wordt dus wel degelijk een HMM gefit. Dat spreekt de alinea hierboven niet
tegen: de blokkade stond er eerst, het model kwam erna.

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
import pandas as pd
import pytest
from hmmlearn.hmm import GaussianHMM

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
from tradebot.risk.hmm_regime import HMMRegimeDetector, Regime
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
#: STAGE C-1: DEZE LIJST IS LEEG, EN DAT IS HET RESULTAAT.
#:
#: Hier stond precies een entry -- `risk/hmm_regime.py`, die `GaussianHMM.predict`
#: (Viterbi) en `predict_proba` (smoothed) gebruikte. Phase 0 heeft dat als DI-1
#: geregistreerd en bewust niet gerepareerd, omdat die fase geen modelgedrag
#: wijzigde; audit §24 eiste REDESIGN naar het M2-contract. Die redesign staat er
#: nu: `hmmlearn` doet nog uitsluitend de EM-schatting, en alle inferentie loopt
#: door `regime.markov.forward_filter`. Sectie 5 hieronder meet dat het verschil
#: echt is en dat het weggehaalde pad dezelfde toets NIET doorstaat.
#:
#: De lijst blijft een TRIPWIRE, en staat leeg scherper dan gevuld: elke entry is
#: een pad waarlangs `P(S_t | F_T)` een positie kan bereiken. Een nieuwe regel
#: hier is een besluit dat in `docs/ARCHITECTURAL_DECISIONS.md` hoort, niet een
#: dat in een testbestand ontstaat.
#: Paden staan met forward slashes, ongeacht platform -- zie `_scan_surface`.
REGISTERED_VIOLATIONS: dict[str, str] = {}


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

    def test_the_register_is_empty_now_that_di1_is_closed(self) -> None:
        """De tripwire, en hij is precies gesprongen zoals bedoeld.

        Deze test eiste tot Stage C-1 EXACT een entry: `risk/hmm_regime.py`, met
        DI-1 als geregistreerde reden. Hij is rood geworden op het moment dat die
        module naar het filtered contract werd herschreven — de vorige versie
        schreef dat er letterlijk bij: *"Hij wordt ook rood wanneer deliverable
        16 klaar is — dan moet de entry eruit, en dat is precies de bedoeling."*

        Vanaf hier eist hij dat het backtest-oppervlak de smoothed route NERGENS
        meer noemt. Hij wordt rood zodra er een overtreding bijkomt, en net zo
        goed zodra iemand er een probeert te legaliseren door de lijst te vullen.
        """
        assert REGISTERED_VIOLATIONS == {}, (
            "De lijst met toegestane overtredingen is niet leeg. Elke entry is "
            "een pad waarlangs P(S_t | F_T) een positie kan bereiken; DI-1 is "
            "gesloten en er hoort geen opvolger te ontstaan.")
        found = set(_scan_surface())
        assert found == set(), (
            f"het backtest-oppervlak noemt de smoothed route weer: "
            f"{sorted(found)}")

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


# --------------------------------------------------------------------------- #
# 5. De regime-detector van de sizinglaag loopt door dezelfde poort
#    STAGE C-1 -- DI-1
# --------------------------------------------------------------------------- #
#: Bars voor de fit. Op een korter venster degenereert de EM-schatting op deze
#: reeks (gemeten op 600 bars: een niet-positief-definiete covariantie, waarop
#: `hmmlearn` in zijn eigen Cholesky crasht). De detector eist er minimaal 50;
#: dat is een ondergrens voor identificeerbaarheid, geen garantie.
_HMM_BARS = 800

#: Kans dat het regime blijft staan. Lager en de toestand wisselt zo vaak dat er
#: niets te schatten valt; veel hoger en er valt niets te beslissen. 0,97 geeft
#: een reeks waarin de toestanden ECHT overlappen, en dat is de enige situatie
#: waarin filtered en smoothed uit elkaar lopen -- zie de moduledocstring van
#: `regime/markov.py`.
_HMM_PERSISTENCE = 0.97

#: Eigen seed: `SEED` hierboven hoort bij de tweetoestandsreeks van sectie 2-4.
_HMM_SEED = 20260829

#: De vensters van de sweep hieronder: expanderend, zoals een backtest ze ziet.
_HMM_CUTS = range(100, _HMM_BARS, 10)


def _regime_returns(n: int = _HMM_BARS) -> pd.Series:
    """Drie toestanden met elk een eigen drift en een eigen volatiliteit.

    Bear/Flat/Bull zoals `hmm_regime.Regime` ze kent, met parameters in de orde
    van dagelijkse cryptoreturns. De reeks is niet bedoeld als realistische
    markt maar als data waarop de EM-schatting drie onderscheidbare toestanden
    vindt en de posterior toch onzeker genoeg blijft om het verschil tussen
    filtered en Viterbi zichtbaar te maken.
    """
    rng = np.random.default_rng(_HMM_SEED)
    state = np.zeros(n, dtype=int)
    for t in range(1, n):
        state[t] = (
            state[t - 1] if rng.random() < _HMM_PERSISTENCE
            else int(rng.integers(0, 3))
        )
    mu = np.array([-0.008, 0.0, 0.008])[state]
    sigma = np.array([0.030, 0.012, 0.018])[state]
    return pd.Series(
        rng.normal(mu, sigma),
        index=pd.date_range("2020-01-01", periods=n, freq="D"),
    )


@pytest.fixture(scope="module")
def fitted_detector() -> HMMRegimeDetector:
    """Een fit voor de hele module; de EM-stap kost ongeveer een halve seconde."""
    return HMMRegimeDetector().fit(_regime_returns())


def _viterbi_twin(detector: HMMRegimeDetector) -> GaussianHMM:
    """Herbouwt het pad dat Stage C-1 heeft VERWIJDERD, uit dezelfde parameters.

    Dit is de enige plek in de repository waar `GaussianHMM.predict` nog wordt
    aangeroepen, en hij staat hier met een reden: zonder hem bewijst
    `test_labels_do_not_change_when_later_bars_arrive` alleen dat er iets
    draait. Pas wanneer diezelfde toets op de weggehaalde route ROOD wordt, meet
    zij het lek in plaats van de aanwezigheid van code.

    Het grijpt bewust in de privetoestand van de detector. Dat is het punt:
    beide routes draaien op EXACT dezelfde bevroren parameters, zodat het
    gemeten verschil de inferentie is en niet de fit.
    """
    params = detector._params
    assert params is not None
    twin = GaussianHMM(n_components=3, covariance_type="full")
    twin.startprob_ = params.start_prob
    twin.transmat_ = params.trans_mat
    twin.means_ = params.means
    twin.covars_ = params.covars
    return twin


class TestTheDetectorHasNoViterbiSurfaceLeft:
    def test_the_method_that_returned_viterbi_no_longer_exists(self) -> None:
        """`predict` is niet hernoemd maar weg -- ook als naam.

        De naam was de helft van het probleem: `detector.predict(...)` las als
        "voorspel", en de aanroep eronder was `GaussianHMM.predict`. Bleef de
        naam bestaan als alias, dan blijft elke oude aanroep werken en zegt geen
        enkele diff dat het contract is gewijzigd.
        """
        assert not hasattr(HMMRegimeDetector, "predict")
        assert not hasattr(HMMRegimeDetector, "predict_proba")
        assert hasattr(HMMRegimeDetector, "filtered_regimes")

    def test_no_hmmlearn_model_survives_the_fit(
        self, fitted_detector: HMMRegimeDetector,
    ) -> None:
        """Wat niet wordt vastgehouden, kan later niet worden aangeroepen.

        Zolang het `GaussianHMM`-object op de detector blijft staan, is
        `self._model.predict(X)` een regel verderop -- en dat is precies de
        route die hier is gesloten.
        """
        assert not hasattr(fitted_detector, "_model")
        assert isinstance(fitted_detector._params, HmmParameters)

    def test_the_probabilities_pass_the_gate_of_this_file(
        self, fitted_detector: HMMRegimeDetector,
    ) -> None:
        labels, filtered = fitted_detector.filtered_regimes(_regime_returns())
        assert require_filtered(filtered, context="hmm regime detector") is filtered
        assert isinstance(filtered, FilteredProbabilities)
        assert filtered.values.shape == (len(labels), 3)

    def test_current_regime_is_the_argmax_of_the_last_filtered_row(
        self, fitted_detector: HMMRegimeDetector,
    ) -> None:
        """De waarde die `portfolio/legacy_sizing.py` opvraagt, en niets anders.

        Dit legt vast WELKE regel het regime bepaalt: de marginale argmax op de
        laatste bar. Onder Viterbi was dat de laatste stap van een pad dat over
        de hele reeks was geoptimaliseerd, en die twee vallen niet samen.
        """
        returns = _regime_returns()
        labels, filtered = fitted_detector.filtered_regimes(returns)
        state = int(np.argmax(filtered.values[-1]))
        expected = Regime(fitted_detector._state_map[state])
        assert fitted_detector.current_regime(returns) == expected
        assert Regime(int(labels[-1])) == expected


class TestTheDetectorIsTruncationInvariant:
    @pytest.mark.parametrize("cut", [200, 400, _HMM_BARS - 1])
    def test_labels_do_not_change_when_later_bars_arrive(
        self, fitted_detector: HMMRegimeDetector, cut: int,
    ) -> None:
        """Dezelfde toets als sectie 4, nu op de detector zelf.

        Bit-exact, niet bij benadering: het forward-algoritme rekent op de
        prefix letterlijk dezelfde recursie uit, dus elke afwijking -- ook in de
        laatste decimaal -- betekent dat er informatie van later is meegekomen.
        """
        returns = _regime_returns()
        labels_full, filtered_full = fitted_detector.filtered_regimes(returns)
        labels_short, filtered_short = fitted_detector.filtered_regimes(
            returns.iloc[: cut + 1])
        n = len(labels_short)
        np.testing.assert_array_equal(labels_full[:n], labels_short)
        np.testing.assert_allclose(
            filtered_full.values[:n], filtered_short.values, rtol=0.0, atol=0.0)

    def test_the_viterbi_path_it_replaced_fails_that_same_test(
        self, fitted_detector: HMMRegimeDetector,
    ) -> None:
        """NEGATIEVE CONTROLE -- en dit is de kern van dit blok.

        Dezelfde bevroren parameters, dezelfde bars, de andere inferentie.
        Gemeten over 70 expanderende vensters (bar 100 t/m 790, stap 10):

            Viterbi:  het oordeel op de NIEUWSTE bar wijzigt in 12 van de 70
                      vensters zodra er latere bars bij komen;
            filtered: in 0 van de 70.

        Die nieuwste bar is niet zomaar een bar. Het is de enige waarde die
        `portfolio/legacy_sizing.py` opvraagt, via `current_regime`, om een
        LONG-positie wel of niet te halveren. Onder de oude route hing die
        halvering dus in ongeveer een op de zes vensters af van koersen die op
        dat moment nog niet bestonden.
        """
        returns = _regime_returns()
        twin = _viterbi_twin(fitted_detector)
        features = fitted_detector._features(returns)
        viterbi_full = twin.predict(features)

        viterbi_changes = sum(
            int(twin.predict(features[:cut])[-1] != viterbi_full[cut - 1])
            for cut in _HMM_CUTS
        )
        assert viterbi_changes > 0, (
            "Het Viterbi-pad kwam ONGESCHONDEN door de truncatietest. Dan meet "
            "die toets niets, en zegt haar groene uitkomst op het filtered pad "
            "hierboven evenmin iets.")

        labels_full, _ = fitted_detector.filtered_regimes(returns)
        filtered_changes = sum(
            int(fitted_detector.filtered_regimes(returns.iloc[:cut])[0][-1]
                != labels_full[cut - 2])
            for cut in _HMM_CUTS
        )
        assert filtered_changes == 0, (
            f"Het filtered pad veranderde zijn oordeel op de nieuwste bar in "
            f"{filtered_changes} van de {len(_HMM_CUTS)} vensters. Dan is het "
            "niet causaal en is DI-1 niet gesloten.")

    def test_the_two_routes_disagree_on_a_material_share_of_the_bars(
        self, fitted_detector: HMMRegimeDetector,
    ) -> None:
        """Het verschil is een ander oordeel, geen andere berekening.

        Viterbi maximaliseert de kans op de GEZAMENLIJKE reeks; de filtered
        argmax maximaliseert de kans per bar. Gemeten op deze 799 bars lopen ze
        op 124 ervan uiteen -- ruim een op de zeven. Was dat aandeel
        verwaarloosbaar, dan zou het herontwerp cosmetisch zijn en zou de
        registratie van DI-1 als lookaheadlek niet kloppen.
        """
        returns = _regime_returns()
        _, filtered = fitted_detector.filtered_regimes(returns)
        twin = _viterbi_twin(fitted_detector)
        viterbi = twin.predict(fitted_detector._features(returns))
        share = float(np.mean(viterbi != filtered.most_likely_state))
        assert share > 0.05, (
            f"filtered en Viterbi verschillen op slechts {share:.1%} van de "
            "bars; controleer of deze reeks wel overlappende toestanden heeft.")
