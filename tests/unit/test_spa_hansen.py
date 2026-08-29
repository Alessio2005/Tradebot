"""Hansen's SPA-kern — `backtest/spa.py::spa_test`. Phase 7/8 Stage B-2.

WAAROM DIT BESTAND ER NIET WAS, EN WAT DAT KOSTTE
=================================================
`spa_test` bestaat sinds Wave 17 en had tot Stage B **nul tests**. Een grep over
de hele suite gaf één treffer, en dat was de module zelf. In die stilte kon een
defect drie fasen lang overleven:

De functie recentreerde de bootstrap met `np.maximum(d_bar, 0.0)` — Hansen's
**lower**-variant, de meest liberale van de drie — en gaf die terug onder de
sleutel `p_value_consistent`. Haar eigen docstring beloofde daarnaast
`p_value_lower` en `p_value_upper`; die sleutels werden nooit teruggegeven, dus
geen enkele aanroeper kon het verschil zien door de uitvoer te lezen.

Elk SPA-oordeel in dit platform is dus geveld met de meest permissieve schatter,
terwijl de rapporten de aanbevolen noemden. Dat is geen afrondingsverschil: op
een verzameling met kansloze kandidaten scheelt het hier 0,18 tegen 0,74.

DE DRIE VARIANTEN VERSCHILLEN IN ÉÉN DING
=========================================
Alleen in de recentreringsfunctie `g` die van de bootstrapgemiddelden wordt
afgetrokken:

    lower       g_l(d_bar) = max(d_bar, 0)
    consistent  g_c(d_bar) = d_bar * 1{d_bar >= -A_k}
    upper       g_u(d_bar) = d_bar

met `A_k = omega_k * sqrt(2 log log T / T)`.

Een kansloos model (`d_bar` diep negatief) draagt onder `lower` niets bij aan het
bootstrapmaximum, onder `upper` een volledige nulgemiddelde verdeling. `lower`
doet dus alsof de verliezers er niet zijn; `upper` neemt ze allemaal even
serieus. Hansen's `consistent` zit ertussen: het laat modellen vallen die
aantoonbaar hopeloos zijn en houdt de rest in de vergelijking.

Ref: Hansen (2005) §3.2; audit §17.1.
"""
from __future__ import annotations

import numpy as np
import pytest

from tradebot.backtest.spa import spa_test

#: Vast, zodat een gefaalde toets exact reproduceerbaar is.
SEED = 11
T = 400
BLOCK = 10
REPS = 1000


@pytest.fixture(scope="module")
def panel() -> dict[str, np.ndarray]:
    """Eén benchmark en drie kandidaten met scherp verschillende kwaliteit.

    De `hopeless`-kandidaat is het instrument van dit bestand: hij is het enige
    dat `lower` en `upper` uit elkaar trekt, en dus het enige waarmee te bewijzen
    is dat de drie p-waarden ook echt drie verschillende dingen berekenen.
    """
    rng = np.random.default_rng(SEED)
    bench = rng.normal(0.0, 0.02, T)
    return {
        "bench": bench,
        "good": bench + rng.normal(0.0015, 0.02, T),
        "marginal": bench + rng.normal(-0.0002, 0.02, T),
        "hopeless": bench + rng.normal(-0.05, 0.02, T),
    }


def _spa(bench: np.ndarray, *cols: np.ndarray) -> dict[str, object]:
    return spa_test(bench, np.column_stack(cols),
                    n_bootstrap=REPS, block_size=BLOCK)


class TestTheContractOfTheReturnValue:
    """De docstring beloofde drie p-waarden. Twee ervan bestonden niet."""

    def test_all_three_p_values_are_returned(self, panel) -> None:
        r = _spa(panel["bench"], panel["good"], panel["hopeless"])
        for key in ("p_value_lower", "p_value_consistent", "p_value_upper"):
            assert key in r, (
                f"{key} ontbreekt. De docstring belooft hem sinds Wave 17; wie "
                f"hem opvroeg kreeg een KeyError en wie hem niet opvroeg kreeg "
                f"stilzwijgend de liberale variant onder een andere naam.")

    def test_the_reject_flag_follows_the_consistent_variant(self, panel) -> None:
        """Niet de liberale. Dat was precies de verwisseling."""
        r = _spa(panel["bench"], panel["good"], panel["hopeless"])
        assert r["reject_null"] == (r["p_value_consistent"] < 0.05)


class TestHansenOrdering:
    def test_lower_le_consistent_le_upper(self, panel) -> None:
        r = _spa(panel["bench"], panel["good"], panel["marginal"],
                 panel["hopeless"])
        assert (r["p_value_lower"] <= r["p_value_consistent"]
                <= r["p_value_upper"]), (
            f"Hansen 3.2 ordent de drie strikt. Gekregen: "
            f"{r['p_value_lower']}, {r['p_value_consistent']}, "
            f"{r['p_value_upper']}.")

    def test_ordering_survives_a_pure_null(self) -> None:
        """De ordening is een eigenschap van g, niet van de data."""
        rng = np.random.default_rng(77)
        bench = rng.normal(0.0, 0.02, T)
        cols = [bench + rng.normal(0.0, 0.02, T) for _ in range(5)]
        r = spa_test(bench, np.column_stack(cols), n_bootstrap=REPS,
                     block_size=BLOCK)
        assert (r["p_value_lower"] <= r["p_value_consistent"]
                <= r["p_value_upper"])


class TestTheThreeVariantsMeasureThreeDifferentThings:
    """DE REGRESSIETEST OP HET GEVONDEN DEFECT.

    Vóór de reparatie was er één recentering en dus één getal. Deze klasse
    faalt zodra iemand daarnaar terugkeert, ook wanneer de drie sleutels blijven
    bestaan en alleen dezelfde waarde dragen.
    """

    def test_a_hopeless_candidate_moves_upper_and_leaves_lower_alone(
        self, panel
    ) -> None:
        """Het scherpste onderscheid dat er is.

        `lower` behandelt een kansloos model alsof het er niet is; `upper` telt
        het volledig mee. Een kandidaat toevoegen die de benchmark met 5 % per
        bar verliest, MOET `upper` verhogen en `lower` onaangeroerd laten. Doen
        beide hetzelfde, dan is er maar één recentering.

        De bootstrap-indices zijn identiek tussen beide aanroepen (`spa_test`
        seedt zijn eigen generator op 42), dus dit is een exacte vergelijking en
        geen steekproefverschil.
        """
        without = _spa(panel["bench"], panel["good"], panel["marginal"])
        with_ = _spa(panel["bench"], panel["good"], panel["marginal"],
                     panel["hopeless"])

        assert with_["p_value_lower"] == without["p_value_lower"], (
            "lower veranderde door een kandidaat die hij per definitie negeert")
        assert with_["p_value_upper"] > without["p_value_upper"], (
            f"upper bewoog niet ({without['p_value_upper']} -> "
            f"{with_['p_value_upper']}) terwijl er een kansloze kandidaat bij "
            f"kwam. De recentering is dan voor alle drie dezelfde en het defect "
            f"uit Stage B-2 is terug.")

    def test_the_consistent_threshold_drops_the_hopeless_candidate(
        self, panel
    ) -> None:
        """Wat `consistent` consistent maakt.

        `A_k = omega_k * sqrt(2 log log T / T)` schaalt met de eigen
        volatiliteit van het verschil. Een kandidaat die daar diep onder zit is
        aantoonbaar hopeloos en valt uit de vergelijking; een kandidaat die er
        net onder zit blijft erin. Precies daarom is `consistent` de aanbevolen
        schatter en niet een compromis tussen de twee andere.
        """
        without = _spa(panel["bench"], panel["good"], panel["marginal"])
        with_ = _spa(panel["bench"], panel["good"], panel["marginal"],
                     panel["hopeless"])
        assert with_["p_value_consistent"] == without["p_value_consistent"], (
            "de kansloze kandidaat veranderde de consistente p-waarde; de "
            "drempel A_k laat hem dan toch meewegen")

    def test_the_gap_widens_with_more_hopeless_candidates(self, panel) -> None:
        """Een enkel toeval kan één verschil opleveren; een trend niet."""
        rng = np.random.default_rng(5)
        bench = panel["bench"]
        gaps = []
        losers: list[np.ndarray] = []
        for _ in range(3):
            losers.append(bench + rng.normal(-0.05, 0.02, T))
            r = _spa(bench, panel["good"], *losers)
            gaps.append(float(r["p_value_upper"]) - float(r["p_value_lower"]))
        assert gaps == sorted(gaps) and gaps[-1] > gaps[0], (
            f"het gat tussen upper en lower groeit niet met het aantal kansloze "
            f"kandidaten: {gaps}")


class TestTheTestDoesNotFireOnNoise:
    """De negatieve controle. Een toets die ook op ruis significant is, meet
    niets — de faseregel die elke statistische conclusie hier bindt."""

    @pytest.mark.parametrize("seed", [3, 17, 101, 2026])
    def test_pure_noise_is_not_declared_superior(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        bench = rng.normal(0.0, 0.02, T)
        cols = [bench + rng.normal(0.0, 0.02, T) for _ in range(6)]
        r = spa_test(bench, np.column_stack(cols), n_bootstrap=REPS,
                     block_size=BLOCK)
        assert r["reject_null"] is False, (
            f"SPA verwierp de nul op zes kandidaten die per constructie geen "
            f"edge hebben (seed={seed}, p={r['p_value_consistent']})")

    def test_a_real_edge_is_detected(self) -> None:
        """De tegenhanger: een toets die nooit verwerpt, meet net zo min iets."""
        rng = np.random.default_rng(31)
        bench = rng.normal(0.0, 0.02, T)
        strong = bench + rng.normal(0.006, 0.02, T)
        r = spa_test(bench, strong.reshape(-1, 1), n_bootstrap=REPS,
                     block_size=BLOCK)
        assert r["p_value_consistent"] < 0.05, (
            f"een edge van 0,6 % per bar over {T} bars werd niet gevonden "
            f"(p={r['p_value_consistent']}) — de toets heeft dan geen "
            f"onderscheidend vermogen")


class TestDeterminism:
    def test_the_same_input_gives_the_same_p_values(self, panel) -> None:
        """`spa_test` seedt intern op 42. Een SPA-uitslag die per aanroep
        verschilt, is geen bewijs maar een trekking."""
        a = _spa(panel["bench"], panel["good"], panel["hopeless"])
        b = _spa(panel["bench"], panel["good"], panel["hopeless"])
        assert a == b

    def test_the_best_strategy_index_points_at_the_best_strategy(
        self, panel
    ) -> None:
        r = _spa(panel["bench"], panel["hopeless"], panel["good"],
                 panel["marginal"])
        assert r["best_strategy_idx"] == 1
