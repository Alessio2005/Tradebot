"""Pariteitsbewijs tussen de vectorized conventie en de authoritative engine.

Fase-opdracht §14, exit-criterium 2. Het bewijs dat vóór verwijdering van de
legacy-engines moet liggen.

WAT PARITEIT HIER KAN BETEKENEN
-------------------------------
`reports/phase5_engine_diff.md` §3 legt uit waarom "alle vier de engines geven
hetzelfde getal" geen toetsbare claim is: engine 1 en 2 beslissen op TRADES
(entry tot barrier, op een event-grid), engine 3 en 4 op EXPOSURES (per bar, op
een bar-grid). Er bestaat geen invoer waarop die twee hetzelfde horen te doen.

Wat wél toetsbaar is, en wat hier wordt bewezen:

1. De engine reproduceert de vectorized conventie zodra je hun enige
   semantische verschil - het executiemoment - gelijktrekt.
2. Dat verschil is precies één bar, en de richting is de conservatieve.
3. Wat er daarna nog tussen zit, is de kwantiteit-versus-gewicht-drift, en die
   is gemeten en begrensd in plaats van weggemiddeld.

HET GEMETEN VERSCHIL
--------------------
Per-bar RMSE tussen de rendementsreeks van de engine en de vectorized conventie
op elke lag, over 60 bars en drie onafhankelijke reeksen:

    seed          lag 1      lag 2      lag 3
    1            17,49 bp    1,26 bp   20,38 bp
    42           19,50 bp    1,03 bp   19,03 bp
    20260825     18,71 bp    1,06 bp   22,43 bp

`shift(2)` is de conventie van de engine, met een scheiding van 15-20x ten
opzichte van beide buren. Dat is geen toeval en geen kalibratie:
`baseline_runner.py` hanteert `held = weights.shift(1)`, wat neerkomt op
*"beslis op de close van t, voer uit op DIE close, verdien vanaf t+1"*. De
engine voert uit op de prijs van `t+1` - dat is wat `venue.latency_bars = 1`
betekent - en verdient dus vanaf `t+2`.

**De vectorized baseline neemt aan dat je kunt handelen op de close waarop je
besluit.** Dat is de stilzwijgende aanname die deze fase zichtbaar maakt.

WAAROM HET SIGNAAL MOET BEWEGEN
-------------------------------
De eerste versie van deze suite draaide op een constante `a_t = 1`. Daarmee
levert vol-targeting een bijna vlakke gewichtenreeks op, en verschillen
`shift(1)` en `shift(2)` nauwelijks: de RMSE lag op 0,448 tegen 0,455 bp en op
een van de drie seeds koos de test de VERKEERDE lag. Een pariteitstest die
groen kan zijn zonder de eigenschap aan te tonen, is geen bewijs. De replay
gebruikt daarom `varying_exposure=True`.
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tradebot.tca.post_trade import close_tca_roundtrip

from ._engine_fixtures import (
    make_engine,
    make_replay,
    tca_tolerance_bps,
)

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[2]

#: Residu op de juiste lag, als per-bar RMSE in bp PER EENHEID BRUTO-EXPOSURE.
#: Dit is de kwantiteit-versus-gewicht-drift: de engine houdt tussen fills een
#: vast AANTAL stuks, waarvan het gewicht meebeweegt met de prijs en met de
#: equity, plus de partial fills en de dust-drempel. Een gewichtenbacktest
#: herweegt per constructie elke bar gratis en kent geen van beide.
#:
#: WAAROM GENORMALISEERD EN NIET ABSOLUUT. Deze drempel stond op een absolute
#: 3,0 bp, gemeten op een boek dat door het PROPFIRM-risicobudget op een gross
#: van ~0,08 werd gehouden. Toen dat mandaat verviel (`docs/RISK_MANDATE.md`) en
#: `sigma_target` van 0,08 naar 0,20 ging, schaalde de gross met exact 2,5x mee
#: en het residu daarmee ook — de test werd rood zonder dat de engine of de
#: shift(2)-conventie was veranderd.
#:
#: Gemeten over drie seeds x {60, 120} bars, in beide risicoregimes:
#:
#:   regime            gross        rmse2/gross     rmse1/gross
#:   propfirm          0,082-0,088  11,7 - 13,0     226 - 260
#:   eigen kapitaal    0,204-0,221  13,6 - 15,0     225 - 260
#:
#: De genormaliseerde grootheid is stabiel over een factor 2,5 aan boekgrootte;
#: de absolute was dat per constructie niet. 25,0 laat ~67% marge boven de
#: slechtste waarneming en blijft ~9x onder het lag-1-residu, dus de test
#: onderscheidt de juiste lag nog even scherp.
DRIFT_RMSE_TOLERANCE_BPS_PER_GROSS = 25.0

#: Minimale scheiding tussen de juiste lag en zijn buren. Zakt die eronder, dan
#: is de latency niet meer effectief en toetst de suite niets.
MIN_LAG_SEPARATION = 5.0


def _weights_from_decisions(result, replay) -> pd.DataFrame:
    """De gewichten die de soevereine laag heeft toegestaan, als panel.

    Beide kanten van de vergelijking krijgen DEZELFDE besluiten. Zou de
    vectorized kant zijn eigen gewichten construeren, dan zou de test het
    verschil tussen twee risicoregimes meten in plaats van tussen twee
    executieconventies.
    """
    symbols = list(replay.prices.columns)
    frame = pd.DataFrame(
        0.0, index=pd.DatetimeIndex([s.ts for s in replay.slices]),
        columns=symbols,
    )
    decision_ts = [s.ts for s in replay.slices[: len(result.decisions)]]
    for ts, decision in zip(decision_ts, result.decisions, strict=True):
        for symbol, weight in decision.permitted_exposure.items():
            frame.at[ts, symbol] = weight
    return frame


def _vectorized_returns(weights: pd.DataFrame, prices: pd.DataFrame,
                        *, lag: int) -> np.ndarray:
    """De vectorized conventie: `gross_t = sum(w_{t-lag} * r_t)`."""
    returns = prices.pct_change(fill_method=None).fillna(0.0)
    return (weights.shift(lag).fillna(0.0) * returns).sum(axis=1).to_numpy()


def _rmse_bps(result, weights: pd.DataFrame, prices: pd.DataFrame,
              *, lag: int) -> float:
    """Per-bar RMSE tussen engine en vectorized, in basispunten.

    Per BAR en niet op de eindequity: twee verschillende conventies kunnen
    toevallig op hetzelfde eindpunt uitkomen, maar niet op hetzelfde pad. Een
    eerdere versie vergeleek eindequity en koos daardoor op één van drie seeds
    de verkeerde lag.
    """
    engine = result.returns.to_numpy()
    vector = _vectorized_returns(weights, prices, lag=lag)
    return float(np.sqrt(np.mean((engine - vector) ** 2))) * 1e4


def _rmse_by_lag(result, weights, prices) -> dict[int, float]:
    return {lag: _rmse_bps(result, weights, prices, lag=lag) for lag in (1, 2, 3)}


def _mean_gross(result) -> float:
    """Gemiddelde bruto-exposure over het pad, uit de besluiten zelf.

    De normalisator van `DRIFT_RMSE_TOLERANCE_BPS_PER_GROSS`. Gemiddeld en niet
    laatste-bar, omdat `_rmse_bps` een RMSE over het hele pad is; een
    eindbar-gross zou een pad met wisselende exposure verkeerd normaliseren.
    """
    grosses = [
        sum(abs(v) for v in d.permitted_exposure.values()) for d in result.decisions
    ]
    assert grosses, "geen besluiten; er is niets te normaliseren"
    mean = sum(grosses) / len(grosses)
    assert mean > 0.0, (
        "het boek draagt nul bruto-exposure; de drift-tolerantie is dan niet "
        "genormaliseerd te toetsen en de test zou alles doorlaten"
    )
    return mean


def _drift_tolerance_bps(result) -> float:
    return DRIFT_RMSE_TOLERANCE_BPS_PER_GROSS * _mean_gross(result)


@pytest.fixture(scope="module")
def setup():
    replay = make_replay(n_bars=60, varying_exposure=True)
    result = make_engine().run(replay.slices, replay.exposures)
    weights = _weights_from_decisions(result, replay)
    closure = close_tca_roundtrip(result, tolerance_bps=tca_tolerance_bps())
    return replay, result, weights, closure


# =========================================================================== #
# 1. De engine reproduceert de vectorized conventie op de juiste lag
# =========================================================================== #
class TestExecutionTimingParity:
    def test_lag_two_is_the_matching_convention(self, setup) -> None:
        """De engine is één bar later dan de vectorized `shift(1)`-conventie."""
        replay, result, weights, _ = setup
        rmse = _rmse_by_lag(result, weights, replay.prices)
        assert min(rmse, key=lambda k: rmse[k]) == 2, (
            f"de engine komt niet overeen met shift(2); rmse={rmse}"
        )

    def test_the_residual_at_lag_two_is_only_the_drift(self, setup) -> None:
        replay, result, weights, _ = setup
        rmse = _rmse_bps(result, weights, replay.prices, lag=2)
        tolerance = _drift_tolerance_bps(result)
        assert rmse < tolerance, (
            f"residu {rmse:.3f} bp overschrijdt de kwantiteit-versus-gewicht-"
            f"drift van {tolerance:.3f} bp "
            f"({DRIFT_RMSE_TOLERANCE_BPS_PER_GROSS} bp x gemiddelde gross "
            f"{_mean_gross(result):.4f})"
        )

    def test_the_execution_timing_actually_costs_something(self, setup) -> None:
        """De latency is geen cosmetische parameter.

        Als de scheiding tussen lag 1 en lag 2 verdwijnt, is de engine
        feitelijk teruggevallen op de vectorized aanname dat je handelt op de
        close waarop je besluit - en dan is exit-criterium 8 leeg.
        """
        replay, result, weights, _ = setup
        rmse = _rmse_by_lag(result, weights, replay.prices)
        assert rmse[1] / rmse[2] > MIN_LAG_SEPARATION
        assert rmse[3] / rmse[2] > MIN_LAG_SEPARATION

    def test_the_gross_book_is_the_arrival_price_book(self, setup) -> None:
        """De vergelijking loopt via het schaduwboek, niet via de nettocurve.

        Een vectorized gross-berekening kent geen spread, impact of fee. Het
        schaduwboek uit de TCA-roundtrip is exact dat: dezelfde fills op
        arrival prices zonder fee.
        """
        _, result, _, closure = setup
        assert closure.arrival_equity > float(result.equity_curve.iloc[-1]), (
            "het bruto boek is niet groter dan het netto boek; de kosten "
            "worden niet afgetrokken"
        )
        assert closure.gap == pytest.approx(
            closure.decomposition.execution_total, abs=closure.tolerance)


# =========================================================================== #
# 2. De kostenlagen zijn additief en toewijsbaar
# =========================================================================== #
class TestCostAttribution:
    def test_net_equals_gross_minus_the_named_costs(self, setup) -> None:
        _, result, _, closure = setup
        d = closure.decomposition
        reconstructed = closure.arrival_equity - d.spread - d.impact - d.fees
        assert reconstructed == pytest.approx(
            float(result.equity_curve.iloc[-1]), abs=closure.tolerance)

    def test_every_cost_layer_is_strictly_positive(self, setup) -> None:
        """Een laag die nul is, is een laag die niet is aangesloten."""
        _, _, _, closure = setup
        d = closure.decomposition
        assert d.spread > 0.0
        assert d.impact > 0.0
        assert d.fees > 0.0
        assert d.funding != 0.0


# =========================================================================== #
# 3. Determinisme en reproduceerbaarheid van het bewijs zelf
# =========================================================================== #
class TestTheProofIsReproducible:
    def test_the_same_replay_gives_the_same_parity_numbers(self) -> None:
        replay = make_replay(n_bars=60, varying_exposure=True)
        a = make_engine().run(replay.slices, replay.exposures)
        b = make_engine().run(replay.slices, replay.exposures)
        ca = close_tca_roundtrip(a, tolerance_bps=tca_tolerance_bps())
        cb = close_tca_roundtrip(b, tolerance_bps=tca_tolerance_bps())
        assert ca.arrival_equity == cb.arrival_equity
        assert ca.realised_equity == cb.realised_equity

    @pytest.mark.parametrize("seed", [1, 42, 20260825])
    def test_the_lag_two_result_holds_across_independent_series(
        self, seed: int
    ) -> None:
        """Niet één gelukkige reeks: de conventie is een eigenschap van de engine."""
        replay = make_replay(n_bars=60, seed=seed, varying_exposure=True)
        result = make_engine().run(replay.slices, replay.exposures)
        weights = _weights_from_decisions(result, replay)
        rmse = _rmse_by_lag(result, weights, replay.prices)
        assert min(rmse, key=lambda k: rmse[k]) == 2, f"rmse={rmse}"
        assert rmse[2] < _drift_tolerance_bps(result), (
            f"rmse={rmse}, gemiddelde gross={_mean_gross(result):.4f}")
        assert rmse[1] / rmse[2] > MIN_LAG_SEPARATION


# =========================================================================== #
# 4. De legacy-engines: wat er nog van bestaat, en wat dat waard is
# =========================================================================== #
def _modules_importing(needle: str) -> list[str]:
    """Modules die `needle` daadwerkelijk IMPORTEREN.

    Alleen echte imports tellen. Een vermelding in een docstring is
    documentatie, geen consument; een eerdere versie van deze test greep op de
    platte tekst en rekende `_kernels.py` en `spread.py` mee omdat hun
    moduledocstring de afhankelijkheid beschrijft.
    """
    hits: set[str] = set()
    for path in list((ROOT / "src").rglob("*.py")) + list((ROOT / "apps").rglob("*.py")):
        if "__pycache__" in str(path) or path.stem == needle:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and needle in node.module:
                hits.add(str(path.relative_to(ROOT)))
            elif isinstance(node, ast.Import) and any(needle in a.name for a in node.names):
                hits.add(str(path.relative_to(ROOT)))
    return sorted(hits)


class TestLegacyEngineInventory:
    def test_per_side_has_no_production_callers(self) -> None:
        """`reports/phase5_engine_diff.md` §2.6: engine 1 is dode code.

        Dat is de reden dat hij zonder pariteitsbewijs mag verdwijnen: er is
        geen gedrag dat behouden moet blijven.
        """
        assert not _modules_importing("per_side")

    def test_no_module_still_imports_a_removed_engine(self) -> None:
        for engine in ("per_side", "bidirectional"):
            assert not _modules_importing(engine), (
                f"{engine} wordt nog geimporteerd; de verwijdering is niet "
                f"compleet"
            )

    def test_the_authoritative_engine_is_the_only_one_left(self) -> None:
        """Exit-criterium 1: exact één authoritative event-driven backtester."""
        backtest = ROOT / "src" / "tradebot" / "backtest"
        removed = ("per_side.py", "bidirectional.py", "portfolio.py")
        still_there = [f for f in removed if (backtest / f).is_file()]
        assert not still_there, (
            f"legacy-engines bestaan nog: {still_there}; verwijderen mag pas "
            f"na dit pariteitsbewijs, maar dan ook echt"
        )
        assert (backtest / "engine.py").is_file()
        assert (backtest / "vectorized.py").is_file()
