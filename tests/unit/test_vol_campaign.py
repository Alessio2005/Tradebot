"""De H1-campagne: wat er per symbool gebeurt en hoe de trials worden geteld.

De bouwstenen staan in `tests/unit/test_vol_competition.py`. Dit bestand toetst
de campagne eromheen, en dan vooral het enige getal dat na deze fase permanent
in de governance blijft staan: **M**, het aantal uitgevoerde trials.

`fase_6_advanced_research.md` §0.9: *"M is heilig."* Een trial die wel is
gedraaid maar niet is geteld, maakt elke deflated Sharpe ratio erna te gunstig —
en dat is geen boekhoudkundige slordigheid maar een systematisch te optimistisch
oordeel over elk model dat daarna wordt getoetst.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradebot.cv.walk_forward import WalkForwardCV
from tradebot.validation.vol_campaign import (
    CampaignResult,
    build_proxy_panel,
    run_campaign,
)

SEED = 20260829


def _garch_panel(n: int = 700, symbols: tuple[str, ...] = ("AAA", "BBB")) -> pd.DataFrame:
    """Twee reeksen met een echte GARCH-structuur, zodat de fits convergeren."""
    index = pd.date_range("2021-01-01", periods=n, freq="D")
    columns = {}
    for offset, symbol in enumerate(symbols):
        rng = np.random.default_rng(SEED + offset)
        omega, alpha, beta = 2e-6, 0.09, 0.88
        var = np.empty(n)
        eps = np.empty(n)
        var[0] = omega / (1.0 - alpha - beta)
        for t in range(n):
            if t:
                var[t] = omega + alpha * eps[t - 1] ** 2 + beta * var[t - 1]
            eps[t] = rng.normal(0.0, np.sqrt(var[t]))
        columns[symbol] = eps
    return pd.DataFrame(columns, index=index)


def _proxies(returns: pd.DataFrame) -> dict[str, dict[str, pd.Series]]:
    return {
        symbol: {"squared_return": returns[symbol] ** 2}
        for symbol in returns.columns
    }


def _cv() -> WalkForwardCV:
    return WalkForwardCV(
        train_size=300, test_size=100, step=100, mode="rolling",
        min_train=300, embargo_bars=5)


def _run(arch_p_values: dict[str, float] | None = None) -> CampaignResult:
    returns = _garch_panel()
    return run_campaign(
        returns=returns, proxies=_proxies(returns),
        arch_p_values=arch_p_values or {"AAA": 0.001, "BBB": 0.001},
        cv=_cv(), primary_proxy="squared_return", horizons=(1,),
        specs=("garch",), seed=SEED, n_control_replicates=25)


@pytest.mark.slow
class TestRunCampaign:
    def test_every_symbol_variant_and_horizon_becomes_one_outcome(self) -> None:
        result = _run()
        assert len(result.outcomes) == 2
        assert {o.symbol for o in result.outcomes} == {"AAA", "BBB"}
        assert result.n_planned_trials == 2

    def test_a_fitted_combination_counts_as_a_trial(self) -> None:
        """Wat is gefit, telt. Dit is de M-boekhouding van de fase."""
        result = _run()
        assert result.n_trials == 2

    def test_a_gate_that_prevented_a_fit_does_not_count_as_a_trial(self) -> None:
        """De keerzijde, en die is even belangrijk.

        Op een reeks met een gesloten ARCH-poort is niets gefit en is dus geen
        parameterruimte doorzocht. Die combinatie meetellen zou M laten groeien
        door een meting die nooit heeft plaatsgevonden — dezelfde fout als de
        HAR-RV-trial die op nul 5m-dekking strandde en daarom niet meetelt.
        """
        result = _run({"AAA": 0.001, "BBB": 0.42})
        assert result.n_planned_trials == 2
        assert result.n_trials == 1
        descoped = [o for o in result.outcomes if o.symbol == "BBB"]
        assert descoped[0].verdict.status == "DESCOPED"
        assert descoped[0].convergence is None

    def test_the_controls_run_on_the_defender_once_per_symbol_and_horizon(
        self,
    ) -> None:
        result = _run()
        assert set(result.controls) == {"AAA|h1", "BBB|h1"}
        for control in result.controls.values():
            assert control.model.startswith("ewma")

    def test_the_record_carries_the_trial_accounting_and_the_controls(
        self,
    ) -> None:
        record = _run().as_record()
        assert record["n_trials"] == 2
        assert record["n_planned_trials"] == 2
        assert record["primary_proxy"] == "squared_return"
        assert set(record["controls"]) == {"AAA|h1", "BBB|h1"}
        assert len(record["outcomes"]) == 2

    def test_an_unknown_primary_proxy_crashes_instead_of_picking_one(
        self,
    ) -> None:
        """Stil een andere proxy kiezen zou de meetlat verruilen zonder dat
        iemand het in het rapport terugziet."""
        returns = _garch_panel()
        with pytest.raises(Exception, match="proxy"):
            run_campaign(
                returns=returns, proxies=_proxies(returns),
                arch_p_values={"AAA": 0.001, "BBB": 0.001}, cv=_cv(),
                primary_proxy="parkinson", horizons=(1,), specs=("garch",),
                seed=SEED, n_control_replicates=10)


class TestBuildProxyPanel:
    """De proxies die de campagne als realisatie gebruikt.

    Twee dingen moeten kloppen voordat een QLIKE-getal iets betekent: de proxy
    staat op DEZELFDE index als de returns die de modellen voorspellen, en de
    tabel in het rapport telt over ALLE symbolen en niet over het eerste.
    """

    def _ohlc(self, n: int = 50) -> dict[str, pd.DataFrame]:
        index = pd.date_range("2021-01-01", periods=n, freq="D")
        rng = np.random.default_rng(2)
        frames = {}
        for symbol in ("AAA", "BBB"):
            close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.02, size=n)))
            high = close * (1.0 + abs(rng.normal(0.0, 0.01, size=n)))
            low = close * (1.0 - abs(rng.normal(0.0, 0.01, size=n)))
            frames[symbol] = pd.DataFrame(
                {"open": close, "high": high, "low": low, "close": close},
                index=index)
        return frames

    def test_the_proxies_land_on_the_index_of_the_returns(self) -> None:
        ohlc = self._ohlc()
        index = ohlc["AAA"].index[1:]  # log-returns verliezen de eerste bar
        proxies, _ = build_proxy_panel(
            ohlc=ohlc, index=index, names=("parkinson", "squared_return"))
        assert set(proxies) == {"AAA", "BBB"}
        for per_symbol in proxies.values():
            assert set(per_symbol) == {"parkinson", "squared_return"}
            for series in per_symbol.values():
                assert series.index.equals(index)

    def test_the_records_count_every_symbol(self) -> None:
        ohlc = self._ohlc()
        index = ohlc["AAA"].index
        _, records = build_proxy_panel(
            ohlc=ohlc, index=index, names=("parkinson",))
        assert records["parkinson"]["n_symbols"] == 2
        assert records["parkinson"]["relative_efficiency_vs_squared_return"] > 1.0
        assert records["parkinson"]["n_valid_bars"] > len(index), (
            "De tabel telt maar één symbool; dan staat er een steekproef in het "
            "rapport die kleiner is dan de steekproef waarop is beslist.")

    def test_an_unknown_proxy_name_crashes(self) -> None:
        with pytest.raises(Exception, match="proxy"):
            build_proxy_panel(
                ohlc=self._ohlc(), index=self._ohlc()["AAA"].index,
                names=("er_bestaat_geen_ewma_proxy",))


@pytest.mark.slow
class TestTheCampaignMeasuresTheProxyPremise:
    """De campagne meet de premisse onder haar eigen meetlat.

    Zonder deze controle rapporteert zij promoties die volledig kunnen rusten op
    een niveauverschil tussen de proxy en de grootheid die de modellen
    voorspellen. Dat is geen theoretisch risico: de eerste echte H1-run mat
    range-proxies op 1,19 tot 2,24 maal de gemiddelde gekwadrateerde return.
    """

    def _run_with_primary(self, factor: float) -> CampaignResult:
        returns = _garch_panel()
        proxies = {
            symbol: {
                "squared_return": returns[symbol] ** 2,
                "inflated": factor * returns[symbol] ** 2,
            }
            for symbol in returns.columns
        }
        return run_campaign(
            returns=returns, proxies=proxies,
            arch_p_values={"AAA": 0.001, "BBB": 0.001}, cv=_cv(),
            primary_proxy="inflated", horizons=(1,), specs=("garch",),
            seed=SEED, n_control_replicates=10)

    def test_an_inflated_primary_proxy_blocks_every_verdict(self) -> None:
        result = self._run_with_primary(2.0)
        assert {o.verdict.status for o in result.outcomes} == {"UNPROVEN"}
        for outcome in result.outcomes:
            assert "proxy_premise_violated" in outcome.verdict.binding
            assert outcome.verdict.proxy_scale_ratio == pytest.approx(2.0, abs=0.01)

    def test_a_calibrated_primary_proxy_lets_the_verdict_through(self) -> None:
        result = self._run_with_primary(1.0)
        assert "UNPROVEN" not in {
            o.verdict.status for o in result.outcomes
            if "proxy_premise_violated" in o.verdict.binding
        } or all(
            "proxy_premise_violated" not in o.verdict.binding
            for o in result.outcomes
        )
        for outcome in result.outcomes:
            assert outcome.verdict.proxy_scale_ratio == pytest.approx(1.0, abs=0.01)

    def test_a_campaign_without_the_reference_proxy_crashes(self) -> None:
        """De referentie is niet optioneel.

        Zonder de gekwadrateerde return is de zuiverheid van de primaire proxy
        niet te meten, en dan zou de campagne een rangorde publiceren waarvan de
        premisse ongetoetst blijft.
        """
        returns = _garch_panel()
        proxies = {
            symbol: {"parkinson_stub": returns[symbol] ** 2}
            for symbol in returns.columns
        }
        with pytest.raises(Exception, match="squared_return"):
            run_campaign(
                returns=returns, proxies=proxies,
                arch_p_values={"AAA": 0.001, "BBB": 0.001}, cv=_cv(),
                primary_proxy="parkinson_stub", horizons=(1,), specs=("garch",),
                seed=SEED, n_control_replicates=10)


@pytest.mark.slow
class TestTheCampaignMeasuresForecastLevels:
    """Waarom het NIVEAU van een forecast in het artefact hoort.

    QLIKE straft een verkeerd niveau even hard als een verkeerde dynamiek. Wint
    een model op een proxy die zelf een niveauverschuiving draagt, dan moet de
    lezer kunnen zien of het model die verschuiving toevallig volgt. Zonder deze
    twee getallen is dat onderscheid niet te maken en blijft "GARCH verslaat
    EWMA" een uitspraak zonder mechanisme.
    """

    def test_both_forecast_levels_are_measured_against_the_unbiased_reference(
        self,
    ) -> None:
        returns = _garch_panel()
        result = run_campaign(
            returns=returns, proxies=_proxies(returns),
            arch_p_values={"AAA": 0.001, "BBB": 0.001}, cv=_cv(),
            primary_proxy="squared_return", horizons=(1,), specs=("garch",),
            seed=SEED, n_control_replicates=10)
        for outcome in result.outcomes:
            assert outcome.forecast_scale > 0.0
            assert outcome.baseline_forecast_scale > 0.0
            record = outcome.as_record()
            assert record["forecast_scale"] == pytest.approx(
                outcome.forecast_scale)
            assert record["baseline_forecast_scale"] == pytest.approx(
                outcome.baseline_forecast_scale)

    def test_the_highest_forecast_reaches_the_verdict(self) -> None:
        """Niet alleen het gemiddelde niveau, ook de PIEK.

        Een gemiddelde verbergt een handvol geëxplodeerde bars; het oordeel
        hangt aan de piek, want dat is waar een forecast ophoudt een forecast te
        zijn. Zonder deze doorgifte staat de poort in `judge_challenger` er wel,
        maar krijgt hij nooit een getal te zien.
        """
        returns = _garch_panel()
        result = run_campaign(
            returns=returns, proxies=_proxies(returns),
            arch_p_values={"AAA": 0.001, "BBB": 0.001}, cv=_cv(),
            primary_proxy="squared_return", horizons=(1,), specs=("garch",),
            seed=SEED, n_control_replicates=10)
        for outcome in result.outcomes:
            assert outcome.forecast_level_ratio >= outcome.forecast_scale
            assert outcome.verdict.forecast_level_ratio == pytest.approx(
                outcome.forecast_level_ratio)
            assert outcome.as_record()["forecast_level_ratio"] == pytest.approx(
                outcome.forecast_level_ratio)


@pytest.mark.slow
class TestTheControlsReachTheVerdict:
    def test_every_outcome_carries_the_control_of_its_own_series(self) -> None:
        """De controle draait per reeks en per horizon, en hoort dáár te binden.

        Zou de campagne één controle over alle symbolen gebruiken, dan zou een
        toets die op BTC wél iets kan zien, een uitspraak op LINK dekken waar
        hij dat niet kan.
        """
        returns = _garch_panel()
        result = run_campaign(
            returns=returns, proxies=_proxies(returns),
            arch_p_values={"AAA": 0.001, "BBB": 0.001}, cv=_cv(),
            primary_proxy="squared_return", horizons=(1,), specs=("garch",),
            seed=SEED, n_control_replicates=25)
        for outcome in result.outcomes:
            control = result.controls[f"{outcome.symbol}|h{outcome.horizon}"]
            assert outcome.verdict.power_control_passed is control.passed
