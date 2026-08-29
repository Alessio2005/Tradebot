"""De QLIKE-competitie H1 — de bouwstenen, apart toetsbaar. Stap 7.

De zware wetenschap staat elders en is daar getoetst: `volatility/garch.py`
(fits en forecasts), `validation/vol_metrics.py` (QLIKE, MZ, DM-HLN) en
`volatility/realized.py` (de proxies). Dit bestand toetst wat de competitie
DAARBOVENOP doet, en dat is precies waar een competitie stukgaat:

  * de uitlijning van de EWMA-forecast op de bar die hij voorspelt;
  * welke bars meedoen — uitsluitend de testvensters van de walk-forward, want
    een titelverdediger die ook op de trainbars scoort, wint op een andere
    steekproef dan zijn uitdager;
  * de gemeten power, want de pre-registratie maakt het verschil tussen
    `FALSIFIED` en `UNPROVEN` afhankelijk van een GETAL en niet van een indruk;
  * de oordeelslogica zelf: vier stop-criteria met een vaste volgorde.

Ref: `conf/research/preregistration_h1_garch_vs_ewma.yaml`;
`Prompts-fases/fase_6_advanced_research.md` stap 7.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradebot.cv.walk_forward import WalkForwardCV
from tradebot.validation.vol_campaign import (
    ChallengerOutcome,
    run_symbol_competition,
)
from tradebot.validation.vol_competition import (
    EXPECTED_EFFECT_SD,
    H1Verdict,
    NegativeControls,
    build_losses,
    ewma_variance_forecast,
    judge_challenger,
    mde_sd_units,
    oos_mask,
    proxy_scale_ratio,
    run_negative_controls,
    shuffled_forecast,
    swap_control_rejection_rate,
)
from tradebot.validation.vol_metrics import (
    DieboldMarianoResult,
    LossSeries,
    diebold_mariano_hln,
)

LAM = 0.94
BURN_IN = 60


def _returns(n: int = 300, *, spike_at: int | None = None) -> pd.Series:
    """Rustige reeks met eventueel één uitschieter, zodat de uitlijning van de
    forecast zichtbaar wordt in plaats van te moeten worden geloofd."""
    values = np.full(n, 0.01, dtype=np.float64)
    values[::2] = -0.01
    if spike_at is not None:
        values[spike_at] = 0.5
    return pd.Series(
        values, index=pd.date_range("2021-01-01", periods=n, freq="D"))


# --------------------------------------------------------------------------- #
# 1. De EWMA-forecast staat op de bar die hij voorspelt
# --------------------------------------------------------------------------- #
class TestEwmaVarianceForecast:
    def test_the_forecast_for_a_bar_does_not_contain_that_bar(self) -> None:
        """De hele competitie hangt hieraan.

        Zou de forecast op bar `t` de return van bar `t` bevatten, dan wint
        EWMA de QLIKE-competitie met een lek in plaats van met een model — en
        dat lek zou er precies uitzien als een goede estimator.
        """
        spike = 200
        forecast = ewma_variance_forecast(
            _returns(spike_at=spike), lam=LAM, burn_in_bars=BURN_IN, horizon=1)
        quiet = float(forecast.iloc[spike - 1])
        at_spike = float(forecast.iloc[spike])
        after_spike = float(forecast.iloc[spike + 1])
        assert at_spike == pytest.approx(quiet, rel=0.05), (
            "De forecast VOOR de uitschieterbar veranderde toen de uitschieter "
            "werd toegevoegd; hij gebruikt dus de return die hij voorspelt.")
        assert after_spike > 10.0 * at_spike, (
            "De uitschieter komt niet terug in de forecast van de bar erna. "
            "Dan is de reeks verkeerd om verschoven en loopt de forecast "
            "achter in plaats van vooruit.")

    @pytest.mark.parametrize("horizon", [1, 5])
    def test_appending_later_bars_changes_nothing(self, horizon: int) -> None:
        full = ewma_variance_forecast(
            _returns(300), lam=LAM, burn_in_bars=BURN_IN, horizon=horizon)
        short = ewma_variance_forecast(
            _returns(300).iloc[:220], lam=LAM, burn_in_bars=BURN_IN,
            horizon=horizon)
        np.testing.assert_allclose(
            full.to_numpy()[:220], short.to_numpy(), rtol=0.0, atol=0.0)

    @pytest.mark.parametrize("horizon", [1, 5])
    def test_the_first_bars_carry_no_forecast(self, horizon: int) -> None:
        """Tijdens de burn-in bestaat er geen schatting; er wordt niet gevuld."""
        forecast = ewma_variance_forecast(
            _returns(300), lam=LAM, burn_in_bars=BURN_IN, horizon=horizon)
        assert forecast.iloc[: BURN_IN + horizon].isna().all()
        assert forecast.iloc[BURN_IN + horizon :].notna().all()

    def test_a_longer_horizon_shifts_the_same_information_further(self) -> None:
        """RiskMetrics is vlak: de h-staps forecast IS de huidige variantie.

        Dat is geen benadering maar de IGARCH-structuur van EWMA. Het legt vast
        dat h=5 dezelfde getallen vijf bars later plaatst en niet stilzwijgend
        met vijf vermenigvuldigt — dat zou een variantie over vijf bars zijn en
        de QLIKE-competitie beslissen op een eenheidsfout.
        """
        returns = _returns(300)
        h1 = ewma_variance_forecast(
            returns, lam=LAM, burn_in_bars=BURN_IN, horizon=1)
        h5 = ewma_variance_forecast(
            returns, lam=LAM, burn_in_bars=BURN_IN, horizon=5)
        np.testing.assert_allclose(
            h5.to_numpy()[104:120], h1.to_numpy()[100:116], rtol=0.0, atol=0.0)


# --------------------------------------------------------------------------- #
# 2. Alleen de testvensters doen mee
# --------------------------------------------------------------------------- #
class TestOosMask:
    def _cv(self) -> WalkForwardCV:
        return WalkForwardCV(
            train_size=500, test_size=100, step=100, mode="rolling",
            min_train=500, embargo_bars=5)

    def test_it_is_exactly_the_union_of_the_test_windows(self) -> None:
        cv = self._cv()
        mask = oos_mask(1742, cv)
        expected = np.zeros(1742, dtype=bool)
        for fold in cv.split(1742):
            expected[fold.test_idx] = True
        np.testing.assert_array_equal(mask, expected)

    def test_the_embargo_zone_is_not_out_of_sample(self) -> None:
        """De bars tussen het geëmbargeerde trainvenster en het testvenster
        horen bij GEEN van beide. Ze meetellen zou de competitie beslechten op
        bars die de walk-forward-structuur bewust heeft uitgesloten."""
        mask = oos_mask(1742, self._cv())
        assert not mask[495:500].any()
        assert mask[500]

    def test_no_bar_before_the_first_training_window_is_scored(self) -> None:
        mask = oos_mask(1742, self._cv())
        assert not mask[:500].any()

    def test_the_certified_window_yields_the_preregistered_twelve_folds(
        self,
    ) -> None:
        """1.200 OOS-bars: exact het getal waarop de power-analyse rust."""
        assert int(oos_mask(1742, self._cv()).sum()) == 1200


# --------------------------------------------------------------------------- #
# 3. De gemeten power, zoals de pre-registratie hem definieert
# --------------------------------------------------------------------------- #
class TestMdeSdUnits:
    #: De drie scenario's uit `preregistration_h1_garch_vs_ewma.yaml`.
    @pytest.mark.parametrize(
        ("ar1", "expected"), [(0.0, 0.072), (0.2, 0.088), (0.4, 0.110)])
    def test_it_reproduces_the_preregistered_scenarios(
        self, ar1: float, expected: float,
    ) -> None:
        mde = mde_sd_units(n_obs=1200, ar1=ar1, effective_series=1.2651339524254563)
        assert mde == pytest.approx(expected, abs=0.001)

    def test_serial_correlation_costs_power(self) -> None:
        assert mde_sd_units(n_obs=1200, ar1=0.4) > mde_sd_units(n_obs=1200, ar1=0.0)

    def test_negative_autocorrelation_never_buys_power(self) -> None:
        """Een negatieve AR(1) verkleint de variantie van het gemiddelde echt,
        maar die winst wordt hier NIET geclaimd: de designfactor stopt op 1.
        Anders zou een toevallig negatieve autocorrelatie een onderpowerde
        toets tot `FALSIFIED` kunnen promoveren."""
        assert mde_sd_units(n_obs=1200, ar1=-0.5) == pytest.approx(
            mde_sd_units(n_obs=1200, ar1=0.0))


# --------------------------------------------------------------------------- #
# 4. De geschudde forecast van de negatieve controle
# --------------------------------------------------------------------------- #
class TestShuffledForecast:
    def test_it_keeps_the_values_and_destroys_the_timing(self) -> None:
        values = np.arange(1.0, 101.0)
        values[3] = np.nan
        shuffled = shuffled_forecast(values, seed=7)
        np.testing.assert_array_equal(np.isnan(shuffled), np.isnan(values))
        np.testing.assert_allclose(
            np.sort(shuffled[~np.isnan(shuffled)]),
            np.sort(values[~np.isnan(values)]))
        assert not np.allclose(
            shuffled[~np.isnan(shuffled)], values[~np.isnan(values)])

    def test_the_same_seed_gives_the_same_permutation(self) -> None:
        values = np.arange(1.0, 51.0)
        np.testing.assert_array_equal(
            shuffled_forecast(values, seed=3), shuffled_forecast(values, seed=3))
        assert not np.array_equal(
            shuffled_forecast(values, seed=3), shuffled_forecast(values, seed=4))


# --------------------------------------------------------------------------- #
# 5. De oordeelslogica — vier criteria, vaste volgorde
# --------------------------------------------------------------------------- #
def _dm(*, p_value: float, mean_diff: float, ar1: float = 0.1,
        n_obs: int = 1200) -> DieboldMarianoResult:
    """Een DM-resultaat met precies de velden waar het oordeel aan hangt."""
    return DieboldMarianoResult(
        model_a="garch", model_b="ewma_0.94", horizon=1,
        mean_loss_differential=mean_diff, dm_statistic=mean_diff * -10.0,
        hln_statistic=mean_diff * -10.0, p_value=p_value, n_obs=n_obs,
        loss_differential_ar1=ar1, n_dropped_to_intersection=0,
        alternative="two-sided")


class TestJudgeChallenger:
    def test_a_significant_improvement_with_power_and_convergence_promotes(
        self,
    ) -> None:
        verdict = judge_challenger(
            arch_p_value=0.001, convergence_ratio=1.0, boundary_ratio=0.0,
            dm=_dm(p_value=0.01, mean_diff=-0.05))
        assert verdict.status == "PROMOTED"
        assert verdict.binding == ()

    def test_no_significance_with_a_powered_test_falsifies(self) -> None:
        verdict = judge_challenger(
            arch_p_value=0.001, convergence_ratio=1.0, boundary_ratio=0.0,
            dm=_dm(p_value=0.4, mean_diff=-0.001, ar1=0.05))
        assert verdict.status == "FALSIFIED"
        assert "no_significant_qlike_improvement" in verdict.binding

    def test_no_significance_on_an_underpowered_test_is_unproven(self) -> None:
        """Het verschil tussen `FALSIFIED` en `UNPROVEN` is een getal.

        Bij een AR(1) van 0,6 is het minimaal detecteerbare effect groter dan
        het verwachte effect van 0,10 SD; deze opzet KAN het effect dan niet
        zien, en afwezigheid van bewijs is geen bewijs van afwezigheid. No-go 8
        van de fase verbiedt hier een falsificatie.
        """
        verdict = judge_challenger(
            arch_p_value=0.001, convergence_ratio=1.0, boundary_ratio=0.0,
            dm=_dm(p_value=0.4, mean_diff=-0.001, ar1=0.6, n_obs=300))
        assert verdict.status == "UNPROVEN"
        assert "underpowered_test_cannot_falsify" in verdict.binding

    def test_significance_in_favour_of_ewma_never_promotes(self) -> None:
        """De valkuil van een tweezijdige toets.

        Een p-waarde onder 0,05 zegt "de twee verschillen", niet "de uitdager
        wint". Wordt alleen op significantie gekeken, dan promoveert een
        variant die AANTOONBAAR SLECHTER is dan EWMA.
        """
        verdict = judge_challenger(
            arch_p_value=0.001, convergence_ratio=1.0, boundary_ratio=0.0,
            dm=_dm(p_value=0.001, mean_diff=+0.05))
        assert verdict.status == "FALSIFIED"

    def test_a_closed_arch_gate_descopes_before_anything_else(self) -> None:
        """Zonder conditionele heteroskedasticiteit is een GARCH-structuur niet
        gerechtvaardigd; er valt dan niets te falsifiëren."""
        verdict = judge_challenger(
            arch_p_value=0.42, convergence_ratio=1.0, boundary_ratio=0.0,
            dm=_dm(p_value=0.01, mean_diff=-0.05))
        assert verdict.status == "DESCOPED"
        assert verdict.binding[0] == "arch_gate_no_conditional_heteroskedasticity"

    def test_too_few_converged_fits_descopes(self) -> None:
        """Een QLIKE-reeks uit 70 % van de vensters is een SELECTIE van de
        gemakkelijke vensters en niet vergelijkbaar."""
        verdict = judge_challenger(
            arch_p_value=0.001, convergence_ratio=0.7, boundary_ratio=0.0,
            dm=_dm(p_value=0.01, mean_diff=-0.05))
        assert verdict.status == "DESCOPED"
        assert "convergence_too_low_to_compare" in verdict.binding

    def test_too_many_boundary_solutions_descopes(self) -> None:
        verdict = judge_challenger(
            arch_p_value=0.001, convergence_ratio=1.0, boundary_ratio=0.5,
            dm=_dm(p_value=0.01, mean_diff=-0.05))
        assert verdict.status == "DESCOPED"
        assert "convergence_too_low_to_compare" in verdict.binding

    def test_a_missing_comparison_is_unproven_and_never_falsified(self) -> None:
        """Geen toets is geen bewijs. `dm=None` ontstaat wanneer de doorsnede
        te klein was om te vergelijken."""
        verdict = judge_challenger(
            arch_p_value=0.001, convergence_ratio=1.0, boundary_ratio=0.0,
            dm=None)
        assert verdict.status == "UNPROVEN"

    def test_the_verdict_carries_the_measured_numbers(self) -> None:
        verdict = judge_challenger(
            arch_p_value=0.001, convergence_ratio=0.95, boundary_ratio=0.02,
            dm=_dm(p_value=0.4, mean_diff=-0.001, ar1=0.05))
        record = verdict.as_record()
        assert record["status"] == "FALSIFIED"
        assert record["mde_sd_units"] > 0.0
        assert record["expected_effect_sd_units"] == EXPECTED_EFFECT_SD
        assert record["dm_p_value"] == pytest.approx(0.4)

    def test_it_is_a_frozen_verdict_object(self) -> None:
        verdict = judge_challenger(
            arch_p_value=0.001, convergence_ratio=1.0, boundary_ratio=0.0,
            dm=_dm(p_value=0.01, mean_diff=-0.05))
        assert isinstance(verdict, H1Verdict)
        with pytest.raises(AttributeError):
            verdict.status = "PROMOTED"  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# 6. De verliesreeksen zien uitsluitend de out-of-sample bars
# --------------------------------------------------------------------------- #
def _rv_and_forecast(n: int = 400) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(11)
    variance = 0.0004 * np.exp(rng.normal(0.0, 0.4, size=n))
    rv = np.maximum(variance * rng.chisquare(df=1, size=n), 1e-12)
    return rv, variance


class TestBuildLosses:
    def test_bars_outside_the_mask_carry_no_loss(self) -> None:
        """De titelverdediger mag niet op meer bars scoren dan zijn uitdager.

        EWMA heeft na de burn-in op ELKE bar een waarde, ook op de trainbars.
        Zou hij daar meescoren, dan wordt hij op een andere steekproef beoordeeld
        dan de GARCH-variant -- inclusief de bars waarop die per constructie niet
        beoordeeld MAG worden.
        """
        rv, forecast = _rv_and_forecast()
        mask = np.zeros(rv.size, dtype=bool)
        mask[100:200] = True
        losses = build_losses(rv, forecast, model="ewma", mask=mask)
        assert set(losses) == {"qlike", "mse_sd", "mae_sd"}
        for loss in losses.values():
            assert loss.n_usable <= 100
            assert np.isnan(loss.values[:100]).all()
            assert np.isnan(loss.values[200:]).all()

    def test_the_full_length_is_kept_so_two_models_stay_comparable(self) -> None:
        rv, forecast = _rv_and_forecast()
        losses = build_losses(
            rv, forecast, model="ewma", mask=np.ones(rv.size, dtype=bool))
        assert losses["qlike"].values.size == rv.size
        assert losses["qlike"].n_total == rv.size


# --------------------------------------------------------------------------- #
# 7. De negatieve controle op de TOETS zelf
# --------------------------------------------------------------------------- #
def _as_loss(values: np.ndarray, model: str) -> LossSeries:
    """Een `LossSeries` rond een kant-en-klare verliesreeks."""
    return LossSeries(
        name="qlike", model=model, values=np.asarray(values, dtype=np.float64),
        n_total=int(values.size), n_usable=int(np.isfinite(values).sum()),
        n_dropped_proxy=0, n_dropped_forecast=0)


def _loss_pair(n: int = 1200) -> tuple[LossSeries, LossSeries]:
    """Twee verliesreeksen waarvan de eerste ECHT beter is."""
    rng = np.random.default_rng(5)
    base = rng.gamma(shape=2.0, scale=0.5, size=n)
    return _as_loss(base * 0.85, "challenger"), _as_loss(base, "ewma")


class TestSwapControl:
    """Verwerpt de toets ook wanneer er niets te vinden is?

    De opzet: neem twee verliesreeksen waarvan de ene aantoonbaar beter is, en
    wissel per bar met kans een half welke van de twee hij is. Het verwachte
    verliesverschil is daarmee per constructie NUL, terwijl schaal, staarten en
    seriële structuur blijven staan. Verwerpt DM-HLN daar veel vaker dan zijn
    eigen alpha, dan verwerpt hij op ruis en zegt een significante uitkomst op
    de echte data niets.
    """

    def test_the_real_difference_is_found(self) -> None:
        """Zonder dit is de controle hieronder betekenisloos: een toets die
        NOOIT verwerpt, haalt elke negatieve controle."""
        a, b = _loss_pair()
        result = diebold_mariano_hln(a, b, horizon=1)
        assert result.significant
        assert result.better == "challenger"

    def test_shuffling_which_model_is_which_stops_the_rejections(self) -> None:
        a, b = _loss_pair()
        rate = swap_control_rejection_rate(
            a, b, horizon=1, n_replicates=200, seed=20260829)
        assert rate <= 0.10, (
            f"DM-HLN verwerpt in {rate:.1%} van de replicaties terwijl het "
            "verwachte verschil nul is. Bij alpha = 0,05 hoort dat rond 5 % te "
            "liggen; hoger betekent dat de toets op ruis verwerpt.")


# --------------------------------------------------------------------------- #
# 8. De campagne per symbool
# --------------------------------------------------------------------------- #
def _garch_returns(n: int = 700) -> pd.Series:
    """Een reeks met een ECHTE GARCH-structuur, zodat de fits convergeren."""
    rng = np.random.default_rng(20260829)
    omega, alpha, beta = 2e-6, 0.09, 0.88
    var = np.empty(n)
    eps = np.empty(n)
    var[0] = omega / (1.0 - alpha - beta)
    for t in range(n):
        if t:
            var[t] = omega + alpha * eps[t - 1] ** 2 + beta * var[t - 1]
        eps[t] = rng.normal(0.0, np.sqrt(var[t]))
    return pd.Series(eps, index=pd.date_range("2021-01-01", periods=n, freq="D"))


def _small_cv() -> WalkForwardCV:
    return WalkForwardCV(
        train_size=300, test_size=100, step=100, mode="rolling",
        min_train=300, embargo_bars=5)


@pytest.mark.slow
class TestRunSymbolCompetition:
    """Een symbool, een variant, een horizon -- op synthetische GARCH-data.

    Klein gehouden: het punt is de BEDRADING (uitlijning, steekproef, oordeel),
    niet de econometrie. Die staat in de toetsen van `volatility/garch.py` en
    `validation/vol_metrics.py`.
    """

    def test_it_returns_one_outcome_per_spec_and_horizon(self) -> None:
        returns = _garch_returns()
        proxy = pd.Series(returns.to_numpy() ** 2, index=returns.index)
        outcomes = run_symbol_competition(
            returns=returns, proxy=proxy, symbol="TEST", cv=_small_cv(),
            horizons=(1,), arch_p_value=0.001, specs=("garch",))

        assert len(outcomes) == 1
        outcome = outcomes[0]
        assert isinstance(outcome, ChallengerOutcome)
        assert outcome.symbol == "TEST"
        assert outcome.horizon == 1
        assert outcome.verdict.status in {
            "PROMOTED", "FALSIFIED", "UNPROVEN", "DESCOPED"}
        assert outcome.convergence.n_fits > 0

    def test_both_models_are_scored_on_the_same_bars(self) -> None:
        """De kern van een eerlijke competitie."""
        returns = _garch_returns()
        proxy = pd.Series(returns.to_numpy() ** 2, index=returns.index)
        cv = _small_cv()
        outcome = run_symbol_competition(
            returns=returns, proxy=proxy, symbol="TEST", cv=cv,
            horizons=(1,), arch_p_value=0.001, specs=("garch",))[0]

        oos = int(oos_mask(returns.size, cv).sum())
        assert outcome.baseline_losses["qlike"].n_usable <= oos
        assert outcome.losses["qlike"].n_usable <= oos
        assert (outcome.baseline_losses["qlike"].values.size
                == outcome.losses["qlike"].values.size)

    def test_a_closed_arch_gate_descopes_without_fitting_anything(self) -> None:
        """De poort staat VOOR de fit, niet erna.

        Zou de campagne eerst fitten en pas daarna de poort lezen, dan is de
        de-scope een administratieve handeling achteraf en heeft de reeks toch
        een QLIKE-uitslag gekregen die iemand kan citeren.
        """
        returns = _garch_returns()
        proxy = pd.Series(returns.to_numpy() ** 2, index=returns.index)
        outcomes = run_symbol_competition(
            returns=returns, proxy=proxy, symbol="TEST", cv=_small_cv(),
            horizons=(1,), arch_p_value=0.42, specs=("garch",))

        assert len(outcomes) == 1
        assert outcomes[0].verdict.status == "DESCOPED"
        assert outcomes[0].convergence is None
        assert outcomes[0].dm is None


    def test_extra_proxies_are_scored_without_touching_the_verdict(self) -> None:
        """De robuustheidstabel mag het oordeel niet kunnen verschuiven.

        QLIKE is proxy-robuust (Patton 2011): elke conditioneel zuivere proxy
        hoort dezelfde rangorde te geven. Dat is een BEWERING over de data, en
        om haar te kunnen toetsen moeten de extra proxies op exact dezelfde
        forecasts worden gescoord als de primaire. Zouden zij hun eigen fits
        krijgen, dan meet de tabel het verschil tussen twee campagnes in plaats
        van tussen twee proxies.
        """
        returns = _garch_returns()
        squared = returns.to_numpy() ** 2
        proxy = pd.Series(squared, index=returns.index)
        cv = _small_cv()
        rng = np.random.default_rng(3)
        noisier = pd.Series(squared * rng.gamma(2.0, 0.5, size=squared.size),
                            index=returns.index)

        plain = run_symbol_competition(
            returns=returns, proxy=proxy, symbol="TEST", cv=cv,
            horizons=(1,), arch_p_value=0.001, specs=("garch",))[0]
        with_extra = run_symbol_competition(
            returns=returns, proxy=proxy, symbol="TEST", cv=cv,
            horizons=(1,), arch_p_value=0.001, specs=("garch",),
            robustness_proxies={"noisier": noisier})[0]

        assert plain.robustness == {}
        assert set(with_extra.robustness) == {"noisier"}
        assert with_extra.verdict.status == plain.verdict.status
        if plain.dm is not None:
            assert with_extra.dm is not None
            assert with_extra.dm.p_value == plain.dm.p_value
            assert (with_extra.robustness["noisier"].p_value
                    != plain.dm.p_value), (
                "De tweede proxy gaf exact dezelfde p-waarde als de primaire. "
                "Dan is hij niet gescoord maar gekopieerd.")


# --------------------------------------------------------------------------- #
# 9. De twee controles samen: kan de toets iets zien, en ziet hij niets te veel?
# --------------------------------------------------------------------------- #
class TestNegativeControls:
    def test_the_test_detects_a_forecast_whose_timing_is_destroyed(self) -> None:
        """De POWER-kant van de controle.

        Een geschudde forecast heeft exact dezelfde marginale verdeling als het
        origineel en heeft alleen zijn timing verloren — precies de eigenschap
        waarop de competitie hoort te beslissen. Ziet DM-HLN dát verschil niet,
        dan kan hij ook het veel kleinere verschil tussen GARCH en EWMA niet
        zien, en betekent elke niet-significante uitslag in dit rapport niets.
        """
        rv, forecast = _rv_and_forecast(1200)
        controls = run_negative_controls(
            rv=rv, forecast=forecast, mask=np.ones(rv.size, dtype=bool),
            horizon=1, seed=20260829, n_replicates=100, model="ewma_0.94")
        assert controls.shuffle_dm.significant
        assert controls.shuffle_dm.better == "ewma_0.94"

    def test_it_does_not_reject_when_there_is_nothing_to_find(self) -> None:
        """De SIZE-kant. Zie `swap_control_rejection_rate`."""
        rv, forecast = _rv_and_forecast(1200)
        controls = run_negative_controls(
            rv=rv, forecast=forecast, mask=np.ones(rv.size, dtype=bool),
            horizon=1, seed=20260829, n_replicates=200, model="ewma_0.94")
        assert 0.0 <= controls.swap_rejection_rate <= 0.10
        assert controls.passed

    def test_a_record_carries_both_sides(self) -> None:
        rv, forecast = _rv_and_forecast(600)
        record = run_negative_controls(
            rv=rv, forecast=forecast, mask=np.ones(rv.size, dtype=bool),
            horizon=1, seed=1, n_replicates=50, model="ewma_0.94").as_record()
        assert record["shuffle_control"]["significant"] is True
        assert 0.0 <= record["swap_rejection_rate"] <= 1.0
        assert isinstance(record["passed"], bool)

    def test_it_is_a_frozen_result(self) -> None:
        rv, forecast = _rv_and_forecast(600)
        controls = run_negative_controls(
            rv=rv, forecast=forecast, mask=np.ones(rv.size, dtype=bool),
            horizon=1, seed=1, n_replicates=20, model="ewma_0.94")
        assert isinstance(controls, NegativeControls)
        with pytest.raises(AttributeError):
            controls.swap_rejection_rate = 0.0  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# 10. De premisse onder de proxy — gemeten, niet aangenomen
# --------------------------------------------------------------------------- #
class TestProxyScaleRatio:
    """QLIKE is alleen proxy-robuust als de proxy CONDITIONEEL ZUIVER is.

    Die zin staat in de pre-registratie als rechtvaardiging voor de
    range-estimator, met de aanname erbij: *"de range-estimators zijn dat onder
    een driftloze GBM binnen de dag"*. Dat is een PREMISSE, en een premisse is
    toetsbaar.

    Waarom het uitmaakt: QLIKE heeft zijn minimum op ``forecast = E[proxy]``.
    Zit er een multiplicatieve factor `c` in de proxy, dan verschuift dat
    minimum mee, en wint het model waarvan het NIVEAU toevallig bij `c` past —
    ongeacht of het de dynamiek beter voorspelt. De rangorde meet dan
    kalibratie tegen een verschoven doel in plaats van voorspelkwaliteit.
    """

    def test_an_unbiased_proxy_scores_one(self) -> None:
        rng = np.random.default_rng(4)
        sigma2 = 0.0004 * np.exp(rng.normal(0.0, 0.3, size=5000))
        squared_return = sigma2 * rng.chisquare(df=1, size=5000)
        assert proxy_scale_ratio(
            sigma2, squared_return, mask=np.ones(5000, dtype=bool),
        ) == pytest.approx(1.0, abs=0.1)

    def test_a_proxy_that_measures_a_larger_quantity_is_visible(self) -> None:
        rng = np.random.default_rng(4)
        sigma2 = 0.0004 * np.exp(rng.normal(0.0, 0.3, size=5000))
        squared_return = sigma2 * rng.chisquare(df=1, size=5000)
        assert proxy_scale_ratio(
            1.5 * sigma2, squared_return, mask=np.ones(5000, dtype=bool),
        ) == pytest.approx(1.5, abs=0.15)

    def test_only_the_masked_bars_count(self) -> None:
        values = np.ones(100)
        reference = np.ones(100)
        values[50:] = 10.0
        mask = np.zeros(100, dtype=bool)
        mask[:50] = True
        assert proxy_scale_ratio(values, reference, mask=mask) == pytest.approx(1.0)


class TestTheProxyPremiseCanOnlyWithholdAPromotion:
    """De poort die na de eerste echte run is toegevoegd.

    Hij kan een `PROMOTED` naar `UNPROVEN` brengen en nooit andersom. Dat is de
    voorwaarde waaronder een controle die ná het zien van de uitkomst is
    toegevoegd, geen researcher degree of freedom is: hij kan de conclusie
    alleen VOORZICHTIGER maken.
    """

    def test_a_violated_premise_turns_a_promotion_into_unproven(self) -> None:
        verdict = judge_challenger(
            arch_p_value=0.001, convergence_ratio=1.0, boundary_ratio=0.0,
            dm=_dm(p_value=0.001, mean_diff=-0.05), proxy_scale_ratio=1.8)
        assert verdict.status == "UNPROVEN"
        assert "proxy_premise_violated" in verdict.binding

    def test_a_violated_premise_also_blocks_a_falsification(self) -> None:
        """Een ongeldige meetlat bewijst niets, ook niet het tegendeel."""
        verdict = judge_challenger(
            arch_p_value=0.001, convergence_ratio=1.0, boundary_ratio=0.0,
            dm=_dm(p_value=0.9, mean_diff=0.001, ar1=0.02), proxy_scale_ratio=1.8)
        assert verdict.status == "UNPROVEN"
        assert "proxy_premise_violated" in verdict.binding

    def test_a_calibrated_proxy_leaves_the_verdict_untouched(self) -> None:
        with_check = judge_challenger(
            arch_p_value=0.001, convergence_ratio=1.0, boundary_ratio=0.0,
            dm=_dm(p_value=0.001, mean_diff=-0.05), proxy_scale_ratio=1.02)
        without = judge_challenger(
            arch_p_value=0.001, convergence_ratio=1.0, boundary_ratio=0.0,
            dm=_dm(p_value=0.001, mean_diff=-0.05))
        assert with_check.status == without.status == "PROMOTED"
        assert with_check.binding == ()

    def test_the_arch_gate_still_comes_first(self) -> None:
        """Een gesloten poort betekent dat er niet is gefit; dan valt er ook
        geen proxy-premisse te schenden."""
        verdict = judge_challenger(
            arch_p_value=0.42, convergence_ratio=1.0, boundary_ratio=0.0,
            dm=None, proxy_scale_ratio=1.8)
        assert verdict.status == "DESCOPED"
        assert verdict.binding[0] == "arch_gate_no_conditional_heteroskedasticity"

    def test_the_measured_ratio_lands_in_the_record(self) -> None:
        record = judge_challenger(
            arch_p_value=0.001, convergence_ratio=1.0, boundary_ratio=0.0,
            dm=_dm(p_value=0.001, mean_diff=-0.05),
            proxy_scale_ratio=1.8).as_record()
        assert record["proxy_scale_ratio"] == pytest.approx(1.8)


class TestDegenerateForecastLevels:
    """Een forecast van 4,5·10²⁵ maal de gemiddelde variantie is geen forecast.

    GEMETEN, 2026-08-29: de gesimuleerde 5-staps forecast van EGARCH met
    Student-t innovaties liep op BTCUSDT op tot 4,5·10²⁵ maal de gemiddelde
    gekwadrateerde return, en op ETHUSDT tot 3,6·10¹⁷. Dat is geen numerieke
    slordigheid maar een eigenschap van het model: de EGARCH-recursie loopt in
    ``ln σ²``, en de verwachting van ``exp`` van een zwaarstaartige random walk
    hoeft niet te bestaan. De simulatie schat dan een moment dat er niet is, en
    het gemiddelde wordt bepaald door een handvol extreme paden.

    QLIKE verbergt dat bijna: hij groeit logaritmisch in een overschatting, dus
    een paar geëxplodeerde bars verschuiven het gemiddelde nauwelijks. Precies
    dat maakt het gevaarlijk — het getal ziet er bruikbaar uit.
    """

    def test_an_exploded_forecast_level_descopes(self) -> None:
        verdict = judge_challenger(
            arch_p_value=0.001, convergence_ratio=1.0, boundary_ratio=0.0,
            dm=_dm(p_value=0.001, mean_diff=-0.05), proxy_scale_ratio=1.0,
            forecast_level_ratio=1e17)
        assert verdict.status == "DESCOPED"
        assert "degenerate_forecast_level" in verdict.binding

    def test_a_plausible_forecast_level_changes_nothing(self) -> None:
        verdict = judge_challenger(
            arch_p_value=0.001, convergence_ratio=1.0, boundary_ratio=0.0,
            dm=_dm(p_value=0.001, mean_diff=-0.05), proxy_scale_ratio=1.0,
            forecast_level_ratio=3.0)
        assert verdict.status == "PROMOTED"
        assert verdict.binding == ()

    def test_it_binds_before_the_proxy_premise(self) -> None:
        """Volgorde: eerst of er een FORECAST is, dan waartegen hij is gemeten.

        Een model dat geen bruikbaar getal levert, hoort niet als `UNPROVEN`
        wegens de meetlat te worden weggeschreven -- dat zou de meetlat de
        schuld geven van een probleem in het model.
        """
        verdict = judge_challenger(
            arch_p_value=0.001, convergence_ratio=1.0, boundary_ratio=0.0,
            dm=_dm(p_value=0.001, mean_diff=-0.05), proxy_scale_ratio=1.8,
            forecast_level_ratio=1e17)
        assert verdict.status == "DESCOPED"
        assert verdict.binding == ("degenerate_forecast_level",)

    def test_the_measured_level_lands_in_the_record(self) -> None:
        record = judge_challenger(
            arch_p_value=0.001, convergence_ratio=1.0, boundary_ratio=0.0,
            dm=_dm(p_value=0.4, mean_diff=-0.001), proxy_scale_ratio=1.0,
            forecast_level_ratio=1e17).as_record()
        assert record["forecast_level_ratio"] == pytest.approx(1e17)


class TestTheEmpiricalPowerControlBindsToo:
    """De gemeten power slaat de berekende power.

    De pre-registratie leidt het minimaal detecteerbare effect AF uit het aantal
    observaties en de AR(1). Dat is een berekening onder aannames. De
    negatieve controle MEET hetzelfde: kan deze toets een forecast waarvan de
    timing volledig is vernietigd, onderscheiden van het origineel?

    GEMETEN, 2026-08-29: op 8 van de 12 reeks/horizon-combinaties kan hij dat
    NIET (p tot 0,90), terwijl de berekende MDE zegt dat de toets gepowerd is.
    Waar berekening en meting elkaar tegenspreken, wint de meting — en een
    toets die een vernietigde forecast niet ziet, kan het veel kleinere
    GARCH-EWMA-verschil zeker niet zien.
    """

    def test_a_failed_power_control_blocks_a_falsification(self) -> None:
        verdict = judge_challenger(
            arch_p_value=0.001, convergence_ratio=1.0, boundary_ratio=0.0,
            dm=_dm(p_value=0.4, mean_diff=-0.001, ar1=0.05),
            power_control_passed=False)
        assert verdict.status == "UNPROVEN"
        assert "underpowered_test_cannot_falsify" in verdict.binding

    def test_a_passed_power_control_leaves_the_falsification_standing(
        self,
    ) -> None:
        verdict = judge_challenger(
            arch_p_value=0.001, convergence_ratio=1.0, boundary_ratio=0.0,
            dm=_dm(p_value=0.4, mean_diff=-0.001, ar1=0.05),
            power_control_passed=True)
        assert verdict.status == "FALSIFIED"

    def test_it_does_not_touch_a_significant_result(self) -> None:
        """Een toets die iets VINDT, heeft zijn power aantoonbaar gehad.

        De controle bewaakt de interpretatie van een NIET-significante uitslag.
        Hem ook op een significante loslaten zou een gevonden effect wegpoetsen
        met het argument dat het niet gevonden had kunnen worden.
        """
        verdict = judge_challenger(
            arch_p_value=0.001, convergence_ratio=1.0, boundary_ratio=0.0,
            dm=_dm(p_value=0.01, mean_diff=-0.05), power_control_passed=False)
        assert verdict.status == "PROMOTED"

    def test_the_control_result_lands_in_the_record(self) -> None:
        record = judge_challenger(
            arch_p_value=0.001, convergence_ratio=1.0, boundary_ratio=0.0,
            dm=_dm(p_value=0.4, mean_diff=-0.001, ar1=0.05),
            power_control_passed=False).as_record()
        assert record["power_control_passed"] is False
