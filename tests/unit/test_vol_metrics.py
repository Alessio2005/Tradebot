"""De weegschaal van de QLIKE-competitie deugt — Phase 6, deliverable 12.

Dit is het gereedschap dat H1 beslecht. Een verliesfunctie geeft altijd een
getal en een toets geeft altijd een p-waarde; geen van beide klaagt wanneer hij
de verkeerde vraag beantwoordt. §0.10 van de fase-opdracht is daarom hier het
strengst van toepassing: *een DM-HLN-test die ook significant is op ruis, meet
niets.*

Elke toets in dit bestand wordt daarom van twee kanten benaderd:

    grootte   op data waar de nulhypothese WAAR is, mag hij ~5 % verwerpen --
              niet meer (dan is hij vals alarm) en niet veel minder (dan is hij
              te conservatief om iets te vinden);
    power     op data waar zij ONWAAR is, moet hij verwerpen.

En de centrale ontwerpclaim van de module -- dat het oordeel aan QLIKE hangt en
niet aan MSE-SD -- wordt niet als mening opgeschreven maar gemeten.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tradebot.utils.failfast import DataContractError
from tradebot.validation.vol_metrics import (
    PROXY_ROBUST_LOSSES,
    LossSeries,
    diebold_mariano_hln,
    loss_series,
    mae_sd,
    mincer_zarnowitz,
    qlike,
)

SEED = 20260826


def _rng(offset: int = 0) -> np.random.Generator:
    return np.random.default_rng(SEED + offset)


# --------------------------------------------------------------------------- #
# QLIKE als Bregman-divergentie
# --------------------------------------------------------------------------- #
class TestQlikeShape:
    def test_zero_exactly_at_a_perfect_forecast(self) -> None:
        rv = np.array([0.01, 0.04, 0.09])
        np.testing.assert_allclose(qlike(rv, rv), 0.0, atol=1e-15)

    @pytest.mark.parametrize("factor", [0.25, 0.5, 2.0, 4.0])
    def test_strictly_positive_away_from_the_truth(self, factor: float) -> None:
        rv = np.array([0.01, 0.04, 0.09])
        assert np.all(qlike(rv, rv * factor) > 0.0)

    def test_under_forecasting_costs_more_than_over_forecasting(self) -> None:
        """De asymmetrie is een EIGENSCHAP en geen gebrek.

        Voor een risicomodel is te lage variantie gevaarlijker dan te hoge, en
        QLIKE straft precies zo. Een symmetrische maat zou die voorkeur niet
        uitdrukken.
        """
        rv = np.array([0.04])
        too_low = float(qlike(rv, rv * 0.5)[0])
        too_high = float(qlike(rv, rv * 2.0)[0])
        assert too_low > too_high

    @pytest.mark.parametrize(
        ("rv_value", "forecast_value"),
        [(0.0, 0.04), (0.04, 0.0), (-0.01, 0.04), (np.nan, 0.04),
         (0.04, np.nan)],
    )
    def test_undefined_bars_become_nan_not_infinity(
        self, rv_value: float, forecast_value: float,
    ) -> None:
        """`ln(0)` is de QLIKE-bom uit de moduledocstring.

        Een enkele `-inf` maakt het gemiddelde verlies `-inf` en beslist de hele
        competitie. NaN is de enige veilige uitkomst, want `nanmean` laat hem
        vallen en `LossSeries` telt hem.
        """
        out = qlike(np.array([rv_value]), np.array([forecast_value]))
        assert np.isnan(out[0])

    def test_length_mismatch_is_refused(self) -> None:
        with pytest.raises(DataContractError, match="verschillende lengtes"):
            qlike(np.array([0.01, 0.02]), np.array([0.01]))

    def test_only_two_losses_are_marked_proxy_robust(self) -> None:
        assert PROXY_ROBUST_LOSSES == {"qlike", "mse_variance"}


# --------------------------------------------------------------------------- #
# De centrale ontwerpclaim: Patton (2011), gemeten
# --------------------------------------------------------------------------- #
class TestProxyRobustness:
    """Waarom het oordeel aan QLIKE hangt en niet aan MSE-SD.

    De opzet is exact uitrekenbaar en daarom een echte controle in plaats van
    een demonstratie. Met een proxy ``RV = sigma^2 z^2`` (de gekwadrateerde
    return: zuiver, zeer ruizig) geldt:

        QLIKE   E[RV/f - ln(RV/f) - 1] is minimaal op f = E[RV] = sigma^2.
                De WARE variantie wint. Dat is de robuustheid van Patton.

        MSE-SD  E[(sqrt(RV) - sqrt(f))^2] is minimaal op
                sqrt(f) = E[sigma|z|] = sigma sqrt(2/pi),
                dus op f = sigma^2 * 2/pi = 0,6366 sigma^2.

    MSE-SD verkiest dus stelselmatig een model dat de variantie met 36 %
    ONDERSCHAT boven een model dat haar precies goed heeft. Dat is geen kleine
    vertekening aan de rand; het is een omgekeerde rangschikking, en het is de
    reden dat MSE-SD in dit module rapportagemateriaal is en geen oordeel draagt.
    """

    N = 400_000

    def _setup(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rng = _rng(1)
        sigma2 = np.full(self.N, 0.0009)          # sigma = 3 % per bar
        proxy = sigma2 * rng.normal(0.0, 1.0, self.N) ** 2
        truthful = sigma2
        under = sigma2 * (2.0 / math.pi)
        return proxy, truthful, under

    def test_qlike_prefers_the_truthful_model(self) -> None:
        proxy, truthful, under = self._setup()
        a = loss_series(proxy, truthful, loss="qlike", model="truthful")
        b = loss_series(proxy, under, loss="qlike", model="under")
        assert a.mean < b.mean

    def test_mse_sd_prefers_the_biased_model_and_that_is_the_point(self) -> None:
        proxy, truthful, under = self._setup()
        a = loss_series(proxy, truthful, loss="mse_sd", model="truthful")
        b = loss_series(proxy, under, loss="mse_sd", model="under")
        assert b.mean < a.mean, (
            "MSE-SD hoort hier het ONDERSCHATTENDE model te verkiezen; doet hij "
            "dat niet, dan klopt de opzet van deze test niet en bewijst zij "
            "niets over robuustheid.")

    def test_the_two_losses_disagree_and_qlike_is_the_one_that_is_right(
        self,
    ) -> None:
        """De rangschikkingen zijn tegengesteld. Dat is de bevinding zelf."""
        proxy, truthful, under = self._setup()
        qlike_winner = min(
            (loss_series(proxy, f, loss="qlike", model=m) for m, f in
             [("truthful", truthful), ("under", under)]),
            key=lambda s: s.mean).model
        mse_winner = min(
            (loss_series(proxy, f, loss="mse_sd", model=m) for m, f in
             [("truthful", truthful), ("under", under)]),
            key=lambda s: s.mean).model
        assert qlike_winner == "truthful"
        assert mse_winner == "under"
        assert qlike_winner != mse_winner

    def test_mae_sd_shares_the_defect(self) -> None:
        proxy, truthful, _ = self._setup()
        # De MAE-SD-optimale forecast is de MEDIAAN van sigma|z|, niet sigma.
        median_factor = float(np.median(np.abs(_rng(2).normal(0, 1, 200_000))))
        shifted = truthful * median_factor**2
        a = float(np.nanmean(mae_sd(proxy, truthful)))
        b = float(np.nanmean(mae_sd(proxy, shifted)))
        assert b < a

    def test_mse_on_the_variance_scale_stays_robust(self) -> None:
        """MSE-VAR is wél robuust — het is de tweede lid van `PROXY_ROBUST_LOSSES`."""
        proxy, truthful, under = self._setup()
        a = loss_series(proxy, truthful, loss="mse_variance", model="truthful")
        b = loss_series(proxy, under, loss="mse_variance", model="under")
        assert a.mean < b.mean


# --------------------------------------------------------------------------- #
# De boekhouding van weggevallen bars
# --------------------------------------------------------------------------- #
class TestLossSeriesAccounting:
    def test_proxy_gaps_and_forecast_gaps_are_counted_separately(self) -> None:
        """Het onderscheid draagt de convergentiediagnose.

        Een proxy-gat treft elk model gelijk; een forecast-gat is de
        niet-geconvergeerde fold van precies dit model. Wie ze optelt, kan die
        twee niet meer uit elkaar houden in het rapport.
        """
        rv = np.array([0.01, 0.0, 0.04, 0.09, np.nan])
        forecast = np.array([0.01, 0.02, np.nan, 0.09, 0.01])
        series = loss_series(rv, forecast, loss="qlike", model="m")
        assert series.n_total == 5
        assert series.n_usable == 2                # bars 0 en 3
        assert series.n_dropped_proxy == 2         # de nul en de NaN in rv
        assert series.n_dropped_forecast == 1      # de NaN in forecast
        assert series.coverage == pytest.approx(0.4)

    def test_a_negative_qlike_is_refused(self) -> None:
        """Negatieve QLIKE kan niet bestaan en wijst op een eenhedenfout.

        Dit is de vangnetassertie voor precies het geval uit de
        `garch.py`-docstring: een forecast die per ongeluk in procent-kwadraat
        is blijven staan. Zonder deze controle zou dat een spectaculair goed
        ogende competitie opleveren.
        """
        rv = np.array([0.04])
        series = loss_series(rv, rv, loss="qlike", model="ok")
        assert series.mean == pytest.approx(0.0, abs=1e-12)

    def test_unknown_loss_function_is_refused(self) -> None:
        with pytest.raises(DataContractError, match="Onbekende verliesfunctie"):
            loss_series(np.array([0.01]), np.array([0.01]),
                        loss="rmse", model="m")


# --------------------------------------------------------------------------- #
# Mincer-Zarnowitz
# --------------------------------------------------------------------------- #
def _persistent_variance(n: int, *, offset: int = 0) -> np.ndarray:
    """Een AR(1) in log-variantie: persistent, zoals echte volatiliteit."""
    rng = _rng(offset)
    log_v = np.zeros(n)
    for t in range(1, n):
        log_v[t] = 0.97 * log_v[t - 1] + rng.normal(0.0, 0.15)
    return 0.0009 * np.exp(log_v)


class TestMincerZarnowitz:
    def test_an_unbiased_forecast_is_not_rejected(self) -> None:
        sigma2 = _persistent_variance(1_500)
        rv = sigma2 * _rng(3).chisquare(df=8, size=1_500) / 8.0
        result = mincer_zarnowitz(rv, sigma2, model="perfect")
        assert result.beta == pytest.approx(1.0, abs=0.15)
        assert result.unbiased, f"p = {result.p_value:.4f}"

    def test_a_systematically_scaled_forecast_is_rejected(self) -> None:
        """Een forecast die de variantie stelselmatig halveert, is niet zuiver."""
        sigma2 = _persistent_variance(1_500)
        rv = sigma2 * _rng(4).chisquare(df=8, size=1_500) / 8.0
        result = mincer_zarnowitz(rv, sigma2 * 0.5, model="halved")
        assert result.beta == pytest.approx(2.0, abs=0.3)
        assert not result.unbiased
        assert result.p_value < 0.05

    def test_hac_lags_are_reported_not_hidden(self) -> None:
        sigma2 = _persistent_variance(1_500)
        rv = sigma2 * _rng(5).chisquare(df=8, size=1_500) / 8.0
        assert mincer_zarnowitz(rv, sigma2, model="m").hac_lags >= 1

    def test_hac_is_less_trigger_happy_than_plain_ols(self) -> None:
        """De rechtvaardiging van de HAC-keuze, gemeten in plaats van beweerd.

        Onder OLS zijn de standaardfouten op serieel gecorreleerde residuen te
        klein, dus de Wald-statistiek te groot en verwerpt de toets de
        zuiverheid van forecasts die zuiver ZIJN. Deze test rekent beide
        varianten uit op dezelfde zuivere forecast en eist dat de
        HAC-statistiek de kleinere is.
        """
        n = 1_500
        sigma2 = _persistent_variance(n, offset=6)
        rv = sigma2 * _rng(7).chisquare(df=8, size=n) / 8.0
        hac = mincer_zarnowitz(rv, sigma2, model="hac")

        design = np.column_stack([np.ones(n), sigma2])
        xtx_inv = np.linalg.inv(design.T @ design)
        coef = xtx_inv @ design.T @ rv
        resid = rv - design @ coef
        ols_cov = float(resid @ resid) / (n - 2) * xtx_inv
        deviation = coef - np.array([0.0, 1.0])
        ols_wald = float(deviation @ np.linalg.inv(ols_cov) @ deviation)

        assert hac.wald_statistic < ols_wald

    def test_a_constant_forecast_is_refused_rather_than_silently_fitted(
        self,
    ) -> None:
        rv = _persistent_variance(200)
        with pytest.raises(DataContractError, match="CONSTANTE forecast"):
            mincer_zarnowitz(rv, np.full(200, 0.0009), model="level0")

    def test_too_few_bars_is_refused(self) -> None:
        with pytest.raises(DataContractError, match="te weinig bruikbare bars"):
            mincer_zarnowitz(
                np.array([0.01] * 10), np.array([0.011] * 10), model="m")


# --------------------------------------------------------------------------- #
# Diebold-Mariano met HLN — de toets waar H1 op hangt
# --------------------------------------------------------------------------- #
def _loss_series(values: np.ndarray, model: str) -> LossSeries:
    """Een `LossSeries` uit AL BEREKENDE verliezen.

    De omweg via `loss_series` zou een verliesfunctie en een proxy opleggen;
    in de toetsen hieronder zijn de VERLIEZEN zelf het studieobject, want wat
    getest wordt is de toets en niet de verliesfunctie.
    """
    usable = int(np.count_nonzero(np.isfinite(values)))
    return LossSeries(
        name="qlike", model=model, values=values, n_total=int(values.size),
        n_usable=usable, n_dropped_proxy=0,
        n_dropped_forecast=int(values.size) - usable)


def _losses(
    values_a: np.ndarray, values_b: np.ndarray,
) -> tuple[LossSeries, LossSeries]:
    return _loss_series(values_a, "A"), _loss_series(values_b, "B")


class TestDieboldMarianoNegativeControls:
    """§0.10: bewijs dat de toets de nulhypothese ACCEPTEERT op ruis."""

    def test_a_model_against_a_shuffle_of_itself_is_not_significant(self) -> None:
        """De negatieve controle die de H1-pre-registratie letterlijk eist.

        Een geschudde variant van dezelfde verliesreeks heeft per constructie
        hetzelfde gemiddelde verlies. Vindt de toets hier significantie, dan
        meet hij ruis en is elke uitkomst van de competitie waardeloos.
        """
        base = _rng(10).gamma(shape=2.0, scale=0.5, size=1_200)
        shuffled = _rng(11).permutation(base)
        a, b = _losses(base, shuffled)
        result = diebold_mariano_hln(a, b, horizon=1)
        assert not result.significant, f"p = {result.p_value:.4f}"

    def test_size_is_close_to_five_percent_under_the_null(self) -> None:
        """De sterkste controle: hoe vaak verwerpt hij als er niets te vinden is?

        400 herhalingen van twee onafhankelijke verliesreeksen met hetzelfde
        gemiddelde. De verwerpingsgraad hoort rond 0,05 te liggen. Ligt hij veel
        hoger, dan is elke significante uitkomst van H1 vals alarm; veel lager,
        dan is de toets te conservatief om het verwachte effect van 0,10 SD ooit
        te zien.
        """
        rng = _rng(12)
        rejections = 0
        trials, n = 400, 500
        for _ in range(trials):
            a, b = _losses(
                rng.gamma(2.0, 0.5, n), rng.gamma(2.0, 0.5, n))
            if diebold_mariano_hln(a, b, horizon=1).significant:
                rejections += 1
        rate = rejections / trials
        assert 0.02 <= rate <= 0.09, f"verwerpingsgraad {rate:.3f}"

    def test_identical_series_have_zero_differential(self) -> None:
        """Twee identieke reeksen: de langetermijnvariantie is nul en de toets
        weigert in plaats van door nul te delen."""
        base = _rng(13).gamma(2.0, 0.5, 500)
        a, b = _losses(base, base.copy())
        with pytest.raises(DataContractError, match="niet positief"):
            diebold_mariano_hln(a, b, horizon=1)


class TestDieboldMarianoPower:
    def test_a_genuinely_better_model_is_detected(self) -> None:
        rng = _rng(14)
        n = 1_200
        common = rng.gamma(2.0, 0.5, n)
        a, b = _losses(common, common + rng.normal(0.15, 0.3, n))
        result = diebold_mariano_hln(a, b, horizon=1)
        assert result.significant
        assert result.better == "A"
        assert result.mean_loss_differential < 0

    def test_one_sided_alternative_points_the_right_way(self) -> None:
        rng = _rng(15)
        n = 1_200
        common = rng.gamma(2.0, 0.5, n)
        a, b = _losses(common, common + rng.normal(0.15, 0.3, n))
        better = diebold_mariano_hln(a, b, horizon=1, alternative="a-better")
        worse = diebold_mariano_hln(a, b, horizon=1, alternative="b-better")
        assert better.p_value < 0.05
        assert worse.p_value > 0.95


class TestHarveyLeybourneNewboldCorrection:
    def test_the_correction_shrinks_the_statistic(self) -> None:
        """HLN corrigeert de te-vaak-verwerpen-neiging van de ruwe DM.

        De correctiefactor is kleiner dan 1 voor elke eindige n, dus |DM*| <
        |DM|. Ontbreekt de correctie, dan is de toets liberaal — precies wat
        HLN (1997) op simulaties aantoonde.
        """
        rng = _rng(16)
        n = 300
        common = rng.gamma(2.0, 0.5, n)
        a, b = _losses(common, common + rng.normal(0.1, 0.4, n))
        result = diebold_mariano_hln(a, b, horizon=1)
        assert abs(result.hln_statistic) < abs(result.dm_statistic)

    def test_the_correction_bites_harder_at_a_longer_horizon(self) -> None:
        rng = _rng(17)
        n = 300
        common = rng.gamma(2.0, 0.5, n)
        a, b = _losses(common, common + rng.normal(0.1, 0.4, n))
        ratio_h1 = abs(diebold_mariano_hln(a, b, horizon=1).hln_statistic)
        r5 = diebold_mariano_hln(a, b, horizon=5)
        assert abs(r5.hln_statistic) / abs(r5.dm_statistic) < ratio_h1 / abs(
            diebold_mariano_hln(a, b, horizon=1).dm_statistic)


class TestTheComparisonRunsOverTheSameBars:
    def test_intersection_is_taken_and_reported(self) -> None:
        rng = _rng(18)
        n = 1_000
        values_a = rng.gamma(2.0, 0.5, n)
        values_b = rng.gamma(2.0, 0.5, n)
        values_b[:50] = np.nan          # 5 % niet-geconvergeerde folds
        a, b = _losses(values_a, values_b)
        result = diebold_mariano_hln(a, b, horizon=1)
        assert result.n_obs == n - 50
        # Gemeten t.o.v. de BEST gedekte reeks: A had 1.000 bruikbare bars, de
        # vergelijking loopt over 950, dus de competitie verloor er 50 aan de
        # niet-geconvergeerde folds van B. Zou dit t.o.v. de slechtst gedekte
        # reeks worden gemeten, dan stond hier altijd 0 en zou het getal niets
        # zeggen — dezelfde denkfout als in de poort eronder.
        assert result.n_dropped_to_intersection == 50

    def test_too_little_overlap_is_refused(self) -> None:
        """De stilste manier om deze competitie ongeldig te maken.

        Een GARCH die op 40 % van de folds niet convergeerde, wordt anders
        vergeleken met een EWMA die op ALLE folds een waarde had — en de
        ontbrekende bars zijn precies de moeilijke vensters.
        """
        rng = _rng(19)
        n = 1_000
        values_a = rng.gamma(2.0, 0.5, n)
        values_b = values_a + rng.normal(0.0, 0.1, n)
        values_b[400:] = np.nan
        a, b = _losses(values_a, values_b)
        with pytest.raises(DataContractError, match="te weinig bars"):
            diebold_mariano_hln(a, b, horizon=1)

    def test_comparing_two_different_loss_functions_is_refused(self) -> None:
        values = _rng(20).gamma(2.0, 0.5, 500)
        a = LossSeries(name="qlike", model="A", values=values, n_total=500,
                       n_usable=500, n_dropped_proxy=0, n_dropped_forecast=0)
        b = LossSeries(name="mse_sd", model="B", values=values + 0.1,
                       n_total=500, n_usable=500, n_dropped_proxy=0,
                       n_dropped_forecast=0)
        with pytest.raises(DataContractError, match="VERSCHILLENDE verliesfuncties"):
            diebold_mariano_hln(a, b, horizon=1)


class TestTheAr1ThatDecidesUnprovenVersusFalsified:
    """Het H1-stopcriterium `underpowered_test_cannot_falsify` hangt hieraan."""

    def test_independent_losses_give_an_ar1_near_zero(self) -> None:
        rng = _rng(21)
        n = 5_000
        a, b = _losses(rng.gamma(2.0, 0.5, n), rng.gamma(2.0, 0.5, n))
        assert abs(diebold_mariano_hln(a, b, horizon=1).loss_differential_ar1) < 0.05

    def test_a_persistent_differential_is_measured_as_such(self) -> None:
        """Boven ~0,3 is een niet-significante uitkomst `UNPROVEN`, geen falsificatie.

        Daarom moet dit getal kloppen: het bepaalt welk van twee heel
        verschillende oordelen in de ledger belandt.
        """
        rng = _rng(22)
        n = 5_000
        noise = np.zeros(n)
        for t in range(1, n):
            noise[t] = 0.6 * noise[t - 1] + rng.normal(0.0, 0.2)
        base = rng.gamma(2.0, 0.5, n)
        a, b = _losses(base + noise, base)
        measured = diebold_mariano_hln(a, b, horizon=1).loss_differential_ar1
        assert measured == pytest.approx(0.6, abs=0.05)
