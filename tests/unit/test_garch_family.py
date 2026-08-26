"""De GARCH-familie doet wat zij belooft — Phase 6, deliverable 10.

Geen enkele test hier fit een GARCH op echte data. Dat is opzet en geen
beperking: wat hier moet worden bewezen zijn de EIGENSCHAPPEN VAN DE OPZET, en
die zijn precies de dingen die een geslaagde fit niet laat zien.

    * dat er geen pad bestaat naar een fit op de volledige sample;
    * dat een niet-geconvergeerde fit geen getal levert en EWMA dat getal ook
      niet voor hem invult;
    * dat de persistentie per variant met de JUISTE formule wordt gemeten;
    * dat de forecast voor bar `t` op bar `t - h` is gemaakt en geen bar later.

Die laatste is de gevaarlijkste. Een indexverschuiving van één bar levert een
QLIKE-competitie op die vlekkeloos draait, prachtige p-waarden geeft en volledig
ongeldig is. Hij wordt hier gemeten met een stub die zijn eigen oorsprongsbar in
de forecastwaarde codeert, zodat de assertie letterlijk luidt: *de forecast op
bar t is gemaakt op bar t - h*.
"""
from __future__ import annotations

import math
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

from tradebot.cv.walk_forward import WalkForwardCV
from tradebot.schemas.config import AdequacyConfig, load_config
from tradebot.utils.failfast import DataContractError
from tradebot.volatility import garch as garch_mod
from tradebot.volatility.garch import (
    GARCH_FAMILY,
    RETURN_SCALE,
    ConvergenceSummary,
    GarchFit,
    GarchSpec,
    NonConvergenceError,
    _abs_moment_student_t,
    _persistence,
    fit_garch_window,
    summarise_convergence,
    walk_forward_variance_forecasts,
)

CFG = load_config("conf/model/adequacy.yaml", AdequacyConfig)
SEED = 20260826


def _returns(n: int, *, seed: int = SEED) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0.0, 0.03, n))


# --------------------------------------------------------------------------- #
# De parameterruimte is die van de pre-registratie
# --------------------------------------------------------------------------- #
class TestFamilyMatchesThePreregistration:
    def test_exactly_the_four_preregistered_variants(self) -> None:
        assert set(GARCH_FAMILY) == {"garch", "gjr_garch", "egarch", "aparch"}

    def test_trial_arithmetic_matches_planned_trials(self) -> None:
        """4 modellen x 6 symbolen x 2 horizonnen = 48, het `planned_trials`.

        Een vijfde variant toevoegen breekt deze test, en dat is de bedoeling:
        de ruimte uitbreiden vereist een NIEUWE pre-registratie en verhoogt `M`
        opnieuw. Een stille uitbreiding is exact de zet die pre-registratie moet
        uitsluiten.
        """
        n_symbols, n_horizons = 6, 2
        assert len(GARCH_FAMILY) * n_symbols * n_horizons == 48

    def test_every_variant_uses_student_t(self) -> None:
        """De verdelingsaanname is expliciet en overal dezelfde."""
        assert {s.dist for s in GARCH_FAMILY.values()} == {"t"}


# --------------------------------------------------------------------------- #
# Het Student-t moment onder de APARCH-persistentie
# --------------------------------------------------------------------------- #
class TestStudentTAbsoluteMoment:
    @pytest.mark.parametrize("nu", [4.5, 6.0, 10.0, 50.0, 500.0])
    def test_second_moment_is_exactly_one(self, nu: float) -> None:
        """Een gestandaardiseerde verdeling heeft per definitie ``E|z|^2 = 1``.

        Dat maakt dit een EXACTE controle op de formule en niet een benadering:
        de gamma-functies moeten analytisch tot precies 1 wegvallen. Elke
        typefout in de log-gamma-uitdrukking breekt hem.
        """
        assert _abs_moment_student_t(2.0, nu) == pytest.approx(1.0, rel=1e-12)

    def test_first_moment_approaches_the_normal_case(self) -> None:
        """Voor grote `nu` gaat de Student-t naar de normaal: ``E|z| = sqrt(2/pi)``."""
        assert _abs_moment_student_t(1.0, 1e6) == pytest.approx(
            math.sqrt(2.0 / math.pi), rel=1e-4)

    @pytest.mark.parametrize(("delta", "nu"), [(4.0, 4.0), (6.0, 5.0)])
    def test_moment_beyond_the_tail_is_infinite_not_an_error(
        self, delta: float, nu: float,
    ) -> None:
        """Een niet-bestaand moment is een RESULTAAT: het model is niet stationair.

        Het als `inf` teruggeven laat de randoplossingsteller hem correct als
        randoplossing tellen. Crashen zou een geldige meting tot een storing
        maken; nul teruggeven zou hem onzichtbaar maken.
        """
        assert _abs_moment_student_t(delta, nu) == math.inf


# --------------------------------------------------------------------------- #
# Persistentie: per variant de juiste formule
# --------------------------------------------------------------------------- #
class TestPersistenceIsVariantSpecific:
    PARAMS: ClassVar[dict[str, float]] = {
        "alpha[1]": 0.08, "beta[1]": 0.88, "gamma[1]": 0.06, "nu": 6.0}

    def test_garch_is_alpha_plus_beta(self) -> None:
        assert _persistence(GARCH_FAMILY["garch"], self.PARAMS) == pytest.approx(
            0.96)

    def test_gjr_includes_half_the_leverage_term(self) -> None:
        """``alpha + gamma/2 + beta``. Wie de gamma weglaat, onderschat precies
        het model dat de leverage moet vangen — en laat het ten onrechte langs
        de IGARCH-poort."""
        assert _persistence(
            GARCH_FAMILY["gjr_garch"], self.PARAMS) == pytest.approx(0.99)

    def test_egarch_is_beta_only(self) -> None:
        """EGARCH is stationair voor ``|beta| < 1`` ONGEACHT alpha.

        `alpha + beta` zou hier 0,96 geven en bij een grotere alpha vals alarm
        slaan op een volkomen stationair model.
        """
        assert _persistence(
            GARCH_FAMILY["egarch"], self.PARAMS) == pytest.approx(0.88)

    def test_aparch_uses_the_asymmetric_moment(self) -> None:
        """Bij ``delta = 2`` reduceert de kappa-term tot ``1 + gamma^2``.

        Want ``0,5[(1+g)^2 + (1-g)^2] = 1 + g^2`` en ``E|z|^2 = 1``. Dat maakt
        de verwachte waarde met de hand na te rekenen en dus een echte controle
        in plaats van een vastgelegde uitvoer.
        """
        params = {**self.PARAMS, "delta": 2.0}
        expected = 0.08 * (1.0 + 0.06**2) * 1.0 + 0.88
        assert _persistence(
            GARCH_FAMILY["aparch"], params) == pytest.approx(expected)

    def test_the_four_formulas_do_not_all_agree(self) -> None:
        """Negatieve controle op deze testklasse zelf.

        Zouden alle vier de varianten hetzelfde getal geven, dan zou een
        implementatie die overal `alpha + beta` doet alle bovenstaande tests
        halen. Dit bewijst dat de tests kunnen onderscheiden.
        """
        params = {**self.PARAMS, "delta": 2.0}
        values = {
            name: _persistence(spec, params)
            for name, spec in GARCH_FAMILY.items()
        }
        assert len(set(round(v, 6) for v in values.values())) == 4, values


# --------------------------------------------------------------------------- #
# Nooit op de volledige sample
# --------------------------------------------------------------------------- #
class TestFullSampleFitIsStructurallyImpossible:
    @pytest.mark.parametrize("train_end", [400, 401])
    def test_train_end_at_or_past_the_end_is_rejected(
        self, train_end: int,
    ) -> None:
        returns = _returns(400)
        with pytest.raises(DataContractError, match="volledige sample"):
            fit_garch_window(
                returns, GARCH_FAMILY["garch"], CFG,
                symbol="TEST", fold_id=0, train_end=train_end)

    def test_the_guard_binds_before_the_adequacy_gate(self) -> None:
        """Volgorde-controle: de sample-poort staat vóór de adequaatheidspoort.

        Beide zouden hier kunnen aanslaan. Zou de adequaatheidspoort eerst
        komen, dan zou een full-sample-fit op een RUIME reeks er stilletjes
        doorheen glippen — precies het geval dat gevaarlijk is.
        """
        returns = _returns(3_000)
        with pytest.raises(DataContractError, match="volledige sample"):
            fit_garch_window(
                returns, GARCH_FAMILY["garch"], CFG,
                symbol="TEST", fold_id=0, train_end=3_000)


class TestAdequacyGateRunsBeforeTheFit:
    def test_short_window_is_refused_by_the_gate_not_by_the_optimizer(
        self,
    ) -> None:
        """Een te kort venster crasht op de POORT en niet ergens in `arch`.

        Het onderscheid telt: de poortboodschap draagt de gemeten en de vereiste
        venstergrootte, en zegt dat de trial niet meetelt in `M`. Een
        optimizer-fout zou dat allemaal niet zeggen.
        """
        returns = _returns(200)
        with pytest.raises(DataContractError, match="Data Adequacy Gate"):
            fit_garch_window(
                returns, GARCH_FAMILY["garch"], CFG,
                symbol="TEST", fold_id=0, train_end=100)

    def test_non_finite_returns_are_a_finding_not_something_to_impute(
        self,
    ) -> None:
        returns = _returns(1_000)
        returns.iloc[17] = np.nan
        with pytest.raises(DataContractError, match="niet-eindige"):
            fit_garch_window(
                returns, GARCH_FAMILY["garch"], CFG,
                symbol="TEST", fold_id=0, train_end=800)


# --------------------------------------------------------------------------- #
# Niet-convergentie levert geen getal, en zeker niet dat van EWMA
# --------------------------------------------------------------------------- #
def _failed_fit(spec: GarchSpec | None = None) -> GarchFit:
    return GarchFit(
        spec=spec or GARCH_FAMILY["garch"], symbol="TEST", fold_id=3,
        n_obs=500, converged=False, message="Maximum iterations exceeded",
        params={}, persistence=float("nan"), at_boundary=True,
        loglikelihood=float("nan"), result=None,
    )


class TestNonConvergenceIsRecordedButNeverSubstituted:
    def test_a_failed_fit_refuses_to_forecast(self) -> None:
        with pytest.raises(NonConvergenceError):
            _failed_fit().forecast_variance(horizon=1, start=400)

    def test_the_message_rules_out_the_ewma_fallback_explicitly(self) -> None:
        """De boodschap is hier het contract.

        Zij moet de volgende ontwikkelaar vertellen WAAROM er geen terugval is,
        anders is de eerstvolgende reflex om er bij een rode run een
        `except: return ewma(...)` omheen te zetten — en dat is precies het
        gedrag dat de competitie ongeldig maakt.
        """
        with pytest.raises(NonConvergenceError) as excinfo:
            _failed_fit().forecast_variance(horizon=1, start=400)
        message = str(excinfo.value)
        assert "EWMA" in message
        assert "NIET" in message
        assert "Maximum iterations exceeded" in message

    def test_a_failed_fit_still_reports_itself(self) -> None:
        """Geregistreerd, niet weggevangen: het record is compleet."""
        record = _failed_fit().as_record()
        assert record["converged"] is False
        assert record["message"] == "Maximum iterations exceeded"
        assert record["fold_id"] == 3

    def test_a_numerical_failure_is_recorded_as_non_convergence(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """`LinAlgError` uit een singuliere Hessiaan IS een convergentieresultaat."""
        class _Boom:
            def fit(self, **_: object) -> None:
                raise np.linalg.LinAlgError("singular matrix")

        monkeypatch.setattr(
            garch_mod, "require_dependency",
            lambda *a, **k: type("M", (), {"arch_model": staticmethod(
                lambda *a, **k: _Boom())})())
        fit = fit_garch_window(
            _returns(1_000), GARCH_FAMILY["garch"], CFG,
            symbol="TEST", fold_id=0, train_end=800)
        assert fit.converged is False
        assert "LinAlgError" in fit.message
        assert fit.at_boundary is True

    def test_a_bug_in_the_code_propagates_instead_of_being_booked_as_a_result(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """De negatieve controle op de smalle `except`.

        Een `AttributeError` — de vorm die een typefout in mijn eigen code
        aanneemt — mag NOOIT als "dit venster convergeerde niet" in de
        convergentieratio belanden. Zou zij dat wel doen, dan zou de conclusie
        "GARCH convergeert slecht op dit universum" kunnen rusten op een
        gebroken aanroep in plaats van op een eigenschap van de data.

        Deze test bewaakt dus een BEVINDING en niet een stijlregel: hij wordt
        rood zodra iemand de `except` weer verbreedt.
        """
        class _Typo:
            def fit(self, **_: object) -> None:
                raise AttributeError("'NoneType' object has no attribute 'x'")

        monkeypatch.setattr(
            garch_mod, "require_dependency",
            lambda *a, **k: type("M", (), {"arch_model": staticmethod(
                lambda *a, **k: _Typo())})())
        with pytest.raises(AttributeError):
            fit_garch_window(
                _returns(1_000), GARCH_FAMILY["garch"], CFG,
                symbol="TEST", fold_id=0, train_end=800)


# --------------------------------------------------------------------------- #
# De convergentiesamenvatting voedt het stop-criterium
# --------------------------------------------------------------------------- #
def _fit(*, converged: bool, boundary: bool, spec_key: str = "garch",
         symbol: str = "TEST", message: str = "ok") -> GarchFit:
    return GarchFit(
        spec=GARCH_FAMILY[spec_key], symbol=symbol, fold_id=0, n_obs=500,
        converged=converged, message=message, params={},
        persistence=1.0 if boundary else 0.95, at_boundary=boundary,
        loglikelihood=-1.0, result=None,
    )


class TestConvergenceSummary:
    def test_ratios_and_the_comparability_verdict(self) -> None:
        fits = [_fit(converged=True, boundary=False) for _ in range(9)]
        fits.append(_fit(converged=False, boundary=False, message="failed"))
        summary = summarise_convergence(fits)
        assert summary.convergence_ratio == pytest.approx(0.90)
        assert summary.boundary_ratio == pytest.approx(0.0)
        assert summary.comparable(CFG) is True

    def test_below_the_convergence_floor_is_not_comparable(self) -> None:
        fits = [_fit(converged=True, boundary=False) for _ in range(8)]
        fits += [_fit(converged=False, boundary=False, message="failed")] * 2
        assert summarise_convergence(fits).comparable(CFG) is False

    def test_boundary_ratio_denominator_is_the_converged_fits(self) -> None:
        """Delen door ALLE fits zou een variant door de randpoort laten glippen.

        Vijf geconvergeerde fits waarvan er drie op de IGARCH-rand staan, is een
        randoplossingsratio van 0,60. Zou de noemer alle tien de fits zijn, dan
        stond er 0,30 — nog steeds boven de drempel, maar het punt is het
        principe: een randoplossing is een eigenschap van een fit die een
        antwoord GAF, en de mislukkingen worden al apart geteld.
        """
        fits = [_fit(converged=True, boundary=True) for _ in range(3)]
        fits += [_fit(converged=True, boundary=False) for _ in range(2)]
        fits += [_fit(converged=False, boundary=False, message="x")] * 5
        summary = summarise_convergence(fits)
        assert summary.n_converged == 5
        assert summary.boundary_ratio == pytest.approx(0.60)
        assert summary.comparable(CFG) is False

    def test_failure_messages_are_counted_for_the_report(self) -> None:
        fits = [_fit(converged=True, boundary=False) for _ in range(5)]
        fits += [_fit(converged=False, boundary=False, message="singular")] * 3
        fits += [_fit(converged=False, boundary=False, message="max iter")]
        summary = summarise_convergence(fits)
        assert summary.failure_messages == {"singular": 3, "max iter": 1}

    @pytest.mark.parametrize(
        ("key_a", "key_b", "sym_a", "sym_b"),
        [("garch", "egarch", "TEST", "TEST"), ("garch", "garch", "BTC", "ETH")],
        ids=["mixed-specs", "mixed-symbols"],
    )
    def test_refuses_to_average_over_unlike_fits(
        self, key_a: str, key_b: str, sym_a: str, sym_b: str,
    ) -> None:
        """Anders zou een variant die op één symbool structureel faalt, door de
        poort glippen op het gemiddelde met de symbolen waar hij wel werkt."""
        fits = [
            _fit(converged=True, boundary=False, spec_key=key_a, symbol=sym_a),
            _fit(converged=True, boundary=False, spec_key=key_b, symbol=sym_b),
        ]
        with pytest.raises(DataContractError, match="ongelijksoortige"):
            summarise_convergence(fits)

    def test_an_empty_summary_is_refused(self) -> None:
        with pytest.raises(DataContractError):
            summarise_convergence([])

    def test_ratios_are_zero_on_a_summary_with_no_fits(self) -> None:
        """De dataclass zelf mag geen ZeroDivisionError kennen."""
        empty = ConvergenceSummary(
            spec="x", symbol="y", n_fits=0, n_converged=0, n_boundary=0,
            failure_messages={})
        assert empty.convergence_ratio == 0.0
        assert empty.boundary_ratio == 0.0


# --------------------------------------------------------------------------- #
# De indexverschuiving — de gevaarlijkste regel van het module
# --------------------------------------------------------------------------- #
class _StubForecast:
    def __init__(self, variance: pd.DataFrame) -> None:
        self.variance = variance


class _StubResult:
    """Codeert de OORSPRONGSBAR in de forecastwaarde zelf.

    Rij `i` (oorsprong ``start + i``) krijgt in elke horizonkolom de waarde
    ``(start + i) * RETURN_SCALE**2``. Na de deling door ``RETURN_SCALE**2`` in
    `forecast_variance` staat er dus letterlijk het barnummer waarop de forecast
    is gemaakt. Elke assertie op de uitvoerreeks wordt daarmee een uitspraak
    over WELKE BAR de forecast heeft voortgebracht — en dat is precies wat een
    lookahead-controle moet kunnen zeggen.
    """

    def __init__(self, n_obs: int) -> None:
        self.n_obs = n_obs

    def forecast(self, *, horizon: int, start: int, reindex: bool,
                 method: str) -> _StubForecast:
        assert reindex is False
        assert method == "analytic"
        origins = np.arange(start, self.n_obs, dtype=np.float64)
        block = np.repeat(
            (origins * RETURN_SCALE**2)[:, None], horizon, axis=1)
        return _StubForecast(pd.DataFrame(block))


@pytest.fixture
def stub_fits(monkeypatch: pytest.MonkeyPatch) -> None:
    """Vervang de fit door een stub. De verschuiving wordt zo alleen getest."""

    def _fake_fit(returns, spec, cfg, *, symbol, fold_id, train_end):
        return GarchFit(
            spec=spec, symbol=symbol, fold_id=fold_id, n_obs=train_end,
            converged=True, message="stub", params={}, persistence=0.95,
            at_boundary=False, loglikelihood=-1.0,
            result=_StubResult(int(returns.size)),
        )

    monkeypatch.setattr(garch_mod, "fit_garch_window", _fake_fit)


@pytest.mark.usefixtures("stub_fits")
class TestForecastAlignment:
    CV = WalkForwardCV(
        train_size=500, test_size=100, step=100, mode="rolling",
        embargo_bars=5)

    @pytest.mark.parametrize("horizon", [1, 5])
    def test_the_forecast_on_bar_t_was_made_on_bar_t_minus_horizon(
        self, horizon: int,
    ) -> None:
        """De assertie luidt letterlijk wat de titel zegt.

        Een verschuiving van één bar in welke richting dan ook maakt deze
        gelijkheid onwaar op elke bar tegelijk. Dat is het verschil tussen een
        geldige QLIKE-competitie en een lek dat nergens anders opvalt.
        """
        returns = _returns(1_100)
        out, fits = walk_forward_variance_forecasts(
            returns, GARCH_FAMILY["garch"], self.CV, CFG,
            symbol="TEST", horizon=horizon)
        observed = out.dropna()
        assert not observed.empty
        origins = observed.index.to_numpy(dtype=np.float64) - horizon
        np.testing.assert_allclose(observed.to_numpy(), origins)
        assert len(fits) > 0

    def test_no_forecast_lands_before_the_first_training_window_ends(self) -> None:
        """Niets in het eerste trainvenster mag een forecast dragen.

        Deze test vond het embargozone-defect: de eerste versie van
        `walk_forward_variance_forecasts` vulde alles tussen het geëmbargeerde
        `train_end` (495) en `test_end`, en schreef dus op de bars 496-499 --
        die in geen van beide vensters liggen.
        """
        returns = _returns(1_100)
        out, _ = walk_forward_variance_forecasts(
            returns, GARCH_FAMILY["garch"], self.CV, CFG,
            symbol="TEST", horizon=1)
        assert out.iloc[:500].isna().all()

    @pytest.mark.parametrize("horizon", [1, 5])
    def test_forecasts_land_only_on_declared_test_bars(
        self, horizon: int,
    ) -> None:
        """De uitvoer staat exact op de unie van de testvensters.

        Niet erbinnen (dan mist de competitie OOS-massa) en niet erbuiten (dan
        wordt zij beslecht op bars die de walk-forward-structuur bewust heeft
        uitgesloten). Beide kanten worden hier gemeten, want een test die maar
        één kant controleert zou de embargozone-bug hebben doorgelaten.
        """
        n = 1_100
        returns = _returns(n)
        out, _ = walk_forward_variance_forecasts(
            returns, GARCH_FAMILY["garch"], self.CV, CFG,
            symbol="TEST", horizon=horizon)
        declared = np.zeros(n, dtype=bool)
        for fold in self.CV.split(n):
            declared[fold.test_idx] = True
        filled = out.notna().to_numpy()
        assert not np.any(filled & ~declared), "forecast buiten elk testvenster"
        # Volledige dekking geldt voor h <= embargo + 1; zie de functiedocstring.
        assert np.array_equal(filled, declared)

    def test_the_rescale_back_to_absolute_units_happens_exactly_once(
        self,
    ) -> None:
        """De stub geeft procent-kwadraat; de uitvoer moet absoluut zijn.

        Zou de deling door ``RETURN_SCALE**2`` ontbreken, dan stond hier een
        factor 10.000 — goed voor een QLIKE-verschil van ~9,2 en een competitie
        die op een eenheidsfout wordt beslecht.
        """
        returns = _returns(1_100)
        out, _ = walk_forward_variance_forecasts(
            returns, GARCH_FAMILY["garch"], self.CV, CFG,
            symbol="TEST", horizon=1)
        first = out.dropna()
        assert first.iloc[0] == pytest.approx(first.index[0] - 1)

    def test_a_horizon_below_one_bar_is_refused(self) -> None:
        with pytest.raises(DataContractError, match="horizon onder 1 bar"):
            walk_forward_variance_forecasts(
                _returns(1_100), GARCH_FAMILY["garch"], self.CV, CFG,
                symbol="TEST", horizon=0)

    def test_a_configuration_with_no_folds_is_an_error_not_an_empty_result(
        self,
    ) -> None:
        """Een lege competitie is een configuratiefout en moet luid falen."""
        with pytest.raises(DataContractError, match="geen enkele fold"):
            walk_forward_variance_forecasts(
                _returns(300), GARCH_FAMILY["garch"], self.CV, CFG,
                symbol="TEST", horizon=1)


class TestNonConvergedFoldLeavesItsBarsEmpty:
    def test_bars_of_a_failed_fold_stay_nan(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """De negatieve controle op de stub-opzet hierboven.

        Faalt elke fit, dan is de hele uitvoerreeks NaN — er wordt niets
        ingevuld — terwijl de fits wél worden geteld. Dat is het gedrag waarop
        het stop-criterium `convergence_too_low_to_compare` steunt: de
        convergentieratio moet kunnen dalen zonder dat er stilletjes getallen
        verschijnen.
        """
        def _always_fails(returns, spec, cfg, *, symbol, fold_id, train_end):
            return GarchFit(
                spec=spec, symbol=symbol, fold_id=fold_id, n_obs=train_end,
                converged=False, message="stub failure", params={},
                persistence=float("nan"), at_boundary=True,
                loglikelihood=float("nan"), result=None)

        monkeypatch.setattr(garch_mod, "fit_garch_window", _always_fails)
        out, fits = walk_forward_variance_forecasts(
            _returns(1_100), GARCH_FAMILY["garch"],
            WalkForwardCV(train_size=500, test_size=100, step=100,
                          mode="rolling", embargo_bars=5),
            CFG, symbol="TEST", horizon=1)
        assert out.isna().all()
        assert len(fits) > 0
        assert summarise_convergence(fits).convergence_ratio == 0.0
        assert summarise_convergence(fits).comparable(CFG) is False
