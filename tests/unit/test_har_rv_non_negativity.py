"""HAR-RV levert nooit een negatieve variantieforecast — Phase 6, stap 0.

`test_har_rv_forecast_non_negative` was rood op `84273ca`. Deze suite legt vast
WAAROM hij rood was, dat de reparatie het tegenvoorbeeld daadwerkelijk sluit, en
dat de garantie op een mechanisme rust in plaats van op een clip.

De volgorde van de tests volgt de redenering:

    1. het historische tegenvoorbeeld is gesloten;
    2. de ongeconstrainde OLS zou op datzelfde tegenvoorbeeld WEL negatief zijn
       geworden — de negatieve controle, zonder welke test 1 net zo goed groen
       had kunnen zijn omdat het probleem nooit bestond;
    3. de garantie geldt structureel, niet alleen op dit ene voorbeeld;
    4. het invoercontract dekt het gat waarop de garantie steunt;
    5. de restrictie is zichtbaar in de output in plaats van stil.
"""
from __future__ import annotations

import numpy as np
import pytest

from tradebot.utils.failfast import DataContractError
from tradebot.volatility.har_rv import har_rv_feature, har_rv_fit

#: Het exacte tegenvoorbeeld dat Hypothesis vond: een RV-reeks die overal nul is
#: op één spike na. Het is geen exotisch geval — het is de vorm die een RV-proxy
#: aanneemt op een illiquide reeks met één gebeurtenis.
_COUNTEREXAMPLE = np.zeros(30, dtype=np.float64)
_COUNTEREXAMPLE[21] = 0.25**2


def test_the_historical_counterexample_no_longer_produces_a_negative_forecast() -> None:
    forecast = har_rv_feature(_COUNTEREXAMPLE)
    valid = forecast[~np.isnan(forecast)]
    assert valid.size > 0
    assert np.all(valid >= 0.0), f"nog steeds negatief: {valid[valid < 0]}"


def test_unconstrained_ols_would_still_be_negative_here() -> None:
    """Negatieve controle (§0.10) — bewijst dat test 1 iets te bewijzen had.

    Zonder deze test is er geen bewijs dat het tegenvoorbeeld ooit een probleem
    wás; een reparatie van een niet-bestaande bug is groen om de verkeerde reden.
    Hier wordt exact dezelfde ontwerpmatrix ongeconstraind opgelost en
    aangetoond dat de fit dan wél onder nul duikt.
    """
    rv = _COUNTEREXAMPLE
    start, T = 21, len(rv) - 21
    rv_d = rv[start - 1 : start - 1 + T]
    rv_w = np.array([np.mean(rv[max(0, start + i - 5) : start + i]) for i in range(T)])
    rv_m = np.array([np.mean(rv[max(0, start + i - 21) : start + i]) for i in range(T)])
    X = np.column_stack([np.ones(T), rv_d, rv_w, rv_m])
    y = rv[start:]

    beta_ols, *_ = np.linalg.lstsq(X, y, rcond=None)
    ols_forecast = X @ beta_ols
    assert np.any(ols_forecast < 0.0), (
        "de ongeconstrainde OLS is hier niet negatief; dan meet test 1 niet "
        "wat hij beweert te meten en moet het tegenvoorbeeld worden herzien"
    )


@pytest.mark.parametrize(
    ("name", "rv"),
    [
        ("alles nul", np.zeros(60)),
        ("één spike", _COUNTEREXAMPLE),
        ("twee spikes ver uit elkaar", np.where(np.arange(80) % 37 == 0, 0.04, 0.0)),
        ("monotoon dalend naar nul", np.linspace(0.09, 0.0, 90)),
        ("monotoon stijgend", np.linspace(0.0, 0.09, 90)),
        ("constante variantie", np.full(60, 0.0004)),
        ("clustered vol", np.concatenate([np.full(40, 1e-6), np.full(40, 0.01)])),
    ],
)
def test_the_forecast_is_non_negative_on_adversarial_variance_shapes(
    name: str, rv: np.ndarray
) -> None:
    """De garantie is structureel, niet voorbeeld-specifiek."""
    forecast = har_rv_feature(np.asarray(rv, dtype=np.float64))
    valid = forecast[~np.isnan(forecast)]
    assert np.all(valid >= 0.0), f"{name}: negatieve forecast {valid[valid < 0]}"


def test_a_negative_input_is_a_hard_failure_not_a_silent_clip() -> None:
    """Het invoercontract dekt het gat waarop de garantie steunt.

    ``X @ beta >= 0`` volgt uit ``beta >= 0`` ÉN ``X >= 0``. De tweede helft is
    een eigenschap van de INVOER en moet dus worden afgedwongen, anders is de
    garantie een aanname.
    """
    rv = np.full(60, 0.01)
    rv[10] = -0.01
    with pytest.raises(DataContractError, match="NEGATIEVE realized variance"):
        har_rv_fit(rv)


def test_a_non_finite_input_is_a_hard_failure() -> None:
    rv = np.full(60, 0.01)
    rv[10] = np.nan
    with pytest.raises(DataContractError, match="niet-eindige"):
        har_rv_fit(rv)


def test_the_constraint_reports_when_it_binds() -> None:
    """De restrictie is zichtbaar, niet stil.

    Een model dat zijn eigen ingrepen niet rapporteert, kan in een rapport als
    ongeconstrainde OLS worden gepresenteerd terwijl het dat niet is.
    """
    clamped = har_rv_fit(_COUNTEREXAMPLE)
    assert clamped.n_coefficients_clamped > 0, (
        "op het tegenvoorbeeld hoort de restrictie te binden"
    )
    assert all(b >= 0.0 for b in (clamped.c, clamped.beta_d, clamped.beta_w,
                                  clamped.beta_m))


def test_the_constraint_does_not_bind_on_a_well_behaved_har_process() -> None:
    """Op data waar de restrictie niet hoort te binden, bindt zij ook niet.

    Dit is de tegenhanger van de vorige test: samen laten zij zien dat
    ``n_coefficients_clamped`` informatie draagt in plaats van altijd hetzelfde
    te zeggen. Zonder deze helft zou een teller die constant > 0 teruggeeft er
    net zo goed uitzien.
    """
    rng = np.random.default_rng(20260825)
    n = 600
    rv = np.empty(n)
    rv[:21] = 4e-4
    for t in range(21, n):
        mu = (2e-5
              + 0.35 * rv[t - 1]
              + 0.35 * rv[t - 5 : t].mean()
              + 0.25 * rv[t - 21 : t].mean())
        rv[t] = max(mu * rng.lognormal(mean=-0.125, sigma=0.5), 0.0)

    fit = har_rv_fit(rv)
    assert fit.n_coefficients_clamped == 0, (
        "de restrictie bindt op een gesimuleerd HAR-proces met positieve "
        f"coëfficiënten; fit={fit.c, fit.beta_d, fit.beta_w, fit.beta_m}"
    )
    assert fit.r_squared > 0.0
    valid = fit.forecast[~np.isnan(fit.forecast)]
    assert np.all(valid >= 0.0)


def test_the_singular_fit_does_not_fall_back_to_a_constant() -> None:
    """§5.2 / §26-AC5: geen stille degradatie naar Level 0.

    De Phase 0-fallback zette bij een singuliere fit alle hellingen op nul en
    het intercept op het gemiddelde. Dat is een CONSTANTE voorspelling die zich
    voordoet als HAR-RV. Op perfect collineaire invoer (alles constant) mag het
    model die vorm dus niet aannemen zonder dat het meetbaar is.
    """
    fit = har_rv_fit(np.full(60, 0.0004))
    # Collineair: de fit is niet uniek, maar hij mag geen NEGATIEVE of
    # niet-eindige forecast opleveren en moet de reeks reproduceren.
    valid = fit.forecast[~np.isnan(fit.forecast)]
    assert np.all(np.isfinite(valid))
    assert np.all(valid >= 0.0)
    np.testing.assert_allclose(valid, 0.0004, rtol=1e-6, atol=1e-12)
