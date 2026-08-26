"""De QLIKE-variantieproxy doet wat zij beweert — Phase 6, deliverable 11.

Deliverable 11 eist non-negativiteit *gegarandeerd en getest*. Die twee woorden
vragen twee verschillende dingen, en dit bestand levert ze allebei:

    gegarandeerd  de bewijzen in `realized.py`, die alle drie steunen op
                  `L <= min(O,C) <= max(O,C) <= H`;
    getest        het bewijs is een redenering en kan een verkeerde aanname
                  bevatten. Hier wordt de premisse geschonden en gecontroleerd
                  dat de code dat merkt in plaats van een negatieve variantie
                  door te geven aan een loss met een logaritme erin.

Daarnaast wordt de ZUIVERHEID gemeten. De constanten `4 ln 2` en `2 ln 2 - 1`
zijn met de hand overgeschreven uit Parkinson (1980) en Garman-Klass (1980); een
typefout daarin geeft een estimator die er volstrekt normaal uitziet en de
variantie stelselmatig met een paar procent misschat. Op zo'n bias wordt een
QLIKE-competitie beslist. De simulatie hieronder zou hem zien.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradebot.utils.failfast import DataContractError, DependencyMissingError
from tradebot.volatility.realized import (
    PROXY_EFFICIENCY,
    build_range_proxies,
    garman_klass_variance,
    parkinson_variance,
    realized_variance_from_intraday,
    require_valid_ohlc,
    rogers_satchell_variance,
    squared_return_variance,
    yang_zhang_variance,
)

SEED = 20260826


def _simulate_gbm_bars(
    n_bars: int, sigma: float, *, mu: float = 0.0, steps: int = 2_000,
    seed: int = SEED, chain: bool = True,
) -> pd.DataFrame:
    """Bars uit een expliciet gesimuleerde GBM, zodat de ware variantie bekend is.

    `sigma` is de standaardafwijking van de log-return OVER EEN BAR. Het pad
    binnen de bar wordt in `steps` stappen gelopen; open, high, low en close
    volgen uit dat pad en zijn dus per constructie consistent. Dat is precies
    de premisse waarop de non-negativiteitsbewijzen rusten.

    `chain=False` laat elke bar op prijs 1 beginnen in plaats van op de close
    van de vorige. DEFECT IN MIJN EIGEN TESTWERK, hier vastgelegd: de eerste
    versie ketende altijd, en de zuiverheidstest mét drift liep daardoor op
    40.000 bars naar een log-prijs van 40.000 x 0,12 = 4.800, waar `np.exp`
    overflowt naar `inf` en elke estimator NaN teruggeeft. De test faalde dus op
    zijn eigen simulatie en niet op de estimator.

    Ketenen is voor Parkinson, Garman-Klass en Rogers-Satchell sowieso
    irrelevant: alle drie zijn functies van log-VERHOUDINGEN BINNEN een bar en
    dus invariant onder het prijsniveau. Alleen de gekwadrateerde return en
    Yang-Zhang gebruiken de overgang tussen bars; die tests ketenen wel.
    """
    rng = np.random.default_rng(seed)
    increments = rng.normal(
        mu / steps, sigma / np.sqrt(steps), size=(n_bars, steps))
    paths = np.cumsum(increments, axis=1)
    if chain:
        log_open = np.concatenate([[0.0], np.cumsum(paths[:, -1])[:-1]])
    else:
        log_open = np.zeros(n_bars)
    log_paths = log_open[:, None] + paths
    return pd.DataFrame({
        "open": np.exp(log_open),
        "high": np.exp(np.maximum(log_paths.max(axis=1), log_open)),
        "low": np.exp(np.minimum(log_paths.min(axis=1), log_open)),
        "close": np.exp(log_paths[:, -1]),
    })


class TestNonNegativity:
    """De garantie zelf, op de drie per-bar estimators."""

    @pytest.mark.parametrize("sigma", [0.005, 0.02, 0.08, 0.25])
    def test_all_estimators_non_negative_on_valid_bars(self, sigma: float) -> None:
        bars = _simulate_gbm_bars(3_000, sigma)
        o, h, lo, c = (bars[k].to_numpy() for k in ("open", "high", "low", "close"))
        assert np.all(parkinson_variance(h, lo) >= 0.0)
        assert np.all(garman_klass_variance(o, h, lo, c) >= 0.0)
        assert np.all(rogers_satchell_variance(o, h, lo, c) >= 0.0)

    def test_flat_bar_gives_exactly_zero_not_a_tiny_negative(self) -> None:
        """Een bar zonder beweging is de scherpste rand van de bewijzen.

        Daar geldt x = y = 0 en zijn alle drie de estimators exact nul. Levert
        een van hen hier -1e-18 op door afrondingsvolgorde, dan is de
        non-negativiteitsassert in `realized.py` de enige die dat vangt — en
        QLIKE zou er `ln(negatief)` van maken.
        """
        one = np.array([100.0])
        assert parkinson_variance(one, one)[0] == 0.0
        assert garman_klass_variance(one, one, one, one)[0] == 0.0
        assert rogers_satchell_variance(one, one, one, one)[0] == 0.0

    def test_efficiency_table_covers_every_proxy(self) -> None:
        bars = _simulate_gbm_bars(200, 0.02)
        assert set(build_range_proxies(bars)) == set(PROXY_EFFICIENCY)


class TestOhlcContractIsEnforcedNotAssumed:
    """De premisse onder alle drie de bewijzen wordt gemeten, niet aangenomen."""

    @pytest.mark.parametrize(
        ("field", "value"),
        [("high", 99.0), ("low", 101.0)],
        ids=["high-below-close", "low-above-open"],
    )
    def test_impossible_bar_is_rejected(self, field: str, value: float) -> None:
        bars = pd.DataFrame({
            "open": [100.0], "high": [100.5], "low": [99.5], "close": [100.0]})
        bars.loc[0, field] = value
        with pytest.raises(DataContractError, match="OHLC-contract"):
            build_range_proxies(bars, name="BROKEN")

    def test_non_positive_price_is_rejected_before_any_logarithm(self) -> None:
        bars = pd.DataFrame({
            "open": [100.0], "high": [101.0], "low": [0.0], "close": [100.0]})
        with pytest.raises(DataContractError, match="niet-positieve prijzen"):
            build_range_proxies(bars, name="ZERO")

    def test_nan_is_a_finding_not_something_to_impute(self) -> None:
        bars = pd.DataFrame({
            "open": [100.0], "high": [np.nan], "low": [99.0], "close": [100.0]})
        with pytest.raises(DataContractError, match="niet-eindige"):
            build_range_proxies(bars, name="GAP")

    def test_violation_slips_past_the_proof_and_is_caught_by_the_measurement(
        self,
    ) -> None:
        """De negatieve controle op de bewijzen zelf (§0.10).

        `garman_klass_variance` wordt hier RECHTSTREEKS aangeroepen op een bar
        die het contract schendt — dus met de poort omzeild. Het bewijs zegt
        `0,5 x^2 >= 0,383 y^2` zolang `x >= |y|`; hier is dat niet zo en moet
        `_assert_non_negative` alsnog aanslaan. Zonder deze test bewijst de
        groene suite alleen dat het contract nooit werd geschonden, niet dat de
        meting eronder werkt.
        """
        o = np.array([100.0])
        c = np.array([130.0])   # close ver boven high: |ln(C/O)| > ln(H/L)
        h = np.array([100.6])
        lo = np.array([99.4])
        with pytest.raises(DataContractError, match="negatieve variantie"):
            garman_klass_variance(o, h, lo, c)


class TestUnbiasedness:
    """Zijn de overgeschreven constanten de juiste?"""

    @pytest.mark.parametrize(
        ("estimator", "name"),
        [(parkinson_variance, "parkinson"),
         (garman_klass_variance, "garman_klass"),
         (rogers_satchell_variance, "rogers_satchell")],
    )
    def test_estimator_recovers_the_true_variance(self, estimator, name) -> None:
        sigma = 0.03
        bars = _simulate_gbm_bars(
            40_000, sigma, steps=500, seed=SEED + 7, chain=False)
        o, h, lo, c = (bars[k].to_numpy() for k in ("open", "high", "low", "close"))
        values = (
            estimator(h, lo) if name == "parkinson" else estimator(o, h, lo, c))
        # Discrete bemonstering van een continu pad ONDERSCHAT de range, dus
        # elke range-estimator zit met eindige `steps` iets onder de waarheid.
        # De band is daarom eenzijdig ruim; een typefout in `4 ln 2` of in
        # `2 ln 2 - 1` verschuift het gemiddelde met tientallen procenten en
        # valt er ruim buiten.
        ratio = float(values.mean()) / sigma**2
        assert 0.90 <= ratio <= 1.05, f"{name}: E[proxy]/sigma^2 = {ratio:.4f}"

    def test_rogers_satchell_survives_drift_where_parkinson_does_not(self) -> None:
        """Het onderscheidende kenmerk van Rogers-Satchell, gemeten.

        Parkinson en Garman-Klass gaan uit van een driftloze GBM binnen de bar.
        Bij een sterke drift meet de range deels de RICHTING in plaats van de
        variantie en overschatten zij. Rogers-Satchell is ook onder drift
        zuiver. Op crypto-dagbars is dit geen theoretisch detail, en het is de
        reden dat de proxykeuze in het rapport hoort.
        """
        sigma, drift = 0.03, 0.12
        bars = _simulate_gbm_bars(
            40_000, sigma, mu=drift, steps=500, seed=SEED + 11, chain=False)
        o, h, lo, c = (bars[k].to_numpy() for k in ("open", "high", "low", "close"))
        rs = float(rogers_satchell_variance(o, h, lo, c).mean()) / sigma**2
        pk = float(parkinson_variance(h, lo).mean()) / sigma**2
        assert rs < pk, f"RS {rs:.4f} zou dichter bij 1 moeten liggen dan PK {pk:.4f}"
        assert abs(rs - 1.0) < abs(pk - 1.0)


class TestWindowAndBurnInSemantics:
    """Geen `ffill`, geen `fillna(0.0)` — dat was de hele reden voor dit module."""

    def test_squared_return_leaves_bar_zero_undefined(self) -> None:
        out = squared_return_variance(np.array([100.0, 101.0, 102.0]))
        assert np.isnan(out[0])
        assert np.all(np.isfinite(out[1:]))

    def test_yang_zhang_burn_in_stays_nan(self) -> None:
        bars = _simulate_gbm_bars(200, 0.02)
        o, h, lo, c = (bars[k].to_numpy() for k in ("open", "high", "low", "close"))
        out = yang_zhang_variance(o, h, lo, c, window=20)
        assert np.all(np.isnan(out[:20]))
        assert np.any(np.isfinite(out[20:]))

    def test_yang_zhang_returns_all_nan_when_series_shorter_than_window(
        self,
    ) -> None:
        bars = _simulate_gbm_bars(10, 0.02)
        o, h, lo, c = (bars[k].to_numpy() for k in ("open", "high", "low", "close"))
        assert np.all(np.isnan(yang_zhang_variance(o, h, lo, c, window=20)))

    def test_zero_bars_are_counted_not_dropped(self) -> None:
        """Een nul-variantiebar is onbruikbaar voor QLIKE en moet zichtbaar zijn.

        `ln(0) = -inf`. Het rapport moet kunnen tonen hoeveel van de steekproef
        daaraan opgaat; stilzwijgend wegfilteren zou de competitie op een
        onbekende deelverzameling beslechten.
        """
        bars = pd.DataFrame({
            "open": [100.0, 100.0, 101.0],
            "high": [100.0, 102.0, 101.0],
            "low": [100.0, 99.0, 101.0],
            "close": [100.0, 101.0, 101.0]})
        proxies = build_range_proxies(bars, name="FLAT")
        assert proxies["parkinson"].n_zero_bars == 2
        assert proxies["parkinson"].n_valid_bars == 1


class TestIntradayRealizedVarianceCrashesInsteadOfSubstituting:
    """De belangrijkste beperking van de fase mag niet onzichtbaar worden."""

    @pytest.mark.parametrize("empty", [None, {}])
    def test_no_intraday_bars_is_a_crash_not_a_daily_fallback(self, empty) -> None:
        with pytest.raises(DependencyMissingError) as excinfo:
            realized_variance_from_intraday(empty, granularity="5m")
        message = str(excinfo.value)
        # De boodschap moet de LEZER van de traceback vertellen waarom er geen
        # terugval is. Een kale `KeyError` zou de volgende ontwikkelaar ertoe
        # verleiden er alsnog een range-estimator in te hangen.
        assert "NIET" in message
        assert "range-estimator" in message


class TestRequireValidOhlcDirectly:
    def test_accepts_bars_that_satisfy_the_premise(self) -> None:
        bars = _simulate_gbm_bars(500, 0.04)
        require_valid_ohlc(
            *(bars[k].to_numpy() for k in ("open", "high", "low", "close")),
            name="OK")
