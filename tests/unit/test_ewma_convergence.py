"""EWMA-eigenschappen — Phase 3, deliverable 11.

De RiskMetrics-recursie is de PRODUCTIE-vol-estimator van dit platform: hij
bepaalt via Naive Risk Parity elke positiegrootte. Een schatter met die rol
verdient toetsen op zijn wiskundige eigenschappen, niet alleen op "draait
zonder crash".

Getoetst worden:
  * de decay is exact geometrisch met factor lambda;
  * een schok landt met precies gewicht `(1 - lambda)` in de variantie;
  * de burn-in is NaN en de eerste waarde valt exact op de gedeclareerde grens;
  * de recursie is causaal, dus truncatie-invariant;
  * herhaalde runs op dezelfde input zijn bit-identiek;
  * lambda en de burn-in komen uit `conf/model/volatility.yaml`;
  * er wordt NOOIT stilzwijgend een volatiliteit van nul geproduceerd.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tradebot.schemas.config import VolatilityConfig, load_config
from tradebot.utils.failfast import DataContractError
from tradebot.volatility.ewma import (
    ewma_variance_causal,
    ewma_volatility,
    ewma_volatility_panel,
)

ROOT = Path(__file__).resolve().parents[2]
VOL_CFG = load_config(ROOT / "conf" / "model" / "volatility.yaml", VolatilityConfig)

SEED = 20260823
N_BARS = 400
BURN_IN = 20


def _index(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2021-01-01", periods=n, freq="1D", tz="UTC")


def _returns(values: list[float] | np.ndarray) -> pd.Series:
    """Returns met een leidende NaN, zoals `diff()` die oplevert."""
    arr = np.concatenate([[np.nan], np.asarray(values, dtype="float64")])
    return pd.Series(arr, index=_index(len(arr)))


def _random_close(*, n: int = N_BARS, seed: int = SEED) -> pd.Series:
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.02, size=n)))
    return pd.Series(close, index=_index(n))


class TestDecayIsGeometric:
    def test_variance_decays_by_lambda_each_bar_after_a_shock(self) -> None:
        """Met r = 0 na de schok geldt sigma^2_{t+k} = lambda^k * sigma^2_t."""
        lam = float(VOL_CFG.ewma_lambda)
        rets = np.zeros(N_BARS, dtype="float64")
        rets[:BURN_IN] = 0.01           # voedt de seed
        rets[BURN_IN + 1] = 0.20        # de schok
        var = ewma_variance_causal(_returns(rets), lam=lam, burn_in_bars=BURN_IN)

        shock_pos = BURN_IN + 2         # +1 voor de leidende NaN
        after = var.to_numpy()[shock_pos:]
        ratios = after[1:11] / after[:10]
        np.testing.assert_allclose(ratios, lam, rtol=1e-12)

    def test_half_life_matches_the_analytic_value(self) -> None:
        """lambda^h = 1/2 geeft h = ln(0.5)/ln(lambda); ~11.2 bars bij 0.94."""
        lam = float(VOL_CFG.ewma_lambda)
        half_life = math.log(0.5) / math.log(lam)
        rets = np.zeros(N_BARS, dtype="float64")
        rets[:BURN_IN] = 0.01
        rets[BURN_IN + 1] = 0.20
        var = ewma_variance_causal(_returns(rets), lam=lam, burn_in_bars=BURN_IN)
        arr = var.to_numpy()
        shock_pos = BURN_IN + 2
        k = int(round(half_life))
        assert arr[shock_pos + k] / arr[shock_pos] == pytest.approx(0.5, rel=0.05)

    def test_shock_enters_with_weight_one_minus_lambda(self) -> None:
        """sigma^2_t - lambda*sigma^2_{t-1} is exact (1 - lambda) * r_t^2."""
        lam = float(VOL_CFG.ewma_lambda)
        shock = 0.20
        rets = np.zeros(N_BARS, dtype="float64")
        rets[:BURN_IN] = 0.01
        rets[BURN_IN + 1] = shock
        var = ewma_variance_causal(_returns(rets), lam=lam, burn_in_bars=BURN_IN)
        arr = var.to_numpy()
        pos = BURN_IN + 2
        contribution = arr[pos] - lam * arr[pos - 1]
        assert contribution == pytest.approx((1.0 - lam) * shock**2, rel=1e-12)


class TestBurnInIsHonest:
    def test_burn_in_is_nan_and_the_boundary_is_exact(self) -> None:
        vol = ewma_volatility(
            _random_close(), lam=VOL_CFG.ewma_lambda, burn_in_bars=BURN_IN,
            annualisation_factor=VOL_CFG.annualisation_factor,
        )
        assert vol.iloc[:BURN_IN].isna().all(), "burn-in bevat een eindige waarde"
        assert vol.iloc[BURN_IN:].notna().all(), "na de burn-in ontbreekt een waarde"
        assert int(vol.isna().sum()) == BURN_IN

    def test_no_implicit_zero_volatility_is_ever_produced(self) -> None:
        """DI-12: fillna(0.0) leverde een oneindig risk-parity gewicht op."""
        vol = ewma_volatility(
            _random_close(), lam=VOL_CFG.ewma_lambda, burn_in_bars=BURN_IN,
            annualisation_factor=VOL_CFG.annualisation_factor,
        )
        assert not bool((vol.dropna() == 0.0).any()), (
            "er is een volatiliteit van exact nul geproduceerd; in Naive Risk "
            "Parity is dat een oneindig gewicht"
        )

    def test_series_shorter_than_the_burn_in_is_all_nan(self) -> None:
        """Te weinig historie levert geen schatting op, niet een gegokte."""
        vol = ewma_volatility(
            _random_close(n=BURN_IN), lam=VOL_CFG.ewma_lambda,
            burn_in_bars=BURN_IN,
            annualisation_factor=VOL_CFG.annualisation_factor,
        )
        assert vol.isna().all()


class TestCausality:
    def test_recursion_is_truncation_invariant(self) -> None:
        close = _random_close()
        full = ewma_volatility(
            close, lam=VOL_CFG.ewma_lambda, burn_in_bars=BURN_IN,
            annualisation_factor=VOL_CFG.annualisation_factor,
        )
        for cut in (N_BARS // 2, 3 * N_BARS // 4, N_BARS - 2):
            truncated = ewma_volatility(
                close.iloc[: cut + 1], lam=VOL_CFG.ewma_lambda,
                burn_in_bars=BURN_IN,
                annualisation_factor=VOL_CFG.annualisation_factor,
            )
            a = truncated.to_numpy()
            b = full.iloc[: cut + 1].to_numpy()
            assert (np.isnan(a) == np.isnan(b)).all()
            np.testing.assert_array_equal(a[~np.isnan(a)], b[~np.isnan(b)])

    def test_a_future_shock_cannot_change_the_past(self) -> None:
        """Een enorme return op de laatste bar raakt geen enkele eerdere waarde."""
        close = _random_close()
        tampered = close.copy()
        tampered.iloc[-1] = float(close.iloc[-1]) * 5.0
        kw = dict(lam=VOL_CFG.ewma_lambda, burn_in_bars=BURN_IN,
                  annualisation_factor=VOL_CFG.annualisation_factor)
        base = ewma_volatility(close, **kw).to_numpy()[:-1]
        after = ewma_volatility(tampered, **kw).to_numpy()[:-1]
        np.testing.assert_array_equal(base[~np.isnan(base)], after[~np.isnan(after)])


class TestDeterminism:
    def test_repeated_runs_are_bit_identical(self) -> None:
        kw = dict(lam=VOL_CFG.ewma_lambda, burn_in_bars=BURN_IN,
                  annualisation_factor=VOL_CFG.annualisation_factor)
        first = ewma_volatility(_random_close(seed=SEED), **kw)
        second = ewma_volatility(_random_close(seed=SEED), **kw)
        pd.testing.assert_series_equal(first, second, check_exact=True)

    def test_panel_matches_the_per_column_estimate(self) -> None:
        """Er wordt niet over symbolen gepoold; elke kolom staat op zichzelf."""
        kw = dict(lam=VOL_CFG.ewma_lambda, burn_in_bars=BURN_IN,
                  annualisation_factor=VOL_CFG.annualisation_factor)
        panel = pd.DataFrame({
            "A": _random_close(seed=SEED),
            "B": _random_close(seed=SEED + 1),
        })
        wide = ewma_volatility_panel(panel, **kw)
        for col in panel.columns:
            expected = ewma_volatility(panel[col], **kw)
            np.testing.assert_array_equal(
                wide[col].to_numpy(), expected.to_numpy()
            )
        assert (wide.dtypes == np.dtype("float64")).all()


class TestParametersComeFromConfig:
    def test_binding_riskmetrics_lambda_lives_in_conf(self) -> None:
        """Audit sectie 9.1/22: Level 1 = EWMA met lambda = 0.94."""
        assert VOL_CFG.estimator == "ewma"
        assert VOL_CFG.ewma_lambda == pytest.approx(0.94)
        assert VOL_CFG.burn_in_bars > 1

    @pytest.mark.parametrize("lam", [0.0, 1.0, -0.5, 1.5])
    def test_lambda_outside_the_unit_interval_crashes(self, lam: float) -> None:
        with pytest.raises(DataContractError, match="lambda"):
            ewma_variance_causal(_returns(np.zeros(N_BARS)), lam=lam,
                                 burn_in_bars=BURN_IN)

    @pytest.mark.parametrize("burn_in", [0, 1])
    def test_burn_in_below_two_returns_crashes(self, burn_in: int) -> None:
        with pytest.raises(DataContractError, match="twee returns"):
            ewma_variance_causal(_returns(np.zeros(N_BARS)),
                                 lam=VOL_CFG.ewma_lambda, burn_in_bars=burn_in)

    def test_non_positive_price_crashes(self) -> None:
        close = _random_close()
        close.iloc[5] = 0.0
        with pytest.raises(DataContractError, match="Niet-positieve prijs"):
            ewma_volatility(close, lam=VOL_CFG.ewma_lambda, burn_in_bars=BURN_IN,
                            annualisation_factor=VOL_CFG.annualisation_factor)
