# tests/unit/test_volatility_edges.py
"""De volatiliteitspaden die de hoofdtests niet raken.

NUMBA-KERNELS
-------------
`coverage.py` ziet niets binnen een `@njit`-functie: die draait als machinecode.
De kernels worden daarom twee keer aangeroepen -- gecompileerd én via
`.py_func`, de Python-bron -- en moeten op afrondingsniveau dezelfde uitkomst
geven (gemeten verschil ~1e-14 relatief). Bitidentiek is het niet: LLVM mag
`log`/`sqrt` en de optelvolgorde anders uitvoeren dan numpy. Daarnaast een
absolute marge van 1e-12, want de sprongvariantie is `sqrt(max(RV - BV, 0))`:
waar RV en BV bijna gelijk zijn valt het verschil weg en vergroot de wortel
een afrondingsfout van 1e-17 tot ~1e-14 absoluut. Dat dekt de bronregels en
bewijst tegelijk dat de gecompileerde versie hetzelfde rekent als wat er staat.
"""
from __future__ import annotations

import dataclasses
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tradebot.cv.walk_forward import WalkForwardCV
from tradebot.schemas.config import AdequacyConfig, load_config
from tradebot.utils.failfast import DataContractError
from tradebot.volatility import garch as garch_mod
from tradebot.volatility import har_rv as har_module
from tradebot.volatility.garman_klass import (
    _garman_klass_kernel,
    _jump_variance_kernel,
    get_garman_klass_volatility,
    get_jump_adjusted_volatility,
)
from tradebot.volatility.parkinson import _parkinson_kernel, get_parkinson_volatility
from tradebot.volatility.realized import (
    build_range_proxies,
    realized_variance_from_intraday,
)
from tradebot.volatility.rogers_satchell import get_rogers_satchell_volatility
from tradebot.volatility.yang_zhang import get_yang_zhang_volatility

ROOT = Path(__file__).resolve().parents[2]
ADEQUACY = load_config(ROOT / "conf/model/adequacy.yaml", AdequacyConfig)
RNG = np.random.default_rng(41)


def _same(compiled: np.ndarray, source: np.ndarray) -> None:
    np.testing.assert_allclose(compiled, source, rtol=1e-12, atol=1e-12,
                               equal_nan=True)


def _ohlc(n: int = 80, freq: str | None = "D") -> pd.DataFrame:
    close = 100.0 * np.exp(np.cumsum(RNG.normal(0.0, 0.02, n)))
    open_ = close * np.exp(RNG.normal(0.0, 0.005, n))
    high = np.maximum(open_, close) * (1.0 + np.abs(RNG.normal(0.0, 0.01, n)))
    low = np.minimum(open_, close) * (1.0 - np.abs(RNG.normal(0.0, 0.01, n)))
    index = pd.date_range("2022-01-01", periods=n, freq=freq or "D", tz="UTC")
    frame = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close},
                         index=index)
    if freq is None:
        frame.index = pd.DatetimeIndex(list(frame.index))  # zonder freq
    return frame


class TestKernelsMatchTheirPythonSource:
    def test_garman_klass(self) -> None:
        f = _ohlc()
        args = (f["open"].to_numpy(), f["high"].to_numpy(), f["low"].to_numpy(),
                f["close"].to_numpy(), 14)
        _same(_garman_klass_kernel(*args), _garman_klass_kernel.py_func(*args))
        short = tuple(a[:5] if isinstance(a, np.ndarray) else a for a in args)
        assert np.isnan(_garman_klass_kernel.py_func(*short)).all()

    def test_parkinson(self) -> None:
        f = _ohlc()
        h, lo = f["high"].to_numpy(), f["low"].to_numpy()
        _same(_parkinson_kernel(h, lo, 14), _parkinson_kernel.py_func(h, lo, 14))
        assert np.isnan(_parkinson_kernel.py_func(h[:5], lo[:5], 14)).all()

    def test_jump_variance_dense_and_sparse(self) -> None:
        close = _ohlc()["close"].to_numpy()
        _same(_jump_variance_kernel(close, 10), _jump_variance_kernel.py_func(close, 10))
        # Een reeks die vooral stilstaat: minder dan de helft van de paren is
        # niet-nul, dus de schaarste-correctie neemt de andere tak.
        sparse = np.full(60, 100.0)
        sparse[::7] = 101.0
        sparse[30] = 0.0  # ongeldige prijs: return 0
        _same(_jump_variance_kernel(sparse, 10), _jump_variance_kernel.py_func(sparse, 10))
        assert np.isnan(_jump_variance_kernel.py_func(close[:5], 10)).all()


class TestEstimatorWrappers:
    def test_short_frames_give_zero(self) -> None:
        short = _ohlc(5)
        assert (get_garman_klass_volatility(short) == 0.0).all()
        assert (get_jump_adjusted_volatility(short) == 0.0).all()
        assert (get_parkinson_volatility(short) == 0.0).all()
        assert (get_rogers_satchell_volatility(short) == 0.0).all()
        assert (get_yang_zhang_volatility(short) == 0.0).all()

    def test_garman_klass_falls_back_to_parkinson_without_an_open(self) -> None:
        frame = _ohlc()
        without_open = get_garman_klass_volatility(frame.drop(columns="open"))
        assert without_open.iloc[-1] == pytest.approx(
            get_parkinson_volatility(frame).iloc[-1])

    def test_jump_adjusted_and_rogers_satchell_in_both_units(self) -> None:
        frame = _ohlc()
        relative = get_jump_adjusted_volatility(frame, window=10)
        absolute = get_jump_adjusted_volatility(frame, window=10, absolute=True)
        assert absolute.iloc[-1] == pytest.approx(
            relative.iloc[-1] * frame["close"].iloc[-1])
        assert (relative >= get_garman_klass_volatility(frame, window=10) - 1e-12).all()
        rs = get_rogers_satchell_volatility(frame, absolute=True)
        assert rs.iloc[-1] > 0.0


class TestYangZhangCadence:
    def test_intraday_bars_are_refused(self) -> None:
        with pytest.raises(ValueError, match="daily"):
            get_yang_zhang_volatility(_ohlc(freq="h"))

    @pytest.mark.parametrize("freq", ["D", "W-SUN", "MS"])
    def test_daily_and_longer_bars_are_accepted(self, freq: str) -> None:
        """Week- en maandbars werden geweigerd: `.nanos` bestaat niet voor een
        niet-vaste offset en die ValueError werd als intraday doorgegooid."""
        assert get_yang_zhang_volatility(_ohlc(freq=freq)).iloc[-1] > 0.0

    def test_without_a_freq_the_spacing_is_inferred_and_warned(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        frame = _ohlc(freq="h")
        frame.index = pd.DatetimeIndex(list(frame.index))
        assert frame.index.freq is None
        with caplog.at_level(logging.WARNING):
            get_yang_zhang_volatility(frame)
        assert "median spacing" in caplog.text


class TestHarRv:
    @staticmethod
    def _rv(n: int = 120) -> np.ndarray:
        return np.abs(RNG.normal(1e-4, 3e-5, n)) + 1e-6

    def test_too_short_a_series_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least"):
            har_module.har_rv_fit(self._rv(10))

    def test_fit_indices_select_the_training_rows(self) -> None:
        rv = self._rv()
        train_only = har_module.har_rv_fit(rv, fit_indices=np.arange(23, 80))
        assert train_only.forecast.shape == rv.shape

    def test_too_few_fit_rows_fall_back_to_the_full_series(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        rv = self._rv()
        with caplog.at_level(logging.WARNING):
            fallback = har_module.har_rv_fit(rv, fit_indices=np.arange(3))
        assert "falling back" in caplog.text
        assert fallback.beta_d == har_module.har_rv_fit(rv).beta_d

    def test_a_long_series_without_fit_indices_warns(self) -> None:
        with pytest.warns(UserWarning, match="without fit_indices"):
            har_module.har_rv_fit(self._rv(5_100))

    def test_a_failing_nnls_is_refused_not_replaced(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def _fail(*args, **kwargs):
            raise RuntimeError("Maximum number of iterations reached.")

        monkeypatch.setattr(har_module, "nnls", _fail)
        with pytest.raises(DataContractError, match="convergeerde niet"):
            har_module.har_rv_fit(self._rv())


class TestRealized:
    def test_intraday_closes_become_daily_realised_variance(self) -> None:
        rv = realized_variance_from_intraday(
            {"2024-01-02": np.array([100.0, 101.0, 100.0]),
             "2024-01-01": np.array([100.0, 100.0])}, granularity="5m")
        assert rv[0] == 0.0
        assert rv[1] == pytest.approx(np.log(1.01) ** 2 + np.log(100 / 101) ** 2)

    def test_every_range_proxy_serialises(self) -> None:
        proxies = build_range_proxies(_ohlc(), name="BTC")
        for proxy in proxies.values():
            record = proxy.as_record()
            assert record["n_valid_bars"] == proxy.n_valid_bars


class TestGarch:
    def test_an_unknown_persistence_kind_is_a_programming_error(self) -> None:
        spec = dataclasses.replace(garch_mod.GARCH_FAMILY["garch"],
                                   persistence_kind="bogus")  # type: ignore[arg-type]
        with pytest.raises(AssertionError, match="onbekende persistence_kind"):
            garch_mod._persistence(spec, {})

    def test_a_real_fit_converges_and_reports_its_persistence(self) -> None:
        """De gewone, geslaagde fit: tot nu toe alleen in trage tests gedraaid."""
        n, omega, alpha, beta = 400, 0.05, 0.08, 0.9
        z = RNG.standard_t(8, n) / np.sqrt(8 / 6)
        r, var = np.empty(n), omega / (1 - alpha - beta)
        for t in range(n):
            r[t] = np.sqrt(var) * z[t]
            var = omega + alpha * r[t] ** 2 + beta * var
        returns = pd.Series(r / 100.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = garch_mod.fit_garch_window(
                returns, garch_mod.GARCH_FAMILY["garch"], ADEQUACY,
                symbol="SYN", fold_id=0, train_end=350)
        assert fit.converged is True
        assert 0.0 < fit.persistence < 1.0
        summary = garch_mod.summarise_convergence([fit])
        assert summary.as_record()["n_fits"] == 1

    def test_a_cv_without_training_rows_is_a_configuration_error(self) -> None:
        cv = WalkForwardCV(train_size=5, test_size=20, step=20, mode="rolling",
                           min_train=5, embargo_bars=10)
        with pytest.raises(DataContractError, match="geen enkele fold"):
            garch_mod.walk_forward_variance_forecasts(
                pd.Series(RNG.normal(0, 0.01, 100)), garch_mod.GARCH_FAMILY["garch"],
                cv, ADEQUACY, symbol="SYN", horizon=1)
