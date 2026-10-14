# tests/unit/test_features_fracdiff_edges.py
"""`features/fracdiff.py`: gewichten, de causale convolutie, de d*-zoektocht en de transformer.

De ADF-test is voor de zoeklogica gestubd: `frac_diff_ffd` geeft een constante
reeks met de waarde `d` terug en `adfuller` leest daar zijn p-waarde uit, zodat
de bisectie op een bekende drempel (d = 0,4) moet landen.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from tradebot.features import fracdiff as fd


def _walk(n: int = 1500, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=n, freq="D", tz="UTC")
    return pd.Series(100.0 + np.cumsum(rng.normal(0.0, 1.0, n)), index=idx, name="px")


@pytest.fixture
def encoded_d(monkeypatch: pytest.MonkeyPatch):
    """`frac_diff_ffd` -> constante reeks met waarde d; `adfuller` -> p uit die waarde."""
    monkeypatch.setattr(fd, "frac_diff_ffd",
                        lambda series, d, threshold=0.0: pd.Series(np.full(100, d)))

    def install(p_of_d):
        monkeypatch.setattr(fd, "adfuller", lambda y, **kw: (0.0, p_of_d(float(y.iloc[0]))))
    return install


class TestWeights:
    @pytest.mark.parametrize("d", [0.0, 1.0, -0.2, 1.5])
    def test_d_must_lie_strictly_inside_the_unit_interval(self, d: float) -> None:
        with pytest.raises(ValueError, match="d must lie in"):
            fd._ffd_weights(d, 1e-5)

    def test_the_threshold_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="threshold must be > 0"):
            fd._ffd_weights(0.5, 0.0)

    def test_weights_start_at_one_and_stop_at_the_threshold(self) -> None:
        w = fd._ffd_weights(0.4, 1e-4)
        assert isinstance(w, np.ndarray) and w[0] == 1.0 and w[1] == pytest.approx(-0.4)
        assert np.abs(w[-1]) >= 1e-4

    def test_the_residual_mass_estimate_is_returned_on_request_and_a_big_one_is_logged(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.DEBUG, logger=fd.logger.name):
            w, residual = fd._ffd_weights(0.9, 1e-2, return_residual=True)
        assert residual > fd.FFD_RESIDUAL_MASS_WARN and "residual-mass ratio" in caplog.text
        assert len(w) > 1


class TestFracDiffFfd:
    def test_it_refuses_anything_but_a_series(self) -> None:
        with pytest.raises(TypeError, match=r"expects a pd\.Series"):
            fd.frac_diff_ffd(np.arange(10.0), d=0.4)                  # type: ignore[arg-type]

    def test_a_series_shorter_than_the_weight_vector_gives_an_empty_result(self) -> None:
        out = fd.frac_diff_ffd(_walk(5), d=0.4, threshold=1e-5)
        assert out.empty and out.name == "px"

    def test_the_output_is_causal_and_starts_after_the_weight_window(self) -> None:
        s = _walk(400)
        w = fd._ffd_weights(0.4, 1e-3)
        out = fd.frac_diff_ffd(s, d=0.4, threshold=1e-3)
        assert len(out) == len(s) - len(w) + 1 and out.index[0] == s.index[len(w) - 1]
        t = out.index[10]
        cut = fd.frac_diff_ffd(s.loc[:t], d=0.4, threshold=1e-3)      # latere data verandert niets
        assert cut.loc[t] == pytest.approx(out.loc[t])
        changed = s.copy()
        changed.loc[changed.index > t] += 1_000.0
        assert fd.frac_diff_ffd(changed, d=0.4, threshold=1e-3).loc[t] == pytest.approx(out.loc[t])

    @pytest.mark.xfail(strict=True, reason=(
        "BEVINDING, niet aangepast (pre-registratie): `np.convolve(values, w[::-1])` kruist het "
        "venster om. Het gewicht w[0] = 1 komt op de OUDSTE waarde in het venster te staan en x_t "
        "krijgt het kleinste gewicht w[L-1]. Dat wijkt af van AFML 5.3.2 en van de docstring van "
        "`_ffd_weights` (w[0] hoort bij x_t). Causaal blijft het wel; de amplituderespons is gelijk."))
    def test_the_current_bar_carries_the_unit_weight(self) -> None:
        w = fd._ffd_weights(0.4, 1e-3)
        spike = np.zeros(len(w) + 5)
        spike[len(w) + 2] = 1.0
        out = fd.frac_diff_ffd(pd.Series(spike), d=0.4, threshold=1e-3)
        assert out.loc[len(w) + 2] == pytest.approx(1.0)


class TestMinFracDiff:
    def test_bisection_lands_on_the_smallest_rejecting_d(self, encoded_d) -> None:
        encoded_d(lambda d: 0.01 if d >= 0.4 else 0.5)
        res = fd.min_frac_diff(_walk(100))
        assert res.converged and 0.4 <= res.d < 0.4 + 1e-5 and res.pvalue == 0.01

    def test_when_even_the_largest_d_is_not_enough_it_reports_non_convergence(self, encoded_d) -> None:
        encoded_d(lambda d: 0.9)
        res = fd.min_frac_diff(_walk(100))
        assert not res.converged and res.d == 0.95 and res.pvalue == 0.9

    def test_when_the_smallest_d_already_rejects_it_is_returned_directly(self, encoded_d) -> None:
        encoded_d(lambda d: 0.001)
        res = fd.min_frac_diff(_walk(100), d_lo=0.1)
        assert res.converged and res.d == 0.1 and res.n_used_obs == 100

    @pytest.mark.parametrize("error", [ValueError("degenerate"), np.linalg.LinAlgError("singular")])
    def test_an_adf_failure_counts_as_not_stationary(self, monkeypatch: pytest.MonkeyPatch, error) -> None:
        def boom(y, **kw):
            raise error
        monkeypatch.setattr(fd, "adfuller", boom)
        res = fd.min_frac_diff(_walk(600))
        assert not res.converged and res.pvalue == 1.0

    def test_too_few_differenced_observations_count_as_not_stationary(self) -> None:
        res = fd.min_frac_diff(_walk(60))                              # venster langer dan de reeks
        assert not res.converged and res.pvalue == 1.0 and res.n_used_obs == 0

    def test_a_real_random_walk_needs_a_d_strictly_between_the_bounds(self) -> None:
        res = fd.min_frac_diff(_walk(1500), threshold=1e-3)
        assert res.converged and 0.05 <= res.d <= 0.95 and res.pvalue <= 0.05


class TestCausalMinFracDiff:
    def test_too_short_a_train_slice_falls_back_to_the_upper_bound(self) -> None:
        assert fd.causal_min_frac_diff(_walk(300), train_end_iloc=40, d_hi=0.8) == 0.8

    def test_only_the_train_slice_decides_d(self) -> None:
        s = _walk(1500)
        train = fd.causal_min_frac_diff(s, train_end_iloc=1000, threshold=1e-3)
        assert train == fd.min_frac_diff(s.iloc[:1000], d_lo=0.1, d_hi=0.9, threshold=1e-3).d
        s2 = s.copy()
        s2.iloc[1000:] = s2.iloc[1000:] * 7.0 + 500.0                # alles na de train-grens verandert
        assert fd.causal_min_frac_diff(s2, train_end_iloc=1000, threshold=1e-3) == train


class TestMinFracDiffTransformer:
    def test_transform_before_fit_is_refused(self) -> None:
        with pytest.raises(RuntimeError, match="before fit"):
            fd.MinFracDiff().transform(pd.DataFrame({"a": [1.0]}))

    def test_fit_skips_excluded_and_short_columns_and_transform_skips_missing_ones(self) -> None:
        df = pd.DataFrame({"px": _walk(500, seed=1), "skip": _walk(500, seed=2),
                           "short": _walk(500, seed=3).where(lambda s: s.index < s.index[100])})
        model = fd.MinFracDiff(threshold=1e-3, exclude=("skip",)).fit(df)
        assert set(model.d_per_col_) == {"px"} and model.is_fitted_
        out = model.transform(df.drop(columns=["px"]).assign(other=1.0))
        assert out.empty and out.index.equals(df.index)              # niets te differentiëren
        full = model.transform(df)
        assert list(full.columns) == [f"ffd_px_d{model.d_per_col_['px']:.3f}"]
        assert isinstance(full.index, pd.DatetimeIndex)

    def test_fit_transform_is_fit_followed_by_transform(self) -> None:
        df = pd.DataFrame({"px": _walk(500, seed=4)})
        a = fd.MinFracDiff(threshold=1e-3).fit_transform(df)
        b = fd.MinFracDiff(threshold=1e-3).fit(df).transform(df)
        pd.testing.assert_frame_equal(a, b)
