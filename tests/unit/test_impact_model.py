"""Het marktimpactmodel — de vorm, de verplichte provenance, en de negatieve tests.

Fase-opdracht §11 en §17. De kern van deze suite is niet dat het model correct
rekent (dat is één test); het is dat het WEIGERT te rekenen zonder expliciete,
herleidbare parameters. Dat is precies wat
`execution/market_impact.py::square_root_impact(eta=0.142)` niet deed.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tradebot.execution.impact_calibration import (
    ETA_BOUND_QUANTILE,
    REQUIRED_DATASETS,
    calibrate_impact,
    missing_calibration_datasets,
    range_implied_eta,
)
from tradebot.execution.impact_model import (
    ImpactParams,
    ImpactStatus,
    square_root_impact,
)
from tradebot.schemas.config import ImpactConfig, load_config
from tradebot.utils.failfast import ConfigContractError, DataContractError

ROOT = Path(__file__).resolve().parents[2]
CONF = ROOT / "conf"
ARTEFACT = ROOT / "artefacts" / "execution" / "impact_params.json"


@pytest.fixture(scope="module")
def cfg() -> ImpactConfig:
    return load_config(CONF / "execution/impact.yaml", ImpactConfig)


@pytest.fixture(scope="module")
def params(cfg: ImpactConfig) -> ImpactParams:
    return ImpactParams(
        eta=cfg.eta, kappa_d=cfg.kappa_d, status=ImpactStatus(cfg.status),
        method=cfg.method, data_hash=cfg.data_hash, sample_size=cfg.sample_size,
        period_start=cfg.period_start, period_end=cfg.period_end,
        instruments=cfg.instruments, eta_ci_low=cfg.eta_ci_low,
        eta_ci_high=cfg.eta_ci_high,
    )


def _params(**overrides: object) -> ImpactParams:
    base: dict[str, object] = {
        "eta": 0.5, "kappa_d": 0.5, "status": ImpactStatus.CALIBRATED,
        "method": "test", "data_hash": "deadbeef", "sample_size": 10,
        "period_start": "2020-01-01", "period_end": "2020-12-31",
        "instruments": ("BTCUSDT",), "eta_ci_low": 0.4, "eta_ci_high": 0.6,
    }
    base.update(overrides)
    return ImpactParams(**base)  # type: ignore[arg-type]


# =========================================================================== #
# 1. De vorm van het model is de voorgeschreven vorm
# =========================================================================== #
class TestModelForm:
    def test_the_formula_is_eta_sigma_sqrt_participation(self) -> None:
        p = _params(eta=0.5)
        est = square_root_impact(order_notional=1_000.0, adv_notional=100_000.0,
                                 sigma_daily=0.04, params=p)
        assert est.impact_fraction == pytest.approx(0.5 * 0.04 * math.sqrt(0.01))
        assert est.impact_bps == pytest.approx(est.impact_fraction * 1e4)
        assert est.participation == pytest.approx(0.01)

    def test_impact_scales_with_the_square_root_of_size(self) -> None:
        p = _params()
        a = square_root_impact(order_notional=100.0, adv_notional=1e6,
                               sigma_daily=0.04, params=p)
        b = square_root_impact(order_notional=400.0, adv_notional=1e6,
                               sigma_daily=0.04, params=p)
        assert b.impact_fraction / a.impact_fraction == pytest.approx(2.0)

    def test_the_sign_of_the_order_does_not_change_the_cost(self) -> None:
        p = _params()
        buy = square_root_impact(order_notional=1_000.0, adv_notional=1e6,
                                 sigma_daily=0.04, params=p)
        sell = square_root_impact(order_notional=-1_000.0, adv_notional=1e6,
                                  sigma_daily=0.04, params=p)
        assert buy.impact_fraction == pytest.approx(sell.impact_fraction)

    def test_permanent_and_temporary_sum_to_the_whole(self) -> None:
        est = square_root_impact(order_notional=1_000.0, adv_notional=1e6,
                                 sigma_daily=0.04, params=_params(kappa_d=0.3))
        assert est.permanent_fraction + est.temporary_fraction == pytest.approx(
            est.impact_fraction)
        assert est.permanent_fraction == pytest.approx(0.3 * est.impact_fraction)

    def test_a_zero_size_order_has_zero_impact(self) -> None:
        est = square_root_impact(order_notional=0.0, adv_notional=1e6,
                                 sigma_daily=0.04, params=_params())
        assert est.impact_fraction == 0.0

    def test_the_cost_helper_returns_quote_currency(self) -> None:
        est = square_root_impact(order_notional=1_000.0, adv_notional=1e6,
                                 sigma_daily=0.04, params=_params())
        assert est.cost(1_000.0) == pytest.approx(1_000.0 * est.impact_fraction)


# =========================================================================== #
# 2. NEGATIEVE TESTS — fase-opdracht §17
# =========================================================================== #
class TestMissingParametersAreHardFailures:
    def test_calling_without_params_is_a_config_contract_error(self) -> None:
        """§17: 'ontbrekende eta -> FAIL'."""
        with pytest.raises(ConfigContractError):
            square_root_impact(order_notional=1.0, adv_notional=1.0,
                               sigma_daily=0.01, params=None)  # type: ignore[arg-type]

    def test_a_zero_eta_cannot_be_constructed(self) -> None:
        """Nul is niet 'geen impact'; het is de aanname dat handelen gratis is."""
        with pytest.raises(ConfigContractError):
            _params(eta=0.0)

    def test_a_negative_eta_cannot_be_constructed(self) -> None:
        with pytest.raises(ConfigContractError):
            _params(eta=-0.1)

    def test_a_non_finite_eta_cannot_be_constructed(self) -> None:
        with pytest.raises(ConfigContractError):
            _params(eta=float("nan"))

    def test_kappa_d_outside_the_unit_interval_is_rejected(self) -> None:
        with pytest.raises(ConfigContractError):
            _params(kappa_d=1.5)

    def test_params_without_provenance_are_rejected(self) -> None:
        for field in ("method", "data_hash"):
            with pytest.raises(ConfigContractError):
                _params(**{field: ""})

    def test_params_without_a_sample_are_rejected(self) -> None:
        with pytest.raises(ConfigContractError):
            _params(sample_size=0)

    def test_params_without_instruments_are_rejected(self) -> None:
        with pytest.raises(ConfigContractError):
            _params(instruments=())

    def test_a_confidence_band_that_excludes_the_estimate_is_rejected(self) -> None:
        with pytest.raises(ConfigContractError):
            _params(eta=0.5, eta_ci_low=0.6, eta_ci_high=0.7)

    def test_a_status_that_is_free_text_is_rejected(self) -> None:
        with pytest.raises(ConfigContractError):
            _params(status="calibrated")


class TestMissingMarketDataAreHardFailures:
    def test_a_zero_adv_is_not_divided_by_an_epsilon(self) -> None:
        """§17: 'ontbrekende orderbook depth -> FAIL'.

        `market_impact.py` had `min_volume: float = 1e-9` en deelde daardoor
        stilzwijgend door bijna-nul: de impact van handelen in een instrument
        dat niet handelde werd een eindig, plausibel getal.
        """
        with pytest.raises(DataContractError):
            square_root_impact(order_notional=1.0, adv_notional=0.0,
                               sigma_daily=0.01, params=_params())

    def test_a_missing_adv_is_a_hard_failure(self) -> None:
        with pytest.raises(DataContractError):
            square_root_impact(order_notional=1.0, adv_notional=float("nan"),
                               sigma_daily=0.01, params=_params())

    def test_a_negative_sigma_is_a_hard_failure(self) -> None:
        with pytest.raises(DataContractError):
            square_root_impact(order_notional=1.0, adv_notional=1e6,
                               sigma_daily=-0.01, params=_params())

    def test_a_non_finite_order_size_is_a_hard_failure(self) -> None:
        with pytest.raises(DataContractError):
            square_root_impact(order_notional=float("inf"), adv_notional=1e6,
                               sigma_daily=0.01, params=_params())


# =========================================================================== #
# 3. De status reist mee
# =========================================================================== #
class TestStatusPropagation:
    def test_the_estimate_carries_the_calibration_status(self) -> None:
        for status in ImpactStatus:
            est = square_root_impact(
                order_notional=1.0, adv_notional=1e6, sigma_daily=0.01,
                params=_params(status=status),
            )
            assert est.status is status
            assert est.as_record()["status"] == status.value

    def test_the_current_repository_state_is_uncalibrated(
        self, params: ImpactParams
    ) -> None:
        """Zolang er geen orderboekdata is, MOET dit UNCALIBRATED zijn.

        Deze test faalt zodra iemand de status op CALIBRATED zet zonder dat de
        onderliggende data er is - en dat is precies de fout die hij moet
        vangen.
        """
        assert params.status is ImpactStatus.IMPACT_UNCALIBRATED
        assert not params.is_calibrated


# =========================================================================== #
# 4. De kalibratie zelf
# =========================================================================== #
class TestCalibration:
    def test_orderbook_and_trades_are_what_a_real_calibration_needs(self) -> None:
        assert set(REQUIRED_DATASETS) == {"orderbook_l2", "trades"}

    def test_the_certified_store_is_missing_both(self) -> None:
        available = ["ohlcv", "funding", "open_interest"]
        assert missing_calibration_datasets(available) == REQUIRED_DATASETS

    def test_nothing_is_missing_when_everything_is_present(self) -> None:
        assert missing_calibration_datasets(
            ["ohlcv", "orderbook_l2", "trades"]) == ()

    def test_range_implied_eta_is_range_over_sigma(self) -> None:
        idx = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
        ohlc = pd.DataFrame(
            {"high": [110.0, 120.0, 105.0], "low": [90.0, 100.0, 95.0],
             "close": [100.0, 110.0, 100.0]}, index=idx)
        sigma = pd.Series([0.10, 0.10, 0.10], index=idx)
        got = range_implied_eta(ohlc, sigma)
        assert got.iloc[0] == pytest.approx((110.0 - 90.0) / 100.0 / 0.10)
        assert len(got) == 3

    def test_bars_without_a_volatility_estimate_are_dropped_not_epsilon_filled(
        self,
    ) -> None:
        idx = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
        ohlc = pd.DataFrame(
            {"high": [110.0] * 3, "low": [90.0] * 3, "close": [100.0] * 3},
            index=idx)
        sigma = pd.Series([np.nan, 0.0, 0.10], index=idx)
        got = range_implied_eta(ohlc, sigma)
        assert len(got) == 1
        assert np.isfinite(got).all()

    def test_misaligned_inputs_are_a_hard_failure(self) -> None:
        idx = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
        ohlc = pd.DataFrame(
            {"high": [1.0] * 3, "low": [1.0] * 3, "close": [1.0] * 3}, index=idx)
        with pytest.raises(DataContractError):
            range_implied_eta(ohlc, pd.Series([0.1, 0.1], index=idx[:2]))

    def test_missing_ohlc_columns_are_a_hard_failure(self) -> None:
        idx = pd.date_range("2024-01-01", periods=2, freq="D", tz="UTC")
        with pytest.raises(DataContractError):
            range_implied_eta(pd.DataFrame({"close": [1.0, 1.0]}, index=idx),
                              pd.Series([0.1, 0.1], index=idx))

    def test_calibration_without_orderbook_data_returns_an_upper_bound(self) -> None:
        idx = pd.date_range("2024-01-01", periods=200, freq="D", tz="UTC")
        rng = np.random.default_rng(0)
        close = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0, 0.02, 200))),
                          index=idx)
        ohlc = pd.DataFrame({
            "high": close * 1.02, "low": close * 0.98, "close": close}, index=idx)
        sigma = pd.Series(0.02, index=idx)
        params, evidence = calibrate_impact(
            {"BTCUSDT": ohlc}, {"BTCUSDT": sigma},
            available_datasets=["ohlcv"],
            data_hashes={"crypto/ohlcv/BTCUSDT/1d": "abc123"},
        )
        assert params.status is ImpactStatus.IMPACT_UNCALIBRATED
        assert evidence.missing_datasets == REQUIRED_DATASETS
        assert "BOVENGRENS" in evidence.reason
        assert params.sample_size == 200
        # De bovengrens is het gekozen kwantiel van (high-low)/close / sigma.
        expected = pd.Series((ohlc["high"] - ohlc["low"]) / ohlc["close"] / 0.02)
        assert params.eta == pytest.approx(
            float(expected.quantile(ETA_BOUND_QUANTILE)))

    def test_calibration_without_instruments_is_a_hard_failure(self) -> None:
        with pytest.raises(DataContractError):
            calibrate_impact({}, {}, available_datasets=["ohlcv"], data_hashes={})

    def test_a_missing_sigma_for_an_instrument_is_a_hard_failure(self) -> None:
        idx = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC")
        ohlc = pd.DataFrame(
            {"high": [1.1] * 5, "low": [0.9] * 5, "close": [1.0] * 5}, index=idx)
        with pytest.raises(DataContractError):
            calibrate_impact({"BTCUSDT": ohlc}, {}, available_datasets=["ohlcv"],
                             data_hashes={"x": "y"})


# =========================================================================== #
# 5. De configuratie is niet met de hand geschreven
# =========================================================================== #
class TestConfigMatchesTheArtefact:
    def test_the_artefact_exists(self) -> None:
        assert ARTEFACT.is_file(), (
            "artefacts/execution/impact_params.json ontbreekt; draai "
            "`python apps/calibrate_impact.py`."
        )

    def test_every_config_value_comes_from_the_calibration_run(
        self, cfg: ImpactConfig
    ) -> None:
        """Bewijst dat `conf/execution/impact.yaml` niet met de hand is gezet.

        Zonder deze test is een gekalibreerde parameter niet te onderscheiden
        van een getal dat iemand mooi vond.
        """
        recorded = json.loads(ARTEFACT.read_text(encoding="utf-8"))["params"]
        assert cfg.eta == pytest.approx(recorded["eta"], rel=1e-6)
        assert cfg.kappa_d == pytest.approx(recorded["kappa_d"], rel=1e-6)
        assert cfg.status == recorded["status"]
        assert cfg.data_hash == recorded["data_hash"]
        assert cfg.sample_size == recorded["sample_size"]
        assert cfg.period_start == recorded["period_start"]
        assert cfg.period_end == recorded["period_end"]
        assert list(cfg.instruments) == recorded["instruments"]
        assert cfg.eta_ci_low == pytest.approx(recorded["eta_ci_low"], rel=1e-6)
        assert cfg.eta_ci_high == pytest.approx(recorded["eta_ci_high"], rel=1e-6)

    def test_the_config_has_no_defaults_at_all(self) -> None:
        """Elk veld verplicht: een onvolledige impactconfig moet crashen."""
        assert all(
            field.is_required() for field in ImpactConfig.model_fields.values()
        ), "ImpactConfig heeft een default gekregen; §11 verbiedt dat."

    def test_an_incomplete_impact_config_is_a_config_contract_error(self) -> None:
        from tradebot.schemas.config import validate_mapping

        with pytest.raises(ConfigContractError):
            validate_mapping(ImpactConfig, {"kappa_d": 0.5}, source="test")


def test_execution_config_no_longer_carries_a_default_eta() -> None:
    """Het dode `impact_eta: 0.142` is weg uit `ExecutionConfig`.

    Zolang het daar stond, was er een tweede, ongelezen bron voor eta met een
    literatuurwaarde als default - precies de constructie die §11 verbiedt.
    """
    from tradebot.schemas.config import ExecutionConfig

    assert "impact_eta" not in ExecutionConfig.model_fields
