"""Verwachte RV, volatility risk premium en het regime: vaste regels, elke tak apart."""
from __future__ import annotations

import dataclasses
import math
from typing import get_args

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from tests.unit.options.market_helpers import CFG
from tradebot.options.iv_metrics import ewma_vol, realized_vol
from tradebot.options.vol_regime import (
    RegimeSettings,
    RVForecast,
    VolRegime,
    VolState,
    classify,
    effective_regimes,
    forecast_rv,
    regime_settings,
)
from tradebot.schemas.options import (
    OptionsConfig,
    RegimeAllowedConfig,
    StrategiesConfig,
    StrategyName,
    VolRegimeConfig,
)

S = RegimeSettings()


def bars(n: int, vol: float = 0.20, seed: int = 7, shock_vol: float | None = None,
         shock_days: int = 5) -> pd.DataFrame:
    """Dagbars met een vaste dagvol; de laatste `shock_days` optioneel met `shock_vol`."""
    rng = np.random.default_rng(seed)
    sig = np.full(n, vol / math.sqrt(252.0))
    if shock_vol is not None:
        sig[-shock_days:] = shock_vol / math.sqrt(252.0)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, sig)))
    opn = np.concatenate([[100.0], close[:-1]]) * np.exp(rng.normal(0.0, sig / 4))
    hi = np.maximum(opn, close) * np.exp(np.abs(rng.normal(0.0, sig / 2)))
    lo = np.minimum(opn, close) * np.exp(-np.abs(rng.normal(0.0, sig / 2)))
    idx = pd.bdate_range("2026-01-02", periods=n)
    return pd.DataFrame({"open": opn, "high": hi, "low": lo, "close": close}, index=idx)


def fc(short: float = 0.20, mid: float = 0.20, long: float | None = 0.20,
       forecast: float = 0.20) -> RVForecast:
    return RVForecast(short, mid, long, forecast)


# --- de RV-verwachting ------------------------------------------------------------------------

def test_the_forecast_mixes_the_three_windows_in_variance():
    b = bars(120)
    f = forecast_rv(b, S)
    assert f is not None and f.rv_long is not None
    rs, rm, rl = (realized_vol(b, w) for w in S.rv_windows)
    assert (f.rv_short, f.rv_mid, f.rv_long) == (rs, rm, rl)
    assert f.forecast == pytest.approx(math.sqrt(0.2 * rs**2 + 0.5 * rm**2 + 0.3 * rl**2))
    assert f.forecast == pytest.approx(0.20, abs=0.05)


def test_without_the_long_window_the_weights_are_rescaled():
    b = bars(40)
    f = forecast_rv(b, S)
    assert f is not None and f.rv_long is None
    assert f.forecast == pytest.approx(math.sqrt((0.2 * f.rv_short**2 + 0.5 * f.rv_mid**2) / 0.7))


def test_without_the_mid_window_there_is_no_forecast():
    assert forecast_rv(bars(15), S) is None


def test_a_vol_shock_shows_as_expansion():
    f = forecast_rv(bars(120, shock_vol=0.80), S)
    assert f is not None and f.expansion > S.stress_rv_expansion


# --- het regime --------------------------------------------------------------------------------

@pytest.mark.parametrize("atm, rv, slope, regime, reason", [
    (None, fc(), 0.01, VolRegime.UNKNOWN, "no_iv"),
    (0.25, None, 0.01, VolRegime.UNKNOWN, "no_rv"),
    (0.40, fc(), -0.03, VolRegime.STRESS, "backwardation"),
    (0.40, fc(short=0.40, mid=0.20), 0.01, VolRegime.STRESS, "rv_expansion"),
    (0.25, fc(), 0.01, VolRegime.RICH, "vrp_rich"),
    (0.25, fc(), None, VolRegime.RICH, "vrp_rich"),
    (0.25, fc(), -0.01, VolRegime.NORMAL, "vrp_normal"),     # licht invers: geen RICH
    (0.215, fc(), 0.01, VolRegime.NORMAL, "vrp_normal"),     # ratio 1,075
    (0.231, fc(forecast=0.20), 0.01, VolRegime.RICH, "vrp_rich"),
    (0.118, fc(forecast=0.10), 0.01, VolRegime.NORMAL, "vrp_normal"),  # ratio ok, < 2 punten
    (0.18, fc(), 0.01, VolRegime.CHEAP, "vrp_cheap"),
])
def test_the_rules_in_order(atm, rv, slope, regime, reason):
    st = classify(atm, rv, slope, S)
    assert (st.regime, st.reason) == (regime, reason)


def test_the_state_reports_the_premium():
    st = classify(0.25, fc(forecast=0.20), 0.01, S)
    assert st.vrp_ratio == pytest.approx(1.25) and st.vrp_points == pytest.approx(0.05)
    ev = st.as_event()
    assert ev["regime"] == "rich" and ev["vrp_points"] == 0.05 and ev["rv_expansion"] == 1.0
    empty = classify(None, None, None, S)
    assert empty.vrp_ratio is None and empty.vrp_points is None
    assert empty.as_event()["rv_forecast"] is None and empty.as_event()["rv_expansion"] is None


def _st(regime: VolRegime) -> VolState:
    return VolState(regime, 0.2, fc(), 0.0, "x")


def test_stress_on_the_market_underlying_holds_for_the_whole_book():
    states = {"SPY": _st(VolRegime.STRESS), "AAPL": _st(VolRegime.RICH)}
    assert effective_regimes(states, "SPY") == {"SPY": VolRegime.STRESS,
                                                "AAPL": VolRegime.STRESS}
    assert effective_regimes(states, None)["AAPL"] is VolRegime.RICH
    calm = {"SPY": _st(VolRegime.NORMAL), "AAPL": _st(VolRegime.RICH)}
    assert effective_regimes(calm, "SPY")["AAPL"] is VolRegime.RICH
    assert effective_regimes({"AAPL": _st(VolRegime.CHEAP)}, "SPY") == {"AAPL": VolRegime.CHEAP}


# --- de config ---------------------------------------------------------------------------------

def test_the_settings_come_from_the_config():
    assert regime_settings(CFG.vol_regime) == RegimeSettings()


def test_the_strategy_names_match_the_strategies():
    assert set(RegimeAllowedConfig.model_fields) == {r.value for r in VolRegime}
    assert set(get_args(StrategyName)) == set(StrategiesConfig.model_fields)


@pytest.mark.parametrize("kw, msg", [
    ({"rv_windows": (20, 5, 60)}, "oplopend"),
    ({"rv_windows": (1, 5, 60)}, "oplopend"),
    ({"cheap_vrp_ratio": 1.2, "rich_vrp_ratio": 1.1}, "cheap_vrp_ratio"),
    ({"stress_term_slope": 0.0, "rich_min_term_slope": -0.01}, "stress_term_slope"),
])
def test_the_config_rejects_inconsistent_rules(kw, msg):
    with pytest.raises(ValueError, match=msg):
        VolRegimeConfig(**kw)


def test_the_market_underlying_must_be_in_the_universe():
    raw = CFG.model_dump()
    raw["vol_regime"]["market_underlying"] = "XYZ"
    with pytest.raises(ValueError, match="market_underlying"):
        OptionsConfig.model_validate(raw)
    raw["vol_regime"]["market_underlying"] = None
    assert OptionsConfig.model_validate(raw).vol_regime.market_underlying is None


# --- σ(h), EWMA en VRP per looptijd -----------------------------------------------------------

def test_at_horizon_runs_from_the_blend_to_the_long_run_rv():
    f = RVForecast(0.40, 0.30, 0.20, 0.40, kappa=4.0)
    assert f.at_horizon(0) == f.forecast and f.at_horizon(-5) == f.forecast
    assert f.at_horizon(1e-6) == pytest.approx(0.40, abs=1e-6)          # h → 0: σ_0
    assert f.at_horizon(365 * 1000) == pytest.approx(0.20, abs=1e-3)    # h → ∞: σ_L
    h = 30 / 365
    w = (1 - math.exp(-4.0 * h)) / (4.0 * h)
    assert f.at_horizon(30) == pytest.approx(math.sqrt(0.04 + (0.16 - 0.04) * w))
    assert f.at_horizon(30) > f.at_horizon(90) > 0.20                   # monotoon naar σ_L


def test_at_horizon_uses_the_mid_window_without_a_long_window():
    f = RVForecast(0.40, 0.30, None, 0.40)
    assert f.at_horizon(365 * 1000) == pytest.approx(0.30, abs=1e-3)


def test_at_horizon_is_flat_for_a_flat_forecast_and_follows_kappa():
    assert fc().at_horizon(45) == pytest.approx(0.20)
    slow = RVForecast(0.40, 0.30, 0.20, 0.40, kappa=0.5)
    fast = RVForecast(0.40, 0.30, 0.20, 0.40, kappa=20.0)
    assert fast.at_horizon(60) < slow.at_horizon(60)


def test_vrp_at_measures_iv_against_sigma_of_the_term():
    st = classify(0.30, RVForecast(0.40, 0.30, 0.20, 0.40), 0.0, S)
    sigma = st.rv.at_horizon(45)
    ratio, points = st.vrp_at(0.30, 45)
    assert ratio == pytest.approx(0.30 / sigma) and points == pytest.approx(0.30 - sigma)
    assert st.vrp_at(0.30, 0) == (st.vrp_ratio, st.vrp_points)
    assert classify(0.30, None, None, S).vrp_at(0.30, 45) is None


def test_ewma_vol_follows_the_recursion_and_needs_two_closes():
    closes = [100.0, 101.0, 99.0, 102.0]
    df = pd.DataFrame({"close": closes})
    r = np.diff(np.log(closes))
    v = r[0] ** 2
    for x in r[1:]:
        v = 0.9 * v + 0.1 * x * x
    assert ewma_vol(df, 0.9) == pytest.approx(math.sqrt(v * 252.0))
    assert ewma_vol(df.iloc[:1], 0.9) is None
    assert ewma_vol(pd.DataFrame({"close": [100.0, 100.0, 100.0]}), 0.9) is None


def test_the_ewma_is_off_by_default_and_blends_with_rescaled_weights():
    b = bars(120)
    base = forecast_rv(b, S)
    assert base is not None and base.kappa == 4.0
    assert forecast_rv(b, dataclasses.replace(S, ewma_weight=0.0)) == base
    s = dataclasses.replace(S, ewma_weight=0.25)
    f = forecast_rv(b, s)
    rs, rm, rl = (realized_vol(b, w) for w in S.rv_windows)
    re_ = ewma_vol(b, s.ewma_lambda)
    har = 0.2 * rs**2 + 0.5 * rm**2 + 0.3 * rl**2
    assert f is not None and f.forecast == pytest.approx(math.sqrt(0.75 * har + 0.25 * re_**2))
    assert f.rv_short == base.rv_short and f.rv_long == base.rv_long
    # Zonder lang venster: de twee andere gewichten (0,2 en 0,5) worden naar 0,75 herschaald.
    short = forecast_rv(bars(30), s)
    rs, rm = realized_vol(bars(30), 5), realized_vol(bars(30), 20)
    re_ = ewma_vol(bars(30), s.ewma_lambda)
    har = (0.2 * rs**2 + 0.5 * rm**2) / 0.7
    assert short is not None and short.forecast == pytest.approx(
        math.sqrt(0.75 * har + 0.25 * re_**2))


def test_an_ewma_of_zero_variance_drops_out_of_the_blend():
    flat = bars(120).assign(close=100.0)    # geen rendement: de EWMA ontbreekt
    s = dataclasses.replace(S, ewma_weight=0.25)
    assert forecast_rv(flat, s) == forecast_rv(flat, S)


def test_the_new_config_keys_reach_the_settings():
    cfg = VolRegimeConfig(ewma_lambda=0.9, ewma_weight=0.3, mean_reversion_kappa=2.0)
    s = regime_settings(cfg)
    assert (s.ewma_lambda, s.ewma_weight, s.mean_reversion_kappa) == (0.9, 0.3, 2.0)
    assert VolRegimeConfig().ewma_weight == 0.0
    for bad in ({"ewma_lambda": 1.0}, {"ewma_weight": 1.5}, {"mean_reversion_kappa": 0.0}):
        with pytest.raises(ValidationError):
            VolRegimeConfig(**bad)
