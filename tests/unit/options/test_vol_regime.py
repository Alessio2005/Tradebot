"""Verwachte RV, volatility risk premium en het regime: vaste regels, elke tak apart."""
from __future__ import annotations

import math
from typing import get_args

import numpy as np
import pandas as pd
import pytest

from tests.unit.options.market_helpers import CFG
from tradebot.options.iv_metrics import realized_vol
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
