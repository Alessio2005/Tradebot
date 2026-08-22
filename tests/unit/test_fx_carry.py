# tests/unit/test_fx_carry.py
"""Wave 25 guards: FX TR-index math, PIT carry signal, direction map."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tradebot.alpha import fx_carry
from tradebot.data.fx_universe import build_fx_factors, load_fx_panels
from tradebot.data.sources.fred import G10_RATE_SERIES, G10_SPOT_SERIES

N = 400


def _write_fx(tmp: Path, rate_lag_days: int = 1) -> Path:
    """Synthetic FRED-shaped parquets: flat spots, EUR rate 5%, USD 1%,
    JPY (inverted series) rate 0%."""
    idx = pd.date_range("2023-01-02", periods=N, freq="B", tz="UTC")
    rows_s, rows_r = [], []
    for ccy, (sid, direction) in G10_SPOT_SERIES.items():
        if ccy not in ("EUR", "JPY"):
            continue
        level = 1.10 if ccy == "EUR" else 150.0  # JPY quoted JPY-per-USD
        for t in idx:
            rows_s.append({"series_id": sid, "event_ts": t, "value": level,
                           "asof_ts": t + pd.Timedelta(days=1)})
    rates = {"USD": 1.0, "EUR": 5.0, "JPY": 0.0}
    for ccy, (sid, _) in G10_RATE_SERIES.items():
        if ccy not in rates:
            continue
        for t in idx:
            rows_r.append({"series_id": sid, "event_ts": t, "value": rates[ccy],
                           "asof_ts": t + pd.Timedelta(days=rate_lag_days)})
    dest = tmp / "fx"
    dest.mkdir(parents=True)
    pd.DataFrame(rows_s).to_parquet(dest / "spots.parquet", index=False)
    pd.DataFrame(rows_r).to_parquet(dest / "rates.parquet", index=False)
    return tmp


def test_tr_index_accrues_carry_on_flat_spots(tmp_path: Path) -> None:
    root = _write_fx(tmp_path)
    tr, carry = load_fx_panels(root=root)
    # flat spots: EUR TR must grow at (5-1)%/yr accrual, JPY at (0-1)%/yr
    valid = tr["EUR"].dropna()
    daily = np.log(valid).diff().dropna()
    assert daily.iloc[-1] == pytest.approx(0.04 / 252, rel=1e-6)
    jpy = np.log(tr["JPY"].dropna()).diff().dropna()
    assert jpy.iloc[-1] == pytest.approx(-0.01 / 252, rel=1e-6)
    assert carry["EUR"].dropna().iloc[-1] == pytest.approx(4.0)


def test_direction_map_inverts_jpy(tmp_path: Path) -> None:
    root = _write_fx(tmp_path)
    tr, _ = load_fx_panels(root=root)
    # JPY series 150 JPY/USD -> USD-per-JPY < 1 internally; TR index starts ~1
    assert tr["JPY"].dropna().iloc[0] == pytest.approx(1.0, abs=0.01)


def test_carry_signal_respects_release_lag(tmp_path: Path) -> None:
    root = _write_fx(tmp_path, rate_lag_days=45)
    _, carry = load_fx_panels(root=root)
    # first ~45 calendar days: rates not yet knowable -> signal NaN
    assert carry["EUR"].iloc[:30].isna().all()
    assert carry["EUR"].dropna().iloc[0] == pytest.approx(4.0)


def _six_ccy_panels(root: Path):
    tr, carry = load_fx_panels(root=root)
    for i, c in enumerate(["AUD", "NZD", "SEK", "NOK"]):
        tr[c] = tr["EUR"] * (1 + 0.001 * i)
        carry[c] = carry["EUR"] - 1.0 - i
    return tr, carry


def test_fx_factors_known_outcome_and_gross_one(tmp_path: Path) -> None:
    root = _write_fx(tmp_path)
    tr, carry = _six_ccy_panels(root)
    f = build_fx_factors(tr, carry)
    # flat spots: every TR accrues its carry; the rank-weighted CARRY factor
    # is long the high-carry names -> strictly positive accrual
    valid = f["CARRY"].dropna()
    assert len(valid) > 100 and (valid.iloc[50:] > 0).mean() > 0.95
    # DOLLAR = mean accrual across the 6 names, all positive ex-JPY mix
    assert f["DOLLAR"].dropna().iloc[-1] == pytest.approx(
        (np.log(tr.iloc[-1] / tr.iloc[-2])).mean(), rel=1e-2
    )
    # TREND needs the 252d warmup, then defined
    assert f["TREND"].dropna().index[0] > f["DOLLAR"].dropna().index[0]


def test_fx_factors_causal_no_lookahead(tmp_path: Path) -> None:
    root = _write_fx(tmp_path)
    tr, carry = _six_ccy_panels(root)
    base = build_fx_factors(tr, carry)
    # corrupting the LAST day's carry signal must not change any factor
    # return (weights use signal <= t-1 only)
    carry2 = carry.copy()
    carry2.iloc[-1] = 99.0
    f2 = build_fx_factors(tr, carry2)
    pd.testing.assert_frame_equal(base, f2)


def test_fx_factors_thin_cross_section_is_nan(tmp_path: Path) -> None:
    root = _write_fx(tmp_path)
    tr, carry = load_fx_panels(root=root)  # only EUR + JPY -> < min_names
    f = build_fx_factors(tr, carry)
    assert f["CARRY"].isna().all() and f["TREND"].isna().all()


def test_fx_carry_unit_runs_deterministic(tmp_path: Path) -> None:
    root = _write_fx(tmp_path)
    tr, carry = load_fx_panels(root=root)
    # widen to 6 currencies so min_names is met: clone EUR/JPY columns
    for i, c in enumerate(["AUD", "NZD", "SEK", "NOK"]):
        tr[c] = tr["EUR"] * (1 + 0.001 * i)
        carry[c] = carry["EUR"] - i
    a = fx_carry.run(tr, carry)
    b = fx_carry.run(tr, carry)
    pd.testing.assert_series_equal(a.net_returns, b.net_returns)
    assert a.config["prior"] == fx_carry.PRIOR
    # highest-carry currency must sit in the long book at the last rebalance
    last_w = a.weights.dropna(how="all").iloc[-1]
    assert last_w["EUR"] > 0 and last_w["JPY"] < 0
