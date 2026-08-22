"""tests/unit/test_open_interest.py — Open-interest data + feature block.

Covers: Bybit OI fetch parsing (mocked), causal per-bar as-of loader,
no-lookahead guarantees, and feature-block correctness/validity.
All tests use synthetic in-memory data — no network, no parquet I/O except
a tmp roundtrip for the loader.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tradebot.data.open_interest import (
    OpenInterestFetcher,
    load_per_bar_open_interest,
)
from tradebot.features.open_interest import compute_oi_features


# ============================================================================
# Fetcher parsing (mocked network)
# ============================================================================
class _FakeResp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def test_fetch_open_interest_parses_bybit_payload(monkeypatch, tmp_path):
    """One page of Bybit-shaped OI rows → tidy DataFrame, then empty cursor."""
    page = {
        "retCode": 0,
        "retMsg": "OK",
        "result": {
            "list": [
                {"openInterest": "12345.6", "timestamp": "1700003600000"},
                {"openInterest": "12300.0", "timestamp": "1700000000000"},
            ],
            "nextPageCursor": "",  # terminate immediately
        },
    }
    monkeypatch.setattr(
        "tradebot.data.open_interest.requests.get",
        lambda *a, **k: _FakeResp(page),
    )
    f = OpenInterestFetcher(data_dir=str(tmp_path), symbol="BTCUSDT", interval="1h")
    df = f.fetch_open_interest()

    assert list(df.columns) == ["openInterest"]
    assert len(df) == 2
    assert df.index.tz is not None                      # UTC-aware
    assert df.index.is_monotonic_increasing             # sorted ascending
    assert df["openInterest"].iloc[-1] == pytest.approx(12345.6)


def test_fetch_open_interest_handles_error_retcode(monkeypatch, tmp_path):
    bad = {"retCode": 10001, "retMsg": "bad", "result": {"list": []}}
    monkeypatch.setattr(
        "tradebot.data.open_interest.requests.get",
        lambda *a, **k: _FakeResp(bad),
    )
    f = OpenInterestFetcher(data_dir=str(tmp_path), symbol="BTCUSDT")
    assert f.fetch_open_interest().empty


def test_fetcher_rejects_bad_interval(tmp_path):
    with pytest.raises(ValueError):
        OpenInterestFetcher(data_dir=str(tmp_path), interval="7m")


# ============================================================================
# Per-bar loader — causal as-of alignment
# ============================================================================
def _write_oi(tmp_path: Path, symbol: str, ts, values) -> Path:
    df = pd.DataFrame({"timestamp": ts, "openInterest": values})
    p = tmp_path / f"oi_crypto_{symbol}.parquet"
    df.to_parquet(p, index=False)
    return tmp_path


def test_loader_missing_file_returns_zeros(tmp_path):
    idx = pd.date_range("2022-01-01", periods=10, freq="1h", tz="UTC")
    out = load_per_bar_open_interest("NOPE", idx, tmp_path)
    assert out.shape == (10,)
    assert np.all(out == 0.0)


def test_loader_asof_is_causal(tmp_path):
    """Each bar must carry the most recent OI obs with obs_ts <= bar_ts."""
    obs_ts = pd.to_datetime(
        ["2022-01-01 00:00", "2022-01-01 01:00", "2022-01-01 02:00"], utc=True
    )
    obs_val = [100.0, 200.0, 300.0]
    _write_oi(tmp_path, "BTCUSDT", obs_ts, obs_val)

    # Bars at :30 fall BETWEEN observations → must take the PRIOR obs only.
    bars = pd.to_datetime(
        ["2022-01-01 00:30", "2022-01-01 01:30", "2022-01-01 02:30"], utc=True
    )
    out = load_per_bar_open_interest("BTCUSDT", pd.DatetimeIndex(bars), tmp_path)
    assert out.tolist() == [100.0, 200.0, 300.0]  # never a future value


def test_loader_pre_history_is_zero(tmp_path):
    obs_ts = pd.to_datetime(["2022-01-01 05:00"], utc=True)
    _write_oi(tmp_path, "BTCUSDT", obs_ts, [500.0])
    bars = pd.date_range("2022-01-01 00:00", periods=8, freq="1h", tz="UTC")
    out = load_per_bar_open_interest("BTCUSDT", bars, tmp_path)
    # Bars before the first obs (00:00..04:00) have no causal OI → 0.0.
    assert out[0] == 0.0 and out[4] == 0.0
    assert out[5] == 500.0  # 05:00 onward carries the level


# ============================================================================
# Feature block — correctness & no-lookahead
# ============================================================================
@pytest.fixture()
def idx() -> pd.DatetimeIndex:
    return pd.date_range("2022-01-01", periods=24 * 40, freq="1h", tz="UTC")


def test_features_shape_and_columns(idx):
    oi = np.linspace(1000, 2000, len(idx))
    out = compute_oi_features(idx, oi, close=None)
    assert list(out.columns) == [
        "feat_meso_oi_zscore_30d",
        "feat_meso_oi_chg_24h",
        "feat_meso_oi_price_divergence",
    ]
    assert len(out) == len(idx)
    assert not out.isna().any().any()


def test_features_length_mismatch_raises(idx):
    with pytest.raises(ValueError):
        compute_oi_features(idx, np.ones(len(idx) - 1))


def test_constant_oi_gives_zero_features(idx):
    out = compute_oi_features(idx, np.full(len(idx), 1500.0), close=None)
    assert np.allclose(out["feat_meso_oi_chg_24h"].to_numpy(), 0.0)
    assert np.allclose(out["feat_meso_oi_zscore_30d"].to_numpy(), 0.0)


def test_monotone_oi_positive_change(idx):
    oi = np.exp(np.linspace(0, 1, len(idx)))  # strictly increasing
    out = compute_oi_features(idx, oi, close=None)
    post = out["feat_meso_oi_chg_24h"].to_numpy()[24 * 2:]
    assert np.all(post > 0.0)  # rising OI → positive 24h change


def test_oi_price_divergence_sign(idx):
    # OI rises, price falls over the window → divergence should be -1.
    oi = np.exp(np.linspace(0, 1, len(idx)))
    close = np.linspace(2000, 1000, len(idx))            # falling price
    out = compute_oi_features(idx, oi, close=close)
    tail = out["feat_meso_oi_price_divergence"].to_numpy()[24 * 2:]
    assert np.all(tail == -1.0)


def test_features_are_causal_no_lookahead(idx):
    """Mutating ONLY the last bar's OI must not change any earlier feature row."""
    rng = np.random.default_rng(7)
    oi = 1000 + np.cumsum(rng.normal(0, 5, len(idx)))
    base = compute_oi_features(idx, oi.copy(), close=None)

    oi_mut = oi.copy()
    oi_mut[-1] += 10_000.0  # huge shock on the final bar only
    mut = compute_oi_features(idx, oi_mut, close=None)

    # Every row except the last must be bit-identical (no future leak).
    pd.testing.assert_frame_equal(base.iloc[:-1], mut.iloc[:-1])


def test_feature_recovers_injected_signal(idx):
    """Validity gate: when a positioning build-up (the trailing 24h OI change)
    drives the forward return, the (shifted) ``feat_meso_oi_chg_24h`` recovers
    a POSITIVE rank-correlation; an OI series unrelated to that target does not.
    Guards against a pipeline that destroys real signal or fabricates a
    spurious one."""
    from scipy.stats import spearmanr

    n = len(idx)
    rng = np.random.default_rng(123)
    drive = rng.normal(0, 1, n)                    # latent positioning impulse
    oi = 1000.0 * np.exp(np.cumsum(drive) * 0.01)  # OI integrates the impulse
    out = compute_oi_features(idx, oi, close=None)
    feat = out["feat_meso_oi_chg_24h"].to_numpy()

    # Target driven by the SAME trailing-24h window the feature measures:
    # positioning build-up over (t-24, t-1] predicts the next return.
    window_sum = pd.Series(drive, index=idx).rolling(24).sum().shift(1)
    target = (window_sum + rng.normal(0, 4, n)).to_numpy()

    m = np.isfinite(feat) & np.isfinite(target) & (feat != 0.0)
    ic_signal = spearmanr(feat[m], target[m]).correlation

    rand_oi = 1000.0 * np.exp(np.cumsum(rng.normal(0, 1, n)) * 0.01)
    feat_r = compute_oi_features(idx, rand_oi, close=None)["feat_meso_oi_chg_24h"].to_numpy()
    ic_rand = spearmanr(feat_r[m], target[m]).correlation

    assert ic_signal > 0.30                 # real signal is clearly recovered
    assert abs(ic_rand) < 0.5 * ic_signal   # unrelated OI shows much weaker IC
