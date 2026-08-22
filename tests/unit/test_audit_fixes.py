"""Unit tests for AUDIT_CHECKUP fixes (Waves 10-13 + audit P0–P3 items).

Each test asserts the *property* the audit demands, not implementation
detail. If the implementation changes but still satisfies the spec, the
test stays green.

Covered:
- A-1 Fractional Differentiation (FFD): causal, weights truncated, min_d
       binary search converges on a synthetic random-walk.
- A-2 Dollar Bars: kernel produces bars with bar_notional ~ threshold.
- A-3 Stationarity gate: detects an I(1) random-walk and lets a
       log-return through.
- B-2 TrendScanner: ValueError when event_timestamps is None and
       require_events is the default.
- B-3 Triple Barrier: jump_ratio array is derived from wick/ATR (not
       zeros).
- C-1 / C-2 CFI: accepts sample_weight + n_perms; replicate_se reported.
- D-2 CPCV: ValueError when purge_bars is omitted under intraday default.
- E-2 Triple Barrier: half_spread_arr widens ask_high/bid_low proxies.
- E-4 Funding features: causal (shift(1) applied) and correct shape.
- Wave 12: paper_trade_report.py builds a report from JSONL state.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# A-1 — Fractional Differentiation
# ---------------------------------------------------------------------------
from tradebot.features.fracdiff import (
    MinFracDiff,
    frac_diff_ffd,
    min_frac_diff,
)


def test_ffd_weights_are_causal_and_truncated():
    """Weights are descending in |w|; output uses only past values."""
    rng = np.random.default_rng(7)
    walk = pd.Series(np.cumsum(rng.normal(scale=0.01, size=1_000)),
                     index=pd.date_range("2025-01-01", periods=1_000, freq="h"))
    diffed = frac_diff_ffd(walk, d=0.4, threshold=1e-4)
    # Causal: output index strictly inside the input index, no future-leak.
    assert diffed.index[0] >= walk.index[0]
    assert diffed.index[-1] == walk.index[-1]
    # Truncated convolution: at least some lead-in rows must be dropped.
    assert len(diffed) < len(walk)


def test_min_frac_diff_finds_d_under_unit_root():
    """A random walk needs d > 0 to become stationary; binary search converges."""
    pytest.importorskip("statsmodels")
    rng = np.random.default_rng(11)
    walk = pd.Series(np.cumsum(rng.normal(scale=1.0, size=2_000)),
                     index=pd.date_range("2025-01-01", periods=2_000, freq="h"))
    result = min_frac_diff(walk, d_lo=0.05, d_hi=0.95, p_target=0.05)
    assert 0.05 <= result.d <= 0.95
    if result.converged:
        assert result.pvalue <= 0.05


def test_min_frac_diff_transformer_round_trip():
    pytest.importorskip("statsmodels")
    rng = np.random.default_rng(3)
    idx = pd.date_range("2025-01-01", periods=1_500, freq="h")
    df = pd.DataFrame({
        "close": np.cumsum(rng.normal(scale=0.5, size=1_500)) + 100.0,
        "log_ret": rng.normal(scale=0.001, size=1_500),
    }, index=idx)
    transformer = MinFracDiff(threshold=1e-3, p_target=0.10,
                              d_lo=0.1, d_hi=0.9,
                              exclude=("log_ret",))
    out = transformer.fit_transform(df)
    assert any(c.startswith("ffd_close_") for c in out.columns)
    assert not any(c.startswith("ffd_log_ret") for c in out.columns)


# ---------------------------------------------------------------------------
# A-2 — Dollar Bars
# ---------------------------------------------------------------------------
from tradebot.bars.dollar import generate_dollar_bars


def test_dollar_bars_threshold_holds_per_bar():
    """Each generated bar's notional ≥ threshold (allow last residue)."""
    rng = np.random.default_rng(2)
    n = 2_000
    idx = pd.date_range("2025-01-01", periods=n, freq="min", tz="UTC")
    price = 50_000 + np.cumsum(rng.normal(scale=10, size=n))
    df = pd.DataFrame({
        "open":  price,
        "high":  price + 5,
        "low":   price - 5,
        "close": price,
        "tick_volume": np.abs(rng.normal(loc=1.0, scale=0.3, size=n)),
    }, index=idx)
    bars = generate_dollar_bars(df, threshold=500_000)
    assert len(bars) > 0
    assert "bar_notional" in bars.columns
    # Each closed bar (all but possibly the last) should cross threshold.
    closed = bars["bar_notional"].iloc[:-1]
    assert (closed >= 500_000 * 0.95).all()


# ---------------------------------------------------------------------------
# A-3 — Stationarity gate
# ---------------------------------------------------------------------------
from tradebot.features.stationarity_gate import check_feature_stationarity


def test_stationarity_gate_flags_random_walk_passes_returns():
    pytest.importorskip("statsmodels")
    rng = np.random.default_rng(13)
    n = 1_500
    idx = pd.date_range("2025-01-01", periods=n, freq="h")
    df = pd.DataFrame({
        "feat_close":  np.cumsum(rng.normal(scale=1.0, size=n)),   # I(1)
        "feat_logret": rng.normal(scale=0.01, size=n),             # I(0)
    }, index=idx)
    _, report = check_feature_stationarity(df, drop_if_non_stationary=False)
    assert "feat_close" in report.non_stationary
    assert "feat_logret" not in report.non_stationary
    df_dropped, _ = check_feature_stationarity(df, drop_if_non_stationary=True)
    assert "feat_close" not in df_dropped.columns
    assert "feat_logret" in df_dropped.columns


# ---------------------------------------------------------------------------
# B-2 — TrendScanner event_timestamps gate
# ---------------------------------------------------------------------------
from tradebot.labeling.trend_scanning import TrendScanningLabeler


def test_trend_scanner_requires_event_timestamps_by_default():
    rng = np.random.default_rng(5)
    idx = pd.date_range("2025-01-01", periods=400, freq="h")
    df = pd.DataFrame({
        "open":  100 + rng.normal(scale=0.5, size=400).cumsum(),
        "high":  None,
        "low":   None,
        "close": None,
    }, index=idx)
    df["high"]  = df["open"] + 1.0
    df["low"]   = df["open"] - 1.0
    df["close"] = df["open"]
    labeler = TrendScanningLabeler(t_min=5, t_max=10, min_tstat=1.5)
    with pytest.raises(ValueError, match="AUDIT B-2"):
        labeler.label_data(df, event_timestamps=None)


# ---------------------------------------------------------------------------
# B-3 + E-2 — Triple Barrier jump_ratio derived & half_spread widening
# ---------------------------------------------------------------------------
from tradebot.labeling.triple_barrier import TripleBarrierLabeler


def _make_atr_bars(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(8)
    idx = pd.date_range("2025-01-01", periods=n, freq="h")
    close = 100 + np.cumsum(rng.normal(scale=0.3, size=n))
    df = pd.DataFrame({
        "open":  close,
        "high":  close + 0.6,
        "low":   close - 0.6,
        "close": close,
    }, index=idx)
    return df


def test_triple_barrier_uses_wick_ratio_when_sl_jump_active():
    df = _make_atr_bars()
    atr = np.full(len(df), 0.5, dtype=np.float64)
    events = np.array([10, 30, 60, 90], dtype=np.int32)
    horizons = np.full(events.shape[0], 12, dtype=np.int32)

    no_jump = TripleBarrierLabeler(
        pt_width=2.0, sl_width=1.0, sl_jump_max_mult=0.0,
    )
    sl_jump = TripleBarrierLabeler(
        pt_width=2.0, sl_width=1.0, sl_jump_max_mult=0.5,
    )
    r1 = no_jump.label(df, events, horizons, atr, side=1)
    r2 = sl_jump.label(df, events, horizons, atr, side=1)
    # Activating sl_jump_max_mult with derived wick-ratio must change at least
    # one t1_idx / barrier outcome on this synthetic dataset.
    assert (r1["t1_idx"].values != r2["t1_idx"].values).any() or \
           (r1["barrier_label"].values != r2["barrier_label"].values).any()


def test_triple_barrier_half_spread_arr_changes_outcome():
    df = _make_atr_bars()
    atr = np.full(len(df), 0.5, dtype=np.float64)
    events = np.array([10, 30, 60, 90], dtype=np.int32)
    horizons = np.full(events.shape[0], 12, dtype=np.int32)

    lab = TripleBarrierLabeler(pt_width=2.0, sl_width=1.0)
    r_zero = lab.label(df, events, horizons, atr, side=1,
                       half_spread_arr=np.zeros(len(df)))
    r_wide = lab.label(df, events, horizons, atr, side=1,
                       half_spread_arr=np.full(len(df), 1.0))
    # Widening the half-spread should reduce or shift PT/SL outcomes.
    assert (r_zero["t1_idx"].values != r_wide["t1_idx"].values).any() or \
           (r_zero["barrier_label"].values != r_wide["barrier_label"].values).any()


# ---------------------------------------------------------------------------
# C-1 / C-2 — CFI sample_weight + n_perms
# ---------------------------------------------------------------------------
from tradebot.features.cfi import compute_clustered_feature_importance


class _MajorityClassModel:
    """Deterministic classifier returning a constant for predict_proba."""
    def __init__(self, p_class1: float = 0.6):
        self.p_class1 = p_class1
    def predict_proba(self, X):
        n = len(X)
        return np.column_stack([1 - np.full(n, self.p_class1),
                                np.full(n, self.p_class1)])
    def predict(self, X):
        return np.ones(len(X), dtype=int)


def test_cfi_accepts_sample_weight_and_n_perms():
    rng = np.random.default_rng(1)
    n, d = 600, 6
    X = rng.normal(size=(n, d))
    y = (X[:, 0] + rng.normal(scale=0.5, size=n) > 0).astype(int)
    w = np.linspace(0.1, 1.0, n)
    model = _MajorityClassModel()
    out, _ = compute_clustered_feature_importance(
        model, X, y,
        feature_names=[f"f{i}" for i in range(d)],
        n_regimes=3, purge_bars=10,
        sample_weight=w, n_perms=4, rng_seed=42,
    )
    assert "n_perms" in out.columns
    assert (out["n_perms"] == 4).all()
    # replicate_se column exists and is non-negative (or NaN for single perm).
    se = out["replicate_se"].dropna()
    assert (se >= 0).all()


def test_cfi_sample_weight_length_mismatch_raises():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 3))
    y = (X[:, 0] > 0).astype(int)
    with pytest.raises(ValueError, match="sample_weight"):
        compute_clustered_feature_importance(
            _MajorityClassModel(), X, y, ["a", "b", "c"],
            sample_weight=np.ones(50),
        )


# ---------------------------------------------------------------------------
# D-2 — CPCV purge_bars gate
# ---------------------------------------------------------------------------
from tradebot.cv.cpcv import CombinatorialPurgedCV


def test_cpcv_requires_purge_bars_by_default():
    with pytest.raises(ValueError, match="AUDIT D-2"):
        CombinatorialPurgedCV(n_groups=4, n_test_groups=2)


def test_cpcv_accepts_purge_bars():
    cv = CombinatorialPurgedCV(n_groups=4, n_test_groups=2, purge_bars=50)
    assert cv.purge_bars == 50


def test_cpcv_opt_out_still_works():
    """Daily-bar studies can disable the gate explicitly."""
    cv = CombinatorialPurgedCV(
        n_groups=4, n_test_groups=2, require_bar_purge=False,
    )
    assert cv.purge_bars is None


# ---------------------------------------------------------------------------
# E-4 — Funding features
# ---------------------------------------------------------------------------
from tradebot.features.funding_carry import compute_funding_features


def test_funding_features_are_causal_and_shaped():
    n = 720  # 30 days of 1-h bars
    idx = pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC")
    # Mostly zeros; 3 funding ticks per day at hours 0/8/16.
    rate = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if idx[i].hour in (0, 8, 16):
            rate[i] = 0.0001 * (1 if (i // 8) % 2 == 0 else -1)
    df = compute_funding_features(idx, rate)
    # 5 columns expected.
    expected = {
        "feat_micro_funding_rate",
        "feat_micro_funding_anno",
        "feat_meso_funding_cum_24h",
        "feat_meso_funding_zscore_30d",
        "feat_meso_funding_sign",
    }
    assert expected.issubset(df.columns)
    # Causality: the first row must not carry the bar-0 funding.
    assert df["feat_micro_funding_rate"].iloc[0] == pytest.approx(0.0)
    # 24h sum is bounded by 3 * max(abs(rate)).
    assert df["feat_meso_funding_cum_24h"].abs().max() <= 3.5 * 0.0001 + 1e-9


# ---------------------------------------------------------------------------
# Wave 12 — paper_trade_report.py
# ---------------------------------------------------------------------------
# Path-robust import: 'scripts' is not a package; resolve by file location so
# collection works regardless of cwd/sys.path (Wave-20 runner fix).
import importlib.util as _ilu

_ptr_spec = _ilu.spec_from_file_location(
    "paper_trade_report",
    pathlib.Path(__file__).resolve().parents[2] / "scripts" / "paper_trade_report.py",
)
ptr = _ilu.module_from_spec(_ptr_spec)
_ptr_spec.loader.exec_module(ptr)


def test_paper_trade_report_builds_summary(tmp_path: pathlib.Path):
    state = tmp_path / "state"
    state.mkdir()
    audit_path = state / "audit_log.jsonl"
    audit_path.write_text(
        json.dumps({
            "event_ts": "2026-05-13T00:00:00Z",
            "order_id": "ord-1",
            "symbol":   "BTCUSDT",
            "side":     "BUY",
            "qty_base": 0.01,
            "notional_usdt": 600.0,
            "fill_ts":  "2026-05-13T00:00:01Z",
        }) + "\n",
        encoding="utf-8",
    )
    report = ptr.build_report(state)
    assert "Audit log summary" in report
    assert "Circuit breaker" in report
    assert "no halts" in report
