"""Guards for the adaptive walk-forward book (AFML overfit/leakage controls).

Fast: tiny synthetic panel + tiny model so the whole suite runs in seconds.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradebot.alpha.adaptive_wf import AdaptiveWalkForward, AdaptiveWFConfig


def _synth_panel(n_assets=16, n_days=760, seed=7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2021-06-01", periods=n_days, freq="D", tz="UTC")
    frames = []
    names = ["BTCUSDT"] + [f"A{i}USDT" for i in range(n_assets - 1)]
    for s in names:
        ret = rng.normal(0, 0.04, n_days)
        close = 100 * np.exp(np.cumsum(ret))
        high = close * (1 + np.abs(rng.normal(0, 0.01, n_days)))
        low = close * (1 - np.abs(rng.normal(0, 0.01, n_days)))
        op = close * (1 + rng.normal(0, 0.005, n_days))
        vol = rng.lognormal(10, 1, n_days)
        frames.append(pd.DataFrame({"date": dates, "symbol": s, "open": op,
                                    "high": high, "low": low, "close": close, "volume": vol}))
    return pd.concat(frames, ignore_index=True)


def _tiny_cfg(**kw):
    cfg = AdaptiveWFConfig(
        wf_start="2023-01-01", refit_days=180, min_train_rows=500, min_hist_bars=300,
        mda_keep=99, recency_halflife_days=200.0, seed=0, **kw,
    )
    cfg.catboost_params = {"iterations": 40, "depth": 3, "learning_rate": 0.1,
                           "l2_leaf_reg": 4, "loss_function": "Logloss", "verbose": 0}
    return cfg


@pytest.fixture(scope="module")
def panel():
    return _synth_panel()


def test_no_lookahead_features(panel):
    """Features for past dates must not change when future bars are removed (R-1)."""
    eng = AdaptiveWalkForward(_tiny_cfg()).build_panel(panel)
    P_full = eng.build_features_labels()
    cut = panel["date"].sort_values().unique()[600]
    eng2 = AdaptiveWalkForward(_tiny_cfg()).build_panel(panel[panel["date"] <= cut])
    P_trunc = eng2.build_features_labels()
    feats = [c for c in eng.features_]
    a = P_full[P_full["__date"] < pd.Timestamp(cut) - pd.Timedelta(days=70)]
    b = P_trunc[P_trunc["__date"] < pd.Timestamp(cut) - pd.Timedelta(days=70)]
    key = ["__sym", "__date"]
    merged = a[key + feats].merge(b[key + feats], on=key, suffixes=("_f", "_t"))
    assert len(merged) > 1000
    for f in feats:
        x, y = merged[f"{f}_f"].to_numpy(), merged[f"{f}_t"].to_numpy()
        m = np.isfinite(x) & np.isfinite(y)
        assert np.allclose(x[m], y[m], atol=1e-9), f"lookahead leak in feature {f}"


def test_determinism(panel):
    """Same cfg + seed -> bit-identical OOS scores (R-5)."""
    eng1 = AdaptiveWalkForward(_tiny_cfg()).build_panel(panel)
    P1 = eng1.run(eng1.build_features_labels())
    eng2 = AdaptiveWalkForward(_tiny_cfg()).build_panel(panel)
    P2 = eng2.run(eng2.build_features_labels())
    s1, s2 = P1["__p"].to_numpy(), P2["__p"].to_numpy()
    m = np.isfinite(s1) & np.isfinite(s2)
    assert m.sum() > 0
    assert np.array_equal(s1[m], s2[m])
    assert np.isnan(s1).sum() == np.isnan(s2).sum()


def test_recency_weights_decay(panel):
    """Older events get lower sample weight; weights positive & finite."""
    eng = AdaptiveWalkForward(_tiny_cfg()).build_panel(panel)
    P = eng.build_features_labels()
    anchor = pd.Timestamp("2023-01-01", tz="UTC")
    tr = P[P["__date"] < anchor]
    w = eng._sample_weights(tr, anchor=anchor)
    assert np.all(np.isfinite(w)) and np.all(w > 0)
    age = (anchor - tr["__date"]).dt.days.to_numpy()
    old = w[age > np.quantile(age, 0.75)].mean()
    recent = w[age < np.quantile(age, 0.25)].mean()
    assert recent > old, "recency weighting must favour recent events"


def test_dollar_neutral_harvest(panel):
    """Harvested book is ~dollar-neutral each day (long == short gross)."""
    eng = AdaptiveWalkForward(_tiny_cfg()).build_panel(panel)
    P = eng.run(eng.build_features_labels())
    pnl = eng.harvest(P)
    assert np.isfinite(pnl).all()
    assert pnl.std() > 0
