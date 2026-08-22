"""Methodological guards for the FINAL multi-sleeve book (CHIEF).

  1. no look-ahead: weights at date t invariant to FUTURE data.
  2. finite, non-trivial weights.
  3. low market beta (regime-robust, not pure long beta).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
from tradebot.alpha.multi_sleeve_book import MultiSleeveBook

PANEL = _ROOT / "artefacts" / "broad_perp_daily_close.parquet"
DVOL = _ROOT / "artefacts" / "dvol_btc.parquet"
FUND = _ROOT / "artefacts" / "funding_5asset.parquet"


def _data():
    if not (PANEL.exists() and DVOL.exists() and FUND.exists()):
        pytest.skip("data caches not present")
    p = pd.read_parquet(PANEL).resample("1D").last()
    dv = pd.read_parquet(DVOL)["dvol"]
    f = pd.read_parquet(FUND)
    return p, dv, f


def test_finite_nontrivial_weights():
    p, dv, f = _data()
    w = MultiSleeveBook().target_weights(p, dvol=dv, funding=f)
    assert np.isfinite(w.to_numpy()).all(), "non-finite weights"
    assert w.abs().sum() > 1e-6, "all-zero weights"


def test_no_lookahead():
    p, dv, f = _data()
    nb = MultiSleeveBook()
    cut = len(p) - 30
    full, _ = nb.weight_history(p, dvol=dv, funding=f)
    trunc, _ = nb.weight_history(p.iloc[:cut], dvol=dv.iloc[:dv.index.get_indexer([p.index[cut]], method="ffill")[0]], funding=f.iloc[:cut])
    common = trunc.index.intersection(full.index)[-5:]
    cols = full.columns.intersection(trunc.columns)
    diff = (full.loc[common, cols] - trunc.loc[common, cols]).abs().to_numpy()
    assert np.nanmax(diff) < 1e-9, f"look-ahead detected: {np.nanmax(diff)}"


def test_low_market_beta():
    p, dv, f = _data()
    combined, _ = MultiSleeveBook().weight_history(p, dvol=dv, funding=f)
    R = p / p.shift(1) - 1.0
    mkt = R.mean(axis=1)
    we = combined.shift(1).reindex(R.index).reindex(columns=R.columns).fillna(0.0)
    pnl = (we * R).sum(axis=1).dropna()
    beta = np.polyfit(mkt.reindex(pnl.index).fillna(0.0), pnl, 1)[0]
    assert abs(beta) < 0.40, f"beta too high (not regime-robust): {beta:.3f}"
