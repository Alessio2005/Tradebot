"""Regression tests for the CHIEF audit 2026-05-28 fixes.

Covers:
  L-1  Live Platt calibrator must be monotone, correctly-signed (no inversion),
       and degenerate fits must fall back to identity (raw passthrough).
  M-1  Deflated Sharpe must respond to the true multiple-testing burden N.
  L-3  PositionTracker.mark must reject missing/invalid prices (no phantom
       100% loss that poisons the live max-drawdown ratchet).
"""
from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# L-1 — calibrator sign / monotonicity / degeneracy guard
# ---------------------------------------------------------------------------
from tradebot.train.calibration import PathSpecificPlattCalibrator


def _fit_discriminative(seed: int = 0, n: int = 4000):
    rng = np.random.default_rng(seed)
    raw = rng.uniform(0.1, 0.6, n)
    # P(y=1) increases with raw → a correctly-signed calibrator must be
    # monotone increasing.
    y = (rng.uniform(0.0, 1.0, n) < raw).astype(int)
    paths = rng.integers(0, 3, n)
    cal = PathSpecificPlattCalibrator(input_is_probability=True)
    cal.fit(raw, y, paths, np.ones(n))
    return cal


def test_calibrator_is_monotone_increasing_not_inverted():
    cal = _fit_discriminative()
    xs = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    out = cal.predict_proba(xs, path_id=None)
    # The pre-fix bug returned sigmoid(-z) ⇒ low raw mapped to HIGH prob.
    assert np.all(np.diff(out) > 0), f"calibrator not monotone increasing: {out}"
    assert out[0] < 0.35, f"low raw prob should map low, got {out[0]:.3f}"
    assert out[-1] > out[0] + 0.2, "insufficient dynamic range — possible collapse"


def test_calibrator_does_not_invert_low_to_high():
    """The exact live failure mode: raw≈0.29 must NOT map to ≈0.72."""
    cal = _fit_discriminative()
    out = float(cal.predict_proba(np.array([0.29]), path_id=None)[0])
    assert out < 0.5, f"raw 0.29 mapped to {out:.3f} — inversion regression"


def test_calibrator_degenerate_falls_back_to_identity():
    """Non-discriminative (clustered) raw probs ⇒ output near base rate."""
    rng = np.random.default_rng(1)
    n = 4000
    raw = rng.normal(0.29, 0.005, n).clip(0.01, 0.99)  # tight cluster, no signal
    y = (rng.uniform(0, 1, n) < 0.30).astype(int)       # base rate ~0.30
    paths = rng.integers(0, 3, n)
    cal = PathSpecificPlattCalibrator(input_is_probability=True)
    cal.fit(raw, y, paths, np.ones(n))
    out = float(cal.predict_proba(np.array([0.29]), path_id=None)[0])
    # Must sit near the base rate (≈0.30), never the inverted ~0.72.
    assert abs(out - 0.30) < 0.15, f"degenerate calibrator output {out:.3f} off base rate"
    assert out < 0.5, "degenerate calibrator must not produce false high confidence"


def test_guard_rejects_inverted_params():
    A, B = PathSpecificPlattCalibrator._guard_params(-2.0, 0.1, "test")
    assert (A, B) == (1.0, 0.0), "negative slope (inverted) must fall back to identity"
    A2, B2 = PathSpecificPlattCalibrator._guard_params(3.0, -0.5, "test")
    assert A2 == 3.0 and B2 == -0.5, "valid positive slope must be preserved"


# ---------------------------------------------------------------------------
# M-1 — Deflated Sharpe responds to the true hypothesis count
# ---------------------------------------------------------------------------
# PHASE 5: `backtest/portfolio.py::_deflated_sharpe` is verwijderd met de
# legacy-engine. De eigenschap die deze test bewaakt is echter niet
# engine-specifiek, en de canonieke implementatie blijft bestaan: audit
# sectie 24 merkt `backtest/metrics.py::DSR` aan als RETAIN (Bailey-LdP,
# met de dimensionaliteitsfix). De test is daarheen verlegd in plaats van
# verwijderd - de bevinding M-1 blijft gelden.
from tradebot.backtest.metrics import deflated_sharpe as _deflated_sharpe


def test_deflated_sharpe_decreases_with_more_hypotheses():
    # De parameters wijken af van de oorspronkelijke test omdat de CONVENTIE
    # verschilt: het verwijderde `portfolio.py::_deflated_sharpe` nam een
    # GEANNUALISEERDE Sharpe, terwijl `metrics.py::deflated_sharpe` de
    # PER-BAR Sharpe neemt (Bailey-LdP; de dimensionaliteitsfix die audit
    # sectie 24 als RETAIN aanmerkt). Bij sr=2.85 per bar is elke DSR
    # numeriek 1.0 en meet de test niets.
    #
    # De bewaakte eigenschap is ongewijzigd: meer beproefde hypothesen ->
    # lagere DSR, en het effect is materieel.
    #
    # PHASE 10, STAP 4A: de handtekening is die van MEASUREMENT_CONTRACT.md §6
    # geworden. `sr_variance`, `skew`, `kurtosis` en `bars_per_year` hebben geen
    # default meer, dus zij staan hier expliciet; `1/n_obs` is de
    # gedocumenteerde normale benadering en zegt dat ook.
    sr = 0.10
    n_obs = 2_000
    moments = dict(n_obs=n_obs, sr_variance=1.0 / n_obs, skew=0.0,
                   kurtosis=3.0, bars_per_year=365.0, approximation="normal")
    dsr_small = _deflated_sharpe(sr, n_trials=10, **moments).dsr
    dsr_large = _deflated_sharpe(sr, n_trials=2000, **moments).dsr
    assert dsr_large < dsr_small, (
        "DSR must shrink as the multiple-testing burden grows "
        f"(N=10 => {dsr_small:.3f}, N=2000 => {dsr_large:.3f})"
    )
    # De ~200x ondertelling verhoogde de gerapporteerde DSR materieel.
    assert (dsr_small - dsr_large) > 0.01


# ---------------------------------------------------------------------------
# L-3 — phantom-loss guard in the mark-to-market path
# ---------------------------------------------------------------------------
import pandas as pd

from tradebot.oms.order import Fill, OrderSide
from tradebot.oms.position_tracker import PositionTracker


def _open_long(tracker: PositionTracker, symbol="ETHUSDT", qty=10.0, price=2000.0):
    fill = Fill(
        order_id="t1",
        fill_price=price,
        fill_qty=qty,
        fill_ts=pd.Timestamp("2026-05-28T00:00:00Z"),
        notional_usdt=qty * price,
    )
    tracker.apply_fill(symbol, fill, OrderSide.BUY)


@pytest.mark.parametrize("bad_price", [0.0, -1.0, float("nan"), float("inf")])
def test_mark_rejects_invalid_price(bad_price):
    tracker = PositionTracker(initial_equity=200_000.0)
    _open_long(tracker)
    tracker.mark("ETHUSDT", 2010.0)         # valid mark
    eq_before = tracker.equity
    tracker.mark("ETHUSDT", bad_price)       # invalid → must be ignored
    eq_after = tracker.equity
    assert eq_after == pytest.approx(eq_before), (
        f"invalid mark {bad_price!r} changed equity "
        f"({eq_before:.2f} → {eq_after:.2f}) — phantom-loss regression"
    )


def test_mark_valid_price_still_updates():
    tracker = PositionTracker(initial_equity=200_000.0)
    _open_long(tracker)
    tracker.mark("ETHUSDT", 2100.0)
    # +100 on 10 units = +1000 unrealised
    assert tracker.equity == pytest.approx(201_000.0, abs=1.0)
