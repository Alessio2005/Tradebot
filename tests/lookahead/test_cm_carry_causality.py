# tests/lookahead/test_cm_carry_causality.py
"""G6 causality guard for cm_carry_energy (R-1).

The unit derives returns from SLOT series, so it carries a failure mode the
other units do not: if the roll calendar can be influenced by how far the data
happens to run, the return on a roll day is computed against the wrong
contract. For natural gas — averaging -27.5%/yr of carry — one misplaced roll
injects ~2.3% of return that nobody earned, which is the same contamination
this program measured on free continuous futures (+25.1%/yr).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradebot.alpha import cm_carry


def _synthetic_term_structure(
    n_days: int = 1600, seed: int = 28, contango: float = 0.01
) -> pd.DataFrame:
    """Long PIT frame with a known monthly roll and a persistent curve slope."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2015-01-01", periods=n_days, freq="B", tz="UTC")
    rows = []
    for prod, base, slope in (
        ("WTI", 50.0, -contango),   # backwardation
        ("NG", 3.0, contango),      # contango
        ("HO", 2.0, contango / 2),
        ("RBOB", 1.8, -contango / 2),
    ):
        spot = base * np.exp(np.cumsum(rng.normal(0.0, 0.02, n_days)))
        for tenor in (1, 2, 3, 4):
            rows.append(pd.DataFrame({
                "product": prod,
                "tenor": tenor,
                "event_ts": idx,
                "settle": spot * (1.0 + slope * (tenor - 1)),
            }))
    df = pd.concat(rows, ignore_index=True)
    df["asof_ts"] = df["event_ts"] + pd.Timedelta(days=1)
    return df


@pytest.mark.parametrize("cut", [900, 1100, 1337])
def test_truncation_invariance_no_future_leakage(cut: int) -> None:
    """Returns up to T must not change when data after T is removed.

    This is where a data-derived roll calendar would betray itself: the last
    bar of a truncated panel is the last bar of ITS month, so a
    ``groupby(month).max()``-style expiry rule would invent a roll there.
    """
    ts = _synthetic_term_structure()
    keep = sorted(ts["event_ts"].unique())[:cut]
    trunc_ts = ts[ts["event_ts"].isin(keep)]

    full = cm_carry.run(ts)
    trunc = cm_carry.run(trunc_ts)
    common = trunc.net_returns.index

    pd.testing.assert_frame_equal(
        full.weights.loc[common], trunc.weights.loc[common]
    )
    pd.testing.assert_series_equal(
        full.net_returns.loc[common], trunc.net_returns.loc[common]
    )


def test_weights_are_held_one_bar_before_earning() -> None:
    """The return on bar t must be earned by the weights formed on bar t-1."""
    res = cm_carry.run(_synthetic_term_structure())
    expected = (
        res.weights.shift(1).fillna(0.0) * res.instrument_returns.fillna(0.0)
    ).sum(axis=1)
    pd.testing.assert_series_equal(
        res.gross_returns, expected, check_names=False
    )
    same_bar = (res.weights * res.instrument_returns.fillna(0.0)).sum(axis=1)
    assert not np.allclose(
        res.gross_returns.fillna(0.0), same_bar.fillna(0.0)
    ), "gross return matches same-bar weights — lookahead"


def test_future_prices_cannot_move_past_weights() -> None:
    ts = _synthetic_term_structure()
    base = cm_carry.run(ts).weights
    dates = sorted(ts["event_ts"].unique())
    bumped = ts.copy()
    bumped.loc[bumped["event_ts"] >= dates[1200], "settle"] *= 1.5
    after = cm_carry.run(bumped).weights

    pd.testing.assert_frame_equal(base.iloc[:1190], after.iloc[:1190])


def test_roll_days_are_monthly_and_not_data_derived() -> None:
    """~12 rolls a year, and the count does not depend on where data stops."""
    ts = _synthetic_term_structure()
    slots = cm_carry.build_slot_panel(ts, "NG")
    rolled = cm_carry.detect_rolls(slots, "NG")
    years = (slots.index[-1] - slots.index[0]).days / 365.25
    assert 10.0 <= rolled.sum() / years <= 14.0, (
        f"{rolled.sum() / years:.2f} rolls/yr is not monthly"
    )

    trunc = cm_carry.detect_rolls(slots.iloc[:1000], "NG")
    common = trunc.index
    pd.testing.assert_series_equal(rolled.loc[common], trunc.loc[common])


def test_roll_return_prices_the_contract_actually_held() -> None:
    """On a roll day the held contract has moved DOWN to slot k-1.

    Under a persistent contango a long position genuinely LOSES the one-month
    spread as its contract converges toward the front. The slot-naive series
    (today's slot 2 over yesterday's slot 2) hides that cost entirely, which is
    the direction of the continuous-futures contamination this program measured
    (+7%/yr WTI, +25%/yr NG — free series looking better than reality).
    """
    ts = _synthetic_term_structure(contango=0.05)
    slots = cm_carry.build_slot_panel(ts, "NG")   # NG is the contango leg
    rolled = cm_carry.detect_rolls(slots, "NG")
    r = cm_carry.contract_consistent_returns(slots, "NG", held_slot=2)
    naive = slots[2] / slots[2].shift(1) - 1.0

    on_roll = rolled & r.notna()
    assert on_roll.sum() > 5, "need roll days to test"
    # the honest series pays the roll; the naive one does not.
    assert r[on_roll].mean() < naive[on_roll].mean() - 0.02, (
        f"roll-day return {r[on_roll].mean():.4f} does not pay the contango "
        f"the slot-naive series hides ({naive[on_roll].mean():.4f})"
    )
    # and it is approximately one slot-step of the curve
    step = (slots[1] / slots[2] - 1.0)[on_roll].mean()
    assert abs(r[on_roll].mean() - (naive[on_roll].mean() + step)) < 0.01


def test_roll_is_charged_as_a_trade() -> None:
    """The roll moves the position back up a slot — that costs spread."""
    ts = _synthetic_term_structure()
    res = cm_carry.run(ts)
    rolled_any = res.rolls.reindex(res.daily_turnover.index).fillna(False).any(axis=1)
    # turnover on roll days must exceed a quiet mid-month hold (which is ~0)
    quiet = res.daily_turnover[~rolled_any]
    on_roll = res.daily_turnover[rolled_any]
    assert on_roll.mean() > quiet.mean() + 0.5, (
        f"roll days show turnover {on_roll.mean():.3f} vs quiet "
        f"{quiet.mean():.3f} — the roll trade is not being charged"
    )


def test_partial_curve_rows_are_dropped_not_filled() -> None:
    """A missing tenor makes the slope undefined; filling it invents curve shape."""
    ts = _synthetic_term_structure()
    ts = ts[~((ts["product"] == "WTI") & (ts["tenor"] == 3)
              & (ts["event_ts"] == sorted(ts["event_ts"].unique())[500]))]
    slots = cm_carry.build_slot_panel(ts, "WTI")
    assert slots.notna().all().all()
    assert sorted(ts["event_ts"].unique())[500] not in slots.index
