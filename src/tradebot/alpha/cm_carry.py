# src/tradebot/alpha/cm_carry.py
"""Wave 28 — energy term-structure carry (Gorton-Rouwenhorst; Koijen et al. 2018).

Prior: a commodity futures curve in backwardation earns a positive roll return
and one in contango a negative one; sorting on the curve slope is one of the
oldest documented commodity premia (Gorton & Rouwenhorst 2006, Fin. Anal. J.
62(2); Koijen, Moskowitz, Pedersen & Vrugt 2018, "Carry", J. Fin. Econ. 127(2)).
This is a DIFFERENT mechanism from ``cm_tsmom`` — the signal is the shape of
the forward curve, not past returns.

Design decisions, fixed BEFORE the first backtest (F12 — no tuning layer), and
deliberately identical to ``cm_tsmom`` wherever a choice is arbitrary, so the
two units are comparable and this one adds no new degrees of freedom:
  * signal   = (F1 - F2) / F1, the pre-registered slope (SETUP_B_ML_PROMPT §3.2),
               annualised x12 for reporting; only its SIGN drives the book.
  * held leg = CONTRACT 2, not contract 1. Slot 1 runs into delivery and its
               liquidity and settlement risk are not what the premium is about.
  * sizing   = sign(carry) / vol(60d), gross normalised to 1, monthly rebalance.
  * horizon  = time-series per product, NOT cross-sectional: four correlated
               energy products cannot support a cross-section (F10).

THE ROLL PROBLEM, AND WHY THIS IS NOT A CONTINUOUS SERIES
----------------------------------------------------------
EIA publishes settlement prices by SLOT ("Contract 1..4"), not by contract. On
a roll day the whole curve shifts down one slot, so ``F2(t)/F2(t-1) - 1`` is
comparing two DIFFERENT contracts and manufactures a return that nobody earned.
That is the same class of error as the free continuous futures this program
already measured at +7.0%/yr (WTI) and +25.1%/yr (nat gas).

So the return is computed contract-consistently:

    non-roll day:  r(t) = F2(t) / F2(t-1) - 1     (same contract both days)
    roll day:      r(t) = F2(t) / F3(t-1) - 1     (today's slot 2 was
                                                   yesterday's slot 3)

which needs no back-adjustment and invents nothing. Roll days are DETECTED, and
the detector is validated by counting: NYMEX energy contracts are monthly, so a
correct detector must find ~12 rolls per year per product. See
``detect_rolls`` and ``roll_diagnostics``.

FROZEN ARCHIVE — the caveat travels with every number produced here: the EIA
series stop at 2024-04-05, so this unit has **no recent OOS window**. It cannot
be run live from this source.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

from tradebot.alpha.xs_unit import CostModel, ann_sharpe

# NYMEX observes the US federal holiday set closely enough for a roll DATE; any
# residual mismatch shifts a roll by at most one session, and
# ``validate_roll_calendar`` measures whether prices agree with the result.
_US_CALENDAR = USFederalHolidayCalendar()

__all__ = [
    "UNIT",
    "PRIOR",
    "COST",
    "run",
    "CarryResult",
    "detect_rolls",
    "expiry_dates",
    "validate_roll_calendar",
    "build_slot_panel",
    "carry_signal",
    "contract_consistent_returns",
    "effective_breadth",
]

UNIT = "cm_carry_energy"
PRIOR = (
    "Gorton & Rouwenhorst (2006), Fin. Anal. J. 62(2) 47-68; "
    "Koijen, Moskowitz, Pedersen & Vrugt (2018), J. Fin. Econ. 127(2) 197-225"
)

_VOL_WINDOW = 60
_TRADING_DAYS = 252
_MONTHS_PER_YEAR = 12

# Futures, not ETFs: there is no borrow fee on a short futures position (margin,
# not stock loan). Commission + half-spread kept at the registered xasset level
# so cm_carry and cm_tsmom are costed on the same basis; NYMEX WTI is tighter
# than this and RBOB wider, so it is a fair average and is stressed at +50%.
COST = CostModel(commission_bps=1.0, half_spread_bps=3.0, borrow_fee_ann=0.0)


@dataclass(frozen=True)
class CarryResult:
    unit: str
    net_returns: pd.Series
    gross_returns: pd.Series
    weights: pd.DataFrame
    daily_turnover: pd.Series
    instrument_returns: pd.DataFrame
    carry: pd.DataFrame
    rolls: pd.DataFrame
    config: dict

    def summary(self, start: str | None = None) -> dict:
        net, gross = self.net_returns.dropna(), self.gross_returns.dropna()
        if start is not None:
            net, gross = net.loc[start:], gross.loc[start:]
        years = net.groupby(net.index.year).apply(lambda r: (1 + r).prod() - 1)
        eq = (1 + net).cumprod()
        dd = float((eq / eq.cummax() - 1.0).min())
        ann_vol = float(net.std() * np.sqrt(_TRADING_DAYS))
        cagr = float((1 + net).prod() ** (_TRADING_DAYS / max(len(net), 1)) - 1)
        return {
            "unit": self.unit,
            "n_bars": int(len(net)),
            "sample_years": round(len(net) / _TRADING_DAYS, 2),
            "net_sharpe": ann_sharpe(net),
            "gross_sharpe": ann_sharpe(gross),
            "net_cagr": cagr,
            "ann_vol": ann_vol,
            "max_drawdown": abs(dd),
            "calmar": cagr / abs(dd) if dd else float("nan"),
            "dd_over_vol": abs(dd) / ann_vol if ann_vol else float("nan"),
            "years_positive_frac": float((years > 0).mean()),
            "n_years": int(len(years)),
            "ann_turnover": float(self.daily_turnover.loc[net.index].sum()
                                  / max(len(net) / _TRADING_DAYS, 1e-9)),
            "config": self.config,
        }


def build_slot_panel(ts: pd.DataFrame, product: str) -> pd.DataFrame:
    """Wide slot panel [event_ts x tenor] for one product, complete tenors only.

    Rows missing any of the four tenors are dropped: a partial curve makes both
    the slope and the roll detector undefined, and silently forward-filling one
    would fabricate curve shape.
    """
    sub = ts[ts["product"] == product]
    wide = sub.pivot_table(index="event_ts", columns="tenor", values="settle")
    wide = wide.reindex(columns=[1, 2, 3, 4]).dropna().sort_index()
    if not isinstance(wide.index, pd.DatetimeIndex) or wide.index.tz is None:
        raise ValueError(f"{product}: slot panel needs a tz-aware UTC index")
    return wide


def expiry_dates(idx: pd.DatetimeIndex, product: str) -> pd.DatetimeIndex:
    """Contract expiry dates from the CME specification.

    Published rules, not inference:
      CL  WTI        3 business days before the 25th calendar day of the month
                     preceding delivery (or before the business day preceding
                     the 25th, when the 25th is not one).
      NG  Henry Hub  3 business days before the 1st of the delivery month.
      HO  ULSD       last business day of the month preceding delivery.
      RB  RBOB       last business day of the month preceding delivery.

    Computed on the US federal business-day calendar, NOT on the panel index.
    An earlier version derived the business days from the data, which made a
    truncated panel unable to see an expiry it would have known about in real
    time — exchange calendars are published years ahead. ``idx`` only supplies
    the date RANGE; no expiry depends on how far the data happens to run.
    """
    idx = pd.DatetimeIndex(idx).sort_values()
    naive = idx.tz_convert(None)
    bday = pd.offsets.CustomBusinessDay(calendar=_US_CALENDAR)

    out: list[pd.Timestamp] = []
    months = pd.period_range(
        naive.min().to_period("M"), naive.max().to_period("M"), freq="M"
    )
    for m in months:
        if product == "WTI":
            ref = pd.Timestamp(year=m.year, month=m.month, day=25)
            # roll back to a business day, then 3 business days before it
            b = ref if _is_bday(ref, bday) else ref - bday
            exp = b - 3 * bday
        elif product == "NG":
            exp = (m + 1).to_timestamp() - 3 * bday
        elif product in ("HO", "RBOB"):
            last_cal = (m + 1).to_timestamp() - pd.Timedelta(days=1)
            exp = last_cal if _is_bday(last_cal, bday) else last_cal - bday
        else:
            raise ValueError(f"no expiry rule registered for {product!r}")
        out.append(exp)
    return pd.DatetimeIndex(sorted(set(out))).tz_localize("UTC")


def _is_bday(ts: pd.Timestamp, bday: pd.offsets.CustomBusinessDay) -> bool:
    return bool(bday.is_on_offset(ts))


def detect_rolls(slots: pd.DataFrame, product: str) -> pd.Series:
    """Boolean series: do the slots shift down one at t? Calendar-derived.

    An earlier version inferred this from prices — "is F1(t) closer to F2(t-1)
    than to F1(t-1)". It does not work: adjacent-slot spreads are small next to
    daily noise, so the test fired 51-78 times a year against a true rate of
    ~12, with a diffuse day-of-month distribution. Since natural gas averages
    -27.5%/yr of carry, mis-assigning its rolls injects ~25%/yr of return that
    nobody earned — the same contamination this program measured on the free
    continuous ``NG=F`` series (+25.1%/yr). So the roll date is taken from the
    contract specification, and ``validate_roll_calendar`` checks that rule
    against the prices where the curve is steep enough to have power.
    """
    idx = pd.DatetimeIndex(slots.index).sort_values()
    exp = expiry_dates(idx, product)
    pos = np.searchsorted(idx.values, exp.values, side="right")
    pos = pos[pos < len(idx)]            # slots shift the day AFTER expiry
    rolled = pd.Series(False, index=idx, name="rolled")
    rolled.iloc[np.unique(pos)] = True
    rolled.iloc[0] = False               # no previous bar => unknowable
    return rolled


def _shift_evidence(slots: pd.DataFrame) -> pd.Series:
    """same_score - shift_score in price units; positive => the slots shifted.

    Absolute differences, not logs: WTI settled NEGATIVE in April 2020 and a
    log transform silently yields NaN there.
    """
    same = sum((slots[k] - slots[k].shift(1)).abs() for k in (1, 2, 3))
    shift = sum((slots[k] - slots[k + 1].shift(1)).abs() for k in (1, 2, 3))
    return (same - shift).rename("evidence")


def validate_roll_calendar(slots: pd.DataFrame, product: str) -> dict:
    """Independent check that the calendar rule matches what prices did.

    ``auc`` is P(shift evidence at a calendar roll > at a non-roll), measured on
    the steepest decile of the curve — the only regime where the test has power.
    """
    rolled = detect_rolls(slots, product)
    ev = _shift_evidence(slots)
    steep = (slots[1] - slots[2]).abs() / slots[1].abs().clip(lower=1e-6)
    mask = steep >= steep.quantile(0.90)

    a = ev[mask & rolled].dropna()
    b = ev[mask & ~rolled].dropna()
    if len(a) and len(b):
        ranks = pd.Series(np.concatenate([a.values, b.values])).rank().to_numpy()
        auc = float(
            (ranks[: len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b))
        )
    else:
        auc = float("nan")

    years = (slots.index[-1] - slots.index[0]).days / 365.25
    return {
        "product": product,
        "n_rolls": int(rolled.sum()),
        "rolls_per_year": float(rolled.sum() / years),
        "median_day_of_month": int(np.median(pd.DatetimeIndex(rolled[rolled].index).day)),
        "shift_evidence_auc_steep_decile": auc,
    }


def contract_consistent_returns(
    slots: pd.DataFrame, product: str, held_slot: int = 2
) -> pd.Series:
    """Return of holding ``held_slot`` and rolling — same contract on both dates.

    On a roll day the held contract has moved up one slot overnight, so the
    comparison base is ``held_slot + 1`` from yesterday. Never compares two
    different contracts, so no back-adjustment is needed and no roll return is
    invented.
    """
    if held_slot not in (2, 3, 4):
        raise ValueError(
            f"held_slot must be 2, 3 or 4 (the held contract moves DOWN to "
            f"slot-1 on a roll), got {held_slot}"
        )
    rolled = detect_rolls(slots, product)
    # Contracts move DOWN a slot as the front expires: the contract held in
    # slot k yesterday sits in slot k-1 today. So the price of the position
    # actually held is today's slot k-1, compared with yesterday's slot k.
    #
    # Comparing today's slot k with yesterday's slot k+1 would also be
    # "contract-consistent", but it prices a contract that was never held —
    # today's slot k was yesterday's slot k+1, which is not the position.
    today = slots[held_slot].where(~rolled, slots[held_slot - 1])
    base = slots[held_slot].shift(1)
    return (today / base - 1.0).rename("ret")


def carry_signal(slots: pd.DataFrame) -> pd.Series:
    """Annualised curve slope ``(F1 - F2) / F1 * 12``. Positive = backwardation."""
    return (((slots[1] - slots[2]) / slots[1]) * _MONTHS_PER_YEAR).rename("carry")


def effective_breadth(weights: pd.DataFrame, returns: pd.DataFrame) -> float:
    """N_eff = N / (1 + (N-1)*rho_bar) — the F10 guard against fake breadth."""
    active = [c for c in weights.columns if (weights[c] != 0).any()]
    if len(active) < 2:
        return float(len(active))
    corr = returns[active].corr().to_numpy()
    n = len(active)
    off = (corr.sum() - np.trace(corr)) / (n * (n - 1))
    return float(n / (1.0 + (n - 1) * off))


def run(
    term_structure: pd.DataFrame,
    cost: CostModel = COST,
    held_slot: int = 2,
    min_products: int = 2,
) -> CarryResult:
    """Run the time-series carry book on the EIA term-structure archive.

    ``term_structure`` is the long PIT frame from ``data.sources.eia``.
    Signals formed on bar t are held into bar t+1, so no row earns its own
    signal (R-1).
    """
    ts = term_structure.copy()
    ts["event_ts"] = pd.to_datetime(ts["event_ts"], utc=True)
    products = sorted(ts["product"].unique())

    rets, carries, rolls = {}, {}, {}
    for p in products:
        slots = build_slot_panel(ts, p)
        rets[p] = contract_consistent_returns(slots, p, held_slot=held_slot)
        carries[p] = carry_signal(slots)
        rolls[p] = detect_rolls(slots, p)

    ret_df = pd.DataFrame(rets).sort_index()
    carry_df = pd.DataFrame(carries).reindex(ret_df.index)
    roll_df = pd.DataFrame(rolls).reindex(ret_df.index).fillna(False)

    # A day with fewer than min_products live curves cannot carry a book.
    enough = carry_df.notna().sum(axis=1) >= min_products
    ret_df, carry_df = ret_df.loc[enough], carry_df.loc[enough]
    roll_df = roll_df.loc[enough]

    vol = ret_df.rolling(_VOL_WINDOW).std()
    raw = (np.sign(carry_df) / vol).replace([np.inf, -np.inf], np.nan)

    # Rebalance on the FIRST bar of each month, decided against the PREVIOUS
    # bar only — the truncation-invariant calendar (W27 fix, W28 §2).
    period = raw.index.tz_convert(None).to_period("M")
    is_rb = pd.Series(period != np.roll(period, 1), index=raw.index)
    is_rb.iloc[0] = True
    w_rb = raw.loc[is_rb.to_numpy()]
    w_rb = w_rb.div(w_rb.abs().sum(axis=1).replace(0, np.nan), axis=0)
    weights = w_rb.reindex(raw.index).ffill().fillna(0.0)

    held = weights.shift(1).fillna(0.0)          # w(t-1) earns r(t) — causal
    gross = (held * ret_df.fillna(0.0)).sum(axis=1)
    rebalance_turnover = (weights - held).abs().sum(axis=1)

    # The roll is a TRADE, not just an accounting step: on a roll day the held
    # contract has moved to slot k-1 and must be sold and re-bought in slot k.
    # That is a full round trip on the held notional, ~12x a year per product.
    # Leaving it out would flatter the book by roughly 12 * 2 * per_side.
    roll_flags = roll_df.reindex(index=held.index, columns=held.columns)
    roll_flags = roll_flags.fillna(False).astype(bool)
    roll_turnover = (held.abs() * roll_flags).sum(axis=1) * 2.0

    turnover = rebalance_turnover + roll_turnover
    net = gross - turnover * cost.per_side

    return CarryResult(
        unit=UNIT,
        net_returns=net,
        gross_returns=gross,
        weights=weights,
        daily_turnover=turnover,
        instrument_returns=ret_df,
        carry=carry_df,
        rolls=roll_df,
        config={
            "signal": "(F1-F2)/F1 annualised x12, sign only",
            "held_slot": held_slot,
            "vol_window": _VOL_WINDOW,
            "rebalance": "ME (first bar of month, truncation-invariant)",
            "products": products,
            "min_products": min_products,
            "commission_bps": cost.commission_bps,
            "half_spread_bps": cost.half_spread_bps,
            "borrow_fee_ann": cost.borrow_fee_ann,
            "prior": PRIOR,
            "data_caveat": "EIA frozen archive ends 2024-04-05 — no recent OOS window",
        },
    )
