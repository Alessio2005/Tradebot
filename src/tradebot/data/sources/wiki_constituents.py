# src/tradebot/data/sources/wiki_constituents.py
"""Point-in-time S&P 500 membership from Wikipedia (free).

Survivorship is THE equities silent killer (G8): backtesting on today's
members excludes every removed loser. This module reconstructs a
point-in-time membership calendar from the Wikipedia page's current-members
table + the "Selected changes" history table.

PIT: a membership change effective on date D is knowable at D (announced
days earlier by S&P DJI) — asof_ts = effective date is therefore
conservative-correct for trading FROM the effective date; we additionally
add a 1-day lag so reconstitution trades never assume same-day knowledge.

Honest limitation (register in DATA_REGISTER.md): the Wikipedia change
table is complete back to ~2000 but sparser before; coverage must be
validated against the universe start date of every backtest (Wave 21
done-criterion).
"""
from __future__ import annotations

import io

import pandas as pd

from .base import ASOF_COL, EVENT_COL, SourceMeta, http_get_text, validate_pit

__all__ = ["META", "fetch_membership_events", "build_membership_calendar"]

_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
META = SourceMeta(
    name="wiki_constituents",
    url=_URL,
    license="free (CC BY-SA; factual data)",
    publication_lag="effective date + 1 day (changes are announced before effectiveness)",
    coverage="S&P 500 current members + change history (~2000->present reliable)",
    survivorship="THIS source is the survivorship fix — includes removals",
    quality_notes="validate change-table completeness vs backtest start (Wave 21)",
)

_LAG = pd.Timedelta(days=1)


def fetch_membership_events() -> pd.DataFrame:
    """Parse both Wikipedia tables into add/remove events.

    Returns PIT frame [symbol, action(add|remove), event_ts, asof_ts].
    Current members without a known add date get event_ts = 1990-01-01 and a
    'date_unknown' flag — they may NOT be assumed members before the first
    verifiable date in a stricter universe build.
    """
    html = http_get_text(_URL)
    tables = pd.read_html(io.StringIO(html))

    def _flat(t: pd.DataFrame) -> pd.DataFrame:
        t = t.copy()
        if isinstance(t.columns, pd.MultiIndex):
            t.columns = [
                "_".join(str(p) for p in c if str(p) not in ("nan", ""))
                for c in t.columns
            ]
        t.columns = [str(c).lower().strip().replace(" ", "_") for c in t.columns]
        return t

    # Robust table selection (page layout shifts; observed StopIteration
    # 2026-06-10): current = table with a symbol/ticker column; changes =
    # table with both an added- and a removed-column.
    current = changes = None
    for t in map(_flat, tables):
        cols = set(t.columns)
        if current is None and any(c in ("symbol", "ticker") for c in cols):
            current = t
        if changes is None and any("added" in c for c in cols) and any(
            "removed" in c for c in cols
        ):
            changes = t
    if current is None or changes is None:
        raise ValueError(
            "Wikipedia page layout changed: could not locate current-members "
            f"and/or changes table (saw {len(tables)} tables)."
        )

    def _col(frame: pd.DataFrame, *needles: str) -> str:
        for c in frame.columns:
            if all(n in c for n in needles):
                return c
        raise ValueError(f"No column matching {needles} in {list(frame.columns)}")

    date_col = _col(changes, "date")
    try:
        added_col = _col(changes, "added", "ticker")
        removed_col = _col(changes, "removed", "ticker")
    except ValueError:  # header variant without 'ticker' sublevel
        added_col = _col(changes, "added")
        removed_col = _col(changes, "removed")

    rows: list[dict] = []
    dates = pd.to_datetime(changes[date_col], errors="coerce", utc=True)
    for i in range(len(changes)):
        if pd.isna(dates.iloc[i]):
            continue
        for col, action in ((added_col, "add"), (removed_col, "remove")):
            sym = changes[col].iloc[i]
            if isinstance(sym, str) and sym.strip():
                rows.append(
                    {"symbol": sym.strip().upper(), "action": action,
                     EVENT_COL: dates.iloc[i], "date_unknown": False}
                )

    # current members not covered by any 'add' event -> unknown-date adds
    sym_col = next(c for c in current.columns if c in ("symbol", "ticker"))
    cur_syms = {str(s).strip().upper() for s in current[sym_col].dropna()}
    known_adds = {r["symbol"] for r in rows if r["action"] == "add"}
    for sym in sorted(cur_syms - known_adds):
        rows.append(
            {"symbol": sym, "action": "add",
             EVENT_COL: pd.Timestamp("1990-01-01", tz="UTC"), "date_unknown": True}
        )

    df = pd.DataFrame(rows)
    df[ASOF_COL] = df[EVENT_COL] + _LAG
    return validate_pit(df, required=("symbol", "action"))


def build_membership_calendar(
    events: pd.DataFrame, dates: pd.DatetimeIndex
) -> pd.DataFrame:
    """Boolean membership matrix [dates x symbols], built ONLY from asof_ts.

    A symbol is in the tradeable set at date t iff its latest event with
    asof_ts <= t is an 'add'. The Wave-21 lookahead test asserts no symbol
    appears before its add became knowable.
    """
    if dates.tz is None:
        raise ValueError("dates must be tz-aware UTC")
    ev = events.sort_values(ASOF_COL, kind="stable")
    symbols = sorted(ev["symbol"].unique())
    cal = pd.DataFrame(False, index=dates, columns=symbols)
    for sym, grp in ev.groupby("symbol"):
        state = False
        prev_t = None
        for _, row in grp.iterrows():
            t = row[ASOF_COL]
            if prev_t is not None and state:
                cal.loc[(cal.index >= prev_t) & (cal.index < t), sym] = True
            state = row["action"] == "add"
            prev_t = t
        if prev_t is not None and state:
            cal.loc[cal.index >= prev_t, sym] = True
    return cal
