# src/tradebot/data/xasset_proxy.py
"""Wave 27 — cross-asset total-return proxy panel (Mandate v3 §5.4).

WHY PROXIES AND NOT FUTURES CONTINUATIONS (measured, 2026-08-10):
    Free continuous futures series (Yahoo ``CL=F`` etc.) are FRONT-MONTH and
    NOT back-adjusted, so they track spot and silently omit the roll yield.
    Measured against the actual rolled vehicles over 2010-2026:

        CL=F  +3.38%/yr   vs USO  -3.63%/yr   ->  +7.01%/yr missing
        NG=F  -5.02%/yr   vs UNG -30.07%/yr   -> +25.05%/yr missing

    A TSMOM backtest on those series would book 7-25%/yr of return that a
    trader cannot collect — an order of magnitude larger than the premium
    being tested. The ETF/ETC panel below is roll-INCLUSIVE by construction
    (NAV holds and rolls the contracts) and dividend-adjusted, which is
    exactly the §5.4 "ETF-proxies" route.

    The other free term-structure source, EIA's NYMEX Contract 1-4, is a
    FROZEN ARCHIVE: it stops at 2024-04-05 ("futures prices after April 5,
    2024, are not available"). Usable for a carry backtest 1980-2024, NOT
    for a unit that must live through the recent window. Registered in
    DATA_REGISTER.md with that limitation.

PIT rule: EOD close -> knowable next calendar day 00:00 UTC (same convention
as stooq/cboe in the register). Units index on ``event_ts`` and apply their
own one-bar hold shift; the as-of stamp is what the register audits.

Survivorship (G8): the panel is built from ETFs alive TODAY. Dead funds
(e.g. closed commodity ETNs) are absent -> a mild upward bias on the long
side. Documented, not silently ignored; the instruments chosen are large
survivors whose closure risk over the sample was negligible.
"""
from __future__ import annotations

import pandas as pd

from .sources.base import EVENT_COL, SourceMeta, stamp_asof, validate_pit

__all__ = [
    "META",
    "XASSET_UNIVERSE",
    "SECTORS",
    "fetch_panel",
    "to_tr_panel",
    "sector_of",
]

META = SourceMeta(
    name="yfinance_xasset_proxy",
    url="https://finance.yahoo.com (yfinance, auto_adjust=True)",
    license="free; ToS: personal/research use, no redistribution (mandate §5.2)",
    publication_lag="EOD close -> next calendar day 00:00 UTC",
    coverage="26 US-listed ETF/ETC total-return series, 6 sectors, 2004/2007->",
    survivorship="alive-today panel; dead funds absent (mild long-side bias)",
    quality_notes=(
        "auto_adjust=True => dividend-adjusted total return; commodity ETCs "
        "are roll-inclusive. Replaces futures continuations, which omit "
        "7-25%/yr of roll drag (measured 2026-08-10)."
    ),
)

# ticker -> (sector, description).  Selection rule, fixed BEFORE any backtest
# (F12: no instrument shopping): liquid (median ADV >= ~$5M over the last
# year), longest available history, and at most one instrument per distinct
# underlying risk. Thin names (CORN/SOYB $1.7M, FXA $0.7M, FXB $1.6M) and
# redundant duplicates (GSG~DBC, DBB) were excluded on that rule.
XASSET_UNIVERSE: dict[str, tuple[str, str]] = {
    # --- commodities (roll-inclusive ETCs) ---
    "USO": ("commodity", "WTI crude, front-month rolled"),
    "UNG": ("commodity", "Henry Hub natural gas, front-month rolled"),
    "GLD": ("commodity", "gold bullion"),
    "SLV": ("commodity", "silver bullion"),
    "PPLT": ("commodity", "platinum bullion"),
    "CPER": ("commodity", "copper futures"),
    "DBA": ("commodity", "agriculture basket"),
    "DBC": ("commodity", "broad commodity basket"),
    "WEAT": ("commodity", "wheat futures"),
    # --- rates ---
    "SHY": ("rates", "UST 1-3y"),
    "IEF": ("rates", "UST 7-10y"),
    "TLT": ("rates", "UST 20y+"),
    "TIP": ("rates", "US TIPS"),
    # --- credit ---
    "LQD": ("credit", "US investment-grade corporates"),
    "HYG": ("credit", "US high yield"),
    # --- equity index ---
    "SPY": ("equity", "S&P 500"),
    "QQQ": ("equity", "Nasdaq 100"),
    "IWM": ("equity", "Russell 2000"),
    "EFA": ("equity", "MSCI EAFE"),
    "EEM": ("equity", "MSCI emerging markets"),
    "EWJ": ("equity", "MSCI Japan"),
    # --- FX ---
    "UUP": ("fx", "USD index long"),
    "FXE": ("fx", "EUR/USD"),
    "FXY": ("fx", "JPY/USD"),
    "FXF": ("fx", "CHF/USD"),
    # --- real assets ---
    "VNQ": ("real_estate", "US REITs"),
}

SECTORS = ("commodity", "rates", "credit", "equity", "fx", "real_estate")

_PUB_LAG = pd.Timedelta(days=1)  # EOD close knowable next calendar day 00:00 UTC


def sector_of(ticker: str) -> str:
    return XASSET_UNIVERSE[ticker][0]


def fetch_panel(
    tickers: tuple[str, ...] | None = None,
    start: str = "2004-01-01",
    end: str | None = None,
) -> pd.DataFrame:
    """Download the adjusted-close panel and return it PIT-validated (long form).

    Columns: ``event_ts``, ``asof_ts``, ``symbol``, ``sector``, ``close``.
    ``close`` is a total-return index level (auto_adjust=True), not a raw price.
    """
    import yfinance as yf

    names = tuple(tickers or XASSET_UNIVERSE)
    raw = yf.download(
        list(names),
        start=start,
        end=end,
        progress=False,
        auto_adjust=True,
        threads=False,
    )
    if raw.empty:
        raise RuntimeError("yfinance returned an empty panel — refusing to proceed")
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    if not isinstance(raw.columns, pd.MultiIndex):
        close.columns = list(names)

    long = (
        close.rename_axis("event_ts")
        .reset_index()
        .melt(id_vars="event_ts", var_name="symbol", value_name="close")
        .dropna(subset=["close"])
    )
    long["event_ts"] = pd.to_datetime(long["event_ts"], utc=True)
    long["sector"] = long["symbol"].map(sector_of)
    long = stamp_asof(long, _PUB_LAG, event_col=EVENT_COL)
    return validate_pit(long, required=("symbol", "close"))


def to_tr_panel(long: pd.DataFrame) -> pd.DataFrame:
    """Long PIT frame -> wide total-return panel indexed by ``event_ts`` (UTC).

    Units consume this and apply their own one-bar hold shift, so the frame
    is deliberately indexed on the EVENT timestamp, not the as-of stamp.
    """
    wide = long.pivot_table(
        index=EVENT_COL, columns="symbol", values="close", aggfunc="last"
    ).sort_index()
    wide.index = pd.DatetimeIndex(wide.index, name="event_ts")
    return wide
