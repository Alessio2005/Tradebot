# src/tradebot/data/sources/eia.py
"""EIA NYMEX futures Contract 1-4 — the only free decades-deep real term structure.

Mandate §5.4 / SETUP_B_ML_PROMPT §4: term-structure carry (Gorton-Rouwenhorst;
Koijen, Moskowitz, Pedersen & Vrugt 2018) needs the SLOPE of a real forward
curve. Free continuous futures (``CL=F``, ``NG=F``) cannot supply it — they are
front-month and not back-adjusted, and were measured to overstate returns by
+7.0%/yr (WTI) and +25.1%/yr (nat gas). Expired contracts are gone from Yahoo,
so the historical front curve is not reconstructable from there at all.

EIA publishes settlement prices for NYMEX Contract 1 through Contract 4 as
free daily series going back to 1983 (WTI), 1993/94 (Henry Hub), 1980 (heating
oil) and 2005 (RBOB). Four tenors is enough for a slope.

TWO HARD CONSTRAINTS, both already paid for once by this program
---------------------------------------------------------------
1. **FROZEN ARCHIVE.** The series stop at **2024-04-05**; EIA no longer
   publishes NYMEX futures prices. Excellent for a 1983-2024 backtest, unusable
   as a live feed. ``assert_archive_bounds`` makes that explicit rather than
   letting a stale tail masquerade as current data, and every carry conclusion
   must carry the label "no recent OOS window".
2. **``.xls`` ENDPOINT ONLY.** The HTML route (``hist/LeafHandler.ashx``)
   truncates silently — it returns a valid-looking page with the tail missing.
   This module refuses to use it.

PIT convention: settlement on day D is knowable at the next calendar day
00:00 UTC — the same conservative rule as ``stooq``/``cboe``.
"""
from __future__ import annotations

import io

import pandas as pd

from .base import EVENT_COL, SourceMeta, stamp_asof, validate_pit

__all__ = [
    "META",
    "PRODUCTS",
    "TENORS",
    "ARCHIVE_LAST_BAR",
    "fetch_contract",
    "fetch_term_structure",
    "assert_archive_bounds",
]

_BASE = "https://www.eia.gov/dnav/{section}/hist_xls/{series}d.xls"
_UA = {"User-Agent": "tradebot-research algulizia@gmail.com"}

# product -> (EIA section, series-id template, unit, human name)
PRODUCTS: dict[str, tuple[str, str, str, str]] = {
    "WTI": ("pet", "RCLC{t}", "USD/bbl", "Cushing WTI crude"),
    "NG": ("ng", "RNGC{t}", "USD/MMBtu", "Henry Hub natural gas"),
    "HO": ("pet", "EER_EPD2F_PE{t}_Y35NY_DPG", "USD/gal", "NY Harbor No.2 heating oil"),
    "RBOB": ("pet", "EER_EPMRR_PE{t}_Y35NY_DPG", "USD/gal", "NY Harbor RBOB gasoline"),
}
TENORS = (1, 2, 3, 4)

# The archive's final settlement date. Verified 2026-08-10.
ARCHIVE_LAST_BAR = pd.Timestamp("2024-04-05", tz="UTC")

_DEFAULT_LAG = pd.Timedelta(days=1)

META = SourceMeta(
    name="eia_nymex_futures",
    url=_BASE.format(section="{pet|ng}", series="<SERIES>"),
    license="free (US government, EIA)",
    publication_lag="settlement EOD -> next calendar day 00:00 UTC (conservative)",
    coverage=(
        "NYMEX Contract 1-4: WTI 1983->, Henry Hub NG 1993/94->, heating oil "
        "1980->, RBOB 2005->; daily. FROZEN: last bar 2024-04-05."
    ),
    survivorship="n/a — continuation tenors, not a name universe",
    quality_notes=(
        "Only free decades-deep real term structure. FROZEN ARCHIVE (stops "
        "2024-04-05): backtest only, never live. Use the .xls endpoint ONLY — "
        "the HTML LeafHandler.ashx route truncates silently."
    ),
)


def _http_get_bytes(url: str, timeout: int = 90) -> bytes:
    """GET a binary resource with the same backoff policy as ``http_get_text``.

    ``base.http_get_text`` decodes to str, which corrupts an OLE2 workbook, so
    this is the binary sibling rather than a reimplementation of the policy.
    """
    import time

    import requests

    for attempt in (1, 2, 3, 4):
        try:
            resp = requests.get(url, headers=_UA, timeout=timeout)
            resp.raise_for_status()
            if not resp.content.startswith(b"\xd0\xcf\x11\xe0"):
                raise ValueError(
                    f"{url} did not return an OLE2 workbook — EIA served an "
                    f"error page or the endpoint moved (first bytes "
                    f"{resp.content[:8]!r})"
                )
            return resp.content
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2.0**attempt)
    raise RuntimeError("unreachable")


def fetch_contract(
    product: str, tenor: int, lag: pd.Timedelta = _DEFAULT_LAG
) -> pd.DataFrame:
    """One (product, tenor) settlement series as a PIT frame.

    Columns: ``[product, tenor, event_ts, settle, asof_ts]``.
    """
    if product not in PRODUCTS:
        raise ValueError(f"unknown product {product!r}, have {sorted(PRODUCTS)}")
    if tenor not in TENORS:
        raise ValueError(f"tenor must be one of {TENORS}, got {tenor}")

    section, template, _unit, _name = PRODUCTS[product]
    url = _BASE.format(section=section, series=template.format(t=tenor))
    raw = _http_get_bytes(url)

    # Sheet layout: row0 title, row1 "Sourcekey", row2 header, row3+ data.
    df = pd.read_excel(io.BytesIO(raw), sheet_name="Data 1", skiprows=2)
    if df.shape[1] < 2:
        raise ValueError(f"{url}: expected >=2 columns, got {list(df.columns)}")
    df = df.iloc[:, :2]
    df.columns = [EVENT_COL, "settle"]

    df[EVENT_COL] = pd.to_datetime(df[EVENT_COL], utc=True, errors="coerce")
    df["settle"] = pd.to_numeric(df["settle"], errors="coerce")
    df = df.dropna(subset=[EVENT_COL, "settle"])
    if df.empty:
        raise ValueError(f"{url}: no usable rows after parsing")

    df["product"] = product
    df["tenor"] = int(tenor)
    df = df[["product", "tenor", EVENT_COL, "settle"]]
    df = stamp_asof(df, lag)
    return validate_pit(df, required=("product", "tenor", "settle"))


def fetch_term_structure(
    products: tuple[str, ...] = tuple(PRODUCTS),
    lag: pd.Timedelta = _DEFAULT_LAG,
) -> pd.DataFrame:
    """All tenors for the given products, long format.

    Long, not wide: the wide pivot is a modelling choice and belongs in the
    unit, not in the data layer.
    """
    frames = [
        fetch_contract(p, t, lag=lag) for p in products for t in TENORS
    ]
    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["asof_ts", "product", "tenor"], kind="stable")
    return validate_pit(out.reset_index(drop=True), required=("product", "tenor", "settle"))


def assert_archive_bounds(df: pd.DataFrame, tolerance_days: int = 5) -> None:
    """Fail loudly if the archive does not end where the register says it does.

    Two failure modes this catches, both of which would silently corrupt a
    carry backtest:

    * the tail is MISSING (the HTML endpoint's silent truncation) — the series
      would look like it ends early and the last years would vanish;
    * the tail EXTENDS past the frozen date — meaning EIA resumed publication,
      in which case the "no recent OOS window" caveat attached to every carry
      conclusion is no longer true and must be revisited, not quietly dropped.
    """
    last = pd.to_datetime(df[EVENT_COL], utc=True).max()
    delta = abs((last - ARCHIVE_LAST_BAR).days)
    if delta > tolerance_days:
        raise ValueError(
            f"EIA archive bound moved: last bar {last.date()} vs registered "
            f"{ARCHIVE_LAST_BAR.date()} ({delta}d). Either the .xls was "
            f"truncated, or EIA resumed publishing — both change what the "
            f"carry backtest means. Investigate before using this data."
        )
