# src/tradebot/data/sources/fred.py
"""FRED — policy/3m rates per currency for FX carry (free, no key via CSV).

PIT rule: every series carries a RELEASE lag (macro data is published after
the observation period). The lag is a REQUIRED argument per series — there
is no safe default for macro releases, so this module refuses to guess.
Reference lags (document per series in the data register):
  - daily market rates (e.g. SOFR, 3m T-bill): 1 business day
  - monthly CPI-style releases: ~45 days
For exact real-time vintages ALFRED exists; for G10 carry the conservative
fixed lag is sufficient and simpler (registered in DATA_REGISTER.md).
"""
from __future__ import annotations

import pandas as pd

from .base import EVENT_COL, SourceMeta, http_get_text, read_csv_text, stamp_asof, validate_pit

__all__ = ["META", "fetch_series", "G10_RATE_SERIES", "G10_SPOT_SERIES"]

META = SourceMeta(
    name="fred",
    url="https://fred.stlouisfed.org/graph/fredgraph.csv?id=<SERIES>",
    license="free (St. Louis Fed; CSV endpoint, no key)",
    publication_lag="EXPLICIT per series (required arg) — e.g. 1bd for daily rates",
    coverage="US + international rates/macro; daily/monthly; decades",
    survivorship="n/a (macro series)",
    quality_notes="'.' = missing in raw CSV; ALFRED available if vintage-exact needed",
)

# G10 3m-rate proxies for the carry signal (Koijen et al. 2018).
# Values: (series_id, release_lag_days) — verify & register on first ingest.
G10_RATE_SERIES: dict[str, tuple[str, int]] = {
    "USD": ("DTB3", 1),
    "EUR": ("IR3TIB01EZM156N", 45),
    "JPY": ("IR3TIB01JPM156N", 45),
    "GBP": ("IR3TIB01GBM156N", 45),
    "CHF": ("IR3TIB01CHM156N", 45),
    "CAD": ("IR3TIB01CAM156N", 45),
    "AUD": ("IR3TIB01AUM156N", 45),
    "NZD": ("IR3TIB01NZM156N", 45),
    "SEK": ("IR3TIB01SEM156N", 45),
    "NOK": ("IR3TIB01NOM156N", 45),
}


# G10 spot series (FRED H.10, daily, 1bd release lag). direction=+1 means
# the series is quoted as USD per 1 unit foreign (what we standardise on);
# -1 means foreign per USD -> INVERT on load. Getting this map wrong flips
# the carry sign silently — it is unit-tested against known 2024 levels.
G10_SPOT_SERIES: dict[str, tuple[str, int]] = {
    "EUR": ("DEXUSEU", +1),
    "JPY": ("DEXJPUS", -1),
    "GBP": ("DEXUSUK", +1),
    "CHF": ("DEXSZUS", -1),
    "CAD": ("DEXCAUS", -1),
    "AUD": ("DEXUSAL", +1),
    "NZD": ("DEXUSNZ", +1),
    "SEK": ("DEXSDUS", -1),
    "NOK": ("DEXNOUS", -1),
}


def fetch_series(
    series_id: str,
    release_lag: pd.Timedelta,
    start_year: int = 1971,
    chunk_years: int = 2,
    pace_s: float = 2.0,
) -> pd.DataFrame:
    """Fetch one FRED series as a PIT frame [series_id, event_ts, value, asof_ts].

    ``release_lag`` is mandatory: the availability moment is observation
    date + lag. Underestimating the lag is lookahead; overestimating only
    costs signal freshness — when in doubt, round up.

    ``start_year`` lets callers skip useless pre-history (e.g. FX carry only
    trades from ~1999 when >=6 rate series exist, so spots before then are
    dead weight). ``chunk_years``/``pace_s`` tune the chunking — see below.
    """
    if release_lag < pd.Timedelta(0):
        raise ValueError("release_lag must be >= 0")
    # Strategy (G8 forensics 2026-06-11 — measured, not guessed):
    #   * fredgraph.csv has a per-YEAR server cost (~6.3s/yr for DEX series):
    #     1mo=0.8s, 1yr=9s, 2yr=15s, 3yr=22s, 5yr=~35s.
    #   * FRED's gateway hard-cuts at ~60s -> any request spanning >~9yr
    #     returns HTTP 504 (verified: an 11yr request 504'd at 60.3s).
    #   * The unbounded full-history request therefore ALWAYS 504s.
    #   * The /data/<SERIES> HTML page is truncated to exactly 1000 rows
    #     (verified: DEXUSEU HTML ended 2002-11 — the original spots bug).
    #   => Only safe path: date-BOUNDED chunks small enough to stay under the
    #      60s gateway, fetched with ENOUGH pacing that rapid-fire requests
    #      don't queue server-side into timeouts. Measured 2026-06-11:
    #      - 3yr/5yr chunks at 0.5s pacing -> the DEX series time out at 60s
    #        (rapid multi-year requests pile up server-side).
    #      - 2yr chunks at 2s pacing -> rock-solid: 5/5 consecutive at ~10-13s.
    #      So 2-year chunks + 2s pacing is the verified-reliable recipe.
    #      The http_get_text backoff still absorbs the rare transient 504.
    import time

    base = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    chunks: list[pd.DataFrame] = []
    for y0 in range(start_year, 2030, chunk_years):
        url = f"{base}&cosd={y0}-01-01&coed={y0 + chunk_years - 1}-12-31"
        c = read_csv_text(http_get_text(url, timeout=45))
        time.sleep(pace_s)
        if len(c):
            chunks.append(c)
    df = pd.concat(chunks, ignore_index=True).drop_duplicates()
    df.columns = [str(c).strip().lower() for c in df.columns]
    date_col = "observation_date" if "observation_date" in df.columns else "date"
    value_col = next(c for c in df.columns if c != date_col)
    df = df.rename(columns={value_col: "value"})
    df["value"] = pd.to_numeric(df["value"].replace(".", None), errors="coerce")
    df = df.dropna(subset=["value"])
    df[EVENT_COL] = pd.to_datetime(df[date_col], utc=True)
    df["series_id"] = series_id
    df = df[["series_id", EVENT_COL, "value"]]
    df = df.sort_values(EVENT_COL, kind="stable").reset_index(drop=True)
    # staleness alarm (G8): a "successful" fetch that ends years ago is a
    # silent-truncation symptom, not a success
    age_days = (pd.Timestamp.now(tz="UTC") - df[EVENT_COL].max()).days
    if age_days > 120:
        raise ValueError(
            f"{series_id}: last observation is {age_days}d old — "
            "truncated/stale fetch, refusing to persist (G8)."
        )
    df = stamp_asof(df, release_lag)
    return validate_pit(df, required=("series_id", "value"))
