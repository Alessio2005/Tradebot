# src/tradebot/data/sources/kenfrench.py
"""Ken French Data Library — daily factor returns (free). Required for G4.

Datasets: FF5 daily (Mkt-RF, SMB, HML, RMW, CMA, RF) + Momentum daily (MOM).

PIT note: the library is updated with a multi-week publication delay; the
default availability lag is a conservative 90 days after the return date.
For the G4 *evaluation* regression (diagnostic, not a trading signal) the
lag is irrelevant; any unit that ever CONDITIONS on factor values must use
``asof_ts`` through ``asof_join`` like every other source.
"""
from __future__ import annotations

import io
import zipfile

import pandas as pd

from .base import EVENT_COL, SourceMeta, stamp_asof, validate_pit

__all__ = ["META", "fetch_ff5_daily", "fetch_momentum_daily", "fetch_factors_daily"]

_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp"
META = SourceMeta(
    name="kenfrench",
    url=f"{_BASE}/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip",
    license="free (Ken French Data Library, academic)",
    publication_lag="monthly refresh; conservative 90d availability lag",
    coverage="US factor returns daily, 1963->present",
    survivorship="n/a (factor returns, CRSP-based)",
    quality_notes="percent units in raw file -> converted to decimal here",
)

_DEFAULT_LAG = pd.Timedelta(days=90)


def _fetch_zip_csv(url: str) -> pd.DataFrame:
    import requests

    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        raw = zf.read(zf.namelist()[0]).decode("utf-8", errors="replace")
    # File has a text preamble; data rows start where col0 is YYYYMMDD.
    lines = raw.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln[:8].strip().isdigit())
    header = lines[start - 1]
    body = "\n".join([header] + [
        ln for ln in lines[start:] if ln[:8].strip().isdigit()
    ])
    df = pd.read_csv(io.StringIO(body))
    df = df.rename(columns={df.columns[0]: "date"})
    df["date"] = pd.to_datetime(df["date"].astype(str).str.strip(), format="%Y%m%d", utc=True)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def fetch_ff5_daily(lag: pd.Timedelta = _DEFAULT_LAG) -> pd.DataFrame:
    df = _fetch_zip_csv(f"{_BASE}/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip")
    for c in df.columns:
        if c != "date":
            df[c] = pd.to_numeric(df[c], errors="coerce") / 100.0  # % -> decimal
    df[EVENT_COL] = df.pop("date")
    df = stamp_asof(df, lag)
    return validate_pit(df, required=("Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"))


def fetch_momentum_daily(lag: pd.Timedelta = _DEFAULT_LAG) -> pd.DataFrame:
    df = _fetch_zip_csv(f"{_BASE}/F-F_Momentum_Factor_daily_CSV.zip")
    df.columns = ["date" if c == "date" else c.strip() for c in df.columns]
    mom_col = next(c for c in df.columns if c.lower().startswith("mom"))
    df = df.rename(columns={mom_col: "MOM"})
    df["MOM"] = pd.to_numeric(df["MOM"], errors="coerce") / 100.0
    df[EVENT_COL] = df.pop("date")
    df = stamp_asof(df[[EVENT_COL, "MOM"]], lag)
    return validate_pit(df, required=("MOM",))


def fetch_factors_daily(lag: pd.Timedelta = _DEFAULT_LAG) -> pd.DataFrame:
    """FF5 + MOM merged on event date — the G4 equities factorset (ex-BAB).

    BAB comes from the AQR data library (separate download, registered
    independently in the data register).
    """
    ff5 = fetch_ff5_daily(lag)
    mom = fetch_momentum_daily(lag)
    merged = ff5.merge(mom[[EVENT_COL, "MOM"]], on=EVENT_COL, how="inner")
    merged = merged.sort_values("asof_ts", kind="stable").reset_index(drop=True)
    return validate_pit(merged, required=("Mkt-RF", "SMB", "HML", "RMW", "CMA", "MOM"))
