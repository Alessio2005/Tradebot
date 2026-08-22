# src/tradebot/data/sources/edgar.py
"""SEC EDGAR — point-in-time fundamentals & filing events (free).

THE PIT rule here (Mandate §7, non-negotiable): asof_ts = the EDGAR
**acceptance timestamp** of the filing (when the market could read it) —
NEVER the fiscal period end date. Using period-end dates is the classic
fundamentals lookahead.

Endpoints (free, no key; UA header with contact required by SEC policy):
  https://data.sec.gov/submissions/CIK##########.json   (filing index)
  https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json (XBRL facts)
Rate limit: max 10 req/s — bulk ingests must sleep.
"""
from __future__ import annotations

import json
import time

import pandas as pd

from .base import ASOF_COL, EVENT_COL, SourceMeta, http_get_text, validate_pit

__all__ = ["META", "fetch_filing_index", "fetch_company_facts"]

META = SourceMeta(
    name="edgar",
    url="https://data.sec.gov/",
    license="free (US government data; UA with contact required)",
    publication_lag="none guessed — asof_ts = filing acceptance timestamp (true PIT)",
    coverage="all SEC registrants; filings 1993->present, XBRL facts ~2009->present",
    survivorship="includes delisted registrants (filings persist)",
    quality_notes="rate limit 10 req/s; acceptance time in ET -> converted to UTC",
)

_ET = "America/New_York"


def _cik10(cik: int | str) -> str:
    return str(int(cik)).zfill(10)


def fetch_filing_index(cik: int | str, forms: tuple[str, ...] = ("10-K", "10-Q", "8-K")) -> pd.DataFrame:
    """Filing-event frame for one company: one row per accepted filing.

    event_ts = acceptance moment; asof_ts = same moment (information IS the
    event — earnings/8-K event studies anchor on this, Bernard-Thomas 1989).
    """
    raw = json.loads(http_get_text(f"https://data.sec.gov/submissions/CIK{_cik10(cik)}.json"))
    recent = raw["filings"]["recent"]
    df = pd.DataFrame(
        {
            "form": recent["form"],
            "accession": recent["accessionNumber"],
            "acceptance": recent["acceptanceDateTime"],
            "report_date": recent["reportDate"],
        }
    )
    df = df[df["form"].isin(forms)].copy()
    acc = pd.to_datetime(df["acceptance"], utc=True)
    df[EVENT_COL] = acc
    df[ASOF_COL] = acc
    df["cik"] = _cik10(cik)
    df = df[["cik", "form", "accession", "report_date", EVENT_COL, ASOF_COL]]
    return validate_pit(df, required=("cik", "form"))


def fetch_company_facts(
    cik: int | str,
    concepts: tuple[str, ...] = (
        "Revenues",
        "NetIncomeLoss",
        "StockholdersEquity",
        "Assets",
        "OperatingIncomeLoss",
        "GrossProfit",
    ),
    taxonomy: str = "us-gaap",
    sleep_s: float = 0.12,
) -> pd.DataFrame:
    """XBRL fundamentals, point-in-time.

    One row per (concept, fiscal period, filing): event_ts = fiscal period
    end (what the number is ABOUT), asof_ts = the filing date it became
    public in (what the market KNEW and when). Downstream features must join
    on asof_ts — quality/profitability (Novy-Marx 2013) built on period-end
    dates is falsified-by-construction.
    """
    raw = json.loads(
        http_get_text(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{_cik10(cik)}.json")
    )
    time.sleep(sleep_s)
    rows: list[dict] = []
    facts = raw.get("facts", {}).get(taxonomy, {})
    for concept in concepts:
        for unit, items in facts.get(concept, {}).get("units", {}).items():
            for it in items:
                if "end" not in it or "filed" not in it or it.get("val") is None:
                    continue
                rows.append(
                    {
                        "cik": _cik10(cik),
                        "concept": concept,
                        "unit": unit,
                        "value": float(it["val"]),
                        "fy": it.get("fy"),
                        "fp": it.get("fp"),
                        "form": it.get("form"),
                        "period_end": it["end"],
                        "filed": it["filed"],
                    }
                )
    if not rows:
        raise ValueError(f"No XBRL facts for CIK {cik} / {concepts}")
    df = pd.DataFrame(rows)
    df[EVENT_COL] = pd.to_datetime(df["period_end"], utc=True)
    # 'filed' is a date; filings accepted during day D are tradeable D+1 open
    # at the earliest -> conservative availability = end of filing day UTC.
    df[ASOF_COL] = pd.to_datetime(df["filed"], utc=True) + pd.Timedelta(days=1)
    df = df.drop(columns=["period_end", "filed"])
    return validate_pit(df, required=("cik", "concept", "value"))
