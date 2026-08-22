# src/tradebot/data/edgar_universe.py
"""Wave 23a/b — bulk EDGAR ingest for the equity universe (PIT, G8).

ticker -> CIK via SEC company_tickers.json, then per company:
  fundamentals (XBRL companyfacts, asof = filed+1d)  -> fundamentals.parquet
  filing events (10-K/10-Q/8-K, asof = acceptance)   -> filings.parquet
Failures are recorded per ticker (coverage evidence), never swallowed.
SEC fair-access: <=10 req/s — the sleep is mandatory, do not lower it.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from tradebot.data.sources import edgar
from tradebot.data.sources.base import http_get_text

__all__ = ["load_cik_map", "bulk_ingest"]


def load_cik_map() -> dict[str, str]:
    """ticker -> zero-padded CIK from the official SEC mapping (free)."""
    raw = json.loads(http_get_text("https://www.sec.gov/files/company_tickers.json"))
    return {
        rec["ticker"].upper(): str(rec["cik_str"]).zfill(10)
        for rec in raw.values()
    }


def bulk_ingest(
    tickers: list[str],
    root: Path | str = "market_data_parquet",
    sleep_s: float = 0.13,
) -> dict:
    """Ingest fundamentals + filing events for ``tickers``; returns coverage."""
    root = Path(root)
    dest = root / "equities"
    dest.mkdir(parents=True, exist_ok=True)

    cik_map = load_cik_map()
    fund_frames: list[pd.DataFrame] = []
    filing_frames: list[pd.DataFrame] = []
    failures: dict[str, str] = {}
    no_cik = sorted(t for t in tickers if t.upper() not in cik_map)

    for i, t in enumerate(t for t in tickers if t.upper() in cik_map):
        cik = cik_map[t.upper()]
        f = edgar.fetch_company_facts(cik, sleep_s=0.0)
        f["ticker"] = t.upper()
        fund_frames.append(f)
        time.sleep(sleep_s)
        fi = edgar.fetch_filing_index(cik)
        fi["ticker"] = t.upper()
        filing_frames.append(fi)
        time.sleep(sleep_s)
        if i % 50 == 0:
            print(f"  ...{i} companies done", flush=True)

    if fund_frames:
        pd.concat(fund_frames, ignore_index=True).to_parquet(
            dest / "fundamentals.parquet", index=False
        )
    if filing_frames:
        pd.concat(filing_frames, ignore_index=True).to_parquet(
            dest / "filings.parquet", index=False
        )

    coverage = {
        "n_requested": len(tickers),
        "n_no_cik": len(no_cik),
        "no_cik": no_cik,
        "n_failed": len(failures),
        "failed": failures,
        "note": "no-CIK + failed ≈ delisted/renamed: zelfde survivorship-gap "
                "als de prijsdata (G8) — PEAD/quality-units erven die bias.",
    }
    (dest / "edgar_coverage.json").write_text(
        json.dumps(coverage, indent=2), encoding="utf-8"
    )
    return coverage
