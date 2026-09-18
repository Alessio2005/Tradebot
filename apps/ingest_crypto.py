"""apps/ingest_crypto.py — dunne CLI rond de PIT crypto-ingestion (deliverable 15).

Respecteert R-6 (<= 80 LOC). Alle logica zit in `tradebot.data.ingestion`.

    python apps/ingest_crypto.py --kind ohlcv --granularity 1d
    python apps/ingest_crypto.py --kind funding --symbols BTCUSDT ETHUSDT
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from tradebot.data.ingestion import BybitV5Client, run_ingestion
from tradebot.data.ingestion.contract import IngestionSource, IngestionSpec
from tradebot.data.ingestion.crypto_sources import (
    FundingSource,
    OhlcvSource,
    OpenInterestSource,
)
from tradebot.data.pit_store import PitStore
from tradebot.data.validation import (
    FUNDING_SPEC,
    OHLCV_SPEC,
    OPEN_INTEREST_SPEC,
    GapLedger,
)
from tradebot.schemas.config import DataConfig, load_config

ROOT = Path(__file__).resolve().parents[1]
_SPECS = {"ohlcv": OHLCV_SPEC, "funding": FUNDING_SPEC,
          "open_interest": OPEN_INTEREST_SPEC}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kind", choices=sorted(_SPECS), default="ohlcv")
    ap.add_argument("--granularity", default="1d")
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--conf", default=str(ROOT / "conf" / "data" / "default.yaml"))
    ap.add_argument("--gap-policy", default=None,
                    help="overschrijft conf; 'register' legt gaten vast i.p.v. crashen")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    cfg = load_config(args.conf, DataConfig)
    symbols = args.symbols or list(cfg.symbols)
    store = PitStore(ROOT / cfg.pit_store_root)
    ledger = GapLedger(ROOT / "artefacts" / "governance" / "gap_ledger.jsonl")
    client = BybitV5Client(category="linear")
    end_ms = int(pd.Timestamp.now(tz="UTC").normalize().value // 10**6)

    rc = 0
    for sym in symbols:
        start_ms = client.instrument_launch_ms(sym)
        spec = IngestionSpec(
            asset_class="crypto", dataset=args.kind, symbol=sym,
            granularity=args.granularity,
            series_spec=_SPECS[args.kind],
            gap_policy=args.gap_policy or cfg.gap_policy,
            max_abs_log_return=cfg.max_abs_log_return,
            allow_price_jumps=cfg.allow_price_jumps,
            source_name=f"bybit-v5/{args.kind}",
            extra_meta={"venue": cfg.venue, "kind": args.kind,
                        "launch_ms": start_ms},
        )
        src: IngestionSource
        if args.kind == "ohlcv":
            src = OhlcvSource(client, start_ms, end_ms)
        elif args.kind == "funding":
            src = FundingSource(client, start_ms, end_ms, cfg.funding_interval_hours)
        else:
            src = OpenInterestSource(client, start_ms, end_ms)

        res = run_ingestion(src, spec, store, ledger)
        print(f"{sym:10s} {args.kind:14s} {res.n_rows:7,} rijen  "
              f"{res.first_iso[:10]}..{res.last_iso[:10]}  "
              f"gaps={res.n_gaps:<3} hash={res.data_hash}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
