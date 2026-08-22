# scripts/w23ab_eval.py
"""Wave 23a/b evaluation — PEAD + quality on EDGAR PIT data (network-free).

  python3 scripts/w23ab_eval.py quality | pead
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tradebot.data.equity_universe import load_price_panel  # noqa: E402
from tradebot.registry import HypothesisLedger, LedgerEntry  # noqa: E402
from tradebot.risk import factor_residual_alpha  # noqa: E402

START = "2010-01-01"  # XBRL fundamentals reliable ~2009+; documented in wave log


def main() -> int:
    which = sys.argv[1]
    panel, cal = load_price_panel(root=ROOT / "market_data_parquet")

    if which == "quality":
        from tradebot.alpha import eq_quality as mod

        fund = pd.read_parquet(ROOT / "market_data_parquet/equities/fundamentals.parquet")
        ev = mod.gpa_events(fund)
        print(f"GP/A events: {len(ev)} over {ev['symbol'].nunique()} tickers")
        res = mod.run(panel, fund, membership=cal)
    else:
        from tradebot.alpha import eq_pead as mod

        fil = pd.read_parquet(ROOT / "market_data_parquet/equities/filings.parquet")
        ev = mod.announcement_events(panel, fil)
        print(f"announcement events: {len(ev)} over {ev['symbol'].nunique()} tickers")
        res = mod.run(panel, fil, membership=cal)

    s = res.summary(start=START)
    print(json.dumps({k: v for k, v in s.items() if k != "config"},
                     indent=1, default=str))

    facs = pd.read_parquet(ROOT / "market_data_parquet/equities/factors_daily.parquet")
    facs.index = pd.to_datetime(facs["event_ts"], utc=True)
    F = facs[["Mkt-RF", "SMB", "HML", "RMW", "CMA", "MOM"]]
    g4 = factor_residual_alpha(res.net_returns.loc[START:], F,
                               unit=res.unit, market="book", periods_per_year=252)
    print(g4.gate_row())
    print("   " + "  ".join(f"{k}={v:+.3f}(t={g4.loading_tstats[k]:+.1f})"
                            for k, v in g4.loadings.items()))

    ledger = HypothesisLedger(ROOT / "artefacts/governance/hypothesis_ledger.json")
    tot = ledger.append(LedgerEntry.from_config(
        wave=23, unit=res.unit, market="equities",
        config={**res.config, "eval_start": START},
        metrics={"net_sharpe": s["net_sharpe"], "net_cagr": s["net_cagr"],
                 "g4_alpha_ann": g4.alpha_ann, "g4_t": g4.t_alpha, "g4_p": g4.p_alpha},
        notes=mod.PRIOR))
    print("ledger total:", tot)
    return 0


if __name__ == "__main__":
    sys.exit(main())
