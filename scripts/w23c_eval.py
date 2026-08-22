# scripts/w23c_eval.py
"""Wave 23c evaluation — eq_overnight (LPS 2019) on the W21 PIT panel.

Prints summary (2000+), G4 vs self-built MKT+UMD, correlation vs the W22
units, and stages the ledger entry. Network-free.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tradebot.alpha import eq_lowvol, eq_overnight, eq_strev, eq_xsmom  # noqa: E402
from tradebot.data.equity_universe import load_price_panel  # noqa: E402
from tradebot.registry import HypothesisLedger, LedgerEntry  # noqa: E402
from tradebot.risk import factor_residual_alpha  # noqa: E402

START = "2000-01-01"
STAGING = ROOT / "artefacts/governance/hypothesis_ledger_staging_w23c.json"


def main() -> int:
    close, cal, open_ = load_price_panel(
        root=ROOT / "market_data_parquet", with_open=True
    )
    res = eq_overnight.run(close, open_, membership=cal)
    s = res.summary(start=START)
    print(f"== {res.unit} ({eq_overnight.PRIOR}) ==")
    print(json.dumps({k: v for k, v in s.items() if k != "config"},
                     indent=2, default=str))

    rets = close.pct_change(fill_method=None)
    mkt = rets.mean(axis=1).rename("MKT")
    # UMD: reuse the archived xsmom unit's NET-free gross construction
    umd = eq_xsmom.run(close, membership=cal).gross_returns.rename("UMD")
    facs = pd.concat([mkt, umd], axis=1).loc[START:]
    g4 = factor_residual_alpha(
        res.net_returns.loc[START:], facs, unit=res.unit,
        market="book", periods_per_year=252,
    )
    print(g4.gate_row())
    print("   loadings: " + "  ".join(
        f"{k}={v:+.3f} (t={g4.loading_tstats[k]:+.1f})"
        for k, v in g4.loadings.items()))

    others = {
        m.UNIT: m.run(close, membership=cal).net_returns.loc[START:]
        for m in (eq_xsmom, eq_strev, eq_lowvol)
    }
    corr = pd.DataFrame({**others, res.unit: res.net_returns.loc[START:]}).corr()
    print("\ncorr vs W22 units:\n", corr[res.unit].round(3))

    HypothesisLedger.append_to_staging(STAGING, LedgerEntry.from_config(
        wave=23, unit=res.unit, market="equities",
        config={**res.config, "eval_start": START},
        metrics={"net_sharpe": s["net_sharpe"], "net_cagr": s["net_cagr"],
                 "g4_t_alpha": g4.t_alpha, "g4_alpha_ann": g4.alpha_ann},
        notes=eq_overnight.PRIOR,
    ))
    print(f"\nstaged -> {STAGING.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
