# scripts/w22_g4_diagnostic.py
"""W22 archiving evidence: HAC residual alpha of the three equity units vs
self-built panel factors (MKT = EW universe return, UMD = 12-1 decile spread).

Self-built factors suffice for ARCHIVING diagnostics (proves what the unit
loads on); the formal Ken-French G4 (mandate §10) is required only when a
unit is ACCEPTED. Network-free by design (sandbox egress is blocked).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tradebot.alpha import eq_lowvol, eq_strev, eq_xsmom  # noqa: E402
from tradebot.alpha.xs_unit import decile_weights  # noqa: E402
from tradebot.data.equity_universe import load_price_panel  # noqa: E402
from tradebot.risk import factor_residual_alpha  # noqa: E402

START = "2000-01-01"


def main() -> int:
    panel, cal = load_price_panel(root=ROOT / "market_data_parquet")
    rets = panel.pct_change(fill_method=None)
    mkt = rets.mean(axis=1).rename("MKT")

    # self-built UMD: gross 12-1 decile spread, monthly weights, no costs
    mom_sig = eq_xsmom.signal_panel(panel)
    mask = cal.reindex(index=mom_sig.index, columns=mom_sig.columns).fillna(False)
    mom_sig = mom_sig.where(mask)
    naive = mom_sig.index.tz_convert(None)
    rb = pd.DatetimeIndex(mom_sig.index.to_series().groupby(naive.to_period("M")).max())
    w = pd.DataFrame({t: decile_weights(mom_sig.loc[t]) for t in rb}).T
    w.index = rb
    w = w.reindex(mom_sig.index).ffill().fillna(0.0)
    umd = (w.shift(1) * rets).sum(axis=1).rename("UMD")

    facs = pd.concat([mkt, umd], axis=1).loc[START:]

    for mod in (eq_xsmom, eq_strev, eq_lowvol):
        res = mod.run(panel, membership=cal)
        g4 = factor_residual_alpha(
            res.net_returns.loc[START:], facs, unit=res.unit,
            market="book", periods_per_year=252,
        )
        print(g4.gate_row())
        print("   loadings: " + "  ".join(
            f"{k}={v:+.3f} (t={g4.loading_tstats[k]:+.1f})"
            for k, v in g4.loadings.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
