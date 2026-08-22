#!/usr/bin/env python
"""Wave 27 — G4 residual-alpha test for cm_tsmom_xasset (Mandate §10 G4).

Three nested specifications, reported together (the g4-self-factor rule:
never report only the flattering one):

  S1  Ken French 6 (Mkt-RF, SMB, HML, RMW, CMA, MOM)      — equity factors
  S2  S1 + PASSIVE  (equal-weight long-only of the SAME panel)
  S3  S2 + per-sector passives (equity / rates / commodity long-only)

S3 is the honest test: if a TSMOM book's alpha survives controls for simply
being long the same instruments, it is timing, not beta. A 2004-2026 sample
spans a historic bond bull market — without S2/S3 a "long bonds most of the
time" book would masquerade as alpha.

    python scripts/w27_g4.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from tradebot.alpha.cm_tsmom import COST, UNIT, run
from tradebot.data.sources.kenfrench import fetch_factors_daily
from tradebot.data.xasset_proxy import sector_of, to_tr_panel
from tradebot.risk.factor_alpha import factor_residual_alpha

PANEL = "market_data_parquet/xasset/tr_panel.parquet"
_KF = ("Mkt-RF", "SMB", "HML", "RMW", "CMA", "MOM")


def main() -> int:
    panel = to_tr_panel(pd.read_parquet(PANEL))
    res = run(panel, cost=COST)
    net = res.net_returns.dropna()
    net.index = pd.DatetimeIndex(net.index).tz_convert(None).normalize()

    rets = panel.pct_change(fill_method=None)
    rets.index = pd.DatetimeIndex(rets.index).tz_convert(None).normalize()

    # passive controls: what you get for simply being long the same things
    passive = pd.DataFrame({"PASSIVE": rets.mean(axis=1)})
    for sec in ("equity", "rates", "commodity"):
        cols = [c for c in rets.columns if sector_of(c) == sec]
        passive[f"PASV_{sec.upper()}"] = rets[cols].mean(axis=1)

    kf = fetch_factors_daily()
    kf = kf.set_index(pd.DatetimeIndex(kf["event_ts"]).tz_convert(None).normalize())
    kf = kf[[c for c in _KF if c in kf.columns]]
    missing = set(_KF) - set(kf.columns)
    if missing:
        raise SystemExit(f"Ken French factors missing: {sorted(missing)}")

    specs = {
        "S1 KenFrench6": kf,
        "S2 +passive": kf.join(passive[["PASSIVE"]], how="inner"),
        "S3 +sector passives": kf.join(passive, how="inner"),
    }

    print(f"unit={UNIT}  n_net={len(net)}  "
          f"{net.index.min().date()} -> {net.index.max().date()}\n")
    for name, fac in specs.items():
        idx = net.index.intersection(fac.index)
        r = factor_residual_alpha(
            net.loc[idx], fac.loc[idx], unit=UNIT, market="book",
            periods_per_year=252,
        )
        print(f"--- {name} ---")
        print(" ", r.gate_row())
        loads = "  ".join(
            f"{k}={v:+.3f}(t={r.loading_tstats[k]:+.1f})"
            for k, v in r.loadings.items()
        )
        print("   loadings:", loads, "\n")

    # correlation of the unit with plain long-only exposure
    idx = net.index.intersection(passive.index)
    print("correlation with passive long-only panel: "
          f"{net.loc[idx].corr(passive['PASSIVE'].loc[idx]):+.3f}")
    print("correlation with passive rates:           "
          f"{net.loc[idx].corr(passive['PASV_RATES'].loc[idx]):+.3f}")
    print("correlation with passive equity:          "
          f"{net.loc[idx].corr(passive['PASV_EQUITY'].loc[idx]):+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
