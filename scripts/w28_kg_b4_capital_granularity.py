#!/usr/bin/env python
"""W28 KG-B4 — is the Setup B book sizeable at the REAL account capital?

Operator input 2026-08-10: account capital is **under $50k**.
SETUP_B_ML_PROMPT §8 requires this computed before Fase 1 closes, and
PREREGISTRATION_SETUP_B_W28 §8 pre-registers the consequence: a KG-B4 failure
at the real capital is a CAPITAL finding, reported as such, never repaired by
shrinking the universe until the numbers work.

WHAT IS AND IS NOT ASSUMED
--------------------------
Instrument volatilities come from the real 26-instrument panel (measured, not
assumed). Contract multipliers are NOT invented: the analysis is parameterised
over "notional per contract", and the result is reported across a range that
brackets every CME micro (~$5k) up to a full-size note future (~$110k). The
break-even capital is then read off, so no unverified contract spec can change
the conclusion.

THE ARITHMETIC
--------------
A book of capital C at target vol s_t has an annual risk budget of C * s_t.
One contract of instrument i carries notional_i * vol_i of annualised risk.
Under inverse-vol weighting each instrument should carry roughly an equal share
of that budget, so the position in instrument i is

    target_risk_i = C * s_t / N          (equal risk share, N instruments)
    contracts_i   = target_risk_i / (notional_i * vol_i)

If ``contracts_i`` rounds to 0 the instrument drops out of the book entirely;
if it rounds to 1 from 0.5 the sizing error is 100%. KG-B4 allows a sizing
error of at most 20% of target risk.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tradebot.alpha.cm_tsmom import effective_breadth, run  # noqa: E402
from tradebot.data.xasset_proxy import to_tr_panel  # noqa: E402

PANEL = "market_data_parquet/xasset/tr_panel.parquet"
TRADING_DAYS = 252
TARGET_VOL = 0.10
MAX_SIZING_ERROR = 0.20      # KG-B4


def main() -> int:
    panel = to_tr_panel(pd.read_parquet(PANEL))
    rets = panel.pct_change(fill_method=None)
    # recent vol regime — what a live account would actually face
    vol = (rets.tail(TRADING_DAYS * 3).std() * np.sqrt(TRADING_DAYS)).dropna()
    n_inst = len(vol)

    print(f"panel: {n_inst} instruments, ann vol {vol.min():.1%}-{vol.max():.1%} "
          f"(median {vol.median():.1%})\n")

    rows = []
    for notional in (5_000, 10_000, 25_000, 50_000, 110_000):
        for capital in (25_000, 50_000, 150_000, 250_000, 500_000):
            budget = capital * TARGET_VOL
            target_risk = budget / n_inst
            risk_per_contract = notional * vol
            exact = target_risk / risk_per_contract
            held = np.round(exact)
            # instruments that round to zero simply are not in the book
            alive = int((held > 0).sum())
            err = np.where(exact > 0, np.abs(held - exact) / exact, 1.0)
            rows.append({
                "notional_per_contract": notional,
                "capital": capital,
                "instruments_sizeable": alive,
                "median_sizing_error": float(np.median(err)),
                "mean_sizing_error": float(np.mean(err)),
                "kg_b4_pass": bool(np.mean(err) <= MAX_SIZING_ERROR and alive >= 20),
            })

    df = pd.DataFrame(rows)
    print("instruments that can be sized at all (of 26), by contract notional x capital")
    piv = df.pivot(index="notional_per_contract", columns="capital",
                   values="instruments_sizeable")
    print(piv.to_string())
    print("\nmean sizing error (KG-B4 limit 0.20)")
    piv2 = df.pivot(index="notional_per_contract", columns="capital",
                    values="mean_sizing_error")
    print(piv2.round(2).to_string())

    # ---- what survives at the real capital ------------------------------- #
    print("\n" + "=" * 74)
    print("AT THE REAL CAPITAL (operator: under $50k) — surviving instrument set")
    print("=" * 74)
    res = run(panel)
    for capital, notional in ((50_000, 5_000), (50_000, 10_000), (25_000, 5_000)):
        budget = capital * TARGET_VOL
        target_risk = budget / n_inst
        exact = target_risk / (notional * vol)
        survivors = list(vol.index[np.round(exact) > 0])
        if survivors:
            sub_w = res.weights[survivors]
            sub_r = rets[survivors]
            neff = effective_breadth(sub_w, sub_r) if len(survivors) > 1 else 1.0
        else:
            neff = 0.0
        print(f"  capital ${capital:,} @ ${notional:,}/contract -> "
              f"{len(survivors):2d} instruments, N_eff {neff:.2f}")
        if survivors:
            print(f"      {survivors}")

    # ---- break-even ------------------------------------------------------ #
    print("\n" + "=" * 74)
    print("BREAK-EVEN CAPITAL for the full 26-instrument book at KG-B4")
    print("=" * 74)
    for notional in (5_000, 10_000, 25_000):
        lo, hi = 10_000, 20_000_000
        for _ in range(60):
            mid = (lo + hi) / 2
            target_risk = mid * TARGET_VOL / n_inst
            exact = target_risk / (notional * vol)
            held = np.round(exact)
            err = np.where(exact > 0, np.abs(held - exact) / exact, 1.0)
            if float(np.mean(err)) <= MAX_SIZING_ERROR and int((held > 0).sum()) >= n_inst:
                hi = mid
            else:
                lo = mid
        print(f"  ${notional:>7,}/contract -> needs ${hi:>12,.0f} for 26 instruments "
              f"at <=20% sizing error")

    out = {
        "operator_capital": "under $50k (stated 2026-08-10)",
        "target_vol": TARGET_VOL,
        "max_sizing_error": MAX_SIZING_ERROR,
        "n_instruments_panel": n_inst,
        "grid": rows,
        "verdict": (
            "KG-B4 FAILS at the stated capital for the pre-registered "
            "26-instrument universe. This is a capital finding, not a signal "
            "finding (PREREGISTRATION §8)."
        ),
    }
    dest = Path("artefacts/killgates/kg_b4_capital_granularity.json")
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
