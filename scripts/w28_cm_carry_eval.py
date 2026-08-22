#!/usr/bin/env python
"""Wave 28 — full evaluation of cm_carry_energy (EIA term-structure carry).

Produces the complete kill-gate table, the three nested G4 specifications, the
roll-calendar validation, the return-construction cross-check against
roll-inclusive ETFs, and the kill-gate artefact that
``pytest tests/killgates -m killgate`` reads.

    python scripts/w28_cm_carry_eval.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tradebot.alpha import cm_carry
from tradebot.alpha.xs_unit import CostModel, ann_sharpe
from tradebot.backtest.dd_shape import max_dd_over_vol_quantile
from tradebot.data.sources.kenfrench import fetch_factors_daily
from tradebot.risk.factor_alpha import factor_residual_alpha

TS_PANEL = "market_data_parquet/commodities/eia_term_structure.parquet"
XASSET = "market_data_parquet/xasset/tr_panel.parquet"
_KF = ("Mkt-RF", "SMB", "HML", "RMW", "CMA", "MOM")
_TRADING_DAYS = 252
SEED = 42


def _bootstrap_p_positive(net: pd.Series, n_boot: int = 2000, block: int = 21) -> dict:
    """Stationary block bootstrap of the annualised Sharpe (Politis-Romano)."""
    rng = np.random.default_rng(SEED)
    x = net.dropna().to_numpy()
    n = len(x)
    n_blocks = int(np.ceil(n / block))
    sharpes = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel() % n
        s = x[idx[:n]]
        sd = s.std(ddof=1)
        sharpes[b] = (s.mean() / sd * np.sqrt(_TRADING_DAYS)) if sd > 0 else 0.0
    return {
        "bootstrap_p_positive": float((sharpes > 0).mean()),
        "bootstrap_p05": float(np.quantile(sharpes, 0.05)),
        "bootstrap_p95": float(np.quantile(sharpes, 0.95)),
    }


def _validate_construction(ts: pd.DataFrame) -> list[dict]:
    """Constructed returns vs roll-inclusive ETF NAVs — the G8 honesty check."""
    xa = pd.read_parquet(XASSET)
    w = xa.pivot(index="event_ts", columns="symbol", values="close")
    w.index = pd.to_datetime(w.index, utc=True)
    etf = w.sort_index().pct_change(fill_method=None)

    rows = []
    for prod, sym in (("WTI", "USO"), ("NG", "UNG")):
        slots = cm_carry.build_slot_panel(ts, prod)
        mine = cm_carry.contract_consistent_returns(slots, prod, held_slot=2).dropna()
        j = pd.concat([mine.rename("eia"), etf[sym].rename("etf")], axis=1,
                      join="inner").dropna()
        j = j[j.index >= "2007-01-01"]
        cagr = lambda s: float((1 + s).prod() ** (_TRADING_DAYS / len(s)) - 1)  # noqa: E731
        rows.append({
            "product": prod, "etf": sym, "n": len(j),
            "corr": float(j["eia"].corr(j["etf"])),
            "cagr_constructed": cagr(j["eia"]), "cagr_etf": cagr(j["etf"]),
            "cagr_diff": cagr(j["eia"]) - cagr(j["etf"]),
        })
    return rows


def main() -> int:
    ts = pd.read_parquet(TS_PANEL)

    print("=" * 78)
    print("ROLL CALENDAR VALIDATION (calendar rule vs what prices did)")
    print("=" * 78)
    roll_val = [cm_carry.validate_roll_calendar(cm_carry.build_slot_panel(ts, p), p)
                for p in ("WTI", "NG", "HO", "RBOB")]
    for r in roll_val:
        print(f"  {r['product']:5s} rolls/yr {r['rolls_per_year']:5.2f}  "
              f"median day-of-month {r['median_day_of_month']:2d}  "
              f"shift-evidence AUC (steep decile) {r['shift_evidence_auc_steep_decile']:.3f}")

    print("\n" + "=" * 78)
    print("RETURN CONSTRUCTION vs ROLL-INCLUSIVE ETFs (G8)")
    print("=" * 78)
    constr = _validate_construction(ts)
    for c in constr:
        print(f"  {c['product']}/{c['etf']}  n={c['n']}  corr={c['corr']:.4f}  "
              f"constructed {c['cagr_constructed']:+.2%}/yr vs ETF "
              f"{c['cagr_etf']:+.2%}/yr  diff {c['cagr_diff']:+.2%}/yr")
    print("  reference: free continuous futures overstate by +7.0%/yr (WTI), "
          "+25.1%/yr (NG)")

    res = cm_carry.run(ts)
    s = res.summary()
    net = res.net_returns.dropna()

    # cost stress +50%
    stressed = cm_carry.run(ts, cost=CostModel(1.5, 4.5, 0.0))
    s_stress = ann_sharpe(stressed.net_returns.dropna())

    # IS/OOS split at the midpoint of the sample
    cut = net.index[len(net) // 2]
    is_s, oos_s = ann_sharpe(net.loc[:cut]), ann_sharpe(net.loc[cut:])
    decay = (is_s - oos_s) / abs(is_s) if is_s else float("nan")

    boot = _bootstrap_p_positive(net)
    n_eff = cm_carry.effective_breadth(res.weights, res.instrument_returns)

    print("\n" + "=" * 78)
    print("UNIT RESULT")
    print("=" * 78)
    for k, v in s.items():
        if k != "config":
            print(f"  {k:24s} {v}")
    print(f"  {'net_sharpe_cost_+50%':24s} {s_stress:.4f}")
    print(f"  {'is_sharpe / oos_sharpe':24s} {is_s:.4f} / {oos_s:.4f}  decay {decay:+.1%}")
    print(f"  {'bootstrap P(S>0)':24s} {boot['bootstrap_p_positive']:.4f} "
          f"(5th pct {boot['bootstrap_p05']:+.3f})")
    print(f"  {'N / N_eff':24s} {len(res.instrument_returns.columns)} / {n_eff:.2f}")

    # ---- G4: three nested specifications --------------------------------- #
    print("\n" + "=" * 78)
    print("G4 — residual alpha, three nested specifications (S3 binds)")
    print("=" * 78)
    net_naive = net.copy()
    net_naive.index = pd.DatetimeIndex(net_naive.index).tz_convert(None).normalize()

    inst = res.instrument_returns.copy()
    inst.index = pd.DatetimeIndex(inst.index).tz_convert(None).normalize()
    passive = pd.DataFrame({"PASSIVE": inst.mean(axis=1)})
    for p in inst.columns:
        passive[f"PASV_{p}"] = inst[p]

    kf = fetch_factors_daily()
    kf = kf.set_index(pd.DatetimeIndex(kf["event_ts"]).tz_convert(None).normalize())
    kf = kf[[c for c in _KF if c in kf.columns]]

    specs = {
        "S1 KenFrench6": kf,
        "S2 +passive energy": kf.join(passive[["PASSIVE"]], how="inner"),
        "S3 +per-product passives": kf.join(passive, how="inner"),
    }
    g4 = {}
    for name, fac in specs.items():
        idx = net_naive.index.intersection(fac.index)
        r = factor_residual_alpha(net_naive.loc[idx], fac.loc[idx],
                                  unit=cm_carry.UNIT, market="book",
                                  periods_per_year=_TRADING_DAYS)
        g4[name] = {"alpha_ann": float(r.alpha_ann), "t": float(r.t_alpha),
                    "p": float(r.p_alpha), "n": len(idx)}
        print(f"  {name:26s} alpha {r.alpha_ann:+.2%}/yr  t={r.t_alpha:+.2f}  "
              f"p={r.p_alpha:.4f}  n={len(idx)}")

    # ---- kill gates ------------------------------------------------------ #
    shape_cap = max_dd_over_vol_quantile(
        s["net_sharpe"], s["sample_years"], 0.95, s["ann_vol"]
    )
    g4_strict = g4["S3 +per-product passives"]
    checks = [
        ("KG-B1 net_sharpe >= 0.40", s["net_sharpe"] >= 0.40, f"{s['net_sharpe']:.3f}"),
        ("KG-B1 years_positive >= 0.60", s["years_positive_frac"] >= 0.60,
         f"{s['years_positive_frac']:.3f}"),
        ("KG-B1 dd_over_vol <= q95 cap", s["dd_over_vol"] <= shape_cap,
         f"{s['dd_over_vol']:.3f} vs cap {shape_cap:.3f}"),
        ("KG-B2 G4-S3 p < 0.05", g4_strict["p"] < 0.05, f"p={g4_strict['p']:.4f}"),
        ("KG-B2 G4-S3 t >= 2.0", g4_strict["t"] >= 2.0, f"t={g4_strict['t']:+.2f}"),
        ("KG-B3 WF sharpe >= 0.30", oos_s >= 0.30, f"{oos_s:.3f}"),
        ("KG-B3 decay <= 40%", decay <= 0.40, f"{decay:+.1%}"),
        ("KG-B3 bootstrap P(S>0) >= 0.75",
         boot["bootstrap_p_positive"] >= 0.75, f"{boot['bootstrap_p_positive']:.3f}"),
        ("G-COST net > 0 @ +50% cost", s_stress > 0, f"{s_stress:.3f}"),
    ]
    print("\n" + "=" * 78)
    print("KILL GATES")
    print("=" * 78)
    for name, ok, val in checks:
        print(f"  {'PASS' if ok else 'FAIL':4s}  {name:34s} {val}")

    passed = all(ok for _, ok, _ in checks)
    verdict = "ACCEPTED" if passed else "ARCHIVED — gate not met"
    print(f"\n  VERDICT: {verdict}")

    artefact = {
        "unit": cm_carry.UNIT, "wave": 28, "evaluated_utc": "2026-08-10",
        "panel": f"{TS_PANEL} (EIA NYMEX Contract 1-4, FROZEN archive ends 2024-04-05)",
        "prior": cm_carry.PRIOR,
        "cost_model": "1.0bp commission + 3.0bp half-spread per side, no borrow (futures)",
        "gates_completed": ["KG-B1 in-sample", "KG-B2 residual alpha",
                            "KG-B3 out-of-sample"],
        "net_sharpe": s["net_sharpe"], "gross_sharpe": s["gross_sharpe"],
        "net_cagr": s["net_cagr"], "ann_vol": s["ann_vol"],
        "max_drawdown": s["max_drawdown"], "calmar": s["calmar"],
        "dd_over_vol": s["dd_over_vol"], "dd_shape_cap_q95": shape_cap,
        "years_positive_frac": s["years_positive_frac"],
        "sample_years": s["sample_years"], "n_bars": s["n_bars"],
        "ann_turnover": s["ann_turnover"],
        "net_sharpe_cost_stress_50pct": s_stress,
        "alpha_p": g4_strict["p"], "alpha_t_strict": g4_strict["t"],
        "alpha_ann_strict": g4_strict["alpha_ann"],
        "g4_specs": g4,
        "wf_sharpe": oos_s, "is_sharpe": is_s, "sharpe_decay": decay,
        **boot,
        "abs_rho_vs_book": None,
        "n_instruments": len(res.instrument_returns.columns), "n_eff": n_eff,
        "roll_calendar_validation": roll_val,
        "return_construction_validation": constr,
        "verdict": verdict,
        "failed": [f"{n}: {v}" for n, ok, v in checks if not ok],
        "passed": [f"{n}: {v}" for n, ok, v in checks if ok],
        "notes": (
            "abs_rho_vs_book is null: no combined_book returns exist in this "
            "working copy, so KG-B3's correlation criterion is NOT RUN rather "
            "than assumed. Data caveat: the EIA archive ends 2024-04-05, so "
            "this unit has no recent OOS window and cannot be run live from "
            "this source."
        ),
    }
    dest = Path("artefacts/killgates/cm_carry.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(artefact, indent=2), encoding="utf-8")
    print(f"  wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
