"""run_adaptive_wf.py — CLI for the adaptive walk-forward book (R-6, stateless).

Loads the free OHLCV panel (+ funding), runs the AFML overfit-proof adaptive
walk-forward engine, prints the forward report (Sharpe / DSR / PBO / per-year /
per-quarter), and writes metrics + the forward equity curve.

Usage:
  python apps/run_adaptive_wf.py [--halflife 365] [--refit 91] [--start 2023-01-01]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
from tradebot.alpha.adaptive_wf import AdaptiveWalkForward, AdaptiveWFConfig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default=str(_ROOT / "artefacts" / "broad_perp_ohlcv.parquet"))
    ap.add_argument("--funding", default=str(_ROOT / "artefacts" / "funding_universe.parquet"))
    ap.add_argument("--halflife", type=float, default=365.0)
    ap.add_argument("--refit", type=int, default=91)
    ap.add_argument("--start", default="2023-01-01")
    ap.add_argument("--out", default=str(_ROOT / "reports" / "adaptive_wf_metrics.json"))
    a = ap.parse_args()

    ohlcv = pd.read_parquet(a.panel)
    funding = pd.read_parquet(a.funding) if Path(a.funding).exists() else None
    cfg = AdaptiveWFConfig(recency_halflife_days=a.halflife, refit_days=a.refit, wf_start=a.start)

    eng = AdaptiveWalkForward(cfg).build_panel(ohlcv, funding)
    P = eng.build_features_labels()
    P = eng.run(P)
    # cache OOS score panel so the PBO grid can be swept offline (conclusive S>=50)
    P.loc[P["__p"].notna(), ["__date", "__sym", "__p"]].to_parquet(
        _ROOT / "artefacts" / "adaptive_wf_scores.parquet")
    r = eng.evaluate(P)

    print(f"\n=== ADAPTIVE WALK-FORWARD BOOK (refit {a.refit}d, recency HL {a.halflife:.0f}d) ===")
    print(f"  refits={r['n_refits']}  frozen-features={r['n_features']}")
    print(f"  forward Sharpe = {r['forward_sharpe']:.2f}   vol = {r['forward_vol']:.0%}   "
          f"MaxDD = {r['max_drawdown']*100:.0f}%")
    print(f"  Deflated Sharpe (n_trials={r['n_trials']}) = {r['deflated_sharpe']:.3f}   "
          f"PBO = {r['pbo']:.3f}")
    # MEASUREMENT_CONTRACT.md §6: waar V[{SR_m}] vandaan komt, hoort naast de DSR
    # te staan en niet alleen in de JSON.
    print(f"  DSR V[SR_m] = {r['dsr_sr_variance']:.3e} ({r['dsr_approximation']})   "
          f"n_obs = {r['n_obs']}  bars_per_year = {r['bars_per_year']:.0f}  "
          f"t_years = {r['t_years']:.3f}")
    print(f"  last 12m: return = {r['last12m_return']*100:+.0f}%   Sharpe = {r['last12m_sharpe']:.2f}")
    print("  forward per-year: " + " ".join(f"{y}:{v*100:+.0f}%" for y, v in r["per_year"].items()))
    print("  recent quarters:  " + " ".join(f"{d}:{v*100:+.0f}%" for d, v in list(r["per_quarter"].items())[-8:]))

    pnl = r.pop("pnl")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({k: v for k, v in r.items() if k != "pnl"}, open(a.out, "w"), indent=2, default=str)
    pnl.to_frame("pnl").to_parquet(_ROOT / "artefacts" / "adaptive_wf_pnl.parquet")
    print(f"  saved -> {a.out}")


if __name__ == "__main__":
    main()
