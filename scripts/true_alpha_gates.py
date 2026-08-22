"""true_alpha_gates.py — CHIEF /goal true-alpha gate battery (G4/G5/G6 + DSR).

Runs the EXISTING 3-sleeve causal-RP neutral book through the hard true-alpha
acceptance gates so we close the breakthrough's #1 caveat ("no purged CPCV /
no factor regression"):

  G4  rolling 90d |beta| to BTC AND to equal-weight basket  (< 0.10)
  G5  net Sharpe in EVERY regime bucket (bull/bear x hi/lo vol)  (> 0)
  G6  alpha after regressing out MKT, TSMOM, XSMOM, CARRY, VRP   (> 0, p<0.05)
      with Newey-West (HAC) standard errors.
  DSR corrected (metrics.deflated_sharpe, dimensional fix 2026-06-08).

All factors are built causally from the broad perp panel; sleeve weights use
data <= t and execute t+1 (inherited from the sleeve builders).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

_spec = importlib.util.spec_from_file_location("cb", ROOT / "scripts/combined_neutral_book.py")
cb = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(cb)
_vspec = importlib.util.spec_from_file_location("vb", ROOT / "scripts/validate_neutral_book.py")
vb = importlib.util.module_from_spec(_vspec); _vspec.loader.exec_module(vb)

PANEL = pd.read_parquet(ROOT / "artefacts/broad_perp_daily_close.parquet").resample("1D").last()
RP = np.log(PANEL / PANEL.shift(1))
RP = RP[RP.notna().sum(axis=1) >= 20]
BTC = RP["BTCUSDT"] if "BTCUSDT" in RP.columns else RP.mean(axis=1)
BASKET = RP.mean(axis=1)


# ── factors (causal) ─────────────────────────────────────────────────────────
def f_tsmom(lookback=50):
    """Time-series momentum factor: cross-asset avg of sign(trailing ret)*next ret."""
    sig = np.sign(np.log(PANEL / PANEL.shift(lookback)))
    return (sig.shift(1) * RP).mean(axis=1).rename("TSMOM")


def f_xsmom(lookback=20):
    """Cross-sectional momentum: long top / short bottom by trailing return, $-neutral."""
    mom = np.log(PANEL / PANEL.shift(lookback))
    z = mom.sub(mom.mean(axis=1), axis=0).div(mom.std(axis=1).replace(0, np.nan), axis=0)
    w = z.sub(z.mean(axis=1), axis=0)
    w = w.div(w.abs().sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    return (w.shift(1) * RP).sum(axis=1).rename("XSMOM")


def f_carry():
    c = cb.carry_pnl(cost_bps=0.0)            # gross carry factor (no cost)
    return c.rename("CARRY")


def f_vrp():
    """Vol-risk-premium proxy: short BTC realized vol shock (implied if available)."""
    p = ROOT / "artefacts/dvol_btc.parquet"
    rv = BTC.rolling(20).std()
    if p.exists():
        d = pd.read_parquet(p)
        iv = d.iloc[:, 0].resample("1D").last().reindex(RP.index).ffill()
        return (iv / 100.0 / np.sqrt(365) - rv).shift(1).rename("VRP")
    return (-rv.diff()).shift(1).rename("VRP")    # fallback realized-vol carry


# ── gates ────────────────────────────────────────────────────────────────────
def gate_g4(combo):
    out = {}
    for name, fac in (("BTC", BTC), ("BASKET", BASKET)):
        x = fac.reindex(combo.index)
        cov = combo.rolling(90).cov(x)
        var = x.rolling(90).var()
        beta = (cov / var).dropna()
        out[name] = (beta.abs().max(), (beta.abs() < 0.10).mean())
    return out


def gate_g5(combo):
    trail = np.log(PANEL["BTCUSDT"] / PANEL["BTCUSDT"].shift(90)).reindex(combo.index)
    vol = BTC.rolling(30).std().reindex(combo.index)
    bull = trail > 0
    hivol = vol > vol.median()
    buckets = {
        "bull/hivol": bull & hivol, "bull/lovol": bull & ~hivol,
        "bear/hivol": ~bull & hivol, "bear/lovol": ~bull & ~hivol,
    }
    res = {}
    for k, m in buckets.items():
        seg = combo[m.fillna(False)]
        sh = seg.mean() / seg.std() * np.sqrt(365) if len(seg) > 5 and seg.std() > 0 else np.nan
        res[k] = (sh, len(seg))
    return res


def gate_g6(combo):
    facs = pd.concat([BTC.rename("MKT"), f_tsmom(), f_xsmom(), f_carry(), f_vrp()], axis=1)
    df = pd.concat([combo.rename("y"), facs], axis=1).dropna()
    X = sm.add_constant(df[["MKT", "TSMOM", "XSMOM", "CARRY", "VRP"]])
    m = sm.OLS(df["y"], X).fit(cov_type="HAC", cov_kwds={"maxlags": 10})
    return m


def main():
    c, s, l, _ = vb.sleeves(5.0)
    combo = vb.causal_rp(c, s, l).dropna()
    print(f"Book: n={len(combo)}  ann_Sharpe={vb.ann_sharpe(combo):+.2f}\n")

    print("=== G4: rolling 90d |beta| (<0.10) ===")
    for name, (mx, frac) in gate_g4(combo).items():
        flag = "PASS" if mx < 0.10 else ("MARGINAL" if frac > 0.9 else "FAIL")
        print(f"  vs {name:6s}: max|beta|={mx:.3f}  frac<0.10={frac*100:.0f}%  [{flag}]")

    print("\n=== G5: net Sharpe per regime bucket (all >0) ===")
    g5 = gate_g5(combo); neg = 0
    for k, (sh, n) in g5.items():
        neg += sh <= 0
        print(f"  {k:12s}: Sharpe={sh:+.2f}  (n={n})")
    print(f"  -> {'PASS' if neg == 0 else 'FAIL'} ({neg} non-positive buckets)")

    print("\n=== G6: alpha after factor regression (HAC), alpha>0 p<0.05 ===")
    m = gate_g6(combo)
    a, ta, pa = m.params["const"], m.tvalues["const"], m.pvalues["const"]
    ann_a = a * 365
    print(f"  alpha/day={a:.5f}  ann_alpha={ann_a*100:+.1f}%  t={ta:+.2f}  p={pa:.4f}  "
          f"[{'PASS' if (a > 0 and pa < 0.05) else 'FAIL'}]")
    print("  factor loadings (t-stat):")
    for f in ["MKT", "TSMOM", "XSMOM", "CARRY", "VRP"]:
        print(f"    {f:6s}: beta={m.params[f]:+.4f}  t={m.tvalues[f]:+.2f}")
    print(f"  R^2={m.rsquared:.3f}")

    print("\n=== DSR (corrected) ===")
    from tradebot.backtest.metrics import deflated_sharpe
    sr_d = combo.mean() / combo.std()
    for nt in (12, 50, 200):
        print(f"  DSR(n_trials={nt:3d}) = {deflated_sharpe(sr_d, n_trials=nt, n_obs=len(combo)):.3f}")


if __name__ == "__main__":
    main()
