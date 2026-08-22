"""adaptive_wf_pbo.py — conclusive CSCV PBO for the adaptive book (Wave 19).

The in-engine PBO used only 8 construction variants (< Bailey-LdP S>=50 floor, so
indicative only). This sweeps 70+ harvest configurations from the CACHED walk-forward
scores (no retrain) and runs CSCV PBO at S>=50 for a statistically conclusive
overfit estimate. Reuses the engine's neutralise/regime/band internals.

Run (after `python apps/run_adaptive_wf.py` has cached scores):
  python scripts/adaptive_wf_pbo.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from tradebot.alpha.adaptive_wf import AdaptiveWalkForward, AdaptiveWFConfig
from tradebot.backtest.pbo import compute_pbo

DAYS = 365.0; COST_BPS = 10.0; TARGET_VOL = 0.40


def main():
    ohlcv = pd.read_parquet(ROOT / "artefacts" / "broad_perp_ohlcv.parquet")
    fund_p = ROOT / "artefacts" / "funding_universe.parquet"
    funding = pd.read_parquet(fund_p) if fund_p.exists() else None
    scores = pd.read_parquet(ROOT / "artefacts" / "adaptive_wf_scores.parquet")

    eng = AdaptiveWalkForward(AdaptiveWFConfig()).build_panel(ohlcv, funding)
    pan = eng._panels
    close, rets, mkt, btc_ret = pan["close"], pan["rets"], pan["mkt"], pan["btc_ret"]
    syms = pan["syms"]
    score = (scores.pivot_table(index="__date", columns="__sym", values="__p", aggfunc="last")
             .reindex(close.index).reindex(columns=syms))
    vol = rets.rolling(30).std().shift(1); logvol = np.log(vol.clip(lower=1e-4))
    betaf = rets.rolling(60).cov(btc_ret).div(btc_ret.rolling(60).var(), axis=0).shift(1)
    rk = score.rank(axis=1, pct=True); sig0 = rk.sub(rk.mean(axis=1), axis=0)
    sig_neu = eng._neutralise(sig0, logvol, betaf)          # computed ONCE
    regime_gross = eng._regime_gross(mkt)                   # computed ONCE

    def pnl_of(sig, span, regime, band):
        s = sig.ewm(span=span).mean()
        g = s.abs().sum(axis=1).replace(0, np.nan); wn = s.div(g, axis=0).fillna(0.0)
        if regime:
            wn = wn.mul(regime_gross, axis=0)
        if band > 0:
            wn = eng._apply_band(wn, band)
        p = ((wn.shift(1) * rets).sum(axis=1)
             - (wn - wn.shift(1)).abs().sum(axis=1) * COST_BPS / 1e4).dropna()
        bv = p.std() * np.sqrt(DAYS)
        return p * (TARGET_VOL / bv) if bv > 0 else p

    variants = {}
    for bname, base in (("raw", sig0), ("neu", sig_neu)):
        for span in (3, 5, 8, 13, 21, 34):
            for regime in (True, False):
                for band in (0.0, 0.004, 0.008):
                    variants[f"{bname}_s{span}_r{int(regime)}_b{band}"] = pnl_of(base, span, regime, band)
    idx = sorted(set().union(*[v.index for v in variants.values()]))
    mat = pd.DataFrame({k: v.reindex(idx) for k, v in variants.items()}).fillna(0.0)
    print(f"variants={mat.shape[1]}  obs={mat.shape[0]}", flush=True)

    res = compute_pbo(mat.to_numpy(), n_subsets=16)
    print(f"\n=== CONCLUSIVE CSCV PBO (S={mat.shape[1]}) ===")
    for k, v in res.items():
        print(f"  {k}: {v}")
    # deployed config = neu, span5, regime on, band 0.004
    dep = "neu_s5_r1_b0.004"
    if dep in variants:
        sh = lambda p: p.mean() / p.std() * np.sqrt(DAYS)
        ranks = pd.Series({k: sh(v) for k, v in variants.items()}).sort_values(ascending=False)
        print(f"\n  deployed config '{dep}': Sharpe={sh(variants[dep]):.2f}, "
              f"rank {list(ranks.index).index(dep)+1}/{len(ranks)} by in-sample Sharpe")


if __name__ == "__main__":
    main()
