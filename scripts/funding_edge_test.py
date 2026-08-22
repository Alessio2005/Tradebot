"""funding_edge_test.py — does free funding-rate data add real edge?

Three honest tests on the broad funding panel:
  A. Directional IC: funding-z (crowding) vs forward N-day return, TS & XS.
  B. Broad cross-sectional funding carry sleeve (MN), ~99 names vs the 5-name carry.
  C. Funding-conditioned reversal (does crowding sharpen the reversal premium?).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
DAYS, COST = 365.0, 10.0

P = pd.read_parquet(ROOT / "artefacts/broad_perp_daily_close_WIDE.parquet").resample("1D").last()
F = pd.read_parquet(ROOT / "artefacts/funding_universe.parquet").resample("1D").sum()
common = [c for c in P.columns if c in F.columns]
P = P[common]; R = np.log(P / P.shift(1)); R = R[R.notna().sum(axis=1) >= 20]
F = F.reindex(R.index)[common].fillna(0.0)
print(f"Funding panel: {len(common)} names aligned, {len(R)} days")


def sh(x): return x.mean() / x.std() * np.sqrt(DAYS) if x.std() > 0 else np.nan


def test_A():
    print("\n=== A. Directional IC: funding-z vs forward return ===")
    fz = F.rolling(7).mean()
    fz = fz.sub(fz.mean(axis=1), axis=0).div(fz.std(axis=1).replace(0, np.nan), axis=0)  # XS z
    for H in (1, 3, 5, 10):
        fwd = np.log(P.shift(-H) / P).reindex(index=fz.index, columns=fz.columns)
        m = pd.DataFrame({"f": fz.shift(1).values.ravel(), "r": fwd.values.ravel()}).dropna()
        ic, _ = spearmanr(m["f"], m["r"])
        print(f"  H={H:2d}d: XS IC(funding_z -> fwd ret) = {ic:+.4f}  (n={len(m)})  "
              f"[+ = high funding predicts UP, − = reversal]")


def test_B():
    print("\n=== B. Broad cross-sectional funding carry sleeve (MN) ===")
    for k in (1, 3, 7):
        car = -F.rolling(k).mean()
        z = car.sub(car.mean(axis=1), axis=0).div(car.std(axis=1).replace(0, np.nan), axis=0)
        w = z.sub(z.mean(axis=1), axis=0); w = w.div(w.abs().sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
        we = w.shift(1).fillna(0.0)
        # carry pnl = price return on weights + funding accrual (short pays/receives)
        price = (we * R).sum(axis=1); fund = -(we * F).sum(axis=1)
        cost = (we - we.shift(1).fillna(0.0)).abs().sum(axis=1) * COST / 1e4
        net = (price + fund - cost).dropna()
        line = f"  k={k}: Sharpe={sh(net):+.2f} | "
        neg = 0
        for y, g in net.groupby(net.index.year):
            r = (1 + g).prod() - 1; neg += r < 0; line += f"{y}:{r*100:+.0f}% "
        print(line + f"| NEG={neg}")


def test_C():
    print("\n=== C. Funding-conditioned reversal (crowding sharpens reversal?) ===")
    m = R.mean(axis=1); resid = R.sub(m, axis=0)
    rev = -resid.rolling(10).sum()
    fz = F.rolling(7).mean(); fz = fz.sub(fz.mean(axis=1), axis=0).div(fz.std(axis=1).replace(0, np.nan), axis=0)
    for mode, sig in (("plain rev", rev),
                      ("rev × |funding_z|", rev * fz.abs()),
                      ("rev + funding tilt", rev + 0.5 * (-fz))):  # short high-funding too
        z = sig.sub(sig.mean(axis=1), axis=0).div(sig.std(axis=1).replace(0, np.nan), axis=0)
        w = z.sub(z.mean(axis=1), axis=0); w = w.div(w.abs().sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
        we = w.shift(1).fillna(0.0)
        net = ((we * R).sum(axis=1) - (we - we.shift(1).fillna(0.0)).abs().sum(axis=1) * COST / 1e4).dropna()
        print(f"  {mode:22s}: Sharpe={sh(net):+.2f}")


if __name__ == "__main__":
    test_A(); test_B(); test_C()
