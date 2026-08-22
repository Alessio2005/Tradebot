"""validate_neutral_book.py — CHIEF out-of-sample validation of the 3-sleeve
market-neutral book. Tests whether the in-sample result survives:
  (1) causal (expanding-window) risk-parity weighting — no full-sample lookahead,
  (2) strict IS(2021-2023) / OOS(2024-2026) split — weights frozen on IS,
  (3) realistic cost sensitivity (5/7/10 bps),
  (4) correctly-unit Deflated Sharpe / t-stat.

Residual caveat (cannot be undone here): the FACTOR SET (carry, residual-reversal,
low-vol) was chosen with full-sample hindsight. The OOS split validates the
WEIGHTING + parameters out-of-sample, not the factor selection.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import importlib.util
spec = importlib.util.spec_from_file_location("cb", ROOT / "scripts/combined_neutral_book.py")
cb = importlib.util.module_from_spec(spec); spec.loader.exec_module(cb)


def lowvol_pnl(cost_bps=5.0):
    P = pd.read_parquet(ROOT / "artefacts/broad_perp_daily_close.parquet").resample("1D").last()
    R = np.log(P / P.shift(1)); R = R[R.notna().sum(axis=1) >= 20]
    vol = R.rolling(20).std()
    z = vol.sub(vol.mean(axis=1), axis=0).div(vol.std(axis=1).replace(0, np.nan), axis=0)
    w = (-z).sub((-z).mean(axis=1), axis=0); w = w.div(w.abs().sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    we = w.shift(1).fillna(0.0)
    cost = (we - we.shift(1).fillna(0.0)).abs().sum(axis=1) * cost_bps / 1e4
    return ((we * R).sum(axis=1) - cost).rename("lowvol")


def sleeves(cost_bps):
    c = cb.carry_pnl(cost_bps=cost_bps)
    s, mkt = cb.statarb_pnl(cost_bps=cost_bps)
    l = lowvol_pnl(cost_bps=cost_bps)
    idx = c.index.intersection(s.index).intersection(l.index)
    return (c.reindex(idx).fillna(0.0), s.reindex(idx).fillna(0.0), l.reindex(idx).fillna(0.0), mkt)


def peryear(net):
    out = {}
    for y, g in net.groupby(net.index.year):
        out[y] = (1 + g).prod() - 1
    return out


def ann_sharpe(net):
    return net.mean() / net.std() * np.sqrt(365) if net.std() > 0 else np.nan


def causal_rp(c, s, l, lookback=252):
    """Expanding/rolling inverse-vol weights using only PAST data (shift 1)."""
    df = pd.concat([c, s, l], axis=1); df.columns = ["c", "s", "l"]
    vol = df.rolling(lookback, min_periods=60).std().shift(1)   # past-only
    iv = 1.0 / vol.replace(0, np.nan)
    w = iv.div(iv.sum(axis=1), axis=0)
    combo = (w * df).sum(axis=1)
    return combo.dropna()


def main():
    print("=== 1) Causal risk-parity (no full-sample lookahead), 5 bps ===")
    c, s, l, mkt = sleeves(5.0)
    combo_causal = causal_rp(c, s, l)
    py = peryear(combo_causal)
    line = "  ".join(f"{y}:{v*100:+.0f}%" for y, v in py.items())
    beta = np.polyfit(mkt.reindex(combo_causal.index).fillna(0), combo_causal, 1)[0]
    neg = sum(v < 0 for v in py.values())
    print(f"  per-jaar: {line}")
    print(f"  Sharpe={ann_sharpe(combo_causal):+.2f}  beta={beta:+.3f}  NEG={neg}")

    print("\n=== 2) Strict IS(2021-2023) -> OOS(2024-2026), weights frozen on IS ===")
    is_mask = combo_causal.index.year <= 2023
    cis, sis, lis = c[c.index.year <= 2023], s[s.index.year <= 2023], l[l.index.year <= 2023]
    iv = np.array([1/cis.std(), 1/sis.std(), 1/lis.std()]); wis = iv/iv.sum()
    print(f"  IS-frozen RP weights: carry={wis[0]:.2f} statarb={wis[1]:.2f} lowvol={wis[2]:.2f}")
    full = wis[0]*c + wis[1]*s + wis[2]*l
    for label, mask in [("IS 2021-2023", full.index.year <= 2023), ("OOS 2024-2026", full.index.year >= 2024)]:
        seg = full[mask]; py = peryear(seg)
        line = "  ".join(f"{y}:{v*100:+.0f}%" for y, v in py.items())
        b = np.polyfit(mkt.reindex(seg.index).fillna(0), seg, 1)[0]
        print(f"  {label}: {line} | Sharpe={ann_sharpe(seg):+.2f} beta={b:+.3f} NEG={sum(v<0 for v in py.values())}")

    print("\n=== 3) Cost sensitivity (causal RP) ===")
    for cb_ in (5.0, 7.0, 10.0):
        cc, ss, ll, _ = sleeves(cb_)
        combo = causal_rp(cc, ss, ll)
        py = peryear(combo)
        print(f"  {cb_:>4.1f}bps: Sharpe={ann_sharpe(combo):+.2f}  NEG_jaren={sum(v<0 for v in py.values())}  "
              f"worst_year={min(py.values())*100:+.0f}%")

    print("\n=== 4) Significance ===")
    sh = ann_sharpe(combo_causal); n = len(combo_causal)
    tstat = sh * np.sqrt(n / 365.0)
    print(f"  Annual Sharpe={sh:.2f}  n_days={n}  t-stat={tstat:.2f}")
    try:
        from tradebot.backtest.metrics import deflated_sharpe
        sr_daily = combo_causal.mean() / combo_causal.std()    # per-day SR (correct units)
        dsr = deflated_sharpe(sr_daily, n_trials=12, n_obs=n)
        print(f"  DSR(per-day SR={sr_daily:.4f}, n_trials=12, n_obs={n}) = {dsr:.3f}  (promote if >0.95)")
    except Exception as e:
        print("  DSR skip:", e)


if __name__ == "__main__":
    main()
