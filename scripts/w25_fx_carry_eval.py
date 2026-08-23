"""w25_fx_carry_eval.py — Wave 25: G10 FX carry unit, full per-unit checklist.

Stappenplan §9: simple literature-conform unit (Koijen et al. 2018) ->
walk-forward OOS net of costs -> G4 factor regression (FX set: DOLLAR,
CARRY, TREND; binding INCLUSIVE per F13/F17-precedent, ex-CARRY reported
as the Koijen-diagnostic) -> correlation vs the crypto MN book (full +
stress windows) -> ledger + serial accept/archive decision.

Run:  python scripts/w25_fx_carry_eval.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tradebot.alpha import fx_carry  # noqa: E402
from tradebot.data.fx_universe import build_fx_factors, load_fx_panels  # noqa: E402
from tradebot.alpha.factor_alpha import factor_residual_alpha  # noqa: E402

DAYS_CRYPTO = 365.0
COST_BPS = 10.0
TARGET_VOL = 0.40


def _ann_sharpe(r: pd.Series, periods: float = 252.0) -> float:
    r = r.dropna()
    return float(r.mean() / r.std() * np.sqrt(periods)) if r.std() > 0 else float("nan")


def crypto_mn_book() -> pd.Series:
    """Reproduce the 5-asset crypto XS MN book (ML-XS + LOWVOL + STATARB, RP,
    40% vol; Sharpe ~0.85) from the persisted artefacts — same math as
    scripts/multi_sleeve_combine.py (kept verbatim; that script prints only).

    NB this is the 5-asset XS book, NOT the broad-75-perp §9 book (Sharpe
    1.15). For the W25 correlation gate the choice is immaterial — FX-vs-
    crypto is cross-asset-class — but the label must be honest."""
    sc = pd.read_parquet(ROOT / "artefacts" / "xs_oos_scores.parquet")
    close = pd.read_parquet(ROOT / "artefacts" / "xs_close_panel.parquet")
    score = sc.pivot_table(index="__date", columns="__sym", values="__p", aggfunc="last")
    score = score.reindex(close.index).reindex(columns=close.columns)
    rets = np.log(close / close.shift(1))
    vol = rets.rolling(30).std().shift(1)
    logvol = np.log(vol.clip(lower=1e-4))
    btc = rets["BTCUSDT"]
    beta = rets.rolling(60).cov(btc).div(btc.rolling(60).var(), axis=0).shift(1)
    resid = rets.sub(beta.mul(btc, axis=0))

    def demean(s: pd.DataFrame) -> pd.DataFrame:
        return s.sub(s.mean(axis=1), axis=0)

    def vt(p: pd.Series) -> pd.Series:
        p = p.dropna()
        bv = p.std() * np.sqrt(DAYS_CRYPTO)
        return p * (TARGET_VOL / bv) if bv > 0 else p

    def pnl_from_w(w: pd.DataFrame) -> pd.Series:
        g = w.abs().sum(axis=1).replace(0, np.nan)
        wn = w.div(g, axis=0).fillna(0.0)
        return (wn.shift(1) * rets).sum(axis=1) - (
            wn - wn.shift(1)
        ).abs().sum(axis=1) * COST_BPS / 1e4

    rank = score.rank(axis=1, pct=True)
    sig = demean(rank)
    sig_n = sig * np.nan
    for dt in sig.index:
        y = sig.loc[dt].dropna()
        if len(y) < 15:
            continue
        X = pd.DataFrame({"lv": logvol.loc[dt], "b": beta.loc[dt]}).reindex(y.index).fillna(0.0)
        X.insert(0, "c", 1.0)
        A = X.values
        if not np.isfinite(A).all():
            sig_n.loc[dt, y.index] = y.values
            continue
        try:
            coef, *_ = np.linalg.lstsq(A, y.values, rcond=None)
            sig_n.loc[dt, y.index] = y.values - A @ coef
        except np.linalg.LinAlgError:
            sig_n.loc[dt, y.index] = y.values
    ml = vt(pnl_from_w(sig_n.ewm(span=5).mean()))
    lowvol = vt(pnl_from_w(-demean(logvol.rank(axis=1, pct=True)).ewm(span=5).mean()))
    statarb = vt(pnl_from_w(-demean(resid.rolling(5).sum().rank(axis=1, pct=True)).ewm(span=3).mean()))

    idx = ml.dropna().index
    book = vt(sum(s.reindex(idx).fillna(0.0) for s in (ml, lowvol, statarb)))
    return book


def main() -> int:
    tr, carry = load_fx_panels(root=ROOT / "market_data_parquet")

    print("=== W25 FX DATA COVERAGE ===")
    for c in tr.columns:
        s, k = tr[c].dropna(), carry[c].dropna()
        print(
            f"  {c}: TR {s.index[0].date()} -> {s.index[-1].date()} ({len(s)})  "
            f"carry {k.index[0].date()} -> {k.index[-1].date()} ({len(k)})"
        )
    breadth = carry.notna().sum(axis=1)
    full = breadth[breadth >= 6]
    eval_start = str(full.index[0].date())
    print(f"  breadth>=6 from {eval_start}; last date {tr.index[-1].date()}")

    res = fx_carry.run(tr, carry)
    summ = res.summary(start=eval_start)
    print("\n=== W25 fx_carry_g10 (net of 1bp half-spread, ME rebalance) ===")
    print(f"  net Sharpe {summ['net_sharpe']:+.2f}  gross {summ['gross_sharpe']:+.2f}  "
          f"net CAGR {summ['net_cagr'] * 100:+.2f}%  turnover/d {summ['avg_daily_turnover']:.4f}  "
          f"n_days {summ['n_days']}")
    yrs = summ["per_year_net"]
    line = "  ".join(f"{y}:{v * 100:+.1f}%" for y, v in sorted(yrs.items()))
    print("  per-year net:\n   " + "\n   ".join(line[i:i + 110] for i in range(0, len(line), 110)))
    pos_years = sum(v > 0 for v in yrs.values())
    print(f"  positive years: {pos_years}/{len(yrs)}")
    # sub-period honesty: pre/post 2008 (carry crash) and last 10y
    net = res.net_returns.loc[eval_start:]
    for label, sl in (
        ("pre-2008", net.loc[:"2007-12-31"]),
        ("2008-2021", net.loc["2008-01-01":"2021-12-31"]),
        ("2022+", net.loc["2022-01-01":]),
        ("last 10y", net.iloc[-2520:]),
    ):
        print(f"  {label:10s} Sharpe {_ann_sharpe(sl):+.2f}  n={len(sl.dropna())}")

    factors = build_fx_factors(tr, carry)
    print("\n=== FX factorset sanity (gross, ann.) ===")
    for c in factors.columns:
        print(f"  {c}: Sharpe {_ann_sharpe(factors[c]):+.2f}  "
              f"ann.ret {factors[c].dropna().mean() * 252 * 100:+.2f}%")

    print("\n=== G4 — binding: full FX set (DOLLAR+CARRY+TREND), HAC ===")
    g4 = factor_residual_alpha(
        net, factors, unit="fx_carry_g10", market="fx", periods_per_year=252
    )
    print("  " + g4.gate_row())
    print("  loadings: " + "  ".join(
        f"{k}={v:+.3f}(t={g4.loading_tstats[k]:+.1f})" for k, v in g4.loadings.items()))

    print("\n=== G4 diagnostic — ex-CARRY (the Koijen question: does the "
          "premium survive dollar+trend?) ===")
    g4x = factor_residual_alpha(
        net, factors[["DOLLAR", "TREND"]], unit="fx_carry_g10_exCARRY",
        market="book", periods_per_year=252,
    )
    print("  " + g4x.gate_row())

    print("\n=== correlation vs crypto MN book ===")
    book = crypto_mn_book()
    a = net.copy()
    a.index = pd.DatetimeIndex(a.index.date)
    b = book.copy()
    b.index = pd.DatetimeIndex(b.index.date)
    j = pd.concat([a.rename("fx"), b.rename("book")], axis=1, join="inner").dropna()
    corr_full = float("nan")
    if len(j) >= 60:
        corr_full = j["fx"].corr(j["book"])
        print(f"  overlap {len(j)} days ({j.index[0].date()} -> {j.index[-1].date()})")
        print(f"  corr full    : {corr_full:+.3f}")
        stress = j.loc["2022-01-01":"2022-12-31"]
        if len(stress) > 60:
            print(f"  corr 2022    : {stress['fx'].corr(stress['book']):+.3f}  (n={len(stress)})")
    else:
        print(f"  overlap {len(j)} days — too thin for a correlation "
              "(crypto book starts 2021; need FX data through 2021+).")

    out = ROOT / "artefacts" / "fx"
    out.mkdir(parents=True, exist_ok=True)
    net.rename("net_ret").to_frame().to_parquet(out / "fx_carry_net.parquet")
    factors.to_parquet(out / "fx_factors_daily.parquet")
    book.rename("net_ret").to_frame().to_parquet(out / "crypto_mn_book_daily.parquet")
    print(f"\npersisted: fx_carry_net, fx_factors_daily, crypto_mn_book_daily -> {out}")

    print("\n=== ACCEPT CRITERIA (stappenplan §9) ===")
    print(f"  net Sharpe >= 0.40 : {summ['net_sharpe']:+.2f}  "
          f"{'PASS' if summ['net_sharpe'] >= 0.40 else 'FAIL'}")
    print(f"  t(alpha)  >= 2.0  : {g4.t_alpha:+.2f}  {'PASS' if g4.passes else 'FAIL'}")
    if corr_full == corr_full:  # not NaN
        print(f"  |corr| vs book < 0.3 : {abs(corr_full):.3f}  "
              f"{'PASS' if abs(corr_full) < 0.3 else 'FAIL'}")
    else:
        print("  |corr| vs book < 0.3 : n/a (insufficient overlap)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
