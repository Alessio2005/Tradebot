"""regime_timing_book.py — directional market-timing sleeve from free alt-data (Wave 16).

Uses the NEW free alt-data (sentiment, DeFi flows, on-chain, DVOL) the prior waves
never touched, to time the crypto market (equal-weight basket, traded via BTC+ETH).
Directional is allowed under mandate v2. Measures the sleeve honestly AND its
correlation to the market-neutral XS-ML book -> does it add an orthogonal √N track?

Run:  python scripts/regime_timing_book.py
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from tradebot.backtest.metrics import deflated_sharpe

DAYS = 365.0; COST_BPS = 5.0; TARGET_VOL = 0.40; H = 10
N_FOLDS, EMBARGO = 6, H + 3


def zt(s, n=60):
    return (s - s.rolling(n).mean()) / s.rolling(n).std()


def main():
    alt = pd.read_parquet(ROOT / "artefacts" / "altdata_macro.parquet")
    close = pd.read_parquet(ROOT / "artefacts" / "xs_close_panel.parquet")
    idx = close.index
    rets = np.log(close / close.shift(1))
    mkt = rets.mean(axis=1)                          # equal-weight crypto market
    btc = close["BTCUSDT"]

    alt = alt.reindex(idx).ffill()
    F = pd.DataFrame(index=idx)
    # sentiment
    F["fng"] = alt["fng"]; F["fng_chg"] = alt["fng"].diff(5)
    # capital flows
    F["tvl_mom"] = np.log(alt["defi_tvl"]).diff(20)
    F["stbl_growth"] = np.log(alt["stbl_supply"]).diff(20)
    F["stbl_chg5"] = np.log(alt["stbl_supply"]).diff(5)
    # on-chain
    F["addr_mom"] = np.log(alt["btc_active_addr"].clip(lower=1)).diff(20)
    F["tx_mom"] = np.log(alt["btc_txcount"].clip(lower=1)).diff(20)
    F["hash_mom"] = np.log(alt["btc_hashrate"].clip(lower=1)).diff(30)
    # options / vol regime
    F["dvol_btc"] = alt["dvol_btc"]; F["dvol_btc_z"] = zt(alt["dvol_btc"])
    F["dvol_eth_z"] = zt(alt["dvol_eth"])
    F["dvol_chg"] = alt["dvol_btc"].diff(5)
    # market technicals
    lr = np.log(btc / btc.shift(1))
    F["btc_ma50"] = btc / btc.rolling(50).mean() - 1
    F["btc_ma200"] = btc / btc.rolling(200).mean() - 1
    F["btc_mom20"] = np.log(btc / btc.shift(20))
    F["btc_vol20"] = lr.rolling(20).std()
    F["mkt_mom20"] = mkt.rolling(20).sum()
    F = F.shift(1)                                   # causal: data <= t-1
    FEATS = list(F.columns)

    # target: forward H-day market return > 0
    fwd = mkt.shift(-1).rolling(H).sum().shift(-(0))   # approx fwd H-day from t+1
    fwd = (np.log(close.shift(-H) / close)).mean(axis=1)   # cleaner: mean fwd H-day log ret
    y = (fwd > 0).astype(int)
    data = pd.concat([F, y.rename("__y"), fwd.rename("__fwd")], axis=1).dropna()
    print(f"timing rows={len(data)}  feats={len(FEATS)}  base up-rate={data['__y'].mean():.3f}", flush=True)

    dt = data.index.values
    edges = pd.to_datetime(np.quantile(pd.to_datetime(dt).astype("int64"), np.linspace(0, 1, N_FOLDS + 1)))
    emb = np.timedelta64(EMBARGO, "D"); oos = pd.Series(np.nan, index=data.index)
    for k in range(N_FOLDS):
        lo, hi = edges[k].to_datetime64(), edges[k + 1].to_datetime64()
        test = (dt >= lo) & (dt < hi) if k < N_FOLDS - 1 else (dt >= lo) & (dt <= hi)
        overlap = (dt >= (lo - emb)) & (dt <= (hi + emb)); train = (~test) & (~overlap)
        if train.sum() < 300:
            continue
        m = CatBoostClassifier(iterations=300, depth=4, learning_rate=0.03, l2_leaf_reg=8,
                               loss_function="Logloss", verbose=0, random_seed=0, subsample=0.8, rsm=0.8)
        m.fit(data.loc[train, FEATS].fillna(0.0), data.loc[train, "__y"])
        oos.iloc[test.nonzero()[0]] = m.predict_proba(data.loc[test, FEATS].fillna(0.0))[:, 1]
    mask = oos.notna()
    auc = roc_auc_score(data.loc[mask, "__y"], oos[mask])
    ic = oos[mask].corr(data.loc[mask, "__fwd"], method="spearman")
    print(f"timing OOS AUC={auc:.4f}  IC(prob vs fwd mkt ret)={ic:+.4f}", flush=True)

    # book: market exposure from prob; trade equal-weight basket (proxy = mkt return)
    prob = oos.reindex(idx).ffill(limit=H)
    def make(longshort):
        if longshort:
            expo = ((prob - 0.5) * 2).clip(-1, 1)          # long/short
        else:
            expo = (prob > 0.5).astype(float)              # long/flat
        pnl = (expo.shift(1) * mkt) - (expo - expo.shift(1)).abs() * COST_BPS / 1e4
        pnl = pnl.dropna(); bv = pnl.std() * np.sqrt(DAYS)
        return pnl * (TARGET_VOL / bv) if bv > 0 else pnl
    for nm, ls in (("long/flat", False), ("long/short", True)):
        p = make(ls); sh = p.mean() / p.std() * np.sqrt(DAYS)
        eq = (1 + p).cumprod(); cagr = eq.iloc[-1] ** (DAYS / len(p)) - 1
        dd = (eq / eq.cummax() - 1).min(); b = p.cov(mkt.reindex(p.index)) / mkt.reindex(p.index).var()
        yr = " ".join(f"{y_}:{((1+g).prod()-1)*100:+.0f}%" for y_, g in p.groupby(p.index.year))
        print(f"\n[timing {nm}] Sharpe={sh:.2f} CAGR={cagr*100:+.0f}% MaxDD={dd*100:.0f}% beta={b:+.2f} "
              f"DSR={deflated_sharpe(p.mean()/p.std(),200,len(p)):.3f}")
        print(f"   per-year: {yr}")
        if nm == "long/short":
            timing = p

    # buy&hold benchmark
    bh = (mkt).dropna(); bhsh = bh.mean() / bh.std() * np.sqrt(DAYS)
    print(f"\n[buy&hold market] Sharpe={bhsh:.2f}")

    # correlation to MN XS book + combination
    try:
        sc = pd.read_parquet(ROOT / "artefacts" / "xs_oos_scores.parquet")
        score = sc.pivot_table(index="__date", columns="__sym", values="__p", aggfunc="last").reindex(idx).reindex(columns=close.columns)
        vol = rets.rolling(30).std().shift(1)
        rk = score.rank(axis=1, pct=True); sig = rk.sub(rk.mean(axis=1), axis=0).ewm(span=5).mean()
        g = sig.abs().sum(axis=1).replace(0, np.nan); w = sig.div(g, axis=0).fillna(0.0)
        xs = ((w.shift(1) * rets).sum(axis=1) - (w - w.shift(1)).abs().sum(axis=1) * 10 / 1e4).dropna()
        xsv = xs.std() * np.sqrt(DAYS); xs = xs * (TARGET_VOL / xsv)
        common = timing.dropna().index.intersection(xs.dropna().index)
        corr = timing.reindex(common).corr(xs.reindex(common))
        comb = (timing.reindex(common).fillna(0) + xs.reindex(common).fillna(0))
        cv = comb.std() * np.sqrt(DAYS); comb = comb * (TARGET_VOL / cv)
        csh = comb.mean() / comb.std() * np.sqrt(DAYS)
        eq = (1 + comb).cumprod(); ccagr = eq.iloc[-1] ** (DAYS / len(comb)) - 1
        cdd = (eq / eq.cummax() - 1).min()
        cyr = " ".join(f"{y_}:{((1+g2).prod()-1)*100:+.0f}%" for y_, g2 in comb.groupby(comb.index.year))
        print(f"\n=== corr(timing, XS-MN) = {corr:+.2f} ===")
        print(f"[COMBINED timing+XS] Sharpe={csh:.2f} CAGR={ccagr*100:+.0f}% MaxDD={cdd*100:.0f}% "
              f"DSR(N2000)={deflated_sharpe(comb.mean()/comb.std(),2000,len(comb)):.3f}")
        print(f"   per-year: {cyr}")
    except Exception as e:
        print("combine n/a:", str(e)[:60])


if __name__ == "__main__":
    main()
