"""Literature-pinned candidates on the six certified perps (development sample only, < 2026-06-24).

Pre-specified BEFORE the first run (trial count for this script = 3: T1, T2, C1; sensitivities are not new trials):
  T1  TSMOM ensemble, long-short: s_i,t = mean_L sign(logP_t - logP_{t-L}), L in {7,14,28,56,112};
      w_i,t = s_i,t * (0.40 / sigma_i,t) / N_live ; sigma = EWMA(0.94) annualised (the repo's vol model).
      Decided on close t, earns from close t to close t+1 (w.shift(1)); sensitivity: w.shift(2).
      Costs: 6.5 bps per unit |dw| (taker 5.5 + half-spread 1.0, conf/execution/fees.yaml),
      funding: a long pays positive funding (f summed per day), a short receives it.
  T2  Same, long-flat (s clipped at 0)  -- literature: the crypto short leg loses money.
  B0  Benchmark: vol-scaled long-only (s = +1), same costs.  B1: BTC buy & hold (unlevered).
  C1  Cash-and-carry yield: short perp + long spot per coin, 1/N notional, held only while the trailing
      7-day mean daily funding > 0; earns realised funding; costs 2 legs x (perp 6.5 + spot 11) bps per switch;
      basis P&L ignored (flagged).  Reported as annual yield, not a directional strategy.
"""
import os, sys, json
import numpy as np, pandas as pd
from pathlib import Path
ROOT = Path("/home/claude/alessio2005/tradebot"); os.chdir(ROOT)
from tradebot.data.weekly_market import load_weekly_market
from tradebot.validation.inference import sharpe_with_se, block_bootstrap_ci
import statsmodels.api as sm

SYMS = ["BTCUSDT","ETHUSDT","SOLUSDT","AVAXUSDT","LINKUSDT","DOTUSDT"]
full = load_weekly_market(ROOT, SYMS)
m = full.truncate(pd.Timestamp("2026-06-24T00:00:00+00:00"))
close = pd.DataFrame({s: m.ohlcv[s]["close"] for s in SYMS})
logp = np.log(close)
ret = close.pct_change()
sig = m.sigma_annual
fund = m.funding.fillna(0.0)        # daily sum of 8h settlements (fraction)
COST = 6.5e-4
LOOKS = [7, 14, 28, 56, 112]

def signal(long_flat=False):
    s = sum(np.sign(logp - logp.shift(L)) for L in LOOKS) / len(LOOKS)
    s = s.where(logp.shift(max(LOOKS)).notna())
    return s.clip(lower=0) if long_flat else s

def book(s, lag=1):
    live = s.notna() & sig.notna()
    n_live = live.sum(axis=1).replace(0, np.nan)
    w = (s * 0.40 / sig).where(live).div(n_live, axis=0).fillna(0.0)
    gross = w.abs().sum(axis=1); w = w.mul(np.minimum(1.0, 4.0 / gross.replace(0, np.nan)).fillna(1.0), axis=0)
    held = w.shift(lag).fillna(0.0)
    gross_ret = (held * ret.fillna(0.0)).sum(axis=1)
    fund_cost = (held * fund).sum(axis=1)           # long pays positive funding
    turnover = w.diff().abs().sum(axis=1).shift(lag - 1).fillna(0.0)
    net = gross_ret - fund_cost - COST * turnover
    return pd.DataFrame({"gross": gross_ret, "net": net, "funding": -fund_cost,
                         "costs": -COST * turnover, "turnover": turnover, "gross_lev": held.abs().sum(axis=1)})

def stats_(r, start, end):
    r = r.loc[start:end]
    se = sharpe_with_se(r, bars_per_year=365.0)
    ci = block_bootstrap_ci(r, bars_per_year=365.0, seed=7)
    ann = r.mean() * 365; vol = r.std() * np.sqrt(365)
    eq = (1 + r).cumprod(); mdd = (eq / eq.cummax() - 1).min()
    return {"sharpe": round(se.sharpe, 2), "se": round(se.se, 2), "ci95": [round(ci.low, 2), round(ci.high, 2)],
            "ann_ret": round(ann, 3), "ann_vol": round(vol, 3), "max_dd": round(mdd, 3),
            "years": round(len(r) / 365, 2)}

out = {}
books = {"T1_trend_LS": book(signal()), "T2_trend_long_flat": book(signal(True)),
         "B0_volscaled_long": book(signal().notna().astype(float).where(signal().notna())),
         "T1_lag2": book(signal(), lag=2), "T2_lag2": book(signal(True), lag=2)}
btc = ret["BTCUSDT"].fillna(0.0)
periods = {"full_2020-09_2026-06": ("2020-09-01", "2026-06-23"), "dev_OOS_2022-01_2026-06": ("2022-01-01", "2026-06-23"),
           "2024-01_2026-06": ("2024-01-01", "2026-06-23")}
for pname, (a, b) in periods.items():
    out[pname] = {k: stats_(v["net"], a, b) for k, v in books.items()}
    out[pname]["B1_BTC_buyhold"] = stats_(btc, a, b)
    out[pname]["T1_gross"] = stats_(books["T1_trend_LS"]["gross"], a, b)
# yearly net Sharpe/returns
yearly = {}
for k in ["T1_trend_LS", "T2_trend_long_flat", "B0_volscaled_long"]:
    r = books[k]["net"].loc["2020-09-01":]
    yearly[k] = {str(y): {"ret": round(g.sum(), 3), "sr": round(g.mean() / g.std() * np.sqrt(365), 2)} for y, g in r.groupby(r.index.year)}
out["yearly"] = yearly
# cost/funding decomposition (OOS)
for k in ["T1_trend_LS", "T2_trend_long_flat"]:
    d = books[k].loc["2022-01-01":]
    out[f"decomp_{k}"] = {"gross_ann": round(d.gross.mean()*365, 3), "funding_ann": round(d.funding.mean()*365, 3),
                          "costs_ann": round(d.costs.mean()*365, 3), "turnover_per_day": round(d.turnover.mean(), 3),
                          "avg_gross_lev": round(d.gross_lev.mean(), 2)}
# alpha of T1 vs B0 (does trend add over vol-scaled beta?) on dev OOS, HAC
for k in ["T1_trend_LS", "T2_trend_long_flat"]:
    y = books[k]["net"].loc["2022-01-01":]; x = sm.add_constant(books["B0_volscaled_long"]["net"].loc["2022-01-01":])
    fit = sm.OLS(y, x).fit(cov_type="HAC", cov_kwds={"maxlags": 10})
    out[f"alpha_{k}_vs_B0"] = {"alpha_ann": round(fit.params.iloc[0]*365, 3), "t": round(fit.tvalues.iloc[0], 2),
                               "beta": round(fit.params.iloc[1], 2)}
# C1 cash-and-carry
f7 = fund.rolling(7, min_periods=7).mean()
on = (f7 > 0).astype(float).where(close.notna(), 0.0)
held = on.shift(1).fillna(0.0)
n = close.notna().sum(axis=1).replace(0, np.nan)
carry = (held * fund).div(n, axis=0).sum(axis=1)
switch_cost = (on.diff().abs().fillna(0.0) * (6.5e-4 + 11e-4)).div(n, axis=0).sum(axis=1)
c_net = carry - switch_cost
always = fund.where(close.notna()).div(n, axis=0).sum(axis=1)
c1 = {}
for y, g in c_net.loc["2020-09-01":].groupby(c_net.loc["2020-09-01":].index.year):
    c1[str(y)] = {"net_yield_ann": round(g.mean()*365, 4), "always_on_gross_ann": round(always.loc[g.index].mean()*365, 4), "days": len(g),
                  "frac_days_on": round(held.loc[g.index].mean().mean(), 2)}
out["C1_cash_and_carry_by_year"] = c1
out["C1_sharpe_dev_OOS"] = stats_(c_net, "2022-01-01", "2026-06-23")
json.dump(out, open("/home/claude/research/trend_carry.json", "w"), indent=1)
pd.DataFrame({k: v["net"] for k, v in books.items()} | {"BTC": btc, "C1": c_net}).to_parquet("/home/claude/research/books.parquet")
print(json.dumps(out, indent=1))
