"""paper_neutral_trader.py — paper-trading runner for the market-neutral book.

Trades the dollar-neutral multi-sleeve NeutralBook (statarb + low-vol; carry
optional) on the established Bybit USDT-perp universe. Daily rebalance, next-
bar fills, explicit taker+slippage costs, futures NAV accounting.

Modes:
  --replay            end-to-end dry-run over the cached/updated daily panel
                      (same code path as live -> proves the system works).
  --step              live daily step: update panel from Bybit, compute target
                      weights from data through the last complete bar, rebalance
                      once, persist state. (Run daily via cron for paper trading.)
  --update-data       only refresh the price panel cache, then exit.

State -> artefacts/paper_trade_neutral/{equity_curve.jsonl,state.json,audit.jsonl}

Methodology: weights use information only through bar t; the broker marks the
t->t+1 move on the PREVIOUS weights before rebalancing -> no look-ahead.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from tradebot.alpha.neutral_book import NeutralBook, NeutralBookConfig
from tradebot.data.perp_feed import build_or_update_panel, fetch_recent_funding

CACHE = _ROOT / "artefacts" / "broad_perp_daily_close.parquet"
STATE_DIR = _ROOT / "artefacts" / "paper_trade_neutral"
INITIAL_EQUITY = 200_000.0
GROSS_LEV = 1.0          # gross exposure = NAV (net ~0 -> conservative)
TAKER_BPS = 4.0          # Bybit USDT-M taker
SLIP_BPS = 2.0           # modest slippage for liquid perps, small size
COST_BPS = TAKER_BPS + SLIP_BPS


class PaperBroker:
    """Futures NAV accounting for a dollar-neutral book (base-qty positions)."""

    def __init__(self, equity: float = INITIAL_EQUITY):
        self.equity = float(equity)
        self.qty: dict[str, float] = {}
        self.last_px: dict[str, float] = {}
        self.records: list[dict[str, Any]] = []

    def step(self, ts: pd.Timestamp, prices: pd.Series, target_w: pd.Series,
             funding: pd.Series | None, gross_lev: float, cost_bps: float) -> dict[str, Any]:
        prices = prices.dropna()
        # 1) mark-to-market the PREVIOUS book on the move into `prices`
        mtm = 0.0
        fund_pnl = 0.0
        for sym, q in self.qty.items():
            if q == 0 or sym not in prices.index:
                continue
            p0 = self.last_px.get(sym)
            if p0 and p0 > 0:
                mtm += q * (prices[sym] - p0)
            if funding is not None and sym in funding.index:
                fund_pnl -= q * prices[sym] * float(funding.get(sym, 0.0))
        self.equity += mtm + fund_pnl

        # 2) target positions (base qty) from dollar-neutral weights
        tw = target_w.reindex(prices.index).fillna(0.0)
        target_notional = tw * self.equity * gross_lev
        target_qty = (target_notional / prices).replace([np.inf, -np.inf], 0.0).fillna(0.0)

        # 3) trade to target, charge cost on traded notional
        traded_notional = 0.0
        all_syms = set(self.qty) | set(target_qty.index)
        new_qty: dict[str, float] = {}
        for sym in all_syms:
            tq = float(target_qty.get(sym, 0.0))
            cq = float(self.qty.get(sym, 0.0))
            px = float(prices.get(sym, self.last_px.get(sym, 0.0)))
            traded_notional += abs(tq - cq) * px
            new_qty[sym] = tq
        cost = traded_notional * cost_bps / 1e4
        self.equity -= cost

        # 4) commit
        self.qty = {s: q for s, q in new_qty.items() if abs(q) > 1e-12}
        self.last_px = {s: float(prices[s]) for s in prices.index}
        gross = sum(abs(q) * self.last_px[s] for s, q in self.qty.items() if s in self.last_px)
        net = sum(q * self.last_px[s] for s, q in self.qty.items() if s in self.last_px)
        rec = {"ts": ts.isoformat(), "equity": round(self.equity, 2),
               "mtm": round(mtm, 2), "funding_pnl": round(fund_pnl, 2),
               "cost": round(cost, 2), "turnover_notional": round(traded_notional, 2),
               "gross_lev": round(gross / self.equity, 3) if self.equity else 0.0,
               "net_lev": round(net / self.equity, 4) if self.equity else 0.0,
               "n_pos": len(self.qty)}
        self.records.append(rec)
        return rec


def _persist(broker: PaperBroker, target_w: pd.Series) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with open(STATE_DIR / "equity_curve.jsonl", "w", encoding="utf-8") as f:
        for r in broker.records:
            f.write(json.dumps(r) + "\n")
    state = {"updated": datetime.now(timezone.utc).isoformat(),
             "equity": broker.equity, "n_positions": len(broker.qty),
             "positions_qty": broker.qty,
             "latest_target_weights": {k: round(float(v), 5) for k, v in target_w.items() if abs(v) > 1e-9}}
    (STATE_DIR / "state.json").write_text(json.dumps(state, indent=2))


def run_replay(panel: pd.DataFrame, with_carry: bool = False) -> None:
    nb = NeutralBook(NeutralBookConfig())
    combined, pnls = nb.weight_history(panel, funding=None)
    broker = PaperBroker()
    dates = combined.index
    for d in dates:
        w = combined.loc[d]
        if w.abs().sum() < 1e-9:
            continue
        broker.step(d, panel.loc[d], w, None, GROSS_LEV, COST_BPS)
    _persist(broker, combined.iloc[-1])
    eq = pd.Series({pd.Timestamp(r["ts"]): r["equity"] for r in broker.records})
    ret = eq.pct_change().dropna()
    sh = ret.mean() / ret.std() * np.sqrt(365) if ret.std() > 0 else float("nan")
    dd = (eq / eq.cummax() - 1).min()
    print(f"REPLAY: {len(broker.records)} rebalances | start={eq.iloc[0]:,.0f} end={eq.iloc[-1]:,.0f} "
          f"({(eq.iloc[-1]/eq.iloc[0]-1)*100:+.0f}%) Sharpe={sh:.2f} MaxDD={dd*100:.1f}%")
    line = ""
    for y, g in eq.groupby(eq.index.year):
        line += f"{y}:{(g.iloc[-1]/g.iloc[0]-1)*100:+.0f}% "
    print("per-jaar:", line)
    print(f"final gross_lev={broker.records[-1]['gross_lev']} net_lev={broker.records[-1]['net_lev']} "
          f"n_pos={broker.records[-1]['n_pos']}  -> state in {STATE_DIR}")


def run_step(panel: pd.DataFrame, with_carry: bool) -> None:
    # use data through the last COMPLETE daily bar (drop today's partial bar)
    panel = panel.iloc[:-1] if len(panel) > 1 else panel
    nb = NeutralBook(NeutralBookConfig())
    funding = None
    if with_carry:
        f = fetch_recent_funding(list(panel.columns))
        funding = f if not f.empty else None
    target_w = nb.target_weights(panel, funding=None)
    # load prior broker state if present
    broker = PaperBroker()
    sp = STATE_DIR / "state.json"
    if sp.exists():
        st = json.loads(sp.read_text())
        broker.equity = float(st.get("equity", INITIAL_EQUITY))
        broker.qty = {k: float(v) for k, v in st.get("positions_qty", {}).items()}
        ecp = STATE_DIR / "equity_curve.jsonl"
        if ecp.exists():
            broker.records = [json.loads(l) for l in ecp.read_text().splitlines() if l.strip()]
        # reconstruct last_px from the latest panel row for held names
        last_row = panel.iloc[-1]
        broker.last_px = {s: float(last_row[s]) for s in broker.qty if s in last_row.index and pd.notna(last_row[s])}
    rec = broker.step(panel.index[-1], panel.iloc[-1], target_w,
                      funding.iloc[-1] if funding is not None and len(funding) else None,
                      GROSS_LEV, COST_BPS)
    _persist(broker, target_w)
    print(f"STEP @ {rec['ts']}: equity={rec['equity']:,.2f} mtm={rec['mtm']:+,.0f} "
          f"cost={rec['cost']:,.0f} gross_lev={rec['gross_lev']} net_lev={rec['net_lev']} n_pos={rec['n_pos']}")
    top = target_w[target_w.abs() > 1e-9].sort_values()
    print("  shorts:", {k: round(v, 3) for k, v in top.head(3).items()})
    print("  longs :", {k: round(v, 3) for k, v in top.tail(3).items()})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", action="store_true")
    ap.add_argument("--step", action="store_true")
    ap.add_argument("--update-data", action="store_true")
    ap.add_argument("--no-fetch", action="store_true", help="use cached panel as-is (no network)")
    ap.add_argument("--carry", action="store_true", help="include live funding-carry sleeve (step only)")
    args = ap.parse_args()

    if args.no_fetch and CACHE.exists():
        panel = pd.read_parquet(CACHE).resample("1D").last()
    else:
        print("Updating price panel from Bybit ...", flush=True)
        panel = build_or_update_panel(CACHE).resample("1D").last()
        print(f"Panel: {panel.index.min().date()}..{panel.index.max().date()} | {panel.shape[1]} perps | {len(panel)} days")

    if args.update_data:
        return
    if args.replay:
        run_replay(panel)
    elif args.step:
        run_step(panel, with_carry=args.carry)
    else:
        print("Specify --replay, --step, or --update-data")


if __name__ == "__main__":
    main()
