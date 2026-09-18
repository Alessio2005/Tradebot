"""paper_multi_sleeve.py — paper-trade runner for the FINAL regime-robust book.

Trades MultiSleeveBook (L/S trend majors + DVOL contrarian + cross-sectional
carry) on the Bybit USDT-perp universe. Daily rebalance, next-bar fills,
explicit taker+slippage costs, futures NAV accounting (reuses PaperBroker).

Validated 2021-2026 (simple returns, 6 bps): Sharpe ~1.07, MaxDD −22% (vs BTC
buy&hold 0.60 / −77%), every calendar year positive (2022 +2% vs market −90%).

Modes:
  --replay      end-to-end dry-run over the cached panel (same code path as live).
  --step        live daily step: refresh data, compute weights from the last
                complete bar, rebalance once, persist. (cron daily for paper.)
  --update-data refresh caches and exit.

State -> artefacts/paper_trade_multisleeve/{equity_curve.jsonl,state.json}.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from apps.paper_neutral_trader import PaperBroker  # reuse the validated broker

from tradebot.alpha.multi_sleeve_book import MultiSleeveBook, MultiSleeveConfig
from tradebot.data.perp_feed import build_or_update_panel

CACHE = _ROOT / "artefacts" / "broad_perp_daily_close.parquet"
DVOL_CACHE = _ROOT / "artefacts" / "dvol_btc.parquet"
FUNDING_CACHE = _ROOT / "artefacts" / "funding_5asset.parquet"
STATE_DIR = _ROOT / "artefacts" / "paper_trade_multisleeve"
INITIAL_EQUITY = 200_000.0
LEVERAGE = 1.5           # RISK DIAL: scales the (naturally light) RP gross.
GROSS_CAP = 1.0          # hard cap on gross exposure (max fully-invested).
# Sharpe (~1.1) and every-year-positive are leverage-invariant; LEVERAGE/GROSS_CAP
# only trade total return against drawdown. Lower them for a calmer DD profile.
COST_BPS = 6.0


def _get(u: str) -> Any:
    if not u.startswith("https://"):
        raise ValueError(
            f"alleen https is toegestaan, kreeg: {u!r}. "
            "Zonder deze controle accepteert uopen ook file:/ en "
            "custom schemes (bandit B310)."
        )
    r = urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"})
    # nosec B310 -- het schema is drie regels hierboven op https vastgezet.
    return json.load(urllib.request.urlopen(r, timeout=25))  # nosec B310


def fetch_dvol_update() -> pd.Series:
    existing = pd.read_parquet(DVOL_CACHE)["dvol"] if DVOL_CACHE.exists() else None
    start = int(existing.index.max().timestamp() * 1000) if existing is not None and len(existing) else int(pd.Timestamp("2021-01-01", tz="UTC").timestamp() * 1000)
    end = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000); cs = start; rows = []
    while cs < end:
        ce = min(cs + 200 * 86400000, end)
        try:
            d = _get(f"https://www.deribit.com/api/v2/public/get_volatility_index_data?currency=BTC&start_timestamp={cs}&end_timestamp={ce}&resolution=86400")["result"]["data"]
        except Exception:
            break
        if not d:
            break
        rows += [(x[0], x[4]) for x in d]; cs = ce + 86400000
    new = pd.Series({pd.to_datetime(t, unit="ms", utc=True): v for t, v in rows})
    dv = new if existing is None else pd.concat([existing, new])
    dv = dv[~dv.index.duplicated()].sort_index().resample("1D").last()
    dv.to_frame("dvol").to_parquet(DVOL_CACHE)
    return dv


def scaled_targets(combined_row: pd.Series) -> pd.Series:
    w = combined_row * LEVERAGE
    g = w.abs().sum()
    if g > GROSS_CAP:
        w = w * (GROSS_CAP / g)
    return w


def _persist(broker: Any, tw: pd.Series) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with open(STATE_DIR / "equity_curve.jsonl", "w", encoding="utf-8") as f:
        for r in broker.records:
            f.write(json.dumps(r) + "\n")
    (STATE_DIR / "state.json").write_text(json.dumps({
        "updated": datetime.now(timezone.utc).isoformat(), "equity": broker.equity,
        "n_positions": len(broker.qty), "positions_qty": broker.qty,
        "latest_target_weights": {k: round(float(v), 5) for k, v in tw.items() if abs(v) > 1e-9},
    }, indent=2))


def run_replay(panel: pd.DataFrame, dvol: pd.Series, funding: pd.DataFrame) -> None:
    nb = MultiSleeveBook(MultiSleeveConfig())
    combined, _ = nb.weight_history(panel, dvol=dvol, funding=funding)
    broker = PaperBroker(INITIAL_EQUITY)
    fund_daily = funding.reindex(panel.index).fillna(0.0) if len(funding.columns) else funding
    for d in combined.index:
        tw = scaled_targets(combined.loc[d])
        if tw.abs().sum() < 1e-9:
            continue
        fr = fund_daily.loc[d] if d in fund_daily.index else None
        broker.step(d, panel.loc[d], tw, fr, gross_lev=1.0, cost_bps=COST_BPS)
    _persist(broker, scaled_targets(combined.iloc[-1]))
    eq = pd.Series({pd.Timestamp(r["ts"]): r["equity"] for r in broker.records})
    ret = eq.pct_change().dropna(); shp = ret.mean() / ret.std() * np.sqrt(365) if ret.std() > 0 else float("nan")
    ddv = (eq / eq.cummax() - 1).min()
    print(f"REPLAY: {len(broker.records)} rebalances | {eq.iloc[0]:,.0f} -> {eq.iloc[-1]:,.0f} "
          f"({(eq.iloc[-1]/eq.iloc[0]-1)*100:+.0f}%) Sharpe={shp:.2f} MaxDD={ddv*100:.1f}%")
    print("per-jaar:", " ".join(f"{y}:{(g.iloc[-1]/g.iloc[0]-1)*100:+.0f}%" for y, g in eq.groupby(eq.index.year)))
    print(f"latest gross_lev={broker.records[-1]['gross_lev']} net_lev={broker.records[-1]['net_lev']} n_pos={broker.records[-1]['n_pos']}")


def run_step(panel: pd.DataFrame, dvol: pd.Series, funding: pd.DataFrame) -> None:
    panel = panel.iloc[:-1] if len(panel) > 1 else panel       # last COMPLETE bar
    nb = MultiSleeveBook(MultiSleeveConfig())
    tw = scaled_targets(nb.target_weights(panel, dvol=dvol, funding=funding))
    broker = PaperBroker(INITIAL_EQUITY)
    sp = STATE_DIR / "state.json"
    if sp.exists():
        st = json.loads(sp.read_text())
        broker.equity = float(st.get("equity", INITIAL_EQUITY))
        broker.qty = {k: float(v) for k, v in st.get("positions_qty", {}).items()}
        ecp = STATE_DIR / "equity_curve.jsonl"
        if ecp.exists():
            broker.records = [json.loads(l) for l in ecp.read_text().splitlines() if l.strip()]
        last = panel.iloc[-1]
        broker.last_px = {s: float(last[s]) for s in broker.qty if s in last.index and pd.notna(last[s])}
    fr = funding.reindex(panel.index).fillna(0.0).iloc[-1] if len(funding.columns) else None
    rec = broker.step(panel.index[-1], panel.iloc[-1], tw, fr, gross_lev=1.0, cost_bps=COST_BPS)
    _persist(broker, tw)
    print(f"STEP @ {rec['ts']}: equity={rec['equity']:,.2f} mtm={rec['mtm']:+,.0f} cost={rec['cost']:,.0f} "
          f"gross_lev={rec['gross_lev']} net_lev={rec['net_lev']} n_pos={rec['n_pos']}")
    t = tw[tw.abs() > 1e-9].sort_values()
    print("  shorts:", {k: round(v, 3) for k, v in t.head(3).items()})
    print("  longs :", {k: round(v, 3) for k, v in t.tail(3).items()})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", action="store_true"); ap.add_argument("--step", action="store_true")
    ap.add_argument("--update-data", action="store_true"); ap.add_argument("--no-fetch", action="store_true")
    args = ap.parse_args()
    if args.no_fetch:
        panel = pd.read_parquet(CACHE).resample("1D").last()
        dvol = pd.read_parquet(DVOL_CACHE)["dvol"]
    else:
        print("Refreshing data ...", flush=True)
        panel = build_or_update_panel(CACHE).resample("1D").last()
        dvol = fetch_dvol_update()
    funding = pd.read_parquet(FUNDING_CACHE) if FUNDING_CACHE.exists() else pd.DataFrame()
    print(f"panel {panel.index.min().date()}..{panel.index.max().date()} ({panel.shape[1]} perps) | "
          f"dvol {dvol.index.max().date()} | funding {funding.shape[1] if len(funding.columns) else 0} assets")
    if args.update_data:
        return
    if args.replay:
        run_replay(panel, dvol, funding)
    elif args.step:
        run_step(panel, dvol, funding)
    else:
        print("Specify --replay, --step, or --update-data")


if __name__ == "__main__":
    main()
