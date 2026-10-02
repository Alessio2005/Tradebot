"""BTC-trend long/flat: één regel, geen parameters die op de holdout zijn gezocht.

REGEL (vastgelegd voor de holdout is gelezen, conf/research/preregistration_trend_btc.yaml):
    signaal_t = close_t > gemiddelde(close_{t-99..t})      (bekend na de close van bar t)
    positie_t = signaal_{t-1}                              (long één eenheid notioneel, of vlak)

De positie op bar t is dus een functie van data tot en met de close van bar t-1. De order wordt op
die close gevuld; elke positiewijziging betaalt één kant (taker + halve spread, uit
`conf/execution/fees.yaml`). Een long betaalt de funding van de bar (positief = long betaalt), zoals
de rest van de repo dat boekt. Geen hefboom, geen vol-target: dat is een tweede knop en dus een
tweede kans om op de dev-sample te passen.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..data.weekly_market import load_weekly_market
from ..features.registry import current_git_sha
from ..labeling.barrier_fills import round_trip_cost
from ..schemas.config import ExecutionConfig, load_config
from ..utils.failfast import DataContractError, require
from .holdout import development_slice, gate_slice

__all__ = ["HYPOTHESIS_ID", "MA_BARS", "summarize", "trend_book", "main"]

HYPOTHESIS_ID = "trend-btc-ma100-v1"
MA_BARS = 100
SYMBOL = "BTCUSDT"
ARTEFACT = "artefacts/governance/trend_btc_holdout.json"


def trend_book(close: pd.Series, funding: pd.Series, *, cost_per_side: float,
               ma_bars: int = MA_BARS) -> pd.DataFrame:
    """Dagelijkse bruto, kosten, funding en netto rendement van de long/flat-regel."""
    require(close.index.is_monotonic_increasing, "close is niet oplopend.", DataContractError)
    ma = close.rolling(ma_bars, min_periods=ma_bars).mean()
    signal = (close > ma).where(ma.notna(), False)
    pos = signal.shift(1, fill_value=False).astype(float)
    ret = close.pct_change().fillna(0.0)
    fund = funding.reindex(close.index).fillna(0.0)
    turnover = pos.diff().abs().fillna(pos.abs())
    out = pd.DataFrame({
        "pos": pos, "gross": pos * ret, "cost": turnover * cost_per_side, "funding": pos * fund,
    })
    out["net"] = out["gross"] - out["cost"] - out["funding"]
    return out


def summarize(net: pd.Series, bench: pd.Series) -> dict[str, Any]:
    """Totaal, jaarlijks rendement, Sharpe en drawdown van een dagreeks; naast de benchmark."""
    def one(x: pd.Series) -> dict[str, float]:
        x = x.astype(float)
        eq = (1.0 + x).cumprod()
        total = float(eq.iloc[-1] - 1.0)
        n = len(x)
        sd = float(x.std())
        return {"n_bars": float(n), "total_return": total,
                "annualised_return": float((1.0 + total) ** (365.0 / n) - 1.0),
                "sharpe": float(x.mean() / sd * np.sqrt(365.0)) if sd > 0 else float("nan"),
                "max_drawdown": float((eq / eq.cummax() - 1.0).min())}
    return {"strategy": one(net), "buy_and_hold": one(bench)}


def main() -> None:
    root = Path.cwd()
    exec_cfg = load_config(root / "conf/execution/fees.yaml", ExecutionConfig)
    cost_per_side = round_trip_cost(exec_cfg) / 2.0
    m = load_weekly_market(root, (SYMBOL,))
    close = m.ohlcv[SYMBOL]["close"]
    book = trend_book(close, m.funding[SYMBOL], cost_per_side=cost_per_side)
    bench = close.pct_change().fillna(0.0)
    lock = root / "artefacts/governance/holdout_lock.json"
    frame = pd.DataFrame({"net": book["net"], "bench": bench, "pos": book["pos"]})
    dev = development_slice(frame, lock_path=lock)
    record: dict[str, Any] = {
        "hypothesis_id": HYPOTHESIS_ID, "rule": f"BTC long boven {MA_BARS}d-gemiddelde, anders vlak",
        "cost_per_side": cost_per_side, "git_sha": current_git_sha(),
        "development": summarize(dev["net"].iloc[MA_BARS:], dev["bench"].iloc[MA_BARS:]),
        "development_window": [str(dev.index[MA_BARS]), str(dev.index[-1])],
    }
    if "--holdout" in sys.argv[1:]:
        hold = gate_slice(frame, lock_path=lock, hypothesis_id=HYPOTHESIS_ID)
        record["holdout"] = summarize(hold["net"], hold["bench"])
        record["holdout_window"] = [str(hold.index[0]), str(hold.index[-1])]
        record["holdout_time_in_market"] = float(hold["pos"].mean())
    (root / ARTEFACT).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n",
                                 encoding="utf-8")
    print(json.dumps({k: record[k] for k in record if k in ("development", "holdout")}, indent=2))


if __name__ == "__main__":
    main()
