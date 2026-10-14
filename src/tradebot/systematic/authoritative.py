"""De eindkandidaat door de authoritative engine (`backtest/engine.py`).

Het screeningsboek (`book.py`) is `NOT_ADMISSIBLE_AS_PROMOTION_EVIDENCE`. Promotiebewijs
komt uitsluitend hieruit: dezelfde doelgewichten als `a_t`, door de soevereine
`RiskEngine` (vol-doel, concentratie-, cluster-, ADV-, bruto- en netto-limieten,
drawdown-breaker) en de `OrderRouter` (fees, spread, impact, participatielimiet,
funding), met `latency_bars = 1`: het besluit op de close van *t* vult op de close
van *t+1*. Dat is de pessimistische timing; het screeningsboek met `lag = 2` hoort
er per constructie het dichtst bij te liggen.

    python -m tradebot.systematic.authoritative            # na de poortlezing
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..backtest.engine import build_slices, exposures_from_frame
from ..backtest.phase5_baseline import build_engine
from ..execution.order_router import SpreadModel, SpreadStatus, VenueSpec
from ..execution.trade_costs import load_impact_params
from ..schemas.config import BacktestConfig, ExecutionConfig, RiskConfig, load_config
from ..schemas.robust_book import robust_book_config
from ..utils.failfast import DataContractError, require
from .book import BookResult
from .evaluate import summarize, yearly
from .market import BARS_PER_YEAR
from .programme import (
    ARTEFACT_DIR,
    LOCK_PATH,
    ROOT,
    build_trial,
    cost_spec,
    load_market,
    run_targets,
)

__all__ = ["run_authoritative"]


def _frame_from_engine(result: Any, index: pd.DatetimeIndex) -> pd.DataFrame:
    net = result.returns.reindex(index).fillna(0.0)
    return pd.DataFrame({"net": net, "gross": net, "funding": 0.0, "fees": 0.0, "spread": 0.0,
                         "slippage": 0.0, "impact": 0.0, "turnover": 0.0,
                         "gross_leverage": 0.0, "net_leverage": 0.0, "n_trades": 0.0},
                        index=index)


def run_authoritative(root: Path = ROOT) -> dict[str, Any]:
    cfg = robust_book_config(root / "conf/model/robust_book.yaml")
    gate = json.loads((root / ARTEFACT_DIR / "gate_read.json").read_text(encoding="utf-8"))
    lock = json.loads((root / LOCK_PATH).read_text(encoding="utf-8"))
    require(any(r["hypothesis_id"] == gate["hypothesis_id"] for r in lock["reads"]),
            "De authoritative run hoort na de geregistreerde poortlezing.", DataContractError)
    name = gate["candidate"]
    included = tuple(gate["included"])

    market = load_market(root, cfg)
    target = build_trial(name, market, cfg, included=included)
    live = market.live(cfg.sizing.min_history_bars)
    usable = live.any(axis=1)
    first = usable.idxmax()
    idx = market.index[market.index >= first]

    risk_cfg = load_config(root / "conf/risk/default.yaml", RiskConfig)
    exe = load_config(root / "conf/execution/fees.yaml", ExecutionConfig)
    bt = load_config(root / "conf/backtest/default.yaml", BacktestConfig)
    venue = VenueSpec(maker_fee_bps=exe.maker_fee_bps, taker_fee_bps=exe.taker_fee_bps,
                      funding_cap_abs=0.02, min_notional=10.0, latency_bars=1)
    spread = SpreadModel(half_spread_bps=exe.assumed_half_spread_bps,
                         status=SpreadStatus.SPREAD_ASSUMED, source="conf/execution/fees.yaml")
    engine = build_engine(risk_cfg, load_impact_params(root / "conf/execution/impact.yaml"),
                          venue, spread, initial_equity=float(bt.initial_equity))

    # Alleen munten die live zijn, krijgen een slice-waarde; de rest valt uit de slice.
    marks = market.close.where(live).loc[idx]
    sigma_d = market.sigma_daily.loc[idx]
    turnover = market.adv_usd.loc[idx]  # bar-volume ~ ADV op dagbasis; alleen voor de participatielimiet
    slices = build_slices(
        marks=marks, sigma_hat=sigma_d * np.sqrt(BARS_PER_YEAR), sigma_daily=sigma_d,
        adv_notional=market.adv_usd.loc[idx], bar_volume_notional=turnover,
        clusters=dict(risk_cfg.clusters), funding_rate=market.funding.loc[idx])
    weights = target.weights.loc[idx].where(live.loc[idx])
    exposures = exposures_from_frame(weights.where(target.rebalance.loc[idx], np.nan, axis=0))
    result = engine.run(slices, exposures)

    # Het screeningsboek op dezelfde bars, lag 1 en lag 2, voor de vergelijking.
    costs = cost_spec(root, cfg)
    screen = {lag: run_targets(target, market, costs, lag=lag) for lag in (1, 2)}
    eng = BookResult(frame=_frame_from_engine(result, market.index),
                     held=screen[2].held, trades=screen[2].trades)

    windows = {
        "w_dev": (pd.Timestamp(cfg.windows.w_dev_start, tz="UTC"),
                  pd.Timestamp(cfg.windows.w_dev_end, tz="UTC")),
        "gate": (pd.Timestamp(cfg.windows.gate_start, tz="UTC"), market.index[-1]),
    }
    out: dict[str, Any] = {
        "candidate": name, "included": list(included), "engine": "backtest.engine.EventDrivenEngine",
        "counts_as_promotion_evidence": True, "risk_policy_hash": result.risk_policy_hash,
        "impact_status": result.impact_status, "spread_status": result.spread_status,
        "audit": result.as_record(), "windows": {}}
    for label, (a, b) in windows.items():
        e = summarize(eng.window(a, b))
        s1 = summarize(screen[1].window(a, b))
        s2 = summarize(screen[2].window(a, b))
        diff = (eng.window(a, b).net - screen[2].window(a, b).net)
        out["windows"][label] = {
            "authoritative": {k: e[k] for k in ("sharpe", "sharpe_se", "cagr", "ann_vol",
                                                "max_drawdown", "sortino", "calmar",
                                                "sharpe_ci_low", "sharpe_ci_high")},
            "screen_lag1": {k: s1[k] for k in ("sharpe", "cagr", "ann_vol", "max_drawdown")},
            "screen_lag2": {k: s2[k] for k in ("sharpe", "cagr", "ann_vol", "max_drawdown")},
            "per_bar_rmse_vs_screen_lag2_bp": float(np.sqrt((diff ** 2).mean()) * 1e4),
            "correlation_vs_screen_lag2": float(np.corrcoef(
                eng.window(a, b).net, screen[2].window(a, b).net)[0, 1]),
            "yearly_authoritative": yearly(eng.window(a, b)),
        }
    (root / ARTEFACT_DIR / "authoritative.json").write_text(
        json.dumps(out, indent=2, sort_keys=True, default=float, ensure_ascii=False),
        encoding="utf-8")
    return out


if __name__ == "__main__":
    require(len(sys.argv) == 1, "Geen argumenten.", DataContractError)
    print(json.dumps(run_authoritative(), indent=2, default=float))
