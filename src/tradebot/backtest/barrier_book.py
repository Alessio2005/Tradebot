"""Een tradeboek met stops en take-profits bij de exchange (spec §10-§12, §17.6).

WAAROM NIET `backtest/engine.py`
================================
De authoritative engine vult orders op de prijs van een bar. Een 1:1-trade met
stops bij de exchange sluit op het BARRIÈRENIVEAU, midden in een bar; dat kan de
engine niet uitdrukken zonder zijn fillmodel te veranderen. Dit boek gebruikt
wel dezelfde soevereine risicolaag (`RiskEngine.decide`), dezelfde
kostenparameters en `square_root_impact`. `tests/unit/test_barrier_book.py`
bewijst de boekhouding (P&L, kosten, funding sluiten op de equity) en de
causaliteit (een exit wordt pas op zijn exitbar gelezen).

DE VOLGORDE OP BAR d
====================
1. Mark-to-market van close d-1 naar close d, of naar de fill als de trade op d
   sluit. Funding over bar d op de positie van d-1 (long betaalt positieve funding).
2. Nieuwe trades met entry op d (event op d-1): toegelaten als p >= p_trade en
   het symbool geen open positie heeft; grootte uit binaire Kelly of een vaste
   risicofractie, maal de correlatiecorrectie.
3. Het gewenste boek gaat door `RiskEngine.decide`; toegestane exposures worden
   posities, tegen kosten. Bestaande posities worden alleen herschaald buiten
   `resize_band`, of naar nul.

WAT `desired` IS
================
Voor een open positie is `desired` haar BEDOELDE exposure (de grootte bij entry), niet
haar huidige, al teruggeschaalde gewicht. Het huidige gewicht terugvoeren laat elke
schaling van de risicolaag (drawdown-breaker, vol-target) elke dag opnieuw op zichzelf
werken: bij een breakerfactor van 0,6 is de positie na ~25 dagen numeriek stof (~1e-10)
en crasht de engine terecht op een concentratie die door dat stof afdrijft. Gemeten op
de synthetische campagnetest. `test_a_long_drawdown_does_not_shrink_a_position_to_dust`
bewaakt dit.

RISICO-EXITS
============
Zet `decide` een open positie op nul (halt, drawdown, limiet), dan sluit zij op
de close van d. Dat is geen barrière-exit: de trade krijgt `risk_exit_bar = d` en
zijn werkelijke `realized_return`. Wie hitrate of R-multiples uit `trades` rekent,
gebruikt `realized_return` en niet `fill_return`; die laatste is het rendement
dat de barrière zou hebben opgeleverd.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from ..data.weekly_market import WeeklyMarket
from ..execution.impact_model import ImpactParams, square_root_impact
from ..risk.binary_kelly import correlation_scale, kelly_fraction_binary
from ..risk.contract import MarketState, RiskState
from ..risk.engine import RiskEngine
from ..utils.failfast import DataContractError, require

__all__ = ["CANDIDATE_COLUMNS", "BookInputs", "BookResult", "SizingRule", "run_barrier_book"]

CANDIDATE_COLUMNS = ("symbol", "entry_bar", "exit_bar", "side", "fill_return",
                     "barrier", "p", "p_low", "p_trade")

#: `risk_exit_bar` van een trade die op zijn barrière of verticale grens sloot.
NO_RISK_EXIT = -1


@dataclass(frozen=True)
class SizingRule:
    mode: Literal["kelly", "fixed"]
    kelly_multiple: float
    fixed_risk_fraction: float
    resize_band: float
    #: De round trip die Kelly als kosten ziet (fractie).
    cost_rt: float


@dataclass(frozen=True)
class BookInputs:
    market: WeeklyMarket
    avg_corr: pd.DataFrame
    #: Kosten per kant: taker + halve spread, als fractie van het notioneel.
    cost_rate: float
    impact: ImpactParams | None


@dataclass(frozen=True)
class BookResult:
    returns: pd.Series
    equity: pd.Series
    trades: pd.DataFrame
    total_fees: float
    total_funding: float
    total_impact: float
    total_pnl: float
    n_resizes: int
    halted: bool


class _Ledger:
    def __init__(self, equity: float) -> None:
        self.equity = equity
        self.fees = self.funding = self.impact = self.pnl = 0.0

    def trade_cost(self, notional: float, inputs: BookInputs, adv: float, sigma: float) -> None:
        fee = abs(notional) * inputs.cost_rate
        imp = 0.0
        if inputs.impact is not None and abs(notional) > 0.0:
            imp = square_root_impact(order_notional=abs(notional), adv_notional=adv,
                                     sigma_daily=sigma, params=inputs.impact).cost(abs(notional))
        self.fees += fee
        self.impact += imp
        self.equity -= fee + imp


def _base_fraction(row: pd.Series, sizing: SizingRule) -> float:
    if sizing.mode == "fixed":
        return sizing.fixed_risk_fraction
    return sizing.kelly_multiple * kelly_fraction_binary(float(row.p_low), float(row.barrier),
                                                         sizing.cost_rt)


def run_barrier_book(
    candidates: pd.DataFrame,
    inputs: BookInputs,
    risk: RiskEngine,
    sizing: SizingRule,
    *,
    equity0: float,
) -> BookResult:
    """Speel de kandidaten af in tijdvolgorde en geef het dagelijkse boekrendement."""
    missing = sorted(set(CANDIDATE_COLUMNS) - set(candidates.columns))
    require(not missing, "Kandidaten missen kolommen.", DataContractError, missing=missing)
    m = inputs.market
    cols = {s: j for j, s in enumerate(m.symbols)}
    close = pd.DataFrame({s: m.ohlcv[s]["close"] for s in m.symbols}).to_numpy(np.float64)
    fund = m.funding.reindex(columns=list(m.symbols)).to_numpy(np.float64)
    sig_a = m.sigma_annual.reindex(columns=list(m.symbols)).to_numpy(np.float64)
    sig_d = m.sigma_daily.reindex(columns=list(m.symbols)).to_numpy(np.float64)
    adv = m.adv_usd.reindex(columns=list(m.symbols)).to_numpy(np.float64)
    corr = inputs.avg_corr.reindex(index=m.grid, columns=list(m.symbols)).to_numpy(np.float64)
    cand = candidates.reset_index(drop=True)
    by_entry: dict[int, list[int]] = {}
    for i, e in enumerate(cand["entry_bar"].to_numpy(dtype=np.int64)):
        by_entry.setdefault(int(e), []).append(i)

    n = len(m.grid)
    book = _Ledger(float(equity0))
    state = RiskState(equity=book.equity, high_water_mark=book.equity,
                      day_start_equity=book.equity)
    qty: dict[str, float] = {}
    open_row: dict[str, int] = {}
    entry_px: dict[str, float] = {}
    trade_idx: dict[str, int] = {}
    intent: dict[str, float] = {}
    trades: list[dict[str, Any]] = []
    returns = np.zeros(n)
    equity = np.full(n, float(equity0))
    resizes = 0

    for d in range(1, n):
        start = book.equity
        # 1. Mark-to-market, funding en exits op de barrière of de verticale close.
        for sym in list(qty):
            j, q = cols[sym], qty[sym]
            prev = close[d - 1, j]
            row = cand.loc[open_row[sym]]
            if int(row.exit_bar) == d:
                fill = entry_px[sym] * (1.0 + float(row.fill_return) * float(row.side))
                move = q * (fill - prev)
                book.trade_cost(q * fill, inputs, adv[d, j], sig_d[d, j])
                trades[trade_idx.pop(sym)]["realized_return"] = float(row.fill_return)
                del qty[sym], open_row[sym], entry_px[sym], intent[sym]
            else:
                move = q * (close[d, j] - prev)
            book.pnl += move
            book.equity += move
            if np.isfinite(fund[d, j]):
                paid = q * prev * fund[d, j]
                book.funding += paid
                book.equity -= paid
        if state.halted:
            returns[d] = (book.equity - start) / start
            equity[d] = book.equity
            continue

        # 2. Het gewenste boek: bestaande posities plus toegelaten nieuwe trades.
        desired = dict(intent)
        pending: dict[str, tuple[int, float]] = {}
        chosen = [i for i in by_entry.get(d, [])
                  if cand.loc[i].symbol not in qty
                  and float(cand.loc[i].p) >= float(cand.loc[i].p_trade)
                  and np.isfinite(adv[d, cols[cand.loc[i].symbol]])
                  and np.isfinite(sig_a[d, cols[cand.loc[i].symbol]])]
        seen: set[str] = set()
        chosen = [i for i in chosen if not (cand.loc[i].symbol in seen or seen.add(cand.loc[i].symbol))]
        for i in chosen:
            row = cand.loc[i]
            same = (sum(1 for q in qty.values() if np.sign(q) == row.side)
                    + sum(1 for k in chosen if cand.loc[k].side == row.side))
            rho = corr[d - 1, cols[row.symbol]]
            f = _base_fraction(row, sizing) * correlation_scale(same, rho if np.isfinite(rho) else 1.0)
            if f <= 0.0:
                continue
            desired[row.symbol] = float(np.clip(row.side * f / float(row.barrier), -1.0, 1.0))
            pending[row.symbol] = (i, f)
        if desired:
            market = MarketState(asof_ts=m.grid[d],
                                 sigma_hat={s: float(sig_a[d, cols[s]]) for s in desired},
                                 adv_usd={s: float(adv[d, cols[s]]) for s in desired})
            state = dataclasses.replace(state, equity=book.equity,
                                        high_water_mark=max(state.high_water_mark, book.equity),
                                        day_start_equity=start)
            decision = risk.decide(desired, market, state)
            state = decision.risk_state_out
            # 3. Toegestane exposures worden posities.
            for sym in sorted(set(desired) | set(qty)):
                j = cols[sym]
                px = close[d, j]
                target_w = float(decision.permitted_exposure.get(sym, 0.0))
                cur_q = qty.get(sym, 0.0)
                if sym in qty and sym not in pending and target_w != 0.0:
                    cur_w = cur_q * px / book.equity
                    if abs(target_w - cur_w) <= sizing.resize_band * abs(cur_w):
                        continue
                target_q = target_w * book.equity / px
                delta = target_q - cur_q
                if delta == 0.0:
                    continue
                book.trade_cost(delta * px, inputs, adv[d, j], sig_d[d, j])
                if target_q == 0.0:
                    # Een risico-exit: de trade sluit op de close van d, niet op zijn barrière.
                    if sym in trade_idx:
                        t = trades[trade_idx.pop(sym)]
                        t["risk_exit_bar"] = d
                        t["realized_return"] = float(t["side"]) * (px / entry_px[sym] - 1.0)
                    qty.pop(sym, None)
                    open_row.pop(sym, None)
                    entry_px.pop(sym, None)
                    intent.pop(sym, None)
                    continue
                qty[sym] = target_q
                if sym in pending:
                    i, f = pending[sym]
                    open_row[sym] = i
                    entry_px[sym] = px
                    intent[sym] = desired[sym]
                    trade_idx[sym] = len(trades)
                    trades.append({"row": i, "symbol": sym, "entry_bar": d,
                                   "exit_bar": int(cand.loc[i].exit_bar),
                                   "side": float(cand.loc[i].side), "qty": target_q,
                                   "entry_price": px, "weight": target_w,
                                   "risk_fraction": abs(target_w) * float(cand.loc[i].barrier),
                                   "requested_risk_fraction": f,
                                   "fill_return": float(cand.loc[i].fill_return),
                                   "barrier": float(cand.loc[i].barrier),
                                   "p": float(cand.loc[i].p), "p_low": float(cand.loc[i].p_low),
                                   "risk_exit_bar": NO_RISK_EXIT,
                                   "realized_return": float("nan")})
                else:
                    resizes += 1
        returns[d] = (book.equity - start) / start
        equity[d] = book.equity

    return BookResult(
        returns=pd.Series(returns, index=m.grid, name="book_return"),
        equity=pd.Series(equity, index=m.grid, name="equity"),
        trades=pd.DataFrame(trades),
        total_fees=book.fees, total_funding=book.funding, total_impact=book.impact,
        total_pnl=book.pnl, n_resizes=resizes, halted=bool(state.halted),
    )
