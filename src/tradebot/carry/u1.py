"""Het Kraken-boek: v9-U1 (de regels van v5-H3, BTC en ETH) op de koersen en funding van Kraken.

    python -I -m tradebot.carry.u1 shadow [--start 2026-10-01] [--out DIR]
    python -I -m tradebot.carry.u1 step   --state S --capital-usd N [--live]
    python -I -m tradebot.carry.u1 status --state S
    python -I -m tradebot.carry.u1 unwind --state S --reason TEKST [--live]

WAT DIT IS. Het enige bevroren boek dat op Kraken EU kan draaien (`carry.venues`): long
spot BTC/ETH als onderpand in de futureswallet, short de perp, alleen waar de carry het
waard is (instap ≥ 15 %/jaar, uitstap < 5 %, twee slots van 35 %). Wat het boek niet
gebruikt, hoort in een geldmarktfonds (v9); dat doet de eigenaar, het boek zegt wanneer.

* `shadow` rekent het boek op de publieke Kraken-data (`data.kraken_feed`) met de fees van
  Kraken en de T-bill van FRED, en rapporteert vanaf het begin van het vooruit-sample
  (2026-10-01). Geen sleutel, geen order. Draait dagelijks op Actions
  (`.github/workflows/kraken-shadow.yml`).
* `step` handelt één nieuwe afgesloten bar: eerst de beurs reconciliëren tegen wat het
  boek na de vorige stap verwachtte (een breuk stopt het boek), dan PRECIES de beslisregel
  van de backtest (`systematic.harvest.basis_decide`) op wat de beurs werkelijk houdt, dan
  orders in de margeveilige volgorde. Standaard dry-run (de adapter stuurt niets);
  `--live` met sleutels in de omgeving (`KRAKEN_SPOT_KEY`, `KRAKEN_SPOT_SECRET`,
  `KRAKEN_FUTURES_KEY`, `KRAKEN_FUTURES_SECRET`).

DE KAS. Het boek rekent in fracties van het BOEKKAPITAAL (`--capital-usd`: beurs plus
fonds), niet van wat er op de beurs staat. Vraagt een besluit meer dan de beurs heeft, dan
handelt de stap niet en meldt hij hoeveel er naar de beurs moet (`funding_request_usd`); het
besluit valt opnieuw op de volgende bar. Heeft het boek een week niets nodig gehad, dan
meldt hij wat er naar het fonds kan (`sweep_out_usd`). Een overboeking vertraagt een
instap dus met een dag of twee; de batterij van v5 meet dat (Sharpe bij lag 2 / basis 0,99).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..data import external_series as ex
from ..data import kraken_feed as kf
from ..schemas.robust_book_v2 import RobustBookV5Config, robust_book_v5_config
from ..systematic import programme_v5 as v5
from ..systematic import programme_v9 as v9
from ..systematic.book import BookResult
from ..systematic.harvest import (
    BasisCosts,
    BasisMarket,
    basis_decide,
    carry_estimate,
    harvest_targets,
    run_basis,
)
from ..utils.failfast import DataContractError, require
from ..utils.hashing import hash_config
from .kraken import KrakenVenue
from .orders import LegOrder, orders_for, unwind_orders
from .runner import after_fills
from .venue import LegFill, ReconciliationBreak, Venue, reconcile_book
from .venues import basis_costs, load_venues

__all__ = ["FORWARD_START", "U1Context", "U1State", "build_context", "load_u1_state", "main",
           "render", "save_u1_state", "shadow", "simulate", "step", "unwind"]

ROOT = v9.ROOT
FORWARD_START = "2026-10-01"
#: Hoeveel `committed`-waarden de staat bewaart: genoeg voor de vrijgaveweek van v9.
HISTORY = 30


# --------------------------------------------------------------------------- #
# Het boek op een markt
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class U1Context:
    v5: RobustBookV5Config
    v9: v9.V9Config
    market: BasisMarket
    costs: BasisCosts
    target: pd.DataFrame
    config_hash: str
    spot_px: np.ndarray
    perp_px: np.ndarray

    @property
    def symbols(self) -> tuple[str, ...]:
        return self.market.symbols


def build_context(root: Path, market: BasisMarket) -> U1Context:
    """De regels van H3 (v5-config), de kosten van Kraken (v9-config) op `market`."""
    c5 = robust_book_v5_config(root / v5.CONFIG)
    c9 = v9.load_config(root / v9.CONFIG)
    venue = load_venues(root / "conf/execution/venues.yaml")[c9.venue.name]
    costs = basis_costs(v5.costs_for(root, c5), venue, capital_usd=c9.venue.capital_usd,
                        maker=c9.venue.maker)
    b = c5.basis
    target = harvest_targets(market, span=b.funding_span, enter_apr=b.enter_apr,
                             exit_apr=b.exit_apr, slots=len(b.majors), notional=b.notional,
                             min_spot_adv_usd=b.min_spot_adv_usd,
                             max_abs_basis=b.max_abs_basis, symbols=b.majors)
    return U1Context(
        v5=c5, v9=c9, market=market, costs=costs, target=target,
        config_hash=hash_config({"basis": b.model_dump(mode="json"), "v9": asdict(c9),
                                 "costs": costs.as_record()}),
        spot_px=market.spot_close.ffill().to_numpy(dtype=np.float64),
        perp_px=market.perp.book.close.ffill().to_numpy(dtype=np.float64))


def simulate(ctx: U1Context) -> BookResult:
    """De backtest van het boek op deze markt: `run_basis`, zoals v5 hem draait."""
    b = ctx.v5.basis
    return run_basis(ctx.target, ctx.market, ctx.costs, lag=ctx.v5.execution.lag_bars,
                     band=b.band, hedge_tolerance=b.hedge_tolerance,
                     maintenance_margin=b.maintenance_margin, margin_floor=b.margin_floor)


# --------------------------------------------------------------------------- #
# Shadow
# --------------------------------------------------------------------------- #
def _f(x: float) -> float | None:
    return float(x) if math.isfinite(x) else None


def shadow(ctx: U1Context, dtb3: pd.Series, *, start: str = FORWARD_START) -> dict[str, Any]:
    """Het boek op Kraken, met de kasrente, en wat het vanaf `start` deed en nu wil."""
    frame = v9.cash_overlay(simulate(ctx).frame, dtb3, ctx.v9)
    s0 = pd.Timestamp(start, tz="UTC")
    fwd = frame.loc[s0:]
    carry = carry_estimate(ctx.market, ctx.v5.basis.funding_span).iloc[-1]
    last = ctx.market.index[-1]

    def total(col: str) -> float:
        return float((1.0 + fwd[col]).prod() - 1.0) if len(fwd) else 0.0

    yearly = {str(y): {"return": float((1.0 + f["net"]).prod() - 1.0),
                       "over_t_bill": float((1.0 + f["excess"]).prod() - 1.0),
                       "n_days": len(f)} for y, f in frame.groupby(frame.index.year)}
    return {
        "as_of": last.isoformat(), "forward_start": s0.isoformat(), "forward_days": len(fwd),
        "forward": {"return": total("net"), "over_t_bill": total("excess"),
                    "book_only": float((fwd["net"] - fwd["cash"]).sum()),
                    "cash": float(fwd["cash"].sum()), "t_bill": float(fwd["rf"].sum()),
                    "max_drawdown": float((1.0 - (1.0 + fwd["net"]).cumprod()
                                           / (1.0 + fwd["net"]).cumprod().cummax()).max())
                    if len(fwd) else 0.0},
        "kraken_history": {"first_bar": frame.index[0].isoformat(), "yearly": yearly},
        "now": {"carry_apr": {s: _f(carry[s]) for s in ctx.symbols},
                "enter_apr": ctx.v5.basis.enter_apr, "exit_apr": ctx.v5.basis.exit_apr,
                "target": {s: float(ctx.target[s].iloc[-1]) for s in ctx.symbols},
                "spot_notional": float(frame["spot_notional"].iloc[-1]),
                "cash_fraction": float(frame["cash_fraction"].iloc[-1])},
        "daily": fwd[["net", "cash", "rf", "excess", "spot_notional", "cash_fraction"]],
    }


def render(rep: Mapping[str, Any]) -> str:
    """Het shadowrapport als Markdown (de samenvatting van de Actions-run)."""
    f, now = rep["forward"], rep["now"]
    lines = [
        "# Kraken-boek (v9-U1), shadow",
        "",
        f"Laatste afgesloten bar: **{rep['as_of'][:10]}**. Vooruit-sample sinds "
        f"{rep['forward_start'][:10]} ({rep['forward_days']} dagen).",
        "",
        "| vooruit | waarde |",
        "| --- | --- |",
        f"| rendement (boek + kas) | {f['return']:+.3%} |",
        f"| boven de T-bill | {f['over_t_bill']:+.3%} |",
        f"| waarvan carryboek | {f['book_only']:+.3%} |",
        f"| waarvan kasrente | {f['cash']:+.3%} |",
        f"| max drawdown | {f['max_drawdown']:.3%} |",
        "",
        "| munt | carry (30d, per jaar) | doel (fractie) |",
        "| --- | --- | --- |",
    ]
    for s, c in now["carry_apr"].items():
        lines.append(f"| {s} | {'—' if c is None else f'{c:.1%}'} | {now['target'][s]:.2f} |")
    status = ("in carry" if now["spot_notional"] > 0.0 else
              f"geen carry boven {now['enter_apr']:.0%}: het boek staat in kas")
    lines += ["", f"Stand: **{status}** (in het fonds: {now['cash_fraction']:.0%}).", "",
              "| jaar (Kraken-data) | rendement | boven de T-bill |", "| --- | --- | --- |"]
    for y, r in rep["kraken_history"]["yearly"].items():
        lines.append(f"| {y} ({r['n_days']} d) | {r['return']:+.2%} | {r['over_t_bill']:+.2%} |")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# De staat tussen twee live-runs
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class U1State:
    config_hash: str
    mode: str
    capital_usd: float
    last_bar: str | None = None
    expected_spot: Mapping[str, float] = field(default_factory=dict)
    expected_perp: Mapping[str, float] = field(default_factory=dict)
    committed: Sequence[float] = ()
    halted: str | None = None

    def as_record(self) -> dict[str, Any]:
        return {**asdict(self), "expected_spot": dict(self.expected_spot),
                "expected_perp": dict(self.expected_perp), "committed": list(self.committed)}


def save_u1_state(state: U1State, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state.as_record(), indent=1, sort_keys=True) + "\n",
                   encoding="utf-8")
    tmp.replace(path)


def load_u1_state(path: Path) -> U1State:
    raw = json.loads(path.read_text(encoding="utf-8"))
    require(set(raw) == {f for f in U1State.__dataclass_fields__}, "Een U1-staat met andere "
            "velden.", DataContractError, got=sorted(raw))
    return U1State(**{**raw, "committed": tuple(raw["committed"])})


# --------------------------------------------------------------------------- #
# Live
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _Decision:
    """`carry.orders.Deltas`: wat er per munt bij of af moet, als fractie van `equity_pre`."""

    da: np.ndarray
    db: np.ndarray
    equity_pre: float


def _committed(new_a: np.ndarray, c9: v9.V9Config) -> float:
    return float(min(1.0, float(new_a.sum()) / c9.base.spot_leg))


def step(ctx: U1Context, state: U1State, venue: Venue) -> tuple[dict[str, Any], U1State]:
    """Eén nieuwe afgesloten bar (zie de moduledocstring). Geen nieuwe bar: niets."""
    require(state.halted is None, "Het boek staat stil; hervat alleen na onderzoek.",
            DataContractError, halted=state.halted)
    require(state.config_hash == ctx.config_hash, "De staat hoort bij een andere config.",
            DataContractError)
    t = len(ctx.market.index) - 1
    bar = ctx.market.index[t]
    if state.last_bar is not None and bar <= pd.Timestamp(state.last_bar):
        return {"bar": bar.isoformat(), "action": "geen nieuwe bar"}, state
    vb = venue.book()
    prices = dict(zip(ctx.symbols, ctx.spot_px[t], strict=True))
    breaks = reconcile_book(state.expected_spot, state.expected_perp, vb, prices=prices)
    if breaks:
        raise ReconciliationBreak("; ".join(breaks))
    cap = float(state.capital_usd)
    a_d = np.array([vb.spot.get(s, 0.0) * ctx.spot_px[t, i] / cap
                    for i, s in enumerate(ctx.symbols)])
    b_d = np.array([vb.perp_short.get(s, 0.0) * ctx.perp_px[t, i] / cap
                    for i, s in enumerate(ctx.symbols)])
    both = (ctx.market.perp.book.close.iloc[t].notna()
            & ctx.market.spot_close.iloc[t].notna()).to_numpy()
    b = ctx.v5.basis
    new_a, _, da, db = basis_decide(a_d, b_d, ctx.target.iloc[t].to_numpy(dtype=np.float64),
                                    both, skip=False, band=b.band,
                                    hedge_tolerance=b.hedge_tolerance,
                                    margin_floor=b.margin_floor)
    committed = _committed(new_a, ctx.v9)
    need = cap * min(1.0, committed + ctx.v9.cash.operating_buffer)
    orders: list[LegOrder] = orders_for(_Decision(da, db, cap), ctx.symbols, ctx.spot_px[t],
                                        ctx.perp_px[t])
    grows = any(not o.reduce_only for o in orders)
    short = max(0.0, need - float(vb.equity_usd))
    fills: list[LegFill] = []
    if grows and short > 1e-6 * cap:
        action = "wacht op overboeking"
        spot, perp = dict(vb.spot), dict(vb.perp_short)
    else:
        fills = venue.execute(orders)
        action = "gehandeld" if fills else "geen handel nodig"
        spot, perp = after_fills(vb.spot, vb.perp_short, fills)
    hist = (tuple(state.committed) + (committed,))[-HISTORY:]
    week = hist[-(ctx.v9.cash.release_after_days + 1):]
    idle = len(week) > ctx.v9.cash.release_after_days and max(week) == 0.0
    sweep = max(0.0, float(vb.equity_usd) - cap * ctx.v9.cash.operating_buffer) if idle else 0.0
    record = {
        "bar": bar.isoformat(), "mode": state.mode, "action": action,
        "exchange_equity_usd": float(vb.equity_usd), "capital_usd": cap,
        "target": {s: float(v) for s, v in zip(ctx.symbols, ctx.target.iloc[t], strict=True)},
        "spot_before": dict(vb.spot), "perp_before": dict(vb.perp_short),
        "orders": [o.as_record() for o in orders],
        "fills": [{"symbol": f.order.symbol, "leg": f.order.leg.value, "qty": f.qty,
                   "price": f.price, "fee_usd": f.fee_usd} for f in fills],
        "committed": committed, "funding_request_usd": short if action.startswith("wacht")
        else 0.0, "sweep_out_usd": sweep}
    return record, replace(state, last_bar=bar.isoformat(), expected_spot=spot,
                           expected_perp=perp, committed=hist)


def unwind(ctx: U1Context, state: U1State, venue: Venue, *,
           reason: str) -> tuple[list[LegFill], U1State]:
    """De kill switch: gehedged plat (eerst de shorts, dan de spot), en de staat op stil."""
    require(bool(reason.strip()), "Een unwind zonder reden.", DataContractError)
    vb = venue.book()
    t = len(ctx.market.index) - 1
    fills = venue.execute(unwind_orders(
        vb.spot, vb.perp_short, dict(zip(ctx.symbols, ctx.spot_px[t], strict=True)),
        dict(zip(ctx.symbols, ctx.perp_px[t], strict=True))))
    return fills, replace(state, expected_spot={}, expected_perp={}, halted=reason)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _venue(ctx: U1Context, state: U1State, *, live: bool,
           http: Any | None) -> KrakenVenue:
    kw: dict[str, Any] = {} if http is None else {"http": http}
    if not live:
        # Dry-run: de adapter rondt af en legt vast, met de stand die het boek verwacht.
        return KrakenVenue(ctx.symbols, dry_run=True, dry_equity_usd=state.capital_usd,
                           dry_book={"spot": dict(state.expected_spot),
                                     "perp": dict(state.expected_perp)}, **kw)
    keys = ("KRAKEN_SPOT_KEY", "KRAKEN_SPOT_SECRET", "KRAKEN_FUTURES_KEY",
            "KRAKEN_FUTURES_SECRET")
    missing = [k for k in keys if not os.environ.get(k)]
    require(not missing, "Live zonder Kraken-sleutels in de omgeving.", DataContractError,
            missing=missing)
    return KrakenVenue(ctx.symbols, spot_key=os.environ[keys[0]], spot_secret=os.environ[keys[1]],
                       futures_key=os.environ[keys[2]], futures_secret=os.environ[keys[3]],
                       dry_run=False, **kw)


def main(argv: Sequence[str] | None = None, *, root: Path = ROOT,
         get: kf.Get | None = None, http: Any | None = None,
         now: pd.Timestamp | None = None) -> dict[str, Any]:
    ap = argparse.ArgumentParser(prog="python -I -m tradebot.carry.u1")
    ap.add_argument("command", choices=("shadow", "step", "status", "unwind"))
    ap.add_argument("--state", type=Path, default=None)
    ap.add_argument("--capital-usd", type=float, default=None)
    ap.add_argument("--start", default=FORWARD_START)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--reason", default=None)
    ap.add_argument("--live", action="store_true")
    args = ap.parse_args(argv)
    if args.command == "status":
        require(args.state is not None, "status vraagt --state.", DataContractError)
        out: dict[str, Any] = load_u1_state(args.state).as_record()
        print(json.dumps(out, sort_keys=True))
        return out
    fetch: Callable[..., Any] = kf._get if get is None else get
    panels = kf.fetch_panels(fetch, now=now)
    ctx = build_context(root, kf.kraken_market(panels, robust_book_v5_config(root / v5.CONFIG)))
    if args.command == "shadow":
        dtb3 = ex.fetch_dtb3(panels.last.tz_convert(None), fetch)
        rep = shadow(ctx, dtb3, start=args.start)
        md = render(rep)
        if args.out is not None:
            args.out.mkdir(parents=True, exist_ok=True)
            (args.out / "summary.md").write_text(md, encoding="utf-8")
            rep["daily"].to_csv(args.out / "shadow_daily.csv", float_format="%.10g",
                                lineterminator="\n")
            (args.out / "report.json").write_text(json.dumps(
                {k: v for k, v in rep.items() if k != "daily"}, indent=1, sort_keys=True)
                + "\n", encoding="utf-8")
        print(md)
        return {k: v for k, v in rep.items() if k != "daily"}
    require(args.state is not None, f"{args.command} vraagt --state.", DataContractError)
    mode = "live" if args.live else "dry_run"
    if args.state.is_file():
        state = load_u1_state(args.state)
        require(state.mode == mode, "Een staat wisselt niet van modus; begin een nieuwe staat "
                "voor live.", DataContractError, state=state.mode, run=mode)
    else:
        require(args.command == "step" and args.capital_usd is not None
                and args.capital_usd > 0.0, "Een nieuwe staat vraagt step met "
                "--capital-usd.", DataContractError)
        state = U1State(config_hash=ctx.config_hash, mode=mode, capital_usd=args.capital_usd)
    venue = _venue(ctx, state, live=args.live, http=http)
    if args.command == "step":
        out, state = step(ctx, state, venue)
    else:
        require(bool(args.reason), "unwind vraagt een --reason.", DataContractError)
        fills, state = unwind(ctx, state, venue, reason=args.reason)
        out = {"closed_legs": len(fills), "halted": state.halted}
    save_u1_state(state, args.state)
    print(json.dumps(out, sort_keys=True, default=float))
    return out


if __name__ == "__main__":
    main(sys.argv[1:])
