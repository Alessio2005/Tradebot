"""De dagelijkse stap van het carryboek: paneel in, besluit, orders, staat uit.

    ctx = load_context(root, panel_root)            # het bevroren v8-boek op dit paneel
    state = init(ctx, start="2026-10-01")           # de backtest tot de start: de beginstand
    records, state = step(ctx, state, PaperVenue()) # elke nieuwe bar, één keer

PARITEIT MET DE BACKTEST
========================
`init` en de paperstap roepen `LeveredBook.step` aan op `lever_arrays` van
`programme_v8.book_spec`: precies wat `programme_v8.simulate` draait. Een shadowboek dat
op dezelfde panelen van `init` tot vandaag stapt, IS de backtest, bar voor bar en bit voor
bit (`tests/unit/test_carry.py`). Het enige wat het toevoegt, is dat elk besluit wordt
vastgelegd op het moment dat het genomen wordt, als concrete orders.

Met een echte venue rekent de stap met wat de beurs houdt (`decide_live`): eerst wordt de
beurs gereconcilieerd tegen wat het boek na de vorige stap verwachtte. Een breuk (een
gemiste fill, een liquidatie, auto-deleveraging, een handmatige ingreep) stopt het boek.

GESETTELDE BARS
===============
De panelen overbruggen datagaten van ≤ 7 bars met de laatste koers
(`breadth.bridge_data_holes`). Dat besluit kent de toekomst: of een gat kort blijft, zie
je pas als de munt terugkomt. Op de rand van een groeiend paneel zou een stap daardoor een
gedwongen exit nemen die de backtest later niet neemt. De shadowstap verwerkt daarom
alleen bars waarna nog minstens `settle` bars in het paneel staan. Een live-datafeed van
de beurs zelf heeft geen bron-gaten en draait met `settle=0`.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..oms.order import OrderSide
from ..schemas.robust_book_v2 import RobustBookV8Config, robust_book_v8_config
from ..systematic import programme_v6 as v6
from ..systematic import programme_v8 as v8
from ..systematic.breadth import MAX_BRIDGED_GAP
from ..systematic.harvest import BasisCosts
from ..systematic.leverage import LeverArrays, LeveredBook, LeverStep, MarginMarket, lever_arrays
from ..systematic.programme_v7 import BookSpec
from ..utils.failfast import DataContractError, require
from ..utils.hashing import hash_config
from .orders import Leg, LegOrder, orders_for, unwind_orders
from .state import CarryState, restore, snapshot
from .venue import LegFill, PaperVenue, ReconciliationBreak, Venue, VenueBook, reconcile_book

__all__ = ["CarryContext", "after_fills", "build_context", "init", "load_context",
           "new_book", "settled_end", "step", "unwind"]


@dataclass(frozen=True)
class CarryContext:
    """Het bevroren boek op één paneel: config, markt, kosten, spec en de voorberekende rijen."""

    cfg: RobustBookV8Config
    candidate: str
    market: MarginMarket
    costs: BasisCosts
    spec: BookSpec
    arrays: LeverArrays
    config_hash: str
    spot_px: np.ndarray
    perp_px: np.ndarray


def build_context(market: MarginMarket, cfg: RobustBookV8Config, costs: BasisCosts, *,
                  candidate: str = v8.CANDIDATES[0]) -> CarryContext:
    spec = v8.book_spec(candidate, market, cfg)
    arrays = lever_arrays(spec.target, market, lag=spec.lag, financing=spec.financing,
                          skip=spec.skip)
    b = market.basis
    return CarryContext(
        cfg=cfg, candidate=candidate, market=market, costs=costs, spec=spec, arrays=arrays,
        config_hash=hash_config({"candidate": candidate, "config": cfg.model_dump(mode="json"),
                                 "costs": costs.as_record()}),
        # De koers waartegen een besluit rekent: de close van de bar, of de laatst bekende
        # (een munt die verdwijnt, wordt gesloten tegen zijn laatste koers, zoals de backtest).
        spot_px=b.spot_close.ffill().to_numpy(dtype=np.float64),
        perp_px=b.perp.book.close.ffill().to_numpy(dtype=np.float64))


def load_context(root: Path, panel_root: Path | None = None, *,
                 config: Path | None = None) -> CarryContext:
    """Het v8-boek op de panelen onder `panel_root` (standaard de bevroren panelen)."""
    cfg = robust_book_v8_config(root / (config or v8.CONFIG))
    market = v6.load_market(root, cfg, panel_root=panel_root)
    return build_context(market, cfg, v6.costs_for(root, cfg))


def new_book(ctx: CarryContext) -> LeveredBook:
    s = ctx.spec
    return LeveredBook(ctx.arrays.symbols, ctx.costs, band=s.band,
                       hedge_tolerance=s.hedge_tolerance, margin=s.margin, adv_cap=s.adv_cap,
                       slice_step=s.slice_step)


def settled_end(ctx: CarryContext, settle: int = MAX_BRIDGED_GAP) -> int:
    """De positie van de laatste bar waarvan de data vastligt (zie de moduledocstring)."""
    require(settle >= 0, "Een negatieve settle-buffer bestaat niet.", DataContractError)
    return len(ctx.arrays.index) - 1 - int(settle)


def _expected(book: LeveredBook, res: LeverStep, ctx: CarryContext,
              t: int) -> tuple[dict[str, float], dict[str, float]]:
    """De hoeveelheden die het boek na de stap op bar t houdt (munten)."""
    spot, perp = {}, {}
    for i, sym in enumerate(book.symbols):
        if book.a[i] > 0.0:
            spot[sym] = float(book.a[i] * res.equity_pre / ctx.spot_px[t, i])
        if book.b[i] > 0.0:
            perp[sym] = float(book.b[i] * res.equity_pre / ctx.perp_px[t, i])
    return spot, perp


def after_fills(spot: Mapping[str, float], perp: Mapping[str, float],
                 fills: Sequence[LegFill]) -> tuple[dict[str, float], dict[str, float]]:
    """Wat de beurs na de fills houdt, gemeten aan de fills zelf (ook voor `carry.u1`)."""
    out = {Leg.SPOT: dict(spot), Leg.PERP: dict(perp)}
    for f in fills:
        book = out[f.order.leg]
        # Spot: kopen = erbij. Perp-short: verkopen = erbij.
        grows = (f.order.side is OrderSide.BUY) == (f.order.leg is Leg.SPOT)
        q = book.get(f.order.symbol, 0.0) + (f.qty if grows else -f.qty)
        if abs(q) > 1e-12:
            book[f.order.symbol] = q
        else:
            book.pop(f.order.symbol, None)
    return out[Leg.SPOT], out[Leg.PERP]


def _record(ctx: CarryContext, t: int, mode: str, res: LeverStep, book: LeveredBook,
            orders: Sequence[LegOrder], fills: Sequence[LegFill]) -> dict[str, Any]:
    row = dict(zip(("gross", "funding", "financing", "liquidation", "fees", "spread",
                    "slippage", "impact", "net", "turnover", "gross_leverage", "net_leverage",
                    "n_trades", "n_held", "spot_notional", "loan", "uni_mmr",
                    "stress_headroom", "governed"), res.row, strict=True))
    return {"bar": ctx.arrays.index[t].isoformat(), "mode": mode, "equity": book.equity,
            "equity_pre": res.equity_pre, "liquidated": res.liquidated,
            **{k: (float(v) if np.isfinite(v) else None) for k, v in row.items()},
            "orders": [o.as_record() for o in orders],
            "fills": [{"symbol": f.order.symbol, "leg": f.order.leg.value, "qty": f.qty,
                       "price": f.price, "fee_usd": f.fee_usd} for f in fills]}


def init(ctx: CarryContext, *, start: str, mode: str = "paper",
         settle: int = MAX_BRIDGED_GAP) -> CarryState:
    """De beginstand op `start`: de backtest over elke bar vóór `start`.

    Het vooruit-sample (`conf/model/robust_book_v8.yaml::forward.start`) begint niet met
    een leeg boek: de backtest draagt zijn posities door. Een shadowboek dat leeg begint,
    zou een ander boek meten dan de vooruit-lezing."""
    require(mode in ("paper", "live"), "Onbekende modus.", DataContractError, mode=mode)
    idx = ctx.arrays.index
    first = int(idx.searchsorted(pd.Timestamp(start, tz="UTC")))
    require(0 < first <= settled_end(ctx, settle) + 1, "De start ligt niet binnen het "
            "gesettelde paneel.", DataContractError, start=start, first=str(idx[0]),
            last=str(idx[-1]))
    book = new_book(ctx)
    for t in range(first):
        res = book.step(ctx.arrays.bar(t))
    spot, perp = _expected(book, res, ctx, first - 1)
    if mode == "live":
        # Een live-boek begint op de beurs, niet in de backtest: wat de backtest op de
        # start houdt, moet eerst worden opgebouwd, en dat doet de eerste live-stap.
        spot, perp = {}, {}
    return snapshot(book, candidate=ctx.candidate, config_hash=ctx.config_hash, mode=mode,
                    last_bar=idx[first - 1], expected_spot=spot, expected_perp=perp)


def _marked(ctx: CarryContext, venue_book: VenueBook, t: int) -> tuple[np.ndarray, np.ndarray]:
    # Een munt die de beurs houdt en het paneel niet kent, breekt al in de reconciliatie
    # (het boek verwacht er niets van): hier telt alleen wat het paneel kent.
    syms = ctx.arrays.symbols
    eq = float(venue_book.equity_usd)
    a_d = np.array([venue_book.spot.get(s, 0.0) * ctx.spot_px[t, i] / eq
                    for i, s in enumerate(syms)])
    b_d = np.array([venue_book.perp_short.get(s, 0.0) * ctx.perp_px[t, i] / eq
                    for i, s in enumerate(syms)])
    return a_d, b_d


def step(ctx: CarryContext, state: CarryState, venue: Venue, *,
         settle: int = MAX_BRIDGED_GAP) -> tuple[list[dict[str, Any]], CarryState]:
    """Verwerk elke gesettelde bar na `state.last_bar`, één keer. Geeft per bar een
    beslisrecord terug, en de nieuwe staat.

    Paper: elke achterstallige bar wordt ingehaald (het boek IS het model). Live: alleen
    precies één nieuwe bar; een live-boek handelt nooit op een oude bar."""
    require(state.halted is None, "Het boek staat stil; hervat alleen na onderzoek.",
            DataContractError, halted=state.halted)
    require(state.candidate == ctx.candidate, "De staat hoort bij een andere kandidaat.",
            DataContractError, state=state.candidate, ctx=ctx.candidate)
    mode = "paper" if venue.is_paper else "live"
    require(state.mode == mode, "Een staat wisselt niet van modus; begin een nieuwe staat "
            "voor live.", DataContractError, state=state.mode, venue=mode)
    idx = ctx.arrays.index
    last = pd.Timestamp(state.last_bar)
    require(last in idx, "De laatste bar van de staat staat niet in het paneel.",
            DataContractError, last_bar=state.last_bar)
    lo, hi = int(idx.get_loc(last)) + 1, settled_end(ctx, settle)
    todo = list(range(lo, hi + 1))
    if mode == "live":
        require(len(todo) <= 1, "Live verwerkt alleen de nieuwste bar; er staan er meer "
                "open (een gemiste run?).", DataContractError, open_bars=len(todo))
    book = new_book(ctx)
    restore(state, book, config_hash=ctx.config_hash)
    records: list[dict[str, Any]] = []
    for t in todo:
        bar = ctx.arrays.bar(t)
        if mode == "live":
            vb = venue.book()
            breaks = reconcile_book(state.expected_spot, state.expected_perp, vb,
                                    prices=dict(zip(book.symbols, ctx.spot_px[t], strict=True)))
            if breaks:
                raise ReconciliationBreak("; ".join(breaks))
            a_d, b_d = _marked(ctx, vb, t)
            res = book.decide_live(bar, a_d, b_d, vb.equity_usd)
        else:
            res = book.step(bar)
        orders = orders_for(res, book.symbols, ctx.spot_px[t], ctx.perp_px[t])
        fills = venue.execute(orders)
        if mode == "live":
            # Wat de beurs na de fills houdt, gemeten aan de fills zelf: een gedeeltelijke
            # fill is de nieuwe waarheid (de volgende stap herbalanceert), alles wat daar
            # nog bovenop verandert, is een breuk bij de volgende reconciliatie.
            spot, perp = after_fills(vb.spot, vb.perp_short, fills)
        else:
            spot, perp = _expected(book, res, ctx, t)
            if isinstance(venue, PaperVenue):
                venue.sync(spot, perp, book.equity)
        records.append(_record(ctx, t, mode, res, book, orders, fills))
        state = snapshot(book, candidate=ctx.candidate, config_hash=ctx.config_hash,
                         mode=mode, last_bar=idx[t], expected_spot=spot, expected_perp=perp)
    return records, state


def unwind(state: CarryState, venue: Venue, *, spot_price: dict[str, float],
           perp_price: dict[str, float], reason: str) -> tuple[list[LegFill], CarryState]:
    """De kill switch: het hele boek gehedged plat (eerst de shorts, dan de spot), en de
    staat op stil. Hoeveelheden van de BEURS, niet van het model."""
    vb = venue.book()
    fills = venue.execute(unwind_orders(vb.spot, vb.perp_short, spot_price, perp_price))
    if isinstance(venue, PaperVenue):
        venue.sync({}, {}, vb.equity_usd)
    return fills, state.with_halt(reason)
