# src/tradebot/oms/router.py
"""Bybit order router — async REST interface (V5 API).

In paper mode (default), delegates to PaperOMS.
In live mode, submits orders to the Bybit V5 REST API via aiohttp.

NOTE (venue migration 2026-06-14): migrated Binance Futures → **Bybit V5**.
Auth changes: Bybit signs ``timestamp + api_key + recv_window + payload`` with
HMAC-SHA256 and passes ``X-BAPI-API-KEY / X-BAPI-TIMESTAMP / X-BAPI-RECV-WINDOW
/ X-BAPI-SIGN`` headers (vs Binance's query ``signature`` + ``X-MBX-APIKEY``).
Endpoints: /v5/order/create, /v5/order/realtime, /v5/position/list (all
``category=linear``).

Bybit API keys are loaded from environment variables:
  BYBIT_API_KEY, BYBIT_API_SECRET

The router is intentionally minimal: it handles only MARKET orders for now.
LIMIT order support can be added post v1.0.0.

Lot size: ``qty`` wordt naar beneden afgerond op de ``qtyStep`` van het
instrument (/v5/market/instruments-info) en onder ``minOrderQty`` geweigerd.

DE BOEKHOUDING VOLGT DE BEURS (audit 2026-10-24, E-3/E-5/E-6)
==============================================================
* Een live fill wordt in het interne boek (de tracker van de ``PaperOMS``)
  geboekt, met zijn fee. De engine leest equity en gewichten uit dat boek; tot
  deze fix bleef het in live-modus leeg, dus stuurde elke rebalance het volle
  doel opnieuw en zag de circuit breaker de live-P&L nooit.
* Een fill is pas een fill als de beurs een EINDstatus meldt. De router peilt
  ``/v5/order/realtime`` tot ``Filled`` (of een andere eindstatus) en weigert een
  order zonder uitgevoerde hoeveelheid. Daarvoor las hij de eerste respons, ook
  een lege, als "volledig gevuld tegen prijs 0".
* Een sluitorder krijgt een unieke ``orderLinkId`` en ``reduceOnly``. Met de
  vaste id ``ord_close_{symbol}`` gaf een tweede noodsluiting de status van de
  eerste terug en plaatste niets.
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import math
import os
import time
import uuid
from decimal import ROUND_DOWN, Decimal
from urllib.parse import urlencode

from .order import Fill, Order, OrderSide, OrderStatus, OrderType
from .paper_oms import PaperOMS

logger = logging.getLogger(__name__)

__all__ = ["OrderRouter"]

_BASE = "https://api.bybit.com"
_CATEGORY = "linear"
_RECV_WINDOW = "5000"
#: Bybit V5 ``orderStatus``-waarden waarna een order niet meer verandert.
_TERMINAL = {"Filled": OrderStatus.FILLED, "PartiallyFilledCanceled": OrderStatus.CANCELLED,
             "Cancelled": OrderStatus.CANCELLED, "Rejected": OrderStatus.REJECTED,
             "Deactivated": OrderStatus.CANCELLED}


class OrderRouter:
    """Async order router for paper and live Bybit (linear) trading.

    Parameters
    ----------
    paper_oms :
        PaperOMS instance used in paper mode. In live mode its tracker is the
        internal book that every live fill is booked into.
    live_mode :
        If True, route orders to Bybit V5 REST.  Requires API keys in env.
    status_polls, status_poll_interval_s :
        How often, and how far apart, the router asks Bybit for the terminal
        status of a placed order before it gives up with an error.
    """

    def __init__(
        self,
        paper_oms: PaperOMS,
        live_mode: bool = False,
        status_polls: int = 20,
        status_poll_interval_s: float = 0.25,
    ) -> None:
        self._paper = paper_oms
        self._live_mode = live_mode
        # Hoe lang de router op een eindstatus wacht: 20 x 0,25 s. Een marketorder is
        # op Bybit in milliseconden klaar; vijf seconden zonder eindstatus is een storing.
        self._status_polls = int(status_polls)
        self._status_poll_interval_s = float(status_poll_interval_s)
        # Orders waarvan de fill al in het interne boek staat: een idempotente retry
        # mag hem niet nog eens boeken.
        self._booked: set[str] = set()
        self._api_key = os.getenv("BYBIT_API_KEY", "")
        self._api_secret = os.getenv("BYBIT_API_SECRET", "")
        self._session: object | None = None  # aiohttp.ClientSession placeholder

        # Wave 15 P0-5.4 — idempotent order placement cache
        self._order_id_cache: dict[str, str] = {}

        # Per-symbol lot-size filters (qtyStep, minOrderQty) from
        # /v5/market/instruments-info, fetched lazily and cached.  Bybit rejects
        # orders whose qty violates qtyStep / minOrderQty, so we comply locally.
        self._symbol_filters: dict[str, tuple[str, str]] = {}

    # ------------------------------------------------------------------
    # Bybit V5 signing helpers
    # ------------------------------------------------------------------

    def _auth_headers(self, payload: str) -> dict[str, str]:
        """Build signed Bybit V5 auth headers for a given payload string.

        ``payload`` is the raw JSON body (POST) or the query string (GET).
        Signature = HMAC_SHA256(secret, timestamp + api_key + recv_window + payload).
        """
        ts = str(int(time.time() * 1000))
        to_sign = ts + self._api_key + _RECV_WINDOW + payload
        sign = hmac.new(
            self._api_secret.encode(), to_sign.encode(), hashlib.sha256
        ).hexdigest()
        return {
            "X-BAPI-API-KEY": self._api_key,
            "X-BAPI-TIMESTAMP": ts,
            "X-BAPI-RECV-WINDOW": _RECV_WINDOW,
            "X-BAPI-SIGN": sign,
            "Content-Type": "application/json",
        }

    async def _get_symbol_filters(self, symbol: str) -> tuple[str, str]:
        """Return (qtyStep, minOrderQty) for ``symbol`` (cached).

        Falls back to ("0.00000001", "0") if the public instruments-info call
        fails, so order submission still proceeds (Bybit then validates).
        """
        if symbol in self._symbol_filters:
            return self._symbol_filters[symbol]
        qty_step, min_qty = "0.00000001", "0"
        import aiohttp

        query = urlencode({"category": _CATEGORY, "symbol": symbol})
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{_BASE}/v5/market/instruments-info?{query}", timeout=5
            ) as resp:
                data = await resp.json()
        rows = (data.get("result") or {}).get("list") or []
        if rows:
            lot = rows[0].get("lotSizeFilter", {}) or {}
            qty_step = str(lot.get("qtyStep", qty_step))
            min_qty = str(lot.get("minOrderQty", min_qty))
        self._symbol_filters[symbol] = (qty_step, min_qty)
        return qty_step, min_qty

    @staticmethod
    def _round_to_step(qty: float, step: str) -> Decimal:
        """Floor ``qty`` to the instrument's ``qtyStep`` (Decimal-exact)."""
        step_d = Decimal(step)
        if step_d <= 0:
            return Decimal(str(qty))
        return (Decimal(str(qty)) / step_d).to_integral_value(
            rounding=ROUND_DOWN
        ) * step_d

    # ------------------------------------------------------------------
    # Public async interface
    # ------------------------------------------------------------------

    async def place_order(self, order: Order) -> Fill:
        """Route an order to paper or live exchange.

        Live: the confirmed fill is booked into the internal book exactly once."""
        if not self._live_mode:
            return self._paper.place_order(order)
        fill, fee = await self._live_place(order)
        self._book(order, fill, fee)
        return fill

    def _book(self, order: Order, fill: Fill, fee: float) -> None:
        """Boek een live fill (en zijn fee) in het interne boek, één keer per order."""
        if order.order_id in self._booked:
            return
        self._booked.add(order.order_id)
        tracker = self._paper.tracker
        tracker.apply_fill(order.symbol, fill, order.side)
        tracker.apply_cash(-fee)
        tracker.mark(order.symbol, fill.fill_price)

    async def close_all(self) -> list[Fill]:
        """Flatten all positions."""
        if not self._live_mode:
            return self._paper.close_all()
        return await self._live_close_all()

    async def get_exchange_positions(self) -> dict[str, float]:
        """Query live exchange for current positions (paper: empty dict)."""
        if not self._live_mode:
            return {}
        return await self._live_get_positions()

    # ------------------------------------------------------------------
    # Live implementation (aiohttp)
    # ------------------------------------------------------------------

    async def _live_place(self, order: Order) -> tuple[Fill, float]:
        """Idempotent MARKET order submission to the Bybit V5 REST API.

        Wave 15 P0-5.4: if the order_id is already in the cache (retry scenario),
        return the cached exchange order status instead of placing a duplicate.
        Returns the fill and its fee (USDT).
        """
        # Phase 0: aiohttp is een harde dependency (pyproject.toml); de
        # try/except ImportError-guard hier is daarmee dode code geworden.
        import aiohttp

        order_id = str(order.order_id)

        # Wave 15 P0-5.4 — idempotent retry: if already placed, query status
        if order_id in self._order_id_cache:
            exchange_id = self._order_id_cache[order_id]
            logger.info(
                "Idempotent retry: order %s already placed as exchange_id=%s",
                order_id, exchange_id,
            )
            return await self._get_order_status(order, exchange_id)

        # Comply with the per-symbol lot-size filter: floor qty to qtyStep and
        # require >= minOrderQty (Bybit rejects violations).
        qty_step, min_qty = await self._get_symbol_filters(order.symbol)
        qty_d = self._round_to_step(order.qty_base, qty_step)
        if qty_d < Decimal(min_qty):
            raise RuntimeError(
                f"Bybit order for {order.symbol} below minOrderQty: "
                f"qty={qty_d} < min={min_qty} (qtyStep={qty_step})."
            )

        # New order — submit to exchange.  Bybit wants Buy/Sell (title-case)
        # and ``orderLinkId`` gives us exchange-side idempotency too.
        body: dict[str, object] = {
            "category": _CATEGORY,
            "symbol": order.symbol,
            "side": order.side.value.capitalize(),  # BUY→Buy, SELL→Sell
            "orderType": "Market",
            "qty": format(qty_d.normalize(), "f"),
            "orderLinkId": order_id,
        }
        if order.reduce_only:
            body["reduceOnly"] = True
        body_str = json.dumps(body, separators=(",", ":"))
        headers = self._auth_headers(body_str)

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{_BASE}/v5/order/create",
                data=body_str,
                headers=headers,
            ) as resp:
                data = await resp.json()

        if data.get("retCode", -1) != 0:
            raise RuntimeError(f"Bybit order failed: {data}")

        result = data.get("result") or {}
        exchange_order_id = str(result.get("orderId", ""))
        # Cache for idempotent retries (Wave 15 P0-5.4)
        self._order_id_cache[order_id] = exchange_order_id
        logger.info(
            "Order placed: internal=%s exchange=%s", order_id, exchange_order_id
        )

        # Market orders fill async on Bybit; poll until a terminal status.
        return await self._get_order_status(order, exchange_order_id)

    async def _get_order_status(self, order: Order,
                                exchange_order_id: str) -> tuple[Fill, float]:
        """Peil Bybit tot de order een eindstatus heeft; geef de werkelijke fill terug.

        Geen eindstatus binnen `status_polls` pogingen, of een eindstatus zonder
        uitgevoerde hoeveelheid, is een fout: de positie is dan onbekend of nul, en
        geen van beide mag als fill worden geboekt."""
        # Phase 0: aiohttp is een harde dependency (pyproject.toml); de
        # try/except ImportError-guard hier is daarmee dode code geworden.
        import aiohttp
        import pandas as pd

        query = urlencode({
            "category": _CATEGORY,
            "symbol": order.symbol,
            "orderId": exchange_order_id,
        })
        row: dict[str, object] = {}
        for attempt in range(self._status_polls):
            if attempt:
                await asyncio.sleep(self._status_poll_interval_s)
            headers = self._auth_headers(query)
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{_BASE}/v5/order/realtime?{query}",
                    headers=headers,
                ) as resp:
                    data = await resp.json()
            rows = (data.get("result") or {}).get("list") or []
            row = rows[0] if rows else {}
            if row.get("orderStatus") in _TERMINAL:
                break
        else:
            raise RuntimeError(
                f"Bybit order {exchange_order_id} ({order.order_id}) has no terminal "
                f"status after {self._status_polls} polls (last: "
                f"{row.get('orderStatus') if row else 'order not found'!r}); the position "
                "is unknown — reconcile before trading on."
            )

        status = _TERMINAL[str(row["orderStatus"])]
        fill_qty = float(str(row.get("cumExecQty") or 0.0))
        avg_price = float(str(row.get("avgPrice") or 0.0))
        fee = float(str(row.get("cumExecFee") or 0.0))
        order.status = status
        if fill_qty <= 0.0:
            raise RuntimeError(
                f"Bybit order {exchange_order_id} ({order.order_id}) ended "
                f"{row['orderStatus']!r} without an executed quantity."
            )
        if not (math.isfinite(avg_price) and avg_price > 0.0):
            raise RuntimeError(
                f"Bybit order {exchange_order_id} reports {fill_qty} executed "
                f"without a valid average price ({row.get('avgPrice')!r})."
            )
        if status is not OrderStatus.FILLED:
            logger.warning("Order %s partially filled: %s of %s (%s).", order.order_id,
                           fill_qty, order.qty_base, row["orderStatus"])
        notional = avg_price * fill_qty
        fill = Fill(
            order_id=order.order_id,
            fill_price=avg_price,
            fill_qty=fill_qty,
            fill_ts=pd.Timestamp.now(tz="UTC"),
            notional_usdt=notional,
            post_trade_cost_bps=fee / notional * 1e4,
            exchange_order_id=exchange_order_id,
        )
        return fill, fee

    async def _live_close_all(self) -> list[Fill]:
        """Sluit elke beurspositie met een reduce-only marketorder.

        Elke aanroep krijgt nieuwe order-id's: een tweede noodsluiting moet ook
        echt een order plaatsen."""
        fills = []
        positions = await self._live_get_positions()
        for symbol, qty in positions.items():
            if abs(qty) < 1e-12:
                continue
            side = OrderSide.SELL if qty > 0 else OrderSide.BUY
            close_order = Order(
                order_id=f"ord_close_{symbol}_{uuid.uuid4().hex[:8]}",
                symbol=symbol,
                side=side,
                order_type=OrderType.MARKET,
                qty_base=abs(qty),
                signal_prob=0.5,
                kelly_fraction=0.0,
                model_version="close",
                git_sha="",
                feature_hash="",
                portfolio_weight=0.0,
                reduce_only=True,
            )
            fills.append(await self.place_order(close_order))
        return fills

    async def _live_get_positions(self) -> dict[str, float]:
        # Phase 0: `except ImportError: return {}` meldde "geen posities" terwijl
        # de werkelijke positie onbekend was - de gevaarlijkste vorm van stille
        # degradatie in een OMS. aiohttp is een harde dependency.
        import aiohttp

        query = urlencode({"category": _CATEGORY, "settleCoin": "USDT"})
        headers = self._auth_headers(query)

        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{_BASE}/v5/position/list?{query}",
                headers=headers,
            ) as resp:
                data = await resp.json()

        rows = (data.get("result") or {}).get("list") or []
        out: dict[str, float] = {}
        for item in rows:
            size = float(item.get("size", 0.0) or 0.0)
            # Bybit reports unsigned size + a side; encode as signed amount.
            signed = size if str(item.get("side", "")).lower() == "buy" else -size
            out[item["symbol"]] = signed
        return out
