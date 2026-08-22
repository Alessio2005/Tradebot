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

Live-validation TODO: Bybit rejects orders whose ``qty`` violates the
per-symbol ``qtyStep`` / ``minOrderQty``.  A precise implementation should
round to the instrument's step from /v5/market/instruments-info; this router
rounds to 8 dp (sufficient for majors, verify per alt before live).
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import time
from decimal import ROUND_DOWN, Decimal
from typing import Dict, Optional, Tuple
from urllib.parse import urlencode

from .order import Fill, Order, OrderStatus
from .paper_oms import PaperOMS

logger = logging.getLogger(__name__)

__all__ = ["OrderRouter"]

_BASE = "https://api.bybit.com"
_CATEGORY = "linear"
_RECV_WINDOW = "5000"


class OrderRouter:
    """Async order router for paper and live Bybit (linear) trading.

    Parameters
    ----------
    paper_oms :
        PaperOMS instance used in paper mode.
    live_mode :
        If True, route orders to Bybit V5 REST.  Requires API keys in env.
    """

    def __init__(
        self,
        paper_oms: PaperOMS,
        live_mode: bool = False,
    ) -> None:
        self._paper = paper_oms
        self._live_mode = live_mode
        self._api_key = os.getenv("BYBIT_API_KEY", "")
        self._api_secret = os.getenv("BYBIT_API_SECRET", "")
        self._session: Optional[object] = None  # aiohttp.ClientSession placeholder

        # Wave 15 P0-5.4 — idempotent order placement cache
        self._order_id_cache: Dict[str, str] = {}

        # Per-symbol lot-size filters (qtyStep, minOrderQty) from
        # /v5/market/instruments-info, fetched lazily and cached.  Bybit rejects
        # orders whose qty violates qtyStep / minOrderQty, so we comply locally.
        self._symbol_filters: Dict[str, Tuple[str, str]] = {}

    # ------------------------------------------------------------------
    # Bybit V5 signing helpers
    # ------------------------------------------------------------------

    def _auth_headers(self, payload: str) -> Dict[str, str]:
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

    async def _get_symbol_filters(self, symbol: str) -> Tuple[str, str]:
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
        """Route an order to paper or live exchange."""
        if not self._live_mode:
            return self._paper.place_order(order)
        return await self._live_place(order)

    async def close_all(self) -> list[Fill]:
        """Flatten all positions."""
        if not self._live_mode:
            return self._paper.close_all()
        return await self._live_close_all()

    async def get_exchange_positions(self) -> Dict[str, float]:
        """Query live exchange for current positions (paper: empty dict)."""
        if not self._live_mode:
            return {}
        return await self._live_get_positions()

    # ------------------------------------------------------------------
    # Live implementation (aiohttp)
    # ------------------------------------------------------------------

    async def _live_place(self, order: Order) -> Fill:
        """Idempotent MARKET order submission to the Bybit V5 REST API.

        Wave 15 P0-5.4: if the order_id is already in the cache (retry scenario),
        return the cached exchange order status instead of placing a duplicate.
        Raises NotImplementedError if aiohttp is not available.
        """
        try:
            import aiohttp
        except ImportError:
            raise NotImplementedError(
                "Live order routing requires aiohttp. "
                "Install with: pip install aiohttp"
            )


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
        body = {
            "category": _CATEGORY,
            "symbol": order.symbol,
            "side": order.side.value.capitalize(),  # BUY→Buy, SELL→Sell
            "orderType": "Market",
            "qty": format(qty_d.normalize(), "f"),
            "orderLinkId": order_id,
        }
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

        # Market orders fill async on Bybit; query realtime status for the
        # average fill price / executed qty.
        return await self._get_order_status(order, exchange_order_id)

    async def _get_order_status(self, order: Order, exchange_order_id: str) -> Fill:
        """Query Bybit for the status of an already-placed order (Wave 15 P0-5.4)."""
        try:
            import aiohttp
        except ImportError:
            raise NotImplementedError("aiohttp required for live order status query.")

        import pandas as pd

        query = urlencode({
            "category": _CATEGORY,
            "symbol": order.symbol,
            "orderId": exchange_order_id,
        })
        headers = self._auth_headers(query)

        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{_BASE}/v5/order/realtime?{query}",
                headers=headers,
            ) as resp:
                data = await resp.json()

        rows = (data.get("result") or {}).get("list") or []
        row = rows[0] if rows else {}
        avg_price = float(row.get("avgPrice", 0.0) or 0.0)
        fill_qty = float(row.get("cumExecQty", order.qty_base) or order.qty_base)
        fill = Fill(
            order_id=order.order_id,
            fill_price=avg_price,
            fill_qty=fill_qty,
            fill_ts=pd.Timestamp.now(tz="UTC"),
            notional_usdt=avg_price * fill_qty,
            exchange_order_id=exchange_order_id,
        )
        order.status = OrderStatus.FILLED
        return fill

    async def _live_close_all(self) -> list[Fill]:
        fills = []
        positions = await self._live_get_positions()
        for symbol, qty in positions.items():
            if abs(qty) < 1e-12:
                continue
            from .order import OrderSide
            from .order import OrderType as OT
            side = OrderSide.SELL if qty > 0 else OrderSide.BUY
            close_order = Order(
                order_id=f"ord_close_{symbol}",
                symbol=symbol,
                side=side,
                order_type=OT.MARKET,
                qty_base=abs(qty),
                signal_prob=0.5,
                kelly_fraction=0.0,
                model_version="close",
                git_sha="",
                feature_hash="",
                portfolio_weight=0.0,
            )
            fills.append(await self._live_place(close_order))
        return fills

    async def _live_get_positions(self) -> Dict[str, float]:
        try:
            import aiohttp
        except ImportError:
            return {}

        query = urlencode({"category": _CATEGORY, "settleCoin": "USDT"})
        headers = self._auth_headers(query)

        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{_BASE}/v5/position/list?{query}",
                headers=headers,
            ) as resp:
                data = await resp.json()

        rows = (data.get("result") or {}).get("list") or []
        out: Dict[str, float] = {}
        for item in rows:
            size = float(item.get("size", 0.0) or 0.0)
            # Bybit reports unsigned size + a side; encode as signed amount.
            signed = size if str(item.get("side", "")).lower() == "buy" else -size
            out[item["symbol"]] = signed
        return out
