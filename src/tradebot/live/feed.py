# src/tradebot/live/feed.py
"""Bybit V5 WebSocket feed — 5-second OHLCV bars via publicTrade aggregation.

NOTE (venue migration 2026-06-14): migrated Binance → **Bybit linear (USDT)**.
Stream mapping:
  Binance {sym}@aggTrade   → Bybit publicTrade.{SYM}   (5s bar aggregation)
  Binance {sym}@bookTicker → Bybit orderbook.1.{SYM}   (best bid/ask, F1)
  Binance REST premiumIndex→ Bybit REST /v5/market/tickers (funding, F2)
Bybit public streams require a JSON subscribe handshake after connect
(``{"op":"subscribe","args":[...]}``) and re-subscription on reconnect.

Live mode: subscribes to ``publicTrade.{SYM}`` and aggregates individual trades
into ``bar_seconds``-second tumbling bars (default 5s).  This gives EXACT parity
with the training data (5s raw bars from the market_data_parquet files): same
bar resolution, same taker_buy_volume for CVD features (Bybit ``S`` field =
aggressor side), same CUSUM step cadence inside ModelSignal.

Paper mode: replays historical bar DataFrames provided by the caller.

Why publicTrade and not kline?
  Bybit kline WS minimum interval is 1m and carries no taker-side breakdown.
  publicTrade delivers every trade individually (~1ms latency); we aggregate
  client-side into 5s tumbling windows — IDENTICAL to how the raw 5s parquets
  in market_data_parquet/{SYM}/*.parquet are produced from the trade tape.

Each completed bar is pushed as a ``BarEvent`` to the engine's async queue.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["BarEvent", "FeedConfig", "Feed"]


@dataclass(frozen=True)
class BarEvent:
    """A single completed OHLCV bar from the feed.

    Attributes
    ----------
    symbol :
        Trading pair.
    ts :
        Bar close timestamp (UTC-aware).
    open, high, low, close, volume :
        OHLCV values.
    taker_buy_volume :
        Taker-buy base-asset volume for the bar.  Non-zero only from
        aggTrade-based bars; 0.0 from kline-based paper-replay bars that
        don't carry this field.  Used by FeaturePipeline for CVD features
        (feat_cvd_flow_ffd, feat_cvd_session_z, feat_rvd_z).
    funding_rate :
        Latest 8h funding rate at bar close (0.0 if unavailable).
    seq :
        Monotonically increasing sequence number per symbol (Wave 15 P0-5.6).
        0 means sequence tracking disabled (e.g. paper-replay without seq data).
    """

    symbol: str
    ts: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float
    taker_buy_volume: float = 0.0
    funding_rate: float = 0.0
    seq: int = 0


@dataclass
class FeedConfig:
    """Feed configuration.

    Attributes
    ----------
    symbols :
        List of trading pairs to subscribe to.
    bar_seconds :
        Target bar duration in seconds.  Default 5 — matches the training
        data resolution exactly (5s raw bars → FeaturePipeline).
        In live mode this controls the aggTrade aggregation window.
    paper_mode :
        If True, replay historical bars from ``historical_bars``.
    replay_delay_s :
        Simulated delay between bars in paper-replay mode (0 = instant).
    interval :
        Kept for backward compatibility / audit logging only.  In live mode
        the actual bar resolution is ``bar_seconds``, not this string.
    """

    symbols: List[str]
    bar_seconds: int = 5
    paper_mode: bool = True
    replay_delay_s: float = 0.0
    interval: str = "5s"    # informational only; live feed uses publicTrade
    funding_interval_min: int = 480  # Bybit funding cycle (min). 480=8h (majors);
                                     # set 240/60 for 4h/1h alts (instruments-info).


class _PartialBar:
    """In-progress aggregation of aggTrade events into one N-second bar.

    Bars are aligned to Unix-epoch boundaries:
      bar_id  = trade_ms // bar_ms
      bar ends at: (bar_id + 1) * bar_ms  (exclusive)
    """

    __slots__ = (
        "bar_id", "open", "high", "low", "close",
        "volume", "taker_buy_vol",
    )

    def __init__(
        self,
        trade_ms: int,
        price: float,
        qty: float,
        is_taker_buy: bool,
        bar_ms: int,
    ) -> None:
        self.bar_id       = trade_ms // bar_ms
        self.open         = price
        self.high         = price
        self.low          = price
        self.close        = price
        self.volume       = qty
        self.taker_buy_vol = qty if is_taker_buy else 0.0

    def update(
        self,
        price: float,
        qty: float,
        is_taker_buy: bool,
    ) -> None:
        if price > self.high:
            self.high = price
        if price < self.low:
            self.low = price
        self.close   = price
        self.volume += qty
        if is_taker_buy:
            self.taker_buy_vol += qty

    def to_bar_event(
        self,
        symbol: str,
        bar_ms: int,
        seq: int,
    ) -> BarEvent:
        close_ts_ms = (self.bar_id + 1) * bar_ms - 1
        return BarEvent(
            symbol=symbol,
            ts=pd.Timestamp(close_ts_ms, unit="ms", tz="UTC"),
            open=self.open,
            high=self.high,
            low=self.low,
            close=self.close,
            volume=self.volume,
            taker_buy_volume=self.taker_buy_vol,
            seq=seq,
        )


class Feed:
    """Market data feed that emits BarEvents.

    Parameters
    ----------
    config :
        Feed configuration.
    queue :
        asyncio.Queue where BarEvents are pushed.
    historical_bars :
        symbol → sorted DataFrame (for paper-replay mode).
        Index must be a UTC-aware DatetimeIndex.
    """

    def __init__(
        self,
        config: FeedConfig,
        queue: asyncio.Queue,
        historical_bars: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> None:
        self._cfg = config
        self._queue = queue
        self._bars = historical_bars or {}
        self._running = False
        # Wave 15 P0-5.6 — per-symbol sequence counters
        self._seq_counters: Dict[str, int] = {}
        # F1 — live L1 best bid/ask per symbol (from orderbook.1 stream).
        # Tuple: (best_bid, best_ask, recv_ms).  Used for realistic spread
        # injection into PaperOMS (replaces synthetic (H-L)/8 estimate).
        self._best_quote: Dict[str, tuple[float, float, int]] = {}
        # F2 — last observed funding rate per symbol (from REST tickers).
        # Updated every poll_interval_s seconds; used to stamp BarEvents that
        # cross an 8h funding boundary (UTC 00:00, 08:00, 16:00).
        self._funding_rate: Dict[str, float] = {}
        # Track last 8h-boundary already paid out per symbol so we never
        # double-bill within one funding interval.
        self._last_funding_bar_id: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Public control
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the feed.  Returns immediately; feed runs as background tasks.

        In live mode this spawns THREE coroutines:
          - _live_ws_aggtrade : publicTrade → 5s OHLCV bars (primary path)
          - _live_ws_bookticker : orderbook.1 best_bid/ask (F1, replaces synthetic spread)
          - _live_funding_poller : REST /v5/market/tickers every 30s (F2, linear only)
        """
        self._running = True
        if self._cfg.paper_mode:
            asyncio.create_task(self._replay())
        else:
            asyncio.create_task(self._live_ws_aggtrade())
            # F1 — book-ticker stream is independent; it just updates an
            # in-memory map.  Engine reads from get_best_quote() per bar.
            asyncio.create_task(self._live_ws_bookticker())
            # F2 — funding poller (Bybit linear tickers REST). Skip when
            # TRADEBOT_FEED_SPOT_FALLBACK=1 (spot has no funding rate).
            import os as _os
            if _os.environ.get("TRADEBOT_FEED_SPOT_FALLBACK", "0") != "1":
                asyncio.create_task(self._live_funding_poller())
            else:
                logger.warning(
                    "Feed: spot fallback active — funding poller disabled "
                    "(spot has no funding rate)."
                )

    async def stop(self) -> None:
        self._running = False

    # ------------------------------------------------------------------
    # F1 — Best-quote accessor (consumed by engine for PaperOMS spread)
    # ------------------------------------------------------------------

    def get_best_quote(self, symbol: str) -> Optional[tuple[float, float]]:
        """Return (best_bid, best_ask) for ``symbol`` if a fresh quote exists.

        Returns None if no quote has arrived yet or the last quote is older
        than 5 seconds (stale).  Engine falls back to synthetic spread when
        this returns None — but the floor of 5 bps is removed to avoid
        double-counting cost (the YAML cost_bps_per_asset is the backtest
        cost; in live the L1 half-spread is the cost).
        """
        q = self._best_quote.get(symbol)
        if q is None:
            return None
        bid, ask, ts_ms = q
        now_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
        if now_ms - ts_ms > 5_000:
            return None  # stale
        return (bid, ask)

    def get_funding_rate(self, symbol: str) -> float:
        """Return the latest cached 8h funding rate (decimal, e.g. 0.0001).

        Returns 0.0 if no value has been polled yet.
        """
        return self._funding_rate.get(symbol, 0.0)

    # ------------------------------------------------------------------
    # Paper-replay
    # ------------------------------------------------------------------

    async def _replay(self) -> None:
        """Interleave bars from all symbols in chronological order."""
        rows: list[tuple[pd.Timestamp, str, pd.Series]] = []
        for symbol, df in self._bars.items():
            for ts, row in df.iterrows():
                rows.append((ts, symbol, row))

        rows.sort(key=lambda x: x[0])
        logger.info("Feed: paper-replay starting, %d bars total.", len(rows))

        for ts, symbol, row in rows:
            if not self._running:
                break
            event = BarEvent(
                symbol=symbol,
                ts=ts if hasattr(ts, "tz") and ts.tz is not None
                    else pd.Timestamp(ts, tz="UTC"),
                open=float(row.get("open", row.get("o", 0.0))),
                high=float(row.get("high", row.get("h", 0.0))),
                low=float(row.get("low", row.get("l", 0.0))),
                close=float(row.get("close", row.get("c", 0.0))),
                volume=float(row.get("volume", row.get("v", 0.0))),
                taker_buy_volume=float(row.get("taker_buy_volume", 0.0)),
                funding_rate=float(row.get("funding_rate", 0.0)),
            )
            await self._queue.put(event)
            if self._cfg.replay_delay_s > 0:
                await asyncio.sleep(self._cfg.replay_delay_s)

        # Signal end of replay
        await self._queue.put(None)
        logger.info("Feed: paper-replay finished.")

    # ------------------------------------------------------------------
    # Live WebSocket — publicTrade → N-second bars (primary live path)
    # ------------------------------------------------------------------

    async def _live_ws_aggtrade(self) -> None:
        """Aggregate Bybit publicTrade events into bar_seconds-second OHLCV bars.

        Bybit kline WS minimum interval is 1m and carries no taker-side split.
        We subscribe to ``publicTrade.{SYM}`` for each symbol and aggregate
        client-side into ``bar_seconds``-second tumbling windows.  This matches
        training exactly: the market_data_parquet 5s bars are produced the same
        way from the trade tape.

        Bar alignment: epoch-aligned boundaries (bar_id = trade_ms // bar_ms).
        A bar is emitted when the FIRST trade of the next window arrives,
        so bars are emitted with ~0ms latency after window close.

        Taker-buy volume is accumulated per bar so that FeaturePipeline
        can compute CVD features (feat_cvd_flow_ffd, feat_cvd_session_z,
        feat_rvd_z) identically to training.  Bybit ``S`` is the aggressor side
        (Buy → taker buy).

        Reconnect policy (P2-3): exponential backoff, cap 60s. Re-subscribes on
        every (re)connect.
        """
        try:
            import websockets
        except ImportError:
            raise RuntimeError(
                "Live feed requires websockets. Install with: pip install websockets"
            )

        import json

        bar_ms = self._cfg.bar_seconds * 1000
        sym_set = {s.upper() for s in self._cfg.symbols}

        # Bybit V5 public stream endpoint.  linear = USDT perps; spot fallback
        # uses the spot stream (no funding) when geo-blocked.
        import os as _os
        spot_fallback = _os.environ.get("TRADEBOT_FEED_SPOT_FALLBACK", "0") == "1"
        if spot_fallback:
            url = "wss://stream.bybit.com/v5/public/spot"
            logger.critical(
                "Feed: TRADEBOT_FEED_SPOT_FALLBACK=1 — connecting to Bybit SPOT "
                "%s. Funding-rate parity LOST. Backtest comparison invalid.", url,
            )
        else:
            url = "wss://stream.bybit.com/v5/public/linear"
            logger.info(
                "Feed: connecting to Bybit LINEAR publicTrade WS %s (bar_seconds=%d) "
                "— parity with training data (USDT perp).",
                url, self._cfg.bar_seconds,
            )

        sub_args = [f"publicTrade.{sym.upper()}" for sym in self._cfg.symbols]
        sub_msg = json.dumps({"op": "subscribe", "args": sub_args})

        _BASE_BACKOFF_S: float = 1.0
        _MAX_BACKOFF_S: float  = 60.0
        attempt: int = 0

        # In-progress bars per symbol.  Initialised on first trade.
        partial: Dict[str, _PartialBar] = {}

        while self._running:
            async with websockets.connect(url, ping_interval=20) as ws:
                await ws.send(sub_msg)  # Bybit subscribe handshake
                if attempt > 0:
                    logger.info(
                        "Feed: WS reconnected after %d attempt(s).", attempt
                    )
                attempt = 0

                async for message in ws:
                    if not self._running:
                        return

                    data = json.loads(message)
                    # Bybit control frames (sub ack / pong) carry no "topic".
                    topic = data.get("topic", "")
                    if not topic.startswith("publicTrade."):
                        continue

                    # data["data"] is a LIST of trades in this push.
                    for ev in data.get("data", []):
                        sym = ev.get("s")
                        if sym not in sym_set:
                            continue

                        price        = float(ev["p"])
                        qty          = float(ev["v"])
                        trade_ms     = int(ev["T"])
                        is_taker_buy = str(ev.get("S", "")).lower() == "buy"
                        bar_id       = trade_ms // bar_ms

                        if sym not in partial:
                            # First trade for this symbol
                            partial[sym] = _PartialBar(
                                trade_ms, price, qty, is_taker_buy, bar_ms
                            )
                            continue
                        if bar_id > partial[sym].bar_id:
                            # Window closed → emit completed bar, start new one
                            self._seq_counters[sym] = (
                                self._seq_counters.get(sym, 0) + 1
                            )
                            bar_event = partial[sym].to_bar_event(
                                sym, bar_ms, self._seq_counters[sym]
                            )
                            # F2 — attach funding rate if this bar crossed an
                            # 8h funding boundary (00:00, 08:00, 16:00 UTC).
                            # We tag the FIRST bar at or after the boundary;
                            # _last_funding_bar_id prevents double-billing if
                            # multiple bars fall within the same 8h window
                            # (only the first carries the non-zero rate, which
                            # PaperOMS.accrue_funding consumes).
                            bar_event = self._maybe_attach_funding(bar_event, bar_ms)
                            await self._queue.put(bar_event)
                            logger.debug(
                                "Feed: emitted 5s bar %s ts=%s o=%.4f c=%.4f "
                                "v=%.2f tbv=%.2f",
                                sym, bar_event.ts, bar_event.open,
                                bar_event.close, bar_event.volume,
                                bar_event.taker_buy_volume,
                            )
                            # Begin the new bar with the triggering trade
                            partial[sym] = _PartialBar(
                                trade_ms, price, qty, is_taker_buy, bar_ms
                            )
                        else:
                            # Same window → accumulate
                            partial[sym].update(price, qty, is_taker_buy)

    # ------------------------------------------------------------------
    # F2 — Funding boundary detection (8h: 00:00, 08:00, 16:00 UTC)
    # ------------------------------------------------------------------

    _FUNDING_BOUNDARY_MS: int = 8 * 3600 * 1000  # default 8h in ms (majors)

    def _maybe_attach_funding(self, bar: "BarEvent", bar_ms: int) -> "BarEvent":
        """Return ``bar`` possibly augmented with funding_rate.

        A non-zero funding_rate is attached on EXACTLY the first bar of each
        Bybit funding cycle (``FeedConfig.funding_interval_min``, default 8h)
        so that PaperOMS.accrue_funding() runs once per cycle.  Cycles are
        tracked per-symbol via ``_last_funding_bar_id`` to prevent double
        accrual within one window.
        """
        sym = bar.symbol
        # Funding bar id = (bar close ms) // (cycle ms).  All bars within the
        # same window share the same id; only the FIRST bar where the id
        # advances gets stamped.
        boundary_ms = int(getattr(self._cfg, "funding_interval_min", 480)) * 60_000
        if boundary_ms <= 0:
            boundary_ms = self._FUNDING_BOUNDARY_MS
        bar_close_ms = int(bar.ts.timestamp() * 1000)
        funding_window_id = bar_close_ms // boundary_ms
        last = self._last_funding_bar_id.get(sym, -1)
        if funding_window_id == last:
            return bar  # already paid this window
        # Stamp this bar with the cached funding rate (0.0 if poller hasn't
        # produced one yet → no accrual, log silently).
        rate = self._funding_rate.get(sym, 0.0)
        self._last_funding_bar_id[sym] = funding_window_id
        if rate == 0.0:
            return bar
        # BarEvent is frozen — recreate with funding_rate set.
        from dataclasses import replace as _replace
        logger.info(
            "Feed [%s]: tagging funding bar window=%d rate=%.6f ts=%s",
            sym, funding_window_id, rate, bar.ts,
        )
        return _replace(bar, funding_rate=rate)

    # ------------------------------------------------------------------
    # F1 — Live book-ticker stream (best_bid / best_ask per symbol)
    # ------------------------------------------------------------------

    async def _live_ws_bookticker(self) -> None:
        """Subscribe to Bybit orderbook.1 streams; update best-bid/ask in memory.

        Bybit ``orderbook.1.{SYM}`` emits a snapshot then deltas whenever the
        top-of-book changes (multiple times per second for liquid perps).  We
        do NOT push bar events from this — it is purely an out-of-band state
        update consumed via :py:meth:`get_best_quote`.  A delta may omit the
        bid OR ask side if only one changed, so we keep the last good level.

        Cost-double-count safeguard (per user note 2026-05-25):
          The L1 half-spread injected via this stream REPLACES the synthetic
          (high-low)/8 estimate in engine.py.  It is NOT added on top of
          ``cost_bps_per_asset`` from prod.yaml — that YAML value is consumed
          ONLY in backtest_portfolio.py:708 (line confirmed via grep), never
          in PaperOMS.  So the live cost stack remains: half-spread + taker
          fee, exactly mirroring the backtest spread_arr × 0.5 + fee model.
        """
        # Phase 0: `except ImportError: return` liet de bookticker-feed
        # STIL afsluiten, waarna het live systeem zonder L1-quotes doordraaide.
        # websockets is een harde dependency.
        import json
        import os as _os

        import websockets

        sym_set = {s.upper() for s in self._cfg.symbols}
        spot_fallback = _os.environ.get("TRADEBOT_FEED_SPOT_FALLBACK", "0") == "1"
        url = (
            "wss://stream.bybit.com/v5/public/spot"
            if spot_fallback
            else "wss://stream.bybit.com/v5/public/linear"
        )
        sub_args = [f"orderbook.1.{sym.upper()}" for sym in self._cfg.symbols]
        sub_msg = json.dumps({"op": "subscribe", "args": sub_args})
        logger.info("Feed: connecting to Bybit orderbook.1 WS %s", url)

        _BASE_BACKOFF_S: float = 1.0
        _MAX_BACKOFF_S: float  = 60.0
        attempt: int = 0

        while self._running:
            async with websockets.connect(url, ping_interval=20) as ws:
                await ws.send(sub_msg)
                attempt = 0
                async for message in ws:
                    if not self._running:
                        return
                    data = json.loads(message)
                    topic = data.get("topic", "")
                    if not topic.startswith("orderbook.1."):
                        continue
                    ob = data.get("data", {})
                    sym = ob.get("s")
                    if sym is None or sym not in sym_set:
                        continue
                    # b/a are [[price, size], ...] sorted best-first; either
                    # may be empty on a one-sided delta.
                    bids = ob.get("b") or []
                    asks = ob.get("a") or []
                    prev = self._best_quote.get(sym)
                    bid = float(bids[0][0]) if bids else (prev[0] if prev else None)
                    ask = float(asks[0][0]) if asks else (prev[1] if prev else None)
                    if bid is None or ask is None:
                        continue
                    ts_ms = int(data.get("ts", 0) or 0)
                    if ts_ms == 0:
                        ts_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
                    self._best_quote[sym] = (bid, ask, ts_ms)

    # ------------------------------------------------------------------
    # F2 — Funding rate poller (REST /v5/market/tickers, category=linear)
    # ------------------------------------------------------------------

    _BYBIT_TICKERS_URL = "https://api.bybit.com/v5/market/tickers"

    @staticmethod
    def _parse_bybit_funding(payload: dict) -> float:
        """Extract ``fundingRate`` from a Bybit V5 tickers payload (0.0 if absent)."""
        if payload.get("retCode", -1) != 0:
            return 0.0
        rows = (payload.get("result") or {}).get("list") or []
        if not rows:
            return 0.0
        return float(rows[0].get("fundingRate", 0.0) or 0.0)

    async def _live_funding_poller(self, poll_interval_s: float = 30.0) -> None:
        """Poll Bybit linear tickers every ``poll_interval_s`` seconds.

        Stores latest ``fundingRate`` per symbol in ``_funding_rate``.
        The publicTrade loop consumes this value when emitting a bar that
        crosses an 8h boundary (see _maybe_attach_funding).

        Rationale: the WebSocket tickers stream pushes delta updates frequently
        and is heavyweight.  We only need the funding rate at the 8h boundary —
        polling at 30s cadence catches all updates without flooding the
        connection.
        """
        # Phase 0: hier stond een 40-regelige synchrone urllib-fallback voor het
        # geval aiohttp ontbrak, met daaromheen een `except Exception` die
        # "funding poller cannot start" logde en gewoon returnde. Netto: de
        # live-loop draaide door met funding_rate op de laatst bekende waarde
        # (of nul), terwijl de backtest wel funding aanrekende - een stille
        # sim-to-live divergentie. aiohttp is een harde dependency.
        import aiohttp

        # F2 FIX 2026-05-26 (Windows / Python 3.13 / aiohttp 3.10):
        #   aiohttp's default TCPConnector tries aiodns, which requires a
        #   SelectorEventLoop on Windows.  Python 3.13 defaults to
        #   ProactorEventLoop → RuntimeError on session construction.
        #   Force the threaded resolver (DNS via the standard thread pool)
        #   to keep parity with Linux behaviour without flipping the global
        #   event-loop policy.
        from aiohttp.resolver import ThreadedResolver
        connector = aiohttp.TCPConnector(resolver=ThreadedResolver())

        # Phase 0: de omhullende `except Exception` logde "Funding-rate parity
        # LOST for this run" en ging door. Verlies van funding-pariteit is geen
        # log-regel maar een contractschending: de live-P&L wijkt dan
        # structureel af van de backtest.
        async with aiohttp.ClientSession(connector=connector) as session:
            while self._running:
                for sym in self._cfg.symbols:
                    url = (
                        f"{self._BYBIT_TICKERS_URL}"
                        f"?category=linear&symbol={sym}"
                    )
                    async with session.get(url, timeout=5) as resp:
                        payload = await resp.json()
                    rate = self._parse_bybit_funding(payload)
                    self._funding_rate[sym] = rate
                    logger.debug(
                        "Feed [%s]: funding poll fundingRate=%.6f",
                        sym, rate,
                    )
                await asyncio.sleep(poll_interval_s)

    # ------------------------------------------------------------------
    # Legacy kline WebSocket (kept for reference / fallback testing only)
    # ------------------------------------------------------------------

    async def _live_ws_kline(self) -> None:
        """Connect to Bybit kline WebSocket streams (NOT used in production).

        This path uses ``kline.{interval}.{SYM}`` topics (minimum interval = 1m).
        Only use this for debugging; production always uses _live_ws_aggtrade()
        which provides 5s resolution matching the training data.
        """
        try:
            import websockets
        except ImportError:
            raise RuntimeError(
                "Live feed requires websockets. Install with: pip install websockets"
            )

        import json

        # Bybit kline interval is a bare number of minutes ("1","3","5",...);
        # map the informational "5s"/"1m" style string to the closest minute.
        iv = str(self._cfg.interval).lower().rstrip("m") or "1"
        if iv.endswith("s"):
            iv = "1"
        url = "wss://stream.bybit.com/v5/public/linear"
        sub_args = [f"kline.{iv}.{sym.upper()}" for sym in self._cfg.symbols]
        sub_msg = json.dumps({"op": "subscribe", "args": sub_args})
        logger.warning(
            "Feed: using LEGACY Bybit kline stream %s — this breaks 5s feature parity.",
            sub_args,
        )

        _BASE_BACKOFF_S: float = 1.0
        _MAX_BACKOFF_S: float  = 60.0
        attempt: int = 0

        while self._running:
            async with websockets.connect(url, ping_interval=20) as ws:
                await ws.send(sub_msg)
                if attempt > 0:
                    logger.info(
                        "Feed: WS reconnected after %d attempt(s).", attempt
                    )
                attempt = 0

                async for message in ws:
                    if not self._running:
                        return
                    data = json.loads(message)
                    topic = data.get("topic", "")
                    if not topic.startswith("kline."):
                        continue
                    symbol = topic.split(".")[-1]
                    for k in data.get("data", []):
                        if not k.get("confirm"):  # confirm=True → bar closed
                            continue
                        self._seq_counters[symbol] = (
                            self._seq_counters.get(symbol, 0) + 1
                        )
                        event = BarEvent(
                            symbol=symbol,
                            ts=pd.Timestamp(int(k["end"]), unit="ms", tz="UTC"),
                            open=float(k["open"]),
                            high=float(k["high"]),
                            low=float(k["low"]),
                            close=float(k["close"]),
                            volume=float(k["volume"]),
                            taker_buy_volume=0.0,  # Bybit kline carries no taker split
                            seq=self._seq_counters[symbol],
                        )
                        await self._queue.put(event)
