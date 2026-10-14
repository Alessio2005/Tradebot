# tests/unit/test_live_feed.py
"""`live/feed.py`: barvorming uit trades, paper-replay, quotes, funding en de achtergrondtaken.

Geen netwerk: `websockets.connect` en `aiohttp.ClientSession` zijn vervangen door
scripts met vooraf bepaalde berichten.
"""
from __future__ import annotations

import asyncio
import json
import logging

import pandas as pd
import pytest

from tradebot.live import feed as fd
from tradebot.live.feed import BarEvent, Feed, FeedConfig, _PartialBar

SYM = "ETHUSDT"
BAR_MS = 5_000


def _run(coro):
    return asyncio.run(coro)


def _drain(queue: asyncio.Queue) -> list:
    items = []
    while not queue.empty():
        items.append(queue.get_nowait())
    return items


# --------------------------------------------------------------------------- #
# bar events and partial bars
# --------------------------------------------------------------------------- #
class TestPartialBar:
    def test_a_bar_aggregates_ohlcv_and_taker_buy_volume(self) -> None:
        bar = _PartialBar(12_345, 100.0, 1.0, True, BAR_MS)
        bar.update(103.0, 2.0, False)
        bar.update(99.0, 0.5, True)
        bar.update(101.0, 1.5, False)
        assert (bar.bar_id, bar.open, bar.high, bar.low, bar.close) == (2, 100.0, 103.0, 99.0, 101.0)
        assert bar.volume == pytest.approx(5.0) and bar.taker_buy_vol == pytest.approx(1.5)

    def test_the_event_closes_one_millisecond_before_the_next_boundary(self) -> None:
        event = _PartialBar(12_345, 100.0, 1.0, False, BAR_MS).to_bar_event(SYM, BAR_MS, seq=7)
        assert event.ts == pd.Timestamp(14_999, unit="ms", tz="UTC") and event.seq == 7
        assert event.taker_buy_volume == 0.0 and event.funding_rate == 0.0

    def test_bar_events_are_immutable(self) -> None:
        event = BarEvent(SYM, pd.Timestamp("2024-01-01", tz="UTC"), 1, 2, 0.5, 1.5, 10)
        with pytest.raises(AttributeError):
            event.close = 9.0                                            # type: ignore[misc]


# --------------------------------------------------------------------------- #
# paper replay
# --------------------------------------------------------------------------- #
def _bars(start: str, n: int, prefix: str = "") -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq="min", tz="UTC")
    cols = {"open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 10.0}
    return pd.DataFrame({prefix + k: v for k, v in cols.items()}, index=idx)


class TestReplay:
    def test_bars_of_all_symbols_come_out_in_time_order_followed_by_an_end_marker(self) -> None:
        queue: asyncio.Queue = asyncio.Queue()
        feed = Feed(FeedConfig(symbols=["A", "B"], paper_mode=True), queue,
                    {"A": _bars("2024-01-01 00:00", 3), "B": _bars("2024-01-01 00:01", 2)})

        async def main():
            await feed.start()
            await asyncio.gather(*feed._tasks)
        _run(main())
        items = _drain(queue)
        assert [(e.symbol, e.ts.minute) for e in items[:-1]] == [("A", 0), ("A", 1), ("B", 1), ("A", 2), ("B", 2)]
        assert items[-1] is None and items[0].close == 1.5 and items[0].taker_buy_volume == 0.0

    def test_short_column_names_and_naive_timestamps_are_understood(self) -> None:
        df = pd.DataFrame({"o": [1.0], "h": [3.0], "l": [0.5], "c": [2.0], "v": [7.0],
                           "taker_buy_volume": [4.0], "funding_rate": [0.0002]},
                          index=pd.DatetimeIndex(["2024-01-01 00:00"]))
        queue: asyncio.Queue = asyncio.Queue()
        feed = Feed(FeedConfig(symbols=["A"]), queue, {"A": df})
        feed._running = True
        _run(feed._replay())
        event = _drain(queue)[0]
        assert (event.open, event.high, event.low, event.close, event.volume) == (1.0, 3.0, 0.5, 2.0, 7.0)
        assert event.taker_buy_volume == 4.0 and event.funding_rate == 0.0002 and str(event.ts.tz) == "UTC"

    def test_a_stopped_feed_stops_replaying_and_a_delay_is_honoured(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sleeps: list[float] = []

        async def fake_sleep(seconds: float) -> None:
            sleeps.append(seconds)
        monkeypatch.setattr(fd.asyncio, "sleep", fake_sleep)
        queue: asyncio.Queue = asyncio.Queue()
        feed = Feed(FeedConfig(symbols=["A"], replay_delay_s=0.25), queue, {"A": _bars("2024-01-01", 3)})
        feed._running = True
        _run(feed._replay())
        assert sleeps == [0.25] * 3
        stopped = Feed(FeedConfig(symbols=["A"]), asyncio.Queue(), {"A": _bars("2024-01-01", 3)})
        _run(stopped._replay())                                           # _running is False
        assert _drain(stopped._queue) == [None]


# --------------------------------------------------------------------------- #
# start, stop, background tasks
# --------------------------------------------------------------------------- #
def _stub_loops(feed: Feed, started: list[str]) -> None:
    for name in ("_replay", "_live_ws_aggtrade", "_live_ws_bookticker", "_live_funding_poller"):
        async def stub(_n=name):
            started.append(_n)
        setattr(feed, name, stub)


class TestStartStop:
    def test_paper_mode_starts_only_the_replay(self) -> None:
        feed, started = Feed(FeedConfig(symbols=[SYM], paper_mode=True), asyncio.Queue()), []
        _stub_loops(feed, started)

        async def main():
            await feed.start()
            await asyncio.gather(*feed._tasks)
        _run(main())
        assert started == ["_replay"] and feed._running

    def test_live_mode_starts_the_trade_book_and_funding_loops(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("TRADEBOT_FEED_SPOT_FALLBACK", raising=False)
        feed, started = Feed(FeedConfig(symbols=[SYM], paper_mode=False), asyncio.Queue()), []
        _stub_loops(feed, started)

        async def main():
            await feed.start()
            await asyncio.gather(*feed._tasks)
        _run(main())
        assert sorted(started) == ["_live_funding_poller", "_live_ws_aggtrade", "_live_ws_bookticker"]

    def test_the_spot_fallback_has_no_funding_poller(self, monkeypatch: pytest.MonkeyPatch,
                                                     caplog: pytest.LogCaptureFixture) -> None:
        monkeypatch.setenv("TRADEBOT_FEED_SPOT_FALLBACK", "1")
        feed, started = Feed(FeedConfig(symbols=[SYM], paper_mode=False), asyncio.Queue()), []
        _stub_loops(feed, started)

        async def main():
            await feed.start()
            await asyncio.gather(*feed._tasks)
        with caplog.at_level(logging.WARNING, logger=fd.logger.name):
            _run(main())
        assert "_live_funding_poller" not in started and "spot fallback active" in caplog.text

    def test_stop_cancels_the_running_tasks(self) -> None:
        feed = Feed(FeedConfig(symbols=[SYM], paper_mode=True), asyncio.Queue())

        async def forever():
            await asyncio.sleep(3600)
        feed._replay = forever                                            # type: ignore[method-assign]

        async def main():
            await feed.start()
            await asyncio.sleep(0)
            task = next(iter(feed._tasks))
            await feed.stop()
            await asyncio.gather(task, return_exceptions=True)
            return task.cancelled()
        assert _run(main()) is True and feed._running is False and feed.task_failure is None

    def test_a_crashing_task_is_remembered_once_and_logged_critically(self, caplog: pytest.LogCaptureFixture) -> None:
        feed = Feed(FeedConfig(symbols=[SYM]), asyncio.Queue())

        async def boom_a():
            raise ValueError("first")

        async def boom_b():
            raise KeyError("second")

        async def main():
            a, b = feed._spawn(boom_a(), "a"), feed._spawn(boom_b(), "b")
            await asyncio.gather(a, b, return_exceptions=True)
            await asyncio.sleep(0)
        with caplog.at_level(logging.CRITICAL, logger=fd.logger.name):
            _run(main())
        assert isinstance(feed.task_failure, ValueError) and feed._tasks == set()
        assert caplog.text.count("achtergrondtaak") == 2


# --------------------------------------------------------------------------- #
# quotes and funding state
# --------------------------------------------------------------------------- #
class TestQuotesAndFunding:
    def test_a_fresh_quote_is_returned_and_a_stale_or_missing_one_is_not(self) -> None:
        feed = Feed(FeedConfig(symbols=[SYM]), asyncio.Queue())
        now_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
        assert feed.get_best_quote(SYM) is None
        feed._best_quote[SYM] = (99.0, 101.0, now_ms)
        assert feed.get_best_quote(SYM) == (99.0, 101.0)
        feed._best_quote[SYM] = (99.0, 101.0, now_ms - 10_000)
        assert feed.get_best_quote(SYM) is None

    def test_funding_defaults_to_zero(self) -> None:
        feed = Feed(FeedConfig(symbols=[SYM]), asyncio.Queue())
        assert feed.get_funding_rate(SYM) == 0.0
        feed._funding_rate[SYM] = 0.0001
        assert feed.get_funding_rate(SYM) == 0.0001

    @pytest.mark.parametrize(("payload", "expected"), [
        ({"retCode": 10001}, 0.0),
        ({"retCode": 0, "result": {"list": []}}, 0.0),
        ({"retCode": 0}, 0.0),
        ({"retCode": 0, "result": {"list": [{"fundingRate": "0.000125"}]}}, 0.000125),
        ({"retCode": 0, "result": {"list": [{"fundingRate": ""}]}}, 0.0),
        ({"retCode": 0, "result": {"list": [{}]}}, 0.0),
        ({}, 0.0),
    ])
    def test_the_bybit_funding_payload_is_parsed_defensively(self, payload: dict, expected: float) -> None:
        assert Feed._parse_bybit_funding(payload) == pytest.approx(expected)


class TestFundingTag:
    def _bar(self, ms: int) -> BarEvent:
        return BarEvent(SYM, pd.Timestamp(ms, unit="ms", tz="UTC"), 1, 1, 1, 1, 1)

    def test_the_first_bar_of_a_funding_window_carries_the_rate_and_later_ones_do_not(self) -> None:
        feed = Feed(FeedConfig(symbols=[SYM]), asyncio.Queue())
        feed._funding_rate[SYM] = 0.0002
        window = 8 * 3600 * 1000
        first = feed._maybe_attach_funding(self._bar(window + 1), BAR_MS)
        again = feed._maybe_attach_funding(self._bar(window + 6_000), BAR_MS)
        nxt = feed._maybe_attach_funding(self._bar(2 * window + 1), BAR_MS)
        assert (first.funding_rate, again.funding_rate, nxt.funding_rate) == (0.0002, 0.0, 0.0002)

    def test_a_zero_rate_marks_the_window_without_tagging(self) -> None:
        feed = Feed(FeedConfig(symbols=[SYM]), asyncio.Queue())
        bar = self._bar(1_000)
        assert feed._maybe_attach_funding(bar, BAR_MS) is bar and feed._last_funding_bar_id[SYM] == 0
        feed._funding_rate[SYM] = 0.5
        assert feed._maybe_attach_funding(self._bar(2_000), BAR_MS).funding_rate == 0.0     # zelfde venster

    def test_a_non_positive_interval_falls_back_to_eight_hours(self) -> None:
        feed = Feed(FeedConfig(symbols=[SYM], funding_interval_min=0), asyncio.Queue())
        feed._funding_rate[SYM] = 0.0003
        assert feed._maybe_attach_funding(self._bar(8 * 3600 * 1000 + 1), BAR_MS).funding_rate == 0.0003

    def test_a_shorter_funding_cycle_is_respected(self) -> None:
        feed = Feed(FeedConfig(symbols=[SYM], funding_interval_min=60), asyncio.Queue())
        feed._funding_rate[SYM] = 0.0004
        assert feed._maybe_attach_funding(self._bar(3_600_000 + 1), BAR_MS).funding_rate == 0.0004


# --------------------------------------------------------------------------- #
# websocket loops
# --------------------------------------------------------------------------- #
class _Ws:
    def __init__(self, feed: Feed, messages: list, stop_after: int | None = None) -> None:
        self.feed, self.messages, self.stop_after, self.sent = feed, messages, stop_after, []

    async def send(self, msg: str) -> None:
        self.sent.append(json.loads(msg))

    def __aiter__(self):
        return self._messages()

    async def _messages(self):
        for k, m in enumerate(self.messages, start=1):
            yield m if isinstance(m, str) else json.dumps(m)
            if self.stop_after is not None and k == self.stop_after:
                self.feed._running = False
        self.feed._running = False


class _Connect:
    def __init__(self, ws: _Ws, calls: list) -> None:
        self.ws, self.calls = ws, calls

    def __call__(self, url: str, **kwargs):
        self.calls.append((url, kwargs))
        return self

    async def __aenter__(self) -> _Ws:
        return self.ws

    async def __aexit__(self, *exc) -> bool:
        return False


def _connect(monkeypatch: pytest.MonkeyPatch, feed: Feed, messages: list, stop_after: int | None = None):
    ws, calls = _Ws(feed, messages, stop_after), []
    monkeypatch.setattr("websockets.connect", _Connect(ws, calls))
    return ws, calls


def _trade(price: float, qty: float, ms: int, side: str = "Buy", sym: str = SYM) -> dict:
    return {"s": sym, "p": str(price), "v": str(qty), "T": ms, "S": side}


def _trades(*events: dict) -> dict:
    return {"topic": f"publicTrade.{SYM}", "data": list(events)}


class TestTradeStream:
    def test_trades_are_aggregated_into_epoch_aligned_bars_with_sequence_numbers(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("TRADEBOT_FEED_SPOT_FALLBACK", raising=False)
        queue: asyncio.Queue = asyncio.Queue()
        feed = Feed(FeedConfig(symbols=[SYM], bar_seconds=5, paper_mode=False), queue)
        feed._running = True
        feed._funding_rate[SYM] = 0.0001
        ws, calls = _connect(monkeypatch, feed, [
            {"topic": "tickers.ETHUSDT", "data": {}},                              # ander topic
            _trades(_trade(100.0, 1.0, 1_000, "Buy"), _trade(5.0, 9.0, 1_500, "Buy", sym="XRPUSDT")),
            _trades(_trade(102.0, 2.0, 2_000, "Sell"), _trade(99.0, 1.0, 4_000, "buy")),
            _trades(_trade(101.0, 3.0, 6_000)),                                    # volgende bar: emit
            _trades(_trade(103.0, 1.0, 11_000)),
        ])
        _run(feed._live_ws_aggtrade())
        first, second = _drain(queue)
        assert calls[0][0] == "wss://stream.bybit.com/v5/public/linear" and calls[0][1] == {"ping_interval": 20}
        assert ws.sent == [{"op": "subscribe", "args": [f"publicTrade.{SYM}"]}]
        assert (first.open, first.high, first.low, first.close) == (100.0, 102.0, 99.0, 99.0)
        assert first.volume == 4.0 and first.taker_buy_volume == 2.0 and first.seq == 1
        assert first.ts == pd.Timestamp(4_999, unit="ms", tz="UTC") and first.funding_rate == 0.0001
        assert second.seq == 2 and second.open == 101.0 and second.funding_rate == 0.0

    def test_the_stream_stops_when_the_feed_is_stopped_mid_stream(self, monkeypatch: pytest.MonkeyPatch) -> None:
        queue: asyncio.Queue = asyncio.Queue()
        feed = Feed(FeedConfig(symbols=[SYM], paper_mode=False), queue)
        feed._running = True
        _connect(monkeypatch, feed, [_trades(_trade(1.0, 1.0, 1_000)), _trades(_trade(2.0, 1.0, 9_000))], stop_after=1)
        _run(feed._live_ws_aggtrade())
        assert _drain(queue) == []                                                 # tweede bericht is niet meer verwerkt

    def test_the_spot_fallback_connects_to_the_spot_endpoint(self, monkeypatch: pytest.MonkeyPatch,
                                                             caplog: pytest.LogCaptureFixture) -> None:
        monkeypatch.setenv("TRADEBOT_FEED_SPOT_FALLBACK", "1")
        feed = Feed(FeedConfig(symbols=[SYM], paper_mode=False), asyncio.Queue())
        feed._running = True
        _, calls = _connect(monkeypatch, feed, [])
        with caplog.at_level(logging.CRITICAL, logger=fd.logger.name):
            _run(feed._live_ws_aggtrade())
        assert calls[0][0] == "wss://stream.bybit.com/v5/public/spot" and "Funding-rate parity LOST" in caplog.text


class TestBookStream:
    def test_best_bid_and_ask_follow_snapshots_and_one_sided_deltas(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("TRADEBOT_FEED_SPOT_FALLBACK", raising=False)
        feed = Feed(FeedConfig(symbols=[SYM], paper_mode=False), asyncio.Queue())
        feed._running = True
        seen: list = []

        def book(bids=None, asks=None, ts=1_700_000_000_000, sym=SYM, topic=f"orderbook.1.{SYM}") -> dict:
            return {"topic": topic, "ts": ts, "data": {"s": sym, "b": bids or [], "a": asks or []}}

        messages = [
            {"topic": "publicTrade.ETHUSDT", "data": {}},                         # ander topic
            {"topic": f"orderbook.1.{SYM}", "data": {}},                          # zonder symbool
            book(sym="XRPUSDT", bids=[["1", "1"]], asks=[["2", "1"]]),            # onbekend symbool
            book(bids=[["99.5", "3"]]),                                           # alleen bid, nog geen eerdere quote
            book(bids=[["99.0", "1"]], asks=[["100.0", "1"]]),                    # snapshot
            book(asks=[["100.5", "2"]], ts=0),                                    # alleen ask, ts ontbreekt
        ]
        ws, calls = _connect(monkeypatch, feed, messages)
        original = Feed.get_best_quote

        async def run():
            await feed._live_ws_bookticker()
            seen.append(dict(feed._best_quote))
        _run(run())
        bid, ask, ts_ms = feed._best_quote[SYM]
        assert (bid, ask) == (99.0, 100.5) and ts_ms > 1_700_000_000_000          # ts=0 wordt de wandklok
        assert ws.sent == [{"op": "subscribe", "args": [f"orderbook.1.{SYM}"]}] and original is Feed.get_best_quote

    def test_the_spot_fallback_and_an_early_stop_are_handled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TRADEBOT_FEED_SPOT_FALLBACK", "1")
        feed = Feed(FeedConfig(symbols=[SYM], paper_mode=False), asyncio.Queue())
        feed._running = True
        quote = {"topic": f"orderbook.1.{SYM}", "ts": 5, "data": {"s": SYM, "b": [["1", "1"]], "a": [["2", "1"]]}}
        _, calls = _connect(monkeypatch, feed, [quote, quote], stop_after=1)
        _run(feed._live_ws_bookticker())
        assert calls[0][0].endswith("/spot") and feed._best_quote[SYM] == (1.0, 2.0, 5)


class TestKlineStream:
    def test_only_confirmed_bars_become_events(self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
        queue: asyncio.Queue = asyncio.Queue()
        feed = Feed(FeedConfig(symbols=[SYM], interval="15m", paper_mode=False), queue)
        feed._running = True
        kline = {"open": "1", "high": "3", "low": "0.5", "close": "2", "volume": "9", "end": 1_700_000_059_999}
        messages = [
            {"topic": "publicTrade.ETHUSDT", "data": []},
            {"topic": f"kline.15.{SYM}", "data": [{**kline, "confirm": False}, {**kline, "confirm": True}]},
            {"topic": f"kline.15.{SYM}", "data": [{**kline, "confirm": True}]},
        ]
        ws, _ = _connect(monkeypatch, feed, messages)
        with caplog.at_level(logging.WARNING, logger=fd.logger.name):
            _run(feed._live_ws_kline())
        events = _drain(queue)
        assert ws.sent == [{"op": "subscribe", "args": [f"kline.15.{SYM}"]}] and "LEGACY" in caplog.text
        assert [e.seq for e in events] == [1, 2] and events[0].close == 2.0 and events[0].taker_buy_volume == 0.0
        assert events[0].ts == pd.Timestamp(1_700_000_059_999, unit="ms", tz="UTC")

    def test_second_resolutions_fall_back_to_the_one_minute_stream_and_a_stop_ends_the_loop(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        feed = Feed(FeedConfig(symbols=[SYM], interval="5s", paper_mode=False), asyncio.Queue())
        feed._running = True
        ws, _ = _connect(monkeypatch, feed, [{"topic": f"kline.1.{SYM}", "data": []}] * 2, stop_after=1)
        _run(feed._live_ws_kline())
        assert ws.sent[0]["args"] == [f"kline.1.{SYM}"]


# --------------------------------------------------------------------------- #
# funding poller
# --------------------------------------------------------------------------- #
class _HttpResp:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    async def json(self) -> dict:
        return self.payload

    async def __aenter__(self) -> _HttpResp:
        return self

    async def __aexit__(self, *exc) -> bool:
        return False


class TestFundingPoller:
    def test_each_symbol_is_polled_and_its_rate_cached(self, monkeypatch: pytest.MonkeyPatch) -> None:
        feed = Feed(FeedConfig(symbols=["AAA", "BBB"], paper_mode=False), asyncio.Queue())
        feed._running = True
        urls: list[str] = []
        payloads = {"AAA": {"retCode": 0, "result": {"list": [{"fundingRate": "0.0002"}]}},
                    "BBB": {"retCode": 10001}}

        class Session:
            def __init__(self, *a, **kw) -> None:
                pass

            async def __aenter__(self) -> Session:
                return self

            async def __aexit__(self, *exc) -> bool:
                return False

            def get(self, url: str, timeout=None):
                urls.append(url)
                return _HttpResp(payloads[url.rsplit("=", 1)[-1]])

        async def stop_after_first_pass(_s: float) -> None:
            feed._running = False
        monkeypatch.setattr("aiohttp.ClientSession", Session)
        monkeypatch.setattr(fd.asyncio, "sleep", stop_after_first_pass)
        _run(feed._live_funding_poller(poll_interval_s=0.0))
        assert urls == [f"{Feed._BYBIT_TICKERS_URL}?category=linear&symbol={s}" for s in ("AAA", "BBB")]
        assert feed.get_funding_rate("AAA") == pytest.approx(0.0002) and feed.get_funding_rate("BBB") == 0.0
