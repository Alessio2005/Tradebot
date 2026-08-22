# ADR-0007 — asyncio over threading for the live engine

**Status:** Accepted · 2026-05-12

## Context

Real-time bar ingestion, model prediction, optimisation, and order
placement. Workload is dominated by network I/O (Binance WS + REST,
optional Slack). CPU-bound steps (Numba kernels, CatBoost predict) are
short relative to network latency.

Three candidates evaluated:

1. **Threading + queues** — Pythonic, but introduces shared state on
   position tracker and order book.
2. **Multiprocessing** — IPC overhead too high for per-bar cycle (≤500 ms).
3. **asyncio** — cooperative concurrency, single-thread event loop, no
   shared-state races.

## Decision

`tradebot.live.engine.LiveEngine` is an `asyncio.Task` graph driven by
a single event loop. Coroutines:

- `Feed.stream()`     → produces `Bar` events.
- `FeatureUpdater.update(bar)`     → CPU-bound; ran in `ThreadPoolExecutor`.
- `SignalRunner.predict(...)`     → CPU-bound; ran in same executor.
- `PortfolioController.optimize(...)` → numpy-bound; same executor.
- `ExecutionController.run(...)`  → coordinates OMS coroutines.
- `OMS.place_order(...)`          → REST/WS sender.

Outside the event loop: a `aiohttp.web.Application` exposing `/metrics`
on port 8000 for Prometheus.

## Reason

- I/O-bound dominant: asyncio is purpose-built.
- Deterministic order of events: `await order_sent; await fill_seen` reads
  like causal chain.
- No mutex needed on `PositionTracker`: only one coroutine touches it.
- CPU-bound bits are isolated via `loop.run_in_executor`. A
  ThreadPoolExecutor with `max_workers=2` is more than enough for 1h-bar
  cadence.

## Trade-off

- Library compat: must use `aiohttp` for Binance, not `requests` — fine.
- Blocking calls must be wrapped — discipline issue, lint-rule guards
  forbidden patterns (`from typing import Awaitable`).
- Debugging: stacks are split across coroutines. Mitigation: structured
  logging via `structlog`, every record tagged with `coroutine_id`.

## References

- Python docs — `asyncio` design rationale.
- Aiohttp performance white-paper (2023).
