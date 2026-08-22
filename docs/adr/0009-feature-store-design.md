# ADR-0009 — Feature Store as Parquet partitions, not a database

**Status:** Accepted · 2026-05-12

## Context

The live engine needs to read features for the latest bar within ~50 ms
after bar close, and append new features at the same cadence. The research
pipeline produces features in bulk, but cannot rerun the full DAG on every
live bar. A feature store solves the gap.

Three options evaluated:

1. **TimescaleDB hypertables** — operationally complex; requires DB pod
   in K8s and SLA contracts.
2. **Redis time-series** — fast reads, but evict policy and persistence
   semantics fight us when audit requires immutability.
3. **DuckDB-managed Parquet partitions** — embedded, no extra process,
   DVC-trackable.

## Decision

`tradebot.featurestore.store.FeatureStore` implements an append-only
Parquet store under `artefacts/feature_store/`. Layout:

```
artefacts/feature_store/
├── BTCUSDT/
│   └── year=2026/
│       └── month=05/
│           └── 0001.parquet
├── ETHUSDT/...
└── _schema.json   # Pandera-derived FeatureSchema
```

DuckDB drives the read API:

```python
duckdb.read_parquet("artefacts/feature_store/BTCUSDT/**/*.parquet")
  .filter(ts >= start, ts <= end)
  .to_df()
```

Write API enforces:

- Pandera schema validation on every `append`.
- No timestamp older than `latest_timestamp(symbol)` (R-1 causal fence).
- File rotation at 10 MB or end of month, whichever comes first.

## Reason

- Zero extra dependencies in production — DuckDB is `pip install duckdb`.
- Parquet files DVC-track cleanly; reverting a corrupt feature is `dvc
  checkout`.
- Read latency: < 5 ms for the last 240 bars of a single symbol
  (benchmarked on a 100 MB partition).
- Write latency: < 10 ms for one bar (target met).

## Trade-off

- Not suited for tick-level data — 1h bars cap throughput requirements
  comfortably.
- Concurrent writers are not safe; the live engine is the sole writer.
  Research reads share-locked.
- Append-only means `delete-then-rewrite` is the only correction
  pattern — keeps audit history intact.

## References

- DuckDB Parquet performance benchmark (DuckDB Labs, 2023).
- AFML chapter 5 (Fractional Differentiation), motivates causal fence.
