# ADR-0010 — Append-only JSONL for the order audit log

**Status:** Accepted · 2026-05-12

## Context

Rule R-8 requires every order/event to be reconstructable post-trade. We
need a substrate that is:

- append-only (no destructive writes after a record lands)
- human-readable for forensic review
- machine-queryable in pandas
- DVC-trackable for reproducibility
- cheap to write (≤ 1 ms per record)

Three candidates considered:

1. **Postgres / Timescale** — relational, but write latency and ops cost
   are wrong for ≤ 50 orders/day.
2. **Apache Iceberg** — overkill for the throughput; ops complexity high.
3. **Append-only JSONL** — one record per line, SHA-256 of every batch.

## Decision

`tradebot.oms.audit_log.AuditLog` writes one JSONL record per order/fill
event under `state/audit_log.jsonl`. Each record contains the 14 mandatory
fields defined by `tradebot.schemas.orders.OrderAuditSchema`:

```
event_ts, order_id, symbol, side, qty_base, notional_usdt, order_type,
signal_prob, kelly_fraction, model_version, git_sha, feature_hash,
portfolio_weight, pre_trade_cost_bps
```

Optional post-trade fields populate when the fill arrives:

```
fill_price, fill_ts, post_trade_cost_bps, circuit_breaker_state
```

A separate `audit_log.sha256.txt` file is rotated each session: it stores
the SHA-256 of the JSONL since the previous rotation. Tamper detection is
just `compare(stored_hash, recomputed_hash)`.

## Reason

- Native Python; no driver required.
- pandas `read_json(lines=True)` parses the whole file in one call.
- Grep-able with standard Unix tools; `jq` makes ad-hoc queries trivial.
- File rotation is `touch` + symlink swap — trivial in a K8s sidecar.
- Append-only by construction (the only write is `open("a").write(...)`).

## Trade-off

- Real-time dashboards over millions of records would need a sink
  (Loki, ClickHouse). Mitigation: we don't have that volume.
- No native indexing → pandas filter must scan the file. For an annual
  archive (~18 k records at 50 orders/day), full-scan is ≤ 200 ms.
- Concurrent writers unsafe. Mitigation: one writer (the live engine);
  external readers open with `pd.read_json(..., lines=True)` which copies.

## References

- AFML chapter 17 (Structural Breaks), motivates audit completeness.
- Rule R-8 in `REFACTOR_BLUEPRINT_v3.md`.
