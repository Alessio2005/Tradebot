# Runbook — Tradebot live engine

> On-call audience. If pager rings, start here.

## Decision tree

```
┌─ Slack alert received ─┐
│                        │
│  CIRCUIT_BREAKER       ├──► §1. Halt acknowledged → §3. Forensics
│  FEED_TIMEOUT          ├──► §2. Feed outage      → §3 if state ≠ flat
│  DRIFT_PSI > 0.20      ├──► §4. Drift incident
│  DAILY_LOSS > 3%       ├──► already halted (see §1)
│  HASH_MISMATCH         ├──► §5. Model mismatch
└────────────────────────┘
```

## 1. CIRCUIT_BREAKER fired

The engine has self-halted. **Do not flip `live.mode` back to `live` until §3
forensics complete.**

```bash
# Inspect last 5 halt events
tail -n 5 state/circuit_log.jsonl | jq

# Confirm all positions flat
python -c "from tradebot.compliance.position_report import generate_position_report; \
           print(generate_position_report.__doc__)"   # for usage
```

Order of operations:
1. Confirm `state/circuit_log.jsonl` last entry. Read the `halt_reason` field.
2. Verify positions are flat: query exchange OR inspect `state/positions.json`.
3. If not flat: do **not** restart the engine; manually flatten via the
   exchange UI and document in incident log.
4. Open the relevant section below by `halt_reason`.

## 2. FEED_TIMEOUT > 30 s

Likely upstream issue (Binance WS or your network egress). The engine has
halted; positions are still open until manual action.

```bash
# Confirm WS health from another host:
wscat -c "wss://stream.binance.com:9443/ws/btcusdt@kline_1h" \
  | head -n 3

# Inspect engine logs:
kubectl -n tradebot logs -l app.kubernetes.io/component=live-engine \
  --tail=200 --since=15m
```

If positions are still open and the market is hostile:
- flatten via REST API (use the manual cancellation script).
- update incident log with the timestamp of manual intervention.

## 3. Forensics (post-halt)

```bash
# 1. State snapshot
cp state/audit_log.jsonl state/circuit_log.jsonl /tmp/forensics_$(date +%F)/

# 2. PR-locked branch
git checkout -b incident/$(date +%F)_$(echo $RANDOM | head -c4)

# 3. Run post-mortem helper
python scripts/paper_trade_report.py --state-dir state/ --output incident_report.md

# 4. Verify reconciler vs exchange
python -c "from tradebot.oms.reconciler import reconcile_with_exchange; \
           reconcile_with_exchange(...)"   # adapt to actual signature
```

Required artefacts in the post-mortem:
- last 100 audit-log records
- circuit-log entry
- equity-curve plot for the last 24 h
- the `feature_hash` of the loaded model
- the exchange WS gap (start_ts, stop_ts, gap duration)
- the model `git_sha` and DVC hash

Once these are attached to a Linear/Jira ticket, you may roll a new pod:

```bash
kubectl -n tradebot rollout restart deployment/tradebot-live
```

## 4. Drift incident (PSI > 0.20)

The feature distribution has shifted vs the reference period. The engine
keeps running — this is an *advisory* alert.

```bash
# Read the latest drift report:
cat artefacts/monitoring/drift_report.json | jq '.critical_features'
```

If critical features ≥ 3, hold the next model promotion. Open a research
ticket and request:
- regenerate the reference period from the latest 90 days
- re-train CPCV with the new reference baseline
- re-run `apps/monitor_drift` and confirm PSI < 0.10 on the new baseline

## 5. HASH_MISMATCH

The `feature_hash` computed at startup differs from the one stored alongside
the model in the registry. **Refuse to trade.**

Common causes:
- Manual edit of `conf/symbols/*.yaml` or `conf/strategy/*.yaml`
- Model promoted with old `params.yaml`
- Mid-deploy: feature columns reordered (rare; pinned via Pandera)

Steps:
1. `git log` `params.yaml`, `conf/`, `src/tradebot/features/` since last
   successful boot.
2. Decide: revert the env, or re-promote the model with the new feature
   hash. Either path requires an MRM update.

## 6. Manual halt

To force-stop the engine without rolling back:

```bash
kubectl -n tradebot scale deployment/tradebot-live --replicas=0
```

The pre-stop hook gracefully cancels open orders (60 s grace). Restart:
```bash
kubectl -n tradebot scale deployment/tradebot-live --replicas=1
```

## 7. Disaster recovery

State volume (`tradebot-live-state` PVC):
- back up `state/` once per hour via cron.
- restore from the latest snapshot before pod restart if the PVC is corrupted.

Artefact volume (`tradebot-artefacts` PVC):
- regenerable from `dvc pull` against the S3/MinIO remote.
- worst case: 6 h cold rebuild (see nightly DAG timing in `dvc.yaml`).

## 8. Escalation matrix

| Severity | Trigger                                       | First responder | Escalate after |
|----------|-----------------------------------------------|-----------------|----------------|
| SEV-1    | Open position with engine halted, market hostile | quant on-call | 10 min         |
| SEV-2    | Engine halted, flat                           | quant on-call   | 30 min         |
| SEV-3    | Drift alert with running engine               | research        | 4 h            |
| SEV-4    | Successful halt during research-hour window   | research        | next morning   |

## 9. Contacts

- Quant on-call: see PagerDuty rotation `tradebot-live`
- Engineering on-call: see PagerDuty `tradebot-infra`
- Slack: `#tradebot-incidents` (immutable, audit-logged)
