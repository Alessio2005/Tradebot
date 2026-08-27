# Runbook — Tradebot live engine

> **STATUS — Phase 7/8, Stage A-2.** Alles ónder §0 beschrijft nog het
> **Wave-tijdperk** en niet het systeem dat Phase 5 heeft gebouwd: de beslisboom
> hangt aan `state/circuit_log.jsonl` en `live.mode`, terwijl de soevereine
> risicolaag met `HaltStore`, `RiskEngine` en `ExecutionContext` er niet in
> voorkomt. **Die herschrijving is Stage D-4 en is nog niet gedaan.** Deze
> sectie staat bewust vooraan zodat niemand het verouderde deel voor actueel
> aanziet.

---

## 0. De referentie-interpreter

Vanaf 2026-08-27 is er precies één interpreter waarop een meting geldig is:

```
D:/venv/tradebot/Scripts/python.exe      Python 3.13.0
```

### 0.1 Waarom dit hier staat

De verhuizing van de C-schijf naar de D-schijf liet een omgeving achter waarin
de import zelf half kapot was. `__editable__.tradebot-0.4.0.pth` wees naar
`C:\Users\algul\Documents\Tradebot\src` — een map die nog bestond maar leeg was.
`import tradebot` slaagde als lege namespace-package; `import tradebot.registry`
faalde met `ModuleNotFoundError`. Elke `python apps/...` steunde daarmee op een
package die er niet was.

Erger, en pas gevonden door te meten: **de testketen was nooit gepind.**
`requirements.lock` pint 476 runtime-packages en nul testtools. Twee machines
die allebei dat bestand volgen, kregen verschillende pytest- en ruff-versies —
en de lint-gate gaf daardoor een ander oordeel over dezelfde broncode:
7 bevindingen onder ruff 0.15.12, 78 onder 0.16.4.

### 0.2 De omgeving opbouwen

```bash
python -m venv D:/venv/tradebot
D:/venv/tradebot/Scripts/python -m pip install -r requirements.lock
D:/venv/tradebot/Scripts/python -m pip install -r requirements-dev.lock
D:/venv/tradebot/Scripts/python -m pip install -e . --no-deps
```

**Beide lockfiles, altijd.** `requirements.lock` alleen levert een omgeving die
de code draait maar het bewijs niet reproduceerbaar meet.

### 0.3 Verifiëren dat de omgeving klopt

```bash
D:/venv/tradebot/Scripts/python -c "import tradebot; print(tradebot.__file__)"
```
Moet `D:\Tradebot\src\tradebot\__init__.py` teruggeven — **nooit** een C:-pad.

```bash
D:/venv/tradebot/Scripts/python -c "from tradebot.registry.lineage import get_git_sha; print(get_git_sha())"
```
Moet een resolvable sha teruggeven. Een lege string betekent dat D-9 heropend is
en dat elk artefact dat vanaf dat moment wordt geschreven **invalide** is.

```bash
D:/venv/tradebot/Scripts/python -m pytest tests/unit -q -p no:randomly
```
`1253 passed, 1 skipped`. De skip vraagt een niet-gecachet broad-perp panel.

### 0.4 De drie ratchets — alle drie moeten groen zijn

```bash
python scripts/check_hardcoded_params.py --strict
```
```bash
python scripts/audit_fallbacks.py --strict
```
```bash
python -m pytest -q -p no:randomly
```

De eerste twee geven **exit 0**; de tweede meldt 37 adviezen en 0 blokkerend.
De suite geeft **exact 4 failures**.

> **Die vier failures horen rood te staan.** Het zijn de pre-geregistreerde
> killgates op `cm_carry` en `cm_tsmom` — KG-B1 in-sample, tweemaal KG-B2
> residual alpha, en KG-B3 out-of-sample. Zijn het er meer, minder, of staan er
> andere namen: dat is een regressie, geen ruis. Noteer de namen, niet alleen
> het aantal.

### 0.5 Na elke verplaatsing van de werkkopie

Verplaats je de boom ooit weer, doe dan **eerst** dit — vóór je iets meet:

```bash
find . -name "__pycache__" -type d -not -path "./.git/*" -prune -exec rm -rf {} +
```
```bash
find . -name "*.pyc" -not -path "./.git/*" -delete
```
```bash
python -m pip uninstall -y tradebot && python -m pip install -e . --no-deps
```

`tests/unit/test_repository_hygiene.py` bewaakt dit: hij faalt zodra een `.pyc`
in de boom een `co_filename` buiten de repository-root draagt. Na de verhuizing
deden **351 van 351** dat, en elke traceback citeerde daardoor een pad dat niet
bestond.

### 0.6 Waar de historie staat

```
D:/backup/tradebot-<sha>.bundle          geverifieerde volledige historie
D:/backup/tradebot-CDRIVE-orphan.bundle  de opgeruimde C-schijf-repo
```

Een bundle is één bestand, bevat de volledige historie en is offline
verifieerbaar met `git bundle verify <pad>`.

> **OPEN PUNT — no-go 2 staat nog ACTIEF.** Beide bundles staan op **dezelfde
> fysieke schijf** als de werkkopie. Er is nog geen remote. Zie
> `reports/phase7_foundation_report.md` §2.3.

---

## Wave-tijdperk (VEROUDERD — vervangen in Stage D-4)

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
