# Runbook — Tradebot live engine

> **Gecontroleerd op 2026-09-01** (Phase 7/8, Stage E-3).
> **LET OP — dit document is NIET geverifieerd.** Het beschrijft het
> Wave-tijdperk en niet het systeem dat Phase 5 heeft gebouwd. Stage D-4
> schrijft een herschrijving voor, getoetst doordat een tweede persoon het
> systeem er uitsluitend op start, halteert en herstart. Die herschrijving
> is nog niet gedaan; tot dat moment is dit document onbetrouwbaar voor
> operationeel gebruik.

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


---

## 0.7 De entrypoints — wat er is en hoe je het aanroept

> **Toegevoegd 2026-09-04, Phase 9 stap 12.** De nulmeting van die fase vond
> **36 van de 98 entrypoints die door niets werden genoemd** — niet door
> `dvc.yaml`, niet door de `Makefile`, niet door `[project.scripts]`, niet
> door `.github/workflows/` en niet door `docs/`. Een entrypoint dat nergens
> staat, wordt bij de volgende opruiming voor dood aangezien. Deze sectie is
> de verwijzing die dat voorkomt.

Drie soorten, en de grens is hard:

| Map | Wat er staat | Mag productiegedrag leveren |
|---|---|---|
| `apps/` | entrypoints van het platform | ja |
| `scripts/` | platformgereedschap: poorten, scanners, registergeneratoren | nee, maar de repo hangt ervan af |
| `research/` | wave-onderzoek; zie `research/README.md` | nee |

### Apps in de DVC-DAG

Deze tien worden door `dvc repro` aangeroepen. Zij vormen de authoritative
keten; alles wat zij bereiken is klasse A in `docs/CODE_REGISTER.md`.

| App | LOC | DVC-stage | Console-script | Waarvoor |
|---|---:|---|---|---|
| `apps/build_features.py` | 289 | `build_features` | `tb-build-features` | DAG Stage 1: Raw data → bars → features → events. |
| `apps/data_sync.py` | 50 | `data_sync` | `tb-data-sync` |  |
| `apps/run_data_adequacy.py` | 75 | `phase6_adequacy` | — | Data Adequacy Gate — meet alle vijf modelklassen VOOR de eerste fit. |
| `apps/run_econometric_diagnostics.py` | 69 | `phase6_econometrics` | — | Econometrische keten + fractionele differentiëring op elke reeks. |
| `apps/run_meta_labeling.py` | 232 | `phase6_h3_meta_labeling` | — | H3 — CatBoost als secondary model. Stappen 12 en 13, deliverable 22. |
| `apps/run_phase5_baseline.py` | 134 | `phase5_revaluation` | `tb-run-baseline` | Phase 5, §20 — herwaardeer de Phase 3-baseline door de volledige keten. |
| `apps/run_regime_benchmark.py` | 192 | `phase6_h2_regime_benchmark` | — | H2 — M0 Causal Vol-Buckets tegen de Markov-familie. Stap 11, deliverable 18. |
| `apps/run_vol_competition.py` | 134 | `phase6_h1_competition` | — | H1 — de QLIKE-competitie draaien. Phase 6 stap 7, deliverable 13. |
| `apps/train_cpcv.py` | 957 | `train_cpcv` | `tb-train-cpcv` | DAG Stage 3: best_params → CPCV ensemble + calibrators. |
| `apps/tune_hparams.py` | 418 | `tune_hparams` | `tb-tune-hparams` | DAG Stage 2: Feature artefacts → best Optuna hyperparams. |

### Overige apps

Zij staan niet in de DAG en draaien op verzoek. Dat is geen gebrek: een
governance-bevriezer of een paper-trader hoort niet in een reproductiepad.

| App | LOC | Console-script | Waarvoor |
|---|---:|---|---|
| `apps/alpha_combine.py` | 100 | — | apps/alpha_combine.py — IC-weighted signal combination stage (Stage 6). |
| `apps/build_equity_universe.py` | 50 | — |  |
| `apps/calibrate_impact.py` | 83 | — | Phase 5, deliverable 5 — kalibreer het marktimpactmodel, of bewijs dat het niet kan. |
| `apps/doctor.py` | 142 | — |  |
| `apps/feature_selection.py` | 443 | — | Per-asset feature selection pipeline. |
| `apps/featurestore_sync.py` | 86 | — | apps/featurestore_sync.py — incremental Parquet append for live feed (Stage 1b). |
| `apps/fetch_factors.py` | 39 | — |  |
| `apps/freeze_monitoring.py` | 82 | — | Bevries de monitoringdrempels vóór de 60-daagse klok. Stage D-3, D-5. |
| `apps/freeze_phase6_preregistrations.py` | 112 | — | Bevries de drie Phase 6-pre-registraties en registreer ze in de ledger. |
| `apps/freeze_preregistration.py` | 107 | — | Bevries een pre-registratie en registreer hem in de hypothese-ledger. |
| `apps/generate_mrm_report.py` | 369 | — | Generate and persist the Model Risk Management report. |
| `apps/ingest_crypto.py` | 86 | — | apps/ingest_crypto.py — dunne CLI rond de PIT crypto-ingestion (deliverable 15). |
| `apps/ingest_edgar.py` | 39 | — |  |
| `apps/ingest_eia.py` | 63 | — |  |
| `apps/ingest_fx.py` | 67 | — |  |
| `apps/ingest_xasset.py` | 50 | — |  |
| `apps/ledger_append.py` | 88 | — |  |
| `apps/live_paper_trader.py` | 658 | — | apps/live_paper_trader.py — Live shadow trader for 14-day MRM validation. |
| `apps/live_trader.py` | 398 | — | apps/live_trader.py — live engine entry point with full model wiring. |
| `apps/monitor_drift.py` | 76 | `tb-monitor-drift` |  |
| `apps/paper_monitor.py` | 305 | — | apps/paper_monitor.py — Live Streamlit dashboard for paper trading. |
| `apps/paper_multi_sleeve.py` | 166 | — | paper-trade runner for the FINAL regime-robust book. |
| `apps/paper_neutral_trader.py` | 201 | — | paper-trading runner for the market-neutral book. |
| `apps/paper_trade_runner.py` | 539 | — | apps/paper_trade_runner.py — Historical feature replay paper trade runner. |
| `apps/regenerate_baseline.py` | 89 | — |  |
| `apps/run_adaptive_wf.py` | 64 | — | CLI for the adaptive walk-forward book (R-6, stateless). |
| `apps/run_baseline.py` | 87 | — | Phase 3, deliverable 14 — draai de Level-1 baseline en de statistische gates. |
| `apps/run_eq_units.py` | 68 | — |  |
| `apps/run_gates.py` | 76 | — | Draai de research-gates lokaal, vóór een PR — Stage B-5. |
| `apps/run_stress.py` | 101 | — | Phase 4, deliverable 13 — draai de stressscenario's en de baseline-overlay. |

### Platformgereedschap in `scripts/`

Vijf ervan zijn POORTEN — gemarkeerd met **poort** hieronder: zij geven exit 1
en zij draaien alle vijf in `.github/workflows/inventory.yml`.

| Script | LOC | Waarvoor |
|---|---:|---|
| `scripts/audit_fallbacks.py` | 387 | **poort** — Phase 0 / Stap 4 - AST-gebaseerde scanner voor stille degradatie. |
| `scripts/build_data_register.py` | 201 | Genereer docs/DATA_REGISTER.md uit de PIT-store — Phase 1, stap 11. |
| `scripts/build_feature_store.py` | 99 | Phase 2, deliverable 9 - bouw de gecertificeerde L3 feature store. |
| `scripts/check_banned_methods.py` | 275 | **poort** — Phase 2 / Stage B-5 — AST-scanner op verboden validatiemethoden. |
| `scripts/check_file_size.py` | 114 | **poort** — Phase 9 / stap 13 - de LOC-ratchet met een cap per bestand. Vervangt het `loc-check`-doel in de `Makefile`, dat nooit is uitgevoerd omdat `make` in deze omgeving niet bestaat. Sluit DI-3. |
| `scripts/check_hardcoded_params.py` | 284 | **poort** — Phase 0 / stap 8 - scanner voor hardcoded parameters (exit criterium 6). |
| `scripts/clean.py` | 131 | Phase 9 / stap 14 - verwijdert tool-caches en runlogs (`.mypy_cache`, `.hypothesis`, `.pytest_cache`, `.ruff_cache`, `catboost_info`, `logs/`, `outputs/`). Vervangt het `clean`-doel in de `Makefile`. **Draai hem droog voordat je hem echt draait** — dit is het enige gereedschap hier dat bestanden verwijdert die NIET in versiebeheer staan en dus niet met `git show` terug te halen zijn. `data/pit_store/` en `artefacts/governance/` staan buiten zijn bereik en `tests/unit/test_clean.py` dwingt af dat hij ze zelfs niet VOORSTELT. |
| `scripts/monitor_retrain.py` | 110 | scripts/monitor_retrain.py — Live progress dashboard for the retrain run. |
| `scripts/paper_trade_report.py` | 74 | Parse live-engine state into a human-readable paper-trade report. |
| `scripts/phase0_baseline.py` | 245 | Phase 0 / Step 1 - repository baseline inventory (evidence artefact). |
| `scripts/phase1_data_gap.py` | 214 | Phase 1 / stap 1 - formele vastlegging van de datalacune (bewijsartefact). |
| `scripts/reachability_map.py` | 411 | **poort** — AST-bereikbaarheidskaart over `src/` — Phase 9, deliverable 1. `--strict` geeft exit 1 op een ongeregistreerde onbereikbare module. |
| `scripts/unwrap_broad_except.py` | 201 | Phase 0 helper - unwrap `except Exception` handlers that swallow errors. |

### Onderzoek

Achtenveertig bestanden in `research/`, met hun herkomst per bestand in
`research/README.md`. Zij staan buiten de DAG, buiten CI en leveren geen
productiegedrag.

### Waarom geen van deze entrypoints is verplaatst of verwijderd

Stap 12 van de fase-opdracht laat drie uitkomsten toe per ongenoemd
entrypoint: een verwijzing toevoegen, verplaatsen naar `research/`, of
verwijderen met registerregel. Voor alle twaalf ongenoemde apps is het de
eerste geworden, en dat is een besluit met een reden:

* **`live_paper_trader.py`** is de seed van **16 modules / 5.264 LOC** in
  `live/`. Verplaatsen naar `research/` verandert die zestien van klasse B
  naar klasse C en beantwoordt daarmee stilzwijgend **openstaand besluit 3**
  uit `docs/PROJECT_STATE.md` §5 — *wordt `live/` aangesloten of herbouwd?*
  Fence 5 van Phase 9 verbiedt dat expliciet.
* **`freeze_preregistration.py`** en **`freeze_phase6_preregistrations.py`**
  schrijven in `artefacts/governance/`, dat append-only is (fence 2). Het
  gereedschap dat een bevroren artefact heeft geschreven, moet vindbaar
  blijven naast het artefact.
* **`ingest_crypto.py`** en **`ingest_edgar.py`** zijn de ingestieweg van de
  PIT-store. Hun bronmodules staan onder een datacontract.
* De overige — `feature_selection.py`, `generate_mrm_report.py`,
  `paper_monitor.py`, `paper_multi_sleeve.py`, `paper_neutral_trader.py`,
  `run_baseline.py`, `run_stress.py` — bereiken elk klasse-B-modules. Ze
  verwijderen betekent die modules onbereikbaar maken, en dan verschuift het
  probleem in plaats van dat het verdwijnt.

Wat wél is veranderd: zij staan nu hier, dus zij zijn genoemd.

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
- worst case: 6 h cold rebuild (see nightly DAG timing in `dvc.yaml`).

> **Gecorrigeerd 2026-08-29 (Phase 7/8).** Hier stond *"regenerable from
> `dvc pull` against the S3/MinIO remote"*. **Die remote bestaat niet** —
> `.dvc/config` bevat uitsluitend `no_scm = True`, en `dvc pull` antwoordt
> `No remote provided and no default remote set`. De claim is nooit waar
> geweest; hij viel pas op toen de eerste CI-run erop strandde.
>
> De gecertificeerde PIT-store staat sinds AD-12 **in git** en komt mee met
> de checkout. De afgeleide artefacten (`artefacts/features`,
> `artefacts/models`, `artefacts/tracks`) zijn reproduceerbaar uit
> `dvc.yaml` op die store.
>
> Dit hele runbook is nog het legacy Wave-document en wordt in Stage D
> herschreven voor het systeem dat Phase 5 heeft gebouwd. Deze correctie
> staat er los van: een document dat een niet-bestaande remote als
> herstelpad opgeeft, is een herstelpad dat in een incident faalt.

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
