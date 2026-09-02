# Tradebot — Architecture Overview (v1.0)

> Audience: new engineers and on-call. For algorithmic detail, see the
> module-level docstrings and the binding audit document
> [`ARCHITECTUUR_AUDIT_2026-08-22.md`](ARCHITECTUUR_AUDIT_2026-08-22.md).
>
> **Geverifieerd tegen de codebase op 2026-09-01** (Phase 7/8, Stage E-3).
> De DAG in §2 is opnieuw uit `dvc.yaml` gelezen; de eerdere verificatie van
> 2026-08-22 dateerde van vóór Phase 5 en beschreef twee stages die sindsdien
> zijn verwijderd.
>
> Phase 0 corrigeert hier **D-5**: dit document verwees naar een
> `REFACTOR_BLUEPRINT_v3.md` in de projectroot dat nooit heeft bestaan. De
> verwijzing is vervangen door het auditdocument, dat vanaf nu het enige
> bindende architectuurdocument is.
>
> **D-8** is eveneens gesloten: de DAG-mappen `artefacts/features/`,
> `artefacts/models/` en `artefacts/tracks/` bestaan nu daadwerkelijk en dragen
> elk een `README.md` met hun artefact-contract.
>
> Waar dit document en het auditdocument elkaar tegenspreken, wint het
> auditdocument.

## 1. Three-layer model

```
┌───────────────────────────────────────────────────────────────┐
│ LAAG 3 — LIVE ENGINE       asyncio event-loop, OMS, recon     │
│   apps/live_trader.py · tradebot/live · tradebot/oms          │
├───────────────────────────────────────────────────────────────┤
│ LAAG 2 — DECISION LAYER    alpha, portfolio, risk, execution  │
│   tradebot/alpha · portfolio · risk · execution · tca         │
├───────────────────────────────────────────────────────────────┤
│ LAAG 1 — RESEARCH PIPELINE data → features → train → backtest │
│   tradebot/data · bars · features · labeling · cv · tune ·    │
│   train · backtest · monitoring · registry · reporting        │
└───────────────────────────────────────────────────────────────┘
```

Layer 1 is self-contained. Layer 2 imports from Layer 1 (read-only).
Layer 3 imports from Layers 1+2 and never writes to their artefacts.

## 2. DAG stages

*Gemeten uit `dvc.yaml` op 2026-09-01. Dit is de VOLLEDIGE DAG; er zijn geen
andere stages.*

| Stage | Commando | Output |
|-------|----------|--------|
| `data_sync` | `apps.data_sync` | `market_data_parquet/<symbool>/` |
| `build_features` | `apps.build_features` | `artefacts/bars/`, `artefacts/features/`, `artefacts/events/` |
| `tune_hparams` | `apps.tune_hparams` | `artefacts/hparams/` |
| `train_cpcv` | `apps.train_cpcv` | `artefacts/models/`, `artefacts/calibrators/`, `artefacts/oos_probs/` |
| `phase5_revaluation` | `apps/run_phase5_baseline.py` | `artefacts/baseline/phase5_revaluation.json` |
| `phase6_adequacy` | `apps/run_data_adequacy.py` | `artefacts/governance/phase6_data_adequacy.json` |
| `phase6_econometrics` | `apps/run_econometric_diagnostics.py` | `artefacts/governance/phase6_econometrics.json` |
| `phase6_h1_competition` | `apps/run_vol_competition.py` | `…/phase6_h1_competition.json`, `reports/GARCH_VS_EWMA_COMPETITION.md` |
| `phase6_h2_regime_benchmark` | `apps/run_regime_benchmark.py` | `…/phase6_h2_regime_benchmark.json`, `reports/M0_VS_HMM_BENCHMARK.md` |
| `phase6_h3_meta_labeling` | `apps/run_meta_labeling.py` | `…/phase6_h3_meta_labeling.json`, `reports/META_LABELING_EVALUATION.md` |

**Wat hier NIET meer staat, en waarom.** Tot Phase 7/8 Stage E-3 droeg deze
tabel twee stages die Phase 5 heeft VERWIJDERD — `backtest_portfolio` en
`make_tearsheet`, die `bidirectional_backtest` en `PortfolioBacktester` dreven.
Audit §16.1 en §24 schrijven consolidatie tot één event-driven engine voor; het
pariteitsbewijs staat in `tests/integration/test_engine_parity.py` en de
vergelijking in `reports/phase5_engine_diff.md`. `dvc.yaml` documenteert de
verwijdering op de plek waar de stages stonden. De opvolger is
`src/tradebot/backtest/engine.py` (`EventDrivenEngine`), met `phase5_revaluation`
als DVC-stage.

Daarmee vervielen ook de outputs `artefacts/portfolio/`, `reports/tearsheets/` en
`artefacts/alpha_signals/`, die dit document nog als bestaand aanwees.

**Apps buiten de DAG.** `apps/featurestore_sync.py`, `apps/alpha_combine.py`,
`apps/live_trader.py` en `apps/paper_trade_runner.py` bestaan wel maar zijn geen
DVC-stage: zij worden met de hand of door de live-loop gestart en produceren
geen gehashte artefacten in de pijplijn.

## 3. Live event flow (per bar-close)

```
Bar(t) received
  → FeatureUpdater.update(bar_t)            ≤  50 ms
  → SignalRunner.predict(features_t)        ≤ 100 ms
  → PortfolioController.optimize(…)         ≤ 200 ms
  → ExecutionController.size_orders(…)
  → OMS.place_orders(…)
  → AuditLog.record(…)
                          target end-to-end < 500 ms
```

Failure conditions monitored by `CircuitBreaker` (Wave 8, R-7):

- max drawdown > 8 % of equity peak
- daily loss > 3 % of NAV
- feed timeout > 30 s
- model `feature_hash` mismatch
- VaR > 1.5 × daily limit
- position older than 48 h with no signal refresh

On halt: cancel open orders, snapshot state, emit Slack alert, refuse
auto-resume. Restart requires human acknowledgement and a new `model_version`
boot.

## 4. Governance flow

1. New model → registered as **challenger** in `ModelCatalog`.
2. `ShadowTrader` runs the challenger alongside the **champion** for ≥ 14 days
   without placing orders.
3. `ChampionChallenger.evaluate(champion_pnl, challenger_pnl)`:
   * Diebold-Mariano test with Harvey-Leybourne-Newbold small-sample
     correction.
   * Promote only when `p < 0.05` **and** Sharpe(challenger) > Sharpe(champion).
4. On promotion: `generate_mrm_report(...)` is produced and stored alongside
   the model JSON in `registry/`.
5. `live.mode=live` deployment refuses to start unless the active model has a
   signed-off MRM report (`approved_by` non-empty).

## 5. R-1 … R-8 — design rules in one line each

| #   | Rule                                                                  |
|-----|-----------------------------------------------------------------------|
| R-1 | Causality is absolute. Feature for bar t uses only data ≤ t-1.        |
| R-2 | Zero root `*.py` files. All code lives under `src/tradebot/`.         |
| R-3 | Stage boundary = Pandera contract. `validate_or_die` is the handshake.|
| R-4 | 800 LOC per file (4-file whitelist with `# LOC-EXCEPTION`).           |
| R-5 | Deterministic. Same `cfg + seed` ⇒ bit-identical output.              |
| R-6 | Stateless CLI apps (≤ 80 LOC). All logic in `src/tradebot/`.          |
| R-7 | Dead-man's switch: 6 conditions, halt on any.                         |
| R-8 | Audit trail: every order has 14 mandatory fields.                     |
