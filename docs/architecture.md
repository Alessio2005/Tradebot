# Tradebot — Architecture Overview (v1.0)

> Audience: new engineers and on-call. For algorithmic detail, see the
> module-level docstrings and the `REFACTOR_BLUEPRINT_v3.md` root document.

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

| Stage | App entry-point          | Output                              |
|-------|---------------------------|-------------------------------------|
| 0     | `data_sync`               | `market_data_parquet/`              |
| 1     | `build_features`          | `artefacts/features/, events/`      |
| 1b    | `featurestore_sync`       | `artefacts/feature_store/`          |
| 2     | `tune_hparams`            | `artefacts/hparams/`                |
| 3     | `train_cpcv`              | `artefacts/models/, calibrators/`   |
| 4     | `backtest_portfolio`      | `artefacts/portfolio/, tracks/`     |
| 5     | `make_tearsheet`          | `reports/tearsheets/`               |
| 6     | `alpha_combine`           | `artefacts/alpha_signals/`          |
| live  | `live_trader`             | live or paper-trade event loop      |

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
