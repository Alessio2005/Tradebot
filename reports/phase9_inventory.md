# Phase 9 - beslistabel over de volledige oppervlakte (stap 7)

> **Gemeten 2026-09-04 op commit `b0ac67d`.** Bereikbaarheid uit
> `scripts/reachability_map.py`, dekking uit `pytest-cov` 7.1.0. Dit document
> is een LEESSTAP: er wordt in deze stap niets verwijderd of verplaatst.

Elke module, app en script krijgt precies een rij en precies een verdict.
Nul rijen zonder verdict.

## De beslistabel zoals toegepast

| Klasse | Dekking | Verdict |
|---|---|---|
| A | >= 70 % | BEHOUDEN |
| A | < 70 % | TESTEN, of ratchet met datum en eigenaar |
| B | >= 70 % | BEHOUDEN |
| B | < 70 % | TESTEN - zie de motivering hieronder waarom niet ARCHIVEREN |
| C | n.v.t. | VERPLAATSEN naar `research/` - tenzij fence 4 |
| D | n.v.t. | BEHOUDEN, met CONTRACT of AMBITIE in het register |
| E | n.v.t. | VERWIJDEREN - tenzij een bestaansassertie |

### Waarom geen enkele B-module ARCHIVEREN krijgt

De opdracht laat bij `B < 70 %` de keuze tussen TESTEN en ARCHIVEREN en eist
een expliciete keuze. Die keuze is **TESTEN voor alle 35**, om een reden die
per module dezelfde is. Gegroepeerd naar de app die ze bereikt:

| Seed-app | Genoemd? | Modules | LOC |
|---|:---:|---:|---:|
| `apps/live_paper_trader.py` | **nee** | 16 | 5264 |
| `apps/build_equity_universe.py` | ja | 5 | 712 |
| `apps/paper_multi_sleeve.py` | **nee** | 3 | 411 |
| `apps/feature_selection.py` | **nee** | 3 | 293 |
| `apps/ingest_edgar.py` | **nee** | 2 | 203 |
| `apps/ingest_eia.py` | ja | 1 | 177 |
| `apps/ingest_xasset.py` | ja | 1 | 160 |
| `apps/ingest_fx.py` | ja | 1 | 129 |
| `apps/run_adaptive_wf.py` | ja | 1 | 122 |
| `apps/fetch_factors.py` | ja | 1 | 89 |
| `apps/featurestore_sync.py` | ja | 1 | 73 |

Het zwaartepunt is `live/` (16 modules / 5.264 LOC via
`apps/live_paper_trader.py`). Archiveren daarvan zou **openstaand besluit 3**
van `PROJECT_STATE.md` par. 5 - *Wordt `live/` aangesloten of herbouwd?* -
stilzwijgend beantwoorden met *herbouwd*, en fence 5 verbiedt precies dat.
De overige groepen zijn ingestion-apps die wel worden genoemd, plus
`selection/` (293 LOC, 0,0 % dekking), dat te klein is om archiveren te
rechtvaardigen en in stap 11 goedkoop te dekken valt.

### Waarom geen enkele D-module ARCHIVEREN krijgt

Klasse D is per definitie *uitsluitend vanuit `tests/` bereikbaar*. Daaruit
volgt iets dat de opdracht niet expliciet maakt: **een D-module verplaatsen
betekent zijn test breken.** Naar `research/archive/` verhuizen laat de import
in `tests/` falen, en dat verandert een `passed` in een `error` - een
regressie tegen exit-criterium 1, dat meer passes toestaat maar geen minder.

Archiveren van een D-module vraagt dus om ook de test te verwijderen, en
daarmee bewijsmateriaal. Alle twintig blijven staan; het register noteert per
module of zij een CONTRACT dragen of een AMBITIE zijn. Achttien dragen een
aanwijsbare afnemer; twee zijn ambitie die blijft staan omdat verplaatsen de
vingerafdruk zou wijzigen.

## Samenvatting

| Verdict | Modules | LOC |
|---|---:|---:|
| BEHOUDEN | 174 | 38393 |
| TESTEN | 110 | 32041 |
| VERWIJDEREN | 9 | 487 |
| VERPLAATSEN naar `research/` | 6 | 566 |
| **totaal** | **299** | **71487** |

## Klasse E - onbereikbaar - 11 modules / 1505 LOC

| Module | LOC | Dekking | Verdict | Afnemer / reden |
|---|---:|---:|---|---|
| `src/tradebot/portfolio/legacy_sizing.py` | 824 | 0.0 % | **BEHOUDEN** | bestaansassertie in `tests/unit/test_risk_alpha_decoupling.py` (Phase 4 exit-criterium 3) + DI-10 + ratchetpost in `check_hardcoded_params.py` |
| `src/tradebot/portfolio/covariance.py` | 194 | 0.0 % | **BEHOUDEN** | bestaansassertie in `tests/unit/test_risk_alpha_decoupling.py` (Phase 4 exit-criterium 3) + DI-10 |
| `src/tradebot/alpha/decay_tracker.py` | 140 | 0.0 % | **VERWIJDEREN** | Wave 18 IC-decay-tracker; nooit aangesloten, niet in `alpha/__init__.py` |
| `src/tradebot/features/macro.py` | 110 | 0.0 % | **VERWIJDEREN** | macro-featuretransformaties; er is geen macro-featurepad in de keten |
| `src/tradebot/data/sources/cboe.py` | 61 | 0.0 % | **VERWIJDEREN** | VIX-TS-bron; het pakketcontract eist een lookahead-test in `tests/lookahead/` (G6) en die bestaat niet |
| `src/tradebot/logging_config.py` | 61 | 0.0 % | **VERWIJDEREN** | de docstring draagt op `setup_logging()` aan te roepen bij elke app-entry; **nul aanroepen** in de hele repository |
| `src/tradebot/alpha/eq_strev_resid.py` | 55 | 0.0 % | **VERWIJDEREN** | Wave 24; `WAVE_LOG.md` en `EXPANSION_RESEARCH` boeken hem als ARCHIVED -> F16 |
| `src/tradebot/types.py` | 25 | 0.0 % | **VERWIJDEREN** | domein-NewTypes en Protocols; **dubbele waarheid** naast `schemas/`, door niets geimporteerd |
| `src/tradebot/bars/tick.py` | 16 | 0.0 % | **VERWIJDEREN** | `raise NotImplementedError`; niet in `bars/__init__.py` |
| `src/tradebot/bars/volume.py` | 16 | 0.0 % | **VERWIJDEREN** | `raise NotImplementedError`; niet in `bars/__init__.py` |
| `src/tradebot/_version.py` | 3 | 0.0 % | **VERWIJDEREN** | de docstring claimt *single source of truth for the package version* - **onwaar**: `tradebot/__init__.py` leest de versie via `importlib.metadata` uit `pyproject.toml` |

## Klasse D - test-only - 20 modules / 3036 LOC

| Module | LOC | Dekking | Verdict | Afnemer / reden |
|---|---:|---:|---|---|
| `src/tradebot/registry/lifecycle.py` | 310 | 100.0 % | **BEHOUDEN** | CONTRACT - DI-15 - `SymbolLifecycle` is gebouwd en wacht op een databron met delisting-historie; `artefacts/governance/symbol_lifecycle.json` |
| `src/tradebot/tca/post_trade.py` | 304 | 95.7 % | **BEHOUDEN** | CONTRACT - `docs/tca_methodology.md`; DI-5 (gesloten) is op deze modules gesloten |
| `src/tradebot/volatility/har_rv.py` | 270 | 81.7 % | **BEHOUDEN** | CONTRACT - DI-18 - HAR-RV is geblokkeerd op een intraday-bron, niet beslist |
| `src/tradebot/monitoring/execution_drift.py` | 228 | 97.1 % | **BEHOUDEN** | CONTRACT - drempels bevroren in `artefacts/governance/monitoring_config_hash.json` (`max_adverse_price_drift_bps`, `max_fill_latency_seconds`, `min_fill_ratio`) |
| `src/tradebot/registry/experiment.py` | 223 | 77.0 % | **BEHOUDEN** | CONTRACT - governance - draagt de experimentregistratie in `artefacts/governance/` |
| `src/tradebot/monitoring/vol_forecast_monitor.py` | 212 | 98.5 % | **BEHOUDEN** | CONTRACT - drempels bevroren in `artefacts/governance/monitoring_config_hash.json` (`vol_forecast_min_obs`, `vol_mz_alpha`, `vol_qlike_degradation_ratio`) |
| `src/tradebot/bars/dollar.py` | 173 | 45.5 % | **BEHOUDEN** | AMBITIE - geen DI, geen AD, geen preregistratie. **Behouden**: de numba-kernel `numba_dollar_bars` staat in `bars/_kernels.py` (klasse A), en verplaatsen breekt `tests/unit/test_audit_fixes.py` |
| `src/tradebot/data/orderbook.py` | 150 | 92.5 % | **BEHOUDEN** | AMBITIE - geen DI, geen AD, geen preregistratie. **Behouden**: verplaatsen breekt `tests/unit/test_wave6.py` |
| `src/tradebot/monitoring/sharpe_monitor.py` | 143 | 96.2 % | **BEHOUDEN** | CONTRACT - DI-7 (gesloten) - de HALT-taak; `tests/unit/test_external_monitors_can_actually_halt.py` rijdt een echte degradatie door een echte breaker |
| `src/tradebot/tca/implementation_shortfall.py` | 136 | 79.4 % | **BEHOUDEN** | CONTRACT - `docs/tca_methodology.md` - de IS-decompositie |
| `src/tradebot/data/funding.py` | 129 | 21.4 % | **BEHOUDEN** | CONTRACT - de fundingreeks is gecertificeerd in `artefacts/governance/data_hashes.json` |
| `src/tradebot/alpha/fx_tsmom.py` | 119 | 85.4 % | **BEHOUDEN** | CONTRACT - `docs/EXPANSION_RESEARCH_2026-08-10.md`: *this file IS the D1 unit*; D1 is de openstaande post uit commit `f7702dd` |
| `src/tradebot/features/funding_carry.py` | 106 | 95.7 % | **BEHOUDEN** | CONTRACT - `docs/SHORT_ALPHA_RESEARCH_2026-05-30.md` (causaliteitseis op de lag) |
| `src/tradebot/features/open_interest.py` | 106 | 96.8 % | **BEHOUDEN** | CONTRACT - de OI-reeks is gecertificeerd in `artefacts/governance/data_hashes.json` |
| `src/tradebot/data/panel.py` | 105 | 100.0 % | **BEHOUDEN** | CONTRACT - `asof_join`-contract; lookahead-test `tests/lookahead/test_asof_join_crypto.py` |
| `src/tradebot/tca/arrival_price.py` | 92 | 74.2 % | **BEHOUDEN** | CONTRACT - `docs/tca_methodology.md` - `arrival_price` is de referentieprijs van de decompositie |
| `src/tradebot/tca/pre_trade.py` | 87 | 100.0 % | **BEHOUDEN** | CONTRACT - `docs/tca_methodology.md`; DI-5 corrigeerde juist een onjuiste claim OVER deze module |
| `src/tradebot/tca/report.py` | 77 | 80.0 % | **BEHOUDEN** | CONTRACT - `docs/tca_methodology.md` + `reports/TCA_CALIBRATION_REPORT.md` |
| `src/tradebot/schemas/orders.py` | 39 | 100.0 % | **BEHOUDEN** | CONTRACT - ordercontract; governance-artefacten verwijzen ernaar |
| `src/tradebot/tca/__init__.py` | 27 | 100.0 % | **BEHOUDEN** | CONTRACT - pakketmarkering van de TCA-laag hierboven |

## Klasse C - research-only - 8 modules / 1108 LOC

| Module | LOC | Dekking | Verdict | Afnemer / reden |
|---|---:|---:|---|---|
| `src/tradebot/alpha/cm_carry.py` | 383 | 75.5 % | **BEHOUDEN** | fence 4 - draagt de pre-geregistreerde killgates KG-B1/B2/B3 |
| `src/tradebot/alpha/cm_tsmom.py` | 159 | 96.8 % | **BEHOUDEN** | fence 4 - draagt de pre-geregistreerde killgate KG-B2 |
| `src/tradebot/backtest/dd_shape.py` | 148 | 92.3 % | **VERPLAATSEN naar `research/`** | sluit DI-8 |
| `src/tradebot/data/fx_universe.py` | 137 | 98.2 % | **VERPLAATSEN naar `research/`** | sluit DI-8 |
| `src/tradebot/alpha/eq_pead.py` | 95 | 82.9 % | **VERPLAATSEN naar `research/`** | sluit DI-8 |
| `src/tradebot/alpha/eq_quality.py` | 81 | 87.5 % | **VERPLAATSEN naar `research/`** | sluit DI-8 |
| `src/tradebot/alpha/eq_overnight.py` | 58 | 93.3 % | **VERPLAATSEN naar `research/`** | sluit DI-8 |
| `src/tradebot/alpha/fx_carry.py` | 47 | 100.0 % | **VERPLAATSEN naar `research/`** | sluit DI-8 |

## Klasse B - operationeel - 67 modules / 12763 LOC

| Module | LOC | Dekking | Verdict | Bereikt via |
|---|---:|---:|---|---|
| `src/tradebot/live/engine.py` | 913 | 34.0 % | TESTEN | apps/live_paper_trader.py |
| `src/tradebot/live/feed.py` | 747 | 22.7 % | TESTEN | apps/live_paper_trader.py |
| `src/tradebot/live/model_signal.py` | 591 | 0.0 % | TESTEN | apps/live_paper_trader.py |
| `src/tradebot/live/feature_updater.py` | 572 | 38.2 % | TESTEN | apps/live_paper_trader.py |
| `src/tradebot/live/circuit_breaker.py` | 466 | 92.5 % | BEHOUDEN | apps/live_paper_trader.py |
| `src/tradebot/execution/impact_calibration.py` | 454 | 88.6 % | BEHOUDEN | apps/calibrate_impact.py |
| `src/tradebot/alpha/adaptive_wf.py` | 362 | 81.9 % | BEHOUDEN | apps/run_adaptive_wf.py |
| `src/tradebot/live/execution_controller.py` | 341 | 87.1 % | BEHOUDEN | apps/live_paper_trader.py |
| `src/tradebot/live/portfolio_controller.py` | 331 | 36.7 % | TESTEN | apps/live_paper_trader.py |
| `src/tradebot/live/signal_runner.py` | 321 | 23.2 % | TESTEN | apps/live_paper_trader.py |
| `src/tradebot/oms/router.py` | 315 | 27.9 % | TESTEN | apps/live_paper_trader.py |
| `src/tradebot/monitoring/drift.py` | 314 | 32.4 % | TESTEN | apps/live_paper_trader.py |
| `src/tradebot/oms/paper_oms.py` | 272 | 69.1 % | TESTEN | apps/live_paper_trader.py |
| `src/tradebot/alpha/xs_unit.py` | 255 | 89.5 % | BEHOUDEN | apps/run_eq_units.py |
| `src/tradebot/compliance/mrm_report.py` | 245 | 79.2 % | BEHOUDEN | apps/generate_mrm_report.py |
| `src/tradebot/compliance/champion_challenger.py` | 225 | 94.7 % | BEHOUDEN | apps/generate_mrm_report.py |
| `src/tradebot/validation/gate_runner.py` | 221 | 93.3 % | BEHOUDEN | apps/run_gates.py |
| `src/tradebot/monitoring/live_drift_monitor.py` | 218 | 0.0 % | TESTEN | apps/live_paper_trader.py |
| `src/tradebot/compliance/shadow_trader.py` | 201 | 95.7 % | BEHOUDEN | apps/generate_mrm_report.py |
| `src/tradebot/validation/walk_forward.py` | 201 | 90.9 % | BEHOUDEN | apps/feature_selection.py |
| `src/tradebot/compliance/position_report.py` | 183 | 94.7 % | BEHOUDEN | apps/generate_mrm_report.py |
| `src/tradebot/oms/reconciler.py` | 183 | 91.1 % | BEHOUDEN | apps/live_paper_trader.py |
| `src/tradebot/monitoring/alerts.py` | 182 | 88.7 % | BEHOUDEN | apps/live_paper_trader.py |
| `src/tradebot/oms/position_tracker.py` | 182 | 80.3 % | BEHOUDEN | apps/live_paper_trader.py |
| `src/tradebot/data/sources/eia.py` | 177 | 0.0 % | TESTEN | apps/ingest_eia.py |
| `src/tradebot/live/judge_gate.py` | 174 | 22.4 % | TESTEN | apps/live_paper_trader.py |
| `src/tradebot/data/perp_feed.py` | 173 | 0.0 % | TESTEN | apps/paper_multi_sleeve.py |
| `src/tradebot/oms/audit_log.py` | 169 | 89.0 % | BEHOUDEN | apps/live_paper_trader.py |
| `src/tradebot/data/xasset_proxy.py` | 160 | 50.0 % | TESTEN | apps/ingest_xasset.py |
| `src/tradebot/featurestore/store.py` | 158 | 90.5 % | BEHOUDEN | apps/featurestore_sync.py |
| `src/tradebot/data/equity_universe.py` | 157 | 0.0 % | TESTEN | apps/build_equity_universe.py |
| `src/tradebot/live/cusum_filter.py` | 157 | 22.0 % | TESTEN | apps/live_paper_trader.py |
| `src/tradebot/data/sources/stooq.py` | 155 | 0.0 % | TESTEN | apps/build_equity_universe.py |
| `src/tradebot/monitoring/metrics.py` | 151 | 79.5 % | BEHOUDEN | apps/live_paper_trader.py |
| `src/tradebot/data/sources/wiki_constituents.py` | 149 | 35.6 % | TESTEN | apps/build_equity_universe.py |
| `src/tradebot/data/sources/base.py` | 143 | 64.6 % | TESTEN | apps/build_equity_universe.py |
| `src/tradebot/selection/mda.py` | 142 | 0.0 % | TESTEN | apps/feature_selection.py |
| `src/tradebot/compliance/circuit_log.py` | 138 | 90.9 % | BEHOUDEN | apps/generate_mrm_report.py |
| `src/tradebot/selection/sfi.py` | 137 | 0.0 % | TESTEN | apps/feature_selection.py |
| `src/tradebot/oms/order.py` | 135 | 98.0 % | BEHOUDEN | apps/live_paper_trader.py |
| `src/tradebot/data/sources/fred.py` | 129 | 22.9 % | TESTEN | apps/ingest_fx.py |
| `src/tradebot/live/state.py` | 128 | 88.7 % | BEHOUDEN | apps/live_paper_trader.py |
| `src/tradebot/backtest/pbo.py` | 122 | 17.6 % | TESTEN | apps/run_adaptive_wf.py |
| `src/tradebot/alpha/multi_sleeve_book.py` | 120 | 29.1 % | TESTEN | apps/paper_multi_sleeve.py |
| `src/tradebot/data/sources/edgar.py` | 120 | 0.0 % | TESTEN | apps/ingest_edgar.py |
| `src/tradebot/alpha/neutral_book.py` | 118 | 33.8 % | TESTEN | apps/paper_multi_sleeve.py |
| `src/tradebot/monitoring/cross_account.py` | 113 | 93.5 % | BEHOUDEN | apps/live_paper_trader.py |
| `src/tradebot/registry/risk_registry.py` | 113 | 100.0 % | BEHOUDEN | apps/run_stress.py |
| `src/tradebot/data/sources/yfinance_backup.py` | 108 | 0.0 % | TESTEN | apps/build_equity_universe.py |
| `src/tradebot/monitoring/prob_calibration.py` | 108 | 34.6 % | TESTEN | apps/live_paper_trader.py |
| `src/tradebot/monitoring/feature_health.py` | 101 | 51.0 % | TESTEN | apps/live_paper_trader.py |
| `src/tradebot/data/sources/kenfrench.py` | 89 | 0.0 % | TESTEN | apps/fetch_factors.py |
| `src/tradebot/data/edgar_universe.py` | 83 | 0.0 % | TESTEN | apps/ingest_edgar.py |
| `src/tradebot/featurestore/backfill.py` | 73 | 40.9 % | TESTEN | apps/featurestore_sync.py |
| `src/tradebot/live/exchange_status.py` | 72 | 32.1 % | TESTEN | apps/live_paper_trader.py |
| `src/tradebot/alpha/eq_lowvol.py` | 71 | 100.0 % | BEHOUDEN | apps/run_eq_units.py |
| `src/tradebot/live/sigterm.py` | 58 | 43.8 % | TESTEN | apps/live_paper_trader.py |
| `src/tradebot/alpha/eq_xsmom.py` | 55 | 100.0 % | BEHOUDEN | apps/run_eq_units.py |
| `src/tradebot/alpha/eq_strev.py` | 46 | 100.0 % | BEHOUDEN | apps/run_eq_units.py |
| `src/tradebot/monitoring/__init__.py` | 27 | 100.0 % | BEHOUDEN | apps/live_paper_trader.py |
| `src/tradebot/live/__init__.py` | 25 | 100.0 % | BEHOUDEN | apps/live_paper_trader.py |
| `src/tradebot/compliance/__init__.py` | 23 | 100.0 % | BEHOUDEN | apps/generate_mrm_report.py |
| `src/tradebot/data/sources/__init__.py` | 23 | 100.0 % | BEHOUDEN | apps/build_equity_universe.py |
| `src/tradebot/featurestore/schema.py` | 23 | 100.0 % | BEHOUDEN | apps/featurestore_sync.py |
| `src/tradebot/oms/__init__.py` | 17 | 100.0 % | BEHOUDEN | apps/live_paper_trader.py |
| `src/tradebot/featurestore/__init__.py` | 14 | 100.0 % | BEHOUDEN | apps/featurestore_sync.py |
| `src/tradebot/selection/__init__.py` | 14 | 0.0 % | TESTEN | apps/feature_selection.py |

## Klasse A - authoritative - 193 modules / 53075 LOC

| Module | LOC | Dekking | Verdict | Bereikt via |
|---|---:|---:|---|---|
| `src/tradebot/labeling/meta.py` | 1181 | 18.0 % | TESTEN | apps/build_features.py |
| `src/tradebot/train/ensemble.py` | 1117 | 8.3 % | TESTEN | apps/build_features.py |
| `src/tradebot/schemas/config.py` | 1094 | 96.5 % | BEHOUDEN | apps/run_data_adequacy.py |
| `src/tradebot/features/regime.py` | 1056 | 11.9 % | TESTEN | apps/build_features.py |
| `src/tradebot/backtest/evaluation.py` | 1054 | 27.2 % | TESTEN | apps/run_data_adequacy.py |
| `src/tradebot/tune/objective.py` | 955 | 10.4 % | TESTEN | apps/tune_hparams.py |
| `src/tradebot/validation/data_adequacy.py` | 804 | 97.6 % | BEHOUDEN | apps/run_data_adequacy.py |
| `src/tradebot/features/base.py` | 785 | 96.6 % | BEHOUDEN | apps/build_features.py |
| `src/tradebot/features/ta.py` | 775 | 9.1 % | TESTEN | apps/build_features.py |
| `src/tradebot/validation/vol_competition.py` | 759 | 100.0 % | BEHOUDEN | apps/run_vol_competition.py |
| `src/tradebot/reporting/phase6_vol_competition.py` | 754 | 0.0 % | TESTEN | apps/run_vol_competition.py |
| `src/tradebot/regime/markov.py` | 729 | 98.2 % | BEHOUDEN | apps/run_data_adequacy.py |
| `src/tradebot/volatility/garch.py` | 653 | 93.3 % | BEHOUDEN | apps/run_vol_competition.py |
| `src/tradebot/features/_ta_kernels.py` | 635 | 14.9 % | TESTEN | apps/build_features.py |
| `src/tradebot/reporting/phase6_meta_labeling.py` | 633 | 97.1 % | BEHOUDEN | apps/run_meta_labeling.py |
| `src/tradebot/execution/market_impact.py` | 621 | 17.4 % | TESTEN | apps/build_features.py |
| `src/tradebot/train/catboost.py` | 600 | 11.9 % | TESTEN | apps/build_features.py |
| `src/tradebot/validation/vol_campaign.py` | 580 | 44.7 % | TESTEN | apps/run_vol_competition.py |
| `src/tradebot/reporting/phase6_regime_benchmark.py` | 578 | 0.0 % | TESTEN | apps/run_regime_benchmark.py |
| `src/tradebot/features/orthogonalize.py` | 555 | 75.1 % | BEHOUDEN | apps/build_features.py |
| `src/tradebot/validation/vol_metrics.py` | 552 | 96.7 % | BEHOUDEN | apps/run_vol_competition.py |
| `src/tradebot/risk/stress_test.py` | 543 | 90.7 % | BEHOUDEN | apps/run_data_adequacy.py |
| `src/tradebot/backtest/accounting.py` | 528 | 95.3 % | BEHOUDEN | apps/run_data_adequacy.py |
| `src/tradebot/registry/preregistration.py` | 528 | 96.0 % | BEHOUDEN | apps/run_data_adequacy.py |
| `src/tradebot/risk/engine.py` | 515 | 97.6 % | BEHOUDEN | apps/run_data_adequacy.py |
| `src/tradebot/validation/meta_label_campaign.py` | 504 | 0.0 % | TESTEN | apps/run_meta_labeling.py |
| `src/tradebot/data/crypto.py` | 501 | 15.0 % | TESTEN | apps/build_features.py |
| `src/tradebot/train/_scalers.py` | 498 | 12.0 % | TESTEN | apps/build_features.py |
| `src/tradebot/risk/limits.py` | 497 | 98.7 % | BEHOUDEN | apps/run_data_adequacy.py |
| `src/tradebot/validation/regime_benchmark.py` | 489 | 69.7 % | TESTEN | apps/run_regime_benchmark.py |
| `src/tradebot/backtest/baseline_report.py` | 488 | 0.0 % | TESTEN | apps/run_data_adequacy.py |
| `src/tradebot/regime/conditioning.py` | 483 | 97.9 % | BEHOUDEN | apps/run_meta_labeling.py |
| `src/tradebot/execution/order_router.py` | 482 | 99.3 % | BEHOUDEN | apps/run_data_adequacy.py |
| `src/tradebot/train/meta_label.py` | 476 | 97.7 % | BEHOUDEN | apps/run_meta_labeling.py |
| `src/tradebot/features/microstructure.py` | 474 | 66.9 % | TESTEN | apps/run_data_adequacy.py |
| `src/tradebot/backtest/engine.py` | 455 | 95.4 % | BEHOUDEN | apps/run_data_adequacy.py |
| `src/tradebot/risk/kill_switches.py` | 454 | 100.0 % | BEHOUDEN | apps/run_data_adequacy.py |
| `src/tradebot/labeling/trend_scanning.py` | 440 | 15.8 % | TESTEN | apps/build_features.py |
| `src/tradebot/regime/student_t.py` | 421 | 95.1 % | BEHOUDEN | apps/run_data_adequacy.py |
| `src/tradebot/data/tradfi_macro.py` | 411 | 12.0 % | TESTEN | apps/build_features.py |
| `src/tradebot/labeling/cusum.py` | 408 | 28.3 % | TESTEN | apps/build_features.py |
| `src/tradebot/execution/simulator.py` | 406 | 34.0 % | TESTEN | apps/build_features.py |
| `src/tradebot/alpha/momentum.py` | 404 | 68.1 % | TESTEN | apps/run_data_adequacy.py |
| `src/tradebot/train/quant_arch.py` | 393 | 66.5 % | TESTEN | apps/build_features.py |
| `src/tradebot/features/pipeline.py` | 388 | 70.9 % | BEHOUDEN | apps/build_features.py |
| `src/tradebot/volatility/realized.py` | 387 | 94.9 % | BEHOUDEN | apps/run_vol_competition.py |
| `src/tradebot/features/cfi.py` | 371 | 69.6 % | TESTEN | apps/build_features.py |
| `src/tradebot/portfolio/hrp.py` | 371 | 99.0 % | BEHOUDEN | apps/run_data_adequacy.py |
| `src/tradebot/features/registry.py` | 369 | 99.0 % | BEHOUDEN | apps/run_data_adequacy.py |
| `src/tradebot/validation/econometrics.py` | 368 | 96.8 % | BEHOUDEN | apps/run_econometric_diagnostics.py |
| `src/tradebot/alpha/base.py` | 366 | 93.6 % | BEHOUDEN | apps/run_data_adequacy.py |
| `src/tradebot/validation/gates.py` | 365 | 86.2 % | BEHOUDEN | apps/run_data_adequacy.py |
| `src/tradebot/labeling/triple_barrier.py` | 360 | 40.1 % | TESTEN | apps/build_features.py |
| `src/tradebot/features/volatility.py` | 356 | 100.0 % | BEHOUDEN | apps/run_data_adequacy.py |
| `src/tradebot/risk/var.py` | 350 | 64.6 % | TESTEN | apps/run_data_adequacy.py |
| `src/tradebot/features/fracdiff.py` | 335 | 86.0 % | BEHOUDEN | apps/run_econometric_diagnostics.py |
| `src/tradebot/validation/sharpe_difference.py` | 335 | 99.0 % | BEHOUDEN | apps/run_regime_benchmark.py |
| `src/tradebot/train/calibration.py` | 333 | 91.7 % | BEHOUDEN | apps/build_features.py |
| `src/tradebot/data/crypto_macro.py` | 332 | 14.7 % | TESTEN | apps/build_features.py |
| `src/tradebot/bars/_kernels.py` | 330 | 3.9 % | TESTEN | apps/build_features.py |
| `src/tradebot/risk/contract.py` | 327 | 100.0 % | BEHOUDEN | apps/run_data_adequacy.py |
| `src/tradebot/cv/cpcv.py` | 326 | 90.2 % | BEHOUDEN | apps/build_features.py |
| `src/tradebot/execution/spread.py` | 314 | 17.9 % | TESTEN | apps/tune_hparams.py |
| `src/tradebot/execution/context.py` | 307 | 97.8 % | BEHOUDEN | apps/run_data_adequacy.py |
| `src/tradebot/backtest/phase5_baseline.py` | 304 | 48.0 % | TESTEN | apps/run_meta_labeling.py |
| `src/tradebot/data/pit_store.py` | 303 | 95.5 % | BEHOUDEN | apps/build_features.py |
| `src/tradebot/labeling/phase6_barriers.py` | 303 | 89.7 % | BEHOUDEN | apps/run_data_adequacy.py |
| `src/tradebot/backtest/baseline_runner.py` | 298 | 52.3 % | TESTEN | apps/run_data_adequacy.py |
| `src/tradebot/alpha/csm_volume_clock.py` | 296 | 17.6 % | TESTEN | apps/run_data_adequacy.py |
| `src/tradebot/risk/stress_report.py` | 294 | 0.0 % | TESTEN | apps/run_data_adequacy.py |
| `src/tradebot/portfolio/constraints.py` | 291 | 91.9 % | BEHOUDEN | apps/run_data_adequacy.py |
| `src/tradebot/registry/hypothesis_ledger.py` | 291 | 97.3 % | BEHOUDEN | apps/run_data_adequacy.py |
| `src/tradebot/backtest/metrics.py` | 289 | 41.4 % | TESTEN | apps/run_data_adequacy.py |
| `src/tradebot/execution/impact_model.py` | 288 | 98.4 % | BEHOUDEN | apps/run_data_adequacy.py |
| `src/tradebot/utils/time.py` | 285 | 72.6 % | BEHOUDEN | apps/build_features.py |
| `src/tradebot/regime/buckets.py` | 280 | 98.6 % | BEHOUDEN | apps/run_data_adequacy.py |
| `src/tradebot/train/reward.py` | 272 | 93.8 % | BEHOUDEN | apps/build_features.py |
| `src/tradebot/risk/hmm_regime.py` | 270 | 98.2 % | BEHOUDEN | apps/run_data_adequacy.py |
| `src/tradebot/alpha/kalman_ou.py` | 269 | 20.7 % | TESTEN | apps/run_data_adequacy.py |
| `src/tradebot/validation/adequacy_report.py` | 269 | 0.0 % | TESTEN | apps/run_data_adequacy.py |
| `src/tradebot/features/transforms.py` | 268 | 95.3 % | BEHOUDEN | apps/build_features.py |
| `src/tradebot/reporting/phase6_econometrics.py` | 268 | 0.0 % | TESTEN | apps/run_econometric_diagnostics.py |
| `src/tradebot/risk/daily_loss_governor.py` | 255 | 98.2 % | BEHOUDEN | apps/run_data_adequacy.py |
| `src/tradebot/bars/runs.py` | 254 | 9.0 % | TESTEN | apps/build_features.py |
| `src/tradebot/validation/feature_importance.py` | 244 | 97.2 % | BEHOUDEN | apps/run_meta_labeling.py |
| `src/tradebot/validation/dsr.py` | 243 | 89.3 % | BEHOUDEN | apps/run_data_adequacy.py |
| `src/tradebot/data/macro.py` | 239 | 35.8 % | TESTEN | apps/build_features.py |
| `src/tradebot/backtest/regime_overlay.py` | 235 | 98.9 % | BEHOUDEN | apps/run_meta_labeling.py |
| `src/tradebot/cv/bootstrap.py` | 232 | 31.1 % | TESTEN | apps/build_features.py |
| `src/tradebot/validation/diagnostics_report.py` | 223 | 0.0 % | TESTEN | apps/run_econometric_diagnostics.py |
| `src/tradebot/portfolio/risk_parity.py` | 218 | 83.6 % | BEHOUDEN | apps/run_data_adequacy.py |
| `src/tradebot/train/schema_guard.py` | 217 | 91.4 % | BEHOUDEN | apps/build_features.py |
| `src/tradebot/data/ingestion/contract.py` | 211 | 96.3 % | BEHOUDEN | apps/build_features.py |
| `src/tradebot/train/checkpoints.py` | 210 | 53.7 % | TESTEN | apps/build_features.py |
| `src/tradebot/registry/trial_counter.py` | 208 | 100.0 % | BEHOUDEN | apps/run_data_adequacy.py |
| `src/tradebot/portfolio/black_litterman.py` | 207 | 84.3 % | BEHOUDEN | apps/run_data_adequacy.py |
| `src/tradebot/data/ingestion/bybit.py` | 206 | 29.5 % | TESTEN | apps/build_features.py |
| `src/tradebot/registry/promotion.py` | 205 | 82.4 % | BEHOUDEN | apps/run_data_adequacy.py |
| `src/tradebot/data/open_interest.py` | 204 | 79.0 % | BEHOUDEN | apps/build_features.py |
| `src/tradebot/risk/vol_targeting.py` | 204 | 100.0 % | BEHOUDEN | apps/run_data_adequacy.py |
| `src/tradebot/backtest/vectorized.py` | 202 | 100.0 % | BEHOUDEN | apps/run_data_adequacy.py |
| `src/tradebot/volatility/ewma.py` | 196 | 100.0 % | BEHOUDEN | apps/build_features.py |
| `src/tradebot/utils/failfast.py` | 193 | 93.8 % | BEHOUDEN | apps/build_features.py |
| `src/tradebot/data/validation/continuity.py` | 191 | 92.0 % | BEHOUDEN | apps/build_features.py |
| `src/tradebot/cv/uniqueness.py` | 186 | 44.3 % | TESTEN | apps/build_features.py |
| `src/tradebot/features/scaling.py` | 186 | 73.9 % | BEHOUDEN | apps/build_features.py |
| `src/tradebot/backtest/spa.py` | 184 | 73.5 % | BEHOUDEN | apps/run_data_adequacy.py |
| `src/tradebot/data/phase6_universe.py` | 180 | 0.0 % | TESTEN | apps/run_data_adequacy.py |
| `src/tradebot/risk/factor_risk.py` | 179 | 95.5 % | BEHOUDEN | apps/run_data_adequacy.py |
| `src/tradebot/cv/walk_forward.py` | 176 | 88.9 % | BEHOUDEN | apps/build_features.py |
| `src/tradebot/data/validation/gaps.py` | 176 | 95.7 % | BEHOUDEN | apps/build_features.py |
| `src/tradebot/features/stationarity_gate.py` | 176 | 73.0 % | BEHOUDEN | apps/build_features.py |
| `src/tradebot/validation/spa.py` | 176 | 93.3 % | BEHOUDEN | apps/run_data_adequacy.py |
| `src/tradebot/bars/imbalance.py` | 175 | 11.4 % | TESTEN | apps/build_features.py |
| `src/tradebot/registry/phase6_power.py` | 173 | 27.8 % | TESTEN | apps/run_vol_competition.py |
| `src/tradebot/risk/kelly.py` | 170 | 96.9 % | BEHOUDEN | apps/run_data_adequacy.py |
| `src/tradebot/train/meta_train.py` | 167 | 14.9 % | TESTEN | apps/build_features.py |
| `src/tradebot/alpha/macro_regime.py` | 166 | 74.0 % | BEHOUDEN | apps/run_data_adequacy.py |
| `src/tradebot/execution/slippage.py` | 163 | 61.8 % | TESTEN | apps/build_features.py |
| `src/tradebot/alpha/research_harness.py` | 161 | 27.4 % | TESTEN | apps/run_data_adequacy.py |
| `src/tradebot/execution/fees.py` | 158 | 77.4 % | BEHOUDEN | apps/build_features.py |
| `src/tradebot/features/momentum.py` | 154 | 100.0 % | BEHOUDEN | apps/run_data_adequacy.py |
| `src/tradebot/utils/arrays.py` | 154 | 81.5 % | BEHOUDEN | apps/build_features.py |
| `src/tradebot/backtest/__init__.py` | 152 | 69.2 % | TESTEN | apps/run_data_adequacy.py |
| `src/tradebot/alpha/combination.py` | 151 | 92.1 % | BEHOUDEN | apps/run_data_adequacy.py |
| `src/tradebot/alpha/factor_alpha.py` | 150 | 100.0 % | BEHOUDEN | apps/run_data_adequacy.py |
| `src/tradebot/risk/drawdown.py` | 149 | 88.1 % | BEHOUDEN | apps/run_data_adequacy.py |
| `src/tradebot/data/ingestion/crypto_sources.py` | 148 | 50.0 % | TESTEN | apps/build_features.py |
| `src/tradebot/risk/liquidity_risk.py` | 148 | 97.5 % | BEHOUDEN | apps/run_data_adequacy.py |
| `src/tradebot/utils/parquet_io.py` | 148 | 33.3 % | TESTEN | apps/build_features.py |
| `src/tradebot/registry/catalog.py` | 145 | 83.3 % | BEHOUDEN | apps/run_data_adequacy.py |
| `src/tradebot/portfolio/markowitz.py` | 142 | 85.5 % | BEHOUDEN | apps/run_data_adequacy.py |
| `src/tradebot/data/validation/outliers.py` | 140 | 97.9 % | BEHOUDEN | apps/build_features.py |
| `src/tradebot/alpha/mean_reversion.py` | 139 | 87.1 % | BEHOUDEN | apps/run_data_adequacy.py |
| `src/tradebot/data/validation/schema.py` | 138 | 97.9 % | BEHOUDEN | apps/build_features.py |
| `src/tradebot/volatility/garman_klass.py` | 135 | 25.6 % | TESTEN | apps/build_features.py |
| `src/tradebot/alpha/carry.py` | 134 | 87.8 % | BEHOUDEN | apps/run_data_adequacy.py |
| `src/tradebot/train/thompson.py` | 130 | 25.6 % | TESTEN | apps/build_features.py |
| `src/tradebot/portfolio/optimizer.py` | 126 | 81.8 % | BEHOUDEN | apps/run_data_adequacy.py |
| `src/tradebot/portfolio/rebalance.py` | 125 | 84.2 % | BEHOUDEN | apps/run_data_adequacy.py |
| `src/tradebot/utils/hashing.py` | 122 | 97.7 % | BEHOUDEN | apps/build_features.py |
| `src/tradebot/features/regime_features.py` | 120 | 26.7 % | TESTEN | apps/build_features.py |
| `src/tradebot/train/seeded.py` | 120 | 42.4 % | TESTEN | apps/build_features.py |
| `src/tradebot/train/__init__.py` | 110 | 100.0 % | BEHOUDEN | apps/build_features.py |
| `src/tradebot/portfolio/equal_weight.py` | 108 | 100.0 % | BEHOUDEN | apps/run_data_adequacy.py |
| `src/tradebot/cv/purge.py` | 107 | 47.1 % | TESTEN | apps/build_features.py |
| `src/tradebot/risk/__init__.py` | 107 | 100.0 % | BEHOUDEN | apps/run_data_adequacy.py |
| `src/tradebot/risk/position_limits.py` | 105 | 95.3 % | BEHOUDEN | apps/run_data_adequacy.py |
| `src/tradebot/risk/beta_hedge.py` | 104 | 22.8 % | TESTEN | apps/run_data_adequacy.py |
| `src/tradebot/alpha/microstructure.py` | 103 | 91.9 % | BEHOUDEN | apps/run_data_adequacy.py |
| `src/tradebot/backtest/_kernels.py` | 100 | 12.8 % | TESTEN | apps/tune_hparams.py |
| `src/tradebot/data/ingestion/legacy.py` | 100 | 37.5 % | TESTEN | apps/build_features.py |
| `src/tradebot/schemas/bars.py` | 98 | 97.9 % | BEHOUDEN | apps/build_features.py |
| `src/tradebot/volatility/yang_zhang.py` | 97 | 14.0 % | TESTEN | apps/build_features.py |
| `src/tradebot/schemas/features.py` | 90 | 60.0 % | TESTEN | apps/build_features.py |
| `src/tradebot/tune/search_space.py` | 89 | 50.0 % | TESTEN | apps/tune_hparams.py |
| `src/tradebot/registry/lineage.py` | 86 | 50.0 % | TESTEN | apps/run_data_adequacy.py |
| `src/tradebot/schemas/labels.py` | 86 | 80.6 % | BEHOUDEN | apps/build_features.py |
| `src/tradebot/train/_bandit_helpers.py` | 84 | 18.9 % | TESTEN | apps/build_features.py |
| `src/tradebot/train/stack.py` | 75 | 84.0 % | BEHOUDEN | apps/build_features.py |
| `src/tradebot/schemas/events.py` | 72 | 95.0 % | BEHOUDEN | apps/build_features.py |
| `src/tradebot/tune/samplers.py` | 70 | 42.1 % | TESTEN | apps/tune_hparams.py |
| `src/tradebot/schemas/folds.py` | 68 | 92.3 % | BEHOUDEN | apps/build_features.py |
| `src/tradebot/features/blocks.py` | 66 | 65.0 % | TESTEN | apps/build_features.py |
| `src/tradebot/alpha/__init__.py` | 65 | 100.0 % | BEHOUDEN | apps/run_data_adequacy.py |
| `src/tradebot/regime/__init__.py` | 56 | 100.0 % | BEHOUDEN | apps/run_data_adequacy.py |
| `src/tradebot/portfolio/__init__.py` | 55 | 100.0 % | BEHOUDEN | apps/run_data_adequacy.py |
| `src/tradebot/tune/storage.py` | 50 | 30.0 % | TESTEN | apps/tune_hparams.py |
| `src/tradebot/data/validation/__init__.py` | 49 | 100.0 % | BEHOUDEN | apps/build_features.py |
| `src/tradebot/execution/__init__.py` | 47 | 100.0 % | BEHOUDEN | apps/build_features.py |
| `src/tradebot/train/adapter.py` | 47 | 28.0 % | TESTEN | apps/train_cpcv.py |
| `src/tradebot/tune/pruning.py` | 44 | 80.0 % | BEHOUDEN | apps/tune_hparams.py |
| `src/tradebot/volatility/parkinson.py` | 43 | 43.8 % | TESTEN | apps/build_features.py |
| `src/tradebot/features/__init__.py` | 37 | 100.0 % | BEHOUDEN | apps/build_features.py |
| `src/tradebot/data/ingestion/__init__.py` | 35 | 100.0 % | BEHOUDEN | apps/build_features.py |
| `src/tradebot/labeling/fixed_horizon.py` | 34 | 50.0 % | TESTEN | apps/build_features.py |
| `src/tradebot/schemas/_common.py` | 32 | 73.3 % | BEHOUDEN | apps/build_features.py |
| `src/tradebot/__init__.py` | 31 | 66.7 % | TESTEN | apps/build_features.py |
| `src/tradebot/schemas/__init__.py` | 30 | 100.0 % | BEHOUDEN | apps/build_features.py |
| `src/tradebot/schemas/oos_predictions.py` | 27 | 84.2 % | BEHOUDEN | apps/build_features.py |
| `src/tradebot/volatility/__init__.py` | 27 | 100.0 % | BEHOUDEN | apps/build_features.py |
| `src/tradebot/labeling/__init__.py` | 26 | 100.0 % | BEHOUDEN | apps/build_features.py |
| `src/tradebot/reporting/__init__.py` | 26 | 100.0 % | BEHOUDEN | apps/run_econometric_diagnostics.py |
| `src/tradebot/exceptions.py` | 25 | 78.3 % | BEHOUDEN | apps/build_features.py |
| `src/tradebot/volatility/rogers_satchell.py` | 22 | 28.6 % | TESTEN | apps/build_features.py |
| `src/tradebot/registry/__init__.py` | 21 | 100.0 % | BEHOUDEN | apps/run_data_adequacy.py |
| `src/tradebot/schemas/hparams.py` | 21 | 100.0 % | BEHOUDEN | apps/build_features.py |
| `src/tradebot/data/__init__.py` | 20 | 100.0 % | BEHOUDEN | apps/build_features.py |
| `src/tradebot/utils/__init__.py` | 16 | 100.0 % | BEHOUDEN | apps/build_features.py |
| `src/tradebot/cv/__init__.py` | 15 | 100.0 % | BEHOUDEN | apps/build_features.py |
| `src/tradebot/validation/__init__.py` | 15 | 100.0 % | BEHOUDEN | apps/run_data_adequacy.py |
| `src/tradebot/tune/__init__.py` | 14 | 100.0 % | BEHOUDEN | apps/tune_hparams.py |
| `src/tradebot/bars/__init__.py` | 5 | 100.0 % | BEHOUDEN | apps/build_features.py |

## Apps - 40 stuks / 7384 LOC

`in DAG` = aangeroepen door een `dvc.yaml`-stage. `genoemd` = genoemd in
`dvc.yaml`, `Makefile`, `[project.scripts]`, `.github/workflows/` of `docs/`.
Het verdict volgt stap 12: een app die nergens wordt genoemd krijgt een
verwijzing, verhuist, of verdwijnt met registerregel.

| App | LOC | > 80 ? | in DAG | genoemd | Verdict (stap 12) |
|---|---:|:---:|:---:|:---:|---|
| `apps/train_cpcv.py` | 957 | ja | ja | ja | BEHOUDEN |
| `apps/live_paper_trader.py` | 658 | ja | - | **nee** | **BESLISSEN** |
| `apps/paper_trade_runner.py` | 539 | ja | - | ja | BEHOUDEN |
| `apps/feature_selection.py` | 443 | ja | - | **nee** | **BESLISSEN** |
| `apps/tune_hparams.py` | 418 | ja | ja | ja | BEHOUDEN |
| `apps/live_trader.py` | 398 | ja | - | ja | BEHOUDEN |
| `apps/generate_mrm_report.py` | 369 | ja | - | **nee** | **BESLISSEN** |
| `apps/paper_monitor.py` | 305 | ja | - | **nee** | **BESLISSEN** |
| `apps/build_features.py` | 289 | ja | ja | ja | BEHOUDEN |
| `apps/run_meta_labeling.py` | 232 | ja | ja | ja | BEHOUDEN |
| `apps/paper_neutral_trader.py` | 201 | ja | - | **nee** | **BESLISSEN** |
| `apps/run_regime_benchmark.py` | 192 | ja | ja | ja | BEHOUDEN |
| `apps/paper_multi_sleeve.py` | 166 | ja | - | **nee** | **BESLISSEN** |
| `apps/doctor.py` | 142 | ja | - | ja | BEHOUDEN |
| `apps/run_phase5_baseline.py` | 134 | ja | ja | ja | BEHOUDEN |
| `apps/run_vol_competition.py` | 134 | ja | ja | ja | BEHOUDEN |
| `apps/freeze_phase6_preregistrations.py` | 112 | ja | - | **nee** | **BESLISSEN** |
| `apps/freeze_preregistration.py` | 107 | ja | - | **nee** | **BESLISSEN** |
| `apps/run_stress.py` | 101 | ja | - | **nee** | **BESLISSEN** |
| `apps/alpha_combine.py` | 100 | ja | - | ja | BEHOUDEN |
| `apps/regenerate_baseline.py` | 89 | ja | - | ja | BEHOUDEN |
| `apps/ledger_append.py` | 88 | ja | - | ja | BEHOUDEN |
| `apps/run_baseline.py` | 87 | ja | - | **nee** | **BESLISSEN** |
| `apps/featurestore_sync.py` | 86 | ja | - | ja | BEHOUDEN |
| `apps/ingest_crypto.py` | 86 | ja | - | **nee** | **BESLISSEN** |
| `apps/calibrate_impact.py` | 83 | ja | - | ja | BEHOUDEN |
| `apps/freeze_monitoring.py` | 82 | ja | - | ja | BEHOUDEN |
| `apps/monitor_drift.py` | 76 | - | - | ja | BEHOUDEN |
| `apps/run_gates.py` | 76 | - | - | ja | BEHOUDEN |
| `apps/run_data_adequacy.py` | 75 | - | ja | ja | BEHOUDEN |
| `apps/run_econometric_diagnostics.py` | 69 | - | ja | ja | BEHOUDEN |
| `apps/run_eq_units.py` | 68 | - | - | ja | BEHOUDEN |
| `apps/ingest_fx.py` | 67 | - | - | ja | BEHOUDEN |
| `apps/run_adaptive_wf.py` | 64 | - | - | ja | BEHOUDEN |
| `apps/ingest_eia.py` | 63 | - | - | ja | BEHOUDEN |
| `apps/build_equity_universe.py` | 50 | - | - | ja | BEHOUDEN |
| `apps/data_sync.py` | 50 | - | ja | ja | BEHOUDEN |
| `apps/ingest_xasset.py` | 50 | - | - | ja | BEHOUDEN |
| `apps/fetch_factors.py` | 39 | - | - | ja | BEHOUDEN |
| `apps/ingest_edgar.py` | 39 | - | - | **nee** | **BESLISSEN** |

## Scripts - 59 stuks / 9093 LOC

Inclusief de in deze fase toegevoegde `scripts/reachability_map.py`.
Het verdict volgt stap 9 en 12: platformgereedschap blijft in `scripts/`,
onderzoek verhuist naar `research/`.

| Script | LOC | genoemd | Verdict (stap 9/12) |
|---|---:|:---:|---|
| `scripts/reachability_map.py` | 405 | **nee** | BEHOUDEN in `scripts/` - platformgereedschap |
| `scripts/audit_fallbacks.py` | 387 | ja | BEHOUDEN in `scripts/` - platformgereedschap |
| `scripts/check_hardcoded_params.py` | 284 | ja | BEHOUDEN in `scripts/` - platformgereedschap |
| `scripts/check_banned_methods.py` | 275 | ja | BEHOUDEN in `scripts/` - platformgereedschap |
| `scripts/diag_short_barrier_sweep.py` | 269 | ja | **BESLISSEN** - gereedschap of onderzoek |
| `scripts/phase0_baseline.py` | 245 | ja | BEHOUDEN in `scripts/` - platformgereedschap |
| `scripts/w28_cm_carry_eval.py` | 236 | ja | **BESLISSEN** - gereedschap of onderzoek |
| `scripts/xs_alpha_rework.py` | 233 | ja | **BESLISSEN** - gereedschap of onderzoek |
| `scripts/diag_funding_short_edge.py` | 232 | ja | **BESLISSEN** - gereedschap of onderzoek |
| `scripts/breadth_ml_book.py` | 216 | ja | **BESLISSEN** - gereedschap of onderzoek |
| `scripts/phase1_data_gap.py` | 214 | **nee** | BEHOUDEN in `scripts/` - platformgereedschap |
| `scripts/build_data_register.py` | 201 | ja | BEHOUDEN in `scripts/` - platformgereedschap |
| `scripts/unwrap_broad_except.py` | 201 | ja | BEHOUDEN in `scripts/` - platformgereedschap |
| `scripts/pooled_xs_book.py` | 195 | ja | **BESLISSEN** - gereedschap of onderzoek |
| `scripts/w25_fx_carry_eval.py` | 194 | **nee** | **BESLISSEN** - gereedschap of onderzoek |
| `scripts/xs_maxfeat.py` | 183 | ja | **BESLISSEN** - gereedschap of onderzoek |
| `scripts/wave20_runner.py` | 171 | ja | **BESLISSEN** - gereedschap of onderzoek |
| `scripts/neutral_carry_reversal.py` | 165 | **nee** | **BESLISSEN** - gereedschap of onderzoek |
| `scripts/walkforward_forward.py` | 162 | ja | **BESLISSEN** - gereedschap of onderzoek |
| `scripts/vol_target_sweep.py` | 161 | **nee** | **BESLISSEN** - gereedschap of onderzoek |
| `scripts/wave_final_eval.py` | 159 | ja | **BESLISSEN** - gereedschap of onderzoek |
| `scripts/meta_label_book.py` | 155 | ja | **BESLISSEN** - gereedschap of onderzoek |
| `scripts/w28_roll_calendar_validation.py` | 155 | **nee** | **BESLISSEN** - gereedschap of onderzoek |
| `scripts/xs_multihorizon.py` | 155 | ja | **BESLISSEN** - gereedschap of onderzoek |
| `scripts/diag_xsectional_ls.py` | 154 | ja | **BESLISSEN** - gereedschap of onderzoek |
| `scripts/w28_kg_b4_capital_granularity.py` | 152 | **nee** | **BESLISSEN** - gereedschap of onderzoek |
| `scripts/w28_dd_shape_calibration.py` | 147 | ja | **BESLISSEN** - gereedschap of onderzoek |
| `scripts/true_alpha_gates.py` | 146 | ja | **BESLISSEN** - gereedschap of onderzoek |
| `scripts/regime_timing_book.py` | 144 | ja | **BESLISSEN** - gereedschap of onderzoek |
| `scripts/trend_robust_sweep.py` | 143 | ja | **BESLISSEN** - gereedschap of onderzoek |
| `scripts/market_neutral_alpha.py` | 139 | ja | **BESLISSEN** - gereedschap of onderzoek |
| `scripts/train_new_assets.py` | 138 | **nee** | **BESLISSEN** - gereedschap of onderzoek |
| `scripts/ml_daily_book.py` | 134 | **nee** | **BESLISSEN** - gereedschap of onderzoek |
| `scripts/broad_statarb.py` | 133 | **nee** | **BESLISSEN** - gereedschap of onderzoek |
| `scripts/w28_seed_ledger.py` | 129 | ja | **BESLISSEN** - gereedschap of onderzoek |
| `scripts/xs_harvest.py` | 128 | ja | **BESLISSEN** - gereedschap of onderzoek |
| `scripts/mn_optimized_exec.py` | 124 | ja | **BESLISSEN** - gereedschap of onderzoek |
| `scripts/multi_sleeve_combine.py` | 122 | ja | **BESLISSEN** - gereedschap of onderzoek |
| `scripts/xs_diagnose.py` | 119 | ja | **BESLISSEN** - gereedschap of onderzoek |
| `scripts/robust_combine.py` | 116 | ja | **BESLISSEN** - gereedschap of onderzoek |
| `scripts/validate_neutral_book.py` | 113 | ja | **BESLISSEN** - gereedschap of onderzoek |
| `scripts/w26_seasonality_diag.py` | 113 | ja | **BESLISSEN** - gereedschap of onderzoek |
| `scripts/fetch_altdata.py` | 111 | ja | **BESLISSEN** - gereedschap of onderzoek |
| `scripts/monitor_retrain.py` | 110 | **nee** | BEHOUDEN in `scripts/` - platformgereedschap |
| `scripts/combined_alpha_book.py` | 109 | **nee** | **BESLISSEN** - gereedschap of onderzoek |
| `scripts/w27_xasset_tsmom_eval.py` | 105 | ja | **BESLISSEN** - gereedschap of onderzoek |
| `scripts/combined_neutral_book.py` | 102 | ja | **BESLISSEN** - gereedschap of onderzoek |
| `scripts/build_feature_store.py` | 99 | **nee** | BEHOUDEN in `scripts/` - platformgereedschap |
| `scripts/w27_g4.py` | 87 | ja | **BESLISSEN** - gereedschap of onderzoek |
| `scripts/fetch_broad_ohlcv.py` | 84 | ja | **BESLISSEN** - gereedschap of onderzoek |
| `scripts/adaptive_wf_pbo.py` | 81 | ja | **BESLISSEN** - gereedschap of onderzoek |
| `scripts/funding_edge_test.py` | 78 | **nee** | **BESLISSEN** - gereedschap of onderzoek |
| `scripts/fetch_broad_universe.py` | 75 | **nee** | **BESLISSEN** - gereedschap of onderzoek |
| `scripts/paper_trade_report.py` | 74 | ja | BEHOUDEN in `scripts/` - platformgereedschap |
| `scripts/w23c_eval.py` | 70 | **nee** | **BESLISSEN** - gereedschap of onderzoek |
| `scripts/fetch_bybit_spot.py` | 69 | **nee** | **BESLISSEN** - gereedschap of onderzoek |
| `scripts/w23ab_eval.py` | 68 | **nee** | **BESLISSEN** - gereedschap of onderzoek |
| `scripts/fetch_funding_universe.py` | 65 | **nee** | **BESLISSEN** - gereedschap of onderzoek |
| `scripts/w22_g4_diagnostic.py` | 59 | **nee** | **BESLISSEN** - gereedschap of onderzoek |

