# CODE REGISTER — de overlevende oppervlakte

> **Geverifieerd tegen de codebase op 2026-09-04** (Phase 9, stap 10).
> Dit document beschrijft wat er in `src/`
> staat, door welk entrypoint het wordt bereikt, hoeveel dekking het draagt en
> of het onder een ratchet staat. Het is bedoeld voor wie het project overneemt
> en wil weten wat hij mag aanraken.
>
> Bereikbaarheid komt uit `scripts/reachability_map.py`, dekking uit
> `pytest-cov` 7.1.0 op de referentie-interpreter uit `docs/runbook.md` par. 0.
> Beide zijn gemeten, niet beweerd.

## De regel die dit register afdwingt

Elke module in klasse A of B draagt **ofwel dekking boven de 70 %-drempel,
ofwel een geregistreerde ratchet met een datum en een eigenaar**. Er is geen
derde optie. "Eigenaar" is een ROL; dit project heeft er nog geen bezet.

| | stap 10 | **nagemeten, stap 16** |
|---|---:|---:|
| modules in `src/` | 290 | **290** |
| LOC | 71.000 | **71.025** |
| dekking, LOC-gewogen | 60,3 % | **60,6 %** |
| dekking, statements (`pytest-cov`) | 55,82 % | **56,64 %** |
| modules boven de drempel | 176 | **178** |
| modules onder een ratchet (A/B) | 110 | **108** |
| modules in een derde categorie | 0 | **0** |
| ratchet-herzieningsdatum | | **2027-03-04** |

> **Nagemeten 2026-09-04 bij het exit-rapport** met
> `reachability_map.py --coverage reports/phase9_coverage_after.json`. De
> ratchetlijst hieronder is ongewijzigd gelaten: zij is geschreven op de
> nulmeting en dat is de referentie waartegen de ratchet krimpt. Twee regels
> zijn er inmiddels **uit gegroeid** — `selection/__init__.py` (100,0 %) en
> `selection/mda.py` (96,2 %), beide dankzij de tests van stap 11 voor het
> pakket dat er nul had. Zij blijven staan zodat zichtbaar is dát ze zijn
> opgelost. De 71.025 LOC tegen 71.000 is geen groei van code maar de acht
> `# LOC-EXCEPTION:`-regels van stap 13 en hun context.
>
> De controle die telt, is omgekeerd uitgevoerd — niet *"staat elke
> ratchet-regel in het register"* maar *"staat elke module onder de drempel in
> het register"*: **108 van 108, nul modules onder de drempel die hier niet
> staan.** Bewijs in `reports/phase9_exit_report.md`, criterium 5.

## Per pakket

| Pakket | Waarvoor het bestaat | Modules | LOC | Dekking | Onder ratchet | Bereikt via | Eigenaar |
|---|---|---:|---:|---:|---:|---|---|
| `features` | de Phase 2-featureregistry en de causale featureblokken | 20 | 7318 | 59.0 % | 7 | `apps/build_features.py` | Research |
| `validation` | data-adequacy, campagnes en statistische poorten | 17 | 6348 | 74.1 % | 5 | `apps/run_data_adequacy.py` | Research |
| `data` | point-in-time ingestion, validatie en de PIT-store | 35 | 5771 | 43.1 % | 20 | `apps/build_features.py` | Engineering |
| `live` | de live/paper-handelsketen; zie openstaand besluit 3 | 14 | 4896 | 38.5 % | 10 | `apps/live_paper_trader.py` | Engineering |
| `train` | CPCV-training, ensembles en meta-labeling | 16 | 4849 | 44.0 % | 10 | `apps/build_features.py` | Research |
| `risk` | de soevereine risicolaag; RiskDecision en de kill switches | 17 | 4671 | 86.8 % | 3 | `apps/run_data_adequacy.py` | Research |
| `backtest` | de authoritative event-driven engine, accounting en evaluatie | 14 | 4559 | 54.6 % | 8 | `apps/run_data_adequacy.py` | Research |
| `alpha` | alpha-units en hun combinatie; levert gewenste exposure aan de risicolaag | 26 | 4373 | 72.0 % | 6 | `apps/run_data_adequacy.py` | Research |
| `execution` | orderrouting, marktimpact en spread-modellering | 10 | 3240 | 62.9 % | 4 | `apps/build_features.py` | Engineering |
| `labeling` | triple-barrier, CUSUM, trend-scanning en meta-labels | 7 | 2752 | 31.1 % | 5 | `apps/build_features.py` | Research |
| `portfolio` | portefeuilleconstructie; grotendeels legacy onder DI-10 | 11 | 2661 | 55.8 % | — | `apps/run_data_adequacy.py` | Research |
| `registry` | lineage, experimenten en de symbol lifecycle | 11 | 2303 | 86.6 % | 2 | `apps/run_data_adequacy.py` | Engineering |
| `reporting` | rapportgeneratie voor de Phase 6-campagnes | 5 | 2259 | 28.4 % | 3 | `apps/run_econometric_diagnostics.py` | Engineering |
| `regime` | regimemodellen en het forward filter | 5 | 1969 | 97.6 % | — | `apps/run_data_adequacy.py` | Research |
| `volatility` | EWMA, GARCH-familie, HAR-RV en range-estimators | 9 | 1830 | 81.6 % | 4 | `apps/build_features.py` | Research |
| `monitoring` | drift-, executie- en Sharpe-bewaking met bevroren drempels | 11 | 1797 | 65.3 % | 4 | `apps/live_paper_trader.py` | Engineering |
| `schemas` | de datacontracten; pandera- en pydantic-modellen | 11 | 1657 | 93.1 % | 1 | `apps/build_features.py` | Engineering |
| `oms` | order management op de live-weg | 7 | 1273 | 69.8 % | 2 | `apps/live_paper_trader.py` | Engineering |
| `tune` | hyperparameteroptimalisatie | 6 | 1222 | 19.4 % | 4 | `apps/tune_hparams.py` | Research |
| `cv` | combinatorisch purged cross-validation en uniqueness | 6 | 1042 | 64.3 % | 3 | `apps/build_features.py` | Research |
| `compliance` | audittrail en handelsbeperkingen | 6 | 1015 | 90.8 % | — | `apps/generate_mrm_report.py` | Engineering |
| `bars` | informatiegedreven bars (AFML par. 2) | 5 | 937 | 14.9 % | 3 | `apps/build_features.py` | Engineering |
| `utils` | hashing, arrays, parquet-io en fail-fast-helpers | 6 | 918 | 76.0 % | 1 | `apps/build_features.py` | Engineering |
| `tca` | transaction cost analysis en de IS-decompositie | 6 | 723 | 88.9 % | — | `tests/integration/test_engine_parity.py` | Research |
| `selection` | feature-selectie (SFI, causale MDA) | 3 | 293 | 0.0 % | 3 | `apps/feature_selection.py` | Research |
| `featurestore` | gecertificeerde L3-featureopslag | 4 | 268 | 78.3 % | 1 | `apps/featurestore_sync.py` | Engineering |
| `(top-level)` | pakketmarkering | 2 | 56 | 71.8 % | 1 | `apps/build_features.py` | Engineering |
| **totaal** | | **290** | **71000** | **60.3 %** | **110** | | |

## Klasse D — test-only, met hun afnemer

Twintig modules worden uitsluitend vanuit `tests/` bereikt. Dat is geen
restcategorie: achttien dragen een aanwijsbaar contract dat op bedrading
wacht, twee zijn ambitie zonder afnemer. Geen van beide groepen verhuist,
want een D-module verplaatsen breekt per definitie zijn eigen test — en dat
verandert een `passed` in een `error`.

| Module | LOC | Dekking | Soort | Grondslag | Waarom het leeft |
|---|---:|---:|:---:|:---:|---|
| `registry/lifecycle.py` | 310 | 100.0 % | **CONTRACT** | DI-15 | `SymbolLifecycle` weigert bars voor de listing of na de delisting; wacht op een databron met delisting-historie |
| `tca/post_trade.py` | 304 | 95.7 % | **CONTRACT** | DI-5 | `docs/tca_methodology.md` |
| `volatility/har_rv.py` | 270 | 81.7 % | **CONTRACT** | AD-23 | buiten het meetdomein, geen afnemer meer |
| `monitoring/execution_drift.py` | 228 | 97.1 % | **CONTRACT** | governance | drempels bevroren in `monitoring_config_hash.json` |
| `registry/experiment.py` | 223 | 77.0 % | **CONTRACT** | governance | experimentregistratie |
| `monitoring/vol_forecast_monitor.py` | 212 | 98.5 % | **CONTRACT** | governance | drempels bevroren in `monitoring_config_hash.json` |
| `bars/dollar.py` | 173 | 45.5 % | **AMBITIE** | — | geen afnemer gevonden. Blijft staan: de numba-kernel staat in `bars/_kernels.py` (klasse A) en verplaatsen breekt `test_audit_fixes.py` |
| `data/orderbook.py` | 150 | 92.5 % | **AMBITIE** | — | geen afnemer gevonden. Blijft staan: verplaatsen breekt `test_wave6.py` |
| `monitoring/sharpe_monitor.py` | 143 | 96.2 % | **CONTRACT** | DI-7 | de HALT-taak; `test_external_monitors_can_actually_halt.py` rijdt een echte degradatie door een echte breaker |
| `tca/implementation_shortfall.py` | 136 | 79.4 % | **CONTRACT** | DI-5 | de IS-decompositie |
| `data/funding.py` | 129 | 21.4 % | **CONTRACT** | governance | fundingreeks gecertificeerd in `data_hashes.json` |
| `alpha/fx_tsmom.py` | 119 | 85.4 % | **CONTRACT** | D1 | `EXPANSION_RESEARCH_2026-08-10.md`: *this file IS the D1 unit* |
| `features/funding_carry.py` | 106 | 95.7 % | **CONTRACT** | research | `SHORT_ALPHA_RESEARCH_2026-05-30.md`, causaliteitseis op de lag |
| `features/open_interest.py` | 106 | 96.8 % | **CONTRACT** | governance | OI-reeks gecertificeerd in `data_hashes.json` |
| `data/panel.py` | 105 | 100.0 % | **CONTRACT** | R-1 | `asof_join`; lookahead-test `test_asof_join_crypto.py` |
| `tca/arrival_price.py` | 92 | 74.2 % | **CONTRACT** | DI-5 | referentieprijs van de decompositie |
| `tca/pre_trade.py` | 87 | 100.0 % | **CONTRACT** | DI-5 | DI-5 corrigeerde een onjuiste claim over deze module |
| `tca/report.py` | 77 | 80.0 % | **CONTRACT** | DI-5 | `reports/TCA_CALIBRATION_REPORT.md` |
| `schemas/orders.py` | 39 | 100.0 % | **CONTRACT** | contract | het ordercontract |
| `tca/__init__.py` | 27 | 100.0 % | **CONTRACT** | DI-5 | pakketmarkering van de TCA-laag |

## Klasse E — onbereikbaar, en toch behouden

De bereikbaarheidsscanner meet imports. Deze twee modules worden door geen
enkele import bereikt en dragen tóch een contract: een test die hun BESTAAN
asserteert. `scripts/reachability_map.py --strict` kent ze als geregistreerde
uitzondering; elke andere klasse-E-module laat de poort rood worden.

| Module | LOC | Waarom behouden |
|---|---:|---|
| `portfolio/legacy_sizing.py` | 824 | bestaansassertie in `test_risk_alpha_decoupling.py` (Phase 4 exit-criterium 3), DI-10, ratchetpost in `check_hardcoded_params.py` |
| `portfolio/covariance.py` | 194 | bestaansassertie in `test_risk_alpha_decoupling.py` (Phase 4 exit-criterium 3), DI-10 |

## De ratchet — klasse A en B onder de drempel

**110 modules / 32041 LOC.** Zij mogen blijven, maar hun dekking mag niet dalen en
de post wordt herzien op **2027-03-04**. Gesorteerd op `LOC x (1 - dekking)`: de
volgorde waarin het bijschrijven van dekking het meeste oplevert.

| # | Module | Klasse | LOC | Dekking | Bereikt via | Eigenaar |
|---:|---|:---:|---:|---:|---|---|
| 1 | `train/ensemble.py` | A | 1117 | 8.3 % | apps/build_features.py | Research |
| 2 | `labeling/meta.py` | A | 1181 | 18.0 % | apps/build_features.py | Research |
| 3 | `features/regime.py` | A | 1056 | 11.9 % | apps/build_features.py | Research |
| 4 | `tune/objective.py` | A | 955 | 10.4 % | apps/tune_hparams.py | Research |
| 5 | `backtest/evaluation.py` | A | 1054 | 27.2 % | apps/run_data_adequacy.py | Research |
| 6 | `reporting/phase6_vol_competition.py` | A | 754 | 0.0 % | apps/run_vol_competition.py | Engineering |
| 7 | `features/ta.py` | A | 775 | 9.1 % | apps/build_features.py | Research |
| 8 | `live/engine.py` | B | 913 | 34.0 % | apps/live_paper_trader.py | Engineering |
| 9 | `live/model_signal.py` | B | 591 | 0.0 % | apps/live_paper_trader.py | Engineering |
| 10 | `reporting/phase6_regime_benchmark.py` | A | 578 | 0.0 % | apps/run_regime_benchmark.py | Engineering |
| 11 | `live/feed.py` | B | 747 | 22.7 % | apps/live_paper_trader.py | Engineering |
| 12 | `features/_ta_kernels.py` | A | 635 | 14.9 % | apps/build_features.py | Research |
| 13 | `train/catboost.py` | A | 600 | 11.9 % | apps/build_features.py | Research |
| 14 | `execution/market_impact.py` | A | 621 | 17.4 % | apps/build_features.py | Engineering |
| 15 | `validation/meta_label_campaign.py` | A | 504 | 0.0 % | apps/run_meta_labeling.py | Research |
| 16 | `backtest/baseline_report.py` | A | 488 | 0.0 % | apps/run_data_adequacy.py | Research |
| 17 | `train/_scalers.py` | A | 498 | 12.0 % | apps/build_features.py | Research |
| 18 | `data/crypto.py` | A | 501 | 15.0 % | apps/build_features.py | Engineering |
| 19 | `labeling/trend_scanning.py` | A | 440 | 15.8 % | apps/build_features.py | Research |
| 20 | `data/tradfi_macro.py` | A | 411 | 12.0 % | apps/build_features.py | Engineering |
| 21 | `live/feature_updater.py` | B | 572 | 38.2 % | apps/live_paper_trader.py | Engineering |
| 22 | `validation/vol_campaign.py` | A | 580 | 44.7 % | apps/run_vol_competition.py | Research |
| 23 | `bars/_kernels.py` | A | 330 | 3.9 % | apps/build_features.py | Engineering |
| 24 | `risk/stress_report.py` | A | 294 | 0.0 % | apps/run_data_adequacy.py | Research |
| 25 | `labeling/cusum.py` | A | 408 | 28.3 % | apps/build_features.py | Research |
| 26 | `data/crypto_macro.py` | A | 332 | 14.7 % | apps/build_features.py | Engineering |
| 27 | `validation/adequacy_report.py` | A | 269 | 0.0 % | apps/run_data_adequacy.py | Research |
| 28 | `reporting/phase6_econometrics.py` | A | 268 | 0.0 % | apps/run_econometric_diagnostics.py | Engineering |
| 29 | `execution/simulator.py` | A | 406 | 34.0 % | apps/build_features.py | Engineering |
| 30 | `execution/spread.py` | A | 314 | 17.9 % | apps/tune_hparams.py | Engineering |
| 31 | `live/signal_runner.py` | B | 321 | 23.2 % | apps/live_paper_trader.py | Engineering |
| 32 | `alpha/csm_volume_clock.py` | A | 296 | 17.6 % | apps/run_data_adequacy.py | Research |
| 33 | `bars/runs.py` | A | 254 | 9.0 % | apps/build_features.py | Engineering |
| 34 | `oms/router.py` | B | 315 | 27.9 % | apps/live_paper_trader.py | Engineering |
| 35 | `validation/diagnostics_report.py` | A | 223 | 0.0 % | apps/run_econometric_diagnostics.py | Research |
| 36 | `monitoring/live_drift_monitor.py` | B | 218 | 0.0 % | apps/live_paper_trader.py | Engineering |
| 37 | `labeling/triple_barrier.py` | A | 360 | 40.1 % | apps/build_features.py | Research |
| 38 | `alpha/kalman_ou.py` | A | 269 | 20.7 % | apps/run_data_adequacy.py | Research |
| 39 | `monitoring/drift.py` | B | 314 | 32.4 % | apps/live_paper_trader.py | Engineering |
| 40 | `live/portfolio_controller.py` | B | 331 | 36.7 % | apps/live_paper_trader.py | Engineering |
| 41 | `data/phase6_universe.py` | A | 180 | 0.0 % | apps/run_data_adequacy.py | Engineering |
| 42 | `data/sources/eia.py` | B | 177 | 0.0 % | apps/ingest_eia.py | Engineering |
| 43 | `data/perp_feed.py` | B | 173 | 0.0 % | apps/paper_multi_sleeve.py | Engineering |
| 44 | `backtest/metrics.py` | A | 289 | 41.4 % | apps/run_data_adequacy.py | Research |
| 45 | `cv/bootstrap.py` | A | 232 | 31.1 % | apps/build_features.py | Research |
| 46 | `backtest/phase5_baseline.py` | A | 304 | 48.0 % | apps/run_meta_labeling.py | Research |
| 47 | `features/microstructure.py` | A | 474 | 66.9 % | apps/run_data_adequacy.py | Research |
| 48 | `data/equity_universe.py` | B | 157 | 0.0 % | apps/build_equity_universe.py | Engineering |
| 49 | `bars/imbalance.py` | A | 175 | 11.4 % | apps/build_features.py | Engineering |
| 50 | `data/sources/stooq.py` | B | 155 | 0.0 % | apps/build_equity_universe.py | Engineering |
| 51 | `data/macro.py` | A | 239 | 35.8 % | apps/build_features.py | Engineering |
| 52 | `validation/regime_benchmark.py` | A | 489 | 69.7 % | apps/run_regime_benchmark.py | Research |
| 53 | `data/ingestion/bybit.py` | A | 206 | 29.5 % | apps/build_features.py | Engineering |
| 54 | `train/meta_train.py` | A | 167 | 14.9 % | apps/build_features.py | Research |
| 55 | `backtest/baseline_runner.py` | A | 298 | 52.3 % | apps/run_data_adequacy.py | Research |
| 56 | `selection/mda.py` | B | 142 | 0.0 % | apps/feature_selection.py | Research |
| 57 | `selection/sfi.py` | B | 137 | 0.0 % | apps/feature_selection.py | Research |
| 58 | `live/judge_gate.py` | B | 174 | 22.4 % | apps/live_paper_trader.py | Engineering |
| 59 | `train/quant_arch.py` | A | 393 | 66.5 % | apps/build_features.py | Research |
| 60 | `alpha/momentum.py` | A | 404 | 68.1 % | apps/run_data_adequacy.py | Research |
| 61 | `registry/phase6_power.py` | A | 173 | 27.8 % | apps/run_vol_competition.py | Engineering |
| 62 | `risk/var.py` | A | 350 | 64.6 % | apps/run_data_adequacy.py | Research |
| 63 | `live/cusum_filter.py` | B | 157 | 22.0 % | apps/live_paper_trader.py | Engineering |
| 64 | `data/sources/edgar.py` | B | 120 | 0.0 % | apps/ingest_edgar.py | Engineering |
| 65 | `alpha/research_harness.py` | A | 161 | 27.4 % | apps/run_data_adequacy.py | Research |
| 66 | `features/cfi.py` | A | 371 | 69.6 % | apps/build_features.py | Research |
| 67 | `data/sources/yfinance_backup.py` | B | 108 | 0.0 % | apps/build_equity_universe.py | Engineering |
| 68 | `cv/uniqueness.py` | A | 186 | 44.3 % | apps/build_features.py | Research |
| 69 | `backtest/pbo.py` | B | 122 | 17.6 % | apps/run_adaptive_wf.py | Research |
| 70 | `volatility/garman_klass.py` | A | 135 | 25.6 % | apps/build_features.py | Research |
| 71 | `data/sources/fred.py` | B | 129 | 22.9 % | apps/ingest_fx.py | Engineering |
| 72 | `utils/parquet_io.py` | A | 148 | 33.3 % | apps/build_features.py | Engineering |
| 73 | `train/checkpoints.py` | A | 210 | 53.7 % | apps/build_features.py | Research |
| 74 | `train/thompson.py` | A | 130 | 25.6 % | apps/build_features.py | Research |
| 75 | `data/sources/wiki_constituents.py` | B | 149 | 35.6 % | apps/build_equity_universe.py | Engineering |
| 76 | `data/sources/kenfrench.py` | B | 89 | 0.0 % | apps/fetch_factors.py | Engineering |
| 77 | `features/regime_features.py` | A | 120 | 26.7 % | apps/build_features.py | Research |
| 78 | `backtest/_kernels.py` | A | 100 | 12.8 % | apps/tune_hparams.py | Research |
| 79 | `alpha/multi_sleeve_book.py` | B | 120 | 29.1 % | apps/paper_multi_sleeve.py | Research |
| 80 | `oms/paper_oms.py` | B | 272 | 69.1 % | apps/live_paper_trader.py | Engineering |
| 81 | `volatility/yang_zhang.py` | A | 97 | 14.0 % | apps/build_features.py | Research |
| 82 | `data/edgar_universe.py` | B | 83 | 0.0 % | apps/ingest_edgar.py | Engineering |
| 83 | `risk/beta_hedge.py` | A | 104 | 22.8 % | apps/run_data_adequacy.py | Research |
| 84 | `data/xasset_proxy.py` | B | 160 | 50.0 % | apps/ingest_xasset.py | Engineering |
| 85 | `alpha/neutral_book.py` | B | 118 | 33.8 % | apps/paper_multi_sleeve.py | Research |
| 86 | `data/ingestion/crypto_sources.py` | A | 148 | 50.0 % | apps/build_features.py | Engineering |
| 87 | `monitoring/prob_calibration.py` | B | 108 | 34.6 % | apps/live_paper_trader.py | Engineering |
| 88 | `train/seeded.py` | A | 120 | 42.4 % | apps/build_features.py | Research |
| 89 | `train/_bandit_helpers.py` | A | 84 | 18.9 % | apps/build_features.py | Research |
| 90 | `data/ingestion/legacy.py` | A | 100 | 37.5 % | apps/build_features.py | Engineering |
| 91 | `execution/slippage.py` | A | 163 | 61.8 % | apps/build_features.py | Engineering |
| 92 | `cv/purge.py` | A | 107 | 47.1 % | apps/build_features.py | Research |
| 93 | `data/sources/base.py` | B | 143 | 64.6 % | apps/build_equity_universe.py | Engineering |
| 94 | `monitoring/feature_health.py` | B | 101 | 51.0 % | apps/live_paper_trader.py | Engineering |
| 95 | `live/exchange_status.py` | B | 72 | 32.1 % | apps/live_paper_trader.py | Engineering |
| 96 | `backtest/__init__.py` | A | 152 | 69.2 % | apps/run_data_adequacy.py | Research |
| 97 | `tune/search_space.py` | A | 89 | 50.0 % | apps/tune_hparams.py | Research |
| 98 | `featurestore/backfill.py` | B | 73 | 40.9 % | apps/featurestore_sync.py | Engineering |
| 99 | `registry/lineage.py` | A | 86 | 50.0 % | apps/run_data_adequacy.py | Engineering |
| 100 | `tune/samplers.py` | A | 70 | 42.1 % | apps/tune_hparams.py | Research |
| 101 | `schemas/features.py` | A | 90 | 60.0 % | apps/build_features.py | Engineering |
| 102 | `tune/storage.py` | A | 50 | 30.0 % | apps/tune_hparams.py | Research |
| 103 | `train/adapter.py` | A | 47 | 28.0 % | apps/train_cpcv.py | Research |
| 104 | `live/sigterm.py` | B | 58 | 43.8 % | apps/live_paper_trader.py | Engineering |
| 105 | `volatility/parkinson.py` | A | 43 | 43.8 % | apps/build_features.py | Research |
| 106 | `features/blocks.py` | A | 66 | 65.0 % | apps/build_features.py | Research |
| 107 | `labeling/fixed_horizon.py` | A | 34 | 50.0 % | apps/build_features.py | Research |
| 108 | `volatility/rogers_satchell.py` | A | 22 | 28.6 % | apps/build_features.py | Research |
| 109 | `selection/__init__.py` | B | 14 | 0.0 % | apps/feature_selection.py | Research |
| 110 | `__init__.py` | A | 31 | 66.7 % | apps/build_features.py | Engineering |

## Wat dit register NIET zegt

Dekking is geen bewijs van juistheid. `PROJECT_STATE.md` waarschuwt letterlijk:
*een groen criterium is pas bewijs als de bijbehorende test rood kan worden om
de reden waarvoor het criterium bestaat.* Een module op 95 % kan nog steeds
uitsluitend op vorm getoetst zijn. De kolom telt regels die een test heeft
aangeraakt, niet gedrag dat een test heeft vastgelegd.
