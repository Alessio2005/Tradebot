# `research/` — de onderzoekstrack

> **Aangemaakt 2026-09-04, Phase 9 stap 9.** Sluit **DI-8**.

Sectie 20 van `docs/ARCHITECTUUR_AUDIT_2026-08-22.md` schrijft een
`research/`-track voor zonder productiecode. Tot deze fase stonden de
wave-onderzoeksscripts naast het platformgereedschap in `scripts/`, en dat was
de reden dat DI-8 sinds Phase 0 doorschoof: verplaatsen zou de verwijzingen in
de wave-documentatie breken. Die verwijzingen zijn in deze commit meeverhuisd,
dus de voorwaarde is vervuld.

## Wat hier staat, en wat niet

| | |
|---|---|
| **`research/`** | onderzoek: wave-evaluaties, boekconstructies, diagnostiek, en de data-pulls die daarbij horen. Draait op verzoek, staat niet in de DAG, en levert geen productiegedrag. |
| **`scripts/`** | platformgereedschap: poorten, scanners en registergeneratoren waar de repository zelf van afhangt. |

De elf bestanden die in `scripts/` zijn gebleven:

```
audit_fallbacks.py          Phase 0 - poort op stille degradatie
check_banned_methods.py     Phase 2 - poort op verboden validatiemethoden
check_hardcoded_params.py   Phase 0 - poort op hardcoded parameters
reachability_map.py         Phase 9 - bereikbaarheidskaart, poort op klasse E
build_data_register.py      Phase 1 - genereert docs/DATA_REGISTER.md
build_feature_store.py      Phase 2 - bouwt de gecertificeerde L3 feature store
phase0_baseline.py          Phase 0 - bewijsartefact
phase1_data_gap.py          Phase 1 - bewijsartefact
unwrap_broad_except.py      Phase 0 - helper bij de fail-fast-opruiming
monitor_retrain.py          operationeel - dashboard op een lopende retrain
paper_trade_report.py       operationeel - leest live-engine-state uit
```

## Herkomst

Alle 48 bestanden komen uit `scripts/` en zijn met `git mv` verplaatst, dus
`git log --follow research/<naam>.py` geeft de volledige historie. De
verhuizing is klasse-neutraal gemeten: de bereikbaarheidskaart geeft vóór en
na exact dezelfde verdeling (C = 8 modules / 1.108 LOC, D = 20 / 3.036).

| Bestand | LOC | Wave | Waarvoor |
|---|---:|:---:|---|
| `adaptive_wf_pbo.py` | 81 | W19 | conclusive CSCV PBO for the adaptive book (Wave 19). |
| `breadth_ml_book.py` | 216 | W14 | 70x2 (99x2) directional ML breadth book (Wave 14). |
| `broad_statarb.py` | 133 | — | breadth-fixed market-neutral cross-sectional stat-arb. |
| `combined_alpha_book.py` | 109 | — | CHIEF step 1: combine the genuine, correctly-measured |
| `combined_neutral_book.py` | 102 | — | CHIEF /goal: regime-independent alpha via |
| `diag_funding_short_edge.py` | 232 | — | H2 diagnostic: is there positive SHORT edge in funding? |
| `diag_short_barrier_sweep.py` | 269 | — | Grondige short-alpha analyse (model-vrij). |
| `diag_xsectional_ls.py` | 154 | — | H3 capstone: heeft de SHORT-LEG edge in een |
| `fetch_altdata.py` | 111 | W16 | free market-wide alt-data panel (Wave 16). |
| `fetch_broad_ohlcv.py` | 84 | W14 | daily OHLCV panel for the 70x2 breadth rework (Wave 14). |
| `fetch_broad_universe.py` | 75 | W2 | wider perp panel for the breadth test (Wave 2). |
| `fetch_bybit_spot.py` | 69 | — | actual Bybit spot daily closes (the live-tradeable venue). |
| `fetch_funding_universe.py` | 65 | — | free, full-history funding rates (Binance) for the |
| `funding_edge_test.py` | 78 | — | does free funding-rate data add real edge? |
| `market_neutral_alpha.py` | 139 | — | strictly dollar/beta-neutral cross-sectional |
| `meta_label_book.py` | 155 | W14 | meta-labeling done right (Wave 14, LdP central technique). |
| `ml_daily_book.py` | 134 | — | daily, bidirectional, low-frequency ML book (Wave: harden). |
| `mn_optimized_exec.py` | 124 | — | leverage + execution optimization of the verified MN book. |
| `multi_sleeve_combine.py` | 122 | W15 | stack orthogonal sleeves toward higher Sharpe (Wave 15). |
| `neutral_carry_reversal.py` | 165 | — | dollar-neutral short-term-reversal + funding-carry. |
| `pooled_xs_book.py` | 195 | W14 | pooled cross-sectional ML book (Wave 14 rework). |
| `regime_timing_book.py` | 144 | W16 | directional market-timing sleeve from free alt-data (Wave 16). |
| `robust_combine.py` | 116 | W17 | best robust combination + the "every year >60%" math (Wave 17). |
| `train_new_assets.py` | 138 | — | research/train_new_assets.py |
| `trend_robust_sweep.py` | 143 | W17 | all-weather long/short trend, multi-timeframe/RR (Wave 17). |
| `true_alpha_gates.py` | 146 | — | CHIEF /goal true-alpha gate battery (G4/G5/G6 + DSR). |
| `validate_neutral_book.py` | 113 | — | CHIEF out-of-sample validation of the 3-sleeve |
| `vol_target_sweep.py` | 161 | — | find the most aggressive risk setting that drives |
| `w22_g4_diagnostic.py` | 59 | W22 | research/w22_g4_diagnostic.py |
| `w23ab_eval.py` | 68 | W23a | research/w23ab_eval.py |
| `w23c_eval.py` | 70 | W23c | research/w23c_eval.py |
| `w25_fx_carry_eval.py` | 194 | W25 | Wave 25: G10 FX carry unit, full per-unit checklist. |
| `w26_seasonality_diag.py` | 113 | W26 | Wave 26 step 1: MEASURE crypto intraday seasonality. |
| `w27_g4.py` | 87 | W27 | !/usr/bin/env python |
| `w27_xasset_tsmom_eval.py` | 105 | W27 | !/usr/bin/env python |
| `w28_cm_carry_eval.py` | 236 | W28 | !/usr/bin/env python |
| `w28_dd_shape_calibration.py` | 147 | W28 | !/usr/bin/env python |
| `w28_kg_b4_capital_granularity.py` | 152 | W28 | !/usr/bin/env python |
| `w28_roll_calendar_validation.py` | 155 | W28 | !/usr/bin/env python |
| `w28_seed_ledger.py` | 129 | W28 | !/usr/bin/env python |
| `walkforward_forward.py` | 162 | W18 | adaptive, recency-weighted forward test (Wave 18). |
| `wave20_runner.py` | 171 | W21 | research/wave20_runner.py |
| `wave_final_eval.py` | 159 | — | capstone gate scorecard for the vol-targeted book. |
| `xs_alpha_rework.py` | 233 | W15 | cross-sectional ML rework, models PERSISTED (Wave 15). |
| `xs_diagnose.py` | 119 | W15 | is the XS-ML book novel alpha or just the low-vol factor? (Wave 15) |
| `xs_harvest.py` | 128 | W15 | harvest the positive XS IC into Sharpe (Wave 15, no retrain). |
| `xs_maxfeat.py` | 183 | W16 | MAX the free cross-sectional feature set (Wave 16). |
| `xs_multihorizon.py` | 155 | W15 | multi-horizon ensemble to lift the XS-ML IC (Wave 15). |
| **totaal** | **6598** | | **48 bestanden** |

## Drie bestanden laden een ander bestand per pad

`mn_optimized_exec.py`, `validate_neutral_book.py` en `true_alpha_gates.py`
laden hun afhankelijkheid met
`importlib.util.spec_from_file_location(..., ROOT / "research/...")`. Dat is
een import die geen importstatement is: geen enkele AST-scanner ziet hem, en
`scripts/reachability_map.py` dus ook niet. De padstrings zijn bij de
verhuizing meeverhuisd; wie hier een bestand hernoemt, moet ze met de hand
nalopen.

## Wat deze track NIET is

Geen archief. Bestanden die hun oordeel al hebben gekregen en waarvan de
implementatie is verwijderd, staan in `reports/phase9_removal_register.md`
met het commando dat ze terughaalt. Wat hier staat, is nog uitvoerbaar.
