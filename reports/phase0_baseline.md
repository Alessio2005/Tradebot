# PHASE 0 - REPOSITORY BASELINE (nulmeting)

**Gegenereerd:** 2026-08-22T16:42:09+00:00  
**Generator:** `scripts/phase0_baseline.py` (read-only)  
**Bindend brondocument:** `docs/ARCHITECTUUR_AUDIT_2026-08-22.md`  
**Status:** bewijsmateriaal - deze nulmeting wordt niet overschreven na remediatie.

Feitelijke staat van de repository *voordat* enige Phase 0-wijziging is doorgevoerd.

---

## 1. Totalen

| Metriek | Waarde |
|---|---|
| Gescande Python-bestanden | 377 |
| Totaal LOC | 68,243 |
| Bestanden > 800 LOC (D-6) | 9 |
| Apps > 80 LOC (D-7) | 19 |
| Bestanden met `quant_architect`-import | 5 |
| Hygiene-bevindingen (shadow/config/DAG) | 17 |

---

## 2. LOC per module

| Module | Bestanden | LOC |
|---|---:|---:|
| `apps` | 30 | 7,911 |
| `scripts` | 53 | 7,468 |
| `tests` | 56 | 6,538 |
| `src/tradebot/features` | 16 | 5,120 |
| `src/tradebot/live` | 14 | 4,736 |
| `src/tradebot/train` | 15 | 4,481 |
| `src/tradebot/data` | 24 | 4,019 |
| `src/tradebot/alpha` | 27 | 3,912 |
| `src/tradebot/backtest` | 12 | 3,814 |
| `src/tradebot/risk` | 13 | 3,031 |
| `src/tradebot/labeling` | 6 | 2,459 |
| `src/tradebot/execution` | 6 | 1,640 |
| `miscellaneous` | 6 | 1,451 |
| `src/tradebot/oms` | 7 | 1,294 |
| `src/tradebot/monitoring` | 9 | 1,286 |
| `src/tradebot/tune` | 6 | 1,223 |
| `src/tradebot/cv` | 6 | 1,042 |
| `src/tradebot/compliance` | 6 | 1,020 |
| `src/tradebot/bars` | 7 | 984 |
| `src/tradebot/portfolio` | 8 | 956 |
| `src/tradebot/registry` | 6 | 801 |
| `src/tradebot/schemas` | 12 | 654 |
| `src/tradebot/tca` | 6 | 534 |
| `src/tradebot/volatility` | 7 | 503 |
| `src/tradebot/utils` | 5 | 502 |
| `src/tradebot/selection` | 3 | 302 |
| `src/tradebot/featurestore` | 4 | 275 |
| `src/tradebot` | 5 | 145 |
| `src/tradebot/reporting` | 2 | 142 |

---

## 3. Bestanden > 800 LOC - D-6 (architecture.md R-4)

| Bestand | LOC | Overschrijding |
|---|---:|---:|
| `src/tradebot/labeling/meta.py` | 1,190 | +390 |
| `src/tradebot/train/ensemble.py` | 1,155 | +355 |
| `apps/backtest_portfolio.py` | 1,093 | +293 |
| `src/tradebot/features/regime.py` | 1,059 | +259 |
| `src/tradebot/backtest/evaluation.py` | 1,044 | +244 |
| `src/tradebot/risk/portfolio.py` | 989 | +189 |
| `src/tradebot/tune/objective.py` | 970 | +170 |
| `apps/train_cpcv.py` | 957 | +157 |
| `src/tradebot/live/engine.py` | 903 | +103 |

> D-6 is **P2** en valt buiten de Phase 0-scope. Vastgelegd als bewijs; remediatie
> is doorgeschoven naar `docs/DEFERRED_ISSUES.md`.

---

## 4. Apps > 80 LOC - D-7 (architecture.md R-6)

**19 van de 30 apps overschrijden de limiet.**

| App | LOC | Overschrijding |
|---|---:|---:|
| `apps/backtest_portfolio.py` | 1,093 | +1,013 |
| `apps/train_cpcv.py` | 957 | +877 |
| `apps/live_paper_trader.py` | 642 | +562 |
| `apps/backtest_multi_alpha.py` | 575 | +495 |
| `apps/paper_trade_runner.py` | 523 | +443 |
| `apps/feature_selection.py` | 427 | +347 |
| `apps/tune_hparams.py` | 418 | +338 |
| `apps/stress_and_optimise.py` | 401 | +321 |
| `apps/live_trader.py` | 397 | +317 |
| `apps/generate_mrm_report.py` | 369 | +289 |
| `apps/paper_monitor.py` | 305 | +225 |
| `apps/build_features.py` | 289 | +209 |
| `apps/paper_neutral_trader.py` | 201 | +121 |
| `apps/paper_multi_sleeve.py` | 166 | +86 |
| `apps/doctor.py` | 142 | +62 |
| `apps/alpha_combine.py` | 100 | +20 |
| `apps/make_tearsheet.py` | 91 | +11 |
| `apps/regenerate_baseline.py` | 89 | +9 |
| `apps/featurestore_sync.py` | 86 | +6 |

> D-7 is **P2** en valt buiten de Phase 0-scope. Nieuwe apps die in Phase 1-3 worden
> toegevoegd respecteren de 80-LOC-limiet wel.

---

## 5. `quant_architect`-importeurs - sectie 5.2

Het pakket `quant_architect` bestaat nergens op het filesystem en staat in geen enkele
dependency-declaratie. Elk van onderstaande bestanden draait daardoor permanent in
gedegradeerde modus, zonder melding.

| Bestand | LOC |
|---|---:|
| `src/tradebot/backtest/portfolio.py` | 677 |
| `src/tradebot/features/scaling.py` | 193 |
| `src/tradebot/train/_scalers.py` | 522 |
| `src/tradebot/train/catboost.py` | 634 |
| `src/tradebot/train/ensemble.py` | 1,155 |

### 5.1 Afwijking t.o.v. het auditdocument

Sectie 5.2 van de audit noemt `tune/objective.py` als vijfde importeur. Dat is feitelijk
onjuist: `tune/objective.py` bevat geen `quant_architect`-import. De werkelijke vijfde
importeur is `src/tradebot/features/scaling.py`. Conform sectie 2.2 (documentatie is een
hypothese, geen waarheid) prevaleert deze meting boven de audittekst.

Alle gevraagde symbolen bestaan wel degelijk als interne module:

| Symbool | Interne bron |
|---|---|
| `EntropyGate` | `tradebot.train.schema_guard` |
| `EntropyGateDecision` | `tradebot.train.schema_guard` |
| `FeatureSchemaGuard` | `tradebot.train.schema_guard` |
| `SchemaMismatchError` | `tradebot.train.schema_guard` |
| `LedoitWolfThompsonSampler` | `tradebot.train.thompson` |
| `NetAlphaResult` | `tradebot.train.reward` |
| `NetAlphaReward` | `tradebot.train.reward` |
| `QuantArchitectStack` | `tradebot.train.stack` |
| `SymmetricQuantileScaler` | `tradebot.train.quant_arch` |

De import is dus niet dood maar **verkeerd geadresseerd**: de functionaliteit is ooit uit
`quant_architect.py` geextraheerd naar `train/`, zonder dat de importsites zijn bijgewerkt.
Stap 5 is daarmee een gerichte heradressering, geen herimplementatie.

---

## 6. Shadow trees, misplaatste build-config en ontbrekende DAG-mappen

| Pad | Bevinding |
|---|---|
| `__pycache__/` | shadow tree aanwezig in de root (49 entries) |
| `.pytest_cache/` | shadow tree aanwezig in de root (7 entries) |
| `.ruff_cache/` | shadow tree aanwezig in de root (11 entries) |
| `catboost_info/` | shadow tree aanwezig in de root (6 entries) |
| `outputs/` | shadow tree aanwezig in de root (14 entries) |
| `logs/` | shadow tree aanwezig in de root (71 entries) |
| `miscellaneous/pyproject.toml` | **build-config buiten de projectroot** - `pip install -e .` vanuit de root is onmogelijk |
| `miscellaneous/Makefile` | **build-config buiten de projectroot** - `pip install -e .` vanuit de root is onmogelijk |
| `miscellaneous/dvc.yaml` | **build-config buiten de projectroot** - `pip install -e .` vanuit de root is onmogelijk |
| `miscellaneous/params.yaml` | **build-config buiten de projectroot** - `pip install -e .` vanuit de root is onmogelijk |
| `miscellaneous/.pre-commit-config.yaml` | **build-config buiten de projectroot** - `pip install -e .` vanuit de root is onmogelijk |
| `miscellaneous/.env` | **build-config buiten de projectroot** - `pip install -e .` vanuit de root is onmogelijk |
| `.git/` | **ontbreekt - D-9**: geen versiebeheer; elk MRM-rapport is invalide |
| `.gitignore` | ontbreekt |
| `artefacts/features/` | **ontbreekt - D-8**: DAG-map uit architecture.md |
| `artefacts/models/` | **ontbreekt - D-8**: DAG-map uit architecture.md |
| `artefacts/tracks/` | **ontbreekt - D-8**: DAG-map uit architecture.md |

