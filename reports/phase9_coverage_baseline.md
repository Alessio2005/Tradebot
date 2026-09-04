# Phase 9 - dekkingsnulmeting (stap 3)

> **Gemeten 2026-09-04 op commit `26cfd90`.** Dit is de dekking VOOR de
> opruiming. De nameting staat in `reports/phase9_exit_report.md`.

## Het commando en het gereedschap

```bash
D:/venv/tradebot/Scripts/python.exe -m pytest -m "not slow and not regression" \
    -p no:randomly --cov=src/tradebot --cov-report=json:coverage.json --cov-report=term -q
```

| | versie | herkomst |
|---|---|---|
| interpreter | Python 3.13.0 | `D:/venv/tradebot/Scripts/python.exe` (RUNBOOK par. 0) |
| `pytest` | 9.1.1 | `requirements-dev.lock` |
| `pytest-cov` | 7.1.0 | `requirements-dev.lock` |
| `coverage` | 7.15.4 | `requirements-dev.lock` |

`-p no:randomly` staat er zodat de meting herhaalbaar is; de
volgorde-onafhankelijkheid is een aparte meting (exit-criterium 6).

## De uitkomst tegen de drempel

**Totaal: 55.82 %** - 13391 van 23991 statements over 299 bestanden.

```
FAIL Required test coverage of 70.0% not reached. Total coverage: 55.82%
```

`pyproject.toml` `[tool.coverage.report] fail_under = 70` wordt **niet
gehaald**. Dat is een gemeten feit en het hoort in dit rapport, niet in een
verlaagde drempel: de fase-opdracht sluit dat laatste expliciet uit.

De testselectie draaide **2752 passed, 4 failed, 23 skipped** over 2779
collected. De vier failures zijn dezelfde vier killgates als in de
vingerafdruk en het aantal skips is gelijk aan dat van de volledige suite.
Het verschil met de 2797 collected van de volledige suite is precies de 18
tests die `slow` of `regression` gemarkeerd zijn - dat is de marker, geen
regressie.

## Dekking per pakket, LOC-gewogen

Gesorteerd van laag naar hoog. De kolom `LOC` is de ruwe regeltelling
(`wc -l`), de dekking is het LOC-gewogen gemiddelde van de modules erin.

| Pakket | Modules | LOC | Dekking |
|---|---:|---:|---:|
| `selection` | 3 | 293 | 0.0 % |
| `bars` | 7 | 969 | 14.4 % |
| `tune` | 6 | 1222 | 19.4 % |
| `(top-level)` | 5 | 145 | 27.7 % |
| `reporting` | 5 | 2259 | 28.4 % |
| `labeling` | 7 | 2752 | 31.1 % |
| `live` | 14 | 4896 | 38.5 % |
| `data` | 36 | 5832 | 42.7 % |
| `train` | 16 | 4849 | 44.0 % |
| `backtest` | 14 | 4559 | 54.6 % |
| `portfolio` | 11 | 2661 | 55.8 % |
| `features` | 21 | 7428 | 58.1 % |
| `execution` | 10 | 3240 | 62.9 % |
| `cv` | 6 | 1042 | 64.3 % |
| `monitoring` | 11 | 1797 | 65.3 % |
| `alpha` | 28 | 4568 | 68.9 % |
| `oms` | 7 | 1273 | 69.8 % |
| `validation` | 17 | 6348 | 74.1 % |
| `utils` | 6 | 918 | 76.0 % |
| `featurestore` | 4 | 268 | 78.3 % |
| `volatility` | 9 | 1830 | 81.6 % |
| `registry` | 11 | 2303 | 86.6 % |
| `risk` | 17 | 4671 | 86.8 % |
| `tca` | 6 | 723 | 88.9 % |
| `compliance` | 6 | 1015 | 90.8 % |
| `schemas` | 11 | 1657 | 93.1 % |
| `regime` | 5 | 1969 | 97.6 % |
| **totaal** | **299** | **71487** | **59.9 %** |

De twee getallen verschillen bewust: 55,82 % is de statement-dekking van
`coverage.py`, 59,9 % is dezelfde meting gewogen naar ruwe regels. De eerste
is de poort; de tweede maakt zichtbaar waar het gewicht zit.

## Waar het werk zit - klasse A en B onder de drempel

**110 modules / 32041 LOC** staan in de authoritative of operationele keten met
minder dan 70 % dekking. Daarvan hebben er **22 een dekking van precies
0,0 %**: geen enkele test raakt ze aan.

Klasse A: 75 modules. Klasse B: 35 modules.

Gesorteerd op `LOC x (1 - dekking)` - de volgorde die stap 11 voorschrijft.
De eerste dertig:

| # | Klasse | LOC | Dekking | Module | Bereikt via |
|---:|:---:|---:|---:|---|---|
| 1 | A | 1117 | 8.3 % | `src/tradebot/train/ensemble.py` | apps/build_features.py |
| 2 | A | 1181 | 18.0 % | `src/tradebot/labeling/meta.py` | apps/build_features.py |
| 3 | A | 1056 | 11.9 % | `src/tradebot/features/regime.py` | apps/build_features.py |
| 4 | A | 955 | 10.4 % | `src/tradebot/tune/objective.py` | apps/tune_hparams.py |
| 5 | A | 1054 | 27.2 % | `src/tradebot/backtest/evaluation.py` | apps/run_data_adequacy.py |
| 6 | A | 754 | 0.0 % | `src/tradebot/reporting/phase6_vol_competition.py` | apps/run_vol_competition.py |
| 7 | A | 775 | 9.1 % | `src/tradebot/features/ta.py` | apps/build_features.py |
| 8 | B | 913 | 34.0 % | `src/tradebot/live/engine.py` | apps/live_paper_trader.py |
| 9 | B | 591 | 0.0 % | `src/tradebot/live/model_signal.py` | apps/live_paper_trader.py |
| 10 | A | 578 | 0.0 % | `src/tradebot/reporting/phase6_regime_benchmark.py` | apps/run_regime_benchmark.py |
| 11 | B | 747 | 22.7 % | `src/tradebot/live/feed.py` | apps/live_paper_trader.py |
| 12 | A | 635 | 14.9 % | `src/tradebot/features/_ta_kernels.py` | apps/build_features.py |
| 13 | A | 600 | 11.9 % | `src/tradebot/train/catboost.py` | apps/build_features.py |
| 14 | A | 621 | 17.4 % | `src/tradebot/execution/market_impact.py` | apps/build_features.py |
| 15 | A | 504 | 0.0 % | `src/tradebot/validation/meta_label_campaign.py` | apps/run_meta_labeling.py |
| 16 | A | 488 | 0.0 % | `src/tradebot/backtest/baseline_report.py` | apps/run_data_adequacy.py |
| 17 | A | 498 | 12.0 % | `src/tradebot/train/_scalers.py` | apps/build_features.py |
| 18 | A | 501 | 15.0 % | `src/tradebot/data/crypto.py` | apps/build_features.py |
| 19 | A | 440 | 15.8 % | `src/tradebot/labeling/trend_scanning.py` | apps/build_features.py |
| 20 | A | 411 | 12.0 % | `src/tradebot/data/tradfi_macro.py` | apps/build_features.py |
| 21 | B | 572 | 38.2 % | `src/tradebot/live/feature_updater.py` | apps/live_paper_trader.py |
| 22 | A | 580 | 44.7 % | `src/tradebot/validation/vol_campaign.py` | apps/run_vol_competition.py |
| 23 | A | 330 | 3.9 % | `src/tradebot/bars/_kernels.py` | apps/build_features.py |
| 24 | A | 294 | 0.0 % | `src/tradebot/risk/stress_report.py` | apps/run_data_adequacy.py |
| 25 | A | 408 | 28.3 % | `src/tradebot/labeling/cusum.py` | apps/build_features.py |
| 26 | A | 332 | 14.7 % | `src/tradebot/data/crypto_macro.py` | apps/build_features.py |
| 27 | A | 269 | 0.0 % | `src/tradebot/validation/adequacy_report.py` | apps/run_data_adequacy.py |
| 28 | A | 268 | 0.0 % | `src/tradebot/reporting/phase6_econometrics.py` | apps/run_econometric_diagnostics.py |
| 29 | A | 406 | 34.0 % | `src/tradebot/execution/simulator.py` | apps/build_features.py |
| 30 | A | 314 | 17.9 % | `src/tradebot/execution/spread.py` | apps/tune_hparams.py |

De volledige lijst van 299 modules met klasse, LOC, dekking en verdict staat
in `reports/phase9_inventory.md`.

## Wat dit betekent voor stap 11

De fase-opdracht wijst `train/` (4.849 LOC, 3 testbestanden), `reporting/`
(2.259, 1), `portfolio/` (2.661, 4), `oms/` (1.273, 4) en `selection/` (293,
**0**) aan als de kop van de lijst. De nagemeten dekking bevestigt dat beeld
en scherpt het aan:

* `selection/` - **0,0 %**, drie modules, geen enkele test. Bevestigd.
* `bars/` - **14,4 %**, staat niet in de opdrachtlijst maar hoort er wel bij.
* `tune/` - **19,4 %**, met `tune/objective.py` (955 LOC, 10,4 %).
* `reporting/` - **28,4 %**, waarvan twee modules op exact 0,0 %:
  `phase6_vol_competition.py` (754 LOC) en `phase6_regime_benchmark.py` (578).
* `oms/` - **69,8 %**, net onder de drempel; de goedkoopste winst van de lijst.
* `portfolio/` - **55,8 %**, maar 1.018 van die 2.661 LOC zijn
  `legacy_sizing.py` + `covariance.py`, die per DI-10 ONGEWIJZIGD moeten
  blijven. Dekking bijschrijven mag; het gedrag aanraken niet.

**Dit is meer werk dan een enkele fase.** 110 modules over 32.041 LOC naar
70 % brengen met gedragstests - geen vormtests, zoals `PROJECT_STATE.md`
waarschuwt - is geen bijvangst van een opruiming. Stap 11 werkt de lijst van
boven naar beneden af en registreert voor elke module die de drempel niet
haalt een ratchet met datum en eigenaar in `docs/CODE_REGISTER.md`. Dat is
de tweede van de twee toegestane uitkomsten; een derde is er niet.
