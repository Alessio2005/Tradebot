# PHASE 2 — FEATURE VALIDATION REPORT

> **Laag:** L3 — Feature Store & Causal Pipeline
> **Bindend brondocument:** `ARCHITECTUUR_AUDIT_2026-08-22.md` — secties 7, 7.1, 7.2, 8, 8.1, 8.2, 10, 10.1, 10.2, 19, 24, 26
> **Voorwaarde:** Phase 0 & Phase 1 afgerond; Data Register gecertificeerd op `git_sha 661f351`.

**Gegenereerd:** 2026-08-23
**`git_sha` van de feature-definities:** `4600bb7`
**Hash-receptuur:** `phase2.v1` (`FEATURE_HASH_VERSION`)
**Data Register:** `artefacts/governance/data_hashes.json` — 18 gecertificeerde reeksen
**Configuratie:** `conf/features/default.yaml`, gevalideerd tegen `schemas/config.py::FeatureConfig`

---

## 0. Scope-afwijking, expliciet vermeld

Het auditdocument (sectie 23) noemt als Phase 2 de **Research & Falsification
Foundation** (L11/L12: DSR-, SPA- en lookahead-gates in CI). De opdracht die
deze fase heeft aangestuurd, beschrijft Phase 2 als de **Deterministic Feature
Engine** (L3). Dit rapport dekt uitsluitend het laatste. De L11/L12-gates uit
`Prompts-fases/fase_2_research_falsification.md` zijn **niet** gebouwd en
blijven openstaan; de causaliteitssuite die hier is opgeleverd is een
bouwsteen daarvoor, geen vervanging ervan.

---

## 1. Wat er is gebouwd

| Deliverable | Artefact | Status |
|---|---|---|
| 1 | `src/tradebot/features/base.py` — `BaseFeature`, `FeaturePipeline`, `CertifiedFrame`, `DataRegister` | opgeleverd |
| 2 | `src/tradebot/features/momentum.py` | opgeleverd |
| 3 | `src/tradebot/features/volatility.py` — **DI-2 gesloten** | opgeleverd |
| 4 | `src/tradebot/features/microstructure.py` — L3-sectie op funding & open interest | opgeleverd |
| 5 | `src/tradebot/features/registry.py` — `FeatureRegistry`, `feature_hash` | opgeleverd |
| 6 | `tests/lookahead/test_feature_causality.py` — 310 tests | opgeleverd |
| 7 | `tests/unit/test_feature_determinism.py` — 21 tests | opgeleverd |
| 8 | `conf/features/default.yaml` + `FeatureConfig` | opgeleverd |
| 9 | `scripts/build_feature_store.py` — 70 LOC (R-6 ≤ 80) | opgeleverd |
| 10 | dit rapport | opgeleverd |

**15 geregistreerde features** over 6 symbolen, opgebouwd uit **18
gecertificeerde PIT-reeksen** (6× ohlcv 1d, 6× funding 8h, 6× open_interest 1d).

---

## 2. Het contract dat elke feature moet halen

`BaseFeature.transform` is geen wrapper maar de poort. Hij weigert:

| Controle | Wat er wordt afgedwongen | Exception |
|---|---|---|
| Provenance | Input is een `CertifiedFrame` met een `data_hash` per bronreeks, geverifieerd tegen het Phase 1-register | `DataContractError` |
| Tijdstandaard | UTC-aware, oplopende, unieke index; `event_ts_ns` en `asof_ts_ns` int64; `asof ≥ event` | `CausalityViolationError` |
| Outputschema | Exact de gedeclareerde `output_columns`, in volgorde | `DataContractError` |
| Dtype | Elke kolom `float64`; geen impliciete conversie | `DataContractError` |
| Eindigheid | Geen ±inf | `DataContractError` |
| **Burn-in** | **Elke waarde binnen `burn_in_period` is NaN. Eén eindige waarde = contractbreuk** | `CausalityViolationError` |
| NaN daarna | NaN na de burn-in uitsluitend waar de INPUT zelf ontbreekt | `DataContractError` |

De burn-in-controle is de directe tegenhanger van DI-2: een opgevulde
opstartfase kán niet meer stilzwijgend passeren, ongeacht waarmee hij is
opgevuld.

### 2.1 Waarom de output op `asof_ts` staat en niet op `event_ts`

De feature-matrix is geïndexeerd op het **beschikbaarheidsmoment**. Een daily
OHLCV-bar opent op `event_ts` en is pas compleet op `asof_ts` (de volgende
middernacht UTC, per de Phase 1-semantiek in `docs/DATA_REGISTER.md` §5). Een
feature die uit die bar volgt, is dus pas op `asof_ts` bruikbaar. Wie de matrix
op `event_ts` indexeert, geeft een consument per constructie 24 uur voorsprong —
zonder dat er ergens een `shift(-1)` in de code staat.

---

## 3. Features, formules, causaliteitsbewijs en hashes

`feature_hash = H(input_data_hashes + feature_class_name + hyperparameter_dict + git_sha)`,
plus `output_columns`, `burn_in_period` en de modulenaam, en voorafgegaan door
`FEATURE_HASH_VERSION`. De hashes hieronder zijn berekend op de
**BTCUSDT-provenance** bij `git_sha 4600bb7`; voor een ander symbool verschilt
alleen de `input_data_hashes`-component en dus de hash.

### 3.1 Volatiliteit (L3, `features/volatility.py`)

| Kolom | Formule | Burn-in | Causaliteitsbewijs | `feature_hash` |
|---|---|---:|---|---|
| `vol_realized_5` | `std(r_{t-4..t}) · √365`, `min_periods = window` | 5 | `rolling` is achterwaarts; `r_0` is NaN, dus het eerste volledige venster sluit op positie 5 | `3c88519bcaff400da1a908331b61fe34` |
| `vol_realized_20` | idem, `w = 20` | 20 | idem | `d2bc462e424a1a075b29d62f8786a481` |
| `vol_realized_60` | idem, `w = 60` | 60 | idem | `cc35ba98964a91ae8fb4386516047bbf` |
| `vol_expanding` | `expanding_std(r_0..r_t) · √365`, `min_periods = 20` | 20 | groeit uitsluitend met bekende historie; **de causale vervanger van DI-2** | `8f1cfa5e66d4925bc70c758d0eb485a9` |
| `vol_ewma` | `σ²_t = λσ²_{t-1} + (1−λ)r²_t`, `λ = 0.94` | 20 | voorwaartse recursie; seed = expanding gemiddelde van `r²` over uitsluitend de eerste 20 returns | `56a27bcbcaad7014006e7ae34539cdf6` |
| `vol_parkinson_20` | `√(mean(ln(H/L)²)/(4 ln 2)) · √365` | 19 | bar-lokale input plus achterwaarts venster; geen return-differentie nodig | `5581d4a7ad8cc55d2027ec68dc28170e` |

**De EWMA-seed is het gevoelige punt.** Een EWMA moet ergens beginnen, en de
gebruikelijke fout is hem te seeden met de variantie over de volledige sample —
dezelfde fout als DI-2, alleen beter verstopt. Hier is de seed het expanding
gemiddelde van `r²` over uitsluitend de eerste `ewma_burn_in_bars` returns, en
vóór dat punt wordt niets vrijgegeven.

**Garman-Klass ontbreekt bewust.** De GK-schatter kan per bar een negatieve
variantie opleveren. Die realisaties vragen een expliciet gedocumenteerde
behandeling (clippen op nul, of overstappen op het variantiedomein), en dat is
een besluit van de volatiliteitsengine (L2, Phase 3/6) — niet van de
feature-laag. Stilzwijgend clippen zou in risk parity een oneindig gewicht
kunnen opleveren; dat risico wordt niet geïntroduceerd om een lijstje compleet
te maken. Parkinson kent het probleem niet: `ln(H/L)²` is per constructie
niet-negatief.

### 3.2 Momentum (L3, `features/momentum.py`)

| Kolom | Formule | Burn-in | Causaliteitsbewijs | `feature_hash` |
|---|---|---:|---|---|
| `mom_logret_5` | `ln P_{t-1} − ln P_{t-6}` (`skip = 1`) | 6 | uitsluitend `shift(+k)`; een negatieve skip wordt door de constructor geweigerd | `63473195ae7a54566db0b0766796b5ef` |
| `mom_logret_20` | `ln P_{t-1} − ln P_{t-21}` | 21 | idem | `9f520f2fb55d84c8a70418f02f8386ae` |
| `mom_logret_60` | `ln P_{t-1} − ln P_{t-61}` | 61 | idem | `dbc38b75165caff5330d4d0ae86838b2` |
| `mom_ewma_spread` | `EWMA_12(ln P) − EWMA_26(ln P)`, `adjust=False` | 25 | zuivere recursie `y_t = (1−a)y_{t-1} + a x_t`; `min_periods = slow_span` maskeert de opstartfase met NaN | `bef7e4d3d3ea7463523596bff07eab7d` |

De `skip_bars = 1` is geen kosmetiek: het rendement van de laatste bar is op `t`
bekend, maar de bekende korte-termijn reversal in crypto maakt hem als
trendsignaal misleidend. De skip staat in `conf/` en gaat mee in de hash.

### 3.3 Microstructuur & carry (L3, `features/microstructure.py`)

| Kolom | Formule | Burn-in | Causaliteitsbewijs | `feature_hash` |
|---|---|---:|---|---|
| `micro_funding_mean_3` | `mean(f_{t-2..t})` | 2 | achterwaarts venster op de via `asof_join` gekoppelde rate | `727f7bf92fca01a1fe1e70a8e23abd44` |
| `micro_funding_mean_21` | `mean(f_{t-20..t})` | 20 | idem | `faa0b675e604794be055159fb4829b69` |
| `micro_funding_zscore` | `(f_t − expanding_mean)/expanding_std`, `min_periods = 60` | 59 | **expanding**, nooit sample-breed; een nul-variantie crasht in plaats van ±inf op te leveren | `e4aedacd2cdc110120827b0eade0ed42` |
| `micro_oi_logchg_1` | `ln OI_t − ln OI_{t-1}` | 1 | uitsluitend `shift(+k)` | `0a403db740859f36e99b4d0939f2325c` |
| `micro_oi_logchg_7` | `ln OI_t − ln OI_{t-7}` | 7 | idem | `2d3cb15474e4d195d79a62be450731ac` |

**De z-score is bewust expanding.** Een z-score op basis van het gemiddelde en
de spreiding over de volledige reeks is dezelfde klasse fout als DI-2 — en in de
praktijk zichtbaarder, omdat hij per constructie rond nul gecentreerd is over
precies het venster dat wordt getoetst.

### 3.4 Multi-granulaire koppeling (exit criterium 4)

8h funding en 1d open interest worden op het 1d bar-raster geprojecteerd,
uitsluitend via `utils.time.asof_join(direction="backward")` met een verplichte
tolerance uit `conf/features/default.yaml` (`funding_tolerance_hours: 8`,
`open_interest_tolerance_hours: 24`).

**Het koppelmoment is de `asof_ts` van de OHLCV-bar, niet zijn `event_ts`.** Dat
is het moment waarop de feature daadwerkelijk beschikbaar is, en het is tevens
de index van de feature-matrix. Koppelen op `event_ts` zou eveneens causaal zijn
maar gooit tot een volledige bar aan bekende funding-informatie weg; koppelen op
een later moment zou een lek zijn. Beide grenzen worden getoetst
(`TestMultiGranularJoinIsBackwardOnly`), inclusief de controle dat geen enkele
gekoppelde rate vóór zijn eerste settlement aan een bar hangt.

---

## 4. Artefacten in de feature store

`data/feature_store/{matrix_hash}.parquet` + `{matrix_hash}.json` (manifest).
`content_hash` is de inhoudshash over de geschreven rijen; `matrix_hash` is de
identiteit van de definitie inclusief `git_sha`.

| Symbool | Bars in | Bars uit (burn-in afgekapt) | Periode | `matrix_hash` | `content_hash` |
|---|---:|---:|---|---|---|
| BTCUSDT | 2.342 | 2.281 | 2020-03-25 → 2026-08-22 | `c2ab73ffac337fe5424ab3bde8f018c1` | `0a12d6f12e7dd9786473d8edf7a29d42` |
| ETHUSDT | 1.987 | 1.926 | 2021-03-15 → 2026-08-22 | `cb114e4f5628c65de18ef976315fadf6` | `497e87a7688c317a91ce4c7f9e87bfb2` |
| SOLUSDT | 1.773 | 1.712 | 2021-10-15 → 2026-08-22 | `59dc0473e7473e7e1c29e7268a2a4b62` | `fabcdfa6305600b39598db4806b35117` |
| AVAXUSDT | 1.803 | 1.742 | 2021-09-15 → 2026-08-22 | `9cd747914cbd6647baf512ee8ad8465c` | `13a7350a32c8a18f0b0e2e47b0b9d601` |
| LINKUSDT | 2.132 | 2.071 | 2020-10-21 → 2026-08-22 | `b3a0d25b410f2ecb5d1e5b5afcbda38b` | `ac2cd7a71bc750111ab481efabca1d5d` |
| DOTUSDT | 1.983 | 1.922 | 2021-03-19 → 2026-08-22 | `343115cc3e4c08166acf1dd0fbd9849e` | `2ad16f393cc2f17f45c349d273f4e255` |

Gezamenlijke burn-in: **61 bars** (het maximum over alle features —
`mom_logret_60` met `window + skip`). De pipeline hanteert bewust het maximum,
zodat een matrix waarvan de burn-in is afgekapt in *elke* kolom bruikbaar is en
niet alleen in de snelste.

### 4.1 Resterende NaN's ná het afkappen van de burn-in

De regel luidt: burn-in afgekapt **of** expliciet gedocumenteerd. Na het
afkappen resteert er op één symbool NaN, en de oorzaak is geen feature maar
ontbrekende brondata:

| Symbool | Kolom | NaN | Oorzaak |
|---|---|---:|---|
| BTCUSDT | `micro_oi_logchg_1` | 72 | open interest begint op 2020-08-05, OHLCV op 2020-03-25 |
| BTCUSDT | `micro_oi_logchg_7` | 78 | idem, plus het 7-bars venster |

Alle overige symbolen: **0 NaN** na het afkappen. Deze telling staat
machineleesbaar in elk manifest (`nan_per_column`), zodat de regel toetsbaar is
en niet alleen beschreven. Het gat wordt **niet** gevuld: dat BTC-perp open
interest pas vanaf augustus 2020 wordt gerapporteerd is een feit over de bron,
geen ontbrekende meting om te interpoleren.

---

## 5. DI-2 — gesloten

**Locatie:** `src/tradebot/risk/hmm_regime.py::HMMRegimeDetector._features`
**Sluitende commit:** `d1e2c25`

```python
# was:
vol = returns.rolling(5, min_periods=2).std().fillna(returns.std())
```

`returns.std()` is de standaarddeviatie over de **volledige sample**. Op bar 1
van een reeks van 2.342 bars kreeg de vol-feature daarmee een waarde die pas in
2026 bekend kon zijn. Het lek is bovendien **systematisch en niet willekeurig**:
de opvulwaarde is per constructie de gemiddelde volatiliteit van het hele
onderzoeksvenster, waardoor een rustige beginperiode structureel te hoog en een
crisisperiode structureel te laag wordt geschat. Elk risicomodel dat erop steunt
is gekalibreerd op informatie die het niet had.

De vervanging gebruikt uitsluitend `[0, t]`: rolling zodra er een volledig
venster is, daarvóór `causal_expanding_std`, en daarvóór NaN — die rijen vallen
weg in de bestaande finite-filter in plaats van te worden opgevuld.

**Bewijs** (`TestDI2IsClosed`, 13 tests):

1. Het oude idioom is aantoonbaar **niet** truncatie-invariant, op elk van de
   zes gecertificeerde symbolen — de toets meet het lek daadwerkelijk.
2. `causal_expanding_std` is bit-exact invariant op dezelfde zes symbolen.
3. Een AST-scan bewijst dat het idioom niet meer als **code** in het bestand
   voorkomt (het staat nog wel in de docstring, als documentatie van de gesloten
   bevinding).

**Scope-grens.** DI-2 was oorspronkelijk aan Phase 6 toegewezen omdat Phase 0
geen modelgedrag wijzigde. De Phase 2-opdracht trekt die toewijzing expliciet
naar voren. **DI-1** — `GaussianHMM.predict` levert het Viterbi-pad, dat
*smoothed* is en dus sectie 10.2 schendt — staat hier volledig los van en blijft
aan **Phase 6** toegewezen. Deze fase raakt de feature, niet het decoderingspad.

---

## 6. Exit criteria

| # | Criterium | Bewijs | Status |
|---|---|---|---|
| 1 | 100% causaliteitsgarantie: alle geregistreerde features slagen voor de truncatie-invariantietest op echte crypto-data | 310 tests in `tests/lookahead/test_feature_causality.py`; 15 features × 6 symbolen × 3 snijpunten, `diff == 0.0` inclusief NaN-patroon | **GEHAALD** |
| 2 | DI-2 volledig opgelost; geen sample-brede statistiek of forward-looking functie | §5; AST-scan + burn-in-guard in `BaseFeature`; geen `center=`, geen `shift(-` in `features/{base,volatility,momentum}.py` of de L3-sectie van `microstructure.py` | **GEHAALD** |
| 3 | Deterministische pipeline: twee runs → bit-exact identieke Parquet en `feature_hash` | `TestBuildScriptIsByteDeterministic` draait het echte script tweemaal en vergelijkt de bytes; handmatig geverifieerd met `cmp`/`md5sum` | **GEHAALD** |
| 4 | Geen data leakage bij joins; uitsluitend `asof_join(direction="backward")` met tolerance | §3.4; `TestMultiGranularJoinIsBackwardOnly`; `asof_join` is de enige merge in de L3-laag en eist een verplichte tolerance | **GEHAALD** |
| 5 | Geen hardcoded feature-parameters; alles uit `conf/features/default.yaml` | `check_hardcoded_params.py --strict` → 0 treffers in de nieuwe modules; ze staan expliciet in de `GOVERNED`-set (budget 0). `test_changed_config_changes_the_matrix_hash` bewijst het van de andere kant | **GEHAALD** |
| 6 | Testsuite breidt uit zonder regressie | 728 → 1.064 tests (+336). Dezelfde 6 pre-existente failures vóór en ná; nul nieuwe | **GEHAALD** |

### 6.1 Testtelling

| | Baseline (`24999fe`) | Na Phase 2 (`4600bb7`) |
|---|---:|---:|
| Verzameld | 728 | 1.064 |
| Gefaald | 6 | 6 |
| Overgeslagen | 21 | 23 |

De 6 failures zijn **pre-existent en ongewijzigd**: 4× `test_expansion_killgates`
(kill-gates op `cm_carry`/`cm_tsmom`) en 2× `test_hypothesis_kernels`
(`har_rv_forecast_non_negative`, `evt_gpd_var_less_than_cf_at_extreme`). Zij
vallen buiten de scope van deze fase en zijn niet aangeraakt.

De 2 extra skips zijn gedocumenteerd: `test_first_value_lands_exactly_on_the_boundary`
slaat de twee open-interest-features op BTCUSDT over, omdat daar de bron later
begint dan het bar-raster en de burn-in-grens dus niet scherp te meten is.

### 6.2 Hygiëne

| Scanner | Uitkomst |
|---|---|
| `scripts/check_hardcoded_params.py --strict` | exit 0 — geen bestand overschrijdt zijn budget |
| `scripts/audit_fallbacks.py --strict` | exit 0 — 0 blokkerende bevindingen; nul `try/except` in de nieuwe modules |
| `ruff check` | clean op alle Phase 2-bestanden |
| `mypy` | clean op `features/{base,volatility,momentum,registry}.py` en op de L3-sectie van `microstructure.py`. De 6 meldingen in `risk/hmm_regime.py` zijn pre-existent (identiek aantal vóór en ná de DI-2-fix) |

---

## 7. Bewijs door falen

Groen zonder bewezen rood is waardeloos. De suite bevat daarom drie negatieve
controles die **moeten** falen:

| Controle | Wat het lek is | Wat het vangt |
|---|---|---|
| `_FutureLeakingFeature` | `close.shift(-1)` | de invariantietoets zelf, op elk van de drie snijpunten |
| `_SampleWideScalingFeature` | `r / r.std()` — normaliseert met de spreiding over de volledige sample | de subtiele variant: geen `shift(-1)` te bekennen, elke rij oogt causaal, en toch verandert elke waarde zodra er data achteraan wordt geplakt |
| `_FilledBurnIn` | exact het DI-2-idioom | de burn-in-guard in `BaseFeature`, met `CausalityViolationError` |

Zonder deze drie zou de suite groen kunnen zijn omdat hij niets meet.

---

## 8. Wat deze fase NIET heeft gedaan

Eerlijk afgebakend, zodat een latere fase niet uitgaat van dekking die er niet is:

1. **Geen L11/L12-gates.** Zie §0. De DSR-, SPA- en promotiegates uit het
   auditdocument bestaan nog niet; deze suite kan er wel op worden aangesloten.
2. **Geen econometrische toetsingsketen (sectie 8.2).** ADF/KPSS, CUSUM en
   Ljung-Box/ARCH draaien niet automatisch op elke reeks die de pipeline
   binnenkomt. `features/stationarity_gate.py` bestaat als legacy-module maar is
   niet aan het L3-contract gekoppeld. **Aanbevolen voor Phase 3.**
3. **Geen Garman-Klass.** Zie §3.1; expliciet uitgesteld met reden.
4. **Geen fractionele differentiëring (sectie 8.1).** `features/fracdiff.py`
   bestaat als legacy-module, is niet naar `BaseFeature` gemigreerd en heeft dus
   geen causaliteitsbewijs. **Aanbevolen voor Phase 6**, samen met de
   Level 2+-modellen die er de consument van zijn.
5. **Geen cross-sectionele features.** Alles is per symbool. Een
   cross-sectionele rank is causaal zolang hij per tijdstip over de assets
   rangschikt, maar hij vereist een panel-variant van het contract. **Phase 3**
   (sectie 11: Cross-Sectional Momentum) heeft hem nodig.
6. **De legacy feature-modules zijn onaangeraakt.** `ta.py`, `regime.py`,
   `orthogonalize.py`, `cfi.py`, `scaling.py` en de order-flow-helft van
   `microstructure.py` draaien nog niet onder het L3-contract en dragen geen
   causaliteitsbewijs. Zij vallen onder DI-12 (Phase 3).
7. **DVC-tracking van de feature store.** `data/feature_store/` staat in
   `.gitignore` en is nog niet als DVC-stage geregistreerd, anders dan
   `data/pit_store`. Het artefact is volledig herbouwbaar uit de PIT-store plus
   `conf/`, en de `matrix_hash` maakt de herbouw verifieerbaar.
8. **Survivorship bias staat nog open (DI-15).** Het universum bestaat uit zes
   ex-post gekozen overlevers. Dat raakt de features niet — zij zijn causaal per
   symbool — maar wel elke claim die er later op wordt gebouwd. Bij elke
   baseline-claim in Phase 3 te vermelden.

---

## 9. Reproductie

```bash
git checkout 4600bb7
python scripts/build_feature_store.py
python -m pytest tests/lookahead/test_feature_causality.py tests/unit/test_feature_determinism.py -q
```

De `matrix_hash` in §4 wordt exact gereproduceerd op deze commit. Op een latere
commit verschuiven alle `matrix_hash`- en `feature_hash`-waarden, omdat `git_sha`
onderdeel is van de receptuur en `current_git_sha()` HEAD leest — ook bij een
docs-only commit. Dat is de conservatieve kant van de fout: een hash mag nooit
beweren dat twee artefacten gelijk zijn terwijl de codetoestand verschilt. De
`content_hash` van elk Parquet-bestand is daarentegen **code-versie-onafhankelijk**
en blijft gelijk zolang de berekening en de brondata gelijk blijven — beide
kolommen staan daarom in de tabel.

---

## 10. Aanbeveling

Phase 2 (L3) is afgerond: alle zes exit criteria zijn gehaald, met bewijs.
De feature-laag is klaar voor Phase 3-consumptie.

**Voordat Phase 3 begint**, twee punten die daar direct aan geraakt worden:

* De **cross-sectionele transform** (punt 5 hierboven) is een harde
  voorwaarde voor Cross-Sectional Momentum en moet onder hetzelfde
  `BaseFeature`-contract worden gebouwd, niet ernaast.
* De **stationariteitsketen** (punt 2) hoort vóór de eerste alpha-claim te
  draaien, niet erna.
