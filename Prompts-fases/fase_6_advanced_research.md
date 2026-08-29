# MASTER-PROMPT: PHASE 6 — ADVANCED RESEARCH TRACKS (GARCH · REGIMES · META-LABELING)

> **Fase:** 6 van 7 · **Prioriteit:** P2
> **Bindend brondocument:** `docs/ARCHITECTUUR_AUDIT_2026-08-22.md` — §6, §8, §8.1, §8.2, §9, §9.1, §9.2, §9.3, §10, §10.1, §10.2, §12, §12.1, §13.1, §17, §22, §23 (Phase 6), §24, §25, §26, §27
> **Voorwaarde:** Phase 0 t/m 5 afgerond. Phase 5 sloot af als **CONDITIONAL PASS** op `84273ca`; de twee gedeeltelijke criteria zijn dataeigenschappen, geen openstaand werk. Zie §0 hieronder — die voorwaarden zijn bindend voor alles wat je in deze fase claimt.
> **Startpunt:** `84273ca` · **Risk `config_hash`:** `1b60cb664fbf9a2a` · **`M` bij aanvang:** `2.715`

---

## 0. ERFENIS UIT PHASE 5 — WAT JE MOET WETEN VOORDAT JE ÉÉN MODEL FIT

Deze sectie is geen achtergrondinformatie. Het is de verzameling randvoorwaarden waaronder elk oordeel in deze fase tot stand komt. Wie ze negeert, produceert een statistisch geldig antwoord op een verkeerd gestelde vraag.

### 0.1 Er is precies één engine, en die is soeverein bedraad

`src/tradebot/backtest/engine.py::EventDrivenEngine` is de enige instantie die mag verklaren wat een model zou hebben opgeleverd. De keten is:

```
MarketEvent -> FillEvent (orders van eerdere bars) -> FundingEvent
            -> SOVEREIGN RISK DECISION -> Order -> (volgende bar) Fill
            -> Accounting
```

Consequenties voor deze fase, niet onderhandelbaar:

| Regel | Mechanisme dat hem afdwingt |
|---|---|
| Elk Phase 6-model levert een **signaal**, nooit een order | `OrderRouter.build_orders()` eist `RiskDecision` als eerste argument |
| Elk model draait onder `config_hash = 1b60cb664fbf9a2a` | `build_orders` verwerpt een besluit met afwijkende hash; `EventDrivenEngine.__init__` weigert bij hash-mismatch |
| Geen model mag zijn eigen limiet meebrengen | statische repository-audit in `tests/integration/test_sovereign_wiring.py` |
| Geen vectorized resultaat als promotiebewijs | `NOT_ADMISSIBLE_AS_PROMOTION_EVIDENCE`; 25 tests in `test_vectorized_not_admissible.py` |
| Accounting sluit na elke bar of crasht | twee invarianten in `Ledger.mark()` |

**Breid de statische audit uit met de Phase 6-modules** (`volatility/`, `regime/`, `labeling/`, `train/`, `portfolio/hrp.py`). Een nieuw onderzoeksspoor is precies het moment waarop iemand een drempel lokaal hardcodeert.

### 0.2 De latency-conventie is `shift(2)`, niet `shift(1)`

`reports/phase5_engine_diff.md` bewees met per-bar RMSE over drie seeds dat de authoritative engine `shift(2)` matcht (1,03–1,26 bp) tegen 17–22 bp voor lag 1 en 3. `baseline_runner.py` hanteerde `shift(1)`: *"beslis op de close van `t`, voer uit op DIE close."*

> **De Phase 3-baseline nam aan dat je kunt handelen op de close waarop je besluit.** Elk Phase 6-model dat op die aanname is gebouwd, meet iets dat niet bestaat.

Elke feature, elke regime-classificatie, elke label en elke modelfit in deze fase respecteert `latency_bars = 1` bovenop de beslisbar. Een GARCH-forecast voor `t+1` die de close van `t` gebruikt is legitiem; een positie die daarop op de close van `t` wordt ingenomen is dat niet.

### 0.3 Kosten zijn gemodelleerd, maar twee componenten zijn gelabeld

| Component | Status | Gevolg voor deze fase |
|---|---|---|
| Fees (Bybit VIP-0) | gemodelleerd | dominante kostenpost — zie 0.4 |
| Funding, latency, partial fills, rejects | gemodelleerd | volledig bruikbaar |
| Marktimpact | **`IMPACT_UNCALIBRATED`**, bovengrens `eta = 2,9919`, `kappa_d = 0,6720` | bindt niet op dit boek (<1 bp bij participatie ~1e-7), maar het **label reist mee** |
| Spread | **`SPREAD_ASSUMED`**, 1,0 bp | Corwin-Schultz verworpen op deze data (32,6–65,9 bp, 33–38 % negatief); zie AD-3 |
| Orderboekdiepte, limit-queues, cancellations | **ONTBREKEN** | geen L2-data in de gecertificeerde store |

**Elke promotieclaim in deze fase draagt beide labels.** Een regel als *"HMM verbetert de OOS Sharpe met 0,08 na kosten"* is onvolledig; correct is *"…na kosten onder `IMPACT_UNCALIBRATED` en `SPREAD_ASSUMED` (1,0 bp)"*. Voeg aan elk rapport een sensitiviteitsanalyse toe: **bij welke aangenomen half-spread verdwijnt de gemeten verbetering?** Verdwijnt zij al bij 3 bp, dan is de promotie een spread-aanname, geen modelresultaat.

### 0.4 Fees domineren, en dat maakt turnover de scherpste vijand van complexiteit

Over 1.743 bars, L3:

| Track | Fees | Spread | Impact | Orders |
|---|---:|---:|---:|---:|
| `long_only_equal_weight` | € 49,89 | € 9,07 | € 36,51 | 985 |
| `xs_momentum_risk_parity` | € 2.106,77 | € 383,05 | € 1.209,08 | 9.655 |

Regimewisselingen, meta-label-filters en HRP-herweging zijn alle drie **turnover-generatoren**. Rapporteer voor elk gepromoveerd model expliciet de turnover-delta ten opzichte van zijn baseline, en de bruto-verbetering die nodig was om de extra fees terug te verdienen. Een model dat de Sharpe bruto met 0,05 verbetert en de turnover verdubbelt, is gefalsificeerd — niet onbewezen.

### 0.5 De baseline halteert, en dat vernielt naïeve conditionering

`long_only_equal_weight` **halteert definitief op 2022-05-10** (LUNA) bij 8,31 % drawdown op de High-Water Mark. De drawdown-breaker is een eenrichtingsdeur.

> Van de 1.743 bars handelt deze track er ~130. Elk regime-conditioneringsexperiment op deze track meet de eerste zes maanden van het venster en daarna niets.

**Verplicht in de pre-registratie van H2 en H3:** benoem vóór de eerste run welke track het primaire signaal levert, en verantwoord de keuze tegen dit haltprobleem. `xs_momentum_risk_parity` handelt het volledige venster (9.655 orders) en is daarmee de enige track met bruikbare OOS-massa. Rapporteer voor elk experiment het **aantal effectief handelende bars**, niet het vensterlengte.

### 0.6 Het universum is klein, en dat is een resultaat, geen excuus

6 gecertificeerde `ohlcv/1d`-reeksen · 1.743 daily bars · 2021-11-15 t/m 2026-08-23 · één cluster.

Dit is de belangrijkste beperking van de hele fase. Zie §3 (Data Adequacy Gate). Voor een 2-state Gaussian HMM, walk-forward met 6 folds en een embargo, houd je per fold enkele honderden observaties over waarin één toestand mogelijk tientallen bars beslaat. **De meest waarschijnlijke uitkomst van H2 is `UNPROVEN — insufficient data`, en dat is een geldig, publiceerbaar eindresultaat** (§6-doctrine). Wat níet geldig is: het label `FALSIFIED` toekennen aan een model dat simpelweg te weinig data had, of `PROMOTED` op basis van een verschil dat binnen de ruis van 1.743 bars valt.

### 0.7 De clusterlimiet is vacuous — en dat raakt HRP direct

Met één cluster geldt `effective_relative_cap(1, 0.60) = 1.0`; `cluster_cap` bindt **nooit**. De labels dragen geen informatie: scheiding binnen-vs-tussen +0,0113 met 95 %-BI [−0,0131, +0,0380], teken wisselt per jaar, zes verschillende partities in zes jaar. Correctie bracht de boekvolatiliteit van 3,26 % naar 7,50 % (93,8 % attainment).

**HRP is een clustering-gebaseerde allocator op een universum waarvan bewezen is dat het geen clusterstructuur heeft.** Registreer dat vóór de run als expliciete verwachting. Vindt HRP niettemin een verbetering boven Inverse Volatility, wees dan extra sceptisch: op zes assets met ρ ≈ 0,73 is het verschil tussen HRP en Inverse Vol enkele basispunten aan gewichten en de kans op overfitting op de recursieve bisectie-ordening is aanzienlijk.

### 0.8 Vier stukken openstaand werk die Phase 5 expliciet naar deze fase doorschuift

`reports/phase5_exit_report.md` §17.4:

1. **Hypothesis-profiel met `derandomize=True`** in `tests/conftest.py` — sluit de onverklaarde zevende failure uit §11.3 definitief. Phase 5 deed dit bewust niet, omdat het testgedrag buiten de fase-scope wijzigt en de faalbasislijn vergelijkbaar moest blijven met `b206895`.
2. **De twee property-failures repareren.** Dit zijn **echte bugs**, geen flakiness, en beide liggen in het hart van deze fase:
   - `test_har_rv_forecast_non_negative` — `har_rv_forecast` kan **negatief** worden. HAR-RV is Level 3 in de vol-hiërarchie en jouw deliverable 2. Een negatieve variantieforecast maakt QLIKE ongedefinieerd (`ln` van een negatief getal), dus deze bug moet **vóór Stap 6** gesloten zijn.
   - `test_evt_gpd_var_less_than_cf_at_extreme` — `evt_gpd_var` levert een niet-eindige waarde op degenerate invoer (`risk/var.py`).
3. **DVC-stage voor de authoritative engine.** Stage 4 en 4.5 zijn verwijderd; de opvolger draait via `apps/run_phase5_baseline.py` en heeft nog geen stage. **De ML-track kan niet reproduceerbaar zijn zonder deze stage** — bouw hem vóór Stap 10.
4. **Orderboekdata insteren** — sluit exit-criterium 7 en Open Question 1. Valt buiten de Phase 6-scope tenzij de data er is; noteer de status ongewijzigd.

De vier killgate-failures (`KG-B1 in-sample-cm_carry`, `KG-B2 residual alpha-cm_tsmom`, `KG-B2 residual alpha-cm_carry`, `KG-B3 out-of-sample-cm_carry`) blijven **rood en dat is correct**: het zijn pre-registreerde go/no-go-poorten op gefalsificeerde hypotheses. Repareer ze niet. Bewijs aan het eind van de fase dat de faalbasislijn is gedaald van 6 naar **4**.

### 0.9 De ledger, en waarom `M` heilig is

```
M_before = 2.715
register_risk_config(1b60cb664fbf9a2a)
M_after  = 2.715
```

Risicoconfiguratie is geen hypothese; 13 tests in `test_risk_config_is_not_a_hypothesis.py` bewijzen dat, inclusief één die aantoont dat een **echte** hypothese `M` wél beweegt. Deze fase is de eerste sinds Phase 3 die `M` massaal laat stijgen: elke GARCH-variant × elk symbool × elke horizon, elk aantal HMM-states, elke barrier-configuratie, elke CatBoost-hyperparameterset.

**Registreer het geplande aantal trials vóór de run en het feitelijke aantal erna.** Wijken die af, verklaar het verschil. Een DSR op een te lage `M` is de meest voorkomende vorm van zelfbedrog in kwantitatief onderzoek, en op een venster van 1.743 bars is hij dodelijk.

### 0.10 Één procedurele les uit Phase 5, letterlijk over te nemen

Phase 5 documenteerde negen defecten in het eigen werk (§12), waarvan vijf **in de bewijsvoering zelf** zaten: een pariteitstest die eindequity vergeleek en op één van drie seeds de verkeerde lag koos; dezelfde test op een constante exposure waardoor de lag niet identificeerbaar was; een wiring-test met niet-bindende limietwaarden die dus niets bewees; een ledger-isolatietest die het echte governance-artefact vervuilde; een onjuiste tussenconclusie door een extractiefout (`tail -3 | head -2`).

> **De les:** een groene test bewijst niets zolang je niet hebt aangetoond dat hij rood kán worden om de juiste reden. Elke statistische toets in deze fase krijgt een **negatieve controle**: injecteer een signaal waarvan je weet dat het geen informatie draagt (gerandomiseerde labels, geshuffelde regime-reeks, permutatie van het RV-doel) en bewijs dat de toets dat correct verwerpt. Een DM-HLN-test die ook significant is op ruis, meet niets.

---

## 1. ROL EN CONTEXT

Je acteert als **Senior Quantitative Researcher — Econometrics & Machine Learning**. Je opereert met de scepsis die past bij financiële tijdreeksen waarin de signaal-ruisverhouding extreem laag is. Jouw taak is niet om complexiteit te introduceren, maar om complexiteit te laten bewijzen dat ze haar plaats verdient — of haar te falsificeren.

Je werkt uitsluitend binnen de kaders van `docs/ARCHITECTUUR_AUDIT_2026-08-22.md`.

**Relevante lagen uit de Target Architecture (§19):**

| Laag | Naam | Rol in deze fase |
|---|---|---|
| **L2** | Volatility Engines | GARCH-familie en HAR-RV als Level 2/3 uitdagers van EWMA |
| **L3** | Regime Engines | M0 Causal Vol-Buckets (baseline) vs. M1/M2 Markov-modellen |
| **L5** | Conditioning Layer (Alpha × Regime Overlay) | alleen toegankelijk voor gepromoveerde regime-modellen |
| **L6** | Model Combination & Ensembles | meta-labeling als secondary model |
| **L10** | Backtesting Engine | **ongewijzigd overnemen**; deze fase herontwerpt hem niet |
| **L11** | Statistical Validation & Falsification | de poort waar elk model doorheen moet |

> **Doctrine (§6):** *"Een model wordt pas gearchiveerd of verwijderd als het bewezen slecht is onder een correct geconfigureerde baseline. Zolang het onbewezen is, verblijft het in de Research Track en krijgt het geen toegang tot de productie-pijplijn."*

**Bindende hiërarchie en regels:**

- **Volatiliteit (§9.1):** Level 0 naïef → **Level 1 EWMA λ=0.94 (BASELINE PRODUCTIE)** → Level 2 GARCH/GJR/EGARCH/APARCH → Level 3 Realized Volatility / HAR-RV.
- **Regimes (§10.1):** **M0 Causal Vol-Buckets = BASELINE PRODUCTIE.** M1 Markov Chain = Research Track. M2 HMM = *Restricted Research Only*. M3 Markov-Switching GARCH = *Exclusief Theoretisch* — **niet implementeren in deze fase**.
- **De Filtered-vs-Smoothed regel (§10.2):** smoothed probabilities `P(S_t | F_T)` gebruiken de volledige dataset. **GEBRUIK IN BACKTESTS IS STRENG VERBODEN.** Uitsluitend filtered probabilities `P(S_t | F_t)` uit het forward-algoritme zijn toegestaan.
- **ML (§12.1):** geen blinde classificatie of regressie op ruwe richting van dagelijkse returns. ML is **secondary model** (meta-labeling, López de Prado): het primaire model bepaalt de richting, het ML-model voorspelt de kans op succes.
- **Portfolio (§13.1):** HRP is Research Track; Markowitz in ruwe vorm is **banned**.

**Kritieke bevindingen die deze fase adresseert (§24):**

| Bevinding | Vereist oordeel | Drempel |
|---|---|---|
| `risk/hmm_regime.py` | **REDESIGN** naar M2 Filtered HMM | QLIKE- of OOS Sharpe-winst t.o.v. M0, na kosten |
| CatBoost Direct Directional | **ARCHIVE** | terugkeer alleen als meta-labeler met **OOS AUC > 0.58** |
| HRP | **RESEARCH ONLY** | toelating vereist turnover-gecorrigeerde OOS Sharpe > Inverse Volatility |

---

## 2. DOEL VAN DE FASE

Voer een gecontroleerde, gepre-registreerde evaluatie uit van alle Level 2+ modellen tegen hun respectieve baselines, en laat uitsluitend modellen met aantoonbare, kostengecorrigeerde OOS-waarde toe tot de productiepijplijn.

Na deze fase heeft elk complex model een expliciet, in de ledger vastgelegd oordeel: **`PROMOTED`**, **`UNPROVEN — insufficient data`** of **`FALSIFIED`** — onderbouwd met de juiste statistische toets, gemeten door de authoritative engine, en gerapporteerd inclusief de kostenlabels waaronder het oordeel geldt.

**Wat expliciet NIET in scope is:**

- Herontwerp van de soevereine risicolaag of de engine (Phase 4/5-territorium; §2.1 van de Phase 5-opdracht verbood het daar, hier geldt hetzelfde).
- Het `live/`-pad en de vijf openstaande C-items (C1–C4, C6) — dat is **Phase 7**.
- M3 Markov-Switching GARCH.
- Nieuwe alpha-hypotheses. Deze fase evalueert *modellen*, geen signalen. Het primaire signaal komt ongewijzigd uit Phase 3.
- Orderboekdata-acquisitie, tenzij de data buiten deze fase beschikbaar komt.

---

## 3. DATA ADEQUACY GATE — VERPLICHT VÓÓR ELKE FIT

Nieuw ten opzichte van eerdere fasen, en de directe consequentie van §0.6. **Geen enkel model mag worden gefit voordat zijn data-adequaatheid is gemeten en geregistreerd.**

Bouw `src/tradebot/validation/data_adequacy.py` met per modelklasse een expliciete, in `conf/model/adequacy.yaml` geconfigureerde eis:

| Modelklasse | Minimumeis | Wat je rapporteert |
|---|---|---|
| GARCH(1,1) / GJR / EGARCH / APARCH | ≥ N observaties per fit-venster (uit conf), convergentie zonder randoplossing | effectieve venstergrootte, aantal fits, convergentieratio |
| HAR-RV | dekking van 1m/5m bars per dag (uit de Phase 1-dekkingsanalyse) | % dagen met voldoende intraday-dekking |
| M2 HMM (k states) | ≥ N observaties **per toestand per fold**, geschat via de filtered occupancy | verwachte occupancy per state per fold |
| Meta-labeling (CatBoost) | ≥ N triple-barrier events per fold ná purging en embargo, én klassebalans binnen bandbreedte | events per fold, positieve-klasseratio |
| HRP | ≥ N observaties voor een stabiele correlatiematrix op 6 assets | conditiegetal en stabiliteit van de matrix over folds |

**De poort werkt als volgt:** haalt een model de eis niet, dan wordt het **niet gefit**. Het krijgt direct het oordeel `UNPROVEN — insufficient data` met de gemeten cijfers erbij, en de trial telt **niet** mee in `M` omdat er geen search heeft plaatsgevonden. Dit voorkomt de twee fouten die anders onvermijdelijk zijn: een gefalsificeerd model dat alleen te weinig data had, en een gepromoveerd model dat op tientallen observaties per toestand is gefit.

Voeg een **power-analyse** toe aan elke pre-registratie: welk effect zou je met 1.743 bars, 6 symbolen en het gekozen aantal folds detecteerbaar kunnen maken bij α = 0,05? Is het minimaal detecteerbare effect groter dan het effect dat je redelijkerwijs verwacht, dan is de test bij voorbaat niet informatief — noteer dat vóór de run, niet erna.

---

## 4. CONCRETE DELIVERABLES

### 4.1 Governance en fundament

| # | Deliverable | Eis |
|---|---|---|
| 1 | **Drie pre-registraties** via `registry/preregistration.py` | H1/H2/H3, elk met nulhypothese, universum, periode, parameterruimte, gepland aantal trials, power-analyse, stop-criteria — vastgelegd vóór de eerste run |
| 2 | `src/tradebot/validation/data_adequacy.py` + `conf/model/adequacy.yaml` | §3; harde poort, geen waarschuwing |
| 3 | `tests/conftest.py` — Hypothesis-profiel `derandomize=True` | sluit exit-report §11.3; sluit de vraag over de zevende failure |
| 4 | **Fix** `har_rv_forecast` (non-negativiteit) en `evt_gpd_var` (eindigheid op degenerate invoer) | faalbasislijn van 6 naar 4; de vier killgates blijven rood |
| 5 | **DVC-stage** voor de authoritative engine (`apps/run_phase5_baseline.py`-opvolger) | reproduceerbaarheid van elke Phase 6-run; vóór Stap 10 gereed |
| 6 | Uitbreiding statische sovereign-audit naar de Phase 6-modules | `test_sovereign_wiring.py`; bewijs dat de uitbreiding een schending detecteert |

### 4.2 Econometrische toetsingsketen

| # | Deliverable | Inhoud |
|---|---|---|
| 7 | `src/tradebot/validation/econometrics.py` | §8.2: ADF & KPSS, CUSUM & structural break tests, Ljung-Box, **Engle ARCH-test** |
| 8 | `src/tradebot/features/fracdiff.py` | §8.1: minimale `d ∈ [0,1]` die stationariteit bereikt (ADF p < 0,05) met maximaal behoud van geheugen; `d` uit `conf/model/` |
| 9 | `reports/ECONOMETRIC_DIAGNOSTICS.md` | per reeks: alle vijf toetsen, gekozen `d`, behouden geheugen, en het ARCH-poortoordeel |

### 4.3 Volatiliteitsspoor (H1)

| # | Deliverable | Eis |
|---|---|---|
| 10 | `src/tradebot/volatility/garch.py` | GARCH(1,1), GJR-GARCH, EGARCH, APARCH conform §9.2; expliciete verdelingsaanname (**Student-t** voor zware staarten); convergentiecontrole; **harde crash bij niet-convergentie, geen fallback naar EWMA** |
| 11 | `src/tradebot/volatility/realized.py` | Realized Variance en HAR-RV op de 1m/5m bars uit Phase 1; range-estimators (`parkinson.py`, `garman_klass.py`, `rogers_satchell.py`, `yang_zhang.py`) als daily proxies; **non-negativiteit gegarandeerd en getest** |
| 12 | `src/tradebot/validation/vol_metrics.py` | **QLIKE** `RV_t / σ̂²_t − ln(RV_t / σ̂²_t) − 1` (de enige proxy-ruis-robuuste loss) · MSE-SD · MAE-SD · **Mincer-Zarnowitz** `RV̂_t = α + β·σ̂²_t + ε_t` met toets op `α = 0, β = 1` · **Diebold-Mariano met Harvey-Leybourne-Newbold correctie** |
| 13 | `reports/GARCH_VS_EWMA_COMPETITION.md` | volledige QLIKE-competitie per symbool × per horizon, DM-HLN p-waarden, MZ-coëfficiënten, RV-proxy-provenance, negatieve controle, expliciet oordeel per model |

### 4.4 Regimespoor (H2)

| # | Deliverable | Eis |
|---|---|---|
| 14 | `src/tradebot/regime/buckets.py` | **M0 Causal Vol-Buckets**: harde drempels op EWMA z-score en ATR-ratio's. Nul latente toestanden, nul geschatte parameters, lekrisico nul. Dit is de te verslaan baseline. |
| 15 | `src/tradebot/regime/markov.py` | M1 Markov Chain en M2 Gaussian/Student-t HMM. **Uitsluitend filtered probabilities via het forward-algoritme.** De smoothed-variant bestaat alleen als expliciet gemarkeerde diagnostiek en is technisch onbereikbaar vanuit de backtest. |
| 16 | `src/tradebot/risk/hmm_regime.py` — **herontworpen** | §24 REDESIGN naar het M2-contract in `regime/markov.py`. De EMA-crossover-fallback is in Phase 0 gesloopt en **mag onder geen enkele omstandigheid terugkeren** (§5.2). |
| 17 | `tests/lookahead/test_filtered_only_enforcement.py` | bewijst dat geen enkel backtest-pad smoothed probabilities kan bereiken; een geïnjecteerde poging crasht met `CausalityViolationError`; inclusief de negatieve controle die aantoont dat de test rood wordt |
| 18 | `reports/M0_VS_HMM_BENCHMARK.md` | M0 vs. M2 Filtered HMM: OOS Sharpe ná transactiekosten, regime-stabiliteit (aantal wisselingen, gemiddelde regimeduur), **turnover-delta en de fees die de wisselingen kosten**, aantal effectief handelende bars per track, spread-sensitiviteit |

### 4.5 Meta-labeling en ML (H3)

| # | Deliverable | Eis |
|---|---|---|
| 19 | `src/tradebot/labeling/triple_barrier.py` | triple-barrier labeling met expliciete horizon; barrier-breedtes en horizon uit `conf/model/`; **respecteert de `shift(2)`-conventie** |
| 20 | `src/tradebot/train/meta_label.py` | CatBoost **uitsluitend** als secondary model: het primaire Phase 3-baselinesignaal bepaalt de richting, het ML-model voorspelt de succeskans (sizing / trade filter). Directionele voorspelling op ruwe returns is **technisch geblokkeerd**, niet afgeraden. |
| 21 | `src/tradebot/validation/feature_importance.py` | **MDI** en **SFI**, beide onder **Purged Cross-Validation met embargo** (§12.1) |
| 22 | `reports/META_LABELING_EVALUATION.md` | OOS AUC met betrouwbaarheidsinterval, precision/recall op het **operationele werkpunt**, events per fold, klassebalans, MDI/SFI, negatieve controle op gerandomiseerde labels, expliciet oordeel tegen **AUC > 0.58** |

### 4.6 Portfolio (research-gated)

| # | Deliverable | Eis |
|---|---|---|
| 23 | `src/tradebot/portfolio/hrp.py` (research-gated) | HRP blijft bestaan maar is **technisch geblokkeerd** voor productiegebruik totdat turnover-gecorrigeerde OOS Sharpe > Inverse Volatility is aangetoond; de gate is een crash, geen vlag |
| 24 | `reports/HRP_VS_INVERSE_VOL.md` | vergelijking inclusief de expliciete vaststelling uit §0.7 dat dit universum geen clusterstructuur heeft, en de stabiliteit van de HRP-boomordening over folds |

### 4.7 Afsluiting

| # | Deliverable | Eis |
|---|---|---|
| 25 | Ledger-entries per model | atomair; `PROMOTED` / `UNPROVEN — insufficient data` / `FALSIFIED` met statistische onderbouwing, `M`-bijdrage, `config_hash`, `data_hash`, git-sha |
| 26 | `docs/FALSIFICATION_REGISTER.md` bijgewerkt | elk gefalsificeerd model met de toets die hem falsificeerde |
| 27 | `docs/ARCHITECTURAL_DECISIONS.md` uitgebreid | elke keuze die niet uit de code leesbaar is (RV-sampling, aantal states, barrier-breedtes, embargo-lengte) |
| 28 | `reports/phase6_exit_report.md` | volledig, inclusief een sectie *"wat er tijdens deze fase mis bleek in mijn eigen werk"* — een exit-rapport dat alleen successen noemt, is geen exit-rapport |

---

## 5. STAPSGEWIJZE UITVOERING

**Stap 0 — Fundament opruimen en de faalbasislijn verlagen.**
Registreer het Hypothesis-profiel met `derandomize=True`. Repareer `har_rv_forecast` (non-negativiteit) en `evt_gpd_var` (eindigheid). Draai de volledige suite driemaal en bewijs: **4 failures, dezelfde vier killgate-namen, elke run.** Dit gebeurt eerst omdat `har_rv_forecast` een deliverable van deze fase is en een negatieve variantieforecast QLIKE ongedefinieerd maakt. Commit: `test(config): register derandomized hypothesis profile`, `fix(volatility): guarantee non-negative HAR-RV forecasts`, `fix(risk): make EVT GPD VaR finite on degenerate input`.

**Stap 1 — Drie pre-registraties schrijven.**
Vóór één regel modelcode. Elk met nulhypothese, universum, periode, parameterruimte, gepland aantal trials, **power-analyse**, gekozen primaire signaaltrack met verantwoording tegen het haltprobleem (§0.5), en stop-criteria:

- **H1:** de GARCH-familie verslaat EWMA(0.94) OOS op QLIKE, significant onder Diebold-Mariano met HLN-correctie.
- **H2:** M2 Filtered HMM levert een superieure OOS Sharpe ná kosten ten opzichte van M0 Causal Vol-Buckets, gemeten door de authoritative engine op een track die het volledige venster handelt.
- **H3:** CatBoost als meta-labeler bereikt OOS AUC > 0.58 op triple-barrier-labels van het Phase 3-baselinesignaal.

Commit: `docs(registry): pre-register phase 6 advanced research waves`.

**Stap 2 — Data Adequacy Gate bouwen.**
`validation/data_adequacy.py` + `conf/model/adequacy.yaml`. Meet en publiceer de adequaatheid van alle vijf modelklassen **voordat** je iets fit. Blijkt hier al dat H2 of H3 onhaalbaar is op dit universum, dan is dat de goedkoopste bevinding van de hele fase — registreer hem en pas de planning aan in plaats van hem te ontdekken na drie weken fitten.

**Stap 3 — Econometrische toetsingsketen bouwen.**
Implementeer `validation/econometrics.py`. Draai ADF, KPSS, CUSUM, Ljung-Box en de Engle ARCH-test op elke reeks die de vol-pipeline binnenkomt. **De ARCH-test is de poortwachter:** is er geen aantoonbare conditionele heteroskedasticiteit, dan is een GARCH-structuur niet gerechtvaardigd en stopt het spoor daar met een gedocumenteerd oordeel.

**Stap 4 — Fractionele differentiëring.**
Bouw `features/fracdiff.py`, bepaal per reeks de minimale `d` en rapporteer het behouden geheugen. `d` is geen constante in code; hij komt uit `conf/model/` en wordt per reeks geregistreerd.

**Stap 5 — Realized Volatility fundament leggen.**
Bouw `realized.py` en construeer de RV-proxy op minuutbasis. **De QLIKE-competitie is alleen zo goed als de proxy.** Documenteer expliciet de microstructuurruis-behandeling, de sampling-frequentie en de dagen met onvoldoende dekking; verwijs naar de dekkingsanalyse uit Phase 1. Ontbreekt intraday-dekking voor een deel van het venster, dan is dat een beperking van de competitie en hoort zij in het rapport, niet in een voetnoot.

**Stap 6 — GARCH-familie implementeren.**
GARCH(1,1), GJR-GARCH, EGARCH, APARCH met Student-t innovaties. Fit uitsluitend op expanding of rolling vensters binnen de walk-forward-structuur — **nooit** op de volledige sample. Niet-convergentie en randoplossingen zijn resultaten die je registreert, geen problemen die je wegvangt. Rapporteer de convergentieratio per model per symbool.

**Stap 7 — QLIKE-competitie draaien.**
Evalueer elk model OOS tegen de RV-proxy. Rapporteer QLIKE, MSE-SD, MAE-SD en de MZ-coëfficiënten. Toets significantie met DM + HLN, per symbool en per horizon, met eerlijke telling van elke trial in `M`. **Voer de negatieve controle uit:** vergelijk EWMA met een geshuffelde variant van zichzelf en bewijs dat DM-HLN daar géén significantie vindt. Promoveert GARCH niet significant, dan blijft EWMA(0.94) de productie-estimator — een compleet en publiceerbaar resultaat.

**Stap 8 — M0 Causal Vol-Buckets bouwen.**
Implementeer de baseline eerst. M0 heeft nul geschatte parameters en nul lekrisico; dat maakt hem tot een oneerlijk sterke tegenstander, en precies daarom is hij de juiste baseline. Drempels uit `conf/model/`.

**Stap 9 — Filtered HMM bouwen en de smoothed-route dichtmetselen.**
Implementeer M2 met het forward-algoritme. Bouw daarna `test_filtered_only_enforcement.py` en bewijs dat het backtest-pad de smoothed-variant niet kan bereiken, plus de negatieve controle die aantoont dat de test een geïnjecteerde poging daadwerkelijk detecteert. **Bouw de blokkade voordat je het model op data loslaat** — anders sluipt de smoothed-variant er tijdens exploratie in en is elk daarna gemeten resultaat besmet.

**Stap 10 — DVC-stage voor de authoritative engine.**
Sluit exit-report §17.4 punt 3. Zonder deze stage is geen enkele ML-run reproduceerbaar, en een niet-reproduceerbare AUC is geen bewijs.

**Stap 11 — M0 vs. HMM benchmarken door de authoritative engine.**
Conditioneer het gekozen Phase 3-baselinesignaal op M0 en op de filtered HMM-probabilities. Draai **beide** door `EventDrivenEngine` met gekalibreerde kosten, `config_hash = 1b60cb664fbf9a2a`, en de `shift(2)`-conventie. Meet: OOS Sharpe ná kosten, aantal effectief handelende bars, regimewisselingen, turnover-delta, fees-delta, en de spread-sensitiviteit (bij welke half-spread verdwijnt de winst?). Vergeet niet dat de accounting-invarianten na elke bar sluiten — een crash hier is een fout in jouw conditionering, niet in de engine.

**Stap 12 — Meta-labeling opzetten.**
Bouw triple-barrier labeling en train CatBoost als secondary model. Het primaire signaal bepaalt de richting; het ML-model voorspelt uitsluitend de succeskans. Purged Walk-Forward met embargo, scaler per fold gefit. Bereken MDI en SFI. Rapporteer OOS AUC met betrouwbaarheidsinterval tegen de drempel 0.58, en de precision/recall op het werkpunt waarop je het filter daadwerkelijk zou inzetten. **Negatieve controle verplicht:** train hetzelfde model op gerandomiseerde labels en bewijs dat de AUC dan rond 0,50 ligt. Ligt hij hoger, dan lekt er informatie in je pipeline.

**Stap 13 — Meta-label-filter door de engine halen.**
Een AUC boven 0,58 is een statistisch resultaat, geen economisch resultaat. Draai het gefilterde signaal door de authoritative engine en meet of het filter na fees netto waarde toevoegt. Een filter dat de helft van de trades weghaalt en de Sharpe met 0,02 verbetert, verdient geen promotie.

**Stap 14 — HRP evalueren onder research-gate.**
Vergelijk HRP met Inverse Volatility op turnover-gecorrigeerd OOS-rendement, met de expliciete vaststelling uit §0.7 erbij. Meet de stabiliteit van de boomordening over folds. Blijft HRP achter of is de ordening instabiel, dan blijft de productiegate dicht. Dat is een geldig en definitief resultaat.

**Stap 15 — Oordelen, registreren, publiceren.**
Ken elk model expliciet een status toe. Draai voor elk gepromoveerd model de volledige Phase 2-gate: 6 lookahead-tests, DSR met eerlijke `M`, SPA over de multi-modelvergelijking. Registreer atomair in de ledger, werk `FALSIFICATION_REGISTER.md` en `ARCHITECTURAL_DECISIONS.md` bij, en publiceer alle vijf de rapporten plus `reports/phase6_exit_report.md`.

---

## 6. CRITERIA & VALIDATIE (EXIT CRITERIA)

| # | Criterium |
|---|---|
| 1 | **Alleen modellen met aantoonbare OOS-waarde zijn goedgekeurd.** Elk model in de productiepijplijn heeft een ledger-entry met statistische onderbouwing, `M`-bijdrage en kostenlabels. |
| 2 | **QLIKE-competitie compleet.** GARCH-familie vs. EWMA(0.94), per symbool en per horizon, met DM-HLN p-waarden, MZ-coëfficiënten en RV-proxy-provenance. GARCH promoveert uitsluitend bij significante OOS-superioriteit; anders blijft EWMA de productie-estimator. |
| 3 | **Filtered-only afgedwongen.** `test_filtered_only_enforcement.py` bewijst dat smoothed probabilities technisch onbereikbaar zijn vanuit elk backtest-pad (§10.2), en de negatieve controle bewijst dat de test werkt. |
| 4 | **M0 vs. HMM beslist.** Het HMM promoveert uitsluitend bij superieure OOS Sharpe **ná transactiekosten** door de authoritative engine (§10.2). Anders: `UNPROVEN` of `FALSIFIED`, en M0 blijft productie-baseline. |
| 5 | **Meta-labeling beoordeeld tegen AUC > 0.58** (§24), mét betrouwbaarheidsinterval en negatieve controle. Onder die drempel blijft CatBoost `ARCHIVED`. Directionele CatBoost op ruwe daily returns is technisch geblokkeerd. |
| 6 | **Economische toets bovenop de statistische.** Elk model dat een statistische drempel haalt, is óók door de authoritative engine gehaald met turnover- en fees-delta gerapporteerd. |
| 7 | **Feature importance rigoureus.** MDI én SFI onder Purged CV met embargo voor elk ML-spoor (§12.1). |
| 8 | **Econometrische keten doorlopen.** ADF, KPSS, CUSUM, Ljung-Box en Engle ARCH gerapporteerd voor elke reeks in de vol-pipeline; het ARCH-poortoordeel expliciet. |
| 9 | **Data Adequacy Gate actief en gerapporteerd.** Elk model heeft een gemeten adequaatheidsscore; elk `UNPROVEN — insufficient data` is met cijfers onderbouwd, niet met een vermoeden. |
| 10 | **Power-analyse per pre-registratie**, met het minimaal detecteerbare effect naast het verwachte effect. |
| 11 | **HRP blijft research-gated** tenzij turnover-gecorrigeerde OOS-superioriteit boven Inverse Volatility is bewezen én de boomordening stabiel is. |
| 12 | **Negatieve controle per statistische toets.** Elke toets is aantoonbaar in staat de nulhypothese te accepteren op ruis. |
| 13 | **Alle Phase 2-gates groen** voor elk gepromoveerd model: 6 lookahead-tests, DSR met eerlijke `M`, SPA over de multi-modelvergelijking. |
| 14 | **`M` correct bijgewerkt.** Gepland aantal trials vóór de run, feitelijk aantal erna, verschil verklaard. `M_before = 2.715` staat in het rapport. |
| 15 | **Faalbasislijn verlaagd van 6 naar 4.** De twee property-bugs zijn gerepareerd; de vier killgates staan nog rood en dat is correct. |
| 16 | **Hypothesis-profiel `derandomize=True` geregistreerd**; drie volledige suite-runs geven identieke resultaten. Exit-report §11.3 is gesloten. |
| 17 | **DVC-stage voor de authoritative engine bestaat**; elke Phase 6-run is reproduceerbaar uit `data_hash` + `config_hash` + git-sha. |
| 18 | **Sovereign-audit uitgebreid** naar `volatility/`, `regime/`, `labeling/`, `train/`, `portfolio/hrp.py`, met een test die bewijst dat de uitbreiding een schending detecteert. |
| 19 | **Nul stille degradatie.** Geen enkel model valt bij niet-convergentie of ontbrekende dependency terug op een eenvoudiger alternatief. |
| 20 | **Exit report compleet**, inclusief de sectie over defecten in het eigen werk. |

---

## 7. ABSOLUTE NO-GO CONDITIES

Is één van deze condities aan het eind van de fase actief, dan is de fase **niet** afgerond, ongeacht hoeveel deliverables er staan.

| # | Conditie |
|---|---|
| 1 | Een smoothed probability is vanuit een backtest-pad bereikbaar |
| 2 | Een model is gepromoveerd op vectorized bewijs |
| 3 | Een model is gepromoveerd zonder de authoritative engine met kosten te hebben doorlopen |
| 4 | Een model valt bij niet-convergentie of ontbrekende dependency terug op EWMA of een EMA-crossover |
| 5 | De EMA-crossover-fallback uit §5.2 is in enige vorm teruggekeerd |
| 6 | `M` is niet bijgewerkt voor alle uitgevoerde trials |
| 7 | Een DSR of SPA is gerapporteerd zonder de trials van deze fase in `M` |
| 8 | Een `FALSIFIED`-oordeel is toegekend aan een model dat de Data Adequacy Gate niet haalde |
| 9 | Een promotieclaim staat in een rapport zonder de labels `IMPACT_UNCALIBRATED` en `SPREAD_ASSUMED` |
| 10 | Een Phase 6-module definieert een eigen risicolimiet |
| 11 | Een label of feature gebruikt informatie van ná `t` |
| 12 | Een scaler of imputer is over folds heen gefit |
| 13 | Een test schrijft naar een echt governance-artefact |
| 14 | Een statistische conclusie is getrokken zonder negatieve controle |
| 15 | HRP is productie-toegankelijk zonder bewijs |

---

## 8. REGELS & HANDELINGSINSTRUCTIES

- **Fail-fast compliance.** Niet-convergerende GARCH-fit, ontbrekende `hmmlearn`, te weinig observaties voor een HMM-fit, ontbrekende intraday-dekking: crashen en registreren. Nul `try/except` fallbacks. De EMA-crossover-fallback uit §5.2 mag onder geen enkele omstandigheid terugkeren.
- **100 % PIT rigor.** Elke modelfit gebruikt uitsluitend data ≤ `t`. Elke regime-classificatie is filtered. Elke label is gepurged en geëmbargood. Elke scaler is per fold gefit. Elke positie respecteert `latency_bars = 1` bovenop de beslisbar (§0.2).
- **Onbewezen ≠ bewezen slecht (§6).** Faalt een model door ontoereikende data, label het `UNPROVEN — insufficient data` mét de gemeten adequaatheidscijfers en laat het in de Research Track. Falsificeer alleen wat aantoonbaar slechter is dan zijn baseline onder correcte condities.
- **Baseline-first.** Elk complex model wordt uitsluitend beoordeeld ten opzichte van zijn expliciete baseline: GARCH vs. EWMA(0.94), HMM vs. M0, HRP vs. Inverse Vol, CatBoost vs. het ongefilterde primaire signaal. Absolute performance is irrelevant.
- **Statistisch is niet economisch.** Een p-waarde is geen euro. Elke statistische winst gaat door de engine en wordt in netto-rendement na fees uitgedrukt, met de turnover-delta erbij.
- **Tel elke trial.** Elke parametervariant, elke symbool-specifieke fit, elke horizon, elk aantal states, elke barrier-breedte telt mee in `M`. Op 1.743 bars maakt een te lage `M` de DSR structureel te optimistisch.
- **Elke toets krijgt een negatieve controle.** Zie §0.10. Een test die niet rood kan worden, bewijst niets.
- **Geen hardcoded variabelen.** λ, `d`, drempels, aantal states, barrier-breedtes, embargo-lengtes, horizons en adequaatheidsminima komen uit `conf/model/`.
- **Atomaire commits.** `feat(volatility): add GJR-GARCH with student-t innovations`, `feat(regime): implement filtered-only HMM with forward algorithm`, `test(lookahead): block smoothed probabilities from backtest paths`, `feat(validation): gate every model fit on measured data adequacy`, `docs(reports): publish GARCH vs EWMA QLIKE competition results`.
- **Rapporteer negatieve resultaten volledig.** Een gefalsificeerd model is een even waardevol resultaat als een gepromoveerd model, en aanzienlijk goedkoper dan de live ontdekking ervan. De meest waarschijnlijke uitkomst van deze fase is dat EWMA en M0 blijven staan — schrijf dat resultaat met dezelfde zorg op als een promotie.
- **Schrijf het eigen-defecten-hoofdstuk terwijl je werkt.** Phase 5 vond vijf van zijn negen defecten in de bewijsvoering zelf. Verwacht hetzelfde hier, en noteer ze op het moment dat je ze vindt.

---

## 9. STARTINSTRUCTIE

> **Begin met Stap 0, niet met Stap 1.** Registreer eerst het Hypothesis-profiel met `derandomize=True` in `tests/conftest.py` en repareer de twee property-bugs: `har_rv_forecast` mag geen negatieve variantie opleveren en `evt_gpd_var` moet eindig zijn op degenerate invoer. Draai daarna de volledige suite driemaal en bewijs dat er exact **4** failures resteren, alle vier de pre-registreerde killgates op `cm_carry` en `cm_tsmom`. Dit gaat vóór de pre-registraties omdat `har_rv_forecast` een Level 3-deliverable van deze fase is en een negatieve variantieforecast QLIKE ongedefinieerd maakt — de competitie in Stap 7 zou dan op een kapotte metriek rusten.
>
> Commit als:
> `test(config): register derandomized hypothesis profile to close the seventh-failure question`
> `fix(volatility): guarantee non-negative HAR-RV forecasts`
> `fix(risk): make EVT GPD VaR finite on degenerate input`
>
> **Ga daarna naar Stap 1:** schrijf de drie pre-registraties (H1 GARCH vs. EWMA op QLIKE, H2 M2 Filtered HMM vs. M0 Causal Vol-Buckets, H3 CatBoost meta-labeling met AUC-drempel 0.58) via `registry/preregistration.py`, elk met nulhypothese, universum, periode, parameterruimte, gepland aantal trials, power-analyse op 1.743 bars en 6 symbolen, de gekozen primaire signaaltrack met verantwoording tegen het haltprobleem uit §0.5, en stop-criteria. Commit als `docs(registry): pre-register phase 6 advanced research waves`.
