# PHASE 7/8 — EXIT REPORT: CONSOLIDATIE, PRODUCTION READINESS & OPLEVERING

> **Deliverable E-5** · afgesloten 2026-09-01
> **git_sha bij afsluiting:** `df96805` plus de niet-gecommitte Stage C/D/E-boom
> **Meetregel:** elk cijfer in dit rapport is op 2026-09-01 gemeten. Waar een
> eerdere meting afweek, staat de afwijking erbij en **prevaleert de nieuwe**.

---

## 0. De uitkomst, eerst

**De fase is niet afgerond.** Vier van de vijf stages hebben hun poort gehaald;
Stage D niet, en Stage E niet volledig. Dat staat vooraan omdat een exit-rapport
dat met successen begint, de lezer traint om het einde over te slaan.

| Stage | Poort | Status |
|---|---|---|
| A — Fundament | drie ratchets exit 0 | **gehaald** (eerdere sessie) |
| B — Governance (D-1) | de gate blokkeert een lekkend model | **gehaald** (eerdere sessie) |
| C — Phase 6 afmaken | H1, H2 en H3 elk een ledger-oordeel | **gehaald**, met één openstaande deliverable (stap 14, HRP) |
| D — Productie | 60 opeenvolgende schone dagen | **NIET gehaald** — de klok is niet gestart, en kan dat nog niet |
| E — Oplevering | zie E1–E5 | **deels** — E2, E3, E4 gehaald; E1 en E5 met beperkingen |

Wat er wél staat: drie beslist hypothesen, een live-keten waarvan de
risicolimieten voor het eerst uit de soevereine policy komen, een HALT die niet
meer met de klok verloopt, en een documentatiegate die drift tegenhoudt in
plaats van hem op te ruimen.

Wat er niet staat: één codepad. `live/` en de Phase 5-keten zijn nog steeds twee
ketens, en zolang dat zo is, kan de 60-daagse klok niet zinvol lopen.

---

## 1. Stage C — de drie oordelen

Alle drie de fase-6 hypothesen zijn beslist. **Geen enkele is gepromoveerd, en
geen enkele is gefalsificeerd.** `M` blijft **2776**.

| | H1 — GARCH vs. EWMA | H2 — HMM vs. M0 | H3 — CatBoost meta-labeling |
|---|---|---|---|
| Oordeel | `UNPROVEN` | `UNPROVEN — insufficient data` | `ARCHIVED` |
| Bindend criterium | proxy-premisse geschonden | bezetting onder de poort | AUC-drempel |
| Ledger-amendement | `ae4823e95f6d5844` | `beb6a14e4f362658` | `b08585c4394488eb` |
| Rapport | `reports/GARCH_VS_EWMA_COMPETITION.md` | `reports/M0_VS_HMM_BENCHMARK.md` | `reports/META_LABELING_EVALUATION.md` |

### 1.1 H3 in het kort

Zes gepre-registreerde specs, alle zes `ARCHIVED`. OOS-AUC **0,4768 tot
0,4842** — alle zes onder 0,50 — met een hoogste conservatieve ondergrens van
0,4113 tegen een drempel van 0,58. De negatieve controle is schoon
(**0,5100** tegen een grens van 0,55), dus de run is geldig; de adequaatheids-
poort bindt niet (442,7 effectieve trainevents per fold tegen een eis van 100),
dus dit is géén `UNPROVEN` zoals H1 en H2. Er was genoeg data, er is gefit, en
er is gemeten.

**Waarom `ARCHIVED` en niet `FALSIFIED`.** De gepre-registreerde actie voor
`auc_below_threshold` is `archive`, en de power-analyse die vóór de run vastlag
noemt het conservatieve scenario **niet-informatief**: de MDE is 0,1029 tegen
een benodigd overschot van 0,08. Een toets die haar effect niet kan detecteren,
kan het ook niet verwerpen.

**De twee lezingen zijn het oneens, en dat hoort er te staan.** Op het NOMINALE
interval ligt de bovengrens van de beste spec op 0,4976 — onder 0,50 — en een
pipeline zonder uniqueness-correctie zou hier dus concluderen dat het model
significant SLECHTER dan willekeurig rangschikt. Dat is precies het interval dat
de pre-registratie vóór de run heeft afgewezen als te smal.

**Reproduceerbaarheid, gemeten:** drie opeenvolgende runs leverden een
bit-identiek artefact (`md5 7e78b12305533aa53f17ad7345e8fd46`).

---

## 2. Stage D — wat dicht is en wat niet

| Criterium | Status | Bewijs |
|---|---|---|
| **D2** — geen limiet defaultet naar oneindig | **groen** | `tests/unit/test_live_limits_are_sovereign.py` |
| **D3** — DI-7 gesloten | **groen** | `tests/unit/test_background_tasks_are_held.py` |
| **D4** — `AlertSeverity` kent `HALT` | **groen** | `tests/unit/test_alert_halt_severity.py` |
| **D5** — drempels gehasht vóór de klok | **groen** | `artefacts/governance/monitoring_config_hash.json` |
| **D9** — `HALTED` overleeft een herstart | **groen** | `tests/unit/test_live_halt_is_irreversible.py` |
| **D1** — één codepad | **rood** | `reports/phase7_divergence_map.md` |
| **D6** — de vier chaos-skips | **rood** | ongewijzigd; zij vragen D1 |
| **D7** — dagelijkse bit-identieke pariteit | **niet toetsbaar** | zie §2.2 |
| **D12** — runbook getoetst door een tweede persoon | **rood** | het runbook is niet herschreven |

### 2.1 De twee gevaarlijkste bevindingen in de live-keten

**Er was geen positielimiet.** `live/execution_controller.py` droeg
`_max_gross_notional: float = float("inf")` met de opmerking *"set via
attributes after init"*. Gemeten: **niets in de hele boom zette dat veld ooit.**
`_check_position_limits` vergeleek elk order met oneindig en kon per constructie
niet vuren. Dat is geen limiet met een ongelukkige default maar een functie die
eruitzag als een limiet — in de enige laag waar een ontbrekende limiet echt geld
kost. Er waren er bovendien twee, niet één: ook `_max_notional_per_symbol`.

**De HALT verliep met de klok.** De breaker schreef zijn trips wél naar schijf
en weigerde wél te starten met een niet-geaccordeerde trip — maar alleen voor
trips van de laatste **24 uur**. Een systeem dat halteerde en 25 uur later
herstartte, kwam schoon op met een trip die nog altijd `acknowledged: false`
droeg. Niemand had de halt opgeheven; de tijd had dat gedaan.
`risk/kill_switches.py::HaltStore.release` had het al opgeschreven: *"een kill
switch die vanzelf opheft is een bypass met een klok eraan."* Dit was moeilijker
te zien dan een ontbrekende persistentie, want het oogde correct.

### 2.2 Waarom D7 niet toetsbaar is

De divergence map vond twee dingen die niet in de D-1-opdracht staan en die
zwaarder wegen dan de bypass die er wél in staat, omdat zij niet een controle
omzeilen maar **de getallen zelf veranderen**:

| | live-keten | Phase 5-keten |
|---|---|---|
| `eta` | 0,1 / 0,142 (defaults) | **2,991922** (gekalibreerd, met provenance) |
| Slippage | 5,0 bp vast | 1,0 bp aangenomen half-spread |
| Featurestack | `features/pipeline.py` | `features/registry.py` → `features/base.py` |

Een factor ~30 op eta en twee volledig gescheiden featurestacks. Een
pariteitstest hierop meet de configuratie, niet het gedrag. **Deze twee gaan
vóór D7**, en D1 gaat vóór D6.

---

## 3. Stage E — oplevering

| Criterium | Status |
|---|---|
| **E1** — reproductiepad op een verse machine doorlopen | **NIET gehaald**; het pad staat in `docs/REPRODUCTION.md`, maar is niet op een schone machine gedraaid |
| **E2** — elke DI een definitief oordeel | **gehaald**; 3 gesloten met bewijs, 6 bewust geaccepteerd, 9 doorgeschoven met voorwaarde |
| **E3** — nul documenten die niet-bestaande artefacten claimen | **gehaald**, en afgedwongen door `tests/unit/test_docs_claim_only_what_exists.py` |
| **E4** — `docs/PROJECT_STATE.md` bestaat | **gehaald** |
| **E5** — dit rapport, inclusief het eigen-defecten-hoofdstuk | **gehaald** |

Bij E-1 bleek `pip install -e .` twee console-scripts aan te maken —
`tb-backtest-portfolio` en `tb-make-tearsheet` — die naar modules wezen die
Phase 5 heeft verwijderd. `make backtest` crashte dus met een
`ModuleNotFoundError`. Een gedocumenteerd commandopad dat niet werkt, is de
duurste soort documentatie: hij kost de lezer vertrouwen op het moment dat hij
het project voor het eerst probeert.

---

## 4. Wat er tijdens deze fase mis bleek in mijn eigen werk

Phase 5 vond vijf van zijn negen defecten in de bewijsvoering zelf. Deze fase
vond er **elf**, en dat is geen toeval: hoe strenger de poort, hoe vaker de
poort zelf het defect draagt.

### 4.1 Het rapport kende zijn conclusie voordat de campagne had gedraaid

`reporting/phase6_meta_labeling.py` droeg vaste tekst waar een meting hoorde:

* `"**Alle zes liggen ONDER 0,50**"` — geschreven vóór de eerste fit;
* `"## 2. De negatieve controle — en zij is schoon"` — in de kop;
* `"SFI komt voor élke feature rond 0,50 uit"` — een interpretatie van een fit
  die niet had plaatsgevonden;
* `"het nominale interval is ongeveer vijf keer zo smal"` — een verhouding als
  constante;
* `campaign['effective_n_conservative'] and 0.103` — een uitdrukking die er als
  berekening uitziet en **altijd** het literal `0.103` oplevert.

Dat de eerste bewering achteraf toevallig juist bleek, maakt het niet minder
ernstig. Het is hetzelfde defect als een toets die niet rood kan worden (§0.10),
alleen een stap later in de keten. Alles is nu afgeleid, en negen tests voeren
payloads in die de andere kant op wijzen.

### 4.2 Het schema was niet het gepre-registreerde schema

`purged_training_index` hield ook de events NA het testvenster vast. Dat is
AFML-purged-K-fold, en de pre-registratie draagt `scheme: purged_walk_forward`
met `train_bars: 500`. Gemeten: elke fold trainde op ~9.650 van de 10.330
events in plaats van op ~2.900.

Drie dingen braken tegelijk, en het derde is het ergste: de Data Adequacy Gate —
de poort die de fit AUTORISEERT — had precies die ~2.900 events per fold gemeten
(`[2867, 2934, 2947, …]`). Poort en run waren het oneens over wat een fold is.
En omdat het gefilterde signaal in stap 13 door de authoritative engine gaat,
werd de netto Sharpe gemeten op een filter dat deels op zijn eigen toekomst was
gefit — een lookahead in de ECONOMISCHE toets, niet alleen in de statistische.

Na de reparatie kwam de negatieve controle bovendien dichter bij 0,50 (0,5100
tegen 0,5216), wat de richting is die je verwacht wanneer toekomstinformatie de
fit verlaat.

### 4.3 Ik schreef een meting op die ik niet had gedaan

In `reports/phase7_divergence_map.md` §5 stond: *"Er is geen schrijfpad naar
schijf."* Onjuist. De breaker journaliseert zijn trips en weigert te starten;
het defect was subtieler en interessanter (§2.1). Ik had het bestaan van
`cb_log_path` gezien en niet doorgelezen.

In datzelfde rapport stonden aanvankelijk verkeerde aantallen: *"elf modules,
waarvan één gedeeld"*. Gemeten zijn het er **18 en 3**. Ik had geteld uit het
hoofd in plaats van het te meten — in een rapport waarvan de hele waarde in de
meting zit.

### 4.4 Ik stempelde documenten "geverifieerd" die ik niet had geverifieerd

Bij E-3 zette ik onder acht documenten een regel *"Geverifieerd tegen de
codebase op 2026-09-01"* met een motivering die deels een bewering was: dat de
audit-secties waren nagelopen (niet gedaan), dat de RISK\_CONTRACT-getallen tegen
de YAML waren gehertoetst (niet gedaan). Dat is exact het defect dat E-3 moet
opheffen, aangebracht dóór E-3. De stempels zeggen nu per document wat er wél en
niet is gemeten, en `docs/runbook.md` draagt géén verificatieclaim maar een
waarschuwing.

### 4.5 Twee poorten die het verkeerde maten

**De R²-guard.** De vol-forecastmonitor gaf een vals HALT-alarm op een perfecte
forecast: bij nulresiduen explodeert de Wald-statistiek. Mijn eerste guard keek
naar `r_squared` — en `forecast = rv × 0,7` heeft óók `r_squared = 1`, met
`beta = 1,43`. De guard onderdrukte dus precies het geval dat moest vuren.
Fitkwaliteit kan de twee niet onderscheiden; de coëfficiënten wel.

**De doc-scanner.** Mijn eerste versie telde elke padverwijzing als een claim,
waardoor het aantal bevindingen STEEG toen ik documenten corrigeerde: een
document dat uitlegt dat een pad niet meer bestaat, moet dat pad noemen. Zelfde
les als bij `min_confidence`: een test die het documenteren van een besluit
verbiedt, dwingt af dat besluiten ongedocumenteerd blijven.

### 4.6 Een test die statistisch niet klopte

Ik bouwde een "zuivere maar ruisige" forecast als `forecast = rv × ruis`. Dat is
errors-in-variables: de ruis zit in de regressor, wat attenuatie geeft
(`beta = 0,892`, `p = 7,2e-5`). Mincer-Zarnowitz meldde terecht vertekening; mijn
test noemde dat een vals alarm. Zuiverheid is `E[RV | forecast] = forecast`, dus
de ruis hoort in de realisatie.

### 4.7 Een val die ik zelf bijna zette

De nieuwe halt-state defaultet naar `artefacts/risk/halt_state.json`, en
`conftest.py` isoleerde alleen het OUDE logbestand. Elke test die een breaker
liet trippen, zou een echte halt-state hebben achtergelaten (no-go 15) — en
omdat die toestand per ontwerp niet verloopt, zou **elke volgende breaker in de
suite hebben geweigerd te starten**. Een testartefact zou de suite hebben
stilgelegd, en de volgende ontwikkelaar zou dat niet op een test hebben
gegokt.

### 4.8 Een bevinding die ik bijna publiceerde en die onjuist was

Ik stond op het punt te schrijven dat de live-keten posities sized met een
ongekalibreerde `eta = 0,142`, omdat `live/portfolio_controller.py`
`execution.market_impact` importeert. Bij het nalopen bleek het uitsluitend
`negative_skew_crisis_multiplier` te importeren. Het echte pad loopt via
`oms/paper_oms.py` → `execution/slippage.py`, en de bevinding is daarmee anders
en preciezer. Zonder die controle had er een onwaarheid in een governance-rapport
gestaan.

---

## 5. Wat er in bestaand werk is gevonden

Naast de eigen defecten, gevonden en gerepareerd of geregistreerd:

| Bevinding | Waar |
|---|---|
| Niets zette ooit `_max_gross_notional`; de limiet kon niet vuren | `live/execution_controller.py` |
| De HALT verliep na 24 uur zonder dat iemand hem accordeerde | `live/circuit_breaker.py` |
| `min_confidence` was dode configuratie in de ene module en een werkende poort in de andere | `live/` |
| `kelly_fraction` werd berekend en nooit toegepast; de docstring beweerde van wel | `live/execution_controller.py` |
| Het falsificatieregister citeerde een ledger-hash die niet bestaat (`5e5e07579587cddd`) | `docs/FALSIFICATION_REGISTER.md` |
| `pip install -e .` maakte twee commando's die crashen | `pyproject.toml`, `Makefile` |
| `architecture.md` beschreef twee DVC-stages die Phase 5 verwijderde | `docs/architecture.md` |
| `tca_methodology.md` noemde een output- en een consumentpad die geen van beide bestaan | `docs/tca_methodology.md` |
| `model_risk_policy.md` beweerde in de tegenwoordige tijd een sign-off-log bij te houden dat niet bestaat | `docs/model_risk_policy.md` |
| `GarchAdequacyConfig` declareerde `max_forecast_level_ratio` twee keer; de CI-mypy-stap stond daardoor rood | `schemas/config.py` |
| `phase6_h2_regime_benchmark` mist `hypothesis_ledger.json` in `deps` terwijl zijn rapport `M` afdrukt | `dvc.yaml` (DI-22) |

---

## 6. De actieve no-go-condities

Van de twintig no-go-condities zijn er bij afsluiting **drie actief**. De fase
is daarmee per definitie niet afgerond.

| # | Conditie | Waarom actief |
|---|---|---|
| 6 | `live/` bevat een tweede implementatie van executie-, accounting- of risicologica | §2.2; drie van de zes componenten |
| 11 | De 60-daagse teller is niet herstart na een crash of onverklaarde drift | vacuous: de teller is nooit gestart |
| 18 | Een permanent overgeslagen test houdt de suite ten onrechte groen | de vier `test_chaos.py`-skips staan er nog |

No-go 11 staat er als vacuous-actief en niet als "niet van toepassing": zolang
er geen klok loopt, is er ook geen bewijs dat hij correct herstart.

---

## 7. Verificatie bij afsluiting

```
python -m pytest tests -q        -> exact 4 failures, alle vier killgates
python -m ruff check src/        -> schoon (CI-scope)
python -m mypy --strict src/tradebot/schemas/config.py src/tradebot/utils/failfast.py
                                 -> schoon
python apps/freeze_monitoring.py --check   -> ongewijzigd
```

De vier failures zijn de pre-geregistreerde killgates op `cm_carry` en
`cm_tsmom`. Zij horen rood te staan. **Een suite met minder dan vier failures
betekent dat er een killgate is uitgeschakeld, niet dat er iets is opgelost.**

Nieuwe tests in deze fase: **116**, verdeeld over negen bestanden. Elke detector
draagt zijn eigen negatieve controle en is aantoonbaar rood geweest op de
toestand van vóór de reparatie.

---

## 8. De drie volgende stappen

1. **Sluit de twee ketens** (§2.2). Alles wat met pariteit of de klok te maken
   heeft, hangt hierachter.
2. **Maak Phase 6 stap 14 af** — HRP tegen Inverse Volatility, turnover-
   gecorrigeerd, onder de research-gate. De laatste openstaande deliverable van
   Stage C.
3. **Herschrijf het runbook en laat een tweede persoon het uitvoeren** (D-4,
   D12). Pas daarna heeft het starten van de 60-daagse klok zin.

*Voor de volledige stand: `docs/PROJECT_STATE.md`. Voor het reproductiepad:
`docs/REPRODUCTION.md`.*
