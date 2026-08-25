# PHASE 5 — EXIT REPORT

**Fase:** 5 van 7 — Execution, Backtesting & Sovereign Risk Integration · **Prioriteit:** P1
**Bindend brondocument:** `docs/ARCHITECTUUR_AUDIT_2026-08-22.md` §15, §15.1, §16, §16.1, §19, §23, §24, §26, §27
**Startpunt:** `b206895` · **Eindpunt:** `84273ca`
**Gegenereerd:** 2026-08-25

---

## 1. Executive status

> # CONDITIONAL PASS

Achttien van de twintig exit-criteria zijn gehaald en bewezen. Twee zijn
**gedeeltelijk** gehaald, en beide om een reden die deze fase niet kón oplossen:

| # | Criterium | Status |
|---|---|---|
| 6 | Impact gekalibreerd óf expliciet `IMPACT_UNCALIBRATED` | **GEHAALD** — `IMPACT_UNCALIBRATED` met conservatieve bovengrens |
| 7 | Execution realism aantoonbaar | **GEDEELTELIJK** — spread is een aanname, orderboekdiepte en limit-queues ontbreken |
| 9 | Backtest/live contract bit-identiek | **GEDEELTELIJK** — bewezen op de gedeelde context; `live/` zelf is Phase 7 |

De reden is in beide gevallen dezelfde en staat al in `docs/DATA_REGISTER.md`
§6: er is geen orderboek-L1/L2, geen trade-tape en geen bid/ask in de
gecertificeerde store. Dat is een eigenschap van de dataset, geen tekortkoming
van deze fase, en het is expliciet gelabeld in plaats van weggewerkt.

**Wat WEL onvoorwaardelijk is bewezen:** er bestaat vanaf nu één causale,
sovereign-governed simulatieketen, en er is aantoonbaar geen weg eromheen.

---

## 2. Architecture

### 2.1 De authoritative engine

`src/tradebot/backtest/engine.py::EventDrivenEngine`. De keten is:

```
MarketEvent -> FillEvent (orders van eerdere bars) -> FundingEvent
            -> SOVEREIGN RISK DECISION -> Order -> (volgende bar) Fill
            -> Accounting
```

Elke pijl is **structureel** afgedwongen, niet afgesproken:

| Eis | Hoe zij onmogelijk te omzeilen is |
|---|---|
| Geen order zonder soeverein besluit | `OrderRouter.build_orders()` neemt `RiskDecision` als verplicht eerste argument; er is geen tweede pad naar een `Order` |
| Geen besluit uit een ander regime | `build_orders` verwerpt een besluit met een afwijkende `config_hash` |
| Geen twee regimes in één run | `EventDrivenEngine.__init__` weigert als router- en engine-hash verschillen |
| Geen lookahead | `Order.__post_init__` verwerpt een order die op of vóór zijn beslisbar kan vullen |
| Geen verdwenen kosten | `Ledger.mark()` controleert na **elke** bar twee invarianten en crasht |
| Geen fill op mid | `SpreadModel` verwerpt een half-spread van nul |

### 2.2 Sovereign layer

Ongewijzigd overgenomen uit Phase 4 (`risk/engine.py`, `risk/contract.py`,
`risk/limits.py`, `risk/kill_switches.py`, `risk/vol_targeting.py`). Deze fase
heeft hem niet herontworpen — de fase-opdracht §2.1 verbiedt dat expliciet — maar
**bedraad**.

### 2.3 Execution path

`execution/context.py` · `execution/order_router.py` · `execution/impact_model.py`
· `execution/impact_calibration.py` · `execution/spread.py` (uitgebreid) ·
`backtest/accounting.py`.

### 2.4 Live contract

`ExecutionContext` is een ABC. `BacktestExecutionContext` en
`LiveExecutionContext` erven beide van `_SharedContext` en overschrijven **geen
enkele** contractmethode; een test faalt zodra dat verandert. De conversie van
`MarketSlice` naar `MarketState`/`RiskState` is één keer geschreven, waardoor
pariteit een eigenschap van de code is in plaats van een meetresultaat.

---

## 3. Sovereign wiring

### 3.1 Vertrekpunt

`reports/phase5_sovereign_wiring_audit.md`: `RiskEngine.decide()` had **drie**
aanroepsites buiten tests, alle drie analyse-instrumenten. **Elf** productiepaden
bereikten een order zonder de laag ooit aan te roepen.

En het gevaarlijke deel: **vier van de zeven lokale limieten waren RUIMER dan de
soevereine policy.**

| Grootheid | Sovereign | Lokaal | Locatie |
|---|---:|---:|---|
| Per-asset cap | 0,25 | **2,00** | `backtest/tracks.py` |
| ADV-participatie | 0,01 | **0,10** | `execution/market_impact.py` |
| Gross (backtest) | 1,50 | **2,00** | `backtest/per_side.py`, `bidirectional.py` |
| Gross notional (live) | 1,50 × equity | **`inf`** | `live/execution_controller.py` |

### 3.2 Modules die nu bedraad zijn

| Module | Wat er is gebeurd |
|---|---|
| `backtest/engine.py` | **nieuw**, sovereign is verplicht bij constructie |
| `execution/order_router.py` | **nieuw**, `RiskDecision` verplicht, hash geverifieerd |
| `portfolio/constraints.py` | `max_weight`/`max_leverage` zijn adapters naar `conf/risk/` |
| `portfolio/optimizer.py` | `constraints` verplicht; default verwijderd |
| `live/portfolio_controller.py` | `constraints` verplicht; lokale `max_weight=0.40` weg (C5 gesloten) |
| `live/engine.py` | bouwt zijn fallback uit `from_risk_config()` |

### 3.3 Lokale constraints verwijderd

| # | Constraint | Hoe |
|---|---|---|
| B1-B3 | leverage- en gross-caps in de drie legacy-engines | met de engines verwijderd |
| B5 | `AssetTrack.per_asset_cap = 2.0` | met `tracks.py` verwijderd |
| D1 | `PortfolioConstraints.max_weight = 0.40` | adapter |
| D2 | `PortfolioConstraints.max_leverage = 1.00` | adapter (waarde was **fout**: 1,00 vs 1,50) |
| D5 | impliciete toepassingsvolgorde | expliciet in `CONSTRAINT_ORDER` |
| C5 | `live/portfolio_controller.py` `max_weight = 0.40` | uit `from_risk_config()` |

### 3.4 Lokale constraints die blijven, met reden

De regel: een constraint mag lokaal blijven wanneer hij **(a)** uitsluitend
verkleint en **(b)** een kosten- of mechanicavraag beantwoordt in plaats van een
risicovraag.

| Constraint | Klasse | Reden |
|---|---|---|
| `min_weight` | execution-only | dust-drempel; een positie van 0,4 % kost meer aan fees dan zij bijdraagt |
| `max_turnover` | execution-only | kostenbeslissing: tracking error tegen transactiekosten |
| `VenueSpec.funding_cap_abs` | market-mechanics | Bybit begrenst funding zélf; een backtest die dat niet doet, modelleert een markt die niet bestaat |
| `VenueSpec.min_notional` | market-mechanics | de venue accepteert niets kleiners |
| fat-finger gate (`live/`) | execution-only | sanity check op één order, niet op de portefeuille |
| `gross_target` (baseline) | conventie | vergelijkingsschaal tussen tracks binnen één rapport |

### 3.5 Bewijs

`tests/integration/test_sovereign_wiring.py` — 33 tests, één per §16-eigenschap,
plus een repository-brede **statische audit** die het authoritative pad scant op
lokaal gedefinieerde limietnamen. Die audit heeft zelf een test die bewijst dat
hij een schending detecteert, zodat hij niet tot decoratie kan verworden.

### 3.6 Wat bewust naar Phase 7 gaat

| # | Locatie | Wat |
|---|---|---|
| C1/C2 | `live/circuit_breaker.py` | eigen drawdown-breaker, **in-memory** HALT-state |
| C3/C4 | `live/execution_controller.py` | eigen notional-limieten, `max_gross_notional = inf` |
| C6 | `live/execution_controller.py` | `min_confidence = 0.55` — modelvertrouwen als sizingparameter |

---

## 4. Execution realism

| Component | Status | Herkomst |
|---|---|---|
| Fees (maker/taker) | **gemodelleerd** | `conf/execution/fees.yaml`, Bybit VIP-0 |
| Funding | **gemodelleerd** | venue-cap ±2 %, per-bar settlement |
| Latency | **gemodelleerd** | `venue.latency_bars = 1`, afgedwongen in `Order` |
| Marktimpact | **gemodelleerd** | `IMPACT_UNCALIBRATED`, bovengrens `eta = 2,99` |
| Spread | **aanname** | `SPREAD_ASSUMED`, 1,0 bp |
| Participatielimiet | **gemodelleerd** | uit `risk.adv_participation_cap` |
| Partial fills | **gemodelleerd** | participatie × bar-volume |
| Geweigerde orders | **gemodelleerd** | `RejectReason` enum |
| Orderboekdiepte | **ONTBREEKT** | geen L2-data |
| Limit-order queues | **ONTBREEKT** | geen diepte, geen intra-bar tijdas |
| Cancellations | **ONTBREEKT** | idem |

### 4.1 Impact — `IMPACT_UNCALIBRATED`

`eta` is **niet identificeerbaar** zonder eigen orders en hun gemeten
prijsrespons. De afgeleide bovengrens: bij `Q = V` geeft het model
`Impact = eta * sigma_d`, en de grootste beweging die die dag optrad is de
dagrange. Over **11.660** symbool-dagen is de p95 daarvan **2,9919**;
`kappa_d = 0,6720`.

**Het rapport zegt er eerlijk bij dat deze grens weinig informatief is.** Voor
een Brownse beweging is de verwachte dagrange `1,60 sigma`; de gemeten mediaan
is **1,42** en de p95 per symbool ligt tussen **2,93 en 3,11** terwijl de
dagvolumes ordes van grootte verschillen. Hij meet volatiliteit, niet
liquiditeit.

Hij is bruikbaar omdat hij op dit boek **niet bindt**: bij een participatie van
~1e-7 blijft de impact onder 1 bp.

### 4.2 Spread — `SPREAD_ASSUMED`

Corwin-Schultz (2012) is geïmplementeerd en op deze data **verworpen**: mediane
half-spreads van 32,6 tot 65,9 bp met 33-38 % negatieve schattingen. Op
BTC-perps is ~33 bp ongeveer 50× te hoog. De estimator identificeert spread uit
het verschil tussen een- en tweedaagse ranges, en bij crypto-volatiliteit
domineert ruis dat verschil. Zie `docs/ARCHITECTURAL_DECISIONS.md` AD-3.

---

## 5. Accounting

`backtest/accounting.py` — dubbele boekhouding met twee invarianten, na **elke**
mutatie gecontroleerd:

```
assets == liabilities + equity
equity - initial_equity == realized + unrealized - fees - funding
```

Geen van de vier legacy-engines had een kasregister; alle vier deden
`equity *= (1 + r - kosten)`.

**Failure cases:** nul. Over alle testruns en de volledige
baseline-herwaardering (1.743 bars × 4 tracks) is geen enkele
`AccountingError` opgetreden.

**29 tests**, inclusief property-tests over willekeurige fill-reeksen en de
flip-casus (long naar short in één fill) die geen legacy-engine modelleerde.

---

## 6. TCA

### 6.1 Sluiting

Bewezen met een **schaduwboekhouding**: dezelfde fills opnieuw geboekt tegen hun
arrival price, zonder fee, tegen dezelfde markprijzen. Het verschil tussen de
twee eindequities moet exact gelijk zijn aan `spread + impact + fees`.

```
residu < 1e-6 × startequity      (tolerantie 0,01 bp uit conf/tca/default.yaml)
```

### 6.2 Decompositie

`spread` · `impact` · `fees` · `funding` · `timing`.

`timing` wordt **gerapporteerd maar telt niet mee** in de sluitingsidentiteit.
Het is de opportuniteitskost van niet-gevulde hoeveelheid — contrafeitelijk
geld dat de kas nooit heeft verlaten. Meetellen zou de roundtrip laten sluiten
op een bedrag dat nergens is betaald.

### 6.3 De negatieve test

Een test blanco't het fee-totaal en bevestigt dat de sluiting dán faalt. Een
identiteit die ook klopt wanneer je er kosten uit weglaat, toetst niets.

---

## 7. Parity

### 7.1 Legacy parity

`reports/phase5_engine_diff.md` corrigeert eerst de premisse: er waren geen vier
concurrerende engines. Er waren **drie** engines, één gedeelde hulpbibliotheek en
een **vijfde**, door de opdracht niet genoemd pad (`baseline_runner.py`).

Pariteit is per-bar RMSE tegen de vectorized conventie op elke lag:

| seed | lag 1 | **lag 2** | lag 3 |
|---|---:|---:|---:|
| 1 | 17,49 bp | **1,26 bp** | 20,38 bp |
| 42 | 19,50 bp | **1,03 bp** | 19,03 bp |
| 20260825 | 18,71 bp | **1,06 bp** | 22,43 bp |

De engine matcht **`shift(2)`** met 15-20× scheiding. `baseline_runner.py`
hanteert `shift(1)`, wat neerkomt op *"beslis op de close van `t`, voer uit op
DIE close"*.

> **De Phase 3-baseline neemt aan dat je kunt handelen op de close waarop je
> besluit.** Dat is de stilzwijgende aanname die deze fase zichtbaar maakt.

Het residu van ~1 bp is kwantiteit-versus-gewicht-drift.

### 7.2 Wat het schrijven van dit bewijs opleverde

De eerste versie vergeleek **eindequity** en koos op één van drie seeds de
verkeerde lag: twee conventies kunnen toevallig op hetzelfde eindpunt uitkomen.
En hij draaide op een constante `a_t = 1`, waarmee vol-targeting een bijna
vlakke gewichtenreeks geeft en `shift(1)` en `shift(2)` 0,448 tegen 0,455 bp
schelen — groen zonder iets aan te tonen. Beide zijn gerepareerd vóórdat het
bewijs de verwijdering licenseerde.

### 7.3 Backtest/live replay parity

13 tests. Op een gedeelde replay zijn equity-curve, elk risicobesluit, elke
order, elke fill en elke boekstaat **bit-identiek**. Het enige verschil is het
`context_kind`-veld in het auditspoor.

---

## 8. Cluster feasibility

`reports/phase5_cluster_concentration_audit.md`. De stelling uit de
fase-opdracht is gereproduceerd: **3,26 %** gerealiseerde boekvolatiliteit tegen
een target van 8 %, oftewel **40,7 %** attainment.

De diagnose gaat één stap verder. De labels dragen **geen informatie**:

| | Waarde |
|---|---|
| Scheiding binnen-vs-tussen | **+0,0113** |
| Bootstrap 95 %-BI | **[−0,0131, +0,0380]** — bevat nul |
| Teken per jaar | **wisselt**: −0,066 / +0,037 / +0,050 / +0,059 / −0,042 / −0,059 |
| Jaarlijkse herclustering | **zes verschillende partities in zes jaar** |

En de uitkomst was pervers: de clusterlimiet duwde `LINKUSDT` van 16,7 % naar
**40,0 %** — exact op `max_concentration` — en vernietigde 58 % van de
bruto-exposure. Een diversificatiebeperking die het meest geconcentreerde
toegestane boek produceerde.

### 8.1 Correctie en impact

| Grootheid | Vóór | Na |
|---|---:|---:|
| Boekvolatiliteit | 3,26 % | **7,50 %** |
| Target-attainment | 40,7 % | **93,8 %** |
| `LINKUSDT`-aandeel | 40,0 % | **16,7 %** |
| `cluster_cap` bindt | 100 % van de bars | **nooit** |

De resterende 6,2 % is de **comonotone bovengrens** in `book_sigma_hat`
(`sum |a_i| * sigma_i`) — bewuste conservatisme, hier gekwantificeerd op 1,073×.

### 8.2 §19 feasibility gate

| Rapportagepunt | Waarde |
|---|---|
| Target volatility | 8,00 % |
| Achieved volatility | 7,50 % |
| **Maximum feasible volatility** | **42,1 %** |
| Binding constraint | `vol_target` (100 % van de bars) |
| Reason for infeasibility | **n.v.t.** |

**`RISK_TARGET_INFEASIBLE` wordt NIET gerapporteerd.** De target is haalbaar;
de shortfall kwam van een limiet, niet van het universum.

### 8.3 Wat expliciet gezegd moet worden

Met één cluster geldt `effective_relative_cap(1, 0.60) = 1.0` en bindt
`cluster_cap` **nooit meer**. De limiet is **vacuous** op dit universum. Zij
blijft in `constraint_order` en wordt actief zodra er een aantoonbaar tweede
cluster is; een test bewijst dat. Tot dat moment is `max_concentration` (0,40)
de enige werkzame spreidingsbescherming.

---

## 9. Hypothesis ledger integrity

```
M_before = 2.715
register_risk_config(1b60cb664fbf9a2a)
M_after  = 2.715
```

Deze fase **wijzigde** de risicoconfiguratie (`47821e47fe2cec30` →
`1b60cb664fbf9a2a`). Zonder de scheiding zou die correctie de DSR van elke
Phase 3-track hebben verlaagd.

`tests/unit/test_risk_config_is_not_a_hypothesis.py` — 13 tests, waaronder één
die bewijst dat een **echte** hypothese `M` wél beweegt, zodat een kapotte
ledger de suite niet groen kan maken.

### 9.1 Een defect in mijn eigen werk

De eerste versie van de ledger-isolatietest schreef naar
`artefacts/governance/risk_config_registry.json` en liet daar een rij met
`git_sha="test"` achter. Een test die een governance-artefact vervuilt,
ondermijnt de auditbaarheid die hij hoort te bewaken. De rij is verwijderd, de
echte registratie staat onder haar commit-sha, en een nieuwe test faalt zodra er
ooit weer een test-rij verschijnt.

---

## 10. Baseline

### 10.1 Vier lagen

| Track | Laag | Netto | Vol | Sharpe | Max DD |
|---|---|---:|---:|---:|---:|
| `long_only_equal_weight` | L0 vectorized | −66,60 % | 71,5 % | +0,039 | 85,2 % |
| | L1 + sovereign | +2,98 % | 7,5 % | +0,119 | 13,8 % |
| | L2 + latency | +3,61 % | 7,6 % | +0,136 | 13,6 % |
| | **L3 + execution** | **−8,02 %** | **2,5 %** | **−0,695** | **8,3 %** |
| `xs_momentum_risk_parity` | L0 vectorized | −27,45 % | 20,5 % | −0,227 | 50,9 % |
| | L1 + sovereign | −0,17 % | 2,3 % | −0,005 | 5,3 % |
| | L2 + latency | −2,93 % | 2,3 % | −0,263 | 6,3 % |
| | **L3 + execution** | **−3,65 %** | **2,3 %** | **−0,331** | **6,6 %** |

Meetvenster: **1.743 bars**, 2021-11-15 t/m 2026-08-23.

### 10.2 De bevinding die eruit springt

`long_only_equal_weight` **halteert op 2022-05-10** — de LUNA-instorting — bij
een drawdown van **8,31 %** op de High-Water Mark, en de drawdown-breaker is een
eenrichtingsdeur, dus het boek handelt daarna nooit meer.

> Phase 3 rapporteerde een max drawdown van **83,5 %** voor deze track. Onder de
> soevereine policy is die drawdown **niet bereikbaar**: het boek wordt op 8 %
> stilgelegd. Het getal 83,5 % beschrijft een positie die geen
> sovereign-governed systeem had mogen aanhouden.

### 10.3 Kosten (L3, volledig venster)

| Track | Fees | Spread | Impact | Fill ratio | Orders | Geweigerd |
|---|---:|---:|---:|---:|---:|---:|
| `long_only_equal_weight` | € 49,89 | € 9,07 | € 36,51 | 99,99 % | 985 | 1 |
| `xs_momentum_risk_parity` | € 2.106,77 | € 383,05 | € 1.209,08 | 100,00 % | 9.655 | 3 |

**Fees domineren.** Impact is ~57 % van de fees ondanks de zeer conservatieve
`eta`; spread is het kleinst. Dat komt doordat het boek klein is ten opzichte van
het dagvolume — precies de reden dat de ruime `eta`-bovengrens bruikbaar is.

### 10.4 Interpretatie

1. **De risicolaag maakt geen alpha.** Alle vier de tracks blijven na kosten
   negatief. Dat is de Phase 3-conclusie, ongewijzigd.
2. **De risicolaag maakt het verlies wel veel kleiner.** L0 → L1 haalt
   `long_only_equal_weight` van −66,6 % naar +3,0 % en de drawdown van 85 % naar
   14 %. Dat is geen alpha; dat is een boek dat 10× kleiner is.
3. **Latency kost geld en het is meetbaar.** L1 → L2 kost
   `xs_momentum_risk_parity` 2,76 procentpunt.
4. **Een slechter resultaat is geen failure.** Fase-opdracht §20 zegt het
   expliciet: een realistischer resultaat is het doel.

---

## 11. Test status

### 11.1 Ratchets

| Ratchet | Baseline `b206895` | Nu `84273ca` |
|---|---|---|
| Verzamelde tests | 1.190 | **1.741** (+551) |
| Falende tests | 6 | **6** — dezelfde namen |
| Geskipte tests | 21 | 23 |
| Hardcoded literals | 330 | **312** |
| `audit_fallbacks.py --strict` | 36 bevindingen, 0 blokkerend | **36, 0 blokkerend** |
| `ruff check` (hele repo) | 272 | **239** |
| `mypy` op Phase 5-bestanden | n.v.t. | **clean** |

### 11.2 De zes falende tests

| Test | Reproduceerbaar? |
|---|---|
| `test_expansion_killgates.py::[KG-B1 in-sample-cm_carry]` | ja, deterministisch |
| `test_expansion_killgates.py::[KG-B2 residual alpha-cm_tsmom]` | ja |
| `test_expansion_killgates.py::[KG-B2 residual alpha-cm_carry]` | ja |
| `test_expansion_killgates.py::[KG-B3 out-of-sample-cm_carry]` | ja |
| `test_hypothesis_kernels.py::test_har_rv_forecast_non_negative` | ja, 6/6 runs |
| `test_hypothesis_kernels.py::test_evt_gpd_var_less_than_cf_at_extreme` | ja, 6/6 runs |

Alle zes zijn **pre-existent**: zij falen identiek op `b206895`, vóór elke
Phase 5-wijziging. De vier killgates zijn pre-registreerde go/no-go-poorten op
`cm_carry` en `cm_tsmom` die gefalsificeerd zijn — zij horen rood te staan.

### 11.3 §22 — de onverklaarde zevende failure

`b206895` legt vast dat één tussentijdse suite-run **7** failures rapporteerde
zonder namen, terwijl drie volledige runs er exact 6 gaven. De fase-opdracht
eist dat dit wordt onderzocht en niet weggewuifd.

**Onderzoek uitgevoerd:**

1. **Exacte tests geïdentificeerd** — de twee property-tests zijn de enige
   non-derandomised Hypothesis-tests in de suite:
   `@settings(max_examples=100, deadline=5000)` zonder `derandomize=True` en
   zonder geregistreerd profiel in `tests/conftest.py`.
2. **Afzonderlijk gereproduceerd** — 6 opeenvolgende runs van
   `tests/property/test_hypothesis_kernels.py`: **exact 2 failures, dezelfde
   namen, elke keer.**
3. **Met en zonder pytest-cache gedraaid** — `-p no:cacheprovider` maakt geen
   verschil.
4. **Drie volledige suite-runs op het startpunt** — exact 6, identieke namen.
5. **Drie volledige suite-runs op het eindpunt** — exact 6, identieke namen.

Totaal: **elf** volledige suite-runs over deze fase (drie op `b206895`, vijf
tussentijds, drie op `84273ca`), plus zes gerichte runs van de property-suite.
Elke run: exact 6 failures, dezelfde namen.

**Correctie op mijn eigen tussenresultaat.** Een eerdere meting in deze fase
leek te laten zien dat één van de twee property-tests alleen faalde mét de
Hypothesis-voorbeelddatabase. Dat was onjuist: mijn extractie gebruikte
`tail -3 | head -2` en ving daardoor slechts de eerste van twee `FAILED`-regels.
Beide falen deterministisch.

**Conclusie:** de zevende failure is in deze fase **niet gereproduceerd** in
elf volledige suite-runs plus zes gerichte runs. De meest waarschijnlijke
verklaring blijft de non-derandomised Hypothesis-configuratie — een zeldzaam
tegenvoorbeeld dat de random search doorgaans niet vindt — maar dat is een
**hypothese, geen vaststelling**, en zij wordt hier als zodanig genoteerd.

**Aanbeveling voor Phase 6:** registreer een Hypothesis-profiel met
`derandomize=True` in `tests/conftest.py`. Dat maakt de suite reproduceerbaar en
zou deze vraag definitief sluiten. Het is bewust **niet** in deze fase gedaan:
het verandert het gedrag van tests buiten de fase-scope, en de faalbasislijn
moest vergelijkbaar blijven met `b206895`.

### 11.4 Nieuwe tests

| Suite | Tests |
|---|---:|
| `tests/unit/test_accounting.py` | 29 |
| `tests/unit/test_impact_model.py` | 37 |
| `tests/unit/test_vectorized_not_admissible.py` | 25 |
| `tests/unit/test_portfolio_constraints_are_not_sovereign.py` | 26 |
| `tests/unit/test_risk_config_is_not_a_hypothesis.py` | 13 |
| `tests/integration/test_sovereign_wiring.py` | 33 |
| `tests/integration/test_tca_roundtrip.py` | 15 |
| `tests/integration/test_engine_parity.py` | 13 |
| `tests/integration/test_backtest_live_parity.py` | 13 |
| `tests/integration/test_cluster_feasibility.py` | 14 |
| `tests/lookahead/test_engine_causality.py` | 20 |

### 11.5 Open issues

1. **De killgate-failures zijn nooit opgelost.** Zij dateren van vóór Phase 3.
2. **De twee property-failures zijn echte bugs**, niet flakiness:
   `har_rv_forecast` kan negatief worden en `evt_gpd_var` levert een niet-eindige
   waarde op een degenerate invoer. Beide zitten in `risk/var.py` en
   `volatility/`, buiten de Phase 5-scope, maar zij horen niet blijvend rood te
   staan.

---

## 12. Wat er tijdens deze fase mis bleek in mijn eigen werk

Opgenomen omdat een exit-rapport dat alleen successen noemt, geen exit-rapport
is.

| # | Defect | Hoe gevonden | Status |
|---|---|---|---|
| 1 | `Ledger.equity` was overschrijfbaar; "equity is afgeleid" was een comment | eigen test faalde | `__slots__` toegevoegd |
| 2 | Ledger-isolatietest schreef naar het echte governance-register | eigen review | tmp_path + bewakende test |
| 3 | Pariteitstest vergeleek eindequity → verkeerde lag op 1 van 3 seeds | seed-parametrisatie | per-bar RMSE |
| 4 | Pariteitstest draaide op constante `a_t` → lag niet identificeerbaar | RMSE-analyse | `varying_exposure` |
| 5 | Wiring-test koos niet-bindende limietwaarden → bewees niets | test faalde | bindende waarden + toelichting |
| 6 | `apply_constraints` liet 2,9e-8 boven de cap staan (lineaire convergentie) | eigen test faalde | gesloten `water_filling_limit` |
| 7 | `day_start_equity` rolde niet → boek halteerde op 1.676 van 1.743 bars | baseline-run | rollen per kalenderdag |
| 8 | `per_side`-consumententest greep op platte tekst i.p.v. imports | test faalde | AST-analyse |
| 9 | Onjuiste tussenconclusie over Hypothesis-DB (extractiefout) | herhaalde meting | gecorrigeerd in §11.3 |

Defect 7 is het belangrijkste: het was op geen enkele synthetische replay van
60 bars zichtbaar en kwam alleen aan het licht door de echte baseline te draaien.

---

## 13. Definitieve exit criteria

| # | Criterium | Status | Bewijs |
|---|---|---|---|
| 1 | Exact één authoritative event-driven backtester | **GEHAALD** | `test_engine_parity.py::TestLegacyEngineInventory` |
| 2 | Legacy engines verwijderd na parity-bewijs | **GEHAALD** | commit `8071dc5` na `d12c3c5` |
| 3 | Vectorized uitgesloten als promotion evidence | **GEHAALD** | 25 tests; gate crasht |
| 4 | Accounting sluit | **GEHAALD** | 2 invarianten, elke bar, 0 failures |
| 5 | TCA sluit | **GEHAALD** | residu < 1e-6 × equity |
| 6 | Impact gekalibreerd óf `IMPACT_UNCALIBRATED` | **GEHAALD** | status + bovengrens + provenance |
| 7 | Execution realism aantoonbaar | **GEDEELTELIJK** | spread = aanname; geen L2-diepte |
| 8 | Causaliteit/lookahead bewezen | **GEHAALD** | 20 tests, incl. negatieve controles |
| 9 | Backtest/live contract bit-identiek | **GEDEELTELIJK** | bewezen op gedeelde context; `live/` = Phase 7 |
| 10 | Sovereign volledig wired voor Phase 5 runtime | **GEHAALD** | 33 tests + statische audit |
| 11 | `portfolio/constraints.py` geen concurrerende authority | **GEHAALD** | 26 tests |
| 12 | Backtest bevat geen eigen risk regime | **GEHAALD** | statische audit op het authoritative pad |
| 13 | Resterende lokale constraints geclassificeerd | **GEHAALD** | §3.4 + wiring audit §3.5 |
| 14 | Risk config in `registry/risk_registry.py` | **GEHAALD** | beide hashes geregistreerd |
| 15 | Risk config verandert `M` niet | **GEHAALD** | 13 tests |
| 16 | Clusterlabels en caps geaudit | **GEHAALD** | `phase5_cluster_concentration_audit.md` |
| 17 | Feasibility van 8 % aangetoond of infeasible gemarkeerd | **GEHAALD** | 93,8 % attainment; max feasible 42,1 % |
| 18 | Phase 3-baseline herberekend | **GEHAALD** | `phase5_revaluation.json`, 4 lagen |
| 19 | Alle failures geïdentificeerd en geclassificeerd | **GEHAALD** | §11.2, §11.3 |
| 20 | Exit report compleet | **GEHAALD** | dit document |

---

## 14. Absolute no-go conditions

| Conditie | Aanwezig? |
|---|---|
| Backtest past lokale risk limit toe zonder sovereign decision | **nee** — structureel onmogelijk |
| Portfolio sizing kan buiten sovereign policy limiteren | **nee** — adapters + `enforce_provenance` |
| Risk config wordt als hypothesis trial geteld | **nee** — 13 tests |
| Clusterlabels blokkeren de feasible space onopgelost | **nee** — gecorrigeerd, impact gemeten |
| TCA sluit niet | **nee** — sluit |
| Accounting sluit niet | **nee** — sluit |
| Impact valt stilzwijgend terug | **nee** — geen enkel default-argument |
| Lookahead aanwezig | **nee** — 20 tests |
| Authoritative engine niet uniek | **nee** — drie verwijderd |
| Parity niet bewezen | **nee** — 13 tests, drie seeds |
| Test failures niet geïdentificeerd | **nee** — alle zes benoemd en pre-existent |

**Geen enkele no-go-conditie is actief.**

---

## 15. Deliverables

| # | Deliverable | Status |
|---|---|---|
| 1 | `src/tradebot/backtest/engine.py` | opgeleverd |
| 2 | `src/tradebot/backtest/accounting.py` | opgeleverd |
| 3 | `src/tradebot/backtest/vectorized.py` | opgeleverd |
| 4 | `src/tradebot/execution/impact_model.py` | opgeleverd |
| 5 | `apps/calibrate_impact.py` | opgeleverd (65 LOC) |
| 6 | `src/tradebot/execution/spread.py` | uitgebreid met Corwin-Schultz |
| 7 | `src/tradebot/execution/slippage.py` | bestond; niet op het authoritative pad |
| 8 | `src/tradebot/execution/order_router.py` | opgeleverd |
| 9 | `src/tradebot/tca/post_trade.py` | uitgebreid met roundtrip-sluiting |
| 10 | `conf/execution/` | `fees.yaml` + **`impact.yaml`** |
| 11 | `conf/tca/default.yaml` | opgeleverd |
| 12 | `tests/integration/test_tca_roundtrip.py` | opgeleverd |
| 13 | `tests/integration/test_engine_parity.py` | opgeleverd |
| 14 | `tests/integration/test_backtest_live_parity.py` | opgeleverd |
| 15 | `tests/lookahead/test_engine_causality.py` | opgeleverd |
| 16 | `reports/TCA_CALIBRATION_REPORT.md` | opgeleverd |
| 17 | `reports/phase5_engine_diff.md` | opgeleverd |
| 18 | `reports/phase5_sovereign_wiring_audit.md` | opgeleverd |
| 19 | `reports/phase5_cluster_concentration_audit.md` | opgeleverd |
| 20 | `reports/phase5_exit_report.md` | dit document |
| 21 | `docs/ARCHITECTURAL_DECISIONS.md` | opgeleverd (10 entries) |
| 22 | `registry/risk_registry.py` + tests | bestond (Phase 4); 13 tests toegevoegd |

**Niet in de lijst, wel opgeleverd:** `execution/context.py`,
`execution/impact_calibration.py`, `backtest/phase5_baseline.py`,
`apps/run_phase5_baseline.py`, `tests/integration/test_sovereign_wiring.py`,
`tests/integration/test_cluster_feasibility.py`, `tests/unit/test_accounting.py`,
`tests/unit/test_impact_model.py`,
`tests/unit/test_vectorized_not_admissible.py`,
`tests/unit/test_portfolio_constraints_are_not_sovereign.py`,
`tests/unit/test_risk_config_is_not_a_hypothesis.py`.

---

## 16. Commits

| Sha | Bericht |
|---|---|
| `977b0c2` | `docs(backtest): forensic comparison of the legacy backtest engines` |
| `d0131a6` | `docs(risk): audit where the sovereign risk layer is and is not wired` |
| `4307089` | `docs(risk): measure why the book lands at 3.2% against an 8% vol target` |
| `d26baf9` | `feat(backtest): add double-entry accounting with two enforced invariants` |
| `bbcca28` | `feat(execution): make the impact parameter mandatory and calibrate what can be` |
| `f4352e9` | `feat(backtest): add the authoritative event-driven engine, sovereign-wired` |
| `d12c3c5` | `feat(risk): wire the sovereign layer into execution and prove it cannot be bypassed` |
| `2e288ea` | `style(tests): drop noqa directives ruff reports as unused` |
| `8071dc5` | `refactor(backtest): remove the legacy engines after the parity proof` |
| `766de6f` | `refactor(portfolio): stop L8 from carrying its own risk policy` |
| `aec701b` | `docs(phase5): record the decisions that are not readable from the code` |
| `84273ca` | `feat(backtest): re-run the Phase 3 baseline through the full Phase 5 chain` |

---

## 17. Phase 7 handoff

### 17.1 Wat nog in `live/` moet worden aangesloten

| # | Locatie | Wat er weg moet | Vervanger |
|---|---|---|---|
| C1 | `live/circuit_breaker.py:79-85` | eigen `max_drawdown_pct`, `max_intraday_drawdown_pct`, `max_daily_loss_pct` | `risk/kill_switches.py` via `ExecutionContext` |
| C2 | `live/circuit_breaker.py` | **in-memory** HALT-state — een procesherstart wist de halt | `risk.kill_switches.HaltStore` (bestaat, persistent, append-only journaal) |
| C3 | `live/execution_controller.py:243-267` | `_check_position_limits` | `risk.max_position_pct` + `risk.gross_cap` uit het besluit |
| C4 | `live/execution_controller.py:95` | `_max_gross_notional = inf` — **onbegrensd bij omissie** | idem |
| C6 | `live/execution_controller.py:61-63` | `min_confidence = 0.55` — modelvertrouwen als sizingparameter | **verwijderen**; audit §14 sluit dit uit |

### 17.2 Welke contracten al bestaan

| Contract | Bestand | Status |
|---|---|---|
| `ExecutionContext` (ABC) | `execution/context.py` | **klaar**, beide implementaties getest |
| `LiveExecutionContext` | `execution/context.py` | **klaar en getest**; alleen de bron ontbreekt |
| `MarketSlice` (replay-formaat) | `execution/context.py` | **klaar** |
| `RiskDecision` / `MarketState` / `RiskState` | `risk/contract.py` | klaar sinds Phase 4 |
| `OrderRouter` | `execution/order_router.py` | **klaar**, venue-agnostisch |
| `Ledger` | `backtest/accounting.py` | **klaar**, bruikbaar als live shadow book |
| `HaltStore` | `risk/kill_switches.py` | klaar sinds Phase 4 |

**Wat Phase 7 feitelijk moet doen:** `live/feed.py` een `MarketSlice` laten
leveren en `live/state.py` de boekstaat. De besluitvorming erboven verandert
niet.

### 17.3 Welke tests Phase 7 moet uitvoeren

1. `LiveExecutionContext` gevoed door `live/feed.py` produceert dezelfde
   `MarketState` als `BacktestExecutionContext` op dezelfde bars.
2. Een paper-run en een backtest over hetzelfde venster leveren identieke
   `RiskDecision`-reeksen.
3. `HaltStore` overleeft een procesherstart en de engine handelt niet door een
   halt heen.
4. `live/execution_controller.py` heeft geen enkel eigen limietveld meer —
   uitbreiding van de statische audit in `test_sovereign_wiring.py` met de
   `live/`-modules.
5. Het live shadow-ledger sluit tegen de exchange-balans binnen een vooraf
   vastgelegde tolerantie.
6. `min_confidence` bestaat niet meer in `live/`.

### 17.4 Wat Phase 6 zou moeten oppakken

1. **Hypothesis-profiel met `derandomize=True`** — sluit §11.3 definitief.
2. **De twee property-failures repareren** — het zijn echte bugs.
3. **Een DVC-stage voor de authoritative engine.** Stage 4 en 4.5 zijn
   verwijderd; de opvolger draait via `apps/run_phase5_baseline.py` en heeft nog
   geen stage, omdat de ML-track eerst een pre-registratie nodig heeft die op
   deze engine is geschreven.
4. **Orderboekdata ingesteren** — sluit exit-criterium 7 en Open Question 1.

---

## 18. Wat de volgende fase moet weten

1. **De soevereine laag is nu overal bedraad waar Phase 5 over gaat, en er is
   aantoonbaar geen weg eromheen.** De statische audit is het mechanisme dat dat
   zo houdt; breid hem uit in plaats van hem te omzeilen.
2. **De baseline is realistischer én slechter, en dat was het doel.** De
   belangrijkste verschuiving is niet een kostenpost maar een **halt**:
   `long_only_equal_weight` stopt op 2022-05-10 en handelt nooit meer.
3. **`IMPACT_UNCALIBRATED` en `SPREAD_ASSUMED` zijn geen tijdelijke labels.**
   Zij verdwijnen pas wanneer er orderboek- en quote-data is.
4. **De clusterlimiet is vacuous.** Op dit universum beschermt alleen
   `max_concentration` tegen concentratie. Zodra het universum verbreedt, wordt
   `cluster_cap` weer actief — en dan moet de labeling opnieuw worden gemeten,
   niet overgenomen.
5. **De risicolaag maakt nog steeds geen alpha.** Alle vier de tracks blijven
   netto negatief. Dat is de openstaande vraag van Phase 3 en geen enkele limiet
   lost hem op.
