# FASE 10 — STAP 12: H-10.2, HET ONEVENWICHTIGE PANEEL

> **Verdict: DESCOPED.** Het mechanisme werkt — dit is de eerste hypothese in fase
> 10 die niet wordt weerlegd. Van de vier stop-criteria binden de eerste twee
> niet: er komen bars bij en de drempel daalt. Het derde bindt wél, en zijn vooraf
> geregistreerde actie is `descope` en niet `archive`: **de toegevoegde bars zijn
> smaller dan de bestaande, dus de drempeldaling overschat de gewonnen
> bewijskracht.**

**Gegenereerd:** 2026-09-14
**git_sha (meting):** `1abc99c`
**Meetvenster:** ontwikkelsample. Gebalanceerd 2021-11-15 → 2025-09-04 (1.390
bars); onevenwichtig **2020-11-21** → 2025-09-04 (**1.749 bars**)
**Pre-registratie:** bevroren, `planned_trials: 1`
**Bron:** `artefacts/governance/phase10_h10_2.json`, geproduceerd door
`apps/run_h10_2_unbalanced_panel.py` op
`src/tradebot/validation/phase10_unbalanced_panel.py` en
`src/tradebot/portfolio/weights.py`
**Trials:** 1 (`min_symbols_per_bar = 2`). Stage C staat nu op 5 van de 25;
`remaining(booked=5) = 20`
**Faseopdracht:** stap 12, R-1, R-2, R-3, R-10

---

## 1. Het resultaat van deze stap

De stapopdracht zegt het zelf: *"Rapporteer de gerealiseerde samplegroei in bars
én in jaren, en de nieuwe t = 2-hurdle bij de nieuwe N. Dat laatste getal is het
eigenlijke resultaat van deze stap."*

| grootheid | gebalanceerd (alle 6) | onevenwichtig (≥ 2) | verschil |
|---|---:|---:|---:|
| eerste bar | 2021-11-15 | **2020-11-21** | −359 bars eerder |
| bars (ontwikkeling) | 1.390 | **1.749** | **+359 (+25,83 %)** |
| jaren | 3,8082 | **4,7918** | +0,9836 |
| **t = 2-drempel** | 1,0249 | **0,9137** | **−0,1112** |
| SE van een ann. Sharpe | 0,5124 | 0,4568 | −0,0556 |

**De nieuwe drempel is 0,9137.** Dat is het antwoord.

"Bruikbaar" is hier `sigma.notna() & adv.notna()` — dezelfde conjunctie die het
huidige venster definieert. Niet de koersen: de risicolaag weigert een bar zonder
ex-ante volatiliteit, dus een koers alleen maakt geen bruikbare bar.

---

## 2. En de prijs ervan, want die staat in dezelfde meting

De 359 toegevoegde bars komen uit de periode waarin het universum nog niet vol
was. Zij dragen dus een **smallere cross-sectie**:

| breedte | bars in het nieuwe venster | waarvan toegevoegd |
|---:|---:|---:|
| 2 namen | 145 | **145** |
| 3 namen | 4 | **4** |
| 4 namen | 180 | **180** |
| 5 namen | 30 | **30** |
| 6 namen | 1.390 | — |

Gemiddelde breedte van de toegevoegde bars: **3,265** tegen **6,000** op het
bestaande venster. Over het hele nieuwe venster zakt zij van 6,000 naar **5,439**.

`2/√T` veronderstelt bars van **gelijke informatie-inhoud**. Deze zijn dat niet.
De drempeldaling van 0,1112 is dus rekenkundig echt en economisch gedeeltelijk
schijn, en dat is precies waarom het derde stop-criterium `descope` voorschrijft
in plaats van `archive`: de winst is bruikbaar, maar zij is kleiner dan het getal
suggereert. Het artefact draagt daarom een `hurdle_caveat`-veld naast de drempel,
zodat het getal niet los van zijn voorbehoud uit het JSON te lichten is.

Wat wél een prettige verrassing is: de toegevoegde bars zijn niet allemaal
2-breed. De helft (180 van 359) draagt vier namen. De verdunning is milder dan de
premisse deed vermoeden.

---

## 3. De stop-criteria, zoals gemeten

| criterium | eis | gemeten | bindt | actie |
|---|---|---:|:-:|---|
| `growth_is_not_material` | `bars_added > 0` | 359 | nee | archive |
| `hurdle_does_not_fall` | `delta < 0` | −0,1112 | nee | archive |
| `added_bars_are_not_thinner…` | `≥ 0` | **−2,7354** | **JA** | **descope** |
| `promotion_requires_all_gates_clear` | 0 bindend | 1 | nee | promote |

Het derde criterium stond vóór de meting in de pre-registratie met de aantekening
dat het **verwacht bindt** — de toegevoegde bars komen per constructie uit de
periode voordat het universum vol was. Het bindt, en de voorspelling is daarmee
uitgekomen in plaats van achteraf verzonnen.

---

## 4. De valkuil uit substap 12.6, gemeten en bevestigd

De stapopdracht waarschuwt: *"Een langere sample maakt de hurdle lager, maar de
baseline-Sharpes moeten dan opnieuw worden gemeten … Een Sharpe van 0,156 gemeten
op 3 symbolen × 3,42 jaar is niet vergelijkbaar met een Sharpe op 6 symbolen ×
4,4 jaar."*

Hermeten op `L0_vectorized`, ontwikkelsample, per venster. **Beide kolommen zijn
`NOT_ADMISSIBLE_AS_PROMOTION_EVIDENCE`**, zoals elke L0-meting:

| track | gebalanceerd (3,8082 j) | onevenwichtig (4,7918 j) |
|---|---:|---:|
| `long_only_equal_weight` | +0,2704 (t = 0,547) | **+0,7216 (t = 1,679)** |
| `long_only_risk_parity` | +0,2239 (t = 0,450) | **+0,7272 (t = 1,659)** |
| `xs_momentum_equal_weight` | +0,1298 (t = 0,281) | +0,0646 (t = 0,153) |
| `xs_momentum_risk_parity` | −0,1051 (t = −0,210) | −0,0343 (t = −0,075) |

**Lees dit niet als een verbetering.** Het patroon verraadt wat er gebeurt: de
twee **long-only** tracks springen met ruim +0,45, terwijl de twee
**cross-sectionele** tracks nauwelijks bewegen of dálen. Een langere sample die
een directionele periode toevoegt, beloont een long-only boek en zegt niets over
het cross-sectionele signaal. Het verschil is dus een uitspraak over **welke bars
zijn toegevoegd**, niet over het systeem — exact de niet-vergelijkbaarheid waar de
stapopdracht voor waarschuwt.

En het beslissende getal: **zelfs op de lagere drempel van 0,9137 blijft de beste
track met t = 1,679 onder 2.** De samplegroei maakt geen enkele track significant.
Zij maakt de vraag meetbaarder, niet beantwoord.

### `phase5_revaluation.json` is NIET als `superseded` gemarkeerd

Substap 12.6 vraagt dat wel. Dat is hier niet gedaan, en de reden is de meting in
§5: **L1, L2 en L3 kunnen op het ragged venster niet worden hermeten.** De oude
waarden `superseded` noemen terwijl er alleen voor L0 een nieuwe waarde bestaat,
zou het artefact half op het ene en half op het andere venster laten staan — een
interne inconsistentie die erger is dan de veroudering die de markering wil
oplossen. Bovendien vervángen de nieuwe L0-getallen de oude niet: zij meten een
ander venster, en §4 laat zien dat het verschil een marktperiode-effect is.
Beide reeksen horen mét hun venster naast elkaar, en dat is wat het artefact doet.

---

## 5. Waarom de ladder niet boven L0 komt — gemeten, niet gekozen

`run_all_layers` op het ragged venster faalt:

```
DataContractError: Ontbrekende of niet-eindige ex-ante volatiliteitsschatting.
De risicolaag valt NIET terug op een laatste bekende waarde en NIET op een
constante vol: een vol-schatting ontbreekt precies wanneer de markt iets doet
wat de schatter niet kent, en dat is het slechtste moment voor een aanname.
```

Dat is **correct gedrag** en geen bug. `backtest/phase5_baseline.py:170-184` bouwt
per bar `sigma_hat={s: float(sigma_hat.at[ts, s]) for s in symbols}` over **alle**
symbolen, en `RiskEngine.decide` weigert een niet-eindige vol expliciet en zonder
terugval. Een bar waarop één naam nog niet bestaat, kan de soevereine laag dus per
constructie niet bereiken.

Ragged doorvoeren tot L1 en hoger vraagt dat `RiskEngine.decide` een **per-bar
subset** van symbolen aanvaardt. Dat is een wijziging in `backtest/` plus een
besluit over het risicocontract, en beide staan buiten de bestandenlijst van deze
stap. De vol-reeks opvullen om er toch door te komen is precies de stille
degradatie die die foutmelding verbiedt, en is daarom niet gedaan.

**Bijkomend gemeten, en het is de diepere oorzaak.**
`src/tradebot/portfolio/equal_weight.py:104` doet
`np.nan_to_num(exposures.to_numpy(dtype="float64"), nan=0.0)`: afwezigheid wordt
op nul gezet **voordat** `normalise_weights` het paneel ooit ziet. De
gebalanceerde aanname zit dus stroomopwaarts ingebakken, niet in de laag die deze
stap aanpast. Voor de brutonormalisatie maakt dat niets uit (`nansum` over NaN is
gelijk aan `sum` over nullen), maar voor de breedtepoort en voor elke
covariantieschatter maakt het alles uit — en dat is eigenschap (2) van
`weights.py`: een `NaN` is afwezigheid, een `0.0` is een genomen besluit.

---

## 6. Correcties op de premisse van de stapopdracht (R-10)

Alle drie stonden **vóór de meting** in de pre-registratie, zodat zij geen
achteraf-verklaring konden worden.

| bewering | stapopdracht | gemeten |
|---|---:|---:|
| samplegroei | "ordegrootte 30 %" | **+25,83 %** (ontwikkeling); +20,60 % (vol venster) |
| nieuwe drempel | "ongeveer 0,95" | **0,9137** |
| ontwikkelsample, gebalanceerd | "3,42 jaar", drempel 1,081 | **3,8082 jaar**, drempel **1,0249** |

De 0,95 uit de stapopdracht hoort bij `min_symbols_per_bar = 3` (gemeten 0,9541),
niet bij de 2 die is geregistreerd. De 3,42 jaar komt nergens uit deze meting;
`docs/MEASUREMENT_CONTRACT.md` legt W_DEV vast op 1.390 bars = 3,8082 jaar.

**Wat wél klopt** — de per-symbool-dekking, en dat is de kern van de premisse:

| symbool | eerste bar | jaren gemeten | stapopdracht |
|---|---|---:|---:|
| BTCUSDT | 2020-03-26 | 6,42 | 6,41 ✓ |
| LINKUSDT | 2020-10-22 | 5,84 | 5,83 ✓ |
| ETHUSDT | 2021-03-16 | 5,44 | 5,44 ✓ |
| DOTUSDT | 2021-03-20 | 5,43 | — |
| AVAXUSDT | 2021-09-16 | 4,94 | — |
| SOLUSDT | 2021-10-16 | 4,86 | — |

Het derde symbool begint op 2021-03-16 tegen de opgegeven 2021-03-15: één dag, en
de strekking van de premisse staat.

---

## Bijlage — herleidbaarheid

**Trials: 1.** `min_symbols_per_bar = 2`, vastgelegd als `planned_trials: 1` in de
bevroren pre-registratie. De poort is `assert_within_budget(5, reset_path=...)`
tegen het bevroren `M_new = 25`; `remaining(booked=5) = 20`. Stage C heeft nu 5
van de 25 geclaimd (4 voor H-10.1, 1 hier).

De waarden 3, 4 en 5 zijn in de premissemeting wél doorgerekend. Dat kost geen
trial: het is dekkingsdiagnostiek op de datamatrix — geen rendement, geen Sharpe,
geen selectie — en de 2 stond vast vóór die meting, want de stapopdracht
registreert hem.

**Boekingsvorm.** `n_trials=0` met `amends`, om dezelfde reden als bij H-10.1:
`active_trial_count()` eist dat de lopende ledger-teller gelijk blijft aan de
bevroren `archived_total = 2776`, en een boeking met `n_trials≥1` laat die functie
weigeren en daarmee elke DSR in fase 10. Dat blijft een open mandaatbesluit; de
controle is niet verzwakt.

**`result: archived` en niet `descoped`.** `hypothesis_ledger.py:37` laat alleen
`{accepted, archived, falsified, interim}` toe. `DESCOPED` staat in
`metrics.verdict` en in de `notes`. Dezelfde beperking als bij H-10.1's
`UNPROVEN`.

**R-3.** Er is geen tweede brutonormalisatie bijgekomen.
`weights.py::normalise_weights` voegt uitsluitend de breedtepoort toe en geeft het
paneel daarna aan de bestaande `equal_weight.py::normalise_to_gross` door,
inclusief diens NaN-behandeling (`np.nansum`) en diens weigering door nul te
delen.

**R-1.** `tests/lookahead/test_unbalanced_panel_causality.py` toetst truncatie en
perturbatie, plus de eigenschap die dit paneel draagt: een bar mag niet veranderen
door of een symbool later verdwijnt. De negatieve controle is
`_gate_on_the_last_bar` — "neem de symbolen die we vandaag hebben" — en zij gaat
aantoonbaar rood, want de lookahead zit niet in de rekenregel maar in de
KOLOMkeuze.

**R-11, eerlijk.** `phase10_unbalanced_panel.py` is geschreven vóór zijn tests, en
dat is de omgekeerde volgorde. De zekerheid is op een andere manier gehaald: elke
assertie is nagelopen tegen drie opzettelijk gebroken varianten van de module
(mutatie op de teller `>=` → `>`, op de drempelformule `2/√T` → `1/√T`, en op de
toegevoegde-bars-verzameling), en elke mutatie maakt ten minste één test rood.
Dat staat ook in de docstring van het testbestand, in plaats van te worden
weggelaten.

Reproductie:

```
python apps/run_h10_2_unbalanced_panel.py
python -m pytest tests/unit/test_unbalanced_panel.py \
                tests/unit/test_panel_growth.py \
                tests/lookahead/test_unbalanced_panel_causality.py -q
```

Het JSON-artefact is de bron van elk getal in dit rapport en draagt bewust geen
`verdict`: het oordeel is een lezing van die getallen tegen de vooraf vastgelegde
beslisregel, en hoort hier te staan en niet in een bestand dat een renderer kan
overschrijven.

**Deze stap is geen DVC-stage.** Zij leest de ontwikkelsample en raakt het
poortsample niet aan (`holdout_lock.json` draagt nog `reads: []`), maar zij
produceert geen invoer voor een volgende stage; het artefact is een meting die het
rapport citeert.
