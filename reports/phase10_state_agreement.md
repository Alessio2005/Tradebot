# FASE 10 — STAP 8: EWMA TEGEN GARCH, ZONDER PROXY

> **Status:** diagnostiek, expliciet niet-promoveerbaar. Er wordt in deze stap
> niets geselecteerd — geen estimator, geen kwantiel, geen horizon, geen
> parameter is op grond van onderstaande getallen gewijzigd — dus kost zij
> **nul trials** (`trials 0  selecteert niets`, eerste regel van de stdout).

**Gegenereerd:** 2026-09-11
**git_sha (meting):** `c4811a4`
**Meetvenster (ontwikkelsample):** 2021-11-16 → 2025-09-04, **1.389
rendement-bars**, 6 symbolen
**Vergelijkingsvenster (waarover elk getal hieronder loopt):** 2023-12-06 →
2025-06-08, **551 bars** — zie §1, dit is niet hetzelfde venster
**Bron:** stdout van `apps/run_state_agreement.py` op
`src/tradebot/regime/state_agreement.py`
**Faseopdracht:** stap 8 (`Prompts-fases/fase_10_herstart_dagbars.md`),
rulings P35, P36 (geamendeerd), P37, R-8, R-10

---

## 0. De beslisregel, zoals zij VOORAF stond

> **Beslisregel, vooraf (substap 8.5).** Wijzen de twee modellen op ≥ 95 % van
> de bars dezelfde toestand toe **en** is `fraction_bars_with_decision_change`
> < 5 %, dan is de keuze tussen hen economisch irrelevant en is H1 ook op de
> toestandsas beslecht — zonder proxy, zonder DM-toets en zonder trial.
> EWMA(0,94) blijft de productie-estimator.
>
> Blijft de overeenstemming daaronder, dan is het verschil een **kandidaat** en
> gaat het als hypothese naar Stage C, met de trial-kosten die daarbij horen.
> Het gaat **niet** stilzwijgend mee als "betere estimator".

Deze regel is vastgelegd vóór de meting. Zij staat als constante in de app
(`MIN_IDENTICAL, MAX_DECISION_CHANGE = 0.95, 0.05`), zodat de run haar niet kan
herformuleren op het getal dat eruit komt. Onder §3 staat wat eruit kwam;
onder §4 staat welke tak dat is. Er is niets bijgesteld om de regel te halen.

---

## 1. De dekking gaat vóór het overeenstemmingsgetal (R-10, ruling P37)

`walk_forward_variance_forecasts` laat elke bar buiten een testvenster op NaN
staan en vult niet. Het GARCH-paneel heeft daardoor gaten die het EWMA-paneel
niet heeft, en die gaten zitten aan **beide** uiteinden van de ontwikkelsample.

| grootheid | vergeleken | totaal | aandeel | weggevallen |
|---|---:|---:|---:|---:|
| (bar, symbool)-**cellen** | **3.306** | 8.334 | **39,67 %** | 5.028 |
| **bars** (elke naam bekend) | **551** | 1.389 | **39,67 %** | 838 |

**Elk getal in dit rapport loopt over 551 bars, 2023-12-06 tot 2025-06-08 —
niet over de 1.389 bars van de ontwikkelsample.** Dat venster ligt niet aan het
eind, waar een lezer het zou zoeken. De opbouw van de 838 weggevallen bars:

- **750 bars aan de voorkant** (2021-11-16 → 2023-12-05). Het eerste
  testvenster van de walk-forward begint op bar 500 (2023-03-31), en daarna
  vraagt `assign_by_variance` nog 250 waarnemingen voordat zijn expanding
  kwantiel bestaat. De eerste GARCH-toestand valt dus op bar 750.
- **88 bars aan de achterkant** (2025-06-09 → 2025-09-04). Na de achtste fold
  resteren er 89 bars, minder dan `test_bars = 100`, dus er is geen negende
  fold en er zijn geen forecasts. Het vergelijkingsvenster eindigt daarmee
  **drie maanden vóór** de ontwikkelsample.

De twee noemers zijn verschillende grootheden (P37: de bar-noemer eist dat
*elk* symbool bekend is en is met zes namen de strengere) maar komen hier op
hetzelfde percentage uit. Dat is geen toeval en ook geen samenvallen van twee
maten: de foldstructuur is identiek voor alle zes de namen, dus het GARCH-gat
staat op elke naam op exact dezelfde bars, en dan is "minstens één naam
onbekend" hetzelfde als "alle namen onbekend". Op een paneel waar de folds per
naam zouden verschillen, lopen deze twee uiteen.

Er is nergens gevuld, nergens voorwaarts gevuld en nergens op EWMA
teruggevallen waar GARCH ontbreekt.

### 1.1 Twee van de zes GARCH-panelen zijn onder H1's eigen poort niet vergelijkbaar

| symbool | garch-bars | folds | geconvergeerd | randoplossingen | `comparable` (H1) |
|---|---:|---:|---:|---:|:--|
| BTCUSDT | 800 | 8 | 100,0 % | **75,0 %** | **NEE** |
| ETHUSDT | 800 | 8 | 100,0 % | **75,0 %** | **NEE** |
| SOLUSDT | 800 | 8 | 100,0 % | 0,0 % | ja |
| AVAXUSDT | 800 | 8 | 100,0 % | 0,0 % | ja |
| LINKUSDT | 800 | 8 | 100,0 % | 0,0 % | ja |
| DOTUSDT | 800 | 8 | 100,0 % | 0,0 % | ja |

Elke fit convergeert. Maar op BTCUSDT en ETHUSDT staat **zes van de acht**
geconvergeerde fits op de IGARCH-rand, tegen een
`max_boundary_solution_ratio` van **0,10** in `conf/model/adequacy.yaml`. De
kolom `comparable` is niet hier uitgerekend maar gelezen uit
`ConvergenceSummary.comparable`, de poort die de H1-campagne zelf hanteert
(R-3): op een randoplossing bestaat de onvoorwaardelijke variantie niet en is
de forecast een random walk in variantie.

**Wat dit wel en niet betekent.** Deze stap sluit die twee namen **niet** uit.
Een naam weglaten omdat zijn fits lelijk zijn, zou van deze telling een keuze
maken, en dan zou stap 8 selecteren in plaats van tellen. Maar het getal in §3
mag niet zonder dit label worden gelezen: op een derde van het paneel komt de
uitdager uit fits die H1's eigen machinerie weigert te vergelijken, en een
random walk in variantie dwaalt — precies het gedrag dat onder een expanding
kwantiel systematisch extra HOOG-toewijzingen oplevert (§3.2).

---

## 2. Wat er is vergeleken, en waartegen

Twee σ̂-panelen op de ontwikkelsample, beide toegewezen met **dezelfde bevroren
kwantielregel** — identieke `low_q = 0,25`, `high_q = 0,75`,
`min_periods = 250`, `lag = 1` uit `conf/model/regime.yaml`:

| | verdediger | uitdager |
|---|---|---|
| model | EWMA(0,94), `ewma_volatility_panel` via `load_phase6_universe` | walk-forward GARCH(1,1)-t, `walk_forward_variance_forecasts`, h = 1 |
| foldstructuur | n.v.t. (causaal recursief) | `WalkForwardCV(500/100/100, rolling, embargo 5)` — dezelfde constructie als `apps/run_vol_competition.py` (R-3) |
| eenheid | volatiliteit | variantie (wortel genomen, zie hieronder) |

**Ruling P35 — dit is een kwantielREGEL en geen niveaudrempel.**
`assign_by_variance` leidt zijn drempels af uit de reeks die hij krijgt, dus
elk model wordt tegen zijn **eigen** verdeling gescoord. GARCH tegen de schaal
van EWMA leggen zou een puur niveauverschil tussen de estimators als
toestandsverschil rapporteren, en dat is niet de vraag van deze stap.

Twee gevolgen, en het tweede is een grens op de claim (R-8):

1. De vergelijking is **invariant onder elke monotone herschaling** van σ̂. Dat
   de GARCH-kant variantie levert en de EWMA-kant volatiliteit, kan de uitkomst
   dus per constructie niet raken; de wortel in `_garch_sigma` is leesbaarheid
   en geen schaalafstemming. Er is er geen gedaan en er was er geen nodig.
   `test_the_comparison_is_invariant_under_a_monotone_rescaling` maakt dat
   meetbaar in plaats van beweerd.
2. Daarom **kan** deze meting ook niet zien dat "GARCH systematisch hoger staat
   dan EWMA". Die onzichtbaarheid is voor de toestandsvraag de juiste
   eigenschap — AD-16 mat dat L7 een cross-sectioneel uniforme schaal er weer
   uitdeelt — maar zij blijft een grens op wat elk getal hieronder mag beweren.

---

## 3. De uitkomst

### 3.1 Overeenstemming op celniveau (clausule 1 van de beslisregel)

| grootheid | waarde | teller/noemer |
|---|---:|---|
| `fraction_identical` (ruw) | **0,6691** | 2.212 / 3.306 |
| **Cohens κ** | **0,4370** | — |

Toevalsovereenstemming p_e = 0,4122; κ = (0,6691 − 0,4122) / (1 − 0,4122) =
0,4370. Dat κ zoveel lager ligt dan het ruwe percentage is precies waarom hij
verplicht is: van de 66,9 % "overeenstemming" is 41,2 procentpunt wat twee
onafhankelijke toewijzers met deze bezettingen ook zonder enige gedeelde
informatie zouden halen.

Per symbool loopt de ruwe overeenstemming van **0,561** (ETHUSDT) tot
**0,744** (BTCUSDT) — het veld is het onderling niet eens, net als in stap 6.

### 3.2 De verwarringsmatrix, gepoold (rijen EWMA, kolommen GARCH)

| EWMA \ GARCH | LAAG | NORMAAL | HOOG | totaal | bezetting |
|---|---:|---:|---:|---:|---:|
| **LAAG** | 334 | 201 | 2 | 537 | 16,24 % |
| **NORMAAL** | 148 | **1.369** | **697** | 2.214 | 66,97 % |
| **HOOG** | 1 | 45 | 509 | 555 | 16,79 % |
| **totaal** | 483 | 1.615 | 1.208 | 3.306 | |
| **bezetting** | 14,61 % | 48,85 % | **36,54 %** | | |

De grootste off-diagonaal is **NORMAAL → HOOG: 697 cellen**. GARCH wijst op dit
venster HOOG toe op 36,5 % van de cellen tegen 16,8 % voor EWMA — meer dan een
factor twee. Onder een expanding kwantiel op 0,75 is dat geen niveau-uitspraak
(zie P35 hierboven, die is onzichtbaar) maar een uitspraak over het **verloop**
van de GARCH-reeks ten opzichte van haar eigen geschiedenis: een σ̂ die door het
vergelijkingsvenster heen omhoog loopt ten opzichte van zijn eigen expanding
verdeling, wordt structureel als HOOG gelabeld. Dat is hetzelfde gedrag als een
random walk in variantie, en op twee van de zes namen is de forecast dat
letterlijk (§1.1).

LAAG en HOOG verwisselen bijna nooit rechtstreeks (2 en 1 cellen): net als de
overgangsmatrix in stap 6 loopt de onenigheid vrijwel altijd via NORMAAL.

### 3.3 Besluitwijziging (clausule 2 van de beslisregel)

De poort is `flat_states = {HIGH}`. Die waarde is **niet hier gekozen**: plan
stap 13.3 draait het tweede spoor met `gate_by_state(flat_states={HIGH})` en
stap 13 boekt daar exact één trial voor met de toelichting *"één configuratie:
`flat_states = {HIGH}`. Géén zoektocht over `flat_states`"*. Zij staat in
`state_agreement.PRE_REGISTERED_FLAT_STATES` mét die bronvermelding.

| telling | bars | van | aandeel |
|---|---:|---:|---:|
| **`n_decision_changes`** — de poort valt anders uit onder `{HIGH}` | **381** | 551 | **69,15 %** |
| `n_any_state_changes` — welke toestandswissel dan ook (**bovengrens**) | 489 | 551 | 88,75 % |

**De verdictregel leest de eerste rij** (geamendeerde P36). De tweede staat
ernaast als conservatieve **bovengrens**: een poort kan toestanden alleen
samenvoegen, nooit splitsen, dus 88,75 % begrenst de besluitwijziging onder
*elke* denkbare toestandspoort. Beide richtingen expliciet:

- Was de bovengrens onder 5 % gebleven, dan had de conclusie gegolden voor elke
  poort die later ook maar gekozen zou worden. Dat is **niet** het geval.
- Omgekeerd bewijst een falende bovengrens niets over een specifieke poort —
  maar dat is hier academisch, want de gepre-registreerde poort zelf faalt met
  69,15 % tegen een grens van 5 %.

Op celniveau zegt precies één van beide modellen HOOG in **745 van de 3.306**
cellen (22,53 %); daarvan is 699 keer GARCH degene die HOOG zegt en 46 keer
EWMA. De poort-overeenstemming per cel is dus 77,47 %, ruim onder de 95 % die
clausule 1 al niet haalt.

---

## 4. Het oordeel

| clausule | eis (vooraf) | gemeten | |
|---|---|---:|:--|
| 1 | `fraction_identical` ≥ 95 % | **66,91 %** | **MIST** (−28,1 pp) |
| 2 | `fraction_bars_with_decision_change` < 5 % | **69,15 %** | **MIST** (13,8× de grens) |
| — | bovengrens, elke poort | *(geen eis; informatief)* | 88,75 % |

**Beide clausules missen, en niet nipt.** De regel vraagt ≥ 95 % en meet
66,91 %; zij vraagt < 5 % en meet 69,15 %. Dit wordt hier niet gelezen als
"dicht bij" en de regel wordt niet bijgesteld.

**Dit is dus de tweede tak van de beslisregel.** Het verschil tussen EWMA(0,94)
en de walk-forward GARCH(1,1)-t op de toestandsas is een **KANDIDAAT** en gaat
als hypothese naar **Stage C, met de trial-kosten die daarbij horen**.

Wat daaruit **niet** volgt, en wat deze stap expliciet weigert te doen:

- **GARCH is niet gepromoveerd.** Er staat hier geen enkel getal dat zegt welk
  van de twee modellen *beter* is. Deze stap telt of ze het eens zijn; wie van
  beiden gelijk heeft, is precies de vraag die zonder variantieproxy niet te
  beantwoorden is en die H1 daarom heeft laten liggen (DI-18, AD-23). Een hoge
  onenigheid is geen bewijs voor de uitdager.
- **EWMA(0,94) blijft de productie-estimator.** Niet omdat deze meting hem
  gelijk geeft, maar omdat hij dat al was en er niets is dat hem verplaatst.
  Een wissel op grond van dit rapport zou een selectie zijn op een diagnose die
  nul trials heeft geboekt, en dat is precies de stille promotie die deze fase
  bestaat om te voorkomen.
- **H1 is op de toestandsas NIET beslecht.** De eerste tak van de regel
  ("economisch irrelevant, dus beslecht zonder trial") is de tak die niet is
  gelopen.

---

## 5. Wat deze meting niet vaststelt (R-8)

**De niveau-as is per constructie onzichtbaar.** Zie §2: elke monotone
herschaling van σ̂ laat de toewijzing ongemoeid. "GARCH staat systematisch hoger
dan EWMA" is met dit instrument niet te zien, in geen van beide richtingen.

**De twee expanding kwantielen rusten op ongelijke geschiedenissen, en dat is
niet weggenomen.** Op de eerste vergeleken bar (2023-12-06) rust het
EWMA-kwantiel op 750 waarnemingen vanaf 2021-11, het GARCH-kwantiel op 250
vanaf 2023-03-31; op de laatste bar 1.300 tegen 800. De referentieverdeling van
de uitdager is dus altijd een striktere, latere deelverzameling van die van de
verdediger. Dat volgt uit ruling P35 — elk model tegen zijn eigen verdeling —
maar het betekent dat een deel van de gemeten onenigheid toe te schrijven is
aan een verschil in **referentieperiode** en niet aan een verschil in
**modeldynamiek**, en dit rapport kan die twee niet uit elkaar halen. Er is
géén tweede variant gedraaid om dat te scheiden: een alternatieve
vensterkeuze zoeken nádat de regel rood staat, is de bijstelling die substap
8.5 verbiedt. Het scheiden van die twee is een legitieme vraag voor **Stage C**
en hoort daar met zijn eigen trial-kosten te landen.

**Eén specificatie, één horizon, één poort.** GARCH(1,1)-t op h = 1, en verder
niets. h = 1 is gekozen omdat de toestand van bar `t` σ̂ op `t-1` leest — dat is
de eenstapsforecast — en niet omdat hij het beste resultaat gaf; er is geen
tweede horizon gedraaid. Hetzelfde geldt voor de drie andere leden van
`GARCH_FAMILY` en voor `flat_states`: één configuratie, geen zoektocht. Wie er
een tweede probeert, draait een zoektocht en betaalt de trial.

**Geen uitspraak over economische impact.** `n_decision_changes` telt bars
waarop de poort anders uitvalt. Wat dat in P&L doet, is niet gemeten en volgt
hier niet uit — dat is stap 13's vraag, met de volledige inferentiekern erop.

**De dekking begrenst alles.** 39,67 % van de ontwikkelsample, in een venster
dat drie maanden vóór het einde ervan ophoudt (§1), en op twee van de zes namen
uit fits die H1's eigen poort niet vergelijkbaar noemt (§1.1). Elk percentage
hierboven is een uitspraak over 2023-12-06 tot 2025-06-08 en over niets anders.

---

## Bijlage — herleidbaarheid

Nul trials; er is geen veld in `conf/model/regime.yaml`,
`conf/model/volatility.yaml` of `conf/model/adequacy.yaml` op grond van dit
rapport gewijzigd. Dat is de voorwaarde waaronder stap 8 nul trials kost, en
zij is gerespecteerd.

Reproductie:
```
python apps/run_state_agreement.py
python -m pytest tests/unit/test_state_agreement.py -q
```

De stdout van de eerste regel is de enige bron van elk getal in dit rapport;
er is bewust geen JSON-artefact (YAGNI — er is geen tweede consument).
