# H1 — DE GARCH-FAMILIE TEGEN EWMA(0.94) OP QLIKE

> **Deliverable 13** · Phase 6 stap 7 · Phase 7/8 Stage C-2 en C-3
> **git_sha:** `8e6cd02`
> **Pre-registratie:** `cef1a3b9a6811d7bde1afc92a2a9503f` (bevroren 2026-08-25T17:52:22.254778+00:00, M bij bevriezing = 2716)
> **Universum:** 6 gecertificeerde reeksen · 1743 bars · 2021-11-15 t/m 2026-08-23
> **Walk-forward:** 12 folds · train 495 · test 100 · embargo 5 bars · 1200 OOS-bars per symbool

## 0. De uitkomst, eerst

**Geen enkel lid van de GARCH-familie is gepromoveerd. EWMA(0.94) blijft de productie-estimator.**

Dat is geen nulresultaat maar een MEETLATRESULTAAT: bij 36 van de 48 combinaties bindt `proxy_premise_violated`. De gepre-registreerde proxy blijkt een andere grootheid te meten dan de modellen voorspellen, en op zo'n meetlat valt niets te promoveren en evenmin iets te falsifiëren. §2.3 meet dat, met het mechanisme erbij.

| Oordeel | Aantal | Betekenis |
|---|---|---|
| PROMOTED | 0 | lagere QLIKE dan EWMA(0.94), significant, gepowerd én geconvergeerd |
| FALSIFIED | 0 | getoetst en geen verbetering gevonden terwijl de toets er wel een had kunnen zien |
| UNPROVEN | 36 | niet vast te stellen: de meetlat is aantoonbaar verschoven (§2.3), of de toets kon het verwachte effect niet zien |
| DESCOPED | 12 | niet vergelijkbaar: ARCH-poort dicht, te weinig convergentie of te veel randoplossingen, of een gedegenereerde forecast (§4.1) |

Welk criterium waar bond:

| Stop-criterium | Combinaties |
|---|---|
| `proxy_premise_violated` | 36 |
| `convergence_too_low_to_compare` | 10 |
| `degenerate_forecast_level` | 2 |

Het onderscheid tussen `FALSIFIED` en `UNPROVEN` is geen nuance maar een regel uit de pre-registratie: no-go 8 van de fase verbiedt een falsificatie-oordeel over een model dat simpelweg te weinig data had. Welke van de twee geldt, hangt af van het gemeten minimaal detecteerbare effect in §6 — een getal, niet een indruk.

## 1. Wat vooraf vastlag

| | |
|---|---|
| Nulhypothese | de verwachte QLIKE van elk familielid is ≥ die van EWMA(0.94) |
| Primaire maat | `oos_qlike` (proxy-robuust, Patton 2011) |
| Toets | Diebold-Mariano met HLN-correctie, tweezijdig, α = 0.05 |
| Geplande trials | 48 (4 varianten × 6 symbolen × 2 horizonnen) |
| Verwacht effect | 0.1 SD van de per-bar verliesverschilreeks (Hansen & Lunde 2005, bovengrens) |

De vier stop-criteria en hun volgorde staan in `conf/research/preregistration_h1_garch_vs_ewma.yaml` en zijn in code afgedwongen in `validation/vol_competition.py::judge_challenger`. De volgorde is bindend: een gesloten ARCH-poort betekent dat een GARCH-structuur op die reeks niet gerechtvaardigd is, en dan valt er niets te falsifiëren.

## 2. Waartegen QLIKE is gemeten — en wat daaraan ontbreekt

### 2.1 Er is geen realized variance. Dat is een meting, geen aanname.

De Data Adequacy Gate meet **0.00 %** van de dagen met voldoende 5m-dekking tegen een eis van 80 % (0 van 1742 dagen). De gecertificeerde store bevat geen (alleen 1d, 8h funding, 1d OI).

**Oordeel HAR-RV: `UNPROVEN — insufficient data`.** Niet `FALSIFIED` — no-go 8 van de fase verbiedt een falsificatie-oordeel over een model dat de Data Adequacy Gate niet haalde. Er is geen enkele HAR-RV-parameter geschat en geen enkele parameterruimte doorzocht; de trial telt daarom **niet** mee in `M`.

Wat daarmee onbeslist blijft, expliciet: **HAR-RV als Level 3-uitdager is niet getoetst.** Niet afgewezen, niet aangenomen — niet getoetst. Het spoor gaat pas open met een intraday-bron, en dat is een inkoopbesluit en geen technische keuze.

### 2.2 De competitie draait dus op een dagelijkse range-proxy

QLIKE vergelijkt een variantieforecast met een PROXY voor de gerealiseerde variantie. Die proxy is hier een range-estimator op dagbars, geen realized variance uit intraday returns. Dat is legitiem — QLIKE is robuust tegen proxy-ruis zolang de proxy conditioneel zuiver is (Patton 2011) — maar de EFFICIENTIE ligt veel lager dan die van 5-minuts-RV, en lagere efficiëntie is direct minder power. **Dit is een beperking van de competitie zelf en niet van de modellen die eraan meedoen.**

| Proxy | Rol hier | Relatieve efficiëntie | Aanname |
|---|---|---|---|
| `garman_klass` | robuustheid | 7.4× | geen drift binnen de bar; gebruikt open en close |
| `parkinson` | robuustheid | 5.2× | geen drift binnen de bar |
| `rogers_satchell` | **primair** | 8.0× | **zuiver ook mét drift** binnen de bar |
| `squared_return` | robuustheid | 1.0× | geen — zuiver, maar de ruisigste van de vier |

De primaire proxy is **`rogers_satchell`**, en die keuze is vóór de run gemaakt op één grond: Rogers-Satchell is de enige range-estimator in dit rijtje die zuiver blijft bij een DRIFT binnen de bar. Parkinson en Garman-Klass veronderstellen driftloosheid; op een reeks die in een jaar verdrievoudigt of halveert, is dat de aanname die het eerst breekt. De overige proxies draaien mee als robuustheidscontrole (§9) op EXACT dezelfde fits, zodat het verschil de proxy is en niet een tweede campagne.

Wat alle vier gemeen hebben: zij meten de variatie BINNEN de dag uit vier prijzen, terwijl de modellen de variantie van de CLOSE-TO-CLOSE-return voorspellen — dat is de reeks waarop zij zijn gefit. Op een driftloze GBM vallen die twee grootheden samen, en op die gelijkheid rust de hele proxykeuze. De volgende paragraaf meet of zij op deze data standhoudt. Zij houdt geen stand.

### 2.3 De premisse onder de proxy — gemeten, en geschonden

De pre-registratie rechtvaardigt de range-estimator met een voorwaarde: *"QLIKE is robuust tegen proxy-ruis zolang de proxy conditioneel zuiver is (Patton 2011), en de range-estimators zijn dat onder een driftloze GBM binnen de dag"*. Die voorwaarde is een PREMISSE, en een premisse is meetbaar.

Waarom zij ertoe doet: QLIKE heeft zijn minimum op ``forecast = E[proxy]``. Draagt de proxy een multiplicatieve factor ten opzichte van de grootheid die de modellen voorspellen — de variantie van de **close-to-close return**, want daarop zijn zij gefit — dan verschuift dat minimum mee. De competitie rangschikt dan op *kalibratie tegen een verschoven doel* in plaats van op voorspelkwaliteit, en het model met het toevallig passende NIVEAU wint.

De referentie is de gekwadrateerde return. Die is de ruisigste proxy die er is, en tegelijk de enige die per constructie zuiver is voor precies de voorspelde grootheid: ``E[r_t² | F_{t-1}] = σ_t²``. Ruis maakt een schatter niet scheef, en het is de scheefheid die de rangorde breekt.

| Symbool | proxy / r² | EWMA-forecast / r² | GARCH-forecast / r² (bereik) |
|---|---|---|---|
| AVAXUSDT | **1.79×** | 1.01× | 1.10× – 1.22× |
| BTCUSDT | **1.30×** | 1.02× | 1.51× – 45091890159650568480489472.00× |
| DOTUSDT | **2.24×** | 1.01× | 0.96× – 1.13× |
| ETHUSDT | **1.27×** | 1.01× | 1.17× – 356940549973063232.00× |
| LINKUSDT | **1.64×** | 1.01× | 1.05× – 1.11× |
| SOLUSDT | **1.41×** | 1.01× | 1.31× – 1.48× |

Twee dingen staan hier naast elkaar, en samen verklaren zij de uitslag van §6 volledig:

1. **De primaire proxy meet een grotere grootheid dan de modellen voorspellen.** Niet marginaal: de factor loopt op tot boven de twee. Op deze data is de intraday-variatie stelselmatig groter dan de close-to-close-variantie — prijzen zwiepen binnen de dag heen en weer en komen terug. Dat is een eigenschap van crypto-dagbars, geen meetfout, maar het maakt de range-estimator ongeschikt als meetlat voor een model dat close-to-close voorspelt.
2. **EWMA(0.94) is nagenoeg perfect gekalibreerd op de close-to-close-variantie** (rond 1,0×), terwijl de GARCH-varianten er systematisch boven zitten. Op een proxy die zelf naar boven is verschoven, is dat een voordeel dat niets met voorspelkwaliteit te maken heeft.

Daarom staat er in `validation/vol_competition.py::judge_challenger` een criterium dat NIET in de bevroren pre-registratie stond: `proxy_premise_violated`. Het is toegevoegd nadat de eerste run de premisse als geschonden mat, en dat is normaal gesproken precies de manoeuvre die pre-registratie uitsluit. Wat het hier toelaatbaar maakt, is de RICHTING: dit criterium kan een `PROMOTED` en een `FALSIFIED` allebei alleen omzetten in `UNPROVEN`. Het kan de conclusie uitsluitend voorzichtiger maken en nooit gunstiger — en het toetst een voorwaarde die de pre-registratie zelf uitspreekt.

**Wat §6 hierdoor NIET zegt.** De DM-uitslagen daar zijn de gepre-registreerde meting op de gepre-registreerde meetlat, en zij staan er onverkort in. Zij zijn geen bewijs voor of tegen de GARCH-familie, omdat de meetlat aantoonbaar een ander doel meet dan het model voorspelt.

## 3. De ARCH-poort — vóór de eerste fit

Stap 3 van de fase: *"De ARCH-test is de poortwachter."* Is er geen aantoonbare conditionele heteroskedasticiteit, dan is een GARCH-structuur op die reeks niet gerechtvaardigd en wordt zij gedescopeerd — zonder fit, zodat er geen QLIKE-getal ontstaat dat los van dit oordeel kan gaan reizen.

| Symbool | Engle-ARCH p | Poort |
|---|---|---|
| AVAXUSDT | 3.26e-15 | open |
| BTCUSDT | 2.15e-12 | open |
| DOTUSDT | 1.72e-06 | open |
| ETHUSDT | 5.47e-13 | open |
| LINKUSDT | 1.33e-16 | open |
| SOLUSDT | 1.59e-50 | open |

## 4. Convergentie en randoplossingen

Niet-convergentie is een RESULTAAT en geen probleem dat wordt weggevangen: een mislukte fit levert geen forecast, en die bars blijven leeg in plaats van te worden gevuld met de waarde van de titelverdediger. Een randoplossing is even informatief: op `persistence ≥ 0,999` bestaat de onvoorwaardelijke variantie niet en is de forecast een random walk in variantie.

| Symbool | Variant | Fits | Geconvergeerd | Op de rand | Vergelijkbaar |
|---|---|---|---|---|---|
| AVAXUSDT | `aparch(p=1,o=1,q=1)-t` | 12 | 100% | 0% | ja |
| AVAXUSDT | `egarch(p=1,o=1,q=1)-t` | 12 | 100% | 0% | ja |
| AVAXUSDT | `garch(p=1,o=0,q=1)-t` | 12 | 100% | 0% | ja |
| AVAXUSDT | `gjr_garch(p=1,o=1,q=1)-t` | 12 | 100% | 0% | ja |
| BTCUSDT | `aparch(p=1,o=1,q=1)-t` | 12 | 100% | 0% | ja |
| BTCUSDT | `egarch(p=1,o=1,q=1)-t` | 12 | 100% | 0% | ja |
| BTCUSDT | `garch(p=1,o=0,q=1)-t` | 12 | 100% | 58% | **nee** |
| BTCUSDT | `gjr_garch(p=1,o=1,q=1)-t` | 12 | 100% | 42% | **nee** |
| DOTUSDT | `aparch(p=1,o=1,q=1)-t` | 12 | 100% | 0% | ja |
| DOTUSDT | `egarch(p=1,o=1,q=1)-t` | 12 | 100% | 0% | ja |
| DOTUSDT | `garch(p=1,o=0,q=1)-t` | 12 | 100% | 0% | ja |
| DOTUSDT | `gjr_garch(p=1,o=1,q=1)-t` | 12 | 100% | 25% | **nee** |
| ETHUSDT | `aparch(p=1,o=1,q=1)-t` | 12 | 100% | 0% | ja |
| ETHUSDT | `egarch(p=1,o=1,q=1)-t` | 12 | 100% | 0% | ja |
| ETHUSDT | `garch(p=1,o=0,q=1)-t` | 12 | 100% | 75% | **nee** |
| ETHUSDT | `gjr_garch(p=1,o=1,q=1)-t` | 12 | 100% | 42% | **nee** |
| LINKUSDT | `aparch(p=1,o=1,q=1)-t` | 12 | 100% | 0% | ja |
| LINKUSDT | `egarch(p=1,o=1,q=1)-t` | 12 | 100% | 0% | ja |
| LINKUSDT | `garch(p=1,o=0,q=1)-t` | 12 | 100% | 0% | ja |
| LINKUSDT | `gjr_garch(p=1,o=1,q=1)-t` | 12 | 100% | 0% | ja |
| SOLUSDT | `aparch(p=1,o=1,q=1)-t` | 12 | 100% | 0% | ja |
| SOLUSDT | `egarch(p=1,o=1,q=1)-t` | 12 | 100% | 0% | ja |
| SOLUSDT | `garch(p=1,o=0,q=1)-t` | 12 | 100% | 0% | ja |
| SOLUSDT | `gjr_garch(p=1,o=1,q=1)-t` | 12 | 100% | 0% | ja |

### 4.1 Forecasts die ophielden forecasts te zijn

Op h > 1 bestaat er voor EGARCH en APARCH geen analytische meerstaps-forecast: hun recursie loopt in ``ln σ²`` respectievelijk ``σ^δ``, en de terugtransformatie heeft geen gesloten vorm. De forecast wordt daarom gesimuleerd. Dat legt een eigenschap van het model bloot die de analytische route zou hebben verborgen: de verwachting van ``exp`` van een zwaarstaartige random walk hoeft niet te bestaan, en dan schat de simulatie een moment dat er niet is.

| Symbool | h | Variant | hoogste forecast / r² |
|---|---|---|---|
| BTCUSDT | 5 | `egarch(p=1,o=1,q=1)-t` | **5.41e+28×** |
| ETHUSDT | 5 | `egarch(p=1,o=1,q=1)-t` | **4.28e+20×** |

Deze combinaties zijn GEDESCOPEERD en niet gefalsifieerd. Het model heeft daar geen bruikbaar getal geleverd; dat afrekenen als "slechter dan EWMA" zou een oordeel vellen over een meting die niet bestaat.

Waarom dit een aparte poort verdient: QLIKE groeit slechts LOGARITMISCH in een overschatting. Een handvol bars met een forecast van 10²⁵ verschuift het gemiddelde verlies nauwelijks, dus het getal blijft er bruikbaar uitzien. Zonder deze controle zou zo'n reeks gewoon meedoen in de rangorde.

## 5. De verliezen

Gemeten op de primaire proxy `rogers_satchell`, uitsluitend op de testvensters van de walk-forward. Beide modellen zien dezelfde bars: EWMA heeft na zijn burn-in overal een waarde, maar wordt hier niet gescoord op de trainbars waarop zijn uitdager per constructie niet beoordeeld mag worden.

| Symbool | h | Variant | QLIKE | QLIKE EWMA | MSE-SD | MAE-SD | Bars |
|---|---|---|---|---|---|---|---|
| AVAXUSDT | 1 | `aparch(p=1,o=1,q=1)-t` | 1.0952 | 1.1677 | 0.00157 | 0.018 | 1200 |
| AVAXUSDT | 1 | `egarch(p=1,o=1,q=1)-t` | 1.0732 | 1.1677 | 0.00156 | 0.0181 | 1200 |
| AVAXUSDT | 1 | `garch(p=1,o=0,q=1)-t` | 1.0234 | 1.1677 | 0.00154 | 0.0179 | 1200 |
| AVAXUSDT | 1 | `gjr_garch(p=1,o=1,q=1)-t` | 1.0431 | 1.1677 | 0.00155 | 0.0181 | 1200 |
| AVAXUSDT | 5 | `aparch(p=1,o=1,q=1)-t` | 1.2602 | 1.3679 | 0.00167 | 0.0198 | 1200 |
| AVAXUSDT | 5 | `egarch(p=1,o=1,q=1)-t` | 1.2331 | 1.3679 | 0.00168 | 0.0201 | 1200 |
| AVAXUSDT | 5 | `garch(p=1,o=0,q=1)-t` | 1.1816 | 1.3679 | 0.00169 | 0.0203 | 1200 |
| AVAXUSDT | 5 | `gjr_garch(p=1,o=1,q=1)-t` | 1.1823 | 1.3679 | 0.0017 | 0.0204 | 1200 |
| BTCUSDT | 1 | `aparch(p=1,o=1,q=1)-t` | 0.6730 | 0.7901 | 0.000272 | 0.0129 | 1200 |
| BTCUSDT | 1 | `egarch(p=1,o=1,q=1)-t` | 0.6617 | 0.7901 | 0.00026 | 0.0125 | 1200 |
| BTCUSDT | 1 | `garch(p=1,o=0,q=1)-t` | 0.6502 | 0.7901 | 0.000239 | 0.0117 | 1200 |
| BTCUSDT | 1 | `gjr_garch(p=1,o=1,q=1)-t` | 0.6488 | 0.7901 | 0.000244 | 0.0119 | 1200 |
| BTCUSDT | 5 | `aparch(p=1,o=1,q=1)-t` | 0.7221 | 0.8744 | 0.000301 | 0.0139 | 1200 |
| BTCUSDT | 5 | `egarch(p=1,o=1,q=1)-t` | 1.3060 | 0.8744 | 2.6e+22 | 4.66e+09 | 1200 |
| BTCUSDT | 5 | `garch(p=1,o=0,q=1)-t` | 0.7115 | 0.8744 | 0.000286 | 0.0132 | 1200 |
| BTCUSDT | 5 | `gjr_garch(p=1,o=1,q=1)-t` | 0.7204 | 0.8744 | 0.000302 | 0.0138 | 1200 |
| DOTUSDT | 1 | `aparch(p=1,o=1,q=1)-t` | 1.5655 | 1.9154 | 0.00203 | 0.0164 | 1200 |
| DOTUSDT | 1 | `egarch(p=1,o=1,q=1)-t` | 1.5479 | 1.9154 | 0.00203 | 0.0165 | 1200 |
| DOTUSDT | 1 | `garch(p=1,o=0,q=1)-t` | 1.6953 | 1.9154 | 0.00205 | 0.017 | 1200 |
| DOTUSDT | 1 | `gjr_garch(p=1,o=1,q=1)-t` | 1.6770 | 1.9154 | 0.00206 | 0.0172 | 1200 |
| DOTUSDT | 5 | `aparch(p=1,o=1,q=1)-t` | 1.8136 | 2.3427 | 0.00209 | 0.0174 | 1200 |
| DOTUSDT | 5 | `egarch(p=1,o=1,q=1)-t` | 1.7812 | 2.3427 | 0.00209 | 0.0177 | 1200 |
| DOTUSDT | 5 | `garch(p=1,o=0,q=1)-t` | 2.0217 | 2.3427 | 0.00214 | 0.0185 | 1200 |
| DOTUSDT | 5 | `gjr_garch(p=1,o=1,q=1)-t` | 1.9288 | 2.3427 | 0.00214 | 0.0186 | 1200 |
| ETHUSDT | 1 | `aparch(p=1,o=1,q=1)-t` | 0.5996 | 0.7393 | 0.000405 | 0.0135 | 1200 |
| ETHUSDT | 1 | `egarch(p=1,o=1,q=1)-t` | 0.5826 | 0.7393 | 0.000401 | 0.0135 | 1200 |
| ETHUSDT | 1 | `garch(p=1,o=0,q=1)-t` | 0.6169 | 0.7393 | 0.00041 | 0.0136 | 1200 |
| ETHUSDT | 1 | `gjr_garch(p=1,o=1,q=1)-t` | 0.6707 | 0.7393 | 0.00042 | 0.014 | 1200 |
| ETHUSDT | 5 | `aparch(p=1,o=1,q=1)-t` | 0.6739 | 0.8569 | 0.000446 | 0.0145 | 1200 |
| ETHUSDT | 5 | `egarch(p=1,o=1,q=1)-t` | 0.7775 | 0.8569 | 3.96e+14 | 5.8e+05 | 1200 |
| ETHUSDT | 5 | `garch(p=1,o=0,q=1)-t` | 0.7065 | 0.8569 | 0.000463 | 0.0149 | 1200 |
| ETHUSDT | 5 | `gjr_garch(p=1,o=1,q=1)-t` | 0.7796 | 0.8569 | 0.000496 | 0.0158 | 1200 |
| LINKUSDT | 1 | `aparch(p=1,o=1,q=1)-t` | 0.9149 | 1.0742 | 0.00125 | 0.0167 | 1200 |
| LINKUSDT | 1 | `egarch(p=1,o=1,q=1)-t` | 0.9014 | 1.0742 | 0.00125 | 0.0166 | 1200 |
| LINKUSDT | 1 | `garch(p=1,o=0,q=1)-t` | 0.9113 | 1.0742 | 0.00125 | 0.0167 | 1200 |
| LINKUSDT | 1 | `gjr_garch(p=1,o=1,q=1)-t` | 0.9109 | 1.0742 | 0.00125 | 0.0166 | 1200 |
| LINKUSDT | 5 | `aparch(p=1,o=1,q=1)-t` | 1.0810 | 1.3084 | 0.00134 | 0.0183 | 1200 |
| LINKUSDT | 5 | `egarch(p=1,o=1,q=1)-t` | 1.1186 | 1.3084 | 0.00134 | 0.0181 | 1200 |
| LINKUSDT | 5 | `garch(p=1,o=0,q=1)-t` | 1.0697 | 1.3084 | 0.00134 | 0.0183 | 1200 |
| LINKUSDT | 5 | `gjr_garch(p=1,o=1,q=1)-t` | 1.0734 | 1.3084 | 0.00135 | 0.0184 | 1200 |
| SOLUSDT | 1 | `aparch(p=1,o=1,q=1)-t` | 0.5096 | 0.5918 | 0.000675 | 0.0176 | 1200 |
| SOLUSDT | 1 | `egarch(p=1,o=1,q=1)-t` | 0.5015 | 0.5918 | 0.000667 | 0.0175 | 1200 |
| SOLUSDT | 1 | `garch(p=1,o=0,q=1)-t` | 0.5047 | 0.5918 | 0.000664 | 0.0176 | 1200 |
| SOLUSDT | 1 | `gjr_garch(p=1,o=1,q=1)-t` | 0.5096 | 0.5918 | 0.000673 | 0.0176 | 1200 |
| SOLUSDT | 5 | `aparch(p=1,o=1,q=1)-t` | 0.5983 | 0.7152 | 0.000808 | 0.0205 | 1200 |
| SOLUSDT | 5 | `egarch(p=1,o=1,q=1)-t` | 0.5931 | 0.7152 | 0.000839 | 0.0205 | 1200 |
| SOLUSDT | 5 | `garch(p=1,o=0,q=1)-t` | 0.5969 | 0.7152 | 0.000812 | 0.0206 | 1200 |
| SOLUSDT | 5 | `gjr_garch(p=1,o=1,q=1)-t` | 0.6000 | 0.7152 | 0.000814 | 0.0206 | 1200 |

Een LAGERE QLIKE is beter; nul is een perfecte forecast. Het verschil in puntschatting beslist niets — zie de volgende paragraaf.

## 6. Diebold-Mariano met HLN-correctie, en de gemeten power

`mean diff` is ``QLIKE(variant) − QLIKE(EWMA)``: negatief betekent dat de uitdager beter was. De p-waarde is tweezijdig; een verwerping in het NADEEL van de uitdager promoveert niets.

`MDE` is het minimaal detecteerbare effect van DEZE toets, in standaarddeviaties van de per-bar verliesverschilreeks, berekend met de GEMETEN AR(1) in plaats van met een vooraf gekozen scenario. Ligt hij boven het verwachte effect, dan kon deze opzet het effect niet zien en is een niet-significante uitkomst `UNPROVEN`.

De kolom `proxy/r²` staat er met opzet naast: waar zij ver van 1,00 ligt, is de p-waarde ernaast een meting op een verschoven meetlat (§2.3) en draagt zij geen oordeel.

| Symbool | h | Variant | mean diff | HLN | p | AR(1) | MDE (SD) | proxy/r² | Oordeel |
|---|---|---|---|---|---|---|---|---|---|
| AVAXUSDT | 1 | `aparch(p=1,o=1,q=1)-t` | -0.0725 | -1.66 | 0.0969 | 0.015 | 0.082 | 1.79 | UNPROVEN |
| AVAXUSDT | 1 | `egarch(p=1,o=1,q=1)-t` | -0.0945 | -2.56 | 0.0106 | 0.005 | 0.081 | 1.79 | UNPROVEN |
| AVAXUSDT | 1 | `garch(p=1,o=0,q=1)-t` | -0.144 | -2.67 | 0.0076 | -0.001 | 0.081 | 1.79 | UNPROVEN |
| AVAXUSDT | 1 | `gjr_garch(p=1,o=1,q=1)-t` | -0.125 | -2.68 | 0.00746 | -0.005 | 0.081 | 1.79 | UNPROVEN |
| AVAXUSDT | 5 | `aparch(p=1,o=1,q=1)-t` | -0.108 | -1.46 | 0.145 | 0.036 | 0.084 | 1.79 | UNPROVEN |
| AVAXUSDT | 5 | `egarch(p=1,o=1,q=1)-t` | -0.135 | -2.17 | 0.0301 | 0.047 | 0.085 | 1.79 | UNPROVEN |
| AVAXUSDT | 5 | `garch(p=1,o=0,q=1)-t` | -0.186 | -2.51 | 0.0121 | 0.033 | 0.084 | 1.79 | UNPROVEN |
| AVAXUSDT | 5 | `gjr_garch(p=1,o=1,q=1)-t` | -0.186 | -2.43 | 0.0151 | 0.029 | 0.083 | 1.79 | UNPROVEN |
| BTCUSDT | 1 | `aparch(p=1,o=1,q=1)-t` | -0.117 | -1.13 | 0.258 | -0.005 | 0.081 | 1.30 | UNPROVEN |
| BTCUSDT | 1 | `egarch(p=1,o=1,q=1)-t` | -0.128 | -1.28 | 0.2 | -0.004 | 0.081 | 1.30 | UNPROVEN |
| BTCUSDT | 1 | `garch(p=1,o=0,q=1)-t` | — | — | — | — | — | 1.30 | DESCOPED |
| BTCUSDT | 1 | `gjr_garch(p=1,o=1,q=1)-t` | — | — | — | — | — | 1.30 | DESCOPED |
| BTCUSDT | 5 | `aparch(p=1,o=1,q=1)-t` | -0.152 | -1.41 | 0.158 | 0.046 | 0.085 | 1.30 | UNPROVEN |
| BTCUSDT | 5 | `egarch(p=1,o=1,q=1)-t` | 0.432 | 2.66 | 0.00787 | 0.053 | — | 1.30 | DESCOPED |
| BTCUSDT | 5 | `garch(p=1,o=0,q=1)-t` | — | — | — | — | — | 1.30 | DESCOPED |
| BTCUSDT | 5 | `gjr_garch(p=1,o=1,q=1)-t` | — | — | — | — | — | 1.30 | DESCOPED |
| DOTUSDT | 1 | `aparch(p=1,o=1,q=1)-t` | -0.35 | -1.21 | 0.228 | -0.006 | 0.081 | 2.24 | UNPROVEN |
| DOTUSDT | 1 | `egarch(p=1,o=1,q=1)-t` | -0.368 | -1.23 | 0.218 | -0.004 | 0.081 | 2.24 | UNPROVEN |
| DOTUSDT | 1 | `garch(p=1,o=0,q=1)-t` | -0.22 | -1.36 | 0.173 | 0.000 | 0.081 | 2.24 | UNPROVEN |
| DOTUSDT | 1 | `gjr_garch(p=1,o=1,q=1)-t` | — | — | — | — | — | 2.24 | DESCOPED |
| DOTUSDT | 5 | `aparch(p=1,o=1,q=1)-t` | -0.529 | -1.26 | 0.208 | 0.006 | 0.081 | 2.24 | UNPROVEN |
| DOTUSDT | 5 | `egarch(p=1,o=1,q=1)-t` | -0.561 | -1.29 | 0.198 | 0.006 | 0.081 | 2.24 | UNPROVEN |
| DOTUSDT | 5 | `garch(p=1,o=0,q=1)-t` | -0.321 | -1.31 | 0.189 | 0.006 | 0.081 | 2.24 | UNPROVEN |
| DOTUSDT | 5 | `gjr_garch(p=1,o=1,q=1)-t` | — | — | — | — | — | 2.24 | DESCOPED |
| ETHUSDT | 1 | `aparch(p=1,o=1,q=1)-t` | -0.14 | -1.13 | 0.261 | 0.004 | 0.081 | 1.27 | UNPROVEN |
| ETHUSDT | 1 | `egarch(p=1,o=1,q=1)-t` | -0.157 | -1.26 | 0.206 | -0.001 | 0.081 | 1.27 | UNPROVEN |
| ETHUSDT | 1 | `garch(p=1,o=0,q=1)-t` | — | — | — | — | — | 1.27 | DESCOPED |
| ETHUSDT | 1 | `gjr_garch(p=1,o=1,q=1)-t` | — | — | — | — | — | 1.27 | DESCOPED |
| ETHUSDT | 5 | `aparch(p=1,o=1,q=1)-t` | -0.183 | -1.58 | 0.113 | 0.026 | 0.083 | 1.27 | UNPROVEN |
| ETHUSDT | 5 | `egarch(p=1,o=1,q=1)-t` | -0.0794 | -0.60 | 0.548 | 0.017 | — | 1.27 | DESCOPED |
| ETHUSDT | 5 | `garch(p=1,o=0,q=1)-t` | — | — | — | — | — | 1.27 | DESCOPED |
| ETHUSDT | 5 | `gjr_garch(p=1,o=1,q=1)-t` | — | — | — | — | — | 1.27 | DESCOPED |
| LINKUSDT | 1 | `aparch(p=1,o=1,q=1)-t` | -0.159 | -1.43 | 0.153 | -0.001 | 0.081 | 1.64 | UNPROVEN |
| LINKUSDT | 1 | `egarch(p=1,o=1,q=1)-t` | -0.173 | -1.37 | 0.172 | -0.001 | 0.081 | 1.64 | UNPROVEN |
| LINKUSDT | 1 | `garch(p=1,o=0,q=1)-t` | -0.163 | -1.50 | 0.134 | 0.001 | 0.081 | 1.64 | UNPROVEN |
| LINKUSDT | 1 | `gjr_garch(p=1,o=1,q=1)-t` | -0.163 | -1.49 | 0.137 | 0.000 | 0.081 | 1.64 | UNPROVEN |
| LINKUSDT | 5 | `aparch(p=1,o=1,q=1)-t` | -0.227 | -1.37 | 0.171 | 0.008 | 0.082 | 1.64 | UNPROVEN |
| LINKUSDT | 5 | `egarch(p=1,o=1,q=1)-t` | -0.19 | -1.51 | 0.132 | 0.011 | 0.082 | 1.64 | UNPROVEN |
| LINKUSDT | 5 | `garch(p=1,o=0,q=1)-t` | -0.239 | -1.41 | 0.158 | 0.008 | 0.082 | 1.64 | UNPROVEN |
| LINKUSDT | 5 | `gjr_garch(p=1,o=1,q=1)-t` | -0.235 | -1.38 | 0.167 | 0.008 | 0.082 | 1.64 | UNPROVEN |
| SOLUSDT | 1 | `aparch(p=1,o=1,q=1)-t` | -0.0823 | -2.48 | 0.0133 | 0.056 | 0.086 | 1.41 | UNPROVEN |
| SOLUSDT | 1 | `egarch(p=1,o=1,q=1)-t` | -0.0903 | -2.60 | 0.00954 | 0.048 | 0.085 | 1.41 | UNPROVEN |
| SOLUSDT | 1 | `garch(p=1,o=0,q=1)-t` | -0.0871 | -2.68 | 0.00736 | 0.070 | 0.087 | 1.41 | UNPROVEN |
| SOLUSDT | 1 | `gjr_garch(p=1,o=1,q=1)-t` | -0.0822 | -2.45 | 0.0145 | 0.064 | 0.086 | 1.41 | UNPROVEN |
| SOLUSDT | 5 | `aparch(p=1,o=1,q=1)-t` | -0.117 | -1.98 | 0.048 | 0.132 | 0.092 | 1.41 | UNPROVEN |
| SOLUSDT | 5 | `egarch(p=1,o=1,q=1)-t` | -0.122 | -2.12 | 0.0339 | 0.128 | 0.092 | 1.41 | UNPROVEN |
| SOLUSDT | 5 | `garch(p=1,o=0,q=1)-t` | -0.118 | -1.98 | 0.0475 | 0.135 | 0.093 | 1.41 | UNPROVEN |
| SOLUSDT | 5 | `gjr_garch(p=1,o=1,q=1)-t` | -0.115 | -1.93 | 0.0538 | 0.135 | 0.093 | 1.41 | UNPROVEN |

## 7. Mincer-Zarnowitz — is de forecast zuiver?

``RV_t = α + β·σ²_t + e_t`` met HAC-standaardfouten. De gezamenlijke nulhypothese is ``(α, β) = (0, 1)``: een zuivere forecast. Verwerping betekent een systematische onder- of overschatting, en die kan naast een goede QLIKE bestaan.

| Symbool | h | Model | α | β | Wald p | R² | Zuiver |
|---|---|---|---|---|---|---|---|
| AVAXUSDT | 1 | `aparch(p=1,o=1,q=1)-t` | 0.00155 | 0.937 | 0.0641 | 0.001 | ja |
| AVAXUSDT | 1 | `ewma_0.94` | 0.0017 | 0.951 | 0.156 | 0.001 | ja |
| AVAXUSDT | 1 | `egarch(p=1,o=1,q=1)-t` | 0.00147 | 0.949 | 0.0975 | 0.001 | ja |
| AVAXUSDT | 1 | `garch(p=1,o=0,q=1)-t` | 0.00137 | 0.977 | 0.196 | 0.002 | ja |
| AVAXUSDT | 1 | `gjr_garch(p=1,o=1,q=1)-t` | 0.00147 | 0.930 | 0.219 | 0.002 | ja |
| AVAXUSDT | 5 | `aparch(p=1,o=1,q=1)-t` | 0.00397 | -0.126 | 0.388 | 0.000 | ja |
| AVAXUSDT | 5 | `ewma_0.94` | 0.00315 | 0.254 | 0.113 | 0.000 | ja |
| AVAXUSDT | 5 | `egarch(p=1,o=1,q=1)-t` | 0.00375 | -0.033 | 0.351 | 0.000 | ja |
| AVAXUSDT | 5 | `garch(p=1,o=0,q=1)-t` | 0.00342 | 0.100 | 0.0133 | 0.000 | nee |
| AVAXUSDT | 5 | `gjr_garch(p=1,o=1,q=1)-t` | 0.0034 | 0.107 | 0.0206 | 0.000 | nee |
| BTCUSDT | 1 | `aparch(p=1,o=1,q=1)-t` | 0.000276 | 0.468 | 3.25e-10 | 0.018 | nee |
| BTCUSDT | 1 | `ewma_0.94` | 0.000381 | 0.623 | 0.0001 | 0.022 | nee |
| BTCUSDT | 1 | `egarch(p=1,o=1,q=1)-t` | 0.000226 | 0.532 | 1.79e-08 | 0.024 | nee |
| BTCUSDT | 5 | `aparch(p=1,o=1,q=1)-t` | 0.000515 | 0.220 | 1.73e-20 | 0.003 | nee |
| BTCUSDT | 5 | `ewma_0.94` | 0.000579 | 0.285 | 1.72e-07 | 0.005 | nee |
| BTCUSDT | 5 | `egarch(p=1,o=1,q=1)-t` | 0.000748 | -0.000 | 0 | 0.000 | nee |
| DOTUSDT | 1 | `aparch(p=1,o=1,q=1)-t` | 0.0013 | 1.534 | 0.0259 | 0.001 | nee |
| DOTUSDT | 1 | `ewma_0.94` | 0.0029 | 0.541 | 0.243 | 0.000 | ja |
| DOTUSDT | 1 | `egarch(p=1,o=1,q=1)-t` | 0.00165 | 1.264 | 0.0648 | 0.001 | ja |
| DOTUSDT | 1 | `garch(p=1,o=0,q=1)-t` | 0.00257 | 0.699 | 0.233 | 0.000 | ja |
| DOTUSDT | 5 | `aparch(p=1,o=1,q=1)-t` | 0.00278 | 0.632 | 0.13 | 0.000 | ja |
| DOTUSDT | 5 | `ewma_0.94` | 0.00402 | -0.105 | 0.365 | 0.000 | ja |
| DOTUSDT | 5 | `egarch(p=1,o=1,q=1)-t` | 0.00304 | 0.453 | 0.261 | 0.000 | ja |
| DOTUSDT | 5 | `garch(p=1,o=0,q=1)-t` | 0.00405 | -0.113 | 0.378 | 0.000 | ja |
| ETHUSDT | 1 | `aparch(p=1,o=1,q=1)-t` | -0.000112 | 1.173 | 0.613 | 0.021 | ja |
| ETHUSDT | 1 | `ewma_0.94` | 0.000491 | 0.823 | 0.00115 | 0.019 | nee |
| ETHUSDT | 1 | `egarch(p=1,o=1,q=1)-t` | 2.22e-05 | 1.019 | 0.89 | 0.023 | ja |
| ETHUSDT | 5 | `aparch(p=1,o=1,q=1)-t` | 0.000824 | 0.430 | 0.0186 | 0.002 | nee |
| ETHUSDT | 5 | `ewma_0.94` | 0.00103 | 0.337 | 1.12e-06 | 0.003 | nee |
| ETHUSDT | 5 | `egarch(p=1,o=1,q=1)-t` | 0.00141 | -0.000 | 0 | 0.000 | nee |
| LINKUSDT | 1 | `aparch(p=1,o=1,q=1)-t` | 0.000834 | 1.131 | 0.204 | 0.002 | ja |
| LINKUSDT | 1 | `ewma_0.94` | 0.00179 | 0.696 | 0.273 | 0.001 | ja |
| LINKUSDT | 1 | `egarch(p=1,o=1,q=1)-t` | 0.000438 | 1.348 | 0.167 | 0.002 | ja |
| LINKUSDT | 1 | `garch(p=1,o=0,q=1)-t` | 0.00105 | 1.009 | 0.285 | 0.002 | ja |
| LINKUSDT | 1 | `gjr_garch(p=1,o=1,q=1)-t` | 0.000947 | 1.064 | 0.249 | 0.002 | ja |
| LINKUSDT | 5 | `aparch(p=1,o=1,q=1)-t` | 0.00319 | -0.031 | 0.362 | 0.000 | ja |
| LINKUSDT | 5 | `ewma_0.94` | 0.0027 | 0.223 | 0.219 | 0.000 | ja |
| LINKUSDT | 5 | `egarch(p=1,o=1,q=1)-t` | 0.00371 | -0.284 | 0.362 | 0.000 | ja |
| LINKUSDT | 5 | `garch(p=1,o=0,q=1)-t` | 0.00281 | 0.152 | 0.387 | 0.000 | ja |
| LINKUSDT | 5 | `gjr_garch(p=1,o=1,q=1)-t` | 0.00309 | 0.018 | 0.336 | 0.000 | ja |
| SOLUSDT | 1 | `aparch(p=1,o=1,q=1)-t` | -0.00101 | 1.485 | 0.275 | 0.060 | ja |
| SOLUSDT | 1 | `ewma_0.94` | 0.000721 | 1.015 | 7.99e-05 | 0.035 | nee |
| SOLUSDT | 1 | `egarch(p=1,o=1,q=1)-t` | -0.00143 | 1.665 | 0.0407 | 0.068 | nee |
| SOLUSDT | 1 | `garch(p=1,o=0,q=1)-t` | -0.000971 | 1.454 | 0.325 | 0.071 | ja |
| SOLUSDT | 1 | `gjr_garch(p=1,o=1,q=1)-t` | -0.00101 | 1.492 | 0.348 | 0.064 | ja |
| SOLUSDT | 5 | `aparch(p=1,o=1,q=1)-t` | 0.00075 | 0.694 | 0.514 | 0.006 | ja |
| SOLUSDT | 5 | `ewma_0.94` | 0.00167 | 0.508 | 3.31e-06 | 0.009 | nee |
| SOLUSDT | 5 | `egarch(p=1,o=1,q=1)-t` | 0.00241 | 0.079 | 1.43e-26 | 0.001 | nee |
| SOLUSDT | 5 | `garch(p=1,o=0,q=1)-t` | 0.000789 | 0.669 | 0.303 | 0.007 | ja |
| SOLUSDT | 5 | `gjr_garch(p=1,o=1,q=1)-t` | 0.000878 | 0.641 | 0.347 | 0.006 | ja |

## 8. De negatieve controles — op de TOETS, niet op de markt

Stap 7: *"een toets die ook op ruis significant is, meet niets"*. Die eis heeft twee kanten, en één ervan alleen is misleidend.

**POWER.** EWMA tegen een geschudde variant van zichzelf: dezelfde waarden, alleen de timing weg. Ziet DM-HLN dát verschil niet, dan kan hij het veel kleinere GARCH-EWMA-verschil zeker niet zien en betekent geen enkele niet-significante uitslag in dit rapport iets.

**SIZE.** Per bar wisselen welke van twee echte verliesreeksen bij welk model hoort. Het verwachte verschil is dan per constructie nul, terwijl schaal, staarten en seriële structuur blijven staan. Verwerpt de toets daar veel vaker dan α, dan verwerpt hij op ruis en is elke significante uitslag hierboven verdacht.

| Reeks | h | Geschud: p | Geschud significant | Verwerping op ruis | Plafond | Geslaagd |
|---|---|---|---|---|---|---|
| AVAXUSDT | 1 | 0.713 | **nee** | 0.0% | 10% | **nee** |
| AVAXUSDT | 5 | 0.815 | **nee** | 0.0% | 10% | **nee** |
| BTCUSDT | 1 | 0.0288 | ja | 3.5% | 10% | ja |
| BTCUSDT | 5 | 0.357 | **nee** | 3.5% | 10% | **nee** |
| DOTUSDT | 1 | 2.26e-06 | ja | 5.5% | 10% | ja |
| DOTUSDT | 5 | 0.901 | **nee** | 0.0% | 10% | **nee** |
| ETHUSDT | 1 | 0.0536 | **nee** | 1.0% | 10% | **nee** |
| ETHUSDT | 5 | 0.357 | **nee** | 0.5% | 10% | **nee** |
| LINKUSDT | 1 | 0.687 | **nee** | 0.0% | 10% | **nee** |
| LINKUSDT | 5 | 0.711 | **nee** | 0.0% | 10% | **nee** |
| SOLUSDT | 1 | 4.64e-07 | ja | 5.0% | 10% | ja |
| SOLUSDT | 5 | 0.00125 | ja | 4.0% | 10% | ja |

Elke controle draait op 200 replicaties en op de TITELVERDEDIGER, niet op de winnaar: zou zij op het winnende model draaien, dan hing haar oordeel af van wie won.

**De uitkomst, en zij is ongemakkelijk.** De SIZE-kant slaagt overal: de toets verwerpt niet op ruis. De POWER-kant faalt op **8 van de 12** reeks/horizon-combinaties — daar onderscheidt DM-HLN een forecast waarvan de timing volledig is vernietigd, NIET van het origineel.

Dat weegt zwaarder dan het lijkt. De pre-registratie leidt haar power AF uit het aantal observaties en de AR(1), en die berekening zegt dat de toets gepowerd is (§6, kolom MDE). De controle MEET hetzelfde en spreekt dat tegen. Waar berekening en meting botsen, wint de meting: een toets die het grootst denkbare verschil niet ziet, ziet het veel kleinere GARCH-EWMA-verschil zeker niet.

Het gevolg is afgedwongen en niet alleen opgeschreven: `judge_challenger` weigert een `FALSIFIED` op een reeks waar deze controle faalt, en geeft `UNPROVEN` — het criterium `underpowered_test_cannot_falsify` uit de pre-registratie, nu gevoed door een meting in plaats van door een afleiding.

## 9. Robuustheid over de proxies

Dezelfde fits, dezelfde bars, een andere realisatie. QLIKE hoort proxy-robuust te zijn zolang de proxy conditioneel zuiver is; deze tabel maakt van die stelling een meting. Zij draagt geen oordeel — het verdict staat op de primaire proxy, zoals vooraf vastgelegd. Zou een tweede proxy het oordeel mogen kantelen, dan was de proxykeuze achteraf gemaakt.

| Symbool | h | Variant | p (rogers_satchell) | p (garman_klass) | p (parkinson) | p (squared_return) |
|---|---|---|---|---|---|---|
| AVAXUSDT | 1 | `aparch(p=1,o=1,q=1)-t` | 0.0969 | 0.0522 | 0.0778 | 0.518 |
| AVAXUSDT | 1 | `egarch(p=1,o=1,q=1)-t` | 0.0106 | 0.00787 | 0.02 | 0.514 |
| AVAXUSDT | 1 | `garch(p=1,o=0,q=1)-t` | 0.0076 | 0.00474 | 0.0104 | 0.517 |
| AVAXUSDT | 1 | `gjr_garch(p=1,o=1,q=1)-t` | 0.00746 | 0.00701 | 0.0202 | 0.752 |
| AVAXUSDT | 5 | `aparch(p=1,o=1,q=1)-t` | 0.145 | 0.0842 | 0.11 | 0.337 |
| AVAXUSDT | 5 | `egarch(p=1,o=1,q=1)-t` | 0.0301 | 0.0252 | 0.05 | 0.345 |
| AVAXUSDT | 5 | `garch(p=1,o=0,q=1)-t` | 0.0121 | 0.0116 | 0.0264 | 0.368 |
| AVAXUSDT | 5 | `gjr_garch(p=1,o=1,q=1)-t` | 0.0151 | 0.0152 | 0.0342 | 0.429 |
| BTCUSDT | 1 | `aparch(p=1,o=1,q=1)-t` | 0.258 | 0.37 | 0.664 | 0.0875 |
| BTCUSDT | 1 | `egarch(p=1,o=1,q=1)-t` | 0.2 | 0.277 | 0.522 | 0.139 |
| BTCUSDT | 5 | `aparch(p=1,o=1,q=1)-t` | 0.158 | 0.247 | 0.518 | 0.121 |
| BTCUSDT | 5 | `egarch(p=1,o=1,q=1)-t` | 0.00787 | 0.00049 | 2.73e-05 | 2.28e-09 |
| DOTUSDT | 1 | `aparch(p=1,o=1,q=1)-t` | 0.228 | 0.2 | 0.195 | 0.215 |
| DOTUSDT | 1 | `egarch(p=1,o=1,q=1)-t` | 0.218 | 0.187 | 0.179 | 0.127 |
| DOTUSDT | 1 | `garch(p=1,o=0,q=1)-t` | 0.173 | 0.146 | 0.155 | 0.58 |
| DOTUSDT | 5 | `aparch(p=1,o=1,q=1)-t` | 0.208 | 0.175 | 0.169 | 0.143 |
| DOTUSDT | 5 | `egarch(p=1,o=1,q=1)-t` | 0.198 | 0.163 | 0.157 | 0.0981 |
| DOTUSDT | 5 | `garch(p=1,o=0,q=1)-t` | 0.189 | 0.156 | 0.16 | 0.363 |
| ETHUSDT | 1 | `aparch(p=1,o=1,q=1)-t` | 0.261 | 0.29 | 0.344 | 0.984 |
| ETHUSDT | 1 | `egarch(p=1,o=1,q=1)-t` | 0.206 | 0.203 | 0.235 | 0.75 |
| ETHUSDT | 5 | `aparch(p=1,o=1,q=1)-t` | 0.113 | 0.12 | 0.166 | 0.87 |
| ETHUSDT | 5 | `egarch(p=1,o=1,q=1)-t` | 0.548 | 0.8 | 0.901 | 0.133 |
| LINKUSDT | 1 | `aparch(p=1,o=1,q=1)-t` | 0.153 | 0.133 | 0.144 | 0.692 |
| LINKUSDT | 1 | `egarch(p=1,o=1,q=1)-t` | 0.172 | 0.152 | 0.164 | 0.751 |
| LINKUSDT | 1 | `garch(p=1,o=0,q=1)-t` | 0.134 | 0.108 | 0.111 | 0.49 |
| LINKUSDT | 1 | `gjr_garch(p=1,o=1,q=1)-t` | 0.137 | 0.115 | 0.122 | 0.592 |
| LINKUSDT | 5 | `aparch(p=1,o=1,q=1)-t` | 0.171 | 0.144 | 0.141 | 0.324 |
| LINKUSDT | 5 | `egarch(p=1,o=1,q=1)-t` | 0.132 | 0.106 | 0.103 | 0.271 |
| LINKUSDT | 5 | `garch(p=1,o=0,q=1)-t` | 0.158 | 0.127 | 0.122 | 0.268 |
| LINKUSDT | 5 | `gjr_garch(p=1,o=1,q=1)-t` | 0.167 | 0.139 | 0.136 | 0.318 |
| SOLUSDT | 1 | `aparch(p=1,o=1,q=1)-t` | 0.0133 | 0.0112 | 0.0621 | 0.509 |
| SOLUSDT | 1 | `egarch(p=1,o=1,q=1)-t` | 0.00954 | 0.00747 | 0.0412 | 0.559 |
| SOLUSDT | 1 | `garch(p=1,o=0,q=1)-t` | 0.00736 | 0.00602 | 0.0405 | 0.558 |
| SOLUSDT | 1 | `gjr_garch(p=1,o=1,q=1)-t` | 0.0145 | 0.0122 | 0.0625 | 0.554 |
| SOLUSDT | 5 | `aparch(p=1,o=1,q=1)-t` | 0.048 | 0.0629 | 0.201 | 0.454 |
| SOLUSDT | 5 | `egarch(p=1,o=1,q=1)-t` | 0.0339 | 0.046 | 0.168 | 0.411 |
| SOLUSDT | 5 | `garch(p=1,o=0,q=1)-t` | 0.0475 | 0.0624 | 0.205 | 0.411 |
| SOLUSDT | 5 | `gjr_garch(p=1,o=1,q=1)-t` | 0.0538 | 0.0709 | 0.221 | 0.427 |

## 10. De ledger: M is niet gegroeid, en dat is de bedoeling

| | |
|---|---|
| M vóór deze run | 2776 |
| Trials geboekt bij het BEVRIEZEN | 48 (`phase6_h1_garch_vs_ewma`, `cef1a3b9a6811d7b`) |
| Trials in deze run gefit | 48 |
| M ná deze run | 2776 |

De 48 trials stonden al in `M` vóór de eerste fit: zij zijn geboekt toen de pre-registratie werd bevroren. Dat is precies goed — wie een parameterruimte vastlegt, heeft die kansen genomen, en `M` hoort niet pas te groeien als de uitkomst bevalt.

Daarom boekt deze run ze NIET opnieuw. Zou hij dat doen, dan ging `M` van 2776 naar 2824 voor onderzoek dat één keer is gedaan. Ondertellen maakt elke deflated Sharpe ratio erna te gunstig, dubbeltellen maakt hem te streng — beide getallen zijn even onwaar. Het oordeel gaat daarom als AMENDEMENT de ledger in: `n_trials = 0`, met `amends` naar de entry hierboven.

Geplande trials: 48. Feitelijk gefit: 48.
Die twee zijn gelijk: geen enkele poort heeft een fit voorkomen, dus elke geboekte combinatie heeft daadwerkelijk een kans gehad om iets te vinden. De boeking bij het bevriezen was dus exact.

EWMA(0.94) telt niet mee: lambda komt uit `conf/model/volatility.yaml` en is niet gevarieerd. De negatieve controles tellen evenmin mee: zij toetsen de toets, niet de markt.

## 11. Wat dit rapport NIET vaststelt

1. **Geen realized variance.** De competitie is beslecht tegen een dagelijkse range-proxy. Met 5-minuts-RV zou de toets meer power hebben en zou een klein effect zichtbaar kunnen worden dat hier onzichtbaar blijft. Zie §2.
2. **HAR-RV is niet getoetst.** `UNPROVEN — insufficient data`, met 0,00 % 5m-dekking. Het model is niet afgewezen.
3. **H1 loopt vast op dezelfde ontbrekende data als HAR-RV.** Een geldige QLIKE-competitie vraagt een proxy die zuiver is voor de voorspelde grootheid. De gekwadrateerde return is dat wel maar is te ruisig om een effect van 0,10 SD te zien; de range-estimators zijn efficiënt maar meten hier aantoonbaar een grotere grootheid (§2.3). Intraday realized variance zou beide oplossen, en die is er niet. Het H1-spoor is daarmee niet beslist maar GEBLOKKEERD, en de blokkade is een inkoopbesluit over data — geen modelleerkeuze.
4. **Geen economisch oordeel.** QLIKE is een statistische maat. Een lagere QLIKE zegt niets over rendement na kosten; dat vergt een run door de authoritative engine met turnover- en fees-delta, en die staat hier niet in.
5. **Twee horizonnen, geen meer.** h = 1 en h = 5 bars. Een langere horizon toevoegen vereist een nieuwe pre-registratie en verhoogt `M` opnieuw.
6. **Per symbool getoetst, niet gepoold.** De pre-registratie toetst per symbool en per horizon. De power-analyse bij bevriezing rekende met 1.265 effectief onafhankelijke reeksen voor een gepoolde toets; de toetsen hier staan elk op één reeks en hebben dus MINDER power dan die scenario's suggereren. Dat is de conservatieve kant: het levert eerder `UNPROVEN` dan `FALSIFIED` op.

---

*Gegenereerd door `apps/run_vol_competition.py` op git_sha `8e6cd02`. Artefact: `artefacts/governance/phase6_h1_competition.json`.*
