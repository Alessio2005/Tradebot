# QUANT RESEARCH ARCHITECTURE — MASTER INSTRUCTION

Je werkt rechtstreeks in een bestaande quant trading repository via Claude Terminal.

Je primaire doel is NIET om direct code te schrijven of bestaande modellen opnieuw te bouwen.

Je primaire doel is om eerst de bestaande repository volledig te begrijpen en vervolgens samen met mij te onderzoeken hoe we van een architectuur met losse sleeves kunnen overstappen naar een geïntegreerde, modelgedreven quant-architectuur.

## 1. FUNDAMENTELE RICHTING

We zijn definitief afgestapt van:

* het oude Setup A/B/C-framework
* het propfirm-framework
* losse sleeves als eindarchitectuur

We werken vanuit een Europees gereguleerd perspectief. De exacte juridische/regulatoire interpretatie moet niet door jou worden aangenomen; focus in deze opdracht primair op de technische en quant-architectuur.

De bestaande sleeves zijn GEEN doelarchitectuur meer.

Een sleeve kan:

* verdwijnen
* worden opgesplitst
* worden gecombineerd
* worden omgevormd tot een modelcomponent
* als feature dienen
* als probabilistische voorspeller dienen
* onderdeel worden van een ensemble
* onderdeel worden van een meta-model
* onderdeel worden van een regime/model-selection layer

Denk dus niet:

`meer sleeves → betere strategie`

maar:

`bestaande modellen → interactie → geïntegreerde voorspelling → decision layer → portfolio/risk → economisch rendement`

## 2. REPOSITORY-FIRST PRINCIPLE

De repository bevat al bestaande quant-technieken, modellen en infrastructuur.

Deze bestaande implementaties zijn het primaire uitgangspunt.

Je mag NIET automatisch nieuwe versies bouwen van technieken die al in de repository bestaan.

Voorbeelden van reeds aanwezige quant-technieken kunnen zijn:

* GARCH
* EWMA
* Gradient Boosting
* volume bars
* andere barconstructies
* meerdere labeling-methodes
* feature engineering
* bestaande ML-modellen
* bestaande signalen
* bestaande sleeves
* backtesting
* validatie
* risk management
* portfolio logic

De regel is:

`inspect → understand → reuse → recombine → restructure → improve`

Niet:

`invent → rewrite → backtest`

## 3. ABSOLUTE REGEL VOOR DE EERSTE FASE

BEGIN NIET MET CODEWIJZIGINGEN.

BEGIN NIET MET REFACTORING.

BEGIN NIET MET HET MAKEN VAN NIEUWE MODELLEN.

BEGIN NIET MET HET VERWIJDEREN VAN SLEEVES.

BEGIN NIET MET HET SCHRIJVEN VAN IMPLEMENTATIEPROMPTS.

De eerste fase is uitsluitend:

`repository investigation + quant architecture brainstorm`

Je mag terminalcommando's uitvoeren om de repository te inspecteren.

Lees bestanden.

Zoek naar modellen.

Volg imports.

Analyseer dependencies.

Inspecteer configs.

Inspecteer tests.

Inspecteer bestaande backtest- en validatiecode.

Maar wijzig tijdens deze eerste fase geen bestanden.

## 4. EERSTE ACTIE: REPOSITORY DISCOVERY

Start met een systematische inventarisatie van de repository.

Gebruik geschikte terminalcommando's om onder andere te onderzoeken:

* directory structure
* README's
* configuratiebestanden
* source directories
* model directories
* feature pipelines
* labeling
* data pipelines
* bar generation
* volatility modules
* prediction modules
* strategy/sleeve modules
* portfolio modules
* risk modules
* validation modules
* backtesting
* tests
* scripts
* notebooks indien relevant

Zoek actief naar termen zoals:

`GARCH`
`EWMA`
`volatility`
`volume bars`
`bars`
`label`
`labeling`
`gradient boosting`
`XGBoost`
`LightGBM`
`CatBoost`
`feature`
`meta model`
`ensemble`
`signal`
`sleeve`
`portfolio`
`risk`
`backtest`
`walk forward`
`validation`
`regime`

Gebruik niet blind deze exacte termen; inspecteer ook naamgeving en architectuur die afwijkend is.

## 5. SLEEVES MOETEN WORDEN ONTLEED

Wanneer je sleeves aantreft, behandel ze niet als strategieën die simpelweg behouden moeten blijven.

Onderzoek per sleeve:

* welke modellen erin zitten
* welke features erin zitten
* welke labels worden gebruikt
* welke data wordt gebruikt
* welke horizon wordt voorspeld
* welke outputs worden geproduceerd
* welke filtering plaatsvindt
* welke thresholding plaatsvindt
* welke position sizing plaatsvindt
* welke risk logic erin zit
* welke onderdelen duplicatie bevatten
* welke onderdelen mogelijk herbruikbaar zijn

De centrale vraag is:

> Wat is de daadwerkelijke quant-informatie binnen deze sleeve?

Een sleeve kan bijvoorbeeld blijken te bestaan uit:

`features → model → threshold → sizing`

waarbij alleen het model interessant is en de rest beter gecentraliseerd kan worden.

## 6. BESTAANDE MODELLEN ALS BOUWBLOKKEN

Voor ieder substantieel bestaand model moet je proberen vast te stellen:

* locatie in repository
* modeltype
* input
* output
* target
* horizon
* feature set
* training procedure
* inference procedure
* validatiemethodologie
* afhankelijkheden
* gebruikte data
* mogelijke leakage-risico's
* huidige rol
* potentiële nieuwe rol

Denk daarbij niet uitsluitend aan:

`model = trading strategy`

Een model kan bijvoorbeeld beter functioneren als:

* feature generator
* probability estimator
* specialist model
* regime model
* volatility model
* auxiliary prediction
* ensemble member
* meta-model input
* ranking model
* uncertainty estimator

## 7. ARCHITECTURE DISCOVERY

Nadat je de repository hebt begrepen, onderzoek je verschillende mogelijke architecturen.

Een voorbeeld is:

`data engine`
→ `bar construction`
→ `market state`
→ `volatility`
→ `labeling`
→ `feature engine`
→ `existing prediction models`
→ `model combination`
→ `decision layer`
→ `position sizing`
→ `portfolio`
→ `risk`

Maar dit is GEEN voorgeschreven architectuur.

Onderzoek actief alternatieven zoals:

* parallel model architecture
* ensemble
* stacking
* meta-model
* mixture-of-experts
* hierarchical models
* regime-conditional models
* volatility-conditional models
* dynamic model selection
* multi-horizon models
* cross-sectional models
* time-series models
* probabilistic prediction frameworks

Je moet niet proberen zoveel mogelijk modellen te combineren.

De vraag is:

> Welke combinatie van bestaande componenten kan daadwerkelijk nieuwe informatie of betere economische beslissingen opleveren?

## 8. PREDICTIVE EDGE IS NIET HET EINDDOEL

Maak continu onderscheid tussen:

`predictive edge`
→ `tradable edge`
→ `portfolio edge`
→ `net economic edge`

Een model dat hoge accuracy heeft maar na:

* transaction costs
* slippage
* turnover
* drawdowns
* latency
* sizing constraints
* portfolio constraints

geen aantrekkelijk rendement produceert, is geen succesvol tradingmodel.

Positieve validatie van een sleeve betekent dus NIET automatisch dat deze sleeve behouden moet blijven.

Onderzoek waarom predictive performance mogelijk niet wordt omgezet in economisch rendement.

## 9. BELANGRIJKE ONDERZOEKSVRAGEN

Onderzoek tijdens de brainstorm onder andere:

### Volatility

Kan volatiliteit:

* alleen als feature dienen?
* conditioning informatie leveren?
* regime detection ondersteunen?
* model selection beïnvloeden?
* model weighting beïnvloeden?
* labeling verbeteren?
* position sizing verbeteren?

### Bars

Onderzoek:

* tijdsbars
* volume bars
* andere informatie-gebaseerde bars

en hoe deze de downstream-modellen beïnvloeden.

### Labeling

Onderzoek of verschillende labels:

* verschillende horizons representeren
* verschillende aspecten van een trade meten
* complementaire informatie leveren
* in een multi-task of hierarchical architecture gebruikt kunnen worden

### Existing ML models

Onderzoek of bestaande Gradient Boosting-modellen:

* standalone predictors
* specialist models
* ensemble members
* meta-model inputs
* probability generators

zouden moeten zijn.

### Model interaction

Onderzoek:

* stacking
* ensemble weighting
* model gating
* dynamic model selection
* regime-based routing
* mixture-of-experts
* confidence weighting

## 10. ECONOMISCHE BOTTLENECK

Zoek expliciet naar het verschil tussen:

`model voorspelt goed`

en:

`model verdient geld`

Analyseer mogelijke bottlenecks zoals:

* slechte thresholding
* verkeerde horizon
* slechte labels
* te hoge turnover
* slechte sizing
* correlatie tussen signalen
* onvoldoende regime awareness
* verkeerde timing
* signalen die individueel goed lijken maar gezamenlijk niet complementair zijn
* te veel trades met lage expected value
* geen onzekerheidsmodellering
* onvoldoende portfolio optimization

## 11. DATA LEAKAGE EN RESEARCH BIAS

Elke architectuur moet rekening houden met:

* look-ahead bias
* data leakage
* temporal leakage
* survivorship bias indien relevant
* selection bias
* multiple testing
* data snooping
* overfitting
* regime instability

Je moet bestaande implementaties kritisch controleren op deze punten.

Maak geen claims over betrouwbaarheid zonder de relevante code te inspecteren.

## 12. FASERING

Werk uiteindelijk toe naar de volgende research map.

### PHASE 0 — BRAINSTORM

Doel:

Begrijpen wat we daadwerkelijk proberen te bouwen.

Output:

* huidige architectuur
* huidige sleeves
* beschikbare modellen
* belangrijkste tekortkomingen
* mogelijke geïntegreerde architecturen
* fundamentele ontwerpkeuzes
* belangrijkste onderzoeksvragen

Nog steeds GEEN codewijzigingen.

### PHASE 1 — REPOSITORY INVENTORY

Maak een technisch overzicht van:

`data`
→ `bars`
→ `volatility`
→ `labeling`
→ `features`
→ `models`
→ `signals`
→ `sleeves`
→ `portfolio`
→ `risk`
→ `validation`

Koppel concrete repositorybestanden/modules aan deze onderdelen.

### PHASE 2 — MODEL MAP

Maak een map van alle bestaande modellen.

Voor ieder model:

`input → transformation → model → output → consumer`

Identificeer overlap en complementariteit.

### PHASE 3 — SLEEVE DECOMPOSITION

Breek iedere sleeve conceptueel af tot zijn onderliggende quant-componenten.

Bepaal:

`KEEP`
`COMBINE`
`REPURPOSE`
`REPLACE`
`REMOVE`

Nog geen codewijziging zonder mijn expliciete opdracht.

### PHASE 4 — ARCHITECTURE OPTIONS

Ontwerp meerdere kandidaatarchitecturen op basis van wat daadwerkelijk in de repository aanwezig is.

Vergelijk:

* expected edge
* robustness
* complexity
* overfitting risk
* interpretability
* computational cost
* implementation difficulty
* economic potential

### PHASE 5 — HYPOTHESES

Formuleer testbare hypotheses.

Bijvoorbeeld:

> Volatility-conditioned routing van bestaande modellen levert betere out-of-sample risk-adjusted performance dan vaste modelweging.

Iedere hypothese moet falsifieerbaar zijn.

### PHASE 6 — EXPERIMENT DESIGN

Bepaal exact hoe hypotheses getest worden.

Focus op:

* walk-forward
* purged validation indien relevant
* embargo indien relevant
* out-of-sample
* regime analysis
* transaction costs
* slippage
* turnover
* robustness

### PHASE 7 — INTEGRATED MODEL ARCHITECTURE

Pas nu ontwerpen we de concrete nieuwe architectuur.

De architectuur moet zo veel mogelijk gebruikmaken van bestaande repositorycomponenten.

### PHASE 8 — ECONOMIC OPTIMIZATION

Optimaliseer:

* prediction thresholds
* decision rules
* confidence
* sizing
* volatility scaling
* turnover
* portfolio allocation
* risk

### PHASE 9 — ROBUSTNESS

Uitgebreide validation en stress testing.

### PHASE 10 — PRODUCTION

Pas na voldoende research validation wordt de nieuwe architectuur production-ready gemaakt.

## 13. OUTPUTFORMAT VOOR DE EERSTE FASE

Nadat je de repository hebt onderzocht, geef je mij een eerste research report met exact deze structuur:

# 1. Repository overview

Wat bestaat er daadwerkelijk?

# 2. Existing quant models

Welke modellen zijn aanwezig en wat doen ze?

# 3. Existing sleeves

Welke sleeves bestaan er en waaruit bestaan ze?

# 4. Current architecture

Hoe werkt het systeem momenteel end-to-end?

# 5. Problems with the current sleeve architecture

Waarom is de huidige structuur waarschijnlijk niet optimaal?

# 6. Reusable components

Welke bestaande componenten zijn potentieel zeer waardevol?

# 7. Redundancy

Welke modellen/features/signalen lijken redundant?

# 8. Architecture opportunities

Welke nieuwe combinaties zijn interessant?

# 9. Candidate architectures

Geef meerdere architectuurideeën met voor- en nadelen.

# 10. Key hypotheses

Welke hypotheses zouden we als eerste moeten onderzoeken?

# 11. Recommended research direction

Welke richting verdient volgens jou verder onderzoek en waarom?

# 12. Next research phase

Welke informatie moeten we eerst verder onderzoeken voordat we naar implementatie gaan?

## 14. BELANGRIJKE BEPERKING

Neem NIET aan dat een complexere architectuur automatisch beter is.

Meer modellen ≠ meer edge.

Meer features ≠ meer edge.

Meer ML ≠ meer edge.

Een goed model in een slechte decision architecture kan minder waardevol zijn dan een eenvoudiger model in een goede architecture.

Zoek dus naar:

`information efficiency`
`model complementarity`
`robustness`
`economic value`

en niet naar maximale technische complexiteit.

## 15. START NU

Begin nu met de repository.

Voer eerst alleen inspectie- en analysecommando's uit.

Wijzig geen bestanden.

Bouw nog niets.

Probeer eerst te begrijpen wat er daadwerkelijk aanwezig is.

Daarna rapporteer je je bevindingen volgens het bovenstaande format.

De eerste deliverable is dus:

`REPOSITORY UNDERSTANDING + QUANT ARCHITECTURE BRAINSTORM`

en NIET:

`CODE IMPLEMENTATION`.
