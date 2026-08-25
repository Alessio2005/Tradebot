# ARCHITECTURAL DECISIONS

> Beslissingen die niet uit de code zijn af te lezen, met de reden erbij.
>
> Dit register bestaat omdat een keuze zonder motivering na een half jaar
> ononderscheidbaar is van een omissie. Elke entry noemt wat er is besloten,
> welk alternatief is afgewezen, en welke test de beslissing bewaakt.
>
> Aanvullend op `docs/adr/` (per-wave ADR's) en `docs/DEFERRED_ISSUES.md`
> (bekende, bewust uitgestelde problemen).

---

## AD-1 — De risicoconfiguratie staat NIET in de hypothese-ledger

**Fase:** 4 (besluit) · 5 (getest en herbevestigd)
**Status:** actief
**Bewaakt door:** `tests/unit/test_risk_config_is_not_a_hypothesis.py` (13 tests)

### Besluit

Risicoconfiguraties worden geregistreerd in `registry/risk_registry.py`, een
apart append-only register met `config_hash`, `git_sha` en de volledige
gevalideerde configuratie. Zij komen **niet** in
`registry/hypothesis_ledger.py`.

### Waarom

`hypothesis_ledger.total_n_hypotheses()` sommeert `n_trials` over alle rijen, en
dat getal is de `M` waarmee de Deflated Sharpe Ratio deflateert
(`backtest/metrics.py`). Elke rij die daarbij komt, verlaagt de DSR van elk
eerder resultaat dat met een live `M` is gemeten.

Een risicoconfiguratie voorspelt niets. Zij wordt tegen geen nulhypothese
getoetst en verdient geen deflatie. Haar toch daar neerzetten zou betekenen dat
de Phase 3-baseline statistisch slechter wordt omdat iemand een limiet heeft
opgeschreven — een koppeling tussen governance en statistiek die niemand heeft
bedoeld en die niemand zou opmerken.

Dit is scherp geworden in Phase 5: de clusterlabels zijn gewijzigd
(`config_hash` `47821e47fe2cec30` → `1b60cb664fbf9a2a`). Zonder deze scheiding
zou die correctie de DSR van elke Phase 3-track hebben verlaagd.

### Afgewezen alternatief

De letterlijke fase-opdracht van Phase 4 schreef voor: *"registreer de
risicoconfiguratie in de ledger met `config_hash`."* Dat is bewust niet gedaan.
De opdracht bedoelde auditbaarheid; de ledger levert die, maar met een
neveneffect dat de opdracht niet noemde.

### Wat het register wél moet leveren

Dezelfde eigenschappen die de ledger auditbaar maken: append-only, atomair
geschreven, gehasht, idempotent. En de volledige configuratie naast de hash —
een hash zonder inhoud bewijst dat er iets veranderde, niet wat.

---

## AD-2 — `eta` is `IMPACT_UNCALIBRATED`, en dat label reist mee

**Fase:** 5
**Status:** actief tot er orderboekdata is
**Bewaakt door:** `tests/unit/test_impact_model.py` (37 tests)

### Besluit

Het marktimpactmodel draait op een **conservatieve bovengrens** voor `eta`, niet
op een schatting. De status `IMPACT_UNCALIBRATED` staat in
`conf/execution/impact.yaml`, in `ImpactParams`, in elke `ImpactEstimate`, en in
elk `BacktestResult`.

### Waarom

`docs/DATA_REGISTER.md` §6 stelt vast dat er geen orderboek-L1/L2 en geen
trade-prints in de gecertificeerde store staan. Zonder eigen orders en hun
gemeten prijsrespons is `eta` niet identificeerbaar.

Wat wél afleidbaar is: bij `Q = V` geeft het model `Impact = eta * sigma_d`, en
de grootste beweging die die dag optrad is de dagrange. Wordt die volledig aan
impact toegeschreven, dan geldt `eta <= (high-low)/close/sigma_d`. Over 11.660
symbool-dagen is de p95 daarvan **2,99**.

### Wat er eerder stond, en waarom dat erger was

`execution/market_impact.py::square_root_impact(eta: float = 0.142)` en
`conf/execution/fees.yaml::impact_eta: 0.142`. Dat getal komt uit
Bouchaud-Bonart op BTC-perpetuals — een andere venue, een andere periode, een
ander universum — en stond achter een **default**. Een caller die `eta` vergat,
kreeg geen fout maar een plausibel ogend getal. Het veld werd bovendien door
geen enkele module gelezen.

`ImpactParams` heeft daarom geen enkel default-argument en kan niet worden
geconstrueerd zonder volledige herkomst.

### Wat dit rapport eerlijk moet melden

De bovengrens is **ruim maar weinig informatief**. Voor een Brownse beweging is
de verwachte dagrange `1,60 sigma`; de gemeten mediaan is 1,42 en de p95 per
symbool ligt tussen 2,93 en 3,11 terwijl de dagvolumes ordes van grootte
verschillen. Hij meet dus vrijwel uitsluitend volatiliteit, niet liquiditeit.

Hij is desondanks bruikbaar omdat hij op dit boek niet bindt: bij een
participatie van ~1e-7 blijft de impact onder 1 bp. Zie
`reports/TCA_CALIBRATION_REPORT.md` §5.1-5.2.

---

## AD-3 — De half-spread is `SPREAD_ASSUMED`, niet gemeten

**Fase:** 5
**Status:** actief tot er quote-data is
**Bewaakt door:** `tests/integration/test_tca_roundtrip.py::TestProvenance`

### Besluit

De half-spread komt uit `conf/execution/fees.yaml::assumed_half_spread_bps`
(1,0 bp) en draagt de status `SPREAD_ASSUMED`. Een half-spread van nul is
onmogelijk te construeren: `SpreadModel.__post_init__` crasht erop.

### Waarom niet gemeten

Er is geen bid/ask in de gecertificeerde store. De standaard uitweg is een
OHLC-gebaseerde estimator, en Corwin-Schultz (2012) is daarvoor de
referentiemethode. Hij is geïmplementeerd
(`execution/spread.py::corwin_schultz_spread`) en **op deze data verworpen**:

| Symbool | Mediane half-spread | Aandeel negatieve schattingen |
|---|---:|---:|
| BTCUSDT | 32,6 bp | 32,6 % |
| ETHUSDT | 44,3 bp | 33,4 % |
| SOLUSDT | 57,6 bp | 37,2 % |
| LINKUSDT | 65,9 bp | 35,7 % |

Een half-spread van 33 bp op BTC-perps is ongeveer 50× te hoog. De estimator
identificeert spread uit het verschil tussen een- en tweedaagse ranges, en bij
crypto-volatiliteit (dagranges van 5-8 %) wordt dat verschil gedomineerd door
ruis. Het aandeel negatieve schattingen — Corwin & Schultz rapporteren 10-20 %
op Amerikaanse aandelen — bevestigt de breakdown.

De functie blijft bestaan als **meetinstrument**, niet als spreadbron. Zodra er
quotes zijn, vervangt een gemeten spread de aanname en wordt de status
`MEASURED`.

### Afgewezen alternatief

Corwin-Schultz tóch gebruiken omdat hij "op de data steunt". Dat zou een
spreadkost opleveren die alles domineert, en de conclusie van elke backtest zou
een eigenschap van een kapotte estimator zijn in plaats van van de strategie.

---

## AD-4 — De clusterlimiet is vacuous op dit universum, en dat staat er

**Fase:** 5
**Status:** actief tot het universum een tweede cluster heeft
**Bewaakt door:** `tests/integration/test_cluster_feasibility.py` (14 tests)

### Besluit

Alle zes symbolen dragen het label `crypto_perp`. Daarmee geldt
`effective_relative_cap(1, 0.60) = max(0.60, 1/1) = 1.0` en bindt
`cluster_cap` nooit meer. De limiet blijft in `constraint_order`.

### Waarom

De vorige indeling (vijf `crypto_l1`, één `crypto_oracle`) is gemeten en
verworpen:

* scheiding binnen-vs-tussen cluster **+0,0113**, bootstrap 95 %-BI
  **[−0,0131, +0,0380]** — bevat nul;
* het teken **wisselt per jaar**; in 2025 en 2026 is `crypto_l1` sterker
  gecorreleerd met `LINKUSDT` dan met zichzelf;
* jaarlijks herhaalde clustering levert **zes verschillende partities in zes
  jaar**.

Het gevolg was pervers: de limiet duwde `LINKUSDT` van 16,7 % naar **40,0 %**
van het boek — exact op `max_concentration` — en vernietigde 58 % van de
bruto-exposure. Een diversificatiebeperking die het meest geconcentreerde
toegestane boek produceerde.

### Afgewezen alternatief

Herlabelen naar majors/alts (3/3). Dat geeft in-sample een 4× betere scheiding
en 93,8 % target-attainment. Verworpen omdat de jaarlijkse clustering die
indeling niet herstelt: kiezen voor de partitie met het beste
volledige-venstergetal terwijl elk deelvenster iets anders aanwijst, is precies
de in-sample selectie die dit platform elders (DSR, SPA, pre-registratie)
afstraft.

### Wat expliciet gezegd moet worden

Een vacuous limiet is geen werkende limiet. Op dit universum is
`max_concentration` (0,40) de enige werkzame spreidingsbescherming.
`cluster_cap` wordt weer actief zodra er een aantoonbaar tweede cluster is — een
aandelen-, FX- of commodity-sleeve. Een test bewijst dat hij dan wél bindt.

---

## AD-5 — Perps worden volledig gefinancierd geboekt

**Fase:** 5
**Status:** actief
**Bewaakt door:** `tests/unit/test_accounting.py` (29 tests)

### Besluit

`backtest/accounting.py` boekt een perpetual als een volledig gefinancierde
positie: een long van `Q` tegen `P` haalt `Q*P` uit de kas en zet `Q*P` in de
positiewaarde. Er is geen margin-rekening.

### Waarom

Margin verandert *welk deel* van het vermogen vaststaat, niet *hoeveel*
vermogen er is. De hefboomvraag is bovendien al beantwoord door L7 (`gross_cap`)
voordat er een order bestaat. Een gefinancierde weergave sluit exact op
`assets == liabilities + equity`, en die identiteit is wat de hele fase
afdwingbaar maakt.

Bij een short is de positiewaarde negatief; die negatieve waarde is de
verplichting om terug te kopen en verschijnt daarom aan de passiefzijde.

### Uitbreidingspad

Wie echte margin wil boeken, voegt een `margin_requirement`-rekening toe zonder
de identiteit aan te raken.

---

## AD-6 — De venue-funding-cap hoort in de executielaag, niet in de boekhouding

**Fase:** 5
**Status:** actief
**Bewaakt door:** `tests/unit/test_accounting.py::test_funding_is_not_capped_by_the_ledger`

### Besluit

`Ledger.apply_funding()` klemt de funding rate **niet**. De ±2 %-cap zit in
`VenueSpec.cap_funding()` en wordt door de engine toegepast vóór het boeken.

### Waarom

`backtest/portfolio.py:408` deed `np.clip(fr, -0.02, 0.02)` binnen de
PnL-berekening, met de motivering dat de exchange dat ook doet. Die motivering
klopt — maar het is een eigenschap van de **venue**, niet van onze
boekhouding. Een boekhouding die zijn eigen invoer corrigeert, verbergt waar de
correctie vandaan kwam en maakt het onmogelijk te zien hoe vaak zij bond.

---

## AD-7 — De authoritative engine executeert één bar later dan de vectorized baseline

**Fase:** 5
**Status:** actief
**Bewaakt door:** `tests/integration/test_engine_parity.py` (11 tests)

### Besluit

`venue.latency_bars = 1`: een order besloten op de close van bar `t` vult tegen
de prijs van bar `t+1`. `Order.__post_init__` verwerpt elke order die eerder
kan vullen.

### Waarom dit een BESLISSING is en geen implementatiedetail

`backtest/baseline_runner.py` hanteert `held = weights.shift(1)`. Dat betekent:
*beslis op de close van `t`, voer uit op DIE close, verdien vanaf `t+1`*. De
Phase 3-baseline neemt dus aan dat je kunt handelen op de close waarop je
besluit.

De engine doet dat niet, en het verschil is meetbaar. Per-bar RMSE tussen de
engine en de vectorized conventie op elke lag, over drie onafhankelijke reeksen:

| seed | lag 1 | **lag 2** | lag 3 |
|---|---:|---:|---:|
| 1 | 17,49 bp | **1,26 bp** | 20,38 bp |
| 42 | 19,50 bp | **1,03 bp** | 19,03 bp |
| 20260825 | 18,71 bp | **1,06 bp** | 22,43 bp |

De engine matcht `shift(2)`, met 15-20× scheiding van beide buren. Het residu
van ~1 bp is kwantiteit-versus-gewicht-drift: de engine houdt tussen fills een
vast aantal stuks, een gewichtenbacktest herweegt elke bar gratis.

### Gevolg voor de baseline

Elke vergelijking met de Phase 3-cijfers moet dit meenemen. Het is geen
kostenpost maar een **timingverschil**, en het gaat de conservatieve kant op.

---

## AD-8 — Vectorized output is technisch uitgesloten van promotie

**Fase:** 5
**Status:** actief
**Bewaakt door:** `tests/unit/test_vectorized_not_admissible.py` (25 tests)

### Besluit

`VectorizedResult.evidence_class` is een `Literal` met precies één toegestane
waarde: `NOT_ADMISSIBLE_AS_PROMOTION_EVIDENCE`.
`registry/promotion.py::promote()` roept `reject_vectorized_evidence()` aan en
**crasht** op vectorized invoer.

### Waarom een technische weigering en geen richtlijn

Audit §16.1 zegt dat vectorized resultaten nooit als promotiebewijs gelden. Een
richtlijn die niemand afdwingt, houdt het één release vol. Het gevaar zit niet
in het rekenen maar in het **citeren**: een getal zonder herkomst verhuist na
een week naar een rapport, en daar is niet meer te zien dat het uit een engine
kwam die geen spread, geen latency en geen partial fills kent.

De detectie werkt daarom ook op platte dicts en op JSON-roundtrips — dat is de
vorm waarin een getal terugkomt.

### Waarom de engine niet gewoon is verwijderd

Een screening van vijftig hypothesen kost de event-driven engine uren en de
vectorized versie seconden. Het instrument is nuttig; alleen zijn output mag
geen bewijs zijn.

---

## AD-9 — Limit-order queues worden niet gemodelleerd

**Fase:** 5
**Status:** actief tot er intraday orderboekdata is
**Bewaakt door:** de moduledocstring van `execution/order_router.py`

### Besluit

De router plaatst uitsluitend **taker**-orders. Limit-order queues, time-to-fill
en cancellations worden niet gesimuleerd.

### Waarom

Twee onafhankelijke redenen, en beide zijn dwingend:

1. `execution/simulator.py` bevat een LOB-queue-simulator die orderboekdiepte
   nodig heeft. Die data bestaat niet voor dit universum
   (`docs/DATA_REGISTER.md` §6).
2. Op een daily grid is er geen intra-bar tijdas waarop een queue betekenis
   heeft.

Taker-only is bovendien de **conservatieve** keuze: taker betaalt de hoogste fee
en de volledige spread.

### Waarom dit expliciet staat en niet is weggelaten

Fase-opdracht §12 noemt limit-order queues als onderdeel van execution realism.
Zij ontbreken; dat is een databeperking en geen omissie, en het verschil tussen
die twee is precies wat een exit-rapport moet vastleggen.

---

## AD-10 — `min_weight` en `max_turnover` blijven in L8

**Fase:** 5
**Status:** actief
**Bewaakt door:** `tests/unit/test_portfolio_constraints_are_not_sovereign.py` (26 tests)

### Besluit

`portfolio/constraints.py` behoudt `min_weight` (dust-drempel) en
`max_turnover` (turnovercap) als lokale, niet-soevereine constraints.
`max_weight` en `max_leverage` zijn adapters naar `risk.max_concentration` en
`risk.gross_cap`.

### Waarom deze twee legitiem lokaal zijn

Beide **verkleinen** exposure en kunnen de soevereine cap dus niet oprekken. En
beide beantwoorden een andere vraag dan risico: een positie van 0,4 % kost meer
aan fees dan zij bijdraagt, en een turnovercap ruilt tracking error tegen
transactiekosten. Dat zijn kostenbeslissingen.

### De regel die hieruit volgt

Een constraint mag buiten de soevereine laag blijven wanneer hij (a) uitsluitend
verkleint en (b) een kosten- of mechanicavraag beantwoordt in plaats van een
risicovraag. Elke andere constraint verhuist of verdwijnt.
`reports/phase5_sovereign_wiring_audit.md` §3.5 past deze regel toe op alle
negentien gevonden constraints.
