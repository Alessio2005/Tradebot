# ARCHITECTURAL DECISIONS

> **Geverifieerd tegen de codebase op 2026-09-01** (Phase 7/8, Stage E-3).
> AD-1 t/m AD-21; AD-22 t/m AD-24 toegevoegd in fase 10, stap 1 (de drie
> mandaatbesluiten uit `docs/MANDATE.md`). GEMETEN: alle 16 testverwijzingen in dit document
> wijzen naar een bestaand bestand. Dat die tests groen zijn, blijkt uit
> de suite; dat zij het JUISTE bewaken, staat per AD in de tekst.

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

---

## AD-11 — De git-historie krijgt een private GitHub-remote

**Fase:** 7/8, Stage A-1 (uitgevoerd 2026-08-29)
**Status:** actief
**Bewaakt door:** `git ls-remote origin`; `gh repo view --json isPrivate`

### Besluit

De volledige historie staat op een **private** GitHub-repository,
`Alessio2005/Tradebot`, als `origin`. Dit is een besluit van de opdrachtgever;
de keuze tussen een hosted remote, een tweede fysieke schijf of beide lag bij
hem en is expliciet op de hosted variant gevallen.

### Waarom dit geen hygiënepunt was

`reports/phase7_foundation_report.md` §2.3 mat het probleem: 94 commits, het
falsificatieregister, de hypothese-ledger met `M = 2776` en de bevroren
pre-registraties stonden op **één fysieke schijf zonder kopie**. De twee
geverifieerde bundles in `D:/backup/` staan op diezelfde schijf en tellen dus
niet.

Dit platform verklaart elk MRM-rapport, elke ledger-entry en elk
feature-artefact zonder resolvable `git_sha` INVALIDE (audit §26). De hele
bewijsketen hing daarmee aan één schijf. Dat is geen opruimwerk maar een single
point of failure onder de volledige governancelaag, en het goedkoopste risico in
het hele Phase 7/8-document om weg te nemen.

### Wat het besluit NIET is

* **Geen back-up van de data.** `data/pit_store.dvc` verwijst naar een
  DVC-remote die hier los van staat; §11.4 van de fase-opdracht noemt *"waar
  staat de DVC-remote?"* als openstaand punt. Dat blijft open.
* **Geen publicatie.** De repo is privé en dat is na de push geverifieerd, niet
  aangenomen. Een controle vóór de push zou niets zeggen over de staat erna.

### De randvoorwaarde die dit besluit oplegt

Het OAuth-token heeft de `workflow`-scope nodig, want de repo bevat zeven
bestanden onder `.github/workflows/`. GitHub weigert server-side élke push die
ze aanraakt zonder die scope. Ze weglaten is geen alternatief: die workflows
ZIJN de gates die Phase 2 en Stage B hebben opgeleverd.

---

## AD-12 — De gecertificeerde PIT-store staat in git, niet onder DVC

**Fase:** 7/8, nagekomen op Stage A-1 (uitgevoerd 2026-08-29)
**Status:** actief
**Bewaakt door:** `tests/unit/test_pit_store_integrity.py` (8 tests);
`research_gates.yml` stap *"Verify the certified PIT store survived the checkout"*

### Besluit

`data/pit_store/` — 228 bestanden, 2,6 MB, de 18 gecertificeerde reeksen uit
Phase 1 — staat vanaf nu **in git**. `data/pit_store.dvc` blijft bestaan, maar
in een andere rol: het is geen pointer naar een remote meer, maar een
**onafhankelijk integriteitsmanifest**.

### Waarom, en wat er werd gemeten

De eerste CI-run in het bestaan van dit project — mogelijk gemaakt doordat
no-go 2 diezelfde dag werd gesloten (AD-11) — liet `research_gates.yml` stranden
op:

```
$ dvc pull
No remote provided and no default remote set.

$ cat .dvc/config
[core]
    no_scm = True
```

**Er was geen DVC-remote.** Niet onbereikbaar, niet verkeerd geconfigureerd:
hij bestond niet. De store stond daarmee op precies één fysieke schijf — exact
het single point of failure dat no-go 2 voor de git-historie aanwees, alleen dan
voor de data waar elk onderzoeksresultaat naar verwijst.

`docs/runbook.md` §7 beweerde intussen dat de artefacten *"regenerable from
`dvc pull` against the S3/MinIO remote"* waren. Die remote bestond nergens.

### Waarom git en niet een cloud-remote

De opdrachtgever heeft gekozen. De drie opties lagen voor:

| Optie | Prijs |
|---|---|
| **git** (gekozen) | DVC verliest zijn versiebeheerrol voor deze dataset |
| cloud-remote (S3/GDrive/Azure) | opzet, kosten, en een secret in de repo-settings |
| tweede fysieke schijf | beschermt tegen schijfverlies, maar laat CI rood |

Doorslaggevend is de **omvang**: 2,6 MB. DVC bestaat om datasets buiten git te
houden die git onwerkbaar zouden maken; op deze schaal is dat argument er niet,
en de reproduceerbaarheid is met git onvoorwaardelijk. Een clone geeft nu een
draaiend platform, zonder credentials en zonder tweede systeem.

**Dit besluit schaalt niet mee.** Groeit de store naar honderden megabytes — met
1m/5m-bars of L2-snapshots, zie §11.1 van de fase-opdracht — dan moet hij terug
naar een remote. De grens ligt bij de eerste dataset die git merkbaar traag
maakt, en dat is een nieuw besluit, geen automatisme.

### De randvoorwaarde die dit oplegt

Zonder `.gitattributes` zou dit besluit de herkomstketen kunnen breken. 114 van
de 228 bestanden zijn JSON met CRLF-regeleindes. Wie cloont met
`core.autocrlf=true` — de default van menig Windows-installatie — krijgt ze
geconverteerd terug, en de DVC-dirhash klopt dan niet meer op een machine waar
niemand iets heeft gewijzigd.

`.gitattributes` pint daarom `data/pit_store/** -text`. Dat bestand is bewust
smal gehouden: een `* text=auto` zou de 476 getrackte `.py`-bestanden in één
commit normaliseren (DI-17), en dat is een besluit voor Stage E.

### Wat de `.dvc` nu doet

`data/pit_store.dvc` legt vast: 228 bestanden, 2.340.376 bytes, dirhash
`e04fff202fdc77d375a990fa99c43c3d.dir`.
`tests/unit/test_pit_store_integrity.py` rekent die hash na met een EIGEN
implementatie — DVC aanroepen om DVC te controleren valt met het gereedschap om
— en heeft een negatieve controle die precies het CRLF-scenario injecteert.

Zonder die test zou het manifest een achtergebleven artefact zijn dat niets
bewaakt. Dat is de klasse die deze fase overal elders opruimt.
---

## AD-13 — Een proxy die een andere grootheid meet, mag geen model promoveren

**Fase:** 6 (H1, stap 7) · besloten tijdens Phase 7/8 Stage C-3
**Status:** actief
**Bewaakt door:** `tests/unit/test_vol_competition.py::TestProxyScaleRatio` en
`::TestTheProxyPremiseCanOnlyWithholdAPromotion`

### Besluit

`validation/vol_competition.py::judge_challenger` draagt een stop-criterium dat
NIET in de bevroren H1-pre-registratie stond: `proxy_premise_violated`. Het meet
`mean(proxy) / mean(r²)` op de gescoorde bars en zet elk oordeel om in
`UNPROVEN` zodra die verhouding meer dan `adequacy.proxy.max_scale_deviation`
van 1 afwijkt.

### Waarom, en waarom dit geen post-hoc criterium is

QLIKE heeft zijn minimum op `forecast = E[proxy]`. Draagt de proxy een
multiplicatieve factor ten opzichte van de grootheid die de modellen
voorspellen — de variantie van de close-to-close return — dan verschuift dat
minimum mee, en rangschikt de competitie op kalibratie tegen een verschoven doel
in plaats van op voorspelkwaliteit.

Dat is geen theoretische zorg. De eerste H1-run (2026-08-29) mat
`mean(rogers_satchell) / mean(r²)` tussen **1,27 en 2,24**, terwijl EWMA(0.94)
op **1,01** zit en de GARCH-varianten op 1,13 tot 1,70. Op die meetlat
promoveerden 13 van de 48 combinaties — en dezelfde vergelijkingen tegen de
gekwadrateerde return, de enige per-bar proxy die per constructie zuiver is voor
de voorspelde grootheid, gaven p = 0,35 tot 0,75. De promotie zat in het
niveauverschil, niet in de dynamiek.

Een criterium toevoegen ná het zien van de uitkomst is normaal gesproken precies
de manoeuvre die pre-registratie uitsluit. Twee dingen maken dit toelaatbaar, en
zij gelden allebei:

1. **Het toetst een premisse die de pre-registratie zelf uitspreekt.** Die
   rechtvaardigt de range-estimator letterlijk met *"zolang de proxy
   conditioneel zuiver is (Patton 2011) … onder een driftloze GBM binnen de
   dag"*. Die premisse is meetbaar, en is gemeten.
2. **Het kan de conclusie alleen voorzichtiger maken.** `PROMOTED` en
   `FALSIFIED` worden allebei `UNPROVEN`; er bestaat geen invoer waarbij dit
   criterium iets promoveert.

### Het afgewezen alternatief

De primaire proxy achteraf verruilen voor de gekwadrateerde return — dan zou de
uitkomst netjes negatief zijn geweest. Dat is een meetlatwissel na het zien van
de uitslag, en die is niet te onderscheiden van dezelfde wissel met een
omgekeerd motief. De gepre-registreerde meetlat blijft dus staan, mét haar
uitslag, en het oordeel erboven zegt dat zij niet geldig is.

### Wat dit blokkeert

H1 is hiermee niet beslist maar geblokkeerd, op dezelfde ontbrekende data als
HAR-RV: een geldige QLIKE-competitie vraagt intraday realized variance. De
gekwadrateerde return is zuiver maar te ruisig — de negatieve controle laat zien
dat de toets daarmee zelfs een forecast met vernietigde timing niet altijd
onderscheidt. Zie DI-18.

---

## AD-14 — Een oordeel is geen nieuwe zoektocht: de amendement-entry

**Fase:** 6 (H1-oordeel) · besloten tijdens Phase 7/8 Stage C-3
**Status:** actief
**Bewaakt door:** `tests/unit/test_ledger_amendment.py` (6 tests)

### Besluit

`registry/hypothesis_ledger.py::LedgerEntry` kent een veld `amends`. Een entry
die het OORDEEL over eerder geboekte trials herziet, draagt `amends` (de
`config_hash` van de geamendeerde entry) en `n_trials = 0`. `append()` weigert
een amendement dat naar een onbekende entry wijst, en `n_trials = 0` blijft
verboden zónder `amends`.

### Waarom

De 48 trials van H1 zijn geboekt toen de pre-registratie werd BEVROREN — precies
goed: wie een parameterruimte vastlegt, heeft die kansen genomen, en `M` hoort
niet pas te groeien als de uitkomst bevalt. Maar de ledger is append-only en
`total_n_hypotheses()` telt `seed_total + Σ n_trials`. Het oordeel boeken als
een tweede entry met dezelfde 48 trials zou `M` van 2.776 naar 2.824 brengen
voor onderzoek dat één keer is gedaan.

Beide fouten zijn even erg en wijzen tegengesteld: **ondertellen** maakt elke
DSR erna te gunstig, **dubbeltellen** maakt hem te streng, en beide getallen
zijn even onwaar.

`FALSIFICATION_REGISTER.md` liep in Wave 28 tegen exact dezelfde muur en koos
toen voor "geen ledger-entry, wel een registerregel". Dat werkte daar, maar het
liet een gat: de ledger kent dan oordelen niet die het register wel kent. Met
`amends` hoeft die keuze niet meer te worden gemaakt.

### Het afgewezen alternatief

Een bestaande entry muteren (`result: interim` → `archived`). Dat maakt de
ledger niet meer append-only, en dan is de vraag "wat wist men wanneer" niet
meer uit het bestand te beantwoorden — precies de eigenschap waarvoor hij
append-only is.


---

## AD-15 — De regime-overlay is één afbeelding zonder vrije parameter

**Fase:** 6 (H2, stap 11) · besloten tijdens Phase 7/8 Stage C-3
**Status:** actief
**Bewaakt door:** `tests/unit/test_regime_conditioning.py`

### Besluit

Elke conditioneerder — M0, M1 en M2 — wordt op precies dezelfde manier op het
primaire signaal gelegd:

    a_geconditioneerd[t] = a_basis[t] * (1 - p_hoog[t])

M0 levert `p_hoog` als indicator `1{bucket = HOOG}`, M1 als eenstapsvoorspelling
`P(S_t = HOOG | S_{t-1})`, M2 als filtered posterior
`P(S_t = onrustigste toestand | F_t)`. Er is geen multiplier per bucket, geen
drempel op de kans en geen schaling.

De factor wordt daarna per bar gedeeld door zijn grootste waarde — dezelfde
normalisatie die `exposures_from_weights` op de allocatorgewichten toepast.

### Waarom

Elke vrije parameter in die afbeelding zou een trial zijn die niet in
pre-registratie `3d3af28730a6c7f9da48d13139522a05` staat, en zij zou de winnaar
kunnen bepalen zonder dat een van de regimemodellen iets had gedaan. Wat de
modellen mogen doen is uitsluitend `p_hoog` schatten; de rest is bedrading.

De normalisatie hoort erbij om twee redenen. De architecturale: de soevereine
laag herschaalt het boek toch naar zijn vol-target, dus de absolute schaal van
de factor is niet waarneembaar (zie AD-16) — hem laten staan zou suggereren dat
er iets wordt gemeten dat er niet is. De numerieke: zonder normalisatie kan een
bar waarop het filter voor elk symbool `P(hoog) ≈ 1` zegt, een boek van 1e-6 van
de equity opleveren, en daarop is een RELATIEVE risicolimiet niet betrouwbaar te
verifiëren (DI-19).

### Het afgewezen alternatief

Een geoptimaliseerde multiplier per regime (bijvoorbeeld 1,0 / 0,7 / 0,3 voor
LAAG / NORMAAL / HOOG). Dat is drie extra vrije parameters per model, dus zes
extra trials boven de zes geboekte, en het maakt van H2 een sizing-experiment
in plaats van een regime-experiment.

---

## AD-16 — Een regime-overlay kan dit boek niet de-grossen, alleen tilten

**Fase:** 6 (H2, stap 11) · besloten tijdens Phase 7/8 Stage C-3
**Status:** actief
**Bewaakt door:**
`tests/unit/test_regime_overlay.py::TestArmsDifferInOneThingOnly`

### Besluit

Elk H2-resultaat wordt gerapporteerd als CROSS-SECTIONELE TILT en niet als
risicoreductie, en het rapport zegt dat expliciet.

### Waarom — dit is een meting, geen interpretatie

De soevereine laag schaalt het hele boek met
`w_t = min(max_leverage, σ_target / σ_boek)`. Vermenigvuldig elke exposure met
dezelfde `c`, dan deelt `w_t` er weer door. Gemeten op synthetische data: elke
exposure halveren verplaatste de gemiddelde bruto notional van 8.300 naar 8.283
en de turnover met 0,06 %. Gemeten op de echte H2-run: de M0-arm draagt 0,976×
de bruto notional van de ongeconditioneerde arm en 1,140× de turnover.

Wat wél doorkomt is de asymmetrie TUSSEN symbolen — elk symbool heeft zijn eigen
regime — en dat is een herverdeling van het risicobudget: haal het weg bij wat nu
onrustig is en geef het aan de rest. Dat is een zinnig experiment, maar het is
een ander experiment dan "verlaag de risico's in een slecht regime", en een
rapport dat het tweede suggereert terwijl het eerste is gemeten, is onjuist.

### Wat dit uitsluit

De vraag of een regime-overlay het RISICO kan verlagen, is op dit boek niet te
beantwoorden zolang L7 soeverein naar een vol-target herschaalt. Dat is geen
tekortkoming van de overlay maar een eigenschap van de architectuur, en zij
staat als beperking in `reports/M0_VS_HMM_BENCHMARK.md` §10.

---

## AD-17 — Het startpunt van elke EM is deterministisch, niet geseed

**Fase:** 6 (H2, deliverable 15) · besloten tijdens Phase 7/8 Stage C-3
**Status:** actief
**Bewaakt door:**
`tests/unit/test_hmm_fit_contract.py::TestReproducibility`

### Besluit

`regime/markov.py::deterministic_start` vervangt de k-means-initialisatie van
`hmmlearn`: een quantielsplit op `|y − mediaan|`, uniforme start- en
overgangskansen, puur numpy. `GaussianHMM` draait met `init_params=""`, zodat
alleen zijn EM-recursie overblijft. De Student-t EM start van de UITKOMST van
die Gaussische fit.

### Waarom

`sklearn.cluster.KMeans` parallelliseert over OpenMP-threads en de
reductievolgorde ligt niet vast. **Gemeten op 900 bars van twee overlappende
toestanden, hmmlearn 0.3.3, zestig identieke fits met `random_state=20260830`:
twee verschillende uitkomsten.** Niet in de laatste decimaal van een
rapportgetal, maar in de parameters waarop het hele H2-oordeel rust. Een
niet-reproduceerbaar getal is geen bewijs.

De split loopt over de absolute afwijking van de mediaan en niet over de return
zelf, omdat k-means op een 1-D returnreeks splitst op NIVEAU — de negatieve
returns in de ene toestand, de positieve in de andere. Dat is een TEKENsplit, en
een vol-regimemodel gaat over SCHAAL.

De gedeelde start is wat de vergelijking tussen `gaussian` en `student_t` eerlijk
maakt: beide EM's zijn lokaal, dus zonder gedeeld startpunt zou een deel van het
verschil het startpunt zijn in plaats van de emissieverdeling.

---

## AD-18 — Het oordeel over de bezetting gaat over het minimum, niet de mediaan

**Fase:** 6 (H2, stap 11) · besloten tijdens Phase 7/8 Stage C-3
**Status:** actief
**Bewaakt door:** `tests/unit/test_regime_benchmark.py::TestOccupancyIsAbsorbing`

### Besluit

Stop-criterium 1 van de H2-pre-registratie (`rarest_state_obs_per_fold < 100`)
wordt toegepast op het MINIMUM over alle (symbool, fold)-fits van een
conditioneerder. De mediaan en het aantal fits onder de poort worden ernaast
gerapporteerd.

### Waarom

Een walk-forward-evaluatie is alleen zinnig wanneer elke fold erin geldig is.
Een netto Sharpe over 1.200 OOS-bars waarvan een deel voortkomt uit een model
dat op die fold niet gefit had mogen worden, is geen schoon getal — en een
mediaan die de poort haalt, verbergt precies dat.

Het verschil is niet theoretisch. Gemeten in de H2-run: `hmm2-diag-student_t`
heeft een mediaan van 377,3 (ruim boven de eis van 100) en een minimum van 27,97,
met 2 van de 72 fits onder de poort. Op de mediaan zou hij zijn beoordeeld, op
het minimum wordt hij gedescopeerd. De strenge lezing kan per constructie geen
promotie fabriceren, de milde wel.

### Wat er daarom NAAST staat

De mediaan en `n_fits_below_gate` staan in het rapport en in het artefact, zodat
een lezer ziet of het oordeel "dit model kan hier niet" luidt of "deze folds
konden niet". Voor `m1-k2` is dat 66 van 72 fits — het eerste. Voor
`hmm2-diag-student_t` 2 van 72 — het tweede.

---

## AD-19 — De purge gebeurt op `t1`, niet op een embargo-aantal

**Fase:** 6 (H3, stap 12) · besloten tijdens Phase 7/8 Stage C-3
**Status:** actief
**Bewaakt door:** `tests/unit/test_meta_label.py::TestPurging`

### Besluit

`train/meta_label.py::purged_training_index` laat elk trainevent vallen waarvan
het interval `[t, t1]` het testvenster raakt — `exit_bar >= test_start` — en legt
de embargo van 5 bars daar BOVENOP als buffer na het venster. De embargo is dus
niet het mechanisme dat de labeloverlap afdekt; hij dekt de seriële correlatie
na het venster af.

### Waarom

De bevroren pre-registratie draagt `embargo_bars: 5` en `label_horizon_bars: 1`
uit `conf/validation/default.yaml`. Die 1 slaat op het **Phase 3-baselinelabel**,
niet op de verticale barrière van **10 bars** die H3 gebruikt. Een
triple-barrier-label van bar `t` gebruikt bars tot en met `t + 1 + H`; met H = 10
overleeft het een embargo van 5 met ruime marge.

Waren die 5 bars het enige mechanisme geweest, dan had elke fold trainevents
bevat waarvan de uitkomst in het testvenster wordt bepaald — precies het lek dat
no-go 11 verbiedt, en precies het lek dat een meta-labeling-AUC optilt zonder
dat er iets is geleerd.

De purge op `t1` is exact in plaats van conservatief-geschat: `exit_bar` is per
event bekend, dus er wordt weggegooid wat daadwerkelijk overlapt en niet wat een
vuistregel vermoedt. De combinatie is strikt sterker dan "embargo ≥ horizon".

### Wat er daarom NAAST staat

`purged_training_index` telt de drie categorieën apart — events IN het
testvenster, events die erin doorlopen (`n_purged_by_overlap`), en events in de
embargozone — en elke fold rapporteert ze in het artefact. Een purge die niets
wegneemt, is aan die telling te zien; een bewering dat er is gepurged, niet.

De verhouding hoort groot te zijn: met een horizon van 10 bars en een event op
vrijwel elke bar raakt een aanzienlijk deel van de trainrand het testvenster.
Een `n_purged_by_overlap` van bijna nul zou betekenen dat de purge langs de data
heen grijpt.

---

## AD-20 — Er zijn twee effectieve steekproefgroottes, en het oordeel valt op de conservatieve

**Fase:** 6 (H3, stap 12) · besloten tijdens Phase 7/8 Stage C-3
**Status:** actief
**Bewaakt door:**
`tests/unit/test_meta_label.py::TestUniqueness`,
`tests/unit/test_meta_label.py::TestWeightedAuc`

### Besluit

Het Hanley-McNeil-interval om de OOS-AUC wordt op **twee** effectieve
steekproefgroottes gerapporteerd, en het oordeel tegen de drempel van 0,58 valt
op de conservatieve:

* **nominaal** — het aantal gescoorde testrijen;
* **conservatief** — datzelfde aantal maal de gemeten uniqueness maal
  `effective_independent_series / n_symbols` uit de bevroren pre-registratie.

De AUC zelf weegt de uniqueness al mee (`weighted_auc`): hij is de kans dat een
willekeurig getrokken geslaagde trade hoger scoort dan een willekeurig getrokken
mislukte, beide getrokken proportioneel aan hun uniqueness.

### Waarom

Het nominale aantal labels overschat de informatie op twee manieren tegelijk, en
ze stapelen:

1. **Overlap in de tijd.** Met een verticale barrière van 10 bars en een event op
   vrijwel elke bar delen buren negen van hun tien toekomstige bars (AFML
   hoofdstuk 4). De gemeten uniqueness is 0,1622.
2. **Afhankelijkheid in de cross-sectie.** Zes perpetuals met een gemeten
   gemiddelde paarsgewijze correlatie van 0,7485 zijn **1,27** onafhankelijke
   reeksen waard, niet zes.

Welke van de twee de "juiste" n is, volgt niet uit de data: twee labels van
verschillende symbolen op dezelfde bar zijn verschillende trades met
verschillende uitkomsten, maar hun uitkomsten zijn gecorreleerd. Omdat die keuze
niet uit de data volgt, is zij **vóór de run** vastgelegd in de pre-registratie
in plaats van erna gekozen — dat laatste is het werkpunt kiezen op de uitkomst,
één abstractieniveau hoger.

Het nominale interval blijft in het rapport staan, en niet uit volledigheid: het
is het interval dat een pipeline zonder uniqueness-correctie zou publiceren. Het
verschil tussen de twee is de reden dat meta-labeling-resultaten uit de
literatuur zo vaak niet repliceren.

### Wat dit NIET is

Het is geen vrijbrief om het brede interval te gebruiken wanneer het uitkomt en
het smalle wanneer dat beter uitkomt. De pre-registratie wijst één van de twee
aan als bindend, en dat is de conservatieve — in beide richtingen. Haalt de
conservatieve ondergrens de drempel, dan promoveert het model ook wanneer het
nominale interval breder zou zijn geweest.

---

## AD-21 — De CatBoost-pipeline heeft geen scaler en geen imputer

**Fase:** 6 (H3, stap 12) · besloten tijdens Phase 7/8 Stage C-3
**Status:** actief
**Bewaakt door:**
`tests/unit/test_meta_label.py::TestTheModelNeverSeesTheTestWindow`

### Besluit

`fit_secondary_model` fit geen scaler en geen imputer. Niet één per fold, en
zeker niet één over folds heen.

### Waarom dat GEEN schending is van "elke scaler per fold gefit"

Die regel beschermt één ding: dat er niets wordt gefit op data buiten het
trainvenster. Dat contract geldt hier onverkort — het is alleen niet met een
scaler te schenden die er niet is.

CatBoost splitst op ORDENINGEN en is daarmee invariant onder elke monotone
transformatie per feature. Een per-fold gefitte scaler zou het model
aantoonbaar niet veranderen; wat hij wél zou doen, is de indruk van
zorgvuldigheid wekken op een plek waar geen zorg nodig is, en de aandacht
weghalen bij de plek waar zij dat wél is — de purge op `t1` (AD-19).

Ontbrekende waarden worden evenmin geïmputeerd. CatBoost verwerkt NaN zelf; een
imputatie zou een SCHATTING toevoegen die niemand heeft gevraagd en die, per
fold gefit, alsnog een vrijheidsgraad introduceert.

### Wat de plaats van de test-garantie inneemt

`test_the_model_never_sees_the_test_window` vervangt de scaler-per-fold-test
door een sterkere: hij vervangt de testrijen door andere waarden en eist dat het
gefitte model onveranderd blijft. Dat dekt niet alleen de scaler maar élke route
waarlangs testdata de fit zou kunnen bereiken — een `eval_set` voor early
stopping bijvoorbeeld, de meest voorkomende manier waarop een 'out-of-sample'
AUC in-sample wordt doordat het aantal iteraties op de testdata wordt gekozen.

---

## AD-22 — De dagbar is zowel de meet- als de handelsresolutie

**Fase:** 10 (mandaatbesluit B-1, stap 1.2)
**Status:** actief
**Bewaakt door:** `docs/MEASUREMENT_CONTRACT.md` §2; daarnaast de
domeinconsistentiepoort `scripts/check_domain_consistency.py` (de
`RESOLUTION`-scan over `conf/`), gebouwd in stap 2, met
`tests/unit/test_domain_consistency.py::test_a_non_daily_resolution_in_conf_is_refused`
als negatieve controle

### Besluit

Het programma kent **één** barresolutie: de dagbar. Die resolutie geldt voor de
meetketen en voor de handelsketen, en zij zijn daarmee dezelfde keten.

`live/feed.py::FeedConfig.bar_seconds` vervalt. Er is geen configuratieveld meer
waarmee een tweede resolutie kan worden gekozen, want een veld dat er is, wordt
gebruikt.

Wat dit sluit:

* het resolutieverschil tussen de twee ketens (backtest op dagbars, `live/` op
  5-secondenbars — zie `reports/phase7_divergence_map.md`);
* het verschil van 5 tegen 6 namen tussen de twee ketens;
* exit-criterium D7, dat hiermee voor het eerst *definieerbaar* wordt: pariteit
  tussen twee ketens is pas een uitspraak wanneer beide op dezelfde klok lopen.

### Waarom

De handelsklok volgt de meetklok, niet andersom. Elke grootheid waarop dit
programma een besluit neemt, is gemeten op een dagbar; een handelsketen die op
5 seconden loopt, neemt dus 17.280 besluiten per dag op informatie die één keer
per dag verandert. Dat is geen fijnere uitvoering van dezelfde strategie maar
een andere strategie, met een andere kostenstructuur en een ander
microstructuurregime — en géén van beide is gemeten.

De keuze is bovendien de enige die het meetdomein van AD-23 respecteert: er is
geen gecertificeerde sub-daagse bron, dus een 5-secondenketen draait per
constructie op ongecertificeerde data.

### Het afgewezen alternatief

Twee resoluties naast elkaar houden en een adapter bouwen die de dagbar naar de
livefrequentie vertaalt, met een pariteitstest die bewijst dat beide hetzelfde
doen.

Afgewezen omdat zo'n pariteitstest **de configuratie meet in plaats van het
gedrag**. Hij slaagt precies wanneer de adapter de sub-daagse variatie
wegmiddelt, en faalt wanneer die variatie ergens doorwerkt — dus hij is groen
zolang de tweede resolutie niets doet, en rood zodra zij iets doet. Een test die
alleen groen is wanneer de functionaliteit die hij bewaakt inert is, bewaakt
niets. De adapter zou daarmee de divergentie niet oplossen maar verbergen.

---

## AD-23 — Het meetdomein is een whitelist van drie bronnen op één frequentie

**Fase:** 10 (mandaatbesluit B-2, stap 1.3)
**Status:** actief
**Bewaakt door:** de whitelist `conf/governance/measurement_domain.yaml` en de
poort `scripts/check_domain_consistency.py`, beide gebouwd in stap 2, met
`tests/unit/test_domain_consistency.py` (zeven tests, waaronder twee negatieve
controles op een rij in de vorm van het echte register) als bewijs dat de poort
rood kan worden

### Besluit

> Het meetdomein van dit programma bestaat uit precies drie gecertificeerde
> bronnen in de PIT-store, geobserveerd op precies één frequentie:
>
> | Bron | Frequentie | Certificering |
> |---|---|---|
> | OHLCV per symbool | **1 bar per dag** | `data/pit_store/`, hash-gecertificeerd |
> | Funding rate | 8-uurs, geaggregeerd naar dagbar | idem, via `features/microstructure.py` |
> | Open interest | 1 observatie per dag | idem |
>
> Elke grootheid die het programma gebruikt, is een functie van deze drie. Een
> grootheid die een fijnere waarneming vereist dan één bar per dag, is **geen
> uitgestelde vraag maar een niet-bestaande vraag**: er is geen bron voor, er
> komt geen bron voor, en er is geen conditie waaronder zij terugkeert.

De toelaatbaarheidsregel, en dit is de operationele kern van dit besluit:

> Een grootheid is toelaatbaar dan en slechts dan wanneer zij een meetbare
> functie is van de drie bronnen in de tabel, geobserveerd op één bar per dag.
> Toelaatbaarheid wordt bewezen door de bron te noemen, niet door de
> afwezigheid van een verbod.

### Wat er binnen het domein wél mag

Dit is de helft die mensen vergeten, en zonder haar wordt er later een
domeinconforme module gesloopt omdat haar naam verkeerd klinkt:

* **Range-gebaseerde variantieschatters mogen.** `parkinson`, `garman_klass`,
  `rogers_satchell` en `squared_return` in `volatility/realized.py` zijn
  functies van dagelijkse OHLC. Zij blijven, ongewijzigd.
* **Funding en open interest mogen.** `features/microstructure.py` bevat
  uitsluitend `FundingRateMean`, `FundingRateZScore` en
  `OpenInterestLogChange`. Alle vijf features die de authoritative
  15-feature-registry als "microstructuur" labelt, komen hiervandaan en zijn
  dagelijkse grootheden. **De naam van dat bestand is misleidend; de inhoud is
  domeinconform.** Stap 15.3 hernoemt het naar `features/positioning.py` zodat
  de naam de inhoud niet meer tegenspreekt.
* **GARCH, EGARCH, GJR en EWMA mogen.** Het zijn dagelijkse modellen op
  dagelijkse rendementen.

Wat er níet mag, is één ding: een waarneming binnen de dag.

### Waarom

Een verbodslijst is nooit volledig. Zij noemt de vormen die iemand al had
bedacht, en zij nodigt uit om een variant te bedenken die er niet op staat. Een
domein is per constructie volledig: wat er niet in zit, zit er niet in,
ongeacht hoe het heet.

Daaruit volgt de vorm van de poort in stap 2. Zij **whitelist** de drie
bronnen: een heropeningsconditie in `FALSIFICATION_REGISTER.md` is geldig
wanneer zij een bron uit de tabel noemt, en ongeldig wanneer zij dat niet doet.
De poort hoeft niet te weten wat er buiten het domein bestaat — en dat is
precies de eigenschap die een blacklist mist.

### Het afgewezen alternatief

Een blacklist van verboden termen in de scanner: sub-daagse frequenties,
orderboektermen, tick- en intradaybegrippen, en wat er verder bedacht is.

Afgewezen om twee redenen die beide fataal zijn. Zij is **niet volledig** —
elke nieuwe naam voor dezelfde grootheid omzeilt haar, en er is geen manier om
dat te merken. En zij **houdt de uitgesloten ruimte levend** door haar op te
sommen: een document dat de sub-daagse waarneming afschaft maar haar in vijftien
verboden termen blijft beschrijven, maakt haar tot een onderwerp in plaats van
tot een niet-bestaande vraag.

---

## AD-24 — De hypothese-ledger wordt eenmalig gereset

**Fase:** 10 (mandaatbesluit B-3, stap 1.4)
**Status:** actief
**Bewaakt door:** het bevroren resetartefact
`artefacts/governance/ledger_reset.json` (`m_new = 25`, geschreven door stap 3)
en `tests/unit/test_ledger_reset.py` — met name
`test_the_frozen_reset_artefact_is_pinned_and_linked_to_the_ledger` (pint dat
het bestand bestaat, `m_new == 25` draagt, en dat zijn hash gelijk is aan de
`data_hash` op het AD-24-amendement in de ledger) en
`test_the_reset_does_not_make_the_gate_permissive` (de M-gevoeligheidstoets die
rood wordt zodra `m_new` wordt verlaagd)

### Besluit

`M = 2776` is niet langer de trial-teller voor dit programma. De ledger wordt
**eenmalig** gereset, onder acht regels die samen het protocol zijn:

> **R1** `M_new` is niet nul. Het is het vooraf geregistreerde, bevroren aantal geplande trials.
>
> **R2** `FALSIFICATION_REGISTER.md` blijft onverkort bindend, **en dat is de prijs van de reset.** De kennis uit de oude 2776 trials lekt wél door — wie weet dat crypto-XS-momentum faalt, kiest een andere kandidaat dan wie dat niet weet. Dat is selectiedruk en zij verdwijnt niet met de teller. Het register is het geheugen dat haar neutraliseert.
>
> **R3** De oude ledger wordt gearchiveerd, niet verwijderd. Een AD-14-amendement (`n_trials = 0`) legt de reset vast.
>
> **R4** Wie een oude fit hergebruikt, erft zijn trials. Het CPCV-ensemble draagt **2.400** Optuna-trials (200 × 6 symbolen × 2 zijden).
>
> **R5** Eén reset. Een tweede maakt `M` een parameter in plaats van een meting.
>
> **R6** `M_new` wordt bevroren vóór de eerste fit.
>
> **R7** De vijf poorten blijven ongewijzigd.
>
> **R8** De reset is een claim over de *toekomstige* zoekruimte, en die claim is door de reset zelf niet verifieerbaar. Daarom is hij gekoppeld aan het bevroren poortsample uit stap 4B. Een kandidaat die de ontwikkelsample overleeft, wordt exact één keer op het poortsample gemeten; dat is de enige meting in dit programma waarvan `M` per constructie 1 is.

### Waarom

De gemeten grond, en zij is zwakker dan het getal 2776 suggereert.

`seed_total = 2363` is **geen meting maar een reconstructie**, uitgevoerd nadat
het oorspronkelijke logboek verloren was gegaan. De itemisatie staat verbatim in
`docs/WAVE_LOG.md` W20 §0.1 en luidt
`2000 (audit §10/§11) + W14:204 + W15:21 + W16:8 + W17:49 + W18:1 + W19:80 = 2363`.
`registry/trial_counter.py` noemt haar met zoveel woorden een **ondergrens**
("M is een ONDERGRENS, geen exacte telling"), en het artefact zelf labelt haar
als `Conservative floor`. Daarbovenop staan 17 geboekte entries met samen 413
trials; **2363 + 413 = 2776**.

Een teller die een ondergrens is, kan niet worden verlaagd door beter te tellen,
en de promotiepoort die eraan hangt is daarmee rekenkundig onbereikbaar: bij
M = 2776 en `W_FULL` vereist DSR ≥ 0,95 een annualiseerde Sharpe van **2,380**
(`docs/MEASUREMENT_CONTRACT.md` §2.5), tegen een best gemeten track van
**0,156**. Er is geen kandidaat denkbaar die dat haalt, en een poort die niets
kan doorlaten, toetst niets.

De reset lost dat op door de teller opnieuw te definiëren over een **disjuncte**
zoekruimte: dagbars (AD-22) binnen het meetdomein (AD-23). Dat is de enige grond
waarop hij verdedigbaar is — niet dat de oude trials niet zijn gedaan, maar dat
zij in een andere ruimte zijn gedaan.

**De reset geeft geen enkele regel uit F1 t/m F20 vrij.** R2 is geen
formaliteit maar de dragende regel van dit besluit: de selectiedruk uit 2776
trials verdwijnt niet met de teller, alleen het boekhoudkundige spoor ervan.
Het falsificatieregister is wat die druk neutraliseert, en het blijft daarom
onverkort bindend — inclusief elke heropeningsconditie, die per AD-23 bovendien
alleen geldig is wanneer zij een bron uit het meetdomein noemt.

### Het afgewezen alternatief

`M` laten staan op 2776 en de DSR-drempel verlagen, of de poort een
`warn_only`-stand geven zodat een marginaal resultaat alsnog kan promoveren.

Afgewezen omdat dat de poort aanpast aan de uitkomst in plaats van aan de
familie. De DSR is een correctie voor **selectiebias**, en haar parameter `M`
is een uitspraak over hoe breed er is gezocht; die uitspraak mag veranderen
wanneer de zoekruimte verandert, maar de drempel mag niet veranderen omdat het
antwoord tegenvalt. Het verschil is precies het verschil tussen een reset met
een grond (R1-R8, met R8 als externe verificatie) en het uithollen van de enige
poort die dit programma tegen zichzelf beschermt — wat de fences van §7
uitdrukkelijk verbieden.

---

## AD-25 — Een toestand conditioneert de samenstelling, nooit de schaal

**Fase:** 10 (stap 7)
**Status:** actief
**Bewaakt door:** `tests/unit/test_state_mapping.py::test_the_mapping_survives_vol_targeting`
en `tests/unit/test_state_mapping.py::test_a_uniform_multiplier_would_fail_that_same_test`

### Besluit

REGEL V, in de gepreciseerde vorm van `Prompts-fases/fase_10_herstart_dagbars.md`
§4.3. Laat `a_t` de exposurevector zijn en `c_t(i)` een toestandsafhankelijke
factor.

* Is `c_t(i) = c_t` voor alle `i` — cross-sectioneel uniform — dan is de
  afbeelding **per constructie een lege operatie**: L7 herschaalt naar
  σ-target en deelt `c_t` er weer uit. Dit is het geval van AD-15 en het is
  gemeten in AD-16.
* Varieert `c_t(i)` over `i`, dan raakt de afbeelding de **samenstelling** en
  is zij niet leeg. Zij kost dan `k − 1` vrije parameters per toestand, en
  elke daarvan is een trial (R-2).
* De enige cross-sectioneel gedifferentieerde afbeelding **zonder** vrije
  parameter is de **poort**: `c_t(i) ∈ {0, 1}`, met de toestandsverzameling
  die op nul gaat vooraf geregistreerd.

**Bindend gevolg:** een toestand mag uitsluitend via een poort op de exposure
worden afgebeeld (`regime/state_mapping.py::gate_by_state`), en de poort moet
aantoonbaar vol-targeting overleven.

### Waarom

De ruwe vorm van REGEL V uit revisie 1 — "elke vermenigvuldiging is redundant"
— is niet waar, en een AD hoort geen onware bewering te bevatten. AD-16 meet
specifiek een cross-sectioneel UNIFORME factor: elke exposure met dezelfde
constante `c` vermenigvuldigen verplaatste, op synthetische data, de
gemiddelde bruto notional van 8.300 naar 8.283 en de turnover met 0,06 %. Op
de echte H2-run droeg de M0-arm 0,976× de bruto notional van de
ongeconditioneerde arm en 1,140× de turnover. Dat is een meting over een
UNIFORME factor, en zij zegt niets over een GEDIFFERENTIEERDE factor — die
raakt wél de samenstelling en overleeft de herschaling wél.

Uit dat onderscheid volgt de vorm van de poort. Een multiplier per toestand
(`c_t(i) = m_{s_t(i)}`) is een gedifferentieerde afbeelding en dus niet leeg —
maar zij voegt `k − 1` vrije parameters toe (bij drie toestanden: twee), en
elke daarvan is een trial onder R-2. Zonder vooraf geregistreerde
trial-kosten zou dit bestand een sizing-experiment worden — precies het
alternatief dat AD-15 al afwees. De poort is de enige vorm die de
samenstelling raakt zonder die rekening: `c_t(i) ∈ {0, 1}` heeft geen vrije
parameter zodra `flat_states` is geregistreerd.

### Het afgewezen alternatief

`a_t(i) × (1 − p_hoog(i))`, de afbeelding die H2 (AD-16) al gebruikte: een
continue demping op de kans dat symbool `i` in het onrustigste regime zit.
Deze afbeelding is inderdaad cross-sectioneel gedifferentieerd — `p_hoog`
verschilt per symbool — en zij is dus geen lege operatie in de zin van AD-16.

Zij is afgewezen omdat zij, ondanks dat, een vrije-parameterkeuze verbergt: de
vorm van de demping (lineair in `p_hoog`, geen vloer, geen niet-lineariteit)
is zelf een keuze die net zo goed anders had kunnen zijn, en elke variant
daarvan is een trial die niet vooraf is geregistreerd voor DEZE fase. Stap 7
conditioneert op een DISCRETE toestand (`VolState`, drie niveaus) en niet op
een continue kans — dat is precies waarom de poort met exact nul vrije
parameters kan: er is geen kansmodel om te dempen, alleen een lidmaatschapstest
`s_t ∈ flat_states`. Een multiplier per toestand blijft daarmee toegestaan
noch verboden voor een latere fase die hem vooraf registreert en betaalt
(§4.3); hij is hier alleen te duur.

### Bewaakt door twee tests, en de tweede is geen bijzaak

`test_the_mapping_survives_vol_targeting` normaliseert het basisboek en het
gepoorte boek elk op hun eigen bruto exposure — de operatie die L7 uitvoert —
en eist dat zij dan nog verschillen. `test_a_uniform_multiplier_would_fail_
that_same_test` is de negatieve controle: hij past dezelfde normalisatie toe
op een boek dat met een UNIFORME 0,5 is vermenigvuldigd, en eist dat dat boek
ná normalisatie identiek is aan het origineel. Zonder die tweede test bewijst
de eerste niets over de POORT specifiek — alleen dat er iets aan het boek is
veranderd, en een schaalafbeelding verandert ook iets aan het boek totdat L7
het er weer uitdeelt.


<!-- MERGE 2026-09-18: deze AD kwam binnen als AD-13 uit
     claude/handelsplan-assets-7r9sz2, dat vertakte toen AD-12 de laatste was.
     Main heeft sindsdien AD-13 t/m AD-25 vergeven, dus hij is hernummerd naar
     AD-26. Alleen het NUMMER is gewijzigd; het besluit is onaangeroerd. -->

---

## AD-26 — Het risicobudget is eigen kapitaal, geen propfirm-compliance

**Fase:** 9 (mandaatwijziging, buiten de fasenummering — exogeen besluit)
**Status:** actief
**Bewaakt door:** `tests/unit/test_config_contracts.py::TestRiskBudgetIsComplete`
(6 tests) · volledige verantwoording in `docs/RISK_MANDATE.md`

### Besluit

Er wordt niet meer met propfirms gewerkt. De drempels in
`conf/risk/default.yaml` die uitsluitend contractnaleving waren, zijn vervangen
door een eigen-kapitaalbudget: `max_drawdown_pct` 0,08 → 0,25,
`daily_loss_limit` 0,03 → 0,10, de de-grossing-trappen 0,04/0,06 → 0,12/0,18,
`daily_var_limit_pct` 0,02 → 0,05 en `max_position_age_h` 48 → 720.

Daarnaast is de fase-4-regel "waar waarden uiteenliepen is de strengste gekozen"
vervangen door een expliciet vastgelegde risicobereidheid: `sigma_target`
0,08 → 0,20, `max_leverage` en `gross_cap` 1,5 → 4,0, `net_cap` 0,60 → 2,0,
`max_position_pct` 0,25 → 0,80.

### Waarom dit één besluit is en niet twee

Omdat het verleidelijk is er twee van te maken en dan de tweede niet op te
schrijven. Het propfirm-mandaat gaf een *reden* voor strakke getallen; de
opruimregel "strengste wint" gaf een *procedure*. Valt de reden weg, dan blijft
de procedure zonder onderbouwing achter — en een drempel die alleen nog bestaat
omdat hij ooit de strengste van drie toevallige waarden was, is geen
risicobeleid. `docs/RISK_MANDATE.md` §1 labelt daarom elke limiet A (propfirm),
B (interne opruiming) of C (marktfeit), zodat "de propfirm is weg" niet als
blanco cheque voor het hele bestand kan dienen.

### Wat het NIET is

Geen verruiming omdat een resultaat tegenviel — de zet die fase 9 §9 als enige
onbeschadigde norm van dit project aanwijst. De grond is exogeen. Ter controle:
de vier bekende poort-failures (KG-B1/B2/B3 op `cm_carry` en `cm_tsmom`) staan
na dit besluit onveranderd rood, geen backtest is opnieuw gedraaid en geen
ledger-entry is aangeraakt.

En het maakt niets verhandelbaar. Break-even is 16,3-57,8 bps per round trip; de
beste gemeten bruto-edge in deze repo is +4,43 bps. Een negatieve verwachting
harder inzetten vergroot alleen de verwachte verliezen.

### Afgewezen alternatief 1 — alles verruimen omdat de propfirm weg is

`adv_participation_cap` (0,01) en `max_concentration` (0,40) zijn bewust
ONGEWIJZIGD. De ADV-cap is liquiditeit, geen bereidheid: boven ~1 % van de ADV
is de impactterm het hele resultaat, en `conf/execution/impact.yaml` staat op
`IMPACT_UNCALIBRATED` — hem verruimen vervalst de kostenkant van élke meting,
inclusief de break-even waaraan elke kandidaat wordt getoetst.
`max_concentration` was nooit propfirm-afgeleid en is per AD-4 de enige werkzame
spreidingsbescherming zolang `cluster_cap` vacuous is.

### Afgewezen alternatief 2 — de limieten laten staan tot er een edge is

Dit is de aantrekkelijke zet ("ruimte die je niet gebruikt kan geen schade
doen") en hij is fout, om één meetbare reden. `max_position_age_h: 48` sloot
elke positie na twee dagen, terwijl funding carry — de enige
niet-gefalsificeerde kandidaat uit fase 9 §6.3 — bij ~1,95 bps/dag ruim zeven
dagen nodig heeft om alleen de 13,0 bps vaste kosten terug te verdienen. Die
limiet gáf geen ruimte weg: hij garandeerde dat de kandidaat negatief zou meten
om een reden die niets met de markt te maken heeft. Hetzelfde geldt voor
`max_drawdown_pct: 0,08` tegen de enige forward-meting in de repo
(`max_drawdown −0,3817`): die halt beëindigde het pad vóór het venster uit was.

Een limiet die een meting onmogelijk maakt, is geen voorzichtigheid maar een
meetfout met een risicomotivering. Fase 9 §0 regel 5 zegt het algemeen: een
niet-uitgevoerde meting is geen negatieve meting.

### Wat de grens van boven zet

`max_drawdown_pct` kan niet hoger dan 0,30. `risk/stress_test.py:68` zet
`GAP_DOWN_FRACTION = 0.30` en de stress-suite eist dat die schok de halt trípt;
daarboven is het S3-scenario non-bindend en toetst de suite niets meer. 0,25
houdt 5 punten marge. Na dit besluit trípt S3 nog steeds zowel de
`daily_loss_governor` als de `drawdown_breaker` — gecontroleerd met
`apps/run_stress.py`, uitvoer in `artefacts/risk/phase4_stress.json`.

### Het defect dat dit besluit blootlegt en niet oplost

`sigma_target` verruimen is mogelijk INERT. `reports/vol_target_sweep.csv` laat
`realized_vol` identiek 0,0742 zien bij target 0,16, 0,24 én 0,35: de vol-target
bindt daar niet, er zit een andere limiet vóór. Die sweep draaide met
risicoparameters buiten `conf/` om, dus het getal is geen eigenschap van deze
config — maar de vraag welke limiet werkelijk bindt, is open. Een risicolaag
waarvan de primaire schaalparameter niet bindt, is kapot, en dat oordeel staat
los van hoe ruim de getallen zijn. Zie `docs/RISK_MANDATE.md` §2.3 en DI-27.
