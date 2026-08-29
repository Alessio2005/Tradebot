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

