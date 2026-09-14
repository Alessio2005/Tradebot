# RISK STRESS REPORT — L7 Independent Risk & Volatility Targeting

> **Fase:** 4 van 7 · **Deliverable:** 12 · **Prioriteit:** P1
> **Bindend brondocument:** `ARCHITECTUUR_AUDIT_2026-08-22.md` — secties 9.1, 13.1, 14, 14.1, 19, 23 (Phase 4), 24
> **Gegenereerd:** 2026-08-25 · **`git_sha`:** `0b84e4a`
> **`config_hash` van de risicoconfiguratie:** `47821e47fe2cec30`
> **Artefact:** `artefacts/risk/phase4_stress.json` · **Register:** `artefacts/governance/risk_config_registry.json`
> **Reproductie:** `python apps/run_stress.py`

---

## 0. Wat hier gemeten is

Twee dingen, met verschillende bewijslast.

**Deel A — de scenario's (§2).** Vier schokken op de `RiskEngine`. Per scenario:
welke exposure vroeg alpha, welke stond risk toe, welke constraint bond, met
welke gemeten waarde tegen welke geconfigureerde drempel. Dit toetst
exit-criterium 1.

**Deel B — de baseline mét risicolaag (§3-§5).** De Phase 3-baselinetracks zijn
**opnieuw gedraaid** met de echte engine in het pad, bar voor bar, met een
causale `σ̂` uit L2 en een meelopende risicostaat. Niets is nagebootst: waar de
kill switches vuren, vuren ze. Dit toetst exit-criteria 6 en 8.

De baselinekant van §3 reproduceert `reports/BASELINE_BENCHMARK.md` §3.1 tot op
de laatste decimaal. Dat is geen toeval maar de controle: dezelfde tracks,
dezelfde folds, dezelfde kostenaanname. Zonder die identiteit is de
vergelijking betekenisloos.

---

## 1. De geconfigureerde drempels

Alle waarden uit `conf/risk/default.yaml`, gehasht als `47821e47fe2cec30`.

| Sleutel | Waarde | Rol |
|---|---:|---|
| `sigma_target` | 0,08 | doelvolatiliteit van het boek |
| `max_leverage` | 1,5 | cap op de vol-targeting scalar |
| `max_position_pct` | 0,25 | absolute exposure per asset |
| `max_concentration` | 0,40 | aandeel van de gross in één asset |
| `max_cluster_concentration` | 0,60 | aandeel van de gross in één cluster |
| `gross_cap` | 1,5 | `Σ\|w_i\|` |
| `net_cap` | 0,60 | `\|Σ w_i\|` |
| `adv_participation_cap` | 0,01 | fractie van ADV per asset |
| `drawdown_breaker_levels` | 4 % → 0,50 · 6 % → 0,25 | getrapte de-grossing |
| `max_drawdown_pct` | 0,08 | harde HALT op de High-Water Mark |
| `daily_loss_limit` | 0,03 | onmiddellijke kill switch |

Waar de oude boom uiteenliep is consequent de **strengste** waarde gekozen; de
herkomst van elke keuze staat als commentaar in het configuratiebestand.

---

## 2. Deel A — Scenario's S1 t/m S4

Uitgangssituatie: zes symbolen, `a_t = 0,6` elk (gross 3,6), `σ̂ = 72,4 %` — de
gemeten Phase 3-baselinevolatiliteit, geen verzonnen getal. Equity 1,0 op de
High-Water Mark.

| Scenario | Gevraagde gross | Toegestane gross | Reductie | HALTED | Bindende constraints |
|---|---:|---:|---:|:--:|---|
| **S1** Vol-shock (×10) | 3,600 | 0,004604 | **99,87 %** | nee | `vol_target`, `cluster_cap` |
| **S2** Correlatie-instorting | 3,600 | 0,110497 | **96,93 %** | nee | `vol_target` |
| **S3** Gap-down −30 % | 3,600 | 0,000000 | **100,00 %** | **ja** | `daily_loss_governor`, `drawdown_breaker` |
| **S4** Alpha-runaway (a_t = ±1) | 6,000 | 0,046041 | **99,23 %** | nee | `vol_target`, `cluster_cap` |

**In elk scenario is de toegestane exposure aantoonbaar kleiner dan de
gevraagde, met registratie van de bindende constraint.** Exit-criterium 1 is
gehaald.

### 2.1 Het auditspoor, per scenario

Elke regel is herleidbaar tot een gemeten waarde én een `conf/risk/`-sleutel.

| Scenario | Constraint | Scope | Gemeten | Drempel | Configuratiesleutel |
|---|---|---|---:|---:|---|
| S1 | `vol_target` | `__book__` | 26,064 | 0,080 | `risk.sigma_target` |
| S1 | `cluster_cap` | `crypto_l1` | 2,000 | 0,600 | `risk.max_cluster_concentration` |
| S2 | `vol_target` | `__book__` | 2,606 | 0,080 | `risk.sigma_target` |
| S3 | `daily_loss_governor` | `__book__` | 0,300 | 0,030 | `risk.daily_loss_limit` |
| S3 | `drawdown_breaker` | `__book__` | 0,300 | 0,080 | `risk.max_drawdown_pct` |
| S4 | `vol_target` | `__book__` | 4,344 | 0,080 | `risk.sigma_target` |
| S4 | `cluster_cap` | `crypto_l1` | 2,000 | 0,600 | `risk.max_cluster_concentration` |

Ter vergelijking: de voorganger registreerde één string,
`reason="vol_mult=0.412 corr_mult=0.883"`, waaruit noch de bindende drempel
noch haar herkomst af te leiden was.

### 2.2 S2 verdient toelichting — en levert een openstaand punt op

De vol-targeting is **per constructie al bestand tegen S2**. Zij schat de
boekvolatiliteit als de comonotone bovengrens `Σ|w_i|·σ_i`, oftewel onder de
aanname ρ = 1. Wanneer de werkelijke correlaties naar 1 gaan, haalt de
werkelijkheid die aanname in en verandert de schatting niet. Dat is precies
waarom die aanname is gekozen: zij kan alleen te veel de-grossen, nooit te
weinig. Een engine op een geschatte covariantiematrix had zijn schatting juist
op dit moment moeten bijstellen — het moment waarop die schatting het minst
betrouwbaar is.

Maar S2 legt ook iets bloot. Zodra alle assets één cluster vormen, **wordt de
clusterlimiet vacuüm**: je kunt niet over clusters spreiden als er één is. De
afdwingbare vorm van een relatieve cap is `max(cap, 1/n_actief)`, en bij één
cluster is dat 1,0. De bescherming die de clusterlimiet in normale tijden
levert, valt dus juist in een correlatiecrisis weg. Wat overeind blijft is de
per-asset `max_concentration`, en die bindt daar ook aantoonbaar.

Dit is wiskundig onvermijdelijk, geen implementatiefout — maar het hoort in de
hoofdtekst en niet in een voetnoot. Zie §6.2 voor de aanbeveling.

### 2.3 De harness weigert een te ruime kalibratie

`RiskStressHarness.run()` **crasht** wanneer een scenario niets laat binden. Uit
de fase-opdracht: *"Als in een scenario geen enkele limiet bindt, is de
kalibratie te ruim en moet hij worden aangescherpt."* Een stressrapport waarin
alles groen is omdat er niets is getest, is erger dan geen rapport. Dat gedrag
is zelf getest (`tests/integration/test_risk_overrules_alpha.py::
TestTheHarnessRefusesALooseCalibration`).

---

## 3. Deel B — Geometrische Compounding & Variance Drag Evaluatie

### 3.1 Waarom dit de kernvraag van de fase is

`BASELINE_BENCHMARK.md` §3.1 legt het probleem bloot: het 1/N-mandje haalt een
**positieve** Sharpe van +0,156 en verliest tegelijk **48,9 %** van het
kapitaal. Dat is geen fout maar rekenkunde. De Sharpe gebruikt het
**rekenkundige** gemiddelde; kapitaal groeit **meetkundig**, en het verschil is
bij benadering

```
g ≈ μ − σ²/2
```

Bij σ = 72 % is `σ²/2` ruim 26 procentpunt. Volatility targeting grijpt
rechtstreeks op die term aan: het schaalt σ omlaag, en de drag valt
**kwadratisch** mee terug.

> **Methodenoot.** De drag wordt hier **gemeten** als `rekenkundig − meetkundig`,
> niet benaderd met `σ²/2`. Die benadering geldt alleen voor kleine returns en
> breekt precies waar het interessant wordt: bij een gap van −30 % lopen `r` en
> `ln(1+r)` merkbaar uiteen.

### 3.2 De vier tracks, met en zonder risicolaag

OOS-venster: **1.615 bars**, 17 folds, purged walk-forward, identiek aan Phase 3.

| Track | | Ann. rekenkundig | Ann. vol | **CAGR (meetkundig)** | Variantie-drag | Totaalrendement | Max DD | Sharpe |
|---|---|---:|---:|---:|---:|---:|---:|
| `long_only_equal_weight` | baseline | +11,3 % | 72,5 % | **−14,1 %** | 25,4 pp | −48,9 % | 83,5 % | 0,156 |
| | **+ risicolaag** | +0,6 % | **3,2 %** | **+0,6 %** | **0,1 pp** | **+2,5 %** | **5,1 %** | 0,190 |
| `long_only_risk_parity` | baseline | +8,8 % | 67,9 % | **−13,5 %** | 22,2 pp | −47,3 % | 82,2 % | 0,129 |
| | **+ risicolaag** | +0,5 % | **2,9 %** | **+0,4 %** | **0,0 pp** | **+2,0 %** | **4,6 %** | 0,168 |
| `xs_momentum_equal_weight` | baseline | −4,8 % | 26,4 % | −7,9 % | 3,2 pp | −30,6 % | 50,8 % | −0,181 |
| | **+ risicolaag** | +0,0 % | **1,5 %** | **+0,0 %** | **0,0 pp** | **+0,1 %** | **3,1 %** | 0,017 |
| `xs_momentum_risk_parity` | baseline | −10,1 % | 21,0 % | −11,6 % | 1,5 pp | −42,1 % | 56,3 % | −0,483 |
| | **+ risicolaag** | −0,1 % | **1,3 %** | **−0,2 %** | **0,0 pp** | **−0,7 %** | **3,0 %** | −0,109 |

### 3.3 Wat hier staat

**De variantie-drag is geëlimineerd.** Op de 1/N-track: van 25,4 procentpunt
naar 0,1. Op de risk-parity-track: van 22,2 naar 0,0. Dat is het directe,
gekwantificeerde effect waar exit-criterium 6 om vraagt.

**Het meetkundige rendement kantelt van negatief naar positief.** 1/N gaat van
−14,1 % CAGR naar +0,6 %; het totaalrendement over het venster van **−48,9 %
naar +2,5 %**. Hetzelfde signaal, hetzelfde venster, dezelfde kosten — alleen
de exposure is geschaald.

**De drawdown daalt met een factor 16.** Van 83,5 % naar 5,1 %. Exit-criterium
8 stelt: *"een risicolaag die de drawdown niet verlaagt, is verkeerd
gekalibreerd."* Op alle vier de tracks daalt hij met een factor 12 tot 17.

**De Sharpe verandert nauwelijks — en dat hoort.** Vol targeting is voor de
Sharpe bij benadering een schaaltransformatie: 0,156 → 0,190 op 1/N. Wie een
Sharpe-sprong had verwacht, verwachtte alpha van een risicolaag. Die levert
zij niet, en hoort zij niet te leveren.

**De momentumtracks blijven verliezen.** `xs_momentum_risk_parity` gaat van
−42,1 % naar −0,7 %: het verlies is met een factor 60 kleiner, maar het is nog
steeds een verlies, en de Sharpe blijft negatief (−0,109). Dat is de eerlijke
uitkomst. De Phase 3-tracks verliezen **bruto** al geld (bruto Sharpe −0,284);
geen risicolaag repareert een signaal zonder edge. Wat de laag doet is het
verlies begrenzen, niet omkeren.

**De turnover daalt fors.** Op `xs_momentum_risk_parity` van 0,179 naar 0,010
per bar — een gevolg van de kleinere posities. Dat verlaagt de kostenpost
navenant, maar verandert niets aan de afwezige edge.

### 3.4 Welke limieten in de baseline-overlay bonden

| Track | `vol_target` | `cluster_cap` | `drawdown_breaker` | `concentration_cap` | HALTED-bars |
|---|---:|---:|---:|---:|---:|
| `long_only_equal_weight` | 1.611 | 8.055 | 227 | — | **0** |
| `long_only_risk_parity` | 1.611 | 8.055 | 73 | — | **0** |
| `xs_momentum_equal_weight` | 1.611 | 7.935 | — | — | **0** |
| `xs_momentum_risk_parity` | 1.611 | 7.935 | — | 268 | **0** |

`cluster_cap` telt één registratie per geraakt symbool, `vol_target` één per
bar; vandaar de schaalverschillen. `vol_target` bindt op 1.611 van de 1.615
bars: op de eerste vier is de EWMA-schatting nog niet eindig, en daar wordt
niet gehandeld in plaats van geraden — er komt dus ook geen besluit en geen
registratie.

**Nul HALTED-bars.** De getrapte de-grossing greep 227 respectievelijk 73 keer
in op de long-only-tracks en hield de drawdown daarmee onder de harde grens van
8 %. De breaker deed dus precies waarvoor hij is ontworpen: ingrijpen in het
traject waar dat nog goedkoop is, zodat de kill switch niet nodig is.

---

## 4. Welke limiet doet het werk? — sensitiviteit

Beide dominante limieten zijn **dragend**. Zet er één uit en de baseline haalt
zijn harde grens en halt permanent.

| Variant | Gem. gross | Realized vol | CAGR | Totaalrendement | Max DD | HALTED-bars (van 1.615) |
|---|---:|---:|---:|---:|---:|---:|
| **Zoals geleverd** | 0,0457 | 3,23 % | **+0,56 %** | **+2,51 %** | 5,06 % | **0** |
| `max_cluster_concentration = 1,0` | 0,0107 | 2,66 % | −1,30 % | −5,64 % | 8,12 % | **1.413** |
| Vol-target uit (`σ_target = 1,0`) | 0,0041 | 3,66 % | +0,60 % | +2,68 % | 5,91 % | **1.600** |

De laatste twee rijen zien er op rendement misleidend goed uit: die boeken zijn
**bevroren**, niet presterend. Ze halen hun drawdown- respectievelijk
dagverlieslimiet vroeg in het venster en handelen daarna niet meer. Zonder
vol-targeting is het boek al na ~11 bars dood.

**De rekensom achter de gerealiseerde vol.** Vol-targeting alleen zou uitkomen
op `w_t = 0,08 / 0,724 = 0,111`, wat bij een baselinevolatiliteit van 72,5 %
een gerealiseerde vol van `0,111 × 72,5 % ≈ 8,05 %` geeft — exact het target.
De clusterlimiet knijpt de gross verder terug naar 0,046, en daarmee de vol
naar 3,2 %. **De clusterlimiet, niet de vol-target, is op dit universum de
dominante constraint.** Zie §6.2.

---

## 5. Kill switches: irreversibiliteit

| Eigenschap | Bewijs |
|---|---|
| Overleeft een procesherstart | `HaltStore` schrijft naar schijf; een vers object op hetzelfde pad ziet de halt |
| Herstelt niet bij marktherstel | Equity terug naar een **nieuwe** High-Water Mark laat de halt staan |
| Herstelt niet bij een nieuwe handelsdag | Een verse `day_start_equity` laat de halt staan |
| Eerste oorzaak blijft staan | `engage()` overschrijft een bestaande halt nooit |
| Alleen handmatig op te heffen | `release()` eist operator én motivering, en journaliseert beide |
| Corrupte state ≠ toestemming | Leeg, afgekapt of niet-object bestand crasht |

Bewezen in `tests/unit/test_kill_switch_irreversibility.py` (23 tests) en in
S3 hierboven, inclusief een herstart-scenario met herstelde markt.

---

## 6. Openstaande punten

### 6.1 De momentumtracks hebben geen edge
De risicolaag verkleint het verlies met een factor 60, maar keert het niet om.
Bruto Sharpe −0,284: het signaal verliest vóór er één bps aan kosten af gaat.
Dat blijft de bevinding van Phase 3 en is geen risicoprobleem.

### 6.2 De clusterlabels zijn een governance-keuze met een scherpe kant
Vijf van de zes symbolen dragen het label `crypto_l1`, met een clusterlimiet van
0,60. Dat forceert het zesde symbool (`LINKUSDT`, het enige `crypto_oracle`)
naar 40 % van de gross — exact op zijn eigen `max_concentration`. Twee gevolgen:

1. het boek kan zijn volatiliteitstarget van 8 % nooit halen; het blijft op
   ~3,2 % steken (§4);
2. in een correlatiecrisis (S2) valt de clusterlimiet weg en is de per-asset
   concentratielimiet de enige overgebleven spreidingsbescherming.

**Aanbeveling voor Phase 5:** herzie de clusterindeling op een breder universum,
of accepteer expliciet dat het vol-target op dit universum niet bindend is en
verlaag `sigma_target` navenant. De huidige situatie is conservatief — het boek
draait onder target, nooit erboven — maar zij is niet ontworpen, zij is een
neveneffect van de labeling.

### 6.3 De liquiditeitslimiet is niet getest op realistische schaal
`adv_participation_cap` staat in het besluitpad en wordt in elke run
geverifieerd, maar bij een equity van één eenheid bindt hij nergens. Hij is
gedekt door unit-tests, niet door de baseline-overlay.

### 6.4 Survivorship bias (DI-15) staat open
Elk cijfer in §3 draait op zes ex-post gekozen overlevers. De vergelijking
tussen baseline en overlay is daar niet gevoelig voor — beide kanten draaien op
hetzelfde universum — maar de absolute niveaus zijn dat wel.

### 6.5 GARCH blijft Phase 6
De vol-estimator is EWMA (λ = 0,94) uit `conf/model/volatility.yaml`. De
competitie tussen vol-modellen is expliciet Phase 6.

---

## 7. Reproductie

```
python apps/run_stress.py
```

Levert `artefacts/risk/phase4_stress.json` met elk besluit, elke bindende
constraint en beide compounding-profielen per track, en registreert de
configuratie onder `config_hash` in
`artefacts/governance/risk_config_registry.json`.

Getoetst door 314 tests, waarvan 33 integratietests die exit-criterium 1
scenario voor scenario bewijzen.
