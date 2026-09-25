# FASE 11 — BREEDTE EN TIJDSCHAAL: DE METING

> **Nul trials.** Dit rapport meet geen gemiddeld rendement, geen Sharpe van een
> signaal en geen IC. Alles hieronder is een tweede moment, een eigenschap van
> een besluitpaneel, omzet, of een simulatie op synthetische rendementen met
> gemiddelde nul (R-15). Eén Sharpe wordt **geciteerd**: de gepubliceerde
> L3-Sharpe van `xs_momentum_equal_weight` bij k = 1 uit de ladder, in §3.4.

**Faseopdracht:** `Prompts-fases/fase_11_breedte_en_tijdschaal.md`
**Bron van elk getal:** `artefacts/governance/phase11_breadth.json`, geproduceerd
door `apps/run_breadth_measurement.py` op `src/tradebot/validation/breadth.py`,
`src/tradebot/validation/signal_clock.py`,
`src/tradebot/validation/phase11_breadth_measurement.py` en
`src/tradebot/validation/phase11_decision_clock_feasibility.py`
**Parameters:** `conf/research/breadth.yaml`, gezet vóór de eerste meting van
elke stap; bootstrap uit `conf/validation/inference.yaml` (10.000 replicaties,
seed 20260905, ci 0,95, bloklengte gekalibreerd)
**Risicobeleid:** §1 en §2 gaan niet door de risicolaag. §3 citeert de
L3-kostenmix van `artefacts/baseline/phase5_revaluation.json`, die beleid
`1b60cb664fbf9a2a` draagt; het geldende beleid is `9961e1613bc907a5`. Hoe §3
met dat verschil omgaat, staat in §3.3.

---

## 1. Breedte per constructie, per venster (stap 2)

De vensters zijn afgeleid en niet opnieuw gedefinieerd: `W_FULL` is het
bruikbare venster (ex-ante volatiliteit en causale ADV op elke naam, zoals
H-10.1), `W_DEV` is `development_slice` met de bevroren split, en `W_GATE` is
het complement. Het poortsample is niet via `gate_slice` gelezen;
`holdout_lock.json` draagt nog `reads: []`.

| venster | bars | eerste | laatste |
|---|---:|---|---|
| `W_DEV` | 1.390 | 2021-11-15 | 2025-09-04 |
| `W_GATE` | 353 | 2025-09-05 | 2026-08-23 |
| `W_FULL` | 1.743 | 2021-11-15 | 2026-08-23 |

Twee grootheden per rij, elk onder haar eigen naam (AD-30). **Onafhankelijke
weddenschappen** is de breedte. Het **ontwerpeffect** is wat de repository tot
nu toe "N_eff" noemde (DI-35).

| venster | constructie | rang | ρ̄ | **onafh. weddenschappen** | 95 %-interval | ontwerpeffect | 95 %-interval |
|---|---|---:|---:|---:|---|---:|---|
| `W_DEV` | directioneel | 6 | +0,7403 | **1,601** | [1,516; 1,684] | 1,276 | [1,240; 1,312] |
| `W_DEV` | dollar-neutraal | 5 | −0,1903 | **4,353** | [4,144; 4,500] | 123,58 | [45,60; 335,91] |
| `W_DEV` | bèta-gehedged (EW) | 5 | −0,1872 | **4,615** | [4,437; 4,717] | 93,58 | [47,06; 167,98] |
| `W_GATE` | directioneel | 6 | +0,8200 | **1,369** | [1,296; 1,468] | 1,176 | [1,143; 1,221] |
| `W_GATE` | dollar-neutraal | 5 | −0,1830 | **3,801** | [3,142; 4,336] | 70,67 | [40,34; 137,77] |
| `W_GATE` | bèta-gehedged (EW) | 5 | −0,1752 | **3,763** | [3,271; 4,157] | 48,33 | [30,05; 82,69] |
| `W_FULL` | directioneel | 6 | +0,7485 | **1,576** | [1,503; 1,647] | 1,265 | [1,234; 1,297] |
| `W_FULL` | dollar-neutraal | 5 | −0,1931 | **4,364** | [4,172; 4,503] | 172,71 | [59,69; 501,85] |
| `W_FULL` | bèta-gehedged (EW) | 5 | −0,1891 | **4,573** | [4,415; 4,674] | 109,67 | [55,38; 188,98] |

Vier lezingen.

**1. Zes namen zijn directioneel 1,6 weddenschap.** Het interval [1,516; 1,684]
sluit elke lezing uit waarin het universum "een beetje breed" is. Het
ontwerpeffect van 1,276 is het getal dat fase 10 gebruikte, en voor haar doel
(een gepoolde t defleren) is het juist.

**2. Neutraliseren tegen het mandje verdriedubbelt de breedte, tot vlak onder
de rangbovengrens.** 4,353 van maximaal 5. Het residu heeft vijf bijna gelijk
verdeelde eigenwaarden: de altcoinbewegingen tegen elkaar zijn werkelijk
verschillend, zodra de gezamenlijke factor eruit is.

**3. Het ontwerpeffect is op die constructie geen getal.** Een puntschatting
van 123,58 met een interval van 45,60 tot 335,91 op zes namen meet de nabijheid
van de noemer tot nul en niets anders. In 2025 alleen geeft zij 465,55. Dat is
de kwantitatieve inhoud van DI-35: wie deze grootheid als breedte leest, leest
ruis die een orde van grootte beweegt.

**4. Het poortvenster is directioneel aantoonbaar smaller.** 1,369
[1,296; 1,468] op `W_GATE` tegen 1,601 [1,516; 1,684] op `W_DEV`: de intervallen
overlappen niet. Dollar-neutraal is het verschil kleiner en niet scheiden
(3,801 [3,142; 4,336] tegen 4,353 [4,144; 4,500]). Een hypothese waarvan het
bewijs op directionele breedte leunt, wordt op een smaller universum gepoort
dan zij is ontwikkeld.

### 1.1 Per kalenderjaar (`W_FULL`)

| jaar | bars | directioneel | dollar-neutraal | bèta-gehedged (EW) | ontwerpeffect, dollar-neutraal |
|---|---:|---|---|---|---:|
| 2021 | 47 | 1,635 [1,397; 1,933] | 3,950 [3,345; 4,171] | 4,136 [3,410; 4,369] | 33,05 |
| 2022 | 365 | 1,430 [1,337; 1,500] | 4,131 [3,515; 4,440] | 4,474 [3,987; 4,677] | 46,52 |
| 2023 | 365 | 1,900 [1,680; 2,148] | 3,925 [3,612; 4,124] | 4,371 [4,066; 4,518] | 65,04 |
| 2024 | 366 | 1,738 [1,538; 1,978] | 4,116 [3,584; 4,451] | 4,401 [3,927; 4,635] | 307,73 |
| 2025 | 365 | 1,406 [1,320; 1,522] | 4,292 [3,692; 4,651] | 4,462 [4,031; 4,717] | 465,55 |
| 2026 | 235 | 1,351 [1,252; 1,485] | 4,067 [3,580; 4,398] | 3,863 [3,435; 4,170] | 51,88 |

De directionele breedte beweegt met het regime: 1,90 in 2023, 1,35 in 2026. De
dollar-neutrale breedte blijft binnen elk jaar tussen 3,9 en 4,3, met overlappende
intervallen. **De breedte die een neutraal boek ziet, is stabiel; die van een
richtingsboek niet.** Het ontwerpeffect springt in dezelfde jaren tussen 33 en 466.

### 1.2 Afwijkingen van de nulmeting van de faseopdracht (R-10)

| grootheid | faseopdracht §3.1 | gemeten | verklaring |
|---|---|---|---|
| interval directioneel, PR, `W_DEV` | [1,518; 1,687] | [1,516; 1,684] | 10.000 replicaties met de seed van de inferentiekern tegen 2.000 met een verkenningsseed; bijlage A van de opdracht voorspelde dit |
| interval dollar-neutraal, PR, `W_DEV` | [4,143; 4,513] | [4,144; 4,500] | idem |
| bèta-gehedged tegen BTC | 2,825 | niet in dit artefact | de constructie staat niet in `conf/research/breadth.yaml`; de opdracht noemde haar als diagnostiek, en zij is niet opnieuw gemeten |

Elke puntschatting reproduceert de faseopdracht op drie decimalen.

---

## 2. De signaalklok (stap 3)

Gewichtspanelen van de vier baseline-tracks, gebouwd precies zoals H-10.1 ze
bouwt (`baseline_weight_tracks`), besluitpaneel `weights.loc[usable].fillna(0.0)`,
venster `W_DEV`. Omzet uit `backtest/vectorized.py::run_vectorized` met kosten
nul: de enige omzetdefinitie, tweezijdig en met de instapbar. De schatter is de
geïntegreerde autocorrelatietijd met een Sokal-venster, `c = 5` en maximaal 200
lags, beide gezet in `conf/research/breadth.yaml` vóór deze meting.

| track | unieke rijen | bars met wijziging | omzet per bar | omzet per jaar | ac1 (mediaan) | τ_int (mediaan) | **onafh. besluiten per jaar** |
|---|---:|---:|---:|---:|---:|---:|---:|
| `long_only_equal_weight` | **1** | **0** | 0,0007 | 0,26 | — | ∞ | **0** |
| `long_only_risk_parity` | 1.390 | 1.389 | 0,0177 | 6,46 | 0,976 | ≈ 84,7 † | ≈ 4,3 † |
| `xs_momentum_equal_weight` | 284 | 678 | 0,1537 | 56,10 | 0,946 | **29,96** | **12,18** |
| `xs_momentum_risk_parity` | 1.390 | 1.389 | 0,1769 | 64,58 | 0,947 | 31,02 | 11,77 |

De twee momentumtracks wisselen per naam 21,9 keer per jaar van teken.

**Waar het venster niet werd bereikt.** De mediaan over zes namen verbergt dat
het Sokal-venster bij sommige namen niet binnen 200 lags viel; hun τ is dan een
ondergrens. Per naam:

| track | BTC | ETH | SOL | AVAX | LINK | DOT |
|---|---:|---:|---:|---:|---:|---:|
| `xs_momentum_equal_weight` | 25,5 | 37,5 | ≥ 68,9 | 29,4 | 30,5 | 19,3 |
| `xs_momentum_risk_parity` | 25,5 | 37,9 | ≥ 58,9 | 31,0 | 31,0 | 20,9 |
| `long_only_risk_parity` | ≥ 100,6 | ≥ 132,7 | ≥ 97,7 | 24,4 | ≥ 71,6 | 39,5 |

- **Momentum:** alleen SOL is afgekapt, en SOL ligt boven de mediaan. De
  mediaan (het gemiddelde van de derde en vierde naam, 29,4 en 30,5) hangt er
  niet van af. **De signaalklok van `xs_momentum_equal_weight` is 30 bars.**
  Die redenering is sinds stap 7 code en geen proza meer:
  `signal_clock.median_is_determined` zegt of elke afgekapte naam strikt boven
  de bovenste mediaannaam ligt, en het artefact draagt per track
  `tau_median_determined` en de τ per naam. Voor beide momentumtracks is het
  antwoord ja, voor `long_only_risk_parity` nee.
- **† Risicopariteit long-only:** vier van de zes namen halen het venster niet,
  en de mediaan valt tussen twee afgekapte waarden. Met 600 lags in plaats van
  200 (een gevoeligheidscontrole buiten de config; de config zelf blijft
  staan) dalen die schattingen juist, van 132,7 naar 90,1 bij ETH: de staart
  van de autocorrelatiefunctie is daar ruis. Het getal van ongeveer 85 bars
  (ongeveer 4 besluiten per jaar) is dus een ordegrootte en geen meting op één
  decimaal. De klok van deze track is de EWMA-volatiliteit, niet een signaal.

### 2.1 De beslisklok tegen de signaalklok

`xs_momentum_equal_weight` vormt zijn signaal over 60 bars
(`conf/model/alpha.yaml`) en herziet het elke bar. Het paneel verandert op 678
van de 1.390 bars (178 per jaar) en draagt 56,1 eenheden omzet per jaar, terwijl
de gewichten ongeveer 12 keer per jaar iets onafhankelijks zeggen. **Het boek
betaalt voor 178 herschikkingen per jaar om 12 besluiten uit te drukken.** Of dat
verschil een kostenlek is dat de netto Sharpe meetbaar verandert, is vraag V3,
en stap 7 rekent vóór elke registratie uit of die vraag op deze sample
beslisbaar is.

### 2.2 H-10.1, verklaard zonder nieuwe meting (stap 3.3)

H-10.1 legde de beslisfrequentie op `long_only_equal_weight`. Dat paneel heeft
**één** unieke rij, nul wijzigingen en daarmee een temporele breedte van nul:
τ_int is oneindig, want de reeks zegt na de eerste bar nooit meer iets nieuws.
Een vasthoudoperatie op zo'n paneel is per constructie de identiteit, en het
verschil tegen k = 1 is dan exact nul. Dat staat in het H-10.1-rapport als
gemeten uitkomst.

Wat hier bijkomt, is dat het vóór de registratie uit de gewichtsmatrix te lezen
was. De pre-registratie voorspelde het zelfs ("vrijwel nul"). De ontbrekende
stap was de conclusie: een hypothese die per constructie niets kan meten, hoort
niet geregistreerd te worden. Dat is R-17 van deze fase.

Dit staat in dit rapport en **niet** in de ledger: die is append-only, en het
oordeel over H-10.1 (`UNPROVEN`) verandert er niet door.

### 2.3 Eén vasthoudoperatie (stap 4)

De repository kent twee implementaties van "een besluit k bars vasthouden":
`alpha/momentum.py::CrossSectionalMomentum` met `rebalance_every_bars = k` (houdt
de signaalexposures vast, kalender vanaf het begin van het featurepaneel) en
`portfolio/decision_frequency.py::hold_decision` (houdt de gewichten vast,
blokken vanaf de eerste bar die hij krijgt). `tests/unit/test_hold_equivalence.py`
legt vast waar dat dezelfde operatie is:

| geval | uitkomst |
|---|---|
| gelijkgewogen sizing, gelijk anker | **bit-identiek**: normaliseren per rij laat identieke rijen identiek |
| risicopariteit, gelijk anker | **ander boek**: de sizing leest een volatiliteit die elke bar verandert |
| gelijkgewogen, anker drie bars verschoven | **ander boek**: de fase van de kalender verschilt |
| de geconfigureerde unit (lookback 60, skip 1), k = 10 | **geweigerd door de unit zelf** |

Het laatste geval stond niet in de faseopdracht en is tijdens het schrijven van
de test gemeten. Met de geconfigureerde lookback is de burn-in 61 bars. De
bars tussen het einde van de burn-in en de eerste rebalancebar dragen een
feature, maar hun exposure komt via `ffill` uit een NaN, en de eigen validatie
van de unit slaat daarop aan. Van alle k van 1 tot en met 61 accepteert de unit
er twee: **1 en 61**. 61 is priem, dus de vasthoudoptie van de unit is met de
geconfigureerde lookback voor geen enkele k tussen 2 en 60 bruikbaar.

**Besluit (DI-36):** `hold_decision` is de ene implementatie van "een besluit
vasthouden". Zij heeft een causaliteitstest met negatieve controle en is door
H-10.1 gebruikt. `rebalance_every_bars` blijft op 1 en wordt uit de unit
gehaald zodra een fase de baseline-unit om een andere reden opnieuw afleidt.
`conf/model/alpha.yaml` is in deze fase niet aangeraakt (fence).

De equivalentietest slaagde bij de eerste run, omdat beide implementaties al
bestonden (R-11). De zekerheid komt uit een mutatie: vasthouden met k + 1 in
plaats van k maakt de test rood.

---

## 3. De haalbaarheidspoort van H-11.2 (stap 7)

> **Uitkomst: ROOD.** H-11.2 wordt niet geregistreerd en kost nul trials. Het
> rood hangt niet af van de ladder die de zusterfase nog moet leveren (§3.3).
> Stap 8 vervalt; dat is de vooraf geregistreerde handeling en geen
> overgeslagen stap.

**De hypothese zoals zij geregistreerd zou zijn.** Op `xs_momentum_equal_weight`
verhoogt het vasthouden van het besluit gedurende k\* bars de netto Sharpe op L3
ten opzichte van k = 1, en het verschil overleeft de toets van H-10.1 bij
M_new = 25.

**k\* = ⌈29,9646⌉ = 30**, de signaalklok van §2, gemeten op het besluitpaneel
voordat er bij k = 30 een rendement bestond. `k_star_from_clock` weigert een
klok die oneindig is of waarvan de mediaan van een afgekapte naam afhangt.

### 3.1 De ingrediënten, op `W_DEV` (1.390 bars)

| grootheid | k = 1 | k\* = 30 | bron |
|---|---:|---:|---|
| omzet per bar | 0,1537 | 0,0315 (−79,5 %) | `run_vectorized`, kosten nul, instapbar inbegrepen |
| σ_boek, geannualiseerd | 0,2330 | 0,2355 | `w'Σw`, Σ de covariantie van `W_DEV` |

Het vasthouden gebeurt met `hold_decision`, verankerd op de eerste bruikbare
bar zoals bij H-10.1, en wordt daarna op `W_DEV` gesneden. De kosten per zijde
zijn 6,5 bp (`taker_fee_bps` 5,5 + `assumed_half_spread_bps` 1,0, via
`CostModel.per_side`).

### 3.2 De detectiegrens, met de echte kern (stap 7.2)

200 paden van synthetische rendementen **met gemiddelde nul** uit de
covariantie van `W_DEV` (seed 20260926). Op elk pad lopen beide
gewichtspanelen, uitgevoerd met één bar vertraging, door
`validation/inference.py::sharpe_difference_test`.

| grootheid | waarde |
|---|---:|
| gemiddeld verschil onder de nul | 0,030 (standaardfout van dat gemiddelde 0,030: de simulatie is niet scheef) |
| spreiding van het verschil | 0,430 |
| mediane standaardfout van de kern (Ledoit–Wolf) | 0,434 |
| **kleinste zichtbare verschil, tweezijdig 95 %** | **0,850** |
| kleinste zichtbare verschil, eenzijdig 95 % | 0,713 |
| idem, tweezijdig, uit de spreiding in plaats van de SE | 0,843 |

De benadering `√((2 − 2ρ) / t_jaar)` uit §3.7 van de faseopdracht gaf 0,438
voor de standaardfout. De kern geeft 0,434. De benadering was juist tot op 1 %.

**Tweezijdig is de poort.** De toets van H-10.1 vraagt een interval dat nul
uitsluit. Eenzijdig is de mildste lezing en dient alleen om te laten zien dat
het rood ook daar standhoudt. Beide grenzen zijn nog **niet** gedefleerd bij
M_new = 25; deflatie zou de lat hoger leggen. Het ongedefleerde getal is dus de
gunstigste grens voor de hypothese.

### 3.3 De kostenwinst bij nul signaalverval, en de ladder (stap 7.1)

Op de L0-kostenas levert de omzetdaling bij 6,5 bp per zijde **0,124** Sharpe
op. Op L3 komt daar impact bij. De verhouding tussen de omzetkosten op L3 (fees
+ impact + spread) en die op de L0-as (fees + spread) is de multiplier m.

**De voorwaarde van stap 7 is niet vervuld.**
`artefacts/baseline/phase11_revaluation.json` staat niet op `main` (laatst
nagekeken op `28cc31b`). De enige ladder is `phase5_revaluation.json` onder
beleid `1b60cb664fbf9a2a`. Onder het geldende beleid is het boek groter. Fees
en spread zijn lineair in de grootte en vallen in Sharpe-eenheden weg. Impact
per eenheid schaalt met de wortel van de grootte (`execution/impact_model.py`).
Hoeveel groter het boek kan zijn, staat in het risicoregister
(`artefacts/governance/risk_config_registry.json`):

| boek | schaal | bron | m | **kostenwinst op L3** |
|---|---:|---|---:|---:|
| de ladder zoals zij is | 1 | `phase5_revaluation.json` | 1,540 | **0,192** |
| verwacht onder het geldende beleid | 2,5 | `sigma_target` 0,20 / 0,08 | 1,853 | 0,231 |
| **bovengrens** onder het geldende beleid | 36,0 | `gross_cap` 4,0 / gemeten L1-boek 0,111 | 4,240 | **0,527** |

De poort vraagt m ≥ **6,83** (tweezijdig) of m ≥ 5,74 (eenzijdig). Het grootste
boek dat het geldende beleid toelaat, geeft m = 4,24. **Geen ladder onder
`9961e1613bc907a5` kan dit rood dus omdraaien.** `gate_verdict` noemt dat
`red_under_every_book` en geeft zonder de nieuwe ladder alleen in dat geval een
definitief `RED`; in elk ander geval `PENDING_7_1`.

Stap 7.1 blijft een openstaande handeling. Zodra de zusterfase haar ladder op
`main` heeft, wordt `feasibility.ladder_artefact` in
`conf/research/breadth.yaml` het nieuwe pad en draait de app opnieuw. m wordt
dan gemeten en niet begrensd. Het oordeel kan daardoor niet meer veranderen,
alleen scherper worden.

### 3.4 De poort (stap 7.3)

| lezing | kostenwinst | nodig | uitkomst |
|---|---:|---:|---|
| ladder, tweezijdig (**de poort**) | 0,192 | 0,850 | **rood**, factor 4,4 te klein |
| verwacht boek, tweezijdig | 0,231 | 0,850 | rood, factor 3,7 |
| bovengrens, eenzijdig (de mildste lezing) | 0,527 | 0,713 | rood, factor 1,35 |

**Correctie op de faseopdracht (R-10): het tweede criterium van H-10.1 is geen
gedefleerd interval.** §7.3 en §8.4 van de opdracht noemen het "het 95 %-CI
sluit nul uit na deflatie bij M = 25". In
`conf/experiment/h10_1_decision_frequency.yaml` heet het criterium
`deflated_interval_includes_zero`, maar zijn metriek is
`dsr_development_best_k ≥ 0,95`: de DSR van de **niveau**-Sharpe van de
vastgehouden reeks (`phase10_decision_frequency_measurement.py`, `dsr_gate` op
de primaire reeks), niet van het verschil. Wie de beslisregel "ongewijzigd"
overneemt, neemt dus een niveaucriterium over. Dat geeft een tweede,
onafhankelijke reden voor rood:

| grootheid | waarde | bron |
|---|---:|---|
| niveau-Sharpe die de DSR vraagt (M = 25, N = 1.390, normaliteit) | 1,869 | `dsr_hurdle`, §3 van het muurrapport |
| gepubliceerde L3-Sharpe, k = 1, volle venster, oude beleid | −0,168 | `phase5_revaluation.json`, **geciteerd** |
| afstand | **2,037** | |
| grootste kostenwinst onder enig boek | 0,527 | §3.3 |

De Sharpe van −0,168 geldt over het volle venster en niet over `W_DEV`, en onder
het oude beleid. Een vergelijking op één decimaal draagt zij dus niet. Wel laat
zij zien in welke orde de afstand ligt: een kostenwinst van hooguit 0,53 tegen
een afstand van 2,04.

**Wat de poort veronderstelt.** Zij veronderstelt dat vasthouden het signaal
niet **verbetert**, dus dat de bruto Sharpe bij k\* niet hoger ligt dan bij
k = 1. Dat is geen stelling. Een signaal met ruis die snel terugvalt, kan van
vasthouden profiteren, omdat het vasthouden dan middelt. Dat maakt de poort
niet ongeldig. Een hypothese die alleen haalbaar is als het signaal door
vasthouden béter wordt, is een andere hypothese dan H-11.2. Zij zou een eigen
registratie en een eigen haalbaarheid nodig hebben: een bruto-verbetering van
ten minste 0,85 − 0,19 = 0,66 Sharpe door alleen vasthouden.

### 3.5 Vraag V3, beantwoord als kostenuitspraak

**De beslisklok is op `xs_momentum_equal_weight` een kostenhefboom van 0,12
Sharpe op de L0-kostenas, 0,19 op de L3-kostenmix van de ladder en naar
verwachting 0,23 onder het geldende beleid.** Als kostengrootheid is dat
meetbaar. Als Sharpe-verschil is het op deze sample niet te beslissen: het
kleinste zichtbare verschil is 0,85.

**De breakeven van H-10.1** (`breakeven_cost_bps`, criterium 4) vraagt het
bruto Sharpe-verschil, een eerste moment dat deze fase niet meet. Onder de
aanname van de poort (bruto-verschil nul) is zij per constructie 0 bp: elke
positieve kost begunstigt dan het vasthouden. Dat getal zegt niets. Wat wel iets
zegt, is dezelfde functie met het bruto-verschil op minus de detectiegrens. Dat
geeft de kosten per zijde waarbij de besparing bij nul verval **zichtbaar**
zou worden:

| grens | kosten per zijde |
|---|---:|
| tweezijdig | **44,4 bp** |
| eenzijdig | 37,3 bp |
| ter vergelijking: wat deze repository aanneemt, L0 | 6,5 bp |
| idem, effectief op de L3-mix van de ladder (6,5 × 1,54) | 10,0 bp |

Om het vasthouden op deze sample als Sharpe-verschil te kunnen zien, zouden de
kosten per zijde **4,4 keer** zo hoog moeten zijn als de effectieve L3-kosten
van de ladder, en bijna 7 keer de aanname op L0.

**De verwachting van de faseopdracht (R-10)** was rood, 0 trials, een
kostenwinst van 0,12 tot 0,19 en een eenzijdige grens van 0,72. Gemeten:
rood, 0 trials, 0,124 tot 0,192, en 0,713. De verwachting klopte, en zij
steunde op een benadering die de kern nu tot op 1 % bevestigt.
