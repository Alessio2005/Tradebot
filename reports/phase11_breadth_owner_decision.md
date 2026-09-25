# FASE 11 — BREEDTE: WAT ALLEEN DE EIGENAAR KAN BESLUITEN

> **Nul trials, nul code.** Dit document legt een besluit voor en neemt het
> niet. Elk getal hieronder komt uit `artefacts/governance/phase11_breadth.json`
> of uit een genoemd bestaand rapport; er is voor dit document niets gemeten.

**Faseopdracht:** `Prompts-fases/fase_11_breedte_en_tijdschaal.md`, stap 9
**Onderbouwing:** `reports/phase11_breadth_and_timescale.md` (breedte, klok,
haalbaarheid) en `reports/phase11_ic_wall.md` (de muur)
**Leesvolgorde:** §1 is wat er is, §2 is wat het zou kosten om het te
veranderen, §3 is de keuze.

---

## 1. De gemeten feiten (stap 9.1)

### 1.1 Hoeveel onafhankelijke weddenschappen dit universum per jaar maakt

Breedte is de participatieratio van de eigenwaarden van de correlatiematrix
(`validation/breadth.py::independent_bets`, AD-30), op `W_DEV` (1.390 dagbars,
zes namen), maal het aantal onafhankelijke besluiten per jaar.

| constructie | onafh. weddenschappen | 95 %-interval | per jaar, klok 1 dag | per jaar, klok 30 dagen |
|---|---:|---|---:|---:|
| directioneel | 1,601 | [1,516; 1,684] | 584 | 19,5 |
| dollar-neutraal | 4,353 | [4,144; 4,500] | 1.589 | 53,0 |
| bèta-gehedged (EW) | 4,615 | [4,437; 4,717] | 1.684 | 56,1 |

Een klok van één dag is de bovengrens: dan zegt elk besluit elke dag iets
nieuws. Het enige signaal dat dit programma nu heeft, het momentum van de
baseline, heeft een klok van 30 dagen (§2 van het meetrapport).

### 1.2 De muur: de IC die een signaal minimaal nodig heeft

Bij de DSR-drempel op `W_DEV` (Sharpe 1,869 bij M = 25, N = 1.390, onder
normaliteit):

| klok | directioneel (formule / simulatie) | dollar-neutraal | bèta-gehedged |
|---:|---|---:|---:|
| 1 dag | 0,077 / 0,042 | 0,047 | 0,046 |
| 5 dagen | 0,173 / 0,094 | 0,105 | 0,102 |
| 30 dagen | 0,423 / 0,231 | 0,257 | 0,249 |

Voor de directionele constructie ligt de muur tussen formule en simulatie. De
afwijking hangt af van hoe gecorreleerd de voorspellingsruis is (§2 van het
muurrapport).

### 1.3 Het poortvenster is het smalste

| constructie | `W_DEV` | `W_GATE` |
|---|---|---|
| directioneel | 1,601 [1,516; 1,684] | **1,369** [1,296; 1,468] |
| dollar-neutraal | 4,353 [4,144; 4,500] | 3,801 [3,142; 4,336] |

Directioneel overlappen de intervallen niet. Op `W_GATE` (353 bars) vraagt
t = 2 alleen al een geannualiseerde Sharpe van 2,034.

### 1.4 De breedte over de jaren

| jaar | directioneel | dollar-neutraal |
|---|---:|---:|
| 2022 | 1,430 | 4,131 |
| 2023 | **1,900** | 3,925 |
| 2024 | 1,738 | 4,116 |
| 2025 | 1,406 | 4,292 |
| 2026 (235 bars) | **1,351** | 4,067 |

De directionele breedte daalt sinds 2023 van 1,90 naar 1,35. De dollar-neutrale
breedte blijft binnen 3,9 tot 4,3, met overlappende intervallen.

### 1.5 Wat dit domein aan IC heeft laten zien

Het diagnostische momentumraster van fase 10 (§5.7) liet **−0,024 tot +0,050**
zien, over 55 cellen en bij een standaardfout van 0,011. Het maximum over 55
cellen is door de selectie zelf omhoog gedrukt. Bij een klok van één dag vraagt
de muur 0,042 tot 0,077. Bij elke klok van vijf dagen of meer vraagt zij
minstens 0,094.

### 1.6 En de tijdschaal als hefboom

Het besluit langer vasthouden levert op het momentumboek hooguit 0,19 Sharpe
aan kosten op (0,23 onder het geldende beleid). Het kleinste verschil dat op
deze sample zichtbaar is, is 0,85. H-11.2 is daarom niet geregistreerd (§3 van
het meetrapport). De temporele breedte biedt op deze sample geen uitweg.

---

## 2. Wat de breedte zou veranderen, en wat elk pad kost (stap 9.2)

Drie paden, zonder aanbeveling. Onder elk pad staat wat de gemeten structuur
erover zegt en wat het kost.

### Pad 1 — Meer Bybit-perpetuals

**Wat het oplevert, volgens de gemeten structuur.** De eerste eigenwaarde van
de ruwe correlatiematrix draagt 78,5 % van de variantie, en de gemiddelde
correlatie is ρ̄ = 0,740. Als nieuwe namen dezelfde gemiddelde correlatie
hebben, stijgt de directionele breedte met het aantal namen N maar naar een
plafond. Bij gelijke correlaties is de participatieratio
`N² / ((1 + (N−1)ρ̄)² + (N−1)(1−ρ̄)²)`, en die gaat voor N → ∞ naar
`1/ρ̄² = 1,82`:

| N | directionele breedte bij ρ̄ = 0,740 |
|---:|---:|
| 6 | 1,60 (gemeten: 1,601) |
| 12 | 1,71 |
| 50 | 1,80 |
| ∞ | 1,82 |

Dat is een rekenvoorbeeld onder één aanname: dat nieuwe namen net zo
gecorreleerd zijn als de huidige. Het is geen meting. Alleen de dollar-neutrale
breedte schaalt mee, met hoogstens N − 1. Dat is precies de breedte waarvan het
bewijs onder **F10** zegt dat zij niet betaalt: *raw reversal +0,30 bij 12
namen, +0,06 bij 99* (`docs/FALSIFICATION_REGISTER.md`). F10 verbiedt de claim
"meer namen is meer breedte", niet de meting ervan.

**Wat het kost.**
- **Survivorship (DI-15).** De publieke Bybit-API geeft alleen nog-verhandelde
  instrumenten: 833 van 833 met status `Trading`, nul delistings. Elke
  toegevoegde naam is een overlever, en elke claim op dat universum draagt die
  vermelding. Zolang er geen bron met delisting-historie is, blijft dat zo.
- **Het meetcontract.** `W_FULL` is gedefinieerd als het gebalanceerde paneel
  van deze zes namen (`docs/MEASUREMENT_CONTRACT.md` §2). Een ander universum
  vraagt een nieuwe versie van dat contract. Elk getal van fase 10 en van deze
  fase geldt voor zes namen.
- **Een onevenwichtig paneel.** Namen die later genoteerd zijn, hebben een
  kortere historie. Een breder universum is daardoor al snel onevenwichtig, en
  dan geldt de kost van pad 2 ook hier.
- **Data.** Certificering per naam in het dataregister. De bron zelf valt
  formeel binnen de drie bronnen van AD-23.

### Pad 2 — Een langer venster

**Wat het oplevert (H-10.2, `reports/phase10_h10_2_unbalanced_panel.md`).**
+359 ontwikkelbars (+25,83 %), en de t = 2-drempel daalt van 1,0249 naar
0,9137. De toegevoegde bars dragen echter gemiddeld **3,265** namen, tegen
6,000 op het bestaande venster. `2/√T` veronderstelt bars met gelijke
informatie-inhoud, en die zijn het niet. H-10.2 is daarom `DESCOPED`: de winst
is echt, maar kleiner dan de drempeldaling suggereert.

**Wat het kost.** `RiskEngine.decide` moet een deelverzameling van namen per
bar accepteren. Dat is een wijziging in het risicocontract, en het
risicocontract is op dit moment van de zusterfase (meetbasis en carry). De
wijziging hoort dus ná haar samenvoeging. Aan de breedte per bar verandert
niets: het is hetzelfde universum, alleen eerder begonnen.

### Pad 3 — Breedte over markten

**Wat het oplevert.** Dit is de enige route die de heropeningscondities van
F10 en F20 open laten. F10: *"breedte over markten/premia (mandaat v3) in
plaats van namen binnen één bètafactor"*. F20, tweede les: *"breedte was hier de
bindende beperking, niet het signaal"*. Hoeveel breedte het oplevert, is binnen
dit domein niet te meten. Dat is precies het punt.

**Wat het kost.** Het ligt buiten het meetdomein (AD-23: drie bronnen, één
frequentie, dit universum). Het is dus geen onderzoeksstap maar een
mandaatbesluit, met een nieuw of gewijzigd AD, nieuwe databronnen (inkoop), een
nieuw meetcontract, en naar het precedent van AD-24 vermoedelijk een nieuw
trialbudget.

---

## 3. Het besluit dat alleen de eigenaar kan nemen (stap 9.3)

De gemeten muur laat drie uitkomsten toe. **Deze fase kiest er geen.**

| uitkomst | wat het betekent | wat het vraagt |
|---|---|---|
| **A. Stoppen** | Het programma erkent dat dit domein op deze sample geen signaal kan dragen dat de poort haalt. §1.2 tegen §1.5 is de onderbouwing. | Een afsluitend rapport. Geen code. |
| **B. Het universum binnen Bybit verbreden** | Pad 1, eventueel met pad 2. De directionele breedte blijft onder ongeveer 1,8. De dollar-neutrale breedte groeit, en F10 zegt dat die niet betaalt. | Een nieuw meetcontract, de survivorship-vermelding op elke claim (DI-15), een risicocontract voor een onevenwichtig paneel, en trials uit het resterende budget. |
| **C. Een nieuw mandaat** | Pad 3: breedte over markten of premies. | Een mandaatbesluit buiten AD-23, inkoop van data, en een nieuw meetdomein. |

**Wat er open staat bij het schrijven.** De zusterfase (meetbasis en carry)
leidt de ladder opnieuw af onder het geldende beleid en toetst carry. Haar
uitkomst is nog niet bekend. Zij kan elk van de drie uitkomsten raken, maar de
getallen van §1 hangen er niet van af: zij zijn tweede momenten en
besluitpanelen, en gaan niet door de risicolaag. De enige uitzondering is §1.6,
en daarvan is aangetoond dat de nieuwe ladder het oordeel niet kan omdraaien
(§3.3 van het meetrapport).

**Trialstand.** Fase 10 besteedde 5 van de 25. Deze fase besteedde 0. De
zusterfase heeft er ten hoogste 8 gepland. Voor uitkomst B blijven er dus ten
minste 12 over.
