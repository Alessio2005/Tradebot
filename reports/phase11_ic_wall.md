# FASE 11 — DE IC-MUUR

> **Nul trials.** De muur is berekend uit de gemeten breedte (tweede momenten),
> de drempels van het meetcontract en een simulatie op synthetische rendementen
> uit de covariantie van `W_DEV`. Er is geen IC van enig echt signaal gemeten
> (R-15).

**Faseopdracht:** `Prompts-fases/fase_11_breedte_en_tijdschaal.md`, stap 5
**Bron van elk getal:** `artefacts/governance/phase11_breadth.json`, blok `wall`,
geproduceerd door `apps/run_breadth_measurement.py` op
`src/tradebot/validation/breadth.py` (`dsr_hurdle`, `t_hurdle_sharpe`,
`required_ic`, `simulate_wall`) en
`src/tradebot/validation/phase11_breadth_measurement.py::measure_wall`
**Parameters:** `conf/research/breadth.yaml` (horizonrooster, IC-rooster,
simulatielengte 200.000, seed 20260925, tolerantie 10 %), gezet vóór de eerste
meting; `M_new = 25` uit `artefacts/governance/ledger_reset.json`; DSR-doel
`1 − dsr_alpha` = 0,95 uit `conf/validation/default.yaml`

---

## 1. De drempels

| venster | bars | DSR ≥ 0,95 bij M = 25 | t = 2 |
|---|---:|---:|---:|
| `W_DEV` | 1.390 | **1,868609** | 1,0249 |
| `W_GATE` | 353 | — | 2,0337 |

De DSR-drempel bestond tot deze fase alleen als overgetypt getal: 1,8686 in de
ledger-notitie en het rapport van H-10.1. Zij is nu een functie,
`dsr_hurdle`, die `backtest/metrics.py::deflated_sharpe` numeriek omkeert en
dus geen tweede DSR-formule is (R-3). De eerste test van die functie
reproduceert 1,868609.

**Voorwaarde bij deze drempels:** normaliteit (scheefheid 0, kurtosis 3) en
`sr_variance = 1/n_obs`, de gedocumenteerde benadering. Een werkelijke
rendementsreeks is scheef en dikstaartig, en haar drempel ligt dan anders. Elke
hypothese rekent haar drempel daarom voor haar eigen reeks opnieuw uit met
`dsr_hurdle`; deze tabel is de referentie en niet de poort.

## 2. Houdt de fundamentele wet op deze structuur? (stap 5.2)

Per constructie: synthetische rendementen uit de covariantie van `W_DEV`, een
voorspelling met opgelegde IC, het boek `w = f / Σ|f|`, en de gerealiseerde
Sharpe uit `inference.sharpe_with_se`, naast `IC · √(breedte · 365)`.

| constructie | IC opgelegd | IC gemeten | Sharpe gerealiseerd | wet | **verhouding** |
|---|---:|---:|---:|---:|---:|
| directioneel | 0,02 | 0,0204 | 0,919 | 0,494 | **1,860** |
| directioneel | 0,05 | 0,0498 | 2,224 | 1,204 | **1,848** |
| directioneel | 0,10 | 0,0987 | 4,350 | 2,386 | **1,823** |
| directioneel | 0,20 | 0,1965 | 8,295 | 4,750 | **1,746** |
| dollar-neutraal | 0,02 | 0,0194 | 0,769 | 0,773 | 0,994 |
| dollar-neutraal | 0,05 | 0,0518 | 2,089 | 2,065 | 1,012 |
| dollar-neutraal | 0,10 | 0,1057 | 4,294 | 4,216 | 1,018 |
| dollar-neutraal | 0,20 | 0,2131 | 8,710 | 8,499 | 1,025 |
| bèta-gehedged (EW) | 0,02 | 0,0184 | 0,817 | 0,755 | 1,082 |
| bèta-gehedged (EW) | 0,05 | 0,0477 | 2,134 | 1,960 | 1,089 |
| bèta-gehedged (EW) | 0,10 | 0,0966 | 4,326 | 3,967 | 1,091 |
| bèta-gehedged (EW) | 0,20 | 0,1944 | 8,705 | 7,982 | 1,090 |

**De vooraf vastgelegde beslisregel** (`formula_tolerance = 0,10`): ligt de
verhouding over het hele rooster binnen 10 % van 1, dan is de formule de muur.
Anders is de gesimuleerde afbeelding de muur, gecorrigeerd met de mediane
verhouding.

| constructie | verhouding | uitkomst | correctie |
|---|---|---|---:|
| dollar-neutraal | 0,994 – 1,025 | **de formule is de muur** | 1 |
| bèta-gehedged (EW) | 1,082 – 1,091 | **de formule is de muur** | 1 |
| directioneel | 1,746 – 1,860 | **de simulatie is de muur** | 1,8355 |

**Dit weerlegt een verwachting van de faseopdracht (R-10).** §2.3 van de
opdracht noteerde op grond van een verkenning dat de wet "binnen ongeveer 15 %"
klopt. Voor de twee neutrale constructies is dat juist, en scherper: 2,5 % en
9 %. Voor de directionele constructie is het fout, met een factor 1,8.

**Waarom de directionele constructie afwijkt, en waarom die afwijking van een
modelkeuze afhangt.** De voorspellingsruis in de simulatie is onafhankelijk
over de namen. Op een universum waarin de eerste factor 78,5 % van de variantie
draagt, zijn zes voorspellingen met onafhankelijke ruis op dezelfde factor samen
meer waard dan de breedte van de rendementen alleen zegt: het gemiddelde van
zes ruizige lezingen van één marktbeweging is minder ruizig dan elk ervan. Is de
ruis net zo gecorreleerd als de rendementen, bijvoorbeeld één marktvisie die op
zes namen wordt gekopieerd, dan verdwijnt dat voordeel en gaat de verhouding
naar 1. **De directionele muur ligt daarom tussen de formule en de simulatie.**
Beide staan hieronder. Voor de neutrale constructies speelt dit niet, omdat het
residu geen gezamenlijke factor meer heeft.

## 3. De muur

IC die een signaal minimaal moet hebben, per horizon `h` in dagbars.

### 3.1 Dollar-neutraal (formule = muur; 4,353 weddenschappen op `W_DEV`, 3,801 op `W_GATE`)

| h | weddenschappen per jaar | DSR, `W_DEV` | t = 2, `W_DEV` | t = 2, `W_GATE` |
|---:|---:|---:|---:|---:|
| 1 | 1.588,8 | **0,0469** | 0,0257 | 0,0546 |
| 5 | 317,8 | 0,1048 | 0,0575 | 0,1221 |
| 10 | 158,9 | 0,1482 | 0,0813 | 0,1727 |
| 30 | 53,0 | **0,2568** | 0,1408 | 0,2990 |
| 60 | 26,5 | 0,3631 | 0,1992 | 0,4229 |

### 3.2 Bèta-gehedged tegen het mandje (formule = muur; 4,615 op `W_DEV`, 3,763 op `W_GATE`)

| h | weddenschappen per jaar | DSR, `W_DEV` | t = 2, `W_DEV` | t = 2, `W_GATE` |
|---:|---:|---:|---:|---:|
| 1 | 1.684,3 | 0,0455 | 0,0250 | 0,0549 |
| 5 | 336,9 | 0,1018 | 0,0558 | 0,1227 |
| 10 | 168,4 | 0,1440 | 0,0790 | 0,1735 |
| 30 | 56,1 | 0,2494 | 0,1368 | 0,3006 |
| 60 | 28,1 | 0,3527 | 0,1934 | 0,4251 |

### 3.3 Directioneel (1,601 weddenschappen op `W_DEV`, 1,369 op `W_GATE`)

Formule, en gesimuleerde muur (formule gedeeld door 1,8355):

| h | DSR, `W_DEV`: formule | DSR, `W_DEV`: simulatie | t = 2, `W_DEV`: formule | t = 2, `W_DEV`: simulatie | t = 2, `W_GATE`: formule | t = 2, `W_GATE`: simulatie |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0,0773 | 0,0421 | 0,0424 | 0,0231 | 0,0910 | 0,0496 |
| 5 | 0,1729 | 0,0942 | 0,0948 | 0,0517 | 0,2035 | 0,1108 |
| 10 | 0,2445 | 0,1332 | 0,1341 | 0,0731 | 0,2877 | 0,1568 |
| 30 | 0,4234 | 0,2307 | 0,2322 | 0,1265 | 0,4984 | 0,2715 |
| 60 | 0,5988 | 0,3263 | 0,3284 | 0,1789 | 0,7048 | 0,3840 |

## 4. En wat dit domein ooit heeft laten zien (stap 5.3)

Geciteerd en niet opnieuw gemeten (R-15). De enige IC's die dit programma
binnen het domein heeft gerapporteerd, komen uit het diagnostische
momentumraster van fase 10 §5.7: **−0,024 tot +0,050** over 55 cellen, bij een
standaardfout van 0,011. Op welke horizon elke cel haar IC mat, staat daar niet.

Naast de muur:

- **Bij een klok van één dag** vraagt de DSR 0,047 (dollar-neutraal), 0,046
  (bèta-gehedged) en 0,042 tot 0,077 (directioneel). Het hoogste gerapporteerde
  getal, 0,050, ligt daar net boven of net onder. Het is bovendien het
  maximum over 55 cellen, en dus zelf door selectie omhoog gedrukt.
- **Bij elke klok van vijf dagen of meer** vraagt de DSR minstens 0,094, bijna
  twee keer het hoogste getal dat dit domein ooit heeft gerapporteerd.
- **Bij de klok van het bestaande momentumsignaal** (h ≈ 30, §2 van het
  meetrapport) vraagt de DSR 0,23 tot 0,42, en de t = 2-drempel op het
  poortvenster alleen al 0,27 tot 0,50.

**Wat dat betekent.** Een signaal dat dit domein ooit door de poort brengt,
moet elke dag opnieuw iets zeggen dat 5 % van de cross-sectionele variatie
verklaart, en moet dat doen met een omzet die op een dagklok niet te betalen is
(§3 van het meetrapport en stap 7). Of die combinatie bestaat, is geen vraag die
deze fase kan beantwoorden. Zij heeft ook geen signaal gemeten. Wel stelt zij
vast dat het domein geen ruimte laat voor iets anders. Stage E legt dat voor
aan de eigenaar.
