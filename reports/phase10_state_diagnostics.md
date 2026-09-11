# FASE 10 — STAP 6: DE TOESTANDSDIAGNOSE

> **Status:** diagnostiek, expliciet niet-promoveerbaar. Er wordt in deze
> stap niets geselecteerd — geen kwantiel, geen `low_q`/`high_q`, geen
> parameter is op grond van onderstaande getallen gewijzigd — dus kost zij
> **nul trials** (`selects_nothing: true`, `trials: 0` in het artefact).

**Gegenereerd:** 2026-09-11
**git_sha (meting):** `5679090`
**Meetvenster (dit rapport):** ontwikkelsample, 2021-11-15 → 2025-09-04,
**1.389 rendement-bars** (1.390 σ̂-bars), 6 symbolen, **3,8055 j**
**Volledig venster (§4.5-basislijn):** 2021-11-15 → 2026-08-23, 1.743 bars,
**4,7753 j**
**Bron:** `artefacts/governance/phase10_state_diagnostics.json`, geproduceerd
door `apps/run_state_diagnostics.py` op `src/tradebot/regime/state_diagnostics.py`
**Faseopdracht:** stap 6 (`Prompts-fases/fase_10_herstart_dagbars.md`), §4.5,
§4.5.3 (H1-eindstand n.v.t. hier), Q7, R-8, R-10

---

## 1. De twee assen naast elkaar

Beide assen komen uit dezelfde meting (`runs.lagged`, de toewijzing met het
lag-contract van stap 5 — de toewijzing waarop daadwerkelijk gehandeld zou
worden). Alle standaardfouten en t-statistieken komen uit
`validation/inference.py::clustered_mean`; deze module rekent er zelf geen uit
(R-3).

### 1.1 Volatiliteitsas — `separation_vol[*].ann_vol`

| Toestand | n (bars) | ann. volatiliteit | 95 %-BI |
|---|---:|---:|---:|
| LAAG | 2.701 | **63,7 %** | [59,2 %, 68,0 %] |
| NORMAAL | 3.293 | **80,9 %** | [76,0 %, 85,5 %] |
| HOOG | 846 | **118,4 %** | [83,9 %, 144,9 %] |

De toestand scheidt op deze as duidelijk: LAAG tegen HOOG is een factor
**1,86**. Het betrouwbaarheidsinterval van HOOG is fors breder dan dat van de
andere twee toestanden — HOOG heeft de minste bars (846) en de minste
episodes (80), en de deviatie is een kwadraat, dus gevoeliger voor de
zwaarste staart van de verdeling.

### 1.2 Richtingsas — `separation_return[*]`, drie lezingen

| Toestand | ann. rendement | 95 %-BI | t (gepoold) | t (geclusterd op datum) | t (N_eff-gedefleerd) |
|---|---:|---:|---:|---:|---:|
| LAAG | +15,0 % | [−71,0 %, +101,1 %] | 0,64 | 0,34 | **0,29** |
| NORMAAL | +41,4 % | [−53,1 %, +135,9 %] | 1,54 | 0,86 | **0,72** |
| HOOG | −3,8 % | [−235,4 %, +227,8 %] | −0,05 | −0,03 | **−0,02** |

Geen enkele gedefleerde t haalt zelfs 1 — de verwachting uit substep 6.5
("geen enkele gedefleerde richtings-t haalt 1") **houdt**. De rij-voor-rij
val van t_gepoold → t_geclusterd → t_gedefleerd is zelf de bevinding van §5
van deze module: zes namen met een gemiddelde paneelcorrelatie
(`rho_bar` ≈ 0,71–0,77 per toestand) leveren `N_eff` rond de 1,2–1,3, dus een
gepoolde t die met een factor ~2 te groot is. Dat de gedefleerde t hier
*kleiner* is dan de gepoolde in elke cel is ook precies wat
`test_the_deflated_t_is_never_larger_than_the_pooled_t` afdwingt.

---

## 2. Bezetting én episodes per toestand per symbool

De gepoolde tabellen hierboven middelen over zes symbolen die het onderling
oneens zijn. Dat is geen ruis: het is de reden waarom §3.7 een alfaclaim op
deze as verbiedt.

| Symbool | Toestand | bezetting | episodes | gem. duur (bars) | ann. vol | ann. rendement | bp/dag |
|---|---|---:|---:|---:|---:|---:|---:|
| BTCUSDT | LAAG | 49,2 % | 30 | 18,7 | 43,1 % | +29,6 % | +8,10 |
| BTCUSDT | NORMAAL | 42,3 % | 41 | 11,8 | 52,1 % | +79,7 % | +21,84 |
| BTCUSDT | HOOG | 8,5 % | 11 | 8,8 | 58,8 % | +25,2 % | +6,90 |
| ETHUSDT | LAAG | 34,8 % | 22 | 18,0 | 52,6 % | +61,7 % | +16,91 |
| ETHUSDT | NORMAAL | 51,5 % | 38 | 15,4 | 68,9 % | +12,7 % | +3,49 |
| ETHUSDT | HOOG | 13,7 % | 17 | 9,2 | 82,7 % | +37,8 % | +10,37 |
| SOLUSDT | LAAG | 44,0 % | 34 | 14,8 | 76,5 % | +39,2 % | +10,73 |
| SOLUSDT | NORMAAL | 43,1 % | 48 | 10,2 | 93,8 % | **+106,5 %** | +29,18 |
| SOLUSDT | HOOG | 12,9 % | 13 | 11,3 | 190,7 % | **−95,5 %** | −26,18 |
| AVAXUSDT | LAAG | 36,8 % | 23 | 18,2 | 73,8 % | **−17,6 %** | −4,83 |
| AVAXUSDT | NORMAAL | 54,8 % | 36 | 17,4 | 95,9 % | +1,8 % | +0,48 |
| AVAXUSDT | HOOG | 8,4 % | 12 | 8,0 | 105,9 % | **+68,4 %** | +18,73 |
| LINKUSDT | LAAG | 33,9 % | 36 | 10,8 | 73,2 % | +17,7 % | +4,86 |
| LINKUSDT | NORMAAL | 48,2 % | 52 | 10,6 | 85,3 % | +63,0 % | +17,25 |
| LINKUSDT | HOOG | 17,9 % | 16 | 12,8 | 108,3 % | +15,3 % | +4,18 |
| DOTUSDT | LAAG | 38,2 % | 18 | 24,2 | 59,1 % | **−45,0 %** | −12,33 |
| DOTUSDT | NORMAAL | 49,0 % | 30 | 18,6 | 77,8 % | +4,4 % | +1,22 |
| DOTUSDT | HOOG | 12,8 % | 11 | 13,3 | 105,9 % | **−49,4 %** | −13,54 |

(Bron: `runs.lagged.per_symbol[*][*]`, velden `occupancy`, `n_episodes`,
`mean_duration_bars`, `ann_vol`, `ann_return`, `mean_per_bar_bp`.)

**De tekens spreken elkaar tegen.** In de toestand HOOG doet SOLUSDT
−95,5 % op jaarbasis (−26,18 bp/dag) terwijl AVAXUSDT in diezelfde toestand
+68,4 % doet (+18,73 bp/dag) — twee symbolen, tegengestelde tekens, dezelfde
toestand, hetzelfde venster. In de toestand LAAG zijn AVAXUSDT (−17,6 %) en
DOTUSDT (−45,0 %) negatief terwijl de andere vier positief zijn. Dit is
dezelfde vorm van tegenspraak die de nulmeting van revisie 1 al liet zien
(§4.5: "DOTUSDT doet +131,6 bp/dag in HOOG, SOLUSDT −45,1 bp") — het teken
wisselt per symbool en per meting, en geen van beide is stabiel genoeg om een
richting aan de toestand toe te kennen. Een gepoolde tabel had dit verborgen.

De bezetting per symbool is ook zelf niet gelijk verdeeld: LINKUSDT zit 17,9 %
van de tijd in HOOG tegen 8,4 % voor AVAXUSDT — bijna een factor 2. De
gepoolde bezetting (39,5 % / 48,1 % / 12,4 %, `runs.lagged.separation_vol[*].occupancy`)
is dus zelf al een mix van zes verschillende symbool-specifieke regimes.

---

## 3. Overgangsmatrix en gemiddelde episodelengte

`runs.lagged.transition_matrix`, gepoold over de zes symbolen, overgangen
geteld per symbool (geen overgang loopt over een symboolgrens):

| van \ naar | LAAG | NORMAAL | HOOG |
|---|---:|---:|---:|
| **LAAG** | 94,00 % | 5,96 % | 0,04 % |
| **NORMAAL** | 4,96 % | 92,67 % | 2,37 % |
| **HOOG** | 0,00 % | 9,35 % | 90,65 % |

De diagonaal ligt overal boven 90 %: een toestand wisselt niet elke bar, en
is in die zin handelbaar (Q7). LAAG en HOOG springen nooit rechtstreeks in
elkaar — de HOOG-rij heeft een exacte 0,00 % naar LAAG en de LAAG-rij een
verwaarloosbare 0,04 % naar HOOG — een toestand loopt altijd via NORMAAL.

Gemiddelde episodelengte (`runs.lagged.separation_vol[*].{n_episodes,mean_duration_bars}`,
gemiddelde van symbool-gemiddelden, dus geen `n_bars / n_episodes`):

| Toestand | episodes | gem. duur (bars) |
|---|---:|---:|
| LAAG | 163 | 17,4 |
| NORMAAL | 245 | 14,0 |
| HOOG | 80 | 10,6 |

HOOG is zowel de kortstlevende toestand (10,6 bars gemiddeld) als de toestand
met de minste episodes (80 over zes symbolen en 3,8 jaar, ruim 13 episodes
per symbool). Dat is precies de combinatie die de bredere
betrouwbaarheidsintervallen in §1.1 en §1.2 verklaart: minder episodes is een
kleinere effectieve steekproef, ongeacht hoeveel individuele bars er in de
toestand vallen.

---

## 4. Wat deze diagnose NIET vaststelt

**Geen richtingsclaim.** Geen enkele gedefleerde t op de richtingsas haalt 1
(§1.2), en de per-symbool-tabel (§2) laat zien dat de tekens elkaar
tegenspreken. Een gedefleerde t onder 1 is geen bewijs van afwezigheid van
een richtingseffect — dat zou zelf een claim zijn die deze meting niet
draagt — maar het is nog veel minder bewijs van aanwezigheid. Er is hier geen
basis om een toestand als "bullish" of "bearish" te lezen, en de huidige
poort-constructie (AD-25, REGEL V) is daar ook niet van afhankelijk: zij
schakelt exposure op basis van de toestand, niet op basis van een
richtingsaanname over die toestand.

**Geen alfaclaim (§3.7).** Een conditioneel gemiddeld rendement per toestand
is geen alfa: er is niets weggecontroleerd — geen markt, geen funding, geen
transactiekosten, geen tijdstrend. Het cijfer in §1.2 en §2 is een
beschrijving van wat er in deze toestand *gebeurde*, niet een schatting van
wat een strategie die op deze toestand handelt, zou *verdienen*.

**De identificatie loopt op VARIANTIE, niet op richting.** "HOOG" betekent in
deze module en in `regime/state.py` uitsluitend "hoge σ̂" (gerealiseerde
volatiliteit ligt daar ook aantoonbaar hoger, §1.1). Het is nooit "hoog
rendement" of "bullish", ook niet toevallig — NORMAAL heeft in §1.2 het
hoogste ann. rendement (+41,4 %) van de drie toestanden, niet HOOG.

**Geen uitspraak over verschillen TUSSEN de twee metingen in dit rapport.**
§5 vergelijkt puntschattingen tussen de gelagde en ongelagde toewijzing en
tussen het volledige venster en de ontwikkelsample. Er is geen
betrouwbaarheidsinterval berekend op die verschillen zelf — alleen op elke
losse cel — en de per-toestand-intervallen in §1.1 overlappen ruim tussen de
twee toewijzingen. §5 beschrijft daarom een richting in de puntschattingen,
niet een statistisch onderscheiden verschil (R-8).

---

## 5. Het effect van de lag

Substep 6.5 vroeg om twee dingen apart te houden: het venster (volledig
4,7753 j → de 3,8055-jaar ontwikkelsample) en de lag (stap 5's causale
verschuiving van de toestand met één bar). Beide effecten zijn hier gemeten
door de app **twee keer** te draaien op precies hetzelfde venster — eenmaal
met `lag=1` (het contract van stap 5) en eenmaal met `lag=0` (de
gelijktijdige, per-constructie-lookahead toewijzing van revisie 1) — zodat
het venstereffect en het lag-effect niet door elkaar lopen.

### 5.1 De drie punten, ann. volatiliteit per toestand

| Meting | venster | lag | LAAG | NORMAAL | HOOG | ratio HOOG/LAAG |
|---|---|---|---:|---:|---:|---:|
| §4.5-basislijn (revisie 1) | 4,7753 j, volledig | 0 (ongelagd) | 62,6 % | 79,2 % | 137,0 % | 2,19 |
| dit rapport, ongelagde reproductie | 3,8055 j, ontwikkelsample | 0 (ongelagd) | 50,4 % | 79,1 % | 141,3 % | 2,81 |
| dit rapport, GELAGD (§1.1, de echte) | 3,8055 j, ontwikkelsample | 1 (stap 5) | 63,7 % | 80,9 % | 118,4 % | 1,86 |

(Bronnen: rij 1 uit `Prompts-fases/fase_10_herstart_dagbars.md` §4.5; rijen 2
en 3 uit `runs.unlagged_revision_1.separation_vol[*].ann_vol` resp.
`runs.lagged.separation_vol[*].ann_vol` in het artefact.)

### 5.2 Venstereffect (rij 1 → rij 2, beide ongelagd)

LAAG 62,6 % → 50,4 % (−12,2 pp), NORMAAL 79,2 % → 79,1 % (−0,1 pp), HOOG
137,0 % → 141,3 % (+4,3 pp). Dit effect raakt vrijwel uitsluitend LAAG; NORMAAL
en HOOG verschuiven nauwelijks tussen het volledige venster en de
ontwikkelsample.

### 5.3 Lag-effect (rij 2 → rij 3, beide op de ontwikkelsample)

LAAG 50,4 % → 63,7 % (+13,3 pp), NORMAAL 79,1 % → 80,9 % (+1,8 pp), HOOG
141,3 % → 118,4 % (−22,9 pp). De HOOG→LAAG-verhouding valt van 2,81 naar
1,86 — de spreiding krimpt met ongeveer een derde. Dit is het effect dat
substep 6.5 expliciet vroeg om te controleren: **"Als de lag de vol-separatie
merkbaar verkleint, is dat een belangrijke bevinding"**, en dat is precies
wat hier gebeurt.

**Dit is de bevinding, en R-10 zegt dat hij gerapporteerd wordt, niet
weggewerkt.** Vergeleken met de gelijktijdige (ongelagde) toewijzing van
revisie 1, comprimeert de causaal-gelagde toewijzing van stap 5 de
vol-separatie zichtbaar: HOOG wordt bijna een kwart minder extreem (141,3 % →
118,4 %) en LAAG stijgt met een derde van het verschil naar de andere
toestanden toe (50,4 % → 63,7 %). De richting van het effect is eenduidig op
beide venstervergelijkingen (rij 2 → rij 3 op de ontwikkelsample, en
rij 1 → rij 3 netto: 62,6 % → 63,7 % LAAG maar 137,0 % → 118,4 % HOOG) en
wijst op dezelfde mechaniek: een deel van de "separatie" in de ongelagde
toewijzing van revisie 1 kwam uit gelijktijdigheid tussen de toestand van bar
`t` en het rendement van bar `t`. Zodra de toestand met één bar wordt
teruggeschoven — zodat zij uitsluitend leunt op informatie die vóór het
gemeten rendement bekend was — valt een deel van dat verschil weg.

**Wat dit niet vaststelt.** Er is één venster-paar en één lag-paar gemeten,
geen betrouwbaarheidsinterval op het VERSCHIL tussen twee metingen, en de
per-toestand-intervallen in §1.1 zijn breed genoeg (HOOG: [83,9 %, 144,9 %])
om de ongelagde 141,3 % en de gelagde 118,4 % beide te bevatten. Dit rapport
claimt daarom een richting in de puntschattingen — de lag verkleint de
gemeten vol-separatie, met name via HOOG — en niet een statistisch
onderscheiden verschil tussen de twee toewijzingen. Een formele toets op dat
verschil (bijvoorbeeld een geblokte/geclusterde toets op het paar
gelagd-minus-ongelagd per symbool-bar) is niet uitgevoerd en zou nodig zijn
voordat iemand dit als meer dan een puntschatting-observatie citeert.

**Op de richtingsas verandert de lag niets aan de kwalitatieve conclusie.**
Gedefleerde t's: ongelagd LAAG +0,34, NORMAAL +0,50, HOOG +0,27
(`runs.unlagged_revision_1.separation_return[*].t_stat_neff_deflated`);
gelagd LAAG +0,29, NORMAAL +0,72, HOOG −0,02 (§1.2). In beide metingen blijft
elke gedefleerde t ruim onder 1 — de conclusie van §1.2 en §4 staat op beide
toewijzingen.

---

## Bijlage — herleidbaarheid

`selects_nothing: true`, `trials: 0` (`phase10_state_diagnostics.json`, top).
Geen enkel veld in `conf/model/regime.yaml` of `conf/model/volatility.yaml`
is op grond van dit rapport gewijzigd; dat is de voorwaarde waaronder stap 6
in de faseopdracht nul trials kost, en zij is hier gerespecteerd.

Reproductie:
```
D:/venv/tradebot/Scripts/python.exe apps/run_state_diagnostics.py
```
