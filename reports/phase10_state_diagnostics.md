# FASE 10 — STAP 6: DE TOESTANDSDIAGNOSE

> **Status:** diagnostiek, expliciet niet-promoveerbaar. Er wordt in deze
> stap niets geselecteerd — geen kwantiel, geen `low_q`/`high_q`, geen
> parameter is op grond van onderstaande getallen gewijzigd — dus kost zij
> **nul trials** (`selects_nothing: true`, `trials: 0` in het artefact).

**Gegenereerd:** 2026-09-11; meting opnieuw gedraaid 2026-09-12 (fixronde 1,
bevinding M-3) — byte-identiek op het `git_sha`-veld na, zie de bijlage
**git_sha (meting):** `8d62d5c`
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

| Symbool | Toestand | bezetting | episodes | gem. duur (bars) | ann. vol | ann. rendement | bp/dag | 95 %-BI (rendement) | t (gecl.) | t (gedefl.) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BTCUSDT | LAAG | 49,2 % | 30 | 18,7 | 43,1 % | +29,6 % | +8,10 | [−38,7 %, +97,8 %] | 0,85 | 0,85 |
| BTCUSDT | NORMAAL | 42,3 % | 41 | 11,8 | 52,1 % | +79,7 % | +21,84 | [−9,2 %, +168,6 %] | 1,76 | 1,76 |
| BTCUSDT | HOOG | 8,5 % | 11 | 8,8 | 58,8 % | +25,2 % | +6,90 | [−199,6 %, +249,9 %] | 0,22 | 0,22 |
| ETHUSDT | LAAG | 34,8 % | 22 | 18,0 | 52,6 % | +61,7 % | +16,91 | [−37,2 %, +160,7 %] | 1,22 | 1,22 |
| ETHUSDT | NORMAAL | 51,5 % | 38 | 15,4 | 68,9 % | +12,7 % | +3,49 | [−93,8 %, +119,3 %] | 0,23 | 0,23 |
| ETHUSDT | HOOG | 13,7 % | 17 | 9,2 | 82,7 % | +37,8 % | +10,37 | [−211,0 %, +286,7 %] | 0,30 | 0,30 |
| SOLUSDT | LAAG | 44,0 % | 34 | 14,8 | 76,5 % | +39,2 % | +10,73 | [−88,8 %, +167,1 %] | 0,60 | 0,60 |
| SOLUSDT | NORMAAL | 43,1 % | 48 | 10,2 | 93,8 % | **+106,5 %** | +29,18 | [−52,2 %, +265,2 %] | **1,32** | **1,32** |
| SOLUSDT | HOOG | 12,9 % | 13 | 11,3 | 190,7 % | **−95,5 %** | −26,18 | [−686,6 %, +495,5 %] | **−0,32** | **−0,32** |
| AVAXUSDT | LAAG | 36,8 % | 23 | 18,2 | 73,8 % | **−17,6 %** | −4,83 | [−152,9 %, +117,6 %] | **−0,26** | **−0,26** |
| AVAXUSDT | NORMAAL | 54,8 % | 36 | 17,4 | 95,9 % | +1,8 % | +0,48 | [−142,0 %, +145,5 %] | 0,02 | 0,02 |
| AVAXUSDT | HOOG | 8,4 % | 12 | 8,0 | 105,9 % | **+68,4 %** | +18,73 | [−338,3 %, +475,1 %] | **0,33** | **0,33** |
| LINKUSDT | LAAG | 33,9 % | 36 | 10,8 | 73,2 % | +17,7 % | +4,86 | [−121,8 %, +157,3 %] | 0,25 | 0,25 |
| LINKUSDT | NORMAAL | 48,2 % | 52 | 10,6 | 85,3 % | +63,0 % | +17,25 | [−73,5 %, +199,4 %] | 0,90 | 0,90 |
| LINKUSDT | HOOG | 17,9 % | 16 | 12,8 | 108,3 % | +15,3 % | +4,18 | [−269,2 %, +299,8 %] | 0,11 | 0,11 |
| DOTUSDT | LAAG | 38,2 % | 18 | 24,2 | 59,1 % | **−45,0 %** | −12,33 | [−151,2 %, +61,3 %] | **−0,83** | **−0,83** |
| DOTUSDT | NORMAAL | 49,0 % | 30 | 18,6 | 77,8 % | +4,4 % | +1,22 | [−118,8 %, +127,7 %] | 0,07 | 0,07 |
| DOTUSDT | HOOG | 12,8 % | 11 | 13,3 | 105,9 % | **−49,4 %** | −13,54 | [−378,7 %, +279,9 %] | **−0,29** | **−0,29** |

(Bron: `runs.lagged.per_symbol[*][*]`, velden `occupancy`, `n_episodes`,
`mean_duration_bars`, `ann_vol`, `ann_return`, `mean_per_bar_bp`, `ci_low`,
`ci_high`, `t_stat_clustered`, `t_stat_neff_deflated`.)

**Waarom de twee t-kolommen identiek zijn, en waarom ze er allebei staan.** Een
per-symbool-cel is een paneel van ÉÉN kolom. `clustered_mean` geeft daar per
constructie `rho_bar = 0` en `neff_factor = 1,0` terug: er is geen
cross-sectionele breedte om voor te defleren, dus de gepoolde, de geclusterde
en de gedefleerde lezing vallen samen. Dat is geen dubbeling maar de eerlijke
mededeling dat op dit niveau alleen de clustering op datum nog werk doet — en
de kolommen staan er in dezelfde vorm als §1.2, zodat geen enkel getekend
rendement in dit rapport zonder zijn t op de bladzijde staat (R-8).

**Geen van de achttien cellen sluit nul uit.** De grootste |t| in de hele
tabel is 1,76 (BTCUSDT, NORMAAL) en elk van de achttien 95 %-intervallen
bevat nul. Wat hieronder "tegenspraak" heet, is dus een tegenspraak tussen
puntschattingen die stuk voor stuk niet van nul te onderscheiden zijn — en
precies dat maakt het een argument tegen een richtingslezing, niet een meting
van een richting.

**De tekens spreken elkaar tegen — en geen van beide tekens is meetbaar.** In
de toestand HOOG doet SOLUSDT −95,5 % op jaarbasis (−26,18 bp/dag,
t = −0,32, BI [−686,6 %, +495,5 %]) terwijl AVAXUSDT in diezelfde toestand
+68,4 % doet (+18,73 bp/dag, t = +0,33, BI [−338,3 %, +475,1 %]) — twee
symbolen, tegengestelde tekens, dezelfde toestand, hetzelfde venster, en twee
t's die allebei onder een derde blijven. In de toestand LAAG zijn AVAXUSDT
(−17,6 %, t = −0,26) en DOTUSDT (−45,0 %, t = −0,83) negatief terwijl de andere
vier positief zijn; ook daar haalt geen enkele t de 1. Dit is
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
| **LAAG** | 94,00 % | 5,96 % | 0,037 % |
| **NORMAAL** | 4,96 % | 92,67 % | 2,37 % |
| **HOOG** | 0,00 % | 9,35 % | 90,65 % |

De diagonaal ligt overal boven 90 %: een toestand wisselt niet elke bar, en
is in die zin handelbaar (Q7). Een sprong tussen de twee uitersten is
uitzonderlijk maar niet onmogelijk, en dat verschil is het noemen waard: HOOG
gaat op dit venster **nooit** rechtstreeks naar LAAG (exact 0,00 %, nul
waarnemingen), terwijl LAAG **precies één keer** rechtstreeks naar HOOG gaat
(0,037 %, één overgang op ~2.700). Vrijwel elke overgang loopt dus via NORMAAL,
maar "nooit" geldt maar voor één van de twee richtingen.

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
§5 vergelijkt puntschattingen tussen de gelagde en de gelijktijdige toewijzing.
Er is geen betrouwbaarheidsinterval berekend op dat verschil zelf — alleen op
elke losse cel — en de per-toestand-intervallen in §1.1 overlappen ruim tussen
de twee toewijzingen. §5 beschrijft daarom een richting in de puntschattingen,
niet een statistisch onderscheiden verschil (R-8).

**En al helemaal geen uitspraak over het VENSTER.** §5.2 legt de
§4.5-basislijn ernaast, maar die rij is met een ANDERE toewijzer gemeten
(`regime/buckets.py::classify_vol_buckets`) en niet met `assign_by_variance`.
Het verschil tussen die rij en dit rapport is de som van een venstereffect en
een regeleffect, en met twee rijen zijn die twee niet te scheiden. Dit rapport
meet dus het effect van de lag, niet dat van het venster (fixronde 1,
bevinding I-1).

---

## 5. Het effect van de lag

Substep 6.5 vroeg om twee effecten apart te houden: het venster (volledig
4,7753 j → de 3,8055-jaar ontwikkelsample) en de lag (stap 5's causale
verschuiving van de toestand met één bar). Eén van die twee is hier schoon te
meten en de andere niet. Dat onderscheid staat vooraan, want het bepaalt wat de
rest van deze paragraaf mag beweren.

**Het lag-effect is schoon gemeten.** De app is **twee keer** gedraaid:
dezelfde code, dezelfde config, dezelfde toewijzer, hetzelfde venster,
dezelfde 1.389 rendement-bars — en precies één verschil, `lag=1` tegen
`lag=0`. Eén variabele beweegt, dus het verschil tussen die twee rijen ís het
lag-effect. Dat is §5.3.

**Het venstereffect is NIET schoon te meten, en werd in de eerste versie van
dit rapport ten onrechte als zodanig gepresenteerd** (fixronde 1, bevinding
I-1). De §4.5-basislijn is niet met `assign_by_variance` gemeten maar met
`regime/buckets.py::classify_vol_buckets` — vaste ±0,5σ-banden op de log van
de causale EWMA-vol, plus een tweede as op de ATR-verhouding. Dat is een
andere schatter met een andere bezetting: §4.5 zit 16,5 / 78,7 / 4,8 % in
LAAG / NORMAAL / HOOG, dit rapport 39,5 / 48,1 / 12,4 %. LAAG scheelt een
factor 2,4 en NORMAAL een factor 1,6; de drie toestanden bevatten domweg
andere bars. Het verschil tussen die rij en dit rapport is daarmee de som van
een venstereffect én een regeleffect, en uit twee rijen zijn twee effecten
niet te scheiden. §5.2 rapporteert dat verschil daarom als het GEZAMENLIJKE
effect van allebei, en nergens als het venster.

### 5.1 De drie punten, ann. volatiliteit per toestand

| Meting | toewijzer | venster | lag | bezetting L/N/H | LAAG | NORMAAL | HOOG | ratio HOOG/LAAG |
|---|---|---|---|---|---:|---:|---:|---:|
| §4.5-basislijn (revisie 1) | `classify_vol_buckets` | 4,7753 j, volledig | 0 | 16,5 / 78,7 / 4,8 % | 62,6 % | 79,2 % | 137,0 % | 2,19 |
| dit rapport, `lag=0` (gelijktijdig) | `assign_by_variance` | 3,8055 j, ontwikkelsample | 0 | 39,5 / 48,2 / 12,4 % | 50,4 % | 79,1 % | 141,3 % | 2,81 |
| dit rapport, GELAGD (§1.1, de echte) | `assign_by_variance` | 3,8055 j, ontwikkelsample | 1 (stap 5) | 39,5 / 48,1 / 12,4 % | 63,7 % | 80,9 % | 118,4 % | 1,86 |

(Bronnen: rij 1 uit `Prompts-fases/fase_10_herstart_dagbars.md` §4.5, kolommen
"aandeel" en "ann. volatiliteit"; rijen 2 en 3 uit
`runs.unlagged_revision_1.separation_vol[*].{occupancy,ann_vol}` resp.
`runs.lagged.separation_vol[*].{occupancy,ann_vol}` in het artefact.)

De kolom **toewijzer** is de kolom die deze tabel eerlijk houdt. Rij 2 is geen
reproductie van rij 1 en wordt hier ook nergens zo genoemd: zij deelt met rij 1
alleen de gelijktijdigheid, niet de regel en niet het venster. Rij 2 en rij 3
verschillen daarentegen in precies één ding.

De artefactsleutel van rij 2 heet `runs.unlagged_revision_1`. Die naam dateert
van vóór fixronde 1 en is niet hernoemd, omdat hernoemen het artefact en elke
verwijzing ernaar zou wijzigen zonder één getal te raken. Lees hem als "de
`lag=0`-variant van dit rapport", nooit als "de meting van revisie 1"; wat rij 2
met revisie 1 deelt, staat hierboven en is alleen de gelijktijdigheid.

### 5.2 Rij 1 → rij 2: venster ÉN toewijzingsregel, niet te scheiden

LAAG 62,6 % → 50,4 % (−12,2 pp), NORMAAL 79,2 % → 79,1 % (−0,1 pp), HOOG
137,0 % → 141,3 % (+4,3 pp). Beide rijen zijn ongelagd, dus de lag zit hier
niet in — maar het venster wél samen met de toewijzer, en die twee zijn uit
deze twee rijen niet uit elkaar te halen. Het is dus een gezamenlijk effect,
en het is met name niet te zeggen dat "het venster LAAG raakt": de toewijzer
die rij 1 produceerde stopt 16,5 % van de bars in LAAG en die van rij 2 39,5 %,
dus LAAG bestaat in de twee rijen uit andere bars. Dit blok staat hier omdat
de faseopdracht de §4.5-basislijn ernaast wil zien, niet omdat er een
venstereffect uit te lezen valt. Wie dat wél wil meten, moet
`classify_vol_buckets` op de ontwikkelsample draaien; dat is geen onderdeel
van stap 6 en is niet gedaan.

### 5.3 Lag-effect (rij 2 → rij 3): de bevinding van deze stap

LAAG 50,4 % → 63,7 % (+13,3 pp), NORMAAL 79,1 % → 80,9 % (+1,8 pp), HOOG
141,3 % → 118,4 % (−22,9 pp). De HOOG→LAAG-verhouding valt van 2,81 naar
1,86 — de spreiding krimpt met ongeveer een derde. Dit is het effect dat
substep 6.5 expliciet vroeg om te controleren: **"Als de lag de vol-separatie
merkbaar verkleint, is dat een belangrijke bevinding"**, en dat is precies
wat hier gebeurt.

**Dit is de bevinding, en R-10 zegt dat hij gerapporteerd wordt, niet
weggewerkt.** De causaal-gelagde toewijzing van stap 5 comprimeert de
vol-separatie zichtbaar ten opzichte van de gelijktijdige toewijzing van
DEZELFDE regel op HETZELFDE venster: HOOG wordt bijna een kwart minder extreem
(141,3 % → 118,4 %) en LAAG stijgt 13,3 pp naar de andere toestanden toe
(50,4 % → 63,7 %). Dit is de enige vergelijking in dit rapport waarin precies
één ding verschilt — geen toewijzer, geen venster, geen config, geen andere
bars, alleen de lag — en dat is de reden dat zij als bevinding telt terwijl
§5.2 dat niet doet. De mechaniek is eenduidig: een deel van de "separatie" in
een gelijktijdige toewijzing komt uit de gelijktijdigheid zelf, want de
toestand van bar `t` is daar een functie van σ̂ op `t`, en σ̂ op `t` bevat
`r_t`. Zodra de toestand met één bar wordt teruggeschoven — zodat zij
uitsluitend leunt op informatie die vóór het gemeten rendement bekend was —
valt een deel van dat verschil weg.

**Wat hier bewust NIET als steunbewijs wordt gebruikt.** Het netto verschil
rij 1 → rij 3 (62,6 % → 63,7 % op LAAG, 137,0 % → 118,4 % op HOOG) wijst
dezelfde kant op, maar het is de som van drie verschillen — toewijzer, venster
én lag. Het voegt aan de vergelijking hierboven niets toe en wordt hier dus
niet als bevestiging opgevoerd. De bevinding rust op rij 2 → rij 3 en heeft
geen steun van elders nodig.

**Wat dit niet vaststelt.** Er is één lag-paar gemeten — en géén venster-paar,
zie §5.2 — er is geen betrouwbaarheidsinterval berekend op het VERSCHIL tussen
de twee metingen, en de per-toestand-intervallen in §1.1 zijn breed genoeg
(HOOG: [83,9 %, 144,9 %]) om de gelijktijdige 141,3 % en de gelagde 118,4 %
beide te bevatten. Dit rapport
claimt daarom een richting in de puntschattingen — de lag verkleint de
gemeten vol-separatie, met name via HOOG — en niet een statistisch
onderscheiden verschil tussen de twee toewijzingen. Een formele toets op dat
verschil (bijvoorbeeld een geblokte/geclusterde toets op het paar
gelagd-minus-ongelagd per symbool-bar) is niet uitgevoerd en zou nodig zijn
voordat iemand dit als meer dan een puntschatting-observatie citeert.

**Op de richtingsas verandert de lag niets aan de kwalitatieve conclusie.**
Gedefleerde t's: gelijktijdig LAAG +0,34, NORMAAL +0,50, HOOG +0,27
(`runs.unlagged_revision_1.separation_return[*].t_stat_neff_deflated`);
gelagd LAAG +0,29, NORMAAL +0,72, HOOG −0,02 (§1.2). In beide metingen blijft
elke gedefleerde t ruim onder 1 — de conclusie van §1.2 en §4 staat op beide
toewijzingen.

---

## 6. Wat de bevinding van §5.3 betekent voor stap 7 en stap 13

Substep 6.5 zegt dat een afwijking *"vóór alle volgende stappen gaat"*. Stap 7
is al gecommit (`5679090`, `1de85eb`) en schakelt exposure op deze toestand,
dus die zin verdient een antwoord in plaats van een stilte. Het antwoord
hieronder is **controllerbesluit P41**; het wordt hier vastgelegd, niet in dit
rapport genomen.

**De mechaniek van stap 7 is niet geraakt.** De claim van stap 7 gaat over
SAMENSTELLING: een toestandspoort verandert wélke namen er in het boek zitten,
en een cross-sectioneel uniforme vermenigvuldiger kan dat per constructie niet,
omdat L7 hem er weer uitdeelt (AD-16, AD-25). Die claim is structureel. Zij
hangt niet af van hoe ver de vol-niveaus van de toestanden uit elkaar liggen,
en een compressie van 2,81× naar 1,86× verandert er dus niets aan.

**Wat de compressie wél verandert, is de prior op de economische vraag van
stap 13.** Een toestand waarvan de vol-separatie 1,86× is in plaats van 2,81×
draagt minder voor een poort om uit te halen. De verwachte OMVANG van een
eventueel poortvoordeel is daarmee kleiner dan revisie 1 suggereerde. Dat is
een verwachting en geen meting (R-10): stap 13 is de stap die hem toetst, en
daar hoort deze zin dan ook thuis.

**Stap 6 beoordeelt stap 7 niet.** Deze stap is diagnostiek, selecteert niets,
en er is op grond van deze getallen geen bestand buiten stap 6 gewijzigd.

---

## Bijlage — herleidbaarheid

`selects_nothing: true`, `trials: 0` (`phase10_state_diagnostics.json`, top).
Geen enkel veld in `conf/model/regime.yaml` of `conf/model/volatility.yaml`
is op grond van dit rapport gewijzigd; dat is de voorwaarde waaronder stap 6
in de faseopdracht nul trials kost, en zij is hier gerespecteerd.

**De `git_sha` van het artefact, en wat hij wel en niet kan zeggen** (fixronde
1, bevinding M-3). De eerste versie van het artefact droeg `git_sha: 5679090`.
Dat is de HEAD van het moment waarop de meting draaide, en op die commit
bestond `src/tradebot/regime/state_diagnostics.py` nog niet — de module was
toen nog niet gecommit. Het veld noemde dus een boom waarin de
metende code ontbrak. De meting is daarom in fixronde 1 opnieuw gedraaid op de
gecommitte boom; het nieuwe artefact is **byte-identiek aan het oude, op die
ene regel na**:

```
$ diff artefact_zoals_gecommit.json artefacts/governance/phase10_state_diagnostics.json
2c2
<   "git_sha": "5679090",
---
>   "git_sha": "8d62d5c",
```

Geen enkel gerapporteerd getal in dit document is daardoor veranderd. `8d62d5c`
bevat `state_diagnostics.py` en `run_state_diagnostics.py` wél, dus de
herkomstclaim klopt nu. Wat een `git_sha` principieel niet kan zijn, is de
commit waarin het artefact zélf terechtkomt: `current_git_sha()` leest HEAD op
het moment van schrijven, en die commit bestaat dan nog niet. De sha noemt dus
de boom waartegen is gemeten, nooit de boom waarin het resultaat landt.

Reproductie:
```
D:/venv/tradebot/Scripts/python.exe apps/run_state_diagnostics.py
```
