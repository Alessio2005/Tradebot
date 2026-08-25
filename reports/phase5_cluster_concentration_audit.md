# PHASE 5 — CLUSTER & CONCENTRATION AUDIT

> **Status:** meting. Elk getal in dit rapport is gereproduceerd op de
> gecertificeerde PIT-store via de echte `RiskEngine`, niet geschat.

**Gegenereerd:** 2026-08-25
**git_sha:** `d0131a6`
**Meetvenster:** 2021-11-15 → 2026-08-23, **1.743 bars** (daily, na EWMA-burn-in en 30-bars ADV-venster)
**Risk `config_hash`:** `47821e47fe2cec30`
**Databron:** `docs/DATA_REGISTER.md` — 6 gecertificeerde `ohlcv/1d`-reeksen
**Fase-opdracht:** §7, §19

---

## 1. Samenvatting

De fase-opdracht stelt dat de clusterindeling de 8 %-volatiliteitstarget
blokkeert en het boek op ~3,2 % houdt. **Die stelling is juist en is hier
gereproduceerd:** 3,26 % gerealiseerde boekvolatiliteit, 40,7 % van de target.

De diagnose gaat echter één stap verder dan de opdracht veronderstelt. Het
probleem is niet dat `LINKUSDT` in het verkeerde vakje zit. Het probleem is dat
**dit universum geen clusterstructuur hééft**, en dat een clusterlimiet op een
structuurloos universum geen diversificatiebeperking is maar een willekeurige
belasting op bruto-exposure — die het boek bovendien *concentreert* in plaats
van spreidt.

| Bevinding | Waarde |
|---|---|
| Gerealiseerde boekvolatiliteit | **3,26 %** |
| `sigma_target` | 8,00 % |
| Attainment | **40,7 %** |
| Oorzaak | `cluster_cap`, bindt op **100 %** van de bars |
| Aandeel `LINKUSDT` bij 1/N zonder clusterlimiet | 16,7 % |
| Aandeel `LINKUSDT` **mét** clusterlimiet | **40,0 %** |
| Statistische scheiding van de labels | +0,0113, 95 %-BI **[−0,0131, +0,0380]** |

---

## 2. §7.1 — Audit van de labels

### 2.1 Bron en methodologie

`conf/risk/default.yaml` zet zes labels: vijf `crypto_l1`, één `crypto_oracle`.
Het commentaar bij het blok noemt de indeling *"monolithisch crypto-perp"* en
verantwoordt haar als *"expliciet vastgelegd in plaats van impliciet gelaten"*.

Er is **geen clustering-methodologie**. De labels zijn een taxonomie van wat het
onderliggende project dóét (een L1-blockchain, een oracle-netwerk), niet een
meting van hoe het instrument beweegt. Voor een risicolimiet is alleen het
tweede relevant.

### 2.2 De labels scheiden niets

Gemeten correlatie over het volledige venster:

| | Gemiddelde ρ | Paren |
|---|---:|---:|
| **Binnen** cluster | 0,7389 | 10 |
| **Tussen** clusters | 0,7277 | 5 |
| **Scheiding** | **+0,0113** | |

Bootstrap (2.000 resamples): 95 %-betrouwbaarheidsinterval
**[−0,0131, +0,0380]**. Het interval bevat nul. De labels dragen geen
aantoonbare informatie over co-beweging.

Ter vergelijking: alle zes mogelijke 5/1-partities, gerangschikt op scheiding:

| Geïsoleerd symbool | Scheiding |
|---|---:|
| `SOLUSDT` | +0,0360 |
| **`LINKUSDT`** (geconfigureerd) | **+0,0113** |
| `DOTUSDT` | +0,0006 |
| `BTCUSDT` | −0,0010 |
| `AVAXUSDT` | −0,0041 |
| `ETHUSDT` | −0,0427 |

De geconfigureerde keuze is niet eens de beste 5/1-splitsing, en de spreiding
over alle zes is kleiner dan het bootstrap-interval van één ervan.

### 2.3 De labels zijn niet stabiel

Scheiding per kalenderjaar:

| Jaar | n | Binnen | Tussen | Scheiding |
|---|---:|---:|---:|---:|
| 2021 | 76 | 0,626 | 0,691 | **−0,0657** |
| 2022 | 365 | 0,810 | 0,773 | +0,0369 |
| 2023 | 365 | 0,656 | 0,606 | +0,0499 |
| 2024 | 366 | 0,704 | 0,644 | +0,0592 |
| 2025 | 365 | 0,788 | 0,830 | **−0,0422** |
| 2026 | 235 | 0,800 | 0,859 | **−0,0588** |

**Het teken wisselt.** In de twee meest recente jaren is `crypto_l1` sterker
gecorreleerd met `LINKUSDT` dan met zichzelf. Een limiet die op deze indeling
steunt, beschermt in 2025 en 2026 tegen het tegenovergestelde van wat hij
beweert te beperken.

### 2.4 Het universum heeft geen stabiele clusterstructuur

Hiërarchische clustering (correlatie-afstand, average linkage) op het volledige
venster geeft bij k=2:

```
{AVAXUSDT, LINKUSDT, DOTUSDT}  vs  {BTCUSDT, ETHUSDT, SOLUSDT}
separation = +0,0447   (4x de geconfigureerde)
```

Dat lijkt een betere indeling — mid-caps versus majors — en over het volledige
venster is de scheiding stabiel positief per jaar (+0,017 tot +0,081).

Maar dezelfde clustering, **per jaar opnieuw uitgevoerd**, herstelt die indeling
niet:

| Jaar | Data-gedreven k=2-partitie | Groottes |
|---|---|---|
| 2021 | `{BTC,ETH,SOL,LINK,DOT}` / `{AVAX}` | 5/1 |
| 2022 | `{BTC,ETH,AVAX,LINK,DOT}` / `{SOL}` | 5/1 |
| 2023 | `{BTC,ETH,SOL,AVAX,DOT}` / `{LINK}` | 5/1 |
| 2024 | `{BTC,ETH,SOL,AVAX}` / `{LINK,DOT}` | 4/2 |
| 2025 | `{ETH,AVAX,LINK,DOT}` / `{BTC,SOL}` | 4/2 |
| 2026 | `{BTC,ETH,SOL,AVAX,LINK}` / `{DOT}` | 5/1 |

**Zes jaren, zes verschillende partities.** Het geïsoleerde symbool wisselt elk
jaar. Dit is wat ruis eruitziet, niet wat structuur eruitziet.

**Conclusie van §7.1:** zes crypto-perpetuals met een gemiddelde paarsgewijze
correlatie van 0,735 vormen **één** economisch cluster. Elke tweedeling ervan is
een fit op ruis. De majors/alts-indeling is dus geen correctie maar een tweede
gok met een gunstiger in-sample getal.

---

## 3. §7.2 — Feasibility

### 3.1 Kan het universum de 8 %-target halen?

**Ja, ruimschoots — de constraint-set maakt de target niet infeasible.**

Random search over de long-only simplex binnen de volledige sovereign
constraint-set (400.000 trekkingen):

| Labelling | Max haalbare σ_p | Target gehaald? |
|---|---:|---|
| Huidige labels (5/1) | **42,1 %** | ja |
| Geen clusterlimiet | 51,0 % | ja |
| Eén cluster per symbool | 51,0 % | ja |
| Gebalanceerd 3/3 | 50,8 % | ja |

De assets zijn zó volatiel (geannualiseerde EWMA-σ: BTC 54 %, ETH 70 %, SOL
92 %, AVAX 93 %, LINK 94 %, DOT 86 %) dat 8 % boekvolatiliteit een *kleine*
positie vereist, geen grote. Er bestaat geen `RISK_TARGET_INFEASIBLE`-situatie.

> **Dit is een andere vraag dan de volgende.** "Bestaat er een toegestane
> portefeuille met 8 % vol?" is ja. "Levert de sovereign pipeline, toegepast op
> een 1/N-alpha, 8 % vol?" is nee. Beide moeten gemeten worden; alleen de eerste
> is een feasibility-vraag.

### 3.2 Wat de pipeline feitelijk oplevert

`RiskEngine.decide()` bar voor bar op een 1/N long-only boek (`a_t = 1` voor elk
symbool), met echte `sigma_hat` uit `volatility/ewma.py` en een causale
30-bars ADV uit de gecertificeerde `turnover`-kolom:

```
sigma_target                 : 0,0800
gerealiseerde boekvolatiliteit: 0,0326      <- 40,7 % van target
gemiddelde gross exposure     : 0,0470
gemiddeld gewicht per symbool : BTC/ETH/SOL/AVAX/DOT 0,0056  ·  LINK 0,0188
```

Bindende limieten over 1.743 bars:

| Limiet | Bindt op | Frequentie |
|---|---:|---|
| `vol_target` | 1.743 bars | **100 %** |
| `cluster_cap` | 8.715 registraties (5 symbolen × 1.743 bars) | **100 %** |
| alle overige | 0 | 0 % |

### 3.3 Attributie

Eén limiet tegelijk geneutraliseerd, opnieuw gemeten over dezelfde 1.743 bars:

| Configuratie | σ_p | Gross | % van target |
|---|---:|---:|---:|
| **Baseline** | **0,0326** | 0,0470 | **40,7 %** |
| `max_cluster_concentration` 0,60 → 1,00 | 0,0750 | 0,1128 | **93,8 %** |
| Labels: één cluster per symbool | 0,0750 | 0,1128 | 93,8 % |
| Labels: gebalanceerd 3/3 | 0,0750 | 0,1128 | 93,8 % |
| `max_concentration` 0,40 → 1,00 | 0,0326 | 0,0470 | 40,7 % |
| `max_position_pct` 0,25 → 1,00 | 0,0326 | 0,0470 | 40,7 % |

**De clusterlimiet is als enige verantwoordelijk.** Concentratie- en
per-asset-limieten dragen exact nul bij. Elke manier om de clusterlimiet te
ontkrachten — de cap verhogen, of labels kiezen waarop hij niet bindt — levert
hetzelfde getal: 7,50 %.

### 3.4 De resterende 6,2 %

Van 8,00 % naar 7,50 % is geen limiet maar de **comonotone bovengrens** in
`risk/vol_targeting.py::book_sigma_hat`, die de boekvolatiliteit schat als
`Σ|a_i|·σ_i`, dus alsof alle correlaties 1 zijn:

```
equal-weight boek, werkelijke sigma      : 0,7148
equal-weight boek, comonotone bovengrens : 0,7670
bovengrens / waarheid                    : 1,073x conservatief
=> vol_target alleen landt op             : 0,0746  (93,2 % van target)
```

Bij een gemiddelde correlatie van 0,735 kost die aanname **6,8 %** van de
target. Dat is een bewuste, gedocumenteerde conservatisme en geen defect: de
bovengrens is per constructie nooit te laag, en dat is precies wat een
risicolimiet moet garanderen. Zij wordt hier gekwantificeerd, niet weggenomen.

**Volledige decompositie van de 8 %:**

```
8,00 %  sigma_target
 −0,54 pp  comonotone bovengrens (bewust conservatief, 1,073x)
= 7,46 %  wat vol_target alleen zou opleveren
 −4,20 pp  cluster_cap op een structuurloos universum   <-- het defect
= 3,26 %  gerealiseerd
```

---

## 4. De perverse uitkomst

Dit is de kern van de bevinding, en de reden dat "accepteer 3,2 % als
conservatief" geen geldige uitweg is.

Bij een 1/N-boek en de huidige labels moet `crypto_l1` (vijf symbolen) onder
60 % van de gross blijven. De limietlaag schaalt dat cluster terug tot het
vaste punt:

| | Zonder clusterlimiet | Mét clusterlimiet |
|---|---:|---:|
| Aandeel `LINKUSDT` | 16,7 % | **40,0 %** |
| Aandeel per L1-symbool | 16,7 % | 12,0 % |
| Behouden gross | 100 % | **41,7 %** |

`LINKUSDT` gaat van 16,7 % naar **40,0 %** van het boek — exact de
`max_concentration`-drempel, die daarmee van niet-bindend naar precies-bindend
gaat.

> **Een diversificatiebeperking produceert het meest geconcentreerde boek dat de
> overige limieten toestaan, en vernietigt daarbij 58 % van de bruto-exposure.**

Dat is geen conservatieve uitkomst. Het is een limiet die het tegenovergestelde
doet van wat hij belooft, op grond van een indeling die de data niet steunt.

---

## 5. §7.3 — Aanbeveling, met rationale en impactanalyse

### 5.1 Wat NIET wordt gedaan

**Herlabelen naar majors/alts (3/3).** In-sample geeft dat een 4× betere
scheiding en een target-attainment van 93,8 %. Het wordt toch afgewezen: §2.4
laat zien dat de jaarlijkse clustering die indeling niet herstelt. Kiezen voor
de partitie met het beste volledige-venstergetal, terwijl elk deelvenster een
andere partitie aanwijst, is precies de vorm van in-sample selectie die dit
platform elders (DSR, SPA, pre-registratie) systematisch afstraft. Een
risicolimiet mag daar niet van uitgezonderd worden.

**`sigma_target` verlagen naar 3,2 %.** Dat zou het symptoom wegdefiniëren en de
gemeten oorzaak intact laten.

### 5.2 Wat WEL wordt gedaan

**De labels corrigeren naar wat gemeten is: één cluster.**

```yaml
clusters:
  BTCUSDT:  crypto_perp
  ETHUSDT:  crypto_perp
  SOLUSDT:  crypto_perp
  AVAXUSDT: crypto_perp
  LINKUSDT: crypto_perp
  DOTUSDT:  crypto_perp
```

**Rationale.** De gemeten correlatiestructuur wijst één groep aan. Dat vastleggen
is geen versoepeling maar een correctie: de limiet stopt met het beperken van
een factor die niet bestaat.

**Gevolg, expliciet en zichtbaar.** Met één cluster geldt
`effective_relative_cap(1, 0.60) = max(0.60, 1/1) = 1.0`. De clusterlimiet
**bindt nooit meer**. Dat is een limiet die vacuous is, en dat moet luid gezegd
worden in plaats van als succes gepresenteerd:

> Op dit universum is `max_cluster_concentration` **vacuous**. Er is één
> economisch cluster; er valt niets tussen clusters te spreiden. De limiet blijft
> in `constraint_order` en wordt actief zodra het universum een tweede
> aantoonbaar cluster bevat — een aandelen-, FX- of commodity-sleeve. Tot dat
> moment is de enige werkzame spreidingsbescherming `max_concentration` (0,40)
> op symboolniveau.

Dit is exact het gedrag dat Phase 4 §3.3 al voor scenario S2 (correlatie-
instorting) beschreef. Het verschil is dat het nu de **normale** toestand blijkt
te zijn, niet de crisistoestand.

### 5.3 Impactanalyse

| Grootheid | Vóór | Na | Δ |
|---|---:|---:|---|
| Gerealiseerde boekvolatiliteit | 3,26 % | 7,50 % | +4,24 pp |
| Target-attainment | 40,7 % | 93,8 % | +53,1 pp |
| Gemiddelde gross exposure | 0,047 | 0,113 | 2,40× |
| Aandeel `LINKUSDT` | 40,0 % | 16,7 % | **−23,3 pp** |
| Max symboolconcentratie | 40,0 % | 16,7 % | −23,3 pp |
| Bindende limieten | `vol_target`, `cluster_cap` | `vol_target` | − |
| `config_hash` | `47821e47fe2cec30` | *wijzigt* | nieuwe registry-entry |

Het boek wordt **groter én minder geconcentreerd**. Dat is ongebruikelijk voor
een limietwijziging en het is precies de diagnose: de oude limiet ruilde
exposure in voor concentratie.

### 5.4 Vereiste regressietests

Vast te leggen vóór de wijziging, niet erna:

1. `cluster_cap` bindt nul keer op een 1/N-boek onder de nieuwe labels.
2. `cluster_cap` bindt **wél** zodra een tweede cluster wordt geïnjecteerd — de
   limiet is vacuous op dít universum, niet kapot.
3. Target-attainment ≥ 90 % op het gecertificeerde venster.
4. `LINKUSDT`-aandeel ≤ `max_concentration` en ≤ 1/n + tolerantie.
5. `config_hash` wijzigt en de nieuwe waarde staat in `risk_config_registry.json`.
6. `hypothesis_ledger.total_n_hypotheses()` verandert **niet** door de nieuwe
   registratie.
7. Een universum zonder label voor een symbool → harde failure (§17).

---

## 6. §19 — Feasibility gate

| Rapportagepunt | Waarde |
|---|---|
| Target volatility | 8,00 % |
| Achieved volatility (vóór) | 3,26 % |
| Achieved volatility (na) | 7,50 % |
| **Maximum feasible volatility** | **42,1 %** (huidige labels) / 51,0 % (zonder clusterlimiet) |
| Concentration binding constraint | `max_concentration` 0,40 — bindt na de clusterlimiet, niet ervoor |
| Cluster binding constraint | `max_cluster_concentration` 0,60 — bindt op 100 % van de bars |
| Symbol binding constraint | geen |
| Reason for infeasibility | **n.v.t. — de target is haalbaar** |

**`RISK_TARGET_INFEASIBLE` wordt NIET gerapporteerd.** De target is met ruime
marge haalbaar; de shortfall komt van een limiet, niet van het universum. Het
onderscheid is bindend voor exit-criterium 17: de fase mag niet als "infeasible"
afsluiten wanneer de oorzaak een corrigeerbaar labelingsdefect is.

---

## 7. Reproductie

Alle getallen in dit rapport komen uit twee read-only probes tegen de
gecertificeerde store en de echte `RiskEngine`. Zij worden in Phase 5 vastgelegd
als `tests/integration/test_cluster_feasibility.py`, zodat elk getal hierboven
een testassertie wordt en niet alleen een rapportregel.
