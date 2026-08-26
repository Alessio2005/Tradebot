# PHASE 6 — ECONOMETRISCHE DIAGNOSE VAN DE VOL-PIJPLIJN

> **Deliverable 9** · Phase 6, stappen 3 en 4 · §8.1, §8.2
> **git_sha:** `f87bac1`
> **Universum:** 6 gecertificeerde reeksen · 1743 bars · 2021-11-15 t/m 2026-08-23

Elke toets hieronder is gedraaid VOORDAT er één model is gefit. Dat is de volgorde die §3 voorschrijft en zij is niet omkeerbaar: een ARCH-toets die je draait nadat een GARCH-fit is mislukt, is geen poort meer maar een verklaring achteraf.

## 0. Waarop is getoetst, en waarom dat drie reeksen zijn

Een toets op de verkeerde reeks levert een keurige p-waarde die niets betekent. Per symbool zijn drie transformaties getoetst:

| Reeks | Wat je verwacht | Waarvoor de uitkomst telt |
|---|---|---|
| `log_price` | eenheidswortel | controle op de data; **niet** het ARCH-poortoordeel |
| `log_return` | stationair, ARCH-effecten aanwezig | **hier geldt het ARCH-poortoordeel** |
| `fracdiff(d*)` | stationair mét behouden geheugen | de reeks die een feature-pijplijn zou gebruiken |

De ARCH-toets op `log_price` staat in de tabellen maar telt niet als poort. Op een niet-stationaire reeks autocorreleren de gekwadrateerde residuen vanzelf, dus de toets verwerpt daar bijna altijd — een artefact van de niet-stationariteit, geen conditionele heteroskedasticiteit.

## 1. Het ARCH-poortoordeel per reeks

Stap 3: *"De ARCH-test is de poortwachter: is er geen aantoonbare conditionele heteroskedasticiteit, dan is een GARCH-structuur niet gerechtvaardigd en stopt het spoor daar met een gedocumenteerd oordeel."*

| Symbool | Engle ARCH LM | p-waarde | Poort | Gevolg voor H1 |
|---|---:|---:|---|---|
| `BTCUSDT` | 81.48 | 2.15e-12 | **OPEN** | GARCH-familie mag worden gefit |
| `ETHUSDT` | 84.58 | 5.47e-13 | **OPEN** | GARCH-familie mag worden gefit |
| `SOLUSDT` | 268.84 | 1.59e-50 | **OPEN** | GARCH-familie mag worden gefit |
| `AVAXUSDT` | 96.07 | 3.26e-15 | **OPEN** | GARCH-familie mag worden gefit |
| `LINKUSDT` | 103.16 | 1.33e-16 | **OPEN** | GARCH-familie mag worden gefit |
| `DOTUSDT` | 49.48 | 1.72e-06 | **OPEN** | GARCH-familie mag worden gefit |

**6 van 6 reeksen** passeren de ARCH-poort.

- `BTCUSDT` — OPEN — Engle ARCH verwerpt op alpha = 0.05 (p = 2.15e-12); een GARCH-structuur is op BTCUSDT:log_return gerechtvaardigd.
- `ETHUSDT` — OPEN — Engle ARCH verwerpt op alpha = 0.05 (p = 5.47e-13); een GARCH-structuur is op ETHUSDT:log_return gerechtvaardigd.
- `SOLUSDT` — OPEN — Engle ARCH verwerpt op alpha = 0.05 (p = 1.59e-50); een GARCH-structuur is op SOLUSDT:log_return gerechtvaardigd.
- `AVAXUSDT` — OPEN — Engle ARCH verwerpt op alpha = 0.05 (p = 3.26e-15); een GARCH-structuur is op AVAXUSDT:log_return gerechtvaardigd.
- `LINKUSDT` — OPEN — Engle ARCH verwerpt op alpha = 0.05 (p = 1.33e-16); een GARCH-structuur is op LINKUSDT:log_return gerechtvaardigd.
- `DOTUSDT` — OPEN — Engle ARCH verwerpt op alpha = 0.05 (p = 1.72e-06); een GARCH-structuur is op DOTUSDT:log_return gerechtvaardigd.

## 2. Stationariteit: ADF en KPSS samen

ADF en KPSS hebben TEGENGESTELDE nulhypotheses. Dat is geen redundantie maar de reden dat zij samen informatiever zijn dan elk apart:

| ADF | KPSS | Betekenis |
|---|---|---|
| verwerpt | verwerpt niet | stationair — eenduidig |
| verwerpt niet | verwerpt | eenheidswortel — eenduidig |
| verwerpt | verwerpt | tegenstrijdig; lange geheugen of structurele breuk |
| verwerpt niet | verwerpt niet | onbeslist — de data kan het niet zeggen |

Een pijplijn die alleen ADF draait, leest de laatste twee gevallen als "prima" en gaat door.

### 2.1 `log_price`

| Symbool | ADF stat | ADF p | KPSS stat | KPSS p | Oordeel |
|---|---:|---:|---:|---:|---|
| `BTCUSDT` | -0.881 | 0.794 | 4.7627 | 0.01 * | eenheidswortel (ADF verwerpt niet, KPSS verwerpt) |
| `ETHUSDT` | -2.604 | 0.0923 | 1.1620 | 0.01 * | eenheidswortel (ADF verwerpt niet, KPSS verwerpt) |
| `SOLUSDT` | -1.558 | 0.505 | 2.5058 | 0.01 * | eenheidswortel (ADF verwerpt niet, KPSS verwerpt) |
| `AVAXUSDT` | -1.910 | 0.328 | 1.6507 | 0.01 * | eenheidswortel (ADF verwerpt niet, KPSS verwerpt) |
| `LINKUSDT` | -2.913 | 0.0439 | 1.0606 | 0.01 * | tegenstrijdig (beide verwerpen) — meestal lange-geheugengedrag of een structurele breuk binnen het venster |
| `DOTUSDT` | -1.663 | 0.45 | 4.3573 | 0.01 * | eenheidswortel (ADF verwerpt niet, KPSS verwerpt) |

### 2.2 `log_return`

| Symbool | ADF stat | ADF p | KPSS stat | KPSS p | Oordeel |
|---|---:|---:|---:|---:|---|
| `BTCUSDT` | -42.676 | 0 | 0.3337 | 0.1 * | stationair (ADF verwerpt, KPSS verwerpt niet) |
| `ETHUSDT` | -42.526 | 0 | 0.1413 | 0.1 * | stationair (ADF verwerpt, KPSS verwerpt niet) |
| `SOLUSDT` | -44.132 | 0 | 0.3264 | 0.1 * | stationair (ADF verwerpt, KPSS verwerpt niet) |
| `AVAXUSDT` | -41.660 | 0 | 0.1251 | 0.1 * | stationair (ADF verwerpt, KPSS verwerpt niet) |
| `LINKUSDT` | -42.534 | 0 | 0.1942 | 0.1 * | stationair (ADF verwerpt, KPSS verwerpt niet) |
| `DOTUSDT` | -12.171 | 1.42e-22 | 0.1660 | 0.1 * | stationair (ADF verwerpt, KPSS verwerpt niet) |

### 2.3 `fracdiff`

| Symbool | ADF stat | ADF p | KPSS stat | KPSS p | Oordeel |
|---|---:|---:|---:|---:|---|
| `BTCUSDT` | -1.876 | 0.344 | 1.7924 | 0.01 * | eenheidswortel (ADF verwerpt niet, KPSS verwerpt) |
| `ETHUSDT` | -2.812 | 0.0566 | 0.8859 | 0.01 * | eenheidswortel (ADF verwerpt niet, KPSS verwerpt) |
| `SOLUSDT` | -2.862 | 0.05 | 1.1731 | 0.01 * | tegenstrijdig (beide verwerpen) — meestal lange-geheugengedrag of een structurele breuk binnen het venster |
| `AVAXUSDT` | -2.932 | 0.0417 | 1.0506 | 0.01 * | tegenstrijdig (beide verwerpen) — meestal lange-geheugengedrag of een structurele breuk binnen het venster |
| `LINKUSDT` | -3.000 | 0.0349 | 0.9105 | 0.01 * | tegenstrijdig (beide verwerpen) — meestal lange-geheugengedrag of een structurele breuk binnen het venster |
| `DOTUSDT` | -3.389 | 0.0113 | 2.8715 | 0.01 * | tegenstrijdig (beide verwerpen) — meestal lange-geheugengedrag of een structurele breuk binnen het venster |

`*` = de KPSS-p-waarde is door `statsmodels` afgekapt op de rand van zijn tabel. Een `p = 0,01` betekent daar `<= 0,01` en een `p = 0,10` betekent `>= 0,10`.

## 3. Autocorrelatie en structurele breuken

| Symbool | Ljung-Box p (returns) | autocorrelatie? | CUSUM stat | kritiek | breuk? |
|---|---:|---|---:|---:|---|
| `BTCUSDT` | 0.319 | nee | 1.297 | 0.948 | ja |
| `ETHUSDT` | 0.953 | nee | 0.959 | 0.948 | ja |
| `SOLUSDT` | 0.332 | nee | 1.358 | 0.948 | ja |
| `AVAXUSDT` | 0.607 | nee | 0.760 | 0.948 | nee |
| `LINKUSDT` | 0.572 | nee | 0.844 | 0.948 | nee |
| `DOTUSDT` | 0.0835 | nee | 0.807 | 0.948 | nee |

**3 van 6 reeksen** vertonen een structurele breuk in het gemiddelde over dit venster. Dat is relevant voor elk model in deze fase: een model dat over één venster wordt gefit en beoordeeld terwijl er een breuk in zit, levert een OOS-prestatie op die een gemiddelde is over twee verschillende regimes en die dus over geen van beide iets zegt.

## 4. Fractionele differentiëring — de gekozen `d` per reeks

Zoekbereik uit `conf/model/fracdiff.yaml`: `d ∈ [0.05, 0.95]`, gewichtsdrempel `0.0001`, ADF-doel `p < 0.05`.

Het punt van FFD is de afruil tussen stationariteit en geheugen. Het CRITERIUM van §8.1 is `d*` zelf — lager is meer geheugen, want de FFD-gewichten dalen monotoon in `d`. De kolommen **geheugen** en **bij d = 1** zijn AFML's correlatiediagnostiek; §4.2 laat zien dat die maat op dit venster niet monotoon is en soms negatief, en dat zij hier dus niet als criterium kan dienen. Zij staat in de tabel omdat haar instabiliteit zelf een resultaat is.

| Symbool | `d*` | ADF p | geheugen | bij `d = 1` | winst | venster | bars weg | bars over |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `BTCUSDT` | 0.3650 | 0.0426 | 0.1026 | 0.0253 | +0.0772 | 316 | 315 | 1428 |
| `ETHUSDT` | 0.1875 | 0.0498 | -0.2385 | 0.0412 | -0.2797 | 507 | 506 | 1237 |
| `SOLUSDT` | 0.3137 | 0.05 | -0.3959 | 0.0229 | -0.4188 | 372 | 371 | 1372 |
| `AVAXUSDT` | 0.2554 | 0.05 | -0.4280 | 0.0279 | -0.4559 | 439 | 438 | 1305 |
| `LINKUSDT` | 0.1668 | 0.05 | -0.5049 | 0.0360 | -0.5408 | 521 | 520 | 1223 |
| `DOTUSDT` | 0.0500 | 0.0154 | 0.3558 | 0.0108 | +0.3450 | 362 | 361 | 1382 |

`d` staat NIET als constante in code. Hij is per reeks gezocht binnen de bovenstaande grenzen en wordt hier per reeks geregistreerd; de zoekruimte komt uit `conf/model/fracdiff.yaml`.

### 4.1 Wat FFD op DEZE reekslengte kost

De kolommen **venster** en **bars weg** zijn de reden dat de gewichtsdrempel in `conf/` op `1e-4` staat en niet op de AFML-waarde `1e-5`. Gemeten op `d ≈ 0,45`:

| drempel | venster | truncatiemassa | bars over van 1.743 |
|---:|---:|---:|---:|
| `1e-2` | 11 | 0,1216 | 1.733 |
| `1e-3` | 49 | 0,0571 | 1.695 |
| `1e-4` | 238 | 0,0271 | **1.506** |
| `1e-5` | 1.163 | 0,0131 | 581 |

AFML gebruikt `1e-5`, maar dat is geschreven voor reeksen van tienduizenden bars. Op 1.743 bars kost `1e-5` er 1.162 — twee derde van de steekproef — en de resterende 581 bars beslaan alleen het laatste deel van het venster. Elke toets op die reeks meet dan een ander tijdvak dan de rest van deze fase.

**Defect in mijn eigen werk, gevonden tijdens deze stap.** De eerste versie van deze tabel had een kolom *warm-up* die consequent `0 bars` rapporteerde. `frac_diff_ffd` zet de eerste rijen niet op NaN maar LAAT ZE VALLEN, en mijn meting telde NaN's. Daardoor zag een transformatie die twee derde van de steekproef opat, eruit alsof zij gratis was. De correctie is een lengteverschil in plaats van een NaN-telling, en zij is de directe aanleiding voor de drempelkeuze hierboven.

### 4.2 De geheugenmaat is op dit venster NIET monotoon in `d`

AFML §5.5 meet behouden geheugen als `corr(FFD_d(x), x)` en toont een curve die netjes daalt van 1 naar 0 als `d` van 0 naar 1 loopt. Op DEZE reeks doet zij dat niet. Gemeten op hetzelfde raster:

| Symbool | `d=0.05` | `d=0.10` | `d=0.20` | `d=0.30` | `d=0.50` | `d=0.70` | `d=0.90` |
|---|---|---|---|---|---|---|---|
| `BTCUSDT` | +0.440 | +0.073 | -0.092 | +0.007 | +0.169 | +0.144 | +0.044 |
| `ETHUSDT` | -0.111 | -0.190 | -0.213 | -0.232 | -0.153 | -0.049 | -0.046 |
| `SOLUSDT` | -0.045 | -0.473 | -0.551 | -0.419 | -0.035 | +0.081 | +0.024 |
| `AVAXUSDT` | -0.201 | -0.450 | -0.415 | -0.325 | -0.146 | +0.004 | -0.021 |
| `LINKUSDT` | -0.113 | -0.466 | -0.482 | -0.282 | -0.099 | -0.028 | -0.017 |
| `DOTUSDT` | +0.356 | +0.188 | +0.142 | +0.126 | +0.104 | +0.132 | +0.044 |

De maat springt van teken en is niet monotoon. De reden is geen rekenfout maar de vorm van dit venster: AFML's curve wordt gemeten op reeksen waarin het NIVEAU door één sterke trend wordt gedomineerd, en 2021-11 t/m 2026-08 is dat niet — piek, instorting, herstel. De FFD-reeks is in essentie een lang gewogen gemiddelde van verleden returns, en dat heeft op een niet-monotoon prijspad geen systematisch verband met het niveau.

**Gevolg voor de interpretatie.** De kolom *geheugen* in tabel 4 is DIAGNOSTIEK en geen criterium. Het criterium van §8.1 is `d` zelf: de minimale `d` die ADF-stationariteit haalt, IS per constructie het maximale behoud van geheugen, want de FFD-gewichten dalen monotoon in `d`. Wie de correlatiekolom als criterium zou gebruiken, zou op deze data een willekeurige `d` kiezen.

### 4.3 De ADF/KPSS-tegenspraak op de FFD-reeks

`min_frac_diff` zoekt de kleinste `d` die ADF onder `p = 0,05` brengt en optimaliseert dus UITSLUITEND tegen ADF. Tabel 2.3 laat zien wat dat oplevert: de ADF-p ligt per constructie net onder 0,05, terwijl KPSS op elke reeks verwerpt. De twee toetsen zijn het oneens, en dat is geen bug maar precies wat het paar hoort te laten zien.

De interpretatie: de minimale `d` in ADF-zin is voor KPSS ONDERGEDIFFERENTIEERD. Wie een reeks nodig heeft die BEIDE toetsen doorstaat, moet een hogere `d` accepteren en dus geheugen inleveren. Deze fase kiest dat niet, omdat §8.1 expliciet de minimale `d` bij ADF-stationariteit voorschrijft — maar het feit hoort in het rapport en niet in een voetnoot, want een feature-pijplijn die deze reeks gebruikt, gebruikt een reeks die KPSS niet-stationair noemt.

## 5. Wat deze diagnose NIET zegt

- Zij zegt niet dat een model dat door de ARCH-poort komt, ook zal winnen. De poort is een noodzakelijke voorwaarde, geen aanwijzing.
- Zij zegt niets over de RV-proxy. QLIKE vergelijkt een forecast met een proxy, en de gecertificeerde store bevat geen intraday-data; die beperking staat in `reports/GARCH_VS_EWMA_COMPETITION.md` en in de Data Adequacy Gate, niet hier.
- Zij zegt niets over de economische waarde van een model. Elke statistische winst moet nog door de authoritative engine.

