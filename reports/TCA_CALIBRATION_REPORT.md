# TCA / MARKET IMPACT CALIBRATION REPORT

**Gegenereerd door:** `apps/calibrate_impact.py`
**git_sha:** `766de6f`
**Status:** **`IMPACT_UNCALIBRATED`**

---

## 1. Wat een echte kalibratie vereist

| Dataset | Aanwezig in de gecertificeerde store? |
|---|---|
| `orderbook_l2` | **nee** |
| `trades` | **nee** |

**Ontbrekend:** `orderbook_l2`, `trades`

Geen orderboek- of trade-data in de gecertificeerde store (ontbreekt: orderbook_l2, trades). eta is niet identificeerbaar zonder eigen orders en hun gemeten prijsrespons; de gerapporteerde waarde is een CONSERVATIEVE BOVENGRENS uit de dagrange, geen schatting. Zie docs/DATA_REGISTER.md sectie 6.

---

## 2. Status

> ## `IMPACT_UNCALIBRATED`
>
> De gerapporteerde `eta` is **geen schatting**. Elk resultaat dat op deze
> parameter draait, moet deze status zichtbaar meedragen; de authoritative
> engine dwingt dat af via `ImpactEstimate.status`.

---

## 3. De afgeleide bovengrens

**Methode:** range-implied upper bound: quantile(0.95) of (high-low)/close / sigma_daily, pooled over symbol-days

Het model geeft bij `Q = V` (een meta-order zo groot als het dagvolume)
`Impact = eta * sigma_d`. De grootste beweging die op die dag daadwerkelijk
optrad is `(high - low) / close`. Wordt die VOLLEDIG aan impact toegeschreven -
niets aan nieuws, niets aan de stroom van anderen - dan volgt
`eta <= (high - low) / close / sigma_d`.

De aanname is maximaal conservatief, dus de werkelijke `eta` ligt hier met
zekerheid onder, en vermoedelijk ver eronder.

### Verdeling over 11,660 symbool-dagen

| Kwantiel | Waarde |
|---|---:|
| p50 | 1.421 |
| p75 | 1.906 |
| p90 | 2.524 |
| **p95 (gekozen bovengrens)** | **2.992** |
| p99 | 4.253 |
| max | 13.283 |

p95 en niet het maximum: het maximum over ~11k
symbool-dagen is een enkele gebeurtenis en meet die dag, niet de
liquiditeitsstructuur.

### Per instrument

| Symbool | Dagen | mediaan | p95 | max | `kappa_d` |
|---|---:|---:|---:|---:|---:|
| `AVAXUSDT` | 1,743 | 1.432 | 2.929 | 11.261 | 0.648 |
| `BTCUSDT` | 2,282 | 1.379 | 2.995 | 7.962 | 0.679 |
| `DOTUSDT` | 1,923 | 1.417 | 3.115 | 13.283 | 0.665 |
| `ETHUSDT` | 1,927 | 1.383 | 2.971 | 9.681 | 0.686 |
| `LINKUSDT` | 2,072 | 1.446 | 2.976 | 12.660 | 0.661 |
| `SOLUSDT` | 1,713 | 1.479 | 2.981 | 8.009 | 0.685 |

---

## 4. De parameters

| Parameter | Waarde |
|---|---:|
| `eta` | **2.9919** |
| `kappa_d` | **0.6720** |
| Onzekerheidsband (`eta`, p75-p99) | [1.906, 4.253] |
| Steekproef | 11,660 symbool-dagen |
| Periode | 2020-05-25 t/m 2026-08-23 |
| Instrumenten | `AVAXUSDT`, `BTCUSDT`, `DOTUSDT`, `ETHUSDT`, `LINKUSDT`, `SOLUSDT` |
| `data_hash` | `4c113cea9085af5e` |

`kappa_d` is de PERMANENTE fractie: `sigma_cc / E[(high-low)/close]`, het deel
van de typische dagbeweging dat de close haalt. Voor de kosten van EEN order
verandert `kappa_d` niets - de volledige impact wordt betaald - maar hij bepaalt
wat er van de beweging overblijft voor de volgende order.

---

## 5. Wat dit betekent voor de kosten

Kosten in basispunten bij `sigma_d` = 4,0 % (typisch voor dit universum):

| Participatie (`Q/V`) | Impact (bps) |
|---|---:|
| 0.01% | 12.0 |
| 0.10% | 37.8 |
| 1.00% | 119.7 |
| 5.00% | 267.6 |
| 10.00% | 378.5 |

Ter vergelijking: de `adv_participation_cap` in `conf/risk/default.yaml` staat op
**1 %**, en de sovereign laag dwingt die af vóór er een order bestaat.

### 5.1 Hoe informatief is deze bovengrens? Nauwelijks.

Dat moet gezegd worden, want het getal ziet er preciezer uit dan het is.

Voor een Brownse beweging is de verwachte range over een periode
`sigma * sqrt(8/pi) ~ 1,60 sigma`. De gemeten mediaan over alle instrumenten
ligt op **1.42**, en de p95 op **2.99** — precies wat een
random walk oplevert. De spreiding tussen instrumenten is bovendien
verwaarloosbaar: de p95 per symbool ligt tussen
2.93 en 3.11, terwijl hun dagvolumes
ordes van grootte verschillen.

**Deze bovengrens meet dus vrijwel uitsluitend gewone volatiliteit, en vrijwel
geen liquiditeitsstructuur.** Dat is geen fout in de afleiding maar de directe
consequentie van het ontbreken van orderboekdata: zonder eigen orders is er
geen signaal waaruit impact te scheiden valt van beweging.

### 5.2 Waarom hij desondanks bruikbaar is

Omdat hij op dit boek niet bindt. Bij de bruto-exposure die de sovereign laag
toestaat (~0,11 van de equity, gespreid over
6 symbolen) is de participatie in de orde van 1e-7 tot
1e-5, en daar levert zelfs een bovengrens van 2.99 een impact van
onder de 1 bp op een liquide dag. De dominante kostenpost blijft de taker-fee.

De bovengrens wordt pas bindend bij een aanzienlijk groter boek of een
aanzienlijk dunnere markt — en dat is precies wanneer je een conservatieve
aanname wílt hebben. Het exit-rapport draait daarom een
gevoeligheidsanalyse over `eta` in plaats van één getal te rapporteren.

---

## 6. Gecertificeerde bronnen

| Reeks | `data_hash` |
|---|---|
| `crypto/ohlcv/AVAXUSDT/1d` | `af05b4f95d0e282111641f85904c22f9` |
| `crypto/ohlcv/BTCUSDT/1d` | `4f21f2c7ab071ddc19da5eb3f38172f3` |
| `crypto/ohlcv/DOTUSDT/1d` | `17be1f4206997f12fd7e1825f123299b` |
| `crypto/ohlcv/ETHUSDT/1d` | `d67c9bb18b2c794ba416067556227a5a` |
| `crypto/ohlcv/LINKUSDT/1d` | `8d6a82f316253c2626d016bbbad804d4` |
| `crypto/ohlcv/SOLUSDT/1d` | `1d64b001993b575f34ef04ce548616c5` |

---

## 7. Wat er nodig is om dit te vervangen door een echte kalibratie

1. Orderboek-L2-snapshots op minstens 1-minuutcadans voor het universum.
2. Trade-prints met richting (taker side).
3. Eigen order-flow met grootte en tijdstempel, of een meta-order-reconstructie
   uit de trade-tape.

Punt 1 en 2 zijn de openstaande items uit `docs/DATA_REGISTER.md` §6; punt 3
komt pas na Phase 7 beschikbaar. Tot dat moment blijft de status
`IMPACT_UNCALIBRATED`, en dat is geen tekortkoming van deze fase maar een
eigenschap van de dataset.
