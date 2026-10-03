# Trend/carry research programme — voorregistratie (v1, 2026-10-03)

Doel van de eigenaar: *"blijf onderzoeken tot er een tevreden Sharpe en een winstgevende strategie is."*
Dit document legt vast wat "tevreden" betekent **voordat** er een nieuwe kandidaat is gedraaid, zodat
het zoeken zelf geen manier wordt om een mooi getal te fabriceren.

## 0. Uitgangspunt (wat al bekend was)

* De wekelijkse ML-meta-label-pijplijn heeft **geen edge**: OOS-AUC 0,49–0,51 (ensemble 0,490, gewogen),
  geschudde labels 0,476–0,529 (`research/inputs/repro_auc.json`). Die route is gesloten; hier wordt niet
  verder aan gedraaid.
* Literatuur-gepinde trend (TSMOM-ensemble, lookbacks 7/14/28/56/112, 6 perps, kosten 6,5 bps/eenheid
  turnover, funding): netto Sharpe 0,99 (T1, long-short) / 1,20 (T2, long-flat) over 2020-09→2026-06, maar
  **0,64 / 0,74 over 2022+** met SE ≈ 0,46–0,51 en 0,35 / 0,44 over 2024+. Een dag vertraging (lag 2) geeft
  0,37 / 0,43 over 2022+. Dat is niet "tevreden".
* Cash-and-carry (C1) levert 2022+ ≈ 4 % per jaar op het totale notioneel — gelijk aan de risicovrije
  voet; het edge is uitgehold (2025: 0,7 %, 2026: −2,9 %) en basisrisico is niet meetbaar zonder spotdata.
  C1 wordt **niet** als strategie meegenomen (alleen genoemd).
* Eerste meting in deze sessie: mijn engine reproduceert T1/T2 exact (0,99/0,64 en 1,20/0,74).

## 1. Data, splitsing en wat nooit wordt aangeraakt

* Zes perps (BTC, ETH, SOL, AVAX, LINK, DOT), dagdata, funding, open interest; `research/lib/data.py`
  laadt uitsluitend `< 2026-06-24` (het bestaande ontwerp-holdout-split).
* **Externe data is niet beschikbaar** (egress-beleid geeft 403 voor Binance/Bybit/OKX/Yahoo/CoinGecko);
  geen extra munten, geen spotdata, geen intraday.
* Drie vensters:
  * **Discovery** 2020-09-01 → 2023-12-31. Nieuwe ideeën worden hier bedacht en gescreend.
  * **Validatie** 2024-01-01 → 2026-06-23. Voor een nieuwe sleeve **eenmalig** gelezen nadat zijn ontwerp
    bevroren is. Een herontwerp na een lezing kost een trial en het venster geldt dan als *ontwikkeling*.
  * **Holdout** 2026-06-24 → 2026-08-23 (60 bars). Eén lezing, aan het eind, voor één bevroren kandidaat.
    Een rooktest, geen bewijs.
* Opmerking: de eerdere sessie heeft T1/T2 al over alle jaren gezien. TSMOM-ontwerp geldt dus niet als
  "ongezien"; alleen de nieuwe sleeves zijn dat wel.

## 2. Conventies (niet aan gedraaid)

* Besluit op close *t*, positie gehouden van close *t* → *t+1* (lag 1). Lag 2 is een stresstest.
* Kosten 6,5 bps per eenheid |Δgewicht| (taker 5,5 + halve spread 1,0); 2× als stresstest.
* Funding per dag op de gehouden positie; long betaalt positieve funding.
* Doel: portefeuillevolatiliteit 20 % (EWMA span 60 van de eigen rendementen, causaal), bruto-hefboom ≤ 4
  (mandaat). Gewichten worden geschaald, niet de rendementen, zodat de extra turnover van het schalen
  betaald wordt.
* Sharpe = dagrendementen met nul op vlakke dagen, ×√365, Lo-SE, circulaire blokbootstrap
  (`validation/inference.py`).

## 3. Wat "tevreden" betekent (de eindkandidaat moet **alles** halen)

Gemeten op 2022-01-01 → 2026-06-23 (4,48 jaar, alle zes munten live), 20 % vol-doel, 6,5 bps:

| # | Eis |
|---|---|
| A1 | netto Sharpe ≥ 1,0 |
| A2 | Sharpe ≥ 0,7 in het validatievenster (2024+) **én** ≥ 0,7 in 2022–2023 |
| A3 | positief netto rendement in minstens 4 van de 5 kalenderjaren (2022, 23, 24, 25, 26H1), geen jaar slechter dan −10 % |
| A4 | max drawdown ≤ 25 % (mandaat) |
| A5 | Sharpe ≥ 0,8 bij dubbele kosten **en** ≥ 0,6 bij lag 2 |
| A6 | ondergrens 95 %-blokbootstrap-interval van de Sharpe > 0 |
| A7 | holdout (60 bars): rendement ≥ −1 × de verwachte 60-daagse σ van de strategie |

**DSR wordt gerapporteerd, geen poort.** Met een eerlijke M ≥ 30 vraagt DSR ≥ 0,95 op 4,5 jaar een Sharpe
van ≈ 1,8; dat is met deze data niet te bewijzen en zou het antwoord altijd "nee" maken. Het getal staat er
wel, naast de eerlijke M.

## 4. Trial-boekhouding

* `research/lib/trials.py`: append-only `research/trials.jsonl`; M = 17 (vooraf: 13 trend/carry +
  4 meta-label) + het aantal verschillende varianten dat hier wordt geboekt.
* **Budget: maximaal 40 nieuwe varianten.** Gevoeligheidsruns (kosten ×2, lag 2, subperiodes, parameter-
  buurt) zijn geen varianten maar worden volledig gerapporteerd, niet geselecteerd.
* Raakt het budget op zonder dat A1–A6 gehaald zijn, dan is het antwoord **"met deze data en deze
  zes munten bestaat geen strategie die dit haalt"**, met de beste kandidaat en zijn eerlijke cijfers.

## 5. Roster v1 (aanvullingen later als gedateerde amendementen, elk een trial)

| Code | Hypothese | Ontwerp (vast) |
|---|---|---|
| TR_LS / TR_LF | trend | TSMOM-ensemble 7/14/28/56/112, long-short resp. long-flat, gelijk risico per live munt, portefeuille-vol-doel |
| TR_BRK | trend | Donchian-doorbraak 20/40/80/160, gelijk gewogen, houden tot tegengestelde doorbraak |
| TR_SM | trend, lagere turnover | idem TR_LS, signaal EMA-gesmoord (span 5) + rustband |
| XS_MOM | cross-sectioneel momentum | rang 28d-rendement, dollar-neutraal, wekelijks |
| XS_REV | korte-termijn reversal | rang 7d-rendement, omgekeerd, dollar-neutraal |
| CARRY_XS | funding-carry | short hoogste/long laagste funding, dollar-neutraal |
| FUND_EXT | extreme funding is contrarian | long/short tegen funding-z > 2 |
| OI_CONF | OI bevestigt trend | trend geschaald met OI-trend |
| RES_REV | residu-reversie | residu t.o.v. markt-PC1, z-score contrarian |
| COMBO | diversificatie | gelijk-risico mix van de overlevenden |

Selectieregel voor COMBO: een sleeve doet mee als zijn discovery-Sharpe > 0 én zijn validatie-Sharpe > 0
(één lezing); gewichten gelijk-risico (inverse realised vol), **niet** geoptimaliseerd.

## 6. Amendement 1 (2026-10-03, na de discovery-screen `stage_c_screen.py`, vóór enige validatie-lezing)

De screen (17 voorspellers, discovery 2021-10 → 2023-12, 17 trials geboekt als `SCREEN_*`) liet **niets**
op |t| ≥ 3 komen. Wat wel (zwak) richting gaf: cross-sectioneel momentum 14–112 d (IC +0,01…+0,03, t ≈ 1,8
bij 28/56/112) en funding-crowding (`fund_z`: IC −0,03, t −1,8; h=5: −0,05). Gevolgen:

* **Geschrapt** zonder verdere trial (screen toont geen of tegengesteld teken): XS_REV (IC −0,015, t −0,9),
  OI_CONF (oi_chg5 TS t −1,2), RES_REV (resid_z20 IC *positief* +0,02, dus geen reversie), FUND_EXT.
* **CARRY_XS → FUND_XS**: de screen steunt `fund_z` (afwijking van de eigen 90-d-geschiedenis), niet het
  funding-*niveau* (fund_lvl7 IC −0,015, t −0,9). Ontwerp: dollar-neutrale XS-contrarian op `fund_z`.
* **XS_MOM** wordt een ensemble over k ∈ {14, 28, 56, 112} (z-score per k, gemiddeld) i.p.v. één k.
* Het budget van 40 nieuwe varianten omvat de 17 screen-trials; er resteren 23.

## 7. Amendement 2 — bevroren ontwerp vóór de eerste validatie-lezing (2026-10-03)

Discovery-resultaten waarop dit gebaseerd is (`stage_c_*.py`, geen validatiedata): `FUND_XS` faalt de
selectieregel (discovery-Sharpe −0,12; 2021: −1,62) en valt af; `TR_BRK_*` (d6-Sharpe 0,11/0,12) en
`TR_SM_LS` (0,65, lagere Sharpe ondanks lagere turnover) vallen af. De lookbacknabijheid is **niet** gebruikt
om een lookbackset te kiezen: (5,10,20,40,80,160) geeft d6 0,30 tegen 1,00 voor (7,14,28,56,112); dat verschil
is binnen de ruis (SE ≈ 0,5) en bewijst de fragiliteit, geen reden om (7,14,28) met 1,21 te nemen.

**Bevroren kandidaat `COMBO_TX`:**

* `TREND_MIX`: score = ½·TSMOM-LS + ½·TSMOM-LF (lookbacks 7/14/28/56/112), gewicht per munt
  `score · 0,40 / σ_i / N_live`, portefeuille-vol-doel 20 % (EWMA span 60, gewichten geschaald).
* `XS_MOM`: score = gemiddelde z-score van vol-genormaliseerd rendement over k = 14/28/56/112, dollar-neutraal,
  inverse-vol, wekelijks herbalanceren (vaste fase), min. 4 live munten, portefeuille-vol-doel 20 %.
* `COMBO_TX` = ½·gewichten(TREND_MIX) + ½·gewichten(XS_MOM), daarna portefeuille-vol-doel 20 %, bruto ≤ 4.
* Selectieregel: een sleeve houdt zijn plek alleen als zijn validatie-Sharpe > 0.

Eén lezing van het validatievenster, daarna eventuele wijziging = nieuwe trial en validatie geldt als
ontwikkeling. Boekhouding: M = 43 op dit moment (17 vooraf + 17 screen + 9 sleeves/varianten); nieuwe
varianten tot nu toe 26 van 40.
