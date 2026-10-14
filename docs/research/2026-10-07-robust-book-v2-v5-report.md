# Robuust boek v2–v5 — breedte, kosten, bevestiging en basiscarry (2026-10-07)

Vervolg op `2026-10-07-robust-book-report.md` (v1). Dezelfde regels: elke trial in de
ledger vóór de run, elke preregistratie bevroren vóór de meting, elke lezing buiten W_DEV
geregistreerd vóór de data terugkomt. De vraag blijft: is een **netto Sharpe > 1 en een
netto CAGR > 15 %** haalbaar met een strategie die robuust genoeg is om op te vertrouwen?

## Samenvatting

- **Netto Sharpe > 1 is robuust haalbaar.** De spot-perp-basiscarry op BTC en ETH (v5-H3)
  haalt het volgende:

  | venster | netto Sharpe | CAGR | max DD |
  | --- | --- | --- | --- |
  | W_DEV | 6,8 | 7,0 % | 0,4 % |
  | backcast 2020 (nooit gemeten) | 11,5 | 12,1 % | 0,2 % |

  Op W_DEV doorstaat H3 elke robuustheidspoort behalve de CAGR: DSR 1,00 bij M = 21,
  PBO 0,002, Sharpe 6,1 bij dubbele kosten, 5,8 bij 50 mln AUM.
- **Netto CAGR > 15 % tegelijk is niet gehaald.**
  - Geen robuuste kandidaat komt in het vaste, ongehefboomde kapitaalmodel boven 9,3 % op
    W_DEV.
  - De basiscarry verdient alleen als er carry is: 2020, 2021 en 2024 wel, 2022–23 en de
    holdout 2025-07 .. 2026-09 niet. Daar nam H3 geen enkele positie.
  - Volgens de bevroren regel bindt een ongedefinieerde holdout-Sharpe. Het formele
    oordeel is daarom "gefalsificeerd". Inhoudelijk is het: geen carry, geen rendement.
- **Prijssignalen houden geen stand.** De prijsgedreven kandidaten (trend, momentum,
  cross-sectionele carry) halen buiten W_DEV geen positieve Sharpe: v2 −0,11, v3 0,00,
  v4 −1,19.
- **De grens.** 15 % CAGR met de basiscarry vraagt ruwweg het dubbele van de notional
  (portfolio margin). De vol blijft dan rond 2 %, maar het echte risico zit niet in de
  vol: beurs- en tegenpartijrisico, liquidatiecascades en de stablecoin. Dat is hefboom
  om de 15 % te halen, en die is in de opdracht uitgesloten. Die keuze ligt bij de
  eigenaar.

M is nu 21 in deze ledger, plus 52 bekende eerdere trials.

## 1. v2 — breedte: het survivorship-vrije Binance-universum

**Probleem uit v1.** Zes Bybit-perps, overlevers van nu, met een onderlinge correlatie van
0,73: ongeveer 1,3 onafhankelijke weddenschappen. Op zo'n universum is geen cross-sectioneel
signaal te meten.

**Data** (`data/binance_vision.py`):
- de volledige dag-historie van elke Binance USDT-M-perp die ooit handelde, ook gedeliste
  munten (LUNA, FTT, SRM, ...): 900 symbolen, 2020-01 tot en met 2026-09;
- elke zip gecontroleerd tegen zijn S3-MD5;
- point-in-time-universum: de top-50 naar 30-daagse ADV, met ≥ 90 bars historie,
  ≥ 5 mln USDT ADV en ≥ 15 % jaarvol (tegen gekoppelde munten);
- korte datagaten overbrugd, delistings gesloten tegen de laatste koers.

**Bevinding over kosten.** De vooraf gekozen Abdi-Ranaldo-spreadschatter (CHL) meet op
dagbars volatiliteit, geen spread: 17,7 bp voor BTCUSDT, terwijl de werkelijke spread
onder 1 bp ligt. v2 rekende daardoor met te hoge kosten. v3 corrigeert dat (post-hoc, en
zo geboekt).

**W_DEV (2021-01 .. 2025-06), netto:**

| trial | Sharpe | CAGR | vol | MDD | omzet/jr | bruto+funding-Sharpe |
| --- | --- | --- | --- | --- | --- | --- |
| X1 cross-sectioneel momentum | −0,14 | −4,7 % | 19,8 % | 39,9 % | 55× | 0,60 |
| X2 funding-carry (long/short) | 0,31 | 4,4 % | 21,7 % | 44,2 % | 68× | 1,13 |
| X3 trend long/short | 0,44 | 5,7 % | 15,5 % | 18,0 % | 23× | 0,72 |
| X4 trend long-flat | **0,77** | 8,2 % | 11,1 % | 15,5 % | 11× | 0,94 |
| X5 combinatie | 0,33 | 4,8 % | 19,9 % | 43,7 % | 94× | 1,49 |
| R1 BTC buy-and-hold | 0,58 | 18,2 % | 60,7 % | 78,9 % | — | — |

De score koos X4. Die bindt op G1, G2, G3 (validate 0,16), G7, G8 en G9.
**Holdout X4: Sharpe −0,11**, gearchiveerd.

## 2. v3 — realistische kosten en partiële aanpassing

Twee wijzigingen, en verder niets:
- een vaste halve spread van 3 bp, met 6 bp als stress;
- Gârleanu-Pedersen partiële aanpassing (κ = 0,2), zodat de omzet bij het signaalverval past.

| trial | Sharpe | CAGR | MDD | 95 %-CI | train / validate |
| --- | --- | --- | --- | --- | --- |
| Y2 carry | 0,83 | 14,8 % | 29,3 % | [−0,09; 1,76] | 1,08 / 0,31 |
| Y4 trend long-flat | 0,86 | 9,6 % | 14,5 % | [−0,04; 1,80] | 1,15 / 0,24 |
| Y5 carry + trend | **1,13** | **20,7 %** | 24,4 % | [0,19; 2,07] | 1,50 / 0,33 |

Y5 haalde de doelcijfers in-sample, maar faalde op G3 (validate 0,33), G5 (kosten), G8
(DSR 0,72 bij M = 17) en G9 (PBO). De vooraf vastgelegde score koos Y4.
**Holdout Y4: Sharpe 0,00, CAGR −0,3 %.**

## 3. v4 — één geregistreerde bevestiging van Y5

Y5 was de enige kandidaat met de doelcijfers op W_DEV. Hem toch op de holdout zetten is
een extra trial. Die is als zodanig geboekt (M = 18), met de criteria vóór de lezing
bevroren.

| | Sharpe | CAGR | MDD |
| --- | --- | --- | --- |
| holdout Y5 (2025-07 .. 2026-09) | **−1,19** (SE 1,01) | **−17,6 %** | **27,0 %** |
| waarvan carry | −1,17 | −18,5 % | 25,2 % |
| waarvan trend | 0,00 | −0,3 % | 10,5 % |

**Gefalsificeerd:** z = −2,29 tegen W_DEV, en een drawdown boven het mandaat.

**De les.** Het carryboek ontving 22,9 %/jaar aan funding en verloor meer op de prijs.
Long lage en short hoge funding is een prijsweddenschap: de short-kant zit in munten die
met hefboom worden gekocht, en die bleven stijgen. Dat leidde naar v5.

## 4. v5 — funding oogsten zonder prijsweddenschap

Ontwerp: `docs/superpowers/specs/2026-10-07-robust-book-v5-basis-design.md`. Per munt staat
even veel notional long op spot als short op de perp. Het rendement is dan de funding,
min de basisverandering en de kosten van twee benen.

**Kapitaalmodel**, zonder portfolio margin:
- 70 % spot en 30 % USDT-marge;
- een margevloer, en liquidatie bij cross-margin;
- delisting van één been sluit beide benen.

Alle parameters lagen vooraf vast. Het spotbeen (481 perps met een spotpaar) is nieuw
gedownload en tegen de MD5 gecontroleerd.

### 4.1 W_DEV (2021-01 .. 2025-06), netto

| trial | Sharpe | CAGR | vol | MDD | train / validate | score |
| --- | --- | --- | --- | --- | --- | --- |
| H1 basis top-50 | 5,66 | 8,6 % | 1,5 % | 1,4 % | 6,75 / 3,36 | 90,1 |
| H2 50 % basis + 50 % trend | 1,57 | 9,3 % | 5,8 % | 6,7 % | 1,98 / 0,66 | 85,9 |
| **H3 basis BTC + ETH** | **6,76** | 7,0 % | 1,0 % | 0,4 % | 6,63 / 9,56 | **98,6** |

**Per jaar, H3:** 2021 +25,9 %, 2022 −0,1 %, 2023 +0,3 %, 2024 +6,9 %, 2025H1 +0,5 %.

**Poorten:**
- H3 bindt alleen op G2 (CAGR):
  - DSR 1,00 bij M = 21 (0,70 bij M = 73) en PBO 0,002;
  - Sharpe bij dubbele kosten 6,11, bij 50 mln AUM 5,76;
  - elke parameterverstoring blijft boven de helft van de basis-Sharpe;
  - nul liquidaties.
- H1 bindt ook op G5: bij dubbele kosten blijft 0,744 van de Sharpe over, tegen een drempel
  van 0,75. Bij 50 mln AUM zakt de Sharpe naar 0,97 (impact in altcoin-spot).
- H1 en H2 binden op G8. De DSR rekent met de spreiding tussen de drie trials, en H1's
  hoge Sharpe trekt de nul-Sharpe omhoog. Dat is de bevroren regel van v2 en die blijft
  staan.

De vooraf vastgelegde score koos H3.

### 4.2 De twee lezingen buiten W_DEV

| H3 | Sharpe | CAGR | MDD | z t.o.v. W_DEV | belegd |
| --- | --- | --- | --- | --- | --- |
| **backcast 2020-04 .. 2020-12** (nooit gemeten) | **11,46** (SE 1,69) | 12,1 % | 0,16 % | **+2,79** | 100 % van de dagen |
| holdout 2025-07 .. 2026-09 (besmet) | ongedefinieerd | 0,0 % | 0,0 % | — | 0 % van de dagen |

**De backcast** is het enige schone out-of-samplebewijs in dit hele programma, en hij
bevestigt de edge ruim: z = +2,79 boven W_DEV.

**De holdout** laat de grens zien. De 30-daagse funding op BTC en ETH kwam in 15 maanden
nooit boven de instapdrempel van 15 %/jaar, dus H3 hield niets aan.

**Formeel oordeel: "falsified_out_of_sample".** De bevroren poortcode telt een niet-eindige
metriek als bindend. Dat oordeel laat ik staan. Inhoudelijk is de holdout geen
weerlegging van de edge, maar wel van de CAGR: 0 % over vijftien maanden.

**Procesincident, zichtbaar in de geschiedenis.**
1. De eerste berekening crashte na de registratie van beide lezingen, omdat de samenvatting
   van een venster zonder positie nul variantie gaf. De slotstanden zijn direct
   gecommit (`37a9a40`).
2. De backcast-uitkomst was op dat moment nergens getoond of opgeslagen.
3. Het afmaken liep via `validation/holdout.py::resume_registered_read`: één keer, met de
   reden in het slot, zonder tweede registratie. De regel voor een venster zonder positie
   lag vast vóór de backcast zichtbaar werd (`af6b9f0`).
4. Kandidaat, parameters en poorten zijn niet veranderd.

## 5. Oordeel over de doelstelling

**Het maximaal realistisch haalbare resultaat, met bewijs dat ik robuust acht:**

| | netto Sharpe | netto CAGR | max DD |
| --- | --- | --- | --- |
| BTC/ETH-basiscarry, W_DEV | 6,8 | 7,0 % | 0,4 % |
| idem, ongemeten backcast 2020 | 11,5 | 12,1 % | 0,2 % |
| idem, holdout 2025–26 | n.v.t. (vlak) | 0,0 % | 0,0 % |
| beste combinatie (H2), W_DEV | 1,6 | 9,3 % | 6,7 % |

**Waarom 15 % er niet in zit zonder hefboom:**

1. **De beperking is hoe vaak er carry is, niet hoeveel.**
   - Als H3 belegd is, ontvangt hij 27 %/jaar funding op de notional, bij gemiddeld 61 %
     notional per been.
   - Maar hij is maar 43 % van de W_DEV-dagen belegd: 91 % in 2021, 9 % in 2022, 2 % in
     2023, 73 % in 2024 en 34 % in 2025H1. In de holdout was dat 0 %.
   - De funding die longs betalen, komt in golven, en tussen de golven staat het boek in
     cash.
2. **Breedte helpt maar beperkt.** De top-10-funding is hoger (52 % in 2021), maar
   altcoin-spot is dun: impact, en een spreiding die bij 50 mln AUM de Sharpe tot 0,97
   drukt. H1 haalt 8,6 %.
3. **Prijsrendement erbij kost robuustheid.** Trend erbij (H2) geeft 9,3 %, maar de
   validate-Sharpe zakt naar 0,66. Elke prijsgedreven sleeve in v2–v4 stortte buiten
   W_DEV in.
4. **15 % met de basiscarry vraagt ruwweg het dubbele van de notional.** Dat kan met
   portfolio margin. Het is een ruwe schaling, geen toets. De vol blijft rond 2 %, maar
   het risico zit in wat de vol niet ziet:
   - de beurs als tegenpartij (FTX 2022);
   - auto-deleveraging en liquidatiecascades bij een wick;
   - het stablecoin-onderpand.

   Dat is precies "hefboom om de 15 % te forceren", en die is uitgesloten.

**Wat een netto CAGR boven 15 % wel zou kunnen dragen, als vervolgonderzoek** (elk een
nieuwe preregistratie, en geen van alle in deze data te toetsen):
- de basiscarry op meerdere beurzen tegelijk (Bybit, OKX, Deribit): meer munten met
  carry, en minder concentratie in één tegenpartij;
- rente op het onbelegde kapitaal (tokenized T-bills, ~4–5 %): geen alfa, maar het vult
  de vlakke jaren;
- een expliciet en door de eigenaar gekozen hefboomkader voor de basiscarry, met
  stress-tests op beursfalen.

## 6. Resterende zwakke punten

- **Daggranulariteit.** De intraday-basis, het precieze tijdstip van de
  funding-afrekening rond de close en de executie tussen twee wallets zijn niet gemodelleerd.
- **Kosten.** Spotspreads zijn een vaste aanname (5 bp, stress 10 bp), niet gemeten uit
  orderboeken.
- **Validatie in de engine.** De authoritative engine (`backtest/engine.py`) kan geen
  tweebeensboek en geen delistings simuleren. v2–v5 zijn screeningsresultaten
  (`NOT_ADMISSIBLE`).
- **Besmette holdout.** De holdout 2025–26 is voor elk programma na v2 besmet. Het enige
  schone bewijs dat nog te halen valt, ligt vooruit in de tijd: papertrading van H3
  vanaf 2026-10, op echte fills.

