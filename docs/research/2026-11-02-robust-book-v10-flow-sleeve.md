# Robuust boek v10: een richtings-sleeve naast de carry (2026-11-02)

De vraag van de eigenaar: het boek mag niet alleen van funding-carry afhangen, en
"instituties gaan netto long", dus speel daar eventueel op in.

v10 laat de carry van U1 (v9) ongemoeid en zet een deel van het geld dat anders in het
geldmarktfonds staat long in BTC- en ETH-spot. Spot zonder hefboom kan op Kraken EU en telt
daar als onderpand. De live dienst (`carry/`) is niet aangeraakt.

## Samenvatting

- **Alle vier de trials zijn verworpen** op de vooraf vastgelegde criteria. Het
  hoofdkandidaat F1 (flow + trend) haalt de CAGR-winst niet, en zijn slechtste jaar ligt
  meer dan 5 pp onder dat van U1.
- **De institutionele flow voegt niets toe boven trend.** De netto long van asset managers
  op de CME (CFTC Traders in Financial Futures) maakt de sleeve slechter, niet beter:
  - F1 (flow + trend) haalt minder dan F3 (alleen trend);
  - F4 (alleen flow) verliest in de holdout ten opzichte van U1.

  "Instituties gaan netto long" is als verhaal waar, maar de wekelijkse CME-positionering
  is als timingsignaal voor BTC/ETH niet bruikbaar.
- **Alleen trend (F3) is het interessantst**, maar ook verworpen:

  | | U1 | F3 trend | F1 flow + trend |
  | --- | --- | --- | --- |
  | CAGR, hele cyclus | 8,0 % | **12,7 %** | 8,9 % |
  | boven de T-bill | 4,9 % | 9,4 % | 5,7 % |
  | Sharpe boven T-bill, maand | 1,27 | 1,24 | 1,00 |
  | max drawdown | 0,3 % | 7,0 % | 8,4 % |
  | holdout 2025-07 .. 2026-09, boven T-bill | −0,8 % | +4,1 % | +1,5 % |
  | slechtste jaar | +1,3 % (2022) | **−5,1 % (2022)** | −3,9 % (2022) |

  F3 voegt bijna 5 pp per jaar toe bij dezelfde Sharpe, maar 2022 wordt een verliesjaar.
  Dat breekt "elk jaar winst", en het criterium "hoogstens 5 pp slechter dan U1 in het
  slechtste jaar" (6,4 pp).
- **Eerlijk gezegd:** de prijsfamilie heeft alle historische vensters al gelezen (X4, Y4,
  Y5 in v2–v4). Ook een positieve uitkomst hier had niets bevestigd. Wat F3 waard is, laat
  alleen het vooruit-sample zien.

## 1. De regels

Vastgelegd in `conf/model/robust_book_v10.yaml` en gecommit vóór de spot- en CFTC-reeksen in
de repo stonden. Niets is op die data getuned. Per dag, met besluit op de close en rendement
over de volgende dag:

- **flow:** (long − short) / open interest van asset managers op het standaard
  CME-contract, hoger dan 4 rapporten eerder. Het rapport van dinsdag telt vanaf zondag
  (verschijnt vrijdag); ouder dan 21 dagen telt het niet.
- **trend:** de close boven zijn gemiddelde van 100 dagen.
- **grootte:** vol-doel per munt τ / σ / 2, σ uit een EWMA van 30 dagen. De rij wordt
  teruggeschaald tot het plafond van de trial en tot het geld dat de carry die dag niet
  vastlegt (min 5 % werkvoorraad). De carry heeft dus altijd voorrang.
- **handel** alleen bij aan/uit of > 25 % afwijking van het doel. Kosten 25 bp per eenheid
  omzet (Kraken spot taker 22 bp + 3 bp halve spread). Geld in de sleeve mist de kasrente.

| trial | flow | trend | vol-doel | plafond |
| --- | --- | --- | --- | --- |
| F1 (hoofdkandidaat) | ja | ja | 10 % | 40 % |
| F2 | ja | ja | 5 % | 20 % |
| F3 | nee | ja | 10 % | 40 % |
| F4 | ja | nee | 10 % | 40 % |

M in de ledger: 32 → 36.

## 2. De criteria en de uitkomst

Alle vijf moeten gelden; hele cyclus 2020-04 .. 2026-09.

| criterium | drempel | F1 | F2 | F3 | F4 |
| --- | --- | --- | --- | --- | --- |
| CAGR boven U1 | ≥ +2 pp | +0,9 ✗ | +0,3 ✗ | +4,7 ✓ | +2,0 ✓ |
| Sharpe boven T-bill (maand) | ≥ 1,0 | 1,00 ✓ | 1,17 ✓ | 1,24 ✓ | 1,05 ✓ |
| max drawdown | ≤ 15 % | 8,4 % ✓ | 3,5 % ✓ | 7,0 % ✓ | 10,7 % ✓ |
| holdout boven U1 | > 0 | +2,3 pp ✓ | +1,1 pp ✓ | +4,9 pp ✓ | −0,0 pp ✗ |
| slechtste jaar t.o.v. U1 | ≤ 5 pp lager | 5,2 ✗ | 2,6 ✓ | 6,4 ✗ | 10,5 ✗ |

## 3. Per jaar (totaalrendement van het boek)

| jaar | U1 | F1 | F2 | F3 | F4 | BTC |
| --- | --- | --- | --- | --- | --- | --- |
| 2020 (apr–dec) | 8,7 % | 9,1 % | 9,0 % | 22,4 % | 10,5 % | +350 % |
| 2021 | 25,6 % | 30,1 % | 27,2 % | 30,8 % | 31,9 % | +63 % |
| 2022 | 1,3 % | −3,9 % | −1,3 % | −5,1 % | −9,2 % | −65 % |
| 2023 | 4,5 % | 11,7 % | 8,2 % | 14,9 % | 18,1 % | +154 % |
| 2024 | 7,9 % | 4,4 % | 5,2 % | 10,0 % | 10,0 % | +120 % |
| 2025 | 3,3 % | 6,9 % | 5,3 % | 7,4 % | 8,1 % | −5 % |
| 2026 (jan–sep) | 2,3 % | 2,3 % | 2,3 % | 5,2 % | 0,3 % | −5 % |

## 4. Wat opvalt

- **De sleeve is klein.** Gemiddeld 4–7 % van het kapitaal: bij een BTC-vol van ~60 % geeft
  een vol-doel van 10 % per munt ~8 %. In carryjaren is er bovendien weinig vrij geld.
- **De kosten zijn hoog voor die grootte**: 4–6 % over de hele cyclus, ~0,7–0,9 % per
  jaar. De band van 25 % op een klein gewicht en het aan/uit van de trend rond het
  gemiddelde draaien veel om. Een bredere band of wekelijks herbalanceren is een
  voor de hand liggende verbetering, maar zou nu, na het lezen, een nieuwe trial zijn.
- **2022 is het probleem.** Elke variant verliest daar: de trend gaf in het voorjaar en de
  zomer valse instappen. Zonder dat jaar zou F3 alle criteria halen.

## 5. Advies

1. **Niet live.** Geen trial haalt zijn criteria, en de live dienst blijft U1.
2. **F3 (alleen trend) als shadow vooruit volgen**, naast U1, zonder geld. Het is de enige
   variant met een duidelijke meerwaarde (+4,7 pp CAGR, holdout positief) bij een
   drawdown die binnen het mandaat blijft. Het oordeel valt dan op data die nog niemand
   heeft gezien.
3. **De CFTC-flow niet verder najagen** als timingsignaal. De data blijft in de repo voor
   wie het wil herzien.

## Reproduceren

```bash
# op Actions (research-run, taak positioning-series): data/external/{spot_close,cftc_tff}.csv
python -I -m tradebot.systematic.programme_v10 freeze   # al gedaan: M = 36
python -I -m tradebot.systematic.programme_v10 run      # artefacts/research/robust_book_v10/
```

Bekend PIT-risico: tijdens de sluiting van de Amerikaanse overheid (2025-10/11)
verschenen de CFTC-rapporten weken te laat. De CFTC publiceert geen historische
releasedatums, dus die weken lezen in de backtest iets te vroeg. Dat bevoordeelt de
flow-trials, en die zijn toch verworpen.
