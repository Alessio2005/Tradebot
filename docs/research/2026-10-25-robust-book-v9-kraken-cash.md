# Robuust boek v9: het carryboek op Kraken EU, met de kasrente (2026-10-25)

De opdracht: verbeter de bestaande strategieën zodat ze elk jaar winstgevend zijn, na de
overstap van Binance naar Kraken (`2026-10-25-venue-kraken-vs-bybit.md`). De regels zijn
die van v5–v8:
- de trial staat in de ledger vóór de run (commit `eab66eb`, M = 32);
- de preregistratie is bevroren vóór de meting (`b366dbfe7cb1cc03d08be352e4b9c9cd`);
- geen drempel is versoepeld.

## Samenvatting

- **Op Kraken EU kan maar één bevroren regelset draaien: v5-H3** (spot-perp-carry in BTC
  en ETH), met de spot als onderpand in de futureswallet. K1, S2 en T2 hebben unified
  margin of altcoins als onderpand nodig, en die zijn er op Kraken EU niet. Die boeken
  "verbeteren" heeft geen zin zolang ze nergens kunnen draaien.
- **Het probleem van H3 was niet verliezen, maar stilzitten.**
  - In 2022, 2023, 2025 en 2026 stond H3 voor 85–100 % in dollars.
  - In de backtest brachten die dollars niets op. In werkelijkheid staan ze in een
    geldmarktfonds.
  - v9 (kandidaat U1) telt dat mee, tegen DTB3 min 50 bp, met conservatieve
    overboekingsregels. Het rekent ook de fees van Kraken.
- **Resultaat: elk kalenderjaar positief.**

  | jaar | U1 | boven de T-bill | waarvan kasrente | in het fonds | H3 (bevroren, Binance) |
  | --- | --- | --- | --- | --- | --- |
  | 2020 (apr–dec) | +8,7 % | +8,6 % | 0,0 % | 18 % | +9,0 % |
  | 2021 | +25,6 % | +25,5 % | 0,0 % | 15 % | +25,9 % |
  | 2022 | **+1,3 %** | −0,7 % | +1,5 % | 88 % | −0,1 % |
  | 2023 | +4,5 % | −0,7 % | +4,2 % | 92 % | +0,3 % |
  | 2024 | +7,9 % | +2,7 % | +1,2 % | 27 % | +6,9 % |
  | 2025 | +3,3 % | −0,8 % | +2,8 % | 79 % | +0,5 % |
  | 2026 (jan–sep) | +2,3 % | −0,5 % | +2,3 % | 95 % | 0,0 % |
  | **hele cyclus (CAGR)** | **8,0 %** | **+4,9 %** | | | 6,2 % |

- **Risico, eerlijk gemeten** (hele cyclus, maandrendementen):

  | | Sharpe | max drawdown |
  | --- | --- | --- |
  | U1 | 2,18 | 0,33 % |
  | U1 boven de T-bill | 1,27 | 1,85 % |

  3 van de 78 maanden waren negatief; de slechtste was −0,08 %.
- **Formeel is v9 gefalsificeerd. Twee vooraf vastgelegde criteria binden:**
  1. **g2:** de CAGR op W_DEV is 8,75 %, onder de 15 % van het mandaat.
  2. **holdout_sharpe_collapse:** in de holdout (2025-07 .. 2026-09) had H3 geen carry. U1
     was daar een geldmarktfonds dat 50 bp kost: +3,96 % tegen +4,83 % voor de T-bill.
     - Boven de T-bill is dat een vrijwel constant klein verlies, met bijna geen
       variantie. De iid-Sharpe daarvan is −1220 en z = −12,7.
     - De grootte van dat getal is een artefact van de lage variantie, maar het teken is
       echt: in de holdout verdiende de strategie niets boven cash.
- **Wat dit is:** een boek dat in carryjaren 5–25 % verdient en in de andere jaren
  ongeveer de T-bill haalt. Dat is "elk jaar winst" in dollars. Het is geen 15 %-strategie,
  en boven cash heeft het in vier van de zeven jaren licht verloren.

## 1. Wat v9 is, en wat niet

- **De invoer is de bevroren dagreeks van v5-H3.** Dezelfde regels, hetzelfde been
  (2 × 35 %), dezelfde drempels: 15 % in, 5 % uit, funding-EWMA van 30 dagen. Er is
  niets opnieuw gefit.
- **Fees van Kraken:** tier $1 mln, taker, 27 bp per paar benen tegen 15,5 bp. Exact via
  de framerij (`carry.venues.fee_overlay`).
- **De kasrente** (`conf/model/robust_book_v9.yaml`, elke waarde vooraf gekozen):
  - het rendement is DTB3 van de vorige dag, min 50 bp, nooit onder nul, op 365 dagen;
  - een vaste werkvoorraad van 5 % blijft op de beurs;
  - geld dat H3 vrijgeeft, gaat pas na 7 dagen naar het fonds;
  - geld dat H3 nodig heeft, staat 2 dagen vóór de transactie op de beurs. Het besluit
    valt een dag vooraf (lag 1).

  Deze regels kunnen alleen rente wegnemen, nooit een transactie beter maken.
- **De Sharpe-poorten meten het rendement boven de T-bill.** Zonder die aftrek is elke
  kasstrategie een Sharpe-wonder. Daarbij komen drie criteria die strenger zijn dan v5–v8:
  - de Sharpe op maandrendementen ≥ 1,0;
  - elk kalenderjaar positief;
  - over de hele cyclus de T-bill verslaan.
- **Wat de handel van H3 betreft, zijn vier maten overgenomen uit de batterij van v5:**
  vertraging, plateau, minimale verstoring en PBO. De regels zijn identiek, dus een nieuwe
  batterij zou dezelfde handel opnieuw meten.

## 2. Alle poorten

| poort | waarde | drempel | bindt |
| --- | --- | --- | --- |
| g1 Sharpe boven T-bill (iid, W_DEV) | 5,14 | ≥ 1,0 | nee |
| g1m idem, op maandrendementen | 1,23 | ≥ 1,0 | nee |
| **g2 netto CAGR W_DEV** | **8,75 %** | ≥ 15 % | **ja** |
| g3 train / validate | 6,02 / 2,89 | ≥ 0,5 | nee |
| g4 max drawdown W_DEV | 0,33 % | ≤ 25 % | nee |
| g5 Sharpe bij 2× kosten / basis | 0,83 | ≥ 0,75 | nee |
| g5 Sharpe bij lag 2 / basis (v5) | 0,99 | ≥ 0,5 | nee |
| g6 plateau / minimale verstoring (v5) | 1,0 / 6,17 | ≥ 0,8 / ≥ 0 | nee |
| g7 bootstrap-ondergrens | 1,99 | > 0 | nee |
| g8 DSR (M = 32, iid) | 1,00 | ≥ 0,95 | nee |
| g9 PBO (v5) | 0,002 | < 0,5 | nee |
| g10 liquidaties W_DEV | 0 | 0 | nee |
| g11 sluitgrens bij 35 % haircut | 2,38× | ≥ 2,0× | nee |
| g13 slechtste kalenderjaar | +1,33 % | ≥ 0 | nee |
| g14 CAGR boven de T-bill | +4,88 % | > 0 | nee |
| backcast z / Sharpe boven T-bill | +3,08 / 10,8 | > −1,645 / > 0 | nee |
| **holdout z** | **−12,7** | > −1,645 | **ja** |

De g8-DSR en de iid-Sharpe zijn met de bevroren schatter gemeten. Ze zijn opgeblazen
(audit 2026-10-24, SK-1). g1m is de eerlijke maat.

## 3. Gevoeligheid

Hele cyclus. Per scenario: CAGR, slechtste jaar en de verliesjaren.

| scenario | CAGR | slechtste jaar | verliesjaren |
| --- | --- | --- | --- |
| U1 (Kraken $1 mln, taker) | 8,0 % | +1,3 % (2022) | geen |
| Kraken $1 mln, maker (12 bp) | 8,2 % | +1,4 % | geen |
| Kraken nieuw account, taker (85 bp) | 7,2 % | +0,9 % | geen |
| H3 op Kraken zonder kasrente | 6,0 % | −0,2 % | 2022 |
| **Belegger in euro's, dollar afgedekt** | 6,4 % | **−0,7 % (2022)** | **2022** |
| alleen de T-bill | 3,0 % | +0,05 % | geen |

**De afgedekte euro-belegger.** Het boek is in dollars. Wie in euro's belegt en de dollar
afdekt, betaalt ongeveer DTB3 − €STR over het hele kapitaal. In 2022 was dat ~2 %, en
daarmee wordt 2022 licht negatief. Zonder afdekking domineert de EUR/USD-koers elk jaar
(±10 %). Voor een Nederlandse belegger is "elk jaar winst" in euro's dus **niet**
gehaald.

## 4. De funding van Kraken tegen die van Binance

Kraken publiceert zijn fundinghistorie (PF_XBTUSD, PF_ETHUSD) vanaf 2025-10-15. Over die
351 dagen:

| | Kraken | Binance | verschil | correlatie |
| --- | --- | --- | --- | --- |
| BTC | 3,0 %/jaar | 3,3 %/jaar | −0,3 pp | 0,62 |
| ETH | 3,3 %/jaar | 2,5 %/jaar | +0,8 pp | 0,58 |

Gemiddeld gelijk, maar met een correlatie van ~0,6 per dag. De in- en uitstap van H3
rekent op de Binance-funding. In die periode stond het boek niet in carry, dus het
verschil verandert U1 niet. Over 2020–2025 is de vergelijking niet te maken: die
Kraken-historie is niet publiek.

## 5. Wat het betekent

- **Als het doel "nooit een verliesjaar in dollars" is, met een drawdown onder 0,5 %:** U1
  doet dat over 6,5 jaar backtest, met 8 % CAGR. Het is in feite een geldmarktfonds dat in
  carryjaren (2020, 2021, 2024) de carry van BTC en ETH oogst.
- **Als het doel 15 % CAGR is:** dat haalt geen boek dat op Kraken EU kan draaien. T2 (het
  beste boek, 13,4 %) heeft unified margin en altcoins als onderpand nodig. Op Kraken EU
  kan dat niet, en Bybit EU heeft geen perps. Dat is een structurele grens van de beurs,
  geen parameter.
- **Of v9 doorgaat, is een besluit voor de eigenaar.** Formeel is v9 gefalsificeerd door
  de vooraf vastgelegde holdoutregel. Het mandaat (15 % CAGR) en het nieuwe doel (elk jaar
  winst) spreken elkaar hier tegen. Een criterium achteraf aanpassen is de zet die
  preregistratie uitsluit. Een nieuw mandaat moet vóór een nieuwe trial worden vastgelegd.

## 6. Wat open blijft

- **Elke lezing hier is besmet.** v5 las W_DEV, de backcast en de holdout, en v9 is
  ontworpen met de jaarcijfers van H3 in de hand. Het eerste schone bewijs is het
  vooruit-sample (vanaf 2026-10-01, slot bevroren op 2026-10-09).
- **De vooruit-lezing van v9 bestaat nog niet.** Die moet v5-H3 vooruit simuleren op de
  prijzen en funding van Kraken. Hij moet gebouwd zijn vóór de eerste lezing, op zijn
  vroegst in april 2027.
- **Spread en impact zijn die van Binance.** Kraken is dunner.
- **De kasrente vraagt een rekening bij een broker met een dollar-geldmarktfonds**, plus
  een route via USDC naar Kraken. De kosten daarvan zijn hier op 50 bp per jaar gezet,
  niet gemeten.
