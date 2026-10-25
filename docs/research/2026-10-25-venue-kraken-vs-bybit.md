# Beurswissel: Kraken of Bybit in plaats van Binance (2026-10-25)

De vraag van de eigenaar: Binance eruit, Kraken of Bybit erin. Wat kan, wat is nodig en wat
is het goedkoopst? Het antwoord geldt voor een **Nederlandse retailklant**. Het is een stand
van zaken op 25 oktober 2026, uit openbare bronnen (onderaan). Voordat er kapitaal heen gaat,
moeten de feetabel en de producttoegang op de site van de beurs zelf nog een keer gecontroleerd
worden: de ontwikkelomgeving kon de pagina's van de beurzen niet zelf openen.

## Samenvatting

- **Bybit valt af.** Bybit EU heeft een MiCA-vergunning (Oostenrijk) en biedt Nederlandse
  klanten alleen spot.
  - Voor perps is een MiFID II-vergunning nodig. Die is aangevraagd, maar niet verleend.
  - Bybit global bedient EER-klanten sinds 1 juli 2026 niet meer.
  - Een carryboek is long spot plus short perp. Zonder perp is er geen boek.
- **Kraken EU kan beide benen**: spot via de MiCA-entiteit in Ierland, perps via de
  MiFID-entiteit in Cyprus. Er zijn vier beperkingen die Binance niet had:
  1. **Spotfees** zijn 4 à 8 keer zo hoog bij een klein account. Bij $1 mln aangehouden
     vermogen is het 22 bp taker of 10 bp maker.
  2. **Retail mag maximaal 2:1 op perps.** ESMA ziet perps als CFD's.
  3. **Geen unified margin.** Spot/margin en perps zitten bij twee juridische entiteiten. De
     USD-lening die in v7/v8 spot financiert en tegelijk de short dekt, bestaat daar niet.
  4. **Perp-onderpand** kan alleen BTC, ETH, USDC, USDT, EUR of GBP zijn. De haircuts zijn
     niet gepubliceerd gevonden.
- **Geen van de vier bevroren boeken draait ongewijzigd op Kraken.**
  - Eén regelset past wel: **v5-H3 (BTC en ETH), met de spot als onderpand in de
    futureswallet.** Dezelfde regels, hetzelfde been van 0,70.
  - De short wordt dan gedekt door 0,70 spot plus 0,30 USD. Perp-notional / onderpand is
    0,70, ruim onder 2:1.
  - In de Binance-indeling (twee wallets) was dat 0,70 / 0,30 = 2,33. Dat zit boven het
    retailplafond.
- **Wat het kost:** op de bevroren dagreeksen verandert de feewissel weinig, want de boeken
  handelen weinig.
  - H3: 6,2 % CAGR met de Binance-fees, 6,0 % bij Kraken taker op $1 mln en 5,3 % bij
    Kraken taker op een nieuw account.
  - Het probleem is niet de fee. Het probleem is wat het boek in een jaar zonder carry doet:
    stilzitten op onbelegde dollars.

## 1. Wat kan waar

| | Binance | Bybit EU | Kraken EU |
| --- | --- | --- | --- |
| Toegestaan voor NL-klanten | nee (sinds 2023) | ja (MiCA, AT) | ja (spot MiCA IE, perps MiFID CY) |
| Perps voor NL/EER | — | **nee** (MiFID aangevraagd, niet verleend) | ja |
| Max. perp-hefboom retail | — | — | 2:1 (CFD-regime) |
| Unified margin (lening + spot + perp in één account) | ja | — | **nee** |
| Wat telt als perp-onderpand | elke munt | — | BTC, ETH, USDC, USDT, EUR, GBP |
| Spotfee taker, $0 / $1 mln | 10 / 10 bp | 10 bp (niet geverifieerd) | 80 / 22 bp |
| Spotfee maker, $0 / $1 mln | 10 / 10 bp | 10 bp (niet geverifieerd) | 40 / 10 bp |
| Perpfee maker / taker | 5,5 / 5,5 bp (bevroren config) | — | 2 / 5 bp |
| Fundinginterval | 8 uur | — | elk uur (gemeten over 4 uur) |

Kraken bepaalt de tier op 30-daags volume **of** op aangehouden vermogen (schema van 9 juli
2026). In `conf/execution/venues.yaml` staat de hele tabel, met de bron per regel.

## 2. Welke boeken kunnen waar

`python -I -m tradebot.carry.venues` schrijft `artefacts/research/venue_comparison.json`.
Per boek staat daarin waarom het op een beurs niet kan (`VenueProfile.blockers`), en wat het
over de hele cyclus 2020-04 .. 2026-09 zou opleveren met de fees van die beurs.

| Boek | Kraken EU | Bybit EU |
| --- | --- | --- |
| v5-H3 majors, twee wallets | perp/marge 2,33 > 2:1; kan alleen met een been ≤ 0,67 of als professionele klant | geen perps |
| **v5-H3 majors, spot als onderpand** | **geen blokkade** | geen perps |
| v6-K1 carry 1×, top-50 | altcoins tellen niet als onderpand | geen perps |
| v7-S2 carry met tranche | geen unified margin; altcoins tellen niet | geen perps |
| v8-T2 S2 + sprongrisicogrens | geen unified margin; altcoins tellen niet | geen perps |

## 3. Wat het kost

Fee per paar benen (spot plus perp, één kant):

| Scenario | bp per paar |
| --- | --- |
| Binance (bevroren model) | 15,5 |
| Kraken, $1 mln, maker | 12,0 |
| Kraken, $1 mln, taker | 27,0 |
| Kraken, nieuw account, taker | 85,0 |

Hele cyclus 2020-04 .. 2026-09. Per cel: CAGR / maand-Sharpe / verliesjaren.

| Boek | Binance | Kraken $1 mln maker | Kraken $1 mln taker | Kraken nieuw, taker |
| --- | --- | --- | --- | --- |
| H3 (spot als onderpand) | 6,2 % / 1,61 / 2022 | 6,2 % / 1,61 / 2022 | 6,0 % / 1,58 / 2022 | 5,3 % / 1,41 / 2022, 2023 |
| K1 | 8,8 % / 1,19 / 2022, 2025, 2026 | 8,9 % / 1,20 / idem | 8,5 % / 1,14 / idem | 6,8 % / 0,93 / idem |
| S2 | 12,9 % / 1,37 / 2025, 2026 | 13,0 % / 1,38 / idem | 12,4 % / 1,33 / idem | 9,9 % / 1,12 / + 2022 |
| T2 | 13,4 % / 1,55 / 2025, 2026 | 13,5 % / 1,56 / idem | 12,9 % / 1,51 / idem | 10,4 % / 1,28 / + 2022 |

Dit is een overlay, en dat is de aanname:
- de fee is exact: de framerij is een sluitende som, en de bevroren boeken rekenen precies
  15,5 bp per paar;
- spread, impact en funding blijven die van Binance;
- de instapdrempels zijn niet opnieuw gekozen voor de hogere kosten.

Kraken heeft voor BTC en ETH diepe boeken, maar dunner dan Binance. De funding op Kraken is
niet dezelfde reeks. Hoeveel die scheelt, meet de workflow in §5.

**Goedkoopst:** Kraken is de enige optie die het boek kan hosten. Met limietorders (maker)
en $1 mln aangehouden vermogen is het goedkoper per paar dan het Binance-model (12 tegen
15,5 bp). Met marktorders op een nieuw account is het ruim vijf keer zo duur (85 bp). De
adapter stuurt nu marktorders. Maker-uitvoering is een vervolgstap: limietorders met een
time-out en daarna een marktorder.

## 4. Wat er nodig is voordat er kapitaal heen gaat

1. **Een Kraken-account met KYC, plus toegang tot derivaten.** Voor complexe producten
   onder MiFID verwacht ik een geschiktheidstoets. Controleer dat bij de onboarding.
2. **Twee sets API-sleutels.**
   - Kraken spot (`api.kraken.com`) met de rechten saldo opvragen, handelen en
     `WalletTransfer` naar de futureswallet.
   - Kraken Futures (`futures.kraken.com`), apart aangemaakt.
   - Geen opnamerecht naar externe adressen.
3. **De haircuts op BTC- en ETH-onderpand.** Die zijn niet gepubliceerd gevonden. Vraag ze
   bij Kraken op en zet ze in `venues.yaml`. Tot dan rekent het boek met de 0,30 USD-buffer
   van H3.
4. **De adapter valideren** (`src/tradebot/carry/kraken.py`).
   - Futures: op `demo-futures.kraken.com`. De handtekening van Futures is volgens de
     documentatie gebouwd, maar niet tegen een gepubliceerde testvector getoetst. Die van
     spot wel.
   - Spot heeft geen testnet. Spot draait daarom eerst in dry-run (de standaard), en daarna
     met het kleinste toegestane bedrag.
5. **Minimale ordergrootte en precisie** per paar komen uit `AssetPairs` en `instruments`.
   De adapter weigert orders onder het minimum. Een boek van een paar duizend euro haalt het
   minimum per munt, maar betaalt dan de duurste tier (§3).
6. **Professionele klantstatus** is niet nodig voor H3 met spot als onderpand. Alleen voor
   hefboom boven 2:1, en daarvoor gelden eigen MiFID-criteria.

## 5. Wat er gebouwd is

- `conf/execution/venues.yaml`: de profielen van Binance, Bybit EU en Kraken EU, met datum en
  bronnen.
- `src/tradebot/carry/venues.py`: profielen laden en valideren, de blokkades per boek, de
  kosten per beurs (`basis_costs`), de fee-overlay en de vergelijking.
- `src/tradebot/carry/kraken.py`: de Kraken-adapter voor het carryboek.
  - Spot `AddOrder` met polling tot de status `closed`.
  - Overboeking van onderpand naar en van de futureswallet.
  - Perp-orders met `reduceOnly` bij het afbouwen.
  - Standaard in dry-run: zonder expliciete opt-in gaat er geen order naar buiten.
- `src/tradebot/data/external_series.py` en `.github/workflows/research-run.yml`.
  - Haalt op een GitHub-runner de T-bill (FRED DTB3) en €STR (ECB) op, plus de dagfunding van Kraken
    (PF_XBTUSD, PF_ETHUSD) en Binance (BTCUSDT, ETHUSDT), tot en met 2026-09-30.
  - Commit het resultaat terug. De ontwikkelomgeving bereikt die hosts zelf niet.

## 6. Wat open blijft

- **De funding van Kraken tegen die van Binance.** Gemeten door de workflow van §5. De uitkomst
  staat in het v9-rapport.
- **Spread en impact op Kraken** zijn niet gemeten. Daarvoor is orderboekdata van Kraken
  nodig.
- **De adapter heeft nog geen echte order gezien** (§4.4).
- **Bybit komt terug in beeld zodra Bybit EU een MiFID-vergunning voor derivaten krijgt.** De
  verwachting is rond november 2026, niet bevestigd. Dan is `bybit_eu` in `venues.yaml` met
  één wijziging een kandidaat.

## Bronnen

- [Kraken: fee schedule](https://www.kraken.com/features/fee-schedule)
- [Kubenote: Kraken fees, een complete gids](https://kubenote.com/kraken-cryptocurrency-fees-a-complete-guide-to-cost-efficient-trading/) (tiers van 9 juli 2026)
- [DollarScout: Kraken review](https://www.dollarscout.net/crypto/reviews/kraken/)
- [Kraken blog: crypto collateral for EU futures](https://blog.kraken.com/product/kraken-derivatives/crypto-collateral-eu-futures)
- [CoinDesk: Kraken rolls out regulated crypto derivatives in Europe](https://www.coindesk.com/business/2025/05/20/kraken-rolls-out-regulated-crypto-derivatives-in-europe)
- [Finance Magnates: ESMA tells firms perpetual futures fall under EU CFD rules](https://www.financemagnates.com/forex/regulation/esma-tells-firms-perpetual-futures-fall-under-eu-cfd-rules/)
- [Kraken: funding rates of perpetuals](https://www.kraken.com/au/pro/perps/crypto-perpetuals/funding-rates)
- [CASP Tracker: Bybit](https://casptracker.eu/exchange/bybit/)
- [Finance Magnates: Bybit pursues MiFID licence for EU derivatives](https://www.financemagnates.com/cryptocurrency/bybit-pursues-mifid-license-for-eu-derivatives-phases-out-mt5-for-in-house-trafi/)
- [Leodex: Bybit EU MiCA migration](https://leodex.io/learn/country-restrictions/bybit-eu-mica-migration)
