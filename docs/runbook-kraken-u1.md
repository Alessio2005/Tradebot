# Draaiboek: het Kraken-boek (v9-U1)

Wat het boek is, wat het nu doet, en hoe het van shadow naar echt geld gaat. Achtergrond:
- `docs/research/2026-10-25-venue-kraken-vs-bybit.md` (waarom Kraken);
- `docs/research/2026-10-25-robust-book-v9-kraken-cash.md` (de cijfers).

## 1. Wat het boek doet

- **Long spot BTC en ETH, short de perp van dezelfde munt.** Het boek verdient de funding
  die de short ontvangt. Een prijsweddenschap is het niet: de twee benen heffen elkaar op.
- **Alleen als de carry het waard is.** Het boek stapt in bij een funding van gemiddeld
  minstens 15 %/jaar over 30 dagen, en stapt uit onder 5 %. Het werkt met twee slots van
  35 % van het kapitaal. Dit zijn de bevroren regels van v5-H3, die niet veranderen.
- **Op Kraken EU staat de spot als onderpand in de futureswallet.** Daar dekt hij de short.
  Er blijft 30 % USD als buffer. Retail mag maximaal 2:1 hefboom; het boek gebruikt 0,7:1.
- **Zonder carry staat het boek in kas.** Dat geld hoort in een dollar-geldmarktfonds
  buiten de beurs. Het boek meldt wanneer het geld naar de beurs moet en wanneer het terug
  kan. De overboeking zelf doet de eigenaar.

## 2. De stand (2026-10-27)

- **Backtest (bevroren H3-reeks, Kraken-fees, kasrente):** elk kalenderjaar positief van
  2020 tot en met 2026, met een CAGR van 8,0 %. Boven de T-bill was dat +4,9 %/jaar. In
  2022, 2023, 2025 en 2026 lag het boek iets onder cash.
- **Volgens de vooraf vastgelegde regels is het boek gefalsificeerd.** Twee redenen:
  1. de CAGR haalt de 15 % van het mandaat niet;
  2. in de holdout verdiende het niets boven cash.

  Kapitaal vraagt dus eerst een nieuw mandaat van de eigenaar, vastgelegd vóór een nieuwe
  trial.
- **Vooruit-sample:** loopt sinds 2026-10-01 en wordt eenmaal gelezen vanaf 2027-04-01,
  met `research-run`, taak `v9-forward`.
- **Shadow:** elke dag op de echte Kraken-data, via `.github/workflows/kraken-shadow.yml`.
  Het resultaat staat in de samenvatting van de run en in een artefact. De shadow verandert
  niets aan het boek. Wie op grond van de shadow het ontwerp wil veranderen, boekt een
  nieuwe trial.

## 3. Dagelijks

| wat | hoe |
| --- | --- |
| Shadow lezen | Actions → kraken-shadow → laatste run → Summary |
| Shadow met de hand | `python -I -m tradebot.carry.u1 shadow --out shadow` |
| Een stap (dry-run) | `python -I -m tradebot.carry.u1 step --state state/u1-dry.json --capital-usd N` |
| Stand van de staat | `python -I -m tradebot.carry.u1 status --state state/u1-dry.json` |
| Alles plat (kill switch) | `python -I -m tradebot.carry.u1 unwind --state S --reason TEKST [--live]` |

**Een stap draait na 00:00 UTC.** Hij gebruikt alleen afgesloten dagbars. Een tweede run
op dezelfde dag doet niets.

**De stap meldt twee bedragen:**
- `funding_request_usd`: zoveel USD moet er naar Kraken voordat het boek instapt. Tot het er
  staat, handelt de stap niet, en het besluit valt op de volgende bar opnieuw;
- `sweep_out_usd`: het boek heeft een week niets nodig gehad, dus dit bedrag kan naar het
  fonds.

## 4. Van dry-run naar echt geld

Doorloop deze stappen in deze volgorde. Elke stap moet slagen voordat de volgende begint.

1. **Mandaat.** Leg vast wat het boek moet halen, vóór een nieuwe trial. Bijvoorbeeld: elk
   jaar positief en de T-bill verslaan, in plaats van 15 % CAGR. Achteraf een criterium
   aanpassen is de zet die preregistratie uitsluit.
2. **Account.** Kraken EU: KYC, Kraken Pro en toegang tot derivaten. Voor complexe
   producten vraagt MiFID een geschiktheidstoets.
3. **Sleutels.** Er zijn twee sets nodig. Zet ze in de omgeving van de runner, nooit in de
   repo:
   - spot (`KRAKEN_SPOT_KEY`, `KRAKEN_SPOT_SECRET`): saldo opvragen en orders plaatsen en
     opvragen. `WalletTransfer` (spot → futures) vraagt volgens Kraken het recht
     *Funds: Query*. Geef geen recht op opnames naar externe adressen;
   - futures (`KRAKEN_FUTURES_KEY`, `KRAKEN_FUTURES_SECRET`): handelen en overboeken naar
     spot. Futures heeft eigen sleutels.

   Beperk beide sleutels tot het IP-adres van de runner.
4. **Haircuts.** Vraag bij Kraken de haircut op BTC- en ETH-onderpand op. De buffer van
   30 % USD dekt een verdubbeling van de koers bij een haircut tot 35 %.
5. **Futures op de demo.** Draai `KrakenVenue(..., dry_run=False, demo=True)` met
   demo-sleutels. Toets: open en sluit een short, met en zonder `reduceOnly`, en controleer
   de handtekening. Spot heeft geen demo.
6. **Dry-run met het echte kapitaal.** Draai minstens twee weken elke dag
   `step --state state/u1-dry.json --capital-usd N`. Vergelijk de orders met de shadow. De
   dry-run gebruikt de echte metadata van Kraken (minimum ordergrootte, precisie).
7. **Live, klein.** Begin een NIEUWE staat (een staat wisselt niet van modus), met een
   klein kapitaal:
   `step --state state/u1-live.json --capital-usd 5000 --live`. Elke run reconcilieert
   eerst de beurs tegen wat het boek verwacht. Bij een verschil (een gemiste fill, een
   handmatige ingreep, een liquidatie) stopt het boek tot iemand het onderzoekt.
8. **Opschalen.** Pas na een paar weken zonder breuk. Boven $400k bezit valt het boek in de
   goedkoopste spot-tier (22 bp taker in plaats van 80 bp).

## 5. Wat het boek NIET doet

- **Het belooft geen winst.** Over 6,5 jaar backtest was elk jaar positief in dollars. Elke
  lezing van die periode is besmet, en het vooruit-sample is nog te kort.
- **Het dekt de dollar niet af.** Voor een belegger in euro's met een afgedekte dollar was
  2022 −0,7 %.
- **Het verplaatst geen geld naar of van het fonds.** Het zegt alleen wanneer.
- **Het handelt geen altcoins.** Op Kraken EU tellen die niet als onderpand.

## Bronnen

- [Kraken: Request Wallet Transfer](https://docs.kraken.com/api-reference/funding/request-wallet-transfer)
  (recht *Funds: Query*; alleen spot → futures, de andere kant gaat via de futures-API)
- [Kraken: market candles (charts API)](https://docs.kraken.com/api-reference/candles/market-candles)
