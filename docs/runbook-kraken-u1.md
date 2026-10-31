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
- **Eerste shadow op de echte Kraken-data** (2026-10-27, bar van 26 oktober):
  - de carry over 30 dagen is 10,4 % voor BTC en 8,2 % voor ETH, onder de instap van 15 %;
  - het boek staat dus in kas;
  - vooruit (27 dagen): +0,25 %, alleen kasrente, en −0,05 % boven de T-bill;
  - op de Kraken-data sinds oktober 2025 is het boek niet ingestapt. 2026 tot nu: +2,5 %,
    −0,5 % boven de T-bill.
- **Shadow:** elke dag op de echte Kraken-data, via `.github/workflows/kraken-shadow.yml`.
  Het resultaat staat in de samenvatting van de run en in een artefact. De shadow verandert
  niets aan het boek. Wie op grond van de shadow het ontwerp wil veranderen, boekt een
  nieuwe trial.

## 3. Op een server (24/7)

De dienst `python -I -m tradebot.carry.service` draait het boek zonder toezicht. Hij doet
elke 15 minuten één ronde:
1. **Heartbeat.** Een bestand voor de healthcheck, en een GET naar `HEARTBEAT_URL`.
2. **Lock.** Draait er al een tweede proces op dezelfde staat, dan slaat de ronde over.
3. **UNWIND-bestand.** Is het er, dan sluit de dienst alles en zet het boek stil.
4. **Watchdog.**
   - De beurs tegen het boek.
   - Een hedge-gat, en de marge.
   - De klok tegen die van Kraken.
5. **De dagelijkse stap.** Eenmaal per dagbar, na 00:10 UTC. Is hij om 06:00 nog niet gelukt,
   dan volgt een alarm.

Een ronde die faalt (netwerk, beurs, data), laat de dienst draaien. De volgende ronde
probeert opnieuw, en na drie mislukte rondes op rij volgt een alarm.

**Server:** een kleine VPS in de EU (1 vCPU, 2 GB) met Docker, of Linux met Python 3.13 en
systemd. Zet de klok gelijk met NTP (chrony of systemd-timesyncd); de nonce van Kraken en de
dagwissel leunen erop.

**Met docker compose** (aanbevolen):

```bash
git clone <repo> /opt/tradebot && cd /opt/tradebot
cp conf/live/u1.example.yaml infra/compose/u1.yaml      # kapitaal, grenzen, mode: dry_run
cp infra/compose/.env.example infra/compose/.env         # alarmkanalen, later de sleutels
chmod 600 infra/compose/.env
docker compose -f infra/compose/docker-compose.yml up -d --build
docker compose -f infra/compose/docker-compose.yml logs -f u1
```

De container is zo ingericht:
- read-only en zonder capabilities;
- hij draait als eigen gebruiker en schrijft alleen in het volume `tradebot-state`;
- na een crash of reboot start hij opnieuw (`restart: unless-stopped`).

**Met systemd:** zie de kop van `infra/systemd/tradebot-u1.service`. De unit herstart altijd
na 30 s, en alleen `/var/lib/tradebot` is schrijfbaar.

**Alarmen.** Zet minstens één kanaal in `.env`: `ALERT_WEBHOOK_URL` (Slack, Mattermost,
Discord) of `TELEGRAM_BOT_TOKEN` met `TELEGRAM_CHAT_ID`.
- *Kritiek:* het boek staat stil, een stap is geweigerd, een hedge-gat, marge, UNWIND.
- *Waarschuwing:* mislukte rondes, de stap te laat, de klok, HALT, zet geld op Kraken.
- *Info:* een dagelijkse samenvatting, en dat er geld naar het fonds kan.

Hetzelfde alarm komt hoogstens eens per uur; een kritiek alarm hoogstens eens per kwartier.

**Een dead man's switch.** Zet `HEARTBEAT_URL`, bijvoorbeeld een gratis check bij
healthchecks.io met een periode van 15 minuten en een grace van 30 minuten. Valt de server
weg of hangt de container, dan kan de dienst zelf niets melden. Dit is dan het enige alarm.

| wat | hoe (compose; met systemd zonder `docker compose ... exec u1`) |
| --- | --- |
| Logs | `docker compose -f infra/compose/docker-compose.yml logs -f u1` |
| Stand | `... exec u1 python -I -m tradebot.carry.u1 status --config /etc/tradebot/u1.yaml` |
| Geen nieuw risico | `... exec u1 touch /var/lib/tradebot/HALT` (weg: `rm`) |
| Alles plat | `... exec u1 sh -c 'echo reden > /var/lib/tradebot/UNWIND'` |
| Hervatten na stilstand | `... exec u1 python -I -m tradebot.carry.u1 resume --config /etc/tradebot/u1.yaml --reason "onderzocht: ..."` |
| Logboek | `/var/lib/tradebot/u1-journal.jsonl` (één regel per stap of ingreep) |
| Vorige staten | `/var/lib/tradebot/backups/` (de laatste 30) |
| Bijwerken | `git pull && docker compose -f infra/compose/docker-compose.yml up -d --build` |

**Stilstand is de veilige toestand.** Het boek gaat alleen weer handelen na `resume`, en dat
doet een mens. Lees eerst het alarm en het logboek. `resume` neemt wat de beurs NU houdt als
nieuwe verwachting.

**Drie gevallen die je moet herkennen:**
- *Een harde crash midden in een stap* (stroom weg, SIGKILL). De orders zijn uitgevoerd,
  maar de staat is niet opgeslagen. De volgende ronde ziet dat de beurs afwijkt van het boek
  en zet het boek stil, met een kritiek alarm. Er wordt niets dubbel gehandeld. Kijk in het
  logboek en op Kraken, en doe dan `resume`.
- *Een update die de config van het boek verandert* (bv. nieuwe fees in
  `conf/execution/venues.yaml`). De stap weigert de oude staat. Accepteer de nieuwe config
  bewust met `resume --accept-config --reason "..."`.
- *"Zet X USD op Kraken".* Een instap koopt spot met USD uit de **spotwallet**. Zet het geld
  daar, niet in de futureswallet. De dienst probeert het elke ronde opnieuw.

**Zonder dienst, met de hand:**
- `python -I -m tradebot.carry.u1 shadow --out shadow` draait de shadow;
- `python -I -m tradebot.carry.u1 step --config C` draait één stap.

De CLI neemt dezelfde lock als de dienst. Draaien ze tegelijk, dan wacht de één.

## 4. Veiligheid: wat het boek tegenhoudt

De grenzen staan in `limits:` van de serverconfig (`carry/safety.py`). Een grens kan een
besluit alleen tegenhouden of risico laten AFBOUWEN; hij kan nooit risico opbouwen.

| controle | wanneer | wat er gebeurt |
| --- | --- | --- |
| Reconciliatie | elke stap en elke ronde | beurs wijkt af van het boek: stilstand + kritiek alarm |
| Verlies van de gehedgede posities | elke stap | meer dan `max_step_loss` sinds de vorige stap: stilstand |
| Kapitaal, ordergrootte, bruto | elke stap | boven het plafond: stap geweigerd, stilstand |
| Live koers tegen de close | elke stap met orders | meer dan `max_price_deviation`: de volgende ronde opnieuw |
| Verse data | elke stap | de dagbar is er nog niet: de volgende ronde opnieuw |
| Order voor order | uitvoering | een fout: stoppen, de beurs lezen, een hedge-gat reduce-only dichten, stilstand |
| Na de uitvoering | elke stap | de beurs wijkt af of een hedge-gat: stilstand |
| Hedge-gat | elke ronde | het overschot reduce-only sluiten, stilstand |
| Marge-headroom | elke ronde | onder `margin_warn`: waarschuwing; onder `margin_critical`: alles sluiten, stilstand |
| Klok | elke ronde | meer dan `max_clock_skew_s` van Kraken: waarschuwing |
| HALT / UNWIND | elke ronde | geen nieuw risico / alles sluiten |
| Eén proces | elke ronde en elke CLI-opdracht | een tweede proces op dezelfde staat wacht |

**Wat niet automatisch gebeurt:** een mislukte order opnieuw proberen. Een tweede poging kan
een dubbele order zijn. De dienst repareert alleen door af te bouwen, en zet het boek stil.

## 5. Van dry-run naar echt geld

Doorloop deze stappen in deze volgorde. Elke stap moet slagen voordat de volgende begint.

1. **Mandaat.** Leg vast wat het boek moet halen, vóór een nieuwe trial. Bijvoorbeeld: elk
   jaar positief en de T-bill verslaan, in plaats van 15 % CAGR. Achteraf een criterium
   aanpassen is de zet die preregistratie uitsluit.
2. **Account.** Kraken EU: KYC, Kraken Pro en toegang tot derivaten. Voor complexe
   producten vraagt MiFID een geschiktheidstoets. Gebruik een eigen account of subaccount
   voor het boek. De reconciliatie ziet elke BTC, ETH of perp-positie die niet van het
   boek is als een breuk, en stopt het boek dan.
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
6. **De dienst in dry-run, met het echte kapitaal.** Laat hem minstens twee weken draaien
   op de server (§3), met `mode: dry_run` en de alarmen aan. Vergelijk de dagelijkse
   samenvatting met de shadow op Actions. Test de noodknoppen ook één keer: HALT, UNWIND en
   `resume`. De dry-run gebruikt de echte metadata van Kraken (minimum ordergrootte,
   precisie).
7. **Live, klein.**
   - Zet `mode: live`, een klein `capital_usd` (bv. 5000) en een strak `max_order_usd`.
   - Geef het een NIEUW state-pad, want een staat wisselt niet van modus.
   - Zet de vier sleutels in `.env` en herstart de dienst.
   - Elke stap reconcilieert eerst de beurs. Bij een verschil (een gemiste fill, een
     handmatige ingreep, een liquidatie) stopt het boek tot iemand het onderzoekt.
8. **Opschalen.** Pas na een paar weken zonder breuk. Boven $400k bezit valt het boek in de
   goedkoopste spot-tier (22 bp taker in plaats van 80 bp).

## 6. Wat het boek NIET doet

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
