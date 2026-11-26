# Runbook: de options-engine (paper)

De options-engine handelt Amerikaanse aandelen- en ETF-opties met regelgebaseerde strategieën.
Alles draait met **fictief geld**: replay op opgeslagen data, of een IBKR-paper-account met
vertraagde marktdata. Live bestaat in de code als schakelaar, maar staat uit en kan vanuit
deze repository niet aan (zie "Live" onderaan).

## Snel: replay op synthetische data

```bash
python -I -m tradebot.options.data.synthetic --out /tmp/markt
python -I -m tradebot.options replay --market /tmp/markt --out /tmp/run
```

`/tmp/run` bevat dan:

- `journal.jsonl`: elke gebeurtenis (keten, kandidaat, risicobesluit, order, fill, positie).
- `state.json`: posities, orders, equity per dag.
- `summary.json`: begin- en eind-equity, aantal posities, en of er iets misging.

De exit-code is 1 als de engine haltte of als er na de laatste dag nog een short optie open
staat die al expireerde. De workflow `options-paper-smoke.yml` doet precies dit.

## Historische simulatie (echte koers en IV, ketens uit het model)

```bash
docker compose -f infra/compose/docker-compose.yml run --rm --no-deps -e IB_CLIENT_ID=18 \
  options-paper python -I -m tradebot.options historical \
  --out /var/lib/tradebot/historical/markt --start 2026-06-01 --end 2026-11-25
docker compose -f infra/compose/docker-compose.yml run --rm --no-deps options-paper \
  python -I -m tradebot.options replay \
  --market /var/lib/tradebot/historical/markt --out /var/lib/tradebot/historical/run --workers 5
```

`historical` vraagt bij IB Gateway (alleen lezen, client-id 18) twee jaar dagbars en IBKR's
dagelijkse 30-daagse IV per onderliggende, en dividenden en earnings bij yfinance. De ketens
worden met QuantLib gereconstrueerd: het IV-niveau is echt, de smile, de spreads en het open
interest komen uit het model (`options.data.historical`). De IV van vóór de startdatum gaat mee
als IV-historie, zodat IVR vanaf dag één werkt. De uitkomst meet de regels op echte koers- en
IV-paden, **niet** echte fills. `--workers 5` bouwt de ketens vooraf parallel: een half jaar
duurt dan ongeveer een half uur tot een uur in plaats van vier uur.

## De paper-dienst tegen IBKR

Vereisten: een IBKR-paper-login (account-id begint met `DU`) en IB Gateway in paper-modus.

### Met Docker

```bash
cp infra/compose/.env.example infra/compose/.env     # TWS_USERID, TWS_PASSWORD, IB_ACCOUNT
docker compose -f infra/compose/docker-compose.yml up -d --build ib-gateway options-paper
docker compose -f infra/compose/docker-compose.yml logs -f options-paper
```

De gateway-container staat vast op `TRADING_MODE: paper`. De dienst verbindt via poort 4004.

### Met systemd

Zie de kop van `infra/systemd/tradebot-options-paper.service`. IB Gateway draait dan op
dezelfde machine in paper-modus op poort 4002.

### Wat de dienst doet

- Elke 15 seconden een ronde; in de sessie elke 5 minuten nieuwe ketens van IBKR.
- Entries alleen op de scantijden (10:00 en 13:30 New York), niet in de eerste 15 en de
  laatste 30 minuten, en niet op een halve dag na 12:30.
- Beheer van open posities elke minuut: winst nemen, stop-loss, sluiten voor expiratie,
  bijsturen of rollen volgens de strategie.
- Elke order gaat eerst door de risicopoort. Fouten die nooit mogen (te grote order, verkeerde
  onderliggende, prijs buiten bid/ask) zijn een `SafetyBreach` en zetten de HALT.
- Na de sluiting: ATM-IV vastleggen, expiraties en assignments verwerken, equity en staat
  bewaren.

## Ingrijpen

Alle bestanden staan in de run-map (`/var/lib/tradebot/options`).

| Wat | Hoe | Effect |
|---|---|---|
| Geen nieuwe entries | `touch HALT` | Beheer en sluiten gaan door |
| Alles sluiten | `touch UNWIND` | Elke positie wordt met voorrang gesloten |
| Hervatten | `rm HALT` of `rm UNWIND` | Volgende ronde weer normaal |
| Stoppen | `docker compose stop options-paper` of `systemctl stop ...` | Open orders geannuleerd, staat bewaard |

Een HALT door de dienst zelf (drawdown, drie fouten op rij, `SafetyBreach`) staat in
`halt/halt.json`. Die gaat pas weg met een expliciete vrijgave via de `HaltStore`
(`release(operator=..., justification=...)`), net als bij het crypto-boek.

## Alarmen en wat te doen

| Alarm | Betekenis | Doen |
|---|---|---|
| `options_halt` | De dienst haltte | Lees `halt/halt.json` en het journal; geef pas vrij als de oorzaak weg is |
| `options_recon_break` | Broker en staat verschillen | Vergelijk posities in TWS met `state.json`; entries op die onderliggende staan stil tot het weer klopt |
| `options_assignment` | Een short leg is assigned | Controleer de aandelenpositie; sluit of laat de strategie verder beheren |
| `options_short_past_expiry` | Een short optie staat na expiratie nog in de staat | Controleer bij IBKR; dit hoort nooit te gebeuren |
| `options_task_*` | Een taak faalde | Lees het journal (`task_error`); drie op rij is HALT |

Assignments bij IBKR komen binnen als positiewijziging. De reconcile ziet het verschil,
blokkeert entries op die onderliggende en alarmeert.

## Configuratie

`conf/options/default.yaml` (schema: `src/tradebot/schemas/options.py`). Belangrijk:

- `account.capital_usd`: 25.000 (fictief). Alle risicogrenzen zijn fracties hiervan.
- `strategies.*.enabled`: CSP, covered call en de wheel staan uit bij 25.000.
- `risk.*`: grenzen op delta, vega, gamma, verlies per onderliggende, buying power,
  scenario-verlies, aantal posities, dagverlies (stop entries 2 %, afbouwen 4 %) en
  drawdown (HALT 10 %).

## Live

Live kan alleen als alle vijf voorwaarden tegelijk waar zijn (`src/tradebot/options/live.py`):
mode `live`, een bevestigingszin in de omgeving, een `LIVE_APPROVED`-bestand met de
account-hash, een echt (geen `DU`) account en een geslaagde paper-killgate. Er is geen CLI-
opdracht voor live, geen config met `mode: live`, en een test bewaakt dat het zo blijft.
