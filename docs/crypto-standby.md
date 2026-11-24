# Crypto in standby (sinds 2026-11-24)

Op 2026-11-24 besloot de eigenaar dat aandelenopties de crypto-strategieën vervangen als
actieve setup van Tradebot. De crypto-stack wordt niet verwijderd: hij staat in **standby**.
Ontwerp: `docs/research/2026-11-24-options-pivot-design.md`.

Standby betekent:

- de code, configs, tests en workflows staan er nog;
- de tests draaien in elke suite mee en de dekking blijft 100 % over het geheel;
- niets ervan draait standaard: geen dienst, geen geplande run;
- met de stappen onderaan staat het weer aan.

Op het moment van de standby draaide er niets en stond er geen geld uit. Er is dus niets
afgebouwd.

## Wat er in standby staat

| Onderdeel | Waar |
|---|---|
| Het Kraken-boek v9-U1 (dienst, adapter, CLI) | `src/tradebot/carry/`, `conf/live/u1.example.yaml`, `docs/runbook-kraken-u1.md` |
| De onderzoeksprogramma's v1-v9 | `src/tradebot/systematic/` |
| Het ML-pad | `src/tradebot/train/`, `tune/`, `labeling/`, `cv/`, `selection/`, `featurestore/` |
| Data van de beurzen | `src/tradebot/data/kraken_feed.py`, `src/tradebot/data/binance_vision.py` |
| Boekconfigs | `conf/model/robust_book_v*.yaml`, `conf/symbols/*USDT.yaml` |
| De U1-dienst | `infra/compose/docker-compose.yml` (profiel `crypto`), `infra/systemd/tradebot-u1.service` |
| Het carry-shadowboek | `infra/k8s/cron-carry-shadow.yaml` (`suspend: true`) |
| De dagelijkse shadow | `.github/workflows/kraken-shadow.yml` (alleen `workflow_dispatch`) |

`.github/workflows/paper-trade-ci.yml` blijft gewoon draaien: de code bestaat nog en moet
blijven werken.

## De laatste stand

Geen nieuwe cijfers; de rapporten van vóór de standby gelden:

- `docs/research/2026-10-25-robust-book-v9-kraken-cash.md`: v9-U1 op Kraken, 8,0 % CAGR over
  de backtest 2020-04 .. 2026-09, haalt de 15 % van het mandaat niet en is op de holdout
  gefalsificeerd.
- `docs/research/2026-10-25-venue-kraken-vs-bybit.md`: waarom Kraken.
- `docs/research/2026-10-24-production-readiness-audit.md`: de audit.

De U1-config staat op `mode: dry_run` en blijft zo.

## Weer aanzetten

Elke stap staat los; zet alleen aan wat nodig is.

1. **De U1-dienst met docker compose:**
   `docker compose -f infra/compose/docker-compose.yml --profile crypto up -d --build u1`.
   Of zet het profiel vast in `.env` naast compose: `COMPOSE_PROFILES=crypto`.
2. **De U1-dienst met systemd:** `systemctl enable --now tradebot-u1`. Uitzetten was
   `systemctl disable --now tradebot-u1`.
3. **Het carry-shadowboek op k8s:** `suspend: false` in `infra/k8s/cron-carry-shadow.yaml`, of
   `kubectl -n tradebot patch cronjob carry-shadow -p '{"spec":{"suspend":false}}'`.
4. **De dagelijkse shadow op GitHub:** zet het `schedule`-blok uit het commentaar in
   `.github/workflows/kraken-shadow.yml` terug onder `on:`. Geplande workflows draaien alleen
   op de standaardbranch.

Daarna geldt `docs/runbook-kraken-u1.md` weer zoals het er staat.

## Wat kan zonder iets aan te zetten

- Een shadow-run met de hand: Actions → `kraken-shadow` → Run workflow, of lokaal
  `python -I -m tradebot.carry.u1 shadow --out shadow`.
- De eenmalige vooruit-lezing van v9 (`python -I -m tradebot.systematic.programme_v9 forward`,
  via `research-run`, taak `v9-forward`) kan vanaf 2027-04-01. Die staat niet ingepland.
