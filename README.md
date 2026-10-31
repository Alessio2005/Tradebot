# Tradebot

Onderzoeksstack voor systematische crypto-strategieën: PIT-data, features,
labeling, CPCV en walk-forward, kosten- en risicomodellen, onderzoekspoorten en
een paper-trade-engine. Code staat in `src/tradebot/`, configuratie in `conf/`,
bevroren onderzoeks- en governance-artefacten in `artefacts/`, en de rapporten in
`docs/research/`. Wat er per onderzoeksronde veranderde, staat in `CHANGELOG.md`.

## Installatie

Python 3.13. De omgeving is de lock: installeer uit de lockfiles, niet uit de
ondergrenzen in `pyproject.toml`.

```bash
python3.13 -m venv .venv && . .venv/bin/activate
pip install -r requirements.lock -r requirements-dev.lock
pip install -e . --no-deps
pre-commit install
```

De dvc-CLI voor het manifest `data/pit_store.dvc` is optioneel:
`pip install -e ".[data]"`. De PIT-store zelf staat in git.

## Stand (2026-10-27)

Er is geen strategie die volgens de vooraf vastgelegde regels kapitaal verdient. Zie
`docs/research/2026-10-24-production-readiness-audit.md` voor de audit.

**Beurs.** Binance is niet beschikbaar in Nederland en Bybit EU heeft geen perps voor
EER-klanten. Kraken EU is de enige beurs die beide benen kan hosten, met beperkingen: 2:1
retail, geen unified margin, alleen BTC/ETH als onderpand. Zie
`docs/research/2026-10-25-venue-kraken-vs-bybit.md` en `conf/execution/venues.yaml`.

**Het boek dat op Kraken kan: v9-U1.** Dat zijn de regels van v5-H3 op BTC en ETH, met de
fees van Kraken en de kasrente.
- Over de backtest 2020-04 .. 2026-09 was elk kalenderjaar positief, met 8,0 % CAGR.
- Het boek haalt de 15 % van het mandaat niet, en is op de holdout gefalsificeerd. Zie
  `docs/research/2026-10-25-robust-book-v9-kraken-cash.md`.
- Het draait elke dag als shadow op de echte Kraken-data (`kraken-shadow.yml`).
- Op een server draait het 24/7 als dienst (`carry/service.py`, docker compose of systemd),
  standaard in dry-run. De dienst heeft harde grenzen, een watchdog, alarmen, een
  heartbeat en HALT/UNWIND-noodknoppen.
- Van dry-run naar echt geld: `docs/runbook-kraken-u1.md`.

```bash
docker compose -f infra/compose/docker-compose.yml up -d --build                # de dienst, 24/7
python -I -m tradebot.carry.u1 status --config /etc/tradebot/u1.yaml            # de stand
python -I -m tradebot.carry.u1 shadow --out shadow                              # Kraken-data, geen order
python -I -m tradebot.carry.u1 step --state state/u1-dry.json --capital-usd N   # dry-run
python -I -m tradebot.carry.u1 status --state state/u1-dry.json
python -I -m tradebot.systematic.programme_v9 forward                           # eenmaal, vanaf 2027-04-01
```

Het v8-T2-boek (Binance, unified margin) draait als shadow op Binance-panelen voor zijn
eigen vooruit-lezing. Op Kraken kan het niet.

```bash
python -I -m tradebot.data.binance_vision forward YYYY-MM   # vooruit-panelen
python -I -m tradebot.carry init --state state/carry.json   # beginstand = backtest tot de start
python -I -m tradebot.carry step --state state/carry.json   # elke nieuwe gesettelde bar
python -I -m tradebot.systematic.restate                    # de bevroren cijfers, eerlijk uitgedrukt
```

## Veelgebruikte commando's

```bash
make lint          # ruff check src/ tests/
make type-check    # mypy op schemas/ en utils/
make test-fast     # alles behalve slow en regression
make coverage      # idem, met dekking (fail_under = 100)
make test          # de hele suite
```

## CI

Elke pull request draait:

| Workflow | Wat hij afdwingt |
|---|---|
| `ci.yml` | ruff en mypy; de snelle tests; en apart slow, regression, benchmark en e2e |
| `hygiene.yml` | lock-installatie, harde dependencies, `git_sha`, config-contracten, `tests/unit` |
| `inventory.yml` | omgeving gelijk aan de lock, 100 % dekking, de killgates |
| `research_gates.yml` | PIT-store-integriteit en de blokkerende onderzoekspoorten |
| `security-scan.yml` | bandit en pip-audit tegen de lockfiles |
| `docs.yml` | markdownlint (alleen bij gewijzigde `.md`-bestanden) |

Op `main` en op een schema draaien daarnaast `nightly-regression.yml`,
`paper-trade-ci.yml` en `kraken-shadow.yml`. Die laatste draait het Kraken-boek elke dag
op de echte data: alleen lezen, geen sleutel, geen order.

`docker.yml` bouwt het image van de dienst bij elke wijziging aan infra, code of config.
Daarbij controleert hij:
- de healthcheck;
- de compose-config;
- op `main`: één echte dry-run-ronde in de container, zoals compose hem draait.

`research-run.yml` start alleen met de hand (Actions -> Run workflow). Hij doet het werk
waar de ontwikkelomgeving geen netwerk voor heeft:
- onderzoeksdata ophalen (T-bill, €STR, funding van Kraken en Binance);
- de vooruit-lezing van v9;
- een shadowrun op een willekeurige branch.

Wat de run schrijft, commit hij terug op de branch.

Labels als `DI-24` of `AD-26` in commentaar en commits verwijzen naar het
register van uitgestelde issues en architectuurbesluiten
(`DEFERRED_ISSUES.md`, `ARCHITECTURAL_DECISIONS.md`). Dat register staat niet
in deze repository; het label zelf en de uitleg ernaast in het commentaar zijn
wat er over is.
