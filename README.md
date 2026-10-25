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

## Stand (2026-10-24)

Er is geen strategie die kapitaal verdient. De enige kandidaat is de spot-perp-carry
v8-T2; zijn echte Sharpe is ~1,5 (niet de 5 uit `sqrt(365)`), en de schone toets is de
vooruit-lezing vanaf april 2027. Zie `docs/research/2026-10-24-production-readiness-audit.md`.

Het carryboek draait als shadowboek met dezelfde code als de backtest:

```bash
python -I -m tradebot.data.binance_vision forward YYYY-MM   # vooruit-panelen
python -I -m tradebot.carry init --state state/carry.json   # beginstand = backtest tot de start
python -I -m tradebot.carry step --state state/carry.json   # elke nieuwe gesettelde bar
python -I -m tradebot.systematic.restate                    # de bevroren cijfers, eerlijk uitgedrukt
```

v9 (2026-10-25): H3 op Kraken EU met de kasrente is elk kalenderjaar positief (CAGR 8 %),
maar haalt de 15 % niet en is op de holdout gefalsificeerd; zie
`docs/research/2026-10-25-robust-book-v9-kraken-cash.md`.

Beurs (2026-10-25): Binance is niet beschikbaar in Nederland, Bybit EU heeft geen perps
voor EER-klanten; Kraken EU is de enige beurs die beide benen kan hosten, met beperkingen
(2:1 retail, geen unified margin, alleen BTC/ETH als onderpand). Zie
`docs/research/2026-10-25-venue-kraken-vs-bybit.md`, `conf/execution/venues.yaml` en
`python -I -m tradebot.carry.venues`. De adapter `tradebot.carry.kraken` draait standaard in
dry-run.

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

Op `main` en op een schema draaien daarnaast `nightly-regression.yml` en
`paper-trade-ci.yml`. `research-run.yml` start alleen met de hand (Actions -> Run workflow):
hij haalt onderzoeksdata op waar de ontwikkelomgeving niet bij kan (T-bill, €STR, funding van
Kraken en Binance) en commit het resultaat terug op de branch.

Labels als `DI-24` of `AD-26` in commentaar en commits verwijzen naar het
register van uitgestelde issues en architectuurbesluiten
(`DEFERRED_ISSUES.md`, `ARCHITECTURAL_DECISIONS.md`). Dat register staat niet
in deze repository; het label zelf en de uitleg ernaast in het commentaar zijn
wat er over is.
