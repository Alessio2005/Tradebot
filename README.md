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
`paper-trade-ci.yml`.

Labels als `DI-24` of `AD-26` in commentaar en commits verwijzen naar het
register van uitgestelde issues en architectuurbesluiten
(`DEFERRED_ISSUES.md`, `ARCHITECTURAL_DECISIONS.md`). Dat register staat niet
in deze repository; het label zelf en de uitleg ernaast in het commentaar zijn
wat er over is.
