# artefact-contract: `artefacts/features/`

> Sluit **D-8** (P0): deze DAG-map werd in `docs/architecture.md` beschreven maar
> bestond niet. Aangemaakt in Phase 0.
> Geverifieerd tegen de codebase op 2026-08-22, Phase 0.

## Wat hier staat

De output van **L1 — Feature Engineering (stateless, causaal)**. Eén artefact per
`(symbol, granularity, feature_set, data_hash)`.

## Contract

| Eis | Regel |
|---|---|
| **Formaat** | Parquet, gepartitioneerd op `(symbol, granularity)` |
| **Tijd** | index in UTC Unix nanoseconden (Phase 1, `utils/time.py`) |
| **Provenance** | elk bestand draagt in zijn sidecar-metadata de `data_hash` van de PIT-partitie waaruit het is afgeleid, plus `git_sha` en `config_hash` |
| **Causaliteit** | elke kolom is uitsluitend een functie van data ≤ `t`. Geen `center=True`, geen fit over de volledige sample |
| **Onveranderlijkheid** | een feature-set wordt nooit ge-edit. Een correctie is een nieuwe `data_hash` en dus een nieuw artefact |
| **Versiebeheer** | DVC-tracked; alleen de `.dvc`-pointer gaat in git |

## Wat hier NIET staat

Modelgewichten (→ `artefacts/models/`), backtest-resultaten (→ `artefacts/tracks/`),
of ruwe marktdata (→ de PIT-store).

## Consument

`src/tradebot/features/pipeline.py` schrijft; L2/L3/L4 lezen. Een lezer die een
artefact zonder geldige `data_hash` aantreft, crasht met `DataContractError`.
