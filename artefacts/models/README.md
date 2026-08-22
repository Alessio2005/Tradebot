# artefact-contract: `artefacts/models/`

> Sluit **D-8** (P0). Aangemaakt in Phase 0.
> Geverifieerd tegen de codebase op 2026-08-22, Phase 0.

## Wat hier staat

Geserialiseerde, **read-only** modelartefacten: gefitte estimators, scalers,
calibrators en hun metadata.

## Contract

| Eis | Regel |
|---|---|
| **Lineage** | elk model draagt `git_sha`, `data_hash`, `config_hash`, `preregistration_id` en `feature_hash`. `registry/lineage.py::verify_lineage` crasht bij elke mismatch |
| **Fit-venster** | een model wordt uitsluitend gefit op data ≤ het einde van zijn trainvenster. Het venster staat in de metadata |
| **Onveranderlijkheid** | modellen worden nooit overschreven. Hertrainen levert een nieuw artefact met een nieuwe `git_sha`/`data_hash`-combinatie |
| **Promotiestatus** | `REGISTERED / TESTED / CANDIDATE / PAPER / CHAMPION`, beheerd door `registry/promotion.py` (Phase 2). Alleen `→ FALSIFIED` is een terugwaartse overgang |
| **Versiebeheer** | DVC-tracked; alleen de `.dvc`-pointer gaat in git |

## Wat hier NIET staat

Features (→ `artefacts/features/`) of P&L-reeksen (→ `artefacts/tracks/`).

## Consument

`train/` schrijft; `live/` en `backtest/` lezen. Een model zonder resolvable
`git_sha` mag niet geladen worden — dat was **D-9**, gesloten in Phase 0.
