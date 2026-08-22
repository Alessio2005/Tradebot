# artefact-contract: `artefacts/tracks/`

> Sluit **D-8** (P0). Aangemaakt in Phase 0.
> Geverifieerd tegen de codebase op 2026-08-22, Phase 0.

## Wat hier staat

Per-asset **P&L-tracks**: de uitkomst van één backtest- of paper-run, als
tijdreeks van signed returns, gevraagde leverage, zijde en funding.

## Contract

| Eis | Regel |
|---|---|
| **Schema** | `schemas/tracks.py::TrackSchema` (Pandera, `strict=True`) |
| **Causaliteit** | een gewicht bepaald op `t` rendeert pas op `t+1` (de `w.shift(1)`-conventie uit `alpha/xs_unit.py`, RETAIN-item sectie 24) |
| **Kosten** | tracks worden **bruto én netto** opgeslagen. Een netto-track zonder gedocumenteerde kostenaanname is invalide |
| **Provenance** | `git_sha`, `data_hash`, `config_hash`, `preregistration_id` en de trial-index `M` |
| **Autoriteit** | alleen tracks uit de event-driven engine (Phase 5) tellen als bewijs voor promotie. Vectorized tracks zijn screening-materiaal (sectie 16.1) |

## Wat hier NIET staat

Aggregaten en rapporten (→ `reports/`), of modelgewichten (→ `artefacts/models/`).

## Consument

`backtest/` schrijft; `validation/` (L11, Phase 2) en `registry/` (L12) lezen.
