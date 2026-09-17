# KETEN A — STATUS

> **Verdict: `CLOSED — premisse niet aangetroffen`.**
> **Geverifieerd tegen de codebase op 2026-09-17**, fase 10 stap 14, op commit
> `21158dd`.
> **Trialrekening: 0.** Er is niets gepromoveerd, niets gedraaid en niets
> gekozen, want er is niets om te promoveren.

## Waarom dit document bestaat en wat het vaststelt

Stap 14 van de fase-10-masterprompt draagt op om "keten A" te sluiten, en
beschrijft haar zo:

> *"`docs/CHAIN_A_STATUS.md` beschrijft een dode keten: `alpha/factor_alpha.py::momentum_alpha`
> bestaat, maar geen actieve configuratie roept hem aan."*

Die zin bevat drie feitelijke beweringen. **Alle drie zijn onjuist**, en dat is
met de repository-geschiedenis vastgesteld en niet met een lezing van de code
van vandaag:

| Bewering | Status | Bewijs |
|---|---|---|
| `docs/CHAIN_A_STATUS.md` bestaat en beschrijft een dode keten | **onjuist** | `git log --all --diff-filter=A -- '*CHAIN*'` geeft nul treffers: het bestand is nooit in enige branch toegevoegd |
| `momentum_alpha` bestaat in `alpha/factor_alpha.py` | **onjuist** | `git log --all -S'momentum_alpha'` geeft precies één commit: `715d30a`, de commit die de masterprompt zélf schreef. De naam heeft nooit in de code gestaan |
| er is een "keten A" | **onjuist** | `grep -rn "keten A\|chain A"` over de hele repository treft uitsluitend de masterprompt zelf (regels 369, 2922, 2928–2933) |

`src/tradebot/alpha/factor_alpha.py` bestaat wél, maar is iets anders: een
HAC/Newey-West-regressie die residuele alfa meet (`factor_residual_alpha`,
`FactorAlphaResult`, `G4_FACTORSETS`). Hij bevat geen momentumsignaal.

**Dit document is daarom geen sluiting van een keten, maar de vastlegging dat de
keten niet bestaat.** Dat is een uitkomst onder R-10: de prompt formuleerde een
verwachting over de codebase, de meting weerlegt haar, en dan is de meting het
antwoord.

## Route (a) of route (b)

Geen van beide is van toepassing. Route (a) vraagt om keten A "éénmalig
diagnostisch te draaien met de bestaande, ongewijzigde parameters"; er is geen
keten om te draaien. Route (b) vraagt om promotie met 2.400 geërfde
Optuna-trials in de deflatie; er is geen kandidaat om te promoveren.

De keuze is daarmee de derde: **sluiten, met nul trials.** Dat is geen omzeiling
van de trialrekening maar haar enige geldige invulling — een trial is de prijs
van een keuze, en hier is geen keuze gemaakt.

### De 2.400 Optuna-trials bestaan wél, maar horen ergens anders

De prompt rekent met "2.400 geërfde Optuna-trials (200 × 6 × 2)". Dat getal
heeft een echte grondslag, alleen niet deze:

- `optuna==4.9.0` staat in `requirements.lock`;
- `src/tradebot/tune/objective.py:758` leest `cfg.training.optuna_trials` met
  **fallback 200**;
- `src/tradebot/tune/` is bereikbaar via `apps/tune_hparams.py` en `dvc.yaml`.

De 200 × 6 is dus het hyperparameterbudget van het **boosting-tuningsysteem**
over zes symbolen, niet van een alfaketen. Er staat geen Optuna-study op schijf
en geen enkele module importeert `optuna` buiten `tune/`. Wie dat budget ooit
in een deflatie wil meenemen, telt het bij `tune/`, niet hier.

## Wat er wél dood is — de meting die stap 14 bedoelde

De generieke vraag achter stap 14 is wél beantwoordbaar: *bestaat er
alfacode die geen enkele actieve configuratie aanroept?* Ja, veertien modules.
Gemeten op `21158dd` met een referentietelling over `conf/`, `apps/` en
`dvc.yaml`, en apart over `src/` en `tests/`:

| Module in `alpha/` | in `__init__` | refs in `src/` | refs in `tests/` | refs in `conf`+`apps`+`dvc` |
|---|---:|---:|---:|---:|
| `csm_volume_clock` | ja | 1 | **0** | **0** |
| `kalman_ou` | ja | 1 | **0** | **0** |
| `research_harness` | ja | 1 | **0** | **0** |
| `mean_reversion` | ja | 2 | **0** | **0** |
| `factor_alpha` | ja | 1 | 3 | **0** |
| `macro_regime` | ja | 1 | 1 | **0** |
| `cm_carry` | nee | 0 | 2 | **0** |
| `cm_tsmom` | nee | 0 | 4 | **0** |
| `eq_overnight` | nee | 0 | 1 | **0** |
| `eq_pead` | nee | 0 | 1 | **0** |
| `eq_quality` | nee | 0 | 1 | **0** |
| `fx_carry` | nee | 0 | 2 | **0** |
| `fx_tsmom` | nee | 0 | 2 | **0** |
| `xs_unit` | nee | 10 | 3 | **0** |

De eerste vier regels zijn het scherpst: aangesloten op `alpha/__init__.py`,
bereikbaar vanuit `src/`, en door **geen test en geen configuratie** aangeroepen.
Dat is de vorm die de prompt beschreef — alleen dertien keer, onder andere namen,
en nooit als één keten.

**Deze tabel is een inventarisatie, geen verdict.** De verdicts vallen in stap 15,
op grond van het meetdomein (AD-23), en daar horen ze ook: negen van de veertien
modules dragen een `cm_`-, `eq_`- of `fx_`-prefix en meten dus commodities,
aandelen of valuta — vermogenstitels die in `conf/governance/measurement_domain.yaml`
niet voorkomen. Hun dood is geen toeval van bedrading maar een gevolg van het
mandaat.

## Gevolgen voor de deliverables

| Deliverable | Status |
|---|---|
| D19 — `reports/phase10_chain_a_score.md`, "keten A gescoord, niet-promoveerbaar" | **niet produceerbaar.** Er valt niets te scoren. Dit document vervangt hem |
| Stap 14.2 — diagnostische run op de ontwikkelsample | **vervallen.** Geen keten om te draaien |
| Stap 14.3 — "verwijder de dode configuratiepaden" | **uitgevoerd, maar ergens anders dan verwacht.** Zie hieronder |

## Het dode configuratiepad dat er wél was

Stap 14.3 draagt op de dode configuratiepaden te verwijderen. Rond keten A zijn
die er niet — de veertien modules hierboven worden juist door géén configuratie
genoemd, dus er is niets dat naar hen wijst. Maar de vraag zelf is wel
beantwoordbaar, en één keer luidt het antwoord ja.

Gemeten over alle 101 letterlijke paden in `dvc.yaml` en alle Hydra-targets in
`conf/`: **één pad wees naar iets dat niet bestaat.**

```text
dvc.yaml:57   deps: - src/tradebot/data/ingestion.py     <- bestaat niet
```

`src/tradebot/data/ingestion` is in commit `be94079` van een MODULE een PAKKET
geworden (`ingestion/__init__.py`, `bybit.py`, `contract.py`, `crypto_sources.py`,
`legacy.py`), maar `dvc.yaml` bleef naar het oude `.py`-pad wijzen. De import in
`apps/build_features.py:117` werkt gewoon — die noemt het pakket, niet het
bestand — en `tests/test_imports.py:51` dekt hem af, dus niets in de testsuite
kon dit zien.

DVC wél. Vóór de reparatie:

```text
ERROR: failed to reproduce 'build_features@ETHUSDT':
[Errno 2] No such file or directory: '.../src/tradebot/data/ingestion.py'
```

Daarmee was **stage 1 van de pijplijn onreproduceerbaar** — de stage die bars,
features en events bouwt. De reparatie is één regel: het pad wijst nu naar de
pakketmap. Na de reparatie lost `dvc repro --dry build_features` op en draait
door naar de commando's.

> Dit is dezelfde maskering als in de CI-poorten van deze branch: de testsuite
> keek naar de import (die werkte), en de enige poort die naar het PAD keek werd
> nooit gedraaid. Een defect dat twee onafhankelijke controles overleeft omdat
> geen van beide de vraag stelde die het zou hebben gevangen.

## Wat dit NIET vaststelt

- Het zegt niets over of de veertien modules moeten verdwijnen. Dat is stap 15.
- Het zegt niets over de kwaliteit van `factor_alpha.py`. Die module is
  domein-conform, getest (3 testbestanden) en blijft ongemoeid.
- Het herroept geen enkele regel uit `docs/FALSIFICATION_REGISTER.md`. `cm_carry`
  (F20) en `cm_tsmom` (W28-ruling) zijn daar al gearchiveerd; deze inventarisatie
  telt ze mee als dode bedrading, niet als een nieuw oordeel.
