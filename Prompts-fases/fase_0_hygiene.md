# MASTER-PROMPT: PHASE 0 — AUDIT & REPOSITORY HYGIENE

> **Fase:** 0 van 7 · **Prioriteit:** P0 (blokkerend voor alle volgende fasen)
> **Bindend brondocument:** `ARCHITECTUUR_AUDIT_2026-08-22.md` — secties 5, 18, 20, 23 (Phase 0), 26
> **Voorwaarde:** Geen enkele fase mag starten voordat de exit criteria van deze fase zijn behaald.

---

## ROL EN CONTEXT

Je acteert als **Principal Software Architect & Quant Platform Engineer** bij een institutionele kwantitatieve handelsdesk. Je bent verantwoordelijk voor de fundering waarop elke latere statistische claim rust: reproduceerbaarheid, dependency-integriteit en cryptografische traceerbaarheid.

Je werkt uitsluitend binnen de kaders van het bindende auditdocument `ARCHITECTUUR_AUDIT_2026-08-22.md`.

**Relevante lagen uit de Target Architecture (sectie 19):**

| Laag | Naam | Rol in deze fase |
|---|---|---|
| **L0** | Data Ingestion, PIT Storage & Validation | Contract-enforcement voorbereiden (geen implementatie — dat is Phase 1) |
| **L12** | Governance, Ledger & Audit Register | `git_sha` moet resolvable worden; MRM-rapporten moeten valide zijn |
| **Cross-cutting** | Alle lagen L0–L13 | Fail-fast contract geldt platform-breed |

**Kritieke bevindingen die deze fase adresseert:**
- **D-9 (P0):** Geen werkende `.git` repository in de hoofdmap → elk MRM-rapport is per definitie invalide.
- **D-10 (P0):** `hmmlearn` ontbreekt in `pyproject.toml` → `risk/hmm_regime.py` valt stilzwijgend terug op een 20/100 EMA-crossover.
- **D-8 (P0):** De DAG-mappen `artefacts/features/`, `models/`, `tracks/` bestaan niet.
- **D-5 (P1):** `REFACTOR_BLUEPRINT_v3.md` wordt gerefereerd maar bestaat niet.
- **Sectie 5.2:** 12 locaties in `src/` met stille degradatie; 5 core modules importeren het niet-bestaande pakket `quant_architect` (`train/ensemble.py`, `train/_scalers.py`, `train/catboost.py`, `tune/objective.py`, `backtest/portfolio.py`).

> **Doctrine (sectie 28):** *"Silent Failures — het stilzwijgend degraderen van statistische modellen naar EMA's — is het grootste operationele risico in de huidige codebase."*

---

## DOEL VAN DE FASE

Herstel versiebeheer, centraliseer configuratie en **elimineer elke vorm van stille degradatie**.

Na deze fase geldt: het systeem crasht luid en onmiddellijk bij elke schending van een dependency-, config- of datacontract. Er bestaat geen enkel codepad meer waarin een complex statistisch model stilzwijgend wordt vervangen door een naïeve benadering. Elke build is bit-reproduceerbaar vanuit `pyproject.toml` + lockfile, en elk artefact is herleidbaar naar een `git_sha`.

---

## CONCRETE DELIVERABLES

1. **`.git` herinitialisatie** — werkende repository in de projectroot, met `.gitignore` die shadow trees uitsluit (`__pycache__/`, `.pytest_cache/`, `.ruff_cache/`, `catboost_info/`, `outputs/`, `logs/`, `*.parquet` buiten DVC).
2. **`src/tradebot/utils/failfast.py`** — expliciete exception-hiërarchie: `TradebotContractError` (basis), `DependencyMissingError`, `DataContractError`, `ConfigContractError`, `CausalityViolationError`. Geen enkele daarvan mag ergens gevangen worden buiten de top-level applicatie-entrypoints.
3. **`pyproject.toml`** — volledige, gepinde dependency-declaratie inclusief `hmmlearn`, `arch`, `statsmodels`, `scipy`, `pandas`, `pyarrow`, `dvc`, `hydra-core`, `pydantic>=2`. Bijbehorende lockfile gecommit.
4. **`src/tradebot/schemas/config.py`** — Pydantic v2 modellen voor alle configuratiedomeinen: `DataConfig`, `VolatilityConfig`, `RiskConfig`, `ExecutionConfig`, `BacktestConfig`, `ValidationConfig`. `model_config = ConfigDict(extra="forbid", frozen=True)`.
5. **`conf/` Hydra composition root** — herstructurering naar `conf/data/`, `conf/model/`, `conf/risk/`, `conf/execution/` conform sectie 20. Elke YAML wordt bij laadtijd gevalideerd tegen zijn Pydantic-schema.
6. **Verwijdering van alle `quant_architect` imports** in de 5 geïdentificeerde modules — vervangen door directe, bestaande modules of door een harde `DependencyMissingError` indien de functionaliteit nog niet bestaat.
7. **`src/tradebot/risk/hmm_regime.py`** — EMA-crossover fallback volledig gesloopt. Ontbreekt `hmmlearn`, dan `raise DependencyMissingError`.
8. **`scripts/audit_fallbacks.py`** — AST-gebaseerde scanner die `src/` doorzoekt op: `try/except ImportError`, bare `except:`, `except Exception` zonder re-raise, en `warnings.warn` gevolgd door een degraded return. Exit code 1 bij elke hit.
9. **`tests/unit/test_no_silent_fallbacks.py`** — pytest-wrapper rond de scanner; faalt de build bij elke nieuwe stille fallback.
10. **`tests/unit/test_config_contracts.py`** — bewijst dat elke YAML in `conf/` laadt onder zijn Pydantic-schema en dat een onbekende sleutel een `ConfigContractError` triggert.
11. **DAG-mappen aangemaakt** (D-8): `artefacts/features/`, `artefacts/models/`, `artefacts/tracks/`, elk met een `.gitkeep` en een `README.md` die het artefact-contract beschrijft.
12. **`docs/DEPENDENCY_CONTRACT.md`** — expliciete lijst van harde afhankelijkheden per laag (L0–L13), met per dependency het gevolg van afwezigheid (altijd: crash, nooit: fallback).
13. **`docs/ARCHITECTURE.md` gecorrigeerd** — verwijzingen naar niet-bestaande documenten (D-5) verwijderd of vervangen door verwijzingen naar dit auditdocument.
14. **CI-workflow `.github/workflows/hygiene.yml`** — draait `audit_fallbacks`, `ruff`, `mypy --strict` op `src/tradebot/schemas/` en `pytest tests/unit`.

---

## STAPSGEWIJZE UITVOERING

**Stap 1 — Nulmeting vastleggen.**
Inventariseer de repository voordat je iets wijzigt. Genereer `reports/phase0_baseline.md` met: aantal bestanden per module, LOC-telling per bestand (markeer alles >800 LOC i.v.m. D-6), en de volledige lijst van bestanden die `quant_architect` importeren. Deze nulmeting is bewijsmateriaal, geen wegwerpartefact.

**Stap 2 — Versiebeheer herstellen.**
Initialiseer `.git`, schrijf de `.gitignore`, en maak één initiële commit die de volledige huidige staat vastlegt (`chore(repo): re-initialise version control with baseline snapshot`). Verifieer daarna dat `registry/lineage.py` een niet-lege `git_sha` teruggeeft. Zolang dit faalt, is D-9 niet opgelost.

**Stap 3 — Fail-fast fundament bouwen.**
Schrijf `src/tradebot/utils/failfast.py`. Definieer de exception-hiërarchie en een helper `require(condition, message, exc_type)` die geen boolean teruggeeft maar direct raiset. Schrijf de unit tests voordat je de exceptions ergens toepast.

**Stap 4 — Fallback-scanner bouwen en de 12 locaties in kaart brengen.**
Implementeer `scripts/audit_fallbacks.py` met Python's `ast`-module (geen regex — regex mist genest gedrag). Draai hem en produceer `reports/phase0_fallback_register.md`: per hit het bestand, regelnummer, het model dat gedegradeerd wordt, en het naïeve alternatief waarnaar wordt teruggevallen. Verwacht minimaal de 12 in sectie 5.2 genoemde locaties.

**Stap 5 — `quant_architect` chirurgisch verwijderen.**
Behandel de 5 modules één voor één, elk met een eigen atomaire commit. Per module: bepaal wat de import feitelijk zou moeten leveren, vervang door een bestaande interne module, of — als de functionaliteit nog niet bestaat — vervang de import door een expliciete `raise DependencyMissingError` op het aanroeppunt. **Verboden:** een lege stub die `None` of een identity-transform teruggeeft; dat is dezelfde stille degradatie in een nieuw jasje.

**Stap 6 — `hmm_regime.py` saneren.**
Verwijder het `try/except ImportError`-blok en de EMA-crossover-tak volledig. Voeg `hmmlearn` toe aan `pyproject.toml`. De module mag in deze fase nog niet herontworpen worden naar het M2 Filtered HMM-model — dat is expliciet Phase 6. In deze fase geldt alleen: geen stille fallback meer.

**Stap 7 — Dependency-contract sluiten.**
Vul `pyproject.toml` aan met alle harde afhankelijkheden, genereer de lockfile, en bouw de omgeving vanaf nul opnieuw op in een schone virtual environment. Documenteer het resultaat in `docs/DEPENDENCY_CONTRACT.md`.

**Stap 8 — Configuratie centraliseren.**
Schrijf de Pydantic-schema's, herstructureer `conf/` naar de indeling uit sectie 20, en vervang elke `os.getenv(...)`, elke hardcoded drempelwaarde en elk magisch getal in `src/` door een configuratieveld. Grep expliciet op numerieke literals in `risk/`, `volatility/` en `execution/` — dat zijn de plekken waar drempels historisch hardcoded raken.

**Stap 9 — Shadow trees en DAG-mappen.**
Verwijder de shadow trees uit versiebeheer. Maak de ontbrekende DAG-mappen aan met hun artefact-contract README's (D-8).

**Stap 10 — Documentatiedrift dichten.**
Corrigeer D-5 en D-8 in `docs/`. Voeg aan elk gecorrigeerd document een regel toe: *"Geverifieerd tegen de codebase op [datum], Phase 0."*

**Stap 11 — CI dichttimmeren.**
Schrijf `.github/workflows/hygiene.yml`. De workflow moet falen — niet waarschuwen — bij elke fallback-hit, elke schema-schending en elke lint-fout.

**Stap 12 — Exit-rapport.**
Genereer `reports/phase0_exit_report.md` met per exit-criterium het bewijs (commando, output, commit-sha).

---

## CRITERIA & VALIDATIE (EXIT CRITERIA)

Deze fase is **uitsluitend** afgerond wanneer elk van onderstaande punten aantoonbaar waar is:

1. **Nul stille fallbacks.** `python scripts/audit_fallbacks.py --strict` retourneert exit code 0 over de volledige `src/`-boom. Nul `try/except ImportError`. Nul bare excepts. Nul `except Exception` zonder re-raise.
2. **100% reproduceerbare build.** Een schone virtual environment, opgebouwd uit uitsluitend `pyproject.toml` + lockfile, draait de volledige testsuite zonder een enkele handmatige installatie.
3. **Valide `git_sha`.** `registry/lineage.py` levert een niet-lege, in de repository resolvable commit-hash. Een MRM-rapport zonder valide `git_sha` moet crashen, niet loggen (D-9 gesloten).
4. **`quant_architect` bestaat nergens meer.** `grep -rn "quant_architect" src/` levert nul treffers in import-statements.
5. **Config-integriteit.** Elke YAML in `conf/` valideert tegen zijn Pydantic-schema; een geïnjecteerde onbekende sleutel triggert aantoonbaar een `ConfigContractError`.
6. **Nul hardcoded parameters.** Geen enkele drempelwaarde, lookback-window, λ-parameter of limiet staat als literal in `src/`. Alles komt uit `conf/`.
7. **CI blokkeert.** Een bewust geïntroduceerde stille fallback in een testbranch laat de `hygiene`-workflow aantoonbaar falen.
8. **DAG-mappen bestaan** met hun artefact-contract (D-8 gesloten).

> **Harde regel:** Phase 1 mag niet starten zolang één van deze acht punten open staat. Er is geen gedeeltelijke afronding.

---

## REGELS & HANDLINGSINSTRUCTIES

- **Fail-fast compliance is absoluut.** Elke module bevat nul `try/except` fallbacks en crasht onmiddellijk bij schending van een contract (Acceptance Criterion 5, sectie 26). Een `except` die logt en doorgaat is een overtreding. Een `except` die een default teruggeeft is een ernstiger overtreding.
- **100% PIT rigor.** Ook in een hygiëne-fase: raak geen enkele datapipeline aan op een manier die causaliteit kan schenden. Bij twijfel: niet aanraken, documenteren, doorschuiven naar Phase 1.
- **Geen hardcoded variabelen.** Elke parameter komt uit `conf/` via een gevalideerd Pydantic-schema. Een magisch getal in productiecode is een build-breaker.
- **Atomaire commits.** Eén logische wijziging per commit, conventional commits: `feat(config): add pydantic schema enforcement for risk domain`, `fix(risk): remove silent EMA fallback in hmm_regime`, `chore(repo): re-initialise git and add gitignore`, `test(hygiene): add AST-based fallback scanner`. Nooit meerdere modules in één commit.
- **Geen scope creep.** Deze fase herstelt de fundering. Je implementeert geen nieuwe modellen, geen nieuwe alpha, geen HMM-herontwerp. Wat niet in de deliverables staat, hoort in een latere fase.
- **Documenteer het niet-oplosbare.** Kom je een probleem tegen dat buiten deze fase valt, schrijf het weg in `docs/DEFERRED_ISSUES.md` met fase-toewijzing. Niets verdwijnt stilzwijgend.
- **Bewijs boven bewering.** Elke claim in het exit-rapport is gekoppeld aan een reproduceerbaar commando en zijn output.

---

## STARTINSTRUCTIE

> **Begin nu met Stap 1:** inventariseer de volledige repository en genereer `reports/phase0_baseline.md` met de LOC-telling per module, de lijst van bestanden >800 LOC (D-6), en de volledige lijst van bestanden die `quant_architect` importeren. Initialiseer daarna `.git` en leg deze nulmeting vast in één commit `chore(repo): re-initialise version control with baseline snapshot`.
