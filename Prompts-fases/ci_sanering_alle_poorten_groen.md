# MASTER-PROMPT: CI-SANERING — ELKE POORT GROEN, GEEN POORT VERZWAKT

> **Branch:** `claude/phase-10-sub-agent-prompt-4oylvb` @ `371755e` · **Prioriteit:** P0 · **Type:** infrastructuur, geen onderzoek
> **Nulmeting:** 2026-09-18, op de referentie-interpreter uit `docs/runbook.md` §0. Elk cijfer in §2 is die dag gemeten, niet uit een eerder rapport overgenomen.
> **Bindende brondocumenten:** `docs/runbook.md` §0, `docs/PROJECT_STATE.md`, `docs/DEFERRED_ISSUES.md` (DI-3, DI-4, DI-8, DI-16, DI-17), `reports/phase9_coverage_baseline.md`, `docs/ARCHITECTURAL_DECISIONS.md`
> **Voor uitvoerders:** stappen gebruiken checkbox-syntaxis (`- [ ]`). Elke stap eindigt in één commit met een meting in de body. Werk in volgorde: stap 0 blokkeert alles.

---

## 0. HET PRINCIPE DAT ALLES STUURT

Deze repository draait om één regel, en die staat in `docs/PROJECT_STATE.md` §1:

> **Een resultaat telt pas wanneer het de poort is gepasseerd die het had kunnen tegenhouden.**

Daaruit volgt de enige manier waarop deze opdracht mag worden uitgevoerd: **groen wordt verdiend door de code te repareren, nooit door de poort te verzwakken.** Een groene CI die is ontstaan door een drempel te verlagen, is schadelijker dan een rode CI, want zij liegt.

### 0.1 Verboden — dit is geen richtlijn maar een harde grens

Je mag NIET, in geen enkele stap, om welke reden dan ook:

| # | Verboden | Waarom |
|---|---|---|
| V1 | `fail_under` verlagen in `pyproject.toml` | `reports/phase9_coverage_baseline.md` sluit dat expliciet uit; de fase die de drempel zette, sloot het ook al uit |
| V2 | Een regel toevoegen aan `[tool.ruff.lint] ignore` of aan `per-file-ignores` | Elke bestaande onderdrukking daar is stuk voor stuk beoordeeld en gemotiveerd; een nieuwe zonder dat oordeel is ruis in een lijst die betekenis heeft |
| V3 | `# type: ignore`, `# noqa` of `--ignore-errors` als bulkmiddel | Eén `# type: ignore` met een reden erachter mag; een reeks ervan om een teller naar nul te brengen niet |
| V4 | Een test verwijderen, skippen, `xfail`-en of hernoemen om hem groen te krijgen | De vier `xfail(strict=True)`-killgates zijn pre-geregistreerd; `inventory.yml` toetst hun verzameling op identiteit |
| V5 | `continue-on-error: true`, `\|\| true`, of een stap uit een workflow halen | Dit is precies wat §2.4 als defect aanmerkt; het toevoegen ervan zou het defect vermenigvuldigen |
| V6 | Een ratchet-budget verhogen (`check_hardcoded_params.py`, `check_file_size.py`, `check_banned_methods.py`) | Een ratchet die omhoog mag, is geen ratchet |
| V7 | Een gemeten getal uit §2 overschrijven zonder het opnieuw te meten | §2 is een nulmeting; die weerleg je met een meting, niet met een aanname |
| V8 | Een mandaat- of databesluit nemen dat aan de eigenaar is (§4) | Zie DI-15, DI-18, DI-21: dataposten zijn inkoopbesluiten |

### 0.2 Wat je wél doet met iets dat niet groen te krijgen is

Je repareert het niet stiekem en je laat het niet weg. Je schrijft het op als **DI-regel** in `docs/DEFERRED_ISSUES.md`, in het bestaande format (ID, gevonden in, probleem, waarom niet nu, toegewezen aan), en je rapporteert de stap als **rood met reden**. Een eerlijk rode poort met een geregistreerde motivering is een geldig eindresultaat van deze opdracht. Een groen gemaakte poort zonder die motivering is dat niet.

---

## 1. DE REFERENTIE-OMGEVING — DOE DIT VOORDAT JE IETS MEET

Een deel van de "fouten" in deze repository bestaat niet. Het zijn artefacten van een verkeerde interpreter. Dat is tijdens de nulmeting van 2026-09-18 daadwerkelijk gebeurd en het kostte drie meetronden voordat het opviel.

### 1.1 De enige interpreter waarop een meting geldig is

```
D:/venv/tradebot/Scripts/python.exe      Python 3.13.0
```

Dit staat in `docs/runbook.md` §0 en het is bindend. Verifieer vóór je eerste meting:

```bash
D:/venv/tradebot/Scripts/python -c "import tradebot; print(tradebot.__file__)"
```

Moet `D:\Tradebot\src\tradebot\__init__.py` geven — **nooit** een `C:`-pad. Geeft het een `C:`-pad, dan meet je een andere codebase dan de repository en is alles wat volgt ongeldig.

### 1.2 Val 1 — `scripts.*` is niet importeerbaar op de globale interpreter

Op de globale `C:\...\Python313` faalt de testcollectie met drie fouten:

```
ModuleNotFoundError: No module named 'scripts.check_file_size'
ModuleNotFoundError: No module named 'scripts.reachability_map'
ModuleNotFoundError: No module named 'scripts.clean'
```

**Dit is geen defect in de repository.** De oorzaak is precies aanwijsbaar: `site-packages/scripts/__init__.py` — meegeleverd door `fitz 0.0.1.dev2` en `google-auth-oauthlib 1.2.1` — is een **reguliere** package, en `D:\Tradebot\scripts\` is een **namespace**-package (geen `__init__.py`). Een reguliere package wint altijd van een namespace-package, ongeacht de volgorde van `sys.path`. De `sys.path.insert(0, ...)` in `tests/conftest.py` helpt daar dus niet tegen.

In de referentie-venv staat geen concurrerende `scripts` en resolveert hij naar `_NamespacePath(['D:\\Tradebot\\scripts', ...])`. Daar is de collectie schoon.

Diagnose in één regel:

```bash
D:/venv/tradebot/Scripts/python -c "import scripts; print(getattr(scripts,'__path__',None) or scripts.__file__)"
```

### 1.3 Val 2 — markdownlint telt bestanden die niet in de repository zitten

`npx markdownlint-cli2@0.13.0 "**/*.md"` geeft lokaal **59** bevindingen (52× MD007, 2× MD050, 2× MD029, 2× MD014, 1× MD038). Alle 59 staan in `.superpowers/`, dat niet getrackt is. Op een verse checkout ziet `docs.yml` er **nul**. Meet Markdown daarom altijd over `git ls-files '*.md'`, nooit over de glob.

### 1.4 Val 3 — gereedschapsversies verschillen per interpreter

| | globale Python 3.13 | referentie-venv (lockfiles) |
|---|---|---|
| `mypy` | 1.17.0 | **2.3.1** |
| `pytest` | 9.0.2 | **9.1.1** |
| `ruff` | 0.15.12 | 0.15.12 |

Mypy 1.17 en 2.3.1 geven een andere foutverzameling op identieke broncode (gemeten: 188 tegen 187, met een andere verdeling over codes). Dit is dezelfde klasse als **DI-16** en het is de reden dat stap 5 bestaat.

---

## 2. NULMETING — gemeten 2026-09-18 op `371755e`, referentie-venv

### 2.1 Elke workflow, elke job, elke stap

| Workflow | Job / stap | Status | Gemeten |
|---|---|---|---|
| `ci.yml` | lint → `ruff check src/ apps/ tests/` | **GROEN** | `All checks passed!` |
| `ci.yml` | lint → `mypy schemas/ utils/ apps/` | **ROOD** | **187 fouten in 34 bestanden** (64 gecontroleerd) |
| `ci.yml` | `loc-guard` | **ROOD** | `FAIL: src/tradebot/backtest/evaluation.py has 1057 lines`, exit 1 |
| `ci.yml` | `test-fast` (pytest + `--cov`) | **ROOD** | 1 failure + dekkingsdrempel; exit 1 |
| `hygiene.yml` | dependency-contract (21 harde deps) | GROEN | alle 21 aanwezig |
| `hygiene.yml` | `git_sha` resolvable | GROEN | `371755e`, `git cat-file -e` OK |
| `hygiene.yml` | `audit_fallbacks.py --strict` | GROEN | exit 0; 39 bevindingen, **0 blokkerend** |
| `hygiene.yml` | `check_hardcoded_params.py --strict` | GROEN | exit 0; 298 literals / 98 bestanden, ratchet 313 |
| `hygiene.yml` | geen `quant_architect`-imports | GROEN | 0 treffers |
| `hygiene.yml` | DAG-mappen + `README.md` | GROEN | 3/3 aanwezig |
| `hygiene.yml` | `ruff` (src + 5 bestanden) | GROEN | `All checks passed!` |
| `hygiene.yml` | `mypy --strict config.py failfast.py` | GROEN | `Success: no issues found in 2 source files` |
| `hygiene.yml` | `pytest tests/unit -q` | **ROOD** | bevat de failure uit §2.2 D4 |
| `inventory.yml` | omgeving == beide lockfiles | GROEN | **0 afwijkingen** |
| `inventory.yml` | `reachability_map.py --strict` | GROEN | exit 0; 3 geregistreerd onbereikbaar |
| `inventory.yml` | `check_file_size.py` | GROEN | exit 0; 10 gecapte bestanden, geen boven zijn cap |
| `inventory.yml` | `check_hardcoded_params.py --strict` | GROEN | exit 0 |
| `inventory.yml` | `audit_fallbacks.py --strict` | GROEN | exit 0 |
| `inventory.yml` | `check_banned_methods.py --strict` | GROEN | exit 0; 0 treffers / 366 bestanden |
| `inventory.yml` | `check_domain_consistency.py --strict` | GROEN | exit 0; 0 openstaande regels |
| `inventory.yml` | `ruff check src/` | GROEN | `All checks passed!` |
| `inventory.yml` | dekking tegen `fail_under=70` | **ROOD** | **59,10 %**; rood **bij ontwerp**, gedocumenteerd |
| `inventory.yml` | de vier killgates staan XFAILED | GROEN | verzameling identiek, 0 XPASS, 0 FAILED |
| `research_gates.yml` | `apps/run_gates.py` (4 poorten) | GROEN | exit 0; `banned_methods`, `lookahead_suite` (646 passed), `promotion_gates` (111), `gate_killgate` (18) |
| `research_gates.yml` | `test_pit_store_integrity.py` | GROEN | draait mee in de suite |
| `docs.yml` | markdownlint-cli2 0.13.0 | GROEN | 0 bevindingen op getrackte `.md` (59 lokaal, alle in ongetrackt `.superpowers/`) |
| `security-scan.yml` | `bandit -r src/ apps/ -ll` | **ONGEMETEN** | bandit staat in geen van beide lockfiles |
| `security-scan.yml` | `pip-audit` | **MEET NIETS** | zie §2.4 |
| `nightly-regression.yml` | `regenerate_baseline.py` | **MEET NIETS** | zie §2.4 |
| `nightly-regression.yml` | slow + regression tests | ONGEMETEN | buiten de snelle selectie |
| `paper-trade-ci.yml` | 7-daagse paper-trade smoke | ONGEMETEN | vereist `dvc repro` + artefactcache |

### 2.2 De vier echte defecten

**D1 — mypy: 187 fouten in 34 bestanden.** Dit is de `lint`-job van `ci.yml`, en daarmee blokkeert het `test-fast` (die `needs: lint` heeft). Verdeling over foutcodes:

| code | n | | code | n |
|---|---|---|---|---|
| `type-arg` | 92 | | `no-any-return` | 8 |
| `misc` | 24 | | `dict-item` | 5 |
| `no-untyped-def` | 18 | | `union-attr` | 4 |
| `call-arg` | 10 | | `assignment` | 4 |
| `arg-type` | 9 | | `unused-ignore` | 1 |
| `no-untyped-call` | 8 | | `operator` | 1 |

Zwaartepunt: `apps/train_cpcv.py` (32), `apps/feature_selection.py` (15), `src/tradebot/schemas/bars.py` (14), `src/tradebot/schemas/events.py` (12), `apps/paper_multi_sleeve.py` (11), `apps/generate_mrm_report.py` (10), `schemas/labels.py` (8), `schemas/features.py` (8), `apps/paper_neutral_trader.py` (7), `apps/live_paper_trader.py` (7).

Bijna de helft (`type-arg`, 92×) is één klasse: een generiek type zonder parameters (`dict`, `tuple`, `Queue`). Dat is mechanisch en risicoloos. De `call-arg`- en `arg-type`-fouten zijn dat **niet** — daar zit mogelijk een echte bug onder. Voorbeeld uit de meting: `apps/live_paper_trader.py:329: Missing positional argument "constraints" in call to "PortfolioControllerConfig"`. Behandel elke `call-arg` en `arg-type` als kandidaat-defect, niet als typeruis.

Mypy meldt daarnaast `pyproject.toml: note: unused section(s): module = ['ccxt.*']` — een dode override.

Waarom `apps/` zo zwaar weegt: `[[tool.mypy.overrides]] module = ["tradebot.schemas.*", "tradebot.utils.arrays", "apps.*"]` zet `strict = true` op precies die scope. Dat is een bewuste keuze. **`apps.*` uit die lijst halen is geen reparatie maar V2 in een andere vorm.**

**D2 — `loc-guard` in `ci.yml` is rood, en de poort zelf is verouderd.** Tien bestanden in `src/` staan boven 800 regels:

| regels | bestand | in de `ci.yml`-whitelist? |
|---|---|---|
| 1226 | `schemas/config.py` | nee |
| 1185 | `labeling/meta.py` | ja |
| 1162 | `validation/inference.py` | nee |
| 1117 | `train/ensemble.py` | ja |
| 1060 | `features/regime.py` | nee |
| 1057 | `backtest/evaluation.py` | nee |
| 984 | `validation/data_adequacy.py` | nee |
| 958 | `tune/objective.py` | ja |
| 917 | `live/engine.py` | nee |
| 824 | `portfolio/legacy_sizing.py` | nee |

De whitelist in de workflow noemt bovendien `risk/portfolio.py`, dat sinds Phase 4 niet bestaat. Een whitelist die zijn eigen scope niet kent, bewaakt niets — dat is exact de constatering die al in de `Makefile` bij `loc-check` staat, en die daar heeft geleid tot de vervanger: **`scripts/check_file_size.py`**, een ratchet met een cap per bestand, gedraaid door `inventory.yml` en getoetst door `tests/unit/test_file_size_ratchet.py` (inclusief het bewijs dat hij rood kán worden). Die ratchet staat groen. De `ci.yml`-job is de oude, vervangen poort die nooit is opgeruimd.

**D3 — de dekkingsdrempel wordt niet gehaald, en dat is een mandaatkwestie.** `pyproject.toml` zet `fail_under = 70`. Gemeten op `371755e`:

```
TOTAL   24987 statements   10219 missed   59 %
FAIL Required test coverage of 70.0% not reached. Total coverage: 59.10%
```

Dat is een stijging ten opzichte van de 55,82 % uit `reports/phase9_coverage_baseline.md` (2026-09-04, commit `26cfd90`), en nog altijd **10,9 procentpunt** onder de eis. In statements: 14.768 gedekt, 17.491 nodig — een gat van **2.723 statements**.

Het raakt **twee** jobs, niet één:

- `inventory.yml` → de expliciete dekkingsstap (staat er bewust, zonder `continue-on-error`, omdat "een drempel die niet draait geen drempel is");
- `ci.yml` → `test-fast`, want `pytest --cov` leest `fail_under` uit `pyproject.toml` en laat de run daarop falen. Die job is dus mede rood om een reden die niet in zijn stapnaam staat.

Zie §4 — dit is de enige post in deze opdracht waarvoor jij geen mandaat hebt.

**D4 — één falende test, en het is een poort die correct afgaat.**

```
FAILED tests/unit/test_docs_claim_only_what_exists.py::TestEveryReferenceResolves::test_the_exception_list_does_not_rot
AssertionError: Deze paden staan als 'bewust afwezig' maar bestaan inmiddels:
['conf/env/prod.yaml']. Haal ze uit KNOWN_ABSENT.
```

`conf/env/prod.yaml` stond op de lijst van bewust ontbrekende paden en bestaat nu. De test doet precies waarvoor hij is geschreven: een uitzonderingslijst die niet meeloopt met de werkelijkheid, is verrot. De reparatie is één regel — het pad uit `KNOWN_ABSENT` halen — maar controleer eerst **waarom** het bestand er is gekomen en of er documentatie bij hoort die er nog niet is. Dit is tevens de reden dat de `hygiene.yml`-stap `pytest tests/unit` rood staat.

De rest van de suite: 4 XFAIL (de pre-geregistreerde killgates, exact de verwachte vier), ~23 SKIPPED (ontbrekende datacaches en nog niet geëvalueerde kill-gate-artefacten), 0 ERROR.

### 2.3 De twee schijnfouten

Beide zijn in §1 uitgewerkt en horen **niet** in de foutentelling: de `scripts.*`-importfouten (verkeerde interpreter) en de 59 markdownlint-bevindingen (ongetrackte bestanden). Rapporteer ze als meetartefact, niet als defect. Wat er wél uit volgt is stap 4: de suite hoort niet stil te vallen op de interpreter waarop hij toevallig draait.

### 2.4 De poorten die niets meten

Een poort die niet rood kan worden, is documentatie. Er staan er drie in de repository:

| Waar | Wat er staat | Wat het doet |
|---|---|---|
| `security-scan.yml` | `pip-audit --require-hashes --disable-pip \|\| true` | `\|\| true` maakt elke uitkomst groen. `--require-hashes` zonder requirements-bestand is bovendien een gebruiksfout, dus de stap faalt intern en zwijgt. **De CVE-poort heeft nooit iets gerapporteerd.** |
| `nightly-regression.yml` | `python apps/regenerate_baseline.py \|\| true` | Slaat de baselineregeneratie stilzwijgend over; de regressietests vergelijken daarna tegen een baseline waarvan niemand weet of hij vers is |
| `ci.yml` | `loc-guard` | D2: dubbele, verouderde poort naast een werkende ratchet |

Daarnaast worden `bandit` en `pip-audit` **ongepind** geïnstalleerd, en `ci.yml` installeert `pip install -e ".[dev]"` in plaats van de lockfiles — `mypy>=1.7` en `ruff>=0.1.8` betekent dat de releasedatum het oordeel bepaalt. Dat is **DI-16**, gemeten in Phase 7/8: ruff 0.15.12 gaf 7 bevindingen, ruff 0.16.4 gaf er 78 op identieke broncode. DI-16 is gesloten voor de lockfile-workflows en **open gebleven voor `ci.yml`**.

### 2.5 Twee omgevingen, twee oordelen

| | Python | Installatie | Gepind? |
|---|---|---|---|
| `ci.yml` | 3.11 | `pip install -e ".[dev]"` | **nee** |
| `hygiene.yml` | 3.13 | beide lockfiles + `-e . --no-deps` | ja |
| `inventory.yml` | 3.13.0 | beide lockfiles + `-e . --no-deps` | ja |
| `research_gates.yml` | 3.13 | beide lockfiles + `-e . --no-deps` | ja |
| `nightly-regression.yml` | 3.11 | `pip install -e ".[dev]"` | **nee** |
| `paper-trade-ci.yml` | 3.11 | `pip install -e ".[dev,ingestion]"` | **nee** |
| referentie-venv | 3.13.0 | beide lockfiles | ja |

Drie workflows meten een andere omgeving dan de referentie, op een andere Python-minor, met ongepinde linters. Elke bevinding die daar ontstaat, is niet reproduceerbaar op de interpreter waarop `docs/runbook.md` §0 meten geldig verklaart.

---

## 3. STAPSGEWIJZE UITVOERING

Elke stap: eerst meten, dan wijzigen, dan opnieuw meten, dan committen met beide getallen in de commit body.

### Stap 0 — reproduceer de nulmeting

- [ ] Verifieer de interpreter (§1.1). Een `C:`-pad = stoppen.
- [ ] Draai het volledige verificatieblok uit §5.2 en leg de uitkomst vast in `reports/ci_sanering_nulmeting.md`.
- [ ] Wijkt een regel af van §2.1, dan is **§2 verouderd, niet jouw meting**. Noteer het verschil expliciet voordat je verdergaat.
- [ ] Noteer per commando de **exitcode**. `docs/runbook.md` §0.3 noemt `1253 passed, 1 skipped` voor `pytest tests/unit`; die referentie klopt niet meer sinds D4 — meet hem opnieuw.

### Stap 1 — de falende test (D4)

Begin hier, niet bij mypy: dit is de goedkoopste reparatie en hij maakt twee stappen in twee workflows groen.

- [ ] Zoek uit waarom `conf/env/prod.yaml` bestaat: welke commit voegde hem toe, en hoort er documentatie bij die nog ontbreekt.
- [ ] Haal het pad uit `KNOWN_ABSENT` in `tests/unit/test_docs_claim_only_what_exists.py`.
- [ ] Controleer of de rest van `KNOWN_ABSENT` nog klopt — als één regel is verrot, is dat een aanwijzing over de hele lijst, geen toeval.
- [ ] **Meet:** `pytest tests/unit -q -p no:randomly` → 0 failures.

### Stap 2 — mypy naar nul (D1)

- [ ] Splits de 187 fouten in twee stapels: **mechanisch** (`type-arg`, `no-untyped-def`, `no-untyped-call`, `no-any-return`) en **verdacht** (`call-arg`, `arg-type`, `assignment`, `union-attr`, `operator`, `dict-item`, `misc`).
- [ ] Werk de mechanische stapel per bestand af, grootste eerst. Echte annotaties — `dict[str, float]`, niet `dict[Any, Any]`; `Queue[Task]`, niet `Queue[Any]`. Een `Any` die je zelf toevoegt om een fout te laten verdwijnen, valt onder V3.
- [ ] Behandel elke fout uit de verdachte stapel als kandidaat-bug. Voor elk: typefout of echte fout? Vind je een echte fout (`live_paper_trader.py:329` is de eerste kandidaat), schrijf er een test bij die zonder de reparatie rood is.
- [ ] Verwijder de dode override `module = ["ccxt.*"]` uit `pyproject.toml`, of motiveer waarom hij blijft.
- [ ] **Meet:** `mypy src/tradebot/schemas/ src/tradebot/utils/ apps/ --ignore-missing-imports` → `Success`. `mypy --strict src/tradebot/schemas/config.py src/tradebot/utils/failfast.py` blijft groen.
- [ ] Draai de volledige suite opnieuw. Een annotatie die gedrag wijzigt, is geen annotatie.

### Stap 3 — `loc-guard` vervangen door de ratchet die al bestaat (D2)

- [ ] Vervang de `loc-guard`-job in `ci.yml` door `python scripts/check_file_size.py`.
- [ ] Schrijf in de workflow, in commentaar, dezelfde redenering die bij `make loc-check` staat: de shell-lus hanteerde een whitelist die een niet-bestaand bestand noemde en zeven bestaande overtreders miste, en de ratchet toetst hetzelfde begrip strenger én is zelf getest.
- [ ] Verboden: de whitelist uitbreiden met de zeven bestanden. Dat is V6 in een andere vorm.
- [ ] **Meet:** de oude shellstap geeft exit 1; `scripts/check_file_size.py` geeft exit 0; `tests/unit/test_file_size_ratchet.py` blijft groen.
- [ ] De tien bestanden boven 800 regels zijn **DI-3** en blijven waar ze zijn. Splitsen valt buiten deze opdracht (§6).

### Stap 4 — de testsuite interpreter-onafhankelijk maken (§1.2)

- [ ] Bewijs eerst de diagnose: laat `scripts.__path__` zien op beide interpreters.
- [ ] Kies een reparatie en motiveer hem. De voor de hand liggende is `scripts/__init__.py` toevoegen, waardoor `D:\Tradebot\scripts` een reguliere package wordt en de `sys.path.insert(0, ...)` uit `tests/conftest.py` weer beslissend is.
- [ ] Controleer daarna **expliciet** of dat iets breekt: `reachability_map.py` (telt `scripts/` nu mee?), `check_file_size.py`, de LOC-ratchet, en elk script dat als `python scripts/x.py` draait.
- [ ] **Meet:** de collectie is schoon op *beide* interpreters, en alle zes de poortscripts geven nog steeds exit 0.
- [ ] Breekt de reparatie iets dat zwaarder weegt: niet doorduwen. Registreer hem als DI met de meting erbij.

### Stap 5 — één omgeving, één oordeel (§2.5, sluit DI-16)

- [ ] Zet `ci.yml`, `nightly-regression.yml` en `paper-trade-ci.yml` op Python 3.13 en op installatie uit beide lockfiles + `pip install -e . --no-deps`, gelijk aan `hygiene.yml` / `inventory.yml` / `research_gates.yml`.
- [ ] Werkt dat voor `paper-trade-ci.yml` niet (de `ingestion`-extra staat niet in de lockfiles), los dat expliciet op — extra toevoegen aan de lock, of de workflow documenteren als bewust afwijkend. Niet stilzwijgend laten staan.
- [ ] Verifieer dat `requires-python = ">=3.10"` in `pyproject.toml` nog klopt met wat er feitelijk wordt getest. Test je alleen 3.13, dan is `>=3.10` een claim zonder bewijs: óf een 3.10-matrix toevoegen, óf de claim aanpassen.
- [ ] **Meet:** `ruff --version` en `mypy --version` zijn in elke workflow gelijk aan de lockfile-pins (`ruff==0.15.12`, `mypy==2.3.1`).
- [ ] Werk de DI-16-regel in `docs/DEFERRED_ISSUES.md` bij: gesloten, met de commit erbij.

### Stap 6 — de poorten die niets meten laten meten (§2.4)

- [ ] `pip-audit`: haal `|| true` weg, haal `--require-hashes --disable-pip` weg, en richt hem op de lockfiles: `pip-audit -r requirements.lock -r requirements-dev.lock`. Pin `pip-audit` in `requirements-dev.lock`.
- [ ] Draai hem één keer en lees de uitkomst. Vindt hij CVE's, dan is dat een **bevinding, geen blokkade voor deze stap**: registreer elke CVE met versie en oordeel in `docs/DEFERRED_ISSUES.md` en laat de stap rood staan als hij rood hoort te staan. Onderdruk niets.
- [ ] `bandit`: pin hem in `requirements-dev.lock` en draai `bandit -r src/ apps/ -ll -q`. Zelfde regel: registreren, niet onderdrukken.
- [ ] `regenerate_baseline.py || true` in `nightly-regression.yml`: maak expliciet wat de bedoeling is. Óf de stap mag falen en dan hoort er een conditie bij die zegt wanneer, óf hij mag niet falen en dan gaat `|| true` eruit. Een derde optie is er niet.
- [ ] **Meet:** laat elke aangepaste stap één keer bewust rood worden (tijdelijk, lokaal) om te bewijzen dat hij dat kán. Zonder dat bewijs is de reparatie niet af — dat is de standaard uit `docs/PROJECT_STATE.md`.

### Stap 7 — de dekking (D3) — meten en voorleggen, niet oplossen

- [ ] Hermeet de dekking na stap 1 t/m 6; die stappen verplaatsen het cijfer.
- [ ] Maak de opsplitsing per pakket: welke modules dragen de 41 % ongedekt, en hoeveel van de 10.219 gemiste statements zitten in de tien grootste ongedekte bestanden.
- [ ] Schrijf `reports/ci_dekkingsplan.md`: het gemeten cijfer, de afstand tot 70, en drie opties met hun kosten — (a) tests schrijven tot 70 %, (b) een dekkingsratchet op het gemeten niveau met 70 als staand doel, (c) de drempel accepteren als permanent rode poort met geregistreerde motivering.
- [ ] **Neem optie (b) of (c) niet zelf.** Beide verlagen feitelijk de eis; dat is een mandaatbesluit (V1, V8). Leg de drie opties voor en stop daar.
- [ ] Tot de eigenaar heeft besloten: de stap blijft rood, en dat rood staat als zodanig in je eindrapport.

---

## 4. WAT "100 % VALIDE" HIER BETEKENT

De opdracht luidt: de hele repo clean en 100 % valide. Dat is haalbaar voor alles behalve één post, en die uitzondering hoort in de opdracht te staan, niet in een voetnoot.

**Wel haalbaar, volledig, in deze opdracht:** de falende test (D4), mypy naar nul (D1), `loc-guard` gerepareerd (D2), drie niet-metende poorten die weer meten (§2.4), één omgeving over alle workflows (§2.5), een testsuite die op elke interpreter collecteert (§1.2). Na stap 6 is elke poort in de repository óf groen, óf rood om een geregistreerde, gemeten reden.

**Niet haalbaar zonder besluit van de eigenaar:** de dekkingsdrempel (D3). 59,10 % naar 70 % is ~2.723 statements aan nieuwe tests. Dat is geen sanering maar een fase op zichzelf.

**Niet haalbaar, punt, en dat is bekend:** DI-15 (survivorship bias), DI-18 (ontbrekende variantieproxy) en DI-21 (H2-bezetting) zijn **dataposten**. Zij gaan niet open met code. `docs/DEFERRED_ISSUES.md` zegt bij DI-21 letterlijk wat er níét mag: *"`k` verlagen of de poort verruimen tot hij opengaat."* Diezelfde zin geldt voor elke poort in deze opdracht.

Een eindrapport dat zegt "alles groen" terwijl D3 is weggepoetst, is een mislukte opdracht. Een eindrapport dat zegt "alles groen behalve de dekkingsdrempel, hier is het gemeten cijfer, hier zijn drie opties, aan u de keuze" is een geslaagde.

---

## 5. EXIT-CRITERIA

### 5.1 De criteria

| # | Criterium | Bewijs |
|---|---|---|
| E1 | `mypy` op de `ci.yml`-scope geeft `Success` | commandouitvoer in het exit-rapport |
| E2 | `mypy --strict` op `config.py` + `failfast.py` blijft groen | idem |
| E3 | `ruff check src/ apps/ tests/` groen, **zonder** nieuwe regel in `ignore` | `git diff pyproject.toml` toont geen uitbreiding van `[tool.ruff.lint] ignore` |
| E4 | Alle zes de poortscripts geven exit 0 | exitcodes in het rapport |
| E5 | 0 FAILED in `pytest -m "not slow and not regression"`, en 0 in `pytest tests/unit` | beide uitvoeren |
| E6 | De vier pre-geregistreerde killgates staan XFAILED, identiek, 0 XPASS, 0 FAILED | de guard uit `inventory.yml` |
| E7 | `apps/run_gates.py` geeft exit 0 op vier poorten | artefact `research_gates.json` |
| E8 | De LOC-poort in `ci.yml` is de ratchet, en `tests/unit/test_file_size_ratchet.py` is groen | `git diff .github/workflows/ci.yml` |
| E9 | De drie niet-metende poorten kunnen aantoonbaar rood worden | per poort één bewust rode run, in het rapport |
| E10 | Elke workflow installeert dezelfde gepinde omgeving als de referentie | `git diff` over de workflows |
| E11 | Testcollectie is schoon op zowel de venv als de globale interpreter | beide uitvoeren in het rapport |
| E12 | Geen enkele drempel, ratchet of ignore-lijst is versoepeld | `git diff` over `pyproject.toml` en de drie ratchet-scripts, expliciet getoond |
| E13 | D3 is gemeten, opgeschreven en als besluit voorgelegd — niet opgelost | `reports/ci_dekkingsplan.md` |
| E14 | Elke niet-gesloten post staat als DI-regel in `docs/DEFERRED_ISSUES.md` | het diff van dat bestand |

### 5.2 Het verificatieblok — draai dit integraal, vóór en ná

```bash
PY="D:/venv/tradebot/Scripts/python.exe"

"$PY" -c "import tradebot; print(tradebot.__file__)"

"$PY" -m ruff check src/ apps/ tests/            ; echo "EXIT=$?"
"$PY" -m ruff check src/                          ; echo "EXIT=$?"
"$PY" -m mypy src/tradebot/schemas/ src/tradebot/utils/ apps/ --ignore-missing-imports ; echo "EXIT=$?"
"$PY" -m mypy --strict src/tradebot/schemas/config.py src/tradebot/utils/failfast.py   ; echo "EXIT=$?"

"$PY" scripts/reachability_map.py --strict        ; echo "EXIT=$?"
"$PY" scripts/check_file_size.py                  ; echo "EXIT=$?"
"$PY" scripts/check_hardcoded_params.py --strict  ; echo "EXIT=$?"
"$PY" scripts/audit_fallbacks.py --strict         ; echo "EXIT=$?"
"$PY" scripts/check_banned_methods.py --strict    ; echo "EXIT=$?"
"$PY" scripts/check_domain_consistency.py --strict; echo "EXIT=$?"

"$PY" -m pytest tests/unit -q -p no:randomly --no-header ; echo "EXIT=$?"
"$PY" -m pytest -m "not slow and not regression" -p no:randomly --cov=src/tradebot --cov-report=term -q ; echo "EXIT=$?"
"$PY" apps/run_gates.py --out artefacts/governance/research_gates.json ; echo "EXIT=$?"

npx --yes markdownlint-cli2@0.13.0 $(git ls-files '*.md') --config .markdownlint.json ; echo "EXIT=$?"
```

Noteer van elk commando de **exitcode**, niet alleen de laatste regel uitvoer. Een pipe naar `tail` verbergt de exitcode — dat is tijdens de nulmeting van 2026-09-18 één keer gebeurd en het maakte zes poorten ten onrechte groen in de eerste ronde.

---

## 6. FENCES — WAT DEZE OPDRACHT NIET AANRAAKT

| Niet aanraken | Waarom |
|---|---|
| De tien bestanden boven 800 regels | DI-3; splitsen is een architectuurbesluit, geen CI-reparatie |
| De 19 apps boven 80 regels | DI-4, idem |
| `scripts/` verplaatsen naar `research/` | DI-8; breekt verwijzingen in de wave-documentatie |
| De hardcoded-parameterbudgetten | DI-9 t/m DI-14; gebonden aan hun eigen fase |
| Regeleindes normaliseren | DI-17; één commit met een diff over de hele boom hoort niet halverwege deze opdracht |
| De 71 `PLR0917`-bevindingen van ruff 0.16.x | DI-16 §2; signatuurontwerp, ~71 aanroepketens, expliciet besluit |
| Elk statistisch contract, elke poortdrempel, elke preregistratie | Dit is een infrastructuuropdracht. Raakt je wijziging een getal waarop een hypothese is beoordeeld, dan doe je het verkeerde |
| `artefacts/governance/*.json` | Append-only ledger; zie DI-20 |
| De 39 `SWALLOWED_EXCEPT`-adviezen van `audit_fallbacks.py` | 0 daarvan zijn blokkerend; de poort staat groen. Ze aanpakken is een aparte, inhoudelijke opdracht |

Raakt een reparatie uit §3 een fence, stop en meld het. Niet doorwerken.

---

## 7. REGELS & COMMITDISCIPLINE

1. **Eén stap, één commit.** De commit body bevat de meting vóór en de meting ná, als getallen.
2. **Meet met exitcodes.** `echo "EXIT=$?"` direct achter het commando, vóór elke pipe.
3. **Nooit een poort verzwakken om hem te passeren** (§0.1). Bij twijfel: dat is het signaal, niet de uitzondering.
4. **Een reparatie is pas af als de bijbehorende test rood kan worden om de reden waarvoor hij bestaat.** Dit is de huisregel uit `docs/PROJECT_STATE.md` en hij geldt hier onverkort.
5. **Rapporteer rood als rood.** Een stap die rood blijft met een geregistreerde reden is een geldig eindresultaat; een stap die groen is gemaakt zonder die reden is dat niet.
6. **Schrijf in het Nederlands**, in de stijl van de bestaande documenten in `docs/` en `reports/`.
7. **Geen nieuwe afhankelijkheden** zonder ze in beide lockfiles te pinnen en `docs/DEPENDENCY_CONTRACT.md` bij te werken.

---

## 8. STARTINSTRUCTIE

Begin met stap 0. Wijzig niets voordat het volledige verificatieblok uit §5.2 is gedraaid en de uitkomst naast §2.1 is gelegd.

Rapporteer daarna, vóór je stap 1 begint, in maximaal één scherm:

1. welke regels uit §2.1 zijn veranderd sinds 2026-09-18, met het nieuwe getal;
2. het gemeten resultaat van `pytest tests/unit` en de gemeten dekking;
3. in welke volgorde je stap 1 t/m 7 doet, en waarom die volgorde;
4. elk punt waarop je denkt dat §0.1 je in de weg zit — dát is het moment om het te zeggen, niet achteraf.
