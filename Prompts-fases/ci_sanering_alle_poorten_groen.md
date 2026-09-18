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

Mypy 1.17 en 2.3.1 geven een andere foutverzameling op identieke broncode (gemeten: 188 tegen 187, met een andere verdeling over codes). Dit is dezelfde klasse als **DI-16** en het is de reden dat stap 3 bestaat.

---

## 2. NULMETING — gemeten 2026-09-18 op `371755e`, referentie-venv

### 2.1 Elke workflow, elke job, elke stap

| Workflow | Job / stap | Status | Gemeten |
|---|---|---|---|
| `ci.yml` | lint → `ruff check src/ apps/ tests/` | **GROEN** | `All checks passed!` (installeert sinds deze branch uit beide lockfiles) |
| `ci.yml` | lint → `mypy schemas/ utils/ apps/` | **ROOD** | **187 fouten in 34 bestanden** (64 gecontroleerd) |
| `ci.yml` | `loc-guard` (= `check_file_size.py`) | GROEN | exit 0; deze branch verving de inline shell-lus al door de ratchet |
| `ci.yml` | `test-fast` (pytest + `--cov`) | **ROOD** | dekkingsdrempel; exit 1. 0 FAILED op een verse checkout |
| `hygiene.yml` | dependency-contract (21 harde deps) | GROEN | alle 21 aanwezig |
| `hygiene.yml` | `git_sha` resolvable | GROEN | `371755e`, `git cat-file -e` OK |
| `hygiene.yml` | `audit_fallbacks.py --strict` | GROEN | exit 0; 39 bevindingen, **0 blokkerend** |
| `hygiene.yml` | `check_hardcoded_params.py --strict` | GROEN | exit 0; 298 literals / 98 bestanden, ratchet 313 |
| `hygiene.yml` | geen `quant_architect`-imports | GROEN | 0 treffers |
| `hygiene.yml` | DAG-mappen + `README.md` | GROEN | 3/3 aanwezig |
| `hygiene.yml` | `ruff` (src + 5 bestanden) | GROEN | `All checks passed!` |
| `hygiene.yml` | `mypy --strict config.py failfast.py` | GROEN | `Success: no issues found in 2 source files` |
| `hygiene.yml` | `pytest tests/unit -q` | GROEN | groen op een verse checkout; lokaal 1 failure, zie §2.3 val 3 |
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

### 2.2 De twee echte defecten

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

**D2 — de dekkingsdrempel wordt niet gehaald, en dat is een mandaatkwestie.** `pyproject.toml` zet `fail_under = 70`. Gemeten op `371755e`:

```
TOTAL   24987 statements   10219 missed   59 %
FAIL Required test coverage of 70.0% not reached. Total coverage: 59.10%
```

Dat is een stijging ten opzichte van de 55,82 % uit `reports/phase9_coverage_baseline.md` (2026-09-04, commit `26cfd90`), en nog altijd **10,9 procentpunt** onder de eis. In statements: 14.768 gedekt, 17.491 nodig — een gat van **2.723 statements**.

Het raakt **twee** jobs, niet één:

- `inventory.yml` → de expliciete dekkingsstap (staat er bewust, zonder `continue-on-error`, omdat "een drempel die niet draait geen drempel is");
- `ci.yml` → `test-fast`, want `pytest --cov` leest `fail_under` uit `pyproject.toml` en laat de run daarop falen. Die job is dus mede rood om een reden die niet in zijn stapnaam staat.

Zie §4 — dit is de enige post in deze opdracht waarvoor jij geen mandaat hebt.

De rest van de suite: 1 FAILED (lokaal; zie §2.3 val 3 — op een verse checkout 0), 4 XFAIL (de pre-geregistreerde killgates, exact de verwachte vier), ~23 SKIPPED (ontbrekende datacaches en nog niet geëvalueerde kill-gate-artefacten), 0 ERROR.

### 2.3 De drie schijnfouten

Geen van drieën hoort in de foutentelling. Rapporteer ze als meetartefact, niet als defect.

**Val 1 en 2** staan in §1.2 en §1.3: de `scripts.*`-importfouten (verkeerde interpreter) en de 59 markdownlint-bevindingen (ongetrackte bestanden).

**Val 3 — de falende test die op CI niet faalt.** Lokaal geeft de suite:

```
FAILED tests/unit/test_docs_claim_only_what_exists.py::TestEveryReferenceResolves::test_the_exception_list_does_not_rot
AssertionError: Deze paden staan als 'bewust afwezig' maar bestaan inmiddels:
['conf/env/prod.yaml']. Haal ze uit KNOWN_ABSENT.
```

Dat leest als een poort die correct afgaat, en dat is het niet. `conf/env/` is **niet getrackt** — `.gitignore` regel 33 (`env/`, geschreven voor virtualenvs) sluit de hele Hydra-`env`-groep uit, en geen van de vier bestanden erin zit in de boom. Het bestand bestaat alleen op deze machine.

Bewijs, en voer dit uit voordat je er iets aan doet:

```bash
git clone --single-branch --branch <branch> . /tmp/fresh
cd /tmp/fresh && ls conf/env/          # bestaat niet
pytest tests/unit/test_docs_claim_only_what_exists.py -q   # 9 passed
```

Gemeten 2026-09-18: in de verse checkout slagen alle negen tests. De `KNOWN_ABSENT`-regel is dus **juist** en het pad eruit halen zou CI kapotmaken, niet repareren.

Wat er wél onder zit, is kleiner en van een andere orde: `_resolves()` oordeelt op bestaan in het bestandssysteem, dus een ongetrackt lokaal bestand kantelt het oordeel van deze poort. Vandaag is dat precies één pad. Dat is dezelfde klasse als val 2 en het hoort geregistreerd te worden — maar de semantiek van een governance-test wijzigen is géén bijvangst van een CI-sanering. Zie §6.

Wat er uit val 1 en 3 samen volgt is stap 2: de suite hoort hetzelfde te zeggen op elke interpreter en in elke werkboom, of expliciet te vermelden dat hij dat niet doet.

### 2.4 De poorten die niets meten

Een poort die niet rood kan worden, is documentatie. Er staan vijf `|| true`-maskers in de workflows:

| Waar | Wat er staat | Wat het doet |
|---|---|---|
| `security-scan.yml:32` | `pip-audit --require-hashes --disable-pip \|\| true` | `\|\| true` maakt elke uitkomst groen. `--require-hashes` zonder requirements-bestand is bovendien een gebruiksfout, dus de stap faalt intern en zwijgt. **De CVE-poort heeft nooit iets gerapporteerd.** |
| `nightly-regression.yml:33` | `python apps/regenerate_baseline.py \|\| true` | Slaat de baselineregeneratie stilzwijgend over; de regressietests vergelijken daarna tegen een baseline waarvan niemand weet of hij vers is |
| `nightly-regression.yml:87` | `python -m apps.monitor_drift \|\| true` | Een driftmonitor die niet mag afgaan, monitort niets |
| `nightly-regression.yml:119` | inline rapportgeneratie, `\|\| true` | Idem; het rapport kan ontbreken zonder dat iets rood wordt |
| `paper-trade-ci.yml:94` | `--output paper_trade_report.md \|\| true` | Idem |

Beoordeel ze niet als één klasse. Een `|| true` op een **rapport**generator is verdedigbaar als het rapport bijvangst is; een `|| true` op een **poort** is dat nooit. `pip-audit` en `regenerate_baseline` zijn poorten. Schrijf per regel op welke van de twee het is, en verwijder het masker waar het een poort maskeert.

Daarnaast worden `bandit` en `pip-audit` **ongepind** geïnstalleerd. Dat is **DI-16**, gemeten in Phase 7/8: ruff 0.15.12 gaf 7 bevindingen, ruff 0.16.4 gaf er 78 op identieke broncode. DI-16 is gesloten voor `ci.yml`, `hygiene.yml`, `inventory.yml` en `research_gates.yml`, en **open gebleven** voor `nightly-regression.yml`, `paper-trade-ci.yml` en `security-scan.yml`.

### 2.5 Twee omgevingen, twee oordelen

| | Python | Installatie | Gepind? |
|---|---|---|---|
| `ci.yml` | **3.11** | beide lockfiles + `-e . --no-deps` | ja |
| `hygiene.yml` | 3.13 | beide lockfiles + `-e . --no-deps` | ja |
| `inventory.yml` | 3.13.0 | beide lockfiles + `-e . --no-deps` | ja |
| `research_gates.yml` | 3.13 | beide lockfiles + `-e . --no-deps` | ja |
| `nightly-regression.yml` | 3.11 | `pip install -e ".[dev]"` | **nee** |
| `paper-trade-ci.yml` | 3.11 | `pip install -e ".[dev,ingestion]"` | **nee** |
| `security-scan.yml` | 3.11 | `pip install bandit[toml]` / `pip-audit` | **nee** |
| referentie-venv | 3.13.0 | beide lockfiles | ja |

Twee dingen staan hier los van elkaar.

**De pins.** Drie workflows installeren nog ongepind. Elke bevinding die daar ontstaat, is niet reproduceerbaar op de interpreter waarop `docs/runbook.md` §0 meten geldig verklaart.

**De Python-minor.** `ci.yml` installeert de lockfiles — die op 3.13 zijn opgelost — op **3.11**. Dat is geen pin-probleem maar een tweede omgeving: dezelfde pins op een andere minor geven een andere resolutie, en in het slechtste geval installeert hij niet eens. De vier gates die het project als geldig erkent draaien op 3.13. Meet of `ci.yml` op 3.11 überhaupt installeert voordat je iets anders concludeert.


---

## 3. STAPSGEWIJZE UITVOERING

Elke stap: eerst meten, dan wijzigen, dan opnieuw meten, dan committen met beide getallen in de commit body.

### Stap 0 — reproduceer de nulmeting

- [ ] Verifieer de interpreter (§1.1). Een `C:`-pad = stoppen.
- [ ] Draai het volledige verificatieblok uit §5.2 en leg de uitkomst vast in `reports/ci_sanering_nulmeting.md`.
- [ ] Wijkt een regel af van §2.1, dan is **§2 verouderd, niet jouw meting**. Noteer het verschil expliciet voordat je verdergaat.
- [ ] Noteer per commando de **exitcode**. `docs/runbook.md` §0.3 noemt `1253 passed, 1 skipped` voor `pytest tests/unit` — meet of die referentie nog klopt.
- [ ] Draai elke rode uitkomst **ook in een verse kloon** voordat je hem een defect noemt. Drie van de vier bevindingen in de eerste meetronde van 2026-09-18 waren dat niet (§2.3).

### Stap 1 — mypy naar nul (D1)

- [ ] Splits de 187 fouten in twee stapels: **mechanisch** (`type-arg`, `no-untyped-def`, `no-untyped-call`, `no-any-return`) en **verdacht** (`call-arg`, `arg-type`, `assignment`, `union-attr`, `operator`, `dict-item`, `misc`).
- [ ] Werk de mechanische stapel per bestand af, grootste eerst. Echte annotaties — `dict[str, float]`, niet `dict[Any, Any]`; `Queue[Task]`, niet `Queue[Any]`. Een `Any` die je zelf toevoegt om een fout te laten verdwijnen, valt onder V3.
- [ ] Behandel elke fout uit de verdachte stapel als kandidaat-bug. Voor elk: typefout of echte fout? Vind je een echte fout (`live_paper_trader.py:329` is de eerste kandidaat), schrijf er een test bij die zonder de reparatie rood is.
- [ ] Verwijder de dode override `module = ["ccxt.*"]` uit `pyproject.toml`, of motiveer waarom hij blijft.
- [ ] **Meet:** `mypy src/tradebot/schemas/ src/tradebot/utils/ apps/ --ignore-missing-imports` → `Success`. `mypy --strict src/tradebot/schemas/config.py src/tradebot/utils/failfast.py` blijft groen.
- [ ] Draai de volledige suite opnieuw. Een annotatie die gedrag wijzigt, is geen annotatie.

### Stap 2 — de testsuite interpreter-onafhankelijk maken (§1.2)

- [ ] Bewijs eerst de diagnose: laat `scripts.__path__` zien op beide interpreters.
- [ ] Kies een reparatie en motiveer hem. De voor de hand liggende is `scripts/__init__.py` toevoegen, waardoor `D:\Tradebot\scripts` een reguliere package wordt en de `sys.path.insert(0, ...)` uit `tests/conftest.py` weer beslissend is.
- [ ] Controleer daarna **expliciet** of dat iets breekt: `reachability_map.py` (telt `scripts/` nu mee?), `check_file_size.py`, de LOC-ratchet, en elk script dat als `python scripts/x.py` draait.
- [ ] **Meet:** de collectie is schoon op *beide* interpreters, en alle zes de poortscripts geven nog steeds exit 0.
- [ ] Breekt de reparatie iets dat zwaarder weegt: niet doorduwen. Registreer hem als DI met de meting erbij.

### Stap 3 — één omgeving, één oordeel (§2.5, sluit DI-16)

- [ ] Zet `ci.yml`, `nightly-regression.yml` en `paper-trade-ci.yml` op Python 3.13 en op installatie uit beide lockfiles + `pip install -e . --no-deps`, gelijk aan `hygiene.yml` / `inventory.yml` / `research_gates.yml`.
- [ ] Werkt dat voor `paper-trade-ci.yml` niet (de `ingestion`-extra staat niet in de lockfiles), los dat expliciet op — extra toevoegen aan de lock, of de workflow documenteren als bewust afwijkend. Niet stilzwijgend laten staan.
- [ ] Verifieer dat `requires-python = ">=3.10"` in `pyproject.toml` nog klopt met wat er feitelijk wordt getest. Test je alleen 3.13, dan is `>=3.10` een claim zonder bewijs: óf een 3.10-matrix toevoegen, óf de claim aanpassen.
- [ ] **Meet:** `ruff --version` en `mypy --version` zijn in elke workflow gelijk aan de lockfile-pins (`ruff==0.15.12`, `mypy==2.3.1`).
- [ ] Werk de DI-16-regel in `docs/DEFERRED_ISSUES.md` bij: gesloten, met de commit erbij.

### Stap 4 — de poorten die niets meten laten meten (§2.4)

- [ ] `pip-audit`: haal `|| true` weg, haal `--require-hashes --disable-pip` weg, en richt hem op de lockfiles: `pip-audit -r requirements.lock -r requirements-dev.lock`. Pin `pip-audit` in `requirements-dev.lock`.
- [ ] Draai hem één keer en lees de uitkomst. Vindt hij CVE's, dan is dat een **bevinding, geen blokkade voor deze stap**: registreer elke CVE met versie en oordeel in `docs/DEFERRED_ISSUES.md` en laat de stap rood staan als hij rood hoort te staan. Onderdruk niets.
- [ ] `bandit`: pin hem in `requirements-dev.lock` en draai `bandit -r src/ apps/ -ll -q`. Zelfde regel: registreren, niet onderdrukken.
- [ ] `regenerate_baseline.py || true` in `nightly-regression.yml`: maak expliciet wat de bedoeling is. Óf de stap mag falen en dan hoort er een conditie bij die zegt wanneer, óf hij mag niet falen en dan gaat `|| true` eruit. Een derde optie is er niet.
- [ ] **Meet:** laat elke aangepaste stap één keer bewust rood worden (tijdelijk, lokaal) om te bewijzen dat hij dat kán. Zonder dat bewijs is de reparatie niet af — dat is de standaard uit `docs/PROJECT_STATE.md`.

### Stap 5 — de dekking (D2) — meten en voorleggen, niet oplossen

- [ ] Hermeet de dekking na stap 1 t/m 4; die stappen verplaatsen het cijfer.
- [ ] Maak de opsplitsing per pakket: welke modules dragen de 41 % ongedekt, en hoeveel van de 10.219 gemiste statements zitten in de tien grootste ongedekte bestanden.
- [ ] Schrijf `reports/ci_dekkingsplan.md`: het gemeten cijfer, de afstand tot 70, en drie opties met hun kosten — (a) tests schrijven tot 70 %, (b) een dekkingsratchet op het gemeten niveau met 70 als staand doel, (c) de drempel accepteren als permanent rode poort met geregistreerde motivering.
- [ ] **Neem optie (b) of (c) niet zelf.** Beide verlagen feitelijk de eis; dat is een mandaatbesluit (V1, V8). Leg de drie opties voor en stop daar.
- [ ] Tot de eigenaar heeft besloten: de stap blijft rood, en dat rood staat als zodanig in je eindrapport.

---

## 4. WAT "100 % VALIDE" HIER BETEKENT

De opdracht luidt: de hele repo clean en 100 % valide. Dat is haalbaar voor alles behalve één post, en die uitzondering hoort in de opdracht te staan, niet in een voetnoot.

**Wel haalbaar, volledig, in deze opdracht:** mypy naar nul (D1), de niet-metende poorten die weer meten (§2.4), één omgeving over alle workflows (§2.5), een testsuite die op elke interpreter collecteert (§1.2). Na stap 4 is elke poort in de repository óf groen, óf rood om een geregistreerde, gemeten reden.

**Niet haalbaar zonder besluit van de eigenaar:** de dekkingsdrempel (D2). 59,10 % naar 70 % is ~2.723 statements aan nieuwe tests. Dat is geen sanering maar een fase op zichzelf.

**Niet haalbaar, punt, en dat is bekend:** DI-15 (survivorship bias), DI-18 (ontbrekende variantieproxy) en DI-21 (H2-bezetting) zijn **dataposten**. Zij gaan niet open met code. `docs/DEFERRED_ISSUES.md` zegt bij DI-21 letterlijk wat er níét mag: *"`k` verlagen of de poort verruimen tot hij opengaat."* Diezelfde zin geldt voor elke poort in deze opdracht.

Een eindrapport dat zegt "alles groen" terwijl D2 is weggepoetst, is een mislukte opdracht. Een eindrapport dat zegt "alles groen behalve de dekkingsdrempel, hier is het gemeten cijfer, hier zijn drie opties, aan u de keuze" is een geslaagde.

---

## 5. EXIT-CRITERIA

### 5.1 De criteria

| # | Criterium | Bewijs |
|---|---|---|
| E1 | `mypy` op de `ci.yml`-scope geeft `Success` | commandouitvoer in het exit-rapport |
| E2 | `mypy --strict` op `config.py` + `failfast.py` blijft groen | idem |
| E3 | `ruff check src/ apps/ tests/` groen, **zonder** nieuwe regel in `ignore` | `git diff pyproject.toml` toont geen uitbreiding van `[tool.ruff.lint] ignore` |
| E4 | Alle zes de poortscripts geven exit 0 | exitcodes in het rapport |
| E5 | 0 FAILED in `pytest -m "not slow and not regression"` en in `pytest tests/unit`, **gemeten in een verse kloon** | beide uitvoeren, in `/tmp/fresh` |
| E6 | De vier pre-geregistreerde killgates staan XFAILED, identiek, 0 XPASS, 0 FAILED | de guard uit `inventory.yml` |
| E7 | `apps/run_gates.py` geeft exit 0 op vier poorten | artefact `research_gates.json` |
| E8 | Elk `\|\| true` dat een POORT maskeert is weg, en die poort kan aantoonbaar rood worden | per poort één bewust rode run, in het rapport |
| E9 | Elke workflow installeert dezelfde gepinde omgeving als de referentie | `git diff` over de workflows |
| E10 | Testcollectie is schoon op zowel de venv als de globale interpreter | beide uitvoeren in het rapport |
| E11 | Geen enkele drempel, ratchet of ignore-lijst is versoepeld | `git diff` over `pyproject.toml` en de drie ratchet-scripts, expliciet getoond |
| E12 | D2 is gemeten, opgeschreven en als besluit voorgelegd — niet opgelost | `reports/ci_dekkingsplan.md` |
| E13 | Elke niet-gesloten post staat als DI-regel in `docs/DEFERRED_ISSUES.md` | het diff van dat bestand |

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
| De semantiek van `test_docs_claim_only_what_exists.py` | §2.3 val 3: `_resolves()` oordeelt op het bestandssysteem, niet op de git-boom, dus een ongetrackt lokaal bestand kantelt het oordeel. CI is groen. Dit herschrijven raakt een governance-poort en vraagt een eigen afweging — registreer het als DI |

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
3. in welke volgorde je stap 1 t/m 5 doet, en waarom die volgorde;
4. elk punt waarop je denkt dat §0.1 je in de weg zit — dát is het moment om het te zeggen, niet achteraf.
