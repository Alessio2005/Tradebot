# Phase 9 — exit-rapport

> **Fase:** 9 — opschoning, consolidatie & testmassa. **Startpunt:** `f7702dd`.
> **Afgerond:** 2026-09-04 over 26 commits.
>
> Per exit-criterium staat hieronder het commando, de output en de commit-sha.
> Alle metingen zijn gedaan op de referentie-interpreter uit `docs/RUNBOOK.md`
> §0 — `D:/venv/tradebot/Scripts/python.exe`, Python 3.13.0. Een meting op de
> systeeminterpreter is in dit project geen meting; dat is de eerste bevinding
> van deze fase en zij staat in `reports/phase9_environment_audit.md`.

---

## Wat de fase heeft gedaan, in één tabel

| | vóór (`f7702dd`) | na (`HEAD`) |
|---|---:|---:|
| modules in `src/` | 299 | **290** |
| LOC in `src/` | 71.487 | **71.025** |
| klasse E — onbereikbaar | 11 / 1.505 LOC | **2 / 1.018 LOC**, beide geregistreerd |
| klasse D — test-only | 20 / 3.036 LOC | **20 / 3.036 LOC**, elk met afnemer |
| bestanden in `scripts/` | 58 | **13** platformgereedschap |
| bestanden in `research/` | — | **48** onderzoek |
| entrypoints (`apps/` + `scripts/`) | 98 | **53** |
| daarvan die nergens worden genoemd | **36** | **0** |
| pakketten zonder enige test | 1 (`selection`) | **0** |
| tests | 2.797 | **2.885** |
| dekking (statements) | 55,82 % | **56,64 %** |
| gedekte statements | 13.391 | **13.480** |

Netto −462 LOC: 487 verwijderd, 25 toegevoegd (acht `# LOC-EXCEPTION:`-regels
en hun context). De opruiming is klein. Dat is de uitkomst, niet een
tekortkoming: van de 39 modules die de nulmeting onbereikbaar noemde, bleken er
28 wél bereikt — zie criterium 3.

De twee regels die er het meest toe doen staan onderaan: er zijn **88 tests bij
gekomen** en **89 statements méér gedekt**. Een opruimfase die alleen ongedekte
code weggooit, laat die laatste teller stilstaan.

---

## Criterium 1 — gedragsneutraliteit

> *`python -m pytest -q` geeft dezelfde verzameling failing node-id's als de
> nulmeting. Minder failures is een gefaalde fase. Een skip die in een error
> verandert telt als regressie.*

```bash
D:/venv/tradebot/Scripts/python.exe -m pytest -q -p no:randomly
```

| | nulmeting (`cfeddf0`) | nameting (`HEAD`) | Δ |
|---|---:|---:|---:|
| collected | 2797 | **2885** | +88 |
| passed | 2770 | **2858** | **+88** |
| failed | 4 | **4** | **0** |
| skipped | 23 | **23** | **0** |
| errors | 0 | **0** | **0** |

**Failed, skipped en errors staan alle drie stil.** Dat is wat het criterium
vraagt: meer passed mag, minder failures niet, en een skip die error wordt telt
als regressie.

### De +88 is nagerekend, niet aangenomen

Vijf nieuwe testbestanden brengen **92** tests:

| Bestand | Tests | Stap |
|---|---:|---|
| `tests/unit/test_reachability_map.py` | 22 | 4 — de spec die vóór de scanner is geschreven |
| `tests/unit/test_selection.py` | 22 | 11 — het pakket dat **nul** tests had |
| `tests/unit/test_oms_router_signing_and_rounding.py` | 22 | 11 — de live-orderweg |
| `tests/unit/test_file_size_ratchet.py` | 14 | 13 |
| `tests/unit/test_clean.py` | 12 | 14 |
| **totaal** | **92** | |

92 toegevoegd tegen +88 netto betekent dat er **vier** verdwenen. Zij zijn
gevonden door de collectie van `f7702dd` in een tijdelijke worktree te draaien
en bestand voor bestand te vergelijken:

| Bestand | vóór | na |
|---|---:|---:|
| `tests/unit/test_alpha_isolation.py` | 51 | **49** |
| `tests/unit/test_risk_alpha_decoupling.py` | 55 | **53** |

Beide bestanden zijn in deze fase **niet aangeraakt**. Zij parametriseren over
een glob van de map zelf:

```python
ALPHA_MODULES = sorted(p for p in ALPHA_DIR.glob("*.py"))
```

Stap 8 verwijderde twee alpha-modules (`decay_tracker.py`, `eq_strev_resid.py`),
dus verloren beide suites elk twee parametrisaties. **Er is geen assertie
verdwenen en geen test uitgezet** — er zijn twee onderwerpen minder om te
scannen, en de suites scannen nog steeds élke alpha-module die bestaat. Dat is
het gedrag dat je van een glob-geparametriseerde poort wilt: hij krimpt met
zijn onderwerp mee en niet met zijn strengheid.

De overige **123 testbestanden hebben tot op de test hetzelfde aantal** vóór en
na — bestand voor bestand vergeleken, nul verschillen buiten deze twee.

De vier node-id's zijn dezelfde vier:

```
tests/killgates/test_expansion_killgates.py::test_futures_unit_against_gate[KG-B1 in-sample-cm_carry]
tests/killgates/test_expansion_killgates.py::test_futures_unit_against_gate[KG-B2 residual alpha-cm_tsmom]
tests/killgates/test_expansion_killgates.py::test_futures_unit_against_gate[KG-B2 residual alpha-cm_carry]
tests/killgates/test_expansion_killgates.py::test_futures_unit_against_gate[KG-B3 out-of-sample-cm_carry]
```

Ook de assertiewaarden zijn ongewijzigd — `cm_carry` net_sharpe 0,09987 en
years_positive_frac 0,4516 tegen KG-B1; `cm_tsmom` alpha_p 0,0793 en `cm_carry`
alpha_p 0,3618 tegen KG-B2; `cm_carry` wf_sharpe −0,2784, sharpe_decay 1,613 en
bootstrap_p_positive 0,723 tegen KG-B3. De poorten falen om dezelfde reden en
met hetzelfde getal, niet toevallig even vaak.

De 23 skips zijn regel voor regel dezelfde, inclusief de elf die overslaan op
een ontbrekend killgate-artefact of een ontbrekende datacache. Nul skips zijn
error geworden.

**Bewijs:** `reports/phase9_fingerprint_before.txt` tegen
`reports/phase9_fingerprint_after.txt`.

### Eén regressie, gevonden en verholpen

Volledigheidshalve: de eerste nameting gaf **vijf** failures. De vijfde was

```
tests/unit/test_docs_claim_only_what_exists.py::TestEveryReferenceResolves::test_no_document_claims_a_path_that_does_not_exist
AssertionError: Deze documenten verwijzen naar paden die niet bestaan:
{'PROJECT_STATE.md': ['reports/phase9_exit_report.md']}
```

`PROJECT_STATE.md` §4 verwees naar dít rapport voordat het bestond. De poort
deed exact zijn werk. Het rapport schrijven lost hem op — niet de verwijzing
weghalen. Vermeld omdat de fase-opdracht een vijfde failure een gefaalde fase
noemt, en de bewering "vier failures" anders niet controleerbaar is.

### Eén verwijdering teruggedraaid vóór ze een commit werd

De werkboom droeg bij aanvang van stap 16 een **ongecommitteerde verwijdering**
van `src/tradebot/artefacts/PROPFIRM_PARALLEL_AUDIT.md` — bijvangst van de
stap 14-sweep die `src/tradebot/artefacts/runbook.md` verplaatste. Dat bestand
draagt een expliciet **BEHOUDEN**-verdict in
`reports/phase9_environment_audit.md` regel 127, het staat niet in het
verwijderregister, en het wordt aangehaald door drie getrackte bestanden:

```
src/tradebot/risk/daily_loss_governor.py:3
tests/unit/test_propfirm_governor.py:3
docs/STRATEGY_AUDIT_CRYPTO_ACCOUNTS_2026-06-14.md:9
```

Hersteld met `git checkout --`. Een document dat de risicolaag als bron noemt,
verdwijnt niet als bijvangst; en een verwijdering zonder registerregel is
precies de faalklasse die criterium 8 afdekt.

---

## Criterium 2 — baseline-identiteit

> *`python apps/run_phase5_baseline.py` levert bit-identieke metrieken. `pytest
> -m regression` staat groen.*

```bash
D:/venv/tradebot/Scripts/python.exe apps/run_phase5_baseline.py
```

De baseline schrijft 16 rijen over 30 velden. Veldsgewijs vergeleken, met NaN
als gelijk aan NaN (een naïeve `==` meldt 245 valse verschillen omdat
`nan != nan` in IEEE-754):

```
REAL differing values: 1 of 485
   /git_sha | f7702dd -> b9dc476
```

**484 van de 485 velden zijn identiek.** Het enige verschil is het
`git_sha`-metadataveld, dat per constructie de commit noteert waarop de run
plaatsvond. Elke metriek — Sharpe, turnover, kosten, fill-ratio's, de
sovereign-clip- en halt-tellers — staat op dezelfde waarde.

**Bewijs:** `reports/phase9_baseline_before.json` tegen
`reports/phase9_baseline_after.json`.

```bash
D:/venv/tradebot/Scripts/python.exe -m pytest -m regression -q -p no:randomly
```

Zie criterium 1; de regressiemarker staat groen en is de voorwaarde waarop
`portfolio/legacy_sizing.py` behouden is gebleven (stap 8, DI-10).

---

## Criterium 3 — klasse E is leeg, of geregistreerd

> *`python scripts/reachability_map.py --strict` retourneert 0.*

```bash
D:/venv/tradebot/Scripts/python.exe scripts/reachability_map.py --strict
```

```
A:  193 modules    53096 LOC
B:   67 modules    12767 LOC
C:    8 modules     1108 LOC
D:   20 modules     3036 LOC
E:    2 modules     1018 LOC
totaal: 290 modules, 71025 LOC

geregistreerd onbereikbaar (2), zie docs/CODE_REGISTER.md:
  tradebot.portfolio.covariance  (194 LOC)
  tradebot.portfolio.legacy_sizing  (824 LOC)

EXIT=0
```

Beide resterende E-modules staan met naam en reden in `docs/CODE_REGISTER.md`
§"Klasse E — onbereikbaar, en toch behouden". Zij dragen een **bestaansassertie**
in `tests/unit/test_risk_alpha_decoupling.py` (Phase 4 exit-criterium 3) plus
DI-10, dat `legacy_sizing.py` ONGEWIJZIGD houdt zodat de Phase 3-baseline
herrekenbaar blijft. Een test die eist dát een module bestaat, is een afnemer
die de importgraaf niet ziet.

### De poort kan rood worden

Vereist door de harde regel. In wegwerpbranch `phase9-gate-proof` op `736a988`
is `src/tradebot/deliberately_unreachable.py` toegevoegd — door niets
geïmporteerd:

```
FAIL: 1 ONGEREGISTREERDE onbereikbare module(s):
  tradebot.deliberately_unreachable  (src/tradebot/deliberately_unreachable.py, 3 LOC)
EXIT=1
```

**Bewijs:** `reports/phase9_gate_proof.txt`, poort 1 van 5. Commit `dd9451d`.

### Waarom de kaart afwijkt van de nulmeting

De fase-opdracht noemde 39 onbereikbare modules; gemeten zijn het er 11. Het
totaal is identiek (299 / 71.487) en klasse C komt tot op de LOC overeen — het
is dus dezelfde boom, en het verschil zit in de resolver. Twee edges verklaren
het gat, en beide zijn échte imports: `from pkg import submodule` (de dotted
naam staat nergens in de tekst) en het uitvoeren van bovenliggende
`__init__.py`'s bij een submodule-import. `src/tradebot/risk/__init__.py`
re-exporteert acht modules regel voor regel; wie `tradebot.risk` importeert,
importeert ze alle acht.

**28 modules die op de nulmeting verwijderbaar leken, droegen een echte
import.** Dat is de reden dat de scanner er is, en waarom stap 5 vóór stap 8
komt. **Bewijs:** `reports/phase9_reachability_audit.md`, commit `b0ac67d`.

---

## Criterium 4 — elke overlevende module heeft een verdict

> *`reports/phase9_inventory.md` bevat 299 rijen plus de apps en scripts, elk
> met klasse, dekking en verdict. Nul rijen zonder verdict.*

`reports/phase9_inventory.md` (commit `f8acf67`) telt **441 tabelregels** over
zeven secties: klasse E (11), D (20), C (8), B (67), A (193), apps (40) en
scripts (59). De samenvatting sluit op het totaal:

| Verdict | Modules | LOC |
|---|---:|---:|
| BEHOUDEN | 174 | 38.393 |
| TESTEN | 110 | 32.041 |
| VERWIJDEREN | 9 | 487 |
| VERPLAATSEN naar `research/` | 6 | 566 |
| **totaal** | **299** | **71.487** |

Nul rijen zonder verdict. De twee plaatsen waar de beslistabel een keuze
toestond, zijn expliciet gemaakt in plaats van opengelaten:

* **B < 70 % → TESTEN of ARCHIVEREN.** Alle 35 krijgen **TESTEN**, gegroepeerd
  naar de app die ze bereikt, met de motivering in §"Waarom geen enkele
  B-module ARCHIVEREN krijgt". Zestien van de 35 hangen aan
  `apps/live_paper_trader.py` — de live-orderweg archiveren tijdens een
  opruimfase zou fence 5 schenden.
* **D → CONTRACT of AMBITIE.** Alle 20 dragen een aanwijsbare afnemer (een
  DI-nummer, een preregistratie of een besluit) en krijgen **BEHOUDEN**; geen
  enkele is als ambitie geclassificeerd. Zij staan met hun afnemer in
  `docs/CODE_REGISTER.md` §"Klasse D — test-only, met hun afnemer". `registry/
  lifecycle.py` is daarvan het schoolvoorbeeld: DI-15 houdt hem levend.

---

## Criterium 5 — dekking is niet gedaald

> *Elke module in klasse A of B zit boven de drempel of staat onder een
> geregistreerde ratchet met datum en eigenaar. Nul modules in een derde
> categorie.*

```bash
D:/venv/tradebot/Scripts/python.exe -m pytest -m "not slow and not regression" \
    -p no:randomly --cov=src/tradebot --cov-report=json --cov-report=term -q
```

| | nulmeting (`26cfd90`) | nameting (`HEAD`) | Δ |
|---|---:|---:|---:|
| statements | 23.991 | 23.800 | −191 |
| gedekt | 13.391 | **13.480** | **+89** |
| **totaal** | **55,82 %** | **56,64 %** | **+0,82 pp** |
| LOC-gewogen over modules | 60,3 % | **60,6 %** | +0,3 pp |

**De dekking is niet gedaald; zij is gestegen** — en op de manier die telt: er
zijn 89 statements méér gedekt, niet alleen 191 ongedekte statements
verdwenen. Was de stijging louter het gevolg van het weggooien van ongedekte
code, dan zou de teller gelijk zijn gebleven.

**Bewijs:** `reports/phase9_coverage_baseline.md` (vóór) tegen
`reports/phase9_coverage_after.txt` en `reports/phase9_coverage_after.json`
(na).

De drempel `[tool.coverage.report] fail_under = 70` wordt **niet gehaald**. Dat
is een gemeten feit en het staat hier in plaats van in een verlaagde drempel;
de fase-opdracht sluit dat laatste expliciet uit.

### Nul modules in een derde categorie

Nagerekend op de nameting, met de kaart opnieuw gedraaid tegen de nieuwe
dekkingsdata:

```bash
D:/venv/tradebot/Scripts/python.exe scripts/reachability_map.py \
    --coverage reports/phase9_coverage_after.json \
    --json reports/phase9_reachability_after.json
```

| | modules |
|---|---:|
| klasse A + B | **260** |
| ├─ boven de drempel (≥ 70 %) | 152 |
| └─ onder een geregistreerde ratchet | **108** |
| **modules in een derde categorie** | **0** |

De sluitende controle is de laatste regel, en die is omgekeerd gedaan: niet
"staat elke ratchet-regel in het register" maar **"staat elke module onder de
drempel in het register"**. Dat is de richting waarin de fout zou zitten.

```
A+B below 70 (de ratchetpopulatie):        108
below 70 maar NIET in het register:          0
in het register maar inmiddels >= 70 %:      2
   selection/__init__.py  100,0 %
   selection/mda.py        96,2 %
```

`docs/CODE_REGISTER.md` noteert **110**; er staan er nu **108** onder. Het
verschil is geen boekhoudfout maar het resultaat van stap 11: het register is
geschreven op de nulmeting, en de tests voor `selection` — het pakket dat
**nul** tests had — hebben die twee modules boven de drempel getild. Een
ratchet die krimpt omdat er dekking is bijgeschreven, doet wat hij moet doen.

Per module noemt het register de klasse, de LOC, de dekking, het entrypoint dat
hem bereikt en een **eigenaar** (Research of Engineering), gesorteerd op
`LOC × (1 − dekking)` — de volgorde waarin bijschrijven het meeste oplevert.
Herzieningsdatum **2027-03-04**. Bovenaan: `train/ensemble.py` (1.117 LOC,
8,3 %), `labeling/meta.py` (1.181, 18,0 %), `features/regime.py` (1.056,
11,9 %).

Buiten A en B — waar het criterium niet over gaat — staan nog vier modules
onder de drempel: de twee geregistreerde E-modules (`legacy_sizing.py`,
`covariance.py`, beide 0 % en beide met een bestaansassertie) en twee
D-modules, `bars/dollar.py` (45,5 %) en `data/funding.py` (21,4 %), die hun
afnemer in het register hebben staan.

---

## Criterium 6 — de omgeving is de lock

> *`pip freeze` en `requirements-dev.lock` verschillen nergens. `pytest-cov` en
> `pytest-randomly` draaien. De suite is groen onder willekeurige testvolgorde.*

```bash
D:/venv/tradebot/Scripts/python.exe -m pytest --version   # 9.1.1
D:/venv/tradebot/Scripts/python.exe -m mypy --version     # 2.3.1 (compiled: yes)
D:/venv/tradebot/Scripts/python.exe -m ruff --version     # 0.15.12
D:/venv/tradebot/Scripts/python.exe -c "import pytest_cov, pytest_randomly, coverage"
```

```
Python 3.13.0
pytest 9.1.1
mypy 2.3.1 (compiled: yes)
ruff 0.15.12
pytest_cov 7.1.0
pytest_randomly OK
coverage 7.15.4
```

Veldsgewijs tegen beide locks, met genormaliseerde pakketnamen:

```
requirements.lock      : 144 pins | missing=0 diff=0
requirements-dev.lock  :  19 pins | missing=0 diff=0
venv-only packages     :   0
```

> Eén ogenschijnlijk verschil is een parseerartefact en geen versieverschil:
> de lockregel `pywin32==312 ; sys_platform == "win32"` draagt een
> environment-marker. Geïnstalleerd is 312, gelockt is 312. Een vergelijker die
> de marker niet afsnijdt, meldt hier een fantoom.

### De suite is groen onder willekeurige testvolgorde

Dit deel van het criterium was in stap 2 expliciet doorgeschoven naar stap 16
(*"de willekeurige-volgordemeting is exit-criterium 6 en volgt apart"*). Hier
is zij:

```bash
D:/venv/tradebot/Scripts/python.exe -m pytest -q -p randomly
```

| | vaste volgorde (`-p no:randomly`) | **willekeurige volgorde (`-p randomly`)** |
|---|---:|---:|
| collected | 2885 | **2885** |
| passed | 2858 | **2858** |
| failed | 4 | **4** |
| skipped | 23 | **23** |
| errors | 0 | **0** |

Dezelfde vier node-id's, `diff` van beide `FAILED`-verzamelingen is leeg.

Dat de volgorde daadwerkelijk is geschud, blijkt uit de uitvoer zelf: de
`FAILED`-regels komen in een andere volgorde binnen (KG-B2, KG-B2, KG-B1, KG-B3
tegen KG-B1, KG-B2, KG-B2, KG-B3 bij de vaste volgorde). Een "willekeurige"
run die exact dezelfde regelvolgorde oplevert, heeft niets geschud.

**Nul volgorde-afhankelijke tests.** Dat is de garantie die een fase die tests
verplaatst en modules verwijdert nodig heeft, en die vóór deze fase nooit was
getoetst omdat het plugin op de verkeerde interpreter werd gezocht.

**Bewijs:** `reports/phase9_random_order.txt`.

### De lock, en waarom stap 2 een bevinding werd

**Dit criterium was vóór aanvang van de fase al waar.** De zeven bevindingen
onder de kop *"de meetinstrumenten zelf zijn stuk"* — ontbrekende `pytest-cov`,
`pytest-randomly`, `pandas` 2.2.3 tegen 2.3.3, `mypy` 1.17.0 tegen 2.3.1 — zijn
allemaal metingen op de **systeeminterpreter**, en die is sinds 2026-08-27 niet
de interpreter waarop een meting in dit project geldt. Op de referentievenv
klopt elke pin. Stap 2 is daarmee geen herstelactie geworden maar een
bevinding: *het gereedschap was niet stuk, de meting was op de verkeerde
interpreter gedaan.* Commit `26cfd90`,
`reports/phase9_environment_audit.md`.

Dat is geen semantisch verschil. Was de herinstallatie blind uitgevoerd zoals
de opdracht voorschreef, dan was een werkende omgeving vervangen op grond van
een verkeerd gelezen meting.

---

## Criterium 7 — de poorten staan en blokkeren

> *`.github/workflows/inventory.yml` faalt aantoonbaar op een bewust
> toegevoegde onbereikbare module. De LOC-poort noemt geen niet-bestaande
> bestanden meer.*

`.github/workflows/inventory.yml` (commit `736a988`) draait zes poorten, alle
blokkerend — `run:` zonder `continue-on-error`:

| # | Poort | Uitkomst op `HEAD` |
|---:|---|---|
| 1 | `reachability_map.py --strict` | EXIT=0 |
| 2 | `check_file_size.py` | EXIT=0 — *"niets boven 800 regels zonder cap, niets boven zijn cap (9 gecapte bestanden)"* |
| 3 | `check_hardcoded_params.py --strict` | EXIT=0 — **302** literals in 98 bestanden, ratchet 317, geen bestand boven zijn budget |
| 4 | `audit_fallbacks.py --strict` | EXIT=0 — 39 bevindingen, **blokkerend: 0** |
| 5 | `check_banned_methods.py --strict` | EXIT=0 |
| 6 | `ruff check src/` | EXIT=0 |

Poort 3 staat op 302 van 317. De opdracht eist dat dit budget niet stijgt; het
is gedaald.

### Alle vijf poorten kunnen rood worden

Niet één, alle vijf, elk op een échte overtreding van precies datgene waartegen
de poort bestaat. Uitgevoerd in wegwerpbranch `phase9-gate-proof` op `736a988`,
daarna verwijderd:

| Poort | Overtreding | Exit |
|---|---|---:|
| `reachability_map.py --strict` | module toegevoegd die door niets wordt geïmporteerd | **1** |
| `check_file_size.py` | één regel bij `data_adequacy.py` (cap 807 → 809 regels) | **1** |
| `check_hardcoded_params.py --strict` | literal `0.0731` in `risk/engine.py`, budget 0 | **1** |
| `audit_fallbacks.py --strict` | `try: import numpy / except ImportError` in `risk/engine.py` | **1** |
| killgate-vergelijking | positieve én negatieve controle op de vier node-id's | **1 / 0** |

De vijfde regel is de belangrijkste: de killgate-poort geeft exit 0 op de
schone boom en exit 1 wanneer de vergelijking iets te melden heeft. Een
vergelijking die altijd groen is, vergelijkt niet.

**Bewijs:** `reports/phase9_gate_proof.txt`, commit `dd9451d`.

### De LOC-poort noemt geen spookbestanden meer

De whitelist in de `Makefile` noemde `risk/portfolio.py`. Dat bestand bestaat
niet:

```bash
$ ls src/tradebot/risk/portfolio.py
ls: cannot access 'src/tradebot/risk/portfolio.py': No such file or directory
```

Erger dan de spookregel was de vorm: de poort stond als shell-lus in een
`Makefile`, en `make` bestaat niet in deze omgeving. **De poort is nooit
uitgevoerd.** Dat is dezelfde faalvorm als `make coverage`. De vervanger is
`scripts/check_file_size.py` met een cap per bestand, getoetst door
`tests/unit/test_file_size_ratchet.py` en afgedwongen door de workflow.
Acht bestanden dragen nu een `# LOC-EXCEPTION:`-regel met hun reden;
`portfolio/legacy_sizing.py` is uitgezonderd omdat DI-10 hem ongewijzigd houdt.

### Twee poorten stonden zelf niet in de runbook

Bij het narekenen van stap 12 in deze stap bleken **twee entrypoints door niets
te worden genoemd** — precies de toestand die stap 12 had opgeheven:

```
scripts/check_file_size.py   (114 LOC, toegevoegd in stap 13)
scripts/clean.py             (131 LOC, toegevoegd in stap 14)
```

Beide zijn ná stap 12 ontstaan, en stap 12 had de tabel in `docs/runbook.md`
§0.7 toen al geschreven. `check_file_size.py` stond nog wel in CI en in DI-3;
`clean.py` stond **nergens** buiten zijn eigen test. Dat laatste is de
vervelendste soort: het is het enige gereedschap in deze repository dat
bestanden verwijdert die **niet** in versiebeheer staan en dus niet met
`git show` terug te halen zijn.

Beide zijn alsnog in §0.7 opgenomen, `clean.py` met de waarschuwing erbij, en
de kop boven de tabel is gecorrigeerd van *"de eerste vier zijn POORTEN"* naar
vijf met een expliciete **poort**-markering per regel — die vier klopten al
niet meer met de zes stappen in de workflow. Nagerekend:

```bash
for f in $(git ls-files apps scripts | grep '\.py$' | xargs -n1 basename); do
  grep -rq "$f" docs/ .github/ dvc.yaml Makefile || echo "UNNAMED: $f"
done
```

Geen uitvoer: **alle 53 entrypoints worden genoemd.**

---

## Criterium 8 — elke verwijdering is omkeerbaar

> *Steekproef van vijf, met output in het exit-rapport.*

`reports/phase9_removal_register.md` (commit `ada34bb`) telt negen regels: 9
modules / 487 LOC. Elk `git show`-commando is uitgevoerd:

| # | Commando | Regels teruggehaald | Register-LOC | sha256 (16) |
|---:|---|---:|---:|---|
| 1 | `git show 5257a52^:src/tradebot/alpha/decay_tracker.py` | 140 | 140 ✅ | `dc144871ff50e319` |
| 2 | `git show e1fa983^:src/tradebot/features/macro.py` | 110 | 110 ✅ | `d0d917d8bee30f1a` |
| 3 | `git show 3d99d45^:src/tradebot/data/sources/cboe.py` | 61 | 61 ✅ | `e3c4a84d0331b0cc` |
| 4 | `git show f9ecba3^:src/tradebot/logging_config.py` | 61 | 61 ✅ | `b2baefccae3ffe95` |
| 5 | `git show 4f12430^:src/tradebot/types.py` | 25 | 25 ✅ | `d3b873b1a2f53066` |

Alle vijf geven exit 0 en leveren het bestand met **exact** het aantal regels
dat de LOC-kolom van het register noemt. De sha256 staat erbij zodat een
volgende lezer kan controleren dat hij hetzelfde bestand terugkrijgt als wat
hier is gemeten.

De negen verwijderingen staan in acht commits, één per pakket
(`5257a52` alpha ×2, `ca58de9` bars ×2, `3d99d45` data, `e1fa983` features,
`4f12430` types + `_version`, `f9ecba3` logging_config), zodat het register per
bestand naar één sha kan wijzen.

---

## Criterium 9 — DI-3, DI-4 en DI-8

> *Gesloten of met een gecorrigeerde meting herbevestigd. Geen enkele DI is
> verdwenen.*

```bash
diff <(git show f7702dd:docs/DEFERRED_ISSUES.md | grep -oE 'DI-[0-9]+' | sort -u) \
     <(grep -oE 'DI-[0-9]+' docs/DEFERRED_ISSUES.md | sort -u)
```

Leeg. **Alle 22 DI-nummers (DI-1 t/m DI-22) staan er nog**, conform fence 6.

| | Status | Wat er is gebeurd |
|---|---|---|
| **DI-8** | **GESLOTEN** | 48 onderzoeksscripts met `git mv` naar `research/`, historie mee. 77 verwijzingen in 35 bestanden bijgewerkt, waaronder 32 in de wave-documentatie — dat was de voorwaarde waarop de post sinds Phase 0 doorschoof. Elf platformgereedschappen blijven in `scripts/`. Klasse-neutraal gemeten: C = 8 modules / 1.108 LOC vóór én na. Commit `43a1ddf`. |
| **DI-3** | **herbevestigd, meting gecorrigeerd** | De post claimde *"nog 2 bestanden >800 LOC"*; gemeten zijn het er **9**, en de twee genoemde stonden toen al op 1.181 en 1.054. De ratchet die *"verdere groei blokkeert"* bestond niet — hij stond in de `Makefile`. Nu: `scripts/check_file_size.py`, afgedwongen door CI, met bewijs dat hij rood wordt op één toegevoegde regel. |
| **DI-4** | **herbevestigd, meting gecorrigeerd** | De post claimde *"18 van 40 apps"* boven 80 LOC; gemeten **27 van 40** — het aandeel is van 63 % naar 68 % **gestegen**. Ook het voorbeeld klopte niet: `apps/freeze_monitoring.py` telt 82 regels, niet 63, en overschrijdt de limiet dus zelf. Alle 40 apps staan nu met LOC, DVC-stage en doel in `docs/runbook.md` §0.7. |

Drie DI's droegen dus een **onjuist getal**, en in twee gevallen was de
maatregel die hen zou moeten bewaken er niet. Dat is de bevinding, niet het
opruimen.

---

## Criterium 10 — de fences staan

> *`git diff --stat <start-sha>..HEAD -- data/pit_store artefacts/governance
> reports/` toont nul verwijderde regels in `data/pit_store/` en
> `artefacts/governance/`.*

```bash
git diff --stat f7702dd..HEAD -- data/pit_store artefacts/governance
```

Leeg. Geen enkel bestand in beide paden is aangeraakt. Scherper gemeten, per
bestandsnaam in plaats van per regel:

```bash
diff <(git ls-tree -r --name-only f7702dd -- artefacts/governance) \
     <(git ls-tree -r --name-only HEAD    -- artefacts/governance)   # leeg
diff <(git ls-tree -r --name-only f7702dd -- data/pit_store) \
     <(git ls-tree -r --name-only HEAD    -- data/pit_store)          # leeg
```

| Fence | Bestanden bij `f7702dd` | Bij `HEAD` | Verwijderde regels |
|---|---:|---:|---:|
| `data/pit_store/` (fence 1, AD-12) | 228 | **228** | **0** |
| `artefacts/governance/` (fence 2, append-only) | 15 | **15** | **0** |

Identiek, bestand voor bestand — geen toevoegingen en geen verwijderingen.

> **Correctie op een eerdere fase-notitie.** Commit `2542d62` noemde
> *"artefacts/governance 16 bestanden"*. Getrackt zijn het er **15**, bij
> aanvang én nu. Het getal 16 was een teltout; de fence zelf staat.

Fence 4 (de vier rode killgates) is criterium 1. Fence 5 (de divergentie) is
in kaart gebracht en **niet opgelost** — zie hieronder. Fence 7 (geen
gedragswijziging) is criterium 2: 484 van 485 baselinevelden identiek.

Ook `reports/` heeft nul verwijderde bestanden: de negen
`portfolio_metrics_*.json`-varianten en elk `phase*_exit_report.md` staan er nog
(fence 3).

---

## Wat de fase bewust NIET heeft gedaan

De opdracht is even scherp over wat er niet mag gebeuren als over wat er moet.

1. **De divergentie live/backtest is gekwantificeerd, niet opgelost** (fence 5,
   stap 15, commit `d314282`). Van de **21 modules / 6.173 LOC** in `live/` en
   `oms/` staat er **nul** in de authoritative afsluiting. Het HALT-pad ligt op
   de soevereine laag — `HaltStore`, `risk/limits.py`,
   `daily_loss_governor.py` worden elk één keer geïmporteerd — maar het
   BESLUIT-pad niet: **nul** verwijzingen naar `RiskDecision`, `RiskEngine` of
   `execution/order_router.py`. Dat is nauwkeuriger dan het *"nul
   verwijzingen"* dat `PROJECT_STATE.md` §3.2 tot nu toe noteerde, en het is
   bijgewerkt. Twee featurestacks, twee impactmodellen, twee barresoluties:
   beschreven, met verwijzing naar openstaand besluit 0 en 3. **Geen van beide
   stacks is aangeraakt.**
2. **`alpha/cm_carry.py` en `alpha/cm_tsmom.py` zijn niet verhuisd** hoewel zij
   klasse C zijn (fence 4). Zij dragen een pre-geregistreerde poort.
3. **Geen enkele drempel, parameter, formule of signatuur heeft een andere
   waarde gekregen** (fence 7). Criterium 2 is daar de meting van.
4. **Geen enkele B-module is gearchiveerd** en **geen enkele D-module is als
   ambitie afgevoerd.** Beide keuzes stonden open in de beslistabel; beide zijn
   naar behouden uitgevallen, met motivering per groep. Bij twijfel behouden.

---

## Nieuwe openstaande post

De dekking staat op de gemeten waarde hierboven tegen een drempel van 70, met
**110 modules / 32.041 LOC** onder een ratchet. Dat is geen bijvangst van een
opruiming maar werk op zichzelf, en het raakt de volgende stap rechtstreeks:
`live/` draagt 38,5 % en `oms/router.py` — de weg waarlangs in productie een
order de deur uit gaat — **27,9 %**. De post staat in `docs/PROJECT_STATE.md`
§6 vóór stap 2, met de ratchet als bewaker en 2027-03-04 als
herzieningsdatum.

---

## Commits van deze fase

26 commits, `f7702dd..HEAD`, 104 bestanden, +12.543 / −5.266.

| Stap | Commit | Onderwerp |
|---|---|---|
| 1 | `cfeddf0` | gedragsvingerafdruk vóór de opruiming |
| 2 | `26cfd90` | de meetinstrumenten zijn niet stuk, de meting was fout |
| 3 | `6e63345` | dekkingsnulmeting per module |
| 4 | `5b8e09b` | falende spec voor de bereikbaarheidsscanner |
| 5 | `97382cd` | AST-gebaseerde scanner |
| 6 | `b0ac67d` | gemeten kaart + false-positive-audit |
| 7 | `f8acf67` | beslistabel over alle 299 modules |
| 8 | `5257a52` … `f9ecba3` | klasse E opruimen, één commit per pakket |
| 8 | `ada34bb` | verwijderregister met herstelcommando per bestand |
| 9 | `43a1ddf` | `research/`-track (sluit DI-8) |
| 10–11 | `54ce8d4`, `916d1f9`, `515b8f5`, `bdca3df` | register, ratchets, `selection` 0 → 96 %, `oms` |
| 12 | `db00c98` | entrypoints: 36 ongenoemd van 98 wordt 0 |
| 13 | `736a988`, `dd9451d` | blokkerende inventarispoort + bewijs dat vijf poorten rood worden |
| 14 | `2542d62` | 120,9 MB caches, vier bestanden uit het pakket |
| 15 | `d314282` | authoritative/live-overlap gekwantificeerd |
| 16 | *dit rapport* | vingerafdruk vergeleken, dekking nagemeten, volgorde-onafhankelijkheid getoetst, exit-rapport |

---

## Oordeel

Alle tien exit-criteria zijn aantoonbaar waar, elk met een commando en zijn
output hierboven. De drie criteria waarvoor de harde regel expliciet bewijs
eist dat de controle **rood kan worden** — 3, 7 en 8 — dragen dat bewijs:
`reports/phase9_gate_proof.txt` voor 3 en 7, en vijf uitgevoerde
`git show`-commando's voor 8.

De opbrengst in LOC is klein: 487 regels weg op 71.487. De opbrengst in kennis
is dat niet. Van de 39 modules die als onbereikbaar te boek stonden, bleken er
**28 een echte import te dragen**; van de zeven "kapotte meetinstrumenten"
bleken er **zeven op de verkeerde interpreter gemeten**; en drie DI's droegen
een getal dat niet klopte. Elk van die vier bevindingen zou tot een verkeerde
handeling hebben geleid als de opdracht op zijn woord was uitgevoerd in plaats
van nagemeten.

Dat is wat de regel van deze fase voorschreef: *opruimen is niet het
verwijderen van wat je niet herkent, maar het aantoonbaar maken van wat er
staat.*
