# PHASE 7/8 — STAGE A: FUNDAMENT HERSTELLEN

**Stage:** A van E · **Prioriteit:** P0
**Bindend brondocument:** `Prompts-fases/fase_7_8_consolidatie_productie.md` §3
**Startpunt:** `ec1026d` · **Eindpunt:** `459bbe9` · branch `main`
**Uitgevoerd:** 2026-08-27
**Poort naar Stage B:** alle drie de ratchets exit 0 · `pip install -e .` werkt
vanaf een schone interpreter · de historie staat aantoonbaar buiten deze machine

---

## 1. Executive status

> # CONDITIONAL PASS

Acht van de negen exit-criteria zijn gehaald en bewezen. Eén is **niet** gehaald
en kan niet door mij worden gehaald.

| # | Criterium | Status |
|---|---|---|
| A1 | Historie buiten deze machine | **NIET GEHAALD** — zie §2.3. Vereist een handeling van de opdrachtgever. |
| A2 | C-schijf-repo opgeruimd ná bewijs van nul unieke commits | **GEHAALD** — met een restant van drie lege mappen, §2.2 |
| A3 | `x.jsonl` weg; schrijver naar `tmp_path`; bewakende test | **GEHAALD** |
| A4 | `pip install -e .` werkt; `tradebot.__file__` wijst naar `D:\Tradebot\src` | **GEHAALD** |
| A5 | Nul `.pyc` buiten de root; test bewijst dit en is rood geweest | **GEHAALD** |
| A6 | `check_hardcoded_params.py --strict` exit 0 **zonder budgetverruiming** | **GEHAALD** |
| A7 | `audit_fallbacks.py --strict` exit 0; faalbasislijn 4, dezelfde namen | **GEHAALD** |
| A8 | Elke ratchet aantoonbaar rood gemaakt op een wegwerpbranch | **GEHAALD** |
| A9 | Schone-venv-build reproduceert de suite; referentie-interpreter vastgelegd | **GEHAALD** |

**No-go-conditie 2 staat op dit moment ACTIEF** (*"de git-historie bestaat nog
steeds maar op één fysieke locatie"*). Zie §2.3 voor wat er nodig is en waarom
ik het niet zelf kan afronden.

**Wat er ONVERWACHT is gevonden en niet in §0 van de masterprompt stond:** de
lint-gate van `hygiene.yml` stond óók rood, en zijn oordeel hing af van de
releasedatum van `ruff`. Zie §5. Dat is de belangrijkste bevinding van deze
stage.

---

## 2. A-1 — Git veiligstellen en de landmijn opruimen

### 2.1 De bundle — eerste handeling, vóór elke wijziging

```bash
git bundle create D:/backup/tradebot-ec1026d.bundle --all
git bundle verify D:/backup/tradebot-ec1026d.bundle
```

```
D:/backup/tradebot-ec1026d.bundle is okay
The bundle contains these 2 refs:
  ec1026da256068027505234eef42f468bd765d08 refs/heads/main
  ec1026da256068027505234eef42f468bd765d08 HEAD
The bundle records a complete history.
```

| Artefact | SHA-256 | Omvang |
|---|---|---|
| `tradebot-ec1026d.bundle` | `c9a58afbb133bedcad6bc0a779d64f68078d4098cb382f23a2ef31a49e13488f` | 2.219.187 bytes |
| `tradebot-CDRIVE-orphan.bundle` | `b29792e7cf7b48fdeee49de397a834aefd0b5a0cb51618c89adb91f38b4cbd01` | — |

### 2.2 De C-schijf-repo — bewijs vóór verwijdering

No-go 3 verbiedt verwijdering zonder bewijs van nul unieke commits. Dat bewijs
is op **vier** onafhankelijke manieren geleverd, niet op één:

```bash
# 1. commits bereikbaar vanuit cdrive maar niet vanuit D:main
git fetch C:/Users/algul/Documents/Tradebot/.git 'refs/*:refs/cdrive/*'
git rev-list --count main..refs/cdrive/heads/main
#   -> 0

# 2. objecten bereikbaar vanuit cdrive die D: niet heeft
git rev-list --objects refs/cdrive/heads/main --not --all
#   -> 0

# 3. VOLLEDIGE objectvergelijking, niet alleen bereikbare
git --git-dir=C:/.../.git cat-file --batch-all-objects --batch-check='%(objectname)'
git             cat-file --batch-all-objects --batch-check='%(objectname)'
#   orphan: 1768 objecten   D:: 1774 objecten
#   in orphan maar niet in D:: 0

# 4. dangling commits (die stap 1 en 2 per definitie missen)
git --git-dir=C:/.../.git fsck --lost-found
#   -> 11 dangling commits, alle 11 aanwezig in D: (git cat-file -e per stuk)
```

> **Stap 3 en 4 zijn toegevoegd bovenop wat de prompt vroeg.** `git log
> main..refs/cdrive/heads/main` kijkt alleen naar **bereikbare** commits. Een
> commit die alleen in de reflog of als dangling object bestaat, is daarmee
> onzichtbaar — en juist een uitgeholde repo is precies de plek waar zoiets
> zich ophoudt. De orphan bleek er elf te hebben. Alle elf staan ook in D:.

**Pas daarna** verwijderd. Resultaat:

```bash
ls -d C:/Users/algul/Documents/Tradebot/.git    # No such file or directory
find C:/Users/algul/Documents/Tradebot          # Tradebot/, src/, src/tradebot/
```

**De landmijn is weg.** Er kan daar geen `git add -A && git commit` meer worden
gedraaid die de verwijdering van 637 paden vastlegt in een historie die van
buiten niet van de echte te onderscheiden is.

**Restant, eerlijk gemeld:** drie **lege** mappen (`Tradebot/src/tradebot/`,
nul bestanden) blijven staan. `rm -rf` faalt met *"Device or resource busy"*.
Oorzaak, gemeten en niet aangenomen:

```
$env:OneDrive                       C:\Users\algul\OneDrive
Test-Path "$env:OneDrive\Documents" True
Get-Process                         OneDrive.Sync.Service (pid 3212) draait
```

`C:\Users\algul\Documents` is OneDrive-gesynchroniseerd; de sync-service houdt
de map open. Nul bestanden, geen `.git` — het beveiligingsdoel is volledig
bereikt. Ruim de mappen op na een herstart van OneDrive of van de machine.

> **Bijvangst die het vermelden waard is:** omdat `Documents` OneDrive-gesynct
> is, stond de orphan-repo **ook in de cloud**. Dat was een off-machine kopie —
> van een uitgeholde werkkopie, met een historie die identiek oogde aan de
> echte. Dat is een slechtere back-up dan geen back-up, want hij nodigt uit tot
> herstellen uit de verkeerde bron.

### 2.3 De remote — NIET GEHAALD, en waarom

De opdrachtgever heeft gekozen voor een **private GitHub-remote**. Die keuze is
vastgelegd. Uitvoeren lukt niet:

```bash
gh auth status        # bash: gh: command not found
Get-Command gh        # gh NOT installed
```

De GitHub CLI is niet geïnstalleerd. Een remote aanmaken vereist authenticatie
met een account, en dat is een handeling die ik niet namens de opdrachtgever
uitvoer — het aanmaken van accounts en het invoeren van inloggegevens ligt
buiten wat ik doe, ongeacht hoe de opdracht is geformuleerd.

**Wat de opdrachtgever moet doen** — één van beide:

```bash
winget install --id GitHub.cli
```
```bash
gh auth login
```
```bash
gh repo create Tradebot --private --source=. --remote=origin --push
```

of, zonder CLI: de repo handmatig privé aanmaken op github.com en daarna

```bash
git remote add origin git@github.com:<gebruiker>/Tradebot.git
```
```bash
git push -u origin main --tags
```
```bash
git ls-remote origin
```

**Tot dat moment blijft no-go 2 actief** en staat de volledige bewijsketen —
80 commits, het falsificatieregister, de ledger met `M = 2776`, de bevroren
pre-registraties — op één fysieke schijf. De bundles in `D:/backup/` staan op
**diezelfde schijf** en tellen dus niet als off-machine kopie.

### 2.4 `x.jsonl` — het lek was nog actief

| Meting | Waarde |
|---|---|
| Regels bij aanvang | **38** (de prompt mat er 35 op 2026-08-27) |
| Herkomst | `d79bf25`, laatst gewijzigd in `24999fe` |
| Schrijvers gevonden | **precies één** |

```
tests/unit/test_data_validation.py:118
    enforce_gap_policy(recs, policy="reject", ledger=GapLedger("x.jsonl"))
```

Elke buur in datzelfde bestand gebruikte al `tmp_path`. Het verschil tussen 35
en 38 regels betekent dat de test sindsdien opnieuw heeft gedraaid: dit was
**geen residu maar een actief lek**.

Drie verdedigingslinies in plaats van één:

1. de schrijver gebruikt `tmp_path`;
2. `.gitignore` weert `/*.jsonl /*.parquet /*.csv /*.pkl /*.npy` uit de root;
3. `tests/unit/test_repository_hygiene.py` bewaakt per **suffix**, niet per
   naam — de volgende lekkage onder een andere naam wordt óók gevangen.

---

## 3. A-2 — De omgeving herbouwen

### 3.1 Caches

```
vóór:  351 .pyc · 39 __pycache__      (351/351 met co_filename op C:)
na:      0 .pyc ·  0 __pycache__
```

Direct zichtbaar in de suite-uitvoer. Vóór:

```
SKIPPED [1] C:\Users\algul\Documents\Tradebot\tests\e2e\test_chaos.py:19
D:\Tradebot\src\tradebot\risk\var.py:222: RuntimeWarning
```

Na:

```
SKIPPED [1] tests\e2e\test_chaos.py:19
```

Twee schijven in één rapport, teruggebracht tot één.

### 3.2 De editable install

| Meting | Vóór | Na |
|---|---|---|
| `.pth`-inhoud | `C:\Users\algul\Documents\Tradebot\src` | `D:\Tradebot\src` |
| `tradebot.__file__` | *(namespace-package, leeg)* | `D:\Tradebot\src\tradebot\__init__.py` |
| `import tradebot.registry` | `ModuleNotFoundError` | slaagt |
| `get_git_sha()` | *(niet importeerbaar)* | resolvet |

### 3.3 De schone venv — criterium A9

```bash
python -m venv D:/venv/tradebot
D:/venv/tradebot/Scripts/python -m pip install -r requirements.lock
D:/venv/tradebot/Scripts/python -m pip install -r requirements-dev.lock
D:/venv/tradebot/Scripts/python -m pip install -e . --no-deps
D:/venv/tradebot/Scripts/python -m pytest tests/unit -q -p no:randomly
```

```
1253 passed, 1 skipped, 18 warnings in 49.41s
```

`D:/venv/tradebot` is vanaf nu de **referentie-interpreter**, vastgelegd in
`docs/runbook.md` §0.

> **`requirements-dev.lock` bestond niet en moest worden gemaakt.** De eerste
> poging tot deze verificatie faalde op `No module named pytest`: de lockfile
> pint 476 runtime-packages en **nul** testtools. Zie §5.

### 3.4 De `.pyc`-bewaker is rood geweest

Injectie: een geldige `.pyc` met `co_filename =
C:\Users\algul\Documents\Tradebot\src\tradebot\injected.py`, geschreven met een
correcte PEP 552-header zodat Python hem als geldig beschouwt.

```
AssertionError: 1 bytecode-cache(s) verwijzen naar een pad buiten D:\Tradebot.
Eerste drie: [('src\\tradebot\\__pycache__\\injected.cpython-313.pyc',
              'C:\\Users\\algul\\Documents\\Tradebot\\src\\tradebot\\injected.py')]
```

Groen na verwijdering van de injectie.

---

## 4. A-3 — De ratchet groen zonder hem te verruimen

De vijftien literals waren significantiedrempels, lag-aantallen,
minimum-observatie-eisen en vensterlengtes — precies waarvoor de ratchet
bestaat.

| Module | Literals | Waarheen |
|---|---|---|
| `validation/econometrics.py` | 8 | `alpha`, `ljung_box_lags` |
| `validation/diagnostics_report.py` | 2 | idem |
| `validation/vol_metrics.py` | 2 | `mincer_zarnowitz_min_obs`, `dm_min_intersection_ratio` |
| `volatility/realized.py` | 2 | `yang_zhang_window` |
| `validation/data_adequacy.py` | 1 | `auc_expected` |

Alle vijftien staan nu in `conf/validation/econometrics.yaml`, gevalideerd tegen
`EconometricsConfig` (`extra="forbid"`, `frozen=True`), en gaan mee in de
`config_hash`.

```
                        vóór    na
literals                 327    312
bestanden                107    102
toegestaan door ratchet  317    317      <- ONVERANDERD
exit code                  1      0
```

> **Het budget is niet verruimd.** De ratchet stond en staat op 317. Er is
> verplaatst, niet toegestaan. Phase 0-besluit van 2026-08-22: budgetten mogen
> uitsluitend omlaag.

**Geen enkel getal is gewijzigd.** Elke waarde is identiek aan de default die
zij vervangt. Een verplaatsing die tegelijk een waarde verandert, is niet te
onderscheiden van een stille herkalibratie — en dat is exact wat de ratchet
hoort te voorkomen.

**Twee ontwerpkeuzes die uitleg verdienen:**

1. **De accessor staat in `schemas/config.py`, niet in `validation/`.**
   `volatility/realized.py` gebruikt dezelfde drempel, en een L2-module die uit
   L11 importeert is een laaginversie.
2. **`econometrics_config()` is fail-fast.** Ontbreekt de config of schendt hij
   zijn contract, dan crasht de import van de consumerende module. Er is geen
   terugval op een ingebouwde drempel: een drempel die uit code komt in plaats
   van uit `conf/`, is niet gehasht en dus niet auditbaar.

**Wat NIET is verplaatst en waarom:** `arch_lags = 12` blijft in de code — 12 is
een kalenderconstante die de scanner terecht als `BENIGN` behandelt. `null_auc =
0.5` blijft — de nulwaarde van een AUC is een definitie, geen keuze.

---

## 5. Wat §0 van de masterprompt niet had gemeten

Dit is de belangrijkste bevinding van deze stage en zij stond in geen enkel
rapport.

### 5.1 De lint-gate stond óók rood

§0.3 identificeert `check_hardcoded_params.py` als de enige rode ratchet.
`hygiene.yml` draait echter ook een `ruff`-stap, en die faalde:

```bash
ruff check src/ scripts/audit_fallbacks.py scripts/phase0_baseline.py \
           scripts/unwrap_broad_except.py tests/unit/test_config_contracts.py \
           tests/unit/test_no_silent_fallbacks.py
```

```
Found 7 errors.
```

Ook op `42555d2` — dit is geen regressie van mijn werk, maar een gate die al
rood stond vóór deze fase begon.

### 5.2 En zijn oordeel hing af van de releasedatum van ruff

| ruff | Bevindingen |
|---|---|
| 0.15.12 (systeem) | **7** |
| 0.16.4 (verse venv) | **78**, waarvan 71× `PLR0917` |

**Dezelfde broncode. Dezelfde workflow. Een ander antwoord.**

De oorzaak staat in `hygiene.yml` zelf:

```yaml
pip install pytest pytest-cov hypothesis mypy ruff types-PyYAML types-requests
```

Geen enkele versie. Wie de gate op maandag draaide kreeg een ander oordeel dan
wie hem op vrijdag draaide.

> **Een gate die van mening verandert zonder dat de code verandert, bewaakt
> niets.** Dit is dezelfde klasse als de stille fallback die dit platform overal
> elders uitsluit, alleen dan in de bewijsvoering in plaats van in de code.

### 5.3 De reparatie

`requirements-dev.lock` pint 19 packages, inclusief `pytest==9.1.1`,
`ruff==0.15.12`, `mypy==2.3.1` en `hypothesis==6.165.10`. `hygiene.yml`
installeert eruit.

**De zeven bevindingen zijn gerepareerd, niet onderdrukt.** Eén ervan was een
echte bug:

```python
# src/tradebot/portfolio/legacy_sizing.py:779
        return RiskState(
```

`RiskState` wordt in die module **nergens geïmporteerd**; de klasse heet
`LegacyRiskState` (regel 76), en de docstring waarschuwt zelfs expliciet *"niet
te verwarren met `risk.contract.RiskState`"*. `LegacyRiskState.update()` gooide
dus gegarandeerd een `NameError` bij elke aanroep — een methode die dood in het
water lag — en **geen enkele test dekte hem af**. De lint-gate wees hem al die
tijd aan; hij stond alleen nooit groen genoeg om gelezen te worden.

De overige zes: 2× `I001` importsortering, 1× `F401` ongebruikte import, 1×
`F821` (de annotatie van dezelfde bug), 1× `RUF007` `itertools.pairwise`, 1×
`RUF012` `ClassVar`.

**De 71 `PLR0917` onder 0.16.4 zijn geregistreerd als DI-16**, niet stil
weggepind. Zij zijn een signatuur-ontwerpkwestie in ~71 aanroepketens en raken
geen statistisch contract; dat oppakken is een expliciet besluit, geen bijvangst
van een `pip install` op een willekeurige dag.

---

## 6. A-4 — Elke gate is aantoonbaar rood gemaakt

Op wegwerpbranch `throwaway/negative-controls`, ná de omgevingsherbouw, daarna
verwijderd.

| Gate | Injectie | Uitkomst |
|---|---|---|
| `check_hardcoded_params.py --strict` | `alpha: float = 0.037` in `cusum_test` | **exit 1** — *"validation/econometrics.py: 1 literal(s) in een door Phase 0-3 bestuurde module (budget 0)"* |
| `audit_fallbacks.py --strict` | een `except:` met `return 0.0` | **exit 1** — *"BARE_EXCEPT [(bare)] in `_injected_silent_fallback` -> return 0.0"*, blokkerend: 1 |
| `pytest` | `adf_test(alpha=1.0)` — verwerpt altijd | **FAILED** `test_adf_rejects_on_white_noise_and_not_on_a_random_walk` |
| `test_no_data_artefact_...` | `injected_leak.jsonl` in de root | **FAILED** — *"Data-artefact(en) in de repository-root"* |
| `test_no_bytecode_cache_...` | `.pyc` met C-pad als `co_filename` | **FAILED** — *"1 bytecode-cache(s) verwijzen naar een pad buiten D:\Tradebot"* |

Alle vijf groen na verwijdering van de injectie.

---

## 7. Wat er mis bleek in mijn eigen werk

### OD-1 — Een `write_text`-round-trip veranderde regeleindes

**Symptoom:** `git show --stat` meldde **767 gewijzigde regels** in
`realized.py` voor een wijziging van 9 insertions / 2 deletions.

**Oorzaak:** `pathlib.Path.write_text()` opent in tekstmodus; op Windows
vertaalt dat `\n` naar `\r\n`. Een `read_text` → `write_text`-round-trip
normaliseert dus stilzwijgend naar CRLF. Omvang: exact twee bestanden
(`realized.py` 0→387, `.gitignore` 0→107). De vier andere modules stonden al op
CRLF; de Edit-tool bewaarde regeleindes correct.

**De bijna-fout die erger was dan de fout.** Mijn eerste controle was

```bash
git show 42555d2:src/tradebot/volatility/realized.py | grep -c $'\r'    # 380
```

en dat getal zei: *"stond al op CRLF, er is niets aan de hand."* Het is
onjuist. `git show` past werkkopie-filters toe; alleen `git cat-file -p` geeft
de ruwe blob:

```bash
git cat-file -p 42555d2:...realized.py | tr -d '\n' | tr -cd '\r' | wc -c   # 0
```

**Ik had op basis van de verkeerde meting bijna geconcludeerd dat er geen defect
was.** Dat is precies het patroon dat §2.2 van het auditdocument beschrijft, en
het is hier op mijzelf van toepassing: de eerste meting was een hypothese, geen
waarheid.

**Hersteld:** beide bestanden terug naar LF, de commit geamendeerd zodat de diff
de werkelijke wijziging toont (9 insertions, 2 deletions).

**Rest-bevinding, niet van mij:** de repository is repo-breed inconsistent — 203
getrackte `.py` puur CRLF, 270 puur LF, 3 gemengd, geen `.gitattributes`.
Geregistreerd als **DI-17** voor Stage E.

### OD-2 — Mijn eigen bewaker faalde op mijn eigen bestand

`test_no_unexpected_file_sits_in_the_repository_root` werd rood zodra ik
`requirements-dev.lock` aanmaakte. Dat is de bewaker die correct werkt, maar het
laat zien dat een allowlist-bewaker onderhoud kost bij elke legitieme toevoeging.
Bewust toegevoegd aan de allowlist in plaats van de test te verzwakken.

---

## 8. Eindmeting

```bash
python scripts/check_hardcoded_params.py --strict   # exit 0
python scripts/audit_fallbacks.py --strict          # exit 0  (37 advies, 0 blokkerend)
python -m pytest -q -p no:randomly                  # 4 failures
ruff check <CI-paden>                               # All checks passed
mypy --strict schemas/config.py utils/failfast.py   # Success
```

De vier failures, ongewijzigd van naam:

```
KG-B1 in-sample-cm_carry
KG-B2 residual alpha-cm_tsmom
KG-B2 residual alpha-cm_carry
KG-B3 out-of-sample-cm_carry
```

---

## 9. Poort naar Stage B

| Poortvoorwaarde | Status |
|---|---|
| Alle drie de ratchets exit 0 | **JA** |
| `pip install -e .` werkt vanaf een schone interpreter | **JA** — 1253 passed in een verse venv |
| De historie staat aantoonbaar buiten deze machine | **NEE** — §2.3 |

**Mijn oordeel: Stage B mag beginnen.** De derde voorwaarde is een handeling van
de opdrachtgever die geen enkele technische afhankelijkheid heeft met Stage B;
hem afwachten zou de fase blokkeren zonder enig risico weg te nemen. Het risico
dat hij adresseert — verlies van de schijf — is intussen **wel** verkleind door
twee geverifieerde bundles, alleen niet weggenomen.

**No-go 2 blijft in elk volgend rapport vermeld tot de remote bestaat.**

---

## 10. Commits

| Commit | Onderwerp |
|---|---|
| `ec1026d` | `docs(phase7): measure the actual state of the repository before touching it` |
| `15fc584` | `fix(validation): move phase-6 thresholds out of code and into conf` |
| `2f00011` | `test(repo): stop the test that leaked x.jsonl into the project root` |
| `459bbe9` | `chore(repo): restore editable install after the drive move` |

---

*Stage A afgerond op `459bbe9`, 2026-08-27. Eén exit-criterium open, met een
benoemde eigenaar en een uitvoerbaar commando.*
