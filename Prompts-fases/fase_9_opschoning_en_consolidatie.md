# MASTER-PROMPT: PHASE 9 — OPSCHONING, CONSOLIDATIE & TESTMASSA

> **Fase:** 9, volgt op `fase_7_8_consolidatie_productie.md` · **Prioriteit:** P1
> **Bindende brondocumenten:** `docs/PROJECT_STATE.md` (herzien 2026-09-02), `docs/DEFERRED_ISSUES.md`, `docs/ARCHITECTUUR_AUDIT_2026-08-22.md` §20 (repository-indeling), `reports/phase7_divergence_map.md`
> **Voorwaarde:** Deze fase raakt geen enkele openstaande onderzoeksvraag. Zij verkleint het oppervlak; zij beslist niets.

---

## ROL EN CONTEXT

Je acteert als **Principal Software Architect & Quant Platform Engineer**. Je erft een repository van **71.487 LOC over 299 modules in `src/`**, **28.042 LOC over 140 testbestanden**, **40 apps**, **58 scripts** en zeven afgeronde fasen aan bewijsmateriaal.

Het platform is gebouwd rond één principe: *een resultaat telt pas wanneer het de poort is gepasseerd die het had kunnen tegenhouden.* Vrijwel alle infrastructuur bestaat om claims te weerleggen, niet om ze te produceren.

**Daaruit volgt de enige regel die deze fase echt regeert:**

> **Opruimen in deze repository is niet het verwijderen van wat je niet herkent. Het is het aantoonbaar maken van wat er staat, en het verwijderen van uitsluitend dat waarvan de overbodigheid is gemeten.**

Een verwijderde module die nog een contract droeg, is een stille degradatie met een schoner ogende `git ls-files`. Dat is exact de faalklasse waar Phase 0 de hele fail-fast-doctrine tegen heeft opgetuigd. Een opruimfase die die doctrine schendt is erger dan geen opruimfase.

**Relevante lagen (Target Architecture §19):** alle. Deze fase is cross-cutting en raakt L0 t/m L13.

---

## DOEL VAN DE FASE

Na deze fase geldt:

1. Van **elke** module in `src/`, elke app en elk script is **gemeten** — niet beweerd — door welk entrypoint hij wordt bereikt en hoeveel testdekking hij draagt.
2. Wat niet bereikbaar is en geen contract draagt, is **verwijderd met een register-entry** die vermeldt wat het was, hoeveel LOC, waarom het weg mocht, en uit welke commit het terug te halen is.
3. Wat wél overleeft draagt **ofwel dekking boven de drempel, ofwel een geregistreerde ratchet met een datum**. Er is geen derde optie.
4. De opruiming is **aantoonbaar gedragsneutraal**: dezelfde testuitslag, dezelfde vier rode killgates, dezelfde baseline-metrieken, geen daling in dekking.
5. Een nieuwe onbereikbare module kan er niet ongemerkt bij komen — CI blokkeert dat.

---

## NULMETING (gemeten 2026-09-04, reproduceer deze vóór je iets wijzigt)

Deze getallen zijn geen achtergrond. Zij zijn de basis waartegen je eindtoestand wordt afgerekend. Als jouw eigen meting hiervan afwijkt, is dát je eerste bevinding, en die schrijf je op voordat je verder gaat.

### Bereikbaarheidskaart van `src/` — vijf klassen

| Klasse | Definitie | Modules | LOC |
|---|---|---:|---:|
| **A — authoritative** | bereikbaar vanuit een `dvc.yaml`-stage | 123 | 41.763 |
| **B — operationeel** | bereikbaar vanuit een app die niet in de DAG staat | 81 | 17.163 |
| **C — research** | uitsluitend bereikbaar vanuit `scripts/` | 8 | 1.108 |
| **D — test-only** | uitsluitend bereikbaar vanuit `tests/` | 48 | 6.932 |
| **E — onbereikbaar** | door niets bereikt | 39 | 4.521 |
| | **totaal** | **299** | **71.487** |

De tien DVC-seeds zijn: `apps/data_sync.py`, `build_features.py`, `tune_hparams.py`, `train_cpcv.py`, `run_phase5_baseline.py`, `run_data_adequacy.py`, `run_econometric_diagnostics.py`, `run_vol_competition.py`, `run_regime_benchmark.py`, `run_meta_labeling.py`.

### Gedragsvingerafdruk (`python -m pytest -q`, 2026-09-04)

**2770 passed · 4 failed · 23 skipped.** De vier failures zijn exact de pre-geregistreerde killgates en zij horen rood te staan:

```
tests/killgates/test_expansion_killgates.py::test_futures_unit_against_gate[KG-B1 in-sample-cm_carry]
tests/killgates/test_expansion_killgates.py::test_futures_unit_against_gate[KG-B2 residual alpha-cm_tsmom]
tests/killgates/test_expansion_killgates.py::test_futures_unit_against_gate[KG-B2 residual alpha-cm_carry]
tests/killgates/test_expansion_killgates.py::test_futures_unit_against_gate[KG-B3 out-of-sample-cm_carry]
```

Dit viertal — niet "vier failures", maar déze vier node-id's — is de vingerafdruk waar de fase op wordt afgerekend. De 23 skips zijn eveneens onderdeel van de afdruk: elf ervan slaan over omdat een killgate-artefact of een datacache ontbreekt, en een opruiming die zo'n artefactpad breekt verandert een skip in een error zonder dat het totaal aantal failures beweegt.

### Entrypoints

* **36 van de 98** entrypoints worden door niets genoemd — niet door `dvc.yaml`, niet door de `Makefile`, niet door `[project.scripts]`, niet door `.github/workflows/`, niet door `docs/`. Uitgesplitst: **12 van de 40 apps** en **24 van de 58 scripts**.
* **27 van de 40 apps** staan boven de 80-LOC-limiet uit `architecture.md` R-6. `DEFERRED_ISSUES.md` DI-4 noteert 18 van 40. **Die meting is verouderd.**

### De meetinstrumenten zelf zijn stuk

Dit is de belangrijkste bevinding van de nulmeting en zij gaat vóór alle andere stappen.

| Bevinding | Bewijs |
|---|---|
| `pytest-cov==7.1.0` staat in `requirements-dev.lock` maar is **niet geïnstalleerd** | `python -c "import pytest_cov"` → `ModuleNotFoundError` |
| `make coverage` en `[tool.coverage.report] fail_under = 70` zijn daarmee **nooit uitgevoerd** | idem |
| `make` bestaat niet in deze omgeving; de `Makefile` is documentatie, geen poort | `make: command not found` |
| `make loc-check` zou **rood** staan op 6 bestanden | `backtest/evaluation.py` 1054, `features/regime.py` 1056, `live/engine.py` 913, `portfolio/legacy_sizing.py` 824, `schemas/config.py` 1094, `validation/data_adequacy.py` 804 |
| De LOC-whitelist noemt `risk/portfolio.py` — **dat bestand bestaat niet** | `ls src/tradebot/risk/portfolio.py` |
| DI-3 claimt "gemeten 2026-09-01: nog 2 bestanden >800 LOC" | Gemeten 2026-09-04: **9**. De ratchet houdt niets tegen |
| De geïnstalleerde omgeving is niet de gelockte omgeving | `pandas` 2.2.3 vs lock 2.3.3 · `catboost` 1.2.8 vs 1.2.10 · `pytest` 9.0.2 vs 9.1.1 · `mypy` 1.17.0 vs 2.3.1 · `pytest-randomly` 4.1.0 **ontbreekt** |

> `pytest-randomly` ontbreekt betekent dat testvolgorde-onafhankelijkheid nooit is getoetst. Een opruimfase die tests verplaatst of verwijdert, heeft juist die garantie nodig.

Phase 0 exit-criterium 2 luidde *"100 % reproduceerbare build"*. Dat criterium is gedrift. Je herstelt het vóór je één regel opruimt, want elke meting in deze fase wordt met dit gereedschap gedaan.

### Testmassa per pakket (statisch: aantal testbestanden dat het pakket noemt)

| Pakket | src LOC | testbestanden | Oordeel |
|---|---:|---:|---|
| `selection` | 293 | **0** | geen enkele test |
| `reporting` | 2.259 | **1** | 2.259 LOC achter één bestand |
| `tune` | 1.222 | 1 | |
| `compliance` | 1.015 | 1 | |
| `bars` | 969 | 1 | |
| `featurestore` | 268 | 1 | |
| `train` | 4.849 | **3** | grootste absolute gat |
| `portfolio` | 2.661 | 4 | |
| `oms` | 1.273 | 4 | live-orderweg |
| `risk` | 4.671 | 28 | goed gedekt |
| `schemas` | 1.657 | 48 | goed gedekt |
| `utils` | 918 | 58 | goed gedekt |

### Werkboom-hygiëne

`.mypy_cache` telt **2.773 bestanden / 106 MB**, `.hypothesis` 555 bestanden, `logs/` 68 bestanden / 5 MB. Alle ongetrackt, dus geen git-probleem — wel een `find`/`grep`/IDE-probleem en een bron van valse treffers in elke scan die je in deze fase draait.

---

## DE BESLISTABEL

Dit is de kern van de fase. Elke module, app en script krijgt precies één rij. Raden is verboden; de klasse komt uit de scanner, de dekking uit `coverage`.

| Klasse | Dekking | Verdict | Toelichting |
|---|---|---|---|
| **A** | ≥ 70 % | **BEHOUDEN** | staat in de DAG en is gedekt; niets te doen |
| **A** | < 70 % | **TESTEN** | het draait in de authoritative keten. Verwijderen is uitgesloten; je schrijft dekking bij tot de drempel, of je registreert een ratchet met een datum en een eigenaar |
| **B** | ≥ 70 % | **BEHOUDEN** | operationeel entrypoint, gedekt |
| **B** | < 70 % | **TESTEN of ARCHIVEREN** | expliciete keuze per module, met naam van het entrypoint dat hem gebruikt. Geen keuze maken is geen uitkomst |
| **C** | n.v.t. | **VERPLAATSEN naar `research/`** | sluit DI-8 — **tenzij** de module een pre-geregistreerde killgate draagt (zie fence 4) |
| **D** | n.v.t. | **CONTRACT of AMBITIE** | zie hieronder; dit is de moeilijkste klasse |
| **E** | n.v.t. | **VERWIJDEREN** | met register-entry — **behalve** de false positives hieronder |

### Klasse D is geen restcategorie

48 modules / 6.932 LOC worden uitsluitend door hun eigen tests bereikt. Dat betekent één van twee dingen, en het verschil is niet cosmetisch:

* **Contract dat op bedrading wacht.** `registry/lifecycle.py` (310 LOC) implementeert `SymbolLifecycle` — `DEFERRED_ISSUES.md` DI-15 zegt letterlijk *"Het contract is wel gebouwd en weigert bars voor de listing of na de delisting, dus de data plugt in zodra hij er is."* Dit verwijderen sluit een openstaand **inkoopbesluit** met een `git rm`. **Behouden, met registerregel.**
* **Ambitie zonder afnemer.** Een module die niemand ooit heeft aangesloten en waarvoor geen open DI, geen preregistratie en geen besluit bestaat. **Archiveren.**

Je bewijst per module welke van de twee het is door te zoeken naar de afnemer: een open DI-nummer, een preregistratie in `artefacts/governance/`, een besluit in `docs/ARCHITECTURAL_DECISIONS.md`, of een expliciete verwijzing in `PROJECT_STATE.md` §5/§6. Vind je er geen, dan is het ambitie.

Let in het bijzonder op dat de helft van `risk/` in klasse D valt — `var.py` (350), `hmm_regime.py` (270), `kelly.py` (170), `factor_risk.py` (179), `drawdown.py` (149), `liquidity_risk.py` (148), `position_limits.py` (105), `beta_hedge.py` (104). `PROJECT_STATE.md` §2 noemt de risicolaag "soeverein in de BACKTEST". Soeverein en uitsluitend door tests bereikt is een spanning die je in het inventarisrapport benoemt, niet oplost.

### Bekende false positives van de scanner

De scanner meet import-bereikbaarheid. Drie klassen treffers zijn daarom onbetrouwbaar en mogen **nooit** op grond van de scanner alleen worden verwijderd:

1. **`__init__.py`-bestanden** verschijnen als onbereikbaar wanneer consumenten submodules direct importeren. In de nulmeting zijn dat er **15**: `backtest/`, `compliance/`, `data/`, `data/sources/`, `execution/`, `features/`, `labeling/`, `oms/`, `reporting/`, `schemas/`, `selection/`, `train/`, `tune/`, `utils/`, `validation/`. Beoordeel per bestand of het re-exports bevat die iemand gebruikt; `backtest/__init__.py` (152 LOC) en `train/__init__.py` (110 LOC) zijn te groot om een leeg pakketmarkering te zijn.
2. **Dynamische registratie.** Zoek expliciet naar `importlib`, `__subclasses__`, entry-point-registries en Hydra `_target_`-strings in `conf/**/*.yaml` voordat je iets uit klasse E of D verwijdert. Een Hydra-target is een import die geen `import`-statement is.
3. **Aanroep vanaf de commandoregel.** `python -m tradebot.x.y` staat in geen enkele importgraaf. Grep `docs/`, `.github/`, `Makefile` en `dvc.yaml` op de modulenaam.

---

## FENCES — WAT DEZE FASE NIET AANRAAKT

Overtreding van een fence is geen slordigheid maar een vernietiging van bewijsmateriaal. Bij twijfel: niet aanraken, opschrijven, doorschuiven.

1. **`data/pit_store/**` blijft ongewijzigd in git.** Het lijkt een fout — 228 ruwe parquet/json-bestanden in versiebeheer naast een `.dvc`-pointer — maar het is een expliciet besluit (AD-12, toegelicht in `.gitignore` regel 53-58): onder DVC stond de store op een remote die niet bestond. Verwijderen of terug-DVC'en is een architectuurbesluit, geen opruiming.
2. **`artefacts/governance/**` is append-only.** De hypothesis-ledger (M = 2776), de preregistraties, `data_hashes.json`, `monitoring_config_hash.json`. Niets wordt hier verwijderd, samengevoegd of "opgeschoond". Ook niet de vier `preregistration_*.json` met onleesbare hex-namen.
3. **`reports/phase*_exit_report.md` en elk bewijsrapport blijven.** Ook de negen `portfolio_metrics_*.json`-varianten: dat zijn gemeten uitkomsten van verschillende configuraties, geen tijdelijke bestanden. Wie ze overbodig acht, moet eerst aantonen dat geen enkel rapport ernaar verwijst.
4. **De vier rode killgates blijven rood en blijven dezelfde vier.** `PROJECT_STATE.md` §7: *"Een suite met minder dan vier failures betekent dat een killgate is uitgeschakeld, niet dat er iets is opgelost."* `alpha/cm_carry.py` en `alpha/cm_tsmom.py` vallen in klasse C (research-only) — zij worden **niet** verplaatst, want zij dragen een pre-geregistreerde poort.
5. **De divergentie live/backtest (`PROJECT_STATE.md` §3.2) wordt in kaart gebracht, niet opgelost.** Twee featurestacks, twee impactmodellen (η = 0,1/0,142 live tegen 2,991922 gekalibreerd), twee barresoluties (5 s tegen dag), vijf namen tegen zes. Consolideren hoort bij openstaand besluit 0 en 3 van de eigenaar. Wie hier één van de twee stacks weggooit, neemt dat besluit stilzwijgend. **Verboden.**
6. **`docs/DEFERRED_ISSUES.md`-entries worden nooit verwijderd.** Sluiten mag, met bewijs. Een DI die verdwijnt zonder sluitingsregel is een verloren openstaande post.
7. **Geen gedragswijziging.** Geen enkele drempel, parameter, formule of signatuur verandert van waarde in deze fase. Hernoemen, verplaatsen en verwijderen mag; herrekenen niet.

---

## CONCRETE DELIVERABLES

1. **`scripts/reachability_map.py`** — AST-gebaseerde scanner (geen regex; regex ziet relatieve imports niet). Resolvet relatieve imports tegen het pakketpad, bouwt de importgraaf over `src/`, en berekent de transitieve afsluiting vanuit vier seedsets: DVC-stages, alle apps, alle scripts, alle tests. Output: JSON + markdown, één rij per module met klasse, LOC, en de seed die hem bereikt. `--strict` geeft exit 1 zodra klasse E niet-leeg is buiten de geregistreerde uitzonderingen.
2. **`tests/unit/test_reachability_map.py`** — bewijst de scanner op een fixture-boom: een relatieve import wordt gevolgd, een `__init__`-re-export wordt gevolgd, een Hydra-`_target_`-string wordt als bereik geteld, en een echt onbereikbare module wordt als E geclassificeerd. Zonder deze test is de scanner een mening.
3. **`reports/phase9_inventory.md`** — de volledige beslistabel ingevuld: 299 modules, 40 apps, 58 scripts. Per rij: pad, LOC, klasse, dekking, verdict, en bij D/E de gevonden of ontbroken afnemer. Dit is het langste document van de fase en het is bewijsmateriaal.
4. **`reports/phase9_coverage_baseline.md`** — dekking per module vóór en na, met het commando en de exacte toolversies waarmee gemeten is.
5. **`reports/phase9_removal_register.md`** — per verwijderd bestand: pad, LOC, klasse, reden, de commit die het verwijderde, en het `git show <sha>^:<pad>`-commando dat het terughaalt. Onherstelbaarheid is de enige echte fout in een opruimfase.
6. **`docs/CODE_REGISTER.md`** — de overlevende oppervlakte. Per pakket: waarvoor het bestaat, welk entrypoint het bereikt, welke dekking het draagt, en of het onder een ratchet staat. Dit is het document dat iemand leest die het project overneemt en wil weten wat hij mag aanraken.
7. **`research/`-track** — de wave-onderzoeksscripts uit `scripts/` verhuisd, met een `research/README.md` die de herkomst per bestand vermeldt, en de verwijzingen in `docs/WAVE_LOG.md` bijgewerkt. Sluit **DI-8**. `scripts/` houdt platformgereedschap — ten minste `audit_fallbacks.py`, `check_banned_methods.py`, `check_hardcoded_params.py`, `build_data_register.py` en de nieuwe `reachability_map.py`; per resterend script beslist stap 7 of het gereedschap of onderzoek is. Van de 58 scripts (8.688 LOC) zijn er 24 nergens genoemd — dat is het zwaartepunt van deze stap.
8. **Herstelde dev-omgeving** — `requirements-dev.lock` en de geïnstalleerde omgeving identiek; `pytest-cov` en `pytest-randomly` werkend; `reports/phase9_coverage_baseline.md` draagt het bewijs.
9. **Gecorrigeerde `Makefile`** — de niet-bestaande whitelist-entry `risk/portfolio.py` weg, de zes actuele overschrijdingen ofwel gesplitst ofwel met naam en reden gewhitelist. Een whitelist die een niet-bestaand bestand noemt is een poort die zijn eigen scope niet kent.
10. **`.github/workflows/inventory.yml`** — draait `reachability_map.py --strict`, `check_hardcoded_params.py --strict`, de LOC-poort en de dekkingsdrempel. Faalt, waarschuwt niet.
11. **`reports/phase9_exit_report.md`** — per exit-criterium het bewijs: commando, output, commit-sha.
12. **Bijgewerkte `docs/DEFERRED_ISSUES.md`** — DI-3, DI-4 en DI-8 gesloten of met een gecorrigeerde meting en een nieuwe voorwaarde herbevestigd. DI-3's cijfer is aantoonbaar fout; dat wordt hersteld, niet stilzwijgend overschreven.
13. **Bijgewerkte `docs/PROJECT_STATE.md`** — §4 en §6 aangevuld met de uitkomst van deze fase.

---

## STAPSGEWIJZE UITVOERING

Elke stap eindigt in een eigen commit. Elke stap die code toevoegt begint met de falende test.

**Stap 1 — Gedragsvingerafdruk vastleggen.**
Vóór alles: leg vast wat het systeem nú doet. Draai `python -m pytest -q` en bewaar de volledige uitvoer in `reports/phase9_fingerprint_before.txt`: aantallen passed/failed/skipped én de node-id's van elke failure. Draai `dvc status` en bewaar welke stages stale zijn. Draai `python apps/run_phase5_baseline.py` en bewaar de metrieken. Commit: `chore(phase9): record behavioural fingerprint before cleanup`.
Deze drie bestanden zijn je enige verdediging tegen de vraag *"heeft de opruiming iets kapotgemaakt?"*.

**Stap 2 — Meetinstrument herstellen.**
Installeer de omgeving opnieuw uit `requirements-dev.lock` in een schone virtualenv. Verifieer `pytest --version`, `mypy --version`, `python -c "import pytest_cov, pytest_randomly"`. Draai daarna dezelfde suite en vergelijk met de vingerafdruk uit stap 1 — een verschil hier is een bevinding over versiedrift, niet over jouw werk, en die schrijf je op vóór je verdergaat. Commit: `fix(env): reinstall dev environment from lock and restore coverage tooling`.

**Stap 3 — Dekkingsnulmeting.**
`python -m pytest -m "not slow and not regression" --cov=src/tradebot --cov-report=json --cov-report=term-missing`. Schrijf `reports/phase9_coverage_baseline.md` met de dekking per module en het totaal tegen de `fail_under = 70`-drempel. Als het totaal onder 70 ligt, is dat een gemeten feit dat in het rapport hoort, niet iets om de drempel voor te verlagen. Commit: `test(phase9): record per-module coverage baseline`.

**Stap 4 — Scanner: falende test eerst.**
Schrijf `tests/unit/test_reachability_map.py` tegen een fixture-boom in `tmp_path` met vier gevallen: relatieve import, `__init__`-re-export, Hydra `_target_`-string in YAML, en een echt onbereikbare module. Draai hem. Verwacht: `ModuleNotFoundError: scripts.reachability_map`. Commit: `test(inventory): add failing spec for reachability scanner`.

**Stap 5 — Scanner bouwen.**
Implementeer `scripts/reachability_map.py` tot de test groen is. Minimale implementatie: geen rapportage-opsmuk, geen kleuren, geen CLI-opties die de test niet vraagt. Commit: `feat(inventory): add AST-based reachability scanner`.

**Stap 6 — De kaart draaien en de false positives uitputten.**
Draai de scanner over de echte boom. Vergelijk met de nulmeting hierboven (123/81/8/48/39). Wijkt jouw uitkomst af, dan zoek je uit waarom vóór je verdergaat — een verschil betekent dat de boom is gewijzigd of dat jouw resolver anders werkt, en beide moeten in het rapport. Loop daarna de drie false-positive-klassen expliciet na: `grep -rn "importlib\|__subclasses__" src/`, `grep -rn "_target_" conf/`, en een grep op elke klasse-E-modulenaam in `docs/`, `.github/`, `Makefile`, `dvc.yaml`. Commit: `docs(phase9): publish measured reachability map with false-positive audit`.

**Stap 7 — De beslistabel invullen.**
Schrijf `reports/phase9_inventory.md`. Elke rij krijgt een verdict uit de tabel hierboven. Voor klasse D noteer je per module de gevonden afnemer (DI-nummer, preregistratie, AD-nummer) of de expliciete constatering dat er geen is. **Dit is een leesstap, geen schrijfstap in `src/`.** Je verwijdert in deze stap niets. Commit: `docs(phase9): complete disposition table for all 299 modules`.

**Stap 8 — Klasse E opruimen, één commit per pakket.**
Verwijder uitsluitend wat na stap 6 en 7 E blijft. Per pakket één atomaire commit. Na elke commit: `python -m pytest -q` en vergelijk met de vingerafdruk. Wijkt er iets af, dan is de verwijdering fout — revert die commit, herclassificeer de module, en noteer waarom de scanner hem miste. Vul `reports/phase9_removal_register.md` bij elke commit aan.
Bijzondere aandacht voor de twee grootste: `labeling/meta.py` (1.181 LOC) en `portfolio/legacy_sizing.py` (824 LOC). De laatste is expliciet ONGEWIJZIGD verhuisd zodat de Phase 3-baseline herrekenbaar blijft (DI-10). Verwijderen mag pas nadat je hebt aangetoond dat `reports/BASELINE_BENCHMARK.md` en de regressietests hem niet nodig hebben — draai `python -m pytest -m regression` en laat de uitkomst zien. Kun je dat niet aantonen: behouden, en de reden in het register.

**Stap 9 — Klasse C verhuizen naar `research/`.**
Verplaats met `git mv` zodat de historie meeverhuist. Werk de verwijzingen in `docs/WAVE_LOG.md` en `docs/AUDIT_WAVES20-25_2026-06-11.md` bij — dat was de voorwaarde waarop DI-8 is doorgeschoven. Laat `alpha/cm_carry.py` en `alpha/cm_tsmom.py` staan waar ze staan (fence 4). Commit: `refactor(research): move wave research scripts to research/ track (closes DI-8)`.

**Stap 10 — Klasse D beslissen.**
Per module: contract of ambitie. Contracten krijgen een regel in `docs/CODE_REGISTER.md` met het DI-nummer of besluit dat hen levend houdt. Ambities gaan naar `research/archive/` met een regel in het removal-register — niet naar `/dev/null`, want de volgende fase moet ze kunnen terugvinden. Eén commit per verdict-groep.

**Stap 11 — Klasse A en B onder de dekkingsdrempel.**
Sorteer op LOC × (1 − dekking) en werk van boven naar beneden. `train/` (4.849 LOC, 3 testbestanden), `reporting/` (2.259 LOC, 1), `portfolio/` (2.661 LOC, 4), `oms/` (1.273 LOC, 4), `selection/` (293 LOC, **0**) staan bovenaan. Per module: eerst de falende test die het feitelijke gedrag vastlegt, dan groen. **Schrijf gedragstests, geen vormtests.** `PROJECT_STATE.md` waarschuwt hier letterlijk voor: *"een groen criterium is pas bewijs als de bijbehorende test rood kan worden om de reden waarvoor het criterium bestaat. Een AST-test die de vorm van een aanroep controleert, is dat niet."*
Haal je de drempel voor een module niet, dan registreer je een ratchet in `docs/CODE_REGISTER.md` met een datum en een eigenaar. Doorschuiven zonder ratchet is de derde optie die dit project zichzelf niet toestaat.

**Stap 12 — Entrypoints saneren.**
De 36 ongenoemde entrypoints: per stuk ofwel een verwijzing in `dvc.yaml`/`Makefile`/`docs/runbook.md` toevoegen, ofwel naar `research/` verplaatsen, ofwel verwijderen met register-entry. Corrigeer bij deze stap de DI-4-meting naar de werkelijke 27 van 40 en herbevestig of sluit de post. Commit per groep.

**Stap 13 — Poorten repareren en dichttimmeren.**
Verwijder `risk/portfolio.py` uit de LOC-whitelist in de `Makefile`. Beslis per overschrijding: splitsen of gewhitelist met reden in de bestandsheader (`# LOC-EXCEPTION: <reden>`). Schrijf `.github/workflows/inventory.yml`. Bewijs dat de workflow blokkeert door in een wegwerpbranch een onbereikbare module toe te voegen en de run rood te zien worden. Zonder dat bewijs is de poort een decoratie. Commit: `ci(inventory): add blocking inventory gate with proof of failure`.

**Stap 14 — Werkboom-hygiëne.**
`.mypy_cache` (106 MB), `.hypothesis`, `.pytest_cache`, `.ruff_cache`, `catboost_info`, `logs/`, `outputs/` opruimen en verifiëren dat `.gitignore` ze allemaal dekt. Voeg een `clean`-doel toe dat ook in deze omgeving werkt — `make` bestaat hier niet, dus lever het als `scripts/clean.py` of documenteer het commando in `docs/runbook.md`. Raak `data/pit_store/` niet aan (fence 1). Commit: `chore(repo): purge tool caches and align gitignore`.

**Stap 15 — Divergentiekaart, zonder besluit.**
Werk `reports/phase7_divergence_map.md` bij met wat de opruiming zichtbaar heeft gemaakt: de overlap tussen de authoritative en de live-afsluiting is **34 modules / 11.255 LOC** van respectievelijk 123 en 65. Beschrijf de duplicatie feitelijk — twee featurestacks, twee impactmodellen, twee barresoluties — en verwijs naar openstaand besluit 0 en 3. **Los niets op.** Commit: `docs(divergence): quantify authoritative/live overlap after inventory`.

**Stap 16 — Vingerafdruk vergelijken en exit-rapport.**
Draai de volledige vingerafdruk opnieuw. Vergelijk regel voor regel met stap 1. Schrijf `reports/phase9_exit_report.md` met per exit-criterium het commando, de output en de commit-sha. Werk `docs/PROJECT_STATE.md` §4 en §6 bij. Commit: `docs(phase9): exit report — inventory closed, surface reduced, fingerprint unchanged`.

---

## CRITERIA & VALIDATIE (EXIT CRITERIA)

Deze fase is uitsluitend afgerond wanneer elk punt aantoonbaar waar is.

1. **Gedragsneutraliteit.** `python -m pytest -q` geeft na de fase **2770 passed · 4 failed · 23 skipped** — of, wanneer stap 11 dekking heeft bijgeschreven, méér passed en onveranderd 4 failed / 23 skipped — met **dezelfde verzameling failing node-id's** als in de nulmeting hierboven. Minder failures is een gefaalde fase, geen betere. Een skip die in een error verandert telt als regressie.
2. **Baseline-identiteit.** `python apps/run_phase5_baseline.py` levert bit-identieke metrieken aan de vingerafdruk uit stap 1. `python -m pytest -m regression` staat groen.
3. **Klasse E is leeg** — of elke resterende E-module staat met naam en reden in `docs/CODE_REGISTER.md`. `python scripts/reachability_map.py --strict` retourneert 0.
4. **Elke overlevende module heeft een verdict.** `reports/phase9_inventory.md` bevat 299 rijen plus de apps en scripts, elk met klasse, dekking en verdict. Nul rijen zonder verdict.
5. **Dekking is niet gedaald** en elke module in klasse A of B zit boven de drempel of staat onder een geregistreerde ratchet met datum en eigenaar. Nul modules in een derde categorie.
6. **De omgeving is de lock.** `pip freeze` en `requirements-dev.lock` verschillen nergens. `pytest-cov` en `pytest-randomly` draaien. De suite is groen onder willekeurige testvolgorde.
7. **De poorten staan en blokkeren.** `.github/workflows/inventory.yml` faalt aantoonbaar op een bewust toegevoegde onbereikbare module. De LOC-poort noemt geen niet-bestaande bestanden meer.
8. **Elke verwijdering is omkeerbaar.** Voor elk pad in `reports/phase9_removal_register.md` haalt het genoteerde `git show`-commando het bestand terug. Steekproef van vijf, met output in het exit-rapport.
9. **DI-3, DI-4 en DI-8 zijn gesloten of met een gecorrigeerde meting herbevestigd.** Geen enkele DI is verdwenen.
10. **De fences staan.** `git diff --stat <start-sha>..HEAD -- data/pit_store artefacts/governance reports/` toont nul verwijderde regels in `data/pit_store/` en `artefacts/governance/`.

> **Harde regel:** een groen criterium telt pas wanneer de bijbehorende controle rood kán worden om de reden waarvoor zij bestaat. Toon dat bij punt 3, 7 en 8 expliciet aan.

---

## REGELS & HANDELINGSINSTRUCTIES

* **Meten gaat vooraf aan verwijderen.** Geen enkel bestand verdwijnt op grond van "ziet er ongebruikt uit". De klasse komt uit de scanner, de dekking uit `coverage`, en beide staan in het inventarisrapport vóór de eerste `git rm`.
* **Fail-fast compliance blijft absoluut.** Nul `try/except ImportError` in `src/`. Een opruiming die een import in een `try` zet om een verwijdering te laten "werken", is precies de stille degradatie die Phase 0 heeft uitgeroeid. `scripts/audit_fallbacks.py --strict` blijft groen.
* **Optimaliseren betekent hier: dubbele waarheid verwijderen.** Niet: hot loops herschrijven, niet: numba erbij, niet: algoritmen vervangen. Elke prestatiewijziging is een gedragswijziging tot het tegendeel is gemeten, en dat meten hoort niet in deze fase.
* **Eén logische wijziging per commit.** `refactor(portfolio): remove unreachable legacy_sizing after regression proof`, `test(train): add behavioural coverage for ensemble fit path`, `chore(repo): purge tool caches`. Nooit meerdere pakketten in één commit; het removal-register moet per bestand naar één sha kunnen wijzen.
* **Verplaatsen met `git mv`.** Historie is bewijs. Een bestand dat als nieuw verschijnt heeft zijn `git log` verloren en daarmee zijn herkomst.
* **Geen hardcoded parameters, geen nieuwe.** `scripts/check_hardcoded_params.py --strict` staat groen op budget 310/317 en mag niet stijgen.
* **Documenteer het niet-oplosbare.** Kom je iets tegen dat buiten deze fase valt — en de divergentie uit fence 5 is daar het grootste voorbeeld van — dan gaat het naar `docs/DEFERRED_ISSUES.md` met fase-toewijzing. Niets verdwijnt stilzwijgend.
* **Bewijs boven bewering.** Elke claim in elk rapport is gekoppeld aan een reproduceerbaar commando en zijn output. "Opgeschoond" is geen bevinding; "39 modules / 4.521 LOC verwijderd, vingerafdruk identiek, register in `reports/phase9_removal_register.md`" wel.
* **Bij twijfel behouden.** Een module te veel bewaren kost LOC. Een module te weinig bewaren kost een contract, en dat merkt niemand tot het moment waarop het ertoe doet.

---

## STARTINSTRUCTIE

> **Begin nu met Stap 1.** Draai `python -m pytest -q` en bewaar de volledige uitvoer inclusief alle failing node-id's in `reports/phase9_fingerprint_before.txt`. Draai daarna `dvc status` en `python apps/run_phase5_baseline.py` en leg beide uitkomsten ernaast. Commit deze drie artefacten als `chore(phase9): record behavioural fingerprint before cleanup`.
>
> Ga daarna naar Stap 2 en herstel de dev-omgeving uit `requirements-dev.lock` — je meet de rest van deze fase met gereedschap dat op dit moment aantoonbaar ontbreekt. Verwijder pas een regel code nadat `reports/phase9_inventory.md` is gecommit.
