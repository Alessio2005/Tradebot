# Phase 9 — verwijderregister

> **Onherstelbaarheid is de enige echte fout in een opruimfase.** Per verwijderd
> bestand: wat het was, hoeveel LOC, waarom het weg mocht, de commit die het
> verwijderde, en het commando dat het terughaalt.
>
> Elk `git show`-commando hieronder is uitgevoerd en gaf het bestand terug met
> het aantal regels dat in de kolom LOC staat. Bewijs staat in
> `reports/phase9_exit_report.md`, exit-criterium 8.

**Startpunt van de fase:** `f7702dd`. **Klasse E vóór:** 11 modules / 1.505 LOC.
**Klasse E na:** 2 modules / 1.018 LOC — beide met een bestaansassertie, zie §3.

**Verwijderd: 9 modules / 487 LOC.** `src/` gaat van 299 modules / 71.487 LOC
naar 290 modules / 71.000 LOC.

---

## 1. Het register

| # | Pad | LOC | Klasse | Reden | Verwijderd door | Terughalen |
|---:|---|---:|:---:|---|---|---|
| 1 | `src/tradebot/alpha/decay_tracker.py` | 140 | E | Wave 18 IC-decay-tracker; nooit aangesloten, niet in `alpha/__init__.py`, geen DI/AD/preregistratie | `5257a52` | `git show 5257a52^:src/tradebot/alpha/decay_tracker.py` |
| 2 | `src/tradebot/features/macro.py` | 110 | E | transformeert macrodata die geen upstream heeft; het universum is zes perpetuals op dagbars | `e1fa983` | `git show e1fa983^:src/tradebot/features/macro.py` |
| 3 | `src/tradebot/data/sources/cboe.py` | 61 | E | de enige PIT-bron zonder ingest-app; mist zowel de `DATA_REGISTER.md`-registratie (G8) als de lookahead-test (G6) die het pakketcontract eist | `3d99d45` | `git show 3d99d45^:src/tradebot/data/sources/cboe.py` |
| 4 | `src/tradebot/logging_config.py` | 61 | E | docstring draagt op `setup_logging()` bij elke app-entry aan te roepen; **nul aanroepen** in de hele repository | `f9ecba3` | `git show f9ecba3^:src/tradebot/logging_config.py` |
| 5 | `src/tradebot/alpha/eq_strev_resid.py` | 55 | E | Wave 24; geboekt als ARCHIVED → F16. Het oordeel blijft, de implementatie gaat | `5257a52` | `git show 5257a52^:src/tradebot/alpha/eq_strev_resid.py` |
| 6 | `src/tradebot/types.py` | 25 | E | dubbele waarheid naast `schemas/` (1.657 LOC, 48 testbestanden); door niets geïmporteerd | `4f12430` | `git show 4f12430^:src/tradebot/types.py` |
| 7 | `src/tradebot/bars/tick.py` | 16 | E | `raise NotImplementedError`; niet in `bars/__init__.py` | `ca58de9` | `git show ca58de9^:src/tradebot/bars/tick.py` |
| 8 | `src/tradebot/bars/volume.py` | 16 | E | `raise NotImplementedError`; niet in `bars/__init__.py` | `ca58de9` | `git show ca58de9^:src/tradebot/bars/volume.py` |
| 9 | `src/tradebot/_version.py` | 3 | E | dubbele waarheid: claimt *single source of truth* voor de versie, terwijl `tradebot/__init__.py` die via `importlib.metadata` uit `pyproject.toml` leest | `4f12430` | `git show 4f12430^:src/tradebot/_version.py` |

Vijf commits, één per pakket:

```
5257a52  refactor(alpha):     remove two unreachable wave-era alpha modules
ca58de9  refactor(bars):      remove two unimplemented AFML bar-type placeholders
3d99d45  refactor(data):      remove the one PIT source that was never wired to an entrypoint
e1fa983  refactor(features):  remove the macro feature layer that has no upstream
4f12430  refactor(tradebot):  remove two modules that claim a truth held elsewhere
f9ecba3  refactor(tradebot):  remove the logging bootstrap that nothing bootstraps
```

## 2. Wat er vóór elke verwijdering is nagelopen

Geen enkel bestand is verdwenen op grond van "ziet er ongebruikt uit". Per
module zijn vijf controles gedraaid, alle vijf met nul treffers:

1. **Importbereikbaarheid** — klasse E in `scripts/reachability_map.py`, dat zijn
   resolver bewijst in `tests/unit/test_reachability_map.py` (17 passed).
2. **Bestaandsasserties** — volledige scan op `.exists()` en `.is_file()` over
   een `src/`-pad in `tests/`. Deze controle redde twee andere modules; zie §3.
3. **Documentatiepoort** — nul backtick-paden in `docs/*.md`, zodat
   `tests/unit/test_docs_claim_only_what_exists.py` groen blijft.
4. **Smoke-test** — geen van de negen staat in `tests/test_imports.py`.
5. **Afnemer** — nul treffers in `artefacts/governance/`,
   `docs/ARCHITECTURAL_DECISIONS.md`, `docs/DEFERRED_ISSUES.md` en
   `docs/PROJECT_STATE.md`.

Na elke commit draaiden `tests/test_imports.py` en de documentatiepoort;
na de laatste de volledige suite tegen de vingerafdruk uit stap 1.

## 3. Wat NIET is verwijderd, en waarom

Twee modules die de scanner als klasse E aanmerkt, blijven staan. Zij dragen een
contract dat geen importgraaf kan zien:

| Pad | LOC | Waarom behouden |
|---|---:|---|
| `src/tradebot/portfolio/legacy_sizing.py` | 824 | `tests/unit/test_risk_alpha_decoupling.py::test_the_entangled_portfolio_module_no_longer_lives_in_risk` asserteert dat dit bestand BESTAAT — het Phase 4 exit-criterium 3-bewijs. Daarnaast DI-10 (bewust ONGEWIJZIGD zodat de Phase 3-baseline herrekenbaar blijft) en een ratchetpost in `scripts/check_hardcoded_params.py` |
| `src/tradebot/portfolio/covariance.py` | 194 | dezelfde bestaansassertie; DI-10 noemt `_safe_corr` expliciet |

De opdracht schrijft voor `legacy_sizing.py` een regressiebewijs voor als
voorwaarde voor verwijdering. Dat bewijs is niet geleverd en hoefde niet
geleverd te worden: de bestaansassertie is een hardere blokkade dan de
regressietest. Een `git rm` zou een groene test rood maken en daarmee
exit-criterium 1 breken, ongeacht wat de regressietests zeggen.

> Dit is het geval waarvoor de fase is geschreven. Het meetinstrument gaf beide
> modules vrij; de handmatige false-positive-audit hield 1.018 LOC tegen. De
> volgorde *meten, dan naloopen, dan pas verwijderen* is precies wat het
> verschil maakte.

## 4. Wat de opdracht verwachtte en wat er gemeten is

De fase-opdracht raamt de verwijdering op **39 modules / 4.521 LOC** en wijst in
stap 8 twee modules aan als "de twee grootste":

| Verwacht | Gemeten | |
|---|---|---|
| `labeling/meta.py` — 1.181 LOC, klasse E | **klasse A** | bereikt vanuit `apps/build_features.py`, seed van een DVC-stage. Verwijderen had de authoritative DAG gebroken |
| `portfolio/legacy_sizing.py` — 824 LOC, klasse E | **klasse E, maar behouden** | bestaansassertie, zie §3 |

Van de 39 verwachte modules bleken er 28 bereikbaar en 2 contractdragend. De
opgeruimde oppervlakte is daarmee **487 LOC in plaats van 4.521** — geen
tegenvaller maar de uitkomst van een meting die de nulmeting corrigeert. Zie
`reports/phase9_reachability_audit.md` §1 voor de twee importedges die het
verschil verklaren.
