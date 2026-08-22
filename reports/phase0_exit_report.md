# PHASE 0 — EXIT REPORT

> **Fase:** 0 van 7 — Audit & Repository Hygiene
> **Bindend brondocument:** `docs/ARCHITECTUUR_AUDIT_2026-08-22.md` — secties 5, 18, 20, 23, 26
> **Baseline-commit:** `73a01a4` (ongewijzigde nulmeting)
> **Exit-commit:** `417acce`
> **Datum:** 2026-08-22

---

## VERDICT

**7 van de 8 exit criteria zijn behaald. Criterium 6 is NIET behaald in de
letterlijke zin (nul literals), maar is wel AFDWINGBAAR gemaakt.**

### Besluit van de opdrachtgever — 2026-08-22

Voorgelegd met drie opties (zie §Openstaand besluit). **Gekozen: optie 1 —
ratchet accepteren.**

Daarmee geldt criterium 6 als *afgedwongen in plaats van geëlimineerd*:

- het budget per bestand is gepind op de gemeten nulstand (336 literals,
  107 bestanden);
- `scripts/check_hardcoded_params.py --strict` faalt bij elke nieuwe literal en
  draait blokkerend in CI;
- budgetten mogen uitsluitend omlaag;
- de opruiming loopt per fase via **DI-9 t/m DI-14** in `docs/DEFERRED_ISSUES.md`;
- modules die Phase 3 nieuw aanmaakt krijgen budget 0 zonder uitzondering.

**Phase 0 is hiermee afgerond en Phase 1 is ontgrendeld.**

Deze afwijking van de letterlijke tekst is bewust, expliciet vastgelegd en
draagt een handhavingsmechanisme. Hij is niet stilzwijgend afgevinkt.

---

## 1. Nul stille fallbacks — ✅ BEHAALD

```bash
python scripts/audit_fallbacks.py --strict
```

```
TOTAAL 35 bevinding(en): SWALLOWED_EXCEPT=35
  blokkerend: 0   advies: 35
exit code 0
```

| Categorie | Nulmeting | Nu |
|---|---:|---:|
| `try/except ImportError` | 34 | **0** |
| bare `except:` | 0 | **0** |
| `except Exception` zonder re-raise | 130 | **0** |
| warn-then-degrade | 2 | **0** |
| **Blokkerend totaal** | **166** | **0** |
| Advies (nauwe, niet-degraderende handlers) | 42 | 35 |

**Belangrijke correctie op het auditdocument.** Sectie 5.2 noemt *"12 locaties"*.
De uitputtende AST-scan vond er **208**. De audittelling was een steekproef.

**De 42 advies-bevindingen zijn stuk voor stuk beoordeeld.** Twaalf bleken geen
numerieke hygiëne maar een echte **modelwissel**, verstopt achter een nauw
exceptietype — en zijn alsnog gerepareerd:

| Module | Wat er stilzwijgend gebeurde |
|---|---|
| `portfolio/black_litterman.py` | `LinAlgError` → **marktprior**: de views (P, Q) verdwenen uit de schatting terwijl het resultaat als Black-Litterman werd gerapporteerd. Tweede site → 1/N. |
| `volatility/har_rv.py` | `LinAlgError` → `beta = [mean(y), 0, 0, 0]`: HAR-RV (**Level 2**) degradeerde naar een **constante voorspelling (Level 0)**. Elke QLIKE-competitie tegen EWMA zou feitelijk EWMA-vs-constante zijn geweest. |
| `train/thompson.py` (2×) | "fallback θ̂" maakt de Thompson-sampler **greedy** — exploratie valt weg terwijl de bandit zichzelf nog als Thompson-sampler rapporteert. De code noemde dit zelf al *"exploration-death fallback"*. |
| `alpha/adaptive_wf.py` (2×) | Bij een gefaalde neutralisatie-regressie werd de **rauwe** y teruggeschreven. Het resultaat heette daarna nog "geneutraliseerd" terwijl de factor-exposure er volledig in zat — een markt-neutraal boek dat op die dagen gewoon directioneel was. |
| `risk/beta_hedge.py` | Huber-IRLS → gewone **OLS**-beta. Juist bij de outliers waarvoor Huber is gekozen, viel de schatter terug op de gevoelige variant — onder de naam "Huber". |
| `backtest/evaluation.py` | `count_git_commits` gaf een **verzonnen** fallback-telling die als trial-count `M` de Deflated Sharpe Ratio in ging. Een te lage `M` maakt DSR structureel te optimistisch. |
| `registry/lineage.py` | `get_file_hash` gaf `""` bij een ontbrekend artefact, en `verify_lineage` slaat een lege hash over. Een **verdwenen modelbestand passeerde de lineage-verificatie**. |
| `features/cfi.py` | `log_loss` → `accuracy`: de scoringsmetriek wisselde **permanent** midden in de feature-importance-berekening. |
| `train/catboost.py` | `return raw`: onuitgelijnde, ongeschaalde features naar een model dat op een andere kolomvolgorde is getraind. |
| `selection/mda.py` | Embargo-blok op `0` i.p.v. `NaN`. Een nul is een *geldige observatie* ("geen beweging"), geen ontbrekende — dat verschuift de MDA-nulverdeling. |

De 35 resterende advies-bevindingen zijn Cholesky-jitter, eigenvalue-clipping en
het overslaan van een corrupte JSONL-regel. Volledig geregistreerd in
`reports/phase0_fallback_register.md`.

**Waarom `SWALLOWED_EXCEPT` niet blokkeert.** Exit criterium 1 noemt exact drie
condities (`ImportError`-fallbacks, bare excepts, `except Exception` zonder
re-raise); deliverable 8 voegt warn-then-degrade toe. Een nauwe
`except LinAlgError` die een bijna-singuliere covariantiematrix regulariseert is
numerieke hygiëne, geen degradatie van een model naar een naïevere benadering.
Die scheiding is expliciet gedocumenteerd in de scanner.

---

## 2. 100% reproduceerbare build — ✅ BEHAALD

```bash
python -m venv C:/tbvenv
C:/tbvenv/Scripts/python -m pip install -r requirements.lock
C:/tbvenv/Scripts/python -m pip install -e . --no-deps
C:/tbvenv/Scripts/python -m pytest tests/unit -q
```

```
429 passed, 1 skipped
```

Nul handmatige installaties. `requirements.lock` bevat de volledige transitieve
closure (135 pakketten, gepind), gegenereerd met `pip-compile --strip-extras`.

**De schone-venv-verificatie vond drie gaten die op de ontwikkelmachine
onzichtbaar waren**, omdat de pakketten daar toevallig al stonden:

| Dependency | Gebruikt in | Symptoom op een schone build |
|---|---|---|
| `tenacity` | `data/crypto.py` (retry-decorators) | `ModuleNotFoundError` bij collectie van 2 testbestanden |
| `websockets`, `prometheus-client` | `live/feed.py`, `monitoring/metrics.py` | lockfile dateerde van vóór hun declaratie |
| `yfinance` | `data/tradfi_macro.py` (module-level) | `ModuleNotFoundError` bij import |

Een AST-scan van **alle 24** third-party top-level imports in `src/` tegen de
lockfile bevestigt dat er geen ongedeclareerde harde dependency meer is. De drie
resterende (`torch`, `cupy`, `mlflow`) zijn bewuste capability-probes via
`has_module`, elk gemotiveerd in `docs/DEPENDENCY_CONTRACT.md`.

---

## 3. Valide `git_sha` — ✅ BEHAALD (D-9 gesloten)

```bash
python -c "from tradebot.registry.lineage import get_git_sha; print(get_git_sha())"
# 417acce
git cat-file -e 417acce^{commit}    # exit 0
```

Bovendien: `get_file_hash` gaf voorheen `""` terug voor een ontbrekend artefact,
en `verify_lineage` sláát een lege hash over. Een verdwenen modelbestand
passeerde daardoor de lineage-verificatie zonder melding. Die fallback is
verwijderd — een ontbrekend artefact crasht nu.

---

## 4. `quant_architect` bestaat nergens meer — ✅ BEHAALD

```bash
grep -rnE '^[[:space:]]*(from|import)[[:space:]]+quant_architect' src/
# geen treffers
```

**Correctie op het auditdocument.** Sectie 5.2 noemt `tune/objective.py` als
vijfde importeur. Dat is feitelijk onjuist; de werkelijke vijfde was
`features/scaling.py`. Conform sectie 2.2 prevaleert de meting.

De import was bovendien niet *dood* maar **verkeerd geadresseerd**: alle negen
symbolen bestonden al in `tradebot.train.{reward,schema_guard,stack,thompson,quant_arch}`.
De ernstigste was `train/ensemble.py`, waar `EntropyGate`, `LedoitWolfThompsonSampler`,
`NetAlphaReward` en `FeatureSchemaGuard` op `typing.Any` werden gezet — waardoor
de entropy-gate nooit een signaal blokkeerde, de bandit terugviel op ruwe
MVN-sampling, de net-alpha reward nooit werd berekend en schema-drift nooit werd
gedetecteerd. Permanent, en zonder melding.

Vijf modules, vijf atomaire commits: `e0b5481`, `cc2c5fa`, `dd5feef`, `8b26883`, `f6a04ae`.

---

## 5. Config-integriteit — ✅ BEHAALD

`src/tradebot/schemas/config.py` levert zeven Pydantic v2-domeinmodellen, alle
met `ConfigDict(extra="forbid", frozen=True)`. `conf/` is geherstructureerd naar
sectie 20 (`conf/data/`, `conf/model/`, `conf/execution/`, `conf/validation/`,
`conf/backtest/`).

`tests/unit/test_config_contracts.py` — 45 tests, alle groen — bewijst:

- elke YAML laadt onder zijn schema;
- een **geïnjecteerde onbekende sleutel** triggert `ConfigContractError` (per bestand geparametriseerd);
- config is onveranderlijk na laden;
- `pydantic.ValidationError` lekt niet naar de aanroeper;
- de bindende auditconstanten staan aantoonbaar in `conf/` en niet in `src/`:
  λ = 0.94, purged walk-forward als enige CV, `embargo ≥ labelhorizon`
  (afgedwongen door een `field_validator`), CPCV/PBO gepind op géén
  gate-bevoegdheid, Bybit funding-interval 8 h, kostenaanname gemarkeerd als
  **voorlopig** tot de η-kalibratie in Phase 5.

---

## 6. Nul hardcoded parameters — ❌ NIET BEHAALD

```bash
python scripts/check_hardcoded_params.py
# 336 numerieke literal(s) in 107 bestand(en).
```

**Gemeten stand: 336 numerieke literals in 107 van de 232 modules onder `src/`.**
Het criterium eist nul. De audit suggereert een handvol; de werkelijkheid is twee
ordes groter.

### Waarom dit binnen Phase 0 niet oplosbaar is

Er zit een **interne tegenstrijdigheid in de fase-specificatie**. Criterium 6
eist dat elke drempel, lookback-window, λ-parameter en limiet uit `src/`
verdwijnt. Diezelfde fase legt onder *Regels & Handelingsinstructies* op:

> *"Geen scope creep. Deze fase herstelt de fundering. Je implementeert geen
> nieuwe modellen, geen nieuwe alpha, geen HMM-herontwerp."*

en

> *"100% PIT rigor. Ook in een hygiëne-fase: raak geen enkele datapipeline aan op
> een manier die causaliteit kan schenden. Bij twijfel: niet aanraken."*

De 336 literals zitten in de signatures van `risk/`, `execution/`, `train/`,
`tune/`, `labeling/` en `backtest/` — precies de modules die de audit in
sectie 23/24 aanwijst voor **REDESIGN in Phase 4, 5 en 6**. Ze nu
parametriseren betekent: elke signature wijzigen, elke call-site meeveranderen,
en dat in Phase 4–6 nogmaals doen. Dat is niet alleen dubbel werk maar ook
risicovol, want een deel van die code heeft geen testdekking.

### Wat er in plaats daarvan is gebouwd

Het criterium is **niet stil afgevinkt**. Er is een **ratchet** gebouwd die het
afdwingbaar maakt:

`scripts/check_hardcoded_params.py`

- legt het budget per bestand vast op de **gemeten nulstand** (336 over 107 bestanden);
- wijst elk bestand toe aan de fase die het opruimt — **DI-9 t/m DI-14** in `docs/DEFERRED_ISSUES.md`;
- **faalt** zodra een bestand méér literals krijgt, of zodra een nieuw bestand met literals verschijnt (dat staat niet in de tabel en krijgt dus budget 0);
- geeft de modules die Phase 3 nieuw aanmaakt (`features/transforms.py`, `alpha/base.py`, `portfolio/risk_parity.py`, `validation/`) **budget 0 zonder uitzondering**;
- draait blokkerend in `.github/workflows/hygiene.yml`.

Budgetten mogen uitsluitend omlaag.

**Aangetoond dat de ratchet werkt:**

```
# sneaky_threshold: float = 0.037 toegevoegd aan volatility/ewma.py
volatility/ewma.py: 3 literal(s) > budget 2
exit code 1
# na terugdraaien
exit code 0
```

### Eén bevinding hieruit verdient aparte aandacht

`volatility/ewma.py` — de estimator die Phase 3 als **baseline** gebruikt —
heeft vandaag:

- **géén λ-parameter** maar een `halflife=20`, terwijl de audit λ = 0.94 bindend voorschrijft;
- `return pd.Series(0.0, ...)` op een lege DataFrame;
- `.ffill().fillna(0.0)` over de burn-in — een **impliciete constante volatiliteit van nul**, wat in Naive Risk Parity (gewicht ∝ 1/σ) een **oneindig gewicht** oplevert.

Dat is precies wat Phase 3 stap 3 verbiedt (*"crashen of NaN propageren — nooit
een impliciete constante volatiliteit"*). Het harden ervan is expliciet Phase 3
stap 3 en is daarom **niet** in Phase 0 uitgevoerd; het staat als **DI-12**
geregistreerd met budget 2 → 0.

---

## 7. CI blokkeert — ✅ BEHAALD

Negatieve controle op branch `test/negative-control-fallback`: een bewust
geïntroduceerde stille fallback in `volatility/ewma.py`.

```
$ python scripts/audit_fallbacks.py --strict
STRICT: build gebroken door 1 blokkerende stille fallback(s).
src/tradebot/volatility/ewma.py:33: IMPORT_FALLBACK [ImportError] in
  _deliberate_silent_fallback -> import numpy as np; return np.full(...)
exit code = 1
```

Drie pytest-gates faalden mee: `test_strict_scan_exits_zero`,
`test_zero_import_fallbacks`, `test_no_blocking_kind_at_all`. Branch verwijderd
na het bewijs; `main` staat weer op exit 0.

`.github/workflows/hygiene.yml` faalt — waarschuwt niet — op: een ontbrekende
harde dependency na lockfile-install, een niet-resolvable `git_sha`, een
blokkerende fallback, een `quant_architect`-import, een overschreden
parameter-budget, een ontbrekende DAG-map, een lint-fout, een
`mypy --strict`-fout of een falende unit test. `continue-on-error` komt in het
bestand niet voor.

---

## 8. DAG-mappen bestaan — ✅ BEHAALD (D-8 gesloten)

`artefacts/features/`, `artefacts/models/` en `artefacts/tracks/` bestaan, elk
met een `README.md` die het artefact-contract vastlegt: formaat, tijdstandaard
(UTC ns), provenance (`data_hash`, `git_sha`, `config_hash`), causaliteit,
onveranderlijkheid, en wie schrijft respectievelijk leest.

---

## Regressiebewijs

| Meting | Baseline `73a01a4` | Exit `417acce` |
|---|---:|---:|
| Tests totaal | 567 | 633 |
| Passed | 538 | 604 |
| Failed | 8 | 6 |
| Skipped | 21 | 21 |

De 8 failures op de baseline zijn geverifieerd in een **aparte git-worktree** op
commit `73a01a4`: exact dezelfde set. **Nul regressies** over de hele fase. Twee
ervan (`test_wave13_docs_exist[CHANGELOG.md]`, `test_wave13_changelog_mentions_audit_fixes`)
zijn opgelost door `CHANGELOG.md` terug te zetten naar de projectroot. De
resterende 6 (4 kill-gate-artefacten, 2 property-tests) zijn pre-existent en
raken geen Phase 0-deliverable.

Alle **232 modules** importeren. `ruff check` op `src/` en de Phase 0-tooling:
schoon (691 → 0). `mypy --strict` op `schemas/config.py` en `utils/failfast.py`:
schoon.

### Twee echte bugs gevonden tijdens de lint-opruiming

| Bestand | Bug |
|---|---|
| `schemas/tracks.py` | `pd` werd gebruikt maar **nooit geïmporteerd** — de UTC-index-check van `TrackSchema` heeft dus nooit gewerkt. |
| `risk/__init__.py` | `Regime` bestond **twee keer** met een volledig andere betekenis: een propfirm-accountregime (`CHALLENGE`/`FUNDED`) en een marktregime (`BEAR`/`FLAT`/`BULL`). De laatste import won, dus `tradebot.risk.Regime` was stilzwijgend het marktregime en de propfirm-variant onbereikbaar. Nu expliciet `AccountRegime`. |

---

## Deliverables

| # | Deliverable | Status |
|---|---|---|
| 1 | `.git` herinitialisatie + `.gitignore` | ✅ `73a01a4` |
| 2 | `utils/failfast.py` | ✅ 32 tests |
| 3 | `pyproject.toml` + lockfile | ✅ 135 pakketten gepind |
| 4 | `schemas/config.py` (Pydantic v2) | ✅ 7 domeinmodellen |
| 5 | `conf/` composition root | ✅ sectie 20-indeling |
| 6 | `quant_architect` verwijderd | ✅ 5 modules, 5 commits |
| 7 | `hmm_regime.py` gesaneerd | ✅ 5 degradatiepaden weg |
| 8 | `scripts/audit_fallbacks.py` | ✅ met negatieve controles |
| 9 | `tests/unit/test_no_silent_fallbacks.py` | ✅ 21 tests |
| 10 | `tests/unit/test_config_contracts.py` | ✅ 45 tests |
| 11 | DAG-mappen + artefact-contracts | ✅ D-8 |
| 12 | `docs/DEPENDENCY_CONTRACT.md` | ✅ |
| 13 | `docs/architecture.md` gecorrigeerd | ✅ D-5 |
| 14 | `.github/workflows/hygiene.yml` | ✅ 10 blokkerende stappen |
| — | `docs/DEFERRED_ISSUES.md` | ✅ DI-1 t/m DI-14 |
| — | `scripts/check_hardcoded_params.py` | ✅ ratchet |

---

## Besluit (afgehandeld)

Criterium 6 is voorgelegd aan de opdrachtgever met de volgende drie opties.
**Optie 1 is gekozen op 2026-08-22.**

1. **Ratchet accepteren** — criterium 6 geldt als afgedwongen (niet als
   geëlimineerd), de 336 literals worden per fase opgeruimd volgens DI-9…DI-14,
   en Phase 1 start. ✅ **GEKOZEN.**
2. **Volledig elimineren vóór Phase 1** — 336 literals in 107 bestanden
   parametriseren, inclusief modules die Phase 4–6 daarna opnieuw herschrijven.
3. **Scope beperken tot `risk/`, `volatility/`, `execution/`** — de drie modules
   die stap 8 expliciet noemt (88 literals), en de rest via de ratchet.

Daarnaast is besloten over de reikwijdte van de Phase 1-ingestie: **alleen daily
OHLCV nu, intraday (5m) later**. Consequentie: Open Question 1 uit sectie 27 —
*"beschikken we over voldoende orderboek-depth om HAR-RV en het TCA-impactmodel
te kalibreren?"* — wordt in het Phase 1 exit-rapport beantwoord met de gemeten
dekking van wat wél is ingested, en met een expliciet *"intraday nog niet
gemeten"* voor de rest.
