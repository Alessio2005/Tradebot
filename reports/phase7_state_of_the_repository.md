# PHASE 7/8 — DE GEMETEN STAAT VAN DE REPOSITORY

**Fase:** 7/8 van 8 — Consolidatie, Production Readiness & Oplevering · **Prioriteit:** P0
**Bindend brondocument:** `Prompts-fases/fase_7_8_consolidatie_productie.md` §0, §12
**Startpunt:** `42555d2` · branch `main` · werkkopie `D:\Tradebot`
**Gemeten:** 2026-08-27 · **Meter:** Head of Trading Systems & Production Operations
**Status:** meting vóór enige wijziging — er is op dit moment nog niets aangeraakt

---

## 0. Waarom dit rapport bestaat

De masterprompt is gedateerd 2026-08-27 en stelt in §10: *"Meet voordat je iets
beweert. Herhaal elke meting bij aanvang en noteer elke afwijking. Waar meting en
rapport verschillen, prevaleert de meting."*

Dit rapport is die herhaling. Per meting: het commando, de uitvoer, en of hij
overeenkomt met §0 van de prompt. **Vier afwijkingen** zijn gevonden; zij staan in
§8 en zijn geen van alle blokkerend, maar drie ervan corrigeren een cijfer dat de
prompt als feit presenteert.

**Samenvattend oordeel: §0 van de masterprompt is materieel correct.** Elke
structurele bevinding — de ontbrekende governancelaag, de kapotte editable install,
de 351 liegende caches, de rode ratchet, de niet-aangesloten live-laag — is
onafhankelijk gereproduceerd.

---

## 1. Git — de historie bestaat, de kopie niet

```bash
git rev-list --count HEAD
git log --oneline -1
git log --oneline --reverse | head -1
git branch
git remote -v
git status --short
du -sh .git
```

| Meting | Uitvoer | §0.1 | Komt overeen |
|---|---|---|---|
| Commit-telling | `77` | 77 | **ja** |
| HEAD | `42555d2 feat(regime): build the M2 filtered-only HMM and wall off the smoothed route` | idem | **ja** |
| Eerste commit | `73a01a4 chore(repo): re-initialise version control with baseline snapshot` | idem | **ja** |
| Branch | `* main` | main | **ja** |
| Remote | *(leeg)* | leeg | **ja** |
| `.git`-omvang | `3.2M` | 3.2M | **ja** |

**D-9 is en blijft gesloten.** De volledige historie van Phase 0 tot en met de
lopende Phase 6 staat er.

**G-1 bevestigd — er is geen remote. Nul off-machine kopie.** De 77 commits, het
falsificatieregister, de hypothese-ledger (`M = 2776`) en de bevroren
pre-registraties staan op één fysieke schijf. Dit is een single point of failure
onder de volledige governance-laag.

### 1.1 Werkkopie-status

```
 M Prompts-fases/fase_6_advanced_research.md
 M x.jsonl
?? Prompts-fases/fase_7_8_consolidatie_productie.md
```

Drie regels waar §0.1 er twee noemt. De derde is de masterprompt van deze fase
zelf, die na de meting van 2026-08-27 is toegevoegd. **Geen afwijking** — verwacht.

---

## 2. G-2 — De tweede repository op de C-schijf

```bash
git --git-dir=C:/Users/algul/Documents/Tradebot/.git rev-list --count HEAD
git --git-dir=C:/Users/algul/Documents/Tradebot/.git log --oneline -1
git --git-dir=C:/Users/algul/Documents/Tradebot/.git \
    --work-tree=C:/Users/algul/Documents/Tradebot status --short | wc -l
ls -la C:/Users/algul/Documents/Tradebot/
```

| Meting | Uitvoer | §0.1 | Komt overeen |
|---|---|---|---|
| Commit-telling | `77` | 77 | **ja** |
| HEAD | `42555d2` (identiek aan D:) | identiek | **ja** |
| Ontbrekende paden in werkkopie | `637` | 637 | **ja** |
| Inhoud werkkopie | alleen `.git/` en `src/` | idem | **ja** |

**De landmijn bestaat.** Een `git add -A && git commit` in die map legt de
verwijdering van 637 paden vast in een historie die van buiten niet van de echte te
onderscheiden is. Er is op dit moment geen bewijs geleverd dat deze repo nul unieke
commits bevat; dat bewijs is Stage A-1 stap 3 en is een **voorwaarde** voor
verwijdering (no-go 3).

---

## 3. G-3 — `x.jsonl`, het uitgelekte testartefact

```bash
wc -l x.jsonl
head -2 x.jsonl
git log --oneline --all -- x.jsonl
```

```
38 x.jsonl
{"asset_class": "crypto", "gap_end_iso": "1970-01-07 00:00:00+00:00", ...,
 "granularity": "1d", "n_missing": 1, "symbol": "B"}
```

Herkomst: `d79bf25 feat(data): add L0 validation layer with gap and adjustment
ledgers`, laatst gewijzigd in `24999fe docs(reports): publish phase 1 exit report`.

**Bevestigd:** git-tracked, in de projectroot, timestamps in 1970, symbool `"B"` —
synthetische testdata. Exact de defectklasse die `phase5_exit_report.md` §9.1
benoemt.

> **Afwijking 1: 38 regels, niet 35.** Zie §8.

---

## 4. E-1 — De editable install wijst naar de C-schijf

```bash
python --version                                    # Python 3.13.0
python -c "import site;print(site.getsitepackages()[-1])"
cat ".../__editable__.tradebot-0.4.0.pth"
python -c "import tradebot; print(tradebot.__path__)"
python -c "from tradebot.registry.lineage import get_git_sha"
```

| Meting | Uitvoer |
|---|---|
| Interpreter | `C:\Users\algul\AppData\Local\Programs\Python\Python313\python` — 3.13.0 |
| `.pth`-inhoud | `C:\Users\algul\Documents\Tradebot\src` |
| `tradebot.__path__` | `_NamespacePath(['C:\\Users\\algul\\Documents\\Tradebot\\src\\tradebot'])` |
| `import tradebot.registry` | `ModuleNotFoundError: No module named 'tradebot.registry'` |

**Volledig bevestigd.** `import tradebot` slaagt als lege namespace-package;
`import tradebot.<wat dan ook>` faalt. De geïnstalleerde package wijst naar de
pre-verhuizingslocatie, die bestaat maar leeg is.

**Phase 0 exit-criterium 2 (100 % reproduceerbare build) is op deze interpreter
aantoonbaar niet meer waar.**

---

## 5. E-2 — Alle bytecode-caches liegen over hun herkomst

```bash
python - <<'PY'
import pathlib, marshal
h=t=0
for p in pathlib.Path('.').rglob('*.pyc'):
    if '.git' in p.parts: continue
    t+=1
    c=marshal.loads(p.read_bytes()[16:])
    if 'Documents' in c.co_filename: h+=1
print(f'{h}/{t} .pyc met oud C-pad')
PY
find . -name "__pycache__" -type d -not -path "./.git/*" | wc -l
```

```
351/351 .pyc met oud C-pad
39
```

Voorbeelden:

| `.pyc` | `co_filename` |
|---|---|
| `tests\__pycache__\conftest.cpython-313-pytest-9.0.2.pyc` | `C:\Users\algul\Documents\Tradebot\tests\conftest.py` |
| `tests\__pycache__\test_adaptive_wf...pyc` | `C:\Users\algul\Documents\Tradebot\tests\test_adaptive_wf.py` |
| `tests\__pycache__\test_cpcv...pyc` | `C:\Users\algul\Documents\Tradebot\tests\test_cpcv.py` |

**Bevestigd: 351/351, 39 `__pycache__`-mappen.** Zichtbaar in de suite-run van §7:
elke SKIP-melding citeert `C:\Users\algul\Documents\Tradebot\...`, terwijl de
RuntimeWarning uit `var.py` `D:\Tradebot\...` citeert. Twee schijven in één rapport.

---

## 6. De drie ratchets — één staat ROOD

```bash
python scripts/check_hardcoded_params.py --strict ; echo $?
python scripts/audit_fallbacks.py --strict ; echo $?
```

### 6.1 `check_hardcoded_params.py --strict` → **exit 1** (ROOD)

```
OVERSCHRIJDINGEN:
  validation/data_adequacy.py:      1 literal(s) in een door Phase 0-3 bestuurde module (budget 0)
  validation/diagnostics_report.py: 2 literal(s) in een door Phase 0-3 bestuurde module (budget 0)
  validation/econometrics.py:       8 literal(s) in een door Phase 0-3 bestuurde module (budget 0)
  validation/vol_metrics.py:        2 literal(s) in een door Phase 0-3 bestuurde module (budget 0)
  volatility/realized.py:           2 literal(s) > budget 0 (geen budget)

327 numerieke literal(s) in 107 bestand(en).
Toegestaan door de ratchet: 317. Bestuurd door Phase 0-3: budget 0.
```

**Exact overeenkomstig §0.3.** 15 literals in vijf modules, 327 totaal, ratchet 317.
`.github/workflows/hygiene.yml` faalt vandaag. Zolang dit zo is, is elke uitspraak
*"de gates zijn groen"* onjuist.

### 6.2 `audit_fallbacks.py --strict` → **exit 0** (GROEN)

```
TOTAAL 37 bevinding(en): SWALLOWED_EXCEPT=37
  blokkerend: 0   advies: 37
```

**Exact overeenkomstig §0.3:** 37 bevindingen, 0 blokkerend.

### 6.3 Suite → **4 failures** (de faalbasislijn)

```bash
python -m pytest -q -p no:randomly --collect-only   # 2011 tests collected
python -m pytest -q -p no:randomly
```

**2.011 tests verzameld** — exact overeenkomstig §0.8.

```
FAILED tests/killgates/test_expansion_killgates.py::test_futures_unit_against_gate[KG-B1 in-sample-cm_carry]
FAILED tests/killgates/test_expansion_killgates.py::test_futures_unit_against_gate[KG-B2 residual alpha-cm_tsmom]
FAILED tests/killgates/test_expansion_killgates.py::test_futures_unit_against_gate[KG-B2 residual alpha-cm_carry]
FAILED tests/killgates/test_expansion_killgates.py::test_futures_unit_against_gate[KG-B3 out-of-sample-cm_carry]
```

**Faalbasislijn = 4, en het zijn exact de vier pre-geregistreerde killgates op
`cm_carry` en `cm_tsmom`.** Zij horen rood te staan. Phase 6 exit-criteria 15 en 16
zijn gehaald.

**Permanente skips die de suite ten onrechte groen houden** (no-go 18):

| Bestand | Aantal | Reden |
|---|---|---|
| `tests/e2e/test_chaos.py` | **4** | *"implement in Wave 20 sprint"* — Stage D-5 |
| `tests/killgates/test_expansion_killgates.py` | 9 | unit nog niet geëvalueerd — legitiem |
| `tests/lookahead/test_feature_causality.py` | 2 | grens niet scherp — legitiem |
| `tests/test_multi_sleeve_book.py` · `test_neutral_book.py` · `test_factor_alpha.py` | 7 | data-caches afwezig |

---

## 7. Aanwezigheid van elk bestand uit §0.4, §0.5 en §0.6

### 7.1 §0.4 — De Phase 2-governancelaag (D-1)

| Bestand | Gemeten |
|---|---|
| `src/tradebot/validation/gates.py` | **ONTBREEKT** |
| `src/tradebot/validation/walk_forward.py` | **ONTBREEKT** |
| `src/tradebot/validation/dsr.py` | **ONTBREEKT** |
| `src/tradebot/validation/spa.py` | **ONTBREEKT** |
| `src/tradebot/registry/trial_counter.py` | **ONTBREEKT** |
| `scripts/check_banned_methods.py` | **ONTBREEKT** |
| `.github/workflows/research_gates.yml` | **ONTBREEKT** |
| `apps/run_gates.py` | **ONTBREEKT** |
| `tests/killgates/test_gate_cannot_be_bypassed.py` | **ONTBREEKT** |
| `tests/lookahead/test_truncation_invariance.py` | **ONTBREEKT** |
| `tests/lookahead/test_temporal_shift_invariance.py` | **ONTBREEKT** |
| `tests/lookahead/test_future_column_poisoning.py` | **ONTBREEKT** |
| `tests/lookahead/test_scaler_fit_causality.py` | **ONTBREEKT** |
| `tests/lookahead/test_label_horizon_purge.py` | **ONTBREEKT** |
| `tests/lookahead/test_determinism_reproducibility.py` | **ONTBREEKT** |

**15 van 15 ontbreken. D-1 is nooit gesloten.** `docs/model_risk_policy.md` beweert
dat promotie wordt geblokkeerd door de lookahead-suite; die bewering is vandaag
onwaar.

Wat er wél is en hergebruikt moet worden:

```
src/tradebot/backtest/metrics.py          DSR (Bailey–López de Prado)
src/tradebot/backtest/spa.py              Hansen's SPA
src/tradebot/cv/walk_forward.py           purged WF
src/tradebot/registry/hypothesis_ledger.py  M = 2776  (geverifieerd, §7.4)
src/tradebot/registry/preregistration.py  528 LOC, 3 Phase 6-registraties bevroren
src/tradebot/backtest/baseline_report.py  dwingt DSR + SPA al af op de baseline
```

### 7.2 §0.5 — Phase 6, stap 9 van 15

| Bestand | Gemeten |
|---|---|
| `src/tradebot/regime/buckets.py` (M0) | AANWEZIG |
| `src/tradebot/regime/markov.py` (M2) | AANWEZIG |
| `src/tradebot/portfolio/hrp.py` | AANWEZIG — **zonder enige gate** |
| `src/tradebot/risk/hmm_regime.py` | AANWEZIG — **levert nog het Viterbi-pad** |
| `src/tradebot/features/fracdiff.py` | AANWEZIG — niet gemigreerd naar L3 |
| `reports/ECONOMETRIC_DIAGNOSTICS.md` | AANWEZIG |
| `src/tradebot/train/meta_label.py` | **ONTBREEKT** |
| `src/tradebot/validation/feature_importance.py` | **ONTBREEKT** |
| `reports/GARCH_VS_EWMA_COMPETITION.md` | **ONTBREEKT** — H1 onbeslist |
| `reports/M0_VS_HMM_BENCHMARK.md` | **ONTBREEKT** — H2 onbeslist |
| `reports/META_LABELING_EVALUATION.md` | **ONTBREEKT** — H3 onbeslist |
| `reports/HRP_VS_INVERSE_VOL.md` | **ONTBREEKT** |

**De twee actieve no-go's zijn beide bevestigd:**

```bash
grep -n "raise\|NOT_ADMISSIBLE\|gate\|research" src/tradebot/portfolio/hrp.py
# (geen enkele treffer)
```

> **No-go 15 is actief.** `portfolio/hrp.py` is productie-toegankelijk zonder enig
> bewijs: geen `raise`, geen gate, geen `NOT_ADMISSIBLE`-markering.

```bash
grep -n "Viterbi\|def predict" src/tradebot/risk/hmm_regime.py
# 161:    def predict(self, returns) -> tuple[np.ndarray, np.ndarray]:
# 164:        DEFERRED (Phase 6, DI-1): `GaussianHMM.predict` levert het Viterbi-pad,
# 166:        smoothed, niet filtered, en schendt de regel uit audit sectie 10.2.
```

> **DI-1 staat open.** De module documenteert zijn eigen schending van §10.2.

**DVC-stages:** `data_sync`, `build_features`, `tune_hparams`, `train_cpcv` — **geen
engine-stage**. Bevestigd.

**Sovereign-audit** — `_AUTHORITATIVE_PATH` in `test_sovereign_wiring.py`:

```python
_AUTHORITATIVE_PATH = (
    "backtest/engine.py", "backtest/accounting.py",
    "execution/order_router.py", "execution/context.py", "execution/impact_model.py",
)
```

Bevestigd: `volatility/`, `regime/`, `labeling/`, `train/` en `portfolio/hrp.py`
worden **niet** gedekt.

### 7.3 §0.6 — Phase 7 is niet begonnen

| Bestand | Gemeten |
|---|---|
| `src/tradebot/monitoring/vol_forecast_monitor.py` | **ONTBREEKT** |
| `src/tradebot/monitoring/execution_drift.py` | **ONTBREEKT** |
| `src/tradebot/registry/champion_challenger.py` | **ONTBREEKT** |
| `src/tradebot/registry/MRM_generator.py` | **ONTBREEKT** |
| `apps/live_dashboard.py` | **ONTBREEKT** |
| `reports/live_dashboard.html` | **ONTBREEKT** |
| `reports/PAPER_TRADING_LOG.md` | **ONTBREEKT** |
| `reports/phase7_divergence_map.md` | **ONTBREEKT** |
| `conf/monitoring/` | **ONTBREEKT** |
| `tests/e2e/test_live_backtest_parity.py` | **ONTBREEKT** |
| `tests/e2e/test_crash_recovery.py` | **ONTBREEKT** |
| `tests/e2e/test_kill_switch_live.py` | **ONTBREEKT** |

**Aanwezig maar niet Phase 7-waardig — bevestigd:**

```python
class AlertSeverity(Enum):
    INFO = "INFO"; WARNING = "WARNING"; CRITICAL = "CRITICAL"     # geen HALT
```

```python
PSI_MODERATE: float = 0.10      # module-constante, niet conf/monitoring/
PSI_CRITICAL: float = 0.20
```

**De vijf C-items staan er woordelijk:**

| # | Meting |
|---|---|
| C6 | `execution_controller.py:62` `min_confidence: float = 0.55` |
| C4 | `execution_controller.py:95` `self._max_gross_notional: float = float("inf")` |
| C3 | `execution_controller.py:243` `def _check_position_limits(...)` |
| C1 | `circuit_breaker.py:54–63` eigen `max_drawdown_pct` / `max_intraday` / `max_daily_loss` |
| C2 | `circuit_breaker.py:266` `def _persist_trip(...)` |

**`live/` gebruikt geen enkele Phase 5-component:**

```bash
grep -rn "execution.context\|execution.order_router\|backtest.accounting\|RiskEngine\|HaltStore" src/tradebot/live/*.py
# (nul treffers)
grep -rn "oms.router\|OrderRouter" src/tradebot/live/*.py
# src/tradebot/live/engine.py:35:  from ..oms.router import OrderRouter
# src/tradebot/live/engine.py:195: self._router = OrderRouter(
```

De import op regel 35 is de **verkeerde** `OrderRouter`: `oms/router.py`, niet de
Phase 5 `execution/order_router.py` die een `RiskDecision` als verplicht eerste
argument eist. **Dat verschil is de bypass die Stage D moet sluiten.**

> **Afwijking 2:** de prompt presenteert de `oms.router`-regel als *treffer* van de
> eerste grep. Dat is zij niet — `OrderRouter` matcht geen van die vijf patronen.
> Zie §8.

### 7.4 De ledger

```python
from tradebot.registry.hypothesis_ledger import HypothesisLedger
HypothesisLedger().total_n_hypotheses()     # 2776
```

| Meting | Waarde |
|---|---|
| `M` | **2776** — exact overeenkomstig de prompt-header |
| `seed_total` | 2363 |
| Ledger-entries | 14 |
| `seed_note` | *"Itemised reconstruction quoted verbatim from `docs/WAVE_LOG.md` W20 §0.1: 2000 (audit §10/§11) + W14:204 + W15:21 + W16…"* |

**Risk `config_hash` `1b60cb664fbf9a2a`** is aanwezig in
`artefacts/governance/risk_config_registry.json` (2 entries: `47821e47fe2cec30` en
`1b60cb664fbf9a2a`) en in `artefacts/baseline/phase5_revaluation.json`,
`conf/research/preregistration_h2_hmm_vs_m0.yaml`, `docs/ARCHITECTURAL_DECISIONS.md`.
**Bevestigd.**

> **Afwijking 3:** `total_n_hypotheses` is een **methode op `HypothesisLedger`**,
> geen module-level functie. Zie §8.

### 7.5 §0.8 — Structurele schuld

| Meting | Gemeten | §0.8 | Komt overeen |
|---|---|---|---|
| Tests verzameld | 2.011 | 2.011 | **ja** |
| D-6 (>800 LOC) | **7** bestanden | 7 | **ja** |
| D-7 (apps >80 LOC) | **22 van 35** | 22 van 35 | **ja** |
| `__pycache__` in boom | 39 | 39 | **ja** |
| Open DI's | DI-1 … DI-15 | 13 open | te verifiëren in Stage E |

D-6, exact:

```
1181 src/tradebot/labeling/meta.py        1054 src/tradebot/backtest/evaluation.py
1117 src/tradebot/train/ensemble.py        955 src/tradebot/tune/objective.py
1056 src/tradebot/features/regime.py       899 src/tradebot/live/engine.py
                                           825 src/tradebot/portfolio/legacy_sizing.py
```

---

## 8. AFWIJKINGEN TEN OPZICHTE VAN §0

Vier. Geen ervan verandert een conclusie; drie corrigeren een cijfer.

| # | Prompt beweert | Gemeten | Gevolg |
|---|---|---|---|
| 1 | `x.jsonl` telt **35 regels** (§0.1 G-3) | **38 regels** | Geen. Het artefact, zijn herkomst en zijn defectklasse zijn ongewijzigd. De prompt telde vermoedelijk vóór de laatste testrun die eraan toevoegde — wat de bevinding juist **versterkt**: het bestand groeit nog steeds. |
| 2 | De grep op Phase 5-componenten levert **1 treffer** (`live/engine.py:35`) (§0.6) | **0 treffers**; de `oms.router`-import bestaat wel, maar matcht dat patroon niet | Geen — de bevinding wordt **strenger**: `live/` gebruikt aantoonbaar nul Phase 5-componenten, niet één verkeerde. |
| 3 | `M = 2776` komt uit `total_n_hypotheses()` (§B-3) | Het is `HypothesisLedger().total_n_hypotheses()`, een **methode** | Alleen voor de aanroep in Stage B-3. `M = 2776` is bevestigd. |
| 4 | **Vijf** `asyncio.create_task(...)` zonder referentie in `live/` (§D-2, DI-7) | **Vier**, alle in `live/feed.py` (regels 226, 228, 231, 236) | DI-7 blijft open en identiek van aard; de omvang is één kleiner. |

**Er zijn geen afwijkingen gevonden die een §0-conclusie omkeren.** Elke
structurele bevinding is gereproduceerd.

---

## 9. Wat deze meting toevoegt aan §0

Drie observaties die de prompt niet noemt en die de planning raken:

1. **`x.jsonl` groeit nog.** 38 in plaats van 35 regels betekent dat de schrijvende
   test sinds 2026-08-27 opnieuw heeft gedraaid. Stage A-1 stap 4 (de test naar
   `tmp_path` verplaatsen) is daarmee niet alleen opruimwerk maar het stoppen van
   een actief lek.

2. **De C-schijf-repo is nog steeds bereikbaar en nog steeds gevaarlijk.** `.git`
   is op 2026-08-27 14:50 aangeraakt — ná de laatste commit. Wat die aanraking was,
   is niet uit de meting af te leiden; de bundle-en-verifieer-volgorde uit A-1 is
   daarmee geen formaliteit.

3. **De vier `test_chaos.py`-skips citeren allemaal een C-schijf-pad.** Zij zijn
   daarmee tegelijk bewijs van E-2 (liegende caches) én van no-go 18 (permanent
   overgeslagen tests die de suite groen houden). Eén handeling in Stage A-2 en één
   in Stage D-5 sluiten ze allebei.

---

## 10. Conclusie en vrijgave naar Stage A

| Poort | Status |
|---|---|
| Elke §0-meting herhaald | **ja** — 100 % gereproduceerd |
| Afwijkingen genoteerd | **ja** — vier, §8 |
| Er is nog niets gewijzigd | **ja** — `git status` telt dezelfde drie regels als bij aanvang |

**De masterprompt mag als betrouwbaar startpunt worden gebruikt.**

Eerste handeling van Stage A-1, conform §12: de git-bundle — vóór enige wijziging,
inclusief het opruimen van de caches. De historie van 77 commits is het enige
artefact in dit project dat niet opnieuw te produceren is.

---

*Gemeten op `42555d2`, 2026-08-27. Geen enkel bestand is tijdens deze meting
gewijzigd.*
