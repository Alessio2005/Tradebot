# MASTER-PROMPT: PHASE 7/8 — CONSOLIDATIE, PRODUCTION READINESS & OPLEVERING

> **Fase:** 7/8 van 8 · **Prioriteit:** P0
> **Bindend brondocument:** `docs/ARCHITECTUUR_AUDIT_2026-08-22.md` — §14.1, §17, §17.1, §18, §18.1, §19, §21, §23, §26, §27, §28
> **Vervangt:** `Prompts-fases/fase_7_production_papertrading.md` (die prompt gaat uit van een voorwaarde die niet is vervuld — zie §0.7)
> **Startpunt:** `42555d2` · branch `main` · werkkopie `D:\Tradebot`
> **Gemeten op:** 2026-08-27 · **`M` bij aanvang:** `2776` · **Risk `config_hash`:** `1b60cb664fbf9a2a`
> **Faalbasislijn:** 4 · **Tests verzameld:** 2.011

---

## 0. DE GEMETEN STAAT VAN DE REPOSITORY

Deze sectie is geen samenvatting van de exit-rapporten. Het is een **meting**, uitgevoerd
op 2026-08-27 op `42555d2`, met de commando's erbij. Waar de meting afwijkt van wat een
rapport beweert, prevaleert de meting (audit §2.2: *documentatie is een hypothese, geen
waarheid*).

Wie deze fase uitvoert **herhaalt eerst elke meting hieronder** en noteert elke afwijking.
Een startpunt dat je niet zelf hebt gemeten, is een aanname.

### 0.1 Over git — de werkaanname klopt niet

> **De opdrachtgever gaat ervan uit dat de git-historie verloren is gegaan bij de
> verhuizing van de C-schijf naar de D-schijf. Dat is feitelijk onjuist.**

```bash
git -C D:/Tradebot rev-list --count HEAD     # 77
git -C D:/Tradebot log --oneline -1          # 42555d2 feat(regime): build the M2 filtered-only HMM...
git -C D:/Tradebot log --oneline --reverse | head -1
#                                            # 73a01a4 chore(repo): re-initialise version control with baseline snapshot
git -C D:/Tradebot branch                    # * main
du -sh D:/Tradebot/.git                      # 3.2M
```

De volledige historie van Phase 0 tot en met de lopende Phase 6 staat er: **77 commits**,
van de Phase 0-nulmeting `73a01a4` tot de M2 Filtered HMM `42555d2`. Auteur
`Tradebot Quant Team <algulizia@gmail.com>`. `registry/lineage.py::get_git_sha()`
resolvet. **D-9 is en blijft gesloten.**

Er zijn wél **drie echte git-problemen**, en ze zijn alle drie een gevolg van de
verhuizing:

#### G-1 — Er is geen remote. Nul off-machine kopie.

```bash
git -C D:/Tradebot remote -v     # (leeg)
```

Dit platform stelt dat elk MRM-rapport, elke ledger-entry en elk feature-artefact zonder
resolvable `git_sha` **invalide** is. Die hele bewijsketen — 77 commits, het
falsificatieregister, de hypothese-ledger met `M = 2776`, de bevroren pre-registraties —
staat op **één fysieke schijf zonder kopie**. De verhuizing van C: naar D: is precies het
soort gebeurtenis dat dit aan het licht brengt; de volgende keer kan hij minder goed
aflopen.

Dit is geen hygiëne-puntje maar een **single point of failure onder de volledige
governance-laag**, en het is het goedkoopste risico in dit hele document om weg te nemen.

#### G-2 — Er staat een tweede, uitgeholde repository op de C-schijf

```bash
git --git-dir=C:/Users/algul/Documents/Tradebot/.git rev-list --count HEAD   # 77
git --git-dir=C:/Users/algul/Documents/Tradebot/.git log --oneline -1        # 42555d2  (identiek)
git --git-dir=C:/Users/algul/Documents/Tradebot/.git \
    --work-tree=C:/Users/algul/Documents/Tradebot status --short | wc -l     # 637
```

`C:\Users\algul\Documents\Tradebot\` bevat nog een **volledige `.git` met dezelfde 77
commits en dezelfde HEAD**, maar de werkkopie is leeggehaald: alleen `src\tradebot\`
(leeg) staat er nog. Git ziet daar **637 verwijderde paden**.

Dat is een landmijn. Wie in die map per ongeluk `git add -A && git commit` draait, legt de
verwijdering van het volledige project vast in een historie die er verder identiek
uitziet. Er bestaat dan een 78e commit die het project wist, in een repo die van buiten
niet van de echte te onderscheiden is.

#### G-3 — De werkkopie is vervuild

```bash
git -C D:/Tradebot status --short
#  M Prompts-fases/fase_6_advanced_research.md
#  M x.jsonl
```

`x.jsonl` staat **in de projectroot, is git-tracked, en is een uitgelekt testartefact**:
35 regels gap-ledger met timestamps in 1970 en symbool `"B"` — synthetische testdata. Dit
is exact de klasse defect die `phase5_exit_report.md` §9.1 zelf benoemt (*"een test die
een governance-artefact vervuilt, ondermijnt de auditbaarheid die hij hoort te bewaken"*),
alleen dan in de root in plaats van in `artefacts/governance/`.

### 0.2 De echte schade van de verhuizing zit niet in git, maar in de omgeving

Dit is de bevinding die het meest kost als hij blijft staan, en hij is met geen enkel
rapport te vinden — alleen door te meten.

#### E-1 — De editable install wijst nog naar de C-schijf

```bash
cat "$(python -c 'import site;print(site.getsitepackages()[-1])')/__editable__.tradebot-0.4.0.pth"
# C:\Users\algul\Documents\Tradebot\src

python -c "import tradebot; print(tradebot.__path__)"
# ['C:\\Users\\algul\\Documents\\Tradebot\\src\\tradebot']    <- LEGE map

python -c "from tradebot.registry.lineage import get_git_sha"
# ModuleNotFoundError: No module named 'tradebot.registry'
```

De geïnstalleerde `tradebot` (0.4.0, systeem-Python 3.13.0) wijst naar de **pre-verhuizing
locatie**, die nog bestaat maar leeg is. Gevolg:

- `import tradebot` **slaagt** — als lege namespace-package;
- `import tradebot.<wat dan ook>` **faalt**;
- elke `python apps/...` of `python scripts/...` die op de geïnstalleerde package steunt,
  faalt of gedraagt zich onvoorspelbaar;
- `pytest` werkt alleen doordat het `src` zelf op het pad zet.

Een platform dat zijn hele bestaansrecht ontleent aan *"nul stille degradatie"* draait op
dit moment op een omgeving waarin de import zelf half kapot is. Phase 0 exit-criterium 2
(*100 % reproduceerbare build*) is op **deze** interpreter aantoonbaar niet meer waar.

#### E-2 — Alle 351 bytecode-caches liegen over hun herkomst

```bash
python - <<'PY'
import pathlib, marshal
h=t=0
for p in pathlib.Path('.').rglob('*.pyc'):
    if '.git' in p.parts: continue
    t+=1
    try:
        c=marshal.loads(p.read_bytes()[16:])
        if 'Documents' in c.co_filename: h+=1
    except Exception: pass
print(f'{h}/{t} .pyc met oud C-pad')
PY
# 351/351 .pyc met oud C-pad
```

Elke `.pyc` in de boom draagt nog `C:\Users\algul\Documents\Tradebot\...` als
`co_filename`. De mtimes zijn bij de verhuizing behouden, dus Python beschouwt ze als
geldig en gebruikt ze. **Elke traceback, elke warning en elke assertion-melding citeert
daardoor een bestandspad dat niet bestaat.** Zichtbaar in de laatste suite-run:

```
SKIPPED [1] C:\Users\algul\Documents\Tradebot\tests\e2e\test_chaos.py:19
D:\Tradebot\src\tradebot\risk\var.py:222: RuntimeWarning
```

Twee schijven in één rapport. Voor een platform dat provenance als kernwaarde voert, is
bewijsmateriaal dat naar een niet-bestaand pad verwijst geen cosmetisch probleem.

39 `__pycache__`-mappen staan in de boom; `.gitignore` sluit ze uit van git, maar niet van
de interpreter.

### 0.3 De ratchets — één staat ROOD

```bash
python scripts/check_hardcoded_params.py --strict ; echo $?
```

```
OVERSCHRIJDINGEN:
  validation/data_adequacy.py:      1 literal(s) in een door Phase 0-3 bestuurde module (budget 0)
  validation/diagnostics_report.py: 2 literal(s) in een door Phase 0-3 bestuurde module (budget 0)
  validation/econometrics.py:       8 literal(s) in een door Phase 0-3 bestuurde module (budget 0)
  validation/vol_metrics.py:        2 literal(s) in een door Phase 0-3 bestuurde module (budget 0)
  volatility/realized.py:           2 literal(s) > budget 0 (geen budget)

327 numerieke literal(s) in 107 bestand(en).
Toegestaan door de ratchet: 317.
exit code = 1
```

> **`.github/workflows/hygiene.yml` faalt vandaag.** De lopende Phase 6-modules hebben 15
> literals in vijf modules met budget 0 gezet. Dat is niet "nog even opruimen": de ratchet
> is het mechanisme dat criterium 6 van Phase 0 überhaupt afdwingbaar maakt, en hij staat
> rood. Zolang dat zo is, is elke uitspraak *"de gates zijn groen"* onjuist.

De andere twee ratchets staan wél goed:

```bash
python scripts/audit_fallbacks.py --strict ; echo $?    # 37 bevindingen, 0 blokkerend, exit 0
python -m pytest -q -p no:randomly                      # 4 failures, 2.011 verzameld
```

**De faalbasislijn is aantoonbaar van 6 naar 4 gedaald.** De vier resterende zijn exact de
pre-geregistreerde killgates op `cm_carry` en `cm_tsmom`; die horen rood te staan. De twee
property-bugs uit `phase5_exit_report.md` §11.5 zijn gerepareerd (`28a5222`, `a5b4dda`) en
het derandomized Hypothesis-profiel staat geregistreerd (`a7ddc5a`). **Phase 6
exit-criteria 15 en 16 zijn hiermee gehaald.**

### 0.4 D-1 is nooit gesloten — de Phase 2-governancelaag bestaat niet

Dit is de grootste structurele bevinding, en zij is niet nieuw:
`reports/phase3_exit_report.md` §0 en §5 melden hem expliciet. Geen enkele latere fase
heeft hem opgepakt.

**Gemeten — geen van deze bestaat:**

| Ontbrekend | Wat het moest doen |
|---|---|
| `src/tradebot/validation/gates.py` | `run_promotion_gates(...)` → onveranderlijk `GateResult` |
| `src/tradebot/validation/walk_forward.py` | Purged WF met expliciete embargo als L11-ingang |
| `src/tradebot/validation/dsr.py` | afdwingende DSR-wrapper, verplichte `M` |
| `src/tradebot/validation/spa.py` | Hansen's SPA als promotiegate |
| `src/tradebot/registry/trial_counter.py` | persistente, eerlijke `M` |
| `scripts/check_banned_methods.py` | AST-scan op `KFold`, `ShuffleSplit`, `shuffle=True` |
| `.github/workflows/research_gates.yml` | de blokkerende CI-gate |
| `apps/run_gates.py` | lokale gate-run vóór een PR |
| `tests/killgates/test_gate_cannot_be_bypassed.py` | de negatieve controle op de gate zelf |
| `tests/lookahead/test_truncation_invariance.py` | D-1 gate 1 |
| `tests/lookahead/test_temporal_shift_invariance.py` | D-1 gate 2 |
| `tests/lookahead/test_future_column_poisoning.py` | D-1 gate 3 |
| `tests/lookahead/test_scaler_fit_causality.py` | D-1 gate 4 |
| `tests/lookahead/test_label_horizon_purge.py` | D-1 gate 5 |
| `tests/lookahead/test_determinism_reproducibility.py` | D-1 gate 6 |

`src/tradebot/registry/promotion.py` bestaat, maar bevat uitsluitend een klasse
`PromotionGates` — **geen state machine** `REGISTERED → TESTED → CANDIDATE → PAPER →
CHAMPION`.

`docs/model_risk_policy.md` beweert nog steeds dat promotie wordt geblokkeerd door de
lookahead-suite. Die bewering is **vandaag onwaar**: er is geen CI-gate die haar afdwingt.

> **Waarom dit Phase 7 blokkeert en niet kan doorschuiven.** Phase 7 exit-criterium 8 eist
> dat voor elk model in productie de vijf acceptatiecriteria uit §26 aantoonbaar gelden,
> waaronder *DSR p < 0,05 na correctie voor `M`*. En §18.1 regel 3 eist champion/challenger-
> promotie op Diebold-Mariano. Beide vereisen precies de machinerie die niet bestaat. Een
> Phase 7 bovenop deze leemte produceert 60 dagen paper-trading waarvan de promotiebeslissing
> aan het eind door niets wordt afgedwongen. Dat is geen productiegereedheid maar een
> duurdere manier om hetzelfde papieren beleid te herhalen dat D-1 aanwijst.

Wat er wél is en hergebruikt moet worden in plaats van herbouwd:
`backtest/metrics.py` (correcte DSR incl. Euler-Mascheroni), `backtest/spa.py`,
`cv/walk_forward.py`, `registry/hypothesis_ledger.py` (`M = 2776`),
`registry/preregistration.py` (528 LOC, drie Phase 6-registraties bevroren),
`backtest/baseline_report.py` (dwingt DSR en SPA al af op de baseline).

### 0.5 Phase 6 staat op stap 9 van 15

**Klaar en gecommit:**

| # | Deliverable | Commit |
|---|---|---|
| 3 | derandomized Hypothesis-profiel | `a7ddc5a` |
| 4 | `har_rv_forecast` non-negatief · `evt_gpd_var` eindig | `28a5222`, `a5b4dda` |
| 2 | `validation/data_adequacy.py` + `conf/model/adequacy.yaml` | `1b604a8` |
| 1 | drie pre-registraties (H1/H2/H3) bevroren | `f87bac1` |
| 7 | `validation/econometrics.py` | `fbd519a` |
| 11 | `volatility/realized.py` + range-estimators | `fd43c2f`, `dbb5f59` |
| 10 | `volatility/garch.py` (GARCH/GJR/EGARCH/APARCH) | `c3df3e3`, `8f96f0f` |
| 12 | `validation/vol_metrics.py` (QLIKE, MZ, DM-HLN) | `ee67919` |
| 19 | `labeling/triple_barrier.py` | `7b36e89` |
| 14 | `regime/buckets.py` — M0 | `97d9926` |
| 15 | `regime/markov.py` — M2 filtered-only | `42555d2` |
| 17 | `tests/lookahead/test_filtered_only_enforcement.py` | `42555d2` |
| 9 | `reports/ECONOMETRIC_DIAGNOSTICS.md` | aanwezig |

**Ontbreekt nog:**

| # | Deliverable | Status |
|---|---|---|
| 5 | DVC-stage voor de authoritative engine | ontbreekt — `dvc.yaml` heeft alleen `data_sync`, `build_features`, `tune_hparams`, `train_cpcv` |
| 6 | sovereign-audit uitgebreid naar Phase 6-modules | `_AUTHORITATIVE_PATH` in `test_sovereign_wiring.py` noemt `volatility/`, `regime/`, `labeling/`, `train/`, `portfolio/hrp.py` niet |
| 8 | `features/fracdiff.py` onder het L3-contract | legacy-module aanwezig, niet gemigreerd |
| 13 | `reports/GARCH_VS_EWMA_COMPETITION.md` | ontbreekt — H1 is niet beslist |
| 16 | `risk/hmm_regime.py` herontworpen | **niet gedaan** — `predict()` levert nog het Viterbi-pad; **DI-1 staat open** |
| 18 | `reports/M0_VS_HMM_BENCHMARK.md` | ontbreekt — H2 is niet beslist |
| 20 | `train/meta_label.py` | ontbreekt |
| 21 | `validation/feature_importance.py` (MDI/SFI) | ontbreekt |
| 22 | `reports/META_LABELING_EVALUATION.md` | ontbreekt — H3 is niet beslist |
| 23 | `portfolio/hrp.py` research-gated | bestand bestaat, **maar zonder enige gate** |
| 24 | `reports/HRP_VS_INVERSE_VOL.md` | ontbreekt |
| 25–28 | ledger-oordelen, registers, exit-rapport | ontbreken |

> **No-go-conditie 15 van Phase 6 is op dit moment actief:** `portfolio/hrp.py` is
> productie-toegankelijk zonder bewijs. Er staat geen `raise`, geen gate, geen
> `NOT_ADMISSIBLE`-markering in het bestand.

**Eén meting die de planning stuurt** — `artefacts/governance/phase6_data_adequacy.json`
op `a5b4dda`:

| Modelklasse | Adequaat |
|---|---|
| `garch:*` (6 symbolen) | **ja** — 12 vensters, 495 obs elk, eis 250 |
| `hmm_k2`, `hmm_k3` | **ja** |
| `meta_labeling` | **ja** |
| `hrp` | **ja** |
| **`har_rv`** | **NEE — 0,00 % van de dagen heeft voldoende 5m-dekking tegen een eis van 80 %** |

HAR-RV is met de huidige data **niet te fitten**. Dat is geen tegenslag maar precies wat
de Data Adequacy Gate hoort te doen, en het bevestigt Open Question 1 uit §27 voor de
derde keer. Deliverable 11 is daarmee gedeeltelijk onuitvoerbaar en het oordeel luidt
`UNPROVEN — insufficient data`, met deze cijfers erbij. **Niet `FALSIFIED`** (Phase 6
no-go 8).

### 0.6 Phase 7 is niet begonnen — en `live/` staat verder van Phase 5 af dan het rapport suggereert

```bash
grep -rn "execution.context\|execution.order_router\|backtest.accounting\|RiskEngine\|HaltStore" src/tradebot/live/*.py
# src/tradebot/live/engine.py:35:  from ..oms.router import OrderRouter
```

Dat is de **enige** treffer, en het is de **verkeerde `OrderRouter`**: `oms/router.py`, niet
de Phase 5 `execution/order_router.py` die een `RiskDecision` als verplicht eerste argument
eist. `live/` gebruikt op dit moment **geen enkele** Phase 5-component: geen
`ExecutionContext`, geen `Ledger`, geen `RiskEngine`, geen `HaltStore`.

De vijf C-items uit `phase5_exit_report.md` §17.1 staan er nog woordelijk:

```
live/execution_controller.py:62   min_confidence: float = 0.55        <- C6, alpha-parameter in L9
live/execution_controller.py:95   self._max_gross_notional = float("inf")   <- C4, onbegrensd bij omissie
live/execution_controller.py:243  def _check_position_limits(...)     <- C3, eigen notional-limieten
live/circuit_breaker.py:79-85     max_drawdown_pct=0.08, max_intraday=0.05, max_daily_loss=0.03
                                                                      <- C1, eigen breaker
live/circuit_breaker.py:266       def _persist_trip(...)              <- C2, journaliseert, herstelt niet
```

**Ontbrekende Phase 7-deliverables (gemeten):** `monitoring/vol_forecast_monitor.py`,
`monitoring/execution_drift.py`, `registry/champion_challenger.py`,
`registry/MRM_generator.py`, `apps/live_dashboard.py`, `reports/live_dashboard.html`,
`reports/PAPER_TRADING_LOG.md`, `reports/phase7_divergence_map.md`, `conf/monitoring/`,
`tests/e2e/test_live_backtest_parity.py`, `tests/e2e/test_crash_recovery.py`,
`tests/e2e/test_kill_switch_live.py`.

**Aanwezig maar niet Phase 7-waardig:**

- `monitoring/alerts.py` — `AlertSeverity` kent `INFO`/`WARNING`/`CRITICAL`, **geen `HALT`**.
  Deliverable 8 eist vier niveaus waarbij `HALT` het systeem uitschakelt zonder te vragen.
- `monitoring/drift.py` — PSI-drempels staan als module-constanten (`PSI_MODERATE = 0.10`),
  niet in `conf/monitoring/`. Deliverable 4 en de faseregel *"alle drempels uit
  `conf/monitoring/`, vóór de start vastgelegd en gehasht"* zijn daarmee geschonden.
- `docs/RUNBOOK.md` — bestaat (157 regels) maar is het **legacy Wave-runbook**, met een
  beslisboom rond `state/circuit_log.jsonl` en `live.mode`. Het beschrijft niet het systeem
  dat Phase 5 heeft gebouwd.
- `tests/e2e/` — bevat `test_chaos.py` (4 tests, alle vier `SKIPPED` met
  *"implement in Wave 20 sprint"*) en `test_paper_trade_smoke.py`.

### 0.7 Waarom deze prompt `fase_7_production_papertrading.md` vervangt

Die prompt opent met: *"Voorwaarde: Phase 0 t/m 6 volledig afgerond. Bit-identieke
backtest-live pariteit uit Phase 5 is een harde ingangsvoorwaarde."*

Gemeten: Phase 6 staat op stap 9 van 15, Phase 2 is nooit gebouwd, en de backtest-live
pariteit uit Phase 5 is bewezen **op de gedeelde `ExecutionContext`** — niet op `live/`,
dat die context niet gebruikt. `phase5_exit_report.md` zegt dat zelf, in criterium 9:
*"GEDEELTELIJK — bewezen op de gedeelde context; `live/` zelf is Phase 7."*

De ingangsvoorwaarde is dus niet vervuld. Deze prompt lost dat op door de fase in **vijf
stages** te knippen met een harde poort tussen elke stage, in plaats van te doen alsof de
voorwaarde geldt.

### 0.8 Overige gemeten feiten

| Meting | Waarde |
|---|---|
| LOC in `src/` | 61.199 over ~230 modules |
| Testbestanden | 101 · 2.011 tests verzameld |
| `conf/`-YAML's | 32 over 12 domeinmappen |
| D-6 (>800 LOC) | **7** bestanden — `labeling/meta.py` 1.181, `train/ensemble.py` 1.117, `features/regime.py` 1.056, `backtest/evaluation.py` 1.054, `tune/objective.py` 955, `live/engine.py` 899, `portfolio/legacy_sizing.py` 825 |
| D-7 (apps >80 LOC) | **22 van 35** |
| Shadow trees in root | `catboost_info/`, `logs/`, `outputs/`, 39× `__pycache__` |
| DVC | `data/pit_store.dvc` (228 files, 2,3 MB); `dvc.yaml` zonder engine-stage |

---

## 1. ROL EN CONTEXT

Je acteert als **Head of Trading Systems & Production Operations**, met daarnaast — en dat
is nieuw ten opzichte van de oorspronkelijke Phase 7 — de rol van **Head of Model Risk
Governance**, omdat de governancelaag die deze fase hoort te consumeren niet blijkt te
bestaan.

Vanaf hier is elke fout niet langer een teleurstellend rapport maar een operationeel
incident. Jouw taak is een systeem dat zichzelf uitschakelt vóórdat het schade aanricht,
en dat elke afwijking tussen simulatie en werkelijkheid zichtbaar maakt op de dag dat zij
ontstaat.

**Relevante lagen (§19):**

| Laag | Rol in deze fase |
|---|---|
| **L9/L10** | Execution & Backtesting — moeten bit-identiek gedrag vertonen met de live controller |
| **L11** | Statistical Validation — **volledig te bouwen** (Stage B); champion/challenger via Diebold-Mariano |
| **L12** | Governance, Ledger & Audit Register — elke promotie cryptografisch traceerbaar |
| **L13** | Live Operations, Paper Trading & Monitoring — **primaire laag**, volledig te bouwen |

**Bindende governance-regel (§18.1, regel 3):**
> *"Een challenger-model vervangt het champion-model pas na minimaal 60 dagen OOS
> paper-trading waarin het de champion statistisch significant verslaat
> (Diebold-Mariano p < 0.05)."*

**Bindende pijplijn (§21):**
`... → Event-Driven Execution Simulation (TCA) → Ledger Registration → Paper Trading (60 dagen) → Production Promotion`

**Doctrine (§28):** *"Pas wanneer dat systeem aantoonbaar live paper-trades uitvoert met
een bewezen statistische edge, zou hij stapsgewijs complexere modellen toelaten via het
falsificatieregister."*

**Erfenis die onverkort geldt** (uit `phase5_exit_report.md` en `fase_6` §0):

- Er is **precies één** authoritative engine: `backtest/engine.py::EventDrivenEngine`.
  Deze fase herontwerpt hem niet en bouwt er geen tweede.
- De latency-conventie is **`shift(2)`**, niet `shift(1)`.
- Elke kostenclaim draagt de labels **`IMPACT_UNCALIBRATED`** (`eta = 2,9919`,
  `kappa_d = 0,6720`) en **`SPREAD_ASSUMED`** (1,0 bp). Zij verdwijnen pas met L2-data.
- **Fees domineren.** Turnover is de scherpste vijand van complexiteit.
- De `cluster_cap` is **vacuous** op dit universum; alleen `max_concentration` (0,40)
  beschermt tegen concentratie.
- Het universum is **survivorship-biased** (DI-15): zes ex-post gekozen overlevers.
- **De risicolaag maakt geen alpha.** Alle vier Phase 3-tracks blijven netto negatief.
  Dat is de openstaande vraag, en geen enkele limiet of monitor lost hem op.

---

## 2. DOEL VAN DE FASE

Breng het platform van *"een gevalideerde onderzoeksketen met een half afgemaakte
onderzoeksfase en een niet-aangesloten live-laag"* naar *"een operationeel, afgedwongen,
overdraagbaar systeem waarvan elke claim door een machine wordt bewaakt in plaats van door
een document"*.

Concreet, na deze fase:

1. De omgeving is herbouwd en de git-historie is buiten deze machine veiliggesteld.
2. Elke CI-gate staat groen — en is aantoonbaar in staat rood te worden.
3. De L11/L12-governancelaag bestaat, draait blokkerend in CI, en heeft een bewezen
   negatieve controle. **D-1 is gesloten.**
4. Phase 6 heeft voor H1, H2 en H3 een expliciet, in de ledger vastgelegd oordeel —
   `PROMOTED`, `UNPROVEN — insufficient data` of `FALSIFIED`.
5. `live/` deelt één codepad met `backtest/`, en dat is aantoonbaar.
6. Het systeem draait 60 opeenvolgende schone dagen paper-trading met werkende monitoring,
   alerts en kill switches.
7. Het project is **overdraagbaar**: een tweede persoon kan het opzetten, starten,
   halteren, herstarten en de resultaten interpreteren op uitsluitend de documentatie.

**Wat expliciet NIET in scope is:**

- Herontwerp van de soevereine risicolaag (Phase 4) of de authoritative engine (Phase 5).
- Nieuwe alpha-hypotheses. Deze fase evalueert en operationaliseert; zij verzint niets.
- M3 Markov-Switching GARCH.
- Orderboekdata-acquisitie, tenzij die data buiten deze fase beschikbaar komt. **Zie §11.1
  — dit is wél de belangrijkste openstaande investeringsvraag en moet als besluit worden
  voorgelegd, niet stil doorgeschoven.**
- Echt kapitaal. Deze fase eindigt bij paper trading.

---

## 3. STAGE A — FUNDAMENT HERSTELLEN

> **Poort naar Stage B:** alle drie de ratchets exit 0, `pip install -e .` werkt vanaf een
> schone interpreter, en de historie staat aantoonbaar buiten deze machine.

Geen enkele meting in dit project is betrouwbaar zolang de omgeving half kapot is. Deze
stage kost uren, niet dagen, en alles erna hangt ervan af.

### A-1 — Git veiligstellen en de landmijn opruimen

1. **Maak eerst een volledige, verifieerbare kopie van `.git`** vóór je iets aanraakt:
   ```bash
   git -C D:/Tradebot bundle create D:/backup/tradebot-42555d2.bundle --all
   git bundle verify D:/backup/tradebot-42555d2.bundle
   ```
   Een bundle is één bestand, bevat de volledige historie en is offline verifieerbaar.
2. **Richt een remote in.** Leg de keuze vast in `docs/ARCHITECTURAL_DECISIONS.md`: een
   private hosted remote, een tweede fysieke schijf, of beide. Dit is een besluit van de
   opdrachtgever (proprietary code, private repo) — **leg het voor, kies niet zelfstandig
   een publieke host**. Push daarna `main` plus alle tags en verifieer met
   `git ls-remote`.
3. **Ruim de C-schijf-repo op.** `C:\Users\algul\Documents\Tradebot\` bevat een `.git` met
   dezelfde 77 commits en een uitgeholde werkkopie. Bundle hem eerst
   (`tradebot-CDRIVE-orphan.bundle`), verifieer dat zijn HEAD `42555d2` is en dat hij geen
   commits bevat die D: niet heeft:
   ```bash
   git -C D:/Tradebot fetch C:/Users/algul/Documents/Tradebot/.git 'refs/*:refs/cdrive/*'
   git -C D:/Tradebot log --oneline main..refs/cdrive/heads/main   # moet leeg zijn
   ```
   **Pas als die uitvoer leeg is** en de bundle is geverifieerd, verwijder je de map.
   Verwijder niets op basis van deze prompt alleen — verifieer eerst.
4. **Ruim de werkkopie op.** `x.jsonl` is een uitgelekt testartefact: verwijder het uit git
   (`git rm`), voeg het patroon toe aan `.gitignore`, en — belangrijker — **vind de test die
   het schrijft en laat hem naar `tmp_path` schrijven**, met een bewakende test die faalt
   zodra er opnieuw een artefact in de root verschijnt. Dat is exact het patroon dat
   `phase5_exit_report.md` §9.1 voor het governance-register toepaste.

### A-2 — De omgeving herbouwen

1. Verwijder alle 351 stale `.pyc` en 39 `__pycache__`-mappen:
   ```bash
   find . -name "__pycache__" -type d -not -path "./.git/*" -prune -exec rm -rf {} +
   find . -name "*.pyc" -not -path "./.git/*" -delete
   ```
2. Deïnstalleer de kapotte editable install en herinstalleer vanaf `D:\Tradebot`:
   ```bash
   python -m pip uninstall -y tradebot
   python -m pip install -e . --no-deps
   python -c "import tradebot; print(tradebot.__file__)"   # moet D:\Tradebot\src\... zijn
   python -c "from tradebot.registry.lineage import get_git_sha; print(get_git_sha())"
   ```
3. **Herhaal de Phase 0-verificatie van criterium 2 in een verse venv**, want de bestaande
   `C:/tbvenv` is van vóór de verhuizing en zijn status is onbekend:
   ```bash
   python -m venv D:/venv/tradebot
   D:/venv/tradebot/Scripts/python -m pip install -r requirements.lock
   D:/venv/tradebot/Scripts/python -m pip install -e . --no-deps
   D:/venv/tradebot/Scripts/python -m pytest tests/unit -q
   ```
   Documenteer welke interpreter vanaf nu de referentie is, in `docs/RUNBOOK.md`.
4. **Bewijs dat de stale-cache-klasse niet kan terugkeren.** Voeg een test toe die elke
   `.pyc` in de boom controleert op een `co_filename` buiten de repository-root. Die test
   moet aantoonbaar rood worden op een geïnjecteerd voorbeeld.

### A-3 — De ratchet groen krijgen zonder hem te verruimen

De 15 literals in `validation/{data_adequacy,diagnostics_report,econometrics,vol_metrics}.py`
en `volatility/realized.py` zijn Phase 6-werk in uitvoering.

**Verplaats ze naar `conf/`.** Significantiedrempels, minimum-observatie-eisen,
dekkingspercentages en sampling-frequenties zijn precies het soort parameter waarvoor de
ratchet bestaat.

- **Verboden:** het budget van deze vijf modules ophogen. Budgetten mogen uitsluitend
  omlaag (Phase 0-besluit, 2026-08-22).
- Elke verplaatste drempel gaat in `conf/model/` of `conf/validation/`, wordt gevalideerd
  tegen een Pydantic-schema, en gaat mee in de `config_hash`.
- Draai daarna `check_hardcoded_params.py --strict` en eis exit 0.

### A-4 — Aantonen dat de gates rood kunnen worden

Voor elk van de drie ratchets: injecteer op een wegwerpbranch een schending, bewijs dat de
gate faalt, en verwijder de branch. Dit is de Phase 0-negatieve-controle en hij moet ná de
omgevingsherbouw opnieuw worden gedraaid, want de omgeving was toen anders.

### A-5 — Stage A-rapport

`reports/phase7_foundation_report.md`: per bevinding uit §0.1–§0.3 de meting vóór, de
handeling, en de meting erna. Inclusief de nieuwe referentie-interpreter, de remote, en de
bundle-hashes.

---

## 4. STAGE B — D-1 SLUITEN: DE GOVERNANCELAAG BOUWEN

> **Poort naar Stage C:** `research_gates.yml` blokkeert aantoonbaar een merge met een
> lekkend model, en `test_gate_cannot_be_bypassed.py` is aantoonbaar rood geweest.

Dit is de fase-2-opdracht, drie fasen te laat, en zij is niet over te slaan (§0.4).
**Bouw hem zoals `fase_2_research_falsification.md` hem voorschrijft** — die prompt blijft
bindend voor deze stage. Hieronder alleen wat er sinds die prompt is veranderd.

### B-1 — Hergebruik, herbouw niet

| Nodig | Bestaat al als | Handeling |
|---|---|---|
| DSR | `backtest/metrics.py` (Bailey–López de Prado, Euler-Mascheroni correct) | **RETAIN** — dunne, afdwingende wrapper in `validation/dsr.py`, `M` verplicht, geen default |
| SPA | `backtest/spa.py` | wrappen als gate |
| Purged WF | `cv/walk_forward.py` | wrappen; embargo uit `conf/validation/` |
| Ledger | `registry/hypothesis_ledger.py` (`M = 2776`) | **RETAIN** — append-only, atomair |
| Pre-registratie | `registry/preregistration.py` (528 LOC) | **RETAIN** — draagt al de drie Phase 6-registraties |
| Handhaving op de baseline | `backtest/baseline_report.py` | het patroon dat `gates.py` generiek moet maken |

### B-2 — De zes lookahead-tests

Bouw ze als **zelfstandige, benoemde bestanden**. De causaliteitsdekking bestaat
grotendeels al, verspreid over `test_feature_causality.py` (310 tests),
`test_baseline_causality.py` (33), `test_engine_causality.py` (20) en
`test_filtered_only_enforcement.py`. **Extraheer en hernoem** in plaats van te dupliceren;
laat de bestaande suites ernaar verwijzen.

Twee gaten die `phase3_exit_report.md` §2 expliciet noemt en die je wél moet bouwen:

- `test_temporal_shift_invariance.py` — **bestaat nergens**;
- `test_future_column_poisoning.py` — bestaat alleen als negatieve-skip-variant, niet als
  generieke kolom-injectie.

**Elke test moet eerst aantoonbaar rood zijn op een bewust lekkend referentiemodel.** Leg
per test vast: welk lek, welke melding, welk commando. Een test die nooit rood is geweest,
is geen test.

### B-3 — `M` eerlijk maken

Bouw `registry/trial_counter.py`. `M = 2776` komt nu uit `total_n_hypotheses()`. Documenteer
de reconstructiemethode én de onzekerheid. **Een te lage `M` maakt de DSR structureel te
optimistisch** — op 1.743 bars is dat dodelijk.

Neem de Phase 3-les over: `M` voor een meting komt uit een **bevroren** pre-registratie,
niet live uit de ledger, anders is de meting niet reproduceerbaar.

### B-4 — Promotie-state-machine

Herschrijf `registry/promotion.py` naar `REGISTERED → TESTED → CANDIDATE → PAPER →
CHAMPION`, eenrichtingsverkeer behalve `→ FALSIFIED`, met per overgang de vereiste
bewijslast. **De overgang `PAPER → CHAMPION` is dezelfde poort die Stage D nodig heeft** —
bouw hem één keer, hier.

### B-5 — Banned methods en CI

`scripts/check_banned_methods.py` (AST, geen regex) plus
`.github/workflows/research_gates.yml` als **required check**. Blokkeren, niet
waarschuwen.

### B-6 — De negatieve controle op de gate zelf

`tests/killgates/test_gate_cannot_be_bypassed.py`: een model met `close.shift(-1)` moet
worden geweigerd. **Laat de gate dit model door, dan is Stage B niet afgerond, ongeacht
hoe groen de rest is.**

### B-7 — Documentatie eerlijk maken

Corrigeer `docs/model_risk_policy.md` zodat het verwijst naar de zes tests die nu
daadwerkelijk bestaan. **D-1 gesloten.** Schrijf `reports/phase2_exit_report.md` — dat
rapport is nooit geschreven en de fase is nooit formeel afgesloten.

---

## 5. STAGE C — PHASE 6 AFMAKEN EN BESLISSEN

> **Poort naar Stage D:** H1, H2 en H3 hebben elk een ledger-oordeel met statistische
> onderbouwing, `M`-bijdrage en beide kostenlabels.

`Prompts-fases/fase_6_advanced_research.md` blijft **onverkort bindend** voor deze stage,
inclusief zijn 20 exit-criteria en 15 no-go-condities. Hieronder alleen de aanpassingen die
uit de meting volgen.

### C-1 — Sluit eerst de twee actieve no-go's

1. **`portfolio/hrp.py` heeft geen research-gate** (no-go 15). Bouw hem als **crash**, niet
   als vlag, vóórdat je iets anders doet.
2. **`risk/hmm_regime.py` levert nog het Viterbi-pad** — smoothed, en dus in strijd met
   §10.2 (DI-1). `regime/markov.py` bestaat inmiddels met het correcte filtered-only
   contract. Herontwerp `hmm_regime.py` naar dat contract, of verwijder hem en herwijs zijn
   consumenten. **De EMA-crossover-fallback mag in geen enkele vorm terugkeren** (no-go 5).
   Breid `test_filtered_only_enforcement.py` uit zodat hij ook dit pad afdekt.

### C-2 — HAR-RV: het oordeel is al gemeten

De Data Adequacy Gate meet **0,00 % 5m-dekking** tegen een eis van 80 %. Het oordeel luidt
`UNPROVEN — insufficient data`, met dat cijfer erbij, en de trial telt **niet** mee in `M`
(er heeft geen search plaatsgevonden). **Niet `FALSIFIED`** — no-go 8.

Rapporteer expliciet wat daarmee niet is getoetst: HAR-RV als Level 3-uitdager blijft
onbeslist, en de QLIKE-competitie draait op de daily range-proxies uit `volatility/realized.py`
in plaats van op echte Realized Variance. Dat is een **beperking van de competitie** en
hoort in `GARCH_VS_EWMA_COMPETITION.md`, niet in een voetnoot.

### C-3 — De vijf openstaande deliverables

Stap 7 (QLIKE-competitie), 11 (M0 vs. HMM door de engine), 12–13 (meta-labeling), 14 (HRP),
plus de resterende infrastructuur: `train/meta_label.py`,
`validation/feature_importance.py`, de DVC-stage en de sovereign-audit-uitbreiding.

**Voor elk oordeel geldt onverkort:**

- negatieve controle per statistische toets (geshuffelde reeks, gerandomiseerde labels,
  permutatie van het RV-doel) — **een toets die ook op ruis significant is, meet niets**;
- economische toets bovenop de statistische: door `EventDrivenEngine`, met
  `config_hash = 1b60cb664fbf9a2a`, `shift(2)`, turnover-delta en fees-delta;
- **spread-sensitiviteit:** bij welke aangenomen half-spread verdwijnt de gemeten
  verbetering? Verdwijnt zij al bij 3 bp, dan is de promotie een spread-aanname;
- het aantal **effectief handelende bars**, niet de vensterlengte — `long_only_equal_weight`
  halteert op 2022-05-10 en handelt daarna nooit meer;
- `M` vóór en ná, met het verschil verklaard.

**Draai voor elk gepromoveerd model de Stage B-gates.** Dat is de reden dat Stage B vóór
Stage C komt: zonder die gates is een promotie in Phase 6 opnieuw een bewering.

### C-4 — Verwacht een negatieve uitkomst en schrijf hem net zo zorgvuldig op

De meest waarschijnlijke uitkomst is dat **EWMA(0,94) en M0 blijven staan** en dat CatBoost
`ARCHIVED` blijft. Dat is een compleet, publiceerbaar resultaat en aanzienlijk goedkoper
dan de live ontdekking ervan.

---

## 6. STAGE D — PHASE 7: PRODUCTIE EN PAPER TRADING

> **Poort naar Stage E:** 60 opeenvolgende schone dagen.

Vanaf hier volgt `Prompts-fases/fase_7_production_papertrading.md` in zijn oorspronkelijke
vorm, met de onderstaande aanscherpingen uit de meting.

### D-1 — Divergence map, en de correctie op zijn premisse

Stap 1 van de oorspronkelijke prompt vraagt om het opsporen van *duplicatie* tussen `live/`
en `backtest/`. De meting laat iets anders zien: er is geen duplicatie maar **twee volledig
gescheiden implementaties**. `live/` gebruikt geen enkele Phase 5-component.

`reports/phase7_divergence_map.md` moet daarom per component vaststellen: wat gebruikt
`live/` nu, wat is de Phase 5-tegenhanger, en wat is het risico op gedragsdivergentie. Let
in het bijzonder op `oms/router.py` versus `execution/order_router.py` — de laatste eist een
`RiskDecision` als verplicht eerste argument, de eerste niet. **Dat verschil is precies de
bypass die deze fase moet sluiten.**

### D-2 — De vijf C-items, met C4 als scherpste

| # | Wat weg moet | Vervanger |
|---|---|---|
| C1 | `circuit_breaker.py` eigen drempels | `risk/kill_switches.py` via `ExecutionContext` |
| C2 | in-memory HALT-state | `risk.kill_switches.HaltStore` — bestaat, persistent, append-only |
| C3 | `_check_position_limits` | `risk.max_position_pct` + `risk.gross_cap` uit het besluit |
| C4 | `_max_gross_notional = float("inf")` | idem — **fail-fast bij ontbrekende config** |
| C6 | `min_confidence = 0.55` | **verwijderen** — audit §14 sluit modelvertrouwen als sizingparameter uit |

> **C4 verdient een aparte regel.** Een default van oneindig betekent dat wie de setter
> vergeet, live handelt zonder gross-limiet. Dat is de *"stilzwijgend degraderen naar geen
> limiet"*-modus die §23 verbiedt, in de enige laag waar hij echt geld kost.

Breid de statische audit uit `test_sovereign_wiring.py` uit naar de `live/`-modules, met een
test die bewijst dat de uitbreiding een schending detecteert.

Sluit ook **DI-7**: vijf `asyncio.create_task(...)` in `live/` zonder vastgehouden
referentie. Een niet-vastgehouden task kan door de GC worden opgeruimd, waardoor een feed-
of monitortaak stilletjes verdwijnt — een stille degradatie in de live-loop.

### D-3 — Monitoring: bouwen én harden

**Nieuw:** `monitoring/vol_forecast_monitor.py` (live QLIKE + Mincer-Zarnowitz),
`monitoring/execution_drift.py` (per order: voorspelde vs. werkelijke fill-prijs, -tijd,
-hoeveelheid).

**Harden van wat bestaat:**

- `monitoring/alerts.py` — voeg `HALT` toe als vierde niveau. Een `HALT` schakelt het
  systeem uit; hij vraagt geen toestemming.
- `monitoring/drift.py` — haal de PSI-drempels uit de module-constanten en zet ze in
  `conf/monitoring/`.
- `conf/monitoring/` bestaat niet. Bouw hem, valideer tegen een Pydantic-schema, en
  **hash de configuratie vóór de 60-daagse periode start**. Een drempel die achteraf wordt
  vastgesteld, is geen drempel.

### D-4 — Runbook: herschrijven, niet aanvullen

`docs/RUNBOOK.md` beschrijft het Wave-tijdperk. Herschrijf hem voor het systeem dat Phase 5
heeft gebouwd. **Toets hem zoals exit-criterium 9 eist:** een tweede persoon start,
halteert en herstart het systeem uitsluitend op basis van dit document. Slaagt dat niet,
dan is het runbook niet af — ongeacht hoe compleet het oogt.

### D-5 — De e2e-tests, en de vier skips

`tests/e2e/test_chaos.py` bevat vier tests die sinds Wave 20 `SKIPPED` staan met
*"implement in Wave 20 sprint"*. Zij dekken precies wat deze fase moet bewijzen: live
engine-integratie, OMS-router, signal handler. **Implementeer ze of verwijder ze** — een
permanent overgeslagen test is een gate die niets bewaakt en de suite ten onrechte groen
houdt.

Bouw daarnaast `test_live_backtest_parity.py`, `test_crash_recovery.py` (SIGKILL op
minimaal drie punten in de loop) en `test_kill_switch_live.py`.

### D-6 — De klok

60 opeenvolgende dagen, dagelijkse pariteitstest, append-only
`reports/PAPER_TRADING_LOG.md`. **De teller herstart bij elke runtime crash en bij elke
onverklaarde execution drift boven de drempel.** Een verklaard verschil (exchange-outage)
wordt gedocumenteerd en breekt de reeks niet; een onverklaard verschil wel.

> **Wat er in de 60 dagen daadwerkelijk draait, moet vooraf worden vastgelegd.** Alle vier
> Phase 3-tracks zijn netto negatief; als Stage C niets promoveert, is de champion de
> Phase 3-baseline plus de soevereine risicolaag — een systeem waarvan bekend is dat het
> geld verliest. **Dat is een geldige paper-trading-champion**, want deze fase toetst
> operationele gereedheid, niet winstgevendheid. Maar het moet expliciet zo worden
> geregistreerd, zodat niemand het later als een verwachting van rendement leest.

---

## 7. STAGE E — PHASE 8: OPLEVERING

> Deze stage bestaat niet in het oorspronkelijke fasemodel. Zij is toegevoegd omdat
> *"kant en klaar"* meer eist dan *"de laatste fase is afgerond"*.

### E-1 — Eén reproductiepad, van niets naar resultaat

Eén gedocumenteerd commando-pad dat vanaf een schone machine leidt tot een gereproduceerd
baselineresultaat: clone → venv → lockfile → `dvc pull` → ingest → features → baseline →
rapport. Getoetst op een machine die het project niet eerder heeft gedraaid.

### E-2 — De backlog eerlijk afsluiten

`docs/DEFERRED_ISSUES.md` telt 13 openstaande DI's, waarvan er meerdere aan fasen zijn
toegewezen die inmiddels voorbij zijn:

| DI | Toegewezen aan | Feitelijke status |
|---|---|---|
| DI-1 | Phase 6 | **open** — Viterbi in `hmm_regime.py` (Stage C-1) |
| DI-3 (D-6) | Phase 5 | **open** — nog 7 bestanden >800 LOC |
| DI-4 (D-7) | Phase 7 | **open** — 22 van 35 apps >80 LOC |
| DI-7 | Phase 7 | **open** — `asyncio.create_task` (Stage D-2) |
| DI-8 | Phase 5 | **open** — legacy research-scripts in `scripts/` |
| DI-11 / DI-14 | Phase 6 / 7 | deels open — hardcoded params |
| DI-15 | doorlopend | **open** — survivorship bias, vereist een tweede databron |

Elke DI krijgt een definitief oordeel: **gesloten met bewijs**, **bewust geaccepteerd met
motivering en eigenaar**, of **doorgeschoven met een concrete voorwaarde**. Een backlog die
naar afgeronde fasen verwijst, is geen backlog maar ruis.

Voor **DI-3 en DI-4** geldt: dit zijn P2-punten die sinds Phase 0 doorschuiven. Kies
expliciet — opruimen of formeel accepteren met een ratchet die verdere groei blokkeert.
Blijven doorschuiven is de derde optie die dit project zichzelf niet toestaat.

### E-3 — Documentatiedrift dichten

Elk document in `docs/` krijgt de regel *"Geverifieerd tegen de codebase op [datum]"* of
wordt gearchiveerd. Bijzondere aandacht voor:

- `docs/architecture.md` — beschrijft de vier backtesters die Phase 5 heeft verwijderd?
- `docs/tca_methodology.md` — DI-5 claimde bestanden die inmiddels wél bestaan;
- `docs/model_risk_policy.md` — na Stage B eindelijk waar;
- `docs/runbook.md` versus `docs/RUNBOOK.md` — op Windows hetzelfde bestand, op Linux niet.
  **Dit is een reële CI-valkuil.** Kies één naam.

### E-4 — Het overdrachtsdocument

`docs/PROJECT_STATE.md`: wat het platform is, wat aantoonbaar waar is, wat aantoonbaar niet
waar is, welke besluiten openstaan en wat de volgende drie stappen zijn. Eén document,
geschreven voor iemand die het project overneemt.

### E-5 — Eindrapport

`reports/phase7_8_exit_report.md`, met per stage het bewijs, en — verplicht — de sectie
**"wat er tijdens deze fase mis bleek in mijn eigen werk"**. Phase 5 vond vijf van zijn
negen defecten in de bewijsvoering zelf. Verwacht hetzelfde hier.

---

## 8. EXIT CRITERIA

### Stage A — Fundament

| # | Criterium |
|---|---|
| A1 | Git-historie aantoonbaar buiten deze machine, met geverifieerde bundle en/of remote |
| A2 | De C-schijf-repo is opgeruimd, ná bewijs dat hij nul unieke commits bevatte |
| A3 | `x.jsonl` weg uit git; de schrijvende test schrijft naar `tmp_path`; bewakende test aanwezig |
| A4 | `pip install -e .` werkt; `tradebot.__file__` wijst naar `D:\Tradebot\src` |
| A5 | Nul `.pyc` met een `co_filename` buiten de repository-root; test bewijst dit en is rood geweest |
| A6 | `check_hardcoded_params.py --strict` exit 0 **zonder enige budgetverruiming** |
| A7 | `audit_fallbacks.py --strict` exit 0; faalbasislijn nog steeds 4, dezelfde vier namen |
| A8 | Elke ratchet aantoonbaar rood gemaakt op een wegwerpbranch |
| A9 | Schone-venv-build reproduceert de suite; referentie-interpreter in het runbook |

### Stage B — Governance (D-1)

| # | Criterium |
|---|---|
| B1 | Alle 6 lookahead-tests bestaan als benoemd bestand en zijn **aantoonbaar rood geweest** |
| B2 | `validation/{gates,walk_forward,dsr,spa}.py` bestaan; DSR weigert te draaien zonder eerlijke `M` |
| B3 | `registry/trial_counter.py` persistent; reconstructiemethode en onzekerheid gedocumenteerd |
| B4 | Promotie-state-machine met eenrichtingsovergangen behalve `→ FALSIFIED` |
| B5 | `check_banned_methods.py` nul treffers; draait blokkerend in CI |
| B6 | `research_gates.yml` blokkeert aantoonbaar een merge met een lekkend model |
| B7 | `test_gate_cannot_be_bypassed.py` weigert het `shift(-1)`-model |
| B8 | Ledger-integriteit: entry zonder `git_sha`, `data_hash`, `config_hash` of `preregistration_id` crasht |
| B9 | `model_risk_policy.md` gecorrigeerd; **D-1 gesloten**; `phase2_exit_report.md` gepubliceerd |

### Stage C — Phase 6

De 20 exit-criteria van `fase_6_advanced_research.md` §6 gelden onverkort, met deze
aanvullingen:

| # | Criterium |
|---|---|
| C1 | `portfolio/hrp.py` is research-gated met een **crash**, niet met een vlag |
| C2 | `risk/hmm_regime.py` levert filtered probabilities, of bestaat niet meer; **DI-1 gesloten** |
| C3 | HAR-RV draagt `UNPROVEN — insufficient data` met de gemeten 0,00 % dekking |
| C4 | H1, H2 en H3 hebben elk een ledger-oordeel met beide kostenlabels |
| C5 | Elk gepromoveerd model is door de **Stage B**-gates gegaan |
| C6 | Sovereign-audit uitgebreid naar `volatility/`, `regime/`, `labeling/`, `train/`, `portfolio/hrp.py` |
| C7 | DVC-stage voor de authoritative engine bestaat; elke run reproduceerbaar |

### Stage D — Productie

De 9 exit-criteria van `fase_7_production_papertrading.md` gelden onverkort, met:

| # | Criterium |
|---|---|
| D1 | `live/` gebruikt de Phase 5-componenten; nul tweede implementaties; statisch bewezen |
| D2 | C1 t/m C6 gesloten; `min_confidence` bestaat niet meer in `live/`; **geen enkele limiet defaultet naar oneindig** |
| D3 | DI-7 gesloten: elke `asyncio.create_task` houdt zijn referentie vast |
| D4 | `AlertSeverity` kent `HALT`; een `HALT` schakelt uit zonder te vragen |
| D5 | Alle monitoringdrempels uit `conf/monitoring/`, gehasht **vóór** de start van de klok |
| D6 | De vier `test_chaos.py`-skips zijn geïmplementeerd of verwijderd |
| D7 | 60 opeenvolgende schone dagen; dagelijkse bit-identieke pariteit |
| D8 | Crash recovery bewezen op ≥3 punten; nul dubbele orders, nul verloren posities |
| D9 | Kill switches vuren live; `HALTED` overleeft een procesherstart |
| D10 | Champion/challenger technisch afgedwongen; negatieve controles op 59 dagen én p = 0,06 |
| D11 | MRM-rapport crasht bij een ontbrekende hash — **D-9 definitief gesloten** |
| D12 | Runbook getoetst door een tweede persoon |

### Stage E — Oplevering

| # | Criterium |
|---|---|
| E1 | Het reproductiepad is op een verse machine doorlopen |
| E2 | Elke DI heeft een definitief oordeel: gesloten, geaccepteerd, of doorgeschoven met voorwaarde |
| E3 | Nul documenten die niet-bestaande artefacten claimen; `runbook`-casusprobleem opgelost |
| E4 | `docs/PROJECT_STATE.md` bestaat |
| E5 | `reports/phase7_8_exit_report.md` compleet, inclusief het eigen-defecten-hoofdstuk |

---

## 9. ABSOLUTE NO-GO CONDITIES

Is één hiervan aan het eind actief, dan is de fase **niet** afgerond, ongeacht hoeveel
deliverables er staan.

| # | Conditie |
|---|---|
| 1 | Een ratchet staat rood, of is groen gemaakt door zijn budget te verruimen |
| 2 | De git-historie bestaat nog steeds maar op één fysieke locatie |
| 3 | De C-schijf-repo is verwijderd zónder bewijs dat hij nul unieke commits bevatte |
| 4 | Een `.pyc` in de boom verwijst naar een pad buiten de repository |
| 5 | Een promotieclaim is gedaan zonder de Stage B-gates |
| 6 | `live/` bevat een tweede implementatie van executie-, accounting- of risicologica |
| 7 | Een limiet in `live/` defaultet naar oneindig, of ontbreekt zonder te crashen |
| 8 | Modelvertrouwen (`min_confidence`) beïnvloedt nog positiegrootte |
| 9 | De HALT-toestand overleeft een procesherstart niet |
| 10 | Een monitoringdrempel is ná de start van de 60-daagse klok vastgesteld |
| 11 | De 60-daagse teller is niet herstart na een crash of onverklaarde drift |
| 12 | Een statistische conclusie is getrokken zonder negatieve controle |
| 13 | Een `FALSIFIED`-oordeel is toegekend aan een model dat de Data Adequacy Gate niet haalde |
| 14 | Een promotieclaim staat in een rapport zonder `IMPACT_UNCALIBRATED` en `SPREAD_ASSUMED` |
| 15 | Een test schrijft naar een echt governance-artefact of naar de repository-root |
| 16 | Een smoothed probability is bereikbaar vanuit een backtest- of live-pad |
| 17 | De EMA-crossover-fallback uit §5.2 is in enige vorm teruggekeerd |
| 18 | Een permanent overgeslagen test houdt de suite ten onrechte groen |
| 19 | `M` is niet bijgewerkt voor alle uitgevoerde trials |
| 20 | Een document claimt een artefact of gate die niet bestaat |

---

## 10. REGELS & HANDELINGSINSTRUCTIES

- **Meet voordat je iets beweert.** Elke claim in dit document is gedateerd 2026-08-27.
  Herhaal elke meting bij aanvang en noteer elke afwijking. Waar meting en rapport
  verschillen, prevaleert de meting (§2.2).
- **Fail-fast compliance, live.** Ontbrekende feed, inconsistente toestand, niet-sluitende
  accounting, ontbrekende configuratie: halteren. Nul `try/except` fallbacks. In productie
  is stilzwijgend doorgaan met gedegradeerde input de duurste denkbare keuze.
- **100 % PIT rigor, ook live.** De live feature-berekening gebruikt exact dezelfde causale
  code als de backtest. Elke afwijking tussen de live en de gesimuleerde waarde op hetzelfde
  tijdstip is een defect, geen ruis. `latency_bars = 1` bovenop de beslisbar.
- **Eén codepad.** Backtest en live delen dezelfde executie-, accounting- en
  risicocomponenten. Een aparte "live-versie" van een functie is verboden.
- **Geen promotie zonder bewijs.** 60 dagen is een minimum, geen richtlijn. DM p < 0,05 is
  een drempel, geen streefwaarde. Geen enkele uitzondering, ook niet bij een indrukwekkende
  equity curve.
- **Geen hardcoded variabelen.** Elke drempel uit `conf/`, gevalideerd tegen een
  Pydantic-schema, gehasht, geregistreerd.
- **Elke gate krijgt een negatieve controle.** Een test die niet rood kan worden, bewijst
  niets. Phase 5 vond vijf van zijn negen defecten in de bewijsvoering zelf.
- **Onbewezen ≠ bewezen slecht (§6).** Faalt een model door ontoereikende data, dan is het
  `UNPROVEN — insufficient data` met de gemeten cijfers erbij.
- **Atomaire commits, conventional.** `chore(repo): restore editable install after the drive
  move`, `fix(validation): move phase-6 thresholds out of code and into conf`,
  `feat(validation): add the promotion gate suite that D-1 has always claimed`,
  `refactor(live): reuse the phase-5 execution path in the live controller`,
  `feat(monitoring): add per-order execution drift measurement`,
  `feat(registry): enforce the 60-day champion challenger promotion gate`.
- **Escaleer, verzwijg niet.** Elk incident in de 60-daagse periode wordt gelogd,
  onderzocht en gepubliceerd — ook wanneer het de teller terugzet.
- **De opdrachtgever beslist over besluiten, jij over uitvoering.** §11 bevat vier vragen
  die geld of scope raken. Leg ze voor; kies ze niet zelf.

---

## 11. WAT DIT PROJECT NOG NODIG HEEFT — analyse buiten de deliverables

Dit is geen deliverablelijst maar de eerlijke stand van zaken over wat dit platform
ontbreekt om meer te zijn dan een zeer goed gevalideerd meetinstrument.

### 11.1 Het echte knelpunt is data, niet code — en dat is een inkoopbesluit

Open Question 1 uit §27 is nu **drie keer** met cijfers beantwoord en het antwoord is drie
keer *nee*:

| Fase | Meting |
|---|---|
| Phase 1 | 0 rijen 1m/5m, 0 L2-snapshots, 0 trade-prints |
| Phase 5 | `IMPACT_UNCALIBRATED`, `SPREAD_ASSUMED`; Corwin-Schultz verworpen (32,6–65,9 bp, 33–38 % negatief) |
| Phase 6 | Data Adequacy: **0,00 %** 5m-dekking tegen een eis van 80 % |

De gevolgen stapelen: HAR-RV is niet te fitten, `eta` is niet identificeerbaar, de spread is
een aanname van 1,0 bp waarvan Phase 6 zelf vaststelt dat de gemeten verbetering al bij 3 bp
kan verdwijnen, en orderboekdiepte, limit-queues en cancellations ontbreken volledig.

**Geen enkele hoeveelheid engineering lost dit op.** Er zijn twee reële opties, en beide
zijn een besluit van de opdrachtgever:

1. **Een live orderboek-sampler starten** — kost niets behalve tijd, maar de dekking
   ontstaat pas vanaf het moment dat hij draait. Start hij aan het begin van deze fase, dan
   is er bij afronding van de 60-daagse periode ~2–3 maanden L1/L2-historie. **Dat is de
   goedkoopste zet in dit hele document en hij is nu al te laat gestart.**
2. **Een externe tick-databron aankopen** — geeft direct historie, kost geld.

> **Aanbeveling: doe optie 1 in week 1 van deze fase, ongeacht wat er over optie 2 wordt
> besloten.** De sampler is een klein, geïsoleerd stuk werk; de kosten van hem níet starten
> lopen elke dag op.

### 11.2 Het universum is te klein, en dat begrenst wat er nog te vinden is

Zes symbolen, 1.743 bars, één cluster, gemiddelde paarsgewijze correlatie 0,735, zes
ex-post gekozen overlevers (DI-15). Consequenties die door de hele stack lopen:

- de `cluster_cap` is **vacuous** — er valt niets tussen clusters te spreiden;
- HRP is een clustering-allocator op een universum zonder clusterstructuur;
- elke DSR op 1.743 bars met `M = 2776` is streng, en dat hoort ook;
- survivorship bias raakt elke absolute claim.

Verbreding van het universum — meer perps, of een tweede asset class — doet **drie** dingen
tegelijk: het maakt `cluster_cap` betekenisvol, het geeft HRP een eerlijke kans, en het
verkleint de survivorship-bias. Van alle onderzoeksrichtingen die openstaan, is dit degene
met de hoogste verwachte opbrengst per eenheid werk.

**Let op:** zodra het universum verbreedt, moet de clusterlabeling **opnieuw gemeten**
worden, niet overgenomen. Dat is de expliciete les uit
`phase5_cluster_concentration_audit.md`.

### 11.3 De openstaande vraag van Phase 3 is nooit beantwoord

> *"De risicolaag maakt geen alpha. Alle vier de tracks blijven netto negatief. Dat is de
> openstaande vraag van Phase 3 en geen enkele limiet lost hem op."*

Vier fasen lang is deze zin doorgegeven. De risicolaag heeft de drawdown met een factor 12
tot 17 verlaagd en de variantie-drag van 25,4 pp naar 0,1 pp gebracht — dat is echt en
waardevol. Maar het onderliggende signaal heeft **bruto** geen edge (bruto Sharpe −0,284).

Dit platform is inmiddels uitzonderlijk goed in het **weerleggen** van alpha en heeft nog
geen enkele bevestigde alpha-unit. Dat is geen tekortkoming — het is precies wat een
falsificatie-platform hoort te doen, en het is oneindig veel beter dan het omgekeerde. Maar
het betekent wel dat de vraag *"wat gaan we hier eigenlijk mee verdienen?"* na Stage E
scherper voorligt dan nu.

**De eerlijke volgorde is: eerst het instrument afmaken (deze fase), dan pas nieuwe alpha
zoeken.** Alpha zoeken op een half-gevalideerd platform is precies hoe de 20 gefalsificeerde
units zijn ontstaan. Maar reserveer die vraag expliciet als de eerste van de volgende fase,
in plaats van hem opnieuw door te schuiven.

### 11.4 Operationele volwassenheid buiten de auditscope

Het auditdocument dekt research-integriteit uitputtend, maar zwijgt over een aantal
productie-eigenschappen die vóór echt kapitaal moeten bestaan:

| Onderwerp | Stand |
|---|---|
| **Secrets-beheer** | `miscellaneous/.env` bestaat; API-sleutels voor live trading vereisen een expliciet beleid |
| **Exchange-account-scheiding** | paper- en live-sleutels mogen niet in dezelfde configuratie kunnen staan |
| **Back-up en herstel van de PIT-store** | `data/pit_store.dvc` bestaat; waar staat de DVC-remote? |
| **Tijdsynchronisatie** | een live systeem op UTC-ns met een driftende systeemklok produceert stille causaliteitsfouten |
| **Afhankelijkheid van één venue** | alles hangt aan Bybit; een venue-uitval is een totale uitval |
| **Kostenbewaking** | fees domineren; een live fee-tier-wijziging verandert elk resultaat |

Geen van deze hoort in Stage A t/m D thuis, maar alle zes horen in `docs/PROJECT_STATE.md`
te staan als bekende, benoemde risico's — in plaats van als aannames.

### 11.5 De structurele schuld die blijft doorschuiven

D-6 (7 bestanden >800 LOC) en D-7 (22 van 35 apps >80 LOC) schuiven sinds Phase 0 door en
staan nu op fasen die al voorbij zijn. Zij raken geen enkel statistisch contract, en dat is
precies waarom ze blijven liggen.

**Kies één keer expliciet:** opruimen, of formeel accepteren met een ratchet die verdere
groei blokkeert — zoals de hardcoded-parameter-ratchet dat voor literals doet. Een derde
keer doorschuiven is het patroon dat dit project bij elk ander onderwerp weigert.

---

## 12. STARTINSTRUCTIE

> **Begin niet met Stage A-1 maar met de meting.**
>
> Herhaal elke meting uit §0 op de huidige HEAD en publiceer
> `reports/phase7_state_of_the_repository.md` met per meting: het commando, de uitvoer, en
> of hij overeenkomt met §0. Noteer elke afwijking expliciet — deze prompt is gedateerd
> 2026-08-27 en elke wijziging sindsdien maakt een aanname ongeldig.
>
> Neem in dat rapport in elk geval op:
> 1. `git rev-list --count HEAD`, `git log --oneline -1`, `git remote -v`, `git status --short`
> 2. de staat van `C:\Users\algul\Documents\Tradebot\.git` — commit-telling, HEAD, en het
>    aantal paden dat zijn werkkopie mist
> 3. de inhoud van `__editable__.tradebot-0.4.0.pth` en de uitkomst van
>    `python -c "import tradebot; print(tradebot.__path__)"`
> 4. het aantal `.pyc` met een `co_filename` buiten de repository-root
> 5. de exit codes van `check_hardcoded_params.py --strict`, `audit_fallbacks.py --strict`
>    en `pytest -q`, met de namen van elke falende test
> 6. de aanwezigheid van elk bestand uit de tabellen in §0.4, §0.5 en §0.6
>
> Commit als: `docs(phase7): measure the actual state of the repository before touching it`
>
> **Ga daarna naar Stage A-1**, en maak als allereerste handeling de git-bundle — vóór enige
> wijziging, inclusief het opruimen van de caches. De historie van 77 commits is het enige
> artefact in dit project dat niet opnieuw te produceren is.
>
> Commit als: `chore(repo): secure the commit history off this machine before any change`
