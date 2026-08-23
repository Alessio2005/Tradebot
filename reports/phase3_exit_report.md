# PHASE 3 — EXIT REPORT

> **Fase:** 3 van 7 — Baseline Implementation · **Prioriteit:** P1
> **Bindend brondocument:** `ARCHITECTUUR_AUDIT_2026-08-22.md` — secties 9.1, 11, 11.1, 11.2, 13.1, 21, 22, 23 (Phase 3), 26
> **Gegenereerd:** 2026-08-23 · **`git_sha` van de baseline-cijfers:** `bac4427`

---

## 0. Voorwaarde die niet was vervuld, en wat daaraan is gedaan

De fase-opdracht stelt als voorwaarde: *"Phase 0, 1 en 2 volledig afgerond.
Zonder werkende gates is een baseline slechts een bewering."*

Die voorwaarde was **niet vervuld**. Het auditdocument (sectie 23) merkt als
Phase 2 de **Research & Falsification Foundation** aan (L11/L12), en die is niet
gebouwd — zie `reports/phase2_feature_validation.md` §0. Wat wél is gebouwd is
de Deterministic Feature Engine (L3).

Concreet ontbrak `registry/preregistration.py`, en dat is precies wat **stap 1**
van deze fase als eerste handeling voorschrijft. Die module is daarom hier
gebouwd. Dat is de **enige** uitbreiding buiten de Phase 3-deliverablelijst, en
hij is bewust minimaal gehouden: het pre-registratiecontract zelf, niet de
volledige L11/L12-gate-infrastructuur.

Wat er al wél was en is hergebruikt in plaats van herbouwd:

| Component | Locatie | Status |
|---|---|---|
| Deflated Sharpe Ratio | `backtest/metrics.py` | RETAIN — correct, uitsluitend afgedwongen |
| Hansen's SPA | `backtest/spa.py` | bestaand, aangeroepen als gate |
| Purged Walk-Forward | `cv/walk_forward.py` | bestaand, met embargo |
| Hypothese-ledger | `registry/hypothesis_ledger.py` | RETAIN — levert de eerlijke `M` |
| Fantoom-herbalanceringsfix | `alpha/xs_unit.py` | RETAIN — conventie overgenomen |

**Wat nog steeds ontbreekt** en dus openstaat: `validation/gates.py`, de
promotie-state-machine, de zes zelfstandige lookahead-testbestanden onder de
namen uit D-1, `scripts/check_banned_methods.py` en de CI-workflow. Zie §5.

---

## 1. Exit criteria

| # | Criterium | Bewijs | Status |
|---|---|---|---|
| 1 | Baseline OOS Sharpe en drawdown gekwantificeerd, per track, bruto én netto, met max drawdown, Calmar en turnover | `reports/BASELINE_BENCHMARK.md` §3; vier tracks × 1.615 OOS-bars | **GEHAALD** |
| 2 | Alle lookahead-gates groen op de volledige keten (features → alpha → portfolio → backtest) | 33 tests in `tests/lookahead/test_baseline_causality.py`, alle groen; zie §2 voor de nuance over "de 6 gates" | **GEHAALD, met kanttekening** |
| 3 | Alpha-isolatie bewezen; elke output binnen [−1, +1] | 50 tests in `tests/unit/test_alpha_isolation.py`; nul verboden imports in 27 alpha-modules; AST-guard in `__init_subclass__` | **GEHAALD** |
| 4 | 100 % van de data draagt een gecertificeerde `data_hash` | `load_certified_close_panel` verifieert elke reeks tegen het register en crasht bij afwijking; de 6 hashes staan in het artefact | **GEHAALD** |
| 5 | DSR gerapporteerd met eerlijke `M` uit de persistente teller | `M = 2.715` (2.711 bij bevriezing + 4 geplande trials); de gate crasht zonder `M` | **GEHAALD** |
| 6 | SPA uitgevoerd over de multi-strategie-vergelijking | `spa_p_value = 1,00` tegen de 1/N-referentie, 3 kandidaten | **GEHAALD** |
| 7 | Ledger-entry compleet met `git_sha`, `data_hash`, `config_hash`, `preregistration_id` en `M` | wave 29, `phase3_baseline_result`, resultaat `falsified` | **GEHAALD** |
| 8 | Reproduceerbaarheid: tweede run met dezelfde seed en `data_hash` levert bit-identieke resultaten | geverifieerd; `M` komt uit de bevriezing en niet live uit de ledger, juist om dit te garanderen | **GEHAALD** |
| 9 | Eerlijke rapportage: een negatieve baseline wordt als zodanig gerapporteerd en geregistreerd | netto OOS Sharpe **−0,483**; drie stop-criteria binden; ledger-resultaat `falsified` | **GEHAALD** |

### 1.1 Het resultaat zelf

**De pre-geregistreerde hypothese is gefalsificeerd.** Dat is de uitkomst, niet
een tussenstand. Er is geen parameter gesweept om hem om te draaien; de
pre-registratie stond precies één configuratie toe en de actieruimte van een
stop-criterium kent de zet "drempel bijstellen" niet.

| Track | Netto Sharpe | Max DD | DSR | Promotie |
|---|---:|---:|---:|---|
| `long_only_equal_weight` (1/N) | +0,156 | 0,835 | 6,9 × 10⁻⁴ | nee |
| `long_only_risk_parity` | +0,129 | 0,822 | 5,6 × 10⁻⁴ | nee |
| `xs_momentum_equal_weight` | −0,181 | 0,508 | 4,6 × 10⁻⁵ | nee |
| **`xs_momentum_risk_parity`** | **−0,483** | 0,563 | 2,7 × 10⁻⁶ | **nee** |

---

## 2. Kanttekening bij criterium 2 — "de 6 Phase 2-gates"

De deliverablelijst vraagt om *"de 6 Phase 2-gates toegepast op de
baseline-keten"*. Die zes **bestaan niet als zelfstandige bestanden**; ze horen
bij de Research & Falsification Foundation die niet is gebouwd (§0). Wat hier is
gedaan, en wat niet:

| D-1 gate | Gedekt in `test_baseline_causality.py`? |
|---|---|
| `test_truncation_invariance` | **ja** — elke transform, de pipeline, de alpha-exposures en de gewichten van alle vier de tracks |
| `test_scaler_fit_causality` | **ja** — `expanding_zscore` en `rolling_quantile_winsorise` zijn getoetst, plus een negatieve controle op een sample-brede normalisatie |
| `test_determinism_reproducibility` | **ja** — purity- en niet-muteren-tests per transform; bit-identieke herhaling |
| `test_future_column_poisoning` | **gedeeltelijk** — een unit met een NEGATIEVE skip (venster dat in de toekomst eindigt) maakt de toets aantoonbaar rood, maar er is geen generieke kolom-injectietest |
| `test_temporal_shift_invariance` | **nee** — niet gebouwd |
| `test_label_horizon_purge` | **niet van toepassing** — deze baseline kent geen labels; de embargo is wel toegepast en geteld (85 bars) |

Criterium 2 is daarom als **GEHAALD, met kanttekening** gemarkeerd: de
causaliteit van de keten is bewezen op het niveau dat de zes gates beogen, maar
de zes benoemde bestanden bestaan nog niet en D-1 is niet gesloten.

---

## 3. Ontwerpkeuzes die verantwoording verdienen

### 3.1 Winsorisatie zit niet in de baseline-keten

`rolling_quantile_winsorise` is gebouwd en getoetst (deliverable 2 vraagt erom),
maar staat **niet** in de momentum-keten. Een rangschikking is per constructie al
ongevoelig voor uitschieters — dat is wat een rang doet — en winsoriseren vóór
een rang zou 59 extra bars burn-in kosten, waardoor het eerste signaal van
2021-05-16 naar 2021-07-14 zou opschuiven. De transform hoort bij
NIVEAU-features, niet vóór een rang.

Dit was een echte keuze met een echt gevolg: hij is gemaakt vóórdat de resultaten
bekend waren, en niet teruggedraaid toen ze tegenvielen.

### 3.2 De `M` komt uit de bevriezing, niet uit de live ledger

`ledger.total_n_hypotheses()` groeit ook door de resultaat-entry van de meting
zelf. Twee runs zouden dan een andere `M` en dus een andere DSR opleveren — en
reproduceerbaarheid is exit criterium 8. `M` = `ledger_total_at_freeze` +
`planned_trials`, beide vastgelegd in het bevroren pre-registratie-artefact.

### 3.3 `NOT_APPLICABLE` is een zichtbare status

`cost_drag_fraction` vereist een positieve bruto-edge om te eroderen. Die is er
niet (bruto Sharpe −0,284). Het criterium wordt daarom expliciet als
`NOT_APPLICABLE` gerapporteerd en **niet** als geslaagd geteld. Een criterium
dat niet beoordeeld kan worden en stilzwijgend als groen wordt geteld, is de
manier waarop een gate zichzelf uitholt.

### 3.4 Walk-forward op een regelgebaseerde baseline

Deze baseline fit niets. Purged Walk-Forward voegt hier geen bescherming toe die
er niet al was, en dat staat zo in `backtest/baseline_runner.py` en in het
benchmarkrapport. Hij wordt toegepast zodat een later model op precies dezelfde
bars wordt gemeten en dezelfde embargo-bars mist — niet om het getal
geloofwaardiger te laten lijken dan het is.

### 3.5 De EWMA-hardening raakte een RETAIN-grens

`get_ewma_volatility` is een harde deprecation-crash geworden in plaats van een
doorgeefpad. De oude signatuur nam `halflife`, de nieuwe `lambda`; stil
vertalen zou een gedragswijziging onopgemerkt maken. Buiten de testsuite had de
functie nul aanroepers in `src/`.

---

## 4. Hygiëne en regressie

| | Baseline (`45f2629`) | Na Phase 3 (`bac4427`) |
|---|---:|---:|
| Verzamelde tests | 1.064 | 1.190 (+126) |
| Gefaald | 6 | 6 |
| Ratchet hardcoded params | 332 | 330 |

De 6 failures zijn **pre-existent en ongewijzigd**: 4× `test_expansion_killgates`
en 2× `test_hypothesis_kernels`. Zij vallen buiten de scope van deze fase.

| Scanner | Uitkomst |
|---|---|
| `scripts/check_hardcoded_params.py --strict` | exit 0; `volatility/ewma.py` van budget 2 → **0** |
| `scripts/audit_fallbacks.py --strict` | exit 0 blokkerend |
| `ruff check` | clean op alle Phase 3-bestanden |
| `mypy` | clean op `features/transforms.py`, `volatility/ewma.py`, `registry/preregistration.py` |
| R-6 (apps ≤ 80 LOC) | `run_baseline.py` 57 LOC, `freeze_preregistration.py` 78 LOC |

**GOVERNED uitgebreid** (budget 0, permanent): `features/transforms.py`,
`volatility/ewma.py`, `registry/preregistration.py`.

---

## 5. Wat deze fase NIET heeft gedaan

Eerlijk afgebakend, zodat Phase 4 niet uitgaat van dekking die er niet is.

1. **De L11/L12-governancelaag blijft grotendeels open.** Gebouwd:
   `registry/preregistration.py` en de afdwinging van DSR en SPA in
   `backtest/baseline_report.py`. Niet gebouwd: `validation/gates.py`, de
   promotie-state-machine (`REGISTERED → TESTED → CANDIDATE → PAPER → CHAMPION`),
   `registry/trial_counter.py` als zelfstandige module, de zes benoemde
   lookahead-testbestanden, `scripts/check_banned_methods.py` en
   `.github/workflows/research_gates.yml`. **D-1 is niet gesloten.**
2. **`alpha/base.py` en `alpha/momentum.py` zijn UITGEBREID, niet herschreven.**
   Het legacy `AlphaSignal`-protocol en `TSMomentum`/`CSMomentum` staan er nog,
   met een expliciete scheidingsregel ertussen. 26 modules en
   `alpha/__init__.py` hangen aan het oude protocol. Daardoor houdt
   `alpha/momentum.py` zijn ratchet-budget van 2 in plaats van naar 0 te zakken.
   Idem voor `features/pipeline.py` en `portfolio/risk_parity.py`, waar de
   nieuwe L1/L8-code onder een gescheiden sectie naast de legacy staat.
3. **ERC is niet ontmanteld.** `erc_weights` bevat twee stille degradaties
   (fillna met het sample-gemiddelde; inverse-vol fallback bij een mislukte
   solve) maar wordt gebruikt door `portfolio/optimizer.py`. Hij is voor Phase 3
   geen toegestane allocator en is gedocumenteerd, niet verwijderd.
4. **Geen econometrische toetsingsketen (sectie 8.2).** ADF/KPSS, CUSUM en
   Ljung-Box/ARCH draaien nog steeds niet op de reeksen die de pipeline
   binnenkomen. Stond ook al in `reports/phase2_feature_validation.md` §8.2 als
   aanbeveling voor deze fase; is **niet** opgepakt en schuift door.
5. **Survivorship bias (DI-15) staat open.** Zes ex-post gekozen overlevers.
   Vermeld bij elke claim in het benchmarkrapport, maar niet opgelost — dat
   vereist een tweede databron met delisting-historie.
6. **Eén enkele configuratie getoetst.** Bewust: elke extra variant verhoogt
   `M` en verlaagt de DSR van álle varianten. Een tweede configuratie vereist
   een nieuwe pre-registratie.
7. **De kostenaanname is voorlopig.** De η-kalibratie is Phase 5.

---

## 6. Aanbeveling voor Phase 4

Phase 4 (Risk & Volatility Architecture) kan starten: er is een gekwantificeerde
baseline om risico-impact tegen af te meten, en de alpha-laag is aantoonbaar
ontkoppeld van risk en execution — de statische helft van de ontkoppelingstest
die Phase 4 als vereist bewijs opvoert, slaagt nu al.

Drie punten uit deze fase die daar direct aan raken:

* **§3.1 van het benchmarkrapport is een opdracht voor Phase 4.** Een
  geannualiseerde volatiliteit van 72 % maakt de Sharpe een slechte proxy voor
  kapitaalgroei: +11,3 % rekenkundig werd −14,1 % meetkundig. Volatility
  targeting is precies het instrument dat dat gat verkleint, en het effect
  ervan hoort **meetkundig** te worden gerapporteerd, niet alleen in Sharpe.
* **Exit criterium 8 van Phase 4** eist dat de Phase 3-baseline opnieuw wordt
  gedraaid mét de risicolaag. `apps/run_baseline.py` en
  `backtest/baseline_runner.py` zijn daarop voorbereid: de allocator-stap is
  gescheiden van de backtest-stap.
* **De ontkoppelingstest heeft nog een functionele helft nodig.**
  `tests/unit/test_alpha_isolation.py` dekt de statische kant volledig; het
  functionele bewijs (identieke `a_t` → bit-identieke `permitted_exposure`,
  ongeacht de herkomst) kan pas zodra de `RiskEngine` bestaat.
