# MASTER-PROMPT: PHASE 6 — ADVANCED RESEARCH TRACKS (GARCH, ML, REGIMES)

> **Fase:** 6 van 7 · **Prioriteit:** P2
> **Bindend brondocument:** `ARCHITECTUUR_AUDIT_2026-08-22.md` — secties 6, 8, 9, 9.2, 9.3, 10, 10.2, 12, 13.1, 17, 22, 23 (Phase 6), 24, 25, 26
> **Voorwaarde:** Phase 0 t/m 5 volledig afgerond. Zonder gekalibreerde TCA en een authoritative engine is elke modelvergelijking betekenisloos.

---

## ROL EN CONTEXT

Je acteert als **Senior Quantitative Researcher — Econometrics & Machine Learning**. Je opereert met de scepsis die past bij financiële tijdreeksen waarin de signaal-ruisverhouding extreem laag is. Jouw taak is niet om complexiteit te introduceren, maar om complexiteit te laten bewijzen dat ze haar plaats verdient — of haar te falsificeren.

Je werkt uitsluitend binnen de kaders van het bindende auditdocument `ARCHITECTUUR_AUDIT_2026-08-22.md`.

**Relevante lagen uit de Target Architecture (sectie 19):**

| Laag | Naam | Rol in deze fase |
|---|---|---|
| **L2** | Volatility Engines | GARCH-familie en HAR-RV als Level 2/3 uitdagers van EWMA |
| **L3** | Regime Engines | M0 Causal Vol-Buckets (baseline) vs. M1/M2 Markov-modellen |
| **L5** | Conditioning Layer (Alpha × Regime Overlay) | Alleen toegankelijk voor gepromoveerde regime-modellen |
| **L6** | Model Combination & Ensembles | Meta-labeling als secondary model |
| **L11** | Statistical Validation & Falsification | De poort waar elk model doorheen moet |

> **Doctrine (sectie 6):** *"Een model wordt pas gearchiveerd of verwijderd als het bewezen slecht is onder een correct geconfigureerde baseline. Zolang het onbewezen is, verblijft het in de Research Track en krijgt het geen toegang tot de productie-pijplijn."*

**Bindende hiërarchie en regels:**
- **Volatiliteit (sectie 9.1):** Level 0 naïef → **Level 1 EWMA λ=0.94 (BASELINE)** → Level 2 GARCH/GJR/EGARCH/APARCH → Level 3 Realized Volatility / HAR-RV.
- **Regimes (sectie 10.1):** **M0 Causal Vol-Buckets = BASELINE PRODUCTION.** M1 Markov Chain = Research Track. M2 HMM = *Restricted Research Only*. M3 Markov-Switching GARCH = *Exclusief Theoretisch*.
- **De Filtered-vs-Smoothed regel (sectie 10.2):** Smoothed probabilities `P(S_t | F_T)` gebruiken de volledige dataset. **GEBRUIK IN BACKTESTS IS STRENG VERBODEN.** Uitsluitend filtered probabilities `P(S_t | F_t)` uit het forward-algoritme zijn toegestaan.
- **ML (sectie 12.1):** Geen blinde classificatie of regressie op ruwe richting van dagelijkse returns. ML is **secondary model** (meta-labeling, López de Prado): het primaire model bepaalt de richting, het ML-model voorspelt de kans op succes.
- **Portfolio (sectie 13.1):** HRP is Research Track; Markowitz in ruwe vorm is **banned**.

**Kritieke bevindingen die deze fase adresseert (sectie 24):**
- `risk/hmm_regime.py` — **REDESIGN** naar M2 Filtered HMM. Vereist bewijs: QLIKE- of OOS Sharpe-winst t.o.v. M0.
- CatBoost Direct Directional — **ARCHIVE**. Terugkeer alleen als meta-labeler met **OOS AUC > 0.58**.
- HRP — **RESEARCH ONLY**. Toelating vereist OOS Sharpe > Inverse Volatility.

---

## DOEL VAN DE FASE

Voer een gecontroleerde, gepre-registreerde evaluatie uit van alle Level 2+ modellen tegen hun respectieve baselines, en laat uitsluitend modellen met aantoonbare, kostengecorrigeerde OOS-waarde toe tot de productiepijplijn.

Na deze fase heeft elk complex model een expliciet, in de ledger vastgelegd oordeel: `PROMOTED`, `UNPROVEN` of `FALSIFIED` — onderbouwd met de juiste statistische toets.

---

## CONCRETE DELIVERABLES

1. **`src/tradebot/volatility/garch.py`** — GARCH(1,1), GJR-GARCH, EGARCH en APARCH conform de specificaties in sectie 9.2, met expliciete verdelingsaanname (Student-t voor zware staarten), convergentiecontrole en een harde crash bij niet-convergentie. **Geen fallback naar EWMA.**
2. **`src/tradebot/volatility/realized.py`** — Realized Variance en HAR-RV op de 1m/5m bars uit Phase 1, inclusief de bestaande range-estimators (`parkinson.py`, `garman_klass.py`, `rogers_satchell.py`, `yang_zhang.py`) als daily proxies.
3. **`src/tradebot/validation/vol_metrics.py`** — de evaluatiemetrics uit sectie 9.3:
   - **QLIKE:** `RV_t / σ̂²_t − ln(RV_t / σ̂²_t) − 1` (de enige proxy-ruis-robuuste loss)
   - MSE-SD en MAE-SD
   - **Mincer-Zarnowitz regressie:** `RV̂_t = α + β·σ̂²_t + ε_t`, toets op `α = 0, β = 1`
   - **Diebold-Mariano met Harvey-Leybourne-Newbold correctie**
4. **`reports/GARCH_VS_EWMA_COMPETITION.md`** — de volledige QLIKE-competitie per symbool, per horizon, met DM-HLN p-waarden en het expliciete oordeel per model.
5. **`src/tradebot/regime/buckets.py`** — **M0 Causal Vol-Buckets**: harde drempels op EWMA z-score en ATR-ratio's. Nul latente toestanden, nul geschatte parameters, lekrisico nul. Dit is de te verslaan baseline.
6. **`src/tradebot/regime/markov.py`** — M1 Markov Chain en M2 Gaussian/Student-t HMM. **Uitsluitend filtered probabilities via het forward-algoritme.** De smoothed-variant (Baum-Welch over de volledige sample) bestaat in de codebase alleen als expliciet gemarkeerde diagnostiek en is technisch onbereikbaar vanuit de backtest.
7. **`src/tradebot/risk/hmm_regime.py` — herontworpen** (sectie 24, REDESIGN) naar het M2-contract in `regime/markov.py`. De EMA-crossover-fallback is in Phase 0 al gesloopt; hier komt de correcte implementatie.
8. **`tests/lookahead/test_filtered_only_enforcement.py`** — bewijst dat geen enkel backtest-pad smoothed probabilities kan bereiken. Een poging daartoe crasht met `CausalityViolationError`.
9. **`reports/M0_VS_HMM_BENCHMARK.md`** — M0 Causal Vol-Buckets versus M2 Filtered HMM: OOS Sharpe-verbetering ná transactiekosten, regime-stabiliteit, en turnover-impact van regimewisselingen.
10. **`src/tradebot/labeling/triple_barrier.py`** — triple-barrier labeling met expliciete horizon, als input voor meta-labeling.
11. **`src/tradebot/train/meta_label.py`** — CatBoost **uitsluitend** als secondary model: het primaire baseline-signaal uit Phase 3 bepaalt de richting, het ML-model voorspelt de succeskans (sizing / trade filter). Directionele voorspelling op ruwe returns is technisch geblokkeerd.
12. **`src/tradebot/validation/feature_importance.py`** — **MDI** (Mean Decrease Impurity) en **SFI** (Single Feature Importance), beide onder Purged Cross-Validation (sectie 12.1).
13. **`reports/META_LABELING_EVALUATION.md`** — OOS AUC, precision/recall bij het operationele werkpunt, en het expliciete oordeel tegen de drempel **AUC > 0.58** (sectie 24).
14. **`src/tradebot/portfolio/hrp.py` (research-gated)** — HRP blijft bestaan maar is technisch geblokkeerd voor productiegebruik totdat turnover-gecorrigeerde OOS Sharpe > Inverse Volatility is aangetoond.
15. **`src/tradebot/features/fracdiff.py`** — fractionele differentiëring (sectie 8.1): minimale `d ∈ [0,1]` die stationariteit bereikt (ADF p < 0.05) met maximaal behoud van geheugen.
16. **`src/tradebot/validation/econometrics.py`** — de verplichte toetsingsketen uit sectie 8.2: ADF & KPSS, CUSUM & structural break tests, Ljung-Box en Engle ARCH-test.
17. **Pre-registraties** voor elk van de drie onderzoekssporen (volatiliteit, regimes, meta-labeling), vastgelegd vóór de eerste run.

---

## STAPSGEWIJZE UITVOERING

**Stap 1 — Drie pre-registraties schrijven.**
Vóór één regel modelcode: registreer de drie hypotheses met hun nulhypotheses, universum, periode, parameterruimte, geplande aantal trials en **stop-criteria**:
- **H1:** GARCH-familie verslaat EWMA(0.94) OOS op QLIKE, significant onder Diebold-Mariano met HLN-correctie.
- **H2:** M2 Filtered HMM levert een superieure OOS Sharpe na kosten ten opzichte van M0 Causal Vol-Buckets.
- **H3:** CatBoost als meta-labeler bereikt OOS AUC > 0.58 op triple-barrier-labels van het Phase 3-baselinesignaal.

**Stap 2 — Econometrische toetsingsketen bouwen.**
Implementeer `validation/econometrics.py`. Draai ADF, KPSS, CUSUM, Ljung-Box en de Engle ARCH-test op elke reeks die de vol-pipeline binnenkomt. **De ARCH-test is de poortwachter:** is er geen aantoonbare conditionele heteroskedasticiteit, dan is een GARCH-structuur niet gerechtvaardigd en stopt het spoor daar.

**Stap 3 — Fractionele differentiëring implementeren.**
Bouw `features/fracdiff.py` en bepaal per reeks de minimale `d` die stationariteit bereikt. Rapporteer per reeks de gekozen `d` en het behouden geheugen.

**Stap 4 — Realized Volatility fundament leggen.**
Bouw `realized.py` en construeer de RV-proxy op minuutbasis. **De QLIKE-competitie is alleen zo goed als de proxy.** Documenteer expliciet de microstructuurruis-behandeling en de sampling-frequentie; verwijs naar de dekkingsanalyse uit Phase 1.

**Stap 5 — GARCH-familie implementeren.**
Bouw GARCH(1,1), GJR-GARCH, EGARCH en APARCH. Fit uitsluitend op expanding of rolling vensters binnen de walk-forward-structuur — **nooit** op de volledige sample. Niet-convergentie is een resultaat dat je registreert, geen probleem dat je wegvangt.

**Stap 6 — QLIKE-competitie draaien.**
Evalueer elk model OOS tegen de RV-proxy. Rapporteer QLIKE, MSE-SD, MAE-SD en de Mincer-Zarnowitz-coëfficiënten. Toets significantie met Diebold-Mariano + HLN. Per symbool, per horizon, met eerlijke telling van elke trial in `M`.

**Stap 7 — M0 Causal Vol-Buckets bouwen.**
Implementeer de baseline eerst. M0 heeft nul geschatte parameters en nul lekrisico; dat maakt hem tot een oneerlijk sterke tegenstander, en precies daarom is hij de juiste baseline.

**Stap 8 — Filtered HMM bouwen en de smoothed-route blokkeren.**
Implementeer het M2 HMM met het forward-algoritme voor filtered probabilities. Bouw daarna `test_filtered_only_enforcement.py` en bewijs dat het backtest-pad de smoothed-variant niet kan bereiken. **Bouw de blokkade voordat je het model op data loslaat** — anders sluipt de smoothed-variant er tijdens exploratie in.

**Stap 9 — M0 vs. HMM benchmarken.**
Conditioneer het Phase 3-baselinesignaal op M0 en op de filtered HMM-probabilities. Draai beide door de authoritative engine uit Phase 5 met gekalibreerde kosten. Meet de OOS Sharpe-verbetering ná kosten en de extra turnover die regimewisselingen veroorzaken.

**Stap 10 — Meta-labeling opzetten.**
Bouw triple-barrier labeling en train CatBoost als secondary model. Het primaire signaal bepaalt de richting; het ML-model voorspelt uitsluitend de succeskans. Gebruik Purged Walk-Forward met embargo. Bereken MDI en SFI. Rapporteer OOS AUC tegen de drempel 0.58.

**Stap 11 — HRP evalueren onder research-gate.**
Vergelijk HRP met Inverse Volatility op turnover-gecorrigeerd OOS-rendement. Blijft HRP achter, dan blijft de productiegate dicht. Dat is een geldig en definitief resultaat.

**Stap 12 — Oordelen en registreren.**
Ken elk model expliciet een status toe: `PROMOTED`, `UNPROVEN — insufficient data` of `FALSIFIED`. Schrijf alle drie de rapporten, registreer atomair in de ledger, en publiceer `reports/phase6_exit_report.md`.

---

## CRITERIA & VALIDATIE (EXIT CRITERIA)

1. **Alleen modellen met aantoonbare OOS-waarde worden goedgekeurd.** Elk model in de productiepijplijn heeft een ledger-entry met de statistische onderbouwing van zijn promotie.
2. **QLIKE-competitie compleet.** GARCH-familie vs. EWMA(0.94), per symbool en per horizon, met Diebold-Mariano + HLN p-waarden. GARCH promoveert uitsluitend bij significante OOS-superioriteit; anders blijft EWMA de productie-estimator.
3. **Filtered-only afgedwongen.** `test_filtered_only_enforcement.py` bewijst dat smoothed probabilities technisch onbereikbaar zijn vanuit elk backtest-pad (sectie 10.2). Een geïnjecteerde poging crasht aantoonbaar.
4. **M0 vs. HMM beslist.** Het HMM promoveert uitsluitend bij superieure OOS Sharpe **na transactiekosten** ten opzichte van M0 (acceptatiecriterium sectie 10.2). Anders: `UNPROVEN` of `FALSIFIED`, en M0 blijft productie-baseline.
5. **Meta-labeling beoordeeld tegen AUC > 0.58** (sectie 24). Onder die drempel blijft CatBoost `ARCHIVED`. Directionele CatBoost op ruwe daily returns is technisch geblokkeerd.
6. **Feature importance rigoureus.** MDI én SFI berekend onder Purged CV voor elk ML-spoor (sectie 12.1).
7. **Econometrische keten doorlopen.** ADF, KPSS, CUSUM, Ljung-Box en Engle ARCH gerapporteerd voor elke reeks in de vol-pipeline.
8. **HRP blijft research-gated** tenzij turnover-gecorrigeerde OOS-superioriteit boven Inverse Volatility is bewezen.
9. **Alle Phase 2-gates groen** voor elk gepromoveerd model: 6 lookahead-tests, DSR met eerlijke `M`, SPA over de multi-modelvergelijking.
10. **Nul stille degradatie.** Geen enkel model valt bij niet-convergentie of ontbrekende dependency terug op een eenvoudiger alternatief.

---

## REGELS & HANDLINGSINSTRUCTIES

- **Fail-fast compliance.** Niet-convergerende GARCH-fit, ontbrekende `hmmlearn`, te weinig observaties voor een HMM-fit: crashen en registreren. Nul `try/except` fallbacks. De EMA-crossover-fallback uit sectie 5.2 mag onder geen enkele omstandigheid terugkeren.
- **100% PIT rigor.** Elke modelfit gebruikt uitsluitend data ≤ `t`. Elke regime-classificatie is filtered. Elke label is gepurged en geëmbargood. Elke scaler is per fold gefit.
- **Onbewezen ≠ bewezen slecht (sectie 6).** Faalt een model door ontoereikende data, label het `UNPROVEN — insufficient data` en laat het in de Research Track. Falsificeer alleen wat aantoonbaar slechter is dan zijn baseline onder correcte condities.
- **Baseline-first.** Elk complex model wordt uitsluitend beoordeeld ten opzichte van zijn expliciete baseline: GARCH vs. EWMA, HMM vs. M0, HRP vs. Inverse Vol, CatBoost vs. het ongefilterde primaire signaal. Absolute performance is irrelevant.
- **Tel elke trial.** Elke parametervariant, elke symbool-specifieke fit, elke horizon telt mee in `M`. Een te lage `M` maakt DSR structureel te optimistisch — dit is de meest voorkomende vorm van zelfbedrog in kwantitatief onderzoek.
- **Geen hardcoded variabelen.** λ, `d`, drempels, aantal states, barrier-breedtes, embargo-lengtes en horizons komen uit `conf/model/`.
- **Atomaire commits.** `feat(volatility): add GJR-GARCH with student-t innovations`, `feat(regime): implement filtered-only HMM with forward algorithm`, `test(lookahead): block smoothed probabilities from backtest paths`, `docs(reports): publish GARCH vs EWMA QLIKE competition results`.
- **Rapporteer negatieve resultaten volledig.** Een gefalsificeerd model is een even waardevol resultaat als een gepromoveerd model, en aanzienlijk goedkoper dan de live ontdekking ervan.

---

## STARTINSTRUCTIE

> **Begin nu met Stap 1:** schrijf de drie pre-registraties (H1 GARCH vs. EWMA op QLIKE, H2 M2 Filtered HMM vs. M0 Causal Vol-Buckets, H3 CatBoost meta-labeling met AUC-drempel 0.58) via `registry/preregistration.py`, elk met nulhypothese, universum, periode, parameterruimte, gepland aantal trials en stop-criteria. Commit als `docs(registry): pre-register phase 6 advanced research waves`.
