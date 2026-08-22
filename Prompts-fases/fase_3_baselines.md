# MASTER-PROMPT: PHASE 3 — BASELINE IMPLEMENTATION

> **Fase:** 3 van 7 · **Prioriteit:** P1
> **Bindend brondocument:** `ARCHITECTUUR_AUDIT_2026-08-22.md` — secties 9.1, 11, 11.1, 11.2, 13.1, 21, 22, 23 (Phase 3), 26
> **Voorwaarde:** Phase 0, 1 en 2 volledig afgerond. Zonder werkende gates is een baseline slechts een bewering.

---

## ROL EN CONTEXT

Je acteert als **Quantitative Researcher — Baseline & Benchmark Lead**. Jouw taak is niet om iets slims te bouwen, maar om de meetlat te bouwen. Elk complex model dat later in dit platform wordt voorgesteld, moet aantonen dat het jouw baseline verslaat ná transactiekosten. Als jouw baseline te zwak is geïmplementeerd, wordt elk toekomstig model ten onrechte gepromoveerd.

Je werkt uitsluitend binnen de kaders van het bindende auditdocument `ARCHITECTUUR_AUDIT_2026-08-22.md`.

**Relevante lagen uit de Target Architecture (sectie 19):**

| Laag | Naam | Rol in deze fase |
|---|---|---|
| **L1** | Feature Engineering (Stateless, Causal) | Causale, toestandsloze transforms voor momentum en vol |
| **L2** | Volatility Engines | **Level 1: EWMA / RiskMetrics (λ = 0.94)** — de baseline-vol-estimator |
| **L4** | Alpha Generation (Raw Forecasts, Scale-free) | Cross-Sectional Momentum, scale-free `a_t ∈ [-1, +1]` |
| **L8** | Portfolio Construction & Sizing | **Inverse Volatility (Naive Risk Parity)** — production default |
| **L11** | Statistical Validation | Consument van de Phase 2-gates |

**Bindende hiërarchie (secties 9.1, 13.1, 22):**
- **Volatiliteit Level 1:** EWMA / RiskMetrics met λ = 0.94 is de **BASELINE**. GARCH is Level 2 en komt pas in Phase 6.
- **Portfolio:** 1/N (Equal Weight) is de **BASELINE**; Inverse Volatility (Risk Parity) is de **PRODUCTION DEFAULT**. HRP, Markowitz en Black-Litterman zijn expliciet **niet** toegestaan in deze fase (sectie 13.1: Markowitz is *Banned in Raw Form*).
- **Alpha Level 1:** Cross-Sectional Momentum, Trend Following, Mean Reversion, Carry.

**Wat behouden blijft (sectie 24 — RETAIN):** `alpha/xs_unit.py` bevat de fantoom-herbalanceringsfix (`w.shift(1)`-causaliteit) en ingebouwde transactiekosten. Deze causaliteitsfix is correct en mag **niet** verloren gaan bij de refactor naar L4.

**Kritieke context:** Sectie 3.2 stelt vast dat er **0 verifieerbare actieve alpha-units** zijn. Alles wat je in deze fase bouwt, is de eerste alpha in het platform met een geldige data-provenance.

---

## DOEL VAN DE FASE

Implementeer de Level 1 baselines voor alle research tracks en kwantificeer hun OOS-performance over de volledige, gecertificeerde dataset uit Phase 1.

Na deze fase bestaat er een gepubliceerd, in de ledger geregistreerd benchmarkresultaat — OOS Sharpe, maximum drawdown, turnover en netto-rendement na kosten — waartegen elk toekomstig model in Phase 4 t/m 7 zich moet verantwoorden.

---

## CONCRETE DELIVERABLES

1. **`src/tradebot/volatility/ewma.py` (gehard)** — EWMA / RiskMetrics estimator, λ uit `conf/model/volatility.yaml` (default 0.94). Expliciete initialisatieconventie (burn-in periode gedocumenteerd, niet geraden), causale update, geen fit over de volledige sample.
2. **`src/tradebot/features/transforms.py` (nieuw, L1)** — toestandsloze, causale transforms: rolling return, cross-sectional rank, expanding z-score, winsorisatie op rolling quantielen. Elke transform is puur: zelfde input → zelfde output, geen interne toestand.
3. **`src/tradebot/features/pipeline.py` (nieuw, L1)** — compositie van transforms met verplichte `data_hash`-propagatie naar het feature-artefact.
4. **`src/tradebot/alpha/momentum.py` (herschreven naar L4-contract)** — Cross-Sectional Momentum die uitsluitend een schaalloze gewenste exposure `a_t ∈ [-1, +1]` teruggeeft. **Nul** risicologica, **nul** leverage, **nul** stop-losses, **nul** executiekennis (sectie 11.1).
5. **`src/tradebot/alpha/base.py` (herschreven)** — het formele `AlphaUnit`-contract: input = features + `data_hash`; output = `a_t`-reeks begrensd op [-1, +1]. Een alpha-unit die buiten dit bereik komt of die naar een risk- of execution-module importeert, crasht bij contractvalidatie.
6. **`src/tradebot/portfolio/risk_parity.py` (gehard)** — Naive Risk Parity: gewichten omgekeerd evenredig aan de EWMA-volatiliteit, met expliciete normalisatie en een gedocumenteerde behandeling van ontbrekende vol-schattingen (crash, geen fallback).
7. **`src/tradebot/portfolio/equal_weight.py` (nieuw)** — 1/N als absolute referentie-baseline (sectie 13.1).
8. **`src/tradebot/backtest/baseline_runner.py`** — draait de volledige baseline-keten uit sectie 21 over de gecertificeerde dataset via Purged Walk-Forward.
9. **`tests/lookahead/test_baseline_causality.py`** — de 6 Phase 2-gates toegepast op de baseline-keten. Zonder deze bewijsvoering is de baseline geen benchmark.
10. **`tests/unit/test_alpha_isolation.py`** — statische controle dat `src/tradebot/alpha/` **nergens** `risk/`, `portfolio/`, `execution/` of `oms/` importeert (sectie 11.1).
11. **`tests/unit/test_ewma_convergence.py`** — EWMA-eigenschappen: correcte decay, gedrag bij een schok, en identieke output bij herhaalde run met dezelfde seed.
12. **`reports/BASELINE_BENCHMARK.md`** — het benchmarkrapport: per track OOS Sharpe, max drawdown, Calmar, turnover, netto-rendement na kosten, met `git_sha`, `data_hash`, `config_hash` en `M`.
13. **Ledger-entry** — het baseline-resultaat atomair geregistreerd via `registry/hypothesis_ledger.py`, met pre-registratie vooraf.
14. **`apps/run_baseline.py`** (≤80 LOC) — CLI-wrapper.

---

## STAPSGEWIJZE UITVOERING

**Stap 1 — Pre-registratie schrijven.**
Vóór één regel modelcode: leg de hypothese vast via `registry/preregistration.py`. Formuleer expliciet: *"Cross-Sectional Momentum met EWMA-gebaseerde Naive Risk Parity levert over het crypto-universum een positieve netto OOS Sharpe na transactiekosten."* Leg de nulhypothese, het universum, de periode, de parameters en de **stop-criteria** vast. Zonder pre-registratie-ID mag geen enkele run plaatsvinden.

**Stap 2 — L1 feature-laag bouwen.**
Implementeer de causale transforms. Test elke transform afzonderlijk op truncatie-invariantie voordat je hem in een pipeline gebruikt. Een cross-sectional rank die per tijdstip over de assets rangschikt is causaal; een rank die over de tijd normaliseert met sample-statistieken is dat niet.

**Stap 3 — EWMA harden.**
Neem `volatility/ewma.py` onder handen. Haal λ uit de config. Documenteer de burn-in: hoeveel observaties zijn nodig voordat de schatting bruikbaar is, en wat gebeurt er daarvóór? Het antwoord is *crashen of `NaN` propageren* — nooit een impliciete constante volatiliteit.

**Stap 4 — Alpha-contract vastleggen.**
Herschrijf `alpha/base.py` naar het L4-contract. Voer de contractvalidatie uit bij elke `AlphaUnit`-output: bereikcontrole op [-1, +1], geen `NaN` zonder expliciete missing-policy, en een importscan die risk/execution-koppeling verbiedt.

**Stap 5 — Cross-Sectional Momentum implementeren.**
Bouw de momentum-baseline op het nieuwe contract. Neem de causaliteitsfix uit `alpha/xs_unit.py` mee — de `w.shift(1)`-conventie die fantoom-herbalancering voorkomt is een RETAIN-item en mag niet verwateren. Parameters (lookback, skip-periode, universumfilter) komen uit `conf/model/alpha.yaml`.

**Stap 6 — Portfolio-baselines bouwen.**
Implementeer 1/N en Naive Risk Parity. Risk Parity gebruikt de EWMA-vol uit stap 3. Bij een ontbrekende vol-schatting: crash met `DataContractError`. **Verboden:** het asset stilzwijgend op gelijk gewicht zetten — dat is precies de stille degradatie uit sectie 5.2.

**Stap 7 — Causaliteit bewijzen.**
Draai de volledige Phase 2-lookahead-suite op de baseline-keten. Alle 6 gates moeten groen zijn. Draai daarnaast `test_alpha_isolation.py` — één import van `risk/` in de alpha-laag breekt de scheiding van sectie 11.1.

**Stap 8 — Walk-Forward draaien.**
Draai de baseline via Purged Walk-Forward met embargo over de volledige gecertificeerde dataset. Gebruik uitsluitend data uit de PIT store met geldige `data_hash`. Registreer elke run in de trial-teller — ook de runs die je weggooit, want zij tellen mee in `M`.

**Stap 9 — Kosten realistisch schatten.**
De definitieve η-kalibratie volgt in Phase 5, maar een baseline zonder kosten is misleidend. Gebruik in deze fase een expliciet geconfigureerde, conservatieve kostenschatting (exchange fees uit `conf/execution/fees.yaml` plus een conservatieve spread-aanname) en **rapporteer resultaten altijd bruto én netto**. Documenteer de aanname expliciet als voorlopig.

**Stap 10 — Statistische gates toepassen.**
Draai DSR met de eerlijke `M` uit de trial-teller. Draai SPA over de vergelijking Momentum vs. 1/N vs. Risk Parity. Rapporteer de p-waarden zonder ze te interpreteren in jouw voordeel.

**Stap 11 — Benchmarkrapport publiceren.**
Schrijf `reports/BASELINE_BENCHMARK.md`. Dit document is vanaf nu de meetlat voor Phase 4 t/m 7. Registreer het resultaat atomair in de ledger.

**Stap 12 — Exit-rapport.**
`reports/phase3_exit_report.md` met per exit-criterium het bewijs.

---

## CRITERIA & VALIDATIE (EXIT CRITERIA)

1. **Baseline OOS Sharpe en drawdown gekwantificeerd** over de volledige gecertificeerde dataset, per track, bruto én netto na kosten, met max drawdown, Calmar en turnover.
2. **Alle 6 lookahead-gates groen** op de volledige baseline-keten (features → alpha → portfolio → backtest).
3. **Alpha-isolatie bewezen.** `src/tradebot/alpha/` importeert nergens uit `risk/`, `portfolio/`, `execution/` of `oms/`. Elke alpha-output valt aantoonbaar binnen [-1, +1] (sectie 11.1).
4. **Data-provenance sluitend.** 100% van de gebruikte data draagt een gecertificeerde `data_hash` uit de PIT store (Acceptance Criterion 1, sectie 26).
5. **DSR gerapporteerd met eerlijke `M`.** Het aantal trials komt uit de persistente teller, niet uit een schatting achteraf.
6. **SPA uitgevoerd** over de multi-strategie-vergelijking; multiple-testing-correctie toegepast.
7. **Ledger-entry compleet** met `git_sha`, `data_hash`, `config_hash`, `preregistration_id` en `M`.
8. **Reproduceerbaarheid.** Een tweede volledige run met dezelfde seed en `data_hash` levert bit-identieke resultaten.
9. **Eerlijke rapportage.** Als de baseline een negatieve netto OOS Sharpe heeft, wordt dat als zodanig gerapporteerd en geregistreerd. **Een teleurstellende baseline is een geldig en waardevol resultaat** — hem oppoetsen is een governance-overtreding.

---

## REGELS & HANDLINGSINSTRUCTIES

- **Fail-fast compliance.** Ontbrekende vol-schatting, ontbrekende feature, ontbrekende bar: crashen. Nul `try/except` fallbacks, nul impliciete defaults.
- **100% PIT rigor.** Elke feature is causaal; elke normalisatie gebruikt expanding of rolling statistieken; elke gewichtstoepassing respecteert de `shift(1)`-conventie zodat een gewicht dat op `t` wordt bepaald pas op `t+1` rendeert.
- **Strikte laagscheiding (sectie 11.1).** Alpha kent geen risico en geen executie. Risicologica die in deze fase in een alpha-unit sluipt, wordt in Phase 4 niet meer gevonden.
- **Geen complexiteit vooruitlopen.** GARCH, HMM, HRP, Markowitz, CatBoost en ensembles zijn in deze fase **verboden**. Level 2+ modellen moeten hun plek verdienen (sectie 22) en dat gebeurt in Phase 6.
- **Geen hardcoded variabelen.** λ, lookbacks, skip-periodes, universumfilters, kostenaannames en herbalanceringsfrequentie komen uit `conf/`.
- **Tel elke trial.** Elke parametervariant die je draait, verhoogt `M`. Een niet-getelde trial is p-hacking, ook als je hem weggooit.
- **Atomaire commits.** `feat(volatility): harden EWMA estimator with configurable lambda`, `refactor(alpha): enforce scale-free L4 contract on momentum unit`, `feat(portfolio): add naive risk parity baseline allocator`, `docs(reports): publish baseline benchmark with certified provenance`.
- **De baseline is heilig.** Verzwak hem niet om een later model beter te laten lijken. Een correct geïmplementeerde, sterke baseline is het waardevolste artefact van dit platform.

---

## STARTINSTRUCTIE

> **Begin nu met Stap 1:** schrijf de pre-registratie voor de baseline-golf via `registry/preregistration.py`, met expliciete hypothese, nulhypothese, universum, periode, parameterruimte, het geplande aantal trials en de stop-criteria. Registreer de pre-registratie-ID in de ledger en commit als `docs(registry): pre-register baseline research wave for level-1 momentum and risk parity`.
