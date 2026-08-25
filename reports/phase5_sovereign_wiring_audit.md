# PHASE 5 — SOVEREIGN WIRING AUDIT

> **Status:** forensische discovery. Dit rapport is geschreven **vóór elke
> codewijziging** van Phase 5 en beschrijft de repository zoals hij op
> `b206895` staat. Het wordt niet achteraf bijgewerkt om de uitkomst te
> flatteren; de voortgang staat in `reports/phase5_exit_report.md`.

**Gegenereerd:** 2026-08-25
**git_sha:** `b206895`
**Bindend brondocument:** `docs/ARCHITECTUUR_AUDIT_2026-08-22.md` §14, §14.1, §15, §16, §19, §24, §26, §27
**Voorgaand bewijs:** `reports/phase4_entanglement_map.md`, `reports/phase4_exit_report.md` §3.1
**Risk `config_hash` op dit sha:** `47821e47fe2cec30`

---

## 0. De bevinding in één zin

De soevereine risicolaag is volledig gebouwd, gevalideerd en getest — en wordt
door **geen enkel productiepad** aangeroepen.

---

## 1. Methode

Repository-wide grep over de patronen uit de fase-opdracht §3
(`max_position`, `max_weight`, `max_exposure`, `max_gross`, `max_net`,
`max_leverage`, `concentration`, `cluster`, `vol_target`, `volatility`,
`drawdown`, `risk_limit`, `position_limit`, `participation`, `risk_budget`,
`cap`, `constraint`, hardcoded percentages), gevolgd door handmatige inspectie
van elke hit in `backtest/`, `portfolio/`, `execution/`, `risk/`, `live/`,
`oms/`, `apps/`, `tests/` en `conf/`.

* **250** ruwe treffers in de niet-`risk/`-modules over **74** bestanden.
* Na ontdubbeling en het wegstrepen van commentaar, metriekberekeningen en
  numerieke clips (`np.clip(idx, 0, n-1)`): **19** feitelijke runtime-constraints
  buiten `risk/`.
* Elk van die 19 is hieronder geclassificeerd.

De Phase 4-entanglement map telde er 16. Het verschil van 3 zit in constraints
die Phase 4 niet als risicolimiet las: de funding-cap (§4.3, F1), de
participatie-cap in `max_size_for_alpha` (F2) en de fat-finger volumecheck
(F3). Alle drie zijn hieronder alsnog geclassificeerd.

---

## 2. Stap 1 — Waar wordt de Sovereign Risk Layer aangeroepen?

`RiskEngine.decide()` is de enige soevereine ingang. Volledige lijst van
aanroepsites in niet-testcode:

| Aanroepsite | Soort | Draait in productie? |
|---|---|---|
| `risk/stress_test.py:241` | S1–S4 stress-harnas | nee — analyse-instrument |
| `risk/stress_report.py:260` | baseline-overlay voor `RISK_STRESS_REPORT.md` | nee — rapportgenerator |
| `apps/run_stress.py:63` | CLI om beide bovenstaande te draaien | nee — analyse-instrument |

**Dat is de volledige lijst.**

### 2.1 Waar hij NIET wordt aangeroepen

| Module | Draait in het beslispad? | Roept sovereign aan? |
|---|---|---|
| `backtest/portfolio.py` (`PortfolioBacktester`) | **ja** | **nee** |
| `backtest/per_side.py` (`internal_backtest`) | **ja** | **nee** |
| `backtest/bidirectional.py` (`bidirectional_backtest`) | **ja** | **nee** |
| `backtest/baseline_runner.py` (Phase 3-baseline) | **ja** | **nee** |
| `portfolio/constraints.py` | **ja** | **nee** |
| `portfolio/legacy_sizing.py` (`PortfolioRiskManager`) | **ja** | **nee** |
| `live/portfolio_controller.py` | **ja** | **nee** |
| `live/execution_controller.py` | **ja** | **nee** |
| `live/circuit_breaker.py` | **ja** | **nee** |
| `oms/router.py` | **ja** | **nee** |
| `apps/backtest_portfolio.py` | **ja** | **nee** |

**Consequentie.** Het getal in `reports/BASELINE_BENCHMARK.md` en elk getal in
`reports/phase3_exit_report.md` is geproduceerd door een pad dat de soevereine
risicolaag niet passeert. Dat is niet een tekortkoming van die rapporten — zij
zeggen het zelf — maar het betekent dat exit-criterium 18 (herberekening van de
Phase 3-baseline door de volledige sovereign + execution pipeline) geen
cosmetische stap is maar de eerste meting van dit systeem.

---

## 3. Stap 2 — Constraint-inventaris

Kolom **Klasse** gebruikt de taxonomie uit de fase-opdracht §3:
`sovereign` · `execution-only` · `accounting invariant` ·
`market-mechanics` · `legacy duplicate` · `bug` · `deferred → Phase 7`.

### 3.1 `backtest/` — de authoritative-engine-kandidaten

| # | Locatie | Constraint | Huidige eigenaar | Sovereign equivalent | Klasse | Actie | Status |
|---|---|---|---|---|---|---|---|
| B1 | `backtest/evaluation.py:239,305-306` | `max_leverage` hard cap op Kelly-leverage; `capped = raw_lev > max_leverage` | L10 evaluatie | `risk.limits.apply_per_asset_cap` (`max_position_pct`) | **legacy duplicate** | verwijderen mét de engine na pariteitsbewijs | OPEN |
| B2 | `backtest/evaluation.py:721-750` | `max_portfolio_leverage` → gross-cap over gelijktijdig open trades | L10 evaluatie | `risk.limits.apply_gross_cap` (`gross_cap`) | **legacy duplicate** | idem | OPEN |
| B3 | `backtest/per_side.py:52` · `backtest/bidirectional.py:58` | `max_leverage: float = 2.0` (eigen default, wijkt af van `conf/risk` 1.5) | L10 engines | `max_position_pct` / `gross_cap` | **legacy duplicate** | idem | OPEN |
| B4 | `backtest/portfolio.py:347` | `risk_manager.size_position(...)` — vol-target, correlatie-multiplier, DD-state, crisis-multiplier, HMM-cap, CVaR | `portfolio/legacy_sizing.py` (L8) | de volledige `RiskEngine` | **legacy duplicate** (tweede risicoregime) | vervangen door `RiskEngine.decide()` | OPEN |
| B5 | `backtest/portfolio.py:206,351` | `per_asset_cap` per `AssetTrack`, default `2.0` (`tracks.py:50`) | L10 track-dataclass | `max_position_pct` | **legacy duplicate** | idem | OPEN |
| B6 | `backtest/portfolio.py:408` | `fr_capped = np.clip(fr, -0.02, 0.02)` — funding-cap | L10 | geen | **market-mechanics** | BLIJFT lokaal — dit modelleert de funding-cap van de exchange, geen risicobudget. Verhuist naar `conf/execution/` als expliciete venue-parameter | OPEN |
| B7 | `backtest/baseline_runner.py:145` | `gross_target` normalisatieconventie | L10 baseline | `gross_cap` | **execution-only** | BLIJFT — het is een vergelijkingsconventie tussen tracks, geen limiet; de module zegt dat zelf. Wordt in Phase 5 ná de sovereign beslissing toegepast, niet ervoor | OPEN |

### 3.2 `portfolio/` — L8

| # | Locatie | Constraint | Huidige eigenaar | Sovereign equivalent | Klasse | Actie | Status |
|---|---|---|---|---|---|---|---|
| D1 | `portfolio/constraints.py:35` | `max_weight: float = 0.40` | L8 | `max_concentration` (0.40) | **legacy duplicate** | pure adapter naar sovereign óf verwijderen | OPEN |
| D2 | `portfolio/constraints.py:37` | `max_leverage: float = 1.0` | L8 | `gross_cap` (1.5) | **legacy duplicate** — en de waarden **wijken af** | idem | OPEN |
| D3 | `portfolio/constraints.py:36` | `min_weight: float = 0.02` | L8 | geen | **execution-only** | BLIJFT — een dust-drempel is een kostenbeslissing (een positie van 0,4 % kost meer aan fees dan hij bijdraagt), geen risicolimiet. Verhuist naar `conf/execution/` | OPEN |
| D4 | `portfolio/constraints.py:38,86-99` | `max_turnover: float = 1.0` + blend-naar-huidig | L8 | geen | **execution-only** | BLIJFT — turnovercap is een kostenbeslissing. Phase 4 classificeerde hem al zo | OPEN |
| D5 | `portfolio/constraints.py:65-87` | volgorde clip→renorm→leverage is **impliciet in de codevolgorde** | L8 | `constraint_order` in `conf/risk/` | **bug** | de sovereign engine crasht als een limiet niet in `constraint_order` staat; hier is de volgorde onzichtbaar en niet configureerbaar | OPEN |
| D6 | `portfolio/legacy_sizing.py:380-546` | `vol_target_multiplier`, `correlation_multiplier`, `_compute_crisis_multiplier` | L8 | `risk.vol_targeting.apply_volatility_target` | **legacy duplicate** | uit het beslispad; module blijft als research-artefact | OPEN |
| D7 | `portfolio/legacy_sizing.py:547-575` | `dd_state()` — vierde onafhankelijke drawdown-implementatie | L8 | `risk.kill_switches.apply_drawdown_breaker` | **legacy duplicate** | idem | OPEN |

### 3.3 `live/` — L13 (formeel Phase 7)

| # | Locatie | Constraint | Huidige eigenaar | Sovereign equivalent | Klasse | Actie | Status |
|---|---|---|---|---|---|---|---|
| C1 | `live/circuit_breaker.py:79-85` | `max_drawdown_pct=0.08`, `max_intraday_drawdown_pct=0.05`, `max_daily_loss_pct=0.03`, `max_position_age_h=48` | L13 | `max_drawdown_pct`, `daily_loss_limit`, `max_position_age_h` | **deferred → Phase 7** | contractgrens vastleggen in Phase 5; bedrading in Phase 7 | OPEN |
| C2 | `live/circuit_breaker.py` reset | HALT-state is **in-memory**; procesherstart wist de halt | L13 | `risk.kill_switches.HaltStore` (persistent) | **bug** | `HaltStore` bestaat en is aansluitbaar; Phase 7 sluit aan | OPEN |
| C3 | `live/execution_controller.py:243-267` | `_check_position_limits` — per-symbol notional + gross notional | L9 | `max_position_pct`, `gross_cap` | **deferred → Phase 7** | idem C1 | OPEN |
| C4 | `live/execution_controller.py:95` | `self._max_gross_notional = float("inf")` — **default oneindig** | L9 | `gross_cap` | **bug** | wie de setter niet aanroept handelt zonder gross-limiet. Dit is de "stil degraderen naar geen limiet"-modus die §23 verbiedt | OPEN |
| C5 | `live/portfolio_controller.py:79` | `max_weight=0.40` via `PortfolioConstraints` | L8/L13 | `max_concentration` | **deferred → Phase 7** | erft van D1 zodra `portfolio/constraints.py` een adapter is | OPEN |
| C6 | `live/execution_controller.py:61-63` | `max_kelly_fraction=0.25`, `min_confidence=0.55`, `max_weight_change=0.10` | L9 | — | **legacy duplicate** (`max_kelly_fraction`) + **alpha-parameter** (`min_confidence`) | `min_confidence` is modelvertrouwen als sizingparameter — exact de bypass die audit §14 uitsluit. Phase 7 | OPEN |

### 3.4 Nieuw t.o.v. Phase 4

| # | Locatie | Constraint | Huidige eigenaar | Sovereign equivalent | Klasse | Actie | Status |
|---|---|---|---|---|---|---|---|
| F1 | `backtest/portfolio.py:408` | funding-cap ±2 %/8h | L10 | geen | **market-mechanics** | zie B6 | OPEN |
| F2 | `execution/market_impact.py:180,187` | `cap_participation: float = 0.10` in `max_size_for_alpha` | L9 | `adv_participation_cap` (0.01) | **legacy duplicate** — en **10× ruimer** dan de sovereign waarde | adapter naar sovereign, of expliciet als execution-mechanica classificeren | OPEN |
| F3 | `live/execution_controller.py` fat-finger | `notional > 0.5 % daily volume` → reject | L9 | `adv_participation_cap` | **execution-only** | BLIJFT — een fat-finger-gate is een operationele sanity check op de órder, geen portefeuillelimiet. Maar de drempel moet uit config komen | OPEN |

### 3.5 Constraints die legitiem buiten de sovereign laag blijven

Volledige lijst, met de architecturale reden. Elke andere constraint moet
verhuizen of verdwijnen — er is geen vierde optie (fase-opdracht §4.2).

| Constraint | Reden dat hij niet-sovereign is |
|---|---|
| `min_weight` / `min_notional_per_trade` (D3) | Kostendrempel. Een positie onder de drempel verliest geld aan fees; dat is een executiebeslissing, geen risicolimiet. Verkleint exposure, dus schendt de monotoniteitsregel niet. |
| `max_turnover` (D4) | Kostenbeslissing. Idem. |
| Funding-cap ±2 % (B6/F1) | Modelleert een **eigenschap van de venue**, niet van ons risicobudget. Bybit begrenst funding zelf; een backtest die dat niet doet, modelleert een markt die niet bestaat. |
| Fat-finger gate (F3) | Operationele sanity check op één order. Beschermt tegen een defecte caller, niet tegen marktrisico. |
| `gross_target` normalisatie (B7) | Vergelijkingsconventie tussen tracks binnen één rapport. Wordt ná de sovereign beslissing toegepast. |
| Numerieke clips (`np.clip(idx, 0, n-1)`) | Index-bounds. Geen limiet. |

---

## 4. Stap 3 — Waar de twee regimes elkaar tegenspreken

Dit is de kern van no-go-conditie §27 bullet 1. Onderstaande waarden gelden
**tegelijk** in de huidige repository:

| Grootheid | Sovereign (`conf/risk/default.yaml`) | Lokaal | Locatie | Factor |
|---|---:|---:|---|---:|
| Gross / leverage cap | **1.50** | 1.00 | `portfolio/constraints.py:37` | 0,67× (strenger) |
| Gross / leverage cap | **1.50** | 2.00 | `backtest/per_side.py:52`, `bidirectional.py:58` | **1,33× ruimer** |
| Per-asset cap | **0.25** | 2.00 | `backtest/tracks.py:50` (`per_asset_cap`) | **8× ruimer** |
| Concentratie | **0.40** | 0.40 | `portfolio/constraints.py:35` | gelijk — maar niet gedeeld |
| Participatie (ADV) | **0.01** | 0.10 | `execution/market_impact.py:180` | **10× ruimer** |
| Gross notional (live) | **1.50 × equity** | `inf` | `live/execution_controller.py:95` | **onbegrensd** |
| Drawdown-halt | **0.08** | 0.08 / 0.05 intraday | `live/circuit_breaker.py:79-80` | gelijk, maar niet-persistent |

**Vier van de zeven lokale limieten zijn RUIMER dan de sovereign policy.** Een
backtest die vandaag draait, kan dus posities aannemen die de soevereine laag
zou hebben afgewezen — en het resultaat draagt geen enkel spoor daarvan.

---

## 5. Stap 4 — Het definitieve contract

Het contract bestaat al en hoeft niet opnieuw te worden ontworpen
(fase-opdracht §2.1: *"Niet opnieuw ontwerpen wat al bestaat"*). De mapping van
de conceptuele namen uit de fase-opdracht §5 naar de bestaande types:

| Fase-opdracht §5 | Bestaand type | Bestand |
|---|---|---|
| `RiskRequest.target_weights` | `desired_exposure: Mapping[str, float]`, `a_t ∈ [-1,+1]` | `risk/contract.py::validate_desired_exposure` |
| `RiskRequest.market_state` / `volatility_state` / `cluster_state` | `MarketState(asof_ts, sigma_hat, adv_usd, cluster)` | `risk/contract.py::MarketState` |
| `RiskRequest.portfolio_state` / `positions` / `cash` | `RiskState(equity, high_water_mark, day_start_equity, halted, …)` | `risk/contract.py::RiskState` |
| `RiskRequest.timestamp` | `MarketState.asof_ts` (tz-aware, verplicht) | idem |
| `SovereignRiskEngine` | `RiskEngine.decide(...)` | `risk/engine.py` |
| `RiskDecision.approved_targets` | `permitted_exposure` | `risk/contract.py::RiskDecision` |
| `RiskDecision.clipped_targets` / `violations` / `applied_limits` | `binding_constraints: tuple[BindingConstraint, ...]` | idem |
| `RiskDecision.status` | `unconstrained: bool` + `risk_state_out.halted` | idem |
| `RiskDecision.config_hash` / `policy_version` | `config_hash` | idem |
| `RiskDecision.risk_snapshot` | `risk_state_out` + `as_record()` | idem |

**Wat ontbreekt en in Phase 5 wordt toegevoegd:**

1. `RiskRequest.execution_context` — er is geen veld waarmee de executielaag
   zijn eigen staat aan de risicobeslissing kan tonen. Nodig voor
   `BacktestExecutionContext` / `LiveExecutionContext` (fase-opdracht §4.3).
2. `RiskDecision.rejected_targets` — het contract kent clipping, geen expliciete
   afwijzing. Een positie die naar nul wordt geclipt is nu niet te onderscheiden
   van een positie die is afgewezen.
3. Er is geen gedeeld `ExecutionContext`-protocol dat backtest en live beide
   implementeren.

---

## 6. Wat dit rapport NIET vaststelt

* Of de vier backtesters onderling identiek rekenen. Dat is
  `reports/phase5_engine_diff.md`.
* Of de clusterlabels correct zijn. Dat is
  `reports/phase5_cluster_concentration_audit.md`.
* Of de sovereign limieten zelf de juiste waarden hebben. Phase 4 heeft ze
  vastgesteld op "strengste van de bestaande"; deze fase bedraadt ze, en
  bevraagt ze alleen waar feasibility dat afdwingt (§19).

---

## 7. Samenvatting per exit-criterium

| Exit-criterium (fase-opdracht §26) | Stand op `b206895` |
|---|---|
| 10 — Sovereign volledig wired | **NIET GEHAALD** — 0 van 11 productiepaden |
| 11 — `portfolio/constraints.py` geen concurrerende authority | **NIET GEHAALD** — D1, D2, D5 |
| 12 — Backtest bevat geen eigen risk regime | **NIET GEHAALD** — B1–B5 |
| 13 — Resterende lokale constraints expliciet geclassificeerd | **GEHAALD door dit rapport** (§3.5) |
| 14 — Risk config in `registry/risk_registry.py` | **GEHAALD in Phase 4** |
| 15 — Risk config verandert `M` niet | **GEHAALD in Phase 4** — `tests/unit/test_risk_config_registry.py` |
