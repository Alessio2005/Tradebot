# PHASE 4 — ENTANGLEMENT MAP: ALPHA ↔ RISK

> **Fase:** 4 van 7 — Risk & Volatility Architecture · **Prioriteit:** P1
> **Bindend brondocument:** `ARCHITECTUUR_AUDIT_2026-08-22.md` — secties 4, 9, 9.1, 11.1, 13.1, 14, 14.1, 19, 23 (Phase 4), 24, 26
> **Stap:** 1 van 11 — *"Verstrengeling in kaart brengen"*
> **Gegenereerd:** 2026-08-23 · **`git_sha` van de gescande boom:** `18d232a`
> **Scope van de scan:** volledige `src/tradebot/`-boom, plus `conf/`, `apps/` en `scripts/` voor zover die risicodrempels vaststellen.

---

## 0. Wat dit document is, en wat het niet is

Dit is het **werkplan én het bewijsstuk** voor auditsectie 4:

> *"Risicolimieten en volatiliteitstargeting bevinden zich gedeeltelijk binnen
> de alpha-units en feature-pipelines, wat een strikte scheiding van
> verantwoordelijkheden onmogelijk maakt."*

Die bewering wordt hieronder niet herhaald maar **geadresseerd**: elke locatie
staat met bestand, regelnummer, gemeten waarde en bestemming. Er is niets
gewijzigd in deze stap — dit is uitsluitend inventarisatie. De eerste
code-wijziging valt in stap 3.

**Meetbaselines vastgelegd vóór de eerste wijziging** (zodat elke latere claim
van "geen regressie" verifieerbaar is):

| Ratchet | Commando | Waarde op `18d232a` |
|---|---|---|
| Testsuite | `pytest -q` | **1.190 tests**, 6 falend, 21 geskipt |
| Pre-existente faalgevallen | — | 4 × `tests/killgates/test_expansion_killgates.py`, 2 × `tests/property/test_hypothesis_kernels.py` |
| Hardcoded-parameters | `scripts/check_hardcoded_params.py` | 330 literals / 105 bestanden — **binnen budget** (exit 0) |
| Fallback-audit | `scripts/audit_fallbacks.py --strict` | 36 bevindingen, **0 blokkerend** |

De 6 faalgevallen zijn pre-existent en vallen buiten Phase 4. Elke afwijking
van deze vier getallen ná stap 11 is een regressie van deze fase.

---

## 1. Kernbevinding

De verstrengeling is **niet** primair een importprobleem. Statisch is de boom
al schoon:

- `alpha/` importeert **nul** keer uit `risk/`, `portfolio/`, `execution/` of
  `oms/` — afgedwongen door de AST-scan in `alpha/base.py` (`assert_alpha_isolation`),
  gebouwd in Phase 3;
- `risk/` importeert **nul** keer uit `alpha/`.

De verstrengeling zit een laag dieper, en is daardoor gevaarlijker:

1. **Risk kent alpha-grootheden zonder ze te importeren.** `risk/kelly.py`
   neemt `expected_alpha` en een meta-label-confidence als *argument*. De
   soevereine risicolaag schaalt daarmee mee met de overtuiging van het model.
   Dat is precies de bypass die sectie 14 uitsluit.
2. **Dezelfde limiet bestaat meervoudig, met verschillende waarden.** Er zijn
   **4 onafhankelijke drawdown-breakers**, **3 vol-targeting-implementaties**
   en **4 positielimietcontroles** in de boom. Geen ervan kent de andere.
3. **Er is geen enkele bron van waarheid voor een drempel.**
   `max_gross_leverage` heeft **vier verschillende waarden** op vijf plaatsen
   (§5). Welke bindt, hangt af van welke app je start.
4. **Het auditspoor bestaat niet.** De enige registratie van een risico-ingreep
   is `SizingDecision.reason`, een geformatteerde string
   (`risk/portfolio.py:801`). Niet machineleesbaar, niet herleidbaar tot een
   geconfigureerde drempel.

> **Consequentie voor de fasevolgorde:** de ontkoppelingstest (stap 9) is met de
> huidige boom **niet falsifieerbaar op imports alleen** — die is al groen. De
> test moet daarom de scherpere vorm krijgen die de opdracht voorschrijft:
> nul alpha-*parameters* in `risk/`, plus bit-identieke `permitted_exposure`.

---

## 2. (a) Risicologica buiten `risk/`

### 2.1 In `alpha/` — L4 doet aan positiegrootte

| # | Locatie | Wat er staat | Waarom dit L7 is | Bestemming |
|---|---|---|---|---|
| A1 | `alpha/adaptive_wf.py:55` | `target_vol: float = 0.40` — *"book vol target (annualised)"* | Volatility targeting is de kern-deliverable van L7 (audit §14.1). Een alpha-unit die zijn eigen boek-vol target zet, is een tweede risicolaag. | `risk/vol_targeting.py` + `conf/risk/` |
| A2 | `alpha/adaptive_wf.py:259` | `return pnl * (cfg.target_vol / bv) if bv > 0 else pnl` | Dezelfde formule als §14.1, maar **zonder `MaxLeverage`-cap** en met een stille pass-through bij `bv <= 0` — een fallback naar "geen targeting". | `risk/vol_targeting.py` (mét cap, mét fail-fast) |
| A3 | `alpha/adaptive_wf.py:63,298` | `regime_degross: float = 0.7` — *"max gross cut in bull mania"* | De-grossing is een risico-ingreep. Hier gebeurt hij regime-conditioneel binnen L4, onzichtbaar voor L7. | `risk/kill_switches.py` (getrapte de-grossing) |
| A4 | `alpha/adaptive_wf.py:339` | `max_drawdown=float((eq / eq.cummax() - 1).min())` | Vierde onafhankelijke drawdown-berekening in de boom. | `risk/kill_switches.py` (HWM-breaker) |
| A5 | `alpha/xs_unit.py:216-223` | BAB leg-leverage: `w[mask] / max(scale, 0.25)` | Expliciete hefboomtoekenning binnen een alpha-unit; de floor `0.25` is een verborgen leveragelimiet. | Splitsen: β-neutralisatie blijft L4-signaal, de leverage-floor gaat naar `risk/limits.py` |

**Grensgeval, expliciet als zodanig geregistreerd — vol-schaling van het signaal.**
`alpha/momentum.py:105-117`, `alpha/csm_volume_clock.py:192-196` en
`alpha/carry.py:113-114` delen het ruwe signaal door een intern berekende
`ann_vol`. Dit is **geen positiegrootte**: het maakt het signaal schaalvrij
zodat het over assets vergelijkbaar is. Maar:

- het gebruikt een **eigen**, ad-hoc vol-schatting (`np.std(...) * sqrt(252)`,
  vensterlengte hardcoded op 63 in `momentum.py:116`), niet de L2 EWMA-schatting
  uit `conf/model/volatility.yaml`;
- de `+ 1e-9` in de noemer is een stille bescherming tegen nul-vol — precies het
  gedrag dat `vol_targeting.py` volgens deliverable 1 moet verbieden.

`alpha/momentum.py:230-232` benoemt dit zelf al: *"Die vol-schaling is precies
wat sectie 11.1 uit L4 verbant"*, en laat de code bewust ongewijzigd onder
**DI-12**. De nieuwe `CrossSectionalMomentum` (`alpha/momentum.py:251`) is
schoon en heeft géén vol-schaling. **Stap 8 raakt de 26 legacy-units niet
functioneel aan**; hij bewijst dat de units die op het L4-contract draaien nul
risicoparameters kennen, en registreert de rest onder DI-12.

### 2.2 In `backtest/` — L10 dwingt limieten af

| # | Locatie | Wat er staat | Waarom dit L7 is | Bestemming |
|---|---|---|---|---|
| B1 | `backtest/evaluation.py:239-306` | Kelly-sizing met `max_leverage` als harde cap; `capped = raw_lev > max_leverage` | Een leveragelimiet in de evaluatielaag. De backtest bepaalt hiermee zijn eigen risicobudget. | `risk/limits.py` (per-asset cap) |
| B2 | `backtest/evaluation.py:722-731` | `max_portfolio_leverage` → gross-cap over gelijktijdig open trades | Dit **is** de Gross Exposure Cap uit §14.1, geïmplementeerd in L10. | `risk/limits.py::gross_exposure_cap` |
| B3 | `backtest/per_side.py:52,63` · `backtest/bidirectional.py:51,58` | `max_leverage: float = 2.0`, `max_portfolio_leverage: float \| None` | Dezelfde twee caps nogmaals, met eigen defaults, in twee backtesters. | idem B1/B2 |
| B4 | `backtest/portfolio.py:313-320` | Roept `PortfolioRiskManager.size_position()` per asset aan | Legitiem *consument*-gedrag, maar de backtester is nu de enige plek waar de risicolaag wordt aangeroepen — er is geen standalone entrypoint. | Blijft consument; krijgt `RiskEngine` i.p.v. `PortfolioRiskManager` |

### 2.3 In `live/` — L13 heeft zijn eigen risicolaag

| # | Locatie | Wat er staat | Waarom dit L7 is | Bestemming |
|---|---|---|---|---|
| C1 | `live/circuit_breaker.py:79-85` | `max_drawdown_pct=0.08`, `max_intraday_drawdown_pct=0.05`, `max_position_age_h=48` | Volwaardige drawdown-breaker + kill switch, volledig los van `risk/`. | Bron voor `risk/kill_switches.py`; `live/` wordt consument |
| C2 | `live/circuit_breaker.py:255-260` | `reset()` zet `circuit_breaker_active = False` | De reset is *handmatig* (docstring: *"requires human intervention"*) maar de state is **in-memory**: een procesherstart wist de HALT. | `risk/kill_switches.py` met **persistente** `HALTED`-state |
| C3 | `live/execution_controller.py:243-267` | `_check_position_limits` — per-symbol notional + gross notional | Positielimieten in de executielaag (L9). | `risk/limits.py`; executie leest het besluit |
| C4 | `live/execution_controller.py:95` | `self._max_gross_notional: float = float("inf")` | **Default = oneindig.** Wie de setter niet aanroept, handelt zonder gross-limiet. Dit is de "stilzwijgend degraderen naar geen limiet"-modus uit de faseregels. | `risk/limits.py`, fail-fast bij ontbrekende config |
| C5 | `live/portfolio_controller.py:79` | `max_weight=0.40` concentratielimiet | Concentration cap in L8. | `risk/limits.py::concentration_cap` |

### 2.4 In `portfolio/` — L8 knipt zelf

| # | Locatie | Wat er staat | Waarom dit L7 is | Bestemming |
|---|---|---|---|---|
| D1 | `portfolio/constraints.py:35-38` | `max_weight=0.40`, `max_leverage=1.0`, `max_turnover=1.0` | Concentratie- en leveragelimieten. | `max_weight`/`max_leverage` → `risk/limits.py`; `max_turnover` blijft L8 (kostenbeslissing, geen risicolimiet) |
| D2 | `portfolio/constraints.py:65-87` | Iteratieve clip naar `max_weight`, daarna leverage-renormalisatie | De volgorde van toepassing is impliciet in de code-volgorde — verboden onder stap 4. | `risk/limits.py`, volgorde uit `conf/risk/` |

---

## 3. (b) Alpha-logica binnen `risk/`

| # | Locatie | Wat er staat | Waarom dit L4 is | Actie |
|---|---|---|---|---|
| E1 | **`risk/factor_alpha.py`** (143 LOC) | HAC/Newey-West residual-alpha-regressie; `G4_FACTORSETS`; `passes` = *"positieve residual alpha, p<0.05, t>=2"* | Een **alpha-bewijsvoeringsmodule**. Hij meet of een strategie alpha heeft. Dat is L4/L11, nooit L7. De audit noemt zijn aanwezigheid in `risk/` symptomatisch. | **Verplaatsen** naar `alpha/factor_alpha.py` (deliverable 6) |
| E2 | `risk/factor_alpha.py:35` | `_T_ALPHA_FLOOR = 2.0` | Een alpha-significantiedrempel, hardcoded, in de risicolaag. | Verhuist mee; drempel naar `conf/` |
| E3 | `risk/kelly.py:89,111,125` | `gap_risk_kelly_size(expected_alpha, ...)` — *"per-bar edge (e.g. from meta-label score)"* | **De ernstigste bevinding.** De risicolaag schaalt de positie met de *verwachte alpha*. Een unit die zichzelf overtuigd noemt, krijgt meer kapitaal — de bypass die sectie 14 uitsluit. | Kelly verlaat het besluitpad van de `RiskEngine`; blijft beschikbaar als L8-sizinghulpmiddel |
| E4 | `risk/kelly.py:133-160` | `meta_label_kelly(...)` — schaalt met meta-label-confidence | Modelvertrouwen als risicoparameter. Identiek bezwaar. | idem E3 |
| E5 | `risk/kelly.py:27-35` | Commentaar verwijst naar *"crypto alpha-decay studies"* en kalibreert de divisor op aangenomen μ-schattingsfout | De risicolaag is gekalibreerd op alpha-eigenschappen van één markt. | Documenteren; parameter naar `conf/risk/` |
| E6 | `schemas/config.py:249` | `RiskConfig.min_signal_confidence: Fraction = 0.55` | Een **alpha-parameter in het risicocontract**. `conf/risk/default.yaml:12` zet hem ook daadwerkelijk. Exit-criterium 2 eist letterlijk *"`risk/` bevat nul alpha-parameters"*. | **Verwijderen** uit `RiskConfig` en uit `conf/risk/default.yaml` |
| E7 | `schemas/config.py:250` | `RiskConfig.max_funding_cost_bps_day: 30.0` | Een carry-/kostendrempel (L9/L4), geen risicolimiet. | Verplaatsen naar `conf/execution/` of `conf/model/alpha.yaml` |
| E8 | `risk/portfolio.py:804-826` | `set_hmm_detector()` + HMM-regimecap op LONG-posities (`size_position:784-795`) | Een **marktregimemodel** (L2/L4) dat binnen de sizingbeslissing van L7 draait, inclusief een `import pandas` binnen de functie. | HMM verlaat het besluitpad; regime-informatie mag alleen als *gemeten marktstaat* binnenkomen, niet als model |
| E9 | `risk/portfolio.py:351` | `self._crisis_anchor_symbol = "BTCUSDT"` | **Symbool-specifieke uitzondering in L7.** Stap 3 verbiedt dit expliciet: *"de module kent geen symbool-specifieke uitzonderingen"*. De crisis-multiplier van het hele boek hangt aan de skew van één ticker. | Vervangen door een geconfigureerde, symbool-agnostische marktstaat-input |
| E10 | `risk/hmm_regime.py:41` | `from ..features.volatility import causal_expanding_std` | L7 → L3-import. Geen alpha, maar wel een laagoverschrijding; de audit merkt `hmm_regime.py` als REDESIGN aan (§24). | Buiten scope Phase 4 (§24 wijst dit naar Phase 6); registreren |

---

## 4. De vermenigvuldiging: dezelfde limiet, N implementaties

Dit is de reden dat "risico overruled alpha" vandaag niet aantoonbaar is — er
is geen *één* laag die overruled.

### 4.1 Drawdown-breakers — vier stuks

| Implementatie | Trigger | Resume | Peak-definitie | Persistent? |
|---|---|---|---|---|
| `risk/portfolio.py:678-706` `dd_state()` | `0.20` | `0.10` | rolling `dd_breaker_lookback_bars=5000` | ✗ in-memory |
| `risk/drawdown.py:39-53` `DrawdownBreaker` | `0.15` | `0.08` | all-time (`lookback_bars=0`) | ✗ in-memory |
| `risk/daily_loss_governor.py:106` `PropfirmGovernor` | uit `PropfirmLimits` | per dag | day-start balance + static DD | ✗ in-memory |
| `live/circuit_breaker.py:79` `CircuitBreaker` | `0.08` (+ `0.05` intraday) | handmatig | peak equity + sessiepiek | ✗ in-memory (`_persist_trip` logt, herstelt niet) |

Vier verschillende antwoorden op *"bij welke drawdown stoppen we?"* — 8%, 15%,
20%, plus een propfirm-variant. Geen ervan overleeft een procesherstart.
**Exit-criterium 5 wordt vandaag door geen enkele implementatie gehaald.**

### 4.2 Volatility targeting — drie stuks

| Implementatie | Formule | Cap | σ̂-bron | Gedrag bij ontbrekende σ̂ |
|---|---|---|---|---|
| `risk/portfolio.py:511-544` | `target / (realized × crisis_mult)` | `2.5`, of `1.5` in transitie | **realized** rolling vol, intern | `return 1.0` — **stille pass-through** |
| `risk/daily_loss_governor.py:236-255` | `(target × regime_mult) / realised` | `3.0` | caller | `return floor` (`0.0`) |
| `alpha/adaptive_wf.py:259` | `pnl × (target / bv)` | géén | `pnl.std() × √365` | `return pnl` — **stille pass-through** |

Geen van de drie gebruikt de ex-ante `σ̂_{t+1|t}` uit L2. Alle drie gebruiken
**realized** (backward-looking) vol. Twee van de drie degraderen stilzwijgend
naar "geen targeting" — exact het gedrag dat deliverable 1 verbiedt.

> **Kalibratienotitie voor stap 3.** `risk/portfolio.py:511` schaalt op
> *realized portfolio-vol*. De Phase 3-baseline draaide op **72%
> geannualiseerde vol** met een geometrische drag van 25 procentpunt
> (rekenkundig +11,3%, geometrisch −14,1%; `reports/BASELINE_BENCHMARK.md`).
> Bij `σ_target = 0.12` en `σ̂ = 0.72` geeft §14.1 `w_t = 0.167`. De bestaande
> cap van 2,5 bindt daar niet — hij bindt juist aan de verkeerde kant, want er
> is op dit moment geen enkele *ondergrens* op de geleverde exposure.

### 4.3 Positielimieten — vier stuks

| Implementatie | per-asset | gross | net | concentratie |
|---|---|---|---|---|
| `risk/position_limits.py:33-36` | `2.0` | `3.0` | `0.9` (imbalance) | `0.8` |
| `risk/portfolio.py:267-269` | `2.0` | `4.0` | `2.5` | — |
| `portfolio/constraints.py:35-38` | — | `1.0` (leverage) | — | `0.40` (max_weight) |
| `live/execution_controller.py:95` | notional | `inf` **(default)** | — | — |

`risk/position_limits.py` is een complete, geteste limietmodule — en wordt door
**geen enkele** productiecode aangeroepen. Alleen `tests/unit/test_risk.py`
importeert hem. Het is dode code die precies doet wat `risk/limits.py` moet
gaan doen.

---

## 5. Configuratiefragmentatie: dezelfde drempel, vier waarden

`max_gross_leverage`, gemeten in de boom op `18d232a`:

| Bron | Waarde | Bindt wanneer |
|---|---|---|
| `conf/risk/default.yaml:6` | **1.5** | alleen als `RiskConfig` daadwerkelijk gelezen wordt |
| `conf/env/prod.yaml:29` | **1.5** | in prod-profiel |
| `conf/conf_config.yaml:282` (namespace `portfolio:`) | **2.0** | wanneer `apps/backtest_portfolio.py` draait |
| `risk/portfolio.py:267` constructordefault | **4.0** | wanneer niets is meegegeven |
| `apps/backtest_portfolio.py:1014` `OmegaConf.select(..., default=4.0)` | **4.0** | wanneer de config-key ontbreekt |

Vijf plaatsen, vier waarden, factor 2,7 verschil tussen de strengste en de
ruimste. Hetzelfde patroon voor `dd_breaker_threshold`: `0.18`
(`conf_config.yaml:307`), `0.20` (`risk/portfolio.py:271`), **`0.35`**
(`apps/backtest_multi_alpha.py:520`), `0.15` (`conf/symbols/*.yaml`), tegenover
`max_drawdown_pct: 0.08` in `conf/risk/default.yaml:9`.

Twee structurele problemen, los van de getallen:

1. **De risicolimieten wonen in de `portfolio:`-namespace**, niet in `risk:`.
   L7-drempels worden bestuurd door L8-configuratie.
2. **Elke lezing heeft een `default=`-fallback.** Ontbreekt de key, dan valt de
   code stil terug op de *ruimste* waarde. De faseregels eisen het omgekeerde:
   ontbrekende limietconfiguratie moet crashen.

---

## 6. Ontmantelingsplan `risk/portfolio.py` (956 LOC → 0)

Per verantwoordelijkheidsgebied, in de volgorde waarin stap 7 ze verplaatst.
Elke rij is één atomaire commit.

| Regels | Verantwoordelijkheid | Bestemming | Wijziging |
|---|---|---|---|
| `69-238` | `_safe_corr`, `_dual_window_ewma_corr`, `_avg_offdiag`, `effective_n_assets` | `portfolio/` (covariantie is allocatie-input) | Verplaatsen. `ewma_lambda=0.94` op `:72` staat als *default-argument* — dit is de door **DI-10** aan Phase 4 toegewezen literal; hij komt uit `conf/model/volatility.yaml` |
| `511-606` | `vol_target_multiplier`, `_compute_crisis_multiplier` | `risk/vol_targeting.py` | **Herschrijven** op ex-ante `σ̂_{t+1\|t}` uit L2; crisis-anchor (`BTCUSDT`, E9) vervalt; fail-fast i.p.v. `return 1.0` |
| `607-677` | `correlation_multiplier` | `portfolio/` | Diversificatieschaling is een allocatiebeslissing (N_eff), geen harde limiet |
| `678-706` | `dd_state` | `risk/kill_switches.py` | **Herschrijven** op HWM + persistente `HALTED` |
| `707-746` | `size_position` stap 1-3 (DD, Kelly, CVaR) | gesplitst | DD → kill switches; **Kelly en CVaR-scaling verlaten het besluitpad** (E3/E4) |
| `747-783` | `size_position` stap 4-5 (per-asset, gross, net caps) | `risk/limits.py` | Volgorde expliciet uit `conf/risk/`, niet uit code-volgorde |
| `784-795` | HMM-regimecap | verwijderen uit L7 | E8 |
| `804-829` | `set_hmm_detector`, `update_cvar_estimates` | verwijderen | Injectiepunten voor alpha-kennis in L7 |
| `830-956` | `update`, `equity_metrics` | `risk/engine.py` | Wordt de `risk_state`-transitie van het contract |
| `37-64` | `RiskState`, `SizingDecision` | `risk/engine.py` | `SizingDecision.reason: str` (`:63`) wordt het **machineleesbare** `binding_constraints` |

**Consumenten die meebewegen** (in stap 7 te herwijzen):
`backtest/portfolio.py:14,41,63`, `apps/backtest_portfolio.py:903,1008`,
`apps/backtest_multi_alpha.py:447,527`, plus de ratchetregel
`scripts/check_hardcoded_params.py:150` (`"risk/portfolio.py": (15, "DI-10 Phase 4")`)
die na ontmanteling vervalt.

---

## 7. Wat al schoon is — niet opnieuw bouwen

| Component | Locatie | Status |
|---|---|---|
| AST-laagscheiding voor alpha | `alpha/base.py` (`FORBIDDEN_LAYERS`, `assert_alpha_isolation`) | **RETAIN** — `FORBIDDEN_LAYERS` bevat al `risk`; de test in `tests/unit/test_alpha_isolation.py:101,118` verwijst al naar `risk.limits` en `risk.vol_targeting` als verboden imports. De modulenamen uit deliverables 1-2 liggen dus al vast. |
| L4-contract `AlphaOutput` | `alpha/base.py` | **RETAIN** — bevat bewust geen positiegrootte, notional of hefboom |
| EWMA-vol-estimator | `volatility/ewma.py` + `conf/model/volatility.yaml` | **RETAIN** — λ=0.94 uit config, NaN-propagatie tijdens burn-in, geen impliciete constante vol. Dit is de σ̂-bron voor stap 3. |
| `PropfirmGovernor` | `risk/daily_loss_governor.py` | **RETAIN als basis** — dagverlies-logica is bruikbaar; mist alleen persistentie |
| `DrawdownBreaker` | `risk/drawdown.py` | **RETAIN als basis** — heeft al hysterese en `scale_factor` voor getrapte de-grossing; mist HWM-persistentie |
| `PositionLimits` | `risk/position_limits.py` | **RETAIN als basis** voor `risk/limits.py` — compleet, getest, ongebruikt |

> **Belangrijke beperking van de bestaande AST-scan.** `assert_alpha_isolation`
> vuurt via `__init_subclass__`, dus **uitsluitend** voor modules die een
> `AlphaUnit` subclassen. Dat zijn er op `18d232a` precies twee:
> `alpha/base.py` en `alpha/momentum.py`. De overige alpha-modules worden
> **niet** gescand. De statische helft van `test_risk_alpha_decoupling.py`
> (stap 9) moet daarom de **volledige** `alpha/`-boom scannen, niet op de
> bestaande hook leunen.

---

## 8. Werkplan stappen 2-11, met afhankelijkheden

| Stap | Levert | Hangt af van | Adresseert |
|---|---|---|---|
| 2 | Interfacecontract `(desired_exposure, market_state, risk_state) → (permitted_exposure, binding_constraints)` | — | Kernbevinding 4 |
| 3 | `risk/vol_targeting.py` | 2, `volatility/ewma.py` | A1, A2, §4.2 |
| 4 | `risk/limits.py` | 2, `risk/position_limits.py` | B1-B3, C3-C5, D1-D2, §4.3 |
| 5 | `risk/kill_switches.py` (persistente `HALTED`) | 2, `risk/drawdown.py`, `risk/daily_loss_governor.py` | A3, A4, C1, C2, §4.1 |
| 6 | `risk/engine.py` + machineleesbaar auditspoor | 3, 4, 5 | Kernbevinding 4 |
| 7 | `risk/portfolio.py` ontmanteld; `factor_alpha.py` → `alpha/` | 3, 4, 5, 6 | §6, E1, E2 |
| 8 | Alpha-units gezuiverd | 7 | A1-A5, E6, E7 |
| 9 | `test_risk_alpha_decoupling.py` | 7, 8 | §7-beperking |
| 10 | S1-S4 + variance-drag-analyse | 6, 7 | Exit 1, 6 |
| 11 | `RISK_STRESS_REPORT.md`, `phase4_exit_report.md`, `config_hash` in ledger | 10 | Exit 6, 7 |

**Volgorde-eis uit de faseregels:** stap 7 verloopt in atomaire commits per
verplaatst verantwoordelijkheidsgebied (§6-tabel), met de testsuite groen na
elke commit — gemeten tegen de baseline in §0.

---

## 9. Openstaande punten die Phase 4 *niet* sluit

1. **DI-12 (legacy alpha-units).** De vol-schaling in `momentum.py`,
   `csm_volume_clock.py` en `carry.py` blijft staan; 26 modules hangen aan de
   legacy-signatuur. Phase 4 bewijst de scheiding op het **contract**, niet op
   de gearchiveerde units. Audit §24 merkt die units al aan als **ARCHIVE**.
2. **`risk/hmm_regime.py` (E10).** Audit §24 merkt hem aan als REDESIGN richting
   M2 Filtered HMM, met bewijslast *"QLIKE / OOS Sharpe winst"* — dat is
   Phase 6. Phase 4 haalt hem uitsluitend uit het **besluitpad** van L7.
3. **Vier overlappende backtesters.** Audit §24 vraagt consolidatie tot één
   event-driven engine met pariteitstest. Buiten scope; Phase 4 wijst alleen de
   risicoaanroepen om.
4. **DI-15 survivorship bias.** Ongewijzigd open; elke stress-uitkomst in stap
   10 draait op zes ex-post gekozen overlevers en moet dat vermelden.

---

## 10. Samenvattende telling

| Categorie | Aantal locaties | Bestanden |
|---|---|---|
| Risicologica buiten `risk/` | **16** | 11 |
| Alpha-logica binnen `risk/` | **10** | 4 (incl. `schemas/config.py`) |
| Duplicate drawdown-breakers | **4** | 4 |
| Duplicate vol-targeting-implementaties | **3** | 3 |
| Duplicate positielimietcontroles | **4** | 4 |
| Bronnen voor `max_gross_leverage` | **5** (4 waarden) | 5 |
| Alpha-parameters in het risicocontract | **2** | `schemas/config.py`, `conf/risk/default.yaml` |
| Persistente `HALTED`-toestanden | **0** | — |
| Machineleesbare auditsporen | **0** | — |

De laatste twee rijen zijn de fase in één regel: er is veel risicologica, op
veel plaatsen, en geen enkele ervan is auditbaar of overleeft een herstart.
