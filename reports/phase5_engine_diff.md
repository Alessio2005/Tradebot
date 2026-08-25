# PHASE 5 — FORENSIC COMPARISON OF THE LEGACY BACKTEST ENGINES

> **Status:** forensische discovery, geschreven **vóór elke codewijziging** van
> Phase 5. Beschrijft de repository op `b206895`.

**Gegenereerd:** 2026-08-25
**git_sha:** `b206895`
**Bindend brondocument:** `docs/ARCHITECTUUR_AUDIT_2026-08-22.md` §16, §16.1, §24
**Fase-opdracht:** §14 (Legacy Engine Consolidation)

---

## 0. Correctie op de premisse van de opdracht

De fase-opdracht §14 noemt vier bestanden als "de vier legacy backtesters":
`evaluation.py`, `portfolio.py`, `bidirectional.py`, `per_side.py`.

Na inspectie is dat niet de feitelijke structuur. Er zijn **geen vier
concurrerende engines**. Er zijn **drie engines, één gedeelde hulpbibliotheek
en één volledig losstaand vijfde pad** dat de opdracht niet noemt:

| Bestand | LOC | Wat het feitelijk is |
|---|---:|---|
| `backtest/evaluation.py` | 1.054 | **Geen engine.** Hulpbibliotheek: DSR, Kelly+CVaR-sizing, block-bootstrap, timeout-monitor, long/short-collision-resolver, bar-by-bar MTM. Geconsumeerd door `per_side` en `bidirectional`. |
| `backtest/per_side.py` | 394 | **Engine 1.** Eén symbool, één zijde (`internal_backtest`). |
| `backtest/bidirectional.py` | 516 | **Engine 2.** Eén symbool, long én short tegelijk (`bidirectional_backtest`). Superset van engine 1. |
| `backtest/portfolio.py` | 664 | **Engine 3.** Multi-asset portefeuillelaag (`PortfolioBacktester`) die per-asset `AssetTrack`-objecten van engine 1 of 2 consumeert. |
| `backtest/baseline_runner.py` | 298 | **Engine 4 — door de opdracht niet genoemd.** Vectorized, gewichten-gebaseerd, produceert de Phase 3-baseline. Deelt geen enkele regel code met 1–3. |

De keten in productie is:

```
apps/backtest_portfolio.py
    → bidirectional_backtest()      per symbool   (engine 2)
    → AssetTrack                     per symbool
    → PortfolioBacktester.run()      over symbolen (engine 3)
```

en daarnaast, volledig gescheiden:

```
apps/run_baseline.py
    → backtest/baseline_report.py
    → build_weight_tracks() + run_baseline_tracks()   (engine 4)
```

`per_side.py` (engine 1) heeft **nul productie-aanroepsites**. Enige verwijzing
buiten zichzelf: `tests/test_imports.py` en `scripts/check_hardcoded_params.py`.

Dit verandert de consolidatieopdracht wezenlijk. Zie §5.

---

## 1. Verschillenmatrix

Classificatie per fase-opdracht §14: `identiek gedrag` · `feature` ·
`legacy bug` · `accounting difference` · `execution difference` ·
`intentional semantic difference`.

### 1.1 Beslissingsdomein

| Aspect | Engine 1 `per_side` | Engine 2 `bidirectional` | Engine 3 `portfolio` | Engine 4 `baseline_runner` | Klasse |
|---|---|---|---|---|---|
| Eenheid | 1 symbool, 1 zijde | 1 symbool, 2 zijden | N symbolen | N symbolen | **feature** |
| Signaalbron | ML-ensemble + meta-label | idem, 2 ensembles | vooraf gebouwde tracks | `AlphaUnit` op `a_t ∈ [-1,1]` | **intentional semantic difference** |
| Positieconcept | trade (entry→barrier) | trade | doorlopende exposure | doorlopend gewicht | **accounting difference** |
| Tijdas | event-bars (CUSUM) | event-bars (CUSUM) | bar-grid | bar-grid | **execution difference** |
| Long/short-conflict | n.v.t. | `resolve_long_short_collision` (delta-neutraal netten) | n.v.t. | n.v.t. | **feature** |

### 1.2 Accounting

| Aspect | Engine 1 | Engine 2 | Engine 3 | Engine 4 | Klasse |
|---|---|---|---|---|---|
| Equity-model | `account_size × Π(1+r)` per trade | idem | `equity ×= (1 + r − kosten)` per bar | `Π(1 + net_r)` per bar | **accounting difference** |
| Cash-register | **geen** | **geen** | **geen** | **geen** | **legacy bug** t.o.v. audit §16.1 (*"Explicit cash/position accounting registers"*) |
| Positie-register in units | **geen** | **geen** | **geen** | **geen** | **legacy bug**, idem |
| Balansidentiteit gecontroleerd | **nee** | **nee** | **nee** | **nee** | **legacy bug** — §10 van de fase-opdracht eist `assets == liabilities + equity` |
| Funding | via `mtm` helper | via `mtm` helper | expliciet, mét ±2 % cap en Δt-correctie | **niet gemodelleerd** | **accounting difference** |
| Fees | `rebalance_cost_bps`, default **0.0** | idem, default **0.0** | `AssetTrack.cost_bps`, default **5.0** | `taker_fee_bps` uit `conf/` | **legacy bug** — een default van 0.0 is de nul-fee fallback die §23 verbiedt |
| Herbalanceerkosten | op MTM-delta | op MTM-delta | op `|Δ signed leverage|` | op turnover | **accounting difference** |

### 1.3 Executieregime

| Aspect | Engine 1 | Engine 2 | Engine 3 | Engine 4 | Klasse |
|---|---|---|---|---|---|
| Fill-timing | volgende bar (`idx+1`) | volgende bar (`idx+1`) | zelfde bar als exposure | `weights.shift(1)` | **execution difference** |
| Spread — mid-data | `compute_dynamic_spread_arr`, fallback `0.0010` | idem | in `cost_bps` verwerkt | `assumed_half_spread_bps` uit `conf/` | **execution difference** |
| Spread — bid/ask aanwezig | **`spread_val = 0.0`** | **`spread_val = 0.0`** | n.v.t. | n.v.t. | **intentional semantic difference, ongetest** — zie §2.1 |
| Latency | geen expliciet model (impliciet 1 bar) | idem | **geen** | **geen** (`shift(1)`) | **execution difference** |
| Partial fills | **nee** | **nee** | **nee** | **nee** | ontbrekend |
| Orderboekdiepte | proxy via `taker_buy/sell_volume` | idem | **nee** | **nee** | **execution difference** |
| Marktimpact | `impact_k=0.5` in spread | idem | **niet toegepast** | **niet toegepast** | **legacy bug** — twee engines rekenen impact, twee niet |
| Order-object | **geen** | **geen** | **geen** | **geen** | ontbrekend |

### 1.4 Risicoregime

| Aspect | Engine 1 | Engine 2 | Engine 3 | Engine 4 | Klasse |
|---|---|---|---|---|---|
| Per-asset cap | `max_leverage=2.0` | `max_leverage=2.0` | `per_asset_cap=2.0` | geen | **legacy duplicate** (wiring audit B1/B3/B5) |
| Gross cap | `max_portfolio_leverage=None` | idem | via `PortfolioRiskManager` | `gross_target` (conventie) | **legacy duplicate** (B2) |
| Vol-targeting | **geen** | **geen** | `vol_target_multiplier` | **geen** | **legacy duplicate** (D6) |
| Drawdown-breaker | **geen** | **geen** | `dd_state()` | **geen** | **legacy duplicate** (D7) |
| Correlatie-de-grossing | **geen** | **geen** | `correlation_multiplier` | **geen** | **legacy duplicate** (D6) |
| Sovereign `RiskEngine` | **nee** | **nee** | **nee** | **nee** | het onderwerp van deze fase |

### 1.5 Statistiek

| Aspect | Engine 1 | Engine 2 | Engine 3 | Engine 4 | Klasse |
|---|---|---|---|---|---|
| Sharpe | `compute_annualised_sharpe` (dagbuckets, √365,25) | idem | `metrics.sharpe_ratio` | `metrics.sharpe_ratio` | **accounting difference** — twee verschillende Sharpe-definities in één repo |
| DSR `M` | via `evaluation.count_git_commits` | idem | `total_n_hypotheses` óf `N_assets × 2` **fallback** | `M` uit de bevroren pre-registratie | **legacy bug** — de commit-count als `M` is geen hypothesetelling |
| Walk-forward / embargo | **geen** | **geen** | **geen** | Purged WF met embargo | **legacy bug** |

---

## 2. Bevindingen die apart benoemd moeten worden

### 2.1 Bid/ask-data zet de spread op nul

`per_side.py:97-101` en `bidirectional.py:98-101`:

```python
if "bid_close" in df_test.columns and "ask_close" in df_test.columns:
    spread_val = 0.0000
```

`per_side.py:97` licht toe: *"Voor bid/ask data zijn de kosten al in de barriers
verwerkt"*. Dat is een verdedigbare conventie — de Triple-Barrier-labels zouden
dan al op bid/ask-prijzen zijn gezet — maar:

* er is **geen test** die aantoont dat de barriers dat feitelijk doen;
* `bidirectional.py` draagt de toelichting niet, alleen de code;
* het effect is dat **betere data een goedkopere fill oplevert**, wat de
  omgekeerde richting is van wat execution realism voorschrijft.

Klasse: **intentional semantic difference, ongetest**. Actie: op het huidige
universum is er geen bid/ask-data (zie `docs/DATA_REGISTER.md` §6), dus deze tak
is dode code. Hij verdwijnt met de engines.

### 2.2 De fee-default is nul

`rebalance_cost_bps: float = 0.0` in beide single-symbol engines. Een caller die
het argument vergeet, backtest kosteloos. `apps/backtest_portfolio.py` zet hem
wél, maar de default is de fallback die §23 expliciet verbiedt.

Klasse: **legacy bug**.

### 2.3 `M` voor de Deflated Sharpe komt uit `git rev-list --count`

`evaluation.py:42 count_git_commits()` telt commits en gebruikt dat getal als
proxy voor het aantal beproefde hypothesen. Dat is geen hypothesetelling: het
telt documentatiecommits mee en mist trials die nooit zijn gecommit. De
persistente `hypothesis_ledger` (engine 4) is de correcte bron.

Klasse: **legacy bug**.

### 2.4 `portfolio.py` heeft een `N_assets × 2`-ondergrens voor `M`

`portfolio.py:500-510` valt terug op `N_assets × 2` wanneer
`total_n_hypotheses` niet is meegegeven. Een stille, te lage `M` maakt de DSR
te gunstig.

Klasse: **legacy bug**.

### 2.5 Twee Sharpe-definities

`execution/spread.py::compute_annualised_sharpe` bucketeert onregelmatige
trades naar dagen en annualiseert met √365,25. `backtest/metrics.py::sharpe_ratio`
annualiseert met `bars_per_year` uit `conf/backtest/`. De twee getallen zijn niet
vergelijkbaar en dragen in de rapporten dezelfde naam.

Klasse: **accounting difference**.

### 2.6 Engine 1 is dood

`per_side.internal_backtest` wordt door geen enkele productie- of testroute
aangeroepen. Enige alias: `run_per_side_backtest`, aanwezig voor
`tests/test_imports.py`.

Klasse: **legacy duplicate**. Kan verdwijnen zonder pariteitsbewijs tegen een
opvolger, omdat er geen gedrag is dat behouden moet blijven — maar wél mét een
repository-wide referentiecontrole.

---

## 3. Wat pariteit hier kan en niet kan betekenen

De fase-opdracht §14 eist een pariteitsbewijs vóór verwijdering. Dat is
uitvoerbaar, maar de vorm moet kloppen met wat de engines zijn:

| Paar | Pariteit zinvol? | Vorm van het bewijs |
|---|---|---|
| Engine 1 ↔ Engine 2 | **ja** | `bidirectional_backtest` met een uitgeschakeld short-ensemble moet `internal_backtest` op dezelfde inputs reproduceren. Beide draaien op dezelfde helpers uit `evaluation.py`. |
| Engine 2 ↔ nieuwe event-driven engine | **gedeeltelijk** | De nieuwe engine kent geen ML-ensemble en geen Triple-Barrier-trades. Pariteit kan alleen op een **gedeelde, gereduceerde casus**: een vaste exposurereeks, dezelfde kosten, dezelfde bars. |
| Engine 3 ↔ nieuwe engine | **ja** | Beide zijn multi-asset en bar-grid. Pariteit op een vaste gewichtenreeks met kosten uit, dan met kosten aan. |
| Engine 4 ↔ nieuwe engine | **ja, en dit is de belangrijkste** | De Phase 3-baseline moet reproduceerbaar zijn op de nieuwe engine vóórdat de nieuwe kostenlagen erbij komen. Zonder deze stap is elk verschil in §20 niet toe te wijzen aan realism of aan de engine. |

**Het pariteitsbewijs is dus niet "vier engines geven hetzelfde getal".** Dat is
onmogelijk: engine 1/2 beslissen op trades, 3/4 op exposures. Het bewijs is
paarsgewijs en op expliciet gereduceerde casussen.

---

## 4. Wat de authoritative engine moet kunnen dat geen van de vier kan

| Eis (fase-opdracht) | Engine 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| §9 strikte event-volgorde market→signal→**risk**→portfolio→order→fill→accounting | ✗ | ✗ | ✗ | ✗ |
| §10 dubbele boekhouding, sluitende balans | ✗ | ✗ | ✗ | ✗ |
| §11 impact met verplichte η (`ConfigContractError` bij ontbreken) | ✗ | ✗ | ✗ | ✗ |
| §12 order-object, partial fills, cancellations | ✗ | ✗ | ✗ | ✗ |
| §13 TCA-roundtrip die sluit | ✗ | ✗ | ✗ | ✗ |
| §16 sovereign wiring | ✗ | ✗ | ✗ | ✗ |

Geen enkele bestaande engine is uitbreidbaar tot dit zonder herbouw van zijn
kern. De nieuwe `backtest/engine.py` is daarom een nieuwbouw, geen refactor —
maar hij erft aantoonbaar de conventies die getest en correct zijn:
`shift(1)`-timing (engine 4), funding-Δt en de funding-cap (engine 3), en de
kostenconventie op turnover (engine 4).

---

## 5. Consolidatievoorstel

| Bestand | Bestemming | Voorwaarde |
|---|---|---|
| `backtest/per_side.py` | **verwijderen** | referentiecontrole; geen pariteitsbewijs nodig (dode code, §2.6) |
| `backtest/bidirectional.py` | **verwijderen** | pariteit engine 1↔2 + engine 2↔nieuw op gereduceerde casus |
| `backtest/portfolio.py` | **verwijderen** | pariteit engine 3↔nieuw op vaste gewichtenreeks |
| `backtest/evaluation.py` | **splitsen** | statistiek (DSR, bootstrap, MTM) blijft als bibliotheek; Kelly-sizing en de leverage-caps verdwijnen naar `risk/` resp. verdwijnen |
| `backtest/baseline_runner.py` | **behouden** als `vectorized.py`-pad | moet `NOT_ADMISSIBLE_AS_PROMOTION_EVIDENCE` gaan dragen (§15) |
| `backtest/engine.py` | **nieuw** | de enige authoritative engine |
| `backtest/accounting.py` | **nieuw** | dubbele boekhouding, vóór de engine gebouwd (§10) |

---

## 6. Wat dit rapport niet vaststelt

Of de nieuwe engine dezelfde getallen produceert. Dat is
`tests/integration/test_engine_parity.py` en `reports/phase5_exit_report.md` §20.
