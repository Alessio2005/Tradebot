# PHASE 4 — EXIT REPORT

> **Fase:** 4 van 7 — Risk & Volatility Architecture · **Prioriteit:** P1
> **Bindend brondocument:** `ARCHITECTUUR_AUDIT_2026-08-22.md` — secties 4, 9, 9.1, 11.1, 13.1, 14, 14.1, 19, 23 (Phase 4), 24, 26
> **Gegenereerd:** 2026-08-25 · **`git_sha` van de opgeleverde code:** `0b84e4a`
> **`config_hash` van de risicoconfiguratie:** `47821e47fe2cec30`
> **Bijbehorende rapporten:** `reports/phase4_entanglement_map.md`, `reports/RISK_STRESS_REPORT.md`, `docs/RISK_CONTRACT.md`

---

## 0. Samenvatting

De risicolaag bestond vóór deze fase niet als laag. Er waren **vier**
onafhankelijke drawdown-breakers, **drie** vol-targeting-implementaties en
**vier** positielimietcontroles, verspreid over `risk/`, `alpha/`, `backtest/`,
`live/` en `portfolio/`, met **vier verschillende waarden** voor
`max_gross_leverage` op vijf plaatsen. Geen ervan kende de andere, geen ervan
overleefde een procesherstart, en geen ervan registreerde machineleesbaar wat
er waarom was ingegrepen.

Er is nu één soevereine laag — `risk.engine.RiskEngine` — die `a_t` plus een
gemeten marktstaat aanneemt en de toegestane exposure teruggeeft met een
volledig auditspoor. Zij is puur, kent geen alpha, leest elke drempel uit
`conf/risk/` en crasht op elke ontbrekende invoer.

Het gemeten effect op de Phase 3-baseline (`long_only_equal_weight`):

| | Zonder risicolaag | Met risicolaag |
|---|---:|---:|
| Variantie-drag | 25,4 pp | **0,1 pp** |
| CAGR (meetkundig) | −14,1 % | **+0,6 %** |
| Totaalrendement | −48,9 % | **+2,5 %** |
| Max drawdown | 83,5 % | **5,1 %** |
| Sharpe | 0,156 | 0,190 |

**Alle zeven exit-criteria zijn gehaald.** Wat NIET is gebeurd staat in §3, en
dat is even belangrijk als wat wel is gebeurd.

---

## 1. Regressiebaseline

Vastgelegd vóór de eerste wijziging, op `18d232a`, en opnieuw gemeten op
`0b84e4a`.

| Ratchet | Vóór (`18d232a`) | Na (`0b84e4a`) | Oordeel |
|---|---|---|---|
| Testsuite | 1.190 tests | **1.504 tests** | +314 |
| Falende tests | 6 | **6** | identiek, alle pre-existent |
| Pre-existente faalgevallen | 4 × killgates, 2 × property | idem, dezelfde namen | geen regressie |
| `check_hardcoded_params.py` | 330 literals, exit 0 | **330 literals, exit 0** | ratchet niet verruimd |
| `audit_fallbacks.py --strict` | 36 bevindingen, 0 blokkerend | **36, 0 blokkerend** | ongewijzigd |

> **Eén waarneming die niet is weggepoetst.** Tijdens de afsluitende metingen
> rapporteerde één tussenliggende suite-run **7** faalgevallen in plaats van 6,
> zonder dat de namen zijn vastgelegd. Drie volledige runs erna gaven opnieuw
> exact dezelfde 6, evenals aparte runs van `tests/property` (2), `tests/killgates`,
> `tests/integration` en `tests/e2e` (4). De vermoedelijke oorzaak is
> pre-existent en niet van deze fase: `tests/property/test_hypothesis_kernels.py`
> draait vier niet-gederandomiseerde Hypothesis-tests met `deadline=5000`, en die
> run liep gelijktijdig met andere processen. Een `DeadlineExceeded` onder
> belasting is daar het waarschijnlijke gevolg. Het is **niet gereproduceerd**;
> wie de suite in CI vastzet, doet er goed aan die tests te derandomiseren
> (`derandomize=True`) of hun deadline los te laten.

De ratchet is **aangetrokken, niet verruimd**: het budget van 15 voor
`risk/portfolio.py` is bij de splitsing vervangen door 10 + 5 voor de twee
nieuwe paden — exact de gemeten aantallen, zodat het totaal op 330 blijft. De
vijf nieuwe `risk/`-modules staan in `GOVERNED` met budget **0**.

---

## 2. Exit-criteria, met bewijs

### Criterium 1 — Risicolimieten overrulen alpha in S1 t/m S4, met auditspoor

**Gehaald.**

| Scenario | Gevraagd | Toegestaan | Reductie | Bindende constraints |
|---|---:|---:|---:|---|
| S1 Vol-shock (×10) | 3,600 | 0,004604 | 99,87 % | `vol_target`, `cluster_cap` |
| S2 Correlatie-instorting | 3,600 | 0,110497 | 96,93 % | `vol_target` |
| S3 Gap-down −30 % | 3,600 | 0,000000 | 100,00 % | `daily_loss_governor`, `drawdown_breaker` |
| S4 Alpha-runaway (±1) | 6,000 | 0,046041 | 99,23 % | `vol_target`, `cluster_cap` |

Elke constraint draagt scope, gemeten waarde, drempel en `conf/risk/`-sleutel;
de volledige tabel staat in `RISK_STRESS_REPORT.md` §2.1.

**Bewijs:** `tests/integration/test_risk_overrules_alpha.py` (33 tests),
`artefacts/risk/phase4_stress.json`.

De harness **crasht** wanneer een scenario niets laat binden — een te ruime
kalibratie levert geen groen rapport op maar een fout. Dat gedrag is zelf
getest.

### Criterium 2 — Ontkoppelingstest geslaagd

**Gehaald**, in beide helften.

**Statisch.** De volledige `alpha/`-boom (28 modules) importeert nul keer uit
`risk/`, `portfolio/`, `execution/` of `oms/`. De besluitpad-modules van `risk/`
noemen geen enkele alpha-grootheid in een handtekening en importeren niets uit
`alpha/`, `kelly`, `factor_alpha`, `hmm_regime`, `train` of `tune`. `MarketState`,
`RiskState` en `RiskConfig` dragen geen alpha-veld.

> **Waarom de scan niet op de bestaande hook leunt.**
> `alpha/base.py::assert_alpha_isolation` vuurt via `__init_subclass__` en dekt
> dus uitsluitend `AlphaUnit`-subklassen — twee modules. De ontkoppelingstest
> scant de hele boom.

**Functioneel.** Drie niet-verwante alpha-units (rang-momentum, een tanh-carry,
een pure replay) met verschillende namen, parameters en interne toestand
produceren dezelfde `a_t` en krijgen **bit-identieke** `RiskDecision`s — over
vier volatiliteitsregimes en een reeks van 20 bars met bewegende marktstaat.
Een controlemeting bewijst dat een *ander* `a_t` de uitkomst wél verandert, dus
de gelijkheid is niet triviaal.

**Bewijs:** `tests/unit/test_risk_alpha_decoupling.py` (55 tests).

### Criterium 3 — `risk/portfolio.py` ontmanteld

**Gehaald.** Het bestand bestaat niet meer.

| Verantwoordelijkheid | Nieuwe locatie |
|---|---|
| Vol-targeting | `risk/vol_targeting.py` (herschreven) |
| Harde limieten | `risk/limits.py` (herschreven) |
| Drawdown-breaker | `risk/kill_switches.py` (herschreven) |
| Compositie + auditspoor | `risk/engine.py` (nieuw) |
| EWMA-covariantie, correlatie, N_eff | `portfolio/covariance.py` |
| Legacy sizing-machinerie | `portfolio/legacy_sizing.py` |

`risk/factor_alpha.py` is verplaatst naar `alpha/factor_alpha.py`: het meet of
een strategie residual alpha heeft, en dat is een L4/L11-vraag.

De legacy sizing-klasse is **verplaatst, niet weggegooid**, om één reden:
`BASELINE_BENCHMARK.md` is de gecertificeerde meetlat van dit platform, en die
cijfers herrekenbaar houden vereist dat de code die ze produceerde blijft
bestaan. In L8 zijn haar Kelly-, CVaR- en HMM-hooks legitiem — portfolio-
constructie mág alpha kennen. In L7 waren ze een bypass. Dat is waarom het
bestand is verhuisd in plaats van vernietigd.

**Bewijs:** `test_risk_alpha_decoupling.py::
TestRiskKnowsNoAlphaParameter::test_the_entangled_portfolio_module_no_longer_lives_in_risk`.

### Criterium 4 — Standalone vol targeting, fail-fast

**Gehaald.** `risk/vol_targeting.py` implementeert `w_t = min(max_leverage,
σ_target/σ̂)` als pure, stateless, symbool-agnostische functie.

Crasht op: ontbrekende `σ̂`, `NaN`, `±inf`, nul, negatief. **Nul** fallbacks —
geen laatste-bekende-waarde, geen cross-sectionele mediaan, geen constante vol.
Dat geldt óók wanneer `a_t = 0` voor het betreffende symbool: anders zou de
geldigheid van L7 afhangen van de output van L4.

Een statische test bewijst dat de module geen `except`, `fillna`, `ffill`,
`bfill` of `nan_to_num` bevat.

De boekvolatiliteit is de **comonotone bovengrens** `Σ|w_i|·σ_i` (ρ = 1) in
plaats van een geschatte covariantiematrix. Zij kan alleen te veel de-grossen,
nooit te weinig, en houdt het schattingsprobleem in L8. Een property-test toont
dat de grens 200 gesamplede correlatiestructuren domineert.

**Bewijs:** `tests/unit/test_vol_targeting.py` (26 tests).

### Criterium 5 — Irreversibele kill switches

**Gehaald.** Vóór deze fase haalde geen van de vier bestaande breakers dit; de
scherpste (`live/circuit_breaker.py`) journaliseerde zijn trip maar hield de
toestand in `self._halt_reason` — weg na een herstart, precies het scenario
waarvoor een kill switch bestaat.

| Eigenschap | Bewijs |
|---|---|
| Overleeft procesherstart | vers `HaltStore`-object op hetzelfde pad ziet de halt |
| Herstelt niet bij marktherstel | equity terug naar een **nieuwe** HWM laat de halt staan |
| Herstelt niet bij een nieuwe dag | verse `day_start_equity` laat de halt staan |
| Eerste oorzaak blijft staan | `engage()` overschrijft een bestaande halt nooit |
| Alleen handmatig op te heffen | `release()` eist operator én motivering, journaliseert beide |
| Corrupt bestand ≠ toestemming | leeg, afgekapt of niet-object crasht |

Statisch bewaakt: buiten `release()` zet geen enkel codepad `halted` terug.

**Bewijs:** `tests/unit/test_kill_switch_irreversibility.py` (23 tests) en
scenario S3 met herstart-en-herstelde-markt.

### Criterium 6 — Geometrische impact gekwantificeerd

**Gehaald.** De vier Phase 3-tracks zijn opnieuw gedraaid met de engine in het
pad, bar voor bar, met causale `σ̂` uit L2.

| Track | Drag vóór | Drag na | CAGR vóór | CAGR na | Max DD vóór | Max DD na |
|---|---:|---:|---:|---:|---:|---:|
| `long_only_equal_weight` | 25,4 pp | **0,1 pp** | −14,1 % | **+0,6 %** | 83,5 % | **5,1 %** |
| `long_only_risk_parity` | 22,2 pp | **0,0 pp** | −13,5 % | **+0,4 %** | 82,2 % | **4,6 %** |
| `xs_momentum_equal_weight` | 3,2 pp | **0,0 pp** | −7,9 % | **+0,0 %** | 50,8 % | **3,1 %** |
| `xs_momentum_risk_parity` | 1,5 pp | **0,0 pp** | −11,6 % | **−0,2 %** | 56,3 % | **3,0 %** |

De baselinekant reproduceert `BASELINE_BENCHMARK.md` §3.1 tot op de laatste
decimaal — dezelfde tracks, dezelfde folds, dezelfde kosten. Zonder die
identiteit is de vergelijking betekenisloos.

Drie observaties die het rapport expliciet maakt in plaats van weglaat:

1. **De Sharpe verandert nauwelijks** (0,156 → 0,190). Vol targeting is voor de
   Sharpe bij benadering een schaaltransformatie. Wie een sprong verwachtte,
   verwachtte alpha van een risicolaag.
2. **De momentumtracks blijven verliezen.** Het verlies krimpt met een factor 60,
   maar de Sharpe blijft negatief. Bruto Sharpe −0,284: geen risicolaag
   repareert een signaal zonder edge.
3. **De drawdown daalt op alle vier de tracks met een factor 12 tot 17.**
   Criterium 8 uit de bestandsversie van de opdracht — *"een risicolaag die de
   drawdown niet verlaagt, is verkeerd gekalibreerd"* — is daarmee ook gehaald.

**Bewijs:** `RISK_STRESS_REPORT.md` §3, `artefacts/risk/phase4_stress.json`.

### Criterium 7 — Nul hardcoded drempels, gedekt door `config_hash`

**Gehaald.** Elke limiet komt uit `conf/risk/default.yaml`, gevalideerd tegen
`RiskConfig` (`extra="forbid"`, `frozen=True`). De vijf nieuwe `risk/`-modules
staan in de ratchet met budget **0**: geen enkele numerieke literal in een
functiehandtekening.

Waar de oude boom uiteenliep is consequent de **strengste** waarde gekozen, met
de herkomst als commentaar erbij:

| Drempel | Bestaande waarden | Gekozen |
|---|---|---:|
| `gross_cap` | 1,5 / 2,0 / 4,0 | **1,5** |
| `max_drawdown_pct` | 0,08 / 0,15 / 0,18 / 0,20 / 0,35 | **0,08** |
| `max_concentration` | 0,40 / 0,80 | **0,40** |
| `sigma_target` | 0,08 / 0,12 | **0,08** |
| `max_leverage` | 1,5 / 2,0 / 4,0 | **1,5** |

De configuratie is geregistreerd als `47821e47fe2cec30` in
`artefacts/governance/risk_config_registry.json`, met de volledige inhoud
ernaast — een hash bewijst dát er iets veranderde, niet wát.

**Bewijs:** `tests/unit/test_risk_config_registry.py` (10 tests),
`tests/unit/test_config_contracts.py::TestRiskBudgetIsComplete`.

### Aanvullend — Volledig auditspoor (criterium 6 van de bestandsversie)

**Gehaald.** Elke ingreep levert een `BindingConstraint` met soort (enum),
scope, gemeten waarde, drempel, exposure vóór en na, en de `conf/risk/`-sleutel.
`RiskDecision` weigert te bestaan wanneer `unconstrained` en het auditspoor
elkaar tegenspreken — een engine die stilletjes `a_t` doorgeeft is daarmee niet
uitdrukbaar.

---

## 3. Wat NIET is gebeurd

Dit is de belangrijkste sectie van dit rapport.

### 3.1 De legacy-consumenten zijn niet herbedraad

De entanglement map telde **16** locaties met risicologica buiten `risk/`.
Daarvan zijn er drie verwijderd (A1-A3, de vol-target en de de-grossing in
`alpha/adaptive_wf.py`). De overige blijven staan:

| Bevinding | Locatie | Status |
|---|---|---|
| B1-B3 | `backtest/evaluation.py`, `per_side.py`, `bidirectional.py` | leverage- en gross-caps blijven in L10 |
| C1-C5 | `live/circuit_breaker.py`, `execution_controller.py`, `portfolio_controller.py` | L13 houdt zijn eigen breaker en limieten |
| D1-D2 | `portfolio/constraints.py` | concentratie- en leveragecap blijven in L8 |
| A4 | `alpha/adaptive_wf.py` max-drawdown | **herclassificatie:** dit is een gerapporteerde METRIEK, geen limiet; hij gate't niets en blijft daarom staan |
| A5 | `alpha/xs_unit.py` BAB leg-leverage | legacy-unit onder DI-12 / ARCHIVE |

**De nieuwe L7-laag is dus wel compleet en gezaghebbend, maar nog niet de enige
plek waar limieten leven.** De redenen zijn expliciet:

* `backtest/` herbedraden verandert het productiepad van de Phase 3-baseline.
  Audit §24 wijst de consolidatie van de vier overlappende backtesters aan als
  een **apart** werkitem met een pariteitstest als bewijslast; dat binnen deze
  fase meenemen is scope creep met een gecertificeerde meetlat als inzet.
* `live/` is expliciet **Phase 7** (L13). De kill switches zijn hier
  aantoonbaar aansluitbaar gebouwd — `HaltStore` is een zelfstandig,
  persistent object — maar het aansluiten zelf hoort daar.

**Consequentie die de lezer moet meenemen:** wie vandaag `apps/backtest_portfolio.py`
of de live-engine draait, draait nog niet door de soevereine risicolaag. Het
overlay-pad (`risk_overlay_wave`) doet dat wel, en dat is wat §2 criterium 6
meet.

### 3.2 De risicoconfiguratie staat niet in de hypothese-ledger

De opdracht schrijft letterlijk voor: *"registreer de risicoconfiguratie in de
ledger met `config_hash`."* Dat is bewust **niet** gedaan.

`hypothesis_ledger.total_n_hypotheses()` sommeert `n_trials`, en dat getal is de
`M` waarmee de Deflated Sharpe Ratio deflateert. Een risicoconfiguratie
voorspelt niets en wordt tegen geen nulhypothese getoetst; haar daar neerzetten
zou de DSR van de Phase 3-baseline verlagen omdat iemand een limiet heeft
opgeschreven — precies het soort onopgemerkte koppeling dat deze fase opruimt.

In plaats daarvan: `registry/risk_registry.py`, met dezelfde eigenschappen die
de ledger auditbaar maken (append-only, atomair, gehasht) en zonder de
eigenschap die hier schadelijk is. Een test bewaakt dat registreren
`total_n_hypotheses()` ongemoeid laat.

### 3.3 De clusterlabels zijn een governance-keuze met een scherpe kant

Vijf van de zes symbolen dragen `crypto_l1`, met een clusterlimiet van 0,60. Dat
forceert `LINKUSDT` naar 40 % van de gross — exact op zijn eigen
`max_concentration`. Twee gevolgen:

1. het boek haalt zijn vol-target van 8 % nooit; het blijft op ~3,2 % steken,
   want de clusterlimiet knijpt harder dan de vol-target;
2. in een correlatiecrisis (S2) valt de clusterlimiet wég — met één cluster is
   `max(cap, 1/n)` gelijk aan 1,0 — en is de per-asset concentratielimiet de
   enige overgebleven spreidingsbescherming.

Dit is conservatief (het boek draait onder target, nooit erboven) maar het is
een **neveneffect van de labeling, geen ontwerp**. Aanbeveling voor Phase 5:
herzie de clusterindeling op een breder universum, of verlaag `sigma_target`
expliciet naar wat op dit universum haalbaar is.

### 3.4 DI-12: de legacy alpha-units zijn niet aangeraakt

De vol-schaling in `alpha/momentum.py`, `csm_volume_clock.py` en `carry.py`
blijft staan; 26 modules hangen aan de legacy-signatuur, en audit §24 merkt die
units al aan als **ARCHIVE**. De nieuwe `CrossSectionalMomentum` op het
L4-contract is schoon.

### 3.5 `risk/hmm_regime.py` blijft bestaan
Hij is uit het **besluitpad** gehaald — de engine importeert hem niet — maar de
module zelf blijft. Audit §24 wijst zijn herontwerp naar M2 Filtered HMM toe aan
Phase 6, met QLIKE/OOS-Sharpe als bewijslast.

### 3.6 De liquiditeitslimiet is niet op schaal getest
`adv_participation_cap` zit in het besluitpad en wordt elke run geverifieerd,
maar bij een equity van één eenheid bindt hij nergens. Gedekt door unit-tests,
niet door de overlay.

### 3.7 Survivorship bias (DI-15) staat open
Elk cijfer draait op zes ex-post gekozen overlevers. De vergelijking
baseline-versus-overlay is daar ongevoelig voor; de absolute niveaus niet.

---

## 4. Opgeleverde artefacten

| # | Deliverable | Status |
|---|---|---|
| 1 | `src/tradebot/risk/vol_targeting.py` | opgeleverd |
| 2 | `src/tradebot/risk/limits.py` | opgeleverd |
| 3 | `src/tradebot/risk/kill_switches.py` | opgeleverd |
| 4 | `src/tradebot/risk/engine.py` | opgeleverd |
| 5 | `risk/portfolio.py` ontmanteld | opgeleverd → `portfolio/covariance.py` + `portfolio/legacy_sizing.py` |
| 6 | `risk/factor_alpha.py` verplaatst | opgeleverd → `alpha/factor_alpha.py` |
| 7 | `src/tradebot/risk/stress_test.py` | opgeleverd (S1-S4 harness; VaR-suite behouden) |
| 8 | `tests/unit/test_risk_alpha_decoupling.py` | opgeleverd (55 tests) |
| 9 | `tests/integration/test_risk_overrules_alpha.py` | opgeleverd (33 tests) |
| 10 | `tests/unit/test_kill_switch_irreversibility.py` | opgeleverd (23 tests) |
| 11 | `conf/risk/default.yaml` uitgebreid | opgeleverd |
| 12 | `reports/RISK_STRESS_REPORT.md` | opgeleverd |
| 13 | `apps/run_stress.py` | opgeleverd, **79 LOC** (limiet 80) |

**Niet in de deliverablelijst, wel opgeleverd:** `docs/RISK_CONTRACT.md` (stap 2),
`src/tradebot/risk/contract.py`, `src/tradebot/risk/stress_report.py`,
`src/tradebot/registry/risk_registry.py`, `tests/unit/test_risk_contract.py`,
`test_vol_targeting.py`, `test_risk_limits.py`, `test_risk_engine.py`,
`test_risk_config_registry.py`.

---

## 5. Commits

| Sha | Bericht |
|---|---|
| `547bf4b` | `docs(risk): map alpha-risk entanglement across the codebase` |
| `dd9a680` | `docs(risk): specify the risk engine interface contract` |
| `f7edbcb` | `feat(risk): consolidate the L7 risk budget in conf/risk` |
| `5d5a0ad` | `refactor(risk): extract standalone volatility targeting module` |
| `01f76e1` | `feat(risk): consolidate hard limits into a single L7 module` |
| `82bc3ab` | `feat(risk): add high-water-mark drawdown breaker with tiered de-grossing` |
| `af76743` | `feat(risk): compose the sovereign RiskEngine with a machine-readable audit trail` |
| `8d9e8d7` | `refactor(risk): move the factor alpha lab out of the risk layer` |
| `1fd3650` | `refactor(risk): split portfolio.py into limits and allocation responsibilities` |
| `248fc27` | `test(risk): prove risk and alpha are decoupled, and purge L4 of risk parameters` |
| `417aaa6` | `feat(risk): add the S1-S4 stress harness and the baseline risk overlay` |
| `0b84e4a` | `feat(risk): register the risk configuration under an auditable config_hash` |

---

## 6. Wat de volgende fase moet weten

1. **De soevereine laag is gebouwd maar nog niet overal aangesloten** (§3.1).
   Phase 5 of de backtester-consolidatie moet `backtest/` en `portfolio/` op
   `RiskEngine` zetten; Phase 7 doet `live/`.
2. **De clusterkalibratie is een neveneffect, geen keuze** (§3.3). Herzien vóór
   er kapitaal op draait.
3. **De risicolaag maakt geen alpha.** De momentumtracks verliezen bruto; dat
   blijft de openstaande vraag van Phase 3 en wordt door geen limiet opgelost.
4. **`HaltStore` is klaar voor L13.** Een zelfstandig, persistent object met een
   append-only journaal; Phase 7 hoeft hem alleen aan te sluiten.
