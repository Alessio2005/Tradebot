# MASTER-PROMPT: PHASE 4 — RISK & VOLATILITY ARCHITECTURE

> **Fase:** 4 van 7 · **Prioriteit:** P1
> **Bindend brondocument:** `ARCHITECTUUR_AUDIT_2026-08-22.md` — secties 9, 9.1, 11.1, 13.1, 14, 14.1, 19, 23 (Phase 4), 24, 26
> **Voorwaarde:** Phase 0 t/m 3 volledig afgerond. Zonder gekwantificeerde baseline is er geen referentie voor risico-impact.

---

## ROL EN CONTEXT

Je acteert als **Head of Risk Architecture / Risk Systems Engineer**. In dit platform is risico geen filter achteraf, maar het soevereine controle-orgaan. Jij bouwt de laag die alpha kan overrulen, en jij bouwt hem zo dat alpha hem nooit kan omzeilen, uitschakelen of beïnvloeden.

Je werkt uitsluitend binnen de kaders van het bindende auditdocument `ARCHITECTUUR_AUDIT_2026-08-22.md`.

**Relevante lagen uit de Target Architecture (sectie 19):**

| Laag | Naam | Rol in deze fase |
|---|---|---|
| **L2** | Volatility Engines (EWMA / GARCH / HAR) | Levert de ex-ante `σ̂_{t+1|t}`; in productie EWMA (λ=0.94) |
| **L4** | Alpha Generation | **Wordt ontkoppeld.** Alpha levert alleen `a_t`, verder niets |
| **L7** | Independent Risk & Volatility Targeting | **Primaire laag.** Volledig te bouwen als soevereine module |
| **L8** | Portfolio Construction & Sizing | Consument van de door risk toegestane exposure |
| **L13** | Live Operations & Monitoring | Kill switches moeten in Phase 7 rechtstreeks aansluitbaar zijn |

> **Doctrine (sectie 14):** *"Risicobeheer is het soevereine controle-orgaan van het handelssysteem. **Risico overruled altijd Alpha.**"*

**Kritieke bevindingen die deze fase adresseert:**
- **Sectie 4:** *"Risicolimieten en volatiliteitstargeting bevinden zich gedeeltelijk binnen de alpha-units en feature-pipelines, wat een strikte scheiding van verantwoordelijkheden onmogelijk maakt."*
- **Sectie 24 — REDESIGN:** `risk/portfolio.py` moet gesplitst worden: risicobeperking ontkoppeld van alpha. Vereist bewijs: **een geslaagde ontkoppelingstest**.
- **Symptomatisch:** de aanwezigheid van `risk/factor_alpha.py` binnen de risk-module is zelf een instantie van de verstrengeling. Alpha-logica hoort in L4, nooit in L7.

**Bindende risicolagen (sectie 14.1):**

Unconditional Volatility Targeting:

    w_t = min( MaxLeverage , σ_target / σ̂_{t+1|t} )

Hard Limits: Max Concentration Cap · Gross & Net Exposure Caps · Drawdown Breaker (High-Water Mark) · Daily Loss Governor (kill switch).

---

## DOEL VAN DE FASE

Volledige, aantoonbare ontkoppeling van Risk (L7) en Alpha (L4), en oplevering van een standalone risicolaag die in stress-simulaties bewijsbaar alpha-posities overrulet.

Na deze fase kan geen enkele alpha-unit meer een positiegrootte, hefboom of limiet beïnvloeden. De risicolaag is de laatste instantie vóór portfolio-constructie, kent zijn eigen ex-ante volatiliteitsschatting, en heeft een kill switch die onder alle omstandigheden vuurt.

---

## CONCRETE DELIVERABLES

1. **`src/tradebot/risk/vol_targeting.py`** — standalone, stateless volatility targeting: `w_t = min(MaxLeverage, σ_target / σ̂_{t+1|t})`. `σ_target` en `MaxLeverage` uit `conf/risk/`. Bij een ontbrekende of niet-eindige `σ̂` crasht de module — geen laatste-bekende-waarde, geen constante vol.
2. **`src/tradebot/risk/limits.py`** — geconsolideerde harde limieten:
   - Max Concentration Cap (per asset en per sector/cluster)
   - Gross Exposure Cap en Net Exposure Cap
   - Per-asset positielimiet, afgeleid van liquiditeit (ADV-fractie)
3. **`src/tradebot/risk/kill_switches.py`** — Drawdown Breaker op High-Water Mark met getrapte de-grossing, en de Daily Loss Governor als onmiddellijke kill switch. Beide met expliciete, geconfigureerde drempels en een onherroepelijke `HALTED`-toestand die alleen handmatig wordt opgeheven.
4. **`src/tradebot/risk/engine.py`** — de soevereine `RiskEngine`: neemt `a_t` (gewenste exposure) plus marktstaat, geeft `permitted_exposure` terug plus een **volledig, machineleesbaar auditspoor** van elke binding constraint. De engine geeft nooit `a_t` ongewijzigd door zonder expliciete registratie dat geen enkele limiet bond.
5. **`src/tradebot/risk/portfolio.py` — opgesplitst en ontmanteld.** Alle risicologica verhuist naar `risk/limits.py` en `risk/engine.py`; alle allocatielogica naar `portfolio/`. Het originele bestand verdwijnt of blijft als dunne, gedeprecieerde re-export met een harde deprecation-crash.
6. **`src/tradebot/risk/factor_alpha.py` — verplaatst naar `alpha/`** of gearchiveerd. Alpha-logica mag niet in L7 wonen.
7. **`src/tradebot/risk/stress_test.py` (herschreven)** — het stress-simulatieframework met minimaal vier scenario's:
   - **S1 Vol-shock:** realized vol vertienvoudigt binnen één bar
   - **S2 Correlatie-instorting:** alle cross-asset correlaties gaan naar 1
   - **S3 Gap-down:** −30% overnight zonder tussenliggende bars
   - **S4 Alpha-runaway:** elke alpha-unit vraagt gelijktijdig maximale exposure `a_t = ±1`
8. **`tests/unit/test_risk_alpha_decoupling.py`** — de **ontkoppelingstest** (sectie 24, vereist bewijs): statisch bewijs dat `alpha/` nergens uit `risk/` importeert én dat `risk/` geen enkele alpha-parameter kent; plus functioneel bewijs dat de risicolaag identiek reageert op een gegeven `a_t`, ongeacht welke alpha-unit hem produceerde.
9. **`tests/integration/test_risk_overrules_alpha.py`** — bewijst in elk stressscenario dat de risicolaag de gevraagde exposure aantoonbaar verkleint, en registreert per scenario welke constraint bond.
10. **`tests/unit/test_kill_switch_irreversibility.py`** — bewijst dat de `HALTED`-toestand niet automatisch herstelt, ook niet wanneer de markt herstelt.
11. **`conf/risk/default.yaml` (uitgebreid)** — `sigma_target`, `max_leverage`, `max_concentration`, `gross_cap`, `net_cap`, `drawdown_breaker_levels`, `daily_loss_limit`, `adv_participation_cap`.
12. **`reports/RISK_STRESS_REPORT.md`** — per scenario: gevraagde exposure, toegestane exposure, bindende constraint, en de resulterende drawdown-reductie ten opzichte van de ongeremde Phase 3-baseline.
13. **`apps/run_stress.py`** (≤80 LOC) — CLI voor het stressframework.

---

## STAPSGEWIJZE UITVOERING

**Stap 1 — Verstrengeling in kaart brengen.**
Scan `src/tradebot/` op elke plek waar risicologica buiten `risk/` staat (leverage-berekeningen, stop-losses, positielimieten, drawdown-checks in alpha-units, feature-pipelines of backtest-code) én op elke plek waar alpha-logica binnen `risk/` staat (o.a. `risk/factor_alpha.py`). Publiceer `reports/phase4_entanglement_map.md`. Dit is het werkplan én het bewijsstuk voor sectie 4.

**Stap 2 — Risicocontract definiëren.**
Schrijf het formele interfacecontract van de `RiskEngine` op vóór je implementeert: welke input, welke output, welk auditspoor. Leg vast dat de engine **pure** functies gebruikt van `(desired_exposure, market_state, risk_state) → (permitted_exposure, binding_constraints)`.

**Stap 3 — Volatility targeting isoleren.**
Bouw `vol_targeting.py` als standalone module. De `σ̂_{t+1|t}` komt uit L2 (EWMA λ=0.94 in productie). De module kent geen alpha, geen symbool-specifieke uitzonderingen en geen historische performance. Bij niet-eindige of ontbrekende `σ̂`: crash.

**Stap 4 — Harde limieten bouwen.**
Implementeer `limits.py`. Elke limiet is een zelfstandige, testbare functie die de gevraagde exposure inperkt en registreert dát hij bond. Volgorde van toepassing is deterministisch en geconfigureerd — nooit impliciet in de code-volgorde.

**Stap 5 — Kill switches bouwen.**
Implementeer de Drawdown Breaker op High-Water Mark met getrapte de-grossing, en de Daily Loss Governor. De `HALTED`-toestand is persistent en overleeft een herstart van het proces. Bouw dit expliciet met het oog op L13 in Phase 7.

**Stap 6 — RiskEngine samenstellen.**
Componeer vol-targeting, limieten en kill switches tot één soevereine engine met volledig auditspoor. Elke beslissing is herleidbaar tot een geconfigureerde drempel en een gemeten waarde.

**Stap 7 — `risk/portfolio.py` ontmantelen.**
Splits het bestand conform sectie 24. Doe dit in kleine, atomaire commits per verplaatst verantwoordelijkheidsgebied, en houd de testsuite na elke commit groen. Verplaats `risk/factor_alpha.py` naar `alpha/` of archiveer het.

**Stap 8 — Alpha-units zuiveren.**
Verwijder elke resterende risicoparameter uit `alpha/`: leverage, stop-loss, drawdown-check, positielimiet. Alpha levert `a_t ∈ [-1, +1]` en verder niets (sectie 11.1). Laat `test_alpha_isolation.py` uit Phase 3 hierop opnieuw draaien.

**Stap 9 — Ontkoppelingstest schrijven en draaien.**
Bouw `test_risk_alpha_decoupling.py`. De functionele helft is de scherpste: voer twee verschillende alpha-units met een identieke `a_t`-reeks door de engine en bewijs dat de output bit-identiek is. Is dat niet zo, dan zit er alpha-kennis in de risicolaag.

**Stap 10 — Stress-simulaties draaien.**
Draai S1 t/m S4. Per scenario: welke exposure vroeg alpha, welke stond risk toe, welke constraint bond, en wat is de drawdown-reductie ten opzichte van de ongeremde Phase 3-baseline. **Als in een scenario geen enkele limiet bindt, is de kalibratie te ruim en moet hij worden aangescherpt.**

**Stap 11 — Rapporteren en registreren.**
Publiceer `reports/RISK_STRESS_REPORT.md` en registreer de risicoconfiguratie in de ledger met `config_hash`. Een risicoconfiguratie zonder hash is niet auditbaar.

**Stap 12 — Exit-rapport.**
`reports/phase4_exit_report.md` met per exit-criterium het bewijs.

---

## CRITERIA & VALIDATIE (EXIT CRITERIA)

1. **Risicolimieten overrulen aantoonbaar alpha-posities in stress-simulaties.** In elk van de scenario's S1 t/m S4 is de toegestane exposure aantoonbaar kleiner dan de gevraagde, met registratie van de bindende constraint.
2. **Ontkoppelingstest geslaagd** (vereist bewijs uit sectie 24). Statisch: `alpha/` importeert nergens uit `risk/`, en `risk/` bevat nul alpha-parameters. Functioneel: identieke `a_t` → bit-identieke `permitted_exposure`, ongeacht de herkomst.
3. **`risk/portfolio.py` bestaat niet meer in zijn verstrengelde vorm.** Risicologica zit in `risk/`, allocatielogica in `portfolio/`.
4. **Volatility targeting is standalone** en faalt hard bij een ontbrekende of niet-eindige `σ̂`. Nul fallbacks naar een constante volatiliteit.
5. **Kill switches vuren en herstellen niet automatisch.** De `HALTED`-toestand overleeft een procesherstart en wordt uitsluitend handmatig opgeheven.
6. **Volledig auditspoor.** Elke risicobeslissing is herleidbaar tot een geconfigureerde drempel en een gemeten waarde; nul niet-geregistreerde ingrepen.
7. **Nul hardcoded drempels.** Elke limiet komt uit `conf/risk/` en is gedekt door een `config_hash` in de ledger.
8. **Baseline-impact gekwantificeerd.** De Phase 3-baseline is opnieuw gedraaid mét de risicolaag; het verschil in Sharpe, drawdown en turnover is gerapporteerd. Een risicolaag die de drawdown niet verlaagt, is verkeerd gekalibreerd.

---

## REGELS & HANDLINGSINSTRUCTIES

- **Risico overruled altijd alpha.** Er bestaat geen override, geen "high-conviction"-uitzondering, geen bypass-flag. Een bypass die vandaag voor onderzoek wordt ingebouwd, is morgen een live verliespost.
- **Fail-fast compliance.** Ontbrekende vol-schatting, ontbrekende limietconfiguratie, corrupte risk-state: crashen. Nul `try/except` fallbacks. Een risicolaag die stilzwijgend degradeert naar "geen limiet" is de gevaarlijkste module in het platform.
- **100% PIT rigor.** De ex-ante volatiliteit gebruikt uitsluitend informatie tot en met `t`. Elke drawdown-berekening en High-Water Mark is causaal. Een limiet die gekalibreerd is op de volledige sample is een lookahead-lek met kapitaalgevolgen.
- **Strikte laagscheiding.** L7 kent geen alpha-signalen, geen modelnamen en geen strategie-identiteit. L4 kent geen limieten. Deze wederzijdse blindheid is de kern van de fase.
- **Geen hardcoded variabelen.** Alle drempels uit `conf/risk/`, gehashed en geregistreerd.
- **Conservatief bij twijfel.** Is een parameter onzeker, kies dan de strengere waarde en documenteer waarom. Asymmetrie in kosten van fouten rechtvaardigt asymmetrie in defaults.
- **Atomaire commits.** `refactor(risk): extract standalone volatility targeting module`, `feat(risk): add high-water-mark drawdown breaker with tiered de-grossing`, `refactor(risk): split portfolio.py into limits and allocation responsibilities`, `test(risk): prove risk engine overrules alpha under vol shock`.
- **Geen nieuwe modellen.** GARCH blijft Phase 6. In deze fase is EWMA de productie-vol-estimator.

---

## STARTINSTRUCTIE

> **Begin nu met Stap 1:** scan de volledige `src/tradebot/`-boom en genereer `reports/phase4_entanglement_map.md` met (a) elke locatie buiten `risk/` waar risicologica staat — leverage, stop-loss, positielimiet, drawdown-check — en (b) elke locatie binnen `risk/` waar alpha-logica staat, te beginnen met `risk/factor_alpha.py`. Commit als `docs(risk): map alpha-risk entanglement across the codebase`.
