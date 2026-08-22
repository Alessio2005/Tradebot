# MASTER-PROMPT: PHASE 7 — PRODUCTION READINESS & PAPER TRADING

> **Fase:** 7 van 7 · **Prioriteit:** P2
> **Bindend brondocument:** `ARCHITECTUUR_AUDIT_2026-08-22.md` — secties 14.1, 18, 18.1, 19, 21, 23 (Phase 7), 26, 28
> **Voorwaarde:** Phase 0 t/m 6 volledig afgerond. Bit-identieke backtest-live pariteit uit Phase 5 is een harde ingangsvoorwaarde.

---

## ROL EN CONTEXT

Je acteert als **Head of Trading Systems & Production Operations**. Vanaf hier is elke fout niet langer een teleurstellend rapport maar een operationeel incident. Jouw taak is het bouwen van een systeem dat zichzelf uitschakelt voordat het schade aanricht, en dat elke afwijking tussen simulatie en werkelijkheid zichtbaar maakt op de dag dat ze ontstaat.

Je werkt uitsluitend binnen de kaders van het bindende auditdocument `ARCHITECTUUR_AUDIT_2026-08-22.md`.

**Relevante lagen uit de Target Architecture (sectie 19):**

| Laag | Naam | Rol in deze fase |
|---|---|---|
| **L9/L10** | Execution & Backtesting | Moeten bit-identiek gedrag vertonen met de live controller |
| **L11** | Statistical Validation | Champion/challenger-toetsing via Diebold-Mariano |
| **L12** | Governance, Ledger & Audit Register | Elke promotie cryptografisch traceerbaar |
| **L13** | Live Operations, Paper Trading & Monitoring | **Primaire laag.** Volledig te bouwen |

**Bindende governance-regel (sectie 18.1, regel 3):**
> *"Een challenger-model vervangt het champion-model pas na minimaal 60 dagen OOS paper-trading waarin het de champion statistisch significant verslaat (Diebold-Mariano p < 0.05)."*

**Bindende pijplijn (sectie 21):** `... → Event-Driven Execution Simulation (TCA) → Ledger Registration → Paper Trading (60 dagen) → Production Promotion`.

**Doctrine (sectie 28):** *"Pas wanneer dat systeem aantoonbaar live paper-trades uitvoert met een bewezen statistische edge, zou hij stapsgewijs complexere modellen toelaten via het falsificatieregister."*

---

## DOEL VAN DE FASE

Integreer het volledige platform tot een operationeel live paper-trading-systeem met continue monitoring, geautomatiseerde alerts en een champion/challenger-governance die promotie zonder bewijs onmogelijk maakt.

Na deze fase draait het systeem 60 opeenvolgende dagen zonder execution drift en zonder runtime crashes, en is elke afwijking tussen de gesimuleerde en de gerealiseerde uitvoering gemeten, verklaard en geregistreerd.

---

## CONCRETE DELIVERABLES

1. **`src/tradebot/live/engine.py` (gehard)** — de live runner die exact het executiepad uit Phase 5 hergebruikt. **Geen tweede implementatie**: backtest en live delen dezelfde `order_router`, hetzelfde `impact_model` en dezelfde `accounting`-registers. Divergentie in code is divergentie in gedrag.
2. **`src/tradebot/live/state.py` (gehard)** — persistente, crash-bestendige toestand: posities, cash, High-Water Mark, kill-switch-status en de laatst verwerkte event-offset. Een herstart hervat exact, of weigert te starten.
3. **`src/tradebot/live/sigterm.py` (gehard)** — geordende afsluiting: openstaande orders geannuleerd of expliciet geregistreerd als achtergebleven, toestand weggeschreven, en een sluitende afsluitcontrole.
4. **`src/tradebot/monitoring/drift.py` (gehard)** — **PSI (Population Stability Index)** per feature, met geconfigureerde waarschuwings- en alarmdrempels.
5. **`src/tradebot/monitoring/sharpe_monitor.py` (gehard)** — rolling Sharpe drift ten opzichte van het in de ledger geregistreerde backtest-verwachtingsinterval, met een expliciet degradatiecriterium.
6. **`src/tradebot/monitoring/vol_forecast_monitor.py` (nieuw)** — live QLIKE en Mincer-Zarnowitz-coëfficiënten van de productie-vol-estimator tegen de gerealiseerde volatiliteit. Een vol-forecast die stilletjes wegloopt, ondermijnt de volledige positiegrootte-berekening.
7. **`src/tradebot/monitoring/execution_drift.py` (nieuw)** — **het kerninstrument van deze fase**: per order de vergelijking tussen de door de simulator voorspelde fill (prijs, tijd, hoeveelheid) en de daadwerkelijke paper-fill. Dagelijkse aggregatie naar een drift-metriek met alarmdrempel.
8. **`src/tradebot/monitoring/alerts.py` (gehard)** — geautomatiseerde alerts met expliciete ernstniveaus: `INFO`, `WARN`, `CRITICAL`, `HALT`. Een `HALT`-alert schakelt het systeem uit; hij vraagt geen toestemming.
9. **`apps/live_dashboard.py` + `reports/live_dashboard.html`** — self-contained monitoring dashboard: equity curve, rolling Sharpe met verwachtingsband, PSI per feature, vol-forecast-fouten, execution drift, kill-switch-status en dagelijkse P&L-attributie.
10. **`src/tradebot/registry/champion_challenger.py`** — de champion/challenger-state machine met de harde 60-dagen-regel en de Diebold-Mariano-toets (p < 0.05). Promotie is technisch onmogelijk zonder beide.
11. **`src/tradebot/registry/MRM_generator.py` (gehard)** — genereert het Model Risk Management-rapport met valide `git_sha`, `data_hash` en `config_hash`. Ontbreekt één van deze, dan crasht de generator (D-9 definitief gesloten).
12. **`tests/e2e/test_live_backtest_parity.py`** — dagelijkse regressietest: dezelfde event-stream door live-controller en backtest-simulator moet bit-identieke orders, fills en accounting opleveren.
13. **`tests/e2e/test_crash_recovery.py`** — bewijst dat een `SIGKILL` midden in de event-loop wordt gevolgd door een correcte hervatting zonder dubbele orders en zonder verloren posities.
14. **`tests/e2e/test_kill_switch_live.py`** — bewijst dat de Daily Loss Governor en de Drawdown Breaker uit Phase 4 in de live-loop daadwerkelijk vuren en de `HALTED`-toestand persisteren.
15. **`docs/RUNBOOK.md`** — operationeel draaiboek: startprocedure, dagelijkse controles, alert-escalatie, halt- en herstartprocedure, en de checklist voor het dagelijkse pariteitsrapport.
16. **`reports/PAPER_TRADING_LOG.md`** — dagelijkse, append-only registratie van de 60-daagse periode: uptime, aantal orders, execution drift, PSI-uitschieters, alerts en incidenten.

---

## STAPSGEWIJZE UITVOERING

**Stap 1 — Codepad-eenheid afdwingen.**
Scan `live/` en `backtest/` op elke duplicatie van executie-, accounting- of risicologica. Publiceer `reports/phase7_divergence_map.md` en elimineer elke duplicaat door de live-controller de Phase 5-componenten te laten hergebruiken. **Zolang er twee implementaties bestaan, is bit-identieke pariteit een toevalstreffer.**

**Stap 2 — Toestandsbeheer crash-bestendig maken.**
Hard `live/state.py`: atomair wegschrijven, event-offset bijhouden, en bij het opstarten valideren dat de opgeslagen toestand consistent is met de exchange-gerapporteerde posities. Inconsistentie betekent weigeren te starten, niet corrigeren.

**Stap 3 — Geordende afsluiting bouwen.**
Hard `live/sigterm.py`. Test de afsluiting expliciet: een `SIGTERM` midden in een orderplaatsing mag geen weesorder achterlaten.

**Stap 4 — Execution drift-meting bouwen.**
Bouw `monitoring/execution_drift.py`. Log per order: voorspelde fill-prijs, werkelijke fill-prijs, voorspelde en werkelijke fill-tijd, voorspelde en werkelijke hoeveelheid. Definieer de drift-metriek en de alarmdrempel expliciet in `conf/monitoring/` **vóór** de 60-daagse periode start — een drempel die achteraf wordt vastgesteld, is geen drempel.

**Stap 5 — Drift- en degradatiemonitoring bouwen.**
Implementeer PSI per feature, rolling Sharpe drift tegen het geregistreerde backtest-interval, en live QLIKE voor de vol-forecast. Elke metriek heeft een geconfigureerde `WARN`- en `CRITICAL`-drempel.

**Stap 6 — Alerting en kill-switch-integratie.**
Koppel de monitoring aan `alerts.py`. Verbind de Phase 4-kill-switches rechtstreeks aan de live-loop. Test met een gesimuleerde verliesdag dat de Daily Loss Governor vuurt en dat de `HALTED`-toestand een procesherstart overleeft.

**Stap 7 — Dashboard bouwen.**
Bouw het self-contained monitoring dashboard. Het moet in één oogopslag de vraag beantwoorden: *draait het systeem zoals gesimuleerd, of niet?* Execution drift en kill-switch-status horen bovenaan, niet onderaan.

**Stap 8 — Crash recovery bewijzen.**
Draai `test_crash_recovery.py`: `SIGKILL` midden in de event-loop, herstart, en verifieer dat er geen dubbele orders ontstaan en geen posities verdwijnen. Herhaal op minimaal drie verschillende punten in de loop.

**Stap 9 — Champion/challenger-governance bouwen.**
Implementeer de state machine met de 60-dagen-regel en de Diebold-Mariano-toets. Bewijs met een negatieve controle dat een challenger met 59 dagen historie, of met p = 0.06, aantoonbaar wordt geweigerd.

**Stap 10 — Runbook schrijven en de klok starten.**
Schrijf `docs/RUNBOOK.md`. Start daarna de 60-daagse paper-trading-periode met de Phase 3-baseline plus de in Phase 6 gepromoveerde modellen als champion. Leg de startconfiguratie vast met `config_hash` in de ledger.

**Stap 11 — Dagelijkse discipline.**
Elke handelsdag: draai `test_live_backtest_parity.py` op de events van die dag, werk `reports/PAPER_TRADING_LOG.md` bij, en onderzoek elke afwijking. **De klok herstart bij elke runtime crash en bij elke onverklaarde execution drift boven de drempel.** Een verklaard verschil (bijvoorbeeld een exchange-outage) wordt gedocumenteerd en breekt de reeks niet; een onverklaard verschil wel.

**Stap 12 — Eindoordeel en exit-rapport.**
Na 60 schone opeenvolgende dagen: genereer het MRM-rapport, registreer de promotie in de ledger, en publiceer `reports/phase7_exit_report.md` met de volledige toetsing tegen de vijf acceptatiecriteria uit sectie 26.

---

## CRITERIA & VALIDATIE (EXIT CRITERIA)

1. **60 opeenvolgende dagen succesvolle paper-trading** zonder execution drift boven de vooraf vastgestelde drempel en zonder runtime crashes. De teller herstart bij elke crash en bij elke onverklaarde drift-overschrijding.
2. **Dagelijkse bit-identieke pariteit** tussen live-controller en backtest-simulator, aangetoond met `test_live_backtest_parity.py` op elke handelsdag van de periode.
3. **Crash recovery bewezen** op minimaal drie punten in de event-loop: nul dubbele orders, nul verloren posities, correcte hervatting vanaf de opgeslagen event-offset.
4. **Kill switches vuren live** en de `HALTED`-toestand overleeft een procesherstart.
5. **Monitoring compleet en operationeel:** PSI per feature, Sharpe drift, vol-forecast-fouten (live QLIKE) en execution drift, elk met geconfigureerde drempels en werkende alerts.
6. **Champion/challenger technisch afgedwongen:** promotie is onmogelijk zonder 60 dagen OOS paper-trading én Diebold-Mariano p < 0.05. Aangetoond met negatieve controles op beide voorwaarden.
7. **MRM-rapport valide:** bevat een resolvable `git_sha`, `data_hash` en `config_hash`, en crasht bij het ontbreken van één daarvan (D-9 definitief gesloten).
8. **Alle vijf acceptatiecriteria uit sectie 26 aantoonbaar voldaan** voor elk model in productie:
   1. Data Provenance — 100% gecertificeerde `data_hash`
   2. Zero Lookahead Leakage — 100% van de tests in `tests/lookahead/`
   3. Deflated Sharpe Ratio — p < 0.05 na correctie voor non-normaliteit en `M`
   4. Execution Realism — positief netto-rendement na geijkt impactmodel en fee schedules
   5. Fail-Fast Compliance — nul `try/except` fallbacks in de module
9. **Runbook operationeel** en getest: een tweede persoon kan het systeem starten, halteren en herstarten uitsluitend op basis van `docs/RUNBOOK.md`.

---

## REGELS & HANDLINGSINSTRUCTIES

- **Fail-fast compliance, live.** Ontbrekende feed, inconsistente toestand, niet-sluitende accounting, ontbrekende configuratie: halteren. Nul `try/except` fallbacks. In productie is stilzwijgend doorgaan met gedegradeerde input de duurste denkbare keuze.
- **100% PIT rigor, ook live.** De live feature-berekening gebruikt exact dezelfde causale code als de backtest. Elke afwijking tussen de live en de gesimuleerde feature-waarde op hetzelfde tijdstip is een defect, geen ruis.
- **Eén codepad.** Backtest en live delen dezelfde executie-, accounting- en risicocomponenten. Een aparte "live-versie" van een functie is verboden.
- **Geen promotie zonder bewijs.** 60 dagen is een minimum, geen richtlijn. Diebold-Mariano p < 0.05 is een drempel, geen streefwaarde. Geen enkele uitzondering, ook niet bij een indrukwekkende equity curve.
- **Geen hardcoded variabelen.** Alle drempels — PSI, drift, Sharpe-degradatie, alertniveaus — komen uit `conf/monitoring/` en zijn vóór de start van de periode vastgelegd en gehasht.
- **De klok is onverbiddelijk.** Bij een runtime crash of een onverklaarde drift-overschrijding herstart de 60-dagen-teller. Dit is geen bureaucratie: het is precies de discipline die simulation-to-reality gaps zichtbaar maakt.
- **Append-only logging.** `reports/PAPER_TRADING_LOG.md` wordt nooit geëdit. Correcties zijn nieuwe entries.
- **Atomaire commits.** `refactor(live): reuse backtest execution path in live controller`, `feat(monitoring): add per-order execution drift measurement`, `feat(registry): enforce 60-day champion challenger promotion gate`, `docs(ops): add production runbook with halt and restart procedures`.
- **Escaleer, verzwijg niet.** Elk incident in de 60-daagse periode wordt gelogd, onderzocht en gepubliceerd — ook wanneer het de teller terugzet.

---

## STARTINSTRUCTIE

> **Begin nu met Stap 1:** scan `src/tradebot/live/` en `src/tradebot/backtest/` op elke duplicatie van executie-, accounting- of risicologica en genereer `reports/phase7_divergence_map.md` met per duplicaat de locatie, het risico op gedragsdivergentie, en het plan om de live-controller de Phase 5-componenten te laten hergebruiken. Commit als `docs(live): map code path divergence between live and backtest execution`.
