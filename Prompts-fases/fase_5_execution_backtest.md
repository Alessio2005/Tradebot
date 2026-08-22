# MASTER-PROMPT: PHASE 5 — EXECUTION & BACKTESTING ENGINE

> **Fase:** 5 van 7 · **Prioriteit:** P1
> **Bindend brondocument:** `ARCHITECTUUR_AUDIT_2026-08-22.md` — secties 15, 15.1, 16, 16.1, 19, 23 (Phase 5), 24, 26
> **Voorwaarde:** Phase 0 t/m 4 volledig afgerond. Zonder soevereine risicolaag simuleer je een systeem dat niet bestaat.

---

## ROL EN CONTEXT

Je acteert als **Execution Architect & Market Microstructure Engineer**. Jouw opdracht is het dichten van de simulation-to-reality gap. Een backtest die marktimpact, spread en latentie negeert, is geen bewijs maar fictie — en fictie die winstgevend lijkt, is de duurste soort.

Je werkt uitsluitend binnen de kaders van het bindende auditdocument `ARCHITECTUUR_AUDIT_2026-08-22.md`.

**Relevante lagen uit de Target Architecture (sectie 19):**

| Laag | Naam | Rol in deze fase |
|---|---|---|
| **L8** | Portfolio Construction & Sizing | Levert de doelgewichten die geëxecuteerd moeten worden |
| **L9** | Execution & Order Routing (TCA, Impact) | **Primaire laag.** Impact, spread, slippage, order lifecycle |
| **L10** | Backtesting Engine (Event-Driven) | **Primaire laag.** Consolidatie van 4 engines naar 1 autoriteit |
| **L13** | Live Operations & Paper Trading | Moet in Phase 7 bit-identiek gedrag vertonen met L9/L10 |

**Kritieke bevindingen die deze fase adresseert:**
- **Sectie 24 — REDESIGN:** Vier overlappende backtesters (`backtest/evaluation.py`, `backtest/portfolio.py`, `backtest/bidirectional.py`, `backtest/per_side.py`) moeten geconsolideerd worden tot **één** event-driven engine. Vereist bewijs: **een pariteits-test**.
- **D-2 (P1):** `apps/calibrate_impact.py` levert volgens de documentatie η en κ_d. **Het bestand bestaat niet.**
- **D-3 (P1):** `tests/integration/test_tca_roundtrip.py` draait volgens de documentatie per PR. **Het bestand bestaat niet.**
- **D-4 (P1):** `conf/tca/default.yaml` bestaat niet.
- **Sectie 3.2:** *"Uncalibrated TCA — de documentatie claimt marktimpact-kalibratie, maar de bijbehorende kalibratietool ontbreekt volledig."*

**Bindend impactmodel (sectie 15.1) — geijkte Square-Root Law:**

    Impact = η · σ_daily · sqrt( OrderSize / DailyVolume )

**Bindende backtest-doctrine (sectie 16.1):**
- **Vectorized Research Engine:** uitsluitend voor snelle hypothese-screening in de exploratieve fase. *"Vectorized resultaten worden nooit geaccepteerd als bewijs voor modelpromotie."*
- **Authoritative Event-Driven Backtester:** de enige wettige autoriteit. Simuleert discrete event-loops, causale order routing en latency, expliciete cash/position accounting registers, en Point-in-Time datafeeds zonder lookahead.

---

## DOEL VAN DE FASE

Consolideer de vier overlappende backtesters tot één authoritative event-driven engine met volledig geïntegreerde TCA, en kalibreer het marktimpactmodel op echte orderboekdata.

Na deze fase bestaat er precies één instantie die mag verklaren wat een strategie zou hebben opgeleverd, en die instantie deelt zijn executiepad bit-identiek met de live paper-trading-controller.

---

## CONCRETE DELIVERABLES

1. **`src/tradebot/backtest/engine.py`** — de authoritative event-driven engine: discrete event-loop op bar- of tick-niveau, causale order routing met expliciete latentie, en een strikt sequentiële verwerking van `market event → signal → risk → portfolio → order → fill → accounting`.
2. **`src/tradebot/backtest/accounting.py`** — expliciete cash- en positieregisters met dubbele boekhouding. Elke fill muteert cash én positie; een run eindigt met een sluitende balanscontrole of crasht.
3. **`src/tradebot/backtest/vectorized.py`** — de vectorized engine, expliciet gemarkeerd als **research-only**. Elke output draagt het label `NOT_ADMISSIBLE_AS_PROMOTION_EVIDENCE`, en de promotiegate uit Phase 2 weigert deze resultaten (sectie 16.1).
4. **De vier legacy engines geconsolideerd.** `evaluation.py`, `portfolio.py`, `bidirectional.py` en `per_side.py` worden vervangen door de nieuwe engine. Hun unieke, correcte gedrag (bidirectionele en per-side accounting) wordt overgenomen als geconfigureerde modus, niet als aparte engine.
5. **`src/tradebot/execution/impact_model.py`** — het geijkte Square-Root Law impactmodel met parameters η en κ_d uit `conf/execution/`. Een niet-gekalibreerde η is een `ConfigContractError`, geen default.
6. **`apps/calibrate_impact.py` (D-2 gesloten, ≤80 LOC)** — kalibreert η en κ_d op de orderboek- en trade-data uit Phase 1, met gerapporteerde betrouwbaarheidsintervallen en de gebruikte `data_hash`.
7. **`src/tradebot/execution/spread.py` en `slippage.py` (gehard)** — dynamische spread- en slippage-modellering op basis van de actuele bid-ask spread en marktvolatiliteit, gevoed door echte L1-data.
8. **`src/tradebot/execution/order_router.py`** — order lifecycle: limit order queues, time-to-fill, partial fills op basis van orderboekdiepte, en expliciete afhandeling van niet-gevulde orders (die verdwijnen niet, ze blijven open of worden geannuleerd volgens beleid).
9. **`src/tradebot/tca/post_trade.py` (gehard)** — post-trade TCA: implementation shortfall ten opzichte van de arrival price, uitgesplitst naar spread-, impact- en timing-component.
10. **`conf/execution/` en `conf/tca/default.yaml` (D-4 gesloten)** — fee schedules per exchange, latentieparameters, η, κ_d, participatielimieten, order-typebeleid.
11. **`tests/integration/test_tca_roundtrip.py` (D-3 gesloten)** — order → fill → TCA → accounting sluit rond: de som van gerealiseerde kosten is gelijk aan het verschil tussen arrival price en effectieve executieprijs, tot op numerieke tolerantie.
12. **`tests/integration/test_engine_parity.py`** — de **pariteits-test** (vereist bewijs, sectie 24): de nieuwe engine reproduceert het gedrag van elk van de vier legacy engines op hun eigen historische testcases, of het verschil is expliciet verklaard en gedocumenteerd als een gecorrigeerde fout.
13. **`tests/integration/test_backtest_live_parity.py`** — bit-identieke pariteit tussen de backtest execution simulator en de live paper-trading logica op een gedeelde event-replay.
14. **`tests/lookahead/test_engine_causality.py`** — bewijst dat de event-loop nooit een prijs gebruikt die op het beslismoment nog niet bekend was.
15. **`reports/TCA_CALIBRATION_REPORT.md`** — de gekalibreerde η, κ_d, hun betrouwbaarheidsintervallen, de gebruikte dataperiode en `data_hash`, en de impact op de Phase 3-baseline.

---

## STAPSGEWIJZE UITVOERING

**Stap 1 — De vier engines forensisch vergelijken.**
Analyseer `evaluation.py`, `portfolio.py`, `bidirectional.py` en `per_side.py`. Publiceer `reports/phase5_engine_diff.md`: welk gedrag is identiek, welk gedrag verschilt, en — het belangrijkste — **welke verschillen zijn feitelijke bugs**. Sectie 4 waarschuwt dat deze overlap *"kweekvijvers voor subtiele simulatiefouten"* creëert. Dit rapport identificeert ze.

**Stap 2 — Event-contract vastleggen.**
Definieer de event-typen (`MarketEvent`, `SignalEvent`, `OrderEvent`, `FillEvent`) en de strikte verwerkingsvolgorde vóór je implementeert. Leg expliciet vast op welk tijdstip elke event beschikbaar is en met welke latentie hij wordt verwerkt.

**Stap 3 — Accounting-registers eerst.**
Bouw `accounting.py` vóór de engine. Dubbele boekhouding: elke transactie muteert cash en positie consistent. Test met handmatig doorgerekende scenario's — long open, partial fill, short, fee, funding payment, close. Een engine bovenop een onbewezen accounting-laag is waardeloos.

**Stap 4 — Event-driven engine bouwen.**
Implementeer de event-loop. De datafeed levert uitsluitend Point-in-Time data uit de Phase 1-store. Een order geplaatst op basis van informatie op `t` mag ten vroegste vullen op `t + latency`.

**Stap 5 — Impactmodel kalibreren (D-2).**
Bouw `apps/calibrate_impact.py` en kalibreer η op de orderboek- en trade-data. Rapporteer betrouwbaarheidsintervallen. **Blijkt de data ontoereikend** — Open Question 1 uit sectie 27 — dan is dat een geldig resultaat: rapporteer het expliciet, gebruik een conservatieve bovengrens-schatting, en markeer alle downstream-resultaten als `IMPACT_UNCALIBRATED`. Verzin geen η.

**Stap 6 — Spread, slippage en fees.**
Voed de spread- en slippage-modellen met echte L1-data. Vul de fee schedules per exchange in `conf/execution/`. Maker- en taker-fees verschillen materieel; behandel ze afzonderlijk.

**Stap 7 — Order lifecycle bouwen.**
Implementeer limit order queues, time-to-fill en partial fills op basis van orderboekdiepte. Modelleer expliciet wat er gebeurt met het onuitgevoerde deel. Een backtest die aanneemt dat alles volledig vult tegen mid, overschat systematisch.

**Stap 8 — TCA sluitend maken (D-3).**
Bouw `test_tca_roundtrip.py`. De som van gerealiseerde kosten moet gelijk zijn aan het verschil tussen arrival price en effectieve executieprijs. Een niet-sluitende TCA betekent dat ergens kosten verdwijnen — en verdwenen kosten zijn winst die in de realiteit niet bestaat.

**Stap 9 — Pariteits-test tegen de legacy engines.**
Draai `test_engine_parity.py`. Elk verschil is óf een bug in de oude engine — documenteer welke, dat is waardevolle informatie — óf een regressie in de nieuwe. Er is geen derde optie.

**Stap 10 — Legacy engines verwijderen.**
Pas na een geslaagde pariteits-test: verwijder de vier oude engines in aparte, atomaire commits. Zolang ze bestaan, zal iemand ze gebruiken.

**Stap 11 — Backtest-live pariteit.**
Bouw `test_backtest_live_parity.py`: laat de live paper-trading-controller (L13) en de backtest-simulator dezelfde event-replay verwerken en bewijs bit-identieke orders, fills en accounting. Dit is het scherpste exit-criterium van de fase en de directe voorwaarde voor Phase 7.

**Stap 12 — Baseline herwaarderen en exit-rapport.**
Draai de Phase 3-baseline opnieuw door de authoritative engine met gekalibreerde kosten. Vergelijk met de voorlopige kostenaanname uit Phase 3. Publiceer `reports/TCA_CALIBRATION_REPORT.md` en `reports/phase5_exit_report.md`.

---

## CRITERIA & VALIDATIE (EXIT CRITERIA)

1. **Bit-identieke resultaten tussen paper-trading logica en backtest execution simulator** op een gedeelde event-replay. Nul afwijkingen in orders, fills, cash en posities.
2. **Pariteits-test geslaagd** (vereist bewijs, sectie 24). De nieuwe engine reproduceert het gedrag van alle vier de legacy engines, of elk verschil is gedocumenteerd als een gecorrigeerde fout met bewijs.
3. **Precies één authoritative engine.** De vier legacy backtesters bestaan niet meer. De vectorized engine bestaat nog, is gelabeld als research-only, en zijn output wordt aantoonbaar door de promotiegate geweigerd (sectie 16.1).
4. **η gekalibreerd op echte data** (D-2 gesloten), met gerapporteerd betrouwbaarheidsinterval en `data_hash`, of expliciet gemarkeerd als `IMPACT_UNCALIBRATED` met conservatieve bovengrens en gedocumenteerde reden.
5. **TCA sluit rond** (D-3 gesloten). `test_tca_roundtrip.py` slaagt binnen numerieke tolerantie; nul verdwenen kosten.
6. **`conf/tca/default.yaml` bestaat en wordt gevalideerd** tegen zijn Pydantic-schema (D-4 gesloten).
7. **Accounting sluit.** Elke backtest-run eindigt met een sluitende balanscontrole of crasht.
8. **Causaliteit bewezen.** `test_engine_causality.py` en de volledige Phase 2-lookahead-suite zijn groen op de event-driven engine.
9. **Execution Realism aangetoond** (Acceptance Criterion 4, sectie 26): de Phase 3-baseline is herwaardeerd met het geijkte impactmodel en de exchange fee schedules, en het netto-resultaat is eerlijk gerapporteerd — ook wanneer de baseline daardoor onrendabel blijkt.

---

## REGELS & HANDLINGSINSTRUCTIES

- **Fail-fast compliance.** Ontbrekende η, ontbrekende fee schedule, ontbrekende orderboekdiepte, niet-sluitende accounting: crashen. Nul `try/except` fallbacks. Een executiemodule die stilzwijgend terugvalt op "nul kosten" produceert de gevaarlijkste output in het platform.
- **100% PIT rigor.** Een order gebruikt uitsluitend informatie die op het beslismoment bekend was. Fills gebeuren op prijzen die ná het beslismoment beschikbaar kwamen. Fills op de close van de bar waarop besloten werd, zijn een lookahead-lek.
- **Conservatief bij onzekerheid.** Is een executieparameter onzeker, kies de pessimistische kant en documenteer de keuze. Optimisme in executie-aannames is de meest voorkomende oorzaak van live teleurstelling.
- **Eén autoriteit.** Vectorized resultaten worden nooit geaccepteerd als promotiebewijs. Dit is niet onderhandelbaar en wordt technisch afgedwongen, niet met een afspraak.
- **Geen hardcoded variabelen.** η, κ_d, fees, latentie, participatielimieten en order-typebeleid komen uit `conf/execution/` en `conf/tca/`.
- **Verwijder pas na bewijs.** De legacy engines gaan pas weg als de pariteits-test slaagt. Daarna gaan ze echt weg.
- **Atomaire commits.** `feat(backtest): add authoritative event-driven engine with explicit accounting`, `feat(execution): calibrate square-root impact model on order book data`, `test(integration): add TCA roundtrip closure test`, `refactor(backtest): remove legacy per_side engine after parity proof`.
- **Documenteer elke gevonden simulatiefout.** De bugs die je in de vier oude engines vindt, horen in `docs/FALSIFICATION_REGISTER.md` — elk historisch resultaat dat op een buggy engine is verkregen, moet worden geherclassificeerd.

---

## STARTINSTRUCTIE

> **Begin nu met Stap 1:** analyseer `backtest/evaluation.py`, `backtest/portfolio.py`, `backtest/bidirectional.py` en `backtest/per_side.py` en genereer `reports/phase5_engine_diff.md` met een functionele vergelijkingsmatrix: identiek gedrag, verschillend gedrag, en per verschil het oordeel of het een feature of een simulatiefout is. Commit als `docs(backtest): forensic comparison of four legacy backtest engines`.
