# MASTER-PROMPT: PHASE 2 — RESEARCH & FALSIFICATION FOUNDATION

> **Fase:** 2 van 7 · **Prioriteit:** P0 (blokkerend voor elke modelpromotie)
> **Bindend brondocument:** `ARCHITECTUUR_AUDIT_2026-08-22.md` — secties 6, 17, 17.1, 18, 18.1, 23 (Phase 2), 26
> **Voorwaarde:** Phase 0 en Phase 1 volledig afgerond. Zonder gecertificeerde `data_hash` is elke statistische toets betekenisloos.

---

## ROL EN CONTEXT

Je acteert als **Head of Model Risk & Quantitative Research Governance**. Jouw functie bestaat niet om onderzoek mogelijk te maken, maar om te voorkomen dat ruis als alpha wordt gecodeerd. Je bouwt de poort waar elk model doorheen moet, en je bouwt hem zo dat niemand — inclusief jijzelf — hem kan omzeilen.

Je werkt uitsluitend binnen de kaders van het bindende auditdocument `ARCHITECTUUR_AUDIT_2026-08-22.md`.

**Relevante lagen uit de Target Architecture (sectie 19):**

| Laag | Naam | Rol in deze fase |
|---|---|---|
| **L11** | Statistical Validation & Falsification | **Primaire laag.** Volledig te automatiseren. |
| **L12** | Governance, Ledger & Audit Register | **Primaire laag.** Ledger koppelen aan CI/CD. |
| **L0/L1** | Data & Features | Consument: elke gate eist een geldige `data_hash` uit Phase 1 |

**Kritieke bevindingen die deze fase adresseert:**
- **D-1 (P0):** `model_risk_policy.md` claimt dat promotie geblokkeerd wordt door 6 specifieke lookahead-tests. **Geen van deze 6 testbestanden bestaat.** De governance is papier zonder tanden.
- **Sectie 6:** De onderzoeksfilosofie (popperiaanse falsificatie, pre-registratie) is solide, maar lijdt onder uitvoeringstekortkomingen.
- **Sectie 18.1, regel 4:** Geen enkele PR wordt gemerged als de lookahead-suite, DSR-check of invariantie-tests falen. Deze regel wordt momenteel niet afgedwongen.

**Wat behouden blijft (sectie 24 — RETAIN):**
- `docs/FALSIFICATION_REGISTER.md` — *"schaars wetenschappelijk goed; uitstekende hygiëne"*. Niet herschrijven.
- `registry/hypothesis_ledger.py` — atomaire, append-only logica. Niet herschrijven.
- `backtest/metrics.py::DSR` — de Bailey–López de Prado dimensionaliteitsfix inclusief Euler-Mascheroni-term is correct. Niet herschrijven; wel afdwingen.

**Statistische methodencategorisering (sectie 17.1) — bindend:**

| Methode | Status | Consequentie voor de CI-gate |
|---|---|---|
| Purged Walk-Forward met Embargo | **ESSENTIAL** | Verplicht; ontbreekt hij, dan faalt de gate |
| Deflated Sharpe Ratio (DSR) | **ESSENTIAL** | Verplicht; p ≥ 0.05 blokkeert promotie |
| Hansen's SPA / White's Reality Check | **ESSENTIAL** | Verplicht bij elke multi-strategie-vergelijking |
| Diebold-Mariano (HLN-correctie) | **USEFUL** | Verplicht bij forecast-vergelijkingen (Phase 6) |
| Combinatorial Purged CV (CPCV) | **OPTIONAL** | Diagnostiek; **nooit** als primaire gate |
| Probability of Backtest Overfitting | **OPTIONAL** | Diagnostiek; ondergeschikt aan OOS Walk-Forward |
| K-Fold CV (unpurged) | **BANNED** | Aanwezigheid in `src/` breekt de build |

---

## DOEL VAN DE FASE

Koppel het falsificatieregister en de hypothese-ledger hard aan de CI/CD-pipeline.

Na deze fase is de statistische validatiepijplijn uit sectie 17 volledig geautomatiseerd en niet-omzeilbaar. Een model kan alleen de staat `CANDIDATE` bereiken door aantoonbaar door Purged Walk-Forward, DSR en SPA te komen, en elke poging om een resultaat te registreren zonder valide `git_sha`, `data_hash` en `config_hash` crasht.

---

## CONCRETE DELIVERABLES

1. **De 6 ontbrekende lookahead-tests (D-1)** in `tests/lookahead/` — elk als zelfstandig, benoemd bestand:
   1. `test_truncation_invariance.py` — signaal op `t` verandert niet wanneer alle data na `t` wordt afgekapt
   2. `test_temporal_shift_invariance.py` — een uniforme tijdverschuiving van de input verschuift de output identiek
   3. `test_future_column_poisoning.py` — injectie van een toekomstige kolom moet het model laten crashen, niet stilzwijgend verbeteren
   4. `test_scaler_fit_causality.py` — elke scaler/normalisatie is uitsluitend gefit op data ≤ `t`
   5. `test_label_horizon_purge.py` — labels met horizon `h` zijn aantoonbaar gepurged en geëmbargood in elke fold
   6. `test_determinism_reproducibility.py` — identieke seed + identieke `data_hash` → bit-identieke output
2. **`src/tradebot/validation/` (nieuwe laag L11)** met:
   - `walk_forward.py` — Purged Walk-Forward met expliciete embargo-parameter
   - `dsr.py` — dunne, afdwingende wrapper rond de bestaande correcte DSR uit `backtest/metrics.py`, met verplichte `M` (aantal trials), `Y` (track length) en `γ` (skew/kurtosis) argumenten
   - `spa.py` — Hansen's SPA / White's Reality Check als promotiegate
   - `gates.py` — de compositie: één `run_promotion_gates(...)` die alle ESSENTIAL-toetsen draait en een onveranderlijk `GateResult` teruggeeft
3. **`src/tradebot/registry/trial_counter.py`** — persistente, append-only teller van het **totale** aantal uitgevoerde trials over alle waves (W1–W28 en verder). Zonder een eerlijke `M` is DSR een leugen.
4. **`src/tradebot/registry/preregistration.py`** — pre-registratie-contract (sectie 18.1, regel 1): hypothese, parameters, universum, periode, en **stop-criteria** worden vastgelegd vóór de eerste run. Een run zonder pre-registratie-ID crasht.
5. **`src/tradebot/registry/promotion.py` (herschreven)** — expliciete state machine: `REGISTERED → TESTED → CANDIDATE → PAPER → CHAMPION`, met per overgang de vereiste bewijslast. Overgangen zijn eenrichtingsverkeer behalve `→ FALSIFIED`.
6. **`scripts/check_banned_methods.py`** — AST-scanner die `KFold`, `ShuffleSplit`, `train_test_split(shuffle=True)` en elke unpurged CV in `src/` detecteert. Exit code 1 bij een hit.
7. **`.github/workflows/research_gates.yml`** — CI-workflow die bij elke PR draait: lookahead-suite (alle 6), banned-methods-scan, DSR-gate en SPA-gate op de geregistreerde kandidaten.
8. **`tests/killgates/test_gate_cannot_be_bypassed.py`** — bewijst met een bewust lekkend testmodel dat de gate daadwerkelijk blokkeert (negatieve controle).
9. **`docs/model_risk_policy.md` (gecorrigeerd)** — verwijst naar de 6 tests die nu daadwerkelijk bestaan; D-1 gesloten.
10. **`docs/FALSIFICATION_REGISTER.md`** — uitgebreid, niet herschreven: alle 20 gefalsificeerde en 4 "geaccepteerde" alpha-units uit sectie 3.2 worden herlabeld naar `INVALID — geen data_hash` totdat ze op de Phase 1-data opnieuw zijn getoetst.
11. **`apps/run_gates.py`** (≤80 LOC) — CLI om de volledige gate-suite lokaal te draaien vóór een PR.

---

## STAPSGEWIJZE UITVOERING

**Stap 1 — Historische claims ongeldig verklaren.**
Doorloop `docs/FALSIFICATION_REGISTER.md` en de hypothese-ledger. Elk resultaat dat is verkregen op data zonder gecertificeerde `data_hash` uit Phase 1 wordt herlabeld naar `INVALID — no certified data provenance`. Dit betreft expliciet de 4 "geaccepteerde" alpha-units uit sectie 3.2. Verwijder niets — de ledger is append-only; je voegt een invalidatie-entry toe.

**Stap 2 — Trial-teller bouwen.**
Implementeer `trial_counter.py` en reconstrueer de historische `M` uit de waves W1–W28 zo eerlijk mogelijk. Documenteer expliciet de reconstructiemethode en de onzekerheid. **Een te lage `M` maakt DSR structureel te optimistisch** — dit is de meest voorkomende manier waarop DSR in de praktijk wordt ondermijnd.

**Stap 3 — Pre-registratiecontract bouwen.**
Implementeer `preregistration.py`. Een pre-registratie legt vast: hypothese, nulhypothese, universum, periode, parameters, het aantal geplande trials, en de stop-criteria. De ID wordt gegenereerd uit een hash van deze inhoud en is verplicht bij elke gate-run.

**Stap 4 — De 6 lookahead-tests schrijven.**
Schrijf ze één voor één, elk met een eigen commit. Elke test moet **eerst aantoonbaar falen** op een bewust lekkend testmodel voordat je hem loslaat op de productiecode. Een test die nooit rood is geweest, is geen test.

**Stap 5 — Purged Walk-Forward implementeren.**
Bouw `walk_forward.py` met expliciete purge- en embargo-parameters. De embargo-lengte is een functie van de labelhorizon en staat in `conf/validation/`. Bewijs met `test_label_horizon_purge.py` dat overlappende labels tussen train en test aantoonbaar verwijderd zijn.

**Stap 6 — DSR-gate afdwingen.**
Wrap de bestaande, correcte DSR uit `backtest/metrics.py`. De wrapper eist `M` uit de trial-teller, `Y`, en de skew/kurtosis-correctie. Verboden: een default-waarde voor `M`. Ontbreekt `M`, dan crasht de gate.

**Stap 7 — SPA / Reality Check implementeren.**
Bouw `spa.py` voor multiple-testing-correctie bij vergelijking van meerdere strategieën tegen een benchmark. Deze gate is verplicht zodra meer dan één kandidaat tegelijk wordt geëvalueerd.

**Stap 8 — Gate-compositie en state machine.**
Bouw `gates.py` en de promotie-state-machine. Het `GateResult` is onveranderlijk en bevat: `git_sha`, `data_hash`, `config_hash`, `preregistration_id`, `M`, en per gate de uitkomst. Schrijven naar de ledger gebeurt atomair via de bestaande `hypothesis_ledger.py`.

**Stap 9 — Banned methods uitroeien.**
Draai `check_banned_methods.py`. Vervang elke unpurged CV in `src/` door Purged Walk-Forward. Laat de scanner daarna permanent in CI draaien.

**Stap 10 — CI dichttimmeren.**
Schrijf `.github/workflows/research_gates.yml`. De workflow blokkeert de merge — niet met een waarschuwing, maar met een falende required check.

**Stap 11 — Negatieve controle.**
Bouw een bewust lekkend model (bijv. een signaal dat `close.shift(-1)` gebruikt) in `tests/killgates/`. Bewijs dat de CI het blokkeert. **Als de gate dit model doorlaat, is de fase niet afgerond, ongeacht hoe groen de rest is.**

**Stap 12 — Documentatie sluiten en exit-rapport.**
Corrigeer `docs/model_risk_policy.md` (D-1) en schrijf `reports/phase2_exit_report.md`.

---

## CRITERIA & VALIDATIE (EXIT CRITERIA)

1. **PR's worden automatisch geblokkeerd** bij lookahead-fouten of DSR-falsificatie. Aangetoond met een testbranch die een lekkend model bevat en waarvan de merge aantoonbaar wordt tegengehouden.
2. **Alle 6 lookahead-tests bestaan, draaien en zijn aantoonbaar rood geweest** op een lekkend referentiemodel (D-1 gesloten).
3. **DSR-gate weigert te draaien zonder eerlijke `M`.** Geen default, geen fallback. Promotie vereist `p < 0.05` na correctie voor non-normaliteit en het totale aantal trials.
4. **Nul banned methods.** `check_banned_methods.py` levert nul treffers over `src/`. Geen unpurged K-Fold, geen shuffled splits.
5. **Ledger-integriteit.** Elke ledger-entry bevat een valide `git_sha`, `data_hash`, `config_hash` en `preregistration_id`. Een entry zonder één van deze velden crasht bij het schrijven.
6. **Pre-registratie verplicht.** Een gate-run zonder pre-registratie-ID crasht aantoonbaar.
7. **Historische claims geherclassificeerd.** De 4 "geaccepteerde" en 20 gefalsificeerde alpha-units uit sectie 3.2 staan als `INVALID — no certified data provenance` in het register.
8. **Purged Walk-Forward met embargo is de enige toegestane CV** in de promotiepijplijn; CPCV en PBO draaien uitsluitend als diagnostiek en hebben geen gate-bevoegdheid.

---

## REGELS & HANDLINGSINSTRUCTIES

- **Fail-fast compliance.** Een gate die niet kan draaien, is een gefaalde gate. Nooit "skip", nooit "warn", nooit een default-parameter. Nul `try/except` fallbacks.
- **100% PIT rigor.** Elke fold, elke scaler, elk label respecteert de tijdsordening. Een scaler die op de volledige sample is gefit, is een lookahead-lek en geen implementatiedetail.
- **Geen hardcoded variabelen.** Embargo-lengte, significantiedrempels, foldgroottes en horizon-parameters komen uit `conf/validation/`.
- **Bewijs door falen.** Elke gate moet aantoonbaar een bekend-slecht model afwijzen voordat je hem vertrouwt op een onbekend model. Groen zonder bewezen rood is waardeloos.
- **Onbewezen ≠ bewezen slecht (sectie 6).** Falsificeer een model niet omdat de infrastructuur ontbrak. Herlabel het naar `UNPROVEN — insufficient data` en laat het in de Research Track. Verwijderen mag alleen bij *bewezen slecht* onder een correct geconfigureerde baseline.
- **Append-only governance.** De ledger en het falsificatieregister worden nooit geëdit of opgeschoond. Correcties zijn nieuwe entries.
- **Atomaire commits.** `test(lookahead): add truncation invariance gate`, `feat(validation): enforce DSR with mandatory trial count M`, `feat(registry): add preregistration contract with stop criteria`, `ci(gates): block merge on lookahead or DSR failure`.
- **Geen alpha-onderzoek in deze fase.** Je bouwt de rechtbank, niet de verdachte. Baselines volgen in Phase 3.

---

## STARTINSTRUCTIE

> **Begin nu met Stap 1:** doorloop `docs/FALSIFICATION_REGISTER.md` en `registry/hypothesis_ledger.py`, en voeg voor elk historisch resultaat zonder gecertificeerde `data_hash` uit Phase 1 een invalidatie-entry toe met status `INVALID — no certified data provenance`. Rapporteer het totale aantal geherclassificeerde entries en commit als `docs(registry): invalidate historical results lacking certified data provenance`.
