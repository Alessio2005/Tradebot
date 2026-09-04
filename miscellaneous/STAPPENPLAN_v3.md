# STAPPENPLAN v3 — uitvoeringsvolgorde bij `ULTIMATE_GOAL_PROMPT.md`

> Voor de agent. Dit plan vertaalt het /goal-mandaat (v3) naar een exacte
> uitvoeringsvolgorde op déze codebase. §-verwijzingen zijn naar het mandaat.
> Wave-nummering loopt door vanaf de audit (laatste = Wave 19): dit plan begint
> bij **Wave 20**. Elke wave eindigt met een wave-log-entry + bijgewerkte
> gate-tabel — geen uitzonderingen.

---

## 0. LEESWIJZER & PARALLEL-MECHANICA

- **Sequentieel** = blokkerend: niet beginnen vóór de dependency klaar én
  geaccepteerd is.
- **Parallel** = mag gelijktijdig (subagents / achtergrond-jobs), op voorwaarde:
  1. **Ledger-discipline:** elke geteste config wordt atomair toegevoegd aan
     `artefacts/governance/hypothesis_ledger.json` (één writer; parallelle waves
     schrijven hun trials naar een eigen staging-file, merge bij wave-afsluiting).
  2. **Geen gedeelde staat:** parallelle units raken elkaars artefacts/configs niet.
  3. **Accept/archiveer-beslissingen** worden altijd serieel genomen (één
     beslismoment per wave-afsluiting), nooit in twee parallelle takken tegelijk.
- **Harde volgorde-wet:** simpel premium eerst OOS levend → pas dan ML-overlay
  (§5.6, anti-F12). Nooit andersom, ook niet "alvast parallel".

### Overzicht-DAG

```
FASE 0  fundament (W20)            ── blokkeert ALLES hierna
  ├─ 0.1 registers ─┐
  ├─ 0.2 PIT-datalaag ─┼─→ FASE 1  equities-units (W21–24)   ─┐
  ├─ 0.3 G4-factorlab ─┘     (XSMOM ∥ REV ∥ BAB) → (PEAD ∥ QUAL ∥ O/N)│
  └─ 0.4 crypto-rebaseline ──→ (loopt parallel met 0.2/0.3)          │
                              FASE 2  FX ∥ commodities (W25–28) ──────┼─→ FASE 3
                                                                      │   boek-
  FASE 3  RP-boek + stress-matrix + vol-target/Kelly (W29–30) ←───────┘   compositie
     └─→ FASE 4  ML-verdieping op geaccepteerde units (W31+)
            └─→ FASE 5  volledige G1–G10 + 14d shadow (W~33+)
                   └─ rood → terug naar expansielijst §5 (volgende unit-wave)
                   └─ groen → EINDE (enige stoptoestand)
```

---

## FASE 0 — FUNDAMENT (Wave 20; ~alles hier blokkeert de rest)

**Start hier. Niets uit Fase 1+ mag beginnen vóór 0.1–0.3 af zijn.**

### 0.1 Registers (sequentieel, eerst — duurt kort)
- Maak `artefacts/governance/hypothesis_ledger.json`: append-only, seed
  `total_n_hypotheses` met de eerlijke stand uit de audit (~2000+; reconstrueer
  uit auditsecties en `artefacts/hparams/*_best_score.json`). Schema: wave, unit,
  config-hash, datum, resultaat.
- Maak `docs/FALSIFICATION_REGISTER.md`: F1–F12 uit het mandaat §3, één regel per
  item met bewijslink. Append-only.
- **Done wanneer:** beide bestanden bestaan, ledger heeft een gedocumenteerde
  startstand, en een helper in `src/tradebot/registry/` kan atomair appenden
  (≤80 LOC CLI in `apps/`, R-6).

### 0.2 Point-in-time multi-markt-datalaag (parallel met 0.3 en 0.4)
- Nieuw: `src/tradebot/data/sources/` met één module per bron (R-4 ≤800 LOC):
  `stooq.py`, `kenfrench.py`, `edgar.py`, `fred.py`, `cboe.py`, `wiki_constituents.py`.
- Elke bron implementeert hetzelfde contract: fetch → Pandera-schema →
  **as-of-kolom = beschikbaarheidsmoment** (publicatie-lag expliciet, §7) →
  parquet onder `market_data_parquet/<markt>/`.
- Generieke as-of-join-utility in `src/tradebot/utils/time.py` (uitbreiden) +
  lookahead-test per bron in `tests/lookahead/` (G6 geldt vanaf dag één).
- `docs/DATA_REGISTER.md`: bron, URL, licentie (gratis), lag, dekking, kwaliteit,
  survivorship-status (G8).
- **Done wanneer:** elke bron een smoke-fetch + schema-validatie + lookahead-test
  groen heeft en in het data-register staat.

### 0.3 G4-factorlab (parallel met 0.2)
- Nieuw: `src/tradebot/risk/factor_alpha.py`: HAC/Newey-West-regressie,
  rapporteert α, t(α), p, loadings. Eén functie, hergebruikt door élke unit.
- Factorsets klaarzetten als data: Ken French (Mkt-RF, SMB, HML, RMW, CMA, MOM)
  via 0.2; AQR BAB (gratis download); crypto MKT+TSMOM (zelf construeren uit
  bestaande data — bestaat al deels in `research/true_alpha_gates.py`, promoveer
  naar `src/tradebot/`, R-2); FX dollar/carry/trend en commodity markt/carry/mom
  zelf construeren zodra 5.3/5.4-data er is.
- Unit-test met een bekende uitkomst (bv. de audit-§9-boekcijfers reproduceren:
  α ≈ +20%/jr, t ≈ 2.65).
- **Done wanneer:** test groen en de functie vanuit één aanroep een complete
  G4-regel voor de gate-tabel produceert.

### 0.4 Crypto-rebaseline (parallel met 0.2/0.3; alleen compute)
- Draai de bestaande validatie integraal opnieuw: `pytest tests/lookahead/`,
  determinisme-suite (R-5), daarna `research/wave_final_eval.py` en
  `research/multi_sleeve_combine.py` → officiële **Wave-20-baseline-gate-tabel**
  (verwacht: Sharpe ~1.0–1.2, CAGR ~50–60% @40% vol, G1 rood — dat is de start).
- Leg de vier bestaande crypto-units vast als geaccepteerde units in de ledger:
  ML-XS (`artefacts/tracks_breadth/`), LOWVOL, REVERSAL-k10, CARRY.
- **Done wanneer:** baseline-gate-tabel in het wave-log staat en bit-identiek
  reproduceert (zelfde cfg+seed).

---

## FASE 1 — EQUITIES (Waves 21–24; prioriteit 1, §5.2)

**Dependency:** Fase 0 volledig. Binnen de fase geldt het datalaag-pad:

### Wave 21 — universum & survivorship (sequentieel; blokkeert alle equity-units)
- Point-in-time S&P 500-constituents (Wikipedia-history) + Russell/NYSE-listings →
  universum-kalender (`market_data_parquet/equities/universe.parquet`): per datum
  de verhandelbare set, inclusief verwijderde namen.
- Stooq-OHLCV bulk-fetch voor het volledige (ook ex-)universum; gaten en
  delisted-dekking documenteren in het data-register (G8).
- **Done wanneer:** ≥15 jaar dagdata, universum-kalender getest (geen naam in de
  set vóór toevoegdatum — lookahead-test), dekking gerapporteerd.

### Waves 22a ∥ 22b ∥ 22c — data-lichte premia (PARALLEL, onderling onafhankelijk)
| Wave | Unit | Literatuur-prior | Data |
|------|------|------------------|------|
| 22a | XS-momentum 12-1 | Jegadeesh-Titman 1993 | alleen OHLCV |
| 22b | korte-termijn reversal (1m) | Jegadeesh 1990 | alleen OHLCV |
| 22c | low-vol/BAB | Frazzini-Pedersen 2014 | alleen OHLCV |

- Elk volgens de **per-unit-checklist (§9 hieronder)**: simpel, literatuur-conform,
  decielen-spread, dollar-neutraal, walk-forward OOS, kosten (commissie + spread +
  borrow-fee op shorts, hard-to-borrow uitgesloten), G4-regressie tegen Ken French
  (verwachting: BAB-unit moet α houden ná MOM/BAB-factoren — anders is het
  factor-loading en telt niet, §1.2).
- Code: `src/tradebot/alpha/eq_xsmom.py`, `eq_strev.py`, `eq_lowvol.py`.

### Waves 23a ∥ 23b ∥ 23c — data-zware premia (PARALLEL; dependency: EDGAR-ingest af)
| Wave | Unit | Prior | Extra data |
|------|------|-------|-----------|
| 23a | PEAD / earnings-drift | Bernard-Thomas 1989 | EDGAR filing-datums + surprises |
| 23b | quality/profitability | Novy-Marx 2013 | EDGAR fundamentals (PIT!) |
| 23c | overnight-vs-intraday | Lou-Polk-Skouras 2019 | open/close uit Stooq |

- EDGAR: as-of = **filing-timestamp**, nooit periode-einddatum (§7). PEAD-events
  via de bestaande event-study-machinerie (labeling op event-datum).

### Wave 24 — equities-tussenbalans (sequentieel; sluit Fase-1-kern af)
- Accept/archiveer-beslissing per unit (serieel), ledger-merge, correlatiematrix
  van geaccepteerde equity-units onderling + vs crypto-units (gewoon + stress-
  venster maart-2020/2022).
- **Go-criterium Fase 3-voorbereiding:** ≥3 geaccepteerde equity-units met
  paarsgewijs \|ρ\| < 0.3 t.o.v. elkaar én t.o.v. crypto-boek.

---

## FASE 2 — FX & COMMODITIES/RATES (Waves 25–28)

**Mag PARALLEL met Fase 1 starten zodra Fase 0 af is** — het zijn onafhankelijke
datapaden en aparte subagent-takken. Houd de beslismomenten wel serieel (§0).

### Wave 25 — FX-datapad + carry (sequentieel binnen FX-tak)
- FRED 3m-rates per G10-valuta + Stooq FX-dagkoersen → carry-signaal
  (rentedifferentieel), `src/tradebot/alpha/fx_carry.py` (Koijen 2018).
- Kosten: spread 0.5–2 bps majors; executieroute registreren (G10).

### Waves 26a ∥ 26b — FX trend + value (PARALLEL na 25-datapad)
- 26a: FX-TSMOM 12m (Moskowitz-Ooi-Pedersen 2012) — let op: F7/F8 golden voor
  crypto; FX/multi-asset-TSMOM is een ander, gedocumenteerd premium.
- 26b: PPP-value (OECD PPP-data, reversion).

### Waves 27a ∥ 27b — commodities/rates (PARALLEL; eigen datapad via Stooq-ETF/futures-continuaties)
- 27a: term-structure carry (Gorton-Rouwenhorst) + XS-momentum op commodity-mandje.
- 27b: bond-TSMOM via treasury-ETF's; VIX-term-structure (CBOE) als **conditioner**
  op equity-units (F5: géén standalone timer).

### Wave 28 — tussenbalans 2: cross-markt correlatiematrix, accept/archiveer serieel.

---

## FASE 3 — BOEKCOMPOSITIE & RISICO (Waves 29–30; STRIKT SEQUENTIEEL)

**Dependency:** ≥8 geaccepteerde units totaal (crypto 4 + nieuw ≥4). Niets hierin
mag parallel met unit-acceptatie — het boek bouwt op een bevroren unit-set.

### Wave 29 — RP-boek op stress-correlaties
- Risk-parity (inverse-vol) over alle geaccepteerde units — géén
  combination-tuning (F11); stacker komt pas in Fase 4 en moet RP verslaan.
- Stress-matrix (§5.5): gewichten toetsen op crisis-vensters; decorrelatie die
  alleen in rust bestaat telt niet.
- G4 op boek-niveau (multi-markt factorset gecombineerd).

### Wave 30 — vol-targeting + leverage + ruïne-preventie
- Vol-target 35–40% op het boek; fractional Kelly ¼–½; netting + no-trade-band +
  DD-cascade-de-grossing (het §11-recept uit de audit: Calmar 1.13→1.33).
- Venue-allocatie: Bybit-EU ≤10× spot / equities ~2–4× / FX-route — kapitaal zó
  verdelen dat target-vol binnen elke cap haalbaar is (§9 mandaat).
- Circuit-breaker-thresholds herijken op doel-vol (8%-DD-breaker is zinloos bij
  40% vol); documenteer de herijking (R-7 blijft actief).
- **Output:** eerste volledige multi-markt gate-tabel G1–G10. Verwachting op dit
  punt: Sharpe ~1.6–2.2 bij 10–15 units (§4-tabel) → G1 nog rood → door.

---

## FASE 4 — ML-VERDIEPING (Waves 31+; per familie PARALLEL)

**Dependency per unit: de simpele variant is geaccepteerd** (harde wet, §0).

- **31x (parallel per markt×premium-familie):** pooled XS-model volgens
  Wave-15-recept — rank-features, relative-winner target, purged+embargoed CV,
  AFML sample-weights, frozen-MDA, calibratie. Doel: IC van het premium verhogen
  (equities-analoog van crypto-IC +0.069). Nooit per-asset (F3/F10).
- **32 (sequentieel na 31):** meta-labeling sizing (p(win|state), alleen sizing)
  + stacker (`train/stack.py`) over units, conditioneel op regime. Acceptatie
  alleen als de stacker RP **OOS** verslaat; anders RP houden (F11).
- AdaptiveWalkForward (`alpha/adaptive_wf.py`) is de standaard-harness; PBO met
  S ≥ 50 per familie (Wave-19-les).
- Elke ML-variant = ledger-entry. Verwachte winst: Sharpe-units van 0.4–0.5 naar
  0.5–0.7 + betere sizing → boek richting 2.2–2.7.

---

## FASE 5 — GATE-RUN & SHADOW (sequentieel; herhaalt tot groen)

1. **Volledige G1–G10** op het eindboek: DSR met cumulatieve ledger-n_trials,
   PBO (CSCV, S≥50) op boek én units, lookahead-suite 100% (incl. alle nieuwe
   bronnen), determinisme, data-register-audit, TCA-stress.
2. **≥14 dagen shadow/paper** (G9): crypto-units via bestaande paper-engine;
   equities/FX-units via gesimuleerde forward-fills op live dagdata, afwijking
   vs backtest binnen tolerantie.
3. **Rood?** → terug naar de expansielijst (§5 mandaat): volgende unit-wave
   (meer premia, meer markten: small-caps, internationale aandelen via Stooq,
   meer commodity-mandjes, cross-market overlays §5.5). Het plan loopt door —
   per §2.3 bestaat er altijd een volgende wave.
4. **Groen?** → enige stoptoestand bereikt; afsluitend rapport + MRM-flow
   (`registry/promotion.py`, champion-challenger) richting live.

---

## 8. PARALLELLISERINGSREGELS (samenvatting)

| Activiteit | Parallel? | Waarom |
|---|---|---|
| Data-ingests van verschillende bronnen (0.2, 21, 25, 27) | ✅ | onafhankelijke I/O, eigen schema's |
| Unit-waves binnen zelfde markt, data-licht (22a/b/c) | ✅ | geen gedeelde staat; eigen ledger-staging |
| FX-tak naast equities-tak (Fase 2 ∥ Fase 1) | ✅ | aparte datapaden |
| ML-overlays per familie (31x) | ✅ | dependency per unit al vervuld |
| Crypto-rebaseline naast dataloog-bouw (0.4 ∥ 0.2/0.3) | ✅ | alleen compute op bestaande pipeline |
| Accept/archiveer-beslissingen | ❌ serieel | één beslismoment, ledger-integriteit |
| ML-overlay vóór acceptatie simpel premium | ❌ verboden | §5.6 / F12 |
| Fase 3 (boek) tijdens lopende unit-acceptaties | ❌ | boek vereist bevroren unit-set |
| Vol-target/leverage vóór RP-boek | ❌ | hefboom op ongedefinieerd boek = betekenisloos |
| Gate-run/shadow vóór Fase 3–4 af | ❌ | meet anders een tussenproduct |
| Harvest-/combinatie-parametersweeps "erbij" | ❌ verboden | F11/F12: tuning is geen edge-bron |

## 9. PER-UNIT CHECKLIST (template voor elke unit-wave)

1. Hypothese + literatuur-citaat in wave-log (prior verplicht, §5).
2. Data via 0.2-contract (PIT, lag, register) + lookahead-test in `tests/lookahead/`.
3. Simpele implementatie in `src/tradebot/alpha/<markt>_<premium>.py` (R-2/R-4).
4. Walk-forward OOS (AdaptiveWalkForward-harness), netto kosten incl. stress.
5. G4-factorregressie (0.3-functie) → α, t, p in de wave-log.
6. Correlatie vs alle geaccepteerde units (gewoon + stress-venster).
7. Ledger-update (alle trials), DSR-herberekening.
8. **Accept-criteria:** netto OOS Sharpe ≥ 0.4 · t(α) ≥ 2 · \|ρ\| < 0.3 vs boek ·
   G6/G7-tests groen. Anders: archiveren met bewijs (register §3 aanvullen).
9. Wave-log-entry + bijgewerkte gate-tabel. → volgende wave.

## 10. DEFINITION OF DONE PER FASE

| Fase | Klaar wanneer |
|---|---|
| 0 | registers live, dataloog + G4-lab getest, baseline-gate-tabel reproduceert bit-identiek |
| 1 | ≥3 geaccepteerde equity-units (\|ρ\|<0.3), tussenbalans-matrix in wave-log |
| 2 | ≥2 geaccepteerde FX/commodity-units, cross-markt-matrix |
| 3 | RP-boek + vol-target/Kelly/ruïne-preventie staan; volledige gate-tabel gedraaid |
| 4 | ML-overlays gevalideerd per familie; stacker vs RP beslist op OOS-bewijs |
| 5 | G1–G10 volledig groen ná ≥14d shadow — **de enige eindtoestand** |

> Startcommando voor de agent, letterlijk: **begin bij Fase 0, stap 0.1.**
> Na elke wave: log, gate-tabel, beslissing, volgende wave. Niet stoppen vóór
> Fase 5 groen is (98% ≠ 100%; een plafond = redirect, §2.3).
