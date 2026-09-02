# WAVE LOG — mandaat v3 (vanaf Wave 20)

> **GEARCHIVEERD — historisch document.** Dit is een verslag van Waves 1 t/m 28 en beschrijft de toestand van toen. Het wordt NIET bijgewerkt: de paden en artefacten die het noemt, zijn die van die periode en bestaan grotendeels niet meer. Voor de huidige toestand, zie `docs/PROJECT_STATE.md`.
> *Als historisch gemarkeerd op 2026-09-01 (Phase 7/8, Stage E-3).*

> Append-only. Per wave: hypothese (+ literatuur-prior), implementatie,
> resultaat, beslissing, beste-tot-nu-toe OOS-stand, volgende aanvalslinie,
> en de VOLLEDIGE gate-tabel G1–G10 (nooit losse highlights). Dit log is het
> bewijs dat het programma niet gestopt is (§2).

---

## Wave 20 — Fase 0: fundament (2026-06-10) — STATUS: deels af, rebaseline open

**Hypothese / doel.** Geen edge-hypothese; infrastructuurwave (stappenplan §0).
Registers, PIT-datalaag en G4-factorlab zijn randvoorwaarde voor elke
volgende unit-wave; zonder eerlijke cumulatieve DSR-teller is elk later
resultaat onboekbaar.

**Geleverd (0.1 — registers):**
- `artefacts/governance/hypothesis_ledger.json` — append-only, seed
  **total_n_hypotheses = 2363** (geïtemiseerde reconstructie: 2000 audit-§10/§11
  + W14:204 + W15:21 + W16:8 + W17:49 + W18:1 + W19:80; conservatieve vloer).
- `docs/FALSIFICATION_REGISTER.md` — F1–F12, bindend, append-only.
- `src/tradebot/registry/hypothesis_ledger.py` (atomair, staging+seriële merge,
  duplicaat-afwijzing, append-only-guards) + `apps/ledger_append.py` (R-6 CLI)
  + `tests/unit/test_hypothesis_ledger.py`.

**Geleverd (0.2 — PIT-datalaag):**
- `src/tradebot/data/sources/`: `base.py` (contract: fetch → validatie →
  `asof_ts` = beschikbaarheidsmoment), `stooq.py`, `kenfrench.py`, `edgar.py`
  (asof = acceptance-timestamp, nooit periode-einde), `fred.py` (release-lag
  VERPLICHT per serie), `cboe.py` (VIX-TS, F5: alleen conditioner),
  `wiki_constituents.py` (PIT S&P-500-kalender = survivorship-fix).
- `tradebot.utils.time.asof_join` — de enige gesanctioneerde PIT-merge.
- `tests/lookahead/test_pit_sources.py` — contract-, join-, lag- en
  universum-kalender-guards (synthetisch, G6).
- `docs/DATA_REGISTER.md` (G8) — alle bronnen geregistreerd; status per bron.

**Geleverd (0.3 — G4-factorlab):**
- `src/tradebot/risk/factor_alpha.py` — `factor_residual_alpha()` (HAC/
  Newey-West; α, t, p, loadings, R², n; `gate_row()` voor de gate-tabel),
  `G4_FACTORSETS` per markt (mandaat §10 G4). Gepromoveerd uit
  `scripts/true_alpha_gates.py::gate_g6` (R-2).
- `tests/unit/test_factor_alpha.py` — synthetische bekende-uitkomst-tests
  (deterministisch, R-5) + audit-§9-reproductie (skipif zonder cached panel).

**OPEN (0.4 — crypto-rebaseline) — BLOKKEREND voor Fase 1+:**
De compute-omgeving was deze sessie onbeschikbaar (sandbox: onvoldoende
schijfruimte). Vereiste run (lokaal):
```
pytest tests/ -q                         # incl. nieuwe 0.1/0.2/0.3-tests
python scripts/wave_final_eval.py artefacts/broad_perp_daily_close_WIDE.parquet 10 0.35
python scripts/multi_sleeve_combine.py
```
Daarna: 4 bestaande crypto-units (ML-XS, LOWVOL, REVERSAL-k10, CARRY) als
`accepted` in de ledger; baseline-gate-tabel bit-identiek reproduceren.

**Gate-tabel (stand = audit-§9/§17-cijfers; HERBEVESTIGING VEREIST in 0.4 —
dit zijn overgenomen, niet deze wave herdraaide waarden):**

| # | Gate | Drempel | Stand (audit) | Verdict |
|---|------|---------|---------------|---------|
| G1 | Netto CAGR OOS | ≥100% | ~47–55% @40% vol | **ROOD** |
| G2 | DSR @ eerlijke n_trials (nu 2363) | >0.95 | ≤0.54 | **ROOD** |
| G3 | PBO (CSCV, S≥50) | <0.20 | 0.51 (harvest) | **ROOD** |
| G4 | residual α, p<0.05 (HAC) | pass | crypto-boek: +20%/jr, t=2.65, p=0.008 | **GROEN (boek, crypto)** |
| G5 | Sharpe per regime-bucket | >0 ×4 | alle 4 positief | **GROEN** |
| G6 | Lookahead-suite | 100% | nieuwe PIT-tests toegevoegd; integrale run open | **OPEN (0.4)** |
| G7 | Determinisme | pass | herbevestiging open | **OPEN (0.4)** |
| G8 | Data-integriteit/register | pass | register live; smoke-fetches open | **OPEN** |
| G9 | Walk-forward + ≥14d shadow | in tol | niet gestart | **ROOD** |
| G10 | Executie-realisme per unit | pass | crypto: ok; nieuwe markten: n.t.b. | **OPEN** |

**Beslissing.** Fundament staat (code + registers); wave sluit pas na 0.4-run.
Eén rode gate = door (§2) — en er zijn er drie plus vier open.

**Beste-tot-nu-toe OOS-stand.** Netto Sharpe ~1.0–1.2 / ~47–55% CAGR @40% vol
(crypto-MN-boek, audit §9/§11) — tussenstand, geen acceptatie.

**Volgende aanvalslinie.** (1) 0.4-rebaseline draaien zodra compute er is;
(2) smoke-fetches 0.2-bronnen; (3) Wave 21: PIT-universum + Stooq-bulk
(equities, §5.2 prioriteit 1); daarna 22a∥22b∥22c (XSMOM 12-1
Jegadeesh-Titman 1993 ∥ STREV Jegadeesh 1990 ∥ BAB Frazzini-Pedersen 2014).

### Addendum 2026-06-10 (zelfde dag) — W21/W22-code voorbereid, compute nog dicht

Sandbox bleef onbeschikbaar na vrijmaken van ruimte (provisioning-fout
herhaalt; vermoedelijk app-herstart of ruimte op de systeemschijf nodig).
Conform §2.4 is de wachttijd benut voor Fase-1-code (0.1–0.3 waren af; dit
zijn implementaties, GEEN geteste hypotheses — er is dus niets aan de ledger
toegevoegd; trials tellen pas bij de eerste echte run):
- **Harness:** `src/tradebot/alpha/xs_unit.py` — gedeelde XS-machinerie
  (decile-gewichten $-neutraal/gross 1, vaste maand-rebalance, kostenmodel
  commissie+halve spread+borrow, `w.shift(1)`-causaliteit, F12: geen
  harvest-tuning).
- **Units:** `eq_xsmom.py` (12-1, JT1993) · `eq_strev.py` (1m, Jegadeesh
  1990) · `eq_lowvol.py` (vol-rank BAB-lite, FP2014; G4-waarschuwing: moet α
  houden ná BAB-factor).
- **Universum (W21):** `src/tradebot/data/equity_universe.py` (wiki→stooq,
  PIT-kalender, coverage-rapport = G8-bewijs van de rest-survivorship-gap) +
  `apps/build_equity_universe.py` (R-6).
- **Eval-orchestrator:** `apps/run_eq_units.py` (summaries, G4-regels,
  correlaties, ledger-staging).
- **Tests:** `tests/unit/test_eq_units.py` (harness-wiskunde handmatig
  nagerekend, signaaldefinities, kosten, membership) +
  `tests/lookahead/test_eq_units_causality.py` (truncatie-invariantie,
  future-perturbatie, determinisme R-5).

**Run-volgorde zodra compute leeft:** (1) `pytest tests/ -q`; (2) 0.4-
rebaseline (zie boven); (3) smoke-fetches; (4) `python
apps/build_equity_universe.py build`; (5) `python apps/run_eq_units.py
--stage artefacts/governance/hypothesis_ledger_staging_w22.json`; (6)
accept/archiveer serieel + ledger-merge + gate-tabel.

### Addendum 2 (2026-06-10) — compute-blokkade gediagnosticeerd: virtualisatie uit

Na vrijmaken van schijfruimte faalt de sandbox-VM definitief met
**HYPERVISOR_VIRT_DISABLED**: hardware-virtualisatie staat uit op de host.
Fix (gebruiker): virtualisatie aanzetten (BIOS/UEFI: Intel VT-x / AMD SVM;
Windows: "Virtual Machine Platform"/Hyper-V aan, daarna herstart).

Mitigatie zodat de wave niet op de VM wacht: **`scripts/wave20_runner.py`**
— one-shot evidence-runner voor de volledige keten (pytest → rebaseline 2×
met bit-identiek-check (G7) → multi_sleeve_combine → smoke-fetches alle
PIT-bronnen → W21-universumbouw → W22-unit-evals met ledger-staging).
Logs naar `artefacts/wave20_runlog/`, status naar `status.json`. De runner
verzamelt alleen bewijs; accept/archiveer en ledger-merge blijven serieel.
Lokaal draaien: `python scripts/wave20_runner.py` (of `--quick` voor een
25-namen-smoke van de universumbouw).

### Addendum 3 (2026-06-10) — 0.4 REBASELINE GEDRAAID: bit-identiek bevestigd

Eerste runner-run (lokaal, gebruiker) — resultaten:
- **Rebaseline 2× bit-identiek = TRUE (G7 pass voor de eval-keten).**
- Boek (99 namen, 10bps, vol-target 35%→40.3% realized): unlevered Sharpe
  1.06; vol-targeted **Sharpe 1.15, CAGR +46.7%**, avg gross 2.4×.
- G2 DSR: 0.76@N12 / 0.54@N50 / 0.35@N200 — rood bij eerlijke teller (2363+).
- G3 PBO 0.55 (S=9 — indicatief, onder Bailey-vloer; volledige CSCV S≥50 in
  Fase 5). G4-β marginaal (max 0.144/0.170; 97–98% <0.10). G5 alle vier
  buckets positief (zwakste bear/lovol +0.23). **G6 α=+20.2%/jr, t=2.65,
  p=0.008 — exact audit-§9: baseline reproduceert.**
- multi_sleeve_combine bevestigt sleeve-stand + √N-plafond (~4 sleeves,
  voor Sharpe 3 zijn er ~43 nodig → mandaat-as: breedte over markten).
- **Ledger bijgewerkt:** rebaseline-entry + 4 crypto-units `accepted`
  (ML-XS, LOWVOL, REVERSAL-k10, CARRY); total_n_hypotheses 2363→**2368**.

Gefaald in run 1 + fixes (alle gerepareerd, herrun vereist):
1. pytest-collectie: `tests/unit/test_audit_fixes.py` importeerde
   `scripts.paper_trade_report` als package → pad-robuuste importlib-load.
2. stooq 404 op custom UA → browser-UA + stooq.pl-fallback + 0.4s sleep.
3. FRED ReadTimeout @60s → timeout 180s.
4. wiki-parser StopIteration op tabel-layout → robuuste tabel-detectie.

Wave 20 sluit zodra de herrun pytest groen toont
(`python scripts/wave20_runner.py --skip-rebaseline`); 0.4-kern is binnen.

### Addendum 4 (2026-06-10) — run 2 verwerkt: 3 pytest-fails + stooq-blokkade

Run 2: pytest bijna groen (alle 0.1–0.3- en W22-guards slaagden); wiki ✓
(1030 events), kenfrench/cboe/edgar ✓. Gefaald + fixes:
1. `test_xsmom_ranks...` — eigen test was ruis-bros (drift-spacing 0.0002 <<
   ruis); vervangen door exacte identiteits-check + Spearman-rangcorrelatie.
2. `CHANGELOG.md` bleek LEEG (2 wave-13-doc-tests rood) — gereconstrueerd
   uit audit/data-dictionary/code-annotaties (AUDIT A-1/D-2/E-4).
3. **Stooq wholesale geblokkeerd** (404 op .com én .pl, vanaf twee
   netwerken; ook hier geverifieerd) — geen UA-kwestie. Gevolg: de
   W21-bulk-fetch hing ~uur op honderden 404-retries. Fixes: fail-fast
   (abort na 8 opeenvolgende misses zonder enige hit) + **yfinance-fallback**
   (`data/sources/yfinance_backup.py`, ToS-caveat geregistreerd, mandaat
   §5.2 "alleen aanvulling") + `build_universe(source="auto")`.
4. FRED 504 op fredgraph.csv — fallback naar `/data/<SERIES>`-tabel
   (endpoint hier geverifieerd werkend, data t/m 2026-06-08).
Survivorship-consequentie yfinance-route: delisted-dekking nóg dunner dan
Stooq → de G8-coverage-rapportage (failed/ontbrekende ex-leden) wordt het
bindende bewijsstuk; accepted units krijgen een survivorship-stress
(fetched-only vs vollledige ledenlijst) vóór acceptatie. Stooq blijft
geregistreerd als primaire bron zodra de host weer levert.

---

## Wave 20 — GESLOTEN (2026-06-10, run 3): pytest 100% groen

Run 3: **pytest volledig groen** (alleen verwachte skips), 5/6 bronnen
verified (stooq extern geblokkeerd, fallback actief). Daarmee is Fase 0
compleet: registers ✓, PIT-datalaag ✓, G4-lab ✓, rebaseline bit-identiek ✓.
Gate-tabel ongewijzigd t.o.v. addendum 3 (G1/G2/G3/G9 rood — door per §2).

## Wave 21 — universum GEBOUWD; Wave 22 run 1 — VOID (data-artifact)

**W21 (yfinance-fallback):** 872 leden-ooit (1030 PIT-events), 684 gefetcht,
188 failed (≈ delisted/renamed = de residuele survivorship-gap, G8),
272 unknown-add-date; panel 16.218 dagen × 667 namen; ~190 verhandelbare
namen/dag. Done-criteria: ≥15 jr ✓, lookahead-getest ✓, dekking
gerapporteerd ✓. **Herbouw nodig met adjusted prices (zie hieronder).**

**W22 run 1 = VOID — silent killer gevangen:** "+16078% in 2012" en
corr(XSMOM,LOWVOL) = −0.92 zijn split-artifacts: yfinance raw close
(auto_adjust=False) maakt van elke split een nep-rendement van ±90%+.
Lessen geboekt; fixes: (1) auto_adjust=True (split-safe, total-return-achtig,
geregistreerd), (2) glitch-guard in de panel-loader (|1d-move| > 300% →
NaN + rapportage), (3) evaluatievenster vanaf 2000-01-01 (pre-1990 was het
boek per constructie leeg, 1990–2000 lidmaatschap onbetrouwbaar — de
0.0-jaren vervuilden elke statistiek), eval_start in de config-hash.
**Ledger: +3 trials (archived, void) → total_n_hypotheses = 2371.**
Staging-file geroteerd; herrun staged met nieuwe config-hash.

## Wave 21 — GESLOTEN; Wave 22 run 3 (eerste geldige eval) + chief-diagnose

**W21 definitief:** panel herbouwd met adjusted prices; glitch-guard maskeerde
1469 punten (0.027% van cellen — gerapporteerd, G8); 667 namen, ~190
verhandelbaar/dag, eval-venster 2000+. Universum-kalender lookahead-getest.
Survivorship-gap: 188 failed + delisted-afwezigheid yfinance — bindend
G8-bewijs in `universe_coverage.json`; stress verplicht vóór unit-acceptatie.

**W22 run 3 (geldig, 2000–2026 netto):**
| Unit | net Sharpe | gross | Diagnose |
|---|---|---|---|
| eq_xsmom_12_1 | −0.34 | −0.27 | premium zelf zwak/afwezig in large-caps post-2000 óf constructie-issue → G4-regressie op MOM beslist (hoge MOM-loading + α≈0 = getrouwe bouw, zwak premium → archiveren; lage loading = bouwfout) |
| eq_strev_1m | **+0.39** | 0.56 | reëel signaal, kosten eten 30%; nét onder de 0.40-lat — beslissing na G4 (let op: 0.39 ≠ 0.40, geen afronding naar "goed genoeg") |
| eq_lowvol v1 | −0.82 | −0.79 | **constructie-misspecificatie**: $-neutraal vol-rank = structureel short-beta in stijgende markt; FP2014 vereist beta-genivelleerde legs. Falsifieert het premium NIET |
Correlaties (gezond na datafix): xsmom↔strev −0.18, xsmom↔lowvol +0.44,
strev↔lowvol −0.27.

**Acties:** (1) `eq_lowvol` v2 = BAB-leg-leverage (`beta_panel` in de
XS-harness, floor 0.25, NaN→1; getest in `test_beta_neutral_leg_scaling`) —
constructiecorrectie binnen dezelfde hypothese, nieuwe config-hash, telt als
trial; (2) FF5+MOM-factorset via `apps/fetch_factors.py` → G4-regels draaien
automatisch mee in de runner (BAB-factor nog TODO — gate-row gelabeld);
(3) ledger +3 (run-3-evals, interim) → **total_n_hypotheses = 2374**.
Accept/archiveer-beslissingen vallen serieel bij W22-afsluiting, ná G4 en
lowvol-v2.

---

## Waves 22 + 23c — GESLOTEN (2026-06-10): alle equity-OHLCV-premia gearchiveerd

**Seriële beslissingen (chief), bewijs in `w22_g4_diag.log` / `w23c_eval.log`:**
| Unit | net Sharpe | G4-diag (zelfgebouwd MKT+UMD, HAC) | Besluit |
|---|---|---|---|
| eq_xsmom_12_1 | −0.34 | UMD-loading 1.00 (tautologisch eigen construct); net α = −kosten | **ARCHIVED → F13** |
| eq_strev_1m | +0.39 | α +0.9%/jr, t=0.77 — geen residual alpha | **ARCHIVED** (lat 0.40 én t(α)≥2 gemist; heropening: residual/industry-adjusted reversal, DLS2014) |
| eq_lowvol v2 | −0.12 | restbèta +0.075 → lever werkte; α −2%/jr t=−1.1 | **ARCHIVED → F14** (premium afwezig, geen artifact) |
| eq_overnight_1m | −0.60 | α −3.0%/jr t=−3.01 (signif. negatief) | **ARCHIVED → F15** (sign-flip verboden, §1.3) |
Ledger: **total_n_hypotheses = 2378** (incl. W23c-merge). Sandbox-compute
werkt; alle evals hierboven door de agent zelf gedraaid. Sandbox-EGRESS is
volledig geblokkeerd → netwerk-ingests blijven één user-commando.

**Chief-conclusie (§2.3-redirect, geen exit):** het S&P-large-cap-universum
(~190 PIT-namen/dag, mega-caps) bevat geen netto verhandelbare plain-OHLCV-
premia 2000–2026. Consistent met de literatuur (premia leven in breedte/
small-caps) en met F10-logica. Deelruimte "equities × OHLCV-prijs-premia ×
S&P-large-caps" = GESLOTEN. Open vervolgpaden, in volgorde: (1) **FX-carry
(W25)** — FRED-rates + FRED DEX-spots, bron verified, unit-code klaar;
(2) **EDGAR-premia (W23a/b: PEAD, quality)** — fundamentals zijn een ándere
informatiebron dan prijs (F6-conform); (3) breder equity-universum zodra
Stooq weer levert (delisted small/mid-caps).

**Fase-1-status:** 0 geaccepteerde equity-units (go-criterium ≥3 niet
gehaald) — de expansielijst verplicht door naar 5.3 (FX) parallel met de
EDGAR-tak. Gate-tabel ongewijzigd (G1–G3, G9 rood; crypto-boek = beste
stand: Sharpe ~1.15 / +47%).

---

## Waves 23a/b + 24 — GESLOTEN (2026-06-10): equities-large-caps definitief uitgeput

Volledig agent-side gedraaid (sandbox). Formele G4 nu tegen échte Ken
French FF5+MOM (`factors_daily.parquet`, 1963–2026).

| Unit | net Sharpe | G4 (FF5+MOM, HAC) | Besluit |
|---|---|---|---|
| eq_xsmom_12_1 (herbevestiging) | −0.34 | MOM-loading **+0.42 (t=53), R²=0.77** → bouw getrouw; α −4.1% t=−5.4 | F13 definitief (premium netto niet oogstbaar hier) |
| eq_strev_1m (herbevestiging) | +0.39 | **α +3.1%/jr t=2.47 PASS** — premium bestaat, Sharpe-lat niet | archived; residual-route geprobeerd → F16 |
| eq_strev_resid_1m (W24, DLS2014) | −0.35 | gross −0.15 (raw was +0.56!); α t=−1.8 | **ARCHIVED → F16** |
| eq_quality_gpa (W23b, PIT-EDGAR) | −0.25 | RMW +0.114 (t=11.8) = getrouw; α t=−1.65 | **ARCHIVED → F17** |
| eq_pead_ar3 (W23a, acceptance-anchor) | −0.18 | α t=−0.6, R² 0.04 | **ARCHIVED → F18** |

Ledger: **total_n_hypotheses = 2381.** Coverage W23b: 319/667 tickers met
GrossProfit (financials vallen uit — G8 genoteerd). Eval-start fundamentals
2010 (XBRL betrouwbaar ~2009+).

**Incident (R-8):** `xs_unit.py` raakte tweemaal afgekapt door een
schrijf-collision/sync-defect tussen file-tool en mount; hersteld en
herbevestigd met 45/45 groene unit+lookahead-tests vóór de W23a/b-evals.
De eerdere evals draaiden aantoonbaar op het intacte bestand.

**Volgende aanvalslinies (volgorde):** (1) FX-carry-eval zodra de
chunked-FRED-heringest gedraaid is (datafout gevonden: daily FRED-series
waren stilzwijgend afgekapt op ~966 rijen — fix + staleness-alarm in
`fred.py`); (2) crypto-uitbouw §5.1 met LOKALE data (intraday-seizoens-
unit op bestaande info-bars — geen netwerk nodig); (3) equities alleen
nog via een breder universum (small/mid-caps, PIT) zodra een bron levert.

---

## Wave 22 — GESLOTEN (2026-06-10): 0/3 accepts; 2 scoped falsificaties

**Lowvol-v2 (beta-levered, FP-conform):** net Sharpe **−0.12** (v1 −0.82 —
de +0.70 delta bevestigt de beta-bleed-diagnose; corr met xsmom 0.44→0.23).
Maar ook correct gebouwd leeft het premium hier niet netto.

**Seriële beslissingen (alle drie via ledger, run in sandbox, 42/42 tests
groen incl. nieuwe beta-scaling-guard):**
- `eq_xsmom_12_1` → **ARCHIVED** (F13, scoped): net −0.34, 9/27 jaren +.
- `eq_lowvol` v1+v2 → **ARCHIVED** (F14, scoped): −0.82 / −0.12.
- `eq_strev_1m` → **ARCHIVED, lat niet gehaald** (géén falsificatie): +0.39 <
  0.40 én decay-staart 2024/25/26 −3.6/−6.6/−7.2% — Wave-18-patroon.
  Heropening: residual/industry-adjusted reversal (Da-Liu-Schaumburg 2014).

**Chief-lezing:** het gratis large-cap-S&P-universum (~190 namen/dag) draagt
de drie klassieke OHLCV-premia netto niet (consistent met McLean-Pontiff
post-publicatie-decay en de bekende small-cap-concentratie van deze premia).
Dit falsifieert de premia NIET — het sluit deze deelruimte (§2.3) en stuurt
naar: event-driven premia in large caps (PEAD — dáár leeft het wel in large
caps), overnight-split (LPS2019 — expliciet large-cap-gedocumenteerd), en
Fase 2 (FX/commodities). G4-factordata (Ken French) blijft TODO: sandbox-
egress geblokkeerd; loopt mee met de eerstvolgende run met netwerk.

**Ledger: total_n_hypotheses = 2377** (merge via `apps/ledger_append.py` —
helper werkt; unlink op mount faalt, staging geneutraliseerd als .merged.bak).

**Gate-tabel (volledig, stand na W22):**

| # | Gate | Drempel | Stand | Verdict |
|---|------|---------|-------|---------|
| G1 | Netto CAGR OOS | ≥100% | boek = crypto-MN ~47% @40% vol; 0 equity-units toegevoegd | **ROOD** |
| G2 | DSR @ n=2377 | >0.95 | ≤0.54 (boek) | **ROOD** |
| G3 | PBO (CSCV, S≥50) | <0.20 | 0.51–0.55 (crypto-harvest) | **ROOD** |
| G4 | residual α (HAC) | pass | crypto-boek PASS (t=2.65); equities: factordata geblokkeerd → n.b. | **DEELS / OPEN** |
| G5 | Sharpe per regime | >0 ×4 | crypto-boek: alle 4 + | **GROEN** |
| G6 | Lookahead-suite | 100% | pytest 100% groen (incl. PIT/eq-guards) | **GROEN** |
| G7 | Determinisme | pass | rebaseline bit-identiek; R-5-tests groen | **GROEN** |
| G8 | Data-integriteit | pass | register live; glitch-guard; survivorship-gap gedocumenteerd; stress nog te draaien | **DEELS** |
| G9 | WF + 14d shadow | in tol | niet gestart | **ROOD** |
| G10 | Executie-realisme | pass | crypto ok; equity-units gearchiveerd (moot) | **GROEN (huidige boek)** |

→ Volgende wave: **23c overnight-split (offline draaibaar, nu)**; daarna 23a
PEAD (EDGAR — netwerk vereist), Fase 2 FX-carry (FRED — netwerk vereist).

---

## Wave 26 — crypto intraday klok-seizoenaliteit (2026-06-11) — GESLOTEN: F19

**Hypothese + prior.** Uur-van-dag/dag-van-week/funding-venster-effecten op
USDT-perps zijn netto oogstbaar (horizon ≥ uren, geen HFT). Priors: Eross,
McGroarty, Urquhart & Wood (2019, RIBAF) — intraday BTC-seizoenaliteit;
Baur, Cahill, Godfrey & Liu (2019) — trading around the clock; practitioner-
lore rond 8h-funding-timestamps (klok-mechanisme; onderscheiden van het
F1-gefalsifieerde funding-VALUE-signaal).

**Implementatie.** `scripts/w26_seasonality_diag.py`: 6 perps, 5s→1h
(43.681 uurbars, 2021-06→2026-05, cache `artefacts/crypto_hourly.parquet`),
EW-markt-uurreturns; diagnostiek vóór unit-bouw (wave-protocol: bruto effect
moet de kostenlat halen vóór er een unit wordt gebouwd).

**Resultaat (vol uitgeschreven in `w26_diag.log`):**
- Uur-van-dag: max t = +2,11 (h04 UTC, +3,2 bps/u) = max over 24 tests →
  niet significant na multiple-comparison; per-jaar-tekens instabiel.
- Dag-van-week: Wed +45 bps/d (t=1,57), Thu −41 (t=−1,67) — niets ≥ |2|.
- Funding-venster (uur vóór/na 00/08/16 UTC): t = −0,09 / −0,47 — dood.
- Kostenlat: beste aaneengesloten uurvenster (in-sample max over 288
  configs, h20+10u) = **+7,4 bps/dag bruto vs 20 bps/dag taker-round-trip**
  → factor ~3 onder de lat. Geen unit gebouwd.

**Beslissing.** ARCHIVED → **F19** (register bijgewerkt; heropening alleen
bij aantoonbare maker-execution — F2-conditie — of ≥50-naams perp-universum
met XS-klok-effecten). Ledger: +321 trials (24 HOD + 7 DOW + 2 funding +
288 vensterscan) → **total_n_hypotheses = 2702**.

**Beste-tot-nu-toe OOS-stand.** Onveranderd: crypto-MN-boek Sharpe 1,15 /
+46,7% netto CAGR @40% vol (W20-rebaseline, bit-identiek).

**Volgende aanvalslinie.** W25 FX-carry (heringest loopt — chunked-CSV-fix);
daarna 26a/b FX-TSMOM + PPP-value; spot-perp-basis (§5.1) zodra spot-klines
geregistreerd en gefetcht zijn; commodities-bron registreren (Stooq-blokkade).

**Gate-tabel (volledig, stand na W26):**

| # | Gate | Drempel | Stand | Verdict |
|---|------|---------|-------|---------|
| G1 | Netto CAGR OOS | ≥100% | crypto-MN-boek +46,7% @40% vol; 0 nieuwe units | **ROOD** |
| G2 | DSR @ n=2702 | >0,95 | ≤0,54 (boek; teller verhoogd 2381→2702) | **ROOD** |
| G3 | PBO (CSCV, S≥50) | <0,20 | 0,51–0,55 (indicatief; volledige run Fase 5) | **ROOD** |
| G4 | residual α (HAC) | pass | crypto-boek PASS (t=2,65); geen nieuwe units | **DEELS** |
| G5 | Sharpe per regime ×4 | >0 | crypto-boek: alle 4 positief | **GROEN** |
| G6 | Lookahead-suite | 100% | suite groen; W26 = diagnostiek op causale uurbars | **GROEN** |
| G7 | Determinisme | pass | W20-rebaseline bit-identiek; diag deterministisch | **GROEN** |
| G8 | Data-integriteit | pass | register actueel; FX-heringest LOOPT (chunked fix); Stooq geblokkeerd | **DEELS** |
| G9 | WF + 14d shadow | in tol | niet gestart (Fase 5) | **ROOD** |
| G10 | Executie-realisme | pass | crypto ok; W26 expliciet op taker-kosten afgewezen | **GROEN (huidig boek)** |

→ Eén rode gate = door (§2). Volgende wave: W25-FX-carry-eval.

---

## Wave 27 — cross-asset TSMOM (2026-08-10) — GESLOTEN: ARCHIVED (lat/G4-strict niet gehaald)

**Hypothese + prior.** De managed-futures-premie (Moskowitz, Ooi & Pedersen 2012,
JFE 104(2) 228-250) is netto oogstbaar over markten heen. Dit is de expliciete
heropeningsconditie van **F7** ("trend in ANDERE asset-klasse") en **F8**
("in equities/FX/commodities is XSMOM een ander, gedocumenteerd premium"), en
staat als §5.4 op de expansielijst. Vaste literatuurparametrisering, vóór de
eerste backtest bevroren (F12): 252d lookback, sign(momentum), 60d inverse-vol,
maandelijkse herbalancering, gross = 1.

**Twee datavondsten die het ontwerp bepaalden (G8).**

1. **Gratis futures-continuaties zijn onbruikbaar voor P&L.** Yahoo `=F` is
   front-month en NIET terug-aangepast: de roll-yield ontbreekt. Gemeten tegen
   de werkelijk doorrollende vehikels 2010-2026:
   `CL=F +3.38%/jr vs USO -3.63%/jr` -> **+7.01%/jr te veel**;
   `NG=F -5.02%/jr vs UNG -30.07%/jr` -> **+25.05%/jr te veel**.
   Een TSMOM-backtest daarop boekt een orde van grootte méér dan het gezochte
   premium. Daarom de §5.4-ETF-proxyroute: ETC-NAVs zijn roll-inclusief.
2. **EIA NYMEX Contract 1-4 is een BEVROREN ARCHIEF.** De enige gratis, decennia-
   diepe echte termijnstructuur (WTI 1983->, Henry Hub 1993/94->, HO, RBOB) stopt
   op **2024-04-05** ("futures prices after April 5, 2024, are not available").
   Bruikbaar voor een carry-backtest 1980-2024, niet voor een unit die door het
   recente venster moet leven. Register bijgewerkt.

**Implementatie.** `data/xasset_proxy.py` (26 US-genoteerde ETF/ETC TR-series,
6 sectoren, 2004-01-02 -> 2026-08-07, PIT-gestempeld, 135.183 rijen);
`alpha/cm_tsmom.py` (unit, mét borrow op de shortkant — anders dan FX, waar
shorts swap-impliciet zijn); `apps/ingest_xasset.py` (R-6, met truncatie- en
staleness-weigering); evals in `scripts/w27_xasset_tsmom_eval.py` + `w27_g4.py`.
Kosten: 1bp commissie + 3bp half-spread per zijde, 0,50%/jr borrow. De TER van
de ETFs zit al ín de aangepaste NAV en wordt niet dubbel geteld.

**R-8 incident — echte causaliteitsfout gevangen door de eigen test.**
`test_truncation_invariance_no_future_leakage` faalde bij eerste run: de
herbalanceerkalender werd afgeleid uit de DATA (`groupby(period).max()`),
waardoor de laatste bar van een afgekapte reeks altijd "maandeinde" werd — een
fantoom-herbalancering die live niet bestaat. Opgelost door te herbalanceren op
de EERSTE bar van elke maand (alleen vergelijking met de vorige bar -> causaal
en truncatie-invariant). Effect op het resultaat: net Sharpe 0,512 -> **0,487**.
**Zelfde patroon zit in `alpha/xs_unit.py` regel ~202 en `alpha/fx_tsmom.py`** —
daar raakt het alleen de turnover van de allerlaatste bar (verwaarloosbaar maar
reëel); apart te scopen, niet stilzwijgend meegewijzigd omdat het de vastgelegde
cijfers van gearchiveerde units zou verschuiven.

**Resultaat (net, na alle kosten, 2004-2026, n=5685 dagen).**

| metriek | waarde |
|---|---|
| net Sharpe | **0,487** (gross 0,543) |
| net CAGR / ann vol | +2,0% / 4,2% |
| MaxDD / Calmar / DD-over-vol | -11,4% / 0,173 / 2,72 |
| jaren positief | **17/23 (73,9%)** |
| IS (->2019) / OOS (2019->) | 0,395 / **0,675** — decay -71% (OOS béter) |
| bootstrap P(Sharpe>0), block=21d, B=2000 | **0,993** (5e pct +0,18) |
| rho vs crypto-boek (`combined_book`, n=1376) | **+0,006** |
| N / N_eff | 26 / **5,19** |

Sectorbijdrage (bruto, geannualiseerd): rates +0,78% (Sharpe 0,62), equity
+0,75% (0,36), commodity +0,44% (0,29), credit +0,23% (0,34), real estate
+0,09% (0,22), FX -0,01% (-0,01). Breed gedragen, niet één sector.

**G4 — drie geneste specificaties (self-factor-regel: nooit alleen de gunstigste).**

| Spec | alpha/jr | t | p | Verdict |
|---|---|---|---|---|
| S1 Ken French 6 (nominale mandaat-set) | +1,8% | +2,33 | 0,020 | PASS |
| S2 + PASSIVE (equal-weight long-only zelfde panel) | +1,7% | +2,26 | 0,024 | PASS |
| **S3 + sector-passives (equity/rates/commodity)** | **+1,4%** | **+1,75** | **0,079** | **FAIL** |

Loadings S3: MOM **+0,119 (t=14,5)** — de bouw is getrouw, precies het
F13/F17-bewijspatroon; PASV_RATES +0,134 (t=5,0) — er zít rentebèta in.
Zodra je controleert voor simpelweg long zijn in dezelfde instrumenten, is de
residuele alpha niet meer significant. S3 is voor een cross-asset-boek de
juiste G4: de "markt" van dit boek ís passieve exposure naar dit panel.

**Kill-gates (vooraf geregistreerd in `EXPANSION_RESEARCH_2026-08-10.md` §3B).**

| Gate | Criterium | Stand | |
|---|---|---|---|
| KG-B1 | net Sharpe >= 0,40 | 0,487 | PASS |
| KG-B1 | jaren positief >= 0,60 | 0,739 | PASS |
| KG-B1 | dd/vol <= 2,5 | 2,72 | **FAIL** |
| KG-B1 | Calmar >= 0,25 | 0,173 | **FAIL** |
| KG-B2 | G4 p < 0,05 (strict) | 0,079 | **FAIL** |
| KG-B3 | WF Sharpe >= 0,30 | 0,675 | PASS |
| KG-B3 | decay <= 40% | -71% | PASS |
| KG-B3 | bootstrap P(S>0) >= 0,75 | 0,993 | PASS |
| KG-B3 | abs(rho) vs boek < 0,30 | 0,006 | PASS |
| KG-B4/B5 | tradeability / paper | niet gedraaid | open |

Geautomatiseerd: `pytest tests/killgates -m killgate` leest
`artefacts/killgates/cm_tsmom.json` en faalt op KG-B1/KG-B2. Dat rood ís het
oordeel, geen defect.

**Beslissing: ARCHIVED — lat niet gehaald.** Géén falsificatie, en dus géén
nieuw F-nummer: het premium is aantoonbaar aanwezig (getrouwe MOM-loading,
17/23 jaren positief, OOS béter dan IS, rho≈0 tegen het crypto-boek). Het haalt
de vooraf vastgelegde lat niet — het eq_strev_1m-patroon (+0,39 < 0,40),
niet het F13-patroon (netto negatief).

*Eerlijkheid over de gates zelf:* de twee KG-B1-misses zijn drawdown-VORM-
criteria die ik zelf uit een Brownse benadering heb gekalibreerd; over 22 jaar
is dd/vol 2,7 voor een trendboek niet uitzonderlijk. Ze zijn vooraf
geregistreerd, dus het oordeel staat — ze achteraf verruimen omdat het
resultaat tegenvalt is precies §1.3. De heropeningsconditie hieronder legt de
juiste volgorde vast: eerst de drempel opnieuw afleiden (horizon-bewust,
vooraf), dán pas opnieuw meten. **De G4-strict-miss is de inhoudelijke reden.**

**Heropening alleen als:** (a) uitvoering op echte futures met eigen
termijnstructuur — dat sluit tegelijk de research/executie-mismatch (het boek
is onderzocht op ETF-proxies maar zou op futures gehandeld worden, en 4,2% vol
vraagt ~9x hefboom die EU-retail op ETFs niet krijgt); óf (b) een vooraf
her-afgeleide, horizon-bewuste drawdown-drempel **plus** een G4-strict-pass;
óf (c) een instrumentenset met hogere N_eff (nu 5,19 op 26 namen: zes
equity-ETFs ≈ 1,20 bets, vier rates ≈ 1,26 — F10 in het klein).

**Ledger:** +4 trials (1 unit-config + 3 geneste G4-specs) -> **2702 -> 2706**.
Staging: `artefacts/governance/hypothesis_ledger_staging_w27.json`.
LET OP: **de canonieke `hypothesis_ledger.json` ontbreekt in beide werkkopieën**
(hoofdrepo én Desktop/Setup A) — de CLI weigert terecht hem aan te maken.
Dit is een openstaand G8-governance-punt: de eerlijke DSR-teller leeft nu
alleen nog in dit wave-log.

**Beste-tot-nu-toe OOS-stand.** Onveranderd: crypto-MN-boek Sharpe 1,15 /
+46,7% netto CAGR @40% vol. Geaccepteerde units: 4.

**Gate-tabel (volledig, stand na W27).**

| # | Gate | Drempel | Stand | Verdict |
|---|------|---------|-------|---------|
| G1 | Netto CAGR OOS | >=100% | +46,7% (crypto-boek); W27 voegt 0 units toe | **ROOD** |
| G2 | DSR @ n=2706 | >0,95 | <=0,54 | **ROOD** |
| G3 | PBO (CSCV, S>=50) | <0,20 | 0,51-0,55 indicatief | **ROOD** |
| G4 | residual alpha (HAC) | pass, boek + elke unit | crypto-boek PASS (t=2,65); cm_tsmom S3 FAIL (t=1,75) | **DEELS** |
| G5 | Sharpe per regime x4 | >0 | crypto-boek alle 4 positief | **GROEN** |
| G6 | Lookahead-suite | 100% | groen incl. 6 nieuwe W27-causaliteitstests; **suite ving een echte fantoom-herbalancering** | **GROEN** |
| G7 | Determinisme | pass | cm_tsmom determinisme-test groen | **GROEN** |
| G8 | Data-integriteit | pass | 2 nieuwe bevindingen geregistreerd (roll-contaminatie, EIA bevroren); **ledger-JSON ontbreekt** | **DEELS/ROOD** |
| G9 | WF + >=14d shadow | in tol | niet gestart | **ROOD** |
| G10 | Executie-realisme | pass | research-instrument (ETF) != executie-instrument (futures); KG-B4 niet gedraaid | **DEELS** |

-> Eén rode gate = door (§2). **Volgende aanvalslinie:** `cm_carry` op het
EIA-energie-archief 1994-2024 (echte 4-tenor termijnstructuur, vier producten)
— gescheiden premium (Gorton-Rouwenhorst / Koijen et al. 2018), ander
mechanisme dan trend, en de datalaag ligt er. Daarna FX-carry heropstarten
(W25 is nooit afgerond) en het small/mid-cap-equity-spoor zodra een
delisting-inclusieve PIT-bron bestaat.

---

## Wave 28 — Setup B Fase 0 + Fase 1 (2026-08-10) — GESLOTEN: cm_carry → F20

**Mandaat.** `SETUP_B_ML_PROMPT.md` (futures-boek met ML-overlay) bovenop
`/goal` v3. Fase 0 = governance (blokkeert alles), Fase 1 = `cm_carry` zonder ML.

### Fase 0 — governance

**0.1 Ledger hersteld.** De canonieke `hypothesis_ledger.json` ontbrak in beide
werkkopieën. `scripts/w28_seed_ledger.py` bouwt hem terug uit de geïtemiseerde
wave-log-keten: seed 2363 + 6 gereconstrueerde wave-rijen = 2702, daarna W27
via de gesanctioneerde CLI-merge → **2706**, exact de twee onafhankelijk
vermelde totalen in dit log. Config-hashes van W20–W26 zijn niet te herstellen;
die rijen staan als `RECONSTRUCTED` op wave-granulariteit. **Trial-aantallen
zijn exact.** Eén inconsistentie opgelost en genoteerd: de W22-afsluiting noemt
2377 waar de lopende keten 2381 geeft; 2381 is de reeks die aansluit op W26's
onafhankelijke +321 → 2702, dus 2377 is een verouderde herformulering.

**0.2 Fantoom-herbalancering — GEREPAREERD, niet gescoped.** Reden 1: het is
een truncatie-variantie-(causaliteits)defect, geen kostenbenadering — onder
`AdaptiveWalkForward` vuurt het bij **elke** fold-grens, en Fase 2/3 draaien
die code. Reden 2: het defect voegt alleen kosten toe, dus gearchiveerde
cijfers waren pessimistisch, en `eq_strev_1m` staat op +0,39 tegen een lat van
0,40. Gemeten (`scripts/w28_phantom_rebalance_impact.py`): de laatste
gewichtsrij raakt de netto P&L uitsluitend via `turnover[-1]`, dus de delta is
exact één getal op één bar — **+0,000226** empirisch op het echte 26-instrument-
paneel, **+0,000543** analytisch worst-case. Tegen een gat van 0,0100: **geen
enkel gearchiveerd oordeel kantelt**; F13–F18 blijven staan.
*Testschuld ingelost:* `test_eq_units_causality` sloot de laatste partiële maand
uit met de opmerking "by construction". Dat was geen constructie maar dit
defect; de uitzondering is weg. `fx_tsmom` had géén lookahead-test — nu wel.

**0.3 Labels.** Setup B = **commodities/cross-asset futures met ML-sizing-
overlay**. Setup C = **VACANT, niet herbestemd**. Gearchiveerd: *beide* oude
claimanten — `MASTER_PLAN_SETUP_B.md` ("Positioning & Events", NULL na zeven
ontwerpen) en Setup C (FX-carry; `Desktop/Trading Setup C/` is leeg, 0 items).
**Bevinding die niet meereist:** `EXPANSION_RESEARCH` §0 waarschuwt dat Setup
C's FX-carry "already accepted at Sharpe 0.40, rho~0.03" zou zijn. Dat is in deze
repo **nergens onderbouwd** — de herstelde ledger bevat geen enkele FX-entry,
dit log zegt tweemaal het tegendeel ("unit-code klaar", "W25 is nooit
afgerond"), en er is geen killgate-artefact. Het getal komt uit het inmiddels
verwijderde plandocument. Er is **geen geaccepteerde FX-carry-unit.**

**0.3b KG-B1 dd-vorm-drempel her-afgeleid — vóór enige meting.** De constanten
`dd_over_vol <= 2,5` en `calmar >= 0,25` steunden op `E[MaxDD] = sigma^2/(2mu)`.
Die identiteit is echt maar beschrijft de drawdown **op een willekeurig moment**
(stationaire wet van een gereflecteerde Brownse beweging), niet het **maximum**
— en het supremum daarvan is onbegrensd, logaritmisch groeiend in de horizon.
Gemeten bij Sharpe 0,40 over 22,6 jaar: formule voorspelt 1,25 vol-eenheden;
drawdown-op-willekeurig-moment **1,04** (dus de formule klopt voor díe grootheid);
**maximum 3,25**; bij 200 jaar 5,36. Gevolg, gemeten: de oude poort verwierp
**99,7%** van de strategieën die écht Sharpe 0,40 hebben. Dat is geen poort maar
een muur — en de consistentie-checker die juist onhaalbare specs moest vangen
liet hem door, omdat hij dezelfde verkeerde constante droeg.
Vervangen door één criterium met een expliciet budget: `dd_over_vol` <= het
**q0,95-kwantiel** van de maximale drawdown die een strategie met de **eigen
gerealiseerde Sharpe over de eigen steekproeflengte** produceert (referentie =
deterministische simulatie, `src/tradebot/backtest/dd_shape.py`, seed 42 — geen
gefitte formule, want een gefitte drempel kan stil van betekenis veranderen).
Calmar-vloer **geschrapt, niet versoepeld**: Calmar is een deterministische
functie van de andere twee, drievoudig tellen maakte de set onhaalbaar.
**Disciplinecheck:** onder het gecorrigeerde criterium haalt `cm_tsmom` KG-B1
wél. Het oordeel verandert **niet** — zijn geregistreerde heropeningsconditie
eist óók een G4-strict-pass, en die staat onveranderd op t=1,75 < 2,0. Blijft
ARCHIVED op KG-B2. Vastgelegd in `test_cm_tsmom_stays_archived_on_g4_strict`.

**0.4 Pre-registratie bevroren.** `docs/PREREGISTRATION_SETUP_B_W28.md`,
sha256/32 `2cc042fd6acc82ad55b517d42c5b79bd`, ledger 2706 bij bevriezing.

### Fase 1 — `cm_carry_energy` (geen ML)

**Hypothese.** Termijnstructuur-carry (Gorton-Rouwenhorst 2006; Koijen,
Moskowitz, Pedersen & Vrugt 2018): een curve in backwardation verdient een
positieve rolopbrengst. Signaal `(F1-F2)/F1`, tijdreeks per product,
inverse-vol, maandelijks — vooraf vastgelegd, geen tuning-laag.

**Datalaag nieuw.** `src/tradebot/data/sources/eia.py` + `apps/ingest_eia.py`:
NYMEX Contract 1-4 voor WTI/NG/HO/RBOB, 126 614 rijen, WTI met alle vier tenors
terug tot **1985**. Seriecodes in het register waren fout (`_RGC_` bestaat niet;
correct is `EER_EPD2F_PE{1-4}_Y35NY_DPG` resp. `EER_EPMRR_...`) — gecorrigeerd.
Archiefgrens 2024-04-05 hard afgedwongen.

**Het echte werk zat in de rol.** EIA publiceert per **slot**, niet per
contract; op een roldag verschuift de hele curve één slot. `F2(t)/F2(t-1)`
vergelijkt dan twee verschillende contracten en verzint rendement — precies de
klasse fout die hier al gemeten is op gratis continue futures (+7,0%/jr WTI,
+25,1%/jr NG). Drie bevindingen:

1. **Prijs-gebaseerde roldetectie werkt niet.** Naburige slot-spreads zijn klein
   naast dagruis: de eerste detector vuurde **51-78x/jr** tegen een waarheid van
   ~12, met een diffuse dag-van-de-maand-verdeling.
2. **De kalender wel.** Roldata komen nu uit de CME-contractspecificatie op de
   US-federale handelsdagkalender (níet uit de data — een uit data afgeleide
   kalender maakt een afgekapt paneel blind voor een expiry die live gewoon
   bekend is). Gevalideerd: **12,00-12,02 rolls/jr**, WTI mediaan dag 22 en 100%
   in [15,25], HO/RBOB mediaan dag 1, en shift-evidence **AUC 0,84-0,96** op het
   steilste deciel (het enige regime met onderscheidend vermogen).
3. **De lookahead-suite ving twee echte fouten** die de eerste meting
   besmetten: (a) het gehouden contract schuift **omláág** naar slot k-1, dus de
   correcte roldag-return is `F1(t)/F2(t-1)`, niet `F2(t)/F3(t-1)` — die tweede
   prijst een contract dat nooit gehouden werd; (b) de rol is een **trade**: de
   positie moet terug van slot 1 naar slot 2, ~12x/jr een volledige round trip.
   Beide gecorrigeerd. Effect: turnover 5,2 → **30,5** turns/jr, net Sharpe
   0,124 → **0,100**.

**Kruisvalidatie tegen roll-inclusieve ETF-NAVs:** WTI/USO corr 0,93, afwijking
**+1,73%/jr**; NG/UNG corr 0,95, **-0,63%/jr**. Tegen de +7,0% / +25,1% van
gratis continue futures — de constructie is betrouwbaar.

**Resultaat (net, 1994-2024, n=7585, 30,1 jaar).**

| metriek | waarde |
|---|---|
| net Sharpe | **+0,100** (gross 0,148 — het verschil is de rol) |
| net CAGR / ann vol | -0,65% / 25,1% |
| MaxDD / dd-over-vol | -80,7% / 3,21 (binnen de q0,95-vormcap 3,82) |
| jaren positief | **14/31 (45,2%)** |
| IS / OOS (helft-splitsing) | +0,454 / **-0,278** — decay **+161%** |
| bootstrap P(S>0), block 21d, B=2000 | 0,723 (< 0,75) |
| N / N_eff | 4 / **1,54** |
| turnover | 30,5 turns/jr |

**G4 — drie geneste specificaties.** S1 Ken French 6: alpha +3,89%/jr t=+0,87 · S2
+passive energy: alpha +4,51%/jr t=+1,03 · **S3 +per-product passives: alpha
-4,56%/jr t=-0,91**. Zodra je controleert voor simpelweg long zijn in dezelfde
vier producten, is er geen residuele alpha — hij is negatief.

**Poorten: 8 van 9 rood.** Alleen de vormcap en de +50%-kostenstress halen het.

**Beslissing: ARCHIVED → F20** (falsificatie, niet "lat net niet"): OOS
negatief én G4-strict negatief. Twee lessen apart geregistreerd: (a) dit is de
**zesde** onafhankelijke IS-positief/OOS-negatief-meting in dit programma;
(b) **breedte was de bindende beperking** — N_eff 1,54, want WTI/HO/RBOB zijn
één raffinagerisico. Grinold met breadth 1,5 geeft geen boek, hoe goed de
carry-schatter ook is. F10, opnieuw, nu in energie.

**Ledger:** +1 (gate-herafleiding, conservatief geteld) +4 (1 unit-config + 3
geneste G4-specs) → 2706 → **2711**.

**Beste-tot-nu-toe OOS-stand.** Onveranderd: crypto-MN-boek Sharpe 1,15 /
+46,7% netto CAGR @40% vol. Geaccepteerde units: 4.

**Gate-tabel (volledig, stand na W28).**

| # | Gate | Drempel | Stand | Verdict |
|---|------|---------|-------|---------|
| G1 | Netto CAGR OOS | >=100% | +46,7% (crypto-boek); W28 voegt 0 units toe | **ROOD** |
| G2 | DSR @ n=2711 | >0,95 | <=0,54 | **ROOD** |
| G3 | PBO (CSCV, S>=50) | <0,20 | 0,51-0,55 indicatief | **ROOD** |
| G4 | residual alpha (HAC) | pass, boek + elke unit | crypto-boek PASS (t=2,65); cm_tsmom S3 t=1,75; cm_carry S3 t=-0,91 | **DEELS** |
| G5 | Sharpe per regime x4 | >0 | crypto-boek alle vier positief | **GROEN** |
| G6 | Lookahead-suite | 100% | groen; **suite ving 2 echte fouten in cm_carry** (rolrichting + ontbrekende rolkosten) vóór acceptatie | **GROEN** |
| G7 | Determinisme | pass | dd_shape-referentie seed-vast; suite groen | **GROEN** |
| G8 | Data-integriteit | pass | **ledger hersteld (was ROOD)**; EIA-module live + seriecodes gecorrigeerd; rolkalender gevalideerd; constructie gekruisvalideerd tegen ETF-NAVs | **GROEN** |
| G9 | WF + >=14d shadow | in tol | niet gestart | **ROOD** |
| G10 | Executie-realisme | pass | research-instrument != executie-instrument; KG-B4 niet gedraaid — **wacht op werkelijk accountkapitaal** | **DEELS** |

→ Eén rode gate = door (§2). **Volgende aanvalslinie:** Fase 2 — `cm_judge`,
meta-label sizer bovenop `cm_tsmom` (primair net **+0,487**, dus sizing van een
levend signaal, geen redding van een dood signaal — §6.2). Daarna pas Fase 3.
