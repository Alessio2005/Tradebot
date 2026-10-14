# CHANGELOG

> Gereconstrueerd 2026-06-10 (Wave 20): het originele bestand bleek leeg.
> Bronnen: docs/TRUE_ALPHA_AUDIT_2026-06-08.md, docs/data_dictionary.md,
> docs/WAVE_LOG.md, code-annotaties (AUDIT-tags in src/).

## DI-31 herzien (2026-10-14) — numpy's kernelkeuze, niet OpenBLAS
- **Symptoom:** `test_the_multiplier_is_causal[hmm3-diag-student_t]` faalde af
  en toe met `atol=0, rtol=0`. De volledige en de afgekapte run fitten
  bitidentieke trainvensters, maar de Student-t EM gaf soms andere bits terug.
- **Oorzaak:** `np.log(u[:, i])` op een kolom-VIEW in de M-stap, opgelost in
  `ad44372`. numpy 1.26 kiest voor float64-`log` de AVX-512-kernel alleen als
  `nomemoverlap` invoer en uitvoer disjunct ziet. Voor kolom `i` neemt die
  controle `stride * n` bytes, 8·i bytes voorbij het einde van `u`, en telt
  hij aansluitende bereiken als overlap. Landt de verse uitvoer in het
  heapblok direct na `u`, dan rekent numpy met scalaire libm: 1 ULP verschil
  op ~25 % van de elementen. 200 EM-iteraties maken daar 1e-12 van in de
  parameters en 1e-14 in de factor. Bij k = 2 haalt de overschrijding het
  volgende blok nooit.
- **Weerlegd: multithreaded OpenBLAS.** De oude regel drijft ook met OpenBLAS
  op één thread (8 van 20 herfits). In een verse Python-3.13-omgeving uit de
  lockfiles bouwt numpy 1.26.4 uit de sdist zonder BLAS, dus `@` raakt OpenBLAS
  daar niet. De OpenBLAS 0.3.30 die DI-31 noemde is die van scipy, en de EM
  roept die niet aan. `threadpool_limits` om de fit veranderde de
  heapgeschiedenis en daarmee waar de uitvoer landde. Dat verklaart waarom
  het de ene run repareerde en de pytest-run juist liet falen.
- **Gemeten (2026-10-14).** Herfits van de acht opgenomen trainvensters na
  een willekeurige heapverstoring:

  | omgeving | oude regel | HEAD |
  |---|---:|---:|
  | Py 3.11, numpy-wheel (OpenBLAS 0.3.23) | 22/30 | 0/120 |
  | Py 3.13, numpy zonder BLAS | 15/40 | 0/360 |

  De vergelijking van de causaliteitstest zelf, onder dezelfde verstoring,
  faalde met de oude regel 12/12 en 5/6 en op HEAD 0/48 en 0/12. Een spion
  over een volledige `build_conditioner` vond geen enkele transcendente
  ufunc-aanroep (numpy, scipy, hmmlearn) met een niet-contigue invoer.
- **Geen numerieke wijziging en geen besluit nodig.** Tolerantie, markers en
  de bevroren H2-getallen blijven ongemoeid. De regel wordt bewaakt door
  `test_transcendentals_never_see_a_strided_view`. Het register
  `docs/DEFERRED_ISSUES.md` staat sinds `57d3fcf` niet meer in de repository;
  deze notitie en de toelichting bij `log_u` in `regime/student_t.py` zijn
  de bijgewerkte beschrijving.

## Robuust boek v6–v8 (2026-10-09) — hefboom, uitvoering, sprongrisico
- **Eigenaarsbesluit:** hoog risico toegestaan; hefboom niet langer uitgesloten.
  Mandaat ongewijzigd: bruto ≤ 4,0, ADV-participatie ≤ 1 %, max drawdown 25 %.
  Rapport: `docs/research/2026-10-09-robust-book-v6-v8-report.md`.
- **Nieuw boek `systematic/leverage.py`:** de basiscarry van v5 in een
  unified-margin-account.
  - Spot als onderpand met haircut; USDT-lening tegen `max(8 %, BTC-carry)`.
  - Liquidatietoets op de intraday **mark price**: nieuwe data, 895/900 perps,
    MD5-geverifieerd.
  - Governor op uniMMR ≥ 3 en bruto ≤ 4; ADV-cap tegen de lopende equity.
  - Gespreide uitvoering (v7), een gefinancierde tranche (v7) en een
    sprongrisicogrens (v8).
- **v6 (M = 24):** 1×/1,5×/2× per been op W_DEV gaf Sharpe 5,4 / 4,3 / 3,7 en CAGR
  12,1 / 13,6 / 14,0 %. Alle drie binden op G2 en G5. Hefboom is bijna NPV-nul, omdat
  de leenrente met de carry meebeweegt.
- **v7 (M = 26):** spreiding over 5 dagen plus een tranche alleen waar carry ≥
  rente + 10 pp. **S2 haalt op W_DEV alle poorten:** Sharpe 5,35, CAGR 16,7 %,
  MDD 1,1 %.
- **De stille killer:** in de holdout 2025–26 liet de carry-screen alleen
  pump-and-dump-microcaps door (dag-σ 0,24–1,05). Dat is adverse selectie, geen
  carry.
  - v6-K1: −8,6 % (z = −14,9), gefalsificeerd.
  - v7-S2: −5,7 % (z = −16,7), gefalsificeerd.
- **v8 (M = 27):** een sprongrisicogrens (dag-σ ≤ 0,20), afgeleid uit de bevroren
  kosten- en instapparameters.
  - T2: W_DEV Sharpe 5,33, CAGR 15,9 %, alle poorten vrij.
  - Backcast 2020: Sharpe 9,0, CAGR 24,7 %.
  - Holdout −1,1 % (DD 1,4 %), formeel gefalsificeerd door de bevroren z-regel.
  - Hele cyclus 2020-04 .. 2026-09: **Sharpe 5,05, CAGR 13,4 %, MDD 2,0 %**.
- **Vooruit-slot** (vanaf 2026-10-01), bevroren op 2026-10-09 vóór alle lezingen: de
  enige schone toets. Lezing vanaf april 2027 via `binance_vision forward` en
  `programme_v8 forward`.
- **Scenario:** met één beurs en ~3 %/jaar kans op een faillissement zakt de echte
  Sharpe naar ~0,8. Meerdere beurzen zijn een voorwaarde voor de doelstelling.
- **Reproduceerbaarheid:** v5 reproduceert bit-identiek op opnieuw opgehaalde data;
  v6 en v7 na elke refactor.

## Mandaatwijziging (2026-09-12) — eigen kapitaal i.p.v. propfirm
- **Risicobudget herijkt.** Er wordt niet meer met propfirms gewerkt; de
  drempels die contractnaleving waren zijn vervangen door een
  eigen-kapitaalbudget. `max_drawdown_pct` 0,08 → 0,25 · `daily_loss_limit`
  0,03 → 0,10 · de-grossing-trappen 0,04/0,06 → 0,12/0,18 ·
  `daily_var_limit_pct` 0,02 → 0,05 · `max_position_age_h` 48 → 720 ·
  `sigma_target` 0,08 → 0,20 · `max_leverage` en `gross_cap` 1,5 → 4,0 ·
  `net_cap` 0,60 → 2,0 · `max_position_pct` 0,25 → 0,80.
- **`adv_participation_cap` (0,01) en `max_concentration` (0,40) ONGEWIJZIGD.**
  Marktfeit resp. de enige werkzame spreidingsbescherming (AD-4); geen
  mandaatgrond om ze te verruimen.
- **`docs/RISK_MANDATE.md`** (nieuw): verantwoording per limiet, met een label
  A (propfirm-afgeleid) / B (interne "strengste wint") / C (marktfeit), zodat
  "de propfirm is weg" geen blanco cheque voor het hele bestand is. AD-26 in
  `docs/ARCHITECTURAL_DECISIONS.md`.
- **Twee limieten maakten een levende kandidaat onmeetbaar**, en dat is de enige
  meetbare winst van deze wijziging: `max_position_age_h: 48` sloot elke positie
  na twee dagen terwijl funding carry (~1,95 bps/dag) ruim zeven dagen nodig
  heeft om alleen de 13,0 bps vaste kosten terug te verdienen; en
  `max_drawdown_pct: 0,08` haltte de enige forward-meting in de repo
  (`max_drawdown −0,3817`) vóór het venster uit was.
- **Geen edge-claim geraakt.** Geen backtest opnieuw gedraaid, geen metriek
  herschreven, geen ledger-entry gewijzigd; de vier bekende poort-failures
  (KG-B1/B2/B3 op `cm_carry`/`cm_tsmom`) staan onveranderd rood.
- **Achterhaald verklaard:** `artefacts/PROPFIRM_PARALLEL_AUDIT.md`,
  `docs/STRATEGY_AUDIT_CRYPTO_ACCOUNTS_2026-06-14.md` en §4.5 van
  `docs/EXPANSION_RESEARCH_2026-08-10.md` ("keep the propfirm numbers — they are
  contractual"). `risk/daily_loss_governor.py` gemarkeerd DORMANT met een
  meetbare heropeningsvoorwaarde.
- **Tests mandaat-onafhankelijk gemaakt** i.p.v. herijkt op nieuwe literals: de
  kill-switch-scenario's leiden hun equity nu af uit `conf/risk/`, en de
  engine-pariteitstolerantie is genormaliseerd per eenheid bruto-exposure
  (gemeten stabiel op 11,7-15,0 bp/gross over een factor 2,5 aan boekgrootte,
  waar de absolute 3,0 bp dat per constructie niet was).
- Nieuw in DEFERRED_ISSUES: DI-27 (de vol-target bindt niet), DI-28 (drie
  tweede bronnen van waarheid voor risicodrempels), DI-29 (dode
  propfirm-machinerie).

## Wave 20 (2026-06-10) — mandaat v3 fundament
- Hypothesis-ledger (append-only, seed 2363) + falsificatieregister F1–F12.
- PIT multi-markt datalaag (`data/sources/`): stooq, kenfrench, edgar, fred,
  cboe, wiki_constituents; `asof_join` als enige gesanctioneerde PIT-merge.
- G4-factorlab: `risk/factor_alpha.py` (HAC/Newey-West residual alpha).
- Crypto-rebaseline bit-identiek gereproduceerd (Sharpe 1.15, CAGR +46.7%,
  G6 α +20.2%/jr t=2.65 — audit §9).
- W21/22-voorbereid: XS-harness + eq_xsmom/eq_strev/eq_lowvol + PIT-universum.

## Waves 14–19 (2026-06-09) — zie TRUE_ALPHA_AUDIT §12–17
- W15: XS-rework — pooled cross-sectioneel recept, OOS IC +0.069, modellen
  gepersisteerd (`artefacts/tracks_breadth/`). W19: AdaptiveWalkForward
  deploy-grade (`alpha/adaptive_wf.py`), PBO S≥50-les.

## Audit-fixes (2026-05-28 → 2026-06-08)
- **AUDIT A-1** — Fractional differentiation featureblok (AFML hf. 5):
  `src/tradebot/features/fracdiff.py`, `ffd_*` features met per-asset d*
  (zie docs/data_dictionary.md).
- **AUDIT D-2** — CPCV purge in bar-space: `CombinatorialPurgedCV` vereist
  expliciete `purge_bars` in intraday-regimes (≥24 bars/dag); kalenderdag-
  purge was te zwak (`src/tradebot/cv/cpcv.py`; afgedwongen met ValueError).
- **AUDIT E-4** — Funding-rate featureblok, causaal (t−1):
  `src/tradebot/features/funding_carry.py`, `funding_*` micro/meso-features.
- DSR-dimensionaliteitsfix (`metrics.deflated_sharpe`): expected-max
  geschaald door SE van de Sharpe-schatter (Bailey & LdP 2014);
  regressie-locked in `tests/regression/test_dsr_dimensional_fix.py`.

## Waves 5–13 — infrastructuur
- CPCV + purge/embargo, triple-barrier/TrendScan-labeling, HRP/RP-portfolio,
  TCA-laag, OMS + audit-log (R-8), circuit-breakers (R-7), paper-engine,
  MRM-rapportage, lookahead-testsuite, determinisme-suite (R-5).
