# CHANGELOG

> Gereconstrueerd 2026-06-10 (Wave 20): het originele bestand bleek leeg.
> Bronnen: docs/TRUE_ALPHA_AUDIT_2026-06-08.md, docs/data_dictionary.md,
> docs/WAVE_LOG.md, code-annotaties (AUDIT-tags in src/).

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
  "de propfirm is weg" geen blanco cheque voor het hele bestand is. AD-13 in
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
- Nieuw in DEFERRED_ISSUES: DI-18 (de vol-target bindt niet), DI-19 (drie
  tweede bronnen van waarheid voor risicodrempels), DI-20 (dode
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
