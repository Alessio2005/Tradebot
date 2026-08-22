# CHANGELOG

> Gereconstrueerd 2026-06-10 (Wave 20): het originele bestand bleek leeg.
> Bronnen: docs/TRUE_ALPHA_AUDIT_2026-06-08.md, docs/data_dictionary.md,
> docs/WAVE_LOG.md, code-annotaties (AUDIT-tags in src/).

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
