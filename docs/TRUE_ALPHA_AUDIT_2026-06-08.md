# TRUE-ALPHA AUDIT — FINAL: STATE (B) (2026-06-08)

**Author:** Chief Quantitative Architect
**Mandate:** ULTIMATE REFACTOR — ≥100% net CAGR via true (market-neutral, regime-
independent, non-trend) alpha, OOS, after all costs, within Bybit-EU 10× spot cap.
**Status:** **FINAL → STATE (B)** (see §9). All levers pulled (cost-realism,
combination, microstructure, breadth, survivorship, ML, vol-target/leverage).
G1 ≥100% CAGR is **not achievable**; **highest verified level ≈47% net CAGR** with
genuine market-neutral non-directional alpha (G5✓ G6✓, p=0.008). Sections 0–7 below
record the wave-by-wave path; §8–9 are the final verdict.

---

## 0. Executive summary

| Gate | Threshold | Current book | Verdict |
|------|-----------|--------------|---------|
| G1 CAGR (OOS, net) | ≥100% | ~33% @ 30% vol (4.1× lev) | **FAIL (far)** |
| G2 DSR (honest n_trials) | >0.95 | 0.87 @ N=12 · 0.70 @ N=50 · 0.51 @ N=200 | **FAIL** |
| G3 PBO | <0.20 | not yet computed | pending |
| G4 rolling \|β\| 90d | <0.10 | max 0.155 (BTC) / 0.187 (basket); 92–97% windows <0.10 | **MARGINAL** |
| G5 Sharpe every regime bucket | >0 | all 4 positive (worst bear/lovol +0.25) | **PASS** |
| G6 α after factor regression | >0, p<0.05 | **+23.1%/yr, t=3.17, p=0.0015** (MKT+TSMOM) | **PASS** |
| G7 lookahead suite | 100% | inherited (causal sleeves) | pending re-run |
| G8 walk-forward/shadow | in tol | not yet | pending |
| G9 determinism | bit-identical | inherited | pending |

**One-line read:** the 3-sleeve causal-RP neutral book is genuine, statistically
significant *market-neutral, non-directional* alpha (G6 PASS), but its **Sharpe
(1.26) is ~2.5× too low** for the 100% CAGR target under a sane (≤5×) leverage
cap, and its **DSR fails once an honest hypothesis count is used**. The gap is a
**Sharpe-magnitude problem**, not a directionality problem.

---

## 1. Silent killer fixed — the DSR gate was dead

`metrics.deflated_sharpe` subtracted the expected-max of N standard normals
(≈1.67 for N=12, a *z-unit* quantity) directly from a per-day Sharpe (~0.07),
giving z ≈ −68 and **DSR ≡ 0 for every strategy ever passed through it**. This is
the "DSR mis-param" flagged in the 2026-05-28 audit.

**Fix** (`metrics.py`, regression-locked in `tests/regression/test_dsr_dimensional_fix.py`):
scale the expected-max by the Sharpe estimator SE → `z = SR̂/σ_SR − e_max_z =
t_stat − e_max_z` (Bailey & López de Prado 2014). The live tuner
(`evaluation.deflated_sharpe_penalty`) and portfolio audit
(`portfolio._deflated_sharpe`) used *different, correct* implementations, so
production gating was unaffected — but every research readout was reading 0.

Corrected book DSR: **0.87 (N=12)**, decaying to 0.51 at N=200 honest trials.

---

## 2. The feasibility frontier — why 100% needs Sharpe ≈ 3

Geometric CAGR ≈ Sharpe·σ − ½σ². At the book's net Sharpe 1.26:

| Target vol | ≈ Leverage (on 7.4% base) | Geometric CAGR |
|---|---|---|
| 30% | 4.1× | **33%** |
| 40% | 5.4× | 42% |
| 50% | 6.8× | 50% |
| 80% | 10.8× (>cap) | 69% (drag eats the rest) |

Sharpe required for 100% geometric CAGR: **3.20 @ 33% vol · 2.70 @ 40% vol.**
The book is at **1.26**. Reaching 100% at today's Sharpe needs ~11× leverage —
over the 10× cap and a liquidation magnet for a statarb book whose loss regime is
dispersion-collapse. **The mandate therefore reduces to: raise net Sharpe 1.26 → ~3.**

---

## 3. G6 interpretation (resolved by mandate author)

§4.2 mandates building carry- and cross-sectional-reversal sleeves; §1.2.3 (G6)
lists carry and XSMOM among factors to regress *out* — mutually exclusive for those
sleeves. **Ruling: G6 = the not-trend-following test** (regress out MKT + TSMOM
only). Harvesting carry and cross-sectional reversal premia *is* legitimate
market-neutral alpha. Under this ruling the book's residual alpha is **+23.1%/yr,
t=3.17, p=0.0015 — PASS**. (Loadings confirm it: MKT β=−0.04, TSMOM β≈+0.02 small;
the engine is −XSMOM/reversal and carry, both market-neutral.)

---

## 4. What has NOT yet been done (the open levers toward Sharpe ~3)

1. **Universe breadth.** Carry sleeve uses **only 5 assets**; statarb/lowvol use 75
   *survivorship-biased* Binance perps. Cross-sectional Sharpe scales with breadth
   → expand the universe (and correct survivorship) is the largest single lever.
2. **No new features.** OFI / microstructure / cross-asset spreads / lead-lag /
   basis-z — none added. §4.4 untouched.
3. **No model work.** Meta-labeling Judge not fed sleeve signals; no stacker; no
   feature selection (MDA/MDI); no probability calibration on sleeve confidence.
4. **No new orthogonal sleeves.** Kalman-OU pairs cointegration, multi-horizon
   residual reversal, VRP — not yet built/validated.
5. **No purged CPCV / PBO** on the sleeve stack (G3 pending).
6. **No beta-hedge overlay** to convert G4 MARGINAL → PASS.
7. **No vol-targeting / leverage** layer to the 30–40% vol target.

---

## 5. Decision

**End-state not yet determined.** The book clears the *quality/directionality*
gates (G5 PASS, G6 PASS, G4 MARGINAL) but fails the *magnitude* gates (G1, G2) by a
wide margin. Per mandate author: this is an interim checkpoint, not a (B) verdict —
the universe-, feature-, and model-levers are unexhausted. **Program continues**:
raise Sharpe toward ~3 via breadth + orthogonal sleeves + ML sizing, tighten G4 with
a hedge overlay, then vol-target/leverage and run the full G1–G9 gate. Conclude (A)
only on all-green OOS-after-cost, or (B) only after these levers demonstrably top out.

**Reproduce:** `python scripts/validate_neutral_book.py` · `python scripts/true_alpha_gates.py`

---

## 6. Wave 1 addendum — cost realism (2026-06-08)

Per-sleeve net Sharpe decomposition at realistic cost exposes that the 1.26
baseline used optimistic 5 bps:

| Sleeve | Sharpe @5 bps | Sharpe @10 bps | daily turnover | note |
|--------|--------------|----------------|----------------|------|
| LOWVOL (−rank vol_20) | 1.32 | **1.22** | 0.14 | workhorse, cost-robust |
| REVERSAL k=1 (breakthrough used k=3) | 0.34 | **−0.51** | 1.44 | cost-DEAD at k=1 |
| REVERSAL k=10 (fix) | 0.70 | **0.41** | 0.46 | horizon fix resurrects it |
| CARRY (5 assets) | ~0.5 | **0.27** | low | weak; only 5 names |

**Honest 10 bps causal-RP book (carry + rev_k10 + lowvol): Sharpe ≈ 0.94**, all
years positive (2021..2026 +5/+6/+16/+28/+14/+15%). Sleeve corr: carry⊥both
(≈0), rev↔lowvol = 0.47 (overlapping cross-sectional sleeves).

**Implications:** (a) the realistic baseline is **~0.94, not 1.26** — the gap to
Sharpe ~3 is *wider* than §2 stated; (b) the book ≈ the lowvol sleeve alone, so
the "diversification" story is thin at real costs; (c) **lowvol is the most
survivorship-biased sleeve** (delisted high-vol losers are excluded) — correcting
this (needs delisted-coin klines) is the top priority and may lower it further;
(d) the real Sharpe lever is *new genuinely-orthogonal sleeves* + *breadth*, not
re-weighting the existing overlapping ones.

---

## 7. Wave 3 addendum — orthogonal sleeves + combination (2026-06-08)

**Microstructure/OFI sleeve — REJECTED.** `runs_imbalance` on 14-min info-bars has
a real but tiny micro-reversal IC (−0.008…−0.019, sign-consistent across 5 assets),
but gross Sharpe ≤0.38 and net@10 bps = −7…−13 (≈32 bars/day × 0.32 turnover). Not
tradeable. Confirms the 2026-05-31 "tick-OFI ≈ noise" finding. (3 variants → DSR n.)

**Combination tuning — no gain.** Causal Sharpe-proportional and μ/σ² schemes
(0.71 / 0.69 full) are WORSE than naive inverse-vol RP (0.94), because rolling-mean
estimates are noisy and chase recent winners. **RP is already optimal + robust.**

| Combination (10 bps) | full Sharpe | IS 21-23 | OOS 24-26 |
|---|---|---|---|
| **RP (inv-vol)** | **0.94** | 0.60 | 1.39 |
| Sharpe-proportional | 0.71 | 0.24 | 1.37 |
| μ/σ² (max-Sharpe) | 0.69 | 0.17 | 1.41 |

**Established ceiling (cached data):** net Sharpe **0.94 full / 1.39 OOS** (10 bps).
At safe leverage (≤5×, ≤40% vol) ⇒ **~24–30% CAGR (full) / ~48% (OOS-favorable)**.
Gap to the Sharpe ~3 needed for 100% remains ~3.4×. Remaining unpulled levers:
breadth + survivorship correction (needs network), and ML meta-label sizing.

---

## 8. Wave 2/4 addendum — breadth + ML (all levers pulled)

**Breadth (Wave 2).** Expanded universe 75 → 99 perps (onboard<2022-07, the max
with multi-regime history; the other ~430 trading perps are 2023+ small-caps).
Result: unlevered Sharpe 0.94 → **1.06**, vol-targeted CAGR 39% → **47%**, G6 t
2.36 → 2.65. **Real but weak lever** (+0.12 Sharpe per +24 names, steep diminishing
returns; history-availability caps breadth at ~100 names). Cannot approach Sharpe 3.
*Survivorship is NOT fixed* — the live klines API returns nothing for delisted
symbols; correcting it would *lower* the low-vol sleeve, not raise it.

**ML meta-label sizing (Wave 4).** Walk-forward GradientBoosting (expanding window,
6 causal regime features) predicting next-day book sign: **mean test AUC = 0.517**
(no skill, matches the 2026-05-31 AUC 0.51). Gating the book by predicted prob
*destroys* Sharpe (OOS 1.59 → 0.76). **The book's returns are un-timeable** — no ML
lever exists.

---

## 9. FINAL DECISION — STATE (B): 100% net CAGR not achievable; ceiling ≈ 47%

**Verdict.** Under the binding constraints (true market-neutral / non-directional
alpha, OOS, after realistic 10 bps cost, Bybit-EU ≤10× spot-margin, this crypto
universe), the **G1 ≥100% net CAGR gate is not achievable**. The mandate is in
end-state **(B)** of §1.3: a rigorous impossibility result plus the highest
verified level.

**Final gate scorecard (best book: 99 names, carry+rev_k10+lowvol, causal-RP,
vol-targeted 40%, 10 bps):**

| Gate | Threshold | Result | Verdict |
|------|-----------|--------|---------|
| G1 net CAGR (geom, OOS, net) | ≥100% | **+47%** | FAIL |
| G2 DSR (corrected) | >0.95 | 0.76 @N12 · 0.54 @N50 | FAIL |
| G3 PBO (CSCV) | <0.20 | 0.55 | FAIL |
| G4 rolling 90d \|β\| | <0.10 | max 0.14–0.17 (97–98% <0.10) | MARGINAL |
| G5 Sharpe every regime | >0 | all 4 positive | **PASS** |
| G6 α vs MKT+TSMOM (HAC) | >0,p<.05 | **+20%/yr, t=2.65, p=0.008** | **PASS** |

The book *is* genuine, statistically-significant market-neutral, non-directional
alpha (G5✓ G6✓). It fails purely on **magnitude and robustness** (G1/G2/G3).

**The mathematical ceiling (why ~Sharpe 1, not 3).** Geometric CAGR = S·σ − ½σ²;
100% at ≤40% vol (≤5× lev) needs **S ≥ 2.7**. Net true-alpha Sharpe in this
universe is **~0.9–1.15** (full) / ~1.5 (OOS, regime-favourable). Sharpe stacks as
√N over *uncorrelated* sleeves, but this universe contains only **~3 genuinely
orthogonal market-neutral premia** — carry (ρ≈0), and the overlapping
reversal/low-vol pair (ρ=0.47). Microstructure/OFI is empirically untradeable
(net −7…−13 @10 bps); directional/trend loads on TSMOM (barred by G6); ML timing
has no skill (AUC 0.52); funding-as-standalone-alpha and lead-lag were already
falsified (2026-05-30/31). To triple S to ~3 needs ~9× more zero-correlation
sleeves that **do not exist** in this universe. Hence S is capped near 1–1.5 and
**CAGR near 40–55%**, not 100%, at any prudent leverage.

**Highest VERIFIED level (the deliverable number):** **≈47% net CAGR**, net Sharpe
≈1.15 at ~40% realized vol (avg gross 2.4×, peak 10×), market-neutral (G4 marginal),
positive in every regime (G5), with significant non-directional alpha (G6,
p=0.008). This is the defensible product; it is **not** 100%.

**To revisit (A) would require relaxing a constraint:** (i) a much larger
*survivorship-free* universe (300+ names incl. delisted — needs historical data
dumps, would *lower* low-vol), (ii) options/perp VRP instruments (barred for EEA
retail), or (iii) accepting directional/trend exposure (violates the true-alpha
definition). None are available under the current mandate.

**Reproduce final:** `python scripts/wave_final_eval.py artefacts/broad_perp_daily_close_WIDE.parquet 10 0.35`

---

## 10. PIVOT — directional ML book (user authorized Option 1: relax G4/G6)

User relaxed market-neutrality; target raised to 100–150% CAGR; directive: **use the
existing ML pipeline + 126-feature set, no hand-made strategies.** Hand-made trend
heuristics (MA-cross, XSMOM) all failed catastrophically (Sharpe ≤0.3, MaxDD −90%+,
leverage→ruin) — and crypto is cross-sectional *reversal* not momentum (XSMOM loads
negative), so directional edge only exists in *time-series* form.

**The existing ML book (`artefacts/tracks/`, 5 assets, triple-barrier + CatBoost
ensemble + Judge meta-label, CPCV-OOS) leveraged to target vol:**

| Target vol | Leverage | CAGR | Sharpe | MaxDD | per-year |
|---|---|---|---|---|---|
| 30% | 3.8× | +84% | 2.18 | −40% | all + |
| **45%** | **5.6×** | **+141%** | **2.18** | **−55%** | 2021..2025 all +, 2026 −31% |
| 60% | 7.5× | +209% | 2.18 | −66% | all + |

DSR (corrected): 0.99 @N=50, **0.92 @N=2000** (honest trial count). Lookahead suite
(G7) PASSES. CAGR ex-2022 = +89%; 2024–26 = +59% — **not a single-regime artifact.**

**HITS the 100–150% CAGR target on paper, but NOT yet a defensible state (A) —
chief flags:**
1. **Shorts are degenerate (short=0.0% every year)** — the book is long-only/
   long-biased (partly beta). The 2026-05-28 "degenerate long-only calibrator" flag
   is unresolved. It profits in 2022 by being *flat 95%* and catching long bursts,
   not by shorting.
2. **Edge decays in 2026** (−31% levered) — the best live-proxy window is negative.
3. **DSR 0.92 < 0.95** at honest n_trials (~2000) — borderline G2 fail.
4. **Only 5 assets** — user wants 70+; concentration risk; G3 PBO not yet run.
5. **−54% MaxDD at 5.6×** — survivable, not ruin, but a worse crash while levered
   approaches it.

**Path to defensible (A):** expand ML to 70+ Bybit assets (daily-feature pipeline on
the 256-name panel), repair/verify the short side, honest DSR/PBO at true n_trials,
and a leverage policy bounding ruin. Target reached in magnitude; integrity work
remains. (Reproduce: load `artefacts/tracks/*.joblib`, Kelly-size, leverage to vol.)

### 10b. CRITICAL CORRECTION — the directional ML edge is NOT tradeable

Deeper scrutiny (the reason §10's "hits target" is WITHDRAWN):

- **OOS AUC of every model = 0.504–0.523** (LONG and SHORT, all 5 assets, n=35k–76k).
  No out-of-sample discriminative skill. Matches the 2026-05-31 AUC-0.51 finding.
- **Honest book reconstruction from the OOS probs** (trade when prob>thr·judge>0.5,
  realize bar-by-bar returns): Sharpe ≈ **0 at every threshold** (−0.66 @0.60, +0.02
  @0.75, −0.01 @0.85), all years ±5%. **No tradeable edge.**
- **Adding the (healthy-prob) SHORTS makes it −3.26** — shorts are anti-predictive;
  the tracks suppressed them correctly, it was not a calibrator bug.
- The `tracks/*.joblib` **Sharpe 2.18 is a construction artifact** of
  `signed_returns × Kelly`, NOT a tradeable strategy. Leveraging it to "141% CAGR"
  would trade pure noise at 5.6× → **live blow-up.**

**Verdict:** the directional ML route does not contain real OOS edge on this
data/feature set. The only VERIFIED, tradeable edge in the entire study remains the
market-neutral book of §9 (~40–55% CAGR, Sharpe ~1–1.5). 100–150% CAGR of
*live-survivable* edge is **not demonstrable** from the current data and features —
not because of insufficient effort, but because the features carry no OOS
directional information (AUC 0.51) and the market-neutral premia are √N-capped.

---

## 11. Added data (funding) + leverage/execution optimization (2026-06-08)

**Funding rates (free, full history, 99 names) — adds NO edge (tested 3 ways):**
- Directional IC funding-z → fwd ret ≈ 0 at all horizons (+0.002/−0.002/−0.001/
  +0.001, n=176k). Funding does not predict direction.
- Broad cross-sectional carry sleeve: Sharpe −0.14…+0.18, NEG in 3–5 yrs — *worse*
  than the 5-name carry. Funding-as-alpha falsification confirmed at breadth.
- Funding-conditioned reversal: conditioning *hurts* (0.61 → 0.42 → −1.03).

**Leverage/execution optimization — safer verified product (`scripts/mn_optimized_exec.py`):**
position-level sleeve netting + no-trade band + ruin-bounded fractional-Kelly
de-grossing:

| Book | CAGR | Sharpe | MaxDD | Calmar | gross |
|------|------|--------|-------|--------|-------|
| baseline (naive RP, flat target) | +56% | 1.19 | −50% | 1.13 | 2.8× |
| **optimized (netting+fracKelly)** | +51% | 1.27 | **−40%** | 1.28 | 1.8× |
| optimized + 2% band | +45% | 1.17 | **−34%** | **1.33** | 1.6× |

Real improvement in DD-adjusted return + ruin safety; DSR still 0.14 @N=2000 and
returns concentrate in 2023–25 — leverage amplifies but cannot manufacture
robustness. **Verified product stands at ~45–55% CAGR, Calmar ~1.3, MaxDD ~−34%.**

---

## 12. Wave 14 — the 70×2 breadth rework (mandate v2, 2026-06-09)

**Mandate v2.** True-alpha (market-neutral) gate **dropped** — directional/trend/
regime edge now permitted. User directive: *"only 10 models now — go to 70×2,
rework, optimize, read López de Prado / Bailey, don't stop until achieved."* This
re-enters the directional region already mapped in §10 (Option 1), now at full
breadth and with the canonical AFML techniques applied directly.

**Data.** Fetched **full daily OHLCV** for 99 long-history (onboard <2022-07) USDT
perps, Binance public, 2021-06→2026-06 (`scripts/fetch_broad_ohlcv.py` →
`artefacts/broad_perp_ohlcv.parquet`, 176.7k rows). Free, point-in-time-honest
(klines are settled at bar close), survivorship caveat unchanged (delisted names
absent — *upward* bias). Costs 10 bps; vol-target 40%.

**A. Per-asset L/S models (the literal 70×2 = 198 models).**
`scripts/breadth_ml_book.py` — per asset: causal 16-feature set, triple-barrier
L/S labels, 6-fold purged+embargoed CV → OOS probs. **198 models scored: median
AUC 0.551, pooled 0.547, 51% of models >0.55** — *looks* skilled. But directional
rank-IC **−0.007** (wrong sign ≈ 0), and the inverse-vol breadth book **LOSES:
Sharpe −0.59, −27% CAGR, MaxDD −84%, DSR 0.000.** Fundamental-Law line at full
breadth: need IC 0.215, measured **0.007** → **31× gap**.

**B. Pooled cross-sectional model (the AFML-correct rework).**
`scripts/pooled_xs_book.py` — ONE LONG + ONE SHORT CatBoost on **all 169,031 events
pooled**, global time-purged CV, then dollar-neutral decile book.
- Pooled OOS AUC = **0.574 (L) / 0.592 (S)** — apparently strong skill.
- Tradeable books: momentum (long top score) **Sharpe −0.89**; reversal (long
  bottom score) **−2.25** (−100% MaxDD = ruin); raw no-ML 20d reversal **+0.06**.
- **Breadth HURT**: raw reversal fell +0.30 (12 names) → +0.06 (99 names) — the
  wider universe adds high-vol small alts where reversal is bid-ask-bounce/idio
  blow-ups that honest costs eat. Opposite of the √N dream.

**C. Meta-labeling done right (LdP central technique).**
`scripts/meta_label_book.py` — primary = cross-sectional reversal (the only signal
with faint + edge); secondary CatBoost predicts p(bet wins), purged CV.
- Meta-model OOS AUC = **0.548**; it *does* filter losers (book −0.40 → −0.19) but
  **cannot turn a ≤0 base signal positive.**

**THE SILENT KILLER (decisive).** Pooled AUC 0.57–0.59 and meta AUC 0.55 *look*
like the breadth lever is working — they are **triple-barrier/path artifacts**: the
model learns "high vol / oversold → more likely to touch a 2-ATR barrier," which is
true but carries **zero directional information**. Every tradeable book built from
these scores is ≤0. This is the §10b "AUC≠edge / construction-artifact" finding,
now reconfirmed at 99 names across three independent model architectures.

**Fundamental Law of Active Management (the rigorous ceiling, Grinold/Bailey).**
IR = IC·√breadth. The measured directional IC is **≈0.02** (often wrong sign). For
IR=3 at realistic crypto breadth (~8 independent assets × ~24 rebalances/yr →
√192 ≈ 14) you need **IC ≈ 0.21** — a 10× gap. Closing it via breadth alone needs
≈22,500 *independent* bets/yr, impossible at daily frequency with ~100 assets that
share one dominant beta factor. **Breadth measures the zero edge more precisely; it
does not create edge.** More models ≠ more Sharpe.

**Verdict (Wave 14).** The 70×2 breadth/ML rework with **per-asset features** is
falsified (book Sharpe −0.59, IC −0.007). BUT this verdict was **premature on
construction** — see Wave 15: the failure was per-asset features learning vol
artifacts, NOT absence of edge.

---

## 13. Wave 15 — cross-sectional rework: REAL signal found (2026-06-09)

Triggered by user (correctly) rejecting Wave 14's "impossible" framing and demanding
genuine construction with persisted models. The methodological gap: every Wave-14
probe used **per-asset features only** → models learned volatility/path artifacts.

**The fix (genuinely new):** cross-sectional features — rank-transform each feature
across the universe daily, BTC-beta residual momentum, market-relative return,
funding-z — and a **relative-winner target** (asset fwd H-day return > XS median).
`scripts/xs_alpha_rework.py`; **6 CatBoost models PERSISTED** to
`artefacts/tracks_breadth/`; OOS scores cached (`xs_oos_scores.parquet`).

**Result — the IC flipped positive and robust:**
- OOS rank-IC = **+0.069**, **positive EVERY year** (2021 +0.159, 2022 +0.068,
  2023 +0.040, 2024 +0.051, 2025 +0.077, 2026 +0.075). Real predictive signal —
  vs Wave-14 per-asset IC −0.007.
- Naive decile book only Sharpe 0.25 → the signal is real but **harvesting** was
  broken (score loads on vol; turnover/tail names eat it).

**Harvesting (`scripts/xs_harvest.py`):** rank-demean + vol/beta-neutralise + EWMA5
smooth + no-trade band → **Sharpe 0.97, +36% CAGR, MaxDD −42%, β −0.07, all 6 years
positive.** A genuine market-neutral sleeve.

**Diagnostics (`scripts/xs_diagnose.py`) — honest character:**
- ex-2022 Sharpe **0.55** (full 0.80); 2022 contributes +94% → crash-alpha tilt.
- **long-only −0.61 / short-only +0.81** → edge is short-side (borrow/squeeze risk).
- corr(ML, low-vol)=+0.03, corr(market)=−0.04 → **orthogonal to BAB**, not just
  rediscovered low-vol.

**Multi-sleeve combine (`scripts/multi_sleeve_combine.py`) — the √N test:**
| Sleeve | Sharpe | β |
|---|---|---|
| ML-XS | 0.71 | −0.02 |
| LOWVOL | 0.87 | −0.29 |
| STATARB | 0.05 | +0.09 |
| TSMOM | 0.20 | −0.12 |
Combined risk-parity (MN) **Sharpe 0.86, +29% CAGR**, β −0.11, but **−46% in 2021**
(bull mania) and **+131% in 2022**. Combining did NOT beat the best single sleeve —
only 2 of 4 sleeves are strong and they partly overlap.

**√N ceiling (demonstrated, not asserted):** avg sleeve Sharpe 0.46 → Sharpe 3 needs
**~43 equally-good uncorrelated sleeves; have ~4** (2 weak). 100% net CAGR still not
reached. **Honest verified product creeps to ~Sharpe 0.9–1.0 / ~50–60% CAGR** with a
genuinely new, persisted ML sleeve. Multi-horizon ensemble (`xs_multihorizon.py`):
H=5/10/20 IC +0.051/+0.053/+0.046, ensemble book Sharpe 0.70 — did **not** beat the
single-horizon; harvested Sharpe swings 0.70–0.97 on smoothing/band params → robust
honest sleeve Sharpe ~0.7–0.8, DSR ~0.03–0.10.

---

## 14. Wave 16 — max the FREE data + feature set (user: "go to the extreme, free")

User challenge: *"is my feature set perfect since you change nothing?"* Correct hit —
prior waves used a thin ~16–34 feature set and **zero new alt-data**. This wave maxes
the free space.

**A. New free alt-data (`scripts/fetch_altdata.py` → `altdata_macro.parquet`, 8 series,
none used before):** Fear&Greed sentiment, DefiLlama TVL + stablecoin supply, BTC
on-chain (active addrs / tx / hashrate), Deribit DVOL (BTC+ETH). (FRED macro network-
blocked here.) All free, point-in-time, lagged 1d.

**B. Directional regime-timing sleeve (`scripts/regime_timing_book.py`):** the alt-data
are MARKET-WIDE → time the market (traded via BTC/ETH basket).
- timing OOS **AUC 0.45, IC −0.10** — alt-data are *anti-predictive / overfit* for
  direction (17 feats on ~1625 daily samples; the simpler DVOL-only timer beat it).
- combined timing+XS Sharpe 1.24 / +52% CAGR **but a 2022 artifact** (+198% that yr,
  ~flat else, DSR 0.21). Not robust. Market timing has breadth ≈ 1 → sample-starved.

**C. Max cross-sectional feature set (`scripts/xs_maxfeat.py`, 52 feats):** added
fractional-diff price, downside/vol-of-vol, Amihud illiquidity, drawdown/52w-high,
per-asset DVOL-beta & BTC-corr (how market-wide alt-data enters a XS model), + ranks.
- 20-asset smoke: IC **+0.015 (WORSE than baseline)**, book −0.33 — new features
  overfit at small sample.
- **Full 99-asset: IC +0.064 (≈ baseline +0.069, no gain)**; per-year IC all + ;
  top feature `xr_ret5` (XS short-term reversal) dominates (12.5) → the signal was
  always ~one thing (XS reversal + illiquidity + vol). Harvested Sharpe 1.16 but
  **less robust** (2021 −47%, 2022 +164%, MaxDD −61%, DSR 0.196) — higher Sharpe is
  bought with crash-concentration, not edge.

**Verdict (Wave 16).** Maxing the free data + feature space does **not** raise the
ceiling: market-wide alt-data is anti-predictive for timing (breadth≈1), and piling
features onto the XS model overfits (IC drops). The binding constraint is **signal-
to-noise of daily crypto cross-sectional returns + the universe being one beta factor
(low effective breadth)** — NOT feature richness. The feature set is not "perfect,"
but the cross-sectional rework (Wave 15) already captured the available IC (+0.069);
neither more features nor more free data exceed it. Honest product stays ~Sharpe
0.8–1.0 / ~50–60% CAGR.

---

## 15. Wave 17 — multi-timeframe / RR trend + the "every year >60%" proof (2026-06-09)

User: experiment with RR ratios & bars/timeframe (bigger TF = more trend); demand a
book that makes **>60% EVERY year** (no 2022-only / 2021-bleed concentration),
"extremely robust, methodologically 100% watertight."

**A. Multi-TF / RR sweep (`scripts/trend_robust_sweep.py`, 40 configs, walk-forward).**
TF {1,2,3,5,7}d × lookbacks {fast,mid,slow,blend} × {long-short, long-flat}, vol-
targeted, 10bps, DSR **deflated by n_trials=40**.
- **No config makes >60% every year.** Most-robust (tf2/fast/LS): +5/+3/+14/−5/+14/
  +6% — every year near-flat but Sharpe 0.37.
- Walk-forward winner (picked on IS≤2023) **loses OOS** (−0.28), DSR 0.052 — sweep
  overfits. Bigger TF → more 2022 trend (+60–86%) but bigger 2024 whipsaw (−27..−37%).
- RR (PT:SL) sweep (`robust_combine.py`): no material effect — trend Sharpe ~0.45
  across 1:1…1:3; RR does not fix robustness.

**B. Best robust combination (`robust_combine.py`):** TREND(0.37) + XS-REV(1.36) +
LOWVOL(0.87), risk-parity → **Combined Sharpe 1.21, MaxDD −43%, DSR 0.232**, per-year
2021 **−15%** / 2022 +110% / 2023 +7% / 2024 +46% / 2025 +115% / 2026 +29%. Still one
negative year and heavily 2022/2025-weighted — NOT every-year >60%, not even all-+.

**C. THE PROOF — "every year >60%" ⟺ Sharpe ≈ 3 (the same barrier).** For ~Gaussian
annual returns, P(year>60%)=0.95 needs μ − 1.645σ > 0.60. With μ = S·σ:
| Sharpe S | feasible? | required vol |
|---|---|---|
| 1.0 | **IMPOSSIBLE at any vol** | — |
| 1.5 | **IMPOSSIBLE at any vol** | — |
| 2.0 | only at vol 169% (ruinous) | 169% |
| 3.0 | yes | 44% (mean 133%) |
| 4.0 | yes | 25% (mean 102%) |
Below Sharpe ~1.65 the worst year **cannot** be held above 60% at *any* leverage —
raising vol raises the mean but raises the downside faster. So "**>60% every year**"
is **mathematically equivalent to (slightly stricter than) the Sharpe ≈ 3 / 100%-CAGR
barrier.** It is not an easier, different goal — it is the *same* wall.

**Verdict (Wave 17).** Demonstrated 6 independent ways now (MN √N-cap, directional-ML-
no-edge, breadth-artifact, XS-rework-capped, max-free-data-no-gain, multi-TF-trend-
weak) that free-data crypto caps at **Sharpe ~1–1.5**. The robustness target reduces
to Sharpe ≈ 3, which the same evidence rules out. Best honest robust product:
**combined Sharpe ~1.2 / ~50–60% CAGR**, MaxDD −43%, with one ~−15% year — real and
deployable, but the worst year is ~−15%, not +60%. The only levers that reach Sharpe
3 (hence >60%/yr) remain the two the mandate forbids: **paid data** (IC) and
**HFT/L2** (breadth).

---

## 16. Wave 18 — adaptive walk-forward forward test (2026-06-09)

User (valid): one static strategy across 5 regimes is the wrong test; judge
robustness FORWARD for the coming year, recency-weighted, current regime; backtest =
validation. Built exactly that (`scripts/walkforward_forward.py`): rolling quarterly
refit, **train only on past < t0−embargo**, exp **recency weights** (half-life 365d),
**regime guard** de-grossing the short-biased book in bull manias. Strict no-future-
leak.

**Forward result:** OOS AUC 0.546, IC **+0.044** (vs in-sample CPCV +0.069 — forward
is harder, as expected). Forward book **Sharpe 0.60, MaxDD −43%, DSR 0.018 (≈0).**
- forward per-year: 2023 +24 / 2024 +32 / 2025 +48 / **2026 −7%** (YTD).
- "last-12-months +108% / Sharpe 1.51" **is ONE quarter**: 2025-08 **+122%**; the
  surrounding recent quarters are 2025-11 **−20%**, 2026-02 **−12%**, 2026-05 +13%.

**Verdict (Wave 18).** Even with the correct forward methodology and current-regime
weighting, the edge is **lumpy / dislocation-driven, not forward-robust** (forward
DSR ≈ 0; 2026 YTD −7%; recent quarters alternate ±). Recency-weighting cannot smooth
it because the only free-data edge (short-junk XS reversal) is *intrinsically*
regime-burst alpha — it pays in occasional deleveraging events and bleeds in calm/
bull periods. Forward robustness needs either a *smooth* edge (none here) or *many
uncorrelated* edges (√N — none here). The headline "+108%" is exactly the cherry-pick
that integrity forbids selling as repeatable. Honest forward expectation of the
deployable book: **~Sharpe 0.6–1.0, positive over time, but with −15..−20% quarters
and rare +50..120% bursts — a diversifier, not a smooth >60%/yr machine.**

---

## 17. Wave 19 — deploy-grade adaptive walk-forward module (AFML, overfit-proof)

Built the adaptive architecture as a proper, tested, integrated module (not a
script): **`src/tradebot/alpha/adaptive_wf.py`** (`AdaptiveWalkForward`),
**`apps/run_adaptive_wf.py`** (R-6 CLI), **`tests/test_adaptive_wf.py`** (4 guards
PASS: no-lookahead, determinism R-5, recency-weight monotonicity, dollar-neutral).

**Overfit controls (all wired):** pure walk-forward (train < t0−embargo only),
purge+embargo, **AFML sample weights** = per-asset uniqueness × return-magnitude ×
recency decay (`cv/uniqueness.py`), **frozen MDA** feature selection on the first
window (`selection/mda.py`, 28/34 kept), regime-guarded harvest, **DSR deflated by
variant count + CSCV PBO** (`backtest/pbo.py`), deterministic.

**Forward result (14 quarterly refits, recency HL 365d, 10 bps, 40% vol):**
- **Forward Sharpe 0.69, MaxDD −38%** (regime guard: −43%→−38%).
- **DSR 0.533 @ n_trials=8; PBO 0.222** — *engine warns S=8 < Bailey-LdP floor of 50
  → PBO indicative only.*
- forward per-year: 2023 +18 / 2024 +58 / 2025 +63 / **2026 −11% (YTD)**.
- "last-12m +92%" = **one quarter** (2025-08 +102%; neighbours −21/+14/−15/−14/+5).

**Conclusive PBO (`scripts/adaptive_wf_pbo.py`, 72 harvest variants, S≥50 floor met):**
**PBO = 0.51** (CSCV, 12,870 combinations). The indicative S=8 value (0.22) was
optimistic; at proper scale the **harvest-construction selection has ~no OOS skill**
(config ranking is noise). The deployed config ranks 56/72 by IS Sharpe — correctly
NOT the in-sample-best, but PBO 0.51 says the tuning adds no robust value.

**Verdict (Wave 19).** The architecture is correct and now deploy-grade, fully
overfit-instrumented — and the instrumentation returns a sober verdict: forward Sharpe
0.69 but **DSR 0.53, PBO 0.51, 2026 YTD −11%, lumpy** (one quarter carries each good
year). The overfit controls did their job: they flag the book as **fragile, not
robustly deployable as a standalone >60%/yr machine.** The signal (forward IC +0.044)
is real but weak; the harvest tuning is noise. Honest product: a **non-stationary-
aware crash/dislocation diversifier, ~Sharpe 0.7, to be sized small inside a broader
book — not a smooth compounder.** Consistent with the Wave-17 proof (>60%/yr ⟺ Sharpe
≈ 3) and the 6-fold demonstration that free-data crypto caps at Sharpe ~1–1.5.
