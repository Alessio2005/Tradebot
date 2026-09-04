# PRE-REGISTRATION — Setup B (futures book with ML overlay), Wave 28

> **GEARCHIVEERD — historisch document.** Dit is een verslag van de pre-registratie van Wave 28 en beschrijft de toestand van toen. Het wordt NIET bijgewerkt: de paden en artefacten die het noemt, zijn die van die periode en bestaan grotendeels niet meer. Voor de huidige toestand, zie `docs/PROJECT_STATE.md`.
> *Als historisch gemarkeerd op 2026-09-01 (Phase 7/8, Stage E-3).*

**Status:** FROZEN on 2026-08-10 before any Setup-B unit was measured.
**Authority:** `SETUP_B_ML_PROMPT.md` §6 Fase 0 · `ULTIMATE_GOAL_PROMPT.md` §11.
**Ledger at freeze:** `total_n_hypotheses = 2706` (restored, see §1).

Anything measured after this freeze is judged by the thresholds in §6. Changing
a threshold afterwards because a result disappointed is `/goal` §1.3 fraud. The
allowed move is to de-scope (drop a unit, honest archive), never to relax.

---

## 0. LABEL ASSIGNMENT (was blocking; now decided)

| Label | Assignment as of 2026-08-10 |
|---|---|
| **Setup A** | crypto market-neutral book — unchanged, 4 accepted units, Sharpe ~1.15 |
| **Setup B** | **commodities / cross-asset futures with an ML sizing overlay** — this document |
| **Setup C** | **VACANT.** Retired, not reassigned. |

### 0.1 What is archived, explicitly

Both of the prior claimants are archived. `SETUP_B_ML_PROMPT.md` §0 asks which
one; the answer is **both**, for different reasons:

1. **`Desktop/Trading Setup B/MASTER_PLAN_SETUP_B.md` — "Positioning & Events"
   book (Rev 2, 2026-08-03).** SUPERSEDED. That programme reached a documented
   `NULL` after seven designs (B-1…B-7) and its own closing finding was
   structural, not sleeve-choice: across 38 instruments the 5th-percentile
   worst 12-month drawdown is 2.30x annualised vol, so a 10% max-drawdown cap
   is a 4.4% vol ceiling and 8%/yr inside it needs a sustained Sharpe of 1.84.
   Nothing in this pre-registration reopens it.

2. **Setup C — FX G10 carry.** DISSOLVED. `Desktop/Trading Setup C/` is an
   empty directory (verified 2026-08-10, 0 items); the plan file is gone.

### 0.2 A claim that does not survive verification — recorded, not inherited

`docs/EXPANSION_RESEARCH_2026-08-10.md` §0 warns against overwriting Setup C
because its FX carry unit was *"already accepted at Sharpe 0.40, rho~0.03 vs
crypto"*. **That claim is unsupported in this repository and is not carried
forward.** Evidence:

* the restored ledger contains **no FX entry of any kind** (7 entries: W20
  crypto, W22 x2, W23c, W24, W26, W27);
* `docs/WAVE_LOG.md` states the opposite twice — W22 close lists FX carry as an
  *open* next path with "unit-code klaar" (code ready), and W27 records
  "W25 is nooit afgerond" (never completed);
* no `artefacts/killgates/fx_carry.json` exists.

The number came from the now-deleted plan document, not from a run in this
program. This is the same failure mode the project already logged once (a
verdict decided on a premise that was never in the source text), so it is
written down rather than quietly inherited. **There is no accepted FX carry
unit.** `alpha/fx_carry.py` is unevaluated code, and `cm_carry` will therefore
be built as a NEW module (per §3.3 of the mandate) rather than by mutating it.

---

## 1. GOVERNANCE REPAIRS COMPLETED BEFORE THIS FREEZE

| # | Defect | Resolution |
|---|---|---|
| 0.1 | canonical `hypothesis_ledger.json` missing from both working copies | Rebuilt by `research/w28_seed_ledger.py` from the itemised wave-log chain (seed 2363 + 6 reconstructed wave rows = 2702), then W27 merged from its staging file through the sanctioned CLI path -> **2706**, matching the two independent totals stated in the log. Config hashes for W20-W26 did not survive; rows are flagged `RECONSTRUCTED` and are wave-granular. Trial COUNTS are exact. |
| 0.2 | phantom rebalancing in `alpha/xs_unit.py`, `alpha/fx_tsmom.py` | **REPAIRED** (not scoped) — see §2. |
| 0.3b | KG-B1 drawdown-shape constants rest on the wrong distribution | **RE-DERIVED** before measuring anything — see §3. |

---

## 2. PHANTOM REBALANCING — decision and evidence

`groupby(period).max()` derived the rebalance calendar from the data, so the
final bar of any panel was always the max of its period and therefore a
rebalance that live trading would not do.

**Decision: repair, not scope.** Two reasons, in order of weight:

1. It is a **truncation-variance (causality) defect, not a cost approximation.**
   In a single full-sample backtest it fires once. Under `AdaptiveWalkForward`
   — the mandate's standard harness for every unit with a learned component,
   quarterly refit — a fold boundary is exactly "where the data happens to run
   out", so it fires at **every fold**. Fase 2 and 3 run those code paths.
2. The defect only *adds* turnover cost, so archived numbers were pessimistic,
   and `eq_strev_1m` was archived at **+0.39 against a 0.40 lat**. A 0.01 gap
   must be measured, not asserted.

**Repair:** the calendar is decided by comparing with the NEXT bar. Identical to
the old behaviour on every complete period; at the ragged edge the next period
is unknowable, so the final bar never rebalances. `cm_tsmom` keeps its own
first-bar-of-month variant (already correct, and its recorded numbers stand).

**Measured impact** (`scripts/w28_phantom_rebalance_impact.py`): the final
weights row enters net P&L only through `turnover[-1]`, so the delta is exactly
one number at one bar.

| quantity | value |
|---|---|
| empirical delta-Sharpe, real 26-instrument panel, n=5685 | **+0.000226** |
| analytic worst case, equity config (full decile flip, 7bp/side, n=6500) | **+0.000543** |
| `eq_strev_1m` distance to its lat | 0.0100 |

**Verdict: NO archived verdict flips** — the repair gain is ~18x too small.
F13-F18 stand as recorded and are not re-frozen.

**Test debt cleared:** `tests/lookahead/test_eq_units_causality.py` previously
*excluded* the final partial month from its truncation comparison, calling the
difference "by construction". It was not construction, it was this defect. The
carve-out is deleted and invariance now holds over the whole common index.
`tests/lookahead/test_fx_tsmom_causality.py` is new — `fx_tsmom` had no
lookahead test at all, which is why the defect survived there unnoticed.

---

## 3. KG-B1 DRAWDOWN-SHAPE RE-DERIVATION (done BEFORE measuring — §5.3 order)

### 3.1 The defect

`tests/killgates/test_expansion_killgates.py::calmar_ceiling` derived its
ceiling from `E[MaxDD] = sigma^2/(2*mu)`, i.e. `Calmar <= 2*S^2`. That identity
is real but it is **the mean drawdown at a random time** — the stationary law of
a reflected Brownian motion with negative drift. It is not the expected
*maximum*, and for reflected BM the all-time supremum is a.s. unbounded, growing
logarithmically in the horizon.

Measured (`research/w28_dd_shape_calibration.py`, seed 42, 20 000 iid-normal
paths, 5685 bars = 22.6y, Sharpe 0.40):

| quantity | vol units |
|---|---|
| what `1/(2S)` predicts | 1.25 |
| drawdown at a random time (what the formula actually describes) | **1.04** |
| MAXIMUM drawdown (what a gate sees) | **3.25** |
| the same at a 200-year horizon (it keeps growing) | 5.36 |

### 3.2 The consequence, measured

The gate then compared a **single realised** drawdown against constants derived
from that mean. False-rejection rate against strategies that *genuinely* have
the required Sharpe, over the registered 22.6-year horizon:

| true Sharpe | rejected by `dd_over_vol <= 2.5` | rejected by `calmar >= 0.25` | rejected by either |
|---|---|---|---|
| 0.40 | 74.4% | 99.7% | **99.7%** |
| 0.50 | 65.5% | 92.4% | 92.4% |
| 0.60 | 54.7% | 65.2% | 65.2% |
| 0.80 | 34.1% | 11.9% | 34.1% |
| 1.00 | 19.1% | 0.8% | 19.1% |

At its own Sharpe floor the gate rejected **99.7%** of qualifying strategies. It
was not a gate, it was a wall — and the consistency checker that existed
precisely to catch unsatisfiable specs passed it, because it carried the wrong
constant. A checker that green-lights an infeasible spec is worse than none: it
converts a design error into confidence.

### 3.3 The replacement

One criterion, at a stated false-rejection budget:

> `dd_over_vol` must not exceed the **q=0.95 quantile** of the maximum drawdown
> that a strategy with **this unit's own realised Sharpe**, over **this unit's
> own sample length**, at **this unit's own vol**, would produce under an
> iid-normal null. Budget: 5% by construction.

* Reference lives in `src/tradebot/backtest/dd_shape.py` (R-2). It **is** a
  deterministic simulation (seed 42, cached), not a fitted formula — a fitted
  closed form reproduced the null to ~5% at low Sharpe but drifted to ~16% at
  Sharpe 1.0, and a gate whose threshold approximates the thing it measures can
  silently change meaning.
* The null is iid normal — no fat tails, no vol clustering — so real strategies
  draw worse maxima and the cap is **conservative**, never generous.
* **The Calmar floor is dropped, not relaxed.** `Calmar = (S - sigma/2)/dd_over_vol`
  is a deterministic function of the Sharpe floor and the shape cap; keeping all
  three triple-counted one piece of evidence, which is what made the set
  unsatisfiable.
* This is a **shape** gate ("is this drawdown pathological given the skill on
  display"), explicitly *not* a risk limit. Absolute drawdown caps (§7) are
  independent and untouched.

### 3.4 The discipline check — this must not resurrect anything

Under the corrected criterion **`cm_tsmom` clears KG-B1.** That is exactly the
situation the mandate warns about, so it is handled explicitly:

* `cm_tsmom`'s registered reopening condition (`SETUP_B_ML_PROMPT` §1.1b)
  requires a re-derived threshold **AND** a G4-strict pass.
* G4-strict was **t = 1.75 < 2.0** and is completely untouched by this work.
* **`cm_tsmom` therefore stays ARCHIVED on KG-B2** — the substantive reason it
  was archived in the first place.

Encoded as `test_cm_tsmom_stays_archived_on_g4_strict`. If that test ever goes
green, someone has moved the substantive gate, not the shape gate.

**Honest disclosure of order:** I read `cm_tsmom`'s numbers (they are in the
wave log) before doing this re-derivation, so I cannot claim blindness. The
protections are that the re-derivation is driven by a *provable mathematical
error* verified on synthetic paths that never touch any unit's returns, and
that it is *incapable* of changing the archived verdict.

---

## 4. FROZEN UNIVERSE AND DATA

**Research panel (frozen, no additions without a new pre-registration):**
`market_data_parquet/xasset/tr_panel.parquet` — 26 roll-inclusive total-return
ETF/ETC series, 2004-01-02 -> 2026-08-07, 5685 bars, PIT-validated.

| sector | n | symbols |
|---|---|---|
| commodity | 9 | CPER DBA DBC GLD PPLT SLV UNG USO WEAT |
| equity | 6 | EEM EFA EWJ IWM QQQ SPY |
| rates | 4 | IEF SHY TIP TLT |
| fx | 4 | FXE FXF FXY UUP |
| credit | 2 | HYG LQD |
| real estate | 1 | VNQ |

`N_eff = 5.19` on 26 names (`cm_tsmom.effective_breadth`) — the F10 guard. Any
claim of breadth is measured with this, never with the raw name count.

**Term structure:** EIA NYMEX Contract 1-4 (`.xls` endpoint only — the HTML
`LeafHandler.ashx` route truncates silently). WTI 1983->, Henry Hub NG 1993/94->,
heating oil 1980->, RBOB. **Frozen archive: stops 2024-04-05.** Every carry
conclusion carries the visible label "no recent OOS window".

**Binding data prohibitions (violation = wave void):**
1. Free futures continuations (`CL=F`, `NG=F`, …) are **forbidden as a return
   series** — front-month, not back-adjusted, measured overstatement +7.0%/yr
   (WTI) and +25.1%/yr (nat gas), an order of magnitude above the premium being
   sought. Signal input only, with written justification in the wave log.
2. Expired contracts are gone from Yahoo — the front curve is **not**
   historically reconstructable. Pre-2024-04 term structure comes from EIA or
   nowhere.
3. `pct_change` is always called with `fill_method=None`.
4. As-of joins go through `tradebot.utils.time.asof_join` and nothing else.

**Survivorship:** the ETF panel holds only funds alive today -> mild upward
bias, stated in every report. Not fixable with free data; declared, not hidden.

---

## 5. MODEL SPECIFICATION (frozen)

### 5.1 Cost model — one chain, used from bar 1

1.0 bp commission + 3.0 bp half-spread **per side**, 0.50%/yr borrow on short
notional. ETF TER is already inside the adjusted NAV and is **not** double
counted. A log-return Sharpe is not a result.

### 5.2 The only permitted ML role in this wave

**`cm_judge` — meta-label sizer over `cm_tsmom` (LdP role 1).** The primary
signal stays the literature premium; the model predicts `p(win | state)` and
moves **position size only, never the sign**.

Legitimate here because `cm_tsmom` is net **+0.487, not <= 0** — this is sizing
a live signal, not rescuing a dead one (`/goal` §6.2). That number is stated in
the wave log entry as required.

* features: `train/meta_train.build_judge_features()` + volatility/regime state
* labels: `train/meta_train.make_judge_labels()`
* weights: `cv/uniqueness.get_sample_weights()` (uniqueness x magnitude x recency)
* CV: `cv/cpcv.CombinatorialPurgedCV`, purged + embargoed
* selection: `selection/mda.causal_mda`, **frozen before the final run**
* calibration: `train/calibration.PathSpecificPlattCalibrator` + ECE
* harness: `alpha/adaptive_wf.AdaptiveWalkForward` (quarterly refit, recency
  half-life ~365d, train < t0 - embargo)

Roles 2 (pooled XS) and 3 (stacker) unlock only in that order, and the stacker
must beat `portfolio/risk_parity.erc_weights` OOS or RP wins (F11).

### 5.3 Nothing gets rebuilt

Forbidden to re-implement: CPCV, walk-forward, PBO/DSR, calibrator, cost model,
risk parity, tearsheet, ledger, Sharpe/Calmar helpers, HAC regression. The
complete new-build list is `SETUP_B_ML_PROMPT` §3.3 and nothing else.

---

## 6. THRESHOLDS (frozen — the whole table is reported every wave)

### 6.1 Kill gates

| Gate | Criterion |
|---|---|
| KG-B1 | net Sharpe >= 0.40 · years positive >= 0.60 · `dd_over_vol` <= q0.95 skill-and-horizon-aware shape cap (§3.3) |
| KG-B2 | G4 **S3** (strict) p < 0.05 **and** t >= 2.0 |
| KG-B3 | WF Sharpe >= 0.30 · decay <= 40% · bootstrap P(S>0) >= 0.75 · abs(rho) vs book < 0.30 |
| KG-B4 | sizing error <= 20% of target risk at the **real** capital and **real** contract granularity; the surviving instrument set must clear KG-B1 again |
| KG-B5 | slippage <= 25% of modelled · fill rate >= 95% · >=1 full roll cycle clean |
| KG-P | book Sharpe with B >= without B **+0.10** · combined MaxDD <= 10% · rho(A,B) < 0.30 |

### 6.2 G4 — three nested specs, all three reported, S3 binds

| Spec | Factors |
|---|---|
| S1 | Ken French 6 (Mkt-RF, SMB, HML, RMW, CMA, MOM) |
| S2 | S1 + `PASSIVE` (equal-weight long-only of the same panel) |
| **S3** | S2 + sector passives (equity / rates / commodity) — **binding** |

Reporting S1 alone is the self-factor error already on this program's record.

### 6.3 ML gates

| # | Gate | Threshold |
|---|---|---|
| ML-1 | ML book beats the non-ML primary OOS | dSharpe >= **+0.10** net |
| ML-2 | ML book beats risk-parity OOS | else RP wins (F11) |
| ML-3 | PBO (CSCV, **S >= 50**) over all model variants | < 0.20 |
| ML-4 | DSR at the honest cumulative `n_trials` incl. every Optuna trial | > 0.95 |
| ML-5 | calibrator ECE + monotone sign check | ECE < 0.05, monotone |
| ML-6 | meta-label changes **sizing only** — sign identity with primary | 100% of bars |
| ML-7 | feature set from `causal_mda`, frozen before the final run | no post-hoc additions |
| ML-8 | remove the ML component -> book stays >= 0 | else ML carries it = fragile |

### 6.4 Trial accounting

Every Optuna trial, model variant, feature set, label definition and
hyperparameter sweep increments the ledger, booked **during** the loop
(`apps/ledger_append.py --staging`), merged at wave close. DSR and PBO run every
wave. PBO always at S >= 50.

Expect and accept: an honest DSR at n ~ 3000 makes a t of 2.1 meaningless. That
is not a reason to lower the counter; it is the reason to aim at an effect large
enough to survive the correction, or to close the subspace and move on.

---

## 7. RISK, AND THE CONSTRAINT THAT IS NOT ABOUT SIGNALS

Propfirm account, static caps. Parametrised, never hardcoded:
`risk/daily_loss_governor.py`, `risk/drawdown.py` (6%/3% after the retrofit),
`risk/position_limits.py`, `live/circuit_breaker.py`.

**Capital granularity is the binding constraint, not signal quality.** A
20-instrument program at 10% vol needs ~**$150k-250k** even with CME micros;
below that it collapses to ~5 instruments and `N_eff` implodes — which is the
low-breadth case that lands at Sharpe 0.2-0.3, i.e. the very problem the book
exists to solve. Alternatives (fewer instruments, CFDs under ESMA caps,
rotating selection) are **different strategies** and must be backtested as such.

> **OPEN INPUT REQUIRED BEFORE FASE 1 CLOSES:** the real account capital. §8 of
> the mandate requires this computed *before* Fase 1, not after Fase 5. Recorded
> here as an explicit open item rather than assumed.

---

## 8. WHAT WOULD FALSIFY THIS PROGRAMME

Stated in advance so it cannot be renegotiated later:

* `cm_carry` net Sharpe <= 0 on the EIA archive -> archive with an F-number;
  energy term-structure carry is closed for this program.
* `cm_judge` failing ML-1 (dSharpe < +0.10) -> meta-label sizing does not add
  over the raw premium on this panel; role 1 closes and role 2 does not unlock.
* `cm_judge` failing ML-8 (book <= 0 without the ML component) -> the result is
  carried by the model, which is the fragile case, and it is archived even if
  ML-1 passed.
* KG-B4 failing at the real account capital -> the book is not tradeable at this
  size; that is a capital finding, not a signal finding, and it is reported as
  such rather than fixed by shrinking the universe until the numbers work.
