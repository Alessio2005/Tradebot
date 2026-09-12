# Strategy Design & Audit — Crypto Propfirm Accounts A & B

> ## ⚠️ ACHTERHAALD — HISTORISCH DOCUMENT
>
> **Er wordt niet meer met propfirms gewerkt.** De accountregels die dit
> document als bindend behandelt ("static 10 % max-DD + 5 % daily", §6.3) zijn
> geen contract meer. Geldend beleid: `docs/RISK_MANDATE.md` +
> `conf/risk/default.yaml`.
>
> §6.3 ("Propfirm fit") is daarmee in zijn geheel achterhaald, inclusief de
> claim **"Status: enforced in engine"** — de governor is opt-in en staat
> default uit (RISK_MANDATE §5.4).
>
> Bewaard als herkomst, niet als geldende eis.


**Author:** CHIEF (Quantitative Architect / Head of ML)
**Date:** 2026-06-14
**Scope:** Concrete strategy specification for the two crypto propfirm accounts
from [PROPFIRM_PARALLEL_AUDIT](../src/tradebot/artefacts/PROPFIRM_PARALLEL_AUDIT.md),
built **on the existing ML stack** (meta-labelled CatBoost on runs-bars). Venue:
Bybit linear perps (Hyrotrader). Account rules: static 10 % max-DD + 5 % daily.
Governor enforced via `risk/daily_loss_governor.py`.

This is a design + audit document, not a deployment sign-off. Sizing uses the
**sober** edge estimate (see §6.6), not the optimistic CPCV number.

---

## 1. The shared ML engine (what your models actually are)

Confirmed from `conf/conf_config.yaml` + `src/tradebot/{labeling,live,bars}`:

| Layer | Spec |
|---|---|
| Base data | 5 s OHLCV from Bybit `publicTrade` (taker buy/sell volume preserved) |
| Bars | **Volume-runs bars** (AFML §2.2), `target_micro_bars=20`, `runs_atr_floor=0.0005` → ~45 min avg, information-driven (not clock) |
| Event sampling | Incremental **CUSUM** filter, per-symbol `cusum_threshold_multiplier` (BTC 1.5, AVAX 2.2, separate SHORT multiplier) |
| Primary labels | **Triple-Barrier**: PT=2.0·ATR, SL=1.0·ATR (asymmetric: `pt_mult=1.25`, `sl_mult=0.85`), `label_horizon=25` runs-bars, candidates [15,20,30], `vol_window=20`, trend-scan `min_tstat=2.0`. Half-spread + execution-delay baked into the barrier (realistic fills). Side-aware → separate LONG / SHORT. |
| Meta-labels | Binary {1 = primary trade profitable net of `profit_threshold_bps=2.0`; 0 = SL/timeout}. **Judge** = CatBoost (300 iters, depth 5, L2=5.0), `min_tstat_filter=1.5` |
| Model | Per-symbol, per-side **CatBoost ContextualBanditEnsemble** (CPCV-trained) + **Platt** calibration; raw features to CatBoost (`use_pca_for_catboost=false`), PCA-orth(0.95) to the bandit |
| Features | 111 (40 micro / 32 meso / 39 macro): runs-features, FFD (auto-`d`, rolling calibration), funding-carry block, **+ new Open-Interest block** (`features/open_interest.py`) |
| Regime filter | EMA-200: SHORT suppressed when close > EMA-200 (no bull-regime false shorts) |
| Portfolio | **HRP** over events, `hrp_lookback_bars=2000` (~83 d), weights ∈ [0.05, 0.40] |

**Key fact:** the per-symbol models predict **absolute direction** (LONG/SHORT
meta-labelled). Both accounts reuse the SAME trained models — the difference is
how the output is *combined into positions* (§2, §3). That shared dependency is
audited as a risk in §6.5.

---

## 2. Account A — ML Cross-Sectional Market-Neutral

**Goal:** harvest the model's *relative-ranking* skill with ~zero market beta →
the profile the static-DD / daily-loss limits reward.

| Item | Specification |
|---|---|
| Assets | ETH, SOL, AVAX, LINK, DOT (the trained `training_universe`; the validated 5-asset HRP set) |
| Bars / TF | Runs-bars (~45 min), CUSUM-event-driven; portfolio rebalance daily on HRP cov of last 2000 events |
| Signal | Per-asset calibrated edge `e_i = P_long,i − P_short,i` from the existing models |
| Construction (v1, **no retrain**) | Cross-sectional demean → long top-2 `e_i`, short bottom-2, 1 name flat; **dollar-neutral** (Σw=0, Σ\|w\|=1) + **BTC beta-hedge** (`risk/beta_hedge.compute_btc_hedge_size`) to drive net beta → 0 |
| Labels | **v1:** existing absolute-direction triple-barrier labels (MN achieved by *construction*, not by relabelling). **Upgrade (v2):** retrain primary on **residualized** labels — triple-barrier on `r_i − mean_j(r_j)` so the model predicts relative outperformance directly (precedent: `alpha/eq_strev_resid.py`, `alpha/neutral_book.py` residual reversal). |
| Sizing | HRP weights × governor `target_gross_multiplier(target_vol=8–9 %, realised)` → gross ≈ 0.55–0.65× native (funded) / 2× (challenge) |
| Expected (sober, funded) | Sharpe ~0.8–1.1, beta ~0, MaxDD ~6–7 % |

**Why MN here:** going long the highest-edge and short the lowest-edge asset
nets out the common (market) component the model is partly predicting, leaving
the relative-value residual. Beta-hedge mops up the rest. This is the deployable
path that **uses the existing models without retraining**.

---

## 3. Account B — ML Directional Crisis-Alpha

**Goal:** harvest the model's *absolute-direction* skill on majors, with a
trend/vol overlay that makes it convex (profits in crashes that hurt A).

| Item | Specification |
|---|---|
| Assets | BTC, ETH (highest liquidity; where directional timing + dvol works) |
| Bars / TF | Runs-bars (~45 min) for ML entries; **meso overlay**: multi-MA trend {50,100,200} + Deribit DVOL z-score (`alpha/multi_sleeve_book.py` TREND+DVOL sleeves) |
| Signal | Existing per-symbol LONG/SHORT models (CUSUM-gated, Judge-filtered, Platt-calibrated, EMA-200 short filter) → directional position; trend sign and DVOL contrarian as a regime gate/scaler |
| Labels | Existing **absolute-direction** triple-barrier (PT 2.0 / SL 1.0 ATR) + meta-labels. No relabelling needed. |
| Sizing | gross ≈ **0.30–0.35×** native (native MaxDD −24 % = 2.4× the floor → must downscale hard); governor + daily-loss intraday stop mandatory |
| Expected (sober, funded) | Sharpe ~0.6–0.9, beta +0.1, **convex** (positive on crash days) |

**Why directional here:** the trend+DVOL overlay turns the model's timing into a
crisis hedge — when A (market-neutral) earns its carry in calm regimes, B pays
off in the violent regimes, and vice-versa. Low correlation *by exposure*, not
by signal source.

---

## 4. Theses (falsifiable)

- **T-A1 (cross-sectional skill):** the meta-labelled model ranks the 5-perp
  cross-section better than chance → a dollar+beta-neutral long-top/short-bottom
  book has positive IC on *residual* returns. **Falsified if** residual IC ≤ 0.02
  out-of-sample (the noise floor from `project_edge_investigation_20260531`).
- **T-A2 (MN by construction):** long-top/short-bottom on absolute-direction edge
  yields realised \|beta\| < 0.15 after BTC-hedge. **Falsified if** rolling 60-day
  beta > 0.15.
- **T-B1 (directional timing):** the LONG/SHORT models + trend/DVOL overlay are
  net-positive on majors after Bybit costs. **Falsified if** OOS Sharpe ≤ 0 or the
  SHORT leg is net-negative EV (consistent with `project_short_alpha_research`:
  naive trend-shorts bleed → shorts must come from DVOL/regime, not raw ML short).
- **T-AB (orthogonality):** corr(A, B) daily P&L < 0.3. **Falsified if** the
  cross-account monitor (`monitoring/cross_account.py`) repeatedly throttles —
  meaning the shared model risk (§6.5) dominates.

---

## 5. Bars, timeframes & labels — summary table

| | Account A (MN) | Account B (directional) |
|---|---|---|
| Base | 5 s Bybit publicTrade | 5 s Bybit publicTrade |
| Trading bar | volume-runs (~45 min) | volume-runs (~45 min) |
| Overlay TF | — | multi-MA {50,100,200} + DVOL z(90) |
| Event trigger | CUSUM (per-symbol) | CUSUM (per-symbol) |
| Primary label | triple-barrier (v1 absolute / v2 residual) | triple-barrier absolute, PT 2·ATR / SL 1·ATR |
| Meta-label | profit > 2 bps net | profit > 2 bps net |
| Rebalance | daily HRP (2000-event cov) | event-driven + daily |

---

## 6. AUDIT (CHIEF)

### 6.1 Causality / look-ahead
Runs-bars + CUSUM are causal by construction; FFD uses **rolling** calibration
(no global-window leak, `rolling_calibration: true`); features `shift(1)`
(funding + OI blocks verified); HRP cov uses only past events; triple-barrier
horizon ≤ available data. Covered by `tests/lookahead/*` and the OI no-lookahead
test. **Status: PASS** (re-run lookahead suite after any relabelling).

### 6.2 Overfit / DSR
5 assets × 2 sides × meta-labels = a real multiple-testing surface. The Deflated
Sharpe must be computed with the full trial count **including** the OI features
just added (each new feature inflates N). The historical CPCV book used
`label_min_tstat=2.0` and meta `min_tstat_filter=1.5` as built-in guards.
**Action:** re-run DSR with updated N before sizing up.

### 6.3 Propfirm fit
- Account A beta~0 → daily-loss/static-DD friendly (the intended design).
- Account B native MaxDD −24 % → **hard downscale to 0.30–0.35×** is non-negotiable;
  the daily-loss governor (`PropfirmGovernor`) is the backstop, not the plan.
- Both: governor enforces soft 3.5 % / hard 4.5 % daily and static-DD trip 6 % /
  resume 3 % (buffered inside the firm's 5 %/10 %). **Status: enforced in engine.**

### 6.4 Costs
Bybit non-VIP taker = **5.5 bps** (higher than the old Binance 5.0 assumption —
`execution/fees.py` updated). Runs-bar turnover is moderate (~45-min bars). The
triple-barrier already bakes half-spread + execution-delay. **Action:** confirm
the backtest cost model uses 5.5 bps (not the legacy 5.0) before the go/no-go.

### 6.5 Shared-model risk (the honest flag)
A and B consume the **same trained models**. A model-degradation event (regime
break, feature drift, calibrator decay — cf. the `project_chief_audit_20260528`
degenerate-calibrator incident) hits **both** accounts simultaneously, defeating
the "independent accounts" premise. Mitigations: (i) `monitoring/cross_account.py`
throttles the lower-Sharpe account when corr spikes; (ii) the FX account C
(different asset class) is the true diversifier; (iii) drift monitors
(`monitoring/live_drift_monitor.py`) + champion-challenger must gate both.
**This is the dominant residual risk — do not pretend A⊥B is free.**

### 6.6 Edge reconciliation (sober vs optimistic)
The memory holds conflicting numbers: 5-asset HRP **CPCV OOS Sharpe 3.40**
(`project_portfolio_results`) vs **deploy-grade walk-forward Sharpe 0.69**
(`project_true_alpha_mandate`, adaptive_wf) vs trend-model **AUC 0.51 / IC 0.02**
(`project_edge_investigation`). The CPCV 3.40 is optimistic (label/calibration
choices, no live frictions); the **walk-forward 0.69 and the AUC 0.51 are the
sober reality**. **All sizing in this document assumes Sharpe ~0.7–1.1**, not 3.4.
If live shadow confirms higher, scale up — never the reverse.

### 6.7 Short side
`project_short_alpha_research`: naive trend-shorts are structurally negative-EV.
Therefore Account B's short exposure must come from the **DVOL/regime gate +
EMA-200-filtered SHORT model**, not from naive ML short signals. Account A shorts
are *relative* (short the weakest of 5), which is benign.

---

## 7. Go / No-Go gates (per account)

1. Re-run lookahead suite after any relabelling → must PASS.
2. Re-run DSR with updated trial count (incl. OI) → DSR > 0 at 95 %.
3. Backtest with Bybit **5.5 bps** cost → Sharpe > 0 net, MaxDD within governor floor.
4. **14-day shadow** (paper, Bybit live feed, governor active): 0 simulated
   daily/static-DD breaches; realised \|beta_A\| < 0.15; corr(A,B) < 0.3.
5. Only then: Account A challenge → fund → add B.

## 8. Open items
- Widen Account A cross-section beyond 5 names (train ~8–10 liquid perps) — a
  5-name cross-section is thin (long-2/short-2). Highest-value enhancement.
- Account A v2 residualized labels (retrain primary on relative returns).
- Vol-target multiplier fully wired into `ExecutionController` sizing.
- OI economic validation (IC/DSR on real Bybit data) — plumbing done, run pending.
