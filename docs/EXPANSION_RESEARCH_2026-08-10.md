# MULTI-ASSET EXPANSION RESEARCH — Setup B (Commodities) & Setup C (Equities)

> **GEARCHIVEERD — historisch document.** Dit is een verslag van de expansie-research van augustus 2026 en beschrijft de toestand van toen. Het wordt NIET bijgewerkt: de paden en artefacten die het noemt, zijn die van die periode en bestaan grotendeels niet meer. Voor de huidige toestand, zie `docs/PROJECT_STATE.md`.
> *Als historisch gemarkeerd op 2026-09-01 (Phase 7/8, Stage E-3).*

**Author:** Lead Quantitative Researcher / Systems Architect (agent)
**Date:** 2026-08-10
**Input:** "Tradebot Multi-Asset Research Prompt v2.0"
**Binding context:** `docs/FALSIFICATION_REGISTER.md` (F1–F19), `docs/DATA_REGISTER.md`,
`docs/AUDIT_WAVES20-25_2026-06-11.md`, `docs/WAVE_LOG.md`, ledger n = 2702.

---

## 0. EXECUTIVE SUMMARY — GO / NO-GO

| Setup | Prompt's proposal | Verdict | One-line reason |
|---|---|---|---|
| **B — Commodities** | GC/CL/NG (+SI/ZW), trend+carry+seasonal, daily bars | 🟡 **CONDITIONAL GO — rescoped** | The premium class is the one thing the register explicitly leaves *open* (F7/F8 reopening conditions name managed futures). But "commodities only, 5 names" throws away most of the breadth that makes it work. Go as **cross-asset futures TSMOM + energy carry**, not as a commodity silo. |
| **C — Equities** | SPY/QQQ/IWM + 8 sector ETFs, XS-momentum + gap-fade | 🔴 **NO-GO as specified** | Both proposed signals are already falsified in this repo on a *strictly larger* universe (F13 XSMOM net −0.34; F15 overnight net −0.60, with post-hoc sign-flip explicitly forbidden). The proposed 11-ETF universe has ~2–8× **less** effective breadth than the ~190 names that already failed. And EEA retail cannot legally buy the proposed instruments (PRIIPs). |

**Top 3 risks (new, discovered during this research):**

1. **Capital granularity kills diversified futures.** A 20-instrument program at 10% vol needs ≈ **$150k–250k** to size each leg correctly *even with CME micros* (§1A.5). Below that you are forced back to ~5 instruments — which is the low-breadth case that lands at Sharpe ≈ 0.2–0.3. This is the binding constraint on Setup B, not signal quality.
2. **PRIIPs blocks the entire Setup C instrument list** for EEA retail (SPY, QQQ, IWM, XLK…). This is a G10 execution failure independent of alpha (§1B.2).
3. **Free full term-structure data effectively does not exist outside energy.** Nasdaq Data Link CHRIS is deprecated/restricted, Stooq's download endpoint is blocked (already in your register). The one genuine free, decades-deep, PIT term structure is **EIA's NYMEX Contract 1–4 series** for WTI/NG/HO/RBOB (§1A.2).

> **CORRECTED 2026-08-10 by Wave 27 measurement — read §1A.2 with these two fixes:**
> **(a)** EIA is a **frozen archive**: the series stop at **2024-04-05** and EIA no longer publishes NYMEX futures prices. Excellent for a 1980–2024 carry backtest; unusable as a live feed, and it leaves the most recent 2.3 years — the window where decay shows up — uncovered.
> **(b)** Free continuous futures (`CL=F`, `NG=F`) are front-month and **not back-adjusted**: measured against the real rolled vehicles they overstate returns by **+7.0%/yr (WTI)** and **+25.1%/yr (nat gas)**. They cannot be used for P&L. The route that works is the mandate's own §5.4 ETF-proxy panel (roll-inclusive NAVs), built and verified in Wave 27.

**Naming collision — resolve before anything is built.** "Setup B" and "Setup C" are already
taken in this program by *different* things:
`Desktop/Trading Setup B/MASTER_PLAN_SETUP_B.md` = "Positioning & Events" book (Rev 2, 2026-08-03, Phase 0);
`Desktop/Trading Setup C/SETUP_PLAN_SETUP_C_fx_carry.md` = FX G10 carry, **already accepted at Sharpe 0.40, ρ≈0.03 vs crypto**.
The v2.0 prompt silently reassigns those labels to commodities/equities. Adopting it as written
would overwrite an accepted unit with a falsified one. Recommendation: name the new work
**Setup D (futures)** and leave B/C as they stand — or explicitly retire the old plans first.

---

## 1. CORRECTIONS TO THE PROMPT'S PREMISES

These are not stylistic quibbles; each one changes what gets built.

### 1.1 The Fundamental Law arithmetic in §1C.1 is wrong by ~50×

The prompt computes equity breadth as `500 stocks × 3 strategies × 50 bets/yr = 50,000` →
`IR ≈ 0.10 × √30,000 ≈ 17`. Nobody in the history of markets has run an IR of 17. The error is
that **BR counts *independent* bets**, and correlated names/strategies do not add independently.
For an equicorrelated cross-section:

```
N_eff = N / (1 + (N-1)·ρ̄)        BR = N_eff × rebalances_per_year
```

Recomputed honestly (ρ̄ = correlation of *cross-sectionally demeaned* returns, which is the
right quantity for a dollar-neutral book):

| Universe | N | ρ̄ (residual) | N_eff | Rebal/yr | BR | IC | **Implied IR** |
|---|---|---|---|---|---|---|---|
| Prompt's Setup C (11 US ETFs) | 11 | 0.30 | 2.8 | 12 | 33 | 0.03 / 0.05 | **0.17 / 0.29** |
| Prompt's Setup C, pessimistic ρ̄ | 11 | 0.50 | 1.8 | 12 | 22 | 0.03 | **0.14** |
| Your falsified S&P universe | 190 | 0.35 | 2.8 | 12 | 34 | 0.03 | **0.17** → observed −0.34…+0.39 ✅ |
| Prompt's Setup B (5 commodities) | 5 | 0.25 | 2.5 | 12 | 30 | 0.05 | **0.27** |
| Commodities, full sector (24) | 24 | 0.20 | 4.3 | 12 | 51 | 0.05 | **0.36** |
| **Cross-asset futures (40, 4 sectors)** | 40 | 0.10 | 8.2 | 12 | 98 | 0.05 | **0.49** |

Two things fall out immediately:

* The model **retrodicts your own results**. It says the S&P large-cap book should have delivered
  IR ≈ 0.17 gross → ≈ 0 to negative net. You measured −0.34 to +0.39 across six constructs. The
  framework is right; the prompt's application of it was not.
* The **only** row that reaches the ≥0.40 mandate lat is the one the prompt does not propose:
  **stop siloing by asset class, and take breadth across futures sectors** (commodities + rates +
  FX + equity index). That is also exactly what the source paper does — Moskowitz, Ooi & Pedersen
  (2012) is 58 instruments across four sectors, not 5 commodities.

To get IR 0.50 out of the prompt's 11-ETF universe you would need **IC ≈ 0.127** — 2.6× the crypto
IC that already failed, and above any post-decay equity XS IC in the literature.

### 1.2 Kill-Gate #1 is internally infeasible

The gate demands, simultaneously, `Sharpe ≥ 0.60`, `MaxDD ≤ 18%`, `Calmar ≥ 3.3`.
Calmar = CAGR / MaxDD, so `Calmar ≥ 3.3` with `MaxDD ≤ 18%` implies **CAGR ≥ 59%**. At Sharpe 0.60
that requires annualised vol ≈ 98% — and a 98%-vol book does not have an 18% drawdown. Conversely,
MaxDD ≤ 18% implies vol ≈ 8–12% for this Sharpe class, hence CAGR ≈ 5–7%, hence **Calmar ≈ 0.3–0.4**.
The Calmar gate as written is off by an order of magnitude and would kill every strategy including
correct ones. Corrected gates in §3B.

Two more gate defects: `Max consecutive losers ≤ 8` is routinely violated by working trend systems
(10–15 losing days in a row is normal); `Latency P99 < 500ms` is meaningless for a daily-settlement
futures book.

### 1.3 "Overnight Gap Fade" is a forbidden sign-flip

Setup C's second signal is the negative of `eq_overnight_1m`, archived as **F15** with
`α = −3.0%/yr, t = −3.01` — significantly negative, i.e. the *persistence* direction lost money.
The register's reopening condition on F15 says, verbatim: *"NOOIT als post-hoc sign-flip (§1.3)"*.
A significantly-negative α is not a discovered short signal; it is a construct that pays the spread
twice. This one is closed by your own governance, not by my judgement.

### 1.4 The Sharpe lat is 0.40, not 0.60

`eq_strev_1m` was archived at **+0.39** with no rounding, establishing the program's per-unit lat at
0.40 net. Introducing a *new* 0.60 lat for Setup B would be inconsistent — and would reject units
that are portfolio-additive. A unit at 0.45 with ρ ≈ 0.05 to the crypto book raises book Sharpe more
than a unit at 0.70 with ρ = 0.6. Keep 0.40 + `|ρ| < 0.30` + G4 α, as the mandate already specifies.

---

## 2. DEEL 1A — SETUP B: COMMODITIES / FUTURES

### 1A.1 Signal matrix (literature-anchored, honestly discounted)

Expected Sharpe columns are **net, retail-cost, post-publication-decay** estimates — not the
in-sample numbers from the papers. The discount applied is ~40–50% on published gross Sharpe
(McLean–Pontiff decay + the retail cost model in `xs_unit.CostModel`).

| # | Signal | Prior (peer-reviewed) | Published gross | **Honest net est.** | Core features | Data need | Verdict |
|---|---|---|---|---|---|---|---|
| B1 | **TSMOM (time-series momentum)** | Moskowitz, Ooi & Pedersen (2012) JFE 104(2) 228-250; Uhl (2025) *Rev. Financial Economics* on speculator-crowding decay | 0.8–1.2 (58 instr., 1985–2009) | **0.30–0.50** | own 12m trailing return, 60d vol scaling | front-month continuous only | 🟢 **BUILD FIRST** — `alpha/fx_tsmom.py` is already this, with different instruments |
| B2 | **Carry / roll-yield (basis)** | Koijen, Moskowitz, Pedersen & Vrugt (2018) JFE 127(2) 197-225 | 0.5–0.8 (commodity leg) | **0.25–0.45** | (F₁−F₂)/F₁ × 12/Δmonths | **needs 2 contract months** | 🟡 **ENERGY ONLY** free (§1A.2) |
| B3 | **Basis-momentum** | Boons & Prado (2019) JF; Fan et al. (2025) *Eur. Fin. Mgmt* 12555 (extended 1–12m formation, α unexplained by carry+mom) | 0.6–0.9 | **0.25–0.45** | momentum of F₁ *minus* momentum of F₂ | needs 2 contract months | 🟡 same data gate as B2; strongest *new* literature |
| B4 | **XS momentum (commodities)** | Erb & Harvey (2006); Qian (2025) *J. Futures Markets* 70022 (factor momentum; **transaction costs erode the gains**, high turnover) | 0.4–0.7 | **0.15–0.35** | 12-1 rank across commodities | front-month only | 🟠 low priority — high turnover, and Qian (2025) is explicit that costs eat it |
| B5 | **Seasonality** | agricultural/weather cycles | 0.2–0.4 claimed | **0.0–0.15** | day-of-year, calendar spreads | front-month | 🔴 **DO NOT BUILD.** You just falsified clock-seasonality in crypto (**F19**) at 7.4bp vs a 20bp lat. Same failure mode: max-over-many-windows in-sample. If built at all, it must be pre-registered with the multiple-comparison correction *before* the scan. |
| B6 | **Vol-spike / regime** | — | — | **conditioner only** | realized vol, VIX-TS | free (cboe verified) | 🟡 **F5 is binding**: alt-data may act as conditioner/de-grosser, never as a timer |

**Empirical reality anchor (use this, not the paper Sharpes):** the SG Trend / CTA index family has
realised roughly 0.25–0.40 Sharpe since 2010 *net of manager fees* (≈0.5–0.6 gross). A well-built
retail futures trend book landing at **0.35–0.55 net** is a success, not a disappointment. Any
backtest of yours that prints 2.0+ on this premium class is a bug — that is the single most
useful prior to carry into Week 5.

**Composite recommendation (replaces the prompt's `0.5·trend + 0.3·carry + 0.2·seasonal`):**

```
Sleeve D1  futures_tsmom     — 12m own-return sign, inverse-vol sized, monthly, 4 sectors
Sleeve D2  energy_carry      — (F1-F2)/F1 annualised, XS-ranked within the 4 energy products
Sleeve D3  energy_basis_mom  — mom(F1) - mom(F2), same panel as D2   [only if D2 clears]
Book       risk-parity across D1..D3 (F11: RP beat every tuned combiner — do not tune weights)
```

Seasonality is dropped (F19-consistent). Weights are risk-parity, not the prompt's `[0.5, 0.3, 0.2]`
— F11 in your register says tuned combination lost to plain RP (0.71/0.69 vs 0.94).

### 1A.2 Data infrastructure — **this is the real Setup B decision**

The prompt's data table is out of date. Verified status as of 2026-08-10:

| Source | Term structure? | History | Cost | Status | Verdict |
|---|---|---|---|---|---|
| **EIA NYMEX Contract 1–4** (WTI, Henry Hub NG, heating oil, RBOB) | ✅ **YES, 4 tenors** | NG C1–C3 from 1994, C4 from 1993; WTI from 1983 | **free**, no key (dnav pages) or free API key (opendata v2) | verified live 2026-08-10 | 🟢 **THE UNLOCK.** Real, PIT, decades-deep, free basis data. Energy only. |
| Nasdaq Data Link `CHRIS` (Wiki continuous) | ✅ (was) | 1990–2018 | was free | **deprecated, access restricted by provider** | 🔴 dead — the prompt's "Quandl/Nasdaq DataLink: Freemium/Excellent" line is stale |
| Stooq futures continuations | ❌ front only | decades | free | **download endpoint blocked** (your `DATA_REGISTER.md`) | 🔴 already blocked; this is what stalled W27 |
| yfinance continuous (`GC=F`, `CL=F`, `NG=F`) | ❌ front only | ~2000→ | free (ToS: personal use) | fallback already registered | 🟡 fine for **TSMOM**; useless for carry |
| yfinance individual months (`GCZ26.CMX`) | ✅ while listed | expired contracts dropped | free | — | 🟡 forward-fillable only; no back-history |
| CME settlements (daily bulletin) | ✅ all tenors | **today only** | free | — | 🟡 archive-forward: start capturing now, usable in ~2 yrs |
| IBKR TWS API | ✅ | daily deep, intraday ~1–2 yr; expired contracts limited | free w/ account + CME market-data subscription (~€10–13/mo) | — | 🟢 **execution + live data**; weak for deep history |
| Databento | ✅ full | CME back-history | pay-as-you-go (~$100s for daily OHLCV history) | — | 🟡 the escape hatch if carry must extend beyond energy |
| CFTC COT | n/a (positioning) | 1986→ | free | — | 🟢 free, PIT, and directly feeds the Uhl (2025) crowding overlay |

**Consequence — the plan splits in two by data availability:**

* **TSMOM (D1)** can be built *today* on free front-month continuous data (yfinance + your existing
  `yfinance_backup.py`), across 4 sectors, no new vendor. **No blocker.**
* **Carry / basis-momentum (D2/D3)** are free and deep **only for the 4 energy products**. That is
  a 4-name cross-section — N_eff ≈ 2 — so it must be judged as a small satellite sleeve
  (expected Sharpe 0.25–0.45, high variance), or it needs Databento money to widen to metals/ags.

**Roll mechanics.** The prompt recommends OI-based rolls + ratio-adjusted stitching; that is
correct, but note that **you do not need to build it for D1**: EIA/Yahoo continuous series are
already rolled. Build your own roll only if you go to Databento. When you do: roll on OI crossover
with a 3-day buffer, **back-adjust by ratio** (never by difference — difference-adjusted series go
negative and break log returns and vol scaling).

**Bar frequency.** Daily. Agreed with the prompt, and additionally: CME daily settlement is the
only price with a defensible PIT timestamp for a free feed, and F2/F19 already establish that this
program cannot pay taker costs at intraday frequency.

### 1A.3 Module audit — Setup B

| Module | Reuse | Required change | Risk / blocker |
|---|---|---|---|
| `data/sources/` | **ADAPT** | new `eia.py` (Contract 1–4) + extend `yfinance_backup.py` to futures tickers; DATA_REGISTER rows + lookahead test per source | EIA revision policy must be checked → treat as T+1 as-of like FRED |
| `alpha/xs_unit.py` | **DIRECT** | `CostModel(commission_bps≈0.5, half_spread_bps≈1.5, borrow_fee_ann=0.0)` — futures are *cheaper* than your equity model | none; contract multipliers live in sizing, not in the harness |
| `alpha/fx_tsmom.py` | **DIRECT → copy** | swap instrument list; this file *is* the D1 unit | 90% of Setup B's "new code" already exists |
| `alpha/fx_carry.py` | **ADAPT** | signal becomes `(F1−F2)/F1` instead of `i_fx − i_usd`; same XS harness, same rebalance | none — structurally identical |
| `risk/factor_alpha.py` (G4) | **ADAPT** | needs a futures factor set; Ken French has no commodity factors → use TSMOM-factor + equity-mkt + a broad commodity index as controls | G4 self-factor tension applies (see `project_g4_selffactor_tension` memory): trend-on-trend is degenerate → report full-set G4 **and** ex-self-factor |
| `registry/hypothesis_ledger.py` | **DIRECT** | book every scan, incl. seasonality windows if attempted | — |
| `portfolio/` (HRP/RP) | **DIRECT** | — | F11: no combiner tuning |
| `risk/daily_loss_governor.py`, `drawdown.py` | **DIRECT** | parameterise DD caps | — |
| `oms/`, `execution/` | **BUILD NEW** | IB adapter (contract objects, roll re-submission, RTH gates, SPAN margin) | **the only genuinely new engineering**; ~2–3 weeks |
| `live/` | **ADAPT** | market-hours + roll-window gates; daily settlement cadence | CME maintenance break, holiday calendar |

**Net:** the research path is ~1 week of work (data + two units), not 3 weeks. The *engineering*
path (IB OMS) is the long pole.

### 1A.4 Broker

Agreed: **Interactive Brokers**, and the prompt's reasoning holds for futures. Additions it misses:

* **Micro contracts are mandatory** at your account size — MGC (10 oz), MCL (100 bbl), MNG,
  M2K/MES/MNQ, micro FX. Commission ≈ $0.25–0.50 + exchange/clearing ≈ $0.35–0.60 per side.
* **Market data is not free**: CME real-time non-professional bundle ≈ €10–13/month. The prompt's
  "Total Cost: $0" is wrong; budget ~€150/yr.
* NinjaTrader/Saxo add nothing here — Saxo's futures spreads and financing are materially worse for
  a systematic book, NinjaTrader routes to a clearing FCM anyway.

### 1A.5 ⚠️ The capital-granularity blocker (new finding, not in the prompt)

A diversified futures book must express ~N_eff independent risk units. Per-instrument target risk
for capital `K` at portfolio vol `σ_p` is roughly `K · σ_p / √N_eff`. One micro contract carries
risk `notional × σ_instrument`:

| Instrument | Micro notional (2026 est.) | Ann. vol | Risk per contract |
|---|---|---|---|
| MGC (10 oz gold) | ~$33k | 15% | ~$5.0k |
| MCL (100 bbl WTI) | ~$7k | 35% | ~$2.5k |
| MES (S&P) | ~$28k | 16% | ~$4.5k |
| M2K (Russell) | ~$12k | 22% | ~$2.6k |

At `K = $50k`, `σ_p = 10%`, `N_eff = 8`: per-instrument budget = `50k × 0.10 / 2.83 ≈ $1.8k` —
**below one micro gold contract**. You would hold 0.35 contracts, i.e. round to 0 or to 1 (3× over-risk).

**Minimum viable capital for a 20-instrument, 4-sector program ≈ $150k–250k.**
Options if capital is below that:
1. **Trade fewer, cheaper-granularity instruments** (MCL, M2K, micro FX) — but N_eff drops and the whole thesis with it.
2. **CFDs** (EEA retail: ESMA leverage caps 1:20 gold, 1:10 other commodities) — fractional sizing solves granularity, at the cost of wider spreads + overnight financing. Must be cost-modelled honestly before it counts as a route.
3. **Rotate**: hold the top-|signal| K instruments each month. This is a *different, unvalidated* strategy — it must be backtested as such, not assumed equivalent.

This belongs in the pre-registration, because it determines whether the backtest you run is the
strategy you can actually trade (G10).

---

## 3. DEEL 1B — SETUP C: EQUITIES

### 1B.1 Signal matrix vs. the falsification register

| Prompt's signal | Prompt's est. Sharpe | **This repo's measured result** | Status |
|---|---|---|---|
| Cross-sectional momentum (12m − 20d) | 0.40–0.60 | `eq_xsmom_12_1` net **−0.34**, gross −0.27, 9/27 yrs positive; G4 MOM-loading +0.42 (t=53), R²=0.77 → *build was faithful, premium absent net* | **F13 — closed** |
| Overnight gap fade | 0.30–0.50 | `eq_overnight_1m` net **−0.60**; α −3.0%/yr **t = −3.01** | **F15 — closed; sign-flip forbidden** |
| Factor-quality momentum | 0.35–0.55 | `eq_quality_gpa` net **−0.25**; RMW loading +0.114 (t=11.8) → faithful; α t=−1.65 | **F17 — closed** |
| (implied) short-term reversal | — | `eq_strev_1m` **+0.39** (< 0.40 lat), α +3.1%/yr t=2.47 **PASS**, but decay tail 2024/25/26 = −3.6/−6.6/−7.2% | archived, decaying |
| (implied) residual reversal | — | `eq_strev_resid_1m` net −0.35, **gross −0.15** | **F16 — closed** |
| (implied) earnings surprise | — | `eq_pead_ar3` net −0.18, α t=−0.6 | **F18 — closed** |
| Volatility term structure | 0.25–0.45 | not tested; **F5 binding** — alt-data as conditioner/de-grosser only, never as timer | needs a different construct |

Six constructs, two independent information sources (price + PIT fundamentals), 2000/2010–2026,
~190 PIT-tradeable names/day → **zero net premia**. The prompt proposes re-running two of those six
on a universe with *less* breadth.

### 1B.2 Universe — and the hard execution blocker

The prompt's Option B universe is `SPY, QQQ, IWM, XLK, XLV, XLY, XLE, XLU, XLRE, XLF, XLI`.

**Every one of these is a US-domiciled ETF, and EEA retail clients cannot buy them.** Under PRIIPs,
any packaged retail product sold to an EU retail investor needs a Key Information Document; US ETF
issuers do not produce one, so EU brokers — Interactive Brokers, DEGIRO, Trading 212 — block the
order at compliance level. This is law, not broker policy.

Available routes, all of which change the strategy:

| Route | Viable? | Cost of the workaround |
|---|---|---|
| UCITS equivalents (IE/LU-domiciled) | partially | no clean 1:1 for all 11; thinner books; **shorting UCITS sector ETFs is hard-to-borrow**, and the strategy is long/short |
| Elective Professional status at IBKR (MiFID: 2 of 3 criteria) | yes, if you qualify | loses retail protections (negative-balance protection, leverage caps) — a real, non-trivial decision |
| Single stocks (PRIIPs does not apply to shares) | yes | but that *is* the ~190-name universe that produced F13–F18 |
| CFDs on indices/sectors | yes | ESMA leverage caps + financing; spread cost typically kills a 0.2-Sharpe edge |

Combined with §1.1: even if you solved PRIIPs tomorrow, the 11-ETF universe implies IR ≈ 0.17–0.29
at realistic IC. Solving a hard legal problem to reach an expected Sharpe below your own lat is not
a good trade.

### 1B.3 Module audit — Setup C

Moot at the universe level, but recorded for the redirect (§1B.5): `data/equity_universe.py`,
`data/edgar_universe.py`, `alpha/xs_unit.py`, `alpha/eq_*.py`, `risk/factor_alpha.py` and the PIT
lookahead suite are **all already built and tested** for equities. There is *no* engineering
blocker on equities — only a data-breadth and an execution blocker. If a small/mid-cap PIT source
ever appears, Setup C is roughly a two-day rerun, not a two-month build.

### 1B.4 Broker

If Setup C is ever revived: IBKR, elective-professional, with borrow availability checked
pre-trade via the TWS shortable-inventory feed. Alpaca does not serve EEA retail; Schwab is not
an option from NL.

### 1B.5 The only legitimate reopening path

Your own register already wrote it (F13/F16/F17 reopening conditions): **small/mid-cap universe with
PIT, delisted-inclusive data**. That is where the momentum/reversal/quality premia are documented to
concentrate, and it is the one thing the S&P-190 tests could not falsify. The blocker is purely
data: you need a delisting-inclusive small/mid-cap panel, and neither Stooq (blocked) nor yfinance
(delisted names largely absent) provides it. **Do not restart equities until that source exists.**
Candidates to price out: Sharadar SEP/SFP via Nasdaq Data Link (~$50–100/mo, delisted-inclusive),
or CRSP via an academic affiliation.

---

## 4. DEEL 2 — ARCHITECTURE

### 2A. Consolidated module matrix

| Module | Setup A (crypto) | Setup B/D (futures) | Setup C (equities) | Consensus |
|---|---|---|---|---|
| `data/sources/` | Bybit | **new: eia.py, yfinance-futures** | done (edgar/wiki/kenfrench) | plugin per source, `asof_join` only |
| `alpha/xs_unit.py` | direct | **direct** | direct | already the shared harness — *no refactor needed* |
| `alpha/*_carry, *_tsmom` | direct | **copy fx_→cm_** | done | thin signal defs on the harness |
| `labeling/`, `train/`, `cv/` | direct | direct (unused for units) | direct | CPCV shared |
| `portfolio/`, `risk/` | direct | direct | direct | RP/HRP + governors are generic |
| `risk/factor_alpha.py` | direct | **adapt (futures factor set)** | direct | per-market factor sets, ex-self-factor variant |
| `oms/`, `execution/`, `live/` | Bybit | **new IB adapter** | new IB adapter | venue connectors behind one OMS interface |

### 2B. Do **not** do the proposed `core/ + adapters/ + strategies/` refactor

The prompt's §2B.1 proposes restructuring the whole repo into `core/`, `adapters/`, `strategies/`.
That refactor is largely **already done under different names**, and doing it again would be a
multi-week, high-risk churn of a suite that is currently green:

* `alpha/xs_unit.py` **is** the core harness — its docstring already states the contract
  ("Every equity/FX/commodity XS unit … is a thin signal definition on top of this harness").
* `alpha/eq_*.py`, `fx_carry.py`, `fx_tsmom.py` **are** the per-asset-class strategy plugins.
* `data/sources/*` **is** the adapter layer, with a registered `base.py` contract and per-source
  lookahead tests.
* `risk/factor_alpha.py` already takes per-market factor sets.

**Minimal delta to support futures (est. 5–7 working days, not 2 weeks):**

```
src/tradebot/data/sources/eia.py          NEW   Contract 1-4 fetch + as-of + Pandera schema
src/tradebot/data/futures_universe.py     NEW   instrument spec: multiplier, tick, sector, roll rule
src/tradebot/alpha/cm_tsmom.py            NEW   = fx_tsmom with a futures instrument list
src/tradebot/alpha/cm_carry.py            NEW   = fx_carry with (F1-F2)/F1 as the signal
src/tradebot/risk/factor_alpha.py         EDIT  add FUTURES factor set + ex-self-factor mode
tests/lookahead/test_futures_causality.py NEW   as-of + roll-window causality guards
tests/unit/test_cm_units.py               NEW   determinism + known-outcome tests
docs/DATA_REGISTER.md                     EDIT  eia / yfinance-futures / IBKR rows
```

`execution/ib_adapter.py` + `oms/` wiring is a separate, larger workstream — start it only after
KG-B1/B2 pass. Building an OMS for a strategy that has not cleared its gates is how the last three
months of equities work would have been wasted if the gates had not held.

### 2C. Data schema

The prompt's unified Parquet schema is sound and matches what exists. Two corrections:

* `volume: int64` fails for FX and for any synthesised continuous series — use `float64` nullable.
* Add a mandatory **`asof_ts`** column to every PIT source. Your `DATA_REGISTER.md` rule 1 already
  requires `asof_join` on `asof_ts`; the prompt's schema omits it, which would silently reintroduce
  the lookahead the whole R-1 discipline exists to prevent.

Futures-specific columns: `contract_month`, `tenor_rank` (1..4), `open_interest`, `roll_flag`,
`adj_factor` (cumulative ratio-adjustment, so raw and adjusted are both recoverable).

---

## 5. DEEL 3 — TIMELINE & KILL-GATES

### 3A. Revised schedule (8 weeks to a decision, not 16)

The prompt's 16-week plan spends weeks 3–4 on a refactor that is unnecessary and weeks 8–10 on a
setup that is already falsified. Cutting both:

| Week | Work | Exit criterion |
|---|---|---|
| **1** | Pre-registration lock: instrument list, cost model, gates, ledger entries booked **before** any backtest. `eia.py` + futures universe spec + lookahead tests. | ledger n updated; suite green |
| **2** | `cm_tsmom` on 4-sector free continuous data. Full-sample CV + per-year table. | **KG-B1** |
| **3** | `cm_carry` (+`cm_basis_mom`) on EIA energy 4-tenor panel. G4 with ex-self-factor. | **KG-B2** |
| **4** | Walk-forward / OOS, bootstrap, PBO. Correlation vs. crypto book and vs. accepted FX carry. | **KG-B3** |
| **5** | Granularity + cost realism at actual account size (§1A.5). Re-run KG-B1 on the *tradeable* instrument subset. | **KG-B4** |
| **6–7** | IB adapter + paper trading, daily settlement cadence. | **KG-B5** |
| **8** | Portfolio integration, stress, GO/NO-GO. | **KG-P** |

Weeks 1–5 are ~all local compute on free data — no vendor spend, no live risk. **If Setup B dies,
it dies in week 2 or 3 for ~€0.** That is the point of front-loading the gates.

### 3B. Corrected kill-gates

Replacing the prompt's set. Every threshold is either from the existing mandate or derived to be
internally consistent (§1.2).

| Gate | When | Criterion | Rationale |
|---|---|---|---|
| **KG-B1** in-sample | wk 2 | net Sharpe **≥ 0.40** after costs; ≥ 60% of calendar years positive; MaxDD ≤ 2.5× ann. vol; **Calmar ≥ 0.30** | mandate lat (eq_strev precedent); Calmar corrected from 3.3 |
| **KG-B2** faithfulness + α | wk 3 | G4 residual α with HAC: **either** p<0.05 **or** a documented faithful factor loading (the F13/F17 standard); ex-self-factor variant reported | prevents both fake alpha and mislabelled beta |
| **KG-B3** OOS | wk 4 | WF Sharpe ≥ 0.30; decay ≤ **40%** vs IS; bootstrap P(S>0) ≥ 0.75; ρ vs crypto book **< 0.30** | 25% decay is too tight for trend (high year-to-year variance); ρ gate is what makes the unit worth owning |
| **KG-B4** tradeability | wk 5 | realised sizing error ≤ 20% of target risk at actual capital; instrument subset that survives re-clears KG-B1 | §1A.5 — the new blocker |
| **KG-B5** paper | wk 6–7 | slippage ≤ 25% of modelled; fill rate ≥ 95% at settlement; no roll failures over ≥1 roll cycle | latency gate dropped (daily book) |
| **KG-P** portfolio | wk 8 | book Sharpe with B ≥ book Sharpe without B **+ 0.10**; combined MaxDD ≤ 10%; ρ(A,B) < 0.30 | the only question that matters at book level |

**Failure protocol (unchanged from mandate §2.3):** a failed gate → falsification-register entry
with evidence link and a reopening condition → redirect to the next item on the expansion list. No
re-tuning of a failed unit, no gate relaxation, no rounding.

### 3C. Automation

Delivered as `tests/killgates/test_expansion_killgates.py` (pytest, marker `killgate`), including a
**gate-consistency test** that fails if a proposed gate set is internally infeasible — the check
that would have caught the Sharpe/MaxDD/Calmar contradiction in §1.2 before anyone spent a week on it.

Run:

```bash
python -m pytest tests/killgates -m killgate -v
```

---

## 6. DEEL 4 — OPERATIONAL ANSWERS

**4.1 Asset selection.**
*Setup B (rescoped):* D1 TSMOM across 4 sectors — energy (CL, NG, HO, RB), metals (GC, SI, HG),
ags (ZC, ZW, ZS), plus **rates (ZN, ZF), equity index (ES, NQ, RTY), FX (6E, 6J)** to get N_eff up.
D2/D3 carry+basis-momentum on the 4 EIA energy products only (free term structure).
*Setup C:* none. Universe closed pending small/mid-cap PIT data.

**4.2 Data sourcing.** Free tier suffices for the whole research phase: yfinance continuous
(TSMOM) + EIA Contract 1–4 (carry) + CFTC COT (crowding overlay) + CBOE VIX-TS (conditioner,
already verified). Paid spend deferred to a single decision point: **only if D2 clears on energy
do you buy Databento history to widen carry to metals/ags.** IBKR market data ≈ €150/yr at go-live.

**4.3 Model formulation.**

```python
# D1 — cm_tsmom  (Moskowitz, Ooi & Pedersen 2012) — literature-conform, FIXED
signal_i(t) = sign( TR_i(t-1) / TR_i(t-1-252) - 1 )
weight_i(t) = signal_i(t) / vol_i(t, 60d)          # inverse-vol, gross-normalised to 1
# monthly rebalance, held to next month-end; costs: 0.5bp comm + 1.5bp half-spread

# D2 — cm_carry  (Koijen, Moskowitz, Pedersen & Vrugt 2018)
carry_i(t)  = (F1_i(t) - F2_i(t)) / F1_i(t) * (12 / months_between(F1, F2))
weights     = decile_weights(rank(carry), long_frac=1/3, short_frac=1/3)   # $-neutral
# monthly rebalance on the EIA 4-product energy panel

# D3 — cm_basis_mom  (Boons & Prado 2019; Fan et al. 2025)   [conditional on D2]
bm_i(t) = mom_12m(F1_i) - mom_12m(F2_i)
```

No tunable harvest layer, no weight optimisation (F12, F11). Book = risk parity over D1–D3.

**4.4 Broker unification.** Yes — IBKR, single API for futures now and equities later if the
universe ever reopens. Add: micros mandatory, CME data subscription budgeted, elective-professional
status is a separate decision tied only to a (currently dead) equity path.

**4.5 Risk & allocation.** The prompt's "Setup A: max 10% daily loss" contradicts your propfirm
plan (`project_propfirm_plan`: 5% daily / 10% static DD, with the governor retrofit to 6%/3%).
Keep the propfirm numbers — they are contractual, not preferences. Allocation: do **not** pre-commit
50/25/25. Setup B gets capital only after KG-P, and then at a weight set by risk parity on measured
correlations, capped at 25% until it has 3 months of live P&L.

---

## 7. DEEL 5 — DELIVERABLES CHECKLIST

| # | Deliverable | Where | Status |
|---|---|---|---|
| 1 | Executive summary + GO/NO-GO + top risks | §0 | ✅ |
| 2 | Data-inventory report | §1A.2, §1B.2, + `DATA_REGISTER.md` rows | ✅ |
| 3 | Signal-matrix tables + literature | §1A.1, §1B.1 | ✅ |
| 4 | Module audit matrix | §1A.3, §1B.3, §2A | ✅ |
| 5 | Architecture specification | §2B (minimal-delta file list) | ✅ |
| 6 | Phased timeline | §3A (8 weeks) | ✅ |
| 7 | Kill-gate automation code | `tests/killgates/test_expansion_killgates.py` | ✅ |
| 8 | Operational spec | §4 | ✅ |
| 9 | Broker comparison + recommendation | §1A.4, §1B.4 | ✅ |

**Not delivered, deliberately:** no new falsification-register entries. The register is
append-only and evidence-bound; this document produced no *measurements*, only a literature and
data review. Setup C's NO-GO rests on the **existing** F13–F18, not on a new F20.

---

## 8. WHAT WOULD CHANGE THESE VERDICTS

* **Setup B → full GO:** `cm_tsmom` clears KG-B1 on the free 4-sector panel, and account capital
  reaches ~$150k (or a cost-honest CFD route is validated).
* **Setup B → NO-GO:** TSMOM lands below 0.40 net on the tradeable subset, or granularity forces
  N_eff below ~4 (at which point the expected IR is ~0.25 and it cannot clear the lat).
* **Setup C → reopen:** a delisting-inclusive small/mid-cap PIT source is registered and verified.
  The code to test it already exists; the rerun is days, not months.

**Sources:**
[Uhl (2025), Speculators and TSMOM in commodity futures, *Rev. Financial Economics*](https://onlinelibrary.wiley.com/doi/full/10.1002/rfe.1228) ·
[Qian (2025), Factor Momentum in Commodity Futures Markets, *J. Futures Markets*](https://onlinelibrary.wiley.com/doi/10.1002/fut.70022?af=R) ·
[Fan et al. (2025), Understanding the Performance of Currency Basis-Momentum, *Eur. Financial Mgmt*](https://onlinelibrary.wiley.com/doi/full/10.1111/eufm.12555) ·
[Koijen, Moskowitz, Pedersen & Vrugt, *Carry* (JFE 2018)](https://pages.stern.nyu.edu/~lpederse/papers/Carry.pdf) ·
[EIA NYMEX Natural Gas Futures Contracts 1–4 (daily, from 1993/94)](https://www.eia.gov/dnav/ng/ng_pri_fut_s1_d.htm) ·
[EIA Open Data API](https://www.eia.gov/opendata/index.php) ·
[Nasdaq Data Link CHRIS deprecation](https://github.com/PacktPublishing/Python-for-Algorithmic-Trading-Cookbook/issues/5) ·
[Databento futures data](https://databento.com/futures) ·
[PRIIPs KID blocks US ETFs for EU retail](https://finorum.com/us-etfs-in-europe/) ·
[UCITS vs US ETFs for EU investors](https://www.eupersonalfinance.eu/articles/ucits-etf)
