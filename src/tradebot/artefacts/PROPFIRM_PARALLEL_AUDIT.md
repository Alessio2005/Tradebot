# Propfirm Parallel-Deployment Audit & Optimalisatieplan

**Auteur:** CHIEF (Quantitative Architect / Head of ML)
**Datum:** 2026-06-14
**Mandaat:** ≥3 ML-strategieën parallel, 1 per propfirm-account, lage drawdown / hoge return.
**Accountprofiel (vastgesteld):** 3 accounts à ~$100k+, **statische** max-DD ~10% + daily-loss ~5%.

---

## 0. CHIEF-eindoordeel

De *structuur* (1 ongecorreleerde strategie per account) is correct en sluit aan op de
eigen ledger-doorbraak (ongecorreleerde markt-neutrale sleeves = de echte edge,
`project_edge_investigation_20260531`). Drie harde waarheden bepalen de uitvoering:

1. **Slechts één strategie is OOS-gevalideerd.** Het crypto-MN-boek (Sharpe ~1.15) staat
   in de ledger. FX-carry is een gedocumenteerde premie maar ledger-onbevestigd (W25).
   Equities-sleeves zijn *gefalsifieerd* (W21-24). → 2 van de 3 strategieën gaan
   **shadow-first**, krijgen geen realgeld vóór een eigen OOS-clean window.
2. **Venue/instrument-mismatch.** Alle edge leeft op Binance USDⓈ-M perps mét funding.
   Funding-carry transfereert **alleen** naar een crypto-perp-propfirm met API. Op een
   FX/CFD-firma bestaat funding niet en klopt het 6 bps-kostenmodel niet (overnight-swap).
3. **Drawdown-geometrie.** `DrawdownBreaker` staat op trigger 15% / resume 8% — boven de
   10%-vloer. Er is **geen intraday daily-loss-governor**. Dat is de #1 ontbrekende
   component en de kern van de retrofit (§4).

**Kern-inzicht:** de winst zit niet in nieuwe alpha (die is er), maar in (a) de juiste
strategie→venue-mapping en (b) een propfirm-risk-governor-laag.

---

## 1. Challenge-snelheid vs. slaagkans (de "10%/jr is te traag"-kwestie)

Bij Sharpe ~1.1 schaalt méér hefboom μ én σ samen → de first-passage-ratio 2μ/σ² zakt met
1/k → je raakt de −10%-faallijn sneller. Barrières +8% / −10%:

| Vol-target | μ/σ (ann) | P(+8% vóór −10%) | Verw. tijd tot pass |
|---|---|---|---|
| 1× (preservation) | 10% / 9% | ~92% | ~10–12 mnd |
| **2× (CHALLENGE)** | **20% / 18%** | **~80%** | **~4–5 mnd** |
| 3× | 30% / 27% | ~73% | ~3 mnd |

**Beleid:** challenge-regime = **~2× vol** + propfirm **zonder tijdslimiet** → ~4–5 mnd, ~80% pass.
Zodra funded → terug naar **1× preservation** (kapitaalbehoud + profit-lock). NB: het echte
risico is niet de terminale DD maar de intraday 5%-daglimiet op een staartdag → daily-governor
(§4.1) is niet optioneel.

---

## 2. Architectuur — 3 accounts, cross-asset orthogonaal

"1 strategie per account" beschermt alleen als de daily-loss-events ongecorreleerd zijn.
Drie crypto-accounts delen crashrisico (exchange-halt, USDT-depeg, funding-spike). Daarom
maximale orthogonaliteit over asset-classes:

| Account | Strategie | Venue | Beta-bron | Validatie |
|---|---|---|---|---|
| **A** | Crypto market-neutral (statarb + carry + lowvol) | Crypto-perp propfirm (API) | ~0, dollar-neutraal | ✅ OOS (Sharpe ~1.15) |
| **B** | Crypto crisis-alpha (trend + dvol) | Crypto-perp propfirm (API) | +0.12, profiteert in crashes | 🟡 in-sample (MaxDD −24% nátief) |
| **C** | G10 FX-carry + FX-TSMOM | FX-CFD propfirm (MT5/cTrader) | onafhankelijk van crypto | 🟡 gedocumenteerd, ledger-onbevestigd |

Waarom niet 3 sleeves van hetzelfde boek: losse sleeves hebben lagere Sharpe én hogere
idiosyncratische DD dan het gecombineerde boek. A houdt het *complete* gevalideerde boek;
B en C voegen écht ongecorreleerde exposure toe (A neutraal, B directioneel-crash, C andere
asset-class).

---

## 3. Per-strategie specificatie (assets + parameters)

### Account A — Crypto Market-Neutral boek
Bron: `alpha/neutral_book.py` (`NeutralBook`, dollar-neutraal, gross 1).

- **Universe (12–18 liquide perps):** BTC, ETH, SOL, BNB, XRP, ADA, AVAX, LINK, DOT, DOGE,
  LTC, MATIC, NEAR, ATOM, ARB, OP, INJ, SUI — gefilterd op (i) verhandelbaar op de firma,
  (ii) 30d-ADV top-20, (iii) funding-data beschikbaar. `min_names=8` harde ondergrens/bar.
- **Sleeves (pre-committed constants — NIET her-tunen):**
  - statarb (Avellaneda-Lee residual reversal): `statarb_k=3`, long laggards
  - lowvol (betting-against-vol): `vol_window=20`
  - carry (funding-carry, short high-funding): `carry_window=3`
  - combinatie: causal risk-parity, `rp_lookback=252`, `rp_min_periods=60`
- **Gross-knop:** nátief MaxDD ~10.4% (CHIEF-remediation) raakt de 10%-vloer.
  - FUNDED (1×): gross ~0.55–0.65× → ann-vol ~8–9%, MaxDD ~6–7%.
  - CHALLENGE (2×): gross ~1.1–1.3× → ann-vol ~16–18%, MaxDD-budget tegen 9% (daily-governor leunt zwaar).
- **Rebalance:** dagelijks bar-close, execute t+1 (no-lookahead ingebouwd).
- **Verwacht (funded, na kosten):** ~8–11%/jr, Sharpe ~1.1, MaxDD ~6–7%, beta ~0.

### Account B — Crypto Crisis-Alpha
Bron: `alpha/multi_sleeve_book.py` (`MultiSleeveBook`) — **alleen TREND + DVOL** (carry-sleeve
weggelaten om overlap met A te vermijden).

- **Trend:** `majors=(BTCUSDT, ETHUSDT)`, `trend_mas=(50,100,200)`, long/short multi-MA.
- **DVOL:** `dvol_symbol=BTCUSDT`, `dvol_z_window=90`, `dvol_cap=1.5` (contrarian, long/flat).
- **Risk-parity:** `rp_lookback=252`, `rp_min_periods=60`.
- **Gross-knop:** nátief MaxDD −24% = 2,4× de vloer → onacceptabel zonder ingreep.
  - FUNDED: gross ~0.30–0.35× → ann-vol ~6%, MaxDD-budget ~7%.
  - CHALLENGE: gross ~0.6–0.7× (2×) — alleen ná shadow-clean.
- **Rol:** convexe hedge — verdient juist op crashdagen die A/C pijn doen.
- **Status:** in-sample → **30 dagen shadow vóór challenge-geld**.

### Account C — G10 FX-Carry + FX-TSMOM
Bron: `alpha/fx_carry.py` + `alpha/fx_tsmom.py`.

- **Universe (9 G10-crosses):** EUR/USD, GBP/USD, USD/JPY, USD/CHF, AUD/USD, NZD/USD,
  USD/CAD, EUR/JPY, EUR/GBP.
- **fx_carry:** signal = i_fx − i_usd (3m rates, PIT/as-of), maandelijkse rebalance,
  long top 1/3 / short bottom 1/3, dollar-neutraal.
- **fx_tsmom:** signal = sign(12m eigen rendement), inverse-vol (`60d`), maandelijks,
  gross 1, NIET dollar-neutraal.
- **Combinatie:** 50/50 risk-parity, of IC-DAMP-combiner (`alpha/combination.py`).
- **Kosten-silent-killer:** code rekent 1 bp half-spread (FX-swap-implied). Propfirm-CFD =
  overnight-swap + bredere spread. Een carry-trade die je dagen vasthoudt kan zijn hele
  rate-differential aan negatieve swap verliezen. → **Swap-tarieven van de firma verifiëren
  en `CostModel(half_spread_bps,…)` herijken vóór deployment.** Dit kan carry-EV omdraaien.
- **Status:** ledger-onbevestigd → shadow-first.

---

## 4. Verplichte risk-governor retrofit (kern van het werk)

Draait over alle drie de accounts. Vier componenten:

**4.1 Daily-loss-governor (NIEUW — hoogste prioriteit).**
Nieuwe module `risk/daily_loss_governor.py`. Houdt `equity_t / day_start_balance − 1` bij
(dag-grens in firma-tijdzone):
- Soft-stop bij **−3.5%**: geen risk-toename, alleen risk-reducerende orders.
- Hard-flatten + halt tot dag-reset bij **−4.5%** (buffer onder 5%; firma meet live).
- Reset `day_start_balance` op de daily-reset van de firma. Wire in `live/engine.py` vóór
  `CircuitBreaker.check`.

**4.2 Statische max-DD-breaker herconfigureren.**
`risk/drawdown.py` `DrawdownConfig`: **`trigger_threshold=0.06`, `resume_threshold=0.03`,
`scale_factor=0.0`**, peak = startbalans (statisch, niet rolling — propfirm meet vs startbalans).
Halt op 6% laat 4% buffer voor slippage/gap.

**4.3 Vol-targeting gekoppeld aan de daglimiet.**
Wire `risk/var.py` → per-account dagelijks risk-budget: size zó dat **1-dags 99%-ES <
0.5 × daily-limit**. `risk/kelly.py` (¼-Kelly cap) schaalt de gross → levert de
gross-multipliers uit §3.

**4.4 Cross-account correlatie-monitor (NIEUW).**
`monitoring/cross_account.py`: als 20d-correlatie tussen twee account-P&L's > 0.4, throttle
de laagste-Sharpe met 50%. Voorkomt dat 3 "onafhankelijke" accounts 1 trade worden.

**4.5 Challenge/Funded-regime-flag.**
Per account een config-flag:
- CHALLENGE: vol-target 2×, daily-governor strikt, doel = +8% halen.
- FUNDED: 1×, profit-lock (na +6% soft-trailing-stop die payout veiligstelt), reset
  peak/day_start na elke payout.

---

## 5. Propfirm-keuze (verdict)

Eis: API-trading is non-negotiable (ML-bot), shorts toegestaan, funding-data aanwezig
(crypto), EA's toegestaan (FX), NL-ingezetenen geaccepteerd, **geen tijdslimiet**.

| Account | **Gekozen firma** | Reden | Alternatief |
|---|---|---|---|
| **A + B** | **Hyrotrader** (Bybit-API) | Echte perps mét funding via Bybit-API, algo/bot toegestaan, shorts, statische-DD-optie, NL-toegankelijk, geen tijdslimiet | Breakout (Hyperliquid, on-chain, funding aanwezig) |
| **C** | **FTMO** (MT5/cTrader) | EA's/algo's toegestaan, NL-geaccepteerd, tijdslimiet verwijderd, brede G10-FX-spreads, `MetaTrader5`-Python-bridge | FundingPips / FunderPro |

**Primaire keuze: Hyrotrader voor de twee crypto-accounts (A, B)** — dáár leeft je
gevalideerde alpha en de API + funding transfereren 1-op-1. **FTMO voor C** als
cross-asset-diversifier.

> ⚠️ **Verifieer huidige voorwaarden zelf** (kennis t/m jan 2026): exacte daily/max-DD,
> statisch vs trailing, API-/EA-policy, fundingsplit, en — voor FTMO — de **overnight-swap-tarieven**
> (bepalend voor C's carry-EV). Pas `CostModel` en breaker-config dáárop aan vóór deployment.

---

## 6. Position-sizing math

MaxDD ≈ (1.5–2.5)×σ_ann. Static-10%-vloer → target MaxDD ≤ ~7% (FUNDED):
- σ_target: **A→8–9%, B→6%, C→7–8%** (funded). Challenge = 2×.
- gross-multiplier = σ_target / σ_native. Crypto-MN σ_native ~14% → ~0.6×.
  Multi-sleeve σ_native ~18–20% → ~0.3×.
- Phase-1 pass-kans (A, funded-sizing): ~92%; challenge-sizing (2×): ~80% in ~4–5 mnd.

---

## 7. Gefaseerde uitrol met go/no-go-gates

| Fase | Duur | Wat | Go-gate |
|---|---|---|---|
| **0. Retrofit** | nu | §4.1–4.5 bouwen + tests (327-suite uitbreiden) | nieuwe modules groen + breaker-config geverifieerd |
| **1. Shadow (paper)** | **≥14 dagen** | alle 3 paper-trade tegen live feed, governors actief, gesimuleerde propfirm-regels | 0 simulated breaches; A-beta \|<0.1\|; B/C-corr met A < 0.3 |
| **2. Challenge A** | tot pass (~4–5 mnd) | alleen A (gevalideerd), challenge-regime 2× | +8% gehaald, 0 regel-breaches |
| **3. B + C challenge** | tot pass | B/C erbij ná eigen 30d shadow-clean | idem per account |
| **4. Funded + scaling** | doorlopend | funded-regime, payout-cadans, cross-account-monitor live | maandelijkse MRM-review |

**Harde regel (G4-discipline):** B en C krijgen geen realgeld vóór een eigen OOS-clean
shadow-window. Niet stilzwijgend versoepelen.

---

## 8. Eerlijke return-verwachting

100% CAGR is niet haalbaar (convergentie Sharpe ~1–1.5, `project_true_alpha_mandate`).
Onder propfirm-caps (gedownscaled, funded): ~7–11%/jr per account. De "hoge return" zit in
**payout-split (80–90%) × 3 ongecorreleerde accounts op kapitaal-dat-niet-van-jou-is**, niet
in het ruwe %. Orde van grootte: ~$20–30k/jr netto op $300k challenge-capital — mits alle
drie funded blijven. De edge hier is *consistentie binnen de regels*, precies waar het
markt-neutrale boek goed in is.

---

## 9. Open beslissing & aanbevolen volgorde

Enige open punt dat alleen jij kunt vastpinnen: de **definitieve firmavoorwaarden**
(Hyrotrader API/funding/DD-details + FTMO swap-tarieven) → herijken `CostModel` en breaker.

Aanbevolen volgorde: **(1) governor-retrofit bouwen → (2) shadow ≥14d → (3) firmavoorwaarden
verifiëren → (4) challenge A**.
