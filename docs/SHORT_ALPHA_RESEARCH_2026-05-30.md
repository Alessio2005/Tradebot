# Short Alpha Research — waar zit positieve edge in shorts?

> **GEARCHIVEERD — historisch document.** Dit is een verslag van het short-alpha-onderzoek van mei 2026 en beschrijft de toestand van toen. Het wordt NIET bijgewerkt: de paden en artefacten die het noemt, zijn die van die periode en bestaan grotendeels niet meer. Voor de huidige toestand, zie `docs/PROJECT_STATE.md`.
> *Als historisch gemarkeerd op 2026-09-01 (Phase 7/8, Stage E-3).*

**Datum:** 2026-05-30 · **Goal:** positieve edge vinden in de short-kant (systeem is nu de-facto long-only)
**Status:** onderzoeksnotitie / hypothese-backlog (geen code-wijziging)

---

## 0. TL;DR

De huidige short-kant heeft **geen** positieve edge, en dat is geen bug maar een
*structureel* gevolg van hoe shorts gegenereerd worden: ze zijn een spiegel van de
long-trend-pipeline (TrendScan + symmetrische Triple Barrier PT=2·ATR/SL=1·ATR),
toegepast op een markt met seculiere opwaartse drift. Empirisch (experimenten
2026-05-29) verlaagt méér short-activiteit de portfolio-Sharpe van **2.92 → 1.31**
en duwt MaxDD van **10.8% → 18.0%** (DD-breaker getript). De "oplossing" die nu in de
pijplijn zit (`long_bear_factor`, longs afschalen in bear-regime) is een *defensieve
substituut* voor shorts — geen short-alpha.

**Kernconclusie:** trend-following directional shorts hebben in crypto een negatieve
*onvoorwaardelijke* EV. Short-alpha moet komen uit bronnen die **drift-agnostisch of
structureel short-biased** zijn — niet uit "voorspel dat de prijs daalt" via dezelfde
long-mirror pipeline. Die bronnen zijn grotendeels **al in de codebase aanwezig** maar
worden niet als short-motor gebruikt.

---

## 1. Diagnose — waarom de huidige shorts niet werken

| # | Oorzaak | Bewijs in code/logs |
|---|---------|---------------------|
| D1 | **Seculiere long-drift.** ~70% van 8h-bars sluit hoger (2021-2026). De onvoorwaardelijke short vecht tegen de drift. | `backtest_portfolio.py:341` |
| D2 | **Barrier-geometrie staat verkeerd voor shorts.** PT=2·ATR / SL=1·ATR: een short moet een 2-ATR daling halen om te winnen maar verliest al bij 1-ATR stijging. In een opdrijvende markt is dat structureel verliesgevend. | `bidirectional.py:114-129`, TrendScan default PT=2.0/SL=1.0 |
| D3 | **Symmetrische CUSUM, asymmetrische markt.** CUSUM triggert op elke grote move, inclusief bull-pullbacks die snel reverten en de SL raken. 53.7% van AVAX-short-events valt in bull-regime en is netto negatief (-0.80 vs +0.80 in bear). | `backtest_portfolio.py:66-83` |
| D4 | **De Judge kan het niet redden.** Short-Judge OOS pos-rate ≈ 53-55% (ETH 54.1%, SOL 52.9%, DOT 54.9%) — nauwelijks boven munt-opgooi. Geen discriminatie. Raw CatBoost short-probs mean ≈ 0.07-0.34. | `logs/retrain_cpcv_*_SHORT.log` |
| D5 | **Calibrator-geschiedenis.** PathSpecificPlatt sign-inversie blies short-probs kunstmatig op boven min_conf → portfolio overspoeld met slechte shorts (de oorspronkelijke Sharpe=-0.02). De "fix" werd `tau_short = tau_long + 0.35` → effectief **shorts uitgezet** (5 shorts i.p.v. honderden). | `calibration.py:326-333`, `backtest_portfolio.py:338-346` |
| D6 | **Empirisch bewijs dat shorts schaden.** `bidir_A` (415 shorts op DOT, regime-gefilterd): portfolio Sharpe **1.31**, MaxDD **18.0%**. `bidir_C` (5 shorts + long-regime-scaler): Sharpe **2.92**, MaxDD **10.8%**. | `logs/bidir_A_shortmargin010.log`, `logs/bidir_C_longregime.log` |

> **De dood-lopende weg:** blijven sleutelen aan `tau_short` / `short_edge_margin` op de
> symmetrische trend-pipeline. Dat is in-sample optimaliseren op een signaal dat geen
> onderliggende edge heeft (D1-D4). Empirisch bevestigd door D6.

---

## 2. Reframe — waar short-alpha wél leeft

Short-edge in een opwaarts-driftende universe komt niet uit richting-voorspelling maar
uit drie families die de drift *niet* als tegenwind hebben:

1. **Structureel short-biased income** → funding-carry (de short *betaalt niet*, hij
   *ontvangt* funding wanneer longs crowded zijn).
2. **Drift-agnostische reversion/microstructuur** → je short een *lokale* overshoot of
   sell-flow, niet een trend; horizon kort genoeg dat drift niet domineert.
3. **Relatieve waarde / market-neutral** → je short de relatieve *verliezer*; EV hangt
   af van dispersie, niet van absolute marktrichting.

Cruciaal: **al deze motoren bestaan al** en geven *signed* `[-1,+1]` signalen (geverifieerd):
`alpha/carry.py`, `alpha/kalman_ou.py`, `alpha/mean_reversion.py`,
`alpha/csm_volume_clock.py`, `alpha/microstructure.py`. Ze worden alleen niet als
short-motor ingezet — de shorts lopen nu via de long-mirror trend-pipeline.

---

## 3. Hypothese-backlog (geprioriteerd op conviction × effort)

### H1 — Asymmetrische barrier-geometrie voor shorts *(laagste effort, structureel)*
**Stelling:** crypto gaat "up the stairs, down the elevator" — sell-offs zijn sneller en
scherper (vol-clustering, leverage-cascades). De short-barrier moet die fat left tail
matchen: **PT ≈ 1·ATR / SL ≈ 2·ATR** en een *kortere* horizon (t_max omlaag) voor shorts,
terwijl longs symmetrisch blijven.
**Test:** her-label alleen de short-kant met geflipte PT/SL + kortere t_max; meet OOS
short-only Trade-Sharpe en win-rate per regime. **Verwacht:** short base-rate stijgt,
shorts winnen sneller in de bear/sell-off, minder timeout-verlies.
**Raakt:** `TrendScanningLabeler` (side="SHORT"-tak), `bidirectional.py:126-129`.

### H2 — Funding-carry short-overlay *(hoogste conviction, structureel positieve EV)*
**Stelling:** wanneer funding extreem positief is (crowded longs, `feat_meso_funding_zscore_30d`
hoog), betaalt de short-kant *geen* funding maar *ontvangt* die — plus de over-leveraged
longs zijn kwetsbaar voor een long-squeeze. Dit is de schoonste "positieve edge in shorts"
omdat de edge de funding-betaling is, niet de prijsvoorspelling.
**Test:** gate shorts op `funding_zscore_30d > τ`; meet carry-income + squeeze-reversion EV.
Combineer met OU-overshoot (H4) voor timing.
**Raakt:** `alpha/carry.py` (al short-capable, regel 6-7), `features/funding_carry.py`
(features bestaan al), gating in de short-tak van de portfolio.

### H3 — Cross-sectionele (relative-value) shorts *(hoogste schaalbaarheid, market-neutral)*
**Stelling:** short de *relatieve* verliezer, long de relatieve winnaar → β-neutraal. De
EV van het short-been hangt af van dispersie tussen de 5 assets, niet van absolute richting,
dus de seculiere drift valt grotendeels weg.
**Test:** draai `CSMomentumVolumeClock` als dedicated short-leg-selector; evalueer als
dollar-neutraal long/short-paar binnen de HRP-laag.
**Raakt:** `alpha/csm_volume_clock.py` (rank-normalized [-1,+1], al bidirectioneel).

### H4 — Mean-reversion / OU als short-motor *(medium effort, al gebouwd)*
**Stelling:** short de overshoot *boven* de OU-mean, cover op de mean. Reversion is
symmetrisch en vecht niet tegen drift zoals trend-shorts — je short lokale overextensie,
geen trend. Veel betere fit voor short-EV dan trend-following.
**Test:** route shorts door `KalmanOUMeanReversion` / `mean_reversion.py` i.p.v. de
trend-pipeline; Judge-gate behouden.
**Raakt:** `alpha/kalman_ou.py`, `alpha/mean_reversion.py` (beide signed, al gebouwd).

### H5 — Microstructuur / OFI-gedreven shorts *(medium effort, event-driven)*
**Stelling:** aggressieve sell-flow / liquidatie-cascades geven echte short-signalen op
korte horizon waar drift niet domineert. Short *in* een long-liquidatie-spiraal.
**Test:** `OFISignal` als short-trigger op micro-horizon; let op spread-verbreding (D3) in
de kostenmodellering.
**Raakt:** `alpha/microstructure.py`, `features/microstructure.py`.

### H6 — Shorts als hedge i.p.v. standalone *(constructie-laag)*
**Stelling:** beoordeel shorts niet op absolute return maar op hun bijdrage aan
**DD-reductie en β-neutralisatie**. Een short die de portfolio-MaxDD van 18% → <10%
drukt verdient zijn plek, ook bij Sharpe ≈ 0 standalone. Dit is precies het gat dat
`long_bear_factor` nu noodgedwongen dicht.
**Test:** evalueer short-tracks op marginale Sharpe/MaxDD-bijdrage in de HRP/risk-laag,
β-geneutraliseerd t.o.v. ETH (de beta-hedge-infra bestaat al).
**Raakt:** `risk/beta_hedge.py`, `risk/portfolio.py`, HRP-laag in `backtest_portfolio.py`.

---

## 4. Aanbevolen volgorde

1. **Diagnostiek eerst** — kwantificeer short label-base-rate en per-regime short-EV
   onder gecorrigeerde barrier-geometrie (H1). Bevestig OOS: "shorts werken alleen in
   bear + met snelle-PT geometrie". Goedkoop, valideert of H1 een echt fundament heeft.
2. **H2 (funding-carry)** — hoogste conviction, kleinste afhankelijkheid van richting.
   Dit is de meest waarschijnlijke bron van *echte* positieve short-EV.
3. **H4 + H3** — herbestem OU en CSM als short-motoren (drift-agnostisch / market-neutral).
   Stop met shorts via de long-mirror trend-pipeline.
4. **H6** — herwaardeer de hele short-kant als hedge/relative-value been, niet als
   standalone richting-bet. Vervangt `long_bear_factor` door een echte short-hedge.

**Niet doen:** verder tunen van `tau_short`/`short_edge_margin` op de trend-pipeline
(D6 bewijst dat dat een dood spoor is).

---

## 4b. EMPIRISCH RESULTAAT — H2 funding-carry diagnostiek (2026-05-30)

Script: `scripts/diag_funding_short_edge.py` (volledig causaal aan signaal-kant,
niet-overlappende samples, kosten + carry inbegrepen). Output: `reports/diag_funding_short_edge.csv`.
5 assets, 1h-bars, horizons 8/24/72h, ~4.97 jr.

**Verdict: H2 in pure vorm grotendeels GEFALSIFIEERD als universe-brede alpha.**

| Bevinding | Bewijs |
|-----------|--------|
| **Funding-z gate generaliseert NIET.** Pooled `fund_z>=1` short-EV is negatief op alle horizons (-0.0012 / -0.0028 / -0.0024). Per-asset: ETH + DOT(72h) positief, maar SOL/AVAX/LINK negatief. | per-asset tabel |
| **Carry-income is verwaarloosbaar** (~0.0001-0.002 per trade) → de funding-*betaling* is NIET de edge. De thesis "short ontvangt carry" levert in de praktijk niets op. | `carry`-kolom overal ≈ 0 |
| **Wat overleeft: bear-regime + lange horizon.** Pooled `fund_z>=1 & bear` @72h: mean +0.0104, Sharpe 1.04, hit 55%. De edge is **prijs-reversie in bear-regime**, niet carry. | pooled tabel |
| **Korte horizon = momentum, niet reversie.** Pooled `fund_z>=2` @8h sterk negatief (Sharpe -3.0): extreme funding op korte termijn = blow-off continuation, reversie pas op ≥72h. | pooled tabel |
| **Sterke per-asset heterogeniteit.** ETH `fund_z>=1`: Sharpe +0.97/+2.38/+1.17. SOL/AVAX: negatief. DOT @72h: Sharpe +4.18 (maar N=17). → geen universe-constante. | per-asset tabel |
| Sign-logica gevalideerd: controle-bucket `fund_z<=-1` overwegend negatief (short tegen de carry verliest), zoals verwacht. | control-bucket |

**Implicatie voor de roadmap:** de **bear-regime-conditie (D3) doet het echte werk**, niet
funding. Dit *verhoogt* de prioriteit van H1 (asymmetrische barrier-geometrie) en de
regime-aanpak (H6/`long_bear_factor`) boven H2. Een naïeve funding-z short-gate bouwen
is niet de moeite waard; funding hooguit als zwakke per-asset secundaire conditioner bij
ETH/DOT op lange horizon.

**Volgende runnable experiment:** H1 — her-label short-kant met PT≈1/SL≈2 + kortere t_max,
gecombineerd met de bear-regime-gate, en meet short-only OOS Trade-Sharpe per regime.

---

## 4c. EMPIRISCH RESULTAAT — H1/H4 barrier+entry sweep & H3 cross-sectioneel (2026-05-30)

Twee model-vrije test-harnassen (geen retrain), volledig causaal aan signaal-kant,
niet-overlappende trades met per-asset kosten:
- `scripts/diag_short_barrier_sweep.py` → `reports/short_barrier_sweep.csv`
  (6 barrier-geometrieën × 4 entry-condities × 5 assets, triple-barrier short-sim)
- `scripts/diag_xsectional_ls.py` → `reports/xsectional_ls.csv`
  (cross-sectionele momentum long/short, sleeve-decompositie)

### H1 (asymmetrische barrier) — **GEFALSIFIEERD voor EV**
De geflipte/snelle geometrie (PT 1/SL 2 "down the elevator") tilt de **win-rate**
van 0.35 → 0.66, maar **mean-return én Sharpe worden slechter**, niet beter. De
symmetrische 2:1 is juist de béste geometrie voor EV: de zeldzame 2-ATR-winsten wegen
zwaarder dan veel kleine winsten. Win-rate ≠ edge. Pooled `all`: symmetric −0.46 Sharpe
vs flipped −0.62 vs tight −1.55.

### H4 (mean-reversion entry "short de overshoot") — **GEFALSIFIEERD**
`reversion` is in élke combinatie de slechtste conditie (pooled −1.06 tot −1.86 Sharpe).
In opwaarts-driftende assets blijft een overextensie doorlopen — je short de winnaar.

### Regime is de enige directionele hefboom (≈ breakeven)
`bear + symmetric_2_1_t24` is de enige pooled bucket die breakeven raakt
(mean_ret 0.0000, Sharpe −0.004). Bear-gating haalt shorts van duidelijk-negatief naar
vlak — defensief, geen alpha. Dit valideert de bestaande `long_bear_factor`-aanpak.

### De edge is DISPERSIE, niet richting (per-asset)
- **DOT**: positief-EV in *alle* condities (Sharpe tot +0.34) — daalde over de sample.
- **SOL bear**: positief (mean +0.0004, Sharpe +0.66) — beste schone bucket.
- **ETH / LINK / AVAX**: negatief overal, bear het minst-slecht.
→ Geen universele short-geometrie; de winst zit in het verschil tussen assets.

### H3 (cross-sectioneel market-neutral) — **DEELS BEVESTIGD; sterkste route**
Cross-sectionele momentum-selectie (short de relatieve verliezer) verwijdert ~90% van
de structurele short-drag: short-leg Sharpe gaat van **−0.40 (onvoorwaardelijk) → ≈ −0.05**.
De short-leg op zichzelf is **vlak, niet positief**, MAAR levert hedge-waarde:

| Config (lb/rb) | long_only Sharpe | short_leg Sharpe | dollar_neutral Sharpe | neutral vol | neutral MaxDD |
|----------------|------------------|------------------|------------------------|-------------|----------------|
| 20d / 7d  | 0.78 | −0.06 | **0.83** | 0.78 | −0.81 |
| 30d / 7d  | 0.15 | −0.02 | 0.16 | 0.73 | −0.76 |
| controle naive_short_all | — | **−0.42** | — | — | −0.98 |

In de beste config (20d/7d) verhoogt het toevoegen van het short-been de Sharpe
(0.78 → 0.83) én verlaagt het vol (0.92 → 0.78) en MaxDD (−0.85 → −0.81). **Shorts
verdienen hun plek als hedge/diversifier, niet als standalone alpha.**

> ⚠️ Caveat: full-sample, géén CPCV, 6 configs geswept → selection-bias. Het H3-effect
> is param-gevoelig (degradeert bij lange lookback). Vereist proper OOS/CPCV + DSR-deflatie
> vóór enige live-conclusie.

### Overkoepelende conclusie
1. **Directionele short-alpha bestaat structureel niet** in dit universe — geen
   barrier-geometrie (H1), entry-regel (H4) of funding-gate (H2) maakt het positief.
2. **Bear-regime-gating** is de enige directionele hefboom en is *defensief* (≈breakeven);
   de bestaande `long_bear_factor` is dus de juiste richting.
3. **De enige veelbelovende short-VALUE = market-neutral cross-sectioneel (H3/H6):**
   bouw shorts als een β-neutrale overlay met de bestaande `CSMomentumVolumeClock`,
   beoordeeld op **portfolio-Sharpe/MaxDD-bijdrage**, niet op standalone short-return,
   en gevalideerd met CPCV + DSR.

---

## 5. Open vragen voor validatie

- Funding-data dekking: is `feat_meso_funding_zscore_30d` voor alle 5 assets over de
  volle 4.97 jr beschikbaar en causaal (lag toegepast)? (`funding_carry.py:72` past
  `rate_lagged` toe — verifiëren op lengte/gaten).
- Kosten: D3 zegt dat short-CUSUM-events tijdens sell-offs 2-3× bredere spread hebben —
  zit die spread-verbreding al in `compute_dynamic_spread_arr`, of onderschatten we
  short-kosten systematisch?
- DSR-boekhouding: elke nieuwe short-hypothese telt mee in `total_n_hypotheses` (nu 2000).
  Houd het aantal geteste short-varianten bij om selection-bias niet te onderschatten.
