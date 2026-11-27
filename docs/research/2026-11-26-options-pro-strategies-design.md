# Options: vier professionele benaderingen en een gevectoriseerde replay

Datum: 2026-11-26. Bouwt voort op PR #29 (vol-regime, VRP, theta harvesting). Alleen replay en
IBKR-paper, fictief geld. Prijzen, IV en Greeks blijven QuantLib (Amerikaans, dividenden);
alles wat sneller wordt, wordt tegen QuantLib geverifieerd.

## 1. Wat er al is

| Benadering | Al aanwezig | Ontbreekt |
|---|---|---|
| Volatility trading | VRP (ATM-IV30 tegenover HAR-vormige RV-verwachting), regimes, long straddle/strangle (earnings, lage IV), calendar | VRP per looptijd, delta hedging, gamma scalping, vol spread op forward-vol |
| Market making | price-walk vanaf mid naar natural | limiet op fair value binnen de spread, gamma management per positie |
| Quantitative | vaste RV-blend, replay | horizon-RV-verwachting, relative value tussen onderliggende, backtest-statistiek, automatische sizing op risico |
| Premium selling | credit spread, iron condor, calendar, short strangle (uit, ongelimiteerd risico) | harde limieten voor de short strangle, VRP-poort per looptijd |

De replay duurt 2,5 minuut voor 30 dagen en drie onderliggende. Profiel: 95 % zit in
QuantLib-`NPV`, 229.000 aanroepen. Ongeveer 40.000 daarvan zijn FD-waarderingen (discreet
dividend in de looptijd) van 2,6 ms; de rest is BAW van 0,06 ms. Een IV-solve kost gemiddeld
14 waarderingen, de Greeks 6,6.

## 2. Eerlijk over wat voor een particulier haalbaar is

* **Market making** in de echte zin (beide kanten quoten, spread verdienen, rebates) is niet
  haalbaar: geen co-locatie, geen market-maker-rebates, en elke fill op onze limiet is
  waarschijnlijk adverse selection. Wat wel zinvol is: niet de spread weggeven (limieten op
  fair value), gamma beheersen en posities bijsturen.
* **Statistische arbitrage** in de echte zin (dispersie, honderden namen, snelle executie) past
  niet bij 25k en vijf onderliggende. Wat wel kan: relative value, dus premie verkopen waar de
  VRP het rijkst is en kopen waar hij het goedkoopst is. Dat is geen arbitrage en heet ook zo
  niet in de code.
* **Delta hedging met aandelen** kan alleen op AAPL en MSFT (`share_eligible`). US-ETF's
  (SPY, QQQ, IWM) mogen vanuit de EU niet als aandeel gekocht worden (geen KID). Hedgen met
  synthetische aandelen of futures is bij 25k te grof (100 delta per stap tegen ±12 delta
  beweging per dag per straddle). ETF-posities sturen daarom bij met opties, zoals nu.
* **Short strangles** hebben onbeperkt risico. Ze blijven standaard uit
  (`risk.allow_undefined_risk: false`); Alessio beslist of ze aan gaan.

## 3. Fasen

Eén branch en één draft-PR, met een commit per taak, op PR #29 als basis. Elke fase laat de CI
groen (100 % dekking, pip-audit, harde dependencies, geen skips).

### Fase A: replay vectoriseren (eerst, want meetbaar en los van de rest)

1. **`pricing/batch.py`**: numpy-kernels over arrays van contracten: Black-Scholes Europees en
   een poort van QuantLib's Barone-Adesi-Whaley, met dezelfde tijdschaling als `ql_env`
   (`t_model`, σ·√f, r·f, q·f). Een gevectoriseerde IV-solver (Newton met bisectie-vangnet,
   dezelfde grenzen, ticks en foutcodes als `implied_vol`) en Greeks met precies de bumps van
   `greeks.py`.
   **Pariteitspoort in de tests**, op een raster (calls/puts, ITM/ATM/OTM, 1-400 dagen,
   vol 5-150 %, r en q, de expiratiedag): prijs |Δ| ≤ 1e-9·spot tegen QuantLib-BAW, IV
   |Δ| ≤ 1e-7, Greeks |Δ| ≤ 1e-6 relatief. Haalt de poort het niet, dan geldt de kernel
   alleen als startwaarde (zie 2) en blijft QuantLib het eindgetal geven.
2. **`ChainBuilder` per batch**: contracten zonder discreet dividend (BAW-scan) gaan in één
   keer door de kernel. Contracten met FD krijgen een startwaarde uit de kernel (BAW op spot
   min de contante dividenden) en een smal Brent-interval in QuantLib, met terugval op het
   volle interval. De FD-uitkomst blijft QuantLib, binnen `iv_accuracy`.
3. **Procespool** voor het QuantLib-deel (`pricing.workers`, standaard `auto` = min(4, cores)),
   met vaste volgorde: dezelfde invoer geeft dezelfde keten.
4. **Meten en vastleggen**: de standaardreplay moet exact hetzelfde resultaat geven (3 trades,
   +303,60 USD) en op 4 cores minstens 5 keer sneller zijn. Schakelaar `pricing.vectorized`
   (standaard aan) zet het oude pad terug.

De beslissing (risicopoort, scenario's, binomiaal) en de orders blijven op het bestaande
QuantLib-pad.

### Fase B: volatility trading

1. **Horizon-RV en VRP per looptijd** (`vol_regime`): de RV-verwachting voor horizon h met
   vaste mean reversion naar de lange RV, `σ²(h) = σ_L² + (σ_0² − σ_L²)·(1 − e^{−κh})/(κh)`,
   κ vast in config (geen fit). Plus een EWMA-component (λ = 0,94, RiskMetrics) in de blend.
   De VRP van een kandidaat is de IV van ZIJN expiratie tegenover σ(h) met h = zijn DTE.
2. **Relative value**: z-score van de VRP over het universum; credit-kandidaten krijgen een
   hogere rang naarmate hun VRP rijker is dan die van de rest, debit-vol-kandidaten naarmate
   hij goedkoper is. Alleen de rangorde verandert, geen poort.
3. **Aandelenorders en een delta-hedger**: `StockOrder` in paper- en IBKR-broker; een overlay
   die per onderliggende de netto delta van de aangewezen posities terugbrengt zodra
   |delta| > `hedging.band_shares`, met een plafond op de aandelenwaarde. Alleen op
   `share_eligible` en alleen voor strategieën in `hedging.strategies`. Bij een HALT alleen
   afbouwen.
4. **Gamma scalping** (`gamma_scalp`): long ATM-straddle als de IV van die looptijd onder
   σ(h)·`max_iv_rv_ratio` ligt (CHEAP), gehedged door de hedger. Exits: winstdoel, vega-stop,
   maximale looptijd, IV terug op RV.
5. **Vol spread op forward-vol** (calendar bijgewerkt): koop de calendar als de forward-vol
   tussen de twee expiraties, `σ_f² = (σ₂²T₂ − σ₁²T₁)/(T₂ − T₁)`, onder σ(h) ligt en de near
   leg rijk is.

### Fase C: premium selling met strak risicobeheer

1. **Short strangle met harde limieten** (blijft uit): alleen op een allow-list (standaard
   SPY, QQQ, IWM), alleen in RICH, maximaal één per onderliggende, een boekbrede grens op het
   stressverlies (20 %-beweging) van alle undefined-risk posities samen, verplichte stop op
   2 × credit, sluiten als een short leg |delta| > 0,30, sluiten bij 21 DTE, en bij STRESS op
   SPY: alles met onbeperkt risico direct dicht (STOP).
2. **Iron condor, credit spread, calendar**: VRP-poort per looptijd uit B1; de theta-dekking
   rekent met σ(h); de credit spread krijgt dezelfde delta-breach-exit.

### Fase D: retail-technieken uit market making

1. **Fair-value limieten**: de eerste limiet is de QuantLib-waarde op de gladde smile
   (`pricing.surface`), begrensd tot bid..ask en afgerond in ons voordeel; de walk concedeert
   nooit voorbij fair value + `max_edge_frac` van de spread; geen entry als de spread breder is
   dan `max_spread_frac_fair` van fair value.
2. **Gamma management per positie**: een short-premiepositie gaat dicht als haar
   gamma-verlies bij een beweging van één dag-RV groter wordt dan `max_gamma_frac_credit` van
   het ontvangen credit (gamma explodeert vlak voor expiratie). De boekgrens op gamma blijft.

### Fase E: kwantitatief, sizing en backtest

1. **Automatische sizing op risico**: eenheden = het minimum van de huidige max-verlies-regel en
   een vol-target-regel (verwachte 1-dags P&L-σ uit delta, gamma en vega bij σ(1 dag) en een
   vaste vol-of-vol ≤ `sizing.daily_risk_frac` van het kapitaal), vermenigvuldigd met een
   drawdown-rem (lineair van 1 naar 0,5 tussen 5 % en 10 % drawdown). Geen Kelly: dat vraagt
   een geschatte winstkans.
2. **Backtest-statistiek en sweep**: `summary.json` krijgt per strategie trades, win rate,
   gemiddelde P&L en max drawdown, en voor het boek Sharpe en Sortino op dagrendement.
   `tradebot.options sweep` draait meerdere markten parallel (procespool). Dit is een
   robuustheidscontrole, geen optimalisatie: er wordt niets op de uitkomst afgesteld.

## 4. Wat niet verandert

Kill switches, HALT-bestanden, de greek-limieten op boekniveau en de live-schakelaar blijven
zoals ze zijn en worden hergebruikt. De crypto-code blijft onaangeroerd. De paper-bot op
Alessio's pc draait door op de code van PR #29; deze PR verandert daar niets aan tot hij
gemerged en opnieuw gebouwd is.

## 5. Afwijkingen tijdens de bouw

Wat er anders is uitgekomen dan hierboven beschreven (rulings uit de bouw):

* **Versnelling 4,4× in plaats van ≥5×** (A4: 137 s naar 31 s op 4 cores). Het QuantLib-deel
  (FD met een discreet dividend) blijft ±23 s en de rest van de engine is serieel. Een
  vergelijkingsmeting op één moment geeft voor de eindstand 41 s tegen 171 s vóór de branch
  (4,2×), en 41 s ook voor de code van na A4; zie het runbook, hoofdstuk "Snelheid".
* **De FD-solve met startwaarde (smal Brent-interval)** vindt de IV binnen `iv_accuracy` op een
  iets ander punt dan het volle interval. Vier `risk_decision`-waarden in het journal
  verschillen daardoor in het vijfde cijfer; beslissingen, fills en equity zijn identiek.
  `pricing.vectorized: false` geeft een byte-gelijk journal.
* **`execution.limit_anchor` staat standaard op `mid`**; `fair` (sectie 3, D1) is opt-in. Een
  entry waarvan de eerste limiet al voorbij het plafond fair + `max_edge_frac` × spread ligt,
  wordt niet gestuurd (`entry_skipped`, reden `max_edge`) in plaats van een order met een
  limiet voorbij het plafond.
* **Gamma management (D2) staat standaard uit** (`risk.max_gamma_loss_frac_credit: null`,
  aanbevolen 0,5); de sleutel heet `max_gamma_loss_frac_credit`, niet `max_gamma_frac_credit`.
* **De drawdown-rem (E1) staat standaard aan**, maar is in de standaardreplay niet actief: de
  grootste drawdown is daar 0,33 %, ver onder de 5 % waar de rem begint.
* **Settlement (E2)**: assignment, uitoefening en expiratie sluiten nu ook de positie in het
  journal (`position_closed`) en een uitvoering rekent elke open positie met dat contract af,
  niet alleen de eerste. Alle aandelenfills (ook de afbouw van de hedge) tellen mee in de
  gerealiseerde P&L van de positie (B4), zodat een vlakke ronde gelijk is aan de cashflow.
* **Short strangle** blijft uit en vraagt `risk.allow_undefined_risk: true`; Alessio beslist.

