# Plan: vier professionele benaderingen en een gevectoriseerde replay

Spec: `docs/research/2026-11-26-options-pro-strategies-design.md` (lees die eerst; sectie 2 zegt
wat bewust NIET gebouwd wordt).

Branch `claude/project-thread-wylxvn`, gebouwd op PR #29. Eén commit (of een paar) per taak.

## Global Constraints

- Python 3.13, pakket `tradebot`, code onder `src/tradebot/options/`, tests onder
  `tests/unit/options/`. Venv: `/home/user/venv` (`source /home/user/venv/bin/activate`).
- **Taal:** docstrings, commentaar, docs en commitberichten in het Nederlands; identifiers,
  config-sleutels en journal-events in het Engels. Volg de stijl van de bestaande modules
  (moduledocstring met de regel/het algoritme, `__all__`, frozen dataclasses met slots).
- **CI moet groen:** 100 % lijn- én branchdekking op `src/tradebot` (`fail_under = 100` in
  pyproject), `ruff check src tests`, `mypy` zoals de CI het draait, geen `pytest.skip`, geen
  `pragma: no cover` voor nieuwe code, geen `continue-on-error`. Geen nieuwe dependency:
  numpy, scipy, pandas en QuantLib zijn er al. `try/except ImportError` is verboden in `src/`.
- **Regelgebaseerd:** vaste regels en vaste parameters uit `conf/options/default.yaml`; niets
  schatten of trainen op historische data, geen optimalisatie op uitkomsten.
- **QuantLib is de autoriteit** voor prijzen, IV en Greeks (Amerikaanse uitoefening,
  discrete dividenden via FD). Een numpy-pad mag alleen gebruikt worden waar een
  pariteitstest het tegen QuantLib vastpint.
- **Config:** elke nieuwe sleutel in `src/tradebot/schemas/options.py` (StrictModel, met
  grenzen via `Field`) én in `conf/options/default.yaml` met een Nederlandse toelichting.
  Onbekende sleutels zijn een fout.
- **Risico:** greek-limieten op boekniveau (`risk/limits.py`), de risicopoort
  (`risk/gate.py`), HALT/kill switches en de live-schakelaar hergebruiken, niet omzeilen.
  Nieuwe ordersoorten gaan door dezelfde poort. Bij HALT alleen risico afbouwen.
- **Standaard gedrag:** short straddle/strangle blijft uit en vraagt
  `risk.allow_undefined_risk: true`. Alleen replay en paper.
- **Regressie:** de standaard synthetische replay
  (`python -I -m tradebot.options.data.synthetic --out M && python -I -m tradebot.options replay --market M --out R`)
  gaf vóór dit plan 3 geopende posities en eind-equity 25303.60. Taken in fase A moeten dat
  exact behouden. Latere taken die het gedrag bewust veranderen, noteren de nieuwe uitkomst
  en waarom in `CHANGELOG.md`.
- Draai tijdens het werk alleen de tests die je raakt; vóór je commit de hele options-suite
  (`pytest tests/unit/options -q -m "not slow and not regression"`, ±30 s) en de
  dekkingscheck op de bestanden die je wijzigde.

## Task A1: numpy-kernel met pariteit tegen QuantLib-BAW

Nieuw `src/tradebot/options/pricing/batch.py`.

- Arrays (numpy float64) per contract: `spot, strike, is_call, t_model, factor, rate,
  div_yield` waarbij `t_model` en `factor` uit `ql_env.scaled_time(now, expiry)` komen
  (QuantLib rekent in hele dagen vanaf `eval_date`, Actual365Fixed; σ_model = σ·√f,
  r_model = r·f, q_model = q·f; disconteringen `exp(−r_model·t_model)`,
  `exp(−q_model·t_model)`, variantie `σ_model²·t_model`). Lees `ql_env.py`, `vanilla.py`,
  `implied_vol.py`, `greeks.py` en `chain/builder.py` eerst.
- `european_price(...)`: Black(-Scholes) op de forward, zoals QuantLib's `BlackCalculator`.
- `baw_price(...)`: een getrouwe poort van QuantLib's `BaroneAdesiWhaleyApproximationEngine`
  (criticalPrice met de seed van Barone-Adesi-Whaley en de Newton-iteratie met tolerantie
  1e-6 op `|LHS−RHS|/K`; de call met `dividendDiscount >= 1` is Europees; `K = 2/var` als
  de rentediscount binnen 1000 ulps van 1 ligt, anders `−2·ln(rD)/(var·(1−rD))`;
  boven/onder de kritieke prijs de intrinsieke waarde). Gemaskeerde iteratie over de batch.
- `implied_vols(prices, ...)`: dezelfde statuslogica als `implied_vol` (EXPIRED, NO_QUOTE,
  NO_BID, BELOW_INTRINSIC met één tick marge, ABOVE_UPPER_BOUND, prijs ≤ ondergrens → ondergrens
  + ½ tick, out-of-range bij `[iv_min, iv_max]` → NO_CONVERGENCE). Oplossen met een
  gevectoriseerde, beveiligde Newton/bisectie op `[iv_min, iv_max]` tot |Δσ| ≤ 1e-12.
  Contracten waar de kernel niet convergeert, krijgen een eigen status "naar QuantLib"
  (geen IV verzinnen): de aanroeper prijst die met het bestaande QuantLib-pad.
- `baw_greeks(...)`: precies de definities van `greeks.py` voor BAW: delta en gamma met
  spotbumps `h = spot_frac·S`, vega centraal met `b = min(vol_bump, ½σ)` ×0,01, rho centraal
  ±`rate_bump` ×0,01 (met de schaling van de rente via `factor`), theta als herwaardering na
  `step = min(1 dag, resterend/2)` met de `scaled_time` van het verschoven tijdstip (die
  verandert mogelijk `eval_date`; reken de arrays voor `now+step` per contract in Python uit).
- **Pariteitstests** (`tests/unit/options/test_pricing_batch.py`) op een vast raster: calls en
  puts, moneyness 0,7-1,3, DTE 0 (expiratiedag, vóór 16:00) t/m 400, vol 0,05-1,5,
  r ∈ {0, 0,04, 0,08}, q ∈ {0, 0,02, 0,06}:
  - `baw_price` tegen `vanilla.price(..., Method.BAW)`: |Δ| ≤ 1e-9·spot;
  - `european_price` tegen `Method.EUROPEAN`: |Δ| ≤ 1e-10·spot;
  - `implied_vols` tegen `implied_vol(..., Method.BAW)`: zelfde status; bij OK |Δσ| ≤ 2·iv_accuracy,
    en de QuantLib-BAW-prijs bij de kernel-IV ligt binnen 1e-8 van de doelprijs;
  - `baw_greeks` tegen `greeks(..., Method.BAW)` bij dezelfde σ: |Δ| ≤ 1e-6·max(1, |greek|).
  Haalt `baw_price` de 1e-9 niet na een serieuze poging, stop dan en rapporteer
  BLOCKED met de gemeten afwijking: de controller beslist dan over de seed-modus.
- Geen integratie in de keten in deze taak.

## Task A2: keten in batches, FD met startwaarde

`chain/builder.py` (en `PricingConfig`):

- Nieuwe sleutel `pricing.vectorized: bool = true`. Uit = het huidige pad, ongewijzigd.
- `ChainBuilder.build` met `vectorized`: verzamel alle contracten die geprijsd moeten worden
  (niet verlopen, niet stale, niet `skip`). Contracten met `Method.BAW` (scan, geen discreet
  dividend) gaan in één batch door `batch.implied_vols` op de mid (en bid/ask met
  `spread_iv`) en `batch.baw_greeks` bij OK. Contracten met status "naar QuantLib" en alle
  niet-BAW-contracten gaan door het bestaande `price_one`-pad. `PricedOption.iv_mid.method`
  en `.method` blijven exact wat het oude pad zou geven.
- FD-contracten (discreet dividend): startwaarde σ₀ = kernel-IV van een BAW op
  `spot − Σ PV(dividend)` (contante waarde tegen r, ex-datum in de looptijd). Geef
  `implied_vol` een optioneel bracket-argument; probeer eerst Brent op
  `[max(iv_min, σ₀−0,03), min(iv_max, σ₀+0,03)]`; omsluit dat geen wortel, dan het volle
  bestaande pad (zelfde terugvallogica). Uitkomst binnen `iv_accuracy` van het oude pad.
- De volgorde van `Chain.options` blijft de volgorde van `raw`.
- Tests: vectorized aan en uit geven op een synthetische keten (met en zonder dividend)
  dezelfde statussen, IV binnen 2·iv_accuracy en Greeks binnen 1e-5 relatief; telling van
  QuantLib-`NPV`-aanroepen of een spy die aantoont dat BAW-contracten niet meer per stuk door
  QuantLib gaan en dat de FD-solve minder evaluaties nodig heeft.

## Task A3: procespool voor het QuantLib-deel

- `pricing.workers: int | "auto" = "auto"` (auto = min(4, os.cpu_count() or 1); 1 = in-proces).
- De QuantLib-contracten uit A2 (FD, binomiaal, terugval) worden bij workers > 1 en meer dan
  `pricing.pool_min_contracts` (standaard 16) in vaste chunks over een
  `concurrent.futures.ProcessPoolExecutor` verdeeld; de uitkomst wordt in de oorspronkelijke
  volgorde teruggezet. De pool wordt lui gemaakt, hergebruikt en gesloten via
  `ChainBuilder.close()` die `Engine.stop` en de CLI aanroepen. De pool-functie is een
  module-level functie die in tests ook in-proces draait (dekking).
- QuantLib's globale `evaluationDate` is per proces; geen threads.
- Tests: workers=2 geeft exact dezelfde keten als workers=1 (dataclass-gelijkheid); de pool
  wordt gesloten; `auto` resolveert.

## Task A4: meten, regressie en documentatie

- Draai de standaard synthetische replay met `pricing.vectorized: false, workers: 1` en met de
  standaard; beide moeten 3 posities en eind-equity 25303.60 geven. Wijkt de nieuwe af, zoek
  de oorzaak (systematic debugging) en los op; accepteer geen afwijking.
- Noteer de tijden in `CHANGELOG.md` en `docs/runbook-options.md` (sectie replay), met het
  aantal cores. Doel ≥ 5× sneller op 4 cores.
- Voeg een test toe die de replay-uitkomst van een kleine synthetische markt met
  `vectorized` aan en uit vergelijkt (zelfde `summary.json` behalve het journalpad).

## Task B1: horizon-RV, EWMA en VRP per looptijd

`vol_regime.py`, `iv_metrics.py` waar nodig, `schemas/options.py`, config.

- `vol_regime.ewma_lambda: 0.94` en `vol_regime.ewma_weight: 0.0` (standaard 0, zodat het
  regime ongewijzigd blijft; Alessio kan hem aanzetten). Bij gewicht > 0 telt de
  EWMA-variantie van de dagrendementen mee in de blend (gewichten herschaald).
- `vol_regime.mean_reversion_kappa: 4.0` (per jaar). `RVForecast.at_horizon(days)`:
  `σ²(h) = σ_L² + (σ_0² − σ_L²)·(1 − e^{−κh})/(κh)`, h in jaren (kalenderdagen/365),
  σ_0 = de bestaande blend (`forecast`), σ_L = `rv_long` (of `rv_mid` zonder lang venster).
- `VolState.vrp_at(iv, dte_days)` geeft (ratio, punten) tegen `σ(h)`.
- `selection.theta_cover` rekent met `σ(h)` van de vroegste expiratie van de kandidaat in
  plaats van `rv.forecast`. Dat verandert gedrag; noteer de nieuwe replay-uitkomst.
- Tests voor de formule (h→0 geeft σ_0, h→∞ geeft σ_L), EWMA, en de poort.

## Task B2: relative value op de VRP

- Nieuw `options/relative_value.py`: per scanronde de VRP-ratio (ATM-IV30 / σ(30 d)) per
  onderliggende met bekende waarde; z-score over het universum (populatie-σ; bij < 3 namen
  of σ = 0 alle z = 0).
- In `Engine._admitted` (of vlak daarna, vóór de rangorde) wordt `rank_score` van een
  toegelaten kandidaat vermenigvuldigd met `1 + w·z` (credit, `net_credit > 0`) of
  `1 − w·z` (debit), begrensd op [0,5, 1,5], met `relative_value.weight: 0.0` standaard
  (aan te zetten). Spoorregel `rv_z:<z>` in `rule_trace`. Geen poort, alleen rangorde.
- Tests.

## Task B3: aandelenorders en de delta-hedger

- Execution: een aandelenorder (`StockOrder`: symbol, qty ±, limit, purpose `HEDGE`) in de
  `BrokerAdapter`-interface, `PaperBroker` (fill op de limiet als die de spot ± halve
  `paper.stock_spread_bps` (standaard 2) raakt; commissie `paper.commission_per_share`,
  standaard 0.005, minimum 1.0 per order) en de IBKR-adapter (`ib_async.Stock(sym, "SMART",
  "USD")`, LMT, DAY; met de bestaande fake-IB in de tests). Fills boeken als aandelen-legs
  (`Leg(contract=None, ...)`) op de positie die gehedged wordt, en in `Book.shares`.
- Nieuw `options/hedging.py` met een pure functie: per onderliggende de netto delta in
  aandelen van alle open posities waarvan de strategie in `hedging.strategies` staat (optie-
  delta × 100 × aantal + hun aandelen-legs). Is `|delta| > hedging.band_shares`, dan een order
  van `−round(delta)` aandelen (afgerond naar hele aandelen), begrensd zodat
  `|aandelen| × spot ≤ hedging.max_notional_frac × kapitaal`. Alleen op
  `universe.share_eligible`. Interval `hedging.interval_seconds: 300`.
- Config `hedging: {enabled: false, strategies: [gamma_scalp], band_shares: 20,
  max_notional_frac: 0.5, interval_seconds: 300}`.
- De order gaat door de risicopoort (beta-delta limiet) en wordt bij HALT alleen uitgevoerd als
  hij de absolute delta verkleint. Bij het sluiten van een positie worden haar aandelen-legs
  in dezelfde ronde terug naar nul gebracht (aandelenorder, purpose `EXIT`).
- Journal-events `hedge_order`, `hedge_fill`. Tests: paper-fill, IBKR-fake, band, plafond,
  HALT-gedrag, sluiten.

## Task B4: gamma scalping

- Nieuwe strategie `gamma_scalp` in `strategies/vol_trading.py`, registreren in de
  registry (na `long_straddle_strangle`), `StrategiesConfig`, `RegimeAllowedConfig`
  (standaard in `cheap` en `normal`) en config, **standaard `enabled: false`** (hij vraagt
  `hedging.enabled`; staat de strategie aan zonder hedging, dan `ConfigContractError`).
- Entry: alleen `share_eligible`; ATM-straddle (strike het dichtst bij de forward) met DTE in
  `dte: [30, 60]`; IV van die expiratie (gemiddelde van de call- en put-IV) ≤
  `max_iv_rv_ratio` (0.90) × σ(h); geen earnings in de looptijd; sizing via `common.size`
  op het debit. `rank_score` = σ(h)/IV − 1.
- Beheer (eerste die geldt): expiratieregel; waarde ≥ `take_profit_multiple_debit` (1.30) ×
  debit (inclusief gerealiseerde hedge-P&L van de aandelen-legs); waarde ≤
  `stop_loss_frac_debit` (0.60) × debit; IV ≥ σ(h) (`iv_normalized`); DTE ≤ `close_dte` (14).
- Tests voor entry, elke exit en de registry-regel.

## Task B5: calendar op forward-vol

- `CalendarSpread` krijgt een tweede toelatingsroute naast de bestaande IVP-regel:
  `forward_vol` met `σ_f² = (σ₂²T₂ − σ₁²T₁)/(T₂ − T₁)` uit de ATM-IV van de twee
  gekozen expiraties (T in jaren, `t_real`). Toegelaten als `σ_f ≤ max_fwd_vol_rv_ratio`
  (1.00) × σ(h₂−h₁) én IV₁ ≥ `min_near_iv_rv_ratio` (1.05) × σ(h₁). Config
  `calendar_spread.entry_rule: ivp | forward_vol | either` (standaard `ivp`, gedrag
  ongewijzigd). Negatieve σ_f² → niet toegelaten (spoorregel). Tests.

## Task C1: short strangle met harde limieten

`strategies/short_vol.py`, `risk/gate.py`, `engine/service.py`, config.

- `short_straddle_strangle` krijgt: `underlyings: [SPY, QQQ, IWM]` (allow-list),
  `max_per_underlying: 1`, `short_delta_exit: 0.30` (een short leg met |delta| groter → CLOSE
  STOP), de bestaande stop op 2 × credit en `close_dte: 21`.
- Boekbreed: `risk.max_undefined_stress_loss_frac: 0.06`: de som van het stressverlies
  (bestaande `STRESS_MOVE` 20 %, per positie, in USD) van alle open undefined-risk posities plus
  de nieuwe mag niet groter zijn dan die fractie van het kapitaal. Afdwingen in de risicopoort
  als extra check (zelfde `CheckResult`-mechaniek), zodat het ook in het journal staat.
- STRESS: is het effectieve regime van de onderliggende STRESS (inclusief STRESS op SPY voor
  het hele boek), dan gaan alle undefined-risk posities dicht met `Urgency.STOP`, reden
  `regime_stress`. Dit is het enige regime-effect op bestaande posities; documenteer het in
  de moduledocstring van `vol_regime`.
- Standaard blijft de strategie uit en blijft `allow_undefined_risk` false.
- Tests voor allow-list, max per onderliggende, delta-exit, boekgrens en STRESS-sluiting.

## Task C2: premie verkopen met VRP per looptijd

- `credit_spread`, `iron_condor`, `short_straddle_strangle`: optionele poort
  `min_vrp_ratio` (standaard `null` = uit, gedrag ongewijzigd; aanbevolen 1.10 in een
  commentaarregel): de gemiddelde IV van de short legs tegenover σ(h) van die expiratie.
  Spoorregel `vrp_h:<ratio>`.
- `credit_spread` krijgt `short_delta_exit` (standaard `null`; aanbevolen 0.40): short leg
  |delta| boven die waarde → CLOSE STOP `delta_breach`, vóór de ITM-stop.
- Tests.

## Task D1: fair-value limieten

`execution/orders.py`, `engine/service.py`, config.

- `execution.limit_anchor: mid | fair` (standaard `mid`: gedrag ongewijzigd).
  Bij `fair`: fair value per leg = QuantLib-waarde (scan-engine van de keten) bij de IV van
  de gladde smile (`pricing.surface`) voor die strike en expiratie; zonder surfacepunt de mid.
  Combo-fair = Σ ratio × fair_leg. `limit_0` = combo-fair, begrensd tot [combo-bid-kant,
  combo-ask-kant] en afgerond op de combo-tick in ons voordeel.
- De walk concedeert nooit voorbij `fair + max_edge_frac × |natural − mid|` richting natural
  (`execution.max_edge_frac: 0.5`) voor ENTRY; exits en STOP houden hun bestaande concessie.
- Entry geweigerd (journal `entry_skipped`, reden `spread_vs_fair`) als
  `|natural − mid| × 2 > execution.max_spread_frac_fair × |fair|` (standaard 0.25).
- Tests met handgemaakte ketens.

## Task D2: gamma management per positie

- `risk/gamma.py`: `gamma_loss_1d(pos, chain, sigma_1d)` = ½·|Γ_pos|·(S·σ_1d)² in USD, alleen
  voor short-gamma posities, met σ_1d = `RVForecast.at_horizon(1)` (geannualiseerd) × √(1/252).
  Dezelfde σ_1d gebruikt E1.
- In `Engine._manage`, vóór de strategie-regels: als
  `gamma_loss_1d > risk.max_gamma_loss_frac_credit × entry_credit_usd` voor een positie met
  credit (standaard 0.5; `null` = uit), dan CLOSE STOP `gamma_risk`. Standaard `null`
  (gedrag ongewijzigd), aanbevolen waarde in commentaar.
- Tests.

## Task E1: automatische sizing op risico

`strategies/common.py` `size`, `SizingConfig`, config.

- `sizing.daily_risk_frac: null` (uit). Aan: per eenheid de verwachte 1-dags P&L-σ
  `sqrt((Δ·S·σ_1d)² + (½·Γ·S²·σ_1d²)²·2 + (V·volvol_1d)²)` in USD (Greeks per eenheid × 100,
  V per volpunt, `sizing.vol_of_vol_daily_points: 1.0`); eenheden = min(bestaande regel,
  floor(daily_risk_frac × kapitaal / σ_unit)).
- `sizing.drawdown_brake: [0.05, 0.10]`: eenheden × factor die lineair van 1 naar 0,5 loopt
  tussen die drawdowns (drawdown van de equity-piek in `EngineState.equity`), afgerond naar
  beneden, minimaal 0. Standaard aan; check of de replay-uitkomst verandert en noteer.
- De Greeks komen uit de keten (`position_greek`); ontbreken ze, dan 0 eenheden.
- Tests.

## Task E2: backteststatistiek en sweep

- `engine/report.py`: uit de journal en `state.equity` per strategie: aantal gesloten posities,
  win rate, gemiddelde en totale P&L, grootste verlies; voor het boek: totaalrendement, max
  drawdown, Sharpe en Sortino op dagrendementen (√252, risicovrij 0). In `summary.json` onder
  `stats` en in `ReplayResult`.
- CLI `tradebot.options sweep --markets DIR1 DIR2 ... --out DIR [--config]`: draait elke markt
  als replay in een procespool (`--workers`, standaard auto) en schrijft `sweep.json` met de
  stats per markt. Geen parameter-grid: dezelfde config op meerdere markten.
- Runbook bijwerken. Tests (kleine markten, workers=1 voor de dekking plus één run met 2).

## Task F1: documentatie en afronding

- `docs/runbook-options.md`: per nieuwe schakelaar wat hij doet en hoe aan te zetten;
  sectie "wat voor een particulier niet haalbaar is" (uit spec sectie 2).
- `README.md` (options-sectie) en `CHANGELOG.md` bijgewerkt met de uiteindelijke replay-uitkomst
  en -tijd.
- Spec bijwerken als er tijdens de bouw rulings zijn genomen die ervan afwijken.
