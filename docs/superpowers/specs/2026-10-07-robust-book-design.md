# Robuust boek v1 — ontwerp en onderzoeksprocedure (2026-10-07)

Status: **bevroren vóór de eerste kandidaat-run**. Dit document en
`conf/research/preregistration_robust_book.yaml` leggen vast wat er wordt getoetst,
waarop, met welke poorten en hoe er gekozen wordt. Elke wijziging na de eerste run is
een gedateerd amendement en kost een trial.

## 1. Vertrekpunt (gemeten, niet aangenomen)

* **Data.** Alleen de gecertificeerde PIT-store: zes Bybit-perpetuals (BTC, ETH, SOL,
  AVAX, LINK, DOT), dagbars 2020-03-26 → 2026-08-23, 8h-funding, dagelijkse open
  interest. Elke externe marktdatahost (Binance, Bybit, OKX, Kraken, Yahoo, Stooq,
  FRED, CoinGecko, Ken French) geeft 403 op de egress-proxy van deze omgeving.
* **Breedte.** Gemiddelde paarsgewijze correlatie van de dagrendementen 0,73. De
  effectieve breedte van het universum is ≈ 1,3 onafhankelijke weddenschappen.
* **Funding.** Een long-perp betaalt over de hele steekproef 5–19 % per jaar
  (LINK 18,9 %, BTC 12,9 %); in 2025–2026 nog 2–5 %. Funding is geen detail maar een
  rendementspost van dezelfde orde als het verwachte overrendement.
* **Executie.** `open(t+1) == close(t)` op elke bar (mediaan 0,0 bp verschil): de markt
  handelt 24/7, dus een besluit op de close van *t* is direct uitvoerbaar tegen ≈ die
  prijs. De authoritative engine vult op de close van *t+1* — een volle dag later dan
  nodig — en is daarmee de pessimistische grens, niet de realistische.
* **Wat al geprobeerd is op deze data** (zie het auditrapport): meer dan 2.776 trials in
  de oude ledger (gereset op 2026-09-26), daarna 4 (wekelijkse meta-label, OOS-AUC 0,49)
  en 47 (trend/carry-programma op de ongemergde branch `elegant-dijkstra`: beste eerlijke
  trendschatting Sharpe 0,3–0,5) plus 1 (BTC-MA100 op `jolly-feynman`). Bij elkaar
  minstens 52 bekende trials op deze zes munten sinds de reset.

**Prior.** Op grond hiervan is de verwachte netto Sharpe van elke kandidaat hieronder
0,3–0,7. Dit programma verwacht dus dat poort G1 (Sharpe ≥ 1) **niet** gehaald wordt.
Het onderzoek is erop ingericht dat eerlijk vast te stellen, niet om het te weerleggen.

## 2. Wat hier nieuw is ten opzichte van eerder werk

1. **Een ontleding in premies in plaats van één signaal.** Beta (vol-gemanaged long
   majors), trend (symmetrisch, long-short) en carry (dollar-neutraal) zijn drie
   economisch verschillende bronnen, met verschillende regimeprofielen: beta verdient in
   bull, trend in langdurige bewegingen beide kanten op en is convex in crashes, carry
   verdient op de prijs van hefboom.
2. **Een kostenmodel dat gewichtsdrift meetelt.** De vorige onderzoeksengine telde
   turnover als `|w_t − w_{t−1}|` en vergat dat een constant gewicht na een prijsbeweging
   moet worden bijgehandeld. Hier: `trade = w_doel − w_gedrift`, met
   `w_gedrift = w (1+r_i)/(1+r_boek)`.
3. **Een volledige P&L-trap**: bruto → fees → spread → impact → funding → netto, per bar,
   met een sluitende identiteit die bij elke run wordt gecontroleerd.
4. **Geen lookbackselectie.** Trend is een gelijk gewogen ensemble over een
   log-raster 5–160 dagen met stap √2 (11 lookbacks), vooraf vastgelegd. Het eerdere
   werk liet zien dat (7…112) en (5…160) elk 0,3 tot 1,0 kunnen geven; het raster is de
   vereniging, niet de winnaar.
5. **De eindkandidaat gaat door de authoritative event-driven engine** (`backtest/engine.py`)
   met de soevereine RiskEngine. Vectorized uitkomsten blijven screeningsmateriaal
   (`NOT_ADMISSIBLE_AS_PROMOTION_EVIDENCE`).

## 3. Vensters

| Venster | Periode | Bars | Gebruik |
| --- | --- | --- | --- |
| warm-up | 2020-03-26 → 2021-11-14 | — | alleen signaalgeschiedenis, nooit gescoord |
| `TRAIN` | 2021-11-15 → 2023-12-31 | 777 | inclusieregel van de combinatie |
| `VALIDATE` | 2024-01-01 → 2025-09-04 | 613 | consistentiepoort |
| `W_DEV` = TRAIN ∪ VALIDATE | 2021-11-15 → 2025-09-04 | 1.390 | alle poorten G1–G9, robuustheid, selectie |
| `GATE` | 2025-09-05 → 2026-08-23 | 353 | **één lezing**, één kandidaat, via `validation/holdout.py` |

De split 2025-09-05 is het bevroren poortsample uit `artefacts/governance/holdout_lock.json`
(gelezen: `[]`). **Voorbehoud:** het ongemergde `elegant-dijkstra`-programma las 2024-01 →
2026-06-23 voor de trendfamilie; `jolly-feynman` las 2026-06-24 → 2026-08-23 voor een
BTC-trendregel. Voor een trendkandidaat is `GATE` dus niet maagdelijk; voor beta en carry
wel. Dat wordt bij de lezing vermeld en niet weggepoetst.

## 4. Conventies (vast, niet aan gedraaid)

* Besluit op de close van *t* met data ≤ *t*; uitgevoerd tegen ≈ close(*t*) plus kosten;
  het nieuwe gewicht rendeert vanaf *t+1* (`lag = 1`). Stresstest: `lag = 2` en `lag = 3`.
* Kosten per eenheid verhandelde notional: taker 5,5 bp + halve spread 1,0 bp
  (`conf/execution/fees.yaml`) plus vierkantswortelimpact
  `η · σ_dag · √(order/ADV)` met η = 2,99 (`conf/execution/impact.yaml`, conservatieve
  bovengrens) bij een boek van $100k (`conf/backtest/default.yaml`). Stress: kosten ×2 en
  ×3, +10 bp extra slippage, boekgrootte $1M / $10M / $100M (capaciteit).
* Funding: de som van de 8h-afrekeningen binnen de bar op de gehouden positie; long betaalt
  positieve funding. Geen rente op vrije USDT (conservatief).
* Volatiliteit: EWMA, span 60, causaal (inclusief het rendement van *t*). Covariantie idem.
* Een munt doet mee zodra hij ≥ 60 bars koersgeschiedenis heeft.
* Sizing per munt: `w_i = S_i · (τ_a / σ̂_i) / N_live` met τ_a = 0,20. Daarna één
  portefeuillefactor `k = min(τ_p / σ̂_p, k_max)` met τ_p = 0,20 en k_max = 2,0;
  σ̂_p uit de EWMA-covariantie. Daarna |w_i| ≤ 1,0 per munt en bruto ≤ 3,0.

## 5. Hypothesen (zeven trials)

| Code | Rol | Hypothese | Ontwerp |
| --- | --- | --- | --- |
| `B1_BTC_HOLD` | referentie | — | 1,0× long BTC-perp, dagelijks herbalanceerd |
| `B2_EW_HOLD` | referentie | — | gelijk gewogen long over de levende munten, bruto 1,0 |
| `C1_TREND_LS` | kandidaat | tijdreeks-momentum: crypto-trends houden aan door trage informatieverwerking en kuddegedrag (Moskowitz-Ooi-Pedersen 2012; Liu-Tsyvinski 2021) | per lookback `z = ln(P_t/P_{t−L}) / (σ̂ √L)`, `s = clip(z, −1, 1)`, `S = gemiddelde over L` (11 lookbacks 5…160), long-short |
| `C2_TREND_LF` | kandidaat | idem, maar zonder short-premie (shorts betalen in crypto vaak negatieve funding en hebben een structureel negatieve drift tegen zich) | `S = max(S_C1, 0)` |
| `C3_VOLMAN_CORE` | kandidaat | marktpremie van crypto + volatiliteitsclustering: inverse-vol-sizing verbetert de Sharpe van de marktpremie (Moreira-Muir 2017) | long BTC en ETH, gelijk risico, `S = +1` |
| `C4_CARRY_XS` | kandidaat | funding is de prijs van hefboom; short de namen waar longs het meest betalen (Schmeling-Schrimpf-Todorov 2023, "Crypto carry") | bekende carry = som funding over 7 bars t/m *t−1*; top-3 short, bodem-3 long, inverse-vol binnen elk been, dollar-neutraal, wekelijks herbalanceren |
| `C5_COMBO` | kandidaat | diversificatie tussen de premies | gelijk gewogen som van de kandidaten C1–C4 waarvan de **TRAIN**-Sharpe > 0 is (één lezing van TRAIN), daarna portefeuillefactor |

C5 gebruikt alleen TRAIN voor de inclusie. VALIDATE en GATE spelen in die keuze geen rol.

## 6. Poorten (alle op `W_DEV`, netto, tenzij anders vermeld)

| # | Eis |
| --- | --- |
| G1 | netto Sharpe ≥ 1,0 |
| G2 | netto CAGR ≥ 15 % bij τ_p = 20 % (geen extra hefboom om de doelstelling te halen) |
| G3 | Sharpe ≥ 0,5 in TRAIN **én** in VALIDATE |
| G4 | max drawdown ≤ 25 % (mandaat) |
| G5 | Sharpe bij 2× kosten ≥ 0,75 × basis **en** bij `lag = 2` ≥ 0,5 × basis |
| G6 | ≥ 80 % van de parameterverstoringen heeft Sharpe > 0,5 × basis, en geen enkele is negatief |
| G7 | ondergrens 95 %-circulaire-blokbootstrap-interval van de Sharpe > 0 |
| G8 | DSR ≥ 0,95 bij M = 7 (dit programma). Ook gerapporteerd bij M = 59 (+52 bekende trials op deze data) |
| G9 | PBO < 0,5 over de familie parameterverstoringen (CSCV, 16 deelblokken) |
| G10 | `GATE`, één lezing: `(SR_gate − SR_dev) / SE_gate > −1,645` **en** max drawdown in `GATE` ≤ 25 % |

Een kandidaat wordt alleen voor papertrading voorgedragen als G1–G10 alle groen zijn.

## 7. Robuustheidsbatterij (gevoeligheden, geen trials, nooit geselecteerd)

Parameterverstoring (lookbackraster ×0,5/×0,75/×1,25/×1,5/×2; vol-span 20/40/90/120;
carryvenster 3/14/30; k_max 1/3), kosten ×0/×2/×3 en +10 bp, `lag` 2/3, signaalruis (20 seeds,
σ = 0,25 × sd van het signaal), ontbrekende data (5 % van de bars bevroren, 20 seeds),
herbalanceren elke 3/7 bars, universum (leave-one-out, alleen BTC+ETH, zonder BTC), startdatum
+3/+6/+9/+12 maanden, einddatum −3/−6/−12 maanden, kalenderjaren, regimes (BTC boven/onder zijn
200-daags gemiddelde, hoge/lage vol, hoge/lage gemiddelde correlatie), circulaire blokbootstrap,
Monte Carlo-drawdownverdeling, boekgrootte $1M/$10M/$100M.

## 8. Selectie: robuustheidsscore (0–100, uitsluitend op `W_DEV`)

| Punten | Component |
| --- | --- |
| 25 | `clip(SR / 1,0, 0, 1)` |
| 15 | consistentie `clip(min(SR_train, SR_val) / SR, 0, 1)`; 0 als SR ≤ 0 |
| 10 | kosten `clip(SR_2x / SR, 0, 1)` |
| 10 | vertraging `clip(SR_lag2 / SR, 0, 1)` |
| 10 | plateau: aandeel verstoringen met SR > 0,5 × basis |
| 10 | zekerheid `clip((Φ(SR/SE) − 0,5) / 0,475, 0, 1)` (Lo-SE) |
| 10 | drawdown `clip(1 − MDD / 0,40, 0, 1)` |
| 5 | overfit `1 − PBO` |
| 5 | universum: aandeel leave-one-out-varianten met SR > 0 |

De kandidaat (C1–C5) met de hoogste score is de enige die `GATE` leest, ook als hij G1–G9 niet
haalt: de opdrachtgever vraagt expliciet om out-of-sample bewijs, en één lezing voor één
hypothese is wat het slot toestaat.

## 9. Trial-boekhouding

Zeven geplande trials (B1, B2, C1–C5) worden in `artefacts/governance/hypothesis_ledger.json`
geboekt vóór het bevriezen van de preregistratie. Gevoeligheidsruns zijn geen trials en worden
volledig gerapporteerd. Elke extra variant na het bevriezen is een amendement met een eigen
ledger-entry.
