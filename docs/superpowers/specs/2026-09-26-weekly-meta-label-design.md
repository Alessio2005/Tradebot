# Ontwerp: wekelijkse ML-meta-label-strategie op dagdata

> **Status:** ontwerp, per sectie goedgekeurd door de eigenaar op 2026-09-26,
> daarna op dezelfde dag herzien na (a) de implementatieplanning en (b) een
> methodologische review door de eigenaar. §17 somt elke herziening op.
> **Vervolg:** implementatieplan `docs/superpowers/plans/2026-09-26-weekly-meta-label-research.md`.
> **Uitgangspunt:** de schone lei van commit `4a9588b`. Er is geen eerdere
> hypothese waarop dit ontwerp leunt; de validatie hieronder beslist opnieuw.

---

## 1. Doel en wat "winstgevend" hier betekent

Eén model, geen sleeves, dat **1 à 2 keer per week** een trade neemt op zes
crypto-perpetuals, met een symmetrische **1:1 triple barrier**, een licht
ML-filter, een volatiliteitsmodel, fracdiff-features en een risicolaag die op
kansrekening is gebouwd.

"Haalbaar winstgevend" is hier een meetbare uitspraak, geen belofte. Elke poort
hieronder heeft een formele definitie in de genoemde sectie; er is geen poort
zonder getal.

| # | Poort | Eis | Definitie |
|---|---|---|---|
| G1 | Geldigheid | op geschudde labels ligt de OOS-AUC in elke replicatie binnen [0,45; 0,55] | §9.4 |
| G2 | Het filter voegt iets toe | ΔSR = SR(gefilterd) − SR(ongefilterd), beide met **dezelfde vaste risicofractie**, op **dezelfde kalenderas** met nul op vlakke dagen: ondergrens 95%-interval (Ledoit-Wolf) > 0 | §9.7 |
| G3 | Edge na kosten | het Kelly-boek: netto Sharpe > 0 én ondergrens 95%-blokbootstrapinterval > 0 | §9.6 |
| G4 | Selectie-eerlijkheid | DSR ≥ 0,95 bij `M = 4` — op ~5 jaar dagdata een netto Sharpe van **≈ 1,2** (1,199 bij n_obs = 1850, gemeten 2026-09-26 met `backtest.metrics.deflated_sharpe`) | §9.6 |
| G5 | Trefkans | Wilson-ondergrens van het gerealiseerde trefpercentage van de genomen trades > hun gemiddelde ex-ante break-even `p_be` | §6.2, §10.2 |
| G6 | Overfit (ondersteunend) | PBO < 0,25 over de vier varianten; grof bij vier varianten | §9.5 |
| G7 | Drawdown | Monte Carlo-kans op −25 % binnen een jaar ≤ 10 % | §10.5 |
| G8 | Adequaatheid | ten minste 100 genomen OOS-trades | §14 |

Haalt het de poorten niet, dan is dat het antwoord (§14).

## 2. Scope

**Binnen:** de gecertificeerde dagdata (OHLCV, funding, open interest) van BTC,
ETH, SOL, AVAX, LINK en DOT; één gepoold meta-label-model; bet sizing,
risicolaag en een reproduceerbare **dagdata-executieconventie** (§12) voor de
backtest; het oordeel en een eenmalige holdout-rooktest. Dit is **Plan 1**.

**Buiten (Plan 2, pas na een `PASS`):** data bijwerken, papertrading, echte
orders met stops bij de exchange, post-only/taker-fill-logica, live-bewaking.

**Buiten (altijd):** intraday-data en dollarbars (de eigenaar koos voor
1d-data; uit dagdata worden dollarbars meerdaags en halveert het aantal
leervoorbeelden), sleeves of meerdere strategieën naast elkaar,
hyperparameter-zoektochten (Optuna), featureselectie op prestatie, diepe of
zwaar getunede modellen, wijziging van het risicomandaat.

## 3. Pijplijn in één oogopslag

```text
PIT-store (1d OHLCV, funding 8h -> dag, OI 1d), gecertificeerd, t/m asof 2026-08-23
  -> CUSUM-events op log-close, drempel h = k * sigma_{t-1}      (primair signaal: richting = doorbraak)
  -> 1:1 triple barrier, verticaal 10 dagen, fills op barrièreniveau (§12)
  -> label: netto na de trade-specifieke kosten van §6.2 positief
  -> ~20 causale features                                          (fracdiff, vol, trend, markt, funding, OI)
  -> LR + ondiepe RF, Platt-gekalibreerd op inner walk-forward OOF-voorspellingen, gemiddeld
  -> handelsdrempel uit OOF-voorspellingen; Kelly op de ondergrens van een Beta-posterior op
     GEREALISEERDE OOF-uitkomsten; correlatiecorrectie
  -> RiskEngine (mandaat conf/risk/default.yaml) -> dagdata-executieconventie (§12)
  -> oordeel op bevroren stop-criteria; daarna eenmalig de holdout-rooktest
```

## 4. Data

- Bron: de bestaande gecertificeerde PIT-store (`data/pit_store`), datasets
  `ohlcv` (1d), `funding` (8h, per dag gesommeerd via
  `data/funding_panel.daily_funding_panel`) en `open_interest` (1d), alles
  langs het data-register (`load_certified_series`). Het meetdomein
  (`conf/governance/measurement_domain.yaml`, AD-23) verandert **niet**.
- **De store wordt in Plan 1 niet bijgewerkt.** Hij is append-only met
  jaarpartities (`PitStore.write` crasht op een afwijkende bestaande partitie);
  dagen toevoegen aan 2026 vraagt een wijziging aan het apparaat en hoort bij
  Plan 2. De laatste bar heeft asof 2026-08-23.
- Per symbool begint het venster op de eerste dag met een geldige
  vol-schatting (burn-in 60 bars). Het paneel is ongebalanceerd: een symbool
  telt mee vanaf zijn eerste geldige dag.

## 5. Events en het primaire signaal

- **Symmetrisch CUSUM-filter** op de log-slotkoers (de bestaande directionele
  kernel in `labeling/cusum.py`), met drempel `h_t = k · σ_{t−1}`,
  `σ` = EWMA-dagvolatiliteit (λ = 0,94).
- **Richting** = richting van de doorbraak: `S⁺ > h` → long, `S⁻ < −h` → short.
- **k wordt vastgelegd op frequentie, nooit op rendement:** gekozen uit een vast
  rooster zodat het boek op de data **vóór de eerste testperiode (2022-01-01)**
  gemiddeld zo dicht mogelijk bij **5,5 events per week** zit (gelijke afstand:
  de grotere k). Leest geen labels; kost geen trial. Bevroren in de
  preregistratie.
- **Handelsfrequentie** (1–2 per week) komt niet van k maar van de
  handelsdrempel (§10.3).
- Een symbool met een open positie krijgt geen nieuwe trade tot die dicht is
  (events blijven trainingsdata).

## 6. Labels en kosten

### 6.1 Labels

- `labeling/vol_barriers.py` met zijn instapconventie: event op de close van
  `t`, fill op de close van `t+1`, barrières gemonitord vanaf `t+2`.
- **Barrières 1:1:** winst- en verliesbarrière beide op `b_i = σ_t · √5`
  (≈ één weekvolatiliteit). Eén breedte in de configuratie, dus 1:1 per constructie.
- **Verticale barrière:** 10 bars.
- **Dubbele touch op één dag:** pessimistisch als verlies (bestaand gedrag).
- **Fillrendement** `r_i` volgens de executieconventie van §12.
- **Doel:** `y_i = 1[ r_i − C_i^label > 0 ]` met `C_i^label` uit §6.2(a).
- **Gewichten:** gemiddelde uniqueness.

### 6.2 Eén kostendefinitie

Voor trade `i` met richting `s_i ∈ {−1, +1}`, entrybar `e_i`, exitbar `x_i`,
barrière `b_i` en notioneel `N_i`, in rendementseenheden van het notioneel:

```text
C_i = c_fix + c_fund,i + c_imp,i          (stopslippage zit in de fillprijs r_i, niet in C_i)

c_fix    = 2 · (τ + h)                      τ = taker fee 5,5 bps, h = halve spread 1,0 bps
                                            (conf/execution/fees.yaml) → 13 bps, beide benen
c_fund,i = s_i · Σ_{d = e_i+1}^{x_i} f_d     f_d = funding van bar d (dagsom van de 8h-afrekeningen);
                                            long betaalt positieve funding
c_imp,i  = I(N_i, e_i) + I(N_i, x_i)         I = square_root_impact, ADV = 30-daags gemiddelde
                                            turnover met één bar lag, σ = dag-σ van die bar
```

Dezelfde definitie heeft drie toepassingen, met expliciet wat bekend is:

| Toepassing | Wanneer | Vorm |
|---|---|---|
| (a) **Label** | achteraf, voor het trainingsdoel | `C_i^label = c_fix + c_fund,i` (gerealiseerd) `+ c_imp,i` bij het **referentie-notioneel** `N_ref,i = (baseline_risk_fraction / b_i) · equity`. Het label hangt zo niet af van de eigen sizing van het model. |
| (b) **Ex ante** | bij het besluit op de eventbar `t`, alleen data ≤ `t` | `Ĉ_i = c_fix + c_slip + ĉ_fund,i + ĉ_imp,i`, met `c_slip = 5 bps` (slechtste geval: stop), `ĉ_fund,i = H · max(0, s_i · f̄_i)` met `H = 10` (maximale houdtijd) en `f̄_i` = gemiddelde dagfunding over bars `t−7 … t−1`, en `ĉ_imp,i = 2 · I(N_max, t)` met `N_max = max_position_pct · equity` (0,80 · equity). Dit is een **conservatieve, trade-specifieke bovengrens** die niet van de sizing afhangt. |
| (c) **P&L** | in de backtest, per bar | gerealiseerde fees `τ + h` per been, funding per bar op de gehouden positie, impact per order met het werkelijke notioneel, slippage in de fill. |

`p_be,i` (§10.2), `p_trade,i` (§10.3) en de Kelly-fractie (§10.4) gebruiken
uitsluitend `Ĉ_i`. Voorbeeld: `b = 9 %`, funding 1,5 bps/dag in de richting
van de positie, impact verwaarloosbaar: `Ĉ = 13 + 5 + 15 = 33 bps` →
`p_be = 0,5 + 0,0033 / 0,18 = 51,8 %`.

## 7. Features

Vast gekozen, ~20 stuks, allemaal causaal: berekend op data t/m de eventbar en
met één bar extra lag voor funding en open interest.

| Groep | Features | Bron |
|---|---|---|
| Trend | fracdiff-logprijs; vol-genormaliseerde rendementen 1/5/20 dagen; afstand tot 20-daagse top/bodem in σ | `features/fracdiff.py` |
| Mean-reversion | Kalman-z-score t.o.v. het gefilterde niveau; OU-halfwaardetijd | `alpha/kalman_ou.py`, rollend opnieuw gefit op 126 bars |
| Volatiliteit | EWMA-σ; Yang-Zhang-ratio (20 d t.o.v. 120-daags gemiddelde); vol-of-vol | `volatility/` |
| Markt | PCA-marktfactor (PC1-rendement 5 d); residu-z-score; gemiddelde onderlinge correlatie, rollend 60 d | nieuw |
| Activiteit | dollarvolume t.o.v. 30-daags gemiddelde | nieuw |
| Crypto | funding-z-score (som 3 d, t.o.v. 90 d, lag 1); OI-verandering 1 d en 5 d (lag 1) | `data/funding_panel.py`, PIT-store |
| Event | richting van de doorbraak | CUSUM |

- **Fracdiff:** één `d*` voor alle symbolen, het maximum van de per-symbool
  ADF-minimale `d` op data **vóór 2022-01-01** (symbolen met ≥ 250 bars).
  FFD-gewichtsdrempel uit `conf/model/fracdiff.yaml` (1e-4): bij `d = 0,4` een
  venster van 282 bars in plaats van 1.458 onder de moduledefault 1e-5.
- **Geen GARCH:** een fit vraagt per symbool 250–500 bars burn-in; dat kost SOL
  en AVAX hun eerste 1–1,5 jaar. EWMA (RiskMetrics) is het volatiliteitsmodel
  voor barrières, CUSUM-drempels, sizing en risicolaag.
- **Kalman rollend:** `KalmanOUMeanReversion` houdt μ bewust statisch na
  `fit()`; zonder refit trekt de z-score naar een oud prijsniveau.
- De stationariteitspoort draait op trainingsdata; MDA/SFI alleen als diagnose.

## 8. Model en kalibratie

Licht en anti-overfit, **geen hyperparameter-zoektocht**; alle instellingen
staan vooraf in de preregistratie.

| Model | Instellingen |
|---|---|
| Logistische regressie | L2, `C = 0,1`, gestandaardiseerde features |
| Random forest | 500 bomen, `max_depth = 3`, `max_features = 1`, `max_samples` = gemiddelde uniqueness, `min_samples_leaf = 50`, `class_weight = "balanced_subsample"` |
| Ensemble (primaire cel) | gemiddelde van de twee gekalibreerde kansen |

**Kalibratie op out-of-fold-voorspellingen (inner walk-forward).** Binnen elk
buitenste trainvenster:

1. Sorteer de trainevents op tijd en deel ze in **K = 4 opeenvolgende blokken**
   met gelijk aantal events.
2. Voor `k = 2, 3, 4`: fit het basismodel op blokken `< k`, gepurged (geen event
   waarvan de exit ≥ de eerste eventbar van blok `k`) en met een embargo van
   `H + 1 = 11` bars; voorspel blok `k`. Dat geeft **OOF-scores** voor blokken 2–4.
3. Fit een Platt-kalibrator (2 parameters) op die OOF-scores tegen hun
   gerealiseerde labels, met uniqueness-gewichten.
4. Fit het basismodel opnieuw op **alle** trainevents; de testkans is
   `kalibrator(score van dit model)`.
5. Bewaar de **gekalibreerde OOF-kansen met hun gerealiseerde labels**: die
   voeden de handelsdrempel (§10.3) en de empirische posterior (§10.4).

Dit is dezelfde constructie als `CalibratedClassifierCV(ensemble=False)`, met
gepurgede tijdsplits in plaats van K-fold. Twee eerlijke kanttekeningen: de
kalibrator is in-sample op de OOF-scores (2 parameters), en het eindmodel is op
meer data gefit dan de binnenmodellen, dus zijn scores kunnen iets verschoven
zijn. Beide zijn bekend en worden geaccepteerd; de buitenste walk-forward meet
het resultaat.

## 9. Validatie en governance

### 9.1 Tijdsplitsing en holdout

- Ontwikkeling t/m 2026-06-23; **holdout = de laatste 60 bars** vanaf
  `2026-06-24T00:00:00+00:00`. Het bestaande slot (`holdout_lock.json`) is nooit
  gelezen en wordt vóór de eerste fit opnieuw bevroren op die split. Eén lezing,
  alleen na een ontwikkeloordeel `PASS`.

### 9.2 Trial-budget

- De ledger is leeg. Het programma boekt zijn **4 trials** (referentie,
  logreg, forest, ensemble) als eerste entry en bevriest **`M = 4`** in de
  parameters van de preregistratie; de DSR leest hem met `frozen_trial_count`.
  Geen `freeze_reset`: na een reset eist `active_trial_count` dat de ledger niet
  meer groeit.

### 9.3 Buitenste walk-forward

- Purged walk-forward, expanderend, eerste testperiode vanaf 2022-01-01,
  testblokken van 91 bars (kwartaal), embargo 11 bars, purge op de exitbar.

### 9.4 Negatieve controle (geldigheid)

- **Labelpermutatie:** hetzelfde ensemble op 5 permutaties van het doel; de
  gewogen OOS-AUC moet in **elke** replicatie binnen [0,45; 0,55] liggen. Zo
  niet, dan lekt de pijplijn informatie en is de run **INVALID**.
- De **lookahead-suite** (tests die invoer na `t` wijzigen) is de tweede,
  code-niveau-controle.
- **Het omgekeerde signaal is géén negatieve controle.** Het wordt gemeten en
  gerapporteerd als diagnose (ongefilterd, vaste risicofractie), maar het beslist
  niets: een doorbraakmodel mag ontdekken dat het tegengestelde werkt. Is het
  omgekeerde beter, dan is dat een onderzoeksbevinding voor een nieuwe
  preregistratie, geen ongeldige run.

### 9.5 CPCV en PBO, exact

**CPCV (gerapporteerd, geen poort):**

- Events op tijd gesorteerd, `N = 6` groepen met gelijk aantal events,
  `k = 2` testgroepen → 15 splits en 5 paden.
- Purge op exit in eventruimte (`event_space_t1`), embargo `t_max + 1` events.
- Per split: het ensemble fit op de trainevents (met zijn eigen inner OOF-kalibratie
  en drempel, §8), voorspelt de testgroepen, en het **vaste-risicofractie-boek**
  handelt de testevents.
- Per testgroep `g` telt het rendement op de kalenderdagen
  `[datum eerste event van g, min(datum laatste exit van g, dag vóór het eerste event van groep g+1)]`,
  zodat groepvensters niet overlappen.
- Paden via `build_cpcv_return_paths`; per pad de Sharpe op dagelijkse
  kalenderrendementen inclusief nullen.
- **Rangorde per pad:** per pad de Sharpe van alle vier varianten (ongefilterd,
  logreg, forest, ensemble, alle met vaste risicofractie) en de rang van het
  ensemble (1 = beste). Gerapporteerd: verdeling van de padsharpes (min,
  mediaan, max, aandeel > 0) en de rang van het ensemble per pad.

**PBO (poort G6, ondersteunend):**

- CSCV (`backtest/pbo.compute_pbo`) op de `T × 4`-matrix van dagelijkse
  OOS-rendementen van de vier vaste-risicovarianten op dezelfde kalenderas,
  `S = 16` deelreeksen (12.870 IS/OOS-splitsingen), maatstaf Sharpe; PBO = kans
  dat de IS-beste variant OOS onder de mediaan valt (logit ≤ 0).
- **Beperking, uitdrukkelijk:** met vier varianten neemt de relatieve rang maar
  vier waarden aan; PBO is dus grof. `PBO < 0,25` is ondersteunend bewijs, geen
  precieze overfitmaat. De eigenlijke bescherming tegen overfit is: geen
  zoektocht, vaste hyperparameters, DSR bij `M = 4`, en de holdout.

### 9.6 Inferentie

- Sharpe met Lo(2002)-SE, circulair-blokbootstrapinterval, DSR bij `M = 4`
  (`validation/inference.py`, `validation/dsr.py`), Wilson-interval op het
  trefpercentage. Alle Sharpes op **dagelijkse kalenderrendementen, met nul op
  vlakke dagen**, geannualiseerd met 365.
- Een boek zonder variantie in zijn rendementen (geen trades) krijgt Sharpe 0,
  intervalondergrens 0 en DSR 0: geen trades is geen edge.

### 9.7 Het filter tegen het ongefilterde signaal (poort G2), exact

- **Twee boeken, één verschil:** het ongefilterde boek neemt elk OOS-event; het
  gefilterde boek neemt alleen events met `p ≥ p_trade` (§10.3). Beide met
  **dezelfde vaste risicofractie** per trade (`baseline_risk_fraction` = 1 % van
  het vermogen op de stop) en dezelfde correlatiecorrectie, dezelfde
  kostenmethodiek (§6.2c), dezelfde risicolaag en dezelfde executieconventie
  (§12). Het enige verschil is de selectie.
- **Tijdas:** elke kalenderdag van de eerste tot de laatste OOS-testdag; op een
  dag zonder positie is het rendement **expliciet 0**.
- **Toets:** `sharpe_difference_test(gefilterd, ongefilterd, align="common_valid")`:
  Ledoit-Wolf (2008), gestudentiseerde circulaire blokbootstrap. Poort: de
  ondergrens van het 95 %-interval op ΔSR > 0.
- **Aanvullend gerapporteerd (geen poort): selectie-nul.** De gekalibreerde
  OOS-kansen worden binnen elke buitenste fold 100 keer gepermuteerd over de
  events van die fold; met dezelfde drempels levert dat per fold ongeveer
  hetzelfde aantal trades (niet exact: `p_be` is trade-specifiek), willekeurig
  gekozen. Permutatie-p-waarde = `(1 + #{Sharpe_perm ≥ Sharpe_obs}) / (1 + 100)`,
  met het vaste-risicoboek van het gefilterde signaal.
- Het **Kelly-boek** (§10.4) is de strategie zelf; poorten G3, G4, G5 en G7
  gaan daarover. G2 gaat alleen over de selectie.

### 9.8 De holdout-rooktest, numeriek

Het model wordt gefit op alle ontwikkelevents waarvan het label vóór de split
eindigt, en scoort **alle** holdout-events (~50), niet alleen de trades.

| # | Toets | Faalt als |
|---|---|---|
| H1 | Brier-verslechtering `ΔB = Brier(model) − Brier(basisfrequentie)`, basisfrequentie = het positieve aandeel in de training | `ΔB > 0,01`, **of** de 95 %-ondergrens van ΔB (circulaire blokbootstrap) > 0 |
| H2 | Verschuiving van de kansverdeling (holdout-kansen tegen de gekalibreerde OOS-kansen van de ontwikkelperiode) | `\|gemiddelde(p_holdout) − gemiddelde(p_dev)\| > 0,05`, **of** tweesteekproef-KS-p < 0,01 |
| H3 | Rendement van het Kelly-boek over de 60 holdout-dagen | lager dan het 1e percentiel van alle 60-daagse rollende rendementen van het Kelly-boek in de ontwikkel-OOS |

Alle drie groen → Plan 2. Eén rood → niet naar papertrading.

## 10. Sizing met kansrekening

### 10.1 De nul uit barrièretheorie

Voor een koers als Brownse beweging met drift μ en volatiliteit σ, met
symmetrische barrières op ±b:

```text
P(winstbarrière eerst) = 1 / (1 + exp(−2μb/σ²))
```

Zonder drift is dat **precies 0,5**: de 1:1-barrière geeft een schone nul, en
het enige wat het filter hoeft te vinden is conditionele drift. Met `b = σ√5` en
een horizon van 10 dagen eindigt zonder drift ≈ **10,8 %** van de trades op de
verticale barrière.

### 10.2 Break-even

Met de ex-ante kosten `Ĉ_i` van §6.2(b):

```text
E = p(b − Ĉ) − (1 − p)(b + Ĉ)   =>   p_be,i = ½ + Ĉ_i / (2 b_i)
```

`p_be` is trade-specifiek: funding en impact verschillen per trade.

### 10.3 Handelsdrempel uit out-of-fold-voorspellingen

```text
p_trade,i = max(p_be,i, q_f)
q_f       = het (1 − φ_f)-kwantiel van de gekalibreerde OOF-kansen van het trainvenster van fold f
φ_f       = min(1, (1,5 trades/week) / (OOF-events per week in dat venster))
```

`q_f` leest alleen voorspellingen, geen uitkomsten, en die voorspellingen zijn
**out-of-fold** (§8, stap 2–3): de drempel zelf is dus out-of-sample bepaald en
niet optimistisch door in-sample-kansen.

### 10.4 Kelly op een echte Beta-posterior

Per eenheid op het spel wint een trade `1 − ε` en verliest hij `1 + ε`, met
`ε_i = Ĉ_i / b_i`. Kelly voor een binaire weddenschap:

```text
f*(p) = (p(1 − ε) − (1 − p)(1 + ε)) / ((1 − ε)(1 + ε))  ≈  2p − 1 − ε
```

`f*` is de fractie van het vermogen die verloren gaat als de stop wordt geraakt;
het notioneel is `f · vermogen / b`.

**Empirische onzekerheid, gescheiden van kalibratie.** Kalibratie (§8) bepaalt
de kans `p`; de onzekerheid daarover komt uit **gerealiseerde uitkomsten**:

1. Deel de gekalibreerde OOF-kansen van het trainvenster in **B = 5 bakken met
   gelijk aantal** (grenzen = OOF-kwantielen, alleen trainingsdata).
2. Voor een trade met kans `p` in bak `j`: `s_j` = aantal OOF-events in bak `j`
   met `y = 1`, `n_j` = aantal OOF-events in bak `j`.
3. Posterior op het werkelijke succespercentage van die bak, met uniforme prior:
   `Beta(1 + s_j, 1 + n_j − s_j)`. `p_low` = het **25 %-kwantiel**.
4. `f_i = 0,25 · f*(p_low,j) · 1/(1 + (k − 1)ρ)` (kwart-Kelly en
   correlatiecorrectie). Is `p_low,j ≤ p_be,i`, dan `f_i = 0`: geen trade.

Er wordt **geen** posterior gebouwd uit modelkansen als pseudo-waarnemingen.

**Gecorreleerde weddenschappen:** met k gelijktijdige posities in dezelfde
richting en onderlinge correlatie ρ is de multivariate Kelly-fractie per positie
de enkelvoudige gedeeld door `1 + (k − 1)ρ` (exact bij gelijk-gecorreleerde,
gelijk-renderende bets). `ρ` = gemiddelde onderlinge correlatie op de eventbar.

**Discretisatie:** een bestaande positie wordt alleen herschaald als het
toegestane gewicht meer dan 25 % afwijkt, of naar nul.

### 10.5 Drawdown en ruïne

Onder fractie-Kelly `c` (continu model, juist geschatte edge) is de kans dat het
vermogen, **vanaf een willekeurig moment**, ooit onder `x` maal het niveau van dat
moment zakt `x^(2/c − 1)`: kwart-Kelly `0,75^7 ≈ 13 %` op −25 %, halve Kelly
≈ 42 %. Over een horizon met veel nieuwe pieken ligt de kans op minstens één zo'n
drawdown hoger. **Poort G7:** een Monte Carlo met circulaire blokbootstrap
(bloklengte ≈ twee weken aan trades) van de **gerealiseerde R-multiples** van de
genomen OOS-trades, met de mediane gerealiseerde risicofractie, over het aantal
trades van één jaar: `P(max drawdown ≥ 25 %) ≤ 10 %`. De Kelly-fractie wordt
daarna niet bijgesteld; een andere fractie is een nieuwe preregistratie.

## 11. Risicolaag

- **Mandaat ongewijzigd** (`conf/risk/default.yaml`, policy `9961e1613bc907a5`):
  vol-target 20 %, max drawdown 25 %, dagverlies 10 %, leverage ≤ 4, max positie
  80 %, concentratie 40 %, ADV-participatie 1 %, max positieleeftijd 720 u.
- Elke dag gaat het gewenste boek door `RiskEngine.decide`; de toegestane
  exposures worden de posities.
- **Strategieregels bovenop het mandaat:** exits op de barrières en na 10 bars
  (§12); correlatiecorrectie (§10.4); een positie die de risicolaag naar nul zet,
  sluit op de close van die bar en wordt gescoord op haar werkelijke rendement.
- In Plan 1 zijn stop en target een **gemodelleerde conventie** (§12), geen echte
  exchange-orders. Echte orders bij de exchange, met stops die ook gelden als de
  bot uitvalt, horen bij Plan 2.

## 12. Uitvoering in Plan 1: de dagdata-executieconventie

Met dagdata valt geen geloofwaardige post-only-tegen-taker-fill te
reconstrueren; de backtest volgt daarom één vaste, reproduceerbare conventie:

| Moment | Fill | Kosten |
|---|---|---|
| Entry | close van bar `t+1` (event op `t`) | taker `τ` + halve spread `h`, impact |
| Target | op het barrièreniveau; opent de bar er met een gat voorbij, dan op de open | `τ + h` (geen maker-aanname), impact |
| Stop | op het barrièreniveau, of op de open bij een gat erdoor; in beide gevallen × (1 ∓ 5 bps slippage) tegen de positie in | `τ + h`, impact |
| Dubbele touch op één bar | als stop | als stop |
| Verticaal | close van bar `e + 10` | `τ + h`, impact |
| Risico-exit | close van de bar waarop de risicolaag naar nul zet | `τ + h`, impact |
| Funding | per bar op de positie die de bar in ging; long betaalt positieve funding | §6.2 |

Geen post-only-, maker- of intrabar-logica in Plan 1. `backtest/engine.py` kan
een fill op barrièreniveau niet uitdrukken; een klein tradeboek
(`backtest/barrier_book.py`) voert de conventie uit met dezelfde
`RiskEngine`, dezelfde kostenparameters en `square_root_impact`, bewezen met een
sluitende boekhouding en een causaliteitstest.

## 13. Live-bewaking en uitrol (Plan 2)

**Bevestigen kan live niet; fouten opsporen wel.** Om een trefkans van 0,55
tegen `p_be` = 0,515 te bevestigen (α = 5 % eenzijdig, power 80 %) zijn
≈ **1.257 trades** nodig — bij 78 per jaar ≈ 16 jaar. Het bewijs komt uit §9;
live wordt ingericht om een **kapotte** edge snel te zien:

- **Beta-Binomiaal-posterior op gerealiseerde live-uitkomsten**:
  `Beta(1 + s, 1 + n − s)` met `s` = live-successen en `n` = live-trades,
  gescheiden van de modelkalibratie; handel stopt als `P(p < p̄_be | data) > 0,9`.
- **SPRT (Wald)** tussen H1: `p = p_backtest` en H0: `p = p̄_be`, α = β = 0,1;
  zakt de log-likelihoodratio onder `ln(β / (1 − α))`, dan stopt de handel.
- **Kalibratie en drift:** Brier, ECE (`monitoring/prob_calibration.py`),
  feature- en driftmonitors.

**Uitrol:** (1) alle poorten van §1 op de ontwikkelsample; (2) de
holdout-rooktest (§9.8), één lezing; (3) Plan 2: data bijwerken en
papertrading ≥ 8 weken; (4) live met een klein deel van het kapitaal (de fractie
kiest de eigenaar), opschalen alleen zolang de live-cijfers binnen het
backtest-interval blijven en geen monitor is afgegaan.

## 14. Stopregels (vooraf vastgelegd)

| Uitkomst | Actie | Oordeel |
|---|---|---|
| G1 faalt (labelpermutatie) | run ongeldig; de trials tellen wel | `INVALID` |
| G8: < 100 genomen OOS-trades | te weinig data voor een uitspraak | `UNPROVEN` (gaat vóór `FALSIFIED`) |
| netto Sharpe van het Kelly-boek ≤ 0 | geen edge na kosten | `FALSIFIED` |
| G2, G3-interval, G4, G5, G6 of G7 faalt | niet aangetoond | `UNPROVEN` |
| H1, H2 of H3 faalt | niet naar papertrading | stop |
| Een live-monitor gaat af | handel stopt, terug naar onderzoek | stop |

Voorrang bij meerdere bindende criteria: `INVALID` > `UNPROVEN` (te weinig
data) > `FALSIFIED` > `UNPROVEN`. In geen enkel geval wordt aan parameters
gedraaid om het resultaat te redden; een nieuwe poging is een nieuwe
preregistratie en telt in M.

## 15. Componenten

**Hergebruikt, ongewijzigd:** `labeling/vol_barriers.py`, `features/fracdiff.py`,
`features/stationarity_gate.py`, `volatility/`, `cv/walk_forward.py`,
`cv/cpcv.py`, `cv/uniqueness.py`, `backtest/pbo.py`, `validation/` (inference,
dsr, holdout), `registry/` (ledger, preregistration, trial_counter),
`risk/engine.py`, `execution/impact_model.py`, `monitoring/prob_calibration.py`.

**Aangepast:** `labeling/cusum.py` (publieke directionele filter),
`train/meta_label.py` (generieke purged walk-forward).

**Nieuw:** richting en k-kalibratie; fills en de kostendefinitie van §6.2; de
dagmarkt op één raster; de vaste featureset; de dataset; de lichte modellen met
inner-walk-forward-kalibratie; de binaire kansrekening; het tradeboek; het
oordeel; de campagne; de preregistratie en het bevriezen; de holdout-rooktest.

## 16. Bekende risico's

- **De DSR-drempel is hoog** (≈ 1,2 netto Sharpe op ~5 jaar). `UNPROVEN` is een
  geldig antwoord.
- **Zes munten bewegen grotendeels als één activum**; het paneel geeft minder
  onafhankelijke informatie dan het aantal events suggereert.
- **Burn-in:** fracdiff (~200–400 bars) en de 126-bars-Kalman kosten SOL en AVAX
  hun eerste maanden; per symbool gerapporteerd.
- **Vroege folds:** de inner walk-forward heeft in het eerste trainvenster weinig
  events per blok; de OOF-drempel en de posterior zijn daar het onzekerst — de
  posterior maakt de inzet dan vanzelf kleiner.
- **PBO is grof** bij vier varianten (§9.5).
- **Dagdata en barrières:** de volgorde binnen een dag is onbekend; de
  pessimistische regel kost rendement maar voorkomt optimisme.
- **Een holdout van 60 dagen** is een rooktest, geen bewijs.

## 17. Herzieningen

### 17.1 Bij het uitwerken van het plan (2026-09-26)

1. Data zoals gecertificeerd, niet bijgewerkt (append-only store) — §4.
2. Holdout = de laatste 60 bars vanaf 2026-06-24; ongelezen slot opnieuw bevroren — §9.1.
3. M via de lege ledger in plaats van `freeze_reset` — §9.2.
4. Geen GARCH-feature (burn-in); EWMA is het volatiliteitsmodel — §7.
5. Kalman rollend opnieuw gefit — §7.
6. Exits op barrièreniveau in een klein tradeboek, omdat `backtest/engine.py`
   alleen op barprijzen vult — §12.
7. FFD-drempel uit `conf/model/fracdiff.yaml` (1e-4 in plaats van 1e-5) — §7.
8. Voorrang in het oordeel: te weinig data gaat vóór `FALSIFIED` — §14.

### 17.2 Na de methodologische review van de eigenaar (2026-09-26)

1. **Eén kostendefinitie**, trade-specifiek, met drie expliciete toepassingen
   (label, ex ante, P&L); de ex-ante vorm is een conservatieve bovengrens — §6.2,
   §10.2. (Voorheen: 0,25 % in §10 tegen 13 bps in §12.)
2. **Geen pseudo-posterior meer.** De sizing gebruikt een echte Beta-posterior op
   gerealiseerde OOF-uitkomsten per kansbak, gescheiden van de kalibratie. Live
   geldt hetzelfde op live-uitkomsten — §10.4, §13. (Voorheen:
   `Beta(1 + p̂·n, 1 + (1 − p̂)·n)`, een shrinkageheuristiek.)
3. **Handelsdrempel en kalibratie uit inner-walk-forward OOF-voorspellingen** —
   §8, §10.3. (Voorheen: een enkele chronologische kalibratiesplit.)
4. **Het omgekeerde signaal is geen negatieve controle meer**, alleen diagnose;
   geldigheid komt uit labelpermutatie en de lookahead-suite — §9.4.
5. **Holdout-rooktest numeriek**: H1 Brier, H2 kansverdeling, H3 rendement — §9.8.
6. **Vergelijking gefilterd tegen ongefilterd exact gedefinieerd**: dezelfde
   vaste risicofractie, dezelfde kalenderas met nullen, dezelfde kosten en
   risicolaag; Ledoit-Wolf-toets; selectie-nul gerapporteerd — §9.7.
7. **PBO als grof, ondersteunend** benoemd; CPCV-padconstructie en rangorde per
   pad vastgelegd — §9.5.
8. **Plan 1 / Plan 2** behouden, met een expliciete dagdata-executieconventie in
   Plan 1 (geen post-only/taker-logica) — §2, §11, §12.
