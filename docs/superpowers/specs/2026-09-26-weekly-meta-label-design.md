# Ontwerp: wekelijkse ML-meta-label-strategie op dagdata

> **Status:** ontwerp, goedgekeurd per sectie door de eigenaar op 2026-09-26.
> **Vervolg:** een implementatieplan (writing-plans) op basis van dit document.
> **Uitgangspunt:** de schone lei van commit `4a9588b`. Er is geen eerdere
> hypothese waarop dit ontwerp leunt; de validatie hieronder beslist opnieuw.

---

## 1. Doel en wat "winstgevend" hier betekent

Eén model, geen sleeves, dat **1 à 2 keer per week** een trade neemt op zes
crypto-perpetuals, met een symmetrische **1:1 triple barrier**, een licht
ML-filter, een volatiliteitsmodel, fracdiff-features en een risicolaag die op
kansrekening is gebouwd.

"Haalbaar winstgevend" is hier een meetbare uitspraak, geen belofte:

| Poort | Eis |
|---|---|
| ML voegt iets toe | netto Sharpe van het gefilterde boek > die van het ongefilterde primaire signaal (gepaarde toets) |
| Edge na kosten | ondergrens 95%-interval op de netto Sharpe > 0 |
| Selectie-eerlijkheid | DSR ≥ 0,95 bij het bevroren budget `M_new = 4` — op ~5 jaar dagdata vraagt dat een netto Sharpe van **≈ 1,2** (gemeten 2026-09-26 met `backtest.metrics.deflated_sharpe`: 1,199 bij M=4, n_obs=1850) |
| Trefkans | ondergrens 95%-interval op het trefpercentage > break-even `p_be` (§10.2) |
| Overfit | PBO < 0,25 (CPCV) |

Haalt het de poorten niet, dan is dat het antwoord (§14).

## 2. Scope

**Binnen:** dagdata (OHLCV, funding, open interest) van BTC, ETH, SOL, AVAX,
LINK en DOT; één gepoold meta-label-model; bet sizing, risicolaag,
papertrading en een klein-kapitaal live-traject.

**Buiten:** intraday-data en dollarbars (de eigenaar koos voor 1d-data; uit
dagdata worden dollarbars meerdaags en halveert het aantal leervoorbeelden),
sleeves of meerdere strategieën naast elkaar, hyperparameter-zoektochten
(Optuna), diepe of zwaar getunede modellen, wijziging van het risicomandaat.

## 3. Pijplijn in één oogopslag

```text
PIT-store (1d OHLCV, funding 8h -> dag, OI 1d)
  -> CUSUM-events op log-close, drempel h = k * sigma_t      (primair signaal: richting = doorbraak)
  -> 1:1 triple barrier (vol_barriers), verticaal 10 dagen   (label: doorzetten ja/nee)
  -> ~20 causale features op dag t-1                         (fracdiff, vol, trend, markt, funding, OI)
  -> LR + ondiepe RF, Platt-gekalibreerd, gemiddeld          (kans p dat de doorbraak doorzet)
  -> handelsdrempel p_trade, Kelly-sizing op posterior-ondergrens, correlatiecorrectie
  -> RiskEngine (mandaat conf/risk/default.yaml) -> orders met exchange-side stop/target
  -> live-bewaking: Beta-posterior, SPRT, kalibratie- en driftmonitors
```

## 4. Data

- Bron: de bestaande gecertificeerde PIT-store (`data/pit_store`), datasets
  `ohlcv` (1d), `funding` (8h, per dag gesommeerd via
  `data/funding_panel.daily_funding_panel`) en open interest (1d). Het
  meetdomein (`conf/governance/measurement_domain.yaml`, AD-23) verandert
  **niet**.
- Stap vooraf: de store bijwerken tot de meetdatum en de `data_hashes.json`
  bijwerken via de bestaande ingestieroute; niets wordt met de hand gewijzigd.
- Per munt begint het venster op de eerste dag met een geldige vol-schatting
  (burn-in 60 bars). Het paneel is ongebalanceerd: een munt telt mee vanaf
  zijn eerste geldige dag.

## 5. Events en het primaire signaal

- **Symmetrisch CUSUM-filter** (`labeling/cusum.py`) op de log-slotkoers, met
  drempel `h_t = k · σ_t`, `σ_t` = EWMA-dagvolatiliteit (λ = 0,94) uit data
  tot en met `t−1`.
- **Richting** = richting van de doorbraak: `S⁺ > h` → long, `S⁻ < −h` → short.
- **k wordt vastgelegd op frequentie, nooit op rendement:** op de
  ontwikkelsample zo gekozen dat het boek gemiddeld **5 à 6 events per week**
  geeft (training; ~1.200+ events). Die kalibratie leest geen labels en kost
  daarom geen trial. De waarde wordt in de preregistratie bevroren.
- **Handelsfrequentie** (1–2 per week) komt niet van k maar van de
  handelsdrempel `p_trade` (§10.3): het model traint op alle doorbraken, je
  handelt alleen de beste.
- Een munt met een open positie genereert geen nieuwe trade tot die positie
  dicht is (events blijven wel trainingsdata).

## 6. Labels

- `labeling/vol_barriers.py` (voorheen `phase6_barriers.py`), met zijn
  bestaande instapvertraging (instap op de eerstvolgende bar na het event).
- **Barrières 1:1:** winst- en verliesbarrière beide op `b = σ_t · √5`
  (≈ één weekvolatiliteit), in de richting van het primaire signaal.
- **Verticale barrière:** 10 kalenderdagen.
- **Label:** 1 als de winstbarrière eerst wordt geraakt; 0 bij de
  verliesbarrière; bij de verticale barrière 1 alleen als het netto rendement
  na kosten > 0.
- **Dubbele touch op één dag** (beide barrières binnen de dagrange, volgorde
  onbekend op dagdata): **pessimistisch als verlies**. De frequentie hiervan
  wordt gerapporteerd.
- **Gewichten:** gemiddelde uniqueness (`cv/uniqueness.py`), zodat
  overlappende labels niet dubbel tellen.

## 7. Features

Vast gekozen, ~20 stuks, allemaal causaal: berekend op data tot en met de
eventdag en met ten minste één bar lag waar de bron dat vraagt (funding, OI).

| Groep | Features | Bron |
|---|---|---|
| Trend | fracdiff-logprijs (d* per munt, per trainfold gefit met `MinFracDiff`); Kalman-trendhelling; vol-genormaliseerde rendementen 1/5/20 dagen; afstand tot 20-daagse top/bodem in σ | `features/fracdiff.py`, `alpha/kalman_ou.py`, nieuw: dunne wrappers |
| Mean-reversion | z-score t.o.v. het Kalman-niveau; OU-halfwaardetijd (rollend) | `alpha/kalman_ou.py`, `alpha/mean_reversion.py` |
| Volatiliteit | EWMA-σ; GARCH(1,1)-voorspelling; verhouding GARCH/realized; Yang-Zhang; vol-of-vol | `volatility/` |
| Markt | PCA-marktfactor (PC1-rendement 5d); residu-z-score van de munt; gemiddelde onderlinge correlatie 60d | nieuw (PCA per trainfold gefit) |
| Activiteit | dollarvolume t.o.v. 30-daags gemiddelde | nieuw |
| Crypto | funding-z-score (lag 1, som laatste 3 dagen); OI-verandering 1d en 5d | `data/funding_panel.py`, `data/open_interest.py` |
| Event | grootte van de CUSUM-doorbraak in σ; richting | CUSUM-uitvoer |

- Alles wat gefit wordt (d*, PCA, schaalparameters, GARCH) wordt **per
  trainfold** gefit en nooit op testdata.
- De stationariteitspoort (`features/stationarity_gate.py`) draait op
  trainingsdata; een feature die hem niet haalt, gaat eruit — vooraf
  vastgelegd, geen handmatige keuze.
- MDA/SFI (`selection/`) worden **gerapporteerd als diagnose**, niet gebruikt
  voor selectie: selectie op uitkomst is een vrijheidsgraad.

## 8. Model

Licht en anti-overfit, **geen hyperparameter-zoektocht**; alle instellingen
staan vooraf in de preregistratie.

| Model | Instellingen |
|---|---|
| Logistische regressie | L2, `C = 0.1`, gestandaardiseerde features |
| Random forest | 500 bomen, `max_depth = 3`, `max_features = 1`, `max_samples` = gemiddelde uniqueness, `min_samples_leaf = 50`, `class_weight = "balanced_subsample"` |
| Ensemble (primaire cel) | gemiddelde van de twee gekalibreerde kansen |

- **Kalibratie:** Platt-schaling op een binnenste, gepurgede split van het
  trainvenster (`train/calibration.py`). Geen isotone kalibratie: die overfit
  bij een paar honderd events.
- `train/meta_label.py` levert de dataset, de purge en de walk-forward; de
  CatBoost-aanroep in `fit_secondary_model` wordt vervangen door een
  modelfabriek voor bovenstaande twee.

## 9. Validatie en governance

- **Holdout: de laatste 2 maanden** vóór de meetdatum. Het bestaande slot
  (`artefacts/governance/holdout_lock.json`) is nooit gelezen (`reads: []`) en
  wordt vóór de eerste fit opnieuw bevroren op de nieuwe splitsdatum.
  Twee maanden geven 4–16 trades — te weinig voor bewijs (bij 8 trades is de
  standaardfout op het trefpercentage ≈ 0,18). De holdout is daarom een
  **rooktest**: het model scoort er **alle** events (~50) en de Brier-score
  mag niet aantoonbaar slechter zijn dan die van de basisfrequentie; het
  resultaat mag niet catastrofaal afwijken van de backtestverdeling. Eén
  lezing, geregistreerd.
- **Trial-budget:** vóór de eerste fit wordt een nieuwe reset bevroren
  (`registry/ledger_reset.freeze_reset`) met **`M_new = 4`**: referentie
  (ongefilterd), LR, RF, ensemble. Primaire cel: het ensemble.
- **Preregistratie** (`registry/preregistration.py`), bevroren vóór de eerste
  fit, met k, de barrières, de features, de modelinstellingen, de
  poortcriteria uit §1 en de stopregels uit §14.
- **Purged walk-forward met embargo:** expanderend venster, elk kwartaal
  hertrainen, test = het volgende kwartaal, embargo = labelhorizon (10 dagen).
- **CPCV** (6 groepen, 2 testgroepen) voor de PBO-verdeling.
- **Negatieve controles, verplicht:**
  1. geschudde labels → OOS-AUC binnen [0,45; 0,55];
  2. een lekvariant die de eigen bar leest moet door de lookahead-suite rood
     worden gemaakt;
  3. het omgekeerde primaire signaal moet slechter zijn dan het echte.
- **Inferentie:** Sharpe met Lo-SE, block-bootstrap-interval,
  datum-geclusterde standaardfouten over het paneel, DSR bij `M_new = 4`
  (`validation/inference.py`, `validation/dsr.py`).

## 10. Sizing met kansrekening

### 10.1 De nul uit barrièretheorie

Voor een koers als Brownse beweging met drift μ en volatiliteit σ, met
symmetrische barrières op ±b:

```text
P(winstbarrière eerst) = 1 / (1 + exp(−2μb/σ²))
```

Zonder drift is dat **precies 0,5**: de 1:1 barrier geeft een schone nul, en
het enige wat het filter hoeft te vinden is conditionele drift. Met
`b = σ√5` en een horizon van 10 dagen eindigt zonder drift ≈ **10,8%** van
de trades op de verticale barrière (reeksformule voor
`P(sup|W| < a)`, `a = √0,5`), wat de labels grotendeels binair houdt.

### 10.2 Break-even

Met barrière `b` en kosten `c` per round trip (fees + spread + funding over de
houdtijd + impact, in rendementseenheden):

```text
E = p(b − c) − (1 − p)(b + c)   =>   p_be = ½ + c / (2b)
```

Bij b ≈ 9% en c ≈ 0,25% is `p_be` ≈ 51,4%.

### 10.3 Wanneer handelen

`p_trade = max(p_be + marge, q)`, waarbij `q` het kwantiel van de
voorspelde kansen in het trainvenster is dat ~1,5 trade per week oplevert.
Die keuze leest alleen voorspellingen, geen uitkomsten, en wordt per
walk-forward-stap herberekend.

### 10.4 Kelly bij 1:1-uitbetaling

Per eenheid op het spel wint een trade `1 − ε` en verliest hij `1 + ε`, met
`ε = c / b`. Kelly voor een binaire weddenschap:

```text
f* = (p(1 − ε) − (1 − p)(1 + ε)) / ((1 − ε)(1 + ε))  ≈  2p − 1 − ε
```

`f*` is de fractie van het vermogen die verloren gaat als de stop wordt
geraakt; het notioneel is `f · vermogen / b`.

- **Schattingsfout:** p komt niet uit de puntschatting maar uit het
  **25%-kwantiel van `Beta(p̂ · n, (1 − p̂) · n)`**, met `p̂` de gekalibreerde
  kans en `n` het aantal kalibratie-events in dezelfde kansklasse (deciel).
  Weinig events in die klasse → breder interval → kleinere inzet.
- **Fractie:** **kwart-Kelly.** Voorbeeld: p = 0,56 en ε = 0,014 geeft
  f* ≈ 0,106, dus ≈ 2,6% van het vermogen op het spel.
- **Gecorreleerde weddenschappen:** met k gelijktijdige posities in dezelfde
  richting, onderlinge correlatie ρ, is de multivariate Kelly-fractie per
  positie de enkelvoudige gedeeld door `1 + (k − 1)ρ` (exact bij
  gelijk-gecorreleerde, gelijk-renderende bets: `Σ⁻¹1` met
  `Σ = σ²((1−ρ)I + ρ11ᵀ)`). Bij ρ ≈ 0,75 en k = 3 is dat een factor 2,5.
- **Discretisatie:** posities in vaste stappen om onnodige omzet te vermijden.

### 10.5 Drawdown en ruïne

Onder fractie-Kelly `c` (in een continu model met juist geschatte edge) is de
kans dat het vermogen, **vanaf een willekeurig moment**, ooit onder `x` maal
het niveau van dat moment zakt:

```text
P = x^(2/c − 1)
```

Kwart-Kelly: `0,75^7 ≈ 13%` op ooit −25% (de mandaatgrens) vanaf een gegeven
moment; halve Kelly: `0,75^3 ≈ 42%`. Over een lange horizon met veel nieuwe
pieken is de kans op minstens één zo'n drawdown hoger; een **Monte Carlo** met
de echte tradeverdeling, de correlatie en de schattingsfout in p rekent die
kans over 1 en 3 jaar uit, en de Kelly-fractie wordt verlaagd als die kans
boven 10% (1 jaar) uitkomt.

## 11. Risicolaag

- **Mandaat ongewijzigd** (`conf/risk/default.yaml`, policy `9961e1613bc907a5`):
  vol-target 20%, max drawdown 25%, dagverlies 10%, leverage ≤ 4, max positie
  80%, concentratie 40%, ADV-participatie 1%, max positieleeftijd 720 u.
- **Strategieregels bovenop het mandaat:**
  - stop en target staan bij de exchange op de barrières (verlies begrensd,
    ook als de bot uitvalt);
  - sluiten op de verticale barrière (10 dagen);
  - correlatiecorrectie uit §10.4 op gelijkgerichte posities;
  - gap-risico: `risk/kelly.gap_risk_kelly_size` begrenst de positie op een
    scenario waarin de stop door een gat heen slipt.
- Kill switches en circuit breakers: de bestaande `risk/kill_switches.py` en
  `live/circuit_breaker.py`.

## 12. Uitvoering

- Instap met post-only limietorders; niet gevuld binnen de time-out → taker.
- Kostenmodel in de backtest: 13 bps vast per round trip
  (`conf/execution/fees.yaml`) plus funding over de houdtijd plus impact
  (`execution/`).
- TCA (`tca/`) meet de werkelijke kosten tegen het model; een structureel
  verschil gaat naar de bewaking (§13).

## 13. Live-bewaking en uitrol

**Bevestigen kan live niet; fouten opsporen wel.** Om een trefkans van 0,55
tegen `p_be` = 0,515 te bevestigen (α = 5% eenzijdig, power 80%) zijn
≈ **1.257 trades** nodig — bij 78 per jaar ≈ 16 jaar. Het bewijs komt dus uit
§9; live wordt ingericht om een **kapotte** edge snel te zien:

- **Beta-Binomiaal-posterior** op het live-trefpercentage, met de backtest als
  prior (gedempt tot het gewicht van ~20 trades); handel stopt als
  `P(p < p_be | data) > 0,9`.
- **SPRT (Wald)** tussen H1: `p = p_backtest` en H0: `p = p_be`, met α = β =
  0,1. Zakt de log-likelihoodratio onder de ondergrens `ln(β / (1 − α))`, dan
  wordt H0 aangenomen (geen edge) en stopt de handel; komt hij boven
  `ln((1 − β) / α)`, dan wordt de toets herstart (blijven bewaken).
- **Kalibratie:** Brier-score en betrouwbaarheid
  (`monitoring/prob_calibration.py`), plus feature- en driftmonitors
  (`monitoring/drift.py`, `monitoring/feature_health.py`).

**Uitrol:**
1. Onderzoek haalt alle poorten van §1 op de ontwikkelsample.
2. Holdout-rooktest (§9), één lezing.
3. Papertrading ≥ 8 weken (`compliance/shadow_trader.py`, `oms/paper_oms.py`).
4. Live met een klein deel van het kapitaal (de fractie kiest de eigenaar);
   opschalen alleen zolang de live-cijfers binnen het backtest-interval
   blijven en geen monitor uit §13 is afgegaan.

## 14. Stopregels (vooraf vastgelegd)

| Uitkomst | Actie |
|---|---|
| Een negatieve controle faalt | run ongeldig, geen oordeel; trials tellen wel |
| Gefilterd ≤ ongefilterd na kosten | het filter voegt niets toe: stoppen, vastleggen |
| DSR < 0,95 of Sharpe-interval omvat 0 | `UNPROVEN`: stoppen, holdout niet lezen |
| Trefkans-ondergrens ≤ `p_be` | geen edge na kosten: stoppen |
| PBO ≥ 0,25 | overfit-risico te hoog: stoppen |
| Holdout-rooktest faalt | niet naar papertrading |
| Live-monitor gaat af | handel stopt, terug naar onderzoek |

In geen van deze gevallen wordt aan parameters gedraaid om het resultaat te
redden; een nieuwe poging is een nieuwe preregistratie en telt in M.

## 15. Componenten

**Hergebruikt, ongewijzigd:** `labeling/cusum.py`, `labeling/vol_barriers.py`,
`features/fracdiff.py`, `features/stationarity_gate.py`, `volatility/`,
`cv/`, `selection/`, `train/calibration.py`, `validation/` (inference, dsr,
gates, holdout), `registry/` (ledger, preregistration, ledger_reset,
trial_budget), `risk/` (engine, kelly, kill_switches), `execution/`, `tca/`,
`monitoring/`, `oms/`, `live/`.

**Aangepast:** `train/meta_label.py` (modelfabriek i.p.v. CatBoost).

**Nieuw** (grenzen worden in het implementatieplan vastgelegd):
- k-kalibratie op eventfrequentie (bij de CUSUM-code);
- de vaste featureset en de nieuwe features (PCA-markt, activiteit,
  Kalman/OU-wrappers);
- bet sizing met posterior-ondergrens, correlatiecorrectie en
  drawdown-Monte Carlo (bij `risk/`);
- SPRT- en Beta-posteriormonitor (bij `monitoring/`);
- een onderzoeksrunner en een dagelijkse beslisrunner voor deze strategie.

## 16. Bekende risico's

- **De DSR-drempel is hoog** (≈ 1,2 netto Sharpe op ~5 jaar). Een eerlijke
  uitkomst kan `UNPROVEN` zijn; dat is een geldig antwoord.
- **Zes munten bewegen grotendeels als één activum** (hoge onderlinge
  correlatie); het paneel geeft minder onafhankelijke informatie dan het
  aantal events suggereert. De geclusterde standaardfouten en de
  correlatiecorrectie in de sizing vangen dat op, maar vergroten de
  onzekerheid.
- **Dagdata en barrières:** de volgorde binnen een dag is onbekend; de
  pessimistische regel kost rendement in de backtest maar voorkomt een
  optimistische vertekening.
- **Een holdout van 2 maanden** is een rooktest, geen bewijs (§9).

## 17. Wijzigingen bij het uitwerken (2026-09-26)

De implementatieplanning (`docs/superpowers/plans/2026-09-26-weekly-meta-label-research.md`)
liep op zeven punten tegen de werkelijkheid van de repository aan. Deze sectie gaat
voor waar zij §4, §7, §9, §10.4 of §12 tegenspreekt.

1. **Data zoals gecertificeerd, niet bijgewerkt.** De PIT-store is append-only met
   jaarpartities (`PitStore.write` crasht bij een afwijkende bestaande partitie), dus
   dagen toevoegen aan 2026 vergt een wijziging aan het apparaat. Plan 1 gebruikt de
   store zoals hij is (laatste bar: asof 2026-08-23). Bijwerken hoort bij Plan 2 (live).
2. **Holdout = de laatste 60 bars:** split `2026-06-24T00:00:00+00:00`. Het huidige slot
   is nooit gelezen (`reads: []`) en wordt opnieuw bevroren.
3. **M via de lege ledger, niet via `freeze_reset`.** Na een reset eist
   `active_trial_count` dat de ledger niet meer groeit; een schone ledger telt zelf. Het
   programma boekt zijn 4 trials als eerste entry en bevriest `M = 4` in de
   preregistratie; de DSR leest hem met `frozen_trial_count`.
4. **GARCH valt uit de features.** Een GARCH-fit vraagt per symbool 250-500 bars burn-in
   (`conf/model/adequacy.yaml`); dat kost SOL en AVAX hun eerste 1-1,5 jaar. EWMA
   (RiskMetrics) blijft het volatiliteitsmodel voor barrières, CUSUM-drempels, sizing en
   risicolaag. Het vol-regime komt uit de Yang-Zhang-ratio en vol-of-vol.
5. **Kalman rollend opnieuw gefit.** `KalmanOUMeanReversion` houdt mu bewust statisch na
   `fit()`; zonder rollende refit trekt de z-score naar een oud prijsniveau. De
   OU-halfwaardetijd komt uit dezelfde fit (`current_halflife`).
6. **Exits op het barrièreniveau, in een klein tradeboek.** `backtest/engine.py` vult
   alleen op barprijzen en kan een stop bij de exchange niet uitdrukken.
   `backtest/barrier_book.py` gebruikt dezelfde `RiskEngine.decide`, dezelfde
   kostenparameters en `square_root_impact`, en wordt bewezen met een boekhoudidentiteit
   en een causaliteitstest.
7. **Posterior met uniforme prior:** `Beta(1 + p_hat*n, 1 + (1 - p_hat)*n)`, zodat hij
   ook bij `n = 0` gedefinieerd is (dan geen informatie, dus geen inzet boven break-even).

**Opsplitsing:** dit is Plan 1 (onderzoek tot en met oordeel en holdout-rooktest). Plan 2
(data bijwerken, papertrading, SPRT- en Beta-bewaking, orders met stops bij de exchange)
wordt pas geschreven na een `PASS`.
