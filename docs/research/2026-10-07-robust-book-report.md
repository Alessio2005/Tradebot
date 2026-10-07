# Robuust boek v1 — audit, onderzoek en oordeel (2026-10-07)

Preregistratie `eecdb3a109bf10972f17acd3b0146716` · ontwerp
`docs/superpowers/specs/2026-10-07-robust-book-design.md` · artefacten
`artefacts/research/robust_book_v1/` · ledger-entries `robust_book_v1_programme` (M = 7)
en `robust_book_v1_verdict` (oordeel, 0 trials).

## Samenvatting

**De doelstelling (netto Sharpe > 1,0 én netto CAGR > 15 %) is met de data in deze repository
niet robuust haalbaar, en dit programma heeft haar niet gehaald.** Van vijf vooraf vastgelegde
kandidaten haalt er geen de poorten. De meest robuuste kandidaat is **trend long-flat (C2)**:

| C2 trend long-flat | Sharpe netto | CAGR | vol | max DD |
| --- | --- | --- | --- | --- |
| W_DEV (screening, uitvoering direct na de close) | 0,71 (SE 0,55) | 7,0 % | 10 % | 13 % |
| W_DEV (**authoritative engine**, vulling een dag later) | **0,39** (SE 0,54) | 3,1 % | 9 % | 14 % |
| Poortsample 2025-09 → 2026-08 (één lezing, bear-markt) | −0,10 (SE 1,20) | −0,8 % | 6 % | 10 % |
| Ter vergelijking: BTC aanhouden op dezelfde poort | −0,70 | −33,7 % | — | 54 % |

C2 is robuust in vorm: hij houdt 0,6–0,8 over elke parameterverstoring en elke weggelaten munt,
dezelfde Sharpe in beide helften en kapitaalbehoud in een bear-markt. Het **niveau** is echter
te laag. Een eerlijke verwachting voor live is een Sharpe van ≈ 0,25–0,35. De Sharpe-claims
boven de 2 uit 2026-05 waren niet reproduceerbaar en (deels) met hefboom opgeblazen; zie §A.9.

**De bottleneck is data, niet machinerie.** Zes munten correleren gemiddeld 0,73: dat is ≈ 1,3
onafhankelijke weddenschap. Elke markt-datahost is in deze omgeving geblokkeerd (403). Wat het
antwoord wél kan veranderen, staat in §N.

---

## A. Audit van de repository

### A.1 Architectuur

* 71.600 regels in `src/tradebot` (≈ 30 packages) en 29.800 regels tests. De lagen zijn: PIT-datastore
  met certificering (`data/`), features, labeling, ML (`train/`, `tune/`), allocators
  (`portfolio/`), een soevereine `RiskEngine`, executie/TCA, een authoritative event-driven engine
  (`backtest/engine.py`) en live/OMS/monitoring/compliance.
* De governance is sterk en werkt: preregistratie met hash-ID's, een append-only
  hypothese-ledger, een holdout-slot met één lezing per hypothese, DSR, PBO, SPA, CPCV en een
  lookahead-suite met negatieve controles.
* Live, OMS, compliance (champion/challenger, MRM, shadow) en k8s zijn volgens de eigen documenten
  nooit op een echte kandidaat gedraaid.

### A.2 Strategieën die er zijn

* **`main`**: de wekelijkse meta-label-strategie (CUSUM-doorbraak, LR/RF-filter, 1:1-barrières,
  binaire Kelly). De meting staat op de ongemergde branch `jolly-feynman` (commit `a1f4ada`):
  OOS-AUC 0,49, gefilterd boek netto Sharpe −0,78, oordeel INVALID.
* **`elegant-dijkstra`** (ongemergd): het trend/carry-programma, M = 47. Geen enkele kandidaat
  haalt de poorten; de beste eerlijke trendschatting is een Sharpe van 0,3–0,5.
* **`jolly-feynman`** (ongemergd): een BTC-MA100-trendregel die de holdout 2026-06-24 →
  2026-08-23 al één keer heeft gelezen.
* **Nieuw** (deze PR): het robuuste boek, `src/tradebot/systematic/`.

### A.3 Data, frequentie en universum

* Zes Bybit-perpetuals (BTC, ETH, SOL, AVAX, LINK, DOT): dagbars 2020-03-26 → 2026-08-23,
  8h-funding en dagelijkse open interest, gecertificeerd per hash.
* Er is geen spot, geen intraday, geen orderboek en geen tweede activaklasse.
* Survivorship: de zes munten zijn de overlevers van nu (geen LUNA, FTT, ...). Dat bevoordeelt
  long-beta.
* Funding is een grote post: een long-perp betaalde gemiddeld 5–19 % per jaar (2021: 24–70 %;
  2025–26: 2–5 %).

### A.4 Executie, kosten, slippage, hefboom en herbalancering

* Kosten: taker 5,5 bp plus een **aangenomen** halve spread van 1,0 bp
  (`cost_assumption_is_provisional: true`).
* Impact: `η · σ · √(order/ADV)` met η = 2,99. Dat is **geen schatting** maar een bovengrens uit
  de dagrange (`IMPACT_UNCALIBRATED`). Voor kleine orders in DOT, LINK en AVAX levert dat tientallen
  bp per trade op, wat vrijwel zeker te hoog is.
* Timing: de authoritative engine vult op de close van *t+1* (`latency_bars = 1`). In deze data
  geldt `open(t+1) == close(t)` op elke bar en de markt is 24/7. De engine is daarmee een volle dag
  pessimistischer dan een uitvoering direct na de close. Voor een dagelijks trendboek kost dat
  ≈ 0,2 Sharpe (§F).
* Risicobeleid: σ-doel 20 %, bruto ≤ 4, concentratie ≤ max(0,40; 1/n), drawdown-breakers op
  12 %/18 % en een halt op 25 %.
* **Verborgen mismatch:** `live/circuit_breaker.py` sluit posities ouder dan 720 uur (30 dagen).
  De backtest kent die regel niet, en een trendboek houdt posities vaak maanden aan.

### A.5 Metriek, en of de Sharpe correct berekend wordt

* `backtest/metrics.py::sharpe_ratio` deelt het rekenkundig gemiddelde door de sd en annualiseert
  met √365. `validation/inference.py::sharpe_with_se` geeft de Lo (2002)-SE met
  Newey-West-HAC-opslag. CAGR is geometrisch; de max drawdown wordt op de equitycurve berekend.
* **Correct.** Ik heb dit onafhankelijk nagerekend: BTC-perp aanhouden over W_DEV geeft uit de
  ruwe parquet met kale pandas SR 0,390 en CAGR 6,8 %. Dat is identiek aan de programma-uitkomst.
* Er bestond een historische fout: een default `bars_per_year = 8760` blies Sharpes met √24 op.
  Die is in fase 10 hersteld.

### A.6 Train/test-splits en backtest-methodologie

* Er zijn meerdere holdout-definities over branches: de gate-split 2025-09-05 op `main`, de
  weekly-holdout 2026-06-24 en de validatie 2024+ op `elegant-dijkstra`.
* Sommige families hebben vensters al gelezen die op `main` als ongelezen geregistreerd staan.
  Zo las de trendfamilie 2024+ en 2026-06-24+. Dit is gedocumenteerd in het poortartefact.

### A.7 Tests

* De snelle suite heeft op de schone HEAD één falende test, die er al was:
  `test_regime_conditioning.py::test_the_multiplier_is_causal[hmm3-diag-student_t]`
  (HMM-niet-convergentie).
* Deze PR voegt 43 tests toe; die zijn groen.

### A.8 Leakage, look-ahead, survivorship en overfitting

* De funding-boeking is causaal: venster `[T−1d, T)` per bar.
* Het nieuwe boek is getoetst met verstoring van de toekomst, per sleeve, voor de combinatie en
  voor de regimemaskers. Een lekkende negatieve controle wordt rood.
* Overfitting is het grootste historische risico: > 2.776 trials op deze data vóór de reset.
* De ledger is op 2026-09-26 op nul gezet. Dat is formeel netjes (archief bevroren), maar de
  zoekgeschiedenis op dezelfde zes munten verdwijnt daarmee uit M.
* Op `jolly-feynman` zijn de DSR en het trialaantal **achteraf** uit de poorten gehaald (`8c6dc6a`).
  Dat is precies de zet die preregistratie moet uitsluiten.

### A.9 De oude Sharpe > 2

De claims uit mei 2026 (2,77–2,92) en de "headline" 2,70 / 4,31 komen uit code en rapporten die
nooit in git hebben gestaan. Ze zijn volgens de eigen fase-9-audit niet reproduceerbaar, en de 4,31
kwam uit verhoogde hefboom (vol-doel 0,12 → 0,25, leverage 4 → 6, Kelly 0,25 → 0,40). Elke latere
herleiding met volledige herkomst gaf een netto Sharpe tussen −0,48 en +0,27.

## B. Problemen, gerangschikt op impact

1. **Breedte van de data.** Een effectieve N van ≈ 1,3 en 4–5 jaar geven een SE van de Sharpe van
   ≈ 0,5. Een Sharpe van 1 is hier niet aantoonbaar en, gezien > 2.800 eerdere pogingen, ook niet
   aannemelijk.
2. **Geen signaal met voldoende niveau.** Trend, beta, carry en ML (AUC 0,49) geven elk netto 0,3–0,7.
3. **Executietiming.** Lag 1 → lag 2 kost ≈ 0,2 Sharpe. De juiste waarde ligt ertussen en is zonder
   intraday-data niet te kalibreren.
4. **Kostenmodel niet gekalibreerd.** De spread is aangenomen en de impact is een bovengrens. Dat
   bepaalt het oordeel over carry (impact 9 %/jaar bij $100k) en over de combinatie.
5. **Funding-drag op long-perps** (≈ 1,6 %/jaar op C2, 3 %/jaar op C3), te vermijden met spot.
6. **Survivorship** in het universum: zes overlevers.
7. **Governance-drift**: een ledger-reset, een achteraf verwijderde DSR-poort, holdout-vensters die
   over branches worden hergebruikt, en onderzoek dat niet gemerged is.
8. **Complexiteit**: 71k regels, met live/OMS/compliance/k8s voor een systeem zonder edge.
9. **Live ≠ backtest**: de 30-dagenregel voor posities, en de engine-latency tegenover live-executie.

**Behouden:** PIT-store en certificering, preregistratie, ledger en holdout-slot,
`validation/inference.py`, DSR/PBO, de authoritative engine met soevereine risico-laag, de
lookahead-suite en `systematic/` (nieuw).

**Bevriezen of verwijderen** tot er een kandidaat is die de poorten haalt: de ML-stack van de
wave-era (`train/`, `tune/`, catboost, bandits), `live/`, `oms/`, `compliance/`, `infra/k8s`,
`featurestore/` en de HMM-regimes. In deze PR is niets verwijderd; dat is een besluit voor de
eigenaar.

## C. Doelarchitectuur

Ik heb niet één signaal ontworpen maar een **ontleding in premies** met verschillende economische
bronnen en regimeprofielen (zie het ontwerpdocument):

| Sleeve | Economische bron | Diversificatie (correlatie dagrendementen op W_DEV) |
| --- | --- | --- |
| C3 vol-gemanaged kern (BTC+ETH) | marktpremie + vol-clustering | 0,93 met BTC |
| C1/C2 trend (log-raster 5–160 d) | traag informatieverwerken, kuddegedrag; convex in crashes | C1 −0,21 met BTC; C2 0,49 |
| C4 funding-carry (dollar-neutraal) | de prijs van hefboom | ≈ 0 met alles |
| C5 combinatie | gelijk risico van wat op TRAIN positief was | — |

Sizing gebeurt per munt met inverse vol, daarna één portefeuillefactor naar 20 % vol (k ≤ 2) en
limieten die het hele boek proportioneel schalen. Het boek rekent trades vanaf het **gedrifte**
gewicht en toont de volledige kostentrap.

## D. Roadmap van experimenten

Uitgevoerd, in deze volgorde en vastgelegd vóór de run:

1. Preregistratie en M = 7 in de ledger.
2. Zeven trials op W_DEV.
3. De volledige robuustheidsbatterij per kandidaat.
4. De poorten G1–G9 en de robuustheidsscore.
5. Eén lezing van het poortsample voor de kandidaat met de hoogste score.
6. De authoritative engine.
7. Het oordeel in de ledger.

De vervolg-roadmap staat in §N.

## E. Wijzigingen aan de repository

| Pad | Wat |
| --- | --- |
| `src/tradebot/systematic/book.py` | boek met drift, fees/spread/slippage/impact/funding en een sluitende identiteit |
| `src/tradebot/systematic/{market,signals,sleeves}.py` | causale signalen, sizing, vol-doel, limieten |
| `src/tradebot/systematic/evaluate.py` | metriek via de repo-implementaties (Lo-SE, bootstrap, DSR, PBO), regimes, score |
| `src/tradebot/systematic/programme.py` | `freeze` / `run` / `gate` / `verdict`; de markt wordt vóór `run` afgekapt op de gate-split |
| `src/tradebot/systematic/authoritative.py` | de eindkandidaat door `backtest/engine.py` |
| `src/tradebot/schemas/robust_book.py`, `conf/model/robust_book.yaml` | het configcontract |
| `conf/research/preregistration_robust_book.yaml`, `artefacts/governance/preregistration_eecdb3a1….json` | de preregistratie (bevroren) |
| `artefacts/governance/hypothesis_ledger.json`, `holdout_lock.json` | M = 7, het oordeel, de ene poortlezing |
| `tests/unit/test_systematic_{book,programme}.py`, `tests/lookahead/test_systematic_causality.py` | 43 tests |

Reproduceren:

```text
python -m tradebot.systematic.programme run      # W_DEV
python -m tradebot.systematic.programme gate     # weigert: de lezing is al gebruikt
python -m tradebot.systematic.authoritative
```

## F. Backtestresultaten vóór en na

"Vóór" is de strategie op `main` (weekly meta-label) en de passieve referenties. "Na" is het
robuuste boek. Alles is netto en op W_DEV (2021-11-15 → 2025-09-04), tenzij anders vermeld.

| | netto Sharpe (SE) | CAGR | vol | max DD |
| --- | --- | --- | --- | --- |
| Weekly meta-label (gefilterd, `jolly-feynman` a1f4ada) | −0,78 | — | — | — |
| B1 BTC aanhouden | 0,39 (0,51) | 6,8 % | 53 % | 77 % |
| B2 gelijk gewogen aanhouden | 0,19 (0,49) | −12,5 % | 74 % | 85 % |
| C1 trend long-short | 0,31 (0,49) | 3,6 % | 15 % | 18 % |
| **C2 trend long-flat** | **0,71 (0,55)** | 7,0 % | 10 % | 13 % |
| C3 vol-gemanaged kern | 0,47 (0,52) | 7,9 % | 21 % | 33 % |
| C4 funding-carry | 0,36 (0,51) | 5,5 % | 21 % | 25 % |
| C5 combinatie (C1, C2, C4) | 0,63 (0,51) | 10,2 % | 18 % | 17 % |
| C2 via de **authoritative engine** | **0,39 (0,54)** | 3,1 % | 9 % | 14 % |

Waarom realiseert C2 maar 10 % vol bij een doel van 20 %? De portefeuillefactor is begrensd op
k ≤ 2 (preregistratie). Bij zwakke of tegenstrijdige trendsignalen staat het boek grotendeels
vlak (gemiddeld bruto 0,11). Dat verlaagt de CAGR, niet de Sharpe.

## G. Netto-ontleding van de kosten (W_DEV, % per jaar van de equity)

| | bruto | funding | fees | spread | impact | netto (rekenk.) |
| --- | --- | --- | --- | --- | --- | --- |
| C1 | +8,9 | −1,7 | −1,2 | −0,2 | −1,1 | +4,7 |
| C2 | +9,8 | −1,6 | −0,5 | −0,1 | −0,3 | +7,2 |
| C3 | +13,1 | −3,0 | −0,2 | −0,0 | −0,0 | +9,8 |
| C4 | +18,1 | +2,1 | −2,9 | −0,5 | −9,2 | +7,6 |
| C5 | +21,7 | −0,5 | −3,0 | −0,5 | −6,5 | +11,3 |

Carry vóór kosten heeft Sharpe 0,96; na de (bovengrens-)impact blijft 0,36 over. Daarom bepaalt de
kalibratie van de impact het oordeel over carry (§N.2).

## H. Walk-forward

Geen enkele parameter wordt uit rendementen geschat behalve de causale EWMA-vol en -covariantie.
Elke bar is dus out-of-sample ten opzichte van een fit. Er zijn twee keuzes die "in-sample"
werden gemaakt: de inclusie in C5 (alleen TRAIN) en de selectie (score op W_DEV).

| Sharpe | TRAIN 21-11 → 23-12 | VALIDATE 24-01 → 25-09 | rollend 12m min / mediaan / max | % rollende jaren > 0 |
| --- | --- | --- | --- | --- |
| C1 | 0,60 | −0,08 | −1,15 / 0,34 / 2,25 | 65 % |
| C2 | 0,71 | 0,71 | −2,30 / 0,73 / 2,70 | 65 % |
| C3 | −0,01 | 1,07 | −1,69 / 0,94 / 2,42 | 79 % |
| C4 | 0,07 | 0,73 | −0,64 / 0,57 / 2,13 | 77 % |
| C5 | 0,57 | 0,73 | −0,41 / 0,72 / 2,61 | 82 % |

C2 per jaar: 2022 −5,9 %, 2023 +26,6 %, 2024 +12,6 %, 2025 (t/m sept.) +0,4 %. Per regime:
bull (BTC > MA200) 1,05, bear −0,58. Het wint in trends, niet in bear-markten (het staat dan vlak
en verliest nauwelijks).

## I. Robuustheid (C2, Sharpe op W_DEV; basis 0,71)

| Test | Uitkomst |
| --- | --- |
| Kosten ×0 / ×2 / ×3 / +10 bp slippage | 0,81 / 0,62 / 0,52 / 0,61 |
| Vertraging lag 2 / lag 3 | 0,52 / 0,47 |
| Lookbackraster ×0,5 … ×2 | 0,65 – 0,77 |
| Vol-span 20 / 40 / 90 / 120 | 0,60 / 0,67 / 0,74 / 0,75 |
| k_max 1 / 3 | 0,76 / 0,66 |
| Signaalruis (20 seeds) | min 0,50 · mediaan 0,67 · max 0,82 |
| 5 % gemiste herbalanceringen (20 seeds) | 0,64 – 0,73 |
| Elke 3 / 7 dagen herbalanceren | 0,66 / 0,92 |
| Leave-one-coin-out | 0,59 – 0,77; alleen BTC+ETH 0,64; zonder BTC 0,65 |
| Startdatum +3 / +6 / +9 / +12 mnd | 0,86 / 0,93 / 0,96 / 1,16 |
| Einddatum −3 / −6 / −12 mnd | 0,68 / 0,72 / 0,68 |
| Monte Carlo (blokbootstrap): kans op DD > 25 % binnen 1 jaar | 0 %; p95 DD 15 % |
| Capaciteit $1M / $10M / $100M | 0,64 / 0,44 / −0,06 (met de bovengrens-η) |

Volledige batterijen van alle kandidaten staan in `artefacts/research/robust_book_v1/C*.json`.
Opvallend:

* C1 sterft bij lag 2 (−0,01).
* C4 en C5 zijn kostenfragiel: ×2 kosten geeft −0,13 en 0,15.
* C3 heeft de meeste capaciteit (0,43 bij $100M), maar is een regime-wed (TRAIN −0,01).

## J. Baseline tegenover de nieuwe strategie

Tegenover BTC aanhouden (0,39, DD 77 %) levert C2 een hogere Sharpe (0,71 in de screening, 0,39 in
de authoritative engine). Dat komt met **een zesde van de drawdown** (13–14 %) en een vijfde van de
vol, maar ook met een vergelijkbare of lagere CAGR (7 % tegen 6,8 %; authoritative 3,1 %).

Op het poortsample is het verschil het grootst: C2 −0,8 % tegen BTC −33,7 %. De toegevoegde waarde
is **risicobeheersing**, niet rendement.

## K. Overfitting en statistische betrouwbaarheid

* **M.** Dit programma heeft 7 trials. Er zijn minstens 52 bekende trials sinds de reset op
  ongemergde branches, en meer dan 2.776 ervoor.
* **DSR.** DSR ≥ 0,95 vraagt een geannualiseerde Sharpe van 0,71 bij M = 7, 0,89 bij M = 14 en
  1,20 bij M = 59. C2 heeft DSR 0,50 / 0,36 / 0,17 en haalt de poort dus niet, ook niet bij de
  mildste M.
* **Bootstrap.** Het 95 %-interval van de Sharpe van C2 is −0,30 … 1,69. Nul valt erbinnen.
* **PBO.** Over de verstoringsfamilie is de PBO 0,69 (C2), 0,77 (C1), 0,74 (C3), 0,27 (C4) en
  0,37 (C5). Bij C1–C3 is dat geen teken van een piek maar van een plateau van bijna gelijke
  varianten waarvan de rangorde ruis is. Het is wel de vooraf vastgelegde poort, en hij bindt.
* **Lo-SE.** De SE van de Sharpe is ≈ 0,5 op 3,8 jaar. Om een echte Sharpe van 0,7 met 80 % power
  te bevestigen zijn ≈ 16 jaar dagdata nodig.

## L. De kans dat het out-of-sample blijft werken

Ik combineer een sceptische prior N(0,2; 0,3²) (de ervaring van alle eerdere trials op deze munten)
met de meting. Bij de screeningswaarde 0,71 ± 0,55 is de posterior 0,32 ± 0,26; bij de
authoritative 0,39 ± 0,54 is hij 0,25 ± 0,26. Daaruit volgt:

* P(echte Sharpe > 0) ≈ 83–89 %
* P(> 0,5) ≈ 15–25 %
* P(> 1,0) < 1 %

Bij 10 % vol en een Sharpe van ≈ 0,3 hoort een verwacht netto rendement van 2–4 % per jaar, met een
max DD in de orde van 15 %. Het poortsample (−0,10) past daarbij. De bear-markt was het slechtste
denkbare regime voor long-flat, en het boek overleefde het vrijwel ongeschonden.

## Fase 9 — oordeel over de doelstelling

**Netto Sharpe > 1 en netto CAGR > 15 % is met deze data niet verantwoord haalbaar.**

* De hoogste robuuste Sharpe die ik meet is 0,4–0,7, afhankelijk van de executieaanname.
* Door volatility drag is de hoogst haalbare CAGR bij Sharpe *S* gelijk aan *S*²/2, bereikt bij
  een vol van *S*:
  * Bij *S* = 0,7 vraagt 15 % CAGR al ≈ 26 % vol, met een verwachte drawdown ruim boven het
    mandaat van 25 %.
  * Bij *S* ≤ 0,55 is 15 % CAGR met **geen enkele** hefboom haalbaar (*S* = 0,39 geeft maximaal
    7,6 %).
* Dit is precies de "leverage om het rendement kunstmatig te halen" die is uitgesloten.

Het maximaal realistische met deze data: **C2 trend long-flat, Sharpe ≈ 0,3–0,5, CAGR 3–7 % bij
≈ 10 % vol, max DD ≈ 15 %.** Opschalen naar 20 % vol geeft ≈ 4–12 % CAGR bij een DD van ≈ 25–30 %.

## M. Resterende zwakke punten

* De impact is een bovengrens en de spread is aangenomen. De echte kosten zijn waarschijnlijk lager
  voor carry en C5, en dat kan hun oordeel veranderen.
* Het poortsample is voor de trendfamilie niet maagdelijk: het is op zusterbranches al gelezen.
* Survivorship: zes overlevers.
* Eén lange regime-opeenvolging: 2023 levert het grootste deel van de winst van C2.
* De screeningsengine en de authoritative engine verschillen ≈ 12 bp per bar (correlatie 0,985),
  vooral door de concentratielimiet van de soevereine laag.
* De PBO is op een plateau-familie slecht interpreteerbaar.

## N. Concrete volgende stappen

1. **De data ontgrendelen (grootste hefboom).**
   * In de omgeving van deze sessie zijn `api.bybit.com` (de bron van de bestaande ingestie in
     `data/ingestion/bybit.py`) en `data.binance.vision` (gearchiveerde dagbars, ook van
     **gedelistete** symbolen, dus survivorship-vrij) geblokkeerd.
   * Toevoegen kan via de omgevingsinstellingen: Network access → Allowed domains (laat
     "Allow package managers" aangevinkt). Zie
     <https://code.claude.com/docs/en/cloud-environments#network-access>.
   * Daarna: een universum van de top-40 tot 60 perps naar point-in-time liquiditeit, inclusief
     gedelistete munten, en een nieuwe preregistratie (v2) met XS-momentum, XS-carry en
     TS-trend. Bij een effectieve breedte van 10+ in plaats van 1,3 is een Sharpe rond 1
     plausibel.
2. **Kosten kalibreren.** Gebruik orderboek-snapshots of eigen papertrade-fills om de spread-aanname
   en de η-bovengrens te vervangen. Daarna carry/C5 opnieuw beoordelen (nieuwe trial).
3. **Executie.** Ofwel uitvoering kort na 00:00 UTC (dan geldt de lag-1-meting, ≈ +0,2 Sharpe),
   ofwel 1h/4h-bars om de slippage tussen besluit en vulling te meten. Geef de authoritative
   engine een sub-dag-latency.
4. **Long-only sleeves in spot** om funding te vermijden (≈ +1,1 %/jaar netto op C2 na hogere
   spotfees).
5. **Andere activaklassen** (futures/ETF-trend). Die liggen achter dezelfde egress-blokkade
   (Stooq, Yahoo, FRED).
6. **Papertrading van C2** alleen als operationele test (fills, latency, funding), niet als
   kapitaalbesluit.
7. **Governance.** Merge of sluit de zusterbranches en boek hun 52 trials in de ledger. Herstel de
   DSR-poort op het weekly-programma. Pas de live-regel van 30 dagen positieleeftijd aan, of
   documenteer die in de backtest.
