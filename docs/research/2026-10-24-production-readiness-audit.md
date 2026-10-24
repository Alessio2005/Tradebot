# Productie-audit: wat staat er, wat is gerepareerd, wat ontbreekt (2026-10-24)

De vraag: is de setup klaar voor productie? Wat zijn de echte cijfers? Zijn de modellen
af, en loopt alles? Deze audit beantwoordt dat. De bevindingen die in de repository te
repareren waren, zijn in dezelfde ronde gerepareerd. Elke bevinding heeft hieronder een
status:
- **opgelost**: met test;
- **gerapporteerd**: een meetfout, opgelost zonder een bevroren oordeel te veranderen;
- **open**: met wie het moet doen.

## Samenvatting

- **Niet productieklaar. Er is geen strategie die kapitaal verdient.**
  - De ledger telt 16 oordelen, en geen enkele PASS.
  - De enige kandidaat is de spot-perp-carry v8-T2. Die is formeel gefalsificeerd op de
    holdout 2025–26. Zijn enige schone toets is de vooruit-lezing, op zijn vroegst in
    april 2027.
- **De gerapporteerde Sharpe van het carryboek is 3 à 5 keer te hoog.**
  - De dagrendementen zijn sterk autogecorreleerd: ρ₁..₃ = 0,49 / 0,55 / 0,45.
  - `sqrt(365)` neemt aan dat ze onafhankelijk zijn.
  - Over de hele cyclus geeft dat 5,05. Op maandrendementen is het **1,55**, op
    kwartaalrendementen 1,08.
  - De echte jaarvol is 8,4 %, niet 2,5 %.
  - Bij de prijsstrategieën (trend, momentum) vallen iid- en maand-Sharpe samen. Het
    effect zit dus in de carry, niet in de methode.
- **De live-stack kon kapitaal kosten.** Gereproduceerd en opgelost:
  - live fills werden nooit geboekt;
  - een halt sloot alleen het papierboek;
  - een lege orderstatus werd "volledig gevuld tegen prijs 0";
  - een tweede noodsluiting plaatste niets.
- **De infra kon niet bouwen of starten.** Ze wees naar een `apps/`-map die niet bestaat,
  draaide op Python 3.11 tegen een eis van ≥ 3.13, en de promotiepoort was een rooktest
  zonder één order. Dat is vervangen door wat werkt.
- **Het carryboek heeft nu een uitvoeringspad.** Het draait met DEZELFDE code als de
  backtest: een shadowboek dat dag voor dag stapt, is de backtest, bit voor bit. Een echte
  beursadapter ontbreekt nog. Daarvoor zijn een beurs, een jurisdictie en een testnet
  nodig, en die keuze ligt bij de eigenaar (§6).

## 1. De cijfers, eerlijk uitgedrukt

Bron: `artefacts/research/restatement.json`, gebouwd met
`python -I -m tradebot.systematic.restate` uit de bevroren dagreeksen. Er is niets
opnieuw gedraaid, en geen poort, oordeel of ledger-entry is veranderd.

### v8-T2: spot-perp-carry, gefinancierde tranche, sprongrisicogrens

| venster | Sharpe (bevroren, √365) | Sharpe maand | Sharpe kwartaal | Lo (60 lags) | CAGR | max DD |
| --- | --- | --- | --- | --- | --- | --- |
| backcast 2020-04 .. 2020-12 | 9,03 | 3,28 | – | 3,29 | 24,7 % | 0,9 % |
| W_DEV 2021-01 .. 2025-06 | 5,33 | 1,62 | 1,13 | 1,33 | 15,9 % | 1,5 % |
| holdout 2025-07 .. 2026-09 | −2,00 | −1,80 | – | −2,17 | −1,1 % | 1,4 % |
| **hele cyclus (6,5 jaar)** | **5,05** | **1,55** | **1,08** | **1,26** | **13,4 %** | **2,0 %** |

**Per jaar:**

| jaar | rendement |
| --- | --- |
| 2020 | +18,1 % |
| 2021 | +61,0 % |
| 2022 | +0,2 % |
| 2023 | +4,0 % |
| 2024 | +15,7 % |
| 2025 | −0,4 % |
| 2026 t/m september | −1,0 % |

**Wat de tabel niet laat zien:**
- **Concentratie:** 58 % van de winst over de hele cyclus komt uit 2021.
- **Typisch jaar:** het mediane rollende 12-maandsrendement is 5,0 %, het slechtste −1,4 %.
- **Onderwater:** de langste periode onder de vorige top duurt 605 dagen.
- **Lo met meer lags:** de schatting zakt verder naarmate meer lags meetellen
  (Bartlett-gewichten): 1,55 bij 30 lags, 1,26 bij 60 en 1,03 bij 120. De autocorrelatie
  loopt over maanden, want funding komt in golven.

**De drawdown is klein, het rendement onvoorspelbaar.** Het risico van dit boek zit niet in
de vol, maar in de staart (beurs, auto-deleveraging, stablecoin) en in de vraag of er
carry is. Op W_DEV zegt de bevroren Sharpe 5,3. De verdedigbare verwachting is een Sharpe
van ~1–1,5 en ~5–13 %/jaar, met carryjaren tot +60 % en droge jaren rond nul.

### De DSR op maandrendementen (W_DEV, 54 maanden)

| M | 27 | 79 | 500 | 2000 |
| --- | --- | --- | --- | --- |
| DSR | 0,999 | 0,987 | 0,81 | 0,50 |
| vereiste jaar-Sharpe | 0,96 | 1,15 | 1,44 | 1,63 |

M = 79 is een ondergrens. De ledger is op 2026-09-26 op besluit van de eigenaar gereset, en
de 52 "bekende eerdere trials" zijn een schatting. Bij een paar honderd echte trials zakt de
DSR onder de bevroren drempel van 0,95.

### Een beursfaillissement, met de vol die bij de Sharpe hoort

Rekenregel: Sharpe = (μ − p·LGD) / √(σ² + p(1−p)·LGD²), met μ = 13,4 % (CAGR over de
cyclus) en LGD = 1 / het aantal beurzen.

| kans p per jaar | 1 beurs | 2 beurzen | 3 beurzen |
| --- | --- | --- | --- |
| 1 % | 0,95 | 1,32 | 1,45 |
| **3 %** | **0,55** | **0,99** | **1,22** |
| 5 % | 0,36 | 0,79 | 1,06 |

- Met σ = 8,4 % (maand) ligt de Sharpe bij één beurs onder 1, bij elke realistische p.
- Het rapport van v6–v8 (§6) rekende met σ = 2,9 % en kwam op 0,79 / 1,69 / 2,46 bij
  p = 3 %.
- **Twee beurzen zijn de ondergrens voor Sharpe ≈ 1. Drie zijn nodig voor marge.**

### De rest van de repo, ter vergelijking

| kandidaat | venster | Sharpe √365 | Sharpe maand | CAGR |
| --- | --- | --- | --- | --- |
| v5-H3, carry BTC/ETH zonder hefboom | hele cyclus | 6,56 | 1,61 | 6,2 % |
| v7-S2 (v8 zonder sprongrisicogrens) | hele cyclus | 4,36 | 1,37 | 12,9 % |
| v6-K1, 1× per been | hele cyclus | 3,70 | 1,19 | 8,8 % |
| v3-Y5, carry + trend (prijs) | W_DEV / holdout | 1,13 / −1,19 | 0,92 / −0,89 | 20,7 % / −17,6 % |
| v2-X4, trend long-flat | W_DEV / holdout | 0,77 / −0,11 | 0,67 / −0,14 | 8,2 % / −1,1 % |
| BTC buy-and-hold | W_DEV | 0,58 | 0,57 | 18,2 % (max DD 79 %) |
| weekly ML-meta-label | OOS 2022–26 | AUC 0,49 | – | DSR 0,004: INVALID |

## 2. De modellen

| model | code | getraind artefact | oordeel |
| --- | --- | --- | --- |
| carry v8-T2 (`systematic/leverage.py`, `programme_v8.py`) | af, reproduceerbaar | n.v.t. (regelgebaseerd) | gefalsificeerd op de holdout; vooruit-lezing ≥ april 2027 |
| carry v5–v7 | af | n.v.t. | gefalsificeerd |
| prijsboeken v1–v4 (trend, momentum, cross-sectionele carry) | af | n.v.t. | gearchiveerd of gefalsificeerd |
| weekly ML-meta-label (`train/light_models.py`) | af | alleen metadata, geen binary | INVALID (AUC 0,49) |
| CatBoost/ensemble (`train/catboost.py`) | af; P²-schaler hersteld 2026-10-18 | geen | nooit gepromoveerd |
| trend/carry in `research/` | af | n.v.t. | haalt geen enkele poort A1–A6 |
| ML-live-engine (`tradebot.live`) | gerepareerd (§4) | heeft geen model om te laden | niet inzetbaar, en niet nodig |

"Compleet" is daarmee het verkeerde woord. De code is af. **Er is geen model met edge**,
behalve de carry, en die is een regel, geen model. Meer ML-capaciteit lost dat niet op: twee
onafhankelijke routes vonden een OOS-AUC van 0,49–0,51 en een rang-IC van −0,003.

## 3. Silent killers (kwantitatief)

| # | bevinding | status |
| --- | --- | --- |
| SK-1 | **Sharpe √365 op autogecorreleerde dagrendementen.** De Lo-SE in `sharpe_with_se` corrigeert de standaardfout, niet de puntschatting. Raakt elke poort, de DSR, de z-toetsen (holdout z = −22) en de robuustheidsscore van v5–v8. | **gerapporteerd.** `validation/serial_sharpe.py` (maand, kwartaal, Lo met Bartlett-gewichten); `systematic/evaluate.summarize` rapporteert ze naast de bevroren Sharpe; `systematic/restate.py` drukt alle bevroren uitkomsten opnieuw uit. Een nieuwe poort is een preregistratie: open, eigenaar. |
| SK-2 | **De vooruit-toets meet het regime, niet de edge.** De bevroren regel toetst z tegen W_DEV-Sharpe 5,33; een droog halfjaar falsificeert vanzelf. Bovendien is de lezing een backtest op nieuwe data, geen echte fills. | **open, eigenaar.** Voorstel in §7: vóór de eerste vooruit-lezing een aanvullend criterium registreren. |
| SK-3 | **De edge is een fundingregime.** Holdout 2025–26: funding +0,04 %/jaar, impact −0,93 %/jaar. Is de markt nu zo, dan is de verwachting ≈ 0 min kosten. | **open.** Een feit, geen defect: kapitaal alleen in een carryregime, of met de ruststand (cash) als bewuste keuze. |
| SK-4 | **Uitvoering in het model is optimistisch.** `lag_bars: 1` betekent uitvoering op de close van de besluitbar zelf. Beide benen worden tegelijk gevuld (geen leggingrisico). Spot-spreads zijn aangenomen, impact is `IMPACT_UNCALIBRATED`. Liquidatie wordt op dagbars getoetst. Onder margestress zakt de buffer tot 0,70× de onderhoudsmarge. | **open.** Meten kan pas met echte fills (§6). Het shadowboek legt elke order vast, zodat de vergelijking klaarstaat. |
| SK-5 | **M is een ondergrens** (ledger-reset 2026-09-26). | **gerapporteerd** (DSR-tabel in §1). |
| SK-6 | **De sprongrisicogrens (σ ≤ 0,20) is ontworpen na de holdout.** | bekend en zo geboekt; alleen de vooruit-lezing kan hem toetsen. |
| SK-7 | **Bron-gaten worden met kennis van de toekomst overbrugd.** Of een gat ≤ 7 bars blijft, weet je pas als de munt terugkomt. Een live-boek op een groeiend paneel zou op de rand een gedwongen exit nemen die de backtest niet neemt. | **opgelost voor het shadowboek:** het verwerkt alleen gesettelde bars (≥ 7 bars data erna). Een live-feed van de beurs zelf heeft deze gaten niet. |

## 4. Engineering

| # | bevinding | status |
| --- | --- | --- |
| E-1 | **Geen uitvoeringspad voor de carry.** De live-stack is een ML-engine op één been voor Bybit; geen spotbeen, lening of uniMMR. | **opgelost (paper/shadow), open (live).** Zie de opbouw onder deze tabel. |
| E-2 | **De infra kon niet bouwen of starten.** `apps/` bestaat niet; Python 3.11 tegen ≥ 3.13; geen installatie uit de lock; `infra/docker/.dockerignore` werd bij context `../..` nooit gelezen; de network policy liet alleen `live-engine` naar buiten. | **opgelost.** Eén image (`Dockerfile.runner`: 3.13, uit de lock, editable). Een maandelijkse `cron-carry-shadow`. De kapotte stubs zijn weg. `tests/unit/test_infra_contract.py` toetst Python-versie, lock, COPY-bronnen, modules, PVC's, egress en pod-security. |
| E-3 | **Live fills werden nooit geboekt.** De engine las equity en gewichten uit het papierboek, dus elke rebalance stuurde het volle doel opnieuw en de breakers zagen de live-P&L niet. | **opgelost** (`oms/router.py`: boeken met fee, één keer per order). |
| E-4 | **Een HALT sloot alleen papierposities.** | **opgelost** (`live/engine.py`: `_halt` via de router). |
| E-5 | **Fantoomfills.** Een lege of nog lopende orderstatus werd "volledig gevuld tegen prijs 0". | **opgelost.** Peilen tot een eindstatus; geen uitvoering is een fout; een gedeeltelijke fill boekt wat er werd uitgevoerd. |
| E-6 | **De noodsluiting werkte één keer.** De `orderLinkId` lag vast; `reduceOnly` ontbrak. | **opgelost.** |
| E-7 | **Reconciliatie ontbrak.** De `Reconciler` werd nergens gebruikt en keek alleen naar het interne boek. Secrets heetten `BINANCE_*`, de router las `BYBIT_*`. | **opgelost.** De reconciliatie draait elke bar in live-modus, over de unie van beide boeken; een blijvend verschil is een halt. De secret-stub is weg: het shadowboek gebruikt geen sleutels. |
| E-8 | **Governance-documenten ontbreken.** `RISK_MANDATE`, `RISK_CONTRACT`, `MEASUREMENT_CONTRACT`, `DATA_REGISTER`, `ARCHITECTURAL_DECISIONS`, `DEFERRED_ISSUES` en `reports/` worden ~60 keer genoemd. | **open, eigenaar.** Ze staan niet in git en zijn niet te reconstrueren zonder de bron. |
| E-9 | **Een fout in het barpad stopte de engine met open posities.** | **opgelost.** Eerst plat via de router, dan de fout naar boven. |
| E-10 | **De rolling Sharpe in de state-file kon nooit iets schrijven.** Hij rekende met 5s-bars en 252 dagen, eiste 17.280 samples, en de historie werd op 5.000 afgekapt. Op rebalancebars werd dubbel gesampled. | **opgelost** (was een strikte xfail). |
| E-11 | **Holdoutsloten werden zonder splittoets gelezen** (v2–v8, vooruit-lezing inbegrepen). Een slot met een vroegere split zou W_DEV overlappen. | **opgelost** (`programme_v2.require_lock_split`, vóór de lezing; was een strikte xfail). |
| E-12 | **De DSR-M werd live uit de ledger gelezen** (v2, v3, v5, v6). Een herhaalde run na een volgend programma gebruikte een andere M. | **opgelost** (`programme_v2.frozen_m`, de stand bij de freeze; was een strikte xfail). |
| E-13 | **De paper-trade-"smoke" was een promotiepoort.** 24 synthetische bars, `SignalRunner(signals=[])`, nul orders. | **opgelost.** Hernoemd naar wat hij is; uit de promotievoorwaarden gehaald. |

**De opbouw van E-1:**
- `systematic/leverage.py`: `run_levered` is opgesplitst in een `LeveredBook` met `step`
  (backtest en paper) en `decide_live` (de beurs geeft de stand). De backtest is
  bit-identiek over 16 scenario's: liquidatie, slicing, skip, ADV-cap, lag 2 en de
  volledige v8-simulatie.
- `carry/`:
  - orders per been in een margeveilige volgorde: eerst de short terug, dan pas het
    onderpand weg;
  - de staat per symbool, exact en atomair, gebonden aan de config-hash;
  - een venue-contract en een paper-venue;
  - reconciliatie tegen de beurs (vangt ook auto-deleveraging);
  - een kill switch die gehedged afbouwt;
  - een CLI.
- **Pariteit:** shadow = backtest, ook met de staat elke dag naar schijf en terug, en op
  een paneel dat dag voor dag groeit (`tests/unit/test_carry.py`).

**Nog open:**
- `features/fracdiff.py`: de gewichten staan in omgekeerde volgorde ten opzichte van AFML
  5.3.2. Het blijft causaal en de amplituderespons is gelijk. Aanpassen verandert bevroren
  features; het blijft een gedocumenteerde xfail.
- DI-32: `.gitignore` regel `env/` slikt ook `conf/env/`.

## 5. Loopt alles?

- CI op `main` is groen in alle 8 workflows.
- Op deze branch, lokaal: ruff en mypy schoon, de volledige suite groen, dekking 100 %.
- Van de zes strikte xfails van vóór deze audit zijn er vijf opgelost: de rolling Sharpe
  (E-10), en voor v2 én v3 de splittoets (E-11) en de DSR-M (E-12). Wat overblijft is de
  fracdiff-bevinding, bewust ongewijzigd.

Groen zegt hier wat het zegt: de code doet wat de tests vragen. Of de strategie geld
verdient, toetst geen enkele test.

## 6. Wat er nog moet gebeuren, en door wie

**Eigenaar (besluiten):**
1. **Beurs en jurisdictie.**
   - Binance trok zich in 2023 terug uit Nederland.
   - Controleer of spot, USDⓈ-M-perps én portfolio margin voor jouw rechtspersoon
     toegankelijk zijn. Kies minstens twee beurzen (§1).
2. **Een aanvullend vooruit-criterium (SK-2)**, bevroren vóór de eerste vooruit-lezing
   (zie §7).
3. **De governance-documenten (E-8)** terugzetten, of de verwijzingen opruimen.
4. **Kapitaal.** Pas na §7, en niet in een regime zonder carry, tenzij cash aanhouden de
   bewuste keuze is.

**Engineering (vraagt toegang tot de beurs):**
1. **Een venue-adapter** volgens `carry/venue.Venue`:
   - Binance spot, USDⓈ-M en portfolio margin;
   - eerst testnet;
   - per order een eindstatus; reduce-only; lot-step-afronding;
   - de beursstand inclusief lening en uniMMR.
2. **Een dagelijkse datafeed** van de beurs (REST: klines, funding, mark price). Daarmee
   kan het boek dagelijks draaien met `settle=0` in plaats van maandelijks op archieven.
3. **TCA.** Elke shadow-order tegen de werkelijke fill, per been. Kalibreer spread, impact
   Y en leenrente (SK-4).
4. **Monitoring voor carry:**
   - funding die omslaat;
   - basis boven 5 %;
   - uniMMR onder 3;
   - een piek in de leenrente;
   - een stablecoin-depeg;
   - auto-deleveraging (de reconciliatie vangt dat nu al als breuk).
5. **Een tweede en derde beurs** (Bybit, OKX). Die hosts zijn in deze ontwikkelomgeving
   geblokkeerd.

## 7. Voorstel: wanneer is het boek klaar voor kapitaal

**Waarom een aanvullend criterium nodig is.** De bevroren vooruit-regel (z < −1,645 tegen
5,33) falsificeert elk droog halfjaar. Dat is een uitspraak over het regime, niet over de
edge.

**Voorstel, als nieuwe preregistratie**, bevroren vóór de eerste lezing van data na
2026-10-01. Alle vier moeten gelden:
1. **Pariteit.** Het shadowboek (`tradebot.carry step`) en de vooruit-backtest geven per bar
   hetzelfde netto rendement. Dit is nu bewezen op synthetische data; daarna op de echte
   vooruit-panelen.
2. **Uitvoering**, zodra er een venue-adapter is: over ≥ 3 maanden testnet of klein
   kapitaal ligt de implementation shortfall per been ≤ 1,5× de modelkosten van dezelfde
   orders, en zijn er nul reconciliatiebreuken die niet verklaard zijn.
3. **Edge, regimebewust.** In de maanden waarin de 30-daagse funding van BTC/ETH boven de
   instapdrempel lag, is het boek belegd en vangt het ≥ 50 % van de funding op zijn
   notional, na kosten. In maanden zonder carry is het verlies ≤ 1 %/jaar (de kosten van
   kijken).
4. **Risico.** Geen liquidatie; drawdown ≤ 25 %; minstens twee beurzen.

Bevriezen is een besluit van de eigenaar: een preregistratie in een append-only ledger
legt vast wat bewijs is. Deze audit doet dat daarom niet zelf.

## 8. Reproduceren

```text
python -I -m tradebot.systematic.restate              # artefacts/research/restatement.json
pytest tests/unit/test_carry.py                       # shadow = backtest, bit voor bit
pytest tests/unit/test_infra_contract.py              # de infra kan bouwen en starten
pytest -m "not slow and not regression" --cov=src/tradebot   # 100 % dekking
```
