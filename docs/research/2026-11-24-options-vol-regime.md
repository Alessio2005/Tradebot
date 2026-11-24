# Options: volatility risk premium, theta harvesting en vol-regimes

Datum: 2026-11-24. Code: `src/tradebot/options/vol_regime.py`, `src/tradebot/options/selection.py`,
config `conf/options/default.yaml` (secties `vol_regime` en `theta_harvest`).

## Waarom

De strategieën verdienen aan drie dingen, en deze laag maakt ze expliciet:

1. **Volatility risk premium (VRP).** Opties zijn gemiddeld duurder dan de beweging die daarna
   komt: de implied vol ligt meestal boven de gerealiseerde vol. Verkopen loont alleen als die
   premie er nu ook is, dus de IV wordt vergeleken met de VERWACHTE RV, niet met een getal uit
   het verleden.
2. **Theta harvesting.** Een short optie verdient tijdswaarde en verliest aan beweging (gamma).
   Per positie moet de tijdswaarde het verwachte bewegingsverlies dekken.
3. **Regimeselectie.** Een strategie opent alleen in de marktomstandigheden waarvoor ze bedoeld
   is. In een vol-schok of bij een inverse termijnstructuur gaat er geen nieuwe short premium
   het boek in.

Er wordt niets getraind of gefit. Elke drempel staat in de config en is vooraf vastgelegd.

## De regels

### Verwachte RV

`σ_RV² = (0,2·RV5² + 0,5·RV20² + 0,3·RV60²) / (som van de gebruikte gewichten)`, met RV de
Yang-Zhang-vol over 5, 20 en 60 handelsdagen, geannualiseerd met √252. Dat is de vorm van
HAR-RV (Corsi 2009) zonder de regressie. Ontbreekt RV60 (te weinig dagen), dan tellen RV5 en
RV20 met hun gewichten herschaald. Zonder RV20 is er geen verwachting en dus geen regime.

### Regime per onderliggende

De eerste regel die geldt, beslist:

| Regime  | Regel                                                                         |
|---------|-------------------------------------------------------------------------------|
| UNKNOWN | geen ATM-IV(30) of geen RV-verwachting                                        |
| STRESS  | IV60 − IV30 ≤ −2 volpunten (backwardation), of RV5 / RV20 ≥ 1,5 (vol-schok)  |
| RICH    | IV30 / σ_RV ≥ 1,15, IV30 − σ_RV ≥ 2 volpunten en helling ≥ −0,5 volpunt       |
| CHEAP   | IV30 / σ_RV < 0,95                                                            |
| NORMAL  | al het andere                                                                 |

STRESS op SPY (`market_underlying`) geldt voor elke onderliggende.

### Wie mag openen

| Regime  | Strategieën die nieuwe posities mogen openen                                        |
|---------|-------------------------------------------------------------------------------------|
| RICH    | credit spread, iron condor, cash-secured put, covered call, wheel, short strangle*  |
| NORMAL  | credit spread, debit vertical, calendar, long straddle/strangle, CSP, CC, wheel      |
| CHEAP   | debit vertical, calendar, long straddle/strangle                                    |
| STRESS  | geen                                                                                |
| UNKNOWN | geen                                                                                |

\* De short strangle laadt alleen met `risk.allow_undefined_risk: true`; die staat uit.

De strategieën houden hun eigen IV-voorwaarde (IV rank, of IV/RV20 zolang er geen 252 dagen
historie is). Het regime komt daar bovenop. Sluiten, stops, rolls en bijsturingen gaan in
elk regime door.

### Theta-dekking

Voor elke kandidaat die premie ontvangt:

`dekking = (θ per dag × 365) / (½ · |Γ| · S² · σ_RV²) ≥ 1`

Voor één gehedgede short optie is dat ongeveer `σ_IV² / σ_RV²`: de VRP op het niveau van de
positie, met de skew en de gekozen strikes erin. Een kandidaat zonder positieve theta valt af.
`min_theta_per_max_loss` (standaard uit) kan daarnaast een minimum θ per dag eisen als fractie
van het maximale verlies. Kandidaten die premie betalen (debit) gaan zonder deze poort door.

## Wat het journal laat zien

- `vol_regime`: bij elke wisseling, met ATM-IV, σ_RV, RV5/RV20, VRP-verhouding en -punten en
  de termijnhelling.
- `candidate`: spoorregels `regime:<regime>:ok` en `theta:cover=<x>:eff=<θ/max verlies>:ok`.
- `scan_summary`: de regimes per onderliggende en het aantal geblokkeerde kandidaten per
  strategie en poort.

## Wat dit niet bewijst

De drempels zijn redelijke standaardwaarden, geen gemeten optimum. Er is geen historische
ketendata, dus het effect op rendement is niet gemeten. Het bewijs moet uit forward
paper-trading komen: per regime het aantal entries en de uitkomst, uit het journal.

Cash-secured puts en covered calls (en de wheel) hebben in de literatuur het sterkste bewijs
(de CBOE PutWrite- en BuyWrite-indexen), maar staan uit. Hun maximale verlies is in v1
`strike × 100` per contract; bij het fictieve kapitaal van 25.000 USD en 2 % maximaal verlies
per positie (500 USD) past daar op geen enkele onderliggende uit het universum één contract van.
Ze aanzetten vraagt een ander sizingbegrip voor gedekte posities, een groter fictief kapitaal,
of beide. De regimeregels voor ze staan al klaar.

## Replay op de synthetische standaardmarkt

De synthetische markt test de engine en de regels, niet het rendement (geen VRP ingebouwd:
de IV-smile en de GBM-vol liggen dicht bij elkaar).

`python -I -m tradebot.options replay` op de standaardmarkt (`tradebot.options.data.synthetic`),
vóór en na deze wijziging: dezelfde 3 posities, dezelfde eindstand (25.303,60 USD fictief), geen
HALT. De poorten blokkeerden hier niets, omdat elke kandidaat al in een passend regime viel:

| Moment           | Onderliggende | Regime   | Waarom                         | Gevolg                               |
|------------------|---------------|----------|--------------------------------|--------------------------------------|
| 2026-12-01 10:00 | TSTA          | RICH     | IV 22,0 % tegen σ_RV 19,0 %    | credit spread, theta-dekking 1,12    |
| 2026-12-01 10:00 | TSTB          | CHEAP    | IV 22,1 % tegen σ_RV 25,9 %    | long straddle (16 DTE) op 2 december |
| 2026-12-02 10:00 | TSTA          | NORMAL   | VRP-verhouding 1,14            | credit spread, theta-dekking 1,05    |
| 2026-12-15 10:00 | TSTA          | STRESS   | RV5 / RV20 = 1,74 na de sprong | geen nieuwe short premium meer       |
