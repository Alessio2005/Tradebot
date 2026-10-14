# Robuust boek v6: de basiscarry met hefboom, in één unified-margin-account

**Status:** ontwerp, geschreven vóór het bevriezen en vóór elke v6-run. Post-hoc na v5 en
zo geboekt (M = 21 + 3 = 24, plus 52 bekende eerdere trials).

**Eigenaarsbesluit (2026-10-09):** "Je mag high risk, high reward spelen." Dat is het
besluit dat het v5-rapport (§5, "Die keuze ligt bij de eigenaar") openliet: hefboom om de
CAGR-doelstelling te halen is niet langer uitgesloten. De rest van het mandaat blijft
staan (`conf/risk/default.yaml`):
- bruto exposure ≤ 4,0 (beide benen);
- ADV-participatie ≤ 1 %;
- max drawdown 25 % als ruïnelijn.

## 1. Diagnose: waarom v5 de CAGR niet haalde

| v5 | W_DEV-Sharpe | CAGR | gemiddelde notional per been | max per been |
| --- | --- | --- | --- | --- |
| H1 basis top-50 | 5,66 | 8,6 % | 0,33 | 0,70 |
| H3 basis BTC + ETH | 6,76 | 7,0 % | 0,26 | 0,70 |

1. **De Sharpe is het probleem niet.** De basiscarry is de enige bron van rendement in
   deze repo die buiten W_DEV standhield: H3 op de ongemeten backcast 2020 haalde een
   Sharpe van 11,5 (z = +2,79). Elk prijssignaal (trend, momentum, cross-sectionele
   carry) zakte buiten W_DEV naar ≤ 0: −0,11, 0,00 en −1,19.
2. **De kapitaalefficiëntie wel.** In het tweewalletmodel van v5 (70 % spot, 30 %
   USDT-marge) stond gemiddeld maar 0,26–0,33 van de equity per been in de markt. Als
   het boek belegd was, ving het ~27–30 %/jaar funding op de notional; op de equity
   werd dat 7–9 %.
3. **"High risk" betekent hier niet: risico zonder premie erbij nemen.** Een
   prijssleeve toevoegen omdat het mag, is een weddenschap zonder aangetoonde edge. Het
   risico dat v6 neemt, is het risico van de edge die er wél is: hefboom op de carry.
   Dat risico zit in de staart (liquidatie, financiering, beurs), niet in de vol
   (1–2 %/jaar).

## 2. Beschrijvende feiten, alleen uit TRAIN (2021-01-01 .. 2023-12-31)

Gemeten vóór elke v6-run, op data die v5 al gebruikte plus de nieuwe mark-price-panelen.
Validate, backcast en holdout zijn hiervoor niet bekeken.

- **BTC-carry** (EWMA 30 d, de financieringsproxy): gemiddeld 31,4 / 4,4 / 7,1 %/jaar in
  2021 / 2022 / 2023, en boven de 8 %-vloer op 84 / 7 / 33 % van de dagen.
- **H1 op TRAIN** (het bestaande v5-artefact): gemiddeld 0,32 notional per been, max 0,80
  (drift), op 73 % van de dagen belegd, 11,3 %/jaar funding op de equity.
- **Intraday-extremen in het point-in-time-universum** (54.215 munt-dagen):

  | | p99 | p99,9 | max |
  | --- | --- | --- | --- |
  | mark-high t.o.v. de vorige close | +31 % | +77 % | +1.131 % |
  | last-high t.o.v. de vorige close | +31 % | +79 % | +1.186 % |
  | basisstress: mark-high − min(spot-high, mark-high) | 0,3 % | 2,2 % | 76 % |

  Het risico van een gehedged short zit dus vooral in de haircut-erosie bij een squeeze
  (`a · h · rally`). De basis zelf is meestal klein, maar heeft een staart. Mark-dekking
  in het universum: 99,5 %.
- **Liquiditeit:** min(spot-ADV, perp-ADV) over 30 dagen heeft een mediaan van 43 mln,
  een p10 van 10 mln, en ligt op 73 % van de munt-dagen boven 20 mln. Bij een boek van
  1 mln en een slot van 0,2 bindt de 1 %-ADV-cap dus op ruwweg een kwart van de
  munt-dagen.

## 3. Het account: unified (portfolio) margin

Alle grootheden zijn fracties van de equity aan het begin van de bar:
- `a_i` = spot long;
- `b_i` = perp short;
- `c = 1 − Σa` = USDT, negatief is een lening.

Perp-P&L en funding rekenen dagelijks af in USDT.

**Rendement over bar *t*:**

    a·r_spot − b·r_perp + b·f − lening · r_fin(t−1) / 365,   lening = max(0, Σa − 1)

**Financiering** `r_fin(t) = max(8 %, 1,0 × BTC-carry(t))`, bekend op de close van *t*.
- *De reden:* een USDT-uitlener kan zelf de BTC-basis oogsten, dus lenen onder dat
  tarief is geen gratis geld. Hefboom betaalt daardoor alleen in munten met meer carry
  dan BTC.
- *Conservatief:* in 2021 rekent dit model ~31 % rente, ruim boven wat Binance toen
  vroeg.
- *Stress:* max(12 %, 1,5 × BTC-carry).
- *Optimistisch, alleen ter informatie:* constant 6 %.

**Onderpand en marge:**
- Spot telt als onderpand tegen `1 − h`: h = 5 % voor BTC en ETH, 20 % voor de rest
  (stress 35 %).
- Haircut-equity `AE = 1 − Σ a·h`.
- Onderhoudsmarge `MM = 2 % · Σb + 5 % · lening` (stress: 3 % op de perp).
- `uniMMR = AE / MM`.

**Liquidatietoets** op de mark price, het hele boek in twee staten:
- *up:* elke short op zijn mark-high; de spot wordt hooguit even ver meegerekend,
  `min(spot-high, mark-high)` (de basis kan alleen pijn doen);
- *down:* elke short op zijn mark-low; de spot op het slechtste van spot-low en mark-low.

Raakt `AE + ΔAE − MM` in een van beide staten nul, dan is het account geliquideerd:
- alles gesloten in die staat, min 1,5 % liquidatievergoeding over de bruto notional;
- geen funding voor die bar;
- de volgende close hedget opnieuw.

Ontbreekt de mark price, dan geldt de last price, die verder wickt. Het boek houdt bij
hoe dicht het bij liquidatie kwam (`stress_headroom`).

**Governor**, na elke trade:
- bruto ≤ 4,0;
- uniMMR ≥ 3.

Wordt een van beide geschonden, dan gaat het hele boek naar het doel, geschaald met de
grootste factor die beide houdt.

**Wat ongewijzigd uit v5 komt** (`harvest.py`):
- de slotregel van H1: top-50, tien slots, instap ≥ 15 %/jaar, uitstap < 5 %, EWMA 30 d;
- band 50 % en hedge-tolerantie 2 %;
- delisting sluit beide benen;
- kosten per been (perp 5,5 bp fee + 3 bp spread, spot 10 + 5 bp, √-impact met Y = 1);
- `lag = 1` en een boek van $1M.

**Nieuw: de ADV-cap van het mandaat.** Geen been is groter dan 1 % van de 30d-ADV van het
dunste been (min(spot, perp)), gerekend tegen de **lopende** equity op het
uitvoeringsmoment. Bij een vaste begin-AUM zou het boek na een verdubbeling van de equity
stilletjes op 2 % zitten. v5 dwong de cap niet af: een slot van 70 k in een munt met
2 mln ADV is 3,5 %.

**Implementatie:** `systematic/leverage.py`, getoetst in
`tests/lookahead/test_levered_carry_causality.py`:
- causaliteit van besluit, financiering en boek, met en zonder ADV-cap;
- de cap houdt elk verhandeld been onder 1 % van de ADV tegen de lopende equity;
- een perfecte hedge verdient funding min financiering;
- de lening is alleen de spot boven de equity;
- de governor houdt bruto en uniMMR;
- een mark-squeeze liquideert, een last-price-wick alleen niet;
- zonder mark geldt de last price.

## 4. De trials (M = 21 + 3 = 24)

| trial | notional per been | bruto max |
| --- | --- | --- |
| K1_CARRY_PM_1X | 1,0 | 2,0 |
| K2_CARRY_PM_1P5X | 1,5 | 3,0 |
| K3_CARRY_PM_2X | 2,0 | 4,0 (= de mandaatcap) |

Dezelfde slotregel, hetzelfde account. Ze verschillen alleen in hefboom: een
dosis-responsreeks, geen zoektocht. K1 meet wat kapitaalefficiëntie zonder lening
oplevert. Boven 2,0 per been zou het mandaat worden opgerekt, en dat besluit ik niet.

**Selectie (vooraf vastgelegd):**
1. Van de kandidaten zonder bindende W_DEV-poort kiest v6 de **laagste hefboom**: het
   minste staartrisico dat de doelstelling haalt.
2. Haalt geen enkele kandidaat alle poorten, dan kiest de robuustheidsscore van v5. Het
   oordeel is dan "gearchiveerd".

## 5. Poorten op W_DEV (2021-01 .. 2025-06)

**G1–G9 zijn ongewijzigd uit v5**, met dezelfde drempels:
- Sharpe ≥ 1 en CAGR ≥ 15 %;
- TRAIN- en VALIDATE-Sharpe ≥ 0,5;
- drawdown ≤ 25 %;
- bij dubbele kosten ≥ 75 % van de Sharpe, bij lag 2 ≥ 50 %;
- plateau ≥ 0,8 en geen negatieve verstoring;
- bootstrap-ondergrens > 0;
- DSR ≥ 0,95 bij M = 24;
- PBO < 0,5.

Doctrine (`registry/preregistration.py`): een drempel wordt nooit versoepeld. Dat geldt
ook voor G5-kosten, waar v5-H1 op 0,744 strandde. Wat wel verandert, en waarom dat geen
versoepeling is:
- **G2 geldt nu in het unified-margin-account.** In v5 sloot de poort "meer notional om
  de CAGR te halen" uit. Dat was de mandaatregel van toen, en het eigenaarsbesluit van
  2026-10-09 vervangt die. De drempel zelf (15 %) blijft.

Nieuw, en alleen strenger:

| poort | maat | bindt als |
| --- | --- | --- |
| G10 | liquidaties op W_DEV | > 0 |
| G11 | liquidaties onder margestress (alts-haircut 35 %, perp-onderhoud 3 %) | > 0 |
| G12a | Sharpe onder financieringsstress | < 1,0 |
| G12b | CAGR onder financieringsstress | < 10 % |

**De batterij** is die van v5:
- verstoringen: span 14/60, drempels 10/3 en 20/8 %, band 0,25/0,75, slots 5/20;
- kosten ×0/×2/×3, +10 bp, impact- en spreadstress;
- capaciteit 10 en 50 mln (de ADV-cap schaalt mee met het boek);
- lag 2/3, ruis, ontbrekende dagen, besluiten om de 3/7 dagen;
- universum top-30/100, start- en einddata, Monte-Carlo-drawdown.

Plus:
- de financieringsstress en de optimistische financiering;
- de margestress;
- de governor uit, om te laten zien wat hij doet.

## 6. Buiten W_DEV: drie lezingen, en wat elk kan bewijzen

Elke lezing staat in haar slot vóór de data terugkomt (R7).

1. **Backcast 2020-04 .. 2020-12** (slot `holdout_lock_binance_backcast_2020.json`, nieuwe
   hypothese-id).
   - *Deels besmet:* v5 las daar H3, dus dat 2020 rijk aan carry was, is bekend.
   - Kan daarom alleen **verwerpen**: z < −1,645 tegen W_DEV, een drawdown > 25 % of
     een liquidatie.
   - Nieuw is wel het liquidatiegedrag bij hefboom in de altcoin-rally van eind 2020.
2. **Holdout 2025-07 .. 2026-09** (slot `holdout_lock_binance_um.json`).
   - *Besmet:* het fundingniveau daar is in v4/v5 gezien.
   - Alleen **verwerpen**: z < −1,645, een drawdown > 25 % of een liquidatie.
   - Een venster zonder enige positie heeft een ongedefinieerde Sharpe, en die bindt
     (de regel van v5, ongewijzigd).
3. **Vooruit vanaf 2026-10-01** (nieuw slot `holdout_lock_forward_2026_10.json`, bevroren
   met deze preregistratie, op 2026-10-09).
   - Data die bij het bevriezen niet bestond: **de enige schone toets**.
   - Gelezen met `programme_v6 forward`, eenmaal, zodra er ≥ 6 volle maanden zijn
     (vroegst 2027-04).
   - Vooraf vastgelegd:
     - *falsificatie* bij z < −1,645 tegen W_DEV, een drawdown > 25 % of een
       liquidatie;
     - *archivering* bij een Sharpe ≤ 0.
   - Alleen een schone vooruit-lezing kan het boek van papier naar kapitaal brengen.

Promotie naar **papertrading** vraagt nul bindende criteria over W_DEV, de backcast en de
holdout.

## 7. Wat dit ontwerp niet kan zien, en hoe het rapport het weegt

- **Beurs als tegenpartij** (FTX 2022). Bij één beurs staat de hele equity daar, bij
  elke hefboom. Het verlies bij een wanbetaling hangt dus nauwelijks af van de hefboom,
  wel van de spreiding over beurzen. Het rapport rekent een verwacht verlies (kans ×
  verlies) van de CAGR af, als scenario, niet als poort.
- **Auto-deleveraging.** Op een crashdag kan de winnende short worden afgebouwd, waardoor
  het spotbeen ongehedged achterblijft. Daggranulariteit ziet dat niet. Het rapport
  geeft een scenario.
- **Echte leenrentes en haircuttabellen per tier.** Die zijn niet in de data te krijgen.
  Daarom zijn ze conservatief gekozen, met stress en een poort op die stress.
- **Intraday-pad.** Twee extremen per dag zijn geen pad. De toets is conservatief: spot
  en mark worden op hun slechtste combinatie gezet, voor het hele boek tegelijk.
