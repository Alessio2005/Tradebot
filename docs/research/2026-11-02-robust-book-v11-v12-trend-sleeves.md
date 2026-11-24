# Robuust boek v11 en v12: trend-ensemble en long/short-trend naast de carry (2026-11-02)

Vervolg op `2026-11-02-robust-book-v10-flow-sleeve.md`. Dezelfde invoer, boekhouding,
kosten en criteria als v10; alleen het signaal verschilt. De live dienst is niet aangeraakt.

## Samenvatting

- **Alle vijf de nieuwe trials zijn verworpen** (M 36 → 41). Over v10–v12 zijn dat negen
  trials, en geen enkele haalt alle vijf de criteria.
- **Het beste boek tot nu toe is v12-H1** (trend long/short: long spot boven de SMA100,
  short perp eronder). Het is **elk jaar positief**, en verdient in de holdout ruim boven
  U1. Het faalt op één criterium: de Sharpe boven de T-bill (0,89 tegen ≥ 1,0). De kosten
  vreten het op: 10,5 % over de cyclus, door het dagelijks omdraaien rond het gemiddelde.
- **Het trend-ensemble uit de literatuur (v11) werkte hier slechter** dan de ene
  SMA100-regel. Ook met een vol-filter bleef 2022 een verliesjaar.

| | U1 | v10-F3 trend long | v11-G1 ensemble | **v12-H1 trend L/S** | v12-H2 ensemble L/S |
| --- | --- | --- | --- | --- | --- |
| CAGR | 8,0 % | 12,7 % | 9,3 % | **11,1 %** | 7,8 % |
| Sharpe boven T-bill (maand) | 1,27 | 1,24 | 1,06 | 0,89 | 0,71 |
| max drawdown | 0,3 % | 7,0 % | 7,4 % | 9,9 % | 8,7 % |
| holdout boven T-bill | −0,8 % | +4,1 % | −1,1 % | **+9,3 %** | −2,1 % |
| slechtste jaar | +1,3 % | −5,1 % | −6,1 % | **+1,1 %** | +0,6 % |
| kosten (cyclus) | 0 | 6,1 % | 2,1 % | 10,5 % | 4,8 % |

## 1. v11: trend-ensemble (3 trials, alle verworpen)

Literatuur: time-series momentum over meerdere horizonnen (Liu & Tsyvinski 2021, RFS;
Zarattini, Pagani & Barbon 2025, "Catching Crypto Trends") en vol-management (Moreira &
Muir 2017, JF).

- Score = het deel van de periodes {20, 40, 80, 160, 320} d met een positief rendement.
- Wekelijks herbalanceren (zondag); uitstappen elke dag.
- G2 zet de score op 0 als de vol van 30 dagen boven zijn 80e percentiel van het jaar
  staat. G3 is G1 op halve sterkte.

| trial | CAGR | Sharpe (maand) | max DD | 2022 | oordeel |
| --- | --- | --- | --- | --- | --- |
| G1 ensemble 10 % | 9,3 % | 1,06 | 7,4 % | −6,1 % | ✗ CAGR, holdout, slechtste jaar |
| G2 + vol-filter | 8,8 % | 1,01 | 8,0 % | −6,7 % | ✗ CAGR, slechtste jaar |
| G3 ensemble 5 % | 8,4 % | 1,18 | 3,2 % | −2,4 % | ✗ CAGR, holdout |

Twee lessen:
- De periode van 320 dagen heeft bijna een jaar historie nodig. Het ensemble stond daardoor
  in 2020 vlak, terwijl de enkele SMA100 daar 12,5 % verdiende. Dat is een ontwerpfout die
  ik pas na de run zag; hem herstellen zou een nieuwe trial zijn.
- Het vol-filter hielp niet: in 2022 was de vol niet extreem hoog, de prijs zakte gewoon.

## 2. v12: long/short-trend (2 trials, beide verworpen)

v10 en v11 lieten zien dat elke long-only sleeve in 2022 5–8 % verloor. De trendliteratuur
haalt het crisisrendement juist uit de short-kant (Moskowitz, Ooi & Pedersen 2012). Op
Kraken EU kan dat via de perp. De short ontvangt de funding; de Binance-funding is de
proxy.

| criterium | drempel | H1 SMA L/S | H2 ensemble L/S |
| --- | --- | --- | --- |
| CAGR boven U1 | ≥ +2 pp | +3,2 ✓ | −0,2 ✗ |
| Sharpe boven T-bill (maand) | ≥ 1,0 | 0,89 ✗ | 0,71 ✗ |
| max drawdown | ≤ 15 % | 9,9 % ✓ | 8,7 % ✓ |
| holdout boven U1 | > 0 | +10,1 pp ✓ | −1,2 pp ✗ |
| slechtste jaar t.o.v. U1 | ≤ 5 pp lager | 0,2 ✓ | 0,7 ✓ |

Per jaar, H1: 2020 +19,9 %, 2021 +24,4 %, 2022 +1,1 %, 2023 +6,2 %, 2024 +3,9 %,
2025 +10,5 %, 2026 (t/m sep) +8,1 %.

De sleeve van H1 verdient vooral in de holdout (2025 +7,0 %, 2026 +5,7 %), toen BTC en
ETH zakten en de carry stillag. In 2021 en 2024 kost hij geld (−1,0 % en −3,7 %): de short
stapte in correcties in en werd uitgestopt.

## 3. Wat dit zegt

1. **Een long-only richtings-sleeve zonder short raakt 2022 altijd.** "Elk jaar winst"
   vraagt een short-kant, of geen richting.
2. **H1 heeft de vorm die de eigenaar zoekt** (elk jaar positief, onafhankelijk van de
   carry, verdient juist als de carry stilligt). Hij faalt op de Sharpe, en dat komt door
   de kosten: met de helft van de kosten is de Sharpe 0,98, zonder kosten 1,08. Een
   hysteresisband rond het gemiddelde, of beide benen via de perp (8 bp in plaats van
   25 bp), zijn de voor de hand liggende remedies. Maar elke aanpassing na het lezen is
   een nieuwe trial: M staat op 41, en de kans dat het beste van 41 trials toeval is,
   wordt groot.
3. **De volgende stap hoort het vooruit-sample te zijn, niet een tiende backtest.** H1 en
   F3 als shadow naast U1, zonder geld, met de regels zoals ze nu zijn bevroren.

## Reproduceren

```bash
python -I -m tradebot.systematic.programme_v11 run   # artefacts/research/robust_book_v11/
python -I -m tradebot.systematic.programme_v12 run   # artefacts/research/robust_book_v12/
```
