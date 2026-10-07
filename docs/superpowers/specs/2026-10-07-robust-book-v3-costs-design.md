# Robuust boek v3 — gecorrigeerde spread en partiële aanpassing (2026-10-07)

Status: **post-hoc na v2, en zo geboekt.** Bevroren vóór de eerste v3-run; drie nieuwe
trials in de ledger (cumulatief M = 17).

## Waarom v3 bestaat (wat v2 liet zien)

1. **Het vooraf gekozen spreadmodel was fout.** De Abdi-Ranaldo-CHL-schatter op dagbars
   gaf BTCUSDT een mediane halve spread van 17,7 bp en ETHUSDT 25 bp (het plafond). De
   werkelijke quoted spread van die perps ligt onder 1 bp. Op 24/7-dagbars meet de schatter
   volatiliteit, geen spread. Dat is te controleren zonder naar een strategieresultaat te
   kijken.
2. **Er is bruto edge, maar de omzet is te hoog.**
   * Vóór kosten haalde X2 (carry) een Sharpe van 1,13 en X5 (carry + trend) 1,49 op W_DEV.
   * Na het v2-kostenmodel bleef 0,31 en 0,33 over.
   * De omzet was 68× en 94× per jaar.

## Wat v3 verandert (en verder niets)

* **Halve spread: vast 3 bp, stress 6 bp.** De top-50 Binance-perps quoteren doorgaans
  0,1–3 bp. 3 bp is daarom conservatief voor het grootste deel van het universum en
  optimistisch voor de staart. Fees (taker 5,5 bp), impact (Y = 1, stress 2,99), funding,
  `lag = 1` en een boek van $1M blijven gelijk aan v2.
* **Partiële aanpassing (Gârleanu-Pedersen 2013).** Elke dag wordt een fractie κ = 0,2 van
  het gat tussen het gedrifte en het doelgewicht verhandeld. Het doel is het laatste
  sleeve-besluit (wekelijks voor carry, dagelijks voor trend). κ = 0,2 is een halfwaardetijd
  van ≈ 3 dagen: langzamer dan de signaalvervalsnelheid van carry, die over weken loopt.
  κ ∈ {0,1; 0,33; 1,0} zit in de verstoringsfamilie (plateau en PBO).

## Kandidaten

| Code | Sleeve | Waarom |
| --- | --- | --- |
| `Y2_CARRY` | v2-X2 | bruto edge, kostengevoelig |
| `Y4_TREND_LF` | v2-X4 | de robuuste v2-winnaar, nu onder de gecorrigeerde kosten |
| `Y5_COMBO` | X2 + X4, gelijk risico, vast (geen inclusieregel) | diversificatie: carry ⟂ trend |

X1 (XS-momentum) en X3 (trend-LS) komen niet terug. X1 faalde zelfs vóór kosten (0,60) en
is in elke deelperiode zwak. X3 wordt gedomineerd door X4.

## Poorten, vensters en boekhouding

* Universum, vensters en poorten G1–G10 zijn die van v2. De DSR geldt bij M = 17.
* De holdout leest via hetzelfde slot (`holdout_lock_binance_um.json`), één keer, voor de
  kandidaat met de hoogste robuustheidsscore.
* **Voorbehoud:** de holdout is voor de trendfamilie al gelezen in v2 (X4). Een v3-lezing
  is dus zwakker bewijs dan een maagdelijke holdout.

## Verwachting vooraf

* Ruwe schatting voor Y5 op W_DEV: een netto Sharpe van 0,8–1,1. Die rekent met de
  bruto-plus-funding uit v2, de lagere omzet en de lagere spread.
* G3 is het grootste risico: de VALIDATE-helft (2024-01 → 2025-06) was voor carry en
  trend-LS negatief.
