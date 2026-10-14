# Robuust boek v8: een sprongrisicogrens voor de carry-screen

**Status:** ontwerp, geschreven vóór het bevriezen en vóór elke v8-run.
- **Post-hoc na de holdout-lezingen van v6 en v7, en zo geboekt** (M = 26 + 1 = 27, plus
  52 bekende eerdere trials).
- **Dit ontwerp kent de holdout 2025–26:** die is daarmee voor de hele carryfamilie
  verbruikt. Het schone bewijs is het vooruit-sample. Het slot daarvan is bevroren op
  2026-10-09, vóór alle v6–v8-lezingen.

Account, poorten, vensters en lezingen zijn die van v6/v7. Het boek is v7-S2.

## 1. Wat de holdout liet zien

| holdout 2025-07 .. 2026-09 | Sharpe | CAGR | max DD | funding | impact |
| --- | --- | --- | --- | --- | --- |
| v6-K1 | −3,08 | −8,6 % | 10,6 % | +0,1 % | −9,0 %/15 mnd |
| v7-S2 | −2,49 | −5,7 % | 7,1 % | −0,75 %/jaar | −3,3 %/jaar |

**Het mechanisme: adverse selectie van de carry-screen.**
- In de holdout lag de BTC/ETH-carry onder de instap van 15 %. De screen liet daarom
  alleen nog pump-and-dump-microcaps door.
- Bij instap hadden die al een dag-σ van:

  | munt | dag-σ |
  | --- | --- |
  | HIFI | 1,05 |
  | BAKE | 0,57 |
  | TUT | 0,50 |
  | ALPINE | 0,29 |
  | BANANAS31 | 0,24 |

- Intraday bewogen ze ±50–85 %. De basis liep op tot 11,8 % en BAKE-spot verdween midden
  in de positie.
- Hun "carry" is een vergoeding voor sprongrisico, geen hefboompremie.
- Elke 50 %-dagbeweging trok de band open, en een 1 %-ADV-trade kostte daar ~10 % impact.
- Munten met σ ≤ 15 % in dezelfde periode (FORM, KITE, DEXE, TUT in oktober 2025) waren
  rond nul.

**Op W_DEV bestond dit regime niet.** De hoogste dag-σ die K1 ooit hield was 0,22. Munten
boven 15 % waren 1,6 % van de notional.

## 2. De grens, afgeleid uit wat al bevroren was

Een volledige rondgang (in en uit, twee benen) op de 1 %-ADV-cap, gespreid over vijf
dagen, kost in het √-impactmodel (η = 1):

    4 · η · σ · √(0,01 / 5) = 0,18 · σ   per eenheid notional.

Een kwartaal carry op de instapdrempel is 15 % / 4 = 3,75 %.
- **Break-even:** σ = 0,0375 / 0,18 = 0,21. Afgerond naar beneden geeft dat
  **σ_max = 0,20 per dag** (≈ 380 % per jaar, zo'n 7× BTC).
- **Waarom het een plafond is:** voor een positie onder de cap is de impact kleiner, dus
  de grens is ruim. De hedge-risico's die het model niet prijst (basis, delisting,
  liquidatie) groeien ook met σ.

**De regel:** een munt waarvan de dag-σ boven 0,20 ligt (het hoogste van spot en perp,
EWMA met λ = 0,94, bekend op de close van *t*) is niet verhandelbaar.
- Geen instap.
- Een gehouden munt gaat eruit, in vijf stappen.
- De gefinancierde tranche hangt aan het basisslot en gaat mee.

Implementatie: `harvest_targets(max_sigma=…)`. Zonder grens is het besluit exact dat van
v5–v7, en v6 en v7 reproduceren bit-identiek.

## 3. Eén trial

| trial | boek | grens |
| --- | --- | --- |
| T2_CARRY_TRANCHE_JUMPCAP | v7-S2 | dag-σ ≤ 0,20 |

**De batterij** is die van v7. In de verstoringsfamilie zit ook de grens zelf (0,15 en
0,30). Ter informatie: dezelfde kandidaat zonder grens (= v7-S2) en ongespreid.

**De poorten** zijn letterlijk die van v6/v7 (getoetst). DSR bij M = 27.

## 4. Wat elke lezing kan bewijzen

- **W_DEV:** de grens bindt daar nauwelijks. T2 hoort dus ≈ v7-S2 te zijn. Wijkt hij
  sterk af, dan doet de grens iets anders dan bedoeld.
- **Backcast 2020:** besmet. Kan alleen verwerpen.
- **Holdout 2025–26:** verbruikt, want deze periode heeft het ontwerp gestuurd.
  - Hij kan alleen nog verwerpen: als het verlies ondanks de grens blijft.
  - Een goed resultaat bevestigt niets.
  - Hij kan ook vlak uitvallen: zonder carry houdt het boek niets. De bevroren regel laat
    een ongedefinieerde Sharpe binden. Dat oordeel blijft staan, en het rapport legt het
    uit.
- **Vooruit vanaf 2026-10-01:** de enige toets die iets kan bevestigen. Eén lezing per
  hypothese-id, na ≥ 6 volle maanden.
