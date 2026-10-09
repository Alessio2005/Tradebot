# Robuust boek v7: uitvoering en financiering van het hefboomboek

**Status:** ontwerp, geschreven vóór het bevriezen en vóór elke v7-run.
- **Post-hoc na v6 (W_DEV), en zo geboekt** (M = 24 + 2 = 26, plus 52 bekende eerdere trials).
- **Vóór elke v6-lezing buiten W_DEV:** dit ontwerp kent alleen W_DEV.

Het account, de poorten, de vensters en de lezingen zijn die van v6
(`2026-10-09-robust-book-v6-levered-carry-design.md`).

## 1. Wat v6 op W_DEV liet zien

| v6 | Sharpe | CAGR | funding | financiering | impact | G5 (2× kosten / basis) |
| --- | --- | --- | --- | --- | --- | --- |
| K1 1× | 5,42 | 12,1 % | 14,2 % | −0,1 % | −1,8 % | 0,72 |
| K2 1,5× | 4,33 | 13,6 % | 20,5 % | −3,4 % | −3,0 % | 0,63 |
| K3 2× | 3,67 | 14,0 % | 24,8 % | −6,4 % | −3,6 % | 0,51 |

Alle drie binden op G2 (CAGR) en G5 (kosten). Ze hebben nul liquidaties, ook onder
margestress.

**De diagnose** (W_DEV, dezelfde configuraties, geen nieuwe trials):

1. **De impact komt van grote, zeldzame trades.**
   - Posities staan lang: een mediaan van 113 dagen, en 36 instappen in 4,5 jaar.
   - Elke instap en uitstap is een heel slot (0,1–0,2 van de equity in één altcoin) in
     één keer.
   - In het √-impactmodel kost een trade ∝ q^1,5.
   - De hedge-reset bij uiteenlopende benen triggert nooit. Zonder band daalt de impact
     maar van 1,80 naar 1,69 %/jaar (K1).
   - De hoofdbron is dus de grootte van de trade, niet het aantal.
2. **De financiering betaalt voor hefboom die niets verdient.**
   - 17 % van de notional zat in munten met carry onder de leenrente
     `max(8 %, BTC-carry)`. Die munten verdienden ~16 %/jaar.
   - Terwijl er geleend werd, kostte de lening gemiddeld 30–42 %/jaar (2021).
   - Hefboom over het hele boek verspreiden leent dus ook voor de laagste carry.

## 2. Wat v7 verandert, en verder niets

1. **Gespreide uitvoering.**
   - Elke tranche van 1/slots (10 % van de equity) gaat in **vijf gelijke dagstappen** in
     of uit.
   - Een resize buiten de band beweegt per dag hooguit één stap.
   - Het doel volgt intussen het laatste besluit.
   - *Waarom 5:* 5 = 1/κ met κ = 0,2, de partiële aanpassing die v3 al vooraf koos
     (halfwaardetijd ~3 dagen, trager dan het verval van carry, dat in weken loopt).
   - In het √-model kost dit √5 = 2,2× minder impact per tranche. Bij propagator-impact
     (gedeeltelijk permanent, langzaam vervallend) is de winst kleiner; dat is de
     modelonzekerheid.
   - **Nooit gespreid:**
     - een verdwenen koers (meteen dicht);
     - een liquidatie;
     - de governor op het boek zelf (drift buiten de grens krimpt meteen).

     De governor geldt bovendien op het doel: past het volledige doel niet, dan krimpt
     het doel.
2. **Een gefinancierde tranche in plaats van uniforme hefboom.**
   - Elke munt met een basisslot (1/slots, uit eigen equity) krijgt een tweede tranche
     van 1/slots:
     - die gaat **aan** bij carry ≥ leenrente + 10 pp;
     - en blijft aan zolang carry ≥ leenrente.
   - *Waarom 10 pp:* het is de hysterese van de slotregel van v5 (instap 15 %, uitstap
     5 %).
   - Hefboom wordt zo alleen gebruikt waar hij verdient wat hij kost. Bij de rente van
     2021 (~31 %) vraagt een tranche ~41 % carry.
   - Onder de financieringsstress (G12) rekent de regel met de stressrente: dan wordt er
     minder geleend.
   - Maximaal 2× per been, dus bruto 4 = de mandaatcap.

## 3. De trials

| trial | basis | gefinancierde tranche | spreiding |
| --- | --- | --- | --- |
| S1_CARRY_PM_SLICED | 1× (= v6-K1) | — | 5 dagen |
| S2_CARRY_PM_TRANCHE | 1× | +1/slots per munt, carry ≥ rente + 10 pp | 5 dagen |

- S1 meet de uitvoering alleen. S2 voegt de gefinancierde tranche toe.
- **Selectie:** dezelfde regel als v6. De laagste hefboom die alle poorten haalt, anders
  de robuustheidsscore (en dan gearchiveerd).

## 4. Poorten, batterij, lezingen

- **De poorten zijn letterlijk die van v6.** Dat wordt getoetst in
  `tests/unit/test_programme_v7_smoke.py`:
  - G1–G9 van v5 (ook G5: dubbele kosten houden ≥ 75 % van de Sharpe);
  - G10–G12 van v6;
  - DSR bij M = 26.
- **De batterij** is die van v6, met in de verstoringsfamilie (plateau en PBO) ook de
  nieuwe parameters:
  - spreiding 3 en 10 dagen;
  - voor S2 de spread 5 en 15 pp.

  Ter informatie, geen poort: dezelfde kandidaat ongespreid (de attributie van de
  uitvoering).
- **De lezingen** zijn die van v6:
  - backcast 2020 en holdout 2025–26, besmet: kunnen alleen verwerpen;
  - het vooruit-sample vanaf 2026-10-01 (het slot van v6, één lezing per hypothese-id),
    na ≥ 6 volle maanden.

## 5. Wat dit niet oplost

- **Beschikbaarheid van carry.**
  - In 2022 en 2025H1 was er weinig funding. Daar verdient ook v7 bijna niets.
  - De CAGR over W_DEV blijft gedreven door 2021 en 2024.
- **De leenrente is een model** (geen data). De stress (12 %, 1,5× BTC-carry) moet de
  doelstelling grotendeels houden (G12).
- **Iteratie op W_DEV.** v6 en v7 zijn beide op W_DEV ontworpen. M telt dat, de DSR
  deflateert ervoor, en geen drempel is versoepeld. Maar een CAGR die na twee rondes
  ontwerp over de 15 % gaat, is zwakker bewijs dan een vooraf voorspelde. Het oordeel
  daarover ligt bij het vooruit-sample.
