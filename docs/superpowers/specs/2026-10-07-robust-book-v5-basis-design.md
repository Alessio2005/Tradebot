# Robuust boek v5: spot-perp-basiscarry

**Status:** ontwerp, vóór het bevriezen geschreven. Post-hoc na v4, en zo geboekt (M = 21).

## 1. Waarom

v2–v4 zochten een prijssignaal en vonden er geen dat buiten W_DEV standhield:

| programma | kandidaat | W_DEV-Sharpe | holdout-Sharpe |
| --- | --- | --- | --- |
| v2 | X4 trend long-flat | 0,77 | −0,11 |
| v3 | Y4 trend long-flat | 0,86 | 0,00 |
| v4 | Y5 carry + trend | 1,13 | −1,19 (gefalsificeerd) |

De ontleding van de Y5-lezing is het aanknopingspunt. Het carrydeel ontving op de holdout
**22,9 %/jaar funding** en verloor meer op de prijs. Cross-sectionele carry (long lage
funding, short hoge) is een prijsweddenschap met een carry-label: de short-kant zit in de
munten waar de longs met hefboom in zitten, en die stijgen door.

v5 oogst dezelfde funding zonder die weddenschap: per munt even veel notional long op spot
en short op de perp. Over één bar is het rendement

    a·r_spot − b·r_perp + b·f − kosten,   met a ≈ b,

dus funding min de verandering van de basis (perp/spot − 1). Economisch is dit een
vergoeding voor het leveren van hefboom aan longs: arbitragekapitaal is schaars en
gesegmenteerd (He, Manela, Ross & von Wachter 2022; Schmeling, Schrimpf & Todorov 2023,
"Crypto Carry"). Het is geen voorspelling, en het kan verdwijnen: meer arbitragekapitaal
sinds 2024 (onder meer Ethena) drukt de funding, en in bear markets is hij laag of negatief.

**Beschrijvende feiten, alleen uit TRAIN (2021–2023):**
- BTC: 30,7 / 4,2 / 7,8 %/jaar funding in 2021 / 2022 / 2023, positief op 88 % van de dagen.
- De tien munten met de hoogste 30-daagse funding: 52 / 4,8 / 12 %/jaar de dag erna.

Validate, de backcast en de holdout zijn voor dit ontwerp niet bekeken. De holdout is wel
indirect besmet: zie §6.

## 2. Data

Het perpbeen is het survivorship-vrije Binance USDT-M-paneel van v2. Het spotbeen is nieuw:
`data/binance_vision.py::sync_spot` haalt de spot-dagklines op uit dezelfde publieke bucket
en verifieert elke zip tegen zijn MD5-ETag.

- 481 van de 900 perps hebben een spotpaar, ook met prijsfactor: `1000PEPEUSDT` hoort bij
  `PEPEUSDT` × 1000. Van de 421 munten die ooit in het universum zaten zijn het er 324. De
  rest bestaat alleen als perp (aandelen-perps, nieuwe tokens) en kan niet gehedged worden.
  Ze vallen dus terecht weg.
- Manifest: `artefacts/data/binance_spot_manifest.json`, 21.199 objecten.
- Spot-tijdstempels staan sinds 2025 in microseconden; de parser zet ze om.
- Integriteit op TRAIN: de mediane |basis| per munt is 0,08 %. De dag-std is 5–7 bp voor
  BTC en ETH. Eén echte mismatch, BTCSTUSDT (een ander activum met dezelfde naam), valt
  weg via de dagregel |basis| ≤ 5 %.
- Bij het testen kwam een parserfout boven: een datarij met wetenschappelijke notatie gold
  als kopregel. Een scan van alle 64.539 echte zips vond geen enkel geval, dus geen
  bestaand paneel verandert.

## 3. Het boek (`systematic/harvest.py`)

**Besluit** (op de close van *t*, causaal, getoetst in `tests/lookahead/test_basis_causality.py`):
- carry = 365 × EWMA(span 30) van de dagfunding, alleen over genoteerde dagen;
- verhandelbaar: beide koersen aanwezig, spot-ADV ≥ 2 mln, |basis| ≤ 5 %, σ en ADV bekend;
- slots met hysterese: een vrij slot gaat naar de munt met de hoogste carry ≥ 15 %/jaar
  in het point-in-time-universum; een gehouden munt blijft zolang zijn carry > 5 %/jaar
  ligt. Een gehouden munt wordt niet verdrongen door een betere (anders churn);
- tien slots van 7 % (H1), of twee van 35 % voor BTC en ETH (H3).

**Uitvoering:**
- *Twee benen.* Gelijke hoeveelheden blijven gehedged terwijl de prijs beweegt. Er wordt
  alleen gehandeld bij instap, uitstap, of als de notional > 50 % van het doel afwijkt
  (band), of als de benen > 2 % uiteenlopen.
- *Kosten per been.* Spot: 10 bp fee (VIP0), 5 bp halve spread en √-impact op spot-ADV
  en -σ. Perp: 5,5 bp fee, 3 bp spread en impact. De P&L-identiteit wordt na elke run
  gecontroleerd.
- *Kapitaalmodel zonder portfolio margin.* 70 % spot en 30 % USDT-marge in de
  futureswallet (cross).
  - Een rally laat de wallet leeglopen. Zakt hij na de trade onder 25 % van de
    perp-notional, dan gaat het hele boek terug naar het doel.
  - Liquidatie: raakt het verlies, met elke short tegelijk op zijn dag-high, de wallet
    min 2 % onderhoudsmarge, dan is de futureswallet weg. De spot houdt zijn eigen
    rendement en de volgende bar hedget opnieuw.
  - Het echte risico is een wick: de wallet weg en de spotwinst weer verdampt.
- *Delisting.* Verdwijnt een van beide koersen, dan worden beide benen gesloten, het
  verdwenen been tegen zijn laatste koers.
- *Wat niet gemodelleerd is.*
  - Rente op idle USDT (conservatief: nul).
  - Tegenpartijrisico van de beurs (FTX-scenario).
  - Funding binnen de bar die door een liquidatie wordt gemist.
  - Wisselende onderhoudsmarges per tier.

## 4. De trials (M = 18 + 3 = 21)

| trial | inhoud |
| --- | --- |
| H1_BASIS | het boek van §3 op de top-50 |
| H2_BASIS_TREND | 50 % kapitaal in H1, 50 % in Y4 (trend long-flat, ongewijzigd uit v3), dagelijks herverdeeld via een kosteloze interne overboeking |
| H3_BASIS_MAJORS | alleen BTC en ETH |

Elke parameter in `conf/model/robust_book_v5.yaml::basis` is vooraf gekozen, met de reden
ernaast in de config. De CAGR-doelstelling geldt in het vaste kapitaalmodel. Meer notional
(portfolio margin) om haar te halen is uitgesloten, net als in v3 extra hefboom boven het
vol-doel.

## 5. Selectie en poorten

Op W_DEV gelden dezelfde poorten G1–G9 als in v3, met DSR bij de cumulatieve M = 21. De
batterij:
- verstoringen: span 14/60, drempels 10/3 en 20/8 %, band 0,25/0,75, slots 5/20, en voor
  H2 de trend-lookbacks ×0,5/×2;
- kosten ×0/×2/×3, +10 bp slippage, impact-stress, spread-stress (perp 6 bp, spot 10 bp);
- capaciteit bij 10 en 50 mln;
- vertraging lag 2/3;
- ruis op de carryschatting (±25 %, 10 seeds) en 5 % ontbrekende besluitdagen (10 seeds);
- besluiten om de 3 of 7 dagen;
- universum top-30 en top-100;
- start- en einddata, en Monte-Carlo-drawdown.

De vooraf vastgelegde robuustheidsscore kiest. Niet de hoogste Sharpe.

## 6. De twee lezingen buiten W_DEV

Beide worden uitgevoerd, wat de eerste ook laat zien (`programme_v5 oos`). Elke lezing
staat in haar slot vóór de data terugkomt.

1. **Backcast 2020-04-01 .. 2020-12-31.** Nooit door enig programma gerapporteerd; 2020
   diende tot nu toe alleen als opwarmdata voor signalen. Slot
   `artefacts/governance/holdout_lock_binance_backcast_2020.json`, gelezen via
   `validation/holdout.py::backcast_gate_slice` (het spiegelbeeld van `gate_slice`).
   Falsificatie: z < −1,645 tegen W_DEV of een drawdown > 25 %. Een niet-positieve
   Sharpe archiveert.
2. **Holdout 2025-07-01 .. 2026-09-30.** Besmet: in v4 zag ik dat de short-kant van de
   carry daar 22,9 %/jaar funding ontving, en dat gaf mede het idee voor v5. Deze lezing
   kan daarom alleen **verwerpen** (z < −1,645 of drawdown > 25 %). Een goed resultaat
   telt niet als bevestiging.

Promotie naar papertrading vraagt nul bindende criteria over W_DEV en beide lezingen.
Dat is papertrading, geen kapitaal: het enige schone bewijs dat daarna nog te halen valt,
is vooruit in de tijd, vanaf 2026-10.

## 7. Wat dit ontwerp niet kan uitsluiten

- **Regimeconcentratie.** Funding is hoog in bullmarkten (2021, H1 2024). Een kandidaat
  die alleen in 2021 verdient, valt op G3; de jaartabel staat in elk artefact.
- **Verval.** Meer arbitragekapitaal drukt de funding. De backcast (2020) en de holdout
  (2025–26) liggen aan weerszijden van W_DEV; samen zeggen ze iets over stabiliteit.
- **Executierisico tussen twee venues** (spot- en futureswallet): de daggranulariteit
  ziet geen intraday-basis.
