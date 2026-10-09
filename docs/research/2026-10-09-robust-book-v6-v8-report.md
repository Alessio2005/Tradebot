# Robuust boek v6–v8 — hefboom, uitvoering en sprongrisico (2026-10-09)

Vervolg op `2026-10-07-robust-book-v2-v5-report.md`. De regels zijn dezelfde:
- elke trial staat in de ledger vóór de run;
- elke preregistratie is bevroren vóór de meting;
- elke lezing buiten W_DEV is geregistreerd vóór de data terugkomt;
- geen drempel is versoepeld.

Nieuw is het mandaat. De eigenaar staat sinds 2026-10-09 **hoog risico** toe ("high risk,
high reward"). Daarmee valt het besluit dat v5 openliet: hefboom om de CAGR te halen. De
rest van het eigen-kapitaalmandaat (`conf/risk/default.yaml`) blijft staan:
- bruto ≤ 4,0;
- ADV-participatie ≤ 1 %;
- max drawdown 25 %.

## Samenvatting

- **Het beste boek is v8-T2.** Het is de spot-perp-basiscarry van v5, met daarbij:
  - een unified-margin-account;
  - een gefinancierde hefboomtranche die alleen aanstaat waar de carry zijn leenrente
    verdient;
  - gespreide uitvoering;
  - een sprongrisicogrens.

  | v8-T2 | netto Sharpe | CAGR | max DD |
  | --- | --- | --- | --- |
  | W_DEV 2021-01 .. 2025-06 (alle 16 poorten vrij) | 5,33 | 15,9 % | 1,5 % |
  | backcast 2020-04 .. 2020-12 | 9,03 | 24,7 % | 0,9 % |
  | holdout 2025-07 .. 2026-09 (verbruikt) | −2,0 | −1,1 % | 1,4 % |
  | **hele cyclus 2020-04 .. 2026-09 (6,5 jaar)** | **5,05** | **13,4 %** | **2,0 %** |

- **Netto Sharpe > 1: ja, robuust**, in elk venster met carry. Maar de backtest-Sharpe is
  niet de echte risicomaat. Bij één beurs en ~3 % kans per jaar op een faillissement zakt
  de werkelijke Sharpe naar ~0,8. Met twee beurzen is het ~1,7, met drie ~2,5 (§6).
  **Meerdere beurzen zijn een voorwaarde voor de doelstelling.**
- **Netto CAGR > 15 %: alleen in jaren met carry.**

  | jaar | rendement |
  | --- | --- |
  | 2020 | +18 % |
  | 2021 | +61 % |
  | 2022 | +0,2 % |
  | 2023 | +4 % |
  | 2024 | +16 % |
  | 2025–26 | −0,4 / −1,0 % |

  Over de hele cyclus is het 13,4 %. Dat is twee punten onder het doel, maar met een
  drawdown van 2 %.
- **De belangrijkste vondst is een stille killer in de carry-screen** (v5–v7).
  - Zonder brede carry (2025–26) liet de screen alleen nog pump-and-dump-microcaps door,
    met een dag-σ van 24–105 %. Hun "carry" is een vergoeding voor sprongrisico.
  - v6-K1 verloor daar −8,6 % CAGR (z = −14,9) en v7-S2 −5,7 % (z = −16,7). Beide zijn
    **gefalsificeerd**.
  - De sprongrisicogrens van v8 brengt dat terug naar −1,1 % (DD 1,4 %).
  - Formeel blijft ook v8 gefalsificeerd volgens de bevroren z-regel, en de holdout is nu
    voor de hele carryfamilie verbruikt.
- **Hefboom zelf levert weinig op.** De rente die je betaalt, beweegt mee met de carry
  die je oogst. Uniforme hefboom (v6) voegde van 1× naar 2× maar +1,9 % CAGR toe, tegen
  een Sharpe van 5,4 → 3,7. Pas gericht ingezet (v7/v8: alleen waar carry ≥ rente +
  10 pp) is hij de moeite waard.
- **Het oordeel ligt nu vooruit.** Het vooruit-slot (vanaf 2026-10-01) is bevroren op
  2026-10-09, vóór alle lezingen in dit rapport. Het is de enige schone toets die nog
  bestaat. De lezing volgt in april 2027 (§8).

M is nu 27 in deze ledger (+ 52 bekende eerdere trials = 79).

## 1. Uitgangspunt, en waarom geen prijssignaal

v2–v5 lieten twee dingen zien:
- **Prijssignalen houden geen stand.** Trend, momentum en cross-sectionele carry haalden
  buiten W_DEV −0,11, 0,00 en −1,19.
- **De basiscarry is de enige edge.** Long spot en short perp haalden Sharpe 5,7–6,8 op
  W_DEV en 11,5 op de ongemeten backcast.

"High risk" betekent daarom hier: meer risico nemen in de edge die er is, niet risico
kopen zonder premie. Het risico van de carry zit niet in de vol (1–3 %/jaar). Het zit in
de staart: financiering, liquidatie, sprongrisico en de beurs. Elk daarvan is hieronder
expliciet gemodelleerd of als scenario doorgerekend.

**Data:** de v5-data is opnieuw opgehaald en reproduceert bit-identiek:
- de objects- en panelhashes en preregistratie 4df1a5d0;
- de Sharpes H1 5,657 / H2 1,565 / H3 6,762.

Nieuw zijn de mark-price-dagbars van 895 van de 900 perps (MD5-geverifieerd,
`artefacts/data/binance_mark_manifest.json`). Binance liquideert op de mark price, niet op
de last price.

## 2. v6 — de carry met hefboom in een unified-margin-account

Ontwerp: `docs/superpowers/specs/2026-10-09-robust-book-v6-levered-carry-design.md`,
preregistratie b62c8bc3, M = 24. Het boek staat in `systematic/leverage.py`:
- **onderpand:** spot met haircut (majors 5 %, alts 20 %);
- **lening:** USDT boven de eigen equity, tegen `max(8 %, BTC-carry)`;
- **liquidatietoets:** op de intraday mark price, voor het hele boek in een up- en een
  down-staat;
- **governor:** uniMMR ≥ 3 en bruto ≤ 4;
- **ADV-cap:** 1 %, tegen de lopende equity.

Die laatste was eerst tegen de begin-AUM geschreven. Dat is vóór de run gevonden en
hersteld, want na een verdubbeling van de equity zat het boek anders stilletjes op 2 %.

| v6, W_DEV | Sharpe | CAGR | MDD | funding | financiering | impact | 2× kosten / basis |
| --- | --- | --- | --- | --- | --- | --- | --- |
| K1 1× | 5,42 | 12,1 % | 2,3 % | 14,2 % | −0,1 % | −1,8 % | 0,72 |
| K2 1,5× | 4,33 | 13,6 % | 3,5 % | 20,5 % | −3,4 % | −3,0 % | 0,63 |
| K3 2× | 3,67 | 14,0 % | 4,9 % | 24,8 % | −6,4 % | −3,6 % | 0,51 |

- Alle drie binden op G2 (CAGR) en G5 (kosten). De score koos K1.
- Er waren nul liquidaties, ook onder margestress (alts-haircut 35 %).

**Wat dit zegt:** hefboom heeft een steil dalend meerrendement. In 2021 kostte de lening
~31–38 % per jaar. Het boek leende ook voor munten die minder opbrachten (17 % van de
notional). Daarnaast kwam de impact van grote, zeldzame trades:
- een heel slot in één altcoin in één keer;
- posities met een mediane looptijd van 113 dagen.

**Gemeten en weerlegd:** mijn eerste vermoeden (hedge-resets, drift-band) klopte niet.
De hedge-reset triggert nooit, en de band verklaart 0,1 %-punt van de 1,8 %.

## 3. v7 — uitvoering en financiering

Ontwerp: `…-v7-execution-design.md`, preregistratie 04af949b, M = 26. Post-hoc na v6
(W_DEV), vóór elke v6-lezing buiten W_DEV. De wijzigingen:

1. **Gespreide uitvoering.** Elke tranche gaat in vijf gelijke dagstappen in en uit (κ =
   0,2, zoals v3 koos). In het √-impactmodel kost een trade q^1,5, dus de impact zakt
   met √5.
2. **Een gefinancierde tranche.** Een tweede tranche van 1/slots per munt staat alleen
   aan zolang carry ≥ leenrente + 10 pp (en blijft aan tot carry onder de rente zakt).

| v7, W_DEV | Sharpe | CAGR | MDD | 2× kosten / basis | financieringsstress | poorten |
| --- | --- | --- | --- | --- | --- | --- |
| S1 1×, gespreid | 6,27 | 13,3 % | 1,0 % | 0,89 | — | bindt G2, G9 |
| **S2 tranche, gespreid** | **5,35** | **16,7 %** | **1,1 %** | **0,86** | CAGR 14,3 % | **alle vrij** |

- **Spreiding halveert de impact** (1,80 → 0,86 %/jaar).
- **Zonder spreiding** zou S2 een Sharpe van 3,87 en een CAGR van 14,1 % hebben
  (attributie, geen trial).
- Voor het eerst in dit programma haalt een kandidaat op W_DEV beide doelen en alle
  poorten.

## 4. De lezingen buiten W_DEV, en de stille killer

| | backcast 2020 | holdout 2025-07 .. 2026-09 |
| --- | --- | --- |
| v6-K1 | Sharpe 11,19, CAGR 21,7 %, MDD 0,4 % | **Sharpe −3,08, CAGR −8,6 %, MDD 10,6 % (z = −14,9)** |
| v7-S2 | Sharpe 9,03, CAGR 24,7 %, MDD 0,9 % | **Sharpe −2,49, CAGR −5,7 %, MDD 7,1 % (z = −16,7)** |

Beide zijn **gefalsificeerd**, met nul liquidaties.

**De forensiek** (op lezingen die al geregistreerd en verbruikt waren):
- Het verlies kwam niet uit funding of prijs. Bij v6-K1 was dat bruto −1,6 % en funding
  +0,1 %, tegen **impact −9,0 % over 15 maanden** bij een gemiddelde spotpositie van ~6 %
  van de equity.
- De BTC/ETH-carry lag de hele holdout onder de instap van 15 %. De enige munten die de
  screen nog doorliet, waren pump-and-dump-microcaps. Hun dag-σ bij instap:

  | munt | dag-σ |
  | --- | --- |
  | HIFI | 1,05 |
  | BAKE | 0,57 |
  | TUT | 0,50 |
  | ALPINE | 0,29 |
  | BANANAS31 | 0,24 |

- Intraday bewogen ze ±50–85 %. De basis liep op tot 11,8 %, BAKE-spot verdween midden
  in de positie, en bij v7 werd de funding na instap zelfs negatief.
- Elke 50 %-dagbeweging trok de band open. Het boek handelde dus bijna dagelijks in een
  munt waar 1 % van de ADV ~10 % impact kost.

**Adverse selectie:** in een regime zonder brede carry zijn de enige munten met "carry"
die waarin de carry een vergoeding is voor sprongrisico. Op W_DEV bestond dit niet: de
hoogste dag-σ die het boek ooit hield was 0,22. Het is een regimebreuk, en precies het
soort fout dat alleen een echte out-of-samplelezing laat zien.

## 5. v8 — een sprongrisicogrens

Ontwerp: `…-v8-jump-risk-design.md`, preregistratie 1b771576, M = 27. Post-hoc na de
holdout-lezingen, en zo geboekt.

**De grens is afgeleid uit wat al bevroren was**, niet gefit:
- een volledige rondgang op de 1 %-ADV-cap, gespreid over vijf dagen, kost
  `4·σ·√(0,01/5) = 0,18·σ`;
- een kwartaal carry op de instapdrempel is 3,75 %;
- break-even ligt bij σ = 0,21, afgerond **0,20 per dag**.

Boven de grens geen instap, en een gehouden munt gaat eruit.

| v8-T2 | Sharpe | CAGR | MDD | opmerking |
| --- | --- | --- | --- | --- |
| W_DEV | 5,33 | 15,9 % | 1,5 % | alle 16 poorten vrij |
| backcast 2020 | 9,03 | 24,7 % | 0,9 % | |
| holdout 2025–26 | −2,00 | −1,1 % | 1,4 % | z = −22,2 → gefalsificeerd (bevroren regel) |

**De W_DEV-poorten van T2 in detail:**
- DSR 1,00 bij M = 27, en ook bij 79;
- PBO 0,17;
- bij dubbele kosten blijft 0,85 van de Sharpe over, met een CAGR van 13,3 %;
- financieringsstress: Sharpe 5,40, CAGR 13,8 %;
- margestress: nul liquidaties, minimale headroom 0,70× de onderhoudsmarge;
- de grens zelf op 0,15 en 0,30 geeft Sharpe 5,31 / 5,35 (een plateau);
- capaciteit: Sharpe 5,36 en CAGR 7,9 % bij $10M; Sharpe 4,69 en CAGR 3,2 % bij $50M.
  De ADV-cap bindt dan.

**De grens doet wat hij moet doen.** Het holdout-verlies krimpt van −8,6 / −5,7 % naar
−1,1 %, de drawdown van 10,6 naar 1,4 %. Wat overblijft is ~0,9 %/jaar impact van
instappen in munten onder de grens waarvan de carry niet aanhield. Zonder carry verdient
het boek niets, en betaalt het een beetje om te kijken.

Formeel is het oordeel "gefalsificeerd". De bevroren z-toets meet de holdout-Sharpe tegen
de W_DEV-Sharpe van 5,3, en in een vlak, licht negatief jaar is dat een enorme afwijking.
Dat oordeel laat ik staan. Inhoudelijk is het geen weerlegging van de carry maar van
**elk-jaar-rendement**: de carry is er niet altijd.

**Ik itereer niet verder op de holdout.** Elke volgende aanpassing zou op een verbruikte
periode gefit zijn.

## 6. Wat de backtest niet ziet

**Beurs als tegenpartij.** Bij één beurs staat de hele equity daar, bij elke hefboom.
Hieronder Sharpe inclusief een sprongverlies (kans p per jaar, verlies LGD van de
equity), met μ = 16,7 % en σ = 2,9 % (v7-S2, W_DEV):

| p per jaar | 1 beurs (LGD 100 %) | 2 beurzen (50 %) | 3 beurzen (33 %) |
| --- | --- | --- | --- |
| 1 % | 1,51 | 2,81 | 3,71 |
| 3 % | **0,79** | 1,69 | 2,46 |
| 5 % | 0,53 | 1,26 | 1,92 |

FTX (2022) en Mt.Gox laten zien dat ~1–3 % per jaar voor een grote beurs geen extreme
aanname is. **Conclusie: met één beurs is Sharpe > 1 niet verdedigbaar; met twee à drie
wel.**

**Auto-deleveraging.**
- Op een crashdag kan de winnende short worden afgebouwd, waarna de spot ongehedged
  staat.
- Scenario: drie slots van 0,2 met 15 % verdere daling vóór de herhedge kosten 9 % van de
  equity. Bij het gemiddelde boek (~0,13 per munt) is dat ~6 %.
- Zeldzaam, maar in de orde van de jaaropbrengst.

**USDT als rekeneenheid.**
- De equity is in feite USDT. Een depeg van 5 % kost 5 % in dollars, bij elke hefboom.
- Mitigatie: idle onderpand in USDC of fiat. De USDT-lening is een gedeeltelijke hedge.

**Leenrente en haircuts per tier** staan niet in de data. Ze zijn conservatief gekozen,
met een poort op hun stress (G11, G12).

**Intraday-pad.** De liquidatietoets zet spot en mark op hun slechtste dagcombinatie
voor het hele boek tegelijk. Dat is strenger dan de werkelijkheid, maar een dag is geen
pad.

## 7. Oordeel over de doelstelling

| | netto Sharpe | netto CAGR | max DD |
| --- | --- | --- | --- |
| v8-T2, hele cyclus 2020-04 .. 2026-09 (backtest) | 5,05 | 13,4 % | 2,0 % |
| idem, carryrijke jaren (2020, 2021, 2024) | 7,7–9,8 | 16–61 % | ≤ 1,6 % |
| idem, droogtejaren (2022, 2023, 2025–26) | −2,0 … 3,7 | −1 … +4 % | ≤ 1,4 % |

**Wat het doel vraagt, en wat dit boek levert:**
1. **Sharpe > 1:** ja, met een ruime marge in de backtest. De echte grens is het
   tegenpartijrisico. Met twee à drie beurzen blijft het boven 1; met één niet.
2. **CAGR > 15 %:** in carryjaren ruim; over de hele cyclus 13,4 %. De beperking is hoe
   vaak er carry is, niet hoe hard het boek die oogst. In de droogte (2022, 2025–26)
   staat het boek terecht grotendeels in cash.
3. **Wat de CAGR over de cyclus nog kan dragen, zonder nieuwe edge te verzinnen:**
   - **rente op het onbelegde kapitaal.** In de droogte staat ~50–100 % van de equity in
     cash. Tegen 4–5 % (USDC/tokenized T-bills) is dat grofweg +1,5–2,5 %/jaar over de
     cyclus. Dat is geen alfa, maar het brengt de cyclus rond 15 %. Ik heb het niet
     gemeten: de rentereeks is hier niet te krijgen.
   - **meer beurzen** (Bybit, OKX). Meer munten met carry, en tegelijk de verlaging van
     LGD uit §6.
   - **een hoger hefboomplafond dan bruto 4.** Dat kan, maar het is de beslissing van de
     eigenaar (`conf/risk/default.yaml`) en een nieuwe trial. De gefinancierde tranche
     zou het alleen gebruiken waar carry > rente + 10 pp.

**Wat ik níet aanraad:** een prijssleeve om de droogtejaren op te vullen. Elke
prijsgedreven kandidaat in v1–v4 stortte buiten W_DEV in. Hoog risico zonder edge is geen
hoog rendement.

## 8. Wat nu: papier, en de vooruit-lezing

1. **Papertraden van v8-T2 vanaf 2026-10** op echte fills, zonder kapitaal.
   - Het boek en alle parameters liggen vast in `conf/model/robust_book_v8.yaml`.
   - Aandachtspunten: de werkelijke spreads en impact in alt-spot, de leenrente, de
     haircut-tiers, en de executie tussen spot en perp.
2. **De vooruit-lezing, eenmaal, vanaf april 2027** (≥ 6 volle maanden na 2026-10-01):

   ```text
   python -I -m tradebot.data.binance_vision forward 2027-03     # eigen panelen + manifesten
   python -I -m tradebot.systematic.programme_v8 forward
   ```

   - De bevroren panelen blijven onaangeroerd. De lezing weigert als de W_DEV-Sharpe op
     de nieuwe panelen niet exact reproduceert.
   - Vooraf vastgelegd:
     - *falsificatie* bij z < −1,645, een drawdown > 25 % of een liquidatie;
     - *archivering* bij een Sharpe ≤ 0.
   - **Let op, dit geldt ook vooruit:** een vlak halfjaar zonder carry valt onder dezelfde
     z-regel. Een droogte vooruit falsificeert dus formeel, zoals de holdout. Dat is
     streng, en zo bevroren.
   - Alleen een vooruit-lezing zonder bindend criterium is een basis voor kapitaal.
3. **Multi-venue-data** (Bybit, OKX) ontgrendelen. Die hosts zijn in deze omgeving
   geblokkeerd (403); zie de netwerkinstellingen van de omgeving. Daarna volgt v9: de
   carry over meerdere beurzen, met de LGD-winst uit §6.
4. **Een rentereeks voor idle USDC/T-bills**, voor de cyclus-CAGR uit §7.

## 9. Governance

| | preregistratie | trials | M na bevriezen | W_DEV | oordeel |
| --- | --- | --- | --- | --- | --- |
| v6 | b62c8bc3 | 3 | 24 | geen kandidaat vrij (G2, G5) | K1 gefalsificeerd (holdout) |
| v7 | 04af949b | 2 | 26 | S2 alle poorten vrij | S2 gefalsificeerd (holdout) |
| v8 | 1b771576 | 1 | 27 | T2 alle poorten vrij | T2 gefalsificeerd (holdout, bevroren z-regel) |

- **De volgorde is zichtbaar in de geschiedenis:** spec → code en tests → bevriezen →
  W_DEV-run → lezingen.
- **v7 is ontworpen vóór de v6-lezingen.** v8 kwam erna, en is zo geboekt.
- **Het vooruit-slot** (`holdout_lock_forward_2026_10.json`) is bevroren met v6 op
  2026-10-09, vóór alle lezingen. Het wordt gedeeld door de carryfamilie, met één lezing
  per hypothese-id.
- **Reproduceerbaarheid:** v5, v6 en v7 reproduceren bit-identiek na elke refactor.
- **Code:**
  - `systematic/leverage.py`: het boek;
  - `systematic/programme_v6.py`: de gedeelde orkestratie (`Programme`);
  - `programme_v7.py` en `programme_v8.py`: alleen wat ze onderscheidt;
  - `data/binance_vision.py`: de mark price en het vooruit-sample.

## 10. Resterende zwakke punten

- **Iteratie op W_DEV.** v6 → v7 → v8 zijn drie rondes ontwerp op hetzelfde venster. M
  telt ze en de DSR deflateert. Maar een CAGR die na drie rondes net over de 15 % komt
  (15,9 %), is zwakker bewijs dan een vooraf voorspelde.
- **Een dunne marge op G2,** en een CAGR die leunt op 2021 (+61 %).
- **Kosten.** Spot-spreads en impact (Y = 1) zijn aannames. Dubbele kosten laten de
  Sharpe op 4,5 en de CAGR op 13,3 %.
- **Daggranulariteit.** De intraday-basis, het tijdstip van de funding-afrekening en de
  executie tussen twee benen zijn niet gemodelleerd.
- **Eén beurs.** Binance-only, met het tegenpartijrisico uit §6.
- **De authoritative engine** (`backtest/engine.py`) kent geen tweebeensboek. Alles hier
  is screening (`NOT_ADMISSIBLE`), zoals in v2–v5.
- **CI op deze branch.** Drie checks zijn rood, en ze waren dat al vóór dit werk:
  - de coverage-drempel van 70 % (nu 63,2 %, was ~61 %);
  - `test_regime_conditioning[hmm3-diag-student_t]`, de vooraf bestaande niet-convergente
    HMM-fit.
