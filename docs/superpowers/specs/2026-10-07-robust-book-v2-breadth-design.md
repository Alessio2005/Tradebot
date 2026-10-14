# Robuust boek v2 — breedte: een survivorship-vrij Binance-universum (2026-10-07)

Status: **bevroren vóór de eerste meting op het nieuwe universum.** De enige data die vóór
dit document is bekeken: de bestandslijst van de bucket (welke symbolen en maanden
bestaan). Er is geen enkele koers of funding van het nieuwe universum geparseerd.

## 1. Waarom v2

v1 (`2026-10-07-robust-book-design.md`) liet zien dat de zes Bybit-munten de bottleneck
zijn: een gemiddelde correlatie van 0,73 geeft ≈ 1,3 onafhankelijke weddenschappen. De
beste robuuste kandidaat haalde een netto Sharpe van 0,71 (screening, lag 1) en 0,39
(authoritative, lag 2).

Volgens de fundamentele wet van actief beheer (IR ≈ IC·√breedte) is breedte de enige
hefboom die het niveau structureel verhoogt zonder meer te zoeken. v2 vergroot het
universum, niet de zoekruimte.

**Bron.** De publieke `data.binance.vision`-bucket, gelezen via het S3-endpoint
(`src/tradebot/data/binance_vision.py`):

* alle USDT-M-perpetuals die ooit hebben gehandeld (900), **inclusief gedeliste**;
* dagbars en 8h-funding van 2020-01 tot en met 2026-09;
* elke zip geverifieerd tegen zijn S3-ETag (MD5), met een manifest in
  `artefacts/data/binance_um_manifest.json`.

## 2. Universum U_t (point-in-time, besloten op de close van *t*)

Een perp hoort op bar *t* bij U_t als:

1. hij op *t* een close heeft en ≥ 90 bars koersgeschiedenis;
2. zijn 30-daagse gemiddelde quote-volume (ADV, t/m *t*) ≥ $5M is;
3. zijn 60-daagse gerealiseerde vol ≥ 15 % per jaar is (sluit gekoppelde/stablecoin-perps uit);
4. hij tot de **top 50** naar ADV hoort (gelijke ADV: symboolnaam beslist).

Datagaten van ≤ 7 dagen binnen de noteringsperiode worden overbrugd met de laatst bekende waarde. De bron mist bijvoorbeeld 2022-02-27 → 03-01 en 2022-04-02/03 voor ≈ 48 munten tegelijk; dat zijn ontbrekende bestanden, geen handelsstops. Langere gaten gelden als delisting plus herintrede. De bucket bevat 31 gedeliste symbolen, waarvan 21 ooit in U_t zaten. Niet elke ooit gedeliste perp is bewaard, dus een rest-survivorship-bias blijft mogelijk.

Gedeliste munten vallen er vanzelf uit. Een positie in een munt waarvan de koers
verdwijnt, wordt op de laatste bekende close gesloten (rendement 0 op de ontbrekende
bar), met kosten tegen de laatste bekende ADV en σ.

## 3. Sleeves (literatuurgepind, niets gezocht)

| Code | Premie | Constructie | Herbalanceren |
| --- | --- | --- | --- |
| `X1_XSMOM` | cross-sectioneel momentum (Liu-Tsyvinski-Wu 2022; Fieberg e.a. 2023) | per L ∈ {7, 14, 28, 56}: `ln(P_t/P_{t−L}) / (σ √L)`, cross-sectionele z (winsor ±3), gemiddeld over L; long de bovenste helft, short de onderste, inverse-vol binnen elk been, elk been bruto 0,5 | wekelijks (vaste fase) |
| `X2_XSCARRY` | funding-carry (Schmeling-Schrimpf-Todorov 2023) | bekende carry = gemiddelde funding over 7 bars t/m *t−1*; short de bovenste helft, long de onderste, inverse-vol, dollar-neutraal | wekelijks |
| `X3_TREND_LS` | tijdreeks-momentum | v1-trendscore (11 lookbacks 5…160) per munt in U_t, `w = S·τ_a/σ / N` | dagelijks |
| `X4_TREND_LF` | idem, long-flat | `S⁺` | dagelijks |
| `X5_COMBO` | diversificatie tussen premies | gelijk-risicosom van de sleeves uit {X1, X2, X4} met **TRAIN**-Sharpe > 0 | dagelijks |

Referenties: `R1_BTC_HOLD` (1,0× BTC) en `R2_EW50_HOLD` (gelijk gewogen long over U_t,
bruto 1,0).

**Sizing.** Elke sleeve wordt naar 20 % vol geschaald met één factor
`k = min(τ_p/σ̂_p, 3)`. σ̂_p komt uit een stromende EWMA-covariantie (span 60, causaal)
over de munten in het boek. Daarna gelden per munt |w| ≤ 0,20 en bruto ≤ 4,0
(mandaat `gross_cap`).

## 4. Kosten en executie (vooraf gekalibreerd, niet achteraf gekozen)

* **Fees.** Taker 5,5 bp (`conf/execution/fees.yaml`; Binance VIP0 is 5,0 bp, dus dit is
  conservatief).
* **Spread.** Halve spread per munt per dag = max(1 bp, ½ · Abdi-Ranaldo-CHL-schatter),
  rollend over 60 dagen en causaal, met een plafond van 25 bp. Dit is een dataschatting en
  geen aanname.
* **Impact.**
  * `Y · σ_dag · √(order/ADV)` met Y = 1,0 als basis: de literatuur geeft Y ≈ 0,5–1 voor
    metaorders (Bouchaud e.a.; Tóth e.a.).
  * De η = 2,99 uit `conf/execution/impact.yaml` dient als **stresstest**. De repo noemt
    die waarde zelf een bovengrens uit de dagrange, geen impactschatting.
* **Boekgrootte.** Basis $1M. Capaciteit gemeten bij $10M en $50M.
* **Timing.** Besluit op de close (00:00 UTC), uitvoering direct tegen ≈ die prijs
  (`lag = 1`, gerechtvaardigd door `open(t+1) == close(t)` in een 24/7-markt).
  Stresstests: `lag = 2` en +10 bp slippage.
* **Funding.** De werkelijke funding wordt geboekt: long betaalt positieve funding.

## 5. Vensters

| Venster | Periode | Gebruik |
| --- | --- | --- |
| warm-up | 2020-01 → 2020-12 | universumvorming, signaalhistorie |
| `TRAIN` | 2021-01-01 → 2023-12-31 | inclusie in X5 |
| `VALIDATE` | 2024-01-01 → 2025-06-30 | consistentiepoort |
| `W_DEV` | TRAIN ∪ VALIDATE | poorten, batterij, selectie |
| `HOLDOUT` | 2025-07-01 → 2026-09-30 | **één lezing**, één kandidaat, eigen slot (`holdout_lock_binance_um.json`) |

**Voorbehoud, eerlijk vermeld.** Ik weet uit v1 dat 2025-09 → 2026-08 voor de grote munten
een bear-markt was. Voor dit universum en deze sleeves is er niets gemeten. Het deel
2026-08-24 → 2026-09-30 heeft niemand in deze repository ooit gezien.

## 6. Poorten, selectie en boekhouding

* **Poorten.** G1–G10 uit v1 (Sharpe ≥ 1, CAGR ≥ 15 %, ≥ 0,5 in beide helften, DD ≤ 25 %,
  kosten- en vertragingsrobuustheid, plateau, bootstrap-ondergrens > 0, DSR ≥ 0,95,
  PBO < 0,5, en geen ineenstorting op de holdout). De DSR geldt bij de cumulatieve
  ledger-M: 7 (v1) + 7 (v2) = 14. Daarnaast wordt hij gerapporteerd bij +52 bekende trials.
* **Selectie.** Dezelfde robuustheidsscore als v1, uitsluitend op W_DEV. Alleen de winnaar
  leest de holdout.
* **Boekhouding.** Zeven trials (R1, R2, X1–X5) worden in de ledger geboekt vóór het
  bevriezen.

## 7. Prior

Voor gediversifieerde crypto-factorportefeuilles na kosten rapporteert de literatuur
Sharpes van 1–2 vóór 2022 en duidelijk lager daarna. Mijn verwachting vooraf voor X5 op
W_DEV is een netto Sharpe van 0,8–1,3. Out-of-sample verwacht ik daar 30–50 % van af.
