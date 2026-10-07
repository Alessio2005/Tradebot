# Robuust boek v2–v5 — breedte, kosten, bevestiging en basiscarry (2026-10-07)

Vervolg op `2026-10-07-robust-book-report.md` (v1). Dezelfde regels: elke trial in de
ledger vóór de run, elke preregistratie bevroren vóór de meting, elke lezing buiten W_DEV
geregistreerd vóór de data terugkomt. De vraag blijft: is een **netto Sharpe > 1 en een
netto CAGR > 15 %** haalbaar met een strategie die robuust genoeg is om op te vertrouwen?

<!-- V5_SUMMARY -->

## 1. v2 — breedte: het survivorship-vrije Binance-universum

**Probleem uit v1.** Zes Bybit-perps, overlevers van nu, met een onderlinge correlatie van
0,73: ongeveer 1,3 onafhankelijke weddenschappen. Op zo'n universum is geen cross-sectioneel
signaal te meten.

**Data** (`data/binance_vision.py`):
- de volledige dag-historie van elke Binance USDT-M-perp die ooit handelde, ook gedeliste
  munten (LUNA, FTT, SRM, ...): 900 symbolen, 2020-01 tot en met 2026-09;
- elke zip gecontroleerd tegen zijn S3-MD5;
- point-in-time-universum: de top-50 naar 30-daagse ADV, met ≥ 90 bars historie,
  ≥ 5 mln USDT ADV en ≥ 15 % jaarvol (tegen gekoppelde munten);
- korte datagaten overbrugd, delistings gesloten tegen de laatste koers.

**Bevinding over kosten.** De vooraf gekozen Abdi-Ranaldo-spreadschatter (CHL) meet op
dagbars volatiliteit, geen spread: 17,7 bp voor BTCUSDT, terwijl de werkelijke spread
onder 1 bp ligt. v2 rekende daardoor met te hoge kosten. v3 corrigeert dat (post-hoc, en
zo geboekt).

**W_DEV (2021-01 .. 2025-06), netto:**

| trial | Sharpe | CAGR | vol | MDD | omzet/jr | bruto+funding-Sharpe |
|---|---|---|---|---|---|---|
| X1 cross-sectioneel momentum | −0,14 | −4,7 % | 19,8 % | 39,9 % | 55× | 0,60 |
| X2 funding-carry (long/short) | 0,31 | 4,4 % | 21,7 % | 44,2 % | 68× | 1,13 |
| X3 trend long/short | 0,44 | 5,7 % | 15,5 % | 18,0 % | 23× | 0,72 |
| X4 trend long-flat | **0,77** | 8,2 % | 11,1 % | 15,5 % | 11× | 0,94 |
| X5 combinatie | 0,33 | 4,8 % | 19,9 % | 43,7 % | 94× | 1,49 |
| R1 BTC buy-and-hold | 0,58 | 18,2 % | 60,7 % | 78,9 % | — | — |

De score koos X4. Die bindt op G1, G2, G3 (validate 0,16), G7, G8 en G9.
**Holdout X4: Sharpe −0,11**, gearchiveerd.

## 2. v3 — realistische kosten en partiële aanpassing

Twee wijzigingen, en verder niets:
- een vaste halve spread van 3 bp, met 6 bp als stress;
- Gârleanu-Pedersen partiële aanpassing (κ = 0,2), zodat de omzet bij het signaalverval past.

| trial | Sharpe | CAGR | MDD | 95 %-CI | train / validate |
|---|---|---|---|---|---|
| Y2 carry | 0,83 | 14,8 % | 29,3 % | [−0,09; 1,76] | 1,08 / 0,31 |
| Y4 trend long-flat | 0,86 | 9,6 % | 14,5 % | [−0,04; 1,80] | 1,15 / 0,24 |
| Y5 carry + trend | **1,13** | **20,7 %** | 24,4 % | [0,19; 2,07] | 1,50 / 0,33 |

Y5 haalde de doelcijfers in-sample, maar faalde op G3 (validate 0,33), G5 (kosten), G8
(DSR 0,72 bij M = 17) en G9 (PBO). De vooraf vastgelegde score koos Y4.
**Holdout Y4: Sharpe 0,00, CAGR −0,3 %.**

## 3. v4 — één geregistreerde bevestiging van Y5

Y5 was de enige kandidaat met de doelcijfers op W_DEV. Hem toch op de holdout zetten is
een extra trial. Die is als zodanig geboekt (M = 18), met de criteria vóór de lezing
bevroren.

| | Sharpe | CAGR | MDD |
|---|---|---|---|
| holdout Y5 (2025-07 .. 2026-09) | **−1,19** (SE 1,01) | **−17,6 %** | **27,0 %** |
| waarvan carry | −1,17 | −18,5 % | 25,2 % |
| waarvan trend | 0,00 | −0,3 % | 10,5 % |

**Gefalsificeerd:** z = −2,29 tegen W_DEV, en een drawdown boven het mandaat.

**De les.** Het carryboek ontving 22,9 %/jaar aan funding en verloor meer op de prijs.
Long lage en short hoge funding is een prijsweddenschap: de short-kant zit in munten die
met hefboom worden gekocht, en die bleven stijgen. Dat leidde naar v5.

<!-- V5_SECTION -->
