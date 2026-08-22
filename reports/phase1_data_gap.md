# PHASE 1 - DATA GAP ANALYSIS

**Gegenereerd:** 2026-08-22T18:29:44+00:00  
**Generator:** `scripts/phase1_data_gap.py` (read-only)  
**Bindend brondocument:** `docs/ARCHITECTUUR_AUDIT_2026-08-22.md` sectie 3.2, 7.1, 7.2  
**Status:** bewijsstuk voor sectie 3.2 - wordt niet overschreven na ingestie.

---

## 1. Wat er FEITELIJK aanwezig is

### 1.1 Legacy store `market_data_parquet/`

| Bestand | Rijen | MB | Tijdkolom | TZ | Van | Tot |
|---|---:|---:|---|---|---|---|
| `market_data_parquet/commodities/eia_term_structure.parquet` | 126,614 | 0.59 | `event_ts` | UTC | 1980-01-02 | 2024-04-05 |
| `market_data_parquet/equities/factors_daily.parquet` | 15,854 | 0.42 | `event_ts` | UTC | 1963-07-01 | 2026-06-30 |
| `market_data_parquet/fx/rates.parquet` | 21,995 | 0.4 | `event_ts` | UTC | 1954-01-04 | 2026-06-16 |
| `market_data_parquet/fx/spots.parquet` | 118,090 | 1.06 | `event_ts` | UTC | 1971-01-04 | 2026-06-12 |
| `market_data_parquet/xasset/tr_panel.parquet` | 135,183 | 0.99 | `event_ts` | UTC | 2004-01-02 | 2026-08-07 |

### 1.2 PIT store `data/pit_store/`

**Leeg.** De PIT-store wordt in stap 3 van deze fase gebouwd en in
stap 5-7 gevuld. Tot dat moment bestaat er geen enkele dataset met een
gecertificeerde `data_hash`, en is dus elk historisch onderzoeksresultaat
per definitie `INVALID` (sectie 7.2). Dat wordt in Phase 2 stap 1 formeel
in het falsificatieregister vastgelegd.

---

## 2. DE CRYPTO-LACUNE (sectie 3.2), gekwantificeerd

Het geconfigureerde universum telt **6** symbolen: `AVAXUSDT`, `BTCUSDT`, `DOTUSDT`, `ETHUSDT`, `LINKUSDT`, `SOLUSDT`.

| Vereiste crypto-dataset | Symbolen vereist | Symbolen aanwezig | Dekking |
|---|---:|---:|---:|
| OHLCV daily | 6 | **0** | **0%** |
| OHLCV 5m | 6 | **0** | **0%** |
| funding rates | 6 | **0** | **0%** |
| open interest | 6 | **0** | **0%** |
| liquidaties | 6 | **0** | **0%** |
| orderboek L1 | 6 | **0** | **0%** |
| orderboek L2 | 6 | **0** | **0%** |

**Er is nul byte crypto-data in de werkkopie.** `market_data_parquet/`
bevat uitsluitend FX-, macro-, commodity- en equity-reeksen. De bevinding
van sectie 3.2 wordt hiermee bevestigd en gekwantificeerd.

### 2.1 Consequentie voor bestaande claims

Het vlaggenschipmodel (*Crypto-MN, Sharpe 1.15*) en de 4 als geaccepteerd
geregistreerde alpha-units zijn in deze werkkopie **niet reproduceerbaar**.
Er zijn **0 verifieerbare actieve alpha-units**.

De projectgeschiedenis vermeldt een dataopslag buiten de repository
(`Desktop/Trading Setup A/B/C`). Die mappen bestaan niet meer op deze
machine; geverifieerd op de datum van dit rapport. De data is dus niet
elders beschikbaar maar daadwerkelijk afwezig en moet opnieuw worden
ingested.

---

## 3. Vereist versus aanwezig, per research track (sectie 7.1)

### Alpha Research

| Dataset | Aanwezig | Toelichting |
|---|---|---|
| crypto OHLCV daily | **NEE** | per symbool, maximale historie |
| crypto OHLCV hourly/intraday | **NEE** | per symbool |
| funding rates (perp) | **NEE** | settlement-ts in UTC ns, interval-metadata |
| open interest | **NEE** | per symbool |
| liquidaties | **NEE** | per symbool |
| cross-sectional spreads | afgeleid | afgeleid uit OHLCV-panel |
| volume profiles | afgeleid | per symbool |

### Volatility Research (GARCH & EWMA)

| Dataset | Aanwezig | Toelichting |
|---|---|---|
| 1m / 5m bars | **NEE** | voor Realized Variance en HAR-RV |
| crypto OHLCV daily | **NEE** | voor GARCH/EGARCH/GJR |

### Regime Research

| Dataset | Aanwezig | Toelichting |
|---|---|---|
| macro-economische tijdreeksen | gedeeltelijk | FRED e.d. |
| implied volatility indices | **NEE** | DVOL / VIX |
| volatility term structures | gedeeltelijk | FX + Fama-French + EIA aanwezig; DVOL/VIX niet |
| cross-asset correlatiematrices | afgeleid | afgeleid |

### Execution & TCA Research

| Dataset | Aanwezig | Toelichting |
|---|---|---|
| orderboek L1 (top-of-book) | **NEE** | snapshot-frequentie gedocumenteerd |
| orderboek L2 (depth) | **NEE** | voor eta-kalibratie, Phase 5 |
| individuele trade-prints | **NEE** |  |
| exchange fee schedules | JA | conf/execution/fees.yaml |
| latentiestatistieken | **NEE** |  |

---

## 4. Reikwijdte van deze fase

Besluit van de opdrachtgever (2026-08-22): **alleen daily OHLCV nu,
intraday later**. Dat betekent voor de exit criteria van deze fase:

| Dataset | Deze fase | Consequentie |
|---|---|---|
| crypto OHLCV daily | **wordt ingested** | basis voor de Phase 3-baseline |
| funding rates | **wordt ingested** | settlement-semantiek in UTC ns |
| open interest | **wordt ingested** | |
| liquidaties | **wordt ingested** waar de API het toelaat | Bybit publiceert geen volledige historie |
| orderboek L1/L2 | **niet** | alleen live te samplen; geen REST-historie |
| OHLCV 5m | **niet** | doorgeschoven; blokkeert HAR-RV en Realized Variance |

**Open Question 1** (sectie 27) - *beschikken we over voldoende
orderboek-depth om HAR-RV en het TCA-impactmodel te kalibreren?* - wordt
in het exit-rapport van deze fase beantwoord met de gemeten dekking van
wat wel is ingested, plus een expliciet *nog niet gemeten* voor intraday
en depth. Het antwoord is daarmee niet *ja* en niet *nee*, maar
*onbeantwoordbaar op de huidige data* - en dat wordt als zodanig
geregistreerd in plaats van geraden.

