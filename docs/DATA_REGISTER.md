# DATA REGISTER

> **Geverifieerd tegen de codebase op 2026-09-01** (Phase 7/8, Stage E-3).
> GEMETEN: de gap-ledger. De dekkings- en hashcijfers in dit document
> zijn NIET opnieuw tegen de PIT-store gedraaid; daarvoor is
> `apps/run_data_adequacy.py` de bron.

> **De enige geldige bron voor de vraag: welke data mag ik gebruiken?**
>
> Een onderzoeksresultaat dat geen `data_hash` uit dit register citeert,
> is per definitie `INVALID` (audit sectie 7.2).

**Gegenereerd:** 2026-08-23T09:18:03+00:00  
**Generator:** `scripts/build_data_register.py`  
**git_sha:** `661f351`  
**DVC dir-hash van de PIT-store:** `e04fff202fdc77d375a990fa99c43c3d.dir`  
**PIT-store root:** `data/pit_store`

---

## 1. Gecertificeerde datasets

| Asset class | Dataset | Symbool | Gran. | Rijen | Van | Tot | `data_hash` |
|---|---|---|---|---:|---|---|---|
| crypto | funding | AVAXUSDT | 8h | 5,408 | 2021-09-15 | 2026-08-23 | `8edfe83d0a02f02a6871a9a95877d054` |
| crypto | funding | BTCUSDT | 8h | 7,025 | 2020-03-25 | 2026-08-23 | `a69aa5fbc2c0e0c3a214c4b672b8f4e8` |
| crypto | funding | DOTUSDT | 8h | 5,951 | 2021-03-18 | 2026-08-23 | `28f49d9a617fd93f36023efc35b568be` |
| crypto | funding | ETHUSDT | 8h | 5,962 | 2021-03-15 | 2026-08-23 | `02aef54a332ac99c5f93205d375f8737` |
| crypto | funding | LINKUSDT | 8h | 6,396 | 2020-10-21 | 2026-08-23 | `ce6cfcd63ddd681178ebb611b87fd621` |
| crypto | funding | SOLUSDT | 8h | 5,680 | 2021-10-15 | 2026-08-23 | `8f2c70dbf68f1fe635c937b2e6b67c36` |
| crypto | ohlcv | AVAXUSDT | 1d | 1,803 | 2021-09-15 | 2026-08-22 | `af05b4f95d0e282111641f85904c22f9` |
| crypto | ohlcv | BTCUSDT | 1d | 2,342 | 2020-03-25 | 2026-08-22 | `4f21f2c7ab071ddc19da5eb3f38172f3` |
| crypto | ohlcv | DOTUSDT | 1d | 1,983 | 2021-03-19 | 2026-08-22 | `17be1f4206997f12fd7e1825f123299b` |
| crypto | ohlcv | ETHUSDT | 1d | 1,987 | 2021-03-15 | 2026-08-22 | `d67c9bb18b2c794ba416067556227a5a` |
| crypto | ohlcv | LINKUSDT | 1d | 2,132 | 2020-10-21 | 2026-08-22 | `8d6a82f316253c2626d016bbbad804d4` |
| crypto | ohlcv | SOLUSDT | 1d | 1,773 | 2021-10-15 | 2026-08-22 | `1d64b001993b575f34ef04ce548616c5` |
| crypto | open_interest | AVAXUSDT | 1d | 1,803 | 2021-09-16 | 2026-08-23 | `5f642bd6e318ad795103682c530dd952` |
| crypto | open_interest | BTCUSDT | 1d | 2,210 | 2020-08-05 | 2026-08-23 | `4fe481c71f8237e9ba2bbd80f0d84e96` |
| crypto | open_interest | DOTUSDT | 1d | 1,984 | 2021-03-19 | 2026-08-23 | `1795c58ed7ab641fd80219a26272b313` |
| crypto | open_interest | ETHUSDT | 1d | 1,988 | 2021-03-15 | 2026-08-23 | `be986e9314c9f28ef3fd147b01bfefd4` |
| crypto | open_interest | LINKUSDT | 1d | 2,132 | 2020-10-22 | 2026-08-23 | `9024d7e536d7ced7e8e23a4dc97eb952` |
| crypto | open_interest | SOLUSDT | 1d | 1,774 | 2021-10-15 | 2026-08-23 | `9267e0accf0055a1cba7c9db1ab5d4e2` |

**Totaal:** funding: 36,422 rijen, ohlcv: 12,020 rijen, open_interest: 11,891 rijen — 18 reeksen, 114 partities.

---

## 2. Bekende gaps

**Nul ontbrekende bars over alle reeksen.** Gemeten met
`data/validation/gaps.py` op de bar-cadans van elke granulariteit.
De gap-ledger (`artefacts/governance/gap_ledger.jsonl`) BESTAAT NIET, en dat
is de correcte toestand: hij wordt pas aangemaakt bij het eerste gat. Een leeg
bestand zou niet te onderscheiden zijn van een ledger die nooit is geschreven.

Er wordt **nooit** geinterpoleerd. `gap_policy` staat in
`conf/data/default.yaml`; bij `reject` breekt een gat de ingestion.

---

## 3. Gemeten uitschieters

Drempel: `max_abs_log_return = 0.35` (uit `conf/data/`), `allow_price_jumps = True`.

| Symbool | Bars | Sprongen > drempel | Grootste \|log-return\| | Zwaarste dag |
|---|---:|---:|---:|---|
| AVAXUSDT | 1,803 | 1 | 0.3582 | 2022-05-11 |
| BTCUSDT | 2,342 | 0 | 0.1782 | - |
| DOTUSDT | 1,983 | 1 | 0.4828 | 2021-05-19 |
| ETHUSDT | 1,987 | 0 | 0.3238 | - |
| LINKUSDT | 2,132 | 1 | 0.4837 | 2021-05-19 |
| SOLUSDT | 1,773 | 2 | 0.8121 | 2022-11-09 |

`allow_price_jumps = true` betekent: **gemeten, beoordeeld en als echte
marktgebeurtenis geaccepteerd**. Crypto kent dagen met bewegingen van
tientallen procenten (12 maart 2020, 19 mei 2021, november 2022); die
weggooien of winsoriseren zou de staartverdeling vervalsen, en juist die
staart bepaalt de drawdown. `high < low` en `volume == 0` blijven
onvoorwaardelijk fataal.

---

## 4. Toegang per research track (sectie 7.1)

| Dataset | Mag gebruikt worden door |
|---|---|
| `funding` | Alpha Research (carry), Execution & TCA Research |
| `ohlcv` | Alpha Research, Volatility Research (daily GARCH/EWMA), Regime Research |
| `open_interest` | Alpha Research (positionering), Regime Research |

---

## 5. Semantiek van de tijdkolommen

| Dataset | `event_ts_ns` | `asof_ts_ns` |
|---|---|---|
| `ohlcv` | openingstijd van de bar | **sluitingstijd** — een daily bar over 3 jan is pas op 4 jan 00:00 UTC compleet |
| `funding` | settlement-moment | settlement-moment — een rate die om 08:00 UTC settelt is **pas dan** bekend |
| `open_interest` | snapshot-moment | snapshot-moment |

Elke koppeling tussen deze reeksen loopt via
`utils.time.asof_join(direction="backward")` met een **verplichte**
tolerance. Bewezen truncatie-invariant in
`tests/lookahead/test_asof_join_crypto.py`.

---

## 6. Wat hier NIET staat

| Ontbrekend | Gevolg |
|---|---|
| OHLCV 1m/5m | Realized Variance en HAR-RV zijn niet te schatten; de Volatility Research track blijft beperkt tot daily (EWMA, GARCH). |
| Orderboek L1/L2 | De eta-kalibratie (Phase 5) kan niet op echte depth-data. Open Question 1 blijft onbeantwoordbaar. |
| Liquidaties | Bybit publiceert geen historische liquidatie-feed via de publieke REST-API. |
| Historische delistings | Het universum bevat uitsluitend nog-actieve symbolen; zie het exit-rapport voor de omvang van de survivorship bias. |

