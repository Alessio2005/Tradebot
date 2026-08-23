# PHASE 1 — EXIT REPORT

> **Fase:** 1 van 7 — Data Foundation & Crypto Ingestion
> **Bindend brondocument:** `docs/ARCHITECTUUR_AUDIT_2026-08-22.md` — secties 3.2, 7, 7.1, 7.2, 20, 23, 26
> **Voorwaarde:** Phase 0 afgerond (ratchet-besluit 2026-08-22)
> **Datum:** 2026-08-23

---

## VERDICT

**6 van de 9 exit criteria volledig behaald. 2 gedeeltelijk. 1 niet haalbaar op
de gekozen bron.**

| # | Criterium | Status |
|---|---|---|
| 1 | `asof_join` truncatie-invariant op crypto | ✅ **BEHAALD** |
| 2 | Unieke, deterministische `data_hash` per dataset | ✅ **BEHAALD** |
| 3 | Crypto-dataset compleet | ⚠️ **GEDEELTELIJK** — scope-besluit |
| 4 | Timezone-integriteit (UTC ns) | ✅ **BEHAALD** |
| 5 | Onveranderlijkheid bewezen | ✅ **BEHAALD** |
| 6 | Survivorship bias geadresseerd | ⚠️ **GEDEELTELIJK** — bronbeperking |
| 7 | Gap-ledger compleet, nul interpolaties | ✅ **BEHAALD** |
| 8 | DVC-lineage sluitend | ✅ **BEHAALD** |
| 9 | Open Question 1 met cijfers beantwoord | ✅ **BEHAALD** (het antwoord is *nee*) |

De twee gedeeltelijke criteria zijn dat om verschillende redenen: **3** door een
expliciet scope-besluit van de opdrachtgever, **6** doordat de venue de data
niet publiceert. Beide staan hieronder gekwantificeerd.

---

## 1. `asof_join` geverifieerd op crypto-data — ✅ BEHAALD

```bash
pytest tests/lookahead/test_asof_join_crypto.py -q
# 29 passed
```

De truncatietest uit stap 10, op de **echte** PIT-store: koppel funding rates
aan OHLCV, kap alle data na `t` af, herhaal de join, eis bit-identieke waarden
op en vóór `t`. Uitgevoerd voor **alle 6 symbolen** op **drie snijpunten** per
symbool (25%, 50%, 75%). **Nul afwijkingen.**

Daarnaast bewezen:

- geen enkele gekoppelde funding rate ligt ná de bar-timestamp;
- geen bar in de store is op zijn openingstijd al als bekend gemarkeerd;
- geen rij in de store is pas in de toekomst kenbaar;
- het panel-hash is deterministisch over twee onafhankelijke loads.

### Twee testaannames die de meting heeft weerlegd

Beide betroffen mijn test, niet de code — en zijn als zodanig gecorrigeerd:

1. Ik nam aan dat een krappere `tolerance` méér bars ongekoppeld zou laten. Onjuist:
   Bybit settelt funding om 00:00/08:00/16:00 UTC en een daily bar opent om
   00:00 UTC, dus **elke bar valt exact samen met een settlement** en
   `allow_exact_matches=True` koppelt hem ongeacht de tolerance. De test meet nu
   die feitelijke eigenschap (`lag == 0` voor elke gekoppelde rij).
2. Het tolerance-contract wordt daarom getoetst op een reeks met een bewust gat
   van 30 dagen. Bars binnen de eerste 8 uur koppelen dan terecht nog; pas
   daarna houdt het op. Met `tolerance=10 jaar` draagt `merge_asof` de rate wél
   de hele maand vooruit — precies waarom de parameter verplicht is gemaakt.

---

## 2. Deterministische `data_hash` per dataset — ✅ BEHAALD

18 reeksen, elk met een `data_hash` geciteerd in `docs/DATA_REGISTER.md`.

Determinisme is op **drie** niveaus aangetoond:

| Niveau | Bewijs |
|---|---|
| Twee onafhankelijke live ingestion-runs | identieke hash per reeks; de append-only store accepteerde de tweede run als no-op |
| Parquet-roundtrip | `tests/integration/test_data_contract_roundtrip.py` — ingest → hash → persist → reload → identieke hash |
| Hash-eigenschappen | invariant onder rij-, kolom- en index-volgorde; gevoelig voor waarde én dtype |

`dataframe_content_hash` hasht bewust de **gesorteerde inhoud**, niet
bestandsmetadata: mtime, pad, compressieniveau en pyarrow-versie zijn
machineafhankelijk, en zouden het hele provenance-contract betekenisloos maken.

---

## 3. Crypto-dataset compleet — ⚠️ GEDEELTELIJK (scope-besluit)

**De lacune uit sectie 3.2 is gesloten voor de gescopede datasets.**
Nulmeting: 0 bytes crypto-data. Nu:

| Dataset | Gran. | Symbolen | Rijen | Periode | Gaps |
|---|---|---:|---:|---|---:|
| `ohlcv` | 1d | 6/6 | 12.020 | 2020-03-25 → 2026-08-22 | **0** |
| `funding` | 8h | 6/6 | 36.422 | 2020-03-25 → 2026-08-23 | **0** |
| `open_interest` | 1d | 6/6 | 11.891 | 2020-08-05 → 2026-08-23 | **0** |

**60.333 rijen, 114 partities, nul ontbrekende bars.**

### Wat bewust NIET is ingested

Conform het besluit van de opdrachtgever (2026-08-22): *alleen daily nu,
intraday later*.

| Ontbrekend | Reden | Gevolg |
|---|---|---|
| OHLCV 1m/5m | scope-besluit | Realized Variance en HAR-RV zijn niet te schatten; Volatility Research blijft beperkt tot daily |
| Orderboek L1/L2 | geen REST-historie; alleen live te samplen | η-kalibratie (Phase 5) heeft geen echte depth-data |
| Liquidaties | Bybit publiceert geen historische liquidatie-feed publiek | positioneringsanalyse mist de liquidatie-component |

### Onafhankelijke validatie van de data-echtheid

Elke gemeten uitschieter boven de drempel is een **herkenbare marktgebeurtenis**
op de juiste dag — sterk bewijs dat de reeksen echt en correct uitgelijnd zijn:

| Datum | Symbool | log-return | Gebeurtenis |
|---|---|---:|---|
| 2021-05-19 | DOT, LINK | −0,48 | de mei-2021 crash |
| 2022-05-11 | AVAX | −0,36 | Terra/LUNA-ineenstorting |
| 2022-11-09 | SOL | −0,81 (−55%) | FTX-ineenstorting |

Nul bars met `high < low`; nul bars met `volume == 0`.

---

## 4. Timezone-integriteit — ✅ BEHAALD

100% van de timestamps in de PIT-store is `int64` UTC Unix nanoseconden. De
store **weigert** een `datetime`-kolom en een `float`-kolom expliciet
(`tests/unit/test_pit_store_immutability.py`).

Een geïnjecteerde naïeve timestamp crasht aantoonbaar:

```python
to_utc_ns(pd.Timestamp("2024-01-01"))
# DataContractError: Naieve (timezone-loze) timestamp geweigerd aan de
# ingestion-grens...
```

Dit verschilt bewust van het bestaande `to_utc`, dat een naïeve timestamp als
UTC aanneemt. Die aanname is verdedigbaar aan de live-kant (exchange-responses
zijn per conventie UTC) maar onaanvaardbaar aan de ingestion-kant: daar is een
naïeve timestamp het symptoom van een bron waarvan de tijdzone niet is
vastgesteld, en een verkeerde aanname verschuift de hele reeks met uren.

---

## 5. Onveranderlijkheid bewezen — ✅ BEHAALD

```bash
pytest tests/unit/test_pit_store_immutability.py -q
# 23 passed
```

Een tweede schrijfactie op dezelfde partitie met **afwijkende** inhoud crasht
met `DataContractError`; met **identieke** inhoud is het een no-op, zodat een
herstarte ingestion-run veilig is. Er bestaat geen `overwrite`-, `force`-,
`replace`- of `mode`-parameter — afgedwongen door een test die de signature
inspecteert.

### De contracten hebben tijdens deze fase twee echte ontwerpfouten gevangen

Beide werden gevonden doordat de append-only guard vuurde, niet doordat ik ze
zag aankomen:

1. **De bar van vandaag is nog niet gesloten** en muteert gedurende de dag. Twee
   ingestion-runs een uur na elkaar gaven een andere `data_hash` voor dezelfde
   partitie. `run_ingestion` laat nu elke rij vallen waarvan `asof_ts_ns` in de
   toekomst ligt — een universele PIT-regel: *wat nog niet kenbaar is, is geen
   observatie*. Zonder die guard zou een backtest een gedeeltelijke dagslot-prijs
   als definitief hebben behandeld.
2. **De voorgeschreven partitiesleutel is ondergespecificeerd.** Deliverable 1
   schrijft `(asset_class, symbol, granularity, date)` voor; OHLCV, funding en
   open interest van hetzelfde symbool op dezelfde granulariteit vallen daarmee
   in dezelfde partitie. De botsing was concreet: BTCUSDT/1d/2020 kreeg 282
   OHLCV-rijen en vervolgens 149 open-interest-rijen. `dataset` is aan de
   sleutel toegevoegd; de afwijking van de audittekst staat gemotiveerd in de
   module-docstring.

---

## 6. Survivorship bias — ⚠️ GEDEELTELIJK (bronbeperking)

**Wat is gebouwd en werkt:**

- `data/validation/continuity.py` levert een **expliciete adjustment factor
  ledger** in plaats van een stilzwijgende backward-ratio-aanpassing. De ruwe
  reeks blijft intact; `apply_adjustments` werkt op een kopie. Achteraf blijft
  daardoor vaststelbaar wélke correctie met welke factor is toegepast — bij een
  in-place aanpassing is dat onmogelijk. Gedetecteerde sprongen zijn
  **kandidaten**, geen automatisch toegepaste correcties.
- `SymbolLifecycle` weigert bars vóór de listing en ná de delisting van een
  symbool.
- `artefacts/governance/symbol_lifecycle.json` bevat de gemeten listing-datum
  per symbool.

**Wat niet kan, en waarom — gemeten:**

```
GET /v5/market/instruments-info?category=linear  →  833 instrumenten
status counts: {'Trading': 833}
non-Trading USDT perps: []
```

De publieke Bybit V5-API retourneert **uitsluitend nog-verhandelde
instrumenten**. Historische delistings zijn via deze bron **niet verkrijgbaar**.
Het universum kan ze dus niet bevatten, hoe het contract ook is gebouwd.

**Omvang van de resterende bias, eerlijk benoemd.** Het universum bestaat uit
zes large-cap symbolen die alle zes nog verhandeld worden en ex-post zijn
gekozen. Dat is een reële selectiebias voor elke Phase 3-claim: een
momentum-strategie op zes overlevers oogt gunstiger dan dezelfde strategie op
een universum dat periodiek opnieuw wordt samengesteld. Dit is geen restrisico
dat later blijkt — het is een bekende, gekwantificeerde beperking die in
`reports/BASELINE_BENCHMARK.md` bij elk resultaat vermeld moet worden.

Geregistreerd als **DI-15** in `docs/DEFERRED_ISSUES.md`, toegewezen aan de
fase waarin een tweede databron met delisting-historie wordt aangesloten.

---

## 7. Gap-ledger compleet, nul interpolaties — ✅ BEHAALD

**Nul ontbrekende bars** over alle 18 reeksen, gemeten op de bar-cadans van elke
granulariteit. De gap-ledger is daardoor leeg — niet omdat er niet gemeten is,
maar omdat er niets te registreren viel.

`grep -rn "fillna(method=\|ffill()" src/tradebot/data/` levert geen enkele
interpolatie in het ingestion-pad. Bij `gap_policy: reject` (de default) breekt
een gat de ingestion; bij `register` wordt het vastgelegd en gaat de ingestion
door. **In beide gevallen wordt het gat weggeschreven** — er bestaat geen pad
waarin een gat verdwijnt zonder spoor, afgedwongen door
`test_gap_is_written_to_the_ledger_even_when_rejected`.

---

## 8. DVC-lineage sluitend — ✅ BEHAALD

```
data/pit_store.dvc → md5 e04fff202fdc77d375a990fa99c43c3d.dir, 228 files, 2,3 MB
```

De `data_hash` per reeks en de DVC dir-hash staan **beide** in
`docs/DATA_REGISTER.md`, samen met de `git_sha`. Alleen de `.dvc`-pointer staat
in git; nul parquet-bestanden zijn gecommit (geverifieerd).

### Drie kapotte stages in `dvc.yaml` gerepareerd

`dvc add` faalde omdat DVC het hele project laadt. Daarbij bleek de legacy
`dvc.yaml` drie structurele defecten te hebben — de pipeline is dus **nooit
uitvoerbaar geweest**, en de "reproducibility guarantee" in zijn eigen header
werd nergens afgedwongen:

| Defect | Effect |
|---|---|
| `${item}` interpoleerde een **dict** in elke stage over `symbol_side_pairs` | `Cannot interpolate data of type 'dict'` — 8 plekken |
| `build_features` claimde de **map** `artefacts/orthogonalizers/` als output | overlappende outputs tussen symbool-instanties |
| `make_tearsheet` had een map als `outs` én een bestand erin als `metrics` | out en metric in dezelfde tracked directory |

---

## 9. Open Question 1 — ✅ BEANTWOORD (met cijfers, en het antwoord is *nee*)

> *"Beschikken we over voldoende historische orderboek-depth data om de HAR-RV
> en TCA-impactmodellen met hoge nauwkeurigheid te kalibreren?"*

**Nee.** Gemeten:

| Vereist voor | Benodigde data | Aanwezig | Dekking |
|---|---|---|---|
| HAR-RV / Realized Variance | 1m of 5m bars | 0 rijen | **0%** |
| η-kalibratie (Phase 5) | L2 depth-snapshots | 0 rijen | **0%** |
| TCA arrival-price analyse | trade-prints | 0 rijen | **0%** |

Bybit publiceert geen historische orderboek-snapshots via de REST-API; L1/L2 is
uitsluitend **live te samplen**, wat betekent dat de dekking pas ontstaat vanaf
het moment dat er een sampler draait. Voor de η-kalibratie in Phase 5 zijn er
daarmee twee reële opties: een live orderboek-sampler starten en maanden
wachten, of een externe tick-databron aankopen. Dat is een besluit voor de
opdrachtgever, geen technische keuze.

Tot dan blijft de kostenaanname in `conf/execution/fees.yaml` expliciet
gemarkeerd als `cost_assumption_is_provisional: true`, en rapporteert Phase 3
**bruto én netto**.

---

## Regressiebewijs

| Meting | Phase 0 exit | Phase 1 exit |
|---|---:|---:|
| Tests totaal | 633 | 728 |
| Passed | 604 | 701 |
| Failed | 6 | 6 |
| Skipped | 21 | 21 |

Dezelfde 6 pre-existente failures (4 kill-gate-artefacten, 2 property-tests).
**Nul regressies.** Beide Phase 0-gates blijven groen:
`audit_fallbacks.py --strict` exit 0, `check_hardcoded_params.py --strict`
exit 0 (budget teruggebracht van 336 naar 332).

---

## Deliverables

| # | Deliverable | Status |
|---|---|---|
| 1 | `data/pit_store.py` append-only | ✅ 23 tests |
| 2 | `data/ingestion/` generiek contract | ✅ volgorde niet te omzeilen |
| 3 | crypto OHLCV ingestion | ✅ daily; intraday doorgeschoven |
| 4 | funding rates met settlement-semantiek | ✅ |
| 5 | open interest | ✅ (liquidaties niet publiek) |
| 6 | orderboek L1/L2 | ❌ geen REST-historie — zie criterium 9 |
| 7 | `data/validation/` vier validators | ✅ 33 tests |
| 8 | `utils/hashing.py` `data_hash` | ✅ |
| 9 | DVC-lineage | ✅ + 3 legacy-defecten gerepareerd |
| 10 | `docs/DATA_REGISTER.md` | ✅ gegenereerd uit de store |
| 11 | `utils/time.py` uitgebreid | ✅ `asof_join`-kern ongewijzigd |
| 12 | `tests/lookahead/test_asof_join_crypto.py` | ✅ 29 tests |
| 13 | `tests/unit/test_pit_store_immutability.py` | ✅ 23 tests |
| 14 | `tests/integration/test_data_contract_roundtrip.py` | ✅ 10 tests |
| 15 | `apps/ingest_crypto.py` | ✅ 85 LOC |

---

## Besluit dat voorligt

Criterium 3 en 6 zijn gedeeltelijk. Geen van beide is op te lossen met de
huidige bron:

- **Criterium 3** is een scope-besluit dat al genomen is. Intraday-ingestie kan
  alsnog draaien wanneer gewenst; de pijplijn ondersteunt het (`--granularity 5m`).
- **Criterium 6** vereist een tweede databron met delisting-historie. Zonder die
  bron blijft het universum survivorship-biased, en dat moet bij elke
  Phase 3-claim vermeld worden.

Phase 2 (Research & Falsification Foundation) heeft geen van beide nodig: die
fase bouwt de rechtbank, niet de verdachte. Mijn aanbeveling is Phase 2 te
starten en de twee openstaande punten mee te nemen als expliciete
randvoorwaarde bij de Phase 3-baseline.
