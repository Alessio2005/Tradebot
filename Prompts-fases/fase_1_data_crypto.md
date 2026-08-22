# MASTER-PROMPT: PHASE 1 — DATA FOUNDATION & CRYPTO INGESTION

> **Fase:** 1 van 7 · **Prioriteit:** P0 (blokkerend voor alle research)
> **Bindend brondocument:** `ARCHITECTUUR_AUDIT_2026-08-22.md` — secties 3.2, 7, 7.1, 7.2, 20, 23 (Phase 1), 26
> **Voorwaarde:** Phase 0 volledig afgerond (nul stille fallbacks, valide `git_sha`, gecentraliseerde config).

---

## ROL EN CONTEXT

Je acteert als **Quant Data Engineer & Point-in-Time Data Architect**. Jouw enige opdracht is het bouwen van een dataspoor waarop een auditor over drie jaar nog kan reconstrueren welke bytes een bepaald backtestresultaat hebben voortgebracht.

Je werkt uitsluitend binnen de kaders van het bindende auditdocument `ARCHITECTUUR_AUDIT_2026-08-22.md`.

**Relevante lagen uit de Target Architecture (sectie 19):**

| Laag | Naam | Rol in deze fase |
|---|---|---|
| **L0** | Data Ingestion, PIT Storage & Validation | **Primaire laag.** Volledig te bouwen. |
| **L1** | Feature Engineering (Stateless, Causal) | Contract vastleggen: stateless, causaal, geen fit-op-toekomst |
| **L12** | Governance, Ledger & Audit Register | Elke dataset krijgt een `data_hash` die de ledger kan citeren |

> **CORE AXIOM (sectie 7):** *"Geen betrouwbare data → geen betrouwbare quant research."*

**Kritieke bevindingen die deze fase adresseert:**
- **Sectie 3.2:** `market_data_parquet/` bevat uitsluitend FX- en macro-data. **Er is geen crypto-OHLCV of perp-data aanwezig.** Het vlaggenschipmodel (*Crypto-MN, Sharpe 1.15*) is niet reproduceerbaar.
- **Sectie 3.2:** 20 van de 26 alpha-units zijn gefalsificeerd en 4 geaccepteerd op data die niet in de repository staat. Er zijn **0 verifieerbare actieve alpha-units**.
- **Sectie 7.2:** Research-resultaten zonder geciteerde `data_hash` worden automatisch als invalid beschouwd.

**Wat behouden blijft (sectie 24 — RETAIN):** `utils/time.py::asof_join` voert een correcte `merge_asof(direction="backward")` uit. Deze functie is inherent causaal en mag **niet** herschreven worden — alleen uitgebreid met contract-guards.

---

## DOEL VAN DE FASE

Herstel de data-integriteit en vul de ontbrekende crypto-dataset aan.

Na deze fase bestaat er één onveranderlijke, versie-beheerde Point-in-Time store waarin elke tijdreeks in UTC Unix-nanoseconden is opgeslagen, elke dataset een cryptografische `data_hash` draagt, en elke join aantoonbaar causaal is. Geen enkel onderzoek in Phase 2 en verder mag data gebruiken die niet uit deze store komt.

---

## CONCRETE DELIVERABLES

1. **`src/tradebot/data/pit_store.py`** — de onveranderlijke Point-in-Time store: append-only Parquet, gepartitioneerd op `(asset_class, symbol, granularity, date)`. Schrijven overschrijft nooit; een tweede schrijfactie op dezelfde partitie met afwijkende inhoud crasht met `DataContractError`.
2. **`src/tradebot/data/ingestion.py` (herschreven)** — één generiek ingestion-contract: `fetch → validate → normalise → hash → persist`. Geen enkele bron mag deze volgorde overslaan.
3. **`src/tradebot/data/crypto.py` (herschreven)** — crypto spot & perp OHLCV ingestion voor het volledige universum (minimaal BTC, ETH, SOL, AVAX, LINK, DOT conform `conf/symbols/`), op daily en 1m/5m granulariteit.
4. **`src/tradebot/data/funding.py` (herschreven)** — perpetual funding rate ingestion met expliciete funding-interval-metadata en settlement-timestamps in UTC ns.
5. **`src/tradebot/data/open_interest.py`** — open interest en liquidatiedata per symbool.
6. **`src/tradebot/data/orderbook.py`** — L1 top-of-book en (waar beschikbaar) L2 depth snapshots, met expliciete registratie van snapshot-frequentie en gaps. Dit is de input voor de η-kalibratie in Phase 5.
7. **`src/tradebot/data/validation/`** — nieuwe submodule met vier validators:
   - `schema.py` — kolomtypen, verplichte velden, monotone timestamps
   - `gaps.py` — detectie van ontbrekende bars per granulariteit, met expliciete gap-ledger
   - `outliers.py` — prijssprongen, nul-volume bars, negatieve spreads
   - `continuity.py` — futures rolls, delistings, symbol renames
8. **`src/tradebot/utils/hashing.py` (uitgebreid)** — deterministische `data_hash` over de gesorteerde inhoud van een dataset (niet over bestandsmetadata — die is machineafhankelijk).
9. **DVC-lineage** — `dvc.yaml` + `.dvc`-bestanden voor elke dataset in de PIT store; `data_hash` en DVC-hash aantoonbaar consistent.
10. **`docs/DATA_REGISTER.md` (herschreven)** — per dataset: bron, granulariteit, periode, `data_hash`, bekende gaps, en de research tracks die hem mogen gebruiken (conform sectie 7.1).
11. **`src/tradebot/utils/time.py` (uitgebreid, niet herschreven)** — `asof_join` behoudt zijn `merge_asof(direction="backward")`-kern; toegevoegd worden: verplichte UTC-ns-assertie, verplichte sortering-assertie, en een `tolerance`-contract dat expliciet moet worden meegegeven.
12. **`tests/lookahead/test_asof_join_crypto.py`** — bewijst causaliteit van `asof_join` op echte crypto-data: een waarde op `t` mag nooit veranderen wanneer alle data na `t` wordt afgekapt.
13. **`tests/unit/test_pit_store_immutability.py`** — bewijst dat overschrijven crasht.
14. **`tests/integration/test_data_contract_roundtrip.py`** — ingest → hash → persist → reload → hash levert een identieke `data_hash`.
15. **`apps/ingest_crypto.py`** (≤80 LOC, conform R-6 uit D-7) — dunne CLI-wrapper rond de ingestion-pipeline.

---

## STAPSGEWIJZE UITVOERING

**Stap 1 — Datalacune formeel vastleggen.**
Inventariseer `market_data_parquet/` en schrijf `reports/phase1_data_gap.md`: welke asset classes, symbolen, granulariteiten en periodes daadwerkelijk aanwezig zijn versus wat sectie 7.1 vereist per research track. Dit rapport is het bewijsstuk voor sectie 3.2.

**Stap 2 — Timestamp-standaard afdwingen.**
Implementeer één centrale conversie naar **UTC Unix nanoseconden** (sectie 7.2) in `utils/time.py`. Elke ingestion-bron passeert deze conversie. Bouw een assertie die naïeve (timezone-loze) timestamps hard afwijst met `DataContractError`.

**Stap 3 — PIT store bouwen.**
Schrijf `data/pit_store.py` met het append-only contract. Test de onveranderlijkheid vóór je er data in schrijft: een tweede schrijfactie met afwijkende inhoud op dezelfde partitie moet crashen.

**Stap 4 — Validatielaag bouwen.**
Implementeer de vier validators. Elke validator retourneert **geen** boolean maar raiset bij schending. Gaps worden niet stilzwijgend geïnterpoleerd — ze worden geregistreerd in een gap-ledger en de gebruikende research track moet expliciet verklaren hoe hij ermee omgaat.

**Stap 5 — Crypto OHLCV ingestion.**
Bouw de crypto-ingestion voor spot en perp. Haal daily bars over de maximaal beschikbare historie, en 1m of 5m bars over de periode die nodig is voor Realized Variance en HAR-RV (sectie 7.1, Volatility Research). Draai elke fetch door de validatielaag.

**Stap 6 — Funding rates, open interest, liquidaties.**
Ingest de perp-specifieke reeksen. **Let op de settlement-semantiek:** een funding rate die om 08:00 UTC settelt is pas op 08:00 UTC bekend, niet om 00:00 UTC van diezelfde dag. Leg deze semantiek expliciet vast in metadata en dwing hem af in de `asof_join`-tolerance.

**Stap 7 — Orderboek-snapshots.**
Ingest L1, en L2 waar beschikbaar. Documenteer de feitelijke snapshot-frequentie en dekking — Open Question 1 in sectie 27 vraagt expliciet of er genoeg depth-data is om HAR-RV en het TCA-impactmodel te kalibreren. Jouw rapport beantwoordt die vraag met cijfers.

**Stap 8 — Survivorship bias en continuïteit.**
Neem historische delistings en faillissementen op in het universum (sectie 7.2). Bouw de futures-roll-correctie als een **expliciete adjustment factor ledger**, niet als een stilzwijgende backward-ratio-aanpassing. De ledger is inspecteerbaar; een stilzwijgende aanpassing is dat niet.

**Stap 9 — Hashing en DVC-lineage.**
Genereer per dataset een deterministische `data_hash` over de gesorteerde inhoud. Koppel DVC. Verifieer dat twee onafhankelijke ingestion-runs op dezelfde bronperiode dezelfde hash opleveren — is dat niet zo, dan is er non-determinisme in de pipeline en moet dat eerst opgelost worden.

**Stap 10 — `asof_join` verifiëren op crypto.**
Draai de truncatietest: koppel funding rates aan OHLCV via `asof_join`, kap vervolgens alle data na tijdstip `t` af, en herhaal. Elke waarde op of vóór `t` moet bit-identiek zijn. Elke afwijking is een lookahead-lek en blokkeert de fase.

**Stap 11 — Data register schrijven.**
Vul `docs/DATA_REGISTER.md` met per dataset alle metadata inclusief `data_hash`. Dit document is vanaf nu de enige geldige bron voor de vraag "welke data mag ik gebruiken".

**Stap 12 — Exit-rapport.**
`reports/phase1_exit_report.md` met per exit-criterium het bewijs.

---

## CRITERIA & VALIDATIE (EXIT CRITERIA)

1. **`asof_join` geverifieerd op crypto-data.** De truncatie-invariantietest in `tests/lookahead/test_asof_join_crypto.py` slaagt op de volledige crypto-dataset, inclusief de funding-rate-koppeling. Nul afwijkingen.
2. **Unieke `data_hash` gegenereerd** voor elke dataset in de PIT store, deterministisch reproduceerbaar over twee onafhankelijke runs, en geciteerd in `docs/DATA_REGISTER.md`.
3. **Crypto-dataset compleet.** OHLCV (daily + intraday), funding rates, open interest en liquidaties aanwezig voor het volledige universum uit `conf/symbols/`. De datalacune uit sectie 3.2 is aantoonbaar gedicht.
4. **Timezone-integriteit.** 100% van de timestamps in de PIT store is UTC Unix nanoseconden. Een geïnjecteerde naïeve timestamp crasht aantoonbaar.
5. **Onveranderlijkheid bewezen.** Een poging tot overschrijven van een bestaande partitie met afwijkende inhoud crasht met `DataContractError`.
6. **Survivorship bias geadresseerd.** Het universum bevat historische delistings; de adjustment factor ledger voor futures rolls is aanwezig en inspecteerbaar.
7. **Gap-ledger compleet.** Elke ontbrekende bar is geregistreerd. Nul stilzwijgende interpolaties in `src/`.
8. **DVC-lineage sluitend.** Elke dataset is DVC-tracked; DVC-hash en `data_hash` zijn consistent.
9. **Open Question 1 beantwoord met cijfers** — dekking en frequentie van orderboekdata gedocumenteerd, inclusief het oordeel of η-kalibratie (Phase 5) en HAR-RV (Phase 6) haalbaar zijn.

---

## REGELS & HANDLINGSINSTRUCTIES

- **Fail-fast compliance.** Nul `try/except` fallbacks. Een ontbrekende bron, een schema-schending of een gat in de data crasht de ingestion. Nooit een lege DataFrame teruggeven en doorgaan.
- **100% PIT rigor — dit is de kernwaarde van deze fase.** Elke koppeling verloopt via `asof_join` met `direction="backward"` en een expliciete tolerance. Elke aggregatie is causaal: `rolling(...).mean()` mag nooit `center=True` bevatten. Elke normalisatie (z-score, winsorisatie, scaling) gebruikt uitsluitend expanding of rolling statistieken — **nooit** statistieken over de volledige sample.
- **Geen stilzwijgende interpolatie.** `fillna(method="ffill")` over een gat is een keuze met statistische gevolgen. Zulke keuzes staan in de config, worden geregistreerd in de gap-ledger, en worden nooit impliciet in een ingestion-functie genomen.
- **Geen hardcoded variabelen.** Symbolen, periodes, granulariteiten, tolerances en API-endpoints komen uit `conf/data/`.
- **Onveranderlijkheid boven gemak.** De PIT store overschrijft nooit. Een correctie op historische data is een nieuwe versie met een nieuwe `data_hash`, niet een edit.
- **Atomaire commits.** `feat(data): add append-only PIT parquet store with partition contract`, `feat(data): ingest crypto perp funding rates in UTC nanoseconds`, `test(lookahead): verify asof_join truncation invariance on crypto`, `fix(data): reject naive timestamps at ingestion boundary`.
- **Geen research in deze fase.** Je bouwt geen features, geen signalen, geen modellen. Elke regel modelcode in deze fase is scope creep.

---

## STARTINSTRUCTIE

> **Begin nu met Stap 1:** inventariseer `market_data_parquet/` volledig en genereer `reports/phase1_data_gap.md` waarin je per research track uit sectie 7.1 aangeeft welke datasets aanwezig zijn, welke ontbreken, en over welke periode. Kwantificeer expliciet de crypto-lacune uit sectie 3.2. Commit dit als `docs(data): register phase 1 data gap analysis`.
